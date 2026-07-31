"""The imitation stage: fit a network to the cost-to-go a corpus of plans shows.

Two objectives are available and the default is the less obvious one.

**Regression** fits ``h(s) ≈ h*(s)``. It is what "learn a heuristic" usually
means, and it optimises a quantity greedy best-first search never reads.

**Ranking** fits the *order* instead. At each state on a plan, the successor the
plan took should sort ahead of the ones it passed over. GBFS pops the minimum of
the open list, so a model uniformly 30 too high guides perfectly while scoring
terribly on RMSE, and a model with excellent RMSE that inverts one pair of
siblings sends the search into the wrong subtree. Chrestien et al. (NeurIPS
2023) make this argument at length and measure the gap; the default here
follows them.

The default is not *pure* ranking. A ranking loss is invariant to any monotone
rescaling of the output, so it pins down no scale at all — which is fine for
GBFS and useless for weighted A*, where the weight multiplies a quantity that
now means nothing. A small regression term anchors it. ``rank_weight`` is that
trade-off, and 0.8 is a reasonable default rather than a tuned optimum.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from .dataset import Corpus
from .heuristic import HeuristicBundle
from .model import MLP, Adam

__all__ = ["TrainConfig", "TrainReport", "train", "evaluate"]


@dataclass
class TrainConfig:
    """Hyper-parameters for the supervised stage."""

    hidden: tuple = (32, 16)
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 0.01
    #: Share of the objective given to ranking; the rest goes to regression.
    rank_weight: float = 0.8
    #: Ranking groups considered per step. Each contributes one forward pass
    #: per successor, so this is the real cost knob.
    rank_batch: int = 16
    #: Samples from satisficing plans carry an upper bound, not ``h*``. Halving
    #: their weight is a middle road between trusting and discarding them.
    suboptimal_weight: float = 0.5
    validation: float = 0.2
    #: Stop after this many epochs without a new best validation score. Set to
    #: 0 to disable and always run every epoch.
    patience: int = 12
    seed: int = 0
    verbose: bool = False

    def to_dict(self) -> dict:
        return {
            "hidden": list(self.hidden),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "rank_weight": self.rank_weight,
            "rank_batch": self.rank_batch,
            "suboptimal_weight": self.suboptimal_weight,
            "validation": self.validation,
            "patience": self.patience,
            "seed": self.seed,
        }


@dataclass
class TrainReport:
    """What happened during training, for plotting and for the record."""

    history: list = field(default_factory=list)
    best_epoch: int = 0
    metrics: dict = field(default_factory=dict)
    seconds: float = 0.0


def train(corpus: Corpus, config: Optional[TrainConfig] = None) -> tuple:
    """Fit a heuristic to ``corpus``. Returns ``(bundle, report)``."""
    config = config or TrainConfig()
    if not corpus.samples:
        raise ValueError("cannot train on an empty corpus")

    rng = random.Random(config.seed)
    train_set, val_set = corpus.split(config.validation, seed=config.seed)
    if not train_set.samples:  # pragma: no cover - guarded by split()
        raise ValueError("the training split came out empty")

    # Normalise targets so the network learns a shape rather than a magnitude.
    targets = [s.target for s in train_set.samples]
    scale = max(1e-6, sum(targets) / len(targets))

    sizes = [corpus.space.size] + list(config.hidden) + [1]
    model = MLP(sizes, seed=config.seed)
    optimiser = Adam(model, lr=config.learning_rate)

    started = time.perf_counter()
    report = TrainReport()
    best_score = math.inf
    best_state = model.to_dict()
    best_epoch = 0
    stale = 0

    order = list(range(len(train_set.samples)))
    for epoch in range(1, config.epochs + 1):
        rng.shuffle(order)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(order), config.batch_size):
            batch = [
                train_set.samples[i] for i in order[start : start + config.batch_size]
            ]
            loss, grad_w, grad_b = _step(
                model, batch, train_set.groups, scale, config, rng
            )
            optimiser.step(grad_w, grad_b)
            epoch_loss += loss
            steps += 1

        scored = val_set if val_set.samples else train_set
        metrics = evaluate(model, scored, scale)
        # Selection is on ranking accuracy where there is any to measure, since
        # that is the quantity search consumes; MAE only breaks ties.
        rank_metrics = evaluate_ranking(model, train_set.groups, scale)
        score = -(rank_metrics.get("top1", 0.0)) + 1e-3 * metrics["mae"]
        report.history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / max(1, steps),
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "train_top1": rank_metrics.get("top1", 0.0),
            }
        )
        if config.verbose:  # pragma: no cover - operator convenience
            print(
                f"  epoch {epoch:3d}  loss {epoch_loss / max(1, steps):8.4f}"
                f"  val MAE {metrics['mae']:7.3f}"
                f"  top-1 {rank_metrics.get('top1', 0.0):5.3f}"
            )
        if score < best_score - 1e-9:
            best_score = score
            best_state = model.to_dict()
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if config.patience and stale >= config.patience:
                break

    model = MLP.from_dict(best_state)
    report.best_epoch = best_epoch
    report.seconds = time.perf_counter() - started

    final = evaluate(model, val_set if val_set.samples else train_set, scale)
    final.update(
        {"train_" + k: v for k, v in evaluate(model, train_set, scale).items()}
    )
    final.update(evaluate_ranking(model, corpus.groups, scale))
    final["held_out_instances"] = len({s.instance for s in val_set.samples})
    report.metrics = final

    bundle = HeuristicBundle(
        corpus.space, model, scale, metrics=final, config=config.to_dict()
    )
    return bundle, report


def _step(model: MLP, batch, groups, scale: float, config: TrainConfig, rng):
    """One optimiser step over a regression batch and a sample of ranking groups."""
    reg_weight = 1.0 - config.rank_weight
    grad_w = None
    grad_b = None
    total_loss = 0.0

    if reg_weight > 0.0 and batch:
        features = [s.features for s in batch]
        outputs, cache = model.forward_batch(features)
        weights = [1.0 if s.optimal else config.suboptimal_weight for s in batch]
        norm = sum(weights) or 1.0
        d_out = []
        for out, sample, weight in zip(outputs, batch, weights):
            target = sample.target / scale
            diff = out - target
            total_loss += reg_weight * weight * diff * diff / norm
            d_out.append(reg_weight * 2.0 * weight * diff / norm)
        grad_w, grad_b = model.backward_batch(cache, d_out)

    if config.rank_weight > 0.0 and groups:
        picked = (
            rng.sample(groups, config.rank_batch)
            if len(groups) > config.rank_batch
            else list(groups)
        )
        flat = []
        spans = []
        for group in picked:
            start = len(flat)
            flat.append(group.chosen)
            flat.extend(group.others)
            spans.append((start, len(flat)))
        outputs, cache = model.forward_batch([f for f, _ in flat])
        d_out = [0.0] * len(flat)
        for start, end in spans:
            # Score each successor by what search will actually compare:
            # the step cost plus the predicted cost-to-go.
            scores = [-(flat[i][1] / scale + outputs[i]) for i in range(start, end)]
            peak = max(scores)
            exps = [math.exp(s - peak) for s in scores]
            total = sum(exps)
            probs = [e / total for e in exps]
            total_loss += (
                -config.rank_weight * math.log(max(probs[0], 1e-12)) / len(spans)
            )
            for offset, prob in enumerate(probs):
                chosen = 1.0 if offset == 0 else 0.0
                d_out[start + offset] = (
                    config.rank_weight * (chosen - prob) / len(spans)
                )
        rank_w, rank_b = model.backward_batch(cache, d_out)
        if grad_w is None:
            grad_w, grad_b = rank_w, rank_b
        else:
            _accumulate(grad_w, rank_w)
            _accumulate(grad_b, rank_b)

    if grad_w is None:  # pragma: no cover - both terms disabled
        grad_w = [[[0.0] * len(r) for r in layer] for layer in model.weights]
        grad_b = [[0.0] * len(b) for b in model.biases]
    return total_loss, grad_w, grad_b


def _accumulate(into, extra) -> None:
    for a, b in zip(into, extra):
        if isinstance(a, list) and a and isinstance(a[0], list):
            _accumulate(a, b)
        else:
            for i in range(len(a)):
                a[i] += b[i]


def evaluate(model: MLP, corpus: Corpus, scale: float) -> dict:
    """Regression error of ``model`` on ``corpus``, in original cost units."""
    if not corpus.samples:  # pragma: no cover - guarded by callers
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "count": 0}
    outputs, _ = model.forward_batch([s.features for s in corpus.samples])
    errors = [out * scale - s.target for out, s in zip(outputs, corpus.samples)]
    count = len(errors)
    return {
        "mae": sum(abs(e) for e in errors) / count,
        "rmse": math.sqrt(sum(e * e for e in errors) / count),
        # Systematically low is dangerous in a different way from
        # systematically high: it makes the search look like A* with a weak
        # heuristic rather than a greedy one, and expands far more.
        "bias": sum(errors) / count,
        "count": count,
    }


def evaluate_ranking(model: MLP, groups, scale: float) -> dict:
    """How often the model prefers the successor the plan actually took.

    This tracks search performance far better than RMSE does. ``top1`` is the
    fraction of decision points where the plan's successor scores strictly
    lowest; ``in_top2`` allows one better-looking sibling, which a search with
    any backtracking will usually survive.
    """
    groups = list(groups or ())
    if not groups:
        return {}
    flat = []
    spans = []
    for group in groups:
        start = len(flat)
        flat.append(group.chosen)
        flat.extend(group.others)
        spans.append((start, len(flat)))
    outputs, _ = model.forward_batch([f for f, _ in flat])
    top1 = 0
    top2 = 0
    for start, end in spans:
        scores = [flat[i][1] / scale + outputs[i] for i in range(start, end)]
        chosen = scores[0]
        better = sum(1 for s in scores[1:] if s < chosen)
        if better == 0:
            top1 += 1
        if better <= 1:
            top2 += 1
    return {
        "top1": top1 / len(spans),
        "in_top2": top2 / len(spans),
        "groups": len(spans),
    }
