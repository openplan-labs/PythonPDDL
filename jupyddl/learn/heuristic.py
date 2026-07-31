"""The learned heuristic itself, and the artefact it is loaded from.

A trained heuristic is three things that must travel together: the feature
vocabulary, the network, and the scale its targets were normalised by. Ship the
network alone and it silently mis-predicts on the first task whose predicate
symbols hash to different slots — no error, just a bad heuristic, which is the
hardest kind of bug to notice because the planner still returns plans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..heuristics.base import Heuristic
from .features import FeatureSpace
from .model import MLP

__all__ = ["HeuristicBundle", "LearnedHeuristic"]

FORMAT_VERSION = 1


@dataclass
class HeuristicBundle:
    """Everything needed to evaluate a learned heuristic on a fresh task."""

    space: FeatureSpace
    model: MLP
    #: Targets are trained in units of ``scale``; predictions are multiplied
    #: back. Regressing raw costs that range over two orders of magnitude
    #: leaves the first layer fighting the output layer for the scale.
    scale: float = 1.0
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "format": FORMAT_VERSION,
            "space": self.space.to_dict(),
            "model": self.model.to_dict(),
            "scale": self.scale,
            "metrics": self.metrics,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HeuristicBundle":
        version = data.get("format", 0)
        if version != FORMAT_VERSION:
            raise ValueError(
                f"unsupported learned-heuristic format {version}; "
                f"this build reads version {FORMAT_VERSION}"
            )
        return cls(
            FeatureSpace.from_dict(data["space"]),
            MLP.from_dict(data["model"]),
            float(data.get("scale", 1.0)),
            dict(data.get("metrics", {})),
            dict(data.get("config", {})),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=1)

    @classmethod
    def load(cls, path: str) -> "HeuristicBundle":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def bind(self, task) -> "LearnedHeuristic":
        return LearnedHeuristic(task, self)


class LearnedHeuristic(Heuristic):
    """Evaluates a trained network as a heuristic.

    Never admissible, and it does not pretend otherwise: nothing in the
    training objective bounds the prediction from above, and a single
    over-estimate costs A* its optimality guarantee. Pair it with ``gbfs``, or
    with ``wastar`` if you want the bounded-suboptimality knob.

    Values are cached per state. The planner already caches heuristic values in
    its open-list bookkeeping, but a re-opened state is re-evaluated there, and
    a network evaluation is far more expensive than a dictionary lookup.
    """

    name = "learned"
    admissible = False

    def __init__(self, task, bundle: HeuristicBundle):
        super().__init__(task)
        self.bundle = bundle
        self.bound = bundle.space.bind(task)
        self.scale = bundle.scale
        self._model = bundle.model
        self._cache: dict = {}
        self.evaluations = 0

    def __call__(self, state) -> float:
        cached = self._cache.get(state)
        if cached is not None:
            return cached
        self.evaluations += 1
        value = self._model(self.bound(state)) * self.scale
        # A goal state must score zero however the network was trained; the
        # planner's termination test does not consult the heuristic, but a
        # non-zero goal estimate distorts every f-value near the goal.
        if self.task.goal_reached(state):
            value = 0.0
        self._cache[state] = value
        return value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LearnedHeuristic({self._model!r}, scale={self.scale:.3g})"
