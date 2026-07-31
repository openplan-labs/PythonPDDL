"""Turn solved plans into supervision.

A plan is a labelled trajectory whether or not it was collected for that
purpose. If ``s0 -a1-> s1 -a2-> ... -> sn`` is a plan, then the cost of the
suffix from ``si`` is an upper bound on ``h*(si)``, and when the plan is
optimal it *is* ``h*(si)``. One solved instance therefore yields one sample per
state on its plan, free.

Two kinds of sample come out of this module, and the difference matters more
than it looks:

**Regression samples** — ``(features(si), cost of the suffix)``. The obvious
thing, and what most of the literature reports.

**Ranking samples** — at each ``si``, the successor the plan takes together
with the siblings it passes over. Greedy best-first search never reads an
h-value; it reads the *order* h imposes on the open list. A model with a
systematic offset of +30 is useless as a cost estimate and a perfect guide, and
a model that is accurate on average but inverts two siblings sends the search
down the wrong subtree. Optimising the order directly is the better-matched
objective (Chrestien et al., NeurIPS 2023) and is why
:func:`jupyddl.learn.train.train` defaults to it.

Sampling only states on the plan leaves a distribution-shift problem: at search
time the planner asks about states no plan ever passed through, and the model
extrapolates. :mod:`jupyddl.learn.rl` is where that gets fixed.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from dataclasses import replace as _replace

from ..task import facts_of, values_of
from .features import FeatureSpace

__all__ = [
    "Sample",
    "RankingGroup",
    "Corpus",
    "samples_from_plan",
    "build_corpus",
    "task_from_state",
]


def task_from_state(task, state):
    """The same task, re-rooted at ``state``.

    Labelling a state means finding a plan *from* it, and the planners all
    start at ``task.initial_state()``. Rather than thread a start state through
    fourteen planner signatures, move the task: everything else — operators,
    goals, axioms, the numeric layer — is unchanged, so every planner and every
    heuristic works on the result unmodified.
    """
    return _replace(task, init=facts_of(state), init_values=values_of(state))


@dataclass
class Sample:
    """One state, its feature vector, and the cost-to-go we believe it has."""

    features: list
    target: float
    #: False when the plan it came from was only satisficing, so ``target`` is
    #: an upper bound. Training can down-weight these rather than discard them.
    optimal: bool = True
    instance: str = ""


@dataclass
class RankingGroup:
    """Successors of one state: the one the plan took, and the ones it did not.

    ``chosen`` and ``others`` hold ``(features, step_cost)``. The step cost has
    to travel with the vector because the comparison GBFS makes at expansion
    time is between ``h(s')`` values, but the comparison that is *correct* is
    between ``c(s, s') + h*(s')`` — with non-uniform action costs those differ.
    """

    chosen: tuple
    others: list = field(default_factory=list)
    instance: str = ""


class Corpus:
    """Regression samples and ranking groups over a shared feature space."""

    def __init__(self, space: FeatureSpace, samples=None, groups=None):
        self.space = space
        self.samples = list(samples or ())
        self.groups = list(groups or ())

    def __len__(self) -> int:
        return len(self.samples)

    def extend(self, other: "Corpus") -> None:
        if other.space != self.space:
            raise ValueError("cannot merge corpora with different feature spaces")
        self.samples.extend(other.samples)
        self.groups.extend(other.groups)

    def split(self, validation: float = 0.2, seed: int = 0):
        """Hold out a fraction of the samples, grouped so nothing leaks.

        The split is by *instance*, not by sample. Two states three steps apart
        on the same plan have nearly identical features and nearly identical
        targets; splitting at random puts one in train and one in validation
        and reports a validation score that measures memorisation.
        """
        instances = sorted({s.instance for s in self.samples})
        if len(instances) < 2:
            # Nothing to hold out by instance; fall back to a sample split and
            # accept that the score is optimistic.
            rng = random.Random(seed)
            order = list(range(len(self.samples)))
            rng.shuffle(order)
            cut = max(1, int(len(order) * (1 - validation)))
            train_idx = set(order[:cut])
            train = [s for i, s in enumerate(self.samples) if i in train_idx]
            val = [s for i, s in enumerate(self.samples) if i not in train_idx]
            return (
                Corpus(self.space, train, self.groups),
                Corpus(self.space, val, []),
            )
        rng = random.Random(seed)
        rng.shuffle(instances)
        cut = max(1, int(len(instances) * (1 - validation)))
        held = set(instances[cut:])
        train = [s for s in self.samples if s.instance not in held]
        val = [s for s in self.samples if s.instance in held]
        groups = [g for g in self.groups if g.instance not in held]
        return Corpus(self.space, train, groups), Corpus(self.space, val, [])

    def target_stats(self) -> dict:
        if not self.samples:
            return {"count": 0}
        targets = [s.target for s in self.samples]
        mean = sum(targets) / len(targets)
        var = sum((t - mean) ** 2 for t in targets) / len(targets)
        return {
            "count": len(targets),
            "groups": len(self.groups),
            "mean": mean,
            "std": math.sqrt(var),
            "min": min(targets),
            "max": max(targets),
            "optimal_fraction": sum(s.optimal for s in self.samples) / len(targets),
        }

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "space": self.space.to_dict(),
            "samples": [
                {
                    "features": s.features,
                    "target": s.target,
                    "optimal": s.optimal,
                    "instance": s.instance,
                }
                for s in self.samples
            ],
            "groups": [
                {
                    "chosen": [list(g.chosen[0]), g.chosen[1]],
                    "others": [[list(f), c] for f, c in g.others],
                    "instance": g.instance,
                }
                for g in self.groups
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Corpus":
        space = FeatureSpace.from_dict(data["space"])
        samples = [
            Sample(
                s["features"],
                s["target"],
                s.get("optimal", True),
                s.get("instance", ""),
            )
            for s in data["samples"]
        ]
        groups = [
            RankingGroup(
                (g["chosen"][0], g["chosen"][1]),
                [(f, c) for f, c in g["others"]],
                g.get("instance", ""),
            )
            for g in data.get("groups", [])
        ]
        return cls(space, samples, groups)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)

    @classmethod
    def load(cls, path: str) -> "Corpus":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def samples_from_plan(
    task,
    plan,
    bound,
    instance: str = "",
    optimal: bool = True,
    ranking: bool = True,
    max_siblings: int = 8,
    rng: Optional[random.Random] = None,
    start_state=None,
):
    """Regression samples and ranking groups along one plan.

    ``bound`` is a :class:`~jupyddl.learn.features.BoundFeatures` for ``task``.
    ``start_state`` is where the plan begins, defaulting to the task's initial
    state; the aggregation stages pass a state the search wandered into and a
    plan found from there.

    Replaying the plan through :meth:`~jupyddl.task.Task.apply` rather than
    ``operator.apply`` is not optional: it is what closes derived predicates and
    carries numeric fluents, and a state missing its derived facts has different
    features from the one the planner will actually see.
    """
    rng = rng or random.Random(0)
    plan = list(plan or ())
    if not plan:
        return [], []

    states = [task.initial_state() if start_state is None else task.close(start_state)]
    for operator in plan:
        states.append(task.apply(operator, states[-1]))

    suffix = [0.0] * (len(plan) + 1)
    for i in range(len(plan) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + plan[i].cost

    samples = [
        Sample(bound(state), suffix[i], optimal, instance)
        for i, state in enumerate(states)
    ]

    groups = []
    if ranking:
        for i, operator in enumerate(plan):
            state = states[i]
            chosen = (bound(states[i + 1]), float(operator.cost))
            siblings = []
            for other in task.operators:
                if other is operator or not other.applicable(state):
                    continue
                successor = task.apply(other, state)
                if facts_of(successor) == facts_of(states[i + 1]):
                    continue  # a different operator reaching the same state
                siblings.append((successor, float(other.cost)))
            if not siblings:
                continue
            # A branching factor in the hundreds would make the ranking loss
            # dominate the epoch; a sample of the siblings carries the signal.
            if len(siblings) > max_siblings:
                siblings = rng.sample(siblings, max_siblings)
            groups.append(
                RankingGroup(
                    chosen,
                    [(bound(s), c) for s, c in siblings],
                    instance,
                )
            )
    return samples, groups


def build_corpus(
    tasks,
    solver,
    space: Optional[FeatureSpace] = None,
    ranking: bool = True,
    optimal: bool = True,
    seed: int = 0,
    on_instance=None,
):
    """Solve every task and collect what the plans teach.

    ``tasks`` is an iterable of ``(name, task)`` and ``solver`` maps a task to a
    :class:`~jupyddl.search.SearchResult`. ``optimal`` states whether that
    solver is cost-optimal, and must be told rather than guessed: a
    ``SearchResult`` records what was found, not whether anything cheaper
    exists, so nothing in the result distinguishes an optimal plan from a
    satisficing one. Getting it wrong mislabels upper bounds as ``h*``.
    ``on_instance`` is called with ``(name, result)`` after each solve, for
    progress reporting.
    """
    tasks = list(tasks)
    if space is None:
        space = FeatureSpace.from_tasks(task for _, task in tasks)
    corpus = Corpus(space)
    rng = random.Random(seed)
    for name, task in tasks:
        result = solver(task)
        if on_instance is not None:
            on_instance(name, result)
        if not result.solved or not result.plan:
            continue
        samples, groups = samples_from_plan(
            task,
            result.plan,
            space.bind(task),
            instance=name,
            optimal=optimal,
            ranking=ranking,
            rng=rng,
        )
        corpus.samples.extend(samples)
        corpus.groups.extend(groups)
    return corpus
