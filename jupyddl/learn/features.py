"""Turn a grounded state into a fixed-length feature vector.

A learned heuristic is only interesting if it *transfers* — trained on small
instances, useful on large ones. That rules out the obvious encoding. A
one-hot over ``task.facts`` has a different length for every instance and
attaches meaning to fact ids that are an artefact of grounding order, so a
model trained on 4-block blocksworld cannot even be evaluated on 8 blocks.

Everything here is therefore keyed on the **predicate symbol** rather than the
grounded atom, and normalised by how many atoms of that symbol exist. Both
choices are what make the vector size-invariant: ``on`` contributes one feature
whether there are four blocks or forty, and its value stays in ``[0, 1]``.

This is the cheap end of a spectrum. The expensive end is a graph network over
the grounded problem (STRIPS-HGN, ASNets, GOOSE), which captures object
identity and relational structure this cannot. The trade is deliberate: a
heuristic that is slower to evaluate than ``hff`` has to be *much* better
informed to win on time, and in a pure-Python planner it will not be. See
``.docs/learned-heuristics.md``.

Cost is ``O(|s| + |goals|)`` per state, with no successor generation.
"""

from __future__ import annotations

import json
import math
import re

from ..task import facts_of

__all__ = ["FeatureSpace", "predicate_of"]

# Grounded facts print Lisp-style — ``(on a b)``, ``(handempty)`` — while
# operators print functionally, ``move(a,b)#2``. Accept both rather than
# assuming: getting this wrong does not raise, it silently gives every ground
# atom its own slot, and the feature vector stops being size-invariant while
# still looking perfectly reasonable.
_SYMBOL = re.compile(r"^\(?\s*([^\s()]+)")


def predicate_of(fact_name: str) -> str:
    """The predicate symbol of a grounded fact string.

    >>> predicate_of("(on b1 b2)")
    'on'
    >>> predicate_of("(handempty)")
    'handempty'
    >>> predicate_of("move(a,b)")
    'move'
    """
    match = _SYMBOL.match(fact_name)
    if not match:
        return fact_name
    return match.group(1).split("(")[0]


class FeatureSpace:
    """Maps a state of one task into a vector comparable across tasks.

    The vocabulary — which predicate symbols get which slot — is fixed at
    construction and travels with the trained model. A task using a symbol the
    vocabulary has never seen contributes nothing rather than shifting every
    other feature along, which is what lets one model serve a whole domain and
    degrade gracefully on a related one.
    """

    #: Features that do not belong to any single predicate.
    GLOBAL_FEATURES = (
        "goal_unsatisfied_fraction",
        "goal_unsatisfied_log",
        "state_size_fraction",
        "goal_satisfied_any",
    )

    def __init__(self, vocabulary):
        self.vocabulary = tuple(vocabulary)
        self._slot = {name: i for i, name in enumerate(self.vocabulary)}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_task(cls, task) -> "FeatureSpace":
        """Vocabulary from one task's predicate symbols."""
        return cls.from_tasks([task])

    @classmethod
    def from_tasks(cls, tasks) -> "FeatureSpace":
        """Vocabulary from several tasks, so one model covers all of them.

        Symbols are sorted rather than encounter-ordered: the vocabulary has to
        be identical whichever order the training instances arrive in, or two
        runs over the same corpus produce models that disagree.
        """
        symbols = set()
        for task in tasks:
            for name in task.facts:
                symbols.add(predicate_of(name))
        return cls(sorted(symbols))

    @property
    def size(self) -> int:
        return 2 * len(self.vocabulary) + len(self.GLOBAL_FEATURES)

    def names(self) -> list:
        """Human-readable feature names, in vector order (for inspection)."""
        rows = [f"true:{p}" for p in self.vocabulary]
        rows += [f"open-goal:{p}" for p in self.vocabulary]
        rows += list(self.GLOBAL_FEATURES)
        return rows

    # -- binding to a task -------------------------------------------------
    def bind(self, task) -> "BoundFeatures":
        """Pre-compute everything about ``task`` that does not vary by state.

        Per-state work then reduces to two counting passes. Binding once per
        task and reusing it is the difference between a heuristic that is cheap
        and one that re-derives the whole predicate table on every node.
        """
        return BoundFeatures(self, task)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"vocabulary": list(self.vocabulary)}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpace":
        return cls(data["vocabulary"])

    def __eq__(self, other) -> bool:
        return isinstance(other, FeatureSpace) and self.vocabulary == other.vocabulary

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FeatureSpace({len(self.vocabulary)} predicates, {self.size} features)"


class BoundFeatures:
    """A :class:`FeatureSpace` specialised to one task."""

    def __init__(self, space: FeatureSpace, task):
        self.space = space
        self.task = task
        width = len(space.vocabulary)

        # fact id -> vocabulary slot, or -1 for a symbol outside the vocabulary.
        self.slot_of = tuple(
            space._slot.get(predicate_of(name), -1) for name in task.facts
        )

        # Per-symbol totals, used as denominators. A symbol the task never
        # grounds keeps a denominator of 1 so its feature is a constant zero
        # rather than a division by zero.
        totals = [0] * width
        for slot in self.slot_of:
            if slot >= 0:
                totals[slot] += 1
        self.fact_totals = tuple(t or 1 for t in totals)

        goal_totals = [0] * width
        for fact in task.goals:
            slot = self.slot_of[fact]
            if slot >= 0:
                goal_totals[slot] += 1
        self.goal_totals = tuple(t or 1 for t in goal_totals)

        self.goals = frozenset(task.goals)
        self.num_goals = len(self.goals) or 1
        self.num_facts = len(task.facts) or 1
        self.width = width
        self.size = space.size

    def __call__(self, state) -> list:
        """The feature vector for ``state``."""
        width = self.width
        slot_of = self.slot_of
        vector = [0.0] * self.size

        facts = facts_of(state)
        for fact in facts:
            slot = slot_of[fact]
            if slot >= 0:
                vector[slot] += 1.0

        open_goals = 0
        for fact in self.goals:
            if fact not in facts:
                open_goals += 1
                slot = slot_of[fact]
                if slot >= 0:
                    vector[width + slot] += 1.0

        fact_totals = self.fact_totals
        goal_totals = self.goal_totals
        for i in range(width):
            vector[i] /= fact_totals[i]
            vector[width + i] /= goal_totals[i]

        base = 2 * width
        vector[base] = open_goals / self.num_goals
        # Counts span orders of magnitude across a scaling ladder; the log keeps
        # the tail from swamping every other feature.
        vector[base + 1] = math.log1p(open_goals)
        vector[base + 2] = len(facts) / self.num_facts
        vector[base + 3] = 1.0 if open_goals < self.num_goals else 0.0
        return vector


def save_space(space: FeatureSpace, path: str) -> None:  # pragma: no cover - thin
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(space.to_dict(), handle)


def load_space(path: str) -> FeatureSpace:  # pragma: no cover - thin
    with open(path, encoding="utf-8") as handle:
        return FeatureSpace.from_dict(json.load(handle))
