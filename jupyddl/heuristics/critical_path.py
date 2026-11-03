"""Critical-path heuristics h^m (Haslum & Geffner, 2000).

``h^m`` estimates the cost of the most expensive size-``m`` subset of atoms.
``h^1`` equals ``h_max``; higher ``m`` is more informative but costs more.

This computes ``h^m`` over the *delete relaxation*, as the relaxation
heuristics here do: it is admissible on tasks without conditional effects, but
it cannot see mutexes the way Haslum and Geffner's full ``h^m`` does.
With conditional effects and ``m >= 2`` it can overestimate, and the
instance's ``admissible`` flag says so.
"""

from __future__ import annotations

import math
from itertools import combinations

from ..task import facts_of
from .base import Heuristic


class CriticalPathHeuristic(Heuristic):
    name = "hm"
    admissible = True

    def __init__(self, task, m: int = 2):
        super().__init__(task)
        self.m = m
        self.goal = task.goals
        self.ops = list(task.relaxed_operators())  # (pre, add, cost)
        if m >= 2 and task.has_conditional_effects:
            self.admissible = False

    def __call__(self, state) -> float:
        return self._hm(state)

    def _hm(self, state) -> float:
        m = self.m
        inf = math.inf
        state = facts_of(state)
        atoms = set(state) | set(self.goal)
        for pre, add, _ in self.ops:
            atoms |= pre
            atoms |= add
        atoms = sorted(atoms)

        # Table of costs for every atom-set of size 1..m.
        table: dict = {}
        sets: list = []
        for size in range(1, m + 1):
            for combo in combinations(atoms, size):
                fs = frozenset(combo)
                table[fs] = 0.0 if fs <= state else inf
                sets.append(fs)

        def cost_of(subset: frozenset) -> float:
            # Cost of an arbitrary atom-set: table lookup, or (if larger than m)
            # the max over its size-m subsets.
            if not subset:
                return 0.0
            if len(subset) <= m:
                return table.get(subset, inf)
            best = 0.0
            for combo in combinations(sorted(subset), m):
                val = table.get(frozenset(combo), inf)
                if val > best:
                    best = val
            return best

        # Value iteration to the fixpoint.
        changed = True
        while changed:
            changed = False
            for target in sets:
                if table[target] == 0.0:
                    continue
                best = table[target]
                for pre, add, cost in self.ops:
                    if not (add & target):
                        continue
                    regressed = (target - add) | pre
                    val = cost + cost_of(regressed)
                    if val < best:
                        best = val
                if best < table[target]:
                    table[target] = best
                    changed = True

        return cost_of(frozenset(self.goal))
