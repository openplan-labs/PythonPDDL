"""Cheap non-relaxation heuristics."""

from __future__ import annotations

from ..task import facts_of
from .base import Heuristic


class BlindHeuristic(Heuristic):
    """0 in a goal state, otherwise the cheapest operator cost. Admissible."""

    name = "blind"
    admissible = True

    def __init__(self, task):
        super().__init__(task)
        costs = [op.cost for op in task.operators if op.cost > 0]
        self.min_cost = min(costs) if costs else 1

    def __call__(self, state) -> float:
        return 0.0 if self.task.goal_reached(state) else float(self.min_cost)


class GoalCountHeuristic(Heuristic):
    """Number of unsatisfied goal facts. Fast, informative, not admissible."""

    name = "goalcount"

    def __call__(self, state) -> float:
        return float(len(self.task.goals - facts_of(state)))
