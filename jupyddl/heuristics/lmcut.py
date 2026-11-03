"""The LM-cut heuristic (Helmert & Domshlak, 2009).

LM-cut repeatedly computes h_max, finds a *landmark cut* of operators separating
the initial state from the goal in the justification graph, adds the cut's
cheapest operator cost to the estimate, and discounts the cut operators. It is
one of the strongest admissible heuristics for optimal planning.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from .base import Heuristic


@dataclass
class _AugOp:
    idx: int
    pre: frozenset
    add: frozenset
    base_cost: int


class LMCutHeuristic(Heuristic):
    name = "lmcut"
    admissible = True

    def __init__(self, task):
        super().__init__(task)
        # Each conditional effect is relaxed into its own operator charging the
        # full action cost, and the cost partitioning can then count it twice.
        if task.has_conditional_effects:
            self.admissible = False
        nf = task.num_facts
        self.INIT = nf  # artificial "always true" fact
        self.GOAL = nf + 1  # artificial goal fact
        self.num = nf + 2

        self.ops: list[_AugOp] = []
        for pre, add, cost in task.relaxed_operators():
            # Every operator must have at least one precondition for the
            # justification graph; use the artificial INIT fact if needed.
            pre = pre if pre else frozenset({self.INIT})
            self.ops.append(_AugOp(len(self.ops), pre, add, cost))
        # Artificial goal operator (cost 0) collecting the real goal.
        self.goal_op = len(self.ops)
        goal_pre = task.goals if task.goals else frozenset({self.INIT})
        self.ops.append(_AugOp(self.goal_op, goal_pre, frozenset({self.GOAL}), 0))

        self.consumers: list[list[int]] = [[] for _ in range(self.num)]
        for op in self.ops:
            for f in op.pre:
                self.consumers[f].append(op.idx)

    def __call__(self, state) -> float:
        base = set(state)
        base.add(self.INIT)
        source = frozenset(base)
        costs = [op.base_cost for op in self.ops]
        total = 0.0

        while True:
            hmax, pcf = self._hmax(source, costs)
            if hmax[self.GOAL] == 0:
                return total
            if math.isinf(hmax[self.GOAL]):
                return math.inf

            goal_zone = self._goal_zone(costs, pcf)
            cut = self._cut(source, goal_zone, pcf, hmax)
            if not cut:  # should not happen when hmax(goal) > 0
                return total  # a lower bound either way; inf would prune
            min_cost = min(costs[oi] for oi in cut)
            total += min_cost
            for oi in cut:
                costs[oi] -= min_cost

    def _hmax(self, source, costs):
        inf = math.inf
        cost = [inf] * self.num
        pcf = [-1] * len(self.ops)
        counter = [len(op.pre) for op in self.ops]
        pq: list = []
        for f in source:
            cost[f] = 0
            heapq.heappush(pq, (0, f))

        def relax(op: _AugOp):
            # Ties go to the higher fact id, so the cut never depends on set order.
            supporter = max(op.pre, key=lambda p: (cost[p], p))
            value = cost[supporter] + costs[op.idx]
            pcf[op.idx] = supporter
            if math.isinf(cost[supporter]):
                return
            for f in op.add:
                if value < cost[f]:
                    cost[f] = value
                    heapq.heappush(pq, (value, f))

        for op in self.ops:
            if counter[op.idx] == 0:
                relax(op)
        while pq:
            c, f = heapq.heappop(pq)
            if c > cost[f]:
                continue
            for op_idx in self.consumers[f]:
                counter[op_idx] -= 1
                if counter[op_idx] == 0:
                    relax(self.ops[op_idx])
        return cost, pcf

    def _goal_zone(self, costs, pcf):
        """Facts that reach the goal through zero-cost justification edges."""
        zone = {self.GOAL}
        changed = True
        while changed:
            changed = False
            for op in self.ops:
                supporter = pcf[op.idx]
                if (
                    costs[op.idx] == 0
                    and supporter >= 0
                    and supporter not in zone
                    and (op.add & zone)
                ):
                    zone.add(supporter)
                    changed = True
        return zone

    def _cut(self, source, goal_zone, pcf, hmax):
        """Operators crossing from the init-reachable region into the goal zone."""
        # Forward-reachable facts from the initial state that stay out of the
        # goal zone (the "before" region).
        before = {f for f in source if f not in goal_zone}
        changed = True
        while changed:
            changed = False
            for op in self.ops:
                supporter = pcf[op.idx]
                if supporter in before and not math.isinf(hmax[supporter]):
                    for f in op.add:
                        if f not in goal_zone and f not in before:
                            before.add(f)
                            changed = True
        cut = [
            op.idx for op in self.ops if pcf[op.idx] in before and (op.add & goal_zone)
        ]
        return cut
