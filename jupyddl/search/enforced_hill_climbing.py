"""Enforced Hill Climbing (EHC), the search that made the FF planner famous.

From each state it runs a breadth-first probe until it reaches a state with a
strictly better heuristic value, then commits to that path and repeats. It is a
fast satisficing strategy (not complete, not optimal); pair it with an
informative heuristic such as FF.
"""

from __future__ import annotations

import time
from collections import deque

from .base import Planner, heuristic_name
from .result import SearchResult, SearchStats


class EnforcedHillClimbing(Planner):
    """Enforced hill climbing: breadth-first probes for a strictly better h."""

    name = "ehc"
    requires_heuristic = True

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        start = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, heuristic_name(heuristic))
        current = task.initial_state()
        current_h = heuristic(current)
        stats.evaluated += 1
        plan: list = []
        cost = 0
        plateau = 0
        while not task.goal_reached(current):
            # Each probe is a fresh breadth-first dive: report it as a new
            # "bound" so traces show the staircase of committed h-values.
            if observer is not None:
                observer.on_bound(float(current_h), plateau, stats)
            plateau += 1
            state, ops, path_cost, new_h = self._probe(
                task, current, current_h, heuristic, stats, observer
            )
            if state is None:  # no strictly-improving state reachable
                stats.runtime = time.perf_counter() - start
                result = SearchResult(False, None, None, stats)
                if observer is not None:
                    observer.on_finish(result)
                return result
            plan.extend(ops)
            cost += path_cost
            current = state
            current_h = new_h
        stats.runtime = time.perf_counter() - start
        result = SearchResult(True, plan, cost, stats)
        if observer is not None:
            observer.on_goal(current, cost, len(plan), stats)
            observer.on_finish(result)
        return result

    def _probe(self, task, start, target_h, heuristic, stats, observer=None):
        """BFS from ``start`` for a state with ``h < target_h`` (or the goal)."""
        visited = {start}
        queue = deque([(start, [], 0)])
        while queue:
            state, ops, cost = queue.popleft()
            stats.expanded += 1
            if observer is not None:
                observer.on_expand(
                    state,
                    g=cost,
                    h=target_h,
                    f=cost + target_h,
                    depth=len(ops),
                    open_size=len(queue),
                    stats=stats,
                )
            for op in task.applicable_operators(state):
                succ = task.apply(op, state)
                stats.generated += 1
                if succ in visited:
                    continue
                visited.add(succ)
                new_ops = ops + [op]
                new_cost = cost + op.cost
                if task.goal_reached(succ):
                    return succ, new_ops, new_cost, 0
                value = heuristic(succ)
                stats.evaluated += 1
                if observer is not None:
                    observer.on_generate(
                        succ,
                        parent=state,
                        action=op.name,
                        g=new_cost,
                        h=value,
                        depth=len(new_ops),
                        stats=stats,
                    )
                if value < target_h:
                    return succ, new_ops, new_cost, value
                queue.append((succ, new_ops, new_cost))
        return None, None, None, None
