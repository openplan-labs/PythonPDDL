"""Iterative-deepening A* (IDA*): optimal, memory-light informed search."""

from __future__ import annotations

import math
import time

from .base import Planner, heuristic_name
from .node import extract_plan, make_child, make_root
from .result import SearchResult, SearchStats


class IDAStarSearch(Planner):
    """IDA*: depth-first search bounded by an increasing ``f = g + h`` threshold.

    Cost-optimal with an admissible heuristic and uses memory linear in the
    solution depth.
    """

    name = "idastar"
    requires_heuristic = True
    optimal = True

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        start = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, heuristic_name(heuristic))
        hcache: dict = {}

        def h_of(state):
            v = hcache.get(state)
            if v is None:
                v = heuristic(state)
                hcache[state] = v
                stats.evaluated += 1
            return v

        start_state = task.initial_state()
        threshold = h_of(start_state)
        root = make_root(start_state)
        iteration = 0
        while not math.isinf(threshold):
            if observer is not None:
                observer.on_bound(float(threshold), iteration, stats)
            iteration += 1
            found, nxt = self._dfs(
                task, root, threshold, h_of, stats, {start_state}, observer
            )
            if found is not None:
                stats.runtime = time.perf_counter() - start
                result = SearchResult(True, extract_plan(found), found.g, stats)
                if observer is not None:
                    observer.on_goal(found.state, found.g, found.depth, stats)
                    observer.on_finish(result)
                return result
            threshold = nxt
        stats.runtime = time.perf_counter() - start
        result = SearchResult(False, None, None, stats)
        if observer is not None:
            observer.on_finish(result)
        return result

    def _dfs(self, task, node, threshold, h_of, stats, on_path, observer=None):
        h = h_of(node.state)
        f = node.g + h
        if f > threshold:
            return None, f
        if task.goal_reached(node.state):
            return node, threshold
        stats.expanded += 1
        if observer is not None:
            observer.on_expand(
                node.state,
                g=node.g,
                h=h,
                f=f,
                depth=node.depth,
                open_size=len(on_path),
                stats=stats,
                parent=None if node.parent is None else node.parent.state,
                action="" if node.action is None else node.action.name,
            )
        minimum = math.inf
        for op in task.applicable_operators(node.state):
            succ = task.apply(op, node.state)
            stats.generated += 1
            if succ in on_path:
                continue
            child = make_child(node, op, succ, op.cost)
            if observer is not None:
                observer.on_generate(
                    succ,
                    parent=node.state,
                    action=op.name,
                    g=child.g,
                    depth=child.depth,
                    stats=stats,
                )
            on_path.add(succ)
            found, nxt = self._dfs(
                task, child, threshold, h_of, stats, on_path, observer
            )
            on_path.discard(succ)
            if found is not None:
                return found, threshold
            minimum = min(minimum, nxt)
        return None, minimum
