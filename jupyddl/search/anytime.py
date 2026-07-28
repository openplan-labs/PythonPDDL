"""Anytime and bounded-optimal planners.

Both of these keep searching after they find a plan: they report the best plan
so far and tighten it while time and memory allow. The trace records each
improvement, so an anytime run plots as a descending staircase of solution
quality rather than a single point.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time

from .base import Planner, heuristic_name
from .node import extract_plan, make_child, make_root
from .result import Budget, SearchResult, SearchStats


class BranchAndBound(Planner):
    """Depth-first branch and bound.

    Explores depth-first, remembering the cheapest plan found so far, and prunes
    any node whose ``g + h`` already reaches that bound. With an admissible
    heuristic the final answer is cost-optimal, and unlike A* the memory stays
    linear in the depth — it just usually visits many more nodes.
    """

    name = "bnb"
    requires_heuristic = True
    optimal = True

    def __init__(self, max_depth: int = 200, max_expansions: int = 200000):
        self.max_depth = max_depth
        # Depth-first branch and bound explores enormously more nodes than A*,
        # so it carries its own default ceiling. Hitting it sets
        # ``stats.truncated`` and returns the best plan found so far, which is
        # then no longer a proof of optimality.
        self.max_expansions = max_expansions

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        started = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, heuristic_name(heuristic))

        best_plan = None
        best_cost = math.inf
        start = task.initial_state()
        if budget is None and self.max_expansions:
            budget = Budget(max_expansions=self.max_expansions).start()

        def h_of(state):
            if heuristic is None:
                return 0.0
            stats.evaluated += 1
            return heuristic(state)

        def descend(node, on_path):
            nonlocal best_plan, best_cost
            if budget is not None and budget.exceeded(stats):
                return
            if task.goal_reached(node.state):
                if node.g < best_cost:
                    best_cost = node.g
                    best_plan = extract_plan(node)
                    if observer is not None:
                        observer.on_bound(float(best_cost), len(on_path), stats)
                        observer.on_goal(node.state, node.g, node.depth, stats)
                return
            if node.depth >= self.max_depth:
                return
            stats.expanded += 1
            if observer is not None:
                observer.on_expand(
                    node.state,
                    g=node.g,
                    h=0.0,
                    f=node.g,
                    depth=node.depth,
                    open_size=len(on_path),
                    stats=stats,
                    parent=None if node.parent is None else node.parent.state,
                    action="" if node.action is None else node.action.name,
                )
            for op in task.applicable_operators(node.state):
                succ = task.apply(op, node.state)
                stats.generated += 1
                if succ in on_path:
                    continue
                child = make_child(node, op, succ, op.cost)
                estimate = h_of(succ)
                if math.isinf(estimate):
                    stats.deadends += 1
                    continue
                if child.g + estimate >= best_cost:
                    continue  # cannot beat the incumbent
                on_path.add(succ)
                descend(child, on_path)
                on_path.discard(succ)

        descend(make_root(start), {start})

        stats.runtime = time.perf_counter() - started
        solved = best_plan is not None
        result = SearchResult(
            solved, best_plan, None if not solved else best_cost, stats
        )
        if observer is not None:
            observer.on_finish(result)
        return result


class AnytimeWeightedAStar(Planner):
    """Weighted A* that keeps going, lowering the weight after each solution.

    The first plan arrives quickly at a high weight; every later solution is
    cheaper, and the run ends with a weight of 1 (plain A*), so the last plan is
    optimal if the heuristic is admissible and the search completes. Each
    improvement is reported through ``on_bound``.
    """

    name = "awastar"
    requires_heuristic = True

    def __init__(self, weights=(5.0, 3.0, 2.0, 1.5, 1.0)):
        self.weights = tuple(float(w) for w in weights)

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        started = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, heuristic_name(heuristic))

        best_plan = None
        best_cost = math.inf
        start = task.initial_state()

        for index, weight in enumerate(self.weights):
            if observer is not None:
                observer.on_bound(weight, index, stats)
            found, cost = self._bounded(
                task, heuristic, weight, best_cost, stats, observer, start, budget
            )
            if stats.truncated:
                break
            if found is not None and cost < best_cost:
                best_plan, best_cost = found, cost
            if best_cost == 0:
                break

        stats.runtime = time.perf_counter() - started
        solved = best_plan is not None
        result = SearchResult(
            solved, best_plan, None if not solved else best_cost, stats
        )
        if observer is not None:
            observer.on_finish(result)
        return result

    def _bounded(
        self, task, heuristic, weight, incumbent, stats, observer, start, budget=None
    ):
        """One weighted-A* pass, pruning anything the incumbent already beats."""
        hcache: dict = {}

        def h_of(state):
            cached = hcache.get(state)
            if cached is None:
                cached = heuristic(state) if heuristic is not None else 0.0
                hcache[state] = cached
                stats.evaluated += 1
            return cached

        h0 = h_of(start)
        if math.isinf(h0):
            return None, math.inf

        counter = itertools.count()
        best_g = {start: 0}
        open_list = [(weight * h0, next(counter), make_root(start))]

        while open_list:
            if budget is not None and budget.exceeded(stats):
                break
            _, _, node = heapq.heappop(open_list)
            if node.g > best_g.get(node.state, math.inf):
                continue
            if node.g >= incumbent:
                continue
            if task.goal_reached(node.state):
                if observer is not None:
                    observer.on_goal(node.state, node.g, node.depth, stats)
                return extract_plan(node), node.g
            stats.expanded += 1
            if observer is not None:
                observer.on_expand(
                    node.state,
                    g=node.g,
                    h=hcache.get(node.state, 0.0),
                    f=node.g + weight * hcache.get(node.state, 0.0),
                    depth=node.depth,
                    open_size=len(open_list),
                    stats=stats,
                    parent=None if node.parent is None else node.parent.state,
                    action="" if node.action is None else node.action.name,
                )
            for op in task.applicable_operators(node.state):
                succ = task.apply(op, node.state)
                new_g = node.g + op.cost
                stats.generated += 1
                if new_g >= best_g.get(succ, math.inf) or new_g >= incumbent:
                    continue
                estimate = h_of(succ)
                if math.isinf(estimate):
                    stats.deadends += 1
                    continue
                best_g[succ] = new_g
                child = make_child(node, op, succ, op.cost)
                if observer is not None:
                    observer.on_generate(
                        succ,
                        parent=node.state,
                        action=op.name,
                        g=new_g,
                        h=estimate,
                        depth=child.depth,
                        stats=stats,
                    )
                heapq.heappush(
                    open_list, (new_g + weight * estimate, next(counter), child)
                )
        return None, math.inf
