"""Local-search and bounded planners: hill climbing, beam search, novelty search.

These trade completeness for speed and memory. They are the planners you reach
for when the state space is far too large to exhaust and a good-enough plan
beats no plan — and they make instructive contrasts against A* in a benchmark.
"""

from __future__ import annotations

import math
import random
import time

from .base import Planner, heuristic_name
from .node import extract_plan, make_child, make_root
from .result import SearchResult, SearchStats
from ..task import facts_of


class HillClimbing(Planner):
    """Steepest-ascent hill climbing with a sideways-move allowance.

    Always steps to the best successor. ``max_sideways`` consecutive moves that
    do not improve ``h`` are tolerated before the search gives up, which is what
    lets it cross small plateaus; ``restarts`` re-runs from the initial state
    with a random tie-break when it does. Not complete and not optimal.
    """

    name = "hc"
    requires_heuristic = True

    def __init__(self, max_sideways: int = 30, restarts: int = 3, seed: int = 0):
        self.max_sideways = max_sideways
        self.restarts = restarts
        self.seed = seed

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        started = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, heuristic_name(heuristic))
        rng = random.Random(self.seed)

        for attempt in range(self.restarts + 1):
            if observer is not None:
                observer.on_bound(float(attempt), attempt, stats)
            outcome = self._climb(task, heuristic, stats, rng, observer, budget)
            if outcome is not None:
                plan, cost, state = outcome
                stats.runtime = time.perf_counter() - started
                result = SearchResult(True, plan, cost, stats)
                if observer is not None:
                    observer.on_goal(state, cost, len(plan), stats)
                    observer.on_finish(result)
                return result

        stats.runtime = time.perf_counter() - started
        result = SearchResult(False, None, None, stats)
        if observer is not None:
            observer.on_finish(result)
        return result

    def _climb(self, task, heuristic, stats, rng, observer, budget=None):
        state = task.initial_state()
        current_h = heuristic(state)
        stats.evaluated += 1
        plan: list = []
        cost = 0
        sideways = 0
        seen = {state}

        while not task.goal_reached(state):
            if budget is not None and budget.exceeded(stats):
                return None
            if observer is not None:
                observer.on_expand(
                    state,
                    g=cost,
                    h=current_h,
                    f=cost + current_h,
                    depth=len(plan),
                    stats=stats,
                )
            stats.expanded += 1

            candidates = []
            for op in task.applicable_operators(state):
                succ = task.apply(op, state)
                stats.generated += 1
                if succ in seen:
                    continue
                value = heuristic(succ)
                stats.evaluated += 1
                if math.isinf(value):
                    stats.deadends += 1
                    continue
                candidates.append((value, op, succ))

            if not candidates:
                return None
            best = min(value for value, _, _ in candidates)
            tied = [entry for entry in candidates if entry[0] == best]
            value, op, succ = rng.choice(tied)

            if value > current_h:
                return None  # every move is strictly worse
            sideways = sideways + 1 if value == current_h else 0
            if sideways > self.max_sideways:
                return None

            seen.add(succ)
            plan.append(op)
            cost += op.cost
            state = succ
            current_h = value

        return plan, cost, state


class BeamSearch(Planner):
    """Breadth-first search that keeps only the ``width`` best nodes per layer.

    Memory is bounded by the beam width rather than by the frontier, which makes
    it usable where greedy best-first runs out of room. Pruning whole layers
    costs completeness and optimality.
    """

    name = "beam"
    requires_heuristic = True

    def __init__(self, width: int = 20):
        self.width = max(1, int(width))

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        started = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, heuristic_name(heuristic))

        start = task.initial_state()
        root = make_root(start)
        if task.goal_reached(start):
            stats.runtime = time.perf_counter() - started
            result = SearchResult(True, [], 0, stats)
            if observer is not None:
                observer.on_finish(result)
            return result

        layer = [root]
        seen = {start}
        depth = 0
        while layer:
            if budget is not None and budget.exceeded(stats):
                break
            depth += 1
            scored = []
            for node in layer:
                stats.expanded += 1
                if observer is not None:
                    observer.on_expand(
                        node.state,
                        g=node.g,
                        depth=node.depth,
                        open_size=len(layer),
                        stats=stats,
                        parent=None if node.parent is None else node.parent.state,
                        action="" if node.action is None else node.action.name,
                    )
                for op in task.applicable_operators(node.state):
                    succ = task.apply(op, node.state)
                    stats.generated += 1
                    if succ in seen:
                        continue
                    seen.add(succ)
                    child = make_child(node, op, succ, op.cost)
                    if task.goal_reached(succ):
                        stats.runtime = time.perf_counter() - started
                        result = SearchResult(True, extract_plan(child), child.g, stats)
                        if observer is not None:
                            observer.on_goal(succ, child.g, child.depth, stats)
                            observer.on_finish(result)
                        return result
                    value = heuristic(succ)
                    stats.evaluated += 1
                    if math.isinf(value):
                        stats.deadends += 1
                        continue
                    scored.append((value, child))

            scored.sort(key=lambda entry: entry[0])
            layer = [child for _, child in scored[: self.width]]
            if observer is not None and layer:
                observer.on_bound(float(depth), depth, stats)

        stats.runtime = time.perf_counter() - started
        result = SearchResult(False, None, None, stats)
        if observer is not None:
            observer.on_finish(result)
        return result


class IteratedWidth(Planner):
    """Iterated Width: blind search pruned by *novelty*.

    A state is novel at width ``w`` when it makes some set of ``w`` facts true
    together for the first time in the search. IW(1) then IW(2)... explores a
    tiny fraction of the space yet solves a surprising number of benchmark
    instances, and it needs no heuristic at all — the appeal is that it is
    domain-independent width, not domain knowledge.
    """

    name = "iw"
    requires_heuristic = False

    def __init__(self, max_width: int = 2):
        self.max_width = max(1, int(max_width))

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        started = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, "")

        for width in range(1, self.max_width + 1):
            if observer is not None:
                observer.on_bound(float(width), width, stats)
            found = self._run(task, width, stats, observer, budget)
            if stats.truncated:
                break
            if found is not None:
                stats.runtime = time.perf_counter() - started
                result = SearchResult(True, extract_plan(found), found.g, stats)
                if observer is not None:
                    observer.on_goal(found.state, found.g, found.depth, stats)
                    observer.on_finish(result)
                return result

        stats.runtime = time.perf_counter() - started
        result = SearchResult(False, None, None, stats)
        if observer is not None:
            observer.on_finish(result)
        return result

    def _run(self, task, width, stats, observer, budget=None):
        from itertools import combinations

        start = task.initial_state()
        if task.goal_reached(start):
            return make_root(start)
        seen_tuples: set = set()

        def novel(state) -> bool:
            """True if some size-<=width fact tuple is seen here for the first time."""
            facts = sorted(facts_of(state))
            fresh = False
            for size in range(1, width + 1):
                if size > len(facts):
                    break
                for combo in combinations(facts, size):
                    if combo not in seen_tuples:
                        seen_tuples.add(combo)
                        fresh = True
            return fresh

        novel(start)
        queue = [make_root(start)]
        while queue:
            if budget is not None and budget.exceeded(stats):
                return None
            node = queue.pop(0)
            stats.expanded += 1
            if observer is not None:
                observer.on_expand(
                    node.state,
                    g=node.g,
                    depth=node.depth,
                    open_size=len(queue),
                    stats=stats,
                    parent=None if node.parent is None else node.parent.state,
                    action="" if node.action is None else node.action.name,
                )
            for op in task.applicable_operators(node.state):
                succ = task.apply(op, node.state)
                stats.generated += 1
                child = make_child(node, op, succ, op.cost)
                if task.goal_reached(succ):
                    return child
                if novel(succ):
                    queue.append(child)
        return None
