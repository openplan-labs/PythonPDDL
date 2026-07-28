"""Planner base class and the shared best-first search engine.

``best_first`` is a graph-search with re-opening, lazy heuristic evaluation and
dead-end pruning (a heuristic value of ``inf`` prunes the state). The concrete
best-first planners (uniform-cost, greedy, A*, weighted-A*) are just different
priority functions over ``(g, h)``.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time

from .node import extract_plan, make_child, make_root
from .result import SearchResult, SearchStats


class Planner:
    """Base class for all planners."""

    name: str = "planner"
    requires_heuristic: bool = False
    optimal: bool = False

    def search(
        self, task, heuristic=None, observer=None, budget=None
    ) -> SearchResult:  # pragma: no cover
        """Run the planner.

        ``budget`` is an optional :class:`~jupyddl.search.result.Budget`; when a
        planner stops because it ran out of budget it sets ``stats.truncated``,
        so "no plan" can be told apart from "gave up".
        """
        raise NotImplementedError


def heuristic_name(heuristic) -> str:
    """Best-effort display name for a heuristic (``""`` when there is none)."""
    return "" if heuristic is None else getattr(heuristic, "name", "custom")


def best_first(
    task, priority, heuristic=None, reopen=True, observer=None, budget=None
) -> SearchResult:
    """Generic best-first graph search.

    ``priority`` maps ``(g, h)`` to the open-list key; ``heuristic`` is an
    optional callable ``state -> number`` (``inf`` marks a dead end).
    ``observer`` is an optional :class:`~jupyddl.trace.SearchObserver` that is
    notified as the search progresses (see :mod:`jupyddl.trace`).
    """
    stats = SearchStats()
    start = time.perf_counter()
    init = task.initial_state()

    hcache: dict = {}

    def h_of(state) -> float:
        if heuristic is None:
            return 0.0
        cached = hcache.get(state)
        if cached is None:
            cached = heuristic(state)
            hcache[state] = cached
            stats.evaluated += 1
        return cached

    root = make_root(init)
    h0 = h_of(init)
    if math.isinf(h0):
        stats.runtime = time.perf_counter() - start
        result = SearchResult(False, None, None, stats)
        if observer is not None:
            observer.on_finish(result)
        return result

    counter = itertools.count()
    best_g: dict = {init: 0}
    open_list = [(priority(0, h0), next(counter), root)]

    while open_list:
        if budget is not None and budget.exceeded(stats):
            break
        _, _, node = heapq.heappop(open_list)
        if node.g > best_g.get(node.state, math.inf):
            continue  # stale, a cheaper path to this state was found
        if task.goal_reached(node.state):
            stats.runtime = time.perf_counter() - start
            result = SearchResult(True, extract_plan(node), node.g, stats)
            if observer is not None:
                observer.on_goal(node.state, node.g, node.depth, stats)
                observer.on_finish(result)
            return result
        stats.expanded += 1
        state = node.state
        if observer is not None:
            hv_node = hcache.get(state, 0.0) if heuristic is not None else 0.0
            observer.on_expand(
                state,
                g=node.g,
                h=hv_node,
                f=priority(node.g, hv_node),
                depth=node.depth,
                open_size=len(open_list),
                stats=stats,
                parent=None if node.parent is None else node.parent.state,
                action="" if node.action is None else node.action.name,
            )
        for op in task.operators:
            if not op.applicable(state):
                continue
            succ = task.apply(op, state)
            new_g = node.g + op.cost
            stats.generated += 1
            prev = best_g.get(succ, math.inf)
            if new_g >= prev:
                continue
            if not math.isinf(prev):
                if not reopen:
                    continue
                stats.reopened += 1
            hv = h_of(succ)
            if math.isinf(hv):
                stats.deadends += 1
                continue
            best_g[succ] = new_g
            child = make_child(node, op, succ, op.cost)
            if observer is not None:
                observer.on_generate(
                    succ,
                    parent=state,
                    action=op.name,
                    g=new_g,
                    h=hv,
                    depth=child.depth,
                    stats=stats,
                )
            heapq.heappush(open_list, (priority(new_g, hv), next(counter), child))

    stats.runtime = time.perf_counter() - start
    result = SearchResult(False, None, None, stats)
    if observer is not None:
        observer.on_finish(result)
    return result
