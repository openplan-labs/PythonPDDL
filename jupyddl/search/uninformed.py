"""Uninformed planners: breadth-first, depth-first and iterative deepening."""

from __future__ import annotations

import time
from collections import deque

from .base import Planner
from .node import extract_plan, make_child, make_root
from .result import SearchResult, SearchStats


class BreadthFirstSearch(Planner):
    """FIFO breadth-first graph search. Optimal in number of actions."""

    name = "bfs"
    optimal = True  # for unit-cost / plan-length

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        start = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, "")
        start_state = task.initial_state()
        root = make_root(start_state)
        if task.goal_reached(start_state):
            stats.runtime = time.perf_counter() - start
            result = SearchResult(True, [], 0, stats)
            if observer is not None:
                observer.on_goal(start_state, 0, 0, stats)
                observer.on_finish(result)
            return result
        visited = {start_state}
        queue = deque([root])
        while queue:
            if budget is not None and budget.exceeded(stats):
                break
            node = queue.popleft()
            stats.expanded += 1
            if observer is not None:
                observer.on_expand(
                    node.state,
                    g=node.g,
                    depth=node.depth,
                    f=node.g,
                    open_size=len(queue),
                    stats=stats,
                    parent=None if node.parent is None else node.parent.state,
                    action="" if node.action is None else node.action.name,
                )
            for op in task.applicable_operators(node.state):
                succ = task.apply(op, node.state)
                stats.generated += 1
                if succ in visited:
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
                if task.goal_reached(succ):
                    stats.runtime = time.perf_counter() - start
                    result = SearchResult(True, extract_plan(child), child.g, stats)
                    if observer is not None:
                        observer.on_goal(succ, child.g, child.depth, stats)
                        observer.on_finish(result)
                    return result
                visited.add(succ)
                queue.append(child)
        stats.runtime = time.perf_counter() - start
        result = SearchResult(False, None, None, stats)
        if observer is not None:
            observer.on_finish(result)
        return result


class DepthFirstSearch(Planner):
    """LIFO depth-first graph search. Complete on finite spaces, not optimal."""

    name = "dfs"

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        start = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, "")
        start_state = task.initial_state()
        root = make_root(start_state)
        visited = {start_state}
        stack = [root]
        while stack:
            if budget is not None and budget.exceeded(stats):
                break
            node = stack.pop()
            if task.goal_reached(node.state):
                stats.runtime = time.perf_counter() - start
                result = SearchResult(True, extract_plan(node), node.g, stats)
                if observer is not None:
                    observer.on_goal(node.state, node.g, node.depth, stats)
                    observer.on_finish(result)
                return result
            stats.expanded += 1
            if observer is not None:
                observer.on_expand(
                    node.state,
                    g=node.g,
                    depth=node.depth,
                    f=node.g,
                    open_size=len(stack),
                    stats=stats,
                    parent=None if node.parent is None else node.parent.state,
                    action="" if node.action is None else node.action.name,
                )
            for op in task.applicable_operators(node.state):
                succ = task.apply(op, node.state)
                stats.generated += 1
                if succ in visited:
                    continue
                visited.add(succ)
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
                stack.append(child)
        stats.runtime = time.perf_counter() - start
        result = SearchResult(False, None, None, stats)
        if observer is not None:
            observer.on_finish(result)
        return result


class IterativeDeepeningSearch(Planner):
    """Iterative-deepening DFS. Optimal in plan length with low memory."""

    name = "iddfs"
    optimal = True

    def __init__(self, max_depth: int = 1000):
        self.max_depth = max_depth

    def search(self, task, heuristic=None, observer=None, budget=None) -> SearchResult:
        stats = SearchStats()
        start = time.perf_counter()
        if observer is not None:
            observer.on_start(task, self.name, "")
        start_state = task.initial_state()
        for limit in range(self.max_depth + 1):
            if observer is not None:
                observer.on_bound(float(limit), limit, stats)
            found, cutoff = self._dls(
                task, make_root(start_state), limit, stats, {start_state}, observer
            )
            if found is not None:
                stats.runtime = time.perf_counter() - start
                result = SearchResult(True, extract_plan(found), found.g, stats)
                if observer is not None:
                    observer.on_goal(found.state, found.g, found.depth, stats)
                    observer.on_finish(result)
                return result
            if not cutoff:  # search exhausted without hitting the depth limit
                break
        stats.runtime = time.perf_counter() - start
        result = SearchResult(False, None, None, stats)
        if observer is not None:
            observer.on_finish(result)
        return result

    def _dls(self, task, node, limit, stats, on_path, observer=None):
        if task.goal_reached(node.state):
            return node, False
        if node.depth == limit:
            return None, True
        stats.expanded += 1
        if observer is not None:
            observer.on_expand(
                node.state,
                g=node.g,
                depth=node.depth,
                f=node.g,
                open_size=len(on_path),
                stats=stats,
                parent=None if node.parent is None else node.parent.state,
                action="" if node.action is None else node.action.name,
            )
        cutoff = False
        for op in task.applicable_operators(node.state):
            succ = task.apply(op, node.state)
            stats.generated += 1
            if succ in on_path:
                continue
            child = make_child(node, op, succ, op.cost)
            on_path.add(succ)
            found, cut = self._dls(task, child, limit, stats, on_path, observer)
            on_path.discard(succ)
            if found is not None:
                return found, False
            cutoff = cutoff or cut
        return None, cutoff
