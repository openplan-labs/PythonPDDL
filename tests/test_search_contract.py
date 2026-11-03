"""The contract every planner keeps: budgets, truncation, and honest guarantees.

The older budget test ran on an instance small enough to finish before any
budget mattered, so three planners that never looked at their budget passed
it. These run where the budget is the only thing that stops the search.
"""

from __future__ import annotations

import io
import os
import time

import pytest

from jupyddl import build_task, solve_task
from jupyddl.live import TerminalDashboard, _clip
from jupyddl.search import OPTIMAL_PLANNERS, PLANNERS, SearchResult, SearchStats
from jupyddl.trace import MultiObserver, SearchObserver

from conftest import REPO_ROOT


def demo(name):
    folder = os.path.join(REPO_ROOT, "demos", name)
    return build_task(
        os.path.join(folder, "domain.pddl"), os.path.join(folder, "problem.pddl")
    )


@pytest.fixture(scope="module")
def hard():
    return demo("blocksworld12")


# Incomplete planners may give up on their own before any budget is reached;
# "no plan" from them was never a claim of unsolvability.
INCOMPLETE = {"hc", "ehc", "beam", "iw"}


def heuristic_for(name):
    return "goalcount" if PLANNERS[name].requires_heuristic else None


@pytest.mark.parametrize("name", sorted(PLANNERS))
def test_every_planner_stops_at_its_expansion_budget(hard, name):
    started = time.perf_counter()
    result = solve_task(hard, name, heuristic_for(name), max_expansions=100)
    assert time.perf_counter() - started < 20
    assert result.stats.expanded <= 100
    if name not in INCOMPLETE:
        assert result.solved or result.truncated


@pytest.mark.parametrize("name", sorted(PLANNERS))
def test_every_planner_stops_at_its_time_limit(hard, name):
    started = time.perf_counter()
    result = solve_task(hard, name, heuristic_for(name), time_limit=0.3)
    assert time.perf_counter() - started < 3
    if name not in INCOMPLETE:
        assert result.solved or result.truncated


def test_cost_optimal_planners_match_dijkstra_under_action_costs():
    task = demo("errands")  # action costs: shortest is not cheapest here
    optimum = solve_task(task, "dijkstra").cost
    for name in OPTIMAL_PLANNERS:
        result = solve_task(
            task, name, "lmcut" if PLANNERS[name].requires_heuristic else None
        )
        assert result.cost == optimum, name
    # bfs is shortest, not cheapest, which is why it is not in the set.
    assert "bfs" not in OPTIMAL_PLANNERS and "iddfs" not in OPTIMAL_PLANNERS
    assert solve_task(task, "bfs").cost > optimum


def test_a_depth_cap_is_reported_as_truncated_not_unsolvable():
    task = demo("hanoi")
    capped = solve_task(task, "iddfs", max_depth=2)
    assert not capped.solved and capped.truncated
    errands = demo("errands")
    shallow = solve_task(errands, "bnb", "lmcut", max_depth=3)
    assert shallow.truncated  # whatever it found is not a proof of optimality


@pytest.mark.parametrize("name", ["idastar", "ehc", "hc", "beam"])
def test_a_missing_heuristic_is_a_clear_error(name):
    with pytest.raises(ValueError, match="needs a heuristic"):
        PLANNERS[name]().search(demo("hanoi"))


def test_multi_observer_accepts_positional_arguments():
    seen = []

    class Recorder(SearchObserver):
        def on_expand(self, state, g=0.0, h=0.0, *rest, **kwargs):
            seen.append((g, h))

    MultiObserver(Recorder(), Recorder()).on_expand("s", 1.0, 2.0)
    assert seen == [(1.0, 2.0), (1.0, 2.0)]


# -- the live dashboard ---------------------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_the_dashboard_logs_sparingly_when_not_on_a_terminal(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr("jupyddl.live.time.perf_counter", clock)
    stream = io.StringIO()  # not a tty
    board = TerminalDashboard(stream=stream)
    board.on_start(demo("hanoi"), "astar", "hff")
    stats = SearchStats()
    for _ in range(75):  # six seconds of 0.08 s repaints
        clock.now += 0.08
        stats.expanded += 1
        board.on_expand("s", g=1, h=1, f=2, stats=stats)
    progress = [line for line in stream.getvalue().splitlines() if "expanded" in line]
    assert 1 <= len(progress) <= 3


def test_the_dashboard_says_a_budget_stopped_the_search():
    stream = io.StringIO()
    board = TerminalDashboard(stream=stream)
    stats = SearchStats(truncated=True)
    board.on_finish(SearchResult(False, None, None, stats))
    assert "budget" in stream.getvalue() and "no plan found" not in stream.getvalue()


def test_lines_are_clipped_to_the_terminal_width_with_colour_intact():
    line = "\x1b[1m" + "x" * 50 + "\x1b[0m" + "y" * 50
    clipped = _clip(line, 30)
    assert clipped.replace("\x1b[1m", "").replace("\x1b[0m", "") == "x" * 30
    assert clipped.endswith("\x1b[0m")
    assert _clip("short", 30) == "short"
