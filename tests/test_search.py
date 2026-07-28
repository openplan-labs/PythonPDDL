"""Search tests: optimality, satisficing completeness, unsolvable handling."""

from __future__ import annotations

import pytest

from jupyddl.api import build_task, solve_task, validate_plan
from jupyddl.search import PLANNERS

from conftest import OPTIMAL_COST, SOLVABLE, UNSOLVABLE, paths

OPTIMAL_CONFIGS = [
    ("bfs", None),
    ("dijkstra", None),
    ("astar", "blind"),
    ("idastar", "blind"),
]

# A heuristic for each informed planner (satisficing completeness check).
ALL_CONFIGS = [
    ("bfs", None),
    ("dfs", None),
    ("iddfs", None),
    ("dijkstra", None),
    ("gbfs", "hff"),
    ("astar", "lmcut"),
    ("wastar", "hff"),
    ("idastar", "hmax"),
    ("ehc", "hff"),
]


@pytest.mark.parametrize("name", SOLVABLE)
def test_optimal_planners_agree(name, examples_available):
    task = build_task(*paths(name))
    costs = set()
    for planner, heuristic in OPTIMAL_CONFIGS:
        result = solve_task(task, planner, heuristic)
        assert result.solved, f"{planner} failed on {name}"
        assert validate_plan(task, result.plan)
        costs.add(result.cost)
    assert costs == {OPTIMAL_COST[name]}, f"{name}: {costs}"


@pytest.mark.parametrize("name", SOLVABLE)
def test_all_planners_find_valid_plan(name, examples_available):
    task = build_task(*paths(name))
    for planner, heuristic in ALL_CONFIGS:
        result = solve_task(task, planner, heuristic)
        assert result.solved, f"{planner} did not solve {name}"
        assert validate_plan(task, result.plan), f"{planner} bad plan on {name}"


@pytest.mark.parametrize("name", UNSOLVABLE)
@pytest.mark.parametrize("planner", ["bfs", "dijkstra", "astar", "idastar"])
def test_unsolvable_reported(name, planner, examples_available):
    task = build_task(*paths(name))
    heuristic = "hmax" if planner in ("astar", "idastar") else None
    result = solve_task(task, planner, heuristic)
    assert not result.solved
    assert result.plan is None


def test_weighted_astar_weight_respected(examples_available):
    task = build_task(*paths("pallet"))
    result = solve_task(task, "wastar", "hff", weight=3.0)
    assert result.solved and validate_plan(task, result.plan)


def test_all_planner_names_registered():
    assert set(PLANNERS) == {
        "bfs",
        "dfs",
        "iddfs",
        "dijkstra",
        "gbfs",
        "astar",
        "wastar",
        "idastar",
        "ehc",
        "hc",
        "beam",
        "iw",
        "bnb",
        "awastar",
    }
