"""The local-search, novelty and anytime planners, plus search budgets."""

from __future__ import annotations

import pytest

from conftest import demo_paths, paths
from jupyddl import build_task, solve_task, validate_plan
from jupyddl.search import (
    INFORMED_PLANNERS,
    OPTIMAL_PLANNERS,
    PLANNERS,
    describe_planners,
    make_planner,
)
from jupyddl.search.result import Budget, SearchStats, make_budget

NEW_PLANNERS = ["hc", "beam", "iw", "bnb", "awastar"]
# Iterated Width is deliberately incomplete: IW(2) prunes any state that makes
# no new pair of facts true, so plenty of solvable instances are out of reach.
# That is the trade it exists to make, not a defect, so it is excluded from the
# "must find a plan" set and covered by its own test instead.
COMPLETE_ENOUGH = [name for name in NEW_PLANNERS if name != "iw"]


def heuristic_for(name):
    return "hff" if name in INFORMED_PLANNERS else None


@pytest.mark.parametrize("planner", COMPLETE_ENOUGH)
def test_new_planners_solve_a_small_instance(examples_available, planner):
    task = build_task(*paths("pallet"))
    result = solve_task(
        task, planner, heuristic_for(planner), max_expansions=200000, time_limit=30
    )
    assert result.solved, f"{planner} failed on a small instance"
    assert validate_plan(task, result.plan)


@pytest.mark.parametrize("planner", NEW_PLANNERS)
def test_new_planners_report_statistics(examples_available, planner):
    task = build_task(*paths("tsp"))
    result = solve_task(task, planner, heuristic_for(planner), time_limit=30)
    assert result.stats.expanded > 0
    assert result.stats.runtime >= 0


def test_branch_and_bound_is_optimal_when_it_finishes():
    task = build_task(*demo_paths("hanoi"))
    optimal = solve_task(task, "astar", "lmcut")
    bnb = solve_task(task, "bnb", "hmax", max_expansions=500000, time_limit=60)
    if bnb.truncated:
        pytest.skip("branch and bound hit its budget on this machine")
    assert bnb.cost == optimal.cost


def test_anytime_weighted_astar_ends_at_the_optimum(examples_available):
    task = build_task(*paths("pallet"))
    optimal = solve_task(task, "astar", "hmax")
    anytime = solve_task(task, "awastar", "hmax", time_limit=60)
    assert anytime.solved
    assert anytime.cost == optimal.cost


def test_iterated_width_needs_no_heuristic():
    """IW is the one informed-looking planner that is purely structural."""
    assert "iw" not in INFORMED_PLANNERS
    task = build_task(*demo_paths("gripper"))
    result = solve_task(task, "iw", None, max_expansions=100000, time_limit=30)
    # IW(2) does not solve everything -- that is the point of the width ladder.
    assert result.stats.expanded > 0


def test_beam_width_changes_the_search(examples_available):
    task = build_task(*paths("pallet"))
    narrow = solve_task(task, "beam", "hff", width=1, time_limit=30)
    wide = solve_task(task, "beam", "hff", width=50, time_limit=30)
    assert wide.stats.expanded >= narrow.stats.expanded


def test_hill_climbing_is_reproducible(examples_available):
    """Random tie-breaking is seeded, so two runs agree."""
    task = build_task(*paths("pallet"))
    first = solve_task(task, "hc", "hff", time_limit=30)
    second = solve_task(task, "hc", "hff", time_limit=30)
    assert first.solved == second.solved
    assert first.cost == second.cost


# ---------------------------------------------------------------- budgets
def test_budget_stops_on_expansions(examples_available):
    task = build_task(*paths("pallet"))
    result = solve_task(task, "bfs", None, max_expansions=5)
    assert result.stats.expanded <= 5
    assert result.truncated


def test_budget_stops_on_time():
    """A short limit must be honoured even when each expansion is expensive."""
    task = build_task(*demo_paths("blocksworld12"))
    result = solve_task(task, "astar", "lmcut", time_limit=0.5)
    assert result.truncated
    # LM-cut expansions cost tens of milliseconds here, so the clock has to be
    # read every expansion rather than every few hundred.
    assert result.stats.runtime < 5


def test_make_budget_returns_none_when_unlimited():
    assert make_budget(None, None) is None
    assert make_budget(10, None) is not None


def test_budget_marks_stats_when_exceeded():
    budget = Budget(max_expansions=3).start()
    stats = SearchStats()
    stats.expanded = 3
    assert budget.exceeded(stats)
    assert stats.truncated


def test_unlimited_budget_never_fires():
    budget = Budget().start()
    stats = SearchStats()
    stats.expanded = 10**6
    assert not budget.exceeded(stats)
    assert not stats.truncated


# --------------------------------------------------------------- registry
def test_registry_metadata_is_consistent():
    described = {row["name"] for row in describe_planners()}
    assert described == set(PLANNERS)
    for row in describe_planners():
        assert row["summary"], f"{row['name']} has no docstring summary"


def test_optimal_and_informed_lists_match_the_classes():
    for name in OPTIMAL_PLANNERS:
        assert make_planner(name).optimal
    for name in INFORMED_PLANNERS:
        assert make_planner(name).requires_heuristic


def test_every_planner_accepts_a_budget(examples_available):
    """A uniform signature is what lets the harness bound any configuration."""
    task = build_task(*paths("tsp"))
    for name in sorted(PLANNERS):
        result = solve_task(
            task, name, heuristic_for(name), max_expansions=2000, time_limit=20
        )
        assert result.stats is not None
