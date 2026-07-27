"""The bundled demo instances must stay solvable, and optimally so.

These double as regression tests for the grounder: between them the demos cover
typing hierarchies, action costs, conditional effects, static-predicate pruning
and an untyped domain.
"""

from __future__ import annotations

import os

import pytest

from conftest import DEMO_OPTIMAL_COST, DEMOS, demo_paths
from jupyddl import build_task, solve_task, validate_plan

DEMO_NAMES = sorted(DEMO_OPTIMAL_COST)


@pytest.fixture(scope="session", autouse=True)
def demos_available():
    if not os.path.isdir(DEMOS):
        pytest.skip("demos folder is missing")


@pytest.mark.parametrize("name", DEMO_NAMES)
def test_demo_grounds(name):
    task = build_task(*demo_paths(name))
    assert task.num_facts > 0
    assert len(task.operators) > 0
    assert task.goals


@pytest.mark.parametrize("name", DEMO_NAMES)
def test_demo_is_solved_optimally(name):
    """A* with an admissible heuristic must hit the known optimal cost."""
    task = build_task(*demo_paths(name))
    result = solve_task(task, "astar", "lmcut")
    assert result.solved
    assert validate_plan(task, result.plan)
    assert result.cost == DEMO_OPTIMAL_COST[name]


@pytest.mark.parametrize("name", DEMO_NAMES)
def test_satisficing_planner_finds_a_valid_plan(name):
    task = build_task(*demo_paths(name))
    result = solve_task(task, "gbfs", "hff")
    assert result.solved
    assert validate_plan(task, result.plan)
    # Greedy search may be suboptimal, but never cheaper than optimal.
    assert result.cost >= DEMO_OPTIMAL_COST[name]


def test_hanoi_matches_the_closed_form():
    """Five discs: the optimal plan is exactly 2**5 - 1 moves."""
    task = build_task(*demo_paths("hanoi"))
    result = solve_task(task, "astar", "lmcut")
    assert result.cost == 2**5 - 1 == 31
    assert result.plan_length == 31


def test_logistics_uses_action_costs():
    """Flying costs 6 and driving 2, so cost and plan length must differ."""
    task = build_task(*demo_paths("logistics"))
    result = solve_task(task, "astar", "lmcut")
    assert result.solved
    assert result.cost != result.plan_length


def test_elevator_has_conditional_effect_free_solution():
    task = build_task(*demo_paths("elevator"))
    result = solve_task(task, "bfs", None)
    assert result.solved
    assert validate_plan(task, result.plan)


def test_gripper_optimal_cost_is_reproduced_by_uninformed_search():
    """Cross-check the optimum with a planner that uses no heuristic at all."""
    task = build_task(*demo_paths("gripper"))
    blind = solve_task(task, "bfs", None)
    informed = solve_task(task, "astar", "lmcut")
    assert blind.cost == informed.cost == DEMO_OPTIMAL_COST["gripper"]
    # ...and the heuristic should have paid for itself.
    assert informed.stats.expanded <= blind.stats.expanded


def test_sokoban_static_predicates_are_pruned():
    """`move-dir` never appears in an effect, so it must not survive grounding."""
    task = build_task(*demo_paths("sokoban"))
    assert not any("move-dir" in fact for fact in task.facts)
