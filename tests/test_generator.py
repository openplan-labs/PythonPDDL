"""The instance generators.

The contract is: whatever a generator emits must parse, ground, and be solvable,
and the same seed must always produce the same bytes. A generator that quietly
emits unsolvable instances would poison every benchmark built on it.
"""

from __future__ import annotations

import os

import pytest

from jupyddl import solve_task, validate_plan
from jupyddl.generator import (
    GENERATORS,
    describe_generators,
    generate,
    write_instance,
)
from jupyddl.grounding import ground
from jupyddl.parser import parse

NAMES = sorted(GENERATORS)


def build(kind, **kwargs):
    domain, problem = generate(kind, **kwargs)
    return ground(parse(domain), parse(problem))


@pytest.mark.parametrize("kind", NAMES)
def test_generated_instances_parse_and_ground(kind):
    task = build(kind, size=3, seed=1)
    assert task.num_facts > 0
    assert task.operators
    assert task.goals


@pytest.mark.parametrize("kind", NAMES)
def test_generated_instances_are_solvable(kind):
    """Every generator plants a reachable goal — including the random one."""
    task = build(kind, size=3, seed=2)
    result = solve_task(task, "gbfs", "hff", max_expansions=100000, time_limit=30)
    assert result.solved, f"{kind} produced an unsolvable instance"
    assert validate_plan(task, result.plan)


@pytest.mark.parametrize("kind", NAMES)
def test_same_seed_gives_identical_bytes(kind):
    first = generate(kind, size=5, seed=7)
    second = generate(kind, size=5, seed=7)
    assert first == second


@pytest.mark.parametrize("kind", NAMES)
def test_different_seeds_usually_differ(kind):
    """Seeds must actually vary the instance, or a ladder is one problem twice."""
    varied = {generate(kind, size=6, seed=seed)[1] for seed in range(6)}
    assert len(varied) > 1, f"{kind} ignores its seed"


@pytest.mark.parametrize("kind", ["gripper", "blocksworld", "logistics"])
def test_size_scales_the_instance(kind):
    small = build(kind, size=2, seed=0)
    large = build(kind, size=6, seed=0)
    assert large.num_facts > small.num_facts
    assert len(large.operators) > len(small.operators)


def test_gripper_plan_length_grows_with_the_ball_count():
    costs = []
    for size in (2, 4, 6):
        task = build("gripper", size=size, seed=0)
        result = solve_task(task, "astar", "lmcut")
        assert result.solved
        costs.append(result.cost)
    assert costs == sorted(costs) and costs[0] < costs[-1]


def test_numeric_generator_produces_numeric_fluents():
    task = build("numeric-transport", size=2, seed=3)
    assert task.numeric
    assert any(name.startswith("(fuel") for name in task.numeric_names)


def test_workshop_generator_produces_durations():
    task = build("workshop", size=2, seed=0)
    assert task.temporal
    assert all(op.duration > 0 for op in task.operators)


def test_rovers_generator_uses_adl():
    task = build("rovers", size=2, seed=1)
    # `report` has a disjunctive precondition, so it splits into several
    # operators that share one action name.
    reports = [op for op in task.operators if op.base_name.startswith("report")]
    assert reports


def test_unknown_generator_is_rejected():
    with pytest.raises(ValueError, match="Unknown generator"):
        generate("not-a-domain")


def test_describe_generators_covers_the_registry():
    described = {entry["name"] for entry in describe_generators()}
    assert described == set(NAMES)
    assert all(entry["summary"] for entry in describe_generators())


def test_write_instance_creates_a_loadable_folder(tmp_path):
    folder = write_instance("gripper", str(tmp_path), size=3, seed=4)
    assert os.path.exists(os.path.join(folder, "domain.pddl"))
    assert os.path.exists(os.path.join(folder, "problem.pddl"))

    from jupyddl import build_task

    task = build_task(
        os.path.join(folder, "domain.pddl"), os.path.join(folder, "problem.pddl")
    )
    result = solve_task(task, "gbfs", "hff")
    assert result.solved and validate_plan(task, result.plan)


def test_generated_ladder_gets_monotonically_harder():
    """A benchmark ladder is only useful if the rungs actually differ."""
    expansions = []
    for size in (2, 4, 6, 8):
        task = build("gripper", size=size, seed=0)
        result = solve_task(task, "bfs", None, max_expansions=200000)
        assert result.solved
        expansions.append(result.stats.expanded)
    assert expansions == sorted(expansions)
    assert expansions[-1] > expansions[0] * 4
