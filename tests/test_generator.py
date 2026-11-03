"""The instance generators.

The contract is: whatever a generator emits must parse, ground, and be solvable,
and the same seed must always produce the same bytes. A generator that quietly
emits unsolvable instances would poison every benchmark built on it.
"""

from __future__ import annotations

import os
import re

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


def _content(text: str) -> str:
    """Problem text without comments or the problem's name, both of which
    carry the seed whether or not anything else does."""
    text = re.sub(r";;[^\n]*", "", text)
    return re.sub(r"\(problem [^\s)]+", "(problem _", text)


@pytest.mark.parametrize("kind", NAMES)
def test_different_seeds_usually_differ(kind):
    """Seeds must actually vary the instance, or a ladder is one problem twice.

    Compared without the name and comments: gripper and workshop used to put
    the seed there and nowhere else, which passed a byte comparison while
    twenty seeds gave one instance — so a split by seed held out a copy of a
    training instance, and the RL stage's tuning and validation sets were the
    same problems.
    """
    varied = {_content(generate(kind, size=6, seed=seed)[1]) for seed in range(6)}
    assert len(varied) > 1, f"{kind} ignores its seed"
    smaller = {_content(generate(kind, size=3, seed=seed)[1]) for seed in range(20)}
    assert len(smaller) >= 4, f"{kind} has only {len(smaller)} instances at size 3"


@pytest.mark.parametrize("kind", NAMES)
def test_no_instance_is_solved_before_it_starts(kind):
    """A goal that holds in the initial state teaches a learner nothing and
    flatters every planner. Blocksworld at two blocks did this for 75 of the
    first 200 seeds."""
    for size in (2, 3, 4):
        for seed in range(40):
            task = build(kind, size=size, seed=seed)
            assert not task.goal_reached(task.initial_state()), (kind, size, seed)


def test_random_strips_never_samples_a_goal_that_already_holds():
    """The goal is drawn from the end of the planted chain, and could land
    entirely on facts the initial state already had."""
    for size in range(9, 13):
        for seed in range(60):
            task = build("random-strips", size=size, seed=seed)
            assert not task.goal_reached(task.initial_state()), (size, seed)


def test_random_strips_survives_a_chain_that_undoes_itself():
    """Seed 11 at size 9 with four actions plants a chain whose end state is a
    subset of its start, so every fact a goal could be drawn from already
    holds. One more planted step gives the goal something to ask for."""
    task = build("random-strips", size=9, seed=11, actions=4)
    assert not task.goal_reached(task.initial_state())
    result = solve_task(task, "astar", "blind")
    assert result.solved and validate_plan(task, result.plan)


def test_only_previously_trivial_instances_changed():
    """The re-draws above happen only where an instance was trivial, so every
    other seed still emits the bytes it always did. ``demos/blocksworld12`` is
    published as exactly this generator's output."""
    demo = os.path.join(os.path.dirname(__file__), "..", "demos", "blocksworld12")
    with open(os.path.join(demo, "problem.pddl"), encoding="utf-8") as handle:
        assert generate("blocksworld", size=12, seed=3)[1] == handle.read()


def test_gripper_and_workshop_seeds_change_where_things_start():
    gripper = build("gripper", size=4, seed=5)
    assert len(gripper.goals) == 4  # every ball still has to move

    def initial(kind, size, seed):
        return generate(kind, size=size, seed=seed)[1].split("(:goal")[0]

    starts = {initial("gripper", 4, seed).count(" rooma)") for seed in range(20)}
    assert len(starts) > 1, "every seed puts the balls in the same rooms"

    stages = {initial("workshop", 3, seed).count("(cut ") for seed in range(20)}
    assert len(stages) > 1, "every seed starts the parts at the same stage"


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
