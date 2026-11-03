"""Regression tests for the task model: each one used to return a wrong answer.

Every case here passed the rest of the suite while being wrong — a plan that
`validate_plan` accepted because it replays through the same `Task.apply`, an
"optimal" cost that was not, a heuristic that crashed on a state type the other
tests never built. They pin PDDL semantics on tiny hand-written tasks whose
right answer is obvious by inspection.
"""

from __future__ import annotations

import math
from functools import partial

import pytest

from jupyddl import ground, parse, solve_task
from jupyddl.heuristics import HEURISTICS, make_heuristic
from jupyddl.parser import PDDLError, UnsupportedFeatureError


def task_of(domain: str, problem: str):
    return ground(parse(domain), parse(problem))


def names(task, state):
    return sorted(task.facts[f] for f in state)


def plan_of(result):
    return [op.name for op in result.plan]


# -- derived predicates ---------------------------------------------------------

LAMP = """
(define (domain lamp)
  (:requirements :strips :derived-predicates :negative-preconditions)
  (:predicates (on) (light) (switched-off) (bad) (dark-seen))
  (:derived (light) (on))
  (:action off :precondition (on) :effect (and (not (on)) (switched-off)))
  (:action check :precondition (and (switched-off) (light)) :effect (bad))
  (:action look-dark :precondition (not (light)) :effect (dark-seen)))
"""


def test_a_derived_fact_is_retracted_when_its_support_goes():
    task = task_of(
        LAMP, "(define (problem p) (:domain lamp) (:init (on)) (:goal (bad)))"
    )
    off = next(op for op in task.operators if op.name == "off")
    after = task.apply(off, task.initial_state())
    assert "(light)" not in names(task, after)
    assert not solve_task(task, "bfs").solved  # check needs light after off


def test_a_negated_derived_fact_holds_exactly_when_it_is_not_derived():
    task = task_of(
        LAMP, "(define (problem p) (:domain lamp) (:init (on)) (:goal (dark-seen)))"
    )
    assert "(not (light))" not in names(task, task.initial_state())
    assert plan_of(solve_task(task, "bfs")) == ["off", "look-dark"]
    # The relaxation must not call that state a dead end: nothing *adds*
    # (not (light)), it holds once (on) is gone.
    for name in ("hmax", "hff", "lmcut", "h2"):
        assert solve_task(task, "astar", name).solved, name


def test_negating_a_derived_predicate_inside_a_rule_is_refused():
    domain = """
    (define (domain d) (:requirements :strips :derived-predicates :negative-preconditions)
      (:predicates (a) (b) (c))
      (:derived (b) (a))
      (:derived (c) (not (b)))
      (:action go :effect (a)))"""
    with pytest.raises(UnsupportedFeatureError, match="stratified"):
        task_of(domain, "(define (problem p) (:domain d) (:init) (:goal (c)))")


# -- negative literals ----------------------------------------------------------

WALK = """
(define (domain walk)
  (:requirements :strips :typing :negative-preconditions)
  (:types loc)
  (:predicates (at ?l - loc) (left ?l - loc))
  (:action move :parameters (?from ?to - loc)
    :precondition (at ?from)
    :effect (and (not (at ?from)) (at ?to)))
  (:action note-absent :parameters (?l - loc)
    :precondition (not (at ?l))
    :effect (left ?l)))
"""


def test_adding_and_deleting_one_atom_leaves_its_complement_false():
    task = task_of(
        WALK,
        """(define (problem p) (:domain walk) (:objects a b - loc)
             (:init (at a)) (:goal (and (left a) (at a))))""",
    )
    stay = next(op for op in task.operators if op.name == "move(a,a)")
    assert names(task, task.apply(stay, task.initial_state())) == [
        "(at a)",
        "(not (at b))",
    ]
    assert solve_task(task, "bfs").cost == 3  # leave a, note it, come back


def test_a_conditional_add_beats_an_unconditional_delete_for_the_complement():
    domain = """
    (define (domain c) (:requirements :strips :negative-preconditions :conditional-effects)
      (:predicates (p) (q) (r))
      (:action t :effect (and (not (p)) (when (q) (p))))
      (:action unq :effect (not (q)))  ; (q) must be a fluent, or `when` folds away
      (:action needs-not-p :precondition (not (p)) :effect (r)))"""
    task = task_of(
        domain, "(define (problem x) (:domain c) (:init (p) (q)) (:goal (r)))"
    )
    t = next(op for op in task.operators if op.name == "t")
    assert t.cond_effects  # the runtime path, not one grounding resolved
    assert names(task, task.apply(t, task.initial_state())) == ["(p)", "(q)"]
    assert plan_of(solve_task(task, "bfs")) == ["unq", "t", "needs-not-p"]


# -- action costs ---------------------------------------------------------------


def test_under_a_total_cost_metric_an_action_without_a_cost_is_free():
    domain = """
    (define (domain free) (:requirements :strips :action-costs)
      (:predicates (a) (b) (g)) (:functions (total-cost))
      (:action step1 :precondition (a) :effect (b))
      (:action step2 :precondition (b) :effect (g))
      (:action pay :precondition (a) :effect (and (g) (increase (total-cost) 1))))"""
    problem = """(define (problem p) (:domain free) (:init (a) (= (total-cost) 0))
      (:goal (g)) (:metric minimize (total-cost)))"""
    task = task_of(domain, problem)
    assert {op.name: op.cost for op in task.operators} == {
        "step1": 0,
        "step2": 0,
        "pay": 1,
    }
    assert solve_task(task, "dijkstra").cost == 0
    # Without the metric every action costs 1 and shortest wins.
    plain = task_of(
        domain, "(define (problem p) (:domain free) (:init (a)) (:goal (g)))"
    )
    assert plan_of(solve_task(plain, "dijkstra")) == ["pay"]


def test_a_cost_inside_when_is_refused_rather_than_always_charged():
    domain = """
    (define (domain toll) (:requirements :strips :conditional-effects :action-costs)
      (:predicates (rush) (there)) (:functions (total-cost))
      (:action go :effect (and (there) (when (rush) (increase (total-cost) 10)))))"""
    problem = """(define (problem p) (:domain toll) (:init (= (total-cost) 0))
      (:goal (there)) (:metric minimize (total-cost)))"""
    with pytest.raises(UnsupportedFeatureError, match="inside a 'when'"):
        task_of(domain, problem)


def test_a_negative_action_cost_is_refused():
    domain = """
    (define (domain n) (:requirements :strips :action-costs)
      (:predicates (g)) (:functions (total-cost))
      (:action refund :effect (and (g) (increase (total-cost) -5))))"""
    problem = """(define (problem p) (:domain n) (:init) (:goal (g))
      (:metric minimize (total-cost)))"""
    with pytest.raises(PDDLError, match="negative cost"):
        task_of(domain, problem)


# -- numeric fluents ------------------------------------------------------------


def test_a_fluent_that_is_never_defined_fails_every_condition_reading_it():
    domain = """
    (define (domain roads) (:requirements :strips :typing :numeric-fluents)
      (:types loc) (:predicates (at ?l - loc))
      (:functions (distance ?a ?b - loc) (fuel))
      (:action drive :parameters (?a ?b - loc)
        :precondition (and (at ?a) (>= (fuel) (distance ?a ?b)))
        :effect (and (not (at ?a)) (at ?b) (decrease (fuel) (distance ?a ?b)))))"""
    problem = """(define (problem p) (:domain roads) (:objects a b c - loc)
      (:init (at a) (= (fuel) 100) (= (distance a b) 5) (= (distance b c) 5))
      (:goal (at c)))"""
    task = task_of(domain, problem)
    assert plan_of(solve_task(task, "bfs")) == ["drive(a,b)", "drive(b,c)"]
    assert "(distance a c)" not in task.numeric_names


# -- heuristics -----------------------------------------------------------------

DETOUR = """
(define (domain z) (:requirements :strips :action-costs)
  (:predicates (s) (x) (g)) (:functions (total-cost))
  (:action cheap :precondition (s) :effect (and (x) (not (s)) (increase (total-cost) 2)))
  (:action free :precondition (x) :effect (and (g) (increase (total-cost) 0)))
  (:action direct :precondition (s) :effect (and (g) (not (s)) (increase (total-cost) 3))))
"""
DETOUR_P = """(define (problem p) (:domain z) (:init (s) (= (total-cost) 0))
  (:goal (g)) (:metric minimize (total-cost)))"""


def test_blind_counts_free_actions_so_astar_stays_optimal():
    task = task_of(DETOUR, DETOUR_P)
    assert (
        solve_task(task, "astar", "blind").cost
        == 2
        == solve_task(task, "dijkstra").cost
    )


@pytest.mark.parametrize("name", sorted(HEURISTICS))
def test_every_heuristic_handles_an_already_satisfied_empty_goal(name):
    domain = """(define (domain e) (:requirements :strips)
      (:predicates (a)) (:action go :effect (a)))"""
    task = task_of(domain, "(define (problem p) (:domain e) (:init) (:goal (and)))")
    assert make_heuristic(name, task)(task.initial_state()) == 0


@pytest.mark.parametrize("name", sorted(HEURISTICS))
def test_every_heuristic_accepts_a_numeric_state(name):
    from conftest import REPO_ROOT
    import os

    folder = os.path.join(REPO_ROOT, "demos", "numeric-transport")
    task = ground(
        parse(open(os.path.join(folder, "domain.pddl")).read()),
        parse(open(os.path.join(folder, "problem.pddl")).read()),
    )
    value = make_heuristic(name, task)(task.initial_state())
    assert value >= 0 and not math.isnan(value)


def test_a_heuristic_class_or_partial_is_built_for_the_task():
    task = task_of(DETOUR, DETOUR_P)
    for factory in (HEURISTICS["lmcut"], partial(HEURISTICS["hm"], m=2)):
        assert make_heuristic(factory, task)(task.initial_state()) == 2


def test_admissibility_flags_follow_conditional_effects():
    plain = task_of(DETOUR, DETOUR_P)
    conditional = task_of(
        """(define (domain c) (:requirements :strips :conditional-effects)
             (:predicates (p) (q))
             (:action t :effect (when (p) (q)))
             (:action unp :effect (not (p))))""",
        "(define (problem x) (:domain c) (:init (p)) (:goal (q)))",
    )
    for name in ("lmcut", "h2"):
        assert make_heuristic(name, plain).admissible
        assert not make_heuristic(name, conditional).admissible
    assert make_heuristic("hmax", conditional).admissible


def test_an_optimal_planner_defaults_to_an_admissible_heuristic():
    # With no heuristic named, A* used to get hff, which is not admissible.
    task = task_of(DETOUR, DETOUR_P)
    assert solve_task(task, "astar").cost == 2
    assert solve_task(task, "idastar").cost == 2


# -- reproducibility ------------------------------------------------------------


def test_grounding_and_search_do_not_depend_on_the_hash_seed():
    # Fact ids used to follow set iteration order over atoms, i.e. string
    # hashing, so LM-cut broke ties differently under each PYTHONHASHSEED and
    # a benchmark could not be reproduced.
    import os
    import subprocess
    import sys

    from conftest import REPO_ROOT

    script = (
        "from jupyddl import build_task, solve_task\n"
        "t = build_task('demos/gripper/domain.pddl', 'demos/gripper/problem.pddl')\n"
        "r = solve_task(t, 'astar', 'lmcut')\n"
        "print(t.facts[:5], r.stats.expanded, r.plan_names())\n"
    )
    outputs = set()
    for seed in ("0", "1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=REPO_ROOT)
        run = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(run.stdout)
    assert len(outputs) == 1, outputs


def test_none_means_blind_for_a_planner_that_needs_a_heuristic():
    # `-H none` used to fall through to the hff default, i.e. the opposite.
    task = task_of(DETOUR, DETOUR_P)
    none = solve_task(task, "astar", "none")
    blind = solve_task(task, "astar", "blind")
    assert (none.stats.expanded, none.stats.evaluated) == (
        blind.stats.expanded,
        blind.stats.evaluated,
    )
    assert solve_task(task, "bfs", "none").solved
