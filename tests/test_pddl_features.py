"""The PDDL requirements beyond plain STRIPS.

Each block here is a self-contained domain exercising one requirement flag, so a
failure names the feature that broke. What every flag is *supposed* to do lives
in :mod:`jupyddl.requirements`; these tests check the implementation matches.
"""

from __future__ import annotations

import pytest

from jupyddl import solve_task, validate_plan
from jupyddl.grounding import ground
from jupyddl.parser import UnsupportedFeatureError, parse
from jupyddl.requirements import REQUIREMENTS, expand, lookup, supported


def build(domain_text: str, problem_text: str):
    return ground(parse(domain_text), parse(problem_text))


def solve(domain_text, problem_text, search="astar", heuristic="lmcut", **kwargs):
    """Ground, solve and validate in one step; returns ``(task, result)``."""
    task = build(domain_text, problem_text)
    result = solve_task(task, search, heuristic, **kwargs)
    if result.solved:
        assert validate_plan(task, result.plan), "planner returned an invalid plan"
    return task, result


def plan_of(task, result):
    return [op.base_name for op in task.visible_plan(result.plan)]


# ==========================================================================
# the registry itself
# ==========================================================================
def test_every_requirement_has_a_summary():
    for requirement in REQUIREMENTS.values():
        assert requirement.name.startswith(":")
        assert requirement.summary
        assert requirement.support in {"native", "compiled", "partial", "rejected"}


def test_adl_expands_to_its_components():
    flags = expand([":adl"])
    assert ":disjunctive-preconditions" in flags
    assert ":conditional-effects" in flags
    assert ":typing" in flags


def test_lookup_tolerates_a_missing_colon():
    assert lookup("strips") is lookup(":strips")
    assert lookup("nonsense-flag") is None


def test_rejected_flags_are_not_supported():
    # `:continuous-effects` is the one flag that stays refused: it needs
    # continuous-time reasoning, which no compilation into a discrete state
    # space can honestly provide.
    assert not supported(":continuous-effects")
    assert supported(":derived-predicates")
    assert supported(":preferences")


def test_unknown_requirement_is_rejected_at_parse_time():
    with pytest.raises(UnsupportedFeatureError, match="unknown requirement"):
        parse("(define (domain d) (:requirements :teleportation) (:predicates (p)))")


@pytest.mark.parametrize("flag", [":continuous-effects"])
def test_unsupported_flags_fail_loudly(flag):
    """Silently ignoring a requirement would produce confidently wrong plans."""
    with pytest.raises(UnsupportedFeatureError):
        parse(f"(define (domain d) (:requirements :strips {flag}) (:predicates (p)))")


@pytest.mark.parametrize(
    "flag",
    [
        ":preferences",
        ":constraints",
        ":timed-initial-literals",
        ":duration-inequalities",
        ":object-fluents",
    ],
)
def test_formerly_rejected_flags_are_now_accepted(flag):
    parse(f"(define (domain d) (:requirements :strips {flag}) (:predicates (p)))")


# ==========================================================================
# :disjunctive-preconditions
# ==========================================================================
DISJUNCTIVE_DOMAIN = """
(define (domain choice)
  (:requirements :strips :disjunctive-preconditions)
  (:predicates (p) (q) (r) (done))
  (:action mkp :precondition () :effect (p))
  (:action mkq :precondition () :effect (q))
  (:action finish :precondition (or (p) (q) (r)) :effect (done)))
"""


def test_disjunctive_precondition_takes_the_available_branch():
    task, result = solve(
        DISJUNCTIVE_DOMAIN,
        "(define (problem c) (:domain choice) (:init (r)) (:goal (done)))",
    )
    assert result.solved and result.cost == 1
    assert plan_of(task, result) == ["finish"]


def test_disjunctive_precondition_splits_into_one_operator_per_disjunct():
    task = build(
        DISJUNCTIVE_DOMAIN,
        "(define (problem c) (:domain choice) (:init (p) (q) (r)) (:goal (done)))",
    )
    finish_ops = [op for op in task.operators if op.base_name == "finish"]
    assert len(finish_ops) == 3
    # ...but they all report the same action to the outside world.
    assert {op.base_name for op in finish_ops} == {"finish"}


def test_goal_needing_an_unachievable_fact_is_unsolvable():
    """`(r)` is never added by any action and is false initially."""
    _, result = solve(
        DISJUNCTIVE_DOMAIN,
        "(define (problem c) (:domain choice) (:init ) (:goal (and (done) (r))))",
    )
    assert not result.solved
    assert not result.truncated, "this is proved unsolvable, not merely abandoned"


def test_disjunctive_goal_is_compiled_and_hidden_from_the_plan():
    """Both branches are achievable here, so the goal really is a disjunction."""
    task, result = solve(
        DISJUNCTIVE_DOMAIN,
        "(define (problem c) (:domain choice) (:init ) (:goal (or (p) (q))))",
    )
    assert result.solved
    assert task.synthetic, "a disjunctive goal should introduce a compiled operator"
    # The synthetic goal-reaching operator is an artefact, not an action: the
    # visible plan is one real step, even though the search took two.
    visible = plan_of(task, result)
    assert visible in (["mkp"], ["mkq"])
    assert len(result.plan) == len(visible) + 1


def test_statically_impossible_goal_branch_is_simplified_away():
    """`(r)` is in no effect and false initially, so that disjunct cannot hold."""
    task, result = solve(
        DISJUNCTIVE_DOMAIN,
        "(define (problem c) (:domain choice) (:init ) (:goal (or (q) (r))))",
    )
    assert result.solved
    # Only one branch survived, so no compilation was needed at all.
    assert not task.synthetic
    assert plan_of(task, result) == ["mkq"]


# ==========================================================================
# :existential-preconditions / :universal-preconditions
# ==========================================================================
QUANTIFIED_DOMAIN = """
(define (domain quantified)
  (:requirements :strips :typing :quantified-preconditions)
  (:types item)
  (:predicates (shiny ?i - item) (clean ?i - item) (polish ?i - item)
               (any-shiny) (all-clean))
  (:action polish-it :parameters (?i - item) :precondition (polish ?i)
    :effect (and (shiny ?i) (clean ?i)))
  (:action note-any :precondition (exists (?i - item) (shiny ?i))
    :effect (any-shiny))
  (:action note-all :precondition (forall (?i - item) (clean ?i))
    :effect (all-clean)))
"""


def test_existential_precondition_needs_one_witness():
    _, result = solve(
        QUANTIFIED_DOMAIN,
        "(define (problem q) (:domain quantified) (:objects a b - item)"
        " (:init (shiny b)) (:goal (any-shiny)))",
    )
    assert result.solved and result.cost == 1


def test_existential_precondition_fails_with_no_witness():
    _, result = solve(
        QUANTIFIED_DOMAIN,
        "(define (problem q) (:domain quantified) (:objects a b - item)"
        " (:init ) (:goal (any-shiny)))",
    )
    assert not result.solved


def test_universal_precondition_needs_every_object():
    _, result = solve(
        QUANTIFIED_DOMAIN,
        "(define (problem q) (:domain quantified) (:objects a b - item)"
        " (:init (clean a) (clean b)) (:goal (all-clean)))",
    )
    assert result.solved and result.cost == 1


def test_universal_precondition_must_be_achieved_for_all():
    """One object short: the planner has to polish the other one first."""
    task, result = solve(
        QUANTIFIED_DOMAIN,
        "(define (problem q) (:domain quantified) (:objects a b - item)"
        " (:init (clean a) (polish b)) (:goal (all-clean)))",
    )
    assert result.solved
    assert plan_of(task, result) == ["polish-it(b)", "note-all"]


# ==========================================================================
# imply / nested negation (NNF)
# ==========================================================================
def test_imply_is_rewritten_as_a_disjunction():
    _, result = solve(
        "(define (domain weather) (:requirements :adl)"
        " (:predicates (rain) (umbrella) (ok))"
        " (:action go :precondition (imply (rain) (umbrella)) :effect (ok)))",
        "(define (problem w) (:domain weather) (:init (rain) (umbrella))"
        " (:goal (ok)))",
    )
    assert result.solved


def test_imply_blocks_when_the_antecedent_holds_alone():
    _, result = solve(
        "(define (domain weather) (:requirements :adl)"
        " (:predicates (rain) (umbrella) (ok))"
        " (:action go :precondition (imply (rain) (umbrella)) :effect (ok)))",
        "(define (problem w) (:domain weather) (:init (rain)) (:goal (ok)))",
    )
    assert not result.solved


def test_negated_disjunction_becomes_a_conjunction():
    """`(not (or a b))` must require both to be false, not either."""
    domain = (
        "(define (domain nnf) (:requirements :adl)"
        " (:predicates (a) (b) (ok))"
        " (:action go :precondition (not (or (a) (b))) :effect (ok)))"
    )
    _, blocked = solve(
        domain, "(define (problem n) (:domain nnf) (:init (a)) (:goal (ok)))"
    )
    assert not blocked.solved
    _, clear = solve(domain, "(define (problem n) (:domain nnf) (:init ) (:goal (ok)))")
    assert clear.solved


# ==========================================================================
# :derived-predicates
# ==========================================================================
DERIVED_DOMAIN = """
(define (domain reach)
  (:requirements :strips :typing :derived-predicates :adl)
  (:types node)
  (:predicates (edge ?a ?b - node) (path ?a ?b - node) (done))
  (:derived (path ?a ?b - node)
    (or (edge ?a ?b)
        (exists (?m - node) (and (edge ?a ?m) (path ?m ?b)))))
  (:action finish :precondition (path n1 n4) :effect (done)))
"""

CHAIN_PROBLEM = """
(define (problem r) (:domain reach) (:objects n1 n2 n3 n4 - node)
  (:init (edge n1 n2) (edge n2 n3) (edge n3 n4)) (:goal (done)))
"""

BROKEN_PROBLEM = """
(define (problem r) (:domain reach) (:objects n1 n2 n3 n4 - node)
  (:init (edge n1 n2) (edge n3 n4)) (:goal (done)))
"""


def test_derived_predicate_computes_a_transitive_closure():
    task, result = solve(DERIVED_DOMAIN, CHAIN_PROBLEM, heuristic="hmax")
    assert result.solved
    assert task.axioms, "the rule should have grounded into axioms"


def test_derived_predicate_is_false_when_the_chain_breaks():
    _, result = solve(DERIVED_DOMAIN, BROKEN_PROBLEM, heuristic="hmax")
    assert not result.solved


def test_derived_facts_are_closed_in_the_initial_state():
    task = build(DERIVED_DOMAIN, CHAIN_PROBLEM)
    closed = task.initial_state()
    derived = {i for i, name in enumerate(task.facts) if name.startswith("(path ")}
    assert derived & set(closed), "reachability should hold in the initial state"


def test_derived_predicates_are_not_treated_as_static():
    """A derived predicate appears in no effect, but it is not a constant."""
    task = build(DERIVED_DOMAIN, CHAIN_PROBLEM)
    assert any(name.startswith("(path ") for name in task.facts)


# ==========================================================================
# :numeric-fluents
# ==========================================================================
NUMERIC_DOMAIN = """
(define (domain fuel)
  (:requirements :strips :typing :numeric-fluents)
  (:types truck loc)
  (:predicates (at ?t - truck ?l - loc) (road ?a ?b - loc))
  (:functions (fuel ?t - truck))
  (:action drive :parameters (?t - truck ?from ?to - loc)
    :precondition (and (at ?t ?from) (road ?from ?to) (>= (fuel ?t) 10))
    :effect (and (not (at ?t ?from)) (at ?t ?to) (decrease (fuel ?t) 10)))
  (:action refuel :parameters (?t - truck)
    :precondition (< (fuel ?t) 10)
    :effect (assign (fuel ?t) 30)))
"""


def test_numeric_fluents_are_part_of_the_state():
    task = build(
        NUMERIC_DOMAIN,
        "(define (problem f) (:domain fuel) (:objects tr - truck a b - loc)"
        " (:init (at tr a) (road a b) (= (fuel tr) 25)) (:goal (at tr b)))",
    )
    assert task.numeric
    assert task.numeric_names == ("(fuel tr)",)
    assert task.init_values == (25.0,)


def test_numeric_precondition_blocks_when_the_value_is_too_low():
    _, result = solve(
        NUMERIC_DOMAIN,
        "(define (problem f) (:domain fuel) (:objects tr - truck a b - loc)"
        " (:init (at tr a) (road a b) (= (fuel tr) 5)) (:goal (at tr b)))",
        heuristic="hmax",
    )
    # Fuel is 5, driving needs 10, and refuelling is possible: so it *is*
    # solvable, just not directly.
    assert result.solved


def test_numeric_goal_forces_an_extra_action():
    """The goal needs more fuel than driving leaves behind, so refuel or bust."""
    task, result = solve(
        NUMERIC_DOMAIN,
        "(define (problem f) (:domain fuel) (:objects tr - truck a b c - loc)"
        " (:init (at tr a) (road a b) (road b c) (= (fuel tr) 25))"
        " (:goal (and (at tr c) (>= (fuel tr) 20))))",
        heuristic="hmax",
    )
    assert result.solved
    assert "refuel(tr)" in plan_of(task, result)


def test_numeric_effects_read_the_pre_state():
    task = build(
        NUMERIC_DOMAIN,
        "(define (problem f) (:domain fuel) (:objects tr - truck a b - loc)"
        " (:init (at tr a) (road a b) (= (fuel tr) 25)) (:goal (at tr b)))",
    )
    drive = next(op for op in task.operators if op.base_name == "drive(tr,a,b)")
    after = drive.apply(task.initial_state())
    assert after.values == (15.0,)


def test_numeric_state_is_hashable_and_distinguishes_values():
    """Two states with the same facts but different numbers are different."""
    from jupyddl.task import State

    facts = frozenset({1, 2})
    assert State(facts, (1.0,)) != State(facts, (2.0,))
    assert len({State(facts, (1.0,)), State(facts, (1.0,))}) == 1


# ==========================================================================
# :durative-actions
# ==========================================================================
TEMPORAL_DOMAIN = """
(define (domain build)
  (:requirements :strips :typing :durative-actions)
  (:types site)
  (:predicates (dug ?s - site) (built ?s - site))
  (:durative-action dig :parameters (?s - site) :duration (= ?duration 3)
    :condition (and (at start (not (dug ?s))))
    :effect (and (at end (dug ?s))))
  (:durative-action build :parameters (?s - site) :duration (= ?duration 5)
    :condition (and (over all (dug ?s)))
    :effect (and (at end (built ?s)))))
"""


def test_durative_actions_carry_a_duration_and_a_makespan():
    task, result = solve(
        TEMPORAL_DOMAIN,
        "(define (problem b) (:domain build) (:objects s1 - site) (:init )"
        " (:goal (built s1)))",
        heuristic="hmax",
    )
    assert task.temporal
    assert result.solved
    assert task.makespan(result.plan) == 8.0  # 3 to dig plus 5 to build
    assert plan_of(task, result) == ["dig(s1)", "build(s1)"]


def test_makespan_is_sequential_across_independent_actions():
    """The compilation is sequential, so two independent jobs do not overlap."""
    task, result = solve(
        TEMPORAL_DOMAIN,
        "(define (problem b) (:domain build) (:objects s1 s2 - site) (:init )"
        " (:goal (and (built s1) (built s2))))",
        heuristic="hmax",
    )
    assert result.solved
    assert task.makespan(result.plan) == 16.0


def test_bounded_duration_takes_the_shortest_feasible_value():
    domain = parse(
        "(define (domain d) (:requirements :durative-actions :duration-inequalities)"
        " (:predicates (p))"
        " (:durative-action a :parameters ()"
        "  :duration (and (>= ?duration 2) (<= ?duration 9))"
        "  :condition (and) :effect (and (at end (p)))))"
    )
    assert domain.actions[0].duration.value == 2.0


def test_strict_duration_inequality_is_rejected():
    """`> 2` has no shortest feasible duration, so there is nothing to pick."""
    with pytest.raises(UnsupportedFeatureError, match="strict duration"):
        parse(
            "(define (domain d) (:requirements :durative-actions"
            " :duration-inequalities) (:predicates (p))"
            " (:durative-action a :parameters () :duration (> ?duration 2)"
            "  :condition (and) :effect (and (at end (p)))))"
        )


# ==========================================================================
# budgets
# ==========================================================================
def test_budget_marks_truncation_rather_than_unsolvability(examples_available):
    from conftest import demo_paths
    from jupyddl import build_task

    task = build_task(*demo_paths("blocksworld12"))
    result = solve_task(task, "astar", "lmcut", max_expansions=50)
    assert not result.solved
    assert result.truncated, "hitting a budget is not the same as being unsolvable"
    assert result.stats.expanded <= 50


def test_no_budget_means_no_truncation(examples_available):
    from conftest import paths
    from jupyddl import build_task

    task = build_task(*paths("tsp"))
    result = solve_task(task, "astar", "lmcut")
    assert result.solved and not result.truncated
