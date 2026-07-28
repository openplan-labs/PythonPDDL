"""PDDL 3 constructs, and the compilations that make them run.

Preferences, trajectory constraints, timed initial literals and object fluents
are all rewritten into the classical core by :mod:`jupyddl.compile`. These tests
pin the *observable* behaviour — which plans are legal, what they cost, when the
plan ends — rather than the shape of the compilation, so the rewrite can be
changed without rewriting the tests.
"""

from __future__ import annotations

import pytest

from jupyddl import solve_task, validate_plan
from jupyddl.grounding import ground
from jupyddl.parser import UnsupportedFeatureError, parse
from jupyddl.task import values_of


def build(domain_text: str, problem_text: str):
    return ground(parse(domain_text), parse(problem_text))


def solve(domain_text, problem_text, heuristic="hmax", **kwargs):
    task = build(domain_text, problem_text)
    kwargs.setdefault("time_limit", 30)
    result = solve_task(task, "astar", heuristic, **kwargs)
    if result.solved:
        assert validate_plan(task, result.plan), "planner returned an invalid plan"
    return task, result


def plan_of(task, result):
    return [op.base_name for op in task.visible_plan(result.plan)]


# ==========================================================================
# :constraints
# ==========================================================================
SAFETY_DOMAIN = """
(define (domain reactor)
  (:requirements :adl :constraints)
  (:predicates (safe) (armed) (done))
  (:action arm :precondition () :effect (and (armed) (not (safe))))
  (:action work :precondition () :effect (done)))
"""


def test_always_is_enforced_on_every_state():
    """Arming breaks the invariant, so the planner must route around it."""
    task, result = solve(
        SAFETY_DOMAIN,
        "(define (problem r) (:domain reactor) (:init (safe))"
        " (:constraints (always (safe))) (:goal (done)))",
    )
    assert result.solved
    assert plan_of(task, result) == ["work"]


def test_always_can_make_a_goal_unreachable():
    _, result = solve(
        SAFETY_DOMAIN,
        "(define (problem r) (:domain reactor) (:init (safe))"
        " (:constraints (always (safe))) (:goal (armed)))",
    )
    assert not result.solved
    assert not result.truncated


def test_without_the_constraint_the_same_goal_is_reachable():
    """Guards against the constraint machinery breaking the plain case."""
    _, result = solve(
        SAFETY_DOMAIN,
        "(define (problem r) (:domain reactor) (:init (safe)) (:goal (armed)))",
    )
    assert result.solved


def test_always_also_constrains_the_final_state():
    """A plan may not end by violating the invariant either."""
    _, result = solve(
        SAFETY_DOMAIN,
        "(define (problem r) (:domain reactor) (:init (safe))"
        " (:constraints (always (safe))) (:goal (and (done) (armed))))",
    )
    assert not result.solved


TOUR_DOMAIN = """
(define (domain tour)
  (:requirements :adl :constraints)
  (:predicates (visited) (home) (done))
  (:action leave :precondition (home) :effect (and (visited) (not (home))))
  (:action back :precondition (visited) :effect (home))
  (:action finish :precondition (home) :effect (done)))
"""

TOUR_GOAL = "(:goal (and (done) (home)))"


def test_sometime_forces_a_detour():
    task, result = solve(
        TOUR_DOMAIN,
        f"(define (problem t) (:domain tour) (:init (home))"
        f" (:constraints (sometime (visited))) {TOUR_GOAL})",
    )
    assert result.solved
    assert "leave" in plan_of(task, result)
    assert result.cost == 3


def test_without_sometime_the_detour_is_skipped():
    task, result = solve(
        TOUR_DOMAIN,
        f"(define (problem t) (:domain tour) (:init (home)) {TOUR_GOAL})",
    )
    assert result.solved
    assert plan_of(task, result) == ["finish"]
    assert result.cost == 1


def test_at_end_is_conjoined_to_the_goal():
    _, result = solve(
        TOUR_DOMAIN,
        "(define (problem t) (:domain tour) (:init (home))"
        " (:constraints (at-end (visited))) (:goal (done)))",
    )
    assert result.solved


SWITCH_DOMAIN = """
(define (domain switch)
  (:requirements :adl :constraints)
  (:predicates (on) (done1) (done2))
  (:action turn-on :precondition () :effect (on))
  (:action turn-off :precondition (on) :effect (not (on)))
  (:action use1 :precondition (on) :effect (done1))
  (:action use2 :precondition (on) :effect (done2)))
"""


def test_at_most_once_permits_a_single_interval():
    task, result = solve(
        SWITCH_DOMAIN,
        "(define (problem s) (:domain switch) (:init )"
        " (:constraints (at-most-once (on))) (:goal (and (done1) (done2))))",
    )
    assert result.solved
    plan = plan_of(task, result)
    assert plan.count("turn-on") == 1, "the switch may only come on once"


RESET_DOMAIN = """
(define (domain reset-switch)
  (:requirements :adl :constraints)
  (:predicates (on) (done1) (reset) (done2))
  (:action turn-on :precondition (not (on)) :effect (on))
  (:action turn-off :precondition (on) :effect (not (on)))
  (:action use1 :precondition (on) :effect (done1))
  (:action service :precondition (and (not (on)) (done1)) :effect (reset))
  (:action use2 :precondition (and (on) (reset)) :effect (done2)))
"""


def test_at_most_once_forbids_a_second_interval():
    """This goal genuinely needs two on-intervals, which the constraint bans.

    `use2` needs the switch on *and* serviced, and servicing needs it off — so
    any plan must switch on, off, then on again. Without the constraint that is
    fine; with it the instance has no solution at all.
    """
    goal = "(:goal (and (done1) (done2)))"
    _, unconstrained = solve(
        RESET_DOMAIN, f"(define (problem s) (:domain reset-switch) (:init ) {goal})"
    )
    assert unconstrained.solved, "the goal is reachable when nothing forbids it"

    _, constrained = solve(
        RESET_DOMAIN,
        "(define (problem s) (:domain reset-switch) (:init )"
        f" (:constraints (at-most-once (on))) {goal})",
    )
    assert not constrained.solved
    assert not constrained.truncated


SEQUENCE_DOMAIN = """
(define (domain sequence)
  (:requirements :adl :constraints)
  (:predicates (opened) (paid) (left))
  (:action open :precondition () :effect (opened))
  (:action pay :precondition () :effect (paid))
  (:action leave :precondition () :effect (left)))
"""


def test_sometime_before_orders_two_events():
    """Leaving is only allowed once paying has already happened."""
    task, result = solve(
        SEQUENCE_DOMAIN,
        "(define (problem q) (:domain sequence) (:init )"
        " (:constraints (sometime-before (left) (paid)))"
        " (:goal (and (left) (opened))))",
    )
    assert result.solved
    plan = plan_of(task, result)
    assert "pay" in plan
    assert plan.index("pay") < plan.index("leave")


def test_sometime_after_creates_an_obligation():
    """Opening obliges paying, at the latest by the end of the plan."""
    task, result = solve(
        SEQUENCE_DOMAIN,
        "(define (problem q) (:domain sequence) (:init )"
        " (:constraints (sometime-after (opened) (paid)))"
        " (:goal (opened)))",
    )
    assert result.solved
    assert "pay" in plan_of(task, result)


def test_sometime_after_is_free_when_never_triggered():
    _, result = solve(
        SEQUENCE_DOMAIN,
        "(define (problem q) (:domain sequence) (:init )"
        " (:constraints (sometime-after (opened) (paid)))"
        " (:goal (left)))",
    )
    assert result.solved
    assert result.cost == 1, "an untriggered obligation should cost nothing"


def test_metric_time_constraints_are_refused_by_name():
    for operator in ("within", "always-within", "hold-after", "hold-during"):
        with pytest.raises(UnsupportedFeatureError, match="metric time"):
            parse(
                "(define (problem q) (:domain d) (:init )"
                f" (:constraints ({operator} 5 (p))) (:goal (p)))"
            )


def test_unknown_constraint_operator_is_named_in_the_error():
    with pytest.raises(UnsupportedFeatureError, match="not a recognised"):
        parse(
            "(define (problem q) (:domain d) (:init )"
            " (:constraints (eventually-maybe (p))) (:goal (p)))"
        )


# ==========================================================================
# :preferences
# ==========================================================================
SHOP_DOMAIN = """
(define (domain shop)
  (:requirements :adl :preferences :action-costs)
  (:predicates (bought-milk) (bought-cake) (home))
  (:functions (total-cost))
  (:action buy-milk :precondition () :effect (and (bought-milk) (increase (total-cost) 1)))
  (:action buy-cake :precondition () :effect (and (bought-cake) (increase (total-cost) 1)))
  (:action go-home :precondition () :effect (and (home) (increase (total-cost) 1))))
"""


def _shop(goal_extra, weight):
    return (
        "(define (problem s) (:domain shop) (:init )"
        f" (:goal (and (home) (preference {goal_extra})))"
        f" (:metric minimize (+ (total-cost) (* {weight} (is-violated cake)))))"
    )


def test_a_cheap_preference_is_satisfied():
    task, result = solve(SHOP_DOMAIN, _shop("cake (bought-cake)", 10))
    assert result.solved
    assert "buy-cake" in plan_of(task, result)
    assert result.cost == 2, "one purchase plus going home; the preference is free"


def test_an_expensive_preference_is_bought_off():
    """Violating costs 0.5, satisfying costs a whole action: pay the penalty."""
    task, result = solve(SHOP_DOMAIN, _shop("cake (bought-cake)", 0.5))
    assert result.solved
    assert "buy-cake" not in plan_of(task, result)
    assert result.cost == pytest.approx(1.5)


def test_preferences_default_to_weight_one_without_a_metric():
    task, result = solve(
        SHOP_DOMAIN,
        "(define (problem s) (:domain shop) (:init )"
        " (:goal (and (home) (preference cake (bought-cake)))))",
    )
    assert result.solved
    # Satisfying costs 1 extra action; violating costs the default weight of 1.
    assert result.cost == 2


def test_hard_goals_still_bind_alongside_preferences():
    task, result = solve(SHOP_DOMAIN, _shop("cake (bought-cake)", 0.5))
    assert result.solved
    assert "go-home" in plan_of(task, result)


def test_preference_cannot_be_satisfied_then_broken():
    """The closing phase freezes the state, so a preference cannot be faked.

    `up` holds only in the middle of this plan: raising it is a prerequisite for
    finishing, and finishing lowers it again for good. A compilation that let
    the planner claim the preference *while* it held — rather than at the end —
    would report the reward without the state ever showing it.
    """
    domain = """
    (define (domain toggle)
      (:requirements :adl :preferences :action-costs)
      (:predicates (up) (done))
      (:functions (total-cost))
      (:action raise :precondition (not (done))
        :effect (and (up) (increase (total-cost) 1)))
      (:action finish :precondition (up)
        :effect (and (done) (not (up)) (increase (total-cost) 1))))
    """
    task, result = solve(
        domain,
        "(define (problem t) (:domain toggle) (:init )"
        " (:goal (and (done) (preference stay-up (up))))"
        " (:metric minimize (+ (total-cost) (* 100 (is-violated stay-up)))))",
    )
    assert result.solved
    # raise + finish = 2, and `up` is false at the end however the plan is
    # arranged, so the 100-point penalty is unavoidable.
    assert result.cost == pytest.approx(102)
    assert plan_of(task, result) == ["raise", "finish"]


def test_soft_trajectory_constraints_are_refused():
    with pytest.raises(UnsupportedFeatureError, match="soft constraint"):
        build(
            "(define (domain d) (:requirements :adl :constraints :preferences)"
            " (:predicates (p) (q)) (:action go :precondition () :effect (p)))",
            "(define (problem q) (:domain d) (:init )"
            " (:constraints (preference nice (always (q)))) (:goal (p)))",
        )


def test_negative_violation_weight_is_rejected():
    """A negative penalty would pay the planner to break the preference."""
    with pytest.raises(Exception, match="negative violation weight"):
        build(
            SHOP_DOMAIN,
            "(define (problem s) (:domain shop) (:init )"
            " (:goal (and (home) (preference cake (bought-cake))))"
            " (:metric minimize (+ (total-cost) (* -3 (is-violated cake)))))",
        )


# ==========================================================================
# :timed-initial-literals
# ==========================================================================
TIL_DOMAIN = """
(define (domain market)
  (:requirements :strips :typing :durative-actions :timed-initial-literals
                 :numeric-fluents :adl)
  (:types item)
  (:predicates (open) (bought ?i - item) (prepared ?i - item))
  (:durative-action prepare :parameters (?i - item) :duration (= ?duration 4)
    :condition (and (at start (not (prepared ?i))))
    :effect (and (at end (prepared ?i))))
  (:durative-action buy :parameters (?i - item) :duration (= ?duration 1)
    :condition (and (over all (open)) (over all (prepared ?i)))
    :effect (and (at end (bought ?i))))
)
"""


def _market(opens_at):
    return (
        "(define (problem m) (:domain market) (:objects milk - item)"
        f" (:init (at {opens_at} (open))) (:goal (bought milk)))"
    )


def test_a_timed_literal_can_force_the_plan_to_wait():
    task, result = solve(TIL_DOMAIN, _market(10))
    assert result.solved
    # Preparing takes 4, so the plan idles until the shop opens at 10 and buys
    # in the hour after: the end time is 11, not the 5 the durations sum to.
    assert task.makespan(result.plan) == 11.0
    assert plan_of(task, result) == ["prepare(milk)", "buy(milk)"]


def test_an_early_timed_literal_costs_no_waiting():
    task, result = solve(TIL_DOMAIN, _market(0))
    assert result.solved
    assert task.makespan(result.plan) == 5.0


def test_the_clock_is_a_real_state_variable():
    task = build(TIL_DOMAIN, _market(10))
    assert task.clock_index is not None
    assert "(__time)" in task.numeric_names
    assert values_of(task.initial_state())[task.clock_index] == 0.0


def test_timed_literals_can_retract_a_fact():
    """`(at 3 (not (open)))` shuts the shop, and the window really closes.

    Buying needs the milk prepared, which takes 4; by then the shop has been
    shut for an hour, so there is no plan. A compilation that let the planner
    ignore a due literal would happily "buy" at t=4.
    """
    _, result = solve(
        TIL_DOMAIN,
        "(define (problem m) (:domain market) (:objects milk - item)"
        " (:init (at 0 (open)) (at 3 (not (open)))) (:goal (bought milk)))",
    )
    assert not result.solved
    assert not result.truncated


def test_a_wider_window_lets_the_same_plan_through():
    """The mirror of the test above: shut at 6 instead of 3 and it works."""
    task, result = solve(
        TIL_DOMAIN,
        "(define (problem m) (:domain market) (:objects milk - item)"
        " (:init (at 0 (open)) (at 6 (not (open)))) (:goal (bought milk)))",
    )
    assert result.solved
    assert task.makespan(result.plan) == 5.0


# ==========================================================================
# :object-fluents
# ==========================================================================
DELIVERY_DOMAIN = """
(define (domain delivery)
  (:requirements :strips :typing :object-fluents :adl)
  (:types package place)
  (:predicates (connected ?a - place ?b - place))
  (:functions (location ?p - package) - place)
  (:action move :parameters (?p - package ?from - place ?to - place)
    :precondition (and (= (location ?p) ?from) (connected ?from ?to))
    :effect (assign (location ?p) ?to)))
"""


def test_object_fluent_reads_and_writes():
    task, result = solve(
        DELIVERY_DOMAIN,
        "(define (problem d) (:domain delivery) (:objects pkg - package a b c - place)"
        " (:init (= (location pkg) a) (connected a b) (connected b c))"
        " (:goal (= (location pkg) c)))",
    )
    assert result.solved
    assert plan_of(task, result) == ["move(pkg,a,b)", "move(pkg,b,c)"]


def test_object_fluent_goal_can_be_unreachable():
    _, result = solve(
        DELIVERY_DOMAIN,
        "(define (problem d) (:domain delivery) (:objects pkg - package a b c - place)"
        " (:init (= (location pkg) a) (connected b c))"
        " (:goal (= (location pkg) c)))",
    )
    assert not result.solved


def test_object_fluent_stays_single_valued():
    """Assigning a new value must retract the old one, or it is a relation."""
    task, result = solve(
        DELIVERY_DOMAIN,
        "(define (problem d) (:domain delivery) (:objects pkg - package a b - place)"
        " (:init (= (location pkg) a) (connected a b))"
        " (:goal (= (location pkg) b)))",
    )
    assert result.solved
    state = task.initial_state()
    for operator in result.plan:
        state = task.apply(operator, state)
    held = [task.facts[f] for f in state if "fn-location" in task.facts[f]]
    assert len(held) == 1, f"the package is in two places at once: {held}"


def test_object_fluent_as_a_nested_term_is_refused():
    with pytest.raises(UnsupportedFeatureError):
        build(
            "(define (domain n) (:requirements :object-fluents :typing)"
            " (:types p l) (:predicates (at ?x - l)) (:functions (loc ?x - p) - l)"
            " (:action go :parameters (?x - p) :precondition (at (loc ?x))"
            "  :effect (at (loc ?x))))",
            "(define (problem q) (:domain n) (:objects o - p r - l)"
            " (:init ) (:goal (at r)))",
        )


def test_arithmetic_on_an_object_fluent_is_refused():
    with pytest.raises(UnsupportedFeatureError, match="returns an object"):
        build(
            DELIVERY_DOMAIN.replace(
                "(assign (location ?p) ?to)", "(increase (location ?p) ?to)"
            ),
            "(define (problem d) (:domain delivery)"
            " (:objects pkg - package a b - place)"
            " (:init (= (location pkg) a)) (:goal (= (location pkg) b)))",
        )


def test_numeric_and_object_fluents_coexist():
    """A `:functions` block may declare both, split by the trailing type."""
    domain = """
    (define (domain mixed)
      (:requirements :strips :typing :fluents :adl)
      (:types truck place)
      (:predicates (road ?a - place ?b - place))
      (:functions (fuel ?t - truck) - number (at-place ?t - truck) - place)
      (:action drive :parameters (?t - truck ?from - place ?to - place)
        :precondition (and (= (at-place ?t) ?from) (road ?from ?to) (>= (fuel ?t) 5))
        :effect (and (assign (at-place ?t) ?to) (decrease (fuel ?t) 5))))
    """
    task, result = solve(
        domain,
        "(define (problem m) (:domain mixed) (:objects tr - truck a b - place)"
        " (:init (= (at-place tr) a) (= (fuel tr) 12) (road a b))"
        " (:goal (and (= (at-place tr) b) (>= (fuel tr) 5))))",
    )
    assert task.numeric, "the numeric half should still be a state variable"
    assert result.solved
    assert plan_of(task, result) == ["drive(tr,a,b)"]


# ==========================================================================
# the compilations stay out of the way
# ==========================================================================
def test_synthetic_operators_are_hidden_from_plans():
    task, result = solve(SHOP_DOMAIN, _shop("cake (bought-cake)", 10))
    visible = task.visible_plan(result.plan)
    assert len(visible) < len(result.plan), "closing actions should be hidden"
    assert all(not op.base_name.startswith("__") for op in visible)


def test_a_plain_domain_gains_no_synthetic_machinery():
    """None of the PDDL 3 compilations should fire on a classical instance."""
    task = build(
        "(define (domain plain) (:requirements :strips) (:predicates (p))"
        " (:action go :precondition () :effect (p)))",
        "(define (problem q) (:domain plain) (:init ) (:goal (p)))",
    )
    assert not task.synthetic
    assert not task.numeric
    assert task.clock_index is None
    assert len(task.operators) == 1
