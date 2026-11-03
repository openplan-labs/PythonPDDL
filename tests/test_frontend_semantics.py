"""Regression tests for the PDDL front end: each one used to be read wrongly.

Text goes through the parser and :mod:`jupyddl.compile` before the grounder sees
it, and a mistake there gives a confident wrong answer — a plan `validate_plan`
accepts, because it replays the *compiled* task, or an "unsolvable" for an
instance with an obvious plan. These pin the semantics on tiny hand-written
instances whose right answer is clear by inspection, or check that a construct
the compilation cannot honour is refused by name rather than mis-planned.
"""

from __future__ import annotations

import pytest

from jupyddl import ground, parse, solve_task, validate_plan
from jupyddl.parser import PDDLError, UnsupportedFeatureError, tokenize
from jupyddl.requirements import expand


def task_of(domain: str, problem: str):
    return ground(parse(domain), parse(problem))


def solve(domain: str, problem: str, search="astar", heuristic="blind"):
    task = task_of(domain, problem)
    result = solve_task(task, search, heuristic, time_limit=30)
    assert not result.truncated, "these instances are tiny; a budget must not stop them"
    if result.solved:
        assert validate_plan(task, result.plan)
    return task, result


def visible(task, result):
    return [op.base_name for op in task.visible_plan(result.plan)]


def final_facts(task, plan):
    state = task.initial_state()
    for op in plan:
        state = task.apply(op, state)
    return {task.facts[f] for f in state}


# -- durative actions -----------------------------------------------------------

LOCK = """
(define (domain lock)
  (:requirements :strips :negative-preconditions :durative-actions)
  (:predicates (busy) (done))
  (:durative-action work :parameters () :duration (= ?duration 3)
    :condition (at start (not (done)))
    :effect (and (at start (busy)) (at end (not (busy))) (at end (done)))))
"""


def test_an_at_end_effect_overrides_an_at_start_one():
    """`busy` is set when the work starts and cleared when it ends."""
    task, result = solve(
        LOCK, "(define (problem p) (:domain lock) (:goal (and (done) (not (busy)))))"
    )
    assert visible(task, result) == ["work"]
    assert "(busy)" not in final_facts(task, result.plan)


def test_an_at_end_condition_reads_the_state_after_the_start_effects():
    domain = """
    (define (domain d) (:requirements :strips :durative-actions)
      (:predicates (busy) (done))
      (:durative-action work :parameters () :duration (= ?duration 3)
        :condition (at end (busy))
        :effect (and (at start (busy)) (at end (done)))))"""
    task, result = solve(domain, "(define (problem p) (:domain d) (:goal (done)))")
    assert visible(task, result) == ["work"]


def test_an_over_all_condition_the_start_breaks_blocks_the_action():
    domain = """
    (define (domain d) (:requirements :strips :durative-actions)
      (:predicates (ready) (done))
      (:durative-action work :parameters () :duration (= ?duration 3)
        :condition (over all (ready))
        :effect (and (at start (not (ready))) (at end (done)))))"""
    _, result = solve(
        domain, "(define (problem p) (:domain d) (:init (ready)) (:goal (done)))"
    )
    assert not result.solved


AIRLOCK = """
(define (domain airlock)
  (:requirements :strips :durative-actions :constraints)
  (:predicates (sealed) (through))
  (:durative-action cycle :parameters () :duration (= ?duration 2)
    :condition (at start (sealed))
    :effect (and (at start (not (sealed))) (at end (sealed)) (at end (through)))))
"""


def test_always_is_checked_between_a_durative_actions_start_and_end():
    """`sealed` is false for the whole of `cycle`, which `always` forbids."""
    problem = (
        "(define (problem p) (:domain airlock) (:init (sealed)) {} (:goal (through)))"
    )
    _, free = solve(AIRLOCK, problem.format(""))
    assert free.solved
    _, constrained = solve(AIRLOCK, problem.format("(:constraints (always (sealed)))"))
    assert not constrained.solved


def test_a_monitor_that_cannot_see_inside_a_durative_action_is_refused():
    with pytest.raises(UnsupportedFeatureError, match="between an action's start"):
        task_of(
            AIRLOCK,
            "(define (problem p) (:domain airlock) (:init (sealed))"
            " (:constraints (at-most-once (sealed))) (:goal (through)))",
        )


def test_a_durative_effect_without_a_time_specifier_is_refused():
    with pytest.raises(PDDLError, match="time specifier"):
        parse(
            "(define (domain d) (:requirements :durative-actions) (:predicates (p))"
            " (:durative-action a :parameters () :duration (= ?duration 1)"
            "  :condition (and) :effect (and (p))))"
        )


# -- sometime-after --------------------------------------------------------------

ALARM = """
(define (domain alarm)
  (:requirements :adl :constraints)
  (:predicates (trig) (resp))
  (:action t :precondition (not (trig)) :effect (trig))
  (:action r :precondition (and (trig) (not (resp))) :effect (resp)))
"""


def test_sometime_after_is_discharged_in_the_final_state():
    """{} -> {trig} -> {trig, resp}: the trigger is answered, by the last state."""
    task, result = solve(
        ALARM,
        "(define (problem p) (:domain alarm)"
        " (:constraints (sometime-after (trig) (resp))) (:goal (resp)))",
    )
    assert visible(task, result) == ["t", "r"]


def test_sometime_after_still_rejects_an_unanswered_trigger():
    """Triggering cuts the power and clears any answer, so none can follow."""
    domain = """
    (define (domain blackout) (:requirements :adl :constraints)
      (:predicates (trig) (resp) (power))
      (:action t :precondition () :effect (and (trig) (not (power)) (not (resp))))
      (:action r :precondition (power) :effect (resp)))"""
    problem = (
        "(define (problem p) (:domain blackout) (:init (power)) {} (:goal (trig)))"
    )
    _, free = solve(domain, problem.format(""))
    assert free.solved
    _, result = solve(
        domain, problem.format("(:constraints (sometime-after (trig) (resp)))")
    )
    assert not result.solved


# -- forall around a trajectory constraint ---------------------------------------

SPOT = """
(define (domain spot)
  (:requirements :adl :constraints)
  (:predicates (lit ?x) (done))
  (:action light :parameters (?x) :precondition ()
    :effect (and (lit ?x) (forall (?y) (when (not (= ?y ?x)) (not (lit ?y))))))
  (:action finish :precondition () :effect (done)))
"""


def test_forall_sometime_gives_each_object_its_own_moment():
    """Only one spot is lit at a time, yet each must be lit at some point."""
    task, result = solve(
        SPOT,
        "(define (problem p) (:domain spot) (:objects a b)"
        " (:constraints (forall (?x) (sometime (lit ?x)))) (:goal (done)))",
    )
    assert sorted(visible(task, result)) == ["finish", "light(a)", "light(b)"]


RESET = """
(define (domain reset)
  (:requirements :adl :constraints)
  (:predicates (on ?x) (used ?x) (serviced ?x) (again ?x))
  (:action turn-on :parameters (?x) :precondition (not (on ?x)) :effect (on ?x))
  (:action turn-off :parameters (?x) :precondition (on ?x) :effect (not (on ?x)))
  (:action use :parameters (?x) :precondition (on ?x) :effect (used ?x))
  (:action service :parameters (?x) :precondition (and (not (on ?x)) (used ?x))
    :effect (serviced ?x))
  (:action reuse :parameters (?x) :precondition (and (on ?x) (serviced ?x))
    :effect (again ?x)))
"""


def test_forall_at_most_once_applies_to_each_object():
    """`again a` needs `a` on, off, then on: two intervals for `a` itself.

    Read as `(at-most-once (forall (?x) (on ?x)))` the constraint never binds,
    since `b` is never on.
    """
    problem = "(define (problem p) (:domain reset) (:objects a b) {} (:goal (again a)))"
    _, free = solve(RESET, problem.format(""))
    assert free.solved
    _, constrained = solve(
        RESET, problem.format("(:constraints (forall (?x) (at-most-once (on ?x))))")
    )
    assert not constrained.solved


def test_a_quantified_soft_constraint_is_refused_by_name():
    with pytest.raises(UnsupportedFeatureError, match="soft constraint 'p'"):
        task_of(
            SPOT,
            "(define (problem q) (:domain spot) (:objects a b)"
            " (:constraints (forall (?x) (preference p (sometime (lit ?x)))))"
            " (:goal (done)))",
        )


# -- preferences -----------------------------------------------------------------

SHOP = """
(define (domain shop)
  (:requirements :adl :preferences :action-costs)
  (:predicates (bought ?x) (home))
  (:functions (total-cost))
  (:action buy :parameters (?x) :precondition ()
    :effect (and (bought ?x) (increase (total-cost) 1)))
  (:action go-home :precondition () :effect (and (home) (increase (total-cost) 1))))
"""


def test_a_preference_under_forall_is_one_preference_per_object():
    """Each unbought item costs 10, buying it costs 1: buy both, then go home."""
    task, result = solve(
        SHOP,
        "(define (problem p) (:domain shop) (:objects a b)"
        " (:goal (and (home) (forall (?x) (preference got (bought ?x)))))"
        " (:metric minimize (+ (total-cost) (* 10 (is-violated got)))))",
    )
    assert result.cost == 3
    assert sorted(visible(task, result)) == ["buy(a)", "buy(b)", "go-home"]


@pytest.mark.parametrize("wrapper", ["not", "or"])
def test_a_preference_outside_and_or_forall_is_refused(wrapper):
    with pytest.raises(PDDLError, match="only appear under 'and' or 'forall'"):
        parse(
            "(define (problem p) (:domain shop) (:objects a)"
            f" (:goal (and (home) ({wrapper} (preference x (bought a))))))"
        )


def test_a_preference_the_metric_omits_costs_nothing_to_violate():
    """Only `milk` is priced, so buying `cake` is a wasted action."""
    task, result = solve(
        SHOP,
        "(define (problem p) (:domain shop) (:objects milk cake)"
        " (:goal (and (home) (preference milk (bought milk))"
        "  (preference cake (bought cake))))"
        " (:metric minimize (+ (total-cost) (* 5 (is-violated milk)))))",
    )
    assert result.cost == 2
    assert sorted(visible(task, result)) == ["buy(milk)", "go-home"]


def test_a_metric_without_total_cost_makes_actions_free():
    """Buying costs nothing the metric counts, so violating at 0.5 is worse."""
    task, result = solve(
        SHOP,
        "(define (problem p) (:domain shop) (:objects milk)"
        " (:goal (and (home) (preference milk (bought milk))))"
        " (:metric minimize (* 0.5 (is-violated milk))))",
    )
    assert result.cost == 0
    assert "buy(milk)" in visible(task, result)


@pytest.mark.parametrize(
    "metric, error",
    [
        ("(:metric maximize (* 5 (is-violated milk)))", "maximize"),
        ("(:metric minimize (fuel-used))", "the metric reads"),
        ("(:metric minimize (* 2 (total-cost)))", "scales"),
        ("(:metric minimize (* (total-cost) (is-violated milk)))", "not a sum"),
    ],
)
def test_a_metric_the_compilation_cannot_price_is_refused(metric, error):
    with pytest.raises(UnsupportedFeatureError, match=error):
        parse(
            "(define (problem p) (:domain shop) (:objects milk)"
            f" (:goal (and (home) (preference milk (bought milk)))) {metric})"
        )


def test_a_metric_naming_an_undeclared_preference_is_an_error():
    with pytest.raises(PDDLError, match="no preference named 'milkk'"):
        task_of(
            SHOP,
            "(define (problem p) (:domain shop) (:objects milk)"
            " (:goal (and (home) (preference milk (bought milk))))"
            " (:metric minimize (+ (total-cost) (* 5 (is-violated milkk)))))",
        )


SPRINT = """
(define (domain sprint)
  (:requirements :strips :durative-actions)
  (:predicates (s1) (s2) (s3) (s4) (s5) (goal))
  (:durative-action slow :parameters () :duration (= ?duration 5)
    :condition (and) :effect (at end (goal)))
  (:action a1 :effect (s1))
  (:action a2 :precondition (s1) :effect (s2))
  (:action a3 :precondition (s2) :effect (s3))
  (:action a4 :precondition (s3) :effect (s4))
  (:action a5 :precondition (s4) :effect (s5))
  (:action a6 :precondition (s5) :effect (goal)))
"""


def test_a_total_time_metric_minimises_the_makespan():
    """Six instantaneous steps take no time; the one durative action takes 5."""
    task, result = solve(
        SPRINT,
        "(define (problem p) (:domain sprint) (:goal (goal))"
        " (:metric minimize (total-time)))",
    )
    assert "slow" not in visible(task, result)
    assert task.makespan(result.plan) == 0


def test_a_total_time_metric_with_timed_literals_is_refused():
    with pytest.raises(UnsupportedFeatureError, match="total-time"):
        task_of(
            SPRINT,
            "(define (problem p) (:domain sprint) (:init (at 3 (s1))) (:goal (goal))"
            " (:metric minimize (total-time)))",
        )


# -- timed initial literals and the PDDL 3 monitors ------------------------------

LAMP = """
(define (domain lamp)
  (:requirements :adl :constraints :timed-initial-literals)
  (:predicates (on) (ready) (done))
  (:action finish :precondition (ready) :effect (done)))
"""


def test_at_most_once_sees_intervals_the_clock_makes():
    """`on` holds, goes off at 1 and comes back at 2: two intervals."""
    problem = (
        "(define (problem p) (:domain lamp)"
        " (:init (on) (at 1 (not (on))) (at 2 (on)) (at 3 (ready))) {} (:goal (done)))"
    )
    _, free = solve(LAMP, problem.format(""))
    assert free.solved
    _, constrained = solve(LAMP, problem.format("(:constraints (at-most-once (on)))"))
    assert not constrained.solved


def test_sometime_after_sees_a_trigger_the_clock_makes():
    """The bell rings from 1 to 2; it must be answered while it rings."""
    domain = """
    (define (domain bell)
      (:requirements :adl :constraints :timed-initial-literals)
      (:predicates (ring) (answered) (ready) (done))
      (:action answer :precondition (ring) :effect (answered))
      (:action finish :precondition (ready) :effect (done)))"""
    task, result = solve(
        domain,
        "(define (problem p) (:domain bell)"
        " (:init (at 1 (ring)) (at 2 (not (ring))) (at 3 (ready)))"
        " (:constraints (sometime-after (ring) (answered))) (:goal (done)))",
    )
    assert visible(task, result) == ["answer", "finish"]


def test_a_timed_literal_cannot_fire_after_the_preferences_are_paid():
    """The report arrives as the lights go off, so keeping them lit is impossible."""
    domain = """
    (define (domain shift)
      (:requirements :adl :preferences :timed-initial-literals)
      (:predicates (lights) (report))
      (:action idle :precondition () :effect ()))"""
    task, result = solve(
        domain,
        "(define (problem p) (:domain shift)"
        " (:init (lights) (at 5 (not (lights))) (at 5 (report)))"
        " (:goal (and (report) (preference keep-lit (lights))))"
        " (:metric minimize (* 7 (is-violated keep-lit))))",
        search="dijkstra",
        heuristic=None,
    )
    assert result.cost == 7
    assert "(lights)" not in final_facts(task, result.plan)


def test_a_plan_cannot_end_before_a_due_timed_literal_fires():
    """The lights go off at 2, during the only action, and nothing relights them."""
    domain = """
    (define (domain night)
      (:requirements :strips :negative-preconditions :durative-actions
                     :timed-initial-literals)
      (:predicates (lights) (done))
      (:durative-action work :parameters () :duration (= ?duration 5)
        :condition (at start (not (done))) :effect (at end (done))))"""
    problem = (
        "(define (problem p) (:domain night) (:init (lights) {})"
        " (:goal (and (done) (lights))))"
    )
    _, early = solve(domain, problem.format("(at 9 (not (lights)))"))
    assert early.solved
    _, result = solve(domain, problem.format("(at 2 (not (lights)))"))
    assert not result.solved


# -- object fluents --------------------------------------------------------------

MOVE = """
(define (domain move)
  (:requirements :typing :object-fluents :constraints :adl)
  (:types place pkg)
  (:predicates (done))
  (:functions (loc ?p - pkg) - place)
  (:action go :parameters (?p - pkg ?to - place) :precondition ()
    :effect (assign (loc ?p) ?to))
  (:action finish :precondition () :effect (done)))
"""


def test_trajectory_constraints_read_object_fluents():
    problem = (
        "(define (problem q) (:domain move) (:objects a b - place p1 - pkg)"
        " (:init (= (loc p1) a)) (:constraints {}) (:goal {}))"
    )
    task, result = solve(MOVE, problem.format("(always (= (loc p1) a))", "(done)"))
    assert visible(task, result) == ["finish"]
    task, result = solve(
        MOVE, problem.format("(sometime (= (loc p1) b))", "(= (loc p1) a)")
    )
    assert visible(task, result) == ["go(p1,b)", "go(p1,a)"]


def test_assigning_one_object_fluent_to_another_is_refused():
    domain = MOVE.replace(
        "(:action finish",
        "(:action follow :parameters (?p ?q - pkg) :precondition ()"
        " :effect (assign (loc ?p) (loc ?q)))\n  (:action finish",
    )
    with pytest.raises(UnsupportedFeatureError, match="another function"):
        task_of(
            domain,
            "(define (problem q) (:domain move) (:objects a - place p1 p2 - pkg)"
            " (:init (= (loc p1) a) (= (loc p2) a)) (:goal (done)))",
        )


# -- grounding does not consume its input -------------------------------------------


def test_grounding_leaves_the_parsed_domain_as_it_was():
    domain = parse(
        "(define (domain d) (:requirements :adl :constraints :timed-initial-literals)"
        " (:predicates (a) (b) (gift))"
        " (:constraints (sometime (b)))"
        " (:action do-a :effect (a)) (:action do-b :effect (b)))"
    )
    first = ground(
        domain,
        parse("(define (problem p) (:domain d) (:init (at 5 (gift))) (:goal (a)))"),
    )
    assert solve_task(first, "bfs").solved
    assert [action.name for action in domain.actions] == ["do-a", "do-b"]

    # The same domain object, a second problem: its constraint still binds.
    second = ground(domain, parse("(define (problem p) (:domain d) (:goal (a)))"))
    result = solve_task(second, "bfs")
    assert sorted(op.base_name for op in second.visible_plan(result.plan)) == [
        "do-a",
        "do-b",
    ]


# -- invalid and malformed input ---------------------------------------------------

HEAD = "(define (domain d) (:predicates (p ?x) (q))"


@pytest.mark.parametrize(
    "text, key",
    [
        (HEAD + " (:action a :preconditon (q) :effect (q)))", ":preconditon"),
        (HEAD + " (:actions a :effect (q)))", ":actions"),
        ("(define (problem x) (:domain d) (:init) (:goals (q)))", ":goals"),
        ("(define (problem x) (:domain d) (:init))", "no :goal"),
    ],
)
def test_a_misspelt_or_missing_section_is_named(text, key):
    with pytest.raises(PDDLError, match=key):
        parse(text)


@pytest.mark.parametrize(
    "text",
    [
        HEAD + " (:action a :parameters (?x -) :effect (q)))",
        HEAD + " (:action a :parameters))",
        HEAD + " (:action a :parameters (?x) :precondition (= ?x) :effect (q)))",
        HEAD + " (:action a :precondition (forall (?y)) :effect (q)))",
        HEAD + " (:action a :effect (when (q))))",
        HEAD + " (:action a :effect (not)))",
        HEAD + " (:action a :effect (increase)))",
        HEAD + " (:constraints))",
        HEAD + " (:functions (f) -))",
        "(define (domain d) (:predicates p))",
        "(define (domain) (:predicates (p)))",
        "(define (problem x) (:domain d) (:init (= (f))) (:goal (q)))",
        "(define (problem x) (:domain d) (:init q) (:goal (q)))",
        "(define (problem x) (:domain d) (:init) (:goal))",
    ],
)
def test_malformed_input_is_a_pddl_error(text):
    with pytest.raises(PDDLError):
        parse(text)


def test_an_unbalanced_parenthesis_reports_its_line():
    with pytest.raises(PDDLError, match="line 3"):
        tokenize("(define (domain d)\n  (:predicates (p))\n  (:action a))))")


@pytest.mark.parametrize("name", ["nan", "inf", "infinity"])
def test_an_object_named_like_a_special_float_is_an_object(name):
    domain = """
    (define (domain visit) (:requirements :strips)
      (:predicates (at ?who ?where) (met))
      (:action meet :parameters (?w) :precondition (at ?w kitchen) :effect (met)))"""
    task, result = solve(
        domain,
        f"(define (problem p) (:domain visit) (:objects {name} kitchen)"
        f" (:init (at {name} kitchen)) (:goal (met)))",
    )
    assert visible(task, result) == [f"meet({name})"]


# -- requirement flags -------------------------------------------------------------

ROADS = """
(define (domain roads) (:requirements :typing :action-costs)
  (:types loc)
  (:predicates (at ?l - loc) (road ?a ?b - loc))
  (:functions (road-length ?a ?b - loc) (total-cost))
  (:action drive :parameters (?a ?b - loc) :precondition (and (at ?a) (road ?a ?b))
    :effect (and (not (at ?a)) (at ?b) (increase (total-cost) (road-length ?a ?b)))))
"""


def test_an_action_cost_can_read_a_static_function():
    """The direct road is 10 long; the detour through b is 2 + 3."""
    task, result = solve(
        ROADS,
        "(define (problem p) (:domain roads) (:objects a b c - loc)"
        " (:init (at a) (road a b) (road b c) (road a c)"
        "  (= (road-length a b) 2) (= (road-length b c) 3) (= (road-length a c) 10))"
        " (:goal (at c)) (:metric minimize (total-cost)))",
    )
    assert result.cost == 5
    assert visible(task, result) == ["drive(a,b)", "drive(b,c)"]
    assert task.numeric_names == (), "a static cost function is not a state variable"


def test_an_action_cost_reading_a_changing_function_is_refused():
    domain = ROADS.replace(
        "(increase (total-cost) (road-length ?a ?b))",
        "(increase (total-cost) (road-length ?a ?b)) (increase (road-length ?a ?b) 1)",
    )
    with pytest.raises(UnsupportedFeatureError, match="static"):
        task_of(
            domain,
            "(define (problem p) (:domain roads) (:objects a b - loc)"
            " (:init (at a) (road a b) (= (road-length a b) 2)) (:goal (at b)))",
        )


def test_fluents_implies_object_fluents():
    assert ":object-fluents" in expand([":fluents"])


def test_ucpop_is_refused_by_name():
    with pytest.raises(UnsupportedFeatureError, match="domain-axioms"):
        parse("(define (domain d) (:requirements :ucpop) (:predicates (p)))")
