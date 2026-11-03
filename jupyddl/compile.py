"""Compile PDDL 3 constructs down to the classical core.

Preferences, trajectory constraints, timed initial literals and object fluents
are all rewritten here, at the AST level, *before* grounding. The grounder and
the search never learn they existed — which is the point: one representation to
optimise, and every new front-end feature is a source-to-source transformation
rather than another special case in the hot loop.

Each compilation is documented in :mod:`jupyddl.requirements`, including what it
costs you. The synthetic actions introduced along the way are all named with a
leading ``__``; :attr:`jupyddl.task.Task.synthetic` collects them so a printed
plan shows only the actions the domain author wrote.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from itertools import product

from .parser.ast import (
    Action,
    AddEffect,
    And,
    Arithmetic,
    Atom,
    Comparison,
    ConjunctiveEffect,
    Constraint,
    DelEffect,
    Domain,
    EqualityConstraint,
    Exists,
    Forall,
    ForallEffect,
    FluentRef,
    IncreaseCostEffect,
    Literal,
    Number,
    NumericEffect,
    Or,
    PDDLError,
    Predicate,
    Preference,
    Problem,
    TimedEffect,
    Truth,
    UnsupportedFeatureError,
    WhenEffect,
)

# Every fact and action this module invents is prefixed, so they cannot collide
# with a domain's own names and are easy to filter out of a plan.
PREFIX = "__"
CLOCK = FluentRef("__time", ())

__all__ = ["compile_problem", "SYNTHETIC_PREFIX"]

SYNTHETIC_PREFIX = PREFIX


# --------------------------------------------------------------------------
# small AST helpers
# --------------------------------------------------------------------------
def _fact(name: str, *args) -> Atom:
    return Atom(name, tuple(args))


def _holds(name: str, *args) -> Literal:
    return Literal(_fact(name, *args), True)


def _absent(name: str, *args) -> Literal:
    return Literal(_fact(name, *args), False)


def _conjoin(*parts):
    """And() over the parts, dropping trivially-true ones."""
    kept = [p for p in parts if p is not None and not isinstance(p, Truth)]
    if not kept:
        return Truth(True)
    if len(kept) == 1:
        return kept[0]
    return And(tuple(kept))


def _free():
    """An explicit zero cost.

    Grounding charges 1 for any action without a cost effect, which is right for
    a domain action and wrong for the bookkeeping this module invents: closing
    the plan or observing that a constraint held must not show up in the metric.
    """
    return IncreaseCostEffect(0.0)


def _also(effect, *extra):
    """Append effects to an existing effect tree."""
    parts = list(effect.parts) if isinstance(effect, ConjunctiveEffect) else [effect]
    parts.extend(extra)
    return ConjunctiveEffect(parts)


def _negate(formula):
    """Negation normal form of ``not formula``."""
    if isinstance(formula, Truth):
        return Truth(not formula.value)
    if isinstance(formula, Literal):
        return Literal(formula.atom, not formula.positive)
    if isinstance(formula, EqualityConstraint):
        return EqualityConstraint(formula.left, formula.right, not formula.positive)
    if isinstance(formula, Comparison):
        flip = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "=": "!=", "!=": "="}
        return Comparison(flip[formula.op], formula.left, formula.right)
    if isinstance(formula, And):
        return Or(tuple(_negate(p) for p in formula.parts))
    if isinstance(formula, Or):
        return And(tuple(_negate(p) for p in formula.parts))
    if isinstance(formula, Forall):
        return Exists(formula.params, _negate(formula.body))
    if isinstance(formula, Exists):
        return Forall(formula.params, _negate(formula.body))
    raise PDDLError(f"cannot negate {formula!r}")


def _is_domain_action(action: Action) -> bool:
    return not action.name.startswith(PREFIX)


def _changes_the_world(action: Action) -> bool:
    """Domain actions, and the firing of a timed literal.

    These are the only actions that change a fact the domain author wrote, so
    every state on a plan's trajectory except the last is the state one of
    them is applied in. The other synthetic actions only move the clock or
    set bookkeeping facts.
    """
    return _is_domain_action(action) or action.name.startswith(f"{PREFIX}fire-til-")


def _declare(domain: Domain, name: str, params=()) -> None:
    if all(p.name != name for p in domain.predicates):
        domain.predicates.append(Predicate(name, list(params)))


def _add_precondition(
    domain: Domain, formula, only_domain_actions=True, during=False
) -> None:
    """Conjoin ``formula`` onto every action's precondition.

    ``during`` also requires it of a durative action's ``late_condition``, i.e.
    in the state between its at-start and at-end effects -- when there is such
    a state: without at-start effects it is the state the action starts in.
    """
    for index, action in enumerate(domain.actions):
        if only_domain_actions and not _is_domain_action(action):
            continue
        changes = {"precondition": _conjoin(action.precondition, formula)}
        if during and _has_start_effects(action.effect):
            changes["late_condition"] = _conjoin(action.late_condition, formula)
        domain.actions[index] = replace(action, **changes)


def _add_conditional_effect(domain: Domain, condition, body) -> None:
    """Give every action that changes the world a ``when condition body`` effect.

    Used for monitors that must not be optional: the planner cannot decline to
    notice that a constraint's trigger became true. That includes a timed
    literal firing: the state it fires in is on the trajectory too, and a
    monitor that skipped it would miss a trigger the clock produced.
    """
    for index, action in enumerate(domain.actions):
        if not _changes_the_world(action):
            continue
        domain.actions[index] = replace(
            action, effect=_also(action.effect, WhenEffect(condition, body))
        )


def _is_empty_effect(effect) -> bool:
    """True when ``effect`` changes no fact and no fluent (a cost is neither)."""
    if isinstance(effect, ConjunctiveEffect):
        return all(_is_empty_effect(part) for part in effect.parts)
    return isinstance(effect, IncreaseCostEffect)


def _has_start_effects(effect) -> bool:
    """True for a durative action whose at-start effects are not empty."""
    if isinstance(effect, TimedEffect):
        return not _is_empty_effect(effect.start)
    if isinstance(effect, ConjunctiveEffect):
        return any(_has_start_effects(part) for part in effect.parts)
    return False


def _metric_mentions(problem: Problem, name: str) -> bool:
    def walk(expression):
        if isinstance(expression, FluentRef):
            return expression.name == name
        if isinstance(expression, Arithmetic):
            return walk(expression.left) or walk(expression.right)
        return False

    return problem.metric is not None and walk(problem.metric[1])


# --------------------------------------------------------------------------
# quantified constraints and preferences
# --------------------------------------------------------------------------
def _object_pool(domain: Domain, problem: Problem) -> dict:
    """``type -> objects``, as the grounder will build it."""
    from .grounding import _build_type_objects  # grounding imports this module

    return _build_type_objects(domain, problem)


def _bindings(params, pool: dict):
    """Every ``{var: object}`` assignment for ``params``, with its objects."""
    variables = [var for var, _ in params]
    for combo in product(*(pool.get(typ, ()) for _, typ in params)):
        yield dict(zip(variables, combo)), combo


def _substitute_term(expression, binding: dict):
    if isinstance(expression, FluentRef):
        if not expression.args and expression.name in binding:
            return FluentRef(binding[expression.name], ())
        args = tuple(binding.get(arg, arg) for arg in expression.args)
        return FluentRef(expression.name, args)
    if isinstance(expression, Arithmetic):
        return Arithmetic(
            expression.op,
            _substitute_term(expression.left, binding),
            _substitute_term(expression.right, binding),
        )
    return expression


def _substitute(formula, binding: dict):
    """``formula`` with the variables in ``binding`` replaced by objects."""
    if isinstance(formula, Literal):
        atom = formula.atom
        args = tuple(binding.get(arg, arg) for arg in atom.args)
        return Literal(Atom(atom.predicate, args), formula.positive)
    if isinstance(formula, EqualityConstraint):
        return EqualityConstraint(
            binding.get(formula.left, formula.left),
            binding.get(formula.right, formula.right),
            formula.positive,
        )
    if isinstance(formula, Comparison):
        return Comparison(
            formula.op,
            _substitute_term(formula.left, binding),
            _substitute_term(formula.right, binding),
        )
    if isinstance(formula, (And, Or)):
        return type(formula)(tuple(_substitute(p, binding) for p in formula.parts))
    if isinstance(formula, (Forall, Exists)):
        # An inner quantifier shadows the outer variable of the same name.
        bound = {var for var, _ in formula.params}
        inner = {var: obj for var, obj in binding.items() if var not in bound}
        return type(formula)(formula.params, _substitute(formula.body, inner))
    return formula


# --------------------------------------------------------------------------
# object fluents
# --------------------------------------------------------------------------
def compile_object_fluents(domain: Domain, problem: Problem):
    """Turn ``(location ?p) - place`` into a predicate plus a uniqueness rule.

    ``(= (location ?p) ?x)`` becomes ``(__fn-location ?p ?x)``, and
    ``(assign (location ?p) ?y)`` clears the old value before setting the new
    one, so the predicate stays single-valued. Object fluents used as *nested
    terms* — ``(at ?t (location ?p))`` — are refused: flattening those needs a
    fresh existential per occurrence, and the equality form above expresses the
    same thing without the guesswork.
    """
    if not domain.object_fluents:
        return domain, problem

    fluents = {fluent.name: fluent for fluent in domain.object_fluents}

    def predicate_name(name: str) -> str:
        return f"{PREFIX}fn-{name}"

    for fluent in domain.object_fluents:
        params = list(fluent.params) + [("?__value", fluent.result_type)]
        _declare(domain, predicate_name(fluent.name), params)

    def object_value(fluent: str, value) -> str:
        """The object a term names, refusing anything that is not one."""
        if isinstance(value, Number):
            raise UnsupportedFeatureError(
                f"'{fluent}' returns an object, so it cannot be compared with "
                "or set to a number"
            )
        if isinstance(value, FluentRef) and (value.args or value.name in fluents):
            # (loc ?q) here would be read as the *object* "loc".
            raise UnsupportedFeatureError(
                f"'{fluent}' is compared with or set to the value of another "
                f"function, {value}; that is not supported — introduce a "
                "variable for it, e.g. (and (= (loc ?q) ?x) ...)"
            )
        if not isinstance(value, FluentRef):
            raise UnsupportedFeatureError(
                f"'{fluent}' returns an object, so arithmetic on it is meaningless"
            )
        return value.name

    def rewrite_condition(formula):
        if isinstance(formula, Comparison) and formula.op in ("=", "!="):
            left, right = formula.left, formula.right
            for a, b in ((left, right), (right, left)):
                if isinstance(a, FluentRef) and a.name in fluents:
                    if isinstance(b, FluentRef) and b.name in fluents:
                        raise UnsupportedFeatureError(
                            "comparing two object fluents directly is not "
                            "supported; introduce a variable for one of them"
                        )
                    value = object_value(a.name, b)
                    atom = _fact(predicate_name(a.name), *a.args, value)
                    return Literal(atom, formula.op == "=")
            return formula
        if isinstance(formula, And):
            return And(tuple(rewrite_condition(p) for p in formula.parts))
        if isinstance(formula, Or):
            return Or(tuple(rewrite_condition(p) for p in formula.parts))
        if isinstance(formula, Forall):
            return Forall(formula.params, rewrite_condition(formula.body))
        if isinstance(formula, Exists):
            return Exists(formula.params, rewrite_condition(formula.body))
        return formula

    def rewrite_effect(effect):
        if isinstance(effect, ConjunctiveEffect):
            return ConjunctiveEffect([rewrite_effect(p) for p in effect.parts])
        if isinstance(effect, TimedEffect):
            return TimedEffect(rewrite_effect(effect.start), rewrite_effect(effect.end))
        if isinstance(effect, ForallEffect):
            return ForallEffect(effect.params, rewrite_effect(effect.body))
        if isinstance(effect, WhenEffect):
            return WhenEffect(
                rewrite_condition(effect.condition), rewrite_effect(effect.body)
            )
        if isinstance(effect, NumericEffect) and effect.target.name in fluents:
            if effect.op != "assign":
                raise UnsupportedFeatureError(
                    f"'{effect.op}' is arithmetic, but '{effect.target.name}' "
                    "returns an object; only 'assign' is meaningful"
                )
            fluent = fluents[effect.target.name]
            name = predicate_name(fluent.name)
            new = object_value(fluent.name, effect.value)
            # Clear whatever the function used to return, then set the new
            # value: that is what keeps it a function rather than a relation.
            clear = ForallEffect(
                [("?__old", fluent.result_type)],
                WhenEffect(
                    _holds(name, *effect.target.args, "?__old"),
                    DelEffect(_fact(name, *effect.target.args, "?__old")),
                ),
            )
            return ConjunctiveEffect(
                [clear, AddEffect(_fact(name, *effect.target.args, new))]
            )
        return effect

    def rewrite_constraint(constraint):
        if isinstance(constraint, Preference):
            return replace(constraint, body=rewrite_constraint(constraint.body))
        args = tuple(rewrite_condition(arg) for arg in constraint.args)
        return replace(constraint, args=args)

    for index, action in enumerate(domain.actions):
        late = action.late_condition
        domain.actions[index] = replace(
            action,
            precondition=rewrite_condition(action.precondition),
            effect=rewrite_effect(action.effect),
            late_condition=None if late is None else rewrite_condition(late),
        )
    for index, rule in enumerate(domain.derived):
        domain.derived[index] = replace(rule, body=rewrite_condition(rule.body))

    problem.goal = rewrite_condition(problem.goal)
    for index, preference in enumerate(problem.preferences):
        problem.preferences[index] = replace(
            preference, body=rewrite_condition(preference.body)
        )
    # Trajectory constraints read the same terms; left alone, an equality on an
    # object fluent would be taken for a numeric comparison.
    domain.constraints = [rewrite_constraint(c) for c in domain.constraints]
    problem.constraints = [rewrite_constraint(c) for c in problem.constraints]

    # `(= (location p1) depot)` in :init becomes a plain fact.
    for fluent_ref, value in problem.init_objects.items():
        if fluent_ref.name not in fluents:
            raise PDDLError(
                f"'{fluent_ref.name}' is assigned an object in :init but is not "
                "declared as an object fluent in :functions"
            )
        problem.init.append(
            _fact(predicate_name(fluent_ref.name), *fluent_ref.args, value)
        )
    problem.init_objects = {}
    return domain, problem


# --------------------------------------------------------------------------
# timed initial literals
# --------------------------------------------------------------------------
def compile_timed_initials(domain: Domain, problem: Problem):
    """Give the model a clock, and make each timed literal fire off it.

    Elapsed time becomes the numeric fluent ``(__time)``, advanced by each
    durative action's duration. Every timed literal gets a zero-cost
    ``__fire-til-k`` action guarded by ``(>= (__time) t)``, and a
    ``__wait-til-k`` action that lets the planner advance the clock to ``t``
    when it wants the literal to happen.

    The literal must not be *skipped*: every domain action therefore carries
    ``(or (< (__time) t) (__til-k))`` for each timed literal, so once the clock
    passes ``t`` nothing else may happen until the literal has fired.

    Literals must also fire **in time order**, which is a separate constraint:
    ``(at 0 (open))`` and ``(at 3 (not (open)))`` describe a shop that opens then
    shuts, but firing them the other way round would leave it open forever. Each
    firing therefore requires every earlier literal to have fired already.

    Because actions do not overlap, a literal scheduled strictly inside an
    action's duration fires immediately after it rather than during it. The
    goal carries the same guards, so a plan cannot end in a state a due literal
    would already have changed.
    """
    if not problem.timed_initials:
        return domain, problem
    if _metric_mentions(problem, "total-time"):
        raise UnsupportedFeatureError(
            "a (total-time) metric with timed initial literals is not supported: "
            "waiting for a literal takes time that no action is charged for, so "
            "the search would not minimise the makespan"
        )

    problem.init_numeric = dict(problem.init_numeric)
    problem.init_numeric.setdefault(CLOCK, 0.0)

    # The clock only moves if something moves it.
    for index, action in enumerate(domain.actions):
        if action.duration is None or not _is_domain_action(action):
            continue
        domain.actions[index] = replace(
            action,
            effect=_also(
                action.effect, NumericEffect("increase", CLOCK, action.duration)
            ),
        )

    guards = []
    earlier: list = []
    # "Literal j is the first one not fired yet, and it is not due": one of
    # these holds exactly when every due literal has fired.
    caught_up = []
    for k, timed in enumerate(sorted(problem.timed_initials, key=lambda t: t.time)):
        marker = f"{PREFIX}til-{k}"
        _declare(domain, marker)
        due = Comparison(">=", CLOCK, Number(timed.time))
        not_due = Comparison("<", CLOCK, Number(timed.time))
        # Everything scheduled before this must already have happened.
        in_order = _conjoin(*[_holds(name) for name in earlier])

        body = (
            AddEffect(timed.literal.atom)
            if timed.literal.positive
            else DelEffect(timed.literal.atom)
        )
        domain.actions.append(
            Action(
                f"{PREFIX}fire-til-{k}",
                [],
                _conjoin(due, _absent(marker), in_order),
                ConjunctiveEffect([body, AddEffect(_fact(marker)), _free()]),
            )
        )
        domain.actions.append(
            Action(
                f"{PREFIX}wait-til-{k}",
                [],
                # Waiting past an event that has not happened yet would skip it.
                _conjoin(not_due, in_order),
                ConjunctiveEffect(
                    [NumericEffect("assign", CLOCK, Number(timed.time)), _free()]
                ),
            )
        )
        guards.append(Or((not_due, _holds(marker))))
        caught_up.append(_conjoin(in_order, _absent(marker), not_due))
        earlier.append(marker)
    caught_up.append(_conjoin(*[_holds(name) for name in earlier]))

    # A due literal blocks everything else until it has fired -- including the
    # end of the plan. Without the goal guard, a durative action that overran a
    # literal could end the plan in a state the literal would have changed.
    # Literals fire in order, so the goal can say it with one disjunct per
    # literal rather than the 2^k a conjunction of the guards expands to.
    for guard in guards:
        _add_precondition(domain, guard)
    problem.goal = _conjoin(problem.goal, Or(tuple(caught_up)))
    problem.timed_initials = []
    return domain, problem


# --------------------------------------------------------------------------
# trajectory constraints
# --------------------------------------------------------------------------
def compile_constraints(domain: Domain, problem: Problem):
    """Compile ``(:constraints ...)`` into preconditions, monitors and goals.

    * ``always phi`` — conjoined onto every action's precondition and onto the
      goal. Every state on a plan's trajectory is either the initial state, a
      state an action is taken from, or the final state, so those three checks
      cover all of them.
    * ``at-end phi`` — conjoined onto the goal.
    * ``sometime phi`` — a zero-cost ``__observe`` action, applicable exactly
      when ``phi`` holds, sets a monitor fact the goal then requires.
    * ``sometime-before phi psi`` — the same monitor for ``psi``, plus
      ``always (phi implies monitor)``.
    * ``sometime-after phi psi`` — a *forced* monitor: every action records an
      outstanding obligation when ``phi`` holds without ``psi``, and discharges
      it when ``psi`` holds. Monitors read the state an action is applied in,
      so the final state is checked by the goal instead: ``psi`` holds there,
      or nothing is outstanding and ``phi`` does not hold either.
    * ``at-most-once phi`` — forced monitors for "phi has held" and "phi has
      since stopped", plus ``always not (phi and stopped)``.

    Forced monitors ride on conditional effects, which the planner cannot
    decline; optional ones ride on actions, which it applies when convenient.
    The difference matters: a constraint the planner could satisfy by *not
    looking* would not be a constraint. Forced monitors ride on the actions
    that fire timed literals as well, for the same reason.

    A constraint under ``forall`` becomes one constraint per binding.
    """
    constraints = list(domain.constraints) + list(problem.constraints)
    if not constraints:
        return domain, problem

    for constraint in constraints:
        if isinstance(constraint, Preference):
            raise UnsupportedFeatureError(
                f"the soft constraint '{constraint.name}' is not supported: "
                "preferences are supported over goals, not over trajectory "
                "constraints"
            )
    constraints = _expand_constraints(domain, problem, constraints)

    # A durative action with at-start effects passes through a state between
    # its start and its end. `always` is checked there (see below), but the
    # monitors only see the states actions are applied in, so they would miss
    # it: refuse rather than accept a plan whose middle broke the constraint.
    if any(_has_start_effects(action.effect) for action in domain.actions):
        for constraint in constraints:
            if constraint.kind not in ("always", "at-end"):
                raise UnsupportedFeatureError(
                    f"a '{constraint.kind}' constraint over a domain whose "
                    "durative actions have at-start effects is not supported: "
                    "its monitor cannot see the state between an action's start "
                    "and end. Only 'always' and 'at-end' are"
                )

    goal_parts = [problem.goal]
    invariants = []

    for index, constraint in enumerate(constraints):
        kind = constraint.kind
        if kind == "always":
            invariants.append(constraint.args[0])
        elif kind == "at-end":
            goal_parts.append(constraint.args[0])
        elif kind == "sometime":
            marker = _observation_monitor(domain, index, constraint.args[0])
            goal_parts.append(_holds(marker))
        elif kind == "sometime-before":
            trigger, earlier = constraint.args
            marker = _observation_monitor(domain, index, earlier)
            # "phi implies the monitor" as an invariant: by the time phi holds,
            # psi must already have been observed.
            invariants.append(Or((_negate(trigger), _holds(marker))))
        elif kind == "sometime-after":
            trigger, follower = constraint.args
            marker = f"{PREFIX}pending-{index}"
            _declare(domain, marker)
            _add_conditional_effect(
                domain,
                _conjoin(trigger, _negate(follower)),
                AddEffect(_fact(marker)),
            )
            _add_conditional_effect(domain, follower, DelEffect(_fact(marker)))
            # No monitor reads the final state, so the goal does: `psi` there
            # discharges any obligation, including one the last action's
            # monitor just recorded; otherwise nothing may be outstanding and
            # the final state must not create a fresh one.
            goal_parts.append(
                Or((follower, _conjoin(_absent(marker), _negate(trigger))))
            )
        elif kind == "at-most-once":
            phi = constraint.args[0]
            seen = f"{PREFIX}amo-seen-{index}"
            closed = f"{PREFIX}amo-closed-{index}"
            _declare(domain, seen)
            _declare(domain, closed)
            _add_conditional_effect(domain, phi, AddEffect(_fact(seen)))
            _add_conditional_effect(
                domain,
                _conjoin(_negate(phi), _holds(seen)),
                AddEffect(_fact(closed)),
            )
            # Holding again after an interval has closed is the second interval.
            invariants.append(Or((_negate(phi), _absent(closed))))
        else:  # pragma: no cover - the parser rejects anything else
            raise UnsupportedFeatureError(f"unsupported constraint '{kind}'")

    for invariant in invariants:
        # Every action, not just the domain's own. The coverage argument above
        # holds only if each state on the trajectory is either taken from by an
        # action that checks the invariant or is the final state — and the
        # actions timed initial literals compile to *change facts*. Exempting
        # them let a plan step through a state the invariant forbade: a literal
        # that clears `(safe)` at t=3 and one that restores it at t=4 fire
        # back-to-back with nothing checking the state in between. `during`
        # also checks the state between a durative action's start and end.
        _add_precondition(domain, invariant, only_domain_actions=False, during=True)
        goal_parts.append(invariant)

    problem.goal = _conjoin(*goal_parts)
    domain.constraints = []
    problem.constraints = []
    return domain, problem


def _expand_constraints(domain: Domain, problem: Problem, constraints) -> list:
    """One constraint per binding of a ``forall`` around it."""
    if not any(constraint.params for constraint in constraints):
        return constraints
    pool = _object_pool(domain, problem)
    expanded = []
    for constraint in constraints:
        if not constraint.params:
            expanded.append(constraint)
            continue
        for binding, _ in _bindings(constraint.params, pool):
            args = tuple(_substitute(arg, binding) for arg in constraint.args)
            expanded.append(Constraint(constraint.kind, args))
    return expanded


def _observation_monitor(domain: Domain, index: int, formula) -> str:
    """A fact the planner can set, for free, whenever ``formula`` holds."""
    marker = f"{PREFIX}seen-{index}"
    _declare(domain, marker)
    domain.actions.append(
        Action(
            f"{PREFIX}observe-{index}",
            [],
            _conjoin(formula, _absent(marker)),
            ConjunctiveEffect([AddEffect(_fact(marker)), _free()]),
        )
    )
    return marker


# --------------------------------------------------------------------------
# preferences
# --------------------------------------------------------------------------
def compile_preferences(domain: Domain, problem: Problem):
    """Turn each soft goal into a priced choice: satisfy it, or pay for it.

    A closing phase makes this sound. ``__close`` ends the plan — every action
    before it, the domain's own and the ones firing timed literals alike,
    requires that it has *not* happened — after which each preference is
    resolved by one of two zero-parameter actions: a free one that requires the
    preference to hold, or one costing the metric's ``(is-violated p)`` weight
    that does not. Cost-optimal search then picks whichever is cheaper, which is
    exactly what minimising the metric means.

    Freezing the state first is the point: without it the planner could satisfy
    a preference halfway through and then break it, and still be paid for it.

    Under a metric, a preference it does not mention costs nothing to violate —
    it has no effect on plan quality. Without a metric each costs 1. A
    preference under ``forall`` is one preference per binding, each priced at
    the name's weight, as ``(is-violated p)`` counts violated instances.
    """
    declared = {preference.name for preference in problem.preferences}
    for name in problem.violation_weights:
        if name not in declared:
            raise PDDLError(
                f"the metric prices (is-violated {name}), but the goal declares "
                f"no preference named '{name}'"
            )
    if not problem.preferences:
        return domain, problem

    default_weight = 0.0 if problem.metric is not None else 1.0
    instances = []  # (name, label, body)
    pool = None
    for preference in problem.preferences:
        if not preference.params:
            instances.append((preference.name, preference.name, preference.body))
            continue
        if pool is None:
            pool = _object_pool(domain, problem)
        for binding, objects in _bindings(preference.params, pool):
            label = "-".join((preference.name,) + tuple(objects))
            instances.append(
                (preference.name, label, _substitute(preference.body, binding))
            )

    closed = f"{PREFIX}closed"
    _declare(domain, closed)
    # Everything stops once the plan is closed, timed literals included: a
    # literal firing afterwards would change the state the preferences were
    # just paid out on.
    _add_precondition(domain, _absent(closed), only_domain_actions=False)

    domain.actions.append(
        Action(
            f"{PREFIX}close",
            [],
            _absent(closed),
            ConjunctiveEffect([AddEffect(_fact(closed)), _free()]),
        )
    )

    goal_parts = [problem.goal, _holds(closed)]
    for index, (name, label, body) in enumerate(instances):
        done = f"{PREFIX}pref-{index}"
        _declare(domain, done)
        weight = float(problem.violation_weights.get(name, default_weight))
        if weight < 0:
            raise PDDLError(
                f"preference '{name}' has a negative violation "
                "weight, which would reward breaking it"
            )
        domain.actions.append(
            Action(
                f"{PREFIX}satisfy-{label}",
                [],
                _conjoin(_holds(closed), _absent(done), body),
                ConjunctiveEffect([AddEffect(_fact(done)), _free()]),
            )
        )
        # The penalty rides on the action cost, so plain cost-optimal search
        # optimises the metric without knowing what a preference is.
        domain.actions.append(
            Action(
                f"{PREFIX}violate-{label}",
                [],
                _conjoin(_holds(closed), _absent(done)),
                ConjunctiveEffect([AddEffect(_fact(done)), _increase_cost(weight)]),
            )
        )
        goal_parts.append(_holds(done))

    problem.goal = _conjoin(*goal_parts)
    problem.preferences = []
    return domain, problem


def _increase_cost(amount: float):
    return IncreaseCostEffect(amount)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def compile_problem(domain: Domain, problem: Problem):
    """Apply every PDDL 3 compilation, in the order they depend on each other.

    Object fluents go first because they rewrite terms everywhere else reads;
    preferences go last because their closing phase must sit outside the
    machinery the other compilations add.

    The inputs are copied first: every compilation edits the domain and problem
    it is given, and a parsed domain is routinely grounded against several
    problems. Compiling in place leaked one problem's synthetic actions into the
    next and consumed the domain's own constraints on the first.
    """
    domain, problem = copy.deepcopy(domain), copy.deepcopy(problem)
    domain, problem = compile_object_fluents(domain, problem)
    domain, problem = compile_timed_initials(domain, problem)
    domain, problem = compile_constraints(domain, problem)
    domain, problem = compile_preferences(domain, problem)
    return domain, problem
