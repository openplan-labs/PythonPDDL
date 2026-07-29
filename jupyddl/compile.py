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

from dataclasses import replace

from .parser.ast import (
    Action,
    AddEffect,
    And,
    Atom,
    Comparison,
    ConjunctiveEffect,
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


def _declare(domain: Domain, name: str, params=()) -> None:
    if all(p.name != name for p in domain.predicates):
        domain.predicates.append(Predicate(name, list(params)))


def _add_precondition(domain: Domain, formula, only_domain_actions=True) -> None:
    """Conjoin ``formula`` onto every action's precondition."""
    for index, action in enumerate(domain.actions):
        if only_domain_actions and not _is_domain_action(action):
            continue
        domain.actions[index] = replace(
            action, precondition=_conjoin(action.precondition, formula)
        )


def _add_conditional_effect(domain: Domain, condition, body) -> None:
    """Give every domain action a ``when condition body`` effect.

    Used for monitors that must not be optional: the planner cannot decline to
    notice that a constraint's trigger became true.
    """
    for index, action in enumerate(domain.actions):
        if not _is_domain_action(action):
            continue
        domain.actions[index] = replace(
            action, effect=_also(action.effect, WhenEffect(condition, body))
        )


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

    def rewrite_condition(formula):
        if isinstance(formula, Comparison) and formula.op in ("=", "!="):
            left, right = formula.left, formula.right
            for a, b in ((left, right), (right, left)):
                if isinstance(a, FluentRef) and a.name in fluents:
                    if isinstance(b, Number):
                        raise UnsupportedFeatureError(
                            f"'{a.name}' returns an object, so it cannot be "
                            "compared with a number"
                        )
                    value = b.name if isinstance(b, FluentRef) else str(b)
                    if isinstance(b, FluentRef) and b.name in fluents:
                        raise UnsupportedFeatureError(
                            "comparing two object fluents directly is not "
                            "supported; introduce a variable for one of them"
                        )
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
            value = effect.value
            new = value.name if isinstance(value, FluentRef) else str(value)
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

    for index, action in enumerate(domain.actions):
        domain.actions[index] = replace(
            action,
            precondition=rewrite_condition(action.precondition),
            effect=rewrite_effect(action.effect),
        )
    for index, rule in enumerate(domain.derived):
        domain.derived[index] = replace(rule, body=rewrite_condition(rule.body))

    problem.goal = rewrite_condition(problem.goal)
    for index, preference in enumerate(problem.preferences):
        problem.preferences[index] = Preference(
            preference.name, rewrite_condition(preference.body)
        )

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
    action's duration fires immediately after it rather than during it.
    """
    if not problem.timed_initials:
        return domain, problem

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
        earlier.append(marker)

    # A due literal blocks everything else until it has fired.
    for guard in guards:
        _add_precondition(domain, guard)
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
      it when ``psi`` holds. The goal requires nothing outstanding.
    * ``at-most-once phi`` — forced monitors for "phi has held" and "phi has
      since stopped", plus ``always not (phi and stopped)``.

    Forced monitors ride on conditional effects, which the planner cannot
    decline; optional ones ride on actions, which it applies when convenient.
    The difference matters: a constraint the planner could satisfy by *not
    looking* would not be a constraint.
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
            goal_parts.append(_absent(marker))
            # ...and the final state must not leave a fresh obligation either.
            goal_parts.append(Or((_negate(trigger), follower)))
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
        # back-to-back with nothing checking the state in between.
        _add_precondition(domain, invariant, only_domain_actions=False)
        goal_parts.append(invariant)

    problem.goal = _conjoin(*goal_parts)
    domain.constraints = []
    problem.constraints = []
    return domain, problem


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

    A closing phase makes this sound. ``__close`` ends the plan — every domain
    action requires that it has *not* happened — after which each preference is
    resolved by one of two zero-parameter actions: a free one that requires the
    preference to hold, or one costing the metric's ``(is-violated p)`` weight
    that does not. Cost-optimal search then picks whichever is cheaper, which is
    exactly what minimising the metric means.

    Freezing the state first is the point: without it the planner could satisfy
    a preference halfway through and then break it, and still be paid for it.
    """
    if not problem.preferences:
        return domain, problem

    closed = f"{PREFIX}closed"
    _declare(domain, closed)
    _add_precondition(domain, _absent(closed))

    domain.actions.append(
        Action(
            f"{PREFIX}close",
            [],
            _absent(closed),
            ConjunctiveEffect([AddEffect(_fact(closed)), _free()]),
        )
    )

    goal_parts = [problem.goal, _holds(closed)]
    for index, preference in enumerate(problem.preferences):
        done = f"{PREFIX}pref-{index}"
        _declare(domain, done)
        weight = float(problem.violation_weights.get(preference.name, 1.0))
        if weight < 0:
            raise PDDLError(
                f"preference '{preference.name}' has a negative violation "
                "weight, which would reward breaking it"
            )
        domain.actions.append(
            Action(
                f"{PREFIX}satisfy-{preference.name}",
                [],
                _conjoin(_holds(closed), _absent(done), preference.body),
                ConjunctiveEffect([AddEffect(_fact(done)), _free()]),
            )
        )
        domain.actions.append(
            Action(
                f"{PREFIX}violate-{preference.name}",
                [],
                _conjoin(_holds(closed), _absent(done)),
                ConjunctiveEffect([AddEffect(_fact(done))]),  # priced below
            )
        )
        # The penalty rides on the action cost, so plain cost-optimal search
        # optimises the metric without knowing what a preference is.
        domain.actions[-1] = replace(
            domain.actions[-1],
            effect=_also(domain.actions[-1].effect, _increase_cost(weight)),
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
    """
    domain, problem = compile_object_fluents(domain, problem)
    domain, problem = compile_timed_initials(domain, problem)
    domain, problem = compile_constraints(domain, problem)
    domain, problem = compile_preferences(domain, problem)
    return domain, problem
