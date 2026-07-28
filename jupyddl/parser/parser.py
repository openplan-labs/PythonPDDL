"""Recursive-descent PDDL parser producing :mod:`jupyddl.parser.ast` objects.

Conditions are parsed straight into **negation normal form**: ``imply`` is
rewritten, ``not`` is pushed down through ``and``/``or``/quantifiers by De
Morgan, and what comes out only ever negates a literal. Quantifiers survive into
the AST because expanding them needs the object pool, which the grounder owns.

Which requirement flags are accepted, and what each one compiles to, is decided
by :mod:`jupyddl.requirements` — not by ad-hoc checks in here.
"""

from __future__ import annotations

from ..requirements import expand as expand_requirements
from ..requirements import lookup as lookup_requirement
from ..requirements import unsupported_reason
from .ast import (
    Action,
    Constraint,
    AddEffect,
    And,
    Arithmetic,
    Atom,
    Comparison,
    ConjunctiveEffect,
    DelEffect,
    DerivedPredicate,
    Domain,
    EqualityConstraint,
    Exists,
    Forall,
    ForallEffect,
    Function,
    FluentRef,
    IncreaseCostEffect,
    Literal,
    Number,
    NumericEffect,
    ObjectFluent,
    Or,
    Preference,
    PDDLError,
    Predicate,
    Problem,
    TimedInitial,
    Truth,
    UnsupportedFeatureError,
    WhenEffect,
)
from .tokenizer import tokenize

COMPARISONS = {"<", "<=", "=", ">=", ">"}
ARITHMETIC = {"+", "-", "*", "/"}
ASSIGN_OPS = {"assign", "increase", "decrease", "scale-up", "scale-down"}
TOTAL_COST = "total-cost"

# Trajectory-constraint operators. The first group is compiled by
# jupyddl.compile; the second needs metric time, which the sequential model does
# not maintain, so it is refused by name rather than approximated.
CONSTRAINT_OPS = {
    "always",
    "sometime",
    "at-end",
    "at-most-once",
    "sometime-before",
    "sometime-after",
}
METRIC_TIME_OPS = {"within", "always-within", "hold-after", "hold-during"}

# Timed-condition / timed-effect markers inside a durative action.
TIME_SPECIFIERS = {"at", "over"}


def _parse_typed_list(items: list) -> list:
    """Parse ``?x ?y - type a b`` into ``[(name, type), ...]`` (default ``object``)."""
    result: list = []
    pending: list = []
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == "-":
            typ = items[i + 1]
            if isinstance(typ, list):
                if typ and typ[0] == "either":
                    raise UnsupportedFeatureError(
                        "'either' types are not supported; declare a supertype instead"
                    )
                raise UnsupportedFeatureError("'either' types are not supported")
            for name in pending:
                result.append((name, typ))
            pending = []
            i += 2
        else:
            if isinstance(tok, list):
                raise PDDLError(f"Unexpected nested list in typed list: {tok}")
            pending.append(tok)
            i += 1
    result.extend((name, "object") for name in pending)
    return result


def _is_number(token) -> bool:
    if isinstance(token, list):
        return False
    try:
        float(token)
    except (TypeError, ValueError):
        return False
    return True


def _parse_expression(expr):
    """Parse a numeric expression: a literal, a fluent, or arithmetic."""
    if not isinstance(expr, list):
        if _is_number(expr):
            return Number(float(expr))
        # A bare name is a zero-arity fluent, e.g. total-cost.
        return FluentRef(expr, ())
    if not expr:
        raise PDDLError("Empty numeric expression")
    head = expr[0]
    if head in ARITHMETIC:
        if len(expr) == 2 and head == "-":  # unary minus
            return Arithmetic("-", Number(0.0), _parse_expression(expr[1]))
        if len(expr) != 3:
            raise UnsupportedFeatureError(
                f"'{head}' takes exactly two arguments in a numeric expression"
            )
        return Arithmetic(head, _parse_expression(expr[1]), _parse_expression(expr[2]))
    if _is_number(head):
        return Number(float(head))
    for arg in expr[1:]:
        if isinstance(arg, list):
            raise UnsupportedFeatureError(
                f"nested term inside a fluent reference is not supported: {expr}"
            )
    return FluentRef(head, tuple(expr[1:]))


def _parse_atom(expr) -> Atom:
    if not expr or isinstance(expr[0], list):
        raise PDDLError(f"Malformed atom: {expr}")
    for arg in expr[1:]:
        if isinstance(arg, list):
            raise UnsupportedFeatureError(
                f"nested/numeric term in atom is not supported: {expr}"
            )
    return Atom(expr[0], tuple(expr[1:]))


def _negate_comparison(comparison: Comparison) -> Comparison:
    flip = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "=": "!="}
    return Comparison(flip[comparison.op], comparison.left, comparison.right)


def parse_condition(expr, positive: bool = True):
    """Parse a condition into negation normal form.

    ``positive`` carries the sign inward, so ``(not (or a b))`` becomes
    ``(and (not a) (not b))`` without ever building an explicit ``Not`` node.
    """
    if not isinstance(expr, list):
        raise PDDLError(f"Malformed condition: {expr}")
    if not expr:  # () means "true"
        return Truth(positive)

    head = expr[0]

    if head == "and":
        parts = tuple(parse_condition(sub, positive) for sub in expr[1:])
        return And(parts) if positive else Or(parts)
    if head == "or":
        parts = tuple(parse_condition(sub, positive) for sub in expr[1:])
        return Or(parts) if positive else And(parts)
    if head == "not":
        if len(expr) != 2:
            raise PDDLError(f"'not' takes exactly one argument: {expr}")
        return parse_condition(expr[1], not positive)
    if head == "imply":
        if len(expr) != 3:
            raise PDDLError(f"'imply' takes exactly two arguments: {expr}")
        # (imply a b) == (or (not a) b)
        parts = (
            parse_condition(expr[1], not positive),
            parse_condition(expr[2], positive),
        )
        return Or(parts) if positive else And(parts)
    if head == "forall":
        params = tuple(_parse_typed_list(expr[1]))
        body = parse_condition(expr[2], positive)
        return Forall(params, body) if positive else Exists(params, body)
    if head == "exists":
        params = tuple(_parse_typed_list(expr[1]))
        body = parse_condition(expr[2], positive)
        return Exists(params, body) if positive else Forall(params, body)
    if head == "=":
        # `=` is equality between terms unless either side is numeric.
        left, right = expr[1], expr[2]
        if isinstance(left, list) or isinstance(right, list) or _is_number(left):
            comparison = Comparison(
                "=", _parse_expression(left), _parse_expression(right)
            )
            return comparison if positive else _negate_comparison(comparison)
        return EqualityConstraint(left, right, positive)
    if head in COMPARISONS:
        comparison = Comparison(
            head, _parse_expression(expr[1]), _parse_expression(expr[2])
        )
        return comparison if positive else _negate_comparison(comparison)

    return Literal(_parse_atom(expr), positive)


def parse_effect(expr):
    if not isinstance(expr, list):
        raise PDDLError(f"Malformed effect: {expr}")
    if not expr:
        return ConjunctiveEffect([])
    head = expr[0]
    if head == "and":
        return ConjunctiveEffect([parse_effect(sub) for sub in expr[1:]])
    if head == "not":
        return DelEffect(_parse_atom(expr[1]))
    if head == "forall":
        return ForallEffect(_parse_typed_list(expr[1]), parse_effect(expr[2]))
    if head == "when":
        return WhenEffect(parse_condition(expr[1]), parse_effect(expr[2]))
    if head in ASSIGN_OPS:
        target = expr[1]
        name = target[0] if isinstance(target, list) else target
        value = _parse_expression(expr[2])
        if name == TOTAL_COST and head == "increase":
            # The classical action-cost shorthand: keep it as an integer cost.
            if isinstance(value, Number):
                return IncreaseCostEffect(value.value)
            raise UnsupportedFeatureError(
                "(increase (total-cost) ...) needs a constant amount"
            )
        args = tuple(target[1:]) if isinstance(target, list) else ()
        return NumericEffect(head, FluentRef(name, args), value)
    if head in ("or", "imply", "exists"):
        raise UnsupportedFeatureError(f"'{head}' is not allowed in an effect")
    return AddEffect(_parse_atom(expr))


def _section_key(section) -> str:
    return section[0] if section and isinstance(section[0], str) else ""


def _check_requirements(requirements) -> None:
    """Refuse a domain up front rather than mis-planning it later."""
    for flag in requirements:
        if lookup_requirement(flag) is None:
            raise UnsupportedFeatureError(
                f"unknown requirement flag '{flag}'; see jupyddl.requirements "
                "for the full list of recognised flags"
            )
    for flag in expand_requirements(requirements):
        reason = unsupported_reason(flag)
        if reason:
            raise UnsupportedFeatureError(reason)


def parse_domain(tokens: list) -> Domain:
    if _section_key(tokens) != "define":
        raise PDDLError("Domain file must start with (define ...)")

    name = ""
    requirements: list = []
    types: dict = {}
    constants: list = []
    predicates: list = []
    functions: list = []
    actions: list = []
    derived: list = []
    constraints: list = []
    object_fluents: list = []

    for section in tokens[1:]:
        key = _section_key(section)
        if key == "domain":
            name = section[1]
        elif key == ":requirements":
            requirements = list(section[1:])
            _check_requirements(requirements)
        elif key == ":types":
            for child, parent in _parse_typed_list(section[1:]):
                # In a :types list, "child - parent" reads name=child, type=parent.
                types[child] = parent
        elif key == ":constants":
            constants = _parse_typed_list(section[1:])
        elif key == ":predicates":
            for pred in section[1:]:
                predicates.append(Predicate(pred[0], _parse_typed_list(pred[1:])))
        elif key == ":functions":
            numeric, objects = _parse_functions(section[1:])
            functions.extend(numeric)
            object_fluents.extend(objects)
        elif key == ":action":
            actions.append(_parse_action(section))
        elif key == ":durative-action":
            actions.append(_parse_durative_action(section))
        elif key == ":derived":
            derived.append(_parse_derived(section))
        elif key == ":constraints":
            constraints.extend(_parse_constraints(section[1]))
    return Domain(
        name,
        requirements,
        types,
        constants,
        predicates,
        actions,
        functions,
        derived,
        object_fluents,
        constraints,
    )


def _parse_functions(items):
    """Split a ``:functions`` block into numeric functions and object fluents.

    A trailing ``- <type>`` applies to the functions declared before it. The
    type ``number`` (or no type at all) means a numeric fluent; anything else
    means the function returns an *object*, which is compiled away by
    :mod:`jupyddl.compile`.

    Returns ``(numeric, object_fluents)``.
    """
    numeric: list = []
    object_fluents: list = []
    pending: list = []
    index = 0
    while index < len(items):
        entry = items[index]
        if entry == "-":
            result_type = items[index + 1]
            index += 2
            for name, params in pending:
                if result_type == "number":
                    numeric.append(Function(name, params))
                else:
                    object_fluents.append(ObjectFluent(name, params, result_type))
            pending = []
            continue
        if isinstance(entry, list) and entry:
            pending.append((entry[0], _parse_typed_list(entry[1:])))
        index += 1
    # Anything left untyped defaults to a number, as PDDL specifies.
    numeric.extend(Function(name, params) for name, params in pending)
    return numeric, object_fluents


def _parse_action(section: list) -> Action:
    name = section[1]
    parameters: list = []
    precondition = Truth(True)
    effect = ConjunctiveEffect([])
    i = 2
    while i < len(section):
        tag = section[i]
        val = section[i + 1]
        if tag == ":parameters":
            parameters = _parse_typed_list(val)
        elif tag == ":precondition":
            precondition = parse_condition(val)
        elif tag == ":effect":
            effect = parse_effect(val)
        i += 2
    return Action(name, parameters, precondition, effect)


def _strip_time_specifier(expr):
    """Unwrap ``(at start X)`` / ``(at end X)`` / ``(over all X)`` to ``X``.

    The sequential compilation collapses the timeline, so the marker is dropped
    once it has been validated. See :mod:`jupyddl.requirements` for what that
    costs you.
    """
    if isinstance(expr, list) and len(expr) == 3 and expr[0] in TIME_SPECIFIERS:
        marker = expr[1]
        if expr[0] == "at" and marker in ("start", "end"):
            return expr[2]
        if expr[0] == "over" and marker == "all":
            return expr[2]
    return expr


def _flatten_timed(expr, out) -> None:
    if not isinstance(expr, list) or not expr:
        return
    if expr[0] == "and":
        for sub in expr[1:]:
            _flatten_timed(sub, out)
        return
    out.append(_strip_time_specifier(expr))


def _parse_durative_action(section: list) -> Action:
    """Compile ``(:durative-action ...)`` into a single sequential action."""
    name = section[1]
    parameters: list = []
    duration = None
    condition_parts: list = []
    effect_parts: list = []

    i = 2
    while i < len(section):
        tag = section[i]
        val = section[i + 1]
        if tag == ":parameters":
            parameters = _parse_typed_list(val)
        elif tag == ":duration":
            duration = _parse_duration(val, name)
        elif tag == ":condition":
            _flatten_timed(val, condition_parts)
        elif tag == ":effect":
            _flatten_timed(val, effect_parts)
        i += 2

    if duration is None:
        raise PDDLError(f"durative action '{name}' has no :duration")

    precondition = parse_condition(["and"] + condition_parts)
    effect = parse_effect(["and"] + effect_parts)
    return Action(name, parameters, precondition, effect, duration=duration)


def _parse_duration(expr, action_name: str):
    """Read a ``:duration``, which may be an equality or a set of bounds.

    ``(= ?duration 5)`` fixes it. ``(and (>= ?duration 2) (<= ?duration 6))``
    bounds it, and the sequential compilation then picks the **shortest feasible
    duration**: with no concurrency and no continuous change, nothing in the
    model prefers a longer action, so the tightest lower bound is the
    makespan-optimal choice. Strict inequalities are refused because they have
    no shortest feasible value.
    """
    lower, upper, exact = _duration_bounds(expr, action_name)

    if exact is not None:
        if lower is not None and exact < lower:
            raise PDDLError(f"duration bounds for '{action_name}' are contradictory")
        if upper is not None and exact > upper:
            raise PDDLError(f"duration bounds for '{action_name}' are contradictory")
        return Number(exact)

    chosen = lower if lower is not None else 0.0
    if upper is not None and chosen > upper:
        raise PDDLError(
            f"duration bounds for '{action_name}' are contradictory: "
            f"the lower bound {chosen:g} exceeds the upper bound {upper:g}"
        )
    if lower is None and upper is None:
        raise PDDLError(f"malformed :duration for '{action_name}'")
    return Number(chosen)


def _duration_bounds(expr, action_name: str):
    """Collect ``(lower, upper, exact)`` from a duration constraint tree."""
    lower = upper = exact = None

    def visit(node):
        nonlocal lower, upper, exact
        if not isinstance(node, list) or not node:
            raise PDDLError(f"malformed :duration for '{action_name}'")
        head = node[0]
        if head == "and":
            for sub in node[1:]:
                visit(sub)
            return
        if head in ("<", ">"):
            raise UnsupportedFeatureError(
                f"strict duration inequality '{head}' in '{action_name}' is not "
                "supported: it has no shortest feasible duration. Use >= or <=."
            )
        if head not in ("=", ">=", "<="):
            raise PDDLError(
                f"unexpected operator '{head}' in the :duration of '{action_name}'"
            )
        if len(node) != 3 or node[1] != "?duration":
            raise UnsupportedFeatureError(
                f"the :duration of '{action_name}' must constrain ?duration "
                "against a constant"
            )
        value = _parse_expression(node[2])
        if not isinstance(value, Number):
            raise UnsupportedFeatureError(
                f"the duration of '{action_name}' must be a constant; durations "
                "that read numeric fluents are not supported"
            )
        amount = float(value.value)
        if head == "=":
            exact = amount
        elif head == ">=":
            lower = amount if lower is None else max(lower, amount)
        else:
            upper = amount if upper is None else min(upper, amount)

    visit(expr)
    return lower, upper, exact


def _parse_derived(section: list) -> DerivedPredicate:
    """Parse ``(:derived (head ?x - t) body)``."""
    head_expr = section[1]
    if not isinstance(head_expr, list) or not head_expr:
        raise PDDLError("malformed (:derived ...) head")
    params = _parse_typed_list(head_expr[1:])
    head = Atom(head_expr[0], tuple(var for var, _ in params))
    body = parse_condition(section[2])
    return DerivedPredicate(head, params, body)


def _parse_constraints(expr) -> list:
    """Parse a ``(:constraints ...)`` body into :class:`Constraint` nodes."""
    out: list = []

    def visit(node):
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if head == "and":
            for sub in node[1:]:
                visit(sub)
            return
        if head == "forall":
            # A quantified constraint expands once the object pool exists; the
            # body is what carries the modality, so recurse into it and let the
            # grounder expand the quantifier inside each operand.
            params = tuple(_parse_typed_list(node[1]))
            for constraint in _parse_constraints(node[2]):
                out.append(
                    Constraint(
                        constraint.kind,
                        tuple(Forall(params, arg) for arg in constraint.args),
                    )
                )
            return
        if head == "preference":
            name = node[1]
            for constraint in _parse_constraints(node[2]):
                out.append(Preference(name, constraint))
            return
        if head in METRIC_TIME_OPS:
            raise UnsupportedFeatureError(
                f"the '{head}' constraint needs metric time, which the "
                "sequential compilation does not maintain"
            )
        if head not in CONSTRAINT_OPS:
            raise UnsupportedFeatureError(
                f"'{head}' is not a recognised trajectory constraint; "
                f"supported: {', '.join(sorted(CONSTRAINT_OPS))}"
            )
        arity = 2 if head in ("sometime-before", "sometime-after") else 1
        operands = node[1 : 1 + arity]
        if len(operands) != arity:
            raise PDDLError(f"'{head}' takes {arity} operand(s): {node}")
        out.append(Constraint(head, tuple(parse_condition(o) for o in operands)))

    visit(expr)
    return out


def _split_preferences(expr, found: list):
    """Strip ``(preference name phi)`` out of a goal, leaving the hard part.

    Returns the goal s-expression with every preference replaced by ``()``
    (trivially true) and appends the extracted ones to ``found``.
    """
    if not isinstance(expr, list) or not expr:
        return expr
    head = expr[0]
    if head == "preference":
        if len(expr) == 3:
            found.append((expr[1], expr[2]))
        elif len(expr) == 2:  # an unnamed preference
            found.append((f"__pref{len(found) + 1}", expr[1]))
        else:
            raise PDDLError(f"malformed preference: {expr}")
        return []
    if head in ("and", "or", "not", "imply"):
        return [head] + [_split_preferences(sub, found) for sub in expr[1:]]
    if head in ("forall", "exists"):
        return [head, expr[1], _split_preferences(expr[2], found)]
    return expr


def _parse_timed_initial(fact):
    """Parse ``(at <time> <literal>)`` from ``:init``; None when it is an atom.

    ``(at 5 (on a b))`` is a timed initial literal, but ``(at truck1 depot)`` is
    an ordinary atom about location -- the number in the second position is what
    distinguishes them.
    """
    if len(fact) != 3 or fact[0] != "at" or not _is_number(fact[1]):
        return None
    body = fact[2]
    if not isinstance(body, list) or not body:
        raise PDDLError(f"malformed timed initial literal: {fact}")
    if body[0] == "not":
        return TimedInitial(float(fact[1]), Literal(_parse_atom(body[1]), False))
    return TimedInitial(float(fact[1]), Literal(_parse_atom(body), True))


def _collect_violation_weights(expression, weights: dict, factor: float = 1.0):
    """Read the ``(is-violated p)`` coefficients out of a metric expression.

    ``(+ (total-cost) (* 4 (is-violated p)))`` prices preference ``p`` at 4.
    Anything not of that shape is ignored; the default weight is 1.
    """
    if isinstance(expression, FluentRef):
        if expression.name == "is-violated" and expression.args:
            name = expression.args[0]
            weights[name] = weights.get(name, 0.0) + factor
        return
    if isinstance(expression, Arithmetic):
        if expression.op == "+":
            _collect_violation_weights(expression.left, weights, factor)
            _collect_violation_weights(expression.right, weights, factor)
        elif expression.op == "-":
            _collect_violation_weights(expression.left, weights, factor)
            _collect_violation_weights(expression.right, weights, -factor)
        elif expression.op == "*":
            left, right = expression.left, expression.right
            if isinstance(left, Number):
                _collect_violation_weights(right, weights, factor * left.value)
            elif isinstance(right, Number):
                _collect_violation_weights(left, weights, factor * right.value)


def parse_problem(tokens: list) -> Problem:
    if _section_key(tokens) != "define":
        raise PDDLError("Problem file must start with (define ...)")

    name = ""
    domain_name = ""
    objects: list = []
    init: list = []
    init_numeric: dict = {}
    goal = Truth(True)
    metric = None
    metric_minimize_cost = False
    preferences: list = []
    constraints: list = []
    timed_initials: list = []
    init_objects: dict = {}
    violation_weights: dict = {}

    for section in tokens[1:]:
        key = _section_key(section)
        if key == "problem":
            name = section[1]
        elif key == ":domain":
            domain_name = section[1]
        elif key == ":objects":
            objects = _parse_typed_list(section[1:])
        elif key == ":requirements":
            _check_requirements(list(section[1:]))
        elif key == ":init":
            for fact in section[1:]:
                if not fact:
                    continue
                if fact[0] == "=":
                    target, value = fact[1], fact[2]
                    fluent = _parse_expression(target)
                    if not isinstance(fluent, FluentRef):
                        continue
                    if _is_number(value):
                        init_numeric[fluent] = float(value)
                    elif not isinstance(value, list):
                        # `(= (location p1) depot)` assigns an object, not a
                        # number: an object fluent, compiled away later.
                        init_objects[fluent] = value
                    continue
                timed = _parse_timed_initial(fact)
                if timed is not None:
                    timed_initials.append(timed)
                    continue
                init.append(_parse_atom(fact))
        elif key == ":goal":
            stripped = _split_preferences(section[1], preferences)
            goal = parse_condition(stripped)
        elif key == ":metric":
            metric, metric_minimize_cost = _parse_metric(section)
            if metric is not None:
                _collect_violation_weights(metric[1], violation_weights)
        elif key == ":constraints":
            constraints.extend(_parse_constraints(section[1]))
    return Problem(
        name,
        domain_name,
        objects,
        init,
        goal,
        metric_minimize_cost,
        init_numeric,
        metric,
        [
            Preference(pref_name, parse_condition(body))
            for pref_name, body in preferences
        ],
        constraints,
        timed_initials,
        init_objects,
        violation_weights,
    )


def _parse_metric(section):
    """Read ``(:metric minimize <expr>)``; returns ``((dir, expr), is_total_cost)``."""
    if len(section) < 3:
        return None, False
    direction = section[1]
    if direction not in ("minimize", "maximize"):
        raise PDDLError(f"unknown metric direction '{direction}'")
    expression = _parse_expression(section[2])
    is_total_cost = (
        isinstance(expression, FluentRef)
        and expression.name == TOTAL_COST
        and direction == "minimize"
    )
    return (direction, expression), is_total_cost


def parse(text: str):
    """Parse PDDL text into a :class:`Domain` or :class:`Problem`."""
    tokens = tokenize(text)
    for section in tokens[1:]:
        key = _section_key(section)
        if key == "domain":
            return parse_domain(tokens)
        if key == "problem":
            return parse_problem(tokens)
    raise PDDLError("Could not determine whether input is a domain or a problem")


def parse_domain_file(path: str) -> Domain:
    with open(path, "r", encoding="utf-8") as handle:
        return parse_domain(tokenize(handle.read()))


def parse_problem_file(path: str) -> Problem:
    with open(path, "r", encoding="utf-8") as handle:
        return parse_problem(tokenize(handle.read()))
