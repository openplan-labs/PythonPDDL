"""Recursive-descent PDDL parser producing :mod:`jupyddl.parser.ast` objects.

Conditions are parsed straight into **negation normal form**: ``imply`` is
rewritten, ``not`` is pushed down through ``and``/``or``/quantifiers by De
Morgan, and what comes out only ever negates a literal. Quantifiers survive into
the AST because expanding them needs the object pool, which the grounder owns.

Which requirement flags are accepted, and what each one compiles to, is decided
by :mod:`jupyddl.requirements` — not by ad-hoc checks in here.
"""

from __future__ import annotations

import re

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
    TimedEffect,
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


def _parse_typed_list(items: list) -> list:
    """Parse ``?x ?y - type a b`` into ``[(name, type), ...]`` (default ``object``)."""
    if not isinstance(items, list):
        raise PDDLError(f"expected a parenthesised list, got '{items}'")
    result: list = []
    pending: list = []
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == "-":
            if i + 1 >= len(items):
                raise PDDLError(f"'-' with no type after it in {_show(items)}")
            typ = items[i + 1]
            if isinstance(typ, list):
                if typ and typ[0] == "either":
                    raise UnsupportedFeatureError(
                        "'either' types are not supported; declare a supertype instead"
                    )
                raise PDDLError(f"malformed type {_show(typ)} in {_show(items)}")
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


# A PDDL number. ``float()`` would also take ``nan``, ``inf`` and ``infinity``,
# which are perfectly good object names.
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?")


def _is_number(token) -> bool:
    return isinstance(token, str) and _NUMBER.fullmatch(token) is not None


def _show(expr) -> str:
    """Render a token tree back as PDDL, for error messages."""
    if isinstance(expr, list):
        return "(" + " ".join(_show(part) for part in expr) + ")"
    return str(expr)


def _expect_arity(expr: list, count: int, what: str) -> None:
    """Raise a PDDLError unless ``expr`` has exactly ``count`` operands."""
    if len(expr) != count + 1:
        plural = "" if count == 1 else "s"
        raise PDDLError(f"{what} takes {count} argument{plural}: {_show(expr)}")


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
    if isinstance(head, list):
        raise PDDLError(f"malformed numeric expression: {_show(expr)}")
    if head in ARITHMETIC:
        if len(expr) == 2 and head == "-":  # unary minus
            return Arithmetic("-", Number(0.0), _parse_expression(expr[1]))
        if len(expr) < 3:
            raise PDDLError(f"'{head}' needs two operands: {_show(expr)}")
        if len(expr) != 3:
            raise UnsupportedFeatureError(
                f"'{head}' takes exactly two arguments in a numeric expression"
            )
        return Arithmetic(head, _parse_expression(expr[1]), _parse_expression(expr[2]))
    if _is_number(head):
        if len(expr) != 1:
            raise PDDLError(f"malformed numeric expression: {_show(expr)}")
        return Number(float(head))
    for arg in expr[1:]:
        if isinstance(arg, list):
            raise UnsupportedFeatureError(
                f"nested term inside a fluent reference is not supported: {expr}"
            )
    return FluentRef(head, tuple(expr[1:]))


def _parse_atom(expr) -> Atom:
    if not isinstance(expr, list) or not expr or isinstance(expr[0], list):
        raise PDDLError(f"Malformed atom: {_show(expr)}")
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
    if isinstance(head, list):
        raise PDDLError(f"Malformed condition: {_show(expr)}")

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
    if head in ("forall", "exists"):
        _expect_arity(expr, 2, f"'{head}'")
        params = tuple(_parse_typed_list(expr[1]))
        body = parse_condition(expr[2], positive)
        universal = (head == "forall") == positive
        return Forall(params, body) if universal else Exists(params, body)
    if head == "preference":
        raise UnsupportedFeatureError(
            f"a preference is only supported in a goal, not here: {_show(expr)}"
        )
    if head == "=":
        _expect_arity(expr, 2, "'='")
        # `=` is equality between terms unless either side is numeric.
        left, right = expr[1], expr[2]
        if isinstance(left, list) or isinstance(right, list) or _is_number(left):
            comparison = Comparison(
                "=", _parse_expression(left), _parse_expression(right)
            )
            return comparison if positive else _negate_comparison(comparison)
        return EqualityConstraint(left, right, positive)
    if head in COMPARISONS:
        _expect_arity(expr, 2, f"'{head}'")
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
    if isinstance(head, list):
        raise PDDLError(f"Malformed effect: {_show(expr)}")
    if head == "and":
        return ConjunctiveEffect([parse_effect(sub) for sub in expr[1:]])
    if head == "not":
        _expect_arity(expr, 1, "'not'")
        return DelEffect(_parse_atom(expr[1]))
    if head == "forall":
        _expect_arity(expr, 2, "'forall'")
        return ForallEffect(_parse_typed_list(expr[1]), parse_effect(expr[2]))
    if head == "when":
        _expect_arity(expr, 2, "'when'")
        return WhenEffect(parse_condition(expr[1]), parse_effect(expr[2]))
    if head in ASSIGN_OPS:
        _expect_arity(expr, 2, f"'{head}'")
        target = expr[1]
        if isinstance(target, list) and (not target or isinstance(target[0], list)):
            raise PDDLError(f"malformed fluent in {_show(expr)}")
        name = target[0] if isinstance(target, list) else target
        value = _parse_expression(expr[2])
        if name == TOTAL_COST:
            if head != "increase":
                raise UnsupportedFeatureError(
                    f"'{head}' on (total-cost) is not supported; action costs "
                    "are written (increase (total-cost) k)"
                )
            # The classical action-cost shorthand: a constant stays a number;
            # an expression must read only static functions, and the grounder
            # evaluates it per action instance (see IncreaseCostEffect).
            if isinstance(value, Number):
                return IncreaseCostEffect(value.value)
            return IncreaseCostEffect(value)
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
        if not isinstance(flag, str):
            raise PDDLError(f"malformed requirement flag {_show(flag)}")
        if lookup_requirement(flag) is None:
            raise UnsupportedFeatureError(
                f"unknown requirement flag '{flag}'; see jupyddl.requirements "
                "for the full list of recognised flags"
            )
    # The flags as written first, then what they imply, in a fixed order: the
    # error should name the flag the author wrote, whatever the hash seed.
    declared = [lookup_requirement(flag).name for flag in requirements]
    implied = sorted(expand_requirements(requirements) - set(declared))
    for flag in declared + implied:
        reason = unsupported_reason(flag)
        if reason:
            raise UnsupportedFeatureError(reason)


# Sections a domain may contain, and ones recognised only to be refused by name.
DOMAIN_SECTIONS = {
    "domain",
    ":requirements",
    ":types",
    ":constants",
    ":predicates",
    ":functions",
    ":action",
    ":durative-action",
    ":derived",
    ":constraints",
}
REFUSED_DOMAIN_SECTIONS = {
    ":axiom": "PDDL 1.2 (:axiom ...) blocks are not supported; write the rule "
    "as (:derived ...) instead",
    ":safety": "PDDL 1.2 (:safety ...) constraints are not supported; "
    "(:constraints (always ...)) expresses the same thing",
    ":timeless": "PDDL 1.2 (:timeless ...) facts are not supported; put them in "
    "each problem's :init instead",
    ":process": "PDDL+ processes are not supported",
    ":event": "PDDL+ events are not supported",
}
PROBLEM_SECTIONS = {
    "problem",
    ":domain",
    ":requirements",
    ":objects",
    ":init",
    ":goal",
    ":constraints",
    ":metric",
    ":length",  # a PDDL 1.2 hint about plan length; it constrains nothing
}


def _describe(section) -> str:
    """``(:action move ...)`` for an error message about that section."""
    if not isinstance(section, list) or not section:
        return _show(section)
    head = section[0]
    if len(section) > 1 and isinstance(section[1], str):
        return f"({head} {section[1]} ...)"
    return f"({head} ...)"


def _check_section(section, allowed: set, refused: dict, kind: str) -> str:
    key = _section_key(section)
    if key in allowed:
        return key
    if key in refused:
        raise UnsupportedFeatureError(refused[key])
    raise PDDLError(
        f"unknown {kind} section {_describe(section)}; expected one of "
        + ", ".join(sorted(allowed))
    )


def _name_of(section, what: str) -> str:
    # Lenient about trailing tokens: `(problem switch a-b-c)` is in the wild.
    if len(section) < 2 or not isinstance(section[1], str):
        raise PDDLError(f"malformed ({section[0]} ...): expected one {what} name")
    return section[1]


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
        key = _check_section(
            section, DOMAIN_SECTIONS, REFUSED_DOMAIN_SECTIONS, "domain"
        )
        try:
            if key == "domain":
                name = _name_of(section, "domain")
            elif key == ":requirements":
                requirements = list(section[1:])
                _check_requirements(requirements)
            elif key == ":types":
                for child, parent in _parse_typed_list(section[1:]):
                    # In a :types list, "child - parent" reads name=child,
                    # type=parent.
                    types[child] = parent
            elif key == ":constants":
                constants = _parse_typed_list(section[1:])
            elif key == ":predicates":
                for pred in section[1:]:
                    if not isinstance(pred, list) or not pred:
                        raise PDDLError(
                            f"malformed predicate declaration '{_show(pred)}'"
                        )
                    _expect_name(pred[0], "predicate")
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
                constraints.extend(_parse_constraints(_body_of(section)))
        except IndexError as exc:  # a list shorter than its keyword promised
            raise PDDLError(f"malformed {_describe(section)}") from exc
    if not name:
        raise PDDLError("the domain has no (domain <name>)")
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


def _expect_name(token, what: str) -> None:
    if not isinstance(token, str):
        raise PDDLError(f"expected a {what} name, got {_show(token)}")


def _body_of(section) -> list:
    """The single body of a ``(:constraints X)`` or ``(:goal X)`` section."""
    if len(section) != 2:
        raise PDDLError(
            f"({section[0]} ...) takes exactly one formula; "
            f"got {len(section) - 1}: {_show(section)}"
        )
    return section[1]


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
            if index + 1 >= len(items) or isinstance(items[index + 1], list):
                raise PDDLError("'-' in :functions must be followed by a type name")
            result_type = items[index + 1]
            index += 2
            for name, params in pending:
                if result_type == "number":
                    numeric.append(Function(name, params))
                else:
                    object_fluents.append(ObjectFluent(name, params, result_type))
            pending = []
            continue
        if not isinstance(entry, list) or not entry:
            raise PDDLError(f"malformed function declaration '{_show(entry)}'")
        _expect_name(entry[0], "function")
        pending.append((entry[0], _parse_typed_list(entry[1:])))
        index += 1
    # Anything left untyped defaults to a number, as PDDL specifies.
    numeric.extend(Function(name, params) for name, params in pending)
    return numeric, object_fluents


def _keyword_pairs(section: list, allowed: tuple, what: str):
    """Yield ``(keyword, value)`` from ``(:action name :k1 v1 :k2 v2 ...)``.

    A misspelt keyword is an error rather than a silently empty precondition:
    ``:preconditon`` used to give an action that was always applicable.
    """
    if len(section) < 2 or not isinstance(section[1], str):
        raise PDDLError(f"({section[0]} ...) needs a name")
    name = section[1]
    rest = section[2:]
    for i in range(0, len(rest), 2):
        tag = rest[i]
        if tag not in allowed:
            raise PDDLError(
                f"unknown key {_show(tag)} in {what} '{name}'; expected "
                + ", ".join(allowed)
            )
        if i + 1 >= len(rest):
            raise PDDLError(f"{tag} in {what} '{name}' has no value")
        yield tag, rest[i + 1]


def _parse_action(section: list) -> Action:
    name = section[1] if len(section) > 1 else ""
    parameters: list = []
    precondition = Truth(True)
    effect = ConjunctiveEffect([])
    keys = (":parameters", ":precondition", ":effect")
    for tag, val in _keyword_pairs(section, keys, "action"):
        if tag == ":parameters":
            parameters = _parse_typed_list(val)
        elif tag == ":precondition":
            precondition = parse_condition(val)
        else:
            effect = parse_effect(val)
    return Action(name, parameters, precondition, effect)


def _split_timed(expr, slots: dict, action_name: str, what: str) -> None:
    """File each timed part of a durative condition or effect under its marker.

    ``slots`` maps ``"at start"`` / ``"at end"`` / ``"over all"`` to the list
    its parts go in; a marker missing from it is not allowed here.
    """
    if not isinstance(expr, list):
        raise PDDLError(f"malformed {what} in durative action '{action_name}'")
    if not expr:
        return
    if expr[0] == "and":
        for sub in expr[1:]:
            _split_timed(sub, slots, action_name, what)
        return
    marker = None
    if len(expr) == 3 and all(isinstance(part, str) for part in expr[:2]):
        marker = f"{expr[0]} {expr[1]}"
    if marker in slots:
        slots[marker].append(expr[2])
        return
    if marker in ("at start", "at end", "over all"):
        raise UnsupportedFeatureError(
            f"'{marker}' is not allowed on an {what} of '{action_name}'"
        )
    if expr[0] in ("when", "forall") and what == "effect":
        raise UnsupportedFeatureError(
            f"a '{expr[0]}' around time specifiers in durative action "
            f"'{action_name}' is not supported; put it inside one instead, "
            f"e.g. (at end ({expr[0]} ...))"
        )
    raise PDDLError(
        f"the {what} {_show(expr)} of durative action '{action_name}' needs a "
        "time specifier: " + ", ".join(f"({m} ...)" for m in slots)
    )


def _parse_durative_action(section: list) -> Action:
    """Compile ``(:durative-action ...)`` into a single sequential action.

    At-start conditions become the precondition; over-all and at-end
    conditions become ``late_condition``, which the grounder checks in the state
    *after* the at-start effects. The effects stay split in a
    :class:`TimedEffect` so the at-end ones can override the at-start ones.
    """
    name = section[1] if len(section) > 1 else ""
    parameters: list = []
    duration = None
    early: list = []
    late: list = []
    start: list = []
    end: list = []

    keys = (":parameters", ":duration", ":condition", ":effect")
    for tag, val in _keyword_pairs(section, keys, "durative action"):
        if tag == ":parameters":
            parameters = _parse_typed_list(val)
        elif tag == ":duration":
            duration = _parse_duration(val, name)
        elif tag == ":condition":
            slots = {"at start": early, "over all": late, "at end": late}
            _split_timed(val, slots, name, "condition")
        else:
            _split_timed(val, {"at start": start, "at end": end}, name, "effect")

    if duration is None:
        raise PDDLError(f"durative action '{name}' has no :duration")

    return Action(
        name,
        parameters,
        parse_condition(["and"] + early),
        TimedEffect(parse_effect(["and"] + start), parse_effect(["and"] + end)),
        duration=duration,
        late_condition=parse_condition(["and"] + late),
    )


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
    if len(section) != 3:
        raise PDDLError(f"(:derived ...) takes a head and a body: {_show(section)}")
    head_expr = section[1]
    if not isinstance(head_expr, list) or not head_expr:
        raise PDDLError("malformed (:derived ...) head")
    _expect_name(head_expr[0], "derived predicate")
    params = _parse_typed_list(head_expr[1:])
    head = Atom(head_expr[0], tuple(var for var, _ in params))
    body = parse_condition(section[2])
    return DerivedPredicate(head, params, body)


def _parse_constraints(expr, params: tuple = ()) -> list:
    """Parse a ``(:constraints ...)`` body into :class:`Constraint` nodes.

    ``params`` accumulates the variables of enclosing ``forall`` blocks. They
    stay on the constraint rather than moving inside its operands, because the
    modal operators do not distribute over a conjunction:
    ``(forall (?x) (sometime (p ?x)))`` says every ``?x`` gets its own moment,
    while ``(sometime (forall (?x) (p ?x)))`` needs one moment for all of them.
    The compiler expands one constraint per binding once the objects are known.
    """
    out: list = []

    def visit(node, params):
        if not isinstance(node, list):
            raise PDDLError(f"malformed constraint: {_show(node)}")
        if not node:
            return
        head = node[0]
        if not isinstance(head, str):
            raise PDDLError(f"malformed constraint: {_show(node)}")
        if head == "and":
            for sub in node[1:]:
                visit(sub, params)
            return
        if head == "forall":
            _expect_arity(node, 2, "'forall'")
            visit(node[2], params + tuple(_parse_typed_list(node[1])))
            return
        if head == "preference":
            if len(node) == 3:
                name, body = node[1], node[2]
            elif len(node) == 2:
                name, body = "(unnamed)", node[1]
            else:
                raise PDDLError(f"malformed preference: {_show(node)}")
            for constraint in _parse_constraints(body, params):
                out.append(Preference(str(name), constraint))
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
        if len(node) != arity + 1:
            raise PDDLError(f"'{head}' takes {arity} operand(s): {_show(node)}")
        operands = tuple(parse_condition(o) for o in node[1:])
        out.append(Constraint(head, operands, params))

    visit(expr, params)
    return out


def _split_preferences(expr, found: list, params: tuple = ()):
    """Strip ``(preference name phi)`` out of a goal, leaving the hard part.

    Returns the goal s-expression with every preference replaced by ``()``
    (trivially true) and appends ``(name, phi, params)`` to ``found``, where
    ``params`` are the variables of the ``forall`` blocks around it.

    PDDL 3 only lets a preference sit under ``and`` and ``forall``. Anywhere
    else it has no meaning — ``(not (preference p phi))`` used to become
    ``(not ())``, a goal that is simply false.
    """
    if not isinstance(expr, list) or not expr:
        return expr
    head = expr[0]
    if head == "preference":
        if len(expr) == 3:
            found.append((str(expr[1]), expr[2], params))
        elif len(expr) == 2:  # an unnamed preference
            found.append((f"__pref{len(found) + 1}", expr[1], params))
        else:
            raise PDDLError(f"malformed preference: {_show(expr)}")
        return []
    if head == "and":
        return [head] + [_split_preferences(sub, found, params) for sub in expr[1:]]
    if head == "forall":
        _expect_arity(expr, 2, "'forall'")
        inner = params + tuple(_parse_typed_list(expr[1]))
        return [head, expr[1], _split_preferences(expr[2], found, inner)]
    if _mentions_preference(expr):
        raise PDDLError(
            f"a preference may only appear under 'and' or 'forall' in a goal, "
            f"not under '{head}': {_show(expr)}"
        )
    return expr


def _mentions_preference(expr) -> bool:
    if not isinstance(expr, list):
        return False
    if expr and expr[0] == "preference":
        return True
    return any(_mentions_preference(sub) for sub in expr)


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
        raise PDDLError(f"malformed timed initial literal: {_show(fact)}")
    if body[0] == "not":
        _expect_arity(body, 1, "'not'")
        return TimedInitial(float(fact[1]), Literal(_parse_atom(body[1]), False))
    return TimedInitial(float(fact[1]), Literal(_parse_atom(body), True))


def _constant(expression):
    """The value of an expression built from numbers alone, else None."""
    if isinstance(expression, Number):
        return float(expression.value)
    if isinstance(expression, Arithmetic):
        left, right = _constant(expression.left), _constant(expression.right)
        if left is None or right is None:
            return None
        if expression.op == "+":
            return left + right
        if expression.op == "-":
            return left - right
        if expression.op == "*":
            return left * right
        if right != 0:
            return left / right
    return None


def _metric_terms(expression, weights: dict, bases: list, factor: float = 1.0):
    """Check a metric is ``base + sum of w * (is-violated p) + constant``.

    ``base`` is ``(total-cost)`` or ``(total-time)`` (or absent); the
    ``(is-violated p)`` coefficients land in ``weights``. Anything else — a
    numeric fluent, a product of two variables, a scaled ``(total-cost)`` — is
    refused, because the compilation prices actions and preferences, not
    arbitrary expressions, and would otherwise optimise something else.
    """
    if _constant(expression) is not None:
        return  # an offset changes no plan's ranking
    if isinstance(expression, FluentRef):
        if expression.name == "is-violated":
            if len(expression.args) != 1:
                raise PDDLError(f"(is-violated ...) names one preference: {expression}")
            name = expression.args[0]
            weights[name] = weights.get(name, 0.0) + factor
            return
        if expression.name in (TOTAL_COST, "total-time") and not expression.args:
            if factor != 1 or expression.name in bases:
                raise UnsupportedFeatureError(
                    f"the metric scales ({expression.name}); only a plain "
                    f"({expression.name}) term is supported"
                )
            bases.append(expression.name)
            return
        raise UnsupportedFeatureError(
            f"the metric reads {expression}; only (total-cost), (total-time), "
            "constant multiples of (is-violated p) and constants are supported"
        )
    if isinstance(expression, Arithmetic):
        left, right = expression.left, expression.right
        if expression.op == "+":
            _metric_terms(left, weights, bases, factor)
            _metric_terms(right, weights, bases, factor)
            return
        if expression.op == "-":
            _metric_terms(left, weights, bases, factor)
            _metric_terms(right, weights, bases, -factor)
            return
        if expression.op == "*":
            if _constant(left) is not None:
                _metric_terms(right, weights, bases, factor * _constant(left))
                return
            if _constant(right) is not None:
                _metric_terms(left, weights, bases, factor * _constant(right))
                return
        if expression.op == "/" and _constant(right):
            _metric_terms(left, weights, bases, factor / _constant(right))
            return
    raise UnsupportedFeatureError(
        "the metric is not a sum of (total-cost), (total-time), constant "
        "multiples of (is-violated p) and constants"
    )


def _init_value(target, value):
    """Read ``(= <fluent> <value>)`` from ``:init``: a number or an object."""
    fluent = _parse_expression(target)
    if not isinstance(fluent, FluentRef):
        raise PDDLError(f"(= {_show(target)} ...) in :init does not name a fluent")
    if isinstance(value, list):
        number = _constant(_parse_expression(value))
        if number is None:
            raise PDDLError(
                f"the value of {fluent} in :init must be a number or an object, "
                f"not {_show(value)}"
            )
        return fluent, number
    if _is_number(value):
        return fluent, float(value)
    # `(= (location p1) depot)` assigns an object, not a number: an object
    # fluent, compiled away later.
    return fluent, value


def parse_problem(tokens: list) -> Problem:
    if _section_key(tokens) != "define":
        raise PDDLError("Problem file must start with (define ...)")

    name = ""
    domain_name = ""
    objects: list = []
    init: list = []
    init_numeric: dict = {}
    goal = None
    metric = None
    metric_minimize_cost = False
    preferences: list = []
    constraints: list = []
    timed_initials: list = []
    init_objects: dict = {}
    violation_weights: dict = {}

    for section in tokens[1:]:
        key = _check_section(section, PROBLEM_SECTIONS, {}, "problem")
        try:
            if key == "problem":
                name = _name_of(section, "problem")
            elif key == ":domain":
                domain_name = _name_of(section, "domain")
            elif key == ":objects":
                objects = _parse_typed_list(section[1:])
            elif key == ":requirements":
                _check_requirements(list(section[1:]))
            elif key == ":init":
                for fact in section[1:]:
                    if not isinstance(fact, list):
                        raise PDDLError(f"expected a fact in :init, got '{fact}'")
                    if not fact:
                        continue
                    if fact[0] == "=":
                        _expect_arity(fact, 2, "'=' in :init")
                        fluent, value = _init_value(fact[1], fact[2])
                        if isinstance(value, float):
                            init_numeric[fluent] = value
                        else:
                            init_objects[fluent] = value
                        continue
                    timed = _parse_timed_initial(fact)
                    if timed is not None:
                        timed_initials.append(timed)
                        continue
                    init.append(_parse_atom(fact))
            elif key == ":goal":
                stripped = _split_preferences(_body_of(section), preferences)
                goal = parse_condition(stripped)
            elif key == ":metric":
                metric, metric_minimize_cost = _parse_metric(section)
                violation_weights = {}
                _metric_terms(metric[1], violation_weights, [])
            elif key == ":constraints":
                constraints.extend(_parse_constraints(_body_of(section)))
        except IndexError as exc:  # a list shorter than its keyword promised
            raise PDDLError(f"malformed {_describe(section)}") from exc
    if goal is None:
        raise PDDLError(f"problem '{name}' has no :goal")
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
            Preference(pref_name, parse_condition(body), params)
            for pref_name, body, params in preferences
        ],
        constraints,
        timed_initials,
        init_objects,
        violation_weights,
    )


def _parse_metric(section):
    """Read ``(:metric minimize <expr>)``; returns ``((dir, expr), is_total_cost)``."""
    if len(section) != 3:
        raise PDDLError(f"malformed :metric: {_show(section)}")
    direction = section[1]
    if direction == "maximize":
        raise UnsupportedFeatureError(
            "'maximize' metrics are not supported: every planner here "
            "minimises cost, so it would silently optimise the opposite"
        )
    if direction != "minimize":
        raise PDDLError(f"unknown metric direction '{direction}'")
    expression = _parse_expression(section[2])
    is_total_cost = isinstance(expression, FluentRef) and expression.name == TOTAL_COST
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
