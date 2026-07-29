"""Grounding: turn a parsed :class:`Domain` + :class:`Problem` into a
grounded :class:`~jupyddl.task.Task`.

Pipeline:

1. Build the ``type -> objects`` table (with type-hierarchy closure).
2. Instantiate every action over all type-consistent parameter tuples. The
   precondition formula is expanded (quantifiers over the object pool) and
   distributed into DNF; **each disjunct becomes its own grounded operator**, so
   the search only ever sees conjunctive preconditions.
3. Detect static predicates (never added, deleted, or derived) and use the
   initial state to prune infeasible action instances and simplify conditions.
4. Compile negative preconditions/goals into *positive normal form* by
   introducing complement facts ``(not ...)`` and maintaining them on every
   operator that touches the underlying atom.
5. Ground derived-predicate rules into :class:`~jupyddl.task.Axiom` objects.
6. Collect numeric fluents, compile their expressions into closures over the
   state's value vector, and encode everything as integer ids.

A disjunctive goal is compiled to a single artificial goal fact achieved by one
zero-cost operator per disjunct; those operators are recorded in
``Task.synthetic`` so they can be hidden when a plan is printed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product

from .parser.ast import (
    AddEffect,
    And,
    Arithmetic,
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
    Problem,
    Truth,
    UnsupportedFeatureError,
    WhenEffect,
)
from .compile import CLOCK as CLOCK_FLUENT
from .compile import SYNTHETIC_PREFIX, compile_problem
from .parser.parser import parse_domain_file, parse_problem_file
from .task import Axiom, CondEffect, Operator, Task

# Distributing a deeply disjunctive precondition into DNF can blow up
# combinatorially. Stop with a clear message rather than exhausting memory.
MAX_DISJUNCTS = 20000

GOAL_FACT = Atom("__goal-reached__", ())


def _ground_atom(atom: Atom, subst: dict) -> Atom:
    return Atom(atom.predicate, tuple(subst.get(a, a) for a in atom.args))


def _ground_fluent(ref: FluentRef, subst: dict) -> FluentRef:
    return FluentRef(ref.name, tuple(subst.get(a, a) for a in ref.args))


def _ground_expression(expr, subst: dict):
    if isinstance(expr, Number):
        return expr
    if isinstance(expr, FluentRef):
        return _ground_fluent(expr, subst)
    if isinstance(expr, Arithmetic):
        return Arithmetic(
            expr.op,
            _ground_expression(expr.left, subst),
            _ground_expression(expr.right, subst),
        )
    raise PDDLError(f"unexpected numeric expression node: {expr!r}")


def _build_type_objects(domain: Domain, problem: Problem):
    """Map every type to the set of objects that inhabit it (hierarchy-aware)."""
    parent = dict(domain.types)

    def ancestors(typ: str):
        chain = [typ]
        seen = {typ}
        cur = typ
        while cur in parent and parent[cur] not in (None, "object", cur):
            cur = parent[cur]
            if cur in seen:
                break
            seen.add(cur)
            chain.append(cur)
        return chain

    type_objects: dict = defaultdict(set)
    declared: set = set()
    for name, typ in list(domain.constants) + list(problem.objects):
        declared.add(name)
        for anc in ancestors(typ):
            type_objects[anc].add(name)
        type_objects["object"].add(name)

    # Robustness: some (toy) problems omit the :objects section and only mention
    # constants in :init/:goal. Treat any such undeclared constant as an object
    # of the root type so untyped domains still ground.
    for name in _harvest_constants(problem):
        if name not in declared:
            type_objects["object"].add(name)

    for typ in set(list(parent) + list(parent.values())):
        type_objects.setdefault(typ, set())
    return {typ: tuple(sorted(objs)) for typ, objs in type_objects.items()}


def _harvest_constants(problem: Problem) -> set:
    found: set = set()
    for atom in problem.init:
        found.update(arg for arg in atom.args if not arg.startswith("?"))
    for ref in problem.init_numeric:
        found.update(arg for arg in ref.args if not arg.startswith("?"))

    def walk(formula):
        if isinstance(formula, Literal):
            found.update(a for a in formula.atom.args if not a.startswith("?"))
        elif isinstance(formula, (And, Or)):
            for part in formula.parts:
                walk(part)
        elif isinstance(formula, (Exists, Forall)):
            walk(formula.body)

    walk(problem.goal)
    return found


# --------------------------------------------------------------------------
# condition formulas -> DNF
# --------------------------------------------------------------------------
@dataclass
class _Disjunct:
    """One ground conjunction: positive atoms, negative atoms, comparisons."""

    pos: set = field(default_factory=set)
    neg: set = field(default_factory=set)
    comparisons: list = field(default_factory=list)

    def merged(self, other: "_Disjunct") -> "_Disjunct":
        return _Disjunct(
            self.pos | other.pos,
            self.neg | other.neg,
            self.comparisons + other.comparisons,
        )

    @property
    def contradictory(self) -> bool:
        return bool(self.pos & self.neg)


TRUE_DNF = [_Disjunct()]
FALSE_DNF: list = []


def _dnf(formula, subst: dict, type_objects: dict) -> list:
    """Expand quantifiers and distribute ``formula`` into a list of disjuncts.

    An empty list means *unsatisfiable*; a list holding one empty disjunct means
    *trivially true*.
    """
    if formula is None:
        return TRUE_DNF

    if isinstance(formula, Truth):
        return TRUE_DNF if formula.value else FALSE_DNF

    if isinstance(formula, Literal):
        atom = _ground_atom(formula.atom, subst)
        if formula.positive:
            return [_Disjunct(pos={atom})]
        return [_Disjunct(neg={atom})]

    if isinstance(formula, EqualityConstraint):
        left = subst.get(formula.left, formula.left)
        right = subst.get(formula.right, formula.right)
        holds = (left == right) == formula.positive
        return TRUE_DNF if holds else FALSE_DNF

    if isinstance(formula, Comparison):
        grounded = Comparison(
            formula.op,
            _ground_expression(formula.left, subst),
            _ground_expression(formula.right, subst),
        )
        return [_Disjunct(comparisons=[grounded])]

    if isinstance(formula, And):
        result = TRUE_DNF
        for part in formula.parts:
            result = _cross(result, _dnf(part, subst, type_objects))
            if not result:
                return FALSE_DNF
        return result

    if isinstance(formula, Or):
        out: list = []
        for part in formula.parts:
            out.extend(_dnf(part, subst, type_objects))
            if len(out) > MAX_DISJUNCTS:
                raise UnsupportedFeatureError(
                    "disjunctive condition expands past "
                    f"{MAX_DISJUNCTS} cases; simplify the domain or split the action"
                )
        return out

    if isinstance(formula, Forall):
        result = TRUE_DNF
        for sub in _quantifier_substitutions(formula.params, subst, type_objects):
            result = _cross(result, _dnf(formula.body, sub, type_objects))
            if not result:
                return FALSE_DNF
        return result

    if isinstance(formula, Exists):
        out = []
        for sub in _quantifier_substitutions(formula.params, subst, type_objects):
            out.extend(_dnf(formula.body, sub, type_objects))
            if len(out) > MAX_DISJUNCTS:
                raise UnsupportedFeatureError(
                    "existential condition expands past "
                    f"{MAX_DISJUNCTS} cases; the object pool is too large"
                )
        return out

    raise PDDLError(f"unexpected condition node: {formula!r}")


def _quantifier_substitutions(params, subst: dict, type_objects: dict):
    pools = [type_objects.get(typ, ()) for (_, typ) in params]
    for combo in product(*pools):
        extended = dict(subst)
        for (var, _), obj in zip(params, combo):
            extended[var] = obj
        yield extended


def _cross(left: list, right: list) -> list:
    """Distribute a conjunction of two DNFs, dropping contradictory disjuncts."""
    if not left or not right:
        return FALSE_DNF
    out = []
    for a in left:
        for b in right:
            merged = a.merged(b)
            if not merged.contradictory:
                out.append(merged)
    if len(out) > MAX_DISJUNCTS:
        raise UnsupportedFeatureError(
            f"conjunction of disjunctions expands past {MAX_DISJUNCTS} cases"
        )
    return out


# --------------------------------------------------------------------------
# effects
# --------------------------------------------------------------------------
@dataclass
class _RawOp:
    name: str
    pre_pos: set
    pre_neg: set
    comparisons: list
    add: set
    delete: set
    cond: list  # (cpos, cneg, cadd, cdel)
    numeric: list  # (op, FluentRef, expression)
    cost: float
    duration: float
    synthetic: bool = False


def _collect_effect(eff, subst, type_objects, cpos, cneg, acc):
    if isinstance(eff, ConjunctiveEffect):
        for part in eff.parts:
            _collect_effect(part, subst, type_objects, cpos, cneg, acc)
    elif isinstance(eff, AddEffect):
        atom = _ground_atom(eff.atom, subst)
        if cpos or cneg:
            acc["cond"].append((frozenset(cpos), frozenset(cneg), {atom}, set()))
        else:
            acc["add"].add(atom)
    elif isinstance(eff, DelEffect):
        atom = _ground_atom(eff.atom, subst)
        if cpos or cneg:
            acc["cond"].append((frozenset(cpos), frozenset(cneg), set(), {atom}))
        else:
            acc["delete"].add(atom)
    elif isinstance(eff, IncreaseCostEffect):
        acc["cost"] += eff.amount
        acc["has_cost"] = True
    elif isinstance(eff, NumericEffect):
        if cpos or cneg:
            raise UnsupportedFeatureError(
                "numeric effects inside a 'when' are not supported"
            )
        acc["numeric"].append(
            (
                eff.op,
                _ground_fluent(eff.target, subst),
                _ground_expression(eff.value, subst),
            )
        )
    elif isinstance(eff, ForallEffect):
        for sub in _quantifier_substitutions(eff.params, subst, type_objects):
            _collect_effect(eff.body, sub, type_objects, cpos, cneg, acc)
    elif isinstance(eff, WhenEffect):
        # A disjunctive effect condition splits into one conditional effect per
        # disjunct, which is exactly equivalent.
        for disjunct in _dnf(eff.condition, subst, type_objects):
            if disjunct.comparisons:
                raise UnsupportedFeatureError(
                    "numeric comparisons inside a 'when' condition are not supported"
                )
            _collect_effect(
                eff.body,
                subst,
                type_objects,
                cpos | disjunct.pos,
                cneg | disjunct.neg,
                acc,
            )
    else:
        raise PDDLError(f"unexpected effect node: {eff!r}")


def _effect_predicates(domain: Domain) -> set:
    preds: set = set()

    def walk(eff):
        if isinstance(eff, ConjunctiveEffect):
            for part in eff.parts:
                walk(part)
        elif isinstance(eff, (AddEffect, DelEffect)):
            preds.add(eff.atom.predicate)
        elif isinstance(eff, (ForallEffect, WhenEffect)):
            walk(eff.body)

    for action in domain.actions:
        walk(action.effect)
    return preds


def _ground_raw_operators(domain, problem, type_objects) -> list:
    raw = []
    for action in domain.actions:
        pools = [type_objects.get(typ, ()) for (_, typ) in action.parameters]
        for combo in product(*pools):
            subst = {var: obj for (var, _), obj in zip(action.parameters, combo)}
            disjuncts = _dnf(action.precondition, subst, type_objects)
            if not disjuncts:
                continue  # precondition is unsatisfiable for this instance

            acc = {
                "add": set(),
                "delete": set(),
                "cond": [],
                "numeric": [],
                "cost": 0.0,
                "has_cost": False,
            }
            _collect_effect(action.effect, subst, type_objects, set(), set(), acc)

            args = ",".join(combo)
            base = f"{action.name}({args})" if combo else action.name
            duration = 0.0
            if action.duration is not None:
                duration = _constant_value(action.duration, subst, action.name)
            if acc["has_cost"]:
                cost = acc["cost"]
            elif duration:
                # A temporal action with no explicit cost: optimise makespan.
                cost = duration
            else:
                cost = 1

            for index, disjunct in enumerate(disjuncts):
                # Only tag the name when the split is real, so classical domains
                # keep the operator names their users expect.
                name = base if len(disjuncts) == 1 else f"{base}#{index + 1}"
                raw.append(
                    _RawOp(
                        name=name,
                        pre_pos=set(disjunct.pos),
                        pre_neg=set(disjunct.neg),
                        comparisons=list(disjunct.comparisons),
                        add=set(acc["add"]),
                        delete=set(acc["delete"]),
                        cond=list(acc["cond"]),
                        numeric=list(acc["numeric"]),
                        cost=cost,
                        duration=duration,
                    )
                )
    return raw


def _constant_value(expr, subst, action_name):
    """Evaluate a duration expression that must be constant."""
    grounded = _ground_expression(expr, subst)
    if isinstance(grounded, Number):
        return float(grounded.value)
    raise UnsupportedFeatureError(
        f"the duration of '{action_name}' must be a constant; "
        "durations that read numeric fluents are not supported"
    )


# --------------------------------------------------------------------------
# numeric compilation
# --------------------------------------------------------------------------
def _collect_fluents(expr, out: set) -> None:
    if isinstance(expr, FluentRef):
        out.add(expr)
    elif isinstance(expr, Arithmetic):
        _collect_fluents(expr.left, out)
        _collect_fluents(expr.right, out)


def _compile_expression(expr, index_of: dict):
    """Compile a ground numeric expression into a ``values -> float`` closure."""
    if isinstance(expr, Number):
        constant = float(expr.value)
        return lambda values: constant
    if isinstance(expr, FluentRef):
        index = index_of[expr]
        return lambda values: values[index]
    if isinstance(expr, Arithmetic):
        left = _compile_expression(expr.left, index_of)
        right = _compile_expression(expr.right, index_of)
        if expr.op == "+":
            return lambda values: left(values) + right(values)
        if expr.op == "-":
            return lambda values: left(values) - right(values)
        if expr.op == "*":
            return lambda values: left(values) * right(values)
        if expr.op == "/":

            def divide(values):
                denominator = right(values)
                if denominator == 0:
                    # Undefined rather than crashing mid-search: an infinite
                    # value makes every comparison against it fail.
                    return float("inf")
                return left(values) / denominator

            return divide
        raise PDDLError(f"unknown arithmetic operator '{expr.op}'")
    raise PDDLError(f"unexpected numeric expression: {expr!r}")


def _compile_comparison(comparison: Comparison, index_of: dict):
    left = _compile_expression(comparison.left, index_of)
    right = _compile_expression(comparison.right, index_of)
    op = comparison.op
    if op == "<":
        return lambda values: left(values) < right(values)
    if op == "<=":
        return lambda values: left(values) <= right(values)
    if op == ">":
        return lambda values: left(values) > right(values)
    if op == ">=":
        return lambda values: left(values) >= right(values)
    if op == "=":
        return lambda values: left(values) == right(values)
    if op == "!=":
        return lambda values: left(values) != right(values)
    raise PDDLError(f"unknown comparison operator '{op}'")


def _compile_numeric_effect(op: str, index: int, value, index_of: dict):
    compute = _compile_expression(value, index_of)
    if op == "assign":
        return (index, compute)
    if op == "increase":
        return (index, lambda values: values[index] + compute(values))
    if op == "decrease":
        return (index, lambda values: values[index] - compute(values))
    if op == "scale-up":
        return (index, lambda values: values[index] * compute(values))
    if op == "scale-down":

        def scale_down(values):
            divisor = compute(values)
            return float("inf") if divisor == 0 else values[index] / divisor

        return (index, scale_down)
    raise PDDLError(f"unknown numeric assignment '{op}'")


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------
@dataclass
class _Encoder:
    fact_ids: dict = field(default_factory=dict)
    comp_ids: dict = field(default_factory=dict)
    names: list = field(default_factory=list)

    def fact(self, atom: Atom) -> int:
        if atom not in self.fact_ids:
            self.fact_ids[atom] = len(self.names)
            self.names.append(str(atom))
        return self.fact_ids[atom]

    def comp(self, atom: Atom) -> int:
        if atom not in self.comp_ids:
            self.comp_ids[atom] = len(self.names)
            self.names.append(f"(not {atom})")
        return self.comp_ids[atom]


def _ground_axioms(domain: Domain, type_objects: dict) -> list:
    """Ground every derived-predicate rule into (head atom, disjunct) pairs."""
    grounded = []
    for rule in domain.derived:
        for subst in _quantifier_substitutions(rule.params, {}, type_objects):
            head = _ground_atom(rule.head, subst)
            for disjunct in _dnf(rule.body, subst, type_objects):
                if disjunct.comparisons:
                    raise UnsupportedFeatureError(
                        "numeric comparisons in a derived predicate are not supported"
                    )
                grounded.append((head, disjunct))
    return grounded


def ground(domain: Domain, problem: Problem) -> Task:
    """Ground ``domain`` + ``problem`` into a :class:`Task`."""
    # PDDL 3 constructs (preferences, trajectory constraints, timed literals,
    # object fluents) are rewritten into the classical core first, so nothing
    # below this line has to know they exist.
    requirements = tuple(domain.requirements)
    domain, problem = compile_problem(domain, problem)
    type_objects = _build_type_objects(domain, problem)
    init_atoms = set(problem.init)

    derived_preds = {rule.head.predicate for rule in domain.derived}
    all_preds = {p.name for p in domain.predicates} | derived_preds
    # A derived predicate never appears in an effect, but it is emphatically not
    # static -- the axioms compute it.
    static_preds = all_preds - _effect_predicates(domain) - derived_preds

    def is_static(atom: Atom) -> bool:
        return atom.predicate in static_preds

    raw_ops = _ground_raw_operators(domain, problem, type_objects)
    axiom_rules = _ground_axioms(domain, type_objects)

    enc = _Encoder()
    tracked_neg: set = set()

    def resolve_literals(pos_atoms, neg_atoms):
        """Drop static literals that hold, fail on ones that do not."""
        pos_fluent, neg_fluent = set(), set()
        for atom in pos_atoms:
            if is_static(atom):
                if atom not in init_atoms:
                    return None
            else:
                pos_fluent.add(atom)
        for atom in neg_atoms:
            if is_static(atom):
                if atom in init_atoms:
                    return None
            else:
                neg_fluent.add(atom)
                tracked_neg.add(atom)
        return pos_fluent, neg_fluent

    # --- resolve static literals, drop infeasible operators ------------------
    resolved = []
    for op in raw_ops:
        pre = resolve_literals(op.pre_pos, op.pre_neg)
        if pre is None:
            continue
        pre_pos_fluent, pre_neg_fluent = pre

        add = set(op.add)
        delete = set(op.delete)
        cond = []
        for cpos, cneg, cadd, cdel in op.cond:
            trigger = resolve_literals(cpos, cneg)
            if trigger is None:
                continue  # this conditional effect can never fire
            cpos_f, cneg_f = trigger
            if not cpos_f and not cneg_f:
                add |= cadd
                delete |= cdel
            else:
                cond.append((cpos_f, cneg_f, cadd, cdel))
        resolved.append((op, pre_pos_fluent, pre_neg_fluent, add, delete, cond))

    # --- goal ----------------------------------------------------------------
    goal_disjuncts = _dnf(problem.goal, {}, type_objects)
    if not goal_disjuncts:
        raise ValueError("Goal condition is self-contradictory")

    resolved_goals = []
    unsolvable = True
    for disjunct in goal_disjuncts:
        parts = resolve_literals(disjunct.pos, disjunct.neg)
        if parts is None:
            continue  # this way of satisfying the goal is statically impossible
        unsolvable = False
        resolved_goals.append((parts[0], parts[1], disjunct.comparisons))
    if not resolved_goals:
        resolved_goals = [(set(), set(), [])]

    # --- axioms --------------------------------------------------------------
    resolved_axioms = []
    for head, disjunct in axiom_rules:
        parts = resolve_literals(disjunct.pos, disjunct.neg)
        if parts is None:
            continue
        resolved_axioms.append((head, parts[0], parts[1]))

    # --- numeric fluents -----------------------------------------------------
    fluents: set = set(problem.init_numeric)
    for op, *_ in resolved:
        for comparison in op.comparisons:
            _collect_fluents(comparison.left, fluents)
            _collect_fluents(comparison.right, fluents)
        for _, target, value in op.numeric:
            fluents.add(target)
            _collect_fluents(value, fluents)
    for _, _, comparisons in resolved_goals:
        for comparison in comparisons:
            _collect_fluents(comparison.left, fluents)
            _collect_fluents(comparison.right, fluents)
    # `total-cost` is bookkeeping handled by operator costs, not a state variable.
    fluents = {ref for ref in fluents if ref.name != "total-cost"}

    ordered_fluents = sorted(fluents, key=str)
    index_of = {ref: i for i, ref in enumerate(ordered_fluents)}
    numeric_names = tuple(str(ref) for ref in ordered_fluents)
    init_values = tuple(
        float(problem.init_numeric.get(ref, 0.0)) for ref in ordered_fluents
    )

    # --- encode facts --------------------------------------------------------
    init_ids = {enc.fact(a) for a in init_atoms if not is_static(a)}
    for atom in tracked_neg:
        cid = enc.comp(atom)
        if atom not in init_atoms:
            init_ids.add(cid)

    def encode_add_del(add_atoms, del_atoms):
        add_ids = {enc.fact(a) for a in add_atoms}
        del_ids = {enc.fact(a) for a in del_atoms}
        for a in add_atoms:
            if a in tracked_neg:
                del_ids.add(enc.comp(a))
        for a in del_atoms:
            if a in tracked_neg:
                add_ids.add(enc.comp(a))
        return frozenset(add_ids), frozenset(del_ids)

    operators = []
    for op, pp, pn, add, delete, cond in resolved:
        precond = {enc.fact(a) for a in pp} | {enc.comp(a) for a in pn}
        add_ids, del_ids = encode_add_del(add, delete)
        cond_effects = []
        for cpos_f, cneg_f, cadd, cdel in cond:
            cond_ids = {enc.fact(a) for a in cpos_f} | {enc.comp(a) for a in cneg_f}
            cadd_ids, cdel_ids = encode_add_del(cadd, cdel)
            if cadd_ids or cdel_ids:
                cond_effects.append(CondEffect(frozenset(cond_ids), cadd_ids, cdel_ids))
        numeric_pre = tuple(_compile_comparison(c, index_of) for c in op.comparisons)
        numeric_eff = tuple(
            _compile_numeric_effect(kind, index_of[target], value, index_of)
            for kind, target, value in op.numeric
        )
        operators.append(
            Operator(
                op.name,
                frozenset(precond),
                add_ids,
                del_ids,
                tuple(cond_effects),
                op.cost,
                numeric_pre,
                numeric_eff,
                op.duration,
            )
        )

    # --- goal encoding, compiling a disjunction into a synthetic fact --------
    synthetic: set = set()
    goal_numeric: tuple = ()
    if len(resolved_goals) == 1:
        goal_pos, goal_neg, comparisons = resolved_goals[0]
        goals = {enc.fact(a) for a in goal_pos} | {enc.comp(a) for a in goal_neg}
        goal_numeric = tuple(_compile_comparison(c, index_of) for c in comparisons)
    else:
        goal_fact = enc.fact(GOAL_FACT)
        goals = {goal_fact}
        for index, (goal_pos, goal_neg, comparisons) in enumerate(resolved_goals):
            name = f"__reach-goal__#{index + 1}"
            synthetic.add(name)
            precond = {enc.fact(a) for a in goal_pos} | {enc.comp(a) for a in goal_neg}
            operators.append(
                Operator(
                    name,
                    frozenset(precond),
                    frozenset({goal_fact}),
                    frozenset(),
                    (),
                    0,
                    tuple(_compile_comparison(c, index_of) for c in comparisons),
                    (),
                    0.0,
                )
            )

    if unsolvable:
        sentinel = len(enc.names)
        enc.names.append("(unsolvable)")
        goals.add(sentinel)  # never produced by any operator

    axioms = tuple(
        Axiom(
            enc.fact(head),
            frozenset(
                {enc.fact(a) for a in body_pos} | {enc.comp(a) for a in body_neg}
            ),
        )
        for head, body_pos, body_neg in resolved_axioms
    )

    # Anything a compilation introduced is bookkeeping, not something the
    # domain author wrote, so keep it out of printed plans.
    synthetic |= {
        op.name for op in operators if op.base_name.startswith(SYNTHETIC_PREFIX)
    }
    temporal = any(op.duration for op in operators)
    clock_index = index_of.get(CLOCK_FLUENT)
    metric = None
    if problem.metric is not None:
        direction, expression = problem.metric
        metric = f"{direction} {expression}"

    return Task(
        name=problem.name or domain.name,
        facts=tuple(enc.names),
        init=frozenset(init_ids),
        goals=frozenset(goals),
        operators=tuple(operators),
        metric_cost=problem.metric_minimize_cost,
        axioms=axioms,
        numeric_names=numeric_names,
        init_values=init_values,
        goal_numeric=goal_numeric,
        temporal=temporal,
        clock_index=clock_index,
        metric=metric,
        requirements=requirements,
        synthetic=frozenset(synthetic),
    )


def ground_files(domain_path: str, problem_path: str) -> Task:
    """Convenience: parse both files and ground them into a :class:`Task`."""
    return ground(parse_domain_file(domain_path), parse_problem_file(problem_path))
