"""Structured AST for PDDL.

Conditions are stored as a **formula tree in negation normal form**: the parser
pushes every ``not`` down to the atoms and rewrites ``imply``, so the grounder
only ever sees negation applied to a literal. Quantifiers stay in the tree
because expanding them needs the object pool, which only exists at grounding
time; the grounder then distributes the formula into DNF and emits one operator
per disjunct.

Numeric fluents and durative actions have their own small node types. See
:mod:`jupyddl.requirements` for exactly which PDDL requirement flags are
supported and how.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class PDDLError(Exception):
    """Base class for all parsing / modelling errors."""


class UnsupportedFeatureError(PDDLError):
    """Raised when a PDDL construct outside the supported subset is used."""


@dataclass(frozen=True)
class Atom:
    """A (possibly lifted) predicate application, e.g. ``(on ?x ?y)``.

    ``args`` holds terms as raw strings: variables keep their leading ``?``
    while constants/objects are stored verbatim.
    """

    predicate: str
    args: tuple = ()

    def __str__(self) -> str:
        if not self.args:
            return f"({self.predicate})"
        return f"({self.predicate} {' '.join(self.args)})"


@dataclass(frozen=True)
class Literal:
    """A positive or negative atom used in preconditions and goals."""

    atom: Atom
    positive: bool = True


@dataclass(frozen=True)
class EqualityConstraint:
    """An ``(= a b)`` (or its negation) constraint over terms."""

    left: str
    right: str
    positive: bool = True


# --- numeric expressions -----------------------------------------------------


@dataclass(frozen=True)
class Number:
    """A numeric literal."""

    value: float


@dataclass(frozen=True)
class FluentRef:
    """A reference to a numeric fluent, e.g. ``(fuel ?truck)``."""

    name: str
    args: tuple = ()

    def __str__(self) -> str:
        if not self.args:
            return f"({self.name})"
        return f"({self.name} {' '.join(self.args)})"


@dataclass(frozen=True)
class Arithmetic:
    """A binary arithmetic expression (``+``, ``-``, ``*``, ``/``).

    Unary minus is parsed as ``(- 0 x)``.
    """

    op: str
    left: object
    right: object


@dataclass(frozen=True)
class Comparison:
    """A numeric comparison used in preconditions and goals."""

    op: str  # one of < <= = >= >
    left: object
    right: object


# --- condition formulas (negation normal form) -------------------------------


@dataclass(frozen=True)
class Truth:
    """The constant ``true`` — what an empty ``(and)`` parses to."""

    value: bool = True


@dataclass(frozen=True)
class And:
    parts: tuple = ()


@dataclass(frozen=True)
class Or:
    parts: tuple = ()


@dataclass(frozen=True)
class Exists:
    params: tuple = ()  # ((var, type), ...)
    body: object = None


@dataclass(frozen=True)
class Forall:
    params: tuple = ()  # ((var, type), ...)
    body: object = None


@dataclass
class Conjunct:
    """One ground DNF disjunct: a conjunction of literals and comparisons.

    Produced by the grounder, not by the parser. ``equalities`` are resolved
    during instantiation and never survive into a grounded operator.
    """

    literals: list = field(default_factory=list)
    equalities: list = field(default_factory=list)
    comparisons: list = field(default_factory=list)


# --- effects -----------------------------------------------------------------


@dataclass
class AddEffect:
    atom: Atom


@dataclass
class DelEffect:
    atom: Atom


@dataclass
class IncreaseCostEffect:
    """``(increase (total-cost) k)`` — the classical action-cost shorthand."""

    amount: float


@dataclass
class NumericEffect:
    """An assignment to a numeric fluent.

    ``op`` is one of ``assign``, ``increase``, ``decrease``, ``scale-up`` or
    ``scale-down``.
    """

    op: str
    target: FluentRef
    value: object  # an arithmetic expression


@dataclass
class ConjunctiveEffect:
    parts: list = field(default_factory=list)


@dataclass
class ForallEffect:
    params: list  # [(variable, type), ...]
    body: object


@dataclass
class WhenEffect:
    condition: object  # a condition formula
    body: object


# --- domain / problem --------------------------------------------------------


@dataclass
class Predicate:
    name: str
    params: list  # [(variable, type), ...]


@dataclass
class Function:
    """A declared numeric function (``:functions``)."""

    name: str
    params: list = field(default_factory=list)


@dataclass
class Action:
    name: str
    parameters: list  # [(variable, type), ...]
    precondition: object  # a condition formula
    effect: object  # one of the *Effect nodes above
    # Set when the action came from a (:durative-action ...) block; the value is
    # its duration, and the compilation is documented in jupyddl.requirements.
    duration: object = None


@dataclass
class DerivedPredicate:
    """A ``(:derived (head ?x) body)`` axiom."""

    head: Atom
    params: list  # [(variable, type), ...]
    body: object  # a condition formula


@dataclass
class Domain:
    name: str
    requirements: list
    types: dict  # child type -> parent type ("object" if none)
    constants: list  # [(name, type), ...]
    predicates: list
    actions: list
    functions: list = field(default_factory=list)
    derived: list = field(default_factory=list)

    @property
    def has_durative_actions(self) -> bool:
        return any(action.duration is not None for action in self.actions)


@dataclass
class Problem:
    name: str
    domain_name: str
    objects: list  # [(name, type), ...]
    init: list  # [Atom, ...]
    goal: object  # a condition formula
    metric_minimize_cost: bool = False
    init_numeric: dict = field(default_factory=dict)  # FluentRef -> float
    metric: object = None  # (direction, expression) or None
