"""What jupyddl does with every PDDL requirement flag.

This module is the single source of truth: the parser consults it to accept or
reject a domain, ``jupyddl requirements`` prints it, the playground renders it,
and the README table is generated from it. If support for a flag changes, it
changes here and everywhere else follows.

Each requirement carries a :class:`Support` level:

``NATIVE``
    Modelled directly by the grounder and the search.
``COMPILED``
    Accepted and compiled into the core representation. The plans are correct
    for the stated semantics, but the compilation may change what the search
    sees (a disjunctive precondition becomes several operators, for instance).
``PARTIAL``
    Accepted with a documented restriction. ``note`` says exactly what is and
    is not covered — read it before trusting results.
``REJECTED``
    Parsed, recognised, and refused with a clear error rather than silently
    mis-planned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

NATIVE = "native"
COMPILED = "compiled"
PARTIAL = "partial"
REJECTED = "rejected"

SUPPORT_ORDER = [NATIVE, COMPILED, PARTIAL, REJECTED]

__all__ = [
    "Requirement",
    "REQUIREMENTS",
    "NATIVE",
    "COMPILED",
    "PARTIAL",
    "REJECTED",
    "SUPPORT_ORDER",
    "lookup",
    "expand",
    "supported",
    "unsupported_reason",
    "summary",
]


@dataclass(frozen=True)
class Requirement:
    """One PDDL requirement flag and what this implementation does with it."""

    name: str  # including the leading colon
    pddl: str  # the PDDL level that introduced it
    support: str
    summary: str
    note: str = ""
    implies: tuple = ()  # flags this one turns on, per the PDDL spec

    @property
    def is_supported(self) -> bool:
        return self.support != REJECTED


def _r(name, pddl, support, summary, note="", implies=()):
    return Requirement(name, pddl, support, summary, note, tuple(implies))


# The full PDDL 1.2 / 2.1 / 2.2 / 3.0 / 3.1 requirement vocabulary.
REQUIREMENTS = {
    r.name: r
    for r in [
        _r(
            ":strips",
            "1.2",
            NATIVE,
            "Add/delete effects over positive literals.",
            "The core representation. Everything else compiles down to this.",
        ),
        _r(
            ":typing",
            "1.2",
            NATIVE,
            "Typed parameters and objects, with type hierarchies.",
            "Subtypes are resolved transitively when building the object pools.",
        ),
        _r(
            ":negative-preconditions",
            "1.2",
            COMPILED,
            "`not` in preconditions and goals.",
            "Compiled to positive normal form: each negated fluent gets a "
            "complement fact that every operator maintains.",
        ),
        _r(
            ":disjunctive-preconditions",
            "1.2",
            COMPILED,
            "`or` in preconditions and goals.",
            "Preconditions are converted to DNF; each disjunct becomes its own "
            "grounded operator. A disjunctive goal becomes a single artificial "
            "goal fact achieved by one zero-cost operator per disjunct.",
        ),
        _r(
            ":equality",
            "1.2",
            NATIVE,
            "`=` between terms.",
            "Evaluated during grounding; infeasible instances are dropped.",
        ),
        _r(
            ":existential-preconditions",
            "1.2",
            COMPILED,
            "`exists` in preconditions and goals.",
            "Expanded over the typed object pool into a disjunction, then "
            "handled like any other disjunction.",
        ),
        _r(
            ":universal-preconditions",
            "1.2",
            COMPILED,
            "`forall` in preconditions and goals.",
            "Expanded over the typed object pool into a conjunction.",
        ),
        _r(
            ":quantified-preconditions",
            "1.2",
            COMPILED,
            "Both `exists` and `forall` in preconditions.",
            "",
            implies=(":existential-preconditions", ":universal-preconditions"),
        ),
        _r(
            ":conditional-effects",
            "1.2",
            NATIVE,
            "`when` and `forall` inside effects.",
            "Kept explicitly in the grounded operator and evaluated against the "
            "state at application time.",
        ),
        _r(
            ":adl",
            "1.2",
            COMPILED,
            "The full ADL feature set.",
            "",
            implies=(
                ":strips",
                ":typing",
                ":negative-preconditions",
                ":disjunctive-preconditions",
                ":equality",
                ":quantified-preconditions",
                ":conditional-effects",
            ),
        ),
        _r(
            ":derived-predicates",
            "2.2",
            NATIVE,
            "`(:derived ...)` axioms.",
            "Rules are grounded and evaluated to a least fixpoint after every "
            "state change, so planners and heuristics always see closed states. "
            "Derived predicates may not appear in action effects.",
        ),
        _r(
            ":action-costs",
            "3.0",
            NATIVE,
            "`(increase (total-cost) k)` and metric minimisation.",
            "Operator costs feed straight into g, so A*/Dijkstra optimise cost.",
        ),
        _r(
            ":numeric-fluents",
            "2.1",
            PARTIAL,
            "Numeric state variables, comparisons and assignments.",
            "Supported: ground numeric fluents, comparisons (< <= = >= >) in "
            "preconditions and goals, and assign/increase/decrease/scale-up/"
            "scale-down effects over arithmetic expressions. Numeric values are "
            "part of the state, so the state space can become infinite — the "
            "delete-relaxation heuristics ignore numeric conditions, which keeps "
            "them admissible but uninformative about them.",
        ),
        _r(
            ":fluents",
            "2.1",
            PARTIAL,
            "Numeric plus object fluents.",
            "Both halves are implemented; see the two rows above for what each "
            "one covers.",
            implies=(":numeric-fluents",),
        ),
        _r(
            ":durative-actions",
            "2.1",
            PARTIAL,
            "`(:durative-action ...)` with timed conditions and effects.",
            "Compiled to sequential actions: at-start and over-all conditions "
            "become the precondition, at-start and at-end effects are merged, "
            "and the duration is carried through so plans report a makespan. "
            "Actions therefore never overlap — this models sequential temporal "
            "planning, not true concurrency, so a plan needing two actions to "
            "run at the same time will not be found.",
        ),
        _r(
            ":duration-inequalities",
            "2.1",
            COMPILED,
            "Durations bounded by inequalities rather than fixed.",
            "Bounds are collected and the shortest feasible duration is chosen. "
            "With no concurrency and no continuous change nothing in the model "
            "prefers a longer action, so the tightest lower bound is "
            "makespan-optimal. Strict `<`/`>` are refused: they have no shortest "
            "feasible value.",
        ),
        _r(
            ":continuous-effects",
            "2.1",
            REJECTED,
            "Effects that change continuously over an action's duration.",
            "Requires continuous-time reasoning, which this planner does not do.",
        ),
        _r(
            ":timed-initial-literals",
            "2.2",
            PARTIAL,
            "Facts that become true (or false) at a given absolute time.",
            "Elapsed time becomes a numeric fluent advanced by action durations. "
            "Each literal gets a firing action guarded on the clock plus a wait "
            "action that advances it, and every domain action is blocked while a "
            "due literal has not fired. Because actions never overlap, a literal "
            "scheduled strictly inside an action's duration fires immediately "
            "after that action rather than during it.",
        ),
        _r(
            ":preferences",
            "3.0",
            PARTIAL,
            "Soft goals scored by a metric.",
            "Goal preferences are compiled into a priced choice: a closing "
            "action freezes the state, then each preference is resolved either "
            "for free (if it holds) or at its `(is-violated p)` weight, so "
            "cost-optimal search minimises the metric. Preferences over "
            "trajectory constraints or inside action preconditions are refused.",
        ),
        _r(
            ":constraints",
            "3.0",
            PARTIAL,
            "State trajectory constraints.",
            "`always`, `at-end`, `sometime`, `sometime-before`, `sometime-after` "
            "and `at-most-once` are compiled into invariants on every action, "
            "monitor facts and extra goal conjuncts. The metric-time forms "
            "(`within`, `always-within`, `hold-after`, `hold-during`) are "
            "refused by name.",
        ),
        _r(
            ":object-fluents",
            "3.1",
            PARTIAL,
            "Functions returning objects rather than numbers.",
            "Compiled to a predicate plus a uniqueness rule: `(= (location ?p) "
            "?x)` becomes a fact and `assign` clears the old value first. Using "
            "an object fluent as a *nested term* — `(at ?t (location ?p))` — is "
            "refused; write the equality form instead.",
        ),
    ]
}


def lookup(name: str) -> Optional[Requirement]:
    """Find a requirement by flag, tolerating a missing leading colon."""
    if not name:
        return None
    key = name if name.startswith(":") else f":{name}"
    return REQUIREMENTS.get(key.lower())


def expand(names) -> set:
    """Close a requirement list under `implies` (`:adl` turns on seven others)."""
    seen: set = set()
    queue = list(names)
    while queue:
        raw = queue.pop()
        requirement = lookup(raw)
        if requirement is None or requirement.name in seen:
            continue
        seen.add(requirement.name)
        queue.extend(requirement.implies)
    return seen


def supported(name: str) -> bool:
    """True when the flag is known *and* not rejected."""
    requirement = lookup(name)
    return requirement is not None and requirement.is_supported


def unsupported_reason(name: str) -> str:
    """A human explanation of why a flag is refused (empty when it is not)."""
    requirement = lookup(name)
    if requirement is None:
        return f"'{name}' is not a PDDL requirement flag this planner recognises"
    if requirement.is_supported:
        return ""
    detail = f" {requirement.note}" if requirement.note else ""
    return f"'{requirement.name}' is not supported: {requirement.summary}{detail}"


def summary() -> dict:
    """Counts per support level, for the docs and the playground badge."""
    counts = {level: 0 for level in SUPPORT_ORDER}
    for requirement in REQUIREMENTS.values():
        counts[requirement.support] += 1
    return counts


def as_rows() -> list:
    """Flat, serialisable rows — used by the CLI, the web UI and the README."""
    return [
        {
            "name": requirement.name,
            "pddl": requirement.pddl,
            "support": requirement.support,
            "summary": requirement.summary,
            "note": requirement.note,
            "implies": list(requirement.implies),
        }
        for requirement in sorted(
            REQUIREMENTS.values(),
            key=lambda r: (SUPPORT_ORDER.index(r.support), r.name),
        )
    ]
