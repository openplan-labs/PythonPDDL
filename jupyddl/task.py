"""Grounded task representation.

A :class:`Task` is a fully grounded planning problem where every fact is an
integer id, so states are cheap frozensets and successor generation is set
arithmetic. Negative preconditions and goals are compiled away into *positive
normal form* by the grounder, so preconditions and goals are purely positive.
Conditional effects are retained explicitly, so ADL domains are solved rather
than rejected.

Three optional layers ride on top, each inert unless the domain uses it:

* **numeric fluents** — when present, a state is a :class:`State` carrying a
  tuple of numbers alongside the fact set instead of a bare frozenset;
* **derived predicates** — :class:`Axiom` rules closed to a fixpoint after every
  state change, so nothing downstream ever sees an unclosed state;
* **durations** — carried per operator so temporal plans can report a makespan.

A classical STRIPS task pays for none of them: the checks are single truthiness
tests on empty tuples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CondEffect:
    """A conditional effect: when ``condition`` holds, apply add/delete."""

    condition: frozenset
    add: frozenset
    delete: frozenset


@dataclass(frozen=True)
class Axiom:
    """A grounded derived-predicate rule: ``body`` implies ``head``."""

    head: int
    body: frozenset


@dataclass(frozen=True)
class State:
    """A state with numeric fluents: a fact set plus a vector of numbers."""

    facts: frozenset
    values: tuple

    def __contains__(self, fact) -> bool:
        return fact in self.facts

    def __iter__(self):
        return iter(self.facts)

    def __len__(self) -> int:
        return len(self.facts)


def facts_of(state) -> frozenset:
    """The fact set of a state, whether or not it carries numbers."""
    return state.facts if isinstance(state, State) else state


def values_of(state) -> tuple:
    """The numeric vector of a state (empty for classical tasks)."""
    return state.values if isinstance(state, State) else ()


@dataclass(frozen=True)
class Operator:
    """A grounded action over integer fact ids.

    ``numeric_pre`` holds callables ``values -> bool`` and ``numeric_eff`` holds
    ``(index, values -> float)`` pairs; both are empty for classical tasks. The
    right-hand sides are all evaluated against the *pre-state*, so simultaneous
    assignments behave as PDDL specifies.
    """

    name: str
    precond: frozenset
    add: frozenset
    delete: frozenset
    cond_effects: tuple = ()
    cost: int = 1
    numeric_pre: tuple = ()
    numeric_eff: tuple = ()
    duration: float = 0.0

    @property
    def base_name(self) -> str:
        """The action as written in the domain, without any compilation tag.

        Splitting a disjunctive precondition produces ``move(a,b)#2``; a plan
        handed to another tool wants ``move(a,b)``. The tag stays on ``name`` so
        traces can still tell the disjuncts apart.
        """
        head, sep, _ = self.name.partition("#")
        return head if sep else self.name

    def applicable(self, state) -> bool:
        if not self.precond <= facts_of(state):
            return False
        if self.numeric_pre:
            values = values_of(state)
            return all(test(values) for test in self.numeric_pre)
        return True

    def apply(self, state):
        """Return the successor state (conditions evaluated in ``state``)."""
        facts = facts_of(state)
        dels = set(self.delete)
        adds = set(self.add)
        for effect in self.cond_effects:
            if effect.condition <= facts:
                dels |= effect.delete
                adds |= effect.add
        # Adds take precedence over deletes (standard STRIPS semantics).
        new_facts = frozenset((facts - dels) | adds)

        if not isinstance(state, State):
            return new_facts
        if not self.numeric_eff:
            return State(new_facts, state.values)
        values = list(state.values)
        old = state.values  # every right-hand side reads the pre-state
        for index, compute in self.numeric_eff:
            values[index] = compute(old)
        return State(new_facts, tuple(values))


@dataclass
class Task:
    """A grounded planning task."""

    name: str
    facts: tuple  # id -> human-readable fact string
    init: frozenset
    goals: frozenset
    operators: tuple
    metric_cost: bool = False
    axioms: tuple = ()
    numeric_names: tuple = ()  # id -> human-readable fluent string
    init_values: tuple = ()
    goal_numeric: tuple = ()  # callables values -> bool
    temporal: bool = False
    metric: Optional[str] = None
    requirements: tuple = ()
    # Operators introduced purely by a compilation step (a disjunctive goal,
    # say). They are real operators to the search but noise in a printed plan.
    synthetic: frozenset = field(default_factory=frozenset)

    @property
    def num_facts(self) -> int:
        return len(self.facts)

    @property
    def numeric(self) -> bool:
        return bool(self.numeric_names)

    def initial_state(self):
        """The initial state, with axioms already closed."""
        if self.numeric:
            return self.close(State(self.init, self.init_values))
        return self.close(self.init)

    def close(self, state):
        """Add every derived fact entailed by ``state`` (least fixpoint).

        Rules are re-scanned until nothing new fires. Derived predicates cannot
        appear in action effects, so this only ever grows the fact set.
        """
        if not self.axioms:
            return state
        facts = set(facts_of(state))
        changed = True
        while changed:
            changed = False
            for axiom in self.axioms:
                if axiom.head not in facts and axiom.body <= facts:
                    facts.add(axiom.head)
                    changed = True
        closed = frozenset(facts)
        if isinstance(state, State):
            return State(closed, state.values)
        return closed

    def apply(self, operator: Operator, state):
        """Apply ``operator`` to ``state`` and re-close the derived predicates."""
        return self.close(operator.apply(state))

    def goal_reached(self, state) -> bool:
        if not self.goals <= facts_of(state):
            return False
        if self.goal_numeric:
            values = values_of(state)
            return all(test(values) for test in self.goal_numeric)
        return True

    def applicable_operators(self, state):
        for operator in self.operators:
            if operator.applicable(state):
                yield operator

    def fact_name(self, fact_id: int) -> str:
        return self.facts[fact_id]

    def state_str(self, state) -> str:
        return "{" + ", ".join(sorted(self.facts[f] for f in facts_of(state))) + "}"

    def makespan(self, plan) -> float:
        """Total duration of a plan under the sequential temporal compilation."""
        return sum(op.duration for op in plan or ())

    def visible_plan(self, plan) -> list:
        """Drop operators that only exist because of a compilation step."""
        if not self.synthetic:
            return list(plan or ())
        return [op for op in (plan or ()) if op.name not in self.synthetic]

    def relaxed_operators(self):
        """Delete-relaxation operator view used by relaxation heuristics.

        Every conditional effect becomes its own relaxed operator whose
        precondition is the action precondition conjoined with the effect
        condition. This is the standard, sound treatment of conditional effects
        under delete relaxation. Numeric preconditions are *ignored*, which only
        makes the relaxation more permissive and so keeps the heuristics
        admissible; axioms enter as zero-cost rules for the same reason.
        Returns a list of ``(precond, add, cost)``.
        """
        relaxed = []
        for operator in self.operators:
            if operator.add:
                relaxed.append((operator.precond, operator.add, operator.cost))
            for effect in operator.cond_effects:
                if effect.add:
                    relaxed.append(
                        (operator.precond | effect.condition, effect.add, operator.cost)
                    )
            if not operator.add and not operator.cond_effects:
                # Keep operators with only delete effects visible (no-op in the
                # relaxation) so nothing downstream assumes non-empty adds.
                relaxed.append((operator.precond, frozenset(), operator.cost))
        for axiom in self.axioms:
            relaxed.append((axiom.body, frozenset({axiom.head}), 0))
        return relaxed
