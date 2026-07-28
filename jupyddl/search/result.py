"""Search result, statistics and resource budgets shared by all planners."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchStats:
    """Bookkeeping collected while searching (for comparative analysis)."""

    expanded: int = 0
    generated: int = 0
    evaluated: int = 0  # heuristic evaluations
    reopened: int = 0
    deadends: int = 0
    runtime: float = 0.0
    # True when the planner stopped because it ran out of budget rather than
    # because it finished. An optimality claim only holds when this is False.
    truncated: bool = False

    def as_dict(self) -> dict:
        return {
            "expanded": self.expanded,
            "generated": self.generated,
            "evaluated": self.evaluated,
            "reopened": self.reopened,
            "deadends": self.deadends,
            "runtime": self.runtime,
            "truncated": self.truncated,
        }


class Budget:
    """A cooperative resource limit checked inside a search loop.

    Python has no safe way to interrupt a running search from outside, so the
    planners poll this instead.

    ``check_every`` throttles how often the clock is read. It defaults to 1 --
    every call -- because a single expansion can cost tens of milliseconds under
    an expensive heuristic like LM-cut, and a coarser interval would overshoot a
    short time limit by orders of magnitude. Reading a monotonic clock costs
    tens of nanoseconds, which is nothing beside an expansion.

    A budget that is hit sets ``stats.truncated``, so a caller can always tell
    "no plan exists" apart from "we stopped looking".
    """

    def __init__(
        self,
        max_expansions: Optional[int] = None,
        time_limit: Optional[float] = None,
        check_every: int = 1,
    ):
        self.max_expansions = max_expansions
        self.time_limit = time_limit
        self.check_every = max(1, int(check_every))
        self._deadline = None

    @property
    def unlimited(self) -> bool:
        return self.max_expansions is None and self.time_limit is None

    def start(self) -> "Budget":
        self._deadline = (
            None if self.time_limit is None else time.perf_counter() + self.time_limit
        )
        return self

    def exceeded(self, stats: SearchStats) -> bool:
        if self.max_expansions is not None and stats.expanded >= self.max_expansions:
            stats.truncated = True
            return True
        if self._deadline is not None and stats.expanded % self.check_every == 0:
            if time.perf_counter() >= self._deadline:
                stats.truncated = True
                return True
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Budget(max_expansions={self.max_expansions}, "
            f"time_limit={self.time_limit})"
        )


def make_budget(max_expansions=None, time_limit=None) -> Optional[Budget]:
    """Build a :class:`Budget`, or ``None`` when nothing is limited."""
    if max_expansions is None and time_limit is None:
        return None
    return Budget(max_expansions, time_limit).start()


@dataclass
class SearchResult:
    """The outcome of a search: a plan (list of operators) plus statistics."""

    solved: bool
    plan: Optional[list] = None  # list[Operator]
    cost: Optional[int] = None
    stats: SearchStats = field(default_factory=SearchStats)

    @property
    def plan_length(self) -> Optional[int]:
        return None if self.plan is None else len(self.plan)

    @property
    def truncated(self) -> bool:
        """The search stopped on a budget, so 'no plan' means 'none found yet'."""
        return self.stats.truncated

    def plan_names(self, canonical: bool = False) -> Optional[list]:
        """Operator names; ``canonical=True`` drops compilation tags like ``#2``."""
        if self.plan is None:
            return None
        if canonical:
            return [op.base_name for op in self.plan]
        return [op.name for op in self.plan]
