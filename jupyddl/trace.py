"""Search instrumentation: observers, events and serialisable search traces.

Every planner accepts an optional ``observer``. While the search runs it emits
structured events (a node was expanded, a successor was generated, a new
``f``-bound was opened, the goal was reached...). Observers turn those events
into whatever you need:

* :class:`TraceRecorder` accumulates them into a :class:`SearchTrace` that can
  be plotted, diffed against another configuration, or exported to JSON for the
  web playground;
* the dashboards in :mod:`jupyddl.viz.live` render them *while the search runs*.

Tracing is entirely opt-in: with no observer the planners never touch this
module, so the default search path keeps its zero-overhead, zero-dependency
behaviour.

Example::

    from jupyddl import build_task, trace_search

    task = build_task("domain.pddl", "problem.pddl")
    result, trace = trace_search(task, "astar", "lmcut")
    trace.save("run.json")
    print(trace.summary())
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

# Event kinds emitted by the planners.
START = "start"
EXPAND = "expand"
GENERATE = "generate"
BOUND = "bound"
GOAL = "goal"
FINISH = "finish"

__all__ = [
    "SearchEvent",
    "SearchObserver",
    "MultiObserver",
    "TraceRecorder",
    "SearchTrace",
    "START",
    "EXPAND",
    "GENERATE",
    "BOUND",
    "GOAL",
    "FINISH",
]


def _finite(value) -> Optional[float]:
    """JSON has no ``inf``/``nan``; map them to ``None``."""
    if value is None:
        return None
    value = float(value)
    if math.isinf(value) or math.isnan(value):
        return None
    return value


@dataclass
class SearchEvent:
    """One instrumented moment of a search.

    ``node``/``parent`` are small integer ids assigned per distinct state, so a
    trace can be replayed as a tree without holding on to the states.
    """

    step: int
    kind: str
    elapsed: float = 0.0
    node: int = -1
    parent: int = -1
    action: str = ""
    g: float = 0.0
    h: float = 0.0
    f: float = 0.0
    depth: int = 0
    open_size: int = 0
    expanded: int = 0
    generated: int = 0
    evaluated: int = 0
    threshold: Optional[float] = None
    iteration: int = 0

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "kind": self.kind,
            "elapsed": round(self.elapsed, 6),
            "node": self.node,
            "parent": self.parent,
            "action": self.action,
            "g": _finite(self.g),
            "h": _finite(self.h),
            "f": _finite(self.f),
            "depth": self.depth,
            "open": self.open_size,
            "expanded": self.expanded,
            "generated": self.generated,
            "evaluated": self.evaluated,
            "threshold": _finite(self.threshold),
            "iteration": self.iteration,
        }


class SearchObserver:
    """Base class for search observers — every hook defaults to a no-op.

    Subclass and override only what you care about. Planners call the hooks
    defensively, so an observer that raises will not corrupt a search: it is the
    observer's job to stay cheap and total.
    """

    def on_start(self, task, planner: str, heuristic: str = "") -> None:
        """Called once before the first node is touched."""

    def on_expand(
        self,
        state,
        g: float = 0.0,
        h: float = 0.0,
        f: float = 0.0,
        depth: int = 0,
        open_size: int = 0,
        stats=None,
        parent=None,
        action: str = "",
    ) -> None:
        """Called when a node is taken off the frontier and expanded."""

    def on_generate(
        self,
        state,
        parent=None,
        action: str = "",
        g: float = 0.0,
        h: float = 0.0,
        depth: int = 0,
        stats=None,
    ) -> None:
        """Called for each successor that enters the frontier."""

    def on_bound(self, threshold: float, iteration: int, stats=None) -> None:
        """Called by iterative-deepening planners when the bound grows."""

    def on_goal(self, state, g: float = 0.0, depth: int = 0, stats=None) -> None:
        """Called when a goal state is reached."""

    def on_finish(self, result) -> None:
        """Called once when the search stops, solved or not."""


class MultiObserver(SearchObserver):
    """Fan one search out to several observers (e.g. record *and* render)."""

    def __init__(self, *observers):
        self.observers = [o for o in observers if o is not None]

    def on_start(self, task, planner: str, heuristic: str = "") -> None:
        for obs in self.observers:
            obs.on_start(task, planner, heuristic)

    def on_expand(self, state, **kwargs) -> None:
        for obs in self.observers:
            obs.on_expand(state, **kwargs)

    def on_generate(self, state, **kwargs) -> None:
        for obs in self.observers:
            obs.on_generate(state, **kwargs)

    def on_bound(self, threshold: float, iteration: int, stats=None) -> None:
        for obs in self.observers:
            obs.on_bound(threshold, iteration, stats)

    def on_goal(self, state, g: float = 0.0, depth: int = 0, stats=None) -> None:
        for obs in self.observers:
            obs.on_goal(state, g, depth, stats)

    def on_finish(self, result) -> None:
        for obs in self.observers:
            obs.on_finish(result)


@dataclass
class SearchTrace:
    """A replayable record of one search run."""

    planner: str = ""
    heuristic: str = ""
    task_name: str = ""
    num_facts: int = 0
    num_operators: int = 0
    events: list = field(default_factory=list)
    solved: bool = False
    cost: Optional[int] = None
    plan: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    stride: int = 1  # >1 when events were thinned to respect ``max_events``

    # ---------------------------------------------------------------- naming
    @property
    def label(self) -> str:
        """``planner/heuristic`` — the name used in plot legends."""
        return f"{self.planner}/{self.heuristic}" if self.heuristic else self.planner

    # ------------------------------------------------------------ accessors
    def of_kind(self, kind: str) -> list:
        return [e for e in self.events if e.kind == kind]

    @property
    def expansions(self) -> list:
        return self.of_kind(EXPAND)

    def series(self, attr: str, kind: str = EXPAND) -> list:
        """Extract one attribute over the events of a kind, e.g. ``series("f")``."""
        return [getattr(e, attr) for e in self.of_kind(kind)]

    def tree_edges(self) -> list:
        """``(parent_id, node_id, action)`` for every expanded non-root node."""
        return [
            (e.parent, e.node, e.action)
            for e in self.expansions
            if e.parent >= 0 and e.node >= 0
        ]

    def bounds(self) -> list:
        """``(expanded_at, threshold)`` for iterative-deepening planners."""
        return [(e.expanded, e.threshold) for e in self.of_kind(BOUND)]

    def summary(self) -> dict:
        """Compact, printable digest of the run."""
        expansions = self.expansions
        h_values = [e.h for e in expansions if e.h is not None]
        return {
            "label": self.label,
            "task": self.task_name,
            "solved": self.solved,
            "cost": self.cost,
            "plan_length": len(self.plan),
            "events": len(self.events),
            "expanded": self.stats.get("expanded", len(expansions)),
            "generated": self.stats.get("generated", 0),
            "evaluated": self.stats.get("evaluated", 0),
            "runtime": self.stats.get("runtime", 0.0),
            "h_initial": h_values[0] if h_values else None,
            "h_min": min(h_values) if h_values else None,
        }

    # --------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict:
        return {
            "format": "jupyddl.trace/1",
            "planner": self.planner,
            "heuristic": self.heuristic,
            "task": self.task_name,
            "num_facts": self.num_facts,
            "num_operators": self.num_operators,
            "solved": self.solved,
            "cost": self.cost,
            "plan": list(self.plan),
            "stats": dict(self.stats),
            "stride": self.stride,
            "events": [e.as_dict() for e in self.events],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str, indent: Optional[int] = None) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json(indent=indent))

    @classmethod
    def from_dict(cls, data: dict) -> "SearchTrace":
        trace = cls(
            planner=data.get("planner", ""),
            heuristic=data.get("heuristic", ""),
            task_name=data.get("task", ""),
            num_facts=data.get("num_facts", 0),
            num_operators=data.get("num_operators", 0),
            solved=data.get("solved", False),
            cost=data.get("cost"),
            plan=list(data.get("plan", [])),
            stats=dict(data.get("stats", {})),
            stride=data.get("stride", 1),
        )
        for raw in data.get("events", []):
            trace.events.append(
                SearchEvent(
                    step=raw.get("step", 0),
                    kind=raw.get("kind", EXPAND),
                    elapsed=raw.get("elapsed", 0.0),
                    node=raw.get("node", -1),
                    parent=raw.get("parent", -1),
                    action=raw.get("action", ""),
                    g=raw.get("g") or 0.0,
                    h=raw.get("h") or 0.0,
                    f=raw.get("f") or 0.0,
                    depth=raw.get("depth", 0),
                    open_size=raw.get("open", 0),
                    expanded=raw.get("expanded", 0),
                    generated=raw.get("generated", 0),
                    evaluated=raw.get("evaluated", 0),
                    threshold=raw.get("threshold"),
                    iteration=raw.get("iteration", 0),
                )
            )
        return trace

    @classmethod
    def load(cls, path: str) -> "SearchTrace":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class TraceRecorder(SearchObserver):
    """Accumulate search events into a :class:`SearchTrace`.

    ``max_events`` bounds memory on big searches: once the cap is hit the
    recorder halves its sampling rate (keeping goal/bound events, which are rare
    and structurally important), so a trace of a million expansions still fits in
    memory while keeping the shape of the curves.
    """

    def __init__(
        self,
        max_events: int = 20000,
        record_generated: bool = False,
    ):
        self.max_events = max(16, int(max_events))
        self.record_generated = record_generated
        self.trace = SearchTrace()
        self._ids: dict = {}
        self._step = 0
        self._start = time.perf_counter()
        self._iteration = 0

    # --------------------------------------------------------------- helpers
    def _id(self, state) -> int:
        if state is None:
            return -1
        node_id = self._ids.get(state)
        if node_id is None:
            node_id = len(self._ids)
            self._ids[state] = node_id
        return node_id

    def _keep(self, kind: str) -> bool:
        """Sampling gate. Structural events are always kept."""
        if kind in (START, GOAL, FINISH, BOUND):
            return True
        return self._step % self.trace.stride == 0

    def _append(self, event: SearchEvent) -> None:
        self.trace.events.append(event)
        if len(self.trace.events) > self.max_events:
            self._thin()

    def _thin(self) -> None:
        """Halve the resolution of the recorded curve, keeping structure."""
        self.trace.stride *= 2
        kept, sampled = [], 0
        for event in self.trace.events:
            if event.kind in (START, GOAL, FINISH, BOUND):
                kept.append(event)
                continue
            if sampled % 2 == 0:
                kept.append(event)
            sampled += 1
        self.trace.events = kept

    def _snapshot(self, stats):
        if stats is None:
            return 0, 0, 0
        return stats.expanded, stats.generated, stats.evaluated

    # ----------------------------------------------------------- observer API
    def on_start(self, task, planner: str, heuristic: str = "") -> None:
        self._start = time.perf_counter()
        self.trace.planner = planner
        self.trace.heuristic = heuristic or ""
        if task is not None:
            self.trace.task_name = getattr(task, "name", "")
            self.trace.num_facts = getattr(task, "num_facts", 0)
            self.trace.num_operators = len(getattr(task, "operators", ()))
        self._append(SearchEvent(step=0, kind=START))

    def on_expand(
        self,
        state,
        g: float = 0.0,
        h: float = 0.0,
        f: float = 0.0,
        depth: int = 0,
        open_size: int = 0,
        stats=None,
        parent=None,
        action: str = "",
    ) -> None:
        self._step += 1
        if not self._keep(EXPAND):
            return
        expanded, generated, evaluated = self._snapshot(stats)
        self._append(
            SearchEvent(
                step=self._step,
                kind=EXPAND,
                elapsed=time.perf_counter() - self._start,
                node=self._id(state),
                parent=self._id(parent),
                action=action,
                g=g,
                h=h,
                f=f if f else g + h,
                depth=depth,
                open_size=open_size,
                expanded=expanded,
                generated=generated,
                evaluated=evaluated,
                iteration=self._iteration,
            )
        )

    def on_generate(
        self,
        state,
        parent=None,
        action: str = "",
        g: float = 0.0,
        h: float = 0.0,
        depth: int = 0,
        stats=None,
    ) -> None:
        if not self.record_generated:
            return
        self._step += 1
        if not self._keep(GENERATE):
            return
        expanded, generated, evaluated = self._snapshot(stats)
        self._append(
            SearchEvent(
                step=self._step,
                kind=GENERATE,
                elapsed=time.perf_counter() - self._start,
                node=self._id(state),
                parent=self._id(parent),
                action=action,
                g=g,
                h=h,
                f=g + h,
                depth=depth,
                expanded=expanded,
                generated=generated,
                evaluated=evaluated,
                iteration=self._iteration,
            )
        )

    def on_bound(self, threshold: float, iteration: int, stats=None) -> None:
        self._step += 1
        self._iteration = iteration
        expanded, generated, evaluated = self._snapshot(stats)
        self._append(
            SearchEvent(
                step=self._step,
                kind=BOUND,
                elapsed=time.perf_counter() - self._start,
                threshold=threshold,
                iteration=iteration,
                expanded=expanded,
                generated=generated,
                evaluated=evaluated,
            )
        )

    def on_goal(self, state, g: float = 0.0, depth: int = 0, stats=None) -> None:
        self._step += 1
        expanded, generated, evaluated = self._snapshot(stats)
        self._append(
            SearchEvent(
                step=self._step,
                kind=GOAL,
                elapsed=time.perf_counter() - self._start,
                node=self._id(state),
                g=g,
                depth=depth,
                f=g,
                expanded=expanded,
                generated=generated,
                evaluated=evaluated,
                iteration=self._iteration,
            )
        )

    def on_finish(self, result) -> None:
        self._step += 1
        self.trace.solved = bool(getattr(result, "solved", False))
        self.trace.cost = getattr(result, "cost", None)
        plan = getattr(result, "plan", None) or []
        self.trace.plan = [op.name for op in plan]
        stats = getattr(result, "stats", None)
        if stats is not None:
            self.trace.stats = stats.as_dict()
        expanded, generated, evaluated = self._snapshot(stats)
        self._append(
            SearchEvent(
                step=self._step,
                kind=FINISH,
                elapsed=time.perf_counter() - self._start,
                expanded=expanded,
                generated=generated,
                evaluated=evaluated,
            )
        )
