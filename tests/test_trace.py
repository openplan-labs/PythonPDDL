"""Search instrumentation: observers, recorders and serialisable traces."""

from __future__ import annotations

import json
import math

import pytest

from conftest import paths
from jupyddl import build_task, solve_task, trace_search
from jupyddl.search import PLANNERS
from jupyddl.trace import (
    BOUND,
    EXPAND,
    FINISH,
    GOAL,
    START,
    MultiObserver,
    SearchObserver,
    SearchTrace,
    TraceRecorder,
)

INFORMED = {"gbfs", "astar", "wastar", "idastar", "ehc"}


class CountingObserver(SearchObserver):
    """Minimal observer that just tallies the hooks it receives."""

    def __init__(self):
        self.starts = 0
        self.expands = 0
        self.generates = 0
        self.bounds = 0
        self.goals = 0
        self.finishes = 0

    def on_start(self, task, planner, heuristic=""):
        self.starts += 1

    def on_expand(self, state, **kwargs):
        self.expands += 1

    def on_generate(self, state, **kwargs):
        self.generates += 1

    def on_bound(self, threshold, iteration, stats=None):
        self.bounds += 1

    def on_goal(self, state, g=0.0, depth=0, stats=None):
        self.goals += 1

    def on_finish(self, result):
        self.finishes += 1


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_every_planner_emits_start_and_finish(examples_available, planner):
    task = build_task(*paths("tsp"))
    observer = CountingObserver()
    heuristic = "hff" if planner in INFORMED else None
    solve_task(task, planner, heuristic, observer=observer)
    assert observer.starts == 1
    assert observer.finishes == 1
    assert observer.expands > 0


@pytest.mark.parametrize("planner", sorted(PLANNERS))
def test_observer_does_not_change_the_result(examples_available, planner):
    """Instrumentation must be transparent: same plan, same statistics."""
    task = build_task(*paths("tsp"))
    heuristic = "hff" if planner in INFORMED else None
    plain = solve_task(task, planner, heuristic)
    traced = solve_task(task, planner, heuristic, observer=CountingObserver())
    assert plain.solved == traced.solved
    assert plain.cost == traced.cost
    assert plain.plan_names() == traced.plan_names()
    assert plain.stats.expanded == traced.stats.expanded
    assert plain.stats.generated == traced.stats.generated


def test_trace_records_the_search(examples_available):
    task = build_task(*paths("tsp"))
    result, trace = trace_search(task, "astar", "lmcut")
    assert trace.planner == "astar"
    assert trace.heuristic == "lmcut"
    assert trace.task_name == task.name
    assert trace.solved and trace.cost == result.cost
    assert trace.plan == result.plan_names()
    assert trace.label == "astar/lmcut"
    assert [e.kind for e in trace.events][0] == START
    assert trace.events[-1].kind == FINISH
    assert trace.of_kind(GOAL)
    assert len(trace.expansions) > 0
    # The recorded statistics agree with the planner's own bookkeeping.
    assert trace.stats["expanded"] == result.stats.expanded


def test_trace_series_and_summary(examples_available):
    task = build_task(*paths("pallet"))
    _, trace = trace_search(task, "astar", "hmax")
    fs = trace.series("f")
    hs = trace.series("h")
    assert len(fs) == len(trace.expansions) == len(hs)
    assert all(value >= 0 for value in hs)
    summary = trace.summary()
    assert summary["solved"] is True
    assert summary["label"] == "astar/hmax"
    assert summary["expanded"] == trace.stats["expanded"]


def test_uninformed_planner_has_no_heuristic_name(examples_available):
    task = build_task(*paths("tsp"))
    _, trace = trace_search(task, "bfs", None)
    assert trace.heuristic == ""
    assert trace.label == "bfs"


def test_iterative_deepening_reports_bounds(examples_available):
    task = build_task(*paths("pallet"))
    _, trace = trace_search(task, "idastar", "hmax")
    bounds = trace.of_kind(BOUND)
    assert bounds, "IDA* should report each new f-bound"
    thresholds = [event.threshold for event in bounds]
    # Bounds are non-decreasing: IDA* only ever raises the limit.
    assert thresholds == sorted(thresholds)


def test_generated_events_are_opt_in(examples_available):
    task = build_task(*paths("tsp"))
    _, quiet = trace_search(task, "astar", "hmax")
    _, loud = trace_search(task, "astar", "hmax", record_generated=True)
    assert quiet.of_kind("generate") == []
    assert loud.of_kind("generate")


def test_tree_edges_reference_recorded_nodes(examples_available):
    task = build_task(*paths("pallet"))
    _, trace = trace_search(task, "astar", "hmax")
    known = {event.node for event in trace.expansions}
    for parent, node, _action in trace.tree_edges():
        assert node in known
        assert parent in known


def test_max_events_bounds_memory(examples_available):
    """A capped recorder thins its samples instead of growing without limit."""
    task = build_task(*paths("pallet"))
    recorder = TraceRecorder(max_events=32)
    solve_task(task, "bfs", None, observer=recorder)
    trace = recorder.trace
    assert len(trace.events) <= 64  # thinning halves, so allow one overshoot
    assert trace.stride > 1
    # Structural events survive thinning.
    assert trace.of_kind(START) and trace.of_kind(FINISH)


def test_round_trip_through_json(examples_available, tmp_path):
    task = build_task(*paths("tsp"))
    _, trace = trace_search(task, "astar", "lmcut")
    path = tmp_path / "trace.json"
    trace.save(str(path))

    restored = SearchTrace.load(str(path))
    assert restored.label == trace.label
    assert restored.cost == trace.cost
    assert restored.plan == trace.plan
    assert len(restored.events) == len(trace.events)
    assert restored.series("f") == trace.series("f")


def test_json_is_serialisable_without_infinities(examples_available):
    """Dead ends produce inf heuristics; JSON has no inf, so they become null."""
    task = build_task(*paths("tsp"))
    _, trace = trace_search(task, "astar", "hmax")
    payload = json.loads(trace.to_json())
    for event in payload["events"]:
        for key in ("g", "h", "f"):
            value = event[key]
            assert value is None or not math.isinf(value)


def test_multi_observer_fans_out(examples_available):
    task = build_task(*paths("tsp"))
    first, second = CountingObserver(), CountingObserver()
    solve_task(task, "astar", "hmax", observer=MultiObserver(first, second))
    assert first.expands == second.expands > 0
    assert first.finishes == second.finishes == 1


def test_trace_search_accepts_an_extra_observer(examples_available):
    task = build_task(*paths("tsp"))
    extra = CountingObserver()
    _, trace = trace_search(task, "astar", "hmax", observer=extra)
    assert extra.expands > 0
    assert len(trace.expansions) > 0


def test_expand_events_carry_consistent_costs(examples_available):
    task = build_task(*paths("pallet"))
    _, trace = trace_search(task, "astar", "hmax")
    for event in trace.expansions:
        assert event.g >= 0
        assert event.h >= 0
        assert event.depth >= 0
        # A* priority is f = g + h.
        assert event.f == pytest.approx(event.g + event.h)


def test_base_observer_hooks_are_all_no_ops():
    """Subclasses may override only what they need."""
    observer = SearchObserver()
    observer.on_start(None, "planner")
    observer.on_expand(None)
    observer.on_generate(None)
    observer.on_bound(1.0, 0)
    observer.on_goal(None)
    observer.on_finish(None)


def test_unsolvable_instance_still_finishes_the_trace(examples_available):
    task = build_task(*paths("vehicle"))
    result, trace = trace_search(task, "bfs", None)
    assert not result.solved
    assert trace.solved is False
    assert trace.plan == []
    assert trace.events[-1].kind == FINISH
    assert trace.of_kind(GOAL) == []


def test_expand_count_matches_stats(examples_available):
    """With no thinning, one expand event is recorded per expansion."""
    task = build_task(*paths("tsp"))
    result, trace = trace_search(task, "astar", "hmax", max_events=100000)
    assert len(trace.of_kind(EXPAND)) == result.stats.expanded
