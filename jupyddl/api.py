"""High-level convenience API: parse + ground + plan + validate."""

from __future__ import annotations

from .grounding import ground_files
from .heuristics import make_heuristic
from .search import make_planner
from .search.result import SearchResult, make_budget
from .task import Task
from .trace import TraceRecorder


def build_task(domain_path: str, problem_path: str) -> Task:
    """Parse and ground a domain/problem pair into a :class:`Task`."""
    return ground_files(domain_path, problem_path)


def solve_task(
    task: Task,
    search: str = "astar",
    heuristic=None,
    observer=None,
    max_expansions=None,
    time_limit=None,
    **planner_kwargs,
) -> SearchResult:
    """Run ``search`` (optionally with ``heuristic``) on an already-ground task.

    Pass ``observer`` (see :mod:`jupyddl.trace`) to record or live-render the
    search as it runs. ``max_expansions`` and ``time_limit`` bound the run; when
    either is hit the planner stops and sets ``result.truncated``, so an empty
    result means "gave up", not "proved unsolvable".
    """
    planner = make_planner(search, **planner_kwargs)
    heur = None
    name = heuristic if heuristic else ("hff" if planner.requires_heuristic else None)
    if name is not None:
        heur = make_heuristic(name, task)
    budget = make_budget(max_expansions, time_limit)
    return planner.search(task, heur, observer=observer, budget=budget)


def solve(
    domain_path: str,
    problem_path: str,
    search: str = "astar",
    heuristic="lmcut",
    observer=None,
    max_expansions=None,
    time_limit=None,
    **planner_kwargs,
) -> SearchResult:
    """Parse, ground and solve a PDDL instance in one call."""
    task = build_task(domain_path, problem_path)
    return solve_task(
        task,
        search=search,
        heuristic=heuristic,
        observer=observer,
        max_expansions=max_expansions,
        time_limit=time_limit,
        **planner_kwargs,
    )


def trace_search(
    task: Task,
    search: str = "astar",
    heuristic=None,
    max_events: int = 20000,
    record_generated: bool = False,
    observer=None,
    max_expansions=None,
    time_limit=None,
    **planner_kwargs,
):
    """Solve ``task`` while recording the search.

    Returns ``(result, trace)`` where ``trace`` is a
    :class:`~jupyddl.trace.SearchTrace` ready to plot, save or replay. An extra
    ``observer`` (a live dashboard, say) is notified alongside the recorder.
    """
    recorder = TraceRecorder(max_events=max_events, record_generated=record_generated)
    if observer is not None:
        from .trace import MultiObserver

        sink = MultiObserver(recorder, observer)
    else:
        sink = recorder
    result = solve_task(
        task,
        search=search,
        heuristic=heuristic,
        observer=sink,
        max_expansions=max_expansions,
        time_limit=time_limit,
        **planner_kwargs,
    )
    return result, recorder.trace


def validate_plan(task: Task, plan) -> bool:
    """Return ``True`` iff applying ``plan`` from the initial state reaches the goal.

    Replays through the task rather than the raw operators, so derived
    predicates are closed and numeric fluents are carried at every step —
    validating against ``task.init`` directly would see neither.
    """
    state = task.initial_state()
    for op in plan or ():
        if not op.applicable(state):
            return False
        state = task.apply(op, state)
    return task.goal_reached(state)
