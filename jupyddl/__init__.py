"""jupyddl: a pure-Python PDDL planning framework.

Quickstart::

    from jupyddl import solve, build_task, trace_search, validate_plan

    result = solve("domain.pddl", "problem.pddl", search="astar", heuristic="lmcut")
    print(result.solved, result.cost, result.plan_names())

    # ...and watch the search itself
    task = build_task("domain.pddl", "problem.pddl")
    result, trace = trace_search(task, "astar", "lmcut")
    trace.save("run.json")
"""

from __future__ import annotations

from .api import build_task, solve, solve_task, trace_search, validate_plan
from .grounding import ground, ground_files
from .heuristics import HEURISTICS, make_heuristic
from .parser import PDDLError, UnsupportedFeatureError, parse
from .search import PLANNERS, SearchResult, make_planner
from .task import Operator, Task
from .trace import (
    MultiObserver,
    SearchEvent,
    SearchObserver,
    SearchTrace,
    TraceRecorder,
)

__version__ = "2.3.0"

__all__ = [
    "solve",
    "solve_task",
    "build_task",
    "trace_search",
    "validate_plan",
    "ground",
    "ground_files",
    "make_planner",
    "make_heuristic",
    "PLANNERS",
    "HEURISTICS",
    "SearchResult",
    "SearchTrace",
    "SearchEvent",
    "SearchObserver",
    "MultiObserver",
    "TraceRecorder",
    "Task",
    "Operator",
    "parse",
    "PDDLError",
    "UnsupportedFeatureError",
    "__version__",
]
