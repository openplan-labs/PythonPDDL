"""Charts, live dashboards and animations for jupyddl searches.

Everything here needs the ``viz`` extra (``pip install 'jupyddl[viz]'``); the
core framework stays dependency-free. The zero-dependency terminal dashboard
lives in :mod:`jupyddl.live` instead, so watching a search never requires
matplotlib.

::

    from jupyddl import build_task, trace_search
    from jupyddl.viz import plot_search_progress, plot_search_tree

    task = build_task("demos/gripper/domain.pddl", "demos/gripper/problem.pddl")
    result, trace = trace_search(task, "astar", "lmcut")
    plot_search_progress(trace, "progress.png")
    plot_search_tree(trace, "tree.png", dark=True)
"""

from __future__ import annotations

from .live import LiveSearchPlot, animate_search
from .plots import (
    plot_benchmark_dashboard,
    plot_plan_timeline,
    plot_planner_comparison,
    plot_search_progress,
    plot_search_tree,
)
from .theme import DARK, LIGHT, palette, rc_params, sequential, series_color

__all__ = [
    "plot_search_progress",
    "plot_planner_comparison",
    "plot_search_tree",
    "plot_benchmark_dashboard",
    "plot_plan_timeline",
    "LiveSearchPlot",
    "animate_search",
    "palette",
    "rc_params",
    "series_color",
    "sequential",
    "LIGHT",
    "DARK",
]
