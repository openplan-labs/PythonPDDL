"""Charts, the theme and the animation writer.

Rendering is checked for "it produced a real image without raising", not for
pixels: the point is that every chart survives the awkward inputs it will meet
(empty traces, unsolved runs, a single configuration, log scales).
"""

from __future__ import annotations

import os

import pytest

from conftest import paths
from jupyddl import build_task, trace_search
from jupyddl.benchmark import discover_instances, run_benchmark
from jupyddl.trace import SearchTrace

matplotlib = pytest.importorskip("matplotlib", reason="needs the viz extra")

from jupyddl.viz import (  # noqa: E402
    animate_search,
    plot_benchmark_dashboard,
    plot_plan_timeline,
    plot_planner_comparison,
    plot_search_progress,
    plot_search_tree,
)
from jupyddl.viz.theme import palette, rc_params, sequential, series_color  # noqa: E402


@pytest.fixture(scope="module")
def trace(examples_available):
    task = build_task(*paths("pallet"))
    _, recorded = trace_search(task, "astar", "hmax")
    return recorded


@pytest.fixture(scope="module")
def traces(examples_available):
    task = build_task(*paths("pallet"))
    out = []
    for planner, heuristic in [("astar", "hmax"), ("gbfs", "hff"), ("bfs", None)]:
        _, recorded = trace_search(task, planner, heuristic)
        out.append(recorded)
    return out


def _written(path):
    return os.path.exists(path) and os.path.getsize(path) > 1000


# ---------------------------------------------------------------- the theme
def test_palette_modes_are_distinct():
    light, dark = palette(False), palette(True)
    assert light["surface"] != dark["surface"]
    assert light["series"] != dark["series"]
    assert len(light["series"]) == len(dark["series"]) == 8


def test_series_colours_are_stable_and_never_run_out():
    assert series_color(0) == series_color(0)
    assert series_color(0) != series_color(1)
    assert series_color(99)  # wraps rather than raising


def test_sequential_ramp_is_ordered_and_clamped():
    assert sequential(0.0) != sequential(1.0)
    assert sequential(-5) == sequential(0.0)
    assert sequential(5) == sequential(1.0)
    assert sequential(float("nan")) == sequential(0.0)


def test_rc_params_cover_both_modes():
    for dark in (False, True):
        params = rc_params(dark)
        assert params["axes.facecolor"] == palette(dark)["surface"]
        assert params["lines.linewidth"] == 2.0


# --------------------------------------------------------------- the charts
@pytest.mark.parametrize("dark", [False, True])
def test_search_progress(trace, tmp_path, dark):
    path = tmp_path / f"progress-{dark}.png"
    plot_search_progress(trace, str(path), dark=dark)
    assert _written(path)


@pytest.mark.parametrize("dark", [False, True])
def test_search_tree(trace, tmp_path, dark):
    path = tmp_path / f"tree-{dark}.png"
    plot_search_tree(trace, str(path), dark=dark)
    assert _written(path)


@pytest.mark.parametrize("dark", [False, True])
def test_plan_timeline(trace, tmp_path, dark):
    path = tmp_path / f"plan-{dark}.png"
    plot_plan_timeline(trace, str(path), dark=dark)
    assert _written(path)


@pytest.mark.parametrize("dark", [False, True])
def test_planner_comparison(traces, tmp_path, dark):
    path = tmp_path / f"compare-{dark}.png"
    plot_planner_comparison(traces, str(path), dark=dark)
    assert _written(path)


def test_comparison_with_a_single_trace(trace, tmp_path):
    path = tmp_path / "one.png"
    plot_planner_comparison([trace], str(path))
    assert _written(path)


def test_plan_timeline_truncates_long_plans(trace, tmp_path):
    path = tmp_path / "short.png"
    plot_plan_timeline(trace, str(path), max_steps=2)
    assert _written(path)


def test_charts_survive_an_empty_trace(tmp_path):
    empty = SearchTrace(planner="astar", heuristic="lmcut", task_name="nothing")
    plot_search_progress(empty, str(tmp_path / "e1.png"))
    plot_search_tree(empty, str(tmp_path / "e2.png"))
    plot_plan_timeline(empty, str(tmp_path / "e3.png"))
    assert _written(tmp_path / "e1.png")
    assert _written(tmp_path / "e3.png")


def test_charts_survive_an_unsolved_search(examples_available, tmp_path):
    task = build_task(*paths("vehicle"))
    _, unsolved = trace_search(task, "bfs", None)
    plot_search_progress(unsolved, str(tmp_path / "u1.png"))
    plot_planner_comparison([unsolved], str(tmp_path / "u2.png"))
    assert _written(tmp_path / "u1.png")
    assert _written(tmp_path / "u2.png")


@pytest.mark.parametrize("dark", [False, True])
def test_benchmark_dashboard(examples_available, tmp_path, dark):
    instances = discover_instances(examples_available)[:4]
    rows = run_benchmark(instances, [("astar", "hmax"), ("bfs", None)])
    path = tmp_path / f"bench-{dark}.png"
    plot_benchmark_dashboard(rows, str(path), dark=dark)
    assert _written(path)


def test_benchmark_dashboard_marks_failures(examples_available, tmp_path):
    """Unsupported and unsolvable instances become explicit gaps, not blanks."""
    instances = [
        inst
        for inst in discover_instances(examples_available)
        if inst.name in {"grid", "vehicle", "tsp"}
    ]
    rows = run_benchmark(instances, [("astar", "hmax")])
    assert any(not row.valid for row in rows)
    path = tmp_path / "gaps.png"
    plot_benchmark_dashboard(rows, str(path))
    assert _written(path)


# ------------------------------------------------------------- the animation
def test_animate_to_gif(trace, tmp_path):
    path = tmp_path / "search.gif"
    animate_search(trace, str(path), fps=4, seconds=0.5, dpi=50)
    assert _written(path)


def test_animate_rejects_an_empty_trace(tmp_path):
    empty = SearchTrace(planner="bfs", task_name="nothing")
    with pytest.raises(ValueError):
        animate_search(empty, str(tmp_path / "nope.gif"), fps=2, seconds=0.5)


def test_live_plot_collects_without_a_display(examples_available, tmp_path):
    from jupyddl import solve_task
    from jupyddl.viz import LiveSearchPlot

    task = build_task(*paths("pallet"))
    live = LiveSearchPlot(every=5)
    result = solve_task(task, "astar", "hmax", observer=live)
    assert result.solved
    path = tmp_path / "live.png"
    live.save(str(path))
    assert _written(path)
