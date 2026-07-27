"""The zero-dependency terminal dashboard."""

from __future__ import annotations

import io

from conftest import paths
from jupyddl import build_task, solve_task
from jupyddl.live import TerminalDashboard, sparkline


class FakeTTY(io.StringIO):
    """A StringIO that claims to be interactive, so ANSI painting is exercised."""

    def isatty(self):
        return True


def test_sparkline_shapes():
    assert sparkline([]) == ""
    assert sparkline([5, 5, 5]) == "▄▄▄"
    line = sparkline([0, 1, 2, 3, 4, 5, 6, 7])
    assert line[0] == "▁" and line[-1] == "█"


def test_sparkline_downsamples_to_width():
    line = sparkline(list(range(1000)), width=20)
    assert len(line) == 20


def test_sparkline_ignores_infinities():
    assert sparkline([float("inf"), 1, 2]) == sparkline([1, 2])


def test_dashboard_on_a_plain_stream(examples_available):
    """A non-tty must not emit escape codes — this is the CI/notebook path."""
    stream = io.StringIO()
    task = build_task(*paths("tsp"))
    result = solve_task(
        task, "astar", "hmax", observer=TerminalDashboard(stream=stream, interval=0)
    )
    output = stream.getvalue()
    assert result.solved
    assert "\x1b[" not in output
    assert "solved" in output


def test_dashboard_on_a_tty_paints_and_finishes(examples_available):
    stream = FakeTTY()
    task = build_task(*paths("pallet"))
    result = solve_task(
        task, "astar", "hmax", observer=TerminalDashboard(stream=stream, interval=0)
    )
    output = stream.getvalue()
    assert result.solved
    assert "\x1b[" in output  # cursor movement / colour
    assert "expanded" in output
    assert "astar/hmax" in output
    assert "✔ solved" in output


def test_dashboard_reports_failure(examples_available):
    stream = io.StringIO()
    task = build_task(*paths("vehicle"))
    result = solve_task(task, "bfs", None, observer=TerminalDashboard(stream=stream))
    assert not result.solved
    assert "no plan found" in stream.getvalue()


def test_dashboard_history_is_bounded(examples_available):
    """Long searches must not grow the dashboard's buffers without limit."""
    stream = io.StringIO()
    dashboard = TerminalDashboard(stream=stream, history=25, interval=0)
    task = build_task(*paths("pallet"))
    solve_task(task, "bfs", None, observer=dashboard)
    assert len(dashboard._f) <= 25
    assert len(dashboard._open) <= 25


def test_dashboard_does_not_change_the_result(examples_available):
    task = build_task(*paths("pallet"))
    plain = solve_task(task, "astar", "hmax")
    watched = solve_task(
        task,
        "astar",
        "hmax",
        observer=TerminalDashboard(stream=io.StringIO(), interval=0),
    )
    assert plain.cost == watched.cost
    assert plain.stats.expanded == watched.stats.expanded
