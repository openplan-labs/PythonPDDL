"""CLI surface added alongside the visualisation work."""

from __future__ import annotations

import json
import os

import pytest

from conftest import demo_paths, paths
from jupyddl.cli import main


def _written(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def test_solve_writes_a_trace(examples_available, tmp_path, capsys):
    out = tmp_path / "trace.json"
    code = main(
        [
            "solve",
            *paths("tsp"),
            "-s",
            "astar",
            "-H",
            "hmax",
            "--trace",
            str(out),
            "--quiet",
        ]
    )
    assert code == 0
    assert _written(out)
    payload = json.loads(out.read_text())
    assert payload["format"] == "jupyddl.trace/1"
    assert payload["planner"] == "astar"
    assert payload["solved"] is True
    assert payload["events"]
    assert "Valid: True" in capsys.readouterr().out


def test_solve_live_does_not_break_the_exit_code(examples_available, capsys):
    code = main(
        ["solve", *paths("tsp"), "-s", "bfs", "-H", "none", "--live", "--quiet"]
    )
    assert code == 0


def test_solve_writes_charts(examples_available, tmp_path):
    pytest.importorskip("matplotlib", reason="needs the viz extra")
    progress = tmp_path / "p.png"
    tree = tmp_path / "t.png"
    plan = tmp_path / "plan.png"
    code = main(
        [
            "solve",
            *paths("pallet"),
            "-s",
            "astar",
            "-H",
            "hmax",
            "--plot",
            str(progress),
            "--tree",
            str(tree),
            "--plan-plot",
            str(plan),
            "--quiet",
        ]
    )
    assert code == 0
    assert _written(progress) and _written(tree) and _written(plan)


def test_solve_unsolvable_returns_one(examples_available, capsys):
    code = main(["solve", *paths("vehicle"), "-s", "bfs", "-H", "none", "--quiet"])
    assert code == 1
    assert "No plan found." in capsys.readouterr().out


def test_benchmark_dashboard(examples_available, tmp_path, capsys):
    pytest.importorskip("matplotlib", reason="needs the viz extra")
    dashboard = tmp_path / "dash.png"
    csv = tmp_path / "rows.csv"
    code = main(
        [
            "benchmark",
            examples_available,
            "--planners",
            "astar,bfs",
            "--heuristic",
            "hmax",
            "--csv",
            str(csv),
            "--dashboard",
            str(dashboard),
        ]
    )
    assert code == 0
    assert _written(dashboard) and _written(csv)
    assert "coverage" in capsys.readouterr().out


def test_animate_writes_a_gif(tmp_path, capsys):
    pytest.importorskip("matplotlib", reason="needs the viz extra")
    out = tmp_path / "search.gif"
    code = main(
        [
            "animate",
            *demo_paths("hanoi"),
            "-o",
            str(out),
            "-s",
            "astar",
            "-H",
            "hmax",
            "--fps",
            "4",
            "--seconds",
            "0.5",
        ]
    )
    assert code == 0
    assert _written(out)


def test_demo_builds_a_gallery(tmp_path, capsys):
    pytest.importorskip("matplotlib", reason="needs the viz extra")
    # One small instance is enough to exercise the whole gallery path.
    root = tmp_path / "instances" / "hanoi"
    root.mkdir(parents=True)
    domain, problem = demo_paths("hanoi")
    (root / "domain.pddl").write_text(open(domain).read())
    (root / "problem.pddl").write_text(open(problem).read())

    gallery = tmp_path / "gallery"
    code = main(["demo", "--root", str(tmp_path / "instances"), "-o", str(gallery)])
    assert code == 0
    produced = os.listdir(gallery)
    assert any(name.endswith("-progress.png") for name in produced)
    assert any(name.endswith("-compare.png") for name in produced)
    assert "benchmark.csv" in produced
    assert "benchmark.png" in produced


def test_benchmark_reports_missing_folder(tmp_path, capsys):
    code = main(["benchmark", str(tmp_path / "nothing-here")])
    assert code == 1
    assert "No instances found" in capsys.readouterr().err
