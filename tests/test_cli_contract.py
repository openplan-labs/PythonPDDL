"""What the command line promises a script: exit codes and one-line errors.

Every failure used to exit 1 -- unsolvable, out of budget and a crash alike --
and bad input came out as a Python traceback.
"""

from __future__ import annotations

import json
import os
import runpy
import sys

import pytest

from jupyddl import __version__, build_task, trace_search
from jupyddl.cli import (
    EXIT_BAD_INPUT,
    EXIT_NO_PLAN,
    EXIT_SOLVED,
    EXIT_TRUNCATED,
    main,
)

from conftest import REPO_ROOT, paths


def demo(name):
    folder = os.path.join(REPO_ROOT, "demos", name)
    return os.path.join(folder, "domain.pddl"), os.path.join(folder, "problem.pddl")


def test_solved_is_zero_and_unsolvable_is_one(examples_available):
    assert main(["solve", *paths("blocksworld"), "--quiet"]) == EXIT_SOLVED
    assert main(["solve", *paths("vehicle"), "-s", "bfs"]) == EXIT_NO_PLAN


def test_a_budget_stop_has_its_own_exit_code(capsys):
    code = main(
        ["solve", *demo("blocksworld12"), "-s", "bfs", "--max-expansions", "20"]
    )
    assert code == EXIT_TRUNCATED
    assert "within the budget" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["solve", "no/such/domain.pddl", "no/such/problem.pddl"],
        ["solve", REPO_ROOT, REPO_ROOT],  # directories, not files
        ["benchmark", "demos", "--planners", "astar,nosuch"],
        ["benchmark", "demos", "--heuristic", "nosuch"],
        ["learn", "blocksworld", "--sizes", "abc"],
    ],
)
def test_bad_input_is_one_line_and_exit_two(argv, capsys, monkeypatch):
    monkeypatch.delenv("JUPYDDL_DEBUG", raising=False)
    assert main(argv) == EXIT_BAD_INPUT
    err = capsys.readouterr().err
    assert err.startswith("jupyddl: error:") and "Traceback" not in err


def test_a_parse_error_is_reported_not_raised(tmp_path, capsys):
    domain = tmp_path / "domain.pddl"
    domain.write_text("(define (domain d) (:predicates (p))")  # unbalanced
    problem = tmp_path / "problem.pddl"
    problem.write_text("(define (problem p) (:domain d) (:init) (:goal (p)))")
    assert main(["solve", str(domain), str(problem)]) == EXIT_BAD_INPUT
    assert "jupyddl: error:" in capsys.readouterr().err


def test_debug_mode_keeps_the_traceback(monkeypatch):
    monkeypatch.setenv("JUPYDDL_DEBUG", "1")
    with pytest.raises(OSError):
        main(["solve", "no/such/domain.pddl", "no/such/problem.pddl"])


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_charts_without_matplotlib_fail_before_searching(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # import now fails
    code = main(["solve", *demo("blocksworld12"), "-s", "bfs", "--plot", "x.png"])
    assert code == EXIT_BAD_INPUT  # and quickly: bfs here would take minutes
    assert "viz extra" in capsys.readouterr().err


def test_a_trace_records_the_plan_the_user_sees():
    task = build_task(*demo("errands"))
    result, trace = trace_search(task, "astar", "lmcut")
    visible = [op.base_name for op in task.visible_plan(result.plan)]
    assert trace.plan == visible
    assert not any(name.startswith("__") or "#" in name for name in trace.plan)


def test_the_workbench_streams_a_dead_end_as_valid_json():
    namespace = runpy.run_path(os.path.join(REPO_ROOT, "web", "bootstrap.py"))
    sent = []
    observer = namespace["WebObserver"](sent.append, batch=1)
    observer.on_expand("s", g=1.0, h=float("inf"), f=float("inf"))
    observer.flush()
    for message in sent:
        json.loads(message, parse_constant=lambda name: pytest.fail(name))
