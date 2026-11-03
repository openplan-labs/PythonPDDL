"""Benchmark harness tests."""

from __future__ import annotations

import csv

from jupyddl.benchmark import (
    discover_instances,
    run_benchmark,
    summarize,
    to_csv,
)

from conftest import EXAMPLES


def test_discover_instances(examples_available):
    names = {inst.name for inst in discover_instances(EXAMPLES)}
    assert {"blocksworld", "dinner", "tsp"} <= names


def test_run_benchmark_and_csv(tmp_path, examples_available):
    instances = [
        i
        for i in discover_instances(EXAMPLES)
        if i.name in ("dinner", "tsp", "grid", "vehicle")
    ]
    configs = [("astar", "lmcut"), ("gbfs", "hff")]
    rows = run_benchmark(instances, configs)
    assert len(rows) == len(instances) * len(configs)

    by_key = {(r.instance, r.planner): r for r in rows}
    assert by_key[("tsp", "astar")].valid
    assert by_key[("tsp", "astar")].cost == 15
    # grid is unsupported -> recorded with an error, not solved.
    assert by_key[("grid", "astar")].error
    assert not by_key[("grid", "astar")].solved
    # vehicle is unsolvable -> solved False but no crash.
    assert not by_key[("vehicle", "astar")].solved

    out = tmp_path / "results.csv"
    to_csv(rows, str(out))
    with open(out, newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert len(loaded) == len(rows)


def test_summarize_coverage(examples_available):
    instances = [i for i in discover_instances(EXAMPLES) if i.name == "tsp"]
    rows = run_benchmark(instances, [("bfs", None)])
    summary = summarize(rows)
    assert summary["bfs"]["coverage"] == 1


def _demos(*names):
    from conftest import DEMOS

    return [i for i in discover_instances(DEMOS) if i.name in names]


def test_planner_arguments_keep_configurations_apart():
    """Keyed on ``planner/heuristic`` alone, ``wastar`` at weights 1 and 5
    merged into one entry reporting four instances for a two-instance set."""
    instances = _demos("gripper", "hanoi")
    configs = [
        ("wastar", "hff", {"weight": 1.0}),
        ("wastar", "hff", {"weight": 5.0}),
    ]
    rows = run_benchmark(instances, configs, max_expansions=20000)
    assert {r.config for r in rows} == {
        "wastar/hff(weight=1.0)",
        "wastar/hff(weight=5.0)",
    }
    summary = summarize(rows)
    assert set(summary) == {"wastar/hff(weight=1.0)", "wastar/hff(weight=5.0)"}
    assert all(agg["instances"] == len(instances) for agg in summary.values())


def test_the_recorded_heuristic_is_the_one_that_ran():
    """``("gbfs", None)`` runs ``hff`` and used to be recorded as ``""``."""
    rows = run_benchmark(
        _demos("hanoi"),
        [("gbfs", None), ("astar", None), ("gbfs", "none"), ("bfs", None)],
        max_expansions=20000,
    )
    assert [(r.planner, r.heuristic) for r in rows] == [
        ("gbfs", "hff"),
        ("astar", "lmcut"),
        ("gbfs", "blind"),
        ("bfs", ""),
    ]
    assert all(r.solved for r in rows)


def test_a_repeated_configuration_is_not_merged():
    rows = run_benchmark(_demos("hanoi"), [("gbfs", "hff"), ("gbfs", None)])
    assert [r.config for r in rows] == ["gbfs/hff", "gbfs/hff#2"]
    assert all(agg["instances"] == 1 for agg in summarize(rows).values())


def test_a_build_failure_is_labelled_like_a_run(tmp_path):
    from jupyddl.benchmark import Instance

    broken = tmp_path / "broken.pddl"
    broken.write_text("(define (domain", encoding="utf-8")
    rows = run_benchmark(
        [Instance("broken", str(broken), str(broken))],
        [("wastar", None, {"weight": 3})],
    )
    assert rows[0].error
    assert (rows[0].heuristic, rows[0].config) == ("hff", "wastar/hff(weight=3)")
