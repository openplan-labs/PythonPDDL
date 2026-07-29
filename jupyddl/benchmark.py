"""Comparative benchmarking of planners/heuristics over PDDL instances.

Example::

    from jupyddl.benchmark import discover_instances, run_benchmark, to_csv
    rows = run_benchmark(discover_instances("pddl-examples"),
                         [("astar", "lmcut"), ("gbfs", "hff")])
    to_csv(rows, "results.csv")
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import asdict, dataclass
from typing import Optional

from .api import build_task, solve_task, validate_plan
from .parser import PDDLError


@dataclass
class Instance:
    name: str
    domain: str
    problem: str


@dataclass
class BenchmarkRow:
    instance: str
    planner: str
    heuristic: str
    solved: bool
    valid: bool
    cost: Optional[int]
    plan_length: Optional[int]
    expanded: int
    generated: int
    evaluated: int
    runtime: float
    error: str = ""
    truncated: bool = False


def discover_instances(root: str) -> list:
    """Find ``<root>/*/`` folders containing both ``domain.pddl`` and ``problem.pddl``."""
    instances = []
    for domain in sorted(glob.glob(os.path.join(root, "*", "domain.pddl"))):
        folder = os.path.dirname(domain)
        problem = os.path.join(folder, "problem.pddl")
        if os.path.exists(problem):
            instances.append(Instance(os.path.basename(folder), domain, problem))
    return instances


def run_benchmark(instances, configs, max_expansions=None, time_limit=None) -> list:
    """Run each ``(planner, heuristic[, kwargs])`` config on each instance.

    ``configs`` entries are ``(planner_name, heuristic_name_or_None)`` or
    ``(planner_name, heuristic_name_or_None, planner_kwargs)``.

    ``max_expansions`` and ``time_limit`` bound every individual run, which is
    what keeps one pathological instance from stalling a whole benchmark. A run
    that stops on its budget is recorded with ``truncated=True`` and does not
    count towards coverage -- "we stopped looking" is not "no plan exists".
    """
    rows: list = []
    for inst in instances:
        try:
            task = build_task(inst.domain, inst.problem)
        except (PDDLError, ValueError) as exc:
            for cfg in configs:
                planner, heuristic = cfg[0], (cfg[1] or "")
                rows.append(
                    BenchmarkRow(
                        inst.name,
                        planner,
                        heuristic,
                        False,
                        False,
                        None,
                        None,
                        0,
                        0,
                        0,
                        0.0,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
            continue
        for cfg in configs:
            planner = cfg[0]
            heuristic = cfg[1]
            kwargs = cfg[2] if len(cfg) > 2 else {}
            try:
                result = solve_task(
                    task,
                    planner,
                    heuristic,
                    max_expansions=max_expansions,
                    time_limit=time_limit,
                    **kwargs,
                )
                valid = bool(result.solved and validate_plan(task, result.plan))
                rows.append(
                    BenchmarkRow(
                        inst.name,
                        planner,
                        heuristic or "",
                        result.solved,
                        valid,
                        result.cost,
                        result.plan_length,
                        result.stats.expanded,
                        result.stats.generated,
                        result.stats.evaluated,
                        round(result.stats.runtime, 6),
                        "",
                        result.stats.truncated,
                    )
                )
            except Exception as exc:  # keep the benchmark going on a single failure
                rows.append(
                    BenchmarkRow(
                        inst.name,
                        planner,
                        heuristic or "",
                        False,
                        False,
                        None,
                        None,
                        0,
                        0,
                        0,
                        0.0,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
    return rows


def to_csv(rows, path: str) -> None:
    fieldnames = (
        list(asdict(rows[0]).keys())
        if rows
        else [f.name for f in BenchmarkRow.__dataclass_fields__.values()]
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize(rows) -> dict:
    """Aggregate coverage and totals per ``planner/heuristic`` configuration."""
    summary: dict = {}
    for row in rows:
        key = f"{row.planner}/{row.heuristic}" if row.heuristic else row.planner
        agg = summary.setdefault(
            key, {"coverage": 0, "expanded": 0, "runtime": 0.0, "instances": 0}
        )
        agg["instances"] += 1
        agg["coverage"] += int(row.valid)
        agg["expanded"] += row.expanded
        agg["runtime"] += row.runtime
    return summary


def plot_summary(rows, path: str, metric: str = "expanded") -> None:
    """Bar chart of a metric per configuration (requires the ``viz`` extra)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = summarize(rows)
    labels = list(summary.keys())
    values = [summary[k][metric] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel(f"total {metric}")
    ax.set_title(f"Planner comparison ({metric})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
