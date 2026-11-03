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

from .api import build_task, resolve_heuristic, solve_task, validate_plan
from .parser import PDDLError
from .search import make_planner


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
    #: The configuration, planner arguments included (``wastar/hff(weight=5)``).
    #: ``planner`` and ``heuristic`` alone cannot tell two weights apart.
    config: str = ""


def discover_instances(root: str) -> list:
    """Find ``<root>/*/`` folders containing both ``domain.pddl`` and ``problem.pddl``."""
    instances = []
    for domain in sorted(glob.glob(os.path.join(root, "*", "domain.pddl"))):
        folder = os.path.dirname(domain)
        problem = os.path.join(folder, "problem.pddl")
        if os.path.exists(problem):
            instances.append(Instance(os.path.basename(folder), domain, problem))
    return instances


def config_label(planner: str, heuristic: str = "", kwargs=None) -> str:
    """Name a configuration: ``planner/heuristic``, plus any planner arguments.

    >>> config_label("wastar", "hff", {"weight": 5})
    'wastar/hff(weight=5)'
    >>> config_label("bfs")
    'bfs'
    """
    label = f"{planner}/{heuristic}" if heuristic else planner
    if kwargs:
        label += "(" + ",".join(f"{k}={v}" for k, v in sorted(kwargs.items())) + ")"
    return label


def row_label(row) -> str:
    """The configuration a row belongs to, for rows that predate ``config``."""
    return row.config or config_label(row.planner, row.heuristic)


def _heuristic_label(heuristic) -> str:
    if heuristic is None:
        return ""
    if isinstance(heuristic, str):
        return heuristic
    return getattr(heuristic, "name", None) or type(heuristic).__name__


def _plan_configs(configs) -> list:
    """Unpack each config and work out what will actually run, and its label.

    The heuristic recorded is the one :func:`~jupyddl.api.solve_task` will run,
    not the one asked for: ``("gbfs", None)`` runs ``hff`` and used to be
    recorded as ``""``, and ``"none"`` runs ``blind`` for a planner that needs
    a heuristic. Two config entries never share a label — a repeated one is
    suffixed ``#2`` — so :func:`summarize` cannot merge them.
    """
    planned = []
    seen: dict = {}
    for cfg in configs:
        planner, requested = cfg[0], cfg[1]
        kwargs = dict(cfg[2]) if len(cfg) > 2 and cfg[2] else {}
        try:
            ran = resolve_heuristic(make_planner(planner, **kwargs), requested)
        except (ValueError, TypeError):
            ran = requested  # an unknown planner or argument: the run records it
        ran = _heuristic_label(ran)
        label = config_label(planner, ran, kwargs)
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label}#{seen[label]}"
        planned.append((planner, requested, kwargs, ran, label))
    return planned


def _failed_row(instance, planner, heuristic, label, exc) -> BenchmarkRow:
    return BenchmarkRow(
        instance,
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
        config=label,
    )


def run_benchmark(instances, configs, max_expansions=None, time_limit=None) -> list:
    """Run each ``(planner, heuristic[, kwargs])`` config on each instance.

    ``configs`` entries are ``(planner_name, heuristic_name_or_None)`` or
    ``(planner_name, heuristic_name_or_None, planner_kwargs)``. Each row
    records the heuristic that actually ran and a ``config`` label that
    includes the planner arguments, so ``wastar`` at two weights stays two
    configurations.

    ``max_expansions`` and ``time_limit`` bound every individual run, which is
    what keeps one pathological instance from stalling a whole benchmark. A run
    that stops on its budget is recorded with ``truncated=True`` and does not
    count towards coverage -- "we stopped looking" is not "no plan exists".
    """
    planned = _plan_configs(configs)
    rows: list = []
    for inst in instances:
        try:
            task = build_task(inst.domain, inst.problem)
        except (PDDLError, ValueError) as exc:
            for planner, _, _, ran, label in planned:
                rows.append(_failed_row(inst.name, planner, ran, label, exc))
            continue
        for planner, requested, kwargs, ran, label in planned:
            try:
                result = solve_task(
                    task,
                    planner,
                    requested,
                    max_expansions=max_expansions,
                    time_limit=time_limit,
                    **kwargs,
                )
                valid = bool(result.solved and validate_plan(task, result.plan))
                rows.append(
                    BenchmarkRow(
                        inst.name,
                        planner,
                        ran,
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
                        label,
                    )
                )
            except Exception as exc:  # keep the benchmark going on a single failure
                rows.append(_failed_row(inst.name, planner, ran, label, exc))
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
    """Aggregate coverage and totals per configuration.

    Keyed on :func:`row_label`, which carries the planner's arguments. Keyed
    on ``planner/heuristic`` alone, ``wastar`` at weights 1 and 5 merged into
    one entry claiming twice the instances the benchmark had.
    """
    summary: dict = {}
    for row in rows:
        key = row_label(row)
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
