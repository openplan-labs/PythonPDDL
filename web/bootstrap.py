"""Python side of the workbench, executed inside Pyodide by ``worker.js``.

Kept as a real ``.py`` file rather than a string embedded in JavaScript so it
stays readable, lintable, and free of template-literal escaping hazards.

``worker.js`` fetches this file, runs it once after the jupyddl sources are in
place, and then calls the ``run_*`` entry points below. Everything crossing the
JS boundary is JSON, so no proxy objects leak out.
"""

import json
import os
import sys
import tempfile
import time

import jupyddl
from jupyddl import build_task, solve_task, validate_plan
from jupyddl.trace import SearchObserver


class WebObserver(SearchObserver):
    """Stream batched search progress to the page.

    Points are buffered and flushed every 'batch' expansions; 'stride' doubles
    once 'budget' samples have been sent, so a search of 200 nodes and one of
    200,000 both stream a bounded number of points and the browser never falls
    behind the planner.
    """

    def __init__(self, emit, batch=24, budget=900):
        self.emit = emit
        self.batch = batch
        self.budget = budget
        self.buffer = []
        self.count = 0
        self.kept = 0
        self.stride = 1
        self.peak_open = 0

    def on_expand(self, state, g=0.0, h=0.0, f=0.0, depth=0, open_size=0,
                  stats=None, parent=None, action=""):
        self.count += 1
        self.peak_open = max(self.peak_open, open_size)
        if self.count % self.stride:
            return
        self.kept += 1
        hv = 0.0 if h is None or h != h else float(h)
        self.buffer.append([
            self.count,
            float(g),
            hv,
            float(f if f else g + hv),
            int(depth),
            int(open_size),
            int(stats.generated) if stats else 0,
            int(stats.evaluated) if stats else 0,
        ])
        if len(self.buffer) >= self.batch:
            self.flush()
        if self.kept >= self.budget:
            self.stride *= 2
            self.kept = self.kept // 2

    def flush(self):
        if self.buffer:
            self.emit(json.dumps({"points": self.buffer, "peak_open": self.peak_open}))
            self.buffer = []


def _task_info(task):
    """Everything the UI shows about a grounded task."""
    return {
        "name": task.name,
        "facts": task.num_facts,
        "operators": len(task.operators),
        "goals": len(task.goals),
        "axioms": len(task.axioms),
        "numeric": list(task.numeric_names),
        "temporal": bool(task.temporal),
        "requirements": list(task.requirements),
        "metric": task.metric or "",
    }


def _ground(domain_text, problem_text):
    """Ground a task from in-memory PDDL text."""
    with tempfile.TemporaryDirectory() as folder:
        domain_path = os.path.join(folder, "domain.pddl")
        problem_path = os.path.join(folder, "problem.pddl")
        with open(domain_path, "w") as handle:
            handle.write(domain_text)
        with open(problem_path, "w") as handle:
            handle.write(problem_text)
        return build_task(domain_path, problem_path)


def _plan_payload(task, result):
    """Canonical action names, with the compilation-only operators removed."""
    return [op.base_name for op in task.visible_plan(result.plan)]


def run_solve(domain_text, problem_text, planner, heuristic, weight, limits, emit):
    """Ground and solve, streaming progress. Returns a JSON summary."""
    options = json.loads(limits) if limits else {}
    started = time.perf_counter()
    task = _ground(domain_text, problem_text)
    ground_seconds = time.perf_counter() - started

    observer = WebObserver(emit)
    kwargs = {"weight": weight} if planner == "wastar" else {}
    heur = None if heuristic in ("none", "", None) else heuristic
    result = solve_task(
        task,
        planner,
        heur,
        observer=observer,
        max_expansions=options.get("max_expansions"),
        time_limit=options.get("time_limit"),
        **kwargs,
    )
    observer.flush()

    valid = bool(result.solved and validate_plan(task, result.plan))
    return json.dumps({
        "task": _task_info(task),
        "ground_seconds": ground_seconds,
        "solved": bool(result.solved),
        "valid": valid,
        "truncated": bool(result.truncated),
        "cost": result.cost,
        "makespan": task.makespan(result.plan) if task.temporal else None,
        "plan": _plan_payload(task, result),
        "stats": result.stats.as_dict(),
        "planner": planner,
        "heuristic": heur or "",
    })


def run_experiment(instances_json, configs_json, limits, emit):
    """Run a full instance x configuration matrix, streaming one row at a time.

    Grounding is done once per instance and shared across configurations, which
    is both faster and fairer: every configuration sees exactly the same task.
    """
    instances = json.loads(instances_json)
    configs = json.loads(configs_json)
    options = json.loads(limits) if limits else {}
    rows = []

    for instance in instances:
        label = instance.get("id") or instance.get("title") or "instance"
        try:
            started = time.perf_counter()
            task = _ground(instance["domain"], instance["problem"])
            ground_seconds = time.perf_counter() - started
            info = _task_info(task)
        except Exception as exc:
            for config in configs:
                heur = config.get("heuristic") or ""
                rows.append({
                    "instance": label,
                    "planner": config["planner"],
                    "heuristic": heur,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "solved": False, "valid": False, "truncated": False,
                    "cost": None, "length": None, "expanded": 0,
                    "generated": 0, "evaluated": 0, "runtime": 0.0,
                })
                emit(json.dumps({"row": rows[-1]}))
            continue

        for config in configs:
            planner = config["planner"]
            heur = config.get("heuristic") or None
            try:
                result = solve_task(
                    task,
                    planner,
                    heur,
                    max_expansions=options.get("max_expansions"),
                    time_limit=options.get("time_limit"),
                )
                valid = bool(result.solved and validate_plan(task, result.plan))
                row = {
                    "instance": label,
                    "planner": planner,
                    "heuristic": heur or "",
                    "solved": bool(result.solved),
                    "valid": valid,
                    "truncated": bool(result.truncated),
                    "cost": result.cost,
                    "length": len(task.visible_plan(result.plan)),
                    "makespan": task.makespan(result.plan) if task.temporal else None,
                    "expanded": result.stats.expanded,
                    "generated": result.stats.generated,
                    "evaluated": result.stats.evaluated,
                    "runtime": round(result.stats.runtime, 6),
                    "facts": info["facts"],
                    "operators": info["operators"],
                    "ground_seconds": round(ground_seconds, 6),
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "instance": label,
                    "planner": planner,
                    "heuristic": heur or "",
                    "solved": False, "valid": False, "truncated": False,
                    "cost": None, "length": None, "expanded": 0,
                    "generated": 0, "evaluated": 0, "runtime": 0.0,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            rows.append(row)
            emit(json.dumps({"row": row}))

    return json.dumps({"rows": rows})


def run_inspect(domain_text, problem_text):
    """Ground without solving, to report what the instance actually contains."""
    task = _ground(domain_text, problem_text)
    sample_ops = [op.base_name for op in task.operators[:12]]
    return json.dumps({
        "task": _task_info(task),
        "sample_operators": sample_ops,
        "init_size": len(task.init),
    })


def run_generate(kind, size, seed, extra):
    """Generate an instance from the reproducible generators."""
    from jupyddl.generator import generate

    kwargs = json.loads(extra) if extra else {}
    domain, problem = generate(kind, size=int(size), seed=int(seed), **kwargs)
    return json.dumps({"domain": domain, "problem": problem})


def describe():
    """Report the runtime and everything the UI needs to populate its menus."""
    from jupyddl.generator import describe_generators
    from jupyddl.heuristics import HEURISTICS
    from jupyddl.requirements import as_rows, summary
    from jupyddl.search import describe_planners

    return json.dumps({
        "version": jupyddl.__version__,
        "planners": describe_planners(),
        "heuristics": sorted(HEURISTICS),
        "requirements": as_rows(),
        "requirement_summary": summary(),
        "generators": describe_generators(),
        "python": sys.version.split()[0],
    })
