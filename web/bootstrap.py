"""Python side of the playground, executed inside Pyodide by ``worker.js``.

Kept as a real ``.py`` file rather than a string embedded in JavaScript so it
stays readable, lintable, and free of template-literal escaping hazards.

``worker.js`` fetches this file, runs it once after the jupyddl sources are in
place, and then calls :func:`run_solve` / :func:`run_race` / :func:`describe`.
Everything crossing the JS boundary is JSON, so no proxy objects leak out.
"""

import json
import os
import sys
import tempfile

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
    return {
        "name": task.name,
        "facts": task.num_facts,
        "operators": len(task.operators),
        "goals": len(task.goals),
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


def run_solve(domain_text, problem_text, planner, heuristic, weight, emit):
    """Ground and solve, streaming progress. Returns a JSON summary."""
    task = _ground(domain_text, problem_text)
    observer = WebObserver(emit)
    kwargs = {"weight": weight} if planner == "wastar" else {}
    heur = None if heuristic in ("none", "", None) else heuristic
    result = solve_task(task, planner, heur, observer=observer, **kwargs)
    observer.flush()

    valid = bool(result.solved and validate_plan(task, result.plan))
    return json.dumps({
        "task": _task_info(task),
        "solved": bool(result.solved),
        "valid": valid,
        "cost": result.cost,
        "plan": result.plan_names() or [],
        "stats": result.stats.as_dict(),
        "planner": planner,
        "heuristic": heur or "",
    })


def run_race(domain_text, problem_text, configs_json):
    """Run several configurations against one grounding; returns their stats."""
    configs = json.loads(configs_json)
    task = _ground(domain_text, problem_text)

    results = []
    for config in configs:
        planner = config["planner"]
        heur = config.get("heuristic") or None
        label = "{}/{}".format(planner, heur) if heur else planner
        try:
            result = solve_task(task, planner, heur)
            valid = bool(result.solved and validate_plan(task, result.plan))
            results.append({
                "label": label,
                "solved": bool(result.solved),
                "valid": valid,
                "cost": result.cost,
                "length": result.plan_length,
                "stats": result.stats.as_dict(),
                "error": "",
            })
        except Exception as exc:  # one bad config must not sink the race
            results.append({
                "label": label,
                "solved": False,
                "valid": False,
                "cost": None,
                "length": None,
                "stats": {},
                "error": "{}: {}".format(type(exc).__name__, exc),
            })
    return json.dumps({"task": _task_info(task), "results": results})


def describe():
    """Report the runtime and the available planners/heuristics to the page."""
    from jupyddl.heuristics import HEURISTICS
    from jupyddl.search import PLANNERS

    return json.dumps({
        "version": jupyddl.__version__,
        "planners": sorted(PLANNERS),
        "heuristics": sorted(HEURISTICS),
        "python": sys.version.split()[0],
    })
