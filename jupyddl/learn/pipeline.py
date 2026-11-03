"""One call from a domain name to a trained heuristic.

The stages are useful separately and tedious to wire together, so this is the
assembled version: generate a ladder of instances, solve the small ones
optimally, imitate those plans, then let the reinforcement stages take over.

The default ladder deliberately trains on instances *smaller* than the ones it
will be used on. That is the whole claim being tested — a heuristic that only
works at the size it was trained on has learned the instance, not the domain —
and :func:`jupyddl.learn.pipeline.evaluate_transfer` is what checks it.
"""

from __future__ import annotations

import time
from typing import Optional

from ..api import solve_task
from ..generator import generate
from ..grounding import ground
from ..parser import parse
from ..search import make_planner
from ..search.result import make_budget
from .dataset import build_corpus
from .features import FeatureSpace
from .heuristic import HeuristicBundle
from .rl import RLConfig, bootstrap, dagger, optimise_search_cost
from .train import TrainConfig, train

__all__ = [
    "tasks_from_generator",
    "solved_corpus",
    "learn_heuristic",
    "evaluate_transfer",
]


def tasks_from_generator(kind: str, sizes, seed: int = 0, seeds_per_size: int = 1):
    """Ground a ladder of generated instances into ``(name, task)`` pairs.

    Several seeds per size is usually a better use of a training budget than
    more sizes: what varies between two 6-block instances is the goal
    structure, which is what has to be learned, whereas what varies between 6
    and 7 blocks is mostly scale, which the features already normalise away.
    """
    tasks = []
    for size in sizes:
        for offset in range(seeds_per_size):
            instance_seed = seed + offset
            domain_text, problem_text = generate(kind, size=size, seed=instance_seed)
            task = ground(parse(domain_text), parse(problem_text))
            tasks.append((f"{kind}-{size:02d}-{instance_seed}", task))
    return tasks


def solved_corpus(
    tasks,
    space: Optional[FeatureSpace] = None,
    planner: str = "astar",
    heuristic: str = "lmcut",
    optimal: bool = True,
    max_expansions: Optional[int] = 20000,
    time_limit: Optional[float] = 30.0,
    ranking: bool = True,
    seed: int = 0,
    on_instance=None,
):
    """Solve every task and turn the plans into a corpus.

    Defaults to an optimal configuration, so the targets really are ``h*``. On
    a ladder where that is out of reach, pass ``planner="gbfs"``,
    ``heuristic="hff"`` and ``optimal=False``: the labels become upper bounds,
    training down-weights them, and that is far better than an empty corpus.
    """

    def solver(task):
        return solve_task(
            task,
            search=planner,
            heuristic=heuristic,
            max_expansions=max_expansions,
            time_limit=time_limit,
        )

    return build_corpus(
        tasks,
        solver,
        space=space,
        ranking=ranking,
        optimal=optimal,
        seed=seed,
        on_instance=on_instance,
    )


def learn_heuristic(
    kind: str,
    sizes=range(3, 7),
    seeds_per_size: int = 2,
    seed: int = 0,
    train_config: Optional[TrainConfig] = None,
    rl_config: Optional[RLConfig] = None,
    dagger_rounds: int = 0,
    bootstrap_sizes=None,
    cem_iterations: int = 0,
    cem_sizes=None,
    verbose: bool = False,
):
    """Generate, solve, imitate and (optionally) reinforce. Returns a bundle.

    Every stage after imitation is off by default. They cost minutes rather
    than seconds and they are not always worth it — which one pays depends on
    the domain, and saying so honestly is more useful than a default that
    quietly triples the runtime.

    ``cem_sizes`` is the one setting worth understanding before using it. The
    direct search-cost stage needs instances with *headroom*: on the training
    ladder the imitated heuristic already expands roughly as many nodes as the
    plan is long, so every perturbation scores the same and the objective is
    flat. Measured on blocksworld, tuning on the training sizes moved the score
    from 12.8 to 12.75 — noise — while tuning on sizes 9-12 moved it from 1605
    to 64, and cut a held-out set the optimiser never saw from 336 to 122.
    Default: one rung above the training ladder.
    """
    train_config = train_config or TrainConfig(seed=seed)
    rl_config = rl_config or RLConfig(seed=seed, verbose=verbose)
    started = time.perf_counter()
    log = []

    # Materialise every size argument before anything reads it. Each is read
    # more than once below, and a generator passed in would be spent by the
    # first read: the ladder was recorded as ``[]`` and ``max()`` of it raised.
    sizes = list(sizes)
    if not sizes:
        raise ValueError("learn_heuristic needs at least one training size")
    if bootstrap_sizes is not None:
        bootstrap_sizes = list(bootstrap_sizes)
    if cem_sizes is not None:
        cem_sizes = list(cem_sizes)

    def note(message):
        log.append(message)
        if verbose:  # pragma: no cover - operator convenience
            print(message)

    tasks = tasks_from_generator(kind, sizes, seed=seed, seeds_per_size=seeds_per_size)
    note(f"ladder: {len(tasks)} instances of {kind}, sizes {sizes}")

    # The vocabulary must span every task the model will ever see, including
    # the larger ones it is only evaluated on: a symbol first encountered at
    # evaluation time would otherwise have no slot.
    if cem_iterations and cem_sizes is None:
        # A rung above the training ladder: far enough that search has room to
        # be improved, near enough that it can still be searched cheaply.
        top = max(sizes)
        cem_sizes = range(top + 2, top + 6)

    all_tasks = list(tasks)
    if bootstrap_sizes:
        all_tasks += tasks_from_generator(
            kind, bootstrap_sizes, seed=seed, seeds_per_size=1
        )
    if cem_sizes:
        # Two disjoint seed families over the same sizes: one the optimiser
        # fits, one it is only ever scored on. Without the split, a thousand
        # parameters tuned against eight instances fit those eight instances
        # and transfer gets worse, not better.
        cem_tasks = tasks_from_generator(
            kind, cem_sizes, seed=seed + 1000, seeds_per_size=2
        )
        cem_validation = tasks_from_generator(
            kind, cem_sizes, seed=seed + 2000, seeds_per_size=2
        )
        all_tasks += cem_tasks + cem_validation
    space = FeatureSpace.from_tasks(task for _, task in all_tasks)

    corpus = solved_corpus(tasks, space=space, seed=seed)
    note(f"corpus: {corpus.target_stats()}")
    if not corpus.samples:
        raise RuntimeError(
            f"no instance of '{kind}' was solved optimally within the budget; "
            "lower the sizes or pass a satisficing configuration"
        )

    bundle, report = train(corpus, train_config)
    note(
        f"imitation: {report.metrics.get('mae', 0):.2f} MAE, "
        f"top-1 {report.metrics.get('top1', 0):.3f}, "
        f"best epoch {report.best_epoch}, {report.seconds:.1f}s"
    )

    if dagger_rounds:
        bundle, corpus, history = dagger(
            bundle,
            tasks,
            corpus,
            rounds=dagger_rounds,
            config=rl_config,
            train_config=train_config,
        )
        note(f"dagger: {history[-1] if history else 'no rounds ran'}")

    if bootstrap_sizes:
        ladder = tasks_from_generator(
            kind, bootstrap_sizes, seed=seed, seeds_per_size=1
        )
        bundle, corpus, history = bootstrap(
            bundle,
            ladder,
            corpus,
            config=rl_config,
            train_config=train_config,
        )
        note(f"bootstrap: {history[-1] if history else 'no rounds ran'}")

    if cem_iterations:
        note(
            f"cem: tuning on {len(cem_tasks)} instances of sizes {list(cem_sizes)}, "
            f"selecting on {len(cem_validation)} held out from it"
        )
        bundle, history = optimise_search_cost(
            bundle,
            cem_tasks,
            iterations=cem_iterations,
            config=rl_config,
            validation_tasks=cem_validation,
        )
        note(f"cem: {history[-1] if history else 'no iterations ran'}")

    bundle.metrics["pipeline"] = {
        "kind": kind,
        "sizes": list(sizes),
        "instances": len(tasks),
        "seconds": time.perf_counter() - started,
        "log": log,
    }
    return bundle


def evaluate_transfer(
    bundle: HeuristicBundle,
    tasks,
    baselines=("hff", "goalcount", "blind"),
    planner: str = "gbfs",
    max_expansions: Optional[int] = 20000,
    time_limit: Optional[float] = 30.0,
):
    """Compare the learned heuristic against baselines on ``tasks``.

    Reports expansions *and* wall-clock, because they can disagree and the
    disagreement is the whole story. A learned heuristic that halves expansions
    while tripling time per node has not helped anyone; one that expands
    slightly more than ``hff`` but evaluates in a fraction of the time may
    still win. Only reporting expansions is how a learned heuristic gets
    published as a success without being one.
    """
    from ..heuristics import make_heuristic

    rows = []
    for name, task in tasks:
        configurations = [("learned", lambda t: bundle.bind(t))]
        configurations += [
            (base, (lambda b: lambda t: make_heuristic(b, t))(base))
            for base in baselines
        ]
        for label, factory in configurations:
            planner_instance = make_planner(planner)
            budget = make_budget(max_expansions, time_limit)
            started = time.perf_counter()
            result = planner_instance.search(task, factory(task), budget=budget)
            rows.append(
                {
                    "instance": name,
                    "heuristic": label,
                    "solved": result.solved,
                    "cost": result.cost,
                    "expanded": result.stats.expanded,
                    "evaluated": result.stats.evaluated,
                    "seconds": time.perf_counter() - started,
                    "truncated": result.truncated,
                    "budget": max_expansions,
                }
            )
    return rows


def summarise_transfer(rows, failure_penalty: float = 2.0) -> dict:
    """Per-heuristic coverage, and means that a low coverage cannot flatter.

    This used to average each heuristic over the runs *it* solved, which
    rewards giving up: a heuristic that solves only the easiest instance
    reports that instance's cost as its mean, and so looks many times better
    than one that also solved the hard ones. Now:

    * ``coverage`` is the fraction solved, and belongs next to every mean.
    * ``mean_expanded`` and ``mean_seconds`` are over **every** instance. An
      unsolved run is charged ``failure_penalty`` times the larger of the
      expansion budget it ran under (``row["budget"]``, when recorded) and the
      most any configuration spent on that instance — so failing is never
      cheaper than solving, by the same rule as
      :func:`jupyddl.learn.rl.search_cost`. Seconds are charged the same way
      against the slowest run on the instance.
    * ``mean_cost`` is over the ``common`` instances that every configuration
      with any coverage solved, since a failure has no plan cost to charge.
      It is ``None`` when there are none.

    ``solved``, ``total``, ``expanded`` and ``seconds`` are the raw sums the
    means are built from (failures charged as above).
    """
    rows = list(rows)
    instances: dict = {}
    for row in rows:
        instances.setdefault(row["instance"], []).append(row)

    summary: dict = {}
    for row in rows:
        summary.setdefault(
            row["heuristic"],
            {"solved": 0, "total": 0, "expanded": 0.0, "seconds": 0.0},
        )
    for runs in instances.values():
        most_expanded = max(
            [run["expanded"] for run in runs] + [run.get("budget") or 0 for run in runs]
        )
        slowest = max(run["seconds"] for run in runs)
        for run in runs:
            entry = summary[run["heuristic"]]
            entry["total"] += 1
            if run["solved"]:
                entry["solved"] += 1
                entry["expanded"] += run["expanded"]
                entry["seconds"] += run["seconds"]
            else:
                entry["expanded"] += failure_penalty * most_expanded
                entry["seconds"] += failure_penalty * slowest

    # Plan cost only compares on instances everyone with any coverage solved.
    # A configuration that solved nothing has no cost to compare and is left
    # out of the intersection rather than emptying it.
    contenders = {name for name, entry in summary.items() if entry["solved"]}
    common = [
        name
        for name, runs in instances.items()
        if all(
            any(run["solved"] for run in runs if run["heuristic"] == heuristic)
            for heuristic in contenders
        )
    ]
    for name, entry in summary.items():
        total = entry["total"] or 1
        entry["coverage"] = entry["solved"] / total
        entry["mean_expanded"] = entry["expanded"] / total
        entry["mean_seconds"] = entry["seconds"] / total
        costs = [
            run["cost"] or 0
            for instance in common
            for run in instances[instance]
            if run["heuristic"] == name and run["solved"]
        ]
        entry["common"] = len(common)
        entry["mean_cost"] = sum(costs) / len(costs) if costs else None
    return summary
