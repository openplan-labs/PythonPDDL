"""The reinforcement stage: stop imitating h*, start minimising search.

Imitation optimises a proxy. What we want is a heuristic that makes GBFS expand
few nodes, and "nodes expanded" is not a differentiable function of the
weights — it comes out the far side of a priority queue, a goal test and a
successor generator. Three mechanisms close that gap, and they fix different
things.

**DAgger** (:func:`dagger`) fixes *covariate shift*. Imitation trains on states
lying on optimal plans; search asks about states that do not, including the
dead ends and detours a mediocre heuristic wanders into. Training on your own
search distribution is the standard fix (Ross et al., 2011). Concretely: run
the current heuristic, keep the states it expanded, label them by solving from
each one, retrain on the union.

**Bootstrapping** (:func:`bootstrap`) fixes *data scarcity at the top of the
ladder*. You cannot label a 20-block instance you cannot solve. But a heuristic
trained on 6 blocks may just crack 8, whose plans then teach it 10 (Arfaee,
Zilles & Holte, 2011). Each round is a policy-improvement step where the policy
is "which instances can I solve at all".

**Direct search-cost optimisation** (:func:`optimise_search_cost`) attacks the
objective itself, with a derivative-free method, because no gradient exists to
follow. The cross-entropy method over the weight vector treats the planner as a
black box returning expansions. Evolutionary strategies are a real alternative
to policy gradients when rollouts are cheap and the parameter vector is modest
(Salimans et al., 2017), which is exactly this setting.

The crucial detail for all three: **start from the imitation solution.** Search
cost is a step function of the weights over most of the space — every candidate
that solves nothing scores identically — so a randomly initialised policy gets
no gradient signal, from any method. Imitation is what puts the optimiser
somewhere the objective can distinguish.

``.docs/rl-for-search.md`` sets out the MDP this corresponds to and why the
obvious policy-gradient formulation is harder than it looks.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Optional

from ..search import make_planner
from ..search.result import make_budget
from .dataset import Corpus, samples_from_plan
from .heuristic import HeuristicBundle
from .train import TrainConfig, train

__all__ = [
    "RLConfig",
    "SearchCost",
    "search_cost",
    "dagger",
    "bootstrap",
    "optimise_search_cost",
]


@dataclass
class RLConfig:
    """Settings shared by the reinforcement stages."""

    planner: str = "gbfs"
    #: Per-instance expansion budget. Doubles as the penalty for not solving,
    #: which is what keeps the objective finite and comparable.
    max_expansions: int = 5000
    time_limit: Optional[float] = None
    #: A failure is charged this multiple of the budget. Greater than 1 so that
    #: solving an instance slowly always beats not solving it — with a factor
    #: of exactly 1 the optimiser is indifferent between them.
    failure_penalty: float = 2.0
    seed: int = 0
    verbose: bool = False


@dataclass
class SearchCost:
    """What one evaluation of the objective measured."""

    score: float
    solved: int
    total: int
    expansions: float
    seconds: float

    @property
    def coverage(self) -> float:
        return self.solved / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "solved": self.solved,
            "total": self.total,
            "expansions": self.expansions,
            "seconds": self.seconds,
            "coverage": self.coverage,
        }


def search_cost(bundle: HeuristicBundle, tasks, config: Optional[RLConfig] = None):
    """Run the planner on every task and score the heuristic. Lower is better.

    The score is the mean expansion count, charging an unsolved instance
    ``failure_penalty * max_expansions``. Reporting mean expansions over only
    the solved instances — the tempting alternative — rewards a heuristic that
    solves one easy instance quickly and abandons the rest.

    Because the penalty is a multiple of the budget, scores are comparable
    **only at a fixed budget**. Halving ``max_expansions`` halves what a
    failure costs, so two runs under different budgets say nothing about each
    other. Every optimiser here holds the budget fixed for exactly that reason.
    """
    config = config or RLConfig()
    tasks = list(tasks)
    if not tasks:
        raise ValueError("search cost needs at least one task")
    planner = make_planner(config.planner)
    solved = 0
    total_expansions = 0.0
    started = time.perf_counter()
    for _, task in tasks:
        heuristic = bundle.bind(task)
        budget = make_budget(config.max_expansions, config.time_limit)
        result = planner.search(task, heuristic, budget=budget)
        if result.solved:
            solved += 1
            total_expansions += result.stats.expanded
        else:
            total_expansions += config.failure_penalty * config.max_expansions
    return SearchCost(
        score=total_expansions / len(tasks),
        solved=solved,
        total=len(tasks),
        expansions=total_expansions / len(tasks),
        seconds=time.perf_counter() - started,
    )


# --------------------------------------------------------------------------
# DAgger: train on the states the search actually visits
# --------------------------------------------------------------------------
def dagger(
    bundle: HeuristicBundle,
    tasks,
    corpus: Corpus,
    labeller=None,
    rounds: int = 2,
    states_per_task: int = 40,
    config: Optional[RLConfig] = None,
    train_config: Optional[TrainConfig] = None,
):
    """Aggregate data from the current heuristic's own search distribution.

    ``labeller`` maps ``(task, state)`` to a plan from that state, or ``None``
    if it cannot find one; the default solves with greedy search and ``hff``.
    Its plans are usually not optimal, so the samples it produces are recorded
    as upper bounds and down-weighted during training — an over-estimated label
    is still far more informative than no label for a state the imitation
    corpus never contained.

    Returns ``(bundle, corpus, history)``.
    """
    config = config or RLConfig()
    train_config = train_config or TrainConfig()
    labeller = labeller or _default_labeller(config)
    tasks = list(tasks)
    rng = random.Random(config.seed)
    history = []

    for round_index in range(1, rounds + 1):
        visited = _collect_visited_states(bundle, tasks, states_per_task, config, rng)
        added = 0
        for name, task, state in visited:
            plan = labeller(task, state)
            if not plan:
                continue
            samples, groups = samples_from_plan(
                task,
                plan,
                corpus.space.bind(task),
                instance=f"{name}#dagger{round_index}",
                optimal=False,
                ranking=True,
                rng=rng,
                start_state=state,
            )
            corpus.samples.extend(samples)
            corpus.groups.extend(groups)
            added += len(samples)
        bundle, _ = train(corpus, train_config)
        cost = search_cost(bundle, tasks, config)
        history.append(
            {
                "round": round_index,
                "added_samples": added,
                "corpus": len(corpus),
                **cost.to_dict(),
            }
        )
        if config.verbose:  # pragma: no cover - operator convenience
            print(
                f"  dagger round {round_index}: +{added} samples, "
                f"coverage {cost.coverage:.2f}, expansions {cost.expansions:.0f}"
            )
    return bundle, corpus, history


def _collect_visited_states(bundle, tasks, states_per_task, config, rng):
    """States the current heuristic expands, sampled along each search."""
    from ..trace import SearchObserver

    class _Collector(SearchObserver):
        def __init__(self):
            self.states = []

        def on_expand(self, state, **kwargs):
            self.states.append(state)

    planner = make_planner(config.planner)
    visited = []
    for name, task in tasks:
        collector = _Collector()
        budget = make_budget(config.max_expansions, config.time_limit)
        planner.search(task, bundle.bind(task), observer=collector, budget=budget)
        states = collector.states
        if not states:
            continue
        if len(states) > states_per_task:
            states = rng.sample(states, states_per_task)
        visited.extend((name, task, state) for state in states)
    return visited


def _default_labeller(config: RLConfig):
    """Label a state by solving from it with a fast satisficing configuration.

    The reference heuristic is built once per task and reused across every
    state of it. No heuristic in the library reads ``task.init`` — they are all
    parameterised by the state they are called on — so the same instance is
    valid for every re-rooting of the task, and rebuilding the relaxed-task
    tables per label would dominate the cost of labelling.
    """
    from ..heuristics import make_heuristic
    from .dataset import task_from_state

    cache: dict = {}

    def label(task, state):
        key = id(task)
        reference = cache.get(key)
        if reference is None:
            reference = make_heuristic("hff", task)
            cache[key] = reference
        planner = make_planner("gbfs")
        budget = make_budget(config.max_expansions, config.time_limit)
        result = planner.search(task_from_state(task, state), reference, budget=budget)
        return result.plan if result.solved else None

    return label


# --------------------------------------------------------------------------
# Bootstrapping: let the heuristic unlock its own training data
# --------------------------------------------------------------------------
def bootstrap(
    bundle: Optional[HeuristicBundle],
    ladder,
    corpus: Corpus,
    rounds: int = 3,
    config: Optional[RLConfig] = None,
    train_config: Optional[TrainConfig] = None,
):
    """Grow the corpus with instances the current heuristic has just cracked.

    ``ladder`` is ``(name, task)`` in increasing difficulty. Each round attempts
    every unsolved instance with the current heuristic under a fixed budget,
    adds whatever it managed, and retrains. Nothing is ever attempted twice
    after it succeeds, so rounds get cheaper as the frontier moves.

    Returns ``(bundle, corpus, history)``.
    """
    config = config or RLConfig()
    train_config = train_config or TrainConfig()
    ladder = list(ladder)
    rng = random.Random(config.seed)
    remaining = list(ladder)
    history = []

    for round_index in range(1, rounds + 1):
        if not remaining:
            break
        planner = make_planner(config.planner)
        newly = []
        still: list = []
        for name, task in remaining:
            budget = make_budget(config.max_expansions, config.time_limit)
            heuristic = None if bundle is None else bundle.bind(task)
            result = planner.search(task, heuristic, budget=budget)
            if result.solved and result.plan:
                newly.append((name, task, result.plan))
            else:
                still.append((name, task))
        for name, task, plan in newly:
            samples, groups = samples_from_plan(
                task,
                plan,
                corpus.space.bind(task),
                instance=f"{name}#boot{round_index}",
                optimal=False,
                ranking=True,
                rng=rng,
            )
            corpus.samples.extend(samples)
            corpus.groups.extend(groups)
        if newly:
            bundle, _ = train(corpus, train_config)
        history.append(
            {
                "round": round_index,
                "newly_solved": [name for name, _, _ in newly],
                "remaining": len(still),
                "corpus": len(corpus),
            }
        )
        if config.verbose:  # pragma: no cover - operator convenience
            print(
                f"  bootstrap round {round_index}: solved {len(newly)}, "
                f"{len(still)} left, corpus {len(corpus)}"
            )
        if len(still) == len(remaining):
            break  # the frontier stopped moving; more rounds cost and teach nothing
        remaining = still
    return bundle, corpus, history


# --------------------------------------------------------------------------
# Direct optimisation of the search cost
# --------------------------------------------------------------------------
def optimise_search_cost(
    bundle: HeuristicBundle,
    tasks,
    iterations: int = 8,
    population: int = 12,
    elite_fraction: float = 0.25,
    sigma: float = 0.15,
    config: Optional[RLConfig] = None,
    validation_tasks=None,
):
    """Tune the weights against expansions with the cross-entropy method.

    The parameter vector is perturbed, each candidate is scored by actually
    planning with it, the best fraction is kept, and the sampling distribution
    is refitted to those. No gradient is involved, which is the point: the
    objective is the planner.

    **The incumbent is selected on ``validation_tasks``, not on ``tasks``.** A
    thousand parameters tuned against eight instances will fit those eight
    instances, and this is not hypothetical: a run that reached 108 expansions
    on its tuning set scored 1734 on held-out instances, nearly five times
    *worse* than the imitated heuristic it started from. Candidates are still
    proposed by their tuning score — that is what the sampling distribution
    refits on — but nothing is returned unless it also improves on instances
    the optimiser is not fitting. Passing no validation set restores the old
    behaviour, and with it the old failure mode.

    ``sigma`` is relative to the standard deviation of the incoming weights, so
    a sensible perturbation scale does not depend on how the network was
    initialised.

    Returns ``(bundle, history)``.
    """
    config = config or RLConfig()
    tasks = list(tasks)
    scoring = list(validation_tasks) if validation_tasks else tasks
    rng = random.Random(config.seed)

    base = bundle.model.get_flat()
    spread = statistics.pstdev(base) if len(base) > 1 else 1.0
    step = max(1e-6, sigma * (spread or 1.0))
    deviations = [step] * len(base)
    mean = list(base)

    elite_count = max(2, int(population * elite_fraction))
    working = bundle.model.copy()
    probe = HeuristicBundle(
        bundle.space, working, bundle.scale, bundle.metrics, bundle.config
    )

    incumbent = SearchCostCandidate(list(base), search_cost(bundle, scoring, config))
    history = [{"iteration": 0, **incumbent.cost.to_dict()}]
    if config.verbose:  # pragma: no cover - operator convenience
        print(
            f"  cem start: validation coverage {incumbent.cost.coverage:.2f}, "
            f"expansions {incumbent.cost.expansions:.0f}"
        )

    for iteration in range(1, iterations + 1):
        candidates = []
        for _ in range(population):
            theta = [m + rng.gauss(0.0, d) for m, d in zip(mean, deviations)]
            working.set_flat(theta)
            candidates.append(
                SearchCostCandidate(theta, search_cost(probe, tasks, config))
            )
        candidates.sort(key=lambda c: c.cost.score)
        elites = candidates[:elite_count]

        mean = [sum(c.theta[i] for c in elites) / len(elites) for i in range(len(mean))]
        deviations = [
            max(
                step * 0.1,
                math.sqrt(
                    sum((c.theta[i] - mean[i]) ** 2 for c in elites) / len(elites)
                ),
            )
            for i in range(len(mean))
        ]

        # CEM's distribution mean is not one of the sampled candidates and can
        # be worse than all of them, so it is scored rather than assumed good.
        working.set_flat(mean)
        centre = SearchCostCandidate(list(mean), search_cost(probe, tasks, config))
        proposal = min([centre] + elites, key=lambda c: c.cost.score)

        working.set_flat(proposal.theta)
        validated = (
            proposal.cost if scoring is tasks else search_cost(probe, scoring, config)
        )
        if validated.score < incumbent.cost.score:
            incumbent = SearchCostCandidate(list(proposal.theta), validated)
        history.append(
            {
                "iteration": iteration,
                "tuning_score": proposal.cost.score,
                **incumbent.cost.to_dict(),
            }
        )
        if config.verbose:  # pragma: no cover - operator convenience
            print(
                f"  cem iter {iteration}: tuning {proposal.cost.score:.0f}, "
                f"validation {validated.score:.0f}, "
                f"incumbent {incumbent.cost.score:.0f} "
                f"(coverage {incumbent.cost.coverage:.2f})"
            )

    tuned = bundle.model.copy()
    tuned.set_flat(incumbent.theta)
    metrics = dict(bundle.metrics)
    metrics["search_cost"] = incumbent.cost.to_dict()
    return (
        HeuristicBundle(bundle.space, tuned, bundle.scale, metrics, bundle.config),
        history,
    )


@dataclass
class SearchCostCandidate:
    theta: list
    cost: SearchCost
