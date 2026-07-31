"""Learned heuristics: imitate a corpus of plans, then optimise search directly.

The pipeline has two stages and they answer different questions.

**Imitation** (:mod:`jupyddl.learn.train`) asks *what does h\\* look like?* It
fits a network to the cost-to-go that solved plans reveal. Cheap, stable, and
limited by a mismatch it cannot see: it is trained on states that lie on
optimal plans, and at search time it is asked about states that do not.

**Reinforcement** (:mod:`jupyddl.learn.rl`) asks the question that actually
matters: *which heuristic expands the fewest nodes?* Search cost is not a
differentiable function of the weights — it runs through a priority queue — so
this stage reaches it three ways: aggregating data from the states search
really visits (DAgger), growing the corpus with instances the current
heuristic just became able to solve (bootstrapping), and optimising expansions
directly with a derivative-free method.

Quickstart::

    from jupyddl.learn import learn_heuristic

    bundle = learn_heuristic("blocksworld", sizes=range(4, 9), seed=0)
    bundle.save("blocksworld.heur.json")

then anywhere a heuristic name is accepted::

    jupyddl solve domain.pddl problem.pddl -s gbfs -H learned:blocksworld.heur.json

Nothing here is imported by the core. :mod:`jupyddl.heuristics` resolves
``learned:`` lazily, so a planner that never asks for one never pays for it.
"""

from __future__ import annotations

from .dataset import Corpus, RankingGroup, Sample, build_corpus, samples_from_plan
from .features import FeatureSpace
from .heuristic import HeuristicBundle, LearnedHeuristic
from .model import MLP, numpy_available
from .pipeline import learn_heuristic, solved_corpus, tasks_from_generator
from .rl import (
    RLConfig,
    bootstrap,
    dagger,
    optimise_search_cost,
    search_cost,
)
from .train import TrainConfig, TrainReport, evaluate, evaluate_ranking, train

__all__ = [
    "FeatureSpace",
    "Sample",
    "RankingGroup",
    "Corpus",
    "build_corpus",
    "samples_from_plan",
    "MLP",
    "numpy_available",
    "HeuristicBundle",
    "LearnedHeuristic",
    "TrainConfig",
    "TrainReport",
    "train",
    "evaluate",
    "evaluate_ranking",
    "RLConfig",
    "dagger",
    "bootstrap",
    "optimise_search_cost",
    "search_cost",
    "learn_heuristic",
    "solved_corpus",
    "tasks_from_generator",
]
