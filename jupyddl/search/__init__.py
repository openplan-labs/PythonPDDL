"""Search algorithms and a name-based planner registry."""

from .anytime import AnytimeWeightedAStar, BranchAndBound
from .base import Planner, best_first, heuristic_name
from .best_first_planners import (
    AStarSearch,
    GreedyBestFirstSearch,
    UniformCostSearch,
    WeightedAStarSearch,
)
from .enforced_hill_climbing import EnforcedHillClimbing
from .ida_star import IDAStarSearch
from .local import BeamSearch, HillClimbing, IteratedWidth
from .node import SearchNode, extract_plan
from .result import SearchResult, SearchStats
from .uninformed import (
    BreadthFirstSearch,
    DepthFirstSearch,
    IterativeDeepeningSearch,
)

# name -> zero-argument factory returning a fresh Planner instance.
PLANNERS = {
    "bfs": BreadthFirstSearch,
    "dfs": DepthFirstSearch,
    "iddfs": IterativeDeepeningSearch,
    "dijkstra": UniformCostSearch,
    "gbfs": GreedyBestFirstSearch,
    "astar": AStarSearch,
    "wastar": WeightedAStarSearch,
    "idastar": IDAStarSearch,
    "ehc": EnforcedHillClimbing,
    "hc": HillClimbing,
    "beam": BeamSearch,
    "iw": IteratedWidth,
    "bnb": BranchAndBound,
    "awastar": AnytimeWeightedAStar,
}

# Planners that are cost-optimal given an admissible heuristic (or none), and
# whose run completed: a result with ``truncated`` set proves nothing. ``bfs``
# and ``iddfs`` are not here -- they minimise plan *length*.
OPTIMAL_PLANNERS = tuple(
    name for name, factory in PLANNERS.items() if getattr(factory, "optimal", False)
)

# Planners that need a heuristic to run at all.
INFORMED_PLANNERS = tuple(
    name
    for name, factory in PLANNERS.items()
    if getattr(factory, "requires_heuristic", False)
)


def make_planner(name: str, **kwargs) -> Planner:
    """Instantiate a planner by name (see :data:`PLANNERS`).

    A name that is not registered triggers plugin discovery once (see
    :mod:`jupyddl.plugins`) before it is reported as unknown.
    """
    factory = PLANNERS.get(name)
    if factory is None:
        from ..plugins import load_plugins

        load_plugins()
        factory = PLANNERS.get(name)
    if factory is None:
        raise ValueError(f"Unknown planner '{name}'. Available: {sorted(PLANNERS)}")
    return factory(**kwargs)


def describe_planners() -> list:
    """Serialisable metadata for the CLI, the benchmark harness and the web UI."""
    rows = []
    for name, factory in sorted(PLANNERS.items()):
        rows.append(
            {
                "name": name,
                "optimal": bool(getattr(factory, "optimal", False)),
                "requires_heuristic": bool(
                    getattr(factory, "requires_heuristic", False)
                ),
                "summary": (factory.__doc__ or "").strip().split("\n")[0],
            }
        )
    return rows


__all__ = [
    "Planner",
    "best_first",
    "heuristic_name",
    "AStarSearch",
    "GreedyBestFirstSearch",
    "UniformCostSearch",
    "WeightedAStarSearch",
    "EnforcedHillClimbing",
    "IDAStarSearch",
    "BreadthFirstSearch",
    "DepthFirstSearch",
    "IterativeDeepeningSearch",
    "HillClimbing",
    "BeamSearch",
    "IteratedWidth",
    "BranchAndBound",
    "AnytimeWeightedAStar",
    "SearchNode",
    "SearchResult",
    "SearchStats",
    "extract_plan",
    "PLANNERS",
    "OPTIMAL_PLANNERS",
    "INFORMED_PLANNERS",
    "make_planner",
    "describe_planners",
]
