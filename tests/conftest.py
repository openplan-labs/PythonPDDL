"""Shared pytest fixtures and constants for the jupyddl test suite."""

from __future__ import annotations

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO_ROOT, "pddl-examples")

# Known optimal costs for the example instances (validated across BFS, Dijkstra
# and A* with an admissible heuristic).
OPTIMAL_COST = {
    "blocksworld": 2,
    "dinner": 1,
    "flip": 3,
    "pallet": 12,
    "switch": 3,
    "tsp": 15,
}

# Solvable instances without conditional effects (relaxation heuristics are
# admissible on these). ``flip`` is solvable but uses conditional effects.
STRIPS_SOLVABLE = ["blocksworld", "dinner", "pallet", "switch", "tsp"]
SOLVABLE = STRIPS_SOLVABLE + ["flip"]
UNSOLVABLE = ["vehicle"]  # broken example data (typos): goal unreachable
UNSUPPORTED = ["grid"]  # numeric fluents, out of scope


def paths(name: str):
    folder = os.path.join(EXAMPLES, name)
    return os.path.join(folder, "domain.pddl"), os.path.join(folder, "problem.pddl")


@pytest.fixture(scope="session")
def examples_available():
    if not os.path.isdir(EXAMPLES) or not os.listdir(EXAMPLES):
        pytest.skip("pddl-examples submodule not initialised")
    return EXAMPLES


DEMOS = os.path.join(REPO_ROOT, "demos")

# Optimal costs for the bundled demo instances. Hanoi is the strongest check:
# five discs must take exactly 2**5 - 1 moves.
DEMO_OPTIMAL_COST = {
    "gripper": 17,
    "blocksworld8": 16,
    "hanoi": 31,
    "logistics": 24,
    "sokoban": 11,
    "elevator": 14,
    "rovers": 11,
    "network": 7,
    "numeric-transport": 11,
    "workshop": 33,
    "errands": 10,
    "timed-market": 4,
}

# Instances an optimal planner cannot finish in reasonable time; the test suite
# checks these with a satisficing planner instead.
SATISFICING_ONLY = {"blocksworld12"}


def demo_paths(name: str):
    folder = os.path.join(DEMOS, name)
    return os.path.join(folder, "domain.pddl"), os.path.join(folder, "problem.pddl")
