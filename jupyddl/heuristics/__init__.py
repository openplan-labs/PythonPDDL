"""Heuristics and a name-based registry."""

from __future__ import annotations

from functools import partial

from .base import Heuristic
from .critical_path import CriticalPathHeuristic
from .delete_relaxation import FFHeuristic, HAddHeuristic, HMaxHeuristic
from .lmcut import LMCutHeuristic
from .simple import BlindHeuristic, GoalCountHeuristic

# name -> callable(task) -> Heuristic
HEURISTICS = {
    "blind": BlindHeuristic,
    "goalcount": GoalCountHeuristic,
    "hmax": HMaxHeuristic,
    "hadd": HAddHeuristic,
    "hff": FFHeuristic,
    "lmcut": LMCutHeuristic,
    "h1": partial(CriticalPathHeuristic, m=1),
    "h2": partial(CriticalPathHeuristic, m=2),
    "hm": CriticalPathHeuristic,
}


# A heuristic that needs more than a task to build takes an argument after a
# colon: ``learned:blocksworld.heur.json``. Keeping this a *string* spec rather
# than a Python object is what makes a trained heuristic usable from the CLI,
# the benchmark harness and the web UI without any of them knowing it exists.
LOADERS = {}


def _load_learned(path: str, task) -> Heuristic:
    """Resolve ``learned:<path>``, importing the learning stack only if asked."""
    from ..learn.heuristic import HeuristicBundle

    return HeuristicBundle.load(path).bind(task)


LOADERS["learned"] = _load_learned


def make_heuristic(name, task) -> Heuristic:
    """Instantiate a heuristic by name (see :data:`HEURISTICS`).

    Accepts a registry name (``"lmcut"``), a parameterised spec
    (``"learned:model.json"``), or an already-built heuristic, which is passed
    through so callers holding a trained model need not round-trip it to disk.
    """
    if callable(name) and not isinstance(name, str):
        return name
    if ":" in name:
        kind, _, argument = name.partition(":")
        loader = LOADERS.get(kind)
        if loader is None:
            raise ValueError(
                f"Unknown parameterised heuristic '{kind}'. "
                f"Available: {sorted(LOADERS)}"
            )
        return loader(argument, task)
    try:
        factory = HEURISTICS[name]
    except KeyError:
        raise ValueError(
            f"Unknown heuristic '{name}'. Available: {sorted(HEURISTICS)}"
            + (f" (or {sorted(LOADERS)} with an argument)" if LOADERS else "")
        ) from None
    return factory(task)


__all__ = [
    "Heuristic",
    "BlindHeuristic",
    "GoalCountHeuristic",
    "HMaxHeuristic",
    "HAddHeuristic",
    "FFHeuristic",
    "LMCutHeuristic",
    "CriticalPathHeuristic",
    "HEURISTICS",
    "LOADERS",
    "make_heuristic",
]
