"""Planners, heuristics and generators from other packages, found by entry point.

jupyddl's registries are plain dictionaries, so code that imports jupyddl can
always add to them directly::

    from jupyddl.search import PLANNERS
    PLANNERS["mine"] = MyPlanner

That only reaches the process that ran the assignment. An *entry point* makes an
extension visible everywhere instead -- the ``jupyddl`` command line, the
benchmark harness, :func:`jupyddl.solve` -- as soon as the package declaring it
is installed. Declare them in that package's ``pyproject.toml``::

    [project.entry-points."jupyddl.planners"]
    mine = "my_package.search:MyPlanner"

    [project.entry-points."jupyddl.heuristics"]
    myh = "my_package.heuristics:MyHeuristic"

    [project.entry-points."jupyddl.generators"]
    corridor = "my_package.generators:generate_corridor"

The value is what the matching registry holds: a planner class (or any
callable returning a :class:`~jupyddl.search.Planner`), a heuristic class (or
any ``task -> Heuristic`` callable), or a generator function
``(size, seed, **kwargs) -> (domain_text, problem_text)``.

Discovery is lazy: nothing here runs when jupyddl is imported. The command line
loads plugins before it builds its argument parser, and the library loads them
the first time it is asked for a name it does not know. Three rules keep a
plugin from breaking anything it was not asked to do:

- a name that is already registered -- a built-in, or something added by hand
  -- always wins, and the clash is reported;
- a plugin that fails to import, or is not callable, is reported with a
  warning and skipped;
- ``JUPYDDL_NO_PLUGINS=1`` switches discovery off, for a run that must not
  depend on whatever else happens to be installed.
"""

from __future__ import annotations

import os
import warnings
from importlib.metadata import entry_points

GROUPS = ("jupyddl.planners", "jupyddl.heuristics", "jupyddl.generators")

#: What discovery registered: ``group -> {name: source}``, where ``source`` is
#: the distribution that declared it.
REGISTERED: dict = {group: {} for group in GROUPS}

#: What discovery skipped and why: ``group -> {name: reason}``.
SKIPPED: dict = {group: {} for group in GROUPS}

_loaded = False


class PluginWarning(RuntimeWarning):
    """A plugin was found but could not be registered."""


def disabled() -> bool:
    """True when ``JUPYDDL_NO_PLUGINS`` is set to anything but ``0`` or empty."""
    return os.environ.get("JUPYDDL_NO_PLUGINS", "").strip() not in ("", "0")


def _registry(group: str) -> dict:
    # Imported here, not at module level: the registries import this module
    # lazily on a lookup miss, and a top-level import would make that a cycle.
    if group == "jupyddl.planners":
        from .search import PLANNERS

        return PLANNERS
    if group == "jupyddl.heuristics":
        from .heuristics import HEURISTICS

        return HEURISTICS
    from .generator import GENERATORS

    return GENERATORS


def _source(entry) -> str:
    dist = getattr(entry, "dist", None)
    return dist.name if dist is not None else entry.value


def _skip(group: str, name: str, reason: str) -> None:
    SKIPPED[group][name] = reason
    warnings.warn(
        f"jupyddl plugin '{name}' ({group}) skipped: {reason}",
        PluginWarning,
        stacklevel=3,
    )


def load_plugins(force: bool = False) -> dict:
    """Register every installed plugin, once per process.

    Returns :data:`REGISTERED`. ``force=True`` scans again, which only matters
    when packages were installed after the first scan.
    """
    global _loaded
    if disabled() or (_loaded and not force):
        return REGISTERED
    _loaded = True
    for group in GROUPS:
        registry = _registry(group)
        for entry in entry_points(group=group):
            name, source = entry.name, _source(entry)
            if group == "jupyddl.heuristics" and ":" in name:
                _skip(group, name, "':' is reserved for parameterised specs")
                continue
            if name in registry and REGISTERED[group].get(name) != source:
                _skip(group, name, f"'{name}' is already registered")
                continue
            try:
                factory = entry.load()
            except Exception as exc:  # a broken plugin must not break jupyddl
                _skip(group, name, f"import of {entry.value} failed: {exc!r}")
                continue
            if not callable(factory):
                _skip(group, name, f"{entry.value} is not callable")
                continue
            registry[name] = factory
            REGISTERED[group][name] = source
            SKIPPED[group].pop(name, None)
    return REGISTERED


def describe_plugins() -> list:
    """One row per discovered plugin, registered or skipped, for the CLI."""
    rows = []
    for group in GROUPS:
        kind = group.rsplit(".", 1)[1][:-1]  # "jupyddl.planners" -> "planner"
        for name, source in sorted(REGISTERED[group].items()):
            rows.append({"kind": kind, "name": name, "source": source, "error": ""})
        for name, reason in sorted(SKIPPED[group].items()):
            rows.append({"kind": kind, "name": name, "source": "", "error": reason})
    return rows


__all__ = [
    "GROUPS",
    "REGISTERED",
    "SKIPPED",
    "PluginWarning",
    "disabled",
    "load_plugins",
    "describe_plugins",
]
