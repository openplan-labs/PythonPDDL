#!/usr/bin/env python3
"""Bundle jupyddl and the demo instances for the browser playground.

The playground runs the *real* library under Pyodide rather than a JavaScript
re-implementation. Because the core is stdlib-only, there is nothing to compile
and no wheel to resolve: we simply hand Pyodide the package sources and let it
import them. This script writes:

* ``web/dist/jupyddl-sources.json`` — ``{module path: source}`` for the pure
  Python part of the package (:mod:`jupyddl.viz` is skipped; it needs matplotlib,
  which the playground does not load);
* ``web/dist/demos.json`` — the bundled demo domains and problems;
* ``web/dist/build.json`` — version and build metadata shown in the footer.

Run it from the repository root::

    python tools/build_web.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "jupyddl")
OUT = os.path.join(ROOT, "web", "dist")

# viz/ needs matplotlib, which the playground deliberately does not ship.
SKIP_DIRS = {"viz", "__pycache__"}

DEMO_ORDER = [
    (
        "gripper",
        "Gripper",
        "A robot with two grippers ferries six balls between rooms.",
    ),
    (
        "blocksworld8",
        "Blocksworld",
        "Eight blocks, two towers in, one inverted tower out.",
    ),
    ("hanoi", "Towers of Hanoi", "Five discs. The optimal plan is exactly 31 moves."),
    (
        "logistics",
        "Logistics",
        "Trucks and a plane deliver three packages, with action costs.",
    ),
    ("sokoban", "Sokoban", "Push two boxes onto their targets in a 4x4 room."),
    ("elevator", "Elevator", "Six floors, four passengers, conditional effects."),
]


def collect_sources() -> dict:
    """Map ``jupyddl/...py`` -> source text for every browser-safe module."""
    sources = {}
    for dirpath, dirnames, filenames in os.walk(PACKAGE):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            full = os.path.join(dirpath, filename)
            key = os.path.relpath(full, ROOT).replace(os.sep, "/")
            with open(full, encoding="utf-8") as handle:
                sources[key] = handle.read()
    return sources


def collect_demos() -> list:
    """Read the demo instances, keeping the curated order and descriptions."""
    demos = []
    for folder, title, blurb in DEMO_ORDER:
        base = os.path.join(ROOT, "demos", folder)
        domain = os.path.join(base, "domain.pddl")
        problem = os.path.join(base, "problem.pddl")
        if not (os.path.exists(domain) and os.path.exists(problem)):
            print(f"  ! missing demo: {folder}", file=sys.stderr)
            continue
        with open(domain, encoding="utf-8") as handle:
            domain_text = handle.read()
        with open(problem, encoding="utf-8") as handle:
            problem_text = handle.read()
        demos.append(
            {
                "id": folder,
                "title": title,
                "blurb": blurb,
                "domain": domain_text,
                "problem": problem_text,
            }
        )
    return demos


def version() -> str:
    namespace: dict = {}
    init = os.path.join(PACKAGE, "__init__.py")
    with open(init, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("__version__"):
                exec(line, namespace)  # noqa: S102 - our own source line
                break
    return namespace.get("__version__", "0.0.0")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    sources = collect_sources()
    demos = collect_demos()

    with open(os.path.join(OUT, "jupyddl-sources.json"), "w", encoding="utf-8") as fh:
        json.dump(sources, fh)
    with open(os.path.join(OUT, "demos.json"), "w", encoding="utf-8") as fh:
        json.dump(demos, fh)
    with open(os.path.join(OUT, "build.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": version(), "modules": len(sources)}, fh)

    total = sum(len(s) for s in sources.values())
    print(f"jupyddl {version()}")
    print(
        f"  {len(sources)} modules ({total / 1024:.0f} KB) -> web/dist/jupyddl-sources.json"
    )
    print(f"  {len(demos)} demos -> web/dist/demos.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
