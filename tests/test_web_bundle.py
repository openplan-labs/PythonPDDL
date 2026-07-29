"""The browser playground bundle.

The playground runs the real package under Pyodide, so the bundle must stay in
step with the sources on disk — a stale ``web/dist`` ships a different library
than the one in the repository.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

WEB = os.path.join(REPO_ROOT, "web")
DIST = os.path.join(WEB, "dist")
BUILDER = os.path.join(REPO_ROOT, "tools", "build_web.py")


@pytest.fixture(scope="module")
def sources():
    path = os.path.join(DIST, "jupyddl-sources.json")
    if not os.path.exists(path):
        pytest.skip("web bundle not built; run python tools/build_web.py")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_bundle_carries_the_core_modules(sources):
    for module in [
        "jupyddl/__init__.py",
        "jupyddl/api.py",
        "jupyddl/trace.py",
        "jupyddl/grounding.py",
        "jupyddl/task.py",
        "jupyddl/parser/parser.py",
        "jupyddl/search/base.py",
        "jupyddl/heuristics/lmcut.py",
    ]:
        assert module in sources, f"{module} is missing from the browser bundle"


def test_bundle_excludes_matplotlib_dependent_code(sources):
    """viz/ needs matplotlib, which the playground deliberately does not load."""
    assert not [name for name in sources if name.startswith("jupyddl/viz/")]


def test_bundle_matches_the_working_tree(sources):
    """Every bundled module is byte-identical to its source file."""
    stale = []
    for name, text in sources.items():
        path = os.path.join(REPO_ROOT, name)
        assert os.path.exists(path), f"{name} is bundled but no longer exists"
        with open(path, encoding="utf-8") as handle:
            if handle.read() != text:
                stale.append(name)
    assert not stale, (
        "web/dist is stale for: "
        + ", ".join(sorted(stale))
        + " — run python tools/build_web.py"
    )


def test_every_demo_is_bundled():
    path = os.path.join(DIST, "demos.json")
    if not os.path.exists(path):
        pytest.skip("web bundle not built")
    with open(path, encoding="utf-8") as handle:
        demos = json.load(handle)
    names = {demo["id"] for demo in demos}
    assert {"gripper", "blocksworld8", "hanoi", "sokoban"} <= names
    for demo in demos:
        assert demo["domain"].strip().startswith("(define") or ";" in demo["domain"]
        assert "(define" in demo["problem"]
        assert demo["title"] and demo["blurb"]


def test_bundle_is_ordered_independently_of_the_filesystem(sources):
    """The bundle is committed, so it must not depend on the walk order.

    ``os.walk`` reports subdirectories in whatever order the filesystem hands
    them over, which differs between a developer's disk and a CI runner. Left
    unsorted the same sources produce byte-different JSON, and the staleness
    check fails for a bundle that is not actually stale.
    """
    assert list(sources) == sorted(sources)


def test_builder_is_reproducible(tmp_path):
    """Running the builder again must not change the committed bundle."""
    before = {}
    for name in ("jupyddl-sources.json", "demos.json", "build.json"):
        path = os.path.join(DIST, name)
        if not os.path.exists(path):
            pytest.skip("web bundle not built")
        with open(path, encoding="utf-8") as handle:
            before[name] = handle.read()

    subprocess.run(
        [sys.executable, BUILDER], cwd=REPO_ROOT, check=True, capture_output=True
    )

    for name, original in before.items():
        with open(os.path.join(DIST, name), encoding="utf-8") as handle:
            assert handle.read() == original, f"{name} changed; commit the rebuild"


def test_playground_assets_exist():
    for name in (
        "index.html",
        "app.js",
        "worker.js",
        "charts.js",
        "style.css",
        "bootstrap.py",
    ):
        assert os.path.exists(os.path.join(WEB, name)), f"web/{name} is missing"


def test_bootstrap_is_valid_python():
    """It is executed inside Pyodide, so a syntax error would only show there."""
    path = os.path.join(WEB, "bootstrap.py")
    with open(path, encoding="utf-8") as handle:
        compile(handle.read(), path, "exec")
