"""The extension guide and the plugin mechanism it documents.

``.docs/EXTENDING.md`` is executable documentation: every ``python`` block in
it runs here, in order, in one namespace. A guide that drifts from the API it
describes fails this test instead of misleading the next contributor.
"""

from __future__ import annotations

import os
import re
import types
import warnings

import pytest

from jupyddl import plugins
from jupyddl.cli import main
from jupyddl.generator import GENERATORS, generate
from jupyddl.heuristics import HEURISTICS, make_heuristic
from jupyddl.search import PLANNERS, make_planner

from conftest import REPO_ROOT

GUIDE = os.path.join(REPO_ROOT, ".docs", "EXTENDING.md")
REGISTRIES = (PLANNERS, HEURISTICS, GENERATORS)


@pytest.fixture
def clean_registries(monkeypatch):
    """Undo whatever a test registers, and reset plugin discovery."""
    saved = [dict(registry) for registry in REGISTRIES]
    monkeypatch.setattr(plugins, "_loaded", False)
    for group in plugins.GROUPS:
        monkeypatch.setitem(plugins.REGISTERED, group, {})
        monkeypatch.setitem(plugins.SKIPPED, group, {})
    yield
    for registry, before in zip(REGISTRIES, saved):
        registry.clear()
        registry.update(before)


def _python_blocks(path: str) -> list:
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.S | re.M)


def test_the_guide_has_examples():
    assert len(_python_blocks(GUIDE)) >= 6


def test_every_example_in_the_guide_runs(clean_registries, capsys):
    namespace: dict = {"__name__": "extending_guide"}
    for index, block in enumerate(_python_blocks(GUIDE)):
        try:
            exec(compile(block, f"EXTENDING.md[block {index}]", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - the failure message
            pytest.fail(f"block {index} of EXTENDING.md raised {exc!r}:\n{block}")
    # What the guide registered is visible to the ordinary entry points.
    assert {"astar-tb", "walk"} <= set(PLANNERS)
    assert "achiever" in HEURISTICS and "corridor" in GENERATORS
    assert generate("corridor", size=5, seed=1) == generate("corridor", size=5, seed=1)


# -- plugins -----------------------------------------------------------------


class _Dist:
    def __init__(self, name):
        self.name = name


class _Entry:
    """Just enough of ``importlib.metadata.EntryPoint`` for discovery."""

    def __init__(self, name, value, target, dist="my-plugin"):
        self.name, self.value, self._target = name, value, target
        self.dist = _Dist(dist)

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def _install(monkeypatch, entries: dict):
    """Pretend a package declaring ``entries`` (group -> list) is installed."""
    monkeypatch.delenv("JUPYDDL_NO_PLUGINS", raising=False)
    monkeypatch.setattr(
        plugins, "entry_points", lambda group: list(entries.get(group, []))
    )


class _Planner:
    name = "plug"

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _heuristic(task):
    return lambda state: 0.0


def test_an_installed_plugin_is_found_on_first_lookup(monkeypatch, clean_registries):
    _install(
        monkeypatch,
        {
            "jupyddl.planners": [_Entry("plug", "pkg:_Planner", _Planner)],
            "jupyddl.heuristics": [_Entry("zero", "pkg:_heuristic", _heuristic)],
        },
    )
    assert "plug" not in PLANNERS
    assert make_planner("plug", depth=3).kwargs == {"depth": 3}
    assert make_heuristic("zero", task=None)(frozenset()) == 0.0
    assert plugins.REGISTERED["jupyddl.planners"] == {"plug": "my-plugin"}


def test_a_plugin_cannot_replace_a_builtin(monkeypatch, clean_registries):
    builtin = PLANNERS["astar"]
    _install(
        monkeypatch, {"jupyddl.planners": [_Entry("astar", "pkg:_Planner", _Planner)]}
    )
    with pytest.warns(plugins.PluginWarning, match="already registered"):
        plugins.load_plugins()
    assert PLANNERS["astar"] is builtin
    assert "astar" in plugins.SKIPPED["jupyddl.planners"]


def test_a_broken_plugin_is_skipped_not_fatal(monkeypatch, clean_registries):
    _install(
        monkeypatch,
        {
            "jupyddl.planners": [
                _Entry("broken", "pkg:gone", ImportError("no module named pkg")),
                _Entry("constant", "pkg:VALUE", 42),
                _Entry("plug", "pkg:_Planner", _Planner),
            ],
            "jupyddl.heuristics": [_Entry("learned:x", "pkg:_heuristic", _heuristic)],
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plugins.load_plugins()
    assert len([w for w in caught if w.category is plugins.PluginWarning]) == 3
    assert set(plugins.SKIPPED["jupyddl.planners"]) == {"broken", "constant"}
    assert "learned:x" in plugins.SKIPPED["jupyddl.heuristics"]
    assert "plug" in PLANNERS  # the good one still lands


def test_discovery_runs_once_and_can_be_switched_off(monkeypatch, clean_registries):
    calls = []

    def entry_points(group):
        calls.append(group)
        return []

    monkeypatch.setattr(plugins, "entry_points", entry_points)
    monkeypatch.setenv("JUPYDDL_NO_PLUGINS", "1")
    plugins.load_plugins()
    assert calls == []
    monkeypatch.delenv("JUPYDDL_NO_PLUGINS")
    plugins.load_plugins()
    plugins.load_plugins()
    assert calls == list(plugins.GROUPS)


def test_the_cli_sees_plugins(monkeypatch, clean_registries, capsys, tmp_path):
    def generate_line(size=3, seed=0):
        """A one-action line world."""
        domain = (
            "(define (domain line) (:requirements :strips)"
            " (:predicates (a) (b)) (:action go :precondition (a) :effect (b)))"
        )
        problem = "(define (problem p) (:domain line) (:init (a)) (:goal (b)))"
        return domain, problem

    planner = types.new_class("Plugged", (PLANNERS["bfs"],))
    _install(
        monkeypatch,
        {
            "jupyddl.planners": [_Entry("plugged", "pkg:Plugged", planner)],
            "jupyddl.generators": [_Entry("line", "pkg:line", generate_line)],
        },
    )
    assert main(["plugins"]) == 0
    listed = capsys.readouterr().out
    assert "plugged" in listed and "line" in listed and "my-plugin" in listed

    assert main(["generate", "line", "-o", str(tmp_path)]) == 0
    folder = next(p for p in tmp_path.iterdir() if p.is_dir())
    domain, problem = str(folder / "domain.pddl"), str(folder / "problem.pddl")
    assert main(["solve", domain, problem, "-s", "plugged", "-H", "none"]) == 0
    assert "go" in capsys.readouterr().out
