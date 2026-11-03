"""The release automation's assumptions, checked before a release depends on them.

release-please bumps three places at once: the manifest, `pyproject.toml`
and `jupyddl.__version__` (found through its `x-release-please-version`
marker). If they disagree, or the marker goes missing, the release PR bumps
some and not others and `release.yml` refuses the result at release time,
the worst moment to find out.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re

from conftest import REPO_ROOT

import jupyddl


def _read(*parts):
    with open(os.path.join(REPO_ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


def _fixup():
    path = os.path.join(REPO_ROOT, "tools", "release_pr_fixup.py")
    spec = importlib.util.spec_from_file_location("release_pr_fixup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_version_release_please_bumps_agrees():
    manifest = json.loads(_read(".release-please-manifest.json"))["."]
    declared = re.search(r'^version = "([^"]+)"', _read("pyproject.toml"), re.M)
    assert manifest == declared.group(1) == jupyddl.__version__


def test_the_version_line_carries_the_release_please_marker():
    line = next(
        line
        for line in _read("jupyddl", "__init__.py").splitlines()
        if line.startswith("__version__")
    )
    assert "x-release-please-version" in line


def test_every_extra_file_release_please_edits_exists():
    config = json.loads(_read("release-please-config.json"))
    for entry in config["packages"]["."]["extra-files"]:
        assert os.path.exists(os.path.join(REPO_ROOT, entry["path"])), entry


AFTER_RELEASE_PLEASE = """# Changelog

Intro.

## [2.5.0](https://example.invalid/compare/v2.4.0...v2.5.0) (2026-11-03)


### Features

* plugins ([#164](https://example.invalid/164))

## [Unreleased]

### Fixed
- Derived facts are retracted.

## [2.4.0]

### Added
- Older notes.
"""


def test_the_hand_written_notes_are_filed_under_the_new_version():
    filed = _fixup().file_unreleased_notes(AFTER_RELEASE_PLEASE)
    assert filed.index("## [Unreleased]") < filed.index("## [2.5.0]")
    assert filed.count("## [Unreleased]") == 1
    # release.yml cuts the GitHub Release notes from the version's section.
    section = re.search(
        r"^## \[2\.5\.0\][^\n]*\n(.*?)(?=^## \[|\Z)", filed, re.M | re.S
    ).group(1)
    assert "* plugins" in section and "Derived facts are retracted." in section
    assert "Older notes." not in section
    assert _fixup().file_unreleased_notes(filed) == filed  # idempotent


def test_the_fixup_leaves_an_unreleased_changelog_alone():
    text = _read("CHANGELOG.md")
    assert _fixup().file_unreleased_notes(text) == text
