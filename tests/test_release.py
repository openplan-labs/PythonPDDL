"""The release automation (tools/release.py), checked before a merge relies on it.

Every merge to main runs `release.yml`, which asks `tools/release.py` what
the merged pull request's commits want released and then cuts it. A mistake
here either ships a version nobody asked for -- which PyPI never lets you take
back -- or silently skips one, so the rules are pinned case by case.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT

import jupyddl


def _release():
    path = os.path.join(REPO_ROOT, "tools", "release.py")
    spec = importlib.util.spec_from_file_location("release_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(*parts):
    with open(os.path.join(REPO_ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


@pytest.mark.parametrize(
    "messages, labels, expected",
    [
        (["fix(lmcut): handle an empty goal"], [], "patch"),
        (["perf: cache h"], [], "patch"),
        (["feat: plugins"], [], "minor"),
        (["fix: a", "feat: b", "docs: c"], [], "minor"),  # the largest wins
        (["feat!: drop Python 3.10"], [], "major"),
        (["fix(api)!: rename solve"], [], "major"),
        (["fix: a\n\nBREAKING CHANGE: solve() returns a tuple"], [], "major"),
        (["docs: guide", "chore(deps): pin", "ci: x", "test: y"], [], None),
        (["Update README.md", "Merge branch 'main'"], [], None),  # not conventional
        (["feat: plugins"], ["release:none"], None),  # a label overrides
        (["docs: typo"], ["release:patch"], "patch"),
        (["fix: a"], ["release:major", "release:patch"], "major"),
        # "BREAKING CHANGE" only counts as a footer of a conventional commit.
        (["Mention BREAKING CHANGE: nothing"], [], None),
    ],
)
def test_the_bump_follows_the_commits(messages, labels, expected):
    assert _release().bump_for(messages, labels) == expected


@pytest.mark.parametrize(
    "part, expected", [("patch", "2.4.1"), ("minor", "2.5.0"), ("major", "3.0.0")]
)
def test_bumped(part, expected):
    assert _release().bumped("2.4.0", part) == expected


CHANGELOG = """# Changelog

## [Unreleased]

### Fixed
- Derived facts are retracted.

## [2.4.0]

- Older notes.
"""


def test_the_unreleased_notes_become_the_release():
    dated = _release().date_changelog(CHANGELOG, "2.5.0", "2026-11-03")
    assert dated.index("## [Unreleased]") < dated.index("## [2.5.0] - 2026-11-03")
    section = re.search(
        r"^## \[2\.5\.0\][^\n]*\n(.*?)(?=^## \[|\Z)", dated, re.M | re.S
    ).group(1)
    assert "Derived facts are retracted." in section and "Older notes" not in section
    unreleased = re.search(r"^## \[Unreleased\]\n(.*?)(?=^## \[)", dated, re.M | re.S)
    assert unreleased.group(1).strip() == ""


def test_an_empty_changelog_takes_the_pr_title_or_refuses():
    empty = CHANGELOG.replace("### Fixed\n- Derived facts are retracted.\n", "")
    dated = _release().date_changelog(empty, "2.4.1", "2026-11-03", "fix: x (#9)")
    assert "## [2.4.1] - 2026-11-03\n\n- fix: x (#9)" in dated
    with pytest.raises(SystemExit):
        _release().date_changelog(empty, "2.4.1", "2026-11-03")


def test_apply_bumps_every_place_the_version_is_written(tmp_path):
    for name in ("pyproject.toml", "CITATION.cff", "CHANGELOG.md"):
        shutil.copy(os.path.join(REPO_ROOT, name), tmp_path / name)
    (tmp_path / "jupyddl").mkdir()
    shutil.copy(os.path.join(REPO_ROOT, "jupyddl", "__init__.py"), tmp_path / "jupyddl")
    release = _release()
    old = release.current_version(tmp_path)
    new = release.apply("minor", root=tmp_path, date="2026-11-03", note="feat: x")
    assert new == release.bumped(old, "minor")
    assert f'version = "{new}"' in (tmp_path / "pyproject.toml").read_text()
    init = (tmp_path / "jupyddl" / "__init__.py").read_text()
    assert f'__version__ = "{new}"' in init
    assert f"version: {new}" in (tmp_path / "CITATION.cff").read_text()
    assert f"## [{new}] - 2026-11-03" in (tmp_path / "CHANGELOG.md").read_text()


def test_commits_since_the_last_tag_are_counted(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.invalid")
    git("config", "user.name", "t")
    history = [("chore: start", "v1.0.0"), ("feat: missed", None), ("docs: x", None)]
    for message, tag in history:
        (tmp_path / "f").write_text(message)
        git("add", "f")
        git("commit", "-qm", message)
        if tag:
            git("tag", tag)
    release = _release()
    messages = release.messages_since_last_tag(tmp_path)
    assert messages == ["docs: x", "feat: missed"]
    # A docs-only merge still releases the feature whose run was dropped.
    assert release.bump_for(["docs: x"] + messages) == "minor"


def test_every_version_in_the_repository_agrees():
    declared = re.search(r'^version = "([^"]+)"', _read("pyproject.toml"), re.M)
    cited = re.search(r"^version: (.+)$", _read("CITATION.cff"), re.M)
    assert declared.group(1) == jupyddl.__version__ == cited.group(1).strip()
