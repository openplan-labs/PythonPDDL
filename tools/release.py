"""Decide and cut a release from what a merged pull request contains.

    python tools/release.py plan [--since-last-tag] < pr.json   # major|minor|patch|none
    python tools/release.py apply PART [--note TEXT] [--date YYYY-MM-DD]

Run by ``release.yml`` on every push to main. No release pull request, nothing
to approve: the merge *is* the release decision.

**plan** reads ``{"labels": [...], "messages": [...]}`` -- the merged pull
request's labels, and its title plus every commit message in it -- and picks
the largest bump any of them asks for, by Conventional Commits:

* ``feat!:``/``fix!:``, or a ``BREAKING CHANGE:`` footer  -> major
* ``feat:``                                                -> minor
* ``fix:`` / ``perf:``                                     -> patch
* anything else (``docs:``, ``chore:``, ``ci:``, ``test:``) -> none

A ``release:major|minor|patch|none`` label on the pull request overrides that,
for the rare merge whose commits say the wrong thing.

``--since-last-tag`` also counts every commit on the branch since the last
``v*`` tag. GitHub keeps only one *pending* run per concurrency group, so of
several quick merges one can be dropped; this way its bump still lands with
the next release instead of being lost.

**apply** bumps the version in ``pyproject.toml``, ``jupyddl/__init__.py`` and
``CITATION.cff``, turns ``## [Unreleased]`` into ``## [X.Y.Z] - date`` under a
fresh empty ``[Unreleased]``, and rebuilds ``web/dist``, which records the
version. If nobody wrote a changelog entry, ``--note`` (the pull request title)
becomes the entry, so a release never ships with an empty section.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PARTS = ("major", "minor", "patch")
SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]*\))?(?P<bang>!)?: \S")
BREAKING = re.compile(r"^BREAKING[ -]CHANGE: ", re.M)


def bump_for(messages, labels=()) -> str | None:
    """The bump a merged pull request asks for, or ``None`` for no release."""
    for part in PARTS + ("none",):
        if f"release:{part}" in labels:
            return None if part == "none" else part
    level = 0
    for message in messages:
        if not message or not message.strip():
            continue
        match = SUBJECT.match(message.strip().splitlines()[0])
        if match is None:
            continue  # not a Conventional Commit: it asks for nothing
        if match["bang"] or BREAKING.search(message):
            level = 3
        elif match["type"] == "feat":
            level = max(level, 2)
        elif match["type"] in ("fix", "perf"):
            level = max(level, 1)
    return {3: "major", 2: "minor", 1: "patch"}.get(level)


def messages_since_last_tag(root: pathlib.Path = ROOT) -> list:
    """Every commit message on HEAD since the newest ``v*`` tag reachable from it."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout

    try:
        last = git("describe", "--tags", "--abbrev=0", "--match", "v[0-9]*").strip()
    except subprocess.CalledProcessError:
        return []  # no release yet: nothing to catch up on
    log = git("log", "--format=%B%x1e", f"{last}..HEAD")
    return [message.strip() for message in log.split("\x1e") if message.strip()]


def bumped(version: str, part: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _replace(path: pathlib.Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    new, count = re.subn(pattern, replacement, text, count=1, flags=re.M)
    if count != 1:
        raise SystemExit(f"{path}: nothing matched {pattern!r}")
    path.write_text(new, encoding="utf-8")


def date_changelog(text: str, version: str, date: str, note: str = "") -> str:
    """``[Unreleased]`` becomes ``[version] - date``; an empty one goes on top."""
    head = "## [Unreleased]\n"
    start = text.index(head) + len(head)
    following = re.search(r"^## \[", text[start:], re.M)
    end = start + following.start() if following else len(text)
    body = text[start:end].strip()
    if not body:
        if not note:
            raise SystemExit("CHANGELOG.md has nothing under [Unreleased]")
        body = f"- {note}"
    return text[:start] + f"\n## [{version}] - {date}\n\n" + body + "\n\n" + text[end:]


def current_version(root: pathlib.Path = ROOT) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def apply(part: str, root: pathlib.Path = ROOT, date: str = "", note: str = "") -> str:
    old = current_version(root)
    new = bumped(old, part)
    date = date or dt.date.today().isoformat()
    _replace(root / "pyproject.toml", r'^version = "[^"]+"', f'version = "{new}"')
    _replace(
        root / "jupyddl" / "__init__.py",
        r'^__version__ = "[^"]+"',
        f'__version__ = "{new}"',
    )
    _replace(root / "CITATION.cff", r"^version: .+$", f"version: {new}")
    changelog = root / "CHANGELOG.md"
    changelog.write_text(
        date_changelog(changelog.read_text(encoding="utf-8"), new, date, note),
        encoding="utf-8",
    )
    return new


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="read the merged PR as JSON on stdin")
    plan.add_argument(
        "--since-last-tag",
        action="store_true",
        help="also count every commit since the last v* tag",
    )
    cut = sub.add_parser("apply", help="bump, date the changelog, rebuild web/dist")
    cut.add_argument("part", choices=PARTS)
    cut.add_argument("--note", default="", help="changelog entry if there is none")
    cut.add_argument("--date", default="", help="release date (default: today)")
    args = parser.parse_args(argv)

    if args.command == "plan":
        pr = json.load(sys.stdin)
        messages = list(pr.get("messages", []))
        if args.since_last_tag:
            messages += messages_since_last_tag()
        print(bump_for(messages, pr.get("labels", [])) or "none")
        return 0
    version = apply(args.part, date=args.date, note=args.note)
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_web.py")], check=True)
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
