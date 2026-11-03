"""Finish a release-please pull request: file the hand-written notes, rebuild the bundle.

release-please bumps the version and writes a changelog section from the
conventional-commit titles of the merged pull requests. Two things in this
repository it cannot know about:

1. **The hand-written notes.** Contributors describe their change under
   ``## [Unreleased]`` in prose, which says far more than a PR title.
   release-please inserts its new section *above* that heading, so this moves
   the notes into the new section, under the generated lists, and puts an
   empty ``## [Unreleased]`` back on top for the next round.
2. **The web bundle.** ``web/dist/build.json`` records the version, and the
   ``pages`` workflow refuses to deploy a bundle that disagrees with the
   sources. A release that bumps the version without rebuilding would break
   the site the moment it merged.

Idempotent: on a changelog that is already in shape it changes nothing, so the
workflow can run it after every push release-please makes to its branch.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
UNRELEASED = re.compile(r"^## \[Unreleased\][^\n]*\n", re.M)
VERSION = re.compile(r"^## \[?\d+\.\d+\.\d+[^\n]*\n", re.M)
HEADING = re.compile(r"^## ", re.M)


def file_unreleased_notes(text: str) -> str:
    """Move the ``[Unreleased]`` notes under the newest version heading."""
    unreleased = UNRELEASED.search(text)
    newest = VERSION.search(text)
    if unreleased is None or newest is None or unreleased.start() < newest.start():
        return text  # nothing to file, or already filed
    following = HEADING.search(text, unreleased.end())
    end = following.start() if following else len(text)
    notes = text[unreleased.end() : end].strip()
    text = text[: unreleased.start()] + text[end:]

    # The newest version's section now ends at the next heading after it.
    newest = VERSION.search(text)
    following = HEADING.search(text, newest.end())
    section_end = following.start() if following else len(text)
    section = text[newest.start() : section_end].rstrip()
    if notes:
        section += "\n\n" + notes
    section += "\n\n"
    return text[: newest.start()] + "## [Unreleased]\n\n" + section + text[section_end:]


def main() -> int:
    original = CHANGELOG.read_text(encoding="utf-8")
    updated = file_unreleased_notes(original)
    if updated != original:
        CHANGELOG.write_text(updated, encoding="utf-8")
        print("filed the [Unreleased] notes under the new version")
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_web.py")], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
