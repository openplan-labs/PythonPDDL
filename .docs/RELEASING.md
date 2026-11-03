# Releasing jupyddl

## The normal way: merge a pull request

Releases are **semantic and automatic**. Nobody picks a version number, and
nobody approves anything. **Merging a pull request is the release decision.**

On every push to `main`, `release.yml` looks up the pull request that was just
merged and takes the largest bump that its title and its commits ask for
(`tools/release.py`):

| Commit or title | Release |
|---|---|
| `fix: …`, `fix(lmcut): …`, `perf: …` | patch, 2.4.0 → 2.4.1 |
| `feat: …` | minor, 2.4.0 → 2.5.0 |
| `feat!: …`, `fix!: …`, or a `BREAKING CHANGE:` footer | major, 2.4.0 → 3.0.0 |
| `docs:`, `chore:`, `ci:`, `test:`, `refactor:`, `build:`, `style:` | none |

- **Override with a label.** Add `release:major`, `release:minor`,
  `release:patch` or `release:none` to the pull request before merging, for
  the rare change whose commits say the wrong thing.
- **Nothing is lost.** Commits on `main` since the last `v*` tag count too.
  So a merge whose run was dropped, because several merges landed at once,
  still releases with the next one.

When a release is due, the workflow:

1. bumps `pyproject.toml`, `jupyddl.__version__` and `CITATION.cff`;
2. turns `## [Unreleased]` in `CHANGELOG.md` into `## [X.Y.Z] - date`, using
   the PR title if nobody wrote an entry;
3. rebuilds `web/dist`, which records the version;
4. commits that to `main` as `chore(release): X.Y.Z`, tags `vX.Y.Z`, and
   starts `pages`;
5. builds, verifies, publishes to PyPI and cuts the GitHub Release, with every
   check described below.

It pushes with the workflow's own token, which is allowed to push commits and
tags. That is all it needs: there is no release pull request, so the *Allow
GitHub Actions to create pull requests* setting is not needed. Those pushes
start no workflows, so nothing loops.

**Requirements:**

- Squash-merge pull requests with the **PR title** as the commit message
  (*Settings → General → Pull Requests*). The title is what `pr-title`
  checks.
- `main` must accept pushes from GitHub Actions. If you add branch protection,
  let `github-actions[bot]` push, as the `format` workflow already needs.
- The `pypi` environment must have no required reviewers, or every release
  waits for one.

## By hand

`.github/workflows/release.yml` builds, verifies, publishes to PyPI and
creates the GitHub Release from the changelog. You can also start it yourself,
in one of two ways.

**From the Actions tab** — no local git, nothing depending on one person's
laptop:

> **Actions → release → Run workflow → target: `pypi`**

That path creates the tag itself, at the commit it just built and tested, so
the tag cannot end up pointing at something that was never verified.

**Or by pushing a tag**, if you prefer:

```bash
git tag -a v2.3.0 -m "jupyddl 2.3.0"
git push origin v2.3.0
```

Both run the same checks and produce the same artifacts. Choosing `testpypi`
from the Actions tab rehearses everything and deliberately creates neither a
tag nor a release.

## One-time setup on PyPI (a human has to do this)

The workflow authenticates with **Trusted Publishing** (OIDC), so there is no
API token stored in this repository — nothing to leak, nothing to rotate, and
nothing that keeps working if someone walks off with a laptop. The trade is one
piece of setup that can only be done by a PyPI maintainer of the project.

On <https://pypi.org/manage/project/jupyddl/settings/publishing/>, add a
publisher with **exactly** these values:

| Field | Value |
|---|---|
| Owner | `openplan-labs` |
| Repository name | `PythonPDDL` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The environment name matters: the `pypi` job declares
`environment: pypi`, and PyPI will reject a token minted from anywhere else.
That scoping is the point — a workflow added later by someone else cannot
publish unless it also runs in that environment.

Repeat on <https://test.pypi.org> with environment name `testpypi` if you want
the dry run below to work.

Until this exists the `pypi` job fails with an OIDC error. Nothing else is
blocked by it: the build still verifies, the tag is still created and the
GitHub Release is still cut with the artifacts attached. Re-running the failed
job once the publisher is configured completes the release.

That decoupling is deliberate. A GitHub Release records that a version was cut
from a verified commit; PyPI is a downstream channel that can fail for reasons
the code has nothing to do with. Losing the release because the upload failed
would leave no record of the version and force a full re-run to get one.

### Optional: require a human to approve each publish

In **Settings → Environments → pypi**, add yourself as a required reviewer.
GitHub then pauses the `pypi` job until someone approves it. Worth it if you
would rather a mistaken tag not reach PyPI unattended.

## Dry run

Validate the whole pipeline without spending a version number:

**Actions → release → Run workflow → target: `testpypi`**

A PyPI release is effectively permanent. A version can be *yanked*, which hides
it from resolvers, but it can never be replaced or re-uploaded — so the version
number is spent either way. Two minutes on TestPyPI is cheap next to that.

## Cutting a release

1. **Land everything on `main`** and confirm CI is green. Steps 2-5 prepare the
   commit; step 6 is either the Actions button above or a tag push.
2. **Bump the version in two places** — they are checked against each other and
   against the tag, and a mismatch fails the build rather than publishing a
   surprise:
   - `pyproject.toml` → `version`
   - `jupyddl/__init__.py` → `__version__`
3. **Close the changelog section.** Rename `## [Unreleased]` to
   `## [X.Y.Z] - YYYY-MM-DD`. The workflow extracts exactly this section as the
   GitHub Release body, so what you write here is what people read.
4. **Rebuild the browser bundle** — `web/dist/build.json` carries the version:
   ```bash
   python tools/build_web.py
   ```
5. Commit, push, merge.
6. **Cut it**: *Actions → release → Run workflow → `pypi`*, or tag the merge
   commit and push the tag.

## What the workflow refuses to do

Each of these is a way a release goes wrong quietly, so each one is a hard
failure rather than a warning:

- **Tag disagrees with `pyproject.toml`.** Publishing `2.3.0` from a tag reading
  `v2.4.0` cannot be undone.
- **`jupyddl.__version__` disagrees with the metadata.** A package that reports
  a different version at runtime than the one you installed is a support ticket
  with no obvious cause.
- **`twine check --strict` finds anything.** Malformed metadata renders as raw
  text on the project page and is only fixable with another release.
- **The built wheel does not work.** It is installed into a clean environment,
  away from the source tree, and made to generate and solve an instance. An
  editable install hides a module missing from the wheel; this does not.

## Versioning

[Semantic versioning](https://semver.org). In practice:

- **patch** — fixes that change no API and no plan output.
- **minor** — new planners, heuristics, generators, CLI commands, or PDDL
  requirement support. Everything that worked still works.
- **major** — a removal or a change in behaviour that existing code would
  notice. The 1.0.0 rewrite that removed Julia is the model.

Note that changing what a heuristic *returns* can change which plan comes back
even when the API is untouched. Plans are not part of the compatibility
promise; costs and validity are.

## History

PyPI carried `0.4.1` — the original Julia-backed wrapper — for a long time
after the pure-Python rewrite landed here, because there was no release
automation and nobody published by hand. `pip install jupyddl` therefore gave
people a different library than this repository documented. That is what this
workflow exists to prevent.
