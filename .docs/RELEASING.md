# Releasing jupyddl

## The normal way: merge the release PR

Releases are **semantic and automatic**. Nobody picks a version number.

1. Every pull request is squash-merged, so its **title** becomes the commit
   on `main`. The `pr-title` check requires a
   [Conventional Commit](https://www.conventionalcommits.org) title:

   | Title | Release it triggers |
   |---|---|
   | `fix: …`, `fix(lmcut): …`, `perf: …` | patch, 2.4.0 → 2.4.1 |
   | `feat: …` | minor, 2.4.0 → 2.5.0 |
   | `feat!: …`, or a `BREAKING CHANGE:` footer | major, 2.4.0 → 3.0.0 |
   | `docs:`, `chore:`, `ci:`, `test:`, `refactor:`, `build:`, `style:` | none on its own; it rides along with the next one |

2. On every push to `main`, `release-please.yml` keeps **one release PR**
   (`chore: release X.Y.Z`) up to date. It bumps `pyproject.toml` and
   `jupyddl.__version__`, writes the changelog section from the merged
   titles, files the hand-written `## [Unreleased]` notes under it, and
   rebuilds `web/dist`, which records the version. The bump is the largest any
   merged title asks for.
3. **Merging the release PR is the release.** release-please tags the merge
   commit and creates the GitHub Release. It then starts `release.yml` on that
   tag, which builds, verifies, publishes to PyPI and attaches the artifacts,
   with the same checks as below.

Want a release after every merge? Merge the release PR right after. Want
batches? Let it collect. Either way the version follows from what went in.

**One-time repository settings** (a maintainer has to do these):

- *Settings → Actions → General → Workflow permissions*: tick **Allow GitHub
  Actions to create and approve pull requests**. Without it release-please
  cannot open its PR.
- *Settings → General → Pull Requests*: allow squash merging and set its
  default commit message to **Pull request title**, so the checked title is
  what lands on `main`.
- Do **not** make `pr-title` (or `tests`) a required check without an
  exception for release-please's PR. A PR opened with the workflow's own
  token starts no workflows, so those checks never report on it.
  `release.yml` repeats every check that matters before publishing.

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
