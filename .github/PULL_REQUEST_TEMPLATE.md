<!-- Title: a Conventional Commit — `fix(scope): …` (patch), `feat: …` (minor),
`feat!: …` (major), or `docs:` / `test:` / `chore:` / `ci:` (no release). It
becomes the squashed commit that decides the next version. -->

## What and why

<!-- One or two sentences: what changes, and the problem it solves. Link the issue if there is one. -->

## How it was checked

<!-- The test you added, the command you ran, the numbers you measured. A performance claim needs the machine and the instance. -->

## Checklist

- [ ] `flake8 jupyddl tests tools` and `pytest` pass locally
- [ ] Anything under `jupyddl/` changed → `python tools/build_web.py` was re-run and `web/dist` is committed
- [ ] A new planner or heuristic cites its paper (name and year) in the docstring
- [ ] If this moves a planner in or out of `OPTIMAL_PLANNERS`, or narrows a guarantee, the description says so
- [ ] `CHANGELOG.md` has an entry under **Unreleased** for anything a user would notice
