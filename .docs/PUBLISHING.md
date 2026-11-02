# Publishing jupyddl

**We intend to publish jupyddl as citable, peer-reviewed research software.**
This page says why, where, what is already in place, and what is left.

## Why

jupyddl's goal is to be a framework other people build on: a place to
implement a planning algorithm in readable Python, check it against the ones
already here, and *explain* it. You can watch a search run, chart it, replay it
in the browser and publish the result. That goal only works if people can find
the project, trust it, and give it credit. A software paper does all three. It
also gives everyone who extends jupyddl something stable to cite.

## The plan

| # | Venue | What it gives us | Status |
|---|---|---|---|
| 1 | [Zenodo](https://zenodo.org) | A DOI for **every release**, so a paper can cite the exact code it ran | Needs a one-time maintainer setup ([below](#zenodo-one-time-setup)) |
| 2 | [JOSS](https://joss.theoj.org) — *Journal of Open Source Software* | Peer review of the software itself, held in public on GitHub. Free to publish and free to read; the paper gets a journal DOI | Planned. [Readiness checklist below](#joss-readiness) |
| 3 | ICAPS workshops | Peer review of the *research* results. **PRL** (Planning and Reinforcement Learning) for the learned-heuristic work, **PlanEdu** or the system-demo track for the workbench | After the evaluation protocol in [roadmap §3](roadmap.md#3-a-proper-evaluation-protocol). The current results are single-seed and the notes say so |
| 4 | arXiv | A preprint of the learned-heuristic write-up, citable immediately | Optional, alongside 3 |

We are not targeting journals that charge to publish (SoftwareX, JORS). JOSS
covers the same ground for free.

## Citing jupyddl today

Use **"Cite this repository"** on the GitHub page. It reads
[`CITATION.cff`](../CITATION.cff). The current preferred citation is a 2021
preprint that describes the *original* project, a thin wrapper around the
Julia `PDDL.jl` parser. When the JOSS paper is accepted, it replaces that
preprint in `CITATION.cff` and in the README's [Cite](../README.md#cite)
section. Each release's Zenodo DOI goes in `CITATION.cff` as an identifier.

## Zenodo one-time setup

This needs a maintainer and takes about ten minutes:

1. Sign in at <https://zenodo.org> with the GitHub account that administers
   `openplan-labs`.
2. Open **Settings → GitHub** and switch on `openplan-labs/PythonPDDL`. Because
   the repository belongs to an organisation, the Zenodo GitHub app may first
   need to be granted access to `openplan-labs` under that organisation's
   *Third-party access* settings.
3. Cut a release as usual ([RELEASING.md](RELEASING.md)). The `release`
   workflow publishes a GitHub Release, and Zenodo archives every published
   release and mints a DOI for it. Zenodo takes its metadata from
   `CITATION.cff`, so there is nothing else to fill in.
4. Add the *concept* DOI to `CITATION.cff` under `identifiers`, and as a badge
   to the README. The concept DOI always resolves to the latest version.

## JOSS readiness

JOSS reviews the software against a public checklist. Here is where we stand.
JOSS revised its scope and submission requirements in 2026, so **re-read the
current [submission guide](https://joss.readthedocs.io/en/latest/submitting.html)
and [policies](https://joss.readthedocs.io/en/latest/policies.html) before
submitting**. This list is a working copy, not a substitute.

| Requirement | Status |
|---|---|
| OSI-approved licence | ✅ Apache-2.0 |
| Public repository, open issue tracker, history in the open | ✅ |
| Substantial scholarly effort | ✅ Parser and grounder covering 20 of 21 PDDL requirement flags, 14 planners, LM-cut and `h^m`, PDDL 3 compiled to the classical core, learned heuristics, search tracing |
| Installation instructions | ✅ README → Install; `pip install jupyddl` |
| Example usage | ✅ README, `demos/`, the browser workbench |
| Automated tests | ✅ CI on Python 3.10–3.14, with and without the optional extras, plus a packaging check on the built wheel |
| Community guidelines | ✅ [`CONTRIBUTING.md`](../.github/CONTRIBUTING.md), [`CODE_OF_CONDUCT.md`](../.github/CODE_OF_CONDUCT.md), [`SECURITY.md`](../.github/SECURITY.md) |
| API documentation | 🟡 Docstrings throughout, but no rendered API reference yet |
| A guide to extending the framework | 🟡 A paragraph in the README's Architecture section; a full guide is in progress |
| Archived release with a DOI | ⏳ Waiting on the Zenodo setup above |
| `paper.md` and `paper.bib` | ⏳ Not written yet. See the outline below |
| Author list, with ORCIDs | ⏳ See [Authorship](#authorship) |
| AI-usage disclosure | ⏳ Required in the paper. See [below](#ai-usage-disclosure) |

### Paper outline

JOSS papers are short, roughly 250–1000 words. The outline:

- **Summary.** A pure-Python PDDL planning framework meant to be read,
  extended and taught with: no native dependencies, every search observable,
  and the same registries behind the CLI, the benchmark harness and the
  browser workbench.
- **Statement of need.** Researchers and students who want to *implement and
  explain* a planning algorithm, not only run one. The production planners are
  fast, but you have to learn their C++ before you can change anything. The
  Python libraries are easy to read but stop at STRIPS, or at parsing.
- **State of the field.** Position jupyddl honestly against Fast Downward,
  pyperplan, the Unified Planning Framework, Tarski and the `pddl` parsing
  library: what each is for, and where jupyddl overlaps and differs.
- **Design.** Conditions in negation normal form, PDDL 3 compiled away before
  grounding, cooperative budgets, zero-overhead observers, and why the core has
  no runtime dependencies.
- **Research use.** The learned-heuristic line, including its negative results
  and the two published claims that were later corrected.
- **AI usage disclosure** and **Acknowledgements**.

### AI usage disclosure

JOSS allows generative AI in software and papers, provided the use is
disclosed and humans keep oversight. Much of the current code was written with
an AI coding assistant. Those commits carry a `Co-Authored-By: Claude` trailer,
so the history records which work was assisted. The paper must state this
plainly. It should say which parts were assisted, and how the results were
checked: the test suite, the observed-equals-unobserved invariant, and
measurements re-taken at render time rather than typed in.

### Authorship

The maintainers settle the author list before submission, following JOSS's
authorship guidance. If you have made a substantial contribution, such as a
planner, a heuristic, a compilation or sustained review, and want to be
considered, say so in your pull request or open an issue.
