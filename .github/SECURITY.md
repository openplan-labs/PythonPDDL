# Security policy

## Supported versions

Fixes go into the latest release on PyPI. Older releases are not patched, so
upgrade before you report.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately through GitHub instead: on the repository's **Security**
tab, choose **Report a vulnerability**. If that option is missing, open an
issue that only asks for a private contact. Leave out all details of the
problem.

Include the input that triggers it (domain and problem files, a trace or
model file) and the exact command you ran. We aim to acknowledge a report
within a week.

## What counts

jupyddl reads planning files and its own JSON artefacts, so these are in
scope:

- Code execution, or reading and writing files outside the paths you gave,
  from a crafted PDDL file, search trace (`SearchTrace.load`) or learned
  heuristic bundle (`learned:<path>`). These formats are plain JSON and text,
  and none of them should ever execute anything.
- Script injection in the browser workbench from a crafted domain, problem or
  trace.

These are **not** vulnerabilities:

- Planning is PSPACE-complete in general. A small PDDL file can ground into an
  enormous task or take a very long time to search. Use a budget
  (`--time-limit`, `--max-expansions`) when you plan on input you do not
  trust.
