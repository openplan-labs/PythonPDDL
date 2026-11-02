---
name: Propose an algorithm
about: A planner, heuristic, compilation or generator you would like to see in jupyddl, or want to add yourself
title: '[algorithm] '
labels: 'feature'
assignees: ''

---

**Which algorithm?**
Name, paper and year, for example *LM-cut — Helmert & Domshlak, ICAPS 2009*.
Link the paper if it is open access.

**What kind is it?**
Planner (`jupyddl.search.PLANNERS`), heuristic (`jupyddl.heuristics.HEURISTICS`),
PDDL compilation (`jupyddl/compile.py`), instance generator
(`jupyddl.generator.GENERATORS`), or something else.

**What does it guarantee?**
Optimal, complete, admissible, anytime? Under which assumptions, for example
"admissible only without conditional effects"?

**Why is it worth having here?**
What does it show or teach that the existing ones do not?

**Do you want to implement it?**
Yes / no / with help. Read [CONTRIBUTING.md](../CONTRIBUTING.md) first.
