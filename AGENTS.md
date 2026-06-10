# Agent instructions

Canonical agent instructions live in **`CLAUDE.md`** (kept tool-agnostic — it
applies to Codex, Claude Code, and any other coding agent working in this
repo). Read it first, then:

- **`INVERSE_EXPERIMENT_PLAN.md`** — the science spec. Canonical on any
  conflict with a pasted task: flag the conflict, don't guess.
- **`EXECUTION_GUIDE.md`** — operational build order and current step.

Non-negotiables (duplicated here so they survive a lazy read; full list in
CLAUDE.md): never edit the verbatim paper skill bodies or `FORWARD_PROMPT`;
`ID_MAP` paper numbering is frozen; bump `DATA_CONTRACT` + the partition test
on any generation/split change; `pytest -q` (57 green) and
`python3 skills_inverse.py` (`ALL GOOD`) before handoff; never exec/eval model
output.
