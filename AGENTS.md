# Agent instructions

This repo implements the experiment specified in **`INVERSE_EXPERIMENT_PLAN.md`**
(canonical design spec) and **`EXECUTION_GUIDE.md`** (step-by-step build order).
**Read both before writing any code.** When a pasted task and the plan disagree,
the plan wins — flag the conflict instead of guessing.

## What this project is (one paragraph)
A scaled-down replication of Yuan et al. (RL-Compositionality, arXiv:2509.25123)
on Llama-3.2-3B, comparing two operations over a fixed 25-skill library of
*invertible* string transforms: **composition** (positive control) vs
**inversion** (the new question). Two-stage curriculum: Stage 1 teaches the
forward skills via RFT; Stage 2 compares RL (GRPO) vs an RFT baseline on the
target operation. The model answers in JSON and never executes code.

## Hard constraints (do not violate)
- `skills_inverse.py` is **PROVIDED and TESTED — do NOT modify it.** Treat
  `SKILLS[name] = (forward, inverse, sampler, kwargs, tier, origin)` as fixed
  ground truth. Import from it; never reimplement a skill.
- **No `exec`/`eval` of model output. No external sandbox.** The verifier runs
  *our* reference functions only, by composing the known `chain`.
- All skills are **injective ⇒ the preimage is unique**, so the inverse reward
  is exact-match (`reference_apply(chain, x̂) == output`), not "any valid preimage."
- Plain importable modules, no install step. Tests live under `tests/` and must
  pass with `pytest -q`.
- Deterministic where seeded; data generators take `--seed`.

## Reward semantics (get these exactly right)
- **forward_reward**: parse `"output"` from the completion; 1.0 iff it equals
  `problem["output"]`, else 0.0.
- **inverse_reward**: parse `"input"` x̂; 0.0 if missing or non-str; else 1.0 iff
  `reference_apply(problem["chain"], x̂) == problem["output"]`, else 0.0.
- `chain=[f1,...,fk]` means `f_k(...f_1(x))`; apply each skill's `kwargs`.

## File manifest (build in EXECUTION_GUIDE order, Steps 1–7)
- `prompts.py` — `FORWARD_PROMPT`, `INVERSE_PROMPT`, `extract_last_json`
- `verifier.py` — `reference_apply`, `forward_reward`, `inverse_reward`,
  `batch_forward_reward`, `batch_inverse_reward`
- `inverse_tasks.py` — data gen (forward + inverse, levels 1–4) with a
  **mandatory round-trip rejection filter** (`compose_inverse(chain, y) == x`)
- `tests/` — `test_verifier.py`, `test_tasks.py`
- `notebooks/inverse_rl_colab.ipynb` — the single resumable Colab notebook
- `requirements.txt`

## Held-out split (Stage-2 inverse training only; forward still taught in Stage 1)
`HELD_OUT = ["atbash","shift_digits","mirror_str","swap_pairs","reverse_words",
"positional_shift","rail_fence_2","riffle_shuffle"]`; the other 17 are `SEEN`.

## Per-step workflow
Each step's exact instructions are pasted to you from the `[CODEX]` blocks in
`EXECUTION_GUIDE.md`. Implement only the requested step, end with its `[CHECK]`
passing, and do not scaffold future steps unless asked (notebook cells 4–8 are
explicitly left as labeled empty stubs until their step).
