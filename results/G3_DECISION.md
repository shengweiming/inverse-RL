# Gate G3 — Coverage Probe Decision Record

**Date:** Stage-2 pre-flight.
**Inputs:** `results/coverage_probe.csv` (500 problems from `inv_l1_seen_train.jsonl`,
v4 contract; k=16 rollouts/problem, temperature 1.0, top_p 1.0 — matched to the
GRPO rollout distribution). Checkpoint: merged `ckpts/stage1_sft`.
**Rule applied:** plan §6 — inverse-only pool iff ≥4 of 6 SEEN skills have
trainable fraction (0 < successes < 16) ≥ 0.2; otherwise 50/50 inverse-L1-SEEN +
forward-L2-SEEN.

## Decision 1 — Training pool: **inverse-only.**

All 6/6 SEEN skills clear the 0.2 trainable-fraction bar:

| skill | n | pass@16 | trainable frac | mean tok |
|---|---|---|---|---|
| repeat_str | 79 | 0.823 | 0.823 | 331 |
| reverse_words | 81 | 1.000 | 1.000* | 333 |
| add_prefix | 100 | 0.820 | 0.810 | 260 |
| add_suffix | 82 | 0.817 | 0.817 | 269 |
| insert_separator | 86 | 0.488 | 0.488 | 331 |
| fancy_brackets | 72 | 0.528 | 0.528 | 349 |

The four genuinely-structural/affix skills sit in the GRPO sweet spot
(p ≈ 0.49–0.82, balanced advantage magnitudes). No forward mixing is needed; the
pool is not coverage-starved. Rule satisfied honestly (decision rule was fixed
before the probe ran).

\* See Caveat below — reverse_words' "trainable 1.000" is a reporting artifact,
not real signal.

## Decision 2 — `max_completion_len`: **keep 512.**

This was an open question after the probe showed p95_tokens = 512 on ~90% of
problems (≥1 truncated rollout per problem). The aggregate diagnostic resolves it
**against** raising the cap:

- **The cap clips tails, not bodies.** Per-problem *mean* completion length: median
  312, max 456 — **zero** problems have a mean near 512. The typical rollout
  finishes with room to spare; truncation hits only a 1–2-rollout tail per problem.
  The model does not need >512 tokens to solve these.
- **Long = lost, not long = thorough.** On the two hard skills, length correlates
  *negatively* with success: insert_separator −0.23, **fancy_brackets −0.69**.
  When this model goes long here it is spiralling/looping, not reasoning more
  carefully. Raising the cap would protect degeneration, not reasoning.

Decision: 512 stays. It correctly truncates junk; the small signal lost to
tail-truncation is in the harmless (under-counting) direction.

**Pending manual confirmation (does not block launch):** read 8–10 of the
*longest* fancy_brackets rollouts in Colab to confirm they are degenerate rambles
rather than coherent-but-unfinished reasoning. The −0.69 correlation makes
degeneration the overwhelming prior; this is a 5-minute verification, not a
reopening. If those runs are coherent, reopen the cap question; if junk,
analysis is fully confirmed. (Inspection cell: `scripts/inspect_capped.py` /
notebook snippet provided alongside this record.)

## Caveat — reverse_words is a zero-gradient free-rider.

reverse_words solves 16/16 on (nearly) every problem → p = 1 → group std = 0 →
advantage masked to 0 → **contributes no gradient**, despite the table's
"trainable 1.000" (the summary counts 0 < c < 16, and a true 16/16 is c = 16; the
1.000 reflects rounding from problems at 15/16). This is expected: reverse_words
is the identity on the space-free input distribution, so its "inverse" is
echo-the-output, which the model already does perfectly. It is harmless ballast in
the pool (≈1/6 of prompts flowing zero gradient early), not a contaminant. **Action:**
log per-skill training-reward composition (plan §10) so its zero contribution is
visible and confirmable during Arm A.

## Decision 3 — Budget B.

Default proposal stands: **~400 GRPO steps × 16 prompts/step × 8 rollouts/prompt
≈ 50k rollouts**, shared identically with the Arm-B RFT baseline. (Revisit only if
the Arm-A smoke run reveals throughput makes 400 steps impractical in one A100
session; checkpoint every 50 steps regardless so the run is resumable across
sessions.)

## Refined Gate G4 (length dynamics).

Propose-and-verify (H3) is *longer* than the model's current confident-short
successes, but it is **structured** length, opposite to today's spiral length.
So G4 must read length jointly with reward, not in isolation:

- **Healthy (strategy emerging):** mean completion length rises **and** reward
  rises together.
- **Alarm (degeneration / length-hacking):** length or capped-fraction rises while
  reward is flat or falling.

Baselines to measure against (from this probe): median rollout ≈ **312 tokens**;
capped-fraction is tail-only today (~0% of problems have mean length near cap).
A rising capped-fraction during training is the early-warning signal.

## Summary of what carries into Stage-2 config.

- Pool: inverse Level-1 SEEN only (6 skills; reverse_words is ballast).
- `max_completion_len = 512`, `max_prompt_len = 2048`.
- Rollout sampling: temperature 1.0, top_p 1.0 (matched to this probe).
- Budget B ≈ 50k rollouts; checkpoint every 50 steps.
- Logging: per-step mean reward, **per-skill reward composition**, mean & capped
  completion length, zero-variance-group fraction → W&B.
- G4 watches length+reward jointly against ~312-token / tail-only baselines.

_Frozen: no further split or pool changes after this record._
