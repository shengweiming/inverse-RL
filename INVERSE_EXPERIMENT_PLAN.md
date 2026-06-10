# Can RL Teach *Inversion*? — Experiment Plan (Llama-3.2-3B)

**Status:** Stage 1 ✅ (Gate G1 passed) · Stage 1.5 ✅ (Gate G2 passed) · Stage 2 design-complete, build pending.
**This document is the canonical science spec.** Operational build order lives in
`EXECUTION_GUIDE.md`; agent rules live in `CLAUDE.md`. When a pasted task and this
plan disagree, this plan wins — flag the conflict instead of guessing.

---

## 1. Research question & hypotheses

Yuan et al. (RL-Compositionality, arXiv:2509.25123) show that RL on Level-2
*compositions* teaches a generalizable composition skill that RFT on the same data
and RL on Level-1 atomic problems do not. We transplant their logic from the
**level axis** to the **task axis**: composition → **inversion**.

The model is trained (Stage 1) only on *forward* application of opaque skills
`func_N`. It exhibits a clean reversal curse at baseline (see §5). The question:

> **H1 (headline).** GRPO on *inverse* problems over SEEN skills improves inverse
> accuracy on HELD-OUT skills — inversion is learned as a transferable operation,
> not as per-skill memorized inverse algorithms.

> **H2 (RL vs RFT).** Iterative rejection fine-tuning on the identical inverse
> problems, at matched rollout budget, transfers substantially less (mirrors the
> paper's RL-vs-RFT contrast, with a deliberately *strengthened* RFT baseline —
> see §6 Arm B).

> **H3 (mechanism, pre-registered).** The transferable strategy is
> **propose-and-verify**: guess a candidate input, simulate the chain *forward*
> in-CoT (a skill the model has at 96.8%), check against the target output,
> revise. Per-skill inverse algorithms cannot transfer across skills; the
> meta-strategy can. Prediction: successful held-out inversions show forward
> simulation of `func_N` on candidate strings in the CoT, and the frequency of
> this pattern rises over RL training.

> **H4 (control).** RL on *forward* problems (Level-2 compositions, SEEN) does
> not improve held-out inversion: it is the incentive to invert, not RL-on-
> anything, that matters. (Mirrors the paper's RL-Level-1 control.)

Negative results are informative: if H1 fails because hard inverse problems never
enter the trainable set (coverage starvation, §7), that is itself a finding about
the limits of on-policy RL — connect to the sharpening / coverage literature.

## 2. Relationship to the original paper

We reuse the paper's surface **wholesale**: their `func_N` numbering, their input
distribution (lowercase ascii, length 3–10, no spaces), their hidden-definition
prompts (byte-identical `FORWARD_PROMPT`), their published Stage-1 RFT corpus,
and their reward semantics (binary programmatic verifier; inverse reward is a
**functional preimage match** — re-execute the chain on the candidate and compare
outputs, exactly their `compute_score_backward`).

What we change: (a) 9 skills instead of 25 (the paper's 9 invertible-and-distinct
string transforms; `func_1≡func_18` and `func_8≡func_19` are behavioral dupes);
(b) Llama-3.2-3B-Instruct instead of 3.1-8B (Colab budget); (c) Stage 2 trains
**inversion** instead of higher-level composition; (d) our RFT baseline is
strengthened relative to theirs (k=8 samples/problem vs their `N_SAMPLES=2`, plus
per-problem cap 2 to kill the easy-problem frequency skew) so the RL-vs-RFT
contrast is robust to the "RFT wasn't tried hard enough" critique;
(e) we add the pass@k divergence analysis as the sharpening-robust evidence
(their Fig. "reranking illusion"): a held-out gap that *grows* with k indicates
support expansion, a shrinking one indicates reranking.

## 3. The 9 skills, parameters, and the split

Chain element = `(skill_name, param)`; params vary per problem and render as
literals in code (`func_1(x, 3)`, `func_5(x, 'qz')`). Registry in
`skills_inverse.py` (forwards are verbatim paper code).

| skill | paper id | param | inverse | class |
|---|---|---|---|---|
| repeat_str | func_1 | n ∈ 2–4 | `s[:len//n]` | trivial |
| reverse_words | func_4 | — | involution | **identity on this distribution** |
| add_prefix | func_5 | 2–4 lc | strip prefix | trivial |
| add_suffix | func_6 | 2–4 lc | strip suffix | trivial |
| rotate_str | func_8 | n ∈ 1–3 | rotate back | structural |
| mirror_str | func_9 | — | first half | structural |
| insert_separator | func_13 | {-,_,\|} | `s[::2]` | structural |
| duplicate_every_char | func_14 | — | `s[::2]` | structural |
| fancy_brackets | func_15 | — | `s[1::3]` | structural (semi-readable) |

**Split (current, post-swap):**
`HELD_OUT = [rotate_str, mirror_str, fancy_brackets]` ·
`SEEN = [repeat_str, reverse_words, add_prefix, add_suffix, insert_separator, duplicate_every_char]`.

**Why the swap** (was `reverse_words` held out): the paper's input generator
never emits spaces and no skill introduces one, so `reverse_words` is the
**identity on every reachable string**. The Stage-1 corpus is the model's only
source of `func_4` semantics, so its "inverse" is echo-the-output — useless as a
held-out structural probe (it scored 0.56 at baseline vs 0.000 for true
structural skills, inflating the held-out mean 3×). It stays SEEN; `mirror_str`
(baseline 0.000) is promoted. `rotate_str` (0.000 over 33 problems) is the
cleanest held-out skill for headline claims; treat `fancy_brackets` with caution
(0.171 baseline — its payload is visually present between brackets).

## 4. Stage 1 (done) — Gate G1 ✅

SFT on the paper's published Stage-1 RFT corpus (hidden-def prompts; ~30k rows
after filtering to our 9 skills), LoRA r=32, lr 1e-4, 2 epochs, A100-80GB,
~83 min. **G1 result: 96.8% forward level-1 accuracy, every skill ≥ 0.75
(lowest: rotate_str 0.846).** Checkpoint: `ckpts/stage1_sft/` on Drive.

## 5. Stage 1.5 (done) — Gate G2 ✅ (reversal-curse baseline)

Inverse pass@1 on level-1 eval problems, hidden-def prompts, k=8 at temp 0.7
(pre-swap split; per-skill numbers are split-independent):

- Trivial mean (affixes + repeat): **0.472** — copy-a-substring inversions work.
- Structural mean: 0.132; **excluding reverse_words: 0.045**.
- rotate_str **0.000**/33 · mirror_str **0.000**/24 · duplicate_every_char
  **0.000**/10 · insert_separator 0.056/18 · fancy_brackets 0.171/35 ·
  reverse_words 0.563/32 (identity passenger, as predicted).

Gate G2 PASS (structural < 0.30 threshold). The ~10× trivial-vs-structural gap is
the contrast Stage 2 needs. Artifacts: `results/stage1p5_inverse_skill_detail.csv`
(per-problem; archive the pre-swap copy), `results/stage1p5_inverse.csv` (summary —
note: the first generated summary was stale-loaded from an old file; trust the
skill detail and regenerate the summary from it).

## 6. Stage 2 — design

Shared rollout budget **B** across arms (set at Gate G3; default proposal:
~400 GRPO steps × 16 prompts × 8 rollouts ≈ 50k rollouts). All arms start from
`ckpts/stage1_sft`. Reward: `verifier.batch_inverse_reward` /
`batch_forward_reward` (binary, programmatic, no reward model).

**Arm A — RL (the experiment).** TRL GRPO, LoRA r=32, G=8 rollouts/prompt,
temp 1.0, KL coef 0, max completion 512. Training pool: inverse Level-1 SEEN
(possibly mixed — see decision rule). Groups with zero reward variance carry no
gradient; log the wasted fraction. Periodic eval hook: inverse pass@1 on a fixed
held-out-L1 slice (n≈60) every ~25 steps → W&B.

**Arm B — strengthened iterative RFT (the baseline).** Loop until budget B is
spent: generate k=8 rollouts/problem at temp 1.0 with the current model →
verifier-filter to correct → **cap 2 kept responses per problem** (kills the
frequency skew toward easy problems; the paper kept all correct responses from
only 2 samples) → drop problems where all k were correct → SFT 1 epoch (lr 2e-5)
→ next iteration. Eval after each iteration. Expect ~3–5 iterations within B.

**Arm C — forward-RL control (optional, run last).** Same GRPO recipe on
*forward Level-2 SEEN compositions* (forward L1 is saturated at 96.8% — no
gradient survives group filtering). Tests H4.

**Training-pool decision rule (pre-registered, resolves at Gate G3):**
run the coverage probe (§7). If **≥4 of 6 SEEN skills have trainable fraction
≥ 0.2** (fraction of problems with 0 < successes < 16 at temp 1.0), train
inverse-only. Otherwise train a **50/50 mix** of inverse-L1-SEEN and
forward-L2-SEEN (precedent: the paper's best configuration was the mixed
L1+2 pool — easy problems keep gradient flowing while hard ones enter as
coverage allows).

## 7. Coverage risk (the known failure mode)

GRPO/DAPO-style training only learns from problems with mixed-correctness
groups. Baseline pass@1 on SEEN structural skills is ~0 (duplicate 0.000,
insert_separator 0.056); if pass@16 at temp 1.0 is also ≈0, RL never trains on
them and "inversion" is learned only from affix-stripping — H1's transfer hope
dies before training starts. Hence the mandatory **coverage probe** before any
trainer is built: 16 rollouts at temp 1.0 over ~500 `inv_l1_seen_train`
problems; report per-skill pass@16 and trainable fraction. The probe result is
Gate **G3** and feeds the decision rule above. (Free preview: the existing G2
detail CSV already contains pass@8 at temp 0.7 per problem.)

## 8. Evaluation protocol & metrics

Fixed eval sets (regenerated under `DATA_CONTRACT v3` after the split swap):
`inv_l1to4_eval.jsonl`, `fwd_l1to4_eval.jsonl` (levels 1–4 × seen/held-out
cells, 100 problems/cell, seeds fixed).

- **Primary:** inverse pass@1 and pass@8, level-1 **HELD-OUT**, per skill and
  macro-averaged (equal weight per skill — cell ns differ).
- Secondary: inverse seen-L1 (did training work at all); forward L1 retention
  (catastrophic-forgetting check, expect ≥0.90); inverse L2 (composition of
  inverses — exploratory).
- **pass@k divergence:** for final Arm-A vs Stage-1 base, sweep k ∈
  {1,2,4,…,64} on held-out L1. Growing gap with k ⇒ support expansion (new
  skill); shrinking ⇒ reranking. This is the sharpening-robust headline figure.
- **Mechanism (H3):** at each saved checkpoint, sample 20–30 CoTs on held-out
  inverse problems; manual read plus a crude automatic proxy (CoT contains a
  candidate string AND a forward application of a `func_N` to it). Report the
  proxy rate over training.

## 9. Gates

- **G1** ✅ forward L1 ≥ 0.75 every skill after Stage 1.
- **G2** ✅ structural inverse pass@1 < 0.30 at baseline (reversal curse exists).
- **G3** — coverage probe complete; pool decision made by the rule in §6;
  budget B fixed. *No trainer runs before G3.*
- **G4** — RL sanity: training reward rises within first ~100 steps; completion
  length not blowing up (>1.5× median baseline = investigate); zero-variance
  fraction < 0.8.
- **G5** — headline: Arm A held-out inverse pass@1 exceeds Stage-1 base by ≥10
  points macro-averaged, and exceeds Arm B at matched budget. Partial passes are
  reportable (see H1 negative-result note).

## 10. Risks & mitigations

- **Coverage starvation** → probe + mixing rule (§6–7).
- **reverse_words contaminates seen-pool training signal** (its inverse is
  echo): cheap reward for the model; acceptable (it is one of six), but report
  per-skill training-reward composition so it's visible.
- **Stale skip-guards** → results CSVs gate regeneration by existence; delete
  the relevant CSV when the eval set changes (see EXECUTION_GUIDE gotchas).
- **Colab session death mid-RL** → checkpoint LoRA adapter to Drive every ~50
  steps; trainer must resume from latest adapter.
- **Reward hacking** — negligible: verifier is exact functional match; only
  degenerate risk is reverse_words echo (above).
- **Budget** — Arm A ≈ 50k rollouts × ≤2560 tokens on A100-80GB: roughly 6–10
  GPU-hours; Arm B similar; probe ~1 L4-hour. Within remaining Colab units.

### Provenance
Paper artifacts verified against the v3 arXiv source and the PRIME-RL repo
(scripts under `bash/section41_42/`): Stage-1 generation ran *with* definitions
visible then stripped them for training; Stage-2 RFT used `N_SAMPLES=2`,
1 epoch/iteration, keep-all-correct filtering; RL was DAPO-style (batch=minibatch
=16, 16 rollouts, temp 1.0, KL=entropy=0, all-correct/all-incorrect groups
filtered), 1 epoch over 50k problems. Stage-2 RFT datasets were never published;
per-iteration sizes (~13–15k instances at their accuracies) are inferred.
