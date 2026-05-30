# Can RL Teach *Inversion*? — Experiment Plan (Llama-3.2-3B MVP)

**One-line summary.** Replicate the two-stage design of Yuan et al. (*“From f(x) and g(x)
to f(g(x))”*, arXiv:2509.25123, PRIME-RL/RL-Compositionality), but compare two **operations**
on a learned skill library: **composition** (their result — here a *positive control*) vs.
**inversion** (the new question). After a model learns a set of forward string-transformations,
test whether **RL (GRPO)** — vs. an **RFT baseline** — can teach it to *invert* them, and whether
that ability generalizes to deeper inverse problems and to held-out skills.

**Scope (MVP).** This is a minimum-viable result on a **single Colab GPU** with **Llama-3.2-3B**,
intended to (a) demonstrate the effect cleanly enough to be credible and (b) serve as a writing
sample / proof-of-competence to recruit a CS advisor. The Qwen cross-check that fully separates
“teach” from “elicit” (see §3) is deliberately deferred to a follow-on; a one-line roadmap is kept
so the omission reads as scoping, not oversight.

This document is the canonical spec for coding agents. The companion `EXECUTION_GUIDE.md` gives the
human-facing, step-by-step Colab/Codex workflow.

---

## 1. Research question & hypotheses

Composition is something base models can already partly do, so “RL teaches composition” is a modest
claim. **Inversion is a sharper test**: the *reversal curse* (Berglund et al. 2023; Zhu et al. 2024)
gives a strong, documented baseline of failure for ordinary supervised training. So:

> **RQ.** Given a model that reliably computes forward functions {f}, can RL teach it to compute
> preimages (`find x s.t. f(x)=y`) in a way that (a) beats an RFT baseline, (b) generalizes to
> inverting *deeper compositions* it never trained on, and (c) transfers to *held-out skills* whose
> forward form it knows but whose inverse it never trained on — and how does this compare to RL
> teaching **composition** on the identical skill set and model?

**Framing (kept light; this is an empirical paper).** Inversion = the **μ-operator** of
computability (search for an argument satisfying a predicate). Composition closes a skill set into
the *primitive-recursive* fragment; minimization is the ingredient reaching full computability. So
the composition-vs-inversion comparison concretely operationalizes the thesis question:
**which closure operations over its primitive skills does RL actually install?** We do **not** claim
the model becomes Turing-complete; we characterize *which operation* RL teaches and how it
generalizes.

**Hypotheses.**
- **H0 (positive control).** On our scaled-down setup, RL teaches **composition** and generalizes to
  deeper compositions better than RFT — reproducing the original’s central finding. *This is the
  control that makes everything else interpretable (see §3, §7).* *Prior: high if the setup is
  sound.*
- **H1 (teachability).** RL on inversion raises level-1 inverse accuracy far above the post-Stage-1
  zero-shot baseline. *Prior: high (~85%).*
- **H2 (RL > RFT on generalization).** RL generalizes to **deeper** inverse levels and **held-out
  skills** substantially better than RFT. *Prior: moderate.*
- **H3 (operation vs. function — the interesting one).** What RL installs is a **search / inversion
  procedure**, not memorized inverse functions: inverse ability transfers to held-out skills, and
  chain-of-thought shows difficulty-scaled simulate-and-search rather than one-shot inversion.
  *Prior: expect procedure-learning over function-learning, but either result is publishable.*

Every outcome is informative. If H2 fails (RL≈RFT) while H0 holds, that is clean evidence that RL
teaches composition but **not** inversion — a real asymmetry. If H3 lands on “procedure,” the
contribution is that even apparent *teaching* is compatible with RL installing **search**, which over
a known forward model is exactly the μ-operator.

---

## 2. Relationship to the original paper

**Kept identical (fidelity where it matters):**
- The **task surface**: a Python snippet defining `main_solution(x)` as a composition of atomic
  string ops; the model answers in JSON **without executing code**.
- The **two-stage protocol**: Stage 1 = atomic-skill acquisition via **RFT**; Stage 2 = the target
  operation, compared across **RL vs RFT**.
- The **“level/depth” ladder** and the **train-shallow / eval-deep** generalization test.
- The **reward semantics** of the repo’s `compute_score_backward` (a predicted input is correct iff
  it reproduces the target output) and `compute_score` (forward output match).
- The **25-skill count** and the **train/eval skill split** idea.

**Changed (and why):**
| Change | Original | Here | Why |
|---|---|---|---|
| Headline | composition only | **composition (control) + inversion (new)** | the operation comparison is the contribution |
| Target operation | composition `g(f(x))` | **inversion `f⁻¹(y)`** (+ composition as control) | sharper test vs. reversal-curse baseline |
| Skill set | 25 ops, many non-invertible | **25 invertible (injective) ops** | inverse must be well-defined & unique |
| Reward execution | `sandbox_fusion` remote sandbox | **local reference-function check** | Colab-friendly, safe (whitelisted ops) |
| Model | Llama-3.1-8B-Instruct | **Llama-3.2-3B-Instruct** | single-GPU Colab feasibility; see §3 |
| RL plumbing | veRL (Ray+vLLM+FSDP, multi-GPU) | **TRL `GRPOTrainer`** (single GPU) | see §3 |

**What “same training” can and cannot mean on Colab (read this).** The original Stage-2 RL is veRL
GRPO on an 8B model — a multi-GPU **cluster** workload (Ray + vLLM rollout workers + FSDP). It will
**not** run in a single Colab notebook on one GPU. We preserve the *training algorithm* (GRPO,
group-relative advantages, verifiable reward, two-stage curriculum) and the *task*, but run it
through TRL on a smaller model. If a multi-GPU machine becomes available later
(Lambda/RunPod/university cluster), the **same data + reward** drop into the original veRL scripts
unchanged to reproduce at 8B.

---

## 3. Architecture & model decisions

### 3a. Repo: **new lean repo, vendoring the original’s skills + verifier** (do NOT fork veRL)
- veRL is thousands of files of cluster infrastructure unusable on Colab; a coding agent will drown
  in it and it cannot run in one notebook anyway.
- We genuinely reuse only (i) the **skill function definitions** and (ii) the **reward semantics**
  (`compute_score` forward, `compute_score_backward` inverse). Both are tiny and reproduced here.
- “Fork for reference, build fresh for execution.” Keep the original cloned read-only for citation
  and the eventual 8B reproduction; develop in the new repo.

```
inverse-rl/
├── INVERSE_EXPERIMENT_PLAN.md      # this file
├── EXECUTION_GUIDE.md              # human step-by-step (companion)
├── skills_inverse.py               # 25 invertible skills — PROVIDED, TESTED
├── inverse_tasks.py                # data gen: forward + inverse, levels 1–4 — agent builds
├── verifier.py                     # local forward + inverse rewards — agent builds
├── prompts.py                      # FORWARD_PROMPT + INVERSE_PROMPT — agent builds
├── tests/                          # pytest: round-trip, injectivity, both rewards, data-gen filter
├── notebooks/
│   └── inverse_rl_colab.ipynb      # the SINGLE runnable notebook — agent builds
└── requirements.txt
```
`skills_inverse.py` is already written and **passes round-trip + injectivity tests for all 25
skills** plus a composition round-trip check; treat it as fixed ground truth.

### 3b. Model: **Llama-3.2-3B-Instruct** (shakedown on Llama-3.2-1B)
The constraints essentially force this choice, and it is a strong one:
- **Colab caps at A100-40GB.** GRPO with **colocated vLLM** shares one GPU between generation and
  training. An 8B model (~16GB bf16 weights) + a vLLM engine + a training process reliably exceeds
  40GB; a **3B** model (~6GB) leaves comfortable headroom. So matching the original’s 8B is
  infeasible on Colab, and Llama-4 Scout (17B-active / 109B-total MoE) needs all experts resident for
  training → also infeasible on 40GB.
- **1B is below the search-emergence floor (~1.5B)** for RLVR on synthetic search tasks; it likely
  won’t develop the internal search that H3 is about. Use 1B only for cheap pipeline shakedown.
- **3B is capable enough:** Llama-3.2-3B matches Llama-3.1-8B on tool use and beats the original
  GPT-4 on MATH — adequate, post-RFT, to execute these string ops in CoT.
- **Clean for the teach claim:** Llama-3.x is non-reasoning and not subject to the “RLVR only
  elicits latent ability” confound documented for Qwen (spurious/random rewards improve Qwen but not
  Llama/OLMo). Staying in the original’s family keeps the comparison honest.
- **Maximally supported:** as of mid-2026 it is the most-pulled small model, so GRPO recipes, vLLM
  support, and fine-tunes are abundant.

**Residual risk:** 3B may underfit **tier-3** skills (vigenère, rail-fence, deterministic-shuffle).
Gate G1 (§7) catches this; the fallback is to lean the headline on tiers 1–2 and report tier-3 as
capacity-limited — **not** to silently switch models.

**Deferred (follow-on, not MVP):** a **Qwen2.5-3B** cross-check. There the Qwen-vs-Llama gap is
itself the elicit-vs-teach measurement; replicating the inversion effect on a non-Qwen model
(Llama) already gives a credible teaching claim for the MVP, and the Qwen contrast strengthens it
later.

---

## 4. The task

**Forward prompt (Stage 1 + composition control; unchanged from original):**
```
You are given a code:

{code}

Can you predict the output of `main_solution("{input}")` without writing any code?
Please reason and put your final answer in the following json format:
{"output": <your output>}, where <your output> should be the final string.
```

**Inverse prompt (Stage 2 inversion; new — mirrors the forward one):**
```
You are given a code:

{code}

The call `main_solution(x)` returned "{output}".
Can you find an input x such that `main_solution(x)` returns exactly "{output}",
without writing any code? Please reason and put your final answer in the following
json format: {"input": <x>}, where <x> should be the input string.
```

- `{code}` = concatenated source of the atomic functions used plus
  `def main_solution(x): return <expr>` — identical construction to the original generator.
- Answer parsing: extract the last complete JSON object; read `"output"` (forward) or `"input"`
  (inverse).
- **No code execution by the model** (matches original). The verifier executes *our* reference
  functions, never model-written code.

**Rewards (`verifier.py`):**
- **Forward / composition** (`compute_score` port): parse `"output"`; reward 1 iff it equals the
  true `main_solution(x)`.
- **Inverse** (`compute_score_backward` port): parse `"input"` x̂; compute
  `ŷ = main_solution_reference(x̂)` by composing our **trusted** `skills_inverse.py` functions for
  this problem’s chain (we know the chain; we never `exec` model output); reward 1 iff `ŷ == y`.
Because all skills are **injective**, the valid preimage is **unique**, so functional-match ==
exact-match and there is no preimage reward-hacking. (Non-injective skills, where “any valid
preimage” becomes load-bearing, are a future extension.)

---

## 5. The 25 invertible skills

All **injective on their sampled domain** ⇒ unique inverse. Provided and tested in
`skills_inverse.py` as `SKILLS[name] = (forward, inverse, sampler, default_kwargs, tier, origin)`.
Origin: **R** = reused from original `string_data.py`, **N** = new.

| # | Skill | Tier | Origin | Inverse type |
|---|---|---|---|---|
| 1 | reverse | 1 | R | involution |
| 2 | swap_case | 1 | N | involution |
| 3 | atbash | 1 | N | involution (letter reflection) |
| 4 | complement_digits | 1 | N | involution on digits |
| 5 | shift_chars (Caesar, k) | 1 | R | parametric bijection |
| 6 | shift_digits (k) | 1 | N | parametric bijection on digits |
| 7 | duplicate_every_char | 1 | R | injective (take every other) |
| 8 | fancy_brackets «c» | 1 | R | injective (strip) |
| 9 | wrap_tag `<<…>>` | 1 | N | injective (strip affixes) |
| 10 | add_prefix | 1 | R | injective (strip prefix) |
| 11 | add_suffix | 2 | R | injective (strip suffix) |
| 12 | rotate_str (n) | 2 | R | parametric bijection |
| 13 | rotate_words (n) | 2 | N | parametric bijection (word level) |
| 14 | repeat_str (n) | 2 | R | injective (first L/n) |
| 15 | mirror_str | 2 | R | injective (first half) |
| 16 | swap_halves (even L) | 2 | N | involution |
| 17 | swap_pairs | 2 | N | involution |
| 18 | insert_separator (sep) | 2 | R | injective (split) |
| 19 | reverse_words | 2 | R | involution (word order) |
| 20 | succ_char (+k on codepoint) | 2 | N | parametric bijection — *successor-flavored* |
| 21 | deterministic_shuffle | 3 | R | fixed permutation (inverse perm) |
| 22 | positional_shift | 3 | N | position-dependent bijection |
| 23 | vigenere (key) | 3 | N | repeating-key cipher |
| 24 | rail_fence_2 | 3 | N | 2-rail transposition |
| 25 | riffle_shuffle (even L) | 3 | N | perfect-shuffle bijection |

**Tiers** drive both the capability gate and the analysis: forward is easy across tiers, but
**inverse difficulty rises with tier** (T3 permutations/ciphers need genuine reasoning/search).
H3’s “procedure not function” prediction is sharpest on T2–T3.

**Skill split for transfer (H3c).** Hold out a fixed subset (8/25, spanning all tiers, mixing R/N)
from **Stage-2 inverse training only**; their *forward* form is still taught in Stage 1. Suggested
held-out: `atbash, shift_digits, mirror_str, swap_pairs, reverse_words, positional_shift,
rail_fence_2, riffle_shuffle`. Train inverse on the other 17; evaluate inverse on all 25, reported as
**seen-inverse** vs **held-out-inverse**.

---

## 6. Levels (the inverse ladder)

`main_solution` at **level k** is a composition of **k** atomic skills `f_k ∘ … ∘ f_1`.
- **Forward task, level k:** predict `y = main_solution(x)`.
- **Inverse task, level k:** recover `x` from `y` — undoing **k** transformations in sequence, i.e.
  **applying an inverse step k times**. This mirrors the original depth ladder and realizes the
  “apply the inverse n times” idea.

**Train-shallowest-nontrivial, eval-deep (parallel design for both operations):**
- **Inversion:** the new skill is non-trivial already at **level 1** (invert one function). Train RL
  on **level 1**; eval generalization on **levels 1–4**.
- **Composition (control):** the new skill is trivial at level 1 (single function already known from
  Stage 1) and first non-trivial at **level 2**. Train RL on **level 2**; eval on **levels 1–4**
  (incl. level-1 retention).
Both: “train at the shallowest level where the operation is non-trivial, test depth generalization.”

**Data-gen correctness filter (mandatory).** Composition of injections is injective, but a random
chain can violate a downstream skill’s sampled domain (e.g. a `-` produced before
`insert_separator`’s inverse, or odd length into an even-only op). The generator MUST: build chain →
sample `x` from the **first** skill’s sampler → compute `y` → verify
`compose_inverse(chain, y) == x`; **reject and resample** otherwise. This guarantees every emitted
problem has a verified **unique** preimage. (Empirically ~90–95% of level-3/4 chains pass.)

**Ablation variant (optional):** “iterate a *single* skill” — `y = f^k(x)`, recover `x` (true
`f⁻ᵏ`). Cleaner isolation of iteration depth; report if time allows.

---

## 7. Protocol

**Stage 0 — base eval.** Untouched model on forward (sanity) and inverse (should be low). Floor.

**Stage 1 — atomic forward RFT (same as original).** Generate level-1 **forward** problems across all
25 skills; rejection-sample base-model rollouts (keep correct), SFT on survivors (RFT).
- **Gate G1:** forward accuracy per skill high (target mean ≥ ~0.90; every tier ≥ ~0.75). If 3B fails
  a tier, drop the worst T3 skills (don’t switch models). The whole experiment rests on the model
  *knowing the forward functions*.

**Stage 2a — COMPOSITION CONTROL (RL vs RFT).** *This is the positive control and a gate.* From the
Stage-1 checkpoint, train **RL (GRPO, reward=forward)** on **level-2** compositions and an **RFT**
baseline likewise; eval forward accuracy on **levels 1–4**.
- **Gate G0:** RL should beat RFT on depth generalization (levels 3–4) — reproducing the original. If
  it does **not**, the scaled-down setup is underpowered → fix (more steps / bigger model) **before**
  spending budget on inversion, because an inversion null would then be uninterpretable.

**Stage 1.5 — inverse baseline (reversal-curse gate).** Evaluate the Stage-1 model **zero-shot on
inversion**, level 1, pass@1 and pass@k (k≈8–16).
- **Gate G2:** inverse accuracy should be **low** (esp. T2–T3) despite high forward accuracy — the
  “not already there” baseline that makes a positive Stage-2b result meaningful. If it’s already
  high, the contrast is weak → lean on harder skills / report honestly.

**Stage 2b — INVERSION (RL vs RFT).** *The experiment.* From the **same** Stage-1 checkpoint, two
arms:
- **RL arm:** TRL `GRPOTrainer`, reward = inverse verifier, train on **level 1** (optionally also a
  level-1+2 mix if budget allows). Log to wandb.
- **RFT arm:** rejection-sample inverse rollouts from Stage-1 model, SFT on correct ones; matched
  data budget where feasible.
- **Eval (both):** levels 1–4, split seen-inverse vs held-out-inverse, pass@1 (+ pass@k).

**Headline comparison:** composition vs inversion depth-generalization, each RL vs RFT, on the same
skills/model/eval (see §8).

---

## 8. Metrics & headline figure

- **Headline figure:** two depth-generalization panels (composition, inversion); within each,
  accuracy vs eval level (1–4) for RL vs RFT. The predicted story: composition shows RL>RFT
  depth-generalization (control reproduces original); inversion is the test case.
- **Inverse pass@1 / pass@k** (k≈8–16), per level, per tier, seen vs held-out.
- **Transfer gap:** held-out-inverse minus seen-inverse pass@1 per model (small gap ⇒ operation-level
  learning ⇒ supports H3).
- **Forward retention:** forward accuracy after Stage-2 RL (catastrophic-forgetting check).
- **CoT diagnostics (for H3):** completion length vs tier/level; rate of explicit
  trial-and-error/self-check phrasing; whether failures are wrong-search vs no-search.

---

## 9. Function vs. procedure — how we read H3

The model never has a runtime verifier (it answers “without writing code”), so it must invert
**internally**. Two distinguishable mechanisms:
- **Direct inversion (function-learning):** short, monotone CoT; accuracy ~flat in inverse depth;
  little benefit from more tokens.
- **Internal simulate-and-search (procedure-learning = μ-operator):** CoT scales with difficulty,
  shows candidate-then-check structure, benefits from more thinking budget, and **transfers to
  held-out skills** (skill-agnostic procedure).
Expected signature: procedure-learning, esp. T2–T3. Report whichever occurs; both are contributions.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Uninterpretable inversion null** (“RL can’t teach it” vs “setup too weak”) | **Composition positive control (Stage 2a, Gate G0)** — the single most important guard for an MVP |
| **3B underfits forward skills** (8B→3B gap) | Tier gating G1; drop hardest T3; **do not** switch models silently |
| **Reversal curse doesn’t fire** (inverse already easy) | Gate G2; emphasize T2–T3; honest reporting |
| **Reward hacking via alternate preimages** | All skills **injective** ⇒ unique preimage; verifier checks functional == exact match |
| **Data-gen emits unsolvable/ambiguous problems** | Mandatory round-trip rejection filter (§6) |
| **Catastrophic forgetting under RL** | Track forward retention; KL control in GRPO; modest LR; LoRA |
| **veRL/Colab incompatibility** | New lean repo + TRL (§3) |
| **Colab crashes mid-run** | Drive checkpointing + resumable cells (EXECUTION_GUIDE) |
| **Budget overrun on 3B** | 1B shakedown; L4 for dev, A100 only for real runs; small G; defer inversion level-1+2 mix |
| **Elicit-vs-teach confound** | Llama (not Qwen) for the MVP; Qwen contrast deferred as the explicit separator |

---

## 11. Compute budget (target: under 500 Colab units; expect ~280–360)

Rough Colab unit costs: **A100-40GB ≈ 11–13 units/hr**, **L4 ≈ 4–5 units/hr**, **T4 ≈ 1.5–2/hr**.
Costs below assume **Llama-3.2-3B** with LoRA + colocated vLLM. RL runs dominate; their wall-clock is
sensitive to `max_completion_length` and group size `G` — extrapolate from the smoke test.

| Phase | Hardware | Est. time | Est. units |
|---|---|---|---|
| Pipeline shakedown (Llama-3.2-1B, tiny data) | L4 | 1–2 h | ~8 |
| Stage 1 forward RFT (3B) | A100 | ~1.5 h | ~18 |
| Stage 2a **composition control**: RL (level-2, ~250 steps) | A100 | 5–8 h | ~70–95 |
| Stage 2a **composition control**: RFT | A100 | 2–3 h | ~30 |
| Stage 1.5 inverse baseline eval | A100 | ~0.5 h | ~6 |
| Stage 2b **inversion**: RL (level-1, ~250 steps) | A100 | 5–8 h | ~70–95 |
| Stage 2b **inversion**: RFT | A100 | 2–3 h | ~30 |
| Final evals (levels 1–4 × splits × ops, pass@k) | A100/L4 | 2–3 h | ~25–35 |
| **Total (MVP-minimal)** | | | **~260–320** |
| *Optional:* inversion RL on level-1+2 mix | A100 | 5–8 h | +70–95 |

**Cost knobs:** group size `G` (start 8), train-prompt count, `max_completion_length` (start 512),
eval `k` (start 8, raise to 16 only for the final), number of train-level conditions (defer the
inversion level-1+2 mix first). Do all logic/unit tests on CPU.

---

## 12. Success criteria

- **Minimum viable result:** clean **G1** (forward learned), **G0** (composition control: RL>RFT
  depth generalization — setup validated), **G2** (inverse not already there), plus a well-measured
  RL-vs-RFT depth-generalization curve for inversion. Publishable / advisor-convincing regardless of
  sign, *because the control makes it interpretable*.
- **Strong result (H2+H3):** inversion RL generalizes to deeper levels **and** transfers to held-out
  skills with a small gap, CoT shows difficulty-scaled search ⇒ “RL installs the inversion/μ-search
  **operation**,” contrasted against the composition control.
- **Equally interesting asymmetry:** composition control holds (H0) but inversion RL≈RFT or fails to
  transfer ⇒ RL teaches composition but not inversion — a clean, novel asymmetry.

---

### Provenance
Reuses task surface, two-stage protocol, level ladder, and the `compute_score` /
`compute_score_backward` reward semantics from **PRIME-RL/RL-Compositionality** (Yuan et al.,
arXiv:2509.25123, Apache-2.0). Skill functions in `skills_inverse.py` adapt that repo’s
`examples/data_preprocess/string_data.py` and add new invertible operations. Cite the original paper
in any write-up.
