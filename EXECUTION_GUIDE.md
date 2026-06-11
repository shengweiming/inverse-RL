# Execution Guide — Inverse-RL Experiment (GitHub + coding agent + Colab)

Operational build-and-run order. The science spec is `INVERSE_EXPERIMENT_PLAN.md`
(canonical on any conflict); agent rules are in `CLAUDE.md`. Steps are tagged
**[YOU]** (Wei, by hand or in Colab) or **[AGENT]** (delegate to a coding agent).

## Workflow model (read first)

- All code lives in this repo; the single Colab notebook
  `notebooks/inverse_rl_colab.ipynb` clones it (Cell 0 — **check `BRANCH`
  points at your current branch before every session**).
- Everything is resumable: datasets, checkpoints, and results live on Drive
  under `/content/drive/MyDrive/inverse-rl/{data,ckpts,results}` plus
  `state.json`. Cells skip work whose artifacts already exist.
- **Skip-guard rule:** results CSVs gate by *existence*. If you change an eval
  set or a split, delete (or rename to archive) the corresponding CSV, or the
  cell will silently serve stale numbers. This already bit us once (see Step 4
  note).
- **Data contract:** Cell 2 writes `data/DATA_CONTRACT.txt`. Any change to
  generation or the seen/held-out split must bump `DATA_CONTRACT` in Cell 2,
  which forces one regeneration of every JSONL.
- GPU policy: **A100-80GB** for anything that trains (Cells 4, 6, 7);
  **L4 is fine** for eval and the coverage probe — but set
  `CFG["eval_gpu_mem_util"] = 0.85` on L4 (the 0.30 default is sized for
  sharing an A100 with a just-trained model).

## Environment gotchas (hard-won; do not rediscover)

1. **transformers pin.** Colab + uv-installed vLLM + pip requirements can leave
   a Franken-install (metadata says 4.56.2, files from another version;
   symptom: `cannot import name 'GenerationMixin'`). Fix:
   `pip uninstall -y transformers && pip install --no-cache-dir transformers==4.56.2`,
   then **restart runtime**. The pin now lives in `requirements.txt`; keep it.
2. **Drive is slow.** vLLM loading a checkpoint through Drive FUSE takes
   minutes and the 2-shard progress bar looks frozen at 0%. For repeated loads,
   `shutil.copytree` the checkpoint to `/content/` first.
3. **Interrupted vLLM loads leak VRAM.** If you interrupt mid-`LLM(...)`, the
   half-built engine is unreachable and ~the whole model's memory is orphaned
   (IPython tracebacks pin the frames). Symptom: OOM at KV-cache allocation
   with far more memory in use than one model. Fix: restart runtime; do not
   fight it in-session.
4. `release_llm()` (Cell 3) frees the engine *and* nulls `_ACTIVE_MODEL`; vLLM
   may still not return all VRAM in-process — restart before training cells.

## Step 0 — [YOU] Session preamble (every Colab session)

Run Cells 0–3. Cell 0: repo clone + installs (verify `BRANCH`); Cell 1: Drive +
`CFG` + state; Cell 2: data generation (no-op if contract matches); Cell 3: eval
utilities. On L4 set `eval_gpu_mem_util` as above.

## Steps 1–4 — DONE (history)

- **Step 1–2** ✅ Core modules + data generation under the paper-surface
  contract (9 skills, per-element params, `func_N` ids, hidden-def rendering,
  x-form only). 57 unit tests green; generator reject rate is 0 by construction.
- **Step 3** ✅ Notebook skeleton with Drive resume.
- **Step 4** ✅ Stage 1 SFT on the paper's published RFT corpus → **G1 PASS
  96.8%** (`ckpts/stage1_sft/`). Stage 1.5 inverse baseline → **G2 PASS**
  (structural 0.132; 0.045 excl reverse_words). ⚠️ Lesson: the first summary
  CSV was stale-loaded from an old file with the same name — per the skip-guard
  rule above, the per-skill **detail** CSV is the artifact of record.

## Step 5 — DONE (history) — Held-out split swap (no GPU)

Landed in two refinements; the code/tests/notebook are on the final v4 split
(`DATA_CONTRACT v4-heldout-duplicate`). Rationale and baselines: plan §3.
- **v3** promoted `mirror_str` to held out and returned `reverse_words` to SEEN
  (it is the identity on every reachable string — see plan §3).
- **v4** (current) swaps `fancy_brackets ↔ duplicate_every_char`:
  `fancy_brackets`'s payload is visually present (baseline 0.171, a weak probe),
  so it moves to SEEN and the clean-structural `duplicate_every_char`
  (baseline 0.000) is promoted, making the held-out cell uniformly the three
  hardest structural skills.

Final state, for reference (touchpoints — all four, or things bite):
1. `inverse_tasks.py`: `HELD_OUT = ["rotate_str", "mirror_str", "duplicate_every_char"]`
   (one line; `SEEN` derives itself).
2. `tests/test_tasks.py` → `test_held_out_seen_partition`: `SEEN` order is
   repeat_str, reverse_words, add_prefix, add_suffix, insert_separator,
   fancy_brackets. `pytest -q` green (72 tests).
3. Notebook Cell 2: `DATA_CONTRACT = "v4-heldout-duplicate"`.
4. Drive: archive `results/stage1p5_inverse_skill_detail.csv` (rename, e.g.
   `..._v2split.csv` — it's a valid pre-swap baseline) and delete
   `results/stage1p5_inverse.csv` (stale anyway).

Also mine the archived detail CSV (free coverage preview): per-skill pass@8 and
trainable fraction (problems with 0 < successes < 8). Then re-run Cell 5 on the
regenerated eval set (one vLLM pass; L4 OK) for the post-swap G2 record.

## Step 6 — [AGENT] Coverage probe (built ✅) → [YOU] run (~1 L4-hour)

Built as `scripts/coverage_probe.py` — a standalone vLLM loader, NOT a
notebook cell (keeps the CLAUDE.md cell-index map valid; as a subprocess it
also returns all VRAM on exit, sidestepping gotchas 3–4). It samples **16
rollouts at temperature 1.0** (top_p 1.0 — the GRPO rollout distribution, not
Cell-3's eval recipe) over 500 problems from the regenerated
`inv_l1_seen_train.jsonl`, scores with `verifier.inverse_reward`, writes
`results/coverage_probe.csv` (one row per problem: skill, success_count/16,
mean & p95 completion tokens) and prints the per-skill G3 decision inputs:
pass@16 and **trainable fraction** (0 < c < 16). Stale-data guards hard-fail
before any generation: `DATA_CONTRACT.txt` next to the data file must read
`v4-heldout-duplicate`, and the file's skill census must be exactly the v4
SEEN set. Resumable: skips entirely if the CSV exists (delete/archive to
re-run); checkpoints partial progress to `coverage_probe.partial.csv` every
100 problems and resumes from it after a session death.

Run after Cells 0–3 on L4. The script needs the **merged** Stage-1 checkpoint
(it refuses a bare LoRA adapter dir); Cell-3 `load_model` produces and caches
the merge:

```python
ckpt = load_model(CFG["model_name"], CKPT_DIR / "stage1_sft")["model_path"]
!python scripts/coverage_probe.py --ckpt {ckpt} --data {DATA_DIR}/inv_l1_seen_train.jsonl --out {RESULTS_DIR}/coverage_probe.csv --gpu-mem-util 0.85
```

(Per gotcha 2, copying the merged checkpoint to `/content/` first makes the
vLLM load much faster.) Acceptance: runs on L4; prints the G3 decision inputs.

## Step 7 — [YOU] Gate G3 decision (minutes)

Apply the pre-registered rule (plan §6): ≥4/6 SEEN skills with trainable
fraction ≥ 0.2 → inverse-only pool; else 50/50 inverse-L1-SEEN +
forward-L2-SEEN. Fix budget **B** (default ~400 steps × 16 prompts × 8
rollouts). Record the decision in `results/G3_DECISION.md` (one paragraph).

## Step 8 — [AGENT] Cell 7: RL arm (build) → [YOU] run (A100, ~6–10 h)

TRL `GRPOTrainer` from `ckpts/stage1_sft` + LoRA r=32. Spec:
- G=8 rollouts/prompt, temp 1.0, `beta`(KL)=0, max prompt 2048 / completion 512,
  batch 16 prompts/step, lr ~1e-6–5e-6 (sweep not required; pick one, log it).
- Reward: `batch_inverse_reward`; if G3 chose the mixed pool, a dispatching
  wrapper keyed on the problem's `task` field (forward → `batch_forward_reward`).
  Chains travel as **JSON-string columns** (Arrow-safe; verifier normalizes).
- Log per step: mean reward, zero-variance-group fraction, completion length,
  per-skill reward composition → W&B.
- Eval hook every ~25 steps: inverse pass@1 on a fixed 60-problem held-out-L1
  slice (greedy), appended to `results/arm_a_evalcurve.csv`.
- Checkpoint adapter to Drive every 50 steps; resume from latest on restart.
- Smoke config: Llama-3.2-1B, 30 steps, 64 problems — must run end-to-end on L4
  before the A100 session. Gate **G4** criteria in plan §9.

## Step 9 — [AGENT] Cell 6: strengthened iterative RFT arm (build) → [YOU] run

Loop until budget B matches Arm A: generate k=8/problem at temp 1.0 (vLLM) →
`inverse_reward` filter → **cap 2 kept per problem** → drop all-correct
problems → SFT 1 epoch (lr 2e-5, reuse Cell-4 SFT machinery) → eval (same
60-problem hook + full level-1 eval) → iterate. Each iteration's data, adapter,
and eval land on Drive under `ckpts/rft_iter{i}/` and `results/arm_b_*.csv`.
Release vLLM before each SFT phase (`release_llm()`); expect a runtime restart
between generation and training if VRAM is not returned.

## Step 10 — [YOU + AGENT] Analysis

- [AGENT] `scripts/analysis.py` or Cell 8: headline table (held-out inverse
  pass@1/@8: base vs Arm A vs Arm B at matched B), pass@k divergence sweep
  (k ∈ {1..64}) for Arm A vs base, training curves, per-skill breakdowns,
  H3 proxy rate (candidate-echo + forward-simulation regex) per checkpoint.
- [YOU] Read 20–30 CoTs per checkpoint on held-out problems for the
  propose-and-verify signature; your eyes are the instrument, the regex is the
  proxy. Arm C (forward-RL control) only if budget remains after G5.

## Artifact inventory (Drive)

- `data/` — `DATA_CONTRACT.txt`, `fwd_l1to4_eval.jsonl`,
  `fwd_l2_seen_comp_train.jsonl`, `inv_l1_seen_train.jsonl`,
  `inv_l1to4_eval.jsonl`, `inv_l1to2_seen_optional.jsonl`
  (legacy `fwd_l1_all.jsonl` may linger; unused).
- `ckpts/` — `stage1_sft/` (G1 model), `arm_a_step*/`, `rft_iter*/`.
- `results/` — `stage1_*.csv`, `stage1p5_*.csv` (+ `_v2split` archive),
  `coverage_probe.csv`, `G3_DECISION.md`, `arm_a_*.csv`, `arm_b_*.csv`.
- `state.json` — cell-level done flags (informational; CSVs are the real guards).

## Quick order of operations (TL;DR)

Swap split (Step 5) → probe (6) → G3 decision (7) → build+smoke both arms
(8–9 builds can proceed in parallel with 6) → run Arm A → run Arm B →
analysis (10) → Arm C if budget allows.
