# Execution Guide — Inverse-RL Experiment (GitHub + Codex + Colab)

Human-facing, do-this-then-that guide implementing `INVERSE_EXPERIMENT_PLAN.md` as **one resumable
Colab notebook** backed by **Google Drive checkpoints** and **wandb**. Steps are **[YOU]** or
**[CODEX]** (copy the block verbatim to the coding agent). Each build step ends with a **[CHECK]**.

**Defaults:** model `meta-llama/Llama-3.2-3B-Instruct`; shakedown on `meta-llama/Llama-3.2-1B-Instruct`.
RL = TRL `GRPOTrainer` with LoRA + colocated vLLM, single A100-40GB. One notebook, sequential,
independently-resumable cells, each guarded by a Drive-artifact check.

> **Why these choices** (one line, in case the agent asks): 8B GRPO + vLLM won’t fit Colab’s 40GB;
> 1B is below the RLVR search-emergence floor; Llama-3.2-3B is the only Llama point that fits 40GB and
> clears the capability floor, and it avoids the Qwen “elicit not teach” confound.

---

## Workflow model (read first)

- **GitHub repo** = source of truth for code. Codex works here.
- **Colab notebook** clones the repo, installs, mounts Drive, runs the experiment. Only place GPUs
  are used.
- **Google Drive** holds everything expensive to recompute (data, checkpoints, results, a tiny
  `state.json`). **Every cell first checks Drive for its output and skips/loads if present** — the
  crash-resume mechanism.
- **wandb** logs the RL runs.

Drive layout the notebook creates:
```
/content/drive/MyDrive/inverse-rl/
├── data/     fwd_l1/ , fwd_l1to4/ , inv_l1/ , inv_l1to4_eval/ , (opt) inv_l1to2/   *.jsonl
├── ckpts/    stage1_rft/ , comp_rl/ , comp_rft/ , inv_rl/ , inv_rft/   (+ checkpoint-* subdirs)
├── results/  *.json , *.csv , *.png , SUMMARY.md
└── state.json
```

---

## Step 0 — [YOU] Accounts, repo, secrets

1. Create a **GitHub repo** `inverse-rl` (private fine). Add `skills_inverse.py` (provided),
   `INVERSE_EXPERIMENT_PLAN.md`, and this guide to the root. Commit.
2. **Llama license:** Llama models are gated on Hugging Face — visit the
   `meta-llama/Llama-3.2-3B-Instruct` and `…-1B-Instruct` pages and **accept the license** with the
   same HF account whose token you’ll use, or the download 401s.
3. Get a **HF token** (read) and a **wandb API key**.
4. In Colab: **Runtime → Change runtime type → A100** for real runs (use **L4** for Steps 1–3 dev).
5. Put secrets in Colab **Secrets** (🔑): `HF_TOKEN`, `WANDB_API_KEY`. Don’t hardcode.

**[CHECK]** Clone your repo locally; confirm `python skills_inverse.py` prints `Total skills: 25` and
`ALL GOOD`.

---

## Step 1 — [CODEX] Prompts + local verifier (forward AND inverse)

> **Paste to Codex:**
>
> In repo `inverse-rl`, create two files.
>
> **`prompts.py`** — define exactly:
> - `FORWARD_PROMPT`: given `{code}`, predict `main_solution("{input}")`, answer JSON
>   `{"output": ...}`, reason without running code.
> - `INVERSE_PROMPT`: given `{code}` and that `main_solution(x)` returned `"{output}"`, find an input
>   `x` reproducing that output exactly, answer JSON `{"input": ...}`, reason without running code.
> - `extract_last_json(text) -> dict | None`: robustly extract the **last** complete JSON object
>   (handle ```` ```json ```` fences and trailing prose).
>
> **`verifier.py`** — local rewards, **no `exec` of model output, no external sandbox**:
> - `reference_apply(chain, x) -> str`: compose forward functions named in `chain` using
>   `skills_inverse.SKILLS` (each entry `(forward, inverse, sampler, kwargs, tier, origin)`), applying
>   `kwargs`. `chain=[f1,...,fk]` means `f_k(...f_1(x))`.
> - `forward_reward(completion_text, problem) -> float`: parse `"output"`; 1.0 iff it equals
>   `problem["output"]` (already = `reference_apply(chain, input)`), else 0.0.
> - `inverse_reward(completion_text, problem) -> float`: parse `"input"` x̂; if missing/not str →
>   0.0; else 1.0 iff `reference_apply(problem["chain"], x̂) == problem["output"]`, else 0.0.
> - `batch_forward_reward` and `batch_inverse_reward` for TRL (map each completion to its problem by
>   index).
> Add `tests/test_verifier.py`: for every skill, a level-1 problem where the true input gets
> forward_reward... (sanity), the true inverse gets inverse_reward 1.0, and a wrong string gets 0.0.
>
> Do not modify `skills_inverse.py`. Plain importable modules (no install).

**[CHECK]** `pytest -q tests/test_verifier.py` passes. Spot check:
`python -c "import verifier,skills_inverse as s,random; random.seed(0); n='vigenere'; f,finv,samp,kw,_,_=s.SKILLS[n]; x=samp(); y=f(x,**kw); print(verifier.inverse_reward('{\"input\": \"%s\"}'%x, {'chain':[n],'output':y}), verifier.forward_reward('{\"output\": \"%s\"}'%y, {'chain':[n],'output':y}))"`
prints `1.0 1.0`.

---

## Step 2 — [CODEX] Data generation (forward + inverse, levels 1–4)

> **Paste to Codex:**
>
> Create **`inverse_tasks.py`** generating JSONL problem sets, reusing the original repo’s
> code-construction style (concatenate used function sources, then
> `def main_solution(x): return <expr>` for the chain).
>
> - `HELD_OUT = ["atbash","shift_digits","mirror_str","swap_pairs","reverse_words",
>   "positional_shift","rail_fence_2","riffle_shuffle"]`; the other 17 are `SEEN`.
> - `make_problem(chain, task) -> dict | None`: sample `x` from the **first** skill’s sampler;
>   compute `y` via `reference_apply`; **verify** `compose_inverse(chain, y) == x` (undo in reverse
>   order using each skill’s inverse+kwargs); if it fails return None (caller resamples). Build the
>   `code` string and render the matching prompt. Return dict with keys: `task` ("forward"|"inverse"),
>   `chain`, `level`, `input`, `output`, `code`, `prompt`, `skills_seen` (all chain skills in SEEN),
>   and the canonical answer string for RFT/SFT targets.
> - Generators:
>   - `gen_forward(n, levels, skills_pool)` — forward problems; used for Stage-1 (`levels=[1]`, all
>     skills), composition-control train (`levels=[2]`, SEEN), and forward eval (`levels=[1,2,3,4]`).
>   - `gen_inverse(n, levels, skills_pool)` — inverse problems; train (`levels=[1]`, SEEN; optional
>     `[1,2]`) and eval (`levels=[1,2,3,4]`, ALL).
>   - `gen_eval(n_per_cell, task)` — levels 1..4 × {seen, held_out} cells, tagged.
> - CLI, deterministic with `--seed`, prints per-level counts + resample reject rate, e.g.:
>   `python inverse_tasks.py --task inverse --levels 1 --n 4000 --pool seen --out data/inv_l1/train.jsonl`
> - `tests/test_tasks.py`: every emitted problem passes the round-trip check and JSON-parses.
>
> Chains length 1..4. Stage-1 forward uses single skills only.

**[CHECK]** `python inverse_tasks.py --task inverse --levels 1 2 --n 40 --pool seen --out /tmp/x.jsonl --seed 0`
then `head -1 /tmp/x.jsonl | python -m json.tool`. `pytest -q tests/test_tasks.py` passes; reject rate
plausible (<~15%).

---

## Step 3 — [CODEX] The single Colab notebook skeleton (Drive + resume)

> **Paste to Codex:**
>
> Create **`notebooks/inverse_rl_colab.ipynb`**: ONE notebook, ordered cells, each guarded so the
> whole thing is **resumable after a Colab crash**. Implement cells 0–3 now; leave clearly-labeled
> empty cells 4–8.
>
> **Cell 0 — setup:** clone the repo (parametrize git URL), `pip install -q -r requirements.txt`,
> import torch, print GPU name. `requirements.txt` pins: `transformers`, `trl>=0.12`, `peft`,
> `datasets`, `accelerate`, `vllm`, `wandb`, `bitsandbytes`, `pandas`, `matplotlib`.
> **Cell 1 — Drive + config:** `drive.mount('/content/drive')`; `ROOT=/content/drive/MyDrive/inverse-rl`
> and subdirs (`data/ ckpts/ results/`), `makedirs(exist_ok=True)`. `CFG` dict: `model_name`
> (default `meta-llama/Llama-3.2-3B-Instruct`), `shakedown_model` (`…-1B-Instruct`), `lora_r=32`,
> `G=8`, `max_prompt_len`, `max_completion_len=512`, `lr=2e-6`, train/eval levels, pass@k `k=8`,
> seeds. Load `HF_TOKEN`, `WANDB_API_KEY` from `google.colab.userdata`; `huggingface_hub.login`.
> Implement `state_load()/state_save(key)` over `ROOT/state.json`, `exists(path)`, `done(key)`.
> **Cell 2 — data:** if Drive files missing, call `inverse_tasks.py` generators to write all needed
> JSONL (`fwd_l1` all-skills, `fwd_l1to4` eval, `fwd_l2_seen` comp-train, `inv_l1_seen` inv-train,
> `inv_l1to4_eval` ALL; optional `inv_l1to2_seen`); else skip. Mark `state["data"]=True`.
> **Cell 3 — eval utilities:** `load_model(path_or_name, adapter=None)` (HF + optional LoRA merge);
> `generate(prompts, n)` via vLLM; `eval_task(model, cells, task, k)` → tidy DataFrame
> (level, split, pass@1, pass@k) using the matching reward; `forward_accuracy(model)` per skill/tier.
> Save DataFrames to `results/`.
>
> Every long cell: check Drive for output → load & skip if present; else compute → save → set state.
> Print a one-line `[skip]`/`[run]` banner.

**[CHECK]** Run cells 0–3 on **L4** with the **1B** model and tiny Step-2 data copied into
`ROOT/data/`. Confirm GPU prints, Drive mounts, `state.json` appears, `eval_task` returns a DataFrame
on the base model (low numbers fine).

---

## Step 4 — [CODEX] Stage 1 forward RFT  →  Stage 1.5 inverse baseline

> **Paste to Codex:**
>
> Fill **Cell 4 (Stage 1 RFT)** and **Cell 5 (Stage 1.5 baseline)**.
>
> **Cell 4 — Stage 1 forward RFT.** If `ckpts/stage1_rft` exists, load & skip. Else: (a) rollout
> `CFG.rollout_k` (e.g. 4) samples per `fwd_l1` problem from the base model via vLLM; keep those with
> `forward_reward==1.0`; (b) build SFT dataset (prompt, correct_completion); (c) LoRA-SFT with TRL
> `SFTTrainer` (rank `CFG.lora_r`, 1–2 epochs, bf16, grad-checkpointing); (d) merge adapter, save
> merged model to `ckpts/stage1_rft` (the Stage-2 base). Log SFT loss to wandb project `inverse-rl`,
> run `stage1-rft`. Compute & save **forward accuracy per skill & tier** to
> `results/stage1_forward.csv`. Print **Gate G1**: PASS if mean ≥0.90 and every tier ≥0.75, else WARN
> with offending skills.
>
> **Cell 5 — Stage 1.5 inverse baseline.** Load `ckpts/stage1_rft`. `eval_task(task="inverse")` on the
> level-1 eval cells (seen+held_out), pass@1 and pass@k. Save `results/stage1p5_inverse.csv`. Print
> **Gate G2**: PASS (good for us) if inverse pass@1 is low, esp. T2–T3; if already high, warn that the
> reversal-curse contrast is weak and suggest restricting to T2–T3.

**[CHECK]** On **A100** with **3B** + real data: G1 should PASS (forward learned); G2 should show
**low** inverse accuracy despite high forward — that gap is the premise. If G1 fails on some T3 skills,
drop them (don’t switch models).

---

## Step 5 — [CODEX] Stage 2a: COMPOSITION CONTROL (positive control + gate)

> **Paste to Codex:**
>
> Build **Cell 6**, a **parametrized** Stage-2 trainer used for *both* operations, then invoke it for
> the composition control. Two functions:
>
> - `run_grpo(task, train_jsonl, train_levels, run_name, out_dir)`: base = merged `ckpts/stage1_rft`;
>   train a **fresh LoRA** with TRL `GRPOTrainer`; reward = `verifier.batch_forward_reward` if
>   `task=="forward"` else `batch_inverse_reward` (pass per-prompt `problem` dicts via dataset columns
>   / closure so the reward can recover each `chain`/`output`). GRPO config: `num_generations=CFG.G`,
>   `max_completion_length=CFG.max_completion_len`, `use_vllm=True`, bf16, grad-checkpointing,
>   `learning_rate=CFG.lr`, default KL/beta, `per_device_train_batch_size` tuned to fit A100-40GB,
>   `max_steps`, `save_steps=25`. **Drive checkpointing:** `output_dir` under `out_dir` on Drive +
>   `resume_from_checkpoint=True` so a Colab restart resumes from the latest Drive checkpoint. wandb:
>   `report_to="wandb"`, log reward mean/std, frac-correct, KL, completion length; add periodic eval
>   logging `eval/{task}_pass@1/{seen,held_out}` per eval level. On finish: merge, save to
>   `out_dir/final`. Idempotent: skip if `final` exists, resume if a checkpoint exists, else fresh.
> - `run_rft(task, train_jsonl, train_levels, run_name, out_dir)`: rollout `CFG.rft_k` samples per
>   train problem from `ckpts/stage1_rft`, keep reward==1.0, LoRA-SFT on (prompt, correct_completion),
>   merge, save. Match #SFT examples to the RL arm’s seen-correct count where feasible (report both).
>
> **Invoke for the composition control:** `run_grpo("forward", "data/fwd_l2_seen", [2],
> "comp-rl", "ckpts/comp_rl")` and `run_rft("forward", "data/fwd_l2_seen", [2], "comp-rft",
> "ckpts/comp_rft")`. Then eval both on forward levels 1–4 (seen+held_out) and save to `results/`.

**[CHECK]** First a **20-step smoke run** (`max_steps=20`, `G=4`): reward nonzero & trending up, wandb
logging, a Drive checkpoint appears; kill the runtime mid-run and re-run → it **resumes** from Drive.
Then the real comp runs (~250 steps). **Gate G0:** comp-RL should beat comp-RFT on levels 3–4. If not,
the setup is underpowered → raise steps or revisit before inversion (an inversion null would otherwise
be uninterpretable).

---

## Step 6 — [CODEX] Stage 2b: INVERSION (the experiment)

> **Paste to Codex:**
>
> Reusing the Cell-6 functions, fill **Cell 7** to run the inversion arms:
> `run_grpo("inverse", "data/inv_l1_seen", [1], "inv-rl", "ckpts/inv_rl")` and
> `run_rft("inverse", "data/inv_l1_seen", [1], "inv-rft", "ckpts/inv_rft")`. (Optional, budget
> permitting: a second RL run on `data/inv_l1to2_seen`, levels `[1,2]`, `ckpts/inv_rl_l1to2`.) Then
> eval all inversion models on inverse levels 1–4 (seen vs held_out), pass@1 + pass@k, save to
> `results/`. Idempotent via Drive as before.

**[CHECK]** Watch wandb: inverse reward should climb; the live H3 readout is
`eval/inverse_pass@1/held_out` rising across levels. Confirm inv-RFT improves **in-distribution**
(sanity) so the depth/transfer comparison is meaningful.

---

## Step 7 — [CODEX] Analysis: operation comparison, transfer, CoT

> **Paste to Codex:**
>
> Fill **Cell 8 (analysis)**. Load comp-RL, comp-RFT, inv-RL, inv-RFT (+ optional inv-RL-l1to2) and
> the Stage-1 baseline. Produce & save to `results/`:
> 1. **Headline figure** (`png`+`csv`): two panels — *composition* and *inversion* — each plotting
>    pass@1 vs eval level (1–4) for RL vs RFT; facet inversion by seen vs held_out. This is the money
>    figure: composition (control) vs inversion (test), RL vs RFT.
> 2. **Transfer-gap table:** held_out minus seen pass@1 per inversion model (small ⇒ operation-level
>    learning ⇒ H3).
> 3. **Forward retention:** forward accuracy of each Stage-2 model vs Stage-1 (forgetting check).
> 4. **CoT diagnostics:** for a fixed problem sample, completion length vs tier/level for RL vs RFT,
>    and a regex tally of search markers (“try/attempt/check/let me verify”); save a few qualitative
>    transcripts per tier.
> Write `results/SUMMARY.md` filling in numbers for G0/G1/G2/H1/H2/H3 and stating which outcome
> obtained.

**[CHECK]** Open the headline figure. Decision read:
- comp-RL>comp-RFT at depth (G0 holds) **and** inv-RL>inv-RFT at depth with inv-RL held_out≈seen ⇒
  **H2+H3 supported** (“RL installs the inversion/μ-search operation”).
- comp-RL>comp-RFT but inv-RL≈inv-RFT (or held_out≪seen) ⇒ **clean asymmetry**: RL teaches
  composition, not inversion. Still novel and publishable.

---

## Checkpointing & resume — what’s saved where

| Artifact | Drive path | Resume behavior |
|---|---|---|
| Generated data | `data/**.jsonl` | regenerated only if missing |
| Stage-1 model | `ckpts/stage1_rft/` (merged) | loaded & skipped if present |
| Stage-2 RL/RFT ckpts | `ckpts/{comp,inv}_{rl,rft}/checkpoint-*` | `resume_from_checkpoint` on restart |
| Stage-2 finals | `ckpts/**/final` | cell skips if present |
| Results/plots | `results/**` | overwritten on rerun of Cell 8 only |
| Phase flags | `state.json` | each cell checks before running |

Models live on Drive, so a disconnect costs only steps since the last `save_steps` (≈25). Keep
`save_steps` small.

---

## wandb logging spec (RL runs)

Project `inverse-rl`; one run per Stage-2 condition (`comp-rl`, `inv-rl`, opt `inv-rl-l1to2`). Log:
`reward/mean`, `reward/std`, `reward/frac_correct`, `objective/kl`, `completions/mean_len`, and
periodic custom `eval/{task}_pass@1/{seen,held_out}` per eval level. The held_out curve is your live
H3 readout during training.

---

## Budget discipline (≤500 units; aim ~260–360)

3B is ~1.5–2× the cost of a 1.5B run, so be disciplined:
- **Shakedown everything on Llama-3.2-1B + L4 + tiny data** before any A100 run.
- A100 only for: Stage-1 RFT, the four Stage-2 runs (comp-RL/RFT, inv-RL/RFT), final evals.
- Start RL at `G=8, max_steps=250, max_completion_len=512`; from the 20-step smoke run, read
  per-step wall-clock and extrapolate before committing. Extend only if reward is still climbing.
- pass@k: `k=8` until the headline figure is stable; `k=16` only for the final.
- Run the four Stage-2 conditions sequentially; Drive checkpoints make a disconnect cheap.
- **First lever if tight:** defer the optional inversion level-1+2 RL run.
- **Second lever:** if 3B underfits forward T3, drop 2–3 T3 skills (don’t switch models).

---

## Quick order of operations (TL;DR)
1. [YOU] repo + **accept Llama license** + secrets + Colab A100/L4.
2. [CODEX] Steps 1–2: `prompts.py`, `verifier.py` (forward+inverse), `inverse_tasks.py` (+tests).
   [CHECK] pytest.
3. [CODEX] Step 3: notebook cells 0–3. [CHECK] L4 + 1B dry run.
4. [CODEX] Step 4: Stage-1 RFT + Stage-1.5 baseline. [CHECK] **G1 pass, G2 low** on 3B/A100.
5. [CODEX] Step 5: composition control (RL vs RFT). [CHECK] smoke + resume, then **Gate G0**.
6. [CODEX] Step 6: inversion (RL vs RFT). [CHECK] inverse reward climbs; RFT sane.
7. [CODEX] Step 7: analysis + headline figure + SUMMARY.md.
