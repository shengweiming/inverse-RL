# CLAUDE.md

Agent guide for the inverse-RL repo. Read `INVERSE_EXPERIMENT_PLAN.md` (science
spec — canonical on any conflict) and `EXECUTION_GUIDE.md` (build order) before
writing code. If a pasted task contradicts the plan, flag it; don't guess.

## What this is

A scaled-down variant of RL-Compositionality (arXiv:2509.25123) on
Llama-3.2-3B-Instruct, testing whether GRPO can teach **inversion** of
pretrained string skills as a transferable operation. Stage 1 (done): SFT on the
paper's published RFT corpus → 96.8% forward accuracy. Stage 1.5 (done):
reversal-curse baseline confirmed (structural inverse ≈ 0.045). Stage 2 (now):
RL vs strengthened iterative RFT on inverse problems, evaluated on held-out
skills. Models answer in JSON; nothing ever executes model output.

## Commands

```bash
pip install -r requirements-dev.txt --break-system-packages  # CPU dev deps
python3 -m pytest -q                  # 72 tests, all green, <1s, no GPU
python3 skills_inverse.py             # registry self-test, must print ALL GOOD
python3 inverse_tasks.py --task inverse --levels 1,2 --n 40 \
    --pool seen --out /tmp/x.jsonl --seed 0   # CLI smoke; expect "rejects: 0"
```

There is no GPU in the dev environment. GPU work happens only in
`notebooks/inverse_rl_colab.ipynb` on Colab (A100 to train, L4 to eval).

## Architecture (four modules + one notebook)

- **`skills_inverse.py`** — the 9 skills. Registry:
  `SKILLS[name] = (forward, inverse, input_sampler, param_sampler, paper_func_id)`.
  Forward bodies are **verbatim paper code — never edit them**. `coerce_param`
  is the single choke point for parameter normalization (numpy ints,
  digit-strings, bool rejection); route all param handling through it.
- **`verifier.py`** — trusted execution + rewards. `normalize_chain` maps every
  wire format (JSON strings, bare names, `[name, param]` pairs, numpy rows) to
  canonical `(name, param)` tuples. `inverse_reward` is a **functional preimage
  match**: re-run the chain on the candidate, compare outputs (paper semantics;
  NOT exact input match — reverse_words admits whitespace-variant preimages).
  Both rewards str-cast before comparing. `batch_*_reward` are TRL-shaped.
- **`inverse_tasks.py`** — problem generation. `ID_MAP` uses the **paper's
  func_N numbering** (1,4,5,6,8,9,13,14,15), never enumerate order.
  `HELD_OUT`/`SEEN` define the split. `render_code` emits hidden-definition
  `main_solution` with params inlined as literals: `func_5(func_1(x, 3), 'qz')`.
  All problems are x-form (constant/binary forms have no recoverable preimage).
- **`prompts.py`** — `FORWARD_PROMPT` is **byte-identical to the paper's**
  (Stage-1 train/eval distribution match). `extract_last_json` parses answers.
- **`notebooks/inverse_rl_colab.ipynb`** — 11 cells, reordered for the Stage-2
  flow. Notebook index 0 markdown header, 1 setup (Cell 0; shell magics, the
  only non-pure-Python cell), 2 Drive/CFG/state (Cell 1), 3 data gen (Cell 2),
  4 eval utilities (Cell 3; vLLM lifecycle `release_llm()`), 5 **Load Stage-1
  model** (locates the merged `stage1_sft` on Drive; the lightweight stand-in
  now that SFT is done), 6 Cell 6 composition-control stub, 7 **Cell 7 Arm-A
  GRPO trainer**, 8 Cell 8 analysis stub. The finished, now-demoted Stage-1 SFT
  (Cell 4) and Stage-1.5 baseline (Cell 5) are archived at the END (indices
  9–10), banner-tagged `[DEMOTED]` — kept for re-run, not in the normal flow.

## Invariants — do not break

1. `FORWARD_PROMPT` stays byte-identical to the paper. No whitespace "fixes".
2. `ID_MAP` paper numbering is frozen; it's part of the trained model's world.
3. Forward skill bodies are verbatim paper code.
4. Generator reject rate is exactly 0 (all skills are total injections); the
   round-trip filter is a safety invariant, and a test asserts zero rejects.
   If your change makes it fire, the change is wrong.
5. Any change to generation or the split: bump `DATA_CONTRACT` in notebook
   Cell 2 AND update `tests/test_tasks.py::test_held_out_seen_partition`.
6. Results CSVs are skip-guards by existence. New eval semantics ⇒ the old CSV
   must be deleted/archived, or cells serve stale numbers silently.
7. For TRL/Arrow datasets, store `chain` as a **JSON-string column**
   (`json.dumps(problem["chain"])`); `normalize_chain` handles it. Mixed-type
   nested columns upset Arrow.
8. Never `exec`/`eval` model output. Rewards only ever run our reference
   functions on the known chain.
9. `pytest -q` green and `python3 skills_inverse.py` printing `ALL GOOD` are
   required before any handoff.

## Notebook editing rules

Edit the `.ipynb` JSON programmatically (json.load → mutate `cells[i]["source"]`
as keepends-split lines → dump with `indent=2`, trailing newline). Every code
cell except setup (notebook index 1) must `ast.parse`. Clear `outputs` and
`execution_count` on edited cells. Don't touch untargeted cells. The deck was
reordered, so **identify cells by their header comment, not a fixed index**
(`find(substr)` over `"".join(cell["source"])`). Current layout: index 0 md
header, 1 setup (Cell 0), 2 Cell 1, 3 Cell 2, 4 Cell 3, 5 Load Stage-1 model,
6 Cell 6 stub, 7 Cell 7 Arm-A GRPO, 8 Cell 8 stub, 9 demoted Cell 4 SFT,
10 demoted Cell 5 baseline.

## Known environment traps (Colab)

- transformers must be pinned (`requirements.txt`); mixed uv/pip installs
  corrupt it (`GenerationMixin` ImportError ⇒ clean reinstall + runtime restart).
- Drive FUSE checkpoint loads are slow (minutes; copy to `/content/` for
  repeated loads). Interrupted vLLM constructor = orphaned VRAM = restart.
- `CFG["eval_gpu_mem_util"]`: 0.30 default (A100 sharing), 0.85 on L4.

## Current state & next tasks

Done: Gates G1, G2 (numbers in plan §4–5); split swap complete and reconciled
across code, tests, notebook, and docs — all on `DATA_CONTRACT v4-heldout-duplicate`,
`HELD_OUT = [rotate_str, mirror_str, duplicate_every_char]`,
`SEEN = [repeat_str, reverse_words, add_prefix, add_suffix, insert_separator, fancy_brackets]`
(rationale in plan §3); coverage probe built (`scripts/coverage_probe.py`,
Step 6 — Colab run on L4 pending); G3 decided (inverse-only pool, B~50k,
`results/G3_DECISION.md`). Cell 7 Arm-A GRPO trainer built (Step 8): TRL
GRPOTrainer + fresh LoRA r=32 on the merged Stage-1 base, pinned DAPO recipe
(beta=0, num_iterations=1, entropy=0, scale_rewards=False), inverse-only
`batch_inverse_reward`, oversample-refill dynamic sampling with pre/post-filter
group logging, per-skill W&B + held-out eval hook, resumable adapter
checkpoints. `SMOKE`/`PREFLIGHT_3B` flags gate the L4 pre-flights before A100;
run pending. Agent tasks queued, in order: Cell 6 iterative-RFT arm (Step 9),
analysis (Step 10). Specs for each are in the guide; gate criteria in the plan.
