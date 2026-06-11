#!/usr/bin/env python3
"""Inspect the LONGEST rollouts on a target skill to confirm the G3 length read.

The coverage probe stored only per-problem aggregates, so the raw texts are gone.
This script regenerates a small batch (same model, same temp 1.0 / top_p 1.0 /
512-cap sampler as the probe) over problems of one skill, then prints the longest
completions so you can eyeball whether long == degenerate (spiral/loop) or
long == coherent-but-unfinished reasoning.

Expected finding (G3_DECISION.md): on fancy_brackets the longest runs are
degenerate, justifying keeping max_completion_len = 512. If they are coherent and
truncated mid-reasoning, reopen the cap question.

Colab usage (after Cells 0-3, on the same merged checkpoint the probe used):

    ckpt = load_model(CFG["model_name"], CKPT_DIR / "stage1_sft")["model_path"]
    !python scripts/inspect_capped.py \
        --ckpt {ckpt} \
        --data {DATA_DIR}/inv_l1_seen_train.jsonl \
        --skill fancy_brackets --n 40 --k 8 --show 10 --gpu-mem-util 0.85

CPU dry-run with a stub (no GPU), prints plumbing only:

    python scripts/inspect_capped.py --data <jsonl> --skill fancy_brackets --selftest
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Sequence

# Score each regenerated rollout with the trusted verifier so you can see whether
# the long ones are also the failing ones (the -0.69 correlation, made concrete).
from verifier import inverse_reward

GenerateFn = Callable[[Sequence[str], int], list[list[tuple[str, int]]]]

CAP = 512  # must match the probe / Stage-2 rollout cap


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select(problems: list[dict[str, Any]], skill: str, n: int, seed: int) -> list[dict[str, Any]]:
    pool = [p for p in problems if p["chain"][0][0] == skill]
    if not pool:
        raise SystemExit(f"no problems for skill {skill!r} in this file "
                         f"(is it in the v4 SEEN set?)")
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def build_vllm_generate_fn(ckpt: Path, gpu_mem_util: float, seed: int) -> GenerateFn:
    ckpt = Path(ckpt)
    if (ckpt / "adapter_config.json").exists():
        raise SystemExit(f"{ckpt} is a LoRA adapter; pass the MERGED model path "
                         f"(load_model(CFG['model_name'], adapter)['model_path']).")
    state: dict[str, Any] = {}

    def generate_fn(prompts: Sequence[str], k: int) -> list[list[tuple[str, int]]]:
        if "llm" not in state:
            from vllm import LLM, SamplingParams  # GPU-only, lazy
            print(f"[run] loading vLLM {ckpt} (gpu_memory_utilization={gpu_mem_util})")
            state["llm"] = LLM(
                model=str(ckpt), tokenizer=str(ckpt), trust_remote_code=True,
                dtype="bfloat16", max_model_len=2048 + CAP,
                gpu_memory_utilization=gpu_mem_util, enforce_eager=True,
            )
            state["params"] = SamplingParams(
                n=k, temperature=1.0, top_p=1.0, max_tokens=CAP, seed=seed,
            )
        outputs = state["llm"].generate(list(prompts), state["params"])
        return [[(c.text, len(c.token_ids)) for c in out.outputs] for out in outputs]

    return generate_fn


def main(argv: Sequence[str] | None = None, generate_fn: GenerateFn | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect longest rollouts on one skill (G3 length check).")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--skill", default="fancy_brackets")
    ap.add_argument("--n", type=int, default=40, help="problems to sample")
    ap.add_argument("--k", type=int, default=8, help="rollouts per problem")
    ap.add_argument("--show", type=int, default=10, help="longest rollouts to print")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true", help="CPU stub, no GPU")
    args = ap.parse_args(argv)

    problems = select(read_jsonl(args.data), args.skill, args.n, args.seed)
    print(f"[run] {len(problems)} {args.skill} problems, k={args.k}, cap={CAP}")

    if args.selftest and generate_fn is None:
        def generate_fn(prompts, k):  # noqa: E306
            out = []
            for i, _ in enumerate(prompts):
                out.append([(f"stub reasoning {j} {{\"input\": \"x\"}}", 100 + 50 * j) for j in range(k)])
            return out

    if generate_fn is None:
        if args.ckpt is None:
            raise SystemExit("--ckpt is required unless --selftest or an injected generate_fn is used")
        generate_fn = build_vllm_generate_fn(args.ckpt, args.gpu_mem_util, args.seed)

    prompts = [p["prompt"] for p in problems]
    completions = generate_fn(prompts, args.k)

    # Flatten to (tokens, truncated?, correct?, problem, text), then sort by length.
    records = []
    for problem, samples in zip(problems, completions):
        for text, ntok in samples:
            correct = inverse_reward(text, problem)
            records.append((ntok, ntok >= CAP, correct, problem, text))
    records.sort(key=lambda r: r[0], reverse=True)

    n_trunc = sum(r[1] for r in records)
    n_corr = sum(r[2] for r in records)
    print(f"[run] {len(records)} rollouts; truncated(>=cap): {n_trunc} "
          f"({n_trunc/len(records):.0%}); correct: {n_corr} ({n_corr/len(records):.0%})")
    long_corr = sum(1 for r in records[: args.show] if r[2] == 1.0)
    print(f"[run] of the {args.show} LONGEST rollouts, correct: {long_corr}/{args.show} "
          f"(expect ~0 if long==lost)\n")

    for rank, (ntok, trunc, correct, problem, text) in enumerate(records[: args.show], 1):
        tag = "TRUNCATED" if trunc else "complete"
        verdict = "CORRECT" if correct == 1.0 else "wrong"
        print("=" * 78)
        print(f"[{rank}] tokens={ntok} {tag} | {verdict} | output={problem['output']!r}")
        print(f"    code: {problem['code'].splitlines()[-1].strip()}")
        print("-" * 78)
        print(text.strip()[:2000])
        print()
    print("Judge: degenerate (repetition/looping/no JSON) => keep cap 512. "
          "Coherent reasoning cut off mid-thought => reopen cap question.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
