#!/usr/bin/env python3
"""Coverage probe — EXECUTION_GUIDE Step 6; produces the Gate-G3 decision inputs (plan §6–7).

Samples ``--k`` rollouts at ``--temp`` from a merged Stage-1 checkpoint over
``--n`` problems drawn from the regenerated ``inv_l1_seen_train.jsonl``,
scores each rollout with ``verifier.inverse_reward`` (functional preimage
match), and writes one CSV row per problem: skill, success_count out of k,
mean and p95 completion tokens. At the end it prints the per-skill summary
that feeds the pre-registered pool decision rule (plan §6): pass@k and the
trainable fraction (problems with 0 < successes < k).

Stale-data guards (hard failures before any generation — a stale v3 file on
Drive already bit us once):
  (a) ``DATA_CONTRACT.txt`` next to ``--data`` must read exactly
      ``v4-heldout-duplicate``;
  (b) the skill census of ``--data`` must equal the v4 SEEN set from
      ``inverse_tasks``, and every problem must be a level-1 inverse problem.

Resumable: if ``--out`` exists, generation is skipped entirely and the
summary is reprinted from it (repo skip-guard convention — delete or archive
the CSV to force a re-run). Partial progress is checkpointed to
``<out-stem>.partial.csv`` every ``--checkpoint-every`` problems and picked
up on restart; the partial file is promoted to ``--out`` on completion.

GPU use is confined to ``build_vllm_generate_fn`` (vLLM is imported lazily,
on the first generation call), so scoring and aggregation run — and unit-test
— on CPU with a stubbed generator. Expected generator signature::

    generate_fn(prompts: list[str], k: int) -> list[list[tuple[str, int]]]

returning, per prompt, k ``(completion_text, completion_tokens)`` pairs.

Summary token columns: ``mean_tokens`` is the exact pooled mean (every
problem contributes k completions); ``p95_tokens`` is the p95 over the
per-problem p95 values — a conservative tail estimate reproducible from the
CSV alone (the CSV is the artifact of record).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inverse_tasks import SEEN
from verifier import chain_skills, inverse_reward

EXPECTED_DATA_CONTRACT = "v4-heldout-duplicate"
FIELDNAMES = ["problem_index", "skill", "success_count", "k", "mean_tokens", "p95_tokens"]
TRAINABLE_THRESHOLD = 0.2  # plan §6: skills with trainable fraction >= 0.2
TRAINABLE_SKILLS_REQUIRED = 4  # plan §6: >= 4 of 6 SEEN skills -> inverse-only pool

GenerateFn = Callable[[Sequence[str], int], list[list[tuple[str, int]]]]


class ProbeError(SystemExit):
    """Hard failure with a message; exits non-zero from the CLI."""

    def __init__(self, message: str) -> None:
        super().__init__(f"coverage_probe: {message}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_data_contract(data_path: Path) -> None:
    """Guard (a): DATA_CONTRACT.txt next to the data file must match exactly."""
    contract_path = Path(data_path).parent / "DATA_CONTRACT.txt"
    if not contract_path.exists():
        raise ProbeError(
            f"no DATA_CONTRACT.txt next to {data_path} — refusing to probe unverified data "
            f"(expected contract {EXPECTED_DATA_CONTRACT!r}; re-run notebook Cell 2)"
        )
    found = contract_path.read_text(encoding="utf-8").strip()
    if found != EXPECTED_DATA_CONTRACT:
        raise ProbeError(
            f"stale data contract {found!r} in {contract_path} "
            f"(expected {EXPECTED_DATA_CONTRACT!r}; re-run notebook Cell 2 to regenerate)"
        )


def check_skill_census(problems: Sequence[dict[str, Any]]) -> None:
    """Guard (b): level-1 inverse problems whose skills are exactly the v4 SEEN set."""
    if not problems:
        raise ProbeError("input JSONL is empty")
    census: set[str] = set()
    for i, problem in enumerate(problems):
        if problem.get("task") != "inverse":
            raise ProbeError(f"problem {i} has task {problem.get('task')!r}; expected 'inverse'")
        skills = chain_skills(problem["chain"])
        if len(skills) != 1:
            raise ProbeError(f"problem {i} is level {len(skills)}; the probe expects level-1 chains")
        census.update(skills)
    expected = set(SEEN)
    if census != expected:
        missing = sorted(expected - census)
        unexpected = sorted(census - expected)
        raise ProbeError(
            "skill census mismatch vs the v4 SEEN set"
            + (f"; missing: {missing}" if missing else "")
            + (f"; unexpected: {unexpected}" if unexpected else "")
            + " — the data file predates the current split; regenerate it"
        )


def select_problems(problems: Sequence[dict[str, Any]], n: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    """Deterministic seeded subset as (original_index, problem) pairs."""
    indexed = list(enumerate(problems))
    if n >= len(indexed):
        return indexed
    selected = random.Random(seed).sample(indexed, n)
    return sorted(selected, key=lambda pair: pair[0])


def p95(values: Sequence[float]) -> float:
    """Nearest-rank 95th percentile (deterministic, no interpolation)."""
    if not values:
        raise ValueError("p95 of empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return float(ordered[rank - 1])


def score_problem(index: int, problem: dict[str, Any], rollouts: Sequence[tuple[str, int]]) -> dict[str, Any]:
    """One CSV row: skill, success count out of k, mean/p95 completion tokens."""
    if not rollouts:
        raise ValueError(f"problem {index}: no rollouts to score")
    successes = sum(int(inverse_reward(text, problem) == 1.0) for text, _tokens in rollouts)
    token_counts = [float(tokens) for _text, tokens in rollouts]
    return {
        "problem_index": index,
        "skill": chain_skills(problem["chain"])[0],
        "success_count": successes,
        "k": len(rollouts),
        "mean_tokens": sum(token_counts) / len(token_counts),
        "p95_tokens": p95(token_counts),
    }


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Atomic CSV write (temp file + rename) so an interrupt never truncates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["problem_index"]):
            writer.writerow(row)
    os.replace(tmp, path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = []
        for raw in csv.DictReader(f):
            rows.append(
                {
                    "problem_index": int(raw["problem_index"]),
                    "skill": raw["skill"],
                    "success_count": int(raw["success_count"]),
                    "k": int(raw["k"]),
                    "mean_tokens": float(raw["mean_tokens"]),
                    "p95_tokens": float(raw["p95_tokens"]),
                }
            )
        return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-skill Gate-G3 inputs: pass@k, trainable fraction, token stats."""
    by_skill: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_skill.setdefault(row["skill"], []).append(row)
    summary: dict[str, dict[str, float]] = {}
    for skill in sorted(by_skill):
        skill_rows = by_skill[skill]
        n = len(skill_rows)
        summary[skill] = {
            "n": n,
            "pass_at_k": sum(1 for r in skill_rows if r["success_count"] > 0) / n,
            "trainable_fraction": sum(1 for r in skill_rows if 0 < r["success_count"] < r["k"]) / n,
            "mean_tokens": sum(r["mean_tokens"] for r in skill_rows) / n,
            "p95_tokens": p95([r["p95_tokens"] for r in skill_rows]),
        }
    return summary


def format_summary(summary: dict[str, dict[str, float]], k: int) -> str:
    lines = [
        f"per-skill coverage summary (k={k}, Gate G3 decision inputs):",
        f"{'skill':<22} {'n':>5} {'pass@' + str(k):>9} {'trainable':>10} {'mean_tok':>9} {'p95_tok':>8}",
    ]
    for skill, stats in summary.items():
        lines.append(
            f"{skill:<22} {stats['n']:>5.0f} {stats['pass_at_k']:>9.3f} "
            f"{stats['trainable_fraction']:>10.3f} {stats['mean_tokens']:>9.1f} {stats['p95_tokens']:>8.1f}"
        )
    trainable_skills = sorted(
        skill for skill, stats in summary.items() if stats["trainable_fraction"] >= TRAINABLE_THRESHOLD
    )
    pool = (
        "inverse-only pool"
        if len(trainable_skills) >= TRAINABLE_SKILLS_REQUIRED
        else "50/50 inverse-L1-SEEN + forward-L2-SEEN mix"
    )
    lines.append(
        f"G3 rule (plan §6): {len(trainable_skills)}/{len(SEEN)} SEEN skills with trainable "
        f"fraction >= {TRAINABLE_THRESHOLD} ({', '.join(trainable_skills) or 'none'}) -> {pool}"
    )
    lines.append("Record the decision in results/G3_DECISION.md (EXECUTION_GUIDE Step 7).")
    return "\n".join(lines)


def run_probe(
    selected: Sequence[tuple[int, dict[str, Any]]],
    generate_fn: GenerateFn,
    k: int,
    out_path: Path,
    checkpoint_every: int = 100,
) -> list[dict[str, Any]]:
    """Score all selected problems, checkpointing partial progress to disk.

    Skip-guard: an existing ``out_path`` short-circuits everything (delete or
    archive it to re-run). A ``<stem>.partial.csv`` from an interrupted run is
    loaded and only the remaining problems are generated.
    """
    out_path = Path(out_path)
    if out_path.exists():
        print(f"[skip] {out_path} exists; loading it (delete/archive to re-run the probe)")
        return load_rows(out_path)

    partial_path = out_path.with_name(out_path.stem + ".partial.csv")
    rows: list[dict[str, Any]] = []
    if partial_path.exists():
        rows = load_rows(partial_path)
        selected_indices = {index for index, _problem in selected}
        for row in rows:
            if row["problem_index"] not in selected_indices or row["k"] != k:
                raise ProbeError(
                    f"{partial_path} does not match the current --n/--k/--seed selection; "
                    f"delete it to restart the probe"
                )
        print(f"[run] resuming from {partial_path}: {len(rows)}/{len(selected)} problems done")

    done = {row["problem_index"] for row in rows}
    pending = [(index, problem) for index, problem in selected if index not in done]

    for start in range(0, len(pending), checkpoint_every):
        chunk = pending[start : start + checkpoint_every]
        completions = generate_fn([problem["prompt"] for _index, problem in chunk], k)
        if len(completions) != len(chunk):
            raise ProbeError(f"generator returned {len(completions)} results for {len(chunk)} prompts")
        for (index, problem), rollouts in zip(chunk, completions):
            rows.append(score_problem(index, problem, rollouts))
        write_rows(partial_path, rows)
        print(f"[run] scored {len(rows)}/{len(selected)} problems -> {partial_path}")

    write_rows(out_path, rows)
    partial_path.unlink(missing_ok=True)
    print(f"[run] wrote {out_path} ({len(rows)} rows)")
    return rows


def build_vllm_generate_fn(ckpt: Path, temp: float, gpu_mem_util: float, seed: int) -> GenerateFn:
    """Lazy standalone vLLM loader; the engine is built on the first call.

    The checkpoint must be a merged model directory (notebook Cell 3
    ``load_model`` merges the Stage-1 LoRA adapter), not a bare PEFT adapter.
    Lengths match the Stage-2 RL config: max prompt 2048 / completion 512.
    """
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise ProbeError(f"checkpoint not found: {ckpt}")
    if (ckpt / "adapter_config.json").exists():
        raise ProbeError(
            f"{ckpt} is a LoRA adapter, not a merged model; merge it first "
            f"(notebook Cell 3: load_model(CFG['model_name'], adapter)['model_path'])"
        )

    state: dict[str, Any] = {}

    def generate_fn(prompts: Sequence[str], k: int) -> list[list[tuple[str, int]]]:
        if "llm" not in state:
            from vllm import LLM, SamplingParams  # GPU-only import, deliberately lazy

            print(f"[run] loading vLLM model {ckpt} (gpu_memory_utilization={gpu_mem_util})")
            state["llm"] = LLM(
                model=str(ckpt),
                tokenizer=str(ckpt),
                trust_remote_code=True,
                dtype="bfloat16",
                max_model_len=2048 + 512,
                gpu_memory_utilization=gpu_mem_util,
                enforce_eager=True,
            )
            # top_p=1.0 on purpose: the probe must match the GRPO rollout
            # distribution (pure temperature sampling), not Cell 3's eval recipe.
            state["params"] = SamplingParams(n=k, temperature=temp, top_p=1.0, max_tokens=512, seed=seed)
        outputs = state["llm"].generate(list(prompts), state["params"])
        return [[(choice.text, len(choice.token_ids)) for choice in out.outputs] for out in outputs]

    return generate_fn


def main(argv: Sequence[str] | None = None, generate_fn: GenerateFn | None = None) -> int:
    parser = argparse.ArgumentParser(description="Coverage probe over inverse level-1 SEEN problems (Gate G3).")
    parser.add_argument("--ckpt", type=Path, required=generate_fn is None, help="merged Stage-1 checkpoint dir")
    parser.add_argument("--data", type=Path, required=True, help="inv_l1_seen_train.jsonl (v4 contract)")
    parser.add_argument("--out", type=Path, default=Path("results/coverage_probe.csv"))
    parser.add_argument("--n", type=int, default=500, help="problems to probe (seeded subset)")
    parser.add_argument("--k", type=int, default=16, help="rollouts per problem")
    parser.add_argument("--temp", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--gpu-mem-util", type=float, default=0.85, help="vLLM gpu_memory_utilization (0.85 for L4)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100, help="problems per partial-CSV checkpoint")
    args = parser.parse_args(argv)
    if args.n <= 0 or args.k <= 0 or args.checkpoint_every <= 0:
        raise ProbeError("--n, --k, and --checkpoint-every must be positive")

    check_data_contract(args.data)
    problems = read_jsonl(args.data)
    check_skill_census(problems)
    selected = select_problems(problems, args.n, args.seed)
    print(f"[run] probing {len(selected)}/{len(problems)} problems, k={args.k}, temp={args.temp}, seed={args.seed}")

    if generate_fn is None and not args.out.exists():
        generate_fn = build_vllm_generate_fn(args.ckpt, args.temp, args.gpu_mem_util, args.seed)
    rows = run_probe(selected, generate_fn, args.k, args.out, args.checkpoint_every)
    print(format_summary(summarize(rows), args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
