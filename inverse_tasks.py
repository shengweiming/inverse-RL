"""Deterministic JSONL task generation for inverse-RL experiments.

The generated ``chain`` always stores true ``skills_inverse.SKILLS`` names for
trusted verification. The canonical rendered code shown to the model hides atomic
function definitions and includes only ``main_solution`` over meaningless
``func_N`` identifiers. Stage-1 rejection sampling can separately render the
minimal forward definitions needed for a chain.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import skills_inverse
from prompts import FORWARD_PROMPT, INVERSE_PROMPT
from skills_inverse import SKILLS
from verifier import reference_apply

Task = Literal["forward", "inverse"]
Pool = Literal["all", "seen", "held_out"]

# Freeze the decontamination map from the registry's insertion order. Do not
# reorder SKILLS after this point: func_0 ... func_24 are part of the dataset
# contract and only affect rendered code, never the trusted chain field.
ID_MAP: dict[str, str] = {skill_name: f"func_{i}" for i, skill_name in enumerate(SKILLS)}
ID_MAP_INV: dict[str, str] = {identifier: skill_name for skill_name, identifier in ID_MAP.items()}

HELD_OUT: list[str] = [
    "atbash",
    "shift_digits",
    "mirror_str",
    "swap_pairs",
    "reverse_words",
    "positional_shift",
    "rail_fence_2",
    "riffle_shuffle",
]
SEEN: list[str] = [name for name in SKILLS if name not in HELD_OUT]
ALL_SKILLS: list[str] = list(SKILLS)

def _replace_render_names(source: str) -> str:
    """Replace true skill identifiers with stable decontaminated render names."""
    if not ID_MAP:
        return source
    pattern = re.compile(r"\b(" + "|".join(re.escape(name) for name in sorted(ID_MAP, key=len, reverse=True)) + r")\b")
    return pattern.sub(lambda match: ID_MAP[match.group(1)], source)


DEPENDENCY_SOURCE: dict[str, tuple[str, ...]] = {
    "atbash": (inspect.getsource(skills_inverse._atbash_ch).strip(),),
    "vigenere": (inspect.getsource(skills_inverse._vig).strip(),),
    "succ_char": ("_LO,_HI = 32,126; _SPAN=_HI-_LO+1",),
    "deterministic_shuffle": ("from math import gcd", inspect.getsource(skills_inverse._mult).strip()),
}


def _expression(chain: Sequence[str]) -> str:
    expr = "x"
    for name in chain:
        expr = f"{name}({expr})"
    return expr


def render_code(chain: Sequence[str], show_defs: bool = True) -> str:
    """Render a ``main_solution`` snippet for a true-name skill chain.

    Hidden-definition snippets (``show_defs=False``) expose only the composed
    ``main_solution`` expression over decontaminated ``func_N`` identifiers.
    Stage-1 rejection sampling snippets (``show_defs=True``) add just the
    forward definitions and explicit module-level dependencies needed by this
    chain, never the whole helper preamble.
    """
    _validate_chain(chain)
    main = f"def main_solution(x):\n    return {_expression(chain)}"
    if not show_defs:
        return _replace_render_names(main)

    blocks: list[str] = []
    seen_blocks: set[str] = set()
    for name in dict.fromkeys(chain):
        for dependency in DEPENDENCY_SOURCE.get(name, ()):
            if dependency not in seen_blocks:
                blocks.append(dependency)
                seen_blocks.add(dependency)
        forward = inspect.getsource(SKILLS[name][0]).strip()
        if forward not in seen_blocks:
            blocks.append(forward)
            seen_blocks.add(forward)
    blocks.append(main)
    return _replace_render_names("\n\n".join(blocks))


def compose_inverse(chain: Sequence[str], y: str) -> str:
    """Undo a true-name chain in reverse order with trusted inverse functions."""
    _validate_chain(chain)
    value = y
    for name in reversed(chain):
        _forward, inverse, _sampler, kwargs, _tier, _origin = SKILLS[name]
        value = inverse(value, **kwargs)
    return value


def make_problem(chain: Sequence[str], task: Task) -> dict[str, Any] | None:
    """Build one verified problem or return ``None`` if round-trip fails."""
    if task not in {"forward", "inverse"}:
        raise ValueError(f"unknown task: {task!r}")
    _validate_chain(chain)

    first = chain[0]
    sampler = SKILLS[first][2]
    x = sampler()
    y = reference_apply(chain, x)
    if compose_inverse(chain, y) != x:
        return None

    code = render_code(chain)
    prompt = (
        FORWARD_PROMPT.format(code=code, input=x)
        if task == "forward"
        else INVERSE_PROMPT.format(code=code, output=y)
    )
    problem: dict[str, Any] = {
        "task": task,
        "chain": list(chain),
        "level": len(chain),
        "input": x,
        "output": y,
        "code": code,
        "prompt": prompt,
        "skills_seen": all(name in SEEN for name in chain),
        "answer": y if task == "forward" else x,
    }
    if task == "forward":
        gen_code = render_code(chain, show_defs=True)
        problem["gen_code"] = gen_code
        problem["gen_prompt"] = FORWARD_PROMPT.format(code=gen_code, input=x)
    else:
        problem["gen_code"] = None
        problem["gen_prompt"] = None
    return problem


def gen_forward(n: int, levels: Sequence[int], skills_pool: Sequence[str]) -> tuple[list[dict[str, Any]], int]:
    """Generate forward/composition problems and return (problems, rejects)."""
    return _generate("forward", n, levels, skills_pool)


def gen_inverse(n: int, levels: Sequence[int], skills_pool: Sequence[str]) -> tuple[list[dict[str, Any]], int]:
    """Generate inverse problems and return (problems, rejects)."""
    return _generate("inverse", n, levels, skills_pool)


def gen_eval(n_per_cell: int, task: Task) -> tuple[list[dict[str, Any]], int]:
    """Generate levels 1..4 × seen/held-out evaluation cells with tags."""
    all_problems: list[dict[str, Any]] = []
    total_rejects = 0
    for level in range(1, 5):
        for split, pool in (("seen", SEEN), ("held_out", HELD_OUT)):
            problems, rejects = _generate(task, n_per_cell, [level], pool)
            for problem in problems:
                problem["eval_split"] = split
                problem["cell"] = f"{task}_l{level}_{split}"
            all_problems.extend(problems)
            total_rejects += rejects
    return all_problems, total_rejects


def _generate(task: Task, n: int, levels: Sequence[int], skills_pool: Sequence[str]) -> tuple[list[dict[str, Any]], int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not levels:
        raise ValueError("at least one level is required")
    for level in levels:
        if level not in {1, 2, 3, 4}:
            raise ValueError(f"unsupported level {level}; expected 1..4")
    pool = list(skills_pool)
    if not pool:
        raise ValueError("skills_pool must not be empty")
    for name in pool:
        if name not in SKILLS:
            raise ValueError(f"unknown skill in pool: {name}")

    problems: list[dict[str, Any]] = []
    rejects = 0
    max_attempts = max(1000, n * 1000)
    attempts = 0
    while len(problems) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"failed to generate {n} problems after {rejects} rejects")
        level = random.choice(list(levels))
        chain = [random.choice(pool) for _ in range(level)]
        problem = make_problem(chain, task)
        if problem is None:
            rejects += 1
            continue
        problems.append(problem)
    return problems, rejects


def _pool_names(pool: Pool) -> list[str]:
    if pool == "all":
        return ALL_SKILLS
    if pool == "seen":
        return SEEN
    if pool == "held_out":
        return HELD_OUT
    raise ValueError(f"unknown pool: {pool}")


def _validate_chain(chain: Sequence[str]) -> None:
    if not chain:
        raise ValueError("chain must not be empty")
    for name in chain:
        if name not in SKILLS:
            raise ValueError(f"unknown skill: {name}")


def _write_jsonl(path: Path, problems: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")


def _parse_levels(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate inverse-RL JSONL problem sets.")
    parser.add_argument("--task", choices=["forward", "inverse"], required=True)
    parser.add_argument("--levels", default="1", help="comma-separated levels, e.g. 1 or 1,2,3,4")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--pool", choices=["all", "seen", "held_out"], default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true", help="generate levels 1..4 × seen/held_out cells using --n per cell")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    levels = _parse_levels(args.levels)
    if args.eval:
        problems, rejects = gen_eval(args.n, args.task)
    elif args.task == "forward":
        problems, rejects = gen_forward(args.n, levels, _pool_names(args.pool))
    else:
        problems, rejects = gen_inverse(args.n, levels, _pool_names(args.pool))

    _write_jsonl(args.out, problems)
    total_attempts = len(problems) + rejects
    reject_rate = rejects / total_attempts if total_attempts else 0.0
    print(f"wrote {len(problems)} problems to {args.out}")
    for level in sorted({problem["level"] for problem in problems}):
        print(f"level {level}: {sum(1 for problem in problems if problem['level'] == level)}")
    print(f"rejects: {rejects} / {total_attempts} ({reject_rate:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
