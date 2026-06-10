"""Deterministic JSONL task generation for inverse-RL experiments (post-pivot).

Surface contract = the paper's, wholesale. ``chain`` stores per-element
``[skill_name, param]`` pairs (param ``null`` for no-param skills) for trusted
verification; rendered code shows ONLY a hidden-definition ``main_solution``
over the paper's meaningless ``func_N`` identifiers, with each element's
parameter inlined as a literal argument — e.g.::

    def main_solution(x):
        return func_5(func_1(x, 3), 'qz')

Every generated problem is x-form only: ``main_solution`` is a composition of
skills applied to the input variable ``x``. The paper's generator also emits
constant-leaf and binary (concat/interlace) forms — fine for forward
prediction, but a constant leaf has no recoverable input and a binary node
has no unique preimage, so neither is usable as an inversion target. (Stage 1
trained on the paper's own corpus including those forms; only OUR generated
inverse/composition/eval problems are restricted.)

The Stage-1 rejection-sampling path (``show_defs`` rendering, ``gen_code`` /
``gen_prompt`` fields) is deleted: Stage 1 now reuses the paper's published
RFT corpus, so nothing ever needs to see atomic definitions.

Storage note for Stage-2 TRL: a mixed-type ``chain`` column (str names next
to int/str/null params) can upset Arrow. If ``datasets`` complains, store
``json.dumps(problem["chain"])`` as a string column — ``verifier`` normalizes
JSON-string chains transparently.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from prompts import FORWARD_PROMPT, INVERSE_PROMPT
from skills_inverse import (
    PAPER_FUNC_ID,
    SKILLS,
    coerce_param,
    has_param,
    render_param_literal,
    sample_param,
)
from verifier import ChainElement, normalize_chain, reference_apply, reference_invert

Task = Literal["forward", "inverse"]
Pool = Literal["all", "seen", "held_out"]

# The decontamination map is the PAPER's func_N numbering, frozen in
# string_data.py's func_name_mapping — not our registry order. These ids are
# part of the Stage-1 SFT distribution and of the dataset contract.
ID_MAP: dict[str, str] = {name: f"func_{PAPER_FUNC_ID[name]}" for name in SKILLS}
ID_MAP_INV: dict[str, str] = {identifier: name for name, identifier in ID_MAP.items()}

# Held-out = the structural skills we test the reversal curse on. The
# data-rich affixes + repeat stay SEEN so Stage-2 RL has signal to learn from.
HELD_OUT: list[str] = ["rotate_str", "mirror_str", "fancy_brackets"]
SEEN: list[str] = [name for name in SKILLS if name not in HELD_OUT]
ALL_SKILLS: list[str] = list(SKILLS)


def instantiate_chain(chain: Sequence[Any]) -> list[ChainElement]:
    """Resolve a chain spec into fully-parameterized ``(name, param)`` pairs.

    Bare names of param skills get a freshly sampled parameter; explicit
    ``(name, param)`` pairs are respected (this is how tests pin params).
    """
    if not chain:
        raise ValueError("chain must not be empty")
    instantiated: list[ChainElement] = []
    for element in chain:
        if isinstance(element, str):
            name, param = element, None
        elif isinstance(element, Sequence) and not isinstance(element, (str, bytes)) and len(element) in (1, 2):
            name = element[0]
            param = element[1] if len(element) == 2 else None
        else:
            raise TypeError(f"cannot interpret chain element: {element!r}")
        if not isinstance(name, str) or name not in SKILLS:
            raise ValueError(f"unknown skill: {name!r}")
        if has_param(name) and param is None:
            param = sample_param(name)
        instantiated.append((name, coerce_param(name, param)))
    return instantiated


def _expression(chain: Sequence[ChainElement]) -> str:
    expr = "x"
    for name, param in chain:
        if param is None:
            expr = f"{ID_MAP[name]}({expr})"
        else:
            expr = f"{ID_MAP[name]}({expr}, {render_param_literal(name, param)})"
    return expr


def render_code(chain: Sequence[Any]) -> str:
    """Render the hidden-definition ``main_solution`` snippet for a chain.

    Output matches the paper's decontaminated surface exactly: a single
    ``def main_solution(x)`` whose body composes ``func_N`` calls with each
    element's parameter shown as a literal argument. Atomic definitions are
    never shown — Stage 1 installs ``func_N`` semantics, the snippet only
    names them.
    """
    return f"def main_solution(x):\n    return {_expression(normalize_chain(chain))}"


def compose_inverse(chain: Sequence[Any], y: str) -> str:
    """Undo a chain in reverse order with trusted parameterized inverses."""
    return reference_invert(chain, y)


def make_problem(chain: Sequence[Any], task: Task) -> dict[str, Any] | None:
    """Build one verified problem or return ``None`` if round-trip fails.

    All 9 skills are total injections, so the round-trip filter should NEVER
    fire (the generator asserts a zero reject rate in tests); it stays as a
    safety invariant against future registry edits.
    """
    if task not in {"forward", "inverse"}:
        raise ValueError(f"unknown task: {task!r}")

    elements = instantiate_chain(chain)
    first_name = elements[0][0]
    sampler = SKILLS[first_name][2]
    x = sampler()
    y = reference_apply(elements, x)
    if compose_inverse(elements, y) != x:
        return None

    code = render_code(elements)
    prompt = (
        FORWARD_PROMPT.format(code=code, input=x)
        if task == "forward"
        else INVERSE_PROMPT.format(code=code, output=y)
    )
    return {
        "task": task,
        "chain": [[name, param] for name, param in elements],
        "level": len(elements),
        "input": x,
        "output": y,
        "code": code,
        "prompt": prompt,
        "skills_seen": all(name in SEEN for name, _param in elements),
        "answer": y if task == "forward" else x,
    }


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
    skill_counts: dict[str, int] = {name: 0 for name in SKILLS}
    for problem in problems:
        for name, _param in problem["chain"]:
            skill_counts[name] += 1
    for name, count in skill_counts.items():
        if count:
            print(f"skill {name}: {count}")
    print(f"rejects: {rejects} / {total_attempts} ({reject_rate:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
