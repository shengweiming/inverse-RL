"""Trusted reference execution and reward functions for inverse-RL tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from prompts import extract_last_json
from skills_inverse import SKILLS


def reference_apply(chain: Sequence[str], x: str) -> str:
    """Apply the named forward skills in order to ``x``.

    ``chain=[f1, ..., fk]`` means ``f_k(...f_1(x))``. Skill definitions and
    default kwargs come exclusively from ``skills_inverse.SKILLS``.
    """
    value = x
    for name in chain:
        forward, _inverse, _sampler, kwargs, _tier, _origin = SKILLS[name]
        value = forward(value, **kwargs)
    return value


def forward_reward(completion_text: str, problem: dict[str, Any]) -> float:
    """Reward exact JSON ``output`` matches for forward/composition tasks."""
    parsed = extract_last_json(completion_text)
    if parsed is None:
        return 0.0
    return 1.0 if parsed.get("output") == problem.get("output") else 0.0


def inverse_reward(completion_text: str, problem: dict[str, Any]) -> float:
    """Reward exact preimages for inverse tasks via trusted forward execution."""
    parsed = extract_last_json(completion_text)
    if parsed is None:
        return 0.0

    candidate = parsed.get("input")
    if not isinstance(candidate, str):
        return 0.0

    try:
        predicted_output = reference_apply(problem["chain"], candidate)
    except (KeyError, TypeError):
        return 0.0
    return 1.0 if predicted_output == problem.get("output") else 0.0


def _resolve_problems(
    completions: Sequence[str],
    problems: Sequence[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Sequence[dict[str, Any]]:
    """Resolve TRL-style problem columns into a per-completion sequence."""
    if problems is None:
        problems = kwargs.get("problems") or kwargs.get("problem")
    if problems is None:
        raise ValueError("batch reward requires a problems/problem sequence")
    if len(problems) != len(completions):
        raise ValueError("number of problems must match number of completions")
    return problems


def batch_forward_reward(
    completions: Sequence[str],
    problems: Sequence[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Vectorized forward reward for TRL reward callbacks."""
    resolved = _resolve_problems(completions, problems, **kwargs)
    return [forward_reward(completion, problem) for completion, problem in zip(completions, resolved)]


def batch_inverse_reward(
    completions: Sequence[str],
    problems: Sequence[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Vectorized inverse reward for TRL reward callbacks."""
    resolved = _resolve_problems(completions, problems, **kwargs)
    return [inverse_reward(completion, problem) for completion, problem in zip(completions, resolved)]
