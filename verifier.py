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


def _reference_apply_on_declared_domains(chain: Sequence[str], x: str) -> str | None:
    """Apply ``chain`` only while each skill round-trips on its declared domain.

    Some provided skills are injective only on their sampled/domain-filtered
    inputs. For inverse reward, model-proposed preimages must stay inside those
    domains; otherwise a malformed string can collide under the raw forward
    implementation (for example odd-length ``riffle_shuffle`` inputs drop a
    character). The trusted inverse functions are used only as domain witnesses:
    each forward step must satisfy ``inverse(forward(value)) == value``.
    """
    value = x
    for name in chain:
        forward, inverse, _sampler, kwargs, _tier, _origin = SKILLS[name]
        next_value = forward(value, **kwargs)
        if inverse(next_value, **kwargs) != value:
            return None
        value = next_value
    return value


def forward_reward(completion_text: str, problem: dict[str, Any]) -> float:
    """Reward exact JSON ``output`` matches for forward/composition tasks."""
    parsed = extract_last_json(completion_text)
    if parsed is None:
        return 0.0
    return 1.0 if parsed.get("output") == problem.get("output") else 0.0


def inverse_reward(completion_text: str, problem: dict[str, Any]) -> float:
    """Reward exact in-domain preimages for inverse tasks."""
    parsed = extract_last_json(completion_text)
    if parsed is None:
        return 0.0

    candidate = parsed.get("input")
    if not isinstance(candidate, str):
        return 0.0

    try:
        predicted_output = _reference_apply_on_declared_domains(problem["chain"], candidate)
    except (KeyError, TypeError):
        return 0.0
    return 1.0 if predicted_output == problem.get("output") else 0.0


def _is_batched_column(value: Any, n_items: int) -> bool:
    """Return True if ``value`` looks like a per-example batch column."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == n_items


def _column_value_at(key: str, value: Any, index: int, n_items: int) -> Any:
    """Read one example from a TRL/HF-style keyword column."""
    if key == "chain" and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        # A single example may be passed as chain=["f1", "f2"] rather than
        # chain=[["f1", "f2"]]. Preserve that full chain for n=1.
        if n_items == 1 and all(isinstance(item, str) for item in value):
            return list(value)

    if _is_batched_column(value, n_items):
        item = value[index]
        # Be permissive for callers that batch level-1 chains as skill-name
        # strings rather than one-element lists.
        if key == "chain" and isinstance(item, str):
            return [item]
        return item

    return value


def _resolve_problems(
    completions: Sequence[str],
    problems: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Sequence[dict[str, Any]]:
    """Resolve nested or columnar TRL problem data into per-completion dicts."""
    n_items = len(completions)

    if problems is None:
        problems = kwargs.get("problems") or kwargs.get("problem")

    if problems is not None:
        if isinstance(problems, dict):
            if n_items != 1:
                raise ValueError("a single problem dict can only be used with one completion")
            return [problems]
        if len(problems) != n_items:
            raise ValueError("number of problems must match number of completions")
        return problems

    problem_keys = {
        key
        for key in kwargs
        if not key.startswith("_") and key not in {"completion", "completions", "prompt", "prompts"}
    }
    if not ({"chain", "output"} & problem_keys):
        raise ValueError("batch reward requires problem dicts or problem columns")

    resolved: list[dict[str, Any]] = []
    for index in range(n_items):
        problem = {
            key: _column_value_at(key, kwargs[key], index, n_items)
            for key in problem_keys
        }
        resolved.append(problem)
    return resolved


def batch_forward_reward(
    completions: Sequence[str],
    problems: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Vectorized forward reward for TRL reward callbacks."""
    resolved = _resolve_problems(completions, problems, **kwargs)
    return [forward_reward(completion, problem) for completion, problem in zip(completions, resolved)]


def batch_inverse_reward(
    completions: Sequence[str],
    problems: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Vectorized inverse reward for TRL reward callbacks."""
    resolved = _resolve_problems(completions, problems, **kwargs)
    return [inverse_reward(completion, problem) for completion, problem in zip(completions, resolved)]
