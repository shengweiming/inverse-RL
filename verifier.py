"""Trusted reference execution and reward functions for inverse-RL tasks.

Post-pivot chain contract: a chain element is a ``(skill_name, param)`` pair
(param ``None`` for no-param skills), and the verifier applies each skill
with THAT problem's parameter. ``normalize_chain`` is the single entry point
that maps every wire format we encounter onto canonical pairs:

* a JSON string (recommended storage for TRL/Arrow datasets — a mixed-type
  ``chain`` column like ``[["repeat_str", 3], ["mirror_str", null]]`` makes
  Arrow unhappy, a JSON-string column never does),
* a bare skill name (no-param skills only),
* a sequence of elements, each a name, a 1-list ``[name]``, or a 2-list
  ``[name, param]`` (tuples and numpy rows also accepted),
* a single ``[name, param]`` pair standing for a level-1 chain.

Reward semantics mirror the paper's ``codeio.py``: the forward reward is
string equality on the predicted output (both sides str-cast — the Gate-G1
lesson); the inverse reward is a FUNCTIONAL preimage match, i.e. the
candidate input is accepted iff re-executing the chain on it reproduces the
recorded output. Non-canonical preimages therefore score 1.0 when they
exist — for these 9 skills that can only happen via ``reverse_words``
whitespace variants, which matches the paper's backward-reward semantics.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from prompts import extract_last_json
from skills_inverse import SKILLS, apply_forward, apply_inverse, coerce_param, has_param

ChainElement = tuple[str, Any]


def _tolist(value: Any) -> Any:
    """Convert numpy arrays/scalars to plain Python containers when possible."""
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _coerce_name(name: Any) -> str:
    name = _tolist(name)
    if not isinstance(name, str):
        raise TypeError(f"chain element name must be a str, got {type(name).__name__} {name!r}")
    if name not in SKILLS:
        raise KeyError(f"unknown skill: {name!r}")
    return str(name)


def _is_param_like(name: str, param: Any) -> bool:
    """True if ``param`` coerces as a parameter for ``name``.

    Skill names are >= 8 characters while paper string params are <= 4, so a
    skill name can never be mistaken for a param (and vice versa); this is
    what keeps the shape predicates below unambiguous.
    """
    try:
        coerce_param(name, _tolist(param))
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _is_chain_element(value: Any) -> bool:
    """Shape predicate: could ``value`` be one chain element?"""
    value = _tolist(value)
    if isinstance(value, str):
        return value in SKILLS
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 1:
            item = _tolist(value[0])
            return isinstance(item, str) and item in SKILLS
        if len(value) == 2:
            name = _tolist(value[0])
            if not (isinstance(name, str) and name in SKILLS):
                return False
            # coerce_param decides: [name, param] for param skills,
            # [name, None] canonical pairs for no-param skills.
            return _is_param_like(name, value[1])
    return False


def _is_chain(value: Any) -> bool:
    """Shape predicate: could ``value`` be one whole chain?"""
    value = _tolist(value)
    if isinstance(value, str):
        if value in SKILLS:
            return True
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return False
        return not isinstance(parsed, str) and _is_chain(parsed)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return False
    if all(_is_chain_element(item) for item in value):
        return True
    # A single [name, param] pair stands for a level-1 chain: its second item
    # (the param) is not itself a valid element, so the all() above fails.
    return _is_chain_element(value)


def _normalize_element(element: Any) -> ChainElement:
    element = _tolist(element)
    if isinstance(element, str):
        name = _coerce_name(element)
        if has_param(name):
            raise ValueError(
                f"{name} requires a parameter; bare-name chain elements are only "
                f"valid for no-param skills (pass [{name!r}, <param>])"
            )
        return (name, None)
    if isinstance(element, Sequence) and not isinstance(element, (str, bytes)):
        if len(element) == 1:
            return _normalize_element(element[0])
        if len(element) == 2:
            name = _coerce_name(element[0])
            return (name, coerce_param(name, _tolist(element[1])))
        raise ValueError(f"chain element must have 1 or 2 items, got {len(element)}: {element!r}")
    raise TypeError(f"cannot interpret chain element: {element!r}")


def normalize_chain(chain: Any) -> list[ChainElement]:
    """Map any supported wire format onto canonical ``(name, param)`` pairs."""
    chain = _tolist(chain)

    if isinstance(chain, str):
        stripped = chain.strip()
        if stripped.startswith(("[", "{", '"')):
            try:
                parsed = json.loads(stripped)
            except ValueError as exc:
                raise ValueError(f"chain string looks like JSON but does not parse: {chain!r}") from exc
            if isinstance(parsed, str):
                return normalize_chain(parsed)
            return normalize_chain(parsed)
        return [_normalize_element(chain)]

    if not isinstance(chain, Sequence) or isinstance(chain, (str, bytes)):
        raise TypeError(f"chain must be a sequence or string, got {type(chain).__name__}")
    if not chain:
        raise ValueError("chain must not be empty")

    # A single [name, param] pair = level-1 chain (only when its items do NOT
    # all read as elements themselves; ["mirror_str", "fancy_brackets"] is a
    # 2-element chain, ["repeat_str", 3] is one element).
    if _is_chain_element(chain) and not all(_is_chain_element(item) for item in chain):
        return [_normalize_element(chain)]

    return [_normalize_element(element) for element in chain]


def chain_skills(chain: Any) -> list[str]:
    """Skill names of a chain in application order, params dropped."""
    return [name for name, _param in normalize_chain(chain)]


def reference_apply(chain: Any, x: str) -> str:
    """Apply the chain's forward skills in order with per-element params.

    ``chain=[(f1, p1), ..., (fk, pk)]`` means ``f_k(...f_1(x, p1)..., pk)``.
    """
    value = x
    for name, param in normalize_chain(chain):
        value = apply_forward(name, value, param)
    return value


def reference_invert(chain: Any, y: str) -> str:
    """Undo the chain with trusted inverses, in reverse order."""
    value = y
    for name, param in reversed(normalize_chain(chain)):
        value = apply_inverse(name, value, param)
    return value


def forward_reward(completion_text: str, problem: dict[str, Any]) -> float:
    """Reward JSON ``output`` matches for forward/composition tasks.

    Both sides are str-cast before comparison (Gate-G1 lesson: models emit
    ``{"output": 3}`` for digit strings; parquet stores everything as str).
    """
    parsed = extract_last_json(completion_text)
    if parsed is None:
        return 0.0
    predicted = parsed.get("output")
    expected = problem.get("output")
    if predicted is None or expected is None:
        return 0.0
    return 1.0 if str(predicted) == str(expected) else 0.0


def inverse_reward(completion_text: str, problem: dict[str, Any]) -> float:
    """Reward functional preimages using the trusted forward reference.

    The candidate ``input`` is accepted iff ``reference_apply(chain, input)``
    reproduces the recorded output (paper backward-reward semantics). Numeric
    candidates are str-cast (harmless leniency, mirrors forward_reward); any
    other non-str candidate scores 0.
    """
    parsed = extract_last_json(completion_text)
    if parsed is None:
        return 0.0

    candidate = parsed.get("input")
    if isinstance(candidate, bool):
        return 0.0
    if isinstance(candidate, (int, float)):
        candidate = str(candidate)
    if not isinstance(candidate, str):
        return 0.0

    expected = problem.get("output")
    if expected is None:
        return 0.0

    try:
        predicted_output = reference_apply(problem["chain"], candidate)
    except (KeyError, TypeError, ValueError):
        return 0.0
    return 1.0 if predicted_output == str(expected) else 0.0


def _is_batched_column(value: Any, n_items: int) -> bool:
    """True if ``value`` looks like a per-example batch column."""
    value = _tolist(value)
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == n_items


def _column_value_at(key: str, value: Any, index: int, n_items: int) -> Any:
    """Read one example from a TRL/HF-style keyword column."""
    value = _tolist(value)

    if key == "chain":
        # Scalar broadcast: a bare name or a JSON-string chain for all rows.
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            # Batch-of-chains first: row count matches and every row reads as
            # a chain (JSON strings, bare names, element lists, or a single
            # [name, param] pair all count).
            if len(value) == n_items and all(_is_chain(item) for item in value):
                return value[index]
            # Otherwise, if the whole value reads as ONE chain, broadcast it.
            # This is what makes chain=[["repeat_str", 3]] with one completion
            # resolve to the chain itself rather than to its first row.
            if _is_chain(value):
                return value

    if _is_batched_column(value, n_items):
        return value[index]
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
