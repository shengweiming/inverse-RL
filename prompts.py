"""Prompt templates and JSON extraction helpers for inverse-RL tasks."""

from __future__ import annotations

import json
from typing import Any


FORWARD_PROMPT = '''Given the following code:
{code}

Predict main_solution("{input}"). Reason without running code. Answer JSON {{"output": ...}}.'''

INVERSE_PROMPT = '''Given the following code:
{code}

main_solution(x) returned "{output}". Find an input x reproducing that output exactly. Reason without running code. Answer JSON {{"input": ...}}.'''


def extract_last_json(text: str) -> dict[str, Any] | None:
    """Return the last complete JSON object found in *text*, or None.

    The scanner is deliberately tolerant of model completions: JSON may appear
    inside ```json fences, followed by prose, preceded by reasoning, or after an
    abandoned ``{`` fragment. It still requires a syntactically valid JSON object
    and never evaluates Python code.
    """
    if not isinstance(text, str):
        return None

    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, dict[str, Any]]] = []

    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, relative_end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append((start, start + relative_end, obj))

    for index in range(len(candidates) - 1, -1, -1):
        start, end, obj = candidates[index]
        contained_in_larger_object = any(
            outer_start < start and end <= outer_end
            for outer_start, outer_end, _outer_obj in candidates
        )
        if not contained_in_larger_object:
            return obj

    return None
