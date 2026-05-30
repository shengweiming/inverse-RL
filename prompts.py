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
    inside ```json fences, followed by prose, or preceded by reasoning. It still
    requires a syntactically valid JSON object and never evaluates Python code.
    """
    if not isinstance(text, str):
        return None

    last_obj: dict[str, Any] | None = None
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue

        if char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : index + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(obj, dict):
                        last_obj = obj
                start = None

    return last_obj
