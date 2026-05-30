import json
import random

import pytest

from prompts import extract_last_json
from skills_inverse import SKILLS
from verifier import (
    batch_forward_reward,
    batch_inverse_reward,
    forward_reward,
    inverse_reward,
    reference_apply,
)


@pytest.mark.parametrize("name", list(SKILLS))
def test_level_one_rewards_for_each_skill(name):
    random.seed(1000 + list(SKILLS).index(name))
    forward, _inverse, sampler, kwargs, _tier, _origin = SKILLS[name]
    x = sampler()
    y = forward(x, **kwargs)
    problem = {"chain": [name], "input": x, "output": y}

    assert reference_apply([name], x) == y
    assert forward_reward(json.dumps({"output": y}), problem) == 1.0
    assert inverse_reward(json.dumps({"input": x}), problem) == 1.0

    wrong = f"{x}__wrong__"
    assert wrong != x
    assert inverse_reward(json.dumps({"input": wrong}), problem) == 0.0


def test_extract_last_json_handles_fences_and_trailing_prose():
    text = '''Reasoning...
```json
{"output": "first"}
```
More prose {not json}
```json
{"input": "last"}
```
trailing prose'''
    assert extract_last_json(text) == {"input": "last"}


def test_extract_last_json_keeps_nested_objects_intact():
    assert extract_last_json('prefix {"outer": {"inner": 1}} suffix') == {"outer": {"inner": 1}}


def test_extract_last_json_ignores_abandoned_brace_before_final_answer():
    text = 'scratch { ... final {"output": "ok"} trailing prose'
    assert extract_last_json(text) == {"output": "ok"}


def test_missing_or_non_string_inverse_input_scores_zero():
    problem = {"chain": ["reverse"], "output": "cba"}
    assert inverse_reward('{"output": "abc"}', problem) == 0.0
    assert inverse_reward('{"input": 123}', problem) == 0.0


def test_batch_rewards_map_completion_to_problem_by_index():
    problems = [
        {"chain": ["reverse"], "output": "cba"},
        {"chain": ["swap_case"], "output": "AbC"},
    ]
    assert batch_inverse_reward(['{"input": "abc"}', '{"input": "aBc"}'], problems) == [1.0, 1.0]
    assert batch_forward_reward(['{"output": "cba"}', '{"output": "wrong"}'], problems=problems) == [1.0, 0.0]
