import json
import random

import numpy as np
import pytest

from prompts import extract_last_json
from skills_inverse import SKILLS, apply_forward, sample_param
from verifier import (
    batch_forward_reward,
    batch_inverse_reward,
    chain_skills,
    forward_reward,
    inverse_reward,
    normalize_chain,
    reference_apply,
    reference_invert,
)


# ----------------------------------------------------------------------
# Level-1 rewards for every skill, with per-problem params
# ----------------------------------------------------------------------

@pytest.mark.parametrize("name", list(SKILLS))
def test_level_one_rewards_for_each_skill(name):
    random.seed(1000 + list(SKILLS).index(name))
    sampler = SKILLS[name][2]
    x = sampler()
    param = sample_param(name)
    y = apply_forward(name, x, param)
    problem = {"chain": [[name, param]], "input": x, "output": y}

    assert reference_apply(problem["chain"], x) == y
    assert forward_reward(json.dumps({"output": y}), problem) == 1.0
    assert inverse_reward(json.dumps({"input": x}), problem) == 1.0

    wrong = f"{x}__wrong__"
    assert wrong != x
    assert inverse_reward(json.dumps({"input": wrong}), problem) == 0.0


# ----------------------------------------------------------------------
# normalize_chain: every supported wire format → canonical pairs
# ----------------------------------------------------------------------

def test_normalize_chain_accepts_tuples_lists_and_json_strings():
    canonical = [("repeat_str", 3), ("mirror_str", None)]
    assert normalize_chain([("repeat_str", 3), ("mirror_str", None)]) == canonical
    assert normalize_chain([["repeat_str", 3], ["mirror_str", None]]) == canonical
    assert normalize_chain('[["repeat_str", 3], ["mirror_str", null]]') == canonical
    assert normalize_chain([["repeat_str", 3], "mirror_str"]) == canonical
    assert normalize_chain([["repeat_str", 3], ["mirror_str"]]) == canonical


def test_normalize_chain_bare_name_forms():
    assert normalize_chain("mirror_str") == [("mirror_str", None)]
    assert normalize_chain(["mirror_str"]) == [("mirror_str", None)]
    assert normalize_chain(["mirror_str", "fancy_brackets"]) == [
        ("mirror_str", None),
        ("fancy_brackets", None),
    ]


def test_normalize_chain_single_pair_reads_as_level_one_chain():
    assert normalize_chain(["repeat_str", 3]) == [("repeat_str", 3)]
    assert normalize_chain(["add_prefix", "qz"]) == [("add_prefix", "qz")]
    assert normalize_chain([["repeat_str", 3]]) == [("repeat_str", 3)]


def test_normalize_chain_coerces_digit_string_and_numpy_params():
    assert normalize_chain([["repeat_str", "3"]]) == [("repeat_str", 3)]
    assert normalize_chain([["rotate_str", np.int64(2)]]) == [("rotate_str", 2)]
    coerced = normalize_chain([["rotate_str", np.int64(2)]])[0][1]
    assert type(coerced) is int


def test_normalize_chain_accepts_numpy_rows():
    rows = np.array([["add_prefix", "qz"], ["add_suffix", "ab"]], dtype=object)
    assert normalize_chain(rows) == [("add_prefix", "qz"), ("add_suffix", "ab")]


def test_normalize_chain_rejects_bare_param_skill_and_param_on_no_param():
    with pytest.raises(ValueError):
        normalize_chain(["rotate_str"])
    with pytest.raises(ValueError):
        normalize_chain("rotate_str")
    with pytest.raises(ValueError):
        normalize_chain([["mirror_str", 2]])
    with pytest.raises((TypeError, ValueError)):
        normalize_chain([["repeat_str", True]])
    with pytest.raises((KeyError, TypeError, ValueError)):
        normalize_chain(["not_a_skill"])
    with pytest.raises(ValueError):
        normalize_chain([])


def test_chain_skills_drops_params():
    assert chain_skills([["repeat_str", 4], "mirror_str"]) == ["repeat_str", "mirror_str"]
    assert chain_skills('[["add_prefix", "ab"]]') == ["add_prefix"]


# ----------------------------------------------------------------------
# Reference execution semantics
# ----------------------------------------------------------------------

def test_double_insert_separator_chain_uses_every_other_char_inverse():
    chain = [["insert_separator", "-"], ["insert_separator", "-"]]
    assert reference_apply([chain[0]], "texq") == "t-e-x-q"
    assert reference_apply(chain, "texq") == "t---e---x---q"
    assert reference_invert(chain, "t---e---x---q") == "texq"

    problem = {"chain": chain, "input": "texq", "output": "t---e---x---q"}
    assert inverse_reward('{"input": "texq"}', problem) == 1.0


def test_reference_apply_orders_chain_first_to_last():
    # [f1, f2] means f2(f1(x)): prefix first, then mirror.
    chain = [["add_prefix", "ab"], "mirror_str"]
    assert reference_apply(chain, "cd") == "abcd" + "abcd"[::-1]


def test_inverse_reward_is_functional_match_not_exact_match():
    # reverse_words admits whitespace-variant preimages globally; the paper's
    # backward reward (and ours) accepts any input that reproduces the output.
    problem = {"chain": ["reverse_words"], "input": "a b", "output": "b a"}
    assert inverse_reward('{"input": "a b"}', problem) == 1.0
    assert inverse_reward('{"input": "a  b"}', problem) == 1.0


# ----------------------------------------------------------------------
# Reward parsing edge cases
# ----------------------------------------------------------------------

def test_forward_reward_str_casts_both_sides():
    problem = {"chain": [["repeat_str", 3]], "output": "121212"}
    assert forward_reward('{"output": 121212}', problem) == 1.0
    assert forward_reward('{"output": "121212"}', problem) == 1.0
    assert forward_reward('{"output": "wrong"}', problem) == 0.0
    assert forward_reward('{"answer": "121212"}', problem) == 0.0
    assert forward_reward("no json here", problem) == 0.0


def test_inverse_reward_str_casts_numeric_candidates_and_rejects_others():
    problem = {"chain": [["repeat_str", 2]], "output": "1212"}
    assert inverse_reward('{"input": "12"}', problem) == 1.0
    assert inverse_reward('{"input": 12}', problem) == 1.0
    assert inverse_reward('{"input": true}', problem) == 0.0
    assert inverse_reward('{"input": ["12"]}', problem) == 0.0
    assert inverse_reward('{"output": "12"}', problem) == 0.0


def test_missing_or_malformed_problem_fields_score_zero():
    assert inverse_reward('{"input": "abc"}', {"chain": ["mirror_str"]}) == 0.0
    assert inverse_reward('{"input": "abc"}', {"chain": ["nope"], "output": "x"}) == 0.0
    assert inverse_reward('{"input": "abc"}', {"chain": ["rotate_str"], "output": "x"}) == 0.0


# ----------------------------------------------------------------------
# extract_last_json (unchanged behavior)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Batch wrappers: TRL payload shapes
# ----------------------------------------------------------------------

def _mk(chain, x):
    y = reference_apply(chain, x)
    return {"chain": chain, "input": x, "output": y}, y


def test_batch_rewards_map_completion_to_problem_by_index():
    p0, y0 = _mk([["repeat_str", 2]], "ab")
    p1, y1 = _mk(["mirror_str"], "cd")
    problems = [p0, p1]
    assert batch_inverse_reward(['{"input": "ab"}', '{"input": "cd"}'], problems) == [1.0, 1.0]
    assert batch_forward_reward([json.dumps({"output": y0}), '{"output": "wrong"}'], problems=problems) == [1.0, 0.0]


def test_batch_rewards_accept_single_problem_dict_for_one_completion():
    p0, _y0 = _mk([["add_prefix", "qz"]], "abc")
    assert batch_inverse_reward(['{"input": "abc"}'], p0) == [1.0]
    with pytest.raises(ValueError):
        batch_inverse_reward(['{"input": "a"}', '{"input": "b"}'], p0)


def test_batch_rewards_reconstruct_problems_from_dataset_columns():
    assert batch_inverse_reward(
        ['{"input": "ab"}', '{"input": "cd"}'],
        chain=[[["repeat_str", 2]], ["mirror_str"]],
        output=["abab", "cddc"],
    ) == [1.0, 1.0]

    assert batch_forward_reward(
        ['{"output": "abab"}', '{"output": "wrong"}'],
        chain=[[["repeat_str", 2]], ["mirror_str"]],
        output=["abab", "cddc"],
    ) == [1.0, 0.0]


def test_batch_rewards_accept_json_string_chain_columns():
    # Recommended Arrow-safe storage: chain serialized as a JSON string column.
    assert batch_inverse_reward(
        ['{"input": "ab"}', '{"input": "cd"}'],
        chain=['[["repeat_str", 2]]', '[["mirror_str", null]]'],
        output=["abab", "cddc"],
    ) == [1.0, 1.0]


def test_batch_rewards_accept_bare_name_chain_columns():
    assert batch_inverse_reward(
        ['{"input": "ab"}', '{"input": "cd"}'],
        chain=["mirror_str", "duplicate_every_char"],
        output=["abba", "ccdd"],
    ) == [1.0, 1.0]


def test_batch_rewards_preserve_full_chain_for_single_completion():
    # chain=[["repeat_str", 3]] with ONE completion is the chain itself
    # (a single [name, param] row), not a batch whose first row is taken.
    assert batch_inverse_reward(
        ['{"input": "ab"}'],
        chain=[["repeat_str", 3]],
        output=["ababab"],
    ) == [1.0]

    # Two-element chain broadcast to one completion.
    assert batch_inverse_reward(
        ['{"input": "ab"}'],
        chain=[["repeat_str", 2], ["mirror_str", None]],
        output=["ababbaba"],
    ) == [1.0]


def test_batch_rewards_accept_per_row_pair_chains():
    # Per-row chains arriving as single [skill, param] pairs (the TRL shape
    # the onboarding called out), mixed with a bare-name row.
    assert batch_inverse_reward(
        ['{"input": "ab"}', '{"input": "cd"}'],
        chain=[["repeat_str", 2], "mirror_str"],
        output=["abab", "cddc"],
    ) == [1.0, 1.0]


def test_batch_rewards_accept_numpy_columns():
    chain_col = np.array(
        [json.dumps([["rotate_str", 1]]), json.dumps([["add_suffix", "xy"]])],
        dtype=object,
    )
    output_col = np.array(["bca", "abxy"], dtype=object)
    assert batch_inverse_reward(
        ['{"input": "abc"}', '{"input": "ab"}'],
        chain=chain_col,
        output=output_col,
    ) == [1.0, 1.0]


def test_batch_rewards_require_problem_data():
    with pytest.raises(ValueError):
        batch_inverse_reward(['{"input": "x"}'])
