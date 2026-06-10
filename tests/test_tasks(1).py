import json
import random
import re

import pytest

import inverse_tasks
from inverse_tasks import (
    ALL_SKILLS,
    HELD_OUT,
    ID_MAP,
    ID_MAP_INV,
    SEEN,
    gen_eval,
    gen_forward,
    gen_inverse,
    instantiate_chain,
    make_problem,
    render_code,
)
from skills_inverse import SKILLS
from verifier import inverse_reward, normalize_chain, reference_apply

PAPER_ID_MAP = {
    "repeat_str": "func_1",
    "reverse_words": "func_4",
    "add_prefix": "func_5",
    "add_suffix": "func_6",
    "rotate_str": "func_8",
    "mirror_str": "func_9",
    "insert_separator": "func_13",
    "duplicate_every_char": "func_14",
    "fancy_brackets": "func_15",
}


# ----------------------------------------------------------------------
# Contract constants
# ----------------------------------------------------------------------

def test_id_map_is_the_paper_func_numbering():
    assert ID_MAP == PAPER_ID_MAP
    assert ID_MAP_INV == {v: k for k, v in PAPER_ID_MAP.items()}


def test_held_out_seen_partition():
    assert HELD_OUT == ["rotate_str", "mirror_str", "duplicate_every_char"]
    assert SEEN == [
        "repeat_str",
        "reverse_words",
        "add_prefix",
        "add_suffix",
        "insert_separator",
        "fancy_brackets",
    ]
    assert set(SEEN) | set(HELD_OUT) == set(SKILLS)
    assert not set(SEEN) & set(HELD_OUT)
    assert ALL_SKILLS == list(SKILLS)
    assert len(ALL_SKILLS) == 9


# ----------------------------------------------------------------------
# render_code: exact paper surface
# ----------------------------------------------------------------------

def test_render_code_int_param_literal():
    assert render_code([("repeat_str", 3)]) == "def main_solution(x):\n    return func_1(x, 3)"


def test_render_code_str_param_literal():
    assert render_code([("add_prefix", "qz")]) == "def main_solution(x):\n    return func_5(x, 'qz')"


def test_render_code_no_param_skill():
    assert render_code(["mirror_str"]) == "def main_solution(x):\n    return func_9(x)"


def test_render_code_nests_first_skill_innermost():
    code = render_code([("repeat_str", 2), ("add_suffix", "ab")])
    assert code == "def main_solution(x):\n    return func_6(func_1(x, 2), 'ab')"


def test_render_code_level_three_mixed():
    chain = [("insert_separator", "-"), "duplicate_every_char", ("rotate_str", 1)]
    assert render_code(chain) == (
        "def main_solution(x):\n    return func_8(func_14(func_13(x, '-')), 1)"
    )


def test_render_code_hides_definitions_and_true_names():
    chain = [("add_prefix", "ab"), "fancy_brackets", ("repeat_str", 4)]
    code = render_code(chain)
    assert code.count("def ") == 1
    assert code.startswith("def main_solution(x):\n    return ")
    for true_name in SKILLS:
        assert true_name not in code


# ----------------------------------------------------------------------
# instantiate_chain
# ----------------------------------------------------------------------

def test_instantiate_chain_respects_pinned_params_and_samples_bare_names():
    random.seed(7)
    elements = instantiate_chain([("rotate_str", 2), "add_prefix", "mirror_str"])
    assert elements[0] == ("rotate_str", 2)
    name, param = elements[1]
    assert name == "add_prefix"
    assert isinstance(param, str) and 2 <= len(param) <= 4 and param.islower()
    assert elements[2] == ("mirror_str", None)


def test_instantiate_chain_rejects_unknown_and_empty():
    with pytest.raises(ValueError):
        instantiate_chain(["nope"])
    with pytest.raises(ValueError):
        instantiate_chain([])


# ----------------------------------------------------------------------
# make_problem: schema and invariants
# ----------------------------------------------------------------------

def test_make_problem_samples_and_stores_params():
    random.seed(11)
    problem = make_problem(["rotate_str"], "inverse")
    assert problem is not None
    ((name, param),) = problem["chain"]
    assert name == "rotate_str"
    assert param in (1, 2, 3)
    assert problem["level"] == 1
    assert problem["skills_seen"] is False  # rotate_str is held out


def test_make_problem_input_is_paper_distribution():
    random.seed(12)
    for _ in range(50):
        problem = make_problem(["mirror_str", "add_suffix"], "inverse")
        assert re.fullmatch(r"[a-z]{3,10}", problem["input"])


def test_make_problem_has_no_generation_prompt_fields():
    random.seed(13)
    problem = make_problem([("repeat_str", 2)], "forward")
    assert "gen_code" not in problem
    assert "gen_prompt" not in problem
    expected_keys = {
        "task", "chain", "level", "input", "output", "code",
        "prompt", "skills_seen", "answer",
    }
    assert set(problem) == expected_keys


def test_make_problem_prompt_and_answer_fields():
    random.seed(14)
    fwd = make_problem([("add_prefix", "qz")], "forward")
    assert fwd["code"] in fwd["prompt"]
    assert f'main_solution("{fwd["input"]}")' in fwd["prompt"]
    assert fwd["answer"] == fwd["output"]

    inv = make_problem([("add_prefix", "qz")], "inverse")
    assert inv["code"] in inv["prompt"]
    assert f'returned "{inv["output"]}"' in inv["prompt"]
    assert inv["answer"] == inv["input"]


def test_make_problem_survives_json_round_trip_into_verifier():
    random.seed(15)
    problem = make_problem(["insert_separator", ("repeat_str", 3), "fancy_brackets"], "inverse")
    revived = json.loads(json.dumps(problem, ensure_ascii=False))
    assert reference_apply(revived["chain"], revived["input"]) == revived["output"]
    assert inverse_reward(json.dumps({"input": revived["input"]}), revived) == 1.0
    # Arrow-safe variant: chain stored as a JSON string column.
    revived["chain"] = json.dumps(problem["chain"])
    assert inverse_reward(json.dumps({"input": problem["input"]}), revived) == 1.0


def test_make_problem_rejects_unknown_task():
    with pytest.raises(ValueError):
        make_problem(["mirror_str"], "sideways")


# ----------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------

def test_generators_are_deterministic_under_seed():
    random.seed(0)
    first, _ = gen_inverse(5, [1, 2], SEEN)
    random.seed(0)
    second, _ = gen_inverse(5, [1, 2], SEEN)
    assert first == second


def test_generator_reject_rate_is_zero_for_total_injections():
    random.seed(1)
    problems, rejects = gen_inverse(200, [1, 2, 3, 4], ALL_SKILLS)
    assert rejects == 0
    assert len(problems) == 200

    random.seed(2)
    problems, rejects = gen_forward(200, [1, 2, 3, 4], ALL_SKILLS)
    assert rejects == 0
    assert len(problems) == 200


def test_generated_chains_verify_and_render_consistently():
    random.seed(3)
    problems, _ = gen_inverse(60, [1, 2, 3], ALL_SKILLS)
    for problem in problems:
        elements = normalize_chain(problem["chain"])
        assert reference_apply(elements, problem["input"]) == problem["output"]
        assert render_code(elements) == problem["code"]
        assert problem["level"] == len(elements)
        assert problem["skills_seen"] == all(name in SEEN for name, _ in elements)


def test_generator_pool_restriction():
    random.seed(4)
    problems, _ = gen_inverse(40, [1, 2], HELD_OUT)
    for problem in problems:
        for name, _param in problem["chain"]:
            assert name in HELD_OUT


def test_gen_eval_tags_cells_and_respects_pools():
    random.seed(5)
    problems, rejects = gen_eval(3, "inverse")
    assert rejects == 0
    assert len(problems) == 3 * 4 * 2
    cells = {problem["cell"] for problem in problems}
    assert cells == {
        f"inverse_l{level}_{split}" for level in (1, 2, 3, 4) for split in ("seen", "held_out")
    }
    for problem in problems:
        pool = SEEN if problem["eval_split"] == "seen" else HELD_OUT
        for name, _param in problem["chain"]:
            assert name in pool


def test_generator_validates_arguments():
    with pytest.raises(ValueError):
        gen_inverse(5, [], SEEN)
    with pytest.raises(ValueError):
        gen_inverse(5, [5], SEEN)
    with pytest.raises(ValueError):
        gen_inverse(5, [1], [])
    with pytest.raises(ValueError):
        gen_inverse(5, [1], ["nope"])
    with pytest.raises(ValueError):
        gen_inverse(-1, [1], SEEN)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def test_cli_writes_jsonl_with_param_chains(tmp_path, capsys):
    out = tmp_path / "probe.jsonl"
    rc = inverse_tasks.main([
        "--task", "inverse", "--levels", "1,2", "--n", "12",
        "--pool", "seen", "--out", str(out), "--seed", "0",
    ])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    for line in lines:
        problem = json.loads(line)
        assert problem["task"] == "inverse"
        assert problem["level"] in (1, 2)
        for name, param in problem["chain"]:
            assert name in SEEN
            assert ID_MAP[name] in problem["code"]
        assert inverse_reward(json.dumps({"input": problem["input"]}), problem) == 1.0
    captured = capsys.readouterr().out
    assert "rejects: 0 /" in captured
