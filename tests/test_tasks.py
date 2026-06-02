import json
import random

import pytest

from inverse_tasks import ALL_SKILLS, HELD_OUT, ID_MAP, SEEN, compose_inverse, gen_eval, make_problem, render_code
from skills_inverse import SKILLS
from verifier import reference_apply


def test_rendered_code_includes_definitions_by_default():
    chain = ["reverse", "shift_chars", "vigenere"]
    code = render_code(chain)
    assert code == render_code(chain, show_defs=True)
    assert "def func_0" in code
    assert "def func_4" in code
    assert "def func_22" in code
    assert "def _vig" in code
    assert "def main_solution(x):\n    return func_22(func_4(func_0(x)))" in code
    assert code.count("def main_solution") == 1
    assert "def reverse" not in code
    assert "def shift_chars" not in code
    assert "def vigenere" not in code


def test_rendered_code_can_emit_hidden_stub_for_diagnostics_only():
    code = render_code(["reverse", "shift_chars", "vigenere"], show_defs=False)
    assert code == "def main_solution(x):\n    return func_22(func_4(func_0(x)))"
    assert code.startswith("def main_solution")
    assert code.count("def ") == 1
    for forbidden in ("import", "_helper", "_LO", "gcd"):
        assert forbidden not in code


@pytest.mark.parametrize("task", ["forward", "inverse"])
def test_generated_problem_round_trips_and_uses_true_chain(task):
    random.seed(123)
    problem = make_problem(["reverse", "shift_chars"], task)
    assert problem is not None
    assert compose_inverse(problem["chain"], problem["output"]) == problem["input"]
    assert json.loads(json.dumps(problem, ensure_ascii=False)) == problem
    assert all(name in SKILLS for name in problem["chain"])
    assert problem["chain"] == ["reverse", "shift_chars"]
    assert reference_apply(problem["chain"], problem["input"]) == problem["output"]

    code = problem["code"]
    assert "def func_0" in code
    assert "def func_4" in code
    assert "def main_solution(x):\n    return func_4(func_0(x))" in code
    assert code in problem["prompt"]
    for name in SKILLS:
        assert f"def {name}" not in code


def test_show_defs_for_self_contained_chain_omits_global_helpers():
    code = render_code(["insert_separator", "swap_case"], show_defs=True)
    assert "def func_17" in code
    assert "def func_1" in code
    assert "def main_solution(x):\n    return func_1(func_17(x))" in code
    for forbidden in ("import", "_helper", "_atbash_ch", "_vig", "_mult", "_LO", "gcd"):
        assert forbidden not in code


def test_forward_gen_code_matches_canonical_code_and_is_executable():
    random.seed(456)
    problem = make_problem(["atbash", "deterministic_shuffle", "vigenere", "succ_char"], "forward")
    assert problem is not None
    namespace = {}
    exec(problem["gen_code"], namespace)
    assert namespace["main_solution"](problem["input"]) == problem["output"]
    assert problem["gen_code"] == problem["code"]
    assert "def func_2" in problem["gen_code"]
    assert "def func_20" in problem["gen_code"]
    assert "def func_22" in problem["gen_code"]
    assert "def func_19" in problem["gen_code"]
    assert "def _atbash_ch" in problem["gen_code"]
    assert "def _mult" in problem["gen_code"]
    assert "def _vig" in problem["gen_code"]
    assert "from math import gcd" in problem["gen_code"]
    assert "_LO" in problem["gen_code"]
    assert problem["gen_prompt"] == problem["prompt"]


def test_inverse_omits_generation_prompt_fields():
    random.seed(789)
    problem = make_problem(["swap_pairs"], "inverse")
    assert problem is not None
    assert problem["gen_code"] is None
    assert problem["gen_prompt"] is None
    assert "def func_16" in problem["code"]


def test_eval_generation_tags_seen_and_held_out_cells():
    random.seed(0)
    problems, rejects = gen_eval(2, "inverse")
    assert rejects >= 0
    assert len(problems) == 16
    cells = {(p["level"], p["eval_split"]) for p in problems}
    assert cells == {(level, split) for level in range(1, 5) for split in ("seen", "held_out")}
    for problem in problems:
        assert "def main_solution" in problem["code"]
        assert "def func_" in problem["code"]
        if problem["eval_split"] == "seen":
            assert set(problem["chain"]).issubset(SEEN)
            assert problem["skills_seen"] is True
        else:
            assert set(problem["chain"]).issubset(HELD_OUT)
            assert problem["skills_seen"] is False


def test_all_skill_constants_are_consistent():
    assert set(HELD_OUT).isdisjoint(SEEN)
    assert set(HELD_OUT) | set(SEEN) == set(ALL_SKILLS) == set(SKILLS)
    assert list(ID_MAP.values()) == [f"func_{i}" for i in range(len(SKILLS))]
