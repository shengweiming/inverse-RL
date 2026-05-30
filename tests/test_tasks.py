import json
import random

import pytest

from inverse_tasks import ALL_SKILLS, HELD_OUT, ID_MAP, SEEN, compose_inverse, gen_eval, make_problem, render_code
from skills_inverse import SKILLS
from verifier import reference_apply


def test_hidden_code_uses_only_remapped_identifiers():
    chain = ["reverse", "shift_chars", "vigenere"]
    code = render_code(chain, show_defs=False)
    assert code == "def main_solution(x):\n    return func_22(func_4(func_0(x)))"
    assert "def func_" not in code
    assert code.count("def ") == 1
    assert "main_solution" in code
    for name in SKILLS:
        assert name not in code
    for name in chain:
        assert ID_MAP[name] in code


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
    assert code.startswith("def main_solution(x):\n    return ")
    assert code.count("def ") == 1
    assert "func_" in code
    for name in SKILLS:
        assert name not in code


def test_forward_gen_code_is_self_contained_and_executable():
    random.seed(456)
    problem = make_problem(["atbash", "deterministic_shuffle", "vigenere", "succ_char"], "forward")
    assert problem is not None
    namespace = {}
    exec(problem["gen_code"], namespace)
    assert namespace["main_solution"](problem["input"]) == problem["output"]
    assert problem["gen_prompt"] is not None


def test_inverse_omits_generation_prompt_fields():
    random.seed(789)
    problem = make_problem(["swap_pairs"], "inverse")
    assert problem is not None
    assert problem["gen_code"] is None
    assert problem["gen_prompt"] is None


def test_eval_generation_tags_seen_and_held_out_cells():
    random.seed(0)
    problems, rejects = gen_eval(2, "inverse")
    assert rejects >= 0
    assert len(problems) == 16
    cells = {(p["level"], p["eval_split"]) for p in problems}
    assert cells == {(level, split) for level in range(1, 5) for split in ("seen", "held_out")}
    for problem in problems:
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
