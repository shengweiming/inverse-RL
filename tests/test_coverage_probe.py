"""CPU unit tests for scripts/coverage_probe.py with a stubbed generator.

Generation is the only GPU-touching piece of the probe and is injected as a
``generate_fn(prompts, k) -> [[(text, n_tokens), ...], ...]`` callable, so the
guards, scoring, aggregation, checkpointing, and the CLI all test here.
"""

from __future__ import annotations

import json
import random

import pytest

from inverse_tasks import SEEN, make_problem
from scripts.coverage_probe import (
    EXPECTED_DATA_CONTRACT,
    ProbeError,
    check_data_contract,
    check_skill_census,
    format_summary,
    load_rows,
    main,
    p95,
    run_probe,
    score_problem,
    select_problems,
    summarize,
)

# ----------------------------------------------------------------------
# Fixtures: real generated problems, stubbed rollouts
# ----------------------------------------------------------------------


def _seen_problems(per_skill: int = 2) -> list[dict]:
    random.seed(1234)
    problems = []
    for _ in range(per_skill):
        for name in SEEN:
            problems.append(make_problem([name], "inverse"))
    assert all(problems)
    return problems


def _correct(problem: dict) -> tuple[str, int]:
    return (json.dumps({"input": problem["input"]}), 40)


_WRONG = ("no json here", 25)


def _stub_generate(problems: list[dict], successes_by_prompt: dict[str, int]):
    """Stub: per prompt, the configured number of correct rollouts, rest wrong."""
    by_prompt = {p["prompt"]: p for p in problems}

    def generate_fn(prompts, k):
        out = []
        for prompt in prompts:
            s = successes_by_prompt.get(prompt, 0)
            out.append([_correct(by_prompt[prompt])] * s + [_WRONG] * (k - s))
        return out

    return generate_fn


def _raising_generate(*_args, **_kwargs):
    raise AssertionError("generate_fn must not be called on the skip path")


def _write_data(tmp_path, problems, contract=EXPECTED_DATA_CONTRACT):
    data = tmp_path / "inv_l1_seen_train.jsonl"
    data.write_text("".join(json.dumps(p) + "\n" for p in problems), encoding="utf-8")
    if contract is not None:
        (tmp_path / "DATA_CONTRACT.txt").write_text(contract + "\n", encoding="utf-8")
    return data


# ----------------------------------------------------------------------
# Stale-data guards
# ----------------------------------------------------------------------


def test_data_contract_guard_missing_stale_and_ok(tmp_path):
    data = _write_data(tmp_path, _seen_problems(1), contract=None)
    with pytest.raises(ProbeError, match="no DATA_CONTRACT.txt"):
        check_data_contract(data)
    (tmp_path / "DATA_CONTRACT.txt").write_text("v3-heldout-mirror\n", encoding="utf-8")
    with pytest.raises(ProbeError, match="stale data contract"):
        check_data_contract(data)
    (tmp_path / "DATA_CONTRACT.txt").write_text(EXPECTED_DATA_CONTRACT + "\n", encoding="utf-8")
    check_data_contract(data)


def test_skill_census_accepts_exact_v4_seen_set():
    check_skill_census(_seen_problems(1))


def test_skill_census_rejects_held_out_contamination():
    random.seed(7)
    problems = _seen_problems(1) + [make_problem(["rotate_str"], "inverse")]
    with pytest.raises(ProbeError, match="unexpected.*rotate_str"):
        check_skill_census(problems)


def test_skill_census_rejects_missing_seen_skill():
    problems = [p for p in _seen_problems(1) if p["chain"][0][0] != "fancy_brackets"]
    with pytest.raises(ProbeError, match="missing.*fancy_brackets"):
        check_skill_census(problems)


def test_skill_census_rejects_wrong_task_level_and_empty():
    random.seed(7)
    forward = [dict(p, task="forward") for p in _seen_problems(1)]
    with pytest.raises(ProbeError, match="expected 'inverse'"):
        check_skill_census(forward)
    level2 = _seen_problems(1) + [make_problem(["add_prefix", "add_suffix"], "inverse")]
    with pytest.raises(ProbeError, match="level-1"):
        check_skill_census(level2)
    with pytest.raises(ProbeError, match="empty"):
        check_skill_census([])


# ----------------------------------------------------------------------
# Selection, scoring, aggregation
# ----------------------------------------------------------------------


def test_select_problems_is_seeded_and_capped():
    problems = _seen_problems(3)
    a = select_problems(problems, 5, seed=0)
    b = select_problems(problems, 5, seed=0)
    c = select_problems(problems, 5, seed=1)
    assert a == b and len(a) == 5
    assert a != c
    assert [i for i, _p in a] == sorted(i for i, _p in a)
    assert select_problems(problems, 10_000, seed=0) == list(enumerate(problems))


def test_p95_nearest_rank():
    assert p95([7.0]) == 7.0
    assert p95(list(range(1, 101))) == 95.0
    assert p95([3.0, 1.0, 2.0]) == 3.0
    with pytest.raises(ValueError):
        p95([])


def test_score_problem_counts_functional_preimages_and_token_stats():
    random.seed(5)
    problem = make_problem([["repeat_str", 2]], "inverse")
    rollouts = [_correct(problem), _WRONG, ("{\"input\": 99}", 12), _correct(problem)]
    row = score_problem(3, problem, rollouts)
    assert row["problem_index"] == 3
    assert row["skill"] == "repeat_str"
    assert row["success_count"] == 2
    assert row["k"] == 4
    assert row["mean_tokens"] == pytest.approx((40 + 25 + 12 + 40) / 4)
    assert row["p95_tokens"] == 40.0


def test_summarize_pass_at_k_and_trainable_fraction():
    def row(skill, c, k=16):
        return {"problem_index": 0, "skill": skill, "success_count": c, "k": k,
                "mean_tokens": 50.0, "p95_tokens": 80.0}

    rows = [row("add_prefix", 16), row("add_prefix", 3), row("add_prefix", 0),
            row("insert_separator", 0), row("insert_separator", 0)]
    summary = summarize(rows)
    assert summary["add_prefix"]["n"] == 3
    assert summary["add_prefix"]["pass_at_k"] == pytest.approx(2 / 3)
    # all-correct (c=16) and all-wrong (c=0) problems carry no GRPO gradient
    assert summary["add_prefix"]["trainable_fraction"] == pytest.approx(1 / 3)
    assert summary["insert_separator"]["pass_at_k"] == 0.0
    assert summary["insert_separator"]["trainable_fraction"] == 0.0
    text = format_summary(summary, 16)
    assert "G3 rule" in text and "50/50" in text


# ----------------------------------------------------------------------
# run_probe: checkpointing, resume, skip-guard
# ----------------------------------------------------------------------


def test_run_probe_writes_csv_and_cleans_partial(tmp_path):
    problems = _seen_problems(2)
    selected = select_problems(problems, len(problems), seed=0)
    successes = {p["prompt"]: i % 4 for i, p in enumerate(problems)}
    out = tmp_path / "coverage_probe.csv"
    rows = run_probe(selected, _stub_generate(problems, successes), k=4, out_path=out, checkpoint_every=5)
    assert out.exists()
    assert not (tmp_path / "coverage_probe.partial.csv").exists()
    assert load_rows(out) == sorted(rows, key=lambda r: r["problem_index"])
    assert {r["problem_index"] for r in rows} == {i for i, _p in selected}
    for (_i, problem), row in zip(selected, sorted(rows, key=lambda r: r["problem_index"])):
        assert row["success_count"] == successes[problem["prompt"]]


def test_run_probe_skips_when_csv_exists(tmp_path):
    problems = _seen_problems(1)
    selected = select_problems(problems, len(problems), seed=0)
    out = tmp_path / "coverage_probe.csv"
    first = run_probe(selected, _stub_generate(problems, {}), k=4, out_path=out, checkpoint_every=100)
    again = run_probe(selected, _raising_generate, k=4, out_path=out, checkpoint_every=100)
    assert again == sorted(first, key=lambda r: r["problem_index"])


def test_run_probe_resumes_from_partial_checkpoint(tmp_path):
    problems = _seen_problems(2)
    selected = select_problems(problems, len(problems), seed=0)
    out = tmp_path / "coverage_probe.csv"
    inner = _stub_generate(problems, {p["prompt"]: 1 for p in problems})
    calls = {"n": 0}

    def dies_after_first_chunk(prompts, k):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("simulated session death")
        return inner(prompts, k)

    with pytest.raises(RuntimeError):
        run_probe(selected, dies_after_first_chunk, k=4, out_path=out, checkpoint_every=4)
    partial = tmp_path / "coverage_probe.partial.csv"
    assert partial.exists() and not out.exists()
    assert len(load_rows(partial)) == 4

    served: list[str] = []

    def counting(prompts, k):
        served.extend(prompts)
        return inner(prompts, k)

    rows = run_probe(selected, counting, k=4, out_path=out, checkpoint_every=4)
    assert len(rows) == len(selected)
    assert len(served) == len(selected) - 4  # only the remaining problems regenerate
    assert out.exists() and not partial.exists()


def test_run_probe_rejects_mismatched_partial(tmp_path):
    problems = _seen_problems(1)
    selected = select_problems(problems, len(problems), seed=0)
    out = tmp_path / "coverage_probe.csv"
    run_probe(selected, _stub_generate(problems, {}), k=4, out_path=out, checkpoint_every=100)
    out.rename(tmp_path / "coverage_probe.partial.csv")
    with pytest.raises(ProbeError, match="--n/--k/--seed"):
        run_probe(selected, _raising_generate, k=8, out_path=out, checkpoint_every=100)


# ----------------------------------------------------------------------
# CLI end-to-end with a stubbed generator
# ----------------------------------------------------------------------


def test_main_end_to_end_with_stub(tmp_path, capsys):
    problems = _seen_problems(3)
    data = _write_data(tmp_path, problems)
    out = tmp_path / "results" / "coverage_probe.csv"
    successes = {p["prompt"]: i % 5 for i, p in enumerate(problems)}
    argv = ["--data", str(data), "--out", str(out), "--n", "12", "--k", "4",
            "--seed", "0", "--checkpoint-every", "5"]

    assert main(argv, generate_fn=_stub_generate(problems, successes)) == 0
    assert out.exists()
    assert len(load_rows(out)) == 12
    printed = capsys.readouterr().out
    assert "G3 rule" in printed and "pass@4" in printed

    # second invocation hits the skip-guard and never generates
    assert main(argv, generate_fn=_raising_generate) == 0
    assert "[skip]" in capsys.readouterr().out


def test_main_hard_fails_on_stale_contract(tmp_path):
    data = _write_data(tmp_path, _seen_problems(1), contract="v3-heldout-mirror")
    with pytest.raises(ProbeError, match="stale data contract"):
        main(["--data", str(data), "--out", str(tmp_path / "x.csv")], generate_fn=_raising_generate)
