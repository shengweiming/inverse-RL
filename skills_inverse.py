"""The 9 paper skills for the inverse-RL experiment (post-pivot contract).

Source of truth: Yuan et al., RL-Compositionality (arXiv 2509.25123),
``examples/data_preprocess/string_data.py``. Forward implementations below are
VERBATIM copies of the paper's functions; we add a trusted inverse for each.
Every skill is a total injection on arbitrary strings (left inverse exists for
all inputs), with one caveat: ``reverse_words`` is only injective up to
whitespace normalization, and on the paper's input distribution (lowercase
ascii, length 3-10, NO spaces — see ``sample_input``) it is the identity.
Stage-1 SFT used exactly that distribution, so generated problems here must
too; multi-word inputs would probe semantics the model never saw.

Post-pivot registry contract::

    SKILLS[name] = (forward, inverse, input_sampler, param_sampler, paper_func_id)

* ``forward(s)`` / ``forward(s, param)`` — verbatim paper implementation.
* ``inverse(s)`` / ``inverse(s, param)`` — trusted left inverse.
* ``input_sampler()`` — paper input distribution (all skills share it).
* ``param_sampler()`` — draws a param from the paper's range, or ``None``
  for the four no-param skills.
* ``paper_func_id`` — int N such that rendered code calls ``func_N``.

A chain element is a ``(skill_name, param)`` pair (param ``None`` for
no-param skills). Params vary per problem and are shown as literal arguments
in rendered code, e.g. ``func_1(x, 3)`` or ``func_5(x, 'qz')``.

Param ranges (paper's ``random_expr``):
    repeat_str        n   ∈ {2, 3, 4}
    rotate_str        n   ∈ {1, 2, 3}
    add_prefix        pre ∈ lowercase, length 2-4
    add_suffix        suf ∈ lowercase, length 2-4
    insert_separator  sep ∈ {'-', '_', '|'}
"""

from __future__ import annotations

import operator
import random
import string
from typing import Any, Callable

LOWER = string.ascii_lowercase


def sample_input() -> str:
    """Paper input distribution: lowercase ascii, length uniform in [3, 10].

    Mirrors ``generate_feasible_input`` in the paper's ``string_data.py``.
    Note: inputs NEVER contain spaces, so ``reverse_words`` acts as the
    identity on every reachable string (no forward skill introduces spaces).
    """
    return "".join(random.choices(LOWER, k=random.randint(3, 10)))


# ======================================================================
# Forward implementations — VERBATIM from the paper's string_data.py
# ======================================================================

def repeat_str(s, n):
    """Repeat the string s exactly n times."""
    return s * n


def reverse_words(s):
    """Reverse the order of words in the string."""
    words = s.split()
    return ' '.join(reversed(words))


def add_prefix(s, pre):
    """Add a fixed prefix to the string."""
    return pre + s


def add_suffix(s, suf):
    """Add a fixed suffix to the string."""
    return s + suf


def rotate_str(s, n):
    """Rotate the string s by n positions using slicing."""
    if not s:
        return s
    n = n % len(s)
    return s[n:] + s[:n]


def mirror_str(s):
    """Append the reversed string to the original."""
    return s + s[::-1]


def insert_separator(s, sep):
    """Insert a fixed separator between every two characters."""
    return sep.join(s)


def duplicate_every_char(s):
    """Duplicate every character in the string."""
    return ''.join(ch * 2 for ch in s)


def fancy_brackets(s):
    """Enclose each character in fancy brackets."""
    return ''.join("«" + ch + "»" for ch in s)


# ======================================================================
# Trusted inverses (left inverses; total on all strings unless noted)
# ======================================================================

def repeat_str_inv(s, n):
    """Undo ``s * n`` by keeping the first len/n characters."""
    return s[: len(s) // n]


def reverse_words_inv(s):
    """``reverse_words`` is an involution on single-space word sequences.

    On the paper input distribution (no spaces) both directions are the
    identity. Globally it is only a left inverse up to whitespace
    normalization (e.g. double spaces collapse) — see module docstring.
    """
    return reverse_words(s)


def add_prefix_inv(s, pre):
    """Strip the known prefix."""
    return s[len(pre):]


def add_suffix_inv(s, suf):
    """Strip the known suffix."""
    return s[: len(s) - len(suf)]


def rotate_str_inv(s, n):
    """Rotate back by n positions (mod length)."""
    if not s:
        return s
    n = n % len(s)
    return s[len(s) - n:] + s[: len(s) - n]


def mirror_str_inv(s):
    """Keep the first half of ``s + s[::-1]``."""
    return s[: len(s) // 2]


def insert_separator_inv(s, sep):
    """Take every other character.

    NOT ``s.replace(sep, '')``: after chained separator insertions the
    payload itself contains separator characters at odd positions, which
    a replace would wrongly delete. ``s[::2]`` is the exact left inverse
    of ``sep.join(s)`` for every input string, including ones that already
    contain ``sep``.
    """
    return s[::2]


def duplicate_every_char_inv(s):
    """Take every other character of a char-doubled string."""
    return s[::2]


def fancy_brackets_inv(s):
    """Take the payload character of each «c» triple."""
    return s[1::3]


# ======================================================================
# Param samplers (paper ranges)
# ======================================================================

def _sample_repeat_n() -> int:
    return random.randint(2, 4)


def _sample_rotate_n() -> int:
    return random.randint(1, 3)


def _sample_affix() -> str:
    return "".join(random.choices(LOWER, k=random.randint(2, 4)))


def _sample_separator() -> str:
    return random.choice(['-', '_', '|'])


# ======================================================================
# Registry — ordered by paper func id. Tuple contract:
# (forward, inverse, input_sampler, param_sampler_or_None, paper_func_id)
# ======================================================================

SKILLS: dict[str, tuple[Callable, Callable, Callable[[], str], Callable[[], Any] | None, int]] = {
    "repeat_str":           (repeat_str,           repeat_str_inv,           sample_input, _sample_repeat_n,  1),
    "reverse_words":        (reverse_words,        reverse_words_inv,        sample_input, None,              4),
    "add_prefix":           (add_prefix,           add_prefix_inv,           sample_input, _sample_affix,     5),
    "add_suffix":           (add_suffix,           add_suffix_inv,           sample_input, _sample_affix,     6),
    "rotate_str":           (rotate_str,           rotate_str_inv,           sample_input, _sample_rotate_n,  8),
    "mirror_str":           (mirror_str,           mirror_str_inv,           sample_input, None,              9),
    "insert_separator":     (insert_separator,     insert_separator_inv,     sample_input, _sample_separator, 13),
    "duplicate_every_char": (duplicate_every_char, duplicate_every_char_inv, sample_input, None,              14),
    "fancy_brackets":       (fancy_brackets,       fancy_brackets_inv,       sample_input, None,              15),
}

PAPER_FUNC_ID: dict[str, int] = {name: spec[4] for name, spec in SKILLS.items()}

INT_PARAM_SKILLS: frozenset[str] = frozenset({"repeat_str", "rotate_str"})
STR_PARAM_SKILLS: frozenset[str] = frozenset({"add_prefix", "add_suffix", "insert_separator"})


def has_param(name: str) -> bool:
    """True if the skill takes a per-problem parameter."""
    if name not in SKILLS:
        raise KeyError(f"unknown skill: {name!r}")
    return SKILLS[name][3] is not None


def sample_param(name: str) -> Any:
    """Draw a parameter from the paper's range, or ``None`` for no-param skills."""
    if name not in SKILLS:
        raise KeyError(f"unknown skill: {name!r}")
    sampler = SKILLS[name][3]
    return None if sampler is None else sampler()


def coerce_param(name: str, param: Any) -> Any:
    """Normalize and validate a chain-element parameter. The single choke point.

    Accepts: native ints and numpy integers (via ``__index__``) and digit
    strings for int-param skills (Arrow/JSON round-trips can string-cast
    mixed columns); plain strings for str-param skills. Rejects: bools,
    missing params for param skills, params supplied to no-param skills,
    and anything of the wrong type.
    """
    if name not in SKILLS:
        raise KeyError(f"unknown skill: {name!r}")

    if not has_param(name):
        if param is not None:
            raise ValueError(f"{name} takes no parameter, got {param!r}")
        return None

    if param is None:
        raise ValueError(f"{name} requires a parameter, got None")

    if name in INT_PARAM_SKILLS:
        if isinstance(param, bool):
            raise TypeError(f"{name} parameter must be an int, got bool {param!r}")
        if isinstance(param, str):
            stripped = param.strip()
            if stripped.isdigit():
                return int(stripped)
            raise TypeError(f"{name} parameter must be an int, got non-numeric string {param!r}")
        try:
            return operator.index(param)
        except TypeError:
            raise TypeError(f"{name} parameter must be an int, got {type(param).__name__} {param!r}") from None

    # str-param skills
    if isinstance(param, str):
        if not param:
            raise ValueError(f"{name} parameter must be a non-empty string")
        return str(param)  # collapse str subclasses (e.g. numpy.str_)
    raise TypeError(f"{name} parameter must be a str, got {type(param).__name__} {param!r}")


def apply_forward(name: str, s: str, param: Any = None) -> str:
    """Apply one forward skill with this problem's parameter."""
    param = coerce_param(name, param)
    forward = SKILLS[name][0]
    return forward(s) if param is None else forward(s, param)


def apply_inverse(name: str, s: str, param: Any = None) -> str:
    """Apply one trusted inverse with this problem's parameter."""
    param = coerce_param(name, param)
    inverse = SKILLS[name][1]
    return inverse(s) if param is None else inverse(s, param)


def render_param_literal(name: str, param: Any) -> str:
    """Render a parameter exactly as the paper's ``random_expr`` does.

    Ints render bare (``func_1(x, 3)``); strings render single-quoted
    (``func_5(x, 'qz')``). The paper's param alphabet (lowercase letters,
    ``- _ |``) can never need escaping; we assert that invariant.
    """
    param = coerce_param(name, param)
    if param is None:
        raise ValueError(f"{name} has no parameter to render")
    if name in INT_PARAM_SKILLS:
        return str(param)
    assert "'" not in param and '"' not in param and "\\" not in param, (
        f"string param {param!r} would need escaping; outside paper alphabet"
    )
    return f"'{param}'"


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    random.seed(0)
    print(f"Total skills: {len(SKILLS)}")
    assert len(SKILLS) == 9

    N = 2000

    # 1) Per-skill round-trip + injectivity keyed by (param, output).
    for name in SKILLS:
        preimage_of: dict[tuple[Any, str], str] = {}
        for _ in range(N):
            x = sample_input()
            p = sample_param(name)
            y = apply_forward(name, x, p)
            assert apply_inverse(name, y, p) == x, (name, x, p, y)
            key = (p, y)
            if key in preimage_of:
                assert preimage_of[key] == x, f"injectivity broken: {name} {key}"
            preimage_of[key] = x
        print(f"  {name:<22} round-trip + injectivity OK over {N} samples")

    # 2) Hostile-domain totality: every skill except reverse_words has a
    #    TOTAL left inverse on arbitrary strings (spaces, separators,
    #    brackets, unicode, empty). reverse_words is excluded: it is only
    #    injective up to whitespace normalization.
    hostile = [
        "", "a", "ab c", "  spaced  ", "a-b_c|d", "««»»", "a'b\"c\\d",
        "tab\tnl\n", "日本語テスト", "x" * 50, "- _ |", "«mixed» - 'q'",
    ]
    for name in SKILLS:
        if name == "reverse_words":
            continue
        for x in hostile:
            for _ in range(5):
                p = sample_param(name)
                y = apply_forward(name, x, p)
                assert apply_inverse(name, y, p) == x, (name, x, p, y)
    print("  hostile-domain total round-trip OK (8 skills; reverse_words excluded by design)")

    # 3) Composition round-trip on random (skill, param) chains, length 1-4.
    names = list(SKILLS)
    ok = 0
    for _ in range(N):
        level = random.randint(1, 4)
        chain = [(random.choice(names),) for _ in range(level)]
        chain = [(n_, sample_param(n_)) for (n_,) in chain]
        x = sample_input()
        y = x
        for n_, p_ in chain:
            y = apply_forward(n_, y, p_)
        back = y
        for n_, p_ in reversed(chain):
            back = apply_inverse(n_, back, p_)
        assert back == x, (chain, x, y, back)
        ok += 1
    assert ok == N
    print(f"  composition round-trip OK on {N} random chains (levels 1-4): 100%")

    # 4) Param literal rendering matches the paper surface.
    examples = [
        ("repeat_str", 3, "3"),
        ("rotate_str", 2, "2"),
        ("add_prefix", "qz", "'qz'"),
        ("add_suffix", "abc", "'abc'"),
        ("insert_separator", "-", "'-'"),
    ]
    for name, p, want in examples:
        got = render_param_literal(name, p)
        assert got == want, (name, p, got, want)
        print(f"  literal: {name}(x, {got})")

    # 5) Coercion edge cases.
    assert coerce_param("repeat_str", "3") == 3
    for bad in (True, False):
        try:
            coerce_param("rotate_str", bad)
            raise AssertionError("bool accepted as int param")
        except TypeError:
            pass
    try:
        coerce_param("mirror_str", 2)
        raise AssertionError("param accepted by no-param skill")
    except ValueError:
        pass
    try:
        coerce_param("add_prefix", None)
        raise AssertionError("missing param accepted")
    except ValueError:
        pass
    print("  coerce_param edge cases OK")

    print("ALL GOOD")
