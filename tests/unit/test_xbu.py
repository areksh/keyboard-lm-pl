import random

import pytest

from pl_keyboard import xbu

# ── is_convertible ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "word,expected",
    [
        ("łóżko", True),  # folds to all base latin
        ("kota", True),
        ("co-to", True),  # hyphen permitted
        ("don't", True),  # apostrophe permitted
        ("ab", True),
        ("a", False),  # too short
        ("ż", False),  # folds to "z" but length 1
        ("", False),
        ("x1", False),  # digit not permitted
        ("café", False),  # é is not Polish, not foldable
        ("--", False),  # no a-z letter
    ],
)
def test_is_convertible(word, expected):
    assert xbu.is_convertible(word) is expected


# ── word_misspelling (misspell isolated) ──────────────────────────────────────


@pytest.fixture
def identity_misspell(monkeypatch):
    # Replace the stochastic misspeller with a deterministic lower-caser so we can
    # test the fold + format wiring in isolation.
    monkeypatch.setattr(xbu, "misspell", lambda w, rng, correctness: w.lower())


def test_word_misspelling_folds_input_keeps_truth_diacritics(identity_misspell):
    out = xbu.word_misspelling("łóżko", random.Random(0))
    assert out == "<XBU><CHAR_L><CHAR_O><CHAR_Z><CHAR_K><CHAR_O><XBC>łóżko <XEC>"


def test_word_misspelling_blank_word_returns_empty():
    assert xbu.word_misspelling("   ", random.Random(0)) == ""


def test_word_misspelling_returns_empty_when_no_letters(identity_misspell):
    assert xbu.word_misspelling("--", random.Random(0)) == ""


# ── cleanup_spacing ───────────────────────────────────────────────────────────


def test_cleanup_spacing():
    assert xbu.cleanup_spacing("a  b") == "a b"
    assert xbu.cleanup_spacing("  a  ") == "a"
    assert xbu.cleanup_spacing("x<XBC> y") == "x<XBC>y"
    assert xbu.cleanup_spacing("x<XEC> y") == "x<XEC>y"


# ── convert_sentence (word_misspelling isolated) ──────────────────────────────


def test_convert_sentence_selects_convertible_words_and_guards_empty(monkeypatch):
    # stub: empty span for "ma", marker span otherwise.
    monkeypatch.setattr(
        xbu,
        "word_misspelling",
        lambda word, rng, correctness: "" if word == "ma" else f"SPAN_{word}",
    )
    rng = _ScriptedRandom(randranges=[0, 0, 0])  # pick idx0, then idx1, then idx2
    out = xbu.convert_sentence("ala ma 123", rng, proportion=1.0)
    # "ala" converted, "ma" span empty -> kept, "123" not convertible -> kept.
    assert out == "SPAN_ala ma 123"


def test_convert_sentence_breaks_when_no_indices_remain(monkeypatch):
    monkeypatch.setattr(xbu, "word_misspelling", lambda w, rng, c: f"X{w}X")
    rng = _ScriptedRandom(randranges=[0, 0])  # proportion 2.0 -> 4 iters, only 2 words
    out = xbu.convert_sentence("ab cd", rng, proportion=2.0)
    assert out == "XabX XcdX"


def test_convert_sentence_no_conversion_collapses_spaces():
    out = xbu.convert_sentence("a  b", random.Random(0), proportion=0.0)
    assert out == "a b"


def test_convert_sentence_applies_token_spacing_cleanup(monkeypatch):
    monkeypatch.setattr(
        xbu,
        "word_misspelling",
        lambda word, rng, c: "<XBU>x<XBC>dom <XEC>",
    )
    rng = _ScriptedRandom(randranges=[0])
    out = xbu.convert_sentence("kot tu", rng, proportion=0.5)
    # the space the join adds after <XEC> is removed by cleanup_spacing.
    assert "<XEC>tu" in out


# ── augment_line ──────────────────────────────────────────────────────────────


def test_augment_line_keeps_original_and_valid_distinct_copies(monkeypatch):
    calls = iter(["", "konwersja", "oryginał"])
    monkeypatch.setattr(xbu, "convert_sentence", lambda *a, **k: next(calls))
    out = xbu.augment_line("oryginał", random.Random(0), copies=3)
    # original always first; "" dropped (falsy); "konwersja" kept; "oryginał"==line dropped.
    assert out == ["oryginał", "konwersja"]


# ── determinism ───────────────────────────────────────────────────────────────


def test_convert_sentence_is_deterministic_with_seeded_rng():
    a = xbu.convert_sentence("zażółć gęślą jaźń kot pies", random.Random(3))
    b = xbu.convert_sentence("zażółć gęślą jaźń kot pies", random.Random(3))
    assert a == b


# ── helpers ───────────────────────────────────────────────────────────────────


class _ScriptedRandom(random.Random):
    def __init__(self, randranges=()):
        super().__init__()
        from collections import deque

        self._randranges = deque(randranges)

    def randrange(self, n):
        return self._randranges.popleft() % n
