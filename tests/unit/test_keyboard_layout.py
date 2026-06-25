import random
import string
from collections import deque

from hypothesis import given
from hypothesis import strategies as st

from pl_keyboard import keyboard_layout as kl


class ScriptedRandom(random.Random):
    """Deterministic stand-in: pops scripted values; gauss returns mu (no noise)
    unless `gausses` is supplied."""

    def __init__(self, randoms=(), randranges=(), gausses=None):
        super().__init__()
        self._randoms = deque(randoms)
        self._randranges = deque(randranges)
        self._gausses = deque(gausses) if gausses is not None else None

    def random(self):
        return self._randoms.popleft()

    def randrange(self, n):
        return self._randranges.popleft() % n

    def gauss(self, mu, sigma):
        if self._gausses is not None:
            return self._gausses.popleft()
        return mu


# ── layout constants ──────────────────────────────────────────────────────────


def test_layout_constants():
    assert kl.SHIFT_KEY == ""
    assert kl.BACKSPACE_KEY == ""
    assert kl.TAP_SIZE == (80.0, 80.0)
    # all 26 letters plus the two control keys are present (28 entries).
    for c in string.ascii_lowercase:
        assert c in kl.KEYBOARD_KEYS
    assert len(kl.KEYBOARD_KEYS) == 28


def test_closest_key_returns_exact_key_at_its_position():
    assert kl.closest_key(*kl.KEYBOARD_KEYS["q"]) == "q"
    assert kl.closest_key(*kl.KEYBOARD_KEYS["m"]) == "m"
    assert kl.closest_key(*kl.KEYBOARD_KEYS[kl.SHIFT_KEY]) == kl.SHIFT_KEY


# ── substitute_keyboard_letters: all branches ─────────────────────────────────


def test_substitute_normal_backspace_and_shift_branches():
    # 'a'->q (normal), 'b'->BACKSPACE (pops the q), 'c'->SHIFT (skipped).
    g = [75.0, 106.0, 1329.0, 515.0, 113.0, 515.0]
    rng = ScriptedRandom(gausses=g)
    assert kl.substitute_keyboard_letters("abc", rng, temperature=1.0) == ""


def test_substitute_backspace_on_empty_output_is_noop():
    rng = ScriptedRandom(gausses=[1329.0, 515.0])  # 'a' lands on BACKSPACE, out empty
    assert kl.substitute_keyboard_letters("a", rng, temperature=1.0) == ""


def test_substitute_skips_non_keyboard_characters():
    rng = ScriptedRandom(gausses=[75.0, 106.0])  # only 'a' consults gauss
    assert kl.substitute_keyboard_letters("a1", rng, temperature=1.0) == "q"


def test_substitute_identity_when_gauss_returns_mean():
    assert kl.substitute_keyboard_letters("dom", ScriptedRandom(), temperature=0.0) == "dom"


# ── transpose / delete helpers: both branches ─────────────────────────────────


def test_transpose_random_too_short_returns_input():
    assert kl.transpose_random_letters("a", ScriptedRandom()) == "a"


def test_transpose_random_swaps_two_indices():
    assert kl.transpose_random_letters("abc", ScriptedRandom(randranges=[0, 2])) == "cba"


def test_transpose_random_retries_until_distinct_index():
    # first index2 equals index1 (0,0) -> loop -> then 1.
    assert kl.transpose_random_letters("abc", ScriptedRandom(randranges=[0, 0, 1])) == "bac"


def test_transpose_adjacent_too_short_returns_input():
    assert kl.transpose_adjacent_letters("a", ScriptedRandom()) == "a"


def test_transpose_adjacent_swaps_neighbours():
    assert kl.transpose_adjacent_letters("abc", ScriptedRandom(randranges=[1])) == "acb"


def test_delete_empty_returns_input():
    assert kl.delete_random_character("", ScriptedRandom()) == ""


def test_delete_removes_indexed_char():
    assert kl.delete_random_character("abc", ScriptedRandom(randranges=[1])) == "ac"


# ── misspell: scripted scenarios pin every branch ─────────────────────────────


def test_misspell_all_branches_taken():
    rng = ScriptedRandom(randoms=[0.9, 0.9, 0.9, 0.0, 0.9, 0.0], randranges=[0, 1, 2, 3])
    assert kl.misspell("dziekuje", rng, correctness=1.0) == "zdekuje"


def test_misspell_all_branches_skipped():
    rng = ScriptedRandom(randoms=[0.1, 0.1, 0.1, 0.5, 0.1])
    assert kl.misspell("dom", rng, correctness=1.0) == "dom"


def test_misspell_trim_skipped_when_word_too_short():
    rng = ScriptedRandom(randoms=[0.1, 0.1, 0.1, 0.5, 0.9])
    assert kl.misspell("a", rng, correctness=1.0) == "a"


def test_misspell_trim_taken_truncates():
    rng = ScriptedRandom(randoms=[0.1, 0.1, 0.1, 0.5, 0.9, 0.7])
    assert kl.misspell("dom", rng, correctness=1.0) == "do"


def test_misspell_strips_apostrophes_and_lowercases():
    # No mutation branches taken; just normalisation + identity substitute.
    rng = ScriptedRandom(randoms=[0.1, 0.1, 0.1, 0.0, 0.1])
    assert kl.misspell("D'Om", rng, correctness=1.0) == "dom"


def test_misspell_empty_word():
    assert kl.misspell("", ScriptedRandom(randoms=[0.1, 0.1, 0.1, 0.0, 0.1]), correctness=1.0) == ""


# ── properties (real seeded rng) ──────────────────────────────────────────────


def test_misspell_is_deterministic_with_seeded_rng():
    assert kl.misspell("dziekuje", random.Random(7)) == kl.misspell("dziekuje", random.Random(7))


@given(
    word=st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=12),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_misspell_output_is_base_latin_and_not_longer(word, seed):
    out = kl.misspell(word, random.Random(seed))
    assert all(c in string.ascii_lowercase for c in out)
    assert len(out) <= len(word)
