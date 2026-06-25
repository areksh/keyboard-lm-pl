import itertools
import random
from collections import deque

from pl_keyboard import datamix


class ScriptedRandom(random.Random):
    # __new__ must swallow the positional arg: on Python 3.10 the C base
    # random.Random.__new__ still seeds from the first positional constructor
    # arg, so letting the list reach it raises "unhashable type: 'list'" (3.11+
    # seeds only in __init__, where we pass nothing, so it slips through there).
    def __new__(cls, randoms):
        return super().__new__(cls)

    def __init__(self, randoms):
        super().__init__()
        self._randoms = deque(randoms)

    def random(self):
        return self._randoms.popleft()


def test_cumulative_thresholds():
    assert datamix.cumulative_thresholds([1, 3]) == [0.25, 1.0]
    assert datamix.cumulative_thresholds([5]) == [1.0]
    # last entry is pinned to exactly 1.0 even when floats wouldn't sum cleanly.
    assert datamix.cumulative_thresholds([1, 1, 1])[-1] == 1.0


def test_mix_selects_sources_by_weighted_random():
    a = itertools.cycle(["a1", "a2"])
    b = itertools.cycle(["b1"])
    rng = ScriptedRandom([0.1, 0.9, 0.1])  # source 0, 1, 0
    out = list(itertools.islice(datamix.mix([a, b], [1, 1], rng), 3))
    assert out == ["a1", "b1", "a2"]


def test_mix_strips_lines():
    src = itertools.cycle(["  spaced  \n"])
    rng = ScriptedRandom([0.1])
    assert next(datamix.mix([src], [1], rng)) == "spaced"


def test_tokenize_and_chunk_yields_fixed_size_chunks():
    encode = lambda s: [int(c) for c in s]  # noqa: E731
    chunks = list(datamix.tokenize_and_chunk(["123", "456"], encode, bos=1, eos=2, context_len=4))
    assert chunks == [[1, 1, 2, 3], [2, 1, 4, 5]]
    assert all(len(c) == 4 for c in chunks)


def test_tokenize_and_chunk_skips_blank_lines():
    encode = lambda s: [int(c) for c in s]  # noqa: E731
    chunks = list(datamix.tokenize_and_chunk(["", "12"], encode, bos=1, eos=2, context_len=4))
    assert chunks == [[1, 1, 2, 2]]
