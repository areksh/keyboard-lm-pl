"""Weighted source mixing and tokenize/chunk — pure streaming helpers.

The training CLI provides infinite per-file line iterators (re-reading each file)
and a real tokenizer's `encode`; here we only do the weighting and chunking, so
everything is testable with in-memory iterables and a fake encoder.
"""

import bisect
import random
from collections.abc import Callable, Iterable, Iterator, Sequence


def cumulative_thresholds(weights: Sequence[float]) -> list[float]:
    """Normalized cumulative weights, e.g. [1, 3] -> [0.25, 1.0]. The last entry
    is pinned to exactly 1.0 so weighted selection can never fall off the end."""
    total = sum(weights)
    acc = 0.0
    out: list[float] = []
    for w in weights:
        acc += w / total
        out.append(acc)
    out[-1] = 1.0
    return out


def mix(
    iterators: Sequence[Iterator[str]],
    weights: Sequence[float],
    rng: random.Random,
) -> Iterator[str]:
    """Infinite weighted-random mix of `iterators` (each assumed infinite)."""
    thresholds = cumulative_thresholds(weights)
    while True:
        i = bisect.bisect_right(thresholds, rng.random())
        yield next(iterators[i]).strip()


def tokenize_and_chunk(
    lines: Iterable[str],
    encode: Callable[[str], list[int]],
    bos: int,
    eos: int,
    context_len: int,
) -> Iterator[list[int]]:
    """Pack `[bos] + encode(line) + [eos]` tokens into fixed `context_len` chunks.
    A trailing partial buffer is held back (never yielded short)."""
    buf: list[int] = []
    for line in lines:
        if not line:
            continue
        buf += [bos, *encode(line), eos]
        while len(buf) >= context_len:
            yield buf[:context_len]
            buf = buf[context_len:]
