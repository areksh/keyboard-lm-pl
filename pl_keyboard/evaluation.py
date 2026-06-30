"""Evaluation metrics + Polish benchmark cases (pure).

Model inference is injected as plain callables (`predict`/`restore`/`score`), so
metric computation is fully unit-testable; loading the real model lives in
`cli/eval_model.py:_load_model` (marked `# pragma: no cover`). The benchmarks
encode exactly the behaviour the project exists to fix: diacritic restoration
(`lozko -> łóżko`) and inflection-aware next-word (`Stanów -> Zjednoczonych`).
"""

import heapq
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

# (base-latin input as typed, expected diacritic-restored truth).
DIACRITIC_BENCHMARK: tuple[tuple[str, str], ...] = (
    ("lozko", "łóżko"),
    ("zazolc", "zażółć"),
    ("gesla", "gęślą"),
    ("jazn", "jaźń"),
    ("wezmy", "weźmy"),
    ("milosc", "miłość"),
    ("piesc", "pięść"),
    ("zrodlo", "źródło"),
)

# (context typed so far, expected next word) — inflected collocations a
# dictionary-only predictor cannot get right.
NEXT_WORD_BENCHMARK: tuple[tuple[str, str], ...] = (
    ("Stanów", "Zjednoczonych"),
    ("Unii", "Europejskiej"),
    ("dzień", "dobry"),
    ("do", "widzenia"),
    ("wszystkiego", "najlepszego"),
    ("na", "przykład"),
)


def run_benchmark(
    cases: Sequence[tuple[str, str]],
    predict: Callable[[str], str],
    normalize: Callable[[str], str] = lambda s: s,
) -> list[tuple[str, str, str, bool]]:
    """Per-case results `(input, expected, got, hit)`: `got = predict(input)`,
    `hit` compares got vs expected after `normalize`. Drives both `accuracy` and
    the --show-examples diagnostic, so each case is shown exactly as scored."""
    results = []
    for inp, expected in cases:
        got = predict(inp)
        results.append((inp, expected, got, normalize(got) == normalize(expected)))
    return results


def accuracy(
    cases: Sequence[tuple[str, str]],
    predict: Callable[[str], str],
    normalize: Callable[[str], str] = lambda s: s,
) -> float:
    """Fraction of cases where `predict(input)` matches the expected output after
    `normalize` is applied to both sides (identity by default). Diacritic
    restoration passes `str.lower`, since a no-context prompt makes the model
    capitalize the restored word. Empty `cases` -> 0.0 (nothing to measure)."""
    if not cases:
        return 0.0
    hits = sum(hit for *_, hit in run_benchmark(cases, predict, normalize))
    return hits / len(cases)


# ── Top-K / KSR / prefix accuracy (keyboard-quality metrics) ──────────────────
#
# These reproduce the German 11_analyze_model.py headline numbers. The tokenizer
# trains with `treat_whitespace_as_suffix=True` (tokenizer_spec.py), so a space
# attaches to the *preceding* token as a trailing boundary symbol — a "complete
# word" is exactly a piece ending in WORD_BOUNDARY. As elsewhere in this module,
# model inference is injected (logits as a plain sequence, a piece table), so the
# math is pure and unit-tested; the torch/sentencepiece glue lives in the CLI.

WORD_BOUNDARY = "▁"  # SentencePiece whitespace marker ("kot▁"), used as a suffix here.


def word_from_piece(piece: str, boundary: str = WORD_BOUNDARY) -> str | None:
    """The lowercased complete word a token piece represents, or ``None``.

    A complete word is a piece ending in `boundary` whose stem is >= 2 chars and
    alphabetic (an internal hyphen is allowed, e.g. "biało-czerwony"). Sub-word
    pieces, specials, punctuation and digits return ``None`` — they are never a
    predictable next word. Polish diacritics are preserved (we lower-case only).
    """
    if not piece.endswith(boundary):
        return None
    word = piece[: -len(boundary)].lower()
    if len(word) < 2 or not word.replace("-", "").isalpha():
        return None
    return word


def word_positions(
    pieces: Sequence[str], boundary: str = WORD_BOUNDARY
) -> Iterator[tuple[int, str]]:
    """Yield ``(pos, true_word)`` at each clean word boundary in `pieces`.

    A position qualifies when both the previous piece and the piece at `pos` are
    complete words — i.e. the model is being asked to predict a fresh next word
    after one just finished, mirroring the keyboard's next-word path. The context
    fed to the model is ``pieces[:pos]`` (so the prediction target is excluded).
    """
    for pos in range(1, len(pieces)):
        if word_from_piece(pieces[pos - 1], boundary) is None:
            continue
        true_word = word_from_piece(pieces[pos], boundary)
        if true_word is not None:
            yield pos, true_word


def topk_words_from_logits(
    logits: Sequence[float],
    pieces: Sequence[str],
    max_words: int = 5,
    scan: int = 150,
    boundary: str = WORD_BOUNDARY,
) -> list[str]:
    """Ranked complete-word predictions from a next-token logit vector.

    Scans the top `scan` logits (most pieces are sub-words/specials, so we look
    past them) and keeps the first `max_words` that are complete words, in logit
    order. `pieces` maps token id -> piece string.
    """
    top_ids = heapq.nlargest(scan, range(len(logits)), key=logits.__getitem__)
    words: list[str] = []
    for tid in top_ids:
        word = word_from_piece(pieces[tid], boundary)
        if word is not None:
            words.append(word)
            if len(words) >= max_words:
                break
    return words


@dataclass(frozen=True)
class TopKReport:
    """Cold-start next-word quality over evaluated positions."""

    positions: int
    accuracy: dict[int, float]  # k -> fraction of positions with the truth in top-k
    ksr: float  # keystroke savings rate: chars saved by top-1 / total chars
    avg_word_len: float


def evaluate_topk(
    records: Sequence[tuple[str, Sequence[str]]], ks: Sequence[int] = (1, 3, 5)
) -> TopKReport:
    """Aggregate Top-K accuracy + KSR over ``(true_word, ranked_words)`` records.

    KSR (the keyboard-LM industry metric) credits a top-1 hit with saving every
    character of the word: ``sum(len(word) for top-1 hits) / sum(len(word))``.
    Empty input yields an all-zero report rather than dividing by zero.
    """
    hits = {k: 0 for k in ks}
    saved = total_chars = 0
    total = 0
    for true_word, ranked in records:
        total += 1
        total_chars += len(true_word)
        for k in ks:
            if true_word in ranked[:k]:
                hits[k] += 1
        if ranked and ranked[0] == true_word:
            saved += len(true_word)
    if total == 0:
        return TopKReport(0, {k: 0.0 for k in ks}, 0.0, 0.0)
    return TopKReport(
        positions=total,
        accuracy={k: hits[k] / total for k in ks},
        ksr=saved / total_chars if total_chars else 0.0,
        avg_word_len=total_chars / total,
    )


def build_prefix_index(
    pieces: Sequence[str], prefix_lens: Sequence[int] = (1, 2, 3), boundary: str = WORD_BOUNDARY
) -> dict[int, dict[str, list[int]]]:
    """Map ``prefix_len -> {prefix_string -> [token ids of words with it]}``.

    Built once from the vocabulary (``pieces`` indexed by token id) and reused to
    simulate prefix-constrained prediction: the keyboard only offers candidates
    starting with what the user has typed so far.
    """
    index: dict[int, dict[str, list[int]]] = {p: {} for p in prefix_lens}
    for tid, piece in enumerate(pieces):
        word = word_from_piece(piece, boundary)
        if word is None:
            continue
        for plen in prefix_lens:
            if len(word) >= plen:
                index[plen].setdefault(word[:plen], []).append(tid)
    return index


def rank_candidates(
    candidate_ids: Sequence[int],
    pieces: Sequence[str],
    logit_of: Callable[[int], float],
    boundary: str = WORD_BOUNDARY,
) -> list[str]:
    """Words for `candidate_ids` ordered by descending logit. Candidates come
    from `build_prefix_index`, so every id maps to a complete word."""
    ranked = sorted(candidate_ids, key=logit_of, reverse=True)
    return [word_from_piece(pieces[tid], boundary) for tid in ranked]  # type: ignore[misc]


@dataclass(frozen=True)
class PrefixReport:
    """Prefix-constrained accuracy ("after N typed chars") per prefix length."""

    positions: dict[int, int]  # prefix_len -> evaluated positions
    accuracy: dict[int, dict[int, float]]  # prefix_len -> {k: fraction}


def evaluate_prefix(
    records: Sequence[tuple[str, dict[int, Sequence[str] | None]]],
    ks: Sequence[int] = (1, 3, 5),
    prefix_lens: Sequence[int] = (1, 2, 3),
) -> PrefixReport:
    """Aggregate prefix accuracy over ``(true_word, {prefix_len: ranked_words})``.

    A prefix length with no ranked candidates (word too short, or no vocab word
    shares the prefix) is recorded as ``None`` for that record and skipped, so it
    never counts against the accuracy at that length.
    """
    hits = {p: {k: 0 for k in ks} for p in prefix_lens}
    totals = {p: 0 for p in prefix_lens}
    for true_word, ranked_by_plen in records:
        for plen in prefix_lens:
            ranked = ranked_by_plen.get(plen)
            if not ranked:
                continue
            totals[plen] += 1
            for k in ks:
                if true_word in ranked[:k]:
                    hits[plen][k] += 1
    accuracy = {
        p: {k: (hits[p][k] / totals[p] if totals[p] else 0.0) for k in ks} for p in prefix_lens
    }
    return PrefixReport(positions=totals, accuracy=accuracy)


def perplexity(total_nll: float, total_tokens: int) -> float:
    """exp(mean negative log-likelihood) over a corpus. Requires >=1 token."""
    if total_tokens <= 0:
        raise ValueError("need at least one token to compute perplexity")
    return math.exp(total_nll / total_tokens)


def corpus_perplexity(
    lines: Sequence[str],
    score: Callable[[str], tuple[float, int]],
) -> float:
    """Perplexity over `lines`, where `score(line) -> (sum_nll, n_tokens)`."""
    total_nll = 0.0
    total_tokens = 0
    for line in lines:
        nll, n_tokens = score(line)
        total_nll += nll
        total_tokens += n_tokens
    return perplexity(total_nll, total_tokens)
