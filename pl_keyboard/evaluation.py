"""Evaluation metrics + Polish benchmark cases (pure).

Model inference is injected as plain callables (`predict`/`restore`/`score`), so
metric computation is fully unit-testable; loading the real model lives in
`cli/eval_model.py:_load_model` (marked `# pragma: no cover`). The benchmarks
encode exactly the behaviour the project exists to fix: diacritic restoration
(`lozko -> łóżko`) and inflection-aware next-word (`Stanów -> Zjednoczonych`).
"""

import math
from collections.abc import Callable, Sequence

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


def accuracy(
    cases: Sequence[tuple[str, str]],
    predict: Callable[[str], str],
) -> float:
    """Fraction of cases where `predict(input)` exactly equals the expected
    output. Empty `cases` -> 0.0 (nothing to measure)."""
    if not cases:
        return 0.0
    hits = sum(1 for inp, expected in cases if predict(inp) == expected)
    return hits / len(cases)


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
