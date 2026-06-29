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
