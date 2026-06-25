"""Turn plain Polish sentences into XBU autocorrect training spans.

This is the Polish-critical step. Mirrors TrainingDataGenerator.kt's
randomlyMisspellWords, but with one essential difference: the *input* (CHAR
tokens) is folded to base latin while the *truth* keeps its diacritics — so the
model learns "lozko -> łóżko". Diacritic words are NOT skipped (the German repo
skips them, which would be fatal for Polish).
"""

import random
import string

from .diacritics import fold_to_ascii
from .keyboard_layout import misspell
from .tokens import BEGIN_CORRECTION, END_CORRECTION, format_word_correction

# Mirrors TrainingDataGenerator.kt permittedCharacters = "a-z'-" (after folding).
PERMITTED_WORD_CHARS = set(string.ascii_lowercase) | {"'", "-"}
_ASCII_LOWER = set(string.ascii_lowercase)


def is_convertible(word: str) -> bool:
    """True if `word` is worth converting to an XBU span: length >= 2, folds to
    permitted characters only, and contains at least one a-z letter."""
    w = word.strip()
    if len(w) < 2:
        return False
    folded = fold_to_ascii(w).lower()
    if not all(c in PERMITTED_WORD_CHARS for c in folded):
        return False
    return any(c in _ASCII_LOWER for c in folded)


def word_misspelling(word: str, rng: random.Random, correctness: float = 0.8) -> str:
    """One word -> "<XBU>...<XBC>truth <XEC>". Folds the typed side to base latin,
    keeps `word` (with diacritics) as the truth. Empty if nothing usable."""
    if not word.strip():
        return ""
    typed = misspell(fold_to_ascii(word), rng, correctness)
    return format_word_correction(typed, word)


def cleanup_spacing(text: str) -> str:
    """Collapse double spaces and remove the spaces a naive join leaves right
    after <XBC>/<XEC> (matches randomlyMisspellWords' trailing replaces)."""
    text = text.strip().replace("  ", " ").replace("  ", " ")
    text = text.replace(f"{BEGIN_CORRECTION} ", BEGIN_CORRECTION)
    text = text.replace(f"{END_CORRECTION} ", END_CORRECTION)
    return text


def convert_sentence(
    sentence: str,
    rng: random.Random,
    proportion: float = 0.333,
    correctness: float = 0.8,
) -> str:
    """Randomly convert ~`proportion` of the convertible words to XBU spans."""
    words = sentence.split(" ")
    chosen: list[int] = []
    for _ in range(int(len(words) * proportion)):
        remaining = [i for i in range(len(words)) if i not in chosen]
        if not remaining:
            break
        idx = remaining[rng.randrange(len(remaining))]
        if is_convertible(words[idx]):
            chosen.append(idx)

    for i in set(chosen):
        span = word_misspelling(words[i], rng, correctness)
        if span:
            words[i] = span

    return cleanup_spacing(" ".join(words))


def augment_line(
    line: str,
    rng: random.Random,
    proportion: float = 0.333,
    correctness: float = 0.8,
    copies: int = 3,
) -> list[str]:
    """Return the original line plus up to `copies` XBU-augmented variants
    (only non-empty variants that differ from the original)."""
    out = [line]
    for _ in range(copies):
        c = convert_sentence(line, rng, proportion, correctness)
        if c and c != line:
            out.append(c)
    return out
