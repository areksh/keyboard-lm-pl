"""Per-line cleaning: a pure `clean_line(str) -> str | None` decision.

Returning None means "drop this line". The function is pure and side-effect-free
so it is 100% unit-testable; the CLI (02_clean_training_data.py) wraps it with
multiprocessing + checkpoint/resume.
"""

import re

from .polish import (
    expand_abbreviations,
    has_foreign_script,
    has_non_polish_latin,
    looks_polish,
)

MIN_WORDS = 3
MAX_WORDS = 60
MAX_DIGIT_RATIO = 0.30
MAX_CAPS_RATIO = 0.50

_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WS_RE = re.compile(r"\s+")

_PUNCT_NORMALIZE = {
    ord("„"): '"',
    ord("”"): '"',
    ord("“"): '"',
    ord("‟"): '"',
    ord("«"): '"',
    ord("»"): '"',
    ord("‘"): "'",
    ord("’"): "'",
    ord("‚"): "'",
    ord("–"): "-",
    ord("—"): "-",
    ord("―"): "-",
    ord("…"): "...",
}


def normalize(text: str) -> str:
    """Strip control chars, drop URLs/emails, normalize quotes/dashes, collapse
    whitespace."""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _CONTROL_RE.sub("", text)
    text = text.translate(_PUNCT_NORMALIZE)
    return _WS_RE.sub(" ", text).strip()


def clean_line(line: str) -> str | None:
    """Return a cleaned Polish line, or None to drop it."""
    text = normalize(line)
    if not text:
        return None

    if has_foreign_script(text) or has_non_polish_latin(text):
        return None

    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return None

    non_space = [c for c in text if not c.isspace()]
    digits = sum(c.isdigit() for c in non_space)
    if digits / len(non_space) > MAX_DIGIT_RATIO:
        return None

    alpha = [c for c in non_space if c.isalpha()]
    if alpha and sum(c.isupper() for c in alpha) / len(alpha) > MAX_CAPS_RATIO:
        return None

    if not looks_polish(text):
        return None

    return expand_abbreviations(text)
