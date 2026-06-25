"""Polish-language helpers for cleaning and language detection.

Deliberately the *inverse* of keyboard-lm-de's filters: we KEEP Polish diacritics
and DROP other scripts / non-Polish latin diacritics, while leaving plain ASCII
English alone (Polish keyboard text mixes English freely).
"""

import re
import string
import unicodedata

from .diacritics import POLISH_LETTERS, has_polish_diacritic

# Distinctly-Polish function words (avoids single letters that collide with
# English tokens like "i"/"a"/"to"). Presence of one strongly implies Polish.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "jest",
        "nie",
        "się",
        "że",
        "oraz",
        "który",
        "która",
        "które",
        "dla",
        "jak",
        "lub",
        "ale",
        "czy",
        "ponieważ",
        "dlatego",
        "też",
        "już",
        "tylko",
        "bardzo",
        "gdzie",
        "kiedy",
        "więc",
        "mam",
        "masz",
        "jestem",
        "na",
        "do",
        "od",
        "po",
        "za",
    }
)

# Whole-token abbreviation expansions (matched case-insensitively).
ABBREVIATIONS: dict[str, str] = {
    "np.": "na przykład",
    "tj.": "to jest",
    "tzn.": "to znaczy",
    "tzw.": "tak zwany",
    "itd.": "i tak dalej",
    "itp.": "i tym podobne",
    "m.in.": "między innymi",
    "ok.": "około",
    "godz.": "godzina",
    "ul.": "ulica",
}

_FOREIGN_SCRIPT_RE = re.compile(
    "["
    "Ͱ-Ͽ"  # Greek
    "Ѐ-ӿ"  # Cyrillic
    "԰-֏"  # Armenian
    "֐-׿"  # Hebrew
    "؀-ۿ"  # Arabic
    "ऀ-ॿ"  # Devanagari
    "぀-ヿ"  # Hiragana + Katakana
    "㐀-䶿"  # CJK Extension A
    "一-鿿"  # CJK Unified
    "가-힯"  # Hangul
    "]"
)


def has_foreign_script(text: str) -> bool:
    """True if `text` contains Cyrillic/Greek/CJK/Arabic/Hebrew/etc. characters."""
    return bool(_FOREIGN_SCRIPT_RE.search(text))


def has_non_polish_latin(text: str) -> bool:
    """True if `text` contains a latin letter with a diacritic that is NOT Polish
    (ä ö ü ß é č ž …). Plain ASCII and Polish diacritics are fine."""
    for ch in text:
        if ch in POLISH_LETTERS or ch in string.ascii_letters:
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if name.startswith("LATIN"):
            return True
    return False


def looks_polish(text: str) -> bool:
    """Cheap language gate: a Polish diacritic or a distinctly-Polish function
    word is enough to keep a line; otherwise treat it as not-Polish."""
    if has_polish_diacritic(text):
        return True
    tokens = {t.strip(string.punctuation).lower() for t in text.split()}
    return bool(tokens & FUNCTION_WORDS)


def expand_abbreviations(text: str) -> str:
    """Expand known Polish abbreviations token-by-token."""
    return " ".join(ABBREVIATIONS.get(tok.lower(), tok) for tok in text.split(" "))
