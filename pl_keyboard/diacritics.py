"""Polish diacritics handling.

Core idea behind the whole project: FUTO Keyboard's autocorrect/swipe path only
knows base-latin characters <CHAR_A>..<CHAR_Z>
(TrainingDataGenerator.kt: permittedCharacters = "abcdefghijklmnopqrstuvwxyz'-").
On a Polish layout the user types base letters on the main layer (diacritics via
long-press) and swipe traces base letters. So the model must learn:

    base-latin input  ->  correctly-spelled/inflected word *with* diacritics

To train that, the XBU/swipe generator folds the *input* side to base latin (this
module) while keeping the *truth* side with its diacritics intact.
"""

import string

# Lower-case fold map. ż and ź both fold to z, ó folds to o — intentional; the
# model disambiguates from context (that is the desired behaviour).
_FOLD_LOWER = {
    "ą": "a",
    "ć": "c",
    "ę": "e",
    "ł": "l",
    "ń": "n",
    "ó": "o",
    "ś": "s",
    "ź": "z",
    "ż": "z",
}

# Full map including upper-case, preserving case (Ł -> L, ł -> l).
DIACRITIC_FOLD: dict[str, str] = {}
for _lo, _base in _FOLD_LOWER.items():
    DIACRITIC_FOLD[_lo] = _base
    DIACRITIC_FOLD[_lo.upper()] = _base.upper()

POLISH_DIACRITICS: frozenset[str] = frozenset(DIACRITIC_FOLD)

# All "legitimately Polish" letters: ASCII a-z/A-Z plus the nine diacritics.
POLISH_LETTERS: frozenset[str] = frozenset(string.ascii_letters + "".join(POLISH_DIACRITICS))


def fold_to_ascii(text: str) -> str:
    """Replace Polish diacritics with their base latin letter, preserving case."""
    return "".join(DIACRITIC_FOLD.get(ch, ch) for ch in text)


def has_polish_diacritic(text: str) -> bool:
    """True if `text` contains at least one Polish diacritic letter."""
    return any(ch in POLISH_DIACRITICS for ch in text)
