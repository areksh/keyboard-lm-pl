"""Faithful port of FUTO Keyboard's misspelling model
(java/.../xlm/TrainingDataGenerator.kt -> WordMisspelling / QWERTYKeyboardLayout).

We reproduce it exactly so the synthetic typos we train on match the noise the
keyboard produces at inference time. Polish uses a standard QWERTY main layer
(diacritics via long-press), so these pixel positions apply unchanged.

Every stochastic function takes an injected `random.Random` for determinism.
Callers must fold Polish diacritics to base latin *before* misspelling (see
pl_keyboard.diacritics): the layout only knows a-z.
"""

import math
import random

TAP_SIZE = (80.0, 80.0)

SHIFT_KEY = ""  # matches kt: const val SHIFT_KEY = ''
BACKSPACE_KEY = ""  # matches kt: const val BACKSPACE_KEY = ''

# Pixel positions copied from QWERTYKeyboardLayout.KEYBOARD_KEYS, including SHIFT
# and BACKSPACE (a noisy tap can land on them, exactly as in the kt).
KEYBOARD_KEYS: dict[str, tuple[float, float]] = {
    "q": (75.0, 106.0),
    "w": (214.0, 106.0),
    "e": (363.0, 106.0),
    "r": (499.0, 106.0),
    "t": (645.0, 106.0),
    "y": (789.0, 106.0),
    "u": (928.0, 106.0),
    "i": (1073.0, 106.0),
    "o": (1216.0, 106.0),
    "p": (1357.0, 106.0),
    "a": (150.0, 312.0),
    "s": (291.0, 312.0),
    "d": (434.0, 312.0),
    "f": (574.0, 312.0),
    "g": (717.0, 312.0),
    "h": (859.0, 312.0),
    "j": (1005.0, 312.0),
    "k": (1140.0, 312.0),
    "l": (1288.0, 312.0),
    SHIFT_KEY: (113.0, 515.0),
    "z": (287.0, 515.0),
    "x": (434.0, 515.0),
    "c": (576.0, 515.0),
    "v": (718.0, 515.0),
    "b": (860.0, 515.0),
    "n": (1003.0, 515.0),
    "m": (1145.0, 515.0),
    BACKSPACE_KEY: (1329.0, 515.0),
}


def closest_key(x: float, y: float) -> str:
    return min(
        KEYBOARD_KEYS,
        key=lambda c: (KEYBOARD_KEYS[c][0] - x) ** 2 + (KEYBOARD_KEYS[c][1] - y) ** 2,
    )


def substitute_keyboard_letters(word: str, rng: random.Random, temperature: float = 0.6) -> str:
    out: list[str] = []
    for ch in word.lower():
        if ch not in KEYBOARD_KEYS:
            continue
        kx, ky = KEYBOARD_KEYS[ch]
        nx = rng.gauss(kx, temperature * TAP_SIZE[0])
        ny = rng.gauss(ky, temperature * TAP_SIZE[1])
        key = closest_key(nx, ny)
        if key == SHIFT_KEY:
            continue  # would uppercase the next char; irrelevant for us
        elif key == BACKSPACE_KEY:
            if out:
                out.pop()
        else:
            out.append(key)
    return "".join(out)


def transpose_random_letters(word: str, rng: random.Random) -> str:
    if len(word) < 2:
        return word
    a = rng.randrange(len(word))
    b = rng.randrange(len(word))
    while b == a:
        b = rng.randrange(len(word))
    lst = list(word)
    lst[a], lst[b] = lst[b], lst[a]
    return "".join(lst)


def transpose_adjacent_letters(word: str, rng: random.Random) -> str:
    if len(word) < 2:
        return word
    i = rng.randrange(len(word) - 1)
    lst = list(word)
    lst[i], lst[i + 1] = lst[i + 1], lst[i]
    return "".join(lst)


def delete_random_character(word: str, rng: random.Random) -> str:
    if not word:
        return word
    i = rng.randrange(len(word))
    return word[:i] + word[i + 1 :]


def misspell(word: str, rng: random.Random, correctness: float = 0.8) -> str:
    """Mirror WordMisspelling.misspellWord(). `word` should already be base latin
    (fold Polish diacritics before calling)."""
    misspelled = word.strip().lower().replace("'", "")

    def get_rand() -> float:
        return rng.random() ** correctness

    if get_rand() > 0.5:
        misspelled = transpose_random_letters(misspelled, rng)
    if get_rand() > 0.5:
        misspelled = transpose_adjacent_letters(misspelled, rng)
    if get_rand() > 0.5:
        misspelled = delete_random_character(misspelled, rng)

    misspelled = substitute_keyboard_letters(misspelled, rng, temperature=1.0 * get_rand())

    # Trim to a partial word, as if the user hasn't finished typing yet.
    if get_rand() > 0.33 and len(misspelled) >= 2:
        new_len = math.ceil((1.0 - (get_rand() ** 2)) * len(misspelled))
        new_len = max(2, min(new_len, len(misspelled)))
        misspelled = misspelled[:new_len]

    return misspelled
