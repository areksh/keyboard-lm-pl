"""FUTO Keyboard special-token contract.

These tokens and the exact prompt format are dictated by the keyboard's native
code and must match byte-for-byte, or the model is rejected / misbehaves:

  - native/jni/.../org_futo_inputmethod_latin_xlm_LanguageModel.cpp Initialize():
        asserts <XBU>, <XBC>, <XEC> and <CHAR_A> resolve to non-zero ids, and that
        <CHAR_A>..<CHAR_Z> occupy 26 *consecutive* ids
        (LETTERS_TO_IDS[i] = LETTERS_TO_IDS[0] + i).
  - java/.../xlm/TrainingDataGenerator.kt:
        line format "{context} <XBU><CHAR_M><CHAR_O>...<XBC>truth <XEC>" — a space
        *after* truth, and NO space after <XBC>/<XEC>.
"""

import string

BEGIN_USER_INPUT = "<XBU>"  # start of what the user typed (char by char)
BEGIN_CORRECTION = "<XBC>"  # start of the corrected / intended word
END_CORRECTION = "<XEC>"  # end of the correction
SWIPE_MODE = "<XC0>"  # only if declaring feature xc0_swipe_typing_v1

# <CHAR_A> .. <CHAR_Z>, in order. MUST stay contiguous in the tokenizer vocab.
CHAR_TOKENS = [f"<CHAR_{c}>" for c in string.ascii_uppercase]

# Base set shipped in every model (autocorrect path). The CHAR tokens are one
# contiguous, ordered block.
SPECIAL_TOKENS = [BEGIN_USER_INPUT, BEGIN_CORRECTION, END_CORRECTION] + CHAR_TOKENS

# Use this list only if you also declare xc0_swipe_typing_v1.
SPECIAL_TOKENS_WITH_SWIPE = SPECIAL_TOKENS + [SWIPE_MODE]

# Characters the keyboard emits as <CHAR_*> input (base latin only).
PERMITTED_INPUT_CHARS = set(string.ascii_lowercase)

_CHAR_MAP = {c: f"<CHAR_{c.upper()}>" for c in string.ascii_lowercase}


def chars_to_tokens(ascii_chars: str) -> str:
    """'lozk' -> '<CHAR_L><CHAR_O><CHAR_Z><CHAR_K>'. Non a-z chars are dropped,
    mirroring TrainingDataGenerator.kt's mapNotNull over the letter map."""
    return "".join(_CHAR_MAP[c] for c in ascii_chars.lower() if c in _CHAR_MAP)


def format_word_correction(typed_ascii: str, truth: str) -> str:
    """Build one XBU training span.

    typed_ascii : what the user "typed" (already folded to base latin, possibly
                  noisy/partial). Only a-z survive into the CHAR tokens.
    truth       : the intended word, WITH its Polish diacritics preserved.

    Returns "" if nothing usable (mirrors formatWordMisspelling returning "").
    """
    char_tokens = chars_to_tokens(typed_ascii.strip())
    truth = truth.strip()
    if not char_tokens or not truth:
        return ""
    # Space after the word is required by the tokenizer; no space after <XBC>/<XEC>.
    return f"{BEGIN_USER_INPUT}{char_tokens}{BEGIN_CORRECTION}{truth} {END_CORRECTION}"
