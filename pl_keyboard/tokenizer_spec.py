"""SentencePiece training spec + the special-token contract verifier.

Both are pure: `training_kwargs` just builds the kwargs dict (the CLI passes it to
spm.SentencePieceTrainer.train), and `verify_special_tokens` takes a piece->id
lookup, so it is tested with plain dicts — no sentencepiece dependency here.
"""

from collections.abc import Callable, Sequence
from os import PathLike

from .tokens import CHAR_TOKENS, SPECIAL_TOKENS

VOCAB_SIZE = 15008  # 3 control + 26 CHAR specials + learned pieces; matches English


def training_kwargs(
    input_paths: Sequence[str | PathLike],
    model_prefix: str,
    vocab_size: int = VOCAB_SIZE,
) -> dict:
    """Kwargs for spm.SentencePieceTrainer.train that match FUTO's format.

    The special tokens are reserved via `user_defined_symbols` (added in order, so
    the CHAR block stays contiguous), hence the learned vocab is reduced by their
    count.
    """
    return {
        "input": ",".join(str(p) for p in input_paths),
        "model_prefix": model_prefix,
        "vocab_size": vocab_size - len(SPECIAL_TOKENS),
        "model_type": "bpe",
        "treat_whitespace_as_suffix": True,  # spaces as suffix ("wort_")
        # Keep a trailing space so it attaches as the word-final "_" suffix. The
        # keyboard predicts the next word via tokenize(context + " ") and depends
        # on that boundary token; the SentencePiece default (True) strips the
        # trailing space, leaving the model in a mid-word state that yields junk
        # completions instead of next-word predictions.
        "remove_extra_whitespaces": False,
        "character_coverage": 0.9999,
        "byte_fallback": True,
        "add_dummy_prefix": False,
        "input_sentence_size": 2_000_000,
        "shuffle_input_sentence": True,
        "user_defined_symbols": SPECIAL_TOKENS,
        "pad_id": 3,
    }


def verify_special_tokens(piece_to_id: Callable[[str], int]) -> dict[str, int]:
    """Assert the keyboard's tokenizer contract (see tokens.py): every special
    token has a non-zero id, and <CHAR_A>..<CHAR_Z> are 26 consecutive ids.
    Returns the {token: id} map. Raises ValueError on any violation."""
    ids: dict[str, int] = {}
    for tok in SPECIAL_TOKENS:
        i = piece_to_id(tok)
        if i <= 0:
            raise ValueError(f"special token {tok!r} missing or reserved (id={i})")
        ids[tok] = i

    base = ids["<CHAR_A>"]
    for offset, tok in enumerate(CHAR_TOKENS):
        if ids[tok] != base + offset:
            raise ValueError(
                f"CHAR tokens not contiguous at {tok!r}: {ids[tok]} != {base + offset}"
            )
    return ids
