import pytest

from pl_keyboard import tokenizer_spec as ts
from pl_keyboard.tokens import CHAR_TOKENS, SPECIAL_TOKENS


def test_training_kwargs():
    kw = ts.training_kwargs(["a.txt", "b.txt"], "out/pl_keyboard")
    assert kw["input"] == "a.txt,b.txt"
    assert kw["model_prefix"] == "out/pl_keyboard"
    assert kw["model_type"] == "bpe"
    assert kw["vocab_size"] == ts.VOCAB_SIZE - len(SPECIAL_TOKENS)
    assert kw["treat_whitespace_as_suffix"] is True
    assert kw["byte_fallback"] is True
    assert kw["add_dummy_prefix"] is False
    # A trailing space must survive normalization so it attaches as a word-final
    # "_" suffix: the keyboard predicts the next word via tokenize(context + " ")
    # and relies on that boundary. With the default (True) the space is stripped,
    # the model sees mid-word context and emits junk completions.
    assert kw["remove_extra_whitespaces"] is False
    assert kw["character_coverage"] == 0.9999
    assert kw["pad_id"] == 3
    assert kw["user_defined_symbols"] == SPECIAL_TOKENS


def _good_mapping() -> dict[str, int]:
    # 0..3 reserved (unk/bos/eos/pad); specials laid out contiguously from 4.
    return {tok: 4 + i for i, tok in enumerate(SPECIAL_TOKENS)}


def test_verify_special_tokens_accepts_contiguous_layout():
    mapping = _good_mapping()
    ids = ts.verify_special_tokens(mapping.__getitem__)
    base = ids["<CHAR_A>"]
    assert [ids[t] for t in CHAR_TOKENS] == list(range(base, base + 26))


def test_verify_rejects_token_at_reserved_id_zero():
    mapping = _good_mapping()
    mapping["<XBU>"] = 0
    with pytest.raises(ValueError, match="missing or reserved"):
        ts.verify_special_tokens(mapping.__getitem__)


def test_verify_rejects_non_contiguous_char_block():
    mapping = _good_mapping()
    mapping["<CHAR_C>"] = 999
    with pytest.raises(ValueError, match="not contiguous"):
        ts.verify_special_tokens(mapping.__getitem__)
