import string

from pl_keyboard import tokens


def test_special_token_constants():
    assert tokens.BEGIN_USER_INPUT == "<XBU>"
    assert tokens.BEGIN_CORRECTION == "<XBC>"
    assert tokens.END_CORRECTION == "<XEC>"
    assert tokens.SWIPE_MODE == "<XC0>"


def test_char_tokens_are_26_in_order():
    assert tokens.CHAR_TOKENS == [f"<CHAR_{c}>" for c in string.ascii_uppercase]
    assert len(tokens.CHAR_TOKENS) == 26


def test_special_tokens_block_layout():
    # 3 control tokens then the 26 CHAR tokens, contiguous and ordered.
    assert tokens.SPECIAL_TOKENS == ["<XBU>", "<XBC>", "<XEC>", *tokens.CHAR_TOKENS]
    assert len(tokens.SPECIAL_TOKENS) == 29
    # swipe variant only appends <XC0> (keeps the CHAR block contiguous).
    assert tokens.SPECIAL_TOKENS_WITH_SWIPE == [*tokens.SPECIAL_TOKENS, "<XC0>"]


def test_permitted_input_chars_is_base_latin():
    assert tokens.PERMITTED_INPUT_CHARS == set(string.ascii_lowercase)


def test_chars_to_tokens_maps_and_filters():
    assert tokens.chars_to_tokens("lozk") == "<CHAR_L><CHAR_O><CHAR_Z><CHAR_K>"
    # uppercases are lowered; non a-z (incl. diacritics, digits, punct) dropped.
    assert tokens.chars_to_tokens("Ab3-ż") == "<CHAR_A><CHAR_B>"
    assert tokens.chars_to_tokens("") == ""
    assert tokens.chars_to_tokens("123") == ""


def test_format_word_correction_exact_contract_string():
    # space AFTER truth, NO space after <XBC>/<XEC> (TrainingDataGenerator.kt).
    assert (
        tokens.format_word_correction("lozk", "łóżko")
        == "<XBU><CHAR_L><CHAR_O><CHAR_Z><CHAR_K><XBC>łóżko <XEC>"
    )


def test_format_word_correction_strips_truth_and_typed():
    assert (
        tokens.format_word_correction("  lo  ", "  dom  ") == "<XBU><CHAR_L><CHAR_O><XBC>dom <XEC>"
    )


def test_format_word_correction_empty_when_unusable():
    assert tokens.format_word_correction("", "dom") == ""
    assert tokens.format_word_correction("123", "dom") == ""  # no a-z chars
    assert tokens.format_word_correction("lo", "") == ""  # blank truth
    assert tokens.format_word_correction("lo", "   ") == ""
