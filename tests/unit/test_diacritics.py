import pytest

from pl_keyboard import diacritics


@pytest.mark.parametrize(
    "src,expected",
    [
        ("łóżko", "lozko"),
        ("żółć", "zolc"),
        ("zażółć gęślą jaźń", "zazolc gesla jazn"),
        ("Łódź", "Lodz"),
        ("ĄĆĘŁŃÓŚŹŻ", "ACELNOSZZ"),
        ("ąćęłńóśźż", "acelnoszz"),
    ],
)
def test_fold_to_ascii_folds_each_diacritic_preserving_case(src, expected):
    assert diacritics.fold_to_ascii(src) == expected


def test_fold_leaves_non_polish_characters_untouched():
    # ASCII, punctuation, digits and even foreign diacritics pass through.
    assert diacritics.fold_to_ascii("hello, world 123") == "hello, world 123"
    assert diacritics.fold_to_ascii("café ü") == "café ü"


def test_fold_is_idempotent():
    text = "Zażółć gęślą jaźń, łódź i wąż"
    once = diacritics.fold_to_ascii(text)
    assert diacritics.fold_to_ascii(once) == once


def test_collisions_are_intentional():
    # ż and ź both fold to z; ó folds to o. Documented, desired behaviour.
    assert diacritics.fold_to_ascii("ż") == diacritics.fold_to_ascii("ź") == "z"
    assert diacritics.fold_to_ascii("ó") == "o"


def test_has_polish_diacritic():
    assert diacritics.has_polish_diacritic("łódź") is True
    assert diacritics.has_polish_diacritic("Ósemka") is True
    assert diacritics.has_polish_diacritic("lodz") is False
    assert diacritics.has_polish_diacritic("") is False


def test_exported_sets_are_consistent():
    # Every diacritic must have a fold entry, and fold targets are base latin a-z.
    for ch in diacritics.POLISH_DIACRITICS:
        assert ch in diacritics.DIACRITIC_FOLD
        assert diacritics.DIACRITIC_FOLD[ch].lower() in "abcdefghijklmnopqrstuvwxyz"
    # POLISH_LETTERS contains base latin plus the diacritics, both cases.
    assert "a" in diacritics.POLISH_LETTERS and "A" in diacritics.POLISH_LETTERS
    assert "ż" in diacritics.POLISH_LETTERS and "Ż" in diacritics.POLISH_LETTERS
    assert "ü" not in diacritics.POLISH_LETTERS
