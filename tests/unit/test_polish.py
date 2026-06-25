import pytest

from pl_keyboard import polish


@pytest.mark.parametrize(
    "text,expected",
    [
        ("to jest kot", True),  # function word "jest"
        ("Idę do domu", True),  # "do"
        ("łódź", True),  # diacritic only
        ("the quick brown fox", False),  # English, no Polish markers
        ("", False),
    ],
)
def test_looks_polish(text, expected):
    assert polish.looks_polish(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("привет мир", True),  # Cyrillic
        ("日本語", True),  # CJK
        ("مرحبا", True),  # Arabic
        ("Ελληνικά", True),  # Greek
        ("zwykły kot", False),  # plain Polish
    ],
)
def test_has_foreign_script(text, expected):
    assert polish.has_foreign_script(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("café", True),  # é
        ("Müller", True),  # ü
        ("groß", True),  # ß
        ("naïve", True),  # ï
        ("łódź jaźń", False),  # all Polish
        ("plain ascii", False),
    ],
)
def test_has_non_polish_latin(text, expected):
    assert polish.has_non_polish_latin(text) is expected


def test_has_non_polish_latin_ignores_non_letters():
    # digits, punctuation and emoji must not be flagged as latin diacritics.
    assert polish.has_non_polish_latin("123 !? 😀") is False
    # control chars have no unicode name -> the ValueError path -> not flagged.
    assert polish.has_non_polish_latin("a\x01b") is False


def test_expand_abbreviations():
    assert polish.expand_abbreviations("np. kot itd.") == "na przykład kot i tak dalej"
    assert polish.expand_abbreviations("Mam m.in. psa") == "Mam między innymi psa"
    assert polish.expand_abbreviations("Np. duży pies") == "na przykład duży pies"
    assert polish.expand_abbreviations("bez skrótów") == "bez skrótów"
