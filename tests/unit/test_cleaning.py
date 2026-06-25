from pl_keyboard import cleaning

# ── normalize ─────────────────────────────────────────────────────────────────


def test_normalize_collapses_whitespace_and_strips():
    assert cleaning.normalize("  hello   world  ") == "hello world"


def test_normalize_removes_urls_and_emails():
    assert cleaning.normalize("zobacz http://x.com/y teraz") == "zobacz teraz"
    assert cleaning.normalize("napisz a.b@c.com tu") == "napisz tu"
    assert cleaning.normalize("www.example.org koniec") == "koniec"


def test_normalize_strips_control_characters():
    assert cleaning.normalize("a\x00b\x07c") == "abc"


def test_normalize_normalizes_quotes_and_dashes():
    assert cleaning.normalize("„cześć” — tak…") == '"cześć" - tak...'
    assert cleaning.normalize("‘test’") == "'test'"


# ── clean_line: the keep path ─────────────────────────────────────────────────


def test_clean_line_keeps_and_expands_abbreviations():
    assert cleaning.clean_line("To jest np. mały kot") == "To jest na przykład mały kot"


# ── clean_line: every drop branch ─────────────────────────────────────────────


def test_clean_line_drops_empty_after_normalize():
    assert cleaning.clean_line("   ") is None
    assert cleaning.clean_line("http://only.example.com") is None


def test_clean_line_drops_foreign_script():
    assert cleaning.clean_line("Привет это jest kot") is None


def test_clean_line_drops_non_polish_latin():
    assert cleaning.clean_line("Das ist mein größer Hund hier") is None


def test_clean_line_drops_too_few_words():
    assert cleaning.clean_line("mały kot") is None


def test_clean_line_drops_too_many_words():
    assert cleaning.clean_line(" ".join(["kot"] * (cleaning.MAX_WORDS + 1))) is None


def test_clean_line_drops_high_digit_ratio():
    assert cleaning.clean_line("mam 1234567 kotów dzisiaj") is None


def test_clean_line_drops_all_caps():
    assert cleaning.clean_line("TO JEST WIELKA PROMOCJA") is None


def test_clean_line_drops_non_polish_english():
    assert cleaning.clean_line("the quick brown fox jumps") is None


def test_clean_line_handles_letterless_line_without_error():
    # no letters -> caps guard must not divide by zero -> dropped as non-Polish.
    assert cleaning.clean_line(". . . .") is None
