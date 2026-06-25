from pl_keyboard import sources


def test_iter_lines_extracts_stripped_nonempty_lines():
    records = [{"text": "  Pierwsze zdanie.  "}, {"text": "Drugie."}]
    assert list(sources.iter_lines(records, "text")) == ["Pierwsze zdanie.", "Drugie."]


def test_iter_lines_splits_multiline_records_into_lines():
    records = [{"text": "Akapit jeden.\n\nAkapit dwa.\n"}]
    assert list(sources.iter_lines(records, "text")) == ["Akapit jeden.", "Akapit dwa."]


def test_iter_lines_skips_missing_and_nonstring_fields():
    records = [{"other": "x"}, {"text": None}, {"text": 123}, {"text": "ok"}]
    assert list(sources.iter_lines(records, "text")) == ["ok"]


def test_sources_registry_entries_are_streamable_specs():
    # Every registered source names the field iter_lines will read.
    for key, src in sources.SOURCES.items():
        assert src.key == key
        assert src.text_field
        assert src.path
        # The spec round-trips through iter_lines using its own declared field.
        record = {src.text_field: "zdanie testowe"}
        assert list(sources.iter_lines([record], src.text_field)) == ["zdanie testowe"]
