import pytest

from pl_keyboard import gguf_meta as gm


def test_keyboard_metadata_default():
    md = gm.keyboard_metadata(["pl"])
    assert md["keyboardlm.languages"] == "pl"
    assert md["keyboardlm.features"] == "xbu_char_autocorrect_v1 char_embed_mixing_v1"
    assert md["keyboardlm.ext_tokenizer_type"] == "sentencepiece"


def test_keyboard_metadata_multilingual_and_custom_features():
    md = gm.keyboard_metadata(["pl", "en"], features="base_v1")
    assert md["keyboardlm.languages"] == "pl en"
    assert md["keyboardlm.features"] == "base_v1"


def test_unsupported_features_filters_known_and_prefixes():
    assert gm.unsupported_features(["base_v1", "xbu_char_autocorrect_v1"]) == []
    # opt_* and _* are explicitly tolerated by ModelPaths.kt; "foo" is not.
    assert gm.unsupported_features(["foo", "opt_bar", "_baz", "char_embed_mixing_v1"]) == ["foo"]


@pytest.mark.parametrize(
    "features,tok,langs,expected",
    [
        (["base_v1"], "sentencepiece", ["pl"], False),
        ([], "sentencepiece", ["pl"], True),  # no features
        (["base_v1"], "None", ["pl"], True),  # no tokenizer
        (["base_v1"], "sentencepiece", [], True),  # no languages
    ],
)
def test_is_unsupported(features, tok, langs, expected):
    assert gm.is_unsupported(features, tok, langs) is expected


def test_validate_accepts_default():
    gm.validate(["pl"], gm.DEFAULT_FEATURES)  # no raise


def test_validate_rejects_unknown_feature():
    with pytest.raises(ValueError, match="unsupported features"):
        gm.validate(["pl"], "foo char_embed_mixing_v1")


def test_validate_rejects_empty_languages():
    with pytest.raises(ValueError, match="Unsupported"):
        gm.validate([], gm.DEFAULT_FEATURES)


def test_validate_rejects_missing_tokenizer():
    with pytest.raises(ValueError, match="Unsupported"):
        gm.validate(["pl"], gm.DEFAULT_FEATURES, tokenizer_type="None")
