from cli import check_tokenizer
from pl_keyboard.tokens import SPECIAL_TOKENS


def _mapping(**overrides):
    m = {tok: 4 + i for i, tok in enumerate(SPECIAL_TOKENS)}
    m.update(overrides)
    return m


def test_check_passes_for_valid_tokenizer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_tokenizer, "_piece_to_id", lambda path: _mapping().__getitem__)
    rc = check_tokenizer.main(["--model", str(tmp_path / "ok.model")])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_check_fails_for_broken_tokenizer(tmp_path, monkeypatch, capsys):
    # <XBU> at reserved id 0 -> verifier raises -> non-zero exit.
    monkeypatch.setattr(
        check_tokenizer, "_piece_to_id", lambda path: _mapping(**{"<XBU>": 0}).__getitem__
    )
    rc = check_tokenizer.main(["--model", str(tmp_path / "bad.model")])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().err
