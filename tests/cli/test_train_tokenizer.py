from pathlib import Path

from cli import train_tokenizer


def test_train_tokenizer_builds_spec_and_invokes_trainer(tmp_path, monkeypatch):
    corpus = tmp_path / "c.txt"
    corpus.write_text("kot pies dom\n", encoding="utf-8")
    captured = {}

    def fake_train(kwargs):
        captured["kwargs"] = kwargs
        Path(kwargs["model_prefix"] + ".model").write_text("stub")

    monkeypatch.setattr(train_tokenizer, "_train", fake_train)
    prefix = tmp_path / "tok" / "pl_keyboard"  # parent must be created

    rc = train_tokenizer.main(
        [
            "--input",
            str(corpus),
            str(tmp_path / "missing.txt"),
            "--model-prefix",
            str(prefix),
            "--vocab-size",
            "500",
        ]
    )

    assert rc == 0
    kw = captured["kwargs"]
    assert kw["vocab_size"] == 500 - 29  # spec subtracts the specials
    assert kw["model_prefix"] == str(prefix)
    assert str(corpus) in kw["input"]
    assert "missing.txt" not in kw["input"]  # non-existent input filtered out


def test_train_tokenizer_no_inputs_returns_error(tmp_path, capsys):
    rc = train_tokenizer.main(["--input", str(tmp_path / "nope.txt"), "--model-prefix", "x"])
    assert rc == 1
    assert "no input" in capsys.readouterr().err
