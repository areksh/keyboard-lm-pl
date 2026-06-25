from cli import train_model


def _corpus(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("kot pies dom\n", encoding="utf-8")
    return f


def test_train_main_uses_tier_and_explicit_vram_autotune(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(train_model, "_train_model", lambda **kw: captured.update(kw))
    rc = train_model.main(
        [
            "--input",
            str(_corpus(tmp_path)),
            "--sp-model",
            "t.model",
            "--output-dir",
            str(tmp_path / "m"),
            "--tier",
            "medium",
            "--vram-bytes",
            str(24 * 1024**3),
        ]
    )
    assert rc == 0
    assert captured["config"]["hidden_size"] == 640  # medium tier
    assert captured["batch_size"] == 64 and captured["grad_accum"] == 4


def test_train_main_applies_arch_overrides_and_explicit_batch(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(train_model, "_train_model", lambda **kw: captured.update(kw))
    rc = train_model.main(
        [
            "--input",
            str(_corpus(tmp_path)),
            "--sp-model",
            "t.model",
            "--output-dir",
            str(tmp_path / "m"),
            "--hidden",
            "48",
            "--layers",
            "3",
            "--heads",
            "4",
            "--ffn",
            "128",
            "--batch-size",
            "8",
            "--grad-accum",
            "2",
        ]
    )
    assert rc == 0
    cfg = captured["config"]
    assert (cfg["hidden_size"], cfg["num_hidden_layers"]) == (48, 3)
    assert (cfg["num_attention_heads"], cfg["intermediate_size"]) == (4, 128)
    assert captured["batch_size"] == 8 and captured["grad_accum"] == 2


def test_train_main_detects_vram_when_unspecified(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(train_model, "_train_model", lambda **kw: captured.update(kw))
    # No --vram-bytes and no explicit batch -> calls _detect_vram (0 without GPU).
    rc = train_model.main(
        [
            "--input",
            str(_corpus(tmp_path)),
            "--sp-model",
            "t.model",
            "--output-dir",
            str(tmp_path / "m"),
            "--tier",
            "low",
        ]
    )
    assert rc == 0
    assert captured["batch_size"] == 1 and captured["grad_accum"] == 256


def test_train_main_no_inputs_returns_error(tmp_path, capsys):
    rc = train_model.main(
        [
            "--input",
            str(tmp_path / "nope.txt"),
            "--sp-model",
            "s",
            "--output-dir",
            str(tmp_path / "m"),
        ]
    )
    assert rc == 1
    assert "no input" in capsys.readouterr().err
