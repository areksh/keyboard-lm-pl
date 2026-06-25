from cli import convert_gguf


def test_convert_main_validates_then_invokes_converter(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(convert_gguf, "_convert", lambda *a: called.setdefault("args", a))
    out = tmp_path / "sub" / "pl.gguf"  # parent must be created

    rc = convert_gguf.main(
        [
            "--model-dir",
            str(tmp_path),
            "--sp-model",
            str(tmp_path / "t.model"),
            "--output",
            str(out),
        ]
    )

    assert rc == 0
    model_dir, sp_model, out_path, name, languages, features = called["args"]
    assert model_dir == str(tmp_path)
    assert languages == ["pl"]
    assert features == convert_gguf.gguf_meta.DEFAULT_FEATURES
    assert out.parent.exists()


def test_convert_main_refuses_unsupported_features(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(convert_gguf, "_convert", lambda *a: None)
    rc = convert_gguf.main(
        [
            "--model-dir",
            "m",
            "--sp-model",
            "s",
            "--output",
            str(tmp_path / "o.gguf"),
            "--features",
            "totally_made_up_feature",
        ]
    )
    assert rc == 1
    assert "refusing to write Unsupported" in capsys.readouterr().err
