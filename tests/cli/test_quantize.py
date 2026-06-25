import subprocess
from types import SimpleNamespace

from cli import quantize


def test_quantize_runs_each_quant_and_reports_paths(tmp_path, monkeypatch, capsys):
    src = tmp_path / "pl-f16.gguf"
    src.write_bytes(b"GGUF")
    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out_dir = tmp_path / "out"
    rc = quantize.main(
        ["--input", str(src), "--output-dir", str(out_dir), "--quants", "Q6_K", "Q8_0"]
    )

    assert rc == 0
    assert [c[3] for c in calls] == ["Q6_K", "Q8_0"]  # quant arg
    assert str(out_dir / "pl-Q6_K.gguf") in calls[0]  # -f16 stripped from stem
    assert "Q8_0 ->" in capsys.readouterr().out


def test_quantize_reports_failures(tmp_path, monkeypatch, capsys):
    src = tmp_path / "pl.gguf"
    src.write_bytes(b"GGUF")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd, capture_output, text: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    # no --output-dir -> defaults to the input's directory.
    rc = quantize.main(["--input", str(src), "--quants", "Q4_0"])
    assert rc == 1
    assert "FAILED boom" in capsys.readouterr().err


def test_quantize_missing_input_returns_error(tmp_path, capsys):
    rc = quantize.main(["--input", str(tmp_path / "nope.gguf")])
    assert rc == 1
    assert "input not found" in capsys.readouterr().err
