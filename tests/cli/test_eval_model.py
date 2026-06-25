import json
import math

import pytest

from cli import eval_model
from pl_keyboard import evaluation


def _fake_model(restore_map=None, predict_map=None, score=None):
    restore_map = restore_map or {}
    predict_map = predict_map or {}
    score = score or (lambda line: (1.0, 2))

    def loader(model_dir, sp_model):
        return (
            lambda folded: restore_map.get(folded, ""),
            lambda context: predict_map.get(context, ""),
            score,
        )

    return loader


def test_eval_missing_model_dir_returns_error(tmp_path, capsys):
    rc = eval_model.main(["--model-dir", str(tmp_path / "nope"), "--sp-model", "t.model"])
    assert rc == 1
    assert "model dir not found" in capsys.readouterr().err


def test_eval_reports_benchmark_accuracies_without_eval_file(tmp_path, monkeypatch, capsys):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    # A model that restores exactly one diacritic case and predicts one next word.
    monkeypatch.setattr(
        eval_model,
        "_load_model",
        _fake_model(restore_map={"lozko": "łóżko"}, predict_map={"Stanów": "Zjednoczonych"}),
    )

    rc = eval_model.main(["--model-dir", str(model_dir), "--sp-model", "t.model"])

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["diacritic_restoration_accuracy"] == 1 / len(evaluation.DIACRITIC_BENCHMARK)
    assert report["next_word_accuracy"] == 1 / len(evaluation.NEXT_WORD_BENCHMARK)
    assert report["perplexity"] is None


def test_eval_computes_perplexity_and_writes_report_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    eval_file = tmp_path / "held.txt"
    eval_file.write_text("ala ma kota\n\nkot ma ale\n", encoding="utf-8")
    report_path = tmp_path / "out" / "report.json"
    monkeypatch.setattr(eval_model, "_load_model", _fake_model(score=lambda line: (1.0, 2)))

    rc = eval_model.main(
        [
            "--model-dir",
            str(model_dir),
            "--sp-model",
            "t.model",
            "--eval-file",
            str(eval_file),
            "--report",
            str(report_path),
        ]
    )

    assert rc == 0
    written = json.loads(report_path.read_text(encoding="utf-8"))
    # 2 non-blank lines, each (1.0 nll, 2 tokens) -> exp(2/4).
    assert written["perplexity"] == pytest.approx(math.exp(0.5))


def test_eval_skips_perplexity_for_blank_eval_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    eval_file = tmp_path / "empty.txt"
    eval_file.write_text("\n   \n", encoding="utf-8")
    monkeypatch.setattr(eval_model, "_load_model", _fake_model())

    rc = eval_model.main(
        ["--model-dir", str(model_dir), "--sp-model", "t.model", "--eval-file", str(eval_file)]
    )
    assert rc == 0
