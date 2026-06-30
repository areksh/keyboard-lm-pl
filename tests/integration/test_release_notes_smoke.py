"""End-to-end (CPU, seconds): tiny tokenizer + random Llama -> run the release-
notes CLI for real, exercising cli/release_notes.py:_load_model (encode / piece
table / logits) and the full Top-K + prefix eval loop on a genuine model.

A random model won't score well; we only assert the harness runs and emits a
well-formed notes document and a reloadable JSON report. The llama-bench path is
not exercised here (the binary isn't available in CI), so no --gguf is passed.

Skipped where the [train] extra isn't installed (CI dev-only job).
"""

import json
import random

import pytest

torch = pytest.importorskip("torch")
spm = pytest.importorskip("sentencepiece")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from cli import release_notes  # noqa: E402
from pl_keyboard import tokenizer_spec  # noqa: E402

_WORDS = (
    "kot pies dom drzewo woda ogień ziemia niebo łąka żaba źródło ćma gęś jaźń "
    "miłość wolność radość książka komputer telefon ulica miasto rzeka góra morze "
    "samochód rower pociąg most droga park szkoła praca rodzina przyjaciel sąsiad"
).split()


def _build_model(tmp_path, arch_config):
    corpus = tmp_path / "corpus.txt"
    rng = random.Random(0)
    with corpus.open("w", encoding="utf-8") as f:
        for _ in range(4000):
            f.write(" ".join(rng.choice(_WORDS) for _ in range(rng.randint(4, 8))) + "\n")
    prefix = tmp_path / "tok"
    spm.SentencePieceTrainer.train(
        **tokenizer_spec.training_kwargs([corpus], str(prefix), vocab_size=600)
    )
    sp_model = f"{prefix}.model"
    vocab = spm.SentencePieceProcessor(model_file=sp_model).get_piece_size()

    cfg = LlamaConfig(vocab_size=vocab, **arch_config)
    model_dir = tmp_path / "hf"
    LlamaForCausalLM(cfg).save_pretrained(model_dir)
    return str(model_dir), sp_model, corpus


def test_release_notes_cli_runs_on_a_real_model(tmp_path, arch_config):
    model_dir, sp_model, corpus = _build_model(tmp_path, arch_config)
    notes_path = tmp_path / "RELEASE.md"
    report_path = tmp_path / "report.json"

    rc = release_notes.main(
        [
            "--model-dir",
            model_dir,
            "--sp-model",
            sp_model,
            "--eval-file",
            str(corpus),
            "--version",
            "v0.0.1-test",
            "--steps",
            "1000",
            "--max-sentences",
            "40",
            "--output",
            str(notes_path),
            "--report",
            str(report_path),
            "--pre-release",
        ]
    )
    assert rc == 0

    notes = notes_path.read_text(encoding="utf-8")
    assert notes.startswith("# v0.0.1-test")
    assert "1,000 steps" in notes
    assert "## Quality" in notes and "## Prefix accuracy" in notes
    assert "Pre-release" in notes
    assert "## Speed" not in notes  # no GGUFs passed

    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Every reported metric must be a valid fraction, and the report must reload
    # through the CLI's own deserializers (the baseline round-trip contract).
    assert 0.0 <= report["quality"]["accuracy"]["1"] <= 1.0
    assert 0.0 <= report["quality"]["ksr"] <= 1.0
    quality = release_notes._load_topk(report["quality"])
    prefix = release_notes._load_prefix(report["prefix"])
    assert quality.positions >= 0
    assert set(prefix.positions) == set(release_notes.PREFIX_LENS)
