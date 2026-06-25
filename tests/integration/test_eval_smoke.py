"""End-to-end (CPU, seconds): tiny tokenizer + random Llama -> run the eval CLI
for real, exercising cli/eval_model.py:_load_model (restore/predict/score).

A random model won't *pass* the benchmarks; we only assert the harness runs and
produces well-formed metrics (accuracies in [0,1], a finite perplexity).
Architecture defaults to a tiny smoke config; pass --tier to use a real preset.

Skipped where the [train] extra isn't installed (CI dev-only job).
"""

import json
import math
import random

import pytest

torch = pytest.importorskip("torch")
spm = pytest.importorskip("sentencepiece")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from cli import eval_model  # noqa: E402
from pl_keyboard import evaluation, tokenizer_spec  # noqa: E402

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


def test_eval_cli_runs_on_a_real_model_and_emits_wellformed_metrics(tmp_path, arch_config):
    model_dir, sp_model, corpus = _build_model(tmp_path, arch_config)
    report_path = tmp_path / "report.json"

    rc = eval_model.main(
        [
            "--model-dir",
            model_dir,
            "--sp-model",
            sp_model,
            "--eval-file",
            str(corpus),
            "--report",
            str(report_path),
        ]
    )
    assert rc == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert 0.0 <= report["diacritic_restoration_accuracy"] <= 1.0
    assert 0.0 <= report["next_word_accuracy"] <= 1.0
    assert math.isfinite(report["perplexity"]) and report["perplexity"] > 0.0
    # The denominators are the real benchmark sizes (the harness scored every case).
    assert len(evaluation.DIACRITIC_BENCHMARK) > 0 and len(evaluation.NEXT_WORD_BENCHMARK) > 0
