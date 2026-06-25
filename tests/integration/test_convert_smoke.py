"""End-to-end (CPU, seconds): tiny tokenizer + random Llama -> convert -> read
the GGUF back and assert it is exactly what the keyboard would accept.

Architecture defaults to a tiny smoke config; pass --tier {low,medium,high} to
run against the real arch preset (see tests/integration/conftest.py).

Skipped where the [train] extra isn't installed (CI dev-only job).
"""

import random

import pytest

torch = pytest.importorskip("torch")
spm = pytest.importorskip("sentencepiece")
gguf = pytest.importorskip("gguf")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from cli import convert_gguf, train_model  # noqa: E402
from pl_keyboard import gguf_meta, tokenizer_spec  # noqa: E402

_WORDS = (
    "kot pies dom drzewo woda ogień ziemia niebo łąka żaba źródło ćma gęś jaźń "
    "miłość wolność radość smutek pradawny książka komputer telefon ulica miasto "
    "rzeka góra morze jezioro chmura słońce księżyc gwiazda kwiat trawa liść "
    "samochód rower pociąg samolot statek most droga ścieżka park ogród szkoła "
    "praca rodzina przyjaciel sąsiad zwierzę roślina jedzenie napój chleb masło "
    "serce dusza umysł myśl uczucie pamięć przyszłość przeszłość teraźniejszość "
    "wczoraj dzisiaj jutro poranek wieczór południe północ wschód zachód"
).split()


def _train_tiny_tokenizer(tmp_path):
    corpus = tmp_path / "corpus.txt"
    rng = random.Random(0)
    with corpus.open("w", encoding="utf-8") as f:
        for _ in range(4000):
            f.write(" ".join(rng.choice(_WORDS) for _ in range(rng.randint(4, 8))) + "\n")
    prefix = tmp_path / "tok"
    spm.SentencePieceTrainer.train(
        **tokenizer_spec.training_kwargs([corpus], str(prefix), vocab_size=600)
    )
    return f"{prefix}.model", corpus


def _read_str(reader, name):
    field = reader.fields[name]
    return bytes(field.parts[field.data[-1]]).decode("utf-8")


def test_convert_produces_keyboard_valid_gguf(tmp_path, arch_config):
    sp_model, _ = _train_tiny_tokenizer(tmp_path)
    vocab = spm.SentencePieceProcessor(model_file=sp_model).get_piece_size()

    cfg = LlamaConfig(vocab_size=vocab, **arch_config)
    model_dir = tmp_path / "hf"
    LlamaForCausalLM(cfg).save_pretrained(model_dir)

    out = tmp_path / "pl.gguf"
    rc = convert_gguf.main(
        ["--model-dir", str(model_dir), "--sp-model", sp_model, "--output", str(out)]
    )
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0

    reader = gguf.GGUFReader(str(out))
    assert _read_str(reader, "keyboardlm.languages") == "pl"
    assert _read_str(reader, "keyboardlm.ext_tokenizer_type") == "sentencepiece"
    features = _read_str(reader, "keyboardlm.features")
    gguf_meta.validate(["pl"], features)  # would raise if the keyboard would reject it
    assert "keyboardlm.ext_tokenizer_data" in reader.fields


def test_train_then_convert_produces_keyboard_valid_gguf(tmp_path, arch_cli_args):
    # The capstone: a real (small) training run -> convert -> keyboard-valid GGUF.
    sp_model, corpus = _train_tiny_tokenizer(tmp_path)
    model_dir = tmp_path / "trained"

    rc = train_model.main(
        [
            "--input",
            str(corpus),
            "--sp-model",
            sp_model,
            "--output-dir",
            str(model_dir),
            *arch_cli_args,
            "--batch-size",
            "2",
            "--grad-accum",
            "1",
            "--steps",
            "2",
        ]
    )
    assert rc == 0
    assert (model_dir / "config.json").exists()

    out = tmp_path / "trained.gguf"
    rc2 = convert_gguf.main(
        ["--model-dir", str(model_dir), "--sp-model", sp_model, "--output", str(out)]
    )
    assert rc2 == 0 and out.exists()
    reader = gguf.GGUFReader(str(out))
    assert _read_str(reader, "keyboardlm.languages") == "pl"
    gguf_meta.validate(["pl"], _read_str(reader, "keyboardlm.features"))
