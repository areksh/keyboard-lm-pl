"""Contract test: a *real* SentencePiece model trained via our spec must satisfy
the keyboard's tokenizer contract — most importantly that <CHAR_A>..<CHAR_Z> land
at 26 consecutive ids (the native code assumes it). This validates the project's
top integration risk for real, not just against a fake mapping.

Skipped automatically where sentencepiece isn't installed (CI dev-only job).
"""

import random

import pytest

spm = pytest.importorskip("sentencepiece")

from pl_keyboard import tokenizer_spec as ts  # noqa: E402
from pl_keyboard.tokens import CHAR_TOKENS, SPECIAL_TOKENS  # noqa: E402

# A pool with Polish diacritics so character_coverage exercises them.
_WORDS = (
    "kot pies dom drzewo woda ogień ziemia niebo łąka żaba źródło ćma gęś jaźń "
    "miłość wolność radość smutek pradawny książka komputer telefon ulica miasto "
    "rzeka góra morze jezioro chmura słońce księżyc gwiazda kwiat trawa liść "
    "samochód rower pociąg samolot statek most droga ścieżka park ogród szkoła "
    "praca rodzina przyjaciel sąsiad zwierzę roślina jedzenie napój chleb masło "
    " serce dusza umysł myśl uczucie pamięć przyszłość przeszłość teraźniejszość "
    "wczoraj dzisiaj jutro poranek wieczór południe północ wschód zachód"
).split()


def _make_corpus(path, n_lines=6000, seed=0):
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n_lines):
            length = rng.randint(4, 9)
            f.write(" ".join(rng.choice(_WORDS) for _ in range(length)) + "\n")


def test_real_sentencepiece_satisfies_char_contiguity_contract(tmp_path):
    corpus = tmp_path / "corpus.txt"
    _make_corpus(corpus)
    prefix = tmp_path / "pl_keyboard"

    # Small vocab so the tiny fixture corpus can support it; everything else is
    # exactly the production spec (byte_fallback, suffix whitespace, specials).
    kwargs = ts.training_kwargs([corpus], str(prefix), vocab_size=600)
    spm.SentencePieceTrainer.train(**kwargs)

    sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")

    # The contract verifier must pass against the real piece->id lookup.
    ids = ts.verify_special_tokens(sp.piece_to_id)

    # And explicitly: every special present & non-zero, CHAR block contiguous.
    for tok in SPECIAL_TOKENS:
        assert sp.piece_to_id(tok) > 0
    base = sp.piece_to_id("<CHAR_A>")
    assert [sp.piece_to_id(t) for t in CHAR_TOKENS] == list(range(base, base + 26))
    assert ids["<CHAR_A>"] == base


def test_real_tokenizer_uses_suffix_whitespace(tmp_path):
    # treat_whitespace_as_suffix=True -> the space attaches to the *preceding*
    # token ("kot_"), which is what the keyboard's format relies on.
    corpus = tmp_path / "corpus.txt"
    _make_corpus(corpus)
    prefix = tmp_path / "pl_keyboard"
    spm.SentencePieceTrainer.train(**ts.training_kwargs([corpus], str(prefix), vocab_size=600))
    sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")

    pieces = sp.encode("kot pies", out_type=str)
    assert any(p.endswith("▁") for p in pieces)
    assert not any(p.startswith("▁") for p in pieces)
