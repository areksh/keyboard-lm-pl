"""04_train_tokenizer: train the SentencePiece tokenizer from the project spec."""

import argparse
import logging
import sys
from pathlib import Path

from cli import _runtime
from pl_keyboard import tokenizer_spec

log = logging.getLogger("pl_keyboard")


def _train(kwargs: dict) -> None:  # pragma: no cover - thin sentencepiece wrapper
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(**kwargs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the SentencePiece tokenizer.")
    p.add_argument("--input", nargs="+", required=True, help="Training text file(s).")
    p.add_argument("--model-prefix", default="data/tokenizer/pl_keyboard")
    p.add_argument("--vocab-size", type=int, default=tokenizer_spec.VOCAB_SIZE)
    _runtime.add_common_args(p)
    args = p.parse_args(argv)
    _runtime.configure(args)

    inputs = [f for f in args.input if Path(f).is_file()]
    if not inputs:
        print("no input files found", file=sys.stderr)
        return 1

    Path(args.model_prefix).parent.mkdir(parents=True, exist_ok=True)
    kwargs = tokenizer_spec.training_kwargs(inputs, args.model_prefix, args.vocab_size)
    log.info("training tokenizer on %d file(s), vocab_size=%d", len(inputs), args.vocab_size)
    log.debug("sentencepiece kwargs: %s", kwargs)
    _train(kwargs)

    print(f"wrote {args.model_prefix}.model  (verify with 04b_check_tokenizer.py)")
    return 0
