"""04b_check_tokenizer: assert a trained tokenizer satisfies the keyboard contract."""

import argparse
import sys

from pl_keyboard import tokenizer_spec


def _piece_to_id(model_path: str):  # pragma: no cover - thin sentencepiece wrapper
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=model_path)
    return sp.piece_to_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify the tokenizer's special-token contract.")
    p.add_argument("--model", required=True, help="Path to the .model file.")
    args = p.parse_args(argv)

    try:
        ids = tokenizer_spec.verify_special_tokens(_piece_to_id(args.model))
    except ValueError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"OK: {len(ids)} special tokens present, <CHAR_A> id={ids['<CHAR_A>']}")
    return 0
