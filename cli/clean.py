"""02_clean_training_data: filter raw text into cleaned Polish training lines."""

import argparse
from pathlib import Path

from pl_keyboard.cleaning import clean_line


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Clean raw text into Polish training lines.")
    p.add_argument("--input", nargs="+", required=True, help="Input text file(s).")
    p.add_argument("--output", required=True, help="Output file (one clean line each).")
    args = p.parse_args(argv)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = seen = 0
    with out.open("w", encoding="utf-8") as fout:
        for inp in args.input:
            with open(inp, encoding="utf-8") as fin:
                for line in fin:
                    seen += 1
                    cleaned = clean_line(line)
                    if cleaned is not None:
                        fout.write(cleaned + "\n")
                        kept += 1

    print(f"kept {kept}/{seen} lines -> {out}")
    return 0
