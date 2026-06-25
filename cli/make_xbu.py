"""05_make_xbu_data: expand clean sentences into XBU autocorrect training lines."""

import argparse
import random
from pathlib import Path

from pl_keyboard.xbu import augment_line


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate XBU autocorrect training data.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--proportion", type=float, default=0.333, help="Fraction of words per sentence to convert."
    )
    p.add_argument(
        "--correctness",
        type=float,
        default=0.8,
        help="Typing correctness 0-1 (lower = more typos).",
    )
    p.add_argument("--copies", type=int, default=3, help="Augmented copies per sentence.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    with out.open("w", encoding="utf-8") as fout:
        for inp in args.input:
            for line in Path(inp).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                for variant in augment_line(
                    line, rng, args.proportion, args.correctness, args.copies
                ):
                    fout.write(variant + "\n")
                    n_out += 1

    print(f"{n_in} sentences -> {n_out} lines -> {out}")
    return 0
