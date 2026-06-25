"""02_clean_training_data: filter raw text into cleaned Polish training lines."""

import argparse
import logging
from pathlib import Path

from cli import _runtime
from pl_keyboard import logging_setup
from pl_keyboard.cleaning import clean_line

log = logging.getLogger("pl_keyboard")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Clean raw text into Polish training lines.")
    p.add_argument("--input", nargs="+", required=True, help="Input text file(s).")
    p.add_argument("--output", required=True, help="Output file (one clean line each).")
    _runtime.add_common_args(p)
    args = p.parse_args(argv)
    _runtime.configure(args)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = seen = 0
    with out.open("w", encoding="utf-8") as fout:
        for inp in args.input:
            log.info("cleaning %s", inp)
            with open(inp, encoding="utf-8") as fin:
                for line in _runtime.progress(fin, desc=f"clean {Path(inp).name}", log=log):
                    seen += 1
                    cleaned = clean_line(line)
                    if cleaned is not None:
                        fout.write(cleaned + "\n")
                        kept += 1
                        log.log(logging_setup.DEV, "kept: %s", cleaned)

    print(f"kept {kept}/{seen} lines -> {out}")
    return 0
