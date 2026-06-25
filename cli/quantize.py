"""08_quantize: produce quantized GGUF variants from an F16 model via llama-quantize."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from cli import _runtime

DEFAULT_QUANTS = ["Q3_K_M", "Q4_0", "Q6_K", "Q8_0"]  # ultra-low -> ultra

log = logging.getLogger("pl_keyboard")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quantize an F16 GGUF into shippable variants.")
    p.add_argument("--input", required=True, help="F16 .gguf to quantize.")
    p.add_argument("--output-dir", default=None, help="Defaults to the input's directory.")
    p.add_argument("--quants", nargs="+", default=DEFAULT_QUANTS)
    p.add_argument("--llama-quantize", default="llama-quantize", help="Path to the binary.")
    _runtime.add_common_args(p)
    args = p.parse_args(argv)
    _runtime.configure(args)

    src = Path(args.input)
    if not src.is_file():
        print(f"input not found: {src}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem.removesuffix("-f16").removesuffix("-F16")
    log.info("quantizing %s into %d variant(s): %s", src, len(args.quants), ", ".join(args.quants))

    failures = 0
    for quant in _runtime.progress(args.quants, desc="quantize", log=log, unit="variant"):
        dst = out_dir / f"{stem}-{quant}.gguf"
        log.debug("running %s -> %s", quant, dst)
        result = subprocess.run(
            [args.llama_quantize, str(src), str(dst), quant], capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"{quant}: FAILED {result.stderr.strip()}", file=sys.stderr)
            failures += 1
        else:
            print(f"{quant} -> {dst}")

    return 1 if failures else 0
