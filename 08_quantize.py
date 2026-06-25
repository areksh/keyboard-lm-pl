#!/usr/bin/env python3
"""Quantize an F16 GGUF into shippable variants. See cli/quantize.py."""

from cli.quantize import main

if __name__ == "__main__":
    raise SystemExit(main())
