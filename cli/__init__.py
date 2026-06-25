"""Thin command-line wrappers around the pure `pl_keyboard` library.

Each module exposes `main(argv)` so it is testable in-process. The numbered
scripts at the repo root (e.g. 02_clean_training_data.py) are one-line shims that
call these. Heavy dependencies (sentencepiece, torch, …) are imported lazily
inside small wrapper functions so importing a CLI module never requires them.
"""
