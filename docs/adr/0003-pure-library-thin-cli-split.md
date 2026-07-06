# 3. Pure library with thin CLI wrappers

- Status: Accepted
- Date: 2026-07-06

## Context

The pipeline mixes format-critical, deterministic logic (tokenizer contract,
diacritic folding, XBU augmentation, GGUF metadata) with heavy, slow, or
external operations (HuggingFace `datasets` streaming, Ollama HTTP, GPU
training, `llama-quantize`, `llama-bench`). To hold the project to strict TDD
with a 100% coverage gate, the testable logic must be reachable without importing
`torch` or touching a GPU, a network, or an external binary.

## Decision

Split the code in two:

- **`pl_keyboard/`** is a pure, dependency-light library (no `torch`/`datasets`
  at import time). All format-critical and Polish logic lives here and is
  exhaustively unit-tested.
- **CLIs `01..10`** are thin wrappers (`cli/` + numbered shims) doing only
  argument parsing and I/O, delegating every decision to the library.
- **Heavy edges** sit behind injectable interfaces that tests fake, with a small
  number of real CPU end-to-end smoke tests exercising the wiring.

## Consequences

- The unit suite runs fast and needs only the `[dev]` extra; the `[train]` extra
  (torch, transformers, datasets, sentencepiece, gguf) is required only for
  actual training/export and the tier smoke tests.
- The 100% branch-coverage gate is honest: pure logic is covered by real tests,
  not by mocking the unit under test. GPU-only lines are `# pragma: no cover`
  with justification.
- Testing stochastic code means injecting a `random.Random` rather than seeding
  globals; heavy dependencies are injected rather than imported at module top.
- Cost: an indirection layer (library + wrapper) instead of monolithic scripts,
  and discipline to keep heavy imports out of `pl_keyboard`.
