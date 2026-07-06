# Contributing

Thanks for your interest in keyboard-lm-pl. This guide states how the project is
built. It is the going-forward bar for all new work; the rationale is recorded in
[ADR-0001](docs/adr/0001-adopt-engineering-practices.md).

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"           # pure library + full test suite, no ML deps
```

The pure `pl_keyboard/` library and the whole test suite run without the heavy ML
dependencies. Training and export need the `[train]` extra (torch, transformers,
datasets, sentencepiece, gguf) on Python 3.10-3.12:

```bash
pip install -e ".[dev,train]"
```

## The checks CI runs

Every push and pull request runs, on Python 3.10 and 3.12:

```bash
ruff check .
ruff format --check .
pytest --cov=pl_keyboard --cov=cli --cov-branch --cov-report=term-missing
```

All three must pass. Coverage is gated at **100% branch coverage**
(`fail_under = 100`); run the suite locally before opening a pull request.

## Testing: test-driven development

The project is built test-first and holds new work to that bar.

- **New code** (a new module, or new logic in an existing one): write the failing
  test first, then the minimal code to pass, then refactor with the suite green.
- **Changing existing behaviour**: adjust or add a narrow test on the span you are
  touching first, then change the code.
- **Fixing a bug**: reproduce it with a failing test that fails *because of the
  bug*, then fix until green. The regression test stays.
- Test behaviour and public API, not implementation details. Mock only true
  external boundaries; never mock the unit under test.
- Keep the heavy edges (torch, datasets, Ollama, `llama-quantize`, `llama-bench`)
  behind injectable interfaces so unit tests stay pure and fast. Make stochastic
  code testable by injecting a `random.Random` rather than seeding globals. See
  [ADR-0003](docs/adr/0003-pure-library-thin-cli-split.md).
- Genuinely GPU-only or external-boundary lines may be excluded with an explicit
  `# pragma: no cover` and a one-line justification, never silently. Do not game
  coverage with assertion-free tests or tests that re-implement the code.

Test layout (match it when adding tests):

- `tests/unit/` - pure library logic.
- `tests/contract/` - the format contract against a real SentencePiece model.
- `tests/cli/` - the thin CLI wrappers.
- `tests/integration/` - real CPU train -> convert -> eval smoke tests (need the
  `[train]` extra; they self-skip otherwise).

Test module basenames must be unique across the tree (there is no `__init__` in
`tests/`), so a CLI test uses a distinct name from its unit counterpart, e.g.
`tests/cli/test_synthetic_cli.py` vs `tests/unit/test_synthetic.py`.

## The format contract

keyboard-lm-pl targets FUTO Keyboard's exact on-device format. The "hard
contract" table in the [README](README.md) lists each requirement and the module
that pins it as an executable test. Treat these as inviolable: the 26 contiguous
`<CHAR_*>` ids ([ADR-0005](docs/adr/0005-contiguous-char-tokens.md)), the
SentencePiece settings, and the training-line format. Run
`python 04b_check_tokenizer.py --model <prefix>.model` before training.

## Commits

The project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
type(optional-scope): imperative subject, <= 72 chars

Body explaining WHY, wrapped at ~72 columns.
```

- Types in use: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`,
  `ci`, `chore`, `style`, `revert`.
- One logical change per commit; each commit leaves the tree working (lint,
  format, and the coverage gate green). Split refactors from behaviour changes.
- Stage only what belongs to the change; never commit data, models, `.gguf`s, or
  the venv (these are in `.gitignore`).
- Write in British English; ASCII only in new files.

## Documentation

Docs ship in the same change as the code. `README.md` / `README.pl.md` are the
developer manual; keep both in sync when a change makes them stale. Record an
architecturally significant decision as a new ADR under `docs/adr/`
(see its [README](docs/adr/README.md)), and add a `CHANGELOG.md` entry under
`Unreleased` for anything user- or contributor-facing.
