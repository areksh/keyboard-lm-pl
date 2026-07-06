# Architecture Decision Records

This directory records the architecturally significant decisions behind
keyboard-lm-pl: one record per decision, not per change. Records explain *why* a
choice was made and what it commits us to, so the rationale survives even when
the code moves.

ADRs are immutable by convention. Do not edit an accepted record to change its
decision; instead add a new ADR that supersedes it, and mark the old one
`Superseded by ADR-NNNN`.

## Format

Each record is `NNNN-short-kebab-title.md` with the sections: Status, Context,
Decision, Consequences, and a date. Copy an existing record as a template.

## Log

| ADR | Title | Status |
|---|---|---|
| [0001](0001-adopt-engineering-practices.md) | Adopt a shared engineering-practices standard | Accepted |
| [0002](0002-fresh-independent-pipeline.md) | A fresh pipeline, not a fork of keyboard-lm-de | Accepted |
| [0003](0003-pure-library-thin-cli-split.md) | Pure library with thin CLI wrappers | Accepted |
| [0004](0004-diacritic-fold-input-keep-truth.md) | Fold diacritics on the input, keep them in the truth | Accepted |
| [0005](0005-contiguous-char-tokens.md) | Twenty-six contiguous CHAR token ids | Accepted |
