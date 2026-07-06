# 2. A fresh pipeline, not a fork of keyboard-lm-de

- Status: Accepted
- Date: 2026-07-06

## Context

jblechert's [keyboard-lm-de](https://github.com/jblechert/keyboard-lm-de) is the
existing reference implementation of a FUTO Keyboard language model (German).
The obvious starting point for a Polish model would be to fork it and adjust.
But the two languages diverge on the single most important axis: German's
pipeline *filters out* diacritic words, whereas for Polish diacritic restoration
(`lozko` -> `łóżko`) is the core feature, so Polish must *invert* that filter and
keep those words. Beyond that, a fork inherits a codebase with no tests and
carries its structure forward.

## Decision

Build a fresh, independent pipeline. Study keyboard-lm-de as the reference for
FUTO's format contract and overall approach, but do not copy its code. Re-derive
the pipeline test-first around the Polish requirements.

## Consequences

- The project owns its structure and can enforce its own testing standard from
  the first commit rather than inheriting an untested base.
- The Polish-critical inversion (keep diacritic words, fold on input) is a
  first-class design decision here rather than a patch over foreign assumptions
  (see ADR-0004).
- Attribution to keyboard-lm-de as the studied reference is maintained in the
  README and credits; the format contract is independently verified against the
  futo-org/android-keyboard source.
- Cost: no reuse of the German repo's working code; every pipeline stage is
  written and tested from scratch.
