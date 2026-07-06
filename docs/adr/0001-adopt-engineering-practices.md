# 1. Adopt a shared engineering-practices standard

- Status: Accepted
- Date: 2026-07-06

## Context

keyboard-lm-pl is a public, contributable repository that builds a Polish
language model for FUTO Keyboard against an exact, unforgiving on-device format
contract. From the outset it has been built test-first with a 100% branch
coverage gate, Conventional Commits, and CI on two platforms (GitHub Actions and
Woodpecker/Codeberg). Those practices were in force but were nowhere written
down, and a few standard artefacts for a project of this maturity were missing:
a CHANGELOG, a contributor guide, and a decision log.

## Decision

Adopt a single, documented engineering-practices standard as the going-forward
bar for the project, and add the artefacts that make it legible to contributors:

- **Testing**: strict TDD (red -> green -> refactor). All new code is held to the
  full bar; the 100% branch coverage gate stays enforced in CI. Genuinely
  GPU-only or external-boundary lines are excluded with an explicit
  `# pragma: no cover` and a justification, never silently.
- **Version control**: Conventional Commits, small atomic commits, each leaving
  the tree working; no rewriting of pushed history.
- **Documentation**: docs ship in the same change as the code. README.md /
  README.pl.md remain the single developer manual (no separate `docs/` manual
  for a project this size); public surface is documented to that bar.
- **Decisions**: architecturally significant decisions are recorded as ADRs in
  `docs/adr/`.

The going-forward rule: new code gets full TDD; editing existing behaviour gets
a narrow characterisation test on the span being changed. Existing, untouched,
already-covered code is not retroactively re-worked.

## Consequences

- Contributors have `CONTRIBUTING.md`, this ADR log, and a `CHANGELOG.md` as the
  written statement of how the project is built.
- The coverage gate, lint, and format checks in CI are the enforced floor; a
  change is not done until code, tests, docs, and commits are consistent.
- This ADR is the log's first entry and documents the adoption itself. Four
  load-bearing decisions already made during initial development are recorded
  retrospectively as ADRs 0002-0005 so their rationale is captured; no further
  retrospective backfill is planned.
