# 5. Twenty-six contiguous CHAR token ids

- Status: Accepted
- Date: 2026-07-06

## Context

FUTO's on-device native code maps a typed/swiped base letter to a token by
indexing off a base id: it assumes the special tokens `<CHAR_A>..<CHAR_Z>` occupy
26 *consecutive* token ids in the SentencePiece vocabulary. If the tokenizer
places them non-contiguously, the keyboard silently reads the wrong tokens and
the model appears broken or the GGUF is shown as "(Unsupported)". This is the
single highest integration risk in the project and cannot be observed from the
Python side at run time.

## Decision

Treat "the 26 `<CHAR_*>` ids are contiguous" as a hard, executable contract.
`pl_keyboard/tokenizer_spec.verify_special_tokens` asserts the contiguity (along
with the presence and ordering of `<XBU><XBC><XEC>` and the SentencePiece
settings `treat_whitespace_as_suffix` and `remove_extra_whitespaces=False`). The
contract is checked three ways: a unit test on the spec logic, a contract test
against a real trained SentencePiece model, and the `04b_check_tokenizer.py` CLI
run against `pl.model` before any training.

## Consequences

- A tokenizer that would break the keyboard fails fast, at tokenizer-build time,
  with a clear error rather than as a mysterious on-device failure.
- Other pieces of the format contract (metadata keys, feature subset, the
  `<XBU><CHAR_...><XBC>truth <XEC>` training-line format, trailing-space
  boundary) are pinned as executable tests in the same spirit; see the README's
  "hard contract" table for the full list and owning modules.
- The contract constrains tokenizer configuration: vocabulary construction must
  keep the CHAR block contiguous, which the tokenizer training step enforces.
