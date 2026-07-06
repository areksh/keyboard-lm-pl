# 4. Fold diacritics on the input, keep them in the truth

- Status: Accepted
- Date: 2026-07-06

## Context

FUTO's autocorrect/swipe path only knows base-latin characters:
`TrainingDataGenerator.kt` sets `permittedCharacters = "a-z'-"`, so the model's
input alphabet is `<CHAR_A>..<CHAR_Z>` only. On a Polish layout the user types
base letters (diacritics are entered by long-press) and swipe traces are over
base letters. Yet the value of the model is producing correctly accented Polish
(`lozko` -> `łóżko`, choosing the right inflection). The training signal must
therefore connect base-latin input to accented output.

## Decision

Train with the input side **diacritic-folded** to base latin and the truth side
**kept with diacritics**:

```
input  (CHAR tokens, folded):  l o z k o
truth  (kept with diacritics): łóżko
```

The fold map is `ą->a ć->c ę->e ł->l ń->n ó->o ś->s ź->z ż->z`
(`pl_keyboard/diacritics.py`). Words containing diacritics are **never skipped**
(the German reference repo skips them, which would be fatal for Polish). The
augmentation is applied in `pl_keyboard/xbu.py`.

## Consequences

- The model learns diacritic restoration and inflection from base-latin typing,
  which is the project's core feature.
- This is the concrete inversion of keyboard-lm-de's diacritic filter that
  motivated a fresh pipeline (see ADR-0002).
- Restoration is scored case-insensitively at eval time, because a no-context
  prompt makes the model emit a sentence-initial capital (`lozko` -> `Łóżko`),
  which still restores the diacritics correctly.
- The fold map and the "never skip diacritic words" rule are correctness-critical
  and covered by unit tests; changing either changes the training signal.
