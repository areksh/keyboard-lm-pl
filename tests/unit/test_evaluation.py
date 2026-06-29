import math

import pytest

from pl_keyboard import evaluation
from pl_keyboard.diacritics import fold_to_ascii


def test_accuracy_is_fraction_of_exact_matches():
    cases = [("a", "x"), ("b", "y"), ("c", "z")]
    predict = {"a": "x", "b": "WRONG", "c": "z"}.__getitem__
    assert evaluation.accuracy(cases, predict) == pytest.approx(2 / 3)


def test_accuracy_empty_cases_is_zero():
    assert evaluation.accuracy([], lambda s: s) == 0.0


def test_accuracy_applies_normalize_to_both_sides():
    # Diacritic restoration is scored case-insensitively: a no-context XBU prompt
    # makes the model emit a sentence-initial capital ("Łóżko"), which still
    # restores the diacritics correctly. normalize folds that away on both sides.
    cases = [("lozko", "łóżko")]
    predict = {"lozko": "Łóżko"}.__getitem__
    assert evaluation.accuracy(cases, predict) == 0.0  # case-sensitive default misses
    assert evaluation.accuracy(cases, predict, normalize=str.lower) == 1.0


def test_run_benchmark_returns_per_case_results_with_hit_flag():
    cases = [("a", "X"), ("b", "y")]
    predict = {"a": "x", "b": "y"}.__getitem__
    # case-insensitive normalize: "x" counts as a hit for "X".
    assert evaluation.run_benchmark(cases, predict, normalize=str.lower) == [
        ("a", "X", "x", True),
        ("b", "y", "y", True),
    ]
    # case-sensitive default: "x" != "X".
    assert evaluation.run_benchmark(cases, predict)[0] == ("a", "X", "x", False)


def test_perplexity_is_exp_of_mean_nll():
    assert evaluation.perplexity(0.0, 5) == pytest.approx(1.0)
    assert evaluation.perplexity(2.0, 2) == pytest.approx(math.e)


def test_perplexity_requires_at_least_one_token():
    with pytest.raises(ValueError, match="at least one token"):
        evaluation.perplexity(0.0, 0)


def test_corpus_perplexity_aggregates_injected_scores():
    # score returns (sum_nll, n_tokens) per line; perplexity over the totals.
    scores = {"ala": (1.0, 2), "kot": (3.0, 2)}
    ppl = evaluation.corpus_perplexity(["ala", "kot"], scores.__getitem__)
    assert ppl == pytest.approx(math.exp(4.0 / 4))


def test_benchmarks_are_input_truth_pairs_with_the_flagship_cases():
    # The cases the project exists to fix must be present.
    assert ("lozko", "łóżko") in evaluation.DIACRITIC_BENCHMARK
    assert ("Stanów", "Zjednoczonych") in evaluation.NEXT_WORD_BENCHMARK
    for inp, truth in (*evaluation.DIACRITIC_BENCHMARK, *evaluation.NEXT_WORD_BENCHMARK):
        assert inp and truth and inp != truth


def test_diacritic_benchmark_inputs_are_the_ascii_fold_of_their_truth():
    # The diacritic benchmark proves "user types base latin -> model restores
    # diacritics", so each input MUST be exactly fold_to_ascii(truth). A typo'd
    # input that isn't the fold would silently benchmark the wrong restoration.
    for typed, truth in evaluation.DIACRITIC_BENCHMARK:
        assert fold_to_ascii(truth).lower() == typed
        # ...and the truth must actually carry diacritics (else nothing to restore).
        assert truth != typed
