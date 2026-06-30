"""Top-K / KSR / prefix-accuracy metrics (pure).

These mirror the German 11_analyze_model.py semantics: a "complete word" is a
SentencePiece piece ending in the word-boundary suffix (the project trains with
`treat_whitespace_as_suffix=True`, so the space attaches as a trailing "▁"). All
the model-specific work — logits, tokenization — is injected, so the metric math
is exercised here without torch.
"""

import pytest

from pl_keyboard import evaluation

B = evaluation.WORD_BOUNDARY  # "▁"


# ── word_from_piece ───────────────────────────────────────────────────────────


def test_word_from_piece_returns_lowercased_stem_of_a_complete_word():
    assert evaluation.word_from_piece("Kot" + B) == "kot"


def test_word_from_piece_rejects_subword_pieces_without_the_boundary():
    # No trailing boundary -> the piece is mid-word, not a predictable next word.
    assert evaluation.word_from_piece("ko") is None


def test_word_from_piece_rejects_too_short_punctuation_and_digits():
    assert evaluation.word_from_piece("a" + B) is None  # single char
    assert evaluation.word_from_piece("!" + B) is None  # not alphabetic
    assert evaluation.word_from_piece("12" + B) is None  # digits
    assert evaluation.word_from_piece(B) is None  # empty stem


def test_word_from_piece_allows_internal_hyphen():
    assert evaluation.word_from_piece("biało-czerwony" + B) == "biało-czerwony"


def test_word_from_piece_keeps_polish_diacritics_in_the_truth():
    assert evaluation.word_from_piece("łóżko" + B) == "łóżko"


# ── word_positions ────────────────────────────────────────────────────────────


def test_word_positions_yields_only_clean_word_boundaries():
    # pieces: <bos>, "kot▁", "pie" (subword), "s▁"? Build a realistic mix.
    pieces = ["<s>", "kot" + B, "bardzo" + B, "ład", "ny" + B, "pies" + B]
    # Positions where BOTH the previous piece and the current piece are complete
    # words: idx2 (prev kot▁, cur bardzo▁) and idx5 (prev ny▁, cur pies▁).
    assert list(evaluation.word_positions(pieces)) == [(2, "bardzo"), (5, "pies")]


def test_word_positions_empty_for_no_boundaries():
    assert list(evaluation.word_positions(["<s>", "abc", "def"])) == []


# ── topk_words_from_logits ────────────────────────────────────────────────────


def test_topk_words_from_logits_ranks_complete_words_by_logit_desc():
    pieces = ["<s>", "kot" + B, "pies" + B, "dom" + B, "x"]
    logits = [0.0, 5.0, 9.0, 3.0, 100.0]  # "x" highest but not a complete word
    # ranked by logit desc among complete words: pies(9) > kot(5) > dom(3).
    assert evaluation.topk_words_from_logits(logits, pieces) == ["pies", "kot", "dom"]


def test_topk_words_from_logits_caps_at_max_words_and_scan_window():
    pieces = ["a" + B, "bb" + B, "cc" + B, "dd" + B, "ee" + B]
    logits = [5.0, 4.0, 3.0, 2.0, 1.0]  # "a▁" rejected (single char)
    assert evaluation.topk_words_from_logits(logits, pieces, max_words=2) == ["bb", "cc"]
    # scan window narrows how deep we look: scan=1 only sees the top logit ("a▁"),
    # which is not a word, so nothing comes back.
    assert evaluation.topk_words_from_logits(logits, pieces, scan=1) == []


# ── evaluate_topk (Top-1/3/5 + KSR) ───────────────────────────────────────────


def test_evaluate_topk_accuracy_ksr_and_avg_word_len():
    records = [
        ("kot", ["kot", "pies", "dom"]),  # top-1 hit
        ("pies", ["dom", "pies", "kot"]),  # top-3 hit, not top-1
        ("ryba", ["dom", "kot", "sok"]),  # miss
    ]
    report = evaluation.evaluate_topk(records, ks=(1, 3))
    assert report.positions == 3
    assert report.accuracy[1] == pytest.approx(1 / 3)
    assert report.accuracy[3] == pytest.approx(2 / 3)
    # KSR: only "kot" is a top-1 hit -> 3 chars saved out of 3+4+4 = 11.
    assert report.ksr == pytest.approx(3 / 11)
    assert report.avg_word_len == pytest.approx(11 / 3)


def test_evaluate_topk_empty_records_is_all_zero():
    report = evaluation.evaluate_topk([], ks=(1, 3, 5))
    assert report.positions == 0
    assert report.accuracy == {1: 0.0, 3: 0.0, 5: 0.0}
    assert report.ksr == 0.0
    assert report.avg_word_len == 0.0


def test_evaluate_topk_handles_a_position_with_no_word_predictions():
    # topk_words_from_logits can legitimately return [] (no complete word in the
    # scan window): that position is a guaranteed miss and saves nothing.
    report = evaluation.evaluate_topk([("kot", [])], ks=(1,))
    assert report.positions == 1
    assert report.accuracy[1] == 0.0
    assert report.ksr == 0.0


def test_evaluate_topk_empty_truth_word_does_not_divide_by_zero():
    # Degenerate guard: a zero-length truth contributes no characters, so KSR is
    # defined as 0.0 rather than raising.
    report = evaluation.evaluate_topk([("", ["x"])], ks=(1,))
    assert report.positions == 1
    assert report.ksr == 0.0
    assert report.avg_word_len == 0.0


# ── prefix accuracy ───────────────────────────────────────────────────────────


def test_build_prefix_index_groups_complete_words_by_prefix():
    # "do" (len 2) exercises the "word shorter than this prefix length" skip at
    # plen=3; "x" is not a complete word and is dropped entirely.
    pieces = ["kot" + B, "kotek" + B, "pies" + B, "do" + B, "x"]
    index = evaluation.build_prefix_index(pieces, prefix_lens=(1, 2, 3))
    assert index[1]["k"] == [0, 1]
    assert index[1]["p"] == [2]
    assert index[1]["d"] == [3]
    assert index[2]["ko"] == [0, 1]
    assert index[2]["do"] == [3]
    assert index[3]["kot"] == [0, 1]
    assert "do" not in index[3]  # "do" is too short for a 3-char prefix


def test_rank_candidates_orders_ids_by_logit_then_maps_to_words():
    pieces = ["kot" + B, "kotek" + B]
    logits = [1.0, 9.0]
    assert evaluation.rank_candidates([0, 1], pieces, logits.__getitem__) == ["kotek", "kot"]


def test_evaluate_prefix_per_length_accuracy_skipping_absent_lengths():
    records = [
        ("kotek", {1: ["kot", "kotek"], 2: ["kotek", "kot"]}),  # 1ch top3 hit, 2ch top1 hit
        ("pies", {1: ["pole", "pies"], 2: None}),  # 2ch skipped (no candidates)
    ]
    report = evaluation.evaluate_prefix(records, ks=(1, 3), prefix_lens=(1, 2))
    assert report.positions == {1: 2, 2: 1}
    assert report.accuracy[1][1] == pytest.approx(0.0)  # neither is top-1 at 1 char
    assert report.accuracy[1][3] == pytest.approx(1.0)  # both within top-3
    assert report.accuracy[2][1] == pytest.approx(1.0)  # "kotek" top-1 at 2 chars


def test_evaluate_prefix_zero_positions_is_zero_not_error():
    report = evaluation.evaluate_prefix([], ks=(1,), prefix_lens=(1,))
    assert report.positions == {1: 0}
    assert report.accuracy == {1: {1: 0.0}}
