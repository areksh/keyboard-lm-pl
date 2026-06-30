"""Release-notes markdown rendering (pure).

Turns the evaluation reports + on-disk GGUF facts into the Markdown tables a
GitHub release uses (mirroring the German release notes). All inputs are plain
data, so no model/filesystem is touched here.
"""

from pl_keyboard import release_notes
from pl_keyboard.evaluation import PrefixReport, TopKReport

# ── small formatters ──────────────────────────────────────────────────────────


def test_format_param_count_rounds_to_whole_millions():
    assert release_notes.format_param_count(136_000_000) == "136M"
    assert release_notes.format_param_count(57_300_000) == "57M"


def test_format_size_mb_uses_mebibytes_rounded():
    assert release_notes.format_size_mb(77 * 1024 * 1024) == "77 MB"


def test_format_pct_and_delta_pp():
    assert release_notes.format_pct(0.292) == "29.2%"
    assert release_notes.format_delta_pp(0.292, 0.266) == "+2.6 pp"
    assert release_notes.format_delta_pp(0.226, 0.256) == "-3.0 pp"


# ── quant detection / ordering ────────────────────────────────────────────────


def test_detect_quant_matches_most_specific_label():
    # "Q3_K_M" must win over the "Q3_K" substring it contains.
    assert release_notes.detect_quant("pl_keyboard-Q3_K_M.gguf") == "Q3_K_M"
    assert release_notes.detect_quant("pl_keyboard-Q8_0.gguf") == "Q8_0"


def test_detect_quant_unknown_is_question_mark():
    assert release_notes.detect_quant("mystery.gguf") == "?"


def test_quant_sort_key_orders_smallest_first_unknown_last():
    names = ["Q8_0", "Q3_K_M", "F16", "weird"]
    assert sorted(names, key=release_notes.quant_sort_key) == ["Q3_K_M", "Q8_0", "F16", "weird"]


# ── llama-bench JSON parsing ──────────────────────────────────────────────────


def test_parse_llama_bench_extracts_prompt_and_generation_speed():
    out = """[
      {"n_prompt": 64, "n_gen": 0, "avg_ts": 1711.0},
      {"n_prompt": 0, "n_gen": 1, "avg_ts": 562.0}
    ]"""
    assert release_notes.parse_llama_bench(out) == (1711.0, 562.0)


def test_parse_llama_bench_falls_back_to_test_name_field():
    # A trailing row matching neither "pp" nor "tg" (e.g. a metadata line) must be
    # ignored, not crash the fallback scan.
    out = (
        '[{"test": "pp64", "avg_ts": 100.0}, '
        '{"test": "tg1", "avg_ts": 50.0}, '
        '{"test": "model_size"}]'
    )
    assert release_notes.parse_llama_bench(out) == (100.0, 50.0)


def test_parse_llama_bench_returns_none_when_unparseable():
    assert release_notes.parse_llama_bench("not json") == (None, None)
    assert release_notes.parse_llama_bench("[]") == (None, None)


# ── architecture summary ──────────────────────────────────────────────────────


def test_param_count_estimates_whole_millions_from_config():
    config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": 15008,
    }
    assert release_notes.param_count(config).endswith("M")


def test_architecture_summary_reports_params_and_shape():
    config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": 15008,
    }
    summary = release_notes.architecture_summary(config)
    assert "12 layers × 768 hidden × 12 heads" in summary
    assert summary.endswith("heads)")
    assert "M parameters" in summary


# ── quality table ─────────────────────────────────────────────────────────────


def _topk():
    return TopKReport(
        positions=1000,
        accuracy={1: 0.292, 3: 0.482, 5: 0.575},
        ksr=0.256,
        avg_word_len=5.0,
    )


def test_quality_table_without_baseline_is_metric_value():
    table = release_notes.quality_table(_topk(), None, ks=(1, 3, 5))
    assert "| Metric | Value |" in table
    assert "| Top-1 accuracy | 29.2% |" in table
    assert "| KSR | 25.6% |" in table
    assert "Δ" not in table  # no comparison column without a baseline


def test_quality_table_with_baseline_adds_delta_column():
    baseline = TopKReport(
        positions=900, accuracy={1: 0.266, 3: 0.438, 5: 0.536}, ksr=0.226, avg_word_len=5.0
    )
    table = release_notes.quality_table(_topk(), baseline, ks=(1, 3, 5))
    assert "| Metric | Baseline | Current | Δ |" in table
    assert "| Top-1 accuracy | 26.6% | 29.2% | +2.6 pp |" in table
    assert "| KSR | 22.6% | 25.6% | +3.0 pp |" in table


# ── prefix table ──────────────────────────────────────────────────────────────


def _prefix():
    return PrefixReport(
        positions={1: 100, 2: 100, 3: 100},
        accuracy={
            1: {1: 0.618, 3: 0.843},
            2: {1: 0.819, 3: 0.950},
            3: {1: 0.903, 3: 0.982},
        },
    )


def test_prefix_table_singular_and_plural_char_labels():
    table = release_notes.prefix_table(_prefix(), None, prefix_lens=(1, 2, 3), ks=(1, 3))
    assert "| 1 char | 61.8% | 84.3% |" in table
    assert "| 2 chars | 81.9% | 95.0% |" in table


def test_prefix_table_with_baseline_adds_delta_columns():
    baseline = PrefixReport(
        positions={1: 100, 2: 100, 3: 100},
        accuracy={1: {1: 0.577, 3: 0.810}, 2: {1: 0.796, 3: 0.937}, 3: {1: 0.891, 3: 0.976}},
    )
    table = release_notes.prefix_table(_prefix(), baseline, prefix_lens=(1, 2, 3), ks=(1, 3))
    assert "Top-1 Δ" in table and "Top-3 Δ" in table
    assert "| 1 char | 61.8% | +4.1 pp | 84.3% | +3.3 pp |" in table


# ── speed + files tables ──────────────────────────────────────────────────────


def test_speed_table_formats_throughput_and_marks_missing():
    rows = [
        ("Q4_0", 77 * 1024 * 1024, 1711.0, 562.0),
        ("Q8_0", 139 * 1024 * 1024, None, None),
    ]
    table = release_notes.speed_table(rows)
    assert "| Q4_0 | 77 MB | 1,711 | 562 |" in table
    assert "| Q8_0 | 139 MB | — | — |" in table


def test_files_table_lists_pattern_quant_and_size():
    rows = [("Q3_K_M", 67 * 1024 * 1024), ("Q8_0", 139 * 1024 * 1024)]
    table = release_notes.files_table(rows)
    assert "| `*-Q3_K_M.gguf` | Q3_K_M | 67 MB |" in table


# ── full document ─────────────────────────────────────────────────────────────


def test_render_release_notes_assembles_all_sections():
    config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "vocab_size": 15008,
    }
    doc = release_notes.render_release_notes(
        version="v0.1.0",
        steps=35000,
        config=config,
        quality=_topk(),
        quality_baseline=None,
        prefix=_prefix(),
        prefix_baseline=None,
        gguf_rows=[("Q4_0", 77 * 1024 * 1024, 1711.0, 562.0)],
    )
    assert doc.startswith("# v0.1.0")
    assert "35,000 steps" in doc
    assert "## Quality" in doc
    assert "## Prefix accuracy" in doc
    assert "## Speed" in doc
    assert "## Files" in doc
    assert "12 layers × 768 hidden × 12 heads" in doc


def test_render_release_notes_omits_speed_table_when_no_ggufs():
    config = {
        "hidden_size": 512,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
        "vocab_size": 15008,
    }
    doc = release_notes.render_release_notes(
        version="v0.1.0",
        steps=1000,
        config=config,
        quality=_topk(),
        quality_baseline=None,
        prefix=_prefix(),
        prefix_baseline=None,
        gguf_rows=[],
    )
    assert "## Speed" not in doc
    assert "## Files" not in doc
    assert "## Quality" in doc  # metrics still render without GGUFs


def test_render_release_notes_pre_release_note_when_flagged():
    config = {
        "hidden_size": 512,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
        "vocab_size": 15008,
    }
    doc = release_notes.render_release_notes(
        version="v0.1.0-pre",
        steps=1000,
        config=config,
        quality=_topk(),
        quality_baseline=None,
        prefix=_prefix(),
        prefix_baseline=None,
        gguf_rows=[],
        pre_release=True,
    )
    assert "Pre-release" in doc
    assert "training is ongoing" in doc.lower()
