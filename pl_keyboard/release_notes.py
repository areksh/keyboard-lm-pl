"""Render GitHub-release notes (Markdown) from the evaluation reports + GGUFs.

Pure: it formats already-computed numbers (the `TopKReport`/`PrefixReport` from
`evaluation`, file sizes, and parsed `llama-bench` output) into the same tables
the German model publishes per release. The CLI gathers the inputs (running the
model and `llama-bench`); everything here is plain data in, Markdown string out,
so it is fully unit-tested.
"""

import json
from collections.abc import Sequence

from pl_keyboard import arch
from pl_keyboard.evaluation import PrefixReport, TopKReport

# Quantization labels smallest-file -> largest. Used both to order table rows and
# (longest-match-first) to detect a quant from a filename.
_QUANT_ORDER: tuple[str, ...] = (
    "Q2_K",
    "Q3_K_M",
    "Q3_K",
    "Q4_0",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "F16",
)


def format_param_count(n: int) -> str:
    """Parameter count as whole millions, e.g. 136_000_000 -> "136M"."""
    return f"{round(n / 1e6)}M"


def format_size_mb(size_bytes: int) -> str:
    """File size in (rounded) mebibytes, matching llama.cpp's own reporting."""
    return f"{size_bytes / 1024 / 1024:.0f} MB"


def format_pct(fraction: float) -> str:
    """A 0..1 fraction as a one-decimal percentage, e.g. 0.292 -> "29.2%"."""
    return f"{fraction * 100:.1f}%"


def format_delta_pp(current: float, baseline: float) -> str:
    """Signed change in percentage points between two 0..1 fractions."""
    return f"{(current - baseline) * 100:+.1f} pp"


def detect_quant(filename: str) -> str:
    """The quantization label embedded in a GGUF filename, or "?" if unknown.

    Matches the most specific label first so "Q3_K_M" is not mis-detected as the
    "Q3_K" substring it contains.
    """
    upper = filename.upper()
    for quant in sorted(_QUANT_ORDER, key=len, reverse=True):
        if quant in upper:
            return quant
    return "?"


def quant_sort_key(name: str) -> int:
    """Sort key ordering known quants smallest-file first, unknowns last."""
    return _QUANT_ORDER.index(name) if name in _QUANT_ORDER else len(_QUANT_ORDER)


def parse_llama_bench(stdout: str) -> tuple[float | None, float | None]:
    """``(prompt_tok_per_s, generation_tok_per_s)`` from ``llama-bench -o json``.

    Prompt-processing rows report ``n_prompt > 0, n_gen == 0`` and generation
    rows the reverse; older builds only carry a ``test`` name ("pp.."/"tg..") —
    both are handled. Anything unparseable yields ``(None, None)`` so the caller
    can still emit the file-size table.
    """
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return (None, None)

    pp = tg = None
    for entry in data:
        speed = entry.get("avg_ts")
        if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
            pp = speed
        elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
            tg = speed
    if pp is None and tg is None:
        for entry in data:
            test = entry.get("test", "")
            if "pp" in test:
                pp = entry.get("avg_ts")
            elif "tg" in test:
                tg = entry.get("avg_ts")
    return (pp, tg)


def param_count(config: dict) -> str:
    """Estimated parameter count as whole millions, e.g. "136M"."""
    return format_param_count(
        arch.estimate_params(config, config.get("vocab_size", arch.VOCAB_SIZE))
    )


def architecture_summary(config: dict) -> str:
    """One-line model shape, e.g. "136M parameters (12 layers × 768 hidden × …)"."""
    return (
        f"{param_count(config)} parameters "
        f"({config['num_hidden_layers']} layers × {config['hidden_size']} hidden "
        f"× {config['num_attention_heads']} heads)"
    )


def quality_table(
    current: TopKReport, baseline: TopKReport | None, ks: Sequence[int] = (1, 3, 5)
) -> str:
    """Cold-start Top-K accuracy + KSR. Adds a baseline/Δ column when a previous
    report is supplied."""
    metrics = [(f"Top-{k} accuracy", current.accuracy[k], k) for k in ks]
    metrics.append(("KSR", current.ksr, None))
    if baseline is None:
        rows = ["| Metric | Value |", "|---|---|"]
        for label, value, _ in metrics:
            rows.append(f"| {label} | {format_pct(value)} |")
        return "\n".join(rows)

    rows = ["| Metric | Baseline | Current | Δ |", "|---|---|---|---|"]
    for label, value, k in metrics:
        base = baseline.accuracy[k] if k is not None else baseline.ksr
        rows.append(
            f"| {label} | {format_pct(base)} | {format_pct(value)} "
            f"| {format_delta_pp(value, base)} |"
        )
    return "\n".join(rows)


def _chars_label(n: int) -> str:
    return f"{n} char" if n == 1 else f"{n} chars"


def prefix_table(
    current: PrefixReport,
    baseline: PrefixReport | None,
    prefix_lens: Sequence[int] = (1, 2, 3),
    ks: Sequence[int] = (1, 3),
) -> str:
    """Prefix-constrained accuracy ("after N typed chars"). With a baseline, each
    Top-k column is followed by its Δ column."""
    if baseline is None:
        header = ["After N chars"] + [f"Top-{k}" for k in ks]
        rows = [_md_row(header), _md_sep(len(header))]
        for plen in prefix_lens:
            cells = [_chars_label(plen)] + [format_pct(current.accuracy[plen][k]) for k in ks]
            rows.append(_md_row(cells))
        return "\n".join(rows)

    header = ["After N chars"]
    for k in ks:
        header += [f"Top-{k}", f"Top-{k} Δ"]
    rows = [_md_row(header), _md_sep(len(header))]
    for plen in prefix_lens:
        cells = [_chars_label(plen)]
        for k in ks:
            cur = current.accuracy[plen][k]
            base = baseline.accuracy[plen][k]
            cells += [format_pct(cur), format_delta_pp(cur, base)]
        rows.append(_md_row(cells))
    return "\n".join(rows)


def speed_table(rows: Sequence[tuple[str, int, float | None, float | None]]) -> str:
    """`llama-bench` throughput per quant; "—" where a measurement is missing."""
    out = ["| Variant | Size | Prompt (t/s) | Prediction (t/s) |", "|---|---|---|---|"]
    for quant, size_bytes, pp, tg in rows:
        out.append(f"| {quant} | {format_size_mb(size_bytes)} | {_speed(pp)} | {_speed(tg)} |")
    return "\n".join(out)


def files_table(rows: Sequence[tuple[str, int]]) -> str:
    """The shipped GGUF variants and their sizes."""
    out = ["| File | Quantization | Size |", "|---|---|---|"]
    for quant, size_bytes in rows:
        out.append(f"| `*-{quant}.gguf` | {quant} | {format_size_mb(size_bytes)} |")
    return "\n".join(out)


def render_release_notes(
    *,
    version: str,
    steps: int,
    config: dict,
    quality: TopKReport,
    quality_baseline: TopKReport | None,
    prefix: PrefixReport,
    prefix_baseline: PrefixReport | None,
    gguf_rows: Sequence[tuple[str, int, float | None, float | None]],
    ks: Sequence[int] = (1, 3, 5),
    prefix_lens: Sequence[int] = (1, 2, 3),
    prefix_ks: Sequence[int] = (1, 3),
    pre_release: bool = False,
) -> str:
    """Assemble the full release-notes document.

    The Speed and Files sections are emitted only when GGUFs were found, so the
    same command produces useful notes mid-training (metrics only) and at release
    time (with the shipped quantizations).
    """
    title = f"# {version} — {steps:,} steps ({param_count(config)})"
    parts = [title, "", f"**Architecture:** {architecture_summary(config)}"]
    if pre_release:
        parts += ["", "> **Pre-release.** Training is ongoing; these numbers are a snapshot."]
    parts += [
        "",
        "## Quality (cold-start next-word, held-out)",
        "",
        quality_table(quality, quality_baseline, ks=ks),
        "",
        "## Prefix accuracy (typing simulation)",
        "",
        prefix_table(prefix, prefix_baseline, prefix_lens=prefix_lens, ks=prefix_ks),
    ]
    if gguf_rows:
        ordered = sorted(gguf_rows, key=lambda r: quant_sort_key(r[0]))
        parts += [
            "",
            "## Speed (llama-bench, 64-token context)",
            "",
            speed_table(ordered),
            "",
            "## Files",
            "",
            files_table([(quant, size) for quant, size, _, _ in ordered]),
        ]
    return "\n".join(parts) + "\n"


def _speed(value: float | None) -> str:
    return f"{value:,.0f}" if value is not None else "—"


def _md_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _md_sep(n: int) -> str:
    return "|" + "---|" * n
