"""10_release_notes: generate GitHub-release notes (Markdown) for a trained model.

Mirrors the German release write-up: an architecture summary (auto-detected from
the checkpoint's ``config.json``), cold-start Top-1/3/5 + KSR and prefix-accuracy
tables computed on a held-out set, and per-quantization size + ``llama-bench``
speed tables. Step count can't be recovered from the checkpoint (the trainer
saves only weights), so it is passed via ``--steps``; ``--baseline`` points at a
previous run's JSON report to add the Δ comparison columns.

The metric math and Markdown rendering are pure (`pl_keyboard.evaluation` /
`pl_keyboard.release_notes`); only model loading (`_load_model`) and the
`llama-bench` subprocess (`_run_bench`) are heavy and `# pragma: no cover`,
exercised by tests/integration/.
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from cli import _runtime
from pl_keyboard import evaluation, release_notes

log = logging.getLogger("pl_keyboard")

CONTEXT_LEN = 256
SCAN = 150  # logits scanned per position to find complete-word candidates
MAX_WORDS = 5
KS = (1, 3, 5)
PREFIX_LENS = (1, 2, 3)
PREFIX_KS = (1, 3)

# (encode sentence -> token ids, token-id -> piece table, context ids -> logits).
Model = tuple[Callable[[str], list[int]], Sequence[str], Callable[[list[int]], Sequence[float]]]


def _load_model(model_dir: str, sp_model: str) -> Model:  # pragma: no cover - heavy torch inference
    import torch
    from sentencepiece import SentencePieceProcessor
    from transformers import LlamaForCausalLM

    sp = SentencePieceProcessor(model_file=sp_model)
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    bos = model.config.bos_token_id or 1
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    def encode(sentence: str) -> list[int]:
        return [bos, *sp.encode(sentence, out_type=int)]

    def get_logits(context_ids: list[int]) -> Sequence[float]:
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([context_ids])).logits
        return logits[0, -1].tolist()

    return encode, pieces, get_logits


def _run_bench(  # pragma: no cover - subprocess against the llama-bench binary
    gguf: Path, llama_bench: str, prompt_tokens: int = 64, gen_tokens: int = 1, reps: int = 3
) -> str:
    import subprocess

    result = subprocess.run(
        [
            llama_bench,
            "-m",
            str(gguf),
            "-p",
            str(prompt_tokens),
            "-n",
            str(gen_tokens),
            "-r",
            str(reps),
            "-o",
            "json",
            "--no-warmup",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.warning("llama-bench failed for %s: %s", gguf.name, result.stderr.strip())
        return ""
    return result.stdout


def _collect_records(
    model: Model,
    sentences: Sequence[str],
    prefix_index: dict[int, dict[str, list[int]]],
    *,
    ctx_len: int = CONTEXT_LEN,
    scan: int = SCAN,
    max_words: int = MAX_WORDS,
    prefix_lens: Sequence[int] = PREFIX_LENS,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, dict[int, list[str] | None]]]]:
    """Run the model over `sentences`, returning the Top-K and prefix records.

    One forward pass per evaluable word position feeds both the cold-start
    ranking and the prefix-constrained ranking (a prefix length is recorded as
    ``None`` when the truth is too short for it or no vocab word shares its
    prefix), so the pure aggregators can compute every metric from one pass.
    """
    encode, pieces, get_logits = model
    topk_records: list[tuple[str, list[str]]] = []
    prefix_records: list[tuple[str, dict[int, list[str] | None]]] = []
    for sentence in _runtime.progress(sentences, desc="evaluate", log=log, unit="sentence"):
        ids = encode(sentence)[:ctx_len]
        sentence_pieces = [pieces[i] for i in ids]
        for pos, true_word in evaluation.word_positions(sentence_pieces):
            logits = get_logits(ids[:pos])
            topk_records.append(
                (true_word, evaluation.topk_words_from_logits(logits, pieces, max_words, scan))
            )
            by_plen: dict[int, list[str] | None] = {}
            for plen in prefix_lens:
                candidates = (
                    prefix_index[plen].get(true_word[:plen], []) if len(true_word) >= plen else []
                )
                by_plen[plen] = (
                    evaluation.rank_candidates(candidates, pieces, logits.__getitem__)
                    if candidates
                    else None
                )
            prefix_records.append((true_word, by_plen))
    return topk_records, prefix_records


def _find_ggufs(patterns: Sequence[str]) -> list[Path]:
    """Expand `patterns` (literal paths or globs) into unique existing GGUFs."""
    found: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if "*" in pattern or "?" in pattern:
            found.extend(sorted(path.parent.glob(path.name)))
        elif path.is_file():
            found.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _gguf_rows(
    gguf_paths: Sequence[Path], run_bench: Callable[[Path], str]
) -> list[tuple[str, int, float | None, float | None]]:
    """Per-GGUF ``(quant, size_bytes, prompt_ts, gen_ts)`` rows for the tables."""
    rows = []
    for path in gguf_paths:
        log.info("benchmarking %s", path.name)
        pp, tg = release_notes.parse_llama_bench(run_bench(path))
        rows.append((release_notes.detect_quant(path.name), path.stat().st_size, pp, tg))
    return rows


def _topk_payload(report: evaluation.TopKReport) -> dict:
    return {
        "positions": report.positions,
        "accuracy": {str(k): v for k, v in report.accuracy.items()},
        "ksr": report.ksr,
        "avg_word_len": report.avg_word_len,
    }


def _prefix_payload(report: evaluation.PrefixReport) -> dict:
    return {
        "positions": {str(p): n for p, n in report.positions.items()},
        "accuracy": {
            str(p): {str(k): v for k, v in row.items()} for p, row in report.accuracy.items()
        },
    }


def _load_topk(payload: dict) -> evaluation.TopKReport:
    return evaluation.TopKReport(
        positions=payload["positions"],
        accuracy={int(k): v for k, v in payload["accuracy"].items()},
        ksr=payload["ksr"],
        avg_word_len=payload["avg_word_len"],
    )


def _load_prefix(payload: dict) -> evaluation.PrefixReport:
    return evaluation.PrefixReport(
        positions={int(p): n for p, n in payload["positions"].items()},
        accuracy={
            int(p): {int(k): v for k, v in row.items()} for p, row in payload["accuracy"].items()
        },
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate release notes for a trained keyboard model.")
    p.add_argument("--model-dir", required=True, help="HuggingFace checkpoint directory.")
    p.add_argument("--sp-model", required=True, help="SentencePiece .model file.")
    p.add_argument("--eval-file", required=True, help="Held-out text used for the metrics.")
    p.add_argument("--version", required=True, help="Release version string, e.g. v0.1.0.")
    p.add_argument("--steps", required=True, type=int, help="Training steps this checkpoint ran.")
    p.add_argument("--gguf", nargs="*", default=[], help="GGUF files or globs to size + benchmark.")
    p.add_argument("--baseline", default=None, help="Previous run's JSON report, for Δ columns.")
    p.add_argument("--llama-bench", default="llama-bench", help="Path to the llama-bench binary.")
    p.add_argument("--max-sentences", type=int, default=500, help="Cap eval sentences (def. 500).")
    p.add_argument("--output", default=None, help="Write the Markdown notes here (also printed).")
    p.add_argument("--report", default=None, help="Write metrics JSON here (reusable baseline).")
    p.add_argument("--pre-release", action="store_true", help="Add the pre-release snapshot note.")
    _runtime.add_common_args(p)
    args = p.parse_args(argv)
    _runtime.configure(args)

    if not Path(args.model_dir).is_dir():
        print(f"model dir not found: {args.model_dir}", file=sys.stderr)
        return 1
    if not Path(args.eval_file).is_file():
        print(f"eval file not found: {args.eval_file}", file=sys.stderr)
        return 1
    config_path = Path(args.model_dir) / "config.json"
    if not config_path.is_file():
        print(f"config.json not found in {args.model_dir}", file=sys.stderr)
        return 1

    sentences = [
        line.strip()
        for line in Path(args.eval_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.max_sentences]
    config = json.loads(config_path.read_text(encoding="utf-8"))

    log.info("loading model from %s", args.model_dir)
    model = _load_model(args.model_dir, args.sp_model)
    prefix_index = evaluation.build_prefix_index(model[1], PREFIX_LENS)
    log.info("scoring %d held-out sentence(s)", len(sentences))
    topk_records, prefix_records = _collect_records(model, sentences, prefix_index)
    quality = evaluation.evaluate_topk(topk_records, ks=KS)
    prefix = evaluation.evaluate_prefix(prefix_records, ks=PREFIX_KS, prefix_lens=PREFIX_LENS)

    quality_baseline = prefix_baseline = None
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        quality_baseline = _load_topk(baseline["quality"])
        prefix_baseline = _load_prefix(baseline["prefix"])

    ggufs = _find_ggufs(args.gguf)
    try:
        gguf_rows = _gguf_rows(ggufs, lambda path: _run_bench(path, args.llama_bench))
    except FileNotFoundError:
        print(f"llama-bench not found: {args.llama_bench}", file=sys.stderr)
        return 1

    notes = release_notes.render_release_notes(
        version=args.version,
        steps=args.steps,
        config=config,
        quality=quality,
        quality_baseline=quality_baseline,
        prefix=prefix,
        prefix_baseline=prefix_baseline,
        gguf_rows=gguf_rows,
        ks=KS,
        prefix_lens=PREFIX_LENS,
        prefix_ks=PREFIX_KS,
        pre_release=args.pre_release,
    )
    print(notes)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(notes, encoding="utf-8")
        log.info("wrote notes to %s", out)
    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": args.version,
            "steps": args.steps,
            "quality": _topk_payload(quality),
            "prefix": _prefix_payload(prefix),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        report.write_text(text, encoding="utf-8")
        log.info("wrote metrics report to %s", report)
    return 0
