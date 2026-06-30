"""CLI orchestration for 10_release_notes: model loading and the llama-bench
subprocess are injected, so the wiring (eval loop, GGUF handling, baseline Δ,
file outputs) is exercised without torch or the binary."""

import json

from cli import release_notes as cli
from pl_keyboard import evaluation

B = evaluation.WORD_BOUNDARY

# A 5-piece toy vocabulary; logits rank pies(9) > kot(5) > dom(3) > "x"/"a" lower.
_PIECES = ["<s>", "kot" + B, "pies" + B, "dom" + B, "x"]
_WORD_TO_ID = {"kot": 1, "pies": 2, "dom": 3}


def _fake_loader(logits=None):
    vec = logits or [0.0, 5.0, 9.0, 3.0, 1.0]

    def encode(sentence):
        return [0, *[_WORD_TO_ID.get(w, 4) for w in sentence.split()]]

    def loader(model_dir, sp_model):
        return encode, _PIECES, lambda context_ids: list(vec)

    return loader


def _checkpoint(tmp_path, *, config=True):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    if config:
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "intermediate_size": 3072,
                    "vocab_size": 15008,
                }
            ),
            encoding="utf-8",
        )
    return model_dir


def _eval_file(tmp_path, text="kot pies dom\n"):
    f = tmp_path / "held.txt"
    f.write_text(text, encoding="utf-8")
    return f


def _args(model_dir, eval_file, *extra):
    return [
        "--model-dir",
        str(model_dir),
        "--sp-model",
        "t.model",
        "--eval-file",
        str(eval_file),
        "--version",
        "v0.1.0",
        "--steps",
        "35000",
        *extra,
    ]


# ── validation ────────────────────────────────────────────────────────────────


def test_missing_model_dir_returns_error(tmp_path, capsys):
    rc = cli.main(_args(tmp_path / "nope", tmp_path / "held.txt"))
    assert rc == 1
    assert "model dir not found" in capsys.readouterr().err


def test_missing_eval_file_returns_error_before_loading(tmp_path, monkeypatch, capsys):
    loaded = False

    def boom(*a):
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(cli, "_load_model", boom)
    rc = cli.main(_args(_checkpoint(tmp_path), tmp_path / "missing.txt"))
    assert rc == 1
    assert "eval file not found" in capsys.readouterr().err
    assert loaded is False


def test_missing_config_json_returns_error(tmp_path, capsys):
    model_dir = _checkpoint(tmp_path, config=False)
    rc = cli.main(_args(model_dir, _eval_file(tmp_path)))
    assert rc == 1
    assert "config.json not found" in capsys.readouterr().err


# ── happy path ────────────────────────────────────────────────────────────────


def test_generates_notes_without_ggufs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_model", _fake_loader())
    rc = cli.main(_args(_checkpoint(tmp_path), _eval_file(tmp_path)))
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("# v0.1.0")
    assert "35,000 steps" in out
    assert "## Quality" in out
    # "pies" is a top-1 hit (logit 9), so Top-1 accuracy is 1 of 2 positions.
    assert "| Top-1 accuracy | 50.0% |" in out
    assert "## Speed" not in out  # no GGUFs -> no speed/files sections
    assert "Δ" not in out  # no baseline -> no comparison


def test_writes_markdown_and_json_report(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_load_model", _fake_loader())
    out_md = tmp_path / "notes" / "RELEASE.md"
    report = tmp_path / "metrics.json"
    rc = cli.main(
        _args(
            _checkpoint(tmp_path),
            _eval_file(tmp_path),
            "--output",
            str(out_md),
            "--report",
            str(report),
        )
    )
    assert rc == 0
    assert out_md.read_text(encoding="utf-8").startswith("# v0.1.0")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["version"] == "v0.1.0"
    assert payload["quality"]["accuracy"]["1"] == 0.5
    assert payload["prefix"]["positions"]["1"] == 2


def test_pre_release_flag_adds_snapshot_note(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_model", _fake_loader())
    rc = cli.main(_args(_checkpoint(tmp_path), _eval_file(tmp_path), "--pre-release"))
    assert rc == 0
    assert "Pre-release" in capsys.readouterr().out


# ── baseline comparison ───────────────────────────────────────────────────────


def test_baseline_report_adds_delta_columns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_model", _fake_loader())
    model_dir, eval_file = _checkpoint(tmp_path), _eval_file(tmp_path)
    report = tmp_path / "prev.json"
    # First run writes a report; feed it back as the baseline of a second run.
    assert cli.main(_args(model_dir, eval_file, "--report", str(report))) == 0
    rc = cli.main(_args(model_dir, eval_file, "--baseline", str(report)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "| Metric | Baseline | Current | Δ |" in out
    # Identical run -> zero deltas, proving the round-trip (de)serialization.
    assert "+0.0 pp" in out


# ── GGUF sizes + llama-bench ──────────────────────────────────────────────────


def test_gguf_sizes_and_bench_speed_tables(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_model", _fake_loader())
    gguf = tmp_path / "pl_keyboard-Q4_0.gguf"
    gguf.write_bytes(b"\0" * (77 * 1024 * 1024))
    bench_json = (
        '[{"n_prompt":64,"n_gen":0,"avg_ts":1711.0},{"n_prompt":0,"n_gen":1,"avg_ts":562.0}]'
    )
    monkeypatch.setattr(cli, "_run_bench", lambda path, binary: bench_json)

    rc = cli.main(_args(_checkpoint(tmp_path), _eval_file(tmp_path), "--gguf", str(gguf)))
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Speed" in out
    assert "| Q4_0 | 77 MB | 1,711 | 562 |" in out
    assert "## Files" in out
    assert "| `*-Q4_0.gguf` | Q4_0 | 77 MB |" in out


def test_missing_llama_bench_binary_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_model", _fake_loader())
    gguf = tmp_path / "m-Q8_0.gguf"
    gguf.write_bytes(b"\0" * 1024)

    def missing(*a):
        raise FileNotFoundError

    monkeypatch.setattr(cli, "_run_bench", missing)
    rc = cli.main(_args(_checkpoint(tmp_path), _eval_file(tmp_path), "--gguf", str(gguf)))
    assert rc == 1
    assert "llama-bench not found" in capsys.readouterr().err


# ── helpers ───────────────────────────────────────────────────────────────────


def test_find_ggufs_expands_globs_dedups_and_skips_missing(tmp_path):
    (tmp_path / "a-Q4_0.gguf").write_bytes(b"x")
    (tmp_path / "b-Q8_0.gguf").write_bytes(b"x")
    glob = str(tmp_path / "*.gguf")
    literal = str(tmp_path / "a-Q4_0.gguf")  # also matched by the glob -> dedup
    missing = str(tmp_path / "nope.gguf")

    found = cli._find_ggufs([glob, literal, missing])
    assert [p.name for p in found] == ["a-Q4_0.gguf", "b-Q8_0.gguf"]


def test_collect_records_marks_too_short_prefix_lengths_none():
    # Position (2, "do"): "do" is 2 chars, so the 3-char prefix length is recorded
    # as None and skipped, while lengths 1 and 2 get ranked candidates.
    pieces = ["<s>", "kot" + B, "do" + B]

    def encode(sentence):
        return [0, 1, 2]

    model = (encode, pieces, lambda ctx: [0.0, 1.0, 9.0])
    index = evaluation.build_prefix_index(pieces, (1, 2, 3))
    topk, prefix = cli._collect_records(model, ["kot do"], index, prefix_lens=(1, 2, 3))

    assert topk == [("do", ["do", "kot"])]  # ranked by logit: do(9) > kot(1)
    (_, by_plen) = prefix[0]
    assert by_plen[1] == ["do"] and by_plen[2] == ["do"]
    assert by_plen[3] is None
