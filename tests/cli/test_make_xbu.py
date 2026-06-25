from cli import make_xbu


def test_make_xbu_writes_original_plus_variants_deterministically(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("zażółć gęślą jaźń kot pies dom\n\nala ma kota tutaj\n", encoding="utf-8")
    out = tmp_path / "out" / "xbu.txt"

    rc = make_xbu.main(["--input", str(src), "--output", str(out), "--seed", "1", "--copies", "2"])

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "zażółć gęślą jaźń kot pies dom" in text  # original always kept
    assert "<XBU>" in text  # at least one augmented variant

    # same seed -> identical output (determinism)
    out2 = tmp_path / "out2.txt"
    make_xbu.main(["--input", str(src), "--output", str(out2), "--seed", "1", "--copies", "2"])
    assert out.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")
