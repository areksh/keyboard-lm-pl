from cli import clean


def test_clean_main_writes_kept_lines_from_multiple_inputs(tmp_path, capsys):
    f1 = tmp_path / "a.txt"
    f1.write_text("To jest mały kot\nПривет это kot tu\n", encoding="utf-8")
    f2 = tmp_path / "b.txt"
    f2.write_text("Idę do dużego domu\nkot pies\n", encoding="utf-8")
    out = tmp_path / "sub" / "out.txt"  # parent dir must be created

    rc = clean.main(["--input", str(f1), str(f2), "--output", str(out)])

    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert "To jest mały kot" in lines  # kept
    assert "Idę do dużego domu" in lines  # kept
    assert "kot pies" not in lines  # dropped: too few words
    assert not any("Привет" in ln for ln in lines)  # dropped: foreign script
    assert "kept" in capsys.readouterr().out
