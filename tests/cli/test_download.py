from cli import download
from pl_keyboard import sources


def test_download_writes_extracted_lines_and_drains_source(tmp_path, monkeypatch, capsys):
    records = [
        {"text": "Pierwsze.\nDrugie."},
        {"text": ""},
        {"other": "skip"},
        {"text": "Trzecie."},
    ]
    monkeypatch.setattr(download, "_load_dataset", lambda source: records)
    out = tmp_path / "raw" / "c4.txt"

    rc = download.main(["--source", "c4", "--output", str(out)])

    assert rc == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["Pierwsze.", "Drugie.", "Trzecie."]
    assert "wrote 3 lines from c4" in capsys.readouterr().out


def test_download_respects_limit_on_an_unbounded_source(tmp_path, monkeypatch):
    def endless(source):
        while True:
            yield {"text": "x y z"}

    seen = {}

    def fake_loader(source):
        seen["s"] = source
        return endless(source)

    monkeypatch.setattr(download, "_load_dataset", fake_loader)
    out = tmp_path / "fw.txt"

    rc = download.main(["--source", "fineweb2", "--output", str(out), "--limit", "2"])

    assert rc == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["x y z", "x y z"]
    # The CLI passed the resolved Source for the requested key through to the loader.
    assert seen["s"] is sources.SOURCES["fineweb2"]
