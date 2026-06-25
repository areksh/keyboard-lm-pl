from cli import synthetic as synthetic_cli


def test_synthetic_writes_generated_lines_for_each_round(tmp_path, monkeypatch, capsys):
    captured = {}

    def fake_loader(host, model):
        captured["host"], captured["model"] = host, model
        return lambda prompt: "To jest pierwsze zdanie.\nA to jest drugie zdanie."

    monkeypatch.setattr(synthetic_cli, "_ollama_client", fake_loader)
    out = tmp_path / "syn" / "data.txt"

    rc = synthetic_cli.main(
        [
            "--output",
            str(out),
            "--topics",
            "sport",
            "kuchnia",
            "--per-topic",
            "2",
            "--rounds",
            "2",
            "--model",
            "bielik",
            "--host",
            "http://example:1234",
        ]
    )

    assert rc == 0
    # 2 topics x 2 lines x 2 rounds = 8 lines.
    assert len(out.read_text(encoding="utf-8").splitlines()) == 8
    assert captured == {"host": "http://example:1234", "model": "bielik"}
    assert "wrote 8 synthetic lines" in capsys.readouterr().out


def test_synthetic_defaults_to_builtin_topics(tmp_path, monkeypatch):
    from pl_keyboard import synthetic

    prompts: list[str] = []

    def fake_loader(host, model):
        def call(prompt):
            prompts.append(prompt)
            return "Pełne zdanie testowe tutaj."

        return call

    monkeypatch.setattr(synthetic_cli, "_ollama_client", fake_loader)
    out = tmp_path / "data.txt"

    rc = synthetic_cli.main(["--output", str(out), "--per-topic", "1"])

    assert rc == 0
    assert len(prompts) == len(synthetic.TOPICS)  # one prompt per built-in topic
