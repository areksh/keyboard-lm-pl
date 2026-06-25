"""03_generate_synthetic: synthesise Polish training lines via a local LLM.

`main` (topic/round orchestration + writing) is unit-tested with the client
mocked; `_ollama_client` does the real HTTP call to an Ollama server and is
marked `# pragma: no cover`.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from pl_keyboard import synthetic


def _ollama_client(host: str, model: str) -> Callable[[str], str]:  # pragma: no cover - network I/O
    import json
    import urllib.request

    def call(prompt: str) -> str:
        body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["response"]

    return call


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate synthetic Polish training lines via Ollama.")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="llama3.1")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--per-topic", type=int, default=20, help="Sentences requested per topic.")
    p.add_argument("--rounds", type=int, default=1, help="Times to loop over the topic list.")
    p.add_argument("--topics", nargs="+", default=list(synthetic.TOPICS))
    p.add_argument("--min-words", type=int, default=3)
    args = p.parse_args(argv)

    client = _ollama_client(args.host, args.model)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for _ in range(args.rounds):
            for line in synthetic.generate(client, args.topics, args.per_topic, args.min_words):
                f.write(line + "\n")
                n += 1

    print(f"wrote {n} synthetic lines -> {out}")
    return 0
