"""03_generate_synthetic: synthesise Polish training lines via a local LLM.

`main` (topic/round orchestration + writing) is unit-tested with the client
mocked; `_ollama_client` does the real HTTP call to an Ollama server and is
marked `# pragma: no cover`.
"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from cli import _runtime
from pl_keyboard import logging_setup, synthetic

log = logging.getLogger("pl_keyboard")


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
    _runtime.add_common_args(p)
    args = p.parse_args(argv)
    _runtime.configure(args)

    client = _ollama_client(args.host, args.model)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "generating via %s @ %s: %d round(s) x %d topic(s)",
        args.model,
        args.host,
        args.rounds,
        len(args.topics),
    )

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for r in _runtime.progress(range(args.rounds), desc="synth rounds", log=log, unit="round"):
            log.debug("round %d/%d", r + 1, args.rounds)
            for line in synthetic.generate(client, args.topics, args.per_topic, args.min_words):
                f.write(line + "\n")
                n += 1
                log.log(logging_setup.DEV, "line: %s", line)

    print(f"wrote {n} synthetic lines -> {out}")
    return 0
