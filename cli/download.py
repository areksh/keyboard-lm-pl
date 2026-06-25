"""01_download: stream a HuggingFace Polish text source to a local file.

`main` (source resolution, line writing, limit) is unit-tested with the dataset
loader mocked; `_load_dataset` does the real `datasets` streaming download and is
marked `# pragma: no cover` (network + heavy dependency).
"""

import argparse
import logging
from pathlib import Path

from cli import _runtime
from pl_keyboard import sources

log = logging.getLogger("pl_keyboard")


def _load_dataset(source: sources.Source):  # pragma: no cover - network + datasets streaming
    import datasets

    return datasets.load_dataset(
        source.path,
        source.config,
        split=source.split,
        streaming=True,
        trust_remote_code=source.trust_remote_code,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stream a Polish text source to disk.")
    p.add_argument("--source", required=True, choices=sorted(sources.SOURCES))
    p.add_argument("--output", required=True, help="Output file (one line per paragraph).")
    p.add_argument("--limit", type=int, default=None, help="Max lines to write (default: all).")
    _runtime.add_common_args(p)
    args = p.parse_args(argv)
    _runtime.configure(args)

    source = sources.SOURCES[args.source]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("streaming %s -> %s (limit=%s)", args.source, out, args.limit)

    n = 0
    lines = sources.iter_lines(_load_dataset(source), source.text_field)
    with out.open("w", encoding="utf-8") as f:
        for line in _runtime.progress(
            lines, desc=f"download {args.source}", log=log, total=args.limit
        ):
            f.write(line + "\n")
            n += 1
            if args.limit is not None and n >= args.limit:
                break

    print(f"wrote {n} lines from {args.source} -> {out}")
    return 0
