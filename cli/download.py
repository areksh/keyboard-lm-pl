"""01_download: stream a HuggingFace Polish text source to a local file.

`main` (source resolution, line writing, limit) is unit-tested with the dataset
loader mocked; `_load_dataset` does the real `datasets` streaming download and is
marked `# pragma: no cover` (network + heavy dependency).
"""

import argparse
from pathlib import Path

from pl_keyboard import sources


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
    args = p.parse_args(argv)

    source = sources.SOURCES[args.source]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as f:
        for line in sources.iter_lines(_load_dataset(source), source.text_field):
            f.write(line + "\n")
            n += 1
            if args.limit is not None and n >= args.limit:
                break

    print(f"wrote {n} lines from {args.source} -> {out}")
    return 0
