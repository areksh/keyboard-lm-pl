"""Registry of streaming text sources + record→line extraction (pure).

The heavy `datasets.load_dataset(..., streaming=True)` call lives in the CLI
(`cli/download.py:_load_dataset`, marked `# pragma: no cover`); everything here
operates on plain record mappings, so it is fully unit-testable with in-memory
dicts. Each source streams records; `iter_lines` turns a record's text field
into one-line-per-paragraph training input for the cleaner (`02`).
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """A HuggingFace streaming dataset and the field its lines come from."""

    key: str  # CLI name
    path: str  # HF dataset path, e.g. "allenai/c4"
    config: str | None  # HF config / subset name
    split: str
    text_field: str
    trust_remote_code: bool = False


# Polish text sources. Paths/configs are sensible defaults; the download CLI
# lets users override them for a source that has moved or been renamed.
SOURCES: dict[str, Source] = {
    "fineweb2": Source("fineweb2", "HuggingFaceFW/fineweb-2", "pol_Latn", "train", "text"),
    "c4": Source("c4", "allenai/c4", "pl", "train", "text"),
    "tatoeba": Source("tatoeba", "Helsinki-NLP/tatoeba", "pol", "train", "text"),
    "opensubtitles": Source("opensubtitles", "open_subtitles", "pl", "train", "text"),
    "wikipedia": Source("wikipedia", "wikimedia/wikipedia", "20231101.pl", "train", "text"),
}


def iter_lines(records: Iterable[Mapping[str, object]], text_field: str) -> Iterator[str]:
    """Yield stripped, non-empty lines from each record's `text_field`.

    Records whose field is missing or not a string are skipped; multi-paragraph
    text is split so each yielded item is a single line the cleaner can judge.
    """
    for record in records:
        value = record.get(text_field)
        if not isinstance(value, str):
            continue
        for line in value.splitlines():
            line = line.strip()
            if line:
                yield line
