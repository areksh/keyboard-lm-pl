"""Synthetic Polish sentence generation: prompt building + response parsing.

The LLM call is injected as a `Callable[[str], str]`, so generation is fully
unit-testable with a fake client; the real Ollama HTTP client lives in
`cli/synthetic.py:_ollama_client` (marked `# pragma: no cover`). Synthetic data
covers registers and loanword/code-switching that web corpora under-represent.
"""

import re
from collections.abc import Callable, Iterator, Sequence

# Everyday Polish domains, including ones rich in tech terms and loanwords.
TOPICS: tuple[str, ...] = (
    "codzienna rozmowa",
    "technologia i komputery",
    "gotowanie i przepisy",
    "sport i rekreacja",
    "podróże i turystyka",
    "zdrowie i medycyna",
    "praca i biuro",
    "szkoła i edukacja",
    "finanse i zakupy",
    "media społecznościowe",
    "filmy i seriale",
    "polityka i wiadomości",
)

_PROMPT = (
    "Napisz {n} naturalnych, potocznych zdań po polsku na temat: {topic}. "
    "Każde zdanie w osobnej linii, bez numeracji i bez myślników na początku. "
    "Używaj typowych zapożyczeń i terminów technicznych tam, gdzie to naturalne."
)

# Leading list markers an LLM tends to add despite instructions: "1. ", "2) ",
# "- ", "* ", "• " — one or more, in any combination.
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s+)+")


def build_prompt(topic: str, n: int) -> str:
    """Polish instruction asking for `n` one-per-line sentences about `topic`."""
    return _PROMPT.format(n=n, topic=topic)


def parse_response(text: str, min_words: int = 3) -> list[str]:
    """Turn a raw LLM reply into clean training lines.

    Strips leading numbering/bullets and a pair of surrounding quotes, then keeps
    only lines with at least `min_words` words (drops fragments and headers).
    """
    lines: list[str] = []
    for raw in text.splitlines():
        line = _LIST_MARKER.sub("", raw).strip()
        if len(line) >= 2 and line[0] in "\"'" and line[-1] == line[0]:
            line = line[1:-1].strip()
        if len(line.split()) >= min_words:
            lines.append(line)
    return lines


def generate(
    client: Callable[[str], str],
    topics: Sequence[str],
    per_topic: int,
    min_words: int = 3,
) -> Iterator[str]:
    """Prompt `client` once per topic and yield the parsed lines, in order."""
    for topic in topics:
        yield from parse_response(client(build_prompt(topic, per_topic)), min_words)
