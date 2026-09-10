"""Heading-aware Markdown chunking."""

from dataclasses import dataclass
import re

CHUNK_PROFILE = "markdown-whitespace-v1-512-64"
MAX_TOKENS = 512
TOKEN_OVERLAP = 64
MAX_CHUNK_BYTES = 262_144
MAX_SECTIONS = 4_096
MAX_HEADING_CHARS = 500

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*(?:\r?\n)?$")
_SETEXT_HEADING = re.compile(r"^ {0,3}(=+|-+)[ \t]*(?:\r?\n)?$")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    heading_path: tuple[str, ...]
    text: str
    source_start: int
    source_end: int


def split_markdown(markdown: str) -> list[Chunk]:
    """Split long Markdown at headings, then into bounded overlapping windows."""
    if len(_TOKEN.findall(markdown)) <= MAX_TOKENS and len(markdown.encode()) <= MAX_CHUNK_BYTES:
        return [Chunk((), markdown, 0, len(markdown))]

    sections: list[tuple[tuple[str, ...], int, int]] = []
    heading_path: tuple[str, ...] = ()
    section_start = 0
    offset = 0
    fence: tuple[str, int] | None = None
    previous: tuple[int, str] | None = None
    too_many_sections = False

    def add_section(path: tuple[str, ...], start: int, end: int) -> None:
        nonlocal too_many_sections
        if end <= start or too_many_sections:
            return
        if len(sections) == MAX_SECTIONS:
            sections.clear()
            too_many_sections = True
            return
        sections.append((path, start, end))

    for line in markdown.splitlines(keepends=True):
        marker = _FENCE_OPEN.match(line)
        if fence is not None:
            closing = re.fullmatch(
                rf" {{0,3}}{re.escape(fence[0])}{{{fence[1]},}}[ \t]*(?:\r?\n)?", line
            )
            if closing:
                fence = None
            previous = None
        elif marker:
            value = marker.group(1)
            fence = value[0], len(value)
            previous = None
        elif heading := _ATX_HEADING.match(line):
            add_section(heading_path, section_start, offset)
            level = len(heading.group(1))
            text = (heading.group(2) or "").rstrip()
            text = re.sub(r"[ \t]+#+[ \t]*$", "", text)
            heading_path = _heading_path(heading_path, level, text)
            section_start = offset
            previous = None
        elif setext := _SETEXT_HEADING.match(line):
            if previous is not None and previous[1].strip():
                heading_start, text = previous
                add_section(heading_path, section_start, heading_start)
                level = 1 if setext.group(1)[0] == "=" else 2
                heading_path = _heading_path(heading_path, level, text.strip())
                section_start = heading_start
            previous = None
        else:
            previous = (offset, line.rstrip("\r\n"))
        offset += len(line)

    add_section(heading_path, section_start, len(markdown))
    if too_many_sections:
        sections = [((), 0, len(markdown))]

    chunks: list[Chunk] = []
    for path, start, end in sections:
        for byte_start, byte_end in _byte_spans(markdown, start, end):
            text = markdown[byte_start:byte_end]
            tokens = list(_TOKEN.finditer(text))
            if len(tokens) <= MAX_TOKENS:
                chunks.append(Chunk(path, text, byte_start, byte_end))
                continue
            for token_start in range(0, len(tokens), MAX_TOKENS - TOKEN_OVERLAP):
                token_end = min(token_start + MAX_TOKENS, len(tokens))
                chunk_start = byte_start if token_start == 0 else byte_start + tokens[token_start].start()
                chunk_end = byte_end if token_end == len(tokens) else byte_start + tokens[token_end].start()
                chunks.append(Chunk(path, markdown[chunk_start:chunk_end], chunk_start, chunk_end))
                if token_end == len(tokens):
                    break
    return chunks


def _heading_path(current: tuple[str, ...], level: int, text: str) -> tuple[str, ...]:
    parents = current[: level - 1]
    remaining = max(0, MAX_HEADING_CHARS - sum(map(len, parents)))
    return parents + (text[:remaining],)


def _byte_spans(text: str, start: int, end: int):
    while start < end:
        if len(text[start:end].encode()) <= MAX_CHUNK_BYTES:
            yield start, end
            return
        low, high = start + 1, end
        while low < high:
            middle = (low + high + 1) // 2
            if len(text[start:middle].encode()) <= MAX_CHUNK_BYTES:
                low = middle
            else:
                high = middle - 1
        yield start, low
        start = low
    if start == end == 0:
        yield 0, 0
