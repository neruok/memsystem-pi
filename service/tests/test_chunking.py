from memsystem.chunking import (
    MAX_CHUNK_BYTES,
    MAX_HEADING_CHARS,
    MAX_SECTIONS,
    MAX_TOKENS,
    TOKEN_OVERLAP,
    split_markdown,
)


def test_markdown_chunks_preserve_headings_offsets_and_overlap():
    long_body = " ".join(f"word{i}" for i in range(MAX_TOKENS + 1))
    markdown = f"Intro\n# Parent\n```text\n# code\n```\n## Child\n{long_body}"

    chunks = split_markdown(markdown)

    assert [chunk.heading_path for chunk in chunks] == [
        (), ("Parent",), ("Parent", "Child"), ("Parent", "Child")
    ]
    assert all(markdown[chunk.source_start:chunk.source_end] == chunk.text for chunk in chunks)
    first_words = chunks[-2].text.split()
    second_words = chunks[-1].text.split()
    assert first_words[-TOKEN_OVERLAP:] == second_words[:TOKEN_OVERLAP]


def test_chunks_bound_bytes_and_heading_amplification():
    chunks = split_markdown("x" * (MAX_CHUNK_BYTES + 1))
    assert len(chunks) == 2
    assert all(len(chunk.text.encode()) <= MAX_CHUNK_BYTES for chunk in chunks)

    markdown = "# a\n" * (MAX_SECTIONS + 1)
    chunks = split_markdown(markdown)
    assert len(chunks) < MAX_SECTIONS
    assert chunks[0].source_start == 0
    assert chunks[-1].source_end == len(markdown)
    assert all(left.source_end >= right.source_start for left, right in zip(chunks, chunks[1:]))

    chunks = split_markdown("# " + "x" * 848_000)
    assert all(sum(map(len, chunk.heading_path)) <= MAX_HEADING_CHARS for chunk in chunks)

    nested = "\n".join(f"{'#' * level} {'x' * 500}" for level in range(1, 7))
    chunks = split_markdown(nested + "\n" + "word " * MAX_TOKENS)
    assert all(sum(map(len, chunk.heading_path)) <= MAX_HEADING_CHARS for chunk in chunks)


def test_common_mark_heading_edges():
    body = "word " * MAX_TOKENS
    markdown = f"Title\n=====\n## empty #\n``` trailing\n# still code\n```\n# foo#\n{body}"

    paths = [chunk.heading_path for chunk in split_markdown(markdown)]

    assert ("Title",) in paths
    assert ("Title", "empty") in paths
    assert ("foo#",) in paths
    assert all("still code" not in path for path in paths)
