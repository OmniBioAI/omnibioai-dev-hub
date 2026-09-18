"""Comprehensive tests for the markdown-aware chunker (processing/chunker.py).

Developer: Manish Kumar <manish@omnibioai.org>
"""
from processing.chunker import (
    MAX_CHARS,
    _split_at_paragraphs,
    _split_at_word_boundary,
    chunk_text,
)

# ---------------------------------------------------------------------------
# _split_at_word_boundary
# ---------------------------------------------------------------------------

class TestSplitAtWordBoundary:
    """Behavior of _split_at_word_boundary: word-aware splitting under a character limit, with a
    hard cut for oversize words."""

    def test_short_text_returned_unchanged(self):
        """Return text within the limit as a single unchanged piece."""
        assert _split_at_word_boundary("hello world", 100) == ["hello world"]

    def test_exact_limit_not_split(self):
        """Leave text that is exactly at the limit unsplit."""
        text = "abcde"
        assert _split_at_word_boundary(text, 5) == [text]

    def test_splits_at_last_space_before_limit(self):
        """Cut at the last space before the limit rather than mid-word."""
        # "ab cde fg" — rfind(' ', 0, 7) hits space at pos 6 → "ab cde" / "fg"
        text = "ab cde fg"
        result = _split_at_word_boundary(text, 7)
        assert result == ["ab cde", "fg"]

    def test_no_space_forces_hard_cut(self):
        """Cut at the limit when the text has no space to break on."""
        text = "abcdefgh"
        result = _split_at_word_boundary(text, 4)
        assert result == ["abcd", "efgh"]

    def test_multiple_passes_all_within_limit(self):
        """Keep every piece within the limit across several splits while preserving all words."""
        text = "one two three four five six seven"
        max_c = 10
        result = _split_at_word_boundary(text, max_c)
        assert all(len(c) <= max_c for c in result)
        # Round-trip: joining should contain all original words
        assert set(text.split()) == set(" ".join(result).split())

    def test_single_word_longer_than_limit(self):
        """Hard-cut a single word longer than the limit into limit-sized pieces."""
        text = "superlongword"
        result = _split_at_word_boundary(text, 5)
        assert result == ["super", "longw", "ord"]

    def test_leading_spaces_stripped_after_cut(self):
        """Strip the leading space from the remainder after cutting at a space."""
        text = "aaa bbb"
        result = _split_at_word_boundary(text, 4)
        assert result == ["aaa", "bbb"]


# ---------------------------------------------------------------------------
# _split_at_paragraphs
# ---------------------------------------------------------------------------

class TestSplitAtParagraphs:
    """Behavior of _split_at_paragraphs: packing paragraphs into size-limited chunks, with a
    word-boundary fallback for oversize paragraphs."""

    def test_short_text_returned_unchanged(self):
        """Return text within the limit as a single unchanged chunk."""
        text = "Para one.\n\nPara two."
        assert _split_at_paragraphs(text, 1000) == [text]

    def test_empty_text_returns_empty(self):
        """Return no chunks for empty text."""
        assert _split_at_paragraphs("", 100) == []

    def test_whitespace_only_returns_empty(self):
        """Return no chunks for whitespace-only text."""
        assert _split_at_paragraphs("   \n\n   ", 100) == []

    def test_splits_at_paragraph_boundary(self):
        """Split two paragraphs that cannot share a chunk at the blank line between them."""
        p1 = "x" * 300
        p2 = "y" * 300
        text = p1 + "\n\n" + p2
        result = _split_at_paragraphs(text, 400)
        assert result == [p1, p2]

    def test_accumulates_small_paragraphs_when_they_fit(self):
        """Keep small paragraphs together in one chunk while they fit within the limit."""
        p1, p2 = "aaa", "bbb"
        text = p1 + "\n\n" + p2
        result = _split_at_paragraphs(text, 100)
        assert len(result) == 1
        assert p1 in result[0] and p2 in result[0]

    def test_three_paras_packed_optimally(self):
        """Pack the first two paragraphs into one chunk and put the large third paragraph in the
        next."""
        # p1+p2 fit together; p3 causes a flush
        p1 = "a" * 200
        p2 = "b" * 200
        p3 = "c" * 300
        text = p1 + "\n\n" + p2 + "\n\n" + p3
        # budget 450: p1+p2 = 402 (fits), adding p3 = 704 (doesn't) → flush
        result = _split_at_paragraphs(text, 450)
        assert len(result) == 2
        assert p1 in result[0] and p2 in result[0]
        assert p3 in result[1]

    def test_oversize_paragraph_uses_word_boundary(self):
        """Split a paragraph longer than the limit at word boundaries so every chunk fits."""
        big_para = "word " * 500  # 2500 chars
        result = _split_at_paragraphs(big_para, 300)
        assert all(len(c) <= 300 for c in result)
        assert len(result) > 1

    def test_oversize_para_flushes_accumulator_first(self):
        """Emit the accumulated small paragraph as its own chunk before splitting an oversize one."""
        small = "s" * 50
        big = "B " * 300  # 600 chars
        text = small + "\n\n" + big
        result = _split_at_paragraphs(text, 100)
        # small goes to one chunk, big gets word-split
        assert result[0] == small
        assert all(len(c) <= 100 for c in result[1:])

    def test_zero_budget_falls_back_gracefully(self):
        """Return non-empty output instead of raising when the size budget is zero."""
        text = "some text here"
        result = _split_at_paragraphs(text, 0)
        assert result  # must not raise or return empty

    def test_multiple_blank_lines_treated_as_single_boundary(self):
        """Treat several consecutive blank lines as one paragraph boundary and keep both paragraphs."""
        p1 = "first"
        p2 = "second"
        text = p1 + "\n\n\n\n" + p2
        # When the combined text fits in the budget, it is returned unchanged.
        # When it exceeds the budget, multiple blank lines collapse to one boundary.
        result = _split_at_paragraphs(text, 1000)
        assert all(p in " ".join(result) for p in [p1, p2])


# ---------------------------------------------------------------------------
# chunk_text — core behavior
# ---------------------------------------------------------------------------

class TestChunkTextBasic:
    """Basic chunk_text contract: empty input, plain text, argument handling, and content
    preservation."""

    def test_empty_string_returns_empty_list(self):
        """Return an empty list for an empty string."""
        assert chunk_text("") == []

    def test_none_equivalent_empty(self):
        """Return an empty list for whitespace-only input."""
        # Edge: whitespace-only body with no content → empty
        assert chunk_text("   \n\n   ") == []

    def test_plain_text_no_headers_single_chunk(self):
        """Return header-less plain text as a single unchanged chunk."""
        text = "Plain text with no markdown whatsoever."
        result = chunk_text(text)
        assert result == [text]

    def test_chunk_size_kwarg_accepted_without_error(self):
        """Accept a chunk_size keyword without error and return a one-element list for short text."""
        result = chunk_text("hello world", chunk_size=2)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_no_empty_strings_in_output(self):
        """Never emit an empty or whitespace-only chunk, even for header-only sections."""
        text = "# A\n\n# B\n\n# C\n"
        chunks = chunk_text(text)
        assert all(c.strip() for c in chunks)

    def test_all_content_preserved(self):
        """Preserve every word of the input across the emitted chunks."""
        text = "# H1\nIntro\n\n## H2\nDetails here\n\n### H3\nLeaf content"
        chunks = chunk_text(text)
        joined = "\n".join(chunks)
        for word in ["Intro", "Details here", "Leaf content"]:
            assert word in joined


# ---------------------------------------------------------------------------
# chunk_text — header splitting
# ---------------------------------------------------------------------------

class TestChunkTextHeaders:
    """Markdown-header-aware chunking: one chunk per section, with a breadcrumb of parent headers."""

    def test_single_header_included_in_chunk(self):
        """Start a header section's chunk with its header line and include its body."""
        text = "# Overview\nThis is the overview section."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].startswith("# Overview")
        assert "overview section" in chunks[0]

    def test_two_h1_headers_produce_two_chunks(self):
        """Produce one chunk per top-level header."""
        text = "# First\nContent A.\n\n# Second\nContent B."
        chunks = chunk_text(text)
        assert len(chunks) == 2

    def test_h1_and_h2_produce_two_chunks(self):
        """Produce separate chunks for a parent section and its child section."""
        text = "# Parent\nIntro.\n\n## Child\nDetail."
        chunks = chunk_text(text)
        assert len(chunks) == 2

    def test_child_header_gets_parent_breadcrumb(self):
        """Prefix a child section's chunk with its parent header, joined by a breadcrumb separator."""
        text = "# Parent\nIntro.\n\n## Child\nDetail."
        chunks = chunk_text(text)
        h2_chunk = next(c for c in chunks if "Detail" in c)
        assert "# Parent" in h2_chunk
        assert "## Child" in h2_chunk
        assert " > " in h2_chunk

    def test_h3_gets_full_three_level_breadcrumb(self):
        """Give an h3 section a breadcrumb containing its h1 and h2 ancestors."""
        text = "# A\nroot\n\n## B\nmid\n\n### C\nleaf"
        chunks = chunk_text(text)
        h3_chunk = next(c for c in chunks if "leaf" in c)
        assert "# A" in h3_chunk
        assert "## B" in h3_chunk
        assert "### C" in h3_chunk

    def test_sibling_header_does_not_appear_in_peer_breadcrumb(self):
        """Exclude a sibling section's header from a peer section's breadcrumb."""
        text = "# Root\nfirst\n\n## X\nunderX\n\n## Y\nunderY"
        chunks = chunk_text(text)
        y_chunk = next(c for c in chunks if "underY" in c)
        assert "## Y" in y_chunk
        assert "## X" not in y_chunk  # sibling must be excluded

    def test_h1_resets_entire_stack(self):
        """Reset the header breadcrumb when a new top-level header starts."""
        text = "# First\nA\n\n## Sub\nB\n\n# Second\nC"
        chunks = chunk_text(text)
        second_chunk = next(c for c in chunks if c.strip().startswith("# Second"))
        # "# First" and "## Sub" should NOT appear in the "# Second" chunk prefix
        assert "# First" not in second_chunk
        assert "## Sub" not in second_chunk

    def test_preamble_before_first_header_is_included(self):
        """Include text that appears before the first header in the output."""
        text = "Preamble text here.\n\n# Section\nContent."
        chunks = chunk_text(text)
        assert any("Preamble text" in c for c in chunks)

    def test_header_only_section_emitted(self):
        """Emit a chunk for a section that consists of a header alone."""
        # A section with a header but no body text still produces a chunk
        text = "# Title Only"
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert "# Title Only" in chunks[0]

    def test_four_headers_produce_four_chunks(self):
        """Produce one chunk per header for four top-level sections."""
        parts = [f"# H{i}\nContent {i}." for i in range(4)]
        text = "\n\n".join(parts)
        assert len(chunk_text(text)) == 4


# ---------------------------------------------------------------------------
# chunk_text — long section splitting
# ---------------------------------------------------------------------------

class TestChunkTextLongSections:
    """Chunking of sections longer than MAX_CHARS: paragraph and word-boundary splits that keep
    every chunk within the limit."""

    def test_long_section_splits_at_paragraph_boundary(self):
        """Split a section longer than MAX_CHARS at a paragraph boundary with every chunk within the
        limit."""
        p1 = "Alpha " * 200   # 1200 chars
        p2 = "Beta " * 200    # 1000 chars
        body = p1 + "\n\n" + p2
        text = "# Section\n" + body
        chunks = chunk_text(text)
        assert len(chunks) >= 2
        assert all(len(c) <= MAX_CHARS for c in chunks)

    def test_all_paragraph_chunks_carry_parent_header(self):
        """Repeat the parent header in every chunk produced from a long section."""
        p1 = "Alpha " * 200
        p2 = "Beta " * 200
        text = "# Section\n" + p1 + "\n\n" + p2
        chunks = chunk_text(text)
        assert all("# Section" in c for c in chunks)

    def test_long_single_paragraph_splits_at_word_boundary(self):
        """Split a single paragraph longer than MAX_CHARS at word boundaries so every chunk fits."""
        big_body = "word " * 500  # 2500 chars, no paragraph breaks
        text = "# Big\n" + big_body
        chunks = chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= MAX_CHARS for c in chunks)

    def test_word_boundary_chunks_no_mid_word_cuts(self):
        """Split long text without cutting through a word."""
        # Each chunk should end on a complete word (or the header line)
        big_body = "word " * 500
        text = "# Big\n" + big_body
        chunks = chunk_text(text)
        for c in chunks:
            body_part = c.replace("# Big", "").strip()
            if body_part:
                # Every word in body_part should be a complete token
                for w in body_part.split():
                    assert w == "word" or w == "# Big" or w.startswith("#")

    def test_realistic_readme_all_within_max_chars(self):
        """Keep every chunk of a realistic multi-section document within MAX_CHARS."""
        def para(n):
            return " ".join(f"word{i}" for i in range(n))

        sections = []
        for i in range(5):
            sections.append(f"# Section {i}\n{para(300)}\n\n## Sub {i}\n{para(200)}")
        text = "\n\n".join(sections)
        chunks = chunk_text(text)
        assert chunks
        assert all(len(c) <= MAX_CHARS for c in chunks)


# ---------------------------------------------------------------------------
# chunk_text — code block protection
# ---------------------------------------------------------------------------

class TestChunkTextCodeBlocks:
    """Handling of fenced code blocks: they are preserved and never fragmented across chunks."""

    def test_short_code_block_preserved(self):
        """Preserve a short fenced code block and its content in the output."""
        text = "# API\nUse it like:\n```bash\nnpm install pkg\n```\nThat's it."
        chunks = chunk_text(text)
        all_text = "\n".join(chunks)
        assert "```bash" in all_text
        assert "npm install pkg" in all_text

    def test_code_block_appears_whole_in_one_chunk(self):
        """Keep a fenced code block intact in exactly one chunk."""
        code = "```python\n" + "x = 1\n" * 30 + "```"
        text = "# Setup\n" + code + "\nDone."
        chunks = chunk_text(text)
        # The fence must appear complete somewhere; not split across chunks
        assert any(code in c for c in chunks)
        assert sum(1 for c in chunks if "```python" in c) == 1

    def test_very_long_code_block_not_fragmented(self):
        """Open a very long fenced code block in exactly one chunk instead of fragmenting it."""
        # 150 lines — fence will exceed 500 chars but must stay together
        fence = "```bash\n" + ("echo hello_world\n" * 150) + "```"
        assert len(fence) > 500
        text = "# Code\n" + fence
        chunks = chunk_text(text)
        opening_count = sum(c.count("```bash") for c in chunks)
        closing_count = sum(c.count("```") for c in chunks)
        # Opening appears exactly once; closing appears at least once (may be in same chunk)
        assert opening_count == 1
        assert closing_count >= 1

    def test_multiple_code_blocks_each_preserved(self):
        """Preserve every code block when a document contains several."""
        block1 = "```python\nprint('hello')\n```"
        block2 = "```bash\nls -la\n```"
        text = f"# A\n{block1}\nMiddle text.\n\n## B\n{block2}\nEnd."
        chunks = chunk_text(text)
        all_text = "\n".join(chunks)
        assert "print('hello')" in all_text
        assert "ls -la" in all_text

    def test_code_block_between_paragraphs_not_disrupted(self):
        """Preserve a code block that sits between paragraphs, along with the surrounding text."""
        preamble = "Before code.\n\n"
        fence = "```\nsome code\n```"
        postamble = "\n\nAfter code."
        text = "# Section\n" + preamble + fence + postamble
        chunks = chunk_text(text)
        all_text = "\n".join(chunks)
        assert "some code" in all_text


# ---------------------------------------------------------------------------
# chunk_text — MAX_CHARS constant
# ---------------------------------------------------------------------------

def test_max_chars_is_2000():
    """Pin the maximum chunk size at 2000 characters."""
    assert MAX_CHARS == 2000
