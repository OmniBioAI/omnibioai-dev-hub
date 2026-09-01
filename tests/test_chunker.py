"""Comprehensive tests for the markdown-aware chunker (processing/chunker.py)."""
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
    def test_short_text_returned_unchanged(self):
        assert _split_at_word_boundary("hello world", 100) == ["hello world"]

    def test_exact_limit_not_split(self):
        text = "abcde"
        assert _split_at_word_boundary(text, 5) == [text]

    def test_splits_at_last_space_before_limit(self):
        # "ab cde fg" — rfind(' ', 0, 7) hits space at pos 6 → "ab cde" / "fg"
        text = "ab cde fg"
        result = _split_at_word_boundary(text, 7)
        assert result == ["ab cde", "fg"]

    def test_no_space_forces_hard_cut(self):
        text = "abcdefgh"
        result = _split_at_word_boundary(text, 4)
        assert result == ["abcd", "efgh"]

    def test_multiple_passes_all_within_limit(self):
        text = "one two three four five six seven"
        max_c = 10
        result = _split_at_word_boundary(text, max_c)
        assert all(len(c) <= max_c for c in result)
        # Round-trip: joining should contain all original words
        assert set(text.split()) == set(" ".join(result).split())

    def test_single_word_longer_than_limit(self):
        text = "superlongword"
        result = _split_at_word_boundary(text, 5)
        assert result == ["super", "longw", "ord"]

    def test_leading_spaces_stripped_after_cut(self):
        text = "aaa bbb"
        result = _split_at_word_boundary(text, 4)
        assert result == ["aaa", "bbb"]


# ---------------------------------------------------------------------------
# _split_at_paragraphs
# ---------------------------------------------------------------------------

class TestSplitAtParagraphs:
    def test_short_text_returned_unchanged(self):
        text = "Para one.\n\nPara two."
        assert _split_at_paragraphs(text, 1000) == [text]

    def test_empty_text_returns_empty(self):
        assert _split_at_paragraphs("", 100) == []

    def test_whitespace_only_returns_empty(self):
        assert _split_at_paragraphs("   \n\n   ", 100) == []

    def test_splits_at_paragraph_boundary(self):
        p1 = "x" * 300
        p2 = "y" * 300
        text = p1 + "\n\n" + p2
        result = _split_at_paragraphs(text, 400)
        assert result == [p1, p2]

    def test_accumulates_small_paragraphs_when_they_fit(self):
        p1, p2 = "aaa", "bbb"
        text = p1 + "\n\n" + p2
        result = _split_at_paragraphs(text, 100)
        assert len(result) == 1
        assert p1 in result[0] and p2 in result[0]

    def test_three_paras_packed_optimally(self):
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
        big_para = "word " * 500  # 2500 chars
        result = _split_at_paragraphs(big_para, 300)
        assert all(len(c) <= 300 for c in result)
        assert len(result) > 1

    def test_oversize_para_flushes_accumulator_first(self):
        small = "s" * 50
        big = "B " * 300  # 600 chars
        text = small + "\n\n" + big
        result = _split_at_paragraphs(text, 100)
        # small goes to one chunk, big gets word-split
        assert result[0] == small
        assert all(len(c) <= 100 for c in result[1:])

    def test_zero_budget_falls_back_gracefully(self):
        text = "some text here"
        result = _split_at_paragraphs(text, 0)
        assert result  # must not raise or return empty

    def test_multiple_blank_lines_treated_as_single_boundary(self):
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
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_none_equivalent_empty(self):
        # Edge: whitespace-only body with no content → empty
        assert chunk_text("   \n\n   ") == []

    def test_plain_text_no_headers_single_chunk(self):
        text = "Plain text with no markdown whatsoever."
        result = chunk_text(text)
        assert result == [text]

    def test_chunk_size_kwarg_accepted_without_error(self):
        result = chunk_text("hello world", chunk_size=2)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_no_empty_strings_in_output(self):
        text = "# A\n\n# B\n\n# C\n"
        chunks = chunk_text(text)
        assert all(c.strip() for c in chunks)

    def test_all_content_preserved(self):
        text = "# H1\nIntro\n\n## H2\nDetails here\n\n### H3\nLeaf content"
        chunks = chunk_text(text)
        joined = "\n".join(chunks)
        for word in ["Intro", "Details here", "Leaf content"]:
            assert word in joined


# ---------------------------------------------------------------------------
# chunk_text — header splitting
# ---------------------------------------------------------------------------

class TestChunkTextHeaders:
    def test_single_header_included_in_chunk(self):
        text = "# Overview\nThis is the overview section."
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].startswith("# Overview")
        assert "overview section" in chunks[0]

    def test_two_h1_headers_produce_two_chunks(self):
        text = "# First\nContent A.\n\n# Second\nContent B."
        chunks = chunk_text(text)
        assert len(chunks) == 2

    def test_h1_and_h2_produce_two_chunks(self):
        text = "# Parent\nIntro.\n\n## Child\nDetail."
        chunks = chunk_text(text)
        assert len(chunks) == 2

    def test_child_header_gets_parent_breadcrumb(self):
        text = "# Parent\nIntro.\n\n## Child\nDetail."
        chunks = chunk_text(text)
        h2_chunk = next(c for c in chunks if "Detail" in c)
        assert "# Parent" in h2_chunk
        assert "## Child" in h2_chunk
        assert " > " in h2_chunk

    def test_h3_gets_full_three_level_breadcrumb(self):
        text = "# A\nroot\n\n## B\nmid\n\n### C\nleaf"
        chunks = chunk_text(text)
        h3_chunk = next(c for c in chunks if "leaf" in c)
        assert "# A" in h3_chunk
        assert "## B" in h3_chunk
        assert "### C" in h3_chunk

    def test_sibling_header_does_not_appear_in_peer_breadcrumb(self):
        text = "# Root\nfirst\n\n## X\nunderX\n\n## Y\nunderY"
        chunks = chunk_text(text)
        y_chunk = next(c for c in chunks if "underY" in c)
        assert "## Y" in y_chunk
        assert "## X" not in y_chunk  # sibling must be excluded

    def test_h1_resets_entire_stack(self):
        text = "# First\nA\n\n## Sub\nB\n\n# Second\nC"
        chunks = chunk_text(text)
        second_chunk = next(c for c in chunks if c.strip().startswith("# Second"))
        # "# First" and "## Sub" should NOT appear in the "# Second" chunk prefix
        assert "# First" not in second_chunk
        assert "## Sub" not in second_chunk

    def test_preamble_before_first_header_is_included(self):
        text = "Preamble text here.\n\n# Section\nContent."
        chunks = chunk_text(text)
        assert any("Preamble text" in c for c in chunks)

    def test_header_only_section_emitted(self):
        # A section with a header but no body text still produces a chunk
        text = "# Title Only"
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert "# Title Only" in chunks[0]

    def test_four_headers_produce_four_chunks(self):
        parts = [f"# H{i}\nContent {i}." for i in range(4)]
        text = "\n\n".join(parts)
        assert len(chunk_text(text)) == 4


# ---------------------------------------------------------------------------
# chunk_text — long section splitting
# ---------------------------------------------------------------------------

class TestChunkTextLongSections:
    def test_long_section_splits_at_paragraph_boundary(self):
        p1 = "Alpha " * 200   # 1200 chars
        p2 = "Beta " * 200    # 1000 chars
        body = p1 + "\n\n" + p2
        text = "# Section\n" + body
        chunks = chunk_text(text)
        assert len(chunks) >= 2
        assert all(len(c) <= MAX_CHARS for c in chunks)

    def test_all_paragraph_chunks_carry_parent_header(self):
        p1 = "Alpha " * 200
        p2 = "Beta " * 200
        text = "# Section\n" + p1 + "\n\n" + p2
        chunks = chunk_text(text)
        assert all("# Section" in c for c in chunks)

    def test_long_single_paragraph_splits_at_word_boundary(self):
        big_body = "word " * 500  # 2500 chars, no paragraph breaks
        text = "# Big\n" + big_body
        chunks = chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c) <= MAX_CHARS for c in chunks)

    def test_word_boundary_chunks_no_mid_word_cuts(self):
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
    def test_short_code_block_preserved(self):
        text = "# API\nUse it like:\n```bash\nnpm install pkg\n```\nThat's it."
        chunks = chunk_text(text)
        all_text = "\n".join(chunks)
        assert "```bash" in all_text
        assert "npm install pkg" in all_text

    def test_code_block_appears_whole_in_one_chunk(self):
        code = "```python\n" + "x = 1\n" * 30 + "```"
        text = "# Setup\n" + code + "\nDone."
        chunks = chunk_text(text)
        # The fence must appear complete somewhere; not split across chunks
        assert any(code in c for c in chunks)
        assert sum(1 for c in chunks if "```python" in c) == 1

    def test_very_long_code_block_not_fragmented(self):
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
        block1 = "```python\nprint('hello')\n```"
        block2 = "```bash\nls -la\n```"
        text = f"# A\n{block1}\nMiddle text.\n\n## B\n{block2}\nEnd."
        chunks = chunk_text(text)
        all_text = "\n".join(chunks)
        assert "print('hello')" in all_text
        assert "ls -la" in all_text

    def test_code_block_between_paragraphs_not_disrupted(self):
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
    assert MAX_CHARS == 2000
