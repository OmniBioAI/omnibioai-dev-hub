import re

MAX_CHARS = 2000

_FENCE_RE = re.compile(r'```.*?```', re.DOTALL)
_HEADER_RE = re.compile(r'^(#{1,3})\s+.+$', re.MULTILINE)


def _split_at_word_boundary(text: str, max_chars: int) -> list[str]:
    """Split text at word boundaries, keeping each piece ≤ max_chars."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    while len(text) > max_chars:
        cut = text.rfind(' ', 0, max_chars)
        if cut <= 0:
            cut = max_chars  # no space found — hard cut
        chunks.append(text[:cut])
        text = text[cut:].lstrip(' ')
    if text:
        chunks.append(text)
    return chunks


def _split_at_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split at blank-line boundaries; fall back to word-boundary for oversize paragraphs."""
    if max_chars <= 0:
        return _split_at_word_boundary(text, MAX_CHARS) if text.strip() else []
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    paragraphs = [p for p in re.split(r'\n\n+', text) if p.strip()]
    result: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            if current_parts:
                result.append('\n\n'.join(current_parts))
                current_parts = []
                current_len = 0
            result.extend(_split_at_word_boundary(para, max_chars))
        else:
            sep = 2 if current_parts else 0
            if current_parts and current_len + sep + len(para) > max_chars:
                result.append('\n\n'.join(current_parts))
                current_parts = [para]
                current_len = len(para)
            else:
                current_parts.append(para)
                current_len += sep + len(para)

    if current_parts:
        result.append('\n\n'.join(current_parts))

    return result


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """
    Markdown-structure-aware chunker. chunk_size is kept for API compatibility.

    1. Fenced code blocks (``` ```) are never split.
    2. Text is divided at H1/H2/H3 header lines as natural section boundaries.
    3. Sections longer than MAX_CHARS are split at paragraph boundaries (\\n\\n).
    4. Paragraphs longer than MAX_CHARS fall back to word-boundary splitting.
    5. Each chunk is prefixed with its ancestor header chain
       (e.g. "# H1 > ## H2\\n") so retrieval context carries section identity.
    """
    if not text:
        return []

    # Protect fenced code blocks — replace with non-splitting placeholders
    fences: list[str] = []

    def _stash(m: re.Match) -> str:
        fences.append(m.group(0))
        return f'\x00FENCE{len(fences) - 1}\x00'

    protected = _FENCE_RE.sub(_stash, text)

    # Split into sections: list of (header_line | None, body_str, header_level)
    sections: list[tuple] = []
    last_end = 0
    current_header: str | None = None
    current_level = 0

    for m in _HEADER_RE.finditer(protected):
        body_before = protected[last_end:m.start()]
        if current_header is not None or body_before.strip():
            sections.append((current_header, body_before, current_level))
        current_header = m.group(0)
        current_level = len(m.group(1))
        last_end = m.end()

    tail = protected[last_end:]
    if current_header is not None or tail.strip():
        sections.append((current_header, tail, current_level))

    # Build chunks, maintaining a header breadcrumb stack
    header_stack: list[str] = []
    all_chunks: list[str] = []

    for header_line, body, level in sections:
        if header_line is not None:
            # Pop siblings / children before appending this header
            header_stack = header_stack[:level - 1]
            header_stack.append(header_line)

        prefix = ' > '.join(header_stack) + '\n' if header_stack else ''
        body = body.strip()
        full = (prefix + body).strip()

        if not full:
            continue

        if len(full) <= MAX_CHARS:
            all_chunks.append(full)
            continue

        # Section too long — split body at paragraph boundaries
        budget = MAX_CHARS - len(prefix)
        sub_bodies = _split_at_paragraphs(body, budget)

        if sub_bodies:
            for sb in sub_bodies:
                chunk = (prefix + sb).strip()
                if chunk:
                    all_chunks.append(chunk)
        else:
            all_chunks.append(full)  # fallback: emit as-is

    # Restore fenced code blocks in every chunk
    restored: list[str] = []
    for chunk in all_chunks:
        for i, fence in enumerate(fences):
            chunk = chunk.replace(f'\x00FENCE{i}\x00', fence)
        restored.append(chunk)

    return [c for c in restored if c]
