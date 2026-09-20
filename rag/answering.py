"""Grounded-answer contract for Ask OmniBioAI (Phase 19).

Pure logic, no network: the engine passes in a `generate` callable, so every
rule here is testable without an LLM.

CONTRACT (answer_contract "ask.v1")

    retrieval (PUBLIC only, relevance >= cutoff, done by the caller)
      -> if NO qualifying chunk: deterministic no-answer, the LLM is NEVER called
      -> pre-check: a name-like term in the question (iOS, Salesforce, REDCap...)
         that the retrieved documentation never mentions: deterministic
         no-answer, the LLM is NEVER called
      -> LLM, constrained to the numbered excerpts, told to cite [n] or reply
         INSUFFICIENT_CONTEXT
      -> verification: the answer is GROUNDED only if it is not the sentinel,
         is not a refusal, cites at least one supplied excerpt, cites no
         excerpt number that was not supplied, and states no name or number
         that appears in none of the supplied excerpts (invented specifics)
      -> otherwise a deterministic no-answer; the model's text is discarded

    A response is `grounded: true` only in the GROUNDED status. In every other
    status `answer` is a fixed system message, `citations` is [], and nothing the
    model said is shown as documentation.

Known limits (kept honest, see the tests): semantic similarity can score a
plausible-sounding but nonexistent capability highly. The deterministic
pre-check catches name-like terms only; a lowercase nonexistent capability
("voice assistant") relies on the model honouring the INSUFFICIENT_CONTEXT
instruction plus the citation/refusal checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

ANSWER_CONTRACT_VERSION = "ask.v1"

SENTINEL = "INSUFFICIENT_CONTEXT"

STATUS_GROUNDED = "GROUNDED"
STATUS_NO_TRUSTED_CONTEXT = "NO_TRUSTED_CONTEXT"
STATUS_INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
STATUS_UNSUPPORTED_TERM = "UNSUPPORTED_TERM"
STATUS_UNCITED_ANSWER = "UNCITED_ANSWER"
STATUS_INVALID_CITATIONS = "INVALID_CITATIONS"
STATUS_UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
STATUS_LLM_UNAVAILABLE = "LLM_UNAVAILABLE"

NO_CONTEXT_MESSAGE = (
    "No sufficiently relevant trusted OmniBioAI documentation was found for this question, "
    "so no answer is given. Ask OmniBioAI only answers from its published documentation."
)
NO_SUPPORT_MESSAGE = (
    "Documentation was found, but it does not contain a supported answer to this question, "
    "so no answer is given. Ask OmniBioAI only answers from what its documentation states."
)
LLM_UNAVAILABLE_MESSAGE = (
    "The answer service is temporarily unavailable, so no answer is given. Please try again."
)

NO_ANSWER_MESSAGES = {
    STATUS_NO_TRUSTED_CONTEXT: NO_CONTEXT_MESSAGE,
    STATUS_INSUFFICIENT_CONTEXT: NO_SUPPORT_MESSAGE,
    STATUS_UNSUPPORTED_TERM: NO_SUPPORT_MESSAGE,
    STATUS_UNCITED_ANSWER: NO_SUPPORT_MESSAGE,
    STATUS_INVALID_CITATIONS: NO_SUPPORT_MESSAGE,
    STATUS_UNSUPPORTED_CLAIM: NO_SUPPORT_MESSAGE,
    STATUS_LLM_UNAVAILABLE: LLM_UNAVAILABLE_MESSAGE,
}

# Deterministic, and explicit about the context window: Ollama's default is too
# small for five excerpts and would silently truncate the instructions.
GENERATION_OPTIONS = {"temperature": 0, "num_ctx": 8192}

CITATION_FIELDS = (
    "repository", "relative_path", "source_revision", "document_id", "chunk_id",
    "content_state", "verification_state",
)

_BRAND_PREFIX = "omnibioai"
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#./-]*[A-Za-z0-9+#]|[A-Za-z]")
_REFUSAL_RE = re.compile(
    r"(do(?:es)? not (?:mention|contain|include|provide|specify|state|describe|say)"
    r"|no (?:mention|information|reference|evidence|details?)"
    r"|not (?:mentioned|provided|specified|stated|documented|described|covered)"
    r"|(?:cannot|can't|could not|couldn't|unable to) (?:find|determine|answer|locate)"
    r"|(?:is|are|isn't|aren't) not (?:mentioned|documented|described|covered|available))",
    re.IGNORECASE,
)


def _lead_sentence(text: str) -> str:
    """A refusal LEADS the answer ("The excerpts do not mention X"); a trailing caveat
    ("Note: this does not mention Y") on an otherwise supported answer is not a refusal."""
    return re.split(r"(?<=[.!?])\s+|\n", text.strip(), maxsplit=1)[0][:300]


def dedupe_chunks(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order-preserving de-duplication by chunk_id (falls back to source+text)."""
    seen: set[Any] = set()
    out = []
    for d in docs:
        key = d.get("chunk_id") or (d.get("source"), d.get("text"))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _cite_field(doc: dict[str, Any], key: str) -> Any:
    citation = doc.get("citation") or {}
    return citation.get(key, doc.get("repo") if key == "repository" else doc.get(key))


def build_grounded_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """The generation prompt: numbered excerpts, and nothing else the model may rely on."""
    excerpts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {_cite_field(c, 'relative_path') or c.get('source', 'unknown')}"
        state = f"content_state={_cite_field(c, 'content_state')}, verification_state={_cite_field(c, 'verification_state')}"
        excerpts.append(f"{header} ({state})\n{c.get('text', '')}")
    body = "\n\n".join(excerpts)
    return f"""You are the OmniBioAI documentation assistant. You answer ONLY from the numbered documentation excerpts below.

RULES
1. Use only facts stated in the excerpts. Do not use outside knowledge, do not guess, do not extrapolate, and do not infer that a feature exists because something similar does.
2. Cite every claim with the number of the excerpt that states it, in square brackets, e.g. [1] or [2][3]. Never cite a number that is not listed below.
3. If the excerpts contain relevant information, answer using only what they state and say plainly which part of the question they do not cover. Reply exactly {SENTINEL} only when the excerpts contain nothing relevant, or when the question is about a specific product, feature, app or capability that the excerpts never mention -- a similar-sounding feature does not count.
4. If excerpts disagree, say so and cite each.
5. Do not describe something as verified, supported or operational unless an excerpt says so; respect each excerpt's content_state and verification_state.
6. Be concise and technical.

EXAMPLES (unrelated to the excerpts below)
Excerpt [1] backups.md: Databases are backed up nightly to object storage. Restore with the recovery tool.
Question: How do I back up the database?
Answer: Databases are backed up nightly to object storage [1]. To restore, use the recovery tool [1].

Excerpt [1] backups.md: Databases are backed up nightly to object storage. Restore with the recovery tool.
Question: How do I use the hologram viewer for backups?
Answer: {SENTINEL}

EXCERPTS
{body}

QUESTION
{query}

ANSWER (cited, or exactly {SENTINEL}):"""


def unsupported_terms(query: str, chunks: list[dict[str, Any]]) -> list[str]:
    """Name-like terms in the question that no supplied excerpt mentions.

    Names (iOS, Salesforce, REDCap, S3, GitHub...) cannot be paraphrased, so a
    name the retrieved documentation never uses means the question is about
    something the documentation does not cover, however similar the surrounding
    words scored. Only name-like tokens are checked: mixed/camel case, ALL CAPS,
    digit-bearing, or capitalised mid-sentence. The OmniBioAI brand is exempt.
    """
    haystack = " ".join(
        f"{c.get('text', '')} {_cite_field(c, 'relative_path') or ''} {c.get('title', '')}" for c in chunks
    ).lower()
    missing: list[str] = []
    for m in _TOKEN_RE.finditer(query):
        token = m.group(0)
        prefix = query[: m.start()]
        between = prefix.rstrip()
        # a capitalised word starting a sentence, bullet or line is not evidence of a name
        sentence_start = (not between) or between[-1] in ".?!:;-*>#" or "\n" in prefix[len(between):]
        low = token.lower()
        if low.startswith(_BRAND_PREFIX) or len(token) < 2:
            continue
        name_like = (
            any(ch.isupper() for ch in token[1:])          # iOS, REDCap, GitHub
            or (token.isupper() and len(token) >= 2)       # AWS, API
            or any(ch.isdigit() for ch in token)           # S3, v2
            or (token[0].isupper() and not sentence_start)  # Salesforce mid-sentence
        )
        if not name_like:
            continue
        variants = {low, low.removesuffix("s"), low.removesuffix("'s")}
        mentioned = any(re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", haystack) for v in variants if v)
        if not mentioned and low not in missing:
            missing.append(low)
    return missing


_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9._-])\d+(?:[.,]\d+)*(?![A-Za-z0-9])")
_MARKUP_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]|`+|\*+|_{2,}|\]\([^)]*\)|https?://\S+")


def unsupported_answer_terms(answer: str, chunks: list[dict[str, Any]], query: str) -> list[str]:
    """Names and numbers the ANSWER states that appear in none of the supplied excerpts (or the question).

    Citations prove the model pointed at an excerpt, not that what it said is in
    it. A name or number found nowhere in the supplied text is outside knowledge
    or invention ("retention is 30 days", "runs on AWS Lambda"), so the answer is
    refused. Same name-like rules as unsupported_terms; markdown and citation
    markers are stripped first.
    """
    cleaned = _MARKUP_RE.sub(" ", answer)
    support = [*chunks, {"text": query}]
    missing = unsupported_terms(cleaned, support)
    haystack = " ".join(f"{c.get('text', '')} {c.get('title', '')}" for c in support).lower()
    body = re.sub(r"(?m)^\s*\d+[.)]\s", " ", cleaned)  # "1. step" list markers are not claims
    for num in _NUMBER_RE.findall(body):
        stated = num.rstrip(".,")
        if len(stated) < 2 and "." not in stated:  # single digits are step numbers / ordinals, not claims
            continue
        if not re.search(rf"(?<![0-9]){re.escape(stated)}(?![0-9])", haystack) and stated not in missing:
            missing.append(stated)
    return missing


def parse_citation_markers(text: str) -> list[int]:
    """All [n] / [n, m] citation numbers in the text, in order of appearance."""
    numbers: list[int] = []
    for group in _MARKER_RE.findall(text):
        numbers.extend(int(x) for x in re.split(r"\s*,\s*", group))
    return numbers


def verify_answer(raw: str, n_chunks: int) -> tuple[str, list[int]]:
    """Decide whether the model's text may be shown as a grounded answer.

    Returns (status, cited_indices). Only STATUS_GROUNDED lets the text through.
    """
    text = (raw or "").strip()
    if not text or SENTINEL in text.upper().replace(" ", "_"):
        return STATUS_INSUFFICIENT_CONTEXT, []
    if _REFUSAL_RE.search(_lead_sentence(text)):
        return STATUS_INSUFFICIENT_CONTEXT, []
    cited = parse_citation_markers(text)
    if not cited:
        return STATUS_UNCITED_ANSWER, []
    if any(n < 1 or n > n_chunks for n in cited):
        return STATUS_INVALID_CITATIONS, []
    return STATUS_GROUNDED, sorted(set(cited))


def build_citations(chunks: list[dict[str, Any]], cited: list[int]) -> list[dict[str, Any]]:
    """Citations for exactly the excerpts the answer cites; `index` matches the [n] in the text."""
    out = []
    for n in cited:
        chunk = chunks[n - 1]
        entry: dict[str, Any] = {"index": n, **{k: _cite_field(chunk, k) for k in CITATION_FIELDS}}
        entry["title"] = chunk.get("title")
        entry["heading"] = chunk.get("heading")
        relevance = chunk.get("relevance")
        entry["relevance"] = round(relevance, 3) if isinstance(relevance, float) else None
        out.append(entry)
    return out


def _result(query: str, *, answer: str, status: str, grounded: bool, citations: list[dict[str, Any]],
            context_used: int, llm_invoked: bool, context: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "query": query,
        "answer": answer,
        "grounded": grounded,
        "answer_status": status,
        "answer_contract": ANSWER_CONTRACT_VERSION,
        "citations": citations,
        "context_used": context_used,
        "llm_invoked": llm_invoked,
        "sources": [c.get("source") for c in context],
        "context": context,
        "version": "v6-faiss",
        **(extra or {}),
    }


def no_answer(query: str, status: str, *, context_used: int = 0, llm_invoked: bool = False,
              context: list[dict[str, Any]] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deterministic no-grounded-answer response. Never carries citations."""
    return _result(query, answer=NO_ANSWER_MESSAGES[status], status=status, grounded=False, citations=[],
                   context_used=context_used, llm_invoked=llm_invoked, context=context or [], extra=extra)


def generate_grounded_answer(query: str, docs: list[dict[str, Any]], generate: Callable[[str], str]) -> dict[str, Any]:
    """Apply the whole contract to already-retrieved, already-policy-filtered `docs`."""
    chunks = dedupe_chunks(docs)
    if not chunks:
        return no_answer(query, STATUS_NO_TRUSTED_CONTEXT)  # the LLM is never reached

    missing = unsupported_terms(query, chunks)
    if missing:
        return no_answer(query, STATUS_UNSUPPORTED_TERM, context_used=len(chunks), context=chunks,
                         extra={"unsupported_terms": missing})

    try:
        raw = generate(build_grounded_prompt(query, chunks))
    except Exception as exc:  # noqa: BLE001 -- LLM boundary: never surface exception text or invent an answer
        return no_answer(query, STATUS_LLM_UNAVAILABLE, context_used=len(chunks), llm_invoked=True,
                         context=chunks, extra={"error_class": type(exc).__name__})

    status, cited = verify_answer(raw, len(chunks))
    if status != STATUS_GROUNDED:
        return no_answer(query, status, context_used=len(chunks), llm_invoked=True, context=chunks)
    invented = unsupported_answer_terms(raw, chunks, query)
    if invented:
        return no_answer(query, STATUS_UNSUPPORTED_CLAIM, context_used=len(chunks), llm_invoked=True, context=chunks,
                         extra={"unsupported_claim_terms": invented})
    return _result(query, answer=raw.strip(), status=STATUS_GROUNDED, grounded=True,
                   citations=build_citations(chunks, cited), context_used=len(chunks), llm_invoked=True, context=chunks)
