"""Answer-contract tests (rag/answering.py): pure logic, no network, no LLM."""

from unittest.mock import MagicMock

import pytest

from rag.answering import (
    ANSWER_CONTRACT_VERSION,
    NO_CONTEXT_MESSAGE,
    NO_SUPPORT_MESSAGE,
    SENTINEL,
    STATUS_GROUNDED,
    STATUS_INSUFFICIENT_CONTEXT,
    STATUS_INVALID_CITATIONS,
    STATUS_LLM_UNAVAILABLE,
    STATUS_NO_TRUSTED_CONTEXT,
    STATUS_UNCITED_ANSWER,
    STATUS_UNSUPPORTED_TERM,
    build_grounded_prompt,
    dedupe_chunks,
    generate_grounded_answer,
    parse_citation_markers,
    unsupported_terms,
    verify_answer,
)


def chunk(n, text="Back up the database nightly and restore with the recovery tool.", path=None, **extra):
    path = path or f"site/docs/admin/doc{n}.md"
    return {
        "text": text, "source": f"omnibioai-docs:{path}@rev{n}", "repo": "omnibioai-docs", "relative_path": path,
        "source_revision": f"rev{n}", "document_id": f"doc{n}", "chunk_id": f"chunk{n}", "content_state": "CURRENT",
        "verification_state": "CONFIGURED", "title": f"Doc {n}", "heading": None, "relevance": 0.7123,
        "citation": {"repository": "omnibioai-docs", "relative_path": path, "source_revision": f"rev{n}",
                     "document_id": f"doc{n}", "chunk_id": f"chunk{n}", "content_state": "CURRENT",
                     "verification_state": "CONFIGURED"},
        **extra,
    }


# ---- 19A: no context -> deterministic answer, LLM never invoked ------------------------------

def test_zero_context_returns_deterministic_no_answer_and_never_calls_the_llm():
    llm = MagicMock(return_value="A confident hallucination [1].")
    r = generate_grounded_answer("what is X?", [], llm)
    llm.assert_not_called()
    assert r["answer"] == NO_CONTEXT_MESSAGE
    assert (r["grounded"], r["answer_status"], r["context_used"], r["citations"], r["llm_invoked"]) == (
        False, STATUS_NO_TRUSTED_CONTEXT, 0, [], False)
    assert r["answer_contract"] == ANSWER_CONTRACT_VERSION


def test_no_context_message_says_no_trusted_documentation_was_found():
    assert "No sufficiently relevant trusted OmniBioAI documentation was found" in NO_CONTEXT_MESSAGE


def test_the_no_context_response_is_identical_for_every_question():
    llm = MagicMock()
    a = generate_grounded_answer("q1", [], llm)
    b = generate_grounded_answer("something else entirely", [], llm)
    assert {k: v for k, v in a.items() if k != "query"} == {k: v for k, v in b.items() if k != "query"}


# ---- 19B: the answer contract -----------------------------------------------------------------

def test_grounded_answer_requires_a_valid_citation_and_returns_only_cited_chunks():
    docs = [chunk(1), chunk(2), chunk(3)]
    llm = MagicMock(return_value="Back up nightly [1] and restore with the recovery tool [3].")
    r = generate_grounded_answer("how do I back up and restore?", docs, llm)
    assert r["grounded"] and r["answer_status"] == STATUS_GROUNDED and r["llm_invoked"]
    assert [c["index"] for c in r["citations"]] == [1, 3]  # chunk 2 was supplied but not cited
    assert [c["chunk_id"] for c in r["citations"]] == ["chunk1", "chunk3"]
    assert r["context_used"] == 3


def test_prompt_contains_the_rules_the_sentinel_and_only_the_supplied_excerpts():
    docs = [chunk(1, text="ALPHA excerpt text"), chunk(2, text="BETA excerpt text")]
    prompt = build_grounded_prompt("my question?", docs)
    assert "ONLY from the numbered documentation excerpts" in prompt
    assert "Do not use outside knowledge" in prompt and "do not infer that a feature exists" in prompt
    assert f"reply with exactly: {SENTINEL}" in prompt
    assert "[1] site/docs/admin/doc1.md" in prompt and "[2] site/docs/admin/doc2.md" in prompt
    assert "ALPHA excerpt text" in prompt and "BETA excerpt text" in prompt and "my question?" in prompt
    assert "If excerpts disagree, say so and cite each" in prompt  # conflicting evidence rule
    assert "content_state=CURRENT, verification_state=CONFIGURED" in prompt


def test_prompt_never_contains_text_that_was_not_supplied():
    prompt = build_grounded_prompt("q", [chunk(1, text="only this")])
    assert "doc2" not in prompt and "chunk2" not in prompt


def test_the_sentinel_yields_no_grounded_answer_and_discards_the_model_text():
    r = generate_grounded_answer("q?", [chunk(1)], MagicMock(return_value=SENTINEL))
    assert (r["grounded"], r["answer_status"], r["answer"], r["citations"]) == (False, STATUS_INSUFFICIENT_CONTEXT, NO_SUPPORT_MESSAGE, [])
    assert r["llm_invoked"] and r["context_used"] == 1


@pytest.mark.parametrize("text,status", [
    ("Restore it with the tool.", STATUS_UNCITED_ANSWER),                       # fluent but uncited
    ("Do this [9].", STATUS_INVALID_CITATIONS),                                # cites an excerpt that was not supplied
    ("Do this [0].", STATUS_INVALID_CITATIONS),
    ("Do this [1] and that [2].", STATUS_INVALID_CITATIONS),                   # only 1 chunk supplied
    ("The excerpts do not mention this [1].", STATUS_INSUFFICIENT_CONTEXT),    # refusal dressed with a citation
    ("There is no information about that in the documentation [1].", STATUS_INSUFFICIENT_CONTEXT),
    ("I could not find anything about it.", STATUS_INSUFFICIENT_CONTEXT),
    ("insufficient context", STATUS_INSUFFICIENT_CONTEXT),
    ("", STATUS_INSUFFICIENT_CONTEXT),
])
def test_ungrounded_model_output_is_never_shown(text, status):
    r = generate_grounded_answer("q?", [chunk(1)], MagicMock(return_value=text))
    assert not r["grounded"] and r["answer_status"] == status
    assert r["answer"] == NO_SUPPORT_MESSAGE and r["citations"] == []
    assert text.strip() == "" or text not in r["answer"]


def test_a_long_answer_with_one_caveat_is_not_treated_as_a_refusal():
    text = ("Back up nightly [1]. " * 30) + "The documentation does not specify the retention period."
    assert len(text) > 400
    assert verify_answer(text, 1)[0] == STATUS_GROUNDED


def test_citation_markers_accept_comma_lists():
    assert parse_citation_markers("A [1, 2] B [3][4].") == [1, 2, 3, 4]
    assert verify_answer("A [1, 2].", 2) == (STATUS_GROUNDED, [1, 2])
    assert verify_answer("A [1, 5].", 2)[0] == STATUS_INVALID_CITATIONS


def test_llm_failure_is_a_deterministic_error_state_not_a_fabricated_answer_or_leaked_exception():
    r = generate_grounded_answer("q?", [chunk(1)], MagicMock(side_effect=ConnectionError("secret-host:11434 refused")))
    assert not r["grounded"] and r["answer_status"] == STATUS_LLM_UNAVAILABLE and r["citations"] == []
    assert "secret-host" not in r["answer"] and "secret-host" not in str(r["context"]) and r["error_class"] == "ConnectionError"


# ---- 19C: citations ----------------------------------------------------------------------------

def test_citations_carry_full_provenance_and_match_the_supplied_chunks():
    docs = [chunk(1), chunk(2)]
    r = generate_grounded_answer("q?", docs, MagicMock(return_value="Fact [2]."))
    (c,) = r["citations"]
    assert c["index"] == 2
    assert {k: c[k] for k in ("repository", "relative_path", "source_revision", "document_id", "chunk_id",
                              "content_state", "verification_state")} == {
        "repository": "omnibioai-docs", "relative_path": "site/docs/admin/doc2.md", "source_revision": "rev2",
        "document_id": "doc2", "chunk_id": "chunk2", "content_state": "CURRENT", "verification_state": "CONFIGURED"}
    assert c["relevance"] == 0.712 and c["title"] == "Doc 2"


def test_every_citation_corresponds_to_a_chunk_actually_supplied_to_generation():
    docs = [chunk(i) for i in range(1, 6)]
    seen = {}
    r = generate_grounded_answer("q?", docs, lambda p: seen.setdefault("prompt", p) and "Yes [1][4][5].")
    for c in r["citations"]:
        assert f"[{c['index']}] {c['relative_path']}" in seen["prompt"]
        assert docs[c["index"] - 1]["chunk_id"] == c["chunk_id"]


def test_duplicate_chunks_are_supplied_and_cited_once():
    docs = [chunk(1), chunk(1), chunk(2)]
    assert [c["chunk_id"] for c in dedupe_chunks(docs)] == ["chunk1", "chunk2"]
    r = generate_grounded_answer("q?", docs, MagicMock(return_value="A [1] B [2]."))
    assert r["context_used"] == 2 and [c["chunk_id"] for c in r["citations"]] == ["chunk1", "chunk2"]


def test_no_answer_states_never_manufacture_citations():
    for llm_text in (SENTINEL, "uncited words", "bad [7]"):
        assert generate_grounded_answer("q?", [chunk(1)], MagicMock(return_value=llm_text))["citations"] == []


def test_citation_fields_are_never_invented_when_absent():
    bare = {"text": "Back up nightly.", "source": "s", "chunk_id": "c1"}
    (c,) = generate_grounded_answer("q?", [bare], MagicMock(return_value="X [1]."))["citations"]
    assert c["repository"] is None and c["source_revision"] is None and c["content_state"] is None


# ---- 19E: plausible nonexistent capabilities -----------------------------------------------------

@pytest.mark.parametrize("question,term", [
    ("What does the OmniBioAI iOS mobile app do?", "ios"),
    ("How do I install the OmniBioAI iOS app?", "ios"),
    ("How do I configure the Salesforce CRM integration?", "salesforce"),
    ("Does OmniBioAI support Snowflake?", "snowflake"),
])
def test_a_nonexistent_named_capability_is_refused_without_calling_the_llm(question, term):
    """High semantic similarity must not become a product claim: the named thing is not in the docs."""
    docs = [chunk(1, text="OmniBioAI runs on Linux. Scenarios are described in the user guide. Mobile users can open the web UI.")]
    llm = MagicMock(return_value="OmniBioAI has an iOS app [1].")
    r = generate_grounded_answer(question, docs, llm)
    llm.assert_not_called()
    assert not r["grounded"] and r["answer_status"] == STATUS_UNSUPPORTED_TERM
    assert term in r["unsupported_terms"] and r["citations"] == [] and r["context_used"] == 1


def test_substring_matches_do_not_count_as_mentions():
    """'ios' inside 'scenarios' is not a mention of iOS."""
    assert unsupported_terms("Is there an iOS app?", [chunk(1, text="Several scenarios are covered.")]) == ["ios"]


def test_names_the_docs_do_use_are_supported_and_reach_generation():
    docs = [chunk(1, text="Connect REDCap with an API token. Results can be stored in S3 or on Azure Blob Storage.")]
    assert unsupported_terms("How do I connect REDCap and S3?", docs) == []
    assert unsupported_terms("How do I use the REDCap integration?", docs) == []


def test_ordinary_words_and_the_brand_are_not_treated_as_names():
    docs = [chunk(1, text="Restore procedures are documented.")]
    assert unsupported_terms("How do I restore the platform after a disaster?", docs) == []
    assert unsupported_terms("What is OmniBioAI and how does OmniBioAI-Studio work?", docs) == []


def test_a_lowercase_nonexistent_capability_relies_on_the_model_and_the_verifier():
    """No name-like token, so the pre-check passes: the model's INSUFFICIENT_CONTEXT is what protects us."""
    docs = [chunk(1, text="Ask the assistant about your data. Audio is not covered.")]
    assert unsupported_terms("how do I use the voice assistant", docs) == []
    r = generate_grounded_answer("how do I use the voice assistant", docs, MagicMock(return_value=SENTINEL))
    assert not r["grounded"] and r["answer_status"] == STATUS_INSUFFICIENT_CONTEXT


# ---- weak / conflicting evidence ----------------------------------------------------------------

def test_conflicting_excerpts_can_be_reported_with_both_citations():
    docs = [chunk(1, text="Retention is 30 days."), chunk(2, text="Retention is 90 days.")]
    r = generate_grounded_answer("What is the retention period?", docs,
                                MagicMock(return_value="The documentation conflicts: 30 days [1] versus 90 days [2]."))
    assert r["grounded"] and [c["index"] for c in r["citations"]] == [1, 2]


def test_weak_evidence_is_refused_not_guessed():
    r = generate_grounded_answer("What is the retention period?", [chunk(1, text="Backups run nightly.")],
                                MagicMock(return_value=SENTINEL))
    assert not r["grounded"] and r["answer"] == NO_SUPPORT_MESSAGE
