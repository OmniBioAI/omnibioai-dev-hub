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
    STATUS_UNSUPPORTED_CLAIM,
    STATUS_UNSUPPORTED_TERM,
    build_grounded_prompt,
    build_judge_prompt,
    dedupe_chunks,
    generate_grounded_answer,
    judge_says_supported,
    parse_citation_markers,
    unsupported_answer_terms,
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
    assert f"Reply exactly {SENTINEL} only when the excerpts contain nothing relevant" in prompt
    assert "a similar-sounding feature does not count" in prompt
    assert "say plainly which part of the question they do not cover" in prompt
    assert "[1] site/docs/admin/doc1.md" in prompt and "[2] site/docs/admin/doc2.md" in prompt
    assert "ALPHA excerpt text" in prompt and "BETA excerpt text" in prompt and "my question?" in prompt
    assert "If excerpts disagree, say so and cite each" in prompt  # conflicting evidence rule
    assert "content_state=CURRENT, verification_state=CONFIGURED" in prompt
    assert "never reuse their content" in prompt


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


def test_a_trailing_caveat_on_a_supported_answer_is_not_treated_as_a_refusal():
    long_text = ("Back up nightly [1]. " * 30) + "The documentation does not specify the retention period."
    assert verify_answer(long_text, 1)[0] == STATUS_GROUNDED
    short = "Every service exposes a basic health endpoint. [1]\n\nNote: this answer does not mention any other features."
    assert verify_answer(short, 1)[0] == STATUS_GROUNDED  # real-model output that an earlier version wrongly refused


def test_a_refusal_that_leads_the_answer_is_still_refused_even_with_later_content():
    assert verify_answer("The excerpts do not mention an iOS app. They describe backups [1].", 1)[0] == STATUS_INSUFFICIENT_CONTEXT


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


# ---- answer-side check: invented names / numbers ------------------------------------------------------

def test_invented_numbers_and_names_in_a_cited_answer_are_refused():
    docs = [chunk(1, text="Backups run nightly to object storage.")]
    for invented in ("Backups are kept for 30 days [1].", "Backups run nightly on AWS Lambda [1].", "Backups use Veeam [1]."):
        r = generate_grounded_answer("How are backups done?", docs, MagicMock(return_value=invented))
        assert not r["grounded"] and r["answer_status"] == STATUS_UNSUPPORTED_CLAIM and r["citations"] == [], invented
        assert r["unsupported_claim_terms"]


def test_specifics_that_are_in_the_excerpts_or_the_question_are_not_flagged():
    docs = [chunk(1, text="Backups run nightly to S3. Retention is 30 days. Use the REDCap importer.")]
    ok = "Backups run nightly to S3 [1]; retention is 30 days [1]. The REDCap importer restores them [1]."
    r = generate_grounded_answer("How are backups done?", docs, MagicMock(return_value=ok))
    assert r["grounded"], r
    assert unsupported_answer_terms("For Zenodo it is 30 days [1].", docs, "What about Zenodo?") == []  # named in the question


def test_capitalised_sentence_and_bullet_starts_are_not_treated_as_invented_names():
    docs = [chunk(1, text="backups run nightly. restore uses the recovery tool.")]
    text = "Backups run nightly [1].\n- Restore uses the recovery tool [1]\n- Verify afterwards [1]\nTherefore, use the tool [1]."
    assert unsupported_answer_terms(text, docs, "how do I back up?") == []


def test_markdown_links_code_and_urls_are_ignored_by_the_answer_check():
    docs = [chunk(1, text="Read the deployment page for details.")]
    text = "See [Deployment](./deployment.md) and https://example.org/Docs for `Details` [1]."
    assert unsupported_answer_terms(text, docs, "q") == []


def test_step_numbers_and_single_digits_are_not_claims_but_multi_digit_numbers_are():
    docs = [chunk(1, text="Backups run nightly. Restore uses the recovery tool.")]
    steps = "1. Back up nightly [1]\n2. Restore with the recovery tool [1]\nUse it 3 times [1]."
    assert unsupported_answer_terms(steps, docs, "q") == []
    assert set(unsupported_answer_terms("Retention is 90 days [1] and 2.5 GB.", docs, "q")) >= {"90", "2.5"}


def test_prompt_examples_use_invented_vocabulary_that_cannot_be_mistaken_for_documentation():
    prompt = build_grounded_prompt("q", [chunk(1)])
    for term in ("zorblax", "quuxctl", "glimmerfrost", "4471"):
        assert term in prompt  # the examples are present ...
    for domain_word in ("backup", "nightly", "object storage", "recovery tool", "database"):
        example_block = prompt.split("EXAMPLES")[1].split("EXCERPTS")[0]
        assert domain_word not in example_block.lower()  # ... and share no vocabulary with the platform docs


def test_an_answer_that_repeats_the_prompt_examples_is_refused_as_leakage():
    r = generate_grounded_answer("What runs on a port?", [chunk(1, text="Services expose health endpoints.")],
                                 MagicMock(return_value="The zorblax service listens on port 4471 [1]."))
    assert not r["grounded"] and r["answer_status"] == STATUS_UNSUPPORTED_CLAIM and "zorblax" in r["unsupported_claim_terms"]


def test_example_terms_are_fine_when_the_excerpts_or_question_really_contain_them():
    docs = [chunk(1, text="The zorblax service listens on port 4471 in this deployment.")]
    r = generate_grounded_answer("Which port does zorblax use?", docs, MagicMock(return_value="It listens on port 4471 [1]."))
    assert r["grounded"]


# ---- entailment judge (unit level) ---------------------------------------------------------------------

def test_judge_verdict_fails_closed():
    assert judge_says_supported("SUPPORTED") and judge_says_supported(" supported. ")
    for bad in ("UNSUPPORTED", "unsupported", "Probably supported", "SUPPORTED, mostly", "", None, "NOT SUPPORTED"):
        assert not judge_says_supported(bad), bad


def test_judge_prompt_shows_only_the_supplied_excerpts_and_the_answer_without_citation_markers():
    docs = [chunk(1, text="ALPHA fact"), chunk(2, text="BETA fact")]
    prompt = build_judge_prompt("It is ALPHA [1][2].", docs)
    assert "[1] ALPHA fact" in prompt and "[2] BETA fact" in prompt and "It is ALPHA ." in prompt
    assert "Reply with exactly one word: SUPPORTED or UNSUPPORTED" in prompt


def test_judge_runs_last_and_only_after_the_cheaper_checks_pass():
    calls = []
    judge = MagicMock(side_effect=lambda p: calls.append(p) or "SUPPORTED")
    docs = [chunk(1)]
    ok = generate_grounded_answer("q?", docs, MagicMock(return_value="Back up nightly [1]."), judge=judge)
    assert ok["grounded"] and ok["entailment_checked"] and len(calls) == 1
    for bad in (SENTINEL, "uncited", "bad [9]", "The zorblax service listens on port 4471 [1]."):
        judge.reset_mock()
        r = generate_grounded_answer("q?", docs, MagicMock(return_value=bad), judge=judge)
        assert not r["grounded"]
        judge.assert_not_called()


def test_judge_rejection_yields_unsupported_claim_with_no_citations():
    r = generate_grounded_answer("q?", [chunk(1)], MagicMock(return_value="Back up nightly [1]."), judge=MagicMock(return_value="UNSUPPORTED"))
    assert (r["grounded"], r["answer_status"], r["citations"], r["entailment"]) == (False, STATUS_UNSUPPORTED_CLAIM, [], "UNSUPPORTED")


# ---- Phase 19.1: the "ANSWER:" protocol label is not an invented name ------------------------------

# Real model output that was wrongly refused in production: faithful document content, plus a trailing
# echo of the prompt's closing "ANSWER (...):" label.
DR_EXCERPT_1 = ("No subsystem currently has a fully proven, off-site or off-primary-disk backup. Most backups that do exist "
                "are stored on the same physical disk as the data they protect. A tool exists to check backup freshness for "
                "the primary database, but it is not yet running on a schedule; detecting a failed backup currently depends "
                "on someone checking manually.")
DR_EXCERPT_2 = ("A restore having succeeded once does not mean it will succeed automatically or repeatedly. "
                "No formal recovery-time or recovery-point objectives are defined. A failed component is restarted in place.")
DR_ANSWER = ("No subsystem currently has a fully proven, off-site or off-primary-disk backup [1]. Most backups that do exist are "
             "stored on the same physical disk as the data they protect [1].\n\nA restore having succeeded once does not mean it "
             "will succeed repeatedly [2].\n\nANSWER: No subsystem currently has a fully proven, off-site or off-primary-disk "
             "backup. Most backups that do exist are stored on the same physical disk as the data they protect [1].")


def _dr_docs():
    return [chunk(1, text=DR_EXCERPT_1, path="site/docs/admin/disaster-recovery.md"),
            chunk(2, text=DR_EXCERPT_2, path="site/docs/admin/disaster-recovery.md")]


def test_the_answer_label_is_protocol_syntax_not_an_unsupported_name():
    docs = [chunk(1, text="Backups are stored on the same disk.")]
    assert unsupported_answer_terms("Backups are stored on the same disk [1].\n\nANSWER: Backups are stored on the same disk [1].", docs, "q") == []
    assert unsupported_answer_terms("ANSWER (cited, or exactly INSUFFICIENT_CONTEXT): Backups are stored on the same disk [1].", docs, "q") == []


def test_only_the_defined_label_forms_are_exempt_at_line_start():
    docs = [chunk(1, text="Backups are stored on the same disk.")]
    # not at the start of a line -> still a name-like token that is absent from the excerpts
    assert unsupported_answer_terms("The ANSWER is that backups are stored on the same disk [1].", docs, "q") == ["answer"]
    # other colon-terminated / all-caps labels are NOT exempt
    for label in ("NOTE", "WARNING", "SUMMARY", "IMPORTANT", "RESULT"):
        assert unsupported_answer_terms(f"Backups are stored on the same disk [1].\n{label}: see above [1].", docs, "q") == [label.lower()], label
    # a parenthetical other than the exact echoed hint is not swallowed by the label rule (it could hide a name)
    assert set(unsupported_answer_terms("ANSWER (Veeam): Backups are stored on the same disk [1].", docs, "q")) >= {"veeam"}


def test_unsupported_names_are_still_rejected_after_the_label():
    docs = [chunk(1, text="Backups are stored on the same disk.")]
    assert unsupported_answer_terms("ANSWER: Backups use Veeam [1].", docs, "q") == ["veeam"]
    r = generate_grounded_answer("q?", docs, MagicMock(return_value="ANSWER: Backups use Veeam [1]."))
    assert not r["grounded"] and r["answer_status"] == STATUS_UNSUPPORTED_CLAIM and "veeam" in r["unsupported_claim_terms"]


def test_the_known_disaster_recovery_answer_now_reaches_the_normal_grounding_checks():
    judge = MagicMock(return_value="SUPPORTED")
    r = generate_grounded_answer("how do I back up and restore the platform after a disaster", _dr_docs(),
                                 MagicMock(return_value=DR_ANSWER), judge=judge)
    assert r["grounded"] and r["answer_status"] == STATUS_GROUNDED and r["entailment_checked"]
    judge.assert_called_once()  # it went THROUGH the fact-check rather than being refused before it
    assert unsupported_answer_terms(DR_ANSWER, _dr_docs(), "how do I back up and restore the platform after a disaster") == []


def test_the_fact_check_stays_active_for_label_bearing_answers():
    r = generate_grounded_answer("q?", _dr_docs(), MagicMock(return_value=DR_ANSWER), judge=MagicMock(return_value="UNSUPPORTED"))
    assert not r["grounded"] and r["answer_status"] == STATUS_UNSUPPORTED_CLAIM and r["entailment"] == "UNSUPPORTED" and r["citations"] == []


def test_citations_stay_constrained_to_supplied_excerpts_for_label_bearing_answers():
    r = generate_grounded_answer("q?", _dr_docs(), MagicMock(return_value=DR_ANSWER), judge=MagicMock(return_value="SUPPORTED"))
    assert [c["chunk_id"] for c in r["citations"]] == ["chunk1", "chunk2"]
    bad = generate_grounded_answer("q?", _dr_docs(), MagicMock(return_value="ANSWER: Backups are on disk [7]."), judge=MagicMock(return_value="SUPPORTED"))
    assert not bad["grounded"] and bad["answer_status"] == STATUS_INVALID_CITATIONS and bad["citations"] == []


@pytest.mark.parametrize("question", [
    "What does the OmniBioAI iOS mobile app do?",
    "How do I install the OmniBioAI Android app?",
    "How do I export OmniBioAI results to Tableau?",
    "Can OmniBioAI send results to WhatsApp?",
    "Is there an Alexa skill for OmniBioAI?",
    "How do I connect OmniBioAI to Salesforce?",
])
def test_nonexistent_named_capabilities_remain_refused_before_the_llm(question):
    docs = [chunk(1, text="OmniBioAI runs on Linux. Mobile users can open the web UI. Backups run nightly.")]
    llm = MagicMock(return_value="ANSWER: Yes, OmniBioAI has that [1].")
    r = generate_grounded_answer(question, docs, llm, judge=MagicMock(return_value="SUPPORTED"))
    llm.assert_not_called()
    assert not r["grounded"] and r["answer_status"] == STATUS_UNSUPPORTED_TERM and r["citations"] == []
