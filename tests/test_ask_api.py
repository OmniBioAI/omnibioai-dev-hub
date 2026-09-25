"""API-level proof of the Ask OmniBioAI contract: a REAL RAGEngine behind the real routes.

Only the two network calls are stubbed (query embedding and LLM generation), so
these tests exercise retrieval policy + the answer contract exactly as served.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.rag import router
from index.vector_store import VectorStore
from rag.answering import NO_CONTEXT_MESSAGE, NO_SUPPORT_MESSAGE
from rag.engine import RAGEngine

DIM = 768
app = FastAPI()
app.include_router(router)


class FlatIndex:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.ntotal = len(self.vectors)

    def reconstruct(self, row):
        return self.vectors[row]

    def search(self, q, k):
        scores = self.vectors @ np.asarray(q, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores)[:k]
        pad = k - len(order)
        return np.concatenate([scores[order], np.full(pad, -1e9)])[None, :], np.concatenate([order, np.full(pad, -1)])[None, :]


def _unit(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _meta(n, visibility, text, path):
    return {"text": text, "source": f"omnibioai-docs:{path}@rev", "repo": "omnibioai-docs", "relative_path": path,
            "source_revision": "rev", "document_id": f"doc{n}", "chunk_id": f"chunk{n}", "content_state": "CURRENT",
            "verification_state": "CONFIGURED", "bundle": "site", "title": f"T{n}", "visibility": visibility,
            "citation": {"repository": "omnibioai-docs", "relative_path": path, "source_revision": "rev", "document_id": f"doc{n}",
                         "chunk_id": f"chunk{n}", "content_state": "CURRENT", "verification_state": "CONFIGURED"}}


@pytest.fixture
def served():
    """PUBLIC docs on axes 0-1, an INTERNAL doc on axis 2 (the closest match to the INTERNAL question)."""
    meta = [
        _meta(0, "PUBLIC", "Back up the database nightly. Restore it with the recovery tool.", "site/docs/admin/dr.md"),
        _meta(1, "PUBLIC", "Connect REDCap with an API token.", "site/docs/integrations/redcap.md"),
        _meta(2, "INTERNAL", "INTERNAL ragbio ingestion secrets and scripts.", "README.md"),
    ]
    vs = VectorStore()
    vs.index, vs.metadata, vs.dim = FlatIndex([_unit(0), _unit(1), _unit(2)]), meta, DIM
    engine = RAGEngine(vs)
    with patch("api.routes.rag.CONTROL_PLANE") as cp:
        cp.get_engine.return_value = engine
        yield engine


def two_stage(answer, verdict="SUPPORTED"):
    """Fake LLM: the answer prompt gets `answer`, the entailment fact-check prompt gets `verdict`."""
    def fake(prompt, **kwargs):
        return verdict if "strict fact checker" in prompt else answer
    return MagicMock(side_effect=fake)


def _ask(path, text, embed_axis, llm, **extra):
    """Run one request with a stubbed query embedding and a stubbed (spy) LLM."""
    with patch("rag.engine.ollama_embed", return_value=_unit(embed_axis)), patch("rag.engine.ollama_generate", llm):
        r = TestClient(app).post(path, json={"query": text, **extra})
    return r


def _stream_events(response):
    return [json.loads(line.replace("data: ", "")) for line in response.iter_lines() if line]


# ---- 19A / 19F: zero context never reaches the LLM, on BOTH paths -----------------------------------

@pytest.mark.parametrize("path", ["/query", "/stream"])
def test_zero_qualifying_context_never_invokes_the_llm(served, path):
    llm = MagicMock(return_value="Fabricated answer [1].")
    r = _ask(path, "how do I bake sourdough bread", embed_axis=5, llm=llm)  # orthogonal to every chunk -> below cutoff
    assert r.status_code == 200
    llm.assert_not_called()
    body = r.json() if path == "/query" else next(e for e in _stream_events(r) if e["type"] == "response")
    if path == "/query":
        assert body["answer"] == NO_CONTEXT_MESSAGE
    else:
        assert body["content"] == NO_CONTEXT_MESSAGE
    assert (body["grounded"], body["context_used"], body["citations"], body["llm_invoked"]) == (False, 0, [], False)
    assert body["answer_status"] == "NO_TRUSTED_CONTEXT"


def test_stream_no_context_event_sequence_is_status_response_done(served):
    r = _ask("/stream", "sourdough", embed_axis=5, llm=MagicMock())
    events = _stream_events(r)
    assert [e["type"] for e in events] == ["status", "response", "done"]
    assert events[0]["context_used"] == 0


# ---- 19B / 19C: grounded answers and citations ---------------------------------------------------------

@pytest.mark.parametrize("path", ["/query", "/stream"])
def test_supported_question_gets_a_grounded_cited_answer(served, path):
    llm = two_stage("Back up nightly and restore with the recovery tool [1].")
    r = _ask(path, "how do I back up and restore the database", embed_axis=0, llm=llm)
    assert llm.call_count == 2  # answer generation + entailment fact-check
    prompt = llm.call_args_list[0].args[0]
    assert "site/docs/admin/dr.md" in prompt and "INTERNAL" not in prompt and "ragbio" not in prompt
    body = r.json() if path == "/query" else next(e for e in _stream_events(r) if e["type"] == "response")
    assert body["grounded"] is True and body["answer_status"] == "GROUNDED"
    (cite,) = body["citations"]
    assert cite["chunk_id"] == "chunk0" and cite["relative_path"] == "site/docs/admin/dr.md"
    assert cite["content_state"] == "CURRENT" and cite["verification_state"] == "CONFIGURED" and cite["index"] == 1


def test_generation_options_are_deterministic_and_have_an_explicit_context_window(served):
    llm = two_stage("Back up nightly [1].")
    _ask("/query", "how do I back up the database", embed_axis=0, llm=llm)
    for call in llm.call_args_list:  # both the answer and the fact-check use the fixed, deterministic options
        assert call.kwargs["options"] == {"temperature": 0, "num_ctx": 8192, "seed": 7}


def test_a_hallucinated_uncited_answer_from_the_model_is_never_shown(served):
    llm = MagicMock(return_value="OmniBioAI can also restore snapshots from any cloud provider automatically.")
    body = _ask("/query", "how do I back up the database", embed_axis=0, llm=llm).json()
    assert body["grounded"] is False and body["answer"] == NO_SUPPORT_MESSAGE and body["citations"] == []
    assert "snapshots" not in json.dumps(body["answer"])


# ---- 19E: adversarial ------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/query", "/stream"])
def test_plausible_nonexistent_capability_is_refused_even_though_it_retrieves_context(served, path):
    """'OmniBioAI iOS app' is semantically close to real docs (embedding axis 0) yet must not become a claim."""
    llm = MagicMock(return_value="Yes, OmniBioAI has an iOS app [1].")
    r = _ask(path, "What does the OmniBioAI iOS mobile app do?", embed_axis=0, llm=llm)
    llm.assert_not_called()
    body = r.json() if path == "/query" else next(e for e in _stream_events(r) if e["type"] == "response")
    assert body["grounded"] is False and body["answer_status"] == "UNSUPPORTED_TERM" and body["citations"] == []
    assert body["context_used"] == 1  # it DID retrieve context, and still refused


def test_internal_content_cannot_be_retrieved_or_reach_the_llm(served):
    llm = MagicMock(return_value="Ingestion uses secrets [1].")
    r = _ask("/query", "tell me about the ragbio ingestion scripts", embed_axis=2, llm=llm)  # axis 2 = INTERNAL chunk
    body = r.json()
    llm.assert_not_called()
    assert body["context_used"] == 0 and body["context"] == [] and body["answer"] == NO_CONTEXT_MESSAGE
    assert "INTERNAL" not in json.dumps(body) and "secrets" not in json.dumps(body)


@pytest.mark.parametrize("path", ["/query", "/stream"])
def test_client_cannot_request_internal_visibility_or_lower_the_cutoff(served, path):
    llm = MagicMock(return_value="x [1].")
    r = _ask(path, "tell me about the ragbio ingestion scripts", embed_axis=2, llm=llm,
             allowed_visibilities=["INTERNAL", "PUBLIC"], visibility="INTERNAL", min_relevance=0.0)
    llm.assert_not_called()
    assert "secrets" not in r.text and "INTERNAL ragbio" not in r.text  # no INTERNAL chunk text anywhere in the response


def test_lowering_min_relevance_from_the_client_still_yields_no_context_for_off_topic_questions(served):
    llm = MagicMock()
    body = _ask("/query", "weather in paris", embed_axis=7, llm=llm, min_relevance=0).json()
    llm.assert_not_called()
    assert body["context_used"] == 0 and body["grounded"] is False


def test_citation_integrity_end_to_end(served):
    llm = two_stage("Restore with the recovery tool [1].")
    body = _ask("/query", "how do I restore", embed_axis=0, llm=llm).json()
    supplied = {c["chunk_id"] for c in body["context"]}
    assert body["citations"] and all(c["chunk_id"] in supplied for c in body["citations"])
    assert all(c["chunk_id"] in {m["chunk_id"] for m in served.vector_store.metadata if m["visibility"] == "PUBLIC"} for c in body["citations"])


def test_llm_outage_is_a_clean_error_state_on_both_paths(served):
    llm = MagicMock(side_effect=ConnectionError("ollama down"))
    for path in ("/query", "/stream"):
        r = _ask(path, "how do I back up the database", embed_axis=0, llm=llm)
        body = r.json() if path == "/query" else next(e for e in _stream_events(r) if e["type"] == "response")
        assert body["grounded"] is False and body["answer_status"] == "LLM_UNAVAILABLE"
        assert "ollama down" not in json.dumps(body)


@pytest.mark.parametrize("path", ["/query", "/stream"])
@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {"query": None}, {}, {"query": "x" * 5000}])
def test_empty_and_malformed_queries_never_reach_embedding_or_llm(served, path, payload):
    llm, embed = MagicMock(), MagicMock()
    with patch("rag.engine.ollama_embed", embed), patch("rag.engine.ollama_generate", llm):
        r = TestClient(app).post(path, json=payload)
    assert r.status_code == 422
    llm.assert_not_called()
    embed.assert_not_called()


# ---- entailment fact-check stage -------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/query", "/stream"])
def test_a_cited_answer_the_fact_checker_rejects_is_never_shown(served, path):
    llm = two_stage("Backups are also mirrored to a second region every hour [1].", verdict="UNSUPPORTED")
    r = _ask(path, "how do I back up the database", embed_axis=0, llm=llm)
    body = r.json() if path == "/query" else next(e for e in _stream_events(r) if e["type"] == "response")
    assert body["grounded"] is False and body["answer_status"] == "UNSUPPORTED_CLAIM" and body["citations"] == []
    assert "mirrored" not in json.dumps(body["answer" if path == "/query" else "content"])


def test_the_fact_checker_fails_closed_on_hedging_and_errors(served):
    for verdict in ("Probably SUPPORTED", "", "I think so"):
        body = _ask("/query", "how do I back up the database", embed_axis=0, llm=two_stage("Back up nightly [1].", verdict=verdict)).json()
        assert body["grounded"] is False, verdict

    def boom(prompt, **kwargs):
        if "strict fact checker" in prompt:
            raise ConnectionError("judge down")
        return "Back up nightly [1]."
    body = _ask("/query", "how do I back up the database", embed_axis=0, llm=MagicMock(side_effect=boom)).json()
    assert body["grounded"] is False and body["answer_status"] == "LLM_UNAVAILABLE" and "judge down" not in json.dumps(body)


def test_the_fact_checker_is_not_reached_when_a_cheaper_check_already_refused(served):
    for text, axis, answer in (("sourdough", 5, "x [1]."), ("What does the OmniBioAI iOS app do?", 0, "x [1]."),
                               ("how do I back up the database", 0, "INSUFFICIENT_CONTEXT"), ("how do I back up the database", 0, "uncited words")):
        llm = two_stage(answer)
        _ask("/query", text, embed_axis=axis, llm=llm)
        assert not any("strict fact checker" in c.args[0] for c in llm.call_args_list), (text, answer)


def test_the_entailment_check_can_be_disabled_explicitly_and_is_reported(served, monkeypatch):
    monkeypatch.setenv("DEVHUB_ENTAILMENT_CHECK", "off")
    llm = MagicMock(return_value="Back up nightly [1].")
    body = _ask("/query", "how do I back up the database", embed_axis=0, llm=llm).json()
    assert llm.call_count == 1 and body["grounded"] is True and body["entailment_checked"] is False


def test_by_default_the_entailment_check_runs_and_is_reported(served):
    body = _ask("/query", "how do I back up the database", embed_axis=0, llm=two_stage("Back up nightly [1].")).json()
    assert body["grounded"] is True and body["entailment_checked"] is True
