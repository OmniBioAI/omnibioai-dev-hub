import json
import os
import traceback

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from api.auth import require_auth
from rag.control_plane import CONTROL_PLANE

router = APIRouter()

# Ordinary callers get PUBLIC content only, and only chunks genuinely relevant
# to the question. Both are decided here, server-side; nothing in the request
# body can widen visibility or lower the cutoff.
PUBLIC_ONLY = {"PUBLIC"}

# Cosine-similarity floor, calibrated on an independent 40+40 answerable /
# unanswerable query set against the Phase 18 candidate (AUC 0.983; keeps
# 39/40 answerable, rejects 38/40 unanswerable). See scripts/calibrate_relevance.py.
# It rejects off-topic queries; it cannot reject plausible-sounding fabricated
# features of the real product (those score like genuine answers).
DEFAULT_MIN_RELEVANCE = 0.64


def _min_relevance() -> float:
    raw = os.getenv("DEVHUB_MIN_RELEVANCE", "").strip()
    if not raw:
        return DEFAULT_MIN_RELEVANCE
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MIN_RELEVANCE  # fail safe: a typo must not silently disable the cutoff
    return value if 0.0 <= value <= 1.0 else DEFAULT_MIN_RELEVANCE


def _debug_tracebacks_enabled() -> bool:
    # Same shared-secret-style convention as api/auth.py's _auth_enabled():
    # a no-op (tracebacks hidden) unless explicitly turned on.
    return os.getenv("DEBUG_TRACEBACKS", "").strip().lower() == "true"


# =========================================================
# REQUEST MODEL
# =========================================================

MAX_QUERY_CHARS = 2000


class QueryRequest(BaseModel):
    # Unknown fields (allowed_visibilities, min_relevance, visibility...) are
    # ignored, never honoured: policy is server-side (PUBLIC_ONLY, cutoff).
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    repo: str | None = Field(default=None, max_length=200)
    bundle: str | None = Field(default=None, max_length=200)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or "\x00" in stripped:
            raise ValueError("query must be non-empty text")
        return stripped


# =========================================================
# ENGINE ACCESS (V6 SAFE)
# =========================================================

def get_engine():
    try:
        engine = CONTROL_PLANE.get_engine()

        if engine is None:
            raise RuntimeError("RAG engine not initialized")

        # V6 SAFETY CHECKS
        if not hasattr(engine, "query"):
            raise RuntimeError("Engine missing V6 query method")

        return engine

    except Exception as e:  # noqa: BLE001 -- boundary: any engine-access failure becomes a clean RuntimeError for the caller
        raise RuntimeError(f"Engine access failed: {e!s}")


# =========================================================
# QUERY ENDPOINT (V6)
# =========================================================

@router.post("/query")
def query(req: QueryRequest, actor: str = Depends(require_auth)):

    try:
        engine = get_engine()
        docs = engine.retrieve(req.query, repo=req.repo, bundle=req.bundle,
                               allowed_visibilities=PUBLIC_ONLY, min_relevance=_min_relevance())
        # Grounded-answer contract (rag/answering.py): zero qualifying context never reaches the LLM.
        result = engine.answer_from_docs(req.query, docs)

        return {
            **result,
            "api_version": "v6"
        }

    except Exception as e:  # noqa: BLE001 -- top-level HTTP boundary: must catch anything to return a clean 500 instead of an unhandled crash
        detail = {"error": str(e)}
        # Stack traces can leak file paths, internals, and other sensitive
        # detail into the HTTP response -- only include one when a deployer
        # has explicitly opted in via DEBUG_TRACEBACKS=true.
        if _debug_tracebacks_enabled():
            detail["trace"] = traceback.format_exc()
        raise HTTPException(status_code=500, detail=detail)


# =========================================================
# STREAMING ENDPOINT (V6 SAFE SSE)
# =========================================================

@router.post("/stream")
def stream(req: QueryRequest, actor: str = Depends(require_auth)):
    """Server-sent events for Ask OmniBioAI.

    The answer is generated, VERIFIED, and only then sent: streaming raw model
    tokens would show an unverified answer before the grounding checks could
    reject it. Events:
      status    {"stage": "retrieved", "context_used": n}
      response  {"content", "grounded", "answer_status", "citations", "context_used", "llm_invoked"}
      done      {}
      error     {"message"}
    With no qualifying context, `response` is the fixed no-answer message and
    the LLM is never invoked -- the same code path as /query.
    """

    def event(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    def event_stream():

        try:
            engine = get_engine()

            docs = engine.retrieve(req.query, repo=req.repo, bundle=req.bundle,
                                   allowed_visibilities=PUBLIC_ONLY, min_relevance=_min_relevance())
            yield event({"type": "status", "stage": "retrieved", "context_used": len(docs)})

            result = engine.answer_from_docs(req.query, docs)
            yield event({
                "type": "response",
                "content": result["answer"],
                "grounded": result["grounded"],
                "answer_status": result["answer_status"],
                "answer_contract": result["answer_contract"],
                "citations": result["citations"],
                "context_used": result["context_used"],
                "llm_invoked": result["llm_invoked"],
            })
            yield event({"type": "done"})

        except Exception as e:  # noqa: BLE001 -- SSE generator boundary: must catch anything to emit an error event instead of killing the stream
            yield event({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )
