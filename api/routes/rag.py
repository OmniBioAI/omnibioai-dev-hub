import json
import os
import traceback

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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

class QueryRequest(BaseModel):
    query: str
    repo: str | None = None
    bundle: str | None = None


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

        # V6 CONTRACT: only query() exists
        result = engine.query(req.query, repo=req.repo, bundle=req.bundle,
                              allowed_visibilities=PUBLIC_ONLY, min_relevance=_min_relevance())

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

    def event_stream():

        try:
            engine = get_engine()

            # V6: no hybrid_retrieve dependency anymore
            # fallback-safe: reuse query pipeline structure

            result = engine.retrieve(req.query, repo=req.repo, bundle=req.bundle,
                                    allowed_visibilities=PUBLIC_ONLY, min_relevance=_min_relevance())
            context = engine.build_context(result)

            # check optional LLM streaming support
            if hasattr(engine, "stream_llm"):
                for token in engine.stream_llm(req.query, context):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            else:
                # fallback: single response
                response = engine.answer(req.query, repo=req.repo, bundle=req.bundle,
                                       allowed_visibilities=PUBLIC_ONLY, min_relevance=_min_relevance())
                yield f"data: {json.dumps({'type': 'response', 'content': response['answer']})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:  # noqa: BLE001 -- SSE generator boundary: must catch anything to emit an error event instead of killing the stream
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )