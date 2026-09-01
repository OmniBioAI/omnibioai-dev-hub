from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import os
import traceback

from rag.control_plane import CONTROL_PLANE
from api.auth import require_auth

router = APIRouter()


def _debug_tracebacks_enabled() -> bool:
    # Same shared-secret-style convention as api/auth.py's _auth_enabled():
    # a no-op (tracebacks hidden) unless explicitly turned on.
    return os.getenv("DEBUG_TRACEBACKS", "").strip().lower() == "true"


# =========================================================
# REQUEST MODEL
# =========================================================

class QueryRequest(BaseModel):
    query: str
    repo: Optional[str] = None
    bundle: Optional[str] = None


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

    except Exception as e:
        raise RuntimeError(f"Engine access failed: {str(e)}")


# =========================================================
# QUERY ENDPOINT (V6)
# =========================================================

@router.post("/query")
def query(req: QueryRequest, actor: str = Depends(require_auth)):

    try:
        engine = get_engine()

        # V6 CONTRACT: only query() exists
        result = engine.query(req.query, repo=req.repo, bundle=req.bundle)

        return {
            **result,
            "api_version": "v6"
        }

    except Exception as e:
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

            result = engine.retrieve(req.query, repo=req.repo, bundle=req.bundle)
            context = engine.build_context(result)

            # check optional LLM streaming support
            if hasattr(engine, "stream_llm"):
                for token in engine.stream_llm(req.query, context):
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            else:
                # fallback: single response
                response = engine.answer(req.query)
                yield f"data: {json.dumps({'type': 'response', 'content': response['answer']})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )