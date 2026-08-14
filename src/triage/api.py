"""Search API.

Deliberately small: phase 1 only needs a way to run retrieval from something
other than the CLI. The alert webhook arrives with the triage agent in a later
phase.

    GET  /healthz
    GET  /v1/stats
    POST /v1/search   {"query": "...", "top_k": 6, "space": "SRE"}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from triage.app import Stack, build_stack
from triage.logging_setup import get_logger

log = get_logger(__name__)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    space: str | None = None
    section_kind: list[str] | None = None
    expand_sections: bool = True
    include_body: bool = True


class Hit(BaseModel):
    chunk_id: str
    score: float
    rerank_score: float | None = None
    source: str
    page_title: str
    breadcrumb: str
    section_kind: str
    url: str
    body: str | None = None


class SearchResponse(BaseModel):
    query: str
    took_ms: float
    stats: dict[str, Any]
    hits: list[Hit]
    context: str | None = None


def create_app(stack: Stack | None = None) -> FastAPI:
    api = FastAPI(title="Runbook RAG", version="0.1.0")
    state: dict[str, Stack | None] = {"stack": stack}

    def get_stack() -> Stack:
        if state["stack"] is None:
            state["stack"] = build_stack()
        return state["stack"]

    @api.on_event("startup")
    def _startup() -> None:
        # Load the model once at boot, not on the first request.
        get_stack()

    @api.on_event("shutdown")
    def _shutdown() -> None:
        if state["stack"] is not None:
            state["stack"].close()

    @api.get("/healthz")
    def healthz() -> dict[str, Any]:
        current = state["stack"]
        return {
            "status": "ok" if current is not None else "starting",
            "embedder": current.embedder.name if current else None,
            "store": current.store.name if current else None,
        }

    @api.get("/v1/stats")
    def stats() -> dict[str, Any]:
        current = get_stack()
        return {
            "chunks": current.store.count(),
            "pages": len(current.store.page_states()),
            "embedder": current.embedder.name,
            "dimension": current.embedder.dimension,
            "store": current.store.name,
        }

    @api.post("/v1/search", response_model=SearchResponse)
    def search(request: SearchRequest) -> SearchResponse:
        current = get_stack()
        filters: dict[str, Any] = {}
        if request.space:
            filters["space_key"] = request.space
        if request.section_kind:
            filters["section_kind"] = request.section_kind

        try:
            result = current.retriever.search(
                request.query,
                top_k=request.top_k,
                filters=filters or None,
                expand_sections=request.expand_sections,
            )
        except Exception as exc:
            log.error("api.search_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return SearchResponse(
            query=result.query,
            took_ms=round(result.took_ms, 2),
            stats=result.stats,
            hits=[
                Hit(
                    chunk_id=s.chunk.chunk_id,
                    score=round(s.score, 6),
                    rerank_score=s.rerank_score,
                    source=s.source,
                    page_title=s.chunk.page_title,
                    breadcrumb=s.chunk.breadcrumb,
                    section_kind=s.chunk.section_kind.value,
                    url=s.chunk.page_url,
                    body=s.chunk.body_text if request.include_body else None,
                )
                for s in result.chunks
            ],
            context=result.as_context() if request.include_body else None,
        )

    return api
