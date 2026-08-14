"""Hybrid retriever.

Pipeline, in order:

    query variants
        -> dense search   (Gemma embeddings, ANN)   top RETRIEVAL_DENSE_K
        -> lexical search (BM25 / Postgres FTS)     top RETRIEVAL_LEXICAL_K
        -> reciprocal rank fusion                   top RETRIEVAL_FUSED_K
        -> cross-encoder rerank (optional)          top RETRIEVAL_FINAL_K
        -> section expansion (optional)

Why each stage earns its place:

* Dense alone misses exact identifiers (`PSQLException`, `ECONNREFUSED`).
* Lexical alone misses paraphrase ("the DB is refusing connections" vs
  "connection pool exhausted").
* RRF fuses them without needing to calibrate two incomparable score scales.
* The cross-encoder fixes the ordering errors both retrievers make, over a
  candidate set small enough to afford it.
* Section expansion pulls in the sibling chunks of a matched section, so a hit
  on "Step 4" returns steps 1-8 rather than a fragment. For runbooks this is
  the difference between a usable answer and a dangerous one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from triage.config import Settings
from triage.embeddings.base import Embedder
from triage.logging_setup import get_logger
from triage.retrieval.bm25 import BM25Index
from triage.retrieval.fusion import reciprocal_rank_fusion
from triage.retrieval.query import build_queries
from triage.retrieval.rerank import Reranker, build_reranker
from triage.types import Chunk, ScoredChunk
from triage.vectorstore.base import VectorStore

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    query: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    expanded: list[Chunk] = field(default_factory=list)
    took_ms: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)

    def as_context(self, include_expanded: bool = True, max_chars: int = 12_000) -> str:
        """Render results as a citable context block."""
        parts: list[str] = []
        budget = max_chars

        source = self.expanded if (include_expanded and self.expanded) else [
            s.chunk for s in self.chunks
        ]
        for i, chunk in enumerate(source, start=1):
            block = (
                f"[{i}] {chunk.page_title} — {chunk.breadcrumb}\n"
                f"    source: {chunk.page_url}\n"
                f"    type: {chunk.section_kind.value}\n\n"
                f"{chunk.body_text}\n"
            )
            if len(block) > budget:
                break
            parts.append(block)
            budget -= len(block)
        return "\n---\n".join(parts)


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
        lexical_index: BM25Index | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.reranker = reranker if reranker is not None else build_reranker(settings)
        self._lexical = lexical_index
        self._lexical_loaded = lexical_index is not None
        # pgvector does full-text search in the database; only Chroma needs the
        # in-process BM25 sidecar.
        self._native_lexical = hasattr(store, "lexical_search")

    # -- lexical index -----------------------------------------------------
    @property
    def lexical(self) -> BM25Index | None:
        if self._native_lexical:
            return None
        if not self._lexical_loaded:
            path = self.settings.lexical_index_path
            if path.exists():
                try:
                    self._lexical = BM25Index.load(path)
                    log.debug("retriever.lexical_loaded", size=self._lexical.size)
                except Exception as exc:
                    log.warning("retriever.lexical_load_failed", error=str(exc))
                    self._lexical = None
            else:
                log.warning(
                    "retriever.lexical_missing",
                    path=str(path),
                    detail="Falling back to dense-only. Run `ingest` to build it.",
                )
            self._lexical_loaded = True
        return self._lexical

    # -- search ------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        expand_sections: bool | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        settings = self.settings
        final_k = top_k or settings.retrieval_final_k
        expand = (
            settings.retrieval_expand_section
            if expand_sections is None
            else expand_sections
        )

        variants = build_queries(query)
        if not variants:
            return RetrievalResult(query=query, took_ms=0.0)

        dense = self._dense(variants, settings.retrieval_dense_k, filters)
        lexical = self._lexical_search(variants, settings.retrieval_lexical_k, filters)

        fused = reciprocal_rank_fusion(
            dense,
            lexical,
            k=settings.retrieval_rrf_k,
            limit=settings.retrieval_fused_k,
            dense_weight=settings.retrieval_dense_weight,
            lexical_weight=settings.retrieval_lexical_weight,
        )

        # The cross-encoder sees the primary query only — concatenating variants
        # would blur the pair it is scoring.
        reranked = self.reranker.rerank(variants[0], fused, final_k)

        expanded = self._expand(reranked) if expand else []

        took = (time.perf_counter() - started) * 1000
        result = RetrievalResult(
            query=query,
            chunks=reranked,
            expanded=expanded,
            took_ms=took,
            stats={
                "variants": len(variants),
                "dense_hits": len(dense),
                "lexical_hits": len(lexical),
                "fused": len(fused),
                "final": len(reranked),
                "expanded": len(expanded),
                "reranker": self.reranker.name,
                "lexical_backend": "native" if self._native_lexical else "bm25",
            },
        )
        log.info("retriever.search", query=query[:120], **result.stats, took_ms=round(took, 1))
        return result

    # -- stages ------------------------------------------------------------
    def _dense(
        self, variants: list[str], k: int, filters: dict[str, Any] | None
    ) -> list[ScoredChunk]:
        merged: dict[str, ScoredChunk] = {}
        # Only the first two variants go dense; the identifier-only variant is
        # lexical fuel and embeds poorly.
        for variant in variants[:2]:
            embedding = self.embedder.embed_query(variant)
            for hit in self.store.query(embedding, k=k, where=filters):
                existing = merged.get(hit.chunk.chunk_id)
                if existing is None or hit.score > existing.score:
                    merged[hit.chunk.chunk_id] = hit
        return sorted(merged.values(), key=lambda s: s.score, reverse=True)[:k]

    def _lexical_search(
        self, variants: list[str], k: int, filters: dict[str, Any] | None
    ) -> list[ScoredChunk]:
        combined = " ".join(variants)

        if self._native_lexical:
            try:
                return self.store.lexical_search(combined, k=k, where=filters)  # type: ignore[attr-defined]
            except Exception as exc:
                log.warning("retriever.native_lexical_failed", error=str(exc))
                return []

        index = self.lexical
        if index is None or index.size == 0:
            return []

        ranked = index.search(combined, k=k)
        if not ranked:
            return []

        chunk_ids = [cid for cid, _ in ranked]
        scores = dict(ranked)
        chunks = self.store.get_chunks(chunk_ids)
        return [
            ScoredChunk(
                chunk=chunk,
                score=scores.get(chunk.chunk_id, 0.0),
                source="lexical",
                lexical_rank=rank,
            )
            for rank, chunk in enumerate(chunks)
        ]

    def _expand(self, results: list[ScoredChunk]) -> list[Chunk]:
        """Replace each hit with its whole section, de-duplicated, order kept.

        A retrieved chunk that is step 4 of 8 is worse than useless during an
        incident. Expansion costs one cheap metadata lookup per hit.
        """
        seen_sections: set[tuple[str, int]] = set()
        seen_chunks: set[str] = set()
        out: list[Chunk] = []

        for result in results:
            chunk = result.chunk
            key = (chunk.page_id, chunk.section_index)
            if key in seen_sections:
                continue
            seen_sections.add(key)
            try:
                siblings = self.store.get_section(chunk.page_id, chunk.section_index)
            except Exception as exc:
                log.warning("retriever.expand_failed", error=str(exc))
                siblings = [chunk]
            for sibling in siblings or [chunk]:
                if sibling.chunk_id not in seen_chunks:
                    seen_chunks.add(sibling.chunk_id)
                    out.append(sibling)
        return out
