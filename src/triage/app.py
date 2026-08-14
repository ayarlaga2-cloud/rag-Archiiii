"""Composition root.

One place that knows how the pieces fit together, so the CLI, the API and the
eval harness all build an identical stack from the same settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from triage.config import Settings, get_settings
from triage.embeddings.base import Embedder
from triage.embeddings.factory import build_embedder
from triage.ingest.chunker import RunbookChunker
from triage.ingest.pipeline import IngestPipeline
from triage.logging_setup import configure_logging, get_logger
from triage.retrieval.retriever import HybridRetriever
from triage.vectorstore.base import VectorStore
from triage.vectorstore.factory import build_vector_store

log = get_logger(__name__)


@dataclass
class Stack:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    chunker: RunbookChunker
    retriever: HybridRetriever
    pipeline: IngestPipeline

    def close(self) -> None:
        for component in (self.store, self.embedder):
            closer = getattr(component, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pragma: no cover - best effort
                    pass


def build_stack(settings: Settings | None = None, configure_logs: bool = True) -> Stack:
    settings = settings or get_settings()
    if configure_logs:
        configure_logging(settings.log_level, settings.log_json)
    settings.ensure_dirs()

    embedder = build_embedder(settings)
    store = build_vector_store(settings)
    store.ensure_collection(embedder.dimension)

    chunker = RunbookChunker(
        max_tokens=settings.chunk_max_tokens,
        min_tokens=settings.chunk_min_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        # Size chunks with the embedding model's own tokenizer, so
        # CHUNK_MAX_TOKENS means Gemma tokens rather than a guess.
        token_counter=getattr(embedder, "count_tokens", None),
    )
    pipeline = IngestPipeline(settings, embedder, store, chunker)
    retriever = HybridRetriever(settings, embedder, store)

    log.info(
        "stack.ready",
        embedder=embedder.name,
        dimension=embedder.dimension,
        store=store.name,
        chunk_max_tokens=settings.chunk_max_tokens,
    )
    return Stack(
        settings=settings,
        embedder=embedder,
        store=store,
        chunker=chunker,
        retriever=retriever,
        pipeline=pipeline,
    )
