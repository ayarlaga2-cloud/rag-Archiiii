"""End-to-end ingest + retrieval against an in-memory store.

Uses the hashing embedder so the whole path runs without torch or network.
Retrieval quality is not asserted here — only that the wiring is correct and
that incremental sync actually skips unchanged pages.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import pytest

from triage.config import Settings
from triage.embeddings.hashing import HashingEmbedder
from triage.ingest.pipeline import IngestPipeline
from triage.retrieval.retriever import HybridRetriever
from triage.types import Chunk, PageState, ScoredChunk, SourceDocument


class InMemoryStore:
    """Minimal VectorStore implementation for tests."""

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, list[float]] = {}
        self.dimension: int | None = None

    @property
    def name(self) -> str:
        return "memory"

    def ensure_collection(self, dimension: int) -> None:
        self.dimension = dimension

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, embeddings):
            self.chunks[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = vector

    def query(self, embedding: list[float], k: int, where: dict[str, Any] | None = None):
        def cosine(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        candidates = list(self.chunks.values())
        if where:
            for key, value in where.items():
                allowed = value if isinstance(value, (list, tuple, set)) else [value]
                candidates = [
                    c for c in candidates
                    if str(getattr(c, key, getattr(c.section_kind, "value", ""))) in allowed
                    or (key == "section_kind" and c.section_kind.value in allowed)
                ]
        scored = sorted(
            candidates,
            key=lambda c: cosine(embedding, self.vectors[c.chunk_id]),
            reverse=True,
        )[:k]
        return [
            ScoredChunk(chunk=c, score=cosine(embedding, self.vectors[c.chunk_id]),
                        source="dense", dense_rank=i)
            for i, c in enumerate(scored)
        ]

    def delete_page(self, page_id: str) -> int:
        ids = [cid for cid, c in self.chunks.items() if c.page_id == page_id]
        for cid in ids:
            self.chunks.pop(cid, None)
            self.vectors.pop(cid, None)
        return len(ids)

    def page_states(self) -> dict[str, PageState]:
        states: dict[str, PageState] = {}
        for chunk in self.chunks.values():
            existing = states.get(chunk.page_id)
            if existing is None:
                states[chunk.page_id] = PageState(
                    chunk.page_id, chunk.page_version, chunk.content_hash, 1
                )
            else:
                existing.chunk_count += 1
        return states

    def iter_chunks(self) -> Iterator[Chunk]:
        return iter(list(self.chunks.values()))

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self.chunks[c] for c in chunk_ids if c in self.chunks]

    def get_page(self, page_id: str) -> list[Chunk]:
        found = [c for c in self.chunks.values() if c.page_id == page_id]
        return sorted(found, key=lambda c: (c.section_index, c.chunk_index))

    def get_section(self, page_id: str, section_index: int) -> list[Chunk]:
        found = [
            c for c in self.chunks.values()
            if c.page_id == page_id and c.section_index == section_index
        ]
        return sorted(found, key=lambda c: c.chunk_index)

    def count(self) -> int:
        return len(self.chunks)

    def reset(self) -> None:
        self.chunks.clear()
        self.vectors.clear()


RUNBOOK = """# Postgres Connection Exhaustion

## Symptoms

Checkout returns 503 and PgBouncer reports clients waiting.

## Diagnosis

Run the pool query:

```bash
psql -p 6432 pgbouncer -c "SHOW POOLS;"
```

## Remediation

1. Terminate the runaway backend.
2. Raise the pool ceiling temporarily.
3. Revert the ceiling within 24 hours.

## Escalation

Page the Data Platform on-call.
"""


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        embedding_provider="hashing",
        vector_backend="chroma",
        data_dir=tmp_path,
        chroma_path=tmp_path / "chroma",
        chunk_max_tokens=200,
        chunk_min_tokens=40,
        chunk_overlap_tokens=0,
        retrieval_final_k=3,
    )


@pytest.fixture()
def stack(settings):
    store = InMemoryStore()
    embedder = HashingEmbedder()
    pipeline = IngestPipeline(settings, embedder, store)
    retriever = HybridRetriever(settings, embedder, store)
    return settings, store, embedder, pipeline, retriever


def make_doc(version: int = 1, markdown: str = RUNBOOK) -> SourceDocument:
    return SourceDocument(
        page_id="123",
        title="Postgres Connection Exhaustion",
        url="https://example.test/123",
        space_key="SRE",
        version=version,
        labels=["runbook"],
        markdown=markdown,
    )


def test_ingest_writes_chunks_and_lexical_index(stack):
    settings, store, _, pipeline, _ = stack
    report = pipeline.ingest_documents([make_doc()])

    assert report.pages_ingested == 1
    assert report.chunks_written > 0
    assert store.count() == report.chunks_written

    # Backend-agnostic: assert the lexical index is populated and searchable,
    # not which file it happens to live in.
    assert pipeline.lexical is not None
    assert pipeline.lexical.size == report.chunks_written
    assert pipeline.lexical.search("pgbouncer", k=5)


def test_ingest_writes_checkpoint_for_resume(stack):
    settings, _, _, pipeline, _ = stack
    pipeline.ingest_documents([make_doc()])

    assert settings.checkpoint_path.exists()
    lines = [
        l for l in settings.checkpoint_path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["page_id"] == "123"
    assert row["chunks"] > 0


def test_resume_skips_pages_using_checkpoint_only(stack):
    """The second run must not need to scan the store to know what is done."""
    _, store, _, pipeline, _ = stack
    pipeline.ingest_documents([make_doc()])

    # Emptying the store leaves only the checkpoint as a source of truth. If
    # resume works off the checkpoint, the page is still skipped.
    store.chunks.clear()
    store.vectors.clear()
    second = pipeline.ingest_documents([make_doc()])
    assert second.pages_skipped == 1
    assert second.pages_ingested == 0


def test_no_resume_clears_checkpoint_but_still_trusts_the_store(stack):
    """`resume=False` and `force=True` are different levers.

    resume=False discards the checkpoint and re-derives what is already indexed
    from the store — an unchanged page is still skipped, because re-embedding it
    would be pure waste. force=True is the flag that re-embeds regardless.
    """
    settings, _, _, pipeline, _ = stack
    pipeline.ingest_documents([make_doc()])
    assert settings.checkpoint_path.exists()

    again = pipeline.ingest_documents([make_doc()], resume=False)
    assert again.pages_skipped == 1
    assert again.pages_ingested == 0

    forced = pipeline.ingest_documents([make_doc()], force=True)
    assert forced.pages_ingested == 1


def test_corrupt_checkpoint_line_is_tolerated(stack):
    settings, _, _, pipeline, _ = stack
    pipeline.ingest_documents([make_doc()])
    # Simulate a hard kill mid-write leaving a torn final line.
    with settings.checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write('{"page_id": "trunc')

    report = pipeline.ingest_documents([make_doc()])
    assert report.pages_skipped == 1  # the intact line still applied


def test_unchanged_page_is_skipped_on_second_run(stack):
    _, _, _, pipeline, _ = stack
    pipeline.ingest_documents([make_doc()])
    second = pipeline.ingest_documents([make_doc()])

    assert second.pages_ingested == 0
    assert second.pages_skipped == 1


def test_force_reingests_unchanged_page(stack):
    _, _, _, pipeline, _ = stack
    pipeline.ingest_documents([make_doc()])
    forced = pipeline.ingest_documents([make_doc()], force=True)
    assert forced.pages_ingested == 1


def test_edited_page_replaces_old_chunks(stack):
    _, store, _, pipeline, _ = stack
    pipeline.ingest_documents([make_doc()])
    before = store.count()

    edited = RUNBOOK.replace("Page the Data Platform on-call.", "Page the DBA rota instead.")
    report = pipeline.ingest_documents([make_doc(version=2, markdown=edited)])

    assert report.pages_ingested == 1
    assert report.chunks_deleted == before  # old chunks removed, not orphaned
    bodies = " ".join(c.body_text for c in store.iter_chunks())
    assert "DBA rota" in bodies
    assert "Data Platform on-call" not in bodies


def test_search_returns_relevant_chunk(stack):
    _, _, _, pipeline, retriever = stack
    pipeline.ingest_documents([make_doc()])

    result = retriever.search("PgBouncer clients waiting 503", expand_sections=False)
    assert result.chunks
    assert result.stats["dense_hits"] > 0
    titles = {s.chunk.page_title for s in result.chunks}
    assert "Postgres Connection Exhaustion" in titles


def test_section_expansion_returns_whole_procedure(stack):
    _, _, _, pipeline, retriever = stack
    pipeline.ingest_documents([make_doc()])

    result = retriever.search("terminate the runaway backend", expand_sections=True)
    combined = " ".join(c.body_text for c in result.expanded)
    # Retrieving step 1 must bring steps 2 and 3 along with it.
    assert "Terminate the runaway backend" in combined
    assert "Revert the ceiling" in combined


def test_empty_query_returns_empty_result(stack):
    _, _, _, pipeline, retriever = stack
    pipeline.ingest_documents([make_doc()])
    assert retriever.search("   ").chunks == []


def test_as_context_renders_citations(stack):
    _, _, _, pipeline, retriever = stack
    pipeline.ingest_documents([make_doc()])
    context = retriever.search("pool exhausted", expand_sections=False).as_context()
    assert "Postgres Connection Exhaustion" in context
    assert "https://example.test/123" in context
