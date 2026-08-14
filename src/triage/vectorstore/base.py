"""Vector store protocol.

This is the seam that makes "local now, another DB later" a config change.
Nothing above this layer knows whether it is talking to Chroma or Postgres.

`page_states()` and `delete_page()` exist so ingest can be incremental: a
re-sync that re-embeds 4,000 unchanged chunks is slow, expensive on a hosted
embedding API, and needlessly churns the index.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable

from triage.types import Chunk, PageState, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    @property
    def name(self) -> str:
        ...

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection/table if absent. Must be idempotent."""

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        ...

    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Nearest neighbours, most similar first. Score is in [0, 1]."""

    def delete_page(self, page_id: str) -> int:
        """Remove every chunk of a page. Returns the count removed."""

    def page_states(self) -> dict[str, PageState]:
        """Current version/hash per ingested page."""

    def iter_chunks(self) -> Iterator[Chunk]:
        """Every chunk — used to build the lexical index."""

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        ...

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Batch fetch. The lexical index returns ids, so this avoids N round trips."""

    def get_section(self, page_id: str, section_index: int) -> list[Chunk]:
        """All chunks of one section, in order — used for section expansion."""

    def get_page(self, page_id: str) -> list[Chunk]:
        """All chunks of one page, in order.

        Must filter in the store, not by scanning every chunk — on a large
        corpus the difference is milliseconds versus minutes.
        """

    def count(self) -> int:
        ...

    def reset(self) -> None:
        """Drop everything. Destructive."""
