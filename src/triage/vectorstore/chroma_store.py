"""Chroma-backed store — the local footprint.

Persists to disk under CHROMA_PATH. Good to a few hundred thousand chunks on a
laptop, which is far beyond any realistic runbook corpus. Swap to pgvector when
you need concurrent writers, real backups, or SQL-side joins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from triage.logging_setup import get_logger
from triage.types import Chunk, PageState, ScoredChunk

log = get_logger(__name__)

_BATCH = 256
_FILTERABLE = {"page_id", "space_key", "section_kind", "page_title", "content_hash"}


def _to_chroma_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate the backend-neutral filter dict into Chroma's query DSL.

    Callers pass `{"space_key": "SRE", "section_kind": ["remediation", "rollback"]}`
    and stay unaware of which store is underneath.
    """
    if not filters:
        return None
    conditions: list[dict[str, Any]] = []
    for key, value in filters.items():
        if key not in _FILTERABLE:
            raise ValueError(f"Unsupported filter field: {key}")
        if isinstance(value, (list, tuple, set)):
            values = list(value)
            if not values:
                continue
            conditions.append({key: {"$in": values}})
        else:
            conditions.append({key: {"$eq": value}})
    if not conditions:
        return None
    # Chroma rejects a bare multi-key dict; single conditions must not be
    # wrapped in $and either.
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class ChromaVectorStore:
    def __init__(self, path: Path, collection: str) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover
            raise ImportError("chromadb is not installed. pip install chromadb") from exc

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = collection
        self._collection = None

    @property
    def name(self) -> str:
        return f"chroma:{self._collection_name}"

    # -- lifecycle ---------------------------------------------------------
    def ensure_collection(self, dimension: int) -> None:
        # Distance is cosine because every embedder here L2-normalizes, which
        # makes cosine and dot-product rank-equivalent and keeps scores in a
        # predictable range.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine", "dimension": dimension},
        )

    def _require(self):
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # -- writes ------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        collection = self._require()

        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start : start + _BATCH],
                documents=[c.body_text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
            )
        log.debug("chroma.upsert", count=len(chunks))

    def delete_page(self, page_id: str) -> int:
        collection = self._require()
        existing = collection.get(where={"page_id": {"$eq": page_id}}, include=[])
        ids = existing.get("ids") or []
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # collection may not exist yet
            pass
        self._collection = None

    # -- reads -------------------------------------------------------------
    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        collection = self._require()
        result = collection.query(
            query_embeddings=[embedding],
            n_results=max(1, k),
            where=_to_chroma_where(where),
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        scored: list[ScoredChunk] = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            chunk = Chunk.from_metadata(cid, meta or {}, doc or "")
            # Chroma returns cosine distance in [0, 2]; map to a [0, 1]
            # similarity so scores mean the same thing across backends.
            similarity = max(0.0, 1.0 - float(dist) / 2.0)
            scored.append(
                ScoredChunk(chunk=chunk, score=similarity, source="dense", dense_rank=rank)
            )
        return scored

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        collection = self._require()
        result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = result.get("ids") or []
        if not ids:
            return None
        docs = result.get("documents") or [""]
        metas = result.get("metadatas") or [{}]
        return Chunk.from_metadata(ids[0], metas[0] or {}, docs[0] or "")

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        collection = self._require()
        found: dict[str, Chunk] = {}
        for start in range(0, len(chunk_ids), _BATCH):
            batch = chunk_ids[start : start + _BATCH]
            result = collection.get(ids=batch, include=["documents", "metadatas"])
            for cid, doc, meta in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            ):
                found[cid] = Chunk.from_metadata(cid, meta or {}, doc or "")
        # Preserve the caller's ordering — it is the lexical ranking.
        return [found[cid] for cid in chunk_ids if cid in found]

    def get_section(self, page_id: str, section_index: int) -> list[Chunk]:
        collection = self._require()
        result = collection.get(
            where={
                "$and": [
                    {"page_id": {"$eq": page_id}},
                    {"section_index": {"$eq": section_index}},
                ]
            },
            include=["documents", "metadatas"],
        )
        chunks = [
            Chunk.from_metadata(cid, meta or {}, doc or "")
            for cid, doc, meta in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            )
        ]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def get_page(self, page_id: str) -> list[Chunk]:
        collection = self._require()
        result = collection.get(
            where={"page_id": {"$eq": page_id}}, include=["documents", "metadatas"]
        )
        chunks = [
            Chunk.from_metadata(cid, meta or {}, doc or "")
            for cid, doc, meta in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            )
        ]
        return sorted(chunks, key=lambda c: (c.section_index, c.chunk_index))

    def iter_chunks(self) -> Iterator[Chunk]:
        collection = self._require()
        offset = 0
        while True:
            result = collection.get(
                limit=_BATCH, offset=offset, include=["documents", "metadatas"]
            )
            ids = result.get("ids") or []
            if not ids:
                return
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            for cid, doc, meta in zip(ids, docs, metas):
                yield Chunk.from_metadata(cid, meta or {}, doc or "")
            offset += len(ids)

    def page_states(self) -> dict[str, PageState]:
        collection = self._require()
        states: dict[str, PageState] = {}
        offset = 0
        while True:
            result = collection.get(limit=_BATCH, offset=offset, include=["metadatas"])
            ids = result.get("ids") or []
            if not ids:
                break
            for meta in result.get("metadatas") or []:
                meta = meta or {}
                page_id = str(meta.get("page_id", ""))
                if not page_id:
                    continue
                existing = states.get(page_id)
                if existing is None:
                    states[page_id] = PageState(
                        page_id=page_id,
                        page_version=int(meta.get("page_version", 0) or 0),
                        content_hash=str(meta.get("content_hash", "")),
                        chunk_count=1,
                    )
                else:
                    existing.chunk_count += 1
            offset += len(ids)
        return states

    def count(self) -> int:
        return int(self._require().count())
