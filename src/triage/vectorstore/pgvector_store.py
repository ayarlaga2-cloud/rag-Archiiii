"""Postgres + pgvector store — the production footprint.

Chosen over Chroma for production because it gives you concurrent writers, real
backups and PITR, row-level security, and — the reason it matters most here —
native full-text search. That lets the lexical half of hybrid retrieval run
inside the database instead of in a rebuilt in-process BM25 index, which is the
part of the local setup that does not scale.

Requires:
    CREATE EXTENSION vector;   -- pgvector >= 0.5 for HNSW
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from triage.logging_setup import get_logger
from triage.types import Chunk, PageState, ScoredChunk

log = get_logger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BATCH = 200

_COLUMNS = (
    "chunk_id, page_id, page_title, page_url, space_key, page_version, "
    "content_hash, section_path, section_kind, section_index, chunk_index, "
    "token_count, labels, body_text"
)


def _to_vector_literal(vector: list[float]) -> str:
    """pgvector accepts a bracketed text literal, so no extra client package."""
    return "[" + ",".join(f"{v:.7f}" for v in vector) + "]"


def _row_to_chunk(row: tuple) -> Chunk:
    (
        chunk_id, page_id, page_title, page_url, space_key, page_version,
        content_hash, section_path, section_kind, section_index, chunk_index,
        token_count, labels, body_text,
    ) = row[:14]
    return Chunk.from_metadata(
        chunk_id,
        {
            "page_id": page_id,
            "page_title": page_title,
            "page_url": page_url,
            "space_key": space_key,
            "page_version": page_version,
            "content_hash": content_hash,
            "section_path": section_path,
            "section_kind": section_kind,
            "section_index": section_index,
            "chunk_index": chunk_index,
            "token_count": token_count,
            "labels": labels,
        },
        body_text or "",
    )


class PgVectorStore:
    def __init__(self, dsn: str, collection: str) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "psycopg is not installed. pip install 'psycopg[binary]'"
            ) from exc
        if not dsn:
            raise ValueError("PGVECTOR_DSN is required when VECTOR_BACKEND=pgvector")
        if not _IDENT_RE.match(collection):
            raise ValueError(
                f"VECTOR_COLLECTION {collection!r} is not a valid SQL identifier"
            )

        self._psycopg = psycopg
        self._dsn = dsn
        self._table = f"chunks_{collection}"
        self._conn = psycopg.connect(dsn, autocommit=True)

    @property
    def name(self) -> str:
        return f"pgvector:{self._table}"

    def close(self) -> None:
        self._conn.close()

    # -- schema ------------------------------------------------------------
    def ensure_collection(self, dimension: int) -> None:
        t = self._table
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    chunk_id      TEXT PRIMARY KEY,
                    page_id       TEXT NOT NULL,
                    page_title    TEXT NOT NULL DEFAULT '',
                    page_url      TEXT NOT NULL DEFAULT '',
                    space_key     TEXT NOT NULL DEFAULT '',
                    page_version  INTEGER NOT NULL DEFAULT 0,
                    content_hash  TEXT NOT NULL DEFAULT '',
                    section_path  TEXT NOT NULL DEFAULT '',
                    section_kind  TEXT NOT NULL DEFAULT 'other',
                    section_index INTEGER NOT NULL DEFAULT 0,
                    chunk_index   INTEGER NOT NULL DEFAULT 0,
                    token_count   INTEGER NOT NULL DEFAULT 0,
                    labels        TEXT NOT NULL DEFAULT '',
                    body_text     TEXT NOT NULL DEFAULT '',
                    embed_text    TEXT NOT NULL DEFAULT '',
                    embedding     vector({dimension}) NOT NULL,
                    tsv           tsvector GENERATED ALWAYS AS (
                                      to_tsvector('english', embed_text)
                                  ) STORED,
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # HNSW: fast approximate ANN. Build after bulk load for large
            # corpora; CREATE INDEX IF NOT EXISTS makes that safe to reorder.
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_embedding_idx ON {t} "
                f"USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS {t}_tsv_idx ON {t} USING gin (tsv)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {t}_page_idx ON {t} (page_id)")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_section_idx ON {t} (page_id, section_index)"
            )
        log.info("pgvector.schema_ready", table=t, dimension=dimension)

    # -- writes ------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        sql = f"""
            INSERT INTO {self._table} (
                chunk_id, page_id, page_title, page_url, space_key, page_version,
                content_hash, section_path, section_kind, section_index,
                chunk_index, token_count, labels, body_text, embed_text, embedding
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET
                page_id=EXCLUDED.page_id,
                page_title=EXCLUDED.page_title,
                page_url=EXCLUDED.page_url,
                space_key=EXCLUDED.space_key,
                page_version=EXCLUDED.page_version,
                content_hash=EXCLUDED.content_hash,
                section_path=EXCLUDED.section_path,
                section_kind=EXCLUDED.section_kind,
                section_index=EXCLUDED.section_index,
                chunk_index=EXCLUDED.chunk_index,
                token_count=EXCLUDED.token_count,
                labels=EXCLUDED.labels,
                body_text=EXCLUDED.body_text,
                embed_text=EXCLUDED.embed_text,
                embedding=EXCLUDED.embedding,
                updated_at=now()
        """
        rows = [
            (
                c.chunk_id, c.page_id, c.page_title, c.page_url, c.space_key,
                c.page_version, c.content_hash, " > ".join(c.section_path),
                c.section_kind.value, c.section_index, c.chunk_index,
                c.token_count, ",".join(c.labels), c.body_text, c.embed_text,
                _to_vector_literal(e),
            )
            for c, e in zip(chunks, embeddings)
        ]
        with self._conn.cursor() as cur:
            for start in range(0, len(rows), _BATCH):
                cur.executemany(sql, rows[start : start + _BATCH])
        log.debug("pgvector.upsert", count=len(rows))

    def delete_page(self, page_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE page_id = %s", (page_id,))
            return cur.rowcount or 0

    def reset(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self._table}")

    # -- filters -----------------------------------------------------------
    def _where_clause(self, filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
        if not filters:
            return "", []
        allowed = {
            "page_id", "space_key", "section_kind", "page_title", "content_hash",
        }
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            if key not in allowed:
                raise ValueError(f"Unsupported filter field: {key}")
            if isinstance(value, (list, tuple, set)):
                values = list(value)
                if not values:
                    continue
                clauses.append(f"{key} = ANY(%s)")
                params.append(values)
            else:
                clauses.append(f"{key} = %s")
                params.append(value)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    # -- reads -------------------------------------------------------------
    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        clause, params = self._where_clause(where)
        vector = _to_vector_literal(embedding)
        sql = f"""
            SELECT {_COLUMNS}, embedding <=> %s::vector AS distance
            FROM {self._table}
            {clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, [vector, *params, vector, max(1, k)])
            rows = cur.fetchall()

        out: list[ScoredChunk] = []
        for rank, row in enumerate(rows):
            distance = float(row[14])
            # Same [0, 1] mapping as the Chroma backend so scores compare.
            similarity = max(0.0, 1.0 - distance / 2.0)
            out.append(
                ScoredChunk(
                    chunk=_row_to_chunk(row),
                    score=similarity,
                    source="dense",
                    dense_rank=rank,
                )
            )
        return out

    def lexical_search(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        """Native full-text search — the production replacement for in-process BM25."""
        clause, params = self._where_clause(where)
        joiner = " AND " if clause else " WHERE "
        sql = f"""
            SELECT {_COLUMNS},
                   ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS rank
            FROM {self._table}
            {clause}{joiner}tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, [query, *params, query, max(1, k)])
            rows = cur.fetchall()
        return [
            ScoredChunk(
                chunk=_row_to_chunk(row),
                score=float(row[14]),
                source="lexical",
                lexical_rank=rank,
            )
            for rank, row in enumerate(rows)
        ]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {self._table} WHERE chunk_id = %s", (chunk_id,)
            )
            row = cur.fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {self._table} WHERE chunk_id = ANY(%s)",
                (chunk_ids,),
            )
            rows = cur.fetchall()
        found = {row[0]: _row_to_chunk(row) for row in rows}
        # Preserve the caller's ordering — it is the lexical ranking.
        return [found[cid] for cid in chunk_ids if cid in found]

    def get_section(self, page_id: str, section_index: int) -> list[Chunk]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {self._table} "
                f"WHERE page_id = %s AND section_index = %s ORDER BY chunk_index",
                (page_id, section_index),
            )
            rows = cur.fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_page(self, page_id: str) -> list[Chunk]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM {self._table} WHERE page_id = %s "
                f"ORDER BY section_index, chunk_index",
                (page_id,),
            )
            rows = cur.fetchall()
        return [_row_to_chunk(r) for r in rows]

    def iter_chunks(self) -> Iterator[Chunk]:
        with self._conn.cursor(name="chunk_cursor") as cur:
            cur.execute(f"SELECT {_COLUMNS} FROM {self._table}")
            for row in cur:
                yield _row_to_chunk(row)

    def page_states(self) -> dict[str, PageState]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT page_id, MAX(page_version), MAX(content_hash), COUNT(*) "
                f"FROM {self._table} GROUP BY page_id"
            )
            rows = cur.fetchall()
        return {
            str(r[0]): PageState(
                page_id=str(r[0]),
                page_version=int(r[1] or 0),
                content_hash=str(r[2] or ""),
                chunk_count=int(r[3] or 0),
            )
            for r in rows
        }

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            row = cur.fetchone()
        return int(row[0]) if row else 0
