"""SQLite FTS5 lexical index — the scalable local backend.

The JSON BM25 index is fine for a few thousand chunks and wrong for a large
Confluence export: it deserializes the entire postings table into memory on
every process start, so a 300k-chunk corpus means a multi-hundred-MB load
before the first query returns, and a full rebuild every time one page changes.

FTS5 fixes all three problems and costs no new dependency — it is compiled into
the `sqlite3` module that ships with CPython:

  * the index lives on disk and is memory-mapped, so startup is constant-time
  * `bm25()` is a native ranking function, same Okapi scoring, done in C
  * rows can be deleted and re-inserted per page, so a 10-page re-sync touches
    10 pages instead of rebuilding 300k

The one design decision worth stating: text is **pre-tokenized with the
project's own SRE-aware tokenizer** before being handed to FTS5, and queries go
through the same tokenizer. FTS5's built-in `unicode61` tokenizer would split
`org.postgresql.util.PSQLException` on the dots and lose the camelCase
boundaries entirely, which is exactly the signal incident queries depend on.
Feeding it a pre-normalized token stream keeps ranking behaviour identical to
the in-memory BM25 implementation while getting FTS5's scale.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from triage.logging_setup import get_logger
from triage.retrieval.bm25 import tokenize
from triage.types import Chunk

log = get_logger(__name__)

_BATCH = 1000


def fts5_available() -> bool:
    """True when the bundled SQLite was compiled with FTS5."""
    try:
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
            return True
        finally:
            con.close()
    except sqlite3.OperationalError:
        return False


def _fts_query(terms: list[str]) -> str:
    """Build a safe FTS5 MATCH expression.

    Every term is quoted as a string literal, which neutralises the FTS5
    operators (`*`, `:`, `^`, `-`, `NEAR`, `AND`/`OR`/`NOT`) that would
    otherwise turn a stack trace pasted into the search box into a syntax error.
    """
    quoted = ['"' + t.replace('"', '""') + '"' for t in terms if t]
    return " OR ".join(quoted)


class SQLiteLexicalIndex:
    """Disk-backed lexical index with per-page incremental updates."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.path), check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    # -- schema ------------------------------------------------------------
    def _ensure_schema(self) -> None:
        self._con.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS lexical USING fts5(
                chunk_id UNINDEXED,
                page_id  UNINDEXED,
                tokens,
                tokenize = 'unicode61'
            )
            """
        )
        self._con.commit()

    # -- writes ------------------------------------------------------------
    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        rows = [
            # embed_text, not body_text: the breadcrumb carries terms the body
            # of a bare step list often omits.
            (c.chunk_id, c.page_id, " ".join(tokenize(c.embed_text)))
            for c in chunks
        ]
        if not rows:
            return 0
        for start in range(0, len(rows), _BATCH):
            self._con.executemany(
                "INSERT INTO lexical (chunk_id, page_id, tokens) VALUES (?, ?, ?)",
                rows[start : start + _BATCH],
            )
        self._con.commit()
        return len(rows)

    def delete_page(self, page_id: str) -> int:
        cur = self._con.execute("DELETE FROM lexical WHERE page_id = ?", (page_id,))
        self._con.commit()
        return cur.rowcount or 0

    def build(self, chunks: Iterable[Chunk]) -> int:
        """Full rebuild. Prefer add_chunks/delete_page for incremental syncs."""
        self._con.execute("DELETE FROM lexical")
        self._con.commit()
        return self.add_chunks(chunks)

    def optimize(self) -> None:
        """Merge the b-tree segments. Worth running after a large bulk load."""
        try:
            self._con.execute("INSERT INTO lexical(lexical) VALUES('optimize')")
            self._con.commit()
        except sqlite3.OperationalError as exc:  # pragma: no cover
            log.warning("lexical.optimize_failed", error=str(exc))

    # -- reads -------------------------------------------------------------
    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        match = _fts_query(terms)
        if not match:
            return []
        try:
            cur = self._con.execute(
                """
                SELECT chunk_id, bm25(lexical) AS rank
                FROM lexical
                WHERE lexical MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match, max(1, k)),
            )
            # FTS5 bm25() is more-negative-is-better; negate so that, as
            # everywhere else in this codebase, higher means more relevant.
            return [(row[0], -float(row[1])) for row in cur.fetchall()]
        except sqlite3.OperationalError as exc:
            log.warning("lexical.query_failed", error=str(exc), query=query[:120])
            return []

    @property
    def size(self) -> int:
        cur = self._con.execute("SELECT count(*) FROM lexical")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:  # pragma: no cover
            pass

    # Kept for interface parity with BM25Index; FTS5 is already on disk.
    def save(self, path: Path | None = None) -> None:
        self._con.commit()

    @classmethod
    def load(cls, path: Path) -> "SQLiteLexicalIndex":
        return cls(path)


def build_lexical_index(data_dir: Path, collection: str, prefer_sqlite: bool = True):
    """Pick the lexical backend: FTS5 when available, JSON BM25 otherwise."""
    if prefer_sqlite and fts5_available():
        return SQLiteLexicalIndex(Path(data_dir) / f"lexical_{collection}.db")

    from triage.retrieval.bm25 import BM25Index

    log.warning(
        "lexical.fts5_unavailable",
        detail="Falling back to the in-memory JSON BM25 index. Fine for small "
        "corpora; expect slow startup and full rebuilds at scale.",
    )
    return BM25Index()
