"""Ingest orchestration: fetch -> normalize -> chunk -> embed -> upsert.

Incremental by default. A page is re-processed only when its Confluence version
number or its normalized-content hash changed, because re-embedding an
unchanged corpus is slow, costs real money on a hosted embedding API, and
needlessly churns the ANN index.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from triage.config import Settings
from triage.embeddings.base import Embedder
from triage.ingest.chunker import RunbookChunker
from triage.ingest.confluence import ConfluenceClient, ConfluencePage
from triage.ingest.normalize import storage_to_markdown
from triage.logging_setup import get_logger
from triage.retrieval.bm25 import BM25Index
from triage.types import SourceDocument
from triage.vectorstore.base import VectorStore

log = get_logger(__name__)


@dataclass
class IngestReport:
    pages_seen: int = 0
    pages_ingested: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    duration_seconds: float = 0.0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "pages_seen": self.pages_seen,
            "pages_ingested": self.pages_ingested,
            "pages_skipped": self.pages_skipped,
            "pages_failed": self.pages_failed,
            "chunks_written": self.chunks_written,
            "chunks_deleted": self.chunks_deleted,
            "duration_seconds": round(self.duration_seconds, 2),
            "failures": self.failures,
        }


def page_to_document(page: ConfluencePage) -> SourceDocument:
    return SourceDocument(
        page_id=page.page_id,
        title=page.title,
        url=page.url,
        space_key=page.space_key,
        version=page.version,
        labels=page.labels,
        markdown=storage_to_markdown(page.storage_html),
        breadcrumbs=page.ancestors,
        updated_at=page.updated_at,
    )


class IngestPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        chunker: RunbookChunker | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.chunker = chunker or RunbookChunker(
            max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            # Size chunks with the embedding model's real tokenizer where one
            # is available, so CHUNK_MAX_TOKENS means what it says.
            token_counter=getattr(embedder, "count_tokens", None),
        )

    # -- sources -----------------------------------------------------------
    def run_confluence(self, force: bool = False, limit: int | None = None) -> IngestReport:
        with ConfluenceClient(self.settings) as client:
            client.verify()  # fail fast on bad credentials
            pages = client.search_pages()
            if limit:
                pages = (p for i, p in enumerate(pages) if i < limit)
            docs = (page_to_document(p) for p in pages)
            return self.ingest_documents(docs, force=force)

    def run_local_markdown(self, directory: Path, force: bool = False) -> IngestReport:
        """Ingest a folder of .md files.

        Useful before Confluence credentials exist, and for reproducible tests.
        """
        directory = Path(directory)
        docs: list[SourceDocument] = []
        for path in sorted(directory.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = path.stem.replace("-", " ").replace("_", " ").title()
            # Honour a leading H1 as the page title if present.
            first = text.lstrip().split("\n", 1)[0].strip()
            if first.startswith("# "):
                title = first[2:].strip()
            docs.append(
                SourceDocument(
                    page_id=f"local:{path.relative_to(directory).as_posix()}",
                    title=title,
                    url=path.resolve().as_uri(),
                    space_key="LOCAL",
                    version=int(path.stat().st_mtime),
                    labels=["runbook", "local"],
                    markdown=text,
                )
            )
        return self.ingest_documents(docs, force=force)

    # -- core --------------------------------------------------------------
    def ingest_documents(
        self, documents: Iterable[SourceDocument], force: bool = False
    ) -> IngestReport:
        started = time.monotonic()
        report = IngestReport()

        self.store.ensure_collection(self.embedder.dimension)
        known = self.store.page_states()
        log.info(
            "ingest.start",
            embedder=self.embedder.name,
            store=self.store.name,
            known_pages=len(known),
            force=force,
        )

        for doc in documents:
            report.pages_seen += 1
            try:
                if not doc.markdown.strip():
                    report.pages_skipped += 1
                    log.debug("ingest.skip_empty", page_id=doc.page_id, title=doc.title)
                    continue

                state = known.get(doc.page_id)
                unchanged = (
                    state is not None
                    and state.page_version == doc.version
                    and state.content_hash == doc.content_hash
                )
                if unchanged and not force:
                    report.pages_skipped += 1
                    continue

                chunks = self.chunker.chunk(doc)
                if not chunks:
                    report.pages_skipped += 1
                    log.warning("ingest.no_chunks", page_id=doc.page_id, title=doc.title)
                    continue

                vectors = self.embedder.embed_documents([c.embed_text for c in chunks])

                # Delete-then-insert: a page whose headings changed produces a
                # different set of chunk ids, so upsert alone would leave
                # orphans pointing at text that no longer exists.
                removed = self.store.delete_page(doc.page_id)
                self.store.upsert(chunks, vectors)

                report.pages_ingested += 1
                report.chunks_written += len(chunks)
                report.chunks_deleted += removed
                log.info(
                    "ingest.page",
                    page_id=doc.page_id,
                    title=doc.title,
                    chunks=len(chunks),
                    replaced=removed,
                )
            except Exception as exc:  # keep going; one bad page is not fatal
                report.pages_failed += 1
                message = f"{doc.page_id} ({doc.title}): {exc}"
                report.failures.append(message)
                log.error("ingest.page_failed", page_id=doc.page_id, error=str(exc))

        if report.chunks_written:
            self.rebuild_lexical_index()

        report.duration_seconds = time.monotonic() - started
        log.info("ingest.complete", **report.as_dict())
        return report

    def rebuild_lexical_index(self) -> int:
        """Rebuild the on-disk BM25 index from the vector store.

        Skipped when the store does its own full-text search (pgvector), where
        the lexical half of hybrid retrieval runs in the database.
        """
        if hasattr(self.store, "lexical_search"):
            log.info("ingest.lexical_native", store=self.store.name)
            return 0

        index = BM25Index()
        count = index.build(self.store.iter_chunks())
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        index.save(self.settings.lexical_index_path)
        log.info(
            "ingest.lexical_index",
            documents=count,
            path=str(self.settings.lexical_index_path),
        )
        return count
