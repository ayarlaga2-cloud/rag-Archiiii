"""Ingest orchestration: fetch -> normalize -> chunk -> embed -> upsert.

Built for a large Confluence corpus, where the run takes hours and the failure
modes are operational rather than algorithmic:

* **Resumable.** Every completed page is appended to a checkpoint file, so an
  interrupted run restarts where it stopped instead of re-embedding everything.
  Embedding is the expensive step; losing an hour of it to a dropped VPN is the
  single most likely way this wastes a day.
* **Constant memory.** Pages stream through one at a time. Nothing accumulates
  the whole corpus — not the documents, not the chunks, not the vectors.
* **Startup is O(checkpoint), not O(corpus).** Asking the vector store to list
  every page means scanning every chunk; at 300k chunks that is minutes before
  the first page is even fetched. The checkpoint answers the same question
  instantly, and falls back to the store scan only when absent.
* **Overlapped I/O.** A prefetch thread keeps fetching and normalizing while
  the main thread embeds, so HTTP latency hides behind CPU work.
* **Incremental lexical updates.** With the FTS5 backend, a 10-page re-sync
  touches 10 pages rather than rebuilding the whole index.
* **Per-page durability.** Each page is deleted-then-upserted and only then
  checkpointed, so a crash can duplicate work but never corrupt a page.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from triage.config import Settings
from triage.embeddings.base import Embedder
from triage.ingest.chunker import RunbookChunker
from triage.ingest.confluence import ConfluenceClient, ConfluencePage
from triage.ingest.normalize import storage_to_markdown
from triage.logging_setup import get_logger
from triage.types import PageState, SourceDocument
from triage.vectorstore.base import VectorStore

log = get_logger(__name__)

_SENTINEL = object()


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
        rate = self.pages_ingested / self.duration_seconds if self.duration_seconds else 0.0
        return {
            "pages_seen": self.pages_seen,
            "pages_ingested": self.pages_ingested,
            "pages_skipped": self.pages_skipped,
            "pages_failed": self.pages_failed,
            "chunks_written": self.chunks_written,
            "chunks_deleted": self.chunks_deleted,
            "duration_seconds": round(self.duration_seconds, 2),
            "pages_per_second": round(rate, 2),
            # Only the first few; a long run can fail on hundreds of pages and
            # dumping all of them buries the summary.
            "failures": self.failures[:20],
            "failures_total": len(self.failures),
        }


class IngestCheckpoint:
    """Append-only record of completed pages.

    JSONL rather than a rewritten dict: appending is atomic enough that a crash
    mid-write loses at most the last line, whereas a truncated rewrite of a
    100k-entry dict loses the entire history.
    """

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._handle = None

    def load(self) -> dict[str, PageState]:
        if not self.enabled or not self.path.exists():
            return {}
        states: dict[str, PageState] = {}
        bad_lines = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                states[row["page_id"]] = PageState(
                    page_id=row["page_id"],
                    page_version=int(row.get("version", 0)),
                    content_hash=str(row.get("hash", "")),
                    chunk_count=int(row.get("chunks", 0)),
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                # A torn final line from a hard kill — skip it, the page will
                # simply be re-ingested.
                bad_lines += 1
        if bad_lines:
            log.warning("checkpoint.skipped_bad_lines", count=bad_lines, path=str(self.path))
        return states

    def record(self, doc: SourceDocument, chunk_count: int) -> None:
        if not self.enabled:
            return
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(
            json.dumps(
                {
                    "page_id": doc.page_id,
                    "version": doc.version,
                    "hash": doc.content_hash,
                    "chunks": chunk_count,
                }
            )
            + "\n"
        )
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def clear(self) -> None:
        self.close()
        if self.path.exists():
            self.path.unlink()


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


def prefetch(source: Iterable, size: int = 8) -> Iterator:
    """Run `source` on a background thread, buffering up to `size` items.

    Confluence pagination and HTML normalization are I/O and parse work that
    would otherwise sit idle while the embedding model runs. The queue is
    bounded so a fast producer cannot pull the whole corpus into memory.
    """
    if size <= 0:
        yield from source
        return

    buffer: queue.Queue = queue.Queue(maxsize=size)

    def produce() -> None:
        try:
            for item in source:
                buffer.put(item)
        except Exception as exc:  # surface it on the consumer thread
            buffer.put(exc)
        finally:
            buffer.put(_SENTINEL)

    worker = threading.Thread(target=produce, daemon=True, name="ingest-prefetch")
    worker.start()

    while True:
        item = buffer.get()
        if item is _SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item


class IngestPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        chunker: RunbookChunker | None = None,
        lexical=None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.chunker = chunker or RunbookChunker(
            max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            token_counter=getattr(embedder, "count_tokens", None),
        )
        self._lexical = lexical
        # pgvector does full-text search in the database; only Chroma needs a
        # separate lexical index alongside it.
        self._native_lexical = hasattr(store, "lexical_search")

    # -- lexical backend ---------------------------------------------------
    @property
    def lexical(self):
        if self._native_lexical:
            return None
        if self._lexical is None:
            from triage.retrieval.sqlite_lexical import build_lexical_index

            self._lexical = build_lexical_index(
                self.settings.data_dir,
                self.settings.vector_collection,
                prefer_sqlite=self.settings.lexical_backend != "json",
            )
        return self._lexical

    @property
    def _lexical_is_incremental(self) -> bool:
        return hasattr(self.lexical, "add_chunks")

    # -- sources -----------------------------------------------------------
    def run_confluence(
        self, force: bool = False, limit: int | None = None, resume: bool = True
    ) -> IngestReport:
        with ConfluenceClient(self.settings) as client:
            client.verify()  # fail fast on bad credentials

            total = None
            try:
                total = client.count_pages()
                log.info("ingest.corpus_size", pages=total, cql=self.settings.effective_cql())
            except Exception as exc:
                log.debug("ingest.count_unavailable", error=str(exc))

            pages = client.search_pages()
            if limit:
                pages = (p for i, p in enumerate(pages) if i < limit)
                total = min(total, limit) if total else limit

            docs = (page_to_document(p) for p in pages)
            return self.ingest_documents(docs, force=force, total=total, resume=resume)

    def run_local_markdown(
        self, directory: Path, force: bool = False, resume: bool = True
    ) -> IngestReport:
        """Ingest a folder of .md files — useful before Confluence is reachable."""
        directory = Path(directory)
        paths = sorted(directory.rglob("*.md"))

        def documents() -> Iterator[SourceDocument]:
            for path in paths:
                text = path.read_text(encoding="utf-8")
                title = path.stem.replace("-", " ").replace("_", " ").title()
                first = text.lstrip().split("\n", 1)[0].strip()
                if first.startswith("# "):
                    title = first[2:].strip()
                yield SourceDocument(
                    page_id=f"local:{path.relative_to(directory).as_posix()}",
                    title=title,
                    url=path.resolve().as_uri(),
                    space_key="LOCAL",
                    version=int(path.stat().st_mtime),
                    labels=["runbook", "local"],
                    markdown=text,
                )

        return self.ingest_documents(
            documents(), force=force, total=len(paths), resume=resume
        )

    # -- core --------------------------------------------------------------
    def ingest_documents(
        self,
        documents: Iterable[SourceDocument],
        force: bool = False,
        total: int | None = None,
        resume: bool = True,
    ) -> IngestReport:
        started = time.monotonic()
        report = IngestReport()
        settings = self.settings

        self.store.ensure_collection(self.embedder.dimension)

        checkpoint = IngestCheckpoint(settings.checkpoint_path, enabled=settings.ingest_checkpoint)
        if not resume:
            checkpoint.clear()

        known = checkpoint.load() if resume else {}
        source = "checkpoint"
        if not known:
            # First run, or the checkpoint was cleared: fall back to asking the
            # store. Slow on a large collection, which is exactly why the
            # checkpoint exists.
            known = self.store.page_states()
            source = "store scan"

        log.info(
            "ingest.start",
            embedder=self.embedder.name,
            store=self.store.name,
            known_pages=len(known),
            known_from=source,
            total=total,
            force=force,
            resume=resume,
        )

        progress_every = max(1, settings.ingest_progress_every)
        last_log = started

        try:
            for doc in prefetch(documents, size=settings.ingest_prefetch):
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

                    # Delete-then-insert: a page whose headings changed produces
                    # a different set of chunk ids, so upsert alone would leave
                    # orphans pointing at text that no longer exists.
                    removed = self.store.delete_page(doc.page_id)
                    self.store.upsert(chunks, vectors)

                    if self._lexical_is_incremental:
                        self.lexical.delete_page(doc.page_id)
                        self.lexical.add_chunks(chunks)

                    # Checkpoint only after the page is durably written, so a
                    # crash re-does the page rather than skipping it.
                    checkpoint.record(doc, len(chunks))

                    report.pages_ingested += 1
                    report.chunks_written += len(chunks)
                    report.chunks_deleted += removed
                    log.debug(
                        "ingest.page",
                        page_id=doc.page_id,
                        title=doc.title,
                        chunks=len(chunks),
                        replaced=removed,
                    )
                except Exception as exc:  # one bad page must not end the run
                    report.pages_failed += 1
                    report.failures.append(f"{doc.page_id} ({doc.title}): {exc}")
                    log.error("ingest.page_failed", page_id=doc.page_id, error=str(exc))

                if report.pages_seen % progress_every == 0:
                    now = time.monotonic()
                    self._log_progress(report, started, now, total)
                    last_log = now
        finally:
            checkpoint.close()

        # A non-incremental backend (JSON BM25) can only be rebuilt wholesale.
        if report.chunks_written and not self._lexical_is_incremental:
            self.rebuild_lexical_index()
        elif report.chunks_written and self._lexical_is_incremental:
            optimize = getattr(self.lexical, "optimize", None)
            if callable(optimize):
                optimize()

        report.duration_seconds = time.monotonic() - started
        log.info("ingest.complete", **report.as_dict())
        return report

    # -- progress ----------------------------------------------------------
    def _log_progress(
        self, report: IngestReport, started: float, now: float, total: int | None
    ) -> None:
        elapsed = max(now - started, 1e-6)
        rate = report.pages_seen / elapsed
        fields = {
            "seen": report.pages_seen,
            "ingested": report.pages_ingested,
            "skipped": report.pages_skipped,
            "failed": report.pages_failed,
            "chunks": report.chunks_written,
            "pages_per_sec": round(rate, 2),
            "elapsed_min": round(elapsed / 60, 1),
        }
        if total:
            remaining = max(0, total - report.pages_seen)
            fields["total"] = total
            fields["percent"] = round(100 * report.pages_seen / total, 1)
            if rate > 0:
                fields["eta_min"] = round(remaining / rate / 60, 1)
        log.info("ingest.progress", **fields)

    # -- lexical -----------------------------------------------------------
    def rebuild_lexical_index(self) -> int:
        """Rebuild the lexical index from the vector store.

        Only needed for the non-incremental JSON backend, or to repair a
        lexical index that drifted out of sync with the store.
        """
        if self._native_lexical:
            log.info("ingest.lexical_native", store=self.store.name)
            return 0

        index = self.lexical
        count = index.build(self.store.iter_chunks())

        # The JSON backend has to be written out explicitly; FTS5 is already
        # on disk.
        save = getattr(index, "save", None)
        if callable(save):
            self.settings.data_dir.mkdir(parents=True, exist_ok=True)
            try:
                save(self.settings.lexical_index_path)
            except TypeError:
                save()

        optimize = getattr(index, "optimize", None)
        if callable(optimize):
            optimize()

        log.info("ingest.lexical_index", documents=count, backend=type(index).__name__)
        return count
