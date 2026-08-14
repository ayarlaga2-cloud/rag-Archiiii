"""Shared domain types.

Kept dependency-free and import-cycle-free so every layer can use them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SectionKind(str, Enum):
    """Semantic role of a runbook section.

    Stored on every chunk so retrieval can bias toward, say, remediation
    steps when an alert is already firing, or toward diagnosis when the
    agent is still narrowing down a cause.
    """

    OVERVIEW = "overview"
    SYMPTOMS = "symptoms"
    IMPACT = "impact"
    PREREQUISITES = "prerequisites"
    DIAGNOSIS = "diagnosis"
    REMEDIATION = "remediation"
    ROLLBACK = "rollback"
    VERIFICATION = "verification"
    ESCALATION = "escalation"
    REFERENCES = "references"
    OTHER = "other"


@dataclass(slots=True)
class SourceDocument:
    """A normalized Confluence page, ready to chunk."""

    page_id: str
    title: str
    url: str
    space_key: str
    version: int
    labels: list[str]
    markdown: str
    breadcrumbs: list[str] = field(default_factory=list)
    updated_at: str | None = None

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.markdown.encode("utf-8")).hexdigest()[:16]


def build_embed_text(
    page_title: str, breadcrumb: str, kind: "SectionKind | str", body: str
) -> str:
    """Contextual-retrieval envelope for the embedded form of a chunk.

    Defined here (not in the chunker) so the store can reconstruct an identical
    `embed_text` on read without importing the ingest layer.
    """
    kind_value = kind.value if isinstance(kind, SectionKind) else str(kind)
    return (
        f"Runbook: {page_title}\n"
        f"Section: {breadcrumb}\n"
        f"Type: {kind_value}\n\n"
        f"{body}"
    )


@dataclass(slots=True)
class Chunk:
    """One retrievable unit.

    `embed_text` carries breadcrumb context so an isolated "Step 3: restart the
    pool" chunk still embeds near "Postgres connection exhaustion". `body_text`
    is what a human (or the model) actually reads.
    """

    chunk_id: str
    page_id: str
    page_title: str
    page_url: str
    space_key: str
    page_version: int
    content_hash: str
    section_path: list[str]
    section_kind: SectionKind
    section_index: int
    chunk_index: int
    body_text: str
    embed_text: str
    token_count: int
    labels: list[str] = field(default_factory=list)

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.section_path) if self.section_path else self.page_title

    @property
    def citation(self) -> str:
        return f"{self.page_title} — {self.breadcrumb} ({self.page_url})"

    def to_metadata(self) -> dict[str, Any]:
        """Flatten for vector-store metadata.

        Scalars only — Chroma rejects lists/dicts. `body_text` is deliberately
        excluded: stores keep it in their document/text column instead of
        duplicating it into metadata.
        """
        return {
            "page_id": self.page_id,
            "page_title": self.page_title,
            "page_url": self.page_url,
            "space_key": self.space_key,
            "page_version": self.page_version,
            "content_hash": self.content_hash,
            "section_path": " > ".join(self.section_path),
            "section_kind": self.section_kind.value,
            "section_index": self.section_index,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "labels": ",".join(self.labels),
        }

    @classmethod
    def from_metadata(cls, chunk_id: str, md: dict[str, Any], body_text: str) -> "Chunk":
        path = [p for p in str(md.get("section_path", "")).split(" > ") if p]
        raw_labels = str(md.get("labels", ""))
        try:
            kind = SectionKind(md.get("section_kind", "other"))
        except ValueError:
            kind = SectionKind.OTHER
        page_title = str(md.get("page_title", ""))
        breadcrumb = " > ".join(path) if path else page_title
        return cls(
            chunk_id=chunk_id,
            page_id=str(md.get("page_id", "")),
            page_title=page_title,
            page_url=str(md.get("page_url", "")),
            space_key=str(md.get("space_key", "")),
            page_version=int(md.get("page_version", 0) or 0),
            content_hash=str(md.get("content_hash", "")),
            section_path=path,
            section_kind=kind,
            section_index=int(md.get("section_index", 0) or 0),
            chunk_index=int(md.get("chunk_index", 0) or 0),
            body_text=body_text,
            embed_text=build_embed_text(page_title, breadcrumb, kind, body_text),
            token_count=int(md.get("token_count", 0) or 0),
            labels=[lbl for lbl in raw_labels.split(",") if lbl],
        )


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    # Where the hit came from, for debugging retrieval quality.
    source: str = "dense"
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None


@dataclass(slots=True)
class PageState:
    """Ingest bookkeeping — drives incremental sync."""

    page_id: str
    page_version: int
    content_hash: str
    chunk_count: int
