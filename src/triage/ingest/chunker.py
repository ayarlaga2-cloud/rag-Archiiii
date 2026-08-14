"""Runbook-aware chunking.

Fixed-size sliding-window chunking is the default in most RAG tutorials and it
is actively harmful for runbooks. It routinely produces chunks like:

    ...4. Drain the node before
    ---
    restarting. This will page the on-call if...

A half-procedure retrieved during an incident is worse than no procedure. So
this chunker is structural rather than positional:

  * Sections come from the Markdown heading tree; every chunk knows its
    breadcrumb, and that breadcrumb is prepended to the embedded text so an
    isolated "Step 3" still embeds near its parent topic (contextual retrieval).
  * Code fences, tables and lists are ATOMIC. A chunk boundary never lands
    inside a command, a table row, or a numbered step.
  * When an atomic block genuinely exceeds the budget it is split at its own
    internal boundaries — between list items, between table rows (repeating the
    header), between code lines — never mid-line.
  * Overlap is sentence-aligned and is suppressed across a code fence, where
    duplicated commands would be actively misleading.
  * Sections are classified (symptoms / diagnosis / remediation / rollback /
    escalation ...) so retrieval can filter or boost by operational intent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

from triage.types import Chunk, SectionKind, SourceDocument, build_embed_text

TokenCounter = Callable[[str], int]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TABLE_RE = re.compile(r"^\s*\|")
_QUOTE_RE = re.compile(r"^\s*>")
_SENTENCE_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z0-9`\-*])")

# Ordered longest-signal-first: "rollback" must win before a generic
# "remediation" match, and "verification" before "steps".
_KIND_PATTERNS: list[tuple[SectionKind, re.Pattern[str]]] = [
    (SectionKind.ROLLBACK, re.compile(r"\b(roll ?back|revert|undo|back ?out)\b", re.I)),
    (SectionKind.ESCALATION, re.compile(r"\b(escalat\w*|on[- ]?call|contact|who to (call|page)|paging|ownership)\b", re.I)),
    (SectionKind.VERIFICATION, re.compile(r"\b(verif\w*|validat\w*|confirm|post[- ]?check|smoke test|health ?check)\b", re.I)),
    (SectionKind.REMEDIATION, re.compile(r"\b(remediat\w*|mitigat\w*|resolution|resolv\w*|fix|repair|recover\w*|runbook steps|steps?|procedure|action plan|how to)\b", re.I)),
    (SectionKind.DIAGNOSIS, re.compile(r"\b(diagnos\w*|triage|investigat\w*|debug\w*|troubleshoot\w*|root cause|analysis|checks?)\b", re.I)),
    (SectionKind.SYMPTOMS, re.compile(r"\b(symptom\w*|signal\w*|alert\w*|detection|what you('ll)? see|indicator)\b", re.I)),
    (SectionKind.IMPACT, re.compile(r"\b(impact|blast radius|severity|customer effect|sla|slo)\b", re.I)),
    (SectionKind.PREREQUISITES, re.compile(r"\b(prerequisit\w*|before you (begin|start)|requirements?|access|permissions?)\b", re.I)),
    (SectionKind.REFERENCES, re.compile(r"\b(reference\w*|see also|links?|related|appendix|further reading)\b", re.I)),
    (SectionKind.OVERVIEW, re.compile(r"\b(overview|summary|description|purpose|scope|about|introduction|context)\b", re.I)),
]


def heuristic_token_count(text: str) -> int:
    """Provider-agnostic token estimate.

    Deliberately not tiktoken — that is OpenAI's tokenizer and would be wrong
    for every embedding model here. Real tokenizers plug in via `token_counter`
    (`LocalEmbedder` exposes its own). Character-based dominates for
    command-heavy text, word-based for prose; taking the max is a safe
    over-estimate, which keeps chunks under the model's real limit.
    """
    if not text:
        return 0
    words = len(text.split())
    return max(int(len(text) / 4) + 1, int(words * 1.3) + 1)


def classify_section(title: str) -> SectionKind:
    if not title.strip():
        return SectionKind.OTHER
    for kind, pattern in _KIND_PATTERNS:
        if pattern.search(title):
            return kind
    return SectionKind.OTHER


@dataclass(slots=True)
class Block:
    """A Markdown block. `atomic` blocks are never split by packing."""

    kind: str  # paragraph | code | list | table | quote
    text: str
    atomic: bool = False


@dataclass(slots=True)
class Section:
    level: int
    title: str
    path: list[str]
    blocks: list[Block] = field(default_factory=list)
    index: int = 0

    @property
    def is_empty(self) -> bool:
        return not any(b.text.strip() for b in self.blocks)


# ---------------------------------------------------------------------------
# Markdown structural parsing
# ---------------------------------------------------------------------------
def split_sections(markdown: str, root_title: str) -> list[Section]:
    """Split Markdown into heading-delimited sections with breadcrumb paths."""
    lines = markdown.split("\n")
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    current_lines: list[str] = []
    current_level = 0
    current_title = root_title

    in_fence = False
    fence_marker = ""

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines)
        path = [title for _, title in stack]
        # Most runbooks repeat the page title as a body H1. Left alone that
        # yields "Title > Title > Symptoms" in every breadcrumb and burns the
        # duplicate tokens in every embedded chunk. The page title is carried
        # separately on the Chunk, so drop it from the path here.
        if path and path[0] == root_title:
            path = path[1:]
        if not path:
            path = [root_title]
        sections.append(
            Section(
                level=current_level,
                title=current_title,
                path=list(path),
                blocks=split_blocks(body),
                index=len(sections),
            )
        )
        current_lines = []

    for line in lines:
        stripped = line.strip()

        # Fence tracking first — a `#` inside a shell snippet is a comment,
        # not a heading.
        if stripped.startswith("```"):
            marker = "`" * (len(stripped) - len(stripped.lstrip("`")))
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif len(marker) >= len(fence_marker):
                in_fence, fence_marker = False, ""
            current_lines.append(line)
            continue

        heading = None if in_fence else _HEADING_RE.match(stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current_level, current_title = level, title
        else:
            current_lines.append(line)

    flush()
    return [s for s in sections if not s.is_empty]


def split_blocks(body: str) -> list[Block]:
    """Group raw lines into Markdown blocks, marking the atomic ones."""
    blocks: list[Block] = []
    lines = body.split("\n")
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code — atomic.
        if stripped.startswith("```"):
            marker = "`" * (len(stripped) - len(stripped.lstrip("`")))
            buf = [line]
            i += 1
            while i < n:
                buf.append(lines[i])
                candidate = lines[i].strip()
                if candidate.startswith("```"):
                    closing = "`" * (len(candidate) - len(candidate.lstrip("`")))
                    if len(closing) >= len(marker):
                        i += 1
                        break
                i += 1
            blocks.append(Block("code", "\n".join(buf), atomic=True))
            continue

        # Table — atomic.
        if _TABLE_RE.match(line):
            buf = []
            while i < n and _TABLE_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            blocks.append(Block("table", "\n".join(buf), atomic=True))
            continue

        # List — atomic. A procedure's numbering only means something whole.
        if _LIST_RE.match(line):
            buf = []
            while i < n:
                nxt = lines[i]
                if _LIST_RE.match(nxt) or (nxt.startswith(("  ", "\t")) and nxt.strip()):
                    buf.append(nxt)
                    i += 1
                elif not nxt.strip() and i + 1 < n and _LIST_RE.match(lines[i + 1]):
                    buf.append(nxt)  # blank line inside a loose list
                    i += 1
                else:
                    break
            blocks.append(Block("list", "\n".join(buf), atomic=True))
            continue

        # Blockquote (info/warning panels land here).
        if _QUOTE_RE.match(line):
            buf = []
            while i < n and (_QUOTE_RE.match(lines[i]) or not lines[i].strip()):
                if not lines[i].strip() and not (i + 1 < n and _QUOTE_RE.match(lines[i + 1])):
                    break
                buf.append(lines[i])
                i += 1
            blocks.append(Block("quote", "\n".join(buf).rstrip(), atomic=False))
            continue

        # Paragraph — runs to the next blank line or block start.
        buf = []
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                i += 1
                break
            s = nxt.strip()
            if s.startswith("```") or _TABLE_RE.match(nxt) or _LIST_RE.match(nxt) or _QUOTE_RE.match(nxt):
                break
            buf.append(nxt)
            i += 1
        text = "\n".join(buf).strip()
        if text:
            blocks.append(Block("paragraph", text, atomic=False))

    return blocks


# ---------------------------------------------------------------------------
# Oversized-block splitting — always at the block's own internal boundaries
# ---------------------------------------------------------------------------
def _split_code(block: Block, max_tokens: int, count: TokenCounter) -> list[str]:
    lines = block.text.split("\n")
    if len(lines) < 3:
        return [block.text]
    fence_open, fence_close = lines[0], lines[-1]
    language = fence_open.strip().lstrip("`")
    body = lines[1:-1]

    pieces: list[str] = []
    buf: list[str] = []
    budget = max_tokens - count(f"{fence_open}\n{fence_close}") - 12

    for line in body:
        line_tokens = count(line)
        if buf and count("\n".join(buf)) + line_tokens > budget:
            pieces.append("\n".join(buf))
            buf = []
        buf.append(line)
    if buf:
        pieces.append("\n".join(buf))

    total = len(pieces)
    out: list[str] = []
    for idx, piece in enumerate(pieces, start=1):
        marker = f"```{language}" if language else "```"
        suffix = f"\n<!-- code block part {idx}/{total} -->" if total > 1 else ""
        out.append(f"{marker}\n{piece}\n```{suffix}")
    return out


def _split_list(block: Block, max_tokens: int, count: TokenCounter) -> list[str]:
    lines = block.text.split("\n")
    items: list[list[str]] = []
    for line in lines:
        if _LIST_RE.match(line) and not line.startswith(("  ", "\t")):
            items.append([line])
        elif items:
            items[-1].append(line)
        else:
            items.append([line])

    pieces: list[str] = []
    buf: list[str] = []
    for item in items:
        item_text = "\n".join(item)
        if buf and count("\n".join(buf) + "\n" + item_text) > max_tokens:
            pieces.append("\n".join(buf))
            buf = []
        buf.append(item_text)
    if buf:
        pieces.append("\n".join(buf))

    total = len(pieces)
    if total > 1:
        # Say so explicitly: a partial procedure the model mistakes for a whole
        # one is the failure mode this whole module exists to prevent.
        pieces = [
            f"{p}\n\n(continued — step list part {i}/{total})"
            for i, p in enumerate(pieces, start=1)
        ]
    return pieces


def _split_table(block: Block, max_tokens: int, count: TokenCounter) -> list[str]:
    lines = block.text.split("\n")
    if len(lines) < 3:
        return [block.text]
    header = lines[:2]  # header row + separator
    header_text = "\n".join(header)
    rows = lines[2:]

    pieces: list[str] = []
    buf: list[str] = []
    for row in rows:
        if buf and count(header_text + "\n" + "\n".join(buf) + "\n" + row) > max_tokens:
            pieces.append(header_text + "\n" + "\n".join(buf))
            buf = []
        buf.append(row)
    if buf:
        pieces.append(header_text + "\n" + "\n".join(buf))
    return pieces


def _split_prose(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    sentences = _SENTENCE_RE.split(text)
    pieces: list[str] = []
    buf: list[str] = []

    for sentence in sentences:
        if buf and count(" ".join(buf + [sentence])) > max_tokens:
            pieces.append(" ".join(buf))
            buf = []
        if count(sentence) > max_tokens:
            # Pathological single sentence — fall back to word boundaries.
            words = sentence.split()
            wbuf: list[str] = []
            for word in words:
                if wbuf and count(" ".join(wbuf + [word])) > max_tokens:
                    pieces.append(" ".join(wbuf))
                    wbuf = []
                wbuf.append(word)
            if wbuf:
                buf = [" ".join(wbuf)]
            continue
        buf.append(sentence)

    if buf:
        pieces.append(" ".join(buf))
    return [p for p in pieces if p.strip()]


def split_oversized_block(block: Block, max_tokens: int, count: TokenCounter) -> list[str]:
    if block.kind == "code":
        return _split_code(block, max_tokens, count)
    if block.kind == "list":
        return _split_list(block, max_tokens, count)
    if block.kind == "table":
        return _split_table(block, max_tokens, count)
    return _split_prose(block.text, max_tokens, count)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
class RunbookChunker:
    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 96,
        overlap_tokens: int = 64,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if min_tokens >= max_tokens:
            raise ValueError("min_tokens must be < max_tokens")
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.count = token_counter or heuristic_token_count

    # -- public ------------------------------------------------------------
    def chunk(self, doc: SourceDocument) -> list[Chunk]:
        sections = split_sections(doc.markdown, doc.title)
        chunks: list[Chunk] = []

        for section in sections:
            kind = self._section_kind(section)
            texts = self._pack_section(section)
            for chunk_index, body in enumerate(texts):
                body = body.strip()
                if not body:
                    continue
                chunks.append(
                    self._build_chunk(doc, section, kind, chunk_index, body)
                )
        return chunks

    # -- internals ---------------------------------------------------------
    def _section_kind(self, section: Section) -> SectionKind:
        """Classify from the section title, falling back up the breadcrumb.

        A bare "Step 1" heading under "## Remediation" is remediation.
        """
        kind = classify_section(section.title)
        if kind is not SectionKind.OTHER:
            return kind
        for ancestor in reversed(section.path[:-1]):
            parent_kind = classify_section(ancestor)
            if parent_kind is not SectionKind.OTHER:
                return parent_kind
        return SectionKind.OTHER

    def _pack_section(self, section: Section) -> list[str]:
        """Greedily pack blocks into chunks, respecting atomicity."""
        chunks: list[str] = []
        buf: list[Block] = []

        def flush() -> None:
            if not buf:
                return
            chunks.append("\n\n".join(b.text for b in buf).strip())
            buf.clear()

        for block in section.blocks:
            block_tokens = self.count(block.text)

            if block_tokens > self.max_tokens:
                flush()
                for piece in split_oversized_block(block, self.max_tokens, self.count):
                    chunks.append(piece.strip())
                continue

            current_tokens = self.count("\n\n".join(b.text for b in buf)) if buf else 0
            if buf and current_tokens + block_tokens > self.max_tokens:
                flush()
            buf.append(block)

        flush()

        chunks = self._merge_undersized(chunks)
        return self._apply_overlap(chunks, section.blocks)

    def _merge_undersized(self, chunks: list[str]) -> list[str]:
        """Fold tiny trailing chunks back into their neighbour.

        A 12-token orphan chunk ("### Rollback") retrieves noisily and tells the
        model nothing.
        """
        if len(chunks) < 2:
            return chunks
        merged: list[str] = [chunks[0]]
        ceiling = int(self.max_tokens * 1.15)
        for chunk in chunks[1:]:
            if self.count(chunk) < self.min_tokens:
                candidate = merged[-1] + "\n\n" + chunk
                if self.count(candidate) <= ceiling:
                    merged[-1] = candidate
                    continue
            merged.append(chunk)
        return merged

    def _apply_overlap(self, chunks: list[str], blocks: list[Block]) -> list[str]:
        """Sentence-aligned overlap, suppressed across code boundaries."""
        if self.overlap_tokens <= 0 or len(chunks) < 2:
            return chunks

        out = [chunks[0]]
        for previous, current in zip(chunks, chunks[1:]):
            tail = self._tail(previous)
            if tail and not previous.rstrip().endswith("```"):
                out.append(f"{tail}\n\n{current}")
            else:
                out.append(current)
        return out

    def _tail(self, text: str) -> str:
        """Last whole sentences of `text` fitting in the overlap budget."""
        # Never carry a fence fragment forward — half a command is a hazard.
        if "```" in text.rsplit("\n\n", 1)[-1]:
            return ""
        sentences = _SENTENCE_RE.split(text.replace("\n", " ").strip())
        picked: list[str] = []
        for sentence in reversed(sentences):
            candidate = [sentence] + picked
            if self.count(" ".join(candidate)) > self.overlap_tokens:
                break
            picked = candidate
        return " ".join(picked).strip()

    def _build_chunk(
        self,
        doc: SourceDocument,
        section: Section,
        kind: SectionKind,
        chunk_index: int,
        body: str,
    ) -> Chunk:
        breadcrumb = " > ".join(section.path)
        # Contextual retrieval: the breadcrumb rides along in the embedded text
        # so an isolated step still lands near its parent topic in vector space.
        embed_text = build_embed_text(doc.title, breadcrumb, kind, body)
        basis = f"{doc.page_id}:{section.index}:{chunk_index}"
        chunk_id = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]

        return Chunk(
            chunk_id=chunk_id,
            page_id=doc.page_id,
            page_title=doc.title,
            page_url=doc.url,
            space_key=doc.space_key,
            page_version=doc.version,
            content_hash=doc.content_hash,
            section_path=list(section.path),
            section_kind=kind,
            section_index=section.index,
            chunk_index=chunk_index,
            body_text=body,
            embed_text=embed_text,
            token_count=self.count(embed_text),
            labels=list(doc.labels),
        )
