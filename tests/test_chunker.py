"""Chunker invariants.

These assert the properties that make the difference between a usable runbook
chunk and a dangerous one — a half-written command, or a procedure that starts
at step 4.
"""

from __future__ import annotations

import pytest

from triage.ingest.chunker import (
    RunbookChunker,
    classify_section,
    heuristic_token_count,
    split_blocks,
    split_sections,
)
from triage.types import SectionKind, SourceDocument

RUNBOOK = """# Redis Eviction Storm

## Symptoms

Cache hit rate collapses below 40% and latency spikes.

## Diagnosis

Check the eviction counter:

```bash
redis-cli INFO stats | grep evicted_keys
```

## Remediation

1. Raise maxmemory on the replica first.
2. Fail over to the replica.
3. Raise maxmemory on the former primary.
4. Fail back once both nodes are stable.

## Escalation

Page the Cache team.
"""


def make_doc(markdown: str, title: str = "Test Runbook") -> SourceDocument:
    return SourceDocument(
        page_id="p1",
        title=title,
        url="https://example.test/p1",
        space_key="SRE",
        version=1,
        labels=["runbook"],
        markdown=markdown,
    )


# --- section parsing -------------------------------------------------------
def test_sections_carry_breadcrumb_path():
    sections = split_sections(RUNBOOK, "Redis Eviction Storm")
    titles = [s.title for s in sections]
    assert "Symptoms" in titles
    assert "Remediation" in titles

    remediation = next(s for s in sections if s.title == "Remediation")
    # The page title is not repeated in the path — it lives on the Chunk.
    assert remediation.path == ["Remediation"]


def test_body_h1_matching_page_title_is_not_duplicated_in_path():
    sections = split_sections("# Redis Eviction Storm\n\n## Symptoms\n\nCache misses.\n", "Redis Eviction Storm")
    symptoms = next(s for s in sections if s.title == "Symptoms")
    assert symptoms.path == ["Symptoms"]


def test_body_h1_differing_from_page_title_is_kept():
    sections = split_sections("# Overview\n\n## Symptoms\n\nCache misses.\n", "Some Other Page")
    symptoms = next(s for s in sections if s.title == "Symptoms")
    assert symptoms.path == ["Overview", "Symptoms"]


def test_hash_inside_code_fence_is_not_a_heading():
    markdown = """# Title

Intro prose, so the root section is not empty.

## Diagnosis

```bash
# this is a shell comment, not a heading
echo hello
```

Trailing prose.
"""
    sections = split_sections(markdown, "Title")
    assert [s.title for s in sections] == ["Title", "Diagnosis"]
    # The shell comment must not have been promoted into a section of its own.
    assert not any("shell comment" in s.title for s in sections)


def test_heading_with_no_body_is_dropped():
    # An H1 immediately followed by an H2 carries no content of its own, so it
    # would only ever produce an orphan chunk.
    sections = split_sections("# Title\n\n## Diagnosis\n\nReal content here.\n", "Title")
    assert [s.title for s in sections] == ["Diagnosis"]


def test_nested_headings_build_full_path():
    markdown = "# A\n\ntext\n\n## B\n\ntext\n\n### C\n\ntext\n"
    sections = split_sections(markdown, "A")
    deepest = next(s for s in sections if s.title == "C")
    assert deepest.path == ["B", "C"]


# --- block splitting -------------------------------------------------------
def test_blocks_are_typed_and_code_is_atomic():
    blocks = split_blocks("Prose here.\n\n```bash\nls -la\n```\n\n| a | b |\n| --- | --- |\n| 1 | 2 |")
    kinds = [b.kind for b in blocks]
    assert "paragraph" in kinds
    assert "code" in kinds
    assert "table" in kinds
    code = next(b for b in blocks if b.kind == "code")
    assert code.atomic is True
    assert code.text.startswith("```bash")
    assert code.text.rstrip().endswith("```")


def test_ordered_list_is_one_atomic_block():
    blocks = split_blocks("1. first\n2. second\n3. third\n")
    assert len(blocks) == 1
    assert blocks[0].kind == "list"
    assert blocks[0].atomic is True


# --- classification --------------------------------------------------------
@pytest.mark.parametrize(
    "title,expected",
    [
        ("Remediation", SectionKind.REMEDIATION),
        ("Rollback", SectionKind.ROLLBACK),
        ("How to roll back the deploy", SectionKind.ROLLBACK),
        ("Escalation", SectionKind.ESCALATION),
        ("Who to page", SectionKind.ESCALATION),
        ("Symptoms", SectionKind.SYMPTOMS),
        ("Diagnosis", SectionKind.DIAGNOSIS),
        ("Verification", SectionKind.VERIFICATION),
        ("Impact", SectionKind.IMPACT),
        ("Overview", SectionKind.OVERVIEW),
        ("Bananas", SectionKind.OTHER),
    ],
)
def test_section_classification(title, expected):
    assert classify_section(title) is expected


def test_rollback_wins_over_remediation():
    # "roll back the fix" contains both signals; rollback must win.
    assert classify_section("Roll back the fix") is SectionKind.ROLLBACK


def test_kind_inherits_from_ancestor_when_title_is_generic():
    markdown = "# Runbook\n\n## Remediation\n\n### Step 1\n\nDo the thing carefully and completely.\n"
    chunks = RunbookChunker().chunk(make_doc(markdown))
    step = next(c for c in chunks if c.section_path[-1] == "Step 1")
    assert step.section_kind is SectionKind.REMEDIATION


# --- the load-bearing invariants ------------------------------------------
def test_code_fences_are_never_split_across_chunks():
    chunks = RunbookChunker(max_tokens=120, min_tokens=20, overlap_tokens=0).chunk(
        make_doc(RUNBOOK)
    )
    for chunk in chunks:
        # An unbalanced fence count means a chunk boundary landed inside a
        # command — the exact failure this design exists to prevent.
        assert chunk.body_text.count("```") % 2 == 0, chunk.body_text


def test_numbered_procedure_stays_in_one_chunk():
    chunks = RunbookChunker(max_tokens=512, min_tokens=32, overlap_tokens=0).chunk(
        make_doc(RUNBOOK)
    )
    holder = [c for c in chunks if "1. Raise maxmemory on the replica" in c.body_text]
    assert len(holder) == 1
    body = holder[0].body_text
    for step in ("1.", "2.", "3.", "4."):
        assert step in body, f"step {step} was separated from its procedure"


def test_oversized_list_splits_between_items_never_mid_item():
    steps = "\n".join(f"{i}. Step number {i} with some explanatory text attached." for i in range(1, 40))
    chunks = RunbookChunker(max_tokens=100, min_tokens=20, overlap_tokens=0).chunk(
        make_doc(f"# Big\n\n## Remediation\n\n{steps}\n")
    )
    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.body_text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                # Every surviving step line must still carry its full sentence.
                assert line.rstrip().endswith("."), line


def test_oversized_code_block_splits_on_line_boundaries_and_stays_fenced():
    body = "\n".join(f"kubectl get pods --namespace ns-{i} --output wide" for i in range(80))
    chunks = RunbookChunker(max_tokens=100, min_tokens=20, overlap_tokens=0).chunk(
        make_doc(f"# Big\n\n## Diagnosis\n\n```bash\n{body}\n```\n")
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.body_text.count("```") % 2 == 0
        for line in chunk.body_text.split("\n"):
            if line.startswith("kubectl"):
                assert line.endswith("wide"), f"command was cut mid-line: {line!r}"


def test_oversized_table_repeats_header_in_every_piece():
    rows = "\n".join(f"| check-{i} | command-{i} | expected-value-{i} |" for i in range(60))
    markdown = f"# T\n\n## Verification\n\n| Check | Command | Expected |\n| --- | --- | --- |\n{rows}\n"
    chunks = RunbookChunker(max_tokens=100, min_tokens=20, overlap_tokens=0).chunk(
        make_doc(markdown)
    )
    table_chunks = [c for c in chunks if "check-" in c.body_text]
    assert len(table_chunks) > 1
    for chunk in table_chunks:
        assert "| Check | Command | Expected |" in chunk.body_text


# --- context and identity --------------------------------------------------
def test_embed_text_carries_breadcrumb_context():
    chunks = RunbookChunker().chunk(make_doc(RUNBOOK))
    chunk = next(c for c in chunks if c.section_kind is SectionKind.REMEDIATION)
    assert "Redis Eviction Storm" in chunk.embed_text
    assert "Remediation" in chunk.embed_text
    # The body itself stays clean for display.
    assert not chunk.body_text.startswith("Runbook:")


def test_chunk_ids_are_deterministic_across_runs():
    first = RunbookChunker().chunk(make_doc(RUNBOOK))
    second = RunbookChunker().chunk(make_doc(RUNBOOK))
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_ids_differ_across_pages():
    doc_a = make_doc(RUNBOOK)
    doc_b = SourceDocument(
        page_id="p2", title="Other", url="u", space_key="SRE", version=1,
        labels=[], markdown=RUNBOOK,
    )
    ids_a = {c.chunk_id for c in RunbookChunker().chunk(doc_a)}
    ids_b = {c.chunk_id for c in RunbookChunker().chunk(doc_b)}
    assert not (ids_a & ids_b)


def test_empty_document_produces_no_chunks():
    assert RunbookChunker().chunk(make_doc("")) == []


def test_min_tokens_must_be_below_max():
    with pytest.raises(ValueError):
        RunbookChunker(max_tokens=100, min_tokens=100)


def test_token_estimate_is_monotonic():
    assert heuristic_token_count("") == 0
    assert heuristic_token_count("one two three") < heuristic_token_count("one two three four five six")
