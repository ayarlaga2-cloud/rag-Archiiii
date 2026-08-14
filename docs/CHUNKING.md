# Chunking runbooks: the method and why

Chunking is the highest-leverage decision in a RAG system and the one most
often made by accident. This document states what this repo does, why, and how
to tune it.

## The short version

**Split on document structure, never on character count. Keep procedures whole.
Carry the heading path into the embedded text.**

Implementation: [`src/triage/ingest/chunker.py`](../src/triage/ingest/chunker.py).

---

## Why the common approach fails here

The default in most tutorials is a fixed-size sliding window — "1000 characters,
200 overlap". On a runbook it produces this:

```
...4. Drain the node before
─── chunk boundary ───
restarting. This will page the on-call if...
```

Retrieve that during an incident and the model confidently tells an engineer to
drain a node, with the consequence sitting in a chunk that was never retrieved.
A half-procedure is worse than no procedure, because it still looks complete.

Runbooks are unusually hostile to naive chunking because they are:

- **procedural** — step 4 is meaningless without steps 1–3, and dangerous without step 5
- **dense in exact identifiers** — `PSQLException`, `ECONNREFUSED`, `cl_waiting`
- **structured** — the heading is half the meaning ("Rollback" vs "Remediation")
- **command-bearing** — a command split across two chunks is a broken command

## Comparison of methods

| Method | Boundary | Runbook verdict |
|---|---|---|
| Fixed-size window | every N chars | Splits commands and procedures. Do not use. |
| Recursive character split | paragraph → sentence → char | Better, but still splits numbered lists and code blocks. |
| Sentence-window | sentence, retrieve neighbours | Good for prose; procedures still fragment. |
| Semantic (embedding-similarity) | topic shift | Expensive, non-deterministic, and it ignores the explicit structure the author already wrote. |
| **Structural + atomic blocks** (this repo) | heading, then block | Boundaries land where the author put them. |

The key insight: a runbook author has *already* marked the semantic boundaries —
with headings, numbered lists, and code fences. Inferring boundaries statistically
when they are stated explicitly in the markup is strictly worse.

---

## The method, stage by stage

### 1. Normalize to Markdown, preserving structure

Confluence stores pages as XHTML with an `ac:` macro namespace. A generic
HTML→text converter silently drops `<ac:structured-macro ac:name="code">` bodies
— which is where every command in a Confluence runbook lives.

[`normalize.py`](../src/triage/ingest/normalize.py) is a purpose-built walker
that preserves the three things chunking depends on:

- heading hierarchy → becomes the breadcrumb
- ordered-list numbering → a procedure is only meaningful whole
- code fences → commands stay verbatim and stay atomic

It also keeps info/warning panels (as labelled blockquotes) and `expand` macro
bodies — collapsed content is often exactly the deep-dive an on-call needs.

### 2. Split on the heading tree

Each section becomes a candidate chunk, carrying its full breadcrumb path
(`Postgres Connection Exhaustion > Remediation`). A `#` inside a fenced code
block is a shell comment, not a heading, and is treated as such.

If the page repeats its own title as a body `H1`, that duplicate is dropped from
the path — otherwise every breadcrumb reads `Title > Title > Section` and you
pay the duplicated tokens in every single vector.

### 3. Classify each section

Sections are tagged: `symptoms`, `impact`, `diagnosis`, `remediation`,
`rollback`, `verification`, `escalation`, `prerequisites`, `references`,
`overview`. Generic headings (`Step 1`) inherit from their ancestor.

This makes intent-scoped retrieval possible — "how do I roll back" can filter to
`section_kind=rollback` instead of hoping the ranker figures it out:


```bash
triage search "how do I roll back checkout-api" --kind rollback
```

### 4. Pack blocks, treating some as atomic

Within a section, content is grouped into blocks: paragraph, code, list, table,
quote. **Code, list and table blocks are atomic** — a chunk boundary never lands
inside one. Blocks are packed greedily up to the token budget.

### 5. Split oversized blocks at their *own* boundaries

When one atomic block genuinely exceeds the budget, it is split at its internal
structure, never mid-line:

| Block | Split at | Extra care |
|---|---|---|
| Code | line boundaries | each piece re-fenced, labelled `part i/n` |
| List | between top-level items | each piece labelled `step list part i/n` |
| Table | between rows | **header row repeated** in every piece |
| Prose | sentence, then word | last resort only |

The `part i/n` labels are deliberate: if a procedure must be split, the text must
say so, or the model will treat a fragment as the whole thing.

### 6. Overlap — sentence-aligned, and suppressed across code

Overlap exists so a sentence spanning a boundary is not lost. It carries whole
sentences, never a fragment, and is **skipped after a code fence** — a duplicated
half-command is actively misleading, not helpful redundancy.

### 7. Merge undersized chunks

A 12-token orphan (`### Rollback` with one line under it) retrieves noisily and
tells the model nothing. Trailing chunks below `chunking.min_tokens` are folded
back into their neighbour.

### 8. Contextual retrieval: embed the breadcrumb

This is the cheapest large win in the whole pipeline. Each chunk is embedded as:

```
Runbook: Postgres Connection Exhaustion
Section: Remediation
Type: remediation

1. Terminate the runaway backend...
```

while `body_text` stays clean for display. Without the envelope, a chunk reading
"1. Terminate the runaway backend" embeds near *any* restart procedure. With it,
it embeds near Postgres connection problems specifically.

### 9. Expand back to the section at retrieval time

Retrieval returns chunks; `retrieval.expand_section: true` then pulls the sibling
chunks of each matched section. A hit on step 4 returns steps 1–8. This costs one
metadata lookup and is the difference between a usable answer and a hazardous one.

---

## Parameters

Set in the `chunking:` block of [`config.yaml`](../config.yaml).

| Setting | Default | Guidance |
|---|---|---|
| `chunking.max_tokens` | 512 | EmbeddingGemma accepts 2048, but a 2048-token chunk averages several topics into one vector and loses precision. 512 keeps roughly one procedure per vector. Raise toward 1024 only for prose-heavy pages. |
| `chunking.min_tokens` | 96 | Below this, trailing chunks merge upward. |
| `chunking.overlap_tokens` | 64 | ~12% of max. Structural boundaries already fall in sensible places, so heavy overlap mostly buys duplicate storage. |

Token counts use **Gemma's own tokenizer** when the model is loaded, so
`max_tokens` means real Gemma tokens rather than a guess.

## Verifying it on your own runbooks

Inspect chunking with no model and no network:

```bash
triage preview path/to/runbook.md
```

Check three things:

1. **No chunk has an odd number of ``` markers** — that would mean a split fence.
2. **Numbered procedures appear whole**, or are explicitly labelled `part i/n`.
3. **`kind` classification looks right** — if your headings use house vocabulary
   ("Fix It", "War Room"), extend `_KIND_PATTERNS` in `chunker.py`.

The invariants above are enforced by tests in
[`tests/test_chunker.py`](../tests/test_chunker.py) — including that oversized
code blocks never split mid-line and oversized tables repeat their header.

## When to revisit

Change chunking, then re-run `triage eval` against the golden set. That is the
whole point of having one — chunk-size arguments are unwinnable without numbers,
and every change here is cheap to measure.
