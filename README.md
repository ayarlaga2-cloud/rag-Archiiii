# Runbook RAG

Retrieval over Confluence runbooks: **Confluence → structural chunking →
EmbeddingGemma → hybrid retrieval (dense + BM25 + RRF + optional rerank)**.

Phase 1 of a larger system. Jira/Slack integration and the triage agent come
later and are deliberately not built yet — this layer has to be measurably good
first, because everything downstream inherits its failures.

---

## Configuration

**One file: `config.yaml`.** Connection details, credentials, models, chunking
and retrieval tuning all live there.

```bash
copy config.example.yaml config.yaml     # Windows
cp   config.example.yaml config.yaml     # macOS / Linux
```

`config.yaml` is **gitignored** — [`config.example.yaml`](config.example.yaml)
is the committed template. That keeps the one-file workflow while making it hard
to commit a token by accident. If you ever copy the file somewhere tracked,
strip the secrets first.

Environment variables override any field and are still the right answer for CI,
where a credential on disk is the wrong shape — e.g. `CONFLUENCE_API_TOKEN`,
`EMBEDDING_PROFILE`. Nothing requires them locally.

`triage config` prints everything that resolved, with credentials shown as
`set`/`-` rather than echoed, so it is safe to paste into a ticket.

Model, store and reranker choices are **named profiles**, so switching is one
line:

```yaml
embedding:
  active: gemma        # <- change this
  profiles:
    gemma:      { provider: gemma,   model: google/embeddinggemma-300m }
    gemma-256:  { provider: gemma,   model: google/embeddinggemma-300m, truncate_dim: 256 }
    bge-small:  { provider: sentence-transformers, model: BAAI/bge-small-en-v1.5 }
    voyage:     { provider: voyage,  model: voyage-3 }
    hashing:    { provider: hashing, model: hashing-384 }
```

Or override for a single command, without editing anything:

```bash
triage --embedding-profile bge-small search "..."
triage --store-profile production stats
triage --config config.staging.yaml ingest confluence
```

Precedence is **environment variable > config.yaml > default**, so CI and
one-off overrides always win. `triage config` prints exactly what resolved,
with secrets masked.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install -e .
copy config.example.yaml config.yaml
```

Verify the pipeline **before** downloading a 2 GB model — the `hashing` profile
runs the full path on numpy alone:

```bash
triage --embedding-profile hashing ingest local samples/runbooks
triage --embedding-profile hashing search "postgres connection pool exhausted"
```

Then switch to the real embedder:

```bash
pip install -r requirements-embeddings.txt
huggingface-cli login          # EmbeddingGemma is a gated model
triage config                  # confirm what resolved
triage check                   # auth + CQL + model + store
```

## Connecting Confluence

Fill in the `confluence:` block of `config.yaml`:

```yaml
confluence:
  base_url: "https://your-site.atlassian.net/wiki"   # /wiki suffix required
  email: "you@example.com"        # same account that made the token
  api_token: "ATATT..."           # id.atlassian.com/manage-profile/security/api-tokens
  space_keys: ["SRE"]
  labels: ["runbook"]
```

The token must belong to the account in `email`, or auth returns 401.

Selection is CQL. The space/label settings compose into
`type = page AND space in ("SRE") AND label in ("runbook")`; set
`confluence.cql` to take full manual control.

```bash
triage check                        # auth + CQL + model + store
triage ingest confluence --limit 5  # trial run on 5 pages first
triage ingest confluence            # full sync
```

Ingest is **incremental** — a page is re-embedded only when its Confluence
version or its normalized-content hash changed. Re-running is cheap.

## Testing retrieval

```bash
triage search "PSQLException too many clients"     # exact identifier
triage search "checkout is 503ing, db refusing connections"   # paraphrase
triage search "how do I roll back" --kind rollback # intent-scoped
triage search "..." --json                          # machine-readable

triage config                   # resolved settings, secrets masked
triage stats                    # index size, page list, lexical health
triage inspect <page-id>        # every chunk of one page, in order
triage preview runbook.md       # chunking only — no model, no network
triage eval evalset/golden.jsonl
```

`triage eval` is the one that matters. Eyeballing a few queries finds the
failures you thought of; a golden set finds the ones you didn't, and turns every
later change into a measurement instead of an argument.

## Architecture

```
Confluence (CQL)
   ↓  confluence.py     paginate, retry on 429/5xx
   ↓  normalize.py      storage-format XHTML → Markdown (macros preserved)
   ↓  chunker.py        heading tree → atomic blocks → contextual envelope
   ↓  gemma.py          EmbeddingGemma, task-prefixed
   ↓  chroma / pgvector
                        ┌── dense ANN ──┐
query → build_queries ──┤               ├── RRF ── rerank ── section expand
                        └── BM25 / FTS ─┘
```

| Layer | Local | Production | Switch by |
|---|---|---|---|
| Embeddings | EmbeddingGemma (CPU) | same, or Voyage | `embedding.active` |
| Vector store | Chroma (on disk) | Postgres + pgvector | `vector_store.active` |
| Lexical | BM25 sidecar (JSON) | Postgres full-text | automatic |
| Rerank | off | cross-encoder / Voyage | `reranker.active` |

Nothing above the store layer knows which backend it is talking to, so the
production move is a config change. The `production` profile additionally moves
the lexical half of retrieval into the database — the part of the local setup
that does not scale.

## Why hybrid retrieval

Dense alone and lexical alone fail in opposite directions, and both failures are
common in incident response:

- **Dense misses exact identifiers.** An embedding model places
  `PSQLException: FATAL: sorry, too many clients already` near "database
  problems" generally — it will not reliably rank the one runbook containing
  that literal string first.
- **Lexical misses paraphrase.** BM25 cannot connect "the DB is refusing
  connections" to "connection pool exhausted" — no shared terms.

RRF fuses them on rank, so no score calibration is needed between two
incomparable scales. The tokenizer is tuned for this corpus: it splits
camelCase and `ACRONYMWord`, and emits both `org.postgresql.util.PSQLException`
and its parts, so a query saying `psqlexception` still matches.

## Chunking

The core of the system. Full rationale in **[docs/CHUNKING.md](docs/CHUNKING.md)**.

In one line: **split on document structure, never on character count; keep
procedures whole; carry the heading path into the embedded text.**

- Sections come from the heading tree; each chunk knows its breadcrumb.
- Code fences, lists and tables are **atomic** — a boundary never lands inside a
  command or between steps of a procedure.
- Oversized blocks split at their *own* boundaries (between list items, between
  table rows with the header repeated, between code lines), labelled `part i/n`.
- Overlap is sentence-aligned and suppressed across code fences.
- Each chunk embeds with a `Runbook: … / Section: … / Type: …` envelope, so an
  isolated "Step 3" still lands near its parent topic.
- At query time, a hit expands to its whole section — retrieving step 4 of 8
  returns all eight.

## EmbeddingGemma notes

`google/embeddinggemma-300m`: 308M params, 768 dims, 2048-token window, CPU-viable.

It is **asymmetric and prompt-driven** — queries and documents take different
task prefixes, and omitting them is a quiet, significant recall loss. This is
handled in [`embeddings/gemma.py`](src/triage/embeddings/gemma.py) via
`encode_query()` / `encode_document()`.

It is Matryoshka-trained, so the `gemma-256` profile shrinks the index with a
small accuracy cost and no retraining — worth measuring against your eval set
once the corpus is real.

It is a **gated** model: accept the licence on HuggingFace, then either run
`huggingface-cli login` or set `embedding.hf_token` in `config.yaml`.

## Tuning order

Change one thing, re-run `triage eval`, keep what wins:

1. **Turn on the reranker** — `reranker.active: local`. Usually the single
   largest gain; a cross-encoder reads query and document jointly instead of
   comparing two independent summaries.
2. **Chunk size** — `chunking.max_tokens`. 512 → 256 raises precision, lowers
   recall; 512 → 1024 the reverse.
3. **Fusion weights** — raise `retrieval.lexical_weight` if your corpus is
   identifier-heavy.
4. **Candidate depth** — `retrieval.dense_k` / `lexical_k`. More candidates only
   helps if a reranker is doing the sorting.

Because profiles are switchable per command, A/B is a one-liner:

```bash
triage --embedding-profile gemma      eval evalset/golden.jsonl
triage --embedding-profile gemma-256  eval evalset/golden.jsonl
```

## Project layout

```
config.yaml            all config incl. credentials (gitignored)
config.example.yaml    committed template — copy it to config.yaml
src/triage/
  config.py            YAML + env loader, profile resolution
  app.py               composition root
  ingest/              confluence · normalize · chunker · pipeline
  embeddings/          gemma · local(ST) · voyage · hashing
  vectorstore/         chroma · pgvector  (one protocol)
  retrieval/           bm25 · fusion · rerank · query · retriever
  evaluation/          recall@k · MRR · nDCG@k
  cli.py  api.py
tests/                 96 tests, no network or model required
samples/runbooks/      two realistic runbooks
evalset/golden.jsonl   starter golden set
```

```bash
pytest        # 96 tests, runs offline
```

## Status

- **Phase 1 (this):** Confluence → chunk → Gemma → hybrid retrieval ✅
- **Phase 2:** Jira + Slack integration
- **Phase 3:** triage agent (PagerDuty incident + Slack summary)
