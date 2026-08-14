"""Command line interface.

    python -m triage.cli --help

Phase-1 workflow:

    triage config                     # what config.yaml actually resolved to
    triage check                      # config + Confluence + model reachable
    triage preview docs/sample.md     # inspect chunking, no model needed
    triage ingest confluence          # fetch, chunk, embed, index
    triage search "postgres pool exhausted"
    triage eval evalset/golden.jsonl  # measure, don't guess

Profiles from config.yaml can be swapped per command without editing anything:

    triage --embedding-profile bge-small search "..."
    triage --store-profile production stats
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from triage.config import ConfigError, get_settings, reset_settings_cache
from triage.logging_setup import configure_logging

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Runbook RAG: Confluence -> chunk -> Gemma embeddings -> hybrid retrieval.",
)
ingest_app = typer.Typer(no_args_is_help=True, help="Load runbooks into the index.")
app.add_typer(ingest_app, name="ingest")

console = Console()


@app.callback()
def main(
    config: Path = typer.Option(
        None, "--config", "-c", help="Path to config.yaml (default: ./config.yaml)."
    ),
    embedding_profile: str = typer.Option(
        "", "--embedding-profile", "-e", help="Override embedding.active for this command."
    ),
    store_profile: str = typer.Option(
        "", "--store-profile", "-s", help="Override vector_store.active for this command."
    ),
    reranker_profile: str = typer.Option(
        "", "--reranker-profile", "-r", help="Override reranker.active for this command."
    ),
    log_level: str = typer.Option("", "--log-level", help="DEBUG | INFO | WARNING | ERROR."),
) -> None:
    """Global overrides. These set the same env vars the loader already reads,
    so precedence stays exactly as documented in config.yaml."""
    if config:
        os.environ["CONFIG_FILE"] = str(config)
    if embedding_profile:
        os.environ["EMBEDDING_PROFILE"] = embedding_profile
    if store_profile:
        os.environ["VECTOR_STORE_PROFILE"] = store_profile
    if reranker_profile:
        os.environ["RERANKER_PROFILE"] = reranker_profile
    if log_level:
        os.environ["LOG_LEVEL"] = log_level
    # Settings are cached; the overrides above must land before the first read.
    reset_settings_cache()


def _settings(log_level: str | None = None):
    try:
        settings = get_settings()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(2) from exc
    configure_logging(log_level or settings.log_level, settings.log_json)
    return settings


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@app.command("config")
def config_show(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Show the fully resolved configuration and where each value came from."""
    settings = _settings()
    summary = settings.summary()

    if as_json:
        console.print_json(json.dumps(summary, default=str))
        return

    console.print(f"[bold]Config file:[/] {summary['config_file']}")
    for section, values in summary.items():
        if section == "config_file":
            continue
        table = Table(title=section, show_header=False, title_justify="left")
        table.add_column("key", style="dim")
        table.add_column("value")
        for key, value in values.items():
            style = ""
            if value in ("MISSING", "-") or str(value).startswith("MISSING"):
                style = "yellow"
            table.add_row(key, f"[{style}]{value}[/{style}]" if style else str(value))
        console.print(table)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
@app.command()
def check(
    skip_model: bool = typer.Option(False, help="Skip loading the embedding model."),
) -> None:
    """Verify configuration, Confluence connectivity and the embedding model."""
    settings = _settings()
    table = Table(title="Active profiles", show_header=True, header_style="bold")
    table.add_column("Section")
    table.add_column("Profile")
    table.add_column("Resolves to")
    table.add_row("embedding", settings.embedding_profile, f"{settings.embedding_provider} / {settings.embedding_model}")
    table.add_row("vector_store", settings.vector_store_profile, f"{settings.vector_backend} / {settings.vector_collection}")
    table.add_row("reranker", settings.reranker_profile, settings.reranker_provider)
    console.print(table)
    console.print(f"[dim]config: {settings.config_file or '(defaults only)'}[/]")

    ok = True

    if settings.confluence_configured:
        from triage.ingest.confluence import ConfluenceClient

        try:
            with ConfluenceClient(settings) as client:
                user = client.verify()
                console.print(
                    f"[green]OK[/] Confluence as "
                    f"[bold]{user.get('displayName', user.get('email', 'unknown'))}[/]"
                )
                console.print(f"     CQL: [dim]{settings.effective_cql()}[/]")
        except Exception as exc:
            ok = False
            console.print(f"[red]FAIL[/] Confluence: {exc}")
    else:
        console.print("[yellow]SKIP[/] Confluence not configured (set CONFLUENCE_* in .env)")

    if not skip_model:
        try:
            from triage.embeddings.factory import build_embedder

            embedder = build_embedder(settings)
            vector = embedder.embed_query("connection pool exhausted")
            console.print(
                f"[green]OK[/] Embedder [bold]{embedder.name}[/] "
                f"-> {len(vector)} dims"
            )
        except Exception as exc:
            ok = False
            console.print(f"[red]FAIL[/] Embedder: {exc}")

    try:
        from triage.vectorstore.factory import build_vector_store

        store = build_vector_store(settings)
        console.print(f"[green]OK[/] Store [bold]{store.name}[/] ({store.count()} chunks)")
    except Exception as exc:
        ok = False
        console.print(f"[red]FAIL[/] Vector store: {exc}")

    raise typer.Exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# preview — chunking inspection without touching a model
# ---------------------------------------------------------------------------
@app.command()
def preview(
    path: Path = typer.Argument(..., help="A .md file, or a Confluence storage-format .html/.xml file."),
    max_tokens: int = typer.Option(0, help="Override CHUNK_MAX_TOKENS."),
    show_body: bool = typer.Option(True, help="Print each chunk body."),
) -> None:
    """Chunk one file and print the result. No model, no network — instant.

    This is the fastest way to sanity-check chunking against a real runbook
    before committing to a full ingest.
    """
    settings = _settings()
    from triage.ingest.chunker import RunbookChunker
    from triage.ingest.normalize import storage_to_markdown
    from triage.types import SourceDocument

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".html", ".xml", ".storage"}:
        markdown = storage_to_markdown(text)
        console.print(Panel(Syntax(markdown[:2000], "markdown"), title="Normalized Markdown"))
    else:
        markdown = text

    title = path.stem.replace("-", " ").replace("_", " ").title()
    first = markdown.lstrip().split("\n", 1)[0].strip()
    if first.startswith("# "):
        title = first[2:].strip()

    doc = SourceDocument(
        page_id=f"preview:{path.name}",
        title=title,
        url=path.resolve().as_uri(),
        space_key="PREVIEW",
        version=1,
        labels=["preview"],
        markdown=markdown,
    )
    chunker = RunbookChunker(
        max_tokens=max_tokens or settings.chunk_max_tokens,
        min_tokens=settings.chunk_min_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    chunks = chunker.chunk(doc)

    table = Table(title=f"{len(chunks)} chunks from {path.name}")
    table.add_column("#", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("kind")
    table.add_column("breadcrumb")
    for i, chunk in enumerate(chunks, start=1):
        table.add_row(str(i), str(chunk.token_count), chunk.section_kind.value, chunk.breadcrumb)
    console.print(table)

    if show_body:
        for i, chunk in enumerate(chunks, start=1):
            console.print(
                Panel(
                    chunk.body_text,
                    title=f"[{i}] {chunk.breadcrumb}  ({chunk.token_count} tok, {chunk.section_kind.value})",
                    border_style="dim",
                )
            )

    oversized = [c for c in chunks if c.token_count > (max_tokens or settings.chunk_max_tokens) * 1.2]
    if oversized:
        console.print(
            f"[yellow]{len(oversized)} chunk(s) exceed the budget by >20% — "
            f"usually one very large atomic block (code/table).[/]"
        )


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
@ingest_app.command("confluence")
def ingest_confluence(
    force: bool = typer.Option(False, help="Re-embed pages even if unchanged."),
    limit: int = typer.Option(0, help="Stop after N pages (for a trial run)."),
) -> None:
    """Fetch runbooks from Confluence, chunk, embed and index them."""
    from triage.app import build_stack

    stack = build_stack()
    try:
        report = stack.pipeline.run_confluence(force=force, limit=limit or None)
    finally:
        stack.close()
    console.print_json(json.dumps(report.as_dict()))
    if report.pages_failed:
        raise typer.Exit(1)


@ingest_app.command("local")
def ingest_local(
    directory: Path = typer.Argument(..., help="Directory of .md runbooks."),
    force: bool = typer.Option(False, help="Re-embed even if unchanged."),
) -> None:
    """Ingest a folder of Markdown files — useful before Confluence is wired up."""
    from triage.app import build_stack

    stack = build_stack()
    try:
        report = stack.pipeline.run_local_markdown(directory, force=force)
    finally:
        stack.close()
    console.print_json(json.dumps(report.as_dict()))


@ingest_app.command("reindex-lexical")
def reindex_lexical() -> None:
    """Rebuild the BM25 sidecar from the vector store without re-embedding."""
    from triage.app import build_stack

    stack = build_stack()
    try:
        count = stack.pipeline.rebuild_lexical_index()
    finally:
        stack.close()
    console.print(f"[green]Lexical index rebuilt:[/] {count} documents")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
@app.command()
def search(
    query: str = typer.Argument(..., help="Free text: a symptom, an error string, a log line."),
    top_k: int = typer.Option(0, help="Override RETRIEVAL_FINAL_K."),
    space: str = typer.Option("", help="Filter by Confluence space key."),
    kind: str = typer.Option("", help="Filter by section kind (remediation, diagnosis, ...)."),
    expand: bool = typer.Option(True, help="Expand each hit to its full section."),
    show_body: bool = typer.Option(True, help="Print chunk bodies."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Run a hybrid search. This is the phase-1 acceptance test."""
    from triage.app import build_stack

    stack = build_stack()
    filters: dict = {}
    if space:
        filters["space_key"] = space
    if kind:
        filters["section_kind"] = [k.strip() for k in kind.split(",") if k.strip()]

    try:
        result = stack.retriever.search(
            query,
            top_k=top_k or None,
            filters=filters or None,
            expand_sections=expand,
        )
    finally:
        stack.close()

    if as_json:
        console.print_json(
            json.dumps(
                {
                    "query": result.query,
                    "took_ms": round(result.took_ms, 1),
                    "stats": result.stats,
                    "hits": [
                        {
                            "score": round(s.score, 5),
                            "rerank_score": s.rerank_score,
                            "source": s.source,
                            "dense_rank": s.dense_rank,
                            "lexical_rank": s.lexical_rank,
                            "page_title": s.chunk.page_title,
                            "breadcrumb": s.chunk.breadcrumb,
                            "section_kind": s.chunk.section_kind.value,
                            "url": s.chunk.page_url,
                            "chunk_id": s.chunk.chunk_id,
                            "body": s.chunk.body_text,
                        }
                        for s in result.chunks
                    ],
                }
            )
        )
        return

    if not result.chunks:
        console.print("[yellow]No results.[/] Has anything been ingested? Try `triage stats`.")
        return

    table = Table(title=f'"{query}"  ({result.took_ms:.0f} ms)')
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("src")
    table.add_column("d", justify="right")
    table.add_column("l", justify="right")
    table.add_column("kind")
    table.add_column("page > section")
    for i, hit in enumerate(result.chunks, start=1):
        score = hit.rerank_score if hit.rerank_score is not None else hit.score
        table.add_row(
            str(i),
            f"{score:.4f}",
            hit.source,
            "-" if hit.dense_rank is None else str(hit.dense_rank + 1),
            "-" if hit.lexical_rank is None else str(hit.lexical_rank + 1),
            hit.chunk.section_kind.value,
            f"{hit.chunk.page_title} > {hit.chunk.breadcrumb}",
        )
    console.print(table)
    console.print(f"[dim]{result.stats}[/]")

    if show_body:
        source = result.expanded if (expand and result.expanded) else [s.chunk for s in result.chunks]
        for i, chunk in enumerate(source, start=1):
            console.print(
                Panel(
                    chunk.body_text,
                    title=f"[{i}] {chunk.page_title} > {chunk.breadcrumb}",
                    subtitle=chunk.page_url,
                    border_style="dim",
                )
            )


# ---------------------------------------------------------------------------
# stats / inspect / reset
# ---------------------------------------------------------------------------
@app.command()
def stats() -> None:
    """Index size, page count and lexical-index health."""
    from triage.app import build_stack

    stack = build_stack()
    try:
        states = stack.store.page_states()
        total = stack.store.count()
        console.print(f"[bold]Store:[/] {stack.store.name}")
        console.print(f"[bold]Embedder:[/] {stack.embedder.name} ({stack.embedder.dimension} dims)")
        console.print(f"[bold]Chunks:[/] {total}")
        console.print(f"[bold]Pages:[/] {len(states)}")

        lexical_path = stack.settings.lexical_index_path
        if hasattr(stack.store, "lexical_search"):
            console.print("[bold]Lexical:[/] native (Postgres full-text)")
        elif lexical_path.exists():
            from triage.retrieval.bm25 import BM25Index

            index = BM25Index.load(lexical_path)
            status = "[green]in sync[/]" if index.size == total else "[yellow]STALE — run `triage ingest reindex-lexical`[/]"
            console.print(f"[bold]Lexical:[/] BM25, {index.size} docs {status}")
        else:
            console.print("[bold]Lexical:[/] [yellow]missing — dense-only retrieval[/]")

        if states:
            table = Table(title="Pages")
            table.add_column("page_id")
            table.add_column("version", justify="right")
            table.add_column("chunks", justify="right")
            for state in sorted(states.values(), key=lambda s: -s.chunk_count)[:40]:
                table.add_row(state.page_id, str(state.page_version), str(state.chunk_count))
            console.print(table)
    finally:
        stack.close()


@app.command()
def inspect(
    page_id: str = typer.Argument(..., help="Confluence page id, or a local: id."),
) -> None:
    """Show every indexed chunk for one page, in order."""
    from triage.app import build_stack

    stack = build_stack()
    try:
        chunks = [c for c in stack.store.iter_chunks() if c.page_id == page_id]
    finally:
        stack.close()

    if not chunks:
        console.print(f"[yellow]No chunks for page {page_id}.[/]")
        raise typer.Exit(1)

    chunks.sort(key=lambda c: (c.section_index, c.chunk_index))
    console.print(f"[bold]{chunks[0].page_title}[/] — {len(chunks)} chunks")
    for chunk in chunks:
        console.print(
            Panel(
                chunk.body_text,
                title=f"{chunk.breadcrumb}  ({chunk.token_count} tok, {chunk.section_kind.value})",
                border_style="dim",
            )
        )


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", help="Confirm deletion."),
) -> None:
    """Drop the collection and the lexical index. Destructive."""
    if not yes:
        console.print("[red]Refusing to delete without --yes[/]")
        raise typer.Exit(1)
    from triage.app import build_stack

    stack = build_stack()
    try:
        stack.store.reset()
        path = stack.settings.lexical_index_path
        if path.exists():
            path.unlink()
    finally:
        stack.close()
    console.print("[green]Index cleared.[/]")


# ---------------------------------------------------------------------------
# eval / serve
# ---------------------------------------------------------------------------
@app.command("eval")
def run_eval(
    golden: Path = typer.Argument(..., help="JSONL golden set."),
    k: int = typer.Option(0, help="Override RETRIEVAL_FINAL_K."),
    show_misses: bool = typer.Option(True, help="Print the cases that missed."),
) -> None:
    """Score retrieval against a golden set: recall@k, MRR, nDCG@k."""
    from triage.app import build_stack
    from triage.evaluation.retrieval_eval import evaluate, load_cases

    cases = load_cases(golden)
    stack = build_stack()
    try:
        report = evaluate(stack.retriever, cases, k=k or stack.settings.retrieval_final_k)
    finally:
        stack.close()

    table = Table(title=f"Retrieval eval ({len(cases)} cases, k={report.k})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("recall@k", f"{report.recall_at_k:.3f}")
    table.add_row("MRR", f"{report.mrr:.3f}")
    table.add_row("nDCG@k", f"{report.ndcg:.3f}")
    console.print(table)

    if show_misses and report.misses:
        miss_table = Table(title=f"{len(report.misses)} misses", header_style="red")
        miss_table.add_column("query")
        miss_table.add_column("top result")
        miss_table.add_column("note")
        for case in report.misses:
            miss_table.add_row(
                case.query[:70],
                (case.retrieved[0] if case.retrieved else "-")[:60],
                case.notes[:40],
            )
        console.print(miss_table)


@app.command()
def serve(
    host: str = typer.Option("", help="Override API_HOST."),
    port: int = typer.Option(0, help="Override API_PORT."),
) -> None:
    """Run the search API (GET /healthz, POST /v1/search)."""
    import uvicorn

    settings = _settings()
    uvicorn.run(
        "triage.api:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    app()
