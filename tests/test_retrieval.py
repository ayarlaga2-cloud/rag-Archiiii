"""Retrieval building blocks: tokenizer, BM25, RRF, query signals."""

from __future__ import annotations

from triage.retrieval.bm25 import BM25Index, tokenize
from triage.retrieval.fusion import reciprocal_rank_fusion
from triage.retrieval.query import build_queries, extract_signals
from triage.types import Chunk, ScoredChunk, SectionKind


def make_chunk(chunk_id: str, text: str, page: str = "Page") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        page_id="p1",
        page_title=page,
        page_url="https://example.test",
        space_key="SRE",
        page_version=1,
        content_hash="abc",
        section_path=[page, "Remediation"],
        section_kind=SectionKind.REMEDIATION,
        section_index=0,
        chunk_index=0,
        body_text=text,
        embed_text=text,
        token_count=len(text.split()),
    )


# --- tokenizer -------------------------------------------------------------
def test_camel_case_is_split_and_kept_whole():
    tokens = tokenize("NullPointerException")
    assert "null" in tokens and "pointer" in tokens and "exception" in tokens


def test_acronym_boundary_is_split():
    tokens = tokenize("PSQLException")
    assert "psql" in tokens and "exception" in tokens


def test_qualified_name_yields_both_full_and_parts():
    tokens = tokenize("org.postgresql.util.PSQLException")
    assert any("postgresql" in t for t in tokens)
    assert "util" in tokens


def test_numeric_status_codes_survive():
    assert "503" in tokenize("returned status 503 repeatedly")


def test_underscored_identifier_yields_parts():
    tokens = tokenize("ERR_POOL_EXHAUSTED")
    assert "pool" in tokens and "exhausted" in tokens


# --- BM25 ------------------------------------------------------------------
def test_bm25_ranks_exact_identifier_first():
    index = BM25Index()
    index.build(
        [
            make_chunk("a", "Generic advice about databases and latency tuning."),
            make_chunk("b", "org.postgresql.util.PSQLException FATAL sorry too many clients already"),
            make_chunk("c", "Kafka consumer lag remediation and partition rebalancing."),
        ]
    )
    ranked = index.search("PSQLException too many clients", k=3)
    assert ranked
    assert ranked[0][0] == "b"


def test_bm25_returns_nothing_for_unknown_terms():
    index = BM25Index()
    index.build([make_chunk("a", "postgres connection pool")])
    assert index.search("zzzz unrelated qqqq", k=5) == []


def test_bm25_empty_index_is_safe():
    assert BM25Index().search("anything", k=5) == []


def test_bm25_roundtrips_through_disk(tmp_path):
    index = BM25Index()
    index.build([make_chunk("a", "pgbouncer pool exhausted"), make_chunk("b", "kafka lag")])
    path = tmp_path / "lex.json"
    index.save(path)

    reloaded = BM25Index.load(path)
    assert reloaded.size == 2
    assert reloaded.search("pgbouncer", k=1)[0][0] == "a"


# --- fusion ----------------------------------------------------------------
def test_rrf_rewards_agreement_between_retrievers():
    shared = make_chunk("shared", "agreed result")
    dense_only = make_chunk("dense", "dense only")
    lexical_only = make_chunk("lexical", "lexical only")

    dense = [ScoredChunk(dense_only, 0.9), ScoredChunk(shared, 0.5)]
    lexical = [ScoredChunk(lexical_only, 8.0), ScoredChunk(shared, 4.0)]

    fused = reciprocal_rank_fusion(dense, lexical, k=60, limit=3)
    # Rank 2 in both beats rank 1 in only one.
    assert fused[0].chunk.chunk_id == "shared"
    assert fused[0].source == "hybrid"


def test_rrf_records_both_ranks():
    shared = make_chunk("shared", "x")
    fused = reciprocal_rank_fusion(
        [ScoredChunk(shared, 0.9)], [ScoredChunk(shared, 1.0)], limit=1
    )
    assert fused[0].dense_rank == 0
    assert fused[0].lexical_rank == 0


def test_rrf_handles_one_empty_side():
    only = make_chunk("a", "x")
    fused = reciprocal_rank_fusion([ScoredChunk(only, 0.5)], [], limit=5)
    assert len(fused) == 1
    assert fused[0].source == "dense"


# --- query signals ---------------------------------------------------------
def test_extracts_exception_and_status_code():
    signals = extract_signals(
        "PSQLException while handling request, upstream returned 503"
    )
    assert any("PSQLException" in e for e in signals.exceptions)
    assert "503" in signals.http_codes


def test_ignores_log_level_noise():
    signals = extract_signals("ERROR WARN INFO something happened")
    assert "ERROR" not in signals.error_codes
    assert "INFO" not in signals.error_codes


def test_extracts_upper_snake_error_code():
    signals = extract_signals("connect failed: ECONNREFUSED on retry")
    assert "ECONNREFUSED" in signals.error_codes


def test_build_queries_adds_identifier_only_variant():
    queries = build_queries(
        "checkout failing with org.postgresql.util.PSQLException and status 503"
    )
    assert len(queries) >= 2
    # The second variant is pure signal — no prose.
    assert "PSQLException" in queries[1]
    assert "checkout" not in queries[1]


def test_build_queries_on_plain_prose_returns_single_variant():
    queries = build_queries("the site feels slow today")
    assert queries == ["the site feels slow today"]


def test_build_queries_empty_input():
    assert build_queries("") == []
