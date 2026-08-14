"""SQLite FTS5 lexical index — the scalable local backend.

Behaviour must match the in-memory BM25 index (same tokenizer, same
higher-is-better score convention) while adding per-page incremental updates.
"""

from __future__ import annotations

import pytest

from triage.retrieval.sqlite_lexical import (
    SQLiteLexicalIndex,
    _fts_query,
    fts5_available,
)
from triage.types import Chunk, SectionKind

pytestmark = pytest.mark.skipif(
    not fts5_available(), reason="SQLite built without FTS5"
)


def make_chunk(chunk_id: str, text: str, page_id: str = "p1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        page_id=page_id,
        page_title="Page",
        page_url="https://example.test",
        space_key="SRE",
        page_version=1,
        content_hash="abc",
        section_path=["Page", "Remediation"],
        section_kind=SectionKind.REMEDIATION,
        section_index=0,
        chunk_index=0,
        body_text=text,
        embed_text=text,
        token_count=len(text.split()),
    )


@pytest.fixture()
def index(tmp_path):
    idx = SQLiteLexicalIndex(tmp_path / "lex.db")
    yield idx
    idx.close()


# --- basics ----------------------------------------------------------------
def test_build_and_search(index):
    count = index.build(
        [
            make_chunk("a", "Generic advice about databases and latency tuning."),
            make_chunk("b", "org.postgresql.util.PSQLException FATAL sorry too many clients"),
            make_chunk("c", "Kafka consumer lag remediation and partition rebalancing."),
        ]
    )
    assert count == 3
    assert index.size == 3

    ranked = index.search("PSQLException too many clients", k=3)
    assert ranked
    assert ranked[0][0] == "b"


def test_scores_are_higher_is_better(index):
    index.build([make_chunk("a", "pgbouncer pool exhausted connections waiting")])
    ranked = index.search("pgbouncer pool exhausted", k=1)
    # FTS5 bm25() is negative-is-better; the wrapper must invert it so the
    # score convention matches every other retriever in the codebase.
    assert ranked[0][1] > 0


def test_results_are_ordered_best_first(index):
    index.build(
        [
            make_chunk("weak", "pgbouncer mentioned once"),
            make_chunk("strong", "pgbouncer pgbouncer pgbouncer pool exhausted pgbouncer"),
        ]
    )
    ranked = index.search("pgbouncer pool", k=2)
    assert [cid for cid, _ in ranked][0] == "strong"


def test_unknown_terms_return_nothing(index):
    index.build([make_chunk("a", "postgres connection pool")])
    assert index.search("zzzz qqqq unrelated", k=5) == []


def test_empty_index_is_safe(index):
    assert index.search("anything", k=5) == []


def test_empty_query_is_safe(index):
    index.build([make_chunk("a", "postgres")])
    assert index.search("", k=5) == []
    assert index.search("   ", k=5) == []


# --- the tokenizer integration --------------------------------------------
def test_camel_case_identifier_is_findable_by_its_parts(index):
    index.build([make_chunk("a", "NullPointerException in the checkout handler")])
    # The project tokenizer splits camelCase before FTS5 ever sees the text.
    assert index.search("pointer exception", k=1)


def test_camel_case_identifier_is_findable_unsplit(index):
    """How an engineer actually types it, straight off a log line."""
    index.build([make_chunk("a", "NullPointerException in the checkout handler")])
    assert index.search("nullpointerexception", k=1)


def test_dotted_class_name_is_findable_by_bare_name(index):
    index.build([make_chunk("a", "org.postgresql.util.PSQLException raised")])
    assert index.search("psqlexception", k=1)


def test_dotted_class_name_is_findable_fully_qualified(index):
    index.build([make_chunk("a", "org.postgresql.util.PSQLException raised")])
    assert index.search("org.postgresql.util.PSQLException", k=1)


# --- incremental updates ---------------------------------------------------
def test_add_chunks_appends(index):
    index.add_chunks([make_chunk("a", "first document about pgbouncer")])
    index.add_chunks([make_chunk("b", "second document about kafka")])
    assert index.size == 2
    assert index.search("kafka", k=1)[0][0] == "b"


def test_delete_page_removes_only_that_page(index):
    index.add_chunks(
        [
            make_chunk("a1", "pgbouncer pool exhausted", page_id="page-1"),
            make_chunk("a2", "restart the pooler", page_id="page-1"),
            make_chunk("b1", "kafka consumer lag", page_id="page-2"),
        ]
    )
    removed = index.delete_page("page-1")
    assert removed == 2
    assert index.size == 1
    assert index.search("pgbouncer", k=5) == []
    assert index.search("kafka", k=5)


def test_reindexing_a_page_does_not_duplicate(index):
    chunk = make_chunk("a1", "pgbouncer pool exhausted", page_id="page-1")
    index.add_chunks([chunk])
    index.delete_page("page-1")
    index.add_chunks([chunk])
    assert index.size == 1


# --- persistence -----------------------------------------------------------
def test_index_survives_reopen(tmp_path):
    path = tmp_path / "lex.db"
    first = SQLiteLexicalIndex(path)
    first.build([make_chunk("a", "pgbouncer pool exhausted")])
    first.close()

    second = SQLiteLexicalIndex.load(path)
    try:
        assert second.size == 1
        assert second.search("pgbouncer", k=1)[0][0] == "a"
    finally:
        second.close()


def test_optimize_is_safe(index):
    index.build([make_chunk("a", "pgbouncer")])
    index.optimize()
    assert index.size == 1


# --- query escaping --------------------------------------------------------
def test_fts_query_quotes_every_term():
    assert _fts_query(["a", "b"]) == '"a" OR "b"'


def test_fts_query_escapes_embedded_quotes():
    assert _fts_query(['say"hi']) == '"say""hi"'


@pytest.mark.parametrize(
    "hostile",
    [
        'NEAR(a b)',
        'foo* AND bar',
        'col:value',
        '^anchor',
        'a OR b NOT c',
        '(unbalanced',
        '"',
        '-minus',
    ],
)
def test_operator_laden_query_does_not_raise(index, hostile):
    """A pasted stack trace must never become an FTS5 syntax error."""
    index.build([make_chunk("a", "postgres connection pool exhausted")])
    assert isinstance(index.search(hostile, k=5), list)


def test_pasted_stack_trace_is_searchable(index):
    index.build([make_chunk("a", "org.postgresql.util.PSQLException too many clients")])
    trace = (
        'org.postgresql.util.PSQLException: FATAL: sorry, too many clients already\n'
        '\tat org.postgresql.core.v3.ConnectionFactoryImpl.doAuthentication(...)\n'
        '\tat com.acme.Checkout$Handler.handle(Checkout.java:142) ~[app.jar:?]'
    )
    assert index.search(trace, k=5)
