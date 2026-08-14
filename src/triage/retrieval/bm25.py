"""BM25 lexical index with SRE-aware tokenization.

Dense retrieval alone is a bad fit for incident response. An embedding model
happily maps `PSQLException: FATAL: sorry, too many clients already` and
`database connection issues` close together, which is useful — but it does not
reliably surface the one runbook that literally contains
`org.postgresql.util.PSQLException`. Exact identifier match is the single
highest-precision signal an on-call has, and it is exactly what BM25 is good at.

The tokenizer is the part that matters. Generic word tokenizers destroy the
identifiers this corpus is made of, so this one:

  * splits camelCase and ACRONYMWord   -> NullPointerException, PSQLException
  * keeps dotted/underscored forms whole AND emits their parts, so
    `org.postgresql.util.PSQLException` matches a query saying `psqlexception`
  * keeps short numeric tokens like `503`, `429`, `oom`
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from triage.types import Chunk

_WORD_RE = re.compile(r"[A-Za-z0-9_./:\-]+")
_CAMEL_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT_RE = re.compile(r"[./:_\-]+")

# Deliberately short. Aggressive stopword removal strips terms that carry real
# meaning in ops text ("down", "up", "no", "not", "over").
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at for from by with as is are
    was were be been being this that these those it its we you they there here
    do does did have has had will would should can could may might must
    """.split()
)

# Numeric/short tokens worth keeping despite the length filter.
_KEEP_SHORT = frozenset({"oom", "cpu", "gc", "io", "ssl", "tls", "dns", "5xx", "4xx"})


def _keep(token: str) -> bool:
    """Single length rule for every emission route.

    Two chars is the floor: `503`, `gc`, `io` carry meaning in ops text, a lone
    `5` or `a` does not. `_KEEP_SHORT` is the escape hatch for anything below.
    """
    if not token or token in _STOPWORDS:
        return False
    return len(token) > 1 or token in _KEEP_SHORT


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for raw in _WORD_RE.findall(text):
        # Per-raw dedupe: the same surface form reached by two routes below
        # would otherwise inflate its term frequency and skew BM25.
        emitted: set[str] = set()

        def add(token: str) -> None:
            if _keep(token) and token not in emitted:
                emitted.add(token)
                tokens.append(token)

        low_raw = raw.lower().strip("./:-_")
        if not low_raw:
            continue

        # 1. The whole surface form: `org.postgresql.util.psqlexception`.
        add(low_raw)

        # 2. Separator-split, WITHOUT camel splitting. This is what makes
        #    `psqlexception` findable — a user reading `PSQLException` out of a
        #    log types it lowercased and unsplit, and would otherwise match
        #    nothing, because step 3 only ever emits `psql` + `exception`.
        for piece in _SPLIT_RE.split(low_raw):
            add(piece)

        # 3. camelCase / ACRONYMWord split, plus the separator-split of each
        #    part: `NullPointerException` -> null, pointer, exception.
        expanded = _CAMEL_2.sub(" ", _CAMEL_1.sub(" ", raw))
        for part in expanded.split():
            low = part.lower().strip("./:-_")
            if not low:
                continue
            add(low)
            for piece in _SPLIT_RE.split(low):
                add(piece)

    return tokens


class BM25Index:
    """Okapi BM25. Rebuilt from the vector store at the end of every ingest."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avg_length: float = 0.0
        # term -> list of (doc_index, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}

    # -- build -------------------------------------------------------------
    def build(self, chunks: Iterable[Chunk]) -> int:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_ids = []
        self.doc_lengths = []

        for chunk in chunks:
            # Index embed_text, not body_text: the breadcrumb ("Postgres
            # connection exhaustion > Remediation") carries terms the body of
            # a step list often omits.
            terms = tokenize(chunk.embed_text)
            index = len(self.doc_ids)
            self.doc_ids.append(chunk.chunk_id)
            self.doc_lengths.append(len(terms))

            frequencies: dict[str, int] = defaultdict(int)
            for term in terms:
                frequencies[term] += 1
            for term, freq in frequencies.items():
                postings[term].append((index, freq))

        self.postings = dict(postings)
        total = sum(self.doc_lengths)
        self.avg_length = (total / len(self.doc_lengths)) if self.doc_lengths else 0.0
        self._compute_idf()
        return len(self.doc_ids)

    def _compute_idf(self) -> None:
        n = len(self.doc_ids)
        self._idf = {}
        for term, posting in self.postings.items():
            df = len(posting)
            # Robertson/Sparck-Jones with the +0.5 smoothing, floored at a small
            # positive value so terms present in most documents cannot score
            # negatively and invert the ranking.
            self._idf[term] = max(1e-9, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    # -- query -------------------------------------------------------------
    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        if not self.doc_ids:
            return []
        terms = tokenize(query)
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_index, freq in posting:
                length = self.doc_lengths[doc_index] or 1
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_length or 1.0)
                )
                scores[doc_index] += idf * (freq * (self.k1 + 1)) / denominator

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, k)]
        return [(self.doc_ids[i], score) for i, score in ranked]

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "avg_length": self.avg_length,
            "postings": {t: [list(p) for p in posting] for t, posting in self.postings.items()},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)  # atomic, so a crash mid-write cannot corrupt the index

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        index.doc_ids = payload["doc_ids"]
        index.doc_lengths = payload["doc_lengths"]
        index.avg_length = payload["avg_length"]
        index.postings = {
            term: [(int(d), int(f)) for d, f in posting]
            for term, posting in payload["postings"].items()
        }
        index._compute_idf()
        return index

    @property
    def size(self) -> int:
        return len(self.doc_ids)
