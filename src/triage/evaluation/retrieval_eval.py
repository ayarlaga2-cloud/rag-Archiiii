"""Retrieval evaluation.

"Does the retrieval look good?" is not a question you can answer by eyeballing
a handful of queries — the failures that matter are the ones you did not think
to try. A 20-30 case golden set turns every later change (chunk size, reranker
on/off, Matryoshka truncation, a different model) into a measurement instead of
an argument.

Metrics:
  recall@k  — was any relevant chunk retrieved at all? The one to watch: if a
              runbook is not in the top k, nothing downstream can recover it.
  MRR       — how high was the first relevant hit? Sensitive to ranking.
  nDCG@k    — full-ranking quality, discounted by position.

Golden set format (JSONL, one case per line):

    {"query": "postgres connection pool exhausted in checkout",
     "relevant_pages": ["Postgres Connection Exhaustion"],
     "notes": "paraphrase, no exact identifiers"}

Match on `relevant_pages` (page titles or ids) rather than chunk ids: chunk ids
change whenever chunking changes, which would make the golden set useless for
the exact comparison you most want to run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from triage.logging_setup import get_logger
from triage.retrieval.retriever import HybridRetriever

log = get_logger(__name__)


@dataclass
class EvalCase:
    query: str
    relevant_pages: list[str] = field(default_factory=list)
    relevant_sections: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalCase":
        return cls(
            query=raw["query"],
            relevant_pages=[str(p) for p in raw.get("relevant_pages", [])],
            relevant_sections=[str(s) for s in raw.get("relevant_sections", [])],
            notes=raw.get("notes", ""),
        )


@dataclass
class CaseResult:
    query: str
    hit: bool
    first_relevant_rank: int | None
    reciprocal_rank: float
    ndcg: float
    retrieved: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class EvalReport:
    k: int
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def recall_at_k(self) -> float:
        return (sum(1 for c in self.cases if c.hit) / len(self.cases)) if self.cases else 0.0

    @property
    def mrr(self) -> float:
        return (sum(c.reciprocal_rank for c in self.cases) / len(self.cases)) if self.cases else 0.0

    @property
    def ndcg(self) -> float:
        return (sum(c.ndcg for c in self.cases) / len(self.cases)) if self.cases else 0.0

    @property
    def misses(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.hit]

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "cases": len(self.cases),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg, 4),
            "misses": [c.query for c in self.misses],
        }


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(EvalCase.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_no} is not a valid eval case: {exc}") from exc
    return cases


def _is_relevant(chunk, case: EvalCase) -> bool:
    if case.relevant_pages:
        haystack = {chunk.page_title.lower(), chunk.page_id.lower()}
        if any(p.lower() in haystack for p in case.relevant_pages):
            if not case.relevant_sections:
                return True
            breadcrumb = chunk.breadcrumb.lower()
            return any(s.lower() in breadcrumb for s in case.relevant_sections)
    return False


def evaluate(
    retriever: HybridRetriever, cases: list[EvalCase], k: int = 6
) -> EvalReport:
    report = EvalReport(k=k)

    for case in cases:
        result = retriever.search(case.query, top_k=k, expand_sections=False)
        retrieved = [s.chunk for s in result.chunks]
        relevance = [1 if _is_relevant(c, case) else 0 for c in retrieved]

        first = next((i for i, rel in enumerate(relevance) if rel), None)
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))
        # Ideal DCG for the number of relevant items actually retrievable in k.
        ideal_count = min(sum(relevance), k) or 1
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal_count))

        report.cases.append(
            CaseResult(
                query=case.query,
                hit=first is not None,
                first_relevant_rank=(first + 1) if first is not None else None,
                reciprocal_rank=(1.0 / (first + 1)) if first is not None else 0.0,
                ndcg=(dcg / idcg) if idcg else 0.0,
                retrieved=[f"{c.page_title} > {c.breadcrumb}" for c in retrieved],
                notes=case.notes,
            )
        )

    log.info("eval.complete", **report.as_dict())
    return report
