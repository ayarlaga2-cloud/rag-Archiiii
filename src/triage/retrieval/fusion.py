"""Reciprocal Rank Fusion.

Dense cosine similarity and BM25 produce scores on incomparable scales, so
combining them by weighted sum requires per-corpus normalisation that drifts
the moment the corpus changes. RRF ignores magnitudes and fuses on rank alone:

    score(d) = sum over retrievers of  1 / (k + rank(d))

k=60 is the value from the original TREC work and is a good default; raising it
flattens the contribution of top ranks.
"""

from __future__ import annotations

from typing import Sequence

from triage.types import ScoredChunk


def reciprocal_rank_fusion(
    dense: Sequence[ScoredChunk],
    lexical: Sequence[ScoredChunk],
    k: int = 60,
    limit: int = 20,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[ScoredChunk]:
    fused: dict[str, ScoredChunk] = {}
    scores: dict[str, float] = {}

    for rank, item in enumerate(dense):
        cid = item.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank + 1)
        fused[cid] = ScoredChunk(
            chunk=item.chunk, score=0.0, source="dense", dense_rank=rank
        )

    for rank, item in enumerate(lexical):
        cid = item.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + lexical_weight / (k + rank + 1)
        existing = fused.get(cid)
        if existing is None:
            fused[cid] = ScoredChunk(
                chunk=item.chunk, score=0.0, source="lexical", lexical_rank=rank
            )
        else:
            existing.lexical_rank = rank
            # Agreement between two independent retrievers is the strongest
            # signal available before reranking — record it.
            existing.source = "hybrid"

    for cid, score in scores.items():
        fused[cid].score = score

    ordered = sorted(fused.values(), key=lambda s: s.score, reverse=True)
    return ordered[: max(1, limit)]
