"""Cross-encoder reranking.

Bi-encoders (any embedding model, Gemma included) encode query and document
independently, so they can only ever compare summaries of meaning. A
cross-encoder reads the pair jointly and is markedly more accurate — but it
costs a forward pass per candidate, so it runs over the ~20 survivors of
fusion, never over the corpus.

This is the single highest-leverage quality upgrade available once ingest is
working. Default is `none` so nothing extra downloads before you ask for it.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from triage.config import Settings
from triage.logging_setup import get_logger
from triage.types import ScoredChunk

log = get_logger(__name__)


@runtime_checkable
class Reranker(Protocol):
    @property
    def name(self) -> str:
        ...

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        ...


class NoopReranker:
    """Pass-through — keeps fusion order."""

    @property
    def name(self) -> str:
        return "none"

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return list(candidates[:top_k])


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for RERANKER_PROVIDER=local.\n"
                "  pip install -r requirements-embeddings.txt"
            ) from exc
        log.info("reranker.loading", model=model_name)
        self._model = CrossEncoder(model_name, **({"device": device} if device else {}))
        self._model_name = model_name

    @property
    def name(self) -> str:
        return f"cross-encoder:{self._model_name}"

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.embed_text) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        ordered = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return list(ordered[:top_k])


class VoyageReranker:
    def __init__(self, api_key: str, model: str = "rerank-2") -> None:
        import httpx

        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required when RERANKER_PROVIDER=voyage")
        self._model = model
        self._client = httpx.Client(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def name(self) -> str:
        return f"voyage:{self._model}"

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        documents = [c.chunk.embed_text for c in candidates]
        response = self._client.post(
            "https://api.voyageai.com/v1/rerank",
            json={
                "query": query,
                "documents": documents,
                "model": self._model,
                "top_k": min(top_k, len(documents)),
            },
        )
        if response.status_code >= 400:
            log.error(
                "reranker.voyage_failed",
                status=response.status_code,
                body=response.text[:300],
            )
            # Reranking is an enhancement, not a dependency — degrade to
            # fusion order rather than failing the whole retrieval.
            return list(candidates[:top_k])

        out: list[ScoredChunk] = []
        for item in response.json().get("data", []):
            candidate = candidates[item["index"]]
            candidate.rerank_score = float(item.get("relevance_score", 0.0))
            out.append(candidate)
        return out[:top_k]

    def close(self) -> None:
        self._client.close()


def build_reranker(settings: Settings) -> Reranker:
    provider = settings.reranker_provider
    if provider == "local":
        return CrossEncoderReranker(
            settings.reranker_model, device=settings.embedding_device
        )
    if provider == "voyage":
        return VoyageReranker(settings.voyage_api_key, settings.voyage_rerank_model)
    return NoopReranker()
