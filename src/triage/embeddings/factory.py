from __future__ import annotations

from triage.config import Settings
from triage.embeddings.base import Embedder


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider

    if provider == "gemma":
        from triage.embeddings.gemma import GemmaEmbedder

        return GemmaEmbedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            truncate_dim=settings.embedding_truncate_dim,
            device=settings.embedding_device,
            hf_token=settings.hf_token,
        )

    if provider == "sentence-transformers":
        from triage.embeddings.local import LocalEmbedder

        return LocalEmbedder(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
        )

    if provider == "voyage":
        from triage.embeddings.voyage import VoyageEmbedder

        return VoyageEmbedder(
            api_key=settings.voyage_api_key,
            model=settings.voyage_embedding_model,
            batch_size=settings.embedding_batch_size,
        )

    if provider == "hashing":
        from triage.embeddings.hashing import HashingEmbedder

        return HashingEmbedder()

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")
