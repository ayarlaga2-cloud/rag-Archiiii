from __future__ import annotations

from triage.config import Settings
from triage.vectorstore.base import VectorStore


def build_vector_store(settings: Settings) -> VectorStore:
    backend = settings.vector_backend

    if backend == "chroma":
        from triage.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore(
            path=settings.chroma_path, collection=settings.vector_collection
        )

    if backend == "pgvector":
        from triage.vectorstore.pgvector_store import PgVectorStore

        return PgVectorStore(
            dsn=settings.pgvector_dsn, collection=settings.vector_collection
        )

    raise ValueError(f"Unknown VECTOR_BACKEND: {backend}")
