"""Embedder protocol.

Documents and queries are embedded through separate methods on purpose:
instruction-tuned retrieval models (BGE, E5, Voyage) want an asymmetric
prefix or an input-type flag, and getting that wrong silently costs several
points of recall.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        """Vector width — the store needs it to create the collection."""

    @property
    def name(self) -> str:
        """Identifier recorded alongside the index, so a model swap is visible."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...

    def count_tokens(self, text: str) -> int:
        """Token count in this model's own tokenizer, for chunk sizing."""
