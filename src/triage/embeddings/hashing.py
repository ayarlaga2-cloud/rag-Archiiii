"""Dependency-free hashed bag-of-ngrams embedder.

Purpose: let the whole pipeline — ingest, store, hybrid retrieval, agent loop —
run end to end with nothing but numpy installed, so wiring bugs surface before
anyone waits on a 2 GB torch download.

Retrieval quality is poor. It has no semantic generalisation whatsoever; it is
lexical overlap projected into a fixed-width vector. Never ship it.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class HashingEmbedder:
    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"hashing:{self._dimension}"

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self._dimension, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        # Unigrams plus bigrams — bigrams give a little word-order signal.
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for gram in grams:
            digest = hashlib.md5(gram.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        return max(len(text) // 4, len(text.split()))
