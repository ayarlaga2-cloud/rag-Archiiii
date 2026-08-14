"""Local sentence-transformers embedder.

Default for the local footprint. `BAAI/bge-small-en-v1.5` is 384-dim, ~130 MB,
runs fine on CPU, and materially outperforms MiniLM on technical retrieval.

BGE models are trained asymmetrically: queries want the retrieval instruction
prefix, documents want none. Skipping the prefix is a quiet recall regression,
so it is applied here rather than left to the caller.
"""

from __future__ import annotations

import functools

from triage.logging_setup import get_logger

log = get_logger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_E5_QUERY_PREFIX = "query: "
_E5_DOC_PREFIX = "passage: "


class LocalEmbedder:
    """Generic sentence-transformers embedder (BGE, E5, MiniLM, ...).

    EmbeddingGemma has its own class — see `triage.embeddings.gemma` — because
    its task prompts differ from the BGE/E5 conventions handled here.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        device: str = "",
        offline: bool = True,
    ) -> None:
        from triage.embeddings._offline import (
            enable_offline_mode,
            looks_like_path,
            resolve_model_path,
        )

        # Before importing sentence_transformers — the HF libraries read the
        # offline env vars at import time.
        if offline:
            enable_offline_mode()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "  pip install -r requirements-embeddings.txt\n"
                "Or switch to the `hashing` profile for offline smoke tests."
            ) from exc

        source = str(resolve_model_path(model_name)) if looks_like_path(model_name) else model_name
        kwargs: dict = {}
        if device:
            kwargs["device"] = device
        if offline:
            kwargs["local_files_only"] = True

        log.info("embedder.loading", model=source, offline=offline)
        try:
            self._model = SentenceTransformer(source, **kwargs)
        except TypeError:
            kwargs.pop("local_files_only", None)
            self._model = SentenceTransformer(source, **kwargs)
        self._model_name = model_name
        self._batch_size = batch_size
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        lowered = model_name.lower()
        self._is_bge = "bge" in lowered
        self._is_e5 = "e5" in lowered
        log.info("embedder.ready", model=model_name, dimension=self._dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [f"{_E5_DOC_PREFIX}{t}" for t in texts] if self._is_e5 else texts
        vectors = self._model.encode(
            prepared,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        if self._is_bge:
            text = f"{_BGE_QUERY_PREFIX}{text}"
        elif self._is_e5:
            text = f"{_E5_QUERY_PREFIX}{text}"
        vector = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )[0]
        return vector.tolist()

    @functools.lru_cache(maxsize=4096)
    def count_tokens(self, text: str) -> int:
        """Real token count from the model's own tokenizer."""
        try:
            return len(self._model.tokenizer.tokenize(text))
        except Exception:  # pragma: no cover - tokenizer variations
            return max(len(text) // 4, len(text.split()))
