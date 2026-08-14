"""EmbeddingGemma (`google/embeddinggemma-300m`) — the default embedder.

Why it fits this corpus:

  * 308M params, runs on CPU; ~768-dim output, so the index stays small.
  * 2048-token input window — comfortably larger than any chunk we produce, so
    no chunk is ever silently truncated mid-procedure.
  * Matryoshka-trained: the same vector can be truncated to 512/256/128 dims
    without retraining, which is a cheap index-size lever later.

The critical detail is that EmbeddingGemma is **asymmetric and prompt-driven**.
It was trained with task prefixes, and omitting them is a quiet, significant
recall regression — the model behaves as if it were answering a different task:

    query    ->  "task: search result | query: {text}"
    document ->  "title: {title} | text: {text}"

sentence-transformers >= 5.0 exposes `encode_query()` / `encode_document()`,
which apply those prefixes from the model's own config. This class uses them
when present and falls back to the literal prefixes otherwise, so the prompts
are applied either way.

EmbeddingGemma is a GATED model. Accept the licence at
https://huggingface.co/google/embeddinggemma-300m, then `huggingface-cli login`
or set HF_TOKEN.
"""

from __future__ import annotations

from triage.logging_setup import get_logger

log = get_logger(__name__)

# Literal prompts from the model card, used when encode_query/encode_document
# are unavailable on the installed sentence-transformers.
QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "

_GATED_HELP = (
    "Could not load google/embeddinggemma-300m.\n"
    "It is a gated model — three things are required:\n"
    "  1. Accept the licence at https://huggingface.co/google/embeddinggemma-300m\n"
    "  2. Authenticate: `huggingface-cli login`, or set HF_TOKEN in .env\n"
    "  3. pip install -r requirements-embeddings.txt "
    "(sentence-transformers>=5.0, transformers>=4.56)\n"
    "To verify the rest of the pipeline first, set EMBEDDING_PROVIDER=hashing."
)


class GemmaEmbedder:
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        batch_size: int = 16,
        truncate_dim: int | None = None,
        device: str = "",
        hf_token: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "  pip install -r requirements-embeddings.txt\n"
                "Or set EMBEDDING_PROVIDER=hashing to smoke-test without it."
            ) from exc

        kwargs: dict = {}
        if truncate_dim:
            kwargs["truncate_dim"] = truncate_dim
        if device:
            kwargs["device"] = device
        if hf_token:
            kwargs["token"] = hf_token

        log.info("embedder.loading", model=model_name, truncate_dim=truncate_dim)
        try:
            self._model = SentenceTransformer(model_name, **kwargs)
        except Exception as exc:  # gated repo, missing auth, old transformers
            raise RuntimeError(f"{_GATED_HELP}\n\nUnderlying error: {exc}") from exc

        self._model_name = model_name
        self._batch_size = batch_size
        self._truncate_dim = truncate_dim
        self._dimension = truncate_dim or int(
            self._model.get_sentence_embedding_dimension()
        )
        # Present from sentence-transformers 5.0; they read the prompt prefixes
        # out of the model's own config rather than hardcoding them.
        self._has_task_encoders = hasattr(self._model, "encode_query") and hasattr(
            self._model, "encode_document"
        )
        if not self._has_task_encoders:
            log.warning(
                "embedder.legacy_sentence_transformers",
                detail="encode_query/encode_document unavailable; "
                "falling back to literal task prefixes. Upgrade to >=5.0.",
            )
        log.info(
            "embedder.ready",
            model=model_name,
            dimension=self._dimension,
            max_seq_length=getattr(self._model, "max_seq_length", None),
        )

    # -- protocol ----------------------------------------------------------
    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        suffix = f"@{self._truncate_dim}" if self._truncate_dim else ""
        return f"gemma:{self._model_name}{suffix}"

    @property
    def max_input_tokens(self) -> int:
        return int(getattr(self._model, "max_seq_length", 2048) or 2048)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # The chunk's embed_text already opens with "Runbook: <title>", so the
        # model's `title:` slot would only duplicate it — `title: none` is
        # correct here, not a shortcut.
        if self._has_task_encoders:
            vectors = self._model.encode_document(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            vectors = self._model.encode(
                [f"{DOCUMENT_PROMPT}{t}" for t in texts],
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        if self._has_task_encoders:
            vector = self._model.encode_query(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
        else:
            vector = self._model.encode(
                [f"{QUERY_PROMPT}{text}"],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
        return vector.tolist()

    def count_tokens(self, text: str) -> int:
        """Real Gemma token count — this is what sizes the chunks."""
        try:
            return len(self._model.tokenizer.tokenize(text))
        except Exception:  # pragma: no cover - tokenizer variations
            return max(len(text) // 4, len(text.split()))
