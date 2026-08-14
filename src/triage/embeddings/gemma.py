"""EmbeddingGemma — loaded from a local directory, fully offline.

Why it fits this corpus:

  * 308M params, runs on CPU; 768-dim output, so the index stays small.
  * 2048-token input window — larger than any chunk we produce, so no chunk is
    ever silently truncated mid-procedure.
  * Matryoshka-trained: the same vector truncates to 512/256/128 dims without
    retraining, which is a cheap index-size lever later.

**This build never contacts huggingface.co.** `embedding.model` is expected to
be a filesystem path to a model folder copied onto the machine; offline mode is
forced before the HF libraries are imported, so a blocked proxy cannot stall
the load. A Hub repo id still works if the network happens to allow it, but
nothing here depends on that.

The critical behavioural detail is that EmbeddingGemma is **asymmetric and
prompt-driven**. It was trained with task prefixes, and omitting them is a
quiet, significant recall regression — the model behaves as if answering a
different task:

    query    ->  "task: search result | query: {text}"
    document ->  "title: {title} | text: {text}"

sentence-transformers >= 5.0 exposes `encode_query()` / `encode_document()`,
which apply those prefixes from the model's own config. This class uses them
when present and falls back to the literal prefixes otherwise, so the prompts
are applied either way.
"""

from __future__ import annotations

from pathlib import Path

from triage.embeddings._offline import (
    enable_offline_mode,
    looks_like_path,
    resolve_model_path,
)
from triage.logging_setup import get_logger

log = get_logger(__name__)

# Literal prompts from the model card, used when encode_query/encode_document
# are unavailable on the installed sentence-transformers.
QUERY_PROMPT = "task: search result | query: "
DOCUMENT_PROMPT = "title: none | text: "

_LOAD_HELP = (
    "Could not load the EmbeddingGemma model.\n"
    "\n"
    "This project runs fully offline — it expects the model as a FOLDER on this\n"
    "machine, not a download. Check, in order:\n"
    "  1. `embedding.profiles.<active>.model` in config.yaml points at the\n"
    "     folder containing modules.json and model.safetensors\n"
    "  2. the whole folder was copied across, not just the weights\n"
    "  3. pip install -r requirements-embeddings.txt\n"
    "     (sentence-transformers>=5.0, transformers>=4.56)\n"
    "\n"
    "To verify the rest of the pipeline without any model at all, switch to\n"
    "the `hashing` profile:  triage --embedding-profile hashing ingest ...\n"
)


class GemmaEmbedder:
    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        batch_size: int = 16,
        truncate_dim: int | None = None,
        device: str = "",
        hf_token: str = "",
        offline: bool = True,
    ) -> None:
        # Must happen before sentence_transformers pulls in transformers /
        # huggingface_hub, which read these vars at import time.
        if offline:
            enable_offline_mode()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "  pip install -r requirements-embeddings.txt\n"
                "Or set the `hashing` profile to smoke-test without it."
            ) from exc

        is_local = looks_like_path(model_name)
        if is_local:
            source: str | Path = resolve_model_path(model_name)
        else:
            source = model_name
            if offline:
                log.warning(
                    "embedder.repo_id_while_offline",
                    model=model_name,
                    detail=(
                        "This looks like a Hub repo id, not a local folder. It will "
                        "only load if the model is already in the local HF cache. "
                        "Point `model` at the copied model folder instead."
                    ),
                )

        kwargs: dict = {}
        if truncate_dim:
            kwargs["truncate_dim"] = truncate_dim
        if device:
            kwargs["device"] = device
        if hf_token and not offline:
            kwargs["token"] = hf_token
        if offline:
            # Belt and braces: the env vars cover most paths, this covers the
            # per-file resolution calls they miss.
            kwargs["local_files_only"] = True

        log.info(
            "embedder.loading",
            model=str(source),
            local=is_local,
            offline=offline,
            truncate_dim=truncate_dim,
        )
        try:
            self._model = SentenceTransformer(str(source), **kwargs)
        except TypeError:
            # Older sentence-transformers does not accept local_files_only.
            kwargs.pop("local_files_only", None)
            self._model = SentenceTransformer(str(source), **kwargs)
        except Exception as exc:
            raise RuntimeError(f"{_LOAD_HELP}\nUnderlying error: {exc}") from exc

        self._model_name = str(source)
        self._display_name = Path(model_name).name if is_local else model_name
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
            model=self._display_name,
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
        return f"gemma:{self._display_name}{suffix}"

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
