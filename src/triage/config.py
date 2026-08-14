"""Configuration.

Single source of truth is `config.yaml` at the repo root — operational knobs
and credentials together, so there is one file to edit. That file is gitignored;
`config.example.yaml` is the committed template.

Model, store and reranker choices are expressed as **named profiles**, so
switching models is a one-line change to an `active:` key rather than a hunt
through flat settings.

Precedence, highest first:

    1. environment variable   (CI, containers, per-command overrides)
    2. config.yaml
    3. built-in default

Environment variables still work for every field, and remain the right choice
for CI or anywhere a file on disk would be the wrong place for a credential.
An optional `.env` is also read if present, but is not required.

The resolved object is a flat `Settings` with the same field names the rest of
the codebase already uses — profiles are a config-authoring convenience, not a
concept the pipeline has to know about.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

EmbeddingProvider = Literal["gemma", "sentence-transformers", "voyage", "hashing"]
VectorBackend = Literal["chroma", "pgvector"]
RerankerProvider = Literal["local", "voyage", "none"]

DEFAULT_CONFIG_FILE = "config.yaml"


class ConfigError(RuntimeError):
    """config.yaml is malformed, or names a profile that does not exist."""


class Settings(BaseModel):
    """Flat, fully-resolved configuration."""

    # --- Confluence --------------------------------------------------------
    confluence_base_url: str = ""
    confluence_email: str = ""
    confluence_api_token: str = ""
    confluence_space_keys: list[str] = Field(default_factory=list)
    confluence_labels: list[str] = Field(default_factory=list)
    confluence_cql: str = ""
    confluence_page_size: int = 50
    confluence_timeout_seconds: float = 45.0

    # --- Embeddings --------------------------------------------------------
    embedding_profile: str = "gemma"
    embedding_provider: EmbeddingProvider = "gemma"
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_batch_size: int = 16
    embedding_truncate_dim: int | None = None
    embedding_device: str = ""
    # Forces HF libraries into local-only mode. Default True: this project is
    # built for environments where huggingface.co is blocked, and a stray Hub
    # call there hangs until the proxy times out rather than failing fast.
    embedding_offline: bool = True
    hf_token: str = ""
    voyage_api_key: str = ""
    voyage_embedding_model: str = "voyage-3"

    # --- Vector store ------------------------------------------------------
    vector_store_profile: str = "local"
    vector_backend: VectorBackend = "chroma"
    vector_collection: str = "runbooks"
    chroma_path: Path = Path("./data/chroma")
    pgvector_dsn: str = ""
    data_dir: Path = Path("./data")

    # --- Chunking ----------------------------------------------------------
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 96
    chunk_overlap_tokens: int = 64

    # --- Ingest (scale) ----------------------------------------------------
    # auto -> SQLite FTS5 when available (scales, incremental)
    # json -> in-memory BM25 sidecar (small corpora only)
    lexical_backend: Literal["auto", "sqlite", "json"] = "auto"
    ingest_checkpoint: bool = True
    ingest_progress_every: int = 25
    ingest_prefetch: int = 8

    # --- Retrieval ---------------------------------------------------------
    retrieval_dense_k: int = 30
    retrieval_lexical_k: int = 30
    retrieval_fused_k: int = 20
    retrieval_final_k: int = 6
    retrieval_rrf_k: int = 60
    retrieval_dense_weight: float = 1.0
    retrieval_lexical_weight: float = 1.0
    retrieval_expand_section: bool = True

    # --- Reranker ----------------------------------------------------------
    reranker_profile: str = "none"
    reranker_provider: RerankerProvider = "none"
    reranker_model: str = "BAAI/bge-reranker-base"
    voyage_rerank_model: str = "rerank-2"

    # --- Service -----------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    log_level: str = "INFO"
    log_json: bool = False

    # --- Provenance --------------------------------------------------------
    config_file: Path | None = None

    @field_validator("confluence_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("chunk_min_tokens")
    @classmethod
    def _min_lt_max(cls, v: int, info) -> int:
        max_tokens = info.data.get("chunk_max_tokens", 512)
        if v >= max_tokens:
            raise ValueError("chunking.min_tokens must be smaller than chunking.max_tokens")
        return v

    @field_validator("embedding_truncate_dim")
    @classmethod
    def _valid_mrl_dim(cls, v: int | None) -> int | None:
        if v is not None and v not in (128, 256, 512, 768):
            raise ValueError(
                "embedding truncate_dim must be one of 128, 256, 512, 768 "
                "(EmbeddingGemma's Matryoshka dimensions)"
            )
        return v

    # --- Derived -----------------------------------------------------------
    @property
    def lexical_index_path(self) -> Path:
        """JSON BM25 sidecar (legacy/small-corpus backend)."""
        return self.data_dir / f"lexical_{self.vector_collection}.json"

    @property
    def lexical_db_path(self) -> Path:
        """SQLite FTS5 index (default backend)."""
        return self.data_dir / f"lexical_{self.vector_collection}.db"

    @property
    def checkpoint_path(self) -> Path:
        """Completed-page log, so an interrupted ingest resumes instantly."""
        return self.data_dir / f"ingest_{self.vector_collection}.checkpoint.jsonl"

    @property
    def confluence_configured(self) -> bool:
        return bool(
            self.confluence_base_url
            and self.confluence_email
            and self.confluence_api_token
        )

    def effective_cql(self) -> str:
        """Build the CQL query that selects runbook pages."""
        if self.confluence_cql:
            return self.confluence_cql
        clauses = ["type = page"]
        if self.confluence_space_keys:
            spaces = ", ".join(f'"{s}"' for s in self.confluence_space_keys)
            clauses.append(f"space in ({spaces})")
        if self.confluence_labels:
            labels = ", ".join(f'"{lbl}"' for lbl in self.confluence_labels)
            clauses.append(f"label in ({labels})")
        return " AND ".join(clauses)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.vector_backend == "chroma":
            self.chroma_path.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict[str, Any]:
        """Resolved view for `triage config`. Secrets are shown as set/missing."""
        return {
            "config_file": str(self.config_file) if self.config_file else "(defaults only)",
            "embedding": {
                "profile": self.embedding_profile,
                "provider": self.embedding_provider,
                "model": self.embedding_model,
                "batch_size": self.embedding_batch_size,
                "truncate_dim": self.embedding_truncate_dim or "full",
                "device": self.embedding_device or "auto",
                "offline": self.embedding_offline,
            },
            "ingest": {
                "lexical_backend": self.lexical_backend,
                "checkpoint": self.ingest_checkpoint,
                "progress_every": self.ingest_progress_every,
                "prefetch": self.ingest_prefetch,
            },
            "vector_store": {
                "profile": self.vector_store_profile,
                "backend": self.vector_backend,
                "collection": self.vector_collection,
                "location": str(self.chroma_path)
                if self.vector_backend == "chroma"
                else ("set" if self.pgvector_dsn else "MISSING (set PGVECTOR_DSN)"),
            },
            "reranker": {
                "profile": self.reranker_profile,
                "provider": self.reranker_provider,
                "model": self.reranker_model if self.reranker_provider != "none" else "-",
            },
            "chunking": {
                "max_tokens": self.chunk_max_tokens,
                "min_tokens": self.chunk_min_tokens,
                "overlap_tokens": self.chunk_overlap_tokens,
            },
            "retrieval": {
                "dense_k": self.retrieval_dense_k,
                "lexical_k": self.retrieval_lexical_k,
                "fused_k": self.retrieval_fused_k,
                "final_k": self.retrieval_final_k,
                "rrf_k": self.retrieval_rrf_k,
                "weights": f"dense={self.retrieval_dense_weight} lexical={self.retrieval_lexical_weight}",
                "expand_section": self.retrieval_expand_section,
            },
            "confluence": {
                "base_url": self.confluence_base_url or "(unset)",
                "email": self.confluence_email or "(unset)",
                "api_token": "set" if self.confluence_api_token else "MISSING",
                "cql": self.effective_cql(),
            },
            "secrets": {
                "CONFLUENCE_API_TOKEN": "set" if self.confluence_api_token else "-",
                "HF_TOKEN": "set" if self.hf_token else "-",
                "VOYAGE_API_KEY": "set" if self.voyage_api_key else "-",
                "PGVECTOR_DSN": "set" if self.pgvector_dsn else "-",
            },
        }


# ---------------------------------------------------------------------------
# .env loading (no extra dependency; pydantic-settings is no longer needed)
# ---------------------------------------------------------------------------
def find_dotenv(explicit: str | Path | None = ".env") -> Path | None:
    """Locate .env: as given, then cwd, then the repo root.

    Without the repo-root fallback, running `triage` from a subdirectory would
    silently load no secrets and fail with a confusing 401.
    """
    if explicit is None:
        return None
    path = Path(explicit)
    if path.is_absolute():
        return path if path.exists() else None
    for candidate in (
        Path.cwd() / path,
        Path(__file__).resolve().parents[2] / path,
    ):
        if candidate.exists():
            return candidate
    return None


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file. Existing vars always win."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # A real environment variable outranks the file, so CI and
        # per-command overrides are never clobbered by it.
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
def _env_str(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else fallback


def _env_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return fallback
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return fallback
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, fallback: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return fallback
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"config.yaml: '{key}' must be a mapping, got {type(value).__name__}")
    return value


def _resolve_profile(
    section: dict[str, Any], section_name: str, env_var: str, default_name: str
) -> tuple[str, dict[str, Any]]:
    """Pick the active profile, letting an env var override the YAML."""
    profiles = _section(section, "profiles")
    name = _env_str(env_var, str(section.get("active") or default_name))

    if not profiles:
        # No profiles block at all — fall back to built-in defaults.
        return name, {}
    if name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none)"
        raise ConfigError(
            f"config.yaml: {section_name}.active = '{name}' but no such profile. "
            f"Available: {available}"
        )
    profile = profiles[name]
    if not isinstance(profile, dict):
        raise ConfigError(f"config.yaml: {section_name}.profiles.{name} must be a mapping")
    return name, profile


def _as_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Tolerate a comma-separated string as well as a YAML list.
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise ConfigError(f"config.yaml: {field} must be a list of strings")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def find_config_file(explicit: str | Path | None = None) -> Path | None:
    """Locate config.yaml: explicit arg, then CONFIG_FILE, then cwd, then repo root."""
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        return path

    from_env = os.environ.get("CONFIG_FILE")
    if from_env:
        path = Path(from_env)
        if not path.exists():
            raise ConfigError(f"CONFIG_FILE points at a missing file: {path}")
        return path

    for candidate in (
        Path.cwd() / DEFAULT_CONFIG_FILE,
        Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_FILE,
    ):
        if candidate.exists():
            return candidate
    return None


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required. pip install pyyaml") from exc

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def load_settings(
    config_path: str | Path | None = None, env_file: str | Path | None = ".env"
) -> Settings:
    """Build Settings from config.yaml + environment."""
    dotenv_path = find_dotenv(env_file)
    if dotenv_path is not None:
        load_dotenv(dotenv_path)

    path = find_config_file(config_path)
    data = load_yaml(path) if path else {}

    confluence = _section(data, "confluence")
    chunking = _section(data, "chunking")
    ingest = _section(data, "ingest")
    retrieval = _section(data, "retrieval")
    service = _section(data, "service")
    logging_cfg = _section(data, "logging")
    paths = _section(data, "paths")

    embedding_section = _section(data, "embedding")
    emb_name, emb = _resolve_profile(
        embedding_section, "embedding", "EMBEDDING_PROFILE", "gemma"
    )
    store_name, store = _resolve_profile(
        _section(data, "vector_store"), "vector_store", "VECTOR_STORE_PROFILE", "local"
    )
    rerank_name, rerank = _resolve_profile(
        _section(data, "reranker"), "reranker", "RERANKER_PROFILE", "none"
    )

    # Providers whose "model" field feeds a differently-named setting.
    embedding_provider = _env_str("EMBEDDING_PROVIDER", str(emb.get("provider", "gemma")))
    embedding_model = _env_str("EMBEDDING_MODEL", str(emb.get("model", "google/embeddinggemma-300m")))
    reranker_provider = _env_str("RERANKER_PROVIDER", str(rerank.get("provider", "none")))
    reranker_model = str(rerank.get("model", "BAAI/bge-reranker-base"))

    truncate_raw = os.environ.get("EMBEDDING_TRUNCATE_DIM", emb.get("truncate_dim"))
    truncate_dim = int(truncate_raw) if truncate_raw not in (None, "", "null") else None

    settings = Settings(
        config_file=path,
        # Confluence
        confluence_base_url=_env_str("CONFLUENCE_BASE_URL", str(confluence.get("base_url", ""))),
        confluence_email=_env_str("CONFLUENCE_EMAIL", str(confluence.get("email", ""))),
        confluence_api_token=_env_str("CONFLUENCE_API_TOKEN", str(confluence.get("api_token", ""))),
        confluence_space_keys=_as_list(
            os.environ.get("CONFLUENCE_SPACE_KEYS", confluence.get("space_keys")),
            "confluence.space_keys",
        ),
        confluence_labels=_as_list(
            os.environ.get("CONFLUENCE_LABELS", confluence.get("labels")),
            "confluence.labels",
        ),
        confluence_cql=_env_str("CONFLUENCE_CQL", str(confluence.get("cql", "") or "")),
        confluence_page_size=_env_int("CONFLUENCE_PAGE_SIZE", int(confluence.get("page_size", 50))),
        confluence_timeout_seconds=_env_float(
            "CONFLUENCE_TIMEOUT_SECONDS", float(confluence.get("timeout_seconds", 45))
        ),
        # Embeddings
        embedding_profile=emb_name,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", int(emb.get("batch_size", 16))),
        embedding_truncate_dim=truncate_dim,
        embedding_device=_env_str("EMBEDDING_DEVICE", str(emb.get("device", "") or "")),
        embedding_offline=_env_bool(
            "EMBEDDING_OFFLINE", bool(embedding_section.get("offline", True))
        ),
        hf_token=_env_str("HF_TOKEN", str(embedding_section.get("hf_token", "") or "")),
        voyage_api_key=_env_str(
            "VOYAGE_API_KEY", str(embedding_section.get("voyage_api_key", "") or "")
        ),
        voyage_embedding_model=(
            embedding_model if embedding_provider == "voyage" else "voyage-3"
        ),
        # Vector store
        vector_store_profile=store_name,
        vector_backend=_env_str("VECTOR_BACKEND", str(store.get("backend", "chroma"))),
        vector_collection=_env_str("VECTOR_COLLECTION", str(store.get("collection", "runbooks"))),
        chroma_path=Path(_env_str("CHROMA_PATH", str(store.get("path", "./data/chroma")))),
        pgvector_dsn=_env_str("PGVECTOR_DSN", str(store.get("dsn", "") or "")),
        data_dir=Path(_env_str("DATA_DIR", str(paths.get("data_dir", "./data")))),
        # Chunking
        chunk_max_tokens=_env_int("CHUNK_MAX_TOKENS", int(chunking.get("max_tokens", 512))),
        chunk_min_tokens=_env_int("CHUNK_MIN_TOKENS", int(chunking.get("min_tokens", 96))),
        chunk_overlap_tokens=_env_int(
            "CHUNK_OVERLAP_TOKENS", int(chunking.get("overlap_tokens", 64))
        ),
        # Ingest / scale
        lexical_backend=_env_str("LEXICAL_BACKEND", str(ingest.get("lexical_backend", "auto"))),
        ingest_checkpoint=_env_bool("INGEST_CHECKPOINT", bool(ingest.get("checkpoint", True))),
        ingest_progress_every=_env_int(
            "INGEST_PROGRESS_EVERY", int(ingest.get("progress_every", 25))
        ),
        ingest_prefetch=_env_int("INGEST_PREFETCH", int(ingest.get("prefetch", 8))),
        # Retrieval
        retrieval_dense_k=_env_int("RETRIEVAL_DENSE_K", int(retrieval.get("dense_k", 30))),
        retrieval_lexical_k=_env_int("RETRIEVAL_LEXICAL_K", int(retrieval.get("lexical_k", 30))),
        retrieval_fused_k=_env_int("RETRIEVAL_FUSED_K", int(retrieval.get("fused_k", 20))),
        retrieval_final_k=_env_int("RETRIEVAL_FINAL_K", int(retrieval.get("final_k", 6))),
        retrieval_rrf_k=_env_int("RETRIEVAL_RRF_K", int(retrieval.get("rrf_k", 60))),
        retrieval_dense_weight=_env_float(
            "RETRIEVAL_DENSE_WEIGHT", float(retrieval.get("dense_weight", 1.0))
        ),
        retrieval_lexical_weight=_env_float(
            "RETRIEVAL_LEXICAL_WEIGHT", float(retrieval.get("lexical_weight", 1.0))
        ),
        retrieval_expand_section=_env_bool(
            "RETRIEVAL_EXPAND_SECTION", bool(retrieval.get("expand_section", True))
        ),
        # Reranker
        reranker_profile=rerank_name,
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
        voyage_rerank_model=(reranker_model if reranker_provider == "voyage" else "rerank-2"),
        # Service
        api_host=_env_str("API_HOST", str(service.get("host", "127.0.0.1"))),
        api_port=_env_int("API_PORT", int(service.get("port", 8080))),
        log_level=_env_str("LOG_LEVEL", str(logging_cfg.get("level", "INFO"))),
        log_json=_env_bool("LOG_JSON", bool(logging_cfg.get("json", False))),
    )
    return settings


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    """Drop the cached Settings — used by tests and by CLI profile overrides."""
    get_settings.cache_clear()
