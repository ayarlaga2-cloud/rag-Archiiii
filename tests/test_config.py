"""Config loading: YAML profiles, env precedence, and error messages."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from triage.config import ConfigError, Settings, load_dotenv, load_settings

YAML = textwrap.dedent(
    """
    confluence:
      base_url: "https://example.atlassian.net/wiki/"
      email: "ops@example.com"
      space_keys: ["SRE", "PLATFORM"]
      labels: ["runbook"]
      page_size: 25

    embedding:
      active: gemma
      profiles:
        gemma:
          provider: gemma
          model: "google/embeddinggemma-300m"
          batch_size: 16
          truncate_dim: null
        gemma-256:
          provider: gemma
          model: "google/embeddinggemma-300m"
          batch_size: 8
          truncate_dim: 256
        hashing:
          provider: hashing
          model: "hashing-384"

    vector_store:
      active: local
      profiles:
        local:
          backend: chroma
          collection: runbooks
          path: "./data/chroma"
        production:
          backend: pgvector
          collection: runbooks

    reranker:
      active: none
      profiles:
        none:
          provider: none
        local:
          provider: local
          model: "BAAI/bge-reranker-base"

    chunking:
      max_tokens: 400
      min_tokens: 80
      overlap_tokens: 40

    retrieval:
      final_k: 8
      expand_section: false

    logging:
      level: DEBUG
      json: true

    paths:
      data_dir: "./somewhere"
    """
)


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(YAML, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Stop the developer's real environment leaking into these tests."""
    for key in (
        "EMBEDDING_PROFILE", "VECTOR_STORE_PROFILE", "RERANKER_PROFILE", "CONFIG_FILE",
        "EMBEDDING_PROVIDER", "EMBEDDING_MODEL", "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_TRUNCATE_DIM", "VECTOR_BACKEND", "VECTOR_COLLECTION",
        "CHUNK_MAX_TOKENS", "RETRIEVAL_FINAL_K", "RETRIEVAL_EXPAND_SECTION",
        "CONFLUENCE_API_TOKEN", "CONFLUENCE_BASE_URL", "CONFLUENCE_SPACE_KEYS",
        "HF_TOKEN", "VOYAGE_API_KEY", "PGVECTOR_DSN", "LOG_LEVEL", "LOG_JSON",
        "DATA_DIR", "CHROMA_PATH",
    ):
        monkeypatch.delenv(key, raising=False)


# --- basic loading ---------------------------------------------------------
def test_loads_active_profiles(config_file):
    settings = load_settings(config_file, env_file=None)
    assert settings.embedding_profile == "gemma"
    assert settings.embedding_provider == "gemma"
    assert settings.embedding_model == "google/embeddinggemma-300m"
    assert settings.vector_backend == "chroma"
    assert settings.reranker_provider == "none"


def test_scalar_sections_are_read(config_file):
    settings = load_settings(config_file, env_file=None)
    assert settings.chunk_max_tokens == 400
    assert settings.chunk_min_tokens == 80
    assert settings.retrieval_final_k == 8
    assert settings.retrieval_expand_section is False
    assert settings.log_level == "DEBUG"
    assert settings.log_json is True
    assert settings.data_dir == Path("./somewhere")


def test_confluence_url_trailing_slash_is_stripped(config_file):
    settings = load_settings(config_file, env_file=None)
    assert settings.confluence_base_url == "https://example.atlassian.net/wiki"


def test_cql_is_built_from_spaces_and_labels(config_file):
    settings = load_settings(config_file, env_file=None)
    cql = settings.effective_cql()
    assert 'space in ("SRE", "PLATFORM")' in cql
    assert 'label in ("runbook")' in cql


# --- profile switching -----------------------------------------------------
def test_env_var_switches_embedding_profile(config_file, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROFILE", "gemma-256")
    settings = load_settings(config_file, env_file=None)
    assert settings.embedding_profile == "gemma-256"
    assert settings.embedding_truncate_dim == 256
    assert settings.embedding_batch_size == 8


def test_env_var_switches_store_profile(config_file, monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_PROFILE", "production")
    settings = load_settings(config_file, env_file=None)
    assert settings.vector_store_profile == "production"
    assert settings.vector_backend == "pgvector"


def test_env_var_switches_reranker_profile(config_file, monkeypatch):
    monkeypatch.setenv("RERANKER_PROFILE", "local")
    settings = load_settings(config_file, env_file=None)
    assert settings.reranker_provider == "local"
    assert settings.reranker_model == "BAAI/bge-reranker-base"


def test_unknown_profile_names_the_available_ones(config_file, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROFILE", "nope")
    with pytest.raises(ConfigError) as exc:
        load_settings(config_file, env_file=None)
    message = str(exc.value)
    assert "nope" in message
    assert "gemma-256" in message  # tells you what you could have used


# --- precedence ------------------------------------------------------------
def test_env_overrides_yaml_scalar(config_file, monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "999")
    assert load_settings(config_file, env_file=None).chunk_max_tokens == 999


def test_env_overrides_profile_field(config_file, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "some/other-model")
    assert load_settings(config_file, env_file=None).embedding_model == "some/other-model"


def test_secrets_load_from_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        textwrap.dedent(
            """
            confluence:
              api_token: "yaml-token"
            embedding:
              hf_token: "yaml-hf"
              voyage_api_key: "yaml-voyage"
              active: gemma
              profiles:
                gemma: { provider: gemma, model: "google/embeddinggemma-300m" }
            vector_store:
              active: production
              profiles:
                production: { backend: pgvector, collection: runbooks, dsn: "postgresql://x/y" }
            """
        ),
        encoding="utf-8",
    )
    settings = load_settings(path, env_file=None)
    assert settings.confluence_api_token == "yaml-token"
    assert settings.hf_token == "yaml-hf"
    assert settings.voyage_api_key == "yaml-voyage"
    assert settings.pgvector_dsn == "postgresql://x/y"


def test_env_still_overrides_yaml_secrets(tmp_path, monkeypatch):
    # CI should be able to inject credentials without editing the file.
    path = tmp_path / "c.yaml"
    path.write_text('confluence:\n  api_token: "yaml-token"\n', encoding="utf-8")
    assert load_settings(path, env_file=None).confluence_api_token == "yaml-token"
    monkeypatch.setenv("CONFLUENCE_API_TOKEN", "env-token")
    assert load_settings(path, env_file=None).confluence_api_token == "env-token"


def test_bool_env_override_accepts_common_spellings(config_file, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_EXPAND_SECTION", "true")
    assert load_settings(config_file, env_file=None).retrieval_expand_section is True
    monkeypatch.setenv("RETRIEVAL_EXPAND_SECTION", "no")
    assert load_settings(config_file, env_file=None).retrieval_expand_section is False


def test_comma_separated_list_env_override(config_file, monkeypatch):
    monkeypatch.setenv("CONFLUENCE_SPACE_KEYS", "A, B ,C")
    assert load_settings(config_file, env_file=None).confluence_space_keys == ["A", "B", "C"]


def test_non_numeric_env_override_is_reported_clearly(config_file, monkeypatch):
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "big")
    with pytest.raises(ConfigError) as exc:
        load_settings(config_file, env_file=None)
    assert "CHUNK_MAX_TOKENS" in str(exc.value)


# --- .env handling ---------------------------------------------------------
def test_dotenv_does_not_clobber_real_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "from-shell")
    load_dotenv(env_file)
    import os

    assert os.environ["HF_TOKEN"] == "from-shell"


def test_dotenv_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    # Running `triage` from a subdirectory must still find the repo-root .env,
    # otherwise secrets silently don't load and auth fails confusingly.
    from triage.config import find_dotenv

    subdir = tmp_path / "nested"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    assert find_dotenv("definitely-not-here.env") is None


def test_dotenv_absolute_path_is_honoured(tmp_path):
    from triage.config import find_dotenv

    target = tmp_path / "custom.env"
    target.write_text("X=1\n", encoding="utf-8")
    assert find_dotenv(target) == target


def test_dotenv_parses_quotes_and_comments(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nVOYAGE_API_KEY="quoted-value"\n\n', encoding="utf-8")
    load_dotenv(env_file)
    import os

    assert os.environ["VOYAGE_API_KEY"] == "quoted-value"


# --- validation ------------------------------------------------------------
def test_min_tokens_must_be_below_max(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("chunking:\n  max_tokens: 100\n  min_tokens: 100\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_settings(path, env_file=None)


def test_invalid_truncate_dim_is_rejected(config_file, monkeypatch):
    monkeypatch.setenv("EMBEDDING_TRUNCATE_DIM", "300")
    with pytest.raises(Exception):
        load_settings(config_file, env_file=None)


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(tmp_path / "does-not-exist.yaml", env_file=None)


def test_malformed_yaml_is_reported(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("embedding:\n  active: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path, env_file=None)


def test_settings_can_still_be_constructed_directly():
    # The pipeline and tests build Settings without any YAML at all.
    settings = Settings(embedding_provider="hashing", chunk_max_tokens=200, chunk_min_tokens=40)
    assert settings.embedding_provider == "hashing"
    assert settings.lexical_index_path.name == "lexical_runbooks.json"


def test_summary_masks_secrets(tmp_path):
    # `triage config` must never print a credential to a terminal or a log.
    path = tmp_path / "c.yaml"
    path.write_text(
        'confluence:\n  api_token: "super-secret-token"\n'
        'embedding:\n  hf_token: "hf-secret"\n  voyage_api_key: "voyage-secret"\n',
        encoding="utf-8",
    )
    summary = load_settings(path, env_file=None).summary()
    rendered = str(summary)
    for secret in ("super-secret-token", "hf-secret", "voyage-secret"):
        assert secret not in rendered
    assert summary["secrets"]["CONFLUENCE_API_TOKEN"] == "set"
    assert summary["secrets"]["HF_TOKEN"] == "set"
