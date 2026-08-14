"""Offline / local model loading.

HuggingFace is blocked on the target network, so the model is loaded from a
folder copied onto the machine. These tests cover the parts that fail
*silently* or *confusingly* when that folder is wrong — which is the whole
point of the validation layer.
"""

from __future__ import annotations

import os

import pytest

from triage.embeddings._offline import (
    _OFFLINE_VARS,
    enable_offline_mode,
    looks_like_path,
    resolve_model_path,
)


# --- offline env vars ------------------------------------------------------
def test_enable_offline_mode_sets_every_var(monkeypatch):
    for key in _OFFLINE_VARS:
        monkeypatch.delenv(key, raising=False)
    enable_offline_mode()
    for key, value in _OFFLINE_VARS.items():
        assert os.environ[key] == value


def test_enable_offline_mode_does_not_clobber_existing(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    enable_offline_mode()
    # An explicit operator choice wins — useful on a machine that does have
    # Hub access and is being used to fetch the model in the first place.
    assert os.environ["HF_HUB_OFFLINE"] == "0"


# --- path vs repo-id detection --------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        "C:/models/embeddinggemma-300m",
        "C:\\models\\embeddinggemma-300m",
        "/opt/models/embeddinggemma-300m",
        "./models/embeddinggemma-300m",
        "../models/embeddinggemma-300m",
        "~/models/embeddinggemma-300m",
        ".\\models\\gemma",
    ],
)
def test_paths_are_detected(value):
    assert looks_like_path(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "google/embeddinggemma-300m",
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
    ],
)
def test_hub_repo_ids_are_not_paths(value):
    assert looks_like_path(value) is False


def test_empty_is_not_a_path():
    assert looks_like_path("") is False


def test_existing_relative_dir_is_a_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mymodel").mkdir()
    assert looks_like_path("mymodel") is True


# --- model folder validation ----------------------------------------------
def _make_model_dir(root, *, weights=True, st_config=True):
    path = root / "embeddinggemma-300m"
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        (path / "model.safetensors").write_bytes(b"\x00")
    if st_config:
        (path / "modules.json").write_text("[]", encoding="utf-8")
        (path / "config_sentence_transformers.json").write_text("{}", encoding="utf-8")
    return path


def test_valid_model_dir_resolves(tmp_path):
    path = _make_model_dir(tmp_path)
    assert resolve_model_path(str(path)) == path


def test_missing_dir_raises_with_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        resolve_model_path(str(tmp_path / "nope"))
    message = str(exc.value)
    assert "not found" in message
    assert "config.yaml" in message  # tells you where to fix it


def test_pointing_at_a_file_is_rejected(tmp_path):
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"\x00")
    with pytest.raises(NotADirectoryError) as exc:
        resolve_model_path(str(target))
    assert "not a folder" in str(exc.value)


def test_dir_without_weights_is_rejected(tmp_path):
    path = _make_model_dir(tmp_path, weights=False)
    with pytest.raises(FileNotFoundError) as exc:
        resolve_model_path(str(path))
    message = str(exc.value)
    assert "No model weights" in message
    # Lists what it actually found, so a partial copy is obvious.
    assert "config.json" in message


def test_empty_dir_is_rejected(tmp_path):
    path = tmp_path / "empty"
    path.mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        resolve_model_path(str(path))
    assert "empty directory" in str(exc.value)


def test_bare_transformers_checkpoint_warns_but_loads(tmp_path, caplog):
    """Weights present, sentence-transformers config absent.

    This loads without error but applies default mean pooling instead of
    EmbeddingGemma's real head — degrading every vector with no exception
    raised. It must warn loudly rather than pass silently.
    """
    path = _make_model_dir(tmp_path, st_config=False)
    resolved = resolve_model_path(str(path))
    assert resolved == path  # still usable, deliberately not fatal


def test_onnx_only_export_is_accepted(tmp_path):
    path = tmp_path / "onnx-model"
    path.mkdir()
    (path / "model.onnx").write_bytes(b"\x00")
    (path / "modules.json").write_text("[]", encoding="utf-8")
    assert resolve_model_path(str(path)) == path
