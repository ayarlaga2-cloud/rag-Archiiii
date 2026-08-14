"""Offline model loading.

In a locked-down corporate environment huggingface.co is blocked, so any
library call that tries to reach the Hub must be prevented from even
attempting it. Two things are needed, in this order:

1. Set the offline env vars **before** `transformers` / `huggingface_hub` are
   imported. They are read at import time; setting them afterwards is a no-op
   and you get a confusing connection timeout instead of a clean local load.
2. Pass `local_files_only=True` on the model constructor, which stops the
   per-file resolution calls that the env vars alone do not always cover.

Without both, loading a model from a local directory still emits Hub requests
to check for updates — which hang until the proxy times them out.
"""

from __future__ import annotations

import os
from pathlib import Path

from triage.logging_setup import get_logger

log = get_logger(__name__)

_OFFLINE_VARS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    # Telemetry pings the Hub too, and fails the same way.
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}

# A sentence-transformers model directory carries these. A bare transformers
# checkpoint does not — and loading one silently applies default mean pooling,
# which is the wrong head for EmbeddingGemma and quietly degrades every vector.
_ST_MARKERS = ("modules.json", "config_sentence_transformers.json")
_WEIGHT_MARKERS = ("model.safetensors", "pytorch_model.bin", "model.onnx")


def enable_offline_mode() -> None:
    """Force every HF library into local-only mode. Call before importing them."""
    already = [k for k in _OFFLINE_VARS if os.environ.get(k)]
    for key, value in _OFFLINE_VARS.items():
        os.environ.setdefault(key, value)
    if not already:
        log.debug("embedder.offline_mode_enabled", vars=sorted(_OFFLINE_VARS))


def looks_like_path(model: str) -> bool:
    """True when `model` is a filesystem path rather than a Hub repo id.

    Hub ids look like `google/embeddinggemma-300m`: one slash, no drive letter,
    no separators beyond that. Anything that exists on disk is a path.
    """
    if not model:
        return False
    candidate = Path(model).expanduser()
    if candidate.exists():
        return True
    # Windows drive letter, absolute posix path, or an explicit relative path.
    return (
        len(model) > 1 and model[1] == ":"
        or model.startswith(("/", "./", "../", "~", ".\\", "..\\"))
        or "\\" in model
    )


def resolve_model_path(model: str) -> Path:
    """Validate a local model directory, with actionable errors."""
    path = Path(model).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Embedding model directory not found: {path}\n"
            "Set `embedding.profiles.<name>.model` in config.yaml to the folder "
            "holding the downloaded model — the one containing modules.json "
            "and model.safetensors.\n"
            "If you copied it from another machine, check the whole folder came "
            "across, not just the weights file."
        )
    if path.is_file():
        raise NotADirectoryError(
            f"Embedding model path points at a file, not a folder: {path}\n"
            "Point it at the directory that contains the model files."
        )

    names = {p.name for p in path.iterdir()}

    if not any(marker in names for marker in _WEIGHT_MARKERS):
        raise FileNotFoundError(
            f"No model weights found in {path}\n"
            f"Expected one of: {', '.join(_WEIGHT_MARKERS)}\n"
            f"Found: {', '.join(sorted(names)) or '(empty directory)'}"
        )

    if not any(marker in names for marker in _ST_MARKERS):
        # Loadable, but the pooling head would be guessed rather than read —
        # wrong for EmbeddingGemma, and wrong silently.
        log.warning(
            "embedder.not_a_sentence_transformers_dir",
            path=str(path),
            detail=(
                "modules.json / config_sentence_transformers.json are missing. "
                "sentence-transformers will fall back to default mean pooling, "
                "which is NOT EmbeddingGemma's head and will degrade retrieval "
                "quality without raising an error. Re-copy the full model "
                "folder from the machine that downloaded it."
            ),
        )

    return path
