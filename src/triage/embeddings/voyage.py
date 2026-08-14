"""Voyage AI embedder — the recommended hosted option alongside Claude.

Anthropic does not serve an embeddings endpoint; Voyage is the partner Anthropic
points at. `input_type` is the asymmetric query/document flag and is worth real
recall, so it is always set.
"""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from triage.logging_setup import get_logger

log = get_logger(__name__)

_API_URL = "https://api.voyageai.com/v1/embeddings"

# Published output dimensions, used to size the collection before the first
# call. Anything unlisted is probed with a one-token request.
_KNOWN_DIMENSIONS = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-3-large": 1024,
    "voyage-code-3": 1024,
    "voyage-2": 1024,
    "voyage-large-2": 1536,
}


class VoyageTransientError(RuntimeError):
    pass


class VoyageEmbedder:
    def __init__(self, api_key: str, model: str = "voyage-3", batch_size: int = 64) -> None:
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is required when EMBEDDING_PROVIDER=voyage")
        self._model = model
        self._batch_size = batch_size
        self._client = httpx.Client(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self._dimension = _KNOWN_DIMENSIONS.get(model, 0)

    @property
    def dimension(self) -> int:
        if not self._dimension:
            self._dimension = len(self._call(["dimension probe"], "document")[0])
        return self._dimension

    @property
    def name(self) -> str:
        return f"voyage:{self._model}"

    @retry(
        retry=retry_if_exception_type((VoyageTransientError, httpx.TransportError)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call(self, texts: list[str], input_type: str) -> list[list[float]]:
        response = self._client.post(
            _API_URL,
            json={"input": texts, "model": self._model, "input_type": input_type},
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise VoyageTransientError(f"{response.status_code} from Voyage")
        if response.status_code >= 400:
            raise RuntimeError(f"Voyage error {response.status_code}: {response.text[:300]}")
        payload = response.json()
        ordered = sorted(payload["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in ordered]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            out.extend(self._call(batch, "document"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._call([text], "query")[0]

    def count_tokens(self, text: str) -> int:
        return max(len(text) // 4, len(text.split()))

    def close(self) -> None:
        self._client.close()
