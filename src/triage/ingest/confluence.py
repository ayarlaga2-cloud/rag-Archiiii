"""Confluence Cloud REST client.

Uses the v1 content API because CQL is the only practical way to select
"every page in these spaces carrying the `runbook` label" in one query.

Auth is HTTP Basic with (account email, API token) — the standard Atlassian
Cloud scheme. Create a token at:
    https://id.atlassian.com/manage-profile/security/api-tokens
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from triage.config import Settings
from triage.logging_setup import get_logger

log = get_logger(__name__)

# Everything the normalizer needs in one round trip per page.
_EXPAND = ",".join(
    [
        "body.storage",
        "version",
        "space",
        "ancestors",
        "metadata.labels",
        "history.lastUpdated",
    ]
)


class ConfluenceError(RuntimeError):
    """Non-retryable Confluence failure (auth, bad CQL, missing page)."""


class ConfluenceTransientError(RuntimeError):
    """Retryable failure — 429 or 5xx."""


@dataclass(slots=True)
class ConfluencePage:
    page_id: str
    title: str
    space_key: str
    version: int
    storage_html: str
    url: str
    labels: list[str] = field(default_factory=list)
    ancestors: list[str] = field(default_factory=list)
    updated_at: str | None = None


class ConfluenceClient:
    """Thin, paginating, retrying wrapper over the Confluence content API."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.confluence_configured:
            raise ConfluenceError(
                "Confluence is not configured. Set CONFLUENCE_BASE_URL, "
                "CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN in .env"
            )
        self.settings = settings
        self.base_url = settings.confluence_base_url
        token = f"{settings.confluence_email}:{settings.confluence_api_token}"
        encoded = base64.b64encode(token.encode("utf-8")).decode("ascii")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=settings.confluence_timeout_seconds,
            headers={
                "Authorization": f"Basic {encoded}",
                "Accept": "application/json",
                "User-Agent": "triage-agent-rag/0.1",
            },
            follow_redirects=True,
        )

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ConfluenceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- HTTP --------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type(
            (ConfluenceTransientError, httpx.TransportError)
        ),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        response = self._client.get(url, params=params)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "?")
            log.warning("confluence.rate_limited", retry_after=retry_after, url=url)
            raise ConfluenceTransientError(f"429 from Confluence (retry-after={retry_after})")
        if response.status_code >= 500:
            raise ConfluenceTransientError(f"{response.status_code} from Confluence")
        if response.status_code == 401:
            raise ConfluenceError(
                "401 from Confluence — check CONFLUENCE_EMAIL and "
                "CONFLUENCE_API_TOKEN (the token must belong to that account)"
            )
        if response.status_code == 403:
            raise ConfluenceError(
                "403 from Confluence — the account lacks read access to the "
                "requested space(s)"
            )
        if response.status_code >= 400:
            raise ConfluenceError(
                f"{response.status_code} from Confluence: {response.text[:400]}"
            )
        return response.json()

    # -- API ---------------------------------------------------------------
    def verify(self) -> dict[str, Any]:
        """Cheap authenticated call — use to validate credentials up front."""
        return self._get("/rest/api/user/current")

    def count_pages(self, cql: str | None = None) -> int | None:
        """Total pages matching the CQL, for progress reporting.

        Cheap: asks for a single result and reads the total off the envelope.
        Confluence does not always populate `totalSize`, so this returns None
        rather than guessing — progress then reports rate without an ETA.
        """
        query = cql or self.settings.effective_cql()
        payload = self._get(
            "/rest/api/content/search", params={"cql": query, "limit": 1}
        )
        total = payload.get("totalSize")
        if total is None:
            total = payload.get("size")
        return int(total) if isinstance(total, int) else None

    def search_pages(self, cql: str | None = None) -> Iterator[ConfluencePage]:
        """Yield every page matching the CQL query, following pagination."""
        query = cql or self.settings.effective_cql()
        log.info("confluence.search", cql=query)

        params: dict[str, Any] = {
            "cql": query,
            "limit": self.settings.confluence_page_size,
            "expand": _EXPAND,
        }
        path = "/rest/api/content/search"
        seen = 0

        while True:
            payload = self._get(path, params=params)
            results = payload.get("results", [])
            for raw in results:
                page = self._to_page(raw)
                if page is not None:
                    seen += 1
                    yield page

            next_link = (payload.get("_links") or {}).get("next")
            if not next_link or not results:
                break
            # `next` is a path relative to the wiki base and already carries
            # every query parameter, so hand it over verbatim.
            path = f"{self.base_url}{next_link}"
            params = None

        log.info("confluence.search_complete", pages=seen)

    def get_page(self, page_id: str) -> ConfluencePage:
        payload = self._get(f"/rest/api/content/{page_id}", params={"expand": _EXPAND})
        page = self._to_page(payload)
        if page is None:
            raise ConfluenceError(f"Page {page_id} has no storage body")
        return page

    # -- mapping -----------------------------------------------------------
    def _to_page(self, raw: dict[str, Any]) -> ConfluencePage | None:
        page_id = str(raw.get("id", ""))
        body = ((raw.get("body") or {}).get("storage") or {}).get("value")
        if not page_id or not body:
            log.debug("confluence.skip_empty_body", page_id=page_id)
            return None

        webui = ((raw.get("_links") or {}).get("webui")) or ""
        # `_links.base` is present on single-page responses but not always on
        # search responses, so fall back to the configured base URL.
        link_base = (raw.get("_links") or {}).get("base") or self.base_url
        url = f"{link_base}{webui}" if webui else link_base

        labels = [
            lbl.get("name", "")
            for lbl in (
                ((raw.get("metadata") or {}).get("labels") or {}).get("results") or []
            )
            if lbl.get("name")
        ]
        ancestors = [a.get("title", "") for a in (raw.get("ancestors") or []) if a.get("title")]
        last_updated = (
            ((raw.get("history") or {}).get("lastUpdated") or {}).get("when")
            or (raw.get("version") or {}).get("when")
        )

        return ConfluencePage(
            page_id=page_id,
            title=raw.get("title", "") or f"Page {page_id}",
            space_key=((raw.get("space") or {}).get("key")) or "",
            version=int(((raw.get("version") or {}).get("number")) or 0),
            storage_html=body,
            url=url,
            labels=labels,
            ancestors=ancestors,
            updated_at=last_updated,
        )
