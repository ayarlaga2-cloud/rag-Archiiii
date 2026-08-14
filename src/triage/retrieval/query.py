"""Query preparation.

Two jobs, both aimed at the lexical half of hybrid retrieval:

1. Pull high-signal identifiers out of free text — exception class names, HTTP
   status codes, error codes, k8s object names, file paths. These are the terms
   that make BM25 land on exactly the right runbook, and they get diluted when
   buried in a paragraph of prose.

2. Emit query variants. A raw alert body is a poor dense-retrieval query: it is
   long, full of timestamps and host ids, and its embedding drifts toward
   "generic log noise". A short symptom phrase plus a pure-signal query
   retrieves better than either alone, and RRF is happy to fuse them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Java/Python/Go exception and error class names.
_EXCEPTION_RE = re.compile(r"\b([A-Za-z_][\w.]*(?:Exception|Error|Failure|Fault|Timeout))\b")
# HTTP status codes appearing as standalone numbers or with a label.
_HTTP_RE = re.compile(r"\b(?:status(?:[ _-]?code)?[ =:]*)?([1-5]\d{2})\b")
# UPPER_SNAKE error identifiers: ECONNREFUSED, OOMKilled, ERR_POOL_EXHAUSTED.
_ERRCODE_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)*)\b")
# Dotted / slashed paths and package names.
_QUALIFIED_RE = re.compile(r"\b(\w+(?:\.\w+){2,})\b")
_PATH_RE = re.compile(r"(/(?:[\w.-]+/){1,}[\w.-]+)")
# Kubernetes-ish object names: my-svc-7d9f8b-abcde
_K8S_RE = re.compile(r"\b([a-z0-9]+(?:-[a-z0-9]+){2,})\b")

# Common HTTP codes carry meaning; timestamps and ports do not.
_MEANINGFUL_HTTP = {
    "400", "401", "403", "404", "409", "413", "429",
    "500", "502", "503", "504", "507",
}
_NOISE_TOKENS = frozenset(
    {"ERROR", "WARN", "INFO", "DEBUG", "FATAL", "TRACE", "CRITICAL", "NULL", "TRUE", "FALSE"}
)


@dataclass
class QuerySignals:
    exceptions: list[str] = field(default_factory=list)
    http_codes: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    qualified_names: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    resources: list[str] = field(default_factory=list)

    def all_terms(self) -> list[str]:
        seen: list[str] = []
        for group in (
            self.exceptions,
            self.error_codes,
            self.http_codes,
            self.qualified_names,
            self.resources,
            self.paths,
        ):
            for term in group:
                if term not in seen:
                    seen.append(term)
        return seen

    def is_empty(self) -> bool:
        return not self.all_terms()


def _dedupe(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
        if len(out) >= limit:
            break
    return out


def extract_signals(text: str, limit_per_group: int = 6) -> QuerySignals:
    if not text:
        return QuerySignals()

    exceptions = _EXCEPTION_RE.findall(text)
    http = [c for c in _HTTP_RE.findall(text) if c in _MEANINGFUL_HTTP]
    codes = [c for c in _ERRCODE_RE.findall(text) if c not in _NOISE_TOKENS and len(c) >= 4]
    qualified = [q for q in _QUALIFIED_RE.findall(text) if not q.replace(".", "").isdigit()]
    paths = _PATH_RE.findall(text)
    resources = [r for r in _K8S_RE.findall(text) if len(r) >= 8]

    return QuerySignals(
        exceptions=_dedupe(exceptions, limit_per_group),
        http_codes=_dedupe(http, limit_per_group),
        error_codes=_dedupe(codes, limit_per_group),
        qualified_names=_dedupe(qualified, limit_per_group),
        paths=_dedupe(paths, limit_per_group),
        resources=_dedupe(resources, limit_per_group),
    )


def build_queries(text: str, max_chars: int = 600) -> list[str]:
    """Return the query variants to run. First entry is the primary query."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    queries = [cleaned[:max_chars]]

    signals = extract_signals(cleaned)
    if not signals.is_empty():
        # A pure-identifier query. Dense retrieval does little with this, but
        # BM25 scores it very sharply — which is the entire point.
        queries.append(" ".join(signals.all_terms()))

    # For a long payload, the first couple of sentences usually carry the
    # human-readable symptom; the tail is stack frames and host ids.
    if len(cleaned) > max_chars:
        head = " ".join(cleaned.split(". ")[:2])[:max_chars]
        if head and head not in queries:
            queries.append(head)

    return queries
