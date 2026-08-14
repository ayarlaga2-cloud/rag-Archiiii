"""Confluence storage format -> Markdown.

Confluence stores pages as XHTML with an `ac:`/`ri:` macro namespace, not as
HTML. A generic html2markdown converter drops exactly the parts of a runbook
that matter: fenced commands live inside `<ac:structured-macro ac:name="code">`
and caveats live inside info/warning panels.

So this is a purpose-built walker. It is deterministic, has no version drift
against a third-party converter, and preserves the three things chunking later
depends on:

  1. heading hierarchy   -> becomes the chunk breadcrumb
  2. ordered-list numbering -> a procedure must never be renumbered or split
  3. code fences         -> commands stay verbatim and stay atomic
"""

from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

_INLINE_WS = re.compile(r"[ \t\r\f\v]+")
_EXCESS_BLANKS = re.compile(r"\n{3,}")

# Panels that carry operational meaning; rendered as labelled blockquotes.
_PANEL_MACROS = {
    "info": "INFO",
    "note": "NOTE",
    "tip": "TIP",
    "warning": "WARNING",
    "panel": "PANEL",
    "caution": "CAUTION",
}
# Macros that carry no retrievable content.
_SKIP_MACROS = {
    "toc",
    "children",
    "pagetree",
    "livesearch",
    "recently-updated",
    "contributors",
    "anchor",
    "gallery",
}
_BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "table", "pre", "blockquote", "hr", "section",
}


def _plain_text(node: Tag) -> str:
    """All text under `node`, including CDATA.

    `Tag.get_text()` filters descendants with an exact `type(...) in types`
    check, so `CData` — which Confluence wraps every code body in — is silently
    dropped. `isinstance` catches it.
    """
    parts: list[str] = []
    for descendant in node.descendants:
        if isinstance(descendant, Comment):
            continue
        if isinstance(descendant, NavigableString):
            parts.append(str(descendant))
    return "".join(parts)


def _macro_param(macro: Tag, name: str) -> str:
    param = macro.find("ac:parameter", attrs={"ac:name": name})
    return _plain_text(param).strip() if isinstance(param, Tag) else ""


def _collapse(text: str) -> str:
    return _INLINE_WS.sub(" ", text)


class _Renderer:
    """Recursive descent over the storage-format tree."""

    def __init__(self) -> None:
        self.attachments: list[str] = []

    # -- entry point -------------------------------------------------------
    def render(self, soup: BeautifulSoup) -> str:
        out = self._children(soup, depth=0)
        out = _EXCESS_BLANKS.sub("\n\n", out)
        return out.strip() + "\n"

    def _children(self, node: Tag, depth: int) -> str:
        return "".join(self._node(child, depth) for child in node.children)

    # -- dispatch ----------------------------------------------------------
    def _node(self, node, depth: int) -> str:
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            return _collapse(str(node))
        if not isinstance(node, Tag):
            return ""

        name = node.name.lower()

        if name in {"script", "style"}:
            return ""
        if name.startswith("ac:") or name.startswith("ri:"):
            return self._confluence_tag(node, name, depth)

        handler = getattr(self, f"_tag_{name.replace('-', '_')}", None)
        if handler is not None:
            return handler(node, depth)
        return self._children(node, depth)

    # -- standard HTML -----------------------------------------------------
    def _tag_p(self, node: Tag, depth: int) -> str:
        text = self._inline(node, depth).strip()
        return f"{text}\n\n" if text else ""

    def _tag_br(self, node: Tag, depth: int) -> str:
        return "\n"

    def _tag_hr(self, node: Tag, depth: int) -> str:
        return "\n---\n\n"

    def _heading(self, node: Tag, depth: int, level: int) -> str:
        text = self._inline(node, depth).strip()
        return f"\n{'#' * level} {text}\n\n" if text else ""

    def _tag_h1(self, n: Tag, d: int) -> str: return self._heading(n, d, 1)
    def _tag_h2(self, n: Tag, d: int) -> str: return self._heading(n, d, 2)
    def _tag_h3(self, n: Tag, d: int) -> str: return self._heading(n, d, 3)
    def _tag_h4(self, n: Tag, d: int) -> str: return self._heading(n, d, 4)
    def _tag_h5(self, n: Tag, d: int) -> str: return self._heading(n, d, 5)
    def _tag_h6(self, n: Tag, d: int) -> str: return self._heading(n, d, 6)

    def _tag_strong(self, node: Tag, depth: int) -> str:
        inner = self._inline(node, depth).strip()
        return f"**{inner}**" if inner else ""

    _tag_b = _tag_strong

    def _tag_em(self, node: Tag, depth: int) -> str:
        inner = self._inline(node, depth).strip()
        return f"*{inner}*" if inner else ""

    _tag_i = _tag_em

    def _tag_del(self, node: Tag, depth: int) -> str:
        inner = self._inline(node, depth).strip()
        return f"~~{inner}~~" if inner else ""

    _tag_s = _tag_del
    _tag_strike = _tag_del

    def _tag_code(self, node: Tag, depth: int) -> str:
        # <code> inside <pre> is handled by _tag_pre; this is the inline case.
        inner = _collapse(_plain_text(node)).strip()
        return f"`{inner}`" if inner else ""

    def _tag_a(self, node: Tag, depth: int) -> str:
        text = self._inline(node, depth).strip()
        href = (node.get("href") or "").strip()
        if not text:
            return href
        if not href or href.startswith("#"):
            return text
        return f"[{text}]({href})"

    def _tag_pre(self, node: Tag, depth: int) -> str:
        code = node.find("code")
        language = ""
        if isinstance(code, Tag):
            for cls in code.get("class") or []:
                if cls.startswith("language-"):
                    language = cls[len("language-"):]
                    break
        body = _plain_text(node).strip("\n")
        return self._fence(body, language)

    def _tag_blockquote(self, node: Tag, depth: int) -> str:
        inner = self._children(node, depth).strip()
        if not inner:
            return ""
        quoted = "\n".join(f"> {line}" if line else ">" for line in inner.split("\n"))
        return f"{quoted}\n\n"

    def _tag_ul(self, node: Tag, depth: int) -> str:
        return self._list(node, depth, ordered=False)

    def _tag_ol(self, node: Tag, depth: int) -> str:
        return self._list(node, depth, ordered=True)

    def _tag_table(self, node: Tag, depth: int) -> str:
        rows: list[list[str]] = []
        header: list[str] | None = None

        for tr in node.find_all("tr"):
            cells = tr.find_all(["th", "td"], recursive=False)
            if not cells:
                cells = tr.find_all(["th", "td"])
            rendered = [
                _collapse(self._inline(cell, depth)).strip().replace("|", "\\|")
                for cell in cells
            ]
            if not any(rendered):
                continue
            is_header = all(c.name == "th" for c in cells) and header is None
            if is_header:
                header = rendered
            else:
                rows.append(rendered)

        if header is None and not rows:
            return ""
        width = max([len(header or [])] + [len(r) for r in rows] or [0])
        if width == 0:
            return ""
        if header is None:
            # Markdown tables require a header row; synthesize a blank one so
            # the structure survives.
            header = [""] * width

        def line(cells: list[str]) -> str:
            padded = cells + [""] * (width - len(cells))
            return "| " + " | ".join(padded) + " |"

        out = [line(header), "|" + "|".join([" --- "] * width) + "|"]
        out.extend(line(r) for r in rows)
        return "\n".join(out) + "\n\n"

    # -- Confluence macros -------------------------------------------------
    def _confluence_tag(self, node: Tag, name: str, depth: int) -> str:
        if name == "ac:structured-macro":
            return self._macro(node, depth)
        if name == "ac:link":
            return self._ac_link(node, depth)
        if name == "ac:image":
            return self._ac_image(node)
        if name == "ac:task-list":
            return self._ac_task_list(node, depth)
        if name in {"ac:rich-text-body", "ac:layout", "ac:layout-section", "ac:layout-cell"}:
            return self._children(node, depth)
        if name == "ac:plain-text-body":
            return _plain_text(node)
        if name in {"ac:parameter", "ac:placeholder", "ac:task-status", "ri:attachment", "ri:page", "ri:user"}:
            return ""
        return self._children(node, depth)

    def _macro(self, node: Tag, depth: int) -> str:
        macro_name = (node.get("ac:name") or "").lower()

        if macro_name in _SKIP_MACROS:
            return ""

        if macro_name in {"code", "noformat"}:
            body = node.find("ac:plain-text-body")
            source = _plain_text(body) if isinstance(body, Tag) else _plain_text(node)
            language = _macro_param(node, "language") if macro_name == "code" else ""
            return self._fence(source.strip("\n"), language)

        if macro_name in _PANEL_MACROS:
            label = _PANEL_MACROS[macro_name]
            title = _macro_param(node, "title")
            body_node = node.find("ac:rich-text-body")
            body = (
                self._children(body_node, depth).strip()
                if isinstance(body_node, Tag)
                else _collapse(_plain_text(node)).strip()
            )
            if not body:
                return ""
            head = f"**{label}: {title}**" if title else f"**{label}**"
            lines = [head, ""] + body.split("\n")
            quoted = "\n".join(f"> {ln}" if ln else ">" for ln in lines)
            return f"\n{quoted}\n\n"

        if macro_name == "expand":
            title = _macro_param(node, "title") or "Details"
            body_node = node.find("ac:rich-text-body")
            body = self._children(body_node, depth).strip() if isinstance(body_node, Tag) else ""
            # Flattened into a sub-heading — collapsed content is often exactly
            # the deep-dive an on-call needs, so it must stay retrievable.
            return f"\n**{title}**\n\n{body}\n\n" if body else ""

        if macro_name == "status":
            title = _macro_param(node, "title")
            return f"`{title}`" if title else ""

        if macro_name in {"jira", "jiraissues"}:
            key = _macro_param(node, "key")
            return f"[Jira {key}]" if key else "[Jira issue]"

        # Unknown macro: keep whatever prose it wraps rather than dropping it.
        body_node = node.find("ac:rich-text-body")
        if isinstance(body_node, Tag):
            return self._children(body_node, depth)
        plain = node.find("ac:plain-text-body")
        if isinstance(plain, Tag):
            return _plain_text(plain)
        return ""

    def _ac_link(self, node: Tag, depth: int) -> str:
        body = node.find("ac:link-body") or node.find("ac:plain-text-link-body")
        text = _collapse(_plain_text(body)).strip() if isinstance(body, Tag) else ""
        target = ""
        page = node.find("ri:page")
        if isinstance(page, Tag):
            target = (page.get("ri:content-title") or "").strip()
        attachment = node.find("ri:attachment")
        if isinstance(attachment, Tag):
            target = (attachment.get("ri:filename") or "").strip()
        label = text or target
        if not label:
            return ""
        if target and text and target != text:
            return f"{text} (see: {target})"
        return label

    def _ac_image(self, node: Tag) -> str:
        attachment = node.find("ri:attachment")
        filename = ""
        if isinstance(attachment, Tag):
            filename = (attachment.get("ri:filename") or "").strip()
        url_node = node.find("ri:url")
        if isinstance(url_node, Tag) and not filename:
            filename = (url_node.get("ri:value") or "").strip()
        if filename:
            self.attachments.append(filename)
            return f"[image: {filename}]"
        return ""

    def _ac_task_list(self, node: Tag, depth: int) -> str:
        lines: list[str] = []
        for task in node.find_all("ac:task", recursive=False):
            status_node = task.find("ac:task-status")
            done = _plain_text(status_node).strip().lower() == "complete" if isinstance(status_node, Tag) else False
            body_node = task.find("ac:task-body")
            body = _collapse(self._inline(body_node, depth)).strip() if isinstance(body_node, Tag) else ""
            if body:
                lines.append(f"- [{'x' if done else ' '}] {body}")
        return "\n".join(lines) + "\n\n" if lines else ""

    # -- helpers -----------------------------------------------------------
    def _fence(self, body: str, language: str = "") -> str:
        if not body.strip():
            return ""
        # Avoid terminating the fence early if the snippet itself contains ```.
        ticks = "```"
        while ticks in body:
            ticks += "`"
        return f"\n{ticks}{language}\n{body}\n{ticks}\n\n"

    def _inline(self, node: Tag, depth: int) -> str:
        """Render children, flattening block structure into one line."""
        raw = self._children(node, depth)
        return _collapse(raw.replace("\n", " ")).strip()

    def _list(self, node: Tag, depth: int, ordered: bool) -> str:
        items = [c for c in node.find_all("li", recursive=False)]
        if not items:
            return ""
        indent = "    " * depth
        lines: list[str] = []

        for index, item in enumerate(items, start=1):
            marker = f"{index}. " if ordered else "- "
            # Separate this item's own content from any nested list beneath it.
            nested: list[Tag] = []
            own_parts: list[str] = []
            for child in item.children:
                if isinstance(child, Tag) and child.name.lower() in {"ul", "ol"}:
                    nested.append(child)
                else:
                    own_parts.append(self._node(child, depth + 1))

            own = "".join(own_parts).strip()
            own_lines = [ln for ln in own.split("\n")]
            if not own_lines or not own_lines[0].strip():
                own_lines = [""]

            lines.append(f"{indent}{marker}{own_lines[0].strip()}")
            # Continuation lines (code fences, extra paragraphs) get hanging
            # indent so the item stays one Markdown block.
            hang = indent + " " * len(marker)
            for cont in own_lines[1:]:
                lines.append(f"{hang}{cont}" if cont.strip() else "")

            for sub in nested:
                sub_md = self._list(sub, depth + 1, ordered=sub.name.lower() == "ol")
                lines.extend(ln for ln in sub_md.rstrip("\n").split("\n"))

        return "\n".join(lines) + "\n\n"


def storage_to_markdown(storage_html: str) -> str:
    """Convert a Confluence storage-format body into Markdown."""
    if not storage_html or not storage_html.strip():
        return ""
    # html.parser keeps namespaced tag names (`ac:structured-macro`) intact and
    # exposes CDATA as a NavigableString subclass, which lxml's HTML mode does
    # not do reliably.
    soup = BeautifulSoup(storage_html, "html.parser")
    return _Renderer().render(soup)


def markdown_preview(markdown: str, limit: int = 400) -> str:
    flat = " ".join(markdown.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def iter_headings(markdown: str) -> Iterable[tuple[int, str]]:
    """Yield (level, title) for every ATX heading outside code fences."""
    in_fence = False
    fence_marker = ""
    for line in markdown.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            marker = stripped[: len(stripped) - len(stripped.lstrip("`"))]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker >= fence_marker:
                in_fence, fence_marker = False, ""
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if match:
            yield len(match.group(1)), match.group(2).strip()
