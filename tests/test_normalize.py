"""Confluence storage-format normalization.

The macro cases matter most: a generic HTML converter silently drops
`<ac:structured-macro>` bodies, which is where every command in a Confluence
runbook actually lives.
"""

from __future__ import annotations

from triage.ingest.normalize import iter_headings, storage_to_markdown


def test_headings_and_paragraphs():
    html = "<h1>Title</h1><p>Some prose.</p><h2>Diagnosis</h2><p>More prose.</p>"
    md = storage_to_markdown(html)
    assert "# Title" in md
    assert "## Diagnosis" in md
    assert "Some prose." in md


def test_code_macro_with_cdata_is_preserved():
    # CDATA is the case a naive get_text() drops entirely.
    html = (
        '<ac:structured-macro ac:name="code">'
        '<ac:parameter ac:name="language">bash</ac:parameter>'
        "<ac:plain-text-body><![CDATA[kubectl get pods -n payments]]></ac:plain-text-body>"
        "</ac:structured-macro>"
    )
    md = storage_to_markdown(html)
    assert "```bash" in md
    assert "kubectl get pods -n payments" in md
    assert md.count("```") == 2


def test_noformat_macro_becomes_unlabelled_fence():
    html = (
        '<ac:structured-macro ac:name="noformat">'
        "<ac:plain-text-body><![CDATA[plain text block]]></ac:plain-text-body>"
        "</ac:structured-macro>"
    )
    md = storage_to_markdown(html)
    assert "```" in md
    assert "plain text block" in md


def test_warning_panel_becomes_labelled_blockquote():
    html = (
        '<ac:structured-macro ac:name="warning">'
        "<ac:rich-text-body><p>Do not restart the primary.</p></ac:rich-text-body>"
        "</ac:structured-macro>"
    )
    md = storage_to_markdown(html)
    assert "**WARNING**" in md
    assert "> Do not restart the primary." in md


def test_panel_title_is_kept():
    html = (
        '<ac:structured-macro ac:name="info">'
        '<ac:parameter ac:name="title">Before you begin</ac:parameter>'
        "<ac:rich-text-body><p>Check access.</p></ac:rich-text-body>"
        "</ac:structured-macro>"
    )
    md = storage_to_markdown(html)
    assert "INFO: Before you begin" in md


def test_expand_macro_content_is_not_lost():
    html = (
        '<ac:structured-macro ac:name="expand">'
        '<ac:parameter ac:name="title">Deep dive</ac:parameter>'
        "<ac:rich-text-body><p>Hidden but important.</p></ac:rich-text-body>"
        "</ac:structured-macro>"
    )
    md = storage_to_markdown(html)
    assert "Deep dive" in md
    assert "Hidden but important." in md


def test_ordered_list_numbering_is_preserved():
    html = "<ol><li>First step</li><li>Second step</li><li>Third step</li></ol>"
    md = storage_to_markdown(html)
    assert "1. First step" in md
    assert "2. Second step" in md
    assert "3. Third step" in md


def test_nested_list_is_indented():
    html = "<ul><li>Outer<ul><li>Inner</li></ul></li></ul>"
    md = storage_to_markdown(html)
    assert "- Outer" in md
    assert "    - Inner" in md


def test_table_becomes_markdown_with_header_separator():
    html = (
        "<table><tbody>"
        "<tr><th>Check</th><th>Expected</th></tr>"
        "<tr><td>Pool</td><td>Zero waiting</td></tr>"
        "</tbody></table>"
    )
    md = storage_to_markdown(html)
    assert "| Check | Expected |" in md
    assert "| --- | --- |" in md
    assert "| Pool | Zero waiting |" in md


def test_table_without_header_row_still_renders():
    html = "<table><tbody><tr><td>a</td><td>b</td></tr></tbody></table>"
    md = storage_to_markdown(html)
    assert "| a | b |" in md
    assert "---" in md


def test_task_list_becomes_checkboxes():
    html = (
        "<ac:task-list>"
        "<ac:task><ac:task-status>complete</ac:task-status>"
        "<ac:task-body>Done thing</ac:task-body></ac:task>"
        "<ac:task><ac:task-status>incomplete</ac:task-status>"
        "<ac:task-body>Pending thing</ac:task-body></ac:task>"
        "</ac:task-list>"
    )
    md = storage_to_markdown(html)
    assert "- [x] Done thing" in md
    assert "- [ ] Pending thing" in md


def test_links_and_inline_formatting():
    html = '<p><strong>Bold</strong> and <a href="https://example.test">a link</a> and <code>inline_code</code>.</p>'
    md = storage_to_markdown(html)
    assert "**Bold**" in md
    assert "[a link](https://example.test)" in md
    assert "`inline_code`" in md


def test_toc_macro_is_dropped():
    html = '<ac:structured-macro ac:name="toc"/><p>Real content.</p>'
    md = storage_to_markdown(html)
    assert "Real content." in md
    assert "toc" not in md.lower()


def test_unknown_macro_keeps_its_prose():
    html = (
        '<ac:structured-macro ac:name="some-future-macro">'
        "<ac:rich-text-body><p>Still useful text.</p></ac:rich-text-body>"
        "</ac:structured-macro>"
    )
    assert "Still useful text." in storage_to_markdown(html)


def test_empty_input_is_safe():
    assert storage_to_markdown("") == ""
    assert storage_to_markdown("   ") == ""


def test_iter_headings_skips_fenced_content():
    md = "# One\n\n```\n# not a heading\n```\n\n## Two\n"
    assert list(iter_headings(md)) == [(1, "One"), (2, "Two")]
