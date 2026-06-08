#!/usr/bin/env python3
"""
HTML → Document IR parser.

Reduces a saved DTU HTML page (data/raw/html/{sha256(url)}.html) to the same
normalized blocks the PDF chunker will consume. The DOM does most of the work
the PDF parser has to infer: <h1>..<h6> *are* the heading hierarchy, <table>
*is* a table, <ul>/<ol> *is* a list. So HTML chunking falls out almost for free
once noise is stripped.

Uses lxml only (already a Scrapy dependency — no new install).

Usage (standalone inspection):
    python ingestion/html_parser.py data/raw/html/abc123.html \
        --url "https://dtu.ac.in/Web/notice" --doc-type notice
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Ensure project root (dtu-chatbot/) is importable when run as a script
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import lxml.html
from lxml.html import HtmlElement

from ingestion.document_ir import Block, Document, Heading, ListBlock, Paragraph, Table

# ---------------------------------------------------------------------------
# Tag classification
# ---------------------------------------------------------------------------

# Tags whose entire subtree is noise — dropped before walking.
_NOISE_TAGS = frozenset({
    "script", "style", "noscript", "template", "svg", "iframe", "form",
    "button", "input", "select", "textarea", "nav", "header", "footer",
    "aside", "link", "meta", "img", "picture", "video", "audio", "map",
})

# Substrings in @class / @id that mark boilerplate containers (menus, etc.).
_NOISE_ATTR_RE = re.compile(
    r"(?:^|[-_\s])(?:nav|menu|sidebar|side-bar|breadcrumb|crumb|header|footer|"
    r"banner|topbar|navbar|skip|social|share|cookie|popup|modal|search|"
    r"pagination|pager|widget|advert|ads?)(?:$|[-_\s])",
    re.IGNORECASE,
)

# High-confidence boilerplate tokens that real CMS templates concatenate into
# single class names (e.g. "ddsmoothmenu", "topnavigation"). Matched anywhere,
# no word boundary — these substrings effectively never appear in content.
_NOISE_SUBSTR_RE = re.compile(
    r"navigation|navbar|smoothmenu|mainmenu|submenu|megamenu|dropdownmenu|"
    r"topnav|sidemenu|breadcrumb|pagebottom|bottom_section|copyright|sitemap|"
    r"masterhead",
    re.IGNORECASE,
)

# Inline styles / attributes that hide an element from view. DTU pages carry
# injected SEO spam inside <div style="display:none">; extracting it would
# poison the knowledge base, so hidden subtrees are dropped wholesale.
_HIDDEN_STYLE_RE = re.compile(
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0)",
    re.IGNORECASE,
)

# Inline-level tags: treated as text, never recursed into as block containers.
_INLINE_TAGS = frozenset({
    "a", "b", "i", "u", "em", "strong", "span", "font", "small", "sub", "sup",
    "abbr", "code", "mark", "label", "time", "cite", "q", "s", "del", "ins",
    "big", "tt", "kbd", "samp", "var", "wbr", "br",
})

_HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}

_WS_RE = re.compile(r"\s+")

# Strip the site-name suffix DTU appends to every <title>.
_TITLE_SUFFIX_RE = re.compile(
    r"\s*\|\s*Delhi Technological University\s*$", re.IGNORECASE
)


def _clean(text: str | None) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def _clean_title(text: str | None) -> str:
    """Page title without the boilerplate site-name suffix."""
    return _TITLE_SUFFIX_RE.sub("", _clean(text)).strip(" |»-–—")


def _is_noise(el: HtmlElement) -> bool:
    """True if this element is structural boilerplate to be skipped wholesale."""
    if not isinstance(el.tag, str):
        return True  # comments / processing instructions
    if el.tag in _NOISE_TAGS:
        return True
    # Hidden content (injected SEO spam lives here).
    if el.get("hidden") is not None or el.get("aria-hidden") == "true":
        return True
    style = el.get("style") or ""
    if style and _HIDDEN_STYLE_RE.search(style):
        return True
    attrs = " ".join(filter(None, (el.get("class"), el.get("id"), el.get("role"))))
    if not attrs:
        return False
    return bool(_NOISE_ATTR_RE.search(attrs) or _NOISE_SUBSTR_RE.search(attrs))


def _has_block_child(el: HtmlElement) -> bool:
    """True if any child is a real block-level element (not inline / not noise)."""
    for child in el:
        if not isinstance(child.tag, str):
            continue
        if child.tag in _INLINE_TAGS:
            continue
        if _is_noise(child):
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Tables — distinguish data tables from layout tables
# ---------------------------------------------------------------------------


def _is_data_table(table: HtmlElement) -> bool:
    """
    Old DTU/government pages use <table> for page layout as often as for data.
    A table is real data if it has header cells or is a tidy multi-row,
    multi-column grid. A table that nests another table is layout scaffolding.
    """
    if table.xpath(".//table"):
        return False  # outer layout wrapper
    if table.xpath(".//th"):
        return True
    rows = table.xpath("./tr | ./tbody/tr | ./thead/tr")
    if len(rows) < 2:
        return False
    max_cols = max((len(r.xpath("./td | ./th")) for r in rows), default=0)
    return max_cols >= 2


def _parse_table(table: HtmlElement) -> Table | None:
    """Extract headers + rows from a data table."""
    caption_el = table.find(".//caption")
    caption = _clean(caption_el.text_content()) if caption_el is not None else None

    header_cells = table.xpath("./thead//th | ./thead//td")
    headers = [_clean(c.text_content()) for c in header_cells]

    body_rows = table.xpath("./tbody/tr") or table.xpath("./tr")
    # No <thead>: if the first row is all <th>, promote it to headers.
    if not headers and body_rows:
        first = body_rows[0]
        if first.xpath("./th") and not first.xpath("./td"):
            headers = [_clean(c.text_content()) for c in first.xpath("./th")]
            body_rows = body_rows[1:]

    rows: list[list[str]] = []
    for tr in body_rows:
        cells = [_clean(c.text_content()) for c in tr.xpath("./td | ./th")]
        if any(cells):
            rows.append(cells)

    if not rows:
        return None
    return Table(rows=rows, headers=[h for h in headers if h is not None], caption=caption)


# ---------------------------------------------------------------------------
# DOM walk → blocks
# ---------------------------------------------------------------------------


def _is_nav_list(el: HtmlElement) -> bool:
    """
    A list whose items are (almost) all bare hyperlinks is a navigation menu,
    not content — e.g. the DTU per-department sidebar (About Us, Vision, People
    ...). Such lists carry no answerable text, so they are dropped. A list with
    real prose around its links (rare) is kept.
    """
    lis = el.findall("li")
    if len(lis) < 3:
        return False
    linkish = 0
    for li in lis:
        li_text = _clean(li.text_content())
        if not li_text:
            continue
        anchor_text = _clean("".join(a.text_content() for a in li.findall(".//a")))
        if anchor_text and len(anchor_text) >= 0.8 * len(li_text):
            linkish += 1
    return linkish >= max(3, int(0.8 * len(lis)))


def _parse_list(el: HtmlElement) -> ListBlock | None:
    if _is_nav_list(el):
        return None
    items = [_clean(li.text_content()) for li in el.findall("li")]
    items = [i for i in items if i]
    if not items:
        return None
    return ListBlock(items=items, ordered=(el.tag == "ol"))


def _walk(el: HtmlElement, blocks: list[Block]) -> None:
    """Recursively reduce *el* to blocks, appending in document order."""
    if _is_noise(el):
        return

    tag = el.tag
    if not isinstance(tag, str):
        return

    if tag in _HEADING_TAGS:
        text = _clean(el.text_content())
        if text:
            blocks.append(Heading(text=text, level=_HEADING_TAGS[tag]))
        return

    if tag in ("ul", "ol"):
        lst = _parse_list(el)
        if lst:
            blocks.append(lst)
        return

    if tag == "table":
        if _is_data_table(el):
            tbl = _parse_table(el)
            if tbl:
                blocks.append(tbl)
            return
        # Layout table: descend to reach the real content inside its cells.

    if tag == "p":
        text = _clean(el.text_content())
        if text:
            blocks.append(Paragraph(text))
        return

    # Generic container. If everything inside is inline, it's one text block;
    # otherwise recurse, capturing this element's own leading text and the
    # tail text that follows each child.
    if tag not in ("table",) and not _has_block_child(el):
        text = _clean(el.text_content())
        if text:
            blocks.append(Paragraph(text))
        return

    lead = _clean(el.text)
    if lead:
        blocks.append(Paragraph(lead))
    for child in el:
        _walk(child, blocks)
        tail = _clean(child.tail)
        if tail:
            blocks.append(Paragraph(tail))


def _strip_noise(tree: HtmlElement) -> None:
    """
    Physically remove every noise / hidden subtree from the tree, in place,
    BEFORE walking. This is essential: lxml's text_content() reads hidden
    descendants too, so a container whose only block child is a
    `display:none` spam div would otherwise re-inject that spam. Stripping up
    front makes every later text_content() call safe. drop_tree() preserves
    each removed node's tail text, so legitimate prose between siblings stays.
    """
    for el in list(tree.iter()):
        if el is tree:
            continue
        if el.getparent() is None:
            continue  # already removed via an ancestor
        if not isinstance(el.tag, str) or _is_noise(el):
            el.drop_tree()


def _pick_root(tree: HtmlElement) -> HtmlElement:
    """Prefer <main>/<article>, else the densest <div>, else <body>."""
    for xp in ("//main", "//article", "//*[@id='content']", "//*[@role='main']"):
        found = tree.xpath(xp)
        if found:
            return found[0]
    body = tree.find(".//body")
    return body if body is not None else tree


def _merge_adjacent_paragraphs(blocks: list[Block]) -> list[Block]:
    """
    The walk can emit a run of short Paragraphs from <br>-separated text or
    inline tails. Coalesce consecutive paragraphs so prose stays whole; the
    chunker re-splits on token budget anyway.
    """
    merged: list[Block] = []
    for b in blocks:
        if (
            isinstance(b, Paragraph)
            and merged
            and isinstance(merged[-1], Paragraph)
        ):
            merged[-1] = Paragraph(text=f"{merged[-1].text} {b.text}".strip())
        else:
            merged.append(b)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_html_string(
    html: str,
    url: str,
    doc_type: str = "unknown",
    title: str | None = None,
    date_published: str | None = None,
) -> Document:
    """Parse raw HTML into a Document IR."""
    tree = lxml.html.fromstring(html) if html.strip() else lxml.html.Element("html")

    if title is None:
        title_el = tree.find(".//title")
        title = _clean(title_el.text_content()) if title_el is not None else ""
        if not title:
            h1 = tree.find(".//h1")
            title = _clean(h1.text_content()) if h1 is not None else url
    title = _clean_title(title) or url

    _strip_noise(tree)
    root = _pick_root(tree)
    blocks: list[Block] = []
    _walk(root, blocks)
    blocks = _merge_adjacent_paragraphs(blocks)

    # Title-only sectioning: the DTU CMS renders content titles as styled text,
    # not <h1-6>, so pages carry no semantic headings. Anchor each page under a
    # single synthesized H1 (the cleaned page title) so every chunk inherits a
    # meaningful breadcrumb. Skip if the document already has real headings.
    if title and not any(isinstance(b, Heading) for b in blocks):
        blocks.insert(0, Heading(text=title, level=1))

    return Document(
        url=url,
        title=title,
        source_format="html",
        doc_type=doc_type,
        date_published=date_published,
        blocks=blocks,
    )


def parse_html_file(
    path: Path,
    url: str,
    doc_type: str = "unknown",
    title: str | None = None,
    date_published: str | None = None,
) -> Document:
    """Parse a saved HTML body file into a Document IR."""
    html = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_html_string(
        html, url=url, doc_type=doc_type, title=title, date_published=date_published
    )


# ---------------------------------------------------------------------------
# Standalone inspection
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="HTML → Document IR inspector")
    parser.add_argument("html_path", type=Path, help="Path to a saved .html body")
    parser.add_argument("--url", required=True, help="Original source URL")
    parser.add_argument("--doc-type", dest="doc_type", default="unknown")
    parser.add_argument("--title", default=None)
    parser.add_argument("--date", dest="date_published", default=None)
    args = parser.parse_args()

    if not args.html_path.exists():
        print(f"File not found: {args.html_path}", file=sys.stderr)
        sys.exit(1)

    doc = parse_html_file(
        args.html_path,
        url=args.url,
        doc_type=args.doc_type,
        title=args.title,
        date_published=args.date_published,
    )

    print(f"Title       : {doc.title}")
    print(f"Source URL  : {doc.url}")
    print(f"Doc type    : {doc.doc_type}")
    print(f"Total blocks: {len(doc.blocks)}")
    kinds: dict[str, int] = {}
    for b in doc.blocks:
        kinds[b.kind] = kinds.get(b.kind, 0) + 1
    print(f"By kind     : {kinds}")
    print("\n--- Sections (heading-bounded, with breadcrumb) ---")
    for sec in doc.iter_sections():
        body_blocks = len(sec.blocks)
        print(f"\n[{sec.breadcrumb_str}]  ({body_blocks} block(s))")
        for b in sec.blocks[:3]:
            if isinstance(b, Table):
                preview = " / ".join(b.linearize()[:2])
            elif isinstance(b, ListBlock):
                preview = b.as_text().replace("\n", " | ")
            else:
                preview = b.text
            print(f"   {b.kind}: {preview[:160]}")


if __name__ == "__main__":
    main()
