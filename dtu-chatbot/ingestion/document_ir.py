"""
Normalized Document IR — the format-agnostic representation that every parser
(PDF, HTML, ...) emits and the chunker consumes.

The whole point: the chunker should never know or care whether content came from
a PDF or an HTML page. Both parsers reduce their source to the same handful of
block types, and all downstream logic (sectioning, chunking, breadcrumb
enrichment) operates on these blocks alone.

Block types:
    Heading    — a section title with a level (h1..h6 / PDF font hierarchy)
    Paragraph  — a run of prose
    ListBlock  — an ordered or unordered list
    Table      — structured rows with optional headers

No external dependencies. Token counting deliberately lives in the chunker, not
here, so this module stays a pure structural description.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Union


def _meaningful(cell: str) -> bool:
    """True if a table cell carries real content (not empty / bullet / dash)."""
    c = cell.strip()
    return bool(c) and any(ch.isalnum() for ch in c)

# ---------------------------------------------------------------------------
# Block types
# ---------------------------------------------------------------------------


@dataclass
class Heading:
    text: str
    level: int = 1
    page: int | None = None          # source page (PDF); None for HTML
    kind: str = field(default="heading", init=False)


@dataclass
class Paragraph:
    text: str
    page: int | None = None
    kind: str = field(default="paragraph", init=False)


@dataclass
class ListBlock:
    items: list[str]
    ordered: bool = False
    page: int | None = None
    kind: str = field(default="list", init=False)

    def as_text(self) -> str:
        """Render the list as plain text, one item per line."""
        if self.ordered:
            return "\n".join(f"{i}. {item}" for i, item in enumerate(self.items, 1))
        return "\n".join(f"- {item}" for item in self.items)


@dataclass
class Table:
    rows: list[list[str]]
    headers: list[str] = field(default_factory=list)
    caption: str | None = None
    page: int | None = None
    kind: str = field(default="table", init=False)

    def linearize(self) -> list[str]:
        """
        Turn each row into a self-contained sentence so it survives retrieval
        as a complete fact. This is the key move for fee/credit/grade tables:
        a student asking "what is the AC hostel fee" should hit one row that
        carries its own column labels, not a fragment of a flattened grid.

        With headers:   "Room Rent: Rs.22,000, Electricity: Rs.2,000, ..."
        Without headers: "Rs.22,000, Rs.2,000, ..."
        Each line is prefixed with the caption when one exists.
        """
        prefix = f"{self.caption}: " if self.caption else ""
        lines: list[str] = []
        for row in self.rows:
            if self.headers:
                pairs: list[str] = []
                for h, c in zip(self.headers, row):
                    if not _meaningful(c):
                        continue           # drop empty / bullet-only cells
                    h = h.strip()
                    pairs.append(f"{h}: {c.strip()}" if _meaningful(h) else c.strip())
                cells = ", ".join(pairs)
            else:
                cells = ", ".join(c.strip() for c in row if _meaningful(c))
            if cells:
                lines.append(prefix + cells)
        return lines

    def as_text(self) -> str:
        """Full table as text — caption, header row, then linearized rows."""
        parts: list[str] = []
        if self.caption:
            parts.append(self.caption)
        if self.headers:
            parts.append(" | ".join(self.headers))
        parts.extend(self.linearize())
        return "\n".join(parts)


Block = Union[Heading, Paragraph, ListBlock, Table]

_KIND_TO_CLASS = {
    "heading": Heading,
    "paragraph": Paragraph,
    "list": ListBlock,
    "table": Table,
}


# ---------------------------------------------------------------------------
# Section view (heading-bounded grouping with breadcrumb)
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """
    A heading and the blocks that fall under it, with the full ancestor chain.

    `breadcrumb` is the heading hierarchy from the document root down to this
    section (inclusive). It is exactly the deterministic context blurb the
    chunker prepends before embedding — free contextual enrichment, no LLM call.
    """
    heading: str
    breadcrumb: list[str]
    blocks: list[Block]
    page: int | None = None

    @property
    def breadcrumb_str(self) -> str:
        return " > ".join(self.breadcrumb)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


@dataclass
class Document:
    url: str
    title: str
    source_format: str               # "pdf" | "html"
    doc_type: str = "unknown"        # ordinance | notice | hostel | ...
    date_published: str | None = None
    blocks: list[Block] = field(default_factory=list)

    # -- section iteration ---------------------------------------------------

    def iter_sections(self) -> Iterator[Section]:
        """
        Walk blocks in document order, splitting at every Heading and tracking
        the heading stack so each Section carries its full breadcrumb.

        Content that appears before the first heading is yielded as a synthetic
        "Preamble" section rooted at the document title, matching the PDF
        chunker's existing convention.
        """
        stack: list[tuple[int, str]] = []       # (level, heading_text)
        # Content before the first heading is filed under the document title
        # (falling back to "Preamble" only when there is no title), so a page
        # with leading prose gets a meaningful heading instead of a placeholder.
        cur_heading = self.title or "Preamble"
        cur_blocks: list[Block] = []
        cur_page: int | None = None

        def breadcrumb() -> list[str]:
            chain = [self.title] if self.title else []
            chain.extend(text for _, text in stack)
            if not stack:
                chain.append(cur_heading)
            # Collapse consecutive duplicates (e.g. title == synthesized H1).
            deduped: list[str] = []
            for c in chain:
                if not deduped or deduped[-1] != c:
                    deduped.append(c)
            return deduped

        for block in self.blocks:
            if isinstance(block, Heading):
                if cur_blocks:
                    yield Section(cur_heading, breadcrumb(), cur_blocks, cur_page)
                    cur_blocks = []
                # Pop any sibling/deeper headings, then push this one.
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                stack.append((block.level, block.text))
                cur_heading = block.text
                cur_page = block.page          # new section starts on heading's page
            else:
                if cur_page is None:
                    cur_page = block.page      # pre-heading content takes first page
                cur_blocks.append(block)

        if cur_blocks:
            yield Section(cur_heading, breadcrumb(), cur_blocks, cur_page)

    # -- serialization (parse once, chunk/eval many times) -------------------

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "source_format": self.source_format,
            "doc_type": self.doc_type,
            "date_published": self.date_published,
            "blocks": [_block_to_dict(b) for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(
            url=d["url"],
            title=d["title"],
            source_format=d["source_format"],
            doc_type=d.get("doc_type", "unknown"),
            date_published=d.get("date_published"),
            blocks=[_block_from_dict(b) for b in d.get("blocks", [])],
        )


def _block_to_dict(b: Block) -> dict:
    if isinstance(b, Heading):
        return {"kind": "heading", "text": b.text, "level": b.level, "page": b.page}
    if isinstance(b, Paragraph):
        return {"kind": "paragraph", "text": b.text, "page": b.page}
    if isinstance(b, ListBlock):
        return {"kind": "list", "items": list(b.items), "ordered": b.ordered, "page": b.page}
    if isinstance(b, Table):
        return {
            "kind": "table",
            "rows": [list(r) for r in b.rows],
            "headers": list(b.headers),
            "caption": b.caption,
            "page": b.page,
        }
    raise TypeError(f"Unknown block type: {type(b)!r}")


def _block_from_dict(d: dict) -> Block:
    kind = d.get("kind")
    page = d.get("page")
    if kind == "heading":
        return Heading(text=d["text"], level=d.get("level", 1), page=page)
    if kind == "paragraph":
        return Paragraph(text=d["text"], page=page)
    if kind == "list":
        return ListBlock(items=list(d["items"]), ordered=d.get("ordered", False), page=page)
    if kind == "table":
        return Table(
            rows=[list(r) for r in d["rows"]],
            headers=list(d.get("headers", [])),
            caption=d.get("caption"),
            page=page,
        )
    raise ValueError(f"Unknown block kind: {kind!r}")
