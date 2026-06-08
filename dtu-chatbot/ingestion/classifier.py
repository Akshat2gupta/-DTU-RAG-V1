"""
Rule-based page classifier for DTU document processing.
No ML. No external dependencies beyond the standard library.

Exposed functions:
    classify_page(text: str) -> str
    classify_document(pages: list[dict]) -> str
"""
from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Compiled patterns (shared across calls)
# ---------------------------------------------------------------------------
_LTP_RE           = re.compile(r"\bL\s*/?\s*T\s*/?\s*P\b")
_COURSE_CODE_RE   = re.compile(r"\d+\.\s+[A-Z]{2,4}-?\d{3}")
_NUMBERED_LIST_RE = re.compile(r"^\d+\.\s+[A-Z]", re.MULTILINE)
_ROLL_NO_RE       = re.compile(r"Roll Number\s*:")
_DATE_COLON_RE    = re.compile(r"\bDate\s*:")
_SHALL_BE_RE      = re.compile(r"\bshall be\b")
_ATTEND_RE        = re.compile(
    r"\battendance\b.*(?:percentage|75%)|(?:percentage|75%).*\battendance\b",
    re.IGNORECASE | re.DOTALL,
)
_DESIGNATION_RE   = re.compile(r"\bDesignation\b")
_NOTICE_HEADER_RE = re.compile(r"Circular No|F\.No|Ref:-")
_POLICY_HDR_RE    = re.compile(r"Regulation|Ordinance|R\.\s*1\s*\(B\)")
_DEAN_COE_RE      = re.compile(r"Dean \(UG\)|Dean \(PG\)|COE\b|HOD\b")
_CGPA_RE          = re.compile(r"CGPA|grade point|SGPA")


def _count(signals: list, text: str) -> int:
    """Count how many signal callables return True for this text."""
    return sum(1 for s in signals if s(text))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_page(text: str) -> str:
    """
    Classify a single page of text. Rules checked in order; first match wins.

    Returns one of: 'syllabus', 'form', 'contact', 'notice', 'policy', 'skip'
    """

    # ── SYLLABUS (3+ signals) ─────────────────────────────────────────────
    if _count([
        lambda t: bool(_LTP_RE.search(t)),
        lambda t: bool(_COURSE_CODE_RE.search(t)),
        lambda t: "Credit Hours" in t or "Contact Hours" in t,
        lambda t: "Reading List" in t or "Reference Books" in t or "Text Books" in t,
        lambda t: "Name of Books" in t or "Authors/Publisher" in t,
        lambda t: len(_NUMBERED_LIST_RE.findall(t)) >= 3,
    ], text) >= 3:
        return "syllabus"

    # ── FORM (2+ signals) ─────────────────────────────────────────────────
    if _count([
        lambda t: "FORM OF APPLICATION" in t,
        lambda t: "Proposed Project Title" in t,
        lambda t: "Signature of" in t,
        lambda t: bool(_ROLL_NO_RE.search(t)),
        lambda t: len(_DATE_COLON_RE.findall(t)) >= 2,
    ], text) >= 2:
        return "form"

    # ── CONTACT (3+ signals) ──────────────────────────────────────────────
    if _count([
        lambda t: t.count("PROFESSOR") >= 3,
        lambda t: t.count("ASSISTANT PROFESSOR") >= 2,
        lambda t: t.count("Ph.D") >= 3,
        lambda t: bool(_DESIGNATION_RE.search(t)),
    ], text) >= 3:
        return "contact"

    # ── NOTICE (2+ signals) ───────────────────────────────────────────────
    if _count([
        lambda t: bool(_NOTICE_HEADER_RE.search(t)),
        lambda t: "All Students" in t or "all students" in t,
        lambda t: "last date" in t or "Last Date" in t or "due date" in t,
        lambda t: "Dean Academic" in t or "Associate Dean" in t,
    ], text) >= 2:
        return "notice"

    # ── POLICY (2+ signals) ───────────────────────────────────────────────
    if _count([
        lambda t: bool(_POLICY_HDR_RE.search(t)),
        lambda t: len(_SHALL_BE_RE.findall(t)) >= 3,
        lambda t: bool(_DEAN_COE_RE.search(t)),
        lambda t: bool(_ATTEND_RE.search(t)),
        lambda t: bool(_CGPA_RE.search(t)),
        # Annexure pages (semester away, summer semester guidelines)
        lambda t: "annexure" in t.lower() and any(
            w in t.lower() for w in ("guidelines", "procedure", "regulation", "student")
        ),
        # Scholarship / financial support pages
        lambda t: any(
            p in t.lower() for p in ("financial support", "merit scholarship", "scholarship to")
        ),
        # Discipline and unfair means penalty pages
        lambda t: any(w in t.lower() for w in ("penalty", "penalties")) and "student" in t.lower(),
        # Industrial / field training evaluation pages
        lambda t: any(w in t.lower() for w in ("industrial training", "field training"))
            and any(w in t.lower() for w in ("evaluation", "procedure")),
        # Unfair means rules
        lambda t: "unfair means" in t.lower(),
        # Semester programme guidelines
        lambda t: any(w in t.lower() for w in ("semester away", "summer semester"))
            and any(w in t.lower() for w in ("register", "credits")),
    ], text) >= 2:
        return "policy"

    return "skip"


def classify_document(pages: list[dict]) -> str:
    """
    Classify an entire document from a list of {page_number, text} dicts.

    Returns:
        'policy'   if > 40 % of non-skip pages are policy
        'syllabus' if > 30 % of non-skip pages are syllabus
        Most common non-skip type otherwise.
        'skip'     if every page classifies as skip.
    """
    counts: Counter[str] = Counter(classify_page(p.get("text", "")) for p in pages)

    non_skip_total = sum(v for k, v in counts.items() if k != "skip")
    if non_skip_total == 0:
        return "skip"

    if counts.get("policy",   0) / non_skip_total > 0.40:
        return "policy"
    if counts.get("syllabus", 0) / non_skip_total > 0.30:
        return "syllabus"

    # Most common non-skip type
    for label, _ in counts.most_common():
        if label != "skip":
            return label

    return "skip"
