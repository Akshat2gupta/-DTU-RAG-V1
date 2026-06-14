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
_NOTICE_HEADER_RE = re.compile(r"Circular No|F\.\s*No|Ref:-")   # "F. No." (OCR) and "F.No"
_DDMMYYYY_RE      = re.compile(r"\d{2}\.\d{2}\.\d{4}")          # 22.09.2025-style schedule dates
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

    # ── TRANSCRIPT / DEGREE-SERVICE FORMS (before FORM) ───────────────────
    # Transcript request forms carry the fee schedule and procedure students
    # ask about — keep them even though they look like blank forms.
    if "transcript" in text.lower() and any(
        w in text.lower() for w in ("fee", "verification", "duplicate")
    ):
        return "policy"

    # ── NCC / NSS pages ───────────────────────────────────────────────────
    if re.search(r"National Cadet Corps|National Service Scheme", text) or \
            re.search(r"\bNCC\b|\bNSS\b", text):
        return "policy"

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
        # Academic-calendar / schedule pages: title phrase or a run of dates
        lambda t: "academic calendar" in t.lower(),
        lambda t: len(_DDMMYYYY_RE.findall(t)) >= 4,
    ], text) >= 2:
        return "notice"

    # ── PLACEMENT (2+ signals) ───────────────────────────────────────────
    if _count([
        lambda t: bool(re.search(r"placement highlight|placement stat|placement record", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"No\.?\s*of Offers|number of offers", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"Average CTC|Median CTC|Highest CTC|Maximum CTC|LPA|lakh per annum", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"Students Registered for Placement|eligible students", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"placement coordinator|Training.*Placement|T&P|TPC|TNP", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"campus recruit|on.campus|off.campus", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"placement brochure|placement policy|placement procedure", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"top recruiter|hiring partner|dream offer|super dream", t, re.IGNORECASE)),
        # Placement brochure / report content (covers MBA brochure pages)
        lambda t: bool(re.search(r"final.?placement.?report|placement.?season|placement.?overview", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"\bCTC\b", t)) and bool(re.search(r"placement|MBA|package|salary", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"recruiters?|recruiting companies|companies? visit", t, re.IGNORECASE))
            and bool(re.search(r"DTU|campus|placement", t, re.IGNORECASE)),
        # Placement-data tables ("No. of Students Placed", "% Students Placed")
        lambda t: bool(re.search(r"\bplaced\b", t, re.IGNORECASE))
            and bool(re.search(r"placement|T&P|students", t, re.IGNORECASE)),
    ], text) >= 2:
        return "placement"

    # ── ADMISSIONS (2+ signals) ──────────────────────────────────────────
    if _count([
        lambda t: bool(re.search(r"joint admission|JAC Delhi|admission counsell", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"seat matrix|choice filling|allotment", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"JEE Main|JEE rank|All India Rank|opening rank|closing rank", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"online registration", t, re.IGNORECASE))
            and bool(re.search(r"admission|rank|seat|counsel", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"B\.?\s*Tech", t, re.IGNORECASE))
            and bool(re.search(r"admission|counsel|seat|programme|eligib", t, re.IGNORECASE)),
        lambda t: bool(re.search(r"eligibility criteria|document verification|reporting", t, re.IGNORECASE)),
    ], text) >= 2:
        return "admissions"

    # ── HOSTEL RULES (2+ signals) ─────────────────────────────────────────
    # Hostel bulletin / allotment-policy pages: rules prose with none of the
    # academic signals below.  Returns 'policy' so downstream page handling
    # (chunker class_counts, extract_pages keep-list) needs no new label.
    if _count([
        lambda t: "hostel" in t.lower(),
        lambda t: any(w in t.lower() for w in ("warden", "allottee", "allotment")),
        lambda t: any(w in t.lower() for w in ("resident", "inmate", "hosteller")),
        lambda t: "mess" in t.lower(),
        lambda t: "ragging" in t.lower(),
    ], text) >= 2:
        return "policy"

    # Anti-ragging regulation pages (UGC text): "ragging" alone is specific
    # enough — it never appears on syllabus/form/contact pages.
    if "ragging" in text.lower():
        return "policy"

    # ── EXAM-BRANCH RULES (both signals) ──────────────────────────────────
    # "RULES FOR RECHECKING OF RESULTS"-style instruction pages attached to
    # exam-branch forms; the blank form page itself still classifies as form.
    if re.search(r"rules\s+for\s+\w+", text, re.IGNORECASE) and \
            len(_NUMBERED_LIST_RE.findall(text)) >= 2:
        return "policy"

    # ── POLICY (2+ signals) ───────────────────────────────────────────────
    if _count([
        lambda t: bool(_POLICY_HDR_RE.search(t)),
        lambda t: len(_SHALL_BE_RE.findall(t)) >= 3,
        lambda t: bool(_DEAN_COE_RE.search(t)),
        lambda t: bool(_ATTEND_RE.search(t)),
        lambda t: bool(_CGPA_RE.search(t)),
        # CGPA-to-percentage conversion notices
        lambda t: bool(_CGPA_RE.search(t)) and "percentage" in t.lower(),
        # Academic grading structure tables (O/A+/A/B grading scheme pages)
        lambda t: bool(re.search(r"letter\s*grade|numerical\s*grade", t, re.IGNORECASE))
            and bool(re.search(r"outstanding|excellent|formula|grade.?point|SGPA|CGPA", t, re.IGNORECASE)),
        # Credit/programme structure tables
        lambda t: bool(re.search(r"minimum\s*credits|maximum\s*credits|programme\s*credits", t, re.IGNORECASE)),
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
    counts: Counter[str] = Counter(
        classify_page(p.get("text") or p.get("raw_text", "")) for p in pages
    )

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
