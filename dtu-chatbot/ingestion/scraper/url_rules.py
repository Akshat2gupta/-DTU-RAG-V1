"""
URL allowlist/denylist filtering for DTU crawler.
All regex patterns compiled once at module load.
"""
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Allowlist patterns
# ---------------------------------------------------------------------------
_ALLOWLIST_RAW: list[tuple[str, str]] = [
    ("ordinance_index",    r"dtu\.ac\.in/Web/Academics/ordinance\.php"),
    ("ordinance_pdf",      r"dtu\.ac\.in/Web/Academics/ordinance/[^/]+\.pdf$"),
    ("notice_index",       r"dtu\.ac\.in/Web/Academics/notice\.php"),
    ("notice_pdf",         r"dtu\.ac\.in/Web/notice/\d{4}/[A-Za-z]+/file\d+\.pdf$"),
    ("scholarship_index",  r"dtu\.ac\.in/Web/Academics/scholarship\.php"),
    ("scholarship_pdf",    r"dtu\.ac\.in/Web/Academics/scholarship/[^/]+\.pdf$"),
    ("dept_about",         r"dtu\.ac\.in/Web/Departments/[^/]+/about/?$"),
    ("dept_scheme",        r"dtu\.ac\.in/Web/Departments/[^/]+/scheme/?"),
    ("dept_faculty",       r"dtu\.ac\.in/Web/Departments/[^/]+/faculty/?"),
    ("hostel_html",        r"hostels\.dtu\.ac\.in/"),
    ("saarthi_html",       r"saarthi\.dtu\.ac\.in/admissions"),   # matches admissions2026_27/ etc.
    ("exam_html",          r"exam\.dtu\.ac\.in/"),
]

_ALLOWLIST: list[tuple[str, re.Pattern]] = [
    (cat, re.compile(pat, re.IGNORECASE))
    for cat, pat in _ALLOWLIST_RAW
]

# ---------------------------------------------------------------------------
# Denylist patterns  (checked FIRST — any match → rejected)
# ---------------------------------------------------------------------------
_DENYLIST_RAW: list[str] = [
    # Exam result PDFs / result pages
    r"exam\.dtu\.ac\.in/.*result",
    r"exam\.dtu\.ac\.in/.*Result",
    r"/result[s]?/",
    r"result[s]?\.php",
    # Auth-gated sub-domains / portals
    r"reg\.exam\.dtu\.ac\.in",
    r"admin\.exam\.dtu\.ac\.in",
    r"erp\.dtu\.ac\.in",
    r"erpapp\.dtu\.ac\.in",
    r"webkiosk\.dtu\.ac\.in",
    r"fees\.dtu\.ac\.in",
    r"payroll\.dtu\.ac\.in",
    r"library\.dtu\.ac\.in/cgi-bin",
    # Login / auth / registration endpoints
    r"/login",
    r"/Login",
    r"/signup",
    r"/register",
    r"/auth",
    r"/portal",
    # Boilerplate / admin pages
    r"/wp-admin",
    r"/wp-login",
    r"\.dtu\.ac\.in/admin",
    # Raw result marksheets, re-evaluation, etc.
    r"/reeval",
    r"/rechecking",
    r"/marksheet",
    r"atkt",
]

_DENYLIST: list[re.Pattern] = [
    re.compile(pat, re.IGNORECASE) for pat in _DENYLIST_RAW
]

# ---------------------------------------------------------------------------
# PDF detection
# ---------------------------------------------------------------------------
_PDF_RE = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)


def is_pdf(url: str) -> bool:
    """Return True if the URL points to a PDF file."""
    return bool(_PDF_RE.search(url))


def is_allowed(url: str) -> bool:
    """
    Return True if url passes filtering rules.
    Denylist is checked first; a single denylist match returns False immediately.
    Then at least one allowlist pattern must match.
    """
    for deny_re in _DENYLIST:
        if deny_re.search(url):
            return False
    for _cat, allow_re in _ALLOWLIST:
        if allow_re.search(url):
            return True
    return False


def classify_url(url: str) -> str:
    """
    Return the category string for a URL.
    Returns 'unknown' if no allowlist pattern matches (after deny check).
    """
    for deny_re in _DENYLIST:
        if deny_re.search(url):
            return "denied"
    for cat, allow_re in _ALLOWLIST:
        if allow_re.search(url):
            # Map index pages to their canonical category
            if cat == "ordinance_index":
                return "ordinance_html"
            if cat == "notice_index":
                return "notice_html"
            if cat == "scholarship_index":
                return "scholarship_html"
            return cat
    return "unknown"
