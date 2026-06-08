"""
URL router: maps a URL to a canonical document-type category.
Delegates all pattern matching to url_rules.py — no regex duplication.
"""
from ingestion.scraper.url_rules import classify_url as _classify

# Category constants — single source of truth
ORDINANCE_PDF    = "ordinance_pdf"
NOTICE_PDF       = "notice_pdf"
SCHOLARSHIP_PDF  = "scholarship_pdf"
DEPT_ABOUT       = "dept_about"
DEPT_SCHEME      = "dept_scheme"
DEPT_FACULTY     = "dept_faculty"
HOSTEL_HTML      = "hostel_html"
SAARTHI_HTML     = "saarthi_html"
EXAM_HTML        = "exam_html"
ORDINANCE_HTML   = "ordinance_html"
NOTICE_HTML      = "notice_html"
SCHOLARSHIP_HTML = "scholarship_html"
UNKNOWN          = "unknown"
DENIED           = "denied"

ALL_CATEGORIES = {
    ORDINANCE_PDF, NOTICE_PDF, SCHOLARSHIP_PDF,
    DEPT_ABOUT, DEPT_SCHEME, DEPT_FACULTY,
    HOSTEL_HTML, SAARTHI_HTML, EXAM_HTML,
    ORDINANCE_HTML, NOTICE_HTML, SCHOLARSHIP_HTML,
    UNKNOWN, DENIED,
}


def classify_url(url: str) -> str:
    """
    Return one of the category constants for *url*.
    Delegates to url_rules.classify_url(); remaps to typed constants.
    """
    return _classify(url)
