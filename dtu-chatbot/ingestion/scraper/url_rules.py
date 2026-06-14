"""
URL allowlist/denylist filtering for DTU crawler.
All regex patterns compiled once at module load.

Architecture: the allowlist is the primary gate for what gets crawled.
Denylist is checked first — a single match immediately rejects the URL.
Then at least one allowlist pattern must match for the URL to be accepted.

Categories map to document types used by the manifest and batch indexer.
"""
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Allowlist patterns
# ---------------------------------------------------------------------------
_ALLOWLIST_RAW: list[tuple[str, str]] = [

    # ── ACADEMICS UG ─────────────────────────────────────────────────────────
    ("ordinance_index",    r"dtu\.ac\.in/Web/Academics/ordinance\.php"),
    ("ordinance_pdf",      r"dtu\.ac\.in/Web/Academics/ordinance/[^/]+\.pdf$"),
    ("notice_index",       r"dtu\.ac\.in/Web/Academics/notice\.php"),
    # Two notice PDF path conventions used by DTU
    ("notice_pdf",         r"dtu\.ac\.in/Web/notice/\d{4}/[A-Za-z]+/file\d+\.pdf$"),
    ("notice_pdf",         r"dtu\.ac\.in/Web/upload/notice/\d{4}/[A-Za-z]+/file\d+\.pdf$"),
    ("scholarship_index",  r"dtu\.ac\.in/Web/Academics/scholarship\.php"),
    ("scholarship_pdf",    r"dtu\.ac\.in/Web/Academics/scholarship/[^/]+\.pdf$"),
    # All other Academics PHP pages (programmes, calendar, anti-ragging, forms…)
    ("academics_php",      r"dtu\.ac\.in/Web/Academics/[^/]+\.php$"),
    ("syllabus_pdf",       r"dtu\.ac\.in/Web/Academics/syllabus/"),
    ("forms_pdf",          r"dtu\.ac\.in/Web/Academics/forms/.*\.pdf$"),
    ("anti_ragging_pdf",   r"dtu\.ac\.in/Web/Academics/anti_ragging/.*\.pdf$"),

    # ── ACADEMICS PG ─────────────────────────────────────────────────────────
    ("pg_academics_php",   r"dtu\.ac\.in/Web/AcademicsPG/[^/]+\.php$"),
    ("pg_pdf",             r"dtu\.ac\.in/Web/AcademicsPG/.*\.pdf$"),
    ("pg_pdf",             r"dtu\.ac\.in/Platforms/academic_PG/phd/.*\.pdf$"),

    # ── DEPARTMENTS ──────────────────────────────────────────────────────────
    ("dept_about",         r"dtu\.ac\.in/Web/Departments/[^/]+/about/?$"),
    ("dept_faculty",       r"dtu\.ac\.in/Web/Departments/[^/]+/faculty"),
    ("dept_scheme",        r"dtu\.ac\.in/Web/Departments/[^/]+/[Ss]cheme"),
    # All other department sub-pages (vision, labs, placements, research…)
    ("dept_subpage",       r"dtu\.ac\.in/Web/Departments/[^/]+/"
                           r"(?:vision|message|PEOs|lab_and_infra|labs|timetable|"
                           r"placements?|placement|events|mou|alumni|patents|projects|"
                           r"publications|researchArea|sponseredProject|conferences|"
                           r"notableAlumni|research|programs?|people|training|resources|"
                           r"contact|biosoc|studentActivities|booksPublished|"
                           r"prominentFaculty|majorStrength|magazine|society|"
                           r"valueproposition|projectmouevent|phdscholars|"
                           r"MDP|AcademicResearch|Conference)"),
    ("dept_pdf",           r"dtu\.ac\.in/Web/Departments/[^/]+/.*\.pdf$"),
    ("faculty_profile",    r"dtu\.ac\.in/modules/faculty_profile_new/faculty_index\.php"),

    # ── ABOUT & ADMINISTRATION ────────────────────────────────────────────────
    ("about_php",          r"dtu\.ac\.in/Web/About/[^/]+\.php$"),
    ("about_pdf",          r"dtu\.ac\.in/Web/About/.*\.pdf$"),
    ("admin_php",          r"dtu\.ac\.in/Web/Administrations?/[^/]+\.php$"),

    # ── R&D AND INSTITUTIONAL BODIES ─────────────────────────────────────────
    ("rnd_html",           r"dtu\.ac\.in/Web/rnd/"),
    ("nceet_html",         r"dtu\.ac\.in/Web/nceet/"),
    ("icc_html",           r"dtu\.ac\.in/Web/ICC/"),
    ("enggcell_html",      r"dtu\.ac\.in/Web/enggcell/"),
    ("vigilance_html",     r"dtu\.ac\.in/Web/vigilance/"),

    # ── GOVERNANCE & QUICK LINKS ──────────────────────────────────────────────
    ("governance_php",     r"dtu\.ac\.in/Web/quick_links/[^/]+\.php$"),
    ("governance_pdf",     r"dtu\.ac\.in/Web/quick_links/.*\.pdf$"),
    ("nirf_html",          r"dtu\.ac\.in/nirf/"),
    ("publications_pdf",   r"dtu\.ac\.in/Web/publications/pdf/.*\.pdf$"),

    # ── ADMISSIONS ────────────────────────────────────────────────────────────
    ("saarthi_html",       r"saarthi\.dtu\.ac\.in/admissions"),
    ("admissions_html",    r"dtu\.ac\.in/Web/Admissions"),
    ("admissions_pdf",     r"dtu\.ac\.in/Web/Admission/brochure/.*\.pdf$"),

    # ── EXAM SUBDOMAIN ────────────────────────────────────────────────────────
    # Broad catch-all; result/marksheet/reeval pages are blocked by denylist first
    ("exam_html",          r"exam\.dtu\.ac\.in/"),
    ("exam_pdf",           r"exam\.dtu\.ac\.in/(?:DateSheet|Notices|downloads1)/.*\.pdf$"),

    # ── HOSTELS SUBDOMAIN ─────────────────────────────────────────────────────
    ("hostel_html",        r"hostels\.dtu\.ac\.in/"),

    # ── TNP SUBDOMAIN ─────────────────────────────────────────────────────────
    ("tnp_html",           r"tnp\.dtu\.ac\.in/(?:index|about|placements|students|contact)\.html$"),
    ("tnp_pdf",            r"tnp\.dtu\.ac\.in/docs/.*\.pdf$"),

    # ── LIBRARY SUBDOMAIN ─────────────────────────────────────────────────────
    ("library_html",       r"library\.dtu\.ac\.in/"),

    # ── PROGRAMME PAGES (kept for backwards compat with already-indexed docs) ─
    ("programme_html",     r"dtu\.ac\.in/Web/Academics/"
                           r"(?:masteroftechnology|bacheloroftechnology|bdes|bba|"
                           r"mba|phd|msc|mca|btech).*\.php$"),
]

_ALLOWLIST: list[tuple[str, re.Pattern]] = [
    (cat, re.compile(pat, re.IGNORECASE))
    for cat, pat in _ALLOWLIST_RAW
]

# ---------------------------------------------------------------------------
# Denylist patterns  (checked FIRST — any match → rejected)
# ---------------------------------------------------------------------------
_DENYLIST_RAW: list[str] = [
    # Exam result PDFs / result pages (block before allowlist)
    r"exam\.dtu\.ac\.in/.*[Rr]esult",
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

    # Saarthi transactional portals (fee payment, hostel allotment, applications)
    r"saarthi\.dtu\.ac\.in/feeModule",
    r"saarthi\.dtu\.ac\.in/hostelModule",
    r"saarthi\.dtu\.ac\.in/hostel/",

    # Separate hostel allotment portal
    r"dtuhostel\.in",

    # Remote library access (login-gated)
    r"dtulibrary\.remotexs\.in",

    # DTU OPAC (external JS app, not crawlable)
    r"dtu\.bestbookbuddies\.com",

    # Login / auth / registration endpoints
    r"/[Ll]ogin",
    r"/[Ss]ignup",
    r"/register",
    r"/[Aa]uth",
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
