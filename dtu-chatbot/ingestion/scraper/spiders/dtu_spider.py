"""
Unified Scrapy spider for all DTU domains.
"""
from __future__ import annotations

import os
from typing import Generator, Any

import scrapy
from scrapy.http import Response, Request

from ingestion.scraper.http_meta import extract_http_meta
from ingestion.scraper.items import CrawlRecord
from ingestion.scraper.url_rules import is_allowed, is_pdf, classify_url

import re
_RESULT_PDF_RE = re.compile(r"result", re.IGNORECASE)
_DATESHEET_RE  = re.compile(r"datesheet|date.?sheet|time.?table", re.IGNORECASE)

# Categories whose pages should have all dept_* sub-page links followed
_DEPT_CATS = frozenset({
    "dept_about", "dept_scheme", "dept_faculty", "dept_subpage",
})

# Categories routed to parse_generic_html
_GENERIC_HTML_CATS = frozenset({
    "academics_php", "pg_academics_php", "about_php", "admin_php",
    "rnd_html", "nceet_html", "icc_html", "enggcell_html", "vigilance_html",
    "governance_php", "nirf_html", "library_html", "faculty_profile",
    "programme_html",
})

# PDF-only categories (no HTML body needed)
_PDF_CATS = frozenset({
    "ordinance_pdf", "notice_pdf", "scholarship_pdf", "syllabus_pdf",
    "forms_pdf", "anti_ragging_pdf", "pg_pdf", "dept_pdf",
    "governance_pdf", "publications_pdf", "about_pdf", "admissions_pdf",
    "exam_pdf", "tnp_pdf",
})


class DtuSpider(scrapy.Spider):
    name = "dtu"
    allowed_domains = [
        "dtu.ac.in",
        "hostels.dtu.ac.in",
        "saarthi.dtu.ac.in",
        "exam.dtu.ac.in",
        "tnp.dtu.ac.in",
        "library.dtu.ac.in",
    ]
    start_urls = [
        # ── Academics UG hub (links to all programme + policy pages) ──────────
        "https://dtu.ac.in/Web/Academics/",
        "https://dtu.ac.in/Web/Academics/ordinance.php",
        "https://dtu.ac.in/Web/Academics/notice.php",
        "https://dtu.ac.in/Web/Academics/scholarship.php",

        # ── Academics PG ─────────────────────────────────────────────────────
        "https://dtu.ac.in/Web/AcademicsPG/",

        # ── About & Administration ────────────────────────────────────────────
        "https://dtu.ac.in/Web/About/",
        "https://dtu.ac.in/Web/Administrations/",

        # ── Governance & quick links ──────────────────────────────────────────
        "https://dtu.ac.in/Web/quick_links/",

        # ── R&D and institutional bodies ──────────────────────────────────────
        "https://dtu.ac.in/Web/rnd/",

        # ── Admissions ────────────────────────────────────────────────────────
        "https://dtu.ac.in/Web/Admissions/",
        "https://saarthi.dtu.ac.in/admissions2026_27/",
        "https://saarthi.dtu.ac.in/admissions2025_26/",

        # ── Training & Placement Cell ─────────────────────────────────────────
        "https://tnp.dtu.ac.in/index.html",

        # ── Library ───────────────────────────────────────────────────────────
        "https://library.dtu.ac.in/",

        # ── Exam portal (index pages only; results are denylist-blocked) ──────
        "https://exam.dtu.ac.in/",
        "https://exam.dtu.ac.in/DateSheet.htm",
        "https://exam.dtu.ac.in/Notices-n-Circulars.htm",

        # ── Hostels (seed sub-pages directly — root nav was not followed) ─────
        "https://hostels.dtu.ac.in/",
        "https://hostels.dtu.ac.in/about.html",
        "https://hostels.dtu.ac.in/fee.html",
        "https://hostels.dtu.ac.in/contact.html",

        # ── Departments — core engineering ────────────────────────────────────
        "https://dtu.ac.in/Web/Departments/CSE/about",
        "https://dtu.ac.in/Web/Departments/CSE/faculty",
        "https://dtu.ac.in/Web/Departments/Electronics/about",
        "https://dtu.ac.in/Web/Departments/Electronics/faculty",
        "https://dtu.ac.in/Web/Departments/Electrical/about",
        "https://dtu.ac.in/Web/Departments/Electrical/faculty",
        "https://dtu.ac.in/Web/Departments/Mechanical/about",
        "https://dtu.ac.in/Web/Departments/Mechanical/faculty",
        "https://dtu.ac.in/Web/Departments/Civil/about",
        "https://dtu.ac.in/Web/Departments/Civil/faculty",
        "https://dtu.ac.in/Web/Departments/InformationTechnology/about",
        "https://dtu.ac.in/Web/Departments/InformationTechnology/faculty",
        "https://dtu.ac.in/Web/Departments/SE/about",
        "https://dtu.ac.in/Web/Departments/SE/faculty",
        "https://dtu.ac.in/Web/Departments/PE/about",
        "https://dtu.ac.in/Web/Departments/PE/faculty",
        "https://dtu.ac.in/Web/Departments/Environment/about",
        "https://dtu.ac.in/Web/Departments/Environment/faculty",
        "https://dtu.ac.in/Web/Departments/COE/about",
        "https://dtu.ac.in/Web/Departments/COE/faculty",

        # ── Departments — science & humanities ────────────────────────────────
        "https://dtu.ac.in/Web/Departments/AppliedMathematics/about",
        "https://dtu.ac.in/Web/Departments/AppliedMathematics/faculty",
        "https://dtu.ac.in/Web/Departments/AppliedPhysics/about",
        "https://dtu.ac.in/Web/Departments/AppliedPhysics/faculty",
        "https://dtu.ac.in/Web/Departments/AppliedChemistry/about",
        "https://dtu.ac.in/Web/Departments/AppliedChemistry/faculty",
        "https://dtu.ac.in/Web/Departments/BioTech/about",
        "https://dtu.ac.in/Web/Departments/BioTech/faculty",
        "https://dtu.ac.in/Web/Departments/Humanities/about",
        "https://dtu.ac.in/Web/Departments/Humanities/faculty",

        # ── Departments — management, centres, east campus ────────────────────
        "https://dtu.ac.in/Web/Departments/DSM/about",
        "https://dtu.ac.in/Web/Departments/DSM/faculty",
        "https://dtu.ac.in/Web/Departments/MCG/about",
        "https://dtu.ac.in/Web/Departments/MCG/faculty",
        "https://dtu.ac.in/Web/Departments/EVRT/about",
        "https://dtu.ac.in/Web/Departments/EVRT/faculty",
        "https://dtu.ac.in/Web/Departments/eastcampus/about",
        "https://dtu.ac.in/Web/Departments/ccdr/about",
    ]

    custom_settings = {
        "JOBDIR": os.environ.get("SCRAPY_JOBDIR", None),
    }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def start_requests(self):
        for url in self.start_urls:
            req = self._make_request(url)
            if req:
                yield req

    def _make_request(self, url: str, cb=None, **kw) -> Request | None:
        if not is_allowed(url):
            return None
        if is_pdf(url):
            return Request(url, method="HEAD", callback=cb or self._handle_pdf, **kw)
        category = classify_url(url)
        callback = self._route_callback(category, cb)
        return Request(url, callback=callback, **kw)

    def _route_callback(self, category: str, override=None):
        if override:
            return override
        routing: dict[str, Any] = {
            # Specialized index parsers
            "ordinance_html":   self.parse_ordinance_index,
            "notice_html":      self.parse_notice_index,
            "scholarship_html": self.parse_scholarship_index,
            "exam_html":        self.parse_exam_portal,
            # Dept pages (all sub-types share the same parser)
            "dept_about":       self.parse_department,
            "dept_scheme":      self.parse_department,
            "dept_faculty":     self.parse_department,
            "dept_subpage":     self.parse_department,
            # Domain-specific parsers
            "hostel_html":      self.parse_hostel,
            "saarthi_html":     self.parse_saarthi,
            "tnp_html":         self.parse_tnp,
            "admissions_html":  self.parse_admissions,
            # PDF-only
            **{cat: self._handle_pdf for cat in _PDF_CATS},
        }
        # Everything else (generic HTML categories) uses the generic parser
        return routing.get(category, self.parse_generic_html)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _build_record(self, response: Response, link_date_label: str | None = None) -> CrawlRecord:
        meta = extract_http_meta(response)
        category = classify_url(response.url)
        return CrawlRecord(
            url=response.url,
            category=category,
            doc_type=category,
            http_status=response.status,
            content_type=meta["content_type"],
            last_modified=meta["last_modified"],
            etag=meta["etag"],
            content_length=meta["content_length"],
            crawl_timestamp=meta["crawl_timestamp"],
            link_date_label=link_date_label,
            html_body=None,
            html_body_path=None,
        )

    def _handle_pdf(self, response: Response, link_date_label: str | None = None):
        item = self._build_record(response, link_date_label)
        item["html_body"] = None
        yield item

    def _yield_html_record(self, response: Response, label: str | None = None):
        item = self._build_record(response, label)
        item["html_body"] = response.text
        yield item

    def _follow_if_allowed(self, url: str, cb=None, meta: dict | None = None) -> Request | None:
        if not url or not is_allowed(url):
            return None
        req = self._make_request(url, cb=cb)
        if req and meta:
            req = req.replace(meta={**req.meta, **meta})
        return req

    # ------------------------------------------------------------------
    # Generic HTML parser — save page + follow any allowed links
    # ------------------------------------------------------------------

    def parse_generic_html(self, response: Response):
        yield from self._yield_html_record(response)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if not is_allowed(url):
                continue
            cb = self._handle_pdf if is_pdf(url) else None
            req = self._follow_if_allowed(url, cb=cb)
            if req:
                yield req

    # ------------------------------------------------------------------
    # Specialized parsers
    # ------------------------------------------------------------------

    def parse_ordinance_index(self, response: Response):
        yield from self._yield_html_record(response)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if is_allowed(url) and is_pdf(url):
                req = self._follow_if_allowed(url, cb=self._handle_pdf)
                if req:
                    yield req

    def parse_notice_index(self, response: Response):
        yield from self._yield_html_record(response)
        for row in response.css("table tr"):
            cells = row.css("td")
            date_label = cells[0].css("::text").get("").strip() if cells else None
            for a in row.css("a[href]"):
                href = a.attrib["href"]
                url = response.urljoin(href)
                if is_allowed(url) and is_pdf(url):
                    req = self._follow_if_allowed(
                        url,
                        cb=self._handle_pdf_with_label,
                        meta={"link_date_label": date_label},
                    )
                    if req:
                        yield req

    def _handle_pdf_with_label(self, response: Response):
        label = response.meta.get("link_date_label")
        yield from self._handle_pdf(response, link_date_label=label)

    def parse_scholarship_index(self, response: Response):
        yield from self._yield_html_record(response)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if is_allowed(url) and is_pdf(url):
                req = self._follow_if_allowed(url, cb=self._handle_pdf)
                if req:
                    yield req

    def parse_department(self, response: Response):
        """Save dept page + follow any dept sub-page, faculty profile, or dept PDF link."""
        yield from self._yield_html_record(response)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if not is_allowed(url):
                continue
            if is_pdf(url):
                req = self._follow_if_allowed(url, cb=self._handle_pdf)
            else:
                cat = classify_url(url)
                if cat in _DEPT_CATS or cat == "faculty_profile":
                    req = self._follow_if_allowed(url)
                else:
                    continue
            if req:
                yield req

    def parse_hostel(self, response: Response):
        yield from self._yield_html_record(response)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if is_allowed(url) and classify_url(url) == "hostel_html":
                req = self._follow_if_allowed(url)
                if req:
                    yield req

    def parse_saarthi(self, response: Response):
        yield from self._yield_html_record(response)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if is_allowed(url) and classify_url(url) == "saarthi_html":
                req = self._follow_if_allowed(url)
                if req:
                    yield req

    def parse_tnp(self, response: Response):
        yield from self._yield_html_record(response)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if not is_allowed(url):
                continue
            if is_pdf(url):
                req = self._follow_if_allowed(url, cb=self._handle_pdf)
            elif classify_url(url) == "tnp_html":
                req = self._follow_if_allowed(url)
            else:
                continue
            if req:
                yield req

    def parse_admissions(self, response: Response):
        yield from self._yield_html_record(response)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if not is_allowed(url):
                continue
            if is_pdf(url):
                req = self._follow_if_allowed(url, cb=self._handle_pdf)
            elif classify_url(url) == "admissions_html":
                req = self._follow_if_allowed(url)
            else:
                continue
            if req:
                yield req

    def parse_exam_portal(self, response: Response):
        yield from self._yield_html_record(response)
        for a in response.css("a[href]"):
            href = a.attrib["href"]
            url = response.urljoin(href)
            if not is_allowed(url):
                continue
            if is_pdf(url):
                # Block result PDFs; allow datesheets and exam notices
                if not _RESULT_PDF_RE.search(url):
                    req = self._follow_if_allowed(url, cb=self._handle_pdf)
                    if req:
                        yield req
            elif classify_url(url) == "exam_html":
                req = self._follow_if_allowed(url)
                if req:
                    yield req

    def parse(self, response: Response):
        yield from self._yield_html_record(response)
