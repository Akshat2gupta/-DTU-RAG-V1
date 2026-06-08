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

# Exam result PDF pattern — blocked even within exam.dtu.ac.in
import re
_RESULT_PDF_RE = re.compile(r"result", re.IGNORECASE)
_DATESHEET_RE  = re.compile(r"datesheet|date.?sheet|time.?table", re.IGNORECASE)


class DtuSpider(scrapy.Spider):
    name = "dtu"
    allowed_domains = [
        "dtu.ac.in",
        "hostels.dtu.ac.in",
        "saarthi.dtu.ac.in",
        "exam.dtu.ac.in",
    ]
    start_urls = [
        "https://dtu.ac.in/Web/Academics/ordinance.php",
        "https://dtu.ac.in/Web/Academics/notice.php",
        "https://dtu.ac.in/Web/Academics/scholarship.php",
        "https://dtu.ac.in/Web/Departments/coe/about/",
        "https://dtu.ac.in/Web/Departments/coe/scheme/",
        "https://dtu.ac.in/Web/Departments/coe/faculty/",
        "https://dtu.ac.in/Web/Departments/ece/about/",
        "https://dtu.ac.in/Web/Departments/ece/scheme/",
        "https://dtu.ac.in/Web/Departments/ece/faculty/",
        "https://hostels.dtu.ac.in/",
        "https://saarthi.dtu.ac.in/admissions/",
        "https://exam.dtu.ac.in/",
    ]

    custom_settings = {
        "JOBDIR": os.environ.get("SCRAPY_JOBDIR", None),
    }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def start_requests(self):
        for url in self.start_urls:
            yield self._make_request(url)

    def _make_request(self, url: str, cb=None, **kw) -> Request:
        """Build a request with appropriate callback and method."""
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
        routing = {
            "ordinance_html":    self.parse_ordinance_index,
            "ordinance_pdf":     self._handle_pdf,
            "notice_html":       self.parse_notice_index,
            "notice_pdf":        self._handle_pdf,
            "scholarship_html":  self.parse_scholarship_index,
            "scholarship_pdf":   self._handle_pdf,
            "dept_about":        self.parse_department,
            "dept_scheme":       self.parse_department,
            "dept_faculty":      self.parse_department,
            "hostel_html":       self.parse_hostel,
            "saarthi_html":      self.parse_saarthi_landing,
            "exam_html":         self.parse_exam_portal,
        }
        return routing.get(category, self.parse_department)

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
    # Parsers
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
        # Table rows: <tr><td>date label</td><td><a href="...pdf">...</a></td></tr>
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
        yield from self._yield_html_record(response)
        for href in response.css("a::attr(href)").getall():
            url = response.urljoin(href)
            if not is_allowed(url):
                continue
            if is_pdf(url):
                req = self._follow_if_allowed(url, cb=self._handle_pdf)
            else:
                category = classify_url(url)
                if category in ("dept_about", "dept_scheme", "dept_faculty"):
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

    def parse_saarthi_landing(self, response: Response):
        # Yield ONE record only — no link following (auth-gated downstream)
        yield from self._yield_html_record(response)

    def parse_exam_portal(self, response: Response):
        yield from self._yield_html_record(response)
        for a in response.css("a[href]"):
            href = a.attrib["href"]
            url = response.urljoin(href)
            if not is_allowed(url):
                continue
            # Only follow datesheet/timetable PDFs; block result PDFs
            if is_pdf(url):
                if _DATESHEET_RE.search(url) and not _RESULT_PDF_RE.search(url):
                    req = self._follow_if_allowed(url, cb=self._handle_pdf)
                    if req:
                        yield req
            elif classify_url(url) == "exam_html":
                req = self._follow_if_allowed(url)
                if req:
                    yield req
