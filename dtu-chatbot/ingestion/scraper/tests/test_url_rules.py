"""
Tests for url_rules.py — no Scrapy or network required.
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import pytest
from ingestion.scraper.url_rules import is_allowed, is_pdf, classify_url


# ===========================================================================
# Fixtures / URL lists
# ===========================================================================

ALLOWED_URLS = [
    # Ordinance
    "https://dtu.ac.in/Web/Academics/ordinance.php",
    "https://dtu.ac.in/Web/Academics/ordinance/ug_ordinance_2022.pdf",
    "https://dtu.ac.in/Web/Academics/ordinance/pg_rules.pdf",
    # Notice index
    "https://dtu.ac.in/Web/Academics/notice.php",
    # Notice PDFs (discovered links)
    "https://dtu.ac.in/Web/notice/2023/Jan/file1234.pdf",
    "https://dtu.ac.in/Web/notice/2024/Mar/file0056.pdf",
    "https://dtu.ac.in/Web/notice/2022/Dec/file9999.pdf",
    # Scholarship
    "https://dtu.ac.in/Web/Academics/scholarship.php",
    "https://dtu.ac.in/Web/Academics/scholarship/merit_2023.pdf",
    "https://dtu.ac.in/Web/Academics/scholarship/sc_st_scholarship.pdf",
    # Departments
    "https://dtu.ac.in/Web/Departments/coe/about/",
    "https://dtu.ac.in/Web/Departments/ece/about/",
    "https://dtu.ac.in/Web/Departments/me/about/",
    "https://dtu.ac.in/Web/Departments/coe/scheme/",
    "https://dtu.ac.in/Web/Departments/ece/scheme/",
    "https://dtu.ac.in/Web/Departments/coe/faculty/",
    "https://dtu.ac.in/Web/Departments/it/faculty/",
    # Hostels
    "https://hostels.dtu.ac.in/",
    "https://hostels.dtu.ac.in/facilities/",
    "https://hostels.dtu.ac.in/rooms/",
    # Saarthi
    "https://saarthi.dtu.ac.in/admissions/",
    "https://saarthi.dtu.ac.in/admissions/ug/",
    # Exam portal
    "https://exam.dtu.ac.in/",
    "https://exam.dtu.ac.in/datesheet/",
]

DENIED_URLS = [
    # Exam results
    "https://exam.dtu.ac.in/result/btech_sem3.pdf",
    "https://exam.dtu.ac.in/Results/2024_odd.pdf",
    # Auth-gated portals
    "https://reg.exam.dtu.ac.in/",
    "https://admin.exam.dtu.ac.in/panel",
    "https://erp.dtu.ac.in/dashboard",
    "https://erpapp.dtu.ac.in/",
    "https://webkiosk.dtu.ac.in/",
    # Login pages
    "https://dtu.ac.in/Web/login",
    "https://saarthi.dtu.ac.in/login",
    "https://dtu.ac.in/auth/token",
    # Result pages
    "https://dtu.ac.in/results/",
    "https://dtu.ac.in/resultsheet.php",
    # Marksheet / reeval
    "https://exam.dtu.ac.in/marksheet/2023.pdf",
    "https://dtu.ac.in/reeval/form.php",
    # Admin
    "https://dtu.ac.in/admin/panel",
    "https://dtu.ac.in/wp-admin/",
]

PDF_URLS = [
    "https://dtu.ac.in/Web/Academics/ordinance/ug.pdf",
    "https://dtu.ac.in/Web/notice/2023/Jan/file1.pdf",
    "https://dtu.ac.in/Web/Academics/scholarship/merit.pdf",
    "https://example.com/document.PDF",
    "https://example.com/doc.pdf?ver=2",
]

NON_PDF_URLS = [
    "https://dtu.ac.in/Web/Academics/ordinance.php",
    "https://dtu.ac.in/Web/Academics/notice.php",
    "https://hostels.dtu.ac.in/",
    "https://exam.dtu.ac.in/datesheet/",
    "https://dtu.ac.in/Web/Departments/coe/about/",
]


# ===========================================================================
# is_allowed() tests
# ===========================================================================

class TestIsAllowed:
    @pytest.mark.parametrize("url", ALLOWED_URLS)
    def test_allowed_urls_pass(self, url):
        assert is_allowed(url), f"Expected allowed: {url}"

    @pytest.mark.parametrize("url", DENIED_URLS)
    def test_denied_urls_blocked(self, url):
        assert not is_allowed(url), f"Expected denied: {url}"

    def test_unknown_url_blocked(self):
        assert not is_allowed("https://random-site.com/page")

    def test_dtu_homepage_blocked(self):
        # Root DTU page is not in allowlist
        assert not is_allowed("https://dtu.ac.in/")

    def test_empty_string_blocked(self):
        assert not is_allowed("")


# ===========================================================================
# is_pdf() tests
# ===========================================================================

class TestIsPdf:
    @pytest.mark.parametrize("url", PDF_URLS)
    def test_pdf_urls_detected(self, url):
        assert is_pdf(url), f"Expected PDF: {url}"

    @pytest.mark.parametrize("url", NON_PDF_URLS)
    def test_non_pdf_urls_rejected(self, url):
        assert not is_pdf(url), f"Expected non-PDF: {url}"

    def test_pdf_uppercase_extension(self):
        assert is_pdf("https://example.com/DOC.PDF")

    def test_pdf_with_query_string(self):
        assert is_pdf("https://example.com/doc.pdf?download=1&token=abc")

    def test_pdf_not_confused_by_pdf_in_path(self):
        # "pdf" in directory name but no .pdf extension
        assert not is_pdf("https://dtu.ac.in/pdf-docs/notice.php")


# ===========================================================================
# classify_url() tests
# ===========================================================================

class TestClassifyUrl:
    def test_ordinance_index(self):
        assert classify_url("https://dtu.ac.in/Web/Academics/ordinance.php") == "ordinance_html"

    def test_ordinance_pdf(self):
        assert classify_url("https://dtu.ac.in/Web/Academics/ordinance/ug.pdf") == "ordinance_pdf"

    def test_notice_index(self):
        assert classify_url("https://dtu.ac.in/Web/Academics/notice.php") == "notice_html"

    def test_notice_pdf(self):
        cat = classify_url("https://dtu.ac.in/Web/notice/2023/Jan/file1234.pdf")
        assert cat == "notice_pdf"

    def test_scholarship_index(self):
        assert classify_url("https://dtu.ac.in/Web/Academics/scholarship.php") == "scholarship_html"

    def test_scholarship_pdf(self):
        assert classify_url("https://dtu.ac.in/Web/Academics/scholarship/merit.pdf") == "scholarship_pdf"

    def test_dept_about(self):
        assert classify_url("https://dtu.ac.in/Web/Departments/coe/about/") == "dept_about"

    def test_dept_scheme(self):
        assert classify_url("https://dtu.ac.in/Web/Departments/ece/scheme/") == "dept_scheme"

    def test_dept_faculty(self):
        assert classify_url("https://dtu.ac.in/Web/Departments/me/faculty/") == "dept_faculty"

    def test_hostel(self):
        assert classify_url("https://hostels.dtu.ac.in/") == "hostel_html"

    def test_saarthi(self):
        assert classify_url("https://saarthi.dtu.ac.in/admissions/") == "saarthi_html"

    def test_exam_portal(self):
        assert classify_url("https://exam.dtu.ac.in/") == "exam_html"

    def test_denied_returns_denied(self):
        assert classify_url("https://erp.dtu.ac.in/") == "denied"

    def test_unknown_url(self):
        assert classify_url("https://random.com/page") == "unknown"

    def test_result_url_denied(self):
        assert classify_url("https://exam.dtu.ac.in/result/sem3.pdf") == "denied"

    def test_multiple_departments(self):
        for dept in ("coe", "ece", "me", "it", "civil", "bt"):
            url = f"https://dtu.ac.in/Web/Departments/{dept}/about/"
            assert classify_url(url) == "dept_about", f"Failed for dept: {dept}"
