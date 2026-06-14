#!/usr/bin/env python3
"""
One-off targeted scrape: exam.dtu.ac.in + hostels.dtu.ac.in deep pages.

Downloads the pages/PDFs that the generic-query eval showed as content gaps
(hostel rules/fees/per-hostel pages, exam forms, CGPA-conversion notice),
saves them under dtu-chatbot/data/raw/, and registers manifest rows so the
existing indexers pick them up:

    python vertical_slice/scrape_missing_pages.py
    python vertical_slice/html_batch_index.py
    python vertical_slice/batch_index.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_HERE     = Path(__file__).resolve().parent
_CHATBOT  = _HERE.parent / "dtu-chatbot"
_HTML_DIR = _CHATBOT / "data" / "raw" / "html"
_PDF_DIR  = _CHATBOT / "data" / "raw" / "pdfs"

sys.path.insert(0, str(_CHATBOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from manifest.manifest import ManifestDB

MANIFEST_DB = _CHATBOT / "manifest" / "manifest.db"

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DTU-RAG-ingest/1.0)"}

# (url, category, document_type, title)
HTML_TARGETS = [
    # exam.dtu.ac.in
    ("https://exam.dtu.ac.in/Performa.htm",            "exam_html", "exam_page", "Examination Branch - Forms and Performa"),
    ("https://exam.dtu.ac.in/Notices-n-Circulars.htm", "exam_html", "exam_page", "Examination Branch - Notices and Circulars"),
    ("https://exam.dtu.ac.in/Contactus.htm",           "exam_html", "exam_page", "Examination Branch - Contact Us"),
    # hostels.dtu.ac.in core pages
    ("https://hostels.dtu.ac.in/about.html",    "hostel_html", "hostel_page", "DTU Hostels - About"),
    ("https://hostels.dtu.ac.in/fee.html",      "hostel_html", "hostel_page", "DTU Hostels - Fee Structure"),
    ("https://hostels.dtu.ac.in/complain.html", "hostel_html", "hostel_page", "DTU Hostels - Rules, Regulations and Complaints"),
    ("https://hostels.dtu.ac.in/contact.html",  "hostel_html", "hostel_page", "DTU Hostels - Contact"),
    # per-hostel pages
    ("https://hostels.dtu.ac.in/t2.html",  "hostel_html", "hostel_page", "Type II Hostel"),
    ("https://hostels.dtu.ac.in/abh.html", "hostel_html", "hostel_page", "Aryabhatt Hostel"),
    ("https://hostels.dtu.ac.in/bah.html", "hostel_html", "hostel_page", "Bhaskaracharya Hostel"),
    ("https://hostels.dtu.ac.in/cvr.html", "hostel_html", "hostel_page", "Sir C.V. Raman Hostel"),
    ("https://hostels.dtu.ac.in/jcb.html", "hostel_html", "hostel_page", "Sir J.C. Bose Hostel"),
    ("https://hostels.dtu.ac.in/vmh.html", "hostel_html", "hostel_page", "Varahmihir Hostel"),
    ("https://hostels.dtu.ac.in/svh.html", "hostel_html", "hostel_page", "Sir Vishveshwarya Hostel"),
    ("https://hostels.dtu.ac.in/hjb.html", "hostel_html", "hostel_page", "Homi Jehangir Bhabha Hostel"),
    ("https://hostels.dtu.ac.in/rmh.html", "hostel_html", "hostel_page", "Ramanujan Hostel"),
    ("https://hostels.dtu.ac.in/apj.html", "hostel_html", "hostel_page", "Dr. APJ Abdul Kalam Hostel"),
    ("https://hostels.dtu.ac.in/snh.html", "hostel_html", "hostel_page", "Sister Nivedita Hostel"),
    ("https://hostels.dtu.ac.in/kch.html", "hostel_html", "hostel_page", "Kalpana Chawla Hostel"),
    ("https://hostels.dtu.ac.in/vlb.html", "hostel_html", "hostel_page", "Virangana Lakshmibai Hostel"),
    # round 2 — student life / NSS / sports (dsw.dtu.ac.in + Community)
    ("https://dtu.ac.in/Web/Community/nss.php",                       "community_html", "community_page", "National Service Scheme (NSS) at DTU"),
    ("https://dsw.dtu.ac.in/Student-Activites/Cultural_Council.php",  "dsw_html", "dsw_page", "DTU Cultural Council - Clubs and Societies"),
    ("https://dsw.dtu.ac.in/Student-Activites/Literary_Council.php",  "dsw_html", "dsw_page", "DTU Literary Council - Clubs and Societies"),
    ("https://dsw.dtu.ac.in/Student-Activites/Technical_Council.php", "dsw_html", "dsw_page", "DTU Technical Council - Clubs and Societies"),
    ("https://dsw.dtu.ac.in/Student-Activites/Innovative_Team.php",   "dsw_html", "dsw_page", "DTU Innovative Teams"),
    ("https://dsw.dtu.ac.in/Student-Activites/Social_Socities.php",   "dsw_html", "dsw_page", "DTU Social Societies (NSS, social sector clubs)"),
    ("https://dsw.dtu.ac.in/Community/Sports_Games.php",              "dsw_html", "dsw_page", "DTU Sports and Games Facilities"),
    # round 2 — about / academics utility pages
    ("https://dtu.ac.in/Web/About/contactus.php",               "about_php",     "about_page",     "DTU Contact Us - Address and How to Reach"),
    ("https://dtu.ac.in/Web/Academics/academic_calender.php",   "academics_php", "academics_page", "DTU Academic Calendar page"),
    ("https://dtu.ac.in/Web/Academics/forms.php",               "academics_php", "academics_page", "DTU Academic Forms (transcript, migration, duplicate)"),
    # round 2 — remaining departments
    ("https://dtu.ac.in/Web/Departments/Design/about/",   "dept_about",   "dept_page", "Department of Design - About"),
    ("https://dtu.ac.in/Web/Departments/Design/faculty/", "dept_faculty", "dept_page", "Department of Design - Faculty"),
    ("https://dtu.ac.in/Web/Departments/DSM/about/",      "dept_about",   "dept_page", "Delhi School of Management - About"),
    ("https://dtu.ac.in/Web/Departments/DSM/faculty/",    "dept_faculty", "dept_page", "Delhi School of Management - Faculty"),
]

PDF_TARGETS = [
    ("https://hostels.dtu.ac.in/pages/about-us/Hostel-Bulletine-2024-25.pdf",
     "hostel_pdf", "hostel_bulletin", "DTU Hostel Bulletin of Information 2024-25"),
    ("https://exam.dtu.ac.in/downloads1/RECHECKING.pdf",
     "exam_pdf", "exam_form", "Application for Rechecking of Answer Sheets"),
    ("https://exam.dtu.ac.in/downloads1/DISCREPANCY_PERFORMA_09062017.pdf",
     "exam_pdf", "exam_form", "Requisition Form for Result and Document Corrections"),
    ("http://dtu.ac.in/Web/notice/2018/sep/file0906.pdf",
     "notice_pdf", "notice", "CGPA to Percentage Conversion Formula Notice"),
    # round 2 — placement stats, NCC, academic calendar, transcript form, NIRF
    ("https://dtu.ac.in/Web/Placement_data.pdf",
     "placement_pdf", "placement_stats", "DTU Placement Data - Students Placed Statistics"),
    ("https://dtu.ac.in/Web/Community/ncc.pdf",
     "community_pdf", "community_page", "National Cadet Corps (NCC) at DTU"),
    ("https://dtu.ac.in/Web/notice/2025/june/file0681.pdf",
     "calendar_pdf", "academic_calendar", "DTU Academic Calendar AY 2025-26 (all UG PG PhD programs)"),
    ("https://dtu.ac.in/Web/Academics/forms/Trarscript_Form.pdf",
     "exam_pdf", "exam_form", "DTU Transcript Application Form"),
    ("https://www.dtu.ac.in/Web/quick_links/nirf/All%20Report-MHRD,%20National%20Institutional%20Ranking%20Framework%20(NIRF)-Overall.pdf",
     "nirf_pdf", "nirf_report", "DTU NIRF Overall Report - placement and institutional data"),
]


def _fetch(url: str) -> bytes | None:
    for verify in (True, False):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=30, verify=verify)
            if r.status_code == 200 and r.content:
                return r.content
            print(f"  HTTP {r.status_code}")
            return None
        except requests.exceptions.SSLError:
            continue
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return None
    return None


def main() -> None:
    _HTML_DIR.mkdir(parents=True, exist_ok=True)
    _PDF_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with ManifestDB(MANIFEST_DB) as db:
        for url, category, doc_type, title in HTML_TARGETS:
            print(f"[html] {url}")
            body = _fetch(url)
            if body is None:
                continue
            sha  = hashlib.sha256(url.encode()).hexdigest()
            path = _HTML_DIR / f"{sha}.html"
            path.write_bytes(body)
            db.insert_document(
                url, category, doc_type, title=title,
                file_path=str(path.relative_to(_CHATBOT)),
                file_size=len(body), scrape_status="done",
            )
            db._conn.execute(
                "UPDATE documents SET download_status='skipped', scrape_status='done', "
                "file_path=?, title=?, category=?, document_type=? WHERE url=?",
                (str(path.relative_to(_CHATBOT)), title, category, doc_type, url),
            )
            db._conn.commit()
            n_ok += 1
            print(f"  -> {len(body):,} bytes  {path.name}")

        for url, category, doc_type, title in PDF_TARGETS:
            print(f"[pdf ] {url}")
            body = _fetch(url)
            if body is None:
                continue
            name = Path(url.split("?")[0]).name
            path = _PDF_DIR / name
            path.write_bytes(body)
            db.insert_document(
                url, category, doc_type, title=title,
                file_path=str(path.relative_to(_CHATBOT)),
                file_size=len(body), scrape_status="done",
            )
            db._conn.execute(
                "UPDATE documents SET download_status='done', scrape_status='done', "
                "file_path=?, title=?, category=?, document_type=? WHERE url=?",
                (str(path.relative_to(_CHATBOT)), title, category, doc_type, url),
            )
            db._conn.commit()
            n_ok += 1
            print(f"  -> {len(body):,} bytes  {path.name}")

    print(f"\n{n_ok}/{len(HTML_TARGETS) + len(PDF_TARGETS)} targets downloaded and registered.")


if __name__ == "__main__":
    main()
