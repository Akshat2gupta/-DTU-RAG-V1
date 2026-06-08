"""
Tests for ingestion/classifier.py
No file I/O. No pdfplumber. No Scrapy. No network.
"""
import sys
import os

# Allow import as both 'python -m pytest' from dtu-chatbot/
# and 'python ingestion/test_classifier.py' directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ingestion.classifier import classify_page, classify_document

# ---------------------------------------------------------------------------
# Shared sample texts
# ---------------------------------------------------------------------------

SYLLABUS_TEXT = """\
L T P Credits
3 1 0 4
1. CE-301 Introduction to Programming
2. CE-302 Data Structures
3. CE-303 Algorithms
Text Books: Algorithm Design by Kleinberg
Reference Books: Introduction to Algorithms by CLRS
"""

FORM_TEXT = """\
FORM OF APPLICATION
Signature of the Applicant
Roll Number: _______________
"""

CONTACT_TEXT = """\
Dr. John Doe
PROFESSOR
ASSISTANT PROFESSOR Jane Smith
PROFESSOR A. Kumar
Ph.D (IIT Delhi)
Ph.D (IIT Bombay)
Ph.D (IIT Madras)
Designation: Associate Professor
"""

NOTICE_TEXT = """\
Circular No: DTU/2022/001
To: All Students of B.Tech Program
Last Date for submission: 15 March 2022
Dean Academic Affairs
"""

POLICY_TEXT = """\
R. 1(B).22 Attendance Rules
The student shall be required to attend all classes.
Minimum attendance shall be 75%.
The Dean (UG) may grant relaxation.
The COE shall maintain records.
Students must maintain 75% attendance percentage.
"""

# ===========================================================================
# SYLLABUS
# ===========================================================================

def test_syllabus_ltp_course_codes_and_textbooks():
    """LTP + course codes + numbered list + Text Books → ≥3 signals → syllabus."""
    assert classify_page(SYLLABUS_TEXT) == "syllabus"


def test_syllabus_ltp_credit_hours_references():
    """L/T/P + Credit Hours + Reference Books → 3 signals → syllabus."""
    text = "L/T/P: 3/1/0\nCredit Hours: 4\nReference Books: CLRS\n"
    assert classify_page(text) == "syllabus"


def test_syllabus_contact_hours_numbered_list_name_of_books():
    """Contact Hours + 3 numbered lines + Name of Books → 3 signals → syllabus."""
    text = (
        "Contact Hours: 4\n"
        "1. Applied Mathematics\n"
        "2. Advanced Calculus\n"
        "3. Linear Algebra\n"
        "Name of Books: Kreyszig\n"
    )
    assert classify_page(text) == "syllabus"


# ===========================================================================
# FORM
# ===========================================================================

def test_form_application_with_roll_number():
    """FORM OF APPLICATION + Signature of + Roll Number → 3 signals → form."""
    assert classify_page(FORM_TEXT) == "form"


def test_form_project_title_with_signature():
    """Proposed Project Title + Signature of → 2 signals → form."""
    text = "Proposed Project Title:\nSignature of Supervisor:\n"
    assert classify_page(text) == "form"


def test_form_two_dates_with_signature():
    """Signature of + Date: ×2 → 2 signals → form."""
    text = "Signature of Dean\nDate:\nDate:\n"
    assert classify_page(text) == "form"


# ===========================================================================
# CONTACT
# ===========================================================================

def test_contact_faculty_list():
    """PROFESSOR ×3 + Ph.D ×3 + Designation → 3 signals → contact."""
    assert classify_page(CONTACT_TEXT) == "contact"


def test_contact_dense_phd_and_designation():
    """PROFESSOR ×3 + Ph.D ×3 + Designation → 3 signals → contact."""
    text = (
        "PROFESSOR\nPROFESSOR\nPROFESSOR\n"
        "Ph.D (IIT)\nPh.D (NIT)\nPh.D (DTU)\n"
        "Designation: Associate Prof\n"
    )
    assert classify_page(text) == "contact"


def test_contact_assistant_professor_phd_professor():
    """PROFESSOR ×3 + ASSISTANT PROFESSOR ×2 + Ph.D ×3 → 3 signals → contact."""
    text = (
        "PROFESSOR PROFESSOR PROFESSOR\n"
        "ASSISTANT PROFESSOR Alpha\nASSISTANT PROFESSOR Beta\n"
        "Ph.D\nPh.D\nPh.D\n"
    )
    assert classify_page(text) == "contact"


# ===========================================================================
# NOTICE
# ===========================================================================

def test_notice_circular_all_students_last_date():
    """Circular No + All Students + Last Date + Dean Academic → 4 signals → notice."""
    assert classify_page(NOTICE_TEXT) == "notice"


def test_notice_fno_and_all_students():
    """F.No + All Students → 2 signals → notice."""
    text = "F.No DTU/123\nAll Students are hereby informed\n"
    assert classify_page(text) == "notice"


def test_notice_ref_and_due_date():
    """Ref:- + due date → 2 signals → notice."""
    text = "Ref:- DTU/2023/456\nPlease submit before the due date.\n"
    assert classify_page(text) == "notice"


# ===========================================================================
# POLICY
# ===========================================================================

def test_policy_r1b_dean_attendance():
    """R. 1(B) + Dean (UG)/COE + attendance+75% → 3 signals → policy."""
    assert classify_page(POLICY_TEXT) == "policy"


def test_policy_cgpa_and_hod_coe():
    """CGPA/grade point + HOD/COE → 2 signals → policy."""
    text = (
        "The CGPA shall be calculated as weighted average.\n"
        "Grade point shall be assigned to each course.\n"
        "HOD shall forward the result to COE.\n"
    )
    assert classify_page(text) == "policy"


def test_policy_sgpa_dean_pg_grade_point():
    """SGPA/CGPA/grade point + Dean (PG) → 2 signals → policy."""
    text = (
        "The SGPA and CGPA shall be calculated.\n"
        "Dean (PG) shall be authorized.\n"
        "Grade point shall be 10.0 for O grade.\n"
    )
    assert classify_page(text) == "policy"


# ===========================================================================
# SKIP
# ===========================================================================

def test_skip_empty_string():
    assert classify_page("") == "skip"


def test_skip_generic_text():
    text = "This is some random text that matches no classification signals."
    assert classify_page(text) == "skip"


# ===========================================================================
# classify_document
# ===========================================================================

def test_document_policy_majority():
    """5 policy pages + 1 syllabus → policy_frac 0.83 > 0.40 → policy."""
    pages = [{"page_number": i, "text": POLICY_TEXT} for i in range(1, 6)]
    pages.append({"page_number": 6, "text": SYLLABUS_TEXT})
    assert classify_document(pages) == "policy"


def test_document_all_skip():
    """All pages unclassifiable → skip."""
    pages = [{"page_number": i, "text": "random unclassifiable text"} for i in range(1, 4)]
    assert classify_document(pages) == "skip"


def test_document_syllabus_majority():
    """4 syllabus pages + 1 skip → syllabus_frac 1.0 > 0.30 → syllabus."""
    pages = [{"page_number": i, "text": SYLLABUS_TEXT} for i in range(1, 5)]
    pages.append({"page_number": 5, "text": "random text"})
    assert classify_document(pages) == "syllabus"
