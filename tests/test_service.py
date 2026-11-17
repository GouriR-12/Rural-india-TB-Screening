"""Tests for UI pure Python service layer (src/ui/service.py).

Validates:
1. Save / Approve flow: submit_case -> PENDING_REVIEW -> review_case(APPROVED) -> Follow-up stored -> build_final_report.
2. Follow-up refusal: Follow-up agent and run_followup_for_case refuse any non-approved cases (PENDING_REVIEW or REJECTED).
3. Gemini fallback / llm_mode: llm_mode returns 'live' or 'fallback'.
4. Demo data seeding: seed_demo_data creates synthetic cases in SQLite cases table.
"""

import json
import os
import sqlite3
import pytest

from src.agents.followup import run_followup
from src.schemas.models import DecisionResult, DoctorReviewResult
from src.ui.service import (
    build_final_report,
    get_case,
    init_db,
    list_cases,
    llm_mode,
    review_case,
    run_followup_for_case,
    seed_demo_data,
    set_llm_mode,
    submit_case,
)


def test_save_approve_flow(tmp_path):
    """Test case intake submission, persistence as PENDING_REVIEW, doctor approval, and final report generation."""
    db_file = str(tmp_path / "test_tb_screening.db")

    form_data = {
        "case_id": "CASE-TEST-APPROVE-001",
        "patient_id": "PAT-TEST-001",
        "patient_name": "Ramesh Patil",
        "age": 45,
        "gender": "M",
        "village": "Junnar",
        "asha_id": "ASHA-PUNE-01",
        "raw_notes": "Patient has severe khansi for 3 weeks, evening bukhar, and vajan kami.",
        "symptoms": ["Cough > 2 weeks", "Fever", "Weight Loss"],
        "duration_days": 21,
        "contact_history": True,
    }
    cxr_image = "data/sample_cxr_abnormal.png"

    # Step 1: Submit case -> runs Intake -> Imaging -> Decision, saves with PENDING_REVIEW
    case_id = submit_case(form_data, cxr_path=cxr_image, db_path=db_file)
    assert case_id == "CASE-TEST-APPROVE-001"

    # Verify database state after submit
    case_row = get_case(case_id, db_path=db_file)
    assert case_row is not None
    assert case_row["id"] == case_id
    assert case_row["village"] == "Junnar"
    assert case_row["status"] == "PENDING_REVIEW"
    assert case_row["doctor_comment"] is None
    assert case_row["reviewed_at"] is None

    state = json.loads(case_row["state_json"])
    assert state["status"] == "PENDING_REVIEW"
    assert state["intake_result"] is not None
    assert state["imaging_result"] is not None
    assert state["decision_result"] is not None
    assert state["followup_result"] is None
    assert state["doctor_review"] is None

    # Step 2: Doctor reviews and approves case
    comment = "Confirmed active pulmonary tuberculosis on clinical and radiographic correlation. Initiating DOTS."
    updated_case = review_case(
        case_id=case_id,
        decision="APPROVED",
        comment=comment,
        doctor_id="DR-PHC-DESHMUKH",
        db_path=db_file,
    )

    assert updated_case["status"] == "APPROVED"
    assert updated_case["doctor_comment"] == comment
    assert updated_case["reviewed_at"] is not None

    updated_state = json.loads(updated_case["state_json"])
    assert updated_state["status"] == "APPROVED"
    assert updated_state["doctor_review"] is not None
    assert updated_state["doctor_review"]["approved"] is True
    assert updated_state["doctor_review"]["doctor_id"] == "DR-PHC-DESHMUKH"
    assert updated_state["followup_result"] is not None
    assert updated_state["followup_result"]["asha_assigned"] == "ASHA-PUNE-01"
    assert "Nikshay" in updated_state["followup_result"]["adherence_plan"]

    # Step 3: Build final report
    report = build_final_report(case_id, db_path=db_file)
    assert isinstance(report, str)

    # Validate report components required by specification
    assert "Ramesh Patil" in report  # patient details
    assert "Junnar" in report  # patient details (village)
    assert "Cough" in report  # symptoms
    assert "Model Provenance" in report  # imaging + provenance
    assert updated_state["imaging_result"]["model_provenance"] in report
    assert "AI Clinical Summary & Risk Synthesis" in report  # AI summary
    assert "APPROVED" in report  # doctor decision
    assert comment in report  # doctor comment
    assert updated_case["reviewed_at"] in report  # doctor timestamp
    assert "Follow-up & Treatment Adherence Plan" in report  # follow-up plan
    assert "Nikshay" in report
    assert "Research prototype, not clinically validated, not for real diagnosis." in report  # disclaimer


def test_followup_refusing_non_approved_cases(tmp_path):
    """Test that Follow-up strictly refuses to run for non-approved cases (PENDING_REVIEW and REJECTED)."""
    db_file = str(tmp_path / "test_tb_refusal.db")

    form_data = {
        "case_id": "CASE-TEST-REJECT-002",
        "patient_id": "PAT-TEST-002",
        "patient_name": "Sunita Devi",
        "age": 28,
        "gender": "F",
        "village": "Pargaon",
        "asha_id": "ASHA-PUNE-05",
        "raw_notes": "Mild allergic cough for 3 days, no fever, no hemoptysis.",
        "symptoms": ["Mild Cough"],
        "duration_days": 3,
        "contact_history": False,
    }
    cxr_image = "data/sample_cxr_normal.png"

    # Step 1: Submit case -> status is PENDING_REVIEW
    case_id = submit_case(form_data, cxr_path=cxr_image, db_path=db_file)
    case_row = get_case(case_id, db_path=db_file)
    assert case_row["status"] == "PENDING_REVIEW"

    # Rule 1 / Specification: Follow-up must raise an error if status != APPROVED
    with pytest.raises(ValueError, match="Cannot initiate follow-up for a non-approved case"):
        run_followup_for_case(case_id, db_path=db_file)

    # Step 2: Doctor rejects case
    rejection_comment = "Seasonal allergic bronchitis. CXR clear. Non-TB illness."
    rejected_case = review_case(
        case_id=case_id,
        decision="REJECTED",
        comment=rejection_comment,
        doctor_id="DR-PHC-DESHMUKH",
        db_path=db_file,
    )

    assert rejected_case["status"] == "REJECTED"
    assert rejected_case["doctor_comment"] == rejection_comment
    assert rejected_case["reviewed_at"] is not None

    rejected_state = json.loads(rejected_case["state_json"])
    assert rejected_state["status"] == "REJECTED"
    assert rejected_state["doctor_review"]["approved"] is False
    # Ensure Follow-up was NOT run on rejection
    assert rejected_state["followup_result"] is None

    # Verification: run_followup_for_case must still raise an error because status is REJECTED
    with pytest.raises(ValueError, match="Cannot initiate follow-up for a non-approved case"):
        run_followup_for_case(case_id, db_path=db_file)

    # Direct agent verification: run_followup itself refuses non-approved DoctorReviewResult
    decision_obj = DecisionResult(**rejected_state["decision_result"])
    doctor_review_obj = DoctorReviewResult(**rejected_state["doctor_review"])
    with pytest.raises(ValueError, match="Cannot initiate follow-up for a non-approved case"):
        run_followup(
            patient_id="PAT-TEST-002",
            village="Pargaon",
            asha_id="ASHA-PUNE-05",
            decision=decision_obj,
            doctor_review=doctor_review_obj,
        )

    # Step 3: Verify final report for rejected case
    report = build_final_report(case_id, db_path=db_file)
    assert "REJECTED" in report
    assert rejection_comment in report
    assert "No follow-up plan scheduled" in report
    assert "Research prototype, not clinically validated, not for real diagnosis." in report


def test_llm_mode_fallback_and_live(monkeypatch):
    """Test llm_mode exposure returning 'live' or 'fallback'."""
    set_llm_mode(None)

    # When GEMINI_API_KEY is empty or missing -> fallback
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert llm_mode() == "fallback"

    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert llm_mode() == "fallback"

    # When GEMINI_API_KEY is provided -> live
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_test_key_123")
    assert llm_mode() == "live"

    # Test explicit override mechanism
    set_llm_mode("fallback")
    assert llm_mode() == "fallback"
    set_llm_mode(None)


def test_seed_demo_data(tmp_path):
    """Test seed_demo_data generates synthetic cases in SQLite cases table."""
    db_file = str(tmp_path / "test_demo_seed.db")
    ids = seed_demo_data(n=50, db_path=db_file)

    assert len(ids) == 50
    assert ids[0] == "CASE-DEMO-0001"
    assert ids[-1] == "CASE-DEMO-0050"

    all_cases = list_cases(db_path=db_file)
    assert len(all_cases) == 50

    # Verify statuses are populated
    statuses = {c["status"] for c in all_cases}
    assert "PENDING_REVIEW" in statuses
    assert "APPROVED" in statuses
    assert "REJECTED" in statuses

    # Verify villages and dates are present
    villages = {c["village"] for c in all_cases}
    assert len(villages) > 1

    # Verify approved cases have follow-up plan in state_json
    approved = [c for c in all_cases if c["status"] == "APPROVED"]
    assert len(approved) > 0
    approved_state = json.loads(approved[0]["state_json"])
    assert approved_state["followup_result"] is not None

    # Verify build_final_report works on a seeded case
    report = build_final_report(ids[0], db_path=db_file)
    assert "Tuberculosis Screening Clinical Report" in report
    assert "Research prototype, not clinically validated, not for real diagnosis." in report

