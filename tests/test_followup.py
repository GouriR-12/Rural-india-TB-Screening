"""Unit tests for Follow-up Agent."""

import pytest
from src.agents.followup import run_followup
from src.schemas.models import DecisionResult, DoctorReviewResult, FollowupResult


def test_followup_approved():
    decision = DecisionResult(
        tb_risk_score=0.85,
        risk_category="High",
        preliminary_recommendation="GeneXpert indicated",
        clinical_summary="High risk presentation",
        requires_doctor_review=True,
    )
    doctor_review = DoctorReviewResult(
        approved=True,
        doctor_id="DR-007",
        doctor_notes="Approved for ATT",
        final_diagnosis="Pulmonary TB",
        prescription="4FDC",
    )
    result = run_followup(
        patient_id="P-001",
        village="Ralegan",
        asha_id="ASHA-10",
        decision=decision,
        doctor_review=doctor_review,
    )
    assert isinstance(result, FollowupResult)
    assert result.asha_assigned == "ASHA-10"
    assert result.dropout_risk_level in ["High", "Medium", "Low"]


def test_followup_non_approved_raises_error():
    """Rule 1: No case can reach Follow-up without doctor approval."""
    decision = DecisionResult(
        tb_risk_score=0.20,
        risk_category="Low",
        preliminary_recommendation="Low probability",
        clinical_summary="Mild signs",
        requires_doctor_review=True,
    )
    doctor_review = DoctorReviewResult(
        approved=False,
        doctor_id="DR-007",
        doctor_notes="Case rejected",
        final_diagnosis="Common Cold",
    )
    with pytest.raises(ValueError, match="Cannot initiate follow-up for a non-approved case"):
        run_followup(
            patient_id="P-002",
            village="Ralegan",
            asha_id="ASHA-10",
            decision=decision,
            doctor_review=doctor_review,
        )
