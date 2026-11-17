"""Unit tests for Decision Agent."""

from unittest.mock import MagicMock
import pytest

from src.agents.decision import (
    build_briefing_prompt,
    compute_deterministic_risk_score,
    generate_deterministic_briefing,
    generate_doctor_briefing,
    order_lab_test,
    run_decision,
)
from src.schemas.models import (
    DecisionResult,
    DoctorBriefing,
    ImagingResult,
    IntakeResult,
    LabOrderPayload,
    LabOrderResult,
)


def test_order_lab_test():
    payload = LabOrderPayload(case_id="C-001-9999", patient_id="P-001", test_type="CBNAAT / GeneXpert")
    result = order_lab_test(payload)
    assert isinstance(result, LabOrderResult)
    assert result.order_status == "Ordered"
    assert "CBNAAT" in result.test_type


def test_decision_synthesis_high_risk():
    intake = IntakeResult(
        extracted_symptoms=["Cough > 2 weeks", "Fever", "Weight Loss", "Hemoptysis"],
        has_cough=True,
        cough_duration_days=21,
        has_fever=True,
        has_weight_loss=True,
        has_hemoptysis=True,
        contact_history=True,
    )
    imaging = ImagingResult(
        abnormality_detected=True,
        tb_probability=0.88,
        findings="Apical cavitary lesions."
    )
    decision = run_decision(
        case_id="C-001",
        patient_name="Suresh Rao",
        age=52,
        intake=intake,
        imaging=imaging,
    )
    assert isinstance(decision, DecisionResult)
    assert decision.risk_category == "High"
    assert decision.tb_risk_score > 0.65
    assert decision.requires_doctor_review is True
    assert decision.briefing is not None
    assert isinstance(decision.briefing, DoctorBriefing)


def test_decision_gemini_briefing_success():
    """Test successful Gemini structured output generation of DoctorBriefing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    expected_briefing = DoctorBriefing(
        key_findings=[
            "Patient: Rajesh Kumar (Age: 48y, Sex: M, Village: Rampur)",
            "Cough duration: 24 days",
            "Reported symptoms: fever, weight loss",
            "CXR: Apical infiltrates in right upper lobe (Probability: 0.82)"
        ],
        evidence_for=[
            "Persistent cough > 3 weeks exceeding NTEP 14-day threshold",
            "Evening fever and constitutional weight loss reported",
            "Positive CXR parenchymal opacity"
        ],
        evidence_against_or_uncertainty=[
            "Hemoptysis not reported",
            "No prior known household contact"
        ],
        suggested_next_tests=[
            "Upfront CBNAAT / GeneXpert molecular testing for MTB and RMP resistance",
            "Two sputum smear examinations (spot and morning)",
            "Immediate Medical Officer clinical evaluation"
        ],
        flags=[
            "High clinical suspicion requiring prompt diagnostic confirmation"
        ]
    )
    mock_response.parsed = expected_briefing
    mock_client.models.generate_content.return_value = mock_response

    intake = IntakeResult(
        extracted_symptoms=["Cough > 2 weeks", "Fever", "Weight Loss"],
        has_cough=True,
        cough_duration_days=24,
        has_fever=True,
        has_weight_loss=True,
    )
    imaging = ImagingResult(
        abnormality_detected=True,
        tb_probability=0.82,
        findings="Apical infiltrates in right upper lobe.",
        model_provenance="real_public_data"
    )

    decision = run_decision(
        case_id="C-TEST-001",
        patient_name="Rajesh Kumar",
        age=48,
        gender="M",
        village="Rampur",
        intake=intake,
        imaging=imaging,
        client=mock_client
    )

    assert mock_client.models.generate_content.call_count == 1
    assert decision.briefing == expected_briefing
    assert "Rajesh Kumar" in decision.briefing.key_findings[0]
    assert len(decision.briefing.evidence_for) == 3
    assert "CBNAAT" in decision.briefing.suggested_next_tests[0]

    # Verify markdown summary formatting
    assert "### Key Findings" in decision.clinical_summary
    assert "### Evidence For Presumptive TB" in decision.clinical_summary
    assert "### Evidence Against / Uncertainty" in decision.clinical_summary
    assert "### Suggested Next Tests" in decision.clinical_summary
    assert "### Clinical Flags" in decision.clinical_summary


def test_decision_gemini_malformed_json_fallback():
    """Test that malformed JSON triggers at most 1 retry and then falls back to deterministic briefing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = "{ malformed_json: true, unexpected_token "
    mock_client.models.generate_content.return_value = mock_response

    intake = IntakeResult(
        extracted_symptoms=["Cough > 2 weeks", "Fever"],
        has_cough=True,
        cough_duration_days=18,
        has_fever=True,
    )
    imaging = ImagingResult(
        abnormality_detected=True,
        tb_probability=0.70,
        findings="Bilateral infiltrates",
        model_provenance="synthetic_demo"
    )

    decision = run_decision(
        case_id="C-TEST-002",
        patient_name="Sunita Devi",
        age=35,
        gender="F",
        village="Chandanpur",
        intake=intake,
        imaging=imaging,
        client=mock_client
    )

    # Exactly 2 calls: initial attempt + 1 retry max
    assert mock_client.models.generate_content.call_count == 2

    # Verify fallback briefing was returned successfully
    assert isinstance(decision.briefing, DoctorBriefing)
    assert any("Sunita Devi" in kf for kf in decision.briefing.key_findings)
    assert any("18 days" in kf or "18" in kf for kf in decision.briefing.key_findings)
    assert any("synthetic_demo" in flag for flag in decision.briefing.flags)
    assert "### Key Findings" in decision.clinical_summary


def test_decision_gemini_no_key_fallback(monkeypatch):
    """Test that absent API key immediately falls back to deterministic briefing with zero LLM calls."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    intake = IntakeResult(
        extracted_symptoms=["Cough > 2 weeks", "Hemoptysis"],
        has_cough=True,
        cough_duration_days=20,
        has_hemoptysis=True,
        contact_history=True
    )
    imaging = ImagingResult(
        abnormality_detected=True,
        tb_probability=0.85,
        findings="Cavitation in left upper zone",
        model_provenance="synthetic_demo"
    )

    decision = run_decision(
        case_id="C-TEST-003",
        patient_name="Anil Kumar",
        age=40,
        gender="M",
        village="Shivaji Nagar",
        intake=intake,
        imaging=imaging,
        client=None
    )

    assert isinstance(decision.briefing, DoctorBriefing)
    assert len(decision.briefing.key_findings) > 0
    assert any("Anil Kumar" in kf for kf in decision.briefing.key_findings)
    assert any("Hemoptysis" in flag or "hemoptysis" in flag.lower() for flag in decision.briefing.flags)
    assert any("synthetic_demo" in flag for flag in decision.briefing.flags)
    assert "### Key Findings" in decision.clinical_summary


def test_decision_prompt_contains_no_invented_fields():
    """Verify prompt only contains input facts, uses 'not available' for missing fields, forbids diagnosis, and flags synthetic demo."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = DoctorBriefing(
        key_findings=["Test finding"],
        evidence_for=["Test evidence"],
        evidence_against_or_uncertainty=["Test uncertainty"],
        suggested_next_tests=["Test next steps"],
        flags=[]
    )
    mock_client.models.generate_content.return_value = mock_response

    # Missing age, village, gender; synthetic_demo imaging provenance
    intake = IntakeResult(
        extracted_symptoms=["Cough > 2 weeks"],
        has_cough=True,
        cough_duration_days=15,
        has_fever=False,
        has_weight_loss=False,
        has_hemoptysis=False,
        contact_history=False,
        age=None,
        village=None,
        gender=None
    )
    imaging = ImagingResult(
        abnormality_detected=True,
        tb_probability=0.72,
        findings="Reticulonodular opacities",
        model_provenance="synthetic_demo"
    )

    run_decision(
        case_id="C-TEST-004",
        patient_name="Anonymous",
        age=None,
        gender=None,
        village=None,
        intake=intake,
        imaging=imaging,
        client=mock_client
    )

    assert mock_client.models.generate_content.call_count == 1
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    prompt = call_kwargs.get("contents", "")

    # Check missing fields say "not available"
    assert "Age: not available" in prompt
    assert "Biological Sex: not available" in prompt
    assert "Village / Sub-center: not available" in prompt
    assert "Patient Name: not available" in prompt

    # Check constraint: never state a diagnosis
    assert "NEVER STATE A DIAGNOSIS" in prompt
    assert "USE ONLY FACTS PRESENT IN THE INPUTS" in prompt
    assert 'Explicitly state "not available" for any unstated or missing attribute' in prompt

    # Check synthetic_demo imaging provenance requirement in prompt
    assert "synthetic_demo" in prompt
    assert 'If imaging provenance is "synthetic_demo", you MUST explicitly mention in both flags and evidence_against_or_uncertainty' in prompt
