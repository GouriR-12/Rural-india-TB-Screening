"""Unit tests for Intake Agent (Multilingual Extraction, LLM Mocking, Risk Scoring).

Covers Hindi (Devanagari & Hinglish), Marathi (Devanagari & Marathish), English,
strict null checking for missing fields, follow-up question generation,
calibrated logistic regression risk score computation, and optional live Gemini LLM integration.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from src.agents.intake import (
    extract_intake,
    run_intake,
    compute_risk_score,
    generate_follow_up_questions,
    detect_language,
)
from src.schemas.models import IntakePayload, IntakeResult, GeminiIntakeExtraction


# --- Legacy / Existing Baseline Compatibility Tests ---

def test_intake_english_notes():
    """Verify entity extraction from English ASHA worker notes."""
    payload = IntakePayload(
        case_id="C-001",
        patient_id="P-001",
        patient_name="Anil Kumar",
        age=38,
        gender="M",
        village="Khed",
        asha_id="ASHA-101",
        raw_notes="Patient has cough and fever for 20 days with noticeable weight loss.",
        symptoms=["Cough > 2 weeks"],
        duration_days=20,
        contact_history=True,
    )
    result = run_intake(payload)
    assert isinstance(result, IntakeResult)
    assert result.has_cough is True
    assert result.has_fever is True
    assert result.has_weight_loss is True
    assert result.cough_duration_days == 20
    assert result.contact_history is True
    assert result.risk_score >= 0.70
    assert result.risk_level == "High"


def test_intake_hindi_marathi_notes():
    """Verify entity extraction from Hindi/Marathi vocabulary."""
    payload = IntakePayload(
        case_id="C-002",
        patient_id="P-002",
        patient_name="Pooja Shinde",
        age=29,
        gender="F",
        village="Shirur",
        asha_id="ASHA-102",
        raw_notes="रुग्णाला खोकला (khokla) आणि ताप आहे, रात्री घाम येतो.",
        symptoms=[],
        duration_days=0,
        contact_history=False,
    )
    result = run_intake(payload)
    assert result.has_cough is True
    assert result.has_fever is True
    assert result.has_night_sweats is True
    assert result.language_detected in ["mr", "hi", "mr/hi"]


# --- 10 Dedicated Multilingual & Strict Extraction Test Cases ---

def test_case_1_english_full_high_risk():
    """Case 1: English full clinical presentation with alarm symptoms and risk factors."""
    notes = (
        "Patient Ramesh Patil, 45-year-old male from Junnar. Severe persistent cough for 21 days "
        "with evening fever, noticeable weight loss, and night sweats. Also observed blood in sputum (hemoptysis). "
        "History of smoking bidi and close contact with a known active TB patient."
    )
    result = extract_intake(notes)
    assert result.age == 45
    assert result.gender == "M"
    assert result.village == "Junnar"
    assert result.has_cough is True
    assert result.cough_duration_days >= 14
    assert result.has_fever is True
    assert result.has_weight_loss is True
    assert result.has_night_sweats is True
    assert result.has_hemoptysis is True
    assert result.contact_history is True
    assert result.risk_factors.get("smoker") is True
    assert result.risk_score >= 0.85
    assert result.risk_level == "High"
    # Ensure contributing factors explain the high score
    assert any("Hemoptysis" in f for f in result.top_contributing_factors)
    assert any("Persistent Cough" in f for f in result.top_contributing_factors)


def test_case_2_hindi_devanagari_high_risk():
    """Case 2: Hindi Devanagari text with classical pulmonary TB symptoms."""
    notes = (
        "मरीज का नाम सुरेश कुमार, उम्र 52 वर्ष, पुरुष, गांव शिरूर। पिछले 3 हफ्ते (21 दिन) से "
        "लगातार खांसी है, शाम को तेज बुखार आता है, वजन काफी कम हुआ है और खांसी में खून (रक्त) आया है। "
        "घर में टीबी मरीज का संपर्क रहा है।"
    )
    result = extract_intake(notes)
    assert result.age == 52
    assert result.gender == "M"
    assert result.village == "शिरूर" or result.village == "Shirur" or result.village is not None
    assert result.has_cough is True
    assert result.cough_duration_days == 21
    assert result.has_fever is True
    assert result.has_weight_loss is True
    assert result.has_hemoptysis is True
    assert result.contact_history is True
    assert result.language_detected in ["hi", "mr"]
    assert result.risk_score >= 0.85
    assert result.risk_level == "High"


def test_case_3_marathi_devanagari_high_risk():
    """Case 3: Marathi Devanagari text from rural primary healthcare camp."""
    notes = (
        "रुग्णाचे नाव दत्तात्रय, वय 48 वर्षे, पुरुष, गाव सासवड. तीन आठवड्यांपासून खोकला (21 दिवस), "
        "संध्याकाळी ताप, वजन कमी झाले आणि खोकल्यातून रक्त पडत आहे. घरात टीबी रुग्णाचा संपर्क होता."
    )
    result = extract_intake(notes)
    assert result.age == 48
    assert result.gender == "M"
    assert result.has_cough is True
    assert result.cough_duration_days == 21
    assert result.has_fever is True
    assert result.has_weight_loss is True
    assert result.has_hemoptysis is True
    assert result.contact_history is True
    assert result.language_detected in ["mr", "hi"]
    assert result.risk_score >= 0.85
    assert result.risk_level == "High"


def test_case_4_hinglish_high_risk():
    """Case 4: Transliterated Hindi (Hinglish) typed by an ASHA worker on a mobile device."""
    notes = (
        "Patient Mohan 38 saal male village Baramati. Khansi 3 hafte se hai, "
        "bukhar aur vajan kam ho raha hai, khansi me khoon aaya. TB contact history positive hai."
    )
    result = extract_intake(notes)
    assert result.age == 38
    assert result.gender == "M"
    assert result.village == "Baramati"
    assert result.has_cough is True
    assert result.cough_duration_days == 21
    assert result.has_fever is True
    assert result.has_weight_loss is True
    assert result.has_hemoptysis is True
    assert result.contact_history is True
    assert result.language_detected in ["hinglish", "hi", "en"]
    assert result.risk_score >= 0.85
    assert result.risk_level == "High"


def test_case_5_marathish_high_risk():
    """Case 5: Transliterated Marathi (Marathish) with regional terminology."""
    notes = (
        "Rukna Ananda 55 varsh male gav Khed. Teen athavadyapasun khokla ahe, "
        "sandhyakali taap yeto, vajan kami jhale ahe, gham yeto. Bidi odhto."
    )
    result = extract_intake(notes)
    assert result.age == 55
    assert result.gender == "M"
    assert result.village == "Khed"
    assert result.has_cough is True
    assert result.cough_duration_days == 21
    assert result.has_fever is True
    assert result.has_weight_loss is True
    assert result.has_night_sweats is True
    assert result.risk_factors.get("smoker") is True
    assert result.language_detected in ["marathish", "mr", "en"]
    assert result.risk_score >= 0.70
    assert result.risk_level == "High"


def test_case_6_english_missing_fields_strict_null():
    """Case 6: Incomplete English notes. Strict rule: Never guess values, return null and follow-up questions."""
    notes = "Patient has mild dry cough for 3 days, no fever, no weight loss."
    result = extract_intake(notes)
    # Demographics and risk factors were NOT specified -> MUST be null
    assert result.age is None
    assert result.gender is None
    assert result.village is None
    assert result.contact_history is False
    assert result.risk_factors.get("tb_contact") is None
    # Verify missing fields and generated follow-up questions
    assert "age" in result.missing_fields
    assert "sex" in result.missing_fields
    assert "village" in result.missing_fields
    assert len(result.follow_up_questions) >= 3
    # Low risk score because no chronic cough or alarm signs
    assert result.risk_score < 0.35
    assert result.risk_level == "Low"


def test_case_7_hindi_devanagari_incomplete_with_followups():
    """Case 7: Brief Hindi input lacking demographics and contact history."""
    notes = "मरीज को 5 दिन से हल्की खांसी और जुकाम है।"
    result = extract_intake(notes)
    assert result.age is None
    assert result.gender is None
    assert result.village is None
    assert result.has_cough is True
    assert result.cough_duration_days == 5
    assert result.has_hemoptysis is False
    assert "age" in result.missing_fields
    assert "village" in result.missing_fields
    assert len(result.follow_up_questions) > 0
    assert result.risk_score < 0.35
    assert result.risk_level == "Low"


def test_case_8_marathi_missing_duration_and_village():
    """Case 8: Marathi input with cough and fever but unspecified duration and missing village."""
    notes = "रुग्ण स्त्री, वय 35 वर्षे. खोकला आहे आणि ताप आहे पण किती दिवस ते माहित नाही."
    result = extract_intake(notes)
    assert result.age == 35
    assert result.gender == "F"
    assert result.village is None
    assert result.has_cough is True
    assert result.cough_duration_days == 0 or result.cough_duration_days is None
    assert "village" in result.missing_fields
    assert "cough_duration_days" in result.missing_fields
    # Verify targeted follow-up question for cough duration
    assert any("दिवस" in q or "days" in q.lower() or "week" in q.lower() for q in result.follow_up_questions)


def test_case_9_hinglish_missing_risk_factors():
    """Case 9: Hinglish input with symptoms but missing comorbidity and contact risk factors."""
    notes = "Sunita 30 female village Paithan, 2 week se cough aur fever hai."
    result = extract_intake(notes)
    assert result.age == 30
    assert result.gender == "F"
    assert result.village == "Paithan"
    assert result.has_cough is True
    assert result.cough_duration_days == 14
    assert result.has_fever is True
    # Risk factors were not mentioned -> must generate questions
    assert "tb_contact" in result.missing_fields
    assert any("contact" in q.lower() or "संपर्क" in q for q in result.follow_up_questions)
    assert any("diabetes" in q.lower() or "मधुमेह" in q for q in result.follow_up_questions)


def test_case_10_english_low_risk_normal_allergy():
    """Case 10: Clear non-TB presentation (acute rhinitis/allergy) yielding low risk score."""
    notes = (
        "Pooja, 24 years female, village Shirdi. Sneezing and mild runny nose for 2 days. "
        "No chronic cough, no fever, no weight loss, no hemoptysis, non-smoker, no TB contact."
    )
    result = extract_intake(notes)
    assert result.age == 24
    assert result.gender == "F"
    assert result.village == "Shirdi"
    assert result.has_hemoptysis is False
    assert result.has_fever is False
    assert result.contact_history is False
    assert result.risk_score < 0.25
    assert result.risk_level == "Low"


# --- LLM Mocking and Live Gemini Integration Tests ---

def test_mocked_gemini_llm_structured_extraction():
    """Verify that Intake Agent integrates with Gemini LLM response_schema via mocking."""
    mock_gemini_extraction = GeminiIntakeExtraction(
        patient_name="Baban Rao",
        age=60,
        sex="M",
        village="Koregaon",
        has_cough=True,
        cough_duration_days=28,
        has_fever=True,
        fever_duration_days=14,
        has_weight_loss=True,
        has_night_sweats=True,
        has_hemoptysis=True,
        tb_contact=True,
        diabetes=True,
        smoker=True,
        hiv=False,
        other_symptoms=["Chest pain"],
        language_detected="mr",
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = mock_gemini_extraction
    mock_client.models.generate_content.return_value = mock_response

    input_text = "Some complex dialect Marathi audio transcript"
    result = extract_intake(input_text, client=mock_client)

    # Verify mock was called with structured configuration
    mock_client.models.generate_content.assert_called_once()
    assert result.patient_name == "Baban Rao"
    assert result.age == 60
    assert result.gender == "M"
    assert result.village == "Koregaon"
    assert result.has_cough is True
    assert result.cough_duration_days == 28
    assert result.has_hemoptysis is True
    assert result.risk_factors.get("diabetes") is True
    assert result.risk_score >= 0.90
    assert result.risk_level == "High"
    assert any("Hemoptysis" in f for f in result.top_contributing_factors)
    assert any("Persistent Cough" in f for f in result.top_contributing_factors)


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Live Gemini API test requires GEMINI_API_KEY in .env")
def test_live_gemini_llm_intake():
    """Optional live test against Google Gemini API when GEMINI_API_KEY is configured."""
    notes = "मरीज रमेश, 42 वर्ष, पुरुष, गांव जुन्नर। 3 हफ्ते से खांसी और बुखार है।"
    result = extract_intake(notes)
    assert isinstance(result, IntakeResult)
    assert result.age == 42
    assert result.gender == "M"
    assert result.has_cough is True
    assert result.cough_duration_days >= 14
    assert result.risk_score > 0.0
