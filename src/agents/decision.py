"""Decision Agent Module.

Synthesizes clinical intake symptoms, laboratory orders, and chest X-ray findings
into an evidence-backed case briefing, preliminary recommendation, and deterministic risk score.

NTEP (National Tuberculosis Elimination Program) Symptom-Screening Logic & Weights:
========================================================================================
Clinical Finding                    Condition                        Score Contribution
----------------------------------------------------------------------------------------
Persistent Cough (>= 2 weeks)       has_cough & cough_days >= 14         +0.35
Mild Cough (< 2 weeks)              has_cough & cough_days < 14          +0.15
Recurrent / Evening Fever           has_fever == True                    +0.15
Unexplained Weight Loss             has_weight_loss == True              +0.15
Hemoptysis (Alarm Sign)             has_hemoptysis == True               +0.25
Drenching Night Sweats              has_night_sweats == True             +0.10
Household / Close Contact History   contact_history == True              +0.20
Chest Radiograph (CXR) Abnormality  abnormality_detected == True         +(tb_prob * 0.40)
----------------------------------------------------------------------------------------
Risk Stratification Categories:
  Score >= 0.65 : High Risk     -> Prioritize for CBNAAT / GeneXpert molecular confirmation
  0.35 <= Score < 0.65 : Moderate -> Sputum microscopy / clinical evaluation
  Score < 0.35 : Low Risk      -> Symptomatic monitoring, re-screen if cough persists
========================================================================================

Uses Google GenAI SDK (Gemini) with strict Pydantic structured output (DoctorBriefing)
to generate narrative case briefings for the Medical Officer.
Supports graceful deterministic offline fallback with a maximum of one retry on error.

DISCLAIMER:
Research prototype, not clinically validated, not for real diagnosis.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional

from src.schemas.models import (
    DecisionResult,
    DoctorBriefing,
    ImagingResult,
    IntakeResult,
    LabOrderPayload,
    LabOrderResult,
)


def get_gemini_client() -> Optional[Any]:
    """Retrieve an authenticated Google GenAI client if credentials exist."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def order_lab_test(payload: LabOrderPayload) -> LabOrderResult:
    """Generate and dispatch lab investigation order (parallel branch step).

    Args:
        payload: Lab request information.

    Returns:
        LabOrderResult: Details of dispatched sputum/molecular test.
    """
    order_id = f"LAB-{payload.case_id[-6:]}" if len(payload.case_id) >= 6 else f"LAB-{payload.case_id}"
    return LabOrderResult(
        order_id=order_id,
        test_type=payload.test_type,
        order_status="Ordered",
        order_timestamp=datetime.now(timezone.utc).isoformat(),
        target_facility="Designated Microscopy Centre (DMC) / PHC"
    )


def compute_deterministic_risk_score(
    intake: IntakeResult,
    imaging: Optional[ImagingResult] = None
) -> float:
    """Calculate transparent deterministic composite TB risk score based on NTEP symptom weights."""
    score = 0.0

    # Symptom weightings based on NTEP clinical screening
    if intake.has_cough and intake.cough_duration_days >= 14:
        score += 0.35
    elif intake.has_cough:
        score += 0.15

    if intake.has_fever:
        score += 0.15
    if intake.has_weight_loss:
        score += 0.15
    if intake.has_hemoptysis:
        score += 0.25
    if intake.has_night_sweats:
        score += 0.10
    if intake.contact_history:
        score += 0.20

    # Imaging weighting
    if imaging and imaging.abnormality_detected:
        score += (imaging.tb_probability * 0.40)

    return min(round(score, 4), 0.98)


def get_risk_category_and_recommendation(score: float) -> tuple[str, str]:
    """Determine clinical triage category and preliminary recommendation from risk score."""
    if score >= 0.65:
        category = "High"
        recommendation = "High suspicion of pulmonary TB. Prioritize for GeneXpert/CBNAAT and immediate Medical Officer evaluation."
    elif score >= 0.35:
        category = "Moderate"
        recommendation = "Moderate risk. Complete sputum smear microscopy and evaluate response to broad-spectrum antibiotic course."
    else:
        category = "Low"
        recommendation = "Low clinical probability. Monitor symptoms; re-screen if cough persists beyond 2 weeks."
    return category, recommendation


def build_briefing_prompt(
    case_id: str,
    patient_name: str,
    age: Optional[int],
    gender: Optional[str],
    village: Optional[str],
    intake: IntakeResult,
    imaging: Optional[ImagingResult],
    lab: Optional[LabOrderResult],
    risk_score: float,
    risk_category: str
) -> str:
    """Construct structured briefing prompt strictly using present facts."""
    age_str = f"{age} years" if age is not None else "not available"
    gender_str = gender if gender else "not available"
    village_str = village if village else (intake.village if intake.village else "not available")
    name_str = patient_name if patient_name and patient_name != "Anonymous" else "not available"

    cough_str = f"Present ({intake.cough_duration_days} days duration)" if intake.has_cough else "not reported"
    fever_str = "Present" if intake.has_fever else "not reported"
    weight_loss_str = "Present" if intake.has_weight_loss else "not reported"
    night_sweats_str = "Present" if intake.has_night_sweats else "not reported"
    hemoptysis_str = "Present (Blood in sputum/cough)" if intake.has_hemoptysis else "not reported"
    contact_str = "Positive household/close contact" if intake.contact_history else "No known TB contact reported"

    rf = intake.risk_factors or {}
    smoker_str = "Reported" if rf.get("smoker") is True else ("Denied" if rf.get("smoker") is False else "not available")
    diabetes_str = "Reported" if rf.get("diabetes") is True else ("Denied" if rf.get("diabetes") is False else "not available")
    hiv_str = "Reported" if rf.get("hiv") is True else ("Denied" if rf.get("hiv") is False else "not available")

    if imaging:
        imaging_prov = imaging.model_provenance
        imaging_prob = f"{imaging.tb_probability:.2f}"
        imaging_abnormal = "Abnormality detected" if imaging.abnormality_detected else "No abnormality detected"
        imaging_findings = imaging.findings if imaging.findings else "not available"
        imaging_note = imaging.confidence_note if imaging.confidence_note else "not available"
    else:
        imaging_prov = "not available"
        imaging_prob = "not available"
        imaging_abnormal = "not available"
        imaging_findings = "not available"
        imaging_note = "not available"

    lab_str = f"{lab.test_type} (Status: {lab.order_status}, Target: {lab.target_facility})" if lab else "not available"

    prompt = f"""You are an expert clinical decision-support AI assisting a primary health center (PHC) Medical Officer in rural India.
Synthesize the frontline screening evidence into a structured case briefing for doctor review.

INPUT CLINICAL DATA:
- Case Identifier: {case_id}
- Patient Name: {name_str}
- Age: {age_str}
- Biological Sex: {gender_str}
- Village / Sub-center: {village_str}
- Reported Symptoms:
  * Cough: {cough_str}
  * Fever: {fever_str}
  * Weight Loss: {weight_loss_str}
  * Night Sweats: {night_sweats_str}
  * Hemoptysis: {hemoptysis_str}
- Epidemiological Contact History: {contact_str}
- Risk Factors / Comorbidities:
  * Smoking: {smoker_str}
  * Diabetes: {diabetes_str}
  * HIV: {hiv_str}
- Chest Radiograph (CXR) Analysis:
  * Status: {imaging_abnormal}
  * Model Provenance: {imaging_prov}
  * TB Probability: {imaging_prob}
  * Radiologic Findings: {imaging_findings}
  * Radiologic Confidence Note: {imaging_note}
- Laboratory Investigation: {lab_str}
- Deterministic Composite Risk Score: {risk_score:.2f} ({risk_category} Risk Category)

STRICT BRIEFING CONSTRAINTS:
1. USE ONLY FACTS PRESENT IN THE INPUTS. Never extrapolate, hallucinate, or assume unstated details.
2. MISSING FIELDS: Explicitly state "not available" for any unstated or missing attribute.
3. NEVER STATE A DIAGNOSIS. Provide observations and risk indices only; Medical Officers maintain authoritative diagnosis. Do not state that the patient has tuberculosis or any definitive disease.
4. IMAGING PROVENANCE: If imaging provenance is "synthetic_demo", you MUST explicitly mention in both flags and evidence_against_or_uncertainty that the radiograph was evaluated by a synthetic demo model lacking clinical validation.
5. Provide structured output with five sections:
   - key_findings: List of factual clinical observations and durations.
   - evidence_for: Findings and risk factors that raise suspicion of active pulmonary tuberculosis.
   - evidence_against_or_uncertainty: Negated symptoms, missing information ("not available"), or imaging uncertainties.
   - suggested_next_tests: Recommended confirmatory laboratory tests (e.g. CBNAAT/GeneXpert, sputum smear) or clinical steps.
   - flags: Urgent safety alerts, alarm signs (e.g. hemoptysis), or model provenance limitations.
"""
    return prompt


def generate_deterministic_briefing(
    patient_name: str,
    age: Optional[int],
    gender: Optional[str],
    village: Optional[str],
    intake: IntakeResult,
    imaging: Optional[ImagingResult],
    lab: Optional[LabOrderResult],
    risk_score: float,
    risk_category: str
) -> DoctorBriefing:
    """Deterministic template fallback when Gemini API key is absent or on error."""
    age_str = f"{age}y" if age is not None else "not available"
    gender_str = gender if gender else "not available"
    village_str = village if village else (intake.village if intake.village else "not available")
    name_str = patient_name if patient_name and patient_name != "Anonymous" else "not available"

    key_findings: List[str] = [
        f"Patient: {name_str} (Age: {age_str}, Sex: {gender_str}, Village: {village_str})",
        f"Cough duration: {intake.cough_duration_days} days" if intake.has_cough else "Cough: not reported",
    ]

    symptoms_present = []
    if intake.has_fever:
        symptoms_present.append("fever")
    if intake.has_weight_loss:
        symptoms_present.append("weight loss")
    if intake.has_night_sweats:
        symptoms_present.append("night sweats")
    if intake.has_hemoptysis:
        symptoms_present.append("hemoptysis")

    key_findings.append(f"Reported symptoms: {', '.join(symptoms_present) if symptoms_present else 'None reported'}")

    if imaging:
        key_findings.append(f"CXR: {imaging.findings} (Probability: {imaging.tb_probability:.2f})")
    else:
        key_findings.append("CXR: not available")

    evidence_for: List[str] = []
    if intake.has_cough and intake.cough_duration_days >= 14:
        evidence_for.append(f"Persistent chronic cough lasting {intake.cough_duration_days} days (>= 14 days NTEP threshold)")
    elif intake.has_cough:
        evidence_for.append(f"Reported cough of {intake.cough_duration_days} days duration")

    if intake.has_fever:
        evidence_for.append("Reported recurrent or evening fever")
    if intake.has_weight_loss:
        evidence_for.append("Reported unexplained weight loss")
    if intake.has_hemoptysis:
        evidence_for.append("Hemoptysis (blood in sputum) alarm sign reported")
    if intake.has_night_sweats:
        evidence_for.append("Reported drenching night sweats")
    if intake.contact_history:
        evidence_for.append("Documented close or household contact with an active TB patient")
    if imaging and imaging.abnormality_detected:
        evidence_for.append(f"Radiographic parenchymal opacities detected on CXR (TB probability: {imaging.tb_probability:.2f})")

    evidence_against: List[str] = []
    if not intake.has_cough:
        evidence_against.append("Cough not reported")
    if not intake.has_fever:
        evidence_against.append("Fever not reported")
    if not intake.has_weight_loss:
        evidence_against.append("Weight loss not reported")
    if not intake.has_hemoptysis:
        evidence_against.append("Hemoptysis not reported")
    if not intake.contact_history:
        evidence_against.append("No known household TB contact reported")
    if age is None:
        evidence_against.append("Patient age: not available")
    if village is None and not intake.village:
        evidence_against.append("Patient village: not available")
    if not imaging:
        evidence_against.append("Chest radiograph: not available")
    elif imaging.model_provenance == "synthetic_demo":
        evidence_against.append("Imaging evaluated with synthetic_demo model; lacks clinical validation on human radiographs")

    suggested_next_tests: List[str] = []
    if risk_category == "High":
        suggested_next_tests.append("CBNAAT / GeneXpert molecular testing for Mycobacterium tuberculosis detection and Rifampicin resistance")
        suggested_next_tests.append("Sputum smear microscopy (spot and early morning samples)")
        suggested_next_tests.append("Immediate Medical Officer clinical examination and triage")
    elif risk_category == "Moderate":
        suggested_next_tests.append("Sputum smear microscopy for acid-fast bacilli")
        suggested_next_tests.append("Broad-spectrum antibiotic trial with follow-up review if symptoms persist")
    else:
        suggested_next_tests.append("Symptomatic care and clinical observation")
        suggested_next_tests.append("Re-screening if cough persists beyond 14 days")

    flags: List[str] = []
    if intake.has_hemoptysis:
        flags.append("URGENT ALARM: Hemoptysis reported - prioritize immediate doctor evaluation")
    if intake.contact_history:
        flags.append("EPIDEMIOLOGY ALERT: Household TB contact positive - consider contact tracing")
    if imaging and imaging.model_provenance == "synthetic_demo":
        flags.append("PROVENANCE NOTICE: Imaging analyzed with synthetic_demo smoke-test fixture - not clinically validated")

    return DoctorBriefing(
        key_findings=key_findings,
        evidence_for=evidence_for,
        evidence_against_or_uncertainty=evidence_against,
        suggested_next_tests=suggested_next_tests,
        flags=flags
    )


def format_briefing_to_markdown(briefing: DoctorBriefing) -> str:
    """Format structured DoctorBriefing into readable markdown narrative summary."""
    lines: List[str] = []

    lines.append("### Key Findings")
    for item in briefing.key_findings:
        lines.append(f"- {item}")

    lines.append("\n### Evidence For Presumptive TB")
    if briefing.evidence_for:
        for item in briefing.evidence_for:
            lines.append(f"- {item}")
    else:
        lines.append("- None reported")

    lines.append("\n### Evidence Against / Uncertainty")
    if briefing.evidence_against_or_uncertainty:
        for item in briefing.evidence_against_or_uncertainty:
            lines.append(f"- {item}")
    else:
        lines.append("- None noted")

    lines.append("\n### Suggested Next Tests")
    for item in briefing.suggested_next_tests:
        lines.append(f"- {item}")

    if briefing.flags:
        lines.append("\n### Clinical Flags")
        for item in briefing.flags:
            lines.append(f"- ⚠️ {item}")

    return "\n".join(lines)


def generate_doctor_briefing(
    prompt: str,
    fallback_briefing: DoctorBriefing,
    client: Optional[Any] = None
) -> DoctorBriefing:
    """Generate structured DoctorBriefing via Gemini LLM with at most one retry before offline fallback."""
    if client is None:
        client = get_gemini_client()

    if client is None:
        return fallback_briefing

    # At most one retry (2 attempts total)
    for attempt in range(2):
        try:
            from google.genai import types
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DoctorBriefing,
                    temperature=0.0,
                )
            )
            if hasattr(response, "parsed") and response.parsed is not None:
                if isinstance(response.parsed, DoctorBriefing):
                    return response.parsed
                return DoctorBriefing(**response.parsed)

            import json
            data = json.loads(response.text)
            return DoctorBriefing(**data)
        except Exception:
            if attempt == 1:
                break

    return fallback_briefing


def run_decision(
    case_id: str,
    patient_name: str,
    age: Optional[int],
    intake: IntakeResult,
    imaging: Optional[ImagingResult] = None,
    lab: Optional[LabOrderResult] = None,
    gender: Optional[str] = None,
    village: Optional[str] = None,
    client: Optional[Any] = None
) -> DecisionResult:
    """Synthesize screening findings into risk assessment and structured clinical case briefing.

    Args:
        case_id: Screening case identifier.
        patient_name: Patient name.
        age: Patient age in years, or None if not available.
        intake: Structured intake findings from Intake Agent.
        imaging: CXR radiology result from Imaging Agent.
        lab: Lab order details.
        gender: Patient sex/gender, or None if not available.
        village: Patient village, or None if not available.
        client: Optional genai.Client instance for unit test mocking.

    Returns:
        DecisionResult: Structured briefing, preliminary recommendation, and deterministic risk score.
    """
    # 1. Compute transparent deterministic risk score
    score = compute_deterministic_risk_score(intake=intake, imaging=imaging)
    category, recommendation = get_risk_category_and_recommendation(score)

    # Resolve demographics from payload or intake
    final_gender = gender or intake.gender
    final_village = village or intake.village
    final_age = age if age is not None else intake.age

    # 2. Build deterministic fallback briefing
    fallback_briefing = generate_deterministic_briefing(
        patient_name=patient_name,
        age=final_age,
        gender=final_gender,
        village=final_village,
        intake=intake,
        imaging=imaging,
        lab=lab,
        risk_score=score,
        risk_category=category
    )

    # 3. Build prompt with strict constraints
    prompt = build_briefing_prompt(
        case_id=case_id,
        patient_name=patient_name,
        age=final_age,
        gender=final_gender,
        village=final_village,
        intake=intake,
        imaging=imaging,
        lab=lab,
        risk_score=score,
        risk_category=category
    )

    # 4. Generate doctor briefing via Gemini or fallback
    briefing = generate_doctor_briefing(
        prompt=prompt,
        fallback_briefing=fallback_briefing,
        client=client
    )

    clinical_summary = format_briefing_to_markdown(briefing)

    return DecisionResult(
        tb_risk_score=score,
        risk_category=category,  # type: ignore[arg-type]
        preliminary_recommendation=recommendation,
        clinical_summary=clinical_summary,
        requires_doctor_review=True,
        briefing=briefing
    )
