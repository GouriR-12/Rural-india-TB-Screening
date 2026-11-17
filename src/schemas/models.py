"""Typed Pydantic schemas for multi-agent TB screening system."""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import operator
from typing import Annotated


class GeminiIntakeExtraction(BaseModel):
    """Strict schema for Gemini JSON extraction from ASHA clinical notes."""
    patient_name: Optional[str] = Field(default=None, description="Patient name if mentioned in text, otherwise null")
    age: Optional[int] = Field(default=None, description="Patient age in years, or null if not explicitly stated")
    sex: Optional[Literal["M", "F", "Other"]] = Field(default=None, description="Biological sex 'M', 'F', 'Other', or null if not stated")
    village: Optional[str] = Field(default=None, description="Village or primary healthcare center name, or null if not stated")
    has_cough: Optional[bool] = Field(default=None, description="Whether cough is reported (true/false), or null if not mentioned")
    cough_duration_days: Optional[int] = Field(default=None, description="Duration of cough in days (e.g. 3 weeks -> 21 days), or null if not stated")
    has_fever: Optional[bool] = Field(default=None, description="Whether fever/evening fever is reported, or null if not mentioned")
    fever_duration_days: Optional[int] = Field(default=None, description="Duration of fever in days, or null if not stated")
    has_weight_loss: Optional[bool] = Field(default=None, description="Unexplained weight loss or loss of appetite, or null if not mentioned")
    has_night_sweats: Optional[bool] = Field(default=None, description="Drenching night sweats, or null if not mentioned")
    has_hemoptysis: Optional[bool] = Field(default=None, description="Blood in sputum or coughing up blood, or null if not mentioned")
    tb_contact: Optional[bool] = Field(default=None, description="Close or household contact with known TB case, or null if not mentioned")
    diabetes: Optional[bool] = Field(default=None, description="History of diabetes or high blood sugar, or null if not mentioned")
    smoker: Optional[bool] = Field(default=None, description="History of smoking bidi/cigarettes/tobacco, or null if not mentioned")
    hiv: Optional[bool] = Field(default=None, description="HIV status known positive/negative, or null if not mentioned")
    other_symptoms: List[str] = Field(default_factory=list, description="Other clinical symptoms mentioned")
    language_detected: str = Field(default="en", description="Detected language: 'hi', 'mr', 'en', 'hinglish', or 'marathish'")


class IntakePayload(BaseModel):
    """Input payload submitted by an ASHA frontline health worker."""
    case_id: str = Field(default="CASE-AUTO", description="Unique screening case identifier")
    patient_id: str = Field(default="PAT-AUTO", description="Unique patient identifier")
    patient_name: str = Field(default="Anonymous", description="Patient full name")
    age: Optional[int] = Field(default=None, ge=0, le=125, description="Patient age in years")
    gender: Optional[Literal["M", "F", "Other"]] = Field(default=None, description="Patient gender")
    village: Optional[str] = Field(default=None, description="Village or primary healthcare sub-center name")
    asha_id: str = Field(default="ASHA-DEFAULT", description="ASHA worker identification number")
    raw_notes: str = Field(
        ...,
        description="Free-text or audio transcript notes in Hindi, Marathi, or English"
    )
    symptoms: List[str] = Field(
        default_factory=list,
        description="Checklist symptoms (e.g. cough > 2 weeks, fever, night sweats, weight loss, hemoptysis)"
    )
    duration_days: int = Field(default=0, ge=0, description="Reported duration of symptoms in days")
    contact_history: bool = Field(default=False, description="Known contact with confirmed TB case")


class IntakeResult(BaseModel):
    """Structured extraction produced by the Intake Agent."""
    extracted_symptoms: List[str] = Field(default_factory=list)
    has_cough: bool = Field(default=False)
    cough_duration_days: int = Field(default=0)
    has_fever: bool = Field(default=False)
    has_night_sweats: bool = Field(default=False)
    has_weight_loss: bool = Field(default=False)
    has_hemoptysis: bool = Field(default=False)
    contact_history: bool = Field(default=False)
    language_detected: str = Field(default="en")
    intake_timestamp: str = Field(default="")
    confidence: float = Field(default=1.0)
    # Extended fields for strict extraction, follow-up triage, and transparent risk scoring
    patient_name: Optional[str] = Field(default=None)
    age: Optional[int] = Field(default=None)
    gender: Optional[str] = Field(default=None)
    village: Optional[str] = Field(default=None)
    risk_factors: Dict[str, Optional[bool]] = Field(default_factory=dict)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: Literal["Low", "Medium", "High"] = Field(default="Low")
    top_contributing_factors: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = Field(default=None)


class ImagingPayload(BaseModel):
    """Input for chest radiograph (CXR) analysis."""
    case_id: str
    patient_id: str
    image_path: Optional[str] = Field(
        default=None,
        description="Path to chest X-ray image (public/synthetic dataset)"
    )


class ImagingResult(BaseModel):
    """Output produced by Imaging Agent (CNN)."""
    abnormality_detected: bool = Field(default=False)
    tb_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    heatmap_path: Optional[str] = Field(
        default=None,
        description="Path to generated Grad-CAM saliency map visualization"
    )
    confidence_note: str = Field(
        default="High confidence",
        description="Confidence assessment note based on model prediction margin"
    )
    findings: str = Field(default="Normal lung fields. No active parenchymal infiltrates.")
    model_version: str = Field(default="densenet121-tb-v1")
    model_provenance: Literal["synthetic_demo", "real_public_data"] = Field(
        default="synthetic_demo",
        description="Dataset provenance: 'synthetic_demo' (smoke-test only) or 'real_public_data'"
    )


class LabOrderPayload(BaseModel):
    """Input for lab order dispatch."""
    case_id: str
    patient_id: str
    test_type: str = Field(default="CBNAAT / GeneXpert")
    priority: Literal["Routine", "Urgent"] = Field(default="Routine")


class LabOrderResult(BaseModel):
    """Output for sputum / molecular lab test order."""
    order_id: str
    test_type: str
    order_status: Literal["Ordered", "Sample Collected", "Completed"] = "Ordered"
    order_timestamp: str = ""
    target_facility: str = "PHC Lab"


class DoctorBriefing(BaseModel):
    """Structured clinical case briefing generated for Medical Officer review."""
    key_findings: List[str] = Field(
        default_factory=list,
        description="Core clinical observations, symptoms, and duration strictly from inputs"
    )
    evidence_for: List[str] = Field(
        default_factory=list,
        description="Clinical indicators and risk factors consistent with presumptive pulmonary TB"
    )
    evidence_against_or_uncertainty: List[str] = Field(
        default_factory=list,
        description="Absence of symptoms, missing fields ('not available'), or imaging uncertainties"
    )
    suggested_next_tests: List[str] = Field(
        default_factory=list,
        description="Suggested laboratory and radiologic confirmation steps"
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Urgent safety or clinical alerts (e.g. hemoptysis alarm, synthetic demo imaging warning)"
    )


class DecisionResult(BaseModel):
    """Clinical synthesis and recommendation produced by Decision Agent."""
    tb_risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite TB risk estimate")
    risk_category: Literal["High", "Moderate", "Low"] = Field(...)
    preliminary_recommendation: str = Field(...)
    clinical_summary: str = Field(...)
    requires_doctor_review: bool = Field(
        default=True,
        description="Must always be True per mandatory doctor gate rule"
    )
    briefing: Optional[DoctorBriefing] = Field(
        default=None,
        description="Structured briefing generated by Gemini or deterministic fallback"
    )


class DoctorReviewPayload(BaseModel):
    """Submission by Medical Officer during doctor review gate."""
    case_id: str
    approved: bool = Field(
        ...,
        description="True = Confirmed/Accepted for TB management; False = Not TB / Re-screen / Refer"
    )
    doctor_id: str = Field(..., description="Medical Officer registration ID")
    doctor_notes: str = Field(..., description="Doctor clinical remarks")
    final_diagnosis: str = Field(
        default="Confirmed Pulmonary TB",
        description="Doctor's final assessment"
    )
    prescription: Optional[str] = Field(
        default=None,
        description="Treatment regimen (e.g. 4FDC / ATT regimen)"
    )


class DoctorReviewResult(BaseModel):
    """Record of doctor sign-off."""
    approved: bool
    doctor_id: str
    doctor_notes: str
    final_diagnosis: str
    prescription: Optional[str] = None
    review_timestamp: str = ""


class FollowupResult(BaseModel):
    """Follow-up plan and dropout risk calculated by Follow-up Agent."""
    dropout_risk_score: float = Field(..., ge=0.0, le=1.0)
    dropout_risk_level: Literal["High", "Medium", "Low"] = Field(...)
    adherence_plan: str = Field(...)
    next_visit_date: str = Field(...)
    asha_assigned: str = Field(...)
    dots_center: str = Field(...)


class SurveillanceResult(BaseModel):
    """Epidemiological metrics recorded by Surveillance Agent."""
    village: str
    cluster_detected: bool = False
    alert_level: Literal["Normal", "Watch", "Outbreak"] = "Normal"
    cumulative_village_cases: int = 1
    surveillance_timestamp: str = ""
    notes: str = ""


class ScreeningState(TypedDict, total=False):
    """LangGraph state schema for TB screening workflow."""
    # Identification
    case_id: str
    patient_id: str
    patient_name: str
    age: int
    gender: str
    village: str
    asha_id: str

    # Raw Inputs
    raw_notes: str
    symptoms: List[str]
    duration_days: int
    contact_history: bool
    image_path: Optional[str]

    # Agent Outputs
    intake_result: Optional[Dict[str, Any]]
    imaging_result: Optional[Dict[str, Any]]
    lab_result: Optional[Dict[str, Any]]
    decision_result: Optional[Dict[str, Any]]
    doctor_review: Optional[Dict[str, Any]]
    followup_result: Optional[Dict[str, Any]]
    surveillance_result: Optional[Dict[str, Any]]

    # Pipeline Control & Status
    status: str
    final_outcome: Optional[str]
    audit_trail: Annotated[List[str], operator.add]
