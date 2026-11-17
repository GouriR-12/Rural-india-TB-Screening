"""UI Backend Service Module for Tuberculosis Screening.

Pure Python backend service (no Streamlit dependencies) providing persistence,
agent orchestration, case review, report generation, and demo seeding.

DISCLAIMER:
Research prototype, not clinically validated, not for real diagnosis.
"""

from datetime import datetime, timedelta, timezone
import json
import os
import random
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

from src.agents.decision import order_lab_test, run_decision
from src.agents.followup import run_followup
from src.agents.imaging import run_imaging
from src.agents.intake import run_intake
from src.agents.surveillance import run_surveillance
from src.schemas.models import (
    DecisionResult,
    DoctorReviewResult,
    FollowupResult,
    ImagingPayload,
    ImagingResult,
    IntakePayload,
    IntakeResult,
    LabOrderPayload,
    LabOrderResult,
    ScreeningState,
)

DEFAULT_DB_PATH = os.path.join("data", "tb_screening.db")
MANDATORY_DISCLAIMER = "Research prototype, not clinically validated, not for real diagnosis."

DEMO_VILLAGES = [
    "Junnar", "Shirur", "Khed", "Ambegaon", "Baramati", "Parner",
    "Sangamner", "Karad", "Dindori", "Barshi", "Kinwat", "Dharni"
]

DEMO_PATIENT_NAMES_M = [
    "Ramesh Patil", "Ganesh Kamble", "Suresh Deshmukh", "Santosh Shinde",
    "Prakash Jadhav", "Sachin Kulkarni", "Deepak Pawar", "Vijay More"
]

DEMO_PATIENT_NAMES_F = [
    "Sunita Devi", "Kavita Gaikwad", "Pooja Bhosale", "Anuradha Mane",
    "Shobha Kadam", "Rekha Chavan", "Sangeeta Kale", "Lata Chavan"
]

_OVERRIDE_LLM_MODE: Optional[str] = None


def set_llm_mode(mode: Optional[str]) -> None:
    """Explicitly override LLM mode ('live', 'fallback', or None to reset)."""
    global _OVERRIDE_LLM_MODE
    _OVERRIDE_LLM_MODE = mode


def llm_mode() -> str:
    """Return 'live' if GEMINI_API_KEY is present, else 'fallback'.

    Returns:
        str: 'live' or 'fallback'
    """
    if _OVERRIDE_LLM_MODE in ("live", "fallback"):
        return _OVERRIDE_LLM_MODE
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return "live" if key else "fallback"


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize SQLite database and create cases table if it does not exist."""
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                village TEXT,
                status TEXT NOT NULL,
                state_json TEXT NOT NULL,
                doctor_comment TEXT,
                reviewed_at TEXT,
                asha_id TEXT
            )
            """
        )
        # Ensure asha_id column exists for existing tables
        cursor = conn.execute("PRAGMA table_info(cases)")
        columns = [row[1] for row in cursor.fetchall()]
        if "asha_id" not in columns:
            conn.execute("ALTER TABLE cases ADD COLUMN asha_id TEXT")
        # Backfill asha_id from state_json if null
        try:
            conn.execute(
                "UPDATE cases SET asha_id = json_extract(state_json, '$.asha_id') WHERE asha_id IS NULL AND json_extract(state_json, '$.asha_id') IS NOT NULL"
            )
        except Exception:
            pass
        conn.commit()


def get_case(case_id: str, db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a single case record from SQLite."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if not row:
            return None
        return dict(row)


def list_cases(status: Optional[str] = None, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all case records, optionally filtered by status."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if status:
            rows = conn.execute(
                "SELECT * FROM cases WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM cases ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_cases_for_asha(asha_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve safe frontline case records for a specific ASHA worker.

    Excludes all AI summaries, imaging outputs, risk scores, and raw state_json.
    Returns only safe frontline fields: id, patient, village, date, status.

    Args:
        asha_id: Frontline ASHA worker ID.
        db_path: Target SQLite database file path.

    Returns:
        List[Dict[str, Any]]: Safe case records containing only id, patient, village, date, status.
    """
    init_db(db_path)
    target_id = asha_id.strip().lower()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, village, status, state_json, asha_id
            FROM cases
            ORDER BY created_at DESC
            """
        ).fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            row_asha = (r["asha_id"] or "").strip().lower()
            patient_name = "Anonymous"

            # Parse patient_name safely; fallback for asha_id if older unmigrated record
            try:
                st_data = json.loads(r["state_json"])
                patient_name = st_data.get("patient_name") or "Anonymous"
                if not row_asha:
                    row_asha = (st_data.get("asha_id") or "").strip().lower()
            except Exception:
                pass

            if row_asha != target_id:
                continue

            raw_status = r["status"]
            if raw_status == "PENDING_REVIEW":
                display_status = "Not reviewed yet"
            elif raw_status == "APPROVED":
                display_status = "Approved"
            elif raw_status == "REJECTED":
                display_status = "Rejected"
            else:
                display_status = raw_status

            created_date = (r["created_at"] or "")[:10]

            results.append({
                "id": r["id"],
                "patient": patient_name,
                "village": r["village"] or "Unknown",
                "date": created_date,
                "status": display_status,
            })

        return results


def submit_case(
    form_data: Dict[str, Any],
    cxr_path: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    """Execute Intake -> Imaging -> Decision and persist case with PENDING_REVIEW status.

    Args:
        form_data: Patient clinical intake metadata and frontline ASHA observations.
        cxr_path: Optional path to patient's chest radiograph file.
        db_path: Target SQLite database file path.

    Returns:
        str: Generated or provided case_id.
    """
    init_db(db_path)

    case_id = str(
        form_data.get("case_id")
        or f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    )
    patient_id = str(form_data.get("patient_id") or f"PAT-{uuid.uuid4().hex[:6].upper()}")
    patient_name = str(form_data.get("patient_name", "Anonymous"))
    village = str(form_data.get("village") or "Unknown Village")
    age = form_data.get("age")
    gender = form_data.get("gender")
    asha_id = str(form_data.get("asha_id", "ASHA-DEFAULT"))
    raw_notes = str(form_data.get("raw_notes", ""))
    symptoms = list(form_data.get("symptoms", []))
    duration_days = int(form_data.get("duration_days", 0))
    contact_history = bool(form_data.get("contact_history", False))
    actual_cxr = cxr_path or form_data.get("image_path")

    # 1. Intake Agent
    intake_payload = IntakePayload(
        case_id=case_id,
        patient_id=patient_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        village=village,
        asha_id=asha_id,
        raw_notes=raw_notes,
        symptoms=symptoms,
        duration_days=duration_days,
        contact_history=contact_history,
    )
    intake_result: IntakeResult = run_intake(intake_payload)

    # 2. Imaging Agent
    imaging_payload = ImagingPayload(
        case_id=case_id,
        patient_id=patient_id,
        image_path=actual_cxr,
    )
    imaging_result: ImagingResult = run_imaging(imaging_payload)

    # 3. Lab Order Dispatch
    lab_payload = LabOrderPayload(
        case_id=case_id,
        patient_id=patient_id,
        test_type="CBNAAT / GeneXpert",
        priority="Urgent" if (intake_result.cough_duration_days or 0) >= 14 else "Routine",
    )
    lab_result: LabOrderResult = order_lab_test(lab_payload)

    # 4. Decision Agent
    decision_result: DecisionResult = run_decision(
        case_id=case_id,
        patient_name=intake_result.patient_name or patient_name,
        age=intake_result.age if intake_result.age is not None else age,
        intake=intake_result,
        imaging=imaging_result,
        lab=lab_result,
        gender=intake_result.gender or gender,
        village=intake_result.village or village,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    resolved_village = intake_result.village or village

    # 5. Compose full ScreeningState
    state: Dict[str, Any] = {
        "case_id": case_id,
        "patient_id": patient_id,
        "patient_name": intake_result.patient_name or patient_name,
        "age": intake_result.age if intake_result.age is not None else age,
        "gender": intake_result.gender or gender,
        "village": resolved_village,
        "asha_id": asha_id,
        "raw_notes": raw_notes,
        "symptoms": intake_result.extracted_symptoms or symptoms,
        "duration_days": intake_result.cough_duration_days or duration_days,
        "contact_history": intake_result.contact_history,
        "image_path": actual_cxr,
        "intake_result": intake_result.model_dump(),
        "imaging_result": imaging_result.model_dump(),
        "lab_result": lab_result.model_dump(),
        "decision_result": decision_result.model_dump(),
        "doctor_review": None,
        "followup_result": None,
        "surveillance_result": None,
        "status": "PENDING_REVIEW",
        "audit_trail": [
            f"[{created_at}] Case {case_id} submitted by ASHA {asha_id}",
            f"[{created_at}] Intake completed: Risk={intake_result.risk_level} ({intake_result.risk_score:.4f})",
            f"[{created_at}] CXR analysis completed: Prob={imaging_result.tb_probability:.4f}",
            f"[{created_at}] Lab order dispatched: {lab_result.test_type}",
            f"[{created_at}] Decision synthesized: Risk={decision_result.risk_category} ({decision_result.tb_risk_score:.4f})",
            f"[{created_at}] Status set to PENDING_REVIEW",
        ],
    }

    # 6. Save into SQLite
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO cases (id, created_at, village, status, state_json, doctor_comment, reviewed_at, asha_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                created_at,
                resolved_village,
                "PENDING_REVIEW",
                json.dumps(state),
                None,
                None,
                asha_id,
            ),
        )
        conn.commit()

    return case_id


def review_case(
    case_id: str,
    decision: Any,
    comment: str,
    doctor_id: str = "DR-MEDICAL-OFFICER",
    db_path: str = DEFAULT_DB_PATH,
) -> Dict[str, Any]:
    """Review a case: on APPROVED run Follow-up and store result; on REJECTED just store status.

    Args:
        case_id: Target screening case ID.
        decision: Decision indication ('APPROVED', True, 'REJECTED', False).
        comment: Doctor's clinical notes / remarks.
        doctor_id: Medical Officer identifier.
        db_path: Target SQLite database file path.

    Returns:
        Dict[str, Any]: Updated case record.

    Raises:
        ValueError: If case is not found or decision is invalid.
    """
    case = get_case(case_id, db_path=db_path)
    if not case:
        raise ValueError(f"Case '{case_id}' not found.")

    state: Dict[str, Any] = json.loads(case["state_json"])

    # Normalize decision
    is_approved: bool
    if isinstance(decision, bool):
        is_approved = decision
    elif isinstance(decision, str):
        dec_clean = decision.strip().upper()
        if dec_clean in ("APPROVED", "APPROVE", "TRUE", "YES"):
            is_approved = True
        elif dec_clean in ("REJECTED", "REJECT", "FALSE", "NO"):
            is_approved = False
        else:
            raise ValueError(f"Invalid decision '{decision}'. Expected APPROVED or REJECTED.")
    else:
        raise ValueError(f"Invalid decision type: {type(decision)}. Expected str or bool.")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    status_str = "APPROVED" if is_approved else "REJECTED"

    if is_approved:
        doctor_review = DoctorReviewResult(
            approved=True,
            doctor_id=doctor_id,
            doctor_notes=comment or "Doctor approved for treatment.",
            final_diagnosis="Confirmed Pulmonary Tuberculosis",
            prescription="Standard 4FDC regimen (Rifampicin, Isoniazid, Pyrazinamide, Ethambutol)",
            review_timestamp=reviewed_at,
        )

        decision_data = state.get("decision_result") or {}
        decision_obj = DecisionResult(**decision_data)

        # Run Follow-up Agent (only for APPROVED)
        followup_result = run_followup(
            patient_id=state.get("patient_id", "PAT-DEFAULT"),
            village=state.get("village", "Unknown Village"),
            asha_id=state.get("asha_id", "ASHA-DEFAULT"),
            decision=decision_obj,
            doctor_review=doctor_review,
        )

        # Run Surveillance Agent
        surv_result = run_surveillance(
            village=state.get("village", "Unknown Village"),
            confirmed=True,
            patient_id=state.get("patient_id", "PAT-DEFAULT"),
        )

        state["doctor_review"] = doctor_review.model_dump()
        state["followup_result"] = followup_result.model_dump()
        state["surveillance_result"] = surv_result.model_dump()
        state["status"] = "APPROVED"
        state["final_outcome"] = "TB Confirmed - Active in DOTS & Public Health Surveillance"
        state.setdefault("audit_trail", []).append(
            f"[{reviewed_at}] Doctor Review APPROVED by {doctor_id}: {comment}"
        )
    else:
        doctor_review = DoctorReviewResult(
            approved=False,
            doctor_id=doctor_id,
            doctor_notes=comment or "Doctor rejected case: Non-TB diagnosis.",
            final_diagnosis="Non-TB Respiratory Ailment",
            prescription=None,
            review_timestamp=reviewed_at,
        )

        state["doctor_review"] = doctor_review.model_dump()
        state["followup_result"] = None
        state["status"] = "REJECTED"
        state["final_outcome"] = "Not TB - refer or re-screen"
        state.setdefault("audit_trail", []).append(
            f"[{reviewed_at}] Doctor Review REJECTED by {doctor_id}: {comment}"
        )

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE cases
            SET status = ?, doctor_comment = ?, reviewed_at = ?, state_json = ?
            WHERE id = ?
            """,
            (status_str, comment, reviewed_at, json.dumps(state), case_id),
        )
        conn.commit()

    return get_case(case_id, db_path=db_path)  # type: ignore[return-value]


def run_followup_for_case(case_id: str, db_path: str = DEFAULT_DB_PATH) -> FollowupResult:
    """Run Follow-up agent for a case. Raises ValueError if case status != APPROVED.

    Rule 1 & Specification: Follow-up must raise an error if status != APPROVED.

    Args:
        case_id: Identifier of the case.
        db_path: Target SQLite database file path.

    Returns:
        FollowupResult: Result from Follow-up Agent.

    Raises:
        ValueError: If case is not found, or if case status is not APPROVED.
    """
    case = get_case(case_id, db_path=db_path)
    if not case:
        raise ValueError(f"Case '{case_id}' not found.")

    if case["status"] != "APPROVED":
        raise ValueError(
            f"Cannot initiate follow-up for a non-approved case (status='{case['status']}'). "
            f"Doctor sign-off is mandatory."
        )

    state = json.loads(case["state_json"])
    decision_obj = DecisionResult(**state["decision_result"])
    doctor_review_dict = state.get("doctor_review") or {
        "approved": True,
        "doctor_id": "DR-MEDICAL-OFFICER",
        "doctor_notes": case.get("doctor_comment") or "Approved",
        "final_diagnosis": "Confirmed Pulmonary Tuberculosis",
        "review_timestamp": case.get("reviewed_at") or datetime.now(timezone.utc).isoformat(),
    }
    doctor_review_obj = DoctorReviewResult(**doctor_review_dict)

    return run_followup(
        patient_id=state.get("patient_id", "PAT-DEFAULT"),
        village=state.get("village", "Unknown Village"),
        asha_id=state.get("asha_id", "ASHA-DEFAULT"),
        decision=decision_obj,
        doctor_review=doctor_review_obj,
    )


def build_final_report(case_id: str, db_path: str = DEFAULT_DB_PATH) -> str:
    """Build a comprehensive clinical summary markdown report for a screening case.

    Includes:
    - Patient details
    - Symptoms
    - Imaging + provenance
    - AI summary
    - Doctor decision / comment / timestamp
    - Follow-up plan if approved
    - Mandatory disclaimer

    Args:
        case_id: Case identifier.
        db_path: Target SQLite database path.

    Returns:
        str: Markdown formatted clinical report.
    """
    case = get_case(case_id, db_path=db_path)
    if not case:
        raise ValueError(f"Case '{case_id}' not found.")

    state: Dict[str, Any] = json.loads(case["state_json"])
    intake = state.get("intake_result") or {}
    imaging = state.get("imaging_result") or {}
    decision = state.get("decision_result") or {}
    followup = state.get("followup_result") or {}
    doc_rev = state.get("doctor_review") or {}

    patient_name = state.get("patient_name", "Anonymous")
    patient_id = state.get("patient_id", "N/A")
    age = state.get("age", "N/A")
    gender = state.get("gender", "N/A")
    village = case.get("village") or state.get("village", "N/A")
    asha_id = state.get("asha_id", "N/A")
    status = case.get("status", "UNKNOWN")
    created_at = case.get("created_at", "N/A")
    reviewed_at = case.get("reviewed_at") or doc_rev.get("review_timestamp") or "Pending Review"
    doctor_comment = case.get("doctor_comment") or doc_rev.get("doctor_notes") or "No doctor comments yet."

    symptoms_list = state.get("symptoms") or intake.get("extracted_symptoms") or []
    symptoms_str = ", ".join(symptoms_list) if symptoms_list else "None reported"
    duration = state.get("duration_days") or intake.get("cough_duration_days", 0)
    contact_hist = "Yes" if state.get("contact_history") or intake.get("contact_history") else "No"

    provenance = imaging.get("model_provenance", "synthetic_demo")
    tb_prob = imaging.get("tb_probability", 0.0)
    abnormality = "Yes" if imaging.get("abnormality_detected") else "No"
    findings = imaging.get("findings", "Not available")
    conf_note = imaging.get("confidence_note", "Not available")
    heatmap = imaging.get("heatmap_path") or "None"

    risk_score = decision.get("tb_risk_score", 0.0)
    risk_cat = decision.get("risk_category", "Unknown")
    prelim_rec = decision.get("preliminary_recommendation", "Not available")
    ai_summary = decision.get("clinical_summary", "No clinical summary available.")

    lines: List[str] = [
        f"# Tuberculosis Screening Clinical Report - `{case_id}`",
        "",
        f"> **Disclaimer**: {MANDATORY_DISCLAIMER}",
        "",
        "## 1. Patient Details",
        f"- **Patient Name**: {patient_name}",
        f"- **Patient ID**: {patient_id}",
        f"- **Age**: {age}",
        f"- **Gender**: {gender}",
        f"- **Village / Sub-center**: {village}",
        f"- **Assigned ASHA Worker**: {asha_id}",
        f"- **Case Registration Date**: {created_at}",
        f"- **Screening Status**: **{status}**",
        "",
        "## 2. Reported Symptoms & Clinical History",
        f"- **Symptoms**: {symptoms_str}",
        f"- **Cough Duration**: {duration} days",
        f"- **Household / Close Contact History**: {contact_hist}",
        f"- **ASHA Clinical Notes**: \"{state.get('raw_notes', '')}\"",
        "",
        "## 3. Chest Radiograph (Imaging) & Provenance",
        f"- **Model Provenance**: `{provenance}`",
        f"- **CNN TB Probability**: `{tb_prob:.4f}`",
        f"- **Radiographic Abnormality Detected**: {abnormality}",
        f"- **Radiologic Findings**: {findings}",
        f"- **Confidence Assessment**: {conf_note}",
        f"- **Grad-CAM Saliency Heatmap**: `{heatmap}`",
        "",
        "## 4. AI Clinical Summary & Risk Synthesis",
        f"- **Composite TB Risk Score**: `{risk_score:.4f}`",
        f"- **Risk Stratification**: **{risk_cat}**",
        f"- **Preliminary AI Recommendation**: {prelim_rec}",
        "",
        "### Clinical Synthesis Narrative",
        f"{ai_summary}",
        "",
        "## 5. Doctor Review & Decision",
        f"- **Review Status**: **{status}**",
        f"- **Medical Officer**: {doc_rev.get('doctor_id', 'Pending')}",
        f"- **Review Timestamp**: {reviewed_at}",
        f"- **Doctor Diagnosis**: {doc_rev.get('final_diagnosis', 'Pending')}",
        f"- **Doctor Comment / Remarks**: {doctor_comment}",
        f"- **Prescription**: {doc_rev.get('prescription') or 'None'}",
        "",
        "## 6. Follow-up & Treatment Adherence Plan",
    ]

    if status == "APPROVED" and followup:
        lines.extend([
            f"- **Adherence Plan**: {followup.get('adherence_plan', 'DOTS treatment initiated.')}",
            f"- **Next ASHA Home Visit**: {followup.get('next_visit_date', 'Scheduled within 7 days')}",
            f"- **Assigned ASHA Worker**: {followup.get('asha_assigned', asha_id)}",
            f"- **Loss-to-Follow-up Risk**: **{followup.get('dropout_risk_level', 'Medium')}** (Score: `{followup.get('dropout_risk_score', 0.25)}`)",
            f"- **DOTS Center**: {followup.get('dots_center', f'{village} Health Sub-Center')}",
        ])
    elif status == "REJECTED":
        lines.append("- *No follow-up plan scheduled. Case was reviewed and rejected as non-tuberculosis.*")
    else:
        lines.append("- *Pending Medical Officer review. Follow-up is initiated exclusively after doctor approval.*")

    lines.extend([
        "",
        "---",
        f"*{MANDATORY_DISCLAIMER}*",
    ])

    return "\n".join(lines)


def seed_demo_data(n: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[str]:
    """Generate n synthetic cases with random villages and dates and seed into SQLite cases table.

    Args:
        n: Number of synthetic demo cases to generate (default 50).
        db_path: Target SQLite database file path.

    Returns:
        List[str]: Generated case IDs.
    """
    init_db(db_path)
    generated_ids: List[str] = []

    now = datetime.now(timezone.utc)

    with sqlite3.connect(db_path) as conn:
        for i in range(1, n + 1):
            case_id = f"CASE-DEMO-{i:04d}"
            generated_ids.append(case_id)

            village = random.choice(DEMO_VILLAGES)
            is_male = (i % 2 == 0)
            patient_name = random.choice(DEMO_PATIENT_NAMES_M if is_male else DEMO_PATIENT_NAMES_F)
            patient_id = f"PAT-DEMO-{1000 + i}"
            age = random.randint(18, 70)
            gender = "M" if is_male else "F"
            asha_id = f"ASHA-{village[:4].upper()}-{random.randint(1, 15):02d}"

            # Randomize creation date across past 60 days
            days_ago = random.randint(1, 60)
            hours_ago = random.randint(0, 23)
            created_dt = now - timedelta(days=days_ago, hours=hours_ago)
            created_at = created_dt.isoformat()

            # Partition statuses: 40% PENDING_REVIEW, 40% APPROVED, 20% REJECTED
            mod = i % 5
            if mod in (0, 1):
                status = "PENDING_REVIEW"
            elif mod in (2, 3):
                status = "APPROVED"
            else:
                status = "REJECTED"

            has_chronic_cough = (i % 3 != 0)
            duration_days = random.randint(14, 40) if has_chronic_cough else random.randint(3, 10)
            has_fever = (i % 2 == 1)
            has_weight_loss = (has_chronic_cough and i % 4 == 0)
            has_hemoptysis = (has_chronic_cough and i % 7 == 0)
            contact_history = (i % 3 == 1)

            symptoms = []
            if duration_days >= 14:
                symptoms.append("Cough > 2 weeks")
            elif duration_days > 0:
                symptoms.append("Cough")
            if has_fever:
                symptoms.append("Fever")
            if has_weight_loss:
                symptoms.append("Weight Loss")
            if has_hemoptysis:
                symptoms.append("Hemoptysis")

            tb_prob = 0.82 if has_chronic_cough and has_fever else 0.18
            risk_score = 0.75 if (has_chronic_cough and has_fever) else 0.25
            risk_cat = "High" if risk_score >= 0.65 else ("Moderate" if risk_score >= 0.35 else "Low")

            intake_res = {
                "patient_name": patient_name,
                "age": age,
                "gender": gender,
                "village": village,
                "extracted_symptoms": symptoms,
                "has_cough": duration_days > 0,
                "cough_duration_days": duration_days,
                "has_fever": has_fever,
                "has_night_sweats": False,
                "has_weight_loss": has_weight_loss,
                "has_hemoptysis": has_hemoptysis,
                "contact_history": contact_history,
                "risk_factors": {"tb_contact": contact_history, "diabetes": False, "smoker": False, "hiv": False},
                "risk_score": risk_score,
                "risk_level": risk_cat,
                "top_contributing_factors": ["Persistent cough >= 14 days"] if duration_days >= 14 else ["Mild presentation"],
                "missing_fields": [],
                "follow_up_questions": [],
                "language_detected": "en",
                "raw_text": f"Patient reports cough for {duration_days} days.",
                "intake_timestamp": created_at,
                "confidence": 0.95,
            }

            imaging_res = {
                "abnormality_detected": bool(tb_prob >= 0.50),
                "tb_probability": tb_prob,
                "heatmap_path": f"data/cxr/heatmaps/{case_id}_gradcam.png" if tb_prob >= 0.50 else None,
                "confidence_note": "High confidence positive" if tb_prob >= 0.50 else "High confidence negative",
                "findings": "Apical infiltrates detected" if tb_prob >= 0.50 else "Clear lung fields bilaterally",
                "model_version": "densenet121-tb-v1",
                "model_provenance": "synthetic_demo",
            }

            decision_res = {
                "tb_risk_score": risk_score,
                "risk_category": risk_cat,
                "preliminary_recommendation": "CBNAAT / GeneXpert molecular testing" if risk_cat == "High" else "Symptomatic care",
                "clinical_summary": f"Case presentation with {', '.join(symptoms) if symptoms else 'mild symptoms'}. CXR probability: {tb_prob:.2f}.",
                "requires_doctor_review": True,
                "briefing": {
                    "key_findings": [f"Patient: {patient_name} ({age}y, {gender}, {village})"],
                    "evidence_for": [f"Cough {duration_days} days"] if duration_days >= 14 else [],
                    "evidence_against_or_uncertainty": ["Imaging from synthetic_demo model"],
                    "suggested_next_tests": ["CBNAAT GeneXpert"],
                    "flags": ["PROVENANCE NOTICE: Synthetic demo model"],
                },
            }

            doctor_comment: Optional[str] = None
            reviewed_at: Optional[str] = None
            doctor_review_dict: Optional[Dict[str, Any]] = None
            followup_dict: Optional[Dict[str, Any]] = None

            if status in ("APPROVED", "REJECTED"):
                review_dt = created_dt + timedelta(hours=random.randint(2, 24))
                reviewed_at = review_dt.isoformat()

                if status == "APPROVED":
                    doctor_comment = "Confirmed active pulmonary TB. Initiating Category 1 4FDC ATT."
                    doctor_review_dict = {
                        "approved": True,
                        "doctor_id": "DR-PHC-OFFICER",
                        "doctor_notes": doctor_comment,
                        "final_diagnosis": "Confirmed Pulmonary Tuberculosis",
                        "prescription": "Category 1 ATT (4FDC daily)",
                        "review_timestamp": reviewed_at,
                    }
                    next_visit = (review_dt + timedelta(days=7)).strftime("%Y-%m-%d")
                    followup_dict = {
                        "dropout_risk_score": 0.35,
                        "dropout_risk_level": "Medium",
                        "adherence_plan": f"DOTS Initiation under Nikshay Poshan Yojana. Weekly home visit by {asha_id}.",
                        "next_visit_date": next_visit,
                        "asha_assigned": asha_id,
                        "dots_center": f"{village} Health Sub-Center",
                    }
                else:
                    doctor_comment = "Clinical and CXR evaluation indicate non-TB upper respiratory infection."
                    doctor_review_dict = {
                        "approved": False,
                        "doctor_id": "DR-PHC-OFFICER",
                        "doctor_notes": doctor_comment,
                        "final_diagnosis": "Acute Bronchitis (Non-TB)",
                        "prescription": None,
                        "review_timestamp": reviewed_at,
                    }

            state = {
                "case_id": case_id,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "age": age,
                "gender": gender,
                "village": village,
                "asha_id": asha_id,
                "raw_notes": f"Frontline notes for {patient_name}.",
                "symptoms": symptoms,
                "duration_days": duration_days,
                "contact_history": contact_history,
                "image_path": "data/sample_cxr_abnormal.png" if tb_prob >= 0.50 else "data/sample_cxr_normal.png",
                "intake_result": intake_res,
                "imaging_result": imaging_res,
                "lab_result": {
                    "order_id": f"LAB-{case_id[-6:]}",
                    "test_type": "CBNAAT / GeneXpert",
                    "order_status": "Ordered",
                    "order_timestamp": created_at,
                    "target_facility": "Designated Microscopy Centre (DMC) / PHC",
                },
                "decision_result": decision_res,
                "doctor_review": doctor_review_dict,
                "followup_result": followup_dict,
                "surveillance_result": None,
                "status": status,
                "final_outcome": "TB Confirmed" if status == "APPROVED" else ("Not TB" if status == "REJECTED" else None),
                "audit_trail": [
                    f"[{created_at}] Case registered and evaluated: {status}",
                ],
            }

            conn.execute(
                """
                INSERT OR REPLACE INTO cases (id, created_at, village, status, state_json, doctor_comment, reviewed_at, asha_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    created_at,
                    village,
                    status,
                    json.dumps(state),
                    doctor_comment,
                    reviewed_at,
                    asha_id,
                ),
            )

        conn.commit()

    return generated_ids

