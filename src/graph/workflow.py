"""LangGraph State Machine for Multi-Agent TB Screening.

Flow:
    intake -> (imaging and lab-order parallel branches) -> decision -> doctor_review
    doctor_review uses LangGraph interrupt for human-in-the-loop Medical Officer review.
    - If approved: followup -> surveillance -> END
    - If rejected: rejected_end (status: "Not TB - refer or re-screen") -> END
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.agents.decision import order_lab_test, run_decision
from src.agents.followup import run_followup
from src.agents.imaging import run_imaging
from src.agents.intake import run_intake
from src.agents.surveillance import run_surveillance
from src.schemas.models import (
    DecisionResult,
    DoctorReviewPayload,
    DoctorReviewResult,
    ImagingPayload,
    ImagingResult,
    IntakePayload,
    IntakeResult,
    LabOrderPayload,
    LabOrderResult,
    ScreeningState,
)


def intake_node(state: ScreeningState) -> Dict[str, Any]:
    """Execute Intake Agent: convert ASHA notes & symptoms to structured JSON with calibrated risk."""
    payload = IntakePayload(
        case_id=state.get("case_id", "CASE-DEFAULT"),
        patient_id=state.get("patient_id", "PAT-DEFAULT"),
        patient_name=state.get("patient_name", "Anonymous"),
        age=state.get("age"),
        gender=state.get("gender"),
        village=state.get("village"),
        asha_id=state.get("asha_id", "ASHA-DEFAULT"),
        raw_notes=state.get("raw_notes", ""),
        symptoms=state.get("symptoms", []),
        duration_days=state.get("duration_days", 0),
        contact_history=state.get("contact_history", False),
    )
    result: IntakeResult = run_intake(payload)
    return {
        "intake_result": result.model_dump(),
        "status": "INTAKE_COMPLETED",
        "audit_trail": [
            f"[{datetime.now(timezone.utc).isoformat()}] Intake completed by ASHA {payload.asha_id}: "
            f"Risk={result.risk_level} ({result.risk_score:.4f}), Lang={result.language_detected}, "
            f"Missing={len(result.missing_fields)} fields"
        ],
    }


def imaging_node(state: ScreeningState) -> Dict[str, Any]:
    """Execute Imaging Agent (Parallel Branch): evaluate CXR via DenseNet-121 CNN + Grad-CAM."""
    payload = ImagingPayload(
        case_id=state.get("case_id", "CASE-DEFAULT"),
        patient_id=state.get("patient_id", "PAT-DEFAULT"),
        image_path=state.get("image_path"),
    )
    result: ImagingResult = run_imaging(payload)
    return {
        "imaging_result": result.model_dump(),
        "audit_trail": [
            f"[{datetime.now(timezone.utc).isoformat()}] CXR analysis completed (DenseNet-121 CNN): "
            f"TB Prob={result.tb_probability:.4f}, Abnormality={result.abnormality_detected}, "
            f"Grad-CAM Heatmap={result.heatmap_path or 'None'}, Provenance={result.model_provenance}"
        ],
    }


def lab_order_node(state: ScreeningState) -> Dict[str, Any]:
    """Execute Lab Order Step (Parallel Branch): dispatch sputum/GeneXpert order."""
    payload = LabOrderPayload(
        case_id=state.get("case_id", "CASE-DEFAULT"),
        patient_id=state.get("patient_id", "PAT-DEFAULT"),
        test_type="CBNAAT / GeneXpert",
        priority="Urgent" if state.get("duration_days", 0) >= 14 else "Routine",
    )
    result: LabOrderResult = order_lab_test(payload)
    return {
        "lab_result": result.model_dump(),
        "audit_trail": [f"[{datetime.now(timezone.utc).isoformat()}] Lab order dispatched: {result.test_type} ({result.order_id})"],
    }


def decision_node(state: ScreeningState) -> Dict[str, Any]:
    """Execute Decision Agent: synthesize intake, imaging, and lab orders."""
    intake_data = state.get("intake_result") or {}
    imaging_data = state.get("imaging_result") or {}
    lab_data = state.get("lab_result") or {}

    intake_obj = IntakeResult(**intake_data)
    imaging_obj = ImagingResult(**imaging_data) if imaging_data else None
    lab_obj = LabOrderResult(**lab_data) if lab_data else None

    result: DecisionResult = run_decision(
        case_id=state.get("case_id", "CASE-DEFAULT"),
        patient_name=state.get("patient_name", "Anonymous"),
        age=state.get("age", 30),
        intake=intake_obj,
        imaging=imaging_obj,
        lab=lab_obj,
    )

    return {
        "decision_result": result.model_dump(),
        "status": "AWAITING_DOCTOR_REVIEW",
        "audit_trail": [
            f"[{datetime.now(timezone.utc).isoformat()}] Decision synthesized: Risk={result.risk_category} ({result.tb_risk_score})"
        ],
    }


def doctor_review_node(state: ScreeningState) -> Dict[str, Any]:
    """Doctor Review Gate: Mandatory Human-in-the-Loop approval before Follow-up.

    Uses LangGraph interrupt to pause execution until a Medical Officer submits
    their diagnosis and sign-off decision.
    """
    decision_info = state.get("decision_result") or {}
    imaging_info = state.get("imaging_result") or {}
    provenance = imaging_info.get("model_provenance", "synthetic_demo")

    # LangGraph interrupt: Pauses execution and yields payload to the caller/UI
    review_input: Dict[str, Any] = interrupt({
        "gate": "MANDATORY_DOCTOR_REVIEW_GATE",
        "case_id": state.get("case_id"),
        "patient_name": state.get("patient_name"),
        "age": state.get("age"),
        "village": state.get("village"),
        "risk_category": decision_info.get("risk_category", "Unknown"),
        "tb_risk_score": decision_info.get("tb_risk_score", 0.0),
        "clinical_summary": decision_info.get("clinical_summary", ""),
        "preliminary_recommendation": decision_info.get("preliminary_recommendation", ""),
        "imaging_result": imaging_info,
        "model_provenance": provenance,
        "synthetic_warning": bool(provenance == "synthetic_demo"),
        "instruction": "Please review clinical evidence and provide approval (True/False), doctor_id, and clinical notes.",
    })

    # Validate review input
    approved = bool(review_input.get("approved", False))
    doctor_id = str(review_input.get("doctor_id", "DR-PHC-OFFICER"))
    doctor_notes = str(review_input.get("doctor_notes", "Clinical review completed."))
    final_diagnosis = str(
        review_input.get(
            "final_diagnosis",
            "Confirmed Pulmonary Tuberculosis" if approved else "Non-TB Respiratory Ailment"
        )
    )
    prescription = review_input.get("prescription")

    review_result = DoctorReviewResult(
        approved=approved,
        doctor_id=doctor_id,
        doctor_notes=doctor_notes,
        final_diagnosis=final_diagnosis,
        prescription=prescription,
        review_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if approved:
        return {
            "doctor_review": review_result.model_dump(),
            "status": "APPROVED",
            "audit_trail": [
                f"[{datetime.now(timezone.utc).isoformat()}] Doctor Gate APPROVED by {doctor_id}: {final_diagnosis}"
            ],
        }
    else:
        return {
            "doctor_review": review_result.model_dump(),
            "status": "REJECTED",
            "final_outcome": "Not TB - refer or re-screen",
            "audit_trail": [
                f"[{datetime.now(timezone.utc).isoformat()}] Doctor Gate REJECTED by {doctor_id}: {doctor_notes}"
            ],
        }


def route_after_doctor_review(state: ScreeningState) -> str:
    """Conditional edge router following Doctor Review Gate.

    Rule 1: No case can reach Follow-up without doctor approval.
    """
    review = state.get("doctor_review") or {}
    if review.get("approved", False) is True:
        return "followup"
    return "rejected_end"


def followup_node(state: ScreeningState) -> Dict[str, Any]:
    """Execute Follow-up Agent: only reachable after explicit doctor approval."""
    decision_data = state.get("decision_result") or {}
    review_data = state.get("doctor_review") or {}

    decision_obj = DecisionResult(**decision_data)
    review_obj = DoctorReviewResult(**review_data)

    result = run_followup(
        patient_id=state.get("patient_id", "PAT-DEFAULT"),
        village=state.get("village", "Unknown Village"),
        asha_id=state.get("asha_id", "ASHA-DEFAULT"),
        decision=decision_obj,
        doctor_review=review_obj,
    )

    return {
        "followup_result": result.model_dump(),
        "status": "FOLLOWUP_ACTIVE",
        "audit_trail": [
            f"[{datetime.now(timezone.utc).isoformat()}] Follow-up scheduled with ASHA {result.asha_assigned} for {result.next_visit_date}"
        ],
    }


def surveillance_node(state: ScreeningState) -> Dict[str, Any]:
    """Execute Surveillance Agent: record epidemiological telemetry."""
    village = state.get("village", "Unknown Village")
    patient_id = state.get("patient_id", "PAT-DEFAULT")

    result = run_surveillance(
        village=village,
        confirmed=True,
        patient_id=patient_id,
    )

    return {
        "surveillance_result": result.model_dump(),
        "status": "COMPLETED",
        "final_outcome": "TB Confirmed - Active in DOTS & Public Health Surveillance",
        "audit_trail": [
            f"[{datetime.now(timezone.utc).isoformat()}] Surveillance telemetry logged for {village} (Alert: {result.alert_level})"
        ],
    }


def rejected_node(state: ScreeningState) -> Dict[str, Any]:
    """Terminal node for cases not approved by Medical Officer."""
    return {
        "status": "CLOSED",
        "final_outcome": "Not TB - refer or re-screen",
        "audit_trail": [
            f"[{datetime.now(timezone.utc).isoformat()}] Case closed with status: Not TB - refer or re-screen"
        ],
    }


def create_screening_workflow(checkpointer: Optional[SqliteSaver] = None):
    """Build and compile the LangGraph StateGraph workflow for TB screening.

    Args:
        checkpointer: Optional SqliteSaver instance for persistent state and interrupt resumption.

    Returns:
        CompiledStateGraph: Executable LangGraph workflow.
    """
    workflow = StateGraph(ScreeningState)

    # Add all agent and gateway nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("imaging", imaging_node)
    workflow.add_node("lab_order", lab_order_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("doctor_review", doctor_review_node)
    workflow.add_node("followup", followup_node)
    workflow.add_node("surveillance", surveillance_node)
    workflow.add_node("rejected_end", rejected_node)

    # Flow: intake -> (imaging and a lab-order step inside decision run on parallel branches) -> decision
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "imaging")
    workflow.add_edge("intake", "lab_order")
    workflow.add_edge("imaging", "decision")
    workflow.add_edge("lab_order", "decision")

    # decision -> doctor_review
    workflow.add_edge("decision", "doctor_review")

    # doctor_review -> conditional branches: approved -> followup; rejected -> rejected_end
    workflow.add_conditional_edges(
        "doctor_review",
        route_after_doctor_review,
        {
            "followup": "followup",
            "rejected_end": "rejected_end",
        },
    )

    # if approved: followup -> surveillance -> END
    workflow.add_edge("followup", "surveillance")
    workflow.add_edge("surveillance", END)

    # if rejected: rejected_end -> END
    workflow.add_edge("rejected_end", END)

    return workflow.compile(checkpointer=checkpointer)


def get_workflow_with_sqlite(db_path: str = "data/tb_screening.db"):
    """Convenience initializer creating a persistent SQLite checkpointer.

    Args:
        db_path: Path to SQLite database file or ':memory:'.

    Returns:
        tuple (compiled_graph, connection, checkpointer)
    """
    if db_path != ":memory:":
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    graph = create_screening_workflow(checkpointer=checkpointer)
    return graph, conn, checkpointer
