"""End-to-end test suite for LangGraph TB Screening Workflow.

Validates:
1. Intake -> Parallel (Imaging & Lab Order) -> Decision -> Doctor Review Gate (Interrupt)
2. Case Resumption via SQLite Checkpointing with Doctor Approval -> Follow-up -> Surveillance
3. Case Resumption via SQLite Checkpointing with Doctor Rejection -> Status "Not TB - refer or re-screen"
4. Verification that Follow-up is NEVER reached when doctor review rejects the case.
"""

import pytest
from langgraph.types import Command
from src.graph.workflow import get_workflow_with_sqlite


def test_workflow_end_to_end_approval():
    """Test full workflow lifecycle with human-in-the-loop doctor approval."""
    # Initialize in-memory SQLite checkpointer
    graph, conn, _ = get_workflow_with_sqlite(":memory:")

    case_id = "CASE-IND-2026-001"
    thread_config = {"configurable": {"thread_id": case_id}}

    initial_input = {
        "case_id": case_id,
        "patient_id": "PAT-9901",
        "patient_name": "Ramesh Patil",
        "age": 45,
        "gender": "M",
        "village": "Ralegan Siddhi",
        "asha_id": "ASHA-MH-042",
        "raw_notes": "Patient reports severe khansi (cough) for 3 weeks with evening fever and vajan kami (weight loss).",
        "symptoms": ["Cough > 2 weeks", "Fever", "Weight Loss"],
        "duration_days": 21,
        "contact_history": True,
        "image_path": "data/sample_cxr_abnormal.png",
    }

    # Step 1: Run workflow until Doctor Review Gate interrupt
    first_pass = graph.invoke(initial_input, config=thread_config)

    # Verify paused state and interrupt payload
    assert "__interrupt__" in first_pass, "Workflow must pause at doctor review gate"
    interrupts = first_pass["__interrupt__"]
    assert len(interrupts) > 0
    gate_data = interrupts[0].value
    assert gate_data["gate"] == "MANDATORY_DOCTOR_REVIEW_GATE"
    assert gate_data["case_id"] == case_id
    assert gate_data["risk_category"] == "High"
    assert "model_provenance" in gate_data
    assert gate_data["model_provenance"] in ["synthetic_demo", "real_public_data"]

    # Verify that before resume, followup and surveillance have NOT run
    current_state = graph.get_state(thread_config)
    state_values = current_state.values
    assert state_values.get("status") == "AWAITING_DOCTOR_REVIEW"
    assert "imaging_result" in state_values and state_values["imaging_result"] is not None
    assert state_values["imaging_result"]["model_provenance"] in ["synthetic_demo", "real_public_data"]
    assert "followup_result" not in state_values or state_values.get("followup_result") is None
    assert "surveillance_result" not in state_values or state_values.get("surveillance_result") is None

    # Step 2: Doctor reviews and approves the case
    doctor_approval = {
        "approved": True,
        "doctor_id": "DR-KULKARNI-PHC",
        "doctor_notes": "Clinical and radiographic findings indicate active microbiologically suspected pulmonary TB. Initiating ATT regimen.",
        "final_diagnosis": "Confirmed Pulmonary Tuberculosis",
        "prescription": "4FDC (Rifampicin, Isoniazid, Pyrazinamide, Ethambutol) Intensive Phase",
    }

    resumed_pass = graph.invoke(Command(resume=doctor_approval), config=thread_config)

    # Step 3: Verify downstream execution completed through Follow-up and Surveillance
    assert resumed_pass.get("status") == "COMPLETED"
    assert "TB Confirmed" in resumed_pass.get("final_outcome", "")

    # Verify Doctor Review details
    doc_rev = resumed_pass.get("doctor_review")
    assert doc_rev["approved"] is True
    assert doc_rev["doctor_id"] == "DR-KULKARNI-PHC"

    # Verify Follow-up Agent ran
    followup = resumed_pass.get("followup_result")
    assert followup is not None
    assert followup["asha_assigned"] == "ASHA-MH-042"
    assert followup["dropout_risk_level"] in ["High", "Medium", "Low"]
    assert "Nikshay" in followup["adherence_plan"]

    # Verify Surveillance Agent ran
    surv = resumed_pass.get("surveillance_result")
    assert surv is not None
    assert surv["village"] == "Ralegan Siddhi"
    assert surv["cluster_detected"] is True

    # Verify Audit Trail
    audit = resumed_pass.get("audit_trail", [])
    assert any("Intake completed" in item for item in audit)
    assert any("CXR analysis completed" in item for item in audit)
    assert any("Lab order dispatched" in item for item in audit)
    assert any("Doctor Gate APPROVED" in item for item in audit)
    assert any("Follow-up scheduled" in item for item in audit)
    assert any("Surveillance telemetry logged" in item for item in audit)

    conn.close()


def test_workflow_end_to_end_rejection():
    """Test full workflow lifecycle with human-in-the-loop doctor rejection (Not TB)."""
    graph, conn, _ = get_workflow_with_sqlite(":memory:")

    case_id = "CASE-IND-2026-002"
    thread_config = {"configurable": {"thread_id": case_id}}

    initial_input = {
        "case_id": case_id,
        "patient_id": "PAT-9902",
        "patient_name": "Sunita Devi",
        "age": 28,
        "gender": "F",
        "village": "Pargaon",
        "asha_id": "ASHA-MH-018",
        "raw_notes": "Mild allergic cough for 3 days, no fever, no hemoptysis.",
        "symptoms": ["Mild Cough"],
        "duration_days": 3,
        "contact_history": False,
        "image_path": "data/sample_cxr_normal.png",
    }

    # Run until interrupt
    first_pass = graph.invoke(initial_input, config=thread_config)
    assert "__interrupt__" in first_pass

    # Doctor reviews and rejects TB diagnosis
    doctor_rejection = {
        "approved": False,
        "doctor_id": "DR-JOSHI-PHC",
        "doctor_notes": "Upper respiratory tract infection / seasonal allergy. CXR is clear. Sputum not indicated.",
        "final_diagnosis": "Acute Bronchitis (Non-TB)",
    }

    resumed_pass = graph.invoke(Command(resume=doctor_rejection), config=thread_config)

    # Verification: Must end with status 'Not TB - refer or re-screen'
    assert resumed_pass.get("status") == "CLOSED"
    assert resumed_pass.get("final_outcome") == "Not TB - refer or re-screen"

    # Mandatory Rule 1: No case can reach Follow-up without doctor approval
    assert resumed_pass.get("followup_result") is None
    assert resumed_pass.get("surveillance_result") is None

    conn.close()
