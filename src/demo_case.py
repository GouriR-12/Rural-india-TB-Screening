"""Demonstration script: Run a dummy TB screening case end-to-end through LangGraph.

Shows:
1. Intake of ASHA worker screening notes (multilingual cues)
2. Parallel execution: Imaging Agent (CXR analysis) & Lab Order Dispatch
3. Decision Agent clinical synthesis & risk stratification
4. LangGraph interrupt pausing at the Mandatory Doctor Review Gate
5. SQLite checkpoint state inspection
6. Doctor review submission & resumption
7. Downstream routing to Follow-up Agent and Surveillance Agent
"""

import json
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.types import Command
from src.graph.workflow import get_workflow_with_sqlite


def run_dummy_demonstration():
    print("=" * 80)
    print("TB SCREENING MULTI-AGENT SYSTEM - END-TO-END DEMO EXECUTION")
    print("=" * 80)

    # 1. Initialize compiled workflow with SQLite checkpointing
    db_file = "data/demo_tb_screening.db"
    graph, conn, _ = get_workflow_with_sqlite(db_file)
    print(f"\n[INIT] SQLite checkpointer active at: {db_file}")

    case_id = "CASE-MAHA-2026-8801"
    config = {"configurable": {"thread_id": case_id}}

    dummy_input = {
        "case_id": case_id,
        "patient_id": "PAT-7742",
        "patient_name": "Ganesh Kamble",
        "age": 42,
        "gender": "M",
        "village": "Junnar",
        "asha_id": "ASHA-PUNE-012",
        "raw_notes": "Patient reports khansi (severe cough) for over 3 weeks, high bukhar in evenings, vajan kami, and occasional hemoptysis.",
        "symptoms": ["Cough > 2 weeks", "Fever", "Weight Loss", "Hemoptysis"],
        "duration_days": 24,
        "contact_history": True,
        "image_path": "data/sample_cxr_abnormal.png",
    }

    print(f"\n[PHASE 1] Starting Screening for Case: {case_id}")
    print(f"Patient: {dummy_input['patient_name']} (Age: {dummy_input['age']}, Village: {dummy_input['village']})")
    print(f"ASHA Notes: \"{dummy_input['raw_notes']}\"")

    # Step 1: Execute graph until interrupt
    print("\n[DISPATCH] Invoking LangGraph workflow (Intake -> Parallel Imaging/Lab -> Decision)...")
    first_pass = graph.invoke(dummy_input, config=config)

    if "__interrupt__" in first_pass:
        interrupt_info = first_pass["__interrupt__"][0].value
        print("\n" + "=" * 35 + " INTERRUPT REACHED " + "=" * 35)
        print(f"Gate Triggered : {interrupt_info['gate']}")
        print(f"Case ID        : {interrupt_info['case_id']}")
        print(f"Patient        : {interrupt_info['patient_name']} (Age: {interrupt_info['age']})")
        print(f"Risk Category  : {interrupt_info['risk_category']} (Score: {interrupt_info['tb_risk_score']})")
        print(f"Recommendation : {interrupt_info['preliminary_recommendation']}")
        print(f"Case Summary   : {interrupt_info['clinical_summary']}")
        print("=" * 89)

    # Inspect SQLite checkpointed state
    saved_state = graph.get_state(config)
    print(f"\n[CHECKPOINT] Workflow paused and safely checkpointed in SQLite.")
    print(f"Current State Status: {saved_state.values.get('status')}")
    print(f"Follow-up state present? {'Yes' if saved_state.values.get('followup_result') else 'No (Mandatory Gate enforced)'}")

    # Step 2: Doctor Review Submission (Human-in-the-loop)
    print("\n[PHASE 2] Medical Officer Reviews Case & Submits Approval...")
    doctor_action = {
        "approved": True,
        "doctor_id": "DR-ANAND-DESHMUKH",
        "doctor_notes": "CXR shows bilateral cavitation. Clinical picture strongly indicates smear-positive pulmonary tuberculosis. Initiate DOTS Category I immediately.",
        "final_diagnosis": "Microbiologically Confirmed Pulmonary Tuberculosis",
        "prescription": "Category 1 ATT (4FDC daily: Isoniazid 75mg, Rifampicin 150mg, Pyrazinamide 400mg, Ethambutol 275mg)",
    }
    print(f"Doctor ID     : {doctor_action['doctor_id']}")
    print(f"Decision      : {'APPROVED' if doctor_action['approved'] else 'REJECTED'}")
    print(f"Doctor Notes  : {doctor_action['doctor_notes']}")
    print(f"Prescription  : {doctor_action['prescription']}")

    # Resume graph execution
    print("\n[RESUME] Sending Command(resume=...) to LangGraph workflow...")
    resumed_pass = graph.invoke(Command(resume=doctor_action), config=config)

    print("\n" + "=" * 35 + " WORKFLOW COMPLETED " + "=" * 35)
    print(f"Final Status       : {resumed_pass.get('status')}")
    print(f"Final Outcome      : {resumed_pass.get('final_outcome')}")

    followup = resumed_pass.get("followup_result", {})
    print(f"\n[FOLLOW-UP AGENT OUTPUT]")
    print(f"- Assigned ASHA     : {followup.get('asha_assigned')}")
    print(f"- Next Visit Date   : {followup.get('next_visit_date')}")
    print(f"- Dropout Risk Level: {followup.get('dropout_risk_level')} (Score: {followup.get('dropout_risk_score')})")
    print(f"- DOTS Center       : {followup.get('dots_center')}")
    print(f"- Adherence Plan    : {followup.get('adherence_plan')}")

    surv = resumed_pass.get("surveillance_result", {})
    print(f"\n[SURVEILLANCE AGENT OUTPUT]")
    print(f"- Village Monitored : {surv.get('village')}")
    print(f"- Cluster Detected  : {surv.get('cluster_detected')}")
    print(f"- Alert Level       : {surv.get('alert_level')}")
    print(f"- Telemetry Remarks : {surv.get('notes')}")

    print("\n[AUDIT TRAIL LOG]")
    for idx, entry in enumerate(resumed_pass.get("audit_trail", []), 1):
        print(f"  {idx}. {entry}")

    print("\n" + "=" * 80)
    conn.close()


if __name__ == "__main__":
    run_dummy_demonstration()
