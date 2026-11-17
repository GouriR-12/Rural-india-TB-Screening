"""Doctor Review Dashboard UI Components.

Provides interface widgets for the Mandatory Doctor Review Gate, including
a prominent warning badge when imaging model provenance is synthetic_demo.

DISCLAIMER:
Research prototype, not clinically validated, not for real diagnosis.
"""

from typing import Any, Dict, Optional


def get_provenance_badge(provenance: str) -> Dict[str, str]:
    """Return styling and text metadata for model provenance badges."""
    if provenance == "synthetic_demo":
        return {
            "level": "warning",
            "title": "DEMO MODEL (SMOKE-TEST FIXTURE)",
            "message": "⚠️ DEMO MODEL: Trained on synthetic cartoon images (synthetic_demo_cxr). Metrics are not evidence of performance and have zero clinical validity.",
            "badge_color": "#ff4b4b",
            "badge_text": "⚠️ SYNTHETIC DEMO MODEL - NOT CLINICALLY VALIDATED",
            "is_synthetic": True,
        }
    else:
        return {
            "level": "info",
            "title": "REAL PUBLIC DATASET MODEL",
            "message": "ℹ️ Model trained on real public benchmark CXR datasets (Shenzhen/Montgomery). Preliminary research prototype - Medical Officer confirmation mandatory.",
            "badge_color": "#0068c9",
            "badge_text": "PUBLIC REFERENCE DATASET (SHENZHEN/MONTGOMERY)",
            "is_synthetic": False,
        }


def format_doctor_review_summary(gate_payload: Dict[str, Any]) -> str:
    """Format markdown briefing summary for the Medical Officer review gate."""
    case_id = gate_payload.get("case_id", "Unknown")
    patient = gate_payload.get("patient_name", "Anonymous")
    age = gate_payload.get("age", "N/A")
    village = gate_payload.get("village", "N/A")
    risk_cat = gate_payload.get("risk_category", "Unknown")
    risk_score = gate_payload.get("tb_risk_score", 0.0)
    summary = gate_payload.get("clinical_summary", "No summary available.")
    recommendation = gate_payload.get("preliminary_recommendation", "Review case.")

    imaging_data = gate_payload.get("imaging_result") or {}
    provenance = gate_payload.get("model_provenance") or imaging_data.get("model_provenance", "synthetic_demo")
    badge_info = get_provenance_badge(provenance)

    lines = [
        f"## Medical Officer Review Gate - Case: `{case_id}`",
        f"**Patient**: {patient} (Age: {age}, Village: {village})",
        f"**Risk Stratification**: **{risk_cat}** (Composite TB Score: `{risk_score:.4f}`)",
        "",
        "### Imaging Findings & Provenance",
        f"> **Provenance Status**: <span style='background-color:{badge_info['badge_color']}; color:white; padding:3px 8px; border-radius:4px; font-weight:bold;'>{badge_info['badge_text']}</span>",
    ]

    if badge_info["is_synthetic"]:
        lines.append(f"> ⚠️ **ATTENTION DOCTOR**: {badge_info['message']}")

    tb_prob = imaging_data.get("tb_probability")
    if tb_prob is not None:
        lines.append(f"- **CNN TB Probability**: `{tb_prob:.4f}`")
        lines.append(f"- **Findings**: {imaging_data.get('findings', 'N/A')}")
        lines.append(f"- **Confidence Note**: {imaging_data.get('confidence_note', 'N/A')}")
        if imaging_data.get("heatmap_path"):
            lines.append(f"- **Grad-CAM Saliency Map**: `{imaging_data.get('heatmap_path')}`")

    lines.extend([
        "",
        "### Clinical Synthesis",
        f"{summary}",
        "",
        f"**Preliminary Recommendation**: {recommendation}",
        "",
        "---",
        "*Mandatory Sign-off Requirement: No case can initiate treatment or proceed to follow-up without explicit Doctor Approval.*",
        "*Research prototype, not clinically validated, not for real diagnosis.*"
    ])

    return "\n".join(lines)


def render_doctor_review_gate(gate_payload: Dict[str, Any]):
    """Streamlit render function for the Doctor Review Gate."""
    try:
        import streamlit as st
    except ImportError:
        # Streamlit not in current runtime scope; return markdown
        return format_doctor_review_summary(gate_payload)

    imaging_data = gate_payload.get("imaging_result") or {}
    provenance = gate_payload.get("model_provenance") or imaging_data.get("model_provenance", "synthetic_demo")
    badge_info = get_provenance_badge(provenance)

    st.subheader(f"Medical Officer Review Gate: {gate_payload.get('case_id')}")

    # Visible Warning Badge when provenance is synthetic_demo
    if badge_info["is_synthetic"]:
        st.error(
            f"### ⚠️ {badge_info['badge_text']}\n\n"
            f"{badge_info['message']}\n\n"
            "**Real-data training is currently BLOCKED (needs Kaggle download by user).**"
        )
    else:
        st.info(badge_info["message"])

    st.markdown(format_doctor_review_summary(gate_payload), unsafe_allow_html=True)

