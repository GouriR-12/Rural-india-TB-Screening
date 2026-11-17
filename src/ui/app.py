"""Rural India Multi-Agent Tuberculosis (TB) Screening Dashboard.

Role-based Streamlit application with role isolation for Frontline ASHA Workers
and Primary Healthcare Medical Officers.

DISCLAIMER:
Research prototype, not clinically validated, not for real diagnosis.
"""

import sys
from pathlib import Path

# Project root inserted into sys.path before any src. imports
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui.doctor_dashboard import get_provenance_badge
from src.ui.service import (
    DEFAULT_DB_PATH,
    MANDATORY_DISCLAIMER,
    build_final_report,
    get_case,
    get_cases_for_asha,
    list_cases,
    llm_mode,
    review_case,
    seed_demo_data,
    submit_case,
)

# Geographic coordinates for rural Maharashtra PHCs and villages
MAHARASHTRA_VILLAGE_COORDS = {
    "Junnar": {"lat": 19.2065, "lon": 73.8767},
    "Shirur": {"lat": 18.8256, "lon": 74.3789},
    "Khed": {"lat": 18.8524, "lon": 73.9056},
    "Ambegaon": {"lat": 19.0167, "lon": 73.8167},
    "Baramati": {"lat": 18.1517, "lon": 74.5770},
    "Parner": {"lat": 19.0028, "lon": 74.4378},
    "Sangamner": {"lat": 19.5772, "lon": 74.2144},
    "Karad": {"lat": 17.2889, "lon": 74.1833},
    "Dindori": {"lat": 20.2017, "lon": 73.8344},
    "Barshi": {"lat": 18.2333, "lon": 75.6833},
    "Kinwat": {"lat": 19.6200, "lon": 78.2000},
    "Dharni": {"lat": 21.4900, "lon": 77.0300},
    "Ralegan Siddhi": {"lat": 18.9167, "lon": 74.4167},
    "Pargaon": {"lat": 18.8000, "lon": 74.2500},
}

DEFAULT_MAHA_LAT = 18.5204
DEFAULT_MAHA_LON = 73.8567

def apply_theme():
    """Inject DiagNova-style CSS: white, navy text, indigo-purple gradient accents."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp input, .stApp textarea, .stApp button { font-family: 'Inter', 'Segoe UI', sans-serif; }
        [data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
        }
        .stApp { background: #ffffff; }
        [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
        .block-container { padding-top: 2.5rem; max-width: 1200px; }

        /* Headings */
        h1, h2, h3, h4 { color: #1e1b4b; font-weight: 700; letter-spacing: -0.01em; }
        h1 { font-weight: 800; }
        p, li, label, .stMarkdown { color: #334155; }

        /* Sidebar */
        [data-testid="stSidebar"] { background: #f5f3ff; border-right: 1px solid #e5e7eb; }

        /* Primary buttons: indigo -> purple gradient, rounded */
        [data-testid="stBaseButton-primary"],
        [data-testid="stBaseButton-primaryFormSubmit"],
        button[kind="primary"], button[kind="primaryFormSubmit"] {
            background: linear-gradient(90deg, #4f46e5 0%, #a855f7 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.2rem !important;
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.25);
        }
        [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stBaseButton-primaryFormSubmit"]:hover { filter: brightness(1.08); }

        /* Secondary buttons: white with navy outline */
        [data-testid="stBaseButton-secondary"], button[kind="secondary"] {
            background: #ffffff !important;
            color: #1e1b4b !important;
            border: 1.5px solid #1e1b4b !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }

        /* Inputs */
        [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
            border-radius: 12px !important;
            background: #ffffff !important;
            border: 1px solid #d9d6f5 !important;
        }

        /* Cards: forms, metrics, expanders, alerts, dataframes */
        [data-testid="stForm"], [data-testid="stMetric"], [data-testid="stExpander"],
        [data-testid="stDataFrame"], [data-testid="stAlert"] {
            border-radius: 16px !important;
        }
        [data-testid="stForm"], [data-testid="stExpander"] {
            background: #ffffff;
            border: 1px solid #e5e7eb !important;
            box-shadow: 0 10px 30px rgba(30, 27, 75, 0.06);
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 16px 18px;
            box-shadow: 0 10px 30px rgba(30, 27, 75, 0.06);
        }
        [data-testid="stMetricValue"] { color: #4f46e5; font-weight: 800; }

        /* Tabs */
        .stTabs [data-baseweb="tab"] { font-weight: 600; color: #475569; }
        .stTabs [aria-selected="true"] { color: #4f46e5 !important; }
        .stTabs [data-baseweb="tab-highlight"] { background-color: #4f46e5 !important; }

        /* Hero + pills (used by login page) */
        .hero-brand { color: #4f46e5; font-weight: 700; font-size: 1.05rem; margin-bottom: 0.4rem; }
        .hero-title { color: #1e1b4b; font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin: 0 0 0.8rem 0; }
        .hero-sub { color: #475569; font-size: 1.1rem; max-width: 640px; margin-bottom: 1.5rem; }
        .hero-wrap {
            background: radial-gradient(circle at 85% 20%, rgba(168,85,247,0.18), transparent 55%),
                        radial-gradient(circle at 70% 90%, rgba(79,70,229,0.12), transparent 50%);
            border-radius: 24px; padding: 2rem 2rem 1rem 2rem; margin-bottom: 1rem;
        }
        [data-testid="stHeader"] { background: #ffffff !important; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        [data-testid="stHeading"], [data-testid="stHeading"] * { color: #1e1b4b !important; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] p { color: #1e1b4b; }
        [data-testid="stBaseButton-primary"] *,
        [data-testid="stBaseButton-primaryFormSubmit"] * { color: #ffffff !important; }
        [data-baseweb="base-input"], [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea, [data-baseweb="select"] div {
        background-color: #ffffff !important; color: #1e1b4b !important;
        }
        input:disabled { -webkit-text-fill-color: #64748b !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_footer():
    """Render small grey caption disclaimer at the bottom of each page."""
    st.markdown("---")
    st.caption("Research prototype, not clinically validated, not for real diagnosis.")


def render_login():
    """Render Login Page with fixed demo accounts."""
    st.markdown(
        "<div class='hero-wrap'>"
        "<div class='hero-brand'>🩺 Rural India TB Screening</div>"
        "<div class='hero-title'>AI assists ASHA workers and doctors in faster, confident TB screening.</div>"
        "<div class='hero-sub'>Frontline workers submit cases, the AI pipeline prepares the clinical picture, "
        "and a doctor always makes the final call.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password").strip()
        login_btn = st.form_submit_button("Log In", type="primary", use_container_width=True)

    if login_btn:
        if username == "asha" and password == "asha123":
            st.session_state["authenticated"] = True
            st.session_state["role"] = "asha"
            st.session_state["username"] = "asha"
            st.session_state["asha_id"] = "ASHA-MH-042"
            st.rerun()
        elif username == "doctor" and password == "doctor123":
            st.session_state["authenticated"] = True
            st.session_state["role"] = "doctor"
            st.session_state["username"] = "doctor"
            st.rerun()
        else:
            st.error("Invalid username or password. Please use fixed demo accounts.")

    render_footer()


def render_asha_portal():
    """Render ASHA Worker interface with tabs 'New Screening' and 'My Cases'."""
    st.title("👩‍⚕️ Frontline ASHA Worker Screening Portal")
    asha_id = st.session_state.get("asha_id", "ASHA-MH-042")

    tab_new, tab_my = st.tabs(["New Screening", "My Cases"])

    with tab_new:
        st.subheader("1. Patient Profile & Community Demographics")

        # Initialize CXR selection state
        if "selected_cxr_path" not in st.session_state:
            st.session_state["selected_cxr_path"] = "data/sample_cxr_abnormal.png"

        c1, c2, c3 = st.columns(3)
        with c1:
            patient_name = st.text_input("Patient Full Name", value="Ramesh Patil")
            age = st.number_input("Age (Years)", min_value=1, max_value=115, value=45)
        with c2:
            gender = st.selectbox("Biological Sex", options=["M", "F", "Other"], index=0)
            worker_id = st.text_input("Frontline ASHA Worker ID", value=asha_id, disabled=True)
        with c3:
            common_villages = list(MAHARASHTRA_VILLAGE_COORDS.keys())
            village = st.selectbox("Village / Health Sub-Center", options=common_villages, index=0)

        st.subheader("2. Symptom Checklist & Clinical Presentation")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sym_cough_2wks = st.checkbox("Persistent Cough ≥ 2 weeks (NTEP Indicator)", value=True)
            duration_days = st.number_input(
                "Total Cough Duration (Days)", min_value=0, max_value=365, value=21 if sym_cough_2wks else 3
            )
            sym_fever = st.checkbox("Recurrent / Evening Rise of Fever", value=True)
            sym_sweats = st.checkbox("Drenching Night Sweats", value=False)
        with col_s2:
            sym_weight_loss = st.checkbox("Unexplained Weight Loss", value=True)
            sym_hemoptysis = st.checkbox("Hemoptysis (Blood in sputum/cough - URGENT)", value=False)
            sym_contact = st.checkbox("Household or Close Contact with Active TB Patient", value=True)

        raw_notes = st.text_area(
            "ASHA Worker Clinical Observations (Supports Hindi, Marathi, English, or Hinglish)",
            value="Patient reports severe khansi for 3 weeks with evening bukhar and vajan kami. Household contact positive.",
            height=90,
        )

        st.subheader("3. Chest Radiograph (CXR)")
        cxr_col1, cxr_col2 = st.columns([3, 2])

        with cxr_col1:
            uploaded_file = st.file_uploader(
                "Upload Patient Chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                os.makedirs(os.path.join("data", "cxr", "uploads"), exist_ok=True)
                uploaded_path = os.path.join("data", "cxr", "uploads", uploaded_file.name)
                with open(uploaded_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state["selected_cxr_path"] = uploaded_path
                st.success(f"Uploaded: {uploaded_file.name}")

            st.write("Or select benchmark radiograph:")
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button("Use Sample CXR: Abnormal", use_container_width=True):
                    st.session_state["selected_cxr_path"] = "data/sample_cxr_abnormal.png"
            with bcol2:
                if st.button("Use Sample CXR: Normal", use_container_width=True):
                    st.session_state["selected_cxr_path"] = "data/sample_cxr_normal.png"

        with cxr_col2:
            selected_path = st.session_state.get("selected_cxr_path")
            if selected_path and os.path.exists(selected_path):
                st.image(selected_path, caption=f"Selected: {os.path.basename(selected_path)}", width=220)
            else:
                st.info("No radiograph selected.")

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.button("🚀 Submit Screening Case", type="primary", use_container_width=True)

        if submit_btn:
            symptoms_list = []
            if sym_cough_2wks or duration_days >= 14:
                symptoms_list.append("Cough > 2 weeks")
            elif duration_days > 0:
                symptoms_list.append("Cough")
            if sym_fever:
                symptoms_list.append("Fever")
            if sym_weight_loss:
                symptoms_list.append("Weight Loss")
            if sym_sweats:
                symptoms_list.append("Night Sweats")
            if sym_hemoptysis:
                symptoms_list.append("Hemoptysis")

            form_data = {
                "patient_name": patient_name.strip() or "Anonymous",
                "age": age,
                "gender": gender,
                "village": village,
                "asha_id": asha_id,
                "raw_notes": raw_notes.strip(),
                "symptoms": symptoms_list,
                "duration_days": int(duration_days),
                "contact_history": bool(sym_contact),
            }

            with st.spinner("Submitting screening case..."):
                case_id = submit_case(form_data, cxr_path=st.session_state.get("selected_cxr_path"))

            # After submit show ONLY: Case submitted. ID: X. Status: Not reviewed yet.
            # Never show AI summary, risk or imaging to ASHA.
            st.success(f"Case submitted. ID: {case_id}. Status: Not reviewed yet.")

    with tab_my:
        st.subheader("My Registered Cases")
        st.caption(f"Screening cases registered by frontline worker `{asha_id}`.")

        # Uses only get_cases_for_asha
        my_cases = get_cases_for_asha(asha_id)

        if not my_cases:
            st.info("No cases registered yet under your ASHA ID.")
        else:
            df_my = pd.DataFrame(my_cases)
            df_my.columns = ["Case ID", "Patient", "Village", "Date", "Status"]
            def _pill(v):
                colors = {
                    "Approved": "background-color:#dcfce7;color:#166534;font-weight:600",
                    "Rejected": "background-color:#fee2e2;color:#991b1b;font-weight:600",
                    "Not reviewed yet": "background-color:#fef3c7;color:#92400e;font-weight:600",
                }
                return colors.get(v, "")

            st.dataframe(df_my.style.map(_pill, subset=["Status"]), use_container_width=True, hide_index=True)


def render_doctor_portal():
    """Render Medical Officer interface with tabs: Case Queue, Final Reports, Surveillance, Demo Data."""
    st.title("👨‍⚕️ Medical Officer Clinical Portal")

    tab_queue, tab_reports, tab_surv, tab_demo = st.tabs([
        "Case Queue",
        "Final Reports",
        "Surveillance",
        "Demo Data",
    ])

    with tab_queue:
        st.subheader("Pending Clinical Review Queue")
        pending_cases = list_cases(status="PENDING_REVIEW")

        if not pending_cases:
            st.info("No cases currently pending review.")
        else:
            case_options = [
                f"{c['id']} — {c.get('village', 'Unknown')} ({c.get('created_at', '')[:10]})"
                for c in pending_cases
            ]
            selected_option = st.selectbox("Select Case to Review", options=case_options, index=0)
            selected_id = selected_option.split(" — ")[0]

            case_record = get_case(selected_id)
            if not case_record:
                st.error(f"Could not load case {selected_id}.")
            else:
                state = json.loads(case_record["state_json"])
                intake = state.get("intake_result") or {}
                imaging = state.get("imaging_result") or {}
                decision = state.get("decision_result") or {}
                provenance = imaging.get("model_provenance", "synthetic_demo")

                st.markdown(f"### Case `{selected_id}`: {state.get('patient_name')} ({state.get('age')}y, {state.get('gender')})")

                # Provenance Warning Badge
                badge_info = get_provenance_badge(provenance)
                st.markdown(
                    f"<div style='background-color:{badge_info['badge_color']}; color:white; padding:6px 14px; border-radius:6px; font-weight:bold; display:inline-block; margin-bottom:12px; font-size: 0.95rem;'>"
                    f"{badge_info['badge_text']}</div>",
                    unsafe_allow_html=True,
                )
                if badge_info.get("is_synthetic"):
                    st.warning(badge_info["message"])

                # Patient & Symptom Summary
                col_info, col_img = st.columns([3, 2])
                with col_info:
                    st.markdown("#### Patient Details & Frontline Intake")
                    st.markdown(f"- **Village**: {case_record.get('village')}")
                    st.markdown(f"- **Assigned ASHA**: {state.get('asha_id')}")
                    st.markdown(f"- **Cough Duration**: {state.get('duration_days')} days")
                    st.markdown(f"- **Contact History**: {'Positive' if state.get('contact_history') else 'Negative'}")
                    st.markdown(f"- **Symptoms**: {', '.join(state.get('symptoms') or [])}")
                    st.markdown(f"- **ASHA Notes**: *\"{state.get('raw_notes')}\"*")

                    st.markdown("#### Clinical Risk Synthesis")
                    st.markdown(f"- **Risk Stratification**: **{decision.get('risk_category')}** (Score: `{decision.get('tb_risk_score', 0.0):.4f}`)")
                    st.markdown(f"- **Preliminary AI Recommendation**: {decision.get('preliminary_recommendation')}")

                with col_img:
                    st.markdown("#### Radiographic Findings")
                    st.markdown(f"- **TB Probability**: `{imaging.get('tb_probability', 0.0):.4f}`")
                    st.markdown(f"- **Abnormality**: {'Detected' if imaging.get('abnormality_detected') else 'Clear'}")
                    st.markdown(f"- **Findings**: {imaging.get('findings', 'N/A')}")

                    # Original CXR
                    cxr_img_path = state.get("image_path")
                    if cxr_img_path and os.path.exists(cxr_img_path):
                        st.image(cxr_img_path, caption=f"CXR: {os.path.basename(cxr_img_path)}", width=200)

                    # Grad-CAM Heatmap if available
                    heatmap_path = imaging.get("heatmap_path")
                    if heatmap_path and os.path.exists(heatmap_path):
                        st.image(heatmap_path, caption="Grad-CAM Saliency Heatmap", width=200)

                with st.expander("Full Clinical AI Narrative Summary", expanded=False):
                    st.markdown(decision.get("clinical_summary", "No summary available."))

                st.markdown("---")
                st.subheader("Medical Officer Review Decision & Sign-off")
                doctor_comment = st.text_area(
                    "Doctor Clinical Notes / Diagnosis Remarks",
                    value="",
                    placeholder="Enter your clinical remarks (optional)",
                    height=80,
                )
                final_comment = doctor_comment.strip() or "No remarks recorded."

                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    if st.button("✅ APPROVE CASE (Initiate Follow-up & DOTS)", type="primary", use_container_width=True):
                        with st.spinner("Finalizing approval and dispatching Follow-up agent..."):
                            review_case(
                                case_id=selected_id,
                                decision="APPROVED",
                                comment=final_comment,
                                doctor_id="DR-PHC-OFFICER",
                            )
                        st.success(f"Case `{selected_id}` APPROVED! Follow-up adherence plan and surveillance scheduled.")
                        st.rerun()

                with col_act2:
                    if st.button("❌ REJECT CASE (Non-TB Illness)", use_container_width=True):
                        with st.spinner("Recording rejection..."):
                            review_case(
                                case_id=selected_id,
                                decision="REJECTED",
                                comment=final_comment,
                                doctor_id="DR-PHC-OFFICER",
                            )
                        st.warning(f"Case `{selected_id}` REJECTED. Case closed as Non-TB without initiating follow-up.")
                        st.rerun()

    with tab_reports:
        st.subheader("Clinical Final Reports Archive")
        all_cases = list_cases()

        if not all_cases:
            st.info("No cases found in database.")
        else:
            report_options = [
                f"{c['id']} — {c.get('village', 'Unknown')} [{c.get('status')}]"
                for c in all_cases
            ]
            rep_selected = st.selectbox("Select Case to Generate / View Final Report", options=report_options)
            target_case_id = rep_selected.split(" — ")[0]

            report_markdown = build_final_report(target_case_id)

            st.download_button(
                label=f"📥 Download Markdown Report ({target_case_id})",
                data=report_markdown,
                file_name=f"{target_case_id}_final_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

            with st.expander("Preview Final Report", expanded=True):
                st.markdown(report_markdown)

    with tab_surv:
        st.subheader("Public Health Surveillance & Telemetry")
        all_cases = list_cases()

        if not all_cases:
            st.info("No surveillance telemetry recorded yet.")
        else:
            total_screened = len(all_cases)
            approved_cases = sum(1 for c in all_cases if c.get("status") == "APPROVED")
            rejected_cases = sum(1 for c in all_cases if c.get("status") == "REJECTED")
            pending_cases = sum(1 for c in all_cases if c.get("status") == "PENDING_REVIEW")

            presumptive_tb = 0
            for c in all_cases:
                try:
                    st_data = json.loads(c.get("state_json", "{}"))
                    dec = st_data.get("decision_result", {})
                    img = st_data.get("imaging_result", {})
                    if dec.get("risk_category") in ("High", "Moderate") or img.get("tb_probability", 0.0) >= 0.50:
                        presumptive_tb += 1
                except Exception:
                    pass

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Total Screened", total_screened)
            with m2:
                st.metric("Presumptive TB", presumptive_tb)
            with m3:
                st.metric("Doctor Approved", approved_cases)
            with m4:
                st.metric("Doctor Rejected", rejected_cases)
            with m5:
                st.metric("Pending Review", pending_cases)

            st.markdown("---")

            st.markdown("#### Screening Volume Over Time")
            date_records = []
            for c in all_cases:
                created = c.get("created_at") or ""
                date_str = created[:10] if len(created) >= 10 else datetime.now().strftime("%Y-%m-%d")
                date_records.append({
                    "Date": date_str,
                    "Status": c.get("status", "UNKNOWN"),
                    "Village": c.get("village", "Unknown"),
                })

            df_dates = pd.DataFrame(date_records)
            if not df_dates.empty:
                df_ts = df_dates.groupby(["Date", "Status"]).size().reset_index(name="Case Count")
                df_ts = df_ts.sort_values("Date")

                fig_ts = px.line(
                    df_ts,
                    x="Date",
                    y="Case Count",
                    color="Status",
                    markers=True,
                    title="Screening Submissions & Case Outcomes Over Time",
                    color_discrete_map={
                        "APPROVED": "#28a745",
                        "REJECTED": "#dc3545",
                        "PENDING_REVIEW": "#ffc107",
                    },
                )
                fig_ts.update_layout(xaxis_title="Date", yaxis_title="Number of Cases", hovermode="x unified")
                st.plotly_chart(fig_ts, use_container_width=True)

            st.markdown("#### Village Geographic Distribution & Screening Density")
            col_map, col_table = st.columns([3, 2])

            village_counts = df_dates.groupby("Village").size().reset_index(name="Cases")
            approved_by_village = (
                df_dates[df_dates["Status"] == "APPROVED"]
                .groupby("Village")
                .size()
                .reset_index(name="Approved")
            )
            village_summary = pd.merge(village_counts, approved_by_village, on="Village", how="left").fillna(0)
            village_summary["Approved"] = village_summary["Approved"].astype(int)

            map_rows = []
            for _, row in village_summary.iterrows():
                vname = row["Village"]
                coords = MAHARASHTRA_VILLAGE_COORDS.get(vname, {"lat": DEFAULT_MAHA_LAT, "lon": DEFAULT_MAHA_LON})
                map_rows.append({
                    "village": vname,
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                    "Cases": int(row["Cases"]),
                    "Approved": int(row["Approved"]),
                })

            df_map = pd.DataFrame(map_rows)

            with col_map:
                if not df_map.empty:
                    if hasattr(px, "scatter_map"):
                        fig_map = px.scatter_map(
                            df_map,
                            lat="lat",
                            lon="lon",
                            size="Cases",
                            color="Approved",
                            hover_name="village",
                            hover_data={"lat": False, "lon": False, "Cases": True, "Approved": True},
                            color_continuous_scale="Reds",
                            size_max=30,
                            zoom=6.5,
                            title="Screening Cases & TB Approvals by Village",
                        )
                        fig_map.update_layout(map_style="open-street-map", margin={"r": 0, "t": 35, "l": 0, "b": 0})
                        st.plotly_chart(fig_map, use_container_width=True)
                    elif hasattr(px, "scatter_mapbox"):
                        fig_map = px.scatter_mapbox(
                            df_map,
                            lat="lat",
                            lon="lon",
                            size="Cases",
                            color="Approved",
                            hover_name="village",
                            hover_data={"lat": False, "lon": False, "Cases": True, "Approved": True},
                            color_continuous_scale="Reds",
                            size_max=30,
                            zoom=6.5,
                            title="Screening Cases & TB Approvals by Village",
                        )
                        fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 35, "l": 0, "b": 0})
                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.map(df_map, latitude="lat", longitude="lon", size="Cases")

            with col_table:
                st.markdown("##### Village Screening Breakdown")
                st.dataframe(village_summary.sort_values("Cases", ascending=False), use_container_width=True, hide_index=True)

    with tab_demo:
        st.subheader("Demo Data Management")
        st.caption("Generate synthetic screening cases for software demonstration, performance benchmarking, and training.")
        if st.button("🎲 Load 50 Demo Cases", type="primary", use_container_width=True):
            with st.spinner("Seeding 50 synthetic demo cases..."):
                ids = seed_demo_data(n=50)
            st.success(f"Successfully loaded {len(ids)} synthetic demo cases into the database!")
            st.rerun()


def main():
    st.set_page_config(
        page_title="Rural India TB Screening & Surveillance",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    apply_theme()

    # Initialize authentication session state
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "role" not in st.session_state:
        st.session_state["role"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None

    if not st.session_state["authenticated"]:
        render_login()
        return

    # Authenticated sidebar navigation
    st.sidebar.title("🩺 TB Screening System")
    username = st.session_state.get("username", "")
    role = st.session_state.get("role", "")
    role_label = "Frontline ASHA Worker" if role == "asha" else "Medical Officer"
    st.sidebar.markdown(f"👤 **Logged in**: `{username}` ({role_label})")

    mode = llm_mode()
    if mode == "live":
        st.sidebar.success("🟢 **LLM**: live (Google Gemini)")
    else:
        st.sidebar.info("⚪ **LLM**: fallback (rule-based)")

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Log Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["role"] = None
        st.session_state["username"] = None
        st.rerun()

    # Route by logged-in role
    if role == "asha":
        render_asha_portal()
    elif role == "doctor":
        render_doctor_portal()

    # Global Mandatory Disclaimer Footer
    render_footer()


if __name__ == "__main__":
    main()
