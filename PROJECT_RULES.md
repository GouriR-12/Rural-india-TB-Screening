# Project Rules & Architecture Specification

## Project Overview
**Multi-agent Tuberculosis (TB) Screening System for Rural India**
A decision-support and workflow orchestration system designed to support frontline healthcare workers (ASHA workers) and primary healthcare medical officers in screening, triaging, tracking, and monitoring tuberculosis cases across rural India.

Workflow hierarchy:
`ASHA Worker Screenings -> Multi-Agent Pipeline -> Doctor Review Gate -> Follow-up & Treatment Adherence / Public Health Surveillance`

---

## Technology Stack

- **Runtime & Language**: Python 3.11
- **Orchestration**: LangGraph (state machine & human-in-the-loop workflow)
- **Computer Vision / Imaging**: PyTorch (Chest X-ray CNN classification)
- **Predictive Analytics**: scikit-learn & XGBoost (Patient dropout / loss-to-follow-up risk)
- **User Interface**: Streamlit (ASHA mobile intake simulation, Doctor review dashboard, Surveillance map & metrics)
- **Database & Persistence**: SQLite (local transactional store for screenings, reviews, and logs)
- **Testing Framework**: pytest
- **LLM Integration**: Google GenAI SDK (`google-genai`)
  - Key loaded exclusively via `GEMINI_API_KEY` from `.env` file (e.g., using `python-dotenv`).
  - **Zero Key Hardcoding Policy**: Never commit or hardcode API keys or credentials.

---

## Core Operational Rules

### Rule 1: Fixed Five Agents with Mandatory Doctor Gate
The system consists strictly of **five** agents:
1. **Intake Agent**: Ingests screening observations, patient symptoms, and history.
2. **Imaging Agent**: Evaluates chest radiographs (CXR) for TB indications.
3. **Decision Agent**: Synthesizes clinical data, symptoms, and image findings into an evidence-backed case summary and preliminary recommendation.
4. **Follow-up Agent**: Manages patient follow-ups, calculates dropout/loss-to-follow-up risk, and drafts adherence intervention plans.
5. **Surveillance Agent**: Aggregates population screening data, performs spatial/demographic clustering, and tracks epidemiological time-series trends.

**Mandatory Doctor Review Gate**:
- A strict human-in-the-loop (HIL) doctor-review gate sits between the **Decision Agent** and the **Follow-up Agent**.
- **No case can reach the Follow-up Agent without explicit doctor review and approval.**

```
[ ASHA Worker ]
       │
       ▼
 [ Intake Agent ] ─────────► [ Imaging Agent ]
                                    │
                                    ▼
                          [ Decision Agent ]
                                    │
                        ╔═══════════╧═══════════╗
                        ║  DOCTOR REVIEW GATE   ║  (Mandatory Human Sign-off)
                        ╚═══════════╤═══════════╝
                                    │ Approved
                                    ▼
                          [ Follow-up Agent ]
                                    │
                                    ▼
                         [ Surveillance Agent ]
```

---

### Rule 2: Targeted, Appropriate Tool Use (No Universal LLM Wrapping)
Do **not** wrap every task in an LLM. Leverage specialized models and analytical algorithms suited for each task:
- **LLM Used Exclusively For**:
  - **Intake Agent**: Multilingual natural language processing (extracting structured clinical entities from Hindi, Marathi, and English notes/transcripts into structured JSON).
  - **Decision Agent**: Narrative case synthesis, clinical explanation generation, and doctor briefing summaries.
- **Dedicated Models (Non-LLM)**:
  - **Imaging Agent**: Convolutional Neural Network (PyTorch) trained on public chest X-ray datasets.
  - **Follow-up Agent (Dropout Risk)**: Tabular machine learning (XGBoost / scikit-learn) based on socioeconomic, adherence, and geographic factors.
  - **Surveillance Agent**: Time-series analytics (e.g., statsmodels / rolling windows) and spatial clustering algorithms (e.g., DBSCAN / K-Means).

---

### Rule 3: Data Ethics, Synthetic Datasets & Mandatory Disclaimers
- **No Real Patient Data**: Never collect, use, or store real protected health information (PHI).
- **Imaging**: Use publicly available, anonymized open chest X-ray datasets (e.g., NIH, Shenzhen, Montgomery, or Kaggle TB datasets) or synthetic proxies.
- **Tabular/Clinical Data**: Generated synthetically to model rural Indian demographic patterns and epidemiological realities safely.
- **Mandatory Disclaimer**:
  The following disclaimer must appear prominently in the `README.md` and in the Streamlit UI footer:
  > **"Research prototype, not clinically validated, not for real diagnosis."**

---

### Rule 4: Modular Architecture & Pydantic Typings
- Every agent must reside in its own dedicated module under `src/agents/`:
  - `src/agents/intake.py`
  - `src/agents/imaging.py`
  - `src/agents/decision.py`
  - `src/agents/followup.py`
  - `src/agents/surveillance.py`
- All inter-agent data contracts, inputs, and outputs must be strictly typed using **Pydantic** models (located in `src/schemas/`).

Project Directory Layout:
```
├── .env.example
├── PROJECT_RULES.md
├── README.md
├── requirements.txt
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── intake.py
│   │   ├── imaging.py
│   │   ├── decision.py
│   │   ├── followup.py
│   │   └── surveillance.py
│   ├── models/           # CNN & XGBoost model definitions / artifacts
│   ├── pipeline/         # LangGraph workflow & doctor review state machine
│   ├── schemas/          # Pydantic data schemas & state models
│   ├── storage/          # SQLite models & repositories
│   └── ui/               # Streamlit application pages & components
└── tests/
    ├── test_intake.py
    ├── test_imaging.py
    ├── test_decision.py
    ├── test_followup.py
    ├── test_surveillance.py
    └── test_workflow.py
```

---

### Rule 5: Test-Driven Quality Assurance
- Every agent and pipeline component must have automated tests under `tests/`.
- All tests (`pytest`) must pass cleanly before any task or implementation milestone is marked as complete.

---

### Rule 6: Documentation Cadence
- After completing each task or milestone, immediately update `README.md` with:
  - Summary of what was built or updated.
  - Clear, reproducible instructions for running code, services, and test suites.
  - Verification of data privacy and safety disclaimers.
