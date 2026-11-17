# Multi-Agent TB Screening System for Rural India

> [!WARNING]
> **Research prototype, not clinically validated, not for real diagnosis.**

A decision-support and workflow orchestration system designed to support frontline healthcare workers (ASHA workers) and primary healthcare medical officers in screening, triaging, tracking, and monitoring tuberculosis (TB) cases across rural India.

---

## Architecture & Workflow

The screening pipeline is orchestrated via **LangGraph** with state persistence backed by **SQLite checkpointing**.

```
[ ASHA Worker Intake ]
          │
          ▼
    [ intake_node ]
          │
    ┌─────┴────────────────┐
    ▼ (Parallel Branch)    ▼ (Parallel Branch)
[ imaging_node ]      [ lab_order_node ]
    │                      │
    └─────┬────────────────┘
          ▼
   [ decision_node ]
          │
╔═════════╧═══════════════════════════════════════════╗
║         MANDATORY DOCTOR REVIEW GATE                ║
║  (LangGraph interrupt pausing for human sign-off)   ║
╚═════════╤═══════════════════════════════════════════╝
          │
          ├── [Approved] ──► [ followup_node ] ──► [ surveillance_node ] ──► END
          │
          └── [Rejected] ──► [ rejected_end ] (Status: "Not TB - refer or re-screen") ──► END
```

### The 5 Core Agents
1. **Intake Agent** (`src/agents/intake.py`): Parses multilingual clinical notes (Hindi/Marathi/English) and checklist symptoms into structured JSON.
2. **Imaging Agent** (`src/agents/imaging.py`): Fine-tuned PyTorch DenseNet-121 deep convolutional network evaluating chest radiographs (CXR) for pulmonary TB opacities and cavitation, paired with Grad-CAM visual heatmaps.
3. **Decision Agent** (`src/agents/decision.py`): Synthesizes symptoms, radiology findings, and parallel lab orders into composite risk stratification and a briefing summary.
4. **Follow-up Agent** (`src/agents/followup.py`): Formulates adherence plans under Nikshay Poshan Yojana and estimates patient loss-to-follow-up / dropout risk. **Enforced Rule**: Can never be reached without doctor approval.
5. **Surveillance Agent** (`src/agents/surveillance.py`): Tracks village-level incidence, flags active clusters, and monitors epidemiological trends.

---

## Project Structure

```
.
├── .env.example              # Environment variables template
├── PROJECT_RULES.md          # Architectural rules & constraints
├── README.md                 # Project documentation & run guide
├── requirements.txt          # Python dependencies
├── data/                     # Local data storage
│   ├── synthetic/            # Synthetic research datasets
│   │   ├── README            # Synthetic notice & disclaimer
│   │   ├── patients.csv      # 3,000 synthetic patient records
│   │   ├── lab_results.csv   # CBNAAT molecular & smear microscopy
│   │   ├── treatment_logs.csv# 6-month daily adherence logs with dropouts
│   │   ├── contacts.csv      # Household contact tracing records
│   │   └── drug_stock.csv    # Monthly district stock & consumption logs
│   └── tb_screening.db       # SQLite workflow checkpoint store
├── models/                   # Model weight checkpoints
├── src/
│   ├── agents/               # 5 Isolated agent modules with Pydantic contracts
│   │   ├── __init__.py
│   │   ├── intake.py
│   │   ├── imaging.py
│   │   ├── decision.py
│   │   ├── followup.py
│   │   └── surveillance.py
│   ├── data/                 # Data generators & storage
│   │   ├── __init__.py
│   │   └── generate_synthetic.py # Probabilistic synthetic dataset generator
│   ├── graph/                # LangGraph state machine & checkpointing
│   │   ├── __init__.py
│   │   └── workflow.py
│   ├── schemas/              # Pydantic data schemas & ScreeningState
│   │   ├── __init__.py
│   │   └── models.py
│   ├── ui/                   # Streamlit UI (subsequent tasks)
│   └── demo_case.py          # End-to-end runnable demonstration
└── tests/                    # Automated pytest test suites
    ├── __init__.py
    ├── test_decision.py
    ├── test_followup.py
    ├── test_imaging.py
    ├── test_intake.py
    ├── test_surveillance.py
    ├── test_synthetic_data.py
    └── test_workflow.py
```

---

## Synthetic Data Generation

The project includes a realistic synthetic dataset generator (`src/data/generate_synthetic.py`) configured with a fixed random seed (`42`) to produce calibrated rural public health datasets under `data/synthetic/`:

| Dataset | Records | Description |
| :--- | :--- | :--- |
| `patients.csv` | 3,000 | Demographics, Maharashtra districts/villages, symptoms, risk factors, and probabilistic TB label. |
| `lab_results.csv` | 5,063 | CBNAAT (GeneXpert) molecular tests and sputum smear microscopy with calibrated sensitivity & specificity noise. |
| `treatment_logs.csv` | 118,272 | 168-day daily dose logs for 704 TB-positive cases. Dropout hazard depends on distance, age, adverse side effects, and missed doses (18.2% dropout rate). |
| `contacts.csv` | 2,441 | Household contacts traced per confirmed case (mean 3.47 contacts), TPT preventive therapy evaluation, and secondary active TB detection. |
| `drug_stock.csv` | 600 | Monthly opening stock, receipts, consumption, and buffer stock warnings across 10 Maharashtra districts for 12 months. |

To regenerate datasets and view summary statistics:
```bash
python -m src.data.generate_synthetic
```

---

## Chest X-Ray Imaging Pipeline & Model Training

The Imaging Agent (`src/agents/imaging.py`) employs a deep convolutional neural network (PyTorch `DenseNet121`) for pulmonary tuberculosis screening support:

### 1. Data Provenance & Real Data Availability
- **Pipeline Smoke-Test Fixture (`synthetic_demo_cxr`)**:
  - The repository includes 240 synthetic geometric demo images (`data/cxr/synthetic_demo_cxr/`) purely as an end-to-end integration smoke-test fixture.
  - **No clinical performance claims**: Metrics generated on synthetic cartoon images are NOT evidence of diagnostic performance.
  - `ImagingResult` outputs an explicit `model_provenance` field (`"synthetic_demo"` or `"real_public_data"`).
  - The Doctor Review UI displays a prominent warning badge whenever `model_provenance == "synthetic_demo"`.

- **Real Public Datasets (Shenzhen & Montgomery)**:
  - Real-data training is currently **BLOCKED (needs Kaggle download by user)** until real public datasets are downloaded.
  - Two download paths are cleanly supported:
    1. **Automated Kaggle API Download**: If credentials exist (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY`), `python src/data/download_cxr.py` fetches the real data automatically.
    2. **Manual Download**:
       - **Shenzhen Dataset**: Download from [Kaggle Shenzhen CXR Set](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen) and extract images into `data/cxr/raw/shenzhen/`.
       - **Montgomery Dataset**: Download from [Kaggle Montgomery CXR Set](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery) and extract images into `data/cxr/raw/montgomery/`.
       - Both datasets follow the verified naming convention: `_0.png` = Normal, `_1.png` = Tuberculosis.
    3. **Process & Index**:
       ```bash
       python src/data/download_cxr.py --process
       ```
       Builds `metadata.csv` with `dataset_source` in `{"shenzhen", "montgomery"}` and extracted patient IDs.

### 2. Training Script (`src/agents/train_imaging.py`)
- **Refuses Validated Report on Synthetic Data**:
  - Training on `synthetic_demo_cxr` refuses to emit a "validated" clinical report and instead outputs `# DEMO MODEL: trained on synthetic cartoon images. Metrics are not evidence of performance.`
- **Real-Data Training**:
  - Requires real data; if real data is absent, real-data training is marked **BLOCKED** and refuses to substitute synthetic data.
  - Resolution: Trains at **224x224** on real data (128x128 on demo fixture).
  - **Patient-Level Stratified Partitioning**: Grouped strictly by `patient_id` (70% Train, 15% Val, 15% Test) with **zero patient data leakage**.
  - **Bootstrap 95% CI**: Reports test set AUC-ROC with a 1000-iteration bootstrap 95% confidence interval, sensitivity, specificity, and confusion matrix.
  - CPU-friendly defaults with `--epochs` flag.

To train:
```bash
# Demo smoke-test training (creates DEMO MODEL report)
python src/agents/train_imaging.py --epochs 3

# Real-data training (requires downloaded datasets; marks BLOCKED if absent)
python src/agents/train_imaging.py --real-data --epochs 5 --img-size 224
```

### 3. Model Diagnostic Report Status
Full evaluation status is documented in [`models/imaging_report.md`](file:///models/imaging_report.md):
- **Current Status**: `synthetic_demo` smoke-test fixture.
- **Real Data Benchmark**: Blocked pending user download of Shenzhen and Montgomery datasets.
- **Prior 1.0000 Claims Removed**: Prior perfect metrics were artifacts of geometric cartoon demo images and have been removed.

### 4. Explainability with Grad-CAM
Every radiograph evaluation produces a Grad-CAM (Gradient-weighted Class Activation Mapping) saliency map hooking activations and gradients from `model.features.denseblock4.denselayer16.conv2`. The colormapped heatmap is saved under `data/cxr/heatmaps/` and surfaced to the Doctor Review Gate with its provenance badge to explain which lung regions influenced the model's prediction.

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Virtual environment (`venv`)

### 2. Environment Setup
```bash
# Clone and enter directory
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### 3. Running Automated Tests
Run the complete test suite including agent unit tests, synthetic data integrity tests, and LangGraph interrupt/resume workflow tests:
```bash
python -m pytest tests -v
```

### 4. Running the End-to-End Demonstration
Run a dummy screening case from ASHA intake through parallel evaluation, doctor review gate interrupt, and resumed follow-up/surveillance:
```bash
python src/demo_case.py
```

---

## Safety & Ethics Notice
- **No Real Patient Data**: Only synthetic clinical records and public anonymized imaging datasets are utilized.
- **Human-in-the-Loop Mandate**: AI agents provide preliminary decision support only. Medical Officers maintain authoritative oversight via mandatory review gates before treatment initiation or follow-up scheduling.
- **Disclaimer**: *Research prototype, not clinically validated, not for real diagnosis.*
