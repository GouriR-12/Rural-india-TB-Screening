"""Unit tests for Imaging Agent, Grad-CAM generation, and training provenance gates."""

import csv
import glob
import os
from PIL import Image
import pytest

from src.agents.imaging import analyze_image, run_imaging
from src.agents.train_imaging import (
    RealDataBlockedError,
    patient_stratified_split,
    save_imaging_report,
    train_imaging_model,
)
from src.data.download_cxr import METADATA_FILE, parse_filename_label
from src.schemas.models import ImagingPayload, ImagingResult


def test_imaging_positive_cxr():
    """Verify DenseNet121 CNN inference on confirmed TB chest radiograph."""
    # Find sample TB image from demo/benchmark directory
    tb_candidates = glob.glob("data/cxr/synthetic_demo_cxr/*TB*.png") or glob.glob("data/cxr/benchmark/*TB*.png")
    assert len(tb_candidates) > 0, "TB demo images must exist"
    test_img = tb_candidates[0]

    payload = ImagingPayload(
        case_id="CASE-TEST-TB-001",
        patient_id="P-001",
        image_path=test_img
    )
    result = run_imaging(payload)

    assert isinstance(result, ImagingResult)
    assert result.abnormality_detected is True
    assert result.tb_probability >= 0.50
    assert result.heatmap_path is not None
    assert os.path.exists(result.heatmap_path), "Grad-CAM heatmap file must be generated"
    assert "positive" in result.confidence_note.lower()
    assert result.model_version == "densenet121-tb-v1"
    assert result.model_provenance in ["synthetic_demo", "real_public_data"]


def test_imaging_normal_cxr():
    """Verify DenseNet121 CNN inference on normal chest radiograph."""
    normal_candidates = glob.glob("data/cxr/synthetic_demo_cxr/*NORMAL*.png") or glob.glob("data/cxr/benchmark/*NORMAL*.png")
    assert len(normal_candidates) > 0, "Normal demo images must exist"
    test_img = normal_candidates[0]

    payload = ImagingPayload(
        case_id="CASE-TEST-NORM-001",
        patient_id="P-002",
        image_path=test_img
    )
    result = run_imaging(payload)

    assert isinstance(result, ImagingResult)
    assert result.abnormality_detected is False
    assert result.tb_probability < 0.50
    assert result.heatmap_path is not None
    assert os.path.exists(result.heatmap_path)
    assert "negative" in result.confidence_note.lower() or "clear" in result.confidence_note.lower()
    assert result.model_provenance in ["synthetic_demo", "real_public_data"]


def test_imaging_missing_path():
    """Verify graceful handling when CXR is pending or not uploaded."""
    payload = ImagingPayload(case_id="CASE-NO-IMG", patient_id="P-003", image_path=None)
    result = run_imaging(payload)

    assert result.abnormality_detected is False
    assert result.heatmap_path is None
    assert "No chest radiograph" in result.findings
    assert result.model_provenance in ["synthetic_demo", "real_public_data"]


def test_imaging_gradcam_overlay():
    """Verify that the generated Grad-CAM heatmap matches dimensions and has valid RGB colors."""
    tb_candidates = glob.glob("data/cxr/synthetic_demo_cxr/*TB*.png") or glob.glob("data/cxr/benchmark/*TB*.png")
    test_img = tb_candidates[0]
    result = analyze_image(test_img, case_id="CASE-GRADCAM-DIM")

    assert result.heatmap_path is not None
    with Image.open(result.heatmap_path) as hmap:
        assert hmap.mode == "RGB"
        with Image.open(test_img) as orig:
            assert hmap.size == orig.size, "Heatmap dimensions must match original image"


def test_imaging_report_file_exists():
    """Verify models/imaging_report.md exists with required DEMO MODEL header and metrics."""
    report_file = os.path.join("models", "imaging_report.md")
    assert os.path.exists(report_file), "models/imaging_report.md must exist"

    with open(report_file, mode="r", encoding="utf-8") as f:
        content = f.read()

    assert "DEMO MODEL: trained on synthetic cartoon images. Metrics are not evidence of performance." in content
    assert "AUC-ROC" in content
    assert "Sensitivity" in content
    assert "Specificity" in content
    assert "Confusion Matrix" in content
    assert "Research prototype, not clinically validated, not for real diagnosis" in content


def test_training_refuses_validated_report_on_synthetic_data(tmp_path):
    """Verify training on synthetic_demo_cxr data refuses to emit a validated clinical report."""
    test_report_path = str(tmp_path / "test_imaging_report.md")

    dummy_metrics = {
        "auc": 0.99,
        "auc_ci_lower": 0.95,
        "auc_ci_upper": 1.00,
        "sensitivity": 0.95,
        "specificity": 0.95,
        "precision": 0.95,
        "f1": 0.95,
        "accuracy": 0.95,
        "tp": 18,
        "tn": 18,
        "fp": 0,
        "fn": 0,
        "threshold": 0.50
    }

    # Emitting report with provenance == "synthetic_demo"
    save_imaging_report(
        metrics=dummy_metrics,
        train_count=168,
        val_count=36,
        test_count=36,
        epochs=3,
        device="cpu",
        img_size=128,
        provenance="synthetic_demo",
        report_path=test_report_path
    )

    with open(test_report_path, mode="r", encoding="utf-8") as f:
        content = f.read()

    # Must be explicitly headed as DEMO MODEL
    assert content.startswith("# DEMO MODEL: trained on synthetic cartoon images. Metrics are not evidence of performance.")
    assert "PIPELINE SMOKE-TEST FIXTURE ONLY" in content
    assert "Real-data training is currently **BLOCKED" in content


def test_real_data_training_blocked_when_absent():
    """Verify train_imaging_model raises RealDataBlockedError when real data is requested but absent."""
    with pytest.raises(RealDataBlockedError, match="BLOCKED"):
        train_imaging_model(
            epochs=1,
            require_real=True
        )


def test_filename_label_convention():
    """Verify _0 (Normal) and _1 (TB) filename convention parsing for Shenzhen and Montgomery."""
    assert parse_filename_label("CHNCXR_0001_0.png") == (0, "CHNCXR_0001")
    assert parse_filename_label("CHNCXR_0350_1.png") == (1, "CHNCXR_0350")
    assert parse_filename_label("MCUCXR_0001_0.png") == (0, "MCUCXR_0001")
    assert parse_filename_label("MCUCXR_0050_1.png") == (1, "MCUCXR_0050")
    assert parse_filename_label("invalid_filename.png") is None
    assert parse_filename_label("CHNCXR_0001_2.png") is None


def test_metadata_dataset_source_labeled_synthetic_demo():
    """Verify that metadata.csv labels the smoke-test set as synthetic_demo_cxr."""
    assert os.path.exists(METADATA_FILE), "metadata.csv must exist"
    with open(METADATA_FILE, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) > 0
    sources = set(r["dataset_source"] for r in reader)
    assert "synthetic_demo_cxr" in sources


def test_patient_stratified_split_zero_leakage():
    """Verify patient_stratified_split guarantees zero patient leakage between train, val, and test."""
    dummy_records = [
        {"image_id": "img1", "patient_id": "P001", "label": 1},
        {"image_id": "img2", "patient_id": "P001", "label": 1},  # Multiple images for P001
        {"image_id": "img3", "patient_id": "P002", "label": 1},
        {"image_id": "img4", "patient_id": "P003", "label": 0},
        {"image_id": "img5", "patient_id": "P004", "label": 0},
        {"image_id": "img6", "patient_id": "P005", "label": 0},
        {"image_id": "img7", "patient_id": "P006", "label": 1},
        {"image_id": "img8", "patient_id": "P007", "label": 0},
    ]

    train_set, val_set, test_set = patient_stratified_split(dummy_records, seed=42)
    train_pids = set(r["patient_id"] for r in train_set)
    val_pids = set(r["patient_id"] for r in val_set)
    test_pids = set(r["patient_id"] for r in test_set)

    # Assert no overlap in patient IDs
    assert train_pids.isdisjoint(val_pids)
    assert train_pids.isdisjoint(test_pids)
    assert val_pids.isdisjoint(test_pids)


def test_doctor_ui_warning_badge_on_synthetic_demo():
    """Verify Doctor UI displays prominent warning badge when provenance is synthetic_demo."""
    from src.ui.doctor_dashboard import get_provenance_badge, format_doctor_review_summary

    badge = get_provenance_badge("synthetic_demo")
    assert badge["is_synthetic"] is True
    assert "DEMO MODEL" in badge["badge_text"]
    assert badge["badge_color"] == "#ff4b4b"

    summary = format_doctor_review_summary({
        "case_id": "CASE-DEMO-001",
        "patient_name": "Ramesh Patil",
        "model_provenance": "synthetic_demo",
        "imaging_result": {
            "tb_probability": 0.88,
            "findings": "Apical lesions",
            "model_provenance": "synthetic_demo"
        }
    })
    assert "SYNTHETIC DEMO MODEL - NOT CLINICALLY VALIDATED" in summary
    assert "ATTENTION DOCTOR" in summary

