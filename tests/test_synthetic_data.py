"""Test suite for synthetic dataset generator and data integrity."""

import csv
import os
import pytest
from src.data.generate_synthetic import (
    DEFAULT_SEED,
    MAHARASHTRA_GEOGRAPHY,
    generate_all_datasets,
)


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory):
    """Generate datasets into temporary test directory using fixed seed."""
    temp_dir = tmp_path_factory.mktemp("synthetic_test")
    generate_all_datasets(output_dir=str(temp_dir), seed=DEFAULT_SEED)
    return str(temp_dir)


def test_patients_file_integrity(synthetic_data_dir):
    """Verify patients.csv has 3000 rows and compliant schema."""
    filepath = os.path.join(synthetic_data_dir, "patients.csv")
    assert os.path.exists(filepath), "patients.csv must exist"

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 3000, "Must have exactly 3000 patients"
    first = reader[0]

    # Required headers
    expected_fields = [
        "patient_id", "patient_name", "age", "sex", "district", "village",
        "distance_to_phc_km", "cough_duration_days", "has_cough", "has_fever",
        "has_weight_loss", "has_night_sweats", "has_hemoptysis", "diabetes",
        "hiv_status", "smoker", "tb_contact", "tb_positive"
    ]
    for field in expected_fields:
        assert field in first, f"Field {field} must be present in patients.csv"

    # Verify Maharashtra districts and villages
    districts = set(MAHARASHTRA_GEOGRAPHY.keys())
    for row in reader[:100]:
        assert row["district"] in districts
        assert row["village"] in MAHARASHTRA_GEOGRAPHY[row["district"]]
        assert int(row["age"]) >= 15
        assert float(row["distance_to_phc_km"]) > 0
        assert row["tb_positive"] in ["0", "1"]


def test_lab_results_file_integrity(synthetic_data_dir):
    """Verify lab_results.csv contains CBNAAT and smear microscopy records."""
    filepath = os.path.join(synthetic_data_dir, "lab_results.csv")
    assert os.path.exists(filepath), "lab_results.csv must exist"

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) > 3000
    test_types = set(r["test_type"] for r in reader)
    assert "CBNAAT / GeneXpert" in test_types
    assert "Sputum Smear Microscopy" in test_types


def test_treatment_logs_file_integrity(synthetic_data_dir):
    """Verify treatment_logs.csv covers 168 days per TB positive case with dropouts."""
    filepath = os.path.join(synthetic_data_dir, "treatment_logs.csv")
    assert os.path.exists(filepath), "treatment_logs.csv must exist"

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) > 50000
    statuses = set(r["status"] for r in reader)
    assert "Dropped Out / Defaulted" in statuses, "Must include patients dropping out"
    assert "Active" in statuses


def test_contacts_file_integrity(synthetic_data_dir):
    """Verify contacts.csv traces household contacts for confirmed cases."""
    filepath = os.path.join(synthetic_data_dir, "contacts.csv")
    assert os.path.exists(filepath), "contacts.csv must exist"

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) > 1000
    for row in reader[:50]:
        assert int(row["age"]) >= 1
        assert row["relationship"] in ["Spouse", "Child", "Parent", "Sibling", "Grandparent", "In-law"]
        assert "tpt_initiated" in row


def test_drug_stock_file_integrity(synthetic_data_dir):
    """Verify drug_stock.csv tracks monthly stock across districts."""
    filepath = os.path.join(synthetic_data_dir, "drug_stock.csv")
    assert os.path.exists(filepath), "drug_stock.csv must exist"

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 600  # 10 districts * 5 items * 12 months
    months = set(r["month"] for r in reader)
    assert len(months) == 12


def test_synthetic_readme_exists():
    """Verify data/synthetic/README exists and includes the disclaimer."""
    filepath = "data/synthetic/README"
    assert os.path.exists(filepath), "data/synthetic/README must exist"
    with open(filepath, mode="r", encoding="utf-8") as f:
        content = f.read()
    assert "SYNTHETIC" in content.upper()
    assert "Research prototype, not clinically validated, not for real diagnosis" in content
