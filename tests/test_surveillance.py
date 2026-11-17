"""Unit tests for Surveillance Agent."""

from src.agents.surveillance import run_surveillance
from src.schemas.models import SurveillanceResult


def test_surveillance_confirmed_case():
    result = run_surveillance(village="Bhimashankar", confirmed=True, patient_id="P-101")
    assert isinstance(result, SurveillanceResult)
    assert result.village == "Bhimashankar"
    assert result.cluster_detected is True
    assert result.alert_level == "Watch"


def test_surveillance_unconfirmed_case():
    result = run_surveillance(village="Bhimashankar", confirmed=False, patient_id="P-102")
    assert result.cluster_detected is False
    assert result.alert_level == "Normal"
