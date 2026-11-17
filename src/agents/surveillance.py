"""Surveillance Agent Module (STUB).

Aggregates screening and confirmed incidence data across geographic villages and PHCs.
Detects localized spatial clusters and monitors epidemiological incidence trends.
Future tasks will integrate statistical time-series forecasting and clustering models.
"""

from datetime import datetime, timezone
from typing import Literal
from src.schemas.models import SurveillanceResult


def run_surveillance(
    village: str,
    confirmed: bool,
    patient_id: str
) -> SurveillanceResult:
    """Record epidemiological observation and evaluate outbreak/clustering risk.

    Args:
        village: Location name.
        confirmed: True if case was confirmed by Medical Officer.
        patient_id: Screened patient identifier.

    Returns:
        SurveillanceResult: Updated surveillance telemetry.
    """
    # Stub cluster detection
    alert: Literal["Normal", "Watch", "Outbreak"] = "Watch" if confirmed else "Normal"

    notes = (
        f"Case recorded for {village}. "
        f"Active surveillance index updated for patient {patient_id}. "
        f"Alert level: {alert}."
    )

    return SurveillanceResult(
        village=village,
        cluster_detected=confirmed,
        alert_level=alert,
        cumulative_village_cases=1 if confirmed else 0,
        surveillance_timestamp=datetime.now(timezone.utc).isoformat(),
        notes=notes
    )
