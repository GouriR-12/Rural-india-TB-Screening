"""Follow-up Agent Module (STUB).

Manages patient follow-up scheduling, calculates treatment adherence and dropout risk,
and coordinates with assigned ASHA workers.
Future tasks will integrate the XGBoost tabular model for loss-to-follow-up prediction.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal
from src.schemas.models import DecisionResult, DoctorReviewResult, FollowupResult


def run_followup(
    patient_id: str,
    village: str,
    asha_id: str,
    decision: DecisionResult,
    doctor_review: DoctorReviewResult
) -> FollowupResult:
    """Generate adherence plan and predict dropout risk for approved TB case.

    Args:
        patient_id: Unique patient identifier.
        village: Patient residence village.
        asha_id: Frontline ASHA worker assigned.
        decision: Decision Agent synthesis.
        doctor_review: Approved doctor review record.

    Returns:
        FollowupResult: Scheduled visits, adherence plan, and risk tier.
    """
    if not doctor_review.approved:
        raise ValueError("Cannot initiate follow-up for a non-approved case. Doctor sign-off is mandatory.")

    # Stub calculation for dropout risk (heuristic placeholder for XGBoost)
    base_dropout = 0.25
    if decision.risk_category == "High":
        base_dropout += 0.20

    risk_level: Literal["High", "Medium", "Low"] = "Medium"
    if base_dropout >= 0.40:
        risk_level = "High"
    elif base_dropout <= 0.20:
        risk_level = "Low"

    # Schedule next ASHA home visit in 7 days
    next_visit = (datetime.now(timezone.utc) + timedelta(days=7)).strftime("%Y-%m-%d")

    adherence_plan = (
        f"DOTS Initiation under Nikshay Poshan Yojana. Weekly ASHA home visit by {asha_id}. "
        f"Prescribed: {doctor_review.prescription or 'Standard 4FDC regimen'}. "
        f"Bi-weekly pill count and adherence SMS reminders configured."
    )

    return FollowupResult(
        dropout_risk_score=round(base_dropout, 2),
        dropout_risk_level=risk_level,
        adherence_plan=adherence_plan,
        next_visit_date=next_visit,
        asha_assigned=asha_id,
        dots_center=f"{village} Health Sub-Center"
    )
