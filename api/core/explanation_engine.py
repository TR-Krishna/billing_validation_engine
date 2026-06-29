"""
Explanation Engine
==================
Generates plain English explanations for every validation result.
No external API — pure Python string generation.

Each explanation has three parts:
  1. Observation  — what the data shows (specific numbers)
  2. Cause        — what likely caused it
  3. Action       — what should be done

Recommendation is one of:
  FIELD_INSPECTION  → HIGH severity, send team immediately
  SCHEDULED_REVIEW  → MEDIUM severity, review within 7 days
  MONITOR           → LOW severity, watch next billing cycle
  NO_ACTION         → completely normal
"""

from typing import Dict, Any


def generate_explanation(
    meter_id:        str,
    billing_period:  str,
    meter_type:      str,
    anomaly_type:    str,
    severity:        str,
    actual_wh:       float,
    predicted_wh:    float,
    hist_avg_wh:     float,
    deviation_pct:   float,
    power_factor:    float,
    hist_avg_pf:     float,
    risk_score:      float,
    revenue_at_risk: float,
    rule_flags:      Dict[str, bool],
    history_months:  int,
    model_confidence:str,
) -> tuple:
    """
    Generate plain English explanation and recommendation.

    Returns
    -------
    (explanation: str, recommendation: str)
    """

    abs_dev    = abs(deviation_pct)
    pf_drop    = round(hist_avg_pf - power_factor, 2)
    period_str = billing_period

    # ── Observation ───────────────────────────────────────────────────────────
    if anomaly_type in ("SuddenDrop", "NTL_Suspected", "BypassSuspected"):
        observation = (
            f"Meter {meter_id} recorded {actual_wh:,.0f} Wh in "
            f"{period_str}, which is {abs_dev:.1f}% below the model "
            f"forecast of {predicted_wh:,.0f} Wh and {abs_dev:.1f}% "
            f"below its historical average of {hist_avg_wh:,.0f} Wh."
        )

    elif anomaly_type == "SuddenSpike":
        observation = (
            f"Meter {meter_id} recorded {actual_wh:,.0f} Wh in "
            f"{period_str}, which is {abs_dev:.1f}% above the model "
            f"forecast of {predicted_wh:,.0f} Wh and {abs_dev:.1f}% "
            f"above its historical average of {hist_avg_wh:,.0f} Wh."
        )

    elif anomaly_type == "ZeroConsumption":
        observation = (
            f"Meter {meter_id} recorded zero consumption in "
            f"{period_str}, despite a historical average of "
            f"{hist_avg_wh:,.0f} Wh per month."
        )

    elif anomaly_type == "BillingMismatch":
        diff = abs(actual_wh - predicted_wh)
        observation = (
            f"Meter {meter_id} shows a billing discrepancy in "
            f"{period_str}. The meter recorded {actual_wh:,.0f} Wh "
            f"but the billing system applied a different amount. "
            f"Difference: {diff:,.0f} Wh."
        )

    elif anomaly_type == "RateSumError":
        observation = (
            f"Meter {meter_id} has a rate allocation error in "
            f"{period_str}. The sum of Rate 1 to Rate 4 does not "
            f"equal the total active energy of {actual_wh:,.0f} Wh. "
            f"This indicates a billing calculation error."
        )

    elif anomaly_type == "PowerFactorAnomaly":
        observation = (
            f"Meter {meter_id} recorded a power factor of "
            f"{power_factor:.2f} in {period_str}, which is below "
            f"the acceptable minimum of 0.85. Historical average "
            f"was {hist_avg_pf:.2f}."
        )

    elif anomaly_type == "PowerFactorDeviation":
        observation = (
            f"Meter {meter_id} recorded a power factor of "
            f"{power_factor:.2f} in {period_str}, a drop of "
            f"{pf_drop:.2f} from its historical average of "
            f"{hist_avg_pf:.2f}."
        )

    elif anomaly_type == "FlatLine":
        observation = (
            f"Meter {meter_id} has recorded the same consumption "
            f"value of approximately {actual_wh:,.0f} Wh for three "
            f"or more consecutive months. The meter may have stopped "
            f"updating."
        )

    elif anomaly_type == "TariffBoundaryGaming":
        observation = (
            f"Meter {meter_id} has recorded consumption just below "
            f"a tariff slab boundary for multiple consecutive months "
            f"in {period_str}. Current reading: {actual_wh:,.0f} Wh."
        )

    elif anomaly_type == "StatisticalAnomaly":
        observation = (
            f"Meter {meter_id} shows an unusual statistical pattern "
            f"in {period_str}. The reading of {actual_wh:,.0f} Wh "
            f"is flagged by the anomaly detection model as outside "
            f"the normal range for this meter."
        )

    else:
        # normal meter
        observation = (
            f"Meter {meter_id} recorded {actual_wh:,.0f} Wh in "
            f"{period_str}, within the expected range."
        )

    # ── Cause ─────────────────────────────────────────────────────────────────
    if anomaly_type == "NTL_Suspected":
        if pf_drop > 0.05:
            cause = (
                f"The simultaneous power factor degradation from "
                f"{hist_avg_pf:.2f} to {power_factor:.2f} alongside "
                f"the consumption drop is consistent with meter "
                f"tampering or partial bypass. Physical consumption "
                f"appears to continue while metered consumption has "
                f"been artificially reduced."
            )
        else:
            cause = (
                f"This pattern is consistent with non-technical loss. "
                f"Energy is being consumed but not fully recorded by "
                f"the meter."
            )

    elif anomaly_type == "BypassSuspected":
        cause = (
            f"Peak demand remains near normal levels while total "
            f"energy consumption has dropped significantly. This is "
            f"a classic bypass signature — electrical loads are "
            f"still operating but some or all energy is flowing "
            f"around the meter."
        )

    elif anomaly_type == "SuddenDrop":
        cause = (
            f"Possible causes include partial meter bypass, meter "
            f"communication fault, or an unrecorded billing period. "
            f"No seasonal or operational pattern explains a drop "
            f"of this magnitude."
        )

    elif anomaly_type == "SuddenSpike":
        cause = (
            f"Possible causes include meter malfunction recording "
            f"accumulated backlog, a billing system error, or "
            f"unauthorised high-power load addition. The reading "
            f"exceeds the model's upper confidence bound."
        )

    elif anomaly_type == "ZeroConsumption":
        cause = (
            f"The meter may have lost communication with HES, the "
            f"physical meter connection may be broken, or the billing "
            f"read was not captured for this period."
        )

    elif anomaly_type == "BillingMismatch":
        cause = (
            f"The billing system applied a different value than what "
            f"the meter physically recorded. This may indicate a "
            f"billing system error, a data pipeline issue, or "
            f"deliberate manipulation of billing records."
        )

    elif anomaly_type == "RateSumError":
        cause = (
            f"The time-of-use rate allocation is internally "
            f"inconsistent. This is typically a billing system "
            f"calculation error or a data corruption issue."
        )

    elif anomaly_type in ("PowerFactorAnomaly", "PowerFactorDeviation"):
        cause = (
            f"Poor power factor may indicate uncorrected inductive "
            f"loads, capacitor bank failure, or possible meter "
            f"tampering affecting current measurement. The utility "
            f"power factor penalty clause may apply."
        )

    elif anomaly_type == "FlatLine":
        cause = (
            f"A meter recording identical values over multiple months "
            f"is likely stuck, not communicating, or being read from "
            f"a cached value. Estimated billing may be in use."
        )

    elif anomaly_type == "TariffBoundaryGaming":
        cause = (
            f"Repeated readings just below a tariff boundary across "
            f"multiple months may indicate deliberate load management "
            f"to avoid higher tariff rates, or coordinated "
            f"consumption manipulation."
        )

    elif anomaly_type == "StatisticalAnomaly":
        cause = (
            f"The combination of features for this meter in this "
            f"period is statistically unusual compared to the "
            f"normal population of meters. Manual review is "
            f"recommended to determine the cause."
        )

    else:
        cause = ""

    # ── Model confidence note ────────────────────────────────────────────────
    confidence_note = ""
    if model_confidence == "INSUFFICIENT":
        confidence_note = (
            f" Note: Only {history_months} months of history available. "
            f"ML models inactive — rules engine only."
        )
    elif model_confidence == "LOW":
        confidence_note = (
            f" Note: Limited history ({history_months} months). "
            f"ML model confidence is low."
        )

    # ── Action ────────────────────────────────────────────────────────────────
    if severity == "HIGH":
        action = (
            f"Immediate field inspection is recommended. Billing "
            f"should be held pending physical verification of the "
            f"meter. Estimated revenue at risk: "
            f"Rs {revenue_at_risk:,.2f}."
        )
        recommendation = "FIELD_INSPECTION"

    elif severity == "MEDIUM":
        action = (
            f"Schedule a field inspection within 7 days. Monitor "
            f"the next billing cycle before escalating. Estimated "
            f"revenue at risk: Rs {revenue_at_risk:,.2f}."
        )
        recommendation = "SCHEDULED_REVIEW"

    elif anomaly_type != "None" and score_qualifies_for_monitor(risk_score):
        action = (
            f"Flag for review in the next billing cycle. No "
            f"immediate action required. Continue monitoring."
        )
        recommendation = "MONITOR"

    else:
        action         = "No action required."
        recommendation = "NO_ACTION"

    # ── Combine ───────────────────────────────────────────────────────────────
    parts = [observation]
    if cause:
        parts.append(cause)
    parts.append(action)
    if confidence_note:
        parts.append(confidence_note)

    explanation = " ".join(parts)

    return explanation, recommendation


def score_qualifies_for_monitor(risk_score: float) -> bool:
    return risk_score > 0