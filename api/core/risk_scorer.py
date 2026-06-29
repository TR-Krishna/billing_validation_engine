"""
Risk Scoring Engine
===================
Combines all signals from Isolation Forest, TFT, and rules engine
into a single risk score from 0 to 100.

Score > 80  → HIGH   → HES alert sent immediately
Score 50-80 → MEDIUM → flagged for scheduled review
Score < 50  → LOW    → logged and monitored

Points are additive — each signal adds independently.
Score is capped at 100.

Scoring breakdown:
  Isolation Forest flagged      → up to 40 points
  TFT deviation > 60%          → 30 points
  TFT deviation 30-60%         → 15 points
  TFT deviation 15-30%         → 5 points
  NTL rule fired                → 25 points
  Bypass rule fired             → 25 points
  Zero consumption rule         → 20 points
  Billing mismatch rule         → 20 points
  Rate sum error rule           → 10 points
  Tariff boundary gaming        → 15 points
  Power factor low rule         → 15 points
  Power factor deviation rule   → 10 points
  Flat line rule                → 10 points
"""

from typing import Dict, Any

INR_PER_WH = 0.008   # Rs 8 per kWh


def compute_risk_score(
    if_is_anomaly:   bool,
    if_score:        float,
    tft_deviation:   float,
    tft_sudden_drop: bool,
    tft_sudden_spike:bool,
    rule_flags:      Dict[str, bool],
    active_wh:       float,
    predicted_wh:    float,
    hist_avg_wh:     float,
    thresholds:      Dict[str, float],
) -> Dict[str, Any]:
    """
    Compute risk score from all signals.

    Parameters
    ----------
    if_is_anomaly   : Isolation Forest flagged this meter
    if_score        : IF anomaly score (lower = more anomalous)
    tft_deviation   : TFT % deviation (actual - predicted) / predicted
    tft_sudden_drop : actual fell below TFT p10
    tft_sudden_spike: actual exceeded TFT p90
    rule_flags      : dict of rule_name → bool from validation_rules
    active_wh       : current month actual energy
    predicted_wh    : TFT p50 prediction
    hist_avg_wh     : this meter's historical average
    thresholds      : from utility config

    Returns
    -------
    Dict with:
      riskScore        : 0 to 100
      severity         : HIGH / MEDIUM / LOW
      anomalyType      : primary anomaly type string
      ntlSuspected     : bool
      preInvoiceFlag   : bool
      revenueAtRiskINR : estimated revenue at risk in rupees
      scoreComponents  : breakdown of what contributed
    """
    score      = 0.0
    components = {}

    tft_high = float(thresholds.get("tftDeviationHighPct",   60.0))
    tft_med  = float(thresholds.get("tftDeviationMediumPct", 30.0))

    # ── Isolation Forest contribution ────────────────────────────────────────
    # More negative score = more anomalous
    # Scale contribution based on how anomalous the score is
    # Typical range: -0.50 (borderline) to -0.75 (very anomalous)
    if if_is_anomaly:
        # normalise score: -0.50 → 0 points, -0.75 → 40 points
        raw          = abs(if_score)
        if_contrib   = min(40.0, max(0.0, (raw - 0.50) / 0.25 * 40.0))
        score       += if_contrib
        components["isolation_forest"] = round(if_contrib, 1)
    else:
        components["isolation_forest"] = 0

    # ── TFT deviation contribution ───────────────────────────────────────────
    abs_dev = abs(tft_deviation)
    if abs_dev >= tft_high:
        tft_contrib = 30
    elif abs_dev >= tft_med:
        tft_contrib = 15
    elif abs_dev >= 15:
        tft_contrib = 5
    else:
        tft_contrib = 0
    score += tft_contrib
    components["tft_deviation"] = tft_contrib

    # ── Rule contributions ───────────────────────────────────────────────────
    rule_points = {
        "ntlSuspected":        25,
        "bypassSuspected":     25,
        "zeroConsumption":     20,
        "billingMismatch":     20,
        "tariffBoundaryGaming":15,
        "powerFactorLow":      15,
        "rateSumError":        10,
        "powerFactorDeviation":10,
        "flatLine":            10,
    }
    rule_total = 0
    for flag, points in rule_points.items():
        if rule_flags.get(flag, False):
            score      += points
            rule_total += points
    components["rules"] = rule_total

    # cap at 100
    score = min(round(score, 2), 100.0)

    # ── Severity ──────────────────────────────────────────────────────────────
    if score >= 80:
        severity = "HIGH"
    elif score >= 50:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # ── Primary anomaly type ─────────────────────────────────────────────────
    # Priority: most severe and actionable first
    if rule_flags.get("ntlSuspected") or rule_flags.get("bypassSuspected"):
        anomaly_type = "NTL_Suspected"
    elif tft_sudden_drop or tft_deviation < -tft_high:
        anomaly_type = "SuddenDrop"
    elif tft_sudden_spike or tft_deviation > tft_high:
        anomaly_type = "SuddenSpike"
    elif rule_flags.get("zeroConsumption"):
        anomaly_type = "ZeroConsumption"
    elif rule_flags.get("billingMismatch"):
        anomaly_type = "BillingMismatch"
    elif rule_flags.get("rateSumError"):
        anomaly_type = "RateSumError"
    elif rule_flags.get("tariffBoundaryGaming"):
        anomaly_type = "TariffBoundaryGaming"
    elif rule_flags.get("powerFactorLow"):
        anomaly_type = "PowerFactorAnomaly"
    elif rule_flags.get("powerFactorDeviation"):
        anomaly_type = "PowerFactorDeviation"
    elif rule_flags.get("flatLine"):
        anomaly_type = "FlatLine"
    elif if_is_anomaly:
        anomaly_type = "StatisticalAnomaly"
    else:
        anomaly_type = "None"

    # ── NTL suspected ─────────────────────────────────────────────────────────
    ntl_suspected = (
        rule_flags.get("ntlSuspected",   False) or
        rule_flags.get("bypassSuspected",False) or
        anomaly_type == "NTL_Suspected"
    )

    # ── Pre-invoice flag ──────────────────────────────────────────────────────
    # Flag billing as pending if deviation is significant
    # or if any HIGH severity rule fired
    pre_invoice_flag = (
        abs(tft_deviation) > tft_med or
        score >= 50 or
        rule_flags.get("billingMismatch", False) or
        rule_flags.get("ntlSuspected",    False)
    )

    # ── Revenue at risk ───────────────────────────────────────────────────────
    # How much revenue might be lost due to this anomaly
    # Based on gap between expected and actual consumption
    if predicted_wh > 0 and active_wh < predicted_wh:
        # actual below prediction — revenue being lost
        lost_wh          = predicted_wh - active_wh
        revenue_at_risk  = round(lost_wh * INR_PER_WH, 2)
    elif hist_avg_wh > 0 and active_wh < hist_avg_wh:
        # no TFT prediction available — use historical average
        lost_wh          = hist_avg_wh - active_wh
        revenue_at_risk  = round(lost_wh * INR_PER_WH, 2)
    else:
        revenue_at_risk  = 0.0

    return {
        "riskScore":         score,
        "severity":          severity,
        "isAnomaly":         score > 0,
        "anomalyType":       anomaly_type,
        "ntlSuspected":      ntl_suspected,
        "preInvoiceFlag":    pre_invoice_flag,
        "revenueAtRiskINR":  revenue_at_risk,
        "scoreComponents":   components,
    }