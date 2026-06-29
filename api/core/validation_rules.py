"""
Validation Rules Engine
=======================
Pure deterministic checks — no ML involved.
Each rule answers one specific billing question with arithmetic.

Handles what ML models cannot:
  BillingMismatch    → billed != metered (exact arithmetic)
  RateSumError       → Rate1+2+3+4 != Total (exact arithmetic)
  ZeroConsumption    → energy = 0 (obvious, rule is perfect)
  FlatLine           → same value 3+ months (sequence check)
  PowerFactorLow     → PF < 0.85 (threshold check)
  PowerFactorDev     → PF dropped from this meter's normal
  NTLSuspected       → drop + PF degradation together
  BypassSuspected    → drop but peak demand stays high
  TariffBoundaryGaming → reading near slab boundary repeatedly

All thresholds come from the utility config file.
No hardcoded values in this module (except slab boundaries — see Rule 9).
"""

from typing import Dict, Any, List


def run_all_rules(
    active_wh:    float,
    billed_wh:    float,
    r1: float, r2: float, r3: float, r4: float,
    power_factor: float,
    peak_demand:  float,
    features:     Dict[str, Any],
    thresholds:   Dict[str, float],
    e_history:    List[float],
) -> Dict[str, bool]:
    """
    Run all validation rule checks.

    Parameters
    ----------
    active_wh    : current month active energy (Wh)
    billed_wh    : current month billed amount (Wh)
    r1..r4       : rate-wise energy split (Wh)
    power_factor : current month average PF
    peak_demand  : current month peak demand (W)
    features     : computed features from feature_engine
    thresholds   : from utility config file
    e_history    : previous months energy, oldest first

    Returns
    -------
    Dict of rule_name → True (anomaly) / False (normal)
    """

    # pull thresholds from config — no hardcoded values
    pf_min        = thresholds.get("powerFactorMin",       0.85)
    rate_tol      = thresholds.get("rateSumTolerancePct",  2.0)
    bill_tol      = thresholds.get("billingMismatchPct",   2.0)
    ntl_drop      = thresholds.get("ntlEnergyDropPct",     85.0)
    ntl_pf_drop   = thresholds.get("ntlPFDropMin",         0.08)
    bypass_drop   = thresholds.get("bypassEnergyDropPct",  80.0)
    flat_months   = int(thresholds.get("flatLineMonths",   3))

    flags = {}

    # ── 1. Billing mismatch ──────────────────────────────────────────────────
    # Billed amount should equal metered amount within tolerance
    if active_wh > 0:
        mismatch_pct = abs(billed_wh - active_wh) / active_wh * 100
        flags["billingMismatch"] = mismatch_pct > bill_tol
    else:
        # if active = 0, any non-zero billed amount is a mismatch
        flags["billingMismatch"] = billed_wh > 0

    # ── 2. Rate sum error ────────────────────────────────────────────────────
    # Rate1 + Rate2 + Rate3 + Rate4 must equal Total within tolerance
    if active_wh > 0:
        rate_sum     = r1 + r2 + r3 + r4
        rate_err_pct = abs(rate_sum - active_wh) / active_wh * 100
        flags["rateSumError"] = rate_err_pct > rate_tol
    else:
        flags["rateSumError"] = False

    # ── 3. Zero consumption ──────────────────────────────────────────────────
    # Active energy = 0 on a meter that previously had consumption
    # Only flag if meter has history (not a brand new meter)
    has_history = len(e_history) >= 3
    flags["zeroConsumption"] = (active_wh == 0.0 and has_history)

    # ── 4. Flat line ─────────────────────────────────────────────────────────
    # Same value recorded for flat_months+ consecutive months
    # Suggests meter has stopped updating or is stuck
    if len(e_history) >= flat_months:
        last_n   = e_history[-flat_months:]
        all_same = len(set(round(v, 0) for v in last_n)) == 1
        non_zero = last_n[0] > 0
        flags["flatLine"] = all_same and non_zero
    else:
        flags["flatLine"] = False

    # ── 5. Power factor below threshold ─────────────────────────────────────
    # PF below minimum acceptable level (typically 0.85 per IS)
    # Only flag if PF reading is valid (> 0)
    flags["powerFactorLow"] = (0 < power_factor < pf_min)

    # ── 6. Power factor deviation ────────────────────────────────────────────
    # PF dropped significantly from this meter's own historical average
    # Indicates possible meter tampering or load change
    pf_dev = float(features.get("PowerFactor_Deviation", 0.0))
    flags["powerFactorDeviation"] = pf_dev < -ntl_pf_drop

    # ── 7. NTL suspected ─────────────────────────────────────────────────────
    # Non-Technical Loss pattern:
    # Consumption drops dramatically AND power factor degrades simultaneously
    # Both signals together is the classic meter tampering signature
    energy_vs_avg  = float(features.get("EnergyVsAvgRatio", 1.0))
    drop_threshold = 1.0 - (ntl_drop / 100.0)   # e.g. 0.15 for 85% drop

    flags["ntlSuspected"] = (
        energy_vs_avg < drop_threshold and    # energy collapsed
        pf_dev        < -ntl_pf_drop          # PF also degraded
    )

    # ── 8. Bypass suspected ──────────────────────────────────────────────────
    # Energy drops dramatically BUT peak demand doesn't drop proportionally
    # Machines are still running (peak demand stays) but
    # energy is being bypassed around the meter
    hist_avg      = float(features.get("Hist_AvgEnergy_Wh",    active_wh))
    hist_avg_peak = float(features.get("Hist_AvgPeakDemand_W", peak_demand))
    peak_to_avg   = float(features.get("PeakDemandToAvgRatio", 1.0))

    bypass_drop_threshold = 1.0 - (bypass_drop / 100.0)  # e.g. 0.20

    if hist_avg > 0:
        energy_dropped = energy_vs_avg < bypass_drop_threshold
        # peak demand stayed relatively normal (above 50% of historical)
        peak_stayed    = peak_to_avg > 0.50
        flags["bypassSuspected"] = energy_dropped and peak_stayed
    else:
        flags["bypassSuspected"] = False

    # ── 9. Tariff boundary gaming ────────────────────────────────────────────
    # Consumer consistently reads just below a tariff slab boundary
    # to avoid stepping into a higher rate bracket
    # Check approximate slab boundaries for Indian LT tariffs (in Wh/month)
    SLAB_BOUNDARIES_WH = [
        100_000,   # 100 kWh
        200_000,   # 200 kWh
        300_000,   # 300 kWh
        500_000,   # 500 kWh
    ]
    gaming = False
    for boundary in SLAB_BOUNDARIES_WH:
        lower = boundary * 0.95   # within 5% below boundary
        if lower <= active_wh < boundary:
            # check if last 2 history months were also near this boundary
            if len(e_history) >= 2:
                near_count = sum(
                    1 for v in e_history[-2:]
                    if lower <= v < boundary
                )
                if near_count >= 1:
                    gaming = True
    flags["tariffBoundaryGaming"] = gaming

    return flags


def get_rule_anomaly_type(flags: Dict[str, bool]) -> str:
    """
    Determine the primary anomaly type from rule flags.
    Priority ordered — most severe and actionable first.
    """
    if flags.get("ntlSuspected"):          return "NTL_Suspected"
    if flags.get("bypassSuspected"):       return "BypassSuspected"
    if flags.get("zeroConsumption"):       return "ZeroConsumption"
    if flags.get("billingMismatch"):       return "BillingMismatch"
    if flags.get("rateSumError"):          return "RateSumError"
    if flags.get("tariffBoundaryGaming"):  return "TariffBoundaryGaming"
    if flags.get("powerFactorLow"):        return "PowerFactorAnomaly"
    if flags.get("powerFactorDeviation"):  return "PowerFactorDeviation"
    if flags.get("flatLine"):              return "FlatLine"
    return "None"


def count_active_flags(flags: Dict[str, bool]) -> int:
    """Count how many rules fired."""
    return sum(1 for v in flags.values() if v)