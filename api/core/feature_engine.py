"""
Feature Engine
==============
Computes all ML features from HES history payload.
Pure in-memory computation — no DB lookup.

This mirrors exactly what the simulator computed during training.
Same formulas, same order, same column names.
If these features change, the models must be retrained.

Features computed (8 for Isolation Forest):
  EnergyVsAvgRatio       this month vs meter's own historical avg
  VsSeasonalAvgRatio     this month vs same month last year
  DeviationInStdDevs     how many std devs from average
  MoMChangePct           month over month % change
  TrendSlope             rising or falling trend
  PowerFactor_Deviation  PF change from this meter's historical avg
  PeakDemandToAvgRatio   peak demand vs historical avg (bypass signal)
  IsSummer               1.0 if CDD > 8 (hot month), 0.0 otherwise

Also computes historical stats used by both models and risk scorer:
  Hist_AvgEnergy_Wh
  Hist_StdDev_Wh
  Hist_SeasonalAvg_Wh
  Hist_AvgPowerFactor
  Hist_AvgPeakDemand_W
"""

import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

def get_is_summer(temperature_c: float) -> float:
    cdd = max(temperature_c - 18.0, 0.0)
    return 1.0 if cdd > 8 else 0.0


def safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    """Division with zero guard."""
    try:
        if b and b != 0:
            return float(a / b)
    except Exception:
        pass
    return fallback


def trend_slope(values: List[float]) -> float:
    """Linear regression slope over a list of values."""
    if len(values) < 2:
        return 0.0
    y = np.array(values, dtype=float)
    if np.std(y) == 0:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return round(float(np.polyfit(x, y, 1)[0]), 4)


def get_season(month: int) -> int:
    """Indian season encoding matching the simulator."""
    if month in [4, 5, 6, 7, 8]:   return 1   # summer
    elif month in [9, 10, 11]:      return 2   # monsoon
    else:                           return 3   # winter


def compute_features(
    active_wh:      float,
    billed_wh:      float,
    r1: float, r2: float, r3: float, r4: float,
    power_factor:   float,
    peak_demand_w:  float,
    apparent_vah:   float,
    e_history:      List[float],
    pf_history:     List[float],
    peak_history:   List[float],
    billing_period: str,
    temperature_c:  float = 28.0,
) -> Dict[str, Any]:
    """
    Compute all features from current reading + history.

    Parameters
    ----------
    active_wh      : current month total active energy (Wh)
    billed_wh      : current month billed amount (Wh)
    r1..r4         : rate-wise energy (Wh)
    power_factor   : current month average PF
    peak_demand_w  : current month peak demand (W)
    apparent_vah   : current month apparent energy (VAh)
    e_history      : previous months active energy, oldest first
    pf_history     : previous months power factor, oldest first
    peak_history   : previous months peak demand, oldest first
    billing_period : current period string e.g. "2026-03"
    temperature_c  : average temperature for the billing month (celsius)
    """

    n = len(e_history)

    try:
        period_month = int(billing_period.split("-")[1])
    except Exception:
        period_month = 1

    if n >= 3:
        avg_e  = float(np.mean(e_history))
        std_e  = float(np.std(e_history))
        std_e  = max(std_e, 1.0)

        same_month_vals = []
        for i in range(n - 1, -1, -1):
            hist_month = ((i) % 12) + 1
            if hist_month == period_month:
                same_month_vals.append(e_history[i])

        seas_avg = float(np.mean(same_month_vals)) \
                   if same_month_vals else avg_e
        seas_avg = max(seas_avg, 1.0)

        avg_pf   = float(np.mean(pf_history)) \
                   if pf_history else 0.95
        avg_peak = float(np.mean(peak_history)) \
                   if peak_history else max(peak_demand_w, 0.01)
        avg_peak = max(avg_peak, 0.01)

        energy_vs_avg   = round(safe_div(active_wh, avg_e),    6)
        vs_seasonal     = round(safe_div(active_wh, seas_avg), 6)
        dev_std         = round(
            safe_div(active_wh - avg_e, std_e if std_e > 0 else 1), 4
        )
        mom_pct         = round(
            safe_div(active_wh - e_history[-1], e_history[-1]) * 100, 4
        ) if e_history[-1] > 0 else 0.0
        t_slope         = trend_slope(e_history[-12:])
        pf_dev          = round(power_factor - avg_pf, 4)
        peak_to_avg     = round(safe_div(peak_demand_w, avg_peak), 6)

    else:
        avg_e         = max(active_wh, 1.0)
        std_e         = 1.0
        seas_avg      = avg_e
        avg_pf        = 0.95
        avg_peak      = max(peak_demand_w, 0.01)
        energy_vs_avg = 1.0
        vs_seasonal   = 1.0
        dev_std       = 0.0
        mom_pct       = 0.0
        t_slope       = 0.0
        pf_dev        = 0.0
        peak_to_avg   = 1.0

    # weather feature
    is_summer = get_is_summer(temperature_c)

    # integrity check features
    rate_sum_diff    = round((r1+r2+r3+r4) - active_wh, 4)
    billing_diff     = round(billed_wh - active_wh, 4)
    billing_diff_pct = round(
        safe_div(billed_wh - active_wh, active_wh) * 100, 4
    ) if active_wh > 0 else 0.0
    zero_rate_count  = sum(1 for r in [r1,r2,r3,r4] if r == 0)
    app_vs_active    = round(
        safe_div(apparent_vah, active_wh), 6
    ) if active_wh > 0 else 1.0

    return {
        # historical stats
        "Hist_AvgEnergy_Wh":     round(avg_e,    2),
        "Hist_StdDev_Wh":        round(std_e,    4),
        "Hist_SeasonalAvg_Wh":   round(seas_avg, 2),
        "Hist_AvgPowerFactor":   round(avg_pf,   4),
        "Hist_AvgPeakDemand_W":  round(avg_peak, 2),

        # 8 IF features — must match FEATURES list in train.py
        "EnergyVsAvgRatio":      energy_vs_avg,
        "VsSeasonalAvgRatio":    vs_seasonal,
        "DeviationInStdDevs":    dev_std,
        "MoMChangePct":          mom_pct,
        "TrendSlope":            t_slope,
        "PowerFactor_Deviation": pf_dev,
        "PeakDemandToAvgRatio":  peak_to_avg,
        "IsSummer":              is_summer,

        # integrity checks — for rules engine
        "RateSumDiff":           rate_sum_diff,
        "BillingVsMeterDiff":    billing_diff,
        "BillingVsMeterDiffPct": billing_diff_pct,
        "ZeroRateCount":         zero_rate_count,
        "ApparentVsActiveRatio": app_vs_active,
    }


def get_if_feature_vector(features: Dict[str, Any]) -> List[float]:
    """
    Extract exactly the 8 features Isolation Forest expects.
    Order must match FEATURES list in ml/isolation_forest/train.py.
    """
    IF_FEATURES = [
        "EnergyVsAvgRatio",
        "VsSeasonalAvgRatio",
        "DeviationInStdDevs",
        "MoMChangePct",
        "TrendSlope",
        "PowerFactor_Deviation",
        "PeakDemandToAvgRatio",
        "IsSummer",
    ]
    return [float(features.get(f, 0.0)) for f in IF_FEATURES]


def get_tft_sequence(
    e_history:    List[float],
    pf_history:   List[float],
    billing_period: str,
    energy_mean:  float,
    energy_std:   float,
    global_stats: Dict[str, Dict],
    encoder_len:  int = 12,
    temperature_c: float = 28.0,
) -> List[List[float]]:
    """
    Build the 12-month continuous feature sequence for TFT.
    Normalises using the same stats from training.
    Pads with mean values if history is shorter than encoder_len.

    Returns list of [energy_norm, pf_norm, month_norm, season_norm]
    for each of the last encoder_len months.
    """
    try:
        period_month  = int(billing_period.split("-")[1])
        period_season = get_season(period_month)
    except Exception:
        period_month  = 1
        period_season = 3

    def norm_g(val: float, col: str) -> float:
        mu  = float(global_stats[col]["mean"])
        std = float(global_stats[col]["std"])
        return float((val - mu) / std) if std > 0 else 0.0

    e_std  = max(float(energy_std), 1.0)
    e_mean = float(energy_mean)

    e_seq  = ([e_mean]  * (encoder_len - len(e_history))
              + list(e_history))[-encoder_len:]
    pf_seq = ([float(global_stats["PowerFactor_Avg"]["mean"])]
              * (encoder_len - len(pf_history))
              + list(pf_history))[-encoder_len:]

    sequence = []
    for i in range(encoder_len):
        e_norm  = float((float(e_seq[i]) - e_mean) / e_std)
        pf_norm = norm_g(float(pf_seq[i]), "PowerFactor_Avg")
        m_norm  = norm_g(float(period_month),  "Month")
        s_norm  = norm_g(float(period_season), "Season")
        is_summer = get_is_summer(temperature_c)
        sequence.append([e_norm, pf_norm, m_norm, s_norm,is_summer])

    return sequence


def get_model_confidence(history_months: int) -> str:
    """
    Assess how confident the ML models can be based on history length.

    < 3 months  → INSUFFICIENT — only rules engine works
    3-5 months  → LOW           — ML works but limited seasonal data
    6-11 months → MEDIUM        — ML works, partial seasonal data
    12+ months  → HIGH          — ML fully effective
    """
    if history_months < 3:
        return "INSUFFICIENT"
    elif history_months < 6:
        return "LOW"
    elif history_months < 12:
        return "MEDIUM"
    else:
        return "HIGH"