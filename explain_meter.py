"""
Meter Analysis — Display What the API Already Computed
=========================================================
Given a result JSON from /api/validate (which already contains
features, IF score, TFT prediction, rules, and risk score),
prints a detailed human-readable breakdown.

Does NOT recompute anything — purely formats what the API returned.

Usage:
  python explain_meter.py --result M2GDDAC1_result.json
  python explain_meter.py --folder "C:/meter_data/Bill"
  python explain_meter.py --file "meter.xlsx"
"""

import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent

NORMAL_RANGES = {
    "EnergyVsAvgRatio":      (0.70, 1.35,  "consumption vs own avg"),
    "VsSeasonalAvgRatio":    (0.72, 1.30,  "consumption vs same month last year"),
    "DeviationInStdDevs":    (-2.5, 2.5,   "std devs from historical avg"),
    "MoMChangePct":          (-25,  25,    "% change from last month"),
    "TrendSlope":            (-500, 500,   "rising/falling trend"),
    "PowerFactor_Deviation": (-0.05, 0.02, "PF drop from meter's normal"),
    "PeakDemandToAvgRatio":  (0.50, 1.60,  "peak demand vs historical avg"),
    "IsSummer":              (0,    1,     "1.0 = hot month, 0.0 = cool month"),
}

IF_FEATURES = list(NORMAL_RANGES.keys())


def display(payload: dict, result: dict):
    meter_id = result.get("meterId", "?")
    period   = result.get("billingPeriod", "?")
    features = result.get("features", {})
    mo       = result.get("modelOutputs", {})
    sc       = result.get("scoreComponents", {})
    rf       = result.get("ruleFlags", {})

    history  = payload.get("history", []) if payload else []
    current  = payload.get("currentReadings", []) if payload else []

    e_hist  = []
    pf_hist = []
    periods = []
    for m in sorted(history, key=lambda x: x.get("billingPeriod", "")):
        periods.append(m.get("billingPeriod", ""))
        e_val, pf_val = 0, 0
        for r in m.get("readings", []):
            if r.get("obisCode") == "1.0.1.8.0.255":
                e_val = float(r.get("value", 0))
            if r.get("obisCode") == "1.0.13.0.0.255":
                pf_val = float(r.get("value", 0))
        e_hist.append(e_val)
        pf_hist.append(pf_val)

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  METER ANALYSIS — {meter_id}  ·  Period: {period}")
    print(sep)

    # ── History summary ────────────────────────────────────────────────────────
    if e_hist:
        avg_e = features.get("Hist_AvgEnergy_Wh", sum(e_hist)/len(e_hist))
        print(f"\n{'─'*65}")
        print(f"  HISTORY — last {len(e_hist)} months")
        print(f"{'─'*65}")
        print(f"  {'Month':<10} {'Energy (Wh)':>14}  {'PF':>6}  {'vs Avg':>8}")
        print(f"  {'─'*10} {'─'*14}  {'─'*6}  {'─'*8}")
        for i, (bp, e) in enumerate(zip(periods, e_hist)):
            pf_h   = pf_hist[i] if i < len(pf_hist) else 0
            vs_avg = round(e / avg_e * 100, 1) if avg_e > 0 else 100
            marker = " ← last month before anomaly" if i == len(e_hist)-1 else ""
            print(f"  {bp:<10} {e:>14,.0f}  {pf_h:>6.3f}  {vs_avg:>7.1f}%{marker}")

        print(f"\n  Historical average   : {features.get('Hist_AvgEnergy_Wh',0):,.0f} Wh")
        print(f"  Historical std dev   : {features.get('Hist_StdDev_Wh',0):,.0f} Wh  (normal variation)")
        print(f"  Avg power factor     : {features.get('Hist_AvgPowerFactor',0):.4f}")
        print(f"  Avg peak demand      : {features.get('Hist_AvgPeakDemand_W',0):.2f} W")

    # ── Current month ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  CURRENT MONTH — {period}")
    print(f"{'─'*65}")
    print(f"  Active energy   : {result.get('actualWh',0):,.0f} Wh")
    print(f"  Power factor    : (avg historical: {features.get('Hist_AvgPowerFactor',0):.4f})")
    print(f"  Peak demand     : (avg historical: {features.get('Hist_AvgPeakDemand_W',0):.2f} W)")

    # ── IF feature analysis ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  ISOLATION FOREST — Feature Analysis")
    print(f"{'─'*65}")
    print(f"  {'Feature':<28} {'Value':>10}  {'Normal Range':>20}  Status")
    print(f"  {'─'*28} {'─'*10}  {'─'*20}  {'─'*10}")

    anomalous_features = []
    for feat in IF_FEATURES:
        if feat not in features:
            continue
        val          = features[feat]
        lo, hi, desc = NORMAL_RANGES[feat]
        in_range     = lo <= val <= hi
        status       = "✓ normal" if in_range else "✗ ANOMALOUS"
        range_str    = f"[{lo}, {hi}]"
        if not in_range:
            anomalous_features.append((feat, val, lo, hi))
        print(f"  {feat:<28} {val:>10.4f}  {range_str:>20}  {status}")

    print(f"\n  IF anomaly score : {mo.get('ifScore', 0):.4f}  (threshold ≈ -0.512)")
    print(f"  IF decision      : {'ANOMALY ✗' if mo.get('ifIsAnomaly') else 'NORMAL ✓'}")

    if anomalous_features:
        print(f"\n  Features outside normal range:")
        for feat, val, lo, hi in anomalous_features:
            _, _, desc = NORMAL_RANGES[feat]
            direction  = "BELOW" if val < lo else "ABOVE"
            print(f"    {feat}")
            print(f"      Value: {val:.4f}  |  Normal: [{lo}, {hi}]  |  {direction} range")
            print(f"      Meaning: {desc}")
    else:
        print(f"\n  All features within normal range")

    if mo.get("ifIsAnomaly"):
        n_anom = len(anomalous_features)
        print(f"\n  Why IF flagged this meter:")
        if n_anom >= 2:
            print(f"    {n_anom} features are simultaneously outside normal range.")
            print(f"    This combination has almost never appeared in normal meters.")
        elif n_anom == 1:
            print(f"    1 feature is extreme enough to push the anomaly score below threshold.")
        else:
            print(f"    The combination of features is statistically unusual")
            print(f"    even though each individual feature appears borderline.")

    # ── TFT prediction ─────────────────────────────────────────────────────────
    p10 = mo.get("tftP10", 0)
    p50 = mo.get("tftPredictedWh", 0)
    p90 = mo.get("tftP90", 0)
    dev = result.get("deviationPct", 0)
    actual_wh = result.get("actualWh", 0)

    print(f"\n{'─'*65}")
    print(f"  TFT PREDICTION — what the model expected for {period}")
    print(f"{'─'*65}")
    print(f"  Based on: last 12 months of this meter's own consumption pattern")
    print(f"\n  P10 lower bound  : {p10:,.0f} Wh  (10th percentile — minimum expected)")
    print(f"  P50 forecast     : {p50:,.0f} Wh  (median forecast — most likely value)")
    print(f"  P90 upper bound  : {p90:,.0f} Wh  (90th percentile — maximum expected)")
    print(f"\n  Actual reading   : {actual_wh:,.0f} Wh")
    print(f"  Deviation        : {dev:.1f}%  ({'+' if dev>0 else ''}{dev:.1f}% from P50)")

    if actual_wh < p10:
        gap = p10 - actual_wh
        print(f"\n  Actual is {gap:,.0f} Wh BELOW the lower bound (P10)")
        print(f"  Getting {actual_wh:,.0f} Wh is outside the expected range → SuddenDrop flagged")
    elif actual_wh > p90:
        gap = actual_wh - p90
        print(f"\n  Actual is {gap:,.0f} Wh ABOVE the upper bound (P90)")
        print(f"  Getting {actual_wh:,.0f} Wh is outside the expected range → SuddenSpike flagged")
    else:
        print(f"\n  Actual is within the expected P10-P90 range")
        print(f"  TFT does not flag this as anomalous")

    # ── Combined summary ───────────────────────────────────────────────────────
    fired_rules = [k for k, v in rf.items() if v]

    print(f"\n{'─'*65}")
    print(f"  COMBINED VERDICT")
    print(f"{'─'*65}")
    print(f"  Isolation Forest : {sc.get('isolation_forest',0):.1f} / 40 pts")
    print(f"  TFT deviation    : {sc.get('tft_deviation',0)} / 30 pts")
    print(f"  Rules engine     : {sc.get('rules',0)} / 50 pts")
    print(f"  ─────────────────────────────")
    print(f"  Total risk score : {result.get('riskScore',0)} / 100")
    print(f"  Severity         : {result.get('severity','?')}")
    print(f"  Anomaly type     : {result.get('anomalyType','?')}")
    if fired_rules:
        print(f"  Rules fired      : {', '.join(fired_rules)}")
    print(f"  Recommendation   : {result.get('recommendation','?')}")
    print(f"\n{sep}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Display the feature breakdown the API already computed"
    )
    parser.add_argument("--result",
                        help="Path to result JSON from parse_and_validate.py")
    parser.add_argument("--payload",
                        help="Path to payload JSON (optional, for history table)")
    parser.add_argument("--folder",
                        help="Parse folder, call API, and display")
    parser.add_argument("--file",
                        help="Parse single file, call API, and display")
    parser.add_argument("--location", default="Chennai")
    args = parser.parse_args()

    # ── Option 1: use saved result + payload JSON ──────────────────────────────
    if args.result:
        result_path  = Path(args.result)
        payload_path = result_path.parent / (result_path.stem.replace('_result','_payload') + '.json')

        with open(result_path) as f:
            result = json.load(f)

        payload = None
        if payload_path.exists():
            with open(payload_path) as f:
                payload = json.load(f)
        else:
            print(f"  Note: payload not found at {payload_path}")
            print(f"  History table will be skipped. Re-run with --save for full detail.")

        display(payload, result)
        return

    # ── Option 2: parse file/folder, call API fresh ────────────────────────────
    sys.path.insert(0, str(BASE_DIR))
    from parse_and_validate import (
        parse_pdf, parse_excel, parse_csv, parse_folder,
        build_payload, call_api, is_cumulative, subtract_cumulative
    )

    if args.folder:
        meter_id, periods = parse_folder(args.folder)
    elif args.file:
        ext = Path(args.file).suffix.lower()
        if ext == '.pdf':
            meter_id, periods = parse_pdf(args.file)
        elif ext in ('.xlsx', '.xls'):
            meter_id, periods = parse_excel(args.file)
        else:
            meter_id, periods = parse_csv(args.file)
    else:
        print("ERROR: Provide --result, --file, or --folder")
        parser.print_help()
        sys.exit(1)

    if len(periods) < 2:
        print("ERROR: Need at least 2 billing periods for analysis")
        sys.exit(1)

    if is_cumulative(periods):
        print(f"  Data type: CUMULATIVE — applying subtraction...")
        periods = subtract_cumulative(periods)

    payload = build_payload(meter_id, periods, location=args.location)
    print(f"\nCalling API for validation ...")
    try:
        result = call_api(payload, "http://127.0.0.1:8000")
    except RuntimeError as e:
        print(f"API Error: {e}")
        sys.exit(1)

    display(payload, result)


if __name__ == "__main__":
    main()