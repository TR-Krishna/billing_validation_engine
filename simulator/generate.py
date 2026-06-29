"""
Billing Validation Engine — Dataset Simulator
==============================================
Generates 360,000 rows of simulated smart meter billing data.
Matches real DLMS/COSEM IS 16444 billing profile structure.

Meter types  : Residential (LT-1), Commercial (LT-2), Industrial (LT-3)
Months       : 36 per meter
Anomalies    : 8 types injected into ~500 random meters
Output       : results/simulated_meters.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

# fetch_weather.py is in BVEngine root, this file is in simulator/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fetch_weather import fetch_monthly_temperature, get_is_summer

np.random.seed(42)

NUM_METERS    = 10000
NUM_MONTHS    = 36
START_DATE    = date(2023, 1, 1)
OUTPUT_PATH   = "results/simulated_meters.csv"
ANOMALY_COUNT = 500
INR_PER_WH    = 0.008

os.makedirs("results", exist_ok=True)

METER_TYPES = {
    "Residential": {
        "tariff": "LT-1", "avg_wh_min": 8000,  "avg_wh_max": 15000,
        "variation": 0.12, "pf_min": 0.92, "pf_max": 0.99,
        "r1": 0.30, "r2": 0.40, "r3": 0.20, "r4": 0.10,
    },
    "Commercial": {
        "tariff": "LT-2", "avg_wh_min": 30000, "avg_wh_max": 60000,
        "variation": 0.08, "pf_min": 0.90, "pf_max": 0.98,
        "r1": 0.25, "r2": 0.45, "r3": 0.20, "r4": 0.10,
    },
    "Industrial": {
        "tariff": "LT-3", "avg_wh_min": 80000, "avg_wh_max": 150000,
        "variation": 0.05, "pf_min": 0.88, "pf_max": 0.97,
        "r1": 0.20, "r2": 0.38, "r3": 0.22, "r4": 0.20,
    },
}

LOCATIONS     = ["Chennai", "Mumbai", "Delhi", "Bangalore", "Hyderabad"]
ANOMALY_TYPES = ["SuddenDrop","SuddenSpike","ZeroConsumption","BillingMismatch",
                 "PowerFactorAnomaly","NTL_Suspected","FlatLine","RateSumError"]

def seasonal_factor(month, mtype):
    return 1.20 if mtype == "Residential" and month in [4,5,6,7,8] else 1.0

def get_season(month):
    if month in [4,5,6,7,8]: return 1
    elif month in [9,10,11]:  return 2
    else:                     return 3

def split_rates(total, cfg):
    r1 = round(total * cfg["r1"] * np.random.uniform(0.93, 1.07), 2)
    r2 = round(total * cfg["r2"] * np.random.uniform(0.93, 1.07), 2)
    r3 = round(total * cfg["r3"] * np.random.uniform(0.93, 1.07), 2)
    r4 = round(max(total - r1 - r2 - r3, 0), 2)
    return r1, r2, r3, r4

def apparent_energy(active, pf):
    return round(active / max(pf, 0.01), 2)

def reactive_energy(active, app):
    r   = round(max(app**2 - active**2, 0)**0.5, 2)
    qi  = round(r * np.random.uniform(0.25, 0.40), 2)
    return qi, round(r - qi, 2)

def safe_div(a, b):
    return round(a / b, 6) if b and b != 0 else 0.0

def slope(vals):
    if len(vals) < 2: return 0.0
    y = np.array(vals, dtype=float)
    return round(float(np.polyfit(np.arange(len(y)), y, 1)[0]), 4) \
           if np.std(y) > 0 else 0.0

def compute_features(active, billed, r1, r2, r3, r4,
                     pf, peak_w, e_hist, pf_hist, peak_hist, bd,
                     temperature_c=28.0):
    n = len(e_hist)
    if n >= 3:
        avg_e    = float(np.mean(e_hist))
        std_e    = max(float(np.std(e_hist)), 1.0)
        same_m   = [e_hist[i] for i in range(n)
                    if (START_DATE+relativedelta(months=i)).month == bd.month]
        seas_avg = float(np.mean(same_m)) if same_m else avg_e
        seas_avg = max(seas_avg, 1.0)
        avg_pf   = float(np.mean(pf_hist)) if pf_hist else 0.95
        avg_peak = max(float(np.mean(peak_hist)), 0.01) if peak_hist else max(peak_w, 0.01)

        ev_avg   = safe_div(active, avg_e)
        vs_seas  = safe_div(active, seas_avg)
        dev_std  = round(safe_div(active - avg_e, std_e), 4)
        mom_pct  = round(safe_div(active - e_hist[-1], e_hist[-1]) * 100, 4) \
                   if e_hist[-1] > 0 else 0.0
        t_slope  = slope(e_hist[-12:])
        pf_dev   = round(pf - avg_pf, 4)
        pk_ratio = round(safe_div(peak_w, avg_peak), 6)
    else:
        avg_e    = max(active, 1.0)
        std_e    = 1.0
        seas_avg = avg_e
        avg_pf   = 0.95
        avg_peak = max(peak_w, 0.01)
        ev_avg   = 1.0
        vs_seas  = 1.0
        dev_std  = 0.0
        mom_pct  = 0.0
        t_slope  = 0.0
        pf_dev   = 0.0
        pk_ratio = 1.0

    app_vah       = apparent_energy(active, pf)
    rate_sum_diff = round((r1+r2+r3+r4) - active, 4)
    bill_diff     = round(billed - active, 4)
    bill_diff_pct = round(safe_div(billed - active, active) * 100, 4) \
                    if active > 0 else 0.0
    zero_rates    = sum(1 for r in [r1,r2,r3,r4] if r == 0)
    app_vs_act    = round(safe_div(app_vah, active), 6) if active > 0 else 1.0
    is_summer     = get_is_summer(temperature_c)

    return dict(
        Hist_AvgEnergy_Wh     = round(avg_e,    2),
        Hist_StdDev_Wh        = round(std_e,    4),
        Hist_SeasonalAvg_Wh   = round(seas_avg, 2),
        Hist_AvgPowerFactor   = round(avg_pf,   4),
        Hist_AvgPeakDemand_W  = round(avg_peak, 2),
        EnergyVsAvgRatio      = round(ev_avg,   6),
        VsSeasonalAvgRatio    = round(vs_seas,  6),
        DeviationInStdDevs    = dev_std,
        MoMChangePct          = mom_pct,
        TrendSlope            = t_slope,
        PowerFactor_Deviation = pf_dev,
        PeakDemandToAvgRatio  = pk_ratio,
        IsSummer              = is_summer,
        RateSumDiff           = rate_sum_diff,
        BillingVsMeterDiff    = bill_diff,
        BillingVsMeterDiffPct = bill_diff_pct,
        ZeroRateCount         = zero_rates,
        ApparentVsActiveRatio = app_vs_act,
    )

# ── Assign meters ──────────────────────────────────────────────────────────────
meter_ids = [f"M-{str(i).zfill(5)}" for i in range(1, NUM_METERS+1)]
type_pool = (["Residential"]*3334 + ["Commercial"]*3333 + ["Industrial"]*3333)
np.random.shuffle(type_pool)

meter_meta = {}
for mid, mtype in zip(meter_ids, type_pool):
    cfg = METER_TYPES[mtype]
    meter_meta[mid] = dict(
        type=mtype, tariff=cfg["tariff"],
        location=np.random.choice(LOCATIONS),
        base_wh=np.random.uniform(cfg["avg_wh_min"], cfg["avg_wh_max"]),
        pf_base=np.random.uniform(cfg["pf_min"],     cfg["pf_max"]),
    )

anomaly_meters = np.random.choice(meter_ids, size=ANOMALY_COUNT, replace=False)
anomaly_map    = {}
for mid in anomaly_meters:
    atype = np.random.choice(ANOMALY_TYPES)
    s     = np.random.randint(6, NUM_MONTHS-4)
    bad   = list(range(s, s+4)) if atype == "FlatLine" \
            else [np.random.randint(6, NUM_MONTHS)]
    anomaly_map[mid] = (atype, bad)

# ── Fetch weather cache ────────────────────────────────────────────────────────
print("Fetching historical weather data ...")
weather_cache = {}
for loc in LOCATIONS:
    for m in range(NUM_MONTHS):
        bd  = START_DATE + relativedelta(months=m)
        key = f"{loc}_{bd.strftime('%Y-%m')}"
        weather_cache[key] = fetch_monthly_temperature(loc, bd.strftime("%Y-%m"))
print(f"  Weather cache ready — {len(weather_cache)} entries\n")

# ── Generate ───────────────────────────────────────────────────────────────────
all_rows        = []
anomaly_summary = {a: 0 for a in ANOMALY_TYPES}

for idx, mid in enumerate(meter_ids):
    if (idx+1) % 1000 == 0:
        print(f"  Processing meter {idx+1:,} / {NUM_METERS:,} ...")

    meta    = meter_meta[mid]
    mtype   = meta["type"]
    cfg     = METER_TYPES[mtype]
    base_wh = meta["base_wh"]
    pf_base = meta["pf_base"]
    atype, bad_months = anomaly_map.get(mid, (None, []))

    e_hist = []; pf_hist = []; peak_hist = []

    for month_idx in range(NUM_MONTHS):
        bd        = START_DATE + relativedelta(months=month_idx)
        var       = np.random.uniform(1-cfg["variation"], 1+cfg["variation"])
        normal_wh = round(base_wh * var * seasonal_factor(bd.month, mtype), 2)
        normal_pf = float(np.clip(
            np.random.uniform(pf_base-0.02, pf_base+0.01), 0.80, 1.00))

        active  = normal_wh
        pf      = normal_pf
        pf_imp  = round(min(pf + np.random.uniform(0, 0.02), 1.0), 4)
        billed  = normal_wh
        label   = "None"
        is_anom = False
        rate_ok = True

        if month_idx in bad_months and atype:
            is_anom = True
            label   = atype
            if   atype == "SuddenDrop":
                active = round(normal_wh * np.random.uniform(0.05, 0.10), 2)
                billed = active
            elif atype == "SuddenSpike":
                active = round(normal_wh * np.random.uniform(3.0, 4.0), 2)
                billed = active
            elif atype == "ZeroConsumption":
                active = 0.0; billed = 0.0
            elif atype == "BillingMismatch":
                billed = round(active * np.random.uniform(0.60, 0.80), 2)
            elif atype == "PowerFactorAnomaly":
                pf     = round(np.random.uniform(0.70, 0.82), 4)
                pf_imp = round(np.random.uniform(0.70, 0.85), 4)
            elif atype == "NTL_Suspected":
                active = round(normal_wh * np.random.uniform(0.05, 0.10), 2)
                billed = active
                pf     = round(np.random.uniform(0.72, 0.80), 4)
                pf_imp = round(np.random.uniform(0.72, 0.82), 4)
            elif atype == "FlatLine":
                active = round(base_wh, 2); billed = active
            elif atype == "RateSumError":
                rate_ok = False
            anomaly_summary[atype] += 1

        if rate_ok:
            r1, r2, r3, r4 = split_rates(active, cfg)
        else:
            r1 = round(active*0.30,2); r2 = round(active*0.40,2)
            r3 = round(active*0.20,2); r4 = round(active*0.15,2)

        app_vah = apparent_energy(active, pf)
        qi, qiv = reactive_energy(active, app_vah)

        if is_anom and atype == "NTL_Suspected":
            peak_w = round((base_wh/720) * np.random.uniform(1.8, 2.5), 2)
        else:
            avg_h  = active / 720 if active > 0 else 0
            peak_w = round(avg_h * np.random.uniform(1.8, 2.8), 2)

        read_type   = "Estimated" if np.random.random() < 0.05 else "Actual"
        bill_inr    = round(billed * INR_PER_WH, 2)
        weather_key = f"{meta['location']}_{bd.strftime('%Y-%m')}"
        temperature = weather_cache.get(weather_key, 28.0)

        feats = compute_features(
            active, billed, r1, r2, r3, r4,
            pf, peak_w, e_hist[:], pf_hist[:], peak_hist[:], bd,
            temperature_c=temperature,
        )

        all_rows.append({
            "MeterId":                  mid,
            "MeterType":                mtype,
            "TariffSlab":               meta["tariff"],
            "Location":                 meta["location"],
            "BillingPeriodEnd":         bd.strftime("%Y-%m-%d"),
            "Month":                    bd.month,
            "Season":                   get_season(bd.month),
            "IsSummer":                 feats["IsSummer"],
            "ActiveEnergy_Total_Wh":    active,
            "ActiveEnergy_Rate1_Wh":    r1,
            "ActiveEnergy_Rate2_Wh":    r2,
            "ActiveEnergy_Rate3_Wh":    r3,
            "ActiveEnergy_Rate4_Wh":    r4,
            "PeakDemand_Total_W":       peak_w,
            "ApparentEnergy_Total_VAh": app_vah,
            "PowerFactor_Avg":          round(pf, 4),
            "PowerFactor_Import":       pf_imp,
            "ReactiveEnergy_QI_VArh":   qi,
            "ReactiveEnergy_QII_VArh":  0.0,
            "ReactiveEnergy_QIII_VArh": 0.0,
            "ReactiveEnergy_QIV_VArh":  qiv,
            "BilledAmount_Wh":          billed,
            "BillingReadType":          read_type,
            "BillAmountINR":            bill_inr,
            **feats,
            "AnomalyLabel":             label,
            "IsAnomalyMonth":           is_anom,
        })

        e_hist.append(active)
        pf_hist.append(pf)
        peak_hist.append(peak_w)

print("\nSaving ...")
df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*56)
print(f"  Rows    : {len(df):,}")
print(f"  Meters  : {NUM_METERS:,}")
print(f"  Months  : {NUM_MONTHS}")
print(f"  Columns : {len(df.columns)}")
print(f"  Output  : {OUTPUT_PATH}")
print("="*56)
print("\nAnomaly breakdown:")
for a, c in anomaly_summary.items():
    print(f"  {a:<22}  {c:>4} rows")
print(f"\n  Normal rows  : {len(df[~df['IsAnomalyMonth']]):,}")
print(f"  Anomaly rows : {len(df[df['IsAnomalyMonth']]):,}")
print("="*56)

ntl = df[df["AnomalyLabel"]=="NTL_Suspected"]
if len(ntl):
    s = ntl.iloc[0]
    print(f"\nSpot check — NTL_Suspected:")
    print(f"  ActiveEnergy        : {s['ActiveEnergy_Total_Wh']:,.2f} Wh")
    print(f"  Hist_AvgEnergy      : {s['Hist_AvgEnergy_Wh']:,.2f} Wh")
    print(f"  EnergyVsAvgRatio    : {s['EnergyVsAvgRatio']:.4f}")
    print(f"  PeakDemand          : {s['PeakDemand_Total_W']:,.2f} W")
    print(f"  Hist_AvgPeakDemand  : {s['Hist_AvgPeakDemand_W']:,.2f} W")
    print(f"  PeakDemandToAvgRatio: {s['PeakDemandToAvgRatio']:.4f}")
    print(f"  PowerFactor_Avg     : {s['PowerFactor_Avg']:.4f}")
    print(f"  PowerFactor_Dev     : {s['PowerFactor_Deviation']:.4f}")
    print(f"  IsSummer            : {s['IsSummer']:.1f}")