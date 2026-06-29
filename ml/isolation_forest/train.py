"""
Isolation Forest — Training Script
====================================
Trains on consumption anomalies only:
  SuddenDrop, SuddenSpike, ZeroConsumption, NTL_Suspected

BillingMismatch, RateSumError, PowerFactorAnomaly, FlatLine
are handled by the rules engine — treated as normal here.

Features (8):
  EnergyVsAvgRatio       consumption vs own historical average
  VsSeasonalAvgRatio     consumption vs same month last year
  DeviationInStdDevs     how many std devs from average
  MoMChangePct           month over month % change
  TrendSlope             rising or falling trend
  PowerFactor_Deviation  PF change from this meter's normal
  PeakDemandToAvgRatio   peak demand vs historical average (bypass signal)
  IsSummer               1.0 if CDD > 8 (hot month), 0.0 otherwise

Output:
  ml/isolation_forest/model.pkl
  ml/isolation_forest/scaler.pkl
  ml/isolation_forest/config.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "results/simulated_meters.csv"
MODEL_DIR    = "ml/isolation_forest"
MODEL_PATH   = f"{MODEL_DIR}/model.pkl"
SCALER_PATH  = f"{MODEL_DIR}/scaler.pkl"
CONFIG_PATH  = f"{MODEL_DIR}/config.pkl"
MIN_HISTORY  = 3
RANDOM_STATE = 42

# ── Tuning knob ────────────────────────────────────────────────────────────────
# Increase if recall is too low (missing anomalies)
# Decrease if precision is too low (too many false alarms)
CONTAMINATION = 0.003

# ── Features ───────────────────────────────────────────────────────────────────
# These 8 are relative to each meter's own history
# so they work the same regardless of meter size or type
# Order matters — must match runtime feature engine exactly
FEATURES = [
    "EnergyVsAvgRatio",
    "VsSeasonalAvgRatio",
    "DeviationInStdDevs",
    "MoMChangePct",
    "TrendSlope",
    "PowerFactor_Deviation",
    "PeakDemandToAvgRatio",
    "IsSummer",
]

# IF handles only these — rules engine handles the rest
CONSUMPTION_ANOMALIES = [
    "SuddenDrop",
    "SuddenSpike",
    "ZeroConsumption",
    "NTL_Suspected",
]

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Step 1: Load ───────────────────────────────────────────────────────────────
print("=" * 60)
print("  Isolation Forest — Training")
print("=" * 60)
print(f"\nStep 1: Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"  Rows    : {len(df):,}")
print(f"  Columns : {len(df.columns)}")

# ── Step 2: Filter early months ─────────────────────────────────────────────────
# Features are unreliable with fewer than 3 months of history
print(f"\nStep 2: Filtering first {MIN_HISTORY} months per meter ...")
df["MonthIndex"] = df.groupby("MeterId").cumcount()
total_before = len(df)
df = df[df["MonthIndex"] >= MIN_HISTORY].copy()
print(f"  Rows after filter : {len(df):,}")
print(f"  Rows removed      : {total_before - len(df):,}")

# ── Step 3: Label consumption anomalies ──────────────────────────────────────────
print(f"\nStep 3: Labelling consumption anomalies ...")
df["AnomalyLabel"] = df["AnomalyLabel"].fillna("None")
df["IsConsumptionAnomaly"] = df["AnomalyLabel"].isin(CONSUMPTION_ANOMALIES)

c_count    = int(df["IsConsumptionAnomaly"].sum())
total_rows = len(df)
print(f"  Consumption anomaly rows : {c_count:,}")
print(f"  Normal rows              : {total_rows - c_count:,}")
print(f"  Natural anomaly rate     : {c_count/total_rows*100:.4f}%")
print(f"  Contamination set to     : {CONTAMINATION*100:.4f}%")
print(f"\n  Note: BillingMismatch, RateSumError,")
print(f"  PowerFactorAnomaly, FlatLine → rules engine")

# ── Step 4: Check features exist ─────────────────────────────────────────────────
print(f"\nStep 4: Checking {len(FEATURES)} features ...")
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print(f"  ERROR — missing columns: {missing}")
    print(f"  If IsSummer is missing, re-run simulator/generate.py first")
    raise SystemExit(1)
for i, f in enumerate(FEATURES, 1):
    print(f"  {i}. {f}")

# fill any nulls
null_counts = df[FEATURES].isnull().sum()
if null_counts.any():
    print(f"\n  Nulls found — filling with 0:")
    print(null_counts[null_counts > 0])
    df[FEATURES] = df[FEATURES].fillna(0)

# ── Step 5: Scale ──────────────────────────────────────────────────────────────
# StandardScaler: mean=0, std=1 for each feature
# Without this, high-magnitude features dominate tree splits
# Scaler is saved — must apply same scaling at runtime
print(f"\nStep 5: Scaling features ...")
X      = df[FEATURES].values.astype(np.float32)
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
print(f"  Feature matrix : {X_sc.shape}")
print(f"  Feature means after scaling (should be ~0):")
for i, f in enumerate(FEATURES):
    print(f"    {f:<30} mean={X_sc[:,i].mean():.4f}")

# ── Step 6: Train ──────────────────────────────────────────────────────────────
print(f"\nStep 6: Training Isolation Forest ...")
print(f"  n_estimators  = 200")
print(f"  contamination = {CONTAMINATION}")
print(f"  n_jobs        = -1  (all CPU cores)")

model = IsolationForest(
    n_estimators  = 200,
    contamination = CONTAMINATION,
    max_samples   = "auto",
    random_state  = RANDOM_STATE,
    n_jobs        = -1,
)
model.fit(X_sc)
print("  Done!")

# ── Step 7: Predict and evaluate ──────────────────────────────────────────────────
print(f"\nStep 7: Evaluating ...")
y_pred  = model.predict(X_sc)        # 1=normal  -1=anomaly
scores  = model.score_samples(X_sc)  # lower = more anomalous
y_true  = np.where(df["IsConsumptionAnomaly"], -1, 1)

precision = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
recall    = recall_score(y_true,    y_pred, pos_label=-1, zero_division=0)
f1        = f1_score(y_true,        y_pred, pos_label=-1, zero_division=0)

print("\n" + "=" * 60)
print("  Results")
print("=" * 60)
print(f"\nOverall metrics:")
print(f"  Precision : {precision:.4f}")
print(f"    of all meters flagged, how many were real anomalies")
print(f"  Recall    : {recall:.4f}")
print(f"    of all real anomalies, how many did we catch")
print(f"  F1 score  : {f1:.4f}")
print(f"    balance between precision and recall")

# confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
print(f"\nConfusion matrix:")
print(f"                      Predicted anomaly   Predicted normal")
print(f"  Actual anomaly      {cm[0][0]:>17}   {cm[0][1]:>15}")
print(f"  Actual normal       {cm[1][0]:>17}   {cm[1][1]:>15}")
print(f"\n  True positives  : {cm[0][0]:>5}  caught real anomalies")
print(f"  False negatives : {cm[0][1]:>5}  missed real anomalies")
print(f"  False positives : {cm[1][0]:>5}  false alarms")
print(f"  True negatives  : {cm[1][1]:>5}  correctly normal")

# per anomaly type
print(f"\nPer anomaly type breakdown:")
df["IF_Prediction"] = y_pred
df["IF_Score"]      = scores

for atype in sorted(df["AnomalyLabel"].dropna().unique()):
    if atype == "None":
        continue
    mask   = df["AnomalyLabel"] == atype
    total  = int(mask.sum())
    caught = int((df.loc[mask, "IF_Prediction"] == -1).sum())
    rate   = caught / total * 100 if total > 0 else 0
    tag    = "✓ IF target   " if atype in CONSUMPTION_ANOMALIES \
             else "→ rules engine"
    print(f"  {atype:<22}  {caught:>3}/{total:<3}  "
          f"({rate:>5.1f}%)  {tag}")

# score separation
real_sc   = scores[y_true == -1]
norm_sc   = scores[y_true ==  1]
sep       = norm_sc.mean() - real_sc.mean()
print(f"\nScore separation : {sep:.4f}")
print(f"  (higher = better separation between anomaly and normal)")
print(f"  Anomaly mean : {real_sc.mean():.4f}")
print(f"  Normal mean  : {norm_sc.mean():.4f}")

# tuning advice
print(f"\nTuning advice:")
if recall < 0.70:
    print(f"  Recall {recall:.2f} is low → try increasing contamination to "
          f"{CONTAMINATION + 0.002:.3f}")
elif precision < 0.08:
    print(f"  Precision {precision:.2f} is low → try decreasing contamination to "
          f"{CONTAMINATION - 0.001:.3f}")
elif sep < 0.10:
    print(f"  Score separation {sep:.4f} is low → features may need review")
else:
    print(f"  Results look good!")

# ── Step 8: Save ───────────────────────────────────────────────────────────────
print(f"\nStep 8: Saving ...")
with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

config = {
    "features":              FEATURES,
    "consumption_anomalies": CONSUMPTION_ANOMALIES,
    "contamination":         CONTAMINATION,
    "n_estimators":          200,
    "trained_on_rows":       total_rows,
    "min_history_months":    MIN_HISTORY,
    "precision":             float(precision),
    "recall":                float(recall),
    "f1":                    float(f1),
    "score_separation":      float(sep),
}
with open(CONFIG_PATH, "wb") as f: pickle.dump(config, f)

print(f"  model.pkl  → {MODEL_PATH}")
print(f"  scaler.pkl → {SCALER_PATH}")
print(f"  config.pkl → {CONFIG_PATH}")

print("\n" + "=" * 60)
print("  Training complete!")
print("=" * 60)
print(f"""
To retrain with different contamination:
  Open ml/isolation_forest/train.py
  Change CONTAMINATION = {CONTAMINATION}
  Run python ml/isolation_forest/train.py again

Runtime usage:
  Load model.pkl and scaler.pkl once at API startup
  Compute the 8 features from HES history
  Apply scaler.transform() to features
  Call model.predict() → -1 anomaly, 1 normal
  Call model.score_samples() → anomaly score for risk engine

Rules engine handles:
  BillingMismatch · RateSumError · PowerFactorAnomaly · FlatLine
""")