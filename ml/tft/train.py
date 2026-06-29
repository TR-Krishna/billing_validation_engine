"""
Temporal Fusion Transformer — Training Script
=============================================
Pure PyTorch implementation — no pytorch-forecasting dependency.
Runs on NVIDIA GPU via CUDA.

Architecture:
  Categorical embeddings (MeterType, TariffSlab, Location)
  + continuous sequence (12 months of 5 features incl. IsSummer)
  → LSTM encoder (2 layers)
  → Multi-head attention (4 heads)
  → Dense output → p10, p50, p90

Training:
  Months 1-30  → training   (180,000 samples)
  Months 31-36 → validation (60,000 samples)
  Loss: Quantile loss (pinball loss)
  Early stopping: patience 7

Output:
  ml/tft/tft_model.pt
  ml/tft/config.pkl
"""

import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# fetch_weather.py is in BVEngine root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from fetch_weather import get_is_summer

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "results/simulated_meters.csv"
MODEL_DIR    = "ml/tft"
MODEL_PATH   = f"{MODEL_DIR}/tft_model.pt"
CONFIG_PATH  = f"{MODEL_DIR}/config.pkl"

ENCODER_LEN  = 12     # look back 12 months
TRAIN_CUTOFF = 30     # months 1-30 for training
HIDDEN_SIZE  = 64
N_HEADS      = 4
DROPOUT      = 0.1
BATCH_SIZE   = 256
MAX_EPOCHS   = 50
LR           = 0.001
PATIENCE     = 7
QUANTILES    = [0.1, 0.5, 0.9]

# continuous features fed into model per timestep — now 5 with IsSummer
CONT_FEATURES = [
    "energy_norm",
    "PowerFactor_Avg_norm",
    "Month_norm",
    "Season_norm",
    "IsSummer",
]

CONSUMPTION_ANOMALIES = [
    "SuddenDrop", "SuddenSpike", "ZeroConsumption", "NTL_Suspected"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  Temporal Fusion Transformer — Training")
print(f"  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ── Step 1: Load and prepare ───────────────────────────────────────────────────
print("\nStep 1: Loading data ...")
df = pd.read_csv(DATA_PATH)
df = df.sort_values(["MeterId","BillingPeriodEnd"]).reset_index(drop=True)
df["TimeIdx"]     = df.groupby("MeterId").cumcount()
df["AnomalyLabel"]= df["AnomalyLabel"].fillna("None")
print(f"  Rows   : {len(df):,}")
print(f"  Meters : {df['MeterId'].nunique():,}")

# encode categoricals as integers
for col in ["MeterType","TariffSlab","Location"]:
    df[col+"_enc"] = pd.factorize(df[col])[0]

cat_dims = {
    "MeterType_enc":  int(df["MeterType_enc"].nunique()),
    "TariffSlab_enc": int(df["TariffSlab_enc"].nunique()),
    "Location_enc":   int(df["Location_enc"].nunique()),
}
print(f"  Categorical dims : {cat_dims}")

# clip outliers
df["ActiveEnergy_Total_Wh"] = df["ActiveEnergy_Total_Wh"].clip(0)
df["PowerFactor_Avg"]       = df["PowerFactor_Avg"].clip(0.01, 1.0)

# per-meter normalisation for target energy
# use only training months to compute stats (no data leakage)
train_df    = df[df["TimeIdx"] < TRAIN_CUTOFF]
meter_stats = train_df.groupby("MeterId")["ActiveEnergy_Total_Wh"].agg(
    ["mean","std"]
).rename(columns={"mean":"energy_mean","std":"energy_std"})
meter_stats["energy_std"] = meter_stats["energy_std"].fillna(1.0).clip(lower=1.0)
df = df.merge(meter_stats, on="MeterId", how="left")
df["energy_norm"] = (
    (df["ActiveEnergy_Total_Wh"] - df["energy_mean"]) / df["energy_std"]
).fillna(0.0)

# global normalisation for other features
global_stats = {}
for col in ["PowerFactor_Avg","Month","Season"]:
    mu  = float(train_df[col].mean())
    std = float(max(train_df[col].std(), 1e-6))
    df[col+"_norm"] = (df[col] - mu) / std
    global_stats[col] = {"mean": mu, "std": std}

# IsSummer is already in the dataset (binary 0.0 or 1.0) — no normalisation needed
if "IsSummer" not in df.columns:
    print("  WARNING: IsSummer column not found — computing from scratch")
    df["IsSummer"] = 0.0  # fallback

print(f"  IsSummer distribution: {df['IsSummer'].value_counts().to_dict()}")
print(f"  Normalisation: per-meter for energy, global for PF/Month/Season, binary for IsSummer")

# ── Step 2: Dataset ────────────────────────────────────────────────────────────
class MeterDataset(Dataset):
    def __init__(self, df, mode="train"):
        self.samples = []
        cat_cols = ["MeterType_enc","TariffSlab_enc","Location_enc"]

        for _, grp in df.groupby("MeterId"):
            grp = grp.sort_values("TimeIdx").reset_index(drop=True)

            if mode == "train":
                idx_range = range(ENCODER_LEN, TRAIN_CUTOFF)
            else:
                idx_range = range(TRAIN_CUTOFF, len(grp))

            for i in idx_range:
                if i < ENCODER_LEN or i >= len(grp):
                    continue

                enc = grp.iloc[i-ENCODER_LEN:i]
                tgt = grp.iloc[i]

                # build continuous sequence — 5 features per timestep
                cont = []
                for _, row in enc.iterrows():
                    cont.append([
                        float(row["energy_norm"]),
                        float(row["PowerFactor_Avg_norm"]),
                        float(row["Month_norm"]),
                        float(row["Season_norm"]),
                        float(row["IsSummer"]),
                    ])

                self.samples.append({
                    "cats":    torch.tensor(
                        grp.iloc[0][cat_cols].values.astype(np.int64),
                        dtype=torch.long),
                    "cont":    torch.tensor(cont, dtype=torch.float32),
                    "target":  torch.tensor(
                        float(tgt["energy_norm"]), dtype=torch.float32),
                    "mean":    torch.tensor(
                        float(tgt["energy_mean"]), dtype=torch.float32),
                    "std":     torch.tensor(
                        float(tgt["energy_std"]),  dtype=torch.float32),
                    "actual":  float(tgt["ActiveEnergy_Total_Wh"]),
                    "anomaly": str(tgt["AnomalyLabel"]),
                    "time_idx":int(tgt["TimeIdx"]),
                })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate(batch):
    return {
        "cats":    torch.stack([b["cats"]   for b in batch]),
        "cont":    torch.stack([b["cont"]   for b in batch]),
        "target":  torch.stack([b["target"] for b in batch]),
        "mean":    torch.stack([b["mean"]   for b in batch]),
        "std":     torch.stack([b["std"]    for b in batch]),
        "actual":  [b["actual"]   for b in batch],
        "anomaly": [b["anomaly"]  for b in batch],
        "time_idx":[b["time_idx"] for b in batch],
    }


print("\nStep 2: Building datasets ...")
train_ds = MeterDataset(df, mode="train")
val_ds   = MeterDataset(df, mode="val")
print(f"  Train samples : {len(train_ds):,}")
print(f"  Val samples   : {len(val_ds):,}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, collate_fn=collate)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, collate_fn=collate)

# ── Step 3: Model ──────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden_size // n_heads
        self.scale    = math.sqrt(self.head_dim)
        self.q  = nn.Linear(hidden_size, hidden_size)
        self.k  = nn.Linear(hidden_size, hidden_size)
        self.v  = nn.Linear(hidden_size, hidden_size)
        self.o  = nn.Linear(hidden_size, hidden_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        B,T,H = x.shape
        nh,hd = self.n_heads, self.head_dim
        Q = self.q(x).view(B,T,nh,hd).transpose(1,2)
        K = self.k(x).view(B,T,nh,hd).transpose(1,2)
        V = self.v(x).view(B,T,nh,hd).transpose(1,2)
        A = torch.softmax((Q @ K.transpose(-2,-1))/self.scale, dim=-1)
        o = (self.dp(A) @ V).transpose(1,2).contiguous().view(B,T,H)
        return self.o(o)


class TFT(nn.Module):
    def __init__(self, cat_dims, n_cont, hidden_size, n_heads, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n+1, min(50,(n+1)//2+1))
            for n in cat_dims.values()
        ])
        emb_total = sum(min(50,(n+1)//2+1) for n in cat_dims.values())
        self.proj  = nn.Linear(emb_total + n_cont, hidden_size)
        self.lstm  = nn.LSTM(hidden_size, hidden_size,
                             num_layers=2, batch_first=True, dropout=dropout)
        self.attn  = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(hidden_size, hidden_size//2)
        self.fc2   = nn.Linear(hidden_size//2, len(QUANTILES))
        self.act   = nn.ReLU()

    def forward(self, cats, cont):
        embs = [e(cats[:,i]) for i,e in enumerate(self.embeddings)]
        embs = torch.cat(embs,-1).unsqueeze(1).expand(-1,cont.shape[1],-1)
        x    = self.proj(torch.cat([embs,cont],-1))
        lo,_ = self.lstm(x)
        lo   = self.norm1(lo)
        ao   = self.attn(lo)
        ao   = self.norm2(lo + ao)
        last = self.drop(ao[:,-1,:])
        return self.fc2(self.act(self.fc1(last)))


model = TFT(cat_dims, len(CONT_FEATURES),
            HIDDEN_SIZE, N_HEADS, DROPOUT).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nStep 3: Model built")
print(f"  Parameters     : {total_params:,}")
print(f"  n_cont         : {len(CONT_FEATURES)} (added IsSummer)")
print(f"  Cont features  : {CONT_FEATURES}")

# ── Step 4: Train ──────────────────────────────────────────────────────────────
def quantile_loss(preds, targets, quantiles):
    loss = 0.0
    for i, q in enumerate(quantiles):
        e     = targets - preds[:,i]
        loss += torch.mean(torch.max(q*e, (q-1)*e))
    return loss / len(quantiles)


optimizer  = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler  = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)
best_val   = float("inf")
patience_c = 0
history    = []

print(f"\nStep 4: Training ...")
print(f"  Max epochs : {MAX_EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  LR         : {LR}")
print()

for epoch in range(1, MAX_EPOCHS+1):
    model.train()
    tr_loss = []
    for b in train_loader:
        cats = b["cats"].to(DEVICE)
        cont = b["cont"].to(DEVICE)
        tgt  = b["target"].to(DEVICE)
        optimizer.zero_grad()
        loss = quantile_loss(model(cats,cont), tgt, QUANTILES)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss.append(loss.item())

    model.eval()
    vl_loss = []
    with torch.no_grad():
        for b in val_loader:
            cats = b["cats"].to(DEVICE)
            cont = b["cont"].to(DEVICE)
            tgt  = b["target"].to(DEVICE)
            vl_loss.append(
                quantile_loss(model(cats,cont), tgt, QUANTILES).item()
            )

    tr = np.mean(tr_loss)
    vl = np.mean(vl_loss)
    scheduler.step(vl)
    history.append({"epoch":epoch, "train":tr, "val":vl})

    saved = ""
    if vl < best_val:
        best_val   = vl
        patience_c = 0
        torch.save(model.state_dict(), MODEL_PATH)
        saved = "  ← saved"
    else:
        patience_c += 1
        if patience_c >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"  Epoch {epoch:>3}/{MAX_EPOCHS}  "
          f"train={tr:.4f}  val={vl:.4f}{saved}")

print(f"\n  Best val loss : {best_val:.4f}")

# ── Step 5: Evaluate ───────────────────────────────────────────────────────────
print("\nStep 5: Month by month evaluation ...")
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True,
                                  map_location=DEVICE))
model.eval()

all_rows = []
with torch.no_grad():
    for b in val_loader:
        cats  = b["cats"].to(DEVICE)
        cont  = b["cont"].to(DEVICE)
        preds = model(cats, cont).cpu().numpy()
        means = b["mean"].numpy()
        stds  = b["std"].numpy()

        p10 = np.clip(preds[:,0]*stds + means, 0, None)
        p50 = np.clip(preds[:,1]*stds + means, 0, None)
        p90 = np.clip(preds[:,2]*stds + means, 0, None)

        for i in range(len(p50)):
            actual  = float(b["actual"][i])
            dev_pct = (actual - p50[i]) / p50[i] * 100 \
                      if p50[i] > 0 else 0.0
            all_rows.append({
                "time_idx":  b["time_idx"][i],
                "actual":    actual,
                "p10":       float(p10[i]),
                "p50":       float(p50[i]),
                "p90":       float(p90[i]),
                "deviation": float(dev_pct),
                "below_p10": actual < float(p10[i]),
                "above_p90": actual > float(p90[i]),
                "anomaly":   b["anomaly"][i],
            })

results = pd.DataFrame(all_rows)

print("\n" + "=" * 60)
print("  Month by Month Predictions")
print("=" * 60)

for tidx in sorted(results["time_idx"].unique()):
    sub   = results[results["time_idx"] == tidx]
    acts  = sub["actual"].values
    p50v  = sub["p50"].values
    p10v  = sub["p10"].values
    p90v  = sub["p90"].values

    mae      = np.mean(np.abs(p50v - acts))
    mask     = acts > 100
    mape     = np.mean(np.abs((p50v[mask]-acts[mask])/acts[mask]))*100 \
               if mask.sum() > 0 else 0
    coverage = np.mean((acts>=p10v) & (acts<=p90v)) * 100

    catches = {}
    for atype in CONSUMPTION_ANOMALIES:
        sub_a = sub[sub["anomaly"]==atype]
        if len(sub_a) == 0:
            continue
        if atype in ["SuddenDrop","ZeroConsumption","NTL_Suspected"]:
            caught = sub_a["below_p10"].sum()
        else:
            caught = sub_a["above_p90"].sum()
        catches[atype] = f"{caught}/{len(sub_a)}"

    print(f"\n  TimeIdx {tidx} (month {tidx+1}):")
    print(f"    MAE      : {mae:,.0f} Wh")
    print(f"    MAPE     : {mape:.2f}%")
    print(f"    Coverage : {coverage:.1f}%")
    if catches:
        for atype, rate in catches.items():
            print(f"    {atype:<22} : {rate}")

# overall
acts_all = results["actual"].values
p50_all  = results["p50"].values
p10_all  = results["p10"].values
p90_all  = results["p90"].values

mae_all      = np.mean(np.abs(p50_all - acts_all))
mask_all     = acts_all > 100
mape_all     = np.mean(np.abs(
    (p50_all[mask_all]-acts_all[mask_all])/acts_all[mask_all]))*100
coverage_all = np.mean((acts_all>=p10_all)&(acts_all<=p90_all))*100

print(f"\n{'='*60}")
print(f"  Overall:")
print(f"  MAE      : {mae_all:,.2f} Wh")
print(f"  MAPE     : {mape_all:.2f}%")
print(f"  Coverage : {coverage_all:.1f}%")

for atype in CONSUMPTION_ANOMALIES:
    sub_a = results[results["anomaly"]==atype]
    if len(sub_a) == 0:
        continue
    if atype in ["SuddenDrop","ZeroConsumption","NTL_Suspected"]:
        caught = sub_a["below_p10"].sum()
    else:
        caught = sub_a["above_p90"].sum()
    rate = caught/len(sub_a)*100
    print(f"  {atype:<22} : {caught}/{len(sub_a)}  ({rate:.1f}%)")

# ── Step 6: Save ───────────────────────────────────────────────────────────────
print(f"\nStep 6: Saving ...")
config = {
    "model_path":   MODEL_PATH,
    "encoder_len":  ENCODER_LEN,
    "hidden_size":  HIDDEN_SIZE,
    "n_heads":      N_HEADS,
    "dropout":      DROPOUT,
    "quantiles":    QUANTILES,
    "cat_dims":     cat_dims,
    "n_cont":       len(CONT_FEATURES),
    "cont_features":CONT_FEATURES,
    "global_stats": global_stats,
    "meter_stats":  meter_stats.to_dict(),
    "mae":          float(mae_all),
    "mape":         float(mape_all),
    "coverage":     float(coverage_all),
    "history":      history,
}
with open(CONFIG_PATH, "wb") as f:
    pickle.dump(config, f)

print(f"  tft_model.pt → {MODEL_PATH}")
print(f"  config.pkl   → {CONFIG_PATH}")
print(f"\n{'='*60}")
print(f"  TFT Training Complete!")
print(f"  MAE  : {mae_all:,.0f} Wh")
print(f"  MAPE : {mape_all:.2f}%")
print(f"{'='*60}")