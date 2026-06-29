"""
Billing Validation Engine — FastAPI Application
================================================
Run with:
  python -m uvicorn api.main:app --reload --port 8000

Endpoints:
  GET  /api/health
  POST /api/validate        ← HES sends meter data, gets validation result
  POST /api/parse           ← HES uploads PDF/Excel, gets parsed readings
  POST /api/parse-folder    ← HES sends a folder path, gets parsed readings
  GET  /api/reports/{file}  ← download generated report
"""

import sys
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import re

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
IF_MODEL_PATH   = BASE_DIR / "ml/isolation_forest/model.pkl"
IF_SCALER_PATH  = BASE_DIR / "ml/isolation_forest/scaler.pkl"
IF_CONFIG_PATH  = BASE_DIR / "ml/isolation_forest/config.pkl"
TFT_MODEL_PATH  = BASE_DIR / "ml/tft/tft_model.pt"
TFT_CONFIG_PATH = BASE_DIR / "ml/tft/config.pkl"
CONFIG_DIR      = BASE_DIR / "config"
REPORTS_DIR     = BASE_DIR / "results"
REPORTS_DIR.mkdir(exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models    = {}
QUANTILES = [0.1, 0.5, 0.9]

OBIS_FIELDS = {
    "1.0.1.8.0.255":  "totalActiveEnergy",
    "1.0.1.8.1.255":  "rate1Energy",
    "1.0.1.8.2.255":  "rate2Energy",
    "1.0.1.8.3.255":  "rate3Energy",
    "1.0.1.8.4.255":  "rate4Energy",
    "1.0.1.6.0.255":  "peakDemand",
    "1.0.9.8.0.255":  "apparentEnergy",
    "1.0.13.0.0.255": "powerFactor",
    "1.0.84.0.0.255": "powerFactorImport",
    "1.0.5.8.0.255":  "reactiveQI",
    "1.0.6.8.0.255":  "reactiveQII",
    "1.0.7.8.0.255":  "reactiveQIII",
    "1.0.8.8.0.255":  "reactiveQIV",
    "0.0.0.1.2.255":  "billingPeriodEnd",
}


# ── TFT Architecture ──────────────────────────────────────────────────────────
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
        B, T, H = x.shape
        nh, hd  = self.n_heads, self.head_dim
        Q = self.q(x).view(B,T,nh,hd).transpose(1,2)
        K = self.k(x).view(B,T,nh,hd).transpose(1,2)
        V = self.v(x).view(B,T,nh,hd).transpose(1,2)
        A = torch.softmax((Q @ K.transpose(-2,-1)) / self.scale, dim=-1)
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
        self.fc1   = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2   = nn.Linear(hidden_size // 2, len(QUANTILES))
        self.act   = nn.ReLU()

    def forward(self, cats, cont):
        embs = [e(cats[:,i]) for i,e in enumerate(self.embeddings)]
        embs = torch.cat(embs,-1).unsqueeze(1).expand(-1, cont.shape[1], -1)
        x    = self.proj(torch.cat([embs, cont], -1))
        lo,_ = self.lstm(x)
        lo   = self.norm1(lo)
        ao   = self.attn(lo)
        ao   = self.norm2(lo + ao)
        last = self.drop(ao[:,-1,:])
        return self.fc2(self.act(self.fc1(last)))


# ── Startup ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("  Billing Validation Engine")
    print(f"  Device : {DEVICE}")
    print("=" * 50)

    print("\nLoading Isolation Forest ...")
    with open(IF_MODEL_PATH,  "rb") as f: models["if_model"]  = pickle.load(f)
    with open(IF_SCALER_PATH, "rb") as f: models["if_scaler"] = pickle.load(f)
    with open(IF_CONFIG_PATH, "rb") as f: models["if_config"] = pickle.load(f)

    print("\nLoading TFT ...")
    with open(TFT_CONFIG_PATH, "rb") as f:
        tft_cfg = pickle.load(f)
    models["tft_config"] = tft_cfg
    tft_model = TFT(
        cat_dims=tft_cfg["cat_dims"], n_cont=tft_cfg["n_cont"],
        hidden_size=tft_cfg["hidden_size"], n_heads=tft_cfg["n_heads"],
        dropout=tft_cfg["dropout"],
    ).to(DEVICE)
    tft_model.load_state_dict(
        torch.load(TFT_MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    tft_model.eval()
    models["tft_model"] = tft_model

    print("\nLoading OBIS configs ...")
    models["obis_configs"] = {}
    for cfg_file in CONFIG_DIR.glob("*.json"):
        with open(cfg_file) as f:
            cfg = json.load(f)
            uid = cfg.get("utilityId", cfg_file.stem)
            models["obis_configs"][uid] = cfg
            print(f"  Loaded: {uid}")

    print("\n  All models ready!")
    print("=" * 50)
    yield
    models.clear()


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Billing Validation Engine",
              version="2.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service":"Billing Validation Engine","version":"2.0.0",
            "status":"running","device":str(DEVICE)}

@app.get("/api/health")
def health():
    return {"status":"healthy","if_loaded":"if_model" in models,
            "tft_loaded":"tft_model" in models,
            "device":str(DEVICE),
            "timestamp":datetime.utcnow().isoformat()+"Z"}


# ── Report download ────────────────────────────────────────────────────────────
@app.get("/api/reports/{filename}")
def download_report(filename: str):
    safe = Path(filename).name
    path = REPORTS_DIR / safe
    if not path.exists():
        raise HTTPException(404, detail=f"Report not found: {filename}")
    media = ("application/vnd.openxmlformats-officedocument"
             ".spreadsheetml.sheet"
             if safe.endswith(".xlsx") else "application/pdf")
    return FileResponse(path=str(path), media_type=media, filename=safe)


# ── Dataset backups — rollback safety net ────────────────────────────────────
@app.get("/api/dataset/backups")
def list_dataset_backups():
    """List available daily backups of simulated_meters.csv, newest first."""
    backup_dir = REPORTS_DIR / "backups"
    if not backup_dir.exists():
        return {"backups": []}
    files = sorted(backup_dir.glob("simulated_meters_*.csv"), reverse=True)
    return {"backups": [f.name for f in files]}


@app.post("/api/dataset/restore")
def restore_dataset_backup(request: dict):
    """
    Restore simulated_meters.csv from a named backup file,
    undoing any appends made since that backup was taken.
    Pass {"backup": "simulated_meters_20260629.csv"}.
    """
    import shutil
    backup_name = request.get("backup", "").strip()
    if not backup_name:
        raise HTTPException(400, detail="backup filename is required")

    backup_path = REPORTS_DIR / "backups" / Path(backup_name).name
    if not backup_path.exists():
        raise HTTPException(404, detail=f"Backup not found: {backup_name}")

    dataset_path = REPORTS_DIR / "simulated_meters.csv"
    shutil.copy2(backup_path, dataset_path)
    return {"restored": backup_name, "datasetPath": str(dataset_path)}


# ── Parse PDF/Excel ────────────────────────────────────────────────────────────
@app.post("/api/parse")
async def parse_file(file: UploadFile = File(...)):
    """
    Accept PDF or Excel upload from HES dashboard.
    Parse EcoSEnter billing profile format.
    Return structured meter data ready for /api/validate.
    """
    filename = file.filename or "upload"
    ext      = Path(filename).suffix.lower()

    if ext not in (".pdf", ".xlsx", ".xls", ".csv"):
        raise HTTPException(400, detail=f"Unsupported file type: {ext}")

    # save to temp file
    content = await file.read()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from parse_and_validate import (
            parse_pdf, parse_excel, parse_csv,
            build_payload, extract_meter_id
        )

        if ext == ".pdf":
            meter_id, periods = parse_pdf(tmp_path)
        elif ext in (".xlsx", ".xls"):
            meter_id, periods = parse_excel(tmp_path)
        else:
            meter_id, periods = parse_csv(tmp_path)

        if not periods:
            raise HTTPException(422,
                detail="No billing periods found. Check file format.")

        # override meter_id from filename if possible
        fn_id = extract_meter_id("", filename)
        if fn_id and fn_id != "UNKNOWN":
            meter_id = fn_id

        payload = build_payload(meter_id, periods)

        # return payload + preview info
        energies = [p.get("1.0.1.8.0.255") for p in periods
                    if p.get("1.0.1.8.0.255") > 0]
        avg_e    = sum(energies)/len(energies) if energies else 0

        return {
            "meterId":       payload["meterId"],
            "meterType":     payload["meterType"],
            "tariffSlab":    payload["tariffSlab"],
            "periodsFound":  len(periods),
            "billingPeriod": payload["billingPeriod"],
            "historyMonths": payload["historyMonths"],
            "avgEnergyWh":   round(avg_e, 0),
            "periods":       [p.period for p in sorted(
                                periods, key=lambda x: x.period or "")],
            "payload":       payload,   # full payload for /api/validate
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Parse error: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Parse folder ───────────────────────────────────────────────────────────────
@app.post("/api/parse-folder")
async def parse_folder_endpoint(request: dict):
    """
    Accept a folder path string from HES dashboard.
    Parses all Excel/PDF files in the folder.
    Returns structured meter data ready for /api/validate.
    """
    folder   = request.get("folder", "").strip()
    location = request.get("location", "Chennai")

    if not folder:
        raise HTTPException(400, detail="folder path is required")

    folder_path = Path(folder)
    if not folder_path.exists():
        raise HTTPException(404, detail=f"Folder not found: {folder}")
    if not folder_path.is_dir():
        raise HTTPException(400, detail=f"Not a directory: {folder}")

    try:
        from parse_and_validate import (
            parse_folder, build_payload,
            is_cumulative, subtract_cumulative
        )

        meter_id, periods = parse_folder(str(folder_path))

        if not periods:
            raise HTTPException(422,
                detail="No billing periods found in folder.")

        if is_cumulative(periods):
            periods = subtract_cumulative(periods)

        payload = build_payload(meter_id, periods, location=location)

        energies = [p.get("1.0.1.8.0.255") for p in periods
                    if p.get("1.0.1.8.0.255", 0) > 0]
        avg_e = sum(energies)/len(energies) if energies else 0

        return {
            "meterId":       payload["meterId"],
            "meterType":     payload["meterType"],
            "tariffSlab":    payload["tariffSlab"],
            "periodsFound":  len(periods),
            "billingPeriod": payload["billingPeriod"],
            "historyMonths": payload["historyMonths"],
            "avgEnergyWh":   round(avg_e, 0),
            "periods":       sorted([p.period for p in periods if p.period]),
            "payload":       payload,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Parse folder error: {str(e)}")


# ── Unified process — auto-detects file vs folder from a single path ──────────
@app.post("/api/process")
async def process_path(request: dict):
    """
    Accept a single path string from the dashboard terminal box.
    Auto-detects whether it's a file or a folder, parses accordingly,
    then immediately runs /api/validate on the result.
    Returns the full validate result in one call.
    """
    raw_path = request.get("path", "").strip().strip('"')
    location = request.get("location", "Chennai")

    if not raw_path:
        raise HTTPException(400, detail="path is required")

    p = Path(raw_path)
    if not p.exists():
        raise HTTPException(404, detail=f"Path not found: {raw_path}")

    from parse_and_validate import (
        parse_pdf, parse_excel, parse_csv, parse_folder,
        build_payload, is_cumulative, subtract_cumulative
    )

    try:
        if p.is_dir():
            meter_id, periods = parse_folder(str(p))
        else:
            ext = p.suffix.lower()
            if ext == ".pdf":
                meter_id, periods = parse_pdf(str(p))
            elif ext in (".xlsx", ".xls"):
                meter_id, periods = parse_excel(str(p))
            elif ext == ".csv":
                meter_id, periods = parse_csv(str(p))
            else:
                raise HTTPException(400, detail=f"Unsupported file type: {ext}")

        if not periods:
            raise HTTPException(422, detail="No billing periods found.")

        if is_cumulative(periods):
            periods = subtract_cumulative(periods)

        if len(periods) < 2:
            raise HTTPException(
                422,
                detail="Need at least 2 billing periods for validation."
            )

        payload = build_payload(meter_id, periods, location=location)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Parse error: {str(e)}")

    # call the same logic as /api/validate directly (avoid HTTP round trip)
    return await validate(payload)


@app.post("/api/validate")
async def validate(request: dict):
    from api.models.request          import ValidationRequest
    from api.core.obis_resolver      import build_resolver
    from api.core.feature_engine     import (
        compute_features, get_if_feature_vector,
        get_tft_sequence, get_model_confidence,
    )
    from api.core.validation_rules   import run_all_rules
    from api.core.risk_scorer        import compute_risk_score
    from api.core.explanation_engine import generate_explanation
    from generate_report             import generate_excel, generate_pdf
    from fetch_weather               import fetch_monthly_temperature

    try:
        req = ValidationRequest(**request)
    except Exception as e:
        raise HTTPException(422, detail=str(e))

    try:
        resolver = build_resolver(models["obis_configs"], req.utilityId)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    thresholds = resolver.thresholds
    current    = resolver.extract_current(req.currentReadings)
    active_wh  = current["activeEnergy"]
    r1,r2,r3,r4= current["rate1"],current["rate2"],current["rate3"],current["rate4"]
    peak_w     = current["peakDemand"]
    app_vah    = current["apparentEnergy"]
    pf         = current["powerFactor"]
    billed_wh  = active_wh

    e_hist     = resolver.extract_history_energy(req.history)
    pf_hist    = resolver.extract_history_pf(req.history)
    peak_hist  = resolver.extract_history_peak(req.history)
    history_months   = len(e_hist)
    model_confidence = get_model_confidence(history_months)

    # fetch weather for this billing month — used as the IsSummer feature
    temperature = fetch_monthly_temperature(req.location, req.billingPeriod)

    features   = compute_features(
        active_wh=active_wh, billed_wh=billed_wh,
        r1=r1, r2=r2, r3=r3, r4=r4,
        power_factor=pf, peak_demand_w=peak_w, apparent_vah=app_vah,
        e_history=e_hist, pf_history=pf_hist, peak_history=peak_hist,
        billing_period=req.billingPeriod,
        temperature_c=temperature,
    )
    hist_avg   = float(features["Hist_AvgEnergy_Wh"])
    hist_pf    = float(features["Hist_AvgPowerFactor"])

    # IF
    X_scaled  = models["if_scaler"].transform(
        np.array([get_if_feature_vector(features)], dtype=np.float32))
    if_pred   = models["if_model"].predict(X_scaled)[0]
    if_score  = float(models["if_model"].score_samples(X_scaled)[0])
    if_anom   = bool(if_pred == -1)
    if model_confidence == "INSUFFICIENT":
        if_anom = False; if_score = -0.40

    # TFT
    tft_cfg   = models["tft_config"]
    gs        = tft_cfg["global_stats"]
    mt_enc    = int({"Residential":0,"Commercial":1,"Industrial":2}.get(req.meterType,0))
    ts_enc    = int({"LT-1":0,"LT-2":1,"LT-3":2}.get(req.tariffSlab,0))
    loc_enc   = int({"Chennai":0,"Mumbai":1,"Delhi":2,"Bangalore":3,"Hyderabad":4}.get(req.location,0))
    e_mean    = float(hist_avg) if hist_avg > 0 else 1.0
    e_std     = float(max(features["Hist_StdDev_Wh"], 1.0))

    cont_seq  = get_tft_sequence(
        e_history=e_hist, pf_history=pf_hist,
        billing_period=req.billingPeriod,
        energy_mean=e_mean, energy_std=e_std,
        global_stats=gs, encoder_len=tft_cfg.get("encoder_len",12),
        temperature_c=temperature,
    )
    cats_t    = torch.tensor([[mt_enc,ts_enc,loc_enc]],dtype=torch.long).to(DEVICE)
    cont_t    = torch.tensor([cont_seq],dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = models["tft_model"](cats_t,cont_t).cpu().numpy()[0]

    p10 = float(max(preds[0]*e_std+e_mean, 0))
    p50 = float(max(preds[1]*e_std+e_mean, 0))
    p90 = float(max(preds[2]*e_std+e_mean, 0))
    tft_deviation    = float((active_wh-p50)/p50*100 if p50>0 else 0.0)
    tft_sudden_drop  = bool(active_wh < p10)
    tft_sudden_spike = bool(active_wh > p90)
    if model_confidence == "INSUFFICIENT":
        tft_deviation=0.0; tft_sudden_drop=tft_sudden_spike=False

    rule_flags = run_all_rules(
        active_wh=active_wh, billed_wh=billed_wh,
        r1=r1, r2=r2, r3=r3, r4=r4,
        power_factor=pf, peak_demand=peak_w,
        features=features, thresholds=thresholds, e_history=e_hist,
    )
    risk = compute_risk_score(
        if_is_anomaly=if_anom, if_score=if_score,
        tft_deviation=tft_deviation,
        tft_sudden_drop=tft_sudden_drop, tft_sudden_spike=tft_sudden_spike,
        rule_flags=rule_flags,
        active_wh=active_wh, predicted_wh=p50, hist_avg_wh=hist_avg,
        thresholds=thresholds,
    )
    explanation, recommendation = generate_explanation(
        meter_id=req.meterId, billing_period=req.billingPeriod,
        meter_type=req.meterType, anomaly_type=risk["anomalyType"],
        severity=risk["severity"], actual_wh=active_wh, predicted_wh=p50,
        hist_avg_wh=hist_avg, deviation_pct=tft_deviation,
        power_factor=pf, hist_avg_pf=hist_pf,
        risk_score=risk["riskScore"], revenue_at_risk=risk["revenueAtRiskINR"],
        rule_flags=rule_flags, history_months=history_months,
        model_confidence=model_confidence,
    )

    result = {
        "meterId":         req.meterId,
        "billingPeriod":   req.billingPeriod,
        "processedAt":     datetime.utcnow().isoformat()+"Z",
        "riskScore":       risk["riskScore"],
        "severity":        risk["severity"],
        "isAnomaly":       risk["isAnomaly"],
        "anomalyType":     risk["anomalyType"],
        "actualWh":        round(active_wh,2),
        "predictedWh":     round(p50,2),
        "deviationPct":    round(tft_deviation,2),
        "historicalAvgWh": round(hist_avg,2),
        "billedAmountWh":  round(billed_wh,2),
        "billAmountINR":   round(billed_wh*0.008,2),
        "revenueAtRiskINR":risk["revenueAtRiskINR"],
        "ntlSuspected":    risk["ntlSuspected"],
        "preInvoiceFlag":  risk["preInvoiceFlag"],
        "historyMonths":   history_months,
        "modelConfidence": model_confidence,
        "modelOutputs": {
            "ifIsAnomaly":    if_anom,
            "ifScore":        round(if_score,4),
            "tftPredictedWh": round(p50,2),
            "tftP10":         round(p10,2),
            "tftP90":         round(p90,2),
            "tftDeviation":   round(tft_deviation,2),
            "tftSuddenDrop":  tft_sudden_drop,
            "tftSuddenSpike": tft_sudden_spike,
        },
        "ruleFlags":       rule_flags,
        "scoreComponents": risk["scoreComponents"],
        "explanation":     explanation,
        "recommendation":  recommendation,
        "features":        features,
        "weather": {
            "temperatureC": temperature,
            "isSummer":     features.get("IsSummer"),
        },
    }

    # generate reports automatically
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id  = re.sub(r'[^A-Za-z0-9_-]', '_', req.meterId)
    safe_per = req.billingPeriod.replace("/","_")
    base     = f"{safe_id}_{safe_per}_{ts}"
    xl_name  = f"{base}.xlsx"
    pdf_name = f"{base}.pdf"

    try:
        generate_excel(result, str(REPORTS_DIR / xl_name))
        generate_pdf(result,   str(REPORTS_DIR / pdf_name))
        result["reports"] = {
            "excel": f"/api/reports/{xl_name}",
            "pdf":   f"/api/reports/{pdf_name}",
        }
    except Exception as e:
        result["reports"] = {"excel":None,"pdf":None,"error":str(e)}

    # auto-append this validation's data into the training dataset —
    # only the current/tested month gets the detected anomalyType,
    # history months are always labeled None
    try:
        append_validation_to_dataset(
            req=req, resolver=resolver,
            active_wh=active_wh, r1=r1, r2=r2, r3=r3, r4=r4,
            peak_w=peak_w, app_vah=app_vah, pf=pf,
            e_hist=e_hist, pf_hist=pf_hist, peak_hist=peak_hist,
            temperature=temperature, features=features,
            anomaly_type=risk["anomalyType"],
        )
    except Exception as e:
        print(f"  WARNING: could not append to training dataset: {e}")

    return result


def append_validation_to_dataset(
    req, resolver, active_wh, r1, r2, r3, r4, peak_w, app_vah, pf,
    e_hist, pf_hist, peak_hist, temperature, features, anomaly_type,
):
    """
    Appends every month seen in this validation request (history +
    current) into results/simulated_meters.csv, in the same column
    schema used by the synthetic generator. Only the current month
    gets labeled with whatever anomalyType the risk scorer detected
    (or "None" if no anomaly) — history months are always "None"
    since we don't have independent ground truth for them.
    """
    import pandas as pd
    import shutil
    from api.core.feature_engine import get_season

    DATASET_PATH = REPORTS_DIR / "simulated_meters.csv"
    BACKUP_DIR   = REPORTS_DIR / "backups"
    TARIFF_MAP   = {"LT-1": "LT-1", "LT-2": "LT-2", "LT-3": "LT-3"}

    # one backup per day, taken right before the FIRST append of that
    # day — gives you a "start of day" snapshot to roll back to,
    # without making a new backup file on every single click
    if DATASET_PATH.exists():
        BACKUP_DIR.mkdir(exist_ok=True)
        today_tag    = datetime.now().strftime("%Y%m%d")
        backup_path  = BACKUP_DIR / f"simulated_meters_{today_tag}.csv"
        if not backup_path.exists():
            shutil.copy2(DATASET_PATH, backup_path)
            print(f"  Backup created: {backup_path}")

    rows = []

    # history months — no anomaly label, recompute features incrementally
    running_e, running_pf, running_peak = [], [], []
    for i, month_obj in enumerate(req.history):
        h_resolved = resolver.extract_history_energy([month_obj])
        h_e   = h_resolved[0] if h_resolved else 0.0
        h_pf  = resolver.extract_history_pf([month_obj])
        h_pf  = h_pf[0] if h_pf else 0.95
        h_pk  = resolver.extract_history_peak([month_obj])
        h_pk  = h_pk[0] if h_pk else 0.0

        bd = datetime.strptime(month_obj.billingPeriod, "%Y-%m")

        rows.append({
            "MeterId":                  req.meterId,
            "MeterType":                req.meterType,
            "TariffSlab":               req.tariffSlab,
            "Location":                 req.location,
            "BillingPeriodEnd":         bd.strftime("%Y-%m-%d"),
            "Month":                    bd.month,
            "Season":                   get_season(bd.month),
            "IsSummer":                 0.0,
            "ActiveEnergy_Total_Wh":    h_e,
            "ActiveEnergy_Rate1_Wh":    0.0,
            "ActiveEnergy_Rate2_Wh":    0.0,
            "ActiveEnergy_Rate3_Wh":    0.0,
            "ActiveEnergy_Rate4_Wh":    0.0,
            "PeakDemand_Total_W":       h_pk,
            "ApparentEnergy_Total_VAh": round(h_e / max(h_pf, 0.01), 2),
            "PowerFactor_Avg":          round(h_pf, 4),
            "PowerFactor_Import":       1.0,
            "ReactiveEnergy_QI_VArh":   0.0,
            "ReactiveEnergy_QII_VArh":  0.0,
            "ReactiveEnergy_QIII_VArh": 0.0,
            "ReactiveEnergy_QIV_VArh":  0.0,
            "BilledAmount_Wh":          h_e,
            "BillingReadType":          "TESTED",
            "BillAmountINR":            0.0,
            "EnergyVsAvgRatio":         1.0,
            "VsSeasonalAvgRatio":       1.0,
            "DeviationInStdDevs":       0.0,
            "MoMChangePct":             0.0,
            "TrendSlope":               0.0,
            "PowerFactor_Deviation":    0.0,
            "PeakDemandToAvgRatio":     1.0,
            "AnomalyLabel":             "None",
            "IsAnomalyMonth":           False,
        })

    # current/tested month — gets the real detected anomalyType
    bd_cur = datetime.strptime(req.billingPeriod, "%Y-%m")
    rows.append({
        "MeterId":                  req.meterId,
        "MeterType":                req.meterType,
        "TariffSlab":               req.tariffSlab,
        "Location":                 req.location,
        "BillingPeriodEnd":         bd_cur.strftime("%Y-%m-%d"),
        "Month":                    bd_cur.month,
        "Season":                   get_season(bd_cur.month),
        "IsSummer":                 features.get("IsSummer", 0.0),
        "ActiveEnergy_Total_Wh":    active_wh,
        "ActiveEnergy_Rate1_Wh":    r1,
        "ActiveEnergy_Rate2_Wh":    r2,
        "ActiveEnergy_Rate3_Wh":    r3,
        "ActiveEnergy_Rate4_Wh":    r4,
        "PeakDemand_Total_W":       peak_w,
        "ApparentEnergy_Total_VAh": app_vah,
        "PowerFactor_Avg":          round(pf, 4),
        "PowerFactor_Import":       1.0,
        "ReactiveEnergy_QI_VArh":   0.0,
        "ReactiveEnergy_QII_VArh":  0.0,
        "ReactiveEnergy_QIII_VArh": 0.0,
        "ReactiveEnergy_QIV_VArh":  0.0,
        "BilledAmount_Wh":          active_wh,
        "BillingReadType":          "TESTED",
        "BillAmountINR":            0.0,
        "EnergyVsAvgRatio":         features.get("EnergyVsAvgRatio", 1.0),
        "VsSeasonalAvgRatio":       features.get("VsSeasonalAvgRatio", 1.0),
        "DeviationInStdDevs":       features.get("DeviationInStdDevs", 0.0),
        "MoMChangePct":             features.get("MoMChangePct", 0.0),
        "TrendSlope":               features.get("TrendSlope", 0.0),
        "PowerFactor_Deviation":    features.get("PowerFactor_Deviation", 0.0),
        "PeakDemandToAvgRatio":     features.get("PeakDemandToAvgRatio", 1.0),
        "AnomalyLabel":             anomaly_type if anomaly_type and anomaly_type != "None" else "None",
        "IsAnomalyMonth":           bool(anomaly_type and anomaly_type != "None"),
    })

    new_df = pd.DataFrame(rows)

    if DATASET_PATH.exists():
        existing_df = pd.read_csv(DATASET_PATH)
        for col in existing_df.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        new_df = new_df[existing_df.columns]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(DATASET_PATH, index=False)
    print(f"  Appended {len(new_df)} rows to {DATASET_PATH} "
          f"(current month labeled: {rows[-1]['AnomalyLabel']})")