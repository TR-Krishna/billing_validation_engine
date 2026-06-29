# BVEngine — Billing Validation Engine

A billing validation engine for smart meter data. It combines two ML models
with a rules engine to detect billing anomalies — meter tampering, non-technical
losses, billing mismatches, and other unusual consumption patterns — before
bills go out.

Built during a Schneider Electric internship project, targeting real smart
meters in Mysuru, India.

---

## Architecture

```
Raw meter readings (OBIS codes)
        │
        ▼
  Feature Engine          → 8 engineered features per month, relative
                             to each meter's own history
        │
        ├──► Isolation Forest   → flags statistically unusual consumption
        │                          patterns (unsupervised)
        │
        ├──► TFT (Temporal      → forecasts expected consumption (P10/P50/P90),
        │     Fusion Transformer)  flags large deviations from forecast
        │
        └──► Rules Engine        → 9 deterministic checks: rate-sum errors,
                                    zero consumption, PF anomalies, NTL/bypass
                                    signatures, tariff boundary gaming, etc.
                  │
                  ▼
            Risk Scorer    → combines all 3 signals into one 0-100 score
                  │
                  ▼
       Explanation Engine  → plain-English explanation + recommendation
                  │
                  ▼
          FastAPI backend  → /api/process, /api/validate endpoints
                  │
                  ▼
         Dashboard (HTML)  → terminal-style UI, type a meter path,
                              see the full breakdown live
```

---

## Project structure

```
api/
  main.py                    FastAPI app — all endpoints live here
  core/
    feature_engine.py        Computes the 8 ML features from raw readings
    obis_resolver.py         Maps OBIS codes (DLMS/COSEM) to named fields
    validation_rules.py      9 deterministic billing rules
    risk_scorer.py           Combines IF + TFT + rules into one score
    explanation_engine.py    Generates plain-English explanation text
  models/
    request.py               Pydantic schema for incoming validation requests
    response.py               Pydantic schema for the API response

ml/
  isolation_forest/
    train.py                 Trains the Isolation Forest model
  tft/
    train.py                 Trains the TFT forecasting model

simulator/
  generate.py                 Generates synthetic training data
                              (10,000 meters x 36 months)

frontend/
  hes-dashboard/
    index.html                Terminal-style dashboard UI

config/
  utility-default.json        Thresholds, OBIS mappings, meter type config

fetch_weather.py               CDD-based seasonal temperature lookup
parse_and_validate.py          Parses real meter Excel/PDF files
generate_report.py             Generates PDF/Excel reports per validation
explain_meter.py                CLI display layer for a validation result
requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

If `pip` isn't recognized but `python` is, use:
```bash
python -m pip install -r requirements.txt --break-system-packages
```

---

## Running it — first time, full pipeline

Run these **in order**, from the project root:

```bash
# 1. Generate synthetic training data
python simulator/generate.py
#    → creates results/simulated_meters.csv
#    (10,000 simulated meters, 36 months each, ~500 with anomalies)

# 2. Train Isolation Forest
python ml/isolation_forest/train.py
#    → saves model.pkl, scaler.pkl, config.pkl into ml/isolation_forest/

# 3. Train TFT (takes longer — neural network training)
python ml/tft/train.py
#    → saves tft_model.pt, config.pkl into ml/tft/

# 4. Start the API
python -m uvicorn api.main:app --reload --port 8000
#    → wait for "All models ready!" in the terminal

# 5. Open the dashboard
#    Just open frontend/hes-dashboard/index.html in a browser
```

**Note:** `results/`, `*.pkl`, and `*.pt` files are gitignored — they're
either generated or too large to commit. You must run steps 1-3 yourself
after cloning before the API will have models to load.

---

## Using the dashboard

Once the API is running and the dashboard is open in a browser:

- **Real meter data**: type a file or folder path into the terminal box
  and press Enter. It auto-detects whether it's a single file or a folder
  of monthly Excel files, parses it, and runs full validation.
- **Synthetic test data**: use the "Generate" controls to create a fake
  meter with a chosen anomaly type (Spike, Drop, Zero, NTL, PF anomaly)
  and run it through the same pipeline instantly.

Every validation — real or synthetic — also auto-appends its data back
into `results/simulated_meters.csv`, so the dataset grows over time.
A daily backup of the dataset is taken automatically before the first
append each day (`results/backups/`), so changes can be rolled back via:

```
POST /api/dataset/restore
body: {"backup": "simulated_meters_YYYYMMDD.csv"}
```

---

## Key API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/process` | POST | Give it a single file or folder path — auto-detects and validates |
| `/api/validate` | POST | Validate a pre-built JSON payload directly |
| `/api/reports/{filename}` | GET | Download a generated PDF/Excel report |
| `/api/dataset/backups` | GET | List available dataset backups |
| `/api/dataset/restore` | POST | Roll back the dataset to a backup |
| `/api/health` | GET | Check if models are loaded |

s