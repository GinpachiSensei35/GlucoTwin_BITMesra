# Project: Digital Twin for Type 2 Diabetes — Glucose Spike Prediction

Submission for the Happiest Health **Digital Twin Challenge 2026** (Unstop).
Hard deadline: **20 Oct 2026, 7:00 PM IST**. Target finish: 20 Oct, 3:00 PM.
Prefer working, simple code over clever code. Do not add features not listed here without asking.

## Goal
A proof-of-concept digital twin of a Type 2 Diabetes patient that fuses:
1. **Static EHR data** (synthetic, from Synthea): demographics, diagnoses, HbA1c, BMI, medications, lipids.
2. **Dynamic wearable data** (synthetic, our own simulator): CGM every 5 min, steps, heart rate, sleep, meals.

It predicts whether glucose will exceed **180 mg/dL within the next 2 hours**, explains why (SHAP),
and supports **what-if simulation** (e.g. smaller meal, 20-min post-meal walk, better sleep → new risk curve).
A doctor-facing dashboard shows all of this.

## Stack
- Python 3.11, pandas, numpy, scikit-learn, LightGBM, SHAP, joblib
- Synthea (Java 17+) for EHR generation, CSV exporter
- FastAPI + uvicorn for the model/twin API
- Streamlit + Plotly for the doctor dashboard
- pytest for basic tests

## Repo layout
```
data/raw/synthea/        # Synthea CSV output (gitignored if large; keep a small sample committed)
data/processed/          # patients.parquet, wearables.parquet, features.parquet
src/ehr/                 # load + clean Synthea CSVs -> one row per patient
src/wearable/            # physiological simulator seeded from each patient's EHR
src/features/            # fuse streams, build features + labels
src/model/               # train, evaluate, explain, save
src/twin/                # what-if simulation: modify inputs, re-simulate, re-predict
api/                     # FastAPI app
dashboard/               # Streamlit app
docs/                    # architecture.pdf, presentation.pdf, images
models/                  # saved model + metrics.json
tests/
```

## Simulator rules (src/wearable)
- Baseline glucose from HbA1c: eAG (mg/dL) = 28.7 × HbA1c − 46.7
- Meals (3/day + optional snack, Indian carb-heavy patterns) raise glucose; peak ~45–75 min, scaled by carbs × insulin resistance (higher BMI / HbA1c → stronger, slower return)
- Steps/exercise lower glucose for 1–3 h after; HR rises with activity
- Poor sleep (< 6 h or low efficiency) → higher next-day insulin resistance
- Metformin users: modestly blunted rise
- Add realistic sensor noise; clip to 40–400 mg/dL
- Must be deterministic given a seed. Must be callable for a single patient with overridden inputs (used by the twin).

## Modelling rules
- Unit = (patient, timestamp). Label = max glucose in (t, t+120 min] > 180.
- Features: current glucose, lags (15/30/60 min), rate of change, rolling stats, carbs in last 2 h, steps in last 1–2 h, mean HR, last night's sleep, time of day, + static EHR fields.
- **Split train/test by patient** (GroupShuffleSplit). Never by row.
- Report: AUROC, AUPRC, recall at fixed precision, median lead time of alerts. Save to models/metrics.json.
- Compare against a naive baseline (current glucose > threshold).

## Honesty rules
- All data is synthetic (DPDP/HIPAA-safe). Say so everywhere.
- State the limitation that the model partly learns our simulator; validation on real CGM data is future work.
- This is a research prototype, not a medical device.

## Conventions
- Keep scripts runnable from repo root: `python -m src.<module>`
- One `Makefile` or `run.sh` with: data, train, api, dashboard.
- Pin versions in requirements.txt.
- Commit after each working phase.
