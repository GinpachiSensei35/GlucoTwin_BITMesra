# GlucoTwin — Digital Twin for Type 2 Diabetes

Submission for the Happiest Health **Digital Twin Challenge 2026**.

A proof-of-concept digital twin that fuses static EHR data with wearable streams (CGM, steps,
heart rate, sleep, meals) to predict whether glucose will exceed **180 mg/dL in the next 2 hours**,
explain the prediction, and run what-if simulations for a doctor-facing dashboard.

> **All data is synthetic** (Synthea EHR + our own wearable simulator). No real patient data is used.
> This is a research prototype, **not a medical device**.

_Work in progress — full documentation to follow._

## Setup

```bash
py -3.11 -m venv .venv            # or: python3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
```

## Run

```bash
make data        # build patients, wearables, features
make train       # train + evaluate, writes models/metrics.json
make api         # FastAPI on :8000
make dashboard   # Streamlit dashboard
```

## License

MIT — see [LICENSE](LICENSE).
