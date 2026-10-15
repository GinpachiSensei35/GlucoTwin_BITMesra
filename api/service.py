"""Service layer shared by the FastAPI app and the Streamlit dashboard.

"Now" is a point inside the 14 simulated days (default DEMO_TIME); every call takes `at`.
All data synthetic. Research prototype, not a medical device.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build import FEATURES, THRESHOLD
from src.model.explain import shap_values
from src.model.reasons import plain_reasons
from src.twin.twin import DAYS, HORIZON_MIN, MEAL_LOOKBACK_MIN, what_if
from src.wearable.simulator import START

PATIENTS = Path("data/processed/patients.parquet")
WEARABLES = Path("data/processed/wearables.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
MODEL = Path("models/model.joblib")

DEMO_PATIENT = "be33d994-f841-40a4-9410-66555cf90fed"  # 51F T2D, 5.2 h sleep, 105 g lunch
DEMO_TIME = pd.Timestamp("2026-01-07 13:25")  # 10 min after her lunch
HISTORY_H = 22  # chart shows 22 h observed + 2 h twin projection = 24 h
DISCLAIMER = ("All data is synthetic (Synthea EHR + GlucoTwin simulator). The model partly learns the "
              "simulator; validation on real CGM data is future work. Research prototype, not a medical device.")


@lru_cache(maxsize=1)
def _patients() -> pd.DataFrame:
    return pd.read_parquet(PATIENTS).set_index("patient_id")


@lru_cache(maxsize=1)
def _wearables() -> pd.DataFrame:
    return pd.read_parquet(WEARABLES)


@lru_cache(maxsize=1)
def _features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH, columns=["patient_id", "timestamp"] + FEATURES)


@lru_cache(maxsize=1)
def _bundle() -> dict:
    return joblib.load(MODEL)


def time_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    return START + pd.Timedelta(hours=HISTORY_H), START + pd.Timedelta(days=DAYS, minutes=-(HORIZON_MIN + 5))


def _at(at) -> pd.Timestamp:
    t = DEMO_TIME if at is None else pd.Timestamp(at)
    t = t.floor("5min")
    lo, hi = time_range()
    if not lo <= t <= hi:
        raise ValueError(f"time must be between {lo} and {hi}")
    return t


def _check(pid: str) -> None:
    if pid not in _patients().index:
        raise KeyError(pid)


def _risk_level(p: float) -> str:
    thr = _bundle()["threshold"]
    return "high" if p >= thr else "moderate" if p >= 0.3 else "low"


def _ehr(pid: str) -> dict:
    p = _patients().loc[pid]
    out = {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k, v in p.items()}
    out["patient_id"] = pid
    return out


def list_patients(at=None) -> dict:
    t = _at(at)
    f = _features()
    rows = f[f["timestamp"] == t].set_index("patient_id")
    risk = _bundle()["model"].predict_proba(rows[FEATURES])[:, 1]
    pts = _patients().loc[rows.index]
    out = pd.DataFrame({
        "patient_id": rows.index, "age": pts["age"].values, "sex": pts["sex"].values,
        "hba1c": pts["hba1c"].values, "bmi": pts["bmi"].values.round(1),
        "diagnosis": np.where(pts["t2d"], "T2D", "Prediabetes"),
        "on_metformin": pts["on_metformin"].values,
        "glucose_now": rows["glucose"].values, "risk": risk.round(4),
    })
    out["risk_level"] = [_risk_level(r) for r in out["risk"]]
    out = out.sort_values("risk", ascending=False)
    return {"at": str(t), "demo_patient": DEMO_PATIENT, "patients": out.to_dict(orient="records")}


def predict(pid: str, at=None) -> dict:
    _check(pid)
    t = _at(at)
    f = _features()
    row = f[(f["patient_id"] == pid) & (f["timestamp"] == t)]
    X = row[FEATURES]
    p = float(_bundle()["model"].predict_proba(X)[:, 1][0])
    sv = shap_values(_bundle()["model"], X)[0]
    return {
        "patient_id": pid, "at": str(t), "risk": round(p, 4), "risk_level": _risk_level(p),
        "alert": p >= _bundle()["threshold"], "alert_threshold": _bundle()["threshold"],
        "glucose_now": float(row["glucose"].iloc[0]), "above_180_now": float(row["glucose"].iloc[0]) > THRESHOLD,
        "reasons": plain_reasons(row.iloc[0], sv, FEATURES, t, top=3),
        "question": "Will glucose exceed 180 mg/dL within the next 2 hours?",
    }


def patient_detail(pid: str, at=None) -> dict:
    _check(pid)
    t = _at(at)
    w = _wearables()
    w = w[w["patient_id"] == pid]
    lo, hi = t - pd.Timedelta(hours=HISTORY_H), t + pd.Timedelta(minutes=HORIZON_MIN)
    win = w[(w["timestamp"] >= lo) & (w["timestamp"] <= hi)]

    f = _features()
    fr = f[(f["patient_id"] == pid) & (f["timestamp"] >= lo) & (f["timestamp"] <= t)]
    risk = _bundle()["model"].predict_proba(fr[FEATURES])[:, 1]

    near = w[(w["carbs"] > 0) & (w["timestamp"] >= t - pd.Timedelta(minutes=MEAL_LOOKBACK_MIN))
             & (w["timestamp"] <= hi)]
    meal = ({"time": str(near["timestamp"].iloc[0]), "carbs": int(near["carbs"].iloc[0])}
            if len(near) else None)
    today = w[w["timestamp"] <= t].iloc[-1]

    return {
        "patient_id": pid, "at": str(t), "ehr": _ehr(pid), "now": predict(pid, t),
        "last_night_sleep_hours": float(today["sleep_hours_last_night"]),
        "last_night_sleep_eff": float(today["sleep_eff_last_night"]),
        "meal_in_window": meal,
        "series": {
            "timestamps": win["timestamp"].astype(str).tolist(),
            "glucose": win["glucose"].tolist(),
            "carbs": win["carbs"].tolist(),
            "steps": win["steps"].tolist(),
            "observed": (win["timestamp"] <= t).tolist(),  # False = twin projection (expected behaviour)
        },
        "risk_series": {"timestamps": fr["timestamp"].astype(str).tolist(), "risk": np.round(risk, 4).tolist()},
        "disclaimer": DISCLAIMER,
    }


def run_whatif(pid: str, at=None, meal_carbs=None, post_meal_walk_min=None, sleep_hours=None, mc_draws: int = 30) -> dict:
    _check(pid)
    return what_if(pid, _at(at), meal_carbs=meal_carbs, post_meal_walk_min=post_meal_walk_min,
                   sleep_hours=sleep_hours, mc_draws=mc_draws)
