"""Digital twin what-if: edit a patient's behaviour around time t, re-simulate, re-predict.

    from src.twin.twin import what_if
    r = what_if(patient_id, "2026-01-07 13:40", post_meal_walk_min=20)

Overrides (all optional):
    meal_carbs          grams for "the meal": first meal in [t-30 min, t+120 min]; inserted at t if none
    post_meal_walk_min  brisk walk (~100 steps/min) starting 15 min after that meal
    sleep_hours         last night's sleep; wake time kept, bedtime moved

Two risk views per scenario:
    sim_spike_prob  twin physics: share of Monte Carlo re-simulations (noise up to t kept as
                    observed, noise after t re-drawn, behaviour fixed) whose max glucose in
                    (t, t+120] exceeds 180
    model_risk      LightGBM P(spike in next 2 h), evaluated every 5 min over the window as the
                    simulated data "arrives" (the model only sees the past, so planned future
                    meals/walks show up in its curve once they happen)

All data synthetic. Research prototype, not a medical device.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build import FEATURES, THRESHOLD, make_features
from src.wearable.simulator import BINS_PER_DAY, START, STEP_MIN, generate_inputs, patient_seed, simulate_glucose

PATIENTS = Path("data/processed/patients.parquet")
MODEL = Path("models/model.joblib")
DAYS = 14  # must match src.wearable.simulate so the baseline equals wearables.parquet
HORIZON_MIN = 120
MEAL_LOOKBACK_MIN = 30
WALK_DELAY_MIN = 15
WALK_STEPS_PER_BIN = 500  # ~100 steps/min
MC_DRAWS = 50


@lru_cache(maxsize=1)
def _patients() -> pd.DataFrame:
    return pd.read_parquet(PATIENTS)


@lru_cache(maxsize=1)
def _model() -> dict:
    return joblib.load(MODEL)


def _bin(t: pd.Timestamp) -> int:
    return int((t - START) / pd.Timedelta(minutes=STEP_MIN))


def apply_overrides(inputs: dict, at: pd.Timestamp, meal_carbs=None, post_meal_walk_min=None,
                    sleep_hours=None) -> tuple[dict, dict]:
    """Return edited copy of simulator inputs plus a description of what changed."""
    tl = inputs["timeline"].copy()
    sleep = inputs["sleep"].copy()
    ts = tl["timestamp"]

    near = (ts >= at - pd.Timedelta(minutes=MEAL_LOOKBACK_MIN)) & \
           (ts <= at + pd.Timedelta(minutes=HORIZON_MIN)) & (tl["carbs"] > 0)
    i = int(np.flatnonzero(near)[0]) if near.any() else _bin(at)
    info = {"meal_time": str(ts.iloc[i]), "meal_carbs_baseline": int(tl.at[i, "carbs"])}

    if meal_carbs is not None:
        tl.at[i, "carbs"] = float(meal_carbs)
    info["meal_carbs_modified"] = int(tl.at[i, "carbs"])

    if post_meal_walk_min:
        start = i + WALK_DELAY_MIN // STEP_MIN
        n = int(np.ceil(post_meal_walk_min / STEP_MIN))
        rows = tl.index[start:start + n]
        tl.loc[rows, "steps"] = np.maximum(tl.loc[rows, "steps"], WALK_STEPS_PER_BIN)
        info["walk"] = {"start": str(ts.iloc[start]), "minutes": n * STEP_MIN}

    if sleep_hours is not None:
        night = _bin(at) // BINS_PER_DAY - 1  # the night ending on the morning of t's day
        j = sleep.index[sleep["night"] == night][0]
        eff = sleep.at[j, "sleep_efficiency"]
        wake = sleep.at[j, "bed_time"] + pd.Timedelta(hours=sleep.at[j, "sleep_hours"] / eff)
        info["sleep_hours_baseline"] = round(float(sleep.at[j, "sleep_hours"]), 2)
        sleep.at[j, "sleep_hours"] = float(sleep_hours)
        sleep.at[j, "bed_time"] = wake - pd.Timedelta(hours=sleep_hours / eff)
        info["sleep_hours_modified"] = float(sleep_hours)

    return {"timeline": tl, "sleep": sleep}, info


def _run_scenario(patient: pd.Series, inputs: dict, seed: int, at: pd.Timestamp, mc_draws: int) -> dict:
    end = at + pd.Timedelta(minutes=HORIZON_MIN)
    pid = patient["patient_id"]

    sim = simulate_glucose(patient, inputs, seed)
    sim.insert(0, "patient_id", pid)
    feats = make_features(sim, _patients()[_patients()["patient_id"] == pid], drop_incomplete=False)
    win = (feats["timestamp"] >= at) & (feats["timestamp"] <= end)
    risk = _model()["model"].predict_proba(feats.loc[win, FEATURES])[:, 1]

    future = (sim["timestamp"] > at) & (sim["timestamp"] <= end)
    spikes = 0
    for k in range(mc_draws):  # past noise kept as observed; only the future is re-drawn
        g = simulate_glucose(patient, inputs, seed, noise_seed=k, noise_from=_bin(at) + 1)["glucose"].to_numpy()
        spikes += g[future.to_numpy()].max() > THRESHOLD

    w = sim.loc[win.to_numpy()]
    return {
        "timestamps": [str(t) for t in w["timestamp"]],
        "glucose": w["glucose"].round(1).tolist(),
        "steps": w["steps"].tolist(),
        "model_risk": np.round(risk, 4).tolist(),
        "model_risk_now": round(float(risk[0]), 4),
        "model_risk_peak": round(float(risk.max()), 4),
        "peak_glucose": float(sim.loc[future, "glucose"].max()),
        "minutes_above_180": int((sim.loc[future, "glucose"] > THRESHOLD).sum() * STEP_MIN),
        "sim_spike_prob": round(spikes / mc_draws, 3) if mc_draws else None,
    }


def what_if(patient_id: str, at, meal_carbs=None, post_meal_walk_min=None, sleep_hours=None,
            mc_draws: int = MC_DRAWS) -> dict:
    """Baseline vs modified 2-hour glucose + risk for one patient at time `at`."""
    pts = _patients()
    if patient_id not in set(pts["patient_id"]):
        raise KeyError(f"unknown patient_id {patient_id}")
    patient = pts.set_index("patient_id").loc[patient_id].copy()
    patient["patient_id"] = patient_id

    at = pd.Timestamp(at).floor(f"{STEP_MIN}min")
    lo = START + pd.Timedelta(hours=2)
    hi = START + pd.Timedelta(days=DAYS) - pd.Timedelta(minutes=HORIZON_MIN + STEP_MIN)
    if not lo <= at <= hi:
        raise ValueError(f"time must be between {lo} and {hi}")

    seed = patient_seed(patient_id)
    base_inputs = generate_inputs(patient, DAYS, seed)
    mod_inputs, changes = apply_overrides(base_inputs, at, meal_carbs, post_meal_walk_min, sleep_hours)

    return {
        "patient_id": patient_id,
        "time": str(at),
        "horizon_min": HORIZON_MIN,
        "overrides": {"meal_carbs": meal_carbs, "post_meal_walk_min": post_meal_walk_min,
                      "sleep_hours": sleep_hours},
        "changes": changes,
        "baseline": _run_scenario(patient, base_inputs, seed, at, mc_draws),
        "modified": _run_scenario(patient, mod_inputs, seed, at, mc_draws),
        "note": "Synthetic data; research prototype, not a medical device.",
    }
