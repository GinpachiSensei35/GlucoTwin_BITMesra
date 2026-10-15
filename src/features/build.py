"""Fuse EHR + wearable streams into model features and labels.

Run: python -m src.features.build
Input:  data/processed/patients.parquet, data/processed/wearables.parquet
Output: data/processed/features.parquet

Unit = (patient, timestamp). Label = max glucose in (t, t+120 min] > 180 mg/dL.
All data is synthetic.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PATIENTS = Path("data/processed/patients.parquet")
WEARABLES = Path("data/processed/wearables.parquet")
OUT = Path("data/processed/features.parquet")

THRESHOLD = 180.0
BIN_MIN = 5
HORIZON = 120 // BIN_MIN  # 24 bins
HISTORY = 120 // BIN_MIN  # need 2 h of history for lags/rolling windows
MEAL_CAP_MIN = 360

STATIC = ["age", "sex_male", "bmi", "hba1c", "t2d", "prediabetes", "hypertension", "on_metformin",
          "total_chol", "ldl", "hdl", "triglycerides"]
DYNAMIC = ["glucose", "glucose_lag15", "glucose_lag30", "glucose_lag60", "roc_15", "roc_30",
           "glucose_mean_60", "glucose_std_60", "glucose_min_120", "glucose_max_120",
           "carbs_1h", "carbs_2h", "mins_since_meal", "steps_1h", "steps_2h", "hr_mean_30",
           "sleep_hours_last_night", "sleep_eff_last_night", "hour_sin", "hour_cos"]
FEATURES = DYNAMIC + STATIC


def _patient_dynamic(w: pd.DataFrame) -> pd.DataFrame:
    """Dynamic features for one patient's 5-min series (sorted by time)."""
    g = w["glucose"]
    f = pd.DataFrame(index=w.index)
    f["glucose"] = g
    for m in (15, 30, 60):
        f[f"glucose_lag{m}"] = g.shift(m // BIN_MIN)
    f["roc_15"] = (g - f["glucose_lag15"]) / 15  # mg/dL per min
    f["roc_30"] = (g - f["glucose_lag30"]) / 30
    f["glucose_mean_60"] = g.rolling(12).mean()
    f["glucose_std_60"] = g.rolling(12).std()
    f["glucose_min_120"] = g.rolling(24).min()
    f["glucose_max_120"] = g.rolling(24).max()

    f["carbs_1h"] = w["carbs"].rolling(12, min_periods=1).sum()
    f["carbs_2h"] = w["carbs"].rolling(24, min_periods=1).sum()
    last_meal = w["timestamp"].where(w["carbs"] > 0).ffill()
    f["mins_since_meal"] = ((w["timestamp"] - last_meal).dt.total_seconds() / 60).fillna(MEAL_CAP_MIN).clip(upper=MEAL_CAP_MIN)

    f["steps_1h"] = w["steps"].rolling(12, min_periods=1).sum()
    f["steps_2h"] = w["steps"].rolling(24, min_periods=1).sum()
    f["hr_mean_30"] = w["hr"].rolling(6, min_periods=1).mean()
    f["sleep_hours_last_night"] = w["sleep_hours_last_night"]
    f["sleep_eff_last_night"] = w["sleep_eff_last_night"]

    hour = w["timestamp"].dt.hour + w["timestamp"].dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Label: max over the next 24 bins, i.e. (t, t+120 min]
    future_max = g[::-1].rolling(HORIZON, min_periods=HORIZON).max()[::-1].shift(-1)
    f["future_max_glucose"] = future_max
    f["label"] = (future_max > THRESHOLD).astype(float).where(future_max.notna())
    return f


def static_features(patients: pd.DataFrame) -> pd.DataFrame:
    s = patients.set_index("patient_id").copy()
    s["sex_male"] = (s["sex"] == "M").astype(int)
    for c in ["t2d", "prediabetes", "hypertension", "on_metformin"]:
        s[c] = s[c].astype(int)
    return s[STATIC]


def make_features(wear: pd.DataFrame, patients: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
    """Features + label for every (patient, timestamp). Usable for one patient (twin) or all."""
    wear = wear.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    parts = [_patient_dynamic(w) for _, w in wear.groupby("patient_id", sort=False)]
    dyn = pd.concat(parts)
    df = pd.concat([wear[["patient_id", "timestamp"]], dyn], axis=1)
    df = df.join(static_features(patients), on="patient_id")
    if drop_incomplete:
        df = df.dropna(subset=["glucose_min_120", "label"])  # first/last 2 h of each patient
        df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    patients = pd.read_parquet(PATIENTS)
    wear = pd.read_parquet(WEARABLES)
    df = make_features(wear, patients)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    at_risk = df["glucose"] <= THRESHOLD
    print(f"Rows: {len(df):,}  Patients: {df['patient_id'].nunique()}  Features: {len(FEATURES)}  (synthetic)")
    print(f"Positive rate, all rows: {df['label'].mean():.3f}")
    print(f"Positive rate, rows with glucose <= {THRESHOLD:.0f} now: {df.loc[at_risk, 'label'].mean():.3f}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
