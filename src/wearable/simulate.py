"""Simulate 14 days of wearable data for every patient.

Run: python -m src.wearable.simulate
Input:  data/processed/patients.parquet
Output: data/processed/wearables.parquet  (synthetic)
"""
from pathlib import Path

import pandas as pd

from src.wearable.simulator import eag, simulate_patient

PATIENTS = Path("data/processed/patients.parquet")
OUT = Path("data/processed/wearables.parquet")
DAYS = 14


def summarize(wear: pd.DataFrame, patients: pd.DataFrame) -> None:
    per = wear.groupby("patient_id").agg(
        mean_glucose=("glucose", "mean"),
        pct_gt180=("glucose", lambda g: 100 * (g > 180).mean()),
        pct_lt70=("glucose", lambda g: 100 * (g < 70).mean()),
        max_glucose=("glucose", "max"),
        steps_per_day=("steps", lambda s: s.sum() / DAYS),
    )
    per = per.join(patients.set_index("patient_id")[["hba1c", "t2d"]])
    per["eag"] = eag(per["hba1c"])
    per["group"] = per["t2d"].map({True: "T2D", False: "Prediabetes"})
    nights = wear.drop_duplicates(["patient_id", "sleep_hours_last_night"])

    print(f"\nRows: {len(wear):,}  Patients: {wear['patient_id'].nunique()}  Days: {DAYS}  (synthetic)")
    print(per.groupby("group")[["hba1c", "eag", "mean_glucose", "max_glucose", "pct_gt180", "pct_lt70",
                                "steps_per_day"]].mean().round(1).to_string())
    print(f"\nCorr(eAG, simulated mean glucose): {per['eag'].corr(per['mean_glucose']):.3f}")
    print(f"Rows with glucose > 180: {100 * (wear['glucose'] > 180).mean():.1f}%")
    print(f"Sleep h/night: mean {nights['sleep_hours_last_night'].mean():.2f}, "
          f"<6h nights {100 * (nights['sleep_hours_last_night'] < 6).mean():.0f}%")
    print(f"HR: mean {wear['hr'].mean():.0f}, p1-p99 {wear['hr'].quantile(.01):.0f}-{wear['hr'].quantile(.99):.0f}")


def main() -> None:
    patients = pd.read_parquet(PATIENTS)
    frames = [simulate_patient(row, DAYS) for _, row in patients.iterrows()]
    wear = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wear.to_parquet(OUT, index=False)
    summarize(wear, patients)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
