"""Load Synthea CSVs -> one row per adult T2D/prediabetes patient.

Run: python -m src.ehr.build
Output: data/processed/patients.parquet

All data is synthetic (Synthea). No real patient data.
"""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/synthea")
OUT = Path("data/processed/patients.parquet")
MIN_PATIENTS = 100
HBA1C_RANGE = (4.0, 15.0)  # %
MAX_AGE = 90

# SNOMED condition codes
T2D_CODES = {"44054006"}  # Synthea's "Diabetes" (metabolic syndrome module = T2D)
PREDIABETES_CODES = {"15777000"}
HYPERTENSION_CODES = {"59621000"}

# LOINC observation codes -> column name
OBS_CODES = {
    "4548-4": "hba1c",
    "39156-5": "bmi",
    "2093-3": "total_chol",
    "18262-6": "ldl",
    "2085-9": "hdl",
    "2571-8": "triglycerides",
}


def load_observations() -> pd.DataFrame:
    obs = pd.read_csv(RAW / "observations.csv", usecols=["DATE", "PATIENT", "CODE", "VALUE"], dtype=str)
    obs = obs[obs["CODE"].isin(OBS_CODES)].copy()
    obs["DATE"] = pd.to_datetime(obs["DATE"], utc=True)
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    return obs.dropna(subset=["VALUE"])


def latest_values(obs: pd.DataFrame) -> pd.DataFrame:
    """Most recent value per patient per code, pivoted wide."""
    latest = obs.sort_values("DATE").groupby(["PATIENT", "CODE"]).tail(1)
    wide = latest.pivot(index="PATIENT", columns="CODE", values="VALUE").rename(columns=OBS_CODES)
    return wide.reindex(columns=list(OBS_CODES.values()))


def condition_flags(ref_date: pd.Timestamp) -> pd.DataFrame:
    cond = pd.read_csv(RAW / "conditions.csv", usecols=["START", "STOP", "PATIENT", "CODE"], dtype=str)
    stop = pd.to_datetime(cond["STOP"], utc=True)
    cond["active"] = stop.isna() | (stop >= ref_date)

    def has(codes, active_only=True):
        mask = cond["CODE"].isin(codes) & (cond["active"] if active_only else True)
        return cond.loc[mask, "PATIENT"].unique()

    flags = pd.DataFrame(index=pd.Index(cond["PATIENT"].unique(), name="PATIENT"))
    flags["t2d"] = flags.index.isin(has(T2D_CODES, active_only=False))
    flags["prediabetes"] = flags.index.isin(has(PREDIABETES_CODES)) & ~flags["t2d"]
    flags["hypertension"] = flags.index.isin(has(HYPERTENSION_CODES))
    return flags


def metformin_flag(ref_date: pd.Timestamp) -> pd.Series:
    meds = pd.read_csv(RAW / "medications.csv", usecols=["START", "STOP", "PATIENT", "DESCRIPTION"], dtype=str)
    meds = meds[meds["DESCRIPTION"].str.contains("metformin", case=False, na=False)]
    stop = pd.to_datetime(meds["STOP"], utc=True)
    active = meds.loc[stop.isna() | (stop >= ref_date), "PATIENT"].unique()
    return pd.Series(True, index=pd.Index(active, name="PATIENT"), name="on_metformin")


def build() -> pd.DataFrame:
    obs = load_observations()
    ref_date = obs["DATE"].max()  # dataset snapshot date, used for age + "active" status

    pts = pd.read_csv(RAW / "patients.csv", usecols=["Id", "BIRTHDATE", "DEATHDATE", "GENDER"], dtype=str)
    pts = pts[pts["DEATHDATE"].isna()].rename(columns={"Id": "PATIENT"}).set_index("PATIENT")
    birth = pd.to_datetime(pts["BIRTHDATE"], utc=True)
    pts["age"] = ((ref_date - birth).dt.days / 365.25).astype(int)
    pts["sex"] = pts["GENDER"]

    df = (
        pts[["age", "sex"]]
        .join(condition_flags(ref_date), how="left")
        .join(metformin_flag(ref_date), how="left")
        .join(latest_values(obs), how="left")
    )
    bool_cols = ["t2d", "prediabetes", "hypertension", "on_metformin"]
    df[bool_cols] = df[bool_cols].fillna(False).astype(bool)

    df = df[(df["age"] >= 18) & (df["t2d"] | df["prediabetes"])]
    df = df.dropna(subset=["hba1c", "bmi"])  # simulator needs both
    # Plausibility filters: Synthea drives some treated diabetics' HbA1c to <4%,
    # which would give a near-zero eAG baseline; it also keeps some patients alive past 100.
    df = df[df["hba1c"].between(*HBA1C_RANGE) & (df["age"] <= MAX_AGE)]
    df = df.reset_index().rename(columns={"PATIENT": "patient_id"})
    df["source"] = "synthea"
    return df


def summarize(df: pd.DataFrame) -> None:
    print(f"\nPatients: {len(df)}  (synthetic, Synthea)")
    print(f"T2D: {df['t2d'].sum()}  Prediabetes: {df['prediabetes'].sum()}  "
          f"Hypertension: {df['hypertension'].sum()}  On metformin: {df['on_metformin'].sum()}")
    print(f"Sex: {df['sex'].value_counts().to_dict()}\n")
    num = ["age", "bmi", "hba1c", "total_chol", "ldl", "hdl", "triglycerides"]
    print(df[num].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(1).to_string())
    print("\nBy group (mean):")
    group = df["t2d"].map({True: "T2D", False: "Prediabetes"})
    print(df.groupby(group)[["age", "bmi", "hba1c", "on_metformin"]].mean().round(2).to_string())


def main() -> None:
    df = build()
    if len(df) < MIN_PATIENTS:
        raise SystemExit(f"Only {len(df)} qualifying patients (< {MIN_PATIENTS}); use src.ehr.generate instead.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    summarize(df)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
