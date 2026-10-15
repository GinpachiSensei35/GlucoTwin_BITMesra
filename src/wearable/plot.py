"""Plot 2 days of simulated wearables for low / medium / high HbA1c patients.

Run: python -m src.wearable.plot
Output: docs/img/sim_<label>_hba1c.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PATIENTS = Path("data/processed/patients.parquet")
WEARABLES = Path("data/processed/wearables.parquet")
OUT_DIR = Path("docs/img")
DAYS_SHOWN = 2
FIRST_DAY = 2  # skip day 0 so "last night" sleep comes from the simulated night


def pick_patients(patients: pd.DataFrame) -> dict:
    """Patient closest to the 5th, 50th percentile and max HbA1c."""
    s = patients.set_index("patient_id")["hba1c"]
    targets = {"low": s.quantile(0.05), "medium": s.median(), "high": s.max()}
    return {label: (s - v).abs().idxmin() for label, v in targets.items()}


def plot_patient(df: pd.DataFrame, pt: pd.Series, label: str, path: Path) -> None:
    fig, (ax_g, ax_a) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.2]})
    t = df["timestamp"]

    # Sleep shading
    asleep = df["asleep"].to_numpy()
    edges = pd.Series(asleep).astype(int).diff().fillna(asleep[0]).to_numpy()
    starts, ends = list(t[edges == 1]), list(t[edges == -1])
    if len(ends) < len(starts):
        ends.append(t.iloc[-1])
    if asleep[0] and (not starts or starts[0] > ends[0]):
        starts.insert(0, t.iloc[0])
    for i, (a, b) in enumerate(zip(starts, ends)):
        for ax in (ax_g, ax_a):
            ax.axvspan(a, b, color="#8a8fd6", alpha=0.15, lw=0, label="asleep" if (i == 0 and ax is ax_g) else None)

    ax_g.axhspan(70, 180, color="#2ca02c", alpha=0.06, lw=0)
    ax_g.axhline(180, color="#d62728", ls="--", lw=1, label="180 mg/dL threshold")
    ax_g.axhline(70, color="#999999", ls=":", lw=1)
    ax_g.plot(t, df["glucose"], color="#1f4e8c", lw=1.4, label="CGM glucose")

    meals = df[df["carbs"] > 0]
    ax_g.scatter(meals["timestamp"], meals["glucose"], marker="v", s=60, color="#e08a00", zorder=5, label="meal")
    for ts, g, c in meals[["timestamp", "glucose", "carbs"]].itertuples(index=False):
        ax_g.annotate(f"{c} g", (ts, g), textcoords="offset points", xytext=(0, -16), ha="center",
                      fontsize=8, color="#a05f00")

    ax_g.set_ylabel("Glucose (mg/dL)")
    ax_g.set_ylim(40, max(260, df["glucose"].max() + 20))
    flags = ", ".join(f for f, on in [("T2D", pt["t2d"]), ("prediabetes", pt["prediabetes"]),
                                       ("metformin", pt["on_metformin"]), ("HTN", pt["hypertension"])] if on)
    ax_g.set_title(f"{label.capitalize()} HbA1c — {pt['hba1c']:.1f}% · BMI {pt['bmi']:.1f} · "
                   f"{int(pt['age'])}{pt['sex']} · {flags}   (synthetic patient)", loc="left", fontsize=11)
    ax_g.legend(loc="upper right", fontsize=8, ncol=5, frameon=False)

    ax_a.bar(t, df["steps"], width=pd.Timedelta(minutes=5), color="#6a9f58", label="steps / 5 min")
    ax_a.set_ylabel("Steps / 5 min")
    ax_hr = ax_a.twinx()
    ax_hr.plot(t, df["hr"], color="#c0392b", lw=0.8, alpha=0.8, label="heart rate")
    ax_hr.set_ylabel("HR (bpm)")
    ax_a.legend(loc="upper left", fontsize=8, frameon=False)
    ax_hr.legend(loc="upper right", fontsize=8, frameon=False)

    fig.text(0.01, 0.005, "All data synthetic (Synthea EHR + GlucoTwin simulator). Research prototype, not a medical device.",
             fontsize=7, color="#666666")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    patients = pd.read_parquet(PATIENTS)
    wear = pd.read_parquet(WEARABLES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = wear["timestamp"].min() + pd.Timedelta(days=FIRST_DAY)
    t1 = t0 + pd.Timedelta(days=DAYS_SHOWN)
    for label, pid in pick_patients(patients).items():
        pt = patients.set_index("patient_id").loc[pid]
        df = wear[(wear["patient_id"] == pid) & (wear["timestamp"] >= t0) & (wear["timestamp"] < t1)]
        path = OUT_DIR / f"sim_{label}_hba1c.png"
        plot_patient(df.reset_index(drop=True), pt, label, path)
        print(f"{label:>6}: HbA1c {pt['hba1c']:.1f}  mean {df['glucose'].mean():.0f}  "
              f"max {df['glucose'].max():.0f}  ->  {path}")


if __name__ == "__main__":
    main()
