"""CLI demo: risk before and after a 20-minute post-meal walk.

Run: python -m src.twin.demo
     python -m src.twin.demo --patient-id <id> --time "2026-01-07 13:40" --walk 20 --meal-carbs 50 --sleep-hours 7.5

Defaults: the median-HbA1c patient, at their lunch on 7 Jan, with a 20-minute walk.
"""
import argparse

import pandas as pd

from src.twin.twin import _patients, what_if
from src.wearable.simulator import START, generate_inputs, patient_seed


def default_case() -> tuple[str, pd.Timestamp]:
    """Median-HbA1c patient (typical case); time = their lunch (first meal 12:00-16:00) on day 2."""
    pts = _patients()
    p = pts.loc[(pts["hba1c"] - pts["hba1c"].median()).abs().idxmin()]
    tl = generate_inputs(p, 14, patient_seed(p["patient_id"]))["timeline"]
    day = START + pd.Timedelta(days=2)
    lunch = tl[(tl["carbs"] > 0) & (tl["timestamp"] >= day + pd.Timedelta(hours=12))
               & (tl["timestamp"] < day + pd.Timedelta(hours=16))]
    return p["patient_id"], lunch["timestamp"].iloc[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patient-id")
    ap.add_argument("--time")
    ap.add_argument("--walk", type=float, default=20, help="post-meal walk minutes")
    ap.add_argument("--meal-carbs", type=float)
    ap.add_argument("--sleep-hours", type=float)
    args = ap.parse_args()

    pid, at = default_case()
    pid = args.patient_id or pid
    at = args.time or at

    r = what_if(pid, at, meal_carbs=args.meal_carbs, post_meal_walk_min=args.walk or None,
                sleep_hours=args.sleep_hours)
    p = _patients().set_index("patient_id").loc[pid]
    b, m, c = r["baseline"], r["modified"], r["changes"]

    print(f"\nPatient {pid[:8]}...  HbA1c {p['hba1c']:.1f}%  BMI {p['bmi']:.1f}  "
          f"{'T2D' if p['t2d'] else 'prediabetes'}{', metformin' if p['on_metformin'] else ''}  (synthetic)")
    print(f"Time {r['time']}  |  meal at {c['meal_time'][11:16]}: {c['meal_carbs_baseline']} g"
          + (f" -> {c['meal_carbs_modified']} g" if args.meal_carbs is not None else ""))
    if "walk" in c:
        print(f"What-if: {c['walk']['minutes']}-min walk starting {c['walk']['start'][11:16]}")
    if "sleep_hours_baseline" in c:
        print(f"What-if: sleep {c['sleep_hours_baseline']} h -> {c['sleep_hours_modified']} h")

    rows = [
        ("Spike probability, twin re-sim (next 2 h)", b["sim_spike_prob"], m["sim_spike_prob"], "{:.0%}"),
        ("ML risk now", b["model_risk_now"], m["model_risk_now"], "{:.0%}"),
        ("ML risk peak over next 2 h", b["model_risk_peak"], m["model_risk_peak"], "{:.0%}"),
        ("Peak glucose (mg/dL)", b["peak_glucose"], m["peak_glucose"], "{:.0f}"),
        ("Minutes above 180", b["minutes_above_180"], m["minutes_above_180"], "{:.0f}"),
    ]
    print(f"\n{'':<42}{'Before':>10}{'After':>10}")
    for name, before, after, fmt in rows:
        print(f"{name:<42}{fmt.format(before):>10}{fmt.format(after):>10}")

    print(f"\n{'time':<8}{'glucose before':>16}{'after':>8}{'ML risk before':>16}{'after':>8}{'steps after':>13}")
    for i in range(0, len(b["timestamps"]), 3):  # every 15 min
        print(f"{b['timestamps'][i][11:16]:<8}{b['glucose'][i]:>16.0f}{m['glucose'][i]:>8.0f}"
              f"{b['model_risk'][i]:>16.0%}{m['model_risk'][i]:>8.0%}{m['steps'][i]:>13}")
    print("\nSynthetic data; research prototype, not a medical device.")


if __name__ == "__main__":
    main()
