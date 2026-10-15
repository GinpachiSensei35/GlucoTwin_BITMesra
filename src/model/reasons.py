"""Turn SHAP values into plain-English reasons a clinician can read.

Related features are grouped (e.g. hour_sin + hour_cos = time of day) and their SHAP summed,
so the top reasons are distinct ideas rather than five versions of "glucose".
"""
import numpy as np
import pandas as pd

GROUPS = {
    "meal": ["carbs_1h", "carbs_2h", "mins_since_meal"],
    "glucose": ["glucose", "glucose_lag15", "glucose_lag30", "glucose_lag60", "roc_15", "roc_30",
                "glucose_mean_60", "glucose_std_60", "glucose_min_120", "glucose_max_120"],
    "activity": ["steps_1h", "steps_2h"],
    "heart_rate": ["hr_mean_30"],
    "sleep": ["sleep_hours_last_night", "sleep_eff_last_night"],
    "time_of_day": ["hour_sin", "hour_cos"],
    "hba1c": ["hba1c"],
    "bmi": ["bmi"],
    "metformin": ["on_metformin"],
    "diagnosis": ["t2d", "prediabetes"],
    "lipids": ["total_chol", "ldl", "hdl", "triglycerides"],
    "demographics": ["age", "sex_male"],
    "hypertension": ["hypertension"],
}


def _describe(group: str, x: pd.Series, ts: pd.Timestamp) -> str:
    if group == "meal":
        if x["mins_since_meal"] >= 360 or x["carbs_2h"] == 0:
            return "No meal logged in the last 2 hours"
        return f"Ate {x['carbs_2h']:.0f} g carbs, {x['mins_since_meal']:.0f} min ago"
    if group == "glucose":
        trend = x["roc_15"]
        word = "rising" if trend > 0.5 else "falling" if trend < -0.5 else "steady"
        return f"Glucose {x['glucose']:.0f} mg/dL and {word} ({trend:+.1f} mg/dL per min)"
    if group == "activity":
        return f"{x['steps_1h']:.0f} steps in the last hour"
    if group == "heart_rate":
        return f"Heart rate {x['hr_mean_30']:.0f} bpm over the last 30 min"
    if group == "sleep":
        return f"Slept {x['sleep_hours_last_night']:.1f} h last night ({x['sleep_eff_last_night']:.0%} efficiency)"
    if group == "time_of_day":
        return f"Time of day ({ts:%H:%M}), matching this patient's usual meal pattern"
    if group == "hba1c":
        return f"HbA1c {x['hba1c']:.1f}% (long-term glucose control)"
    if group == "bmi":
        return f"BMI {x['bmi']:.1f}"
    if group == "metformin":
        return "On metformin" if x["on_metformin"] else "Not on metformin"
    if group == "diagnosis":
        return "Diagnosed type 2 diabetes" if x["t2d"] else "Prediabetes diagnosis"
    if group == "lipids":
        return f"Lipids: LDL {x['ldl']:.0f}, HDL {x['hdl']:.0f}, TG {x['triglycerides']:.0f} mg/dL"
    if group == "demographics":
        return f"Age {x['age']:.0f}, {'male' if x['sex_male'] else 'female'}"
    if group == "hypertension":
        return "Hypertension" if x["hypertension"] else "No hypertension"
    return group


def plain_reasons(x: pd.Series, shap_row: np.ndarray, features: list[str], ts: pd.Timestamp, top: int = 3) -> list[dict]:
    contrib = dict(zip(features, shap_row))
    out = []
    for group, cols in GROUPS.items():
        cols = [c for c in cols if c in contrib]
        if not cols:
            continue
        s = float(sum(contrib[c] for c in cols))
        out.append({"factor": group, "text": _describe(group, x, ts), "shap": round(s, 3),
                    "direction": "raises risk" if s > 0 else "lowers risk"})
    out.sort(key=lambda r: -abs(r["shap"]))
    return out[:top]
