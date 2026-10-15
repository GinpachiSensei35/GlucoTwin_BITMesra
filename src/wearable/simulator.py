"""Physiological wearable simulator, seeded from each patient's EHR row.

Two stages so the twin can edit behaviour and re-simulate:
    inputs = generate_inputs(patient, days, seed)     # meals, steps, sleep (behaviour)
    df     = simulate_glucose(patient, inputs, seed)  # CGM + HR (physiology + noise)
Noise uses its own RNG stream, so edited inputs see identical sensor noise.

All output is synthetic. Research prototype, not a medical device.
"""
import numpy as np
import pandas as pd

STEP_MIN = 5
BINS_PER_DAY = 24 * 60 // STEP_MIN
START = pd.Timestamp("2026-01-05")  # a Monday; arbitrary fixed origin

# Physiology constants (hand-tuned for plausibility, see docs)
BASAL_FRAC = 0.89        # fasting baseline as a fraction of eAG (meals add the rest)
MIN_BASAL = 80.0         # mg/dL; non-insulin users rarely sit below this fasting
MEAL_GAIN = 0.75         # mg/dL peak rise per gram of carbs at insulin resistance 1.0
METFORMIN_FACTOR = 0.85  # blunted meal rise
EXERCISE_GAIN = 14.0     # mg/dL drop per 1000 steps at kernel peak
DAWN_AMP = 6.0           # mg/dL early-morning rise at insulin resistance 1.0
MEAL_WINDOW = 8 * 12     # bins a meal response is tracked (8 h)
EXERCISE_WINDOW = 3 * 12  # bins an activity effect lasts (3 h)

# Indian carb-heavy day: (name, mean hour, sd hour, mean carbs g)
MEALS = [
    ("breakfast", 8.75, 0.4, 50),  # poha / idli / paratha
    ("lunch", 13.75, 0.4, 85),     # rice + dal + roti
    ("snack", 17.5, 0.3, 28),      # chai + biscuits / namkeen (optional)
    ("dinner", 21.0, 0.4, 75),     # roti / rice + sabzi
]


def eag(hba1c: float) -> float:
    """Estimated average glucose (mg/dL) from HbA1c (%)."""
    return 28.7 * hba1c - 46.7


def patient_seed(patient_id: str, base_seed: int = 0) -> int:
    return (int(str(patient_id).replace("-", "")[:8], 16) + base_seed) % 2**32


def patient_params(patient, seed: int) -> dict:
    """Stable per-patient physiology + habits, derived from EHR and seed."""
    rng = np.random.default_rng([seed, 0])
    hba1c, bmi, age = float(patient["hba1c"]), float(patient["bmi"]), float(patient["age"])
    ir = float(np.clip(1 + 0.04 * (bmi - 25) + 0.25 * (hba1c - 5.5), 0.7, 2.5))
    age_factor = np.clip(1.3 - age / 100, 0.4, 1.0)  # older -> walk less
    return {
        "basal": max(BASAL_FRAC * eag(hba1c), MIN_BASAL),
        "ir": ir,
        "metformin": bool(patient["on_metformin"]),
        "appetite": float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.4)),
        "activity": float(rng.lognormal(0, 0.35) * age_factor),
        "p_morning_walk": float(rng.uniform(0, 0.6) * age_factor),
        "p_evening_walk": float(rng.uniform(0, 0.5) * age_factor),
        "p_snack": float(rng.uniform(0.3, 0.9)),
        "sleep_mean": float(np.clip(rng.normal(6.7, 0.7), 4.5, 8.5)),
        "eff_mean": float(np.clip(rng.normal(0.87, 0.04), 0.70, 0.97)),
        "hr_rest": float(rng.normal(72, 6) + 0.3 * (bmi - 25)),
    }


def _bin(t: pd.Timestamp) -> int:
    return int((t - START) / pd.Timedelta(minutes=STEP_MIN))


def asleep_mask(timestamps: pd.Series, sleep: pd.DataFrame) -> np.ndarray:
    ts = pd.DatetimeIndex(timestamps)
    mask = np.zeros(len(ts), dtype=bool)
    for bed, hours, eff in sleep[["bed_time", "sleep_hours", "sleep_efficiency"]].itertuples(index=False):
        wake = bed + pd.Timedelta(hours=hours / eff)  # time in bed = sleep / efficiency
        mask |= (ts >= bed) & (ts < wake)
    return mask


def generate_inputs(patient, days: int = 14, seed: int = 0) -> dict:
    """Behaviour: meals (carbs per 5-min bin), steps per bin, nightly sleep."""
    prm = patient_params(patient, seed)
    rng = np.random.default_rng([seed, 1])
    n = days * BINS_PER_DAY
    ts = START + pd.to_timedelta(np.arange(n) * STEP_MIN, "min")
    carbs = np.zeros(n)
    steps = np.zeros(n)

    # Night d starts on the evening of day d; night -1 covers the morning of day 0.
    nights = []
    for d in range(-1, days):
        bed = START + pd.Timedelta(days=d, hours=23) + pd.Timedelta(minutes=rng.normal(0, 40))
        hours = float(np.clip(rng.normal(prm["sleep_mean"], 0.8), 3.5, 9.5))
        eff = float(np.clip(rng.normal(prm["eff_mean"], 0.04), 0.60, 0.98))
        nights.append((d, bed, hours, eff))
    sleep = pd.DataFrame(nights, columns=["night", "bed_time", "sleep_hours", "sleep_efficiency"])

    def add(arr, start: pd.Timestamp, n_bins: int, value: float):
        i = _bin(start)
        lo, hi = max(i, 0), min(i + n_bins, n)
        if lo < hi:
            arr[lo:hi] += value

    for d in range(days):
        day = START + pd.Timedelta(days=d)
        prev = sleep.iloc[d]  # night d-1
        wake = prev.bed_time + pd.Timedelta(hours=prev.sleep_hours / prev.sleep_efficiency)

        meal_times = {}
        for name, mu, sd, grams in MEALS:
            t = day + pd.Timedelta(hours=rng.normal(mu, sd))
            g = grams * prm["appetite"] * rng.lognormal(0, 0.15)
            if name == "breakfast":
                t = max(t, wake + pd.Timedelta(minutes=45))
            if name == "snack" and rng.random() > prm["p_snack"]:
                continue
            meal_times[name] = t
            add(carbs, t, 1, round(g))

        if rng.random() < prm["p_morning_walk"]:
            start = wake + pd.Timedelta(minutes=rng.uniform(10, 40))
            add(steps, start, int(rng.integers(5, 9)), rng.uniform(420, 560))  # 25-40 min
        if rng.random() < prm["p_evening_walk"]:
            start = meal_times["dinner"] + pd.Timedelta(minutes=rng.uniform(20, 45))
            add(steps, start, int(rng.integers(3, 6)), rng.uniform(400, 520))  # 15-25 min

    # Background ambulation while awake: bursty, ~2-5k steps/day
    active = rng.random(n) < 0.35
    steps += active * rng.gamma(2.0, 40 * prm["activity"], n)
    steps[asleep_mask(ts, sleep)] = 0

    timeline = pd.DataFrame({"timestamp": ts, "carbs": carbs, "steps": np.round(steps)})
    return {"timeline": timeline, "sleep": sleep}


def _sleep_multiplier(hours, eff):
    """Next-day insulin-resistance multiplier from last night's sleep (>= 1)."""
    short = np.clip((6.0 - hours) / 2.0, 0, 1)
    poor = np.clip((0.85 - eff) / 0.15, 0, 1)
    return 1 + 0.15 * short + 0.10 * poor


def simulate_glucose(patient, inputs: dict, seed: int = 0) -> pd.DataFrame:
    """Physiology: CGM glucose + heart rate from behaviour inputs."""
    prm = patient_params(patient, seed)
    tl, sleep = inputs["timeline"], inputs["sleep"]
    n = len(tl)
    carbs = tl["carbs"].to_numpy(float)
    asleep = asleep_mask(tl["timestamp"], sleep)
    steps = np.where(asleep, 0, tl["steps"].to_numpy(float))

    # Last night's sleep for each bin's calendar day
    day_idx = np.arange(n) // BINS_PER_DAY
    last_night = sleep.set_index("night").reindex(day_idx - 1)
    sl_hours = last_night["sleep_hours"].to_numpy()
    sl_eff = last_night["sleep_efficiency"].to_numpy()
    mult = _sleep_multiplier(sl_hours, sl_eff)
    ir_t = prm["ir"] * mult

    g = prm["basal"] * (1 + 0.3 * (mult - 1))

    # Dawn phenomenon: small rise ~4-8 am
    hour = (np.arange(n) * STEP_MIN % 1440) / 60
    g = g + DAWN_AMP * prm["ir"] * np.exp(-0.5 * ((hour - 6.0) / 1.2) ** 2)

    # Meals: gamma-like response, peak at 45-75 min; higher IR -> bigger, later, slower return
    met = METFORMIN_FACTOR if prm["metformin"] else 1.0
    for i in np.flatnonzero(carbs):
        ir = ir_t[i]
        amp = carbs[i] * MEAL_GAIN * ir * met
        tp = 45 + 30 * np.clip((ir - 1) / 1.5, 0, 1)
        k = 2.0 / (1 + 0.4 * (ir - 1))
        lag = np.arange(min(MEAL_WINDOW, n - i)) * STEP_MIN
        g[i:i + len(lag)] += amp * (lag / tp) ** k * np.exp(k * (1 - lag / tp))

    # Activity: lowers glucose for 1-3 h, peak effect ~30 min after. Full effect on a
    # post-meal excursion, ~30% when fasting (muscle uptake scales with available glucose).
    lag = np.arange(EXERCISE_WINDOW) * STEP_MIN
    kern = (lag / 30) * np.exp(1 - lag / 30)
    drop = EXERCISE_GAIN * np.convolve(steps / 1000, kern)[:n]
    g = g - drop * np.clip(0.3 + (g - prm["basal"]) / 60, 0.3, 1)

    # Noise: slow physiological drift + CGM sensor noise (both AR(1))
    rng = np.random.default_rng([seed, 2])
    drift = _ar1(rng, n, phi=0.98, sd=1.5)
    sensor = _ar1(rng, n, phi=0.7, sd=2.5)
    glucose = np.clip(g + drift + sensor, 40, 400)

    # Heart rate: rises with activity (smoothed), drops in sleep
    effort = pd.Series(steps).ewm(span=3).mean().to_numpy()
    hr = prm["hr_rest"] + 0.07 * effort - 9 * asleep + rng.normal(0, 2, n)

    return pd.DataFrame({
        "timestamp": tl["timestamp"].to_numpy(),
        "glucose": np.round(glucose, 1),
        "hr": np.round(hr).astype(int),
        "steps": steps.astype(int),
        "carbs": carbs.astype(int),
        "asleep": asleep,
        "sleep_hours_last_night": np.round(sl_hours, 2),
        "sleep_eff_last_night": np.round(sl_eff, 3),
    })


def _ar1(rng, n: int, phi: float, sd: float) -> np.ndarray:
    e = rng.normal(0, sd, n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi**2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def simulate_patient(patient, days: int = 14, seed: int | None = None, inputs: dict | None = None) -> pd.DataFrame:
    """Convenience: one patient end to end. Pass `inputs` to override behaviour (twin)."""
    if seed is None:
        seed = patient_seed(patient["patient_id"])
    if inputs is None:
        inputs = generate_inputs(patient, days, seed)
    df = simulate_glucose(patient, inputs, seed)
    df.insert(0, "patient_id", patient["patient_id"])
    return df
