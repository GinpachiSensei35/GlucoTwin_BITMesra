"""Metrics: AUROC, AUPRC, recall at fixed precision, median alert lead time."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.features.build import THRESHOLD


def recall_at_precision(y, score, target: float) -> tuple[float, float]:
    """Best recall with precision >= target, and the score threshold that achieves it."""
    prec, rec, thr = precision_recall_curve(y, score)
    ok = prec[:-1] >= target  # last point has no threshold
    if not ok.any():
        return 0.0, float("nan")
    i = np.argmax(np.where(ok, rec[:-1], -1))
    return float(rec[i]), float(thr[i])


def alert_lead_times(df: pd.DataFrame, score: np.ndarray, thr: float, horizon_min: int = 120) -> dict:
    """For each crossing above THRESHOLD, minutes between the first alert in the preceding
    `horizon_min` and the crossing. df must hold patient_id, timestamp, glucose (time-sorted)."""
    d = df[["patient_id", "timestamp", "glucose"]].copy()
    d["alert"] = (score >= thr) & (d["glucose"] <= THRESHOLD)
    leads, n_events = [], 0
    for _, p in d.groupby("patient_id", sort=False):
        ts = p["timestamp"].to_numpy()
        g = p["glucose"].to_numpy()
        alert_ts = ts[p["alert"].to_numpy()]
        crossings = np.flatnonzero((g[:-1] <= THRESHOLD) & (g[1:] > THRESHOLD)) + 1
        for c in crossings:
            tc = ts[c]
            if tc - ts[0] < np.timedelta64(horizon_min, "m"):
                continue  # not enough history to have alerted
            n_events += 1
            win = alert_ts[(alert_ts >= tc - np.timedelta64(horizon_min, "m")) & (alert_ts < tc)]
            if len(win):
                leads.append((tc - win.min()) / np.timedelta64(1, "m"))
    return {
        "events": n_events,
        "events_alerted": len(leads),
        "event_recall": round(len(leads) / n_events, 4) if n_events else None,
        "median_lead_min": float(np.median(leads)) if leads else None,
    }


def score_report(y, score, series: pd.DataFrame, series_score: np.ndarray, precision_target: float) -> dict:
    """Ranking metrics on (y, score); lead time on the full time series (series, series_score)."""
    rec, thr = recall_at_precision(y, score, precision_target)
    out = {
        "auroc": round(float(roc_auc_score(y, score)), 4),
        "auprc": round(float(average_precision_score(y, score)), 4),
        f"recall_at_precision_{precision_target}": round(rec, 4),
        "alert_threshold": None if np.isnan(thr) else round(thr, 4),
    }
    if not np.isnan(thr):
        out.update(alert_lead_times(series, series_score, thr))
    return out
