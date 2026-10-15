"""Train LightGBM spike model, compare with naive baseline, explain with SHAP.

Run: python -m src.model.train
Input:  data/processed/features.parquet
Output: models/model.joblib, models/metrics.json, models/shap_importance.csv, docs/img/shap_summary.png

Population: rows where current glucose <= 180 (predicting an upcoming spike, not an ongoing one).
All data is synthetic; the model partly learns our simulator. Not a medical device.
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.features.build import FEATURES, THRESHOLD
from src.model.evaluate import score_report
from src.model.explain import global_importance, shap_values

FEATURES_PATH = Path("data/processed/features.parquet")
MODEL_DIR = Path("models")
SHAP_PLOT = Path("docs/img/shap_summary.png")
SEED = 42
TEST_SIZE = 0.25
PRECISION_TARGET = 0.8
SHAP_SAMPLE = 3000

PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)


def split_by_patient(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Boolean train/test masks; every patient lands entirely on one side."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
    train = np.zeros(len(df), dtype=bool)
    train[train_idx] = True
    return train, ~train


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    train_mask, test_mask = split_by_patient(df)
    train_pts = set(df.loc[train_mask, "patient_id"])
    test_pts = set(df.loc[test_mask, "patient_id"])
    assert not train_pts & test_pts, "patient leakage between train and test"

    at_risk = (df["glucose"] <= THRESHOLD).to_numpy()
    tr = df[train_mask & at_risk]
    te_all = df[test_mask]
    te = df[test_mask & at_risk]

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(tr[FEATURES], tr["label"])

    p_te = model.predict_proba(te[FEATURES])[:, 1]
    base_te = te["glucose"].to_numpy()  # naive: current glucose > threshold (swept)
    # Lead-time scoring needs the full test series (crossings happen from <=180 rows)
    p_all = model.predict_proba(te_all[FEATURES])[:, 1]
    base_all = te_all["glucose"].to_numpy()

    model_rep = score_report(te["label"], p_te, te_all, p_all, PRECISION_TARGET)
    base_rep = score_report(te["label"], base_te, te_all, base_all, PRECISION_TARGET)

    all_rows = {
        "model": {"auroc": round(roc_auc_score(te_all["label"], p_all), 4),
                  "auprc": round(average_precision_score(te_all["label"], p_all), 4)},
        "baseline": {"auroc": round(roc_auc_score(te_all["label"], base_all), 4),
                     "auprc": round(average_precision_score(te_all["label"], base_all), 4)},
    }

    # SHAP on a sample of test rows
    X_shap = te[FEATURES].sample(min(SHAP_SAMPLE, len(te)), random_state=SEED)
    imp = global_importance(model, X_shap)
    MODEL_DIR.mkdir(exist_ok=True)
    imp.to_csv(MODEL_DIR / "shap_importance.csv", index=False)
    SHAP_PLOT.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values(model, X_shap), X_shap, max_display=15, show=False)
    plt.title("SHAP: drivers of predicted spike >180 mg/dL in 2 h (synthetic data)", fontsize=10)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT, dpi=110)
    plt.close()

    metrics = {
        "note": "All data synthetic (Synthea EHR + GlucoTwin simulator). Model partly learns the simulator; "
                "validation on real CGM data is future work. Research prototype, not a medical device.",
        "task": "P(max glucose in (t, t+120 min] > 180 mg/dL)",
        "eval_population": "test-patient rows with current glucose <= 180 mg/dL",
        "split": {"method": "GroupShuffleSplit by patient", "test_size": TEST_SIZE, "seed": SEED,
                  "train_patients": len(train_pts), "test_patients": len(test_pts),
                  "train_rows": int(len(tr)), "test_rows": int(len(te)),
                  "test_positive_rate": round(float(te["label"].mean()), 4)},
        "precision_target": PRECISION_TARGET,
        "model": {"name": "LightGBM", **model_rep},
        "baseline": {"name": "current glucose > threshold", **base_rep},
        "all_test_rows": all_rows,
        "top_features": imp.head(10).round(4).to_dict(orient="records"),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    joblib.dump({"model": model, "features": FEATURES, "threshold": model_rep["alert_threshold"],
                 "precision_target": PRECISION_TARGET}, MODEL_DIR / "model.joblib")

    print_report(metrics)


def print_report(m: dict) -> None:
    s = m["split"]
    print(f"\nSplit: {s['train_patients']} train / {s['test_patients']} test patients, "
          f"{s['test_rows']:,} test rows, positive rate {s['test_positive_rate']:.3f}  (synthetic)")
    print(f"Population: {m['eval_population']}\n")
    keys = ["auroc", "auprc", f"recall_at_precision_{m['precision_target']}", "alert_threshold",
            "events", "events_alerted", "event_recall", "median_lead_min"]
    table = pd.DataFrame({"LightGBM": [m["model"].get(k) for k in keys],
                          "Naive (glucose > thr)": [m["baseline"].get(k) for k in keys]}, index=keys)
    print(table.to_string())
    print(f"\nAll test rows (incl. currently >180): model AUROC {m['all_test_rows']['model']['auroc']}, "
          f"baseline AUROC {m['all_test_rows']['baseline']['auroc']}")
    print("\nTop 10 features (mean |SHAP|):")
    for i, r in enumerate(m["top_features"], 1):
        print(f"  {i:>2}. {r['feature']:<24} {r['mean_abs_shap']:.4f}")
    print("\nSaved models/model.joblib, models/metrics.json, models/shap_importance.csv, docs/img/shap_summary.png")


if __name__ == "__main__":
    main()
