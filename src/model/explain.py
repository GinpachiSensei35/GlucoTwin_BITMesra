"""SHAP explanations for the LightGBM spike model."""
import warnings

import numpy as np
import pandas as pd
import shap


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Per-row SHAP values (log-odds contribution to 'spike') for each feature."""
    with warnings.catch_warnings():  # shap warns that LightGBM binary output format changed
        warnings.simplefilter("ignore", UserWarning)
        sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):  # older shap: one array per class
        sv = sv[1]
    return np.asarray(sv)


def global_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    sv = shap_values(model, X)
    imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def explain_row(model, x: pd.DataFrame, top: int = 5) -> list[dict]:
    """Top contributing features for a single row (for the API / dashboard)."""
    sv = shap_values(model, x)[0]
    order = np.argsort(-np.abs(sv))[:top]
    return [{"feature": x.columns[i], "value": float(x.iloc[0, i]), "shap": round(float(sv[i]), 4)} for i in order]
