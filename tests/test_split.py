"""Train/test split must be by patient: no patient on both sides."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.train import split_by_patient

FEATURES_PATH = Path("data/processed/features.parquet")


def _assert_disjoint(df: pd.DataFrame) -> None:
    train, test = split_by_patient(df)
    assert train.sum() > 0 and test.sum() > 0
    assert (train ^ test).all()  # every row on exactly one side
    assert not set(df.loc[train, "patient_id"]) & set(df.loc[test, "patient_id"])


def test_split_no_patient_overlap_synthetic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"patient_id": rng.choice([f"p{i}" for i in range(50)], 5000)})
    _assert_disjoint(df)


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="run `make data` first")
def test_split_no_patient_overlap_real_features():
    df = pd.read_parquet(FEATURES_PATH, columns=["patient_id"])
    _assert_disjoint(df)
