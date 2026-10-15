"""What-if twin: no-op overrides change nothing; walks and smaller meals lower glucose."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (Path("data/processed/patients.parquet").exists() and Path("models/model.joblib").exists()),
    reason="run `make data` and `make train` first",
)

AT = "2026-01-07 13:15"


@pytest.fixture(scope="module")
def pid():
    from src.twin.twin import _patients
    pts = _patients()
    return pts.loc[(pts["hba1c"] - pts["hba1c"].median()).abs().idxmin(), "patient_id"]


def test_no_override_equals_baseline(pid):
    from src.twin.twin import what_if
    r = what_if(pid, AT, mc_draws=5)
    assert r["baseline"]["glucose"] == r["modified"]["glucose"]
    assert r["baseline"]["model_risk"] == r["modified"]["model_risk"]


def test_walk_and_smaller_meal_lower_peak(pid):
    from src.twin.twin import what_if
    walk = what_if(pid, AT, post_meal_walk_min=20, mc_draws=5)
    meal = what_if(pid, AT, meal_carbs=30, mc_draws=5)
    for r in (walk, meal):
        assert r["modified"]["peak_glucose"] < r["baseline"]["peak_glucose"]
        assert r["modified"]["sim_spike_prob"] <= r["baseline"]["sim_spike_prob"]
        # the past is unchanged by a future intervention
        assert r["modified"]["glucose"][0] == r["baseline"]["glucose"][0]


def test_better_sleep_does_not_raise_peak(pid):
    from src.twin.twin import what_if
    r = what_if(pid, AT, sleep_hours=8.0, mc_draws=5)
    assert r["modified"]["peak_glucose"] <= r["baseline"]["peak_glucose"]
