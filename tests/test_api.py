"""API smoke tests against the real processed data and model."""
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (Path("data/processed/features.parquet").exists() and Path("models/model.joblib").exists()),
    reason="run `make data` and `make train` first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def test_patients_sorted_by_risk(client):
    r = client.get("/patients").json()
    risks = [p["risk"] for p in r["patients"]]
    assert len(risks) > 100 and risks == sorted(risks, reverse=True)


def test_patient_detail_and_predict(client):
    from api.service import DEMO_PATIENT
    d = client.get(f"/patients/{DEMO_PATIENT}").json()
    assert d["ehr"]["t2d"] and len(d["now"]["reasons"]) == 3
    assert len(d["series"]["glucose"]) == 24 * 12 + 1
    p = client.post("/predict", json={"patient_id": DEMO_PATIENT}).json()
    assert 0 <= p["risk"] <= 1


def test_whatif_walk_lowers_peak(client):
    from api.service import DEMO_PATIENT
    r = client.post("/whatif", json={"patient_id": DEMO_PATIENT, "post_meal_walk_min": 20, "mc_draws": 5}).json()
    assert r["modified"]["peak_glucose"] < r["baseline"]["peak_glucose"]


def test_errors(client):
    assert client.get("/patients/nope").status_code == 404
    assert client.post("/predict", json={"patient_id": "nope"}).status_code == 404
    assert client.get("/patients", params={"at": "2030-01-01"}).status_code == 422
