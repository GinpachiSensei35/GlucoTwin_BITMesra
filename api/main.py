"""GlucoTwin API.

Run: uvicorn api.main:app --port 8800   (docs at /docs)
All data synthetic. Research prototype, not a medical device.
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from api import service

app = FastAPI(
    title="GlucoTwin API",
    description="Digital twin for type 2 diabetes: predicts glucose > 180 mg/dL within 2 h and "
                "simulates what-if scenarios. " + service.DISCLAIMER,
    version="0.1.0",
)


class PredictRequest(BaseModel):
    patient_id: str
    at: str | None = Field(None, description="timestamp inside the simulated period; default demo time")


class WhatIfRequest(PredictRequest):
    meal_carbs: float | None = Field(None, ge=0, le=300, description="grams for the meal near `at`")
    post_meal_walk_min: float | None = Field(None, ge=0, le=120)
    sleep_hours: float | None = Field(None, ge=2, le=12, description="last night's sleep")
    mc_draws: int = Field(30, ge=1, le=200)


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as e:
        raise HTTPException(404, f"unknown patient {e}")
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/")
def root():
    lo, hi = service.time_range()
    return {"service": "GlucoTwin", "demo_patient": service.DEMO_PATIENT, "demo_time": str(service.DEMO_TIME),
            "valid_time_range": [str(lo), str(hi)], "disclaimer": service.DISCLAIMER}


@app.get("/patients")
def patients(at: str | None = Query(None)):
    return _call(service.list_patients, at)


@app.get("/patients/{patient_id}")
def patient(patient_id: str, at: str | None = Query(None)):
    return _call(service.patient_detail, patient_id, at)


@app.post("/predict")
def predict(req: PredictRequest):
    return _call(service.predict, req.patient_id, req.at)


@app.post("/whatif")
def whatif(req: WhatIfRequest):
    return _call(service.run_whatif, req.patient_id, req.at, req.meal_carbs, req.post_meal_walk_min,
                 req.sleep_hours, req.mc_draws)
