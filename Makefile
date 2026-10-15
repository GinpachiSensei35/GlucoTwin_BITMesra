# Run from repo root. Assumes the venv at .venv (see README).
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: data train api dashboard test

data:
	$(PY) -m src.ehr.build
	$(PY) -m src.wearable.simulate
	$(PY) -m src.features.build

train:
	$(PY) -m src.model.train

api:
	$(PY) -m uvicorn api.main:app --reload --port 8800

dashboard:
	$(PY) -m streamlit run dashboard/app.py

test:
	$(PY) -m pytest -q
