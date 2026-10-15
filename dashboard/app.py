"""GlucoTwin doctor dashboard.

Run from repo root: streamlit run dashboard/app.py
Uses the API at $GLUCOTWIN_API (default http://localhost:8800); falls back to calling the
service layer in-process if the API is not running.
All data synthetic. Research prototype, not a medical device.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import service  # noqa: E402

API = os.environ.get("GLUCOTWIN_API", "http://localhost:8800")
RISK_COLORS = {"high": "#d62728", "moderate": "#ff9f1c", "low": "#2ca02c"}

st.set_page_config(page_title="GlucoTwin", page_icon="🩸", layout="wide")


# ---------- data access: HTTP API, or in-process fallback ----------
@st.cache_data(ttl=60, show_spinner=False)
def api_up() -> bool:
    try:
        r = requests.get(f"{API}/", timeout=1)
        return r.ok and r.json().get("service") == "GlucoTwin"  # not some other app on the port
    except (requests.RequestException, ValueError):
        return False


@st.cache_data(show_spinner=False)
def get(path: str, at: str):
    if api_up():
        r = requests.get(f"{API}{path}", params={"at": at}, timeout=30)
        r.raise_for_status()
        return r.json()
    if path == "/patients":
        return service.list_patients(at)
    return service.patient_detail(path.rsplit("/", 1)[1], at)


@st.cache_data(show_spinner=False)
def post_whatif(payload: tuple) -> dict:
    body = dict(payload)
    if api_up():
        r = requests.post(f"{API}/whatif", json=body, timeout=120)
        r.raise_for_status()
        return r.json()
    return service.run_whatif(**body)


# ---------- sidebar ----------
st.sidebar.title("🩸 GlucoTwin")
st.sidebar.caption("Digital twin for type 2 diabetes · predicts glucose > 180 mg/dL in the next 2 h")
lo, hi = service.time_range()
day = st.sidebar.date_input("Simulated date", service.DEMO_TIME.date(), min_value=lo.date(), max_value=hi.date())
clock = st.sidebar.time_input("Time (\"now\")", service.DEMO_TIME.time(), step=300)
at = pd.Timestamp.combine(day, clock).floor("5min")
at = min(max(at, lo), hi)
if st.sidebar.button("↺ Reset to demo patient & time", width="stretch"):
    st.session_state.clear()
    st.rerun()
st.sidebar.caption(f"Data source: {'API ' + API if api_up() else 'in-process (API not running)'}")
st.sidebar.warning(service.DISCLAIMER, icon="⚠️")

# ---------- patient list ----------
plist = get("/patients", str(at))
df = pd.DataFrame(plist["patients"])
df.insert(0, "demo", df["patient_id"].eq(service.DEMO_PATIENT).map({True: "★", False: ""}))
df["id"] = df["patient_id"].str[:8]
ids = df["patient_id"].tolist()

if "pid" not in st.session_state:
    st.session_state.pid = service.DEMO_PATIENT

tab_list, tab_patient = st.tabs(["Patients by current risk", "Patient"])

with tab_list:
    st.subheader(f"Patients sorted by risk of > 180 mg/dL in the next 2 h · {at:%a %d %b %H:%M}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", len(df))
    c2.metric("High risk", int((df["risk_level"] == "high").sum()))
    c3.metric("Above 180 now", int((df["glucose_now"] > 180).sum()))
    view = df[["demo", "id", "age", "sex", "diagnosis", "hba1c", "bmi", "on_metformin", "glucose_now", "risk", "risk_level"]]
    event = st.dataframe(
        view, hide_index=True, width="stretch", height=520, on_select="rerun", selection_mode="single-row",
        column_config={
            "risk": st.column_config.ProgressColumn("risk", format="percent", min_value=0, max_value=1),
            "glucose_now": st.column_config.NumberColumn("glucose now", format="%.0f mg/dL"),
            "hba1c": st.column_config.NumberColumn("HbA1c %", format="%.1f"),
            "on_metformin": st.column_config.CheckboxColumn("metformin"),
        },
    )
    if event.selection.rows:
        st.session_state.pid = ids[event.selection.rows[0]]
        st.success(f"Opened patient {st.session_state.pid[:8]} — see the **Patient** tab.")
    st.caption("★ = demo patient. Select a row to open that patient.")

# ---------- patient page ----------
with tab_patient:
    pid = st.selectbox("Patient", ids, index=ids.index(st.session_state.pid),
                       format_func=lambda p: f"{p[:8]}{'  ★ demo' if p == service.DEMO_PATIENT else ''}")
    st.session_state.pid = pid
    d = get(f"/patients/{pid}", str(at))
    ehr, now = d["ehr"], d["now"]

    # EHR summary card + current status
    left, right = st.columns([3, 2])
    with left:
        with st.container(border=True):
            dx = "Type 2 diabetes" if ehr["t2d"] else "Prediabetes"
            meds = "metformin" if ehr["on_metformin"] else "no glucose-lowering meds"
            st.markdown(f"#### {dx} · {ehr['age']} y {'male' if ehr['sex'] == 'M' else 'female'} · {meds}")
            m = st.columns(4)
            m[0].metric("HbA1c", f"{ehr['hba1c']:.1f} %")
            m[1].metric("BMI", f"{ehr['bmi']:.1f}")
            m[2].metric("Hypertension", "Yes" if ehr["hypertension"] else "No")
            m[3].metric("Last night's sleep", f"{d['last_night_sleep_hours']:.1f} h")
            lip = ", ".join(f"{k.upper() if k != 'total_chol' else 'Total chol'} {ehr[k]:.0f}"
                            for k in ["total_chol", "ldl", "hdl", "triglycerides"] if ehr.get(k) is not None)
            st.caption(f"Lipids (mg/dL): {lip or 'not recorded'} · Patient {pid[:8]} · synthetic EHR (Synthea)")
    with right:
        with st.container(border=True):
            color = RISK_COLORS[now["risk_level"]]
            st.markdown(f"<div style='font-size:0.9rem;color:gray'>Risk of > 180 mg/dL in next 2 h</div>"
                        f"<div style='font-size:2.6rem;font-weight:700;color:{color}'>{now['risk']:.0%} "
                        f"<span style='font-size:1rem'>{now['risk_level'].upper()}</span></div>",
                        unsafe_allow_html=True)
            st.markdown(f"Glucose now **{now['glucose_now']:.0f} mg/dL** · alert threshold {now['alert_threshold']:.0%}")
            st.markdown("**Why (top 3 reasons):**")
            for r in now["reasons"]:
                arrow = "🔺" if r["direction"] == "raises risk" else "🔻"
                st.markdown(f"{arrow} {r['text']} — *{r['direction']}*")

    # What-if controls (above the chart so the overlay is visible right away)
    meal = d["meal_in_window"]
    with st.expander("🧪 What-if simulation", expanded=True):
        with st.form("whatif"):
            c = st.columns(3)
            base_carbs = meal["carbs"] if meal else 0
            carbs = c[0].slider(f"Meal carbs (g){' · meal at ' + meal['time'][11:16] if meal else ' · adds a meal now'}",
                                0, 160, int(base_carbs), step=5)
            walk = c[1].slider("Post-meal walk (min, starts 15 min after meal)", 0, 60, 0, step=5)
            sleep = c[2].slider("Last night's sleep (h)", 3.0, 9.0, round(d["last_night_sleep_hours"] * 2) / 2, step=0.5)
            run = st.form_submit_button("Run twin ▶", type="primary")
        key = f"wi_{pid}_{at}"
        if run:
            payload = {"patient_id": pid, "at": str(at),
                       "meal_carbs": None if carbs == base_carbs else carbs,
                       "post_meal_walk_min": walk or None,
                       "sleep_hours": None if abs(sleep - d["last_night_sleep_hours"]) < 0.25 else sleep}
            with st.spinner("Re-simulating the next 2 hours…"):
                st.session_state[key] = post_whatif(tuple(payload.items()))
        wi = st.session_state.get(key)
        if wi:
            b, mo = wi["baseline"], wi["modified"]
            k = st.columns(4)
            k[0].metric("Spike probability (twin)", f"{mo['sim_spike_prob']:.0%}",
                        f"{(mo['sim_spike_prob'] - b['sim_spike_prob']) * 100:+.0f} pts", delta_color="inverse")
            k[1].metric("Peak glucose", f"{mo['peak_glucose']:.0f} mg/dL",
                        f"{mo['peak_glucose'] - b['peak_glucose']:+.0f}", delta_color="inverse")
            k[2].metric("Minutes > 180", mo["minutes_above_180"],
                        f"{mo['minutes_above_180'] - b['minutes_above_180']:+d}", delta_color="inverse")
            k[3].metric("ML risk, peak over 2 h", f"{mo['model_risk_peak']:.0%}",
                        f"{(mo['model_risk_peak'] - b['model_risk_peak']) * 100:+.0f} pts", delta_color="inverse")
            st.caption("Spike probability: share of twin re-simulations (future noise re-drawn) exceeding 180 mg/dL "
                       "in the next 2 h. ML risk: LightGBM prediction as the simulated data arrives.")

    # 24 h CGM chart with predicted risk band
    s = pd.DataFrame(d["series"]).rename(columns={"timestamps": "timestamp"})
    s["timestamp"] = pd.to_datetime(s["timestamp"])
    rs = pd.DataFrame(d["risk_series"]).rename(columns={"timestamps": "timestamp"})
    rs["timestamp"] = pd.to_datetime(rs["timestamp"])
    obs, proj = s[s["observed"]], s[s["timestamp"] >= at]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=obs["timestamp"], y=obs["glucose"], name="CGM (observed)",
                             line=dict(color="#1f4e8c", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=proj["timestamp"], y=proj["glucose"], name="Twin projection (as usual)",
                             line=dict(color="#1f4e8c", width=2, dash="dot")), row=1, col=1)
    if wi:
        fig.add_trace(go.Scatter(x=pd.to_datetime(wi["modified"]["timestamps"]), y=wi["modified"]["glucose"],
                                 name="What-if", line=dict(color="#2ca02c", width=3)), row=1, col=1)
    meals = s[s["carbs"] > 0]
    fig.add_trace(go.Scatter(x=meals["timestamp"], y=meals["glucose"], mode="markers+text", name="Meal",
                             text=[f"{c} g" for c in meals["carbs"]], textposition="bottom center",
                             marker=dict(symbol="triangle-down", size=12, color="#e08a00")), row=1, col=1)
    # Shapes after traces: plotly skips shapes on subplots that are still empty
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.06, line_width=0, row=1, col=1)
    fig.add_hline(y=180, line_dash="dash", line_color="#d62728", row=1, col=1,
                  annotation_text="180 mg/dL", annotation_position="top left")
    fig.add_vrect(x0=at, x1=at + pd.Timedelta(hours=2), fillcolor=RISK_COLORS[now["risk_level"]], opacity=0.10,
                  line_width=0, row=1, col=1, annotation_text=f"next 2 h · risk {now['risk']:.0%}",
                  annotation_position="top right")

    fig.add_trace(go.Scatter(x=rs["timestamp"], y=rs["risk"], name="Predicted risk", fill="tozeroy",
                             line=dict(color="#d62728", width=1), fillcolor="rgba(214,39,40,0.25)"), row=2, col=1)
    if wi:
        fig.add_trace(go.Scatter(x=pd.to_datetime(wi["modified"]["timestamps"]), y=wi["modified"]["model_risk"],
                                 name="What-if risk", line=dict(color="#2ca02c", width=2)), row=2, col=1)
    fig.add_hline(y=now["alert_threshold"], line_dash="dot", line_color="gray", row=2, col=1)
    for r in (1, 2):
        fig.add_vline(x=at, line_color="black", line_width=1, row=r, col=1)
    fig.update_yaxes(title_text="Glucose (mg/dL)", range=[40, max(260, s["glucose"].max() + 20)], row=1, col=1)
    fig.update_yaxes(title_text="Risk", range=[0, 1], tickformat=".0%", row=2, col=1)
    fig.update_layout(height=580, margin=dict(l=70, r=20, t=50, b=20), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Last 22 h observed + next 2 h twin projection. Bottom: model risk of exceeding 180 mg/dL within "
               "2 h at each moment; dotted line = alert threshold (80% precision). Synthetic data.")
