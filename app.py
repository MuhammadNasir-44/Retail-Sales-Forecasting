"""
US Retail Sales — Forecast Dashboard (Streamlit)
================================================

An interactive dashboard built entirely in Python. It loads the monthly retail
series, backtests three forecasting models, and lets the user choose a forecast
horizon — all rendered with Streamlit + Plotly.

Run locally:
    streamlit run app.py

Deploy (free): push to GitHub and connect the repo at share.streamlit.io.

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = Path(__file__).parent / "data" / "us_retail_sales.csv"
INK, ACCENT, WARN, GREY = "#1d3557", "#2a9d8f", "#e76f51", "#adb5bd"
MONTHS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="US Retail Sales Forecast",
                   page_icon="📈", layout="wide")


# ----------------------------- data & models -----------------------------
@st.cache_data
def load_series() -> pd.Series:
    df = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "RSXFSN": "sales"})
    return df.set_index("date")["sales"].asfreq("MS")


def _mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


@st.cache_data
def backtest(test_months: int) -> pd.DataFrame:
    """Backtest the three models on a held-out window; return a metrics table."""
    s = load_series()
    train, test = s.iloc[:-test_months], s.iloc[-test_months:]

    last_year = train.iloc[-12:].values
    naive = np.array([last_year[i % 12] for i in range(len(test))])

    hw = ExponentialSmoothing(train, trend="add", seasonal="mul",
                              seasonal_periods=12).fit()
    hw_pred = hw.forecast(len(test)).values

    sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False,
                     enforce_invertibility=False).fit(disp=False)
    sarima_pred = np.asarray(sarima.forecast(len(test)))

    rows = []
    for name, pred in [("Seasonal-naive", naive),
                       ("Holt-Winters", hw_pred),
                       ("SARIMA", sarima_pred)]:
        rows.append({
            "Model": name,
            "MAE ($M)": round(mean_absolute_error(test, pred)),
            "RMSE ($M)": round(np.sqrt(mean_squared_error(test, pred))),
            "MAPE (%)": round(_mape(test, pred), 2),
        })
    return pd.DataFrame(rows)


@st.cache_data
def forecast(horizon: int) -> pd.DataFrame:
    """Fit SARIMA on all data and forecast `horizon` months with a 95% band."""
    s = load_series()
    model = SARIMAX(s, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
    res = model.get_forecast(horizon)
    ci = res.conf_int(alpha=0.05)
    return pd.DataFrame({
        "date": res.predicted_mean.index,
        "forecast": res.predicted_mean.values,
        "lower": ci.iloc[:, 0].values,
        "upper": ci.iloc[:, 1].values,
    })


# ----------------------------- UI -----------------------------
s = load_series()

st.title("📈 US Retail Sales — Forecast Dashboard")
st.caption("Monthly US retail & food-services sales (FRED: RSXFSN, 1992–2026). "
           "Built in Python with statsmodels, Plotly, and Streamlit.")

with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Forecast horizon (months)", 6, 24, 12)
    test_months = st.slider("Backtest window (months)", 12, 36, 24, step=6)
    st.markdown("---")
    st.markdown("**Model:** SARIMA (1,1,1)(1,1,1)₁₂ — the backtest winner.")

metrics = backtest(test_months)
fc = forecast(horizon)
best = metrics.loc[metrics["MAPE (%)"].idxmin()]

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Forecast · next {horizon} mo", f"${fc['forecast'].sum()/1000:,.0f}B")
c2.metric("Best model (MAPE)", f"{best['MAPE (%)']:.2f}%", best["Model"])
seasonal = s.groupby(s.index.month).mean()
seasonal = seasonal / seasonal.mean() * 100
c3.metric("Peak month", MONTHS[int(seasonal.idxmax())],
          f"{seasonal.max():.0f}% of avg")
c4.metric("Latest actual", f"${s.iloc[-1]/1000:,.0f}B", f"{s.index[-1]:%b %Y}")

# Main forecast chart
st.subheader("History & forecast")
hist = s.iloc[-36:]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(fc["date"]) + list(fc["date"][::-1]),
    y=list(fc["upper"]) + list(fc["lower"][::-1]),
    fill="toself", fillcolor="rgba(42,157,143,0.15)",
    line=dict(color="rgba(0,0,0,0)"), name="95% band", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual",
                         line=dict(color=INK, width=2)))
fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast",
                         line=dict(color=ACCENT, width=2.5)))
fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis_title="$ millions / month",
                  legend=dict(orientation="h", y=1.05),
                  hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, width='stretch')

col_a, col_b = st.columns([3, 2])

with col_a:
    st.subheader("Seasonal pattern")
    sfig = go.Figure(go.Bar(
        x=[MONTHS[m] for m in seasonal.index], y=seasonal.values,
        marker_color=[INK if v == seasonal.max() else ACCENT for v in seasonal.values]))
    sfig.add_hline(y=100, line_dash="dash", line_color=GREY)
    sfig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis_title="Index (100 = avg month)",
                       template="plotly_white")
    st.plotly_chart(sfig, width='stretch')

with col_b:
    st.subheader("Model backtest")
    st.dataframe(metrics, hide_index=True, width='stretch')
    st.caption("Lower is better. SARIMA wins on all three measures.")

st.subheader("Forecast detail")
show = fc.copy()
show["date"] = show["date"].dt.strftime("%b %Y")
show = show.rename(columns={"date": "Month", "forecast": "Forecast ($M)",
                            "lower": "Low ($M)", "upper": "High ($M)"})
for c in ["Forecast ($M)", "Low ($M)", "High ($M)"]:
    show[c] = show[c].round().astype(int)
st.dataframe(show, hide_index=True, width='stretch')

st.markdown("---")
st.caption("Source code: github.com/MuhammadNasir-44/Retail-Sales-Forecasting")
