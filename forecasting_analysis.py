"""
US Retail Sales Forecasting — time-series analysis
==================================================

End-to-end time-series project: forecast monthly US retail & food-services
sales, so a business could plan inventory, staffing, and cash flow ahead of
demand.

Pipeline
--------
1. Load the monthly series (FRED: RSXFSN, not seasonally adjusted)
2. Explore — trend and the strong yearly seasonality (holiday peaks)
3. Decompose into trend / seasonal / residual
4. Split into train / test (hold out the last 24 months)
5. Model — a seasonal-naive baseline vs. Holt-Winters exponential smoothing
6. Evaluate on the held-out test set (MAE, RMSE, MAPE)
7. Refit on all data and forecast the next 12 months (with a confidence band)

Run:  python forecasting_analysis.py
Prints metrics to the console, saves charts to ./images/, and writes the
forecast to forecast.csv (used by the dashboard).

Author: Muhammad Nasiruddin
Dataset: FRED "Advance Retail Sales: Retail and Food Services" (RSXFSN), public.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_PATH = Path(__file__).parent / "data" / "us_retail_sales.csv"
IMG_DIR = Path(__file__).parent / "images"
BASE = Path(__file__).parent
INK, ACCENT, GREY, BAND = "#1d3557", "#2a9d8f", "#adb5bd", "#2a9d8f"

TEST_MONTHS = 24     # hold out the last two years to test on
FORECAST_MONTHS = 12  # how far ahead to forecast after refitting


def load() -> pd.Series:
    """Load the monthly retail-sales series indexed by date."""
    df = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "RSXFSN": "sales"})
    s = df.set_index("date")["sales"].asfreq("MS")  # month-start frequency
    print(f"Series: {len(s)} months, {s.index.min():%Y-%m} to {s.index.max():%Y-%m}")
    print(f"Latest month: ${s.iloc[-1]:,.0f}M   (values are $ millions)\n")
    return s


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (%)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def explore(s: pd.Series) -> None:
    """Plot the full history and a seasonal decomposition."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(s.index, s.values / 1000, color=INK, linewidth=1)
    ax.set_title("US retail & food-services sales, 1992–present")
    ax.set_ylabel("$ billions / month")
    ax.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}B")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "history.png", dpi=110)
    plt.close(fig)

    # Multiplicative decomposition — seasonality grows with the trend.
    result = seasonal_decompose(s, model="multiplicative", period=12)
    fig = result.plot()
    fig.set_size_inches(8, 6)
    fig.suptitle("Seasonal decomposition (trend · seasonal · residual)", y=1.01)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "decomposition.png", dpi=110)
    plt.close(fig)

    # Average seasonal shape by calendar month.
    monthly = s.groupby(s.index.month).mean()
    monthly = monthly / monthly.mean() * 100
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(1, 13), monthly.values, color=ACCENT)
    ax.axhline(100, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("Index (100 = average month)")
    ax.set_title("Seasonal pattern — December is the peak")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "seasonality.png", dpi=110)
    plt.close(fig)

    peak, trough = monthly.idxmax(), monthly.idxmin()
    names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print(f"Peak month: {names[peak]} ({monthly[peak]:.0f}% of average)")
    print(f"Low month:  {names[trough]} ({monthly[trough]:.0f}% of average)\n")


def analyze_growth(s: pd.Series) -> None:
    """Deeper look: year-over-year growth and the rolling 12-month total.

    Seasonality tells you *within-year* shape; YoY growth strips the season out
    to show the underlying momentum of the business.
    """
    yoy = s.pct_change(12) * 100  # same month, one year earlier
    rolling_year = s.rolling(12).sum()

    recent = yoy.dropna().iloc[-12:]
    print("Year-over-year growth, last 12 months:")
    print(f"  average: {recent.mean():+.1f}%   latest: {yoy.iloc[-1]:+.1f}% "
          f"({s.index[-1]:%Y-%m})\n")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(rolling_year.index, rolling_year.values / 1000, color=INK, linewidth=1)
    ax1.set_title("Rolling 12-month total sales (trend, seasonality removed)")
    ax1.set_ylabel("$ billions / yr")
    ax1.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}B")

    ax2.plot(yoy.index, yoy.values, color=ACCENT, linewidth=0.9)
    ax2.axhline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax2.set_title("Year-over-year growth rate")
    ax2.set_ylabel("% vs. year ago")
    ax2.yaxis.set_major_formatter(lambda x, _: f"{x:,.0f}%")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "growth.png", dpi=110)
    plt.close(fig)


def seasonal_naive(train: pd.Series, steps: int) -> np.ndarray:
    """Baseline: predict each month = the same month one year earlier."""
    last_year = train.iloc[-12:].values
    return np.array([last_year[i % 12] for i in range(steps)])


def evaluate_models(s: pd.Series) -> None:
    """Backtest baseline vs. Holt-Winters on a held-out test set."""
    train, test = s.iloc[:-TEST_MONTHS], s.iloc[-TEST_MONTHS:]
    print(f"Train: {len(train)} months   Test: {len(test)} months "
          f"({test.index.min():%Y-%m} to {test.index.max():%Y-%m})\n")

    # 1) Seasonal-naive baseline.
    base_pred = seasonal_naive(train, len(test))

    # 2) Holt-Winters: additive trend + multiplicative seasonality.
    hw = ExponentialSmoothing(
        train, trend="add", seasonal="mul", seasonal_periods=12
    ).fit()
    hw_pred = hw.forecast(len(test)).values

    # 3) SARIMA: a statistical model with non-seasonal + seasonal terms.
    #    order (p,d,q) = (1,1,1), seasonal (P,D,Q,s) = (1,1,1,12).
    sarima = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    sarima_pred = np.asarray(sarima.forecast(len(test)))

    print("Backtest on the held-out last 24 months:")
    print(f"{'Model':<26}{'MAE ($M)':>12}{'RMSE ($M)':>12}{'MAPE (%)':>11}")
    for name, pred in [("Seasonal-naive baseline", base_pred),
                       ("Holt-Winters smoothing", hw_pred),
                       ("SARIMA (1,1,1)(1,1,1)12", sarima_pred)]:
        mae = mean_absolute_error(test, pred)
        rmse = np.sqrt(mean_squared_error(test, pred))
        print(f"{name:<26}{mae:>12,.0f}{rmse:>12,.0f}{mape(test, pred):>11.2f}")
    print()

    # Plot actual vs. all models on the test window.
    fig, ax = plt.subplots(figsize=(8, 4))
    ctx = s.iloc[-TEST_MONTHS - 24:]
    ax.plot(ctx.index, ctx.values / 1000, color=INK, linewidth=1.2, label="Actual")
    ax.plot(test.index, base_pred / 1000, "--", color=GREY, label="Seasonal-naive")
    ax.plot(test.index, hw_pred / 1000, color=ACCENT, linewidth=1.6,
            label="Holt-Winters")
    ax.plot(test.index, sarima_pred / 1000, color="#e76f51", linewidth=1.6,
            label="SARIMA")
    ax.set_title("Backtest — forecast vs. actual (held-out 24 months)")
    ax.set_ylabel("$ billions / month")
    ax.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}B")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "backtest.png", dpi=110)
    plt.close(fig)


def forecast_future(s: pd.Series) -> pd.DataFrame:
    """Refit the winning model (SARIMA) on ALL data and forecast 12 months.

    SARIMA won the backtest, so it's used for the live forecast. Its state-space
    form also gives proper confidence intervals (no residual-spread hack needed).
    """
    model = SARIMAX(
        s, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    fc_res = model.get_forecast(FORECAST_MONTHS)
    fc = fc_res.predicted_mean
    ci = fc_res.conf_int(alpha=0.05)  # 95% interval
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values

    out = pd.DataFrame(
        {"date": fc.index, "forecast": fc.values, "lower": lower, "upper": upper}
    )
    out.to_csv(BASE / "forecast.csv", index=False)

    print(f"Next {FORECAST_MONTHS}-month forecast (SARIMA):")
    for _, r in out.iterrows():
        print(f"  {r['date']:%Y-%m}   ${r['forecast']:,.0f}M "
              f"(95% CI ${r['lower']:,.0f} – ${r['upper']:,.0f})")
    total = out["forecast"].sum()
    print(f"\nForecast total for the next year: ${total:,.0f}M "
          f"(${total/1000:,.1f}B)\n")

    # Plot recent history + forecast with band.
    fig, ax = plt.subplots(figsize=(8, 4))
    hist = s.iloc[-36:]
    ax.plot(hist.index, hist.values / 1000, color=INK, linewidth=1.2, label="History")
    ax.plot(out["date"], out["forecast"] / 1000, color=ACCENT, linewidth=1.8,
            label="Forecast")
    ax.fill_between(out["date"], lower / 1000, upper / 1000,
                    color=BAND, alpha=0.18, label="95% band")
    ax.set_title("12-month retail-sales forecast (SARIMA)")
    ax.set_ylabel("$ billions / month")
    ax.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}B")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "forecast.png", dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    s = load()
    explore(s)
    analyze_growth(s)
    evaluate_models(s)
    forecast_future(s)
    print("Saved charts to images/ and forecast.csv")


if __name__ == "__main__":
    main()
