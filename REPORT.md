# Retail Sales Forecasting — Business Analysis Report

**Prepared by:** Muhammad Nasiruddin
**Dataset:** FRED "Advance Retail Sales: Retail & Food Services" (RSXFSN),
monthly, not seasonally adjusted — 414 months, 1992–2026
**Question:** How much will US retail sell over the next 12 months, and when?

---

## Executive summary

Retail demand is seasonal and trend-driven, so planning inventory, staffing, and
cash flow by gut feel leads to costly over- or under-stocking. Using 30+ years of
monthly data, this project builds a forecasting model that learns both the
long-run trend and the repeating yearly pattern, and projects the year ahead with
a measured accuracy.

> **The model forecasts ~$8.2 trillion in US retail & food-services sales over
> the next 12 months, with the usual December peak (~$751B — about 116% of an
> average month). Its typical error is 2.7%.**

Three models were compared on data they had never seen. **SARIMA** was the clear
winner, roughly **halving** the error of a naive baseline — accurate enough for
real planning decisions, and honestly measured rather than assumed.

---

## Why this matters (the business case)

- **Inventory & supply chain** — knowing December will run ~16% above an average
  month (and February ~12% below) lets a retailer stock up and wind down at the
  right times, avoiding both stockouts and dead inventory.
- **Staffing** — seasonal hiring and shift planning can be scheduled to the
  forecast rather than reacting late.
- **Cash flow & budgeting** — a 12-month revenue projection with a confidence
  range feeds directly into financial planning.

## How the forecast was validated

Rather than trusting a single model, three were **backtested** — trained on all
but the last 24 months, then scored on how well they predicted that held-out
period (which they never saw during training):

| Model | MAE ($M) | RMSE ($M) | MAPE (error) |
|-------|:--------:|:---------:|:------------:|
| Seasonal-naive baseline | 33,829 | 39,007 | 5.26% |
| Holt-Winters smoothing | 26,911 | 29,823 | 4.22% |
| **SARIMA (1,1,1)(1,1,1)₁₂** | **17,514** | **20,857** | **2.71%** |

**SARIMA won on every measure**, cutting the baseline's error roughly in half.
A 2.71% average error means a typical monthly forecast lands within ~£3 in every
£100 of actual sales — reliable enough to plan against. SARIMA is therefore used
for the live forecast, with its built-in 95% confidence interval.

![Backtest](images/backtest.png)

## What the forecast says

![12-month forecast](images/forecast.png)

- **Next 12 months:** ~$8.2T in total sales, continuing the long-run upward trend.
- **December peak:** ~$751B — the single biggest month, driven by holiday
  shopping, and consistent every year in the data.
- **February trough:** the quietest month at ~88% of average.
- **Momentum:** stripping out seasonality, underlying growth has run ~3–4% a year
  recently (with a one-off ~17% rebound in 2021 after the pandemic dip).

![Seasonal pattern](images/seasonality.png)

## Recommendations

1. **Plan capacity to the seasonal curve, not the average.** Build inventory and
   staffing toward the Nov–Dec peak and trim through Jan–Feb.
2. **Use the confidence band, not just the point forecast.** Plan a base case on
   the central line and stress-test against the upper/lower bounds.
3. **Refresh monthly.** The model retrains in seconds as new data arrives, so the
   forecast should be regenerated each month to stay current.
4. **Watch the residuals for regime changes.** A sustained miss (as in 2020–21)
   is an early signal that conditions have shifted and plans need revisiting.

## Method note

- **Data:** monthly FRED RSXFSN, not seasonally adjusted (so seasonality is
  modelled explicitly rather than removed).
- **Models:** seasonal-naive baseline · Holt-Winters exponential smoothing ·
  SARIMA — compared by MAE, RMSE, and MAPE on a 24-month hold-out.
- **Deliverables:** the analysis script, an interactive Streamlit dashboard, a
  SQL analysis layer, unit tests, and a Jupyter notebook walkthrough — all in
  this repository.

## Tech

Python · pandas · statsmodels · scikit-learn · SQL (SQLite) · Streamlit · Plotly · pytest
