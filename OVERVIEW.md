# Project Overview — US Retail Sales Forecasting

A complete, one-page tour of this project: what it does, how it was built, what it
found, and where everything lives. For the business-focused write-up, see
[`REPORT.md`](REPORT.md); for setup and commands, see [`README.md`](README.md).

---

## 1. What this project is

An end-to-end **time-series forecasting** project. It takes 30+ years of monthly
US retail-sales data and builds a model that predicts the next 12 months —
accurately enough for a business to plan inventory, staffing, and cash flow. It
also ships an interactive dashboard, a SQL analysis layer, tests, and a written
report, so it stands as a full data-science deliverable rather than just a script.

- **Type:** unsupervised/statistical time-series forecasting (regression family)
- **Language/stack:** Python + SQL (pandas, statsmodels, scikit-learn, SQLite,
  Streamlit, Plotly, pytest)
- **Author:** Muhammad Nasiruddin

## 2. The data

**FRED — "Advance Retail Sales: Retail & Food Services" (RSXFSN)**, monthly, *not*
seasonally adjusted (so the seasonality is visible and modelled explicitly).
**414 months, 1992–2026**, in US$ millions.

## 3. How it works (method)

1. **Explore** — plot the long-run trend and decompose the series into trend,
   seasonal, and residual components.
2. **Feature/seasonal analysis** — quantify the monthly seasonal pattern and the
   year-over-year growth (seasonality removed).
3. **Backtest three models** on a 24-month hold-out the models never see in
   training:
   - Seasonal-naive baseline (this month = same month last year)
   - Holt-Winters exponential smoothing
   - SARIMA (1,1,1)(1,1,1)₁₂
4. **Evaluate** with MAE, RMSE, and MAPE, and pick the winner.
5. **Forecast** — refit the winner on all data and project 12 months with a 95%
   confidence interval.
6. **Cross-check with SQL** — reproduce the seasonal/growth findings in SQL.

## 4. Key results

| Model | MAPE (error) |
|-------|:------------:|
| Seasonal-naive baseline | 5.26% |
| Holt-Winters smoothing | 4.22% |
| **SARIMA (winner)** | **2.71%** |

- **SARIMA won**, roughly halving the baseline's error.
- **Next-12-month forecast:** ~**$8.2 trillion** in total sales.
- **December peak:** ~$751B (about **116%** of an average month); **February** is
  the low point (~88%).
- **Momentum:** ~3–4% year-over-year growth recently, after a ~17% rebound in 2021.

## 5. What's in the repo (file map)

| File / folder | What it is |
|---------------|------------|
| `forecasting_analysis.py` | Main pipeline: load → explore → backtest → forecast; saves charts + `forecast.csv` |
| `app.py` | Interactive **Streamlit + Plotly** dashboard |
| `sql/queries.sql` | SQL analysis queries (yearly totals, seasonal index, top months, YoY growth) |
| `sql_analysis.py` | Runs the SQL against SQLite; saves results to `sql_output/` |
| `sql_output/*.csv` | The SQL query results, viewable directly on GitHub |
| `forecasting_walkthrough.ipynb` | Narrated Jupyter notebook (with a pandas-vs-SQL check) |
| `tests/` | Unit tests for the forecasting helpers and SQL layer (9 tests) |
| `images/` | Generated charts (history, decomposition, seasonality, backtest, forecast, growth) |
| `REPORT.md` | Executive business analysis report |
| `README.md` | Setup, commands, and results summary |
| `data/us_retail_sales.csv` | The source dataset (from FRED) |

## 6. How to run

```bash
pip install -r requirements.txt

python forecasting_analysis.py   # run the full analysis + charts
python sql_analysis.py           # run the SQL analysis
streamlit run app.py             # launch the interactive dashboard
pytest                           # run the unit tests
```

## 7. Skills this project demonstrates

Time-series forecasting (Holt-Winters, SARIMA) · model backtesting & evaluation ·
seasonal decomposition · SQL (aggregation + window functions) · interactive
dashboards (Streamlit/Plotly) · unit testing (pytest) · clear technical and
business communication.

---

*Part of my data-science portfolio, alongside a customer-segmentation project
(clustering) and a churn-prediction project (classification) — together covering
forecasting, unsupervised, and supervised machine learning.*
