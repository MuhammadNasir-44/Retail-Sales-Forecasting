"""Unit tests for the SQL analysis layer.

Run with:  pytest
"""

import sqlite3

import pandas as pd
import pytest

from sql_analysis import load_queries, run


@pytest.fixture
def conn():
    """A tiny in-memory retail_sales table: two full years of simple data."""
    c = sqlite3.connect(":memory:")
    dates = pd.date_range("2020-01-01", periods=24, freq="MS").strftime("%Y-%m-%d")
    # Year 2020 sums to 1200, year 2021 to 2400 (a clean +100% for testing).
    sales = [100] * 12 + [200] * 12
    pd.DataFrame({"date": dates, "sales": sales}).to_sql(
        "retail_sales", c, index=False)
    return c


def test_load_queries_has_expected_names():
    """All four named queries are parsed out of queries.sql."""
    q = load_queries()
    assert {"yearly_totals", "seasonal_index", "top_months", "yoy_growth"} <= set(q)
    assert "SELECT" in q["yearly_totals"].upper()  # it's a query (may lead with a comment)


def test_yearly_totals(conn):
    """Yearly totals aggregate correctly (values are in $ billions)."""
    q = load_queries()
    out = run(conn, q["yearly_totals"]).set_index("year")
    assert out.loc["2020", "total_sales_billions"] == pytest.approx(1.2)
    assert out.loc["2021", "total_sales_billions"] == pytest.approx(2.4)
    assert out.loc["2020", "months"] == 12


def test_yoy_growth_computes_percentage(conn):
    """2021 is double 2020 → +100% year-over-year."""
    q = load_queries()
    out = run(conn, q["yoy_growth"]).set_index("year")
    assert out.loc["2021", "yoy_growth_pct"] == pytest.approx(100.0)


def test_yoy_growth_excludes_partial_years():
    """A partial final year must not appear in the YoY output."""
    c = sqlite3.connect(":memory:")
    dates = (list(pd.date_range("2020-01-01", periods=12, freq="MS")
                  .strftime("%Y-%m-%d"))
             + list(pd.date_range("2021-01-01", periods=3, freq="MS")
                    .strftime("%Y-%m-%d")))          # 2021 is only 3 months
    pd.DataFrame({"date": dates, "sales": [100] * 15}).to_sql(
        "retail_sales", c, index=False)
    out = run(c, load_queries()["yoy_growth"])
    assert "2021" not in set(out["year"])            # partial year dropped
