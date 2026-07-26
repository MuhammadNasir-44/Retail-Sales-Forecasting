"""Unit tests for the forecasting helpers.

Run with:  pytest
"""

import numpy as np
import pandas as pd
import pytest

from forecasting_analysis import load, mape, seasonal_naive


def test_mape_zero_when_perfect():
    """A perfect forecast has 0% error."""
    y = np.array([100, 200, 300], dtype=float)
    assert mape(y, y) == pytest.approx(0.0)


def test_mape_known_value():
    """10% off everywhere → MAPE of 10%."""
    y = np.array([100.0, 200.0])
    pred = np.array([110.0, 220.0])
    assert mape(y, pred) == pytest.approx(10.0)


def test_seasonal_naive_repeats_last_year():
    """The baseline repeats the final 12 months of training data."""
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    train = pd.Series(range(24), index=idx, dtype=float)
    pred = seasonal_naive(train, 12)
    # Last 12 training values are 12..23; the forecast should equal them.
    assert list(pred) == list(range(12, 24))


def test_seasonal_naive_wraps_beyond_12():
    """Forecasting >12 steps wraps around to reuse the seasonal pattern."""
    idx = pd.date_range("2000-01-01", periods=12, freq="MS")
    train = pd.Series(range(12), index=idx, dtype=float)
    pred = seasonal_naive(train, 15)
    assert len(pred) == 15
    assert list(pred[:12]) == list(range(12))
    assert list(pred[12:]) == [0, 1, 2]  # wrapped


def test_load_series_shape():
    """The real dataset loads as a monthly (MS) float series with no gaps."""
    s = load()
    assert isinstance(s, pd.Series)
    assert s.index.freqstr == "MS"
    assert len(s) > 300           # 30+ years of monthly data
    assert s.notna().all()        # no missing months
    assert (s > 0).all()          # sales are positive
