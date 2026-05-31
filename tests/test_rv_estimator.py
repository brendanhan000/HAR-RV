"""
Phase 1 unit tests for rv_estimator.py.

Tests:
  1. Known-input RV: synthetic constant returns produce exact RV.
  2. No NaN leakage: cross-day log returns are NaN, not leaked into RV.
  3. Trading-day alignment: RV index contains only dates present in intraday data.
  4. Min-bars filter: days below threshold are dropped.
  5. Yang-Zhang: positive, finite values on known OHLCV input.
  6. Yang-Zhang NaN on first row (no previous close).
  7. Annualization factor applied correctly.
  8. rv_to_vol = sqrt(RV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rv_estimator import compute_rv, compute_yang_zhang, rv_to_vol


# ---------------------------------------------------------------------------
# Helpers: build synthetic DataFrames
# ---------------------------------------------------------------------------


def _make_intraday(
    dates: list[str],
    ret_per_bar: float = 0.001,
    bars_per_day: int = 78,  # 6.5 hours * 12 5-min bars/hour
) -> pd.DataFrame:
    """
    Synthetic 5-min intraday bars.
    Each bar within a day has exactly `ret_per_bar` log return.
    Close price is cumulative exp of returns, reset at open of each day.
    """
    rows = []
    for d in dates:
        base_ts = pd.Timestamp(d, tz="America/New_York") + pd.Timedelta(hours=9, minutes=30)
        price = 100.0
        for i in range(bars_per_day):
            ts = base_ts + pd.Timedelta(minutes=5 * i)
            close = price * np.exp(ret_per_bar)
            rows.append(
                {"timestamp": ts, "open": price, "high": close, "low": price, "close": close, "volume": 1000}
            )
            price = close
    return pd.DataFrame(rows)


def _make_config(
    annualize: bool = False,
    min_bars: int = 10,
    estimator: str = "intraday",
):
    """Minimal Config-like object for testing."""
    from types import SimpleNamespace

    rv_cfg = SimpleNamespace(
        annualize=annualize,
        trading_days_per_year=252,
        min_bars_per_day=min_bars,
        estimator=estimator,
    )
    return SimpleNamespace(rv=rv_cfg)


def _make_daily_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-04", periods=n, freq="B")
    close = 100 * np.exp(rng.normal(0, 0.01, n).cumsum())
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(close, open_) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(close, open_) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close}
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeRV:
    def test_known_input_exact(self):
        """RV for 2 days of constant returns should equal bars * ret^2."""
        dates = ["2020-01-02", "2020-01-03"]
        ret = 0.001
        bars = 78
        df = _make_intraday(dates, ret_per_bar=ret, bars_per_day=bars)
        cfg = _make_config(annualize=False, min_bars=10)
        rv = compute_rv(df, cfg)
        # First bar of each day has NaN return (no cross-day carry) → bars-1 contribute
        expected = (bars - 1) * ret**2
        np.testing.assert_allclose(rv.values, expected, rtol=1e-9)

    def test_no_nan_in_output(self):
        """RV series should contain no NaN values after min-bar filter."""
        dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
        df = _make_intraday(dates, bars_per_day=78)
        cfg = _make_config(annualize=False, min_bars=10)
        rv = compute_rv(df, cfg)
        assert rv.isna().sum() == 0, "RV contains unexpected NaN values"

    def test_cross_day_log_return_excluded(self):
        """The overnight close-to-open return must not enter the sum of squares."""
        # One day with 1 bar, close=110. Next day with 1 bar, open != 110.
        # If cross-day returns leaked, RV of day 2 would include (log(200/110))^2.
        rows = [
            {
                "timestamp": pd.Timestamp("2020-01-02 09:30", tz="America/New_York"),
                "open": 100, "high": 110, "low": 100, "close": 110, "volume": 1,
            },
            {
                "timestamp": pd.Timestamp("2020-01-03 09:30", tz="America/New_York"),
                "open": 200, "high": 210, "low": 200, "close": 210, "volume": 1,
            },
        ]
        df = pd.DataFrame(rows)
        cfg = _make_config(annualize=False, min_bars=1)
        rv = compute_rv(df, cfg)
        # Each day has 1 bar; that bar's return is NaN (first bar of day).
        # Sum of squares of non-NaN returns = 0 for both days.
        # Cross-day return (log(200/110)) must NOT appear in either day's RV.
        assert pd.Timestamp("2020-01-02") in rv.index, "Day 1 should not be dropped"
        assert pd.Timestamp("2020-01-03") in rv.index, "Day 2 should not be dropped"
        assert rv.loc[pd.Timestamp("2020-01-02")].item() == pytest.approx(0.0)
        assert rv.loc[pd.Timestamp("2020-01-03")].item() == pytest.approx(0.0)

    def test_trading_day_alignment(self):
        """RV index must only contain calendar dates present in intraday data."""
        dates = ["2020-01-02", "2020-01-06", "2020-01-09"]  # skip 3,7,8
        df = _make_intraday(dates, bars_per_day=78)
        cfg = _make_config(annualize=False)
        rv = compute_rv(df, cfg)
        expected_dates = {pd.Timestamp(d) for d in dates}
        assert set(rv.index) == expected_dates

    def test_min_bars_filter_drops_thin_days(self):
        """Days with fewer bars than min_bars_per_day must be dropped."""
        # Day 1: 78 bars; Day 2: 5 bars (half-day sim)
        rows_d1 = _make_intraday(["2020-01-02"], bars_per_day=78)
        rows_d2 = _make_intraday(["2020-01-03"], bars_per_day=5)
        df = pd.concat([rows_d1, rows_d2], ignore_index=True)
        cfg = _make_config(annualize=False, min_bars=30)
        rv = compute_rv(df, cfg)
        assert pd.Timestamp("2020-01-02") in rv.index
        assert pd.Timestamp("2020-01-03") not in rv.index

    def test_annualization_factor(self):
        """Annualized RV should equal daily RV * 252."""
        dates = ["2020-01-02", "2020-01-03"]
        df = _make_intraday(dates, ret_per_bar=0.001, bars_per_day=78)
        cfg_raw = _make_config(annualize=False)
        cfg_ann = _make_config(annualize=True)
        rv_raw = compute_rv(df, cfg_raw)
        rv_ann = compute_rv(df, cfg_ann)
        np.testing.assert_allclose(rv_ann.values, rv_raw.values * 252, rtol=1e-9)


class TestYangZhang:
    def test_positive_and_finite(self):
        df = _make_daily_ohlcv(n=100)
        cfg = _make_config(annualize=False)
        yz = compute_yang_zhang(df, cfg)
        valid = yz.dropna()
        assert (valid > 0).all(), "YZ variance must be positive"
        assert np.isfinite(valid.values).all()

    def test_first_row_is_nan(self):
        df = _make_daily_ohlcv(n=50)
        cfg = _make_config(annualize=False)
        yz = compute_yang_zhang(df, cfg)
        assert np.isnan(yz.iloc[0]), "First YZ row must be NaN (no previous close)"

    def test_annualization(self):
        df = _make_daily_ohlcv(n=50)
        cfg_raw = _make_config(annualize=False)
        cfg_ann = _make_config(annualize=True)
        yz_raw = compute_yang_zhang(df, cfg_raw).dropna()
        yz_ann = compute_yang_zhang(df, cfg_ann).dropna()
        np.testing.assert_allclose(yz_ann.values, yz_raw.values * 252, rtol=1e-9)

    def test_reasonable_magnitude(self):
        """Annualized YZ vol should be in plausible equity range (5%–200%)."""
        df = _make_daily_ohlcv(n=500)
        cfg = _make_config(annualize=True)
        yz = compute_yang_zhang(df, cfg).dropna()
        vol = np.sqrt(yz)
        assert (vol > 0.005).all() and (vol < 2.0).all(), (
            f"YZ vol outside plausible range: min={vol.min():.3f} max={vol.max():.3f}"
        )


class TestRVToVol:
    def test_sqrt_relationship(self):
        rv = pd.Series([0.04, 0.09, 0.16], index=pd.date_range("2020-01-01", periods=3))
        vol = rv_to_vol(rv)
        np.testing.assert_allclose(vol.values, [0.2, 0.3, 0.4])

    def test_name(self):
        rv = pd.Series([0.04], name="RV")
        assert rv_to_vol(rv).name == "RVol"
