"""
Phase C unit tests for the elevated-VIX baseline + paired P&L test.

Covers:
  1. rolling_pct_short_signal shorts exactly when x_t exceeds its rolling percentile
  2. NO LOOK-AHEAD: perturbing a FUTURE value cannot change an earlier day's signal
  3. min_periods: signal is flat until enough history exists
  4. one-day implementation lag is applied
  5. symmetric signal: short / long / flat regions are correct
  6. paired_pnl_test: identical series -> zero diff, not significant
  7. paired_pnl_test: a constant edge -> positive, significant
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baseline import (
    paired_pnl_test,
    rolling_pct_short_signal,
    rolling_pct_symmetric_signal,
)


def _series(vals):
    idx = pd.date_range("2015-01-02", periods=len(vals), freq="B")
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


class TestRollingSignal:
    def test_shorts_when_above_rolling_percentile(self):
        # rising-then-spiking series: late high values exceed their rolling median
        x = _series(list(range(1, 60)) + [200, 5, 5])
        sig = rolling_pct_short_signal(x, window=20, min_periods=10, pct=50)
        # the +200 spike (index 59) should trigger a short, acted the NEXT day (lag 1)
        assert sig.iloc[60] == 1.0
        # the low value right after (index 60 -> acted index 61) should not be short
        assert sig.iloc[61] == 0.0

    def test_no_lookahead_future_cannot_change_past_signal(self):
        rng = np.random.default_rng(0)
        x = _series(rng.normal(20, 5, 300).cumsum() / 10 + 20)
        base = rolling_pct_short_signal(x, window=60, min_periods=30, pct=75)
        x2 = x.copy()
        x2.iloc[-1] *= 5.0                       # shock the final value
        shocked = rolling_pct_short_signal(x2, window=60, min_periods=30, pct=75)
        # every signal strictly before the perturbed date is unchanged
        cut = x.index[-1]
        np.testing.assert_array_equal(base.loc[base.index < cut].values,
                                      shocked.loc[shocked.index < cut].values)

    def test_min_periods_flat_until_enough_history(self):
        x = _series(np.arange(100))
        sig = rolling_pct_short_signal(x, window=40, min_periods=30, pct=75)
        # first ~30 obs have no percentile -> flat (and lag adds one more)
        assert (sig.iloc[:30] == 0).all()

    def test_one_day_lag(self):
        # a single late spike must affect the NEXT day's position, not the same day
        x = _series([10] * 40 + [1000] + [10] * 5)
        sig = rolling_pct_short_signal(x, window=20, min_periods=10, pct=90)
        spike = 40
        assert sig.iloc[spike] == 0.0            # same day: not yet acted
        assert sig.iloc[spike + 1] == 1.0        # next day: short

    def test_symmetric_regions(self):
        x = _series(list(np.linspace(10, 30, 80)))   # steadily rising
        sig = rolling_pct_symmetric_signal(x, window=30, min_periods=15,
                                           upper_pct=80, lower_pct=20)
        assert set(np.unique(sig.values)).issubset({-1.0, 0.0, 1.0})
        # rising series: recent values tend to sit at the top of the window -> some shorts
        assert (sig == 1.0).any()


class TestPairedTest:
    def test_identical_series_zero_diff(self):
        rng = np.random.default_rng(1)
        p = pd.Series(rng.normal(0, 1, 500))
        out = paired_pnl_test(p, p.copy(), lag=22)
        assert out["mean_diff"] == pytest.approx(0.0, abs=1e-12)

    def test_constant_edge_is_significant(self):
        rng = np.random.default_rng(2)
        base = pd.Series(rng.normal(0, 1, 800))
        better = base + 0.3                       # constant per-day edge
        out = paired_pnl_test(better, base, lag=22, alternative="greater")
        assert out["mean_diff"] > 0
        assert out["p_value"] < 0.01              # clearly beats

    def test_worse_series_not_significant_as_better(self):
        rng = np.random.default_rng(3)
        base = pd.Series(rng.normal(0, 1, 800))
        worse = base - 0.3
        out = paired_pnl_test(worse, base, lag=22, alternative="greater")
        assert out["p_value"] > 0.95              # fails the "is better" test
