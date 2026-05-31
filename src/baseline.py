"""
Phase C: the benchmark that matters — "sell vol when VIX is elevated."

The Phase-3 verdict named a naive baseline the HAR-VRP model failed to beat: short
vol whenever VIX is high relative to its own recent range, flat otherwise. Phase C
builds that baseline, runs it through the SAME convex P&L + costs as the signal, and
tests whether HAR-VRP adds any RELIABLE INCREMENTAL edge.

NO LOOK-AHEAD: every threshold is a ROLLING percentile using only data <= t (an
expanding/rolling quantile), and signals are lagged one day before they trade. This is
applied to BOTH the VIX baseline and the VRP signal, so neither gets a full-sample
threshold advantage (the Phase-4 signal had used full-sample quantiles).

Public API:
  rolling_pct_short_signal(x, window, min_periods, pct)        -> {0,+1} short/flat
  rolling_pct_symmetric_signal(x, window, min_periods, up, lo) -> {-1,0,+1}
  paired_pnl_test(pnl1, pnl2, lag)                             -> HAC test of E[pnl1-pnl2]
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Point-in-time rolling-percentile signals
# ---------------------------------------------------------------------------


def _rolling_quantile(x: pd.Series, window: int, min_periods: int, q: float) -> pd.Series:
    """Rolling q-quantile using only data up to and including each date t."""
    return x.rolling(window=window, min_periods=min_periods).quantile(q)


def rolling_pct_short_signal(
    x: pd.Series,
    window: int,
    min_periods: int,
    pct: float,
) -> pd.Series:
    """
    Short-vol (+1) whenever x_t exceeds its rolling `pct`-th percentile (computed on
    data <= t); flat (0) otherwise. Returns a 1-day-lagged position series so the
    decision at t is acted on at t+1 (no look-ahead).
    """
    thr = _rolling_quantile(x, window, min_periods, pct / 100.0)
    sig = (x > thr).astype(float)
    sig[thr.isna()] = 0.0          # not enough history yet -> flat
    sig = sig.shift(1).fillna(0.0)  # implementation lag
    sig.name = "signal"
    return sig


def rolling_pct_symmetric_signal(
    x: pd.Series,
    window: int,
    min_periods: int,
    upper_pct: float,
    lower_pct: float,
) -> pd.Series:
    """
    Symmetric version: short vol (+1) when x_t > rolling upper percentile, long vol
    (-1) when x_t < rolling lower percentile, else flat. Point-in-time, 1-day lagged.
    """
    up = _rolling_quantile(x, window, min_periods, upper_pct / 100.0)
    lo = _rolling_quantile(x, window, min_periods, lower_pct / 100.0)
    sig = pd.Series(0.0, index=x.index, name="signal")
    sig[x > up] = 1.0
    sig[x < lo] = -1.0
    sig[up.isna() | lo.isna()] = 0.0
    sig = sig.shift(1).fillna(0.0)
    return sig


# ---------------------------------------------------------------------------
# Diebold-Mariano-style paired test on the daily P&L difference
# ---------------------------------------------------------------------------


def paired_pnl_test(
    pnl1: pd.Series,
    pnl2: pd.Series,
    lag: int,
    alternative: str = "greater",
) -> dict:
    """
    HAC (Newey-West) test of H0: E[pnl1 - pnl2] = 0 on the daily P&L difference.

    Positions overlap (a rolled straddle is held ~`lag` days), so the difference is
    autocorrelated; we use a Newey-West variance with `lag` lags. This is the
    Diebold-Mariano construction applied to P&L instead of forecast loss.

    alternative='greater' tests H1: pnl1 has strictly higher mean P&L than pnl2
    (i.e., the signal beats the baseline). Returns t-stat, p-value, mean diff.
    """
    df = pd.concat([pnl1.rename("a"), pnl2.rename("b")], axis=1).dropna()
    d = (df["a"] - df["b"]).values
    n = len(d)
    mean_d = float(np.mean(d))
    if n < 30 or np.allclose(d, 0.0):
        return {"t_stat": np.nan, "p_value": np.nan, "mean_diff": mean_d, "n": n, "lag": lag}

    ols = sm.OLS(d, np.ones(n)).fit()
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=max(lag, 1), use_correction=True)
    var = float(hac.cov_params()[0, 0])
    se = np.sqrt(var) if var > 0 else np.nan
    t_stat = mean_d / se if se and np.isfinite(se) else np.nan

    if not np.isfinite(t_stat):
        p = np.nan
    elif alternative == "greater":
        p = float(stats.t.sf(t_stat, df=n - 1))     # H1: mean_d > 0
    elif alternative == "less":
        p = float(stats.t.cdf(t_stat, df=n - 1))
    else:
        p = float(2 * stats.t.sf(abs(t_stat), df=n - 1))

    return {
        "t_stat": round(float(t_stat), 3) if np.isfinite(t_stat) else np.nan,
        "p_value": round(p, 4) if np.isfinite(p) else np.nan,
        "mean_diff": round(mean_d, 5),
        "ann_diff": round(mean_d * 252, 4),
        "n": n,
        "lag": lag,
    }
