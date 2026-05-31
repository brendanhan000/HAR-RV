"""
Phase B (bounded) unit tests for the intraday RV estimators.

  1. naive subsampled RV recovers the true variance on a noise-free random walk
  2. volatility signature is ~flat with NO microstructure noise
  3. with added quote-bounce noise, naive 1-min RV inflates vs coarser sampling
  4. TSRV removes most of that noise bias (closer to truth than naive 1-min)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rv_intraday import (
    realized_variance_subsampled,
    two_scale_rv,
    volatility_signature,
)


def _make_1min(n_days=40, bars=390, sigma_min=0.0008, noise_sd=0.0, seed=0):
    """Synthetic 1-min RTH bars: true log-price is a random walk with per-minute
    std `sigma_min`; observed close adds iid noise of std `noise_sd` (microstructure)."""
    rng = np.random.default_rng(seed)
    rows = []
    for di in range(n_days):
        day = pd.Timestamp("2025-01-06") + pd.Timedelta(days=di)
        base = day.tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        true_lp = np.log(100.0) + np.cumsum(rng.normal(0, sigma_min, bars))
        obs_lp = true_lp + rng.normal(0, noise_sd, bars) if noise_sd > 0 else true_lp
        close = np.exp(obs_lp)
        for i in range(bars):
            ts = base + pd.Timedelta(minutes=i)
            rows.append({"timestamp": ts, "open": close[i], "high": close[i],
                         "low": close[i], "close": close[i], "volume": 1000})
    return pd.DataFrame(rows)


def test_naive_rv_recovers_true_variance_no_noise():
    bars, sigma = 390, 0.0008
    df = _make_1min(n_days=60, bars=bars, sigma_min=sigma, noise_sd=0.0, seed=1)
    rv = realized_variance_subsampled(df, step_bars=1, annualize=False)
    true_daily_var = (bars - 1) * sigma ** 2
    # average across days should be close to the true integrated variance
    assert rv.mean() == pytest.approx(true_daily_var, rel=0.15)


def test_signature_flat_without_noise():
    df = _make_1min(n_days=50, noise_sd=0.0, seed=2)
    sig = volatility_signature(df, steps_bars=[1, 2, 5, 10, 15], base_minutes=1, annualize=False)
    hi = sig.loc[1, "mean_RV"]      # 1-min
    lo = sig.loc[15, "mean_RV"]     # 15-min
    # no noise -> roughly flat across sampling frequency
    assert abs(hi - lo) / lo < 0.20


def test_noise_inflates_high_frequency_rv():
    df = _make_1min(n_days=50, sigma_min=0.0008, noise_sd=0.0006, seed=3)
    sig = volatility_signature(df, steps_bars=[1, 5, 15], base_minutes=1, annualize=False)
    # quote-bounce noise inflates the 1-min RV well above the 15-min RV
    assert sig.loc[1, "mean_RV"] > 1.3 * sig.loc[15, "mean_RV"]


def test_tsrv_corrects_noise_bias():
    bars, sigma, noise = 390, 0.0008, 0.0006
    df = _make_1min(n_days=60, bars=bars, sigma_min=sigma, noise_sd=noise, seed=4)
    true_var = (bars - 1) * sigma ** 2
    naive_1min = realized_variance_subsampled(df, step_bars=1, annualize=False).mean()
    tsrv = two_scale_rv(df, K=5, annualize=False).mean()
    # naive is biased high; TSRV is much closer to the truth
    assert naive_1min > 1.5 * true_var
    assert abs(tsrv - true_var) < abs(naive_1min - true_var)
    assert tsrv == pytest.approx(true_var, rel=0.5)
