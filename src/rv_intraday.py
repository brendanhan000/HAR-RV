"""
Phase B (bounded): microstructure-noise-robust intraday realized variance.

Context: the full Phase B (refit HAR on intraday RV over 2006-2024, re-run the
walk-forward vs VIX) needs the whole sample. Polygon's FREE tier only reaches ~2 years
of intraday history, which is too short to refit/validate. So this module supports the
one thing that window CAN answer: was the Yang-Zhang DAILY estimator we fed the model a
materially worse RV *measure* than a proper intraday estimator?

Estimators
----------
  realized_variance_subsampled : naive sum of squared k-min log returns (per day).
  volatility_signature         : average RV as a function of sampling interval — the
                                 classic microstructure-noise diagnostic (RV inflates at
                                 high frequency if quote-bounce noise is present).
  two_scale_rv (TSRV)          : Zhang-Mykland-Ait-Sahalia (2005) two-scale estimator,
                                 which removes the noise bias by combining a fast (all
                                 ticks) and slow (K subgrids) scale.

All RV values are annualized (×252) when annualize=True, matching src.rv_estimator.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_pull import filter_rth

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-day log-price arrays (regular trading hours only)
# ---------------------------------------------------------------------------


def _daily_logprice_arrays(intraday_df: pd.DataFrame) -> Dict[pd.Timestamp, np.ndarray]:
    """Return {date -> array of within-day log close prices}, RTH only, time-sorted."""
    df = filter_rth(intraday_df).copy()
    df["date"] = df["timestamp"].dt.normalize().dt.tz_localize(None)
    df = df.sort_values("timestamp")
    out: Dict[pd.Timestamp, np.ndarray] = {}
    for d, g in df.groupby("date"):
        lp = np.log(g["close"].to_numpy(dtype=float))
        if lp.size >= 2:
            out[d] = lp
    return out


def _annualize(series: pd.Series, annualize: bool, tdays: int) -> pd.Series:
    return series * tdays if annualize else series


# ---------------------------------------------------------------------------
# Naive subsampled RV + volatility signature
# ---------------------------------------------------------------------------


def realized_variance_subsampled(
    intraday_df: pd.DataFrame,
    step_bars: int,
    annualize: bool = True,
    tdays: int = 252,
    min_returns: int = 5,
) -> pd.Series:
    """
    Naive realized variance from log returns sampled every `step_bars` base bars.
    e.g. base = 1-min bars, step_bars=5 -> 5-minute RV.
    """
    days = _daily_logprice_arrays(intraday_df)
    rec = {}
    for d, lp in days.items():
        sub = lp[::step_bars]
        if sub.size - 1 < min_returns:
            continue
        r = np.diff(sub)
        rec[d] = float(np.sum(r * r))
    s = pd.Series(rec).sort_index()
    s.index.name = "date"
    s.name = f"RV_{step_bars}step"
    return _annualize(s, annualize, tdays)


def volatility_signature(
    intraday_df: pd.DataFrame,
    steps_bars: List[int],
    base_minutes: int = 1,
    annualize: bool = True,
    tdays: int = 252,
) -> pd.DataFrame:
    """
    Average annualized RV at each sampling interval. A rising RV toward the
    highest frequency (smallest step) signals microstructure noise.

    Returns a DataFrame indexed by sampling-interval-in-minutes with columns
    'mean_RV', 'mean_RVol_pct', 'n_days'.
    """
    rows = []
    for step in steps_bars:
        rv = realized_variance_subsampled(intraday_df, step, annualize, tdays)
        rows.append({
            "interval_min": step * base_minutes,
            "mean_RV": float(rv.mean()),
            "mean_RVol_pct": float(np.sqrt(rv.mean()) * 100),
            "n_days": int(rv.notna().sum()),
        })
    return pd.DataFrame(rows).set_index("interval_min").sort_index()


# ---------------------------------------------------------------------------
# Two-Scale Realized Variance (TSRV) — noise-robust
# ---------------------------------------------------------------------------


def two_scale_rv(
    intraday_df: pd.DataFrame,
    K: int = 5,
    annualize: bool = True,
    tdays: int = 252,
    min_returns: int = 20,
) -> pd.Series:
    """
    Two-scale realized variance (Zhang, Mykland & Ait-Sahalia 2005), per day.

    Using all n intraday returns ("fast") and K non-overlapping subgrids ("slow"):
        RV_all   = sum of squared one-step returns
        RV_avg   = (1/K) * sum_k RV on subgrid k
        n_bar    = (n - K + 1) / K
        TSRV     = RV_avg - (n_bar / n) * RV_all          (debiases the noise)
        TSRV_adj = TSRV / (1 - n_bar / n)                 (small-sample adjustment)

    K subgrids on 1-min data = a slow scale sampled every K minutes. Floored at 0.
    """
    days = _daily_logprice_arrays(intraday_df)
    rec = {}
    for d, lp in days.items():
        n = lp.size - 1
        if n < min_returns or n < K:
            continue
        r_all = np.diff(lp)
        rv_all = float(np.sum(r_all * r_all))

        rv_sub = 0.0
        for k in range(K):
            sub = lp[k::K]
            if sub.size >= 2:
                rk = np.diff(sub)
                rv_sub += float(np.sum(rk * rk))
        rv_avg = rv_sub / K

        n_bar = (n - K + 1) / K
        tsrv = rv_avg - (n_bar / n) * rv_all
        tsrv *= 1.0 / (1.0 - n_bar / n)     # small-sample adjustment
        rec[d] = max(tsrv, 0.0)

    s = pd.Series(rec).sort_index()
    s.index.name = "date"
    s.name = f"TSRV_K{K}"
    return _annualize(s, annualize, tdays)
