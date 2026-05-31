"""
Phase 1: Realized variance / volatility estimators.

Public API:
  compute_rv(intraday_df, cfg)          -> pd.Series (daily RV, annualized or raw)
  compute_yang_zhang(daily_df, cfg)     -> pd.Series (daily YZ variance)
  build_rv_series(cfg)                  -> pd.Series (RV, using whichever estimator)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data_pull import add_log_returns, filter_rth, load_or_fetch_daily, load_or_fetch_intraday

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intraday realized variance
# ---------------------------------------------------------------------------


def compute_rv(intraday_df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Daily realized variance = sum of squared intraday log returns.

    Parameters
    ----------
    intraday_df : raw intraday bars (pre-filter OK)
    cfg         : full Config

    Returns
    -------
    pd.Series indexed by date (tz-naive Timestamp), name="RV".
    Values are annualized (×252) if cfg.rv.annualize else raw daily.
    Days with fewer than cfg.rv.min_bars_per_day valid returns are dropped.
    """
    df = filter_rth(intraday_df)
    df = add_log_returns(df)

    # Per-day: count valid returns and sum of squares
    grouped = df.groupby("date")
    rv_raw = grouped["log_ret"].apply(lambda x: (x ** 2).sum())
    # Count total bars per day (not just valid returns) — the first bar of every
    # day always has NaN return by construction (no cross-day return), so counting
    # notna() would systematically under-count by 1 and drop full days.
    n_bars = grouped["log_ret"].size()

    # Drop days with insufficient bars (half-days, data gaps)
    bad_days = n_bars[n_bars < cfg.rv.min_bars_per_day].index
    if len(bad_days):
        log.warning(
            "Dropping %d days with < %d bars: %s … (showing first 5)",
            len(bad_days), cfg.rv.min_bars_per_day,
            list(bad_days[:5]),
        )
    rv_raw = rv_raw[~rv_raw.index.isin(bad_days)]

    if cfg.rv.annualize:
        rv = rv_raw * cfg.rv.trading_days_per_year
    else:
        rv = rv_raw

    rv.name = "RV"
    return rv.sort_index()


# ---------------------------------------------------------------------------
# Yang-Zhang range-based estimator (fallback)
# ---------------------------------------------------------------------------


def compute_yang_zhang(daily_df: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Yang-Zhang (2000) range-based variance estimator.

    Combines overnight return variance, open-to-close return variance,
    and Rogers-Satchell variance with the optimal weighting k.

    Parameters
    ----------
    daily_df : DataFrame with columns date, open, high, low, close.
    cfg      : full Config.

    Returns
    -------
    pd.Series indexed by date, name="RV_YZ".
    Annualized (×252) if cfg.rv.annualize.

    Note: the first row is NaN (needs previous close for overnight return).
    """
    df = daily_df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    o = np.log(df["open"])
    h = np.log(df["high"])
    l = np.log(df["low"])
    c = np.log(df["close"])
    c_prev = c.shift(1)

    # Overnight return (close-to-open)
    ret_o = o - c_prev
    # Open-to-close return
    ret_c = c - o

    # Rogers-Satchell component (no drift assumption needed)
    rs = (h - o) * (h - c) + (l - o) * (l - c)

    N = len(df)
    # YZ optimal k (Yang-Zhang 2000, eq. 20)
    # k = 0.34 / (1.34 + (N+1)/(N-1))
    k = 0.34 / (1.34 + (N + 1) / (N - 1))

    var_o = ret_o.var(ddof=1)  # overnight variance
    var_c = ret_c.var(ddof=1)  # open-to-close variance
    var_rs = rs.mean()          # Rogers-Satchell (no ddof needed, it's mean of demeaned)

    # Per-day YZ variance (daily, not summary)
    yz_daily = (ret_o - ret_o.mean()) ** 2 + k * (ret_c - ret_c.mean()) ** 2 + (1 - k) * rs

    yz_daily.index = df["date"]
    yz_daily.index.name = "date"
    yz_daily.name = "RV_YZ"
    yz_daily.iloc[0] = np.nan  # first row has no previous close

    if cfg.rv.annualize:
        yz_daily = yz_daily * cfg.rv.trading_days_per_year

    return yz_daily.sort_index()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def build_rv_series(cfg: Config) -> pd.Series:
    """
    Build the daily RV series using config-specified estimator.
    Falls back to Yang-Zhang if intraday fetch fails or estimator=="yang_zhang".
    """
    if cfg.rv.estimator == "yang_zhang":
        daily = load_or_fetch_daily(cfg)
        return compute_yang_zhang(daily, cfg)

    # Try intraday; fall back on error
    try:
        intraday = load_or_fetch_intraday(cfg)
        rv = compute_rv(intraday, cfg)
        if len(rv) < 100:
            raise ValueError(f"Only {len(rv)} RV days — suspiciously short.")
        return rv
    except Exception as exc:
        log.warning("Intraday RV failed (%s); falling back to Yang-Zhang.", exc)
        daily = load_or_fetch_daily(cfg)
        return compute_yang_zhang(daily, cfg)


# ---------------------------------------------------------------------------
# Convenience: realized volatility (annualized)
# ---------------------------------------------------------------------------


def rv_to_vol(rv_series: pd.Series) -> pd.Series:
    """Convert realized variance → annualized realized volatility (sqrt)."""
    return np.sqrt(rv_series).rename("RVol")


# ---------------------------------------------------------------------------
# Summary stats + plot
# ---------------------------------------------------------------------------


def rv_summary(rv: pd.Series) -> pd.DataFrame:
    vol = rv_to_vol(rv)
    stats = {
        "N_days": len(rv),
        "RV_mean": rv.mean(),
        "RV_median": rv.median(),
        "RV_std": rv.std(),
        "RV_min": rv.min(),
        "RV_max": rv.max(),
        "RVol_mean_pct": vol.mean() * 100,
        "RVol_median_pct": vol.median() * 100,
        "RVol_max_pct": vol.max() * 100,
    }
    return pd.DataFrame(stats, index=["value"]).T


def plot_rv(rv: pd.Series, output_dir: str | Path = "outputs/figures") -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vol = rv_to_vol(rv) * 100  # percent

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(rv.index, rv.values, lw=0.8, color="steelblue")
    axes[0].set_ylabel("RV (annualized variance)")
    axes[0].set_title("SPY Realized Variance — daily")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[1].plot(vol.index, vol.values, lw=0.8, color="darkorange")
    axes[1].set_ylabel("RVol (annualized %, √RV×100)")
    axes[1].set_title("SPY Realized Volatility — daily")

    fig.tight_layout()
    out = Path(output_dir) / "rv_timeseries.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved RV plot to %s", out)
    return out
