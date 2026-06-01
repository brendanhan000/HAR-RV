"""
Signal 1 — USO/BNO (WTI-Brent) beta-adjusted z-score, OU mean-reversion.

Construction (all POINT-IN-TIME; nothing uses data > t to decide period t):
  1. hedge ratio beta_t : rolling OLS of log(USO) on log(BNO) over `beta_window`
     (window ENDS at t -> uses prices <= t). spread_t = logUSO_t - beta_t*logBNO_t - alpha_t
     is the current deviation from the rolling cointegration line (a demeaned residual).
  2. z_t = (spread_t - rolling_mean) / rolling_std over `z_window`.
  3. OU diagnostics: rolling AR(1) on the spread -> half-life of reversion (is it reverting
     fast enough to trade?); rolling ADF p-value -> is the residual still stationary, or has
     the cointegration broken (the structural-break / "runs against you forever" risk)?
  4. signal: enter when |z| > z_entry, exit when |z| < z_exit (state machine). Position is
     lagged one day before it trades.

P&L: beta-hedged spread, per $1 of USO notional. Long-spread (+1) = long $1 USO / short
$beta BNO. Daily P&L = pos_{t-1} * (r_USO_t - beta_{t-1}*r_BNO_t); costs charged on the
gross (1+beta) notional turned over. This is a real percentage-return stream (not abstract
units), so skew/Sharpe are directly interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover
    adfuller = None


# ---------------------------------------------------------------------------
# Point-in-time hedge ratio + spread + z-score
# ---------------------------------------------------------------------------


def rolling_hedge_ratio(log_uso: pd.Series, log_bno: pd.Series, window: int):
    """Rolling OLS beta_t (cov/var) and alpha_t of log(USO) ~ log(BNO), window ending at t."""
    cov = log_uso.rolling(window).cov(log_bno)
    var = log_bno.rolling(window).var()
    beta = cov / var
    alpha = log_uso.rolling(window).mean() - beta * log_bno.rolling(window).mean()
    return beta.rename("beta"), alpha.rename("alpha")


def build_spread(uso: pd.Series, bno: pd.Series, beta_window: int) -> pd.DataFrame:
    """Aligned frame with prices, point-in-time hedge ratio, and the residual spread."""
    df = pd.concat([uso.rename("uso"), bno.rename("bno")], axis=1).dropna().sort_index()
    df["log_uso"] = np.log(df["uso"])
    df["log_bno"] = np.log(df["bno"])
    beta, alpha = rolling_hedge_ratio(df["log_uso"], df["log_bno"], beta_window)
    df["beta"] = beta
    df["alpha"] = alpha
    df["spread"] = df["log_uso"] - df["beta"] * df["log_bno"] - df["alpha"]
    return df


def zscore(spread: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of the spread (window ends at t; used <= t to decide, then lagged)."""
    mu = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    z = (spread - mu) / sd
    return z.rename("z")


# ---------------------------------------------------------------------------
# OU / cointegration diagnostics (the tail check)
# ---------------------------------------------------------------------------


def ou_halflife(spread: pd.Series) -> float:
    """Half-life of mean reversion from an AR(1) fit: spread_t = a + b*spread_{t-1}.
    half-life = -ln2 / ln(b). Returns np.inf if b>=1 (no reversion)."""
    s = spread.dropna()
    if len(s) < 10:
        return np.nan
    x = s.shift(1).dropna()
    y = s.loc[x.index]
    b, a = np.polyfit(x.values, y.values, 1)
    if b >= 1.0 or b <= 0.0:
        return np.inf
    return float(-np.log(2.0) / np.log(b))


def rolling_diagnostics(spread: pd.Series, window: int, step: int = 21) -> pd.DataFrame:
    """Rolling half-life and ADF p-value on a `step`-day grid (point-in-time: each uses the
    trailing `window` ending at that date). ADF p<0.05 => residual still stationary/reverting."""
    rows = []
    idx = spread.index
    for i in range(window, len(idx), step):
        w = spread.iloc[i - window:i].dropna()      # strictly trailing window (ends at t-1)
        if len(w) < window // 2:
            continue
        hl = ou_halflife(w)
        adf_p = np.nan
        if adfuller is not None and w.std() > 0:
            try:
                adf_p = float(adfuller(w.values, maxlag=1, regression="c", autolag=None)[1])
            except Exception:
                pass
        rows.append({"date": idx[i], "half_life": hl, "adf_p": adf_p})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["half_life", "adf_p"])


# ---------------------------------------------------------------------------
# Signal (entry/exit state machine) + lag
# ---------------------------------------------------------------------------


def generate_spread_signal(z: pd.Series, z_entry: float, z_exit: float) -> pd.Series:
    """
    Mean-reversion entry/exit on the z-score (decided at t from data <= t):
      enter LONG spread (+1)  when z < -z_entry  (spread cheap)
      enter SHORT spread (-1) when z >  z_entry  (spread rich)
      exit to flat (0)        when |z| < z_exit  (reverted toward the mean)
    Returns the RAW (un-lagged) position; the P&L lags it one day before trading.
    """
    pos = np.zeros(len(z))
    state = 0
    zv = z.values
    for i, zi in enumerate(zv):
        if np.isnan(zi):
            state = 0
        elif state == 0:
            if zi > z_entry:
                state = -1
            elif zi < -z_entry:
                state = 1
        else:  # in a position: hold until reversion past the exit band
            if abs(zi) < z_exit:
                state = 0
        pos[i] = state
    return pd.Series(pos, index=z.index, name="position_raw")


# ---------------------------------------------------------------------------
# P&L
# ---------------------------------------------------------------------------


@dataclass
class SpreadResult:
    frame: pd.DataFrame        # everything aligned (prices, beta, spread, z, position, returns)
    gross_return: pd.Series    # pos_{t-1} * (r_uso - beta_{t-1}*r_bno)
    cost: pd.Series            # transaction cost on (1+beta) gross notional turned over
    net_return: pd.Series      # gross - cost
    position: pd.Series        # traded position (lagged one day)


def spread_pnl(df: pd.DataFrame, raw_position: pd.Series, one_way_bps: float) -> SpreadResult:
    """
    Beta-hedged spread P&L per $1 USO notional, net of one-way ETF transaction cost on the
    full (1+beta) gross notional each time the position changes. Position and hedge ratio are
    LAGGED one day (decision at t-1 trades at t) -> no look-ahead in the realised return.
    """
    out = df.copy()
    r_uso = out["uso"].pct_change()
    r_bno = out["bno"].pct_change()
    position = raw_position.reindex(out.index).fillna(0.0).shift(1).fillna(0.0)  # lag to trade
    beta_lag = out["beta"].shift(1)
    gross = position * (r_uso - beta_lag * r_bno)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (1.0 + beta_lag.abs().fillna(0.0)) * (one_way_bps / 10_000.0)
    net = (gross - cost).fillna(0.0)
    out["r_uso"], out["r_bno"] = r_uso, r_bno
    out["position"] = position
    out["gross_return"], out["cost"], out["net_return"] = gross.fillna(0.0), cost, net
    return SpreadResult(frame=out, gross_return=gross.fillna(0.0), cost=cost,
                        net_return=net, position=position)


# ---------------------------------------------------------------------------
# Benchmark: OU one-step forecast vs random walk (does reversion beat "no reversion"?)
# ---------------------------------------------------------------------------


def ou_vs_rw_forecast_errors(spread: pd.Series, beta_window: int):
    """
    Point-in-time one-step forecasts of the spread:
      RW : forecast spread_{t+1} = spread_t   (no reversion)
      OU : forecast spread_{t+1} = spread_t + (1-b)*(theta - spread_t), b,theta from a
           trailing AR(1) over `beta_window` ending at t (data <= t).
    Returns (err_rw, err_ou) squared-error series on the overlap — feed to a DM test.
    """
    s = spread.dropna()
    fc_rw, fc_ou, actual, dates = [], [], [], []
    vals = s.values
    for i in range(beta_window, len(s) - 1):
        w = vals[i - beta_window:i]            # trailing window, ends at t-1 (<= t info)
        x, y = w[:-1], w[1:]
        if np.std(x) == 0:
            continue
        b, a = np.polyfit(x, y, 1)
        cur = vals[i]
        theta = a / (1 - b) if b < 1 else cur
        fc_ou.append(cur + (1 - b) * (theta - cur) if 0 < b < 1 else cur)
        fc_rw.append(cur)
        actual.append(vals[i + 1])
        dates.append(s.index[i + 1])
    actual = np.array(actual)
    err_rw = pd.Series((actual - np.array(fc_rw)) ** 2, index=dates, name="se_rw")
    err_ou = pd.Series((actual - np.array(fc_ou)) ** 2, index=dates, name="se_ou")
    return err_rw, err_ou
