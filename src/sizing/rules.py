"""
Phase 2 — Sizing rules (config-driven, defendable defaults).

Each rule maps point-in-time risk/edge estimates (all lagged: data <= t-1) to a
LEVERAGE w_t >= 0 — a multiple of the strategy's flat (unit) position. The overlay
(overlay.py) composes them in this fixed order:

    base rule (vol-target OR fractional Kelly)
        -> clip to [0, max_leverage]
        -> drawdown brake (optional, causal on realised sized equity)
        -> HARD PER-SESSION CAP (binding, applied LAST)

Design notes that matter for the verdict
-----------------------------------------
* A CONSTANT leverage change (e.g. the cap forcing 0.2x) scales P&L and drawdown
  together, so it moves absolute risk but leaves Sharpe and the drawdown/annual RATIO
  unchanged. Only the TIME-VARYING part of leverage (vol-target de-levering in
  high-vol regimes; the brake cutting after a drawdown) can change risk-ADJUSTED
  numbers. We keep these separable so Phase 3 can attribute the effect.
* Kelly uses an ESTIMATED edge; the edge is the noisy part. `simulate_kelly_ruin`
  shows deep-drawdown probability as a function of edge-estimate error — the reason
  the default is fractional (1/4) Kelly, never full.
* Vol target and Kelly fraction are PRIORS, not tuned to the backtest.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Point-in-time edge estimate (for Kelly and the cap)
# ---------------------------------------------------------------------------


def lagged_mean(returns: pd.Series, window: Optional[int] = None, min_periods: int = 21) -> pd.Series:
    """
    Point-in-time mean return (the 'edge' mu_hat), LAGGED so mu_hat[t] uses data <= t-1.
    window=None -> expanding mean (stable, slow); else a rolling window.
    """
    r = returns.astype(float)
    if window is None:
        m = r.expanding(min_periods=min_periods).mean()
    else:
        m = r.rolling(window=window, min_periods=min_periods).mean()
    out = m.shift(1)
    out.name = "mu_hat"
    return out


# ---------------------------------------------------------------------------
# Base rule 1: volatility targeting
# ---------------------------------------------------------------------------


def vol_target_leverage(
    sigma_hat: pd.Series,
    target_vol: float,
    vol_floor: float = 0.0,
    max_leverage: float = np.inf,
) -> pd.Series:
    """
    w_t = target_vol / max(sigma_hat_t, vol_floor), clipped to [0, max_leverage].

    sigma_hat is the lagged ANNUALISED return-volatility of the strategy's own stream.
    `vol_floor` prevents the flat-stretch blow-up (when the strategy sits flat its
    trailing vol collapses to ~0 -> naive leverage -> infinity). NaN where sigma_hat NaN.
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be > 0")
    denom = sigma_hat.clip(lower=vol_floor) if vol_floor > 0 else sigma_hat
    w = target_vol / denom
    w = w.clip(lower=0.0, upper=max_leverage)
    w.name = "w_vol_target"
    return w


# ---------------------------------------------------------------------------
# Base rule 2: fractional Kelly
# ---------------------------------------------------------------------------


def kelly_leverage(
    mu_hat: pd.Series,
    var_hat: pd.Series,
    kelly_fraction: float,
    max_leverage: float = np.inf,
    allow_short: bool = False,
) -> pd.Series:
    """
    Fractional-Kelly leverage  w_t = kelly_fraction * mu_hat_t / var_hat_t.

    mu_hat (lagged mean return) and var_hat (lagged return variance) must be in the
    SAME per-period return units (e.g. both daily on returns-of-capital). Full Kelly is
    kelly_fraction=1; we DEFAULT to 1/4 and never recommend full (see simulate_kelly_ruin).

    allow_short=False clips negative leverage to 0 (don't flip the strategy on a noisy
    negative edge estimate — just stand down). Clipped to [.., max_leverage].
    """
    if not (0.0 < kelly_fraction <= 1.0):
        raise ValueError("kelly_fraction must be in (0, 1]; full Kelly is 1.0 and is discouraged")
    with np.errstate(divide="ignore", invalid="ignore"):
        w = kelly_fraction * mu_hat / var_hat
    lower = -max_leverage if allow_short else 0.0
    w = w.clip(lower=lower, upper=max_leverage)
    w.name = "w_kelly"
    return w


# ---------------------------------------------------------------------------
# Drawdown brake (optional, causal)
# ---------------------------------------------------------------------------


def drawdown_brake_factor(
    drawdown: pd.Series,
    threshold: float,
    floor: float,
    max_dd: float,
) -> pd.Series:
    """
    Map a (lagged, point-in-time) drawdown series -> leverage multiplier in [floor, 1].

    multiplier = 1                         for dd <= threshold
               = linear ramp 1 -> floor    for threshold < dd < max_dd
               = floor                      for dd >= max_dd
    `drawdown` is a positive fraction of peak equity (0 = at high-water mark). The
    overlay feeds it the realised SIZED equity drawdown through t-1 (so it is causal).
    """
    if not (0 <= threshold < max_dd):
        raise ValueError("require 0 <= threshold < max_dd")
    if not (0 <= floor <= 1):
        raise ValueError("floor must be in [0,1]")
    dd = drawdown.clip(lower=0.0)
    ramp = (dd - threshold) / (max_dd - threshold)
    mult = 1.0 - (1.0 - floor) * ramp.clip(lower=0.0, upper=1.0)
    mult.name = "brake_mult"
    return mult


# ---------------------------------------------------------------------------
# Hard per-session cap  (BINDING, applied LAST)
# ---------------------------------------------------------------------------


def hard_session_cap(
    leverage: pd.Series,
    ann_expected_pnl: pd.Series,
    adverse_loss: pd.Series,
    cap_fraction: float,
) -> pd.Series:
    """
    Bind leverage so a single adverse SESSION cannot lose more than `cap_fraction` of the
    strategy's ANNUAL EXPECTED P&L (default 5%). Applied AFTER every other rule.

        w_cap_t = cap_fraction * ann_expected_pnl_t / adverse_loss_t
        w_final = min(leverage, w_cap)            (>= 0)

    All inputs are per-UNIT-exposure and point-in-time (lagged):
      ann_expected_pnl : estimate of annual P&L per unit (e.g. expanding-mean daily * 252)
      adverse_loss     : adverse one-session loss per unit (e.g. cap_loss_sigma * daily sigma)
    If ann_expected_pnl <= 0 (no demonstrated edge yet) the cap forces w -> 0 (stand down).

    NOTE this cap is a RATIO of annual edge to session loss, hence scale-invariant: it does
    not depend on the notional. For a strategy whose annual P&L is small relative to its
    session risk, the cap binds hard and dominates the other rules — that is a finding,
    not a bug.
    """
    if not (0 < cap_fraction <= 1):
        raise ValueError("cap_fraction must be in (0,1]")
    with np.errstate(divide="ignore", invalid="ignore"):
        w_cap = cap_fraction * ann_expected_pnl / adverse_loss
    w_cap = w_cap.clip(lower=0.0)
    capped = pd.concat([leverage, w_cap], axis=1).min(axis=1)
    capped = capped.where(adverse_loss > 0, 0.0)  # no loss estimate -> no position
    capped.name = "w_capped"
    return capped


# ---------------------------------------------------------------------------
# Kelly ruin / deep-drawdown sensitivity to EDGE-ESTIMATE ERROR
# ---------------------------------------------------------------------------


def simulate_kelly_ruin(
    empirical_returns: Sequence[float],
    sharpe_ann: float,
    vol_ann: float,
    kelly_fractions: Sequence[float] = (0.25, 0.5, 1.0),
    edge_errors: Sequence[float] = tuple(np.round(np.arange(-0.5, 0.51, 0.1), 2)),
    horizon: int = 1260,
    n_sims: int = 4000,
    ruin_dd: float = 0.5,
    periods_per_year: int = 252,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Monte-Carlo deep-drawdown ('ruin') probability when Kelly is sized with a MIS-ESTIMATED
    edge. This is THE key caveat for Kelly sizing, not a footnote.

    Construction (C-free, parameterised by the strategy's own shape):
      * standardise `empirical_returns` to zero mean / unit variance -> keeps the real skew
        and fat left tail (a short-vol book's tail is what bankrupts an over-bettor);
      * rebuild per-period returns with the strategy's daily mean/vol implied by
        (sharpe_ann, vol_ann): sigma_d = vol_ann/sqrt(ppy), mu_d = (sharpe_ann/sqrt(ppy))*sigma_d;
      * full Kelly leverage L* = mu_d / sigma_d^2;
      * you BELIEVE the edge is (1+eps) of truth, so you bet c*(1+eps)*L*;
      * compound W_{t+1} = W_t * (1 + lev * R_t); a period with 1+lev*R <= 0 is a wipe-out.

    Returns a tidy DataFrame: one row per (kelly_fraction, edge_error) with
      p_deep_dd  : P(max drawdown >= ruin_dd over the horizon)
      p_wipeout  : P(equity effectively ruined, <= 1e-6 of start)
      median_cagr: median annualised growth (so you see the upside you trade away too)
    """
    z = np.asarray(empirical_returns, dtype=float)
    z = z[np.isfinite(z)]
    z = (z - z.mean()) / z.std(ddof=0)             # standardised shape (skew/kurt preserved)

    sig_d = vol_ann / np.sqrt(periods_per_year)
    mu_d = (sharpe_ann / np.sqrt(periods_per_year)) * sig_d
    full_kelly = mu_d / sig_d ** 2

    rng = np.random.default_rng(seed)
    rows = []
    for c in kelly_fractions:
        for eps in edge_errors:
            lev = c * (1.0 + eps) * full_kelly
            draws = rng.choice(z, size=(n_sims, horizon), replace=True)
            R = mu_d + sig_d * draws
            growth = np.clip(1.0 + lev * R, 1e-12, None)   # <=0 -> ruin
            eq = np.cumprod(growth, axis=1)
            peak = np.maximum.accumulate(eq, axis=1)
            maxdd = (1.0 - eq / peak).max(axis=1)
            wipeout = eq[:, -1] <= 1e-6
            cagr = eq[:, -1] ** (periods_per_year / horizon) - 1.0
            rows.append({
                "kelly_fraction": c,
                "edge_error": eps,
                "leverage_x": round(float(lev), 3),
                "p_deep_dd": round(float((maxdd >= ruin_dd).mean()), 4),
                "p_wipeout": round(float(wipeout.mean()), 4),
                "median_cagr": round(float(np.median(cagr)), 4),
            })
    return pd.DataFrame(rows)
