"""
Phase 2/3 — the drop-in sizing overlay.

`apply_overlay(returns, signal, cfg)` takes ANY strategy's per-period return stream
(plus, optionally, its raw position signal) and returns a SIZED position per period,
the resulting sized P&L net of turnover cost, and every intermediate so the effect of
each rule is auditable. It is strategy-agnostic: nothing here knows about straddles.

Composition (fixed order; the cap is last and binding):
    R = returns / reference_capital                      # work in return-on-capital units
    sigma_hat = lagged annualised vol of R               # (risk.py, data <= t-1)
    w_base    = vol-target OR fractional-Kelly leverage
              -> clip [0, max_leverage]
    w_brake   = w_base * drawdown_brake(realised sized equity)   # optional, causal
    w_final   = min(w_brake, hard_session_cap)            # BINDING, applied last

NO LOOK-AHEAD: w_t is built only from estimates of data <= t-1 (the estimators lag
internally) and, for the brake, from realised sized P&L <= t-1. The position w_t is
then applied to period t's return: sized_pnl_t = w_t * returns_t. The strategy's own
signal_t is its (already-lagged) decision for t, so position_t = w_t * signal_t is
known at the open of t; turnover_t = |position_t - position_{t-1}|.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.sizing.risk import estimate_vol
from src.sizing.rules import (
    drawdown_brake_factor,
    hard_session_cap,
    kelly_leverage,
    lagged_mean,
    vol_target_leverage,
)


@dataclass
class SizedResult:
    leverage: pd.Series          # final leverage w_t (>=0), NaN during warm-up
    position: pd.Series          # w_t * signal_t (the traded position)
    sized_pnl: pd.Series         # net sized P&L (w_{t}*returns_t - turnover cost)
    gross_sized_pnl: pd.Series   # w_t * returns_t (before turnover cost)
    turnover: pd.Series          # |position_t - position_{t-1}|
    turnover_cost: pd.Series
    w_base: pd.Series            # leverage from the base rule (pre brake/cap)
    w_cap: pd.Series             # the per-session cap ceiling
    brake_mult: pd.Series        # drawdown-brake multiplier in [floor,1]
    cap_binding: pd.Series       # bool: did the cap bind that day?
    sigma_hat: pd.Series         # annualised vol estimate (return units)
    flat_pnl: pd.Series          # the input stream (for flat-vs-sized comparison)
    meta: dict


def _causal_brake_and_cap(
    w_base: np.ndarray,
    w_cap: np.ndarray,
    R: np.ndarray,
    brake_enabled: bool,
    threshold: float,
    floor: float,
    max_dd: float,
):
    """
    Walk forward applying (optional) drawdown brake then the hard cap, tracking realised
    sized equity so the brake responds to the BOOK's drawdown through t-1. Returns
    (w_final, brake_mult). Pure/causal: w[t] uses only equity built from R[<t].
    """
    n = len(w_base)
    w_final = np.zeros(n)
    brake_mult = np.ones(n)
    equity = 0.0          # cumulative sized return-on-capital (starts at 0; peak tracks 1+equity)
    peak = 0.0
    for t in range(n):
        wb = w_base[t] if np.isfinite(w_base[t]) else 0.0
        if brake_enabled:
            dd = 0.0 if (1.0 + peak) <= 0 else max(0.0, (peak - equity) / (1.0 + peak))
            if dd <= threshold:
                m = 1.0
            elif dd >= max_dd:
                m = floor
            else:
                m = 1.0 - (1.0 - floor) * (dd - threshold) / (max_dd - threshold)
            brake_mult[t] = m
            wb = wb * m
        capval = w_cap[t]
        cap = 0.0 if np.isnan(capval) else capval   # NaN -> stand down; +inf -> cap disabled
        wt = max(min(wb, cap), 0.0)
        w_final[t] = wt
        # realise the period and advance equity (R[t] known only after the position is set)
        ret = wt * (R[t] if np.isfinite(R[t]) else 0.0)
        equity += ret
        peak = max(peak, equity)
    return w_final, brake_mult


def apply_overlay(
    returns: pd.Series,
    signal: Optional[pd.Series],
    cfg,
) -> SizedResult:
    """
    Size `returns` (a strategy's flat/unit per-period P&L stream) with the configured rules.

    Parameters
    ----------
    returns : per-period P&L of the strategy at FLAT (unit) sizing.
    signal  : optional strategy position ({-1,0,+1} or continuous), already lagged by the
              strategy. Used for the traded position & turnover; if None, treated as all-1
              (always in the market). sized_pnl uses `returns` directly either way.
    cfg     : full Config (reads cfg.sizing.risk and cfg.sizing.rules).
    """
    rc = cfg.sizing.risk
    ru = cfg.sizing.rules
    ppy = rc.periods_per_year

    returns = returns.astype(float).sort_index()
    if signal is None:
        signal = pd.Series(1.0, index=returns.index)
    signal = signal.reindex(returns.index).fillna(0.0).astype(float)

    C = ru.reference_capital
    R = returns / C                                   # return-on-capital units

    # --- point-in-time risk & edge (all lagged inside the estimators) ---
    # Condition the RISK estimate on in-position days: a frequently-flat strategy's raw stream
    # is mostly zeros, so its trailing vol collapses and naive leverage spikes at re-entry.
    # Estimates are computed on the active sub-series (each value lagged), then forward-filled
    # back to the full index -> day t carries the most recent estimate from active data < t.
    active = (signal != 0)
    if ru.condition_on_active and (~active).any() and active.any():
        R_basis = R[active]
    else:
        R_basis = R
    sigma_ann = estimate_vol(R_basis, rc).reindex(R.index).ffill()    # annualised vol of R
    sigma_daily = sigma_ann / np.sqrt(ppy)
    var_daily = sigma_daily ** 2
    mu_daily = lagged_mean(R_basis, window=ru.kelly_edge_window,
                           min_periods=rc.ewma_min_periods).reindex(R.index).ffill()

    # --- base leverage rule ---
    vol_floor = ru.vol_floor_frac * ru.target_vol
    if ru.method == "vol_target":
        w_base = vol_target_leverage(sigma_ann, ru.target_vol, vol_floor=vol_floor,
                                     max_leverage=ru.max_leverage)
    elif ru.method == "kelly":
        w_base = kelly_leverage(mu_daily, var_daily, ru.kelly_fraction,
                                max_leverage=ru.max_leverage)
    else:
        raise ValueError(f"unknown sizing method '{ru.method}' (use 'vol_target' or 'kelly')")

    # --- hard per-session cap inputs (point-in-time, per unit, in R units) ---
    if ru.annual_expected_pnl is not None:
        ann_expected = pd.Series(ru.annual_expected_pnl / C, index=returns.index)
    else:
        ann_expected = lagged_mean(R, window=None, min_periods=rc.ewma_min_periods) * ppy
    adverse_loss = ru.cap_loss_sigma * sigma_daily
    if ru.cap_enabled:
        w_cap = hard_session_cap(w_base, ann_expected, adverse_loss, ru.cap_fraction)
    else:
        w_cap = pd.Series(np.inf, index=returns.index)

    # --- compose: brake (causal) then cap; w_final >= 0 ---
    w_final_arr, brake_arr = _causal_brake_and_cap(
        w_base.values.astype(float),
        w_cap.values.astype(float),
        R.values.astype(float),
        brake_enabled=ru.brake_enabled,
        threshold=ru.brake_threshold,
        floor=ru.brake_floor,
        max_dd=ru.brake_max_dd,
    )
    leverage = pd.Series(w_final_arr, index=returns.index, name="leverage")
    brake_mult = pd.Series(brake_arr, index=returns.index, name="brake_mult")
    # mark warm-up (no sigma estimate yet) as NaN leverage for reporting (position still 0)
    leverage_report = leverage.where(sigma_ann.notna())

    # --- traded position, turnover, sized P&L ---
    position = (leverage * signal).rename("position")
    turnover = position.diff().abs().fillna(position.abs()).rename("turnover")
    turnover_cost = (ru.turnover_cost_per_unit * turnover).rename("turnover_cost")
    gross_sized = (leverage * returns).rename("gross_sized_pnl")
    sized_pnl = (gross_sized - turnover_cost).rename("sized_pnl")

    cap_binding = ((w_cap.values < w_base.fillna(np.inf).values) & np.isfinite(w_cap.values))
    cap_binding = pd.Series(cap_binding, index=returns.index, name="cap_binding")

    meta = {
        "method": ru.method,
        "reference_capital": C,
        "target_vol": ru.target_vol,
        "vol_floor": vol_floor,
        "max_leverage": ru.max_leverage,
        "kelly_fraction": ru.kelly_fraction,
        "cap_enabled": ru.cap_enabled,
        "cap_fraction": ru.cap_fraction,
        "cap_loss_sigma": ru.cap_loss_sigma,
        "brake_enabled": ru.brake_enabled,
        "turnover_cost_per_unit": ru.turnover_cost_per_unit,
        "mean_leverage": float(leverage_report.mean()),
        "pct_days_cap_binding": float(cap_binding[sigma_ann.notna()].mean() * 100),
    }

    return SizedResult(
        leverage=leverage_report,
        position=position,
        sized_pnl=sized_pnl,
        gross_sized_pnl=gross_sized,
        turnover=turnover,
        turnover_cost=turnover_cost,
        w_base=w_base,
        w_cap=w_cap,
        brake_mult=brake_mult,
        cap_binding=cap_binding,
        sigma_hat=sigma_ann,
        flat_pnl=returns,
        meta=meta,
    )
