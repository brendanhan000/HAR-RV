"""
Phase A: Convex (true short-gamma) P&L model.

WHY THIS EXISTS
---------------
The Phase-4 P&L was a vega-LINEAR proxy:  pnl = signal * (implied_var - realized_var).
That is linear in realized variance and therefore understates the tail: a single -10%
day enters it only through a 22-day *average*, smoothing the very risk that defines a
short-vol book. This module replaces it with the actual position the signal implies — a
short (or long) at-the-money STRADDLE — repriced with Black-Scholes along the realized
price/vol path. Losses then grow ~QUADRATICALLY with the daily move (short gamma) and
the book also bleeds vega when implied vol (VIX) spikes. Greeks are explicit.

WHAT THE SIGNAL IMPLIES
-----------------------
  signal = +1 (short vol)  ->  SHORT an ATM straddle  (short gamma, +theta, short vega)
  signal = -1 (long vol)   ->  LONG  an ATM straddle  (long  gamma, -theta, long  vega)

Each active day opens one unit (a "ladder", mirroring the linear model's daily-overlapping
booking) held for `hold_days` trading days and delta-hedged daily. The book therefore
carries up to `hold_days` overlapping straddles; on a crash EVERY open straddle takes a
gamma+vega hit at once — the realistic compounding the linear proxy averaged away.

P&L DECOMPOSITION (per open straddle, per day, for a SHORT position; long = negate)
  full P&L  = -(V_t - V_{t-1})  +  h_{t-1} * (S_t - S_{t-1})  -  hedge_cost
              \_______________/    \____________________/
               option mark-to-mkt     stale delta hedge
  The first two terms together = gamma + theta + vega + higher-order, because V is fully
  repriced at (S_t, tau_t, sigma_t=VIX_t). We DO NOT add a separate vega term (no double
  count). For transparency we also attribute an approximate greek decomposition:
      gamma_pnl ~ -0.5 * Gamma$_{t-1} * (dS)^2     (short: a loss for any move)
      theta_pnl ~ -Theta_{t-1} * dt                (short: a gain, Theta<0)
      vega_pnl  ~ -Vega_{t-1}  * dSigma            (short: loss when vol rises)
  whose sum reconciles with the full P&L up to second order.

COSTS (config-driven, convex-aware)
  - option bid/ask in implied-vol points, charged on open and close, and WIDENED in
    stress (top-decile VIX) — exactly when a short-vol book is forced to transact.
  - delta-hedge slippage on every share traded.

This remains a HYPOTHESIS TEST, not a trading system. See ASSUMPTIONS at the bottom.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import Config
from src.vrp_signal import build_vrp, generate_signal

log = logging.getLogger(__name__)

_SQRT2PI_INV = 1.0 / np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Black-Scholes toolkit (ATM straddle = call + put on same strike)
# ---------------------------------------------------------------------------


def _d1_d2(S, K, tau, sigma, r, q):
    sig_sqrt = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / sig_sqrt
    d2 = d1 - sig_sqrt
    return d1, d2


def straddle_price(S, K, tau, sigma, r=0.0, q=0.0) -> float:
    """BS price of a call+put struck at K (one straddle, per unit contract)."""
    if tau <= 0:
        return float(abs(S - K))  # intrinsic of a straddle = |S - K|
    d1, d2 = _d1_d2(S, K, tau, sigma, r, q)
    disc_r, disc_q = np.exp(-r * tau), np.exp(-q * tau)
    call = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    put = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    return float(call + put)


def straddle_greeks(S, K, tau, sigma, r=0.0, q=0.0) -> Dict[str, float]:
    """
    Greeks of a call+put straddle (per unit contract).
      delta : dV/dS                 (~0 at ATM, signed away from it)
      gamma : d2V/dS2               (>0; '$gamma' = gamma*S^2)
      vega  : dV/dsigma  per 1.00 vol
      theta : dV/dt      per 1 year (calendar); <0 for a long straddle
    """
    if tau <= 0:
        return {"delta": float(np.sign(S - K)), "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1, d2 = _d1_d2(S, K, tau, sigma, r, q)
    disc_q = np.exp(-q * tau)
    disc_r = np.exp(-r * tau)
    pdf = _SQRT2PI_INV * np.exp(-0.5 * d1 * d1)

    delta = disc_q * (2.0 * norm.cdf(d1) - 1.0)            # call + put deltas
    gamma = 2.0 * disc_q * pdf / (S * sigma * np.sqrt(tau))
    vega = 2.0 * S * disc_q * pdf * np.sqrt(tau)           # per 1.00 vol
    # theta (per year) for call+put, including carry terms
    term = -(S * disc_q * pdf * sigma) / (2.0 * np.sqrt(tau))   # shared time-decay core
    theta_call = term - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    theta_put = term + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
    theta = theta_call + theta_put
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}


# ---------------------------------------------------------------------------
# Position book
# ---------------------------------------------------------------------------


@dataclass
class _Straddle:
    sign: float            # +1 short-vol position (we SHORT the straddle) / -1 long-vol
    K: float               # strike
    n: float               # number of contracts (sizing to vega_notional)
    days_left: int         # trading days until we close
    tau: float             # current time to expiry (years)
    S_prev: float          # spot at last mark
    sigma_prev: float      # IV at last mark
    V_prev: float          # straddle value at last mark (per contract)
    hedge_shares: float    # shares currently held to hedge (signed, position-level)
    sigma_entry: float     # IV at entry (used when iv_mark == "entry": freeze vega P&L)


@dataclass
class ConvexResult:
    daily_pnl: pd.Series           # net daily book P&L
    gross_pnl: pd.Series           # before costs
    costs: pd.Series               # option spread + hedge slippage
    option_cost: pd.Series
    hedge_cost: pd.Series
    signal: pd.Series              # position series (mirrors linear model)
    vrp: pd.Series
    equity_curve: pd.Series
    greek_pnl: pd.DataFrame        # gamma / theta / vega / delta-residual attribution
    stats: dict
    label: str


def _spread_volpts(vix_t: float, cfg: Config) -> float:
    """Round-trip option spread in vol points, widened when VIX is above a fixed
    (no-look-ahead) stress level — spreads blow out exactly when a short-vol book
    is forced to transact."""
    base = cfg.convex_pnl.option_bid_ask_vol
    if vix_t >= cfg.convex_pnl.stress_vix_level:
        return base * cfg.convex_pnl.stress_spread_mult
    return base


def run_convex_backtest(
    forecasts: pd.DataFrame,
    spy_close: pd.Series,
    vix: pd.Series,
    cfg: Config,
    upper_threshold: Optional[float] = None,
    lower_threshold: Optional[float] = None,
    delta_hedge: bool = True,
    label: Optional[str] = None,
    signal: Optional[pd.Series] = None,
) -> ConvexResult:
    """
    Run a vol signal under the convex straddle P&L model.

    Parameters
    ----------
    forecasts       : WFResult.forecasts (must contain 'vix','har'; index = dates)
    spy_close       : SPY close price series (the realized path the book is hedged on)
    vix             : VIX close series (annualized vol %, used as the IV mark)
    cfg             : full Config (reads cfg.convex_pnl)
    upper/lower     : VRP thresholds (defaults: Q75/Q25, same as Phase 4)
    delta_hedge     : True -> isolate vol P&L (gamma/theta/vega). False -> naked
                      straddle held to horizon (adds directional delta risk).
    signal          : optional pre-computed position series ({-1,0,+1}, ALREADY 1-day
                      lagged). When given, it is used verbatim instead of the VRP
                      signal — this is how Phase C feeds the elevated-VIX baseline
                      through the identical P&L/cost engine.

    Returns
    -------
    ConvexResult
    """
    cp = cfg.convex_pnl
    if cp is None:
        raise ValueError("config.convex_pnl section is required for Phase A")

    label = label or ("convex_delta_hedged" if delta_hedge else "convex_naked")

    vrp = build_vrp(forecasts, cfg)   # kept for reporting (pct_vrp_positive) regardless
    if signal is None:
        # default: the Phase-4 VRP signal with (optionally full-sample) thresholds
        valid = vrp.dropna()
        if upper_threshold is None:
            upper_threshold = float(valid.quantile(0.75))
        if lower_threshold is None:
            lower_threshold = float(valid.quantile(0.25))
        signal = generate_signal(vrp, upper_threshold, lower_threshold)  # {-1,0,+1}, 1-day lag
    else:
        # caller-supplied signal (already lagged); align to the forecast index
        signal = signal.reindex(forecasts.index).fillna(0.0).astype(float)
        if upper_threshold is None:
            upper_threshold = float("nan")
        if lower_threshold is None:
            lower_threshold = float("nan")

    # --- align the realized path to the forecast/signal index ---
    idx = forecasts.index
    S = spy_close.reindex(idx).astype(float)
    V = vix.reindex(idx).astype(float)
    # forward-fill rare gaps so the hedge path is continuous
    S = S.ffill()
    V = V.ffill()

    r, q = cp.risk_free_rate, cp.dividend_yield
    hold_days = cp.hold_days
    tenor_days = cp.hold_days + cp.tenor_buffer_days
    dt = 1.0 / 252.0
    slip = cp.underlying_slippage_bps / 10_000.0

    book: List[_Straddle] = []
    rows = []  # per-day records
    single_position = not cp.ladder_daily   # default: one straddle, rolled at horizon

    dates = list(idx)
    for i, t in enumerate(dates):
        S_t = float(S.iloc[i])
        sigma_t = float(V.iloc[i]) / 100.0
        if not np.isfinite(S_t) or not np.isfinite(sigma_t) or sigma_t <= 0:
            rows.append(_empty_row(t))
            continue
        sig_t = float(signal.iloc[i]) if np.isfinite(signal.iloc[i]) else 0.0
        spread_t = _spread_volpts(float(V.iloc[i]), cfg)

        day_gross = 0.0
        day_opt_cost = 0.0
        day_hedge_cost = 0.0
        g_gamma = g_theta = g_vega = g_resid = 0.0

        # ---- 1) mark / hedge existing straddles, then decide roll/flip closes ----
        survivors: List[_Straddle] = []
        for st in book:
            # IV used to mark this straddle: live VIX, or frozen at entry (no vega P&L)
            sigma_mark = sigma_t if cp.iv_mark == "vix" else st.sigma_entry
            tau_new = max(st.tau - dt, 1.0 / 252.0)  # floor avoids tau=0 singularity
            V_now = straddle_price(S_t, st.K, tau_new, sigma_mark, r, q)

            # full mark-to-market P&L for the position (sign: short straddle = +1 -> -dV)
            dV = V_now - st.V_prev
            option_pnl = -st.sign * st.n * dV
            hedge_pnl = st.hedge_shares * (S_t - st.S_prev)
            gross = option_pnl + hedge_pnl

            # greek attribution at the PRE-move state (for transparency)
            gk = straddle_greeks(st.S_prev, st.K, st.tau, st.sigma_prev, r, q)
            dS = S_t - st.S_prev
            dSig = sigma_mark - st.sigma_prev
            gamma_pnl = -st.sign * st.n * 0.5 * gk["gamma"] * dS * dS
            theta_pnl = -st.sign * st.n * gk["theta"] * dt
            vega_pnl = -st.sign * st.n * gk["vega"] * dSig
            # delta P&L net of the (stale) hedge -> residual / discretization
            delta_pnl_opt = -st.sign * st.n * gk["delta"] * dS
            resid = gross - (gamma_pnl + theta_pnl + vega_pnl + delta_pnl_opt + hedge_pnl)

            # post-move greeks (used for both the hedge rebalance and any close cost)
            gk_now = straddle_greeks(S_t, st.K, tau_new, sigma_mark, r, q)

            # rebalance the delta hedge to the new delta (if hedging)
            if delta_hedge:
                target = st.sign * st.n * gk_now["delta"]   # shares to offset position delta
                trade = target - st.hedge_shares
                hcost = abs(trade) * S_t * slip if cp.hedge_cost else 0.0
                st.hedge_shares = target
            else:
                hcost = 0.0

            day_gross += gross
            day_hedge_cost += hcost
            g_gamma += gamma_pnl
            g_theta += theta_pnl
            g_vega += vega_pnl
            g_resid += resid + delta_pnl_opt + hedge_pnl  # lump delta+hedge into residual

            # advance state
            st.tau = tau_new
            st.S_prev = S_t
            st.sigma_prev = sigma_mark
            st.V_prev = V_now
            st.days_left -= 1

            # close on roll (horizon reached) or, in single-position mode, a signal flip/flat
            roll_due = st.days_left <= 0
            flip = single_position and (st.sign != sig_t)
            if roll_due or flip:
                day_opt_cost += st.n * gk_now["vega"] * (spread_t * 0.5)  # close = half round-trip
                if delta_hedge and cp.hedge_cost:
                    day_hedge_cost += abs(st.hedge_shares) * S_t * slip   # unwind the hedge
            else:
                survivors.append(st)
        book = survivors

        # ---- 2) open a new straddle ----
        #   ladder mode : one new straddle each active day (overlapping book)
        #   single mode : open only when flat (covers initial entry, rolls, and flips)
        if sig_t != 0.0 and (cp.ladder_daily or len(book) == 0):
            tau0 = tenor_days / 252.0
            K = S_t  # ATM (spot)
            gk0 = straddle_greeks(S_t, K, tau0, sigma_t, r, q)
            # size to target vega notional: n * vega(per 1.00 vol) * 0.01 = vega_notional per vol-pt
            vega_per_volpt = gk0["vega"] * 0.01
            n = cp.vega_notional / vega_per_volpt if vega_per_volpt > 0 else 0.0
            V0 = straddle_price(S_t, K, tau0, sigma_t, r, q)
            hedge0 = sig_t * n * gk0["delta"] if delta_hedge else 0.0
            book.append(_Straddle(
                sign=sig_t, K=K, n=n, days_left=hold_days, tau=tau0,
                S_prev=S_t, sigma_prev=sigma_t, V_prev=V0, hedge_shares=hedge0,
                sigma_entry=sigma_t,
            ))
            day_opt_cost += n * gk0["vega"] * (spread_t * 0.5)  # open = half round-trip

        day_cost = day_opt_cost + day_hedge_cost
        rows.append({
            "date": t,
            "gross": day_gross,
            "option_cost": day_opt_cost,
            "hedge_cost": day_hedge_cost,
            "cost": day_cost,
            "net": day_gross - day_cost,
            "g_gamma": g_gamma,
            "g_theta": g_theta,
            "g_vega": g_vega,
            "g_resid": g_resid,
            "n_open": len(book),
        })

    df = pd.DataFrame(rows).set_index("date")
    net = df["net"].fillna(0.0)
    gross = df["gross"].fillna(0.0)
    costs = df["cost"].fillna(0.0)
    equity = net.cumsum()

    greek_pnl = df[["g_gamma", "g_theta", "g_vega", "g_resid"]].rename(
        columns={"g_gamma": "gamma", "g_theta": "theta", "g_vega": "vega", "g_resid": "delta_resid"}
    )

    stats = _convex_stats(net, gross, costs, signal, vrp, upper_threshold,
                          lower_threshold, df, label)

    return ConvexResult(
        daily_pnl=net, gross_pnl=gross, costs=costs,
        option_cost=df["option_cost"].fillna(0.0), hedge_cost=df["hedge_cost"].fillna(0.0),
        signal=signal, vrp=vrp, equity_curve=equity, greek_pnl=greek_pnl,
        stats=stats, label=label,
    )


def _empty_row(t) -> dict:
    return {"date": t, "gross": 0.0, "option_cost": 0.0, "hedge_cost": 0.0, "cost": 0.0,
            "net": 0.0, "g_gamma": 0.0, "g_theta": 0.0, "g_vega": 0.0, "g_resid": 0.0,
            "n_open": 0}


# ---------------------------------------------------------------------------
# Statistics (parallels src.vrp_signal._compute_stats for apples-to-apples)
# ---------------------------------------------------------------------------


def _convex_stats(net, gross, costs, signal, vrp, upper_thr, lower_thr, df, label) -> dict:
    active = net[signal != 0]
    n_active = int((df["n_open"] > 0).sum())
    ann_net = float(net.mean() * 252)
    ann_vol = float(net.std() * np.sqrt(252))
    sharpe = ann_net / ann_vol if ann_vol > 0 else np.nan

    cum = net.cumsum()
    dd = cum - cum.cummax()
    max_dd = float(dd.min())
    worst = float(net.min())
    worst_dt = net.idxmin()

    # unitless tail ratios — these are directly comparable across P&L models / sizings
    dd_over_ann = abs(max_dd) / abs(ann_net) if ann_net != 0 else np.nan
    worst_over_ann = abs(worst) / abs(ann_net) if ann_net != 0 else np.nan
    daily_sd = float(net.std())
    worst_z = worst / daily_sd if daily_sd > 0 else np.nan

    return {
        "label": label,
        "upper_threshold": round(upper_thr, 5),
        "lower_threshold": round(lower_thr, 5),
        "n_days_total": int(len(net)),
        "n_days_active": n_active,
        "total_net_pnl": round(float(net.sum()), 4),
        "total_gross_pnl": round(float(gross.sum()), 4),
        "total_costs": round(float(costs.sum()), 4),
        "ann_net_pnl": round(ann_net, 5),
        "ann_vol_pnl": round(ann_vol, 5),
        "sharpe_ratio": round(sharpe, 3) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": round(max_dd, 4),
        "worst_day_pnl": round(worst, 4),
        "worst_day_date": str(worst_dt.date()) if hasattr(worst_dt, "date") else str(worst_dt),
        "dd_over_ann_pnl": round(dd_over_ann, 2) if np.isfinite(dd_over_ann) else np.nan,
        "worst_over_ann_pnl": round(worst_over_ann, 3) if np.isfinite(worst_over_ann) else np.nan,
        "worst_day_zscore": round(worst_z, 2) if np.isfinite(worst_z) else np.nan,
        "win_rate_active": round(float((active > 0).mean()), 3) if len(active) else np.nan,
        "pct_vrp_positive": round(100 * float((vrp.dropna() > 0).mean()), 1),
    }


# ===========================================================================
# ASSUMPTIONS — what would invalidate the convex results
#   1. VIX (30 cal. days, SPX) used as the SPY straddle IV and tenor mark; a
#      ~22-trading-day option is close but not identical. Skew is ignored (ATM only).
#   2. We mark IV to VIX daily but use a single ATM straddle, not a full strip; the
#      true vega/gamma profile of a desk book differs.
#   3. r=q=0 by default: removes spurious carry from the delta hedge but ignores real
#      financing of the share hedge and the option premium.
#   4. Liquidity: we can always trade the ATM straddle and rebalance the hedge daily at
#      the modeled (stress-widened) cost. In 2008/2020 fills were worse than modeled.
#   5. Discrete daily hedging leaves gap risk; intraday gaps (limit-down) are not modeled
#      beyond the close-to-close move, so even THIS convex model can understate the tail.
# ===========================================================================
