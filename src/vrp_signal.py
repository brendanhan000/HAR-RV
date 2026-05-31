"""
Phase 4: VRP signal and cost-aware backtest.

THIS IS A HYPOTHESIS TEST, NOT A TRADING SYSTEM.
All assumptions are explicit and configurable. Results do not imply
live tradability. See ASSUMPTIONS section at bottom of this module.

Design:
  VRP_t = implied_var_t - HAR_forecast_t
        = (VIX_t / 100)^2 - HAR_RV_forecast_t    [both annualized variance]

  Signal:
    VRP_t > upper_threshold  → short vol (collect premium)
    VRP_t < lower_threshold  → long vol  (hedge tail risk)
    otherwise                → flat

  Position sizing: fixed notional (1 unit) per signal.
  Entry/exit: next-day open (1-day implementation lag).
  P&L proxy: short vol gains when RV_realized < implied_var at entry.
    daily_pnl = position * (VRP_entry - (actual_RV_t - implied_var_entry))

  More precisely, we proxy short-vol P&L as:
    short_vol_pnl_t = position_t * (implied_var_entry - actual_RV_t)
  where actual_RV_t is the realized variance over the holding period.

  Costs deducted per trade:
    - bid_ask: option_bid_ask_vega (config) converted to variance units
    - slippage: slippage_bps on notional
    - financing: daily financing on short premium received

ASSUMPTIONS (what would invalidate these results):
  1. VRP proxy: VIX^2 ≠ true implied variance. VIX is a model-free measure
     of 30-day implied vol; we use it as a proxy for h-day implied variance.
     Mismatch in horizon (22 vs 30 calendar days) introduces noise.
  2. P&L proxy: we do not model actual option positions (delta, gamma, vega).
     The pnl = implied_var - realized_var is a vega-linear approximation.
  3. Transaction costs are rough estimates; actual option bid-ask spreads
     are wider in stress periods when short-vol positions are most painful.
  4. No margin, capital requirements, or broker constraints modeled.
  5. No volatility-of-volatility risk; gamma/vanna/volga effects ignored.
  6. The backtest period includes GFC (2008) and COVID (2020) which are
     known in advance — a live system would face these cold.
  7. Short-vol strategies have extreme left-tail risk. Max drawdown and
     worst day matter more than Sharpe ratio.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import Config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VRP construction
# ---------------------------------------------------------------------------


def build_vrp(
    forecasts: pd.DataFrame,
    cfg: Config,
) -> pd.Series:
    """
    Compute VRP_t = implied_var_t - har_forecast_t.

    Uses 'vix' column (annualized variance = (VIX/100)^2) and 'har' forecast.
    Both must already be in annualized variance units.

    Returns pd.Series of VRP, NaN where either input is missing.
    """
    if "vix" not in forecasts.columns or "har" not in forecasts.columns:
        raise ValueError("forecasts must contain 'vix' and 'har' columns")

    vrp = forecasts["vix"] - forecasts["har"]
    vrp.name = "VRP"
    log.info(
        "VRP: n=%d  mean=%.5f  std=%.5f  pct_positive=%.1f%%",
        vrp.notna().sum(),
        vrp.mean(),
        vrp.std(),
        100 * (vrp > 0).mean(),
    )
    return vrp


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def generate_signal(
    vrp: pd.Series,
    upper_threshold: float,
    lower_threshold: float,
) -> pd.Series:
    """
    Generate daily position signal from VRP.

    Parameters
    ----------
    vrp              : VRP series (annualized variance units)
    upper_threshold  : VRP > upper → short vol (+1)
    lower_threshold  : VRP < lower → long vol  (-1)

    Returns
    -------
    pd.Series of {-1, 0, +1}, shifted by 1 day (implementation lag:
    signal at t is acted on at t+1 open).
    """
    signal = pd.Series(0, index=vrp.index, name="signal", dtype=float)
    signal[vrp > upper_threshold] = 1.0    # short vol
    signal[vrp < lower_threshold] = -1.0   # long vol
    signal[vrp.isna()] = 0.0

    # 1-day implementation lag
    signal = signal.shift(1).fillna(0)
    return signal


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def _daily_transaction_cost(
    signal: pd.Series,
    cfg: Config,
) -> pd.Series:
    """
    Estimate transaction costs per day.

    Costs are incurred on the day a position is entered or exited
    (signal changes from 0→nonzero or nonzero→0 or flips sign).

    Cost per trade (one round-trip):
      bid_ask  : cfg.transaction_costs.option_bid_ask_vega converted to
                 variance units: we approximate 1 vega ≈ 0.01 annualized vol,
                 so vega cost → variance cost ≈ bid_ask_vega * 0.02
      slippage : slippage_bps / 10000 of notional (taken as 1.0)
    """
    trades = (signal.diff().abs() > 0).astype(float)
    # bid-ask in variance units (rough: 1 vega point ≈ 0.01 ann_vol → 2*0.01*vol cost)
    bid_ask_var = cfg.transaction_costs.option_bid_ask_vega * 0.02
    slippage = cfg.transaction_costs.slippage_bps / 10_000
    cost_per_trade = bid_ask_var + slippage
    return trades * cost_per_trade


def _daily_financing_cost(
    signal: pd.Series,
    cfg: Config,
) -> pd.Series:
    """Daily financing cost on short-vol positions (borrow cost on short premium)."""
    daily_rate = cfg.transaction_costs.financing_rate_annual / 252
    return signal.abs() * daily_rate


# ---------------------------------------------------------------------------
# P&L engine
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    daily_pnl: pd.Series          # net daily P&L
    gross_pnl: pd.Series          # before costs
    costs: pd.Series              # transaction + financing
    signal: pd.Series             # position series
    vrp: pd.Series                # VRP series
    equity_curve: pd.Series       # cumulative P&L
    stats: dict                   # summary statistics


def run_backtest(
    forecasts: pd.DataFrame,
    cfg: Config,
    upper_threshold: Optional[float] = None,
    lower_threshold: Optional[float] = None,
) -> BacktestResult:
    """
    Cost-aware hypothesis-test backtest of the VRP signal.

    Parameters
    ----------
    forecasts        : WFResult.forecasts DataFrame (actual, har, vix columns)
    cfg              : full Config
    upper_threshold  : VRP level to trigger short-vol (default: VRP 75th pct)
    lower_threshold  : VRP level to trigger long-vol  (default: VRP 25th pct)

    Returns
    -------
    BacktestResult

    P&L model (variance-space, proxy):
      gross_pnl_t = signal_{t-1} * (vrp_entry_approx - (actual_t - vix_t))
                  ≈ signal_{t-1} * (vix_t - actual_t)
      This is: short vol gains when implied > realized.
      Actual option P&L scales with vega and vol-of-vol; this is a linear proxy.
    """
    vrp = build_vrp(forecasts, cfg)

    # Default thresholds: upper/lower quartile of VRP distribution
    valid_vrp = vrp.dropna()
    if upper_threshold is None:
        upper_threshold = float(valid_vrp.quantile(0.75))
    if lower_threshold is None:
        lower_threshold = float(valid_vrp.quantile(0.25))

    log.info(
        "VRP thresholds: upper=%.5f  lower=%.5f", upper_threshold, lower_threshold
    )

    signal = generate_signal(vrp, upper_threshold, lower_threshold)

    # Gross P&L: position * (implied_var - realized_var) on that day
    # i.e. signal at t-1 (already shifted) × (vix_t - actual_t)
    implied_var = forecasts["vix"]
    actual_var  = forecasts["actual"]
    pnl_per_unit = implied_var - actual_var    # positive when vol seller wins

    gross_pnl = signal * pnl_per_unit

    # Costs
    tc = _daily_transaction_cost(signal, cfg)
    fc = _daily_financing_cost(signal, cfg)
    total_cost = tc + fc

    net_pnl = gross_pnl - total_cost
    net_pnl = net_pnl.fillna(0.0)

    equity = net_pnl.cumsum()

    stats = _compute_stats(net_pnl, gross_pnl, signal, vrp, upper_threshold, lower_threshold)

    return BacktestResult(
        daily_pnl=net_pnl,
        gross_pnl=gross_pnl.fillna(0.0),
        costs=total_cost,
        signal=signal,
        vrp=vrp,
        equity_curve=equity,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _compute_stats(
    net_pnl: pd.Series,
    gross_pnl: pd.Series,
    signal: pd.Series,
    vrp: pd.Series,
    upper_thr: float,
    lower_thr: float,
) -> dict:
    active = net_pnl[signal != 0]
    n_days_active = len(active)
    n_days_total  = len(net_pnl)

    total_net   = float(net_pnl.sum())
    total_gross = float(gross_pnl.sum())
    total_costs = float((gross_pnl - net_pnl).sum())

    ann_net    = float(net_pnl.mean() * 252)
    ann_vol    = float(net_pnl.std() * np.sqrt(252))
    sharpe     = ann_net / ann_vol if ann_vol > 0 else np.nan

    # Drawdown
    cum = net_pnl.cumsum()
    roll_max = cum.cummax()
    dd = cum - roll_max
    max_dd = float(dd.min())

    # Worst single day
    worst_day_val = float(net_pnl.min())
    worst_day_date = net_pnl.idxmin()

    # Win rate (active days only)
    win_rate = float((active > 0).mean()) if len(active) > 0 else np.nan

    # Short-vol specific: how often implied > realized (unconditional)
    pct_vrp_positive = float((vrp.dropna() > 0).mean())

    return {
        "upper_threshold": round(upper_thr, 5),
        "lower_threshold": round(lower_thr, 5),
        "n_days_total": n_days_total,
        "n_days_active": n_days_active,
        "pct_active": round(100 * n_days_active / n_days_total, 1),
        "total_net_pnl": round(total_net, 5),
        "total_gross_pnl": round(total_gross, 5),
        "total_costs": round(total_costs, 5),
        "ann_net_pnl": round(ann_net, 5),
        "ann_vol_pnl": round(ann_vol, 5),
        "sharpe_ratio": round(sharpe, 3) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": round(max_dd, 5),
        "worst_day_pnl": round(worst_day_val, 5),
        "worst_day_date": str(worst_day_date.date()) if hasattr(worst_day_date, "date") else str(worst_day_date),
        "win_rate_active": round(win_rate, 3) if np.isfinite(win_rate) else np.nan,
        "pct_vrp_positive": round(100 * pct_vrp_positive, 1),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_backtest(result: BacktestResult, output_dir: str | Path = "outputs/figures") -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(13, 14))

    # Panel 1: VRP over time with threshold bands
    ax = axes[0]
    ax.plot(result.vrp.index, result.vrp.values, lw=0.6, color="steelblue", label="VRP")
    ax.axhline(result.stats["upper_threshold"], color="green", lw=1, linestyle="--",
               label=f"Short-vol threshold ({result.stats['upper_threshold']:.4f})")
    ax.axhline(result.stats["lower_threshold"], color="red", lw=1, linestyle="--",
               label=f"Long-vol threshold ({result.stats['lower_threshold']:.4f})")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("VRP (ann. variance)")
    ax.set_title("Variance Risk Premium: VIX² − HAR Forecast")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 2: signal
    ax2 = axes[1]
    ax2.fill_between(result.signal.index, result.signal.values, 0,
                     where=result.signal > 0, alpha=0.5, color="green", label="Short vol (+1)")
    ax2.fill_between(result.signal.index, result.signal.values, 0,
                     where=result.signal < 0, alpha=0.5, color="red", label="Long vol (−1)")
    ax2.set_ylabel("Position")
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title("VRP Signal (1-day lag applied)")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 3: equity curve (net of costs)
    ax3 = axes[2]
    ax3.plot(result.equity_curve.index, result.equity_curve.values, lw=0.9, color="black",
             label="Net equity curve")
    ax3.plot(result.gross_pnl.cumsum().index, result.gross_pnl.cumsum().values,
             lw=0.7, color="grey", linestyle="--", label="Gross (pre-cost)")
    # Shade drawdown
    cum = result.equity_curve
    roll_max = cum.cummax()
    ax3.fill_between(cum.index, cum.values, roll_max.values, alpha=0.25, color="red",
                     label="Drawdown")
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_ylabel("Cumulative P&L (ann. variance units)")
    ax3.set_title(f"Equity Curve — Net of Costs  "
                  f"(Sharpe={result.stats['sharpe_ratio']:.2f}, "
                  f"MaxDD={result.stats['max_drawdown']:.4f})")
    ax3.legend(fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel 4: daily P&L histogram
    ax4 = axes[3]
    pnl_active = result.daily_pnl[result.signal != 0]
    ax4.hist(pnl_active.values, bins=80, color="steelblue", edgecolor="none", alpha=0.7)
    ax4.axvline(result.stats["worst_day_pnl"], color="red", lw=1.5,
                label=f"Worst day: {result.stats['worst_day_pnl']:.4f} ({result.stats['worst_day_date']})")
    ax4.axvline(0, color="black", lw=0.7)
    ax4.set_xlabel("Daily P&L (ann. variance units)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Daily P&L Distribution (active days only) — left tail is the risk")
    ax4.legend(fontsize=8)

    fig.suptitle(
        "VRP Signal Backtest — HYPOTHESIS TEST ONLY\n"
        "Not a trading recommendation. See assumptions in src/vrp_signal.py.",
        fontsize=9, color="darkred", y=1.01,
    )
    fig.tight_layout()
    out = Path(output_dir) / "vrp_backtest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved backtest plot to %s", out)
    return out
