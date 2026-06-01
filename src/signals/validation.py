"""
Shared validation utilities for the four-signal adversarial study.

Everything here is signal-agnostic: a strategy is just a per-period return stream
(+ optional position series). The point of the module is to make the HONEST
referee identical across signals — same holdout discipline, same metrics, same
multiple-testing correction — so no signal gets a softer test than another.

Key pieces:
  split_dev_holdout      — strict never-seen holdout (dev = contaminated/in-sample)
  perf_metrics           — net Sharpe, maxDD, DD/ann, worst day, SKEW, hit rate, turnover
  worst_sustained_excursion — max adverse run & underwater duration (the tail that
                              kills mean-reversion when a spread stops reverting)
  block_bootstrap_sharpe_p  — p-value for Sharpe>0 under a stationary block bootstrap
                              (handles autocorrelation + small holdout samples honestly)
  benjamini_hochberg     — multiple-comparison correction across the four signals
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Holdout discipline
# ---------------------------------------------------------------------------


def split_dev_holdout(obj, holdout_start: str):
    """
    Split a Series/DataFrame at `holdout_start` (inclusive) into (dev, holdout).
    dev  = everything strictly before holdout_start  -> treat as IN-SAMPLE/contaminated.
    holdout = holdout_start onward                   -> STRICT never-seen OOS.
    """
    ts = pd.Timestamp(holdout_start)
    dev = obj[obj.index < ts]
    holdout = obj[obj.index >= ts]
    return dev, holdout


# ---------------------------------------------------------------------------
# Performance metrics (on a per-period RETURN stream; arithmetic, dollar-neutral OK)
# ---------------------------------------------------------------------------


def perf_metrics(
    returns: pd.Series,
    position: Optional[pd.Series] = None,
    periods_per_year: int = TRADING_DAYS,
    cost_drag: Optional[pd.Series] = None,
) -> dict:
    """
    Summary stats for a return stream. Equity is cumulative (additive) P&L — the
    right convention for a dollar-neutral spread / overlay return per unit notional.

    Returns a dict: n, n_active, ann_return, ann_vol, sharpe, max_drawdown,
    dd_over_ann (|maxDD|/annual return), worst_day, skew, kurtosis, hit_rate (active
    days), turnover_per_year (if position given), cost_drag_per_year (if given).
    """
    r = returns.fillna(0.0).astype(float)
    n = int((r != 0).count())
    active = r[r != 0] if position is None else r[position.reindex(r.index).fillna(0) != 0]
    ann_return = float(r.mean() * periods_per_year)
    ann_vol = float(r.std() * np.sqrt(periods_per_year))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    eq = r.cumsum()
    mdd = float((eq - eq.cummax()).min())
    worst = float(r.min())
    n_years = len(r) / periods_per_year
    out = {
        "n_days": len(r),
        "n_active": int((active != 0).sum()),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "dd_over_ann": abs(mdd) / abs(ann_return) if ann_return != 0 else np.nan,
        "worst_day": worst,
        "skew": float(stats.skew(r.values)) if r.std() > 0 else np.nan,
        "kurtosis": float(stats.kurtosis(r.values)) if r.std() > 0 else np.nan,
        "hit_rate": float((active > 0).mean()) if len(active) else np.nan,
    }
    if position is not None:
        pos = position.reindex(r.index).fillna(0.0)
        out["turnover_per_year"] = float(pos.diff().abs().sum()) / n_years if n_years else np.nan
        out["pct_in_market"] = float((pos != 0).mean())
    if cost_drag is not None:
        out["cost_drag_per_year"] = float(cost_drag.reindex(r.index).fillna(0).sum()) / n_years if n_years else np.nan
    return out


# ---------------------------------------------------------------------------
# Tail: worst sustained adverse excursion (the mean-reversion killer)
# ---------------------------------------------------------------------------


def worst_sustained_excursion(returns: pd.Series, position: Optional[pd.Series] = None) -> dict:
    """
    Characterise the *sustained* tail, not just the worst day — a spread that stops
    reverting runs against you for a long time.

    Returns:
      max_drawdown            : deepest peak-to-trough of cumulative P&L
      max_drawdown_days       : longest stretch below a prior high-water mark
      worst_trade_mae         : worst within-trade adverse excursion (if `position` given;
                                a trade = a maximal run of constant non-zero position)
      worst_trade_days        : duration of that worst trade
    """
    r = returns.fillna(0.0).astype(float)
    eq = r.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = float(dd.min())
    # longest underwater stretch
    underwater = dd < -1e-12
    longest, cur = 0, 0
    for u in underwater.values:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)

    out = {"max_drawdown": max_dd, "max_drawdown_days": int(longest)}

    if position is not None:
        pos = position.reindex(r.index).fillna(0.0).values
        rr = r.values
        worst_mae, worst_dur = 0.0, 0
        i = 0
        while i < len(pos):
            if pos[i] == 0:
                i += 1
                continue
            j = i
            cum = 0.0
            mae = 0.0
            while j < len(pos) and pos[j] == pos[i]:
                cum += rr[j]
                mae = min(mae, cum)        # worst cumulative P&L reached during the trade
                j += 1
            if mae < worst_mae:
                worst_mae, worst_dur = mae, j - i
            i = j
        out["worst_trade_mae"] = float(worst_mae)
        out["worst_trade_days"] = int(worst_dur)
    return out


# ---------------------------------------------------------------------------
# Significance: block bootstrap for Sharpe>0 (autocorrelation-robust, small-sample honest)
# ---------------------------------------------------------------------------


def block_bootstrap_sharpe_p(
    returns: pd.Series,
    n_boot: int = 5000,
    block: int = 10,
    seed: int = 0,
) -> dict:
    """
    Stationary block-bootstrap p-value for H0: Sharpe <= 0 vs H1: Sharpe > 0.

    Resamples overlapping blocks (length `block`) to preserve autocorrelation, recenters
    each resample to mean zero (imposes the null), and asks how often the bootstrap Sharpe
    exceeds the observed Sharpe. Honest for short, autocorrelated holdout samples.
    """
    r = returns.fillna(0.0).astype(float).values
    n = len(r)
    if n < block * 3 or r.std() == 0:
        return {"sharpe": float(np.nan), "p_value": np.nan, "n": n}
    obs_sharpe = r.mean() / r.std() * np.sqrt(TRADING_DAYS)
    rng = np.random.default_rng(seed)
    centered = r - r.mean()                 # impose H0: zero mean
    n_blocks = int(np.ceil(n / block))
    count = 0
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        sample = centered[idx[:n]]
        s = sample.mean() / sample.std() * np.sqrt(TRADING_DAYS) if sample.std() > 0 else 0.0
        if s >= obs_sharpe:
            count += 1
    return {"sharpe": float(obs_sharpe), "p_value": (count + 1) / (n_boot + 1), "n": n}


# ---------------------------------------------------------------------------
# Multiple-testing correction across the four signals
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvalues: Dict[str, float], fdr: float = 0.05) -> dict:
    """
    Benjamini-Hochberg FDR control. Given a dict {name: p}, return per-name
    {p, bh_threshold, significant} and the overall largest p that passes.

    With k tests, the i-th smallest p (rank i) passes if p_(i) <= (i/k)*fdr. Everything
    ranked at/below the largest passing rank is declared significant. This is the
    correct lens for "we tested 4 signals; ~1 looks good by chance."
    """
    items = [(k, v) for k, v in pvalues.items() if v is not None and np.isfinite(v)]
    k = len(items)
    if k == 0:
        return {"k": 0, "fdr": fdr, "crit_p": 0.0, "results": {}}
    items.sort(key=lambda kv: kv[1])
    crit_rank = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= (i / k) * fdr:
            crit_rank = i
    crit_p = (crit_rank / k) * fdr if crit_rank else (1 / k) * fdr
    results = {}
    for i, (name, p) in enumerate(items, start=1):
        results[name] = {
            "p_value": p,
            "rank": i,
            "bh_bound": (i / k) * fdr,
            "significant": i <= crit_rank,
        }
    return {"k": k, "fdr": fdr, "crit_p": crit_p, "n_significant": crit_rank, "results": results}
