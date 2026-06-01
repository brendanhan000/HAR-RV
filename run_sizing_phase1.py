"""
Sizing overlay — PHASE 1 runner: point-in-time risk estimation.

Demonstrates the risk estimators on the REAL elevated-VIX baseline return stream
(the strategy whose risk/sizing was the defining failure of the prior project):
  * EWMA vs rolling-window annualised volatility of the strategy's OWN P&L;
  * a no-look-ahead demonstration on real data (corrupt the future, prove the
    present estimate is byte-identical);
  * warm-up / alignment;
  * a figure of |P&L| with both vol estimates over 2004-2024.

NO SIZING IS APPLIED HERE — Phase 1 is only the risk INPUT. Sizing rules are Phase 2.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import load_config
from src.sizing.risk import estimate_vol, realized_vol_ewma, realized_vol_rolling

SEP = "=" * 78
cfg = load_config("config.yaml")
rc = cfg.sizing.risk


def _load_baseline_pnl() -> pd.Series:
    """Load the cached elevated-VIX baseline net P&L; rebuild from source if missing."""
    cache = Path("data/raw/baseline_elevated_vix_pnl.parquet")
    if cache.exists():
        return pd.read_parquet(cache)["net"]
    # rebuild (≈2s)
    from src.rv_estimator import build_rv_series
    from src.data_pull import load_or_fetch_daily, load_or_fetch_vix_daily
    from src.walk_forward import run_walk_forward
    from src.pnl_convex import run_convex_backtest
    from src.baseline import rolling_pct_short_signal

    rv = build_rv_series(cfg)
    vix = load_or_fetch_vix_daily(cfg).set_index("date")["vix_close"]
    spy = load_or_fetch_daily(cfg).set_index("date")["close"]
    wf = run_walk_forward(rv, vix, cfg)
    idx = wf.forecasts.index
    bc = cfg.baseline
    sig = rolling_pct_short_signal(vix.reindex(idx).ffill(), bc.signal_window,
                                   bc.signal_min_periods, bc.short_percentile)
    res = run_convex_backtest(wf.forecasts, spy, vix, cfg, delta_hedge=True,
                              signal=sig, label="elevated_vix")
    out = pd.DataFrame({"net": res.daily_pnl, "gross": res.gross_pnl, "signal": res.signal})
    out.to_parquet(cache)
    return res.daily_pnl


pnl = _load_baseline_pnl().rename("ret")

print(f"\n{SEP}")
print("SIZING OVERLAY — PHASE 1: point-in-time risk estimation")
print(f"  return stream = elevated-VIX baseline NET P&L (vega_notional=1 units)")
print(f"  {len(pnl)} periods  [{pnl.index[0].date()} -> {pnl.index[-1].date()}]")
print(f"  estimator (config) = {rc.estimator!r}  | demean={rc.demean}  annualize={rc.annualize}")
print(f"  rolling: window={rc.window}, min_periods={rc.rolling_min_periods}")
print(f"  ewma:    halflife={rc.ewma_halflife}, lambda={rc.ewma_lambda}, min_periods={rc.ewma_min_periods}")
print(SEP)

# --- the two estimators (annualised), strictly point-in-time/lagged ---
sig_ewma = realized_vol_ewma(pnl, halflife=rc.ewma_halflife, min_periods=rc.ewma_min_periods,
                             demean=rc.demean, annualize=True, periods_per_year=rc.periods_per_year)
sig_roll = realized_vol_rolling(pnl, window=rc.window, min_periods=rc.rolling_min_periods,
                                demean=rc.demean, annualize=True, periods_per_year=rc.periods_per_year)
sig_cfg = estimate_vol(pnl, rc)   # config dispatcher (== one of the above)

# --- warm-up / alignment ---
print("\n--- Alignment & warm-up (NaN until enough history, then valid) ---")
print(f"  index identical to input : {sig_ewma.index.equals(pnl.index) and sig_roll.index.equals(pnl.index)}")
print(f"  EWMA  first valid at idx {sig_ewma.notna().idxmax()!s:.10}  "
      f"({int(sig_ewma.isna().sum())} leading NaN)")
print(f"  Roll  first valid at idx {sig_roll.notna().idxmax()!s:.10}  "
      f"({int(sig_roll.isna().sum())} leading NaN)")

# --- THE no-look-ahead demonstration, on real data ---
print("\n--- No-look-ahead demonstration (corrupt the future, inspect the present) ---")
k = len(pnl) - 60                       # corrupt the last 60 periods
pnl_bad = pnl.copy()
pnl_bad.iloc[k:] = -500.0               # inject a fake catastrophe
sig_bad = realized_vol_ewma(pnl_bad, halflife=rc.ewma_halflife, min_periods=rc.ewma_min_periods,
                            demean=rc.demean, annualize=True, periods_per_year=rc.periods_per_year)
# the estimate that SIZES period k uses only data < k -> identical despite r[k:] corruption
diff_through_k = (sig_ewma.iloc[: k + 1] - sig_bad.iloc[: k + 1]).abs().max()
diff_after_k = (sig_ewma.iloc[k + 1:] - sig_bad.iloc[k + 1:]).abs().max()
print(f"  corrupted r[{k}:] to -500 (a fake crash)")
print(f"  max |Δσ̂| over periods 0..k (these SIZE periods 0..k) = {diff_through_k:.3e}  -> unchanged ✓")
print(f"  max |Δσ̂| over periods k+1..end (these see r[k])       = {diff_after_k:.3e}  -> changed (correct)")
assert diff_through_k == 0.0, "LOOK-AHEAD LEAK"

# --- how the estimate tracks real vol regimes ---
print("\n--- Annualised σ̂ of the strategy's own P&L at key dates (EWMA | rolling) ---")
for d in ["2007-06-01", "2008-10-15", "2011-08-08", "2017-01-03", "2020-03-16", "2020-08-03"]:
    ts = pnl.index[pnl.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]]
    e = sig_ewma.get(ts, np.nan)
    r = sig_roll.get(ts, np.nan)
    print(f"  {ts.date()}:  EWMA={e:8.2f}   rolling={r:8.2f}")

q = sig_ewma.dropna()
print(f"\n  EWMA σ̂ distribution (ann.):  min={q.min():.2f}  median={q.median():.2f}  "
      f"p95={q.quantile(.95):.2f}  max={q.max():.2f}")
print(f"  ratio max/median = {q.max()/q.median():.1f}x  (the strategy's own risk is far from constant)")

# --- figure ---
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax[0].plot(pnl.index, pnl.values, lw=0.5, color="grey", alpha=0.8)
ax[0].axhline(0, color="black", lw=0.5)
ax[0].set_ylabel("daily net P&L\n(vega_notional=1)")
ax[0].set_title("Elevated-VIX baseline: own return stream and its point-in-time risk estimate")
ax[1].plot(sig_roll.index, sig_roll.values, lw=0.8, color="darkorange",
           label=f"rolling {rc.window}d (lagged)")
ax[1].plot(sig_ewma.index, sig_ewma.values, lw=0.9, color="steelblue",
           label=f"EWMA hl={rc.ewma_halflife} (lagged)")
ax[1].set_ylabel("annualised σ̂ of P&L")
ax[1].legend(fontsize=8)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
out = Path("outputs/figures/sizing_phase1_risk.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nFigure saved: {out}")
print(SEP)
print("PHASE 1 COMPLETE — risk INPUT only. No position sizing applied yet (that is Phase 2).")
print(SEP)
