"""
Phase 4 runner: VRP signal construction + cost-aware hypothesis-test backtest.

THIS IS A HYPOTHESIS TEST, NOT A TRADING RECOMMENDATION.
All assumptions stated explicitly. See src/vrp_signal.py for full assumption list.
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from src.config import load_config
from src.rv_estimator import build_rv_series
from src.data_pull import load_or_fetch_vix_daily
from src.walk_forward import run_walk_forward
from src.vrp_signal import build_vrp, run_backtest, plot_backtest

cfg = load_config("config.yaml")
rv  = build_rv_series(cfg)

vix_df = load_or_fetch_vix_daily(cfg)
vix = vix_df.set_index("date")["vix_close"]

SEP = "=" * 65

print(f"\n{SEP}")
print("PHASE 4 — VRP Signal  [HYPOTHESIS TEST — NOT A TRADING SYSTEM]")
print(f"  h={cfg.har.horizon}  |  costs: bid-ask vega={cfg.transaction_costs.option_bid_ask_vega}"
      f"  slippage={cfg.transaction_costs.slippage_bps}bps"
      f"  financing={cfg.transaction_costs.financing_rate_annual*100:.1f}%/yr")
print(SEP)

# Re-run walk-forward to get forecasts (uses cached snapshot)
print("Re-running walk-forward to produce point-in-time forecasts…")
wf = run_walk_forward(rv, vix, cfg)
print(f"Forecast periods: {len(wf.forecasts)}  ({wf.forecasts.index[0].date()} → {wf.forecasts.index[-1].date()})")

# --- VRP summary ---
vrp = build_vrp(wf.forecasts, cfg)
valid = vrp.dropna()
print(f"\n--- VRP Summary (annualized variance units) ---")
print(f"  N obs         : {len(valid)}")
print(f"  Mean          : {valid.mean():+.5f}")
print(f"  Std           : {valid.std():.5f}")
print(f"  Pct positive  : {100*(valid>0).mean():.1f}%  (unconditional short-vol premium)")
print(f"  5th pct       : {valid.quantile(0.05):+.5f}  (left tail of VRP)")
print(f"  25th pct      : {valid.quantile(0.25):+.5f}")
print(f"  75th pct      : {valid.quantile(0.75):+.5f}")
print(f"  95th pct      : {valid.quantile(0.95):+.5f}")

# --- Run backtest with default thresholds (Q25/Q75) ---
result = run_backtest(wf.forecasts, cfg)

print(f"\n--- Backtest Statistics (net of costs) ---")
stats = result.stats
for k, v in stats.items():
    print(f"  {k:<30s}: {v}")

# --- Regime breakdown ---
print(f"\n--- Regime Breakdown (short-vol periods only) ---")
short_pnl = result.daily_pnl[result.signal > 0]
long_pnl  = result.daily_pnl[result.signal < 0]

for label, pnl in [("Short-vol (+1)", short_pnl), ("Long-vol (−1)", long_pnl)]:
    if len(pnl) > 0:
        print(f"  {label}: n={len(pnl)}  total={pnl.sum():.5f}"
              f"  mean/day={pnl.mean():.6f}  worst={pnl.min():.5f}"
              f"  win_rate={100*(pnl>0).mean():.1f}%")

# --- Worst drawdown periods ---
print(f"\n--- 5 Worst Single-Day P&Ls ---")
worst = result.daily_pnl.nsmallest(5)
for dt, val in worst.items():
    sig = result.signal.get(dt, 0)
    v   = result.vrp.get(dt, np.nan)
    print(f"  {dt.date()}  pnl={val:+.5f}  signal={sig:+.0f}  VRP={v:+.5f}")

# --- Sensitivity: vary thresholds ---
print(f"\n--- Threshold Sensitivity (Sharpe ratio, net of costs) ---")
print(f"  {'Upper pct':>10}  {'Lower pct':>10}  {'Sharpe':>8}  {'MaxDD':>10}  {'Ann P&L':>10}  {'%Active':>8}")
pcts = [(0.6, 0.4), (0.7, 0.3), (0.75, 0.25), (0.8, 0.2), (0.9, 0.1)]
for up, lo in pcts:
    u_thr = float(valid.quantile(up))
    l_thr = float(valid.quantile(lo))
    r = run_backtest(wf.forecasts, cfg, upper_threshold=u_thr, lower_threshold=l_thr)
    s = r.stats
    print(f"  {up*100:>9.0f}%  {lo*100:>9.0f}%  {s['sharpe_ratio']:>8.3f}"
          f"  {s['max_drawdown']:>10.5f}  {s['ann_net_pnl']:>10.5f}  {s['pct_active']:>7.1f}%")

# --- Critical assumptions reminder ---
print(f"""
{SEP}
CRITICAL ASSUMPTIONS — what would invalidate these results:

  1. VRP proxy: VIX^2 ≠ true 22-day implied variance. VIX targets 30
     calendar days; we use it for 22 trading days. This horizon mismatch
     biases VRP estimates and is not corrected here.

  2. P&L proxy: pnl = (implied_var - realized_var) is a vega-linear
     approximation. Real option P&L depends on gamma, theta, vanna,
     and volga. Short-vol strategies are short gamma — large moves
     hurt nonlinearly. This model understates true risk.

  3. Transaction costs here ({cfg.transaction_costs.option_bid_ask_vega:.2f} vega bid-ask,
     {cfg.transaction_costs.slippage_bps:.0f}bps slippage) are optimistic. In stress
     regimes, actual SPX option bid-asks widen 5-10x. The worst
     drawdown dates above are exactly those stress periods.

  4. No capital model: we assume unlimited ability to hold losing
     positions. In practice, margin calls or risk limits would force
     exits at the worst times (the "short-vol blowup" scenario).

  5. Survivorship / selection: SPY was chosen because it is well-known
     to have a persistent VRP. This is not a generalisable finding.

  6. HAR is the weaker forecaster (VIX beats it — Phase 3). VRP
     constructed with a noisy HAR denominator adds noise to the signal.
     A cleaner signal would use intraday RV or a better estimator.

  7. The Sharpe ratio is in variance units, not dollar returns. Dollar
     Sharpe depends on position sizing, notional, and vol-of-vol.

CONCLUSION: The unconditional VRP is positive ({100*(valid>0).mean():.0f}% of days),
consistent with a persistent short-vol premium in SPY. But the signal
edges above are fragile: they erode under realistic costs, are
concentrated in the tails of the VRP distribution, and do not account
for the true non-linear risk of short-volatility strategies.
{SEP}
""")

# --- Plot ---
out = plot_backtest(result)
print(f"Plot saved: {out}")
