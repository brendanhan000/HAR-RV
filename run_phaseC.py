"""
Phase C runner — does the HAR-VRP signal beat the naive "sell vol when VIX is
elevated" baseline, net of the convex P&L + costs from Phase A?

This is THE GATE. Both strategies:
  * use the SAME convex delta-hedged short-straddle P&L and the SAME costs;
  * use ROLLING, point-in-time percentile thresholds (data <= t) — no full-sample
    look-ahead for either side;
  * are compared on net Sharpe, max drawdown, drawdown/annual ratio, worst day, and a
    Diebold-Mariano-style HAC test on the daily P&L difference.

If HAR-VRP does not beat the baseline net of convex costs, the signal adds no reliable
incremental edge and we recommend trading the baseline (or nothing).  HYPOTHESIS TEST.
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
from src.data_pull import load_or_fetch_daily, load_or_fetch_vix_daily
from src.walk_forward import run_walk_forward
from src.vrp_signal import build_vrp
from src.pnl_convex import run_convex_backtest
from src.baseline import (
    rolling_pct_short_signal,
    rolling_pct_symmetric_signal,
    paired_pnl_test,
)

SEP = "=" * 80
cfg = load_config("config.yaml")
bc = cfg.baseline
W, MP = bc.signal_window, bc.signal_min_periods

print(f"\n{SEP}")
print("PHASE C — HAR-VRP signal vs the elevated-VIX baseline  [THE GATE]")
print(f"  convex P&L + costs (Phase A) | rolling {W}d / min {MP}d point-in-time percentiles")
print(SEP)

# --- data + point-in-time forecasts ---
rv = build_rv_series(cfg)
vix = load_or_fetch_vix_daily(cfg).set_index("date")["vix_close"]
spy = load_or_fetch_daily(cfg).set_index("date")["close"]
wf = run_walk_forward(rv, vix, cfg)
idx = wf.forecasts.index
print(f"Forecast periods: {len(idx)}  ({idx[0].date()} → {idx[-1].date()})")

# --- the two signal variables (both point-in-time) ---
vrp = build_vrp(wf.forecasts, cfg)              # VIX^2 - HAR forecast (HAR is walk-forward)
vix_lvl = vix.reindex(idx).ffill()              # raw VIX level for the baseline

HOLD = cfg.convex_pnl.hold_days                 # HAC lag for the paired test (overlap)


def stats(pnl: pd.Series, signal: pd.Series) -> dict:
    pnl = pnl.fillna(0.0)
    ann = float(pnl.mean() * 252)
    vol = float(pnl.std() * np.sqrt(252))
    sd = float(pnl.std())
    cum = pnl.cumsum()
    mdd = float((cum - cum.cummax()).min())
    worst = float(pnl.min())
    return {
        "pct_short": round(100 * float((signal > 0).mean()), 1),
        "pct_active": round(100 * float((signal != 0).mean()), 1),
        "sharpe": ann / vol if vol else float("nan"),
        "ann": ann,
        "mdd": mdd,
        "dd_over_ann": abs(mdd) / abs(ann) if ann else float("nan"),
        "worst": worst,
        "worst_z": worst / sd if sd else float("nan"),
    }


def run(signal, label):
    res = run_convex_backtest(wf.forecasts, spy, vix, cfg, delta_hedge=True,
                              signal=signal, label=label)
    return res


# --- head-to-head at the configured percentile (short/flat for both) ---
P = bc.short_percentile
sig_base = rolling_pct_short_signal(vix_lvl, W, MP, P)            # elevated-VIX baseline
sig_vrp = rolling_pct_short_signal(vrp, W, MP, P)                # HAR-VRP, short/flat
sig_vrp_sym = rolling_pct_symmetric_signal(vrp, W, MP, P, bc.long_percentile)  # + long leg

res_base = run(sig_base, "elevated_vix")
res_vrp = run(sig_vrp, "har_vrp_short")
res_sym = run(sig_vrp_sym, "har_vrp_symmetric")

rows = [
    (f"Elevated-VIX baseline (VIX>{P}pct)", res_base, sig_base),
    (f"HAR-VRP short/flat   (VRP>{P}pct)", res_vrp, sig_vrp),
    (f"HAR-VRP symmetric    (+long leg)", res_sym, sig_vrp_sym),
]

print(f"\n--- Head-to-head, NET of convex costs (same P&L engine, same costs) ---\n")
hdr = (f"  {'strategy':<36}{'%short':>7}{'Sharpe':>8}{'AnnP&L':>9}"
       f"{'MaxDD':>9}{'DD/Ann':>8}{'WorstDay':>10}")
print(hdr + "\n  " + "-" * (len(hdr) - 2))
for name, res, sig in rows:
    s = stats(res.daily_pnl, sig)
    print(f"  {name:<36}{s['pct_short']:>6.0f}%{s['sharpe']:>8.2f}{s['ann']:>9.2f}"
          f"{s['mdd']:>9.1f}{s['dd_over_ann']:>8.1f}{s['worst']:>10.2f}")

# also show GROSS, since net is cost-dominated and both pay similar costs
print(f"\n--- Same, GROSS (pre-cost) — isolates signal quality ---\n")
print(hdr + "\n  " + "-" * (len(hdr) - 2))
for name, res, sig in rows:
    s = stats(res.gross_pnl, sig)
    print(f"  {name:<36}{s['pct_short']:>6.0f}%{s['sharpe']:>8.2f}{s['ann']:>9.2f}"
          f"{s['mdd']:>9.1f}{s['dd_over_ann']:>8.1f}{s['worst']:>10.2f}")

# --- Diebold-Mariano-style paired tests: does HAR-VRP beat the baseline? ---
print(f"\n--- Paired HAC test  H1: strategy mean daily P&L > baseline  (lag={HOLD}) ---")
for tag, res in [("HAR-VRP short/flat", res_vrp), ("HAR-VRP symmetric", res_sym)]:
    for kind, a, b in [("NET", res.daily_pnl, res_base.daily_pnl),
                       ("GROSS", res.gross_pnl, res_base.gross_pnl)]:
        t = paired_pnl_test(a, b, lag=HOLD, alternative="greater")
        verdict = ("beats baseline" if (t["p_value"] is not None and t["p_value"] < 0.05
                   and t["mean_diff"] > 0) else "does NOT beat baseline")
        print(f"  {tag:<20} {kind:<5}: ann_diff={t['ann_diff']:+8.3f}  "
              f"t={t['t_stat']}  p={t['p_value']}  -> {verdict}")

# --- robustness: sweep the short percentile for BOTH (matched activity) ---
print(f"\n--- Percentile sweep (NET Sharpe | DD/Ann) — VIX baseline vs HAR-VRP short/flat ---\n")
print(f"  {'short pct':>9}{'  base %short':>13}{'  base Sharpe':>13}{'  VRP Sharpe':>13}"
      f"{'  base DD/Ann':>13}{'  VRP DD/Ann':>12}")
print("  " + "-" * 72)
for p in [50, 60, 70, 75, 80, 90]:
    sb = rolling_pct_short_signal(vix_lvl, W, MP, p)
    sv = rolling_pct_short_signal(vrp, W, MP, p)
    rb = run(sb, f"vix{p}")
    rvp = run(sv, f"vrp{p}")
    Sb, Sv = stats(rb.daily_pnl, sb), stats(rvp.daily_pnl, sv)
    print(f"  {p:>8}%{Sb['pct_short']:>12.0f}%{Sb['sharpe']:>13.2f}{Sv['sharpe']:>13.2f}"
          f"{Sb['dd_over_ann']:>13.1f}{Sv['dd_over_ann']:>12.1f}")

# --- crisis behavior of the baseline (does shorting elevated VIX blow up too?) ---
print(f"\n--- Crisis net P&L (own units): baseline vs HAR-VRP short/flat ---")
for lbl, lo, hi in [("2008 GFC (Sep–Dec)", "2008-09-01", "2008-12-31"),
                    ("2020 COVID (Feb–Apr)", "2020-02-15", "2020-04-30")]:
    b = res_base.daily_pnl.loc[lo:hi]
    v = res_vrp.daily_pnl.loc[lo:hi]
    print(f"  {lbl:<22} baseline net={b.sum():+8.2f} (worst {b.min():+7.2f})   "
          f"HAR-VRP net={v.sum():+8.2f} (worst {v.min():+7.2f})")

# --- GATE verdict ---
sb = stats(res_base.daily_pnl, sig_base)
sv = stats(res_vrp.daily_pnl, sig_vrp)
dm = paired_pnl_test(res_vrp.daily_pnl, res_base.daily_pnl, lag=HOLD, alternative="greater")
beats = (sv["sharpe"] > sb["sharpe"]) and (dm["p_value"] is not None and dm["p_value"] < 0.05)

print(f"\n{SEP}\nGATE VERDICT\n{SEP}")
if beats:
    print("  HAR-VRP BEATS the elevated-VIX baseline net of convex costs (higher Sharpe")
    print("  AND a statistically significant P&L advantage). The signal adds incremental")
    print("  edge — proceed to Phase D (tail management).")
else:
    print("  HAR-VRP does NOT beat the elevated-VIX baseline net of convex costs.")
    print(f"    net Sharpe: HAR-VRP {sv['sharpe']:+.2f}  vs  baseline {sb['sharpe']:+.2f}")
    print(f"    paired test: ann_diff={dm['ann_diff']:+.3f}  p={dm['p_value']} (not significant > 0)")
    print("  The HAR-VRP signal adds NO reliable incremental edge over selling elevated")
    print("  VIX. Recommendation: trade the simpler baseline (or nothing). Do NOT proceed")
    print("  to Phase D to rescue the signal.")
print(SEP)
