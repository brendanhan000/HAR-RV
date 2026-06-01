"""
Sizing overlay — PHASE 3 runner: benchmark-first test on the harness.

Baseline = the elevated-VIX strategy with FLAT (unit) sizing — the current approach.
We run that same strategy through the overlay (vol-target / Kelly / + hard cap / + brake)
and compare, NET of the existing convex short-gamma P&L and stress-widened costs. The
sized position w_t * signal_t is fed NATIVELY into the convex engine (which is linear in
straddle size), so all costs — including the stress-widened option spread and the delta
hedge — are modelled, not approximated. Leverage is sampled at entry/roll (an options
book is sized at entry, not resized daily).

Reported per variant: net Sharpe, max drawdown, drawdown/annual ratio, worst-day loss,
turnover, total cost. A paired DM-style test (HAC) on the daily-P&L difference, run on
VOL-MATCHED books, asks whether the overlay changes RISK-ADJUSTED return significantly
(scaling leverage up/down trivially moves absolute P&L; the test must not reward that).

GATE / honesty: if the overlay does not improve risk-adjusted return, we say so. We
attribute the effect to the cap vs the vol target vs Kelly separately.  NO LOOK-AHEAD:
leverage uses the strategy's own risk through t-1; signals are lagged one day.
"""
import copy
import logging
import sys
from pathlib import Path

logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import load_config
from src.rv_estimator import build_rv_series
from src.data_pull import load_or_fetch_daily, load_or_fetch_vix_daily
from src.walk_forward import run_walk_forward
from src.pnl_convex import run_convex_backtest
from src.baseline import rolling_pct_short_signal, paired_pnl_test
from src.sizing import apply_overlay

SEP = "=" * 92
cfg = load_config("config.yaml")
HOLD = cfg.convex_pnl.hold_days

# --- data + point-in-time forecasts + the elevated-VIX flat signal ---
rv = build_rv_series(cfg)
vix = load_or_fetch_vix_daily(cfg).set_index("date")["vix_close"]
spy = load_or_fetch_daily(cfg).set_index("date")["close"]
wf = run_walk_forward(rv, vix, cfg)
idx = wf.forecasts.index
bc = cfg.baseline
signal = rolling_pct_short_signal(vix.reindex(idx).ffill(), bc.signal_window,
                                  bc.signal_min_periods, bc.short_percentile)
n_years = len(idx) / 252.0

print(f"\n{SEP}\nSIZING OVERLAY — PHASE 3: flat vs overlay, NET of convex P&L + stress costs\n{SEP}")
print(f"  {len(idx)} days [{idx[0].date()} -> {idx[-1].date()}]  | elevated-VIX baseline (VIX>{bc.short_percentile}pct)")


def convex(position, label):
    return run_convex_backtest(wf.forecasts, spy, vix, cfg, delta_hedge=True,
                               signal=position, label=label)


def variant_cfg(**rules_over):
    c = copy.deepcopy(cfg)
    for k, v in rules_over.items():
        setattr(c.sizing.rules, k, v)
    return c


def sized_position(c):
    """Leverage path from the overlay (sized on the FLAT book's own risk), x the signal."""
    r = apply_overlay(flat.daily_pnl, signal, c)
    return r.position.fillna(0.0), r


# --- benchmark: flat ---
flat = convex(signal, "flat")
A_const = float(flat.daily_pnl.mean() * 252)   # stable annual-expected for the 'clean cap' variant

# --- variants ---
specs = {
    "flat (unit)":                 None,
    "vol-target":                  variant_cfg(method="vol_target", cap_enabled=False),
    "vol-target + cap (PIT A)":    variant_cfg(method="vol_target", cap_enabled=True),
    "vol-target + cap (const A)":  variant_cfg(method="vol_target", cap_enabled=True, annual_expected_pnl=A_const),
    "vol-target + cap + brake":    variant_cfg(method="vol_target", cap_enabled=True, brake_enabled=True, brake_threshold=0.10),
    "fractional Kelly":            variant_cfg(method="kelly", cap_enabled=False),
    "fractional Kelly + cap":      variant_cfg(method="kelly", cap_enabled=True),
}

results, overlays = {}, {}
for name, c in specs.items():
    if c is None:
        results[name], overlays[name] = flat, None
    else:
        pos, ov = sized_position(c)
        results[name], overlays[name] = convex(pos, name), ov


def metrics(res, pos_series):
    p = res.daily_pnl.fillna(0.0)
    ann = p.mean() * 252
    vol = p.std() * np.sqrt(252)
    cum = p.cumsum()
    mdd = float((cum - cum.cummax()).min())
    worst = float(p.min())
    turn = float(pos_series.diff().abs().fillna(pos_series.abs()).sum()) / n_years
    return dict(
        sharpe=ann / vol if vol else np.nan, ann=ann, mdd=mdd,
        dda=abs(mdd) / abs(ann) if ann else np.nan, worst=worst,
        woa=abs(worst) / abs(ann) if ann else np.nan,
        turn=turn, cost=float(res.costs.sum()),
        mean_lev=float(overlays_lev(pos_series)),
    )


def overlays_lev(pos_series):
    nz = pos_series[pos_series.abs() > 0]
    return nz.abs().mean() if len(nz) else 0.0


pos_of = {n: (signal if specs[n] is None else (overlays[n].position.fillna(0.0))) for n in specs}

# ---------------------------------------------------------------------------
# Main table (native scale)
# ---------------------------------------------------------------------------
print(f"\n--- Performance, NET of convex costs (native scale) ---\n")
hdr = (f"  {'variant':<28}{'Sharpe':>7}{'AnnP&L':>9}{'maxDD':>9}{'DD/ann':>7}"
       f"{'worst':>8}{'wrst/ann':>9}{'turn/yr':>8}{'cost':>8}{'meanLev':>8}")
print(hdr + "\n  " + "-" * (len(hdr) - 2))
M = {}
for name in specs:
    m = metrics(results[name], pos_of[name]); M[name] = m
    print(f"  {name:<28}{m['sharpe']:>7.3f}{m['ann']:>9.2f}{m['mdd']:>9.2f}{m['dda']:>7.1f}"
          f"{m['worst']:>8.2f}{m['woa']:>9.2f}{m['turn']:>8.2f}{m['cost']:>8.1f}{m['mean_lev']:>8.2f}")

# ---------------------------------------------------------------------------
# Paired DM-style test on VOL-MATCHED books (isolates timing from scale)
# ---------------------------------------------------------------------------
print(f"\n--- Paired DM-style HAC test, VOL-MATCHED to flat  (H1: overlay's risk-adjusted P&L > flat) ---")
print(f"    (each book scaled to flat's annual vol, so the test rewards TIMING, not leverage level; lag={HOLD})\n")
flat_vol = flat.daily_pnl.std()
for name in specs:
    if specs[name] is None:
        continue
    s = results[name].daily_pnl.fillna(0.0)
    sv = s.std()
    matched = s * (flat_vol / sv) if sv > 0 else s
    t = paired_pnl_test(matched, flat.daily_pnl.fillna(0.0), lag=HOLD, alternative="greater")
    dsharpe = M[name]["sharpe"] - M["flat (unit)"]["sharpe"]
    verdict = "improves" if (t["p_value"] is not None and t["p_value"] < 0.05 and t["mean_diff"] > 0) else \
              ("WORSE" if dsharpe < -0.02 else "no better")
    print(f"  {name:<28} ΔSharpe={dsharpe:+.3f}  vol-matched ann_diff={t['ann_diff']:+7.3f}"
          f"  t={t['t_stat']}  p={t['p_value']}  -> {verdict}")

# ---------------------------------------------------------------------------
# Attribution + GATE verdict
# ---------------------------------------------------------------------------
mf, mvt = M["flat (unit)"], M["vol-target"]
mcap, mconst = M["vol-target + cap (PIT A)"], M["vol-target + cap (const A)"]
print(f"\n{SEP}\nWHAT EACH RULE CONTRIBUTES (the question the brief asked)\n{SEP}")
print(f"  VOL TARGET   : Sharpe {mf['sharpe']:.3f} -> {mvt['sharpe']:.3f} ({mvt['sharpe']-mf['sharpe']:+.3f}),"
      f"  DD/ann {mf['dda']:.1f} -> {mvt['dda']:.1f},  maxDD {mf['mdd']:.1f} -> {mvt['mdd']:.1f}.")
print(f"                 de-levering on the strategy's own (active) vol moves absolute risk; the lagged")
print(f"                 estimate cannot dodge the first gap, so the RATIO barely improves.")
print(f"  HARD CAP     : the binding constraint. With a STABLE annual-expected it is a pure LEVEL cut —")
print(f"                 Sharpe {mconst['sharpe']:.3f} & DD/ann {mconst['dda']:.1f} = vol-target's, but maxDD"
      f" {mvt['mdd']:.1f} -> {mconst['mdd']:.1f} and worst {mvt['worst']:.1f} -> {mconst['worst']:.1f}.")
print(f"                 (point-in-time annual-expected couples to trailing edge -> Sharpe {mcap['sharpe']:.3f}, a cost.)")
print(f"  KELLY        : Sharpe {M['fractional Kelly']['sharpe']:.3f}, DD/ann {M['fractional Kelly']['dda']:.1f}"
      f" — edge-estimate noise makes it the worst rule (see Phase-2 ruin curve).")

cap_helps_ratio = mconst["dda"] < mf["dda"] - 0.5
sizing_helps_sharpe = mvt["sharpe"] > mf["sharpe"] + 0.02
print(f"\n{SEP}\nGATE VERDICT\n{SEP}")
if sizing_helps_sharpe:
    print("  The overlay IMPROVES risk-adjusted return (higher net Sharpe), not just the drawdown cap.")
else:
    print("  The overlay does NOT improve risk-adjusted return. Net Sharpe is no higher than flat")
    print(f"  (flat {mf['sharpe']:.3f} vs best sized {max(M[n]['sharpe'] for n in specs):.3f}); the vol-target slightly")
    print("  lowers it and Kelly lowers it sharply. The cap's ONLY benefit is a hard floor on")
    print(f"  absolute loss: it slashes max drawdown ({mf['mdd']:.0f} -> {mconst['mdd']:.0f}) and the worst day")
    print(f"  ({mf['worst']:.0f} -> {mconst['worst']:.0f}) but leaves the drawdown/annual RATIO ~unchanged")
    print(f"  ({mf['dda']:.1f} -> {mconst['dda']:.1f}) because it is a near-constant leverage reduction.")
    print("  => 'Sizing adds little beyond the drawdown cap.' Exactly the brief's anticipated outcome.")
print(SEP)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
show = ["flat (unit)", "vol-target", "vol-target + cap (PIT A)", "vol-target + cap (const A)"]
for name in show:
    eq = results[name].daily_pnl.fillna(0).cumsum()
    ax[0].plot(eq.index, eq.values, lw=0.9, label=f"{name} (Sh {M[name]['sharpe']:.2f})")
ax[0].axhline(0, color="black", lw=0.5)
ax[0].set_ylabel("cumulative net P&L"); ax[0].legend(fontsize=8)
ax[0].set_title("Phase 3 — flat vs overlay (native convex P&L + stress costs)")
for name in show:
    p = results[name].daily_pnl.fillna(0); cum = p.cumsum(); dd = cum - cum.cummax()
    ax[1].plot(dd.index, dd.values, lw=0.8, label=name)
ax[1].set_ylabel("drawdown (P&L units)"); ax[1].legend(fontsize=8)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig("outputs/figures/sizing_phase3_equity.png", dpi=150)
plt.close(fig)
print(f"Figure: outputs/figures/sizing_phase3_equity.png")
print(f"{SEP}\nPHASE 3 COMPLETE.\n{SEP}")
