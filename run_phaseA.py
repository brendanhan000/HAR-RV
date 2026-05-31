"""
Phase A runner — convex (true short-gamma) P&L vs the vega-linear proxy.

Same VRP signal, same thresholds (Q25/Q75), same walk-forward forecasts as Phase 4.
ONLY the P&L model changes. Reports how Sharpe, max drawdown, worst-day loss and the
drawdown/annual-P&L ratio move when we price the actual short-straddle position with
Black-Scholes along the realized path instead of the linear implied-minus-realized proxy.

  HYPOTHESIS TEST — NOT A TRADING SYSTEM.
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
from src.vrp_signal import run_backtest as run_linear_backtest
from src.pnl_convex import run_convex_backtest

SEP = "=" * 78
cfg = load_config("config.yaml")

print(f"\n{SEP}")
print("PHASE A — Convex short-gamma P&L vs vega-linear proxy  [HYPOTHESIS TEST]")
print(SEP)

# --- data + point-in-time forecasts (identical to Phase 4) ---
rv = build_rv_series(cfg)
vix_df = load_or_fetch_vix_daily(cfg)
vix = vix_df.set_index("date")["vix_close"]
spy = load_or_fetch_daily(cfg).set_index("date")["close"]

print("Re-running walk-forward (cached snapshot) to reproduce Phase-4 forecasts…")
wf = run_walk_forward(rv, vix, cfg)
print(f"Forecast periods: {len(wf.forecasts)}  "
      f"({wf.forecasts.index[0].date()} → {wf.forecasts.index[-1].date()})\n")

# --- three P&L models on the SAME signal/thresholds ---
lin = run_linear_backtest(wf.forecasts, cfg)                          # vega-linear proxy
cvx = run_convex_backtest(wf.forecasts, spy, vix, cfg, delta_hedge=True)   # delta-hedged
nak = run_convex_backtest(wf.forecasts, spy, vix, cfg, delta_hedge=False)  # naked straddle


def series_stats(pnl: pd.Series, signal: pd.Series) -> dict:
    """Unitless, sizing-invariant tail ratios computed from ANY daily P&L series
    (gross or net) — so the linear→convex change can be read off directly."""
    pnl = pnl.fillna(0.0)
    ann = float(pnl.mean() * 252)
    vol = float(pnl.std() * np.sqrt(252))
    sd = float(pnl.std())
    cum = pnl.cumsum()
    max_dd = float((cum - cum.cummax()).min())
    worst = float(pnl.min())
    active = pnl[signal != 0]
    return {
        "sharpe": ann / vol if vol else float("nan"),
        "ann": ann,
        "max_dd": max_dd,
        "worst": worst,
        "dd_over_ann": abs(max_dd) / abs(ann) if ann else float("nan"),
        "worst_over_ann": abs(worst) / abs(ann) if ann else float("nan"),
        "worst_z": worst / sd if sd else float("nan"),
        "win": float((active > 0).mean()) if len(active) else float("nan"),
    }


def table(title, rows):
    print(f"\n--- {title} ---\n")
    hdr = (f"  {'model':<30}{'Sharpe':>8}{'AnnP&L':>10}{'DD/Ann':>9}"
           f"{'Worst/Ann':>11}{'WorstZ':>8}{'Win%':>7}")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for name, r in rows:
        print(f"  {name:<30}{r['sharpe']:>8.2f}{r['ann']:>10.2f}{r['dd_over_ann']:>9.1f}"
              f"{r['worst_over_ann']:>11.2f}{r['worst_z']:>8.1f}{100*r['win']:>6.0f}%")


# (1) GROSS — isolates the pure effect of CONVEXITY (no costs in either model)
gross_rows = [
    ("Vega-LINEAR proxy",      series_stats(lin.gross_pnl, lin.signal)),
    ("CONVEX delta-hedged",    series_stats(cvx.gross_pnl, cvx.signal)),
    ("CONVEX naked straddle",  series_stats(nak.gross_pnl, nak.signal)),
]
table("GROSS P&L (pre-cost) — the pure convexity effect on the same signal", gross_rows)
print("    -> Smoothing in the linear proxy flatters its Sharpe and hides the tail.")
print("       Convex daily mark-to-market shows the real left tail (WorstZ, DD/Ann).")

# (2) NET — layer in realistic, stress-widened option spreads + hedge costs
net_rows = [
    ("Vega-LINEAR proxy",      series_stats(lin.daily_pnl, lin.signal)),
    ("CONVEX delta-hedged",    series_stats(cvx.daily_pnl, cvx.signal)),
    ("CONVEX naked straddle",  series_stats(nak.daily_pnl, nak.signal)),
]
table("NET P&L (after costs) — realistic convex costs added", net_rows)
print("    DD/Ann=max drawdown / annual P&L  |  Worst/Ann=worst day / annual P&L")
print("    WorstZ=worst day in daily-σ units")

# --- gross vs net per model (own units) — see how much costs bite ---
print(f"\n--- Gross → Net per model (own P&L units) ---")
for res, lbl in [(lin, "linear_proxy"), (cvx, cvx.label), (nak, nak.label)]:
    g, n, c = res.gross_pnl.sum(), res.daily_pnl.sum(), res.costs.sum()
    print(f"  {lbl:<22}: gross={g:+10.2f}  costs={-c:+10.2f}  net={n:+10.2f}")

# --- where does the convex P&L come from? greek decomposition (delta-hedged) ---
g = cvx.greek_pnl.sum()
print(f"\n--- Convex (delta-hedged) P&L attribution (total over sample) ---")
print(f"    gamma  : {g['gamma']:+.3f}   (short gamma — pays on every move, the tail)")
print(f"    theta  : {g['theta']:+.3f}   (time decay the short-vol book earns)")
print(f"    vega   : {g['vega']:+.3f}   (mark-to-VIX; bleeds when implied vol spikes)")
print(f"    delta+resid: {g['delta_resid']:+.3f}   (hedge slippage + higher order)")
print(f"    net gross  : {cvx.gross_pnl.sum():+.3f}")

# --- is the edge gamma-theta (variance premium) or vega (a vol-level bet)? ---
import copy
cfg_entry = copy.deepcopy(cfg)
cfg_entry.convex_pnl.iv_mark = "entry"          # freeze IV => kill vega P&L
cvx_g = run_convex_backtest(wf.forecasts, spy, vix, cfg_entry, delta_hedge=True)
# validation: an UNCONDITIONAL always-short hedged straddle should harvest the premium
uncond = run_convex_backtest(wf.forecasts, spy, vix, cfg_entry, delta_hedge=True,
                             upper_threshold=-1e9, lower_threshold=-1e18)
gu = uncond.greek_pnl.sum()
print(f"\n--- Gamma-theta (variance premium) vs vega (vol-level bet) ---")
print(f"    SIGNAL, IV marked to VIX  : gross {cvx.gross_pnl.sum():+8.1f}  "
      f"(gamma+theta {g['gamma']+g['theta']:+.0f}, vega {g['vega']:+.0f})")
print(f"    SIGNAL, IV frozen (no vega): gross {cvx_g.gross_pnl.sum():+8.1f}  "
      f"-> the signal's gamma-theta premium alone")
print(f"    VALIDATION unconditional short, IV frozen: gamma+theta "
      f"{gu['gamma']+gu['theta']:+.0f} (engine DOES harvest the premium when always short)")
print(f"    Read: the signal's positive gross is mostly VEGA (a bet VIX falls), not the")
print(f"          gamma-theta variance premium — and vega is the leg that blows up.")

# --- crisis stress: 2008 GFC and 2020 COVID windows ---
# Reference each crisis loss to the model's GROSS annual edge (positive), so the
# multiple is unambiguous even when the net annual P&L is negative.
print(f"\n--- Crisis windows: net P&L (own units), as a multiple of gross annual edge ---")
def window(res, lo, hi):
    p = res.daily_pnl.loc[lo:hi]
    return p.sum(), p.min(), p.idxmin()
n_years = (wf.forecasts.index[-1] - wf.forecasts.index[0]).days / 365.25
for lbl, lo, hi in [("2008 GFC (Sep–Dec)", "2008-09-01", "2008-12-31"),
                    ("2020 COVID (Feb–Apr)", "2020-02-15", "2020-04-30")]:
    print(f"  {lbl}")
    for res, nm in [(lin, "linear "), (cvx, "convexΔh"), (nak, "convexNk")]:
        tot, wd, wdt = window(res, lo, hi)
        gross_ann = res.gross_pnl.sum() / n_years           # positive reference
        mult = tot / abs(gross_ann) if gross_ann else float("nan")
        print(f"    {nm}: net={tot:+9.3f}  ({mult:+5.1f}× gross-ann edge)  "
              f"worst={wd:+8.3f} on {str(wdt.date())}")

# --- 5 worst days, convex delta-hedged ---
print(f"\n--- 5 worst days — CONVEX delta-hedged (vs the same dates' linear P&L) ---")
worst = cvx.daily_pnl.nsmallest(5)
for dt, val in worst.items():
    lv = lin.daily_pnl.get(dt, float("nan"))
    vx = float(vix.reindex([dt]).iloc[0])
    print(f"  {dt.date()}  convex={val:+.4f}   linear={lv:+.4f}   VIX={vx:.1f}")

# --- plot ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(2, 1, figsize=(13, 9))
    # GROSS (pre-cost) curves, rescaled by POSITIVE factors to the linear gross total,
    # so the comparison isolates convexity (shape) without sign-flips or cost effects.
    def rescale_gross(res):
        cum = res.gross_pnl.cumsum()
        tot = res.gross_pnl.sum()
        k = (lin.gross_pnl.sum() / tot) if tot else 1.0
        return cum * k, k
    lge = lin.gross_pnl.cumsum()
    ax[0].plot(lge.index, lge.values, lw=1.1, color="black", label="Vega-linear proxy")
    cve, kc = rescale_gross(cvx)
    nve, kn = rescale_gross(nak)
    ax[0].plot(cve.index, cve.values, lw=1.0, color="crimson",
               label=f"Convex delta-hedged (×{kc:.2g})")
    ax[0].plot(nve.index, nve.values, lw=0.9, color="darkorange", alpha=0.8,
               label=f"Convex naked (×{kn:.2g})")
    ax[0].axhline(0, color="grey", lw=0.5)
    ax[0].set_title("GROSS (pre-cost) equity curves — same signal, rescaled to a common total. "
                    "Convexity exposes drawdowns the linear proxy hides.")
    ax[0].legend(fontsize=8)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # daily P&L distributions (active days), standardized to unit daily σ
    for res, c, nm in [(lin, "black", "linear"), (cvx, "crimson", "convex Δ-hedged"),
                       (nak, "darkorange", "convex naked")]:
        p = res.daily_pnl[res.signal != 0]
        p = p / p.std()
        ax[1].hist(p.values, bins=120, histtype="step", color=c, label=nm, lw=1.1)
    ax[1].set_xlim(-12, 6)
    ax[1].axvline(0, color="grey", lw=0.5)
    ax[1].set_title("Daily P&L distribution, standardized (active days) — left tail is the risk")
    ax[1].set_xlabel("daily P&L in units of own daily σ")
    ax[1].legend(fontsize=8)

    fig.suptitle("PHASE A — convex short-gamma P&L vs vega-linear proxy  "
                 "[HYPOTHESIS TEST, not a trading system]", color="darkred", fontsize=10)
    fig.tight_layout()
    out = Path("outputs/figures/phaseA_convex_vs_linear.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"\n[plot skipped: {e}]")

print(f"\n{SEP}")
print("Phase A complete. Convexity makes the tail worse — see DD/Ann and WorstZ above.")
print(SEP)
