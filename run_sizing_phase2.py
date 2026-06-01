"""
Sizing overlay — PHASE 2 runner: the sizing RULES and their properties.

Demonstrates, on the real elevated-VIX baseline stream:
  1. Vol targeting    — the point-in-time leverage path (and where the cap binds);
  2. Fractional Kelly — deep-drawdown ('ruin') probability vs EDGE-ESTIMATE ERROR
                        (the key caveat: why we default to 1/4 Kelly, never full);
  3. Hard per-session cap — binding behaviour; clean (constant-A) vs point-in-time-A;
  4. Drawdown brake   — leverage cut after a sized-book drawdown.

This is NOT the verdict. The rigorous flat-vs-sized benchmark comparison (net of convex
costs + turnover, with a paired DM test) is Phase 3. Vol target & Kelly fraction here are
PRIORS from config, not tuned to the backtest.
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
from src.sizing import apply_overlay, simulate_kelly_ruin

SEP = "=" * 80
cfg = load_config("config.yaml")
ru = cfg.sizing.rules
pnl = pd.read_parquet("data/raw/baseline_elevated_vix_pnl.parquet")
net, sig = pnl["net"], pnl["signal"]


def metrics(p):
    p = p.fillna(0.0)
    ann = p.mean() * 252
    vol = p.std() * np.sqrt(252)
    cum = p.cumsum()
    mdd = (cum - cum.cummax()).min()
    return dict(ann=ann, sharpe=ann / vol if vol else np.nan, mdd=mdd,
                dda=abs(mdd) / abs(ann) if ann else np.nan, worst=p.min())


def with_rules(**ov):
    c = copy.deepcopy(cfg)
    for k, v in ov.items():
        setattr(c.sizing.rules, k, v)
    return apply_overlay(net, sig, c)


print(f"\n{SEP}\nSIZING OVERLAY — PHASE 2: sizing rules (config-driven priors)\n{SEP}")
print(f"  target_vol={ru.target_vol}  vol_floor={ru.vol_floor_frac}*target  max_lev={ru.max_leverage}")
print(f"  kelly_fraction={ru.kelly_fraction}  cap={ru.cap_fraction} of annual exp. P&L @ {ru.cap_loss_sigma}σ"
      f"  condition_on_active={ru.condition_on_active}")

# ===========================================================================
# 1. Vol-target leverage path
# ===========================================================================
r_vt = with_rules(method="vol_target", cap_enabled=False)
lev = r_vt.leverage.dropna()
print(f"\n{'-'*80}\n1. VOL-TARGET LEVERAGE PATH (no cap)\n{'-'*80}")
print(f"  leverage  min/median/mean/p95/max = {lev.min():.2f}/{lev.median():.2f}/{lev.mean():.2f}/"
      f"{lev.quantile(.95):.2f}/{lev.max():.2f}")
print(f"  de-levers in high-vol regimes; floor caps it at 1/{ru.vol_floor_frac:.2f}="
      f"{1/ru.vol_floor_frac:.0f}x, hard ceiling {ru.max_leverage:.0f}x")

# ===========================================================================
# 2. FRACTIONAL KELLY — ruin vs edge-estimate error  (THE key caveat)
# ===========================================================================
act = net[sig != 0]
sharpe_act = float(act.mean() / act.std() * np.sqrt(252))
print(f"\n{'-'*80}\n2. FRACTIONAL KELLY — RUIN vs EDGE-ESTIMATE ERROR\n{'-'*80}")
print(f"  parameterised by the strategy's realised ACTIVE-day Sharpe = {sharpe_act:.2f}, at a "
      f"{ru.target_vol*100:.0f}% vol scale,\n  with the empirical active-return SHAPE (fat left tail) bootstrapped.")
print(f"  Full Kelly leverage = Sharpe/vol = {sharpe_act/ru.target_vol:.1f}x.  You only have an")
print(f"  ESTIMATE of the edge; the x-axis is how wrong it is.\n")

ruin = simulate_kelly_ruin(
    act.values, sharpe_ann=sharpe_act, vol_ann=ru.target_vol,
    kelly_fractions=(0.25, 0.5, 1.0),
    edge_errors=tuple(np.round(np.arange(-0.5, 0.51, 0.25), 2)),
    horizon=1260, n_sims=6000, ruin_dd=0.5, seed=cfg.random_seed,
)
print(f"  {'Kelly':>6} {'edgeErr':>8} {'lev(x)':>7} {'P(DD>=50%)':>11} {'P(wipeout)':>11} {'medCAGR':>9}")
for _, row in ruin.iterrows():
    print(f"  {row.kelly_fraction:>6.2f} {row.edge_error:>+8.0%} {row.leverage_x:>7.2f}"
          f" {row.p_deep_dd:>11.1%} {row.p_wipeout:>11.1%} {row.median_cagr:>+9.1%}")
print("\n  => Over-estimating the edge (right side) inflates ruin sharply at FULL Kelly; quarter")
print("     Kelly stays comparatively safe even at +50% edge error. This is WHY the default is 1/4.")

# ===========================================================================
# 3. Hard per-session cap
# ===========================================================================
r_cap = with_rules(method="vol_target", cap_enabled=True)
A_const = float(net.mean() * 252)
r_cap_const = with_rules(method="vol_target", cap_enabled=True, annual_expected_pnl=A_const)
print(f"\n{'-'*80}\n3. HARD PER-SESSION CAP (binding, applied last)\n{'-'*80}")
print(f"  cap binds on {r_cap.meta['pct_days_cap_binding']:.0f}% of in-sample days "
      f"(annual P&L is small vs session risk -> the cap dominates).")
m_vt, m_cap, m_cc = metrics(r_vt.sized_pnl), metrics(r_cap.sized_pnl), metrics(r_cap_const.sized_pnl)
print(f"  {'variant':<28}{'Sharpe':>8}{'DD/ann':>8}{'maxDD':>9}{'worstDay':>10}")
for lbl, m in [("vol-target (no cap)", m_vt),
               ("+cap, A=point-in-time", m_cap),
               ("+cap, A=constant (clean)", m_cc)]:
    print(f"  {lbl:<28}{m['sharpe']:>8.3f}{m['dda']:>8.1f}{m['mdd']:>9.2f}{m['worst']:>10.2f}")
print("  => with a STABLE annual-expected, the cap is a pure LEVEL cut: Sharpe & DD/ann unchanged,")
print("     absolute drawdown slashed. With a point-in-time estimate it also de-levers on weak")
print("     trailing edge (pro-cyclical), which costs risk-adjusted return. The cap buys a tail")
print("     FLOOR, not better risk-adjusted return — exactly the expected outcome.")

# ===========================================================================
# 4. Drawdown brake (optional)
# ===========================================================================
r_brake = with_rules(method="vol_target", cap_enabled=False, brake_enabled=True)
r_brake10 = with_rules(method="vol_target", cap_enabled=False, brake_enabled=True, brake_threshold=0.10)
m_brake, m_brake10 = metrics(r_brake.sized_pnl), metrics(r_brake10.sized_pnl)
print(f"\n{'-'*80}\n4. DRAWDOWN BRAKE (optional; de-levers after sized-book drawdown)\n{'-'*80}")
print(f"  default threshold={ru.brake_threshold:.0%}: DORMANT — the sized book's maxDD "
      f"({abs(m_vt['mdd'])/ru.reference_capital*100:.0f}% of capital) sits just under it.")
print(f"  {'variant':<30}{'Sharpe':>8}{'DD/ann':>8}{'maxDD':>9}{'%braked':>9}")
print(f"  {'vol-target (no brake)':<30}{m_vt['sharpe']:>8.3f}{m_vt['dda']:>8.1f}{m_vt['mdd']:>9.2f}{0:>8.0f}%")
print(f"  {'+ brake @20% (default)':<30}{m_brake['sharpe']:>8.3f}{m_brake['dda']:>8.1f}{m_brake['mdd']:>9.2f}"
      f"{(r_brake.brake_mult<1).mean()*100:>8.0f}%")
print(f"  {'+ brake @10% (engages)':<30}{m_brake10['sharpe']:>8.3f}{m_brake10['dda']:>8.1f}{m_brake10['mdd']:>9.2f}"
      f"{(r_brake10.brake_mult<1).mean()*100:>8.0f}%")
print("  => at DISCIPLINED leverage the book rarely draws down enough to trip the brake — it is a")
print("     backstop for aggressive sizing, not a routine contributor (a finding in itself).")

# ===========================================================================
# Figures
# ===========================================================================
# Fig 1: leverage path + sigma + cap
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax[0].plot(r_vt.sigma_hat.index, r_vt.sigma_hat.values, lw=0.8, color="steelblue")
ax[0].set_ylabel("σ̂ (ann., active-conditioned)")
ax[0].set_title("Phase 2 — vol-target leverage path (point-in-time)")
ax[1].plot(r_vt.leverage.index, r_vt.leverage.values, lw=0.7, color="darkorange", label="vol-target leverage")
ax[1].plot(r_cap.leverage.index, r_cap.leverage.values, lw=0.7, color="firebrick", alpha=0.8,
           label="after hard cap")
ax[1].axhline(1.0, color="black", lw=0.5, ls=":")
ax[1].set_ylabel("leverage w")
ax[1].set_ylim(0, ru.max_leverage * 1.05)
ax[1].legend(fontsize=8)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig("outputs/figures/sizing_phase2_leverage.png", dpi=150)
plt.close(fig)

# Fig 2: ruin curve
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for c in (0.25, 0.5, 1.0):
    sub = ruin[ruin.kelly_fraction == c]
    ax[0].plot(sub.edge_error * 100, sub.p_deep_dd * 100, marker="o", label=f"{c:g} Kelly")
    ax[1].plot(sub.edge_error * 100, sub.median_cagr * 100, marker="o", label=f"{c:g} Kelly")
ax[0].set_xlabel("edge-estimate error (%)"); ax[0].set_ylabel("P(drawdown ≥ 50%) over 5y (%)")
ax[0].set_title("Kelly ruin risk vs edge error"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("edge-estimate error (%)"); ax[1].set_ylabel("median CAGR (%)")
ax[1].set_title("…and the growth you trade for it"); ax[1].legend(fontsize=8)
ax[1].axhline(0, color="black", lw=0.5)
fig.suptitle("Phase 2 — fractional Kelly: deep-drawdown probability vs edge-estimate error", y=1.02)
fig.tight_layout()
fig.savefig("outputs/figures/sizing_phase2_kelly_ruin.png", dpi=150)
plt.close(fig)

print(f"\nFigures: outputs/figures/sizing_phase2_leverage.png, sizing_phase2_kelly_ruin.png")
print(f"{SEP}\nPHASE 2 COMPLETE — rules built & characterised. Phase 3 = benchmark-first verdict.\n{SEP}")
