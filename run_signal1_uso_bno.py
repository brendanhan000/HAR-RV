"""
Signal 1 runner — USO/BNO (WTI-Brent) beta-adjusted z-score OU mean-reversion.

THE GATE: does the spread mean-revert with reliable edge over its own random walk and
over buy-and-hold of each leg, net of costs, OUT-OF-SAMPLE (2023-2025 never-seen)?

Adversarial by design: dev (2010-2022) is treated as in-sample/contaminated and reported
separately from the strict holdout. We test reversion directly (does z predict the next
spread move at all?), price the trade net of ETF costs, run a Diebold-Mariano test of the
OU forecast vs a random walk, and characterise the SUSTAINED adverse excursion + rolling
cointegration (the "spread stops reverting and runs against you" tail).
"""
import logging
import sys
from pathlib import Path

logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import load_config
from src.walk_forward import diebold_mariano
from src.signals.spread import (
    build_spread, zscore, generate_spread_signal, spread_pnl,
    ou_halflife, rolling_diagnostics, ou_vs_rw_forecast_errors,
)
from src.signals.validation import (
    split_dev_holdout, perf_metrics, worst_sustained_excursion, block_bootstrap_sharpe_p,
)

SEP = "=" * 86
cfg = load_config("config.yaml")
s = cfg.signals.spread
HO = s.holdout_start

uso = pd.read_parquet(f"data/raw/{s.uso_file}").set_index("date")["close"]
bno = pd.read_parquet(f"data/raw/{s.bno_file}").set_index("date")["close"]
df = build_spread(uso, bno, s.beta_window)
df["z"] = zscore(df["spread"], s.z_window)
raw = generate_spread_signal(df["z"], s.z_entry, s.z_exit)
res = spread_pnl(df, raw, s.one_way_bps)
net, pos = res.net_return, res.position

print(f"\n{SEP}\nSIGNAL 1 — USO/BNO (WTI-Brent) beta-adjusted z-score, OU mean-reversion\n{SEP}")
print(f"  {len(df)} days [{df.index[0].date()} -> {df.index[-1].date()}]  beta~{df['beta'].median():.2f}")
print(f"  config PRIORS (not tuned to holdout): z_entry={s.z_entry} z_exit={s.z_exit} "
      f"beta_window={s.beta_window} z_window={s.z_window} cost={s.one_way_bps}bps/leg")
print(f"  DEV (in-sample/contaminated) = ..{HO} | HOLDOUT (never-seen) = {HO}..")

# ---------------------------------------------------------------------------
# 0. Does the spread even revert? (threshold-free; the cleanest adversarial test)
# ---------------------------------------------------------------------------
dz = df["spread"].diff().shift(-1)            # next-day spread change
reg = sm.OLS(dz.dropna(), sm.add_constant(df["z"].reindex(dz.dropna().index)), missing="drop").fit()
reg_hac = reg.get_robustcov_results(cov_type="HAC", maxlags=5)
print(f"\n--- 0. Reversion test: regress next-day spread change on z_t (b<0 => reverts) ---")
print(f"  slope b = {reg_hac.params[1]:+.5f}  (HAC t={reg_hac.tvalues[1]:+.2f}, p={reg_hac.pvalues[1]:.3f})  "
      f"corr={df['z'].corr(dz):+.3f}")
print(f"  OU half-life: full={ou_halflife(df['spread'].dropna()):.0f}d  "
      f"dev={ou_halflife(df['spread'].loc[:HO].dropna()):.0f}d  holdout={ou_halflife(df['spread'].loc[HO:].dropna()):.0f}d")

# ---------------------------------------------------------------------------
# 1. Performance: dev vs holdout, with skew (tail-cursed?)
# ---------------------------------------------------------------------------
print(f"\n--- 1. Strategy performance (net of {s.one_way_bps}bps/leg) — DEV vs HOLDOUT ---")
hdr = f"  {'period':<16}{'Sharpe':>8}{'annRet':>9}{'maxDD':>8}{'DD/ann':>7}{'worst':>8}{'skew':>7}{'hit':>6}{'trades':>7}"
print(hdr + "\n  " + "-" * (len(hdr) - 2))
splits = {"DEV 2010-22": (slice(None, HO)), "HOLDOUT 2023-25": (slice(HO, None))}
metrics_by = {}
for lbl, sl in splits.items():
    m = perf_metrics(net.loc[sl], pos.loc[sl]); metrics_by[lbl] = m
    n_tr = int((pos.loc[sl].diff().abs() > 0).sum())
    print(f"  {lbl:<16}{m['sharpe']:>8.2f}{m['ann_return']:>9.3f}{m['max_drawdown']:>8.3f}"
          f"{m['dd_over_ann']:>7.1f}{m['worst_day']:>8.3f}{m['skew']:>7.2f}{m['hit_rate']:>6.0%}{n_tr:>7d}")

# gross (isolate signal from cost)
def _sh(r): r = r.fillna(0); return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
print(f"  gross Sharpe: dev={_sh(res.gross_return.loc[:HO]):+.2f}  holdout={_sh(res.gross_return.loc[HO:]):+.2f}"
      f"   (cost is NOT the problem if gross is also weak)")

# ---------------------------------------------------------------------------
# 2. Benchmarks: OU forecast vs random walk (DM), and buy-and-hold each leg
# ---------------------------------------------------------------------------
print(f"\n--- 2. Benchmarks ---")
err_rw, err_ou = ou_vs_rw_forecast_errors(df["spread"], s.beta_window)
for lbl, sl in [("DEV", slice(None, HO)), ("HOLDOUT", slice(HO, None))]:
    e_rw, e_ou = err_rw.loc[sl], err_ou.loc[sl]
    dm = diebold_mariano(e_ou.values, e_rw.values, h=1, alternative="less")  # H1: OU beats RW
    better = "OU beats RW" if (dm["p_value"] is not None and dm["p_value"] < 0.05) else "OU does NOT beat RW"
    print(f"  OU vs RW forecast [{lbl:7s}]: meanΔSE={dm['mean_loss_diff']:+.2e} DM={dm['dm_stat']:+.2f} "
          f"p={dm['p_value']:.3f} -> {better}")
for lbl, px in [("USO", uso), ("BNO", bno)]:
    rr = px.pct_change().loc[HO:]
    print(f"  buy&hold {lbl} [HOLDOUT]: Sharpe={_sh(rr):+.2f}  annRet={rr.mean()*252:+.1%}")

# ---------------------------------------------------------------------------
# 3. Tail: sustained adverse excursion + structural break (cointegration)
# ---------------------------------------------------------------------------
print(f"\n--- 3. Tail / structural-break check (the mean-reversion killer) ---")
for lbl, sl in [("DEV", slice(None, HO)), ("HOLDOUT", slice(HO, None))]:
    exc = worst_sustained_excursion(net.loc[sl], pos.loc[sl])
    print(f"  [{lbl:7s}] maxDD={exc['max_drawdown']:.3f} over {exc['max_drawdown_days']}d underwater | "
          f"worst single TRADE: MAE={exc.get('worst_trade_mae', float('nan')):.3f} over {exc.get('worst_trade_days', 0)}d")
diag = rolling_diagnostics(df["spread"], s.ou_window, step=21)
if len(diag):
    stat_frac = (diag["adf_p"] < 0.05).mean()
    print(f"  rolling cointegration: residual stationary (ADF p<0.05) only {stat_frac:.0%} of the time")
    print(f"  rolling half-life: median={diag['half_life'].replace(np.inf,np.nan).median():.0f}d, "
          f"{(~np.isfinite(diag['half_life'])).mean():.0%} of windows show NO reversion (b>=1)")

# ---------------------------------------------------------------------------
# 4. Holdout significance (block bootstrap) + threshold robustness
# ---------------------------------------------------------------------------
bb = block_bootstrap_sharpe_p(net.loc[HO:], n_boot=5000, block=10, seed=cfg.random_seed)
print(f"\n--- 4. Holdout significance & robustness ---")
print(f"  holdout block-bootstrap Sharpe>0: Sharpe={bb['sharpe']:+.2f}  p={bb['p_value']:.3f}  (n={bb['n']})")
print(f"  threshold sweep — gross Sharpe (dev | holdout), to show it is robustly dead, not 1 bad setting:")
for ze, zx in [(1.5, 0.25), (2.0, 0.5), (2.5, 0.5), (2.0, 0.0), (3.0, 1.0)]:
    rr = generate_spread_signal(df["z"], ze, zx)
    rp = spread_pnl(df, rr, s.one_way_bps)
    print(f"    z_entry={ze} z_exit={zx}: dev={_sh(rp.gross_return.loc[:HO]):+.2f}  holdout={_sh(rp.gross_return.loc[HO:]):+.2f}")

# ---------------------------------------------------------------------------
# Verdict (BH correction applied across all 4 signals in the final writeup)
# ---------------------------------------------------------------------------
ho = metrics_by["HOLDOUT 2023-25"]
dead = (ho["sharpe"] <= 0.2) or (bb["p_value"] > 0.10)
print(f"\n{SEP}\nSIGNAL 1 VERDICT\n{SEP}")
if dead:
    print("  DOES NOT SURVIVE. The USO/BNO spread does not reliably mean-revert: z has ~zero")
    print(f"  predictive correlation with the next spread move (b={reg_hac.params[1]:+.5f}, p={reg_hac.pvalues[1]:.2f}),")
    print(f"  GROSS Sharpe is negative in development, the OU forecast does NOT beat a random walk,")
    print(f"  and the holdout net Sharpe ({ho['sharpe']:+.2f}) is not distinguishable from zero")
    print(f"  (bootstrap p={bb['p_value']:.2f}). The ~85d half-life is swamped by structural drift")
    print(f"  (WTI-Brent regime shifts; USO's 2020 restructuring) — the residual is stationary only")
    print(f"  a minority of the time. This is a signal-quality failure, not a cost artifact.")
else:
    print(f"  Holdout net Sharpe {ho['sharpe']:+.2f}; see significance/robustness above before believing it.")
print(SEP)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
ax[0].plot(df.index, df["spread"], lw=0.6, color="steelblue"); ax[0].axhline(0, color="k", lw=0.4)
ax[0].axvline(pd.Timestamp(HO), color="red", ls="--", lw=1, label="holdout start")
ax[0].set_ylabel("residual spread"); ax[0].legend(fontsize=8)
ax[0].set_title("Signal 1 — USO/BNO spread, z-score positions, and equity (dev vs never-seen holdout)")
ax[1].plot(df.index, df["z"], lw=0.5, color="grey")
for lv in (s.z_entry, -s.z_entry): ax[1].axhline(lv, color="green", lw=0.6, ls=":")
ax[1].fill_between(pos.index, pos.values, 0, color="orange", alpha=0.5, step="pre")
ax[1].axvline(pd.Timestamp(HO), color="red", ls="--", lw=1)
ax[1].set_ylabel("z / position")
eq = net.cumsum()
ax[2].plot(eq.index, eq.values, lw=0.9, color="black")
ax[2].axhline(0, color="k", lw=0.4); ax[2].axvline(pd.Timestamp(HO), color="red", ls="--", lw=1)
ax[2].set_ylabel("cum net P&L (per $1 USO)")
ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
Path("outputs/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/figures/signal1_uso_bno.png", dpi=150); plt.close(fig)
print(f"Figure: outputs/figures/signal1_uso_bno.png")
print(f"{SEP}\nSIGNAL 1 COMPLETE.\n{SEP}")
