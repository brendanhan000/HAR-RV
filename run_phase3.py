"""Phase 3 runner: walk-forward validation + benchmarks + DM test + verdict."""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import load_config
from src.rv_estimator import build_rv_series
from src.data_pull import load_or_fetch_vix_daily
from src.walk_forward import run_walk_forward, oos_mz_regression

cfg = load_config("config.yaml")
rv  = build_rv_series(cfg)

# Load VIX
try:
    vix_df = load_or_fetch_vix_daily(cfg)
    vix = vix_df.set_index("date")["vix_close"]
    print(f"VIX loaded: {len(vix)} days  [{vix.index[0].date()} → {vix.index[-1].date()}]")
except Exception as e:
    print(f"WARNING: VIX unavailable ({e}) — running without VIX benchmark.")
    vix = None

SEP = "=" * 65
print(f"\n{SEP}")
print(f"PHASE 3 — Walk-Forward Validation")
print(f"  horizon h={cfg.har.horizon}  |  min_train={cfg.walk_forward.min_train_days}"
      f"  |  embargo={cfg.walk_forward.embargo_days}  |  refit_every={cfg.walk_forward.refit_every_days}")
print(SEP)

wf = run_walk_forward(rv, vix, cfg)

print(f"\nForecast periods: {len(wf.forecasts)}")
print(f"Date range: {wf.forecasts.index[0].date()} → {wf.forecasts.index[-1].date()}")

print("\n--- Out-of-Sample Loss Metrics (PRIMARY: QLIKE) ---")
print(wf.metrics.to_string())

print("\n--- OOS R² (HAR vs each benchmark, based on MSE) ---")
for key, val in wf.oos_r2.items():
    bench = key.replace("har_vs_", "")
    sign = "✓ better" if val > 0 else "✗ worse"
    print(f"  {key:25s}: {val:+.4f}  ({sign})")

print("\n--- Diebold-Mariano Tests (H1: HAR has lower QLIKE than benchmark) ---")
print("  (p < 0.05 → HAR significantly beats benchmark at 5% level)")
for key, dm in wf.dm_tests.items():
    bench = key.replace("har_vs_", "")
    p = dm['p_value']
    stat = dm['dm_stat']
    diff = dm['mean_loss_diff']
    sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "n.s."))
    print(f"  vs {bench:12s}:  DM={stat:+.3f}  p={p:.4f} {sig}  (mean ΔQLIKE={diff:+.5f})")

print("\n--- Mincer-Zarnowitz Regression (OOS) ---")
print("  Ideal: intercept ≈ 0, slope ≈ 1")
for m in ["har", "log_har", "rw", "roll_vol"]:
    if m in wf.forecasts.columns:
        mz = oos_mz_regression(wf.forecasts, m, cfg.har.horizon)
        print(f"  {m:12s}: intercept={mz['mz_intercept']:.5f} (p={mz['mz_intercept_pval']:.3f})"
              f"  slope={mz['mz_slope']:.4f} (p={mz['mz_slope_pval']:.3f})"
              f"  R²={mz['mz_r2']:.4f}")

print(f"\n{SEP}")
print(wf.verdict)
print(SEP)

# --- Plot: forecast vs actual + cumulative QLIKE ---
fig, axes = plt.subplots(3, 1, figsize=(13, 10))

# Panel 1: actual vs HAR forecast
ax = axes[0]
ax.plot(wf.forecasts.index, wf.forecasts["actual"], lw=0.7, color="black", label="Actual RV")
ax.plot(wf.forecasts.index, wf.forecasts["har"], lw=0.8, color="steelblue", label="HAR forecast", alpha=0.8)
if vix is not None and wf.forecasts["vix"].notna().any():
    ax.plot(wf.forecasts.index, wf.forecasts["vix"], lw=0.7, color="green",
            label="VIX implied var", alpha=0.6, linestyle="--")
ax.set_ylabel("RV (annualized)")
ax.set_title("Out-of-Sample: Actual vs HAR Forecast")
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Panel 2: cumulative QLIKE advantage over random walk
from src.walk_forward import qlike as ql_fn
actual_arr = wf.forecasts["actual"].values
rw_arr     = wf.forecasts["rw"].values
har_arr    = wf.forecasts["har"].values
cum_adv_rw  = np.cumsum(ql_fn(actual_arr, rw_arr)  - ql_fn(actual_arr, har_arr))
ax2 = axes[1]
ax2.plot(wf.forecasts.index, cum_adv_rw, lw=0.9, color="steelblue")
ax2.axhline(0, color="black", lw=0.7, linestyle="--")
ax2.set_ylabel("Cumulative QLIKE advantage\n(HAR − RW, higher = HAR better)")
ax2.set_title("Cumulative QLIKE: HAR vs Random Walk")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Panel 3: rolling 252-day OOS R² vs RW
from src.walk_forward import squared_error as se_fn
se_har = se_fn(actual_arr, har_arr)
se_rw  = se_fn(actual_arr, rw_arr)
roll_r2 = 1 - (
    pd.Series(se_har, index=wf.forecasts.index).rolling(252).mean()
    / pd.Series(se_rw, index=wf.forecasts.index).rolling(252).mean()
)
ax3 = axes[2]
ax3.plot(roll_r2.index, roll_r2.values, lw=0.9, color="darkorange")
ax3.axhline(0, color="black", lw=0.7, linestyle="--")
ax3.set_ylabel("Rolling OOS R² vs RW (252-day)")
ax3.set_title("HAR Predictive Edge Over Time (rolling 1-year)")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.tight_layout()
out = Path("outputs/figures/walk_forward.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out}")
