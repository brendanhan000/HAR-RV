"""Phase 1 runner: build RV series, print summary, save plot."""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.rv_estimator import build_rv_series, rv_summary, plot_rv, rv_to_vol

cfg = load_config("config.yaml")

print("=" * 60)
print("PHASE 1 — Realized Variance")
print("=" * 60)

rv = build_rv_series(cfg)
print(f"\nRV series: {len(rv)} trading days  [{rv.index[0].date()} → {rv.index[-1].date()}]")

summary = rv_summary(rv)
print("\nSummary statistics:")
print(summary.to_string())

plot_path = plot_rv(rv)
print(f"\nPlot saved: {plot_path}")
