"""Phase 2 runner: fit HAR-RV and log-HAR on full SPY history, print diagnostics."""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.config import load_config
from src.rv_estimator import build_rv_series
from src.har_model import (
    build_har_features,
    build_log_har_features,
    coef_table,
    fit_har,
    fit_log_har,
    insample_diagnostics,
)

cfg = load_config("config.yaml")
rv = build_rv_series(cfg)

features     = build_har_features(rv, cfg)
log_features = build_log_har_features(rv, cfg)

har_result     = fit_har(features, cfg)
loghar_result  = fit_log_har(log_features, cfg)

SEP = "=" * 65

print(f"\n{SEP}")
print(f"PHASE 2 — HAR-RV  (h={cfg.har.horizon} trading days, n={har_result.n_obs})")
print(f"Estimator: Yang-Zhang daily  |  NW lags: {har_result.hac_lags_used}")
print(SEP)

print("\n--- HAR Coefficient Table (HAC/Newey-West SEs) ---")
ct = coef_table(har_result)
print(ct.drop(columns="note", errors="ignore").to_string())

print("\n--- log-HAR Coefficient Table (HAC/Newey-West SEs) ---")
ct_log = coef_table(loghar_result)
print(ct_log.drop(columns="note", errors="ignore").to_string())

print(f"\n  log-HAR Jensen bias correction (σ²_ε = {loghar_result.log_bias_correction:.6f})")

print("\n--- In-sample Diagnostics ---")
diag_har    = insample_diagnostics(har_result)
diag_loghar = insample_diagnostics(loghar_result)
diag_df = pd.DataFrame([diag_har, diag_loghar]).set_index("model")
print(diag_df.T.to_string())

print(f"""
--- Interpretation notes ---
* Primary loss is QLIKE (Patton 2011 proper loss for variance).
* HAC SEs used throughout; naive OLS SEs are invalid for h={cfg.har.horizon} (overlapping targets).
* MZ slope ≈ 1, intercept ≈ 0 would indicate an unbiased forecast.
  MZ slope < 1 → forecast over-reacts; > 1 → under-reacts.
* DW < 2 indicates positive residual autocorrelation (expected for h>1).
""")
