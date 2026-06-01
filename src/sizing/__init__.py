"""
Reusable risk & position-sizing overlay.

A drop-in layer that takes ANY strategy's per-period raw return stream plus a
point-in-time risk estimate and produces a sized position. It is deliberately
strategy-agnostic: it knows nothing about straddles, VRP, or z-scores — only a
return series and (optionally) a covariance for the multi-position case.

Phases (built incrementally, gated):
  1. risk.py   — point-in-time risk estimation (EWMA / rolling vol; LW covariance)
  2. rules.py  — sizing rules (vol target, fractional Kelly, hard cap, drawdown brake)   [later]
  3. overlay.py— the drop-in that wires a return stream + risk -> sized position          [later]

CARDINAL RULE: every input used to size period t uses only data <= t-1. No estimate
that sizes period t may peek at period t's own return. This is enforced by lagging
(shift) inside the estimators and verified by unit tests.
"""
from src.sizing.risk import (  # noqa: F401
    CovEstimates,
    estimate_cov,
    estimate_vol,
    ledoit_wolf_cov,
    realized_vol_ewma,
    realized_vol_rolling,
    rolling_cov,
    sample_cov,
)
from src.sizing.rules import (  # noqa: F401
    drawdown_brake_factor,
    hard_session_cap,
    kelly_leverage,
    lagged_mean,
    simulate_kelly_ruin,
    vol_target_leverage,
)
from src.sizing.overlay import SizedResult, apply_overlay  # noqa: F401

__all__ = [
    # risk
    "realized_vol_rolling", "realized_vol_ewma", "estimate_vol",
    "sample_cov", "ledoit_wolf_cov", "rolling_cov", "estimate_cov", "CovEstimates",
    # rules
    "vol_target_leverage", "kelly_leverage", "lagged_mean",
    "hard_session_cap", "drawdown_brake_factor", "simulate_kelly_ruin",
    # overlay
    "apply_overlay", "SizedResult",
]
