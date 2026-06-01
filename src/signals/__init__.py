"""
Adversarial validation of four live signals through the existing honest harness.

Each signal gets: a STRICT never-seen holdout (dev period = in-sample/contaminated),
its OWN correct benchmark and P&L model, point-in-time inputs (no look-ahead), and a
report of net Sharpe, max drawdown, drawdown/annual ratio, worst day, turnover/cost,
hit rate, return skew, and a DM-style test vs its benchmark. Significance claims are
corrected for multiple testing (Benjamini-Hochberg) across the four signals.

Shared utilities live in `validation.py`; each signal has its own module:
  spread.py   — Signal 1: USO/BNO (WTI-Brent) beta-adjusted z-score OU mean-reversion
  (turtle.py, smartflow.py, regime.py to follow)
"""
