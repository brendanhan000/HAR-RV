"""
Phase 1 — Point-in-time risk estimation of a strategy's OWN return stream.

The sizing overlay scales a strategy by an estimate of that strategy's own
volatility (single-position) or covariance (multi-position). Every estimate is
computed STRICTLY from data <= t-1 so that the number used to size period t never
sees period t's realized return. We enforce this by computing the rolling/EWMA
statistic (which, as pandas computes it, is inclusive of t) and then `.shift(1)`,
so the value carried at index t is a function of returns up to and including t-1.

  realized_vol_rolling(r, ...) -> sigma_hat  (point-in-time, lagged)
  realized_vol_ewma(r, ...)    -> sigma_hat  (RiskMetrics-style, lagged)
  estimate_vol(r, risk_cfg)    -> dispatcher reading config

Multi-position (gated; the single-position path never touches this):
  sample_cov(X)        -> (Sigma, 0.0)
  ledoit_wolf_cov(X)   -> (Sigma_shrunk, shrinkage_delta)   [closed-form LW 2004]
  rolling_cov(df, ...) -> CovEstimates (one point-in-time matrix per date, data < t)

Conventions
-----------
* Returns are a strategy's per-period P&L/return stream (zero-mean is the default
  assumption for risk targeting — RiskMetrics convention — but `demean=True` gives
  the centred sample std). The two differ negligibly for near-zero-mean daily
  strategy returns; the flag is exposed and documented, not silently chosen.
* `annualize=True` returns ANNUALISED volatility: sigma *= sqrt(periods_per_year).
  Covariances are scaled by periods_per_year (variance units), i.e. the SQUARE of
  the vol annualisation, so a covariance and its diagonal vol stay consistent.
* The estimators return the raw estimate (which may be 0 for constant returns or
  NaN before enough history). Flooring/clamping for leverage is the SIZING RULE's
  job (Phase 2), not the estimator's — keep estimation honest and separate.

STRETCH (noted, not implemented): RMT / Marchenko-Pastur eigenvalue denoising of the
covariance matrix (clip sample eigenvalues below the MP upper edge to their average).
Worthwhile when #strategies approaches the lookback length; Ledoit-Wolf already gives
a well-conditioned shrinkage estimator for the handful-of-strategies case here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_returns(returns: pd.Series) -> pd.Series:
    """Validate/normalise a return stream: sorted Series of float, name preserved."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series indexed by period")
    if not returns.index.is_monotonic_increasing:
        returns = returns.sort_index()
    return returns.astype(float)


def _vol_annualizer(annualize: bool, periods_per_year: int) -> float:
    """Volatility annualisation factor (sqrt of periods/year); 1.0 if not annualising."""
    return float(np.sqrt(periods_per_year)) if annualize else 1.0


# ---------------------------------------------------------------------------
# Single-position volatility estimators (point-in-time / lagged)
# ---------------------------------------------------------------------------


def realized_vol_rolling(
    returns: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
    demean: bool = False,
    ddof: int = 0,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Rolling-window realized volatility of a return stream, LAGGED so the value at
    index t uses only returns[t-window .. t-1] (strictly before t).

    Parameters
    ----------
    returns          : per-period return/P&L stream (pandas Series).
    window           : lookback length in periods.
    min_periods      : min observations required before emitting a value
                       (default = window: require a full window).
    demean           : False -> zero-mean RMS sqrt(mean(r^2)) (RiskMetrics convention);
                       True  -> centred sample std with `ddof`.
    ddof             : delta-dof for the centred std (only used when demean=True).
    annualize        : multiply by sqrt(periods_per_year).
    periods_per_year : annualisation base.

    Returns
    -------
    pd.Series aligned to `returns.index`, named "sigma_hat". NaN until enough history.
    The shift(1) is what guarantees no look-ahead: sigma_hat[t] never uses r[t].
    """
    r = _as_returns(returns)
    if window < 2:
        raise ValueError("window must be >= 2")
    if min_periods is None:
        min_periods = window
    min_periods = max(2, int(min_periods))

    if demean:
        stat = r.rolling(window=window, min_periods=min_periods).std(ddof=ddof)
    else:
        # zero-mean RMS: sqrt(mean(r^2)) over the window
        stat = np.sqrt((r ** 2).rolling(window=window, min_periods=min_periods).mean())

    sigma = stat.shift(1) * _vol_annualizer(annualize, periods_per_year)
    sigma.name = "sigma_hat"
    return sigma


def realized_vol_ewma(
    returns: pd.Series,
    halflife: Optional[float] = None,
    lam: Optional[float] = None,
    span: Optional[float] = None,
    min_periods: int = 1,
    demean: bool = False,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Exponentially-weighted (RiskMetrics-style) realized volatility, LAGGED so the
    value at index t uses only returns up to and including t-1.

    Decay is specified by exactly one of:
      lam      : RiskMetrics decay lambda; sigma^2_t = lam*sigma^2_{t-1} + (1-lam)*r^2
                 (pandas alpha = 1 - lam). RiskMetrics daily default is lam=0.94.
      halflife : periods for the weight to halve (alpha = 1 - 0.5**(1/halflife)).
      span     : pandas span (alpha = 2/(span+1)).

    Other parameters mirror `realized_vol_rolling`. `adjust=False` (recursive form)
    is used so the estimate is the genuine RiskMetrics recursion.
    """
    r = _as_returns(returns)
    n_decay = sum(x is not None for x in (halflife, lam, span))
    if n_decay != 1:
        raise ValueError("specify exactly one of halflife, lam, span")

    if lam is not None:
        if not (0.0 < lam < 1.0):
            raise ValueError("lam (RiskMetrics decay) must be in (0,1)")
        ewm_kwargs = dict(alpha=1.0 - lam)
    elif halflife is not None:
        ewm_kwargs = dict(halflife=halflife)
    else:
        ewm_kwargs = dict(span=span)

    if demean:
        stat = r.ewm(min_periods=min_periods, adjust=False, **ewm_kwargs).std()
    else:
        ms = (r ** 2).ewm(min_periods=min_periods, adjust=False, **ewm_kwargs).mean()
        stat = np.sqrt(ms)

    sigma = stat.shift(1) * _vol_annualizer(annualize, periods_per_year)
    sigma.name = "sigma_hat"
    return sigma


def estimate_vol(returns: pd.Series, risk_cfg) -> pd.Series:
    """
    Config-driven dispatcher. `risk_cfg` is a RiskConfig (or any object exposing the
    same attributes). Returns the lagged point-in-time volatility estimate.
    """
    est = risk_cfg.estimator.lower()
    if est == "rolling":
        return realized_vol_rolling(
            returns,
            window=risk_cfg.window,
            min_periods=risk_cfg.rolling_min_periods,
            demean=risk_cfg.demean,
            ddof=risk_cfg.ddof,
            annualize=risk_cfg.annualize,
            periods_per_year=risk_cfg.periods_per_year,
        )
    if est == "ewma":
        return realized_vol_ewma(
            returns,
            halflife=(None if risk_cfg.ewma_lambda is not None else risk_cfg.ewma_halflife),
            lam=risk_cfg.ewma_lambda,
            min_periods=risk_cfg.ewma_min_periods,
            demean=risk_cfg.demean,
            annualize=risk_cfg.annualize,
            periods_per_year=risk_cfg.periods_per_year,
        )
    raise ValueError(f"unknown risk estimator '{risk_cfg.estimator}' (use 'ewma' or 'rolling')")


# ---------------------------------------------------------------------------
# Multi-position covariance estimators (gated; single-position never uses these)
# ---------------------------------------------------------------------------


def sample_cov(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Plain sample covariance (MLE, divisor n) of an (n_obs x p) return matrix.
    Returns (Sigma, shrinkage=0.0) to share the signature of `ledoit_wolf_cov`.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = (Xc.T @ Xc) / n   # Apple-Accelerate matmul raises spurious FP flags; result is exact
    return S, 0.0


def ledoit_wolf_cov(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) shrinkage covariance toward a scaled-identity target.

    Shrinks the (ill-conditioned) sample covariance S toward F = mu*I, where
    mu = mean variance, with the closed-form optimal intensity

        delta* = b^2 / d^2,   b^2 = min(b_bar^2, d^2)

    using the normalised Frobenius norm <A,B> = trace(A B')/p:
        m       = <S, I>                = trace(S)/p
        d^2     = ||S - m I||^2
        b_bar^2 = (1/n^2) * sum_k ||x_k x_k' - S||^2   (clipped to d^2)
    The cross term collapses analytically (sum_k x_k' S x_k = n*trace(S^2)), giving
        sum_k ||x_k x_k' - S||^2 = sum_k ||x_k||^4 - n*||S||^2
    so no Python loop over observations is needed.

    Returns (Sigma_shrunk, delta) with delta in [0,1]. delta -> 0 means "trust the
    sample" (lots of data); delta -> 1 means "trust the identity" (noisy / p ~ n).
    Sigma is positive semi-definite by construction (convex combo of two PSD matrices).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_obs x p)")
    n, p = X.shape
    if n < 2:
        raise ValueError("need >= 2 observations for a covariance estimate")

    Xc = X - X.mean(axis=0, keepdims=True)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        S = (Xc.T @ Xc) / n   # Apple-Accelerate matmul raises spurious FP flags; result is exact
    m = np.trace(S) / p
    F = m * np.eye(p)

    d2 = np.sum((S - F) ** 2) / p
    if d2 <= 0:                       # S already == m*I (e.g. p==1): nothing to shrink
        return S, 0.0

    q = (Xc ** 2).sum(axis=1)         # ||x_k||^2 per observation
    phi = float((q ** 2).sum() - n * np.sum(S ** 2))   # sum_k ||x_k x_k' - S||^2
    phi = max(phi, 0.0)
    b_bar2 = phi / (n ** 2) / p
    b2 = min(b_bar2, d2)
    delta = b2 / d2

    Sigma = delta * F + (1.0 - delta) * S
    return Sigma, float(delta)


@dataclass
class CovEstimates:
    """Point-in-time covariance matrices: one per date, each from data STRICTLY < t."""
    assets: List[str]
    matrices: Dict[pd.Timestamp, np.ndarray] = field(default_factory=dict)
    shrinkage: Dict[pd.Timestamp, float] = field(default_factory=dict)

    def at(self, t) -> Optional[np.ndarray]:
        return self.matrices.get(t)

    @property
    def dates(self) -> List[pd.Timestamp]:
        return list(self.matrices.keys())


def rolling_cov(
    returns_df: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    method: str = "ledoit_wolf",
    annualize: bool = False,
    periods_per_year: int = 252,
) -> CovEstimates:
    """
    Point-in-time covariance series for a multi-strategy return panel.

    For each date t, the covariance is estimated from the last `window` returns that
    occur STRICTLY BEFORE t (rows < t), mirroring the shift(1) discipline of the
    scalar estimators — so the matrix used to size period t never sees period t.

    Parameters
    ----------
    returns_df  : (T x p) panel of strategy returns (columns = strategies).
    window      : lookback length.
    min_periods : min rows required before emitting a matrix (default window).
    method      : "ledoit_wolf" (shrunk) or "sample".
    annualize   : scale covariance by periods_per_year (variance units), keeping it
                  consistent with annualised vols on the diagonal.

    Returns
    -------
    CovEstimates (matrices keyed by date; only dates with >= min_periods history).
    """
    if min_periods is None:
        min_periods = window
    estimator = {"ledoit_wolf": ledoit_wolf_cov, "sample": sample_cov}.get(method)
    if estimator is None:
        raise ValueError(f"unknown covariance method '{method}'")

    var_scale = float(periods_per_year) if annualize else 1.0
    df = returns_df.sort_index()
    idx = df.index
    out = CovEstimates(assets=list(df.columns))

    for i, t in enumerate(idx):
        if i < min_periods:                  # not enough history strictly before t
            continue
        win = df.iloc[max(0, i - window):i]   # rows strictly before t (no row i)
        win = win.dropna()
        if len(win) < min_periods:
            continue
        Sigma, delta = estimator(win.values)
        out.matrices[t] = Sigma * var_scale
        out.shrinkage[t] = delta
    return out


def estimate_cov(returns_df: pd.DataFrame, cov_cfg) -> CovEstimates:
    """Config-driven covariance dispatcher (multi-position path)."""
    return rolling_cov(
        returns_df,
        window=cov_cfg.window,
        min_periods=cov_cfg.min_periods,
        method=cov_cfg.method,
        annualize=getattr(cov_cfg, "annualize", False),
        periods_per_year=getattr(cov_cfg, "periods_per_year", 252),
    )
