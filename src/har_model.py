"""
Phase 2: HAR-RV model (Corsi 2009) and log-HAR variant.

Public API:
  build_har_features(rv, cfg)         -> pd.DataFrame (RV_d, RV_w, RV_m, target)
  fit_har(features, cfg)              -> HARResult
  fit_log_har(features, cfg)          -> HARResult
  insample_diagnostics(result)        -> dict

HARResult holds coefficients, HAC standard errors, t-stats, p-values,
in-sample fitted values, and residuals.

IMPORTANT invariants enforced here:
  - Features at time t use ONLY data <= t (rolling windows look backward only).
  - When h > 1, targets are overlapping averages → HAC/Newey-West SEs are
    MANDATORY. Naive OLS SEs are stored but never surfaced as primary.
  - Log-HAR back-transforms include Jensen's inequality bias correction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

from src.config import Config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def build_har_features(rv: pd.Series, cfg: Config) -> pd.DataFrame:
    """
    Build the HAR feature matrix from a daily RV series.

    For each date t:
      RV_d_t  = RV_t                           (daily)
      RV_w_t  = mean(RV_{t-4..t})             (5-day, inclusive)
      RV_m_t  = mean(RV_{t-21..t})            (22-day, inclusive)
      target  = mean(RV_{t+1..t+h})           (h-day-ahead average)

    All windows use only data <= t for features (strictly causal).
    Target requires h future observations; rows at the tail are dropped.

    Returns a DataFrame with columns:
      RV_d, RV_w, RV_m, target
    indexed by date t. Rows with any NaN are dropped.
    """
    h = cfg.har.horizon
    d_lag = cfg.har.lags["daily"]    # 1
    w_lag = cfg.har.lags["weekly"]   # 5
    m_lag = cfg.har.lags["monthly"]  # 22

    rv = rv.sort_index().copy()

    # Features: rolling means (min_periods = full window to avoid partial warmup)
    rv_d = rv                                                    # daily = RV_t itself
    rv_w = rv.rolling(window=w_lag, min_periods=w_lag).mean()
    rv_m = rv.rolling(window=m_lag, min_periods=m_lag).mean()

    # Target: forward-looking mean over next h days
    # shift(-1) so target at t = mean of t+1..t+h
    target = rv.shift(-1).rolling(window=h, min_periods=h).mean().shift(-(h - 1))

    df = pd.DataFrame({
        "RV_d": rv_d,
        "RV_w": rv_w,
        "RV_m": rv_m,
        "target": target,
    })

    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    log.info(
        "HAR features: %d rows (dropped %d for warmup/tail, h=%d)",
        n_after, n_before - n_after, h,
    )
    return df


def build_log_har_features(rv: pd.Series, cfg: Config) -> pd.DataFrame:
    """
    Same as build_har_features but in log(RV) space.
    Returns log_RV_d, log_RV_w, log_RV_m, log_target.
    """
    features = build_har_features(rv, cfg)
    log_features = np.log(features)
    log_features.columns = ["log_RV_d", "log_RV_w", "log_RV_m", "log_target"]
    return log_features


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class HARResult:
    model_type: str                 # "HAR" or "log-HAR"
    horizon: int
    params: pd.Series               # coefficient estimates
    hac_se: pd.Series               # HAC/Newey-West standard errors (primary)
    ols_se: pd.Series               # naive OLS SEs (stored but not primary for h>1)
    hac_tstat: pd.Series
    hac_pvalue: pd.Series
    fitted: pd.Series               # in-sample fitted values (original RV space)
    residuals: pd.Series            # in-sample residuals (original RV space)
    target: pd.Series               # actual targets (original RV space)
    r2_insample: float
    hac_lags_used: int
    n_obs: int
    # For log-HAR only
    log_bias_correction: Optional[float] = None
    residual_variance: Optional[float] = None


# ---------------------------------------------------------------------------
# HAC lag selection
# ---------------------------------------------------------------------------


def _hac_lags(n: int, h: int, cfg_lags: Optional[int]) -> int:
    """
    Choose number of Newey-West lags.
    Rule: max(h, floor(T^(1/3))). Config can override.
    For h=1 we still use NW but with a small lag to be conservative.
    """
    if cfg_lags is not None:
        return cfg_lags
    auto = max(h, int(np.floor(n ** (1 / 3))))
    return auto


# ---------------------------------------------------------------------------
# HAR fit
# ---------------------------------------------------------------------------


def fit_har(features: pd.DataFrame, cfg: Config) -> HARResult:
    """
    Fit the base HAR-RV model (Corsi 2009) by OLS with HAC standard errors.

      target_t = b0 + b_d*RV_d_t + b_w*RV_w_t + b_m*RV_m_t + e_t

    Parameters
    ----------
    features : output of build_har_features()
    cfg      : full Config

    Returns
    -------
    HARResult with HAC SEs as primary inference.
    """
    h = cfg.har.horizon
    y = features["target"]
    X = sm.add_constant(features[["RV_d", "RV_w", "RV_m"]])

    ols_res = sm.OLS(y, X).fit()
    n = len(y)
    lags = _hac_lags(n, h, cfg.har.hac_lags)

    # Newey-West with bandwidth = lags
    hac_res = ols_res.get_robustcov_results(cov_type="HAC", maxlags=lags, use_correction=True)

    # get_robustcov_results returns arrays; re-attach param names from OLS result
    pnames = ols_res.params.index
    hac_params = pd.Series(hac_res.params, index=pnames, name="coef")
    hac_se     = pd.Series(hac_res.bse,    index=pnames, name="hac_se")
    hac_tstat  = pd.Series(hac_res.tvalues, index=pnames, name="t_stat")
    hac_pvalue = pd.Series(hac_res.pvalues, index=pnames, name="p_value")

    fitted = pd.Series(ols_res.fittedvalues, index=features.index, name="fitted_HAR")
    residuals = pd.Series(ols_res.resid, index=features.index, name="resid_HAR")

    # Clamp fitted to non-negative (variance must be >= 0)
    fitted = fitted.clip(lower=0)

    r2 = 1 - residuals.var() / y.var()

    log.info(
        "HAR fit  h=%d  n=%d  NW_lags=%d  R²=%.4f  coefs: %s",
        h, n, lags, r2, dict(hac_params.round(4)),
    )

    return HARResult(
        model_type="HAR",
        horizon=h,
        params=hac_params,
        hac_se=hac_se,
        ols_se=pd.Series(ols_res.bse, index=pnames, name="ols_se"),
        hac_tstat=hac_tstat,
        hac_pvalue=hac_pvalue,
        fitted=fitted,
        residuals=residuals,
        target=y.rename("target"),
        r2_insample=r2,
        hac_lags_used=lags,
        n_obs=n,
    )


# ---------------------------------------------------------------------------
# Log-HAR fit
# ---------------------------------------------------------------------------


def fit_log_har(features: pd.DataFrame, cfg: Config) -> HARResult:
    """
    Fit HAR in log(RV) space.

      log(target_t) = b0 + b_d*log(RV_d_t) + b_w*log(RV_w_t) + b_m*log(RV_m_t) + e_t

    Back-transform: RV_hat = exp(log_fitted + 0.5 * sigma_e^2)  [Jensen correction]

    Parameters
    ----------
    features : output of build_log_har_features() — columns log_RV_d etc.
    cfg      : full Config

    Returns
    -------
    HARResult with fitted values in ORIGINAL RV space (after bias correction).
    """
    h = cfg.har.horizon
    y = features["log_target"]
    X = sm.add_constant(features[["log_RV_d", "log_RV_w", "log_RV_m"]])

    ols_res = sm.OLS(y, X).fit()
    n = len(y)
    lags = _hac_lags(n, h, cfg.har.hac_lags)

    hac_res = ols_res.get_robustcov_results(cov_type="HAC", maxlags=lags, use_correction=True)

    # Re-attach param names (get_robustcov_results strips them)
    pnames = ols_res.params.index

    log_fitted = pd.Series(ols_res.fittedvalues, index=features.index)
    log_resid = pd.Series(ols_res.resid, index=features.index)
    resid_var = float(log_resid.var())

    # Jensen's inequality bias correction: E[exp(X)] = exp(E[X] + 0.5*Var(X))
    if cfg.har.log_bias_correction:
        fitted_rv = np.exp(log_fitted + 0.5 * resid_var)
        correction = resid_var
    else:
        fitted_rv = np.exp(log_fitted)
        correction = 0.0

    fitted_rv = fitted_rv.rename("fitted_logHAR")
    fitted_rv = fitted_rv.clip(lower=0)

    # Targets in original space (exp of log_target)
    target_rv = np.exp(features["log_target"]).rename("target")
    residuals_rv = (target_rv - fitted_rv).rename("resid_logHAR")

    r2 = 1 - residuals_rv.var() / target_rv.var()

    log.info(
        "log-HAR fit  h=%d  n=%d  NW_lags=%d  R²=%.4f  bias_correction=%.6f",
        h, n, lags, r2, correction,
    )

    return HARResult(
        model_type="log-HAR",
        horizon=h,
        params=pd.Series(hac_res.params, index=pnames, name="coef"),
        hac_se=pd.Series(hac_res.bse, index=pnames, name="hac_se"),
        ols_se=pd.Series(ols_res.bse, index=pnames, name="ols_se"),
        hac_tstat=pd.Series(hac_res.tvalues, index=pnames, name="t_stat"),
        hac_pvalue=pd.Series(hac_res.pvalues, index=pnames, name="p_value"),
        fitted=fitted_rv,
        residuals=residuals_rv,
        target=target_rv,
        r2_insample=r2,
        hac_lags_used=lags,
        n_obs=n,
        log_bias_correction=correction,
        residual_variance=resid_var,
    )


# ---------------------------------------------------------------------------
# In-sample diagnostics
# ---------------------------------------------------------------------------


def insample_diagnostics(result: HARResult) -> dict:
    """
    Return a dict of in-sample diagnostic statistics.

    Includes:
      - R² (in RV space)
      - RMSE, MAE
      - QLIKE loss: mean(RV/RV_hat - log(RV/RV_hat) - 1)
      - Durbin-Watson statistic on residuals
      - Mincer-Zarnowitz R² (regress actual on fitted; test b0=0, b1=1)
      - MZ slope and intercept (with NW SEs)
    """
    y = result.target
    yhat = result.fitted.reindex(y.index)

    rmse = float(np.sqrt(((y - yhat) ** 2).mean()))
    mae = float((y - yhat).abs().mean())

    # QLIKE: proper loss for variance forecasts (Patton 2011)
    # QLIKE = mean(RV/RV_hat - log(RV/RV_hat) - 1)
    # Requires yhat > 0; clip to small positive to avoid log(0)
    yhat_safe = yhat.clip(lower=1e-10)
    ratio = y / yhat_safe
    qlike = float((ratio - np.log(ratio) - 1).mean())

    dw = float(durbin_watson(result.residuals.dropna()))

    # Mincer-Zarnowitz regression: y = a + b*yhat + u
    mz_X = sm.add_constant(yhat.rename("forecast"))
    mz_res = sm.OLS(y, mz_X).fit()
    lags = _hac_lags(len(y), result.horizon, None)
    mz_hac = mz_res.get_robustcov_results(cov_type="HAC", maxlags=lags, use_correction=True)

    # get_robustcov_results strips param names — re-attach from OLS result
    mz_pnames  = mz_res.params.index
    mz_params  = pd.Series(mz_hac.params,  index=mz_pnames)
    mz_pvalues = pd.Series(mz_hac.pvalues, index=mz_pnames)

    mz_r2 = float(mz_res.rsquared)
    mz_intercept = float(mz_params["const"])
    mz_slope = float(mz_params["forecast"])
    mz_intercept_pval = float(mz_pvalues["const"])
    mz_slope_pval = float(mz_pvalues["forecast"])

    return {
        "model": result.model_type,
        "horizon": result.horizon,
        "n_obs": result.n_obs,
        "R2_insample": round(result.r2_insample, 4),
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "QLIKE": round(qlike, 6),
        "Durbin_Watson": round(dw, 4),
        "MZ_R2": round(mz_r2, 4),
        "MZ_intercept": round(mz_intercept, 6),
        "MZ_slope": round(mz_slope, 4),
        "MZ_intercept_pval": round(mz_intercept_pval, 4),
        "MZ_slope_pval": round(mz_slope_pval, 4),
        "HAC_lags": result.hac_lags_used,
    }


# ---------------------------------------------------------------------------
# Pretty-print coefficient table
# ---------------------------------------------------------------------------


def coef_table(result: HARResult) -> pd.DataFrame:
    """
    Return a tidy DataFrame of coefficients with HAC inference.
    For h > 1, includes a warning column reminding that OLS SEs are invalid.
    """
    df = pd.DataFrame({
        "coef": result.params,
        "hac_se": result.hac_se,
        "t_stat": result.hac_tstat,
        "p_value": result.hac_pvalue,
        "sig": result.hac_pvalue.apply(
            lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        ),
    })
    if result.horizon > 1:
        df["note"] = "HAC SEs; OLS SEs invalid for h>1"
    return df
