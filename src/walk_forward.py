"""
Phase 3: Expanding-window walk-forward validation.

Design invariants:
  - Expanding window: train on all data up to t, forecast t+1..t+h.
  - Embargo: after each forecast, skip `embargo_days` before next train window
    ends, ensuring no overlap between the training target and test target.
  - Refit frequency: refit every `refit_every_days` steps (not every day).
  - No future data leaks: VIX aligned strictly to forecast date.
  - Benchmarks: random walk, rolling historical vol, VIX-implied vol.

Public API:
  run_walk_forward(rv, vix, cfg)  -> WFResult
  verdict_table(wf)               -> pd.DataFrame
  diebold_mariano(e1, e2, h)      -> dict
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.config import Config
from src.har_model import (
    HARResult,
    _hac_lags,
    build_har_features,
    build_log_har_features,
    fit_har,
    fit_log_har,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def qlike(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """
    QLIKE loss per observation: actual/forecast - log(actual/forecast) - 1.
    Proper loss for variance forecasts (Patton 2011).
    Forecast clipped to 1e-10 to avoid division by zero.
    """
    f = np.clip(forecast, 1e-10, None)
    r = actual / f
    return r - np.log(r) - 1.0


def squared_error(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return (actual - forecast) ** 2


def absolute_error(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return np.abs(actual - forecast)


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------


def diebold_mariano(
    loss1: np.ndarray,
    loss2: np.ndarray,
    h: int,
    alternative: str = "less",
) -> dict:
    """
    Diebold-Mariano (1995) test: H0: E[d_t] = 0, where d_t = loss1_t - loss2_t.
    H1 (alternative='less'): model1 has strictly lower expected loss than model2.

    Uses HAC variance estimate with lag = h-1 (Harvey, Leybourne, Newbold 1997
    small-sample correction applied).

    Parameters
    ----------
    loss1, loss2 : per-observation loss arrays for models 1 and 2
    h            : forecast horizon (used for HAC lag selection)
    alternative  : 'less' | 'greater' | 'two-sided'

    Returns
    -------
    dict with keys: dm_stat, p_value, mean_loss_diff, n
    Positive dm_stat means model1 has higher average loss than model2.
    """
    d = loss1 - loss2
    n = len(d)
    mean_d = np.mean(d)

    # Newey-West variance of d with lag = h-1
    lag = max(h - 1, 0)
    d_series = pd.Series(d)
    d_df = pd.DataFrame({"d": d_series})
    ols = sm.OLS(d_series, np.ones(n)).fit()
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=lag, use_correction=True)
    var_d = float(hac.cov_params()[0, 0])

    # Harvey-Leybourne-Newbold small-sample correction
    hlnb = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    if mean_d == 0.0:
        return {"dm_stat": 0.0, "p_value": 0.5, "mean_loss_diff": 0.0, "n": n, "h": h}
    se_d = np.sqrt(var_d) * hlnb if var_d > 0 else np.nan
    dm_stat = mean_d / se_d if se_d and np.isfinite(se_d) else np.nan

    if not np.isfinite(dm_stat):
        p_value = np.nan
    elif alternative == "less":
        p_value = float(stats.t.cdf(dm_stat, df=n - 1))
    elif alternative == "greater":
        p_value = float(stats.t.sf(dm_stat, df=n - 1))
    else:
        p_value = float(2 * stats.t.sf(abs(dm_stat), df=n - 1))

    return {
        "dm_stat": round(float(dm_stat), 4) if np.isfinite(dm_stat) else np.nan,
        "p_value": round(p_value, 4) if np.isfinite(p_value) else np.nan,
        "mean_loss_diff": round(float(mean_d), 6),
        "n": n,
        "h": h,
    }


# ---------------------------------------------------------------------------
# Walk-forward result container
# ---------------------------------------------------------------------------


@dataclass
class WFResult:
    forecasts: pd.DataFrame          # columns: date, actual, har, log_har, rw, roll_vol, vix
    metrics: pd.DataFrame            # per-model summary metrics
    dm_tests: Dict[str, dict]        # DM test results vs each benchmark
    oos_r2: Dict[str, float]         # OOS R² vs each benchmark
    verdict: str                     # plain-English verdict


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------


def _fit_and_predict(
    rv_train: pd.Series,
    rv_test_start: pd.Timestamp,
    cfg: Config,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Fit HAR and log-HAR on rv_train, return one h-step-ahead forecast
    aligned to rv_test_start.

    Returns (har_forecast, log_har_forecast) in original RV space.
    Returns (None, None) if training data is too short.
    """
    try:
        features = build_har_features(rv_train, cfg)
        if len(features) < 50:
            return None, None
        har_res = fit_har(features, cfg)

        log_features = build_log_har_features(rv_train, cfg)
        loghar_res = fit_log_har(log_features, cfg)

        # Forecast: plug in the last available row of features
        last = features.iloc[-1]
        X = np.array([1.0, last["RV_d"], last["RV_w"], last["RV_m"]])
        har_fc = float(np.dot(har_res.params.values, X))
        har_fc = max(har_fc, 0.0)

        log_last = log_features.iloc[-1]
        X_log = np.array([1.0, log_last["log_RV_d"], log_last["log_RV_w"], log_last["log_RV_m"]])
        log_fitted_val = float(np.dot(loghar_res.params.values, X_log))
        bias = loghar_res.residual_variance or 0.0
        log_har_fc = float(np.exp(log_fitted_val + 0.5 * bias)) if cfg.har.log_bias_correction else float(np.exp(log_fitted_val))
        log_har_fc = max(log_har_fc, 0.0)

        return har_fc, log_har_fc

    except Exception as exc:
        log.debug("Fit failed at %s: %s", rv_test_start, exc)
        return None, None


def run_walk_forward(
    rv: pd.Series,
    vix: Optional[pd.Series],
    cfg: Config,
) -> WFResult:
    """
    Expanding-window walk-forward validation.

    Parameters
    ----------
    rv  : daily RV series (annualized variance), sorted by date
    vix : daily VIX close series (annualized vol %, not variance), or None
    cfg : full Config

    Returns
    -------
    WFResult
    """
    rv = rv.sort_index().dropna()
    h = cfg.har.horizon
    min_train = cfg.walk_forward.min_train_days
    refit_every = cfg.walk_forward.refit_every_days
    embargo = cfg.walk_forward.embargo_days
    roll_window = cfg.walk_forward.rolling_vol_window

    # Build the full target series: mean(RV_{t+1..t+h}) for each t
    target_full = rv.shift(-1).rolling(window=h, min_periods=h).mean().shift(-(h - 1))

    records = []
    last_fit_idx = -1
    har_params = None
    loghar_params = None
    loghar_bias = 0.0

    rv_arr = rv.values
    rv_idx = rv.index

    n = len(rv)
    first_test = min_train + embargo   # first index we can use as test origin

    log.info(
        "Walk-forward: n=%d  h=%d  min_train=%d  embargo=%d  refit_every=%d",
        n, h, min_train, embargo, refit_every,
    )

    for i in range(first_test, n - h):
        t = rv_idx[i]
        actual = target_full.loc[t]
        if pd.isna(actual):
            continue

        # --- Refit if due ---
        if (i - first_test) % refit_every == 0 or har_params is None:
            train_end = i - embargo          # embargo: exclude last `embargo` days
            if train_end < min_train:
                continue
            rv_train = rv.iloc[:train_end]

            try:
                features = build_har_features(rv_train, cfg)
                if len(features) < 50:
                    continue
                har_res = fit_har(features, cfg)
                har_params = har_res.params.values

                log_features = build_log_har_features(rv_train, cfg)
                loghar_res = fit_log_har(log_features, cfg)
                loghar_params = loghar_res.params.values
                loghar_bias = loghar_res.residual_variance or 0.0
            except Exception as exc:
                log.debug("Refit failed at %s: %s", t, exc)
                continue

            last_fit_idx = i

        # --- Generate forecasts ---
        # Features use data up to t (inclusive)
        rv_d = float(rv.iloc[i])
        rv_w = float(rv.iloc[max(0, i - 4):i + 1].mean())
        rv_m = float(rv.iloc[max(0, i - 21):i + 1].mean())

        # HAR forecast
        X = np.array([1.0, rv_d, rv_w, rv_m])
        har_fc = max(float(np.dot(har_params, X)), 0.0)

        # log-HAR forecast
        log_d = np.log(max(rv_d, 1e-12))
        log_w = np.log(max(rv_w, 1e-12))
        log_m = np.log(max(rv_m, 1e-12))
        X_log = np.array([1.0, log_d, log_w, log_m])
        log_fv = float(np.dot(loghar_params, X_log))
        if cfg.har.log_bias_correction:
            log_har_fc = max(float(np.exp(log_fv + 0.5 * loghar_bias)), 0.0)
        else:
            log_har_fc = max(float(np.exp(log_fv)), 0.0)

        # --- Benchmarks ---
        # Random walk: forecast = RV_t (last observed)
        rw_fc = rv_d

        # Rolling historical vol: mean of last roll_window RV values
        roll_fc = float(rv.iloc[max(0, i - roll_window + 1):i + 1].mean())

        # VIX benchmark: VIX is annualized vol (%) → convert to annualized variance
        # VIX at time t is forward-looking implied vol for next 30 calendar days.
        # Align to forecast date t; use VIX^2 / 100^2 as implied variance.
        vix_fc = np.nan
        if vix is not None:
            vix_val = vix.get(t, np.nan)
            if not pd.isna(vix_val):
                vix_fc = (vix_val / 100.0) ** 2   # VIX% → annualized variance

        records.append({
            "date": t,
            "actual": actual,
            "har": har_fc,
            "log_har": log_har_fc,
            "rw": rw_fc,
            "roll_vol": roll_fc,
            "vix": vix_fc,
        })

    forecasts = pd.DataFrame(records).set_index("date")
    log.info("Walk-forward complete: %d forecast periods", len(forecasts))

    # --- Compute metrics ---
    metrics_rows = []
    models = ["har", "log_har", "rw", "roll_vol", "vix"]
    actual = forecasts["actual"].values

    for m in models:
        fc = forecasts[m].values
        mask = np.isfinite(fc) & np.isfinite(actual)
        if mask.sum() < 10:
            continue
        a_m, f_m = actual[mask], fc[mask]
        ql = qlike(a_m, f_m)
        se = squared_error(a_m, f_m)
        ae = absolute_error(a_m, f_m)
        metrics_rows.append({
            "model": m,
            "n": int(mask.sum()),
            "QLIKE": round(float(ql.mean()), 6),
            "RMSE": round(float(np.sqrt(se.mean())), 6),
            "MAE": round(float(ae.mean()), 6),
        })

    metrics = pd.DataFrame(metrics_rows).set_index("model")

    # --- OOS R² vs each benchmark ---
    # OOS R²(model, bench) = 1 - MSE(model) / MSE(bench)
    oos_r2 = {}
    bench_names = [b for b in ["rw", "roll_vol", "vix"] if b in metrics.index]
    har_mse = metrics.loc["har", "RMSE"] ** 2 if "har" in metrics.index else np.nan
    for b in bench_names:
        bench_mse = metrics.loc[b, "RMSE"] ** 2
        oos_r2[f"har_vs_{b}"] = round(1 - har_mse / bench_mse, 4)

    # --- Diebold-Mariano tests: HAR vs each benchmark on QLIKE ---
    dm_tests = {}
    for b in bench_names:
        mask = np.isfinite(forecasts["har"].values) & np.isfinite(forecasts[b].values) & np.isfinite(actual)
        if mask.sum() < 30:
            continue
        loss_har = qlike(actual[mask], forecasts["har"].values[mask])
        loss_b   = qlike(actual[mask], forecasts[b].values[mask])
        # H1: HAR has lower QLIKE → alternative='less' (dm_stat < 0 favors HAR)
        dm_tests[f"har_vs_{b}"] = diebold_mariano(loss_har, loss_b, h=h, alternative="less")

    # --- Verdict ---
    verdict = _build_verdict(metrics, dm_tests, oos_r2)

    return WFResult(
        forecasts=forecasts,
        metrics=metrics,
        dm_tests=dm_tests,
        oos_r2=oos_r2,
        verdict=verdict,
    )


def _build_verdict(
    metrics: pd.DataFrame,
    dm_tests: dict,
    oos_r2: dict,
) -> str:
    lines = ["WALK-FORWARD VERDICT"]
    lines.append("-" * 50)

    if "har" not in metrics.index:
        return "INCONCLUSIVE — insufficient forecast data."

    har_qlike = metrics.loc["har", "QLIKE"]

    passed = []
    failed = []

    for key, dm in dm_tests.items():
        bench = key.replace("har_vs_", "")
        bench_qlike = metrics.loc[bench, "QLIKE"] if bench in metrics.index else np.nan
        r2 = oos_r2.get(key, np.nan)
        dm_p = dm.get("p_value", np.nan)
        diff = dm.get("mean_loss_diff", np.nan)

        beat = (not pd.isna(bench_qlike)) and har_qlike < bench_qlike
        sig  = (not pd.isna(dm_p)) and dm_p < 0.05

        lines.append(
            f"  HAR vs {bench:12s}:  QLIKE {har_qlike:.4f} vs {bench_qlike:.4f}"
            f"  OOS-R²={r2:+.4f}  DM p={dm_p:.3f}  {'BEAT ✓' if beat else 'FAIL ✗'}"
            f"  {'sig' if sig else 'n.s.'}"
        )
        (passed if beat else failed).append(bench)

    lines.append("")
    if "rw" in passed:
        lines.append("GATE CLEARED: HAR beats the random-walk benchmark on QLIKE.")
        lines.append("Phase 4 (VRP signal) may proceed.")
    else:
        lines.append("GATE FAILED: HAR does NOT beat the random-walk benchmark.")
        lines.append("Do NOT proceed to Phase 4. The model has no demonstrated edge.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mincer-Zarnowitz regression (OOS)
# ---------------------------------------------------------------------------


def oos_mz_regression(forecasts: pd.DataFrame, model: str, h: int) -> dict:
    """OOS Mincer-Zarnowitz: regress actual on model forecast with HAC SEs."""
    df = forecasts[["actual", model]].dropna()
    y = df["actual"]
    X = sm.add_constant(df[model].rename("forecast"))
    ols = sm.OLS(y, X).fit()
    lags = _hac_lags(len(y), h, None)
    hac = ols.get_robustcov_results(cov_type="HAC", maxlags=lags, use_correction=True)
    pnames = ols.params.index
    params  = pd.Series(hac.params,  index=pnames)
    pvalues = pd.Series(hac.pvalues, index=pnames)
    return {
        "model": model,
        "mz_intercept": round(float(params["const"]), 6),
        "mz_slope": round(float(params["forecast"]), 4),
        "mz_intercept_pval": round(float(pvalues["const"]), 4),
        "mz_slope_pval": round(float(pvalues["forecast"]), 4),
        "mz_r2": round(float(ols.rsquared), 4),
        "n": len(y),
    }
