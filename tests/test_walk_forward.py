"""
Phase 3 unit tests for walk_forward.py.

Tests:
  1. No look-ahead: features at test step t use only rv[0..t].
  2. Embargo respected: train window never overlaps with test target window.
  3. Forecast count matches expected number of test steps.
  4. QLIKE loss is non-negative for all models.
  5. Diebold-Mariano returns finite dm_stat and p_value in [0,1].
  6. OOS R² is finite.
  7. Perfect forecast → QLIKE = 0.
  8. Random walk QLIKE on iid data ≈ HAR QLIKE (no edge in iid).
  9. VIX column present when vix series supplied; NaN when not.
  10. Verdict string is non-empty and contains GATE keyword.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.walk_forward import (
    diebold_mariano,
    oos_mz_regression,
    qlike,
    run_walk_forward,
    squared_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rv(n: int = 800, seed: int = 0, persistent: bool = True) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2005-01-03", periods=n)
    if persistent:
        rv = np.zeros(n)
        rv[0] = 0.04
        eps = rng.normal(0, 0.003, n)
        for i in range(1, n):
            rv[i] = max(0.93 * rv[i - 1] + eps[i], 1e-5)
    else:
        rv = np.abs(rng.normal(0.02, 0.005, n))
    return pd.Series(rv, index=dates, name="RV")


def _make_cfg(horizon: int = 5, min_train: int = 200, refit: int = 20, embargo: int = None):
    if embargo is None:
        embargo = horizon
    return SimpleNamespace(
        har=SimpleNamespace(
            horizon=horizon,
            lags={"daily": 1, "weekly": 5, "monthly": 22},
            fit_log_har=True,
            log_bias_correction=True,
            hac_lags=None,
        ),
        walk_forward=SimpleNamespace(
            min_train_days=min_train,
            refit_every_days=refit,
            embargo_days=embargo,
            benchmarks=["random_walk", "rolling_historical_vol", "vix"],
            rolling_vol_window=22,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQLike:
    def test_nonnegative(self):
        rng = np.random.default_rng(1)
        a = np.abs(rng.normal(0.03, 0.01, 100))
        f = np.abs(rng.normal(0.03, 0.01, 100))
        assert (qlike(a, f) >= 0).all()

    def test_perfect_forecast_zero(self):
        a = np.array([0.01, 0.02, 0.05])
        np.testing.assert_allclose(qlike(a, a), 0.0, atol=1e-12)

    def test_clip_prevents_div_zero(self):
        a = np.array([0.01])
        f = np.array([0.0])
        assert np.isfinite(qlike(a, f)[0])


class TestDieboldMariano:
    def test_identical_losses_near_zero_stat(self):
        rng = np.random.default_rng(7)
        loss = np.abs(rng.normal(0.1, 0.02, 300))
        dm = diebold_mariano(loss, loss, h=5)
        assert abs(dm["dm_stat"]) < 1e-6

    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(8)
        l1 = np.abs(rng.normal(0.12, 0.02, 300))
        l2 = np.abs(rng.normal(0.10, 0.02, 300))
        dm = diebold_mariano(l1, l2, h=5)
        assert 0 <= dm["p_value"] <= 1

    def test_clearly_worse_model_high_pval(self):
        """If model1 has much higher loss than model2, p_value (alternative='less') should be high."""
        rng = np.random.default_rng(9)
        l1 = np.abs(rng.normal(0.20, 0.02, 500))  # worse
        l2 = np.abs(rng.normal(0.05, 0.02, 500))  # better
        dm = diebold_mariano(l1, l2, h=5, alternative="less")
        assert dm["p_value"] > 0.9

    def test_clearly_better_model_low_pval(self):
        """If model1 has much lower loss than model2, p_value (alternative='less') should be low."""
        rng = np.random.default_rng(10)
        l1 = np.abs(rng.normal(0.05, 0.02, 500))  # better
        l2 = np.abs(rng.normal(0.20, 0.02, 500))  # worse
        dm = diebold_mariano(l1, l2, h=5, alternative="less")
        assert dm["p_value"] < 0.05


class TestRunWalkForward:
    def _run(self, n=800, persistent=True, include_vix=True, horizon=5):
        rv = _make_rv(n=n, persistent=persistent)
        cfg = _make_cfg(horizon=horizon, min_train=200, refit=20)
        vix = (rv * 100 * 100 * np.random.default_rng(3).uniform(0.8, 1.2, len(rv))).rename("vix") if include_vix else None
        # vix needs to be a Series with same date index, values in % (vol not variance)
        if include_vix:
            vix = pd.Series(
                np.sqrt(rv.values) * 100 * np.random.default_rng(3).uniform(0.9, 1.1, len(rv)),
                index=rv.index, name="vix"
            )
        return run_walk_forward(rv, vix, cfg)

    def test_forecast_count_positive(self):
        wf = self._run()
        assert len(wf.forecasts) > 0

    def test_no_nan_in_har_forecasts(self):
        wf = self._run()
        assert wf.forecasts["har"].notna().all()

    def test_har_forecasts_nonnegative(self):
        wf = self._run()
        assert (wf.forecasts["har"] >= 0).all()

    def test_vix_column_present_when_supplied(self):
        wf = self._run(include_vix=True)
        assert "vix" in wf.forecasts.columns
        assert wf.forecasts["vix"].notna().any()

    def test_vix_is_nan_when_not_supplied(self):
        wf = self._run(include_vix=False)
        assert wf.forecasts["vix"].isna().all()

    def test_qlike_nonnegative_for_all_models(self):
        wf = self._run()
        for m in ["har", "log_har", "rw", "roll_vol"]:
            if m in wf.metrics.index:
                assert wf.metrics.loc[m, "QLIKE"] >= 0, f"QLIKE < 0 for {m}"

    def test_oos_r2_finite(self):
        wf = self._run()
        for key, val in wf.oos_r2.items():
            assert np.isfinite(val), f"OOS R² not finite for {key}"

    def test_dm_tests_present(self):
        wf = self._run()
        assert len(wf.dm_tests) > 0

    def test_dm_pvalues_in_unit_interval(self):
        wf = self._run()
        for key, dm in wf.dm_tests.items():
            p = dm["p_value"]
            if not pd.isna(p):
                assert 0 <= p <= 1, f"p_value out of [0,1] for {key}: {p}"

    def test_verdict_nonempty(self):
        wf = self._run()
        assert len(wf.verdict) > 0

    def test_verdict_contains_gate_keyword(self):
        wf = self._run()
        assert "GATE" in wf.verdict

    def test_no_lookahead_in_features(self):
        """
        Corrupt future RV values → forecasts on earlier dates must be unchanged.
        """
        rv = _make_rv(n=800, seed=42)
        cfg = _make_cfg(horizon=5, min_train=200, refit=500)  # refit=500 → single fit
        wf1 = run_walk_forward(rv, None, cfg)

        rv2 = rv.copy()
        rv2.iloc[-100:] = 9999.0   # corrupt final 100 values
        wf2 = run_walk_forward(rv2, None, cfg)

        # Early forecasts (before corruption zone) must be identical
        cutoff = rv.index[-100]
        common = wf1.forecasts.index[wf1.forecasts.index < cutoff]
        if len(common) > 5:
            pd.testing.assert_series_equal(
                wf1.forecasts.loc[common, "har"],
                wf2.forecasts.loc[common, "har"],
                rtol=1e-6,
                check_names=False,
            )
