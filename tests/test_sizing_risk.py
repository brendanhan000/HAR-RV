"""
Phase 1 unit tests — point-in-time risk estimation for the sizing overlay.

The cardinal rule is NO LOOK-AHEAD: the volatility/covariance used to size period t
must be a function of returns STRICTLY BEFORE t (data <= t-1). The decisive tests:

  * corrupting r[t] (and everything after) must NOT change sigma_hat[t];
  * it MAY only change sigma_hat[t+1], sigma_hat[t+2], ...

Plus alignment (same index, correct NaN warm-up), correctness vs hand-computed
values, annualisation, and the Ledoit-Wolf covariance properties (PSD, shrinkage in
[0,1], better conditioning, reduces to sample with abundant data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.sizing.risk import (
    CovEstimates,
    estimate_cov,
    estimate_vol,
    ledoit_wolf_cov,
    realized_vol_ewma,
    realized_vol_rolling,
    rolling_cov,
    sample_cov,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(n: int = 500, seed: int = 0, vol: float = 0.01, mu: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    return pd.Series(mu + rng.normal(0, vol, n), index=dates, name="ret")


def _make_vol_clustered(n: int = 800, seed: int = 1) -> pd.Series:
    """GARCH-ish returns so vol genuinely varies (a real estimator should track it)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    sig2 = np.empty(n)
    sig2[0] = 1e-4
    r = np.empty(n)
    r[0] = rng.normal(0, np.sqrt(sig2[0]))
    for i in range(1, n):
        sig2[i] = 5e-6 + 0.9 * sig2[i - 1] + 0.08 * r[i - 1] ** 2
        r[i] = rng.normal(0, np.sqrt(sig2[i]))
    return pd.Series(r, index=dates, name="ret")


def _risk_cfg(estimator="ewma", **over):
    base = dict(
        estimator=estimator, window=63, rolling_min_periods=63,
        ewma_halflife=21, ewma_lambda=None, ewma_min_periods=21,
        demean=False, ddof=0, annualize=False, periods_per_year=252, vol_floor=0.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ===========================================================================
# THE no-look-ahead tests (rolling + EWMA)
# ===========================================================================


class TestNoLookAhead:
    @pytest.mark.parametrize("est", ["rolling", "ewma"])
    def test_corrupt_future_does_not_change_present(self, est):
        """sigma_hat[t] uses data < t, so corrupting r[k..] leaves sigma_hat[:k+1] intact."""
        r = _make_vol_clustered(n=600, seed=2)
        f = realized_vol_rolling if est == "rolling" else realized_vol_ewma
        kw = dict(window=63) if est == "rolling" else dict(halflife=21)
        s1 = f(r, **kw)

        k = 400
        r2 = r.copy()
        r2.iloc[k:] = 50.0  # corrupt r[k] and everything after
        s2 = f(r2, **kw)

        # values that size periods 0..k (i.e. indices 0..k) must be byte-identical:
        # sigma_hat[k] depends only on r[<k], which we did NOT touch.
        pd.testing.assert_series_equal(s1.iloc[: k + 1], s2.iloc[: k + 1], check_names=False)

    @pytest.mark.parametrize("est", ["rolling", "ewma"])
    def test_own_return_not_used(self, est):
        """Changing ONLY r[k] must leave sigma_hat[k] unchanged but move sigma_hat[k+1]."""
        r = _make_vol_clustered(n=600, seed=3)
        f = realized_vol_rolling if est == "rolling" else realized_vol_ewma
        kw = dict(window=63) if est == "rolling" else dict(halflife=21)
        s1 = f(r, **kw)

        k = 300
        r2 = r.copy()
        r2.iloc[k] = r.iloc[k] + 10.0  # perturb a single period's own return
        s2 = f(r2, **kw)

        # sizing period k must be unaffected by r[k] itself
        assert s1.iloc[k] == pytest.approx(s2.iloc[k]), "sigma_hat[k] leaked its own return"
        # the very next period DOES see r[k] -> must differ
        assert not np.isclose(s1.iloc[k + 1], s2.iloc[k + 1]), "sigma_hat[k+1] ignored new info"

    def test_dispatcher_is_also_lagged(self):
        """The config-driven entry point inherits the no-look-ahead guarantee."""
        r = _make_vol_clustered(n=500, seed=4)
        s1 = estimate_vol(r, _risk_cfg("ewma"))
        r2 = r.copy()
        r2.iloc[350:] = 99.0
        s2 = estimate_vol(r2, _risk_cfg("ewma"))
        pd.testing.assert_series_equal(s1.iloc[:351], s2.iloc[:351], check_names=False)


# ===========================================================================
# Alignment + warm-up
# ===========================================================================


class TestAlignment:
    def test_index_preserved_rolling(self):
        r = _make_returns(300)
        s = realized_vol_rolling(r, window=63)
        assert s.index.equals(r.index)
        assert len(s) == len(r)
        assert s.name == "sigma_hat"

    def test_index_preserved_ewma(self):
        r = _make_returns(300)
        s = realized_vol_ewma(r, halflife=21)
        assert s.index.equals(r.index)
        assert len(s) == len(r)

    def test_rolling_warmup_nan_then_valid(self):
        """min_periods=window=W -> indices 0..W-1 NaN, valid from index W (the shift)."""
        W = 63
        r = _make_returns(300)
        s = realized_vol_rolling(r, window=W, min_periods=W)
        assert s.iloc[:W].isna().all()
        assert s.iloc[W:].notna().all()

    def test_ewma_warmup_nan_then_valid(self):
        m = 21
        r = _make_returns(300)
        s = realized_vol_ewma(r, halflife=21, min_periods=m)
        assert s.iloc[:m].isna().all()
        assert s.iloc[m:].notna().all()

    def test_unsorted_input_is_sorted(self):
        r = _make_returns(120)
        shuffled = r.sample(frac=1.0, random_state=0)
        s = realized_vol_rolling(shuffled, window=20)
        assert s.index.equals(r.index)  # output comes back in sorted order


# ===========================================================================
# Correctness vs hand-computed values
# ===========================================================================


class TestCorrectness:
    def test_rolling_rms_matches_manual(self):
        """Default (demean=False) is zero-mean RMS over the strictly-prior window."""
        r = _make_returns(200, seed=5)
        W = 30
        s = realized_vol_rolling(r, window=W, min_periods=W, demean=False)
        t = 120
        manual = np.sqrt(np.mean(r.iloc[t - W : t].values ** 2))  # rows t-W..t-1
        assert s.iloc[t] == pytest.approx(manual, rel=1e-12)

    def test_rolling_demean_matches_std(self):
        r = _make_returns(200, seed=6)
        W = 30
        s = realized_vol_rolling(r, window=W, min_periods=W, demean=True, ddof=0)
        t = 120
        manual = r.iloc[t - W : t].std(ddof=0)
        assert s.iloc[t] == pytest.approx(manual, rel=1e-12)

    def test_ewma_matches_riskmetrics_recursion(self):
        r = _make_returns(200, seed=7)
        lam = 0.94
        s = realized_vol_ewma(r, lam=lam, min_periods=1)
        # manual RiskMetrics recursion on r^2 (adjust=False), then shift by one
        x = (r.values ** 2)
        y = np.empty_like(x)
        y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = lam * y[i - 1] + (1 - lam) * x[i]
        manual_sigma = np.sqrt(y)
        t = 150
        assert s.iloc[t] == pytest.approx(manual_sigma[t - 1], rel=1e-10)

    def test_ewma_lambda_equals_equivalent_halflife(self):
        r = _make_returns(300, seed=8)
        lam = 0.94
        hl = np.log(0.5) / np.log(lam)  # halflife with the same alpha
        s_lam = realized_vol_ewma(r, lam=lam, min_periods=1)
        s_hl = realized_vol_ewma(r, halflife=hl, min_periods=1)
        pd.testing.assert_series_equal(s_lam, s_hl, check_names=False, rtol=1e-9)

    def test_annualization_scales_by_sqrt_periods(self):
        r = _make_returns(200, seed=9)
        raw = realized_vol_rolling(r, window=40, annualize=False)
        ann = realized_vol_rolling(r, window=40, annualize=True, periods_per_year=252)
        ratio = (ann / raw).dropna()
        assert np.allclose(ratio.values, np.sqrt(252), rtol=1e-12)

    def test_zero_returns_give_zero_vol(self):
        r = pd.Series(np.zeros(100), index=pd.bdate_range("2010-01-04", periods=100))
        assert realized_vol_rolling(r, window=20).dropna().abs().max() == 0.0
        assert realized_vol_ewma(r, halflife=10, min_periods=5).dropna().abs().max() == 0.0

    def test_constant_returns_zero_under_demean(self):
        r = pd.Series(np.full(100, 0.003), index=pd.bdate_range("2010-01-04", periods=100))
        s = realized_vol_rolling(r, window=20, demean=True, ddof=0)
        assert s.dropna().abs().max() == pytest.approx(0.0, abs=1e-15)

    def test_estimator_tracks_vol_regime(self):
        """Sanity: estimator rises in the high-vol half of a regime-switch series."""
        rng = np.random.default_rng(11)
        lo = rng.normal(0, 0.005, 300)
        hi = rng.normal(0, 0.03, 300)
        r = pd.Series(np.concatenate([lo, hi]), index=pd.bdate_range("2010-01-04", periods=600))
        s = realized_vol_ewma(r, halflife=21)
        assert s.iloc[500] > 3 * s.iloc[250]


# ===========================================================================
# Estimator input validation
# ===========================================================================


class TestValidation:
    def test_ewma_requires_one_decay_param(self):
        r = _make_returns(50)
        with pytest.raises(ValueError):
            realized_vol_ewma(r)  # none
        with pytest.raises(ValueError):
            realized_vol_ewma(r, halflife=10, span=20)  # two

    def test_ewma_lambda_range(self):
        r = _make_returns(50)
        with pytest.raises(ValueError):
            realized_vol_ewma(r, lam=1.5)

    def test_rolling_window_minimum(self):
        r = _make_returns(50)
        with pytest.raises(ValueError):
            realized_vol_rolling(r, window=1)

    def test_dispatcher_unknown_estimator(self):
        r = _make_returns(50)
        with pytest.raises(ValueError):
            estimate_vol(r, _risk_cfg("garch"))

    def test_non_series_rejected(self):
        with pytest.raises(TypeError):
            realized_vol_rolling(np.arange(50).astype(float), window=10)


# ===========================================================================
# Ledoit-Wolf shrinkage covariance
# ===========================================================================


class TestLedoitWolf:
    def test_shrinkage_in_unit_interval(self):
        rng = np.random.default_rng(20)
        X = rng.normal(0, 1, (60, 5))
        _, delta = ledoit_wolf_cov(X)
        assert 0.0 <= delta <= 1.0

    def test_psd_and_symmetric(self):
        rng = np.random.default_rng(21)
        X = rng.normal(0, 1, (40, 6))
        Sigma, _ = ledoit_wolf_cov(X)
        assert np.allclose(Sigma, Sigma.T)
        eig = np.linalg.eigvalsh(Sigma)
        assert eig.min() > -1e-10

    def test_better_conditioned_than_sample_when_p_near_n(self):
        rng = np.random.default_rng(22)
        X = rng.normal(0, 1, (30, 12))  # p close to n -> sample cov ill-conditioned
        S, _ = sample_cov(X)
        Sigma, delta = ledoit_wolf_cov(X)
        assert delta > 0.0
        assert np.linalg.cond(Sigma) < np.linalg.cond(S)

    def test_reduces_toward_sample_with_abundant_data(self):
        """True cov FAR from the scaled-identity target + n>>p -> LW trusts the data (delta -> 0)."""
        rng = np.random.default_rng(23)
        L = np.array([[1, 0, 0, 0], [0.9, 0.4, 0, 0], [0.8, 0.1, 0.3, 0], [0.2, 0.2, 0.2, 0.5]])
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            X = (rng.normal(0, 1, (8000, 4)) @ L.T) * np.array([0.5, 1.0, 2.0, 3.0])  # Accelerate FP-flag quirk
        _, delta = ledoit_wolf_cov(X)
        assert delta < 0.05

    def test_full_shrinkage_when_target_is_the_truth(self):
        """If the true cov IS scaled identity, the sample's deviations are pure noise -> delta -> 1."""
        rng = np.random.default_rng(27)
        X = rng.normal(0, 1, (40, 8))  # iid unit variance == the shrinkage target
        _, delta = ledoit_wolf_cov(X)
        assert delta > 0.8

    def test_shrinks_spurious_offdiagonals(self):
        """True cov is identity; LW should pull spurious sample correlations toward 0."""
        rng = np.random.default_rng(24)
        X = rng.normal(0, 1, (40, 8))
        S, _ = sample_cov(X)
        Sigma, _ = ledoit_wolf_cov(X)
        off = ~np.eye(8, dtype=bool)
        assert np.abs(Sigma[off]).sum() < np.abs(S[off]).sum()

    def test_scalar_case_no_shrinkage(self):
        rng = np.random.default_rng(25)
        X = rng.normal(0, 1, (100, 1))
        Sigma, delta = ledoit_wolf_cov(X)
        assert delta == 0.0
        assert Sigma.shape == (1, 1)
        assert Sigma[0, 0] == pytest.approx(X.var(ddof=0), rel=1e-12)

    def test_sample_cov_matches_numpy(self):
        rng = np.random.default_rng(26)
        X = rng.normal(0, 1, (200, 4))
        S, delta = sample_cov(X)
        assert delta == 0.0
        np.testing.assert_allclose(S, np.cov(X, rowvar=False, bias=True), rtol=1e-12)


# ===========================================================================
# Point-in-time rolling covariance (multi-position path)
# ===========================================================================


def _make_panel(n=600, p=3, seed=30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    return pd.DataFrame(rng.normal(0, 0.01, (n, p)),
                        index=dates, columns=[f"strat{i}" for i in range(p)])


class TestRollingCov:
    def test_no_lookahead(self):
        df = _make_panel()
        c1 = rolling_cov(df, window=126, min_periods=126, method="ledoit_wolf")
        df2 = df.copy()
        df2.iloc[400:] = 9.0  # corrupt from row 400 on
        c2 = rolling_cov(df2, window=126, min_periods=126, method="ledoit_wolf")
        # matrices dated on/before row 400 use only rows < their date -> unchanged
        cutoff = df.index[400]
        for t, M in c1.matrices.items():
            if t <= cutoff:
                np.testing.assert_allclose(M, c2.matrices[t], rtol=1e-12,
                                           err_msg=f"look-ahead leak at {t.date()}")

    def test_matrix_uses_strictly_prior_window(self):
        df = _make_panel()
        W = 126
        c = rolling_cov(df, window=W, min_periods=W, method="sample", annualize=False)
        i = 300
        t = df.index[i]
        expected, _ = sample_cov(df.iloc[i - W : i].values)  # rows i-W .. i-1
        np.testing.assert_allclose(c.at(t), expected, rtol=1e-12)

    def test_warmup_no_matrix_before_min_periods(self):
        df = _make_panel()
        c = rolling_cov(df, window=200, min_periods=150)
        assert all(t >= df.index[150] for t in c.dates)

    def test_all_matrices_psd(self):
        df = _make_panel()
        c = rolling_cov(df, window=150, min_periods=150, method="ledoit_wolf")
        for M in c.matrices.values():
            assert np.linalg.eigvalsh(M).min() > -1e-10

    def test_annualization_scales_variance(self):
        df = _make_panel()
        raw = rolling_cov(df, window=150, method="sample", annualize=False)
        ann = rolling_cov(df, window=150, method="sample", annualize=True, periods_per_year=252)
        t = raw.dates[len(raw.dates) // 2]
        np.testing.assert_allclose(ann.at(t), raw.at(t) * 252, rtol=1e-12)

    def test_estimate_cov_dispatch(self):
        df = _make_panel()
        cov_cfg = SimpleNamespace(method="ledoit_wolf", window=150, min_periods=150,
                                  annualize=True, periods_per_year=252)
        c = estimate_cov(df, cov_cfg)
        assert isinstance(c, CovEstimates)
        assert c.assets == list(df.columns)
        assert len(c.dates) > 0
