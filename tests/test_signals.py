"""
Tests for the signal-validation framework (shared validation utils + Signal 1 spread).

Centre of gravity: the no-look-ahead guarantee (corrupt the future -> traded positions
over [0..k] byte-identical), correct OU/cointegration mechanics, correct P&L signs, and
the honest-referee utilities (Benjamini-Hochberg, block-bootstrap, sustained excursion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.spread import (
    build_spread, zscore, generate_spread_signal, spread_pnl, ou_halflife,
)
from src.signals.validation import (
    split_dev_holdout, perf_metrics, worst_sustained_excursion,
    block_bootstrap_sharpe_p, benjamini_hochberg,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _cointegrated(n=900, seed=0, b=0.9):
    """BNO ~ random walk; USO = BNO + stationary AR(1) residual (genuinely reverting)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n)
    log_bno = np.cumsum(rng.normal(0, 0.01, n)) + np.log(20)
    resid = np.zeros(n)
    for i in range(1, n):
        resid[i] = b * resid[i - 1] + rng.normal(0, 0.02)
    log_uso = log_bno + resid
    return (pd.Series(np.exp(log_uso), index=dates, name="uso"),
            pd.Series(np.exp(log_bno), index=dates, name="bno"))


# ---------------------------------------------------------------------------
# No look-ahead (the cardinal test)
# ---------------------------------------------------------------------------


class TestNoLookAhead:
    def test_corrupt_future_leaves_positions_unchanged(self):
        uso, bno = _cointegrated(900, seed=1)
        df = build_spread(uso, bno, beta_window=120)
        df["z"] = zscore(df["spread"], window=40)
        pos1 = spread_pnl(df, generate_spread_signal(df["z"], 2.0, 0.5), 5.0).position

        k = 600
        uso2 = uso.copy(); uso2.iloc[k:] *= 1.5    # corrupt the future
        bno2 = bno.copy(); bno2.iloc[k:] *= 0.7
        df2 = build_spread(uso2, bno2, beta_window=120)
        df2["z"] = zscore(df2["spread"], window=40)
        pos2 = spread_pnl(df2, generate_spread_signal(df2["z"], 2.0, 0.5), 5.0).position

        # positions traded on days strictly before the corruption use only data <= t-1
        pd.testing.assert_series_equal(pos1.iloc[:k], pos2.iloc[:k], check_names=False)


# ---------------------------------------------------------------------------
# Hedge ratio / spread / OU mechanics
# ---------------------------------------------------------------------------


class TestSpreadMechanics:
    def test_hedge_ratio_recovers_unit_beta(self):
        uso, bno = _cointegrated(900, seed=2)
        df = build_spread(uso, bno, beta_window=250)
        assert df["beta"].dropna().median() == pytest.approx(1.0, abs=0.25)

    def test_spread_is_mean_zero_residual(self):
        uso, bno = _cointegrated(900, seed=3)
        df = build_spread(uso, bno, beta_window=250)
        assert abs(df["spread"].dropna().mean()) < 0.02

    def test_ou_halflife_recovers_known_ar1(self):
        rng = np.random.default_rng(4)
        b = 0.9
        x = np.zeros(4000)
        for i in range(1, len(x)):
            x[i] = b * x[i - 1] + rng.normal(0, 0.1)
        hl = ou_halflife(pd.Series(x))
        assert hl == pytest.approx(-np.log(2) / np.log(b), rel=0.25)   # ~6.6 days

    def test_ou_halflife_random_walk_is_infinite(self):
        rng = np.random.default_rng(5)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 3000)))
        assert ou_halflife(rw) == np.inf or ou_halflife(rw) > 500


# ---------------------------------------------------------------------------
# Signal state machine + P&L signs
# ---------------------------------------------------------------------------


class TestSignalAndPnL:
    def test_entry_exit_state_machine(self):
        z = pd.Series([0, 1.0, 2.5, 1.0, 0.3, -1.0, -2.5, -0.3, 0.0],
                      index=pd.bdate_range("2012-01-02", periods=9))
        pos = generate_spread_signal(z, z_entry=2.0, z_exit=0.5)
        # rich (z>2) -> short (-1), hold until |z|<0.5, then cheap (z<-2) -> long (+1)
        assert pos.iloc[2] == -1 and pos.iloc[3] == -1     # entered short, still |z|>0.5
        assert pos.iloc[4] == 0                            # |z|=0.3<0.5 -> exit
        assert pos.iloc[6] == 1                            # z=-2.5 -> long
        assert pos.iloc[7] == 0                            # |z|=0.3 -> exit

    def test_long_spread_profits_when_uso_outperforms(self):
        dates = pd.bdate_range("2012-01-02", periods=5)
        uso = pd.Series([10, 10, 11, 11, 11], index=dates, dtype=float)   # USO jumps day 3
        bno = pd.Series([20, 20, 20, 20, 20], index=dates, dtype=float)   # BNO flat
        df = pd.DataFrame({"uso": uso, "bno": bno})
        df["beta"] = 1.0
        pos_raw = pd.Series([0, 1, 1, 0, 0], index=dates, dtype=float)    # long spread day 2->
        res = spread_pnl(df, pos_raw, one_way_bps=0.0)
        # position lags one day -> long active on day index 2 (the +10% USO day) -> positive
        assert res.gross_return.iloc[2] > 0

    def test_cost_reduces_return_on_turnover(self):
        uso, bno = _cointegrated(400, seed=6)
        df = build_spread(uso, bno, 120); df["z"] = zscore(df["spread"], 40)
        raw = generate_spread_signal(df["z"], 2.0, 0.5)
        free = spread_pnl(df, raw, one_way_bps=0.0).net_return.sum()
        dear = spread_pnl(df, raw, one_way_bps=20.0).net_return.sum()
        assert dear < free


# ---------------------------------------------------------------------------
# Validation utilities (the honest referee)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_split_is_strict(self):
        s = pd.Series(range(10), index=pd.bdate_range("2022-12-20", periods=10))
        dev, ho = split_dev_holdout(s, "2023-01-01")
        assert dev.index.max() < pd.Timestamp("2023-01-01") <= ho.index.min()
        assert len(dev) + len(ho) == len(s)

    def test_perf_metrics_skew_and_sharpe_sign(self):
        rng = np.random.default_rng(7)
        # unambiguous positive drift (mean ~11 SE above 0) so the Sharpe sign is deterministic
        r = pd.Series(rng.normal(0.005, 0.01, 500), index=pd.bdate_range("2012-01-02", periods=500))
        m = perf_metrics(r)
        assert m["sharpe"] > 0 and np.sign(m["sharpe"]) == np.sign(r.mean()) and np.isfinite(m["skew"])

    def test_worst_sustained_excursion_flags_long_underwater(self):
        # steady losing stream -> underwater nearly the whole time, large maxDD
        r = pd.Series([-0.001] * 300, index=pd.bdate_range("2012-01-02", periods=300))
        exc = worst_sustained_excursion(r, position=pd.Series(1.0, index=r.index))
        assert exc["max_drawdown"] < 0 and exc["max_drawdown_days"] >= 290

    def test_block_bootstrap_zero_mean_not_significant(self):
        rng = np.random.default_rng(8)
        r = pd.Series(rng.normal(0, 0.01, 600), index=pd.bdate_range("2012-01-02", periods=600))
        assert block_bootstrap_sharpe_p(r, n_boot=2000, seed=1)["p_value"] > 0.10

    def test_block_bootstrap_strong_signal_significant(self):
        rng = np.random.default_rng(9)
        r = pd.Series(rng.normal(0.0015, 0.005, 600), index=pd.bdate_range("2012-01-02", periods=600))
        assert block_bootstrap_sharpe_p(r, n_boot=2000, seed=1)["p_value"] < 0.05

    def test_benjamini_hochberg_threshold(self):
        bh = benjamini_hochberg({"a": 0.001, "b": 0.04, "c": 0.20, "d": 0.50}, fdr=0.05)
        # ranks/bounds: a .0125, b .025, c .0375, d .05 -> only a (0.001<=0.0125) passes
        assert bh["results"]["a"]["significant"] is True
        assert bh["results"]["b"]["significant"] is False
        assert bh["n_significant"] == 1

    def test_benjamini_hochberg_all_null(self):
        bh = benjamini_hochberg({"a": 0.6, "b": 0.7, "c": 0.8, "d": 0.9}, fdr=0.05)
        assert bh["n_significant"] == 0
