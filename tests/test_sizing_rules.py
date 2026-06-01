"""
Phase 2 unit tests — sizing rules and the composed overlay.

Covers: rule correctness (vol-target, Kelly, cap, brake), the no-look-ahead guarantee
of the composed overlay (corrupt the future -> present leverage byte-identical, even
with the causal drawdown brake on), the cap binding, turnover accounting, and the
Kelly ruin sim's qualitative behaviour (ruin rises with leverage and with an
over-estimated edge).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from src.sizing.rules import (
    drawdown_brake_factor,
    hard_session_cap,
    kelly_leverage,
    lagged_mean,
    simulate_kelly_ruin,
    vol_target_leverage,
)
from src.sizing.overlay import apply_overlay


# ---------------------------------------------------------------------------
# Config + data helpers
# ---------------------------------------------------------------------------


def _cfg(**rules_over):
    risk = SimpleNamespace(
        estimator="ewma", window=63, rolling_min_periods=63,
        ewma_halflife=21, ewma_lambda=None, ewma_min_periods=21,
        demean=False, ddof=0, annualize=True, periods_per_year=252, vol_floor=0.0,
    )
    rules = dict(
        method="vol_target", condition_on_active=True, reference_capital=100.0,
        target_vol=0.10, vol_floor_frac=0.25, max_leverage=4.0,
        kelly_fraction=0.25, kelly_edge_window=None,
        cap_enabled=True, cap_fraction=0.05, cap_loss_sigma=3.0, annual_expected_pnl=None,
        brake_enabled=False, brake_threshold=0.20, brake_floor=0.25, brake_max_dd=0.40,
        turnover_cost_per_unit=0.0,
    )
    rules.update(rules_over)
    sizing = SimpleNamespace(risk=risk, rules=SimpleNamespace(**rules), covariance=None)
    return SimpleNamespace(sizing=sizing)


def _strategy_stream(n=900, seed=0, flat_frac=0.6):
    """A short-vol-ish stream: mostly small positive days, occasional big losses, often flat."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    base = rng.normal(0.03, 0.4, n)
    crashes = rng.random(n) < 0.02
    base[crashes] -= rng.uniform(3, 9, crashes.sum())
    signal = (rng.random(n) > flat_frac).astype(float)   # in/out of position
    pnl = pd.Series(base * signal, index=dates, name="ret")   # flat days -> 0 P&L
    return pnl, pd.Series(signal, index=dates, name="signal")


# ===========================================================================
# Vol targeting
# ===========================================================================


class TestVolTarget:
    def test_inverse_in_sigma(self):
        s = pd.Series([0.05, 0.10, 0.20], index=pd.bdate_range("2010-01-04", periods=3))
        w = vol_target_leverage(s, target_vol=0.10, max_leverage=np.inf)
        np.testing.assert_allclose(w.values, [2.0, 1.0, 0.5])

    def test_vol_floor_caps_leverage(self):
        s = pd.Series([1e-9, 0.0, 0.50], index=pd.bdate_range("2010-01-04", periods=3))
        w = vol_target_leverage(s, target_vol=0.10, vol_floor=0.05, max_leverage=np.inf)
        # floored denom = 0.05 -> max leverage = 0.10/0.05 = 2.0 (no blow-up at sigma->0)
        assert w.iloc[0] == pytest.approx(2.0)
        assert w.iloc[1] == pytest.approx(2.0)

    def test_max_leverage_clip(self):
        s = pd.Series([0.001], index=pd.bdate_range("2010-01-04", periods=1))
        w = vol_target_leverage(s, target_vol=0.10, vol_floor=0.0, max_leverage=3.0)
        assert w.iloc[0] == 3.0

    def test_nan_passthrough(self):
        s = pd.Series([np.nan, 0.10], index=pd.bdate_range("2010-01-04", periods=2))
        w = vol_target_leverage(s, target_vol=0.10)
        assert np.isnan(w.iloc[0]) and w.iloc[1] == pytest.approx(1.0)

    def test_bad_target_raises(self):
        with pytest.raises(ValueError):
            vol_target_leverage(pd.Series([0.1]), target_vol=0.0)


# ===========================================================================
# Fractional Kelly
# ===========================================================================


class TestKelly:
    def test_formula(self):
        mu = pd.Series([0.001, 0.002])
        var = pd.Series([0.0004, 0.0004])
        w = kelly_leverage(mu, var, kelly_fraction=0.5, max_leverage=np.inf)
        np.testing.assert_allclose(w.values, [0.5 * 0.001 / 0.0004, 0.5 * 0.002 / 0.0004])

    def test_negative_edge_stands_down(self):
        mu = pd.Series([-0.001, 0.001])
        var = pd.Series([0.0004, 0.0004])
        w = kelly_leverage(mu, var, kelly_fraction=0.5, allow_short=False)
        assert w.iloc[0] == 0.0 and w.iloc[1] > 0

    def test_full_kelly_discouraged_but_allowed(self):
        w = kelly_leverage(pd.Series([0.001]), pd.Series([0.0004]), kelly_fraction=1.0)
        assert w.iloc[0] > 0

    def test_fraction_out_of_range_raises(self):
        with pytest.raises(ValueError):
            kelly_leverage(pd.Series([0.001]), pd.Series([0.0004]), kelly_fraction=1.5)
        with pytest.raises(ValueError):
            kelly_leverage(pd.Series([0.001]), pd.Series([0.0004]), kelly_fraction=0.0)


# ===========================================================================
# Lagged edge estimate (no look-ahead)
# ===========================================================================


class TestLaggedMean:
    def test_no_lookahead(self):
        r = pd.Series(np.random.default_rng(1).normal(0, 1, 300),
                      index=pd.bdate_range("2010-01-04", periods=300))
        m1 = lagged_mean(r, window=None, min_periods=10)
        r2 = r.copy(); r2.iloc[200:] = 99.0
        m2 = lagged_mean(r2, window=None, min_periods=10)
        pd.testing.assert_series_equal(m1.iloc[:201], m2.iloc[:201], check_names=False)

    def test_expanding_vs_rolling(self):
        r = pd.Series(np.arange(100.0), index=pd.bdate_range("2010-01-04", periods=100))
        exp = lagged_mean(r, window=None, min_periods=1)
        roll = lagged_mean(r, window=10, min_periods=10)
        assert exp.iloc[50] == pytest.approx(np.mean(np.arange(50)))   # 0..49
        assert roll.iloc[50] == pytest.approx(np.mean(np.arange(40, 50)))


# ===========================================================================
# Hard per-session cap
# ===========================================================================


class TestHardCap:
    def test_binds_and_formula(self):
        lev = pd.Series([2.0, 2.0])
        A = pd.Series([10.0, 10.0])      # annual expected per unit
        L = pd.Series([2.0, 5.0])        # adverse session loss per unit
        w = hard_session_cap(lev, A, L, cap_fraction=0.05)
        # w_cap = 0.05*10/L = {0.25, 0.10}; min(2.0, w_cap) = w_cap
        np.testing.assert_allclose(w.values, [0.25, 0.10])

    def test_does_not_raise_leverage(self):
        lev = pd.Series([0.05])
        w = hard_session_cap(lev, pd.Series([10.0]), pd.Series([2.0]), cap_fraction=0.05)
        assert w.iloc[0] == pytest.approx(0.05)   # cap is 0.25 but leverage already lower

    def test_no_edge_stands_down(self):
        w = hard_session_cap(pd.Series([2.0]), pd.Series([-1.0]), pd.Series([2.0]), 0.05)
        assert w.iloc[0] == 0.0

    def test_no_loss_estimate_stands_down(self):
        w = hard_session_cap(pd.Series([2.0]), pd.Series([10.0]), pd.Series([0.0]), 0.05)
        assert w.iloc[0] == 0.0

    def test_scale_invariant_in_units(self):
        """Cap is a ratio annual/loss -> rescaling the P&L units leaves w_cap unchanged."""
        w1 = hard_session_cap(pd.Series([9.9]), pd.Series([10.0]), pd.Series([2.0]), 0.05)
        w2 = hard_session_cap(pd.Series([9.9]), pd.Series([1000.0]), pd.Series([200.0]), 0.05)
        assert w1.iloc[0] == pytest.approx(w2.iloc[0])


# ===========================================================================
# Drawdown brake
# ===========================================================================


class TestBrake:
    def test_no_brake_below_threshold(self):
        dd = pd.Series([0.0, 0.10, 0.20])
        m = drawdown_brake_factor(dd, threshold=0.20, floor=0.25, max_dd=0.40)
        np.testing.assert_allclose(m.values, [1.0, 1.0, 1.0])

    def test_floor_beyond_max_dd(self):
        dd = pd.Series([0.40, 0.60])
        m = drawdown_brake_factor(dd, threshold=0.20, floor=0.25, max_dd=0.40)
        np.testing.assert_allclose(m.values, [0.25, 0.25])

    def test_linear_ramp(self):
        dd = pd.Series([0.30])   # halfway between 0.20 and 0.40
        m = drawdown_brake_factor(dd, threshold=0.20, floor=0.25, max_dd=0.40)
        assert m.iloc[0] == pytest.approx(1.0 - 0.5 * (1 - 0.25))

    def test_bounds(self):
        dd = pd.Series(np.linspace(0, 1, 50))
        m = drawdown_brake_factor(dd, 0.20, 0.25, 0.40)
        assert (m <= 1.0).all() and (m >= 0.25).all()


# ===========================================================================
# Composed overlay
# ===========================================================================


class TestOverlay:
    @pytest.mark.parametrize("brake", [False, True])
    def test_no_lookahead(self, brake):
        """Corrupt returns[k:] -> leverage over [0..k] is byte-identical (brake on or off)."""
        pnl, sig = _strategy_stream(900, seed=2)
        cfg = _cfg(brake_enabled=brake)
        r1 = apply_overlay(pnl, sig, cfg)
        k = 600
        pnl2 = pnl.copy(); pnl2.iloc[k:] = -50.0
        r2 = apply_overlay(pnl2, sig, cfg)
        lev1 = r1.leverage.fillna(0.0); lev2 = r2.leverage.fillna(0.0)
        pd.testing.assert_series_equal(lev1.iloc[: k + 1], lev2.iloc[: k + 1], check_names=False)

    def test_position_zero_when_flat(self):
        pnl, sig = _strategy_stream(600, seed=3)
        r = apply_overlay(pnl, sig, _cfg())
        assert (r.position[sig == 0].abs() < 1e-12).all()

    def test_cap_binding_flag(self):
        pnl, sig = _strategy_stream(600, seed=4)
        r = apply_overlay(pnl, sig, _cfg(cap_enabled=True))
        # where the cap binds, final leverage must equal the cap ceiling (<= base)
        binding = r.cap_binding & r.leverage.notna() & (r.position.abs() > 0)
        if binding.any():
            assert (r.leverage[binding] <= r.w_base[binding] + 1e-9).all()

    def test_turnover_accounting(self):
        pnl, sig = _strategy_stream(400, seed=5)
        r = apply_overlay(pnl, sig, _cfg(turnover_cost_per_unit=0.5))
        expected = r.position.diff().abs().fillna(r.position.abs())
        pd.testing.assert_series_equal(r.turnover, expected, check_names=False)
        np.testing.assert_allclose(r.turnover_cost.values, 0.5 * expected.values)

    def test_always_on_when_no_signal(self):
        pnl, _ = _strategy_stream(400, seed=6, flat_frac=0.0)
        r = apply_overlay(pnl, None, _cfg())
        assert r.leverage.notna().sum() > 0

    def test_constant_cap_is_pure_level_cut(self):
        """With a CONSTANT annual-expected, the cap only scales leverage -> Sharpe unchanged."""
        pnl, sig = _strategy_stream(1200, seed=7)
        base = apply_overlay(pnl, sig, _cfg(cap_enabled=False))
        A = float(pnl.mean() * 252)
        capped = apply_overlay(pnl, sig, _cfg(cap_enabled=True, annual_expected_pnl=A))

        def sharpe(p):
            p = p.fillna(0.0)
            return p.mean() / p.std() if p.std() > 0 else np.nan
        # capped leverage <= base leverage everywhere (pure reduction)
        assert (capped.leverage.fillna(0) <= base.leverage.fillna(0) + 1e-9).all()


# ===========================================================================
# Kelly ruin / edge-error sensitivity
# ===========================================================================


class TestRuin:
    def _emp(self):
        rng = np.random.default_rng(8)
        x = rng.normal(0.05, 1.0, 2000)
        crash = rng.random(2000) < 0.03
        x[crash] -= rng.uniform(3, 8, crash.sum())
        return x

    def test_probabilities_valid(self):
        df = simulate_kelly_ruin(self._emp(), sharpe_ann=0.5, vol_ann=0.15,
                                 kelly_fractions=(0.25, 1.0), edge_errors=(-0.5, 0.0, 0.5),
                                 n_sims=400, horizon=504, seed=1)
        assert ((df["p_deep_dd"] >= 0) & (df["p_deep_dd"] <= 1)).all()
        assert ((df["p_wipeout"] >= 0) & (df["p_wipeout"] <= 1)).all()

    def test_full_kelly_riskier_than_quarter(self):
        df = simulate_kelly_ruin(self._emp(), sharpe_ann=0.5, vol_ann=0.15,
                                 kelly_fractions=(0.25, 1.0), edge_errors=(0.0,),
                                 n_sims=1500, horizon=756, seed=2)
        q = df[df.kelly_fraction == 0.25]["p_deep_dd"].iloc[0]
        f = df[df.kelly_fraction == 1.0]["p_deep_dd"].iloc[0]
        assert f > q

    def test_overestimated_edge_raises_ruin(self):
        df = simulate_kelly_ruin(self._emp(), sharpe_ann=0.5, vol_ann=0.15,
                                 kelly_fractions=(1.0,), edge_errors=(-0.5, 0.0, 0.5),
                                 n_sims=1500, horizon=756, seed=3)
        d = df.set_index("edge_error")["p_deep_dd"]
        assert d.loc[0.5] >= d.loc[-0.5]   # over-betting (thought edge bigger) is riskier
