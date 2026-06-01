"""
Phase A unit tests for the convex (short-gamma) P&L model.

Covers:
  Black-Scholes toolkit
    1. put-call parity / straddle = call + put
    2. gamma, vega, theta match finite differences
    3. ATM straddle delta ~ 0
    4. long straddle P&L = -(short straddle P&L)
    5. a delta-hedged SHORT straddle's one-day loss grows ~QUADRATICALLY in the move
  Backtest engine
    6. greek attribution (gamma+theta+vega+delta_resid) reconciles with gross P&L
    7. short vol harvests the variance premium when implied > realized (gamma+theta > 0)
    8. long vol mirrors it (loses the premium)
    9. iv_mark="entry" produces exactly zero vega P&L
   10. wider option spread => strictly higher costs / lower net
   11. NO LOOK-AHEAD: perturbing a future price cannot change an earlier day's P&L
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.pnl_convex import (
    run_convex_backtest,
    straddle_greeks,
    straddle_price,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides) -> SimpleNamespace:
    base = dict(
        position_type="straddle",
        hold_days=22,
        tenor_buffer_days=2,
        iv_mark="vix",
        ladder_daily=False,
        vega_notional=1.0,
        risk_free_rate=0.0,
        dividend_yield=0.0,
        option_bid_ask_vol=0.01,
        stress_spread_mult=3.0,
        stress_vix_level=30.0,
        underlying_slippage_bps=1.0,
        hedge_cost=True,
    )
    base.update(overrides)
    return SimpleNamespace(convex_pnl=SimpleNamespace(**base))


def _synthetic_market(n=160, ann_vol=0.08, vix_level=18.0, seed=7):
    """Calm market: realized vol `ann_vol` < implied `vix_level`% (positive premium)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-02", periods=n, freq="B")
    daily = rng.normal(0, ann_vol / np.sqrt(252), n)
    spy = pd.Series(100 * np.exp(np.cumsum(daily)), index=dates, name="close")
    vix = pd.Series(vix_level, index=dates, name="vix_close", dtype=float)
    implied_var = (vix / 100.0) ** 2
    # realized forward-var proxy for the 'actual' column (linear model only; unused here)
    fwd = pd.Series(daily, index=dates).pow(2).rolling(22).mean().shift(-21) * 252
    forecasts = pd.DataFrame({
        "actual": fwd.fillna(ann_vol ** 2),
        "vix": implied_var,
        "har": implied_var * 0.5,   # VRP = vix - har > 0
    }, index=dates)
    return forecasts, spy, vix


# ---------------------------------------------------------------------------
# Black-Scholes toolkit
# ---------------------------------------------------------------------------


class TestBlackScholes:
    def test_put_call_parity_and_straddle(self):
        from scipy.stats import norm
        from src.pnl_convex import _d1_d2
        S, K, tau, sig, r, q = 100.0, 105.0, 0.25, 0.2, 0.03, 0.01
        d1, d2 = _d1_d2(S, K, tau, sig, r, q)
        call = S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        put = K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)
        # parity
        assert call - put == pytest.approx(S * np.exp(-q * tau) - K * np.exp(-r * tau), abs=1e-10)
        # straddle = call + put
        assert straddle_price(S, K, tau, sig, r, q) == pytest.approx(call + put, abs=1e-10)

    def test_greeks_match_finite_difference(self):
        S, K, tau, sig = 100.0, 100.0, 30 / 365, 0.20
        gk = straddle_greeks(S, K, tau, sig)
        h = 1e-3
        gamma_fd = (straddle_price(S + h, K, tau, sig) - 2 * straddle_price(S, K, tau, sig)
                    + straddle_price(S - h, K, tau, sig)) / h ** 2
        vega_fd = (straddle_price(S, K, tau, sig + h) - straddle_price(S, K, tau, sig - h)) / (2 * h)
        theta_fd = -(straddle_price(S, K, tau + 1 / 365, sig)
                     - straddle_price(S, K, tau - 1 / 365, sig)) / (2 / 365)
        assert gk["gamma"] == pytest.approx(gamma_fd, rel=1e-4)
        assert gk["vega"] == pytest.approx(vega_fd, rel=1e-5)
        assert gk["theta"] == pytest.approx(theta_fd, rel=1e-2)

    def test_atm_delta_near_zero_and_theta_negative(self):
        gk = straddle_greeks(100.0, 100.0, 30 / 365, 0.20)
        assert abs(gk["delta"]) < 0.05      # ATM straddle is ~delta-neutral
        assert gk["gamma"] > 0 and gk["vega"] > 0
        assert gk["theta"] < 0              # a long straddle decays

    def test_delta_hedged_short_loss_is_convex(self):
        """A delta-hedged short straddle's loss grows ~quadratically with |move|.
        Hold tau fixed so we isolate the convex PRICE response (no theta offset)."""
        S0, sig, tau = 100.0, 0.20, 30 / 365
        h0 = straddle_greeks(S0, S0, tau, sig)["delta"]  # shares to neutralize short straddle
        V0 = straddle_price(S0, S0, tau, sig)

        def loss(move):
            S1 = S0 * (1 + move)
            V1 = straddle_price(S1, S0, tau, sig)        # same tau -> pure spot convexity
            return -(-(V1 - V0) + h0 * (S1 - S0))        # loss = -(short option + stale hedge)

        l1, l2, l4 = loss(0.01), loss(0.02), loss(0.04)
        # super-linear and ~quadratic: each doubling of the move ~quadruples the loss
        assert l1 > 0 and l2 > 0 and l4 > 0
        assert 3.0 < l2 / l1 < 5.0
        assert 3.0 < l4 / l2 < 5.0


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


class TestConvexEngine:
    def test_greek_attribution_reconciles_with_gross(self):
        cfg = _cfg()
        fc, spy, vix = _synthetic_market()
        r = run_convex_backtest(fc, spy, vix, cfg, upper_threshold=-1e9, lower_threshold=-1e18)
        recon = r.greek_pnl.sum(axis=1)            # gamma+theta+vega+delta_resid, per day
        # daily reconciliation: attribution == gross book P&L to numerical precision
        np.testing.assert_allclose(recon.values, r.gross_pnl.values, atol=1e-6)

    def test_short_harvests_premium_when_implied_exceeds_realized(self):
        # IV frozen so we isolate gamma-theta; implied 18% >> realized 8%
        cfg = _cfg(iv_mark="entry")
        fc, spy, vix = _synthetic_market(ann_vol=0.08, vix_level=18.0)
        r = run_convex_backtest(fc, spy, vix, cfg, upper_threshold=-1e9, lower_threshold=-1e18)
        g = r.greek_pnl.sum()
        assert r.gross_pnl.sum() > 0            # makes money short vol in a calm tape
        assert g["gamma"] + g["theta"] > 0      # the variance premium is harvested
        assert g["vega"] == pytest.approx(0.0, abs=1e-9)  # IV frozen -> no vega

    def test_long_vol_mirrors_short(self):
        cfg = _cfg(iv_mark="entry")
        fc, spy, vix = _synthetic_market(ann_vol=0.08, vix_level=18.0)
        # always-long: VRP < lower for every day
        r = run_convex_backtest(fc, spy, vix, cfg, upper_threshold=1e18, lower_threshold=1e9,
                                delta_hedge=True)
        assert r.gross_pnl.sum() < 0            # buying rich vol in a calm tape loses
        g = r.greek_pnl.sum()
        assert g["gamma"] + g["theta"] < 0

    def test_iv_mark_entry_zero_vega(self):
        cfg = _cfg(iv_mark="entry")
        # even with a moving VIX, frozen-IV marking must produce exactly zero vega P&L
        fc, spy, vix = _synthetic_market()
        vix = vix + pd.Series(np.linspace(0, 20, len(vix)), index=vix.index)  # VIX ramps up
        r = run_convex_backtest(fc, spy, vix, cfg, upper_threshold=-1e9, lower_threshold=-1e18)
        assert r.greek_pnl["vega"].abs().sum() == pytest.approx(0.0, abs=1e-9)

    def test_wider_spread_increases_costs(self):
        fc, spy, vix = _synthetic_market()
        cheap = run_convex_backtest(fc, spy, vix, _cfg(option_bid_ask_vol=0.005),
                                    upper_threshold=-1e9, lower_threshold=-1e18)
        dear = run_convex_backtest(fc, spy, vix, _cfg(option_bid_ask_vol=0.05),
                                   upper_threshold=-1e9, lower_threshold=-1e18)
        assert dear.costs.sum() > cheap.costs.sum()
        assert dear.daily_pnl.sum() < cheap.daily_pnl.sum()

    def test_no_lookahead_future_price_cannot_change_past_pnl(self):
        cfg = _cfg()
        fc, spy, vix = _synthetic_market()
        base = run_convex_backtest(fc, spy, vix, cfg, upper_threshold=-1e9, lower_threshold=-1e18)
        # shock the FINAL price by +50%; nothing before the last few days may change
        spy2 = spy.copy()
        spy2.iloc[-1] = spy2.iloc[-1] * 1.5
        shocked = run_convex_backtest(fc, spy2, vix, cfg, upper_threshold=-1e9, lower_threshold=-1e18)
        # all days strictly before the perturbed date are identical
        cutoff = spy.index[-1]
        a = base.daily_pnl.loc[base.daily_pnl.index < cutoff]
        b = shocked.daily_pnl.loc[shocked.daily_pnl.index < cutoff]
        np.testing.assert_allclose(a.values, b.values, atol=1e-12)


# ---------------------------------------------------------------------------
# Continuous (leveraged) positions — the sizing overlay feeds w_t * signal_t.
# Magnitude = leverage (straddle size at entry), sign = direction. {-1,0,+1}
# is the leverage==1 special case, so the engine stays backward compatible.
# ---------------------------------------------------------------------------


class TestContinuousLeverage:
    def _short_signal(self, idx):
        return pd.Series(1.0, index=idx, name="signal")

    def test_leverage_scales_pnl_linearly(self):
        """A constant leverage k scales every day's P&L by exactly k (engine is linear in size)."""
        cfg = _cfg()
        fc, spy, vix = _synthetic_market()
        sig = self._short_signal(fc.index)
        base = run_convex_backtest(fc, spy, vix, cfg, signal=sig, label="1x")
        for k in (0.5, 2.0, 3.0):
            scaled = run_convex_backtest(fc, spy, vix, cfg, signal=k * sig, label=f"{k}x")
            np.testing.assert_allclose(scaled.daily_pnl.values, k * base.daily_pnl.values, atol=1e-9)

    def test_sign_separates_from_size(self):
        """Direction lives in the sign: a long book mirrors the short book's GROSS P&L at any size."""
        cfg = _cfg()
        fc, spy, vix = _synthetic_market()
        sig = self._short_signal(fc.index)
        short2 = run_convex_backtest(fc, spy, vix, cfg, signal=2.0 * sig)
        long2 = run_convex_backtest(fc, spy, vix, cfg, signal=-2.0 * sig)
        np.testing.assert_allclose(short2.gross_pnl.values, -long2.gross_pnl.values, atol=1e-9)

    def test_zero_leverage_no_position(self):
        cfg = _cfg()
        fc, spy, vix = _synthetic_market()
        flat = run_convex_backtest(fc, spy, vix, cfg, signal=pd.Series(0.0, index=fc.index))
        assert flat.daily_pnl.abs().sum() == pytest.approx(0.0, abs=1e-12)

    def test_leverage_at_entry_held_through_roll(self):
        """Leverage is sampled at entry/roll (an options book is sized at entry, not daily).
        Changing leverage AFTER entry but before a roll does not retroactively resize."""
        cfg = _cfg(hold_days=22, tenor_buffer_days=2)
        fc, spy, vix = _synthetic_market(n=40)
        sig = pd.Series(1.0, index=fc.index)
        sig.iloc[5:] = 2.0          # leverage doubles on day 5, but entry was day 0
        r = run_convex_backtest(fc, spy, vix, cfg, signal=sig)
        base = run_convex_backtest(fc, spy, vix, cfg, signal=pd.Series(1.0, index=fc.index))
        # within the first hold window the position was opened at lev=1 -> identical to base there
        np.testing.assert_allclose(r.daily_pnl.values[:20], base.daily_pnl.values[:20], atol=1e-9)
