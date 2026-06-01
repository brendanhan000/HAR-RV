# Risk & Position-Sizing Overlay — Final Verdict (Phase 4)

**Question (the prime directive):** does a disciplined, point-in-time sizing overlay improve a
strategy's **risk-adjusted** outcome and **reduce its drawdown-to-annual ratio**, net of costs —
or does it only cap drawdown, or add nothing over flat sizing?

**Answer: it does NOT improve risk-adjusted return, and does NOT reduce the drawdown/annual
ratio.** On the surviving elevated-VIX short-vol strategy, every overlay variant earns a *lower*
net Sharpe than flat sizing, and the drawdown/annual ratio gets *worse*, not better. The overlay's
**one defensible contribution is the hard per-session cap as an adaptive, binding loss limit**: it
slashes *absolute* drawdown and worst-day loss (capital preservation / blow-up prevention) by
running the book smaller — but since flat sizing's Sharpe and DD/annual ratio are scale-invariant,
**trading flat at a smaller notional dominates the overlay on both risk-adjusted metrics.** Dynamic
vol-targeting and fractional Kelly actively *hurt* here. This is the brief's anticipated outcome —
*"sizing adds little beyond the drawdown cap"* — and, if anything, stronger.

> Research hypothesis test, not a trading system. SPY 2006–2024, Yang-Zhang daily RV, VIX as the
> implied-vol mark, convex (short-gamma) straddle P&L with stress-widened costs. The sizing inputs
> (vol target, Kelly fraction, cap) are config **priors**, not tuned to the backtest.

---

## One table — flat vs the overlay (net of convex P&L + stress-widened costs)

Sized position `w_t · signal_t` run **natively** through the convex engine (linear in straddle
size; leverage sampled at entry/roll). DD/Ann = |max drawdown| ÷ annual P&L (lower = better tail).
Sharpe and DD/Ann are **scale-invariant** (independent of the notional); maxDD, worst-day and
turnover are absolute. Vol-matched paired DM test asks whether the overlay beats flat at *equal
risk* (rewards timing, not leverage level).

| Variant | net Sharpe | max DD | **DD/Ann** | worst day | turnover/yr | vol-matched DM vs flat |
|---|--:|--:|--:|--:|--:|---|
| **flat (unit)** — benchmark | **0.385** | −71.0 | **9.1** | −19.0 | 12.0 | — |
| vol-target | 0.269 | −22.9 | 13.4 | −6.9 | 4.3 | worse (t −1.50) |
| vol-target + hard cap (stable *A*) | 0.269 | −4.7 | 13.4 | −1.4 | 0.9 | worse (t −1.50) |
| vol-target + hard cap (point-in-time *A*) | 0.169 | −3.6 | 22.2 | −1.5 | 0.8 | worse (t −1.53) |
| fractional (¼) Kelly | 0.112 | −57.2 | 27.3 | −23.2 | 15.0 | **worse, sig** (t −1.96, p .975) |
| fractional Kelly + hard cap | 0.169 | −3.6 | 22.2 | −1.5 | 0.8 | worse (t −1.53) |

**No variant beats flat on Sharpe; none reduces DD/Ann.** The cap binds on 100% of days, so the
base rule underneath it barely matters (Kelly+cap ≡ vol-target+cap).

---

## What the overlay BUYS, and what it COSTS

**Buys (only this):**
- **A hard floor on catastrophic loss.** The per-session cap bounds the worst session to ~5% of
  annual expected P&L, cutting **max drawdown −71 → −5** and **worst day −19 → −1.4**. This is real
  and valuable for survival — a short-vol book's defining failure is the blow-up, and the cap
  forbids it *adaptively* (it de-levers as estimated session risk rises), which a fixed notional
  does not.

**Costs:**
- **Lower risk-adjusted return.** Dynamic vol-targeting drops Sharpe 0.385 → 0.269 and *worsens*
  DD/Ann 9.1 → 13.4: the lagged vol estimate can't dodge the first gap day, so it cuts the recovery
  more than it cuts the drawdown. Kelly is worse still (0.112).
- **Capped upside.** Under the cap the book runs at ~0.05–0.3× — annual P&L falls from 7.78 to
  0.16–0.35. You give up most of the return for the loss floor. The *same* absolute-risk reduction
  is available by simply trading flat smaller, *without* the Sharpe hit.
- **Turnover: not a cost here.** Sized at entry, the book trades *less* (lower notional: 12 → ~1
  trades/yr), so costs fall. The "sizing costs money" concern bites for daily-rebalanced liquid
  instruments, not an entry-sized options book.

**The honest synthesis:** to take less risk, *shrink the flat book* — you keep Sharpe 0.385 and
DD/Ann 9.1. Use the overlay only for the **cap**, and treat it as a risk *limit* (blow-up
insurance), not a performance enhancer. Do not run vol-targeting or Kelly on this strategy.

---

## Why it behaves this way (mechanism, not just the score)

1. **A constant leverage change is scale-invariant.** The cap forces a near-constant ~0.05–0.07×;
   scaling P&L and drawdown together leaves Sharpe and DD/Ann unchanged (verified: the *stable-A*
   cap reproduces vol-target's 0.269 / 13.4 exactly, only smaller). So the cap moves *absolute*
   risk, never the *ratio*. Only the *time-varying* part of leverage can change risk-adjusted
   numbers — and here it changes them for the worse.
2. **Lagged vol can't dodge the gap.** σ̂ₜ uses data ≤ t−1. Before a vol spike the strategy is calm
   (often flat), so σ̂ is low and leverage high — exactly into the first gap day. Vol-targeting
   only de-levers *after* vol has risen, missing the mean-reversion bounce that is the short-vol
   edge. Net: it trims the recovery more than the drawdown.
3. **Kelly's edge is the fragile input.** The strategy's *active-day* Sharpe (1.20) tempts a full
   Kelly of ~12× leverage. The Monte-Carlo ruin analysis (Phase 2) shows the asymmetry: at full
   Kelly, P(50% drawdown) ≈ 100% even with a *perfectly* estimated edge; at ¼ Kelly it is ~9% at
   the true edge but ~47% if the edge is over-estimated by 50%. Edge — not vol — is what bankrupts
   you, which is why the default is ¼ Kelly and why Kelly is the wrong tool for a strategy whose
   edge is this uncertain.
4. **A frequently-flat strategy needs active-day risk.** The baseline is flat 74% of days; on the
   raw zero-inflated stream trailing vol collapses to ~0 and naive leverage spikes at re-entry
   (Sharpe 0.385 → 0.155). Conditioning the risk estimate on in-position days fixes this (→ 0.269)
   and is the overlay's default.

---

## Recommended default config — and WHY each value (priors, not backtest-fit)

```yaml
sizing:
  risk:
    estimator: "ewma"          # reacts to vol regime shifts faster than a flat window
    ewma_halflife: 21          # ~1 trading month: responsive but not jumpy (RiskMetrics-family prior)
    annualize: true            # vol target is stated in annualised terms
  rules:
    condition_on_active: true  # ESSENTIAL for a frequently-flat strategy (else leverage spikes at re-entry)
    target_vol: 0.10           # 10% — conventional institutional risk budget. LEVEL is immaterial to the
                               #       verdict (Sharpe & DD/Ann are scale-invariant); it only sets gross size.
    vol_floor_frac: 0.25       # cap flat-stretch leverage at 1/0.25 = 4x (a backstop, rarely binds once
                               #   risk is conditioned on active days)
    max_leverage: 4.0          # absolute sanity ceiling
    kelly_fraction: 0.25       # 1/4 Kelly, NEVER full — full Kelly is ~12x here and ~certainly ruinous;
                               #   edge is estimated with large error (see ruin curve). Default fractional.
    cap_enabled: true          # the overlay's one worthwhile rule: a binding per-session loss limit
    cap_fraction: 0.05         # one adverse session may risk <= 5% of annual expected P&L (risk-budget
                               #   convention; a year's edge shouldn't be lost in a day)
    cap_loss_sigma: 3.0        # adverse session = 3 daily sigma (~99% VaR) — conservative for a fat left tail
    annual_expected_pnl: <set> # PREFER a forward planning estimate -> clean level cap (Sharpe/ratio
                               #   preserved). Leave null to estimate point-in-time (expanding mean), which
                               #   works but couples leverage to trailing edge (a small Sharpe cost).
    brake_enabled: false       # optional backstop; dormant at disciplined leverage. Turn on for aggressive sizing.
```

None of these are fit to maximise backtest return. `target_vol` and `kelly_fraction` are stated
priors; the verdict metrics (Sharpe, DD/Ann) do not depend on `target_vol` or `reference_capital`
at all. The one judgement call is `cap_fraction = 5%` and `cap_loss_sigma = 3` — both standard
risk-budgeting conventions, and the conclusion is robust to them (the cap reduces absolute risk at
any setting and never improves the ratio).

**For this strategy specifically:** trade **flat at a notional sized so a 3σ session is tolerable**
(equivalently, run the overlay with `cap_enabled: true`, `method: vol_target`, and *disregard* the
vol-target's dynamics — or simply use the cap as a static position limit). Do **not** enable Kelly.

---

## Reusable as a drop-in (the interface delivered)

`apply_overlay(returns, signal, cfg) -> SizedResult` takes ANY strategy's per-period return stream
(+ optional position signal) and a point-in-time risk estimate, and returns a sized position,
sized P&L, leverage path, turnover, and per-rule diagnostics. It is strategy-agnostic — it knows
nothing about straddles — so it applies unchanged to the elevated-VIX baseline, the USO/crude
z-score trade, or any future harnessed signal.

- **Single-position:** scalar EWMA/rolling vol of the strategy's own stream (`src/sizing/risk.py`).
- **Multi-position:** Ledoit-Wolf shrinkage covariance (`ledoit_wolf_cov`, `rolling_cov`),
  point-in-time; gated so the single-position path stays scalar. RMT/Marchenko-Pastur eigenvalue
  denoising is noted as a stretch (worthwhile when #strategies approaches the lookback).
- **No look-ahead:** every sizing input uses data ≤ t−1 (estimators lag internally; the drawdown
  brake reads realised sized equity ≤ t−1). Enforced by unit tests that corrupt the future and
  assert the present leverage is byte-identical.

---

## What would invalidate this conclusion / what it rests on

**Would change it:**
- A strategy whose **own volatility is persistent and forecastable** (vol clusters that the lagged
  σ̂ catches *before* the loss) — there vol-targeting genuinely improves DD/Ann. Short-vol is the
  hard case (the risk arrives as a gap). The overlay should be re-run on the USO z-score trade,
  where the result may differ.
- A **higher and more stable edge**, where Kelly's edge-estimate error is small relative to the
  edge — then fractional Kelly adds growth. This strategy's edge is too uncertain for that.

**Rests on:**
- Convex short-ATM-straddle P&L, daily close-to-close hedging, VIX as the IV mark, stress-widened
  costs — the Phase-A engine (so intraday gap risk is *understated*; the real tail is worse).
- Leverage sampled at entry/roll (entry-sized options book). Daily-resized sizing would add
  turnover with marginal benefit.
- Sizing inputs strictly point-in-time; vol target / Kelly fraction / cap are config priors.

---

## Reproduce

```bash
python3 run_sizing_phase1.py    # point-in-time risk estimation (+ no-look-ahead demo on real data)
python3 run_sizing_phase2.py    # sizing rules: leverage path, Kelly ruin-vs-edge-error, cap, brake
python3 run_sizing_phase3.py    # THE TEST: flat vs overlay, net of convex P&L + stress costs, DM test
python3 -m pytest -q            # 147 tests
```

New code: `src/sizing/{risk,rules,overlay}.py` (+ `config.py` sizing dataclasses, `config.yaml`
`sizing:` block); convex engine extended for continuous leverage (`src/pnl_convex.py`). Figures:
`outputs/figures/sizing_phase{1,2,3}_*.png`. Tests: `tests/test_sizing_{risk,rules}.py`,
`tests/test_pnl_convex.py` (+ continuous-leverage). All sizing parameters live in `config.yaml`.
