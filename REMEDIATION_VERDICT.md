# HAR-VRP Remediation Pass — Final Verdict (Phase E)

**Question (the prime directive):** after fixing the four named defects, does the HAR-VRP
signal add **reliable incremental edge over a naive "sell vol when VIX is elevated"
baseline, net of a convex (true short-gamma) cost/P&L model?**

**Answer: No.** The HAR-VRP signal does not beat the elevated-VIX baseline — it is
strictly worse, gross and net, at every threshold and lookback we tested. The apparent
Sharpe-1.0 edge from the original study was an artifact of (i) a P&L proxy that smoothed
away the tail and (ii) an unrealistically light cost assumption. **Recommendation: trade
the simple baseline, or nothing — and note the baseline itself is only marginal.**

> This is a hypothesis-test research project, not a trading system. SPY 2006–2024,
> Yang-Zhang daily RV (no Polygon history on the free tier), VIX as the implied-vol mark.

---

## One table — the whole remediation

All ratios are unitless and sizing-invariant, so they compare across P&L models. "DD/Ann"
= max drawdown ÷ annual P&L (higher = worse tail). Same VRP signal until the Phase-C row.

| # | What changed | Sharpe (net) | DD/Ann | Worst day | Verdict |
|---|---|--:|--:|--:|---|
| Phase 4 | **Linear** vega P&L proxy (original) | **1.02** | 7.7 | 0.86× ann | flattering artifact |
| A | **Convex** short-straddle P&L, **gross** (pure convexity) | 0.56 | 5.7 | 1.41× ann | convexity ≈ halves Sharpe, fattens tail |
| A | **Convex** P&L, **net** of realistic stress-widened costs | **−0.79** | 22.2 | 1.04× ann | costs flip it negative |
| B (bounded) | Intraday TSRV vs Yang-Zhang daily RV (≈8 mo) | — | — | — | YZ already faithful (corr 0.95): RV input was **not** the weak link |
| C | **HAR-VRP vs elevated-VIX baseline**, convex net | −0.28 **vs +0.38** | 35.2 **vs 9.1** | −31 **vs −19** | **GATE FAILED — baseline wins** |
| D | Tail management | — | — | — | **not performed** (gate failed; do not rescue) |

**Phase C detail (the gate), net of convex costs, point-in-time rolling thresholds:**

| Strategy (short/flat) | Net Sharpe | Gross Sharpe | DD/Ann | Worst day |
|---|--:|--:|--:|--:|
| Elevated-VIX baseline (VIX > rolling 75th pct) | **+0.38** | **0.88** | 9.1 | −19.0 |
| HAR-VRP (VRP > rolling 75th pct) | −0.28 | 0.68 | 35.2 | −31.1 |

Paired HAC test, H1 "HAR-VRP beats baseline": **net p = 0.999, gross p = 0.83** (i.e.
HAR-VRP is significantly *worse*, not better). The baseline wins at **every** percentile
{50,60,70,75,80,90} and **every** rolling window {378…1260 d}, gross and net, and in both
the 2008 and 2020 crises.

---

## Why it fails (the mechanism, not just the score)

1. **The linear proxy hid the risk.** Its daily P&L was a forward-22-day *average*
   realized variance — artificially smooth. That inflated Sharpe (1.11 gross) and erased
   the left tail. Pricing the actual short straddle with Black-Scholes along the realized
   path roughly halves the gross Sharpe and reveals worst days that the proxy scored at
   ≈ 0 (e.g. 2011-08-08, VIX 48: convex −20.6 vs linear −0.001).
2. **The only convex edge is short-vega, not the variance premium.** With implied vol
   frozen, the signal's gamma-theta P&L is ≈ 0 (−16); the positive gross is almost
   entirely vega (+178) — a bet that elevated VIX falls. We validated the engine isn't
   broken: an *unconditional* always-short hedged straddle harvests +718 of gamma-theta,
   so the premium is real — the **signal's timing** just doesn't capture it. And vega is
   the leg that detonates in a crisis.
3. **VRP = VIX² − HAR degrades a cleaner signal.** The edge that exists is "short vol
   when implied is rich / mean-reverting," and the **VIX level captures that directly**.
   Subtracting a noisy HAR forecast (the leg VIX already beats, Phase 3) only adds noise.
   That is exactly why VIX-ranking beats VRP-ranking even before costs.
4. **The RV input was not the bottleneck.** On the ~8 months of intraday data the free
   tier allows, SPY shows negligible microstructure noise (1-min RVol only +1.1% vs
   5-min) and **Yang-Zhang daily tracks the noise-robust intraday measure at 95%
   correlation, ~unbiased**. A sharper RV would not have rescued HAR.

---

## Final verdict (one paragraph)

Net of a convex short-gamma P&L and realistic, stress-widened option costs, the HAR-VRP
signal **adds no reliable incremental edge over simply selling vol when VIX is elevated** —
it earns a lower (negative) net Sharpe, a deeper drawdown-to-P&L ratio, and a worse worst
day, and a Diebold-Mariano-style paired test rejects any P&L advantage (it is significantly
worse). The result is robust across thresholds, lookback windows, and the 2008/2020
crises, and it holds **gross as well as net**, so it is a signal-quality problem, not a
cost artifact. The HAR machinery actively hurts: VRP is a noisier version of the VIX-level
bet it is trying to express. **We did not proceed to Phase D (tail management):** the gate
rule forbids rescuing a signal that has not first proven incremental edge. The honest
recommendation is to **trade the elevated-VIX baseline or nothing** — with the caveat that
even the baseline is only marginal (net Sharpe ≈ 0.38, a ~9× drawdown/annual ratio, and a
single-day loss ~9× its annual P&L); it is the *better* of the two, not a good strategy.

---

## What would invalidate this conclusion / what it rests on

**Would overturn it:**
- An **intraday-RV HAR that beats VIX out-of-sample** on QLIKE / Diebold-Mariano over the
  full 2006–2024 sample. We could not test this — Polygon's free tier reaches only ~2
  years of intraday history (confirmed: pre-2024 minute bars return HTTP 403
  "upgrade your plan"). The *bounded* check we could run argues against it (YZ ≈ intraday),
  but it is not the full out-of-sample test. **This is the one open door.**
- A materially different, defensible cost or structure assumption that closes the
  baseline's *gross* advantage (unlikely — the baseline wins pre-cost at every setting).

**Assumptions / data it rests on:**
- SPY 2006–2024; Yang-Zhang **daily** RV (not intraday) as the HAR target.
- VIX (30-cal-day, SPX) used as the SPY straddle IV and tenor mark; ATM only (no skew).
- Convex P&L = one rolled, delta-hedged short ATM straddle; daily close-to-close hedging
  (so intraday gap risk is *understated* — the real tail is worse, not better).
- Costs: 1 vol-pt round-trip option spread, ×3 when VIX>30; 1 bp hedge slippage. All
  config-driven in `config.yaml`.
- No look-ahead: every threshold/percentile uses only data ≤ t; signals lagged one day.
- Bounded Phase B used ~8 months (2025-09 → 2026-05), a calm window; YZ may degrade more
  than this in crises, but the bulk of any sample is normal-vol where YZ looks faithful.

---

## Reproduce

```bash
python3 run_phase4.py       # original linear-proxy backtest (baseline numbers)
python3 run_phaseA.py       # convex short-gamma P&L vs linear proxy
python3 run_phaseC.py       # THE GATE: HAR-VRP vs elevated-VIX baseline
python3 run_phaseB_lite.py  # bounded intraday-RV vs Yang-Zhang (needs POLYGON_API_KEY once)
python3 -m pytest -q        # 76 tests
```

Figures: `outputs/figures/phaseA_convex_vs_linear.png`,
`outputs/figures/phaseB_intraday_rv.png`. New code: `src/pnl_convex.py`,
`src/baseline.py`, `src/rv_intraday.py` (+ tests). All parameters live in `config.yaml`.
