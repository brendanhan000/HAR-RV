"""
Phase B (bounded) — intraday RV vs the Yang-Zhang daily estimator we actually used.

The free Polygon tier only reaches ~2 years of intraday history, far too short to refit
HAR and re-run the walk-forward over 2006-2024. So this does the bounded thing that IS
answerable on the accessible window: build a microstructure-noise-robust intraday RV
(two-scale realized variance + a volatility-signature noise diagnostic) and ask whether
Yang-Zhang DAILY was a materially worse RV *measure*. It CANNOT change the Phase C gate
verdict (no out-of-sample HAR-vs-VIX test is possible on ~8 months).  HYPOTHESIS TEST.

First run fetches 1-min SPY from Polygon (needs POLYGON_API_KEY) and caches it; later
runs read the cache and need no key / no quota.
"""
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import requests

from src.config import load_config
from src.rv_estimator import compute_yang_zhang
from src.data_pull import _fetch_yfinance_daily
from src.rv_intraday import (
    realized_variance_subsampled,
    two_scale_rv,
    volatility_signature,
)

SEP = "=" * 80
cfg = load_config("config.yaml")

START, END = "2025-09-15", "2026-05-30"     # ~8.5 months inside the free-tier window
CACHE = Path("data/raw/spy_1min_recent.parquet")


def fetch_1min_free_tier(ticker, start, end, key) -> pd.DataFrame:
    """Paginated 1-min fetch that respects the free tier's 5 calls/min (13s spacing)."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}
    out, page = [], 0
    while url:
        r = requests.get(url, params=params, timeout=60)
        j = r.json()
        if j.get("status") not in ("OK", "DELAYED") or r.status_code != 200:
            raise RuntimeError(f"Polygon error {r.status_code}: {j.get('message')}")
        out.extend(j.get("results", []))
        url = j.get("next_url")
        params = {"apiKey": key}
        page += 1
        print(f"  page {page}: cumulative bars={len(out)}")
        if url:
            time.sleep(13)   # free tier: 5 req/min
    df = pd.DataFrame(out)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[
        ["timestamp", "open", "high", "low", "close", "volume"]]


print(f"\n{SEP}\nPHASE B (bounded) — intraday RV vs Yang-Zhang daily  [free-tier window]\n{SEP}")

# --- 1) intraday 1-min bars (cache-aware) ---
if CACHE.exists():
    print(f"Loading cached 1-min bars from {CACHE}")
    bars = pd.read_parquet(CACHE)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert("America/New_York")
else:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        sys.exit("No cache and no POLYGON_API_KEY set — cannot fetch. Set the env var once.")
    print(f"Fetching SPY 1-min {START}→{END} from Polygon (free tier, ~5/min)…")
    bars = fetch_1min_free_tier("SPY", START, END, key)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(CACHE, index=False)
    print(f"  saved {len(bars)} bars to {CACHE} (gitignored)")

n_days = bars["timestamp"].dt.normalize().nunique()
print(f"Intraday bars: {len(bars):,}  spanning {n_days} sessions "
      f"({bars['timestamp'].min().date()} → {bars['timestamp'].max().date()})")

# --- 2) volatility signature (microstructure-noise diagnostic) ---
sig = volatility_signature(bars, steps_bars=[1, 2, 3, 5, 10, 15, 30], base_minutes=1)
print(f"\n--- Volatility signature: mean annualized RVol by sampling interval ---")
print(f"  (rising toward 1-min => quote-bounce noise; flat => clean)")
base = sig.loc[5, "mean_RVol_pct"]
for iv, row in sig.iterrows():
    flag = "  <= 1-min" if iv == 1 else ("  <= 5-min (study-style)" if iv == 5 else "")
    print(f"   {iv:>3}-min : RVol {row['mean_RVol_pct']:5.2f}%   "
          f"({row['mean_RVol_pct']/base - 1:+5.1%} vs 5-min){flag}")

# --- 3) intraday RV estimators (RTH, open-to-close) ---
rv_1m = realized_variance_subsampled(bars, step_bars=1)
rv_5m = realized_variance_subsampled(bars, step_bars=5)
tsrv = two_scale_rv(bars, K=5)

# --- 4) daily OHLC -> overnight gap + Yang-Zhang (same window) ---
daily = _fetch_yfinance_daily("SPY", START, END)
daily["date"] = pd.to_datetime(daily["date"])
yz = compute_yang_zhang(daily, cfg)
# add the overnight gap variance so the intraday measure is full-day, like YZ
d = daily.set_index("date")
overnight = np.log(d["open"] / d["close"].shift(1)) ** 2 * cfg.rv.trading_days_per_year
tsrv_fullday = (tsrv + overnight.reindex(tsrv.index)).dropna()

# --- 5) compare the daily RV measures over the window ---
comp = pd.DataFrame({"YZ": yz, "TSRV_fullday": tsrv_fullday,
                     "naive_5min_oc": rv_5m, "naive_1min_oc": rv_1m}).dropna()
print(f"\n--- Daily RV measures over {len(comp)} common sessions (annualized variance) ---")
print(f"  {'measure':<16}{'mean RVol%':>11}{'corr vs TSRV':>14}{'mean ratio vs TSRV':>20}")
ref = comp["TSRV_fullday"]
for col in ["TSRV_fullday", "YZ", "naive_5min_oc", "naive_1min_oc"]:
    rvolpct = np.sqrt(comp[col].mean()) * 100
    corr = comp[col].corr(ref)
    ratio = (comp[col] / ref).replace([np.inf, -np.inf], np.nan).mean()
    print(f"  {col:<16}{rvolpct:>10.2f}%{corr:>14.3f}{ratio:>19.2f}x")

rmse_yz = float(np.sqrt(((comp["YZ"] - comp["TSRV_fullday"]) ** 2).mean()))
print(f"\n  YZ vs full-day TSRV : corr={comp['YZ'].corr(ref):.3f}  RMSE={rmse_yz:.5f}  "
      f"mean(YZ)/mean(TSRV)={comp['YZ'].mean()/ref.mean():.2f}x")

# --- 6) plot: signature + measure time series ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(2, 1, figsize=(13, 9))
    ax[0].plot(sig.index, sig["mean_RVol_pct"], "o-", color="steelblue")
    ax[0].axvline(5, color="grey", ls="--", lw=0.8, label="5-min (study-style)")
    ax[0].set_xlabel("sampling interval (min)"); ax[0].set_ylabel("mean annualized RVol %")
    ax[0].set_title("Volatility signature plot — SPY (rising toward 1-min ⇒ microstructure noise)")
    ax[0].legend(fontsize=8)
    for col, c in [("TSRV_fullday", "black"), ("YZ", "crimson"), ("naive_5min_oc", "darkorange")]:
        ax[1].plot(comp.index, np.sqrt(comp[col]) * 100, lw=0.9, label=col, color=c, alpha=0.85)
    ax[1].set_ylabel("annualized RVol %"); ax[1].legend(fontsize=8)
    ax[1].set_title("Daily realized vol: intraday TSRV (full-day) vs Yang-Zhang vs naive 5-min")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("PHASE B (bounded, ~8 months) — intraday RV measure check  "
                 "[cannot reopen the Phase C gate]", color="darkred", fontsize=10)
    fig.tight_layout()
    out = Path("outputs/figures/phaseB_intraday_rv.png"); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nPlot saved: {out}")
except Exception as e:
    print(f"[plot skipped: {e}]")

print(f"\n{SEP}\nBounded Phase B complete — see whether YZ tracks the intraday measure above.\n{SEP}")
