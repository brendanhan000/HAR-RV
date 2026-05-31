"""
Data ingestion: Polygon.io intraday bars with Yang-Zhang daily fallback.

Rules:
- If a local snapshot exists, load it (reproducibility).
- Otherwise fetch from Polygon and save the snapshot.
- Never overwrite an existing snapshot — append only if extending range.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import Config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------

_POLYGON_BASE = "https://api.polygon.io"


def _polygon_get(url: str, params: dict, api_key: str) -> dict:
    params = {**params, "apiKey": api_key}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch_polygon_aggs(
    ticker: str,
    start: str,
    end: str,
    multiplier: int,
    timespan: str,
    api_key: str,
) -> pd.DataFrame:
    """Fetch aggregate bars from Polygon, handling pagination."""
    url = f"{_POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}

    results = []
    while url:
        data = _polygon_get(url, params, api_key)
        results.extend(data.get("results", []))
        url = data.get("next_url")
        params = {}  # next_url already has params baked in
        if url:
            time.sleep(0.12)  # be polite to rate limiter

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Polygon returns millisecond epoch in 't'
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(
        "America/New_York"
    )
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()


def _fetch_polygon_daily(
    ticker: str, start: str, end: str, api_key: str
) -> pd.DataFrame:
    return _fetch_polygon_aggs(ticker, start, end, 1, "day", api_key)


def _fetch_polygon_intraday(
    ticker: str, start: str, end: str, bar_minutes: int, api_key: str
) -> pd.DataFrame:
    return _fetch_polygon_aggs(ticker, start, end, bar_minutes, "minute", api_key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_or_fetch_intraday(cfg: Config) -> pd.DataFrame:
    """
    Return intraday bar DataFrame. Uses local snapshot if present, else fetches
    from Polygon and saves. Columns: timestamp (tz-aware ET), open, high, low,
    close, volume.
    """
    snap = cfg.data.snapshot_path
    snap.parent.mkdir(parents=True, exist_ok=True)

    if snap.exists():
        log.info("Loading intraday snapshot from %s", snap)
        df = pd.read_parquet(snap)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
            "America/New_York"
        )
        return df

    api_key = cfg.data.polygon_api_key
    if not api_key:
        raise EnvironmentError(
            f"Set env var {cfg.data.polygon_api_key_env} to use Polygon."
        )

    log.info("Fetching %d-min bars for %s from Polygon …", cfg.data.bar_size, cfg.data.ticker)
    df = _fetch_polygon_intraday(
        cfg.data.ticker,
        cfg.data.start_date,
        cfg.data.end_date,
        cfg.data.bar_size,
        api_key,
    )

    if df.empty:
        raise ValueError("Polygon returned no intraday data.")

    df.to_parquet(snap, index=False)
    log.info("Saved %d rows to %s", len(df), snap)
    return df


def _fetch_yfinance_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLCV via yfinance. Returns date, open, high, low, close, volume."""
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for {ticker}.")
    raw = raw.reset_index()
    raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
    raw = raw.rename(columns={"index": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    return raw[["date", "open", "high", "low", "close", "volume"]].copy()


def load_or_fetch_daily(cfg: Config) -> pd.DataFrame:
    """
    Return daily OHLCV DataFrame. Used for Yang-Zhang fallback.
    Columns: date (pd.Timestamp, tz-naive), open, high, low, close, volume.
    Tries Polygon first; falls back to yfinance if no API key.
    """
    snap = cfg.data.daily_snapshot_path
    snap.parent.mkdir(parents=True, exist_ok=True)

    if snap.exists():
        log.info("Loading daily snapshot from %s", snap)
        df = pd.read_parquet(snap)
        df["date"] = pd.to_datetime(df["date"])
        return df

    api_key = cfg.data.polygon_api_key
    if api_key:
        log.info("Fetching daily bars for %s from Polygon …", cfg.data.ticker)
        df = _fetch_polygon_daily(
            cfg.data.ticker, cfg.data.start_date, cfg.data.end_date, api_key
        )
        if df.empty:
            raise ValueError("Polygon returned no daily data.")
        df = df.rename(columns={"timestamp": "date"})
        df["date"] = df["date"].dt.normalize().dt.tz_localize(None)
    else:
        log.info(
            "No Polygon key — fetching daily bars for %s from yfinance …",
            cfg.data.ticker,
        )
        df = _fetch_yfinance_daily(cfg.data.ticker, cfg.data.start_date, cfg.data.end_date)

    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.to_parquet(snap, index=False)
    log.info("Saved %d rows to %s", len(df), snap)
    return df


def load_or_fetch_vix_daily(cfg: Config) -> pd.DataFrame:
    """
    Return daily VIX close series. Columns: date, vix_close.
    Tries Polygon; if missing, raises clearly.
    """
    snap = Path(cfg.data.raw_dir) / cfg.walk_forward.vix_snapshot_file
    snap.parent.mkdir(parents=True, exist_ok=True)

    if snap.exists():
        log.info("Loading VIX snapshot from %s", snap)
        df = pd.read_parquet(snap)
        df["date"] = pd.to_datetime(df["date"])
        return df

    api_key = cfg.data.polygon_api_key
    if api_key:
        log.info("Fetching daily VIX from Polygon …")
        df = _fetch_polygon_daily("I:VIX", cfg.data.start_date, cfg.data.end_date, api_key)
        if df.empty:
            raise ValueError("Polygon returned no VIX data.")
        df = df.rename(columns={"timestamp": "date", "close": "vix_close"})
        df["date"] = df["date"].dt.normalize().dt.tz_localize(None)
    else:
        log.info("No Polygon key — fetching VIX from yfinance (^VIX) …")
        raw = _fetch_yfinance_daily("^VIX", cfg.data.start_date, cfg.data.end_date)
        df = raw[["date", "close"]].rename(columns={"close": "vix_close"})

    df = df[["date", "vix_close"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.to_parquet(snap, index=False)
    log.info("Saved %d VIX rows to %s", len(df), snap)
    return df


# ---------------------------------------------------------------------------
# Intraday cleaning
# ---------------------------------------------------------------------------

# Regular trading session for US equities (ET)
_RTH_OPEN = pd.Timedelta(hours=9, minutes=30)
_RTH_CLOSE = pd.Timedelta(hours=16, minutes=0)


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular trading hours (9:30–16:00 ET, inclusive of open bar)."""
    ts = df["timestamp"]
    time_of_day = ts.dt.hour * 60 + ts.dt.minute
    open_min = 9 * 60 + 30
    close_min = 16 * 60  # last bar starts at 15:55 for 5-min bars
    mask = (time_of_day >= open_min) & (time_of_day < close_min)
    return df[mask].copy()


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute within-day log returns on close prices.
    Returns are set to NaN at the first bar of each day (no cross-day return).
    A flag column 'cross_day_gap' marks rows where a cross-day return WOULD
    have been computed — to make the invariant explicit and testable.
    """
    df = df.sort_values("timestamp").copy()
    df["date"] = df["timestamp"].dt.normalize().dt.tz_localize(None)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # Flag and nullify cross-day (overnight) returns
    day_changed = df["date"] != df["date"].shift(1)
    df["cross_day_gap"] = day_changed
    df.loc[day_changed, "log_ret"] = np.nan
    return df
