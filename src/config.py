from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    ticker: str
    start_date: str
    end_date: str
    bar_size: int
    data_source: str
    polygon_api_key_env: str
    raw_dir: str
    snapshot_file: str
    daily_snapshot_file: str

    @property
    def polygon_api_key(self) -> Optional[str]:
        return os.environ.get(self.polygon_api_key_env)

    @property
    def snapshot_path(self) -> Path:
        return Path(self.raw_dir) / self.snapshot_file

    @property
    def daily_snapshot_path(self) -> Path:
        return Path(self.raw_dir) / self.daily_snapshot_file


@dataclass
class RVConfig:
    estimator: str
    annualize: bool
    trading_days_per_year: int
    min_bars_per_day: int


@dataclass
class HARConfig:
    horizon: int
    lags: dict
    fit_log_har: bool
    log_bias_correction: bool
    hac_lags: Optional[int]


@dataclass
class WalkForwardConfig:
    min_train_days: int
    refit_every_days: int
    embargo_days: int
    benchmarks: List[str]
    rolling_vol_window: int
    vix_ticker: str
    vix_snapshot_file: str


@dataclass
class TransactionCostsConfig:
    option_bid_ask_vega: float
    slippage_bps: float
    financing_rate_annual: float


@dataclass
class ConvexPnLConfig:
    position_type: str
    hold_days: int
    tenor_buffer_days: int
    iv_mark: str
    ladder_daily: bool
    vega_notional: float
    risk_free_rate: float
    dividend_yield: float
    option_bid_ask_vol: float
    stress_spread_mult: float
    stress_vix_level: float
    underlying_slippage_bps: float
    hedge_cost: bool


@dataclass
class BaselineConfig:
    signal_window: int
    signal_min_periods: int
    short_percentile: float
    long_percentile: float


# --- Signal-validation study (each signal gets its own config block) ---


@dataclass
class SpreadConfig:
    """Signal 1 — USO/BNO beta-adjusted z-score OU mean-reversion. Thresholds are PRIORS
    (standard pairs-trade values), fixed on dev-period reasoning, NOT tuned to the holdout."""
    uso_file: str
    bno_file: str
    holdout_start: str         # strict never-seen OOS begins here; before = dev/in-sample
    beta_window: int           # rolling hedge-ratio (cointegration) window
    z_window: int              # rolling z-score window
    z_entry: float             # enter when |z| > this
    z_exit: float              # exit to flat when |z| < this
    one_way_bps: float         # one-way ETF transaction cost per leg (bps of notional)
    ou_window: int             # window for rolling OU half-life / ADF diagnostics


@dataclass
class SignalsConfig:
    holdout_fdr: float = 0.05  # Benjamini-Hochberg FDR across the four signals
    spread: Optional[SpreadConfig] = None


@dataclass
class Config:
    random_seed: int
    data: DataConfig
    rv: RVConfig
    har: HARConfig
    walk_forward: WalkForwardConfig
    transaction_costs: TransactionCostsConfig
    convex_pnl: Optional[ConvexPnLConfig] = None
    baseline: Optional[BaselineConfig] = None
    signals: Optional[SignalsConfig] = None


def _build_signals(raw_signals: Optional[dict]) -> Optional[SignalsConfig]:
    if not raw_signals:
        return None
    spread_raw = raw_signals.get("spread")
    spread = SpreadConfig(**spread_raw) if spread_raw else None
    return SignalsConfig(holdout_fdr=raw_signals.get("holdout_fdr", 0.05), spread=spread)


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    convex = raw.get("convex_pnl")
    baseline = raw.get("baseline")
    return Config(
        random_seed=raw["random_seed"],
        data=DataConfig(**raw["data"]),
        rv=RVConfig(**raw["rv"]),
        har=HARConfig(**raw["har"]),
        walk_forward=WalkForwardConfig(**raw["walk_forward"]),
        transaction_costs=TransactionCostsConfig(**raw["transaction_costs"]),
        convex_pnl=ConvexPnLConfig(**convex) if convex else None,
        baseline=BaselineConfig(**baseline) if baseline else None,
        signals=_build_signals(raw.get("signals")),
    )
