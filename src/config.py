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
class Config:
    random_seed: int
    data: DataConfig
    rv: RVConfig
    har: HARConfig
    walk_forward: WalkForwardConfig
    transaction_costs: TransactionCostsConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        random_seed=raw["random_seed"],
        data=DataConfig(**raw["data"]),
        rv=RVConfig(**raw["rv"]),
        har=HARConfig(**raw["har"]),
        walk_forward=WalkForwardConfig(**raw["walk_forward"]),
        transaction_costs=TransactionCostsConfig(**raw["transaction_costs"]),
    )
