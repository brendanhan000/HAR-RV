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


# --- Sizing overlay (reusable layer) ---------------------------------------


@dataclass
class RiskConfig:
    """Point-in-time risk estimator settings (priors, NOT backtest-tuned)."""
    estimator: str                # "ewma" | "rolling"
    window: int                   # rolling-window lookback (periods)
    rolling_min_periods: int
    ewma_halflife: float          # EWMA half-life (periods)
    ewma_lambda: Optional[float]  # RiskMetrics decay; overrides halflife if set
    ewma_min_periods: int
    demean: bool                  # False = zero-mean RMS (RiskMetrics); True = centred std
    ddof: int                     # only used when demean=True
    annualize: bool
    periods_per_year: int
    vol_floor: float              # estimator stays honest; rules clamp (Phase 2)


@dataclass
class CovarianceConfig:
    """Multi-position covariance settings (gated off for single-strategy use)."""
    method: str                   # "ledoit_wolf" | "sample"
    window: int
    min_periods: int
    annualize: bool = True
    periods_per_year: int = 252


@dataclass
class RulesConfig:
    """Sizing-rule settings. Vol target & Kelly fraction are PRIORS, not backtest-tuned."""
    method: str                       # base leverage rule: "vol_target" | "kelly"
    condition_on_active: bool         # estimate the strategy's risk on in-position days only
                                      # (a frequently-flat strategy's raw-stream vol collapses to
                                      # ~0 in flat stretches -> naive leverage spikes at re-entry)
    reference_capital: float          # notional the P&L is measured against (sets gross scale)
    target_vol: float                 # annualised vol target (stream units; 0.10 = 10% for %-returns)
    vol_floor_frac: float             # floor sigma_hat at this * target_vol (flat-stretch guard)
    max_leverage: float
    kelly_fraction: float             # default 1/4; never full
    kelly_edge_window: Optional[int]  # None = expanding-mean edge
    cap_enabled: bool
    cap_fraction: float               # one adverse session <= this fraction of annual expected P&L
    cap_loss_sigma: float             # adverse session loss per unit = this * daily sigma_hat
    annual_expected_pnl: Optional[float]  # None = point-in-time estimate; else absolute override
    brake_enabled: bool
    brake_threshold: float
    brake_floor: float
    brake_max_dd: float
    turnover_cost_per_unit: float


@dataclass
class SizingConfig:
    risk: RiskConfig
    covariance: Optional[CovarianceConfig] = None
    rules: Optional[RulesConfig] = None


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
    sizing: Optional[SizingConfig] = None


def _build_sizing(raw_sizing: Optional[dict]) -> Optional[SizingConfig]:
    if not raw_sizing:
        return None
    risk = RiskConfig(**raw_sizing["risk"])
    cov_raw = raw_sizing.get("covariance")
    cov = CovarianceConfig(**cov_raw) if cov_raw else None
    rules_raw = raw_sizing.get("rules")
    rules = RulesConfig(**rules_raw) if rules_raw else None
    return SizingConfig(risk=risk, covariance=cov, rules=rules)


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
        sizing=_build_sizing(raw.get("sizing")),
    )
