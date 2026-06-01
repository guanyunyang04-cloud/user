"""Lightweight primitives for traditional quantitative research."""

from .backtest import BacktestResult, long_only_backtest
from .dataset import load_daily_bars, load_daily_snapshot, load_daily_status, load_manifest, load_universe
from .dataset_v2 import load_daily_universe, load_pit_daily_bars, load_pit_daily_status, load_pit_manifest, load_pit_snapshot
from .factors import annualized_volatility, momentum, moving_average, simple_returns, zscore
from .metrics import annualized_return, max_drawdown, sharpe_ratio, volatility
from .portfolio import equal_weight, normalize_long_only, rank_long_short
from .universe import (
    filter_sh_sz_a_shares,
    filter_sh_sz_mainboard_a_shares,
    is_active_common_stock_info,
    is_sh_sz_a_share,
    is_sh_sz_mainboard_a_share,
    is_st_name,
)

__all__ = [
    "BacktestResult",
    "annualized_return",
    "annualized_volatility",
    "equal_weight",
    "filter_sh_sz_a_shares",
    "filter_sh_sz_mainboard_a_shares",
    "is_active_common_stock_info",
    "is_sh_sz_a_share",
    "is_sh_sz_mainboard_a_share",
    "is_st_name",
    "load_daily_bars",
    "load_daily_universe",
    "load_daily_snapshot",
    "load_daily_status",
    "load_manifest",
    "load_pit_daily_bars",
    "load_pit_daily_status",
    "load_pit_manifest",
    "load_pit_snapshot",
    "load_universe",
    "long_only_backtest",
    "max_drawdown",
    "momentum",
    "moving_average",
    "normalize_long_only",
    "rank_long_short",
    "sharpe_ratio",
    "simple_returns",
    "volatility",
    "zscore",
]
