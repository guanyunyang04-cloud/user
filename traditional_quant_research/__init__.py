"""Lightweight primitives for traditional quantitative research."""

from .backtest import BacktestResult, long_only_backtest
from .dataset import load_daily_bars, load_daily_snapshot, load_daily_status, load_manifest, load_universe
from .dataset_v2 import (
    load_daily_universe,
    load_pit_daily_bars,
    load_pit_daily_status,
    load_pit_manifest,
    load_pit_snapshot,
    load_quality_report,
    load_tradeable_panel,
)
from .diagnostics import ic_by_date, quantile_returns, summarize_factor_ic, top_n_backtest
from .factors import annualized_volatility, momentum, moving_average, simple_returns, zscore
from .metrics import annualized_return, max_drawdown, sharpe_ratio, volatility
from .portfolio import equal_weight, normalize_long_only, rank_long_short
from .research_panel import (
    add_baseline_score,
    add_cross_sectional_zscores,
    build_factor_label_panel,
    default_factor_columns,
    load_baseline_factor_panel,
    panel_summary,
)
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
    "add_baseline_score",
    "add_cross_sectional_zscores",
    "annualized_return",
    "annualized_volatility",
    "build_factor_label_panel",
    "default_factor_columns",
    "equal_weight",
    "filter_sh_sz_a_shares",
    "filter_sh_sz_mainboard_a_shares",
    "ic_by_date",
    "is_active_common_stock_info",
    "is_sh_sz_a_share",
    "is_sh_sz_mainboard_a_share",
    "is_st_name",
    "load_daily_bars",
    "load_daily_universe",
    "load_daily_snapshot",
    "load_daily_status",
    "load_baseline_factor_panel",
    "load_manifest",
    "load_pit_daily_bars",
    "load_pit_daily_status",
    "load_pit_manifest",
    "load_pit_snapshot",
    "load_quality_report",
    "load_tradeable_panel",
    "load_universe",
    "long_only_backtest",
    "max_drawdown",
    "momentum",
    "moving_average",
    "normalize_long_only",
    "panel_summary",
    "quantile_returns",
    "rank_long_short",
    "sharpe_ratio",
    "simple_returns",
    "summarize_factor_ic",
    "top_n_backtest",
    "volatility",
    "zscore",
]
