"""Stable contracts shared by minute research data, model, and replay code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MORNING_DECISION_START = "093100000"
MORNING_DECISION_END = "112900000"
AFTERNOON_DECISION_START = "130100000"
AFTERNOON_DECISION_END = "145500000"
EXIT_WINDOW_START = "093500000"
EXIT_WINDOW_END = "100000000"
EXPECTED_SESSION_BARS = 240
EXPECTED_DECISION_BARS = 234
MINIMUM_DAILY_HISTORY = 60
DAILY_HISTORY_LOOKBACK_OPEN_DAYS = 260
MINUTE_HISTORY_LOOKBACK_OPEN_DAYS = 66
MOVING_AVERAGE_PERIODS = (5, 10, 20, 30, 60, 120, 240)
DAILY_WINDOWS = MOVING_AVERAGE_PERIODS
MINUTE_WINDOWS = MOVING_AVERAGE_PERIODS
SIXTY_MINUTE_WINDOWS = MOVING_AVERAGE_PERIODS
SIXTY_MINUTE_BARS_PER_SESSION = 4
MAXIMUM_DAILY_LABEL_HORIZON = 10

KEY_COLUMNS = ("symbol", "trade_date", "bar_time")

CORE_BASE_FEATURE_COLUMNS = (
    "minute_index",
    "trading_minute_ordinal",
    "symbol_minute_sequence",
    "minute_fraction",
    "time_sin",
    "time_cos",
    "crossed_lunch_from_previous_bar",
    "crossed_overnight_from_previous_bar",
    "calendar_days_from_previous_bar",
    "return_1m",
    "return_3m",
    "return_15m",
    "return_from_previous_close",
    "return_from_open",
    "bar_range",
    "bar_close_location",
    "cumulative_range",
    "cumulative_close_location",
    "vwap_deviation",
    "realized_volatility_15m",
    "downside_volatility_15m",
    "volume_acceleration_5_20",
    "amount_curve_surprise",
    "log_bar_amount",
    "price_impact_1m",
    "trend_efficiency",
    "breakout_20m",
    "drawdown_from_day_high",
    "rebound_from_day_low",
    "auction_gap",
    "auction_amount_to_daily20",
    "previous_return_1d",
    "previous_log_amount_20d",
    "daily_liquidity_rank",
    "history_120d_available",
    "history_240d_available",
    "previous_log_total_market_value",
    "previous_log_circulating_market_value",
    "previous_turnover_rate",
    "previous_pe",
    "previous_pb",
    "intraday_float_turnover",
    "intraday_log_circulating_market_value",
    "corporate_action_today",
    "cash_dividend_per_10",
    "bonus_share_per_10",
    "transfer_share_per_10",
    "minute_daily_momentum_interaction_20",
    "minute_daily_momentum_interaction_60",
    "short_long_momentum_spread",
    "last_completed_60m_bucket",
    "m60_history_bar_count",
    "partial_60m_return",
    "partial_60m_range",
    "partial_60m_close_location",
    "partial_60m_log_amount",
    "m1_m60_trend_alignment",
    "m60_daily_trend_alignment",
    "three_timeframe_trend_alignment",
    "m1_m60_sma_spread_20",
    "m60_daily_sma_spread_20",
)

MULTISCALE_MINUTE_FEATURE_COLUMNS = tuple(
    name
    for window in MINUTE_WINDOWS
    for name in (
        f"return_{window}m",
        f"moving_average_deviation_{window}m",
        f"rolling_range_{window}m",
        f"realized_volatility_{window}m",
        f"volume_ratio_{window}m",
    )
)

SIXTY_MINUTE_FEATURE_COLUMNS = tuple(
    name
    for window in SIXTY_MINUTE_WINDOWS
    for name in (
        f"m60_return_{window}bar",
        f"m60_close_to_sma_{window}bar",
        f"m60_range_{window}bar",
        f"m60_volatility_{window}bar",
        f"m60_amount_ratio_{window}bar",
    )
)

DAILY_MULTISCALE_FEATURE_COLUMNS = tuple(
    name
    for window in DAILY_WINDOWS
    for name in (
        f"previous_return_{window}d",
        f"previous_close_to_sma_{window}d",
        f"previous_volatility_{window}d",
        f"previous_amount_ratio_{window}d",
    )
)

BASE_FEATURE_COLUMNS = (
    CORE_BASE_FEATURE_COLUMNS
    + MULTISCALE_MINUTE_FEATURE_COLUMNS
    + SIXTY_MINUTE_FEATURE_COLUMNS
    + DAILY_MULTISCALE_FEATURE_COLUMNS
)

CROSS_SECTION_FEATURE_COLUMNS = (
    "market_return_mean",
    "market_return_5m_mean",
    "market_breadth_positive",
    "market_return_dispersion",
    "market_log_total_amount",
    "industry_return_mean",
    "industry_return_5m_mean",
    "industry_breadth_positive",
    "industry_return_dispersion",
    "industry_amount_share",
    "industry_strength_rank",
    "stock_return_rank",
    "stock_return_5m_rank",
    "stock_volume_acceleration_rank",
    "stock_amount_rank",
    "industry_stock_return_rank",
    "industry_stock_amount_rank",
    "market_residual_return",
    "industry_residual_return",
    "industry_residual_return_5m",
)

MODEL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + CROSS_SECTION_FEATURE_COLUMNS


class MinuteV2Error(RuntimeError):
    """Raised when a minute-v2 contract or causal invariant is violated."""


@dataclass(frozen=True)
class MinuteV2Config:
    """Causal A-share minute-research assumptions."""

    board: str = "main"
    exclude_st: bool = True
    expected_session_bars: int = EXPECTED_SESSION_BARS
    morning_decision_start: str = MORNING_DECISION_START
    morning_decision_end: str = MORNING_DECISION_END
    afternoon_decision_start: str = AFTERNOON_DECISION_START
    afternoon_decision_end: str = AFTERNOON_DECISION_END
    exit_window_start: str = EXIT_WINDOW_START
    exit_window_end: str = EXIT_WINDOW_END
    candidate_background_percent: int = 1
    minimum_daily_liquidity_rank: float = 0.30
    candidate_stock_rank_floor: float = 0.80
    candidate_industry_rank_floor: float = 0.70
    candidate_industry_stock_rank_floor: float = 0.55
    candidate_volume_rank_floor: float = 0.80
    maximum_delayed_exit_days: int = 5
    maximum_participation_rate: float = 0.01
    commission_bps: float = 3.0
    transfer_fee_bps: float = 0.1
    stamp_tax_bps_before_20230828: float = 10.0
    stamp_tax_bps_after_20230828: float = 5.0
    slippage_bps: float = 7.0
    maximum_trading_days_per_month: int = 0
    processing_days_per_chunk: int = 1
    duckdb_threads: int = 4
    memory_floor_gib: float = 0.5
    duckdb_memory_limit_gib: float = 4.0
    minimum_daily_history: int = MINIMUM_DAILY_HISTORY
    daily_history_lookback_open_days: int = DAILY_HISTORY_LOOKBACK_OPEN_DAYS
    minute_history_lookback_open_days: int = MINUTE_HISTORY_LOOKBACK_OPEN_DAYS

    def validate(self) -> None:
        if self.board != "main":
            raise MinuteV2Error(f"unsupported_minute_v2_board:{self.board}")
        if self.expected_session_bars != EXPECTED_SESSION_BARS:
            raise MinuteV2Error(
                f"minute_v2_session_bar_contract_invalid:{self.expected_session_bars}"
            )
        if not 0 <= int(self.candidate_background_percent) <= 100:
            raise MinuteV2Error("minute_v2_candidate_background_percent_invalid")
        if not 0.0 <= float(self.minimum_daily_liquidity_rank) < 1.0:
            raise MinuteV2Error("minute_v2_liquidity_rank_invalid")
        for name, value in (
            ("stock", self.candidate_stock_rank_floor),
            ("industry", self.candidate_industry_rank_floor),
            ("industry_stock", self.candidate_industry_stock_rank_floor),
            ("volume", self.candidate_volume_rank_floor),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise MinuteV2Error(f"minute_v2_candidate_{name}_rank_invalid")
        if not 0.0 < float(self.maximum_participation_rate) <= 1.0:
            raise MinuteV2Error("minute_v2_participation_rate_invalid")
        if float(self.memory_floor_gib) < 0.5:
            raise MinuteV2Error("minute_v2_memory_floor_too_small")
        if not 0.25 <= float(self.duckdb_memory_limit_gib) <= 4.0:
            raise MinuteV2Error("minute_v2_duckdb_memory_limit_invalid")
        if not 1 <= int(self.processing_days_per_chunk) <= 23:
            raise MinuteV2Error("minute_v2_processing_chunk_invalid")
        if not 0 <= int(self.maximum_trading_days_per_month) <= 23:
            raise MinuteV2Error("minute_v2_month_day_sample_invalid")
        if int(self.minimum_daily_history) != MINIMUM_DAILY_HISTORY:
            raise MinuteV2Error("minute_v2_minimum_daily_history_contract_invalid")
        if int(self.daily_history_lookback_open_days) < max(DAILY_WINDOWS) + 2:
            raise MinuteV2Error("minute_v2_daily_history_lookback_too_short")
        if int(self.minute_history_lookback_open_days) < 62:
            raise MinuteV2Error("minute_v2_minute_history_lookback_too_short")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def is_decision_bar(value: str) -> bool:
    bar_time = str(value)
    return (
        MORNING_DECISION_START <= bar_time <= MORNING_DECISION_END
        or AFTERNOON_DECISION_START <= bar_time <= AFTERNOON_DECISION_END
    )


__all__ = [
    "AFTERNOON_DECISION_END",
    "AFTERNOON_DECISION_START",
    "BASE_FEATURE_COLUMNS",
    "CORE_BASE_FEATURE_COLUMNS",
    "CROSS_SECTION_FEATURE_COLUMNS",
    "DAILY_HISTORY_LOOKBACK_OPEN_DAYS",
    "DAILY_MULTISCALE_FEATURE_COLUMNS",
    "DAILY_WINDOWS",
    "EXIT_WINDOW_END",
    "EXIT_WINDOW_START",
    "EXPECTED_DECISION_BARS",
    "EXPECTED_SESSION_BARS",
    "KEY_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "MAXIMUM_DAILY_LABEL_HORIZON",
    "MINIMUM_DAILY_HISTORY",
    "MINUTE_HISTORY_LOOKBACK_OPEN_DAYS",
    "MINUTE_WINDOWS",
    "MOVING_AVERAGE_PERIODS",
    "MULTISCALE_MINUTE_FEATURE_COLUMNS",
    "MORNING_DECISION_END",
    "MORNING_DECISION_START",
    "MinuteV2Config",
    "MinuteV2Error",
    "SIXTY_MINUTE_BARS_PER_SESSION",
    "SIXTY_MINUTE_FEATURE_COLUMNS",
    "SIXTY_MINUTE_WINDOWS",
    "is_decision_bar",
]
