"""Stable contracts shared by minute-v2 data, model, and replay code."""

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

KEY_COLUMNS = ("symbol", "trade_date", "bar_time")

BASE_FEATURE_COLUMNS = (
    "minute_index",
    "minute_fraction",
    "time_sin",
    "time_cos",
    "return_1m",
    "return_3m",
    "return_5m",
    "return_15m",
    "return_30m",
    "return_60m",
    "return_from_previous_close",
    "return_from_open",
    "bar_range",
    "bar_close_location",
    "cumulative_range",
    "cumulative_close_location",
    "vwap_deviation",
    "realized_volatility_5m",
    "realized_volatility_15m",
    "realized_volatility_30m",
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
    "previous_return_5d",
    "previous_return_20d",
    "previous_volatility_20d",
    "previous_log_amount_20d",
    "daily_liquidity_rank",
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
    """Frozen first-version research assumptions."""

    board: str = "main"
    exclude_st: bool = True
    expected_session_bars: int = EXPECTED_SESSION_BARS
    morning_decision_start: str = MORNING_DECISION_START
    morning_decision_end: str = MORNING_DECISION_END
    afternoon_decision_start: str = AFTERNOON_DECISION_START
    afternoon_decision_end: str = AFTERNOON_DECISION_END
    exit_window_start: str = EXIT_WINDOW_START
    exit_window_end: str = EXIT_WINDOW_END
    periodic_sample_every: int = 15
    random_negative_percent: int = 1
    minimum_daily_liquidity_rank: float = 0.30
    maximum_delayed_exit_days: int = 5
    maximum_participation_rate: float = 0.01
    commission_bps: float = 3.0
    transfer_fee_bps: float = 0.1
    stamp_tax_bps_before_20230828: float = 10.0
    stamp_tax_bps_after_20230828: float = 5.0
    slippage_bps: float = 7.0
    maximum_trading_days_per_month: int = 0
    processing_days_per_chunk: int = 1
    duckdb_threads: int = 2
    memory_floor_gib: float = 4.0

    def validate(self) -> None:
        if self.board != "main":
            raise MinuteV2Error(f"unsupported_minute_v2_board:{self.board}")
        if self.expected_session_bars != EXPECTED_SESSION_BARS:
            raise MinuteV2Error(
                f"minute_v2_session_bar_contract_invalid:{self.expected_session_bars}"
            )
        if not 1 <= int(self.periodic_sample_every) <= EXPECTED_SESSION_BARS:
            raise MinuteV2Error("minute_v2_periodic_sample_invalid")
        if not 0 <= int(self.random_negative_percent) <= 100:
            raise MinuteV2Error("minute_v2_random_negative_percent_invalid")
        if not 0.0 <= float(self.minimum_daily_liquidity_rank) < 1.0:
            raise MinuteV2Error("minute_v2_liquidity_rank_invalid")
        if not 0.0 < float(self.maximum_participation_rate) <= 1.0:
            raise MinuteV2Error("minute_v2_participation_rate_invalid")
        if float(self.memory_floor_gib) < 1.0:
            raise MinuteV2Error("minute_v2_memory_floor_too_small")
        if not 1 <= int(self.processing_days_per_chunk) <= 5:
            raise MinuteV2Error("minute_v2_processing_chunk_invalid")
        if not 0 <= int(self.maximum_trading_days_per_month) <= 23:
            raise MinuteV2Error("minute_v2_month_day_sample_invalid")

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
    "CROSS_SECTION_FEATURE_COLUMNS",
    "EXIT_WINDOW_END",
    "EXIT_WINDOW_START",
    "EXPECTED_DECISION_BARS",
    "EXPECTED_SESSION_BARS",
    "KEY_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "MORNING_DECISION_END",
    "MORNING_DECISION_START",
    "MinuteV2Config",
    "MinuteV2Error",
    "is_decision_bar",
]
