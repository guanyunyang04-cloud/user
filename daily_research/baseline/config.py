from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ResearchConfig:
    # Date range (YYYYMMDD). end_date empty means latest from data source.
    start_date: str = "20180101"
    end_date: str = ""

    # Universe / benchmark / execution
    universe: List[str] = field(default_factory=list)
    universe_scope: str = "all_a"
    benchmark: str = "000300.SH"
    execution_mode: str = "close"
    rebalance_freq: str = "1d"
    industry_neutral: bool = False
    enable_market_regime_filter: bool = False
    regime_ma_window: int = 60
    regime_vol_window: int = 20
    regime_max_annual_vol: float = 0.28
    regime_allowed_quadrants: List[str] = field(default_factory=lambda: [
        "trend_up_low_vol",
        "trend_up_high_vol",
    ])
    enable_industry_cap: bool = False
    max_industry_weight: float = 0.40
    industry_candidate_buffer: int = 5
    enable_style_cap: bool = False
    max_style_weight: float = 0.60

    # Portfolio style
    holding_count: int = 5
    top_n: Optional[int] = None  # legacy alias
    weighting_method: str = "equal"
    max_weight: float = 0.25
    score_threshold: float = 0.0
    score_clip: float = 3.0
    turnover_limit: float = 2.00
    min_hold_days: int = 1

    # Trading filters
    min_price: float = 2.0
    max_price: float = 300.0
    min_adv20: float = 50_000.0

    # Risk controls (per-position)
    stop_loss: float = -0.10
    take_profit: float = 0.25

    # Factor group weights
    factor_group_weights: Dict[str, float] = field(default_factory=lambda: {
        "trend": 0.00,
        "volume": 0.40,
        "volatility": 0.40,
        "structure": 0.20,
        "signal": 0.00,
    })

    # Factor weights inside groups
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        "mom_5": 0.00,
        "mom_20": 0.00,
        "mom_60": 0.00,
        "ma_gap_10": 0.00,
        "ma_gap_20_60": 0.00,
        "trend_slope_20": 0.00,
        "breakout_20": 0.00,
        "vol_ratio_5_20": 0.00,
        "breakout_volume": 0.00,
        "volume_contraction": 0.50,
        "price_volume_divergence": 0.50,
        "atr_14_pct": 0.25,
        "volatility_20": 0.45,
        "volatility_contraction": 0.30,
        "range_position_20": 0.00,
        "drawdown_20": 0.00,
        "close_strength": 1.00,
        "body_strength": 0.00,
        "up_day_ratio_10": 0.00,
        "trend_streak": 0.00,
        "kama_gap": 0.00,
        "kama_slope": 0.00,
        "long_regime_flag": 0.00,
        "mbuy_flag": 0.00,
        "zjtp_flag": 0.00,
        "hcw_flag": 0.00,
    })

    # Group membership
    factor_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        "trend": [
            "mom_5",
            "mom_20",
            "mom_60",
            "ma_gap_10",
            "ma_gap_20_60",
            "trend_slope_20",
            "breakout_20",
        ],
        "volume": [
            "vol_ratio_5_20",
            "breakout_volume",
            "volume_contraction",
            "price_volume_divergence",
        ],
        "volatility": [
            "atr_14_pct",
            "volatility_20",
            "volatility_contraction",
        ],
        "structure": [
            "range_position_20",
            "drawdown_20",
            "close_strength",
            "body_strength",
            "up_day_ratio_10",
            "trend_streak",
        ],
        "signal": [
            "kama_gap",
            "kama_slope",
            "long_regime_flag",
            "mbuy_flag",
            "zjtp_flag",
            "hcw_flag",
        ],
    })

    def __post_init__(self):
        if self.top_n is not None:
            self.holding_count = int(self.top_n)
        self.holding_count = max(int(self.holding_count), 1)
        self.execution_mode = str(self.execution_mode).lower()
        if self.execution_mode not in {"close", "next_open"}:
            raise ValueError("execution_mode must be 'close' or 'next_open'.")
        self.weighting_method = str(self.weighting_method).lower()
        if self.weighting_method not in {"equal", "score"}:
            raise ValueError("weighting_method must be 'equal' or 'score'.")
        self.universe_scope = str(self.universe_scope).lower()
        self.regime_ma_window = max(int(self.regime_ma_window), 2)
        self.regime_vol_window = max(int(self.regime_vol_window), 2)
        self.regime_allowed_quadrants = [
            str(name).strip().lower()
            for name in self.regime_allowed_quadrants
            if str(name).strip()
        ]
