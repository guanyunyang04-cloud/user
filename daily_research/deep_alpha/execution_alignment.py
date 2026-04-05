from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import pandas as pd
import numpy as np

from daily_research.baseline.backtest import backtest
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.external_target_weight_bridge import apply_rebalance_schedule, build_target_weight_bridge
from daily_research.baseline.regime import apply_market_regime_filter, compute_market_regime_state


EXECUTION_ALIGNMENT_REGIME_MA_WINDOW = 50
EXECUTION_ALIGNMENT_REGIME_VOL_WINDOW = 20
EXECUTION_ALIGNMENT_REGIME_MAX_ANNUAL_VOL = 0.32
EXECUTION_ALIGNMENT_REGIME_TREND_FLAT_BAND = 0.01
EXECUTION_ALIGNMENT_REGIME_VOL_TRANSITION_BAND = 0.10
EXECUTION_ALIGNMENT_ALLOWED_QUADRANTS = (
    "trend_up_low_vol",
    "trend_up_high_vol",
)
ROBUSTNESS_WINDOW_DAYS = (63, 126)


@dataclass(frozen=True)
class ExecutionAlignmentProfile:
    name: str
    description: str
    rebalance_freq: str = "1d"
    rebalance_offset_mode: str = "single"
    rebalance_anchor_date: str = ""
    target_weight_top_k: int = 0
    target_weight_min_weight: float = 0.0
    target_weight_power: float = 1.0
    target_weight_full_invest: bool = False
    use_market_regime_filter: bool = False


@dataclass
class ExecutionAlignmentArtifact:
    mode: str
    objective_metric: str
    selected_profile: str
    selected_profile_description: str
    selected_profile_spec: dict[str, object]
    candidate_profiles: list[str]
    objective_rows: list[dict]
    selected_bridge_meta: dict[str, object]
    selected_train_metrics: dict[str, float]
    valid_metrics: dict[str, float]
    valid_score_frame: pd.DataFrame
    valid_target_weights: pd.DataFrame


_BASE_PROFILE_REGISTRY: dict[str, ExecutionAlignmentProfile] = {
    "raw_1d": ExecutionAlignmentProfile(
        name="raw_1d",
        description="Keep raw 1d target-weight output with no additional execution alignment.",
    ),
    "topk1_1d_regoff": ExecutionAlignmentProfile(
        name="topk1_1d_regoff",
        description="1d direct target-weight bridge cropped to top-k 1 with market regime filter off.",
        target_weight_top_k=1,
        use_market_regime_filter=False,
    ),
    "topk2_1d_regoff": ExecutionAlignmentProfile(
        name="topk2_1d_regoff",
        description="1d direct target-weight bridge cropped to top-k 2 with market regime filter off.",
        target_weight_top_k=2,
        use_market_regime_filter=False,
    ),
    "topk3_1d_regoff": ExecutionAlignmentProfile(
        name="topk3_1d_regoff",
        description="1d direct target-weight bridge cropped to top-k 3 with market regime filter off.",
        target_weight_top_k=3,
        use_market_regime_filter=False,
    ),
    "regoff_k2_10d_ensemble_native_anchor": ExecutionAlignmentProfile(
        name="regoff_k2_10d_ensemble_native_anchor",
        description="Current default execution candidate: 10d anchored all-offset ensemble, top-k 2, regime filter off.",
        rebalance_freq="10d",
        rebalance_offset_mode="all",
        rebalance_anchor_date="2025-01-02",
        target_weight_top_k=2,
        use_market_regime_filter=False,
    ),
    "regon_k1_10d_ensemble_native_anchor": ExecutionAlignmentProfile(
        name="regon_k1_10d_ensemble_native_anchor",
        description="More aggressive execution comparator: 10d anchored all-offset ensemble, top-k 1, regime filter on.",
        rebalance_freq="10d",
        rebalance_offset_mode="all",
        rebalance_anchor_date="2025-01-02",
        target_weight_top_k=1,
        use_market_regime_filter=True,
    ),
}


def _build_curated_profit_profile(
    *,
    name: str,
    description: str,
    rebalance_freq: str,
    rebalance_offset_mode: str,
    target_weight_top_k: int,
    use_market_regime_filter: bool,
) -> ExecutionAlignmentProfile:
    return ExecutionAlignmentProfile(
        name=name,
        description=description,
        rebalance_freq=rebalance_freq,
        rebalance_offset_mode=rebalance_offset_mode,
        rebalance_anchor_date="" if str(rebalance_freq) == "1d" else "2025-01-02",
        target_weight_top_k=target_weight_top_k,
        use_market_regime_filter=use_market_regime_filter,
    )


_CURATED_PROFIT_REGISTRY: dict[str, ExecutionAlignmentProfile] = {
    "regoff_k1_3d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regoff_k1_3d_ensemble_native_anchor",
        description="3d anchored all-offset ensemble, top-k 1, regime filter off.",
        rebalance_freq="3d",
        rebalance_offset_mode="all",
        target_weight_top_k=1,
        use_market_regime_filter=False,
    ),
    "regoff_k2_3d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regoff_k2_3d_ensemble_native_anchor",
        description="3d anchored all-offset ensemble, top-k 2, regime filter off.",
        rebalance_freq="3d",
        rebalance_offset_mode="all",
        target_weight_top_k=2,
        use_market_regime_filter=False,
    ),
    "regoff_k1_5d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regoff_k1_5d_ensemble_native_anchor",
        description="5d anchored all-offset ensemble, top-k 1, regime filter off.",
        rebalance_freq="5d",
        rebalance_offset_mode="all",
        target_weight_top_k=1,
        use_market_regime_filter=False,
    ),
    "regoff_k2_5d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regoff_k2_5d_ensemble_native_anchor",
        description="5d anchored all-offset ensemble, top-k 2, regime filter off.",
        rebalance_freq="5d",
        rebalance_offset_mode="all",
        target_weight_top_k=2,
        use_market_regime_filter=False,
    ),
    "regoff_k3_5d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regoff_k3_5d_ensemble_native_anchor",
        description="5d anchored all-offset ensemble, top-k 3, regime filter off.",
        rebalance_freq="5d",
        rebalance_offset_mode="all",
        target_weight_top_k=3,
        use_market_regime_filter=False,
    ),
    "regoff_k1_10d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regoff_k1_10d_ensemble_native_anchor",
        description="10d anchored all-offset ensemble, top-k 1, regime filter off.",
        rebalance_freq="10d",
        rebalance_offset_mode="all",
        target_weight_top_k=1,
        use_market_regime_filter=False,
    ),
    "regoff_k3_10d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regoff_k3_10d_ensemble_native_anchor",
        description="10d anchored all-offset ensemble, top-k 3, regime filter off.",
        rebalance_freq="10d",
        rebalance_offset_mode="all",
        target_weight_top_k=3,
        use_market_regime_filter=False,
    ),
    "regon_k2_10d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regon_k2_10d_ensemble_native_anchor",
        description="10d anchored all-offset ensemble, top-k 2, regime filter on.",
        rebalance_freq="10d",
        rebalance_offset_mode="all",
        target_weight_top_k=2,
        use_market_regime_filter=True,
    ),
    "regoff_k1_20d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regoff_k1_20d_ensemble_native_anchor",
        description="20d anchored all-offset ensemble, top-k 1, regime filter off.",
        rebalance_freq="20d",
        rebalance_offset_mode="all",
        target_weight_top_k=1,
        use_market_regime_filter=False,
    ),
    "regoff_k2_20d_ensemble_native_anchor": _build_curated_profit_profile(
        name="regoff_k2_20d_ensemble_native_anchor",
        description="20d anchored all-offset ensemble, top-k 2, regime filter off.",
        rebalance_freq="20d",
        rebalance_offset_mode="all",
        target_weight_top_k=2,
        use_market_regime_filter=False,
    ),
}

PROFILE_REGISTRY: dict[str, ExecutionAlignmentProfile] = {
    **_BASE_PROFILE_REGISTRY,
    **_CURATED_PROFIT_REGISTRY,
}

LEGACY_CORE_PROFILE_SET_NAME = "legacy_core_v1"
PROFIT_MAX_PROFILE_SET_NAME = "profit_max_v1"
PROFILE_SET_REGISTRY: dict[str, tuple[str, ...]] = {
    LEGACY_CORE_PROFILE_SET_NAME: (
        "raw_1d",
        "topk2_1d_regoff",
        "regoff_k2_10d_ensemble_native_anchor",
        "regon_k1_10d_ensemble_native_anchor",
    ),
    PROFIT_MAX_PROFILE_SET_NAME: (
        "raw_1d",
        "topk1_1d_regoff",
        "topk2_1d_regoff",
        "topk3_1d_regoff",
        "regoff_k1_3d_ensemble_native_anchor",
        "regoff_k2_3d_ensemble_native_anchor",
        "regoff_k1_5d_ensemble_native_anchor",
        "regoff_k2_5d_ensemble_native_anchor",
        "regoff_k3_5d_ensemble_native_anchor",
        "regoff_k1_10d_ensemble_native_anchor",
        "regoff_k2_10d_ensemble_native_anchor",
        "regoff_k3_10d_ensemble_native_anchor",
        "regon_k1_10d_ensemble_native_anchor",
        "regon_k2_10d_ensemble_native_anchor",
        "regoff_k1_20d_ensemble_native_anchor",
        "regoff_k2_20d_ensemble_native_anchor",
    ),
}
DEFAULT_AUTO_PROFILE_SET_NAME = PROFIT_MAX_PROFILE_SET_NAME
DEFAULT_AUTO_PROFILE_NAMES = list(PROFILE_SET_REGISTRY[DEFAULT_AUTO_PROFILE_SET_NAME])


def default_auto_profile_argument() -> str:
    return DEFAULT_AUTO_PROFILE_SET_NAME


def _normalize_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw in {None, ""}:
        return default
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(raw)


def profile_to_spec(profile: ExecutionAlignmentProfile) -> dict[str, object]:
    return dict(asdict(profile))


def build_profile_from_spec(
    spec: dict[str, Any] | None,
    *,
    fallback_name: str = "custom_policy",
    fallback_description: str = "",
) -> ExecutionAlignmentProfile:
    payload = spec if isinstance(spec, dict) else {}
    return ExecutionAlignmentProfile(
        name=str(payload.get("name", "") or fallback_name).strip() or fallback_name,
        description=str(payload.get("description", "") or fallback_description).strip()
        or "Custom execution policy resolved from the promoted research artifact.",
        rebalance_freq=str(payload.get("rebalance_freq", "1d") or "1d"),
        rebalance_offset_mode=str(payload.get("rebalance_offset_mode", "single") or "single"),
        rebalance_anchor_date=str(payload.get("rebalance_anchor_date", "") or ""),
        target_weight_top_k=int(payload.get("target_weight_top_k", 0) or 0),
        target_weight_min_weight=float(payload.get("target_weight_min_weight", 0.0) or 0.0),
        target_weight_power=float(payload.get("target_weight_power", 1.0) or 1.0),
        target_weight_full_invest=_normalize_bool(payload.get("target_weight_full_invest", False), False),
        use_market_regime_filter=_normalize_bool(payload.get("use_market_regime_filter", False), False),
    )


def resolve_profile(*, name: str = "", spec: dict[str, Any] | None = None) -> ExecutionAlignmentProfile:
    if isinstance(spec, dict) and spec:
        return build_profile_from_spec(
            spec,
            fallback_name=str(name or "custom_policy"),
            fallback_description="Execution policy restored from serialized strategy metadata.",
        )
    return get_profile(name)


def resolve_profile_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise KeyError("Empty execution alignment profile name.")
    if normalized not in PROFILE_REGISTRY:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        raise KeyError(f"Unknown execution alignment profile: {name}. Available: {available}")
    return normalized


def get_profile(name: str) -> ExecutionAlignmentProfile:
    return PROFILE_REGISTRY[resolve_profile_name(name)]


def parse_profile_name_list(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_AUTO_PROFILE_NAMES)
    out: list[str] = []
    seen: set[str] = set()
    for item in str(raw).split(","):
        name = str(item).strip()
        if not name:
            continue
        if name in PROFILE_SET_REGISTRY:
            expanded = PROFILE_SET_REGISTRY[name]
        else:
            expanded = (resolve_profile_name(name),)
        for resolved in expanded:
            if resolved in seen:
                continue
            out.append(resolved)
            seen.add(resolved)
    if not out:
        raise ValueError("Execution alignment profile list resolved to empty.")
    return out


def list_profile_lines() -> list[str]:
    lines = ["profiles:"]
    for key in sorted(PROFILE_REGISTRY):
        profile = PROFILE_REGISTRY[key]
        lines.append(f"- {profile.name}: {profile.description}")
    lines.append("profile_sets:")
    for key in sorted(PROFILE_SET_REGISTRY):
        lines.append(f"- {key}: {', '.join(PROFILE_SET_REGISTRY[key])}")
    lines.append(f"default_auto_profile_set: {DEFAULT_AUTO_PROFILE_SET_NAME}")
    lines.append(f"default_auto_profiles: {', '.join(DEFAULT_AUTO_PROFILE_NAMES)}")
    return lines


def _resolve_window_anchor_date(index: pd.Index, rebalance_anchor_date: str) -> str:
    raw = str(rebalance_anchor_date or "").strip()
    if not raw or len(index) == 0:
        return raw
    dt_index = pd.DatetimeIndex(index)
    anchor_ts = pd.Timestamp(raw)
    if anchor_ts > dt_index.max():
        return str(pd.Timestamp(dt_index[0]).date())
    return raw


def _build_exec_cfg(
    *,
    benchmark: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    holding_count: int,
    max_weight: float,
    min_adv20: float,
    min_price: float,
    max_price: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    profile: ExecutionAlignmentProfile,
) -> ResearchConfig:
    return ResearchConfig(
        start_date=str(period_start.date()).replace("-", ""),
        end_date=str(period_end.date()).replace("-", ""),
        benchmark=benchmark,
        execution_mode="next_open",
        holding_count=holding_count,
        weighting_method="score",
        rebalance_freq=profile.rebalance_freq,
        max_weight=max_weight,
        min_adv20=min_adv20,
        min_price=min_price,
        max_price=max_price,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
        enable_market_regime_filter=bool(profile.use_market_regime_filter),
        regime_ma_window=EXECUTION_ALIGNMENT_REGIME_MA_WINDOW,
        regime_vol_window=EXECUTION_ALIGNMENT_REGIME_VOL_WINDOW,
        regime_max_annual_vol=EXECUTION_ALIGNMENT_REGIME_MAX_ANNUAL_VOL,
        regime_trend_flat_band=EXECUTION_ALIGNMENT_REGIME_TREND_FLAT_BAND,
        regime_vol_transition_band=EXECUTION_ALIGNMENT_REGIME_VOL_TRANSITION_BAND,
        regime_allowed_quadrants=list(EXECUTION_ALIGNMENT_ALLOWED_QUADRANTS),
    )


def _rank_key(metrics: dict[str, float], objective_metric: str) -> tuple[float, float, float, float]:
    if objective_metric == "robust_composite":
        return (
            float(metrics.get("worst_window_excess_sharpe", float("-inf"))),
            float(metrics.get("weak_window_excess_annual_return", float("-inf"))),
            float(metrics.get("mean_window_excess_annual_return", float("-inf"))),
            float(metrics.get("excess_annual_return", 0.0)),
            float(metrics.get("excess_sharpe", 0.0)),
            float(metrics.get("excess_max_drawdown", metrics.get("max_drawdown", 0.0))),
            -float(metrics.get("avg_turnover", 0.0)),
        )
    primary = float(metrics.get(objective_metric, 0.0))
    secondary_name = "excess_annual_return" if objective_metric == "excess_sharpe" else "excess_sharpe"
    secondary = float(metrics.get(secondary_name, 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    avg_turnover = float(metrics.get("avg_turnover", 0.0))
    return (primary, secondary, max_drawdown, -avg_turnover)


def _annualized_return_from_total(total_return: float, trading_days: int) -> float:
    if trading_days <= 0 or np.isnan(total_return):
        return float("nan")
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (252.0 / trading_days) - 1.0)


def _segment_metrics_from_equity(equity_df: pd.DataFrame, start_idx: int, end_idx: int) -> dict[str, float]:
    segment = equity_df.iloc[start_idx : end_idx + 1].copy()
    if segment.empty:
        return {
            "annual_return": float("nan"),
            "excess_annual_return": float("nan"),
            "excess_sharpe": float("nan"),
            "excess_max_drawdown": float("nan"),
        }

    prev_idx = start_idx - 1 if start_idx > 0 else start_idx
    path = equity_df.iloc[prev_idx : end_idx + 1].copy()
    portfolio_path = path["portfolio_equity"].astype(float) / float(path["portfolio_equity"].iloc[0])
    excess_path = path["excess_equity"].astype(float) / float(path["excess_equity"].iloc[0])
    portfolio_returns = segment["portfolio_return"].astype(float)
    excess_returns = segment["excess_return"].astype(float)
    total_return = float(portfolio_path.iloc[-1] - 1.0)
    excess_total_return = float(excess_path.iloc[-1] - 1.0)
    annual_return = _annualized_return_from_total(total_return, len(segment))
    excess_annual_return = _annualized_return_from_total(excess_total_return, len(segment))
    excess_annual_vol = float(excess_returns.std() * np.sqrt(252)) if len(excess_returns) > 1 else 0.0
    return {
        "annual_return": annual_return,
        "excess_annual_return": excess_annual_return,
        "excess_sharpe": float(excess_annual_return / excess_annual_vol) if excess_annual_vol > 0 else float("nan"),
        "excess_max_drawdown": float((excess_path / excess_path.cummax() - 1.0).min()),
        "max_drawdown": float((portfolio_path / portfolio_path.cummax() - 1.0).min()),
        "avg_turnover": float(segment["turnover"].mean()) if "turnover" in segment.columns else float("nan"),
    }


def _build_robustness_metrics(equity_df: pd.DataFrame) -> dict[str, float | int | list[int]]:
    rows: list[dict[str, float | int]] = []
    available_windows: list[int] = []
    for window_days in ROBUSTNESS_WINDOW_DAYS:
        if len(equity_df) < window_days:
            continue
        available_windows.append(int(window_days))
        for start_idx in range(0, len(equity_df) - window_days + 1):
            end_idx = start_idx + window_days - 1
            row = _segment_metrics_from_equity(equity_df, start_idx, end_idx)
            row["window_days"] = int(window_days)
            rows.append(row)

    if not rows:
        return {
            "robust_window_days": available_windows,
            "robust_window_count": 0,
            "mean_window_excess_annual_return": float("nan"),
            "weak_window_excess_annual_return": float("nan"),
            "worst_window_excess_sharpe": float("nan"),
            "worst_window_excess_max_drawdown": float("nan"),
        }

    frame = pd.DataFrame(rows)
    return {
        "robust_window_days": available_windows,
        "robust_window_count": int(len(frame)),
        "mean_window_excess_annual_return": float(frame["excess_annual_return"].mean()),
        "weak_window_excess_annual_return": float(frame["excess_annual_return"].min()),
        "worst_window_excess_sharpe": float(frame["excess_sharpe"].min()),
        "worst_window_excess_max_drawdown": float(frame["excess_max_drawdown"].min()),
    }


def evaluate_profile(
    *,
    raw_target_weights: pd.DataFrame,
    raw_score_frame: pd.DataFrame,
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    open_df: pd.DataFrame,
    benchmark_open: pd.Series,
    benchmark: str,
    holding_count: int,
    max_weight: float,
    min_adv20: float,
    min_price: float,
    max_price: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    profile: ExecutionAlignmentProfile,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    resolved_anchor_date = _resolve_window_anchor_date(raw_target_weights.index, profile.rebalance_anchor_date)
    aligned_target_weights, bridge_meta = build_target_weight_bridge(
        raw_target_weights,
        rebalance_freq=profile.rebalance_freq,
        rebalance_offset=0,
        rebalance_offset_mode=profile.rebalance_offset_mode,
        rebalance_anchor_date=resolved_anchor_date,
        top_k=profile.target_weight_top_k,
        min_weight=profile.target_weight_min_weight,
        power=profile.target_weight_power,
        full_invest=bool(profile.target_weight_full_invest),
    )
    aligned_scores, score_schedule_meta = apply_rebalance_schedule(
        raw_score_frame.fillna(0.0),
        rebalance_freq=profile.rebalance_freq,
        rebalance_offset=0,
        rebalance_offset_mode=profile.rebalance_offset_mode,
        rebalance_anchor_date=resolved_anchor_date,
    )
    exec_cfg = _build_exec_cfg(
        benchmark=benchmark,
        period_start=pd.Timestamp(aligned_target_weights.index.min()),
        period_end=pd.Timestamp(aligned_target_weights.index.max()),
        holding_count=holding_count,
        max_weight=max_weight,
        min_adv20=min_adv20,
        min_price=min_price,
        max_price=max_price,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
        profile=profile,
    )
    regime_state = compute_market_regime_state(benchmark_close, exec_cfg)
    if exec_cfg.enable_market_regime_filter:
        aligned_target_weights, aligned_scores = apply_market_regime_filter(
            target_weights=aligned_target_weights,
            target_scores=aligned_scores,
            regime_state=regime_state,
        )
    period_index = aligned_target_weights.index
    equity_df, action_df, metrics = backtest(
        close=close.reindex(period_index),
        benchmark_close=benchmark_close.reindex(period_index),
        target_weights=aligned_target_weights.reindex(period_index),
        target_scores=aligned_scores.reindex(period_index).fillna(0.0),
        config=exec_cfg,
        regime_on=regime_state["regime_on"].reindex(period_index),
        open_df=open_df.reindex(period_index),
        benchmark_open=benchmark_open.reindex(period_index),
    )
    robustness_metrics = _build_robustness_metrics(equity_df)
    del action_df
    meta = {
        "profile_name": profile.name,
        "profile_description": profile.description,
        "rebalance_freq": str(profile.rebalance_freq),
        "rebalance_offset_mode": str(bridge_meta.get("rebalance_offset_mode", profile.rebalance_offset_mode)),
        "rebalance_anchor_date": str(bridge_meta.get("rebalance_anchor_date", profile.rebalance_anchor_date)),
        "target_weight_top_k": int(bridge_meta.get("target_weight_top_k", profile.target_weight_top_k)),
        "target_weight_min_weight": float(bridge_meta.get("target_weight_min_weight", profile.target_weight_min_weight)),
        "target_weight_power": float(bridge_meta.get("target_weight_power", profile.target_weight_power)),
        "target_weight_full_invest": bool(bridge_meta.get("target_weight_full_invest", profile.target_weight_full_invest)),
        "market_regime_filter": bool(profile.use_market_regime_filter),
        "transaction_cost_bps": float(transaction_cost_bps),
        "slippage_bps": float(slippage_bps),
        "sell_tax_bps": float(sell_tax_bps),
        "rebalance_offsets": bridge_meta.get("rebalance_offsets", [0]),
        "rebalance_sleeve_count": int(bridge_meta.get("rebalance_sleeve_count", 1)),
        "score_rebalance_offset_mode": str(score_schedule_meta.get("rebalance_offset_mode", profile.rebalance_offset_mode)),
    }
    metrics = dict(metrics)
    metrics.update(robustness_metrics)
    metrics.update(meta)
    return metrics, aligned_scores, aligned_target_weights, meta


def fit_execution_alignment(
    *,
    mode: str,
    objective_metric: str,
    candidate_profiles: Iterable[str],
    train_raw_target_weights: pd.DataFrame,
    train_raw_score_frame: pd.DataFrame,
    valid_raw_target_weights: pd.DataFrame,
    valid_raw_score_frame: pd.DataFrame,
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    open_df: pd.DataFrame,
    benchmark_open: pd.Series,
    benchmark: str,
    holding_count: int,
    max_weight: float,
    min_adv20: float,
    min_price: float,
    max_price: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> ExecutionAlignmentArtifact:
    names = [resolve_profile_name(name) for name in candidate_profiles]
    if not names:
        raise ValueError("Execution alignment candidate profile list is empty.")

    objective_rows: list[dict] = []
    best_name = names[0]
    best_metrics: dict[str, float] | None = None
    best_meta: dict[str, object] = {}
    for profile_name in names:
        profile = get_profile(profile_name)
        train_metrics, _, _, train_meta = evaluate_profile(
            raw_target_weights=train_raw_target_weights,
            raw_score_frame=train_raw_score_frame,
            close=close,
            benchmark_close=benchmark_close,
            open_df=open_df,
            benchmark_open=benchmark_open,
            benchmark=benchmark,
            holding_count=holding_count,
            max_weight=max_weight,
            min_adv20=min_adv20,
            min_price=min_price,
            max_price=max_price,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            sell_tax_bps=sell_tax_bps,
            profile=profile,
        )
        row = dict(train_meta)
        row.update(
            {
                "phase": "train_eval",
                "objective_metric": str(objective_metric),
                "annual_return": float(train_metrics.get("annual_return", 0.0)),
                "excess_annual_return": float(train_metrics.get("excess_annual_return", 0.0)),
                "excess_sharpe": float(train_metrics.get("excess_sharpe", 0.0)),
                "mean_window_excess_annual_return": float(train_metrics.get("mean_window_excess_annual_return", float("nan"))),
                "weak_window_excess_annual_return": float(train_metrics.get("weak_window_excess_annual_return", float("nan"))),
                "worst_window_excess_sharpe": float(train_metrics.get("worst_window_excess_sharpe", float("nan"))),
                "worst_window_excess_max_drawdown": float(train_metrics.get("worst_window_excess_max_drawdown", float("nan"))),
                "max_drawdown": float(train_metrics.get("max_drawdown", 0.0)),
                "avg_turnover": float(train_metrics.get("avg_turnover", 0.0)),
            }
        )
        objective_rows.append(row)
        if best_metrics is None or _rank_key(train_metrics, objective_metric) > _rank_key(best_metrics, objective_metric):
            best_name = profile_name
            best_metrics = dict(train_metrics)
            best_meta = dict(train_meta)

    selected_profile = get_profile(best_name)
    combined_target_weights = pd.concat([train_raw_target_weights, valid_raw_target_weights], axis=0).sort_index()
    combined_target_weights = combined_target_weights[~combined_target_weights.index.duplicated(keep="last")]
    combined_score_frame = pd.concat([train_raw_score_frame, valid_raw_score_frame], axis=0).sort_index()
    combined_score_frame = combined_score_frame[~combined_score_frame.index.duplicated(keep="last")]
    _, combined_scores, combined_targets, valid_meta = evaluate_profile(
        raw_target_weights=combined_target_weights,
        raw_score_frame=combined_score_frame,
        close=close,
        benchmark_close=benchmark_close,
        open_df=open_df,
        benchmark_open=benchmark_open,
        benchmark=benchmark,
        holding_count=holding_count,
        max_weight=max_weight,
        min_adv20=min_adv20,
        min_price=min_price,
        max_price=max_price,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
        profile=selected_profile,
    )
    valid_index = valid_raw_target_weights.index
    valid_scores = combined_scores.reindex(valid_index).fillna(0.0)
    valid_target_weights = combined_targets.reindex(valid_index).fillna(0.0)
    valid_cfg = _build_exec_cfg(
        benchmark=benchmark,
        period_start=pd.Timestamp(valid_index.min()),
        period_end=pd.Timestamp(valid_index.max()),
        holding_count=holding_count,
        max_weight=max_weight,
        min_adv20=min_adv20,
        min_price=min_price,
        max_price=max_price,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
        profile=selected_profile,
    )
    valid_regime_state = compute_market_regime_state(benchmark_close, valid_cfg)
    _, _, valid_metrics = backtest(
        close=close.reindex(valid_index),
        benchmark_close=benchmark_close.reindex(valid_index),
        target_weights=valid_target_weights,
        target_scores=valid_scores,
        config=valid_cfg,
        regime_on=valid_regime_state["regime_on"].reindex(valid_index),
        open_df=open_df.reindex(valid_index),
        benchmark_open=benchmark_open.reindex(valid_index),
    )
    valid_metrics = dict(valid_metrics)
    valid_metrics.update(valid_meta)
    objective_rows.append(
        {
            **valid_meta,
            "phase": "valid_selected",
            "objective_metric": str(objective_metric),
            "annual_return": float(valid_metrics.get("annual_return", 0.0)),
            "excess_annual_return": float(valid_metrics.get("excess_annual_return", 0.0)),
            "excess_sharpe": float(valid_metrics.get("excess_sharpe", 0.0)),
            "mean_window_excess_annual_return": float(valid_metrics.get("mean_window_excess_annual_return", float("nan"))),
            "weak_window_excess_annual_return": float(valid_metrics.get("weak_window_excess_annual_return", float("nan"))),
            "worst_window_excess_sharpe": float(valid_metrics.get("worst_window_excess_sharpe", float("nan"))),
            "worst_window_excess_max_drawdown": float(valid_metrics.get("worst_window_excess_max_drawdown", float("nan"))),
            "max_drawdown": float(valid_metrics.get("max_drawdown", 0.0)),
            "avg_turnover": float(valid_metrics.get("avg_turnover", 0.0)),
        }
    )
    return ExecutionAlignmentArtifact(
        mode=str(mode),
        objective_metric=str(objective_metric),
        selected_profile=selected_profile.name,
        selected_profile_description=selected_profile.description,
        selected_profile_spec=profile_to_spec(selected_profile),
        candidate_profiles=names,
        objective_rows=objective_rows,
        selected_bridge_meta=valid_meta,
        selected_train_metrics={} if best_metrics is None else best_metrics,
        valid_metrics=valid_metrics,
        valid_score_frame=valid_scores,
        valid_target_weights=valid_target_weights,
    )
