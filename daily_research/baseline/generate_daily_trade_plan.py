from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import (
    build_prepared_bundle_with_cache,
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
from daily_research.baseline.cli_utils import (
    load_stock_list_from_file,
    parse_csv_list,
    parse_horizon_weights,
    parse_int_tuple,
    parse_state_ensemble_weights,
    parse_state_horizon_profiles,
    parse_stock_list,
)
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    get_latest_completed_trading_date,
    get_next_trading_date,
    load_industry_map_from_tq,
    load_style_map_from_tq,
    load_universe_from_tq,
)
from daily_research.baseline.external_target_weight_bridge import (
    build_target_weight_bridge,
    load_value_panel,
    sanitize_target_weight_row,
)
from daily_research.baseline.soft_state_sizing import apply_soft_state_sizing, resolve_soft_state_profile
from daily_research.baseline.ml_alpha import (
    MLAplhaConfig,
    blend_scores,
    load_ml_artifact,
    predict_ml_scores_for_date_bundle,
    resolve_horizon_weights,
    rolling_ml_scores_multi,
)
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter, resolve_regime_label_series
from daily_research.baseline.state_profiles import validate_state_profile_selector
from daily_research.progress import StageProgress


@dataclass
class PositionSnapshot:
    stock: str
    shares: int
    cost_price: float = 0.0


@dataclass
class AccountStateInput:
    positions_df: pd.DataFrame
    cash: float | None
    source: str
    path: str


def parse_args():
    parser = argparse.ArgumentParser(description="Generate end-of-day trade plan TXT for manual execution")
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None, help="Path to txt/csv file containing stock codes.")
    parser.add_argument("--start-date", default="20210101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2", help=argparse.SUPPRESS)
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--rebalance-offset", type=int, default=0, help="Optional rebalance phase offset in trading days for N-day schedules.")
    parser.add_argument("--rebalance-anchor-date", default="", help="Optional trading-date anchor for rebalance phase alignment, e.g. 2025-01-02.")
    parser.add_argument(
        "--rebalance-offset-mode",
        choices=["single", "all"],
        default="single",
        help="single uses one rebalance phase; all averages every offset sleeve into a phase-robust ensemble.",
    )
    parser.add_argument("--positions-file", default="daily_research/execution/current_positions.csv")
    parser.add_argument("--cash", type=float, default=None, help="Optional cash override. If omitted, will try to read from current_positions.csv account snapshot row.")
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--output-dir", default="daily_research/execution/output")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--model-artifact", default="daily_research/execution/models/latest_ml_model.joblib")
    parser.add_argument("--train-on-the-fly", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--external-score-csv", default="", help="Optional external upstream reference score CSV.")
    parser.add_argument("--external-score-column", default="model_score", help="Score column name inside external score CSV.")
    parser.add_argument("--external-target-weight-csv", default="", help="Optional external target-weight CSV, e.g. deep_alpha daily_target_weight_panel.csv")
    parser.add_argument("--external-target-weight-column", default="target_weight", help="Target-weight column name inside external target-weight CSV.")
    parser.add_argument("--external-watch-target-weight-csv", default="", help=argparse.SUPPRESS)
    parser.add_argument("--external-watch-target-weight-column", default="target_weight", help=argparse.SUPPRESS)
    parser.add_argument("--external-watch-score-csv", default="", help=argparse.SUPPRESS)
    parser.add_argument("--external-watch-score-column", default="score", help=argparse.SUPPRESS)
    parser.add_argument("--external-target-weight-semantics", default="", help=argparse.SUPPRESS)
    parser.add_argument("--external-target-weight-cap-mode", default="", help=argparse.SUPPRESS)
    parser.add_argument("--external-target-weight-cap-note", default="", help=argparse.SUPPRESS)
    parser.add_argument("--external-score-panel-role", default="", help=argparse.SUPPRESS)
    parser.add_argument("--external-score-reference-metadata-json", default="", help=argparse.SUPPRESS)
    parser.add_argument("--external-model-manifest", default="", help=argparse.SUPPRESS)
    parser.add_argument("--external-model-warn-trading-days", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--external-model-max-trading-days", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--target-weight-top-k", type=int, default=0, help="Optional top-k crop applied to direct target-weight rows. 0 keeps all names.")
    parser.add_argument("--target-weight-min-weight", type=float, default=0.0, help="Optional minimum weight threshold applied to direct target-weight rows before renormalization.")
    parser.add_argument("--target-weight-power", type=float, default=1.0, help="Optional power transform applied to positive direct target weights before renormalization.")
    parser.add_argument("--target-weight-full-invest", action="store_true", help="When using direct target weights, renormalize positive rows to 100%% gross even if the source leaves cash.")
    parser.add_argument(
        "--soft-state-profile",
        choices=["off", "quadrant_guard_v1", "trend_guard_v1", "market_state_guard_v1"],
        default="off",
        help="Optional soft state-conditioned gross exposure overlay applied after target weights are built.",
    )
    parser.add_argument(
        "--soft-state-selector",
        choices=["quadrant", "market_state", "trend_bucket", "vol_bucket"],
        default="",
        help="Optional selector override for soft-state sizing. Defaults to the profile's native selector.",
    )
    parser.add_argument(
        "--soft-state-gross-map",
        default="",
        help="Optional label:gross map override, e.g. trend_up_low_vol:1.0,trend_down_high_vol:0.55",
    )
    parser.add_argument("--candidate-label", default="", help="Optional label shown in outputs for external score candidates.")
    parser.add_argument("--stale-model-warn-trading-days", type=int, default=1)
    parser.add_argument("--stale-model-max-trading-days", type=int, default=3)
    parser.add_argument("--allow-stale-model", action="store_true")
    parser.add_argument("--transaction-cost-bps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--slippage-bps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sell-tax-bps", type=float, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.0)

    parser.add_argument("--no-market-regime-filter", action="store_true")
    parser.add_argument("--regime-ma-window", type=int, default=60)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-trend-flat-band", type=float, default=0.01)
    parser.add_argument("--regime-vol-transition-band", type=float, default=0.10)
    parser.add_argument(
        "--regime-state-selector",
        choices=["quadrant", "market_state", "trend_bucket", "vol_bucket"],
        default="quadrant",
        help="State selector used by score switching and ML blend. Defaults to compatibility quadrant.",
    )
    parser.add_argument(
        "--regime-quadrants",
        default="trend_up_low_vol,trend_up_high_vol",
        help="Allowed regime selectors. Supports legacy quadrants plus richer labels such as trend_up_vol_low, trend_flat, vol_mid.",
    )

    parser.add_argument("--no-style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--industry-cap", action="store_true")
    parser.add_argument("--max-industry-weight", type=float, default=0.40)

    parser.add_argument("--ml-target-horizon", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument("--ml-target-horizons", default="5,10,20", help=argparse.SUPPRESS)
    parser.add_argument("--ml-horizon-weights", default="5:0.2,10:0.3,20:0.5", help=argparse.SUPPRESS)
    parser.add_argument("--ml-state-horizon-profiles", default="", help=argparse.SUPPRESS)
    parser.add_argument("--ml-train-window-days", type=int, default=504, help=argparse.SUPPRESS)
    parser.add_argument("--ml-retrain-every-days", type=int, default=21, help=argparse.SUPPRESS)
    parser.add_argument("--ml-min-train-dates", type=int, default=120, help=argparse.SUPPRESS)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600, help=argparse.SUPPRESS)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000, help=argparse.SUPPRESS)
    parser.add_argument("--ml-random-seed", type=int, default=7, help=argparse.SUPPRESS)
    parser.add_argument("--ml-model-family", choices=["histgb", "etr", "lgbm"], default="histgb", help=argparse.SUPPRESS)
    parser.add_argument("--lgbm-n-estimators", type=int, default=260, help=argparse.SUPPRESS)
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70, help=argparse.SUPPRESS)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20, help=argparse.SUPPRESS)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument("--ensemble-state-weights", default="", help=argparse.SUPPRESS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-auto-trim-history", action="store_true")
    return parser.parse_args()


def _load_artifact_meta(artifact_path: Path) -> dict[str, Any]:
    meta_path = artifact_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_timestamp(raw: object) -> pd.Timestamp | None:
    if raw in {None, ""}:
        return None
    try:
        return pd.Timestamp(raw)
    except Exception:
        return None


def _assess_model_freshness(
    *,
    artifact_meta: dict[str, Any],
    latest_signal_date: pd.Timestamp,
    trading_dates: pd.Index,
    warn_trading_days: int,
    max_trading_days: int,
) -> dict[str, Any]:
    latest_data_date = _safe_timestamp(artifact_meta.get("latest_data_date"))
    trained_at = _safe_timestamp(artifact_meta.get("trained_at"))
    result: dict[str, Any] = {
        "status": "unknown",
        "status_text": "unknown",
        "artifact_latest_data_date": str(latest_data_date.date()) if latest_data_date is not None else "",
        "trained_at": str(trained_at) if trained_at is not None else "",
        "trading_day_lag": None,
        "calendar_day_lag": None,
        "warnings": [],
        "should_block": False,
    }

    if latest_data_date is None:
        result["warnings"].append("Model metadata is missing latest_data_date; freshness cannot be verified.")
        return result

    latest_signal_date = pd.Timestamp(latest_signal_date)
    result["calendar_day_lag"] = int((latest_signal_date - latest_data_date).days)

    calendar = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values().unique()
    if latest_signal_date in calendar and latest_data_date in calendar:
        trading_day_lag = int(calendar.get_loc(latest_signal_date) - calendar.get_loc(latest_data_date))
        result["trading_day_lag"] = trading_day_lag
    else:
        trading_day_lag = None

    if trading_day_lag is None:
        result["warnings"].append(
            "Trading-calendar lag could not be computed; verify manually that the model is up to date."
        )
        return result

    if trading_day_lag < 0:
        result["status"] = "future"
        result["status_text"] = "future-dated"
        result["warnings"].append(
            "Model latest_data_date is later than the current signal date; check the data definition."
        )
        return result

    if (max_trading_days > 0 and trading_day_lag >= int(max_trading_days)) or (
        warn_trading_days > 0 and trading_day_lag >= int(warn_trading_days)
    ):
        result["status"] = "lagged"
        result["status_text"] = "lagged"
        return result

    result["status"] = "fresh"
    result["status_text"] = "fresh"
    return result


def _assess_external_signal_freshness(
    *,
    source_signal_date: pd.Timestamp,
    trading_dates: pd.Index,
    warn_trading_days: int,
    max_trading_days: int,
) -> dict[str, Any]:
    latest_completed = pd.Timestamp(get_latest_completed_trading_date())
    result = _assess_model_freshness(
        artifact_meta={"latest_data_date": str(pd.Timestamp(source_signal_date).date())},
        latest_signal_date=latest_completed,
        trading_dates=trading_dates,
        warn_trading_days=warn_trading_days,
        max_trading_days=max_trading_days,
    )
    result["latest_completed_trading_date"] = str(latest_completed.date())
    return result


def _coerce_positive_int(raw: object) -> int:
    try:
        value = int(raw)
    except Exception:
        return 0
    return value if value > 0 else 0


def _assess_external_model_retrain_freshness(
    *,
    manifest_path: Path,
    latest_signal_date: pd.Timestamp,
    trading_dates: pd.Index,
    warn_trading_days: int,
    max_trading_days: int,
) -> dict[str, Any]:
    manifest = _load_json_payload(manifest_path)
    policy = manifest.get("retrain_frequency_policy") if isinstance(manifest.get("retrain_frequency_policy"), dict) else {}
    resolved_warn = _coerce_positive_int(warn_trading_days) or _coerce_positive_int(policy.get("warn_after_trading_days"))
    resolved_max = _coerce_positive_int(max_trading_days) or _coerce_positive_int(policy.get("block_after_trading_days"))
    preferred_label = str(policy.get("preferred_label", "")).strip()
    comparison_window = str(policy.get("comparison_window", "")).strip()
    leaderboard_csv = str(policy.get("leaderboard_csv", "")).strip()
    train_end_date = _safe_timestamp(
        manifest.get("train_end_date")
        or manifest.get("train_end")
        or manifest.get("latest_trainable_date")
    )
    created_at = _safe_timestamp(manifest.get("created_at"))
    launch_cutoff_date = _safe_timestamp(manifest.get("launch_cutoff_date"))
    if launch_cutoff_date is None and created_at is not None:
        launch_cutoff_date = pd.Timestamp(created_at).normalize()
    result: dict[str, Any] = {
        "status": "unknown",
        "status_text": "not-configured",
        "train_end_date": str(train_end_date.date()) if train_end_date is not None else "",
        "launch_cutoff_date": str(launch_cutoff_date.date()) if launch_cutoff_date is not None else "",
        "created_at": str(created_at) if created_at is not None else "",
        "manifest_path": str(manifest_path),
        "active_production_run_dir": str(manifest.get("active_production_run_dir", "")),
        "preferred_label": preferred_label,
        "comparison_window": comparison_window,
        "leaderboard_csv": leaderboard_csv,
        "warn_trading_days": int(resolved_warn),
        "max_trading_days": int(resolved_max),
        "trading_day_lag": None,
        "latest_completed_trading_date": str(pd.Timestamp(latest_signal_date).date()),
        "warnings": [],
        "should_block": False,
        "target_weight_semantics": str(manifest.get("target_weight_semantics", "")),
        "target_weight_cap_mode": str(manifest.get("target_weight_cap_mode", "")),
        "target_weight_cap_note": str(manifest.get("target_weight_cap_note", "")),
        "score_panel_role": str(manifest.get("score_panel_role", "")),
        "score_reference_metadata_json": str(manifest.get("score_reference_metadata_json", "")),
        "transaction_cost_bps": manifest.get("transaction_cost_bps"),
        "slippage_bps": manifest.get("slippage_bps"),
        "sell_tax_bps": manifest.get("sell_tax_bps"),
    }
    if not manifest:
        result["warnings"].append(
            f"Production manifest not found: {manifest_path}; cannot verify whether the underlying model is "
            "past its configured retrain cadence."
        )
        return result
    if train_end_date is None:
        result["warnings"].append(
            f"Production manifest is missing train_end_date: {manifest_path}; cannot verify retrain freshness."
        )
        return result
    if launch_cutoff_date is None:
        result["warnings"].append(
            f"Production manifest is missing launch_cutoff_date: {manifest_path}; cannot verify the freshness of "
            "the latest retrain deployment."
        )
        return result
    if resolved_warn <= 0 and resolved_max <= 0:
        result["status_text"] = "thresholds-missing"
        result["warnings"].append(
            f"Production manifest does not provide retrain thresholds: {manifest_path}; cannot verify configured "
            "retrain freshness against research conclusions."
        )
        return result

    freshness = _assess_model_freshness(
        artifact_meta={
            "latest_data_date": str(launch_cutoff_date.date()),
            "trained_at": str(created_at) if created_at is not None else "",
        },
        latest_signal_date=pd.Timestamp(latest_signal_date),
        trading_dates=trading_dates,
        warn_trading_days=resolved_warn,
        max_trading_days=resolved_max,
    )
    result.update(
        {
            "status": str(freshness.get("status", "unknown")),
            "status_text": str(freshness.get("status_text", "unknown")),
            "trading_day_lag": freshness.get("trading_day_lag"),
            "should_block": bool(freshness.get("should_block", False)),
        }
    )
    lag = freshness.get("trading_day_lag")
    latest_signal_text = str(pd.Timestamp(latest_signal_date).date())
    if freshness.get("status") == "future":
        result["warnings"].extend(list(freshness.get("warnings", [])))
    return result


def _normalize_positions_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["stock", "shares", "cost_price"])
    required = {"stock", "shares"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Positions file missing required columns: {sorted(missing)}")
    out = df.copy()
    if "cost_price" not in out.columns:
        out["cost_price"] = 0.0
    out["stock"] = out["stock"].astype(str).str.upper().str.strip()
    out["shares"] = pd.to_numeric(out["shares"], errors="coerce").fillna(0).astype(int)
    out["cost_price"] = pd.to_numeric(out["cost_price"], errors="coerce").fillna(0.0)
    out = out[(out["stock"] != "") & (out["shares"] > 0)].copy()
    return out[["stock", "shares", "cost_price"]]


def _load_account_state(path_str: str) -> AccountStateInput:
    path = Path(path_str)
    if not path.exists():
        return AccountStateInput(
            positions_df=pd.DataFrame(columns=["stock", "shares", "cost_price"]),
            cash=None,
            source="missing",
            path=str(path),
        )
    df = pd.read_csv(path)
    columns = {str(col).strip().lower(): col for col in df.columns}
    if "record_type" not in columns:
        return AccountStateInput(
            positions_df=_normalize_positions_frame(df),
            cash=None,
            source="positions_only",
            path=str(path),
        )

    record_type_col = columns["record_type"]
    typed = df.copy()
    typed[record_type_col] = typed[record_type_col].astype(str).str.strip().str.lower()

    account_rows = typed[typed[record_type_col].eq("account")].copy()
    cash_value: float | None = None
    if not account_rows.empty:
        cash_col = columns.get("available_cash", columns.get("cash"))
        if cash_col:
            cash_series = pd.to_numeric(account_rows[cash_col], errors="coerce").dropna()
            if not cash_series.empty:
                cash_value = float(cash_series.iloc[-1])

    position_rows = typed[typed[record_type_col].isin(["position", "holding", "hold"])].copy()
    if position_rows.empty and {"stock", "shares"} <= set(columns):
        stock_col = columns["stock"]
        position_rows = typed[typed[stock_col].notna()].copy()
        position_rows = position_rows[position_rows[record_type_col] != "account"].copy()

    return AccountStateInput(
        positions_df=_normalize_positions_frame(position_rows),
        cash=cash_value,
        source="account_snapshot",
        path=str(path),
    )


def _build_scores_from_artifact(
    cfg: ResearchConfig,
    artifact,
    prepared_bundle: Dict[str, object],
    style_map: pd.DataFrame | None,
    industry_map: pd.Series | None,
):
    factor_bundle = prepared_bundle["factor_bundle"]
    regime_state = prepared_bundle["regime_state"]
    score_none = prepared_bundle["score_none"]
    score_v2 = prepared_bundle["score_v2"]
    filter_mask = prepared_bundle["filter_mask"]
    feature_frames = prepared_bundle["feature_frames"]
    market_features = prepared_bundle["market_features"]
    artifact_cfg = MLAplhaConfig(**artifact.ml_config)
    enhanced_profile = str(getattr(artifact_cfg, "enhanced_profile", "up_low_breakout_v2") or "up_low_breakout_v2")
    latest_date = factor_bundle["raw_inputs"]["Close"].index.max()
    state_label_series = resolve_regime_label_series(regime_state, cfg.regime_state_selector)
    latest_regime_label = str(state_label_series.loc[latest_date]) if latest_date in state_label_series.index else ""
    latest_quadrant = str(regime_state.loc[latest_date, "quadrant"])

    model_score_row, _ = predict_ml_scores_for_date_bundle(
        models=artifact.models,
        feature_frames=feature_frames,
        market_features=market_features,
        filter_mask=filter_mask,
        dt=latest_date,
        feature_names=artifact.feature_names,
        horizon_weights=resolve_horizon_weights(artifact_cfg, latest_regime_label),
    )
    model_score = pd.DataFrame(np.nan, index=[latest_date], columns=score_none.columns)
    if not model_score_row.empty:
        model_score.loc[latest_date, model_score_row.index] = model_score_row.values

    score_none_latest = score_none.loc[[latest_date]]
    score_v2_latest = score_v2.loc[[latest_date]]
    final_score_raw = blend_scores(
        model_score,
        score_none_latest,
        score_v2_latest,
        artifact_cfg,
        quadrant_series=state_label_series.loc[[latest_date]],
    )
    target_weights = build_target_weights(final_score_raw, cfg, industry_map=industry_map, style_map=style_map)
    final_score = final_score_raw.copy()
    if cfg.enable_market_regime_filter:
        target_weights, final_score = apply_market_regime_filter(target_weights, final_score.fillna(0.0), regime_state)

    summary_values = list(artifact.train_summary.values()) if artifact.train_summary else []
    train_end = max((item.get("train_end", "") for item in summary_values), default="")
    training_log = pd.DataFrame(
        [
            {
                "source": "artifact",
                "artifact_path": "",
                "trained_at": artifact.trained_at,
                "horizons": ",".join(sorted(artifact.models.keys())),
                "train_end": train_end,
                "quadrant": latest_quadrant,
                "regime_selector": cfg.regime_state_selector,
                "regime_label": latest_regime_label,
                "enhanced_profile": enhanced_profile,
            }
        ]
    )
    return factor_bundle, regime_state, score_none, score_v2, model_score, final_score_raw, final_score, target_weights, training_log


def _build_scores_on_the_fly(
    cfg: ResearchConfig,
    ml_cfg: MLAplhaConfig,
    prepared_bundle: Dict[str, object],
    style_map: pd.DataFrame | None,
    industry_map: pd.Series | None,
):
    factor_bundle = prepared_bundle["factor_bundle"]
    regime_state = prepared_bundle["regime_state"]
    score_none = prepared_bundle["score_none"]
    score_v2 = prepared_bundle["score_v2"]
    filter_mask = prepared_bundle["filter_mask"]
    feature_frames = prepared_bundle["feature_frames"]
    market_features = prepared_bundle["market_features"]
    state_label_series = resolve_regime_label_series(regime_state, cfg.regime_state_selector)
    benchmark_close = prepared_bundle["benchmark_close"]
    benchmark_open = prepared_bundle["benchmark_open"]
    model_score, training_log = rolling_ml_scores_multi(
        feature_frames=feature_frames,
        market_features=market_features,
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        open_df=factor_bundle["raw_inputs"]["Open"],
        benchmark_open=benchmark_open,
        filter_mask=filter_mask,
        regime_state=regime_state,
        config=ml_cfg,
        state_label_series=state_label_series,
    )
    final_score_raw = blend_scores(model_score, score_none, score_v2, ml_cfg, quadrant_series=state_label_series)
    target_weights = build_target_weights(final_score_raw, cfg, industry_map=industry_map, style_map=style_map)
    final_score = final_score_raw.copy()
    if cfg.enable_market_regime_filter:
        target_weights, final_score = apply_market_regime_filter(target_weights, final_score.fillna(0.0), regime_state)

    return factor_bundle, regime_state, score_none, score_v2, model_score, final_score_raw, final_score, target_weights, training_log


def _resolve_column_name(columns: pd.Index, requested: str, *, label: str) -> str:
    lookup = {str(col).strip().lower(): str(col) for col in columns}
    key = str(requested or "").strip().lower()
    if key in lookup:
        return lookup[key]
    raise ValueError(f"{label} column not found: {requested}. Available columns: {list(columns)}")


def _resolve_external_value_column(columns: pd.Index, requested: str, *, label: str) -> str:
    candidates = [str(requested or "").strip()]
    lowered = str(requested or "").strip().lower()
    if lowered in {"model_score", "model_decision_score"}:
        candidates.extend(["score", "latest_score", "model_decision_score", "model_score", "ml_score"])
    elif lowered == "latest_score":
        candidates.extend(["score", "model_decision_score", "model_score", "ml_score"])
    elif lowered == "score":
        candidates.extend(["model_decision_score", "model_score", "latest_score", "ml_score"])
    elif lowered == "ml_score":
        candidates.extend(["model_decision_score", "model_score", "score", "latest_score"])
    elif lowered == "target_weight":
        candidates.append("weight")

    for candidate in candidates:
        try:
            return _resolve_column_name(columns, candidate, label=label)
        except ValueError:
            continue
    raise ValueError(f"{label} column not found: {requested}. Available columns: {list(columns)}")


def _load_external_candidate_rows(
    raw_df: pd.DataFrame,
    *,
    stock_column: str,
    value_column: str,
    latest_market_date: pd.Timestamp,
    label: str,
    requested_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    stock_col = _resolve_column_name(raw_df.columns, stock_column, label="stock")
    value_col = _resolve_external_value_column(raw_df.columns, value_column, label=label)
    columns_lower = {str(col).strip().lower(): str(col) for col in raw_df.columns}
    date_col = columns_lower.get("date")

    working = raw_df.copy()
    signal_date = pd.Timestamp(latest_market_date)
    if date_col is not None:
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
        working = working[working[date_col].notna()].copy()
        if requested_date is not None:
            signal_date = pd.Timestamp(requested_date)
            working = working[working[date_col].eq(signal_date)].copy()
        else:
            working = working[working[date_col] <= pd.Timestamp(latest_market_date)].copy()
            if working.empty:
                raise ValueError(f"{label} CSV has no usable rows on or before latest market date: {latest_market_date.date()}")
            signal_date = pd.Timestamp(working[date_col].max())
            working = working[working[date_col].eq(signal_date)].copy()
        if working.empty:
            raise ValueError(f"{label} CSV has no usable rows for signal date: {signal_date.date()}")

    rows = working[[stock_col, value_col]].copy()
    rows.columns = ["stock", "value"]
    rows["stock"] = rows["stock"].astype(str).str.upper().str.strip()
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows = rows[(rows["stock"] != "") & rows["value"].notna()].copy()
    if rows.empty:
        raise ValueError(f"No usable stock/{label} rows found in external CSV.")
    rows = rows.drop_duplicates(subset=["stock"], keep="last").reset_index(drop=True)
    return rows, signal_date


def _load_external_panel_row(
    *,
    panel_csv: Path,
    value_column: str,
    panel_label: str,
    value_name: str,
    latest_market_date: pd.Timestamp,
    requested_date: pd.Timestamp,
    market_index: pd.DatetimeIndex,
    allowed_columns: pd.Index,
) -> tuple[pd.Series, pd.Timestamp]:
    header = pd.read_csv(panel_csv, nrows=0)
    resolved_value_column = value_column
    columns_lower = {str(col).strip().lower(): str(col) for col in header.columns}
    if "date" in columns_lower and "stock" in columns_lower:
        resolved_value_column = _resolve_external_value_column(
            header.columns,
            value_column,
            label=panel_label,
        )
    panel = load_value_panel(
        panel_csv,
        panel_format="auto",
        date_column="date",
        stock_column="stock",
        value_column=resolved_value_column,
        panel_label=panel_label,
        value_name=value_name,
    )
    panel = (
        panel.reindex(index=market_index, columns=allowed_columns)
        .sort_index()
        .loc[lambda df: df.index <= latest_market_date]
    )
    if panel.empty:
        raise ValueError(f"{panel_label} CSV has no aligned dates inside the current execution calendar: {panel_csv}")
    requested_ts = pd.Timestamp(requested_date)
    valid_index = panel.index[panel.index <= requested_ts]
    if len(valid_index) == 0:
        raise ValueError(f"{panel_label} CSV has no usable rows on or before signal date: {requested_ts.date()}")
    signal_date = pd.Timestamp(valid_index.max())
    row = pd.to_numeric(panel.loc[signal_date], errors="coerce").reindex(allowed_columns)
    return row, signal_date


def _build_scores_from_external_csv(
    cfg: ResearchConfig,
    prepared_bundle: Dict[str, object],
    style_map: pd.DataFrame | None,
    industry_map: pd.Series | None,
    *,
    external_score_csv: Path,
    external_score_column: str,
    candidate_label: str,
):
    factor_bundle = prepared_bundle["factor_bundle"]
    regime_state = prepared_bundle["regime_state"]
    latest_market_date = pd.Timestamp(factor_bundle["raw_inputs"]["Close"].index.max())
    if pd.isna(latest_market_date):
        raise RuntimeError("No valid latest date found while loading external score candidate.")

    score_df = pd.read_csv(external_score_csv)
    if score_df.empty:
        raise ValueError(f"External score CSV is empty: {external_score_csv}")

    raw_scores, signal_date = _load_external_candidate_rows(
        score_df,
        stock_column="stock",
        value_column=external_score_column,
        latest_market_date=latest_market_date,
        label="external score",
    )
    raw_scores = raw_scores.rename(columns={"value": "score"})

    allowed_columns = pd.Index(factor_bundle["raw_inputs"]["Close"].columns.astype(str))
    usable = raw_scores[raw_scores["stock"].isin(allowed_columns)].copy()
    dropped = raw_scores[~raw_scores["stock"].isin(allowed_columns)].copy()
    if usable.empty:
        raise ValueError(
            "External score CSV has no overlap with current execution universe. "
            f"score_csv={external_score_csv}"
        )

    aligned_scores = pd.Series(np.nan, index=allowed_columns, dtype=float)
    aligned_scores.loc[usable["stock"]] = usable["score"].to_numpy(dtype=float)
    final_score_raw = pd.DataFrame([aligned_scores], index=[signal_date])
    final_score_raw.index.name = "date"

    score_none = pd.DataFrame(0.0, index=[signal_date], columns=allowed_columns)
    score_v2 = pd.DataFrame(0.0, index=[signal_date], columns=allowed_columns)
    model_score = final_score_raw.copy()

    target_weights = build_target_weights(final_score_raw, cfg, industry_map=industry_map, style_map=style_map)
    final_score = final_score_raw.copy()
    if cfg.enable_market_regime_filter:
        target_weights, final_score = apply_market_regime_filter(target_weights, final_score.fillna(0.0), regime_state)

    label = str(candidate_label or external_score_csv.resolve().parent.name).strip()
    resolved_score_column = _resolve_external_value_column(score_df.columns, external_score_column, label="external score")
    training_log = pd.DataFrame(
        [
            {
                "source": "external_score_csv",
                "candidate_label": label,
                "score_csv_path": str(external_score_csv),
                "score_column": str(resolved_score_column),
                "loaded_rows": int(len(raw_scores)),
                "usable_rows": int(len(usable)),
                "dropped_rows": int(len(dropped)),
                "signal_date": str(signal_date.date()),
                "train_end": "",
                "trained_at": "",
            }
        ]
    )
    return factor_bundle, regime_state, score_none, score_v2, model_score, final_score_raw, final_score, target_weights, training_log


def _build_scores_from_external_target_weight_csv(
    cfg: ResearchConfig,
    prepared_bundle: Dict[str, object],
    *,
    external_target_weight_csv: Path,
    external_target_weight_column: str,
    candidate_label: str,
    rebalance_offset: int,
    rebalance_offset_mode: str,
    rebalance_anchor_date: str,
    target_weight_top_k: int,
    target_weight_min_weight: float,
    target_weight_power: float,
    target_weight_full_invest: bool,
    external_score_csv: Path | None = None,
    external_score_column: str = "model_score",
):
    factor_bundle = prepared_bundle["factor_bundle"]
    regime_state = prepared_bundle["regime_state"]
    latest_market_date = pd.Timestamp(factor_bundle["raw_inputs"]["Close"].index.max())
    if pd.isna(latest_market_date):
        raise RuntimeError("No valid latest date found while loading external target-weight candidate.")

    allowed_columns = pd.Index(factor_bundle["raw_inputs"]["Close"].columns.astype(str))
    market_index = pd.DatetimeIndex(factor_bundle["raw_inputs"]["Close"].index).sort_values().unique()

    weight_df = pd.read_csv(external_target_weight_csv)
    if weight_df.empty:
        raise ValueError(f"External target-weight CSV is empty: {external_target_weight_csv}")

    raw_weights, source_signal_date = _load_external_candidate_rows(
        weight_df,
        stock_column="stock",
        value_column=external_target_weight_column,
        latest_market_date=latest_market_date,
        label="external target weight",
    )
    raw_weights = raw_weights.rename(columns={"value": "target_weight"})
    usable = raw_weights[raw_weights["stock"].isin(allowed_columns)].copy()
    dropped = raw_weights[~raw_weights["stock"].isin(allowed_columns)].copy()
    if usable.empty and not raw_weights.empty:
        raise ValueError(
            "External target-weight CSV has no overlap with current execution universe. "
            f"target_weight_csv={external_target_weight_csv}"
        )

    target_weight_panel = load_value_panel(
        external_target_weight_csv,
        panel_format="auto",
        date_column="date",
        stock_column="stock",
        value_column=external_target_weight_column,
        panel_label="external target weight",
        value_name="target_weight",
    )
    target_weight_panel = (
        target_weight_panel.reindex(index=market_index, columns=allowed_columns)
        .sort_index()
        .loc[lambda df: df.index <= latest_market_date]
        .fillna(0.0)
    )
    if target_weight_panel.empty:
        raise ValueError(
            "External target-weight CSV has no aligned dates inside the current execution calendar. "
            f"target_weight_csv={external_target_weight_csv}"
        )

    target_weights, bridge_meta = build_target_weight_bridge(
        target_weight_panel,
        rebalance_freq=cfg.rebalance_freq,
        rebalance_offset=rebalance_offset,
        rebalance_offset_mode=rebalance_offset_mode,
        rebalance_anchor_date=rebalance_anchor_date,
        top_k=target_weight_top_k,
        min_weight=target_weight_min_weight,
        power=target_weight_power,
        full_invest=target_weight_full_invest,
    )
    target_weights.index.name = "date"
    signal_date = pd.Timestamp(target_weights.index.max())
    aligned_weights = sanitize_target_weight_row(target_weights.loc[signal_date].reindex(allowed_columns).fillna(0.0))

    source_score_row = pd.Series(np.nan, index=allowed_columns, dtype=float)
    score_context_status = "weight_proxy"
    score_context_warning = ""
    if external_score_csv is not None:
        try:
            score_df = pd.read_csv(external_score_csv)
            if score_df.empty:
                raise ValueError(f"External score CSV is empty: {external_score_csv}")
            raw_scores, score_signal_date = _load_external_candidate_rows(
                score_df,
                stock_column="stock",
                value_column=external_score_column,
                latest_market_date=latest_market_date,
                label="external score",
                requested_date=signal_date,
            )
            if pd.Timestamp(score_signal_date) != pd.Timestamp(signal_date):
                raise ValueError(
                    "External score CSV signal date does not match target-weight signal date. "
                    f"score_date={score_signal_date.date()} target_weight_date={signal_date.date()}"
                )
            raw_scores = raw_scores.rename(columns={"value": "score"})
            usable_scores = raw_scores[raw_scores["stock"].isin(allowed_columns)].copy()
            source_score_row.loc[usable_scores["stock"]] = usable_scores["score"].to_numpy(dtype=float)
            score_context_status = "external_score"
        except ValueError as exc:
            score_context_warning = str(exc)

    final_score_raw = pd.DataFrame([aligned_weights], index=[signal_date])
    final_score_raw.index.name = "date"
    score_none = pd.DataFrame(0.0, index=[signal_date], columns=allowed_columns)
    score_v2 = pd.DataFrame(0.0, index=[signal_date], columns=allowed_columns)
    model_score = pd.DataFrame([source_score_row], index=[signal_date])
    model_score.index.name = "date"

    final_score = final_score_raw.copy()
    if cfg.enable_market_regime_filter:
        target_weights, final_score = apply_market_regime_filter(target_weights, final_score.fillna(0.0), regime_state)

    label = str(candidate_label or external_target_weight_csv.resolve().parent.name).strip()
    training_log = pd.DataFrame(
        [
            {
                "source": "external_target_weight_csv",
                "candidate_label": label,
                "target_weight_csv_path": str(external_target_weight_csv),
                "score_csv_path": str(external_score_csv) if external_score_csv is not None else "",
                "target_weight_column": str(external_target_weight_column),
                "score_column": str(external_score_column),
                "loaded_rows": int(len(raw_weights)),
                "usable_rows": int(len(usable)),
                "dropped_rows": int(len(dropped)),
                "source_signal_date": str(pd.Timestamp(source_signal_date).date()),
                "score_context_status": score_context_status,
                "score_context_warning": score_context_warning,
                "signal_date": str(signal_date.date()),
                "rebalance_freq": str(cfg.rebalance_freq),
                "rebalance_offset_mode": str(bridge_meta.get("rebalance_offset_mode", rebalance_offset_mode)),
                "rebalance_offset": bridge_meta.get("rebalance_offset"),
                "rebalance_sleeve_count": int(bridge_meta.get("rebalance_sleeve_count", 1)),
                "rebalance_anchor_date": str(bridge_meta.get("rebalance_anchor_date", rebalance_anchor_date or "")),
                "target_weight_top_k": int(bridge_meta.get("target_weight_top_k", target_weight_top_k)),
                "target_weight_min_weight": float(bridge_meta.get("target_weight_min_weight", target_weight_min_weight)),
                "target_weight_power": float(bridge_meta.get("target_weight_power", target_weight_power)),
                "target_weight_full_invest": bool(bridge_meta.get("target_weight_full_invest", target_weight_full_invest)),
                "train_end": "",
                "trained_at": "",
            }
        ]
    )
    return factor_bundle, regime_state, score_none, score_v2, model_score, final_score_raw, final_score, target_weights, training_log


def _resolve_plan_display_mode(model_info: Dict[str, Any]) -> str:
    mode = str(model_info.get("mode", "")).strip()
    if mode in {"research_candidate_target_weight_csv", "research_candidate_csv"}:
        return "research_candidate"
    return "native_model"


def _load_optional_json_dict(path_like: Any) -> Dict[str, Any]:
    path_text = str(path_like or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_score_reference_metadata(model_info: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(model_info)
    payload = _load_optional_json_dict(model_info.get("score_reference_metadata_json", ""))
    if not payload:
        return merged
    if "candidate_active_now" in payload:
        merged["effective_live_candidate_active"] = bool(payload.get("candidate_active_now", False))
    if payload.get("latest_month"):
        merged["effective_live_month"] = str(payload.get("latest_month", ""))
    if payload.get("latest_trigger_key"):
        merged["effective_live_trigger_key"] = str(payload.get("latest_trigger_key", ""))
    if payload.get("latest_selected_profile"):
        merged["effective_live_execution_profile"] = str(payload.get("latest_selected_profile", ""))
    if payload.get("effective_execution_profile"):
        merged["effective_live_execution_profile"] = str(payload.get("effective_execution_profile", ""))
    if payload.get("effective_execution_profile_description"):
        merged["effective_live_execution_profile_description"] = str(payload.get("effective_execution_profile_description", ""))
    if payload.get("live_target_weight_mode"):
        merged["effective_live_target_weight_mode"] = str(payload.get("live_target_weight_mode", ""))
    if isinstance(payload.get("effective_execution_bridge_meta"), dict):
        merged["effective_live_execution_bridge_meta"] = dict(payload.get("effective_execution_bridge_meta", {}))
    if payload.get("note"):
        merged["effective_live_score_note"] = str(payload.get("note", ""))
    if payload.get("weight_generation_note"):
        merged["effective_live_weight_generation_note"] = str(payload.get("weight_generation_note", ""))
    return merged


def _resolve_candidate_display_config(model_info: Dict[str, Any]) -> Dict[str, Any]:
    return _resolve_candidate_display_config_v2(model_info)


def _resolve_candidate_display_config_v2(model_info: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(model_info.get("mode", "")).strip()
    status = str(model_info.get("score_context_status", "")).strip().lower()
    score_panel_role = str(model_info.get("score_panel_role", "")).strip().lower()
    if mode == "research_candidate_target_weight_csv":
        if score_panel_role == "execution_preweight_score_panel":
            return {
                "execution_score_label": "执行后排序值",
                "show_source_score": True,
                "source_score_label": "上游参考分",
                "execution_proxy_source": "final_score",
                "source_candidate_source": "model_score",
            }
        return {
            "execution_score_label": "参考排序分",
            "show_source_score": status == "external_score",
            "source_score_label": "上游参考分",
            "execution_proxy_source": "final_score",
            "source_candidate_source": "model_score",
        }
    return {
        "execution_score_label": "候选分数",
        "show_source_score": False,
        "source_score_label": "",
        "execution_proxy_source": "final_score",
        "source_candidate_source": "model_score",
    }


def _decorate_candidate_display_fields(
    df: pd.DataFrame,
    *,
    execution_proxy_source: str = "final_score",
    source_candidate_source: str = "model_score",
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        if "execution_proxy_score" not in out.columns:
            out["execution_proxy_score"] = pd.Series(dtype=float)
        if "source_candidate_score" not in out.columns:
            out["source_candidate_score"] = pd.Series(dtype=float)
        return out
    out = df.copy()
    out["execution_proxy_score"] = pd.to_numeric(out.get(execution_proxy_source), errors="coerce")
    if source_candidate_source:
        out["source_candidate_score"] = pd.to_numeric(out.get(source_candidate_source), errors="coerce")
    else:
        out["source_candidate_score"] = np.nan
    return out


def _export_plan_frame(df: pd.DataFrame, *, model_info: Dict[str, Any], frame_kind: str) -> pd.DataFrame:
    display_mode = _resolve_plan_display_mode(model_info)
    if display_mode != "research_candidate":
        return df

    mode = str(model_info.get("mode", "")).strip()
    watchlist_mode = str(model_info.get("watchlist_mode", "")).strip()
    out = df.copy()
    if frame_kind == "action":
        base_columns = [
            "stock",
            "action",
            "shares",
            "price",
            "est_value",
            "reason",
            "current_weight",
            "target_weight",
            "cost_price",
        ]
        if out.empty:
            for column in base_columns:
                if column not in out.columns:
                    out[column] = pd.Series(dtype=float if "weight" in column or column in {"price", "est_value", "cost_price"} else object)
            if "execution_proxy_score" not in out.columns:
                out["execution_proxy_score"] = pd.Series(dtype=float)
            if "source_candidate_score" not in out.columns:
                out["source_candidate_score"] = pd.Series(dtype=float)
        if mode == "research_candidate_target_weight_csv":
            out = out[base_columns + ["execution_proxy_score", "source_candidate_score"]]
            out = out.rename(columns={"source_candidate_score": "upstream_reference_score"})
        else:
            out = out[base_columns + ["execution_proxy_score"]]
            out = out.rename(columns={"execution_proxy_score": "candidate_score"})
        return out

    if frame_kind == "watch":
        if watchlist_mode == "raw_model_selection":
            if "model_decision_score" not in out.columns and "model_score" in out.columns:
                out["model_decision_score"] = pd.to_numeric(out["model_score"], errors="coerce")
            base_columns = ["date", "stock", "target_weight", "model_decision_score"]
            if out.empty:
                for column in base_columns:
                    if column not in out.columns:
                        out[column] = pd.Series(dtype=float if column in {"target_weight", "model_decision_score"} else object)
            return out[base_columns]
        base_columns = ["date", "stock", "target_weight"]
        if out.empty:
            for column in base_columns:
                if column not in out.columns:
                    out[column] = pd.Series(dtype=float if column == "target_weight" else object)
            if "execution_proxy_score" not in out.columns:
                out["execution_proxy_score"] = pd.Series(dtype=float)
            if "source_candidate_score" not in out.columns:
                out["source_candidate_score"] = pd.Series(dtype=float)
        if mode == "research_candidate_target_weight_csv":
            out = out[base_columns + ["execution_proxy_score", "source_candidate_score"]]
            out = out.rename(
                columns={
                    "execution_proxy_score": "execution_proxy_score",
                    "source_candidate_score": "upstream_reference_score",
                }
            )
        else:
            out = out[base_columns + ["execution_proxy_score"]]
            out = out.rename(columns={"execution_proxy_score": "candidate_score"})
        return out

    return out


def _round_buy_shares(delta_value: float, price: float, lot_size: int) -> int:
    if delta_value <= 0 or price <= 0:
        return 0
    raw = int(delta_value / price)
    return (raw // lot_size) * lot_size


def _round_sell_shares(current_shares: int, target_delta_value: float, price: float, lot_size: int) -> int:
    if current_shares <= 0 or price <= 0:
        return 0
    desired = int(abs(target_delta_value) / price)
    if desired >= current_shares:
        return current_shares
    rounded = (desired // lot_size) * lot_size
    return min(max(rounded, 0), current_shares)


def _build_trade_plan(
    latest_date: pd.Timestamp,
    close_row: pd.Series,
    target_weight_row: pd.Series,
    final_score_row: pd.Series,
    score_none_row: pd.Series,
    score_v2_row: pd.Series,
    model_score_row: pd.Series,
    positions_df: pd.DataFrame,
    cash: float,
    lot_size: int,
    display_mode: str = "native_model",
    execution_proxy_source: str = "final_score",
    source_candidate_source: str = "model_score",
) -> tuple[pd.DataFrame, Dict[str, float]]:
    latest_price = close_row.dropna()
    pos = positions_df.copy()
    if pos.empty:
        pos = pd.DataFrame(columns=["stock", "shares", "cost_price"])
    input_position_count = int(len(pos))
    priced_mask = pos["stock"].isin(latest_price.index)
    unpriced_pos = pos[~priced_mask].copy()
    pos = pos[priced_mask].copy()
    unpriced_position_records = [
        {
            "stock": str(row.stock),
            "shares": int(row.shares),
            "cost_price": float(row.cost_price),
        }
        for row in unpriced_pos.itertuples(index=False)
    ]

    current_value_map = {
        row.stock: float(row.shares) * float(latest_price.get(row.stock, 0.0))
        for row in pos.itertuples(index=False)
    }
    total_equity = float(cash) + float(sum(current_value_map.values()))

    target_weight_row = target_weight_row[target_weight_row > 0].sort_values(ascending=False)
    target_value_map = {stock: total_equity * float(weight) for stock, weight in target_weight_row.items()}
    current_shares_map = {row.stock: int(row.shares) for row in pos.itertuples(index=False)}
    current_cost_map = {row.stock: float(row.cost_price) for row in pos.itertuples(index=False)}

    rows = []
    blocked_buy_rows = []
    available_cash = float(cash)
    planned_shares_map = current_shares_map.copy()

    for row in unpriced_pos.itertuples(index=False):
        stock = str(row.stock)
        target_weight = float(target_weight_row.get(stock, 0.0))
        target_value = float(target_value_map.get(stock, 0.0))
        rows.append(
            {
                "stock": stock,
                "action": "保留" if target_weight > 0 else "卖出",
                "shares": int(row.shares),
                "price": np.nan,
                "est_value": np.nan,
                "reason": "缺少价格，人工核对后保留" if target_weight > 0 else "池外持仓，默认执行建议退出",
                "current_weight": np.nan,
                "target_weight": target_weight,
                "final_score": float(final_score_row.get(stock, 0.0)),
                "score_none": float(score_none_row.get(stock, 0.0)),
                "score_v2": float(score_v2_row.get(stock, 0.0)),
                "model_score": float(model_score_row.get(stock, 0.0)),
                "cost_price": float(row.cost_price),
                "advisory_only": True,
                "target_value": target_value,
            }
        )
        if target_weight <= 0:
            planned_shares_map[stock] = 0

    for stock, shares in sorted(current_shares_map.items()):
        price = float(latest_price.get(stock, 0.0))
        current_value = float(shares * price)
        target_value = float(target_value_map.get(stock, 0.0))
        delta_value = target_value - current_value
        if target_value <= 0:
            sell_shares = shares
            sell_value = sell_shares * price
            available_cash += sell_value
            planned_shares_map[stock] = max(shares - sell_shares, 0)
            rows.append(
                {
                    "stock": stock,
                    "action": "卖出",
                    "shares": sell_shares,
                    "price": price,
                    "est_value": sell_value,
                    "reason": "调出目标组合",
                    "current_weight": current_value / total_equity if total_equity > 0 else 0.0,
                    "target_weight": 0.0,
                    "final_score": float(final_score_row.get(stock, 0.0)),
                    "score_none": float(score_none_row.get(stock, 0.0)),
                    "score_v2": float(score_v2_row.get(stock, 0.0)),
                    "model_score": float(model_score_row.get(stock, 0.0)),
                    "cost_price": float(current_cost_map.get(stock, 0.0)),
                }
            )
        elif delta_value < -price * lot_size:
            sell_shares = _round_sell_shares(shares, delta_value, price, lot_size)
            if sell_shares > 0:
                sell_value = sell_shares * price
                available_cash += sell_value
                planned_shares_map[stock] = max(shares - sell_shares, 0)
                rows.append(
                    {
                        "stock": stock,
                        "action": "减仓",
                        "shares": sell_shares,
                        "price": price,
                        "est_value": sell_value,
                        "reason": "目标仓位下降",
                        "current_weight": current_value / total_equity if total_equity > 0 else 0.0,
                        "target_weight": target_value / total_equity if total_equity > 0 else 0.0,
                        "final_score": float(final_score_row.get(stock, 0.0)),
                        "score_none": float(score_none_row.get(stock, 0.0)),
                        "score_v2": float(score_v2_row.get(stock, 0.0)),
                        "model_score": float(model_score_row.get(stock, 0.0)),
                        "cost_price": float(current_cost_map.get(stock, 0.0)),
                    }
                )

    # Convert target weights to position values before comparing against cash and lot size.
    for stock, target_value in target_value_map.items():
        price = float(latest_price.get(stock, 0.0))
        current_shares = int(current_shares_map.get(stock, 0))
        current_value = float(current_shares * price)
        delta_value = float(target_value - current_value)
        target_weight = float(target_weight_row.get(stock, 0.0))
        if delta_value <= 0:
            continue
        if price <= 0:
            blocked_buy_rows.append(
                {
                    "stock": str(stock),
                    "target_weight": target_weight,
                    "target_value": float(target_value),
                    "price": float(price),
                    "minimum_lot_value": None,
                    "available_cash": float(available_cash),
                    "reason": "missing_price",
                }
            )
            continue
        minimum_lot_value = float(price * lot_size)
        if delta_value <= minimum_lot_value:
            blocked_buy_rows.append(
                {
                    "stock": str(stock),
                    "target_weight": target_weight,
                    "target_value": float(target_value),
                    "price": float(price),
                    "minimum_lot_value": minimum_lot_value,
                    "available_cash": float(available_cash),
                    "reason": "target_value_below_one_lot",
                }
            )
            continue
        planned_buy_value = min(delta_value, available_cash)
        buy_shares = _round_buy_shares(planned_buy_value, price, lot_size)
        if buy_shares <= 0:
            blocked_buy_rows.append(
                {
                    "stock": str(stock),
                    "target_weight": target_weight,
                    "target_value": float(target_value),
                    "price": float(price),
                    "minimum_lot_value": minimum_lot_value,
                    "available_cash": float(available_cash),
                    "reason": "available_cash_below_one_lot",
                }
            )
            continue
        est_value = float(buy_shares * price)
        available_cash -= est_value
        planned_shares_map[stock] = current_shares + buy_shares
        rows.append(
            {
                "stock": stock,
                "action": "买入" if current_shares == 0 else "加仓",
                "shares": buy_shares,
                "price": price,
                "est_value": est_value,
                "reason": "进入目标组合" if current_shares == 0 else "目标仓位上升",
                "current_weight": current_value / total_equity if total_equity > 0 else 0.0,
                "target_weight": target_weight,
                "final_score": float(final_score_row.get(stock, 0.0)),
                "score_none": float(score_none_row.get(stock, 0.0)),
                "score_v2": float(score_v2_row.get(stock, 0.0)),
                "model_score": float(model_score_row.get(stock, 0.0)),
                "cost_price": float(current_cost_map.get(stock, 0.0)),
            }
        )

    action_df = pd.DataFrame(rows)
    action_df = _decorate_candidate_display_fields(
        action_df,
        execution_proxy_source=execution_proxy_source,
        source_candidate_source=source_candidate_source,
    )
    if not action_df.empty:
        action_priority = {"卖出": 0, "减仓": 1, "买入": 2, "加仓": 3, "保留": 4}
        action_df["action_priority"] = action_df["action"].map(action_priority).fillna(99)
        if display_mode == "research_candidate":
            action_df = action_df.sort_values(
                ["action_priority", "target_weight", "final_score", "est_value"],
                ascending=[True, False, False, False],
            ).reset_index(drop=True)
        else:
            action_df = action_df.sort_values(
                ["action_priority", "final_score", "est_value"],
                ascending=[True, False, False],
            ).reset_index(drop=True)
        action_df = action_df.drop(columns=["action_priority"])

    execution_date = get_next_trading_date(latest_date)
    raw_target_position_count = int((target_weight_row > 0).sum())
    actionable_target_position_count = int(sum(1 for shares in planned_shares_map.values() if int(shares) > 0))
    summary = {
        "signal_date": str(latest_date.date()),
        "execution_date": str(execution_date or ""),
        "cash_input": float(cash),
        "total_equity": float(total_equity),
        "estimated_cash_after_plan": float(available_cash),
        "current_position_count": input_position_count,
        "priced_position_count": int(len(current_shares_map)),
        "unpriced_position_count": int(len(unpriced_position_records)),
        "unpriced_positions": unpriced_position_records,
        "unpriced_action_suggestions": [
            {
                "stock": str(row["stock"]),
                "action": str(row["action"]),
                "reason": str(row["reason"]),
                "shares": int(row["shares"]),
                "target_weight": float(row["target_weight"]),
            }
            for row in rows
            if bool(row.get("advisory_only", False))
        ],
        "target_position_count": actionable_target_position_count,
        "raw_target_position_count": raw_target_position_count,
        "actionable_target_position_count": actionable_target_position_count,
        "blocked_buy_candidate_count": int(len(blocked_buy_rows)),
        "blocked_buy_candidates": blocked_buy_rows[:15],
        "lot_size": int(lot_size),
        "price_basis": "signal_close",
    }
    return action_df, summary


def _build_hold_table(
    latest_date: pd.Timestamp,
    close_row: pd.Series,
    target_weight_row: pd.Series,
    final_score_row: pd.Series,
    positions_df: pd.DataFrame,
) -> pd.DataFrame:
    latest_price = close_row.dropna()
    pos = positions_df.copy()
    rows = []
    for row in pos.itertuples(index=False):
        stock = row.stock
        shares = int(row.shares)
        raw_price = latest_price.get(stock, np.nan)
        price = float(raw_price) if pd.notna(raw_price) else np.nan
        target_weight = float(target_weight_row.get(stock, 0.0))
        if pd.isna(price):
            market_value = np.nan
            status = "缺少价格，人工核对后保留" if target_weight > 0 else "池外持仓，默认执行建议退出"
        else:
            market_value = shares * price
            status = "目标持有" if target_weight > 0 else "待卖出"
        rows.append(
            {
                "date": latest_date,
                "stock": stock,
                "shares": shares,
                "close": price,
                "market_value": market_value,
                "cost_price": float(row.cost_price),
                "target_weight": target_weight,
                "final_score": float(final_score_row.get(stock, 0.0)),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _build_watchlist(
    latest_date: pd.Timestamp,
    final_score_row: pd.Series,
    target_weight_row: pd.Series,
    score_none_row: pd.Series,
    score_v2_row: pd.Series,
    model_score_row: pd.Series,
    top_n: int = 15,
    display_mode: str = "native_model",
    execution_proxy_source: str = "final_score",
    source_candidate_source: str = "model_score",
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": latest_date,
            "stock": final_score_row.index,
            "final_score": final_score_row.values,
            "target_weight": target_weight_row.reindex(final_score_row.index).fillna(0.0).values,
            "score_none": score_none_row.reindex(final_score_row.index).values,
            "score_v2": score_v2_row.reindex(final_score_row.index).values,
            "model_score": model_score_row.reindex(final_score_row.index).values,
        }
    )
    df = _decorate_candidate_display_fields(
        df,
        execution_proxy_source=execution_proxy_source,
        source_candidate_source=source_candidate_source,
    )
    if display_mode == "research_candidate":
        return (
            df.sort_values(["target_weight", "execution_proxy_score", "stock"], ascending=[False, False, True])
            .head(top_n)
            .reset_index(drop=True)
    )
    return df.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)


def _build_raw_model_watchlist(
    latest_date: pd.Timestamp,
    *,
    model_score_row: pd.Series,
    target_weight_row: pd.Series,
    top_n: int = 15,
) -> pd.DataFrame:
    aligned_index = pd.Index(model_score_row.index.astype(str))
    df = pd.DataFrame(
        {
            "date": latest_date,
            "stock": aligned_index,
            "target_weight": target_weight_row.reindex(aligned_index).fillna(0.0).values,
            "model_decision_score": pd.to_numeric(model_score_row.reindex(aligned_index), errors="coerce").values,
        }
    )
    df["model_score"] = df["model_decision_score"]
    df = df[(df["target_weight"] > 0.0) | (df["model_decision_score"].fillna(0.0) > 0.0)].copy()
    if df.empty:
        return df
    return (
        df.sort_values(["target_weight", "model_decision_score", "stock"], ascending=[False, False, True])
        .head(top_n)
        .reset_index(drop=True)
    )


_TERMINAL_ACTION_LABELS = {
    "卖出": "Sell",
    "减仓": "Trim",
    "买入": "Buy",
    "加仓": "Add",
    "保留": "Keep",
}

_TERMINAL_REASON_LABELS = {
    "调出目标组合": "Removed from target portfolio",
    "目标仓位下降": "Target weight decreased",
    "进入目标组合": "Entered target portfolio",
    "目标仓位上升": "Target weight increased",
    "池外持仓，默认执行建议退出": "Out-of-pool holding, default execution suggests exit",
    "缺少价格，人工核对后保留": "Missing price, keep after manual verification",
}


def _terminal_action_preview(action_df: pd.DataFrame) -> pd.DataFrame:
    preview = action_df[["stock", "action", "shares", "price", "reason"]].copy()
    preview["action"] = preview["action"].map(lambda value: _TERMINAL_ACTION_LABELS.get(str(value), str(value)))
    preview["reason"] = preview["reason"].map(lambda value: _TERMINAL_REASON_LABELS.get(str(value), str(value)))
    return preview


def _apply_soft_state_overlay(
    *,
    args: argparse.Namespace,
    regime_state: pd.DataFrame,
    target_weights: pd.DataFrame,
    model_info: dict[str, Any],
    training_log: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], pd.DataFrame]:
    profile = resolve_soft_state_profile(
        profile=args.soft_state_profile,
        selector=(args.soft_state_selector or None),
        gross_map_raw=(args.soft_state_gross_map or None),
    )
    scaled_weights, soft_state_scale, soft_state_meta = apply_soft_state_sizing(
        target_weights,
        regime_state,
        profile_meta=profile,
    )
    model_info.update(soft_state_meta)
    if training_log is not None and not training_log.empty:
        training_log.loc[:, "soft_state_profile"] = str(soft_state_meta.get("soft_state_profile", "off"))
        training_log.loc[:, "soft_state_enabled"] = bool(soft_state_meta.get("soft_state_enabled", False))
        training_log.loc[:, "soft_state_selector"] = str(soft_state_meta.get("soft_state_selector", "quadrant"))
        training_log.loc[:, "soft_state_scale_mean"] = float(soft_state_meta.get("soft_state_scale_mean", 1.0))
        training_log.loc[:, "soft_state_scale_active_ratio"] = float(
            soft_state_meta.get("soft_state_scale_active_ratio", 0.0)
        )
        training_log.loc[:, "soft_state_gross_map"] = json.dumps(
            soft_state_meta.get("soft_state_gross_map", {}),
            ensure_ascii=False,
            sort_keys=True,
        )
    return scaled_weights, soft_state_scale, soft_state_meta, training_log


def _write_trade_plan_txt(
    path: Path,
    summary: Dict[str, float],
    regime_state_row: pd.Series,
    action_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    watch_df: pd.DataFrame,
    model_info: Dict[str, str],
):
    blocked_reason_labels = {
        "missing_price": "缺少最新价格",
        "target_value_below_one_lot": "目标金额不足一手",
        "available_cash_below_one_lot": "可用现金不足一手",
    }

    def _fmt_metric(value: object) -> str:
        try:
            number = float(value)
        except Exception:
            return "nan"
        if np.isnan(number):
            return "nan"
        return f"{number:.3f}"

    display_mode = _resolve_plan_display_mode(model_info)
    candidate_display = _resolve_candidate_display_config_v2(model_info)
    execution_score_label = str(candidate_display.get("execution_score_label", "候选分数"))
    show_source_score = bool(candidate_display.get("show_source_score", False))
    source_score_label = str(candidate_display.get("source_score_label", "上游参考分"))
    candidate_mode = str(model_info.get("mode", "")).strip()
    watchlist_mode = str(model_info.get("watchlist_mode", "")).strip()
    target_weight_semantics = str(model_info.get("target_weight_semantics", "")).strip()
    target_weight_cap_mode = str(model_info.get("target_weight_cap_mode", "")).strip()
    target_weight_cap_note = str(model_info.get("target_weight_cap_note", "")).strip()
    score_panel_role = str(model_info.get("score_panel_role", "")).strip()
    effective_live_target_weight_mode = str(model_info.get("effective_live_target_weight_mode", "")).strip()
    effective_live_execution_profile = str(model_info.get("effective_live_execution_profile", "")).strip()
    effective_live_execution_profile_description = str(model_info.get("effective_live_execution_profile_description", "")).strip()
    effective_live_trigger_key = str(model_info.get("effective_live_trigger_key", "")).strip()
    effective_live_score_note = str(model_info.get("effective_live_score_note", "")).strip()
    effective_live_weight_generation_note = str(model_info.get("effective_live_weight_generation_note", "")).strip()
    effective_live_bridge_meta = (
        model_info.get("effective_live_execution_bridge_meta", {})
        if isinstance(model_info.get("effective_live_execution_bridge_meta"), dict)
        else {}
    )
    transaction_cost_bps = model_info.get("transaction_cost_bps")
    slippage_bps = model_info.get("slippage_bps")
    sell_tax_bps = model_info.get("sell_tax_bps")
    lines: List[str] = []
    lines.append("每日盘后策略（次日开盘执行）")
    lines.append("=" * 36)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"信号日期: {summary['signal_date']}")
    if summary.get("execution_date"):
        lines.append(f"执行日期: {summary['execution_date']}")
    lines.append("执行方式: 盘后生成建议，下一交易日开盘手工执行")
    lines.append(f"市场状态: {regime_state_row.get('quadrant', '')}")
    if bool(model_info.get("market_regime_filter_enabled", True)):
        lines.append(f"市场过滤: 开启 | 允许开仓: {'是' if bool(regime_state_row.get('regime_on', False)) else '否'}")
    else:
        lines.append("市场过滤: 关闭 | 当前市场状态仅展示，不拦截开仓")
    lines.append(f"模型来源: {model_info.get('mode', '')}")
    if model_info.get("artifact_path"):
        lines.append(f"模型文件: {model_info['artifact_path']}")
    if model_info.get("candidate_label"):
        lines.append(f"候选标签: {model_info['candidate_label']}")
    if model_info.get("candidate_score_csv"):
        if candidate_mode == "research_candidate_target_weight_csv" and score_panel_role == "execution_preweight_score_panel":
            score_file_label = "上游参考分文件"
        else:
            score_file_label = "上游参考分文件" if candidate_mode == "research_candidate_target_weight_csv" else "候选分数文件"
        lines.append(f"{score_file_label}: {model_info['candidate_score_csv']}")
    if model_info.get("candidate_target_weight_csv"):
        lines.append(f"候选权重文件: {model_info['candidate_target_weight_csv']}")
    if model_info.get("watch_candidate_score_csv"):
        lines.append(f"研究候选综合决策分文件: {model_info['watch_candidate_score_csv']}")
    if model_info.get("watch_candidate_target_weight_csv"):
        lines.append(f"研究候选原始权重文件: {model_info['watch_candidate_target_weight_csv']}")
    if model_info.get("candidate_usable_rows") is not None and model_info.get("candidate_total_rows") is not None:
        coverage_label = "候选权重覆盖度" if candidate_mode == "research_candidate_target_weight_csv" else "候选分数覆盖度"
        lines.append(
            f"{coverage_label}: "
            f"{model_info.get('candidate_usable_rows', 0)}/{model_info.get('candidate_total_rows', 0)} "
            f"(dropped={model_info.get('candidate_dropped_rows', 0)})"
        )
    if display_mode == "research_candidate":
        lines.append("展示模式: 研究候选模式")
        lines.append(f"执行排序口径: 先按目标权重，再按{execution_score_label}")
        if show_source_score:
            lines.append(f"{source_score_label}: 仅作来源参考，不参与执行排序")
        if candidate_mode == "research_candidate_target_weight_csv" and score_panel_role == "execution_preweight_score_panel":
            lines.append("执行语义说明: 最终执行以桥接后的目标权重与执行后排序值为准；上游参考分只是来源参考，不保证与最终权重单调一致。")
        if candidate_mode == "research_candidate_target_weight_csv":
            if target_weight_semantics:
                lines.append(f"权重语义: {target_weight_semantics}")
            if target_weight_cap_mode:
                lines.append(f"单票上限语义: {target_weight_cap_mode}")
            if target_weight_cap_note:
                lines.append(f"权重说明: {target_weight_cap_note}")
            if score_panel_role:
                lines.append(f"分数面板角色: {score_panel_role}")
            if transaction_cost_bps is not None and slippage_bps is not None and sell_tax_bps is not None:
                lines.append(
                    "成本口径: "
                    f"transaction {float(transaction_cost_bps):.1f} bps | "
                    f"slippage {float(slippage_bps):.1f} bps | "
                    f"sell_tax {float(sell_tax_bps):.1f} bps"
                )
            if effective_live_target_weight_mode or effective_live_execution_profile:
                live_parts: list[str] = []
                if effective_live_target_weight_mode:
                    live_parts.append(f"mode={effective_live_target_weight_mode}")
                if effective_live_execution_profile:
                    live_parts.append(f"profile={effective_live_execution_profile}")
                if effective_live_trigger_key:
                    live_parts.append(f"trigger={effective_live_trigger_key}")
                if effective_live_bridge_meta:
                    bridge_parts: list[str] = []
                    if effective_live_bridge_meta.get("rebalance_freq"):
                        bridge_parts.append(str(effective_live_bridge_meta.get("rebalance_freq", "")))
                    if effective_live_bridge_meta.get("rebalance_offset_mode"):
                        bridge_parts.append(f"offset={effective_live_bridge_meta.get('rebalance_offset_mode', '')}")
                    if effective_live_bridge_meta.get("rebalance_sleeve_count") is not None:
                        bridge_parts.append(f"sleeves={effective_live_bridge_meta.get('rebalance_sleeve_count')}")
                    if effective_live_bridge_meta.get("target_weight_top_k") is not None:
                        bridge_parts.append(f"topk={effective_live_bridge_meta.get('target_weight_top_k')}")
                    if bridge_parts:
                        live_parts.append("bridge=" + "/".join(str(part) for part in bridge_parts if str(part)))
                lines.append("当前有效执行态: " + " | ".join(live_parts))
            if effective_live_execution_profile_description:
                lines.append(f"有效执行说明: {effective_live_execution_profile_description}")
            if effective_live_score_note:
                lines.append(f"分数说明: {effective_live_score_note}")
            if effective_live_weight_generation_note:
                lines.append(f"权重生成说明: {effective_live_weight_generation_note}")
    if model_info.get("trained_at"):
        lines.append(f"模型训练时间: {model_info['trained_at']}")
    if model_info.get("train_end"):
        lines.append(f"模型训练样本截止: {model_info['train_end']}")
    if model_info.get("source_signal_date") and model_info.get("source_signal_date") != model_info.get("latest_data_date"):
        lines.append(f"候选源信号日: {model_info['source_signal_date']}")
    if model_info.get("execution_signal_date") and model_info.get("execution_signal_date") != model_info.get("source_signal_date"):
        lines.append(f"候选执行信号日: {model_info['execution_signal_date']}")
    if model_info.get("latest_data_date"):
        lines.append(f"{model_info.get('latest_data_label', '模型最新数据日')}: {model_info['latest_data_date']}")
    if model_info.get("freshness_status"):
        freshness_line = f"{model_info.get('freshness_label', '模型新鲜度')}: {model_info['freshness_status']}"
        if model_info.get("trading_day_lag") is not None:
            freshness_line += f" | 交易日滞后 {model_info['trading_day_lag']}"
        if model_info.get("latest_completed_trading_date"):
            freshness_line += f" | 最新完成交易日 {model_info['latest_completed_trading_date']}"
        lines.append(freshness_line)
    if model_info.get("production_model_train_end_date"):
        lines.append(f"底层模型训练样本截止: {model_info['production_model_train_end_date']}")
    if model_info.get("production_model_launch_cutoff_date"):
        lines.append(f"底层模型最近一次上线截止: {model_info['production_model_launch_cutoff_date']}")
    if model_info.get("production_model_retrain_policy"):
        lines.append(f"底层模型重训策略: {model_info['production_model_retrain_policy']}")
    if model_info.get("production_model_retrain_window"):
        lines.append(f"重训研究比较窗: {model_info['production_model_retrain_window']}")
    if model_info.get("production_model_retrain_status"):
        retrain_line = f"{model_info.get('production_model_retrain_label', '底层模型重训时效')}: {model_info['production_model_retrain_status']}"
        if model_info.get("production_model_retrain_trading_day_lag") is not None:
            retrain_line += f" | 交易日滞后 {model_info['production_model_retrain_trading_day_lag']}"
        lines.append(retrain_line)
    history_window = model_info.get("history_window")
    if isinstance(history_window, dict) and history_window.get("effective_start_date"):
        lines.append(
            "历史窗口: "
            f"{history_window.get('effective_start_date')} -> {history_window.get('end_date') or 'latest'} "
            f"(required_trading_days={history_window.get('required_trading_days')})"
        )
    if model_info.get("horizons"):
        lines.append(f"模型周期: {model_info['horizons']}")
    if model_info.get("model_family"):
        lines.append(f"模型族: {model_info['model_family']}")
    if model_info.get("horizon_weights"):
        lines.append(f"模型周期权重: {model_info['horizon_weights']}")
    if model_info.get("state_horizon_profiles"):
        lines.append(f"状态周期权重: {model_info['state_horizon_profiles']}")
    if model_info.get("state_ensemble_weights"):
        lines.append(f"状态集成权重: {model_info['state_ensemble_weights']}")
    if model_info.get("lgbm_n_estimators"):
        lines.append(f"LGBM Trees: {model_info['lgbm_n_estimators']}")
    if model_info.get("enhanced_profile"):
        lines.append(f"Enhanced Profile: {model_info['enhanced_profile']}")
    if model_info.get("soft_state_enabled"):
        lines.append(
            "Soft State Sizing: "
            f"{model_info.get('soft_state_profile', 'custom')} | "
            f"selector={model_info.get('soft_state_selector', '')} | "
            f"mean_gross={_fmt_metric(model_info.get('soft_state_scale_mean'))} | "
            f"active_ratio={_fmt_metric(model_info.get('soft_state_scale_active_ratio'))}"
        )
    validation_summary = model_info.get("validation_summary")
    if isinstance(validation_summary, dict):
        combined = validation_summary.get("combined")
        if isinstance(combined, dict):
            snapshot_parts: list[str] = []
            for key, label in (("full", "full"), ("recent_252d", "recent252d"), ("recent_63d", "recent63d")):
                item = combined.get(key)
                if not isinstance(item, dict) or int(item.get("obs_days", 0) or 0) <= 0:
                    continue
                snapshot_parts.append(
                    f"{label} IC {_fmt_metric(item.get('mean_rank_ic'))}/IR {_fmt_metric(item.get('rank_ic_ir'))}"
                )
            if snapshot_parts:
                lines.append("模型验证: " + " | ".join(snapshot_parts))
    warnings = model_info.get("warnings") or []
    if warnings:
        lines.append("模型提醒:")
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.append(f"总资产估算: {summary['total_equity']:.2f}")
    if summary.get("cash_source_text"):
        lines.append(f"现金来源: {summary['cash_source_text']}")
    if summary.get("positions_source_mtime"):
        lines.append(f"持仓文件时间: {summary['positions_source_mtime']}")
    lines.append(f"输入现金: {summary['cash_input']:.2f}")
    lines.append(f"计划后剩余现金估算: {summary['estimated_cash_after_plan']:.2f}")
    if summary.get("current_position_count") is not None:
        priced_position_count = int(summary.get("priced_position_count", summary.get("current_position_count", 0)) or 0)
        current_position_count = int(summary.get("current_position_count", 0) or 0)
        unpriced_position_count = int(summary.get("unpriced_position_count", 0) or 0)
        if current_position_count > 0:
            lines.append(
                f"持仓识别: 输入持仓 {current_position_count} | 可估价 {priced_position_count} | 缺少价格 {unpriced_position_count}"
            )
        if unpriced_position_count > 0:
            lines.append("持仓识别提醒:")
            for row in list(summary.get("unpriced_positions") or [])[:8]:
                cost_price = pd.to_numeric(pd.Series([row.get('cost_price', np.nan)]), errors='coerce').iloc[0]
                cost_text = f"{float(cost_price):.2f}" if pd.notna(cost_price) else "缺失"
                lines.append(
                    f"- {row.get('stock', '')} | 持股 {int(row.get('shares', 0) or 0)} 股 | 成本价 {cost_text} | "
                    "不在当前执行价格宇宙或缺少最新收盘价，未纳入自动估值与自动调仓。"
                )
            lines.append("- 上述持仓不会被解释成“无持仓”或“建议继续持有”，需要人工先核对。")
            lines.append("- 当前总资产估算不含上述缺少价格持仓的市值。")
    blocked_buy_count = int(summary.get("blocked_buy_candidate_count", 0) or 0)
    if blocked_buy_count > 0:
        lines.append(
            f"买入约束提醒: 有 {blocked_buy_count} 个正目标因一手/现金约束未转成可执行买入（lot_size={int(summary.get('lot_size', 0) or 0)}）。"
        )
        for row in list(summary.get("blocked_buy_candidates") or [])[:5]:
            min_lot_value = row.get("minimum_lot_value")
            min_lot_text = f"{float(min_lot_value):.2f}" if min_lot_value is not None else "缺失"
            lines.append(
                f"- {row.get('stock', '')} | 目标权重 {float(row.get('target_weight', 0.0)):.2%} | "
                f"目标金额 {float(row.get('target_value', 0.0)):.2f} | 一手门槛 {min_lot_text} | "
                f"可用现金 {float(row.get('available_cash', 0.0)):.2f} | 原因 {blocked_reason_labels.get(str(row.get('reason', '')), str(row.get('reason', '')))}"
            )
    if display_mode == "research_candidate":
        lines.append("价格口径: 以下数量按信号日收盘价估算，仅用于把目标权重换算成股数；次日开盘请按实际开盘价与目标权重执行。")
    else:
        lines.append("价格口径: 以下数量按信号日收盘价估算，次日开盘请按实际开盘价微调。")
    lines.append("")

    if action_df.empty:
        lines.append("一、次日开盘建议动作")
        if int(summary.get("unpriced_position_count", 0) or 0) > 0:
            lines.append("- 当前未生成明确自动调仓动作；存在缺少价格/不在执行价格宇宙的持仓，系统无法安全估值并自动给出调仓股数。")
        elif int(summary.get("raw_target_position_count", 0) or 0) > 0 and int(summary.get("blocked_buy_candidate_count", 0) or 0) > 0:
            lines.append("- 当前未生成明确调仓动作；模型存在正目标，但按当前总资产、收盘价和一手约束，暂时无法形成可执行买入股数。")
        else:
            lines.append("- 当前无明确调仓动作，建议次日开盘保持现有仓位。")
    else:
        lines.append("一、次日开盘建议动作")
        for idx, row in action_df.iterrows():
            price_text = f"{float(row['price']):.2f}" if pd.notna(row.get("price", np.nan)) else "待核对"
            est_value_text = f"{float(row['est_value']):.2f}" if pd.notna(row.get("est_value", np.nan)) else "待核对"
            current_weight_text = (
                f"{float(row['current_weight']):.2%}" if pd.notna(row.get("current_weight", np.nan)) else "待核对"
            )
            lines.append(
                f"{idx + 1}. {row['action']} {row['stock']} | 估算数量 {int(row['shares'])} 股 | 估算价格基准 {price_text} | "
                f"估算金额 {est_value_text} | 原因: {row['reason']}"
            )
            if display_mode == "research_candidate":
                detail = (
                    f"   当前权重 {current_weight_text} "
                    f"-> 目标权重 {row['target_weight']:.2%} | "
                    f"{execution_score_label} {row['execution_proxy_score']:.4f}"
                )
                if show_source_score and pd.notna(row.get("source_candidate_score", np.nan)):
                    detail += f" | {source_score_label} {row['source_candidate_score']:.4f}"
                lines.append(detail)
            else:
                lines.append(
                    f"   当前权重 {row['current_weight']:.2%} -> 目标权重 {row['target_weight']:.2%} | "
                    f"综合分 {row['final_score']:.4f} | 模型综合决策分 {row['model_score']:.4f} | none {row['score_none']:.4f} | v2 {row['score_v2']:.4f}"
                )

    lines.append("")
    lines.append("二、当前持仓概览")
    if hold_df.empty:
        lines.append("- 当前无持仓。")
    else:
        for _, row in hold_df.iterrows():
            close_text = f"{float(row['close']):.2f}" if pd.notna(row.get("close", np.nan)) else "缺失"
            market_value_text = f"{float(row['market_value']):.2f}" if pd.notna(row.get("market_value", np.nan)) else "缺失"
            lines.append(
                f"- {row['stock']} | 持股 {int(row['shares'])} 股 | 收盘 {close_text} | 市值 {market_value_text} | "
                f"目标权重 {row['target_weight']:.2%} | 状态 {row['status']}"
            )

    lines.append("")
    if display_mode == "research_candidate":
        lines.append("三、研究候选观察名单")
        if watchlist_mode == "raw_model_selection":
            lines.append("- 观察名单使用未经过执行桥的模型原始候选权重与模型综合决策分。")
            for _, row in watch_df.iterrows():
                model_decision_score = pd.to_numeric(
                    pd.Series([row.get("model_decision_score", row.get("model_score", np.nan))]),
                    errors="coerce",
                ).iloc[0]
                score_text = f"{float(model_decision_score):.4f}" if pd.notna(model_decision_score) else "nan"
                lines.append(
                    f"- {row['stock']} | 原始目标权重 {row['target_weight']:.2%} | 模型综合决策分 {score_text}"
                )
        else:
            for _, row in watch_df.iterrows():
                line = (
                    f"- {row['stock']} | 目标权重 {row['target_weight']:.2%} | "
                    f"{execution_score_label} {row['execution_proxy_score']:.4f}"
                )
                if show_source_score and pd.notna(row.get("source_candidate_score", np.nan)):
                    line += f" | {source_score_label} {row['source_candidate_score']:.4f}"
                lines.append(line)
    else:
        lines.append("三、候选观察名单")
        for _, row in watch_df.iterrows():
            lines.append(
                f"- {row['stock']} | 综合分 {row['final_score']:.4f} | 目标权重 {row['target_weight']:.2%} | "
                f"模型综合决策分 {row['model_score']:.4f} | none {row['score_none']:.4f} | v2 {row['score_v2']:.4f}"
            )

    lines.append("")
    lines.append("四、执行提示")
    lines.append("- 本文件用于盘后生成、次日开盘执行。")
    lines.append("- 次日开盘前先核对可用现金、持仓与竞价情况。")
    lines.append("- 先处理卖出/减仓，再处理买入/加仓。")
    lines.append("- 若次日开盘出现明显跳空，请优先按目标权重而不是按估算股数机械执行。")
    lines.append("- 若实际可用资金与本文件不同，请以卖出后实际资金为准调整买入数量。")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    validate_state_profile_selector(args.enhanced_profile, args.regime_state_selector)
    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        execution_mode="next_open",
        holding_count=args.holding_count,
        weighting_method="score",
        rebalance_freq=args.rebalance_freq,
        score_threshold=args.score_threshold,
        max_weight=args.max_weight,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        enable_market_regime_filter=not args.no_market_regime_filter,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_trend_flat_band=args.regime_trend_flat_band,
        regime_vol_transition_band=args.regime_vol_transition_band,
        regime_state_selector=args.regime_state_selector,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )
    stocks = parse_stock_list(args.stocks)
    file_stocks = load_stock_list_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if stocks:
        cfg.universe = stocks

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=parse_horizon_weights(args.ml_horizon_weights),
        state_horizon_weights=parse_state_horizon_profiles(args.ml_state_horizon_profiles),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=args.ml_retrain_every_days,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        lgbm_n_estimators=args.lgbm_n_estimators,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        state_ensemble_weights=parse_state_ensemble_weights(args.ensemble_state_weights),
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    artifact = None
    artifact_meta: dict[str, Any] = {}
    artifact_path = Path(args.model_artifact)
    external_score_path = Path(args.external_score_csv).expanduser() if str(args.external_score_csv).strip() else None
    external_target_weight_path = (
        Path(args.external_target_weight_csv).expanduser() if str(args.external_target_weight_csv).strip() else None
    )
    external_watch_target_weight_path = (
        Path(args.external_watch_target_weight_csv).expanduser()
        if str(args.external_watch_target_weight_csv).strip()
        else None
    )
    external_watch_score_path = (
        Path(args.external_watch_score_csv).expanduser() if str(args.external_watch_score_csv).strip() else None
    )
    effective_profile = args.enhanced_profile
    effective_ml_cfg = ml_cfg
    if not args.train_on_the_fly:
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {artifact_path}. Run daily_research/execution/update_model.py first."
            )
        artifact = load_ml_artifact(artifact_path)
        artifact_meta = _load_artifact_meta(artifact_path)
        effective_ml_cfg = MLAplhaConfig(**artifact.ml_config)
        effective_profile = str(getattr(effective_ml_cfg, "enhanced_profile", "up_low_breakout_v2") or "up_low_breakout_v2")
        validate_state_profile_selector(effective_profile, cfg.regime_state_selector)

    if args.data_source == "tq":
        if not cfg.universe and cfg.universe_scope == "all_a":
            print("[1/9] Loading all-A universe from TQ...")
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe:
            raise ValueError("TQ mode currently requires --universe-scope all_a when --stocks is not provided.")
    elif not args.csv_folder:
        raise ValueError("CSV mode requires --csv-folder.")

    history_window = resolve_history_window(
        cfg=cfg,
        ml_cfg=effective_ml_cfg,
        requested_start_date=args.start_date,
        end_date=args.end_date,
        mode="infer",
        auto_trim_history=not args.no_auto_trim_history,
    )
    print(
        f"[2/9] Inference history window: {history_window.effective_start_date} -> "
        f"{history_window.end_date or 'latest'} | required_trading_days={history_window.required_trading_days}"
    )
    print(f"[3/9] Preparing market data, stocks={len(cfg.universe)} benchmark={cfg.benchmark}")
    raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        universe=cfg.universe,
        benchmark=cfg.benchmark,
        history_window=history_window,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
    print(
        f"[4/9] raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | "
        f"{raw_cache_meta['cache_path']}"
    )

    print("[5/9] Loading current account snapshot...")
    account_state = _load_account_state(args.positions_file)
    positions_df = account_state.positions_df
    if args.cash is not None:
        effective_cash = float(args.cash)
        cash_source = "cli_override"
        cash_source_text = f"命令行 --cash 覆盖 ({args.positions_file})"
    elif account_state.cash is not None:
        effective_cash = float(account_state.cash)
        cash_source = account_state.source
        cash_source_text = f"{Path(args.positions_file).name} 的 account 行"
    else:
        effective_cash = 0.0
        cash_source = "default_zero"
        cash_source_text = "未提供 account 行现金，按 0 处理"
        print(
            "[warning] The current positions file has no account cash row and --cash was not provided; "
            "build this plan with 0 cash."
        )

    prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
        raw_df_dict=raw_df_dict,
        raw_cache_key=raw_cache_meta["cache_key"],
        cfg=cfg,
        enhanced_profile=effective_profile,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )
    print(
        f"[6/9] factor cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | "
        f"{prepared_cache_meta['cache_path']}"
    )
    df_dict = prepared_bundle["df_dict"]
    style_map = None
    industry_map = None
    if cfg.enable_style_cap and args.data_source == "tq":
        print("[7/9] Loading style mapping...")
        style_map = load_style_map_from_tq(list(df_dict["Close"].columns))
    if cfg.enable_industry_cap and args.data_source == "tq":
        industry_map = load_industry_map_from_tq(list(df_dict["Close"].columns))

    print("[8/9] Computing advanced scores...")
    if args.train_on_the_fly:
        factor_bundle, regime_state, score_none, score_v2, model_score, final_score_raw, final_score_filtered, target_weights, training_log = _build_scores_on_the_fly(
            cfg, ml_cfg, prepared_bundle, style_map, industry_map
        )
        model_info = {
            "mode": "实时训练",
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "horizons": ",".join(str(h) for h in ml_cfg.target_horizons),
            "model_family": str(ml_cfg.model_family),
            "lgbm_n_estimators": int(ml_cfg.lgbm_n_estimators) if str(ml_cfg.model_family) == "lgbm" else None,
            "state_ensemble_weights": ml_cfg.state_ensemble_weights or {},
            "state_horizon_profiles": ";".join(
                f"{state}=" + ",".join(f"{k}:{v:.2f}" for k, v in sorted(weights.items()))
                for state, weights in (ml_cfg.state_horizon_weights or {}).items()
            ),
            "history_window": history_window_to_dict(history_window),
            "enhanced_profile": str(ml_cfg.enhanced_profile),
        }
    else:
        factor_bundle, regime_state, score_none, score_v2, model_score, final_score_raw, final_score_filtered, target_weights, training_log = _build_scores_from_artifact(
            cfg, artifact, prepared_bundle, style_map, industry_map
        )
        if not training_log.empty:
            training_log.loc[:, "artifact_path"] = str(artifact_path)
        model_info = {
            "mode": "离线模型产物",
            "artifact_path": str(artifact_path),
            "trained_at": str(training_log.iloc[0].get("trained_at", "")) if not training_log.empty else "",
            "train_end": str(training_log.iloc[0].get("train_end", "")) if not training_log.empty else "",
            "horizons": str(training_log.iloc[0].get("horizons", "")) if not training_log.empty else "",
            "model_family": str(artifact.ml_config.get("model_family", "histgb")),
            "lgbm_n_estimators": (
                int(artifact.ml_config.get("lgbm_n_estimators", 260))
                if str(artifact.ml_config.get("model_family", "histgb")) == "lgbm"
                else None
            ),
            "horizon_weights": ",".join(
                f"{k}:{v:.2f}" for k, v in sorted((artifact.ml_config.get("target_horizon_weights") or {}).items())
            ),
            "state_ensemble_weights": artifact.ml_config.get("state_ensemble_weights") or {},
            "state_horizon_profiles": ";".join(
                f"{state}=" + ",".join(f"{k}:{v:.2f}" for k, v in sorted(weights.items()))
                for state, weights in (artifact.ml_config.get("state_horizon_weights") or {}).items()
            ),
            "history_window": history_window_to_dict(history_window),
            "enhanced_profile": str(artifact.ml_config.get("enhanced_profile", "up_low_breakout_v2")),
        }

    signal_date = final_score_raw.dropna(how="all").index.max()
    if pd.isna(signal_date):
        raise RuntimeError("No valid latest date found for trade plan generation.")

    freshness_info: dict[str, Any] = {}
    validation_summary: dict[str, Any] = {}
    if not args.train_on_the_fly:
        validation_summary = (
            artifact_meta.get("validation_summary")
            if isinstance(artifact_meta.get("validation_summary"), dict)
            else {}
        )
        freshness_info = _assess_model_freshness(
            artifact_meta=artifact_meta,
            latest_signal_date=pd.Timestamp(signal_date),
            trading_dates=factor_bundle["raw_inputs"]["Close"].index,
            warn_trading_days=args.stale_model_warn_trading_days,
            max_trading_days=args.stale_model_max_trading_days,
        )
        if not validation_summary:
            freshness_info.setdefault("warnings", []).append(
                "Model metadata does not yet include validation_summary; run update_model.py to build a fresh artifact."
            )
        if freshness_info.get("should_block") and args.allow_stale_model:
            freshness_info.setdefault("warnings", []).append(
                "stale model allowed by --allow-stale-model; proceed carefully."
            )
        for warning in freshness_info.get("warnings", []):
            print(f"[warning] {warning}")
        if freshness_info.get("should_block") and not args.allow_stale_model:
            raise RuntimeError(
                "Model freshness reached the blocking threshold."
                f" latest_data_date={freshness_info.get('artifact_latest_data_date', '')},"
                f" trading_day_lag={freshness_info.get('trading_day_lag')}."
                " Run daily_research/execution/update_model.py first, or pass --allow-stale-model explicitly."
            )
        model_info.update(
            {
                "latest_data_date": str(artifact_meta.get("latest_data_date", "")),
                "freshness_status": str(freshness_info.get("status_text", "")),
                "trading_day_lag": freshness_info.get("trading_day_lag"),
                "warnings": list(freshness_info.get("warnings", [])),
                "validation_summary": validation_summary,
            }
        )

    print("[9/9] Building the post-close plan and next-open execution suggestions...")
    target_weights, soft_state_scale, soft_state_meta, training_log = _apply_soft_state_overlay(
        args=args,
        regime_state=regime_state,
        target_weights=target_weights,
        model_info=model_info,
        training_log=training_log,
    )
    display_mode = _resolve_plan_display_mode(model_info)
    candidate_display = _resolve_candidate_display_config_v2(model_info)
    display_score_frame = final_score_filtered if display_mode == "research_candidate" else final_score_raw
    action_df, summary = _build_trade_plan(
        latest_date=signal_date,
        close_row=factor_bundle["raw_inputs"]["Close"].loc[signal_date],
        target_weight_row=target_weights.loc[signal_date],
        final_score_row=display_score_frame.loc[signal_date],
        score_none_row=score_none.loc[signal_date],
        score_v2_row=score_v2.loc[signal_date],
        model_score_row=model_score.loc[signal_date],
        positions_df=positions_df,
        cash=effective_cash,
        lot_size=args.lot_size,
        display_mode=display_mode,
        execution_proxy_source=str(candidate_display.get("execution_proxy_source", "final_score")),
        source_candidate_source=str(candidate_display.get("source_candidate_source", "model_score")),
    )
    summary["positions_source"] = str(account_state.path)
    summary["positions_source_mode"] = str(account_state.source)
    summary["cash_source"] = str(cash_source)
    summary["cash_source_text"] = str(cash_source_text)
    summary["positions_source_mtime"] = (
        datetime.fromtimestamp(Path(account_state.path).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        if account_state.path and Path(account_state.path).exists()
        else ""
    )
    summary["history_window"] = history_window_to_dict(history_window)
    summary["cache"] = {
        "raw": raw_cache_meta,
        "prepared": prepared_cache_meta,
    }
    summary.update(soft_state_meta)
    if not args.train_on_the_fly:
        summary["model_artifact"] = str(artifact_path)
        summary["model_latest_data_date"] = str(artifact_meta.get("latest_data_date", ""))
        summary["model_freshness"] = freshness_info
        summary["model_validation"] = validation_summary
    raw_watch_df: pd.DataFrame | None = None
    if (
        display_mode == "research_candidate"
        and external_watch_target_weight_path is not None
        and external_watch_score_path is not None
    ):
        market_index = pd.DatetimeIndex(factor_bundle["raw_inputs"]["Close"].index)
        allowed_columns = pd.Index(factor_bundle["raw_inputs"]["Close"].columns.astype(str))
        latest_market_date = pd.Timestamp(market_index.max())
        raw_watch_target_weight_row, raw_watch_target_signal_date = _load_external_panel_row(
            panel_csv=external_watch_target_weight_path,
            value_column=args.external_watch_target_weight_column,
            panel_label="external raw-model watch target-weight",
            value_name="target_weight",
            latest_market_date=latest_market_date,
            requested_date=pd.Timestamp(signal_date),
            market_index=market_index,
            allowed_columns=allowed_columns,
        )
        raw_watch_score_row, raw_watch_score_signal_date = _load_external_panel_row(
            panel_csv=external_watch_score_path,
            value_column=args.external_watch_score_column,
            panel_label="external raw-model watch score",
            value_name="score",
            latest_market_date=latest_market_date,
            requested_date=pd.Timestamp(signal_date),
            market_index=market_index,
            allowed_columns=allowed_columns,
        )
        raw_watch_df = _build_raw_model_watchlist(
            latest_date=pd.Timestamp(signal_date),
            model_score_row=raw_watch_score_row,
            target_weight_row=raw_watch_target_weight_row,
            top_n=15,
        )
        model_info.update(
            {
                "watchlist_mode": "raw_model_selection",
                "watch_candidate_target_weight_csv": str(external_watch_target_weight_path),
                "watch_candidate_score_csv": str(external_watch_score_path),
                "watch_target_weight_signal_date": str(pd.Timestamp(raw_watch_target_signal_date).date()),
                "watch_score_signal_date": str(pd.Timestamp(raw_watch_score_signal_date).date()),
            }
        )
    hold_df = _build_hold_table(
        latest_date=signal_date,
        close_row=factor_bundle["raw_inputs"]["Close"].loc[signal_date],
        target_weight_row=target_weights.loc[signal_date],
        final_score_row=display_score_frame.loc[signal_date],
        positions_df=positions_df,
    )
    watch_df = (
        raw_watch_df
        if raw_watch_df is not None
        else _build_watchlist(
            latest_date=signal_date,
            final_score_row=display_score_frame.loc[signal_date].dropna(),
            target_weight_row=target_weights.loc[signal_date],
            score_none_row=score_none.loc[signal_date],
            score_v2_row=score_v2.loc[signal_date],
            model_score_row=model_score.loc[signal_date],
            top_n=15,
            display_mode=display_mode,
            execution_proxy_source=str(candidate_display.get("execution_proxy_source", "final_score")),
            source_candidate_source=str(candidate_display.get("source_candidate_source", "model_score")),
        )
    )
    action_export_df = _export_plan_frame(action_df, model_info=model_info, frame_kind="action")
    watch_export_df = _export_plan_frame(watch_df, model_info=model_info, frame_kind="watch")

    print("Writing output files...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.experiment_tag.strip() or signal_date.strftime("%Y%m%d")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    txt_path = run_dir / "daily_trade_plan.txt"
    latest_txt_path = output_dir / "latest_trade_plan.txt"
    _write_trade_plan_txt(
        txt_path,
        summary=summary,
        regime_state_row=regime_state.loc[signal_date],
        action_df=action_df,
        hold_df=hold_df,
        watch_df=watch_df,
        model_info=model_info,
    )
    _write_trade_plan_txt(
        latest_txt_path,
        summary=summary,
        regime_state_row=regime_state.loc[signal_date],
        action_df=action_df,
        hold_df=hold_df,
        watch_df=watch_df,
        model_info=model_info,
    )

    action_export_df.to_csv(run_dir / "actions_today.csv", index=False, encoding="utf-8-sig")
    hold_df.to_csv(run_dir / "holdings_snapshot.csv", index=False, encoding="utf-8-sig")
    watch_export_df.to_csv(run_dir / "watchlist.csv", index=False, encoding="utf-8-sig")
    training_log.to_csv(run_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": soft_state_scale.index, "soft_state_scale": soft_state_scale.values}).to_csv(
        run_dir / "soft_state_scale.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(run_dir / "plan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"output_dir={run_dir}")
    print(f"latest_plan_file={latest_txt_path}")
    if not action_df.empty:
        print(_terminal_action_preview(action_df).to_string(index=False))
    else:
        print("No rebalance action for today.")


def main_with_progress():
    args = parse_args()
    validate_state_profile_selector(args.enhanced_profile, args.regime_state_selector)
    cfg = ResearchConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        universe_scope=args.universe_scope,
        benchmark=args.benchmark,
        execution_mode="next_open",
        holding_count=args.holding_count,
        weighting_method="score",
        rebalance_freq=args.rebalance_freq,
        score_threshold=args.score_threshold,
        max_weight=args.max_weight,
        min_adv20=args.min_adv20,
        min_price=args.min_price,
        max_price=args.max_price,
        enable_market_regime_filter=not args.no_market_regime_filter,
        regime_ma_window=args.regime_ma_window,
        regime_vol_window=args.regime_vol_window,
        regime_max_annual_vol=args.regime_max_annual_vol,
        regime_trend_flat_band=args.regime_trend_flat_band,
        regime_vol_transition_band=args.regime_vol_transition_band,
        regime_state_selector=args.regime_state_selector,
        regime_allowed_quadrants=parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )
    stocks = parse_stock_list(args.stocks)
    file_stocks = load_stock_list_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if stocks:
        cfg.universe = stocks

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=parse_horizon_weights(args.ml_horizon_weights),
        state_horizon_weights=parse_state_horizon_profiles(args.ml_state_horizon_profiles),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=args.ml_retrain_every_days,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        lgbm_n_estimators=args.lgbm_n_estimators,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        state_ensemble_weights=parse_state_ensemble_weights(args.ensemble_state_weights),
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    artifact = None
    artifact_meta: dict[str, Any] = {}
    artifact_path = Path(args.model_artifact)
    external_score_path = Path(args.external_score_csv).expanduser() if str(args.external_score_csv).strip() else None
    external_target_weight_path = (
        Path(args.external_target_weight_csv).expanduser() if str(args.external_target_weight_csv).strip() else None
    )
    external_watch_target_weight_path = (
        Path(args.external_watch_target_weight_csv).expanduser()
        if str(args.external_watch_target_weight_csv).strip()
        else None
    )
    external_watch_score_path = (
        Path(args.external_watch_score_csv).expanduser() if str(args.external_watch_score_csv).strip() else None
    )
    effective_profile = args.enhanced_profile
    effective_ml_cfg = ml_cfg

    with StageProgress(total=9, label="Trade plan") as progress:
        with progress.stage("Prepare model and research universe", f"source={args.data_source}"):
            if external_score_path is not None or external_target_weight_path is not None:
                if args.train_on_the_fly:
                    raise ValueError("--external-score-csv / --external-target-weight-csv cannot be combined with --train-on-the-fly.")
                if external_score_path is not None and not external_score_path.exists():
                    raise FileNotFoundError(f"External score CSV not found: {external_score_path}")
                if external_target_weight_path is not None and not external_target_weight_path.exists():
                    raise FileNotFoundError(f"External target-weight CSV not found: {external_target_weight_path}")
                if external_watch_target_weight_path is not None and not external_watch_target_weight_path.exists():
                    raise FileNotFoundError(
                        f"External raw-model watch target-weight CSV not found: {external_watch_target_weight_path}"
                    )
                if external_watch_score_path is not None and not external_watch_score_path.exists():
                    raise FileNotFoundError(
                        f"External raw-model watch score CSV not found: {external_watch_score_path}"
                    )
            elif not args.train_on_the_fly:
                if not artifact_path.exists():
                    raise FileNotFoundError(
                        f"Model artifact not found: {artifact_path}. Run daily_research/execution/update_model.py first."
                    )
                artifact = load_ml_artifact(artifact_path)
                artifact_meta = _load_artifact_meta(artifact_path)
                effective_ml_cfg = MLAplhaConfig(**artifact.ml_config)
                effective_profile = str(
                    getattr(effective_ml_cfg, "enhanced_profile", "up_low_breakout_v2") or "up_low_breakout_v2"
                )
                validate_state_profile_selector(effective_profile, cfg.regime_state_selector)
            if args.data_source == "tq":
                if not cfg.universe and cfg.universe_scope == "all_a":
                    cfg.universe = load_universe_from_tq(cfg.universe_scope)
                elif not cfg.universe:
                    raise ValueError("TQ mode without --stocks currently requires --universe-scope all_a.")
            elif not args.csv_folder:
                raise ValueError("CSV mode requires --csv-folder.")

        with progress.stage("Resolve inference history window", args.start_date):
            history_window = resolve_history_window(
                cfg=cfg,
                ml_cfg=effective_ml_cfg,
                requested_start_date=args.start_date,
                end_date=args.end_date,
                mode="infer",
                auto_trim_history=not args.no_auto_trim_history,
            )
            progress.log(
                f"infer history: {history_window.effective_start_date} -> "
                f"{history_window.end_date or 'latest'} | required_trading_days={history_window.required_trading_days}"
            )

        with progress.stage("Load market data", f"stocks={len(cfg.universe)} benchmark={cfg.benchmark}"):
            raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
                data_source=args.data_source,
                csv_folder=args.csv_folder,
                universe=cfg.universe,
                benchmark=cfg.benchmark,
                history_window=history_window,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                progress_desc="Load trade-plan market data",
                progress_position=1,
            )
            progress.log(
                f"raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | {raw_cache_meta['cache_path']}"
            )

        with progress.stage("Load account snapshot", Path(args.positions_file).name):
            account_state = _load_account_state(args.positions_file)
            positions_df = account_state.positions_df
            if args.cash is not None:
                effective_cash = float(args.cash)
                cash_source = "cli_override"
                cash_source_text = f"--cash override ({args.positions_file})"
            elif account_state.cash is not None:
                effective_cash = float(account_state.cash)
                cash_source = account_state.source
                cash_source_text = f"{Path(args.positions_file).name} account row"
            else:
                effective_cash = 0.0
                cash_source = "default_zero"
                cash_source_text = "no account cash provided, fallback to 0"
                progress.log(
                    "[warning] positions file has no account cash row and --cash was not provided; fallback to 0."
                )

        with progress.stage("Build prepared research inputs", effective_profile):
            prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
                raw_df_dict=raw_df_dict,
                raw_cache_key=raw_cache_meta["cache_key"],
                cfg=cfg,
                enhanced_profile=effective_profile,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
            )
            progress.log(
                f"factor cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | "
                f"{prepared_cache_meta['cache_path']}"
            )
            df_dict = prepared_bundle["df_dict"]

        with progress.stage("Load constraint mappings", "industry/style"):
            style_map = None
            industry_map = None
            if cfg.enable_style_cap and args.data_source == "tq":
                style_map = load_style_map_from_tq(list(df_dict["Close"].columns))
            if cfg.enable_industry_cap and args.data_source == "tq":
                industry_map = load_industry_map_from_tq(list(df_dict["Close"].columns))

        with progress.stage("Compute portfolio scores", "artifact/live/external"):
            if external_target_weight_path is not None:
                (
                    factor_bundle,
                    regime_state,
                    score_none,
                    score_v2,
                    model_score,
                    final_score_raw,
                    final_score_filtered,
                    target_weights,
                    training_log,
                ) = _build_scores_from_external_target_weight_csv(
                    cfg,
                    prepared_bundle,
                    external_target_weight_csv=external_target_weight_path,
                    external_target_weight_column=args.external_target_weight_column,
                    candidate_label=args.candidate_label,
                    rebalance_offset=args.rebalance_offset,
                    rebalance_offset_mode=args.rebalance_offset_mode,
                    rebalance_anchor_date=args.rebalance_anchor_date,
                    target_weight_top_k=args.target_weight_top_k,
                    target_weight_min_weight=args.target_weight_min_weight,
                    target_weight_power=args.target_weight_power,
                    target_weight_full_invest=bool(args.target_weight_full_invest),
                    external_score_csv=external_score_path,
                    external_score_column=args.external_score_column,
                )
                candidate_label = str(training_log.iloc[0].get("candidate_label", "")) if not training_log.empty else ""
                loaded_rows = int(training_log.iloc[0].get("loaded_rows", 0)) if not training_log.empty else 0
                usable_rows = int(training_log.iloc[0].get("usable_rows", 0)) if not training_log.empty else 0
                dropped_rows = int(training_log.iloc[0].get("dropped_rows", 0)) if not training_log.empty else 0
                model_info = {
                    "mode": "research_candidate_target_weight_csv",
                    "candidate_label": candidate_label,
                    "candidate_target_weight_csv": str(external_target_weight_path),
                    "candidate_score_csv": str(external_score_path) if external_score_path is not None else "",
                    "candidate_total_rows": loaded_rows,
                    "candidate_usable_rows": usable_rows,
                    "candidate_dropped_rows": dropped_rows,
                    "rebalance_freq": str(cfg.rebalance_freq),
                    "rebalance_offset_mode": str(training_log.iloc[0].get("rebalance_offset_mode", args.rebalance_offset_mode)) if not training_log.empty else str(args.rebalance_offset_mode),
                    "rebalance_offset": training_log.iloc[0].get("rebalance_offset") if not training_log.empty else args.rebalance_offset,
                    "rebalance_sleeve_count": int(training_log.iloc[0].get("rebalance_sleeve_count", 1)) if not training_log.empty else 1,
                    "rebalance_anchor_date": str(training_log.iloc[0].get("rebalance_anchor_date", args.rebalance_anchor_date or "")) if not training_log.empty else str(args.rebalance_anchor_date or ""),
                    "target_weight_top_k": int(training_log.iloc[0].get("target_weight_top_k", args.target_weight_top_k)) if not training_log.empty else int(args.target_weight_top_k),
                    "target_weight_min_weight": float(training_log.iloc[0].get("target_weight_min_weight", args.target_weight_min_weight)) if not training_log.empty else float(args.target_weight_min_weight),
                    "target_weight_power": float(training_log.iloc[0].get("target_weight_power", args.target_weight_power)) if not training_log.empty else float(args.target_weight_power),
                    "target_weight_full_invest": bool(training_log.iloc[0].get("target_weight_full_invest", args.target_weight_full_invest)) if not training_log.empty else bool(args.target_weight_full_invest),
                    "source_signal_date": str(training_log.iloc[0].get("source_signal_date", "")) if not training_log.empty else "",
                    "execution_signal_date": str(training_log.iloc[0].get("signal_date", "")) if not training_log.empty else "",
                    "score_context_status": str(training_log.iloc[0].get("score_context_status", "")) if not training_log.empty else "",
                    "history_window": history_window_to_dict(history_window),
                    "enhanced_profile": "external_target_weight_candidate",
                    "target_weight_semantics": str(args.external_target_weight_semantics or ""),
                    "target_weight_cap_mode": str(args.external_target_weight_cap_mode or ""),
                    "target_weight_cap_note": str(args.external_target_weight_cap_note or ""),
                    "score_panel_role": str(args.external_score_panel_role or ""),
                    "score_reference_metadata_json": str(args.external_score_reference_metadata_json or ""),
                    "transaction_cost_bps": args.transaction_cost_bps,
                    "slippage_bps": args.slippage_bps,
                    "sell_tax_bps": args.sell_tax_bps,
                }
                model_info = _merge_score_reference_metadata(model_info)
                warnings: list[str] = []
                if dropped_rows > 0:
                    warnings.append(
                        f"external target-weight rows outside the current execution universe were dropped: {dropped_rows}"
                    )
                score_context_warning = (
                    str(training_log.iloc[0].get("score_context_warning", "")) if not training_log.empty else ""
                ).strip()
                if score_context_warning:
                    warnings.append(f"external score context fallback: {score_context_warning}")
                if warnings:
                    model_info["warnings"] = warnings
                freshness_info: dict[str, Any] = {}
                validation_summary: dict[str, Any] = {}
            elif external_score_path is not None:
                (
                    factor_bundle,
                    regime_state,
                    score_none,
                    score_v2,
                    model_score,
                    final_score_raw,
                    final_score_filtered,
                    target_weights,
                    training_log,
                ) = _build_scores_from_external_csv(
                    cfg,
                    prepared_bundle,
                    style_map,
                    industry_map,
                    external_score_csv=external_score_path,
                    external_score_column=args.external_score_column,
                    candidate_label=args.candidate_label,
                )
                candidate_label = str(training_log.iloc[0].get("candidate_label", "")) if not training_log.empty else ""
                loaded_rows = int(training_log.iloc[0].get("loaded_rows", 0)) if not training_log.empty else 0
                usable_rows = int(training_log.iloc[0].get("usable_rows", 0)) if not training_log.empty else 0
                dropped_rows = int(training_log.iloc[0].get("dropped_rows", 0)) if not training_log.empty else 0
                model_info = {
                    "mode": "research_candidate_csv",
                    "candidate_label": candidate_label,
                    "candidate_score_csv": str(external_score_path),
                    "candidate_total_rows": loaded_rows,
                    "candidate_usable_rows": usable_rows,
                    "candidate_dropped_rows": dropped_rows,
                    "source_signal_date": str(training_log.iloc[0].get("signal_date", "")) if not training_log.empty else "",
                    "execution_signal_date": str(training_log.iloc[0].get("signal_date", "")) if not training_log.empty else "",
                    "score_context_status": "external_score",
                    "history_window": history_window_to_dict(history_window),
                    "enhanced_profile": "external_score_candidate",
                }
                if dropped_rows > 0:
                    model_info["warnings"] = [
                        f"external score rows outside the current execution universe were dropped: {dropped_rows}"
                    ]
                freshness_info: dict[str, Any] = {}
                validation_summary: dict[str, Any] = {}
            elif args.train_on_the_fly:
                (
                    factor_bundle,
                    regime_state,
                    score_none,
                    score_v2,
                    model_score,
                    final_score_raw,
                    final_score_filtered,
                    target_weights,
                    training_log,
                ) = _build_scores_on_the_fly(cfg, ml_cfg, prepared_bundle, style_map, industry_map)
                model_info = {
                    "mode": "实时训练",
                    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "horizons": ",".join(str(h) for h in ml_cfg.target_horizons),
                    "model_family": str(ml_cfg.model_family),
                    "lgbm_n_estimators": int(ml_cfg.lgbm_n_estimators) if str(ml_cfg.model_family) == "lgbm" else None,
                    "state_ensemble_weights": ml_cfg.state_ensemble_weights or {},
                    "state_horizon_profiles": ";".join(
                        f"{state}=" + ",".join(f"{k}:{v:.2f}" for k, v in sorted(weights.items()))
                        for state, weights in (ml_cfg.state_horizon_weights or {}).items()
                    ),
                    "history_window": history_window_to_dict(history_window),
                    "enhanced_profile": str(ml_cfg.enhanced_profile),
                }
                freshness_info: dict[str, Any] = {}
                validation_summary: dict[str, Any] = {}
            else:
                (
                    factor_bundle,
                    regime_state,
                    score_none,
                    score_v2,
                    model_score,
                    final_score_raw,
                    final_score_filtered,
                    target_weights,
                    training_log,
                ) = _build_scores_from_artifact(cfg, artifact, prepared_bundle, style_map, industry_map)
                if not training_log.empty:
                    training_log.loc[:, "artifact_path"] = str(artifact_path)
                model_info = {
                    "mode": "离线模型产物",
                    "artifact_path": str(artifact_path),
                    "trained_at": str(training_log.iloc[0].get("trained_at", "")) if not training_log.empty else "",
                    "train_end": str(training_log.iloc[0].get("train_end", "")) if not training_log.empty else "",
                    "horizons": str(training_log.iloc[0].get("horizons", "")) if not training_log.empty else "",
                    "model_family": str(artifact.ml_config.get("model_family", "histgb")),
                    "lgbm_n_estimators": (
                        int(artifact.ml_config.get("lgbm_n_estimators", 260))
                        if str(artifact.ml_config.get("model_family", "histgb")) == "lgbm"
                        else None
                    ),
                    "horizon_weights": ",".join(
                        f"{k}:{v:.2f}" for k, v in sorted((artifact.ml_config.get("target_horizon_weights") or {}).items())
                    ),
                    "state_ensemble_weights": artifact.ml_config.get("state_ensemble_weights") or {},
                    "state_horizon_profiles": ";".join(
                        f"{state}=" + ",".join(f"{k}:{v:.2f}" for k, v in sorted(weights.items()))
                        for state, weights in (artifact.ml_config.get("state_horizon_weights") or {}).items()
                    ),
                    "history_window": history_window_to_dict(history_window),
                    "enhanced_profile": str(artifact.ml_config.get("enhanced_profile", "up_low_breakout_v2")),
                }

                signal_date = final_score_raw.dropna(how="all").index.max()
                if pd.isna(signal_date):
                    raise RuntimeError("No valid latest date found for trade plan generation.")
                validation_summary = (
                    artifact_meta.get("validation_summary")
                    if isinstance(artifact_meta.get("validation_summary"), dict)
                    else {}
                )
                freshness_info = _assess_model_freshness(
                    artifact_meta=artifact_meta,
                    latest_signal_date=pd.Timestamp(signal_date),
                    trading_dates=factor_bundle["raw_inputs"]["Close"].index,
                    warn_trading_days=args.stale_model_warn_trading_days,
                    max_trading_days=args.stale_model_max_trading_days,
                )
                if not validation_summary:
                    freshness_info.setdefault("warnings", []).append(
                        "artifact metadata has no validation_summary; rerun update_model.py for a fresh artifact."
                    )
                if freshness_info.get("should_block") and args.allow_stale_model:
                    freshness_info.setdefault("warnings", []).append("stale artifact allowed by --allow-stale-model")
                for warning in freshness_info.get("warnings", []):
                    progress.log(f"[warning] {warning}")
                if freshness_info.get("should_block") and not args.allow_stale_model:
                    raise RuntimeError(
                        "Model artifact reached stale blocking threshold. Run daily_research/execution/update_model.py "
                        "or pass --allow-stale-model explicitly."
                    )
                model_info.update(
                    {
                        "latest_data_date": str(artifact_meta.get("latest_data_date", "")),
                        "latest_data_label": "模型最新数据日",
                        "freshness_status": str(freshness_info.get("status_text", "")),
                        "freshness_label": "模型新鲜度",
                        "trading_day_lag": freshness_info.get("trading_day_lag"),
                        "warnings": list(freshness_info.get("warnings", [])),
                        "validation_summary": validation_summary,
                    }
                )

        with progress.stage("Build trade-plan data", cfg.rebalance_freq):
            signal_date = final_score_raw.dropna(how="all").index.max()
            if pd.isna(signal_date):
                raise RuntimeError("No valid latest date found for trade plan generation.")
            model_info.setdefault("market_regime_filter_enabled", bool(cfg.enable_market_regime_filter))
            model_info.setdefault("execution_signal_date", str(pd.Timestamp(signal_date).date()))
            display_mode = _resolve_plan_display_mode(model_info)
            if external_score_path is not None or external_target_weight_path is not None:
                source_signal_date = _safe_timestamp(model_info.get("source_signal_date")) or pd.Timestamp(signal_date)
                freshness_info = _assess_external_signal_freshness(
                    source_signal_date=source_signal_date,
                    trading_dates=factor_bundle["raw_inputs"]["Close"].index,
                    warn_trading_days=args.stale_model_warn_trading_days,
                    max_trading_days=args.stale_model_max_trading_days,
                )
                warnings = list(model_info.get("warnings", []))
                warnings.extend(freshness_info.get("warnings", []))
                if freshness_info.get("should_block") and args.allow_stale_model:
                    warnings.append("stale external candidate allowed by --allow-stale-model")
                for warning in freshness_info.get("warnings", []):
                    progress.log(f"[warning] {warning}")
                if freshness_info.get("should_block") and not args.allow_stale_model:
                    raise RuntimeError(
                        "External candidate panel reached stale blocking threshold. "
                        "Refresh the active manifest-driven research candidate pipeline, "
                        "or pass --allow-stale-model explicitly. "
                        "Use --legacy-ml only when you intentionally want the old legacy chain."
                    )
                model_info.update(
                    {
                        "latest_data_date": str(pd.Timestamp(source_signal_date).date()),
                        "latest_data_label": "候选源信号日",
                        "freshness_status": str(freshness_info.get("status_text", "")),
                        "freshness_label": "候选信号新鲜度",
                        "latest_completed_trading_date": freshness_info.get("latest_completed_trading_date"),
                        "trading_day_lag": freshness_info.get("trading_day_lag"),
                        "warnings": warnings,
                    }
                )
                external_model_manifest = (
                    Path(args.external_model_manifest).expanduser()
                    if str(args.external_model_manifest).strip()
                    else None
                )
                if external_model_manifest is not None:
                    retrain_info = _assess_external_model_retrain_freshness(
                        manifest_path=external_model_manifest,
                        latest_signal_date=pd.Timestamp(signal_date),
                        trading_dates=factor_bundle["raw_inputs"]["Close"].index,
                        warn_trading_days=args.external_model_warn_trading_days,
                        max_trading_days=args.external_model_max_trading_days,
                    )
                    warnings = list(model_info.get("warnings", []))
                    warnings.extend(retrain_info.get("warnings", []))
                    if retrain_info.get("should_block") and args.allow_stale_model:
                        warnings.append(
                            "stale production retrain cadence allowed by --allow-stale-model; proceed carefully."
                        )
                    for warning in retrain_info.get("warnings", []):
                        progress.log(f"[warning] {warning}")
                    if retrain_info.get("should_block") and not args.allow_stale_model:
                        raise RuntimeError(
                            "External candidate underlying production model reached retrain blocking threshold. "
                            "Run daily_research/execution/update_default_candidate_production.py or pass --allow-stale-model explicitly."
                        )
                    model_info.update(
                        {
                            "warnings": warnings,
                            "production_model_train_end_date": str(retrain_info.get("train_end_date", "")),
                            "production_model_launch_cutoff_date": str(retrain_info.get("launch_cutoff_date", "")),
                            "production_model_retrain_status": str(retrain_info.get("status_text", "")),
                            "production_model_retrain_label": "底层模型重训时效",
                            "production_model_retrain_trading_day_lag": retrain_info.get("trading_day_lag"),
                            "production_model_retrain_warn_trading_days": retrain_info.get("warn_trading_days"),
                            "production_model_retrain_max_trading_days": retrain_info.get("max_trading_days"),
                            "production_model_retrain_manifest": str(retrain_info.get("manifest_path", "")),
                            "production_model_retrain_policy": str(retrain_info.get("preferred_label", "")),
                            "production_model_retrain_window": str(retrain_info.get("comparison_window", "")),
                            "production_model_retrain_leaderboard_csv": str(retrain_info.get("leaderboard_csv", "")),
                            "production_model_active_run_dir": str(retrain_info.get("active_production_run_dir", "")),
                            "target_weight_semantics": str(
                                model_info.get("target_weight_semantics", "") or retrain_info.get("target_weight_semantics", "")
                            ),
                            "target_weight_cap_mode": str(
                                model_info.get("target_weight_cap_mode", "") or retrain_info.get("target_weight_cap_mode", "")
                            ),
                            "target_weight_cap_note": str(
                                model_info.get("target_weight_cap_note", "") or retrain_info.get("target_weight_cap_note", "")
                            ),
                            "score_panel_role": str(
                                model_info.get("score_panel_role", "") or retrain_info.get("score_panel_role", "")
                            ),
                            "score_reference_metadata_json": str(
                                model_info.get("score_reference_metadata_json", "")
                                or retrain_info.get("score_reference_metadata_json", "")
                            ),
                            "transaction_cost_bps": (
                                retrain_info.get("transaction_cost_bps")
                                if retrain_info.get("transaction_cost_bps") is not None
                                else model_info.get("transaction_cost_bps")
                            ),
                            "slippage_bps": (
                                retrain_info.get("slippage_bps")
                                if retrain_info.get("slippage_bps") is not None
                                else model_info.get("slippage_bps")
                            ),
                            "sell_tax_bps": (
                                retrain_info.get("sell_tax_bps")
                                if retrain_info.get("sell_tax_bps") is not None
                                else model_info.get("sell_tax_bps")
                            ),
                        }
                    )
            target_weights, soft_state_scale, soft_state_meta, training_log = _apply_soft_state_overlay(
                args=args,
                regime_state=regime_state,
                target_weights=target_weights,
                model_info=model_info,
                training_log=training_log,
            )
            candidate_display = _resolve_candidate_display_config_v2(model_info)
            display_score_frame = final_score_filtered if display_mode == "research_candidate" else final_score_raw
            action_df, summary = _build_trade_plan(
                latest_date=signal_date,
                close_row=factor_bundle["raw_inputs"]["Close"].loc[signal_date],
                target_weight_row=target_weights.loc[signal_date],
                final_score_row=display_score_frame.loc[signal_date],
                score_none_row=score_none.loc[signal_date],
                score_v2_row=score_v2.loc[signal_date],
                model_score_row=model_score.loc[signal_date],
                positions_df=positions_df,
                cash=effective_cash,
                lot_size=args.lot_size,
                display_mode=display_mode,
                execution_proxy_source=str(candidate_display.get("execution_proxy_source", "final_score")),
                source_candidate_source=str(candidate_display.get("source_candidate_source", "model_score")),
            )
            summary["positions_source"] = str(account_state.path)
            summary["positions_source_mode"] = str(account_state.source)
            summary["cash_source"] = str(cash_source)
            summary["cash_source_text"] = str(cash_source_text)
            summary["positions_source_mtime"] = (
                datetime.fromtimestamp(Path(account_state.path).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                if account_state.path and Path(account_state.path).exists()
                else ""
            )
            summary["history_window"] = history_window_to_dict(history_window)
            summary["cache"] = {
                "raw": raw_cache_meta,
                "prepared": prepared_cache_meta,
            }
            summary.update(soft_state_meta)
            if external_target_weight_path is not None:
                summary["candidate_target_weight_csv"] = str(external_target_weight_path)
                if external_score_path is not None:
                    summary["candidate_score_csv"] = str(external_score_path)
                if external_watch_target_weight_path is not None:
                    summary["watch_candidate_target_weight_csv"] = str(external_watch_target_weight_path)
                if external_watch_score_path is not None:
                    summary["watch_candidate_score_csv"] = str(external_watch_score_path)
                summary["candidate_label"] = str(model_info.get("candidate_label", ""))
                summary["target_weight_semantics"] = str(model_info.get("target_weight_semantics", ""))
                summary["target_weight_cap_mode"] = str(model_info.get("target_weight_cap_mode", ""))
                summary["target_weight_cap_note"] = str(model_info.get("target_weight_cap_note", ""))
                summary["score_panel_role"] = str(model_info.get("score_panel_role", ""))
                summary["score_reference_metadata_json"] = str(model_info.get("score_reference_metadata_json", ""))
                summary["effective_live_target_weight_mode"] = str(model_info.get("effective_live_target_weight_mode", ""))
                summary["effective_live_execution_profile"] = str(model_info.get("effective_live_execution_profile", ""))
                summary["effective_live_trigger_key"] = str(model_info.get("effective_live_trigger_key", ""))
                summary["effective_live_score_note"] = str(model_info.get("effective_live_score_note", ""))
                summary["effective_live_weight_generation_note"] = str(model_info.get("effective_live_weight_generation_note", ""))
                summary["transaction_cost_bps"] = model_info.get("transaction_cost_bps")
                summary["slippage_bps"] = model_info.get("slippage_bps")
                summary["sell_tax_bps"] = model_info.get("sell_tax_bps")
                summary["candidate_total_rows"] = int(model_info.get("candidate_total_rows", 0) or 0)
                summary["candidate_usable_rows"] = int(model_info.get("candidate_usable_rows", 0) or 0)
                summary["candidate_dropped_rows"] = int(model_info.get("candidate_dropped_rows", 0) or 0)
                if model_info.get("production_model_train_end_date"):
                    summary["production_model_train_end_date"] = str(model_info.get("production_model_train_end_date", ""))
                    summary["production_model_launch_cutoff_date"] = str(model_info.get("production_model_launch_cutoff_date", ""))
                    summary["production_model_retrain_status"] = str(model_info.get("production_model_retrain_status", ""))
                    summary["production_model_retrain_trading_day_lag"] = model_info.get("production_model_retrain_trading_day_lag")
                    summary["production_model_retrain_warn_trading_days"] = model_info.get("production_model_retrain_warn_trading_days")
                    summary["production_model_retrain_max_trading_days"] = model_info.get("production_model_retrain_max_trading_days")
                    summary["production_model_retrain_manifest"] = str(model_info.get("production_model_retrain_manifest", ""))
                    summary["production_model_retrain_policy"] = str(model_info.get("production_model_retrain_policy", ""))
                    summary["production_model_retrain_window"] = str(model_info.get("production_model_retrain_window", ""))
            elif external_score_path is not None:
                summary["candidate_score_csv"] = str(external_score_path)
                summary["candidate_label"] = str(model_info.get("candidate_label", ""))
                summary["candidate_total_rows"] = int(model_info.get("candidate_total_rows", 0) or 0)
                summary["candidate_usable_rows"] = int(model_info.get("candidate_usable_rows", 0) or 0)
                summary["candidate_dropped_rows"] = int(model_info.get("candidate_dropped_rows", 0) or 0)
            elif not args.train_on_the_fly:
                summary["model_artifact"] = str(artifact_path)
                summary["model_latest_data_date"] = str(artifact_meta.get("latest_data_date", ""))
                summary["model_freshness"] = freshness_info
                summary["model_validation"] = validation_summary
            raw_watch_df: pd.DataFrame | None = None
            if (
                display_mode == "research_candidate"
                and external_watch_target_weight_path is not None
                and external_watch_score_path is not None
            ):
                market_index = pd.DatetimeIndex(factor_bundle["raw_inputs"]["Close"].index)
                allowed_columns = pd.Index(factor_bundle["raw_inputs"]["Close"].columns.astype(str))
                latest_market_date = pd.Timestamp(market_index.max())
                raw_watch_target_weight_row, raw_watch_target_signal_date = _load_external_panel_row(
                    panel_csv=external_watch_target_weight_path,
                    value_column=args.external_watch_target_weight_column,
                    panel_label="external raw-model watch target-weight",
                    value_name="target_weight",
                    latest_market_date=latest_market_date,
                    requested_date=pd.Timestamp(signal_date),
                    market_index=market_index,
                    allowed_columns=allowed_columns,
                )
                raw_watch_score_row, raw_watch_score_signal_date = _load_external_panel_row(
                    panel_csv=external_watch_score_path,
                    value_column=args.external_watch_score_column,
                    panel_label="external raw-model watch score",
                    value_name="score",
                    latest_market_date=latest_market_date,
                    requested_date=pd.Timestamp(signal_date),
                    market_index=market_index,
                    allowed_columns=allowed_columns,
                )
                raw_watch_df = _build_raw_model_watchlist(
                    latest_date=pd.Timestamp(signal_date),
                    model_score_row=raw_watch_score_row,
                    target_weight_row=raw_watch_target_weight_row,
                    top_n=15,
                )
                model_info.update(
                    {
                        "watchlist_mode": "raw_model_selection",
                        "watch_candidate_target_weight_csv": str(external_watch_target_weight_path),
                        "watch_candidate_score_csv": str(external_watch_score_path),
                        "watch_target_weight_signal_date": str(pd.Timestamp(raw_watch_target_signal_date).date()),
                        "watch_score_signal_date": str(pd.Timestamp(raw_watch_score_signal_date).date()),
                    }
                )
            hold_df = _build_hold_table(
                latest_date=signal_date,
                close_row=factor_bundle["raw_inputs"]["Close"].loc[signal_date],
                target_weight_row=target_weights.loc[signal_date],
                final_score_row=display_score_frame.loc[signal_date],
                positions_df=positions_df,
            )
            watch_df = (
                raw_watch_df
                if raw_watch_df is not None
                else _build_watchlist(
                    latest_date=signal_date,
                    final_score_row=display_score_frame.loc[signal_date].dropna(),
                    target_weight_row=target_weights.loc[signal_date],
                    score_none_row=score_none.loc[signal_date],
                    score_v2_row=score_v2.loc[signal_date],
                    model_score_row=model_score.loc[signal_date],
                    top_n=15,
                    display_mode=display_mode,
                    execution_proxy_source=str(candidate_display.get("execution_proxy_source", "final_score")),
                    source_candidate_source=str(candidate_display.get("source_candidate_source", "model_score")),
                )
            )
            action_export_df = _export_plan_frame(action_df, model_info=model_info, frame_kind="action")
            watch_export_df = _export_plan_frame(watch_df, model_info=model_info, frame_kind="watch")

        with progress.stage("Write execution files", "txt/csv/json"):
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            run_name = args.experiment_tag.strip() or signal_date.strftime("%Y%m%d")
            run_dir = output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            txt_path = run_dir / "daily_trade_plan.txt"
            latest_txt_path = output_dir / "latest_trade_plan.txt"
            _write_trade_plan_txt(
                txt_path,
                summary=summary,
                regime_state_row=regime_state.loc[signal_date],
                action_df=action_df,
                hold_df=hold_df,
                watch_df=watch_df,
                model_info=model_info,
            )
            _write_trade_plan_txt(
                latest_txt_path,
                summary=summary,
                regime_state_row=regime_state.loc[signal_date],
                action_df=action_df,
                hold_df=hold_df,
                watch_df=watch_df,
                model_info=model_info,
            )

            action_export_df.to_csv(run_dir / "actions_today.csv", index=False, encoding="utf-8-sig")
            hold_df.to_csv(run_dir / "holdings_snapshot.csv", index=False, encoding="utf-8-sig")
            watch_export_df.to_csv(run_dir / "watchlist.csv", index=False, encoding="utf-8-sig")
            training_log.to_csv(run_dir / "training_log.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"date": soft_state_scale.index, "soft_state_scale": soft_state_scale.values}).to_csv(
                run_dir / "soft_state_scale.csv",
                index=False,
                encoding="utf-8-sig",
            )
            with open(run_dir / "plan_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"output_dir={run_dir}")
    print(f"latest_plan_file={latest_txt_path}")
    if not action_df.empty:
        print(_terminal_action_preview(action_df).to_string(index=False))
    else:
        print("No rebalance action for today.")


main = main_with_progress


if __name__ == "__main__":
    main()
