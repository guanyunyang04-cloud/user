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
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import (
    get_next_trading_date,
    load_industry_map_from_tq,
    load_style_map_from_tq,
    load_universe_from_tq,
)
from daily_research.baseline.ml_alpha import (
    MLAplhaConfig,
    blend_scores,
    load_ml_artifact,
    predict_ml_scores_for_date_bundle,
    resolve_horizon_weights,
    rolling_ml_scores_multi,
)
from daily_research.baseline.portfolio import build_target_weights
from daily_research.baseline.regime import apply_market_regime_filter


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
    parser.add_argument("--rebalance-freq", default="5d")
    parser.add_argument("--positions-file", default="daily_research/execution/current_positions.csv")
    parser.add_argument("--cash", type=float, default=None, help="Optional cash override. If omitted, will try to read from current_positions.csv account snapshot row.")
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--output-dir", default="daily_research/execution/output")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--model-artifact", default="daily_research/execution/models/latest_ml_model.joblib")
    parser.add_argument("--train-on-the-fly", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stale-model-warn-trading-days", type=int, default=1)
    parser.add_argument("--stale-model-max-trading-days", type=int, default=3)
    parser.add_argument("--allow-stale-model", action="store_true")

    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.0)

    parser.add_argument("--no-market-regime-filter", action="store_true")
    parser.add_argument("--regime-ma-window", type=int, default=60)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")

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
    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70, help=argparse.SUPPRESS)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20, help=argparse.SUPPRESS)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10, help=argparse.SUPPRESS)
    parser.add_argument("--ensemble-state-weights", default="", help=argparse.SUPPRESS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-auto-trim-history", action="store_true")
    return parser.parse_args()


def _parse_stocks(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [stock.strip().upper() for stock in raw.split(",") if stock.strip()]


def _load_stocks_from_file(path: str | None) -> List[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"stocks file not found: {path}")
    text = file_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            tokens.extend(item.strip() for item in line.split(",") if item.strip())
        else:
            tokens.append(line)
    return [token.upper() for token in tokens]


def _parse_csv_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _parse_int_tuple(raw: str | None, fallback: int) -> tuple[int, ...]:
    if not raw:
        return (int(fallback),)
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    return values or (int(fallback),)


def _parse_horizon_weights(raw: str | None) -> dict[int, float]:
    if not raw:
        return {}
    out: dict[int, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        horizon_raw, weight_raw = item.split(":", 1)
        out[int(horizon_raw.strip())] = float(weight_raw.strip())
    return out


def _parse_state_horizon_profiles(raw: str | None) -> dict[str, dict[int, float]]:
    if not raw:
        return {}
    out: dict[str, dict[int, float]] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        state_raw, weights_raw = chunk.split("=", 1)
        out[state_raw.strip()] = _parse_horizon_weights(weights_raw)
    return out


def _parse_state_ensemble_weights(raw: str | None) -> dict[str, dict[str, float]]:
    if not raw:
        return {}
    out: dict[str, dict[str, float]] = {}
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        state_raw, weights_raw = chunk.split("=", 1)
        weights: dict[str, float] = {}
        for item in weights_raw.split(","):
            item = item.strip()
            if not item:
                continue
            name_raw, value_raw = item.split(":", 1)
            weights[name_raw.strip().lower()] = float(value_raw.strip())
        out[state_raw.strip()] = weights
    return out


def _load_artifact_meta(artifact_path: Path) -> dict[str, Any]:
    meta_path = artifact_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
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
        "status_text": "未知",
        "artifact_latest_data_date": str(latest_data_date.date()) if latest_data_date is not None else "",
        "trained_at": str(trained_at) if trained_at is not None else "",
        "trading_day_lag": None,
        "calendar_day_lag": None,
        "warnings": [],
        "should_block": False,
    }

    if latest_data_date is None:
        result["warnings"].append("模型元数据缺少 latest_data_date，无法确认是否过期。")
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
        result["warnings"].append("无法用交易日历计算模型滞后天数，请人工确认模型是否最新。")
        return result

    if trading_day_lag < 0:
        result["status"] = "future"
        result["status_text"] = "数据日超前"
        result["warnings"].append("模型 latest_data_date 晚于当前信号日，请检查数据口径。")
        return result

    if max_trading_days > 0 and trading_day_lag >= int(max_trading_days):
        result["status"] = "blocked"
        result["status_text"] = "过期拦截"
        result["should_block"] = True
        result["warnings"].append(
            f"模型 latest_data_date={latest_data_date.date()}，相对当前信号日滞后 {trading_day_lag} 个交易日，达到拦截阈值 {max_trading_days}。"
        )
        return result

    if warn_trading_days > 0 and trading_day_lag >= int(warn_trading_days):
        result["status"] = "warning"
        result["status_text"] = "过期提醒"
        result["warnings"].append(
            f"模型 latest_data_date={latest_data_date.date()}，相对当前信号日滞后 {trading_day_lag} 个交易日，请优先先跑 update_model.py。"
        )
        return result

    result["status"] = "fresh"
    result["status_text"] = "最新"
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
    latest_quadrant = str(regime_state.loc[latest_date, "quadrant"])

    ml_score_row, _ = predict_ml_scores_for_date_bundle(
        models=artifact.models,
        feature_frames=feature_frames,
        market_features=market_features,
        filter_mask=filter_mask,
        dt=latest_date,
        feature_names=artifact.feature_names,
        horizon_weights=resolve_horizon_weights(artifact_cfg, latest_quadrant),
    )
    ml_score = pd.DataFrame(np.nan, index=[latest_date], columns=score_none.columns)
    if not ml_score_row.empty:
        ml_score.loc[latest_date, ml_score_row.index] = ml_score_row.values

    score_none_latest = score_none.loc[[latest_date]]
    score_v2_latest = score_v2.loc[[latest_date]]
    final_score_raw = blend_scores(
        ml_score,
        score_none_latest,
        score_v2_latest,
        artifact_cfg,
        quadrant_series=regime_state.loc[[latest_date], "quadrant"],
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
                "enhanced_profile": enhanced_profile,
            }
        ]
    )
    return factor_bundle, regime_state, score_none, score_v2, ml_score, final_score_raw, final_score, target_weights, training_log


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
    benchmark_close = prepared_bundle["benchmark_close"]
    benchmark_open = prepared_bundle["benchmark_open"]
    ml_score, training_log = rolling_ml_scores_multi(
        feature_frames=feature_frames,
        market_features=market_features,
        close=factor_bundle["raw_inputs"]["Close"],
        benchmark_close=benchmark_close,
        open_df=factor_bundle["raw_inputs"]["Open"],
        benchmark_open=benchmark_open,
        filter_mask=filter_mask,
        regime_state=regime_state,
        config=ml_cfg,
    )
    final_score_raw = blend_scores(ml_score, score_none, score_v2, ml_cfg, quadrant_series=regime_state["quadrant"])
    target_weights = build_target_weights(final_score_raw, cfg, industry_map=industry_map, style_map=style_map)
    final_score = final_score_raw.copy()
    if cfg.enable_market_regime_filter:
        target_weights, final_score = apply_market_regime_filter(target_weights, final_score.fillna(0.0), regime_state)

    return factor_bundle, regime_state, score_none, score_v2, ml_score, final_score_raw, final_score, target_weights, training_log


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
    ml_score_row: pd.Series,
    positions_df: pd.DataFrame,
    cash: float,
    lot_size: int,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    latest_price = close_row.dropna()
    pos = positions_df.copy()
    if pos.empty:
        pos = pd.DataFrame(columns=["stock", "shares", "cost_price"])
    pos = pos[pos["stock"].isin(latest_price.index)].copy()

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
    available_cash = float(cash)

    for stock, shares in sorted(current_shares_map.items()):
        price = float(latest_price.get(stock, 0.0))
        current_value = float(shares * price)
        target_value = float(target_value_map.get(stock, 0.0))
        delta_value = target_value - current_value
        if target_value <= 0:
            sell_shares = shares
            sell_value = sell_shares * price
            available_cash += sell_value
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
                    "ml_score": float(ml_score_row.get(stock, 0.0)),
                    "cost_price": float(current_cost_map.get(stock, 0.0)),
                }
            )
        elif delta_value < -price * lot_size:
            sell_shares = _round_sell_shares(shares, delta_value, price, lot_size)
            if sell_shares > 0:
                sell_value = sell_shares * price
                available_cash += sell_value
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
                        "ml_score": float(ml_score_row.get(stock, 0.0)),
                        "cost_price": float(current_cost_map.get(stock, 0.0)),
                    }
                )

    # Convert target weights to position values before comparing against cash and lot size.
    for stock, target_value in target_value_map.items():
        price = float(latest_price.get(stock, 0.0))
        current_shares = int(current_shares_map.get(stock, 0))
        current_value = float(current_shares * price)
        delta_value = float(target_value - current_value)
        if delta_value <= price * lot_size:
            continue
        planned_buy_value = min(delta_value, available_cash)
        buy_shares = _round_buy_shares(planned_buy_value, price, lot_size)
        if buy_shares <= 0:
            continue
        est_value = float(buy_shares * price)
        available_cash -= est_value
        rows.append(
            {
                "stock": stock,
                "action": "买入" if current_shares == 0 else "加仓",
                "shares": buy_shares,
                "price": price,
                "est_value": est_value,
                "reason": "进入目标组合" if current_shares == 0 else "目标仓位上升",
                "current_weight": current_value / total_equity if total_equity > 0 else 0.0,
                "target_weight": float(target_weight_row.get(stock, 0.0)),
                "final_score": float(final_score_row.get(stock, 0.0)),
                "score_none": float(score_none_row.get(stock, 0.0)),
                "score_v2": float(score_v2_row.get(stock, 0.0)),
                "ml_score": float(ml_score_row.get(stock, 0.0)),
                "cost_price": float(current_cost_map.get(stock, 0.0)),
            }
        )

    action_df = pd.DataFrame(rows)
    action_df = action_df.sort_values(["action", "final_score"], ascending=[True, False]).reset_index(drop=True) if not action_df.empty else action_df

    execution_date = get_next_trading_date(latest_date)
    summary = {
        "signal_date": str(latest_date.date()),
        "execution_date": str(execution_date or ""),
        "cash_input": float(cash),
        "total_equity": float(total_equity),
        "estimated_cash_after_plan": float(available_cash),
        "current_position_count": int(len(current_shares_map)),
        "target_position_count": int((target_weight_row > 0).sum()),
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
    pos = pos[pos["stock"].isin(latest_price.index)].copy()
    rows = []
    for row in pos.itertuples(index=False):
        stock = row.stock
        shares = int(row.shares)
        price = float(latest_price.get(stock, 0.0))
        target_weight = float(target_weight_row.get(stock, 0.0))
        rows.append(
            {
                "date": latest_date,
                "stock": stock,
                "shares": shares,
                "close": price,
                "market_value": shares * price,
                "cost_price": float(row.cost_price),
                "target_weight": target_weight,
                "final_score": float(final_score_row.get(stock, 0.0)),
                "status": "目标持有" if target_weight > 0 else "待卖出",
            }
        )
    return pd.DataFrame(rows)


def _build_watchlist(
    latest_date: pd.Timestamp,
    final_score_row: pd.Series,
    target_weight_row: pd.Series,
    score_none_row: pd.Series,
    score_v2_row: pd.Series,
    ml_score_row: pd.Series,
    top_n: int = 15,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": latest_date,
            "stock": final_score_row.index,
            "final_score": final_score_row.values,
            "target_weight": target_weight_row.reindex(final_score_row.index).fillna(0.0).values,
            "score_none": score_none_row.reindex(final_score_row.index).values,
            "score_v2": score_v2_row.reindex(final_score_row.index).values,
            "ml_score": ml_score_row.reindex(final_score_row.index).values,
        }
    )
    return df.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)


def _write_trade_plan_txt(
    path: Path,
    summary: Dict[str, float],
    regime_state_row: pd.Series,
    action_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    watch_df: pd.DataFrame,
    model_info: Dict[str, str],
):
    def _fmt_metric(value: object) -> str:
        try:
            number = float(value)
        except Exception:
            return "nan"
        if np.isnan(number):
            return "nan"
        return f"{number:.3f}"

    lines: List[str] = []
    lines.append("每日盘后策略（次日开盘执行）")
    lines.append("=" * 36)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"信号日期: {summary['signal_date']}")
    if summary.get("execution_date"):
        lines.append(f"执行日期: {summary['execution_date']}")
    lines.append("执行方式: 盘后生成建议，下一交易日开盘手工执行")
    lines.append(f"市场状态: {regime_state_row.get('quadrant', '')}")
    lines.append(f"允许开仓: {'是' if bool(regime_state_row.get('regime_on', False)) else '否'}")
    lines.append(f"模型来源: {model_info.get('mode', '')}")
    if model_info.get("artifact_path"):
        lines.append(f"模型文件: {model_info['artifact_path']}")
    if model_info.get("trained_at"):
        lines.append(f"模型训练时间: {model_info['trained_at']}")
    if model_info.get("train_end"):
        lines.append(f"模型训练样本截止: {model_info['train_end']}")
    if model_info.get("latest_data_date"):
        lines.append(f"模型最新数据日: {model_info['latest_data_date']}")
    if model_info.get("freshness_status"):
        freshness_line = f"模型新鲜度: {model_info['freshness_status']}"
        if model_info.get("trading_day_lag") is not None:
            freshness_line += f" | 交易日滞后 {model_info['trading_day_lag']}"
        lines.append(freshness_line)
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
    if model_info.get("enhanced_profile"):
        lines.append(f"Enhanced Profile: {model_info['enhanced_profile']}")
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
    lines.append(f"输入现金: {summary['cash_input']:.2f}")
    lines.append(f"计划后剩余现金估算: {summary['estimated_cash_after_plan']:.2f}")
    lines.append("价格口径: 以下数量按信号日收盘价估算，次日开盘请按实际开盘价微调。")
    lines.append("")

    if action_df.empty:
        lines.append("一、次日开盘建议动作")
        lines.append("- 当前无明确调仓动作，建议次日开盘保持现有仓位。")
    else:
        lines.append("一、次日开盘建议动作")
        for idx, row in action_df.iterrows():
            lines.append(
                f"{idx + 1}. {row['action']} {row['stock']} | 估算数量 {int(row['shares'])} 股 | 信号日收盘参考 {row['price']:.2f} | "
                f"估算金额 {row['est_value']:.2f} | 原因: {row['reason']}"
            )
            lines.append(
                f"   当前权重 {row['current_weight']:.2%} -> 目标权重 {row['target_weight']:.2%} | "
                f"综合分 {row['final_score']:.4f} | ML {row['ml_score']:.4f} | none {row['score_none']:.4f} | v2 {row['score_v2']:.4f}"
            )

    lines.append("")
    lines.append("二、当前持仓概览")
    if hold_df.empty:
        lines.append("- 当前无持仓。")
    else:
        for _, row in hold_df.iterrows():
            lines.append(
                f"- {row['stock']} | 持股 {int(row['shares'])} 股 | 收盘 {row['close']:.2f} | 市值 {row['market_value']:.2f} | "
                f"目标权重 {row['target_weight']:.2%} | 状态 {row['status']}"
            )

    lines.append("")
    lines.append("三、候选观察名单")
    for _, row in watch_df.iterrows():
        lines.append(
            f"- {row['stock']} | 综合分 {row['final_score']:.4f} | 目标权重 {row['target_weight']:.2%} | "
            f"ML {row['ml_score']:.4f} | none {row['score_none']:.4f} | v2 {row['score_v2']:.4f}"
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
        regime_allowed_quadrants=_parse_csv_list(args.regime_quadrants),
        enable_style_cap=not args.no_style_cap,
        max_style_weight=args.max_style_weight,
        enable_industry_cap=args.industry_cap,
        max_industry_weight=args.max_industry_weight,
    )
    stocks = _parse_stocks(args.stocks)
    file_stocks = _load_stocks_from_file(args.stocks_file)
    if stocks or file_stocks:
        stocks = list(dict.fromkeys(stocks + file_stocks))
    if stocks:
        cfg.universe = stocks

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=_parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=_parse_horizon_weights(args.ml_horizon_weights),
        state_horizon_weights=_parse_state_horizon_profiles(args.ml_state_horizon_profiles),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=args.ml_retrain_every_days,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        ensemble_ml_weight=args.ensemble_ml_weight,
        ensemble_none_weight=args.ensemble_none_weight,
        ensemble_v2_weight=args.ensemble_v2_weight,
        state_ensemble_weights=_parse_state_ensemble_weights(args.ensemble_state_weights),
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    artifact = None
    artifact_meta: dict[str, Any] = {}
    artifact_path = Path(args.model_artifact)
    effective_profile = args.enhanced_profile
    effective_ml_cfg = ml_cfg
    if not args.train_on_the_fly:
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"模型产物不存在: {artifact_path}。请先运行 daily_research/execution/update_model.py。"
            )
        artifact = load_ml_artifact(artifact_path)
        artifact_meta = _load_artifact_meta(artifact_path)
        effective_ml_cfg = MLAplhaConfig(**artifact.ml_config)
        effective_profile = str(getattr(effective_ml_cfg, "enhanced_profile", "up_low_breakout_v2") or "up_low_breakout_v2")

    if args.data_source == "tq":
        if not cfg.universe and cfg.universe_scope == "all_a":
            print("[1/9] 正在从 TQ 加载全A股票池...")
            cfg.universe = load_universe_from_tq(cfg.universe_scope)
        elif not cfg.universe:
            raise ValueError("TQ 模式下，未指定 --stocks 时目前仅支持 --universe-scope all_a。")
    elif not args.csv_folder:
        raise ValueError("CSV 模式需要提供 --csv-folder。")

    history_window = resolve_history_window(
        cfg=cfg,
        ml_cfg=effective_ml_cfg,
        requested_start_date=args.start_date,
        end_date=args.end_date,
        mode="infer",
        auto_trim_history=not args.no_auto_trim_history,
    )
    print(
        f"[2/9] 推理历史窗口: {history_window.effective_start_date} -> "
        f"{history_window.end_date or 'latest'} | required_trading_days={history_window.required_trading_days}"
    )
    print(f"[3/9] 正在准备行情数据，股票数: {len(cfg.universe)}，基准: {cfg.benchmark}")
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

    print("[5/9] 正在读取当前账号快照...")
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
            "[warning] 当前 positions 文件未提供 account 行现金，且未传入 --cash；"
            " 本次按 0 现金生成计划。"
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
        print("[7/9] 正在加载风格映射...")
        style_map = load_style_map_from_tq(list(df_dict["Close"].columns))
    if cfg.enable_industry_cap and args.data_source == "tq":
        industry_map = load_industry_map_from_tq(list(df_dict["Close"].columns))

    print("[8/9] 正在计算先进版分数...")
    if args.train_on_the_fly:
        factor_bundle, regime_state, score_none, score_v2, ml_score, final_score_raw, final_score_filtered, target_weights, training_log = _build_scores_on_the_fly(
            cfg, ml_cfg, prepared_bundle, style_map, industry_map
        )
        model_info = {
            "mode": "实时训练",
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "horizons": ",".join(str(h) for h in ml_cfg.target_horizons),
            "model_family": str(ml_cfg.model_family),
            "state_horizon_profiles": ";".join(
                f"{state}=" + ",".join(f"{k}:{v:.2f}" for k, v in sorted(weights.items()))
                for state, weights in (ml_cfg.state_horizon_weights or {}).items()
            ),
            "history_window": history_window_to_dict(history_window),
            "enhanced_profile": str(ml_cfg.enhanced_profile),
        }
    else:
        factor_bundle, regime_state, score_none, score_v2, ml_score, final_score_raw, final_score_filtered, target_weights, training_log = _build_scores_from_artifact(
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
            "horizon_weights": ",".join(
                f"{k}:{v:.2f}" for k, v in sorted((artifact.ml_config.get("target_horizon_weights") or {}).items())
            ),
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
                "模型元数据尚未包含 validation_summary，建议先运行 update_model.py 生成新产物。"
            )
        if freshness_info.get("should_block") and args.allow_stale_model:
            freshness_info.setdefault("warnings", []).append("已使用 --allow-stale-model 放行过期模型，请谨慎执行。")
        for warning in freshness_info.get("warnings", []):
            print(f"[warning] {warning}")
        if freshness_info.get("should_block") and not args.allow_stale_model:
            raise RuntimeError(
                "模型已达到过期拦截阈值。"
                f" latest_data_date={freshness_info.get('artifact_latest_data_date', '')},"
                f" trading_day_lag={freshness_info.get('trading_day_lag')}。"
                "请先运行 daily_research/execution/update_model.py；如确需继续，可显式传入 --allow-stale-model。"
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

    print("[9/9] 正在生成盘后策略与次日开盘执行建议...")
    action_df, summary = _build_trade_plan(
        latest_date=signal_date,
        close_row=factor_bundle["raw_inputs"]["Close"].loc[signal_date],
        target_weight_row=target_weights.loc[signal_date],
        final_score_row=final_score_raw.loc[signal_date],
        score_none_row=score_none.loc[signal_date],
        score_v2_row=score_v2.loc[signal_date],
        ml_score_row=ml_score.loc[signal_date],
        positions_df=positions_df,
        cash=effective_cash,
        lot_size=args.lot_size,
    )
    summary["positions_source"] = str(account_state.path)
    summary["positions_source_mode"] = str(account_state.source)
    summary["cash_source"] = str(cash_source)
    summary["cash_source_text"] = str(cash_source_text)
    summary["history_window"] = history_window_to_dict(history_window)
    summary["cache"] = {
        "raw": raw_cache_meta,
        "prepared": prepared_cache_meta,
    }
    if not args.train_on_the_fly:
        summary["model_artifact"] = str(artifact_path)
        summary["model_latest_data_date"] = str(artifact_meta.get("latest_data_date", ""))
        summary["model_freshness"] = freshness_info
        summary["model_validation"] = validation_summary
    hold_df = _build_hold_table(
        latest_date=signal_date,
        close_row=factor_bundle["raw_inputs"]["Close"].loc[signal_date],
        target_weight_row=target_weights.loc[signal_date],
        final_score_row=final_score_raw.loc[signal_date],
        positions_df=positions_df,
    )
    watch_df = _build_watchlist(
        latest_date=signal_date,
        final_score_row=final_score_raw.loc[signal_date].dropna(),
        target_weight_row=target_weights.loc[signal_date],
        score_none_row=score_none.loc[signal_date],
        score_v2_row=score_v2.loc[signal_date],
        ml_score_row=ml_score.loc[signal_date],
        top_n=15,
    )

    print("正在写入输出文件...")
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

    action_df.to_csv(run_dir / "actions_today.csv", index=False, encoding="utf-8-sig")
    hold_df.to_csv(run_dir / "holdings_snapshot.csv", index=False, encoding="utf-8-sig")
    watch_df.to_csv(run_dir / "watchlist.csv", index=False, encoding="utf-8-sig")
    training_log.to_csv(run_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    with open(run_dir / "plan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"输出目录: {run_dir}")
    print(f"最新建议文件: {latest_txt_path}")
    if not action_df.empty:
        print(action_df[["stock", "action", "shares", "price", "reason"]].to_string(index=False))
    else:
        print("今日无明确调仓动作。")


if __name__ == "__main__":
    main()
