from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_TRAINING_PACK_MANIFEST = Path(
    "quant_data_platform/data/memmap/training_pack/"
    "mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/"
    "qdp_training_pack_manifest.json"
)
DEFAULT_ROLES = ("validation", "test")
DEFAULT_TOP_KS = (1, 3, 5, 10)
DEFAULT_LABEL_HORIZONS = (1, 2, 4)
DEFAULT_SCORE_COLUMNS = (
    "score_strong_continuation",
    "score_trend_pullback",
    "score_industry_spread",
    "score_combined_shortline",
)


CONTINUATION_WEIGHTS = {
    "raw_limit_up_like_1d": 0.12,
    "raw_amount_ratio_5_20": 0.10,
    "raw_volume_ratio_5_20": 0.06,
    "intraday_close_position": 0.12,
    "intraday_last_30m_ret": 0.10,
    "intraday_close_to_vwap": 0.10,
    "intraday_price_above_vwap_share": 0.08,
    "intraday_cum_vwap_slope": 0.08,
    "intraday_close_pressure_30m": 0.08,
    "industry_ret_5_excess": 0.08,
    "industry_positive_share_5": 0.04,
    "stock_ret_5_minus_industry": 0.04,
}

TREND_PULLBACK_TREND_WEIGHTS = {
    "ret_20d": 0.16,
    "cs_rank_ret_20d": 0.14,
    "stock_ret_20_minus_industry": 0.10,
    "industry_ret_5_excess": 0.08,
    "industry_positive_share_5": 0.06,
}
TREND_PULLBACK_PULLBACK_WEIGHTS = {
    "ret_1d": 0.10,
    "raw_close_from_prev_close_1d": 0.08,
    "stock_ret_5_minus_industry": 0.06,
}
TREND_PULLBACK_SUPPORT_WEIGHTS = {
    "intraday_low_to_close_ret": 0.10,
    "intraday_close_position": 0.10,
    "intraday_last_30m_ret": 0.08,
    "intraday_close_to_vwap": 0.06,
    "raw_amount_ratio_5_20": 0.04,
}

INDUSTRY_SPREAD_INDUSTRY_WEIGHTS = {
    "industry_ret_5_excess": 0.14,
    "industry_rank_ret_5": 0.12,
    "industry_positive_share_5": 0.10,
    "industry_rank_turn": 0.06,
    "industry_member_count_log": 0.03,
}
INDUSTRY_SPREAD_LAGGARD_WEIGHTS = {
    "stock_ret_5_minus_industry": 0.12,
    "stock_ret_20_minus_industry": 0.08,
}
INDUSTRY_SPREAD_CONFIRM_WEIGHTS = {
    "intraday_last_30m_ret": 0.08,
    "intraday_close_position": 0.08,
    "raw_amount_ratio_5_20": 0.06,
    "turn_ratio_5_20": 0.04,
    "intraday_close_pressure_30m": 0.04,
    "intraday_price_above_vwap_share": 0.03,
}
MECHANISM_FEATURES = tuple(
    dict.fromkeys(
        [
            *CONTINUATION_WEIGHTS.keys(),
            *TREND_PULLBACK_TREND_WEIGHTS.keys(),
            *TREND_PULLBACK_PULLBACK_WEIGHTS.keys(),
            *TREND_PULLBACK_SUPPORT_WEIGHTS.keys(),
            *INDUSTRY_SPREAD_INDUSTRY_WEIGHTS.keys(),
            *INDUSTRY_SPREAD_LAGGARD_WEIGHTS.keys(),
            *INDUSTRY_SPREAD_CONFIRM_WEIGHTS.keys(),
        ]
    )
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return str(path.resolve())


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _parse_csv_ints(raw: str | Iterable[int] | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = [int(item) for item in raw]
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out or default)


def _parse_csv_strings(raw: str | Iterable[str] | None, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out or default)


def _resolve_path(path: str | Path, *, root: Path | None = None) -> Path:
    out = Path(path)
    if not out.is_absolute() and root is not None:
        out = root / out
    return out


def _rank_percentile(values: pd.Series | np.ndarray, *, high_is_good: bool = True) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype="float64"))
    ranks = series.rank(method="average", pct=True, na_option="keep").to_numpy(dtype="float64")
    ranks = np.nan_to_num(ranks, nan=0.5, posinf=0.5, neginf=0.5)
    if not high_is_good:
        ranks = 1.0 - ranks
    return np.clip(ranks, 0.0, 1.0)


def _weighted_rank_score(frame: pd.DataFrame, weights: dict[str, float], *, inverse: bool = False) -> np.ndarray:
    parts: list[np.ndarray] = []
    resolved_weights: list[float] = []
    for column, weight in weights.items():
        if column not in frame.columns:
            continue
        parts.append(_rank_percentile(frame[column], high_is_good=not inverse))
        resolved_weights.append(float(weight))
    if not parts:
        return np.full((len(frame),), 0.5, dtype="float64")
    weight_array = np.asarray(resolved_weights, dtype="float64")
    weight_sum = float(weight_array.sum())
    if weight_sum <= 0.0:
        return np.full((len(frame),), 0.5, dtype="float64")
    stacked = np.vstack(parts)
    return np.average(stacked, axis=0, weights=weight_array)


def add_shortline_mechanism_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized mechanism scores using same-date cross-sectional ranks."""

    if frame.empty:
        out = frame.copy()
        for column in DEFAULT_SCORE_COLUMNS:
            out[column] = np.asarray([], dtype="float64")
        return out
    groups: list[pd.DataFrame] = []
    group_keys = ["role", "date"] if "role" in frame.columns else ["date"]
    for _, group in frame.groupby(group_keys, sort=False, dropna=False):
        work = group.copy()
        continuation = _weighted_rank_score(work, CONTINUATION_WEIGHTS)
        trend = _weighted_rank_score(work, TREND_PULLBACK_TREND_WEIGHTS)
        pullback = _weighted_rank_score(work, TREND_PULLBACK_PULLBACK_WEIGHTS, inverse=True)
        support = _weighted_rank_score(work, TREND_PULLBACK_SUPPORT_WEIGHTS)
        trend_pullback = 0.45 * trend + 0.25 * pullback + 0.30 * support
        industry = _weighted_rank_score(work, INDUSTRY_SPREAD_INDUSTRY_WEIGHTS)
        laggard = _weighted_rank_score(work, INDUSTRY_SPREAD_LAGGARD_WEIGHTS, inverse=True)
        confirm = _weighted_rank_score(work, INDUSTRY_SPREAD_CONFIRM_WEIGHTS)
        industry_spread = 0.45 * industry + 0.25 * laggard + 0.30 * confirm
        mechanism_frame = pd.DataFrame(
            {
                "continuation": continuation,
                "trend_pullback": trend_pullback,
                "industry_spread": industry_spread,
            },
            index=work.index,
        )
        combined = (
            0.40 * _rank_percentile(mechanism_frame["continuation"])
            + 0.35 * _rank_percentile(mechanism_frame["trend_pullback"])
            + 0.25 * _rank_percentile(mechanism_frame["industry_spread"])
        )
        work["score_strong_continuation"] = continuation
        work["score_trend_pullback"] = trend_pullback
        work["score_industry_spread"] = industry_spread
        work["score_combined_shortline"] = combined
        groups.append(work)
    return pd.concat(groups, axis=0).sort_index(kind="mergesort")


def _entry_valid_mask(frame: pd.DataFrame) -> pd.Series:
    if "entry_tradeable" in frame.columns:
        entry_tradeable = pd.to_numeric(frame["entry_tradeable"], errors="coerce").fillna(0.0) > 0.5
    else:
        entry_tradeable = pd.Series(True, index=frame.index)
    if "entry_limit_up_buy_blocked" in frame.columns:
        limit_blocked = pd.to_numeric(frame["entry_limit_up_buy_blocked"], errors="coerce").fillna(0.0) > 0.5
    else:
        limit_blocked = pd.Series(False, index=frame.index)
    if "entry_suspended_or_no_open" in frame.columns:
        suspended = pd.to_numeric(frame["entry_suspended_or_no_open"], errors="coerce").fillna(0.0) > 0.5
    else:
        suspended = pd.Series(False, index=frame.index)
    return entry_tradeable & ~limit_blocked & ~suspended


def _exit_label(label_horizon: int) -> str:
    return f"D+{int(label_horizon) + 1}_open"


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _mean_or_nan(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def _build_daily_rows(
    frame: pd.DataFrame,
    *,
    score_columns: tuple[str, ...],
    top_ks: tuple[int, ...],
    label_horizons: tuple[int, ...],
    round_trip_cost_bps: float,
    filter_entry: bool,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    cost = float(round_trip_cost_bps) / 10000.0
    rows: list[dict[str, Any]] = []
    group_keys = ["role", "date"] if "role" in frame.columns else ["date"]
    for group_values, group in frame.groupby(group_keys, sort=True, dropna=False):
        if isinstance(group_values, tuple):
            role = str(group_values[0])
            date = str(group_values[1])
        else:
            role = ""
            date = str(group_values)
        valid_entry = _entry_valid_mask(group)
        for score_column in score_columns:
            if score_column not in group.columns:
                continue
            scored = group.loc[pd.to_numeric(group[score_column], errors="coerce").notna()].copy()
            if scored.empty:
                continue
            scored["_score"] = pd.to_numeric(scored[score_column], errors="coerce").astype("float64")
            scored["_entry_valid"] = valid_entry.loc[scored.index].astype(bool)
            scored = scored.sort_values("_score", ascending=False, kind="mergesort")
            selection_pool = scored.loc[scored["_entry_valid"]].copy() if filter_entry else scored
            for top_k in top_ks:
                prefilter = scored.head(min(int(top_k), len(scored))).copy()
                selected = selection_pool.head(min(int(top_k), len(selection_pool))).copy()
                candidate_blocked_rate = float((~prefilter["_entry_valid"]).mean()) if not prefilter.empty else float("nan")
                final_blocked_rate = float((~selected["_entry_valid"]).mean()) if not selected.empty else float("nan")
                selected_stocks = ",".join(selected.get("stock", pd.Series(dtype=str)).astype(str).tolist())
                for label_horizon in label_horizons:
                    abs_col = f"future_cum_return_{int(label_horizon)}d"
                    excess_col = f"future_cum_excess_return_{int(label_horizon)}d"
                    if abs_col not in selected.columns or excess_col not in selected.columns:
                        continue
                    selected_abs = _safe_numeric(selected, abs_col).dropna()
                    selected_excess = _safe_numeric(selected, excess_col).dropna()
                    if selected_abs.empty or selected_excess.empty:
                        continue
                    all_pool = selection_pool if filter_entry else scored
                    all_abs = _safe_numeric(all_pool, abs_col).dropna()
                    all_excess = _safe_numeric(all_pool, excess_col).dropna()
                    if all_abs.empty or all_excess.empty:
                        continue
                    gross_abs = float(selected_abs.mean())
                    gross_excess = float(selected_excess.mean())
                    all_abs_mean = float(all_abs.mean())
                    all_excess_mean = float(all_excess.mean())
                    row: dict[str, Any] = {
                        "role": role,
                        "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                        "score_column": str(score_column),
                        "mechanism": str(score_column).replace("score_", ""),
                        "top_k": int(top_k),
                        "label_horizon": int(label_horizon),
                        "exit_label": _exit_label(int(label_horizon)),
                        "available_count": int(len(scored)),
                        "entry_valid_available_count": int(scored["_entry_valid"].sum()),
                        "prefilter_selected_count": int(len(prefilter)),
                        "selected_count": int(len(selected_abs)),
                        "candidate_blocked_rate": candidate_blocked_rate,
                        "final_blocked_rate": final_blocked_rate,
                        "selected_stocks": selected_stocks,
                        "score_mean": float(selected["_score"].mean()) if not selected.empty else float("nan"),
                        "score_min": float(selected["_score"].min()) if not selected.empty else float("nan"),
                        "score_max": float(selected["_score"].max()) if not selected.empty else float("nan"),
                        "gross_abs_return": gross_abs,
                        "net_abs_return": gross_abs - cost,
                        "gross_excess_return": gross_excess,
                        "net_excess_return": gross_excess - cost,
                        "all_mean_abs_return": all_abs_mean,
                        "all_mean_excess_return": all_excess_mean,
                        "abs_vs_all_mean": gross_abs - all_abs_mean,
                        "excess_vs_all_mean": gross_excess - all_excess_mean,
                        "abs_hit_rate": float((selected_abs > 0.0).mean()),
                        "excess_hit_rate": float((selected_excess > 0.0).mean()),
                        "abs_outperform_all_mean_rate": float((selected_abs > all_abs_mean).mean()),
                        "excess_outperform_all_mean_rate": float((selected_excess > all_excess_mean).mean()),
                    }
                    rank_col = f"future_rank_{int(label_horizon)}d"
                    if rank_col in selected.columns:
                        row["selected_future_rank_mean"] = _mean_or_nan(_safe_numeric(selected, rank_col))
                    industry_rank_col = f"future_industry_rank_{int(label_horizon)}d"
                    if industry_rank_col in selected.columns:
                        row["selected_future_industry_rank_mean"] = _mean_or_nan(_safe_numeric(selected, industry_rank_col))
                    rows.append(row)
    return pd.DataFrame(rows)


def _monthly_rows(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    work = daily.copy()
    work["month"] = pd.to_datetime(work["date"], errors="coerce").dt.to_period("M").astype(str)
    group_cols = ["role", "score_column", "mechanism", "top_k", "label_horizon", "exit_label", "month"]
    agg = (
        work.groupby(group_cols, as_index=False)
        .agg(
            date_count=("date", "nunique"),
            net_abs_return=("net_abs_return", "mean"),
            net_excess_return=("net_excess_return", "mean"),
            gross_abs_return=("gross_abs_return", "mean"),
            gross_excess_return=("gross_excess_return", "mean"),
            abs_hit_rate=("abs_hit_rate", "mean"),
            excess_hit_rate=("excess_hit_rate", "mean"),
            candidate_blocked_rate=("candidate_blocked_rate", "mean"),
        )
        .reset_index(drop=True)
    )
    return agg


def _series_stats(values: pd.Series, *, label_horizon: int) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype("float64")
    if numeric.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "positive_rate": float("nan"),
            "window_sharpe_like": float("nan"),
        }
    std = float(numeric.std(ddof=0))
    scale = math.sqrt(252.0 / max(int(label_horizon), 1))
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": std,
        "p25": float(numeric.quantile(0.25)),
        "p75": float(numeric.quantile(0.75)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "positive_rate": float((numeric > 0.0).mean()),
        "window_sharpe_like": float(numeric.mean() / std * scale) if std > 0.0 else float("nan"),
    }


def _monthly_metrics(monthly: pd.DataFrame, value_col: str) -> dict[str, Any]:
    if monthly.empty or value_col not in monthly.columns:
        return {"positive_month_rate": float("nan"), "worst_month": "", "worst_month_mean": float("nan")}
    values = pd.to_numeric(monthly[value_col], errors="coerce")
    valid = monthly.loc[values.notna()].copy()
    if valid.empty:
        return {"positive_month_rate": float("nan"), "worst_month": "", "worst_month_mean": float("nan")}
    valid["_value"] = pd.to_numeric(valid[value_col], errors="coerce")
    worst = valid.sort_values("_value", ascending=True).iloc[0]
    best = valid.sort_values("_value", ascending=False).iloc[0]
    return {
        "positive_month_rate": float((valid["_value"] > 0.0).mean()),
        "worst_month": str(worst["month"]),
        "worst_month_mean": float(worst["_value"]),
        "best_month": str(best["month"]),
        "best_month_mean": float(best["_value"]),
    }


def _summary_rows(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["role", "score_column", "mechanism", "top_k", "label_horizon", "exit_label"]
    monthly_groups = {keys: group for keys, group in monthly.groupby(group_cols, sort=False)} if not monthly.empty else {}
    for keys, group in daily.groupby(group_cols, sort=True):
        role, score_column, mechanism, top_k, label_horizon, exit_label = keys
        net_abs = _series_stats(group["net_abs_return"], label_horizon=int(label_horizon))
        net_excess = _series_stats(group["net_excess_return"], label_horizon=int(label_horizon))
        month_group = monthly_groups.get(keys, pd.DataFrame())
        monthly_abs = _monthly_metrics(month_group, "net_abs_return")
        monthly_excess = _monthly_metrics(month_group, "net_excess_return")
        row: dict[str, Any] = {
            "role": str(role),
            "score_column": str(score_column),
            "mechanism": str(mechanism),
            "top_k": int(top_k),
            "label_horizon": int(label_horizon),
            "exit_label": str(exit_label),
            "date_count": int(group["date"].nunique()),
            "selected_count_mean": float(group["selected_count"].mean()),
            "available_count_mean": float(group["available_count"].mean()),
            "entry_valid_available_count_mean": float(group["entry_valid_available_count"].mean()),
            "candidate_blocked_rate_mean": float(group["candidate_blocked_rate"].mean()),
            "final_blocked_rate_mean": float(group["final_blocked_rate"].mean()),
            "gross_abs_mean": float(group["gross_abs_return"].mean()),
            "net_abs_mean": net_abs["mean"],
            "net_abs_mean_per_day": net_abs["mean"] / max(int(label_horizon), 1),
            "net_abs_median": net_abs["median"],
            "net_abs_p25": net_abs["p25"],
            "net_abs_p75": net_abs["p75"],
            "net_abs_min": net_abs["min"],
            "net_abs_positive_rate": net_abs["positive_rate"],
            "net_abs_window_sharpe_like": net_abs["window_sharpe_like"],
            "gross_excess_mean": float(group["gross_excess_return"].mean()),
            "net_excess_mean": net_excess["mean"],
            "net_excess_mean_per_day": net_excess["mean"] / max(int(label_horizon), 1),
            "net_excess_median": net_excess["median"],
            "net_excess_p25": net_excess["p25"],
            "net_excess_p75": net_excess["p75"],
            "net_excess_min": net_excess["min"],
            "net_excess_positive_rate": net_excess["positive_rate"],
            "net_excess_window_sharpe_like": net_excess["window_sharpe_like"],
            "all_mean_abs_return": float(group["all_mean_abs_return"].mean()),
            "all_mean_excess_return": float(group["all_mean_excess_return"].mean()),
            "abs_vs_all_mean": float(group["abs_vs_all_mean"].mean()),
            "excess_vs_all_mean": float(group["excess_vs_all_mean"].mean()),
            "abs_hit_rate_mean": float(group["abs_hit_rate"].mean()),
            "excess_hit_rate_mean": float(group["excess_hit_rate"].mean()),
            "abs_outperform_all_mean_rate": float(group["abs_outperform_all_mean_rate"].mean()),
            "excess_outperform_all_mean_rate": float(group["excess_outperform_all_mean_rate"].mean()),
            "net_abs_positive_month_rate": monthly_abs["positive_month_rate"],
            "net_abs_worst_month": monthly_abs["worst_month"],
            "net_abs_worst_month_mean": monthly_abs["worst_month_mean"],
            "net_excess_positive_month_rate": monthly_excess["positive_month_rate"],
            "net_excess_worst_month": monthly_excess["worst_month"],
            "net_excess_worst_month_mean": monthly_excess["worst_month_mean"],
        }
        for optional in ("selected_future_rank_mean", "selected_future_industry_rank_mean"):
            if optional in group.columns:
                row[optional] = _mean_or_nan(group[optional])
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["role", "net_abs_mean", "net_abs_positive_month_rate", "abs_hit_rate_mean"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def _leaderboard(summary: pd.DataFrame, *, limit: int = 20) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    ranked = summary.sort_values(
        ["net_abs_mean", "net_abs_positive_month_rate", "abs_hit_rate_mean", "abs_vs_all_mean"],
        ascending=[False, False, False, False],
    ).head(int(limit))
    cols = [
        "role",
        "mechanism",
        "top_k",
        "label_horizon",
        "exit_label",
        "date_count",
        "net_abs_mean",
        "net_abs_mean_per_day",
        "gross_abs_mean",
        "net_excess_mean",
        "abs_vs_all_mean",
        "excess_vs_all_mean",
        "abs_hit_rate_mean",
        "net_abs_positive_month_rate",
        "net_abs_worst_month",
        "net_abs_worst_month_mean",
        "candidate_blocked_rate_mean",
    ]
    available = [column for column in cols if column in ranked.columns]
    return [dict(row) for row in ranked.loc[:, available].to_dict(orient="records")]


def _choose_dates(sample_index: pd.DataFrame, *, role: str, max_dates: int) -> tuple[str, ...]:
    role_mask = sample_index["role"].astype(str).eq(str(role))
    dates = pd.to_datetime(sample_index.loc[role_mask, "date"], errors="coerce").dropna()
    unique_dates = sorted({pd.Timestamp(item).strftime("%Y-%m-%d") for item in dates.tolist()})
    if int(max_dates) > 0 and len(unique_dates) > int(max_dates):
        positions = np.linspace(0, len(unique_dates) - 1, int(max_dates), dtype=np.int64)
        unique_dates = [unique_dates[int(pos)] for pos in positions.tolist()]
    return tuple(unique_dates)


def build_shortline_stage0_diagnostics_from_frame(
    *,
    frame: pd.DataFrame,
    output_root: str | Path,
    run_tag: str = "shortline_stage0_diagnostics",
    top_ks: str | Iterable[int] | None = None,
    label_horizons: str | Iterable[int] | None = None,
    score_columns: str | Iterable[str] | None = None,
    round_trip_cost_bps: float = 20.0,
    filter_entry: bool = True,
    leaderboard_limit: int = 20,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_top_ks = _parse_csv_ints(top_ks, default=DEFAULT_TOP_KS)
    resolved_horizons = _parse_csv_ints(label_horizons, default=DEFAULT_LABEL_HORIZONS)
    scored = add_shortline_mechanism_scores(frame)
    resolved_scores = tuple(column for column in _parse_csv_strings(score_columns, default=DEFAULT_SCORE_COLUMNS) if column in scored.columns)
    daily = _build_daily_rows(
        scored,
        score_columns=resolved_scores,
        top_ks=resolved_top_ks,
        label_horizons=resolved_horizons,
        round_trip_cost_bps=float(round_trip_cost_bps),
        filter_entry=bool(filter_entry),
    )
    monthly = _monthly_rows(daily)
    summary = _summary_rows(daily, monthly)
    daily_csv = _write_frame(root / "shortline_stage0_daily.csv", daily)
    monthly_csv = _write_frame(root / "shortline_stage0_monthly.csv", monthly)
    summary_csv = _write_frame(root / "shortline_stage0_summary.csv", summary)
    report = {
        "status": "completed" if not summary.empty else "empty",
        "created_at": _now(),
        "run_tag": str(run_tag),
        "output_root": str(root.resolve()),
        "contract": {
            "diagnostic_kind": "shortline_after_close_stage0_fixed_next_open",
            "research_only": True,
            "not_model_training": True,
            "active_artifact_impact": "unchanged",
            "entry_semantics": "D close signal; D+1 open fixed-entry baseline; optional entry-tradeability filter",
            "t_plus_1_semantics": "label_horizon=1 means D+1 open entry to D+2 open exit; label_horizon=2 means D+3 open exit; label_horizon=4 means D+5 open exit",
            "do_not_claim": "D+1 open to D+1 close is not executable under A-share T+1",
            "feature_semantics": "same-date cross-sectional ranks over normalized alpha_v2 features; absolute thresholds are intentionally avoided",
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "filter_entry": bool(filter_entry),
            "top_ks": [int(item) for item in resolved_top_ks],
            "label_horizons": [int(item) for item in resolved_horizons],
            "score_columns": list(resolved_scores),
        },
        "input": {
            "row_count": int(len(frame)),
            "date_count": int(pd.to_datetime(frame["date"], errors="coerce").nunique()) if "date" in frame.columns else 0,
            "stock_count": int(frame["stock"].astype(str).nunique()) if "stock" in frame.columns else 0,
            "roles": sorted(frame["role"].astype(str).unique().tolist()) if "role" in frame.columns and not frame.empty else [],
        },
        "outputs": {
            "daily_csv": daily_csv,
            "monthly_csv": monthly_csv,
            "summary_csv": summary_csv,
        },
        "leaderboard": _leaderboard(summary, limit=int(leaderboard_limit)),
    }
    report_json = _write_json(root / "shortline_stage0_report.json", report)
    report["outputs"]["report_json"] = report_json
    _write_markdown(root / "shortline_stage0_report.md", report)
    report["outputs"]["report_md"] = str((root / "shortline_stage0_report.md").resolve())
    _write_json(root / "shortline_stage0_report.json", report)
    return report


def _open_label_array(manifest: dict[str, Any], name: str, *, root: Path) -> np.memmap:
    meta = dict(dict(manifest.get("label_arrays", {}) or {}).get(name, {}) or {})
    if not meta:
        raise ValueError(f"training pack missing required label array: {name}")
    path = _resolve_path(str(meta.get("path", "") or ""), root=root)
    shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
    dtype = str(meta.get("dtype", "float32") or "float32")
    if not path.exists():
        raise FileNotFoundError(f"label array not found: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _load_feature_panel(manifest: dict[str, Any], *, root: Path) -> np.memmap:
    meta = dict(manifest.get("date_major_feature_panel", {}) or {})
    path_raw = str(meta.get("path", "") or manifest.get("date_major_feature_panel_path", "") or "")
    shape_raw = meta.get("shape") or manifest.get("date_major_feature_panel_shape") or []
    dtype = str(meta.get("dtype") or manifest.get("date_major_feature_dtype") or manifest.get("feature_dtype", "float16") or "float16")
    path = _resolve_path(path_raw, root=root)
    shape = tuple(int(item) for item in list(shape_raw or []))
    if len(shape) != 3:
        raise ValueError("training pack missing date-major feature panel shape.")
    if not path.exists():
        raise FileNotFoundError(f"date-major feature panel not found: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _date_frame_from_pack(
    *,
    group: pd.DataFrame,
    feature_panel: np.memmap,
    feature_columns: list[str],
    feature_indices: dict[str, int],
    labels: dict[str, np.memmap],
    label_horizons: tuple[int, ...],
    horizon_to_risk_pos: dict[int, int],
) -> pd.DataFrame:
    sample_pos = pd.to_numeric(group["_sample_pos"], errors="coerce").to_numpy(dtype=np.int64, copy=True)
    date_pos_values = pd.to_numeric(group["global_date_pos"], errors="coerce").to_numpy(dtype=np.int64, copy=True)
    stock_pos = pd.to_numeric(group["global_stock_pos"], errors="coerce").to_numpy(dtype=np.int64, copy=True)
    if len(np.unique(date_pos_values)) != 1:
        raise ValueError("date group contains multiple global_date_pos values.")
    date_pos = int(date_pos_values[0])
    selected_feature_names = [name for name in MECHANISM_FEATURES if name in feature_indices]
    selected_feature_positions = np.asarray([int(feature_indices[name]) for name in selected_feature_names], dtype=np.int64)
    features = (
        np.asarray(feature_panel[date_pos, stock_pos[:, None], selected_feature_positions[None, :]], dtype=np.float32)
        if selected_feature_positions.size
        else np.zeros((len(group), 0), dtype=np.float32)
    )
    out: dict[str, Any] = {
        "role": group["role"].astype(str).to_numpy(),
        "date": group["date"].astype(str).to_numpy(),
        "stock": group["stock"].astype(str).to_numpy(),
        "entry_tradeable": np.asarray(labels["entry_tradeable"][sample_pos], dtype=np.float32),
        "entry_limit_up_buy_blocked": np.asarray(labels["entry_limit_up_buy_blocked"][sample_pos], dtype=np.float32),
        "entry_suspended_or_no_open": np.asarray(labels["entry_suspended_or_no_open"][sample_pos], dtype=np.float32),
    }
    for col_idx, column in enumerate(selected_feature_names):
        out[column] = features[:, col_idx]
    for horizon in label_horizons:
        label_idx = int(horizon) - 1
        if label_idx < 0 or label_idx >= int(labels["cumulative_return_1to20"].shape[1]):
            continue
        out[f"future_cum_return_{int(horizon)}d"] = np.asarray(labels["cumulative_return_1to20"][sample_pos, label_idx], dtype=np.float32)
        out[f"future_cum_excess_return_{int(horizon)}d"] = np.asarray(
            labels["cumulative_excess_return_1to20"][sample_pos, label_idx],
            dtype=np.float32,
        )
        if "rank_1to20" in labels:
            out[f"future_rank_{int(horizon)}d"] = np.asarray(labels["rank_1to20"][sample_pos, label_idx], dtype=np.float32)
        if int(horizon) in horizon_to_risk_pos and "industry_rank_by_horizon" in labels:
            risk_pos = int(horizon_to_risk_pos[int(horizon)])
            out[f"future_industry_rank_{int(horizon)}d"] = np.asarray(
                labels["industry_rank_by_horizon"][sample_pos, risk_pos],
                dtype=np.float32,
            )
    return pd.DataFrame(out)


def build_shortline_stage0_diagnostics(
    *,
    manifest_json: str | Path = DEFAULT_TRAINING_PACK_MANIFEST,
    output_root: str | Path | None = None,
    run_tag: str = "shortline_stage0_diagnostics",
    roles: str | Iterable[str] | None = None,
    max_dates_per_role: int = 0,
    top_ks: str | Iterable[int] | None = None,
    label_horizons: str | Iterable[int] | None = None,
    score_columns: str | Iterable[str] | None = None,
    round_trip_cost_bps: float = 20.0,
    filter_entry: bool = True,
    leaderboard_limit: int = 20,
    write_scored_sample: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    if str(manifest.get("artifact_type", "")) != "qdp_training_pack_v1":
        raise ValueError(f"Stage 0 shortline diagnostics require qdp_training_pack_v1: {manifest_path}")
    resolved_roles = _parse_csv_strings(roles, default=DEFAULT_ROLES)
    resolved_top_ks = _parse_csv_ints(top_ks, default=DEFAULT_TOP_KS)
    resolved_horizons = _parse_csv_ints(label_horizons, default=DEFAULT_LABEL_HORIZONS)
    resolved_scores = _parse_csv_strings(score_columns, default=DEFAULT_SCORE_COLUMNS)
    out_root = Path(output_root) if output_root is not None else Path("daily_research/output/path_policy/shortline_stage0") / str(run_tag)
    out_root.mkdir(parents=True, exist_ok=True)

    sample_index_path = _resolve_path(str(manifest.get("sample_index_path", "") or ""), root=root)
    sample_index = pd.read_parquet(sample_index_path)
    sample_index["date"] = pd.to_datetime(sample_index["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    feature_columns = [str(item) for item in list(manifest.get("feature_columns", []) or [])]
    feature_indices = {column: idx for idx, column in enumerate(feature_columns)}
    missing_mechanism_features = [column for column in MECHANISM_FEATURES if column not in feature_indices]
    feature_panel = _load_feature_panel(manifest, root=root)
    labels = {
        "cumulative_return_1to20": _open_label_array(manifest, "cumulative_return_1to20", root=root),
        "cumulative_excess_return_1to20": _open_label_array(manifest, "cumulative_excess_return_1to20", root=root),
        "rank_1to20": _open_label_array(manifest, "rank_1to20", root=root),
        "industry_rank_by_horizon": _open_label_array(manifest, "industry_rank_by_horizon", root=root),
        "entry_tradeable": _open_label_array(manifest, "entry_tradeable", root=root),
        "entry_limit_up_buy_blocked": _open_label_array(manifest, "entry_limit_up_buy_blocked", root=root),
        "entry_suspended_or_no_open": _open_label_array(manifest, "entry_suspended_or_no_open", root=root),
    }
    cumulative_horizons = tuple(int(item) for item in list(manifest.get("cumulative_horizons", []) or []))
    horizon_to_risk_pos = {int(horizon): pos for pos, horizon in enumerate(cumulative_horizons)}

    selected_dates_by_role = {
        role: set(_choose_dates(sample_index, role=role, max_dates=int(max_dates_per_role)))
        for role in resolved_roles
    }
    mask = sample_index["role"].astype(str).isin(set(resolved_roles))
    if int(max_dates_per_role) > 0:
        date_allowed = pd.Series(False, index=sample_index.index)
        for role, dates in selected_dates_by_role.items():
            date_allowed |= sample_index["role"].astype(str).eq(role) & sample_index["date"].isin(dates)
        mask &= date_allowed
    selected_index = sample_index.loc[mask].copy()
    if selected_index.empty:
        raise ValueError(f"no samples selected for roles={resolved_roles}, max_dates_per_role={max_dates_per_role}")

    daily_parts: list[pd.DataFrame] = []
    scored_sample_parts: list[pd.DataFrame] = []
    progress_path = out_root / "shortline_stage0_progress.json"
    grouped = list(selected_index.groupby(["role", "date"], sort=True))
    processed_groups = 0
    processed_rows = 0
    for (_, _), group in grouped:
        date_frame = _date_frame_from_pack(
            group=group,
            feature_panel=feature_panel,
            feature_columns=feature_columns,
            feature_indices=feature_indices,
            labels=labels,
            label_horizons=resolved_horizons,
            horizon_to_risk_pos=horizon_to_risk_pos,
        )
        scored = add_shortline_mechanism_scores(date_frame)
        daily = _build_daily_rows(
            scored,
            score_columns=resolved_scores,
            top_ks=resolved_top_ks,
            label_horizons=resolved_horizons,
            round_trip_cost_bps=float(round_trip_cost_bps),
            filter_entry=bool(filter_entry),
        )
        if not daily.empty:
            daily_parts.append(daily)
        if bool(write_scored_sample):
            scored_sample_parts.append(
                scored[
                    [
                        "role",
                        "date",
                        "stock",
                        "entry_tradeable",
                        "entry_limit_up_buy_blocked",
                        "entry_suspended_or_no_open",
                        *[column for column in DEFAULT_SCORE_COLUMNS if column in scored.columns],
                    ]
                ].copy()
            )
        processed_groups += 1
        processed_rows += int(len(group))
        if processed_groups % 100 == 0 or processed_groups == len(grouped):
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "processed_groups": int(processed_groups),
                    "total_groups": int(len(grouped)),
                    "processed_rows": int(processed_rows),
                    "selected_rows": int(len(selected_index)),
                },
            )
    daily_all = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    monthly = _monthly_rows(daily_all)
    summary = _summary_rows(daily_all, monthly)
    daily_csv = _write_frame(out_root / "shortline_stage0_daily.csv", daily_all)
    monthly_csv = _write_frame(out_root / "shortline_stage0_monthly.csv", monthly)
    summary_csv = _write_frame(out_root / "shortline_stage0_summary.csv", summary)
    scored_sample_csv = ""
    if scored_sample_parts:
        scored_sample_csv = _write_frame(out_root / "shortline_stage0_scored_sample.csv", pd.concat(scored_sample_parts, ignore_index=True))
    report = {
        "status": "completed" if not summary.empty else "empty",
        "created_at": _now(),
        "run_tag": str(run_tag),
        "manifest_json": str(manifest_path.resolve()),
        "output_root": str(out_root.resolve()),
        "contract": {
            "diagnostic_kind": "shortline_after_close_stage0_fixed_next_open",
            "research_only": True,
            "not_model_training": True,
            "active_artifact_impact": "unchanged",
            "execution_mode": str(manifest.get("execution_mode", "")),
            "entry_semantics": "D close signal; D+1 open fixed-entry baseline; optional entry-tradeability filter",
            "t_plus_1_semantics": "label_horizon=1 means D+1 open entry to D+2 open exit; label_horizon=2 means D+3 open exit; label_horizon=4 means D+5 open exit",
            "do_not_claim": "D+1 open to D+1 close is not executable under A-share T+1",
            "feature_semantics": "same-date cross-sectional ranks over normalized alpha_v2 features; absolute thresholds are intentionally avoided",
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "filter_entry": bool(filter_entry),
            "roles": list(resolved_roles),
            "top_ks": [int(item) for item in resolved_top_ks],
            "label_horizons": [int(item) for item in resolved_horizons],
            "score_columns": list(resolved_scores),
            "max_dates_per_role": int(max_dates_per_role),
        },
        "input": {
            "selected_row_count": int(len(selected_index)),
            "selected_group_count": int(len(grouped)),
            "roles": list(resolved_roles),
            "selected_dates_by_role": {role: sorted(dates) for role, dates in selected_dates_by_role.items()},
            "feature_count": int(len(feature_columns)),
            "mechanism_feature_count": int(len(MECHANISM_FEATURES) - len(missing_mechanism_features)),
            "missing_mechanism_features": missing_mechanism_features,
            "label_schema_name": str(manifest.get("label_schema_name", "")),
            "label_schema_version": int(manifest.get("label_schema_version", 0) or 0),
        },
        "outputs": {
            "daily_csv": daily_csv,
            "monthly_csv": monthly_csv,
            "summary_csv": summary_csv,
            "scored_sample_csv": scored_sample_csv,
        },
        "leaderboard": _leaderboard(summary, limit=int(leaderboard_limit)),
    }
    report_json = _write_json(out_root / "shortline_stage0_report.json", report)
    report["outputs"]["report_json"] = report_json
    _write_markdown(out_root / "shortline_stage0_report.md", report)
    report["outputs"]["report_md"] = str((out_root / "shortline_stage0_report.md").resolve())
    _write_json(out_root / "shortline_stage0_report.json", report)
    _write_json(progress_path, {"status": "completed", "report_json": report_json})
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Shortline Stage 0 Diagnostics",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Research-only: `{report.get('contract', {}).get('research_only', True)}`",
        f"- Not model training: `{report.get('contract', {}).get('not_model_training', True)}`",
        f"- Round-trip cost bps: `{report.get('contract', {}).get('round_trip_cost_bps', '')}`",
        f"- Entry filter: `{report.get('contract', {}).get('filter_entry', '')}`",
        "",
        "## Semantics",
        "",
        f"- Entry: {report.get('contract', {}).get('entry_semantics', '')}",
        f"- T+1: {report.get('contract', {}).get('t_plus_1_semantics', '')}",
        f"- Boundary: {report.get('contract', {}).get('do_not_claim', '')}",
        "",
        "## Input",
        "",
        f"- Rows: `{report.get('input', {}).get('selected_row_count', 0)}`",
        f"- Date groups: `{report.get('input', {}).get('selected_group_count', 0)}`",
        f"- Roles: `{','.join(report.get('input', {}).get('roles', []))}`",
        "",
        "## Leaderboard",
        "",
    ]
    leaderboard = list(report.get("leaderboard", []) or [])
    if leaderboard:
        lines.append("| role | mechanism | topK | exit | net_abs | per_day | net_excess | hit | month_pos | blocked |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in leaderboard[:20]:
            lines.append(
                "| {role} | {mechanism} | {top_k} | {exit_label} | {net_abs:.4f} | {per_day:.4f} | {net_excess:.4f} | {hit:.3f} | {month_pos:.3f} | {blocked:.3f} |".format(
                    role=row.get("role", ""),
                    mechanism=row.get("mechanism", ""),
                    top_k=int(row.get("top_k", 0) or 0),
                    exit_label=row.get("exit_label", ""),
                    net_abs=float(row.get("net_abs_mean", 0.0) or 0.0),
                    per_day=float(row.get("net_abs_mean_per_day", 0.0) or 0.0),
                    net_excess=float(row.get("net_excess_mean", 0.0) or 0.0),
                    hit=float(row.get("abs_hit_rate_mean", 0.0) or 0.0),
                    month_pos=float(row.get("net_abs_positive_month_rate", 0.0) or 0.0),
                    blocked=float(row.get("candidate_blocked_rate_mean", 0.0) or 0.0),
                )
            )
    else:
        lines.append("- No usable rows.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Daily CSV: `{report.get('outputs', {}).get('daily_csv', '')}`",
            f"- Monthly CSV: `{report.get('outputs', {}).get('monthly_csv', '')}`",
            f"- Summary CSV: `{report.get('outputs', {}).get('summary_csv', '')}`",
            f"- Report JSON: `{report.get('outputs', {}).get('report_json', '')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 0 after-close shortline fixed-next-open diagnostics over a QDP training pack.")
    parser.add_argument("--manifest-json", default=str(DEFAULT_TRAINING_PACK_MANIFEST))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-tag", default="shortline_stage0_diagnostics")
    parser.add_argument("--roles", default="validation,test")
    parser.add_argument("--max-dates-per-role", type=int, default=0)
    parser.add_argument("--top-ks", default="1,3,5,10")
    parser.add_argument("--label-horizons", default="1,2,4")
    parser.add_argument("--score-columns", default="")
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--filter-entry", dest="filter_entry", action="store_true", default=True)
    parser.add_argument("--no-filter-entry", dest="filter_entry", action="store_false")
    parser.add_argument("--leaderboard-limit", type=int, default=20)
    parser.add_argument("--write-scored-sample", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_shortline_stage0_diagnostics(
        manifest_json=args.manifest_json,
        output_root=args.output_root or None,
        run_tag=args.run_tag,
        roles=args.roles,
        max_dates_per_role=int(args.max_dates_per_role),
        top_ks=args.top_ks,
        label_horizons=args.label_horizons,
        score_columns=args.score_columns or None,
        round_trip_cost_bps=float(args.round_trip_cost_bps),
        filter_entry=bool(args.filter_entry),
        leaderboard_limit=int(args.leaderboard_limit),
        write_scored_sample=bool(args.write_scored_sample),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(report.get("outputs", {}).get("report_json", ""))
    return report


if __name__ == "__main__":
    main()
