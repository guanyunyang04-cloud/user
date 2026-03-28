from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
import pickle

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
    parse_named_windows,
    parse_stock_list,
)
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_industry_map_from_tq, load_style_map_from_tq, load_universe_from_tq
from daily_research.baseline.diagnose_advanced_ml_ensemble import DEFAULT_FOCUS_STATE, _run_candidate
from daily_research.baseline.ml_alpha import (
    EXPANDED_MARKET_FEATURE_PROFILE,
    LEGACY_MARKET_FEATURE_PROFILE,
    MLAplhaConfig,
    SUPPORTED_MARKET_FEATURE_PROFILES,
    build_ml_feature_bundle,
    combine_per_horizon_ml_scores,
    normalize_market_feature_profile,
    rolling_ml_scores_multi_detail,
    select_market_feature_profile,
)
from daily_research.baseline.scan_advanced_ml_attack_defense_controller import (
    _build_dynamic_candidates,
    _build_static_candidates,
    _parse_weight_triplet,
    _run_dynamic_candidate,
)
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership
from daily_research.progress import StageProgress


DEFAULT_WINDOWS = "recent_full:20250307:20260327,weak_window_20250905_20260319:20250905:20260319"
DEFAULT_PROFILES = f"{LEGACY_MARKET_FEATURE_PROFILE},{EXPANDED_MARKET_FEATURE_PROFILE}"
DEFAULT_DYNAMIC_OFFENSE = "ml:0.25,none:0.22,v2:0.53"
DEFAULT_DYNAMIC_DEFENSE = "ml:0.25,none:0.235,v2:0.515"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formally compare legacy 7-feature and expanded 24-feature market feature profiles under the same advanced_ml execution stack."
    )
    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--stocks", default=None)
    parser.add_argument("--stocks-file", default=None)
    parser.add_argument("--start-date", default="20190101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-scope", default="all_a")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument(
        "--rolling-liquidity-pool",
        choices=["liquid300", "liquid500", "liquid800"],
        default="liquid500",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--enhanced-profile", default="up_low_breakout_v2")
    parser.add_argument("--focus-state", default=DEFAULT_FOCUS_STATE)
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument("--anchor-profile", default="")
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--output-dir", default="")

    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.0)

    parser.add_argument("--no-market-regime-filter", action="store_true")
    parser.add_argument("--regime-ma-window", type=int, default=50)
    parser.add_argument("--regime-vol-window", type=int, default=20)
    parser.add_argument("--regime-max-annual-vol", type=float, default=0.32)
    parser.add_argument("--regime-quadrants", default="trend_up_low_vol,trend_up_high_vol")

    parser.add_argument("--no-style-cap", action="store_true")
    parser.add_argument("--max-style-weight", type=float, default=0.50)
    parser.add_argument("--industry-cap", action="store_true")
    parser.add_argument("--max-industry-weight", type=float, default=0.40)

    parser.add_argument("--ml-target-horizon", type=int, default=20)
    parser.add_argument("--ml-target-horizons", default="5,10,20")
    parser.add_argument("--ml-horizon-weights", default="5:0.2,10:0.3,20:0.5")
    parser.add_argument("--ml-train-window-days", type=int, default=504)
    parser.add_argument("--ml-retrain-every-days", type=int, default=21)
    parser.add_argument("--ml-min-train-dates", type=int, default=120)
    parser.add_argument("--ml-max-samples-per-day", type=int, default=600)
    parser.add_argument("--ml-max-train-rows", type=int, default=200000)
    parser.add_argument("--ml-random-seed", type=int, default=7)
    parser.add_argument("--ml-model-family", choices=["lgbm"], default="lgbm")
    parser.add_argument("--lgbm-n-estimators", type=int, default=520)

    parser.add_argument("--ensemble-ml-weight", type=float, default=0.70)
    parser.add_argument("--ensemble-none-weight", type=float, default=0.20)
    parser.add_argument("--ensemble-v2-weight", type=float, default=0.10)
    parser.add_argument("--offense-state-weights", default="ml:0.25,none:0.20,v2:0.55")
    parser.add_argument("--defense-state-weights", default="ml:0.25,none:0.25,v2:0.50")

    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--dynamic-offense-state-weights", default=DEFAULT_DYNAMIC_OFFENSE)
    parser.add_argument("--dynamic-defense-state-weights", default=DEFAULT_DYNAMIC_DEFENSE)
    parser.add_argument("--dynamic-offense-trend-gap-min", type=float, default=0.024192)
    parser.add_argument("--dynamic-offense-max-annual-vol", type=float, default=0.176128)
    parser.add_argument("--dynamic-offense-benchmark-ret10-min", type=float, default=0.014717)

    parser.add_argument("--windows", default=DEFAULT_WINDOWS)
    parser.add_argument("--auto-trim-history", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def _resolve_weak_window_name(windows: list[tuple[str, str, str]]) -> str:
    for name, _, _ in windows:
        if str(name).startswith("weak_window_"):
            return str(name)
    raise ValueError("windows must include one weak_window_* entry")


def _subset_raw_df_dict_to_stocks(
    raw_df_dict: dict[str, pd.DataFrame],
    benchmark: str,
    stocks: list[str],
) -> dict[str, pd.DataFrame]:
    keep = [benchmark] + [stock for stock in stocks if stock != benchmark]
    out: dict[str, pd.DataFrame] = {}
    for field, frame in raw_df_dict.items():
        cols = [col for col in keep if col in frame.columns]
        out[field] = frame.reindex(columns=cols)
    return out


def _load_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _profile_ml_score_cache_path(prepared_cache_key: str, ml_cfg: MLAplhaConfig, market_feature_profile: str) -> Path:
    payload = {
        "kind": "market_feature_profile_shared_ml_scores",
        "score_engine_version": 1,
        "prepared_cache_key": str(prepared_cache_key),
        "market_feature_profile": str(market_feature_profile),
        "ml_config": {
            "target_horizon": int(ml_cfg.target_horizon),
            "target_horizons": list(ml_cfg.target_horizons),
            "target_horizon_weights": {str(k): float(v) for k, v in (ml_cfg.target_horizon_weights or {}).items()},
            "train_window_days": int(ml_cfg.train_window_days),
            "retrain_every_days": int(ml_cfg.retrain_every_days),
            "min_train_dates": int(ml_cfg.min_train_dates),
            "max_samples_per_day": int(ml_cfg.max_samples_per_day),
            "max_train_rows": int(ml_cfg.max_train_rows),
            "random_seed": int(ml_cfg.random_seed),
            "model_family": str(ml_cfg.model_family),
            "lgbm_n_estimators": int(ml_cfg.lgbm_n_estimators),
            "train_regime_only": bool(ml_cfg.train_regime_only),
            "execution_mode": str(ml_cfg.execution_mode),
        },
    }
    cache_key = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    cache_dir = Path("daily_research/cache/advanced_ml/ml_scores")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.pkl"


def _safe_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.3f}"


def _safe_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "nan"
    return f"{float(value):.2%}"


def _parse_profiles(raw: str) -> list[str]:
    profiles: list[str] = []
    seen: set[str] = set()
    for item in parse_csv_list(raw):
        normalized = normalize_market_feature_profile(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        profiles.append(normalized)
    if not profiles:
        profiles = [EXPANDED_MARKET_FEATURE_PROFILE]
    return profiles


def _resolve_anchor_profile(anchor_profile: str, profiles: list[str]) -> str:
    raw = str(anchor_profile or "").strip()
    if raw:
        return normalize_market_feature_profile(raw)
    if len(profiles) <= 2:
        return ""
    if EXPANDED_MARKET_FEATURE_PROFILE in profiles:
        return EXPANDED_MARKET_FEATURE_PROFILE
    return str(profiles[0])


def _metric_fields(weak_window_name: str, focus_state: str) -> list[str]:
    return [
        "full_annual_return",
        "full_excess_annual_return",
        "full_excess_sharpe",
        "recent_full_annual_return",
        "recent_full_excess_annual_return",
        "recent_full_excess_sharpe",
        f"{weak_window_name}_annual_return",
        f"{weak_window_name}_excess_annual_return",
        f"{weak_window_name}_excess_sharpe",
        f"{focus_state}_{weak_window_name}_excess_sharpe",
    ]


def _build_base_candidate(base_weights: dict[str, float]) -> dict[str, Any]:
    return {
        "label": "base_global",
        "blend_kind": "base_control",
        "weights": dict(base_weights),
        "state_ensemble_weights": {},
        "distance_to_default": 0.0,
        "profile": "base",
    }


def _make_pairwise_df(
    results_df: pd.DataFrame,
    profiles: list[str],
    weak_window_name: str,
    focus_state: str,
) -> pd.DataFrame:
    if len(profiles) != 2:
        return pd.DataFrame()

    left_profile, right_profile = profiles
    metrics = _metric_fields(weak_window_name, focus_state)

    rows: list[dict[str, Any]] = []
    for label, label_df in results_df.groupby("label", sort=False):
        left_df = label_df.loc[label_df["market_feature_profile"].eq(left_profile)]
        right_df = label_df.loc[label_df["market_feature_profile"].eq(right_profile)]
        if left_df.empty or right_df.empty:
            continue
        left_row = left_df.iloc[0]
        right_row = right_df.iloc[0]
        row = {
            "label": label,
            "blend_kind": str(left_row["blend_kind"]),
            "left_profile": left_profile,
            "right_profile": right_profile,
            "left_feature_count": int(left_row["market_feature_count"]),
            "right_feature_count": int(right_row["market_feature_count"]),
        }
        for metric in metrics:
            left_value = pd.to_numeric(pd.Series([left_row.get(metric)]), errors="coerce").iloc[0]
            right_value = pd.to_numeric(pd.Series([right_row.get(metric)]), errors="coerce").iloc[0]
            row[f"{left_profile}_{metric}"] = left_value
            row[f"{right_profile}_{metric}"] = right_value
            row[f"delta_{right_profile}_minus_{left_profile}_{metric}"] = (
                float(right_value - left_value) if pd.notna(left_value) and pd.notna(right_value) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _make_anchor_comparison_df(
    results_df: pd.DataFrame,
    anchor_profile: str,
    weak_window_name: str,
    focus_state: str,
) -> pd.DataFrame:
    if not anchor_profile:
        return pd.DataFrame()

    metrics = _metric_fields(weak_window_name, focus_state)
    rows: list[dict[str, Any]] = []
    for label, label_df in results_df.groupby("label", sort=False):
        anchor_df = label_df.loc[label_df["market_feature_profile"].eq(anchor_profile)]
        if anchor_df.empty:
            continue
        anchor_row = anchor_df.iloc[0]
        for profile, profile_df in label_df.groupby("market_feature_profile", sort=False):
            if str(profile) == anchor_profile:
                continue
            compare_row = profile_df.iloc[0]
            row = {
                "label": label,
                "blend_kind": str(compare_row["blend_kind"]),
                "anchor_profile": anchor_profile,
                "compare_profile": str(profile),
                "anchor_feature_count": int(anchor_row["market_feature_count"]),
                "compare_feature_count": int(compare_row["market_feature_count"]),
            }
            for metric in metrics:
                anchor_value = pd.to_numeric(pd.Series([anchor_row.get(metric)]), errors="coerce").iloc[0]
                compare_value = pd.to_numeric(pd.Series([compare_row.get(metric)]), errors="coerce").iloc[0]
                row[f"anchor_{metric}"] = anchor_value
                row[f"compare_{metric}"] = compare_value
                row[f"delta_{metric}"] = (
                    float(compare_value - anchor_value) if pd.notna(anchor_value) and pd.notna(compare_value) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _render_summary(
    *,
    latest_data_date: str,
    history_window: dict[str, Any],
    results_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    anchor_comparison_df: pd.DataFrame,
    profiles: list[str],
    weak_window_name: str,
    focus_state: str,
    dynamic_enabled: bool,
    anchor_profile: str,
) -> str:
    lines: list[str] = []
    lines.append("# Market Feature Profile Comparator")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- latest_data_date: {latest_data_date}")
    lines.append(
        f"- history_window: {history_window['effective_start_date']} -> {history_window['end_date']} (mode={history_window['mode']})"
    )
    lines.append(f"- profiles: {profiles}")
    if anchor_profile:
        lines.append(f"- anchor_profile: {anchor_profile}")
    lines.append(f"- focus_state: {focus_state}")
    lines.append("")

    lines.append("## Best candidate by profile")
    for profile in profiles:
        profile_df = results_df.loc[results_df["market_feature_profile"].eq(profile)].copy()
        if profile_df.empty:
            continue
        ranked = profile_df.sort_values(
            ["full_excess_sharpe", f"{weak_window_name}_excess_sharpe", f"{focus_state}_{weak_window_name}_excess_sharpe"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        for _, row in ranked.head(3).iterrows():
            lines.append(
                f"- {profile} | {row['label']}: full excess Sharpe {_safe_num(row['full_excess_sharpe'])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(row[f'{weak_window_name}_excess_sharpe'])}, "
                f"{focus_state}_{weak_window_name} excess Sharpe {_safe_num(row[f'{focus_state}_{weak_window_name}_excess_sharpe'])}, "
                f"feature_count {int(row['market_feature_count'])}"
            )
    lines.append("")

    if not pairwise_df.empty:
        left_profile, right_profile = profiles
        delta_full = f"delta_{right_profile}_minus_{left_profile}_full_excess_sharpe"
        delta_weak = f"delta_{right_profile}_minus_{left_profile}_{weak_window_name}_excess_sharpe"
        delta_focus_weak = f"delta_{right_profile}_minus_{left_profile}_{focus_state}_{weak_window_name}_excess_sharpe"
        delta_full_ret = f"delta_{right_profile}_minus_{left_profile}_full_excess_annual_return"

        lines.append("## Candidate-by-candidate deltas")
        ordered = pairwise_df.sort_values(["blend_kind", "label"], ascending=[True, True]).reset_index(drop=True)
        for _, row in ordered.iterrows():
            lines.append(
                f"- {row['label']}: {right_profile} - {left_profile} | "
                f"full excess Sharpe {_safe_num(row[delta_full])}, "
                f"full excess annual {_safe_pct(row[delta_full_ret])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(row[delta_weak])}, "
                f"{focus_state}_{weak_window_name} excess Sharpe {_safe_num(row[delta_focus_weak])}"
            )
        lines.append("")

        live_row = pairwise_df.loc[pairwise_df["label"].eq("trend_up_low_vol_ml25_none25_v250")]
        offense_row = pairwise_df.loc[pairwise_df["label"].eq("trend_up_low_vol_ml25_none20_v255")]
        if not live_row.empty and not offense_row.empty:
            live_row = live_row.iloc[0]
            offense_row = offense_row.iloc[0]
            lines.append("## Direct answer")
            lines.append(
                f"- Live default `v250`: {right_profile} - {left_profile} = "
                f"full excess Sharpe {_safe_num(live_row[delta_full])}, "
                f"full excess annual {_safe_pct(live_row[delta_full_ret])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(live_row[delta_weak])}."
            )
            lines.append(
                f"- Offense comparator `v255`: {right_profile} - {left_profile} = "
                f"full excess Sharpe {_safe_num(offense_row[delta_full])}, "
                f"full excess annual {_safe_pct(offense_row[delta_full_ret])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(offense_row[delta_weak])}."
            )
            if dynamic_enabled:
                dynamic_rows = pairwise_df.loc[pairwise_df["blend_kind"].eq("dynamic_controller")]
                if not dynamic_rows.empty:
                    best_dynamic = dynamic_rows.sort_values([delta_full, delta_weak], ascending=[False, False]).iloc[0]
                    lines.append(
                        f"- Strongest dynamic delta row is `{best_dynamic['label']}`: "
                        f"full excess Sharpe {_safe_num(best_dynamic[delta_full])}, "
                        f"{weak_window_name} excess Sharpe {_safe_num(best_dynamic[delta_weak])}."
                    )
            lines.append("")

    if not anchor_comparison_df.empty:
        lines.append(f"## Anchor deltas vs `{anchor_profile}`")
        ordered = anchor_comparison_df.sort_values(["blend_kind", "label", "compare_profile"], ascending=[True, True, True])
        for _, row in ordered.iterrows():
            lines.append(
                f"- {row['label']} | {row['compare_profile']} - {anchor_profile}: "
                f"full excess Sharpe {_safe_num(row['delta_full_excess_sharpe'])}, "
                f"full excess annual {_safe_pct(row['delta_full_excess_annual_return'])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(row[f'delta_{weak_window_name}_excess_sharpe'])}, "
                f"{focus_state}_{weak_window_name} excess Sharpe {_safe_num(row[f'delta_{focus_state}_{weak_window_name}_excess_sharpe'])}"
            )
        lines.append("")

        offense_label = "trend_up_low_vol_ml25_none20_v255"
        live_label = "trend_up_low_vol_ml25_none25_v250"
        offense_df = results_df.loc[results_df["label"].eq(offense_label)].copy()
        live_df = results_df.loc[results_df["label"].eq(live_label)].copy()
        if not offense_df.empty and not live_df.empty:
            best_offense = offense_df.sort_values(
                ["full_excess_sharpe", f"{weak_window_name}_excess_sharpe", f"{focus_state}_{weak_window_name}_excess_sharpe"],
                ascending=[False, False, False],
            ).iloc[0]
            best_live = live_df.sort_values(
                [f"{weak_window_name}_excess_sharpe", f"{focus_state}_{weak_window_name}_excess_sharpe", "full_excess_sharpe"],
                ascending=[False, False, False],
            ).iloc[0]
            lines.append("## Frontier read")
            lines.append(
                f"- Best offense profile for `v255` is `{best_offense['market_feature_profile']}`: "
                f"full excess Sharpe {_safe_num(best_offense['full_excess_sharpe'])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(best_offense[f'{weak_window_name}_excess_sharpe'])}, "
                f"{focus_state}_{weak_window_name} excess Sharpe {_safe_num(best_offense[f'{focus_state}_{weak_window_name}_excess_sharpe'])}."
            )
            lines.append(
                f"- Best defense/live profile for `v250` is `{best_live['market_feature_profile']}`: "
                f"full excess Sharpe {_safe_num(best_live['full_excess_sharpe'])}, "
                f"{weak_window_name} excess Sharpe {_safe_num(best_live[f'{weak_window_name}_excess_sharpe'])}, "
                f"{focus_state}_{weak_window_name} excess Sharpe {_safe_num(best_live[f'{focus_state}_{weak_window_name}_excess_sharpe'])}."
            )
            if str(best_offense["market_feature_profile"]) != anchor_profile:
                matched_live = live_df.loc[live_df["market_feature_profile"].eq(best_offense["market_feature_profile"])]
                if not matched_live.empty:
                    matched_live = matched_live.iloc[0]
                    lines.append(
                        f"- Under the same `{best_offense['market_feature_profile']}` profile, live `v250` becomes: "
                        f"full excess Sharpe {_safe_num(matched_live['full_excess_sharpe'])}, "
                        f"{weak_window_name} excess Sharpe {_safe_num(matched_live[f'{weak_window_name}_excess_sharpe'])}, "
                        f"{focus_state}_{weak_window_name} excess Sharpe {_safe_num(matched_live[f'{focus_state}_{weak_window_name}_excess_sharpe'])}."
                    )
            if dynamic_enabled:
                dynamic_df = results_df.loc[results_df["blend_kind"].eq("dynamic_controller")].copy()
                if not dynamic_df.empty:
                    best_dynamic = dynamic_df.sort_values(
                        ["full_excess_sharpe", f"{weak_window_name}_excess_sharpe"],
                        ascending=[False, False],
                    ).iloc[0]
                    lines.append(
                        f"- Best dynamic-controller profile is `{best_dynamic['market_feature_profile']}` via `{best_dynamic['label']}`: "
                        f"full excess Sharpe {_safe_num(best_dynamic['full_excess_sharpe'])}, "
                        f"{weak_window_name} excess Sharpe {_safe_num(best_dynamic[f'{weak_window_name}_excess_sharpe'])}."
                    )
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
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

    profiles = _parse_profiles(args.profiles)
    anchor_profile = _resolve_anchor_profile(args.anchor_profile, profiles)
    if anchor_profile and anchor_profile not in profiles:
        profiles.append(anchor_profile)
    invalid_profiles = [profile for profile in profiles if profile not in SUPPORTED_MARKET_FEATURE_PROFILES]
    if invalid_profiles:
        raise ValueError(f"Unsupported profiles: {invalid_profiles}")

    base_weights = {
        "ml": float(args.ensemble_ml_weight),
        "none": float(args.ensemble_none_weight),
        "v2": float(args.ensemble_v2_weight),
    }
    offense_weights = _parse_weight_triplet(args.offense_state_weights)
    defense_weights = _parse_weight_triplet(args.defense_state_weights)
    dynamic_offense_weights = _parse_weight_triplet(args.dynamic_offense_state_weights)
    dynamic_defense_weights = _parse_weight_triplet(args.dynamic_defense_state_weights)
    windows = parse_named_windows(args.windows)
    weak_window_name = _resolve_weak_window_name(windows)
    focus_state = str(args.focus_state).strip().lower() or DEFAULT_FOCUS_STATE

    ml_cfg = MLAplhaConfig(
        target_horizon=args.ml_target_horizon,
        target_horizons=parse_int_tuple(args.ml_target_horizons, args.ml_target_horizon),
        target_horizon_weights=parse_horizon_weights(args.ml_horizon_weights),
        enhanced_profile=args.enhanced_profile,
        train_window_days=args.ml_train_window_days,
        retrain_every_days=args.ml_retrain_every_days,
        min_train_dates=args.ml_min_train_dates,
        max_samples_per_day=args.ml_max_samples_per_day,
        max_train_rows=args.ml_max_train_rows,
        random_seed=args.ml_random_seed,
        model_family=args.ml_model_family,
        lgbm_n_estimators=args.lgbm_n_estimators,
        ensemble_ml_weight=base_weights["ml"],
        ensemble_none_weight=base_weights["none"],
        ensemble_v2_weight=base_weights["v2"],
        train_regime_only=cfg.enable_market_regime_filter,
        execution_mode=cfg.execution_mode,
    )

    candidates: list[dict[str, Any]] = [_build_base_candidate(base_weights)]
    candidates.extend(
        _build_static_candidates(
            focus_state=focus_state,
            base_weights=base_weights,
            offense_weights=offense_weights,
            defense_weights=defense_weights,
        )
    )
    if not args.skip_dynamic:
        dynamic_candidates = _build_dynamic_candidates(
            focus_state=focus_state,
            base_weights=base_weights,
            offense_weight_grid=[dynamic_offense_weights],
            defense_weight_grid=[dynamic_defense_weights],
            trend_gap_grid=[float(args.dynamic_offense_trend_gap_min)],
            max_vol_grid=[float(args.dynamic_offense_max_annual_vol)],
            benchmark_ret10_grid=[float(args.dynamic_offense_benchmark_ret10_min)],
        )
        candidates.extend(dynamic_candidates[:1])

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("daily_research/output")
        / (args.experiment_tag.strip() or f"market_feature_profile_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    latest_data_date = ""
    history_window = None
    results_rows: list[dict[str, Any]] = []

    with StageProgress(total=8, label="market-feature comparator") as progress:
        with progress.stage("load universe", args.universe_scope):
            if args.data_source == "tq" and not cfg.universe:
                cfg.universe = load_universe_from_tq(cfg.universe_scope)
            if not cfg.universe:
                raise ValueError("Universe is empty after loading.")

        with progress.stage("resolve history", args.start_date):
            history_window = resolve_history_window(
                cfg=cfg,
                ml_cfg=ml_cfg,
                requested_start_date=args.start_date,
                end_date=args.end_date,
                mode="train",
                auto_trim_history=args.auto_trim_history,
            )
            progress.log(
                f"history window: {history_window.effective_start_date} -> {history_window.end_date} "
                f"(required_trading_days={history_window.required_trading_days})"
            )

        with progress.stage("load market data", f"stocks={len(cfg.universe)}"):
            raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
                data_source=args.data_source,
                csv_folder=args.csv_folder,
                universe=cfg.universe,
                benchmark=cfg.benchmark,
                history_window=history_window,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
                progress_desc="load raw bars",
                progress_position=1,
            )
            latest_data_date = history_window.end_date
            progress.log(f"raw cache: {'hit' if raw_cache_meta['cache_hit'] else 'build'} | {raw_cache_meta['cache_path']}")

        with progress.stage("build rolling pool", args.rolling_liquidity_pool or "fixed"):
            rolling_pool_artifact = None
            rolling_membership_mask = None
            prepared_raw_df_dict = raw_df_dict
            if args.rolling_liquidity_pool:
                raw_universe_df_dict = {key: frame.drop(columns=[cfg.benchmark], errors="ignore") for key, frame in raw_df_dict.items()}
                rolling_pool_artifact = build_rolling_liquidity_membership(
                    close_frame=raw_universe_df_dict["Close"],
                    amount_frame=raw_universe_df_dict["Amount"],
                    pool_name=args.rolling_liquidity_pool,
                    signal_start_date=cfg.start_date,
                    signal_end_date=args.end_date,
                    rebalance_every_days=args.pool_rebalance_days,
                    adv_window=args.pool_adv_window,
                    min_price=cfg.min_price,
                    max_price=cfg.max_price,
                )
                rolling_membership_mask = rolling_pool_artifact.membership_frame
                rolling_union = rolling_membership_mask.columns[rolling_membership_mask.any(axis=0)].tolist()
                if not rolling_union:
                    raise RuntimeError(f"Rolling {args.rolling_liquidity_pool} pool is empty for the requested window.")
                prepared_raw_df_dict = _subset_raw_df_dict_to_stocks(raw_df_dict, cfg.benchmark, rolling_union)
                progress.log(f"rolling pool union size: {len(rolling_union)}")

        with progress.stage("build prepared bundle", args.enhanced_profile):
            prepared_raw_cache_key = (
                raw_cache_meta["cache_key"]
                if rolling_pool_artifact is None
                else f"{raw_cache_meta['cache_key']}|{args.rolling_liquidity_pool}|{args.pool_rebalance_days}|{args.pool_adv_window}"
            )
            prepared_bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
                raw_df_dict=prepared_raw_df_dict,
                raw_cache_key=prepared_raw_cache_key,
                cfg=cfg,
                enhanced_profile=args.enhanced_profile,
                use_cache=not args.no_cache,
                refresh_cache=args.refresh_cache,
            )
            progress.log(
                f"prepared cache: {'hit' if prepared_cache_meta['cache_hit'] else 'build'} | {prepared_cache_meta['cache_path']}"
            )
            current_membership_mask = None
            if rolling_membership_mask is not None:
                current_membership_mask = rolling_membership_mask.reindex(
                    index=prepared_bundle["df_dict"]["Close"].index,
                    columns=prepared_bundle["df_dict"]["Close"].columns,
                ).fillna(False)
                prepared_bundle["filter_mask"] = prepared_bundle["filter_mask"] & current_membership_mask
                prepared_bundle["score_none"] = prepared_bundle["score_none"].where(current_membership_mask)
                prepared_bundle["score_v2"] = prepared_bundle["score_v2"].where(current_membership_mask)
                feature_frames, market_features = build_ml_feature_bundle(
                    prepared_bundle["factor_bundle"],
                    prepared_bundle["regime_state"],
                    prepared_bundle["score_none"],
                    prepared_bundle["score_v2"],
                )
                prepared_bundle["feature_frames"] = feature_frames
                prepared_bundle["market_features"] = market_features

        with progress.stage("load exposure maps", "style/industry"):
            candidate_columns = [col for col in prepared_bundle["df_dict"]["Close"].columns if col != cfg.benchmark]
            industry_map = None
            style_map = None
            if cfg.enable_industry_cap and args.data_source == "tq":
                industry_map = load_industry_map_from_tq(candidate_columns)
            if cfg.enable_style_cap and args.data_source == "tq":
                style_map = load_style_map_from_tq(candidate_columns)

        with progress.stage("run feature profiles", ",".join(profiles)):
            training_logs_dir = output_root / "training_logs"
            training_logs_dir.mkdir(parents=True, exist_ok=True)
            for profile in profiles:
                selected_market_features = select_market_feature_profile(prepared_bundle["market_features"], profile)
                progress.log(f"profile {profile}: feature_count={len(selected_market_features)}")
                ml_score_cache_path = _profile_ml_score_cache_path(prepared_cache_meta["cache_key"], ml_cfg, profile)
                cached_ml_payload = None if args.refresh_cache else _load_pickle(ml_score_cache_path)
                shared_training_log = pd.DataFrame()
                shared_per_horizon_scores = None
                if cached_ml_payload is not None and isinstance(cached_ml_payload, dict):
                    shared_training_log = cached_ml_payload.get("training_log", pd.DataFrame())
                    shared_per_horizon_scores = cached_ml_payload.get("per_horizon_scores")
                elif cached_ml_payload is not None:
                    shared_per_horizon_scores = cached_ml_payload

                if shared_per_horizon_scores is None:
                    _, shared_training_log, shared_per_horizon_scores = rolling_ml_scores_multi_detail(
                        feature_frames=prepared_bundle["feature_frames"],
                        market_features=selected_market_features,
                        close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
                        benchmark_close=prepared_bundle["benchmark_close"],
                        filter_mask=prepared_bundle["filter_mask"],
                        regime_state=prepared_bundle["regime_state"],
                        config=ml_cfg,
                        open_df=prepared_bundle["factor_bundle"]["raw_inputs"]["Open"],
                        benchmark_open=prepared_bundle["benchmark_open"],
                    )
                    if not args.no_cache:
                        _save_pickle(
                            ml_score_cache_path,
                            {
                                "training_log": shared_training_log,
                                "per_horizon_scores": shared_per_horizon_scores,
                            },
                        )
                if not shared_training_log.empty:
                    shared_training_log.to_csv(
                        training_logs_dir / f"{profile}_training_log.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )

                ml_score = combine_per_horizon_ml_scores(
                    per_horizon_scores=shared_per_horizon_scores,
                    close=prepared_bundle["factor_bundle"]["raw_inputs"]["Close"],
                    regime_state=prepared_bundle["regime_state"],
                    config=ml_cfg,
                )
                if current_membership_mask is not None:
                    ml_score = ml_score.where(current_membership_mask)

                for candidate in candidates:
                    if str(candidate["blend_kind"]) == "dynamic_controller":
                        row, _ = _run_dynamic_candidate(
                            candidate=candidate,
                            cfg=cfg,
                            ml_score=ml_score,
                            prepared_bundle=prepared_bundle,
                            current_membership_mask=current_membership_mask,
                            industry_map=industry_map,
                            style_map=style_map,
                            windows=windows,
                            focus_state=focus_state,
                        )
                    else:
                        row, _ = _run_candidate(
                            candidate=candidate,
                            cfg=cfg,
                            ml_cfg=ml_cfg,
                            shared_per_horizon_scores=shared_per_horizon_scores,
                            prepared_bundle=prepared_bundle,
                            current_membership_mask=current_membership_mask,
                            industry_map=industry_map,
                            style_map=style_map,
                            windows=windows,
                            focus_state=focus_state,
                        )
                    row["market_feature_profile"] = profile
                    row["market_feature_count"] = int(len(selected_market_features))
                    row["market_feature_names"] = json.dumps(sorted(selected_market_features), ensure_ascii=False)
                    results_rows.append(row)

        with progress.stage("write outputs", output_root.name):
            results_df = pd.DataFrame(results_rows)
            profile_order = {profile: idx for idx, profile in enumerate(profiles)}
            blend_order = {
                "base_control": 0,
                "shortlist_static": 1,
                "dynamic_controller": 2,
            }
            results_df["profile_sort_order"] = results_df["market_feature_profile"].map(profile_order).fillna(99)
            results_df["blend_sort_order"] = results_df["blend_kind"].map(blend_order).fillna(99)
            results_df = results_df.sort_values(
                ["blend_sort_order", "label", "profile_sort_order"],
                ascending=[True, True, True],
            ).reset_index(drop=True)
            pairwise_df = (
                _make_pairwise_df(results_df, profiles, weak_window_name, focus_state)
                if not anchor_profile
                else pd.DataFrame()
            )
            anchor_comparison_df = _make_anchor_comparison_df(
                results_df,
                anchor_profile=anchor_profile,
                weak_window_name=weak_window_name,
                focus_state=focus_state,
            )
            summary_text = _render_summary(
                latest_data_date=latest_data_date,
                history_window=history_window_to_dict(history_window),
                results_df=results_df,
                pairwise_df=pairwise_df,
                anchor_comparison_df=anchor_comparison_df,
                profiles=profiles,
                weak_window_name=weak_window_name,
                focus_state=focus_state,
                dynamic_enabled=not args.skip_dynamic,
                anchor_profile=anchor_profile,
            )

            results_df.to_csv(output_root / "profile_results.csv", index=False, encoding="utf-8-sig")
            if not pairwise_df.empty:
                pairwise_df.to_csv(output_root / "pairwise_comparison.csv", index=False, encoding="utf-8-sig")
            if not anchor_comparison_df.empty:
                anchor_comparison_df.to_csv(output_root / "anchor_comparison.csv", index=False, encoding="utf-8-sig")

            verdict = {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "latest_data_date": latest_data_date,
                "history_window": history_window_to_dict(history_window),
                "profiles": profiles,
                "anchor_profile": anchor_profile,
                "focus_state": focus_state,
                "candidate_labels": results_df["label"].tolist(),
                "dynamic_enabled": bool(not args.skip_dynamic),
                "profile_best_by_full_excess_sharpe": [],
                "pairwise": json.loads(pairwise_df.to_json(orient="records", force_ascii=False)) if not pairwise_df.empty else [],
                "anchor_comparison": (
                    json.loads(anchor_comparison_df.to_json(orient="records", force_ascii=False))
                    if not anchor_comparison_df.empty
                    else []
                ),
            }
            for profile in profiles:
                profile_df = results_df.loc[results_df["market_feature_profile"].eq(profile)]
                if profile_df.empty:
                    continue
                best_row = profile_df.sort_values(
                    ["full_excess_sharpe", f"{weak_window_name}_excess_sharpe", f"{focus_state}_{weak_window_name}_excess_sharpe"],
                    ascending=[False, False, False],
                ).iloc[0]
                verdict["profile_best_by_full_excess_sharpe"].append(
                    {
                        "market_feature_profile": profile,
                        "label": str(best_row["label"]),
                        "market_feature_count": int(best_row["market_feature_count"]),
                        "full_excess_sharpe": float(best_row["full_excess_sharpe"]),
                        "full_excess_annual_return": float(best_row["full_excess_annual_return"]),
                        f"{weak_window_name}_excess_sharpe": float(best_row[f"{weak_window_name}_excess_sharpe"]),
                        f"{focus_state}_{weak_window_name}_excess_sharpe": float(
                            best_row[f"{focus_state}_{weak_window_name}_excess_sharpe"]
                        ),
                    }
                )

            (output_root / "summary.md").write_text(summary_text, encoding="utf-8")
            (output_root / "verdict.json").write_text(
                json.dumps(verdict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
