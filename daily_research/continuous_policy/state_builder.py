from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import (
    HistoryWindow,
    build_prepared_bundle_with_cache,
    history_window_to_dict,
    load_raw_data_with_cache,
    resolve_history_window,
)
from daily_research.baseline.config import ResearchConfig
from daily_research.baseline.data_provider import load_universe_from_tq
from daily_research.baseline.ml_alpha import MLAplhaConfig
from daily_research.execution.liquidity_universe import build_rolling_liquidity_membership, get_named_pool_file
from daily_research.execution.strategy_manifest import load_strategy_manifest


warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


DEFAULT_LABEL_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)
DEFAULT_POOL_REBALANCE_DAYS = 21
DEFAULT_POOL_ADV_WINDOW = 20
DEFAULT_SCORE_BLEND_WEIGHTS = (0.5, 0.5)
ALPHA_PRIOR_NONE = "none"
ALPHA_PRIOR_ACTIVE_EXECUTION = "active_execution_strategy"
DEFAULT_ALPHA_PRIOR_SOURCE = ALPHA_PRIOR_NONE
ALPHA_PRIOR_NONE_ALIASES: frozenset[str] = frozenset({"", "none", "off", "false", "0"})
ALPHA_PRIOR_FRAME_NAMES: tuple[str, ...] = (
    "alpha_prior_score_raw",
    "alpha_prior_score_z",
    "alpha_prior_rank_pct",
    "alpha_prior_target_weight",
    "alpha_prior_selected",
    "alpha_prior_score_delta_1d",
    "alpha_prior_score_delta_5d",
    "alpha_prior_weight_delta_1d",
    "alpha_prior_coverage",
)
LEARNED_ALL_A_POOL_NAMES: frozenset[str] = frozenset(
    {
        "all_a",
        "learned_all_a",
        "full_a",
        "full_market_a",
        "whole_a",
    }
)
STATE_SEQUENCE_BASES: tuple[str, ...] = (
    "score_blend",
    "score_delta_1d",
    "score_delta_5d",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "vol_20d",
    "volatility_expansion",
    "distance_to_20d_high",
    "alpha_prior_score_z",
    "alpha_prior_rank_pct",
    "alpha_prior_target_weight",
    "alpha_prior_selected",
)
STATE_SEQUENCE_LAGS: tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class PreparedPolicyInputs:
    universe: tuple[str, ...]
    pool_name: str
    benchmark: str
    data_source: str
    csv_folder: str
    start_date: str
    end_date: str
    requested_start_date: str
    history_window: HistoryWindow
    raw_cache_meta: dict[str, Any]
    prepared_cache_meta: dict[str, Any]
    close: pd.DataFrame
    open_: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    benchmark_close: pd.Series
    benchmark_open: pd.Series
    score_none: pd.DataFrame
    score_v2: pd.DataFrame
    score_blend: pd.DataFrame
    feature_frames: dict[str, pd.DataFrame]
    market_features: dict[str, pd.Series]
    membership_frame: pd.DataFrame
    rolling_pool_summary: dict[str, Any]
    alpha_prior_summary: dict[str, Any]
    derived_frames: dict[str, pd.DataFrame]
    metadata_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata_summary: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        return {
            "universe_size": len(self.universe),
            "pool_name": self.pool_name,
            "benchmark": self.benchmark,
            "data_source": self.data_source,
            "csv_folder": self.csv_folder,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "requested_start_date": self.requested_start_date,
            "history_window": history_window_to_dict(self.history_window),
            "raw_cache_meta": dict(self.raw_cache_meta),
            "prepared_cache_meta": dict(self.prepared_cache_meta),
            "rolling_pool_summary": dict(self.rolling_pool_summary),
            "alpha_prior_summary": dict(self.alpha_prior_summary),
            "metadata_summary": dict(self.metadata_summary),
        }


def _read_pool_file(path: Path) -> list[str]:
    lines = [str(line).strip().upper() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    return [line for line in lines if line]


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def resolve_active_policy_defaults() -> dict[str, str]:
    manifest = load_strategy_manifest()
    return {
        "benchmark": str(manifest.get("benchmark", "000300.SH") or "000300.SH"),
        "pool_name": str(
            manifest.get("liquidity_pool_name", "")
            or manifest.get("rolling_liquidity_pool", "")
            or manifest.get("liquidity_pool", "")
            or "liquid500"
        ),
        "start_date": str(manifest.get("backtest_start_date", "20250318") or "20250318"),
        "end_date": str(manifest.get("effective_validation_end", "") or manifest.get("promoted_at", "") or "").strip()[:10].replace("-", ""),
    }


def normalize_policy_pool_name(pool_name: str | None) -> str:
    resolved = str(pool_name or "").strip().lower()
    if resolved in LEARNED_ALL_A_POOL_NAMES:
        return "learned_all_a"
    return resolved


def is_learned_all_a_pool_name(pool_name: str | None) -> bool:
    return normalize_policy_pool_name(pool_name) == "learned_all_a"


def resolve_policy_universe(
    *,
    pool_name: str,
    extra_stocks: Iterable[str] | None = None,
    max_universe_size: int = 0,
) -> list[str]:
    extras = _unique_preserve_order(extra_stocks or [])
    resolved_pool_name = normalize_policy_pool_name(pool_name)
    universe: list[str] = []
    if is_learned_all_a_pool_name(resolved_pool_name):
        universe = load_universe_from_tq("all_a")
    elif resolved_pool_name:
        try:
            pool_file = get_named_pool_file(resolved_pool_name)
            if pool_file.exists():
                universe = _read_pool_file(pool_file)
        except Exception:
            universe = []
    if not universe:
        universe = load_universe_from_tq("all_a")
    universe = _unique_preserve_order([*universe, *extras])
    if max_universe_size and int(max_universe_size) > 0:
        capped = universe[: int(max_universe_size)]
        universe = _unique_preserve_order([*capped, *extras])
    return universe


def _default_research_config(*, universe: list[str], benchmark: str) -> ResearchConfig:
    cfg = ResearchConfig(
        universe=list(universe),
        universe_scope="custom",
        benchmark=str(benchmark or "000300.SH"),
        execution_mode="close",
        rebalance_freq="1d",
        min_price=2.0,
        max_price=300.0,
        min_adv20=50_000.0,
        score_clip=3.0,
        regime_state_selector="quadrant",
    )
    return cfg


def _default_ml_config(label_horizons: Iterable[int]) -> MLAplhaConfig:
    horizons = tuple(int(item) for item in label_horizons if int(item) > 0) or DEFAULT_LABEL_HORIZONS
    return MLAplhaConfig(
        target_horizon=max(horizons),
        target_horizons=horizons,
        train_window_days=504,
        retrain_every_days=21,
        min_train_dates=120,
        max_samples_per_day=600,
        max_train_rows=250_000,
        random_seed=7,
        execution_mode="close",
    )


def _safe_pct_change(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    return frame.pct_change(periods=periods, fill_method=None)


def _build_membership_frame(
    *,
    close: pd.DataFrame,
    amount: pd.DataFrame,
    pool_name: str,
    requested_start_date: str,
    end_date: str,
    pool_rebalance_days: int,
    pool_adv_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_pool_name = normalize_policy_pool_name(pool_name)
    if is_learned_all_a_pool_name(resolved_pool_name):
        membership = close.notna().astype(bool)
        return membership, {
            "pool_name": resolved_pool_name,
            "selection_scope": "all_a",
            "selection_mode": "learned_full_market",
            "member_count_median": int(membership.sum(axis=1).median()) if not membership.empty else 0,
        }

    if not resolved_pool_name:
        membership = close.notna().astype(bool)
        return membership, {"pool_name": "", "member_count_median": int(membership.sum(axis=1).median()) if not membership.empty else 0}

    artifact = build_rolling_liquidity_membership(
        close_frame=close,
        amount_frame=amount,
        pool_name=resolved_pool_name,
        signal_start_date=requested_start_date,
        signal_end_date=end_date,
        rebalance_every_days=int(pool_rebalance_days),
        adv_window=int(pool_adv_window),
        min_price=2.0,
        max_price=300.0,
    )
    summary = {
        "pool_name": artifact.pool_name,
        "selection_scope": "rolling_liquidity_pool",
        "selection_mode": "exogenous_pool_mask",
        "pool_size": int(artifact.pool_size),
        "signal_start_date": artifact.signal_start_date,
        "signal_end_date": artifact.signal_end_date,
        "rebalance_every_days": int(artifact.rebalance_every_days),
        "adv_window": int(artifact.adv_window),
        "member_count_median": int(artifact.membership_frame.sum(axis=1).median()) if not artifact.membership_frame.empty else 0,
    }
    return artifact.membership_frame.astype(bool), summary


def _empty_alpha_prior_frames(close: pd.DataFrame, *, summary: dict[str, Any] | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    zero = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
    frames = {name: zero.copy() for name in ALPHA_PRIOR_FRAME_NAMES}
    payload = {
        "source": ALPHA_PRIOR_NONE,
        "status": "disabled",
        "score_panel_csv": "",
        "target_weight_panel_csv": "",
        "coverage_mean": 0.0,
        "selected_count_mean": 0.0,
        "selected_weight_mean": 0.0,
    }
    if summary:
        payload.update(summary)
    return frames, payload


def _resolve_manifest_alpha_paths(manifest: dict[str, Any]) -> tuple[str, str]:
    score_panel = str(
        manifest.get("source_score_panel_csv", "")
        or manifest.get("trade_plan_score_panel_csv", "")
        or ""
    ).strip()
    target_weight_panel = str(
        manifest.get("source_target_weight_panel_csv", "")
        or manifest.get("trade_plan_target_weight_panel_csv", "")
        or ""
    ).strip()
    return score_panel, target_weight_panel


def _first_existing_path(base_dir: Path, names: Iterable[str]) -> str:
    for name in names:
        candidate = base_dir / str(name)
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _resolve_alpha_prior_paths(
    *,
    source: str,
    score_panel: str = "",
    target_weight_panel: str = "",
) -> dict[str, Any]:
    source_text = str(source or DEFAULT_ALPHA_PRIOR_SOURCE).strip()
    explicit_score = str(score_panel or "").strip()
    explicit_target = str(target_weight_panel or "").strip()
    if source_text.lower() in ALPHA_PRIOR_NONE_ALIASES and not (explicit_score or explicit_target):
        return {"source": ALPHA_PRIOR_NONE, "status": "disabled", "score_panel_csv": "", "target_weight_panel_csv": ""}

    resolved_source = source_text or "explicit"
    resolved_score = explicit_score
    resolved_target = explicit_target
    source_meta: dict[str, Any] = {}
    source_lower = source_text.lower()

    if source_lower in {ALPHA_PRIOR_ACTIVE_EXECUTION, "active", "active_manifest", "manifest"}:
        manifest = load_strategy_manifest()
        manifest_score, manifest_target = _resolve_manifest_alpha_paths(manifest)
        resolved_score = resolved_score or manifest_score
        resolved_target = resolved_target or manifest_target
        source_meta = {
            "manifest_strategy_name": str(manifest.get("strategy_name", "") or ""),
            "manifest_candidate_label": str(manifest.get("candidate_label", "") or ""),
            "manifest_score_panel_role": str(manifest.get("score_panel_role", "") or ""),
        }
        resolved_source = ALPHA_PRIOR_ACTIVE_EXECUTION
    elif source_text:
        candidate = Path(source_text).expanduser()
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".json":
            manifest = json.loads(candidate.read_text(encoding="utf-8-sig"))
            manifest_score, manifest_target = _resolve_manifest_alpha_paths(manifest if isinstance(manifest, dict) else {})
            resolved_score = resolved_score or manifest_score
            resolved_target = resolved_target or manifest_target
            source_meta = {"manifest_json": str(candidate.resolve())}
            resolved_source = str(candidate.resolve())
        elif candidate.exists() and candidate.is_dir():
            resolved_score = resolved_score or _first_existing_path(
                candidate,
                (
                    "execution_aligned_daily_live_score_panel.csv",
                    "execution_aligned_daily_score_panel.csv",
                    "daily_live_score_panel.csv",
                    "daily_score_panel.csv",
                    "score_panel.csv",
                ),
            )
            resolved_target = resolved_target or _first_existing_path(
                candidate,
                (
                    "execution_aligned_daily_live_target_weight_panel.csv",
                    "execution_aligned_daily_target_weight_panel.csv",
                    "daily_live_target_weight_panel.csv",
                    "daily_target_weight_panel.csv",
                    "target_weight_panel.csv",
                ),
            )
            resolved_source = str(candidate.resolve())
        elif candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".csv":
            resolved_score = resolved_score or str(candidate.resolve())
            resolved_source = str(candidate.resolve())

    return {
        "source": resolved_source,
        "status": "resolved",
        "score_panel_csv": str(Path(resolved_score).expanduser().resolve()) if resolved_score else "",
        "target_weight_panel_csv": str(Path(resolved_target).expanduser().resolve()) if resolved_target else "",
        **source_meta,
    }


def _load_long_value_panel(panel_path: str | Path, value_column: str) -> pd.DataFrame:
    path = Path(panel_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Alpha-prior panel file not found: {path}")
    panel = pd.read_csv(path)
    required = {"date", "stock", value_column}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Alpha-prior panel {path} is missing columns: {missing}")
    working = panel.loc[:, ["date", "stock", value_column]].copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    working["stock"] = working["stock"].astype(str).str.strip().str.upper()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    return working.pivot_table(index="date", columns="stock", values=value_column, aggfunc="last").sort_index()


def _build_alpha_prior_frames(
    *,
    close: pd.DataFrame,
    source: str = DEFAULT_ALPHA_PRIOR_SOURCE,
    score_panel: str = "",
    target_weight_panel: str = "",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    resolved = _resolve_alpha_prior_paths(source=source, score_panel=score_panel, target_weight_panel=target_weight_panel)
    if str(resolved.get("status", "")) == "disabled":
        return _empty_alpha_prior_frames(close, summary=resolved)

    score_path = str(resolved.get("score_panel_csv", "") or "").strip()
    target_path = str(resolved.get("target_weight_panel_csv", "") or "").strip()
    if not score_path and not target_path:
        return _empty_alpha_prior_frames(close, summary={**resolved, "status": "missing_panels"})

    try:
        raw_score = _load_long_value_panel(score_path, "score") if score_path else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        raw_target = (
            _load_long_value_panel(target_path, "target_weight")
            if target_path
            else pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        )
    except Exception as exc:
        return _empty_alpha_prior_frames(close, summary={**resolved, "status": "load_failed", "error": str(exc)})

    aligned_score = raw_score.reindex(index=close.index, columns=close.columns)
    aligned_target = raw_target.reindex(index=close.index, columns=close.columns)
    coverage = (aligned_score.notna() | aligned_target.notna()).astype(float)

    score_mean = aligned_score.mean(axis=1)
    score_std = aligned_score.std(axis=1, ddof=0).replace(0.0, np.nan)
    score_z = aligned_score.sub(score_mean, axis=0).div(score_std, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rank_pct = aligned_score.rank(axis=1, pct=True, method="average").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score_raw = aligned_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target_weight = aligned_target.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    selected = (target_weight > 1e-8).astype(float)

    frames = {
        "alpha_prior_score_raw": score_raw,
        "alpha_prior_score_z": score_z,
        "alpha_prior_rank_pct": rank_pct,
        "alpha_prior_target_weight": target_weight,
        "alpha_prior_selected": selected,
        "alpha_prior_score_delta_1d": score_z.diff(1).replace([np.inf, -np.inf], np.nan).fillna(0.0),
        "alpha_prior_score_delta_5d": score_z.diff(5).replace([np.inf, -np.inf], np.nan).fillna(0.0),
        "alpha_prior_weight_delta_1d": target_weight.diff(1).replace([np.inf, -np.inf], np.nan).fillna(0.0),
        "alpha_prior_coverage": coverage.fillna(0.0),
    }
    summary = {
        **resolved,
        "status": "loaded",
        "score_panel_rows": int(raw_score.count().sum()) if not raw_score.empty else 0,
        "target_weight_panel_rows": int(raw_target.count().sum()) if not raw_target.empty else 0,
        "coverage_mean": float(frames["alpha_prior_coverage"].mean().mean()) if not close.empty else 0.0,
        "selected_count_mean": float(frames["alpha_prior_selected"].sum(axis=1).mean()) if not close.empty else 0.0,
        "selected_weight_mean": float(frames["alpha_prior_target_weight"].sum(axis=1).mean()) if not close.empty else 0.0,
        "score_date_min": str(raw_score.index.min().date()) if not raw_score.empty else "",
        "score_date_max": str(raw_score.index.max().date()) if not raw_score.empty else "",
        "target_weight_date_min": str(raw_target.index.min().date()) if not raw_target.empty else "",
        "target_weight_date_max": str(raw_target.index.max().date()) if not raw_target.empty else "",
    }
    return frames, summary


def prepare_policy_inputs(
    *,
    pool_name: str,
    start_date: str,
    end_date: str = "",
    benchmark: str = "000300.SH",
    data_source: str = "tq",
    csv_folder: str = "",
    lake_dataset_id: str = "",
    data_lake_root: str = "",
    lake_min_trading_days: int = 2,
    pool_view_id: str = "",
    pool_view_spec: dict[str, Any] | None = None,
    sector_board_view_id: str = "",
    sector_board_view_spec: dict[str, Any] | None = None,
    extra_stocks: Iterable[str] | None = None,
    max_universe_size: int = 0,
    pool_rebalance_days: int = DEFAULT_POOL_REBALANCE_DAYS,
    pool_adv_window: int = DEFAULT_POOL_ADV_WINDOW,
    label_horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
    alpha_prior_source: str = DEFAULT_ALPHA_PRIOR_SOURCE,
    alpha_prior_score_panel: str = "",
    alpha_prior_target_weight_panel: str = "",
    require_lake_benchmark_open: bool = False,
    refresh_cache: bool = False,
    auto_trim_history: bool = True,
    progress_desc: str = "continuous policy prepare",
) -> PreparedPolicyInputs:
    resolved_pool_name = normalize_policy_pool_name(pool_name)
    resolved_data_source = str(data_source or "tq").strip().lower()
    if resolved_data_source == "lake":
        from daily_research.data_lake import DEFAULT_POLICY_INPUT_LAKE_DATASET_ID, ResearchDataLake, load_policy_inputs_from_lake

        requested_universe: list[str] | None = None
        if resolved_pool_name and not is_learned_all_a_pool_name(resolved_pool_name):
            try:
                pool_file = get_named_pool_file(resolved_pool_name)
                if pool_file.exists():
                    requested_universe = _read_pool_file(pool_file)
            except Exception:
                requested_universe = None
        return load_policy_inputs_from_lake(
            lake=ResearchDataLake(str(data_lake_root or "").strip() or None),
            dataset_id=str(lake_dataset_id or DEFAULT_POLICY_INPUT_LAKE_DATASET_ID),
            start_date=str(start_date or "").strip() or "20250318",
            end_date=str(end_date or "").strip(),
            pool_name=resolved_pool_name,
            benchmark=str(benchmark or "000300.SH"),
            universe=requested_universe,
            pool_view_id=str(pool_view_id or ""),
            pool_view_spec=pool_view_spec,
            sector_board_view_id=str(sector_board_view_id or ""),
            sector_board_view_spec=sector_board_view_spec,
            extra_stocks=list(extra_stocks or []),
            max_universe_size=int(max_universe_size or 0),
            min_trading_days=int(lake_min_trading_days or 1),
            alpha_prior_source=alpha_prior_source,
            alpha_prior_score_panel=alpha_prior_score_panel,
            alpha_prior_target_weight_panel=alpha_prior_target_weight_panel,
            require_benchmark_open=bool(require_lake_benchmark_open),
        )
    universe = resolve_policy_universe(
        pool_name=resolved_pool_name,
        extra_stocks=extra_stocks,
        max_universe_size=max_universe_size,
    )
    cfg = _default_research_config(universe=universe, benchmark=benchmark)
    ml_cfg = _default_ml_config(label_horizons)
    history_window = resolve_history_window(
        cfg,
        ml_cfg,
        requested_start_date=str(start_date or "").strip() or "20250318",
        end_date=str(end_date or "").strip(),
        mode="train",
        auto_trim_history=bool(auto_trim_history),
    )
    raw_df_dict, raw_cache_meta = load_raw_data_with_cache(
        data_source=resolved_data_source,
        csv_folder=str(csv_folder or "").strip() or None,
        universe=universe,
        benchmark=str(benchmark or "000300.SH"),
        history_window=history_window,
        use_cache=True,
        refresh_cache=bool(refresh_cache),
        progress_desc=progress_desc,
        progress_position=0,
    )
    bundle, prepared_cache_meta = build_prepared_bundle_with_cache(
        raw_df_dict=raw_df_dict,
        raw_cache_key=str(raw_cache_meta.get("cache_key", "") or ""),
        cfg=cfg,
        enhanced_profile="up_low_breakout_v2",
        use_cache=True,
        refresh_cache=bool(refresh_cache),
    )

    df_dict = bundle["df_dict"]
    close = df_dict["Close"].copy().sort_index()
    open_ = df_dict["Open"].copy().sort_index()
    high = df_dict["High"].copy().sort_index()
    low = df_dict["Low"].copy().sort_index()
    volume = df_dict["Volume"].copy().sort_index()
    amount = df_dict["Amount"].copy().sort_index()
    score_none = bundle["score_none"].copy().sort_index()
    score_v2 = bundle["score_v2"].copy().sort_index()
    score_blend = score_none * float(DEFAULT_SCORE_BLEND_WEIGHTS[0]) + score_v2 * float(DEFAULT_SCORE_BLEND_WEIGHTS[1])

    membership_frame, rolling_pool_summary = _build_membership_frame(
        close=close,
        amount=amount,
        pool_name=resolved_pool_name,
        requested_start_date=str(start_date or "").strip() or history_window.requested_start_date,
        end_date=history_window.end_date,
        pool_rebalance_days=pool_rebalance_days,
        pool_adv_window=pool_adv_window,
    )
    alpha_prior_frames, alpha_prior_summary = _build_alpha_prior_frames(
        close=close,
        source=alpha_prior_source,
        score_panel=alpha_prior_score_panel,
        target_weight_panel=alpha_prior_target_weight_panel,
    )

    returns_1d = _safe_pct_change(close, 1)
    rolling_high_20 = close.rolling(20, min_periods=1).max()
    rolling_high_60 = close.rolling(60, min_periods=1).max()
    rolling_low_20 = close.rolling(20, min_periods=1).min()
    derived_frames = {
        "ret_1d": returns_1d,
        "ret_3d": _safe_pct_change(close, 3),
        "ret_5d": _safe_pct_change(close, 5),
        "ret_10d": _safe_pct_change(close, 10),
        "ret_20d": _safe_pct_change(close, 20),
        "vol_5d": returns_1d.rolling(5).std(),
        "vol_20d": returns_1d.rolling(20).std(),
        "adv20": amount.rolling(20).mean(),
        "volume_ratio_5_20": volume.rolling(5).mean().div(volume.rolling(20).mean().replace(0, np.nan)),
        "score_blend": score_blend,
        "score_delta_1d": score_blend.diff(1),
        "score_delta_5d": score_blend.diff(5),
        "score_delta_accel": score_blend.diff(1).sub(score_blend.diff(5).div(5.0)),
        "ret_accel_5_20": _safe_pct_change(close, 5).sub(_safe_pct_change(close, 20).div(4.0)),
        "distance_to_20d_high": close.div(rolling_high_20.replace(0, np.nan)).sub(1.0),
        "distance_to_60d_high": close.div(rolling_high_60.replace(0, np.nan)).sub(1.0),
        "distance_to_20d_low": close.div(rolling_low_20.replace(0, np.nan)).sub(1.0),
        "volatility_expansion": returns_1d.rolling(5).std().div(returns_1d.rolling(20).std().replace(0, np.nan)).sub(1.0),
        "adv_ratio_5_20": amount.rolling(5).mean().div(amount.rolling(20).mean().replace(0, np.nan)).sub(1.0),
        **alpha_prior_frames,
    }
    for base_name in STATE_SEQUENCE_BASES:
        frame = derived_frames.get(base_name)
        if frame is None:
            continue
        for lag in STATE_SEQUENCE_LAGS:
            derived_frames[f"{base_name}_lag{int(lag)}"] = frame.shift(int(lag))

    return PreparedPolicyInputs(
        universe=tuple(universe),
        pool_name=resolved_pool_name,
        benchmark=str(benchmark or "000300.SH"),
        data_source=resolved_data_source,
        csv_folder=str(csv_folder or ""),
        start_date=str(start_date or "").strip() or history_window.requested_start_date,
        end_date=str(end_date or "").strip() or history_window.end_date,
        requested_start_date=history_window.requested_start_date,
        history_window=history_window,
        raw_cache_meta=raw_cache_meta,
        prepared_cache_meta=prepared_cache_meta,
        close=close,
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        benchmark_close=bundle["benchmark_close"].copy().sort_index(),
        benchmark_open=bundle["benchmark_open"].copy().sort_index(),
        score_none=score_none,
        score_v2=score_v2,
        score_blend=score_blend,
        feature_frames={key: value.copy().sort_index() for key, value in bundle["feature_frames"].items()},
        market_features={key: value.copy().sort_index() for key, value in bundle["market_features"].items()},
        membership_frame=membership_frame.reindex(index=close.index, columns=close.columns, fill_value=False),
        rolling_pool_summary=rolling_pool_summary,
        alpha_prior_summary=alpha_prior_summary,
        derived_frames=derived_frames,
    )


def _series_from_mapping(mapping: dict[str, float | int], universe: Iterable[str]) -> pd.Series:
    return pd.Series({stock: float(mapping.get(stock, 0.0) or 0.0) for stock in universe}, dtype=float)


def _safe_series_value(series: pd.Series, signal_dt: pd.Timestamp) -> float:
    try:
        if signal_dt in series.index:
            return float(series.loc[signal_dt])
        normalized = pd.to_datetime(series.index).normalize()
        matches = np.flatnonzero(normalized == signal_dt.normalize())
        if len(matches) > 0:
            return float(series.iloc[int(matches[-1])])
    except Exception:
        return np.nan
    return np.nan


def _days_since_for_mapping(
    mapping: dict[str, str],
    *,
    universe: Iterable[str],
    signal_dt: pd.Timestamp,
) -> pd.Series:
    values: dict[str, float] = {}
    normalized_signal = pd.Timestamp(signal_dt).normalize()
    for stock in universe:
        raw_value = str(mapping.get(str(stock), "") or "").strip()
        if not raw_value:
            values[str(stock)] = np.nan
            continue
        try:
            mapped_dt = pd.Timestamp(raw_value).normalize()
            values[str(stock)] = float((normalized_signal - mapped_dt).days)
        except Exception:
            values[str(stock)] = np.nan
    return pd.Series(values, dtype=float)


def build_cross_section_state(
    prepared: PreparedPolicyInputs,
    *,
    date: pd.Timestamp | str,
    portfolio_state: Any | None = None,
) -> pd.DataFrame:
    signal_dt = pd.Timestamp(date).normalize()
    if signal_dt not in prepared.close.index:
        raise KeyError(f"Signal date is not available in prepared close frame: {signal_dt:%Y-%m-%d}")

    universe = list(prepared.universe)
    close_row = prepared.close.loc[signal_dt].reindex(universe)
    membership_row = prepared.membership_frame.loc[signal_dt].reindex(universe).fillna(False)

    weights_map = getattr(portfolio_state, "weight_map", lambda: {})()
    entry_map = getattr(portfolio_state, "entry_price_map", lambda: {})()
    peak_map = getattr(portfolio_state, "peak_price_map", lambda: {})()
    hold_day_map = getattr(portfolio_state, "hold_days_map", lambda: {})()
    last_buy_map = getattr(portfolio_state, "last_buy_date_map", lambda: {})()
    last_sell_map = getattr(portfolio_state, "last_sell_date_map", lambda: {})()
    last_reduce_map = getattr(portfolio_state, "last_reduce_date_map", lambda: {})()
    last_exit_map = getattr(portfolio_state, "last_exit_date_map", lambda: {})()
    last_action_map = getattr(portfolio_state, "last_action_label_map", lambda: {})()
    portfolio_features = getattr(portfolio_state, "portfolio_features", lambda: {})()

    current_weight = _series_from_mapping(weights_map, universe)
    entry_price = pd.Series({stock: float(entry_map.get(stock, np.nan)) for stock in universe}, dtype=float)
    peak_price = pd.Series({stock: float(peak_map.get(stock, np.nan)) for stock in universe}, dtype=float)
    hold_days = pd.Series({stock: float(hold_day_map.get(stock, 0) or 0) for stock in universe}, dtype=float)
    days_since_last_buy = _days_since_for_mapping(last_buy_map if isinstance(last_buy_map, dict) else {}, universe=universe, signal_dt=signal_dt)
    days_since_last_sell = _days_since_for_mapping(last_sell_map if isinstance(last_sell_map, dict) else {}, universe=universe, signal_dt=signal_dt)
    days_since_last_reduce = _days_since_for_mapping(last_reduce_map if isinstance(last_reduce_map, dict) else {}, universe=universe, signal_dt=signal_dt)
    days_since_last_exit = _days_since_for_mapping(last_exit_map if isinstance(last_exit_map, dict) else {}, universe=universe, signal_dt=signal_dt)
    last_action = pd.Series({stock: str((last_action_map or {}).get(stock, "") or "").strip().lower() for stock in universe}, dtype=object)
    holding_flag = (current_weight > 1e-8).astype(float)

    row = pd.DataFrame(
        {
            "date": signal_dt.strftime("%Y-%m-%d"),
            "stock": universe,
            "current_price": close_row.to_numpy(dtype=float),
            "in_pool": membership_row.astype(float).to_numpy(dtype=float),
            "current_weight": current_weight.to_numpy(dtype=float),
            "holding_flag": holding_flag.to_numpy(dtype=float),
            "hold_days": hold_days.to_numpy(dtype=float),
            "days_since_last_buy": days_since_last_buy.to_numpy(dtype=float),
            "days_since_last_sell": days_since_last_sell.to_numpy(dtype=float),
            "days_since_last_reduce": days_since_last_reduce.to_numpy(dtype=float),
            "days_since_last_exit": days_since_last_exit.to_numpy(dtype=float),
        }
    )
    row.index = universe

    for feature_name in ("z_score_none", "z_score_v2", "adv20_rank", "price_rank", "ma20_gap", "ma60_gap", "volume_rank"):
        frame = prepared.feature_frames.get(feature_name)
        if frame is not None and signal_dt in frame.index:
            row[feature_name] = frame.loc[signal_dt].reindex(universe).to_numpy(dtype=float)
        else:
            row[feature_name] = np.nan

    row["score_none"] = prepared.score_none.loc[signal_dt].reindex(universe).to_numpy(dtype=float)
    row["score_v2"] = prepared.score_v2.loc[signal_dt].reindex(universe).to_numpy(dtype=float)
    row["score_blend"] = prepared.score_blend.loc[signal_dt].reindex(universe).to_numpy(dtype=float)
    for frame_name in (
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vol_5d",
        "vol_20d",
        "score_delta_1d",
        "score_delta_5d",
        "score_delta_accel",
        "ret_accel_5_20",
        "volume_ratio_5_20",
        "distance_to_20d_high",
        "distance_to_60d_high",
        "distance_to_20d_low",
        "volatility_expansion",
        "adv_ratio_5_20",
        *ALPHA_PRIOR_FRAME_NAMES,
        *[f"{base_name}_lag{lag}" for base_name in STATE_SEQUENCE_BASES for lag in STATE_SEQUENCE_LAGS],
    ):
        frame = prepared.derived_frames[frame_name]
        row[frame_name] = frame.loc[signal_dt].reindex(universe).to_numpy(dtype=float)

    safe_entry_price = entry_price.where(entry_price > 0)
    safe_peak_price = peak_price.where(peak_price > 0)
    row["entry_price"] = safe_entry_price.to_numpy(dtype=float)
    row["peak_price"] = safe_peak_price.to_numpy(dtype=float)
    row["unrealized_pnl"] = np.where(
        holding_flag.to_numpy(dtype=float) > 0,
        close_row.to_numpy(dtype=float) / safe_entry_price.to_numpy(dtype=float) - 1.0,
        0.0,
    )
    row["drawdown_from_peak"] = np.where(
        holding_flag.to_numpy(dtype=float) > 0,
        close_row.to_numpy(dtype=float) / safe_peak_price.to_numpy(dtype=float) - 1.0,
        0.0,
    )
    row["hold_days_clip20"] = hold_days.clip(lower=0.0, upper=20.0).to_numpy(dtype=float)
    row["holding_age_short"] = ((hold_days > 0) & (hold_days <= 3)).astype(float).to_numpy(dtype=float)
    row["holding_age_swing"] = ((hold_days >= 4) & (hold_days <= 10)).astype(float).to_numpy(dtype=float)
    row["holding_age_extended"] = (hold_days >= 11).astype(float).to_numpy(dtype=float)
    row["position_age_phase"] = np.clip(hold_days.to_numpy(dtype=float) / 15.0, 0.0, 1.0)
    row["reentry_cooldown"] = np.where(
        holding_flag.to_numpy(dtype=float) > 0.5,
        0.0,
        np.clip((6.0 - np.nan_to_num(days_since_last_sell.to_numpy(dtype=float), nan=99.0)) / 6.0, 0.0, 1.0),
    )
    # Consolidate the many assembled feature blocks before adding late diagnostic columns.
    row = row.copy()
    row["recent_buy_flag"] = np.where(np.nan_to_num(days_since_last_buy.to_numpy(dtype=float), nan=99.0) <= 3.0, 1.0, 0.0)
    row["recent_sell_flag"] = np.where(np.nan_to_num(days_since_last_sell.to_numpy(dtype=float), nan=99.0) <= 5.0, 1.0, 0.0)
    for action_name in ("open", "hold", "add", "reduce", "exit"):
        row[f"last_action_is_{action_name}"] = (last_action == action_name).astype(float).to_numpy(dtype=float)
    vol20 = np.abs(row["vol_20d"].to_numpy(dtype=float))
    row["pnl_to_vol20"] = np.divide(
        row["unrealized_pnl"].to_numpy(dtype=float),
        vol20,
        out=np.zeros_like(vol20, dtype=float),
        where=vol20 > 1e-8,
    )
    row["drawdown_to_vol20"] = np.divide(
        row["drawdown_from_peak"].to_numpy(dtype=float),
        vol20,
        out=np.zeros_like(vol20, dtype=float),
        where=vol20 > 1e-8,
    )
    row["pnl_from_entry"] = row["unrealized_pnl"].astype(float)
    row["price_from_local_peak"] = (-pd.Series(row["distance_to_20d_high"], index=universe).fillna(0.0)).to_numpy(dtype=float)
    row["signal_decay_speed"] = (
        np.clip(-pd.Series(row["score_delta_1d"], index=universe).fillna(0.0).to_numpy(dtype=float), 0.0, None) * 0.55
        + np.clip(-pd.Series(row["score_delta_accel"], index=universe).fillna(0.0).to_numpy(dtype=float), 0.0, None) * 0.45
    )
    held_pnl = pd.Series(row["unrealized_pnl"], index=universe).where(holding_flag > 0.5)
    row["pnl_rank_in_portfolio"] = held_pnl.rank(pct=True, method="average").fillna(0.0).to_numpy(dtype=float)
    held_drawdown = pd.Series(row["drawdown_from_peak"], index=universe).where(holding_flag > 0.5)
    row["drawdown_rank_in_portfolio"] = held_drawdown.rank(pct=True, method="average").fillna(0.0).to_numpy(dtype=float)

    rank_series = pd.Series(row["score_blend"], index=universe).rank(pct=True, method="average")
    row["score_rank_pct"] = rank_series.reindex(universe).to_numpy(dtype=float)
    score_blend_series = pd.Series(row["score_blend"], index=universe, dtype=float).replace([np.inf, -np.inf], np.nan)
    finite_scores = score_blend_series.dropna()
    row["score_cross_mean"] = float(finite_scores.mean()) if not finite_scores.empty else 0.0
    row["score_cross_std"] = float(finite_scores.std(ddof=0)) if len(finite_scores) > 1 else 0.0
    row["score_cross_top5_mean"] = float(finite_scores.nlargest(min(5, len(finite_scores))).mean()) if not finite_scores.empty else 0.0
    row["score_cross_positive_share"] = float((finite_scores > 0).mean()) if not finite_scores.empty else 0.0

    for market_name, market_series in prepared.market_features.items():
        row[market_name] = _safe_series_value(market_series, signal_dt)

    aggregate = portfolio_features() if callable(portfolio_features) else {}
    aggregate = aggregate if isinstance(aggregate, dict) else {}
    row["portfolio_cash_weight"] = float(aggregate.get("cash_weight", 1.0))
    row["portfolio_gross_exposure"] = float(aggregate.get("gross_exposure", float(np.nansum(row["current_weight"]))))
    row["portfolio_holding_count"] = float(aggregate.get("holding_count", float((holding_flag > 0).sum())))
    row["portfolio_concentration_hhi"] = float(aggregate.get("concentration_hhi", float(np.square(current_weight).sum())))
    row["portfolio_recent_turnover_5d"] = float(aggregate.get("recent_turnover_5d", 0.0))
    row["portfolio_recent_turnover_20d"] = float(aggregate.get("recent_turnover_20d", 0.0))
    row["portfolio_recent_return_5d"] = float(aggregate.get("recent_return_5d", 0.0))
    row["portfolio_recent_return_20d"] = float(aggregate.get("recent_return_20d", 0.0))
    row["portfolio_average_hold_days"] = float(aggregate.get("average_hold_days", float(hold_days[hold_days > 0].mean() if (hold_days > 0).any() else 0.0)))
    row["portfolio_cash_change_5d"] = float(aggregate.get("cash_change_5d", 0.0))
    row["portfolio_cash_weight_mean_20d"] = float(aggregate.get("cash_weight_mean_20d", float(aggregate.get("cash_weight", 1.0))))
    row["portfolio_cash_weight_vol_20d"] = float(aggregate.get("cash_weight_vol_20d", 0.0))
    row["portfolio_cash_deficit"] = float(aggregate.get("cash_deficit", 0.0))
    row["portfolio_turnover_pressure"] = float(aggregate.get("turnover_pressure", 0.0))
    row["portfolio_recent_positive_return_share_20d"] = float(aggregate.get("recent_positive_return_share_20d", 0.0))
    row["portfolio_drawdown_20d"] = float(aggregate.get("portfolio_drawdown_20d", 0.0))
    row["portfolio_return_vol_20d"] = float(aggregate.get("return_vol_20d", 0.0))
    row["recent_reversal_count_20d"] = float(aggregate.get("recent_reversal_count_20d", 0.0))
    row["recent_reversal_rate_20d"] = float(aggregate.get("recent_reversal_rate_20d", 0.0))
    row["recent_reduce_count_10d"] = float(aggregate.get("recent_reduce_count_10d", 0.0))
    row["recent_exit_count_10d"] = float(aggregate.get("recent_exit_count_10d", 0.0))
    row["recent_add_count_10d"] = float(aggregate.get("recent_add_count_10d", 0.0))
    row["recent_open_count_10d"] = float(aggregate.get("recent_open_count_10d", 0.0))
    benchmark_trend_gap_value = float(row["benchmark_trend_gap"].iloc[0]) if "benchmark_trend_gap" in row.columns else 0.0
    benchmark_vol_ratio_value = float(row["benchmark_vol_ratio"].iloc[0]) if "benchmark_vol_ratio" in row.columns else 0.0
    benchmark_trend_gap_value = benchmark_trend_gap_value if np.isfinite(benchmark_trend_gap_value) else 0.0
    benchmark_vol_ratio_value = benchmark_vol_ratio_value if np.isfinite(benchmark_vol_ratio_value) else 0.0
    portfolio_drawdown_value = float(row["portfolio_drawdown_20d"].iloc[0]) if "portfolio_drawdown_20d" in row.columns else 0.0
    portfolio_drawdown_value = portfolio_drawdown_value if np.isfinite(portfolio_drawdown_value) else 0.0
    portfolio_cash_deficit_value = float(row["portfolio_cash_deficit"].iloc[0]) if "portfolio_cash_deficit" in row.columns else 0.0
    portfolio_cash_deficit_value = portfolio_cash_deficit_value if np.isfinite(portfolio_cash_deficit_value) else 0.0
    turnover_pressure_value = float(row["portfolio_turnover_pressure"].iloc[0]) if "portfolio_turnover_pressure" in row.columns else 0.0
    turnover_pressure_value = turnover_pressure_value if np.isfinite(turnover_pressure_value) else 0.0
    reversal_rate_value = float(row["recent_reversal_rate_20d"].iloc[0]) if "recent_reversal_rate_20d" in row.columns else 0.0
    reversal_rate_value = reversal_rate_value if np.isfinite(reversal_rate_value) else 0.0
    row["market_downside_pressure"] = (
        0.55 * max(-benchmark_trend_gap_value, 0.0)
        + 0.25 * max(benchmark_vol_ratio_value, 0.0)
        + 0.20 * max(-portfolio_drawdown_value - 0.02, 0.0)
    )
    row["portfolio_cash_pressure"] = (
        portfolio_cash_deficit_value
        + 0.35 * max(turnover_pressure_value - 0.75, 0.0)
        + 0.18 * max(reversal_rate_value - 0.18, 0.0)
    )
    row["reduce_reversal_pressure"] = (
        0.45 * row["recent_reversal_rate_20d"].astype(float)
        + 0.30 * np.clip(row["recent_reduce_count_10d"].astype(float) / 6.0, 0.0, 1.0)
        + 0.25 * np.clip(row["signal_decay_speed"].astype(float), 0.0, 1.0)
    )
    row["exit_reentry_pressure"] = (
        0.40 * row["reentry_cooldown"].astype(float)
        + 0.30 * row["recent_sell_flag"].astype(float)
        + 0.30 * row["recent_reversal_rate_20d"].astype(float)
    )
    row["cash_regime_pressure"] = (
        0.55 * row["market_downside_pressure"].astype(float)
        + 0.30 * row["portfolio_cash_pressure"].astype(float)
        + 0.15 * np.clip(row["portfolio_turnover_pressure"].astype(float), 0.0, 1.0)
    )
    row["hold_continuity_pressure"] = (
        0.40 * row["holding_age_short"].astype(float)
        + 0.35 * np.clip(row["position_age_phase"].astype(float), 0.0, 1.0)
        + 0.25 * np.clip(row["score_rank_pct"].astype(float) - 0.70, 0.0, 1.0)
    )
    row.index = pd.Index([str(stock).strip().upper() for stock in universe], name="stock_code")
    row["stock"] = [str(stock).strip().upper() for stock in universe]
    return row


def build_daily_state_features(state_frame: pd.DataFrame) -> dict[str, float]:
    if state_frame.empty:
        return {}
    numeric = state_frame.select_dtypes(include=[np.number])
    summary = {
        "pool_breadth": float(state_frame["in_pool"].mean()),
        "score_blend_mean": float(numeric.get("score_blend", pd.Series(dtype=float)).mean()),
        "score_blend_std": float(numeric.get("score_blend", pd.Series(dtype=float)).std()),
        "score_rank_top10_mean": float(state_frame.nlargest(min(10, len(state_frame)), "score_rank_pct")["score_rank_pct"].mean()),
        "positive_score_share": float((numeric.get("score_blend", pd.Series(dtype=float)) > 0).mean()),
        "held_share": float(numeric.get("holding_flag", pd.Series(dtype=float)).mean()),
        "held_weight": float(numeric.get("current_weight", pd.Series(dtype=float)).sum()),
        "unrealized_pnl_mean": float(numeric.get("unrealized_pnl", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).mean()),
        "candidate_edge_mean": float(
            numeric.get("score_blend", pd.Series(dtype=float))
            .where(state_frame["in_pool"] > 0.5)
            .replace([np.inf, -np.inf], np.nan)
            .mean()
        ),
        "candidate_edge_top5": float(
            state_frame.loc[state_frame["in_pool"] > 0.5, "score_blend"].nlargest(min(5, max(int((state_frame["in_pool"] > 0.5).sum()), 1))).mean()
            if bool((state_frame["in_pool"] > 0.5).any())
            else 0.0
        ),
    }
    alpha_score = numeric.get("alpha_prior_score_z", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan)
    alpha_rank = numeric.get("alpha_prior_rank_pct", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan)
    alpha_weight = numeric.get("alpha_prior_target_weight", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    alpha_selected = numeric.get("alpha_prior_selected", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    alpha_coverage = numeric.get("alpha_prior_coverage", pd.Series(dtype=float)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "alpha_prior_score_z" in state_frame.columns:
        alpha_top = state_frame.nlargest(min(5, len(state_frame)), "alpha_prior_score_z")
    else:
        alpha_top = pd.DataFrame()
    summary.update(
        {
            "alpha_prior_coverage_mean": float(alpha_coverage.mean()) if len(alpha_coverage) else 0.0,
            "alpha_prior_selected_count": float((alpha_selected > 0.5).sum()) if len(alpha_selected) else 0.0,
            "alpha_prior_selected_weight": float(alpha_weight.sum()) if len(alpha_weight) else 0.0,
            "alpha_prior_score_mean": float(alpha_score.mean()) if len(alpha_score.dropna()) else 0.0,
            "alpha_prior_score_top5_mean": float(alpha_top["alpha_prior_score_z"].mean()) if not alpha_top.empty else 0.0,
            "alpha_prior_rank_top10_mean": (
                float(state_frame.nlargest(min(10, len(state_frame)), "alpha_prior_rank_pct")["alpha_prior_rank_pct"].mean())
                if "alpha_prior_rank_pct" in state_frame.columns and len(alpha_rank.dropna())
                else 0.0
            ),
        }
    )
    for key in (
        "portfolio_cash_weight",
        "portfolio_gross_exposure",
        "portfolio_holding_count",
        "portfolio_concentration_hhi",
        "portfolio_recent_turnover_5d",
        "portfolio_recent_turnover_20d",
        "portfolio_recent_return_5d",
        "portfolio_recent_return_20d",
        "portfolio_average_hold_days",
        "portfolio_cash_change_5d",
        "portfolio_cash_weight_mean_20d",
        "portfolio_cash_weight_vol_20d",
        "portfolio_cash_deficit",
        "portfolio_turnover_pressure",
        "portfolio_recent_positive_return_share_20d",
        "portfolio_drawdown_20d",
        "portfolio_return_vol_20d",
        "recent_reversal_count_20d",
        "recent_reversal_rate_20d",
        "recent_reduce_count_10d",
        "recent_exit_count_10d",
        "recent_add_count_10d",
        "recent_open_count_10d",
        "market_downside_pressure",
        "portfolio_cash_pressure",
        "reduce_reversal_pressure",
        "exit_reentry_pressure",
        "cash_regime_pressure",
        "hold_continuity_pressure",
        "benchmark_trend_gap",
        "benchmark_annual_vol",
        "benchmark_vol_gap",
        "benchmark_vol_ratio",
        "regime_on",
    ):
        if key in numeric.columns:
            summary[key] = float(numeric[key].iloc[0])
    return summary
