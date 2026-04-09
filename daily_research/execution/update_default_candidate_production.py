from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import torch

import daily_research.deep_alpha.run_deep_alpha_research as research_main
from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.baseline.external_target_weight_bridge import build_target_weight_bridge
from daily_research.deep_alpha.family_epoch_budget import (
    DEFAULT_LATEST_MANIFEST_PATH,
    default_min_epochs_for_budget,
    resolve_epoch_budget_for_family,
)
from daily_research.deep_alpha.execution_alignment import DEFAULT_AUTO_PROFILE_NAMES, resolve_profile
from daily_research.deep_alpha.research_objective import (
    CHECKPOINT_SELECTION_OBJECTIVES,
    DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE,
    DEFAULT_EXECUTION_ALIGNMENT_MODE,
    DEFAULT_EXECUTION_ALIGNMENT_OBJECTIVE,
    DEFAULT_EXECUTION_ALIGNMENT_SELL_TAX_BPS,
    DEFAULT_EXECUTION_ALIGNMENT_SLIPPAGE_BPS,
    DEFAULT_EXECUTION_ALIGNMENT_TRANSACTION_COST_BPS,
    DEFAULT_RESEARCH_OBJECTIVE_MODE,
)
from daily_research.execution.strategy_manifest import (
    DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST,
    build_active_strategy_manifest,
    load_strategy_manifest,
    write_strategy_manifest,
)


FORMAL_SOURCE_RUN = Path(
    "daily_research/output/short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1/runs/state_liquidity_listwise_v1_20250318_20260331"
)
PRODUCTION_ROOT = Path(
    "daily_research/output/deep_alpha_short_alpha_execalign_production_default"
)
DEFAULT_STATIC_FALLBACK_PROFILE = "regoff_k1_5d_ensemble_native_anchor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote the current formal deep_alpha winner into a production full-fit model. "
            "Formal holdout evidence stays in the source run; production retraining uses all "
            "labelable data available before launch, then refreshes live panels to the latest "
            "completed trading date."
        )
    )
    parser.add_argument("--source-run-dir", default=str(FORMAL_SOURCE_RUN))
    parser.add_argument("--production-root", default=str(PRODUCTION_ROOT))
    parser.add_argument(
        "--end-date",
        default="",
        help="Production launch cutoff date. Defaults to latest completed trading date.",
    )
    parser.add_argument("--experiment-tag", default="", help="Optional run tag under daily_research/output.")
    parser.add_argument(
        "--research-objective-mode",
        default="",
        help="Optional override passed to the production fresh retrain. Empty means inherit from the formal source run.",
    )
    parser.add_argument(
        "--checkpoint-selection-objective",
        default="",
        help=(
            "Optional override passed to the production fresh retrain. "
            f"Available values include: {', '.join(CHECKPOINT_SELECTION_OBJECTIVES)}. "
            "Empty means inherit from the formal source run."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-min-improvement",
        type=float,
        default=None,
        help="Optional override passed to the production fresh retrain. Empty means inherit from the formal source run.",
    )
    parser.add_argument(
        "--execution-alignment-objective",
        default="",
        help="Optional override passed to the production fresh retrain. Empty means inherit from the formal source run.",
    )
    parser.add_argument(
        "--execution-alignment-mode",
        default="",
        help="Optional override passed to the production fresh retrain. Empty means inherit from the formal source run.",
    )
    parser.add_argument(
        "--execution-alignment-profile",
        default="",
        help="Optional override passed to the production fresh retrain. Empty means inherit from the formal source run.",
    )
    parser.add_argument(
        "--execution-alignment-candidate-profiles",
        default="",
        help="Optional override passed to the production fresh retrain. Empty means inherit from the formal source run.",
    )
    parser.add_argument(
        "--static-fallback-profile",
        default="",
        help=(
            "Optional static fallback execution profile written into production_root. "
            "Empty means reuse the current production manifest value when available, "
            "otherwise fall back to the active strategy leaderboard default."
        ),
    )
    parser.add_argument(
        "--strategy-manifest-path",
        default=str(DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST),
        help="Where to write the active execution strategy manifest after promotion.",
    )
    parser.add_argument(
        "--family-epoch-budget-manifest",
        default=str(DEFAULT_LATEST_MANIFEST_PATH),
        help="Family epoch budget manifest used to choose the starting production retrain budget.",
    )
    parser.add_argument("--strategy-name", default="", help="Optional name written into the active execution manifest.")
    parser.add_argument(
        "--strategy-panel-mode",
        choices=["auto", "raw", "execution_aligned"],
        default="auto",
        help="Panel mode written into the active execution manifest. auto follows the promoted research winner.",
    )
    parser.add_argument(
        "--activate-strategy",
        dest="activate_strategy",
        action="store_true",
        help="Write the promoted strategy into the active execution manifest so daily execution follows it by default.",
    )
    parser.add_argument(
        "--no-activate-strategy",
        dest="activate_strategy",
        action="store_false",
        help="Skip writing the active execution manifest.",
    )
    parser.set_defaults(activate_strategy=True)
    return parser.parse_args()


def _load_source_config(source_run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_path = source_run_dir / "metrics.json"
    model_path = source_run_dir / "deep_alpha_model.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Formal source metrics.json not found: {metrics_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Formal source deep_alpha_model.pt not found: {model_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    config = dict(artifact.get("config", {}))
    if not config:
        raise RuntimeError(f"Formal source run has no serialized config: {model_path}")
    return metrics, config


def _format_prediction_horizons(raw: Any) -> str:
    if isinstance(raw, (list, tuple)):
        return ",".join(str(int(x)) for x in raw)
    if raw is None:
        return "5,10,20"
    return str(raw)


def _format_task_loss_weights(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    parts: list[str] = []
    for key, value in sorted(raw.items()):
        if str(key).startswith("fwd_excess_"):
            horizon = str(key).replace("fwd_excess_", "")
            parts.append(f"{horizon}:{float(value)}")
        elif str(key) == "risk_downside_20":
            parts.append(f"downside:{float(value)}")
    return ",".join(parts)


def _format_score_horizon_weights(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    return ",".join(f"{int(k)}:{float(v)}" for k, v in sorted(raw.items(), key=lambda item: int(item[0])))


def _format_state_thresholds(raw: Any) -> str:
    if isinstance(raw, (list, tuple)):
        return ",".join(str(float(x)) for x in raw)
    return str(raw or "")


def _format_name_list(raw: Any, *, fallback: list[str] | tuple[str, ...] | None = None) -> str:
    values: list[str] = []
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if str(item).strip()]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item).strip() for item in raw if str(item).strip()]
    if not values and fallback is not None:
        values = [str(item).strip() for item in fallback if str(item).strip()]
    return ",".join(values)


def _infer_research_time_unit(metrics: dict[str, Any], cfg: dict[str, Any]) -> str:
    explicit = str(metrics.get("research_time_unit", cfg.get("research_time_unit", "")) or "").strip().lower()
    if explicit in {"trading_days", "calendar_months"}:
        return explicit
    if any(
        key in metrics or key in cfg
        for key in ("valid_months", "train_eval_window_months", "adaptive_task_window_months")
    ):
        return "calendar_months"
    return "trading_days"


def _infer_family_key_for_source(source_run_dir: Path, cfg: dict[str, Any]) -> str:
    run_hint = str(source_run_dir).lower()
    if "dynamic_graph" in run_hint:
        return "dynamic_graph"
    if "state_liquidity" in run_hint or "short_alpha" in run_hint:
        return "short_alpha"
    if bool(cfg.get("structure_context", False)) or bool(cfg.get("aux_structure_task", False)) or bool(cfg.get("structure_prototype_task", False)):
        return "structure"
    return "baseline"


def _load_metrics_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _has_manifest_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _merge_strategy_metrics(source_metrics: dict[str, Any], production_metrics: dict[str, Any]) -> dict[str, Any]:
    merged = dict(source_metrics)
    for key, value in production_metrics.items():
        if _has_manifest_value(value):
            merged[key] = value
    return merged


def _append_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value in {None, ""}:
        return
    cmd.extend([flag, str(value)])


def _append_flag(cmd: list[str], flag: str, enabled: bool, *, negative_flag: str | None = None) -> None:
    if enabled:
        cmd.append(flag)
    elif negative_flag:
        cmd.append(negative_flag)


def _load_long_target_weight_panel(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["stock"] = raw["stock"].astype(str).str.upper().str.strip()
    raw["target_weight"] = pd.to_numeric(raw["target_weight"], errors="coerce")
    raw = raw.dropna(subset=["date", "stock", "target_weight"])
    return (
        raw.sort_values(["date", "stock"])
        .drop_duplicates(subset=["date", "stock"], keep="last")
        .pivot(index="date", columns="stock", values="target_weight")
        .sort_index()
        .fillna(0.0)
    )


def _panel_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "stock", "target_weight"])
    return (
        frame.stack(dropna=False)
        .rename("target_weight")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "stock"})
    )


def _resolve_default_static_execution_profile() -> str:
    production_manifest = _load_metrics_payload(PRODUCTION_ROOT / "production_retrain_manifest.json")
    manifest_profile = str(production_manifest.get("static_fallback_profile", "") or "").strip()
    if manifest_profile:
        return manifest_profile
    manifest = load_strategy_manifest(DEFAULT_ACTIVE_EXECUTION_STRATEGY_MANIFEST)
    rows = manifest.get("global_deployable_summary_rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            candidate_name = str(row.get("candidate_name", "") or "").strip().lower()
            best_profile = str(row.get("best_profile", "") or "").strip()
            if candidate_name == "state_liquidity_listwise_v1" and best_profile:
                return best_profile
    return DEFAULT_STATIC_FALLBACK_PROFILE


def _materialize_static_fallback_panel(
    *,
    production_root: Path,
    static_profile: str,
) -> None:
    raw_panel_path = production_root / "daily_live_target_weight_panel.csv"
    if not raw_panel_path.exists():
        raise FileNotFoundError(f"Missing research raw live target-weight panel: {raw_panel_path}")
    raw_panel = _load_long_target_weight_panel(raw_panel_path)
    profile = resolve_profile(name=static_profile)
    bridged_panel, bridge_meta = build_target_weight_bridge(
        raw_panel,
        rebalance_freq=profile.rebalance_freq,
        rebalance_offset=0,
        rebalance_offset_mode=profile.rebalance_offset_mode,
        rebalance_anchor_date=profile.rebalance_anchor_date,
        top_k=profile.target_weight_top_k,
        min_weight=profile.target_weight_min_weight,
        power=profile.target_weight_power,
        full_invest=bool(profile.target_weight_full_invest),
    )
    _panel_to_long(bridged_panel).to_csv(
        production_root / "static_fallback_daily_live_target_weight_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (production_root / "static_fallback_daily_live_target_weight_meta.json").write_text(
        json.dumps(
            {
                "static_fallback_profile": profile.name,
                "static_fallback_profile_description": profile.description,
                "target_weight_semantics": "research_raw_target_weight",
                "target_weight_cap_mode": "follow_research_raw_no_global_cap",
                "target_weight_cap_note": (
                    "Static fallback is derived from the uncapped research raw live target-weight panel, "
                    "then bridged with the current default execution profile."
                ),
                "bridge_meta": bridge_meta,
                "source_raw_panel_csv": str(raw_panel_path.resolve()),
                "output_panel_csv": str((production_root / "static_fallback_daily_live_target_weight_panel.csv").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_execution_aligned_score_reference(*, production_root: Path) -> None:
    metrics = _load_metrics_payload(production_root / "metrics.json")
    static_meta = _load_metrics_payload(production_root / "static_fallback_daily_live_target_weight_meta.json")
    execution_profile = str(metrics.get("execution_alignment_profile", "") or "").strip()
    execution_profile_spec = (
        dict(metrics.get("execution_alignment_selected_profile_spec", {}))
        if isinstance(metrics.get("execution_alignment_selected_profile_spec"), dict)
        else {}
    )
    effective_bridge_meta = (
        dict(metrics.get("execution_alignment_selected_bridge_meta", {}))
        if isinstance(metrics.get("execution_alignment_selected_bridge_meta"), dict)
        else {}
    )
    if not effective_bridge_meta and isinstance(static_meta.get("bridge_meta"), dict):
        effective_bridge_meta = dict(static_meta.get("bridge_meta", {}))
    effective_profile_description = str(
        execution_profile_spec.get("description", "")
        or execution_profile_spec.get("profile_description", "")
        or static_meta.get("static_fallback_profile_description", "")
        or ""
    ).strip()
    payload = {
        "role": "execution_preweight_score_panel",
        "effective_execution_profile": execution_profile,
        "effective_execution_profile_description": effective_profile_description,
        "latest_selected_profile": execution_profile,
        "live_target_weight_mode": "execution_aligned_live",
        "effective_execution_bridge_meta": effective_bridge_meta,
        "note": "Displayed score is the execution pre-weight score from the promoted production full-fit live panel.",
        "weight_generation_note": (
            "Current live weights come from the promoted production full-fit "
            "execution-aligned live target-weight panel; the displayed pre-weight "
            "score is the signal-day execution-preweight score only, so it does not "
            "need to be monotonic with final weight."
        ),
    }
    (production_root / "daily_live_score_reference.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_universe_for_source(cfg: research_main.DeepAlphaConfig, args: SimpleNamespace) -> list[str]:
    stocks_file = args.stocks_file or None
    universe = research_main.load_stocks_from_file(stocks_file) if stocks_file else []
    if args.data_source != "tq":
        return universe
    if args.rolling_liquidity_pool:
        try:
            return research_main.load_universe_from_tq(cfg.universe_scope)
        except Exception:
            cached = research_main._load_cached_rolling_pool_union(
                pool_name=args.rolling_liquidity_pool,
                start_date=cfg.start_date,
                end_date=cfg.end_date,
            )
            if cached:
                return cached
            raise
    if not universe and cfg.universe_scope == "all_a":
        return research_main.load_universe_from_tq(cfg.universe_scope)
    if not universe:
        raise ValueError("TQ mode without stocks requires a valid stocks_file or universe_scope=all_a.")
    return universe


def _resolve_training_dates(
    *,
    source_run_dir: Path,
    latest_completed_date: str,
) -> tuple[str, str, int]:
    metrics, raw_cfg = _load_source_config(source_run_dir)
    cfg = research_main.DeepAlphaConfig(**raw_cfg)
    cfg.end_date = str(latest_completed_date)

    args = SimpleNamespace(
        data_source="tq",
        csv_folder=None,
        stocks_file=str(metrics.get("stocks_file", "") or ""),
        liquidity_pool=str(metrics.get("liquidity_pool", "") or ""),
        rolling_liquidity_pool=str(metrics.get("rolling_liquidity_pool", "") or ""),
        pool_rebalance_days=int(metrics.get("rolling_pool_rebalance_days", 21) or 21),
        pool_adv_window=int(metrics.get("rolling_pool_adv_window", 20) or 20),
        relation_layer=bool(metrics.get("relation_layer", False)),
        use_cache=True,
        refresh_cache=False,
    )
    universe = _load_universe_for_source(cfg, args)
    raw_df_dict, raw_key = research_main.load_raw_market_data(
        cfg,
        args,
        universe,
        progress_desc="Production retrain: load market data",
    )
    benchmark_open = raw_df_dict["Open"][cfg.benchmark].copy()
    df_dict, benchmark_close = research_main.split_benchmark_from_universe(raw_df_dict, cfg.benchmark)
    if args.rolling_liquidity_pool:
        rolling_pool_artifact, _ = research_main.load_cached_or_build_rolling_pool(
            args=args,
            cfg=cfg,
            df_dict=df_dict,
            raw_key=raw_key,
        )
        rolling_union = rolling_pool_artifact.membership_frame.columns[
            rolling_pool_artifact.membership_frame.any(axis=0)
        ].tolist()
        if not rolling_union:
            raise RuntimeError(
                f"Rolling {args.rolling_liquidity_pool} production retrain pool is empty for the requested window."
            )
        df_dict = research_main.subset_df_dict_to_stocks(df_dict, rolling_union)
    close = df_dict["Close"]
    target_names = list(metrics.get("target_names", []) or [])
    breakout_event_task = any(str(name).startswith("event_breakout_") for name in target_names)
    clean_breakout_event_task = any(str(name).startswith("event_clean_breakout_") for name in target_names)
    target_frames = research_main.build_targets(
        close,
        benchmark_close,
        cfg.prediction_horizons,
        open_df=df_dict["Open"],
        benchmark_open=benchmark_open,
        execution_mode="next_open",
        breakout_event_horizon=int(metrics.get("breakout_event_horizon", getattr(cfg, "breakout_event_horizon", 5)) or 5),
        breakout_event_threshold=float(
            metrics.get("breakout_event_threshold", getattr(cfg, "breakout_event_threshold", 0.08)) or 0.08
        ),
        breakout_event_pullback_limit=float(
            metrics.get("breakout_event_pullback_limit", getattr(cfg, "breakout_event_pullback_limit", 0.03)) or 0.03
        ),
        breakout_event_task=breakout_event_task,
        clean_breakout_event_task=clean_breakout_event_task,
    )

    valid_mask: pd.DataFrame | None = None
    for frame in target_frames.values():
        frame_valid = frame.reindex(index=close.index, columns=close.columns).notna()
        valid_mask = frame_valid if valid_mask is None else (valid_mask & frame_valid)
    if valid_mask is None:
        raise RuntimeError("Failed to build any production targets for latest trainable date resolution.")

    valid_dates = list(valid_mask.index[valid_mask.any(axis=1)])
    if len(valid_dates) == 0:
        raise RuntimeError(
            "No labelable dates found for production retraining. "
            "Check the market data window and target horizon configuration."
        )
    latest_trainable_date = pd.Timestamp(valid_dates[-1]).strftime("%Y%m%d")
    internal_monitor_days = min(3, len(valid_dates))
    internal_monitor_start_date = pd.Timestamp(valid_dates[-internal_monitor_days]).strftime("%Y%m%d")
    return latest_trainable_date, internal_monitor_start_date, int(internal_monitor_days)


def _build_retrain_command(
    *,
    source_run_dir: Path,
    family_epoch_budget_manifest: str,
    latest_completed_date: str,
    latest_trainable_date: str,
    internal_monitor_start_date: str,
    internal_monitor_days: int,
    experiment_tag: str,
    research_objective_mode_override: str = "",
    checkpoint_selection_objective_override: str = "",
    checkpoint_selection_min_improvement_override: float | None = None,
    execution_alignment_objective_override: str = "",
    execution_alignment_mode_override: str = "",
    execution_alignment_profile_override: str = "",
    execution_alignment_candidate_profiles_override: str = "",
    static_fallback_profile_override: str = "",
) -> list[str]:
    metrics, cfg = _load_source_config(source_run_dir)
    family_key = _infer_family_key_for_source(source_run_dir, cfg)
    epoch_budget = resolve_epoch_budget_for_family(
        family_key,
        manifest_path=family_epoch_budget_manifest,
        fallback_epochs=int(cfg.get("epochs", 8) or 8),
    )
    resolved_default_static_profile = str(
        static_fallback_profile_override or _resolve_default_static_execution_profile()
    ).strip()
    script_path = Path("daily_research/deep_alpha/run_deep_alpha_research.py").resolve()
    cmd: list[str] = [sys.executable, str(script_path)]

    _append_arg(cmd, "--data-source", "tq")
    _append_arg(cmd, "--start-date", cfg.get("start_date", "20210101"))
    _append_arg(cmd, "--end-date", latest_completed_date)
    _append_arg(cmd, "--benchmark", cfg.get("benchmark", "000300.SH"))

    rolling_pool = str(metrics.get("rolling_liquidity_pool", "") or "").strip()
    liquidity_pool = str(metrics.get("liquidity_pool", "") or "").strip()
    stocks_file = str(metrics.get("stocks_file", "") or "").strip()
    if rolling_pool:
        _append_arg(cmd, "--rolling-liquidity-pool", rolling_pool)
        _append_arg(cmd, "--pool-rebalance-days", int(metrics.get("rolling_pool_rebalance_days", 21) or 21))
        _append_arg(cmd, "--pool-adv-window", int(metrics.get("rolling_pool_adv_window", 20) or 20))
    elif liquidity_pool:
        _append_arg(cmd, "--liquidity-pool", liquidity_pool)
    elif stocks_file:
        _append_arg(cmd, "--stocks-file", stocks_file)

    _append_arg(cmd, "--prediction-horizons", _format_prediction_horizons(cfg.get("prediction_horizons")))
    _append_arg(cmd, "--return-loss-mode", cfg.get("return_loss_mode", "top_bottom_bce"))
    _append_arg(cmd, "--return-target-transform", cfg.get("return_target_transform", "raw"))
    _append_arg(cmd, "--return-top-frac", cfg.get("return_top_frac", 0.2))
    _append_arg(cmd, "--return-bottom-frac", cfg.get("return_bottom_frac", 0.2))
    _append_arg(cmd, "--task-loss-weights", _format_task_loss_weights(cfg.get("target_loss_weights")))
    _append_arg(cmd, "--score-horizon-weights", _format_score_horizon_weights(cfg.get("score_horizon_weights")))
    _append_arg(cmd, "--score-rank-blend", cfg.get("score_rank_blend", 0.35))
    _append_arg(cmd, "--score-downside-penalty", cfg.get("score_downside_penalty", 0.25))
    _append_arg(cmd, "--score-risk-mode", cfg.get("score_risk_mode", "subtract"))
    _append_arg(cmd, "--score-risk-gate-threshold", cfg.get("score_risk_gate_threshold", 0.35))
    _append_arg(cmd, "--score-risk-state-thresholds", _format_state_thresholds(metrics.get("score_risk_state_thresholds")))
    _append_arg(cmd, "--score-head-method", metrics.get("score_head_method", "manual"))
    _append_flag(cmd, "--adaptive-task-weights", bool(metrics.get("adaptive_task_weights", False)))
    research_time_unit = _infer_research_time_unit(metrics, cfg)
    _append_arg(cmd, "--research-time-unit", research_time_unit)
    _append_arg(cmd, "--adaptive-task-window-days", metrics.get("adaptive_task_window_days", cfg.get("train_eval_window_days", 126)))
    _append_arg(cmd, "--adaptive-task-window-months", metrics.get("adaptive_task_window_months", cfg.get("adaptive_task_window_months", 6)))

    _append_arg(cmd, "--train-end-date", latest_trainable_date)
    _append_arg(cmd, "--valid-start-date", internal_monitor_start_date)
    _append_arg(cmd, "--valid-days", internal_monitor_days)
    _append_arg(cmd, "--valid-months", 0)
    _append_arg(cmd, "--train-eval-window-days", cfg.get("train_eval_window_days", 126))
    _append_arg(cmd, "--train-eval-window-months", cfg.get("train_eval_window_months", 6))

    _append_arg(cmd, "--batch-size", cfg.get("batch_size", 256))
    _append_arg(cmd, "--num-workers", cfg.get("num_workers", 0))
    _append_flag(cmd, "--pin-memory", bool(cfg.get("pin_memory", True)), negative_flag="--no-pin-memory")
    _append_arg(cmd, "--hidden-dim", cfg.get("hidden_dim", 96))
    _append_arg(cmd, "--encoder-family", cfg.get("encoder_family", "patch_transformer"))
    _append_arg(cmd, "--patch-len", cfg.get("patch_len", 5))
    _append_arg(cmd, "--pretrained-encoder-path", cfg.get("pretrained_encoder_path", ""))
    _append_arg(cmd, "--return-head-mode", cfg.get("return_head_mode", "shared"))
    _append_arg(cmd, "--context-dim", cfg.get("context_dim", 16))
    _append_flag(cmd, "--state-context", bool(cfg.get("state_context", False)))
    _append_flag(cmd, "--liquidity-context", bool(cfg.get("liquidity_context", False)))
    _append_flag(cmd, "--structure-context", bool(cfg.get("structure_context", False)))
    _append_flag(cmd, "--aux-structure-task", bool(cfg.get("aux_structure_task", False)))
    _append_arg(cmd, "--aux-structure-loss-weight", cfg.get("aux_structure_loss_weight", 0.1))
    _append_arg(cmd, "--aux-structure-label-smoothing", cfg.get("aux_structure_label_smoothing", 0.05))
    _append_flag(cmd, "--structure-prototype-task", bool(cfg.get("structure_prototype_task", False)))
    _append_arg(cmd, "--structure-prototype-loss-weight", cfg.get("structure_prototype_loss_weight", 0.05))
    _append_arg(cmd, "--structure-prototype-temperature", cfg.get("structure_prototype_temperature", 0.2))
    _append_arg(cmd, "--transformer-heads", cfg.get("transformer_heads", 4))
    _append_arg(cmd, "--transformer-layers", cfg.get("transformer_layers", 2))
    _append_arg(cmd, "--dropout", cfg.get("dropout", 0.10))
    _append_arg(cmd, "--learning-rate", cfg.get("learning_rate", 1e-3))
    _append_arg(cmd, "--weight-decay", cfg.get("weight_decay", 1e-4))
    _append_arg(cmd, "--epochs", epoch_budget)
    _append_arg(cmd, "--min-epochs", default_min_epochs_for_budget(epoch_budget))
    _append_arg(cmd, "--early-stop-patience", max(int(cfg.get("early_stop_patience", 2)), 8))
    _append_arg(cmd, "--lr-plateau-patience", max(int(cfg.get("lr_plateau_patience", 1)), 4))
    _append_arg(cmd, "--lr-plateau-factor", cfg.get("lr_plateau_factor", 0.5))
    _append_arg(cmd, "--min-improvement", cfg.get("min_improvement", 1e-4))
    _append_flag(cmd, "--use-amp", bool(cfg.get("use_amp", True)), negative_flag="--no-amp")
    _append_flag(cmd, "--safe-runtime-profile", bool(cfg.get("safe_runtime_profile", True)), negative_flag="--no-safe-runtime-profile")
    _append_arg(cmd, "--ranking-loss-weight", cfg.get("ranking_loss_weight", 0.0))
    _append_arg(cmd, "--listwise-loss-weight", cfg.get("listwise_loss_weight", 0.0))
    _append_arg(cmd, "--listwise-temperature", cfg.get("listwise_temperature", 0.35))
    _append_arg(cmd, "--max-rank-pairs-per-group", cfg.get("max_rank_pairs_per_group", 2048))
    _append_arg(cmd, "--random-seed", cfg.get("random_seed", 7))
    _append_arg(cmd, "--market-state-count", cfg.get("market_state_count", 4))
    _append_arg(cmd, "--holding-count", cfg.get("holding_count", 5))
    _append_arg(cmd, "--rebalance-freq", cfg.get("rebalance_freq", "1d"))
    _append_arg(cmd, "--max-weight", cfg.get("max_weight", 0.25))
    _append_flag(cmd, "--relation-layer", bool(metrics.get("relation_layer", False)))
    _append_flag(cmd, "--liquidity-layer", bool(cfg.get("liquidity_layer", False)))
    _append_arg(cmd, "--liquidity-bucket-count", cfg.get("liquidity_bucket_count", 5))
    _append_flag(cmd, "--dynamic-graph-layer", bool(cfg.get("dynamic_graph_layer", False)))
    _append_arg(cmd, "--dynamic-graph-top-k", cfg.get("dynamic_graph_top_k", 8))
    _append_arg(cmd, "--dynamic-graph-temperature", cfg.get("dynamic_graph_temperature", 0.35))
    _append_arg(cmd, "--dynamic-graph-industry-boost", cfg.get("dynamic_graph_industry_boost", 0.15))
    _append_arg(cmd, "--dynamic-graph-style-boost", cfg.get("dynamic_graph_style_boost", 0.05))
    _append_flag(cmd, "--short-alpha-features", bool(metrics.get("short_alpha_features", cfg.get("short_alpha_features", False))))
    _append_arg(cmd, "--breakout-event-horizon", metrics.get("breakout_event_horizon", cfg.get("breakout_event_horizon", 5)))
    _append_arg(
        cmd,
        "--breakout-event-threshold",
        metrics.get("breakout_event_threshold", cfg.get("breakout_event_threshold", 0.08)),
    )
    _append_arg(
        cmd,
        "--breakout-event-pullback-limit",
        metrics.get("breakout_event_pullback_limit", cfg.get("breakout_event_pullback_limit", 0.03)),
    )
    _append_arg(
        cmd,
        "--breakout-event-loss-weight",
        metrics.get("breakout_event_loss_weight", cfg.get("breakout_event_loss_weight", 0.0)),
    )
    _append_arg(
        cmd,
        "--clean-breakout-event-loss-weight",
        metrics.get("clean_breakout_event_loss_weight", cfg.get("clean_breakout_event_loss_weight", 0.0)),
    )
    _append_arg(cmd, "--min-adv20", cfg.get("min_adv20", 50_000.0))
    _append_arg(cmd, "--min-price", cfg.get("min_price", 2.0))
    _append_arg(cmd, "--max-price", cfg.get("max_price", 300.0))

    research_objective_mode = str(
        research_objective_mode_override
        or metrics.get("research_objective_mode", "")
        or DEFAULT_RESEARCH_OBJECTIVE_MODE
    ).strip()
    checkpoint_selection_objective = str(
        checkpoint_selection_objective_override
        or metrics.get("checkpoint_selection_objective", "")
        or DEFAULT_CHECKPOINT_SELECTION_OBJECTIVE
    ).strip()
    checkpoint_selection_min_improvement = (
        checkpoint_selection_min_improvement_override
        if checkpoint_selection_min_improvement_override is not None
        else metrics.get("checkpoint_selection_min_improvement", cfg.get("min_improvement", 1e-4))
    )
    _append_arg(cmd, "--research-objective-mode", research_objective_mode)
    _append_arg(cmd, "--checkpoint-selection-objective", checkpoint_selection_objective)
    _append_arg(cmd, "--checkpoint-selection-min-improvement", checkpoint_selection_min_improvement)

    execution_alignment_mode = str(
        execution_alignment_mode_override
        or metrics.get("execution_alignment_mode", "")
        or ""
    ).strip()
    if research_objective_mode == DEFAULT_RESEARCH_OBJECTIVE_MODE and execution_alignment_mode in {"", "off"}:
        execution_alignment_mode = DEFAULT_EXECUTION_ALIGNMENT_MODE
    elif not execution_alignment_mode:
        execution_alignment_mode = "off"
    execution_alignment_objective = str(
        execution_alignment_objective_override
        or metrics.get("execution_alignment_objective", "")
        or DEFAULT_EXECUTION_ALIGNMENT_OBJECTIVE
    ).strip()
    execution_alignment_profile = str(
        execution_alignment_profile_override
        or metrics.get("execution_alignment_profile", "")
        or ""
    ).strip()
    if not execution_alignment_profile and research_objective_mode == DEFAULT_RESEARCH_OBJECTIVE_MODE:
        execution_alignment_profile = resolved_default_static_profile
    execution_alignment_candidate_profiles = _format_name_list(
        execution_alignment_candidate_profiles_override or metrics.get("execution_alignment_candidate_profiles"),
        fallback=DEFAULT_AUTO_PROFILE_NAMES,
    )
    if execution_alignment_mode != "off":
        _append_arg(cmd, "--execution-alignment-mode", execution_alignment_mode)
        _append_arg(cmd, "--execution-alignment-objective", execution_alignment_objective)
        _append_arg(
            cmd,
            "--execution-alignment-transaction-cost-bps",
            metrics.get(
                "execution_alignment_transaction_cost_bps",
                DEFAULT_EXECUTION_ALIGNMENT_TRANSACTION_COST_BPS,
            ),
        )
        _append_arg(
            cmd,
            "--execution-alignment-slippage-bps",
            metrics.get(
                "execution_alignment_slippage_bps",
                DEFAULT_EXECUTION_ALIGNMENT_SLIPPAGE_BPS,
            ),
        )
        _append_arg(
            cmd,
            "--execution-alignment-sell-tax-bps",
            metrics.get(
                "execution_alignment_sell_tax_bps",
                DEFAULT_EXECUTION_ALIGNMENT_SELL_TAX_BPS,
            ),
        )
        if execution_alignment_mode == "profile":
            _append_arg(
                cmd,
                "--execution-alignment-profile",
                execution_alignment_profile or resolved_default_static_profile,
            )
        else:
            _append_arg(cmd, "--execution-alignment-candidate-profiles", execution_alignment_candidate_profiles)

    _append_arg(cmd, "--experiment-tag", experiment_tag)
    return cmd


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_production_manifest(
    *,
    production_root: Path,
    source_run_dir: Path,
    run_dir: Path,
    latest_completed_date: str,
    latest_trainable_date: str,
    internal_monitor_start_date: str,
    internal_monitor_days: int,
    train_start_date: str,
    static_fallback_profile: str,
    strategy_manifest_path: Path | None = None,
    activate_strategy: bool = False,
) -> None:
    timestamp = pd.Timestamp.now().isoformat()
    retrain_frequency_root = Path("daily_research/output/deep_alpha_retrain_frequency_formal_20260402_r1").resolve()
    retrain_frequency_csv = retrain_frequency_root / "frequency_summary_common_window.csv"
    manifest = {
        "mode": "production_fullfit",
        "policy": {
            "research_protocol": "最近一年 formal holdout 只用于研究判决，不与 production full-fit 混报。",
            "production_protocol": "winner 冻结后，使用截至上线前的全部可标注数据重训一次，再刷新到最新完成交易日上线。",
            "evidence_boundary": "production full-fit 结果不得回填成 formal holdout 证据。",
        },
        "retrain_frequency_policy": {
            "runner": "daily_research/deep_alpha/run_retrain_frequency_formal_matrix.py",
            "research_output_dir": str(retrain_frequency_root),
            "leaderboard_csv": str(retrain_frequency_csv),
            "leaderboard_basis": "common comparison window",
            "comparison_window": "2025-03-18 -> 2026-03-27",
            "preferred_cadence": "monthly_calendar",
            "preferred_label": "Retrain Monthly",
            "secondary_cadence": "quarterly_63d",
            "secondary_label": "Retrain 63D",
            "auto_retrain_enabled": True,
            "auto_retrain_mode": "monthly_calendar",
            "auto_retrain_trigger": "next_calendar_month_after_launch_cutoff",
            "auto_retrain_fallback_trading_days": 21,
            "warn_after_trading_days": 21,
            "block_after_trading_days": 63,
            "notes": "2026-04-02 formal matrix: Monthly > 63D > Freeze 1Y > 21D. Execution side should auto retrain after the next calendar month boundary, keep a 21-trading-day fallback reminder, and block once it drifts past the 63-trading-day guardrail unless explicitly overridden.",
        },
        "source_formal_run_dir": str(source_run_dir.resolve()),
        "active_production_run_dir": str(run_dir.resolve()),
        "production_root": str(production_root.resolve()),
        "target_weight_semantics": "research_raw_target_weight",
        "target_weight_cap_mode": "follow_research_raw_no_global_cap",
        "raw_live_target_weight_panel_csv": str((production_root / "daily_live_target_weight_panel.csv").resolve()),
        "portfolio_capped_live_target_weight_panel_csv": str(
            (production_root / "portfolio_capped_daily_live_target_weight_panel.csv").resolve()
        ),
        "static_fallback_profile": str(static_fallback_profile),
        "static_fallback_daily_live_target_weight_panel_csv": str(
            (production_root / "static_fallback_daily_live_target_weight_panel.csv").resolve()
        ),
        "static_fallback_daily_live_target_weight_meta_json": str(
            (production_root / "static_fallback_daily_live_target_weight_meta.json").resolve()
        ),
        "active_execution_strategy_manifest": "" if strategy_manifest_path is None else str(strategy_manifest_path.resolve()),
        "activate_strategy_after_sync": bool(activate_strategy),
        "train_start_date": str(train_start_date),
        "train_end_date": str(latest_trainable_date),
        "launch_cutoff_date": str(latest_completed_date),
        "internal_monitor_start_date": str(internal_monitor_start_date),
        "internal_monitor_days": int(internal_monitor_days),
        "created_at": timestamp,
    }
    (production_root / "production_retrain_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_lines = [
        "# Production Full-Fit",
        "",
        f"- source_formal_run_dir: `{source_run_dir.as_posix()}`",
        f"- active_production_run_dir: `{run_dir.as_posix()}`",
        f"- train_start_date: `{train_start_date}`",
        f"- train_end_date: `{latest_trainable_date}`",
        f"- launch_cutoff_date: `{latest_completed_date}`",
        f"- internal_monitor_start_date: `{internal_monitor_start_date}`",
        f"- internal_monitor_days: `{internal_monitor_days}`",
        f"- target_weight_semantics: `research_raw_target_weight`",
        f"- target_weight_cap_mode: `follow_research_raw_no_global_cap`",
        f"- raw_live_target_weight_panel_csv: `{(production_root / 'daily_live_target_weight_panel.csv').as_posix()}`",
        f"- portfolio_capped_live_target_weight_panel_csv: `{(production_root / 'portfolio_capped_daily_live_target_weight_panel.csv').as_posix()}`",
        f"- static_fallback_profile: `{static_fallback_profile}`",
        f"- static_fallback_daily_live_target_weight_panel_csv: `{(production_root / 'static_fallback_daily_live_target_weight_panel.csv').as_posix()}`",
        f"- active_execution_strategy_manifest: `{'' if strategy_manifest_path is None else strategy_manifest_path.as_posix()}`",
        "- retrain_frequency_policy: `Retrain Monthly` preferred, auto retrain on next calendar month boundary, fallback remind at `21` trading days, block at `63` trading days",
        f"- retrain_frequency_leaderboard: `{retrain_frequency_csv.as_posix()}`",
        "- note: 本目录只用于日常 production 信号，不作为 formal holdout 证据。",
    ]
    (production_root / "production_retrain_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def _sync_production_root(
    *,
    run_dir: Path,
    production_root: Path,
    source_run_dir: Path,
    latest_completed_date: str,
    latest_trainable_date: str,
    internal_monitor_start_date: str,
    internal_monitor_days: int,
    train_start_date: str,
    static_fallback_profile: str = "",
    strategy_manifest_path: Path | None = None,
    activate_strategy: bool = False,
) -> None:
    production_root.mkdir(parents=True, exist_ok=True)
    resolved_static_fallback_profile = str(static_fallback_profile or _resolve_default_static_execution_profile()).strip()
    for name in [
        "metrics.json",
        "deep_alpha_model.pt",
        "score_head_artifact.pkl",
        "risk_gate_artifact.pkl",
        "execution_alignment_artifact.pkl",
        "train_history.csv",
        "daily_live_score_panel.csv",
        "daily_live_target_weight_panel.csv",
        "portfolio_capped_daily_live_target_weight_panel.csv",
        "execution_aligned_daily_live_score_panel.csv",
        "execution_aligned_daily_live_target_weight_panel.csv",
    ]:
        _copy_if_exists(run_dir / name, production_root / name)
    _materialize_static_fallback_panel(
        production_root=production_root,
        static_profile=resolved_static_fallback_profile,
    )
    _write_execution_aligned_score_reference(production_root=production_root)
    _write_production_manifest(
        production_root=production_root,
        source_run_dir=source_run_dir,
        run_dir=run_dir,
        latest_completed_date=latest_completed_date,
        latest_trainable_date=latest_trainable_date,
        internal_monitor_start_date=internal_monitor_start_date,
        internal_monitor_days=internal_monitor_days,
        train_start_date=train_start_date,
        static_fallback_profile=resolved_static_fallback_profile,
        strategy_manifest_path=strategy_manifest_path,
        activate_strategy=activate_strategy,
    )


def _activate_strategy(
    *,
    source_run_dir: Path,
    production_root: Path,
    strategy_manifest_path: Path,
    strategy_name: str,
    strategy_panel_mode: str,
) -> tuple[Path, dict[str, Any]]:
    source_metrics = _load_metrics_payload(source_run_dir / "metrics.json")
    production_metrics = _load_metrics_payload(production_root / "metrics.json")
    strategy_metrics = _merge_strategy_metrics(source_metrics, production_metrics)
    resolved_strategy_name = (
        str(strategy_name).strip()
        or str(strategy_metrics.get("experiment_tag", "")).strip()
        or production_root.name
    )
    payload = build_active_strategy_manifest(
        source_run_dir=source_run_dir,
        production_root=production_root,
        source_metrics=source_metrics,
        strategy_metrics=strategy_metrics,
        panel_mode=strategy_panel_mode,
        strategy_name=resolved_strategy_name,
        promoted_at=pd.Timestamp.now().isoformat(),
    )
    manifest_path = write_strategy_manifest(payload, path=strategy_manifest_path)
    return manifest_path, payload


def main() -> None:
    args = parse_args()
    source_run_dir = Path(args.source_run_dir).resolve()
    production_root = Path(args.production_root).resolve()
    strategy_manifest_path = Path(args.strategy_manifest_path).resolve()
    latest_completed_date = pd.Timestamp(args.end_date or get_latest_completed_trading_date()).strftime("%Y%m%d")
    latest_trainable_date, internal_monitor_start_date, internal_monitor_days = _resolve_training_dates(
        source_run_dir=source_run_dir,
        latest_completed_date=latest_completed_date,
    )
    _, cfg = _load_source_config(source_run_dir)
    train_start_date = str(cfg.get("start_date", "20210101"))
    experiment_tag = (
        args.experiment_tag.strip()
        or f"deep_alpha_short_alpha_execfirst_production_fullfit_{latest_completed_date}_r1"
    )

    cmd = _build_retrain_command(
        source_run_dir=source_run_dir,
        family_epoch_budget_manifest=str(args.family_epoch_budget_manifest),
        latest_completed_date=latest_completed_date,
        latest_trainable_date=latest_trainable_date,
        internal_monitor_start_date=internal_monitor_start_date,
        internal_monitor_days=internal_monitor_days,
        experiment_tag=experiment_tag,
        research_objective_mode_override=str(args.research_objective_mode or ""),
        checkpoint_selection_objective_override=str(args.checkpoint_selection_objective or ""),
        checkpoint_selection_min_improvement_override=args.checkpoint_selection_min_improvement,
        execution_alignment_objective_override=str(args.execution_alignment_objective or ""),
        execution_alignment_mode_override=str(args.execution_alignment_mode or ""),
        execution_alignment_profile_override=str(args.execution_alignment_profile or ""),
        execution_alignment_candidate_profiles_override=str(args.execution_alignment_candidate_profiles or ""),
        static_fallback_profile_override=str(args.static_fallback_profile or ""),
    )
    print("production_mode=full_fit_retrain")
    print(f"source_formal_run={source_run_dir}")
    print(f"production_run_tag={experiment_tag}")
    print(f"latest_completed_date={latest_completed_date}")
    print(f"latest_trainable_date={latest_trainable_date}")
    print(f"internal_monitor_start_date={internal_monitor_start_date}")
    print(f"internal_monitor_days={internal_monitor_days}")
    print(
        "default_static_fallback_profile="
        f"{str(args.static_fallback_profile or _resolve_default_static_execution_profile()).strip()}"
    )
    if args.research_objective_mode:
        print(f"override_research_objective_mode={args.research_objective_mode}")
    if args.checkpoint_selection_objective:
        print(f"override_checkpoint_selection_objective={args.checkpoint_selection_objective}")
    if args.checkpoint_selection_min_improvement is not None:
        print(f"override_checkpoint_selection_min_improvement={args.checkpoint_selection_min_improvement}")
    if args.execution_alignment_objective:
        print(f"override_execution_alignment_objective={args.execution_alignment_objective}")
    if args.execution_alignment_mode:
        print(f"override_execution_alignment_mode={args.execution_alignment_mode}")
    if args.execution_alignment_profile:
        print(f"override_execution_alignment_profile={args.execution_alignment_profile}")
    if args.execution_alignment_candidate_profiles:
        print(f"override_execution_alignment_candidate_profiles={args.execution_alignment_candidate_profiles}")
    if args.static_fallback_profile:
        print(f"override_static_fallback_profile={args.static_fallback_profile}")
    subprocess.run(cmd, check=True)

    run_dir = (Path("daily_research/output") / experiment_tag).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Production retrain run directory not found after training: {run_dir}")
    _sync_production_root(
        run_dir=run_dir,
        production_root=production_root,
        source_run_dir=source_run_dir,
        latest_completed_date=latest_completed_date,
        latest_trainable_date=latest_trainable_date,
        internal_monitor_start_date=internal_monitor_start_date,
        internal_monitor_days=internal_monitor_days,
        train_start_date=train_start_date,
        static_fallback_profile=str(args.static_fallback_profile or _resolve_default_static_execution_profile()).strip(),
        strategy_manifest_path=strategy_manifest_path,
        activate_strategy=bool(args.activate_strategy),
    )
    if args.activate_strategy:
        active_manifest_path, active_payload = _activate_strategy(
            source_run_dir=source_run_dir,
            production_root=production_root,
            strategy_manifest_path=strategy_manifest_path,
            strategy_name=args.strategy_name,
            strategy_panel_mode=args.strategy_panel_mode,
        )
        print(f"active_execution_strategy_manifest={active_manifest_path}")
        print(f"active_execution_strategy_name={active_payload.get('strategy_name', '')}")
        print(f"active_execution_panel_mode={active_payload.get('panel_mode', '')}")
        print(f"active_execution_candidate_label={active_payload.get('candidate_label', '')}")
    print(f"production_root={production_root}")
    print(f"active_production_run={run_dir}")


if __name__ == "__main__":
    main()
