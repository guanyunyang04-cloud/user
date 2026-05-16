from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.continuous_policy.label_builder import LABEL_CONFIGS, build_future_path_metrics
from daily_research.continuous_policy.model import fit_policy_models
from daily_research.continuous_policy.decision_core_v6 import (
    DECISION_CORE_V6_ARTIFACT_FILENAME,
    DECISION_CORE_V6_LOSS_PROFILE_NAMES,
    DECISION_CORE_V6_PROFILE,
    DECISION_CORE_V6_STRICT_GOLD_DATASET_ID,
    decision_core_v6_profile_name,
    fit_policy_models_decision_core_v6,
)
from daily_research.continuous_policy.model_core_v4 import CORE_V4_LOSS_PROFILE_NAMES, fit_policy_models_core_v4
from daily_research.continuous_policy.model_hier_v4 import fit_policy_models_v4
from daily_research.continuous_policy.model_portfolio_set_v5 import (
    PORTFOLIO_SET_V5_ARTIFACT_FILENAME,
    PORTFOLIO_SET_V5_DEFAULT_STRICT_GOLD_DATASET_ID,
    PORTFOLIO_SET_V5_DFL_PG_V1_VERSION,
    PORTFOLIO_SET_V5_LOSS_PROFILE_NAMES,
    fit_policy_models_portfolio_set_v5,
)
from daily_research.continuous_policy.model_v2 import DECODER_PROFILE_NAMES, fit_policy_models_v2
from daily_research.continuous_policy.model_seq_v3 import (
    DAILY_HEAD_LAYOUT_CHOICES,
    DAILY_HEAD_LAYOUT_MONOLITHIC_V1,
    DEFAULT_LOSS_PROFILE,
    LOSS_PROFILE_NAMES,
    fit_policy_models_v3,
)
from daily_research.continuous_policy.pipeline_utils import (
    BUDGET_OBJECTIVE_CHOICES,
    DEFAULT_BUDGET_OBJECTIVE,
    build_training_matrices,
    select_daily_feature_columns,
    select_feature_columns,
)
from daily_research.continuous_policy.portfolio_simulator import (
    BUDGET_CALIBRATION_CHOICES,
    BUDGET_SEMANTICS_CHOICES,
    DEFAULT_BUDGET_CALIBRATION,
    DEFAULT_BUDGET_SEMANTICS,
    DEFAULT_EXECUTION_SEMANTICS,
    EXECUTION_SEMANTICS_CHOICES,
)
from daily_research.continuous_policy.runtime import (
    MODELS_ROOT,
    now_iso,
    read_json,
    safe_print_json,
    timestamp_tag,
    update_latest_summary,
    write_json,
)
from daily_research.continuous_policy.runtime_progress import JsonlProgressSink
from daily_research.continuous_policy.state_builder import DEFAULT_ALPHA_PRIOR_SOURCE, prepare_policy_inputs, resolve_active_policy_defaults
from daily_research.continuous_policy.training_dataset_cache import (
    build_training_dataset_cache_spec,
    load_reusable_training_dataset,
    save_reusable_training_dataset,
)
from daily_research.data_lake import build_label_completeness_summary
from daily_research.continuous_policy.training_contracts import (
    TRAINER_BACKENDS,
    TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
    TRAINER_BACKEND_FORMAL_CORE_V4,
    TRAINER_BACKEND_FORMAL_HIER_V4,
    TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
    TRAINER_BACKEND_FORMAL_V2,
    TRAINER_BACKEND_FORMAL_SEQ_V3,
    TRAINER_BACKEND_PROTOTYPE_V1,
    build_training_contract,
    normalize_trainer_backend,
    summarize_training_contract,
)


def build_parser() -> argparse.ArgumentParser:
    defaults = resolve_active_policy_defaults()
    parser = argparse.ArgumentParser(description="Train the continuous portfolio policy stack.")
    parser.add_argument(
        "--pool-name",
        default=defaults["pool_name"] or "liquid500",
        help="Rolling liquidity pool name, or `all_a` / `learned_all_a` to let the policy learn selection over the whole A-share universe.",
    )
    parser.add_argument("--start-date", default=defaults["start_date"] or "20250318")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default=defaults["benchmark"] or "000300.SH")
    parser.add_argument("--data-source", default="tq", choices=("tq", "csv", "lake"))
    parser.add_argument(
        "--decision-core",
        default="",
        choices=("", "v6"),
        help="Explicit decision core selector. v6 requires formal_torch_decision_core_v6 and strict Gold training.",
    )
    parser.add_argument("--csv-folder", default="")
    parser.add_argument(
        "--lake-dataset-id",
        default="",
        help="Optional policy_input_bundle dataset id for data-source=lake prepare paths. Training v5 still defaults to strict Gold unless --training-dataset-id overrides it.",
    )
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--max-universe-size", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--skip-multiplier", type=float, default=2.0)
    parser.add_argument(
        "--execution-semantics",
        default=DEFAULT_EXECUTION_SEMANTICS,
        choices=EXECUTION_SEMANTICS_CHOICES,
        help="How portfolio execution records lifecycle actions: semantic_preserving_v1 keeps model intent separate from weight-change orders.",
    )
    parser.add_argument(
        "--budget-semantics",
        default=DEFAULT_BUDGET_SEMANTICS,
        choices=BUDGET_SEMANTICS_CHOICES,
        help="How candidate budget is applied: action_budget_split_v1 protects existing lifecycle actions and budgets new entries separately.",
    )
    parser.add_argument(
        "--budget-calibration",
        default=DEFAULT_BUDGET_CALIBRATION,
        choices=BUDGET_CALIBRATION_CHOICES,
        help="Optional portfolio-level gross/candidate/turnover calibration.",
    )
    parser.add_argument(
        "--budget-objective",
        default=DEFAULT_BUDGET_OBJECTIVE,
        choices=BUDGET_OBJECTIVE_CHOICES,
        help="Daily budget target objective. result_value_v1 adjusts the five controller targets from future return/risk value signals.",
    )
    parser.add_argument(
        "--alpha-prior-source",
        default=DEFAULT_ALPHA_PRIOR_SOURCE,
        help="Alpha prior source: none, active_execution_strategy, a manifest JSON, a run directory, or an explicit CSV source.",
    )
    parser.add_argument("--alpha-prior-score-panel", default="")
    parser.add_argument("--alpha-prior-target-weight-panel", default="")
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--training-dataset-cache-mode",
        default="auto",
        choices=("auto", "refresh", "off"),
        help="Reuse constructed training matrices by fingerprint. auto loads/writes, refresh rebuilds, off disables.",
    )
    parser.add_argument(
        "--training-dataset-store",
        default="lake",
        choices=("lake", "legacy"),
        help="Reusable training dataset store. lake writes DuckDB/Parquet catalog records and preserves legacy pickle fallback.",
    )
    parser.add_argument(
        "--data-lake-root",
        default="",
        help="Optional root for the local DuckDB/Parquet research data lake.",
    )
    parser.add_argument(
        "--training-dataset-id",
        default="",
        help="Optional DuckDB/Parquet Gold training dataset id to load directly. r65 portfolio-set v5 defaults to the full-universe strict Gold dataset when available.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and cache training matrices, write previews and summary, then exit without fitting a model.",
    )
    parser.add_argument(
        "--label-preset",
        default="balanced_v2",
        choices=tuple(sorted(LABEL_CONFIGS)),
        help="Lifecycle teacher label preset used to build the training target set.",
    )
    parser.add_argument(
        "--trainer-backend",
        default=TRAINER_BACKEND_FORMAL_V2,
        choices=TRAINER_BACKENDS,
        help="prototype_gbdt_v1 stays shadow-only; formal_torch_v2 / formal_torch_seq_v3 / formal_torch_hier_v4 are promotable epoch/resume backends; core_v4/v5 remain shadow-only.",
    )
    parser.add_argument(
        "--decoder-profile",
        default="default_v2",
        choices=DECODER_PROFILE_NAMES,
        help="Behavior decoder profile used by the torch formal backend.",
    )
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--min-epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--sequence-layers", type=int, default=1)
    parser.add_argument("--daily-hidden-dim", type=int, default=96)
    parser.add_argument(
        "--daily-head-layout",
        default=DAILY_HEAD_LAYOUT_MONOLITHIC_V1,
        choices=DAILY_HEAD_LAYOUT_CHOICES,
        help="Daily controller architecture. split_v2 separates exposure/deployment/lifecycle heads and enables budget timing auxiliaries.",
    )
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--daily-dropout", type=float, default=0.05)
    parser.add_argument(
        "--loss-profile",
        default=DEFAULT_LOSS_PROFILE,
        choices=tuple(
            sorted(
                set(LOSS_PROFILE_NAMES)
                | set(CORE_V4_LOSS_PROFILE_NAMES)
                | set(PORTFOLIO_SET_V5_LOSS_PROFILE_NAMES)
                | set(DECISION_CORE_V6_LOSS_PROFILE_NAMES)
            )
        ),
        help="Loss contract for seq_v3: use default imitation balance or teacher-aux continuous-primary profiles.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--resume-mode", default="strict", choices=("strict", "fresh"))
    parser.add_argument("--protocol-progress-jsonl", default="")
    parser.add_argument("--tag", default="")
    return parser


def _loss_profile_was_explicit(raw_argv: list[str]) -> bool:
    return any(token == "--loss-profile" or token.startswith("--loss-profile=") for token in raw_argv)


def _apply_backend_default_loss(args: argparse.Namespace, raw_argv: list[str]) -> str:
    backend = normalize_trainer_backend(args.trainer_backend)
    if str(getattr(args, "decision_core", "") or "").strip() == "v6":
        backend = TRAINER_BACKEND_FORMAL_DECISION_CORE_V6
        args.trainer_backend = backend
    if backend == TRAINER_BACKEND_FORMAL_DECISION_CORE_V6:
        if not _loss_profile_was_explicit(raw_argv):
            args.loss_profile = DECISION_CORE_V6_PROFILE
        decision_core_v6_profile_name(args.loss_profile)
    if backend == TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5 and not _loss_profile_was_explicit(raw_argv):
        args.loss_profile = PORTFOLIO_SET_V5_DFL_PG_V1_VERSION
    if backend == TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5 and str(args.loss_profile or "").strip() in {
        "portfolio_set_v5_dfl_pg_v1_r69_value_arbitration",
        "portfolio_set_v5_dfl_pg_v1_r71_multistage_regret",
        "portfolio_set_v5_dfl_pg_v1_r74_lake_behavior_quality",
    }:
        raise ValueError(
            f"{args.loss_profile} is a historical rXX continuous_policy profile and is rejected for new training; "
            f"use {DECISION_CORE_V6_PROFILE} with --decision-core v6."
        )
    return backend


def _validate_v6_train_args(args: argparse.Namespace, backend: str) -> None:
    if backend != TRAINER_BACKEND_FORMAL_DECISION_CORE_V6:
        return
    if str(getattr(args, "decision_core", "") or "").strip() != "v6":
        raise ValueError("formal_torch_decision_core_v6 requires --decision-core v6.")
    if str(getattr(args, "data_source", "") or "").strip().lower() != "lake":
        raise ValueError("formal_torch_decision_core_v6 requires --data-source lake.")
    training_dataset_id = str(getattr(args, "training_dataset_id", "") or "").strip()
    if training_dataset_id != DECISION_CORE_V6_STRICT_GOLD_DATASET_ID:
        raise ValueError(
            "formal_torch_decision_core_v6 requires "
            f"--training-dataset-id {DECISION_CORE_V6_STRICT_GOLD_DATASET_ID}."
        )
    if str(getattr(args, "lake_dataset_id", "") or "").strip() == "":
        raise ValueError("formal_torch_decision_core_v6 requires an explicit --lake-dataset-id.")
    decision_core_v6_profile_name(getattr(args, "loss_profile", ""))


def _build_common_train_summary(
    *,
    run_tag: str,
    trained_at: str,
    prepared,
    args: argparse.Namespace,
    feature_names: list[str],
    daily_feature_names: list[str],
    sample_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    teacher_summary: dict[str, object],
    training_contract: dict[str, object],
) -> dict[str, object]:
    return {
        "run_tag": run_tag,
        "trained_at": trained_at,
        "pool_name": prepared.pool_name,
        "benchmark": prepared.benchmark,
        "start_date": args.start_date,
        "end_date": args.end_date or prepared.end_date,
        "universe_size": len(prepared.universe),
        "feature_count": len(feature_names),
        "daily_feature_count": len(daily_feature_names),
        "sample_rows": int(len(sample_frame)),
        "daily_rows": int(len(daily_frame)),
        "label_preset": str(teacher_summary.get("label_preset", args.label_preset)),
        "trainer_backend": str(training_contract.get("trainer_backend", "") or ""),
        "decoder_profile": str(getattr(args, "decoder_profile", "default_v2") or "default_v2"),
        "loss_profile": str(getattr(args, "loss_profile", DEFAULT_LOSS_PROFILE) or DEFAULT_LOSS_PROFILE),
        "daily_head_layout": str(getattr(args, "daily_head_layout", DAILY_HEAD_LAYOUT_MONOLITHIC_V1) or DAILY_HEAD_LAYOUT_MONOLITHIC_V1),
        "execution_semantics": str(getattr(args, "execution_semantics", DEFAULT_EXECUTION_SEMANTICS) or DEFAULT_EXECUTION_SEMANTICS),
        "budget_semantics": str(getattr(args, "budget_semantics", DEFAULT_BUDGET_SEMANTICS) or DEFAULT_BUDGET_SEMANTICS),
        "budget_calibration": str(getattr(args, "budget_calibration", DEFAULT_BUDGET_CALIBRATION) or DEFAULT_BUDGET_CALIBRATION),
        "budget_objective": str(getattr(args, "budget_objective", DEFAULT_BUDGET_OBJECTIVE) or DEFAULT_BUDGET_OBJECTIVE),
        "alpha_prior_source": str(getattr(args, "alpha_prior_source", DEFAULT_ALPHA_PRIOR_SOURCE) or DEFAULT_ALPHA_PRIOR_SOURCE),
        "alpha_prior_score_panel": str(getattr(args, "alpha_prior_score_panel", "") or ""),
        "alpha_prior_target_weight_panel": str(getattr(args, "alpha_prior_target_weight_panel", "") or ""),
        "training_contract": summarize_training_contract(training_contract),
        "action_distribution": {
            str(key): int(value)
            for key, value in sample_frame["action_label"].astype(str).value_counts().sort_index().items()
        },
        "duration_distribution": {
            str(key): int(value)
            for key, value in sample_frame["planned_holding_bucket"].astype(str).value_counts().sort_index().items()
        },
        "teacher_summary": teacher_summary,
        "prepared_summary": prepared.to_summary(),
    }


def _try_reuse_decision_core_v6_training(
    *,
    run_root: Path,
    args: argparse.Namespace,
    training_contract: dict[str, object],
    progress_sink: JsonlProgressSink,
) -> dict[str, object] | None:
    if str(getattr(args, "resume_mode", "") or "") != "strict":
        return None
    artifact_path = run_root / DECISION_CORE_V6_ARTIFACT_FILENAME
    summary_path = run_root / "train_summary.json"
    if not artifact_path.exists() or not summary_path.exists():
        return None
    summary = read_json(summary_path)
    diagnostics = dict(summary.get("training_diagnostics", {}) or {})
    summary_contract = dict(summary.get("training_contract", {}) or {})
    cache_summary = dict(summary.get("training_dataset_cache", {}) or {})
    required_dataset_id = str(getattr(args, "training_dataset_id", "") or "").strip() or DECISION_CORE_V6_STRICT_GOLD_DATASET_ID
    checks = {
        "artifact_schema": str(diagnostics.get("artifact_schema", "") or "") == "continuous_policy_decision_core_v6_artifact",
        "trainer_backend": str(summary.get("trainer_backend", "") or summary_contract.get("trainer_backend", "") or "")
        == TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
        "loss_profile": str(summary.get("loss_profile", "") or "") == DECISION_CORE_V6_PROFILE,
        "dataset_id": str(cache_summary.get("dataset_id", "") or diagnostics.get("training_dataset_id", "") or "")
        == required_dataset_id,
        "completed_epochs": int(diagnostics.get("completed_epochs", 0) or 0) >= int(training_contract.get("min_epochs", 32) or 32),
        "best_epoch_edge": int(diagnostics.get("best_epoch", 0) or 0)
        <= max(int(diagnostics.get("completed_epochs", 0) or 0) - 2, 0),
        "model_artifact_path": Path(str(summary.get("model_artifact_path", artifact_path) or artifact_path)).expanduser().exists(),
    }
    if not all(checks.values()):
        progress_sink.emit(
            "train_strict_resume_rejected",
            decision_core_version="v6",
            existing_artifact_path=str(artifact_path.resolve()),
            failed_checks=[name for name, ok in checks.items() if not ok],
        )
        return None
    reused_summary = dict(summary)
    reused_diagnostics = dict(reused_summary.get("training_diagnostics", {}) or {})
    reused_diagnostics["strict_resume_reused_artifact"] = True
    reused_diagnostics["strict_resume_reused_at"] = now_iso()
    reused_summary["training_diagnostics"] = reused_diagnostics
    reused_summary["model_artifact_path"] = str(artifact_path.resolve())
    write_json(summary_path, reused_summary)
    update_latest_summary("train", reused_summary)
    progress_sink.emit(
        "train_strict_resume_reused_artifact",
        decision_core_version="v6",
        artifact_path=str(artifact_path.resolve()),
        completed_epochs=int(reused_diagnostics.get("completed_epochs", 0) or 0),
        best_epoch=int(reused_diagnostics.get("best_epoch", 0) or 0),
    )
    return reused_summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    backend = _apply_backend_default_loss(args, raw_argv)
    _validate_v6_train_args(args, backend)
    run_tag = str(args.tag or timestamp_tag("train"))
    run_root = MODELS_ROOT / run_tag
    run_root.mkdir(parents=True, exist_ok=True)
    progress_sink = JsonlProgressSink(
        args.protocol_progress_jsonl if str(args.protocol_progress_jsonl or "").strip() else None,
        run_tag=run_tag,
        stage="train",
    )
    data_prepare_stop = threading.Event()

    def data_prepare_heartbeat() -> None:
        while not data_prepare_stop.wait(120.0):
            progress_sink.emit("train_data_prepare_heartbeat")

    heartbeat_thread: threading.Thread | None = None
    if getattr(progress_sink, "path", None) is not None:
        progress_sink.emit("train_data_prepare_start")
        heartbeat_thread = threading.Thread(
            target=data_prepare_heartbeat,
            name=f"continuous-policy-train-prepare-{run_tag}",
            daemon=True,
        )
        heartbeat_thread.start()

    training_contract = build_training_contract(
        trainer_backend=backend,
        runtime_env="yolos",
        requested_epochs=args.epochs,
        min_epochs=args.min_epochs,
        resume_mode=args.resume_mode,
    )
    if backend == TRAINER_BACKEND_FORMAL_DECISION_CORE_V6:
        reused_summary = _try_reuse_decision_core_v6_training(
            run_root=run_root,
            args=args,
            training_contract=training_contract,
            progress_sink=progress_sink,
        )
        if reused_summary is not None:
            data_prepare_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
            safe_print_json(reused_summary)
            return 0
    direct_dataset_record = None
    direct_dataset_id = str(getattr(args, "training_dataset_id", "") or "").strip()
    if not direct_dataset_id and backend == TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5:
        direct_dataset_id = PORTFOLIO_SET_V5_DEFAULT_STRICT_GOLD_DATASET_ID
    if not direct_dataset_id and backend == TRAINER_BACKEND_FORMAL_DECISION_CORE_V6:
        direct_dataset_id = DECISION_CORE_V6_STRICT_GOLD_DATASET_ID
    if direct_dataset_id:
        try:
            from daily_research.data_lake import ResearchDataLake

            direct_dataset_record = ResearchDataLake(str(getattr(args, "data_lake_root", "") or "") or None).load_training_dataset(direct_dataset_id)
        except Exception:
            if str(getattr(args, "training_dataset_id", "") or "").strip():
                raise
            direct_dataset_record = None

    if direct_dataset_record is not None:
        sample_frame = direct_dataset_record.sample_frame.copy()
        daily_frame = direct_dataset_record.daily_frame.copy()
        teacher_summary = dict(direct_dataset_record.teacher_summary)
        label_summary = dict(direct_dataset_record.label_completeness_summary or {})
        if backend in {TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5, TRAINER_BACKEND_FORMAL_DECISION_CORE_V6} and not bool(label_summary.get("is_training_safe", False)):
            raise ValueError(f"portfolio-set v5 requires a strict training-safe Gold dataset: {direct_dataset_id}")
        prepared_summary = {
            "pool_name": str(direct_dataset_record.metadata.get("universe", args.pool_name) or args.pool_name),
            "benchmark": str(direct_dataset_record.metadata.get("benchmark", args.benchmark) or args.benchmark),
            "start_date": str(direct_dataset_record.metadata.get("start_date", args.start_date) or args.start_date),
            "end_date": str(direct_dataset_record.metadata.get("end_date", args.end_date) or args.end_date),
            "universe_size": int(direct_dataset_record.metadata.get("universe_size", 0) or 0),
            "source": "data_lake_direct",
            "dataset_id": str(direct_dataset_record.dataset_id),
        }

        class _PreparedSummary:
            pool_name = prepared_summary["pool_name"]
            benchmark = prepared_summary["benchmark"]
            end_date = prepared_summary["end_date"]
            universe = list(sample_frame["stock"].astype(str).unique()) if "stock" in sample_frame.columns else []

            def to_summary(self) -> dict[str, object]:
                return dict(prepared_summary)

        prepared = _PreparedSummary()
        training_dataset_cache_spec = dict(direct_dataset_record.metadata.get("parameters", {}) or {})
        training_dataset_cache_summary: dict[str, object] = {
            "mode": "direct_dataset_id",
            "status": "hit",
            "store": "data_lake",
            "dataset_id": str(direct_dataset_record.dataset_id),
            "cache_key": str(direct_dataset_record.fingerprint),
            "cache_dir": str(direct_dataset_record.root.resolve()),
            "sample_rows": int(len(sample_frame)),
            "daily_rows": int(len(daily_frame)),
            "label_completeness_summary": label_summary,
        }
        progress_sink.emit(
            "train_dataset_cache_hit",
            cache_key=str(direct_dataset_record.fingerprint),
            dataset_id=str(direct_dataset_record.dataset_id),
            sample_rows=int(len(sample_frame)),
            daily_rows=int(len(daily_frame)),
        )
    else:
        prepared = prepare_policy_inputs(
            pool_name=args.pool_name,
            start_date=args.start_date,
            end_date=args.end_date,
            benchmark=args.benchmark,
            data_source=args.data_source,
            csv_folder=args.csv_folder,
            lake_dataset_id=args.lake_dataset_id,
            data_lake_root=args.data_lake_root,
            max_universe_size=args.max_universe_size,
            pool_rebalance_days=args.pool_rebalance_days,
            pool_adv_window=args.pool_adv_window,
            alpha_prior_source=args.alpha_prior_source,
            alpha_prior_score_panel=args.alpha_prior_score_panel,
            alpha_prior_target_weight_panel=args.alpha_prior_target_weight_panel,
            refresh_cache=args.refresh_cache,
            progress_desc="continuous policy train",
        )
        prepared_summary = prepared.to_summary()
        training_dataset_cache_spec = build_training_dataset_cache_spec(
            args=args,
            prepared_summary=prepared_summary,
            universe=list(prepared.universe),
        )
        training_dataset_cache_summary = {}
        sample_frame = pd.DataFrame()
        daily_frame = pd.DataFrame()
        teacher_summary = {}
    cache_mode = str(getattr(args, "training_dataset_cache_mode", "auto") or "auto")
    cache_record = None
    if direct_dataset_record is None:
        training_dataset_cache_summary = {
            "mode": cache_mode,
            "status": "off" if cache_mode == "off" else "miss",
            "store": str(getattr(args, "training_dataset_store", "lake") or "lake"),
            "cache_key": "",
            "cache_dir": "",
        }
    if direct_dataset_record is None and cache_mode != "off" and cache_mode != "refresh" and not bool(getattr(args, "refresh_cache", False)):
        cache_record = load_reusable_training_dataset(
            spec=training_dataset_cache_spec,
            lake_root=str(getattr(args, "data_lake_root", "") or "") or None,
            prefer_data_lake=str(getattr(args, "training_dataset_store", "lake") or "lake") == "lake",
            zone="strict_train",
        )
    if direct_dataset_record is None and cache_record is not None:
        sample_frame = cache_record.sample_frame.copy()
        daily_frame = cache_record.daily_frame.copy()
        teacher_summary = dict(cache_record.teacher_summary)
        training_dataset_cache_summary = {
            "mode": cache_mode,
            "status": "hit",
            "store": str(cache_record.store),
            "cache_key": cache_record.cache_key,
            "cache_dir": str(cache_record.cache_dir.resolve()),
            "sample_rows": int(len(sample_frame)),
            "daily_rows": int(len(daily_frame)),
        }
        progress_sink.emit(
            "train_dataset_cache_hit",
            cache_key=cache_record.cache_key,
            sample_rows=int(len(sample_frame)),
            daily_rows=int(len(daily_frame)),
        )
    elif direct_dataset_record is None:
        if cache_mode != "off":
            progress_sink.emit("train_dataset_cache_miss", cache_mode=cache_mode)
        future_metrics = build_future_path_metrics(prepared)
        sample_frame, daily_frame, teacher_summary = build_training_matrices(
            prepared=prepared,
            future_metrics=future_metrics,
            start_date=args.start_date,
            end_date=args.end_date,
            transaction_cost_bps=args.transaction_cost_bps,
            slippage_bps=args.slippage_bps,
            sell_tax_bps=args.sell_tax_bps,
            random_seed=args.random_seed,
            skip_multiplier=args.skip_multiplier,
            label_preset=args.label_preset,
            execution_semantics=args.execution_semantics,
            budget_semantics=args.budget_semantics,
            budget_calibration=args.budget_calibration,
            budget_objective=args.budget_objective,
        )
        if cache_mode != "off":
            label_completeness_summary = build_label_completeness_summary(
                sample_frame=sample_frame,
                daily_frame=daily_frame,
                zone="strict_train",
                available_trade_dates=list(prepared.close.index),
                max_forward_horizon=max(int(item) for item in getattr(future_metrics, "horizons", (20,))),
            )
            saved_cache = save_reusable_training_dataset(
                spec=training_dataset_cache_spec,
                sample_frame=sample_frame,
                daily_frame=daily_frame,
                teacher_summary=teacher_summary,
                lake_root=str(getattr(args, "data_lake_root", "") or "") or None,
                prefer_data_lake=str(getattr(args, "training_dataset_store", "lake") or "lake") == "lake",
                zone="strict_train",
                label_completeness_summary=label_completeness_summary,
            )
            training_dataset_cache_summary = {
                "mode": cache_mode,
                "status": "refreshed" if cache_mode == "refresh" or bool(getattr(args, "refresh_cache", False)) else "stored",
                "store": str(saved_cache.store),
                "cache_key": saved_cache.cache_key,
                "cache_dir": str(saved_cache.cache_dir.resolve()),
                "sample_rows": int(len(sample_frame)),
                "daily_rows": int(len(daily_frame)),
            }
            progress_sink.emit(
                "train_dataset_cache_store",
                cache_key=saved_cache.cache_key,
                sample_rows=int(len(sample_frame)),
                daily_rows=int(len(daily_frame)),
            )
    feature_names = select_feature_columns(sample_frame)
    daily_feature_names = select_daily_feature_columns(daily_frame)
    trained_at = now_iso()
    train_summary = _build_common_train_summary(
        run_tag=run_tag,
        trained_at=trained_at,
        prepared=prepared,
        args=args,
        feature_names=feature_names,
        daily_feature_names=daily_feature_names,
        sample_frame=sample_frame,
        daily_frame=daily_frame,
        teacher_summary=teacher_summary,
        training_contract=training_contract,
    )
    train_summary["training_dataset_cache"] = training_dataset_cache_summary
    progress_sink.emit(
        "train_data_prepare_complete",
        sample_rows=int(len(sample_frame)),
        daily_rows=int(len(daily_frame)),
        feature_count=int(len(feature_names)),
        daily_feature_count=int(len(daily_feature_names)),
    )
    data_prepare_stop.set()
    if heartbeat_thread is not None:
        heartbeat_thread.join(timeout=1.0)

    if bool(getattr(args, "prepare_only", False)):
        sample_frame.head(4000).to_csv(run_root / "training_samples_preview.csv", index=False, encoding="utf-8-sig")
        daily_frame.to_csv(run_root / "daily_training_targets.csv", index=False, encoding="utf-8-sig")
        summary_payload = {
            **train_summary,
            "prepare_only": True,
            "training_diagnostics": {
                "status": "prepare_only_complete",
                "trainer_backend": str(training_contract.get("trainer_backend", "") or ""),
                "train_sample_rows": int(len(sample_frame)),
                "daily_rows": int(len(daily_frame)),
                "model_fit_skipped": True,
            },
            "model_artifact_path": "",
            "sample_preview_csv": str((run_root / "training_samples_preview.csv").resolve()),
            "daily_training_targets_csv": str((run_root / "daily_training_targets.csv").resolve()),
        }
        write_json(run_root / "train_summary.json", summary_payload)
        safe_print_json(summary_payload)
        return 0

    if backend == TRAINER_BACKEND_FORMAL_V2:
        artifact = fit_policy_models_v2(
            sample_frame=sample_frame,
            daily_frame=daily_frame,
            feature_names=feature_names,
            daily_feature_names=daily_feature_names,
            random_seed=args.random_seed,
            train_summary=train_summary,
            trained_at=trained_at,
            training_contract=training_contract,
            run_root=run_root,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            daily_hidden_dim=args.daily_hidden_dim,
            dropout=args.dropout,
            daily_dropout=args.daily_dropout,
            early_stop_patience=args.early_stop_patience,
            resume_mode=args.resume_mode,
        )
        artifact_path = run_root / "continuous_policy_v2_artifact.pt"
        training_diagnostics = dict(artifact.training_diagnostics or {})
    elif backend == TRAINER_BACKEND_FORMAL_CORE_V4:
        artifact = fit_policy_models_core_v4(
            sample_frame=sample_frame,
            daily_frame=daily_frame,
            feature_names=feature_names,
            daily_feature_names=daily_feature_names,
            random_seed=args.random_seed,
            train_summary=train_summary,
            trained_at=trained_at,
            training_contract=training_contract,
            run_root=run_root,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=max(int(args.hidden_dim), 128),
            sequence_layers=max(int(args.sequence_layers), 1),
            daily_hidden_dim=args.daily_hidden_dim,
            dropout=max(float(args.dropout), 0.10),
            daily_dropout=args.daily_dropout,
            early_stop_patience=args.early_stop_patience,
            resume_mode=args.resume_mode,
            loss_profile=args.loss_profile,
            progress_sink=progress_sink,
        )
        artifact_path = run_root / "continuous_policy_core_v4_artifact.pt"
        training_diagnostics = dict(artifact.training_diagnostics or {})
    elif backend == TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5:
        artifact = fit_policy_models_portfolio_set_v5(
            sample_frame=sample_frame,
            daily_frame=daily_frame,
            feature_names=feature_names,
            daily_feature_names=daily_feature_names,
            random_seed=args.random_seed,
            train_summary=train_summary,
            trained_at=trained_at,
            training_contract=training_contract,
            run_root=run_root,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            batch_size=max(1, min(int(args.batch_size), 2)),
            learning_rate=args.learning_rate,
            model_dim=max(int(args.hidden_dim), 16),
            temporal_layers=max(int(args.sequence_layers), 1),
            cross_layers=max(int(args.sequence_layers), 1),
            latent_count=max(4, int(args.daily_hidden_dim)),
            dropout=max(float(args.dropout), 0.05),
            early_stop_patience=args.early_stop_patience,
            resume_mode=args.resume_mode,
            loss_profile=args.loss_profile,
            progress_sink=progress_sink,
        )
        artifact_path = run_root / PORTFOLIO_SET_V5_ARTIFACT_FILENAME
        training_diagnostics = dict(artifact.training_diagnostics or {})
    elif backend == TRAINER_BACKEND_FORMAL_DECISION_CORE_V6:
        artifact = fit_policy_models_decision_core_v6(
            sample_frame=sample_frame,
            daily_frame=daily_frame,
            feature_names=feature_names,
            daily_feature_names=daily_feature_names,
            random_seed=args.random_seed,
            train_summary=train_summary,
            trained_at=trained_at,
            training_contract=training_contract,
            run_root=run_root,
            epochs=max(int(args.epochs), 32),
            min_epochs=max(int(args.min_epochs), 32),
            batch_size=max(1, min(int(args.batch_size), 2)),
            learning_rate=args.learning_rate,
            model_dim=max(int(args.hidden_dim), 16),
            temporal_layers=max(int(args.sequence_layers), 1),
            cross_layers=max(int(args.sequence_layers), 1),
            latent_count=max(4, int(args.daily_hidden_dim)),
            dropout=max(float(args.dropout), 0.05),
            early_stop_patience=args.early_stop_patience,
            resume_mode=args.resume_mode,
            decision_profile=args.loss_profile,
            progress_sink=progress_sink,
        )
        artifact_path = run_root / DECISION_CORE_V6_ARTIFACT_FILENAME
        training_diagnostics = dict(artifact.training_diagnostics or {})
    elif backend == TRAINER_BACKEND_FORMAL_HIER_V4:
        artifact = fit_policy_models_v4(
            sample_frame=sample_frame,
            daily_frame=daily_frame,
            feature_names=feature_names,
            daily_feature_names=daily_feature_names,
            random_seed=args.random_seed,
            train_summary=train_summary,
            trained_at=trained_at,
            training_contract=training_contract,
            run_root=run_root,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            batch_size=max(2, min(int(args.batch_size), 8)),
            learning_rate=min(float(args.learning_rate), 1.0e-3),
            model_dim=max(int(args.hidden_dim), 256),
            temporal_layers=3,
            cross_layers=3,
            dropout=max(float(args.dropout), 0.10),
            early_stop_patience=args.early_stop_patience,
            resume_mode=args.resume_mode,
        )
        artifact_path = run_root / "continuous_policy_hier_v4_artifact.pt"
        training_diagnostics = dict(artifact.training_diagnostics or {})
    elif backend == TRAINER_BACKEND_FORMAL_SEQ_V3:
        artifact = fit_policy_models_v3(
            sample_frame=sample_frame,
            daily_frame=daily_frame,
            feature_names=feature_names,
            daily_feature_names=daily_feature_names,
            random_seed=args.random_seed,
            train_summary=train_summary,
            trained_at=trained_at,
            training_contract=training_contract,
            run_root=run_root,
            epochs=args.epochs,
            min_epochs=args.min_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=max(int(args.hidden_dim), 224),
            sequence_hidden_dim=max(int(args.hidden_dim // 2), 96),
            sequence_layers=max(int(args.sequence_layers), 1),
            daily_hidden_dim=args.daily_hidden_dim,
            daily_head_layout=args.daily_head_layout,
            dropout=max(float(args.dropout), 0.10),
            daily_dropout=args.daily_dropout,
            early_stop_patience=args.early_stop_patience,
            resume_mode=args.resume_mode,
            loss_profile=args.loss_profile,
            progress_sink=progress_sink,
        )
        artifact_path = run_root / "continuous_policy_v3_seq_artifact.pt"
        training_diagnostics = dict(artifact.training_diagnostics or {})
    else:
        artifact = fit_policy_models(
            sample_frame=sample_frame,
            daily_frame=daily_frame,
            feature_names=feature_names,
            daily_feature_names=daily_feature_names,
            random_seed=args.random_seed,
            train_summary=train_summary,
            trained_at=trained_at,
        )
        artifact_path = artifact.save(run_root / "continuous_policy_artifact.pkl")
        training_diagnostics = {
            "trainer_backend": TRAINER_BACKEND_PROTOTYPE_V1,
            "status": "prototype_complete",
            "device": "cpu",
            "resume_mode": "",
            "completed_epochs": 0,
            "best_epoch": 0,
            "notes": [
                "prototype_gbdt_v1 is a non-epoch sklearn teacher/prototype backend.",
                "It remains shadow-only and is not promotion-eligible.",
                f"Decoder profile recorded for downstream compatibility: {args.decoder_profile}.",
            ],
        }

    sample_frame.head(4000).to_csv(run_root / "training_samples_preview.csv", index=False, encoding="utf-8-sig")
    daily_frame.to_csv(run_root / "daily_training_targets.csv", index=False, encoding="utf-8-sig")
    summary_payload = {
        **train_summary,
        "training_diagnostics": training_diagnostics,
        "model_artifact_path": str(Path(artifact_path).resolve()),
        "sample_preview_csv": str((run_root / "training_samples_preview.csv").resolve()),
        "daily_training_targets_csv": str((run_root / "daily_training_targets.csv").resolve()),
    }
    write_json(run_root / "train_summary.json", summary_payload)
    update_latest_summary("train", summary_payload)
    safe_print_json(summary_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
