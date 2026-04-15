from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.continuous_policy.label_builder import LABEL_CONFIGS, build_future_path_metrics
from daily_research.continuous_policy.model import fit_policy_models
from daily_research.continuous_policy.model_hier_v4 import fit_policy_models_v4
from daily_research.continuous_policy.model_v2 import DECODER_PROFILE_NAMES, fit_policy_models_v2
from daily_research.continuous_policy.model_seq_v3 import fit_policy_models_v3
from daily_research.continuous_policy.pipeline_utils import (
    build_training_matrices,
    select_daily_feature_columns,
    select_feature_columns,
)
from daily_research.continuous_policy.runtime import MODELS_ROOT, now_iso, timestamp_tag, update_latest_summary, write_json
from daily_research.continuous_policy.state_builder import prepare_policy_inputs, resolve_active_policy_defaults
from daily_research.continuous_policy.training_contracts import (
    TRAINER_BACKENDS,
    TRAINER_BACKEND_FORMAL_HIER_V4,
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
    parser.add_argument("--data-source", default="tq", choices=("tq", "csv"))
    parser.add_argument("--csv-folder", default="")
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)
    parser.add_argument("--max-universe-size", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--skip-multiplier", type=float, default=2.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--refresh-cache", action="store_true")
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
        help="prototype_gbdt_v1 stays shadow-only; formal_torch_v2 / formal_torch_seq_v3 / formal_torch_hier_v4 are promotable epoch/resume backends.",
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
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--daily-dropout", type=float, default=0.05)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--resume-mode", default="strict", choices=("strict", "fresh"))
    parser.add_argument("--tag", default="")
    return parser


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = normalize_trainer_backend(args.trainer_backend)
    run_tag = str(args.tag or timestamp_tag("train"))
    run_root = MODELS_ROOT / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    training_contract = build_training_contract(
        trainer_backend=backend,
        runtime_env="yolos",
        requested_epochs=args.epochs,
        min_epochs=args.min_epochs,
        resume_mode=args.resume_mode,
    )
    prepared = prepare_policy_inputs(
        pool_name=args.pool_name,
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark=args.benchmark,
        data_source=args.data_source,
        csv_folder=args.csv_folder,
        max_universe_size=args.max_universe_size,
        pool_rebalance_days=args.pool_rebalance_days,
        pool_adv_window=args.pool_adv_window,
        refresh_cache=args.refresh_cache,
        progress_desc="continuous policy train",
    )
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
            dropout=max(float(args.dropout), 0.10),
            daily_dropout=args.daily_dropout,
            early_stop_patience=args.early_stop_patience,
            resume_mode=args.resume_mode,
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
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
