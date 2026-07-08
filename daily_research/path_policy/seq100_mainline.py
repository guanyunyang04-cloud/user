from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from daily_research.path_policy.qdp_v2_sequence_path_training import (
    INPUT_CHANNEL_PROFILE_ALL,
    INPUT_CHANNEL_PROFILE_DAILY_ONLY,
    INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY,
    INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE,
    SUMMARY_LOSS_PROFILE_BASE,
    SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
    SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
    main as sequence_training_main,
)

DEFAULT_STORE_VIEW = Path("daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json")
DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/sequence_path_training")
DEFAULT_RUN_TAG = "seq100_todayclose_path_only_mainline"
DEFAULT_SUMMARY_V2_RUN_TAG = "seq100_todayclose_path_only_summary_v2"
DEFAULT_SUMMARY_V2_NO60_RUN_TAG = "seq100_todayclose_path_only_summary_v2_no60"
DEFAULT_DAILY_ONLY_RUN_TAG = "seq100_todayclose_path_only_daily_only"
DEFAULT_NO_INTRADAY_RUN_TAG = "seq100_todayclose_path_only_no_intraday_summary"
DEFAULT_NO_LIMIT_RUN_TAG = "seq100_todayclose_path_only_no_limit_structure"
DEFAULT_DIRECT_VALUE_5D_RUN_TAG = "seq100_direct_value_5d"
DEFAULT_DIRECT_VALUE_10D_RUN_TAG = "seq100_direct_value_10d"
DEFAULT_DIRECT_VALUE_60D_RUN_TAG = "seq100_direct_value_60d"
DEFAULT_TOP_K = "1,3,5,10,20,50,100"

ACTIVE_CONCEPTS = {
    "research_store_view": "Lightweight manifest under daily_research/data/research_store/views.",
    "seq100_x84_input": "Past 100 trading days of daily raw, daily state, intraday summary, and limit structure.",
    "today_close_anchor": "Future path returns are anchored on the signal-day close.",
    "future60_ohlc_path": "The default model predicts the future 60-day OHLC return path.",
    "path_trade_value_v2": "Ranking value is derived from the predicted path, including return, wait, drawdown, and cost semantics.",
    "path_value_spread": "TopK evaluation reports selected path value versus the same-day universe mean; it is not live PnL.",
}

COMPARISON_CONCEPTS = {
    "table_path60_baseline": "Tabular LightGBM-style baseline for aligned path60 comparison.",
    "path_only_next_open": "Retained anchor comparison; not the default mainline.",
    "rank_heavy_top1": "Observation branch for narrow Top1 behavior; not the default ranking objective.",
    "summary_v2_multi_horizon_ohlc": "Explicit experiment that keeps OHLC output but expands summary loss to OHLC-derived 5/10/20/40/60-day constraints.",
    "summary_v2_no60": "Explicit single-factor experiment that keeps summary_v2 OHLC constraints but removes the full 60-day window.",
    "daily_only_no_minute": "Explicit input ablation that keeps labels/loss fixed but removes intraday_summary and limit_structure input channels.",
    "no_intraday_summary": "Explicit input ablation that removes intraday_summary while retaining limit_structure.",
    "no_limit_structure": "Explicit input ablation that removes limit_structure while retaining intraday_summary.",
    "direct_value_rank_5d": "Explicit sequence ranker that directly learns 5-day path_trade_value_v2 instead of predicting future OHLC.",
    "direct_value_rank_10d": "Explicit sequence ranker that directly learns 10-day path_trade_value_v2 instead of predicting future OHLC.",
    "direct_value_rank_60d": "Explicit sequence ranker that directly learns 60-day path_trade_value_v2 instead of predicting future OHLC.",
}

ARCHIVED_CONCEPTS = {
    "alpha_v2": "Historical infrastructure and evidence line.",
    "path20": "Historical or compatibility horizon; not the current target definition.",
    "symbol_embedding": "Rejected as default until evidence changes.",
    "residual_score": "Comparison branch only.",
    "richer_target": "Comparison branch only.",
    "ohlcva_unified": "Paused branch; higher output complexity did not improve the mainline.",
}


@dataclass(frozen=True)
class TodayClosePathOnlyProfile:
    store_view: Path = DEFAULT_STORE_VIEW
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_tag: str = DEFAULT_RUN_TAG
    epochs: int = 10
    batch_size: int = 512
    hidden_dim: int = 128
    layers: int = 2
    dropout: float = 0.10
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    path_loss_weight: float = 0.45
    summary_loss_weight: float = 0.20
    richer_loss_weight: float = 0.0
    value_loss_weight: float = 0.20
    rank_loss_weight: float = 0.15
    rank_max_per_side: int = 64
    device: str = "auto"
    prediction_mode: str = "compact"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    top_k: str = DEFAULT_TOP_K
    max_samples_per_split: int = 0
    summary_loss_profile: str = SUMMARY_LOSS_PROFILE_BASE
    input_channel_profile: str = INPUT_CHANNEL_PROFILE_ALL
    model_type: str = "gru_path_value"
    direct_value_horizon: int = 0


def build_todayclose_path_only_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    return [
        "train",
        "--store-view",
        str(profile.store_view),
        "--output-root",
        str(profile.output_root),
        "--run-tag",
        profile.run_tag,
        "--epochs",
        str(profile.epochs),
        "--batch-size",
        str(profile.batch_size),
        "--model-type",
        profile.model_type,
        "--hidden-dim",
        str(profile.hidden_dim),
        "--layers",
        str(profile.layers),
        "--dropout",
        str(profile.dropout),
        "--learning-rate",
        str(profile.learning_rate),
        "--weight-decay",
        str(profile.weight_decay),
        "--path-loss-weight",
        str(profile.path_loss_weight),
        "--summary-loss-weight",
        str(profile.summary_loss_weight),
        "--richer-loss-weight",
        str(profile.richer_loss_weight),
        "--value-loss-weight",
        str(profile.value_loss_weight),
        "--rank-loss-weight",
        str(profile.rank_loss_weight),
        "--summary-loss-profile",
        profile.summary_loss_profile,
        "--input-channel-profile",
        profile.input_channel_profile,
        "--direct-value-horizon",
        str(profile.direct_value_horizon),
        "--rank-max-per-side",
        str(profile.rank_max_per_side),
        "--device",
        profile.device,
        "--top-k",
        profile.top_k,
        "--max-samples-per-split",
        str(profile.max_samples_per_split),
        "--prediction-mode",
        profile.prediction_mode,
        "--early-stopping-patience",
        str(profile.early_stopping_patience),
        "--early-stopping-min-delta",
        str(profile.early_stopping_min_delta),
    ]


def build_todayclose_summary_v2_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    summary_profile = TodayClosePathOnlyProfile(
        store_view=profile.store_view,
        output_root=profile.output_root,
        run_tag=profile.run_tag,
        epochs=profile.epochs,
        batch_size=profile.batch_size,
        hidden_dim=profile.hidden_dim,
        layers=profile.layers,
        dropout=profile.dropout,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        path_loss_weight=profile.path_loss_weight,
        summary_loss_weight=profile.summary_loss_weight,
        richer_loss_weight=profile.richer_loss_weight,
        value_loss_weight=profile.value_loss_weight,
        rank_loss_weight=profile.rank_loss_weight,
        rank_max_per_side=profile.rank_max_per_side,
        device=profile.device,
        prediction_mode=profile.prediction_mode,
        early_stopping_patience=profile.early_stopping_patience,
        early_stopping_min_delta=profile.early_stopping_min_delta,
        top_k=profile.top_k,
        max_samples_per_split=profile.max_samples_per_split,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
        input_channel_profile=profile.input_channel_profile,
        model_type=profile.model_type,
        direct_value_horizon=profile.direct_value_horizon,
    )
    return build_todayclose_path_only_train_argv(summary_profile)


def build_todayclose_summary_v2_no60_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    summary_profile = TodayClosePathOnlyProfile(
        store_view=profile.store_view,
        output_root=profile.output_root,
        run_tag=profile.run_tag,
        epochs=profile.epochs,
        batch_size=profile.batch_size,
        hidden_dim=profile.hidden_dim,
        layers=profile.layers,
        dropout=profile.dropout,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        path_loss_weight=profile.path_loss_weight,
        summary_loss_weight=profile.summary_loss_weight,
        richer_loss_weight=profile.richer_loss_weight,
        value_loss_weight=profile.value_loss_weight,
        rank_loss_weight=profile.rank_loss_weight,
        rank_max_per_side=profile.rank_max_per_side,
        device=profile.device,
        prediction_mode=profile.prediction_mode,
        early_stopping_patience=profile.early_stopping_patience,
        early_stopping_min_delta=profile.early_stopping_min_delta,
        top_k=profile.top_k,
        max_samples_per_split=profile.max_samples_per_split,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
        input_channel_profile=profile.input_channel_profile,
        model_type=profile.model_type,
        direct_value_horizon=profile.direct_value_horizon,
    )
    return build_todayclose_path_only_train_argv(summary_profile)


def build_todayclose_no_intraday_summary_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    ablation_profile = TodayClosePathOnlyProfile(
        store_view=profile.store_view,
        output_root=profile.output_root,
        run_tag=profile.run_tag,
        epochs=profile.epochs,
        batch_size=profile.batch_size,
        hidden_dim=profile.hidden_dim,
        layers=profile.layers,
        dropout=profile.dropout,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        path_loss_weight=profile.path_loss_weight,
        summary_loss_weight=profile.summary_loss_weight,
        richer_loss_weight=profile.richer_loss_weight,
        value_loss_weight=profile.value_loss_weight,
        rank_loss_weight=profile.rank_loss_weight,
        rank_max_per_side=profile.rank_max_per_side,
        device=profile.device,
        prediction_mode=profile.prediction_mode,
        early_stopping_patience=profile.early_stopping_patience,
        early_stopping_min_delta=profile.early_stopping_min_delta,
        top_k=profile.top_k,
        max_samples_per_split=profile.max_samples_per_split,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_BASE,
        input_channel_profile=INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY,
        model_type=profile.model_type,
        direct_value_horizon=profile.direct_value_horizon,
    )
    return build_todayclose_path_only_train_argv(ablation_profile)


def build_todayclose_no_limit_structure_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    ablation_profile = TodayClosePathOnlyProfile(
        store_view=profile.store_view,
        output_root=profile.output_root,
        run_tag=profile.run_tag,
        epochs=profile.epochs,
        batch_size=profile.batch_size,
        hidden_dim=profile.hidden_dim,
        layers=profile.layers,
        dropout=profile.dropout,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        path_loss_weight=profile.path_loss_weight,
        summary_loss_weight=profile.summary_loss_weight,
        richer_loss_weight=profile.richer_loss_weight,
        value_loss_weight=profile.value_loss_weight,
        rank_loss_weight=profile.rank_loss_weight,
        rank_max_per_side=profile.rank_max_per_side,
        device=profile.device,
        prediction_mode=profile.prediction_mode,
        early_stopping_patience=profile.early_stopping_patience,
        early_stopping_min_delta=profile.early_stopping_min_delta,
        top_k=profile.top_k,
        max_samples_per_split=profile.max_samples_per_split,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_BASE,
        input_channel_profile=INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE,
        model_type=profile.model_type,
        direct_value_horizon=profile.direct_value_horizon,
    )
    return build_todayclose_path_only_train_argv(ablation_profile)


def build_todayclose_daily_only_train_argv(profile: TodayClosePathOnlyProfile) -> list[str]:
    daily_only_profile = TodayClosePathOnlyProfile(
        store_view=profile.store_view,
        output_root=profile.output_root,
        run_tag=profile.run_tag,
        epochs=profile.epochs,
        batch_size=profile.batch_size,
        hidden_dim=profile.hidden_dim,
        layers=profile.layers,
        dropout=profile.dropout,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        path_loss_weight=profile.path_loss_weight,
        summary_loss_weight=profile.summary_loss_weight,
        richer_loss_weight=profile.richer_loss_weight,
        value_loss_weight=profile.value_loss_weight,
        rank_loss_weight=profile.rank_loss_weight,
        rank_max_per_side=profile.rank_max_per_side,
        device=profile.device,
        prediction_mode=profile.prediction_mode,
        early_stopping_patience=profile.early_stopping_patience,
        early_stopping_min_delta=profile.early_stopping_min_delta,
        top_k=profile.top_k,
        max_samples_per_split=profile.max_samples_per_split,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_BASE,
        input_channel_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
        model_type=profile.model_type,
        direct_value_horizon=profile.direct_value_horizon,
    )
    return build_todayclose_path_only_train_argv(daily_only_profile)


def build_todayclose_direct_value_train_argv(profile: TodayClosePathOnlyProfile, *, horizon: int) -> list[str]:
    direct_profile = TodayClosePathOnlyProfile(
        store_view=profile.store_view,
        output_root=profile.output_root,
        run_tag=profile.run_tag,
        epochs=profile.epochs,
        batch_size=profile.batch_size,
        hidden_dim=profile.hidden_dim,
        layers=profile.layers,
        dropout=profile.dropout,
        learning_rate=profile.learning_rate,
        weight_decay=profile.weight_decay,
        path_loss_weight=0.0,
        summary_loss_weight=0.0,
        richer_loss_weight=0.0,
        value_loss_weight=0.50,
        rank_loss_weight=0.50,
        rank_max_per_side=profile.rank_max_per_side,
        device=profile.device,
        prediction_mode=profile.prediction_mode,
        early_stopping_patience=profile.early_stopping_patience,
        early_stopping_min_delta=profile.early_stopping_min_delta,
        top_k=profile.top_k,
        max_samples_per_split=profile.max_samples_per_split,
        summary_loss_profile=SUMMARY_LOSS_PROFILE_BASE,
        input_channel_profile=profile.input_channel_profile,
        model_type="gru_direct_value",
        direct_value_horizon=int(horizon),
    )
    return build_todayclose_path_only_train_argv(direct_profile)


def mainline_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mainline_id": "seq100_todayclose_path_only",
        "default_store_view": str(DEFAULT_STORE_VIEW),
        "active_concepts": ACTIVE_CONCEPTS,
        "comparison_concepts": COMPARISON_CONCEPTS,
        "archived_concepts": ARCHIVED_CONCEPTS,
        "default_train_profile": asdict(TodayClosePathOnlyProfile()),
        "summary_v2_train_profile": asdict(
            TodayClosePathOnlyProfile(
                run_tag=DEFAULT_SUMMARY_V2_RUN_TAG,
                early_stopping_patience=2,
                summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
            )
        ),
        "summary_v2_no60_train_profile": asdict(
            TodayClosePathOnlyProfile(
                run_tag=DEFAULT_SUMMARY_V2_NO60_RUN_TAG,
                early_stopping_patience=2,
                summary_loss_profile=SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
            )
        ),
        "daily_only_train_profile": asdict(
            TodayClosePathOnlyProfile(
                run_tag=DEFAULT_DAILY_ONLY_RUN_TAG,
                early_stopping_patience=2,
                input_channel_profile=INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            )
        ),
        "input_ablation_train_profiles": {
            "no_intraday_summary": asdict(
                TodayClosePathOnlyProfile(
                    run_tag=DEFAULT_NO_INTRADAY_RUN_TAG,
                    early_stopping_patience=2,
                    input_channel_profile=INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY,
                )
            ),
            "no_limit_structure": asdict(
                TodayClosePathOnlyProfile(
                    run_tag=DEFAULT_NO_LIMIT_RUN_TAG,
                    early_stopping_patience=2,
                    input_channel_profile=INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE,
                )
            ),
        },
        "direct_value_train_profiles": {
            "5d": asdict(
                TodayClosePathOnlyProfile(
                    run_tag=DEFAULT_DIRECT_VALUE_5D_RUN_TAG,
                    batch_size=2048,
                    early_stopping_patience=2,
                    path_loss_weight=0.0,
                    summary_loss_weight=0.0,
                    value_loss_weight=0.50,
                    rank_loss_weight=0.50,
                    model_type="gru_direct_value",
                    direct_value_horizon=5,
                )
            ),
            "10d": asdict(
                TodayClosePathOnlyProfile(
                    run_tag=DEFAULT_DIRECT_VALUE_10D_RUN_TAG,
                    batch_size=2048,
                    early_stopping_patience=2,
                    path_loss_weight=0.0,
                    summary_loss_weight=0.0,
                    value_loss_weight=0.50,
                    rank_loss_weight=0.50,
                    model_type="gru_direct_value",
                    direct_value_horizon=10,
                )
            ),
            "60d": asdict(
                TodayClosePathOnlyProfile(
                    run_tag=DEFAULT_DIRECT_VALUE_60D_RUN_TAG,
                    batch_size=2048,
                    early_stopping_patience=2,
                    path_loss_weight=0.0,
                    summary_loss_weight=0.0,
                    value_loss_weight=0.50,
                    rank_loss_weight=0.50,
                    model_type="gru_direct_value",
                    direct_value_horizon=60,
                )
            ),
        },
        "evidence_boundary": (
            "Sequence path-value metrics are research evidence. They do not activate "
            "active_execution_strategy.json, live/default, broker, or trade-plan changes."
        ),
        "next_decision_surface": "execution_layer_backtest",
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_sequence_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "sequence_path_training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing sequence_path_training_summary.json under {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split_rows = _read_csv_rows(run_dir / "split_metrics.csv")
    topk_rows = _read_csv_rows(run_dir / "topk_metrics.csv")
    value_column = str(summary.get("value_column", "") or "")

    slim_splits = {}
    for row in split_rows:
        split = row.get("split", "")
        if split in {"validation", "test"}:
            slim_splits[split] = {
                "row_count": int(float(row.get("row_count") or 0)),
                "date_count": int(float(row.get("date_count") or 0)),
                "rank_ic_mean": float(row.get("rank_ic_mean") or "nan"),
                "rank_ic_positive_day_rate": float(row.get("rank_ic_positive_day_rate") or "nan"),
                "value_column": row.get("value_column", summary.get("value_column", "")),
            }

    slim_topk: dict[str, dict[str, dict[str, float]]] = {"validation": {}, "test": {}}
    for row in topk_rows:
        split = row.get("split", "")
        top_k = row.get("top_k", "")
        if split in slim_topk and top_k in {"1", "3", "10"}:
            slim_topk[split][top_k] = {
                "path_value_spread": float(
                    row.get(f"alpha_{value_column}")
                    or row.get("alpha_path_trade_value_v2_60d")
                    or row.get("alpha_path_trade_value_60d")
                    or "nan"
                ),
                "best_exit_day_mean": float(row.get("selected_best_exit_day_mean") or "nan"),
                "hit_10pct_rate": float(row.get("selected_hit_10pct_rate") or "nan"),
                "loss_5pct_rate": float(row.get("selected_loss_5pct_rate") or "nan"),
            }

    return {
        "run_dir": str(run_dir),
        "generated_at": summary.get("generated_at", ""),
        "pack_manifest": summary.get("pack_manifest", ""),
        "input_channel_profile": summary.get("input_channel_profile", ""),
        "input_channels": summary.get("input_channels", []),
        "model_input_dim": dict(summary.get("model", {}) or {}).get("input_dim", ""),
        "price_anchor": summary.get("price_anchor", ""),
        "value_column": summary.get("value_column", ""),
        "best_epoch": summary.get("best_epoch", ""),
        "loss_weights": summary.get("loss_weights", {}),
        "splits": slim_splits,
        "topk": slim_topk,
        "evidence_boundary": mainline_contract()["evidence_boundary"],
    }


def _print_payload(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Narrow CLI for the seq100 today-close path-only mainline.")
    sub = parser.add_subparsers(dest="command", required=True)

    contract = sub.add_parser("contract", help="Print the active concept contract.")
    contract.add_argument("--json", action="store_true")

    train = sub.add_parser("train", help="Train the fixed today-close path-only mainline.")
    _add_train_args(train, default_run_tag=DEFAULT_RUN_TAG, default_early_stopping_patience=3)

    train_summary_v2 = sub.add_parser(
        "train-summary-v2",
        help="Train the explicit multi-horizon OHLC summary-loss experiment.",
    )
    _add_train_args(train_summary_v2, default_run_tag=DEFAULT_SUMMARY_V2_RUN_TAG, default_early_stopping_patience=2)

    train_summary_v2_no60 = sub.add_parser(
        "train-summary-v2-no60",
        help="Train the summary_v2 single-factor experiment without the full 60-day window.",
    )
    _add_train_args(train_summary_v2_no60, default_run_tag=DEFAULT_SUMMARY_V2_NO60_RUN_TAG, default_early_stopping_patience=2)

    train_daily_only = sub.add_parser(
        "train-daily-only",
        help="Train the explicit no-minute daily-only input ablation.",
    )
    _add_train_args(train_daily_only, default_run_tag=DEFAULT_DAILY_ONLY_RUN_TAG, default_early_stopping_patience=2)

    train_no_intraday = sub.add_parser(
        "train-no-intraday-summary",
        help="Train the input ablation that removes intraday_summary only.",
    )
    _add_train_args(train_no_intraday, default_run_tag=DEFAULT_NO_INTRADAY_RUN_TAG, default_early_stopping_patience=2)

    train_no_limit = sub.add_parser(
        "train-no-limit-structure",
        help="Train the input ablation that removes limit_structure only.",
    )
    _add_train_args(train_no_limit, default_run_tag=DEFAULT_NO_LIMIT_RUN_TAG, default_early_stopping_patience=2)

    direct_value_5d = sub.add_parser("train-direct-value-5d", help="Train the direct 5-day path-value ranker.")
    _add_train_args(
        direct_value_5d,
        default_run_tag=DEFAULT_DIRECT_VALUE_5D_RUN_TAG,
        default_early_stopping_patience=2,
        default_batch_size=2048,
    )

    direct_value_10d = sub.add_parser("train-direct-value-10d", help="Train the direct 10-day path-value ranker.")
    _add_train_args(
        direct_value_10d,
        default_run_tag=DEFAULT_DIRECT_VALUE_10D_RUN_TAG,
        default_early_stopping_patience=2,
        default_batch_size=2048,
    )

    direct_value_60d = sub.add_parser("train-direct-value-60d", help="Train the direct 60-day path-value ranker.")
    _add_train_args(
        direct_value_60d,
        default_run_tag=DEFAULT_DIRECT_VALUE_60D_RUN_TAG,
        default_early_stopping_patience=2,
        default_batch_size=2048,
    )

    summarize = sub.add_parser("summarize", help="Print a compact sequence run summary.")
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--json", action="store_true")
    return parser


def _add_train_args(
    parser: argparse.ArgumentParser,
    *,
    default_run_tag: str,
    default_early_stopping_patience: int,
    default_batch_size: int = 512,
) -> None:
    parser.add_argument("--store-view", type=Path, default=DEFAULT_STORE_VIEW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=default_run_tag)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=int(default_batch_size))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=int(default_early_stopping_patience))
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "contract":
        _print_payload(mainline_contract(), as_json=bool(args.json))
        return 0

    if args.command == "summarize":
        _print_payload(summarize_sequence_run(Path(args.run_dir)), as_json=bool(args.json))
        return 0

    direct_horizon_by_command = {
        "train-direct-value-5d": 5,
        "train-direct-value-10d": 10,
        "train-direct-value-60d": 60,
    }
    input_profile_by_command = {
        "train-daily-only": INPUT_CHANNEL_PROFILE_DAILY_ONLY,
        "train-no-intraday-summary": INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY,
        "train-no-limit-structure": INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE,
    }
    summary_profile_by_command = {
        "train-summary-v2": SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
        "train-summary-v2-no60": SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
    }
    train_commands = {
        "train",
        "train-summary-v2",
        "train-summary-v2-no60",
        "train-daily-only",
        "train-no-intraday-summary",
        "train-no-limit-structure",
        *direct_horizon_by_command,
    }
    if args.command in train_commands:
        direct_horizon = int(direct_horizon_by_command.get(args.command, 0))
        profile = TodayClosePathOnlyProfile(
            store_view=Path(args.store_view),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=str(args.device),
            max_samples_per_split=int(args.max_samples_per_split),
            path_loss_weight=0.0 if int(direct_horizon) > 0 else 0.45,
            summary_loss_weight=0.0 if int(direct_horizon) > 0 else 0.20,
            richer_loss_weight=0.0,
            value_loss_weight=0.50 if int(direct_horizon) > 0 else 0.20,
            rank_loss_weight=0.50 if int(direct_horizon) > 0 else 0.15,
            early_stopping_patience=int(args.early_stopping_patience),
            early_stopping_min_delta=float(args.early_stopping_min_delta),
            summary_loss_profile=str(summary_profile_by_command.get(args.command, SUMMARY_LOSS_PROFILE_BASE)),
            input_channel_profile=str(input_profile_by_command.get(args.command, INPUT_CHANNEL_PROFILE_ALL)),
            model_type="gru_direct_value" if direct_horizon > 0 else "gru_path_value",
            direct_value_horizon=direct_horizon,
        )
        if args.command == "train-summary-v2":
            train_argv = build_todayclose_summary_v2_train_argv(profile)
        elif args.command == "train-summary-v2-no60":
            train_argv = build_todayclose_summary_v2_no60_train_argv(profile)
        elif args.command == "train-daily-only":
            train_argv = build_todayclose_daily_only_train_argv(profile)
        elif args.command == "train-no-intraday-summary":
            train_argv = build_todayclose_no_intraday_summary_train_argv(profile)
        elif args.command == "train-no-limit-structure":
            train_argv = build_todayclose_no_limit_structure_train_argv(profile)
        elif direct_horizon > 0:
            train_argv = build_todayclose_direct_value_train_argv(profile, horizon=direct_horizon)
        else:
            train_argv = build_todayclose_path_only_train_argv(profile)
        if bool(args.dry_run):
            _print_payload({"profile": asdict(profile), "argv": train_argv}, as_json=bool(args.json))
            return 0
        if bool(args.json):
            train_argv.append("--json")
        return sequence_training_main(train_argv)

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
