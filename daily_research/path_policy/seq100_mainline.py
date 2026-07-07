from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from daily_research.path_policy.qdp_v2_sequence_path_training import main as sequence_training_main

DEFAULT_STORE_VIEW = Path("daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json")
DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/sequence_path_training")
DEFAULT_RUN_TAG = "seq100_todayclose_path_only_mainline"
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
        "gru_path_value",
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


def mainline_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mainline_id": "seq100_todayclose_path_only",
        "default_store_view": str(DEFAULT_STORE_VIEW),
        "active_concepts": ACTIVE_CONCEPTS,
        "comparison_concepts": COMPARISON_CONCEPTS,
        "archived_concepts": ARCHIVED_CONCEPTS,
        "default_train_profile": asdict(TodayClosePathOnlyProfile()),
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
                    row.get("alpha_path_trade_value_v2_60d")
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
    train.add_argument("--store-view", type=Path, default=DEFAULT_STORE_VIEW)
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    train.add_argument("--max-samples-per-split", type=int, default=0)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--json", action="store_true")

    summarize = sub.add_parser("summarize", help="Print a compact sequence run summary.")
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "contract":
        _print_payload(mainline_contract(), as_json=bool(args.json))
        return 0

    if args.command == "summarize":
        _print_payload(summarize_sequence_run(Path(args.run_dir)), as_json=bool(args.json))
        return 0

    if args.command == "train":
        profile = TodayClosePathOnlyProfile(
            store_view=Path(args.store_view),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            device=str(args.device),
            max_samples_per_split=int(args.max_samples_per_split),
        )
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
