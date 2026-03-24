from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from daily_research.baseline.data_provider import (
    get_latest_completed_trading_date,
    load_daily_from_csv,
    load_daily_from_tq,
)


OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
STAGE_SEQUENCE = ("backbone", "score_head", "ranking")
SCORE_HEAD_CHOICES = ("manual", "ridge", "lgbm")


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    train_end: str
    valid_start: str
    valid_end: str

    @property
    def label(self) -> str:
        return f"{self.valid_start.replace('-', '')}_{self.valid_end.replace('-', '')}"


@dataclass(frozen=True)
class RecipeSpec:
    encoder_family: str
    use_pretrain: bool
    score_head_method: str
    ranking_profile: str
    ranking_loss_weight: float
    listwise_loss_weight: float

    @property
    def slug(self) -> str:
        encoder = "gru" if self.encoder_family == "gru" else "patch"
        pretrain = "maskedpre" if self.use_pretrain else "nopre"
        return f"enc-{encoder}__pre-{pretrain}__score-{self.score_head_method}__rank-{self.ranking_profile}"

    @property
    def display_name(self) -> str:
        return self.slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal sufficient deep_alpha comparison matrix in phased walk-forward form."
    )
    parser.add_argument("--phase", choices=["backbone", "score_head", "ranking", "all"], default="all")
    parser.add_argument("--root-tag", default=f"deep_alpha_minimal_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--dry-run", action="store_true", help="Only write the matrix plan and command list; do not execute runs.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse finished runs under the same root-tag.")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--window-count", type=int, default=3)
    parser.add_argument("--valid-days", type=int, default=252)

    parser.add_argument("--data-source", choices=["tq", "csv"], default="tq")
    parser.add_argument("--csv-folder", default=None)
    parser.add_argument("--start-date", default="20220101")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--rolling-liquidity-pool", choices=["liquid300", "liquid500", "liquid800"], default="liquid500")
    parser.add_argument("--pool-rebalance-days", type=int, default=21)
    parser.add_argument("--pool-adv-window", type=int, default=20)

    parser.add_argument("--lookback-window", type=int, default=120)
    parser.add_argument("--prediction-horizons", default="5,10,20")
    parser.add_argument("--holding-count", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="1d")
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--min-adv20", type=float, default=50_000.0)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--max-price", type=float, default=300.0)

    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--patch-len", type=int, default=5)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--safe-runtime-profile", action="store_true")
    parser.add_argument("--no-safe-runtime-profile", dest="safe_runtime_profile", action="store_false")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--market-state-count", type=int, default=4)

    parser.add_argument("--return-loss-mode", choices=["regression", "top_rest_bce", "top_bottom_bce"], default="regression")
    parser.add_argument("--return-target-transform", choices=["raw", "cs_rank", "cs_zscore"], default="raw")
    parser.add_argument("--score-risk-mode", choices=["subtract", "gate", "state_gate", "state_liquidity_gate"], default="state_gate")
    parser.add_argument("--score-risk-state-thresholds", default="0.0,0.2,0.35,0.5,0.65")
    parser.add_argument("--train-eval-window-days", type=int, default=126)
    parser.add_argument("--score-head-candidates", default="manual,ridge,lgbm")

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--lr-plateau-patience", type=int, default=1)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-improvement", type=float, default=1e-4)

    parser.add_argument("--ranking-loss-weight-on", type=float, default=0.5)
    parser.add_argument("--listwise-loss-weight-on", type=float, default=0.25)

    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument("--pretrain-weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--pretrain-min-epochs", type=int, default=8)
    parser.add_argument("--pretrain-early-stop-patience", type=int, default=3)
    parser.add_argument("--pretrain-lr-plateau-patience", type=int, default=2)
    parser.add_argument("--pretrain-lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--pretrain-min-improvement", type=float, default=1e-4)
    parser.add_argument("--pretrain-valid-days", type=int, default=63)
    parser.add_argument("--pretrain-mask-ratio", type=float, default=0.40)
    parser.add_argument("--auto-extend-undertrained", action="store_true")
    parser.add_argument("--no-auto-extend-undertrained", dest="auto_extend_undertrained", action="store_false")
    parser.add_argument("--epoch-extend-step", type=int, default=4)
    parser.add_argument("--max-total-epochs", type=int, default=20)

    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")

    parser.set_defaults(
        skip_existing=True,
        pin_memory=True,
        use_amp=True,
        safe_runtime_profile=True,
        auto_extend_undertrained=True,
        use_cache=True,
    )
    return parser.parse_args()


def _parse_score_head_candidates(raw: str) -> list[str]:
    candidates = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not candidates:
        raise ValueError("score-head-candidates 不能为空。")
    invalid = [item for item in candidates if item not in SCORE_HEAD_CHOICES]
    if invalid:
        raise ValueError(f"Unsupported score_head candidates: {invalid}")
    return candidates


def _matrix_root(root_tag: str) -> Path:
    return OUTPUT_ROOT / Path(root_tag)


def _run_dir(root_tag: str, recipe: RecipeSpec, window: WalkForwardWindow) -> Path:
    return OUTPUT_ROOT / Path(root_tag) / "runs" / f"{recipe.slug}_{window.label}"


def _run_tag(root_tag: str, recipe: RecipeSpec, window: WalkForwardWindow) -> str:
    return (Path(root_tag) / "runs" / f"{recipe.slug}_{window.label}").as_posix()


def _pretrain_dir(root_tag: str, window: WalkForwardWindow) -> Path:
    return OUTPUT_ROOT / Path(root_tag) / "pretrain" / f"patch_masked_{window.label}"


def _pretrain_tag(root_tag: str, window: WalkForwardWindow) -> str:
    return (Path(root_tag) / "pretrain" / f"patch_masked_{window.label}").as_posix()


def _selected_path(matrix_root: Path, stage_name: str) -> Path:
    return matrix_root / f"stage_{stage_name}_selected.json"


def _commands_path(matrix_root: Path, stage_name: str) -> Path:
    return matrix_root / f"stage_{stage_name}_commands.txt"


def _runs_path(matrix_root: Path, stage_name: str) -> Path:
    return matrix_root / f"stage_{stage_name}_runs.csv"


def _summary_path(matrix_root: Path, stage_name: str) -> Path:
    return matrix_root / f"stage_{stage_name}_recipe_summary.csv"


def _summary_md_path(matrix_root: Path, stage_name: str) -> Path:
    return matrix_root / f"stage_{stage_name}_summary.md"


def _load_benchmark_dates(args: argparse.Namespace) -> pd.DatetimeIndex:
    if args.data_source == "tq":
        end_date = args.end_date or pd.Timestamp(get_latest_completed_trading_date()).strftime("%Y%m%d")
        df_dict = load_daily_from_tq([args.benchmark], start_date=args.start_date, end_date=end_date)
        return pd.DatetimeIndex(pd.to_datetime(df_dict["Close"].index))

    if not args.csv_folder:
        raise ValueError("CSV mode requires --csv-folder.")
    df_dict = load_daily_from_csv(args.csv_folder)
    close = df_dict["Close"]
    if args.benchmark not in close.columns:
        raise ValueError(f"Benchmark {args.benchmark} not found in CSV close columns.")
    dates = pd.DatetimeIndex(pd.to_datetime(close.index))
    if args.end_date:
        dates = dates[dates <= pd.Timestamp(args.end_date)]
    return dates


def _build_recent_nonoverlap_windows(
    dates: pd.DatetimeIndex,
    *,
    valid_days: int,
    window_count: int,
) -> list[WalkForwardWindow]:
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    required = int(valid_days) * int(window_count) + 1
    if len(dates) < required:
        raise ValueError(
            f"Not enough trading dates for {window_count} windows x {valid_days} valid_days. "
            f"Need at least {required}, got {len(dates)}."
        )

    windows: list[WalkForwardWindow] = []
    end_pos = len(dates) - 1
    for _ in range(int(window_count)):
        start_pos = end_pos - int(valid_days) + 1
        if start_pos <= 0:
            raise ValueError("Unable to build the requested number of non-overlapping walk-forward windows.")
        train_end_pos = start_pos - 1
        windows.append(
            WalkForwardWindow(
                index=0,
                train_end=str(pd.Timestamp(dates[train_end_pos]).date()),
                valid_start=str(pd.Timestamp(dates[start_pos]).date()),
                valid_end=str(pd.Timestamp(dates[end_pos]).date()),
            )
        )
        end_pos = train_end_pos

    windows.reverse()
    return [
        WalkForwardWindow(
            index=idx + 1,
            train_end=window.train_end,
            valid_start=window.valid_start,
            valid_end=window.valid_end,
        )
        for idx, window in enumerate(windows)
    ]


def _backbone_recipes() -> list[RecipeSpec]:
    return [
        RecipeSpec(
            encoder_family="gru",
            use_pretrain=False,
            score_head_method="manual",
            ranking_profile="plain",
            ranking_loss_weight=0.0,
            listwise_loss_weight=0.0,
        ),
        RecipeSpec(
            encoder_family="patch_transformer",
            use_pretrain=False,
            score_head_method="manual",
            ranking_profile="plain",
            ranking_loss_weight=0.0,
            listwise_loss_weight=0.0,
        ),
        RecipeSpec(
            encoder_family="patch_transformer",
            use_pretrain=True,
            score_head_method="manual",
            ranking_profile="plain",
            ranking_loss_weight=0.0,
            listwise_loss_weight=0.0,
        ),
    ]


def _score_head_recipes(base_recipe: RecipeSpec, candidates: Iterable[str]) -> list[RecipeSpec]:
    return [
        RecipeSpec(
            encoder_family=base_recipe.encoder_family,
            use_pretrain=base_recipe.use_pretrain,
            score_head_method=method,
            ranking_profile="plain",
            ranking_loss_weight=0.0,
            listwise_loss_weight=0.0,
        )
        for method in candidates
    ]


def _ranking_recipes(base_recipe: RecipeSpec, args: argparse.Namespace) -> list[RecipeSpec]:
    return [
        RecipeSpec(
            encoder_family=base_recipe.encoder_family,
            use_pretrain=base_recipe.use_pretrain,
            score_head_method=base_recipe.score_head_method,
            ranking_profile="plain",
            ranking_loss_weight=0.0,
            listwise_loss_weight=0.0,
        ),
        RecipeSpec(
            encoder_family=base_recipe.encoder_family,
            use_pretrain=base_recipe.use_pretrain,
            score_head_method=base_recipe.score_head_method,
            ranking_profile="ranked",
            ranking_loss_weight=float(args.ranking_loss_weight_on),
            listwise_loss_weight=float(args.listwise_loss_weight_on),
        ),
    ]


def _selected_recipe_from_file(path: Path) -> RecipeSpec | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    recipe_payload = payload.get("recipe") or {}
    return RecipeSpec(**recipe_payload)


def _relative_posix(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _append_cache_flags(cmd: list[str], args: argparse.Namespace) -> None:
    if args.refresh_cache:
        cmd.append("--refresh-cache")
    if not args.use_cache:
        cmd.append("--no-cache")


def _append_runtime_flags(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend(["--num-workers", str(args.num_workers)])
    if args.pin_memory:
        cmd.append("--pin-memory")
    else:
        cmd.append("--no-pin-memory")
    if args.use_amp:
        cmd.append("--use-amp")
    else:
        cmd.append("--no-amp")
    if args.safe_runtime_profile:
        cmd.append("--safe-runtime-profile")
    else:
        cmd.append("--no-safe-runtime-profile")


def _build_pretrain_cmd(
    args: argparse.Namespace,
    *,
    window: WalkForwardWindow,
) -> list[str]:
    cmd = [
        sys.executable,
        "daily_research/deep_alpha/pretrain_deep_alpha_encoder.py",
        "--data-source",
        args.data_source,
        "--rolling-liquidity-pool",
        args.rolling_liquidity_pool,
        "--pool-rebalance-days",
        str(args.pool_rebalance_days),
        "--pool-adv-window",
        str(args.pool_adv_window),
        "--start-date",
        args.start_date,
        "--benchmark",
        args.benchmark,
        "--lookback-window",
        str(args.lookback_window),
        "--prediction-horizons",
        args.prediction_horizons,
        "--train-end-date",
        window.train_end,
        "--valid-start-date",
        window.valid_start,
        "--valid-days",
        str(args.valid_days),
        "--pretrain-valid-days",
        str(args.pretrain_valid_days),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--patch-len",
        str(args.patch_len),
        "--transformer-heads",
        str(args.transformer_heads),
        "--transformer-layers",
        str(args.transformer_layers),
        "--dropout",
        str(args.dropout),
        "--learning-rate",
        str(args.pretrain_learning_rate),
        "--weight-decay",
        str(args.pretrain_weight_decay),
        "--epochs",
        str(args.pretrain_epochs),
        "--min-epochs",
        str(args.pretrain_min_epochs),
        "--early-stop-patience",
        str(args.pretrain_early_stop_patience),
        "--lr-plateau-patience",
        str(args.pretrain_lr_plateau_patience),
        "--lr-plateau-factor",
        str(args.pretrain_lr_plateau_factor),
        "--min-improvement",
        str(args.pretrain_min_improvement),
        "--mask-ratio",
        str(args.pretrain_mask_ratio),
        "--random-seed",
        str(args.random_seed),
        "--market-state-count",
        str(args.market_state_count),
        "--min-adv20",
        str(args.min_adv20),
        "--min-price",
        str(args.min_price),
        "--max-price",
        str(args.max_price),
        "--experiment-tag",
        _pretrain_tag(args.root_tag, window),
    ]
    if args.end_date:
        cmd.extend(["--end-date", args.end_date])
    if args.csv_folder:
        cmd.extend(["--csv-folder", args.csv_folder])
    if args.auto_extend_undertrained:
        cmd.append("--auto-extend-undertrained")
    else:
        cmd.append("--no-auto-extend-undertrained")
    cmd.extend(["--epoch-extend-step", str(args.epoch_extend_step)])
    cmd.extend(["--max-total-epochs", str(args.max_total_epochs)])
    _append_runtime_flags(cmd, args)
    _append_cache_flags(cmd, args)
    return cmd


def _build_finetune_cmd(
    args: argparse.Namespace,
    *,
    window: WalkForwardWindow,
    recipe: RecipeSpec,
    pretrained_artifact: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "daily_research/deep_alpha/run_deep_alpha_research.py",
        "--data-source",
        args.data_source,
        "--rolling-liquidity-pool",
        args.rolling_liquidity_pool,
        "--pool-rebalance-days",
        str(args.pool_rebalance_days),
        "--pool-adv-window",
        str(args.pool_adv_window),
        "--start-date",
        args.start_date,
        "--benchmark",
        args.benchmark,
        "--lookback-window",
        str(args.lookback_window),
        "--prediction-horizons",
        args.prediction_horizons,
        "--train-end-date",
        window.train_end,
        "--valid-start-date",
        window.valid_start,
        "--valid-days",
        str(args.valid_days),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--encoder-family",
        recipe.encoder_family,
        "--patch-len",
        str(args.patch_len),
        "--transformer-heads",
        str(args.transformer_heads),
        "--transformer-layers",
        str(args.transformer_layers),
        "--dropout",
        str(args.dropout),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--epochs",
        str(args.epochs),
        "--min-epochs",
        str(args.min_epochs),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--lr-plateau-patience",
        str(args.lr_plateau_patience),
        "--lr-plateau-factor",
        str(args.lr_plateau_factor),
        "--min-improvement",
        str(args.min_improvement),
        "--return-loss-mode",
        args.return_loss_mode,
        "--return-target-transform",
        args.return_target_transform,
        "--score-risk-mode",
        args.score_risk_mode,
        "--score-risk-state-thresholds",
        args.score_risk_state_thresholds,
        "--score-head-method",
        recipe.score_head_method,
        "--train-eval-window-days",
        str(args.train_eval_window_days),
        "--ranking-loss-weight",
        str(recipe.ranking_loss_weight),
        "--listwise-loss-weight",
        str(recipe.listwise_loss_weight),
        "--random-seed",
        str(args.random_seed),
        "--market-state-count",
        str(args.market_state_count),
        "--holding-count",
        str(args.holding_count),
        "--rebalance-freq",
        args.rebalance_freq,
        "--max-weight",
        str(args.max_weight),
        "--min-adv20",
        str(args.min_adv20),
        "--min-price",
        str(args.min_price),
        "--max-price",
        str(args.max_price),
        "--experiment-tag",
        _run_tag(args.root_tag, recipe, window),
    ]
    if args.end_date:
        cmd.extend(["--end-date", args.end_date])
    if args.csv_folder:
        cmd.extend(["--csv-folder", args.csv_folder])
    if recipe.use_pretrain and pretrained_artifact is not None:
        cmd.extend(["--pretrained-encoder-path", _relative_posix(pretrained_artifact)])
    _append_runtime_flags(cmd, args)
    _append_cache_flags(cmd, args)
    return cmd


def _run_command(cmd: list[str]) -> None:
    print(f"[run] {subprocess.list2cmdline(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def _maybe_execute(cmd: list[str], run_dir: Path, *, skip_existing: bool, dry_run: bool) -> str:
    metrics_path = run_dir / "metrics.json"
    if skip_existing and metrics_path.exists():
        print(f"[skip] Existing run found: {run_dir}")
        return "reused"
    if dry_run:
        print(f"[plan] {subprocess.list2cmdline(cmd)}")
        return "planned"
    _run_command(cmd)
    return "executed"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_record(
    *,
    stage_name: str,
    recipe: RecipeSpec,
    window: WalkForwardWindow,
    run_dir: Path,
    pretrain_dir: Path | None,
) -> dict[str, object]:
    metrics = _load_json(run_dir / "metrics.json")
    holdout = metrics.get("holdout_backtest", {})
    training = metrics.get("training_diagnostics", {})
    record: dict[str, object] = {
        "stage": stage_name,
        "recipe_slug": recipe.slug,
        "recipe_display_name": recipe.display_name,
        "window_index": window.index,
        "window_label": window.label,
        "train_end": metrics.get("train_end", window.train_end),
        "valid_start": metrics.get("valid_start", window.valid_start),
        "valid_end": window.valid_end,
        "run_dir": _relative_posix(run_dir),
        "encoder_family": recipe.encoder_family,
        "use_pretrain": bool(recipe.use_pretrain),
        "score_head_method": recipe.score_head_method,
        "ranking_profile": recipe.ranking_profile,
        "ranking_loss_weight": recipe.ranking_loss_weight,
        "listwise_loss_weight": recipe.listwise_loss_weight,
        "finetune_status": training.get("status", ""),
        "finetune_epochs_completed": training.get("epochs_completed", 0),
        "finetune_best_epoch": training.get("best_epoch", 0),
        "excess_total_return": holdout.get("excess_total_return"),
        "excess_annual_return": holdout.get("excess_annual_return"),
        "excess_sharpe": holdout.get("excess_sharpe"),
        "excess_max_drawdown": holdout.get("excess_max_drawdown"),
        "avg_turnover": holdout.get("avg_turnover"),
        "avg_holding_count": holdout.get("avg_holding_count"),
        "regime_active_ratio": holdout.get("regime_active_ratio"),
    }
    if pretrain_dir is not None and (pretrain_dir / "metrics.json").exists():
        pretrain_metrics = _load_json(pretrain_dir / "metrics.json")
        pretrain_training = pretrain_metrics.get("training_diagnostics", {})
        record.update(
            {
                "pretrain_dir": _relative_posix(pretrain_dir),
                "pretrain_status": pretrain_training.get("status", ""),
                "pretrain_epochs_completed": pretrain_training.get("epochs_completed", 0),
                "pretrain_best_epoch": pretrain_training.get("best_epoch", 0),
                "pretrain_final_valid_loss": pretrain_metrics.get("final_valid_loss"),
            }
        )
    else:
        record.update(
            {
                "pretrain_dir": "",
                "pretrain_status": "",
                "pretrain_epochs_completed": 0,
                "pretrain_best_epoch": 0,
                "pretrain_final_valid_loss": None,
            }
        )
    return record


def _summarize_stage(records_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for recipe_slug, group in records_df.groupby("recipe_slug", sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "recipe_slug": recipe_slug,
                "recipe_display_name": first["recipe_display_name"],
                "encoder_family": first["encoder_family"],
                "use_pretrain": bool(first["use_pretrain"]),
                "score_head_method": first["score_head_method"],
                "ranking_profile": first["ranking_profile"],
                "ranking_loss_weight": float(first["ranking_loss_weight"]),
                "listwise_loss_weight": float(first["listwise_loss_weight"]),
                "window_count": int(len(group)),
                "finetune_undertrained_count": int(group["finetune_status"].eq("undertrained").sum()),
                "pretrain_undertrained_count": int(group["pretrain_status"].eq("undertrained").sum()),
                "mean_excess_sharpe": float(group["excess_sharpe"].mean()),
                "min_excess_sharpe": float(group["excess_sharpe"].min()),
                "mean_excess_total_return": float(group["excess_total_return"].mean()),
                "mean_excess_annual_return": float(group["excess_annual_return"].mean()),
                "mean_excess_max_drawdown": float(group["excess_max_drawdown"].mean()),
                "mean_avg_turnover": float(group["avg_turnover"].mean()),
                "mean_regime_active_ratio": float(group["regime_active_ratio"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _select_stage_winner(summary_df: pd.DataFrame) -> tuple[pd.Series, str]:
    eligible = summary_df[
        (summary_df["finetune_undertrained_count"] == 0)
        & (summary_df["pretrain_undertrained_count"] == 0)
    ].copy()
    selection_scope = "eligible_only"
    pool = eligible
    if pool.empty:
        selection_scope = "all_recipes_due_to_undertraining"
        pool = summary_df.copy()
    ordered = pool.sort_values(
        by=[
            "mean_excess_sharpe",
            "min_excess_sharpe",
            "mean_excess_total_return",
            "mean_excess_max_drawdown",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return ordered.iloc[0], selection_scope


def _write_stage_summary_markdown(
    *,
    stage_name: str,
    summary_df: pd.DataFrame,
    winner_row: pd.Series,
    selection_scope: str,
    path: Path,
) -> None:
    lines = [
        f"# Stage {stage_name}",
        "",
        f"selected_recipe: `{winner_row['recipe_slug']}`",
        f"selection_scope: `{selection_scope}`",
        "",
        "## Leaderboard",
        "",
        "| recipe | mean_excess_sharpe | min_excess_sharpe | mean_excess_total_return | finetune_undertrained | pretrain_undertrained |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.sort_values("mean_excess_sharpe", ascending=False).iterrows():
        lines.append(
            "| "
            f"`{row['recipe_slug']}` | "
            f"{row['mean_excess_sharpe']:.3f} | "
            f"{row['min_excess_sharpe']:.3f} | "
            f"{row['mean_excess_total_return']:.4f} | "
            f"{int(row['finetune_undertrained_count'])} | "
            f"{int(row['pretrain_undertrained_count'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plan(
    *,
    matrix_root: Path,
    args: argparse.Namespace,
    windows: list[WalkForwardWindow],
) -> None:
    matrix_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "phase": args.phase,
        "root_tag": args.root_tag,
        "dry_run": bool(args.dry_run),
        "window_count": int(args.window_count),
        "valid_days": int(args.valid_days),
        "windows": [asdict(window) | {"label": window.label} for window in windows],
        "base_config": {
            "data_source": args.data_source,
            "start_date": args.start_date,
            "end_date": args.end_date or pd.Timestamp(get_latest_completed_trading_date()).strftime("%Y%m%d"),
            "benchmark": args.benchmark,
            "rolling_liquidity_pool": args.rolling_liquidity_pool,
            "pool_rebalance_days": int(args.pool_rebalance_days),
            "pool_adv_window": int(args.pool_adv_window),
            "lookback_window": int(args.lookback_window),
            "prediction_horizons": args.prediction_horizons,
            "return_loss_mode": args.return_loss_mode,
            "return_target_transform": args.return_target_transform,
            "score_risk_mode": args.score_risk_mode,
            "train_eval_window_days": int(args.train_eval_window_days),
            "rebalance_freq": args.rebalance_freq,
            "holding_count": int(args.holding_count),
            "max_weight": float(args.max_weight),
        },
    }
    (matrix_root / "matrix_plan.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(window) | {"label": window.label} for window in windows]).to_csv(
        matrix_root / "windows.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_commands(path: Path, commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [subprocess.list2cmdline(cmd) for cmd in commands]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _stage_recipes(
    *,
    stage_name: str,
    args: argparse.Namespace,
    matrix_root: Path,
) -> list[RecipeSpec]:
    if stage_name == "backbone":
        return _backbone_recipes()
    if stage_name == "score_head":
        base_recipe = _selected_recipe_from_file(_selected_path(matrix_root, "backbone"))
        if base_recipe is None:
            raise RuntimeError("stage_score_head requires stage_backbone_selected.json. Run backbone stage first.")
        return _score_head_recipes(base_recipe, _parse_score_head_candidates(args.score_head_candidates))
    if stage_name == "ranking":
        base_recipe = _selected_recipe_from_file(_selected_path(matrix_root, "score_head"))
        if base_recipe is None:
            raise RuntimeError("stage_ranking requires stage_score_head_selected.json. Run score_head stage first.")
        return _ranking_recipes(base_recipe, args)
    raise ValueError(f"Unsupported stage: {stage_name}")


def _execute_stage(
    *,
    stage_name: str,
    args: argparse.Namespace,
    windows: list[WalkForwardWindow],
    matrix_root: Path,
) -> None:
    recipes = _stage_recipes(stage_name=stage_name, args=args, matrix_root=matrix_root)
    records: list[dict[str, object]] = []
    commands: list[list[str]] = []

    for recipe in recipes:
        for window in windows:
            pretrain_dir: Path | None = None
            if recipe.use_pretrain:
                pretrain_dir = _pretrain_dir(args.root_tag, window)
                pretrain_cmd = _build_pretrain_cmd(args, window=window)
                commands.append(pretrain_cmd)
                _maybe_execute(
                    pretrain_cmd,
                    pretrain_dir,
                    skip_existing=args.skip_existing,
                    dry_run=args.dry_run,
                )

            run_dir = _run_dir(args.root_tag, recipe, window)
            finetune_cmd = _build_finetune_cmd(
                args,
                window=window,
                recipe=recipe,
                pretrained_artifact=None if pretrain_dir is None else pretrain_dir / "pretrained_encoder.pt",
            )
            commands.append(finetune_cmd)
            _maybe_execute(
                finetune_cmd,
                run_dir,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )

            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                records.append(
                    _load_run_record(
                        stage_name=stage_name,
                        recipe=recipe,
                        window=window,
                        run_dir=run_dir,
                        pretrain_dir=pretrain_dir,
                    )
                )

    _write_commands(_commands_path(matrix_root, stage_name), commands)
    if not records:
        print(f"[info] No completed records available yet for stage={stage_name}.")
        return

    records_df = pd.DataFrame(records)
    records_df.to_csv(_runs_path(matrix_root, stage_name), index=False, encoding="utf-8-sig")
    summary_df = _summarize_stage(records_df)
    winner_row, selection_scope = _select_stage_winner(summary_df)
    summary_df = summary_df.copy()
    summary_df["selected"] = summary_df["recipe_slug"].eq(str(winner_row["recipe_slug"]))
    summary_df.to_csv(_summary_path(matrix_root, stage_name), index=False, encoding="utf-8-sig")
    _write_stage_summary_markdown(
        stage_name=stage_name,
        summary_df=summary_df,
        winner_row=winner_row,
        selection_scope=selection_scope,
        path=_summary_md_path(matrix_root, stage_name),
    )

    payload = {
        "stage": stage_name,
        "selection_scope": selection_scope,
        "recipe": {
            "encoder_family": str(winner_row["encoder_family"]),
            "use_pretrain": bool(winner_row["use_pretrain"]),
            "score_head_method": str(winner_row["score_head_method"]),
            "ranking_profile": str(winner_row["ranking_profile"]),
            "ranking_loss_weight": float(winner_row["ranking_loss_weight"]),
            "listwise_loss_weight": float(winner_row["listwise_loss_weight"]),
        },
        "recipe_slug": str(winner_row["recipe_slug"]),
        "summary": {
            "mean_excess_sharpe": float(winner_row["mean_excess_sharpe"]),
            "min_excess_sharpe": float(winner_row["min_excess_sharpe"]),
            "mean_excess_total_return": float(winner_row["mean_excess_total_return"]),
            "finetune_undertrained_count": int(winner_row["finetune_undertrained_count"]),
            "pretrain_undertrained_count": int(winner_row["pretrain_undertrained_count"]),
        },
    }
    _selected_path(matrix_root, stage_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Stage {stage_name} selected recipe: {payload['recipe_slug']} ({selection_scope})")


def main() -> None:
    args = parse_args()
    matrix_root = _matrix_root(args.root_tag)
    trading_dates = _load_benchmark_dates(args)
    windows = _build_recent_nonoverlap_windows(
        trading_dates,
        valid_days=args.valid_days,
        window_count=args.window_count,
    )
    _write_plan(matrix_root=matrix_root, args=args, windows=windows)

    stages = STAGE_SEQUENCE if args.phase == "all" else (args.phase,)
    for stage_name in stages:
        try:
            _execute_stage(
                stage_name=stage_name,
                args=args,
                windows=windows,
                matrix_root=matrix_root,
            )
        except RuntimeError as exc:
            if args.dry_run and args.phase == "all":
                print(f"[info] Dry-run stopped before stage={stage_name}: {exc}")
                break
            raise


if __name__ == "__main__":
    main()
