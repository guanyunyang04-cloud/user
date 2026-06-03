from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.execution.freeze_guard import assert_task_allowed
from daily_research.path_policy import v2_research_reset_baseline as v2


PYTHON = v2.PYTHON
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
DATA_LAKE_ROOT = PROJECT_ROOT / "daily_research/output/research_data_lake"
RUN_TAG = "v2_score_backtest_bridge_20260602_01"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
DEFAULT_SCORE_COLUMN = "pred_decision_score"
DEFAULT_HOLDING_COUNT = 20
DEFAULT_MAX_WEIGHT = 0.08
DEFAULT_REBALANCE_FREQ = "5d"
DEFAULT_COST_BPS = 3.0
DEFAULT_SLIPPAGE_BPS = 7.0
DEFAULT_SELL_TAX_BPS = 10.0


@dataclass(frozen=True)
class BridgePaths:
    output_root: Path
    seed_score_panel_csv: Path
    ensemble_score_panel_csv: Path
    backtest_score_panel_csv: Path
    manifest_json: Path
    report_json: Path
    report_md: Path
    backtest_command_json: Path
    backtest_stdout: Path
    backtest_stderr: Path


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
    return str(value)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_source_manifest(study_dir: Path) -> tuple[dict[str, Any], Path, str]:
    manifest_path = study_dir / "forecast_dataset_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest:
        return manifest, manifest_path, "forecast_dataset_manifest_json"
    summary_path = study_dir / "study_summary.json"
    summary = _read_json(summary_path)
    summary_manifest = summary.get("dataset_manifest", {}) if isinstance(summary, dict) else {}
    if isinstance(summary_manifest, dict) and summary_manifest:
        return dict(summary_manifest), summary_path, "study_summary_dataset_manifest"
    return {}, manifest_path, "missing"


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", str(ACTIVE_MANIFEST)], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _default_seed_study_tags() -> tuple[str, ...]:
    return tuple(f"mh_v2_reset_tradeable_mainboard_seed{int(seed)}_20260601_01" for seed in v2.SEEDS)


def _parse_csv_values(raw: str | tuple[str, ...] | list[str], *, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw, (tuple, list)):
        values = list(raw)
    else:
        values = str(raw or "").split(",")
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out or default)


def _paths(output_root: str | Path, *, role: str) -> BridgePaths:
    root = Path(output_root)
    suffix = str(role or "test").strip().lower()
    return BridgePaths(
        output_root=root,
        seed_score_panel_csv=root / f"v2_seed_score_panel_{suffix}.csv",
        ensemble_score_panel_csv=root / f"v2_ensemble_score_panel_{suffix}.csv",
        backtest_score_panel_csv=root / f"v2_ensemble_score_panel_{suffix}_for_backtest.csv",
        manifest_json=root / "v2_score_backtest_bridge_manifest.json",
        report_json=root / "v2_score_backtest_bridge_report.json",
        report_md=root / "v2_score_backtest_bridge_report.md",
        backtest_command_json=root / "v2_shared_backtest_command.json",
        backtest_stdout=root / "v2_shared_backtest_stdout.log",
        backtest_stderr=root / "v2_shared_backtest_stderr.log",
    )


def validate_source_study(
    study_dir: str | Path,
    *,
    dataset_id: str,
    pool_view_id: str,
    feature_profile: str,
) -> dict[str, Any]:
    resolved = Path(study_dir)
    manifest, manifest_path, manifest_source = _load_source_manifest(resolved)
    blockers: list[str] = []
    if not manifest:
        blockers.append("missing_dataset_manifest")
    if str(manifest.get("source_market_dataset_id", "")) != str(dataset_id):
        blockers.append("source_market_dataset_id_mismatch")
    if str(manifest.get("source_pool_view_id", "")) != str(pool_view_id):
        blockers.append("source_pool_view_id_mismatch")
    if str(manifest.get("feature_profile", "")) != str(feature_profile):
        blockers.append("feature_profile_mismatch")
    label_semantics = dict(manifest.get("label_semantics", {}) or {})
    if str(label_semantics.get("label_semantics", "")) != "next_open_entry_to_future_open":
        blockers.append("label_semantics_mismatch")
    return {
        "status": "ok" if not blockers else "blocked",
        "study_dir": str(resolved),
        "manifest_path": str(manifest_path),
        "manifest_source": str(manifest_source),
        "blockers": blockers,
        "source_market_dataset_id": str(manifest.get("source_market_dataset_id", "")),
        "source_pool_view_id": str(manifest.get("source_pool_view_id", "")),
        "source_pool_view_kind": str(manifest.get("source_pool_view_kind", "")),
        "source_pool_view_name": str(manifest.get("source_pool_view_name", "")),
        "feature_profile": str(manifest.get("feature_profile", "")),
        "feature_store_shape": list(manifest.get("feature_store_shape", []) or []),
        "feature_count": int(len(manifest.get("feature_columns", []) or [])),
        "label_semantics": label_semantics,
    }


def _load_prediction_scores(
    study_dir: str | Path,
    *,
    study_tag: str,
    role: str,
    score_column: str,
) -> pd.DataFrame:
    path = Path(study_dir) / f"forecast_predictions_{role}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction CSV for role={role}: {path}")
    required = {"date", "stock", score_column}
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV missing required columns {missing}: {path}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stock"] = out["stock"].astype(str).str.strip().str.upper()
    out["score"] = pd.to_numeric(out[score_column], errors="coerce")
    out = out.loc[out["date"].notna() & out["stock"].ne("") & out["score"].notna()].copy()
    if out.empty:
        raise ValueError(f"Prediction CSV has no usable score rows: {path}")
    keep = ["date", "stock", "score"]
    optional = [
        "pred_decision_score",
        "future_decision_score",
        "pred_best_horizon",
        "future_best_horizon",
        "history_bucket",
        "history_valid_ratio",
    ]
    optional.extend([column for column in out.columns if str(column).startswith("future_hit_label_")])
    keep.extend([column for column in optional if column in out.columns and column not in keep])
    out = out.loc[:, keep].copy()
    out.insert(2, "seed_study_tag", str(study_tag))
    out.insert(3, "role", str(role))
    return out.sort_values(["date", "stock", "seed_study_tag"]).reset_index(drop=True)


def _ensemble_scores(seed_scores: pd.DataFrame, *, required_seed_count: int) -> pd.DataFrame:
    grouped = seed_scores.groupby(["date", "stock"], as_index=False)
    ensemble = grouped.agg(
        score=("score", "mean"),
        score_std=("score", "std"),
        seed_count=("seed_study_tag", "nunique"),
    )
    if "future_decision_score" in seed_scores.columns:
        future = grouped["future_decision_score"].mean().rename(columns={"future_decision_score": "future_decision_score"})
        ensemble = ensemble.merge(future, on=["date", "stock"], how="left")
    if "pred_best_horizon" in seed_scores.columns:
        horizon = grouped["pred_best_horizon"].agg(lambda value: float(pd.to_numeric(value, errors="coerce").median()))
        horizon = horizon.rename(columns={"pred_best_horizon": "pred_best_horizon_median"})
        ensemble = ensemble.merge(horizon, on=["date", "stock"], how="left")
    hit_columns = [column for column in seed_scores.columns if str(column).startswith("future_hit_label_")]
    for column in hit_columns:
        hit = grouped[column].mean().rename(columns={column: column})
        ensemble = ensemble.merge(hit, on=["date", "stock"], how="left")
    if required_seed_count > 0:
        ensemble = ensemble.loc[ensemble["seed_count"].astype(int) >= int(required_seed_count)].copy()
    return ensemble.sort_values(["date", "stock"]).reset_index(drop=True)


def _daily_rank_ic(frame: pd.DataFrame, *, score_column: str, target_column: str) -> dict[str, Any]:
    if frame.empty or score_column not in frame.columns or target_column not in frame.columns:
        return {"available": False}
    values: list[float] = []
    for _, group in frame.groupby("date"):
        work = group[[score_column, target_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(work) < 3:
            continue
        corr = work[score_column].rank().corr(work[target_column].rank())
        if pd.notna(corr):
            values.append(float(corr))
    return {
        "available": bool(values),
        "mean": float(np.mean(values)) if values else 0.0,
        "min": float(np.min(values)) if values else 0.0,
        "date_count": int(len(values)),
    }


def _top_bottom_spread(frame: pd.DataFrame, *, score_column: str, target_column: str, frac: float = 0.2) -> dict[str, Any]:
    if frame.empty or score_column not in frame.columns or target_column not in frame.columns:
        return {"available": False}
    values: list[float] = []
    for _, group in frame.groupby("date"):
        work = group[[score_column, target_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(work) < 5:
            continue
        k = max(1, int(round(len(work) * float(frac))))
        top = work.nlargest(k, score_column)[target_column].mean()
        bottom = work.nsmallest(k, score_column)[target_column].mean()
        values.append(float(top - bottom))
    return {
        "available": bool(values),
        "mean": float(np.mean(values)) if values else 0.0,
        "min": float(np.min(values)) if values else 0.0,
        "date_count": int(len(values)),
    }


def _top20_hit_lift(frame: pd.DataFrame, *, score_column: str) -> dict[str, Any]:
    hit_columns = [column for column in frame.columns if str(column).startswith("future_hit_label_")]
    if frame.empty or not hit_columns:
        return {"available": False}
    work = frame.copy()
    hit_values = work[hit_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    work["_future_hit_any"] = hit_values.max(axis=1)
    values: list[float] = []
    for _, group in work.groupby("date"):
        daily = group[[score_column, "_future_hit_any"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(daily) < 20:
            continue
        k = min(20, len(daily))
        values.append(float(daily.nlargest(k, score_column)["_future_hit_any"].mean() - daily["_future_hit_any"].mean()))
    return {
        "available": bool(values),
        "mean": float(np.mean(values)) if values else 0.0,
        "min": float(np.min(values)) if values else 0.0,
        "date_count": int(len(values)),
    }


def _seed_score_correlation(seed_scores: pd.DataFrame) -> dict[str, Any]:
    pivot = seed_scores.pivot_table(index=["date", "stock"], columns="seed_study_tag", values="score", aggfunc="mean")
    if pivot.shape[1] < 2:
        return {"available": False}
    rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations([str(column) for column in pivot.columns], 2):
        pair = pivot[[left, right]].dropna()
        corr = pair[left].rank().corr(pair[right].rank()) if len(pair) >= 3 else np.nan
        rows.append({"left": left, "right": right, "spearman": float(corr) if pd.notna(corr) else None, "rows": int(len(pair))})
    finite = [float(row["spearman"]) for row in rows if row.get("spearman") is not None]
    return {
        "available": bool(finite),
        "pair_count": int(len(rows)),
        "spearman_mean": float(np.mean(finite)) if finite else 0.0,
        "spearman_min": float(np.min(finite)) if finite else 0.0,
        "pairs": rows,
    }


def _seed_top_overlap(seed_scores: pd.DataFrame, *, top_k: int = 20) -> dict[str, Any]:
    tags = [str(item) for item in sorted(seed_scores["seed_study_tag"].dropna().unique())]
    if len(tags) < 2:
        return {"available": False}
    daily_values: list[float] = []
    for _, daily in seed_scores.groupby("date"):
        sets: dict[str, set[str]] = {}
        for tag, group in daily.groupby("seed_study_tag"):
            ranked = group.sort_values("score", ascending=False).head(int(top_k))
            sets[str(tag)] = set(ranked["stock"].astype(str))
        overlaps: list[float] = []
        for left, right in itertools.combinations(tags, 2):
            if left not in sets or right not in sets:
                continue
            denominator = max(1, min(len(sets[left]), len(sets[right]), int(top_k)))
            overlaps.append(float(len(sets[left] & sets[right]) / denominator))
        if overlaps:
            daily_values.append(float(np.mean(overlaps)))
    return {
        "available": bool(daily_values),
        "top_k": int(top_k),
        "mean_pairwise_overlap": float(np.mean(daily_values)) if daily_values else 0.0,
        "min_pairwise_overlap": float(np.min(daily_values)) if daily_values else 0.0,
        "date_count": int(len(daily_values)),
    }


def _panel_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"row_count": 0, "date_count": 0, "symbol_count": 0, "start_date": "", "end_date": ""}
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    score = pd.to_numeric(frame.get("score", pd.Series(dtype=float)), errors="coerce")
    return {
        "row_count": int(len(frame)),
        "date_count": int(dates.nunique()),
        "symbol_count": int(frame["stock"].astype(str).nunique()) if "stock" in frame.columns else 0,
        "start_date": dates.min().strftime("%Y-%m-%d") if not dates.empty else "",
        "end_date": dates.max().strftime("%Y-%m-%d") if not dates.empty else "",
        "score_mean": float(score.mean()) if score.notna().any() else 0.0,
        "score_std": float(score.std()) if score.notna().any() else 0.0,
        "score_min": float(score.min()) if score.notna().any() else 0.0,
        "score_max": float(score.max()) if score.notna().any() else 0.0,
    }


def _backtest_command(
    *,
    score_panel_csv: Path,
    output_root: Path,
    experiment_tag: str,
    dataset_id: str,
    data_lake_root: str | Path,
    start_date: str,
    end_date: str,
    benchmark: str,
    holding_count: int,
    max_weight: float,
    rebalance_freq: str,
    rebalance_offset_mode: str,
    rebalance_anchor_date: str,
    score_threshold: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    use_market_regime_filter: bool,
    no_cache: bool,
) -> list[str]:
    command = [
        PYTHON,
        "-m",
        "daily_research.baseline.backtest_external_score_panel",
        "--score-panel-csv",
        str(score_panel_csv),
        "--score-panel-format",
        "long",
        "--date-column",
        "date",
        "--stock-column",
        "stock",
        "--score-column",
        "score",
        "--data-source",
        "lake",
        "--lake-dataset-id",
        str(dataset_id),
        "--data-lake-root",
        str(data_lake_root),
        "--universe-scope",
        "all_a",
        "--benchmark",
        str(benchmark),
        "--start-date",
        pd.Timestamp(start_date).strftime("%Y%m%d"),
        "--end-date",
        pd.Timestamp(end_date).strftime("%Y%m%d"),
        "--holding-count",
        str(int(holding_count)),
        "--max-weight",
        str(float(max_weight)),
        "--rebalance-freq",
        str(rebalance_freq),
        "--rebalance-offset-mode",
        str(rebalance_offset_mode),
        "--rebalance-anchor-date",
        str(rebalance_anchor_date),
        "--score-threshold",
        str(float(score_threshold)),
        "--transaction-cost-bps",
        str(float(transaction_cost_bps)),
        "--slippage-bps",
        str(float(slippage_bps)),
        "--sell-tax-bps",
        str(float(sell_tax_bps)),
        "--output-dir",
        str(output_root / "shared_backtest"),
        "--experiment-tag",
        str(experiment_tag),
        "--candidate-label",
        "daily_research_v2_score_bridge_research_candidate",
    ]
    if not use_market_regime_filter:
        command.append("--no-market-regime-filter")
    if no_cache:
        command.append("--no-cache")
    return command


def _command_option(command: list[str], option: str) -> str:
    if option not in command:
        return ""
    idx = command.index(option)
    if idx + 1 >= len(command):
        return ""
    return str(command[idx + 1])


def _collect_backtest_artifacts(command: list[str]) -> dict[str, Any]:
    output_dir = _command_option(command, "--output-dir")
    experiment_tag = _command_option(command, "--experiment-tag")
    if not output_dir or not experiment_tag:
        return {"available": False}
    run_dir = Path(output_dir) / experiment_tag
    metrics_path = run_dir / "metrics.json"
    monthly_path = run_dir / "monthly_backtest_diagnostics.json"
    metrics = _read_json(metrics_path)
    monthly = _read_json(monthly_path)
    core_keys = [
        "annual_return",
        "excess_annual_return",
        "excess_sharpe",
        "max_drawdown",
        "avg_turnover",
        "total_trading_cost_return",
        "annual_return_cost_drag",
        "excess_annual_return_cost_drag",
        "weeklyized_return",
        "excess_weeklyized_return",
        "rebalance_freq",
        "rebalance_offset_mode",
        "rebalance_sleeve_count",
        "holding_count_target",
    ]
    return {
        "available": bool(metrics_path.exists()),
        "output_dir": str(run_dir),
        "metrics_json": str(metrics_path),
        "monthly_backtest_diagnostics_json": str(monthly_path),
        "core_metrics": {key: metrics.get(key) for key in core_keys if key in metrics},
        "monthly_backtest_diagnostics": monthly or metrics.get("monthly_backtest_diagnostics", {}),
    }


def _run_backtest(command: list[str], *, paths: BridgePaths) -> dict[str, Any]:
    assert_task_allowed("candidate-backtest")
    _write_json(paths.backtest_command_json, {"command": command, "created_at": _now()})
    paths.backtest_stdout.parent.mkdir(parents=True, exist_ok=True)
    with paths.backtest_stdout.open("w", encoding="utf-8", errors="replace") as stdout, paths.backtest_stderr.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True, check=False)
    result = {
        "status": "completed" if int(proc.returncode) == 0 else "failed",
        "returncode": int(proc.returncode),
        "command_json": str(paths.backtest_command_json),
        "stdout": str(paths.backtest_stdout),
        "stderr": str(paths.backtest_stderr),
    }
    result["artifacts"] = _collect_backtest_artifacts(command)
    return result


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    bridge = dict(report.get("bridge", {}) or {})
    diagnostics = dict(report.get("diagnostics", {}) or {})
    backtest = dict(report.get("shared_backtest", {}) or {})
    lines = [
        "# V2 Score Backtest Bridge",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Source anchor: `{bridge.get('source_anchor_run_tag', '')}`",
        f"- Dataset: `{bridge.get('dataset_id', '')}`",
        f"- Pool: `{bridge.get('pool_view_id', '')}`",
        f"- Feature profile: `{bridge.get('feature_profile', '')}`",
        f"- Score column: `{bridge.get('score_column', '')}`",
        f"- Role: `{bridge.get('role', '')}`",
        f"- Backtest status: `{backtest.get('status', 'not_requested')}`",
        "",
        "## Outputs",
        "",
        f"- Seed score panel: `{bridge.get('seed_score_panel_csv', '')}`",
        f"- Ensemble score panel: `{bridge.get('ensemble_score_panel_csv', '')}`",
        f"- Backtest score panel: `{bridge.get('backtest_score_panel_csv', '')}`",
        "",
        "## Diagnostics",
        "",
        f"- Ensemble rows: `{diagnostics.get('ensemble_panel', {}).get('row_count', 0)}`",
        f"- Date count: `{diagnostics.get('ensemble_panel', {}).get('date_count', 0)}`",
        f"- Symbol count: `{diagnostics.get('ensemble_panel', {}).get('symbol_count', 0)}`",
        f"- Future score rank IC mean: `{diagnostics.get('ensemble_future_rank_ic', {}).get('mean', '')}`",
        f"- Future score top-bottom spread mean: `{diagnostics.get('ensemble_future_top_bottom_spread', {}).get('mean', '')}`",
        f"- Top20 hit lift mean: `{diagnostics.get('ensemble_top20_hit_lift', {}).get('mean', '')}`",
        f"- Seed score Spearman mean: `{diagnostics.get('seed_score_correlation', {}).get('spearman_mean', '')}`",
        f"- Seed top20 overlap mean: `{diagnostics.get('seed_top20_overlap', {}).get('mean_pairwise_overlap', '')}`",
        "",
        "## Shared Backtest",
        "",
        f"- Output dir: `{backtest.get('artifacts', {}).get('output_dir', '')}`",
        f"- Annual return: `{backtest.get('artifacts', {}).get('core_metrics', {}).get('annual_return', '')}`",
        f"- Excess annual return: `{backtest.get('artifacts', {}).get('core_metrics', {}).get('excess_annual_return', '')}`",
        f"- Excess Sharpe: `{backtest.get('artifacts', {}).get('core_metrics', {}).get('excess_sharpe', '')}`",
        f"- Max drawdown: `{backtest.get('artifacts', {}).get('core_metrics', {}).get('max_drawdown', '')}`",
        f"- Monthly positive ratio: `{backtest.get('artifacts', {}).get('monthly_backtest_diagnostics', {}).get('positive_month_ratio', '')}`",
        f"- Negative month count: `{backtest.get('artifacts', {}).get('monthly_backtest_diagnostics', {}).get('negative_month_count', '')}`",
        "",
        "## Boundary",
        "",
        "- research_only: `true`",
        "- promotion_allowed: `false`",
        "- active_execution_strategy_expected_diff: `none`",
        "- score panel is not a production target-weight panel.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_v2_score_bridge(
    *,
    output_root: str | Path | None = None,
    run_tag: str = RUN_TAG,
    source_anchor_run_tag: str = v2.RUN_TAG,
    seed_study_tags: tuple[str, ...] | list[str] | str = (),
    studies_root: str | Path = STUDIES_ROOT,
    dataset_id: str = v2.DATASET_ID,
    pool_view_id: str = v2.V2_STRICT_POOL_VIEW_ID,
    feature_profile: str = v2.FEATURE_PROFILE,
    role: str = "test",
    score_column: str = DEFAULT_SCORE_COLUMN,
    require_all_seeds: bool = True,
    run_backtest: bool = False,
    data_lake_root: str | Path = DATA_LAKE_ROOT,
    benchmark: str = "000300.SH",
    holding_count: int = DEFAULT_HOLDING_COUNT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ,
    rebalance_offset_mode: str = "all",
    score_threshold: float = -999.0,
    transaction_cost_bps: float = DEFAULT_COST_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    sell_tax_bps: float = DEFAULT_SELL_TAX_BPS,
    use_market_regime_filter: bool = False,
    no_cache: bool = False,
    enforce_active_artifact_clean: bool = True,
) -> dict[str, Any]:
    if enforce_active_artifact_clean and _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    tags = _parse_csv_values(seed_study_tags, default=_default_seed_study_tags())
    root = Path(output_root) if output_root is not None else STUDIES_ROOT / str(run_tag)
    paths = _paths(root, role=role)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    source_root = Path(studies_root)
    validations: list[dict[str, Any]] = []
    seed_frames: list[pd.DataFrame] = []
    for tag in tags:
        study_dir = source_root / tag
        validation = validate_source_study(
            study_dir,
            dataset_id=dataset_id,
            pool_view_id=pool_view_id,
            feature_profile=feature_profile,
        )
        validations.append({"study_tag": str(tag), **validation})
        if validation["status"] != "ok":
            continue
        seed_frames.append(_load_prediction_scores(study_dir, study_tag=str(tag), role=role, score_column=score_column))
    blockers = [f"{item['study_tag']}:{blocker}" for item in validations for blocker in item.get("blockers", [])]
    if blockers:
        report = {
            "schema_version": 1,
            "status": "blocked",
            "run_tag": str(run_tag),
            "blockers": blockers,
            "source_study_validations": validations,
            "updated_at": _now(),
            "boundary": {
                "research_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            },
        }
        _write_json(paths.report_json, report)
        _write_json(paths.manifest_json, report)
        _write_markdown(paths.report_md, report)
        return report
    if not seed_frames:
        raise ValueError("No source prediction frames were loaded for v2 score bridge.")

    seed_scores = pd.concat(seed_frames, ignore_index=True).sort_values(["date", "stock", "seed_study_tag"])
    required_seed_count = len(tags) if require_all_seeds else 1
    ensemble = _ensemble_scores(seed_scores, required_seed_count=required_seed_count)
    if ensemble.empty:
        raise ValueError("Ensemble score panel is empty after seed-count filtering.")
    backtest_panel = ensemble.loc[:, ["date", "stock", "score"]].copy()
    seed_scores_export = seed_scores.copy()
    ensemble_export = ensemble.copy()
    for frame in (seed_scores_export, ensemble_export, backtest_panel):
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    seed_scores_export.to_csv(paths.seed_score_panel_csv, index=False, encoding="utf-8-sig")
    ensemble_export.to_csv(paths.ensemble_score_panel_csv, index=False, encoding="utf-8-sig")
    backtest_panel.to_csv(paths.backtest_score_panel_csv, index=False, encoding="utf-8-sig")

    panel_dates = pd.to_datetime(backtest_panel["date"], errors="coerce").dropna()
    start_date = panel_dates.min().strftime("%Y-%m-%d")
    end_date = panel_dates.max().strftime("%Y-%m-%d")
    command = _backtest_command(
        score_panel_csv=paths.backtest_score_panel_csv,
        output_root=paths.output_root,
        experiment_tag=f"{run_tag}_shared_engine",
        dataset_id=dataset_id,
        data_lake_root=data_lake_root,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        holding_count=holding_count,
        max_weight=max_weight,
        rebalance_freq=rebalance_freq,
        rebalance_offset_mode=rebalance_offset_mode,
        rebalance_anchor_date=start_date,
        score_threshold=score_threshold,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
        use_market_regime_filter=use_market_regime_filter,
        no_cache=no_cache,
    )
    shared_backtest = {"status": "not_requested", "command": command}
    if run_backtest:
        shared_backtest = _run_backtest(command, paths=paths)
        shared_backtest["command"] = command

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "run_tag": str(run_tag),
        "created_at": _now(),
        "research_program": v2.RESEARCH_PROGRAM,
        "study_family": "v2_score_backtest_bridge",
        "source_anchor_run_tag": str(source_anchor_run_tag),
        "source_seed_study_tags": list(tags),
        "dataset_id": str(dataset_id),
        "pool_view_id": str(pool_view_id),
        "feature_profile": str(feature_profile),
        "score_column": str(score_column),
        "role": str(role),
        "ensemble_policy": {
            "method": "mean_by_date_symbol",
            "require_all_seeds": bool(require_all_seeds),
            "required_seed_count": int(required_seed_count),
        },
        "score_panel_csv": str(paths.backtest_score_panel_csv),
        "seed_score_panel_csv": str(paths.seed_score_panel_csv),
        "ensemble_score_panel_csv": str(paths.ensemble_score_panel_csv),
        "execution_candidate_contract": {
            "research_only": True,
            "shadow_only": True,
            "candidate_backtest_task_allowed_under_freeze": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
            "output_semantics": "pred_decision_score ensemble score panel",
            "target_weight_semantics": "shared_backtest_rank_score_top_k_candidate_mapping_not_production_panel",
            "score_to_weight_mapping": {
                "holding_count": int(holding_count),
                "max_weight": float(max_weight),
                "rebalance_freq": str(rebalance_freq),
                "rebalance_offset_mode": str(rebalance_offset_mode),
                "score_threshold": float(score_threshold),
                "market_regime_filter": bool(use_market_regime_filter),
                "costs": {
                    "transaction_cost_bps": float(transaction_cost_bps),
                    "slippage_bps": float(slippage_bps),
                    "sell_tax_bps": float(sell_tax_bps),
                },
            },
        },
    }
    diagnostics = {
        "seed_panel": _panel_summary(seed_scores_export.rename(columns={"score": "score"})),
        "ensemble_panel": _panel_summary(ensemble_export.rename(columns={"score": "score"})),
        "ensemble_future_rank_ic": _daily_rank_ic(ensemble, score_column="score", target_column="future_decision_score"),
        "ensemble_future_top_bottom_spread": _top_bottom_spread(ensemble, score_column="score", target_column="future_decision_score"),
        "ensemble_top20_hit_lift": _top20_hit_lift(ensemble, score_column="score"),
        "seed_score_correlation": _seed_score_correlation(seed_scores),
        "seed_top20_overlap": _seed_top_overlap(seed_scores, top_k=min(20, int(holding_count))),
    }
    status = "completed" if not run_backtest or shared_backtest.get("status") == "completed" else "backtest_failed"
    report = {
        "schema_version": 1,
        "status": status,
        "run_tag": str(run_tag),
        "bridge": {
            **manifest,
            "manifest_json": str(paths.manifest_json),
            "report_json": str(paths.report_json),
            "report_md": str(paths.report_md),
            "backtest_score_panel_csv": str(paths.backtest_score_panel_csv),
        },
        "source_study_validations": validations,
        "diagnostics": diagnostics,
        "shared_backtest": shared_backtest,
        "boundary": manifest["execution_candidate_contract"],
        "updated_at": _now(),
    }
    _write_json(paths.manifest_json, manifest)
    _write_json(paths.report_json, report)
    _write_markdown(paths.report_md, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bridge daily_research v2 forecast scores into research-only candidate backtest inputs.")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--source-anchor-run-tag", default=v2.RUN_TAG)
    parser.add_argument("--seed-study-tags", default=",".join(_default_seed_study_tags()))
    parser.add_argument("--studies-root", default=str(STUDIES_ROOT))
    parser.add_argument("--dataset-id", default=v2.DATASET_ID)
    parser.add_argument("--pool-view-id", default=v2.V2_STRICT_POOL_VIEW_ID)
    parser.add_argument("--feature-profile", default=v2.FEATURE_PROFILE)
    parser.add_argument("--role", choices=["validation", "test"], default="test")
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--allow-partial-seeds", action="store_true")
    parser.add_argument("--run-backtest", action="store_true")
    parser.add_argument("--data-lake-root", default=str(DATA_LAKE_ROOT))
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-count", type=int, default=DEFAULT_HOLDING_COUNT)
    parser.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--rebalance-freq", default=DEFAULT_REBALANCE_FREQ)
    parser.add_argument("--rebalance-offset-mode", choices=["single", "all"], default="all")
    parser.add_argument("--score-threshold", type=float, default=-999.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--sell-tax-bps", type=float, default=DEFAULT_SELL_TAX_BPS)
    parser.add_argument("--use-market-regime-filter", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    output_root = args.output_root or str(STUDIES_ROOT / str(args.run_tag))
    report = build_v2_score_bridge(
        output_root=output_root,
        run_tag=args.run_tag,
        source_anchor_run_tag=args.source_anchor_run_tag,
        seed_study_tags=args.seed_study_tags,
        studies_root=args.studies_root,
        dataset_id=args.dataset_id,
        pool_view_id=args.pool_view_id,
        feature_profile=args.feature_profile,
        role=args.role,
        score_column=args.score_column,
        require_all_seeds=not bool(args.allow_partial_seeds),
        run_backtest=bool(args.run_backtest),
        data_lake_root=args.data_lake_root,
        benchmark=args.benchmark,
        holding_count=int(args.holding_count),
        max_weight=float(args.max_weight),
        rebalance_freq=args.rebalance_freq,
        rebalance_offset_mode=args.rebalance_offset_mode,
        score_threshold=float(args.score_threshold),
        transaction_cost_bps=float(args.transaction_cost_bps),
        slippage_bps=float(args.slippage_bps),
        sell_tax_bps=float(args.sell_tax_bps),
        use_market_regime_filter=bool(args.use_market_regime_filter),
        no_cache=bool(args.no_cache),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status={report.get('status')} run_tag={args.run_tag} output_root={output_root}")
    return 0 if str(report.get("status")) != "backtest_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
