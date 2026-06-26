from __future__ import annotations

import argparse
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
from daily_research.path_policy import v2_candidate_review_matrix as candidate_matrix
from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy import v2_score_backtest_bridge as bridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
DATA_LAKE_ROOT = PROJECT_ROOT / "quant_data_platform/data/lake"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
RUN_TAG = "v2_state_sizing_matrix_20260602_01"
SOURCE_MATRIX_RUN_TAG = "v2_selective_throttle_matrix_narrow_20260602_01"
SOURCE_VARIANT_ID = "h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80"


@dataclass(frozen=True)
class StateSizingVariant:
    variant_id: str
    scale_mode: str
    stress_scale: float
    drawdown_threshold: float
    monthly_loss_threshold: float
    min_scale: float
    holding_count: int
    max_weight: float
    rebalance_freq: str
    rebalance_offset_mode: str
    transaction_cost_bps: float
    slippage_bps: float
    sell_tax_bps: float


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


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", str(ACTIVE_MANIFEST)], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _parse_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _parse_strings(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _default_source_backtest_dir(matrix_run_tag: str, variant_id: str) -> Path:
    return STUDIES_ROOT / str(matrix_run_tag) / "shared_backtest" / f"{matrix_run_tag}_{variant_id}"


def _wide_from_long(path: str | Path, value_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "stock", value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns {missing}: {path}")
    work = frame.loc[:, ["date", "stock", value_column]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["stock"] = work["stock"].astype(str).str.strip().str.upper()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.loc[work["date"].notna() & work["stock"].ne("") & work[value_column].notna()].copy()
    if work.empty:
        raise ValueError(f"No usable rows in {path}")
    return work.pivot_table(index="date", columns="stock", values=value_column, aggfunc="mean").sort_index()


def _long_from_wide(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "stock", value_column])
    try:
        stacked = frame.stack(future_stack=True)
    except TypeError:
        stacked = frame.stack(dropna=True)
    out = stacked.dropna().rename(value_column).reset_index()
    out.columns = ["date", "stock", value_column]
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["stock"] = out["stock"].astype(str)
    return out.sort_values(["date", "stock"]).reset_index(drop=True)


def _load_equity(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "excess_equity"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Equity curve missing required columns {missing}: {path}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["excess_equity"] = pd.to_numeric(out["excess_equity"], errors="coerce")
    out = out.loc[out["date"].notna() & out["excess_equity"].notna()].copy()
    if out.empty:
        raise ValueError(f"Equity curve has no usable rows: {path}")
    return out.sort_values("date").reset_index(drop=True)


def build_state_scale(
    equity: pd.DataFrame,
    *,
    scale_mode: str,
    stress_scale: float,
    drawdown_threshold: float,
    monthly_loss_threshold: float,
    min_scale: float,
) -> tuple[pd.Series, dict[str, Any]]:
    work = equity.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").set_index("date")
    excess = work["excess_equity"].astype(float).replace([np.inf, -np.inf], np.nan).ffill().dropna()
    if excess.empty:
        raise ValueError("excess_equity is empty after cleaning.")
    drawdown = excess / excess.cummax() - 1.0
    month_return = excess.groupby(excess.index.to_period("M")).transform(lambda s: float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) else 0.0)
    prev_month_return = month_return.groupby(month_return.index.to_period("M")).first()
    prev_month_return = prev_month_return.shift(1)
    prev_month_by_date = pd.Series(
        [float(prev_month_return.get(period, 0.0)) if pd.notna(prev_month_return.get(period, np.nan)) else 0.0 for period in excess.index.to_period("M")],
        index=excess.index,
    )
    mode = str(scale_mode or "drawdown").strip().lower()
    drawdown_stress = drawdown <= -abs(float(drawdown_threshold))
    month_stress = prev_month_by_date <= -abs(float(monthly_loss_threshold))
    if mode == "drawdown":
        stress = drawdown_stress
    elif mode == "prev_month_loss":
        stress = month_stress
    elif mode == "drawdown_or_prev_month_loss":
        stress = drawdown_stress | month_stress
    elif mode in {"none", "off"}:
        stress = pd.Series(False, index=excess.index)
    else:
        raise ValueError(f"Unsupported scale_mode: {scale_mode}")
    scale_value = max(float(stress_scale), float(min_scale))
    scale = pd.Series(1.0, index=excess.index)
    scale.loc[stress] = scale_value
    scale = scale.clip(lower=float(min_scale), upper=1.0)
    meta = {
        "scale_mode": mode,
        "stress_scale": float(stress_scale),
        "drawdown_threshold": float(drawdown_threshold),
        "monthly_loss_threshold": float(monthly_loss_threshold),
        "min_scale": float(min_scale),
        "stress_day_count": int(stress.sum()),
        "stress_day_ratio": float(stress.mean()) if len(stress) else 0.0,
        "drawdown_stress_day_count": int(drawdown_stress.sum()),
        "prev_month_loss_stress_day_count": int(month_stress.sum()),
        "min_observed_drawdown": float(drawdown.min()),
        "mean_scale": float(scale.mean()),
    }
    return scale, meta


def apply_state_sizing(target_weights: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    aligned = scale.reindex(target_weights.index).ffill().fillna(1.0).clip(lower=0.0, upper=1.0)
    return target_weights.mul(aligned, axis=0).fillna(0.0)


def _variants(
    *,
    scale_modes: tuple[str, ...],
    stress_scales: tuple[float, ...],
    drawdown_thresholds: tuple[float, ...],
    monthly_loss_thresholds: tuple[float, ...],
    holding_count: int,
    max_weight: float,
    rebalance_freq: str,
    rebalance_offset_mode: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> list[StateSizingVariant]:
    out: list[StateSizingVariant] = []
    for mode in scale_modes:
        for stress_scale in stress_scales:
            for dd in drawdown_thresholds:
                for ml in monthly_loss_thresholds:
                    variant_id = (
                        f"state_{mode}_s{stress_scale:g}_dd{dd:g}_ml{ml:g}_"
                        f"h{holding_count}_mw{int(round(max_weight * 1000)):03d}_rb{rebalance_freq}_{rebalance_offset_mode}"
                    ).replace(".", "p")
                    out.append(
                        StateSizingVariant(
                            variant_id=variant_id,
                            scale_mode=str(mode),
                            stress_scale=float(stress_scale),
                            drawdown_threshold=float(dd),
                            monthly_loss_threshold=float(ml),
                            min_scale=0.0,
                            holding_count=int(holding_count),
                            max_weight=float(max_weight),
                            rebalance_freq=str(rebalance_freq),
                            rebalance_offset_mode=str(rebalance_offset_mode),
                            transaction_cost_bps=float(transaction_cost_bps),
                            slippage_bps=float(slippage_bps),
                            sell_tax_bps=float(sell_tax_bps),
                        )
                    )
    return out


def _backtest_command(
    variant: StateSizingVariant,
    *,
    score_panel_csv: Path,
    target_weight_panel_csv: Path,
    output_root: Path,
    run_tag: str,
    dataset_id: str,
    data_lake_root: str | Path,
    start_date: str,
    end_date: str,
    benchmark: str,
    no_cache: bool,
) -> list[str]:
    command = [
        bridge.PYTHON,
        "-m",
        "daily_research.baseline.backtest_external_score_panel",
        "--score-panel-csv",
        str(score_panel_csv),
        "--score-panel-format",
        "long",
        "--target-weight-panel-csv",
        str(target_weight_panel_csv),
        "--target-weight-panel-format",
        "long",
        "--date-column",
        "date",
        "--stock-column",
        "stock",
        "--score-column",
        "score",
        "--target-weight-column",
        "target_weight",
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
        str(start_date).replace("-", ""),
        "--end-date",
        str(end_date).replace("-", ""),
        "--holding-count",
        str(int(variant.holding_count)),
        "--max-weight",
        str(float(variant.max_weight)),
        "--rebalance-freq",
        str(variant.rebalance_freq),
        "--rebalance-offset-mode",
        str(variant.rebalance_offset_mode),
        "--rebalance-anchor-date",
        str(start_date),
        "--transaction-cost-bps",
        str(float(variant.transaction_cost_bps)),
        "--slippage-bps",
        str(float(variant.slippage_bps)),
        "--sell-tax-bps",
        str(float(variant.sell_tax_bps)),
        "--output-dir",
        str(output_root / "shared_backtest"),
        "--experiment-tag",
        f"{run_tag}_{variant.variant_id}",
        "--candidate-label",
        "daily_research_v2_state_sizing_research_candidate",
        "--no-market-regime-filter",
    ]
    if no_cache:
        command.append("--no-cache")
    return command


def _run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    assert_task_allowed("candidate-backtest")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True, check=False)
    return {"returncode": int(proc.returncode), "stdout": str(stdout_path), "stderr": str(stderr_path)}


def _result(
    variant: StateSizingVariant,
    *,
    command: list[str],
    run_result: dict[str, Any] | None,
    scale_meta: dict[str, Any],
    target_weight_panel_csv: Path,
) -> dict[str, Any]:
    surrogate = candidate_matrix.Variant(
        variant_id=variant.variant_id,
        holding_count=variant.holding_count,
        max_weight=variant.max_weight,
        rebalance_freq=variant.rebalance_freq,
        rebalance_offset_mode=variant.rebalance_offset_mode,
        transaction_cost_bps=variant.transaction_cost_bps,
        slippage_bps=variant.slippage_bps,
        sell_tax_bps=variant.sell_tax_bps,
        use_market_regime_filter=False,
    )
    item = candidate_matrix._variant_result(surrogate, command=command, run_result=run_result)
    item["variant"] = variant.__dict__
    item["scale_meta"] = scale_meta
    item["target_weight_panel_csv"] = str(target_weight_panel_csv)
    return item


def _summary_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = candidate_matrix._summary_frame(results)
    for idx, item in enumerate(results):
        meta = dict(item.get("scale_meta", {}) or {})
        for key in ["stress_day_ratio", "mean_scale", "min_observed_drawdown"]:
            frame.loc[idx, key] = meta.get(key)
    return frame


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = dict(report.get("summary", {}) or {})
    best = dict(summary.get("best_by_excess_sharpe", {}) or {})
    lines = [
        "# V2 State Sizing Matrix",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Source backtest: `{report.get('source_backtest_dir', '')}`",
        f"- Variant count: `{summary.get('variant_count', 0)}`",
        f"- Completed backtests: `{summary.get('completed_backtest_count', 0)}`",
        f"- Promotion-review eligible variants: `{summary.get('promotion_review_eligible_count', 0)}`",
        "",
        "## Boundary",
        "",
        "- research_only: `true`",
        "- promotion_allowed: `false`",
        "- active_execution_strategy_expected_diff: `none`",
        "- State sizing uses only candidate path state available up to the signal date; it is not a production promotion.",
    ]
    if best:
        lines.extend(
            [
                "",
                "## Best Completed Variant",
                "",
                f"- Variant: `{best.get('variant_id', '')}`",
                f"- Excess Sharpe: `{best.get('excess_sharpe', '')}`",
                f"- Positive month ratio: `{best.get('positive_month_ratio', '')}`",
                f"- Negative months: `{best.get('negative_month_count', '')}`",
                f"- Max drawdown: `{best.get('max_drawdown', '')}`",
                f"- Issue flags: `{best.get('issue_flags', '')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_state_sizing_matrix(
    *,
    run_tag: str = RUN_TAG,
    source_matrix_run_tag: str = SOURCE_MATRIX_RUN_TAG,
    source_variant_id: str = SOURCE_VARIANT_ID,
    source_backtest_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    dataset_id: str = v2.DATASET_ID,
    pool_view_id: str = v2.V2_STRICT_POOL_VIEW_ID,
    data_lake_root: str | Path = DATA_LAKE_ROOT,
    benchmark: str = "000300.SH",
    scale_modes: tuple[str, ...] = ("drawdown", "prev_month_loss", "drawdown_or_prev_month_loss"),
    stress_scales: tuple[float, ...] = (0.50, 0.70),
    drawdown_thresholds: tuple[float, ...] = (0.03, 0.05),
    monthly_loss_thresholds: tuple[float, ...] = (0.01,),
    holding_count: int = 20,
    max_weight: float = 0.08,
    rebalance_freq: str = "5d",
    rebalance_offset_mode: str = "all",
    transaction_cost_bps: float = 3.0,
    slippage_bps: float = 7.0,
    sell_tax_bps: float = 10.0,
    run_backtests: bool = False,
    no_cache: bool = False,
    enforce_active_artifact_clean: bool = True,
) -> dict[str, Any]:
    if enforce_active_artifact_clean and _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = Path(output_root) if output_root is not None else STUDIES_ROOT / str(run_tag)
    root.mkdir(parents=True, exist_ok=True)
    source_dir = Path(source_backtest_dir) if source_backtest_dir is not None else _default_source_backtest_dir(source_matrix_run_tag, source_variant_id)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source backtest dir not found: {source_dir}")

    source_target = _wide_from_long(source_dir / "aligned_daily_target_weight_panel.csv", "target_weight")
    source_score = _wide_from_long(source_dir / "aligned_daily_score_panel.csv", "score")
    equity = _load_equity(source_dir / "equity_curve.csv")
    start_date = pd.Timestamp(source_target.index.min()).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(source_target.index.max()).strftime("%Y-%m-%d")
    source_score_path = root / "source_score_context.csv"
    _long_from_wide(source_score, "score").to_csv(source_score_path, index=False, encoding="utf-8-sig")

    variants = _variants(
        scale_modes=scale_modes,
        stress_scales=stress_scales,
        drawdown_thresholds=drawdown_thresholds,
        monthly_loss_thresholds=monthly_loss_thresholds,
        holding_count=holding_count,
        max_weight=max_weight,
        rebalance_freq=rebalance_freq,
        rebalance_offset_mode=rebalance_offset_mode,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
    )
    results: list[dict[str, Any]] = []
    for variant in variants:
        scale, scale_meta = build_state_scale(
            equity,
            scale_mode=variant.scale_mode,
            stress_scale=variant.stress_scale,
            drawdown_threshold=variant.drawdown_threshold,
            monthly_loss_threshold=variant.monthly_loss_threshold,
            min_scale=variant.min_scale,
        )
        scaled_target = apply_state_sizing(source_target, scale)
        target_path = root / "target_weight_panels" / f"{variant.variant_id}.csv"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _long_from_wide(scaled_target, "target_weight").to_csv(target_path, index=False, encoding="utf-8-sig")
        scale_path = root / "state_scales" / f"{variant.variant_id}.csv"
        scale_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": scale.index.strftime("%Y-%m-%d"), "state_scale": scale.values}).to_csv(scale_path, index=False, encoding="utf-8-sig")
        command = _backtest_command(
            variant,
            score_panel_csv=source_score_path,
            target_weight_panel_csv=target_path,
            output_root=root,
            run_tag=run_tag,
            dataset_id=dataset_id,
            data_lake_root=data_lake_root,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            no_cache=no_cache,
        )
        run_result = None
        if run_backtests:
            run_result = _run_command(
                command,
                stdout_path=root / "logs" / f"{variant.variant_id}.stdout.log",
                stderr_path=root / "logs" / f"{variant.variant_id}.stderr.log",
            )
        scale_meta["state_scale_csv"] = str(scale_path)
        results.append(_result(variant, command=command, run_result=run_result, scale_meta=scale_meta, target_weight_panel_csv=target_path))

    summary_csv = root / "v2_state_sizing_matrix_summary.csv"
    frame = _summary_frame(results)
    frame.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    completed = frame.loc[frame["status"].eq("completed")].copy() if not frame.empty else pd.DataFrame()
    best: dict[str, Any] = {}
    if not completed.empty and completed["excess_sharpe"].notna().any():
        best = completed.sort_values(["promotion_review_eligible", "excess_sharpe"], ascending=[False, False]).iloc[0].to_dict()
    summary = {
        "variant_count": int(len(results)),
        "completed_backtest_count": int(frame["status"].eq("completed").sum()) if not frame.empty else 0,
        "promotion_review_eligible_count": int(frame["promotion_review_eligible"].sum()) if not frame.empty else 0,
        "best_by_excess_sharpe": best,
    }
    report = {
        "schema_version": 1,
        "status": "completed" if not run_backtests or all(item.get("status") == "completed" for item in results) else "backtest_failed",
        "run_tag": str(run_tag),
        "research_program": v2.RESEARCH_PROGRAM,
        "study_family": "v2_state_sizing_matrix",
        "source_matrix_run_tag": str(source_matrix_run_tag),
        "source_variant_id": str(source_variant_id),
        "source_backtest_dir": str(source_dir),
        "dataset_id": str(dataset_id),
        "pool_view_id": str(pool_view_id),
        "date_range": {"start": start_date, "end": end_date},
        "summary_csv": str(summary_csv),
        "summary": summary,
        "results": results,
        "boundary": {
            "research_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
            "execution_state_required": "frozen_skeleton_only",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_state_sizing_matrix_report.json", report)
    _write_json(root / "v2_state_sizing_matrix_manifest.json", {k: v for k, v in report.items() if k != "results"})
    _write_markdown(root / "v2_state_sizing_matrix_report.md", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run research-only v2 second-stage state sizing matrix.")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--source-matrix-run-tag", default=SOURCE_MATRIX_RUN_TAG)
    parser.add_argument("--source-variant-id", default=SOURCE_VARIANT_ID)
    parser.add_argument("--source-backtest-dir", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dataset-id", default=v2.DATASET_ID)
    parser.add_argument("--pool-view-id", default=v2.V2_STRICT_POOL_VIEW_ID)
    parser.add_argument("--data-lake-root", default=str(DATA_LAKE_ROOT))
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--scale-modes", default="drawdown,prev_month_loss,drawdown_or_prev_month_loss")
    parser.add_argument("--stress-scales", default="0.50,0.70")
    parser.add_argument("--drawdown-thresholds", default="0.03,0.05")
    parser.add_argument("--monthly-loss-thresholds", default="0.01")
    parser.add_argument("--run-backtests", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_state_sizing_matrix(
        run_tag=args.run_tag,
        source_matrix_run_tag=args.source_matrix_run_tag,
        source_variant_id=args.source_variant_id,
        source_backtest_dir=args.source_backtest_dir or None,
        output_root=args.output_root or None,
        dataset_id=args.dataset_id,
        pool_view_id=args.pool_view_id,
        data_lake_root=args.data_lake_root,
        benchmark=args.benchmark,
        scale_modes=_parse_strings(args.scale_modes, ("drawdown", "prev_month_loss", "drawdown_or_prev_month_loss")),
        stress_scales=_parse_floats(args.stress_scales, (0.50, 0.70)),
        drawdown_thresholds=_parse_floats(args.drawdown_thresholds, (0.03, 0.05)),
        monthly_loss_thresholds=_parse_floats(args.monthly_loss_thresholds, (0.01,)),
        run_backtests=bool(args.run_backtests),
        no_cache=bool(args.no_cache),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status={report.get('status')} run_tag={args.run_tag} output_root={args.output_root or STUDIES_ROOT / args.run_tag}")
    return 0 if str(report.get("status")) != "backtest_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
