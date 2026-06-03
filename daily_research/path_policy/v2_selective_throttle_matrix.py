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
from daily_research.path_policy import v2_bad_month_attribution as attribution
from daily_research.path_policy import v2_candidate_review_matrix as matrix
from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy import v2_score_backtest_bridge as bridge


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
DATA_LAKE_ROOT = PROJECT_ROOT / "daily_research/output/research_data_lake"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
RUN_TAG = "v2_selective_throttle_matrix_20260602_01"
SOURCE_BRIDGE_RUN_TAG = bridge.RUN_TAG


@dataclass(frozen=True)
class ThrottleVariant:
    variant_id: str
    holding_count: int
    max_weight: float
    rebalance_freq: str
    rebalance_offset_mode: str
    transaction_cost_bps: float
    slippage_bps: float
    sell_tax_bps: float
    throttle_mode: str
    penalty: float
    recent_return_window: int
    volatility_window: int
    recent_return_quantile: float
    volatility_quantile: float


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


def _parse_ints(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    values = [int(float(item.strip())) for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _parse_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _parse_strings(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _source_score_panel(source_bridge_root: str | Path) -> Path:
    return matrix._source_score_panel(source_bridge_root)


def _score_panel_dates(score_panel_csv: Path) -> tuple[str, str]:
    return matrix._score_panel_dates(score_panel_csv)


def _load_score_panel(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "stock", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Score panel missing required columns {missing}: {path}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stock"] = out["stock"].astype(str).str.strip().str.upper()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.loc[out["date"].notna() & out["stock"].ne("") & out["score"].notna()].copy()
    if out.empty:
        raise ValueError(f"Score panel has no usable rows: {path}")
    return out.sort_values(["date", "stock"]).reset_index(drop=True)


def _wide_score(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot_table(index="date", columns="stock", values="score", aggfunc="mean").sort_index()


def _stack_non_null(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    try:
        stacked = frame.stack(future_stack=True)
    except TypeError:
        stacked = frame.stack(dropna=True)
    return stacked.dropna().rename(value_name).reset_index()


def _cross_section_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, method="average")


def _risk_masks(
    close: pd.DataFrame,
    *,
    recent_return_window: int,
    volatility_window: int,
    recent_return_quantile: float,
    volatility_quantile: float,
) -> dict[str, pd.DataFrame]:
    returns = close.pct_change()
    recent = close.pct_change(max(int(recent_return_window), 1))
    volatility = returns.rolling(max(int(volatility_window), 2), min_periods=max(2, min(5, int(volatility_window)))).std()
    recent_pct = _cross_section_percentile(recent)
    volatility_pct = _cross_section_percentile(volatility)
    runup = recent_pct >= float(recent_return_quantile)
    high_vol = volatility_pct >= float(volatility_quantile)
    return {
        "recent_runup": runup.fillna(False),
        "high_volatility": high_vol.fillna(False),
        "recent_runup_or_high_volatility": (runup | high_vol).fillna(False),
        "recent_runup_and_high_volatility": (runup & high_vol).fillna(False),
    }


def build_adjusted_score_panel(
    score_frame: pd.DataFrame,
    close: pd.DataFrame,
    *,
    throttle_mode: str,
    penalty: float,
    recent_return_window: int,
    volatility_window: int,
    recent_return_quantile: float,
    volatility_quantile: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scores = _wide_score(score_frame)
    close = close.reindex(index=scores.index, columns=scores.columns).sort_index()
    mode = str(throttle_mode or "none").strip().lower()
    masks = _risk_masks(
        close,
        recent_return_window=recent_return_window,
        volatility_window=volatility_window,
        recent_return_quantile=recent_return_quantile,
        volatility_quantile=volatility_quantile,
    )
    if mode in {"none", "off"}:
        mask = pd.DataFrame(False, index=scores.index, columns=scores.columns)
    elif mode in masks:
        mask = masks[mode].reindex_like(scores).fillna(False)
    else:
        raise ValueError(f"Unsupported throttle_mode: {throttle_mode}")

    adjusted = scores.copy()
    adjusted = adjusted.mask(mask, adjusted - float(penalty))
    rows = _stack_non_null(adjusted, "score")
    rows.columns = ["date", "stock", "score"]
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    rows["stock"] = rows["stock"].astype(str)

    original_top = scores.rank(axis=1, ascending=False, method="first") <= 20
    adjusted_top = adjusted.rank(axis=1, ascending=False, method="first") <= 20
    top_mask = mask & original_top
    stats = {
        "throttle_mode": mode,
        "penalty": float(penalty),
        "recent_return_window": int(recent_return_window),
        "volatility_window": int(volatility_window),
        "recent_return_quantile": float(recent_return_quantile),
        "volatility_quantile": float(volatility_quantile),
        "row_count": int(len(rows)),
        "guarded_cell_count": int(mask.sum().sum()),
        "guarded_cell_ratio": float(mask.sum().sum() / max(mask.size, 1)),
        "guarded_original_top20_cell_count": int(top_mask.sum().sum()),
        "original_top20_retained_ratio": float((original_top & adjusted_top).sum().sum() / max(original_top.sum().sum(), 1)),
    }
    return rows.sort_values(["date", "stock"]).reset_index(drop=True), stats


def _variants(
    *,
    holding_counts: tuple[int, ...],
    max_weights: tuple[float, ...],
    rebalance_freqs: tuple[str, ...],
    rebalance_offset_modes: tuple[str, ...],
    transaction_cost_bps_values: tuple[float, ...],
    slippage_bps_values: tuple[float, ...],
    sell_tax_bps_values: tuple[float, ...],
    throttle_modes: tuple[str, ...],
    penalties: tuple[float, ...],
    recent_return_windows: tuple[int, ...],
    volatility_windows: tuple[int, ...],
    recent_return_quantiles: tuple[float, ...],
    volatility_quantiles: tuple[float, ...],
) -> list[ThrottleVariant]:
    out: list[ThrottleVariant] = []
    for holding_count in holding_counts:
        for max_weight in max_weights:
            for rebalance_freq in rebalance_freqs:
                for offset_mode in rebalance_offset_modes:
                    for cost in transaction_cost_bps_values:
                        for slippage in slippage_bps_values:
                            for tax in sell_tax_bps_values:
                                for mode in throttle_modes:
                                    for penalty in penalties:
                                        for rrw in recent_return_windows:
                                            for vw in volatility_windows:
                                                for rrq in recent_return_quantiles:
                                                    for vq in volatility_quantiles:
                                                        variant_id = (
                                                            f"h{holding_count}_mw{int(round(max_weight * 1000)):03d}_"
                                                            f"rb{rebalance_freq}_{offset_mode}_"
                                                            f"c{cost:g}_{slippage:g}_{tax:g}_"
                                                            f"thr_{mode}_p{penalty:g}_rrw{rrw}_vw{vw}_"
                                                            f"rq{int(round(rrq * 100)):02d}_vq{int(round(vq * 100)):02d}"
                                                        ).replace(".", "p")
                                                        out.append(
                                                            ThrottleVariant(
                                                                variant_id=variant_id,
                                                                holding_count=int(holding_count),
                                                                max_weight=float(max_weight),
                                                                rebalance_freq=str(rebalance_freq),
                                                                rebalance_offset_mode=str(offset_mode),
                                                                transaction_cost_bps=float(cost),
                                                                slippage_bps=float(slippage),
                                                                sell_tax_bps=float(tax),
                                                                throttle_mode=str(mode),
                                                                penalty=float(penalty),
                                                                recent_return_window=int(rrw),
                                                                volatility_window=int(vw),
                                                                recent_return_quantile=float(rrq),
                                                                volatility_quantile=float(vq),
                                                            )
                                                        )
    return out


def _variant_command(
    variant: ThrottleVariant,
    *,
    adjusted_score_panel_csv: Path,
    output_root: Path,
    run_tag: str,
    dataset_id: str,
    data_lake_root: str | Path,
    start_date: str,
    end_date: str,
    benchmark: str,
    no_cache: bool,
) -> list[str]:
    return bridge._backtest_command(
        score_panel_csv=adjusted_score_panel_csv,
        output_root=output_root,
        experiment_tag=f"{run_tag}_{variant.variant_id}",
        dataset_id=dataset_id,
        data_lake_root=data_lake_root,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        holding_count=variant.holding_count,
        max_weight=variant.max_weight,
        rebalance_freq=variant.rebalance_freq,
        rebalance_offset_mode=variant.rebalance_offset_mode,
        rebalance_anchor_date=start_date,
        score_threshold=-999.0,
        transaction_cost_bps=variant.transaction_cost_bps,
        slippage_bps=variant.slippage_bps,
        sell_tax_bps=variant.sell_tax_bps,
        use_market_regime_filter=False,
        no_cache=no_cache,
    )


def _run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    assert_task_allowed("candidate-backtest")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True, check=False)
    return {"returncode": int(proc.returncode), "stdout": str(stdout_path), "stderr": str(stderr_path)}


def _result(
    variant: ThrottleVariant,
    *,
    command: list[str],
    run_result: dict[str, Any] | None,
    throttle_stats: dict[str, Any],
    adjusted_score_panel_csv: Path,
) -> dict[str, Any]:
    base = matrix._variant_result(
        matrix.Variant(
            variant_id=variant.variant_id,
            holding_count=variant.holding_count,
            max_weight=variant.max_weight,
            rebalance_freq=variant.rebalance_freq,
            rebalance_offset_mode=variant.rebalance_offset_mode,
            transaction_cost_bps=variant.transaction_cost_bps,
            slippage_bps=variant.slippage_bps,
            sell_tax_bps=variant.sell_tax_bps,
            use_market_regime_filter=False,
        ),
        command=command,
        run_result=run_result,
    )
    base["variant"] = variant.__dict__
    base["adjusted_score_panel_csv"] = str(adjusted_score_panel_csv)
    base["throttle_stats"] = throttle_stats
    return base


def _summary_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = matrix._summary_frame(results)
    for idx, item in enumerate(results):
        stats = dict(item.get("throttle_stats", {}) or {})
        for key in ["guarded_cell_ratio", "guarded_original_top20_cell_count", "original_top20_retained_ratio"]:
            frame.loc[idx, key] = stats.get(key)
    return frame


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = dict(report.get("summary", {}) or {})
    best = dict(summary.get("best_by_excess_sharpe", {}) or {})
    lines = [
        "# V2 Selective Throttle Matrix",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Source bridge: `{report.get('source_bridge_run_tag', '')}`",
        f"- Variant count: `{summary.get('variant_count', 0)}`",
        f"- Completed backtests: `{summary.get('completed_backtest_count', 0)}`",
        f"- Promotion-review eligible variants: `{summary.get('promotion_review_eligible_count', 0)}`",
        "",
        "## Boundary",
        "",
        "- research_only: `true`",
        "- promotion_allowed: `false`",
        "- active_execution_strategy_expected_diff: `none`",
        "- Selective throttle uses only historical close-derived state and model score; it is not a production target-weight panel.",
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


def build_selective_throttle_matrix(
    *,
    run_tag: str = RUN_TAG,
    source_bridge_run_tag: str = SOURCE_BRIDGE_RUN_TAG,
    source_bridge_root: str | Path | None = None,
    output_root: str | Path | None = None,
    dataset_id: str = v2.DATASET_ID,
    pool_view_id: str = v2.V2_STRICT_POOL_VIEW_ID,
    data_lake_root: str | Path = DATA_LAKE_ROOT,
    benchmark: str = "000300.SH",
    holding_counts: tuple[int, ...] = (20,),
    max_weights: tuple[float, ...] = (0.06, 0.08),
    rebalance_freqs: tuple[str, ...] = ("5d",),
    rebalance_offset_modes: tuple[str, ...] = ("all",),
    transaction_cost_bps_values: tuple[float, ...] = (3.0,),
    slippage_bps_values: tuple[float, ...] = (7.0,),
    sell_tax_bps_values: tuple[float, ...] = (10.0,),
    throttle_modes: tuple[str, ...] = ("recent_runup_or_high_volatility",),
    penalties: tuple[float, ...] = (0.05, 0.10),
    recent_return_windows: tuple[int, ...] = (5, 10),
    volatility_windows: tuple[int, ...] = (20,),
    recent_return_quantiles: tuple[float, ...] = (0.80, 0.90),
    volatility_quantiles: tuple[float, ...] = (0.80, 0.90),
    run_backtests: bool = False,
    no_cache: bool = False,
    enforce_active_artifact_clean: bool = True,
) -> dict[str, Any]:
    if enforce_active_artifact_clean and _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = Path(output_root) if output_root is not None else STUDIES_ROOT / str(run_tag)
    root.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_bridge_root) if source_bridge_root is not None else STUDIES_ROOT / str(source_bridge_run_tag)
    score_panel_csv = _source_score_panel(source_root)
    score_frame = _load_score_panel(score_panel_csv)
    start_date, end_date = _score_panel_dates(score_panel_csv)
    symbols = sorted(score_frame["stock"].unique().tolist())
    close = attribution._load_close_panel(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        dataset_id=dataset_id,
        data_lake_root=data_lake_root,
        benchmark=benchmark,
        use_cache=not bool(no_cache),
    )

    variants = _variants(
        holding_counts=holding_counts,
        max_weights=max_weights,
        rebalance_freqs=rebalance_freqs,
        rebalance_offset_modes=rebalance_offset_modes,
        transaction_cost_bps_values=transaction_cost_bps_values,
        slippage_bps_values=slippage_bps_values,
        sell_tax_bps_values=sell_tax_bps_values,
        throttle_modes=throttle_modes,
        penalties=penalties,
        recent_return_windows=recent_return_windows,
        volatility_windows=volatility_windows,
        recent_return_quantiles=recent_return_quantiles,
        volatility_quantiles=volatility_quantiles,
    )
    results: list[dict[str, Any]] = []
    for variant in variants:
        adjusted, stats = build_adjusted_score_panel(
            score_frame,
            close,
            throttle_mode=variant.throttle_mode,
            penalty=variant.penalty,
            recent_return_window=variant.recent_return_window,
            volatility_window=variant.volatility_window,
            recent_return_quantile=variant.recent_return_quantile,
            volatility_quantile=variant.volatility_quantile,
        )
        adjusted_path = root / "adjusted_score_panels" / f"{variant.variant_id}.csv"
        adjusted_path.parent.mkdir(parents=True, exist_ok=True)
        adjusted.to_csv(adjusted_path, index=False, encoding="utf-8-sig")
        command = _variant_command(
            variant,
            adjusted_score_panel_csv=adjusted_path,
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
        results.append(_result(variant, command=command, run_result=run_result, throttle_stats=stats, adjusted_score_panel_csv=adjusted_path))

    summary_csv = root / "v2_selective_throttle_matrix_summary.csv"
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
        "study_family": "v2_selective_throttle_matrix",
        "source_bridge_run_tag": str(source_bridge_run_tag),
        "source_bridge_root": str(source_root),
        "source_score_panel_csv": str(score_panel_csv),
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
    _write_json(root / "v2_selective_throttle_matrix_report.json", report)
    _write_json(root / "v2_selective_throttle_matrix_manifest.json", {k: v for k, v in report.items() if k != "results"})
    _write_markdown(root / "v2_selective_throttle_matrix_report.md", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run research-only v2 selective high-score risk throttle candidate matrix.")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--source-bridge-run-tag", default=SOURCE_BRIDGE_RUN_TAG)
    parser.add_argument("--source-bridge-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dataset-id", default=v2.DATASET_ID)
    parser.add_argument("--pool-view-id", default=v2.V2_STRICT_POOL_VIEW_ID)
    parser.add_argument("--data-lake-root", default=str(DATA_LAKE_ROOT))
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--holding-counts", default="20")
    parser.add_argument("--max-weights", default="0.06,0.08")
    parser.add_argument("--rebalance-freqs", default="5d")
    parser.add_argument("--rebalance-offset-modes", default="all")
    parser.add_argument("--transaction-cost-bps-values", default="3")
    parser.add_argument("--slippage-bps-values", default="7")
    parser.add_argument("--sell-tax-bps-values", default="10")
    parser.add_argument("--throttle-modes", default="recent_runup_or_high_volatility")
    parser.add_argument("--penalties", default="0.05,0.10")
    parser.add_argument("--recent-return-windows", default="5,10")
    parser.add_argument("--volatility-windows", default="20")
    parser.add_argument("--recent-return-quantiles", default="0.80,0.90")
    parser.add_argument("--volatility-quantiles", default="0.80,0.90")
    parser.add_argument("--run-backtests", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = build_selective_throttle_matrix(
        run_tag=args.run_tag,
        source_bridge_run_tag=args.source_bridge_run_tag,
        source_bridge_root=args.source_bridge_root or None,
        output_root=args.output_root or None,
        dataset_id=args.dataset_id,
        pool_view_id=args.pool_view_id,
        data_lake_root=args.data_lake_root,
        benchmark=args.benchmark,
        holding_counts=_parse_ints(args.holding_counts, (20,)),
        max_weights=_parse_floats(args.max_weights, (0.06, 0.08)),
        rebalance_freqs=_parse_strings(args.rebalance_freqs, ("5d",)),
        rebalance_offset_modes=_parse_strings(args.rebalance_offset_modes, ("all",)),
        transaction_cost_bps_values=_parse_floats(args.transaction_cost_bps_values, (3.0,)),
        slippage_bps_values=_parse_floats(args.slippage_bps_values, (7.0,)),
        sell_tax_bps_values=_parse_floats(args.sell_tax_bps_values, (10.0,)),
        throttle_modes=_parse_strings(args.throttle_modes, ("recent_runup_or_high_volatility",)),
        penalties=_parse_floats(args.penalties, (0.05, 0.10)),
        recent_return_windows=_parse_ints(args.recent_return_windows, (5, 10)),
        volatility_windows=_parse_ints(args.volatility_windows, (20,)),
        recent_return_quantiles=_parse_floats(args.recent_return_quantiles, (0.80, 0.90)),
        volatility_quantiles=_parse_floats(args.volatility_quantiles, (0.80, 0.90)),
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
