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
DATA_LAKE_ROOT = PROJECT_ROOT / "daily_research/output/research_data_lake"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
RUN_TAG = "v2_score_state_calibration_matrix_20260602_01"
SOURCE_TEST_BRIDGE_RUN_TAG = bridge.RUN_TAG


@dataclass(frozen=True)
class ScoreStateCalibrationVariant:
    variant_id: str
    calibration_mode: str
    penalty: float
    score_quantile: float
    recent_return_quantile: float
    volatility_quantile: float
    reversal_quantile: float
    recent_return_window: int
    volatility_window: int
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


def _parse_ints(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    values = [int(float(item.strip())) for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _parse_strings(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _load_ensemble_panel(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "stock", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ensemble panel missing required columns {missing}: {path}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stock"] = out["stock"].astype(str).str.strip().str.upper()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.loc[out["date"].notna() & out["stock"].ne("") & out["score"].notna()].copy()
    if out.empty:
        raise ValueError(f"Ensemble panel has no usable rows: {path}")
    return out.sort_values(["date", "stock"]).reset_index(drop=True)


def _wide_value(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    work = frame.loc[:, ["date", "stock", value_column]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["stock"] = work["stock"].astype(str).str.strip().str.upper()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.loc[work["date"].notna() & work["stock"].ne("") & work[value_column].notna()].copy()
    return work.pivot_table(index="date", columns="stock", values=value_column, aggfunc="mean").sort_index()


def _long_score_with_payload(score: pd.DataFrame, payload: pd.DataFrame, value_column: str = "score") -> pd.DataFrame:
    try:
        stacked = score.stack(future_stack=True)
    except TypeError:
        stacked = score.stack(dropna=True)
    out = stacked.dropna().rename(value_column).reset_index()
    out.columns = ["date", "stock", value_column]
    out["date"] = pd.to_datetime(out["date"])
    out["stock"] = out["stock"].astype(str).str.strip().str.upper()
    payload_norm = payload.copy()
    payload_norm["date"] = pd.to_datetime(payload_norm["date"], errors="coerce")
    payload_norm["stock"] = payload_norm["stock"].astype(str).str.strip().str.upper()
    columns_to_keep = [column for column in payload.columns if column not in {value_column}]
    merged = out.merge(payload_norm.loc[:, columns_to_keep], on=["date", "stock"], how="left")
    return merged.sort_values(["date", "stock"]).reset_index(drop=True)


def _cross_section_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, method="average")


def _state_masks(
    close: pd.DataFrame,
    score: pd.DataFrame,
    *,
    recent_return_window: int,
    volatility_window: int,
    score_quantile: float,
    recent_return_quantile: float,
    volatility_quantile: float,
    reversal_quantile: float,
) -> dict[str, pd.DataFrame]:
    close = close.reindex(columns=score.columns).sort_index()
    returns = close.pct_change()
    recent = close.pct_change(max(int(recent_return_window), 1)).shift(1).reindex(index=score.index, columns=score.columns)
    volatility = (
        returns.rolling(max(int(volatility_window), 2), min_periods=max(2, min(5, int(volatility_window))))
        .std()
        .shift(1)
        .reindex(index=score.index, columns=score.columns)
    )
    reversal = returns.shift(1).reindex(index=score.index, columns=score.columns)
    score_hi = _cross_section_percentile(score) >= float(score_quantile)
    runup_hi = _cross_section_percentile(recent) >= float(recent_return_quantile)
    vol_hi = _cross_section_percentile(volatility) >= float(volatility_quantile)
    reversal_hi = _cross_section_percentile(reversal) >= float(reversal_quantile)
    return {
        "score_runup": (score_hi & runup_hi).fillna(False),
        "score_volatility": (score_hi & vol_hi).fillna(False),
        "score_reversal": (score_hi & reversal_hi).fillna(False),
        "score_volatility_or_reversal": (score_hi & (vol_hi | reversal_hi)).fillna(False),
        "score_runup_or_volatility": (score_hi & (runup_hi | vol_hi)).fillna(False),
        "score_runup_or_volatility_or_reversal": (score_hi & (runup_hi | vol_hi | reversal_hi)).fillna(False),
    }


def adjust_score_panel(
    panel: pd.DataFrame,
    close: pd.DataFrame,
    *,
    calibration_mode: str,
    penalty: float,
    score_quantile: float,
    recent_return_quantile: float,
    volatility_quantile: float,
    reversal_quantile: float,
    recent_return_window: int,
    volatility_window: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    score = _wide_value(panel, "score")
    masks = _state_masks(
        close,
        score,
        recent_return_window=recent_return_window,
        volatility_window=volatility_window,
        score_quantile=score_quantile,
        recent_return_quantile=recent_return_quantile,
        volatility_quantile=volatility_quantile,
        reversal_quantile=reversal_quantile,
    )
    mode = str(calibration_mode or "score_volatility").strip().lower()
    if mode in {"none", "off"}:
        mask = pd.DataFrame(False, index=score.index, columns=score.columns)
    elif mode in masks:
        mask = masks[mode].reindex_like(score).fillna(False)
    else:
        raise ValueError(f"Unsupported calibration_mode: {calibration_mode}")
    adjusted = score.mask(mask, score - float(penalty))
    adjusted_panel = _long_score_with_payload(adjusted, panel, value_column="score")
    original_top = score.rank(axis=1, ascending=False, method="first") <= 20
    adjusted_top = adjusted.rank(axis=1, ascending=False, method="first") <= 20
    meta = {
        "calibration_mode": mode,
        "penalty": float(penalty),
        "score_quantile": float(score_quantile),
        "recent_return_quantile": float(recent_return_quantile),
        "volatility_quantile": float(volatility_quantile),
        "reversal_quantile": float(reversal_quantile),
        "recent_return_window": int(recent_return_window),
        "volatility_window": int(volatility_window),
        "guarded_cell_count": int(mask.sum().sum()),
        "guarded_cell_ratio": float(mask.sum().sum() / max(mask.size, 1)),
        "guarded_original_top20_cell_count": int((mask & original_top).sum().sum()),
        "original_top20_retained_ratio": float((original_top & adjusted_top).sum().sum() / max(original_top.sum().sum(), 1)),
    }
    return adjusted_panel, meta


def _validation_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    rank_ic = bridge._daily_rank_ic(frame, score_column="score", target_column="future_decision_score")
    spread = bridge._top_bottom_spread(frame, score_column="score", target_column="future_decision_score")
    hit = bridge._top20_hit_lift(frame, score_column="score")
    score = 0.0
    if rank_ic.get("available"):
        score += float(rank_ic.get("mean") or 0.0)
    if spread.get("available"):
        score += float(spread.get("mean") or 0.0)
    if hit.get("available"):
        score += float(hit.get("mean") or 0.0) * 2.0
    return {
        "rank_ic_mean": rank_ic.get("mean"),
        "rank_ic_min": rank_ic.get("min"),
        "spread_mean": spread.get("mean"),
        "spread_min": spread.get("min"),
        "top20_hit_lift_mean": hit.get("mean"),
        "top20_hit_lift_min": hit.get("min"),
        "validation_selection_score": float(score),
        "rank_ic_available": bool(rank_ic.get("available", False)),
        "spread_available": bool(spread.get("available", False)),
        "hit_lift_available": bool(hit.get("available", False)),
    }


def _variants(
    *,
    calibration_modes: tuple[str, ...],
    penalties: tuple[float, ...],
    score_quantiles: tuple[float, ...],
    recent_return_quantiles: tuple[float, ...],
    volatility_quantiles: tuple[float, ...],
    reversal_quantiles: tuple[float, ...],
    recent_return_windows: tuple[int, ...],
    volatility_windows: tuple[int, ...],
    holding_count: int,
    max_weight: float,
    rebalance_freq: str,
    rebalance_offset_mode: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> list[ScoreStateCalibrationVariant]:
    out: list[ScoreStateCalibrationVariant] = []
    for mode in calibration_modes:
        for penalty in penalties:
            for sq in score_quantiles:
                for rrq in recent_return_quantiles:
                    for vq in volatility_quantiles:
                        for rvq in reversal_quantiles:
                            for rrw in recent_return_windows:
                                for vw in volatility_windows:
                                    variant_id = (
                                        f"cal_{mode}_p{penalty:g}_sq{int(round(sq * 100)):02d}_"
                                        f"rq{int(round(rrq * 100)):02d}_vq{int(round(vq * 100)):02d}_"
                                        f"rvq{int(round(rvq * 100)):02d}_rrw{rrw}_vw{vw}_"
                                        f"h{holding_count}_mw{int(round(max_weight * 1000)):03d}"
                                    ).replace(".", "p")
                                    out.append(
                                        ScoreStateCalibrationVariant(
                                            variant_id=variant_id,
                                            calibration_mode=str(mode),
                                            penalty=float(penalty),
                                            score_quantile=float(sq),
                                            recent_return_quantile=float(rrq),
                                            volatility_quantile=float(vq),
                                            reversal_quantile=float(rvq),
                                            recent_return_window=int(rrw),
                                            volatility_window=int(vw),
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


def _load_close_for_panel(
    panel: pd.DataFrame,
    *,
    dataset_id: str,
    data_lake_root: str | Path,
    benchmark: str,
    no_cache: bool,
) -> pd.DataFrame:
    from daily_research.path_policy import v2_bad_month_attribution as attribution

    dates = pd.to_datetime(panel["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Cannot load close panel without panel dates.")
    symbols = sorted(panel["stock"].astype(str).str.upper().unique().tolist())
    load_start = (dates.min() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    load_end = dates.max().strftime("%Y-%m-%d")
    return attribution._load_close_panel(
        symbols=symbols,
        start_date=load_start,
        end_date=load_end,
        dataset_id=dataset_id,
        data_lake_root=data_lake_root,
        benchmark=benchmark,
        use_cache=not bool(no_cache),
    )


def _panel_dates(frame: pd.DataFrame) -> tuple[str, str]:
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Score panel has no usable dates.")
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def _backtest_command(
    variant: ScoreStateCalibrationVariant,
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
    variant: ScoreStateCalibrationVariant,
    *,
    command: list[str],
    run_result: dict[str, Any] | None,
    validation_row: dict[str, Any],
    adjusted_score_panel_csv: Path,
    calibration_meta: dict[str, Any],
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
    item["validation_selection"] = validation_row
    item["calibration_meta"] = calibration_meta
    item["adjusted_score_panel_csv"] = str(adjusted_score_panel_csv)
    return item


def _summary_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = candidate_matrix._summary_frame(results)
    for idx, item in enumerate(results):
        validation = dict(item.get("validation_selection", {}) or {})
        meta = dict(item.get("calibration_meta", {}) or {})
        for key in [
            "validation_selection_score",
            "rank_ic_mean",
            "spread_mean",
            "top20_hit_lift_mean",
            "guarded_cell_ratio",
            "original_top20_retained_ratio",
        ]:
            frame.loc[idx, key] = validation.get(key, meta.get(key))
    return frame


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = dict(report.get("summary", {}) or {})
    best = dict(summary.get("best_by_validation_then_test", {}) or {})
    lines = [
        "# V2 Score State Calibration Matrix",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Variant count: `{summary.get('variant_count', 0)}`",
        f"- Test backtests: `{summary.get('test_backtest_count', 0)}`",
        f"- Promotion-review eligible variants: `{summary.get('promotion_review_eligible_count', 0)}`",
        "",
        "## Boundary",
        "",
        "- research_only: `true`",
        "- promotion_allowed: `false`",
        "- active_execution_strategy_expected_diff: `none`",
        "- Validation labels select calibration variants; test backtests remain research candidates, not production target weights.",
    ]
    if best:
        lines.extend(
            [
                "",
                "## Best Selected Variant",
                "",
                f"- Variant: `{best.get('variant_id', '')}`",
                f"- Validation score: `{best.get('validation_selection_score', '')}`",
                f"- Excess Sharpe: `{best.get('excess_sharpe', '')}`",
                f"- Positive month ratio: `{best.get('positive_month_ratio', '')}`",
                f"- Negative months: `{best.get('negative_month_count', '')}`",
                f"- Max drawdown: `{best.get('max_drawdown', '')}`",
                f"- Issue flags: `{best.get('issue_flags', '')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_bridge_panel(
    *,
    role: str,
    bridge_root: Path,
    run_tag: str,
    dataset_id: str,
    pool_view_id: str,
    feature_profile: str,
    enforce_active_artifact_clean: bool,
) -> Path:
    panel = bridge_root / f"v2_ensemble_score_panel_{role}.csv"
    if panel.exists():
        return panel
    bridge.build_v2_score_bridge(
        output_root=bridge_root,
        run_tag=run_tag,
        role=role,
        dataset_id=dataset_id,
        pool_view_id=pool_view_id,
        feature_profile=feature_profile,
        run_backtest=False,
        enforce_active_artifact_clean=enforce_active_artifact_clean,
    )
    if not panel.exists():
        raise FileNotFoundError(f"Bridge ensemble panel was not produced: {panel}")
    return panel


def build_score_state_calibration_matrix(
    *,
    run_tag: str = RUN_TAG,
    output_root: str | Path | None = None,
    validation_ensemble_csv: str | Path | None = None,
    test_ensemble_csv: str | Path | None = None,
    source_test_bridge_run_tag: str = SOURCE_TEST_BRIDGE_RUN_TAG,
    dataset_id: str = v2.DATASET_ID,
    pool_view_id: str = v2.V2_STRICT_POOL_VIEW_ID,
    feature_profile: str = v2.FEATURE_PROFILE,
    data_lake_root: str | Path = DATA_LAKE_ROOT,
    benchmark: str = "000300.SH",
    calibration_modes: tuple[str, ...] = ("score_volatility", "score_reversal", "score_volatility_or_reversal"),
    penalties: tuple[float, ...] = (0.02, 0.05, 0.08),
    score_quantiles: tuple[float, ...] = (0.80,),
    recent_return_quantiles: tuple[float, ...] = (0.80,),
    volatility_quantiles: tuple[float, ...] = (0.80,),
    reversal_quantiles: tuple[float, ...] = (0.80,),
    recent_return_windows: tuple[int, ...] = (5,),
    volatility_windows: tuple[int, ...] = (20,),
    top_n_test_backtests: int = 4,
    holding_count: int = bridge.DEFAULT_HOLDING_COUNT,
    max_weight: float = bridge.DEFAULT_MAX_WEIGHT,
    rebalance_freq: str = bridge.DEFAULT_REBALANCE_FREQ,
    rebalance_offset_mode: str = "all",
    transaction_cost_bps: float = bridge.DEFAULT_COST_BPS,
    slippage_bps: float = bridge.DEFAULT_SLIPPAGE_BPS,
    sell_tax_bps: float = bridge.DEFAULT_SELL_TAX_BPS,
    run_backtests: bool = False,
    no_cache: bool = False,
    enforce_active_artifact_clean: bool = True,
) -> dict[str, Any]:
    if enforce_active_artifact_clean and _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = Path(output_root) if output_root is not None else STUDIES_ROOT / str(run_tag)
    root.mkdir(parents=True, exist_ok=True)
    validation_bridge_root = root / "validation_bridge"
    test_bridge_root = STUDIES_ROOT / str(source_test_bridge_run_tag)
    validation_panel_path = (
        Path(validation_ensemble_csv)
        if validation_ensemble_csv is not None
        else _ensure_bridge_panel(
            role="validation",
            bridge_root=validation_bridge_root,
            run_tag=f"{run_tag}_validation_bridge",
            dataset_id=dataset_id,
            pool_view_id=pool_view_id,
            feature_profile=feature_profile,
            enforce_active_artifact_clean=enforce_active_artifact_clean,
        )
    )
    test_panel_path = (
        Path(test_ensemble_csv)
        if test_ensemble_csv is not None
        else _ensure_bridge_panel(
            role="test",
            bridge_root=test_bridge_root,
            run_tag=source_test_bridge_run_tag,
            dataset_id=dataset_id,
            pool_view_id=pool_view_id,
            feature_profile=feature_profile,
            enforce_active_artifact_clean=enforce_active_artifact_clean,
        )
    )
    validation_panel = _load_ensemble_panel(validation_panel_path)
    test_panel = _load_ensemble_panel(test_panel_path)
    validation_close = _load_close_for_panel(validation_panel, dataset_id=dataset_id, data_lake_root=data_lake_root, benchmark=benchmark, no_cache=no_cache)
    test_close = _load_close_for_panel(test_panel, dataset_id=dataset_id, data_lake_root=data_lake_root, benchmark=benchmark, no_cache=no_cache)
    variants = _variants(
        calibration_modes=calibration_modes,
        penalties=penalties,
        score_quantiles=score_quantiles,
        recent_return_quantiles=recent_return_quantiles,
        volatility_quantiles=volatility_quantiles,
        reversal_quantiles=reversal_quantiles,
        recent_return_windows=recent_return_windows,
        volatility_windows=volatility_windows,
        holding_count=holding_count,
        max_weight=max_weight,
        rebalance_freq=rebalance_freq,
        rebalance_offset_mode=rebalance_offset_mode,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        sell_tax_bps=sell_tax_bps,
    )
    validation_rows: list[dict[str, Any]] = []
    for variant in variants:
        adjusted_validation, meta = adjust_score_panel(
            validation_panel,
            validation_close,
            calibration_mode=variant.calibration_mode,
            penalty=variant.penalty,
            score_quantile=variant.score_quantile,
            recent_return_quantile=variant.recent_return_quantile,
            volatility_quantile=variant.volatility_quantile,
            reversal_quantile=variant.reversal_quantile,
            recent_return_window=variant.recent_return_window,
            volatility_window=variant.volatility_window,
        )
        metrics = _validation_metrics(adjusted_validation)
        validation_rows.append({**variant.__dict__, **meta, **metrics})
    validation_frame = pd.DataFrame(validation_rows)
    validation_csv = root / "v2_score_state_calibration_validation_matrix.csv"
    validation_frame.to_csv(validation_csv, index=False, encoding="utf-8-sig")
    selected = validation_frame.sort_values(
        ["validation_selection_score", "top20_hit_lift_mean", "rank_ic_mean"],
        ascending=[False, False, False],
    ).head(max(int(top_n_test_backtests), 0))
    selected_ids = set(selected["variant_id"].astype(str).tolist())
    results: list[dict[str, Any]] = []
    start_date, end_date = _panel_dates(test_panel)
    for variant in variants:
        if str(variant.variant_id) not in selected_ids:
            continue
        adjusted_test, meta = adjust_score_panel(
            test_panel,
            test_close,
            calibration_mode=variant.calibration_mode,
            penalty=variant.penalty,
            score_quantile=variant.score_quantile,
            recent_return_quantile=variant.recent_return_quantile,
            volatility_quantile=variant.volatility_quantile,
            reversal_quantile=variant.reversal_quantile,
            recent_return_window=variant.recent_return_window,
            volatility_window=variant.volatility_window,
        )
        adjusted_path = root / "adjusted_test_score_panels" / f"{variant.variant_id}.csv"
        adjusted_path.parent.mkdir(parents=True, exist_ok=True)
        adjusted_test.loc[:, ["date", "stock", "score"]].assign(date=lambda df: pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")).to_csv(
            adjusted_path,
            index=False,
            encoding="utf-8-sig",
        )
        command = _backtest_command(
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
        validation_row = validation_frame.loc[validation_frame["variant_id"].astype(str).eq(str(variant.variant_id))].iloc[0].to_dict()
        results.append(
            _result(
                variant,
                command=command,
                run_result=run_result,
                validation_row=validation_row,
                adjusted_score_panel_csv=adjusted_path,
                calibration_meta=meta,
            )
        )
    summary_csv = root / "v2_score_state_calibration_test_summary.csv"
    summary_frame = _summary_frame(results)
    summary_frame.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    completed = summary_frame.loc[summary_frame["status"].eq("completed")].copy() if not summary_frame.empty else pd.DataFrame()
    best: dict[str, Any] = {}
    if not completed.empty:
        best = completed.sort_values(
            ["promotion_review_eligible", "validation_selection_score", "excess_sharpe"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()
    summary = {
        "variant_count": int(len(variants)),
        "selected_variant_count": int(len(results)),
        "test_backtest_count": int(summary_frame["status"].eq("completed").sum()) if not summary_frame.empty else 0,
        "promotion_review_eligible_count": int(summary_frame["promotion_review_eligible"].sum()) if not summary_frame.empty else 0,
        "best_by_validation_then_test": best,
    }
    report = {
        "schema_version": 1,
        "status": "completed" if not run_backtests or all(item.get("status") == "completed" for item in results) else "backtest_failed",
        "run_tag": str(run_tag),
        "research_program": v2.RESEARCH_PROGRAM,
        "study_family": "v2_score_state_calibration_matrix",
        "dataset_id": str(dataset_id),
        "pool_view_id": str(pool_view_id),
        "feature_profile": str(feature_profile),
        "validation_ensemble_csv": str(validation_panel_path),
        "test_ensemble_csv": str(test_panel_path),
        "validation_matrix_csv": str(validation_csv),
        "summary_csv": str(summary_csv),
        "summary": summary,
        "validation_selection": selected.to_dict("records"),
        "results": results,
        "boundary": {
            "research_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
            "selection_role": "validation",
            "candidate_backtest_role": "test",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_score_state_calibration_matrix_report.json", report)
    _write_json(root / "v2_score_state_calibration_matrix_manifest.json", {k: v for k, v in report.items() if k != "results"})
    _write_markdown(root / "v2_score_state_calibration_matrix_report.md", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run validation-selected v2 score/local-state calibration matrix.")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--validation-ensemble-csv", default="")
    parser.add_argument("--test-ensemble-csv", default="")
    parser.add_argument("--source-test-bridge-run-tag", default=SOURCE_TEST_BRIDGE_RUN_TAG)
    parser.add_argument("--dataset-id", default=v2.DATASET_ID)
    parser.add_argument("--pool-view-id", default=v2.V2_STRICT_POOL_VIEW_ID)
    parser.add_argument("--feature-profile", default=v2.FEATURE_PROFILE)
    parser.add_argument("--data-lake-root", default=str(DATA_LAKE_ROOT))
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--calibration-modes", default="score_volatility,score_reversal,score_volatility_or_reversal")
    parser.add_argument("--penalties", default="0.02,0.05,0.08")
    parser.add_argument("--score-quantiles", default="0.80")
    parser.add_argument("--recent-return-quantiles", default="0.80")
    parser.add_argument("--volatility-quantiles", default="0.80")
    parser.add_argument("--reversal-quantiles", default="0.80")
    parser.add_argument("--recent-return-windows", default="5")
    parser.add_argument("--volatility-windows", default="20")
    parser.add_argument("--top-n-test-backtests", type=int, default=4)
    parser.add_argument("--run-backtests", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_score_state_calibration_matrix(
        run_tag=args.run_tag,
        output_root=args.output_root or None,
        validation_ensemble_csv=args.validation_ensemble_csv or None,
        test_ensemble_csv=args.test_ensemble_csv or None,
        source_test_bridge_run_tag=args.source_test_bridge_run_tag,
        dataset_id=args.dataset_id,
        pool_view_id=args.pool_view_id,
        feature_profile=args.feature_profile,
        data_lake_root=args.data_lake_root,
        benchmark=args.benchmark,
        calibration_modes=_parse_strings(args.calibration_modes, ("score_volatility", "score_reversal", "score_volatility_or_reversal")),
        penalties=_parse_floats(args.penalties, (0.02, 0.05, 0.08)),
        score_quantiles=_parse_floats(args.score_quantiles, (0.80,)),
        recent_return_quantiles=_parse_floats(args.recent_return_quantiles, (0.80,)),
        volatility_quantiles=_parse_floats(args.volatility_quantiles, (0.80,)),
        reversal_quantiles=_parse_floats(args.reversal_quantiles, (0.80,)),
        recent_return_windows=_parse_ints(args.recent_return_windows, (5,)),
        volatility_windows=_parse_ints(args.volatility_windows, (20,)),
        top_n_test_backtests=int(args.top_n_test_backtests),
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
