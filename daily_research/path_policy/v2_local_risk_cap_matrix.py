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
from daily_research.path_policy import v2_state_sizing_matrix as state_sizing


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
DATA_LAKE_ROOT = PROJECT_ROOT / "daily_research/output/research_data_lake"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
RUN_TAG = "v2_local_risk_cap_matrix_20260602_01"
SOURCE_MATRIX_RUN_TAG = "v2_selective_throttle_matrix_narrow_20260602_01"
SOURCE_VARIANT_ID = "h20_mw080_rb5d_all_c3_7_10_thr_recent_runup_or_high_volatility_p0p05_rrw5_vw20_rq80_vq80"


@dataclass(frozen=True)
class LocalRiskCapVariant:
    variant_id: str
    cap_mode: str
    stress_scale: float
    score_quantile: float
    recent_return_quantile: float
    volatility_quantile: float
    reversal_quantile: float
    recent_return_window: int
    volatility_window: int
    redistribution_mode: str
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


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", str(ACTIVE_MANIFEST)], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _parse_floats(raw: str, default: tuple[float, ...]) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _parse_strings(raw: str, default: tuple[str, ...]) -> tuple[str, ...]:
    values = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _parse_ints(raw: str, default: tuple[int, ...]) -> tuple[int, ...]:
    values = [int(float(item.strip())) for item in str(raw or "").split(",") if item.strip()]
    return tuple(dict.fromkeys(values or list(default)))


def _default_source_backtest_dir(matrix_run_tag: str, variant_id: str) -> Path:
    return STUDIES_ROOT / str(matrix_run_tag) / "shared_backtest" / f"{matrix_run_tag}_{variant_id}"


def _cross_section_percentile(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True, method="average")


def _local_state_panels(
    close: pd.DataFrame,
    *,
    score: pd.DataFrame,
    recent_return_window: int,
    volatility_window: int,
) -> dict[str, pd.DataFrame]:
    close = close.reindex(index=score.index, columns=score.columns).sort_index()
    returns = close.pct_change()
    recent_return = close.pct_change(max(int(recent_return_window), 1)).shift(1)
    volatility = (
        returns.rolling(max(int(volatility_window), 2), min_periods=max(2, min(5, int(volatility_window))))
        .std()
        .shift(1)
    )
    reversal = returns.shift(1)
    return {
        "score_pct": _cross_section_percentile(score),
        "recent_return_pct": _cross_section_percentile(recent_return),
        "volatility_pct": _cross_section_percentile(volatility),
        "reversal_pct": _cross_section_percentile(reversal),
    }


def _risk_mask(
    state: dict[str, pd.DataFrame],
    *,
    cap_mode: str,
    score_quantile: float,
    recent_return_quantile: float,
    volatility_quantile: float,
    reversal_quantile: float,
) -> pd.DataFrame:
    score_hi = state["score_pct"] >= float(score_quantile)
    runup_hi = state["recent_return_pct"] >= float(recent_return_quantile)
    vol_hi = state["volatility_pct"] >= float(volatility_quantile)
    reversal_hi = state["reversal_pct"] >= float(reversal_quantile)
    mode = str(cap_mode or "score_volatility_or_reversal").strip().lower()
    if mode == "score_volatility":
        mask = score_hi & vol_hi
    elif mode == "score_reversal":
        mask = score_hi & reversal_hi
    elif mode == "score_volatility_or_reversal":
        mask = score_hi & (vol_hi | reversal_hi)
    elif mode == "score_runup_or_volatility_or_reversal":
        mask = score_hi & (runup_hi | vol_hi | reversal_hi)
    elif mode == "volatility_or_reversal":
        mask = vol_hi | reversal_hi
    elif mode == "recent_runup":
        mask = runup_hi
    elif mode in {"none", "off"}:
        template = state["score_pct"]
        mask = pd.DataFrame(False, index=template.index, columns=template.columns)
    else:
        raise ValueError(f"Unsupported cap_mode: {cap_mode}")
    return mask.fillna(False)


def _redistribute_row(
    original: pd.Series,
    scaled: pd.Series,
    risk: pd.Series,
    score: pd.Series,
    *,
    max_weight: float,
    redistribution_mode: str,
) -> pd.Series:
    out = scaled.fillna(0.0).clip(lower=0.0).copy()
    source_sum = float(original.fillna(0.0).clip(lower=0.0).sum())
    if source_sum <= 0:
        return out
    target_sum = min(source_sum, 1.0)
    cap = max(float(max_weight), 1e-12)
    eligible = (original.fillna(0.0) > 1e-12) & (~risk.fillna(False)) & (out < cap - 1e-12)
    mode = str(redistribution_mode or "source_weight").strip().lower()
    for _ in range(8):
        shortfall = target_sum - float(out.sum())
        if shortfall <= 1e-12 or not bool(eligible.any()):
            break
        if mode == "score":
            basis = score.where(eligible).clip(lower=0.0).fillna(0.0)
            if float(basis.sum()) <= 0:
                basis = original.where(eligible).fillna(0.0)
        elif mode == "equal":
            basis = pd.Series(np.where(eligible, 1.0, 0.0), index=out.index)
        elif mode == "off":
            break
        else:
            basis = original.where(eligible).fillna(0.0)
        basis_sum = float(basis.sum())
        if basis_sum <= 0:
            break
        add = basis / basis_sum * shortfall
        out = np.minimum(out + add, cap)
        out = pd.Series(out, index=scaled.index)
        eligible = eligible & (out < cap - 1e-12)
    return out.fillna(0.0)


def apply_local_risk_caps(
    target_weights: pd.DataFrame,
    score: pd.DataFrame,
    close: pd.DataFrame,
    *,
    cap_mode: str,
    stress_scale: float,
    score_quantile: float,
    recent_return_quantile: float,
    volatility_quantile: float,
    reversal_quantile: float,
    recent_return_window: int,
    volatility_window: int,
    redistribution_mode: str,
    max_weight: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    score = score.reindex(index=target_weights.index, columns=target_weights.columns).sort_index()
    state = _local_state_panels(close, score=score, recent_return_window=recent_return_window, volatility_window=volatility_window)
    mask = _risk_mask(
        state,
        cap_mode=cap_mode,
        score_quantile=score_quantile,
        recent_return_quantile=recent_return_quantile,
        volatility_quantile=volatility_quantile,
        reversal_quantile=reversal_quantile,
    ).reindex(index=target_weights.index, columns=target_weights.columns).fillna(False)
    held = target_weights.fillna(0.0) > 1e-12
    active_mask = mask & held
    scaled = target_weights.fillna(0.0).mask(active_mask, target_weights.fillna(0.0) * float(stress_scale))
    adjusted_rows = []
    for date in target_weights.index:
        adjusted_rows.append(
            _redistribute_row(
                target_weights.loc[date],
                scaled.loc[date],
                active_mask.loc[date],
                score.loc[date],
                max_weight=max_weight,
                redistribution_mode=redistribution_mode,
            )
        )
    adjusted = pd.DataFrame(adjusted_rows, index=target_weights.index, columns=target_weights.columns).fillna(0.0)
    released = (target_weights.fillna(0.0) - scaled.fillna(0.0)).clip(lower=0.0)
    meta = {
        "cap_mode": str(cap_mode),
        "stress_scale": float(stress_scale),
        "score_quantile": float(score_quantile),
        "recent_return_quantile": float(recent_return_quantile),
        "volatility_quantile": float(volatility_quantile),
        "reversal_quantile": float(reversal_quantile),
        "recent_return_window": int(recent_return_window),
        "volatility_window": int(volatility_window),
        "redistribution_mode": str(redistribution_mode),
        "active_mask_cell_count": int(active_mask.sum().sum()),
        "active_mask_cell_ratio": float(active_mask.sum().sum() / max(int(held.sum().sum()), 1)),
        "mean_source_gross": float(target_weights.fillna(0.0).sum(axis=1).mean()),
        "mean_adjusted_gross": float(adjusted.sum(axis=1).mean()),
        "mean_released_weight": float(released.sum(axis=1).mean()),
        "max_adjusted_weight": float(adjusted.max().max()) if not adjusted.empty else 0.0,
    }
    diagnostics = state_sizing._long_from_wide(active_mask.astype(float), "risk_cap_mask")
    return adjusted, meta, diagnostics


def _variants(
    *,
    cap_modes: tuple[str, ...],
    stress_scales: tuple[float, ...],
    score_quantiles: tuple[float, ...],
    recent_return_quantiles: tuple[float, ...],
    volatility_quantiles: tuple[float, ...],
    reversal_quantiles: tuple[float, ...],
    recent_return_windows: tuple[int, ...],
    volatility_windows: tuple[int, ...],
    redistribution_modes: tuple[str, ...],
    holding_count: int,
    max_weight: float,
    rebalance_freq: str,
    rebalance_offset_mode: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
) -> list[LocalRiskCapVariant]:
    out: list[LocalRiskCapVariant] = []
    for mode in cap_modes:
        for scale in stress_scales:
            for sq in score_quantiles:
                for rrq in recent_return_quantiles:
                    for vq in volatility_quantiles:
                        for rvq in reversal_quantiles:
                            for rrw in recent_return_windows:
                                for vw in volatility_windows:
                                    for redist in redistribution_modes:
                                        variant_id = (
                                            f"local_{mode}_s{scale:g}_sq{int(round(sq * 100)):02d}_"
                                            f"rq{int(round(rrq * 100)):02d}_vq{int(round(vq * 100)):02d}_"
                                            f"rvq{int(round(rvq * 100)):02d}_rrw{rrw}_vw{vw}_redist_{redist}_"
                                            f"h{holding_count}_mw{int(round(max_weight * 1000)):03d}"
                                        ).replace(".", "p")
                                        out.append(
                                            LocalRiskCapVariant(
                                                variant_id=variant_id,
                                                cap_mode=str(mode),
                                                stress_scale=float(scale),
                                                score_quantile=float(sq),
                                                recent_return_quantile=float(rrq),
                                                volatility_quantile=float(vq),
                                                reversal_quantile=float(rvq),
                                                recent_return_window=int(rrw),
                                                volatility_window=int(vw),
                                                redistribution_mode=str(redist),
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
    variant: LocalRiskCapVariant,
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
    command = state_sizing._backtest_command(
        state_sizing.StateSizingVariant(
            variant_id=variant.variant_id,
            scale_mode="local_risk_cap",
            stress_scale=variant.stress_scale,
            drawdown_threshold=0.0,
            monthly_loss_threshold=0.0,
            min_scale=0.0,
            holding_count=variant.holding_count,
            max_weight=variant.max_weight,
            rebalance_freq=variant.rebalance_freq,
            rebalance_offset_mode=variant.rebalance_offset_mode,
            transaction_cost_bps=variant.transaction_cost_bps,
            slippage_bps=variant.slippage_bps,
            sell_tax_bps=variant.sell_tax_bps,
        ),
        score_panel_csv=score_panel_csv,
        target_weight_panel_csv=target_weight_panel_csv,
        output_root=output_root,
        run_tag=run_tag,
        dataset_id=dataset_id,
        data_lake_root=data_lake_root,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
        no_cache=no_cache,
    )
    command[command.index("--candidate-label") + 1] = "daily_research_v2_local_risk_cap_research_candidate"
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
    variant: LocalRiskCapVariant,
    *,
    command: list[str],
    run_result: dict[str, Any] | None,
    cap_meta: dict[str, Any],
    target_weight_panel_csv: Path,
    risk_mask_csv: Path,
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
    item["cap_meta"] = cap_meta
    item["target_weight_panel_csv"] = str(target_weight_panel_csv)
    item["risk_mask_csv"] = str(risk_mask_csv)
    return item


def _summary_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = candidate_matrix._summary_frame(results)
    for idx, item in enumerate(results):
        meta = dict(item.get("cap_meta", {}) or {})
        for key in ["active_mask_cell_ratio", "mean_source_gross", "mean_adjusted_gross", "mean_released_weight", "max_adjusted_weight"]:
            frame.loc[idx, key] = meta.get(key)
    return frame


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = dict(report.get("summary", {}) or {})
    best = dict(summary.get("best_by_excess_sharpe", {}) or {})
    lines = [
        "# V2 Local Risk Cap Matrix",
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
        "- Local caps use historical close-derived state, source target weights, and score context; they are not production target weights.",
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


def build_local_risk_cap_matrix(
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
    cap_modes: tuple[str, ...] = ("score_volatility_or_reversal", "score_runup_or_volatility_or_reversal"),
    stress_scales: tuple[float, ...] = (0.40, 0.60),
    score_quantiles: tuple[float, ...] = (0.80,),
    recent_return_quantiles: tuple[float, ...] = (0.80,),
    volatility_quantiles: tuple[float, ...] = (0.80,),
    reversal_quantiles: tuple[float, ...] = (0.80,),
    recent_return_windows: tuple[int, ...] = (5,),
    volatility_windows: tuple[int, ...] = (20,),
    redistribution_modes: tuple[str, ...] = ("source_weight",),
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

    source_target = state_sizing._wide_from_long(source_dir / "aligned_daily_target_weight_panel.csv", "target_weight")
    source_score = state_sizing._wide_from_long(source_dir / "aligned_daily_score_panel.csv", "score")
    source_score = source_score.reindex(index=source_target.index, columns=source_target.columns)
    start_date = pd.Timestamp(source_target.index.min()).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(source_target.index.max()).strftime("%Y-%m-%d")
    source_score_path = root / "source_score_context.csv"
    state_sizing._long_from_wide(source_score, "score").to_csv(source_score_path, index=False, encoding="utf-8-sig")

    symbols = sorted(set(source_target.columns.astype(str)) | set(source_score.columns.astype(str)))
    from daily_research.path_policy import v2_bad_month_attribution as attribution

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
        cap_modes=cap_modes,
        stress_scales=stress_scales,
        score_quantiles=score_quantiles,
        recent_return_quantiles=recent_return_quantiles,
        volatility_quantiles=volatility_quantiles,
        reversal_quantiles=reversal_quantiles,
        recent_return_windows=recent_return_windows,
        volatility_windows=volatility_windows,
        redistribution_modes=redistribution_modes,
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
        adjusted, cap_meta, risk_diag = apply_local_risk_caps(
            source_target,
            source_score,
            close,
            cap_mode=variant.cap_mode,
            stress_scale=variant.stress_scale,
            score_quantile=variant.score_quantile,
            recent_return_quantile=variant.recent_return_quantile,
            volatility_quantile=variant.volatility_quantile,
            reversal_quantile=variant.reversal_quantile,
            recent_return_window=variant.recent_return_window,
            volatility_window=variant.volatility_window,
            redistribution_mode=variant.redistribution_mode,
            max_weight=variant.max_weight,
        )
        target_path = root / "target_weight_panels" / f"{variant.variant_id}.csv"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        state_sizing._long_from_wide(adjusted, "target_weight").to_csv(target_path, index=False, encoding="utf-8-sig")
        risk_path = root / "risk_masks" / f"{variant.variant_id}.csv"
        risk_path.parent.mkdir(parents=True, exist_ok=True)
        risk_diag.to_csv(risk_path, index=False, encoding="utf-8-sig")
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
        cap_meta["risk_mask_csv"] = str(risk_path)
        results.append(_result(variant, command=command, run_result=run_result, cap_meta=cap_meta, target_weight_panel_csv=target_path, risk_mask_csv=risk_path))

    summary_csv = root / "v2_local_risk_cap_matrix_summary.csv"
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
        "study_family": "v2_local_risk_cap_matrix",
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
    _write_json(root / "v2_local_risk_cap_matrix_report.json", report)
    _write_json(root / "v2_local_risk_cap_matrix_manifest.json", {k: v for k, v in report.items() if k != "results"})
    _write_markdown(root / "v2_local_risk_cap_matrix_report.md", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run research-only v2 local risk cap target-weight matrix.")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--source-matrix-run-tag", default=SOURCE_MATRIX_RUN_TAG)
    parser.add_argument("--source-variant-id", default=SOURCE_VARIANT_ID)
    parser.add_argument("--source-backtest-dir", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dataset-id", default=v2.DATASET_ID)
    parser.add_argument("--pool-view-id", default=v2.V2_STRICT_POOL_VIEW_ID)
    parser.add_argument("--data-lake-root", default=str(DATA_LAKE_ROOT))
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument("--cap-modes", default="score_volatility_or_reversal,score_runup_or_volatility_or_reversal")
    parser.add_argument("--stress-scales", default="0.40,0.60")
    parser.add_argument("--score-quantiles", default="0.80")
    parser.add_argument("--recent-return-quantiles", default="0.80")
    parser.add_argument("--volatility-quantiles", default="0.80")
    parser.add_argument("--reversal-quantiles", default="0.80")
    parser.add_argument("--recent-return-windows", default="5")
    parser.add_argument("--volatility-windows", default="20")
    parser.add_argument("--redistribution-modes", default="source_weight")
    parser.add_argument("--run-backtests", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_local_risk_cap_matrix(
        run_tag=args.run_tag,
        source_matrix_run_tag=args.source_matrix_run_tag,
        source_variant_id=args.source_variant_id,
        source_backtest_dir=args.source_backtest_dir or None,
        output_root=args.output_root or None,
        dataset_id=args.dataset_id,
        pool_view_id=args.pool_view_id,
        data_lake_root=args.data_lake_root,
        benchmark=args.benchmark,
        cap_modes=_parse_strings(args.cap_modes, ("score_volatility_or_reversal", "score_runup_or_volatility_or_reversal")),
        stress_scales=_parse_floats(args.stress_scales, (0.40, 0.60)),
        score_quantiles=_parse_floats(args.score_quantiles, (0.80,)),
        recent_return_quantiles=_parse_floats(args.recent_return_quantiles, (0.80,)),
        volatility_quantiles=_parse_floats(args.volatility_quantiles, (0.80,)),
        reversal_quantiles=_parse_floats(args.reversal_quantiles, (0.80,)),
        recent_return_windows=_parse_ints(args.recent_return_windows, (5,)),
        volatility_windows=_parse_ints(args.volatility_windows, (20,)),
        redistribution_modes=_parse_strings(args.redistribution_modes, ("source_weight",)),
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
