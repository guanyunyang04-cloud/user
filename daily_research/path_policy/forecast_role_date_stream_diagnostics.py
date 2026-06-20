from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from daily_research.path_policy.forecast_checkpoint_topk_reselection import (
    _load_checkpoint,
    _model_from_checkpoint,
    _parse_csv_ints,
    _parse_csv_strings,
    _resolve_device,
    _safe_name,
    _write_frame,
    _write_json,
)
from daily_research.path_policy.forecast_dataset import load_forecast_memmap_dataset
from daily_research.path_policy.forecast_role_date_sample_diagnostics import (
    DEFAULT_ALPHA_SCORE_COLUMNS,
    _indices_for_dates,
    _resolve_checkpoints,
    _sample_role_dates,
)
from daily_research.path_policy.forecast_training import (
    _ForecastDatasetView,
    _predict_indices,
    forecast_prediction_metrics,
)
from daily_research.path_policy.personal_topk_diagnostics import (
    DEFAULT_HORIZONS,
    DEFAULT_SELECTION_TOP_KS,
    DEFAULT_TOP_KS,
    _add_personal_selection_scores,
    _build_daily_rows,
    _leaderboard,
    _selection_leaderboard,
    _summarize_daily,
)


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
    return value


def _chunked(items: tuple[str, ...], size: int) -> list[tuple[str, ...]]:
    chunk_size = max(int(size), 1)
    return [tuple(items[pos : pos + chunk_size]) for pos in range(0, len(items), chunk_size)]


def _light_prediction_frame_for_dataset_indices(
    dataset: Any,
    *,
    indices: np.ndarray,
    predictions: dict[str, np.ndarray],
    family: str,
    target_scale: float,
) -> pd.DataFrame:
    idx = np.asarray(indices, dtype=int)
    rows = dataset.sample_index.iloc[idx].reset_index(drop=True)
    scale = float(target_scale)
    mu = predictions["mu"] / scale
    aux = predictions["aux"] / scale
    horizons = tuple(int(item) for item in dataset.cumulative_horizons)
    y_cum = dataset.y_cum_excess[idx]
    columns: dict[str, Any] = {
        "date": rows["date"].astype(str).tolist(),
        "stock": rows["stock"].astype(str).to_numpy(),
        "role": rows["role"].astype(str).to_numpy(),
        "model_family": str(family),
    }
    for pos, item_horizon in enumerate(horizons):
        horizon = int(item_horizon)
        columns[f"future_cum_excess_return_{horizon}d"] = y_cum[:, pos]
        columns[f"future_rank_{horizon}d"] = dataset.y_rank_by_horizon[idx, pos]
        columns[f"future_path_max_drawdown_{horizon}d"] = dataset.y_drawdown_by_horizon[idx, pos]
        columns[f"future_path_worst_1d_{horizon}d"] = dataset.y_worst_by_horizon[idx, pos]
        columns[f"future_path_upside_capture_{horizon}d"] = dataset.y_upside_by_horizon[idx, pos]
        columns[f"pred_cum_mu_{horizon}d"] = mu[:, :horizon].sum(axis=1)
        columns[f"pred_aux_cum_{horizon}d"] = aux[:, pos]
    risk_start = len(horizons)
    risk_aux = aux[:, risk_start : risk_start + len(horizons) * 3].reshape(-1, len(horizons), 3)
    for pos, item_horizon in enumerate(horizons):
        horizon = int(item_horizon)
        columns[f"pred_aux_downside_floor_{horizon}d"] = risk_aux[:, pos, 0]
        columns[f"pred_aux_worst_1d_{horizon}d"] = risk_aux[:, pos, 1]
        columns[f"pred_aux_upside_{horizon}d"] = risk_aux[:, pos, 2]
    return pd.DataFrame(columns)


def _aggregate_metric_rows(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_rows:
        return {"status": "insufficient_or_incomplete", "reason": "empty_metric_rows"}
    total_rows = sum(int(row.get("row_count", 0) or 0) for row in metric_rows)
    total_dates = sum(int(row.get("date_count", 0) or 0) for row in metric_rows)
    out: dict[str, Any] = {
        "status": "completed",
        "row_count": int(total_rows),
        "date_count": int(total_dates),
    }
    keys = sorted({key for row in metric_rows for key in row.keys()})
    for key in keys:
        if key in {"status", "row_count", "date_count", "forecast_horizon", "cumulative_horizons", "selected_signal_profile"}:
            continue
        values: list[tuple[float, int]] = []
        for row in metric_rows:
            value = row.get(key)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append((numeric, int(row.get("date_count", 0) or 0)))
        if values:
            weight_sum = sum(weight for _, weight in values)
            out[key] = (
                float(sum(value * weight for value, weight in values) / weight_sum)
                if weight_sum > 0
                else float(np.mean([value for value, _ in values]))
            )
    first = metric_rows[0]
    out["forecast_horizon"] = int(first.get("forecast_horizon", 0) or 0)
    out["cumulative_horizons"] = list(first.get("cumulative_horizons", []) or [])
    return out


def build_role_date_stream_diagnostics(
    *,
    manifest_json: str | Path,
    study_root: str | Path,
    checkpoints: str | Iterable[str],
    output_root: str | Path | None = None,
    role: str = "train",
    max_dates: int = 0,
    date_chunk_size: int = 96,
    batch_size: int = 8192,
    device: str = "auto",
    amp: bool = True,
    target_scale: float = 100.0,
    score_columns: str | Iterable[str] | None = None,
    horizons: str | Iterable[int] | None = None,
    top_ks: str | Iterable[int] | None = None,
    selection_top_ks: str | Iterable[int] | None = None,
    selection_horizons: str | Iterable[int] | None = None,
    selection_min_date_count: int = 20,
    selection_profile: str = "personal_time_efficiency_v2",
    round_trip_cost_bps: float = 20.0,
    write_daily: bool = True,
) -> dict[str, Any]:
    study_path = Path(study_root)
    root = Path(output_root) if output_root is not None else study_path / f"role_date_stream_{role}"
    root.mkdir(parents=True, exist_ok=True)

    dataset = load_forecast_memmap_dataset(manifest_json)
    if not hasattr(dataset, "sample_index"):
        raise ValueError("role date stream diagnostics require a memmap/training-pack dataset.")
    selected_dates = _sample_role_dates(dataset.sample_index, role=role, max_dates=int(max_dates))
    if not selected_dates:
        raise ValueError(f"no dates selected for role={role!r}.")

    checkpoint_paths = _resolve_checkpoints(study_path, checkpoints)
    resolved_device = _resolve_device(device)
    amp_enabled = bool(amp) and resolved_device.type == "cuda"
    resolved_horizons = _parse_csv_ints(horizons, default=DEFAULT_HORIZONS)
    resolved_top_ks = _parse_csv_ints(top_ks, default=DEFAULT_TOP_KS)
    resolved_selection_top_ks = _parse_csv_ints(selection_top_ks, default=DEFAULT_SELECTION_TOP_KS)
    resolved_selection_horizons = _parse_csv_ints(selection_horizons, default=DEFAULT_HORIZONS)
    resolved_score_columns = _parse_csv_strings(score_columns) or DEFAULT_ALPHA_SCORE_COLUMNS
    date_chunks = _chunked(selected_dates, int(date_chunk_size))

    rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = _load_checkpoint(checkpoint_path)
        checkpoint_static_fields = tuple(
            str(item)
            for item in list(
                dict(checkpoint.get("model_config", {}) or {}).get("static_context_fields", []) or []
            )
        )
        dataset_view = _ForecastDatasetView(
            dataset,
            static_context_fields_override=checkpoint_static_fields or None,
        )
        model = _model_from_checkpoint(checkpoint, dataset_view).to(resolved_device)
        chunk_daily: list[pd.DataFrame] = []
        chunk_metrics: list[dict[str, Any]] = []
        checkpoint_root = root / _safe_name(checkpoint_path) / str(role)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        progress_path = checkpoint_root / "stream_progress.json"
        processed_rows = 0
        for chunk_idx, chunk_dates in enumerate(date_chunks, start=1):
            chunk_indices = _indices_for_dates(dataset.sample_index, role=role, dates=chunk_dates)
            if len(chunk_indices) == 0:
                continue
            predictions = _predict_indices(
                model,
                dataset_view,
                chunk_indices,
                batch_size=int(batch_size),
                device=resolved_device,
                amp_enabled=amp_enabled,
                target_scale=float(target_scale),
            )
            frame = _light_prediction_frame_for_dataset_indices(
                dataset,
                indices=chunk_indices,
                predictions=predictions,
                family=str(checkpoint.get("model_family", "")),
                target_scale=float(target_scale),
            )
            chunk_metrics.append(forecast_prediction_metrics(frame))
            daily = _build_daily_rows(
                frame,
                score_columns=tuple(str(item) for item in resolved_score_columns),
                horizons=resolved_horizons,
                top_ks=resolved_top_ks,
                round_trip_cost_bps=float(round_trip_cost_bps),
            )
            if not daily.empty:
                chunk_daily.append(daily)
            processed_rows += int(len(chunk_indices))
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "checkpoint": str(checkpoint_path.resolve()),
                    "role": str(role),
                    "chunk": int(chunk_idx),
                    "chunk_count": int(len(date_chunks)),
                    "processed_rows": int(processed_rows),
                    "processed_dates": int(sum(len(item) for item in date_chunks[:chunk_idx])),
                    "total_dates": int(len(selected_dates)),
                    "last_chunk_start": chunk_dates[0],
                    "last_chunk_end": chunk_dates[-1],
                },
            )
            del predictions, frame, daily
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        daily_all = pd.concat(chunk_daily, ignore_index=True) if chunk_daily else pd.DataFrame()
        summary = _add_personal_selection_scores(
            _summarize_daily(daily_all),
            selection_top_ks=resolved_selection_top_ks,
            selection_horizons=resolved_selection_horizons,
            min_date_count=int(selection_min_date_count),
            selection_profile=str(selection_profile),
        )
        summary_csv = _write_frame(checkpoint_root / "personal_topk_summary.csv", summary)
        daily_csv = ""
        if bool(write_daily):
            daily_csv = _write_frame(checkpoint_root / "personal_topk_daily.csv", daily_all)
        selection_leaderboard = _selection_leaderboard(summary, limit=20)
        selected = dict(selection_leaderboard[0]) if selection_leaderboard else {}
        metrics = _aggregate_metric_rows(chunk_metrics)
        report = {
            "status": "completed" if not summary.empty else "empty",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_epoch": int(checkpoint.get("epoch", 0) or 0),
            "checkpoint_kind": str(checkpoint.get("checkpoint_kind", "")),
            "role": str(role),
            "date_count": int(len(selected_dates)),
            "row_count": int(processed_rows),
            "start_date": selected_dates[0],
            "end_date": selected_dates[-1],
            "metrics": metrics,
            "contract": {
                "sample_mode": "complete_role_dates_stream",
                "max_dates": int(max_dates),
                "date_chunk_size": int(date_chunk_size),
                "selection_profile": str(selection_profile),
                "selection_top_ks": [int(item) for item in resolved_selection_top_ks],
                "selection_horizons": [int(item) for item in resolved_selection_horizons],
                "selection_min_date_count": int(selection_min_date_count),
                "not_a_backtest": True,
            },
            "outputs": {
                "summary_csv": summary_csv,
                "daily_csv": daily_csv,
            },
            "leaderboard": _leaderboard(summary, limit=20),
            "selection_leaderboard": selection_leaderboard,
            "selected_candidate": selected,
        }
        report_json = _write_json(checkpoint_root / "personal_topk_report.json", report)
        report["outputs"]["report_json"] = report_json
        _write_json(checkpoint_root / "personal_topk_report.json", report)
        _write_json(progress_path, {"status": "completed", "report_json": report_json})
        rows.append(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_epoch": int(checkpoint.get("epoch", 0) or 0),
                "checkpoint_kind": str(checkpoint.get("checkpoint_kind", "")),
                "role": str(role),
                "date_count": int(len(selected_dates)),
                "row_count": int(processed_rows),
                "rank_ic_1d": metrics.get("rank_ic_1d"),
                "rank_ic_3d": metrics.get("rank_ic_3d"),
                "rank_ic_5d": metrics.get("rank_ic_5d"),
                "rank_ic_10d": metrics.get("rank_ic_10d"),
                "rank_ic_20d": metrics.get("rank_ic_20d"),
                "top_bottom_spread_1d": metrics.get("top_bottom_spread_1d"),
                "top_bottom_spread_5d": metrics.get("top_bottom_spread_5d"),
                "top_bottom_spread_20d": metrics.get("top_bottom_spread_20d"),
                "personal_selection_score": selected.get("personal_selection_score"),
                "selected_score_column": selected.get("score_column", ""),
                "selected_top_k": selected.get("top_k", 0),
                "selected_horizon": selected.get("horizon", 0),
                "selected_net_mean": selected.get("net_mean"),
                "selected_net_mean_per_day": selected.get("net_mean_per_day"),
                "selected_excess_vs_all_mean": selected.get("excess_vs_all_mean"),
                "selected_excess_vs_all_mean_per_day": selected.get("excess_vs_all_mean_per_day"),
                "selected_hit_rate_mean": selected.get("hit_rate_mean"),
                "selected_positive_month_rate": selected.get("positive_month_rate"),
                "selected_worst_month_mean": selected.get("worst_month_mean"),
                "selection_profile": selected.get("personal_selection_profile", str(selection_profile)),
                "topk_report_json": report_json,
                "topk_summary_csv": summary_csv,
            }
        )
        del model, chunk_daily, daily_all, summary
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_summary = pd.DataFrame(rows)
    if not run_summary.empty:
        run_summary = run_summary.sort_values(
            ["personal_selection_score", "selected_net_mean_per_day", "rank_ic_20d"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    summary_csv = _write_frame(root / f"role_date_stream_{role}_summary.csv", run_summary)
    output = {
        "status": "completed" if not run_summary.empty else "empty",
        "study_root": str(study_path.resolve()),
        "manifest_json": str(Path(manifest_json).resolve()),
        "output_root": str(root.resolve()),
        "role": str(role),
        "date_count": int(len(selected_dates)),
        "checkpoint_count": int(len(checkpoint_paths)),
        "contract": {
            "sample_mode": "complete_role_dates_stream",
            "max_dates": int(max_dates),
            "date_chunk_size": int(date_chunk_size),
            "selection_profile": str(selection_profile),
            "not_a_backtest": True,
        },
        "outputs": {"summary_csv": summary_csv},
        "selected_checkpoint": dict(run_summary.iloc[0].to_dict()) if not run_summary.empty else {},
        "rows": [dict(row) for row in run_summary.to_dict(orient="records")],
    }
    report_json = _write_json(root / f"role_date_stream_{role}_summary.json", output)
    output["outputs"]["report_json"] = report_json
    _write_json(root / f"role_date_stream_{role}_summary.json", output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run streaming complete-date forecast diagnostics for one role.")
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--role", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--max-dates", type=int, default=0)
    parser.add_argument("--date-chunk-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--target-scale", type=float, default=100.0)
    parser.add_argument("--score-columns", default="")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--top-ks", default="1,3,5,10,20")
    parser.add_argument("--selection-top-ks", default="1,3,5")
    parser.add_argument("--selection-horizons", default="1,3,5,10,20")
    parser.add_argument("--selection-min-date-count", type=int, default=20)
    parser.add_argument("--selection-profile", default="personal_time_efficiency_v2")
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--write-daily", dest="write_daily", action="store_true", default=True)
    parser.add_argument("--no-write-daily", dest="write_daily", action="store_false")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_role_date_stream_diagnostics(
        manifest_json=args.manifest_json,
        study_root=args.study_root,
        checkpoints=args.checkpoints,
        output_root=args.output_root or None,
        role=args.role,
        max_dates=int(args.max_dates),
        date_chunk_size=int(args.date_chunk_size),
        batch_size=int(args.batch_size),
        device=args.device,
        amp=bool(args.amp),
        target_scale=float(args.target_scale),
        score_columns=args.score_columns or None,
        horizons=args.horizons,
        top_ks=args.top_ks,
        selection_top_ks=args.selection_top_ks,
        selection_horizons=args.selection_horizons,
        selection_min_date_count=int(args.selection_min_date_count),
        selection_profile=args.selection_profile,
        round_trip_cost_bps=float(args.round_trip_cost_bps),
        write_daily=bool(args.write_daily),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(report.get("outputs", {}).get("report_json", ""))
    return report


if __name__ == "__main__":
    main()
