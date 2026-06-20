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
from daily_research.path_policy.forecast_training import (
    _ForecastDatasetView,
    _prediction_frame_for_dataset_indices,
    _predict_indices,
    forecast_prediction_metrics,
)
from daily_research.path_policy.personal_topk_diagnostics import (
    DEFAULT_HORIZONS,
    DEFAULT_SELECTION_TOP_KS,
    DEFAULT_TOP_KS,
    build_personal_topk_diagnostics_from_frame,
)


DEFAULT_ALPHA_SCORE_COLUMNS = (
    "pred_cum_mu_1d",
    "pred_cum_mu_3d",
    "pred_cum_mu_5d",
    "pred_cum_mu_10d",
    "pred_cum_mu_20d",
    "pred_aux_cum_1d",
    "pred_aux_cum_3d",
    "pred_aux_cum_5d",
    "pred_aux_cum_10d",
    "pred_aux_cum_20d",
    "pred_aux_upside_1d",
    "pred_aux_upside_3d",
    "pred_aux_upside_5d",
    "pred_aux_upside_10d",
    "pred_aux_upside_20d",
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


def _resolve_checkpoints(study_root: Path, checkpoints: str | Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in _parse_csv_strings(checkpoints):
        path = Path(item)
        if not path.is_absolute():
            path = study_root / path
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        out.append(path.resolve())
    if not out:
        raise ValueError("at least one checkpoint is required.")
    return out


def _sample_role_dates(sample_index: pd.DataFrame, *, role: str, max_dates: int) -> tuple[str, ...]:
    role_mask = sample_index["role"].astype(str).eq(str(role))
    dates = pd.to_datetime(sample_index.loc[role_mask, "date"], errors="coerce").dropna()
    unique_dates = sorted({pd.Timestamp(item).strftime("%Y-%m-%d") for item in dates.tolist()})
    if int(max_dates) > 0 and len(unique_dates) > int(max_dates):
        positions = np.linspace(0, len(unique_dates) - 1, int(max_dates), dtype=np.int64)
        unique_dates = [unique_dates[int(pos)] for pos in positions.tolist()]
    return tuple(unique_dates)


def _indices_for_dates(sample_index: pd.DataFrame, *, role: str, dates: tuple[str, ...]) -> np.ndarray:
    if not dates:
        return np.empty((0,), dtype=np.int64)
    role_values = sample_index["role"].astype(str)
    date_values = pd.to_datetime(sample_index["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    mask = role_values.eq(str(role)) & date_values.isin(set(dates))
    return sample_index.index[mask].to_numpy(dtype=np.int64)


def build_role_date_sample_diagnostics(
    *,
    manifest_json: str | Path,
    study_root: str | Path,
    checkpoints: str | Iterable[str],
    output_root: str | Path | None = None,
    role: str = "train",
    max_dates: int = 12,
    batch_size: int = 2048,
    device: str = "auto",
    amp: bool = True,
    target_scale: float = 100.0,
    score_columns: str | Iterable[str] | None = None,
    horizons: str | Iterable[int] | None = None,
    top_ks: str | Iterable[int] | None = None,
    selection_top_ks: str | Iterable[int] | None = None,
    selection_horizons: str | Iterable[int] | None = None,
    selection_min_date_count: int = 1,
    selection_profile: str = "personal_topk_v1",
    round_trip_cost_bps: float = 20.0,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
) -> dict[str, Any]:
    study_path = Path(study_root)
    root = Path(output_root) if output_root is not None else study_path / f"role_date_sample_{role}"
    root.mkdir(parents=True, exist_ok=True)

    dataset = load_forecast_memmap_dataset(manifest_json)
    if not hasattr(dataset, "sample_index"):
        raise ValueError("role date sample diagnostics require a memmap/training-pack dataset.")
    selected_dates = _sample_role_dates(dataset.sample_index, role=role, max_dates=int(max_dates))
    selected_indices = _indices_for_dates(dataset.sample_index, role=role, dates=selected_dates)
    if len(selected_indices) == 0:
        raise ValueError(f"no samples selected for role={role!r}.")

    checkpoint_paths = _resolve_checkpoints(study_path, checkpoints)
    resolved_device = _resolve_device(device)
    amp_enabled = bool(amp) and resolved_device.type == "cuda"
    resolved_horizons = _parse_csv_ints(horizons, default=DEFAULT_HORIZONS)
    resolved_top_ks = _parse_csv_ints(top_ks, default=DEFAULT_TOP_KS)
    resolved_selection_top_ks = _parse_csv_ints(selection_top_ks, default=DEFAULT_SELECTION_TOP_KS)
    resolved_selection_horizons = _parse_csv_ints(selection_horizons, default=DEFAULT_HORIZONS)
    resolved_score_columns = ",".join(_parse_csv_strings(score_columns) or DEFAULT_ALPHA_SCORE_COLUMNS)

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
        predictions = _predict_indices(
            model,
            dataset_view,
            selected_indices,
            batch_size=int(batch_size),
            device=resolved_device,
            amp_enabled=amp_enabled,
            target_scale=float(target_scale),
        )
        frame = _prediction_frame_for_dataset_indices(
            dataset,
            indices=selected_indices,
            predictions=predictions,
            family=str(checkpoint.get("model_family", "")),
            target_scale=float(target_scale),
            decision_cost_bps=float(decision_cost_bps),
            decision_hit_threshold_bps=float(decision_hit_threshold_bps),
            decision_drawdown_penalty=float(decision_drawdown_penalty),
            loss_profile=str(
                dict(checkpoint.get("training_config", {}) or {}).get(
                    "loss_profile",
                    dict(checkpoint.get("optimizer_config", {}) or {}).get("loss_profile", ""),
                )
                or ""
            ),
        )
        metrics = forecast_prediction_metrics(frame)
        report = build_personal_topk_diagnostics_from_frame(
            frame=frame,
            output_root=root / _safe_name(checkpoint_path) / str(role),
            run_tag=f"{_safe_name(checkpoint_path)}_{role}_date_sample",
            prediction_label=str(checkpoint_path.resolve()),
            score_columns=resolved_score_columns,
            horizons=resolved_horizons,
            top_ks=resolved_top_ks,
            selection_top_ks=resolved_selection_top_ks,
            selection_horizons=resolved_selection_horizons,
            selection_min_date_count=int(selection_min_date_count),
            selection_profile=str(selection_profile),
            round_trip_cost_bps=float(round_trip_cost_bps),
        )
        selected = dict(report.get("selected_candidate", {}) or {})
        rows.append(
            {
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_epoch": int(checkpoint.get("epoch", 0) or 0),
                "checkpoint_kind": str(checkpoint.get("checkpoint_kind", "")),
                "role": str(role),
                "sample_date_count": int(len(selected_dates)),
                "sample_row_count": int(len(selected_indices)),
                "sample_start_date": selected_dates[0] if selected_dates else "",
                "sample_end_date": selected_dates[-1] if selected_dates else "",
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
                "selection_profile": selected.get("personal_selection_profile", str(selection_profile)),
                "topk_report_json": report.get("outputs", {}).get("report_json", ""),
                "topk_summary_csv": report.get("outputs", {}).get("summary_csv", ""),
            }
        )
        del model, predictions, frame
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["personal_selection_score", "selected_net_mean", "rank_ic_20d"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    summary_csv = _write_frame(root / f"role_date_sample_{role}_summary.csv", summary)
    output = {
        "status": "completed" if not summary.empty else "empty",
        "study_root": str(study_path.resolve()),
        "manifest_json": str(Path(manifest_json).resolve()),
        "output_root": str(root.resolve()),
        "role": str(role),
        "selected_dates": list(selected_dates),
        "sample_row_count": int(len(selected_indices)),
        "checkpoint_count": int(len(checkpoint_paths)),
        "contract": {
            "sample_mode": "complete_role_dates_evenly_spaced",
            "selection_profile": str(selection_profile),
            "max_dates": int(max_dates),
            "selection_top_ks": [int(item) for item in resolved_selection_top_ks],
            "selection_horizons": [int(item) for item in resolved_selection_horizons],
            "not_a_backtest": True,
        },
        "outputs": {"summary_csv": summary_csv},
        "selected_checkpoint": dict(summary.iloc[0].to_dict()) if not summary.empty else {},
        "rows": [dict(row) for row in summary.to_dict(orient="records")],
    }
    report_json = _write_json(root / f"role_date_sample_{role}_summary.json", output)
    output["outputs"]["report_json"] = report_json
    _write_json(root / f"role_date_sample_{role}_summary.json", output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run compact complete-date forecast diagnostics for one role.")
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--role", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--max-dates", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--target-scale", type=float, default=100.0)
    parser.add_argument("--score-columns", default="")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--top-ks", default="1,3,5,10,20")
    parser.add_argument("--selection-top-ks", default="1,3,5")
    parser.add_argument("--selection-horizons", default="1,3,5,10,20")
    parser.add_argument("--selection-min-date-count", type=int, default=1)
    parser.add_argument("--selection-profile", default="personal_topk_v1")
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--decision-cost-bps", type=float, default=20.0)
    parser.add_argument("--decision-hit-threshold-bps", type=float, default=20.0)
    parser.add_argument("--decision-drawdown-penalty", type=float, default=0.25)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_role_date_sample_diagnostics(
        manifest_json=args.manifest_json,
        study_root=args.study_root,
        checkpoints=args.checkpoints,
        output_root=args.output_root or None,
        role=args.role,
        max_dates=int(args.max_dates),
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
        decision_cost_bps=float(args.decision_cost_bps),
        decision_hit_threshold_bps=float(args.decision_hit_threshold_bps),
        decision_drawdown_penalty=float(args.decision_drawdown_penalty),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(report.get("outputs", {}).get("report_json", ""))
    return report


if __name__ == "__main__":
    main()
