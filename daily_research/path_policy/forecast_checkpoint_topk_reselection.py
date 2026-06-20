from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from daily_research.path_policy.forecast_dataset import load_forecast_memmap_dataset
from daily_research.path_policy.forecast_training import (
    _ForecastDatasetView,
    _prediction_frame_for_dataset_indices,
    _predict_indices,
    forecast_prediction_metrics,
    make_forecast_model,
)
from daily_research.path_policy.personal_topk_diagnostics import (
    DEFAULT_HORIZONS,
    DEFAULT_SCORE_COLUMNS,
    DEFAULT_SELECTION_HORIZONS,
    DEFAULT_SELECTION_TOP_KS,
    DEFAULT_TOP_KS,
    build_personal_topk_diagnostics_from_frame,
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


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return str(path.resolve())


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _parse_csv_strings(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _parse_csv_ints(raw: str | Iterable[int] | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = [int(item) for item in raw]
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out or default)


def _resolve_device(value: str) -> torch.device:
    raw = str(value or "auto").strip().lower()
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    epoch_match = re.search(r"_epoch(\d+)\.pt$", name)
    if epoch_match:
        return (0, int(epoch_match.group(1)), name)
    if "_best.pt" in name:
        return (1, 0, name)
    if "_last.pt" in name:
        return (2, 0, name)
    return (3, 0, name)


def _discover_checkpoints(study_root: Path, checkpoint_globs: tuple[str, ...], max_checkpoints: int = 0) -> list[Path]:
    paths: list[Path] = []
    for pattern in checkpoint_globs:
        for path in study_root.glob(pattern):
            if path.is_file():
                paths.append(path.resolve())
    unique = sorted(set(paths), key=_checkpoint_sort_key)
    if int(max_checkpoints) > 0:
        unique = unique[: int(max_checkpoints)]
    return unique


def _load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"forecast checkpoint payload must be a dictionary: {path}")
    if "state_dict" not in payload:
        raise ValueError(f"forecast checkpoint missing state_dict: {path}")
    return payload


def _model_from_checkpoint(checkpoint: dict[str, Any], dataset_view: _ForecastDatasetView) -> torch.nn.Module:
    model_config = dict(checkpoint.get("model_config", {}) or {})
    training_config = dict(checkpoint.get("training_config", {}) or {})
    family = str(checkpoint.get("model_family", model_config.get("model_family", "")) or "")
    if not family:
        raise ValueError("forecast checkpoint missing model_family.")
    checkpoint_features = [str(item) for item in list(checkpoint.get("feature_columns", []) or [])]
    if checkpoint_features and checkpoint_features != list(dataset_view.feature_columns):
        raise ValueError("forecast checkpoint feature_columns do not match the dataset.")
    model = make_forecast_model(
        family,
        input_dim=int(dataset_view.input_dim),
        hidden_dim=int(model_config.get("hidden_dim", training_config.get("hidden_dim", 192)) or 192),
        horizon=int(dataset_view.horizon),
        dropout=float(model_config.get("dropout", training_config.get("dropout", 0.15)) or 0.15),
        gru_layers=int(model_config.get("gru_layers", training_config.get("gru_layers", 2)) or 2),
        transformer_layers=int(model_config.get("transformer_layers", training_config.get("transformer_layers", 4)) or 4),
        transformer_heads=int(model_config.get("transformer_heads", training_config.get("transformer_heads", 6)) or 6),
        patch_sizes=tuple(int(item) for item in list(model_config.get("patch_sizes", training_config.get("patch_sizes", [4, 20])) or [4, 20])),
        cumulative_horizons=dataset_view.cumulative_horizons,
        output_profile=str(model_config.get("output_profile", training_config.get("output_profile", "forecast_path_v1")) or "forecast_path_v1"),
        static_context_vocab_sizes=dict(model_config.get("static_context_vocab_sizes", {}) or {}),
        static_context_embedding_dims=dict(model_config.get("static_context_embedding_dims", {}) or {}),
        static_context_fields=tuple(str(item) for item in list(model_config.get("static_context_fields", []) or [])),
        static_context_dropout=float(model_config.get("static_context_dropout", 0.20) or 0.20),
        slot_count=int(model_config.get("slot_count", 8) or 8),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _safe_name(path: Path) -> str:
    return path.stem.replace("forecast_model_", "").replace("_", "-")


def build_checkpoint_topk_reselection(
    *,
    manifest_json: str | Path,
    study_root: str | Path,
    output_root: str | Path | None = None,
    checkpoint_globs: str | Iterable[str] | None = None,
    checkpoints: str | Iterable[str] | None = None,
    role: str = "validation",
    batch_size: int = 512,
    device: str = "auto",
    amp: bool = True,
    target_scale: float = 100.0,
    score_columns: str | Iterable[str] | None = None,
    horizons: str | Iterable[int] | None = None,
    top_ks: str | Iterable[int] | None = None,
    selection_top_ks: str | Iterable[int] | None = None,
    selection_horizons: str | Iterable[int] | None = None,
    selection_min_date_count: int = 20,
    selection_profile: str = "personal_topk_v1",
    round_trip_cost_bps: float = 20.0,
    decision_cost_bps: float = 20.0,
    decision_hit_threshold_bps: float = 20.0,
    decision_drawdown_penalty: float = 0.25,
    max_checkpoints: int = 0,
    max_samples_per_role: int = 0,
) -> dict[str, Any]:
    study_path = Path(study_root)
    root = Path(output_root) if output_root is not None else study_path / "personal_checkpoint_reselection"
    root.mkdir(parents=True, exist_ok=True)
    explicit = [Path(item) for item in _parse_csv_strings(checkpoints)]
    if explicit:
        checkpoint_paths = [path if path.is_absolute() else study_path / path for path in explicit]
        checkpoint_paths = [path.resolve() for path in checkpoint_paths if path.exists()]
    else:
        patterns = _parse_csv_strings(checkpoint_globs) or ("forecast_model_*_epoch*.pt",)
        checkpoint_paths = _discover_checkpoints(study_path, patterns, max_checkpoints=int(max_checkpoints))
    if not checkpoint_paths:
        raise ValueError(f"no checkpoints found under {study_path}")

    resolved_device = _resolve_device(device)
    amp_enabled = bool(amp) and resolved_device.type == "cuda"
    dataset = load_forecast_memmap_dataset(manifest_json, max_samples_per_role=int(max_samples_per_role))
    base_dataset_view = _ForecastDatasetView(dataset)
    base_role_indices = base_dataset_view.role_indices(role)
    if len(base_role_indices) == 0:
        raise ValueError(f"dataset has no samples for role={role!r}")

    resolved_horizons = _parse_csv_ints(horizons, default=DEFAULT_HORIZONS)
    resolved_top_ks = _parse_csv_ints(top_ks, default=DEFAULT_TOP_KS)
    resolved_selection_top_ks = _parse_csv_ints(selection_top_ks, default=DEFAULT_SELECTION_TOP_KS)
    resolved_selection_horizons = _parse_csv_ints(selection_horizons, default=DEFAULT_SELECTION_HORIZONS)
    resolved_score_columns = ",".join(_parse_csv_strings(score_columns) or DEFAULT_SCORE_COLUMNS)

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
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
        role_indices = dataset_view.role_indices(role)
        model = _model_from_checkpoint(checkpoint, dataset_view).to(resolved_device)
        predictions = _predict_indices(
            model,
            dataset_view,
            role_indices,
            batch_size=int(batch_size),
            device=resolved_device,
            amp_enabled=amp_enabled,
            target_scale=float(target_scale),
        )
        frame = _prediction_frame_for_dataset_indices(
            dataset,
            indices=role_indices,
            predictions=predictions,
            family=str(checkpoint.get("model_family", "")),
            target_scale=float(target_scale),
            decision_cost_bps=float(decision_cost_bps),
            decision_hit_threshold_bps=float(decision_hit_threshold_bps),
            decision_drawdown_penalty=float(decision_drawdown_penalty),
            loss_profile=str(checkpoint.get("training_config", {}).get("loss_profile", checkpoint.get("optimizer_config", {}).get("loss_profile", "")) or ""),
        )
        metrics = forecast_prediction_metrics(frame)
        report = build_personal_topk_diagnostics_from_frame(
            frame=frame,
            output_root=root / _safe_name(checkpoint_path) / str(role),
            run_tag=f"{_safe_name(checkpoint_path)}_{role}",
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
        row = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_kind": str(checkpoint.get("checkpoint_kind", "")),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0) or 0),
            "role": str(role),
            "model_family": str(checkpoint.get("model_family", "")),
            "personal_selection_score": selected.get("personal_selection_score"),
            "selected_score_column": selected.get("score_column", ""),
            "selected_top_k": selected.get("top_k", 0),
            "selected_horizon": selected.get("horizon", 0),
            "selected_net_mean": selected.get("net_mean"),
            "selected_net_mean_per_day": selected.get("net_mean_per_day"),
            "selected_hit_rate_mean": selected.get("hit_rate_mean"),
            "selected_positive_month_rate": selected.get("positive_month_rate"),
            "selection_profile": selected.get("personal_selection_profile", str(selection_profile)),
            "rank_ic_20d": metrics.get("rank_ic_20d"),
            "top_bottom_spread_20d": metrics.get("top_bottom_spread_20d"),
            "rank_ic_upside_20d": metrics.get("rank_ic_upside_20d"),
            "decision_score_rank_ic": metrics.get("decision_score_rank_ic"),
            "topk_report_json": report.get("outputs", {}).get("report_json", ""),
            "topk_summary_csv": report.get("outputs", {}).get("summary_csv", ""),
        }
        rows.append(row)
        reports.append(report)
        del model, predictions, frame
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["personal_selection_score", "selected_net_mean", "rank_ic_20d"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    summary_csv = _write_frame(root / f"checkpoint_personal_topk_reselection_{role}.csv", summary)
    selected_row = dict(summary.iloc[0].to_dict()) if not summary.empty else {}
    output = {
        "status": "completed" if not summary.empty else "empty",
        "study_root": str(study_path.resolve()),
        "manifest_json": str(Path(manifest_json).resolve()),
        "output_root": str(root.resolve()),
        "role": str(role),
        "checkpoint_count": int(len(checkpoint_paths)),
        "contract": {
            "selection_profile": str(selection_profile),
            "not_a_backtest": True,
            "max_samples_per_role": int(max_samples_per_role),
            "selection_top_ks": [int(item) for item in resolved_selection_top_ks],
            "selection_horizons": [int(item) for item in resolved_selection_horizons],
            "selection_min_date_count": int(selection_min_date_count),
        },
        "outputs": {
            "summary_csv": summary_csv,
        },
        "selected_checkpoint": selected_row,
        "rows": [dict(row) for row in summary.to_dict(orient="records")],
    }
    report_json = _write_json(root / f"checkpoint_personal_topk_reselection_{role}.json", output)
    output["outputs"]["report_json"] = report_json
    _write_json(root / f"checkpoint_personal_topk_reselection_{role}.json", output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reselect forecast checkpoints with personal small-capital top-K validation diagnostics.")
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--checkpoint-globs", default="forecast_model_*_epoch*.pt")
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--role", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--target-scale", type=float, default=100.0)
    parser.add_argument("--score-columns", default="")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--top-ks", default="1,3,5,10,20")
    parser.add_argument("--selection-top-ks", default="1,3,5")
    parser.add_argument("--selection-horizons", default="5,10,20")
    parser.add_argument("--selection-min-date-count", type=int, default=20)
    parser.add_argument("--selection-profile", default="personal_topk_v1")
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--decision-cost-bps", type=float, default=20.0)
    parser.add_argument("--decision-hit-threshold-bps", type=float, default=20.0)
    parser.add_argument("--decision-drawdown-penalty", type=float, default=0.25)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--max-samples-per-role", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_checkpoint_topk_reselection(
        manifest_json=args.manifest_json,
        study_root=args.study_root,
        output_root=args.output_root or None,
        checkpoint_globs=args.checkpoint_globs,
        checkpoints=args.checkpoints or None,
        role=args.role,
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
        max_checkpoints=int(args.max_checkpoints),
        max_samples_per_role=int(args.max_samples_per_role),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(report["outputs"]["report_json"])
    return report


if __name__ == "__main__":
    main()
