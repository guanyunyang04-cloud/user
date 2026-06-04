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

from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy import v2_local_state_input_scout as local_state
from daily_research.path_policy.forecast_training import forecast_loss_profile_contract, make_forecast_model
from daily_research.path_policy.output_aux_profile_comparison import (
    build_output_aux_profile_comparison,
    profile_aggregate_rows,
    score_variant_comparison_rows,
    write_output_aux_profile_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
RUN_TAG = "v2_arch_fusion_scout_20260603_01"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
DEFAULT_SCORE_COLUMN = "pred_decision_score"
MODEL_FAMILY = "hybrid_expert_fusion_static_context"
TIER2_MODEL_FAMILY = "regime_routed_multi_expert_horizon_v1"
SUPPORTED_MODEL_FAMILIES = (MODEL_FAMILY, TIER2_MODEL_FAMILY)
FEATURE_PROFILE = local_state.FEATURE_PROFILE
OUTPUT_PROFILE = v2.OUTPUT_PROFILE
SELECTION_PROFILE = v2.SELECTION_PROFILE
HORIZON_GRID = v2.HORIZON_GRID
LOSS_PROFILE = "horizon_30d_soft_penalty_v1"
DEFAULT_SEEDS = (7,)
CONFIRM_SEEDS = (7, 11, 19)
DEFAULT_EXPERTS = (
    "gru_sequence_static_context",
    "patch_transformer_static_context",
    "dlinear_sequence",
)
MODEL_FAMILY_CONFIGS: dict[str, dict[str, Any]] = {
    MODEL_FAMILY: {
        "tier": "tier1_hybrid_expert_fusion",
        "fusion_version": "hybrid_expert_fusion_static_context_v1",
        "study_tag_prefix": "mh_v2_arch_fusion_hybrid_expert",
        "expert_families": list(DEFAULT_EXPERTS),
        "forecast_hidden_dim": 192,
        "forecast_gru_layers": 2,
        "forecast_transformer_layers": 4,
        "forecast_transformer_heads": 6,
        "forecast_patch_sizes": "4,20",
        "target_parameter_band": "4m_to_5m",
    },
    TIER2_MODEL_FAMILY: {
        "tier": "tier2_regime_routed_multi_expert",
        "fusion_version": "regime_routed_multi_expert_horizon_v1",
        "study_tag_prefix": "mh_v2_arch_fusion_regime_routed_multi_expert",
        "expert_families": [
            "gru_path_continuity",
            "patch_transformer_multiscale_events",
            "dlinear_low_frequency_trend",
            "stock_mixer_cross_section",
            "local_state_volatility_reversal",
        ],
        "forecast_hidden_dim": 256,
        "forecast_gru_layers": 2,
        "forecast_transformer_layers": 4,
        "forecast_transformer_heads": 8,
        "forecast_patch_sizes": "4,10,20",
        "target_parameter_band": "8m_to_15m",
        "router_diagnostics": [
            "expert_weight_by_month",
            "expert_weight_by_regime",
            "bad_month_expert_attribution",
            "router_entropy",
            "high_volatility_reversal_expert_weights",
            "expert_collapse_check",
        ],
    },
}


@dataclass(frozen=True)
class FusionPaths:
    output_root: Path
    validation_panel_csv: Path
    test_panel_csv: Path
    weight_report_json: Path
    report_json: Path
    report_md: Path


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


def _parse_csv_values(raw: str | tuple[str, ...] | list[str], *, default: tuple[str, ...]) -> tuple[str, ...]:
    values = list(raw) if isinstance(raw, (tuple, list)) else str(raw or "").split(",")
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out or default)


def _parse_seeds(raw: str | tuple[int, ...] | list[int] | None, default: tuple[int, ...] = DEFAULT_SEEDS) -> tuple[int, ...]:
    if raw is None:
        return tuple(int(seed) for seed in default)
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        values = [str(item) for item in raw]
    parsed = tuple(int(float(item)) for item in values)
    return parsed or tuple(int(seed) for seed in default)


def _paths(output_root: str | Path | None = None) -> FusionPaths:
    root = Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG
    return FusionPaths(
        output_root=root,
        validation_panel_csv=root / "fusion_score_panel_validation.csv",
        test_panel_csv=root / "fusion_score_panel_test.csv",
        weight_report_json=root / "fusion_weight_report.json",
        report_json=root / "v2_arch_fusion_scout_report.json",
        report_md=root / "v2_arch_fusion_scout_report.md",
    )


def _anchor_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def _model_family_config(model_family: str | None = None) -> dict[str, Any]:
    family = str(model_family or MODEL_FAMILY).strip()
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported arch fusion model_family: {family}")
    return dict(MODEL_FAMILY_CONFIGS[family])


def _study_tag(seed: int, *, model_family: str | None = None) -> str:
    config = _model_family_config(model_family)
    return f"{config['study_tag_prefix']}_seed{int(seed)}_20260603_01"


def _seed7_manifest_path() -> Path:
    return STUDIES_ROOT / local_state._study_tag(7) / "forecast_dataset_manifest.json"


def _loss_contract() -> dict[str, Any]:
    contract = forecast_loss_profile_contract(LOSS_PROFILE, cumulative_horizons=HORIZON_GRID, forecast_horizon=30)
    if str(contract.get("required_output_profile", "")) != OUTPUT_PROFILE:
        raise ValueError(f"loss_profile_output_mismatch: {LOSS_PROFILE} requires {contract.get('required_output_profile')}")
    return contract


def _training_command(*, seed: int, model_family: str | None = None) -> list[str]:
    family = str(model_family or MODEL_FAMILY).strip()
    config = _model_family_config(family)
    tag = _study_tag(seed, model_family=family)
    command = [
        v2.PYTHON,
        "-m",
        "daily_research.path_policy.run_alpha_path20_protocol",
        "--stage",
        "forecast-walkforward-study",
        "--tag",
        tag,
        "--data-source",
        "lake",
        "--lake-dataset-id",
        v2.DATASET_ID,
        "--pool-name",
        "rolling_liquid500_tradeable_mainboard_v2",
        "--pool-view-id",
        v2.V2_STRICT_POOL_VIEW_ID,
        "--benchmark",
        "000300.SH",
        "--start-date",
        "20180101",
        "--end-date",
        "20241231",
        "--max-universe-size",
        "0",
        "--execution-mode",
        "next_open",
        "--forecast-dataset-mode",
        "memmap",
        "--forecast-memmap-manifest",
        str(_seed7_manifest_path()),
        "--forecast-train-start-year",
        "2019",
        "--forecast-train-end-year",
        "2022",
        "--forecast-validation-year",
        "2023",
        "--forecast-test-year",
        "2024",
        "--forecast-model-families",
        family,
        "--forecast-feature-profile",
        FEATURE_PROFILE,
        "--forecast-include-static-context",
        "--forecast-max-feature-columns",
        "192",
        "--forecast-cumulative-horizons",
        HORIZON_GRID,
        "--forecast-horizon",
        "30",
        "--forecast-output-profile",
        OUTPUT_PROFILE,
        "--forecast-loss-profile",
        LOSS_PROFILE,
        "--forecast-selection-profile",
        SELECTION_PROFILE,
        "--forecast-decision-cost-bps",
        "20",
        "--forecast-decision-hit-threshold-bps",
        "10",
        "--forecast-decision-drawdown-penalty",
        "0.10",
        "--forecast-seeds",
        str(int(seed)),
        "--forecast-epochs",
        "24",
        "--forecast-min-epochs",
        "8",
        "--forecast-early-stop-patience",
        "6",
        "--forecast-checkpoint-every-n-epochs",
        "4",
        "--forecast-device",
        "cuda",
    ]
    command.extend(
        [
            "--forecast-hidden-dim",
            str(int(config["forecast_hidden_dim"])),
            "--forecast-gru-layers",
            str(int(config["forecast_gru_layers"])),
            "--forecast-transformer-layers",
            str(int(config["forecast_transformer_layers"])),
            "--forecast-transformer-heads",
            str(int(config["forecast_transformer_heads"])),
            "--forecast-patch-sizes",
            str(config["forecast_patch_sizes"]),
        ]
    )
    return command


def estimate_default_parameter_count(model_family: str | None = None) -> int:
    family = str(model_family or MODEL_FAMILY).strip()
    config = _model_family_config(family)
    model = make_forecast_model(
        family,
        input_dim=192,
        hidden_dim=int(config["forecast_hidden_dim"]),
        horizon=30,
        gru_layers=int(config["forecast_gru_layers"]),
        transformer_layers=int(config["forecast_transformer_layers"]),
        transformer_heads=int(config["forecast_transformer_heads"]),
        patch_sizes=tuple(int(item) for item in str(config["forecast_patch_sizes"]).split(",") if str(item).strip()),
        static_context_vocab_sizes={
            "symbol": 6000,
            "exchange": 8,
            "industry": 128,
            "liquidity_bucket": 6,
            "price_bucket": 6,
        },
        static_context_embedding_dims={
            "symbol": 16,
            "exchange": 4,
            "industry": 8,
            "liquidity_bucket": 4,
            "price_bucket": 4,
        },
        output_profile=OUTPUT_PROFILE,
        cumulative_horizons=tuple(int(item) for item in HORIZON_GRID.split(",") if item.strip()),
    )
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _load_manifest(study_dir: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = study_dir / "forecast_dataset_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest:
        return manifest, manifest_path
    summary_path = study_dir / "study_summary.json"
    summary = _read_json(summary_path)
    summary_manifest = summary.get("dataset_manifest", {}) if isinstance(summary, dict) else {}
    return (dict(summary_manifest), summary_path) if isinstance(summary_manifest, dict) else ({}, manifest_path)


def validate_source_study(
    study_dir: str | Path,
    *,
    dataset_id: str,
    pool_view_id: str,
    feature_profile: str,
) -> dict[str, Any]:
    resolved = Path(study_dir)
    manifest, manifest_path = _load_manifest(resolved)
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
        "blockers": blockers,
        "source_market_dataset_id": str(manifest.get("source_market_dataset_id", "")),
        "source_pool_view_id": str(manifest.get("source_pool_view_id", "")),
        "feature_profile": str(manifest.get("feature_profile", "")),
        "label_semantics": label_semantics,
    }


def _load_predictions(study_dir: Path, *, expert_name: str, role: str, score_column: str) -> pd.DataFrame:
    path = study_dir / f"forecast_predictions_{role}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing_prediction_csv:{path}")
    frame = pd.read_csv(path)
    required = {"date", "stock", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    keep = ["date", "stock", score_column]
    for column in ("future_decision_score", "pred_best_horizon", "future_best_horizon"):
        if column in frame.columns:
            keep.append(column)
    for column in frame.columns:
        if str(column).startswith("future_hit_label_"):
            keep.append(str(column))
    out = frame.loc[:, list(dict.fromkeys(keep))].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["stock"] = out["stock"].astype(str)
    out["expert_name"] = str(expert_name)
    out["score"] = pd.to_numeric(out[score_column], errors="coerce")
    return out.dropna(subset=["date", "stock", "score"])


def _rank_z_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    values: list[pd.Series] = []
    for _, group in out.groupby(["expert_name", "date"], sort=False):
        rank = group["score"].rank(method="average")
        std = float(rank.std(ddof=0))
        normalized = (rank - float(rank.mean())) / std if std > 1.0e-12 else rank * 0.0
        values.append(normalized)
    out["score_rank_z"] = pd.concat(values).sort_index() if values else pd.Series(dtype=float)
    return out


def _wide_scores(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = _rank_z_by_date(frame)
    wide = ranked.pivot_table(index=["date", "stock"], columns="expert_name", values="score_rank_z", aggfunc="mean")
    meta_columns = [
        column
        for column in ranked.columns
        if column not in {"date", "stock", "expert_name", "score", "score_rank_z", DEFAULT_SCORE_COLUMN}
    ]
    meta = ranked.drop_duplicates(["date", "stock"]).set_index(["date", "stock"])[meta_columns]
    return wide.join(meta, how="inner").reset_index()


def _fit_nonnegative_weights(
    validation_wide: pd.DataFrame,
    *,
    experts: tuple[str, ...],
    target_column: str = "future_decision_score",
) -> tuple[dict[str, float], dict[str, Any]]:
    work = validation_wide.dropna(subset=[*experts, target_column]).copy()
    if work.empty:
        weights = {expert: 1.0 / max(len(experts), 1) for expert in experts}
        return weights, {"status": "fallback_equal_weight", "reason": "empty_validation_target"}
    y = pd.to_numeric(work[target_column], errors="coerce")
    scores: dict[str, float] = {}
    for expert in experts:
        x = pd.to_numeric(work[expert], errors="coerce")
        corr = x.rank().corr(y.rank())
        scores[expert] = max(float(corr), 0.0) if pd.notna(corr) else 0.0
    total = sum(scores.values())
    if total <= 1.0e-12:
        weights = {expert: 1.0 / max(len(experts), 1) for expert in experts}
        return weights, {"status": "fallback_equal_weight", "reason": "nonpositive_validation_correlations", "raw_scores": scores}
    weights = {expert: float(value / total) for expert, value in scores.items()}
    return weights, {"status": "fit_from_validation_rank_ic", "raw_scores": scores}


def _apply_weights(wide: pd.DataFrame, *, experts: tuple[str, ...], weights: dict[str, float]) -> pd.DataFrame:
    out = wide.copy()
    score = np.zeros(len(out), dtype=float)
    for expert in experts:
        score += pd.to_numeric(out[expert], errors="coerce").fillna(0.0).to_numpy(dtype=float) * float(weights.get(expert, 0.0))
    out["score"] = score
    return out


def _daily_rank_ic(frame: pd.DataFrame, *, score_column: str = "score", target_column: str = "future_decision_score") -> dict[str, Any]:
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


def _write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# v2 Architecture Fusion Scout",
        "",
        f"- status: `{report.get('status', '')}`",
        f"- fusion_version: `{report.get('fusion_version', '')}`",
        f"- evidence_grade: `{report.get('fusion_evidence_grade', '')}`",
        f"- research_only: `{report.get('boundary', {}).get('research_only', True)}`",
        f"- promotion_allowed: `{report.get('boundary', {}).get('promotion_allowed', False)}`",
    ]
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_training_tasks(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    model_family: str | None = None,
) -> list[dict[str, Any]]:
    root = _anchor_root(output_root)
    resolved_seeds = _parse_seeds(seeds)
    family = str(model_family or MODEL_FAMILY).strip()
    config = _model_family_config(family)
    contract = _loss_contract()
    parameter_count = estimate_default_parameter_count(family)
    tasks: list[dict[str, Any]] = []
    for seed in resolved_seeds:
        tag = _study_tag(int(seed), model_family=family)
        tasks.append(
            {
                "tag": tag,
                "seed": int(seed),
                "model_family": family,
                "tier": str(config["tier"]),
                "fusion_version": str(config["fusion_version"]),
                "expert_families": list(config["expert_families"]),
                "target_parameter_band": str(config["target_parameter_band"]),
                "estimated_parameter_count": int(parameter_count),
                "router_diagnostics": list(config.get("router_diagnostics", [])),
                "loss_profile": LOSS_PROFILE,
                "loss_profile_contract": contract,
                "study_dir": str(STUDIES_ROOT / tag),
                "stdout": str(root / f"{tag}_stdout.log"),
                "stderr": str(root / f"{tag}_stderr.log"),
                "command": _training_command(seed=int(seed), model_family=family),
                "research_program": v2.RESEARCH_PROGRAM,
                "study_family": "v2_arch_fusion_scout",
                "source_market_dataset_id": v2.DATASET_ID,
                "source_pool_view_id": v2.V2_STRICT_POOL_VIEW_ID,
                "feature_profile": FEATURE_PROFILE,
                "output_profile": OUTPUT_PROFILE,
                "horizon_grid": HORIZON_GRID,
                "evidence_grade": "scout_only" if tuple(resolved_seeds) == (7,) else "evidence_grade_candidate",
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            }
        )
    return tasks


def write_task_list(
    output_root: str | Path | None = None,
    *,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    model_family: str | None = None,
    run_tag: str = RUN_TAG,
) -> Path:
    root = _anchor_root(output_root)
    path = root / "v2_arch_fusion_training_task_list.json"
    resolved_seeds = _parse_seeds(seeds)
    family = str(model_family or MODEL_FAMILY).strip()
    config = _model_family_config(family)
    parameter_count = estimate_default_parameter_count(family)
    payload = {
        "schema_version": 1,
        "run_tag": str(run_tag),
        "research_program": v2.RESEARCH_PROGRAM,
        "study_family": "v2_arch_fusion_scout",
        "created_at": _now(),
        "stage": "neural_fusion",
        "model_family": family,
        "tier": str(config["tier"]),
        "fusion_version": str(config["fusion_version"]),
        "expert_families": list(config["expert_families"]),
        "target_parameter_band": str(config["target_parameter_band"]),
        "estimated_parameter_count": int(parameter_count),
        "router_diagnostics": list(config.get("router_diagnostics", [])),
        "seeds": list(resolved_seeds),
        "loss_profile": LOSS_PROFILE,
        "loss_profile_contract": _loss_contract(),
        "training_task_count": len(resolved_seeds),
        "training_tasks": build_training_tasks(output_root=root, seeds=resolved_seeds, model_family=family),
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
    }
    _write_json(path, payload)
    return path


def run_training_tasks(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = DEFAULT_SEEDS,
    model_family: str | None = None,
    run_tag: str = RUN_TAG,
    skip_existing: bool = True,
) -> dict[str, Any]:
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    family = str(model_family or MODEL_FAMILY).strip()
    config = _model_family_config(family)
    for task in build_training_tasks(output_root=root, seeds=seeds, model_family=family):
        summary_path = Path(task["study_dir"]) / "study_summary.json"
        if skip_existing and summary_path.exists():
            completed.append(str(task["tag"]))
            results.append({"tag": task["tag"], "status": "skipped_existing", "study_summary_json": str(summary_path)})
            continue
        with Path(task["stdout"]).open("w", encoding="utf-8", errors="replace") as stdout, Path(task["stderr"]).open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr:
            proc = subprocess.Popen(task["command"], cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True)
            returncode = proc.wait()
        row = {
            "tag": task["tag"],
            "model_family": task["model_family"],
            "returncode": int(returncode),
            "stdout": task["stdout"],
            "stderr": task["stderr"],
        }
        if returncode == 0:
            completed.append(str(task["tag"]))
        else:
            failed.append(str(task["tag"]))
        results.append(row)
        if returncode != 0:
            break
    payload = {
        "schema_version": 1,
        "status": "completed" if not failed else "failed",
        "run_tag": str(run_tag),
        "model_family": family,
        "tier": str(config["tier"]),
        "fusion_version": str(config["fusion_version"]),
        "completed_tags": completed,
        "failed_tags": failed,
        "results": results,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_arch_fusion_training_summary.json", payload)
    return payload


def _read_study_summary(tag: str) -> dict[str, Any]:
    return _read_json(STUDIES_ROOT / str(tag) / "study_summary.json")


def _metric(summary: dict[str, Any], section: str, key: str) -> float:
    direct = summary.get(section, {})
    nested = dict(summary.get("training_summary", {}) or {}).get(section, {})
    value = dict(direct or nested or {}).get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _summary_section(summary: dict[str, Any]) -> dict[str, Any]:
    training = dict(summary.get("training_summary", {}) or {})
    return training if training else summary


def _finite_metric(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def run_neural_fusion_comparison(
    *,
    output_root: str | Path | None = None,
    seeds: tuple[int, ...] | list[int] | str | None = CONFIRM_SEEDS,
    model_family: str | None = None,
    run_tag: str = RUN_TAG,
) -> dict[str, Any]:
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_seeds = _parse_seeds(seeds, default=CONFIRM_SEEDS)
    family = str(model_family or MODEL_FAMILY).strip()
    config = _model_family_config(family)
    study_dirs: list[Path] = []
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for seed in resolved_seeds:
        tag = _study_tag(int(seed), model_family=family)
        summary = _read_study_summary(tag)
        if not summary:
            missing.append(tag)
            continue
        study_dirs.append(STUDIES_ROOT / tag)
        training_summary = _summary_section(summary)
        rows.append(
            {
                "tag": tag,
                "seed": int(seed),
                "status": str(summary.get("status", "")),
                "selected_model_family": str(training_summary.get("selected_model_family", "")),
                "evidence_verdict": str(summary.get("evidence_verdict", "")),
                "rank_ic_mean": _metric(summary, "test_metrics", "decision_score_rank_ic"),
                "top_bottom_spread_mean": _metric(summary, "test_metrics", "decision_score_top_bottom_spread"),
                "hit_lift_mean": _metric(summary, "test_metrics", "decision_hit_lift_top20_mean"),
                "monthly_positive_rate": 0.0,
                "negative_month_count": 0.0,
                "thirty_d_concentration": 0.0,
            }
        )
    audit_report: dict[str, Any] = {}
    audit_paths: dict[str, Path] = {}
    if study_dirs:
        audit_report = build_output_aux_profile_comparison(study_dirs, run_tag=str(run_tag))
        audit_paths = write_output_aux_profile_comparison(audit_report, root)
        audit_rows = {
            (str(row.get("study_tag", "")), int(row.get("seed", 0) or 0)): row
            for row in score_variant_comparison_rows(audit_report)
            if row.get("role") == "test" and row.get("score_name") == DEFAULT_SCORE_COLUMN
        }
        for row in rows:
            audit = audit_rows.get((str(row.get("tag", "")), int(row.get("seed", 0) or 0)))
            if not audit:
                continue
            row["rank_ic_mean"] = _finite_metric(audit.get("decision_score_rank_ic"), row["rank_ic_mean"])
            row["top_bottom_spread_mean"] = _finite_metric(
                audit.get("decision_score_top_bottom_spread"),
                row["top_bottom_spread_mean"],
            )
            row["hit_lift_mean"] = _finite_metric(audit.get("decision_hit_lift_top20_mean"), row["hit_lift_mean"])
            row["monthly_positive_rate"] = _finite_metric(audit.get("monthly_spread_positive_rate"))
            row["negative_month_count"] = _finite_metric(audit.get("negative_month_count"))
            row["thirty_d_concentration"] = _finite_metric(audit.get("thirty_d_concentration"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        aggregate = {"status": "blocked", "blockers": ["missing_all_neural_fusion_studies", *missing]}
    else:
        audit_aggregates = [
            row
            for row in profile_aggregate_rows(audit_report)
            if row.get("role") == "test" and row.get("score_name") == DEFAULT_SCORE_COLUMN
        ] if audit_report else []
        audit_aggregate = audit_aggregates[0] if audit_aggregates else {}
        aggregate = {
            "status": "completed" if not missing else "incomplete",
            "seed_count": int(len(frame)),
            "missing_tags": missing,
            "rank_ic_min": float(frame["rank_ic_mean"].min()),
            "top_bottom_spread_min": float(frame["top_bottom_spread_mean"].min()),
            "hit_lift_min": float(frame["hit_lift_mean"].min()),
            "monthly_positive_rate_mean": _finite_metric(
                audit_aggregate.get("monthly_positive_rate_mean", frame["monthly_positive_rate"].mean())
            ),
            "negative_month_count_max": _finite_metric(
                audit_aggregate.get("negative_month_count_max", frame["negative_month_count"].max())
            ),
            "thirty_d_concentration_mean": _finite_metric(
                audit_aggregate.get("thirty_d_concentration_mean", frame["thirty_d_concentration"].mean())
            ),
            "forecast_prediction_audit_status": str(audit_report.get("status", "")) if audit_report else "not_available",
        }
    if not frame.empty:
        frame.to_csv(root / "v2_arch_fusion_neural_comparison.csv", index=False, encoding="utf-8-sig")
    report = {
        "schema_version": 1,
        "status": aggregate["status"],
        "run_tag": str(run_tag),
        "study_family": "v2_arch_fusion_scout",
        "tier": str(config["tier"]),
        "fusion_version": str(config["fusion_version"]),
        "model_family": family,
        "expert_families": list(config["expert_families"]),
        "seeds": list(resolved_seeds),
        "aggregate": aggregate,
        "comparison_csv": str(root / "v2_arch_fusion_neural_comparison.csv") if not frame.empty else "",
        "forecast_prediction_audit": {
            "status": str(audit_report.get("status", "")) if audit_report else "not_available",
            "paths": {key: str(path) for key, path in audit_paths.items()},
        },
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_arch_fusion_neural_comparison_report.json", report)
    return report


def run_late_fusion(
    *,
    output_root: str | Path | None = None,
    run_tag: str = RUN_TAG,
    expert_study_tags: tuple[str, ...] | list[str] | str = (),
    expert_names: tuple[str, ...] | list[str] | str = DEFAULT_EXPERTS,
    studies_root: str | Path = STUDIES_ROOT,
    dataset_id: str = v2.DATASET_ID,
    pool_view_id: str = v2.V2_STRICT_POOL_VIEW_ID,
    feature_profile: str = v2.FEATURE_PROFILE,
    score_column: str = DEFAULT_SCORE_COLUMN,
    enforce_active_artifact_clean: bool = True,
) -> dict[str, Any]:
    if enforce_active_artifact_clean and _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    experts = _parse_csv_values(expert_names, default=DEFAULT_EXPERTS)
    tags = _parse_csv_values(expert_study_tags, default=experts)
    if len(tags) != len(experts):
        raise ValueError("expert_study_tags and expert_names must have the same count.")
    paths = _paths(output_root or (STUDIES_ROOT / str(run_tag)))
    paths.output_root.mkdir(parents=True, exist_ok=True)
    validations: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    source_root = Path(studies_root)
    for expert, tag in zip(experts, tags):
        study_dir = source_root / str(tag)
        validation = validate_source_study(
            study_dir,
            dataset_id=dataset_id,
            pool_view_id=pool_view_id,
            feature_profile=feature_profile,
        )
        validations.append({"expert_name": str(expert), "study_tag": str(tag), **validation})
        if validation["status"] != "ok":
            continue
        validation_frames.append(_load_predictions(study_dir, expert_name=str(expert), role="validation", score_column=score_column))
        test_frames.append(_load_predictions(study_dir, expert_name=str(expert), role="test", score_column=score_column))
    blockers = [
        f"{item['expert_name']}:{item['study_tag']}:{blocker}"
        for item in validations
        for blocker in item.get("blockers", [])
    ]
    if blockers:
        report = {
            "schema_version": 1,
            "status": "blocked",
            "run_tag": str(run_tag),
            "fusion_version": "late_fusion_rankz_v1",
            "blockers": blockers,
            "source_study_validations": validations,
            "boundary": {
                "research_only": True,
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            },
            "updated_at": _now(),
        }
        _write_json(paths.report_json, report)
        _write_markdown(paths.report_md, report)
        return report
    validation_wide = _wide_scores(pd.concat(validation_frames, ignore_index=True))
    test_wide = _wide_scores(pd.concat(test_frames, ignore_index=True))
    weights, fit_diagnostics = _fit_nonnegative_weights(validation_wide, experts=experts)
    validation_fused = _apply_weights(validation_wide, experts=experts, weights=weights)
    test_fused = _apply_weights(test_wide, experts=experts, weights=weights)
    validation_fused.to_csv(paths.validation_panel_csv, index=False, encoding="utf-8-sig")
    test_fused.to_csv(paths.test_panel_csv, index=False, encoding="utf-8-sig")
    weight_report = {
        "schema_version": 1,
        "fusion_version": "late_fusion_rankz_v1",
        "weight_fit_role": "validation",
        "test_label_used_for_weight_fit": False,
        "expert_families": list(experts),
        "expert_study_tags": list(tags),
        "weights": weights,
        "fit_diagnostics": fit_diagnostics,
    }
    _write_json(paths.weight_report_json, weight_report)
    report = {
        "schema_version": 1,
        "status": "completed",
        "run_tag": str(run_tag),
        "research_program": v2.RESEARCH_PROGRAM,
        "study_family": "v2_arch_fusion_scout",
        "fusion_version": "late_fusion_rankz_v1",
        "fusion_evidence_grade": "scout_only",
        "dataset_id": str(dataset_id),
        "pool_view_id": str(pool_view_id),
        "feature_profile": str(feature_profile),
        "score_column": str(score_column),
        "expert_families": list(experts),
        "expert_study_tags": list(tags),
        "fusion_weight_report_json": str(paths.weight_report_json),
        "fusion_score_panel_validation_csv": str(paths.validation_panel_csv),
        "fusion_score_panel_test_csv": str(paths.test_panel_csv),
        "diagnostics": {
            "validation_rank_ic": _daily_rank_ic(validation_fused),
            "test_rank_ic": _daily_rank_ic(test_fused),
            "weight_fit": fit_diagnostics,
        },
        "source_study_validations": validations,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
            "test_label_used_for_weight_fit": False,
        },
        "updated_at": _now(),
    }
    _write_json(paths.report_json, report)
    _write_markdown(paths.report_md, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run daily_research v2 architecture fusion scouts.")
    parser.add_argument("--stage", choices=["late_fusion", "neural_fusion"], default="late_fusion")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--expert-study-tags", default="")
    parser.add_argument("--expert-names", default=",".join(DEFAULT_EXPERTS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--model-family", default=MODEL_FAMILY, choices=SUPPORTED_MODEL_FAMILIES)
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--confirm-seeds", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--studies-root", default=str(STUDIES_ROOT))
    parser.add_argument("--dataset-id", default=v2.DATASET_ID)
    parser.add_argument("--pool-view-id", default=v2.V2_STRICT_POOL_VIEW_ID)
    parser.add_argument("--feature-profile", default=v2.FEATURE_PROFILE)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    output_root = args.output_root or str(STUDIES_ROOT / str(args.run_tag))
    neural_action_requested = bool(args.write_task_list or args.run_training or args.run_comparison)
    if args.stage == "late_fusion" and not neural_action_requested:
        report = run_late_fusion(
            output_root=output_root,
            run_tag=args.run_tag,
            expert_study_tags=args.expert_study_tags,
            expert_names=args.expert_names,
            studies_root=args.studies_root,
            dataset_id=args.dataset_id,
            pool_view_id=args.pool_view_id,
            feature_profile=args.feature_profile,
            score_column=args.score_column,
        )
    else:
        resolved_seeds = CONFIRM_SEEDS if bool(args.confirm_seeds) else _parse_seeds(args.seeds)
        model_family = str(args.model_family)
        config = _model_family_config(model_family)
        actions: dict[str, Any] = {}
        if args.write_task_list or not (args.run_training or args.run_comparison):
            actions["task_list"] = str(
                write_task_list(
                    output_root=output_root,
                    seeds=resolved_seeds,
                    model_family=model_family,
                    run_tag=str(args.run_tag),
                )
            )
        if args.run_training:
            actions["training_summary"] = run_training_tasks(
                output_root=output_root,
                seeds=resolved_seeds,
                model_family=model_family,
                run_tag=str(args.run_tag),
                skip_existing=not bool(args.no_skip_existing),
            )
        if args.run_comparison:
            actions["comparison"] = run_neural_fusion_comparison(
                output_root=output_root,
                seeds=resolved_seeds,
                model_family=model_family,
                run_tag=str(args.run_tag),
            )
        report = {
            "schema_version": 1,
            "status": "completed",
            "run_tag": str(args.run_tag),
            "stage": "neural_fusion",
            "model_family": model_family,
            "tier": str(config["tier"]),
            "fusion_version": str(config["fusion_version"]),
            "expert_families": list(config["expert_families"]),
            "seeds": list(resolved_seeds),
            "actions": actions,
            "boundary": {
                "research_only": True,
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            },
            "updated_at": _now(),
        }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status={report.get('status')} run_tag={args.run_tag} output_root={output_root}")
    return 0 if str(report.get("status")) == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
