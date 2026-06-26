from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant_data_platform.lake import ResearchDataLake, load_pool_view
from daily_research.path_policy.decision_score_diagnostics import build_diagnostics
from daily_research.path_policy.output_aux_profile_comparison import (
    build_output_aux_profile_comparison,
    profile_aggregate_rows,
    write_output_aux_profile_comparison,
)
from daily_research.path_policy.target_calibration_audit import build_target_calibration_audit, parse_target_grid


PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
RUN_TAG = "mh_rebuild_mainboard_anchor_20260601_01"
RESEARCH_PROGRAM = "alpha_multi_horizon_utility_policy_v1"
STUDY_FAMILY = "mainboard_rebuild_baseline"
DATASET_ID = "policy_input_bundle__45e3d8c059ba718426a9f887"
CORRECTED_POOL_VIEW_ID = "policy_pool_view__74f45f4f83263bccd64a8027"
WRONG_UNIVERSE_POOL_VIEW_ID = "policy_pool_view__8f9367ce305548e3ecd04feb"
OLD_STAGE28_DATASET_ID = "policy_input_bundle__7c8f58d851bce8179e1e9e2d"
OLD_STAGE28_POOL_VIEW_ID = "policy_pool_view__c11400fa72ad263f3d1eecfa"
OLD_STAGE28_FEATURE_SHAPE = [1699, 2430, 156]
OLD_STAGE28_TRAIN_ROWS = 452034
OLD_STAGE28_THIRTY_D_CONCENTRATION = 0.776934
OLD_STAGE28_NEGATIVE_MONTH_MAX = 2
SHORT_V5B_ROOT = PROJECT_ROOT / "daily_research/output/short_expert_policy_v5b_execalign_production_default"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
FEATURE_PROFILE = "raw_kline_context_no_alpha_prior_v1"
MODEL_FAMILY = "gru_sequence_static_context"
LOSS_PROFILE = "target_norm_head_constraint_v1"
OUTPUT_PROFILE = "decision_utility_v1"
SELECTION_PROFILE = "decision_utility"
HORIZON_GRID = "1,2,3,5,8,10,15,20,30"
SEEDS = (7, 11, 19)
EXCLUDED_PREFIXES = ("300", "301", "688", "689")


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _study_tag(seed: int) -> str:
    return f"mh_rebuild_mainboard_target_norm_head_constraint_raw_seed{int(seed)}_20260601_01"


def _anchor_root(output_root: str | Path | None = None) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG


def _seed7_manifest_path() -> Path:
    return STUDIES_ROOT / _study_tag(7) / "forecast_dataset_manifest.json"


def _training_command(*, seed: int) -> list[str]:
    tag = _study_tag(seed)
    command = [
        PYTHON,
        "-m",
        "daily_research.path_policy.run_alpha_path20_protocol",
        "--stage",
        "forecast-walkforward-study",
        "--tag",
        tag,
        "--data-source",
        "lake",
        "--lake-dataset-id",
        DATASET_ID,
        "--pool-name",
        "rolling_liquid500_mainboard",
        "--pool-view-id",
        CORRECTED_POOL_VIEW_ID,
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
        "--forecast-train-start-year",
        "2019",
        "--forecast-train-end-year",
        "2022",
        "--forecast-validation-year",
        "2023",
        "--forecast-test-year",
        "2024",
        "--forecast-model-families",
        MODEL_FAMILY,
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
    if int(seed) != 7:
        command.extend(["--forecast-memmap-manifest", str(_seed7_manifest_path())])
    return command


def build_mainboard_training_tasks(*, output_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _anchor_root(output_root)
    tasks: list[dict[str, Any]] = []
    for seed in SEEDS:
        tag = _study_tag(seed)
        tasks.append(
            {
                "tag": tag,
                "seed": int(seed),
                "study_dir": str(STUDIES_ROOT / tag),
                "stdout": str(root / f"{tag}_stdout.log"),
                "stderr": str(root / f"{tag}_stderr.log"),
                "command": _training_command(seed=int(seed)),
                "research_program": RESEARCH_PROGRAM,
                "study_family": STUDY_FAMILY,
                "source_market_dataset_id": DATASET_ID,
                "source_pool_view_id": CORRECTED_POOL_VIEW_ID,
                "wrong_universe_pool_view_id": WRONG_UNIVERSE_POOL_VIEW_ID,
                "feature_profile": FEATURE_PROFILE,
                "model_family": MODEL_FAMILY,
                "loss_profile": LOSS_PROFILE,
                "horizon_grid": HORIZON_GRID,
                "reuses_seed7_memmap_manifest": int(seed) != 7,
                "seed7_memmap_manifest": str(_seed7_manifest_path()) if int(seed) != 7 else "",
                "shadow_only": True,
                "promotion_allowed": False,
                "active_execution_strategy_expected_diff": "none",
            }
        )
    return tasks


def write_mainboard_task_list(output_root: str | Path | None = None) -> Path:
    root = _anchor_root(output_root)
    path = root / "mainboard_rebuild_task_list.json"
    payload = {
        "schema_version": 1,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "created_at": _now(),
        "training_task_count": len(SEEDS),
        "training_tasks": build_mainboard_training_tasks(output_root=root),
        "boundary": "research-only / shadow-only; execution remains frozen_skeleton_only.",
    }
    _write_json(path, payload)
    return path


def _symbol_code(symbol: str) -> str:
    return str(symbol or "").strip().upper().split(".", 1)[0]


def _prefix_counts(symbols: Iterable[str], prefixes: Iterable[str] = EXCLUDED_PREFIXES) -> dict[str, int]:
    out = {str(prefix): 0 for prefix in prefixes}
    for symbol in symbols:
        code = _symbol_code(str(symbol))
        for prefix in out:
            if code.startswith(prefix):
                out[prefix] += 1
    return out


def validate_corrected_pool_view(
    *,
    pool_view_id: str = CORRECTED_POOL_VIEW_ID,
    data_lake_root: str | Path | None = None,
) -> dict[str, Any]:
    lake = ResearchDataLake(str(data_lake_root or "").strip() or None)
    pool = load_pool_view(lake=lake, pool_view_id=str(pool_view_id))
    membership = pool.membership_frame.fillna(False).astype(bool)
    active_symbols = [str(column) for column in membership.columns if bool(membership[column].any())]
    daily_counts = membership.sum(axis=1).astype(int)
    parameters = dict(pool.metadata.get("parameters", {}) or {})
    source_cache = dict(pool.metadata.get("source_cache", {}) or {})
    quality = dict(source_cache.get("quality_report", {}) or {})
    excluded_counts = _prefix_counts(active_symbols)
    blockers: list[str] = []
    if str(pool.dataset_id) != CORRECTED_POOL_VIEW_ID:
        blockers.append("unexpected_pool_view_id")
    if str(parameters.get("source_market_dataset_id", "")) != DATASET_ID:
        blockers.append("source_market_dataset_mismatch")
    if str(parameters.get("view_kind", "")) != "rolling_liquidity":
        blockers.append("pool_view_kind_mismatch")
    if str(parameters.get("view_name", "")) != "rolling_liquid500_mainboard":
        blockers.append("pool_view_name_mismatch")
    observed_prefixes = tuple(str(item) for item in parameters.get("exclude_symbol_prefixes", []) or [])
    if observed_prefixes != EXCLUDED_PREFIXES:
        blockers.append("exclude_symbol_prefixes_mismatch")
    if sum(excluded_counts.values()) != 0:
        blockers.append("excluded_prefix_active_symbols_present")
    if daily_counts.empty or int(daily_counts.min()) != 500 or int(daily_counts.max()) != 500:
        blockers.append("daily_member_count_not_exactly_500")
    return {
        "schema_version": 1,
        "status": "ok" if not blockers else "blocked",
        "pool_view_id": str(pool.dataset_id),
        "parameters": parameters,
        "quality_report": quality,
        "active_universe_size": int(len(active_symbols)),
        "membership_symbols": int(len(membership.columns)),
        "daily_member_count_min": int(daily_counts.min()) if not daily_counts.empty else 0,
        "daily_member_count_median": float(daily_counts.median()) if not daily_counts.empty else 0.0,
        "daily_member_count_max": int(daily_counts.max()) if not daily_counts.empty else 0,
        "active_excluded_prefix_counts": excluded_counts,
        "active_excluded_prefix_total": int(sum(excluded_counts.values())),
        "blockers": blockers,
    }


def validate_mainboard_memmap_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = _read_json(path)
    feature_columns = [str(item) for item in manifest.get("feature_columns", []) or []]
    stock_values = [str(item) for item in manifest.get("stock_values", []) or []]
    shape = [int(item) for item in manifest.get("feature_store_shape", []) or []]
    label_semantics = dict(manifest.get("label_semantics", {}) or {})
    sample_count_by_role = {str(key): int(value) for key, value in dict(manifest.get("sample_count_by_role", {}) or {}).items()}
    blockers: list[str] = []
    if not path.exists():
        blockers.append("missing_manifest")
    if str(manifest.get("source_market_dataset_id", "")) != DATASET_ID:
        blockers.append("source_market_dataset_mismatch")
    if str(manifest.get("source_pool_view_id", "")) != CORRECTED_POOL_VIEW_ID:
        blockers.append("source_pool_view_mismatch")
    if str(manifest.get("source_pool_view_name", "")) != "rolling_liquid500_mainboard":
        blockers.append("source_pool_view_name_mismatch")
    if len(shape) != 3:
        blockers.append("invalid_feature_store_shape")
    elif int(shape[2]) != len(feature_columns):
        blockers.append("feature_shape_column_count_mismatch")
    if int(manifest.get("feature_count_after_cap", len(feature_columns)) or 0) != len(feature_columns):
        blockers.append("feature_count_after_cap_mismatch")
    if str(label_semantics.get("label_semantics", "")) != "next_open_entry_to_future_open":
        blockers.append("label_semantics_not_next_open")
    if str(manifest.get("execution_mode", "")) != "next_open" or int(manifest.get("next_open_label_extra_trading_day", 0) or 0) != 1:
        blockers.append("next_open_execution_metadata_missing")
    for role in ("train", "validation", "test"):
        if int(sample_count_by_role.get(role, 0)) <= 0:
            blockers.append(f"sample_count_missing_{role}")
    excluded_counts = _prefix_counts(stock_values)
    if sum(excluded_counts.values()) != 0:
        blockers.append("excluded_prefix_stock_values_present")
    return {
        "schema_version": 1,
        "status": "ok" if not blockers else "blocked",
        "manifest_path": str(path),
        "source_market_dataset_id": str(manifest.get("source_market_dataset_id", "")),
        "source_pool_view_id": str(manifest.get("source_pool_view_id", "")),
        "source_pool_view_name": str(manifest.get("source_pool_view_name", "")),
        "feature_profile": str(manifest.get("feature_profile", "")),
        "feature_count": int(len(feature_columns)),
        "feature_count_before_cap": int(manifest.get("feature_count_before_cap", len(feature_columns)) or 0),
        "feature_count_after_cap": int(manifest.get("feature_count_after_cap", len(feature_columns)) or 0),
        "feature_store_shape": shape,
        "feature_group_counts": dict(manifest.get("feature_group_counts", {}) or {}),
        "sample_count_by_role": sample_count_by_role,
        "label_semantics": label_semantics,
        "execution_mode": str(manifest.get("execution_mode", "")),
        "next_open_label_extra_trading_day": int(manifest.get("next_open_label_extra_trading_day", 0) or 0),
        "stock_count": int(len(stock_values) if stock_values else (shape[1] if len(shape) >= 2 else 0)),
        "stock_values_excluded_prefix_counts": excluded_counts,
        "blockers": blockers,
    }


def _study_summary_lake_coverage(seed: int = 7) -> dict[str, Any]:
    summary = _read_json(STUDIES_ROOT / _study_tag(seed) / "study_summary.json")
    prepared = dict(summary.get("prepared_summary", {}) or {})
    raw_cache = dict(prepared.get("raw_cache_meta", {}) or {})
    return dict(raw_cache.get("lake_coverage_report", {}) or {})


def _aggregate_gate_row(anchor_root: Path) -> dict[str, Any]:
    path = anchor_root / "profile_aggregate.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    rows = frame[(frame.get("role") == "test") & (frame.get("score_name") == "pred_decision_score")]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def corrected_gate_status(anchor_root: str | Path) -> dict[str, Any]:
    row = _aggregate_gate_row(Path(anchor_root))
    if not row:
        return {"status": "blocked", "reason": "missing_profile_aggregate_pred_decision_score", "checks": {}}
    checks = {
        "seed_count_ge_3": int(row.get("seed_count", 0) or 0) >= 3,
        "rank_ic_min_positive": _finite_float(row.get("rank_ic_min")) > 0.0,
        "spread_min_positive": _finite_float(row.get("spread_min")) > 0.0,
        "hit_lift_min_positive": _finite_float(row.get("hit_lift_min")) > 0.0,
        "monthly_positive_rate_mean_ge_075": _finite_float(row.get("monthly_positive_rate_mean")) >= 0.75,
        "negative_month_count_max_le_2": int(row.get("negative_month_count_max", 99) or 99) <= OLD_STAGE28_NEGATIVE_MONTH_MAX,
        "thirty_d_concentration_not_worse_than_stage28": _finite_float(row.get("thirty_d_concentration_mean"), 1.0)
        <= OLD_STAGE28_THIRTY_D_CONCENTRATION,
    }
    passed = all(checks.values())
    near_pass = bool(
        checks["seed_count_ge_3"]
        and checks["rank_ic_min_positive"]
        and checks["spread_min_positive"]
        and sum(1 for value in checks.values() if not value) <= 1
    )
    return {
        "status": "pass" if passed else "near_pass" if near_pass else "fail",
        "checks": checks,
        "row": {str(key): _json_default(value) for key, value in row.items()},
    }


def _find_legacy_feature_manifest(search_roots: Iterable[str | Path]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for root in search_roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("forecast_dataset_manifest.json"):
            payload = _read_json(path)
            shape = [int(item) for item in payload.get("feature_store_shape", []) or []]
            columns = [str(item) for item in payload.get("feature_columns", []) or []]
            if len(shape) == 3 and int(shape[2]) == 156 and len(columns) == 156:
                candidates.append(
                    {
                        "manifest_path": str(path),
                        "source_market_dataset_id": str(payload.get("source_market_dataset_id", "")),
                        "source_pool_view_id": str(payload.get("source_pool_view_id", "")),
                        "feature_store_shape": shape,
                        "feature_columns": columns,
                    }
                )
    return candidates[0] if candidates else {}


def build_feature_schema_diff(
    *,
    current_manifest_path: str | Path,
    legacy_manifest_path: str | Path = "",
    search_roots: Iterable[str | Path] = (PROJECT_ROOT / "daily_research/output/path_policy/studies", PROJECT_ROOT / "daily_research/cache"),
) -> dict[str, Any]:
    current = _read_json(Path(current_manifest_path))
    current_columns = [str(item) for item in current.get("feature_columns", []) or []]
    legacy_payload: dict[str, Any] = {}
    if str(legacy_manifest_path or "").strip():
        path = Path(legacy_manifest_path)
        payload = _read_json(path)
        legacy_payload = {
            "manifest_path": str(path),
            "source_market_dataset_id": str(payload.get("source_market_dataset_id", "")),
            "source_pool_view_id": str(payload.get("source_pool_view_id", "")),
            "feature_store_shape": [int(item) for item in payload.get("feature_store_shape", []) or []],
            "feature_columns": [str(item) for item in payload.get("feature_columns", []) or []],
        }
    else:
        legacy_payload = _find_legacy_feature_manifest(search_roots)
    legacy_columns = [str(item) for item in legacy_payload.get("feature_columns", []) or []]
    if not legacy_columns:
        return {
            "schema_version": 1,
            "status": "legacy156_not_recovered",
            "current_manifest_path": str(current_manifest_path),
            "current_feature_count": int(len(current_columns)),
            "current_feature_store_shape": [int(item) for item in current.get("feature_store_shape", []) or []],
            "old_expected_feature_store_shape": OLD_STAGE28_FEATURE_SHAPE,
            "legacy_manifest_path": "",
            "feature_schema_shift": "unresolved_root_cause",
            "legacy156_profile_allowed": False,
            "reason": "No local forecast_dataset_manifest.json with exactly 156 feature columns was found; do not fabricate a legacy156 profile.",
        }
    current_set = set(current_columns)
    legacy_set = set(legacy_columns)
    return {
        "schema_version": 1,
        "status": "completed",
        "current_manifest_path": str(current_manifest_path),
        "legacy_manifest_path": str(legacy_payload.get("manifest_path", "")),
        "current_feature_count": int(len(current_columns)),
        "legacy_feature_count": int(len(legacy_columns)),
        "current_feature_store_shape": [int(item) for item in current.get("feature_store_shape", []) or []],
        "legacy_feature_store_shape": [int(item) for item in legacy_payload.get("feature_store_shape", []) or []],
        "shared_feature_count": int(len(current_set & legacy_set)),
        "legacy_only_feature_count": int(len(legacy_set - current_set)),
        "current_only_feature_count": int(len(current_set - legacy_set)),
        "legacy_only_features": sorted(legacy_set - current_set),
        "current_only_features": sorted(current_set - legacy_set),
        "shared_features": sorted(current_set & legacy_set),
        "legacy156_profile_allowed": True,
    }


def build_short_v5b_bridge_status(*, corrected_gate: dict[str, Any], short_v5b_root: str | Path = SHORT_V5B_ROOT) -> dict[str, Any]:
    root = Path(short_v5b_root)
    if not root.exists():
        return {
            "status": "blocked_missing_short_v5b_payload",
            "short_v5b_root": str(root),
            "corrected_gate_status": str(corrected_gate.get("status", "")),
            "bridge_allowed_now": False,
            "required_common_protocol": [
                "common mainboard-only pool",
                "common transaction/slippage/tax costs",
                "common rebalance policy",
                "common benchmark",
                "common train/validation/test dates",
                "common score panel and target-weight panel semantics",
                "common next-open execution assumption",
            ],
        }
    return {
        "status": "deferred_until_corrected_gate_passes" if corrected_gate.get("status") not in {"pass", "near_pass"} else "ready_for_same_protocol_design",
        "short_v5b_root": str(root),
        "corrected_gate_status": str(corrected_gate.get("status", "")),
        "bridge_allowed_now": corrected_gate.get("status") in {"pass", "near_pass"},
    }


def build_lineage_equivalence_audit(
    *,
    anchor_root: str | Path,
    seed_manifest_path: str | Path,
    pool_validation: dict[str, Any],
    memmap_validation: dict[str, Any],
    feature_schema_diff: dict[str, Any],
    corrected_gate: dict[str, Any],
    bridge_status: dict[str, Any],
) -> dict[str, Any]:
    lake_coverage = _study_summary_lake_coverage(seed=7)
    provider_chain = []
    lake_manifest = _read_json(PROJECT_ROOT / "quant_data_platform/data/lake/manifest_latest.json")
    for record in lake_manifest.get("datasets", []) or []:
        if isinstance(record, dict) and str(record.get("dataset_id", "")) == DATASET_ID:
            params = dict(record.get("parameters", {}) or {})
            provider_chain = [str(item) for item in params.get("provider_chain", []) or []]
            break
    return {
        "schema_version": 1,
        "status": "completed",
        "run_tag": RUN_TAG,
        "anchor_root": str(anchor_root),
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "wrong_universe_diagnostic": {
            "pool_view_id": WRONG_UNIVERSE_POOL_VIEW_ID,
            "classification": "wrong_universe_diagnostic",
            "reason": "Contains ChiNext/STAR prefixes and is not old Stage 2.8 or short_v5b equivalent.",
        },
        "current_mainboard_lineage": {
            "source_market_dataset_id": DATASET_ID,
            "provider_chain": provider_chain,
            "pool_view_id": CORRECTED_POOL_VIEW_ID,
            "pool_validation": pool_validation,
            "memmap_manifest_path": str(seed_manifest_path),
            "memmap_validation": memmap_validation,
            "benchmark_open": {
                "benchmark_open_source": str(lake_coverage.get("benchmark_open_source", "")),
                "benchmark_open_rows": int(lake_coverage.get("benchmark_open_rows", 0) or 0),
                "require_benchmark_open": bool(lake_coverage.get("require_benchmark_open", False)),
            },
        },
        "historical_stage28_boundary": {
            "dataset_id": OLD_STAGE28_DATASET_ID,
            "pool_view_id": OLD_STAGE28_POOL_VIEW_ID,
            "feature_store_shape": OLD_STAGE28_FEATURE_SHAPE,
            "train_rows": OLD_STAGE28_TRAIN_ROWS,
            "evidence_level": "brain_confirmed_text_only_unless_legacy_manifest_recovered",
        },
        "feature_schema_diff": feature_schema_diff,
        "corrected_gate": corrected_gate,
        "short_v5b_bridge_status": bridge_status,
        "interpretation": [
            "Corrected mainboard rebuild is a new-lineage mainboard baseline, not an old Stage 2.8 payload replay.",
            "Do not compare against short_v5b until the same-protocol bridge has a file-backed old payload or explicit substitute.",
            "Execution remains frozen_skeleton_only until research evidence closes.",
        ],
    }


def _audit_markdown(audit: dict[str, Any]) -> str:
    memmap = audit["current_mainboard_lineage"]["memmap_validation"]
    pool = audit["current_mainboard_lineage"]["pool_validation"]
    gate = audit["corrected_gate"]
    feature = audit["feature_schema_diff"]
    bridge = audit["short_v5b_bridge_status"]
    lines = [
        "# Mainboard Rebuild Lineage Equivalence Audit",
        "",
        f"- status: `{audit.get('status')}`",
        f"- run_tag: `{audit.get('run_tag')}`",
        "- boundary: research-only / shadow-only; execution remains frozen.",
        f"- dataset: `{audit['current_mainboard_lineage']['source_market_dataset_id']}`",
        f"- corrected pool: `{audit['current_mainboard_lineage']['pool_view_id']}`",
        f"- wrong-universe diagnostic pool: `{audit['wrong_universe_diagnostic']['pool_view_id']}`",
        "",
        "## Mainboard Pool",
        "",
        f"- pool status: `{pool.get('status')}`",
        f"- active universe: `{pool.get('active_universe_size')}`",
        f"- daily member count: `{pool.get('daily_member_count_min')}/{pool.get('daily_member_count_median')}/{pool.get('daily_member_count_max')}`",
        f"- excluded prefix active total: `{pool.get('active_excluded_prefix_total')}`",
        "",
        "## Memmap",
        "",
        f"- memmap status: `{memmap.get('status')}`",
        f"- feature store shape: `{memmap.get('feature_store_shape')}`",
        f"- sample rows: `{memmap.get('sample_count_by_role')}`",
        f"- label semantics: `{memmap.get('label_semantics', {}).get('label_semantics', '')}`",
        "",
        "## Gate",
        "",
        f"- corrected gate status: `{gate.get('status')}`",
        f"- checks: `{gate.get('checks')}`",
        "",
        "## Feature Schema",
        "",
        f"- feature schema status: `{feature.get('status')}`",
        f"- current feature count: `{feature.get('current_feature_count')}`",
        f"- legacy156 profile allowed: `{feature.get('legacy156_profile_allowed')}`",
        "",
        "## short_v5b Bridge",
        "",
        f"- bridge status: `{bridge.get('status')}`",
        f"- bridge allowed now: `{bridge.get('bridge_allowed_now')}`",
        "",
    ]
    return "\n".join(lines)


def write_lineage_equivalence_audit(audit: dict[str, Any], output_root: str | Path) -> dict[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "mainboard_lineage_equivalence_audit.json"
    md_path = root / "mainboard_lineage_equivalence_audit.md"
    _write_json(json_path, audit)
    md_path.write_text(_audit_markdown(audit), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _progress_payload(*, status: str, completed: list[str], failed: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "completed_tags": completed,
        "failed_tags": failed,
        "updated_at": _now(),
    }


def run_training_tasks(*, output_root: str | Path | None = None, skip_existing: bool = True) -> dict[str, Any]:
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    for task in build_mainboard_training_tasks(output_root=root):
        summary_path = Path(task["study_dir"]) / "study_summary.json"
        if skip_existing and summary_path.exists():
            completed.append(task["tag"])
            results.append({"tag": task["tag"], "status": "skipped_existing", "study_summary_json": str(summary_path)})
            continue
        if bool(task.get("reuses_seed7_memmap_manifest")) and not _seed7_manifest_path().exists():
            failed.append(task["tag"])
            results.append({"tag": task["tag"], "status": "blocked_missing_seed7_memmap_manifest", "manifest": str(_seed7_manifest_path())})
            break
        with Path(task["stdout"]).open("w", encoding="utf-8", errors="replace") as stdout, Path(task["stderr"]).open("w", encoding="utf-8", errors="replace") as stderr:
            proc = subprocess.Popen(task["command"], cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True)
            returncode = proc.wait()
        row = {"tag": task["tag"], "returncode": int(returncode), "stdout": task["stdout"], "stderr": task["stderr"]}
        if returncode == 0:
            completed.append(task["tag"])
        else:
            failed.append(task["tag"])
        results.append(row)
        _write_json(root / "mainboard_training_progress.json", _progress_payload(status="running" if not failed else "blocked", completed=completed, failed=failed))
        if returncode != 0:
            break
    status = "completed" if not failed and len(completed) == len(SEEDS) else "blocked"
    payload = {
        "schema_version": 1,
        "status": status,
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "completed_tags": completed,
        "failed_tags": failed,
        "results": results,
        "updated_at": _now(),
    }
    _write_json(root / "mainboard_training_progress.json", _progress_payload(status=status, completed=completed, failed=failed))
    _write_json(root / "mainboard_training_summary.json", payload)
    return payload


def run_comparison_and_audits(
    *,
    output_root: str | Path | None = None,
    legacy_feature_manifest: str | Path = "",
) -> dict[str, Any]:
    root = _anchor_root(output_root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = build_mainboard_training_tasks(output_root=root)
    report = build_output_aux_profile_comparison([task["study_dir"] for task in tasks], run_tag=RUN_TAG)
    comparison_paths = write_output_aux_profile_comparison(report, root)
    tags = [task["tag"] for task in tasks]
    target_audit = build_target_calibration_audit(tags, studies_root=STUDIES_ROOT, target_grid=parse_target_grid("20:10:0.10"), cumulative_horizons=HORIZON_GRID)
    target_path = root / "mainboard_target_calibration_audit.json"
    _write_json(target_path, target_audit)
    decision_audit = build_diagnostics(tags, studies_root=STUDIES_ROOT, target_audit=target_audit)
    decision_path = root / "mainboard_decision_score_diagnostics.json"
    _write_json(decision_path, decision_audit)
    pool_validation = validate_corrected_pool_view()
    pool_validation_path = root / "mainboard_pool_validation.json"
    _write_json(pool_validation_path, pool_validation)
    seed7_manifest = _seed7_manifest_path()
    memmap_validation = validate_mainboard_memmap_manifest(seed7_manifest)
    memmap_validation_path = root / "mainboard_memmap_validation.json"
    _write_json(memmap_validation_path, memmap_validation)
    feature_diff = build_feature_schema_diff(current_manifest_path=seed7_manifest, legacy_manifest_path=legacy_feature_manifest)
    feature_diff_path = root / "feature_schema_diff_report.json"
    _write_json(feature_diff_path, feature_diff)
    gate = corrected_gate_status(root)
    bridge = build_short_v5b_bridge_status(corrected_gate=gate)
    lineage = build_lineage_equivalence_audit(
        anchor_root=root,
        seed_manifest_path=seed7_manifest,
        pool_validation=pool_validation,
        memmap_validation=memmap_validation,
        feature_schema_diff=feature_diff,
        corrected_gate=gate,
        bridge_status=bridge,
    )
    lineage_paths = write_lineage_equivalence_audit(lineage, root)
    summary = {
        "schema_version": 1,
        "status": "completed" if pool_validation["status"] == "ok" and memmap_validation["status"] == "ok" else "completed_with_blockers",
        "run_tag": RUN_TAG,
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "comparison_paths": {key: str(path) for key, path in comparison_paths.items()},
        "target_calibration_audit_json": str(target_path),
        "decision_score_diagnostics_json": str(decision_path),
        "pool_validation_json": str(pool_validation_path),
        "memmap_validation_json": str(memmap_validation_path),
        "feature_schema_diff_report_json": str(feature_diff_path),
        "lineage_equivalence_audit_paths": {key: str(path) for key, path in lineage_paths.items()},
        "corrected_gate": gate,
        "short_v5b_bridge_status": bridge,
        "updated_at": _now(),
    }
    _write_json(root / "mainboard_rebuild_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run corrected mainboard-only rebuild baseline and audits.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--validate-pool", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-comparison", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--legacy-feature-manifest", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    outputs: dict[str, Any] = {"run_tag": RUN_TAG, "output_root": args.output_root}
    if args.write_task_list or not (args.validate_pool or args.run_training or args.run_comparison):
        outputs["task_list"] = str(write_mainboard_task_list(args.output_root))
    if args.validate_pool:
        payload = validate_corrected_pool_view()
        path = _anchor_root(args.output_root) / "mainboard_pool_validation.json"
        _write_json(path, payload)
        outputs["pool_validation"] = payload
        outputs["pool_validation_json"] = str(path)
    if args.run_training:
        outputs["training_summary"] = run_training_tasks(output_root=args.output_root, skip_existing=not args.no_skip_existing)
    if args.run_comparison:
        outputs["comparison_summary"] = run_comparison_and_audits(output_root=args.output_root, legacy_feature_manifest=args.legacy_feature_manifest)
    if args.json:
        print(json.dumps(outputs, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status=ok run_tag={RUN_TAG} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
