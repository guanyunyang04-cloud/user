from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from daily_research.path_policy.forecast_training import FORECAST_LOSS_PROFILES
from daily_research.path_policy import v2_research_reset_baseline as v2
from daily_research.path_policy import v2_candidate_review_matrix as candidate_matrix


PYTHON = v2.PYTHON
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
DATA_LAKE_ROOT = PROJECT_ROOT / "daily_research/output/research_data_lake"
ACTIVE_MANIFEST = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
RUN_TAG = "v2_high_return_model_discovery_20260604_01"
RESEARCH_PROGRAM = v2.RESEARCH_PROGRAM
STUDY_FAMILY = "v2_high_return_model_discovery"

TRADITIONAL_BAOSTOCK_V2_1_DATASET_ID = "policy_input_bundle__2082fee5bb1760972d8c9012"
SAME_PERIOD_POOL_VIEW_ID = "policy_pool_view__d7a56d5164470b590e4f5a40"
LONG_HISTORY_POOL_VIEW_ID = "policy_pool_view__eb690dd0becc330f029c21bd"
SMOKE32_POOL_VIEW_ID = "policy_pool_view__04f11f51d7c0c7ae32038147"
SMOKE32_MEMMAP_MANIFEST = (
    PROJECT_ROOT
    / "daily_research/output/path_policy/data_expansion/traditional_baostock_v2_1_augmented_memmap_smoke32_20260604_01/forecast_dataset_manifest.json"
)
FEATURE_PROFILE = "raw_kline_context_v2_tradeable_local_state_industry_metrics_v1"
POOL_NAME = "rolling_liquid500_tradeable_mainboard_v2"
BENCHMARK = "000300.SH"
HORIZON_GRID = "1,2,3,5,8,10,15,20,30"
OUTPUT_PROFILE = "decision_utility_v1"
LOSS_PROFILE = "horizon_30d_soft_penalty_v1"
DEFAULT_LOSS_PROFILES = (LOSS_PROFILE,)
SELECTION_PROFILE = "decision_utility"
SCORE_COLUMN = "pred_decision_score"
SMALL_CAPITAL_GATE_ID = candidate_matrix.SMALL_CAPITAL_GATE_ID
SCOUT_MATRIX_ENTRY_EXCESS_SHARPE_MIN = 1.0
SCOUT_MATRIX_ENTRY_POSITIVE_MONTH_RATIO_MIN = 0.55

SCOUT_SEED = 7
CONFIRM_SEEDS = (7, 11, 19)
DEFAULT_MODEL_FAMILIES = (
    "gru_sequence_static_context",
    "hybrid_expert_fusion_static_context",
    "regime_routed_multi_expert_horizon_v1",
)


@dataclass(frozen=True)
class DataSpec:
    key: str
    dataset_id: str
    pool_view_id: str
    start_date: str
    end_date: str
    split_keys: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class SplitSpec:
    key: str
    train_start_year: int
    train_end_year: int
    validation_year: int
    test_year: int
    description: str


@dataclass(frozen=True)
class ModelSpec:
    family: str
    alias: str
    hidden_dim: int
    gru_layers: int
    transformer_layers: int
    transformer_heads: int
    patch_sizes: str


DATA_SPECS: dict[str, DataSpec] = {
    "same_period_augmented": DataSpec(
        key="same_period_augmented",
        dataset_id=TRADITIONAL_BAOSTOCK_V2_1_DATASET_ID,
        pool_view_id=SAME_PERIOD_POOL_VIEW_ID,
        start_date="20180101",
        end_date="20241231",
        split_keys=("same",),
        description="2018-2024 augmented same-period A/B substrate",
    ),
    "same_period_augmented_pilot": DataSpec(
        key="same_period_augmented_pilot",
        dataset_id=TRADITIONAL_BAOSTOCK_V2_1_DATASET_ID,
        pool_view_id=SAME_PERIOD_POOL_VIEW_ID,
        start_date="20220101",
        end_date="20241231",
        split_keys=("pilot",),
        description="2022-2024 full-pool augmented pilot substrate, throughput-only",
    ),
    "long_history_augmented": DataSpec(
        key="long_history_augmented",
        dataset_id=TRADITIONAL_BAOSTOCK_V2_1_DATASET_ID,
        pool_view_id=LONG_HISTORY_POOL_VIEW_ID,
        start_date="20170101",
        end_date="20241231",
        split_keys=("a", "b", "c", "long"),
        description="2017-2024 benchmark-covered augmented long-history substrate",
    ),
    "smoke32_augmented": DataSpec(
        key="smoke32_augmented",
        dataset_id=TRADITIONAL_BAOSTOCK_V2_1_DATASET_ID,
        pool_view_id=SMOKE32_POOL_VIEW_ID,
        start_date="20180101",
        end_date="20241231",
        split_keys=("same",),
        description="32-symbol augmented smoke substrate, pipeline-only",
    ),
}

SPLIT_SPECS: dict[str, SplitSpec] = {
    "same": SplitSpec(
        key="same",
        train_start_year=2019,
        train_end_year=2022,
        validation_year=2023,
        test_year=2024,
        description="current v2 same-period A/B split",
    ),
    "pilot": SplitSpec(
        key="pilot",
        train_start_year=2022,
        train_end_year=2022,
        validation_year=2023,
        test_year=2024,
        description="short full-pool pilot split for augmented throughput diagnostics",
    ),
    "a": SplitSpec(
        key="a",
        train_start_year=2017,
        train_end_year=2020,
        validation_year=2021,
        test_year=2022,
        description="early walk-forward split",
    ),
    "b": SplitSpec(
        key="b",
        train_start_year=2018,
        train_end_year=2021,
        validation_year=2022,
        test_year=2023,
        description="middle walk-forward split",
    ),
    "c": SplitSpec(
        key="c",
        train_start_year=2019,
        train_end_year=2022,
        validation_year=2023,
        test_year=2024,
        description="current test-year walk-forward split",
    ),
    "long": SplitSpec(
        key="long",
        train_start_year=2017,
        train_end_year=2022,
        validation_year=2023,
        test_year=2024,
        description="long-history train, current validation/test split",
    ),
}

MODEL_SPECS: dict[str, ModelSpec] = {
    "gru_sequence_static_context": ModelSpec(
        family="gru_sequence_static_context",
        alias="gru",
        hidden_dim=192,
        gru_layers=2,
        transformer_layers=4,
        transformer_heads=6,
        patch_sizes="4,20",
    ),
    "hybrid_expert_fusion_static_context": ModelSpec(
        family="hybrid_expert_fusion_static_context",
        alias="hybrid",
        hidden_dim=192,
        gru_layers=2,
        transformer_layers=4,
        transformer_heads=6,
        patch_sizes="4,20",
    ),
    "regime_routed_multi_expert_horizon_v1": ModelSpec(
        family="regime_routed_multi_expert_horizon_v1",
        alias="regime_routed",
        hidden_dim=256,
        gru_layers=2,
        transformer_layers=4,
        transformer_heads=8,
        patch_sizes="4,10,20",
    ),
}


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


def _parse_csv_strings(raw: str | tuple[str, ...] | list[str], default: tuple[str, ...]) -> tuple[str, ...]:
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


_LOSS_PROFILE_ALIASES = {
    "horizon_30d_soft_penalty_v1": "h30soft",
    "score_monthly_robust_v1": "monthly_robust",
    "risk_drawdown_reweighted_v1": "risk_dd",
    "horizon_entropy_regularized_v1": "hentropy",
    "topn_excess_rank_v1": "topn",
    "score_to_weight_proxy_v1": "s2w",
    "bad_month_aware_v1": "badmonth",
}


def _loss_profile_alias(loss_profile: str) -> str:
    value = str(loss_profile or "").strip()
    if value in _LOSS_PROFILE_ALIASES:
        return _LOSS_PROFILE_ALIASES[value]
    return value.replace("_v1", "").replace("horizon_", "h").replace("score_", "s").replace("risk_", "r")[:48]


def _parse_loss_profiles(
    raw: str | tuple[str, ...] | list[str] | None,
    default: tuple[str, ...] = DEFAULT_LOSS_PROFILES,
) -> tuple[str, ...]:
    profiles = _parse_csv_strings(raw or (), default)
    unknown = sorted(set(profiles) - set(FORECAST_LOSS_PROFILES))
    if unknown:
        raise ValueError(f"unknown_loss_profiles:{unknown}")
    return profiles


def _parse_seeds(raw: str | tuple[int, ...] | list[int] | None, default: tuple[int, ...] = (SCOUT_SEED,)) -> tuple[int, ...]:
    if raw is None:
        return tuple(default)
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        values = [str(item) for item in raw]
    parsed = tuple(int(float(item)) for item in values)
    return tuple(dict.fromkeys(parsed or default))


def _validate_single_seed_policy(seeds: tuple[int, ...], *, allow_single_seed_scout: bool) -> None:
    if len(seeds) == 1 and not allow_single_seed_scout:
        raise ValueError("single_seed_scout_requires_explicit_allow_single_seed_scout")


def _root(output_root: str | Path | None = None, run_tag: str = RUN_TAG) -> Path:
    return Path(output_root) if output_root is not None else STUDIES_ROOT / str(run_tag)


def _run_tag_suffix(run_tag: str) -> str:
    match = re.search(r"(\d{8}_\d+)$", str(run_tag or ""))
    return match.group(1) if match else "20260604_01"


def _tag(
    *,
    data_key: str,
    split_key: str,
    model_family: str,
    seed: int,
    run_tag: str = RUN_TAG,
    loss_profile: str = LOSS_PROFILE,
    include_loss_alias: bool = False,
) -> str:
    model_alias = MODEL_SPECS[model_family].alias
    data_alias = {
        "same_period_augmented": "same",
        "same_period_augmented_pilot": "same_pilot",
        "long_history_augmented": "long",
        "smoke32_augmented": "smoke32",
    }.get(data_key, str(data_key).replace("_augmented", ""))
    loss_part = f"{_loss_profile_alias(loss_profile)}_" if bool(include_loss_alias) else ""
    return f"mh_v2_hrd_{loss_part}{data_alias}_{split_key}_{model_alias}_seed{int(seed)}_{_run_tag_suffix(run_tag)}"


def _bridge_tag(forecast_tag: str) -> str:
    return f"v2_hr_bridge_{forecast_tag}"


def _matrix_tag(forecast_tag: str) -> str:
    return f"v2_hr_matrix_{forecast_tag}"


def _forecast_command(
    *,
    tag: str,
    data_spec: DataSpec,
    split_spec: SplitSpec,
    model_spec: ModelSpec,
    seed: int,
    loss_profile: str,
    epochs: int,
    min_epochs: int,
    patience: int,
    memmap_manifest: str | Path | None,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
) -> list[str]:
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
        data_spec.dataset_id,
        "--pool-name",
        POOL_NAME,
        "--pool-view-id",
        data_spec.pool_view_id,
        "--benchmark",
        BENCHMARK,
        "--start-date",
        data_spec.start_date,
        "--end-date",
        data_spec.end_date,
        "--max-universe-size",
        "0",
        "--execution-mode",
        "next_open",
        "--forecast-dataset-mode",
        "memmap",
    ]
    if memmap_manifest is not None:
        command.extend(["--forecast-memmap-manifest", str(memmap_manifest)])
    command.extend(
        [
            "--forecast-train-start-year",
            str(int(split_spec.train_start_year)),
            "--forecast-train-end-year",
            str(int(split_spec.train_end_year)),
            "--forecast-validation-year",
            str(int(split_spec.validation_year)),
            "--forecast-test-year",
            str(int(split_spec.test_year)),
            "--forecast-model-families",
            model_spec.family,
            "--forecast-feature-profile",
            FEATURE_PROFILE,
            "--forecast-include-static-context",
            "--forecast-max-feature-columns",
            "192",
            "--forecast-max-samples-per-role",
            str(max(int(max_samples_per_role), 0)),
            "--forecast-max-samples-per-date-per-role",
            str(max(int(max_samples_per_date_per_role), 0)),
            "--forecast-cumulative-horizons",
            HORIZON_GRID,
            "--forecast-horizon",
            "30",
            "--forecast-output-profile",
            OUTPUT_PROFILE,
            "--forecast-loss-profile",
            str(loss_profile),
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
            str(int(epochs)),
            "--forecast-min-epochs",
            str(int(min_epochs)),
            "--forecast-early-stop-patience",
            str(int(patience)),
            "--forecast-checkpoint-every-n-epochs",
            "4",
            "--forecast-device",
            "cuda",
            "--forecast-amp",
            "--forecast-hidden-dim",
            str(int(model_spec.hidden_dim)),
            "--forecast-gru-layers",
            str(int(model_spec.gru_layers)),
            "--forecast-transformer-layers",
            str(int(model_spec.transformer_layers)),
            "--forecast-transformer-heads",
            str(int(model_spec.transformer_heads)),
            "--forecast-patch-sizes",
            str(model_spec.patch_sizes),
        ]
    )
    return command


def _quick_bridge_command(
    *,
    forecast_tag: str,
    data_spec: DataSpec,
    output_root: Path,
) -> list[str]:
    return [
        PYTHON,
        "-m",
        "daily_research.path_policy.v2_score_backtest_bridge",
        "--run-tag",
        _bridge_tag(forecast_tag),
        "--output-root",
        str(output_root / "bridges" / _bridge_tag(forecast_tag)),
        "--source-anchor-run-tag",
        forecast_tag,
        "--seed-study-tags",
        forecast_tag,
        "--dataset-id",
        data_spec.dataset_id,
        "--pool-view-id",
        data_spec.pool_view_id,
        "--feature-profile",
        FEATURE_PROFILE,
        "--score-column",
        SCORE_COLUMN,
        "--role",
        "test",
        "--allow-partial-seeds",
        "--run-backtest",
        "--holding-count",
        "20",
        "--max-weight",
        "0.12",
        "--rebalance-freq",
        "10d",
        "--rebalance-offset-mode",
        "all",
        "--transaction-cost-bps",
        "10",
        "--slippage-bps",
        "5",
        "--sell-tax-bps",
        "10",
        "--json",
    ]


def _candidate_matrix_command(*, forecast_tag: str, data_spec: DataSpec, output_root: Path) -> list[str]:
    bridge_run_tag = _bridge_tag(forecast_tag)
    return [
        PYTHON,
        "-m",
        "daily_research.path_policy.v2_candidate_review_matrix",
        "--run-tag",
        _matrix_tag(forecast_tag),
        "--source-bridge-run-tag",
        bridge_run_tag,
        "--source-bridge-root",
        str(output_root / "bridges" / bridge_run_tag),
        "--output-root",
        str(output_root / "matrices" / _matrix_tag(forecast_tag)),
        "--dataset-id",
        data_spec.dataset_id,
        "--pool-view-id",
        data_spec.pool_view_id,
        "--holding-counts",
        "10,20,30",
        "--max-weights",
        "0.08,0.12,0.16",
        "--rebalance-freqs",
        "5d,10d,20d",
        "--rebalance-offset-modes",
        "all",
        "--transaction-cost-bps-values",
        "10",
        "--slippage-bps-values",
        "5",
        "--sell-tax-bps-values",
        "10",
        "--market-regime-filter-modes",
        "off",
        "--run-backtests",
        "--json",
    ]


def build_forecast_tasks(
    *,
    output_root: str | Path | None = None,
    run_tag: str = RUN_TAG,
    datasets: str | tuple[str, ...] | list[str] = ("same_period_augmented", "long_history_augmented"),
    splits: str | tuple[str, ...] | list[str] = (),
    model_families: str | tuple[str, ...] | list[str] = DEFAULT_MODEL_FAMILIES,
    loss_profiles: str | tuple[str, ...] | list[str] | None = DEFAULT_LOSS_PROFILES,
    seeds: str | tuple[int, ...] | list[int] | None = (SCOUT_SEED,),
    epochs: int = 16,
    min_epochs: int = 6,
    patience: int = 5,
    allow_single_seed_scout: bool = True,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
) -> list[dict[str, Any]]:
    root = _root(output_root, run_tag)
    resolved_datasets = _parse_csv_strings(datasets, ("same_period_augmented", "long_history_augmented"))
    resolved_splits = _parse_csv_strings(splits, ())
    resolved_models = _parse_csv_strings(model_families, DEFAULT_MODEL_FAMILIES)
    resolved_loss_profiles = _parse_loss_profiles(loss_profiles, DEFAULT_LOSS_PROFILES)
    resolved_seeds = _parse_seeds(seeds)
    _validate_single_seed_policy(resolved_seeds, allow_single_seed_scout=allow_single_seed_scout)
    unknown_datasets = sorted(set(resolved_datasets) - set(DATA_SPECS))
    unknown_models = sorted(set(resolved_models) - set(MODEL_SPECS))
    unknown_splits = sorted(set(resolved_splits) - set(SPLIT_SPECS))
    if unknown_datasets:
        raise ValueError(f"unknown_datasets:{unknown_datasets}")
    if unknown_models:
        raise ValueError(f"unknown_model_families:{unknown_models}")
    if unknown_splits:
        raise ValueError(f"unknown_splits:{unknown_splits}")

    tasks: list[dict[str, Any]] = []
    include_loss_alias = len(resolved_loss_profiles) > 1 or any(profile != LOSS_PROFILE for profile in resolved_loss_profiles)
    for data_key in resolved_datasets:
        data_spec = DATA_SPECS[data_key]
        split_keys = resolved_splits or data_spec.split_keys
        split_keys = tuple(key for key in split_keys if key in data_spec.split_keys)
        if not split_keys:
            raise ValueError(f"no_compatible_splits_for_dataset:{data_key}")
        for split_key in split_keys:
            split_spec = SPLIT_SPECS[split_key]
            group_anchor_manifest: Path | None = SMOKE32_MEMMAP_MANIFEST if data_key == "smoke32_augmented" else None
            group_anchor_tag = ""
            for model_family in resolved_models:
                model_spec = MODEL_SPECS[model_family]
                for loss_profile in resolved_loss_profiles:
                    for seed in resolved_seeds:
                        tag = _tag(
                            data_key=data_key,
                            split_key=split_key,
                            model_family=model_family,
                            seed=int(seed),
                            run_tag=run_tag,
                            loss_profile=loss_profile,
                            include_loss_alias=include_loss_alias,
                        )
                        study_dir = STUDIES_ROOT / tag
                        if data_key == "smoke32_augmented":
                            memmap_manifest = SMOKE32_MEMMAP_MANIFEST
                            group_anchor_tag = "prebuilt_smoke32_augmented_memmap"
                        elif group_anchor_manifest is None:
                            memmap_manifest = None
                            group_anchor_tag = tag
                            group_anchor_manifest = study_dir / "forecast_dataset_manifest.json"
                        else:
                            memmap_manifest = group_anchor_manifest
                        candidate_prefix = f"{_loss_profile_alias(loss_profile)}_" if include_loss_alias else ""
                        task = {
                            "tag": tag,
                            "candidate_id": f"{candidate_prefix}{data_key}_{split_key}_{model_spec.alias}_seed{int(seed)}",
                            "dataset_key": data_key,
                            "dataset_id": data_spec.dataset_id,
                            "pool_view_id": data_spec.pool_view_id,
                            "feature_profile": FEATURE_PROFILE,
                            "split_key": split_key,
                            "split": {
                                "train_start_year": split_spec.train_start_year,
                                "train_end_year": split_spec.train_end_year,
                                "validation_year": split_spec.validation_year,
                                "test_year": split_spec.test_year,
                            },
                            "model_family": model_family,
                            "loss_profile": loss_profile,
                            "output_profile": OUTPUT_PROFILE,
                            "selection_profile": SELECTION_PROFILE,
                            "seed": int(seed),
                            "epochs": int(epochs),
                            "min_epochs": int(min_epochs),
                            "patience": int(patience),
                            "study_dir": str(study_dir),
                            "stdout": str(root / "logs" / f"{tag}_stdout.log"),
                            "stderr": str(root / "logs" / f"{tag}_stderr.log"),
                            "forecast_memmap_manifest": str(group_anchor_manifest),
                            "builds_forecast_memmap": memmap_manifest is None,
                            "depends_on_manifest_task_tag": "" if memmap_manifest is None else group_anchor_tag,
                            "reuses_forecast_memmap_manifest": memmap_manifest is not None,
                            "command": _forecast_command(
                                tag=tag,
                                data_spec=data_spec,
                                split_spec=split_spec,
                                model_spec=model_spec,
                                seed=int(seed),
                                loss_profile=loss_profile,
                                epochs=int(epochs),
                                min_epochs=int(min_epochs),
                                patience=int(patience),
                                memmap_manifest=memmap_manifest,
                                max_samples_per_role=max_samples_per_role,
                                max_samples_per_date_per_role=max_samples_per_date_per_role,
                            ),
                            "quick_bridge_command": _quick_bridge_command(
                                forecast_tag=tag,
                                data_spec=data_spec,
                                output_root=root,
                            ),
                            "candidate_matrix_command": _candidate_matrix_command(
                                forecast_tag=tag,
                                data_spec=data_spec,
                                output_root=root,
                            ),
                            "research_program": RESEARCH_PROGRAM,
                            "study_family": STUDY_FAMILY,
                            "evidence_grade": (
                                "smoke_only"
                                if data_key == "smoke32_augmented"
                                else (
                                    "pilot_only"
                                    if data_key == "same_period_augmented_pilot"
                                    else ("scout_only" if len(resolved_seeds) == 1 else "finalist_confirmation_input")
                                )
                            ),
                            "smoke_only": data_key == "smoke32_augmented",
                            "pilot_only": data_key == "same_period_augmented_pilot",
                            "single_seed_scout_only": len(resolved_seeds) == 1,
                            "max_samples_per_role": int(max_samples_per_role),
                            "max_samples_per_date_per_role": int(max_samples_per_date_per_role),
                            "shadow_only": True,
                            "promotion_allowed": False,
                            "active_execution_strategy_expected_diff": "none",
                        }
                        tasks.append(task)
    return tasks


def write_task_list(
    output_root: str | Path | None = None,
    *,
    run_tag: str = RUN_TAG,
    datasets: str | tuple[str, ...] | list[str] = ("same_period_augmented", "long_history_augmented"),
    splits: str | tuple[str, ...] | list[str] = (),
    model_families: str | tuple[str, ...] | list[str] = DEFAULT_MODEL_FAMILIES,
    loss_profiles: str | tuple[str, ...] | list[str] | None = DEFAULT_LOSS_PROFILES,
    seeds: str | tuple[int, ...] | list[int] | None = (SCOUT_SEED,),
    epochs: int = 16,
    min_epochs: int = 6,
    patience: int = 5,
    allow_single_seed_scout: bool = True,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
    enforce_active_artifact_clean: bool = True,
) -> Path:
    if enforce_active_artifact_clean and _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _root(output_root, run_tag)
    tasks = build_forecast_tasks(
        output_root=root,
        run_tag=run_tag,
        datasets=datasets,
        splits=splits,
        model_families=model_families,
        loss_profiles=loss_profiles,
        seeds=seeds,
        epochs=epochs,
        min_epochs=min_epochs,
        patience=patience,
        allow_single_seed_scout=allow_single_seed_scout,
        max_samples_per_role=max_samples_per_role,
        max_samples_per_date_per_role=max_samples_per_date_per_role,
    )
    resolved_seeds = _parse_seeds(seeds)
    resolved_loss_profiles = _parse_loss_profiles(loss_profiles, DEFAULT_LOSS_PROFILES)
    payload = {
        "schema_version": 1,
        "run_tag": str(run_tag),
        "research_program": RESEARCH_PROGRAM,
        "study_family": STUDY_FAMILY,
        "created_at": _now(),
        "stage": "high_return_model_discovery",
        "objective": "single-seed multi-split high-return model scout before three-seed finalist confirmation",
        "dataset_ids": sorted({task["dataset_id"] for task in tasks}),
        "pool_view_ids": sorted({task["pool_view_id"] for task in tasks}),
        "feature_profile": FEATURE_PROFILE,
        "loss_profile": resolved_loss_profiles[0] if len(resolved_loss_profiles) == 1 else "",
        "loss_profiles": list(resolved_loss_profiles),
        "output_profile": OUTPUT_PROFILE,
        "selection_profile": SELECTION_PROFILE,
        "seeds": list(resolved_seeds),
        "single_seed_scout_only": len(resolved_seeds) == 1,
        "evidence_grade_input_or_model_quality_pass": len(resolved_seeds) >= 3,
        "training_task_count": len(tasks),
        "quick_bridge_task_count": len(tasks),
        "candidate_matrix_task_count": len(tasks),
        "max_samples_per_role": int(max_samples_per_role),
        "max_samples_per_date_per_role": int(max_samples_per_date_per_role),
        "smoke_only_task_count": int(sum(1 for task in tasks if bool(task.get("smoke_only", False)))),
        "pilot_only_task_count": int(sum(1 for task in tasks if bool(task.get("pilot_only", False)))),
        "training_tasks": tasks,
        "quick_bridge_tasks": [
            {
                "tag": task["tag"],
                "candidate_id": task["candidate_id"],
                "loss_profile": task["loss_profile"],
                "output_profile": task["output_profile"],
                "selection_profile": task["selection_profile"],
                "bridge_run_tag": _bridge_tag(str(task["tag"])),
                "command": task["quick_bridge_command"],
                "single_seed_scout_only": bool(task["single_seed_scout_only"]),
                "smoke_only": bool(task.get("smoke_only", False)),
                "pilot_only": bool(task.get("pilot_only", False)),
                "promotion_allowed": False,
            }
            for task in tasks
        ],
        "candidate_matrix_tasks": [
            {
                "tag": task["tag"],
                "candidate_id": task["candidate_id"],
                "loss_profile": task["loss_profile"],
                "output_profile": task["output_profile"],
                "selection_profile": task["selection_profile"],
                "matrix_run_tag": _matrix_tag(str(task["tag"])),
                "command": task["candidate_matrix_command"],
                "single_seed_scout_only": bool(task["single_seed_scout_only"]),
                "smoke_only": bool(task.get("smoke_only", False)),
                "pilot_only": bool(task.get("pilot_only", False)),
                "promotion_allowed": False,
            }
            for task in tasks
        ],
        "high_return_scout_policy": {
            "forecast_first": True,
            "single_seed_allowed_only_as_scout": True,
            "multi_split_before_multi_seed": True,
            "three_seed_reserved_for_finalists": True,
            "promotion_gate_deferred": True,
            "research_gate_id": SMALL_CAPITAL_GATE_ID,
            "benchmark": BENCHMARK,
            "matrix_entry_thresholds": {
                "excess_annual_return_gt": 0.0,
                "excess_sharpe_gt": 0.0,
                "excess_sharpe_ge": SCOUT_MATRIX_ENTRY_EXCESS_SHARPE_MIN,
                "positive_month_ratio_ge": SCOUT_MATRIX_ENTRY_POSITIVE_MONTH_RATIO_MIN,
            },
            "research_grade_thresholds": dict(candidate_matrix.SMALL_CAPITAL_RESEARCH_THRESHOLDS),
            "quick_bridge_default": {
                "holding_count": 20,
                "max_weight": 0.12,
                "rebalance_freq": "10d",
                "rebalance_offset_mode": "all",
                "costs_bps": {"transaction": 10, "slippage": 5, "sell_tax": 10},
            },
            "candidate_matrix_grid": {
                "holding_counts": [10, 20, 30],
                "max_weights": [0.08, 0.12, 0.16],
                "rebalance_freqs": ["5d", "10d", "20d"],
                "costs_bps": {"transaction": 10, "slippage": 5, "sell_tax": 10},
            },
        },
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
            "paper_live_broker_allowed": False,
        },
    }
    path = root / "v2_high_return_model_discovery_task_list.json"
    _write_json(path, payload)
    return path


def _run_command(command: list[str], *, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        proc = subprocess.run(command, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True, check=False)
    return {"returncode": int(proc.returncode), "stdout": str(stdout_path), "stderr": str(stderr_path)}


def run_forecast_tasks(
    *,
    output_root: str | Path | None = None,
    run_tag: str = RUN_TAG,
    datasets: str | tuple[str, ...] | list[str] = ("same_period_augmented", "long_history_augmented"),
    splits: str | tuple[str, ...] | list[str] = (),
    model_families: str | tuple[str, ...] | list[str] = DEFAULT_MODEL_FAMILIES,
    loss_profiles: str | tuple[str, ...] | list[str] | None = DEFAULT_LOSS_PROFILES,
    seeds: str | tuple[int, ...] | list[int] | None = (SCOUT_SEED,),
    epochs: int = 16,
    min_epochs: int = 6,
    patience: int = 5,
    allow_single_seed_scout: bool = False,
    skip_existing: bool = True,
    max_tasks: int = 0,
    max_samples_per_role: int = 0,
    max_samples_per_date_per_role: int = 0,
) -> dict[str, Any]:
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _root(output_root, run_tag)
    root.mkdir(parents=True, exist_ok=True)
    tasks = build_forecast_tasks(
        output_root=root,
        run_tag=run_tag,
        datasets=datasets,
        splits=splits,
        model_families=model_families,
        loss_profiles=loss_profiles,
        seeds=seeds,
        epochs=epochs,
        min_epochs=min_epochs,
        patience=patience,
        allow_single_seed_scout=allow_single_seed_scout,
        max_samples_per_role=max_samples_per_role,
        max_samples_per_date_per_role=max_samples_per_date_per_role,
    )
    task_list_path = write_task_list(
        output_root=root,
        run_tag=run_tag,
        datasets=datasets,
        splits=splits,
        model_families=model_families,
        loss_profiles=loss_profiles,
        seeds=seeds,
        epochs=epochs,
        min_epochs=min_epochs,
        patience=patience,
        allow_single_seed_scout=allow_single_seed_scout,
        max_samples_per_role=max_samples_per_role,
        max_samples_per_date_per_role=max_samples_per_date_per_role,
        enforce_active_artifact_clean=False,
    )
    completed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    launched = 0
    for task in tasks:
        if int(max_tasks) > 0 and launched >= int(max_tasks):
            results.append({"tag": task.get("tag", ""), "status": "deferred_by_max_tasks"})
            continue
        summary_path = Path(str(task["study_dir"])) / "study_summary.json"
        if skip_existing and summary_path.exists():
            completed.append(str(task["tag"]))
            results.append({"tag": task["tag"], "status": "skipped_existing", "study_summary_json": str(summary_path)})
            continue
        launched += 1
        result = _run_command(
            list(task["command"]),
            stdout_path=Path(str(task["stdout"])),
            stderr_path=Path(str(task["stderr"])),
        )
        row = {
            "tag": task["tag"],
            "candidate_id": task["candidate_id"],
            "loss_profile": task["loss_profile"],
            **result,
        }
        results.append(row)
        if int(result["returncode"]) == 0:
            completed.append(str(task["tag"]))
        else:
            failed.append(str(task["tag"]))
            break
    payload = {
        "schema_version": 1,
        "status": "completed" if not failed else "failed",
        "run_tag": str(run_tag),
        "task_list_path": str(task_list_path),
        "completed_tags": completed,
        "failed_tags": failed,
        "launched_task_count": int(launched),
        "max_tasks": int(max_tasks),
        "loss_profiles": list(_parse_loss_profiles(loss_profiles, DEFAULT_LOSS_PROFILES)),
        "max_samples_per_role": int(max_samples_per_role),
        "max_samples_per_date_per_role": int(max_samples_per_date_per_role),
        "results": results,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_high_return_model_discovery_forecast_run_summary.json", payload)
    return payload


def _load_task_list_payload(root: Path, task_list_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    path = Path(task_list_path) if task_list_path is not None else root / "v2_high_return_model_discovery_task_list.json"
    payload = _read_json(path)
    return payload, path


def _task_forecast_completed(task: dict[str, Any]) -> bool:
    summary = _read_json(Path(str(task.get("study_dir", ""))) / "study_summary.json")
    return str(summary.get("status", "")).strip() == "completed"


def run_quick_bridge_tasks(
    *,
    output_root: str | Path | None = None,
    run_tag: str = RUN_TAG,
    task_list_path: str | Path | None = None,
    skip_existing: bool = True,
    require_completed_forecast: bool = True,
    max_tasks: int = 0,
) -> dict[str, Any]:
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _root(output_root, run_tag)
    payload, path = _load_task_list_payload(root, task_list_path)
    tasks = list(payload.get("training_tasks", []) or [])
    completed: list[str] = []
    skipped: list[dict[str, Any]] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    launched = 0
    for task in tasks:
        tag = str(task.get("tag", ""))
        if int(max_tasks) > 0 and launched >= int(max_tasks):
            skipped.append({"tag": tag, "reason": "deferred_by_max_tasks"})
            continue
        report_path = root / "bridges" / _bridge_tag(tag) / "v2_score_backtest_bridge_report.json"
        if require_completed_forecast and not _task_forecast_completed(task):
            skipped.append({"tag": tag, "reason": "awaiting_completed_forecast"})
            continue
        if skip_existing and report_path.exists():
            completed.append(tag)
            results.append({"tag": tag, "status": "skipped_existing", "report_json": str(report_path)})
            continue
        launched += 1
        result = _run_command(
            list(task.get("quick_bridge_command", [])),
            stdout_path=root / "logs" / f"{_bridge_tag(tag)}_stdout.log",
            stderr_path=root / "logs" / f"{_bridge_tag(tag)}_stderr.log",
        )
        row = {
            "tag": tag,
            "candidate_id": str(task.get("candidate_id", "")),
            "loss_profile": str(task.get("loss_profile", "")),
            **result,
            "report_json": str(report_path),
        }
        results.append(row)
        if int(result["returncode"]) == 0:
            completed.append(tag)
        else:
            failed.append(tag)
            break
    status = "completed"
    if failed:
        status = "failed"
    elif not completed and skipped:
        status = "awaiting_forecast_results"
    elif not tasks:
        status = "missing_tasks"
    summary = {
        "schema_version": 1,
        "status": status,
        "run_tag": str(run_tag),
        "task_list_path": str(path),
        "completed_tags": completed,
        "skipped": skipped,
        "failed_tags": failed,
        "launched_task_count": int(launched),
        "max_tasks": int(max_tasks),
        "loss_profiles": list(payload.get("loss_profiles", []) or []),
        "results": results,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_high_return_model_discovery_quick_bridge_run_summary.json", summary)
    return summary


def run_candidate_matrix_tasks(
    *,
    output_root: str | Path | None = None,
    run_tag: str = RUN_TAG,
    task_list_path: str | Path | None = None,
    skip_existing: bool = True,
    shortlist_only: bool = True,
    max_tasks: int = 0,
) -> dict[str, Any]:
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_MANIFEST} has uncommitted diff.")
    root = _root(output_root, run_tag)
    payload, path = _load_task_list_payload(root, task_list_path)
    tasks = list(payload.get("training_tasks", []) or [])
    allowed_tags: set[str] | None = None
    if shortlist_only:
        report = collect_high_return_report(output_root=root, run_tag=run_tag, task_list_path=path)
        allowed_tags = {
            str(row.get("tag", ""))
            for row in report.get("leaderboard", [])
            if _matrix_entry_candidate(row)
            and _has_forecast_result(row)
            and not bool(row.get("smoke_only", False))
            and not bool(row.get("pilot_only", False))
        }
    completed: list[str] = []
    skipped: list[dict[str, Any]] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []
    launched = 0
    for task in tasks:
        tag = str(task.get("tag", ""))
        if int(max_tasks) > 0 and launched >= int(max_tasks):
            skipped.append({"tag": tag, "reason": "deferred_by_max_tasks"})
            continue
        if allowed_tags is not None and tag not in allowed_tags:
            skipped.append({"tag": tag, "reason": "not_matrix_entry_candidate"})
            continue
        bridge_report = root / "bridges" / _bridge_tag(tag) / "v2_score_backtest_bridge_report.json"
        matrix_report = root / "matrices" / _matrix_tag(tag) / "v2_candidate_review_matrix_report.json"
        if not bridge_report.exists():
            skipped.append({"tag": tag, "reason": "missing_quick_bridge_report"})
            continue
        if skip_existing and matrix_report.exists():
            completed.append(tag)
            results.append({"tag": tag, "status": "skipped_existing", "report_json": str(matrix_report)})
            continue
        launched += 1
        result = _run_command(
            list(task.get("candidate_matrix_command", [])),
            stdout_path=root / "logs" / f"{_matrix_tag(tag)}_stdout.log",
            stderr_path=root / "logs" / f"{_matrix_tag(tag)}_stderr.log",
        )
        row = {
            "tag": tag,
            "candidate_id": str(task.get("candidate_id", "")),
            "loss_profile": str(task.get("loss_profile", "")),
            **result,
            "report_json": str(matrix_report),
        }
        results.append(row)
        if int(result["returncode"]) == 0:
            completed.append(tag)
        else:
            failed.append(tag)
            break
    status = "completed"
    if failed:
        status = "failed"
    elif not completed and shortlist_only:
        status = "awaiting_matrix_entry_candidates"
    elif not tasks:
        status = "missing_tasks"
    summary = {
        "schema_version": 1,
        "status": status,
        "run_tag": str(run_tag),
        "task_list_path": str(path),
        "matrix_entry_only": bool(shortlist_only),
        "completed_tags": completed,
        "skipped": skipped,
        "failed_tags": failed,
        "launched_task_count": int(launched),
        "max_tasks": int(max_tasks),
        "loss_profiles": list(payload.get("loss_profiles", []) or []),
        "results": results,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_high_return_model_discovery_candidate_matrix_run_summary.json", summary)
    return summary


def _metric(payload: dict[str, Any], section: str, key: str) -> float:
    training = dict(payload.get("training_summary", {}) or {})
    section_payload = dict(payload.get(section, {}) or training.get(section, {}) or {})
    try:
        value = float(section_payload.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _summary_section(summary: dict[str, Any]) -> dict[str, Any]:
    training = dict(summary.get("training_summary", {}) or {})
    return training if training else summary


def _load_bridge_metrics(output_root: Path, forecast_tag: str) -> dict[str, Any]:
    report = _read_json(output_root / "bridges" / _bridge_tag(forecast_tag) / "v2_score_backtest_bridge_report.json")
    shared = dict(report.get("shared_backtest", {}) or {})
    artifacts = dict(shared.get("artifacts", {}) or {})
    return {
        "bridge_status": str(report.get("status", "")),
        "backtest_status": str(shared.get("status", "")),
        "core_metrics": dict(artifacts.get("core_metrics", {}) or {}),
        "monthly_backtest_diagnostics": dict(artifacts.get("monthly_backtest_diagnostics", {}) or {}),
    }


def _load_matrix_metrics(output_root: Path, forecast_tag: str) -> dict[str, Any]:
    matrix_root = output_root / "matrices" / _matrix_tag(forecast_tag)
    report = _read_json(matrix_root / "v2_candidate_review_matrix_report.json")
    small_report = _read_json(matrix_root / f"{SMALL_CAPITAL_GATE_ID}_report.json")
    summary = dict(report.get("summary", {}) or {})
    small_summary = dict(small_report.get("summary", {}) or {})
    best = dict(
        small_summary.get("best_by_balanced_return_score")
        or summary.get("best_by_small_capital_balanced_return_score")
        or {}
    )
    research_grade = dict(
        small_summary.get("best_research_grade_candidate")
        or summary.get("best_small_capital_research_grade_candidate")
        or {}
    )
    return {
        "matrix_status": str(report.get("status", "")),
        "matrix_report_json": str(matrix_root / "v2_candidate_review_matrix_report.json") if report else "",
        "small_capital_gate_id": SMALL_CAPITAL_GATE_ID,
        "small_capital_positive_transfer_count": int(
            small_summary.get("positive_transfer_count", summary.get("small_capital_positive_transfer_count", 0)) or 0
        ),
        "small_capital_research_grade_candidate_count": int(
            small_summary.get(
                "research_grade_candidate_count",
                summary.get("small_capital_research_grade_candidate_count", 0),
            )
            or 0
        ),
        "best_small_capital_variant_id": str(best.get("variant_id", "")),
        "best_small_capital_balanced_return_score": _float_metric(best, "small_capital_balanced_return_score"),
        "best_small_capital_excess_annual_return": _float_metric(best, "excess_annual_return"),
        "best_small_capital_excess_sharpe": _float_metric(best, "excess_sharpe"),
        "best_small_capital_positive_month_ratio": _float_metric(best, "positive_month_ratio"),
        "best_small_capital_worst_month": _float_metric(
            best,
            "worst_monthly_excess_return",
            _float_metric(best, "worst_monthly_return"),
        ),
        "research_grade_small_capital_variant_id": str(research_grade.get("variant_id", "")),
    }


def _float_metric(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _has_forecast_result(row: dict[str, Any]) -> bool:
    return str(row.get("forecast_status", "")).strip() == "completed" or bool(str(row.get("evidence_verdict", "")).strip())


def _positive_transfer(row: dict[str, Any]) -> bool:
    return (
        str(row.get("backtest_status", "")).strip() == "completed"
        and _float_metric(row, "excess_annual_return") > 0.0
        and _float_metric(row, "excess_sharpe") > 0.0
    )


def _matrix_entry_candidate(row: dict[str, Any]) -> bool:
    return (
        _positive_transfer(row)
        and _float_metric(row, "excess_sharpe") >= SCOUT_MATRIX_ENTRY_EXCESS_SHARPE_MIN
        and _float_metric(row, "positive_month_ratio") >= SCOUT_MATRIX_ENTRY_POSITIVE_MONTH_RATIO_MIN
    )


def _score_row(row: dict[str, Any]) -> dict[str, float]:
    if not _has_forecast_result(row):
        return {
            "forecast_strength_score": 0.0,
            "return_potential_score": 0.0,
            "transfer_score": 0.0,
            "instability_penalty": 0.0,
            "repairability_score": 0.0,
            "high_return_score": 0.0,
        }
    rank_ic = _float_metric(row, "rank_ic")
    spread = _float_metric(row, "top_bottom_spread")
    hit = _float_metric(row, "hit_lift")
    annual_return = _float_metric(row, "annual_return")
    excess_annual_return = _float_metric(row, "excess_annual_return")
    excess_sharpe = _float_metric(row, "excess_sharpe")
    max_drawdown = _float_metric(row, "max_drawdown")
    positive_month_ratio = _float_metric(row, "positive_month_ratio")
    negative_month_count = _float_metric(row, "negative_month_count")
    worst_monthly_return = _float_metric(row, "worst_monthly_return")

    forecast_strength_score = 10.0 * rank_ic + 25.0 * spread + 2.0 * hit
    return_potential_score = 2.0 * max(excess_annual_return, 0.0) + 0.75 * max(annual_return, 0.0)
    transfer_score = 0.75 * max(excess_sharpe, 0.0) + 0.50 * max(positive_month_ratio, 0.0)
    instability_penalty = 0.0
    instability_penalty += max(0.0, abs(min(max_drawdown, 0.0)) - 0.35) * 2.0
    instability_penalty += max(0.0, negative_month_count - 5.0) * 0.15
    instability_penalty += max(0.0, abs(min(worst_monthly_return, 0.0)) - 0.12) * 1.5
    repairability_score = max(0.0, 0.25 - max(0.0, instability_penalty))
    high_return_score = forecast_strength_score + return_potential_score + transfer_score + repairability_score - instability_penalty
    return {
        "forecast_strength_score": forecast_strength_score,
        "return_potential_score": return_potential_score,
        "transfer_score": transfer_score,
        "instability_penalty": instability_penalty,
        "repairability_score": repairability_score,
        "high_return_score": high_return_score,
    }


def _split_consistency(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if _has_forecast_result(row)]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in completed:
        key = (
            str(row.get("dataset_key", "")),
            str(row.get("model_family", "")),
            str(row.get("loss_profile", "")),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset_key, model_family, loss_profile), items in groups.items():
        scores = [_float_metric(row, "high_return_score") for row in items]
        transfer_scores = [_float_metric(row, "transfer_score") for row in items]
        split_keys = [str(row.get("split_key", "")) for row in items]
        best = max(items, key=lambda row: _float_metric(row, "high_return_score"))
        worst = min(items, key=lambda row: _float_metric(row, "high_return_score"))
        out.append(
            {
                "dataset_key": dataset_key,
                "model_family": model_family,
                "loss_profile": loss_profile,
                "completed_split_count": len(items),
                "split_keys": split_keys,
                "positive_high_return_split_count": int(sum(1 for value in scores if value > 0.0)),
                "positive_transfer_split_count": int(sum(1 for row in items if bool(row.get("positive_transfer", False)))),
                "matrix_entry_split_count": int(sum(1 for row in items if bool(row.get("matrix_entry_candidate", False)))),
                "small_capital_research_grade_split_count": int(
                    sum(1 for row in items if int(row.get("small_capital_research_grade_candidate_count", 0) or 0) > 0)
                ),
                "high_return_score_mean": float(np.mean(scores)) if scores else 0.0,
                "high_return_score_min": float(np.min(scores)) if scores else 0.0,
                "transfer_score_mean": float(np.mean(transfer_scores)) if transfer_scores else 0.0,
                "best_split_key": str(best.get("split_key", "")),
                "best_tag": str(best.get("tag", "")),
                "worst_split_key": str(worst.get("split_key", "")),
                "worst_tag": str(worst.get("tag", "")),
            }
        )
    return sorted(out, key=lambda row: (_float_metric(row, "positive_transfer_split_count"), _float_metric(row, "high_return_score_mean")), reverse=True)


def _bad_month_issue_hint(row: dict[str, Any]) -> str:
    worst_month = _float_metric(row, "worst_monthly_return")
    max_drawdown = _float_metric(row, "max_drawdown")
    negative_count = _float_metric(row, "negative_month_count")
    if worst_month <= -0.12:
        return "deep_bad_month_preview"
    if max_drawdown <= -0.35:
        return "drawdown_stress_preview"
    if negative_count >= 4:
        return "frequent_bad_months_preview"
    if worst_month < 0.0:
        return "ordinary_bad_month_preview"
    return "no_bad_month_preview"


def _bad_month_preview(rows: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if str(row.get("backtest_status", "")).strip() == "completed"
        and (_float_metric(row, "negative_month_count") > 0.0 or _float_metric(row, "worst_monthly_return") < 0.0)
    ]
    candidates.sort(key=lambda row: (_float_metric(row, "worst_monthly_return"), -_float_metric(row, "negative_month_count")))
    preview: list[dict[str, Any]] = []
    for row in candidates[: int(limit)]:
        preview.append(
            {
                "tag": str(row.get("tag", "")),
                "candidate_id": str(row.get("candidate_id", "")),
                "dataset_key": str(row.get("dataset_key", "")),
                "loss_profile": str(row.get("loss_profile", "")),
                "split_key": str(row.get("split_key", "")),
                "model_family": str(row.get("model_family", "")),
                "negative_month_count": _float_metric(row, "negative_month_count"),
                "worst_monthly_return": _float_metric(row, "worst_monthly_return"),
                "max_drawdown": _float_metric(row, "max_drawdown"),
                "issue_hint": _bad_month_issue_hint(row),
            }
        )
    return preview


def _write_markdown(path: str | Path, report: dict[str, Any]) -> None:
    leaderboard = list(report.get("leaderboard", []) or [])[:10]
    shortlist = list(report.get("shortlist", []) or [])[:5]
    small_capital = dict(report.get("small_capital_gate_summary", {}) or {})
    lines = [
        "# V2 High Return Model Discovery",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Completed forecast count: `{report.get('completed_forecast_count', 0)}`",
        f"- Completed backtest count: `{report.get('completed_backtest_count', 0)}`",
        f"- Smoke completed forecast count: `{report.get('smoke_completed_forecast_count', 0)}`",
        f"- Pilot completed forecast count: `{report.get('pilot_completed_forecast_count', 0)}`",
        f"- Shortlist count: `{report.get('shortlist_count', 0)}`",
        f"- Positive transfer count: `{small_capital.get('positive_transfer_count', 0)}`",
        f"- Matrix entry candidate count: `{small_capital.get('matrix_entry_candidate_count', 0)}`",
        f"- Small-capital research-grade matrix count: `{small_capital.get('research_grade_matrix_count', 0)}`",
        "",
        f"## {SMALL_CAPITAL_GATE_ID}",
        "",
        "- quick bridge entry requires excess_annual_return > `0` and excess_sharpe > `0`.",
        "- candidate matrix entry requires excess_sharpe >= `1.0` and positive_month_ratio >= `0.55`.",
        "- research-grade matrix threshold is excess_sharpe >= `1.2`, excess_annual_return >= `0.25`, positive_month_ratio >= `0.60`, worst_month >= `-0.15`.",
        "",
        "## Shortlist",
        "",
    ]
    if shortlist:
        for row in shortlist:
            lines.append(
                "- `{tag}` {dataset}/{split}/{model}/{loss}: score=`{score:.6f}`, excess_return=`{excess:.6f}`, excess_sharpe=`{sharpe:.6f}`".format(
                    tag=row.get("tag", ""),
                    dataset=row.get("dataset_key", ""),
                    split=row.get("split_key", ""),
                    model=row.get("model_family", ""),
                    loss=row.get("loss_profile", ""),
                    score=_float_metric(row, "high_return_score"),
                    excess=_float_metric(row, "excess_annual_return"),
                    sharpe=_float_metric(row, "excess_sharpe"),
                )
            )
    else:
        lines.append("- No formal shortlist yet.")
    lines.extend(["", "## Leaderboard Preview", ""])
    for row in leaderboard:
        lines.append(
            "- `{tag}` {grade} loss=`{loss}`: score=`{score:.6f}`, forecast=`{forecast}`, bridge=`{bridge}`, smoke=`{smoke}`".format(
                tag=row.get("tag", ""),
                grade=row.get("evidence_grade", ""),
                loss=row.get("loss_profile", ""),
                score=_float_metric(row, "high_return_score"),
                forecast=row.get("forecast_status", ""),
                bridge=row.get("backtest_status", ""),
                smoke=bool(row.get("smoke_only", False)),
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- research_only: `true`",
            "- shadow_only: `true`",
            "- promotion_allowed: `false`",
            "- smoke_only and pilot_only rows are excluded from formal shortlist and completed model-quality evidence.",
            "- active_execution_strategy_expected_diff: `none`",
        ]
    )
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_high_return_report(
    *,
    output_root: str | Path | None = None,
    run_tag: str = RUN_TAG,
    task_list_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(output_root, run_tag)
    tasks_payload = _read_json(task_list_path or root / "v2_high_return_model_discovery_task_list.json")
    tasks = list(tasks_payload.get("training_tasks", []) or [])
    rows: list[dict[str, Any]] = []
    for task in tasks:
        tag = str(task.get("tag", ""))
        summary = _read_json(Path(str(task.get("study_dir", STUDIES_ROOT / tag))) / "study_summary.json")
        training = _summary_section(summary)
        bridge_metrics = _load_bridge_metrics(root, tag)
        core = dict(bridge_metrics.get("core_metrics", {}) or {})
        monthly = dict(bridge_metrics.get("monthly_backtest_diagnostics", {}) or {})
        row = {
            "tag": tag,
            "candidate_id": str(task.get("candidate_id", "")),
            "dataset_key": str(task.get("dataset_key", "")),
            "dataset_id": str(task.get("dataset_id", "")),
            "pool_view_id": str(task.get("pool_view_id", "")),
            "feature_profile": str(task.get("feature_profile", "")),
            "split_key": str(task.get("split_key", "")),
            "model_family": str(task.get("model_family", "")),
            "loss_profile": str(task.get("loss_profile", tasks_payload.get("loss_profile", LOSS_PROFILE))),
            "output_profile": str(task.get("output_profile", tasks_payload.get("output_profile", OUTPUT_PROFILE))),
            "selection_profile": str(task.get("selection_profile", tasks_payload.get("selection_profile", SELECTION_PROFILE))),
            "seed": int(task.get("seed", 0) or 0),
            "max_samples_per_role": int(task.get("max_samples_per_role", tasks_payload.get("max_samples_per_role", 0)) or 0),
            "max_samples_per_date_per_role": int(
                task.get("max_samples_per_date_per_role", tasks_payload.get("max_samples_per_date_per_role", 0)) or 0
            ),
            "forecast_status": str(summary.get("status", "")),
            "evidence_verdict": str(summary.get("evidence_verdict", "")),
            "rank_ic": _metric(training, "test_metrics", "decision_score_rank_ic"),
            "top_bottom_spread": _metric(training, "test_metrics", "decision_score_top_bottom_spread"),
            "hit_lift": _metric(training, "test_metrics", "decision_hit_lift_top20_mean"),
            "bridge_status": bridge_metrics.get("bridge_status", ""),
            "backtest_status": bridge_metrics.get("backtest_status", ""),
            "annual_return": _float_metric(core, "annual_return"),
            "excess_annual_return": _float_metric(core, "excess_annual_return"),
            "excess_sharpe": _float_metric(core, "excess_sharpe"),
            "max_drawdown": _float_metric(core, "max_drawdown"),
            "positive_month_ratio": _float_metric(monthly, "positive_month_ratio", _float_metric(monthly, "positive_months_ratio")),
            "negative_month_count": _float_metric(monthly, "negative_month_count"),
            "worst_monthly_return": _float_metric(
                monthly,
                "worst_monthly_excess_return",
                _float_metric(monthly, "worst_monthly_return"),
            ),
            "single_seed_scout_only": bool(task.get("single_seed_scout_only", True)),
            "smoke_only": bool(task.get("smoke_only", False)),
            "pilot_only": bool(task.get("pilot_only", False)),
            "evidence_grade": str(task.get("evidence_grade", "")),
        }
        row["positive_transfer"] = _positive_transfer(row)
        row["matrix_entry_candidate"] = _matrix_entry_candidate(row)
        row.update(_load_matrix_metrics(root, tag))
        row.update(_score_row(row))
        rows.append(row)
    leaderboard = sorted(rows, key=lambda item: float(item.get("high_return_score", 0.0)), reverse=True)
    shortlist = [
        row
        for row in leaderboard
        if _matrix_entry_candidate(row)
        and _has_forecast_result(row)
        and not bool(row.get("smoke_only", False))
        and not bool(row.get("pilot_only", False))
    ][:3]
    completed_forecast_count = sum(1 for row in rows if str(row.get("forecast_status", "")).strip() == "completed")
    completed_backtest_count = sum(1 for row in rows if str(row.get("backtest_status", "")).strip() == "completed")
    smoke_completed_forecast_count = sum(
        1 for row in rows if bool(row.get("smoke_only", False)) and str(row.get("forecast_status", "")).strip() == "completed"
    )
    pilot_completed_forecast_count = sum(
        1 for row in rows if bool(row.get("pilot_only", False)) and str(row.get("forecast_status", "")).strip() == "completed"
    )
    matrix_entry_rows = [row for row in rows if bool(row.get("matrix_entry_candidate", False))]
    research_grade_matrix_rows = [
        row for row in rows if int(row.get("small_capital_research_grade_candidate_count", 0) or 0) > 0
    ]
    best_matrix_row = max(
        matrix_entry_rows,
        key=lambda row: (_float_metric(row, "high_return_score"), _float_metric(row, "excess_sharpe")),
        default={},
    )
    best_research_grade_row = max(
        research_grade_matrix_rows,
        key=lambda row: (
            _float_metric(row, "best_small_capital_balanced_return_score"),
            _float_metric(row, "best_small_capital_excess_sharpe"),
        ),
        default={},
    )
    small_capital_gate_summary = {
        "gate_id": SMALL_CAPITAL_GATE_ID,
        "positive_transfer_count": int(sum(1 for row in rows if bool(row.get("positive_transfer", False)))),
        "matrix_entry_candidate_count": int(len(matrix_entry_rows)),
        "research_grade_matrix_count": int(len(research_grade_matrix_rows)),
        "matrix_entry_thresholds": {
            "excess_annual_return_gt": 0.0,
            "excess_sharpe_gt": 0.0,
            "excess_sharpe_ge": SCOUT_MATRIX_ENTRY_EXCESS_SHARPE_MIN,
            "positive_month_ratio_ge": SCOUT_MATRIX_ENTRY_POSITIVE_MONTH_RATIO_MIN,
        },
        "research_grade_thresholds": dict(candidate_matrix.SMALL_CAPITAL_RESEARCH_THRESHOLDS),
        "best_matrix_entry_tag": str(best_matrix_row.get("tag", "")),
        "best_research_grade_tag": str(best_research_grade_row.get("tag", "")),
        "best_research_grade_variant_id": str(best_research_grade_row.get("research_grade_small_capital_variant_id", "")),
    }
    status = "missing_tasks"
    if rows and completed_forecast_count <= 0:
        status = "awaiting_forecast_results"
    elif rows and completed_forecast_count > 0:
        status = "completed"
    report = {
        "schema_version": 1,
        "status": status,
        "run_tag": str(run_tag),
        "study_family": STUDY_FAMILY,
        "task_list_path": str(task_list_path or root / "v2_high_return_model_discovery_task_list.json"),
        "loss_profiles": list(tasks_payload.get("loss_profiles", []) or []),
        "completed_forecast_count": int(completed_forecast_count),
        "completed_backtest_count": int(completed_backtest_count),
        "smoke_completed_forecast_count": int(smoke_completed_forecast_count),
        "pilot_completed_forecast_count": int(pilot_completed_forecast_count),
        "leaderboard": leaderboard,
        "split_consistency": _split_consistency(rows),
        "bad_month_preview": _bad_month_preview(rows),
        "shortlist": shortlist,
        "shortlist_count": len(shortlist),
        "small_capital_gate_summary": small_capital_gate_summary,
        "scoring_policy": {
            "intended_use": "research-only high-return scout ranking",
            "not_a_promotion_gate": True,
            "three_seed_required_before_model_quality_evidence": True,
            "shortlist_requires_positive_transfer": True,
            "shortlist_requires_excess_sharpe_ge": SCOUT_MATRIX_ENTRY_EXCESS_SHARPE_MIN,
            "shortlist_requires_positive_month_ratio_ge": SCOUT_MATRIX_ENTRY_POSITIVE_MONTH_RATIO_MIN,
            "small_capital_gate_id": SMALL_CAPITAL_GATE_ID,
        },
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
        "outputs": {
            "report_json": str(root / "v2_high_return_model_discovery_report.json"),
            "report_md": str(root / "v2_high_return_model_discovery_report.md"),
        },
        "updated_at": _now(),
    }
    _write_json(root / "v2_high_return_model_discovery_report.json", report)
    _write_markdown(root / "v2_high_return_model_discovery_report.md", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research-only high-return model discovery orchestration for daily_research v2.")
    parser.add_argument("--run-tag", default=RUN_TAG)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--datasets", default="same_period_augmented,long_history_augmented")
    parser.add_argument("--splits", default="")
    parser.add_argument("--model-families", default=",".join(DEFAULT_MODEL_FAMILIES))
    parser.add_argument("--loss-profiles", default=",".join(DEFAULT_LOSS_PROFILES))
    parser.add_argument("--seeds", default=str(SCOUT_SEED))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--allow-single-seed-scout", action="store_true")
    parser.add_argument("--write-task-list", action="store_true")
    parser.add_argument("--run-forecast", action="store_true")
    parser.add_argument("--run-quick-bridge", action="store_true")
    parser.add_argument("--run-candidate-matrix", action="store_true")
    parser.add_argument("--run-all-matrices", action="store_true")
    parser.add_argument("--collect-report", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--forecast-max-samples-per-role", type=int, default=0)
    parser.add_argument("--forecast-max-samples-per-date-per-role", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    output_root = args.output_root or str(STUDIES_ROOT / str(args.run_tag))
    actions: dict[str, Any] = {}
    if args.write_task_list:
        path = write_task_list(
            output_root=output_root,
            run_tag=args.run_tag,
            datasets=args.datasets,
            splits=args.splits,
            model_families=args.model_families,
            loss_profiles=args.loss_profiles,
            seeds=args.seeds,
            epochs=int(args.epochs),
            min_epochs=int(args.min_epochs),
            patience=int(args.early_stop_patience),
            allow_single_seed_scout=bool(args.allow_single_seed_scout),
            max_samples_per_role=int(args.forecast_max_samples_per_role),
            max_samples_per_date_per_role=int(args.forecast_max_samples_per_date_per_role),
        )
        actions["task_list"] = str(path)
    if args.run_forecast:
        actions["forecast_run"] = run_forecast_tasks(
            output_root=output_root,
            run_tag=args.run_tag,
            datasets=args.datasets,
            splits=args.splits,
            model_families=args.model_families,
            loss_profiles=args.loss_profiles,
            seeds=args.seeds,
            epochs=int(args.epochs),
            min_epochs=int(args.min_epochs),
            patience=int(args.early_stop_patience),
            allow_single_seed_scout=bool(args.allow_single_seed_scout),
            skip_existing=bool(args.skip_existing),
            max_tasks=int(args.max_tasks),
            max_samples_per_role=int(args.forecast_max_samples_per_role),
            max_samples_per_date_per_role=int(args.forecast_max_samples_per_date_per_role),
        )
    if args.run_quick_bridge:
        actions["quick_bridge_run"] = run_quick_bridge_tasks(
            output_root=output_root,
            run_tag=args.run_tag,
            skip_existing=bool(args.skip_existing),
            max_tasks=int(args.max_tasks),
        )
    if args.run_candidate_matrix:
        actions["candidate_matrix_run"] = run_candidate_matrix_tasks(
            output_root=output_root,
            run_tag=args.run_tag,
            skip_existing=bool(args.skip_existing),
            shortlist_only=not bool(args.run_all_matrices),
            max_tasks=int(args.max_tasks),
        )
    if args.collect_report:
        actions["report"] = collect_high_return_report(output_root=output_root, run_tag=args.run_tag)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "run_tag": str(args.run_tag),
        "stage": "high_return_model_discovery",
        "actions": actions,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "promotion_allowed": False,
            "active_execution_strategy_expected_diff": "none",
        },
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status=completed run_tag={args.run_tag} output_root={output_root} actions={','.join(actions) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
