from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from daily_research.path_policy.seq100_walkforward import (
    APPROVED_DEVELOPMENT_CONTRACT_PATH,
    DEFAULT_PYTHON,
    WORKSPACE_ROOT,
    _validated_development_fold_training_contract,
    _validated_fold_training_contract,
    approved_development_contract_binding,
    development_view_path,
    purged_view_path,
    verify_development_walkforward_view,
    verify_purged_walkforward_view,
)


CONTRACT_ID = "seq100_pit_adjusted_global_tail_contract_20260711"
INNER_OOS_YEARS = (2018, 2019, 2020, 2021)
OUTER_OOS_YEARS = (2022, 2023, 2024, 2025)
DEVELOPMENT_YEARS = (2022, 2023, 2024, 2025)
DEVELOPMENT_SEED = 7
DEFAULT_DEVELOPMENT_PROFILES = ("baseline", "hard_st")
SCREEN_SEEDS = (7,)
CONFIRM_SEEDS = (7, 17, 29)
DEFAULT_ROOT = Path("daily_research/output/path_policy/studies/seq100_pit_adjusted_global_tail_generation_20260711_v1")
DEFAULT_SOURCE_VIEW = Path(
    "daily_research/data/research_store/sequence_pack/"
    "qdp_v2_seq100_path60_todayclose_pit_adjusted_2012_2025_v1/manifest.json"
)
DEFAULT_STORE_ROOT = Path(
    "daily_research/data/research_store/walkforward/"
    "seq100_pit_adjusted_global_tail_contract_20260711_v1"
)
DEFAULT_DEVELOPMENT_ROOT = Path(
    "daily_research/output/path_policy/studies/"
    "seq100_candidate_complete_development_walkforward_20260711_v1"
)
DEFAULT_DEVELOPMENT_STORE_ROOT = Path(
    "daily_research/data/research_store/walkforward/"
    "seq100_candidate_complete_development_walkforward_20260711_v1"
)
REGISTRY_SCHEMA_VERSION = 2
EVIDENCE_POLICY_RUN_ARTIFACTS = "run_artifacts_only"
EVIDENCE_POLICY_SYNTHETIC_ALLOWED = "synthetic_metrics_allowed_for_tests"
CODE_PROVENANCE_PATHS = (
    Path("daily_research/path_policy/seq100_mainline.py"),
    Path("daily_research/path_policy/qdp_v2_sequence_path_training.py"),
    Path("daily_research/path_policy/seq100_walkforward.py"),
    Path("daily_research/path_policy/seq100_research_generation.py"),
    Path("daily_research/brain/references/seq100_pit_adjusted_global_tail_contract_20260711.json"),
)
DEVELOPMENT_CODE_PROVENANCE_PATHS = (
    Path("daily_research/path_policy/seq100_mainline.py"),
    Path("daily_research/path_policy/qdp_v2_sequence_path_pack.py"),
    Path("daily_research/path_policy/qdp_v2_sequence_path_training.py"),
    Path("daily_research/path_policy/seq100_walkforward.py"),
    Path("daily_research/path_policy/seq100_research_generation.py"),
    APPROVED_DEVELOPMENT_CONTRACT_PATH,
)

PROFILE_COMMANDS = {
    "baseline": "train-daily-only-summary-v2-ohlcva-aux-low",
    "hard_st": "train-daily-only-summary-v2-ohlcva-aux-low-hard-st",
    "hard_st_global_tail": "train-daily-only-summary-v2-ohlcva-aux-low-hard-st-global-tail",
}
PROFILE_ORDER = tuple(PROFILE_COMMANDS)
METRIC_NAMES = (
    "top3_opportunity_alpha",
    "top3_realized_plan_alpha",
    "oracle_regret",
    "top10_opportunity_alpha",
    "daily_rank_ic",
    "fill_rate",
    "path_mae",
)
DEVELOPMENT_TOP_K = (1, 3, 5, 10)
DEVELOPMENT_METRIC_NAMES = tuple(
    f"top{top_k}_{name}"
    for top_k in DEVELOPMENT_TOP_K
    for name in (
        "opportunity_alpha",
        "net_realized_plan_return_base_alpha",
        "net_realized_plan_return_stress_alpha",
        "net_realized_plan_value_base_alpha",
        "net_realized_plan_value_stress_alpha",
        "oracle_regret",
        "entry_fill_rate",
        "realized_plan_coverage",
    )
) + (
    "development_total_loss",
    "daily_rank_ic",
    "path_mae",
)
DEVELOPMENT_SELECTION_POLICY = {
    "role": "development_model_selection",
    "primary": "mean_topk_net_realized_plan_return_base_alpha",
    "direction": "maximize",
    "top_k": list(DEVELOPMENT_TOP_K),
    "aggregation": "equal_development_year_weight",
    "eligibility": {
        "mean_top3_net_realized_plan_return_base_alpha_min": 0.0,
        "mean_top10_net_realized_plan_return_base_alpha_min": 0.0,
        "mean_top3_net_realized_plan_return_stress_alpha_min": 0.0,
        "positive_top3_development_years_min": 3,
        "all_topk_realized_plan_coverage": 1.0,
    },
    "tie_breakers": [
        "worst_development_year_top3_net_realized_plan_return_base_alpha",
        "leave_best_year_out_top3_net_realized_plan_return_base_alpha",
        "mean_topk_net_realized_plan_value_base_alpha",
        "mean_topk_opportunity_alpha",
        "daily_rank_ic",
    ],
    "required_years": list(DEVELOPMENT_YEARS),
    "historical_test_set": None,
    "historical_outer_audit": False,
}
SELECTION_POLICY = {
    "primary": "top3_opportunity_alpha",
    "direction": "maximize",
    "tie_breakers": ["worst_fold_top3_opportunity_alpha", "top3_realized_plan_alpha", "daily_rank_ic"],
    "guardrails_relative_to_baseline": {
        "top3_realized_plan_alpha_min_delta": -0.005,
        "oracle_regret_max_delta": 0.01,
        "top10_opportunity_alpha_min_delta": -0.005,
        "daily_rank_ic_min_delta": -0.01,
        "worst_fold_top3_opportunity_alpha_min_delta": -0.01,
        "fill_rate_min_delta": -0.05,
        "path_mae_max_ratio": 1.10,
    },
    "finite_required": list(METRIC_NAMES),
    "folds_required": list(INNER_OOS_YEARS),
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _workspace_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else WORKSPACE_ROOT / raw


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_workspace_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _workspace_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(registry)
    declared = str(payload.pop("registry_sha256", ""))
    if not declared or _payload_sha256(payload) != declared:
        raise ValueError("registry digest mismatch")
    return dict(registry)


def _current_code_provenance(
    paths: Sequence[Path] = CODE_PROVENANCE_PATHS,
) -> dict[str, Any]:
    files = [
        {"path": str(_workspace_path(path).resolve()), "sha256": _file_sha256(path)}
        for path in paths
    ]
    return {"files": files, "sha256": _payload_sha256(files)}


def _validate_code_provenance(expected: Mapping[str, Any]) -> dict[str, Any]:
    expected_files = list(dict(expected or {}).get("files", []) or [])
    paths = tuple(Path(str(item.get("path", "") or "")) for item in expected_files)
    current = _current_code_provenance(paths or CODE_PROVENANCE_PATHS)
    if current != dict(expected or {}):
        raise ValueError("registered research code provenance changed")
    return current


def _validate_source_view(
    source_view: str | Path,
    *,
    require_corrected_contract: bool,
) -> tuple[Path, dict[str, Any], str]:
    source_path = _workspace_path(source_view).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = _read_json(source_path)
    source_sha256 = _file_sha256(source_path)
    if not require_corrected_contract:
        return source_path, source, source_sha256
    if str(source.get("artifact_type", "")) != "qdp_v2_sequence_path_pack":
        raise ValueError("research generation requires a qdp_v2_sequence_path_pack source")
    if int(source.get("lookback_days", 0) or 0) != 100 or int(source.get("forward_days", 0) or 0) != 60:
        raise ValueError("research generation source must use lookback=100 and forward=60")
    semantics = dict(source.get("data_semantics", {}) or {})
    expected_semantics = {
        "price_adjustment": "back_adjust",
        "entry_rule": "open_below_limit_tick",
        "pit_universe_domain": "pit_signal_universe",
    }
    for key, expected in expected_semantics.items():
        if semantics.get(key) != expected:
            raise ValueError(f"research generation source data_semantics.{key} must equal {expected!r}")
    if semantics.get("unfilled_samples_retained") is not True:
        raise ValueError("research generation source must retain unfilled samples")
    labels = dict(source.get("label_arrays", {}) or {})
    ohlcva = dict(labels.get("future_ohlcva_path", {}) or {})
    if str(ohlcva.get("price_anchor", "")) != "today_close":
        raise ValueError("research generation source must use today_close path anchoring")
    masks = dict(source.get("masks", {}) or {})
    required_masks = {"entry_filled", "signal_eligible", "tradable", "price_observed", "va_aux_valid"}
    missing_masks = sorted(required_masks.difference(masks))
    if missing_masks:
        raise ValueError(f"research generation source is missing corrected masks: {missing_masks}")
    return source_path, source, source_sha256


def _validate_development_source_view(
    source_view: str | Path,
    *,
    require_corrected_contract: bool,
) -> tuple[Path, dict[str, Any], str]:
    source_path, source, source_sha256 = _validate_source_view(
        source_view,
        require_corrected_contract=bool(require_corrected_contract),
    )
    if not require_corrected_contract:
        return source_path, source, source_sha256
    approved = approved_development_contract_binding()
    if dict(source.get("research_contract", {}) or {}) != approved:
        raise ValueError("development source is not bound to the approved successor contract")
    if dict(source.get("development_contract", {}) or {}) != approved:
        raise ValueError("development source contract alias differs from the approved successor contract")
    candidate_path = _workspace_path(str(source.get("candidate_index_path", "") or "")).resolve()
    if not candidate_path.is_file():
        raise ValueError("development source requires a candidate_index_path")
    if int(source.get("candidate_count", 0) or 0) <= 0:
        raise ValueError("development source candidate_count must be positive")
    if int(source.get("execution_tail_days", 0) or 0) != 20:
        raise ValueError("development source execution_tail_days must equal 20")
    max_dependency = int(source.get("max_label_dependency_days", 0) or 0)
    if max_dependency != 80:
        raise ValueError("development source max_label_dependency_days must equal 80")
    if source.get("dependency_padding_complete") is not True:
        raise ValueError("development source requires complete dependency padding")
    if int(source.get("available_dependency_padding_days", 0) or 0) < max_dependency:
        raise ValueError("development source does not expose the full dependency padding window")
    execution_arrays = dict(source.get("execution_arrays", {}) or {})
    required_execution_arrays = {
        "entry_open_raw",
        "entry_up_limit_raw",
        "exit_close_raw",
        "exit_down_limit_raw",
    }
    missing_execution_arrays = sorted(required_execution_arrays.difference(execution_arrays))
    if missing_execution_arrays:
        raise ValueError(f"development source is missing execution arrays: {missing_execution_arrays}")
    masks = dict(source.get("masks", {}) or {})
    required_execution_masks = {"candidate_eligible", "exit_has_valid_bar_volume", "exit_sellable"}
    missing_execution_masks = sorted(required_execution_masks.difference(masks))
    if missing_execution_masks:
        raise ValueError(f"development source is missing execution masks: {missing_execution_masks}")
    if not dict(source.get("execution_cost_contract", {}) or {}):
        raise ValueError("development source requires an execution_cost_contract")
    terminal = dict(source.get("terminal_execution_contract", {}) or {})
    if str(terminal.get("unresolved_after_tail", "")) != "apply_precommitted_recovery_fraction":
        raise ValueError("development source requires precommitted terminal recovery")
    execution_views = dict(source.get("execution_views", {}) or {})
    if not {"exit_close_raw_path", "exit_down_limit_raw_path", "exit_sellable_path"}.issubset(execution_views):
        raise ValueError("development source is missing execution tail views")
    return source_path, source, source_sha256


def _fold_bindings(
    *,
    years: Sequence[int],
    store_root: Path,
    source_view_sha256: str,
    train_start_year: int,
    require_existing: bool,
) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for year in years:
        view_path = purged_view_path(int(year), store_root=store_root).resolve()
        if not view_path.is_file():
            if require_existing:
                raise FileNotFoundError(
                    f"missing prebuilt fold for {year}: {view_path}; build and verify folds before registry init"
                )
            bindings[str(int(year))] = {
                "oos_year": int(year),
                "view_path": str(view_path),
                "view_sha256": "",
                "fold_training_contract_sha256": "",
                "source_view_sha256": str(source_view_sha256),
                "train_start_year": int(train_start_year),
            }
            continue
        verification = verify_purged_walkforward_view(view_path)
        if str(verification.get("status", "")) != "ok":
            raise ValueError(f"fold verification failed for {year}: {verification.get('blockers', [])}")
        manifest = _read_json(view_path)
        provenance = dict(manifest.get("source_view_provenance", {}) or {})
        if str(provenance.get("manifest_sha256", "")) != str(source_view_sha256):
            raise ValueError(f"fold {year} is not derived from the registered source view")
        purge = dict(manifest.get("purged_walkforward", {}) or {})
        if int(purge.get("train_start_year", -1)) != int(train_start_year):
            raise ValueError(f"fold {year} train_start_year does not match the registered contract")
        fold_contract = _validated_fold_training_contract(manifest)
        bindings[str(int(year))] = {
            "oos_year": int(year),
            "view_path": str(view_path),
            "view_sha256": _file_sha256(view_path),
            "fold_training_contract_sha256": str(fold_contract["sha256"]),
            "source_view_sha256": str(source_view_sha256),
            "train_start_year": int(train_start_year),
        }
    return bindings


def _development_fold_bindings(
    *,
    years: Sequence[int],
    store_root: Path,
    source_view_sha256: str,
    train_start_year: int,
    require_existing: bool,
) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    approved = approved_development_contract_binding()
    for year in years:
        view_path = development_view_path(int(year), store_root=store_root).resolve()
        if not view_path.is_file():
            if require_existing:
                raise FileNotFoundError(
                    f"missing prebuilt development fold for {year}: {view_path}; build and verify folds first"
                )
            bindings[str(int(year))] = {
                "development_year": int(year),
                "view_path": str(view_path),
                "view_sha256": "",
                "development_fold_training_contract_sha256": "",
                "candidate_index_sha256": "",
                "execution_cost_contract_sha256": "",
                "source_view_sha256": str(source_view_sha256),
                "train_start_year": int(train_start_year),
                "research_contract_sha256": str(approved["contract_sha256"]),
            }
            continue
        verification = verify_development_walkforward_view(view_path)
        if str(verification.get("status", "")) != "ok":
            raise ValueError(
                f"development fold verification failed for {year}: {verification.get('blockers', [])}"
            )
        manifest = _read_json(view_path)
        if dict(manifest.get("research_contract", {}) or {}) != approved:
            raise ValueError(f"development fold {year} contract identity changed")
        if dict(manifest.get("development_contract", {}) or {}) != approved:
            raise ValueError(f"development fold {year} contract alias changed")
        provenance = dict(manifest.get("source_view_provenance", {}) or {})
        if str(provenance.get("manifest_sha256", "")) != str(source_view_sha256):
            raise ValueError(f"development fold {year} is not derived from the registered source")
        walkforward = dict(manifest.get("development_walkforward", {}) or {})
        if int(walkforward.get("development_year", -1)) != int(year):
            raise ValueError(f"development fold {year} year metadata mismatch")
        if int(walkforward.get("train_start_year", -1)) != int(train_start_year):
            raise ValueError(f"development fold {year} train_start_year mismatch")
        if int(walkforward.get("max_label_dependency_days", 0) or 0) != 80:
            raise ValueError(f"development fold {year} does not use the approved 80-day dependency")
        fold_contract = _validated_development_fold_training_contract(manifest)
        execution_cost_sha256 = _payload_sha256(dict(manifest.get("execution_cost_contract", {}) or {}))
        bindings[str(int(year))] = {
            "development_year": int(year),
            "view_path": str(view_path),
            "view_sha256": _file_sha256(view_path),
            "development_fold_training_contract_sha256": str(fold_contract["sha256"]),
            "sample_index_sha256": str(fold_contract["sample_index_sha256"]),
            "candidate_index_sha256": str(fold_contract["candidate_index_sha256"]),
            "execution_cost_contract_sha256": execution_cost_sha256,
            "source_view_sha256": str(source_view_sha256),
            "train_start_year": int(train_start_year),
            "research_contract_sha256": str(approved["contract_sha256"]),
        }
    return bindings


def _write_json(path: str | Path, payload: Mapping[str, Any], *, immutable: bool) -> Path:
    target = _workspace_path(path)
    serialized = json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if target.exists() and immutable:
        if target.read_text(encoding="utf-8") == serialized:
            return target
        raise FileExistsError(f"immutable artifact already exists with different content: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, target)
    return target


def _write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("historical outer audit has already been consumed or claimed") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    return target


@contextmanager
def _exclusive_update_lock(path: str | Path):
    lock_path = _workspace_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"research ledger is already being updated: {lock_path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": _now()}, ensure_ascii=False))
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _validated_study_root(payload: Mapping[str, Any], supplied_root: str | Path) -> Path:
    declared = _workspace_path(str(payload.get("study_root", "") or "")).resolve()
    supplied = _workspace_path(supplied_root).resolve()
    if declared != supplied:
        raise ValueError("study root does not match the immutable research registry")
    return declared


def _validated_ledger_path(payload: Mapping[str, Any], ledger_path: str | Path) -> Path:
    study_root = _workspace_path(str(payload.get("study_root", "") or "")).resolve()
    expected = (study_root / "result_ledger.json").resolve()
    supplied = _workspace_path(ledger_path).resolve()
    if supplied != expected:
        raise ValueError("ledger path must be the canonical study result_ledger.json")
    return expected


def _parse_ints(raw: str | Iterable[int]) -> tuple[int, ...]:
    if not isinstance(raw, str):
        values = tuple(int(item) for item in raw)
    else:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("seeds must be a non-empty unique sequence")
    return values


def _validate_exact_years(years: Sequence[int], expected: Sequence[int], *, label: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in years)
    if values != tuple(expected):
        raise ValueError(f"{label} years are frozen at {list(expected)}")
    return values


def _job_id(phase: str, profile: str, year: int, seed: int) -> str:
    return f"{phase}:{profile}:oos{int(year)}:seed{int(seed)}"


def _training_command(
    *,
    phase: str,
    profile: str,
    year: int,
    seed: int,
    epochs: int,
    max_samples_per_split: int,
    root: Path,
    view_path: Path,
    python_executable: Path,
) -> list[str]:
    command = [
        str(_workspace_path(python_executable)),
        "-m",
        "daily_research.path_policy.seq100_mainline",
        PROFILE_COMMANDS[profile],
        "--store-view",
        str(view_path.resolve()),
        "--output-root",
        str((root / "runs" / phase).resolve()),
        "--run-tag",
        f"seq100_pit_{phase}_{profile}_oos{int(year)}_seed{int(seed)}",
        "--epochs",
        str(int(epochs)),
        "--device",
        "cuda",
        "--seed",
        str(int(seed)),
        "--early-stopping-patience",
        "0",
        "--prediction-mode",
        "none",
        "--evaluation-mode",
        "fixed_oos",
    ]
    if int(max_samples_per_split) > 0:
        command.extend(["--max-samples-per-split", str(int(max_samples_per_split))])
    command.append("--json")
    return command


def _fold_build_commands(
    *,
    years: Sequence[int],
    source_view: Path,
    store_root: Path,
    train_start_year: int,
    python_executable: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "oos_year": int(year),
            "view_path": str(purged_view_path(int(year), store_root=store_root).resolve()),
            "command": [
                str(_workspace_path(python_executable)),
                "-m",
                "daily_research.path_policy.seq100_walkforward",
                "build-fold",
                "--source-view",
                str(_workspace_path(source_view).resolve()),
                "--oos-year",
                str(int(year)),
                "--train-start-year",
                str(int(train_start_year)),
                "--store-root",
                str(_workspace_path(store_root).resolve()),
                "--json",
            ],
        }
        for year in years
    ]


def _build_jobs(
    *,
    phase: str,
    profiles: Sequence[str],
    years: Sequence[int],
    seeds: Sequence[int],
    epochs: int,
    max_samples_per_split: int,
    root: Path,
    fold_bindings: Mapping[str, Mapping[str, Any]],
    python_executable: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for profile in profiles:
        if profile not in PROFILE_COMMANDS:
            raise ValueError(f"unsupported profile: {profile}")
        for year in years:
            fold_binding = dict(fold_bindings.get(str(int(year)), {}) or {})
            if not fold_binding:
                raise ValueError(f"missing fold binding for OOS year {year}")
            view_path = _workspace_path(str(fold_binding.get("view_path", "") or "")).resolve()
            for seed in seeds:
                job_id = _job_id(phase, profile, int(year), int(seed))
                job = {
                    "job_id": job_id,
                    "phase": phase,
                    "profile": profile,
                    "profile_command": PROFILE_COMMANDS[profile],
                    "oos_year": int(year),
                    "seed": int(seed),
                    "epochs": int(epochs),
                    "max_samples_per_split": int(max_samples_per_split),
                    "evaluation_mode": "fixed_oos",
                    "checkpoint_policy": "final_epoch",
                    "view_path": str(view_path),
                    "view_sha256": str(fold_binding.get("view_sha256", "")),
                    "fold_training_contract_sha256": str(
                        fold_binding.get("fold_training_contract_sha256", "")
                    ),
                    "source_view_sha256": str(fold_binding.get("source_view_sha256", "")),
                    "command": _training_command(
                        phase=phase,
                        profile=profile,
                        year=int(year),
                        seed=int(seed),
                        epochs=int(epochs),
                        max_samples_per_split=int(max_samples_per_split),
                        root=root,
                        view_path=view_path,
                        python_executable=python_executable,
                    ),
                }
                job["expected_resolved_training_config"] = _expected_resolved_profile_config(job)
                jobs.append(job)
    return jobs


def _normalize_development_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(item).strip() for item in profiles if str(item).strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("development profiles must be a non-empty unique sequence")
    unknown = sorted(set(values).difference(PROFILE_COMMANDS))
    if unknown:
        raise ValueError(f"unsupported development profiles: {unknown}")
    if len(values) > 5:
        raise ValueError("approved development generation allows at most five candidates")
    return values


def _development_job_id(profile: str, year: int) -> str:
    return f"development:{profile}:year{int(year)}:seed{DEVELOPMENT_SEED}"


def _development_training_command(
    *,
    profile: str,
    year: int,
    maximum_epochs: int,
    patience: int,
    min_delta: float,
    minimum_complete_epochs: int,
    root: Path,
    view_path: Path,
    python_executable: Path,
) -> list[str]:
    return [
        str(_workspace_path(python_executable)),
        "-m",
        "daily_research.path_policy.seq100_mainline",
        PROFILE_COMMANDS[profile],
        "--store-view",
        str(view_path.resolve()),
        "--output-root",
        str((root / "runs" / "development").resolve()),
        "--run-tag",
        f"seq100_development_{profile}_{int(year)}_seed{DEVELOPMENT_SEED}",
        "--epochs",
        str(int(maximum_epochs)),
        "--device",
        "cuda",
        "--seed",
        str(DEVELOPMENT_SEED),
        "--max-samples-per-split",
        "0",
        "--early-stopping-patience",
        str(int(patience)),
        "--early-stopping-min-delta",
        str(float(min_delta)),
        "--early-stopping-metric",
        "development_total_loss",
        "--early-stopping-mode",
        "min",
        "--min-complete-epochs",
        str(int(minimum_complete_epochs)),
        "--prediction-mode",
        "none",
        "--evaluation-mode",
        "development",
        "--json",
    ]


def _development_fold_build_commands(
    *,
    years: Sequence[int],
    source_view: Path,
    store_root: Path,
    train_start_year: int,
    python_executable: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "development_year": int(year),
            "view_path": str(development_view_path(int(year), store_root=store_root).resolve()),
            "command": [
                str(_workspace_path(python_executable)),
                "-m",
                "daily_research.path_policy.seq100_walkforward",
                "build-development-fold",
                "--source-view",
                str(_workspace_path(source_view).resolve()),
                "--development-year",
                str(int(year)),
                "--train-start-year",
                str(int(train_start_year)),
                "--store-root",
                str(_workspace_path(store_root).resolve()),
                "--json",
            ],
        }
        for year in years
    ]


def _expected_development_resolved_profile_config(job: Mapping[str, Any]) -> dict[str, Any]:
    from daily_research.path_policy.seq100_mainline import PROFILE_SPECS

    profile = PROFILE_SPECS[str(job["profile_command"])].default_profile()
    return {
        "epochs": int(job["maximum_epochs"]),
        "batch_size": int(profile.batch_size),
        "model_type": str(profile.model_type),
        "hidden_dim": int(profile.hidden_dim),
        "layers": int(profile.layers),
        "dropout": float(profile.dropout),
        "symbol_embedding_dim": 16,
        "learning_rate": float(profile.learning_rate),
        "weight_decay": float(profile.weight_decay),
        "path_loss_weight": float(profile.path_loss_weight),
        "path_loss_profile": str(profile.path_loss_profile),
        "summary_loss_weight": float(profile.summary_loss_weight),
        "richer_loss_weight": float(profile.richer_loss_weight),
        "price_delta_loss_weight": float(profile.price_delta_loss_weight),
        "va_level_loss_weight": float(profile.va_level_loss_weight),
        "va_delta_loss_weight": float(profile.va_delta_loss_weight),
        "value_loss_weight": float(profile.value_loss_weight),
        "rank_loss_weight": float(profile.rank_loss_weight),
        "residual_score_weight": 0.25,
        "residual_penalty_weight": 0.01,
        "summary_loss_profile": str(profile.summary_loss_profile),
        "input_channel_profile": str(profile.input_channel_profile),
        "direct_value_horizon": int(profile.direct_value_horizon),
        "rank_max_per_side": int(profile.rank_max_per_side),
        "device": "cuda",
        "amp": True,
        "seed": DEVELOPMENT_SEED,
        "top_k": [int(item) for item in str(profile.top_k).split(",") if item],
        "max_samples_per_split": 0,
        "prediction_mode": "none",
        "early_stopping_patience": int(job["early_stopping"]["patience"]),
        "early_stopping_min_delta": float(job["early_stopping"]["min_delta"]),
        "early_stopping_metric": "development_total_loss",
        "early_stopping_mode": "min",
        "minimum_complete_epochs": int(job["early_stopping"]["minimum_complete_epochs"]),
        "evaluation_mode": "development",
        "path_value_gradient_profile": str(profile.path_value_gradient_profile),
        "rank_training_profile": str(profile.rank_training_profile),
        "rank_batch_size": int(profile.rank_batch_size),
        "rank_interval": int(profile.rank_interval),
        "prefetch_batches": int(profile.prefetch_batches),
    }


def _build_development_jobs(
    *,
    profiles: Sequence[str],
    years: Sequence[int],
    maximum_epochs: int,
    patience: int,
    min_delta: float,
    minimum_complete_epochs: int,
    root: Path,
    fold_bindings: Mapping[str, Mapping[str, Any]],
    python_executable: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for profile in profiles:
        for year in years:
            binding = dict(fold_bindings.get(str(int(year)), {}) or {})
            if not binding:
                raise ValueError(f"missing development fold binding for {year}")
            view_path = _workspace_path(str(binding.get("view_path", "") or "")).resolve()
            job = {
                "job_id": _development_job_id(profile, int(year)),
                "phase": "development",
                "profile": str(profile),
                "profile_command": PROFILE_COMMANDS[str(profile)],
                "development_year": int(year),
                "seed": DEVELOPMENT_SEED,
                "maximum_epochs": int(maximum_epochs),
                "max_samples_per_split": 0,
                "evaluation_mode": "development",
                "checkpoint_policy": "best_development_total_loss",
                "early_stopping": {
                    "metric": "development_total_loss",
                    "mode": "min",
                    "patience": int(patience),
                    "min_delta": float(min_delta),
                    "minimum_complete_epochs": int(minimum_complete_epochs),
                    "restore_best_checkpoint": True,
                },
                "view_path": str(view_path),
                "view_sha256": str(binding.get("view_sha256", "")),
                "development_fold_training_contract_sha256": str(
                    binding.get("development_fold_training_contract_sha256", "")
                ),
                "candidate_index_sha256": str(binding.get("candidate_index_sha256", "")),
                "execution_cost_contract_sha256": str(
                    binding.get("execution_cost_contract_sha256", "")
                ),
                "source_view_sha256": str(binding.get("source_view_sha256", "")),
                "research_contract_sha256": str(binding.get("research_contract_sha256", "")),
                "metric_names": list(DEVELOPMENT_METRIC_NAMES),
            }
            job["command"] = _development_training_command(
                profile=str(profile),
                year=int(year),
                maximum_epochs=int(maximum_epochs),
                patience=int(patience),
                min_delta=float(min_delta),
                minimum_complete_epochs=int(minimum_complete_epochs),
                root=root,
                view_path=view_path,
                python_executable=python_executable,
            )
            job["expected_resolved_training_config"] = _expected_development_resolved_profile_config(job)
            jobs.append(job)
    return jobs


def initialize_development_registry(
    *,
    root: str | Path = DEFAULT_DEVELOPMENT_ROOT,
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    store_root: str | Path = DEFAULT_DEVELOPMENT_STORE_ROOT,
    profiles: Sequence[str] = DEFAULT_DEVELOPMENT_PROFILES,
    development_years: Sequence[int] = DEVELOPMENT_YEARS,
    seed: int = DEVELOPMENT_SEED,
    maximum_epochs: int = 10,
    patience: int = 2,
    min_delta: float = 0.0,
    minimum_complete_epochs: int = 1,
    train_start_year: int = 2012,
    python_executable: str | Path = DEFAULT_PYTHON,
    require_corrected_source: bool = True,
    require_existing_folds: bool = True,
    evidence_policy: str = EVIDENCE_POLICY_RUN_ARTIFACTS,
    kpi_portfolio_contract_path: str | Path | None = None,
    additional_provenance_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    years = _validate_exact_years(development_years, DEVELOPMENT_YEARS, label="development")
    selected_profiles = _normalize_development_profiles(profiles)
    if int(seed) != DEVELOPMENT_SEED:
        raise ValueError(f"development discovery seed is frozen at {DEVELOPMENT_SEED}")
    if int(maximum_epochs) != 10:
        raise ValueError("approved development maximum_epochs is frozen at 10")
    if int(patience) != 2:
        raise ValueError("approved development early-stopping patience is frozen at 2")
    if int(minimum_complete_epochs) != 1:
        raise ValueError("approved development minimum_complete_epochs is frozen at 1")
    if not math.isfinite(float(min_delta)) or float(min_delta) < 0.0:
        raise ValueError("development early-stopping min_delta must be finite and non-negative")
    if int(train_start_year) != 2012:
        raise ValueError("approved development train_start_year is frozen at 2012")
    if evidence_policy not in {EVIDENCE_POLICY_RUN_ARTIFACTS, EVIDENCE_POLICY_SYNTHETIC_ALLOWED}:
        raise ValueError(f"unsupported evidence policy: {evidence_policy}")
    if evidence_policy == EVIDENCE_POLICY_RUN_ARTIFACTS and (
        not bool(require_corrected_source) or not bool(require_existing_folds)
    ):
        raise ValueError("run-artifact development registries require corrected source and existing folds")
    if evidence_policy == EVIDENCE_POLICY_RUN_ARTIFACTS and kpi_portfolio_contract_path is None:
        raise ValueError(
            "formal development registration is blocked until a KPI/portfolio net-execution contract is provided"
        )
    source_path, _, source_sha256 = _validate_development_source_view(
        source_view,
        require_corrected_contract=bool(require_corrected_source),
    )
    store_root_path = _workspace_path(store_root).resolve()
    fold_bindings = _development_fold_bindings(
        years=years,
        store_root=store_root_path,
        source_view_sha256=source_sha256,
        train_start_year=int(train_start_year),
        require_existing=bool(require_existing_folds),
    )
    approved = approved_development_contract_binding()
    jobs = _build_development_jobs(
        profiles=selected_profiles,
        years=years,
        maximum_epochs=int(maximum_epochs),
        patience=int(patience),
        min_delta=float(min_delta),
        minimum_complete_epochs=int(minimum_complete_epochs),
        root=root_path,
        fold_bindings=fold_bindings,
        python_executable=_workspace_path(python_executable),
    )
    contract_paths = (
        ()
        if kpi_portfolio_contract_path is None
        else (_workspace_path(kpi_portfolio_contract_path).resolve(),)
    )
    extra_paths = (
        *contract_paths,
        *tuple(_workspace_path(path).resolve() for path in additional_provenance_paths),
    )
    missing_extra = [str(path) for path in extra_paths if not path.is_file()]
    if missing_extra:
        raise FileNotFoundError(f"additional development provenance paths are missing: {missing_extra}")
    provenance_paths = (*DEVELOPMENT_CODE_PROVENANCE_PATHS, *extra_paths)
    if len({str(_workspace_path(path).resolve()) for path in provenance_paths}) != len(provenance_paths):
        raise ValueError("development provenance paths must be unique")
    code_provenance = _current_code_provenance(provenance_paths)
    body = {
        "schema_version": 1,
        "artifact_type": "seq100_development_walkforward_registry",
        "contract_id": str(approved["contract_id"]),
        "research_contract": approved,
        "status": "registered",
        "study_root": str(root_path.resolve()),
        "source_view": str(source_path.resolve()),
        "source_view_sha256": source_sha256,
        "store_root": str(store_root_path),
        "fold_bindings": fold_bindings,
        "evidence_policy": str(evidence_policy),
        "code_provenance": code_provenance,
        "additional_provenance_paths": [str(path) for path in extra_paths],
        "kpi_portfolio_contract": (
            None
            if kpi_portfolio_contract_path is None
            else {
                "path": str(contract_paths[0]),
                "sha256": _file_sha256(contract_paths[0]),
            }
        ),
        "train_start_year": int(train_start_year),
        "development_years": list(years),
        "evidence_role": "historical_development_for_checkpoint_model_and_champion_selection",
        "historical_test_set": None,
        "historical_outer_audit": False,
        "profiles": [
            {"profile": profile, "profile_command": PROFILE_COMMANDS[profile]}
            for profile in selected_profiles
        ],
        "training_protocol": {
            "seed": DEVELOPMENT_SEED,
            "max_samples_per_split": 0,
            "full_fold_data_required": True,
            "low_budget_screen_used_for_selection": False,
            "maximum_epochs": int(maximum_epochs),
            "early_stopping": {
                "metric": "development_total_loss",
                "mode": "min",
                "patience": int(patience),
                "min_delta": float(min_delta),
                "minimum_complete_epochs": int(minimum_complete_epochs),
                "restore_best_checkpoint": True,
            },
        },
        "kpi_contract": {
            "metric_names": list(DEVELOPMENT_METRIC_NAMES),
            "top_k": list(DEVELOPMENT_TOP_K),
            "optimization_metric": "development_total_loss",
            "checkpoint_selection_uses_topk": False,
            "champion_primary": "equal_weight_top1_3_5_10_alpha_net_realized_plan_return_base",
            "net_execution_columns_required": [
                "gross_realized_plan_return",
                "net_realized_plan_return_base",
                "net_realized_plan_return_stress",
                "net_realized_plan_value_base",
                "net_realized_plan_value_stress",
                "realized_plan_covered",
            ],
            "legacy_gross_realized_plan_columns_allowed_for_selection": False,
            "realized_plan_coverage_required": 1.0,
            "base_and_double_slippage_scenarios_required": True,
            "execution_cost_contract_sha256_required": True,
            "model_selection": DEVELOPMENT_SELECTION_POLICY,
        },
        "fold_builds": _development_fold_build_commands(
            years=years,
            source_view=source_path,
            store_root=store_root_path,
            train_start_year=int(train_start_year),
            python_executable=_workspace_path(python_executable),
        ),
        "jobs": jobs,
        "protected_boundaries": {
            "active_execution_changed": False,
            "qdp_active_changed": False,
            "legacy_artifacts_overwritten": False,
            "retired_generation_resumed": False,
        },
    }
    registry = {**body, "registry_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "development_registry.json", registry, immutable=True)
    _initialize_ledger(root_path)
    return {**registry, "registry_path": str(path.resolve())}


def initialize_candidate_registry(
    *,
    root: str | Path = DEFAULT_ROOT,
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    screen_seeds: Sequence[int] = SCREEN_SEEDS,
    screen_epochs: int = 2,
    screen_max_samples_per_split: int = 0,
    train_start_year: int = 2012,
    python_executable: str | Path = DEFAULT_PYTHON,
    require_corrected_source: bool = True,
    require_existing_folds: bool = True,
    evidence_policy: str = EVIDENCE_POLICY_RUN_ARTIFACTS,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    source_path, _, source_sha256 = _validate_source_view(
        source_view,
        require_corrected_contract=bool(require_corrected_source),
    )
    store_root_path = _workspace_path(store_root).resolve()
    years = _validate_exact_years(INNER_OOS_YEARS, INNER_OOS_YEARS, label="inner OOS")
    seeds = _parse_ints(screen_seeds)
    if int(screen_epochs) < 2 and "hard_st_global_tail" in PROFILE_ORDER:
        raise ValueError("screen_epochs must be at least 2 so global-tail uses prior-epoch hard negatives")
    if int(screen_max_samples_per_split) < 0:
        raise ValueError("screen_max_samples_per_split must be non-negative")
    if evidence_policy not in {EVIDENCE_POLICY_RUN_ARTIFACTS, EVIDENCE_POLICY_SYNTHETIC_ALLOWED}:
        raise ValueError(f"unsupported evidence policy: {evidence_policy}")
    if evidence_policy == EVIDENCE_POLICY_RUN_ARTIFACTS and (
        not bool(require_corrected_source) or not bool(require_existing_folds)
    ):
        raise ValueError("run-artifact registries require corrected-source and existing-fold verification")
    fold_bindings = _fold_bindings(
        years=years,
        store_root=store_root_path,
        source_view_sha256=source_sha256,
        train_start_year=int(train_start_year),
        require_existing=bool(require_existing_folds),
    )
    jobs = _build_jobs(
        phase="screen",
        profiles=PROFILE_ORDER,
        years=years,
        seeds=seeds,
        epochs=int(screen_epochs),
        max_samples_per_split=int(screen_max_samples_per_split),
        root=root_path,
        fold_bindings=fold_bindings,
        python_executable=_workspace_path(python_executable),
    )
    code_provenance = _current_code_provenance()
    body = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "artifact_type": "seq100_candidate_registry",
        "contract_id": CONTRACT_ID,
        "status": "registered",
        "study_root": str(root_path.resolve()),
        "source_view": str(source_path.resolve()),
        "source_view_sha256": source_sha256,
        "store_root": str(store_root_path),
        "fold_bindings": fold_bindings,
        "evidence_policy": str(evidence_policy),
        "code_provenance": code_provenance,
        "train_start_year": int(train_start_year),
        "inner_oos_years": list(years),
        "historical_outer_oos_years": list(OUTER_OOS_YEARS),
        "outer_evidence_grade": "selection_aware_historical_oos",
        "profiles": [
            {"profile": profile, "profile_command": PROFILE_COMMANDS[profile]}
            for profile in PROFILE_ORDER
        ],
        "phase": {
            "name": "screen",
            "seeds": list(seeds),
            "epochs": int(screen_epochs),
            "max_samples_per_split": int(screen_max_samples_per_split),
        },
        "selection_policy": SELECTION_POLICY,
        "fold_builds": _fold_build_commands(
            years=years,
            source_view=source_path,
            store_root=store_root_path,
            train_start_year=int(train_start_year),
            python_executable=_workspace_path(python_executable),
        ),
        "jobs": jobs,
        "protected_boundaries": {
            "active_execution_changed": False,
            "qdp_active_changed": False,
            "legacy_artifacts_overwritten": False,
        },
    }
    registry = {**body, "registry_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "candidate_registry.json", registry, immutable=True)
    _initialize_ledger(root_path)
    return {**registry, "registry_path": str(path.resolve())}


def _initialize_ledger(root: Path) -> dict[str, Any]:
    path = root / "result_ledger.json"
    if path.exists():
        return _read_json(path)
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_research_result_ledger",
        "created_at": _now(),
        "updated_at": _now(),
        "results": {},
    }
    _write_json(path, payload, immutable=False)
    return payload


def _registry_jobs(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    validated = _validated_registry(registry)
    raw_jobs = list(validated.get("jobs", []) or [])
    jobs = {str(item["job_id"]): dict(item) for item in raw_jobs}
    if len(jobs) != len(raw_jobs):
        raise ValueError("registry contains duplicate job ids")
    return jobs


def _finite_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"metric {name} must be finite")
    return parsed


def _command_option(command: Sequence[str], name: str) -> str:
    try:
        index = list(command).index(name)
    except ValueError as exc:
        raise ValueError(f"registered command is missing {name}") from exc
    if index + 1 >= len(command):
        raise ValueError(f"registered command has no value for {name}")
    return str(command[index + 1])


def _expected_resolved_profile_config(job: Mapping[str, Any]) -> dict[str, Any]:
    from daily_research.path_policy.seq100_mainline import PROFILE_SPECS

    profile_command = str(job["profile_command"])
    profile = PROFILE_SPECS[profile_command].default_profile()
    return {
        "epochs": int(job["epochs"]),
        "batch_size": int(profile.batch_size),
        "model_type": str(profile.model_type),
        "hidden_dim": int(profile.hidden_dim),
        "layers": int(profile.layers),
        "dropout": float(profile.dropout),
        "symbol_embedding_dim": 16,
        "learning_rate": float(profile.learning_rate),
        "weight_decay": float(profile.weight_decay),
        "path_loss_weight": float(profile.path_loss_weight),
        "path_loss_profile": str(profile.path_loss_profile),
        "summary_loss_weight": float(profile.summary_loss_weight),
        "richer_loss_weight": float(profile.richer_loss_weight),
        "price_delta_loss_weight": float(profile.price_delta_loss_weight),
        "va_level_loss_weight": float(profile.va_level_loss_weight),
        "va_delta_loss_weight": float(profile.va_delta_loss_weight),
        "value_loss_weight": float(profile.value_loss_weight),
        "rank_loss_weight": float(profile.rank_loss_weight),
        "residual_score_weight": 0.25,
        "residual_penalty_weight": 0.01,
        "summary_loss_profile": str(profile.summary_loss_profile),
        "input_channel_profile": str(profile.input_channel_profile),
        "direct_value_horizon": int(profile.direct_value_horizon),
        "rank_max_per_side": int(profile.rank_max_per_side),
        "device": "cuda",
        "amp": True,
        "seed": int(job["seed"]),
        "top_k": [int(item) for item in str(profile.top_k).split(",") if item],
        "max_samples_per_split": int(job.get("max_samples_per_split", 0) or 0),
        "prediction_mode": "none",
        "early_stopping_patience": 0,
        "early_stopping_min_delta": 0.0,
        "evaluation_mode": "fixed_oos",
        "path_value_gradient_profile": str(profile.path_value_gradient_profile),
        "rank_training_profile": str(profile.rank_training_profile),
        "rank_batch_size": int(profile.rank_batch_size),
        "rank_interval": int(profile.rank_interval),
        "prefetch_batches": int(profile.prefetch_batches),
    }


def _validate_development_run_job_binding(summary: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    command = list(job.get("command", []) or [])
    expected = {
        "run_tag": _command_option(command, "--run-tag"),
        "seed": DEVELOPMENT_SEED,
        "development_year": int(job["development_year"]),
        "pack_manifest": str(Path(str(job["view_path"])).resolve()),
    }
    actual = {
        "run_tag": str(summary.get("run_tag", "")),
        "seed": int(summary.get("seed", -1)),
        "development_year": int(summary.get("development_year", -1)),
        "pack_manifest": str(Path(str(summary.get("pack_manifest", ""))).resolve()),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(f"development run summary does not match registered job for {key}")
    output_dir = _workspace_path(str(summary.get("output_dir", "") or "")).resolve()
    if not output_dir.is_dir():
        raise ValueError("development run summary output_dir is missing")
    if str(summary.get("evaluation_mode", "")) != "development":
        raise ValueError("development results require development evaluation mode")
    if list(summary.get("evaluation_splits", []) or []) != ["development"]:
        raise ValueError("development results must evaluate only the development split")
    if str(summary.get("evaluation_split", "")) != "development":
        raise ValueError("development evaluation_split must be development")
    if str(summary.get("checkpoint_policy", "")) != "best_development_total_loss":
        raise ValueError("development results require best_development_total_loss checkpoints")
    maximum_epochs = int(job["maximum_epochs"])
    completed_epochs = int(summary.get("completed_epochs", -1))
    best_epoch = int(summary.get("best_epoch", -1))
    if int(summary.get("epochs", -1)) != maximum_epochs:
        raise ValueError("development run maximum epoch budget changed")
    if not 1 <= best_epoch <= completed_epochs <= maximum_epochs:
        raise ValueError("development best/completed epoch provenance is invalid")
    if int(summary.get("best_optimizer_step", 0) or 0) <= 0:
        raise ValueError("development run must report best_optimizer_step")
    if int(summary.get("optimizer_step_count", 0) or 0) <= 0:
        raise ValueError("development run must report optimizer_step_count")
    if int(summary.get("max_samples_per_split", -1)) != 0:
        raise ValueError("development jobs require all fold training samples")
    early = dict(summary.get("early_stopping", {}) or {})
    registered_early = dict(job.get("early_stopping", {}) or {})
    for key in ("metric", "mode", "patience", "min_delta", "minimum_complete_epochs"):
        if early.get(key) != registered_early.get(key):
            raise ValueError(f"development early-stopping config mismatch for {key}")
    if bool(early.get("restore_best_checkpoint", False)) is not True:
        raise ValueError("development run must restore the best checkpoint")
    if not math.isfinite(float(early.get("best_value", math.nan))):
        raise ValueError("development early-stopping best_value must be finite")
    resolved = dict(summary.get("resolved_training_config", {}) or {})
    expected_resolved = dict(job.get("expected_resolved_training_config", {}) or {})
    if not expected_resolved:
        raise ValueError("development job is missing resolved training-config snapshot")
    for key, expected_value in expected_resolved.items():
        if resolved.get(key) != expected_value:
            raise ValueError(f"development resolved training config mismatch for {key}")

    approved = approved_development_contract_binding()
    if dict(summary.get("research_contract", {}) or {}) != approved:
        raise ValueError("development run is not bound to the approved contract")
    view_path = _workspace_path(str(job["view_path"])).resolve()
    expected_view_sha = str(job.get("view_sha256", ""))
    if expected_view_sha:
        if not view_path.is_file() or _file_sha256(view_path) != expected_view_sha:
            raise ValueError("registered development fold view changed")
        manifest = _read_json(view_path)
        fold_contract = _validated_development_fold_training_contract(manifest)
        expected_fold_sha = str(job.get("development_fold_training_contract_sha256", ""))
        if str(fold_contract.get("sha256", "")) != expected_fold_sha:
            raise ValueError("registered development fold contract changed")
        summary_fold = dict(summary.get("development_fold_training_contract", {}) or {})
        if str(summary_fold.get("sha256", "")) != expected_fold_sha:
            raise ValueError("run summary development fold contract mismatch")
        if str(fold_contract.get("candidate_index_sha256", "")) != str(
            job.get("candidate_index_sha256", "")
        ):
            raise ValueError("registered development candidate index changed")
        candidate_path = _workspace_path(str(manifest.get("candidate_index_path", "") or "")).resolve()
        if str(Path(str(summary.get("candidate_index_path", ""))).resolve()) != str(candidate_path):
            raise ValueError("development summary candidate_index_path mismatch")
        if str(summary.get("candidate_index_sha256", "")) != str(
            fold_contract.get("candidate_index_sha256", "")
        ):
            raise ValueError("development summary candidate index SHA mismatch")
        execution_cost_sha256 = _payload_sha256(dict(manifest.get("execution_cost_contract", {}) or {}))
        if execution_cost_sha256 != str(job.get("execution_cost_contract_sha256", "")):
            raise ValueError("registered execution cost contract changed")
        if str(summary.get("execution_cost_contract_sha256", "")) != execution_cost_sha256:
            raise ValueError("development summary execution cost contract SHA mismatch")

    selection = dict(summary.get("sample_selection", {}) or {})
    train_selection = dict(selection.get("train", {}) or {})
    candidate_selection = dict(selection.get("development_candidates", {}) or {})
    if not train_selection or not candidate_selection:
        raise ValueError("development summary is missing train/candidate sample-selection provenance")
    if (
        str(train_selection.get("policy", "")) != "all_rows"
        or int(train_selection.get("requested_max_samples", -1)) != 0
        or int(train_selection.get("selected_row_count", -1))
        != int(train_selection.get("original_row_count", -2))
    ):
        raise ValueError("development training must consume all supervised train rows")
    if (
        str(candidate_selection.get("policy", "")) != "all_candidates"
        or int(candidate_selection.get("selected_row_count", -1))
        != int(candidate_selection.get("original_row_count", -2))
    ):
        raise ValueError("development evaluation must score the full candidate universe")
    expected_value_profile = "hard_st" if str(job["profile"]) in {"hard_st", "hard_st_global_tail"} else "smooth_current"
    expected_rank_profile = "global_tail_512" if str(job["profile"]) == "hard_st_global_tail" else "local_chunk"
    if str(summary.get("path_value_gradient_profile", "")) != expected_value_profile:
        raise ValueError("development path-value profile does not match registered profile")
    if str(summary.get("rank_training_profile", "")) != expected_rank_profile:
        raise ValueError("development ranking profile does not match registered profile")


def _validate_run_job_binding(summary: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("evaluation_mode", "")) == "development":
        _validate_development_run_job_binding(summary, job)
        return
    command = list(job.get("command", []) or [])
    expected = {
        "run_tag": _command_option(command, "--run-tag"),
        "seed": int(job["seed"]),
        "fold_year": int(job["oos_year"]),
        "pack_manifest": str(Path(str(job["view_path"])).resolve()),
    }
    actual = {
        "run_tag": str(summary.get("run_tag", "")),
        "seed": int(summary.get("seed", -1)),
        "fold_year": int(summary.get("fold_year", -1)),
        "pack_manifest": str(Path(str(summary.get("pack_manifest", ""))).resolve()),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(f"run summary does not match registered job for {key}")
    output_dir = _workspace_path(str(summary.get("output_dir", "") or "")).resolve()
    if not output_dir.is_dir():
        raise ValueError("run summary output_dir is missing")
    if str(summary.get("evaluation_mode", "")) != "fixed_oos":
        raise ValueError("research-generation results require fixed_oos evaluation")
    if str(summary.get("checkpoint_policy", "")) != "final_epoch":
        raise ValueError("research-generation results require final_epoch checkpoints")
    expected_epochs = int(job["epochs"])
    for field in ("epochs", "completed_epochs", "best_epoch"):
        if int(summary.get(field, -1)) != expected_epochs:
            raise ValueError(f"run summary does not match registered job for {field}")
    expected_max_samples = int(job.get("max_samples_per_split", 0) or 0)
    if int(summary.get("max_samples_per_split", -1)) != expected_max_samples:
        raise ValueError("run summary does not match registered job for max_samples_per_split")
    resolved = dict(summary.get("resolved_training_config", {}) or {})
    expected_resolved = dict(job.get("expected_resolved_training_config", {}) or {})
    if not expected_resolved:
        raise ValueError("registered job is missing its resolved training-config snapshot")
    for key, expected_value in expected_resolved.items():
        if resolved.get(key) != expected_value:
            raise ValueError(f"resolved training config does not match registered job for {key}")
    view_path = _workspace_path(str(job["view_path"])).resolve()
    expected_view_sha = str(job.get("view_sha256", ""))
    if expected_view_sha:
        if not view_path.is_file() or _file_sha256(view_path) != expected_view_sha:
            raise ValueError("registered fold view is missing or its SHA-256 changed")
        fold_manifest = _read_json(view_path)
        fold_contract = _validated_fold_training_contract(fold_manifest)
        expected_fold_sha = str(job.get("fold_training_contract_sha256", ""))
        if str(fold_contract.get("sha256", "")) != expected_fold_sha:
            raise ValueError("registered fold training contract changed")
        summary_fold = dict(summary.get("fold_training_contract", {}) or {})
        if str(summary_fold.get("sha256", "")) != expected_fold_sha:
            raise ValueError("run summary fold training contract does not match the registered fold")
        provenance = dict(fold_manifest.get("source_view_provenance", {}) or {})
        if str(provenance.get("manifest_sha256", "")) != str(job.get("source_view_sha256", "")):
            raise ValueError("registered fold source identity changed")
    selection = dict(summary.get("sample_selection", {}) or {})
    train_selection = dict(selection.get("train", {}) or {})
    oos_selection = dict(selection.get("oos", {}) or {})
    if not train_selection or not oos_selection:
        raise ValueError("run summary is missing train/OOS sample-selection provenance")
    if int(train_selection.get("requested_max_samples", -1)) != expected_max_samples:
        raise ValueError("train sample-selection limit does not match the registered job")
    if expected_max_samples > 0 and int(train_selection.get("original_row_count", 0)) > expected_max_samples:
        if str(train_selection.get("policy", "")) != "date_complete_even_spread_v1":
            raise ValueError("capped training must use complete dates spread across history")
        if int(train_selection.get("selected_row_count", expected_max_samples + 1)) > expected_max_samples:
            raise ValueError("capped training exceeded the registered sample limit")
    if str(oos_selection.get("policy", "")) != "all_rows":
        raise ValueError("OOS evaluation must remain full-universe")
    if int(oos_selection.get("selected_row_count", -1)) != int(oos_selection.get("original_row_count", -2)):
        raise ValueError("OOS evaluation sample selection is incomplete")
    expected_value_profile = "smooth_current"
    expected_rank_profile = "local_chunk"
    if str(job["profile"]) in {"hard_st", "hard_st_global_tail"}:
        expected_value_profile = "hard_st"
    if str(job["profile"]) == "hard_st_global_tail":
        expected_rank_profile = "global_tail_512"
    if str(summary.get("path_value_gradient_profile", "")) != expected_value_profile:
        raise ValueError("run path-value gradient profile does not match registered profile")
    if str(summary.get("rank_training_profile", "")) != expected_rank_profile:
        raise ValueError("run ranking profile does not match registered profile")


def _development_metrics_from_run_dir(
    root: Path,
    *,
    summary: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, str]]]:
    checkpoint_path = _workspace_path(str(summary.get("best_checkpoint", "") or "")).resolve()
    if checkpoint_path.parent != root.resolve() or not checkpoint_path.is_file():
        raise ValueError("development best checkpoint must be an existing file inside the run directory")
    topk_path = root / "topk_metrics.csv"
    split_path = root / "split_metrics.csv"
    topk = pd.read_csv(topk_path)
    split = pd.read_csv(split_path)
    if "split" in topk.columns:
        topk = topk.loc[topk["split"].astype(str).eq("development")]
    if "split" in split.columns:
        split = split.loc[split["split"].astype(str).eq("development")]
    if len(split) != 1:
        raise ValueError("development outputs must contain exactly one development split row")

    def pick(row: pd.Series, names: Sequence[str], label: str) -> float:
        for name in names:
            if name in row.index and pd.notna(row[name]):
                return _finite_float(row[name], name=label)
        raise ValueError(f"development output is missing {label}; tried {list(names)}")

    metrics: dict[str, float] = {}
    for top_k in DEVELOPMENT_TOP_K:
        rows = topk.loc[pd.to_numeric(topk["top_k"], errors="coerce").eq(int(top_k))]
        if len(rows) != 1:
            raise ValueError(f"development outputs require exactly one Top{top_k} row")
        row = rows.iloc[0]
        prefix = f"top{top_k}"
        metrics[f"{prefix}_opportunity_alpha"] = pick(
            row, ["alpha_opportunity_value"], f"{prefix}_opportunity_alpha"
        )
        for metric_name in (
            "net_realized_plan_return_base",
            "net_realized_plan_return_stress",
            "net_realized_plan_value_base",
            "net_realized_plan_value_stress",
        ):
            metrics[f"{prefix}_{metric_name}_alpha"] = pick(
                row,
                [f"alpha_{metric_name}"],
                f"{prefix}_{metric_name}_alpha",
            )
        metrics[f"{prefix}_oracle_regret"] = pick(
            row, ["selected_oracle_regret", "alpha_oracle_regret"], f"{prefix}_oracle_regret"
        )
        metrics[f"{prefix}_entry_fill_rate"] = pick(
            row,
            ["selected_entry_fill_rate", "selected_realized_fill_rate"],
            f"{prefix}_entry_fill_rate",
        )
        metrics[f"{prefix}_realized_plan_coverage"] = pick(
            row,
            ["selected_realized_plan_coverage"],
            f"{prefix}_realized_plan_coverage",
        )
        if metrics[f"{prefix}_realized_plan_coverage"] != 1.0:
            raise ValueError(f"{prefix} candidate-complete net realized-plan coverage must equal 1.0")
    early = dict(summary.get("early_stopping", {}) or {})
    metrics["development_total_loss"] = _finite_float(
        early.get("best_value", math.nan), name="development_total_loss"
    )
    split_row = split.iloc[0]
    metrics["daily_rank_ic"] = pick(split_row, ["rank_ic_mean"], "daily_rank_ic")
    metrics["path_mae"] = pick(split_row, ["path_mae"], "path_mae")
    if set(metrics) != set(DEVELOPMENT_METRIC_NAMES):
        raise AssertionError("development KPI extraction drifted from the registered metric contract")
    evidence_paths = [
        root / "sequence_path_training_summary.json",
        topk_path,
        split_path,
        root / "daily_topk_metrics.csv",
        root / "daily_rank_ic.csv",
        root / "training_history.csv",
        root / "topk_candidates.parquet",
        checkpoint_path,
    ]
    evidence_files: list[dict[str, str]] = []
    for path in evidence_paths:
        resolved_path = path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"registered development evidence is missing: {resolved_path}")
        evidence_files.append({"path": str(resolved_path), "sha256": _file_sha256(resolved_path)})
    return metrics, evidence_files


def _metrics_from_run_dir(
    run_dir: str | Path,
    *,
    job: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, str]]]:
    root = _workspace_path(run_dir)
    summary_path = root / "sequence_path_training_summary.json"
    summary = _read_json(summary_path)
    _validate_run_job_binding(summary, job)
    if _workspace_path(str(summary.get("output_dir", "") or "")).resolve() != root.resolve():
        raise ValueError("run directory does not match summary output_dir")
    if str(job.get("evaluation_mode", "")) == "development":
        return _development_metrics_from_run_dir(root, summary=summary, job=job)
    checkpoint_path = _workspace_path(str(summary.get("best_checkpoint", "") or "")).resolve()
    if checkpoint_path.parent != root.resolve() or checkpoint_path.name != "final_model.pt":
        raise ValueError("fixed-OOS evidence must bind the final checkpoint inside the run directory")
    topk_path = root / "topk_metrics.csv"
    split_path = root / "split_metrics.csv"
    topk = pd.read_csv(topk_path)
    split = pd.read_csv(split_path)
    if "split" in topk.columns:
        topk = topk.loc[topk["split"].astype(str).eq("oos")]
    if "split" in split.columns:
        split = split.loc[split["split"].astype(str).eq("oos")]
    top3 = topk.loc[pd.to_numeric(topk["top_k"], errors="coerce").eq(3)]
    top10 = topk.loc[pd.to_numeric(topk["top_k"], errors="coerce").eq(10)]
    if len(top3) != 1 or len(top10) != 1 or len(split) != 1:
        raise ValueError("run outputs must contain exactly one OOS Top3, Top10, and split row")
    row3, row10, split_row = top3.iloc[0], top10.iloc[0], split.iloc[0]

    def pick(row: pd.Series, names: Sequence[str], label: str) -> float:
        for name in names:
            if name in row.index and pd.notna(row[name]):
                return _finite_float(row[name], name=label)
        raise ValueError(f"run output is missing {label}; tried {list(names)}")

    metrics = {
        "top3_opportunity_alpha": pick(row3, ["alpha_opportunity_value"], "top3_opportunity_alpha"),
        "top3_realized_plan_alpha": pick(row3, ["alpha_realized_plan_value"], "top3_realized_plan_alpha"),
        "oracle_regret": pick(row3, ["selected_oracle_regret", "alpha_oracle_regret"], "oracle_regret"),
        "top10_opportunity_alpha": pick(row10, ["alpha_opportunity_value"], "top10_opportunity_alpha"),
        "daily_rank_ic": pick(split_row, ["rank_ic_mean"], "daily_rank_ic"),
        "fill_rate": pick(row3, ["selected_entry_fill_rate", "selected_realized_fill_rate"], "fill_rate"),
        "path_mae": pick(split_row, ["path_mae"], "path_mae"),
    }
    evidence_paths = [
        summary_path,
        topk_path,
        split_path,
        root / "daily_topk_metrics.csv",
        root / "daily_rank_ic.csv",
        root / "training_history.csv",
        root / "topk_candidates.parquet",
        checkpoint_path,
    ]
    evidence_files: list[dict[str, str]] = []
    for path in evidence_paths:
        resolved_path = path.resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"registered run evidence is missing: {resolved_path}")
        evidence_files.append({"path": str(resolved_path), "sha256": _file_sha256(resolved_path)})
    return metrics, evidence_files


def register_result(
    *,
    root: str | Path,
    registry_path: str | Path,
    job_id: str,
    metrics: Mapping[str, Any] | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    registry = _validated_registry(_read_json(registry_path))
    root_path = _validated_study_root(registry, root_path)
    _validate_code_provenance(dict(registry.get("code_provenance", {}) or {}))
    job = _registry_jobs(registry).get(str(job_id))
    if job is None:
        raise KeyError(f"job is not registered: {job_id}")
    if (metrics is None) == (run_dir is None):
        raise ValueError("provide exactly one of metrics or run_dir")
    if run_dir is not None:
        normalized, evidence_files = _metrics_from_run_dir(run_dir, job=job)
        evidence_source = "run_artifacts"
    else:
        if str(registry.get("evidence_policy", "")) != EVIDENCE_POLICY_SYNTHETIC_ALLOWED:
            raise ValueError("this registry accepts only verifiable run artifacts")
        assert metrics is not None
        metric_names = (
            DEVELOPMENT_METRIC_NAMES
            if str(job.get("evaluation_mode", "")) == "development"
            else METRIC_NAMES
        )
        missing = sorted(set(metric_names) - set(metrics))
        extra = sorted(set(metrics) - set(metric_names))
        if missing or extra:
            raise ValueError(f"metrics mismatch: missing={missing}, extra={extra}")
        normalized = {name: _finite_float(metrics[name], name=name) for name in metric_names}
        evidence_files = []
        evidence_source = "synthetic_metrics"
    record_body: dict[str, Any] = {
        "job_id": str(job_id),
        "registry_sha256": str(registry["registry_sha256"]),
        "profile": str(job["profile"]),
        "phase": str(job["phase"]),
        "seed": int(job["seed"]),
        "metrics": normalized,
        "evidence_source": evidence_source,
        "evidence_files": evidence_files,
    }
    if str(job.get("evaluation_mode", "")) == "development":
        record_body["development_year"] = int(job["development_year"])
        record_body["checkpoint_policy"] = "best_development_total_loss"
        if run_dir is not None:
            run_summary = _read_json(_workspace_path(run_dir) / "sequence_path_training_summary.json")
            early = dict(run_summary.get("early_stopping", {}) or {})
            record_body["training_outcome"] = {
                "best_epoch": int(run_summary["best_epoch"]),
                "completed_epochs": int(run_summary["completed_epochs"]),
                "best_optimizer_step": int(run_summary["best_optimizer_step"]),
                "optimizer_step_count": int(run_summary["optimizer_step_count"]),
                "stopped_early": bool(early.get("stopped_early", False)),
                "development_total_loss": float(early["best_value"]),
            }
    else:
        record_body["oos_year"] = int(job["oos_year"])
    record = {**record_body, "result_sha256": _payload_sha256(record_body)}
    ledger_path = _validated_ledger_path(registry, root_path / "result_ledger.json")
    with _exclusive_update_lock(root_path / "result_ledger.lock"):
        ledger = _initialize_ledger(root_path)
        results = dict(ledger.get("results", {}) or {})
        existing = results.get(str(job_id))
        if existing is not None and existing != record:
            raise ValueError(f"result already registered with different evidence: {job_id}")
        results[str(job_id)] = record
        ledger["results"] = results
        ledger["updated_at"] = _now()
        _write_json(ledger_path, ledger, immutable=False)
    return record


def _completed_results(
    registry: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    validated_registry = _validated_registry(registry)
    _validate_code_provenance(dict(validated_registry.get("code_provenance", {}) or {}))
    jobs = _registry_jobs(validated_registry)
    results = dict(ledger.get("results", {}) or {})
    missing = sorted(set(jobs) - set(results))
    if missing:
        raise RuntimeError(f"registered matrix is incomplete; missing {len(missing)} jobs, first={missing[:3]}")
    records: list[dict[str, Any]] = []
    for job_id, job in jobs.items():
        record = dict(results[job_id])
        declared_result_sha = str(record.get("result_sha256", ""))
        record_body = {key: value for key, value in record.items() if key != "result_sha256"}
        if not declared_result_sha or _payload_sha256(record_body) != declared_result_sha:
            raise ValueError(f"result digest mismatch: {job_id}")
        if str(record.get("registry_sha256", "")) != str(validated_registry.get("registry_sha256", "")):
            raise ValueError(f"result registry binding mismatch: {job_id}")
        binding_keys = ["profile", "phase", "seed"]
        binding_keys.append(
            "development_year" if str(job.get("evaluation_mode", "")) == "development" else "oos_year"
        )
        for key in binding_keys:
            if record.get(key) != job.get(key):
                raise ValueError(f"result job binding mismatch for {job_id}: {key}")
        evidence_source = str(record.get("evidence_source", ""))
        policy = str(validated_registry.get("evidence_policy", EVIDENCE_POLICY_RUN_ARTIFACTS))
        if policy == EVIDENCE_POLICY_RUN_ARTIFACTS and evidence_source != "run_artifacts":
            raise ValueError(f"result lacks required run-artifact evidence: {job_id}")
        for item in list(record.get("evidence_files", []) or []):
            evidence_path = _workspace_path(str(item.get("path", "") or "")).resolve()
            if not evidence_path.is_file() or _file_sha256(evidence_path) != str(item.get("sha256", "")):
                raise ValueError(f"result evidence file changed: {job_id}: {evidence_path}")
        records.append(record)
    return records


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = [
        {
            "profile": str(item["profile"]),
            "oos_year": int(item["oos_year"]),
            "seed": int(item["seed"]),
            **{name: float(dict(item["metrics"])[name]) for name in METRIC_NAMES},
        }
        for item in records
    ]
    frame = pd.DataFrame(rows)
    output: dict[str, dict[str, Any]] = {}
    for profile, group in frame.groupby("profile", sort=False):
        fold_values = group.groupby("oos_year", sort=True)["top3_opportunity_alpha"].mean()
        output[str(profile)] = {
            "profile": str(profile),
            "job_count": int(len(group)),
            "seed_count": int(group["seed"].nunique()),
            "fold_count": int(group["oos_year"].nunique()),
            **{name: float(group[name].mean()) for name in METRIC_NAMES},
            "worst_fold_top3_opportunity_alpha": float(fold_values.min()),
            "fold_top3_opportunity_alpha": {str(int(year)): float(value) for year, value in fold_values.items()},
        }
    return output


def _guardrail_verdict(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    policy = dict(SELECTION_POLICY["guardrails_relative_to_baseline"])
    checks = {
        "top3_realized_plan_alpha": float(candidate["top3_realized_plan_alpha"])
        >= float(baseline["top3_realized_plan_alpha"]) + float(policy["top3_realized_plan_alpha_min_delta"]),
        "oracle_regret": float(candidate["oracle_regret"])
        <= float(baseline["oracle_regret"]) + float(policy["oracle_regret_max_delta"]),
        "top10_opportunity_alpha": float(candidate["top10_opportunity_alpha"])
        >= float(baseline["top10_opportunity_alpha"]) + float(policy["top10_opportunity_alpha_min_delta"]),
        "daily_rank_ic": float(candidate["daily_rank_ic"])
        >= float(baseline["daily_rank_ic"]) + float(policy["daily_rank_ic_min_delta"]),
        "worst_fold": float(candidate["worst_fold_top3_opportunity_alpha"])
        >= float(baseline["worst_fold_top3_opportunity_alpha"])
        + float(policy["worst_fold_top3_opportunity_alpha_min_delta"]),
        "fill_rate": float(candidate["fill_rate"])
        >= float(baseline["fill_rate"]) + float(policy["fill_rate_min_delta"]),
        "path_mae": float(candidate["path_mae"])
        <= float(baseline["path_mae"]) * float(policy["path_mae_max_ratio"]),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def select_profiles(
    *,
    registry_path: str | Path,
    ledger_path: str | Path,
) -> dict[str, Any]:
    registry = _validated_registry(_read_json(registry_path))
    canonical_ledger = _validated_ledger_path(registry, ledger_path)
    records = _completed_results(registry, _read_json(canonical_ledger))
    aggregates = _aggregate_records(records)
    baseline = aggregates.get("baseline")
    if baseline is None:
        raise RuntimeError("baseline is required for guardrail comparison")
    ranked: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        aggregate = aggregates.get(profile)
        if aggregate is None:
            continue
        verdict = {"passed": True, "checks": {"baseline_reference": True}} if profile == "baseline" else _guardrail_verdict(aggregate, baseline)
        ranked.append({**aggregate, "guardrails": verdict})
    ranked.sort(
        key=lambda item: (
            bool(dict(item["guardrails"])["passed"]),
            float(item["top3_opportunity_alpha"]),
            float(item["worst_fold_top3_opportunity_alpha"]),
            float(item["top3_realized_plan_alpha"]),
            float(item["daily_rank_ic"]),
        ),
        reverse=True,
    )
    return {
        "registry_sha256": str(registry["registry_sha256"]),
        "phase": str(dict(registry.get("phase", {}))["name"]),
        "ranking": ranked,
        "winner": str(ranked[0]["profile"]),
    }


def _aggregate_development_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = [
        {
            "profile": str(item["profile"]),
            "development_year": int(item["development_year"]),
            "seed": int(item["seed"]),
            **{name: float(dict(item["metrics"])[name]) for name in DEVELOPMENT_METRIC_NAMES},
        }
        for item in records
    ]
    frame = pd.DataFrame(rows)
    output: dict[str, dict[str, Any]] = {}
    for profile, group in frame.groupby("profile", sort=False):
        if sorted(group["development_year"].astype(int).tolist()) != list(DEVELOPMENT_YEARS):
            raise ValueError(f"development profile {profile} does not have exactly one result per required year")
        if set(group["seed"].astype(int).tolist()) != {DEVELOPMENT_SEED}:
            raise ValueError(f"development profile {profile} does not use only seed {DEVELOPMENT_SEED}")
        by_year = group.set_index("development_year").sort_index()
        top3_years = by_year["top3_net_realized_plan_return_base_alpha"].astype(float)
        leave_best = top3_years.drop(index=top3_years.idxmax())
        topk_net_return_base = [
            float(group[f"top{k}_net_realized_plan_return_base_alpha"].mean())
            for k in DEVELOPMENT_TOP_K
        ]
        topk_net_value_base = [
            float(group[f"top{k}_net_realized_plan_value_base_alpha"].mean())
            for k in DEVELOPMENT_TOP_K
        ]
        topk_opportunity = [float(group[f"top{k}_opportunity_alpha"].mean()) for k in DEVELOPMENT_TOP_K]
        output[str(profile)] = {
            "profile": str(profile),
            "job_count": int(len(group)),
            "seed": DEVELOPMENT_SEED,
            "development_year_count": int(group["development_year"].nunique()),
            **{name: float(group[name].mean()) for name in DEVELOPMENT_METRIC_NAMES},
            "mean_topk_net_realized_plan_return_base_alpha": float(
                sum(topk_net_return_base) / len(topk_net_return_base)
            ),
            "mean_topk_net_realized_plan_value_base_alpha": float(
                sum(topk_net_value_base) / len(topk_net_value_base)
            ),
            "mean_topk_opportunity_alpha": float(sum(topk_opportunity) / len(topk_opportunity)),
            "worst_development_year_top3_net_realized_plan_return_base_alpha": float(top3_years.min()),
            "leave_best_year_out_top3_net_realized_plan_return_base_alpha": float(leave_best.mean()),
            "positive_top3_development_years": int((top3_years > 0.0).sum()),
            "development_year_top3_net_realized_plan_return_base_alpha": {
                str(int(year)): float(value) for year, value in top3_years.items()
            },
            "development_year_topk": {
                str(int(year)): {
                    f"top{k}_net_realized_plan_return_base_alpha": float(
                        by_year.loc[year, f"top{k}_net_realized_plan_return_base_alpha"]
                    )
                    for k in DEVELOPMENT_TOP_K
                }
                for year in by_year.index
            },
        }
    return output


def select_development_profiles(
    *,
    registry_path: str | Path,
    ledger_path: str | Path,
) -> dict[str, Any]:
    registry = _validated_registry(_read_json(registry_path))
    if str(registry.get("artifact_type", "")) != "seq100_development_walkforward_registry":
        raise ValueError("development selection requires a development walkforward registry")
    canonical_ledger = _validated_ledger_path(registry, ledger_path)
    approved = approved_development_contract_binding()
    if dict(registry.get("research_contract", {}) or {}) != approved:
        raise ValueError("development registry contract identity changed")
    evidence_policy = str(registry.get("evidence_policy", EVIDENCE_POLICY_RUN_ARTIFACTS))
    source_path, _, source_sha256 = _validate_development_source_view(
        str(registry["source_view"]),
        require_corrected_contract=(evidence_policy == EVIDENCE_POLICY_RUN_ARTIFACTS),
    )
    if source_sha256 != str(registry["source_view_sha256"]):
        raise ValueError("development source changed before selection")
    current_fold_bindings = _development_fold_bindings(
        years=DEVELOPMENT_YEARS,
        store_root=_workspace_path(str(registry["store_root"])).resolve(),
        source_view_sha256=source_sha256,
        train_start_year=int(registry["train_start_year"]),
        require_existing=(evidence_policy == EVIDENCE_POLICY_RUN_ARTIFACTS),
    )
    if current_fold_bindings != dict(registry.get("fold_bindings", {}) or {}):
        raise ValueError("development fold bindings changed before selection")
    records = _completed_results(registry, _read_json(canonical_ledger))
    aggregates = _aggregate_development_records(records)
    profiles = [str(item["profile"]) for item in list(registry.get("profiles", []) or [])]
    ranked: list[dict[str, Any]] = []
    for profile in profiles:
        aggregate = aggregates.get(profile)
        if aggregate is None:
            raise RuntimeError(f"development aggregate is missing profile: {profile}")
        checks = {
            "mean_top3_net_realized_plan_return_base_alpha_positive": float(
                aggregate["top3_net_realized_plan_return_base_alpha"]
            ) > 0.0,
            "mean_top10_net_realized_plan_return_base_alpha_positive": float(
                aggregate["top10_net_realized_plan_return_base_alpha"]
            ) > 0.0,
            "mean_top3_net_realized_plan_return_stress_alpha_positive": float(
                aggregate["top3_net_realized_plan_return_stress_alpha"]
            ) > 0.0,
            "at_least_three_positive_top3_years": int(aggregate["positive_top3_development_years"]) >= 3,
            "candidate_complete_realized_plan_coverage": all(
                float(aggregate[f"top{k}_realized_plan_coverage"]) == 1.0
                for k in DEVELOPMENT_TOP_K
            ),
        }
        ranked.append({**aggregate, "eligibility": {"passed": all(checks.values()), "checks": checks}})
    ranked.sort(
        key=lambda item: (
            bool(dict(item["eligibility"])["passed"]),
            float(item["mean_topk_net_realized_plan_return_base_alpha"]),
            float(item["worst_development_year_top3_net_realized_plan_return_base_alpha"]),
            float(item["leave_best_year_out_top3_net_realized_plan_return_base_alpha"]),
            float(item["mean_topk_net_realized_plan_value_base_alpha"]),
            float(item["mean_topk_opportunity_alpha"]),
            float(item["daily_rank_ic"]),
        ),
        reverse=True,
    )
    winner = str(ranked[0]["profile"]) if ranked and bool(ranked[0]["eligibility"]["passed"]) else None
    return {
        "registry_sha256": str(registry["registry_sha256"]),
        "evidence_role": str(registry["evidence_role"]),
        "development_years": list(DEVELOPMENT_YEARS),
        "selection_policy": DEVELOPMENT_SELECTION_POLICY,
        "ranking": ranked,
        "winner": winner,
        "historical_test_set": None,
        "historical_outer_audit": False,
        "source_view": str(source_path),
    }


def freeze_development_champion(
    *,
    root: str | Path,
    registry_path: str | Path,
    ledger_path: str | Path,
    champion: str,
) -> dict[str, Any]:
    registry = _validated_registry(_read_json(registry_path))
    root_path = _validated_study_root(registry, root)
    canonical_ledger = _validated_ledger_path(registry, ledger_path)
    selection = select_development_profiles(registry_path=registry_path, ledger_path=canonical_ledger)
    selected = selection.get("winner")
    if selected is None:
        raise ValueError("no development profile passes the deployment-oriented eligibility gates")
    if str(champion) != str(selected):
        raise ValueError(f"explicit development champion {champion!r} is not the policy winner {selected!r}")
    records = _completed_results(registry, _read_json(canonical_ledger))
    body = {
        "schema_version": 1,
        "artifact_type": "seq100_development_selected_champion",
        "contract_id": str(registry["contract_id"]),
        "research_contract": dict(registry["research_contract"]),
        "status": "development_selected",
        "study_root": str(root_path.resolve()),
        "champion": str(selected),
        "profile_command": PROFILE_COMMANDS[str(selected)],
        "development_registry": str(_workspace_path(registry_path).resolve()),
        "development_registry_sha256": str(registry["registry_sha256"]),
        "source_view": str(registry["source_view"]),
        "source_view_sha256": str(registry["source_view_sha256"]),
        "fold_bindings": dict(registry["fold_bindings"]),
        "code_provenance": dict(registry["code_provenance"]),
        "ledger_path": str(canonical_ledger.resolve()),
        "ledger_sha256": _file_sha256(canonical_ledger),
        "development_result_sha256s": {
            str(item["job_id"]): str(item["result_sha256"])
            for item in sorted(records, key=lambda value: str(value["job_id"]))
        },
        "selection": selection,
        "development_years_consumed": list(DEVELOPMENT_YEARS),
        "historical_test_set": None,
        "historical_outer_audit": False,
        "next_evidence": "retrain_through_2025_then_begin_2026_true_forward_lockbox",
        "active_execution_changed": False,
        "qdp_active_changed": False,
    }
    frozen = {**body, "freeze_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "development_champion.json", frozen, immutable=True)
    return {**frozen, "freeze_path": str(path.resolve())}


def _candidate_run_directories(job: Mapping[str, Any]) -> list[Path]:
    command = list(job.get("command", []) or [])
    output_root = _workspace_path(_command_option(command, "--output-root")).resolve()
    run_tag = _command_option(command, "--run-tag")
    if not output_root.is_dir():
        return []
    return sorted(
        (path for path in output_root.glob(f"{run_tag}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def run_development_registry(
    *,
    root: str | Path,
    registry_path: str | Path,
    max_jobs: int = 0,
) -> dict[str, Any]:
    registry = _validated_registry(_read_json(registry_path))
    if str(registry.get("artifact_type", "")) != "seq100_development_walkforward_registry":
        raise ValueError("runner requires a development walkforward registry")
    root_path = _validated_study_root(registry, root)
    _validate_code_provenance(dict(registry.get("code_provenance", {}) or {}))
    if int(max_jobs) < 0:
        raise ValueError("max_jobs must be non-negative")
    ledger_path = _validated_ledger_path(registry, root_path / "result_ledger.json")
    ledger = _initialize_ledger(root_path)
    registered = set(dict(ledger.get("results", {}) or {}))
    jobs = list(_registry_jobs(registry).values())
    pending = [job for job in jobs if str(job["job_id"]) not in registered]
    if int(max_jobs) > 0:
        pending = pending[: int(max_jobs)]
    state_path = root_path / "runner_logs" / "development" / "runner_state.json"
    completed_now: list[str] = []
    for job in pending:
        job_id = str(job["job_id"])
        state = {
            "schema_version": 1,
            "artifact_type": "seq100_development_runner_state",
            "status": "recovering_or_running",
            "registry_sha256": str(registry["registry_sha256"]),
            "job_id": job_id,
            "completed_now": list(completed_now),
            "updated_at": _now(),
        }
        _write_json(state_path, state, immutable=False)
        recovered = False
        recovery_errors: list[str] = []
        for run_dir in _candidate_run_directories(job):
            try:
                register_result(
                    root=root_path,
                    registry_path=registry_path,
                    job_id=job_id,
                    run_dir=run_dir,
                )
            except (FileNotFoundError, KeyError, ValueError) as exc:
                recovery_errors.append(f"{run_dir}:{exc}")
                continue
            recovered = True
            break
        if not recovered:
            command = [str(item) for item in list(job.get("command", []) or [])]
            completed = subprocess.run(command, cwd=WORKSPACE_ROOT, check=False)
            if int(completed.returncode) != 0:
                _write_json(
                    state_path,
                    {
                        **state,
                        "status": "failed",
                        "returncode": int(completed.returncode),
                        "recovery_errors": recovery_errors,
                        "updated_at": _now(),
                    },
                    immutable=False,
                )
                raise RuntimeError(f"development job failed with exit code {completed.returncode}: {job_id}")
            candidates = _candidate_run_directories(job)
            if not candidates:
                raise FileNotFoundError(f"development job produced no discoverable run directory: {job_id}")
            register_result(
                root=root_path,
                registry_path=registry_path,
                job_id=job_id,
                run_dir=candidates[0],
            )
        completed_now.append(job_id)
    final_ledger = _read_json(ledger_path)
    remaining = sorted(set(_registry_jobs(registry)) - set(dict(final_ledger.get("results", {}) or {})))
    final_state = {
        "schema_version": 1,
        "artifact_type": "seq100_development_runner_state",
        "status": "completed" if not remaining else "paused_at_job_boundary",
        "registry_sha256": str(registry["registry_sha256"]),
        "completed_now": completed_now,
        "registered_job_count": len(_registry_jobs(registry)) - len(remaining),
        "total_job_count": len(_registry_jobs(registry)),
        "remaining_job_ids": remaining,
        "updated_at": _now(),
    }
    _write_json(state_path, final_state, immutable=False)
    return {**final_state, "runner_state_path": str(state_path.resolve())}


def create_confirmation_registry(
    *,
    root: str | Path,
    candidate_registry_path: str | Path,
    ledger_path: str | Path,
    confirm_seeds: Sequence[int] = CONFIRM_SEEDS,
    confirm_epochs: int = 10,
    shortlist_size: int = 2,
    python_executable: str | Path = DEFAULT_PYTHON,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    candidate = _validated_registry(_read_json(candidate_registry_path))
    root_path = _validated_study_root(candidate, root_path)
    _validated_ledger_path(candidate, ledger_path)
    code_provenance = _validate_code_provenance(dict(candidate.get("code_provenance", {}) or {}))
    selection = select_profiles(registry_path=candidate_registry_path, ledger_path=ledger_path)
    eligible = [
        str(item["profile"])
        for item in list(selection["ranking"])
        if bool(dict(item["guardrails"])["passed"]) and str(item["profile"]) != "baseline"
    ]
    if int(shortlist_size) < 1:
        raise ValueError("shortlist_size must be at least 1")
    shortlist = ["baseline", *eligible[: max(int(shortlist_size) - 1, 0)]]
    seeds = _parse_ints(confirm_seeds)
    if "hard_st_global_tail" in shortlist and int(confirm_epochs) < 2:
        raise ValueError("confirm_epochs must be at least 2 for prior-epoch global-tail hard negatives")
    if int(confirm_epochs) <= 0:
        raise ValueError("confirm_epochs must be positive")
    store_root = _workspace_path(str(candidate["store_root"])).resolve()
    source_path, _, source_sha256 = _validate_source_view(
        str(candidate["source_view"]),
        require_corrected_contract=(
            str(candidate.get("evidence_policy", "")) == EVIDENCE_POLICY_RUN_ARTIFACTS
        ),
    )
    if source_sha256 != str(candidate["source_view_sha256"]):
        raise ValueError("candidate source view changed before confirmation")
    fold_bindings = _fold_bindings(
        years=INNER_OOS_YEARS,
        store_root=store_root,
        source_view_sha256=source_sha256,
        train_start_year=int(candidate["train_start_year"]),
        require_existing=(str(candidate.get("evidence_policy", "")) == EVIDENCE_POLICY_RUN_ARTIFACTS),
    )
    if fold_bindings != dict(candidate.get("fold_bindings", {}) or {}):
        raise ValueError("candidate fold bindings changed before confirmation")
    jobs = _build_jobs(
        phase="confirmation",
        profiles=shortlist,
        years=INNER_OOS_YEARS,
        seeds=seeds,
        epochs=int(confirm_epochs),
        max_samples_per_split=0,
        root=root_path,
        fold_bindings=fold_bindings,
        python_executable=_workspace_path(python_executable),
    )
    body = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "artifact_type": "seq100_confirmation_registry",
        "contract_id": CONTRACT_ID,
        "status": "registered",
        "study_root": str(root_path.resolve()),
        "parent_candidate_registry": str(_workspace_path(candidate_registry_path).resolve()),
        "parent_registry_sha256": str(candidate["registry_sha256"]),
        "source_view": str(source_path),
        "source_view_sha256": source_sha256,
        "store_root": str(store_root),
        "fold_bindings": fold_bindings,
        "train_start_year": int(candidate["train_start_year"]),
        "evidence_policy": str(candidate.get("evidence_policy", EVIDENCE_POLICY_RUN_ARTIFACTS)),
        "code_provenance": code_provenance,
        "screen_selection": selection,
        "shortlist": shortlist,
        "inner_oos_years": list(INNER_OOS_YEARS),
        "phase": {"name": "confirmation", "seeds": list(seeds), "epochs": int(confirm_epochs)},
        "selection_policy": SELECTION_POLICY,
        "jobs": jobs,
    }
    registry = {**body, "registry_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "confirmation_registry.json", registry, immutable=True)
    return {**registry, "registry_path": str(path.resolve())}


def freeze_champion(
    *,
    root: str | Path,
    confirmation_registry_path: str | Path,
    ledger_path: str | Path,
    champion: str,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    registry = _validated_registry(_read_json(confirmation_registry_path))
    root_path = _validated_study_root(registry, root_path)
    _validated_ledger_path(registry, ledger_path)
    code_provenance = _validate_code_provenance(dict(registry.get("code_provenance", {}) or {}))
    registry_policy = str(registry.get("evidence_policy", EVIDENCE_POLICY_RUN_ARTIFACTS))
    source_path, _, source_sha256 = _validate_source_view(
        str(registry["source_view"]),
        require_corrected_contract=(registry_policy == EVIDENCE_POLICY_RUN_ARTIFACTS),
    )
    if source_sha256 != str(registry["source_view_sha256"]):
        raise ValueError("confirmation source view changed before champion freeze")
    current_inner_bindings = _fold_bindings(
        years=INNER_OOS_YEARS,
        store_root=_workspace_path(str(registry["store_root"])).resolve(),
        source_view_sha256=source_sha256,
        train_start_year=int(registry["train_start_year"]),
        require_existing=(registry_policy == EVIDENCE_POLICY_RUN_ARTIFACTS),
    )
    if current_inner_bindings != dict(registry.get("fold_bindings", {}) or {}):
        raise ValueError("confirmation fold bindings changed before champion freeze")
    ledger = _read_json(ledger_path)
    records = _completed_results(registry, ledger)
    selection = select_profiles(registry_path=confirmation_registry_path, ledger_path=ledger_path)
    selected = str(selection["winner"])
    if str(champion) != selected:
        raise ValueError(f"explicit champion {champion!r} is not the policy winner {selected!r}")
    body = {
        "schema_version": 1,
        "artifact_type": "seq100_frozen_champion",
        "contract_id": CONTRACT_ID,
        "status": "frozen",
        "study_root": str(root_path.resolve()),
        "champion": selected,
        "profile_command": PROFILE_COMMANDS[selected],
        "confirmation_registry": str(_workspace_path(confirmation_registry_path).resolve()),
        "confirmation_registry_sha256": str(registry["registry_sha256"]),
        "source_view": str(source_path),
        "source_view_sha256": str(registry["source_view_sha256"]),
        "store_root": str(registry["store_root"]),
        "fold_bindings": dict(registry.get("fold_bindings", {}) or {}),
        "train_start_year": int(registry["train_start_year"]),
        "evidence_policy": str(registry.get("evidence_policy", EVIDENCE_POLICY_RUN_ARTIFACTS)),
        "code_provenance": code_provenance,
        "confirmation_result_sha256s": {
            str(item["job_id"]): str(item["result_sha256"])
            for item in sorted(records, key=lambda value: str(value["job_id"]))
        },
        "selection": selection,
        "inner_oos_years_consumed": list(INNER_OOS_YEARS),
        "historical_outer_oos_years_unconsumed": list(OUTER_OOS_YEARS),
        "outer_evidence_grade": "selection_aware_historical_oos",
        "active_execution_changed": False,
        "qdp_active_changed": False,
    }
    frozen = {**body, "freeze_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "frozen_champion.json", frozen, immutable=True)
    return {**frozen, "freeze_path": str(path.resolve())}


def begin_outer_audit(
    *,
    root: str | Path,
    freeze_path: str | Path,
    source_view: str | Path | None = None,
    store_root: str | Path | None = None,
    epochs: int = 10,
    seed: int = 7,
    train_start_year: int = 2012,
    python_executable: str | Path = DEFAULT_PYTHON,
    require_corrected_source: bool = True,
    require_existing_folds: bool = True,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    frozen = _read_json(freeze_path)
    root_path = _validated_study_root(frozen, root_path)
    if str(frozen.get("status", "")) != "frozen" or not str(frozen.get("freeze_sha256", "")):
        raise ValueError("outer audit requires an explicit immutable frozen champion")
    declared_freeze_sha256 = str(frozen["freeze_sha256"])
    freeze_body = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if _payload_sha256(freeze_body) != declared_freeze_sha256:
        raise ValueError("frozen champion digest mismatch")
    code_provenance = _validate_code_provenance(dict(frozen.get("code_provenance", {}) or {}))
    frozen_source = _workspace_path(str(frozen.get("source_view", "") or "")).resolve()
    requested_source = frozen_source if source_view is None else _workspace_path(source_view).resolve()
    if requested_source != frozen_source:
        raise ValueError("outer audit source view must match the frozen inner source")
    source_path, _, source_sha256 = _validate_source_view(
        requested_source,
        require_corrected_contract=bool(require_corrected_source),
    )
    if source_sha256 != str(frozen.get("source_view_sha256", "")):
        raise ValueError("outer audit source SHA-256 must match the frozen inner source")
    frozen_policy = str(frozen.get("evidence_policy", EVIDENCE_POLICY_RUN_ARTIFACTS))
    if frozen_policy == EVIDENCE_POLICY_RUN_ARTIFACTS and (
        not bool(require_corrected_source) or not bool(require_existing_folds)
    ):
        raise ValueError("run-artifact outer audits require corrected-source and existing-fold verification")
    frozen_store = _workspace_path(str(frozen.get("store_root", "") or "")).resolve()
    requested_store = frozen_store if store_root is None else _workspace_path(store_root).resolve()
    if requested_store != frozen_store:
        raise ValueError("outer audit store root must match the frozen inner store")
    if int(train_start_year) != int(frozen.get("train_start_year", -1)):
        raise ValueError("outer audit train_start_year must match the frozen inner contract")
    current_inner_bindings = _fold_bindings(
        years=INNER_OOS_YEARS,
        store_root=requested_store,
        source_view_sha256=source_sha256,
        train_start_year=int(train_start_year),
        require_existing=bool(require_existing_folds),
    )
    if current_inner_bindings != dict(frozen.get("fold_bindings", {}) or {}):
        raise ValueError("frozen inner fold bindings changed before outer audit")
    outer_fold_bindings = _fold_bindings(
        years=OUTER_OOS_YEARS,
        store_root=requested_store,
        source_view_sha256=source_sha256,
        train_start_year=int(train_start_year),
        require_existing=bool(require_existing_folds),
    )
    claim_path = root_path / "outer_audit_consumption.json"
    if claim_path.exists():
        raise RuntimeError("historical outer audit has already been consumed or claimed")
    profile = str(frozen["champion"])
    if int(epochs) <= 0:
        raise ValueError("outer audit epochs must be positive")
    if profile == "hard_st_global_tail" and int(epochs) < 2:
        raise ValueError("outer audit epochs must be at least 2 for prior-epoch global-tail hard negatives")
    jobs = _build_jobs(
        phase="outer_audit",
        profiles=(profile,),
        years=OUTER_OOS_YEARS,
        seeds=(int(seed),),
        epochs=int(epochs),
        max_samples_per_split=0,
        root=root_path,
        fold_bindings=outer_fold_bindings,
        python_executable=_workspace_path(python_executable),
    )
    body = {
        "schema_version": 1,
        "artifact_type": "seq100_outer_audit_registry",
        "contract_id": CONTRACT_ID,
        "status": "claimed",
        "study_root": str(root_path.resolve()),
        "freeze_path": str(_workspace_path(freeze_path).resolve()),
        "freeze_sha256": declared_freeze_sha256,
        "champion": profile,
        "source_view": str(source_path),
        "source_view_sha256": source_sha256,
        "store_root": str(requested_store),
        "fold_bindings": outer_fold_bindings,
        "train_start_year": int(train_start_year),
        "evidence_policy": str(frozen.get("evidence_policy", EVIDENCE_POLICY_RUN_ARTIFACTS)),
        "code_provenance": code_provenance,
        "historical_outer_oos_years": list(OUTER_OOS_YEARS),
        "evidence_grade": "selection_aware_historical_oos",
        "selection_aware": True,
        "repeat_allowed": False,
        "fold_builds": _fold_build_commands(
            years=OUTER_OOS_YEARS,
            source_view=source_path,
            store_root=requested_store,
            train_start_year=int(train_start_year),
            python_executable=_workspace_path(python_executable),
        ),
        "jobs": jobs,
        "claimed_at": _now(),
    }
    claim = {**body, "registry_sha256": _payload_sha256(body)}
    path = _write_json_exclusive(claim_path, claim)
    return {**claim, "registry_path": str(path.resolve())}


def finalize_outer_audit(
    *,
    root: str | Path,
    ledger_path: str | Path,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    claim_path = root_path / "outer_audit_consumption.json"
    claim = _validated_registry(_read_json(claim_path))
    root_path = _validated_study_root(claim, root_path)
    ledger_path = _validated_ledger_path(claim, ledger_path)
    _validate_code_provenance(dict(claim.get("code_provenance", {}) or {}))
    freeze_path = _workspace_path(str(claim.get("freeze_path", "") or "")).resolve()
    frozen = _read_json(freeze_path)
    declared_freeze_sha = str(frozen.get("freeze_sha256", ""))
    freeze_body = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        not declared_freeze_sha
        or _payload_sha256(freeze_body) != declared_freeze_sha
        or declared_freeze_sha != str(claim.get("freeze_sha256", ""))
    ):
        raise ValueError("outer audit freeze binding changed before finalization")
    evidence_policy = str(claim.get("evidence_policy", EVIDENCE_POLICY_RUN_ARTIFACTS))
    source_path, _, source_sha256 = _validate_source_view(
        str(claim["source_view"]),
        require_corrected_contract=(evidence_policy == EVIDENCE_POLICY_RUN_ARTIFACTS),
    )
    if source_sha256 != str(claim["source_view_sha256"]):
        raise ValueError("outer audit source changed before finalization")
    store_root = _workspace_path(str(claim["store_root"])).resolve()
    train_start_year = int(claim["train_start_year"])
    current_outer_bindings = _fold_bindings(
        years=OUTER_OOS_YEARS,
        store_root=store_root,
        source_view_sha256=source_sha256,
        train_start_year=train_start_year,
        require_existing=(evidence_policy == EVIDENCE_POLICY_RUN_ARTIFACTS),
    )
    if current_outer_bindings != dict(claim.get("fold_bindings", {}) or {}):
        raise ValueError("outer fold bindings changed before finalization")
    current_inner_bindings = _fold_bindings(
        years=INNER_OOS_YEARS,
        store_root=store_root,
        source_view_sha256=source_sha256,
        train_start_year=train_start_year,
        require_existing=(evidence_policy == EVIDENCE_POLICY_RUN_ARTIFACTS),
    )
    if current_inner_bindings != dict(frozen.get("fold_bindings", {}) or {}):
        raise ValueError("frozen inner fold bindings changed before finalization")
    ledger_resolved = _workspace_path(ledger_path).resolve()
    ledger = _read_json(ledger_resolved)
    records = _completed_results(claim, ledger)
    aggregates = _aggregate_records(records)
    champion = str(claim["champion"])
    body = {
        "schema_version": 1,
        "artifact_type": "seq100_outer_audit_result",
        "contract_id": CONTRACT_ID,
        "status": "completed",
        "study_root": str(root_path.resolve()),
        "outer_registry": str(claim_path.resolve()),
        "outer_registry_sha256": str(claim["registry_sha256"]),
        "freeze_sha256": str(claim["freeze_sha256"]),
        "source_view": str(source_path),
        "source_view_sha256": source_sha256,
        "store_root": str(store_root),
        "fold_bindings": current_outer_bindings,
        "ledger_path": str(ledger_resolved),
        "ledger_sha256": _file_sha256(ledger_resolved),
        "outer_result_sha256s": {
            str(item["job_id"]): str(item["result_sha256"])
            for item in sorted(records, key=lambda value: str(value["job_id"]))
        },
        "champion": champion,
        "historical_outer_oos_years": list(OUTER_OOS_YEARS),
        "evidence_grade": "selection_aware_historical_oos",
        "selection_aware": True,
        "repeat_allowed": False,
        "aggregate": aggregates[champion],
        "completed_at": _now(),
        "active_execution_changed": False,
        "qdp_active_changed": False,
    }
    result = {**body, "result_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "outer_audit_result.json", result, immutable=True)
    return {**result, "result_path": str(path.resolve())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selection-safe seq100 research-generation orchestration.")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    init.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    init.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    init.add_argument("--screen-seeds", default=",".join(map(str, SCREEN_SEEDS)))
    init.add_argument("--screen-epochs", type=int, default=2)
    init.add_argument(
        "--screen-max-samples-per-split",
        type=int,
        default=0,
        help="Training-only complete-date cap for the screen; inner OOS metrics remain full-universe.",
    )

    register = sub.add_parser("register-result")
    register.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    register.add_argument("--registry", type=Path, required=True)
    register.add_argument("--job-id", required=True)
    register.add_argument("--run-dir", type=Path, required=True)

    advance = sub.add_parser("advance-confirmation")
    advance.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    advance.add_argument("--candidate-registry", type=Path, required=True)
    advance.add_argument("--ledger", type=Path, required=True)
    advance.add_argument("--confirm-seeds", default=",".join(map(str, CONFIRM_SEEDS)))
    advance.add_argument("--confirm-epochs", type=int, default=10)
    advance.add_argument("--shortlist-size", type=int, default=2)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    freeze.add_argument("--confirmation-registry", type=Path, required=True)
    freeze.add_argument("--ledger", type=Path, required=True)
    freeze.add_argument("--champion", choices=PROFILE_ORDER, required=True)

    outer = sub.add_parser("begin-outer-audit")
    outer.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    outer.add_argument("--freeze", type=Path, required=True)
    outer.add_argument("--source-view", type=Path)
    outer.add_argument("--store-root", type=Path)
    outer.add_argument("--epochs", type=int, default=10)
    outer.add_argument("--seed", type=int, default=7)

    finalize = sub.add_parser("finalize-outer-audit")
    finalize.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    finalize.add_argument("--ledger", type=Path, required=True)

    development = sub.add_parser(
        "init-development",
        help="Register the candidate-complete 2022-2025 development walkforward matrix.",
    )
    development.add_argument("--root", type=Path, default=DEFAULT_DEVELOPMENT_ROOT)
    development.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    development.add_argument("--store-root", type=Path, default=DEFAULT_DEVELOPMENT_STORE_ROOT)
    development.add_argument("--profiles", default=",".join(DEFAULT_DEVELOPMENT_PROFILES))
    development.add_argument("--maximum-epochs", type=int, default=10)
    development.add_argument("--patience", type=int, default=2)
    development.add_argument("--min-delta", type=float, default=0.0)
    development.add_argument("--minimum-complete-epochs", type=int, default=1)
    development.add_argument("--kpi-portfolio-contract", type=Path)
    development.add_argument("--additional-provenance-path", type=Path, action="append", default=[])

    run_development = sub.add_parser("run-development")
    run_development.add_argument("--root", type=Path, default=DEFAULT_DEVELOPMENT_ROOT)
    run_development.add_argument("--registry", type=Path, required=True)
    run_development.add_argument("--max-jobs", type=int, default=0)

    select_development = sub.add_parser("select-development")
    select_development.add_argument("--registry", type=Path, required=True)
    select_development.add_argument("--ledger", type=Path, required=True)

    freeze_development = sub.add_parser("freeze-development")
    freeze_development.add_argument("--root", type=Path, default=DEFAULT_DEVELOPMENT_ROOT)
    freeze_development.add_argument("--registry", type=Path, required=True)
    freeze_development.add_argument("--ledger", type=Path, required=True)
    freeze_development.add_argument("--champion", choices=PROFILE_ORDER, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        result = initialize_candidate_registry(
            root=args.root,
            source_view=args.source_view,
            store_root=args.store_root,
            screen_seeds=_parse_ints(args.screen_seeds),
            screen_epochs=args.screen_epochs,
            screen_max_samples_per_split=args.screen_max_samples_per_split,
        )
    elif args.command == "register-result":
        result = register_result(
            root=args.root,
            registry_path=args.registry,
            job_id=args.job_id,
            run_dir=args.run_dir,
        )
    elif args.command == "advance-confirmation":
        result = create_confirmation_registry(
            root=args.root,
            candidate_registry_path=args.candidate_registry,
            ledger_path=args.ledger,
            confirm_seeds=_parse_ints(args.confirm_seeds),
            confirm_epochs=args.confirm_epochs,
            shortlist_size=args.shortlist_size,
        )
    elif args.command == "freeze":
        result = freeze_champion(
            root=args.root,
            confirmation_registry_path=args.confirmation_registry,
            ledger_path=args.ledger,
            champion=args.champion,
        )
    elif args.command == "begin-outer-audit":
        result = begin_outer_audit(
            root=args.root,
            freeze_path=args.freeze,
            source_view=args.source_view,
            store_root=args.store_root,
            epochs=args.epochs,
            seed=args.seed,
        )
    elif args.command == "finalize-outer-audit":
        result = finalize_outer_audit(root=args.root, ledger_path=args.ledger)
    elif args.command == "init-development":
        result = initialize_development_registry(
            root=args.root,
            source_view=args.source_view,
            store_root=args.store_root,
            profiles=tuple(item.strip() for item in str(args.profiles).split(",") if item.strip()),
            maximum_epochs=int(args.maximum_epochs),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            minimum_complete_epochs=int(args.minimum_complete_epochs),
            kpi_portfolio_contract_path=args.kpi_portfolio_contract,
            additional_provenance_paths=tuple(args.additional_provenance_path),
        )
    elif args.command == "run-development":
        result = run_development_registry(
            root=args.root,
            registry_path=args.registry,
            max_jobs=int(args.max_jobs),
        )
    elif args.command == "select-development":
        result = select_development_profiles(
            registry_path=args.registry,
            ledger_path=args.ledger,
        )
    else:
        result = freeze_development_champion(
            root=args.root,
            registry_path=args.registry,
            ledger_path=args.ledger,
            champion=args.champion,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
