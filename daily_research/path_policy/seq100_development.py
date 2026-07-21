"""Single-current-study seq100 rolling development workflow.

The current workflow intentionally has one mutable study, one daily-only pack,
three rolling development folds, and no registry/freeze/promotion layer.
Historical orchestration remains available from ``seq100_research_generation``
but is not imported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from daily_research.path_policy import qdp_v2_sequence_path_pack as pack_builder
from daily_research.path_policy import seq100_walkforward as walkforward
from daily_research.path_policy.seq100_exit_policy_audit import CandidateCompleteAuditPack
from daily_research.path_policy.seq100_exit_policy_price_basis_diagnostic import (
    _basis_summaries,
    _candidate_bridge_rows,
)
from daily_research.path_policy.seq100_mainline import (
    PROFILE_SPECS,
    build_todayclose_path_only_train_argv,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
QDP_ROOT = Path("quant_data_platform/data/qdp_v2")
STUDY_ROOT = Path(
    "daily_research/output/path_policy/studies/seq100_corrected_rolling_2023_2025_v1"
)
STORE_ROOT = Path("daily_research/data/research_store/seq100_current")
PROFILE_COMMAND = "train-daily-only-summary-v2-ohlcva-aux-low"
PROFILE_NAME = "baseline"
DEVELOPMENT_YEARS = (2023, 2024, 2025)
SEED = 7
LOOKBACK_DAYS = 100
FORWARD_DAYS = 60
EXECUTION_TAIL_DAYS = 20
SIGNAL_START = "2010-01-04"
SIGNAL_END = "2025-12-31"
EXPECTED_QDP_AS_OF = "2026-07-16"
MEMORY_GUARD_GIB = 0.5
MEMORY_INTERVAL_SECONDS = 1.0
MEMORY_CONSECUTIVE_BREACHES = 2
OPERATIONAL_COMMANDS = ("contract", "prepare", "run", "status", "summarize")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _workspace_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw.resolve() if raw.is_absolute() else (WORKSPACE_ROOT / raw).resolve()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _contract_semantics() -> dict[str, Any]:
    profile = PROFILE_SPECS[PROFILE_COMMAND].default_profile()
    profile_parameters = asdict(profile)
    for transient in ("store_view", "output_root", "run_tag"):
        profile_parameters.pop(transient, None)
    return {
        "schema_version": 1,
        "contract_id": "seq100_corrected_rolling_2023_2025_v1",
        "data": {
            "qdp_expected_as_of": EXPECTED_QDP_AS_OF,
            "signal_start": SIGNAL_START,
            "signal_end": SIGNAL_END,
            "lookback_days": LOOKBACK_DAYS,
            "prediction_forward_days": FORWARD_DAYS,
            "execution_tail_days": EXECUTION_TAIL_DAYS,
            "max_label_dependency_days": FORWARD_DAYS + EXECUTION_TAIL_DAYS,
            "input_channels": ["daily_raw", "daily_state"],
            "candidate_source": ["universe_snapshot", "security_status"],
            "price_semantics": "back-adjusted model paths; raw exchange execution",
        },
        "development_protocol": {
            "years": list(DEVELOPMENT_YEARS),
            "method": "purged_expanding_development_walkforward",
            "split_roles": {"fit": "train", "evaluation": "development"},
            "train_start_year": 2010,
            "purge_rule": "max_label_dependency_date_idx < development_start_date_idx",
            "seed": SEED,
            "historical_test_set": None,
            "final_fit_in_scope": False,
        },
        "early_stopping": {
            "metric": "development_total_loss",
            "mode": "min",
            "minimum_complete_epochs": 1,
            "maximum_epochs": 10,
            "patience": 2,
            "restore_best_checkpoint": True,
        },
        "profile": {
            "command": PROFILE_COMMAND,
            "name": PROFILE_NAME,
            "input_dim": 32,
            "batch_size": 512,
            "parameters": profile_parameters,
        },
        "metric_contract": {
            "primary": "equal_year_mean_top3_net_realized_plan_return_base_alpha",
            "year_weighting": "equal",
            "top_k": [1, 3, 5, 10],
            "daily_cohort_cash_cny": 1_000_000.0,
            "unfilled_entry": "retain_cash_without_rank_replacement",
            "winner_or_promotion": False,
        },
        "protected_boundaries": {
            "update_qdp": False,
            "change_active_execution": False,
            "create_final_model": False,
        },
    }


def current_contract() -> dict[str, Any]:
    contract = _contract_semantics()
    return {
        "schema_version": 1,
        "workflow_id": "seq100_corrected_rolling_development",
        "operational_commands": list(OPERATIONAL_COMMANDS),
        "contract": contract,
        "contract_sha256": _canonical_digest(contract),
        "source_pack_required_for_contract": False,
        "historical_orchestrator_is_current": False,
        "memory_guard": {
            "minimum_available_gib": MEMORY_GUARD_GIB,
            "interval_seconds": MEMORY_INTERVAL_SECONDS,
            "consecutive_breaches": MEMORY_CONSECUTIVE_BREACHES,
        },
    }


def _read_active(qdp_root: Path) -> dict[str, Any]:
    path = qdp_root / "active" / "active.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("active_as_of_date", "")) != EXPECTED_QDP_AS_OF:
        raise ValueError(
            f"current study is frozen to QDP {EXPECTED_QDP_AS_OF}; observed "
            f"{payload.get('active_as_of_date', '')}"
        )
    return payload


def _dataset_manifest(qdp_root: Path, active: Mapping[str, Any], domain: str) -> dict[str, Any]:
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, "") or "")
    if not dataset_id:
        raise ValueError(f"active QDP is missing {domain}")
    path = qdp_root / "datasets" / domain / dataset_id / "dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _open_dates(qdp_root: Path, active: Mapping[str, Any]) -> list[str]:
    manifest = _dataset_manifest(qdp_root, active, "trading_calendar")
    values: set[str] = set()
    for shard in manifest.get("shards", []):
        table = pq.read_table(
            qdp_root / str(shard["path"]), columns=["trade_date", "is_open"]
        ).to_pydict()
        values.update(
            str(date)
            for date, is_open in zip(table["trade_date"], table["is_open"])
            if bool(is_open)
        )
    return sorted(values)


def _security_identity_listing_dates(
    qdp_root: Path, active: Mapping[str, Any]
) -> dict[str, str]:
    manifest = _dataset_manifest(qdp_root, active, "security_identity")
    listing_dates: dict[str, str] = {}
    for shard in manifest.get("shards", []):
        table = pq.read_table(
            qdp_root / str(shard["path"]), columns=["current_symbol", "list_date"]
        ).to_pydict()
        for symbol, list_date in zip(table["current_symbol"], table["list_date"]):
            normalized_symbol = str(symbol or "")
            normalized_date = str(list_date or "")
            if not normalized_symbol or len(normalized_date) != 10:
                raise ValueError("security_identity contains an invalid symbol/list_date")
            if normalized_symbol in listing_dates:
                raise ValueError(f"security_identity contains duplicate symbol {normalized_symbol}")
            listing_dates[normalized_symbol] = normalized_date
    return listing_dates


def _latest_full_audit(qdp_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(
        (qdp_root / "audits").glob("database_audit_*.json"),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("mode", "")) == "full":
            if str(payload.get("status", "")) != "ok" or int(payload.get("finding_count", -1)) != 0:
                raise ValueError(f"latest full QDP audit is not clean: {path}")
            return path, payload
    raise FileNotFoundError("no full QDP database audit is available")


def _preflight_audit_metrics(audit: Mapping[str, Any]) -> dict[str, Any]:
    checks = dict(audit.get("cross_dataset_checks", {}) or {})
    factor = dict(checks.get("factor_semantics", {}) or {})
    exclusions = dict(checks.get("permanent_exclusions", {}) or {})
    residuals = dict(exclusions.get("residuals", {}) or {})
    continuity = dict(checks.get("long_suspension_continuity", {}) or {})
    required = {
        "factor_status": factor.get("status"),
        "factor_excessive_change_symbol_year_count": factor.get(
            "excessive_change_symbol_year_count"
        ),
        "factor_uncompensated_change_count": factor.get("uncompensated_change_count"),
        "regression_600076_2024_change_count": factor.get(
            "regression_600076_2024_change_count"
        ),
        "permanent_exclusion_residual_rows": residuals.get("total_residual_rows"),
        "continuity_status": continuity.get("status"),
        "continuity_break_count": continuity.get("continuity_break_count"),
    }
    if (
        required["factor_status"] != "ok"
        or int(required["factor_excessive_change_symbol_year_count"] or 0) != 0
        or int(required["factor_uncompensated_change_count"] or 0) != 0
        or int(required["regression_600076_2024_change_count"] or 0) != 0
        or int(required["permanent_exclusion_residual_rows"] or 0) != 0
        or required["continuity_status"] != "ok"
    ):
        raise ValueError(f"QDP semantic preflight failed: {required}")
    return required


def prepare_dry_run(
    *, qdp_root: str | Path = QDP_ROOT, store_root: str | Path = STORE_ROOT
) -> dict[str, Any]:
    qdp = _workspace_path(qdp_root)
    store = _workspace_path(store_root)
    active = _read_active(qdp)
    audit_path, audit = _latest_full_audit(qdp)
    audit_metrics = _preflight_audit_metrics(audit)
    dates = _open_dates(qdp, active)
    sample_start_idx = dates.index(SIGNAL_START)
    sample_end_idx = max(idx for idx, date in enumerate(dates) if date <= SIGNAL_END)
    panel_end_idx = sample_end_idx + FORWARD_DAYS + EXECUTION_TAIL_DAYS
    if panel_end_idx >= len(dates):
        raise ValueError("QDP does not contain the required 80-day dependency padding")
    panel_dates = dates[sample_start_idx : panel_end_idx + 1]
    qdp_symbol_count = int(dict(active.get("scope", {}) or {}).get("symbol_count", 0) or 0)
    if qdp_symbol_count != 3041:
        raise ValueError(f"current study requires 3,041 QDP identities; observed {qdp_symbol_count}")
    identity_dates = _security_identity_listing_dates(qdp, active)
    if len(identity_dates) != qdp_symbol_count:
        raise ValueError(
            "security_identity count does not match active scope: "
            f"{len(identity_dates)} != {qdp_symbol_count}"
        )
    dependency_padding_end = dates[panel_end_idx]
    eligible_pack_symbols = {
        symbol for symbol, list_date in identity_dates.items() if list_date <= dependency_padding_end
    }
    post_dependency_listings = sorted(
        symbol for symbol, list_date in identity_dates.items() if list_date > dependency_padding_end
    )
    symbol_count = len(eligible_pack_symbols)
    n_dates = len(panel_dates)
    # Daily panels, scalar execution arrays/masks, a 60x6 sharded label store,
    # and path summaries.  Parquet indexes are estimated separately.
    estimated_array_bytes = int(
        n_dates * symbol_count * (
            (len(pack_builder.DAILY_RAW_FEATURES) + len(pack_builder.RAW_SIGNAL_COLUMNS)) * 4
            + 4 * 4
            + 16
            + FORWARD_DAYS * 6 * 4
            + 20 * 4
        )
    )
    old_store_bytes = 0
    research_store = store.parent
    if research_store.exists():
        old_store_bytes = sum(
            path.stat().st_size for path in research_store.rglob("*") if path.is_file()
        )
    safe_ends: dict[str, str] = {}
    for year in DEVELOPMENT_YEARS:
        start_idx = next(idx for idx, date in enumerate(dates) if date >= f"{year}-01-01")
        safe_ends[str(year)] = dates[start_idx - (FORWARD_DAYS + EXECUTION_TAIL_DAYS) - 1]
    return {
        "status": "dry_run",
        "writes_performed": False,
        "qdp_root": str(qdp),
        "qdp_active_as_of": active["active_as_of_date"],
        "qdp_full_audit": str(audit_path.resolve()),
        "qdp_preflight": audit_metrics,
        "signal_window": [SIGNAL_START, SIGNAL_END],
        "first_valid_lookback_signal": dates[sample_start_idx + LOOKBACK_DAYS - 1],
        "dependency_padding_end": dependency_padding_end,
        "panel_date_count": n_dates,
        "symbol_count": symbol_count,
        "qdp_symbol_count": qdp_symbol_count,
        "post_dependency_listing_count": len(post_dependency_listings),
        "post_dependency_listing_symbols": post_dependency_listings,
        "development_years": list(DEVELOPMENT_YEARS),
        "safe_train_signal_end": safe_ends,
        "estimated_array_bytes": estimated_array_bytes,
        "existing_research_store_bytes": int(old_store_bytes),
        "feature_profile": "daily_only",
        "minute_data_read": False,
    }


def _study_path(study_root: str | Path) -> Path:
    return _workspace_path(study_root) / "study.json"


def _new_study_payload(dry_run: Mapping[str, Any]) -> dict[str, Any]:
    contract = _contract_semantics()
    return {
        "schema_version": 1,
        "artifact_type": "seq100_current_study",
        "study_id": "seq100_corrected_rolling_2023_2025_v1",
        "contract": contract,
        "contract_sha256": _canonical_digest(contract),
        "runtime": {
            "status": "preparing",
            "updated_at": _now(),
            "completed_years": [],
        },
        "preflight": dict(dry_run),
        "material": {},
    }


def _load_study(study_root: str | Path = STUDY_ROOT) -> tuple[Path, dict[str, Any]]:
    path = _study_path(study_root)
    if not path.is_file():
        raise FileNotFoundError(f"study is not prepared: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("artifact_type", "")) != "seq100_current_study":
        raise ValueError(f"not a current seq100 study: {path}")
    contract = dict(payload.get("contract", {}) or {})
    if str(payload.get("contract_sha256", "")) != _canonical_digest(contract):
        raise ValueError("study contract digest does not match its semantic contract")
    return path, payload


def _update_runtime(study_path: Path, study: dict[str, Any], **changes: Any) -> None:
    runtime = dict(study.get("runtime", {}) or {})
    runtime.update(changes)
    runtime["updated_at"] = _now()
    study["runtime"] = runtime
    _write_json(study_path, study)


def _assert_no_other_research_process() -> None:
    current = os.getpid()
    excluded = {current}
    try:
        parent = psutil.Process(current).parent()
        while parent is not None:
            excluded.add(int(parent.pid))
            parent = parent.parent()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    markers = (
        "qdp_v2_sequence_path_training",
        "seq100_development run",
        "qdp_v2_sequence_path_pack build",
    )
    matches: list[int] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            if int(process.info["pid"]) in excluded:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            if any(marker in command for marker in markers):
                matches.append(int(process.info["pid"]))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if matches:
        raise RuntimeError(f"another research build/train process is active: pids={matches}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _cleanup_research_store(store_root: Path) -> None:
    research_store = store_root.parent.resolve()
    expected = (WORKSPACE_ROOT / "daily_research" / "data" / "research_store").resolve()
    if research_store != expected:
        raise ValueError(f"refusing to clean a non-canonical research store: {research_store}")
    research_store.mkdir(parents=True, exist_ok=True)
    for child in research_store.iterdir():
        if child.name == ".gitkeep":
            continue
        _remove_path(child)


def _guarded_command(command: Sequence[str], *, log_path: Path) -> None:
    guard = [
        str(PYTHON),
        str((WORKSPACE_ROOT / "tools" / "memory_guard.py").resolve()),
        "--min-available-gb",
        str(MEMORY_GUARD_GIB),
        "--interval-seconds",
        str(MEMORY_INTERVAL_SECONDS),
        "--consecutive-breaches",
        str(MEMORY_CONSECUTIVE_BREACHES),
        "--log-json",
        str(log_path.resolve()),
        "--",
        *[str(item) for item in command],
    ]
    completed = subprocess.run(guard, cwd=WORKSPACE_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"guarded command failed with exit code {completed.returncode}")


def _replace_path_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_path_prefix(item, old, new) for key, item in value.items()}
    return value


def _relocate_pack_manifest(manifest_path: Path, old_root: Path, new_root: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relocated = _replace_path_prefix(payload, str(old_root.resolve()), str(new_root.resolve()))
    target = new_root / "pack" / "manifest.json"
    _write_json(target, relocated)
    for auxiliary in ("progress.json", "memory_observations.json"):
        path = new_root / "pack" / auxiliary
        if path.is_file():
            content = json.loads(path.read_text(encoding="utf-8"))
            _write_json(path, _replace_path_prefix(content, str(old_root.resolve()), str(new_root.resolve())))
    return relocated


def _pack_preflight(
    *, manifest_path: Path, fold_results: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = pack_builder.validate_sequence_pack(manifest_path)
    if str(validation.get("status", "")) != "ok":
        raise ValueError(f"sequence pack validation failed: {validation.get('blockers', [])}")
    channels = set(dict(manifest.get("feature_channels", {}) or {}))
    if channels != {"daily_raw", "daily_state"}:
        raise ValueError(f"daily-only pack contains unexpected feature channels: {sorted(channels)}")
    input_masks = list(
        dict(manifest.get("data_semantics", {}) or {}).get("model_input_mask_features", [])
        or []
    )
    model_input_dim = sum(
        len(list(dict(meta).get("columns", []) or []))
        for meta in dict(manifest.get("feature_channels", {}) or {}).values()
    ) + len(input_masks)
    if model_input_dim != 32:
        raise ValueError(f"daily-only model input dimension is not 32: {model_input_dim}")
    qdp_root = Path(str(manifest.get("qdp_root", "") or ""))
    active = {"datasets": dict(manifest.get("active_datasets", {}) or {})}
    identity_dates = _security_identity_listing_dates(qdp_root, active)
    date_values = [str(item) for item in list(manifest.get("date_values", []) or [])]
    if not date_values:
        raise ValueError("pack has no date_values")
    panel_end = date_values[-1]
    expected_symbols = {
        symbol for symbol, list_date in identity_dates.items() if list_date <= panel_end
    }
    pack_symbols = {str(item) for item in list(manifest.get("symbol_values", []) or [])}
    unexpected_missing = sorted(expected_symbols.difference(pack_symbols))
    unexpected_extra = sorted(pack_symbols.difference(expected_symbols))
    if unexpected_missing or unexpected_extra:
        raise ValueError(
            "pack symbol scope differs from identities listed by the dependency cutoff: "
            f"missing={unexpected_missing[:10]}, extra={unexpected_extra[:10]}"
        )
    post_dependency_listings = sorted(
        symbol for symbol, list_date in identity_dates.items() if list_date > panel_end
    )
    symbol_scope = {
        "qdp_identity_count": len(identity_dates),
        "pack_symbol_count": len(pack_symbols),
        "dependency_padding_end": panel_end,
        "post_dependency_listing_count": len(post_dependency_listings),
        "post_dependency_listing_symbols": post_dependency_listings,
        "unexpected_missing_count": 0,
        "unexpected_extra_count": 0,
    }
    candidates = pd.read_parquet(str(manifest["candidate_index_path"]), columns=["trade_date"])
    signal_2026_count = int(candidates["trade_date"].astype(str).str.startswith("2026").sum())
    if signal_2026_count:
        raise ValueError(f"pack contains {signal_2026_count} forbidden 2026 signal rows")
    fold_audits: dict[str, Any] = {}
    for year, result in fold_results.items():
        verification = walkforward.verify_development_walkforward_view(result["view_path"])
        if str(verification.get("status", "")) != "ok":
            raise ValueError(f"development fold {year} failed verification: {verification}")
        contract = dict(verification.get("development_walkforward", {}) or {})
        if int(contract.get("label_dependency_overlap_count", -1)) != 0:
            raise ValueError(f"development fold {year} has label dependency overlap")
        fold_audits[str(year)] = verification
    return {
        "pack_validation": validation,
        "feature_channels": sorted(channels),
        "model_input_mask_features": input_masks,
        "model_input_dim": model_input_dim,
        "signal_2026_count": signal_2026_count,
        "symbol_scope": symbol_scope,
        "folds": fold_audits,
    }


def prepare_current_study(
    *,
    qdp_root: str | Path = QDP_ROOT,
    study_root: str | Path = STUDY_ROOT,
    store_root: str | Path = STORE_ROOT,
    cleanup: bool = True,
) -> dict[str, Any]:
    _assert_no_other_research_process()
    qdp = _workspace_path(qdp_root)
    study_dir = _workspace_path(study_root)
    store = _workspace_path(store_root)
    dry_run = prepare_dry_run(qdp_root=qdp, store_root=store)
    study_path = study_dir / "study.json"
    if study_path.is_file():
        _, existing = _load_study(study_dir)
        existing_status = str(dict(existing.get("runtime", {}) or {}).get("status", ""))
        existing_manifest = Path(
            str(dict(existing.get("material", {}) or {}).get("pack_manifest", "") or "")
        )
        if existing_status in {
            "prepared",
            "running",
            "runs_completed",
            "completed",
        } and existing_manifest.is_file():
            validation = pack_builder.validate_sequence_pack(existing_manifest)
            if str(validation.get("status", "")) == "ok":
                material = dict(existing.get("material", {}) or {})
                fold_views = dict(material.get("fold_views", {}) or {})
                fold_results = {
                    year: {"view_path": str(fold_views.get(str(year), "") or "")}
                    for year in DEVELOPMENT_YEARS
                }
                if not all(
                    Path(str(result["view_path"])).is_file()
                    for result in fold_results.values()
                ):
                    raise FileNotFoundError("prepared study is missing one or more fold views")
                material["preflight"] = _pack_preflight(
                    manifest_path=existing_manifest,
                    fold_results=fold_results,
                )
                existing["material"] = material
                existing["preflight"] = dict(dry_run)
                completed_years = [
                    year
                    for year in DEVELOPMENT_YEARS
                    if len(_completed_run_dirs(study_dir, year)[0]) == 1
                ]
                partial_exists = any(
                    bool(_completed_run_dirs(study_dir, year)[1])
                    for year in DEVELOPMENT_YEARS
                )
                if completed_years == list(DEVELOPMENT_YEARS):
                    reconciled_status = (
                        "completed"
                        if (study_dir / "research_summary.json").is_file()
                        else "runs_completed"
                    )
                elif completed_years or partial_exists:
                    reconciled_status = "running"
                else:
                    reconciled_status = "prepared"
                _update_runtime(
                    study_path,
                    existing,
                    status=reconciled_status,
                    completed_years=completed_years,
                    current_year=None,
                    error=None,
                )
                return existing
        recoverable_manifest = store / "pack" / "manifest.json"
        recoverable_fold_root = store / "folds"
        recoverable_folds = {
            year: {
                "view_path": str(
                    walkforward.development_view_path(
                        year, store_root=recoverable_fold_root
                    ).resolve()
                )
            }
            for year in DEVELOPMENT_YEARS
        }
        if existing_status == "prepare_failed" and recoverable_manifest.is_file() and all(
            Path(str(result["view_path"])).is_file()
            for result in recoverable_folds.values()
        ):
            material_preflight = _pack_preflight(
                manifest_path=recoverable_manifest,
                fold_results=recoverable_folds,
            )
            existing["preflight"] = dict(dry_run)
            existing["material"] = {
                "store_root": str(store),
                "pack_manifest": str(recoverable_manifest),
                "fold_root": str(recoverable_fold_root),
                "fold_views": {
                    str(year): str(recoverable_folds[year]["view_path"])
                    for year in DEVELOPMENT_YEARS
                },
                "preflight": material_preflight,
            }
            _update_runtime(
                study_path,
                existing,
                status="prepared",
                completed_years=[],
                error=None,
            )
            return json.loads(study_path.read_text(encoding="utf-8"))
    study = _new_study_payload(dry_run)
    _write_json(study_path, study)
    if cleanup:
        _cleanup_research_store(store)
    else:
        _remove_path(store)
    staging = store.parent / f".{store.name}.staging"
    _remove_path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    logs = study_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    build_command = [
        str(PYTHON),
        "-m",
        "daily_research.path_policy.qdp_v2_sequence_path_pack",
        "build",
        "--qdp-root",
        str(qdp),
        "--output-root",
        str(staging),
        "--run-tag",
        "pack",
        "--lookback-days",
        str(LOOKBACK_DAYS),
        "--forward-days",
        str(FORWARD_DAYS),
        "--start-date",
        SIGNAL_START,
        "--end-date",
        SIGNAL_END,
        "--train-years",
        "2010-2022",
        "--validation-years",
        "2023-2025",
        "--test-years",
        "",
        "--no-legacy-ohlc-label",
        "--label-shard-size",
        "64",
        "--price-anchor",
        "today_close",
        "--price-adjustment",
        "back_adjust",
        "--entry-rule",
        "open_below_limit_tick",
        "--sample-filter",
        pack_builder.SAMPLE_FILTER_CURRENT_QDP,
        "--suspension-fill",
        pack_builder.SUSPENSION_FILL_CARRY_CLOSE,
        "--feature-profile",
        pack_builder.FEATURE_PROFILE_DAILY_ONLY,
        "--execution-tail-days",
        str(EXECUTION_TAIL_DAYS),
        "--minimum-free-memory-gb",
        str(MEMORY_GUARD_GIB),
        "--research-contract",
        str(study_path),
        "--json",
    ]
    try:
        _guarded_command(build_command, log_path=logs / "memory_guard_prepare.json")
        staging_manifest = staging / "pack" / "manifest.json"
        if not staging_manifest.is_file():
            raise FileNotFoundError("pack build completed without a manifest")
        os.replace(staging, store)
        manifest_path = store / "pack" / "manifest.json"
        _relocate_pack_manifest(manifest_path, staging, store)
        fold_root = store / "folds"
        fold_payload = walkforward.build_development_walkforward_folds(
            source_view=manifest_path,
            development_years=DEVELOPMENT_YEARS,
            train_start_year=2010,
            store_root=fold_root,
            overwrite=False,
        )
        fold_results = {
            int(item["development_year"]): item for item in fold_payload["folds"]
        }
        material_preflight = _pack_preflight(
            manifest_path=manifest_path, fold_results=fold_results
        )
        study["material"] = {
            "store_root": str(store),
            "pack_manifest": str(manifest_path),
            "fold_root": str(fold_root),
            "fold_views": {
                str(year): str(fold_results[year]["view_path"])
                for year in DEVELOPMENT_YEARS
            },
            "preflight": material_preflight,
        }
        _update_runtime(study_path, study, status="prepared", completed_years=[])
    except Exception as exc:
        _update_runtime(study_path, study, status="prepare_failed", error=str(exc))
        raise
    return json.loads(study_path.read_text(encoding="utf-8"))


def _completed_run_dirs(study_dir: Path, year: int) -> tuple[list[Path], list[Path]]:
    root = study_dir / "runs" / "development"
    matches = sorted(root.glob(f"seq100_development_{PROFILE_NAME}_{year}_seed{SEED}_*"))
    completed = [path for path in matches if (path / "sequence_path_training_summary.json").is_file()]
    partial = [path for path in matches if path not in completed]
    return completed, partial


def _validate_run_summary(path: Path, year: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("best_epoch", 0)) < 1:
        raise ValueError(f"fold {year} has no valid best epoch")
    if str(payload.get("evaluation_mode", "")) != "development":
        raise ValueError(f"fold {year} did not use development evaluation mode")
    sample_selection = dict(payload.get("sample_selection", {}) or {})
    development = dict(sample_selection.get("development_candidates", {}) or {})
    if int(development.get("selected_date_count", 0)) <= 0:
        raise ValueError(f"fold {year} has no candidate-complete development dates")
    return payload


def run_current_study(
    *, study_root: str | Path = STUDY_ROOT, max_folds: int = 0
) -> dict[str, Any]:
    _assert_no_other_research_process()
    study_path, study = _load_study(study_root)
    study_dir = study_path.parent
    runtime = dict(study.get("runtime", {}) or {})
    if str(runtime.get("status", "")) not in {"prepared", "running", "run_failed"}:
        raise ValueError(f"study is not runnable from status={runtime.get('status', '')}")
    fold_views = dict(dict(study.get("material", {}) or {}).get("fold_views", {}) or {})
    completed_years: list[int] = []
    launched = 0
    _update_runtime(study_path, study, status="running", error=None)
    try:
        for year in DEVELOPMENT_YEARS:
            completed, partial = _completed_run_dirs(study_dir, year)
            if len(completed) > 1:
                raise ValueError(f"fold {year} has multiple completed runs")
            if completed:
                _validate_run_summary(completed[0] / "sequence_path_training_summary.json", year)
                completed_years.append(year)
                continue
            for path in partial:
                _remove_path(path)
            if max_folds > 0 and launched >= int(max_folds):
                break
            view = Path(str(fold_views.get(str(year), "") or ""))
            if not view.is_file():
                raise FileNotFoundError(f"fold view is missing for {year}: {view}")
            profile = replace(
                PROFILE_SPECS[PROFILE_COMMAND].default_profile(),
                store_view=view,
                output_root=study_dir / "runs" / "development",
                run_tag=f"seq100_development_{PROFILE_NAME}_{year}_seed{SEED}_current",
                epochs=10,
                batch_size=512,
                seed=SEED,
                top_k="1,3,5,10",
                prediction_mode="compact",
                evaluation_mode="development",
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                early_stopping_metric="development_total_loss",
                early_stopping_mode="min",
                min_complete_epochs=1,
                max_samples_per_split=0,
            )
            train_argv = build_todayclose_path_only_train_argv(profile)
            command = [
                str(PYTHON),
                "-m",
                "daily_research.path_policy.qdp_v2_sequence_path_training",
                *train_argv,
                "--development-contract",
                str(study_path),
                "--json",
            ]
            _update_runtime(
                study_path,
                study,
                status="running",
                completed_years=completed_years,
                current_year=year,
            )
            _guarded_command(
                command,
                log_path=study_dir / "logs" / f"memory_guard_fold_{year}.json",
            )
            completed, _ = _completed_run_dirs(study_dir, year)
            if len(completed) != 1:
                raise RuntimeError(f"fold {year} did not produce exactly one completed run")
            _validate_run_summary(completed[0] / "sequence_path_training_summary.json", year)
            completed_years.append(year)
            launched += 1
            _update_runtime(
                study_path,
                study,
                status="running",
                completed_years=completed_years,
                current_year=None,
            )
        final_status = "runs_completed" if completed_years == list(DEVELOPMENT_YEARS) else "running"
        _update_runtime(
            study_path,
            study,
            status=final_status,
            completed_years=completed_years,
        )
    except Exception as exc:
        _update_runtime(
            study_path,
            study,
            status="run_failed",
            completed_years=completed_years,
            error=str(exc),
        )
        raise
    return status_current_study(study_root=study_root)


def status_current_study(*, study_root: str | Path = STUDY_ROOT) -> dict[str, Any]:
    study_path, study = _load_study(study_root)
    years: dict[str, Any] = {}
    for year in DEVELOPMENT_YEARS:
        completed, partial = _completed_run_dirs(study_path.parent, year)
        years[str(year)] = {
            "status": "completed" if len(completed) == 1 else ("partial" if partial else "pending"),
            "completed_run": str(completed[0]) if len(completed) == 1 else None,
            "partial_runs": [str(path) for path in partial],
        }
    return {
        "study": str(study_path),
        "runtime": dict(study.get("runtime", {}) or {}),
        "pack_manifest": dict(study.get("material", {}) or {}).get("pack_manifest"),
        "years": years,
        "available_memory_gib": float(psutil.virtual_memory().available / 1024**3),
        "h_free_gib": float(shutil.disk_usage(WORKSPACE_ROOT).free / 1024**3),
    }


def _finite_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"metric {column} contains non-finite values")
    return float(values.mean())


def _diagnostic_mean_and_coverage(
    frame: pd.DataFrame, column: str
) -> tuple[float, float]:
    """Average an optional diagnostic without weakening executable-data gates.

    Opportunity labels are intentionally not part of candidate eligibility, so a
    selected candidate can legitimately lack that future-only diagnostic.  Score
    and execution metrics still use ``_finite_mean`` and therefore remain strict.
    """

    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        raise ValueError(f"diagnostic metric {column} has no finite values")
    return float(values[finite].mean()), float(finite.mean())


def _year_metrics(run_dir: Path, year: int) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _validate_run_summary(run_dir / "sequence_path_training_summary.json", year)
    daily = pd.read_csv(run_dir / "daily_topk_metrics.csv")
    daily = daily[daily["split"].astype(str).eq("development")].copy()
    if daily.empty or not set(daily["top_k"].astype(int).unique()) == {1, 3, 5, 10}:
        raise ValueError(f"fold {year} daily TopK evidence is incomplete")
    metrics: dict[str, Any] = {
        "development_year": year,
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
    }
    for top_k in (1, 3, 5, 10):
        group = daily[daily["top_k"].astype(int).eq(top_k)]
        metrics[f"top{top_k}_selected_base"] = _finite_mean(
            group, "selected_net_realized_plan_return_base"
        )
        metrics[f"top{top_k}_universe_base"] = _finite_mean(
            group, "universe_net_realized_plan_return_base"
        )
        metrics[f"top{top_k}_base_alpha"] = _finite_mean(
            group, "alpha_net_realized_plan_return_base"
        )
        metrics[f"top{top_k}_stress_alpha"] = _finite_mean(
            group, "alpha_net_realized_plan_return_stress"
        )
        metrics[f"top{top_k}_selected_stress"] = _finite_mean(
            group, "selected_net_realized_plan_return_stress"
        )
        metrics[f"top{top_k}_universe_stress"] = _finite_mean(
            group, "universe_net_realized_plan_return_stress"
        )
        opportunity_alpha, opportunity_day_coverage = _diagnostic_mean_and_coverage(
            group, "alpha_opportunity_value"
        )
        metrics[f"top{top_k}_opportunity_alpha"] = opportunity_alpha
        metrics[f"top{top_k}_opportunity_alpha_day_coverage"] = (
            opportunity_day_coverage
        )
        metrics[f"top{top_k}_selected_opportunity_label_coverage"] = _finite_mean(
            group, "selected_opportunity_value_coverage"
        )
    top3 = daily[daily["top_k"].astype(int).eq(3)]
    average_exit_day, exit_day_observation_day_coverage = (
        _diagnostic_mean_and_coverage(top3, "selected_realized_plan_exit_day")
    )
    metrics.update(
        {
            "top3_entry_fill_rate": _finite_mean(top3, "selected_entry_fill_rate"),
            "top3_execution_return_coverage": _finite_mean(
                top3, "selected_realized_plan_return_coverage"
            ),
            "top3_average_exit_day": average_exit_day,
            "top3_exit_day_observation_day_coverage": (
                exit_day_observation_day_coverage
            ),
            "top3_terminal_recovery_rate": float(
                pd.to_numeric(
                    top3["selected_terminal_recovery_count_base"], errors="coerce"
                ).sum()
                / max(1.0, pd.to_numeric(top3["selected_count"], errors="coerce").sum())
            ),
            "top3_unresolved_exit_rate": float(
                pd.to_numeric(
                    top3["selected_terminal_recovery_count_base"], errors="coerce"
                ).sum()
                / max(1.0, pd.to_numeric(top3["selected_count"], errors="coerce").sum())
            ),
            "top3_opportunity_execution_gap": (
                metrics["top3_opportunity_alpha"]
                - metrics["top3_base_alpha"]
            ),
        }
    )
    split_metrics = list(summary.get("split_metrics", []) or [])
    development_split = next(
        (item for item in split_metrics if str(item.get("split", "")) == "development"),
        None,
    )
    if not development_split:
        raise ValueError(f"fold {year} summary has no development split metrics")
    metrics["rank_ic"] = float(development_split["rank_ic_mean"])
    metrics["candidate_score_coverage"] = float(
        development_split["candidate_complete_score_coverage"]
    )
    if metrics["candidate_score_coverage"] != 1.0:
        raise ValueError(f"fold {year} candidate score coverage is incomplete")
    if metrics["top3_execution_return_coverage"] != 1.0:
        raise ValueError(f"fold {year} Top3 execution return coverage is incomplete")
    return metrics, summary


def _write_summary_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    yearly = list(summary["yearly_metrics"])
    lines = [
        "# Seq100 Corrected Rolling 2023-2025 Result",
        "",
        "- Status: completed / report only / no final fit / no QDP or execution change.",
        "- Profile: daily-only summary_v2 with low-weight OHLCVA auxiliary supervision.",
        "- Primary metric: equal-year mean cost-after Top3 executable alpha.",
        "",
        "| year | best epoch | Top1 alpha | Top3 alpha | Top5 alpha | Top10 alpha | stress Top3 | rank IC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in yearly:
        lines.append(
            "| {development_year} | {best_epoch} | {top1_base_alpha:.4%} | "
            "{top3_base_alpha:.4%} | {top5_base_alpha:.4%} | {top10_base_alpha:.4%} | "
            "{top3_stress_alpha:.4%} | {rank_ic:.6f} |".format(**row)
        )
    aggregate = dict(summary["equal_year_metrics"])
    lines.extend(
        [
            "",
            f"Equal-year Top3 alpha: `{aggregate['top3_base_alpha']:.4%}`.",
            f"Positive Top3 years: `{aggregate['positive_top3_years']}/3`.",
            f"Worst-year Top3 alpha: `{aggregate['worst_year_top3_base_alpha']:.4%}`.",
            f"Suggested final-fit epochs (not executed): `{summary['suggested_final_fit_epochs']}`.",
            "",
            "No winner, champion, final model, deployment, QDP update, or execution change was created.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_current_study(*, study_root: str | Path = STUDY_ROOT) -> dict[str, Any]:
    study_path, study = _load_study(study_root)
    study_dir = study_path.parent
    yearly: list[dict[str, Any]] = []
    run_summaries: dict[str, Any] = {}
    for year in DEVELOPMENT_YEARS:
        completed, _ = _completed_run_dirs(study_dir, year)
        if len(completed) != 1:
            raise ValueError(f"summarize requires exactly one completed fold for {year}")
        metrics, run_summary = _year_metrics(completed[0], year)
        yearly.append(metrics)
        run_summaries[str(year)] = {
            "run_dir": str(completed[0]),
            "summary": str(completed[0] / "sequence_path_training_summary.json"),
            "optimizer_steps": int(
                run_summary.get(
                    "optimizer_step_count", run_summary.get("optimizer_steps", 0)
                )
                or 0
            ),
        }
    equal: dict[str, Any] = {}
    numeric_keys = sorted(
        key
        for key in yearly[0]
        if key not in {"development_year", "best_epoch", "completed_epochs"}
    )
    for key in numeric_keys:
        equal[key] = float(np.mean([float(row[key]) for row in yearly]))
    equal["positive_top3_years"] = int(
        sum(float(row["top3_base_alpha"]) > 0.0 for row in yearly)
    )
    equal["worst_year_top3_base_alpha"] = float(
        min(float(row["top3_base_alpha"]) for row in yearly)
    )

    pack_manifest = Path(str(dict(study.get("material", {}) or {})["pack_manifest"]))
    bridge_rows = _candidate_bridge_rows(
        pack=CandidateCompleteAuditPack(pack_manifest),
        source_study_root=study_dir,
        years=DEVELOPMENT_YEARS,
        profiles=(PROFILE_NAME,),
    )
    basis_year, basis_equal = _basis_summaries(bridge_rows)
    bridge_path = study_dir / "price_basis_selected_candidate_bridge.parquet"
    bridge_rows.to_parquet(bridge_path, index=False)
    basis_year.to_csv(study_dir / "price_basis_rank_bucket_year.csv", index=False)
    basis_equal.to_csv(study_dir / "price_basis_rank_bucket_equal_year.csv", index=False)

    result = {
        "schema_version": 1,
        "status": "completed",
        "study_id": study["study_id"],
        "profile": PROFILE_COMMAND,
        "seed": SEED,
        "development_years": list(DEVELOPMENT_YEARS),
        "primary_metric": "equal_year_mean_top3_net_realized_plan_return_base_alpha",
        "yearly_metrics": yearly,
        "equal_year_metrics": equal,
        "suggested_final_fit_epochs": int(median([row["best_epoch"] for row in yearly])),
        "final_fit_performed": False,
        "winner": None,
        "deployment_changed": False,
        "qdp_changed": False,
        "price_basis": {
            "selected_candidate_row_count": int(len(bridge_rows)),
            "rank_bucket_equal_year": basis_equal.to_dict("records"),
            "bridge_path": str(bridge_path),
        },
        "runs": run_summaries,
    }
    summary_json = study_dir / "research_summary.json"
    summary_md = study_dir / "research_summary.md"
    _write_json(summary_json, result)
    _write_summary_markdown(summary_md, result)
    _update_runtime(
        study_path,
        study,
        status="completed",
        completed_years=list(DEVELOPMENT_YEARS),
        research_summary_json=str(summary_json),
        research_summary_md=str(summary_md),
        error=None,
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-current-study corrected seq100 workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    contract = sub.add_parser("contract", help="Print the fixed current research contract.")
    contract.add_argument("--compact", action="store_true")
    contract.add_argument("--experiment", choices=("", "structured-path-v1"), default="")
    diagnose = sub.add_parser("diagnose", help="Stream checkpoint diagnostics without wide predictions.")
    diagnose.add_argument("--experiment", choices=("structured-path-v1",), default="structured-path-v1")
    diagnose.add_argument("--source-study-root", type=Path, default=None)
    diagnose.add_argument("--output-root", type=Path, default=None)
    prepare = sub.add_parser("prepare", help="Build the one daily-only pack and three folds.")
    prepare.add_argument("--experiment", choices=("", "structured-path-v1"), default="")
    prepare.add_argument("--qdp-root", type=Path, default=QDP_ROOT)
    prepare.add_argument("--study-root", type=Path, default=None)
    prepare.add_argument("--store-root", type=Path, default=STORE_ROOT)
    prepare.add_argument("--dry-run", action="store_true")
    run = sub.add_parser("run", help="Run pending folds sequentially.")
    run.add_argument("--experiment", choices=("", "structured-path-v1"), default="")
    run.add_argument("--study-root", type=Path, default=None)
    run.add_argument("--max-folds", type=int, default=0)
    run.add_argument("--max-tasks", type=int, default=0)
    status = sub.add_parser("status", help="Report pack and fold progress.")
    status.add_argument("--experiment", choices=("", "structured-path-v1"), default="")
    status.add_argument("--study-root", type=Path, default=None)
    summarize = sub.add_parser("summarize", help="Aggregate all three completed folds.")
    summarize.add_argument("--experiment", choices=("", "structured-path-v1"), default="")
    summarize.add_argument("--study-root", type=Path, default=None)
    vintages = sub.add_parser(
        "evaluate-vintages",
        help="Compare checkpoint vintages on the frozen 2025 candidate set.",
    )
    vintages.add_argument(
        "--experiment", choices=("structured-path-v1",), default="structured-path-v1"
    )
    vintages.add_argument("--study-root", type=Path, default=None)
    vintages.add_argument("--output-root", type=Path, default=None)
    vintages.add_argument("--max-tasks", type=int, default=0)
    vintage_task = sub.add_parser(
        "evaluate-vintage-task",
        help=argparse.SUPPRESS,
    )
    vintage_task.add_argument(
        "--experiment", choices=("structured-path-v1",), default="structured-path-v1"
    )
    vintage_task.add_argument("--suite-contract", type=Path, required=True)
    vintage_task.add_argument(
        "--profile",
        choices=("legal_flat_baseline", "structured_joint_turnover"),
        required=True,
    )
    vintage_task.add_argument("--vintage", choices=(2023, 2024, 2025), type=int, required=True)
    prepare_2026 = sub.add_parser(
        "prepare-2026-fold",
        help="Build the frozen 2026 dependency overlay and development fold.",
    )
    prepare_2026.add_argument("--study-root", type=Path, default=None)
    run_2026 = sub.add_parser(
        "run-2026-fold",
        help="Train the two 2026 profile checkpoints sequentially.",
    )
    run_2026.add_argument("--study-root", type=Path, default=None)
    run_2026.add_argument("--max-tasks", type=int, default=0)
    status_2026 = sub.add_parser(
        "status-2026-fold",
        help="Report 2026 fold preparation and training progress.",
    )
    status_2026.add_argument("--study-root", type=Path, default=None)
    evaluate_2026 = sub.add_parser(
        "evaluate-2026-vintages",
        help="Compare 2023-2026 checkpoints on the complete frozen 2026 window.",
    )
    evaluate_2026.add_argument("--output-root", type=Path, default=None)
    evaluate_2026.add_argument("--max-tasks", type=int, default=0)
    evaluate_2026_task = sub.add_parser(
        "evaluate-2026-vintage-task",
        help=argparse.SUPPRESS,
    )
    evaluate_2026_task.add_argument("--suite-contract", type=Path, required=True)
    evaluate_2026_task.add_argument(
        "--profile",
        choices=("legal_flat_baseline", "structured_joint_turnover"),
        required=True,
    )
    evaluate_2026_task.add_argument(
        "--vintage", choices=(2023, 2024, 2025, 2026), type=int, required=True
    )
    evaluate_2026_task.add_argument("--force-inference", action="store_true")
    migrate_integrity = sub.add_parser(
        "migrate-2026-contract-v2",
        help="Archive the completed v1 comparison metadata and migrate it to integrity v2.",
    )
    migrate_integrity.add_argument("--output-root", type=Path, default=None)
    build_compact_report = sub.add_parser(
        "build-compact-report",
        help="Archive the accumulated Seq100 report and replace it with the compact report.",
    )
    build_compact_report.add_argument("--artifact-path", type=Path, default=None)
    verify_integrity = sub.add_parser(
        "verify-research-integrity",
        help="Verify the Seq100 integrity v2 contract, task bundles, archive, and report.",
    )
    verify_integrity.add_argument("--output-root", type=Path, default=None)
    prepare_ablation = sub.add_parser(
        "prepare-structured-input-ablation",
        help="Build the Structured 180-day, turnover, and intraday ablation material.",
    )
    prepare_ablation.add_argument("--study-root", type=Path, default=None)
    run_ablation = sub.add_parser(
        "run-structured-input-ablation",
        help="Run pending Structured input-ablation folds sequentially.",
    )
    run_ablation.add_argument("--study-root", type=Path, default=None)
    run_ablation.add_argument("--max-tasks", type=int, default=0)
    status_ablation = sub.add_parser(
        "status-structured-input-ablation",
        help="Report Structured input-ablation stage, fold, batch, and monitor status.",
    )
    status_ablation.add_argument("--study-root", type=Path, default=None)
    review_ablation = sub.add_parser(
        "review-structured-input-stage1-capital",
        help="Review 100x32 versus 180x32 with fixed exits and finite capital.",
    )
    review_ablation.add_argument("--study-root", type=Path, default=None)
    review_ablation_stage = sub.add_parser(
        "review-structured-input-stage-capital",
        help="Run the finite-capital review for a completed Stage 2 or Stage 3.",
    )
    review_ablation_stage.add_argument("--study-root", type=Path, default=None)
    review_ablation_stage.add_argument("--stage", type=int, choices=(2, 3), required=True)
    capital_speed = sub.add_parser(
        "review-structured-input-capital-speed",
        help="Evaluate complete models with Top1/Top3, D2-D60, and own exits.",
    )
    capital_speed.add_argument("--study-root", type=Path, default=None)
    capital_speed.add_argument("--stage", type=int, choices=(1, 2), required=True)
    capital_speed.add_argument("--max-jobs", type=int, default=0)
    verify_capital_speed = sub.add_parser(
        "verify-structured-input-capital-speed",
        help="Verify a completed joint-model capital-speed review.",
    )
    verify_capital_speed.add_argument("--study-root", type=Path, default=None)
    verify_capital_speed.add_argument("--stage", type=int, choices=(1, 2), required=True)
    summarize_capital_speed = sub.add_parser(
        "summarize-structured-input-capital-speed",
        help="Build the completed pre-Stage-3 capital-speed summary.",
    )
    summarize_capital_speed.add_argument("--study-root", type=Path, default=None)
    summarize_ablation = sub.add_parser(
        "summarize-structured-input-ablation",
        help="Evaluate stage gates and build the Structured input-ablation summary.",
    )
    summarize_ablation.add_argument("--study-root", type=Path, default=None)
    verify_ablation = sub.add_parser(
        "verify-structured-input-ablation",
        help="Verify Structured input-ablation data, tasks, reports, and boundaries.",
    )
    verify_ablation.add_argument("--study-root", type=Path, default=None)
    prepare_window = sub.add_parser(
        "prepare-structured-training-window",
        help="Build bounded Structured 180x35 training-window fold views.",
    )
    prepare_window.add_argument("--study-root", type=Path, default=None)
    run_window = sub.add_parser(
        "run-structured-training-window",
        help="Run one pending bounded-history training task by default.",
    )
    run_window.add_argument("--study-root", type=Path, default=None)
    run_window.add_argument("--max-tasks", type=int, default=1)
    evaluate_window = sub.add_parser(
        "evaluate-structured-training-window",
        help="Run resumable cohort and continuous-account window evaluation.",
    )
    evaluate_window.add_argument("--study-root", type=Path, default=None)
    evaluate_window.add_argument("--max-jobs", type=int, default=0)
    status_window = sub.add_parser(
        "status-structured-training-window",
        help="Report bounded-history experiment state for manual diagnostics.",
    )
    status_window.add_argument("--study-root", type=Path, default=None)
    summarize_window = sub.add_parser(
        "summarize-structured-training-window",
        help="Build or read the final bounded-history comparison summary.",
    )
    summarize_window.add_argument("--study-root", type=Path, default=None)
    verify_window = sub.add_parser(
        "verify-structured-training-window",
        help="Verify bounded views, tasks, results, monitor state, and boundaries.",
    )
    verify_window.add_argument("--study-root", type=Path, default=None)
    prepare_180_2026 = sub.add_parser(
        "prepare-structured-180x35-2026",
        help="Build the Structured 180x35 2026 fold and compact feature suffix.",
    )
    prepare_180_2026.add_argument("--study-root", type=Path, default=None)
    run_180_2026 = sub.add_parser(
        "run-structured-180x35-2026",
        help="Train the fixed one-epoch Structured 180x35 2026 checkpoint.",
    )
    run_180_2026.add_argument("--study-root", type=Path, default=None)
    run_180_2026.add_argument("--max-tasks", type=int, default=1)
    evaluate_180_2026 = sub.add_parser(
        "evaluate-structured-180x35-2026",
        help="Run resumable full-label, extended-score, and fixed-D7 account jobs.",
    )
    evaluate_180_2026.add_argument("--study-root", type=Path, default=None)
    evaluate_180_2026.add_argument("--max-jobs", type=int, default=1)
    status_180_2026 = sub.add_parser(
        "status-structured-180x35-2026",
        help="Report Structured 180x35 2026 preparation, training, and evaluation state.",
    )
    status_180_2026.add_argument("--study-root", type=Path, default=None)
    summarize_180_2026 = sub.add_parser(
        "summarize-structured-180x35-2026",
        help="Build the Structured 180x35 2026 D7 comparison summary.",
    )
    summarize_180_2026.add_argument("--study-root", type=Path, default=None)
    verify_180_2026 = sub.add_parser(
        "verify-structured-180x35-2026",
        help="Verify the Structured 180x35 2026 study and protected boundaries.",
    )
    verify_180_2026.add_argument("--study-root", type=Path, default=None)
    diagnose_180_2026 = sub.add_parser(
        "diagnose-structured-180x35-2026",
        help="Compare the 2025 checkpoint and scan execution without changing the default decision.",
    )
    diagnose_180_2026.add_argument("--output-root", type=Path, default=None)
    verify_diagnose_180_2026 = sub.add_parser(
        "verify-structured-180x35-2026-diagnostics",
        help="Verify the 2025 checkpoint comparison and 2026 execution scan.",
    )
    verify_diagnose_180_2026.add_argument("--output-root", type=Path, default=None)
    evaluate_vintages_180_2026 = sub.add_parser(
        "evaluate-structured-180x35-2026-vintages",
        help="Evaluate the 2023-2026 Structured 180x35 checkpoints on common 2026 material.",
    )
    evaluate_vintages_180_2026.add_argument("--output-root", type=Path, default=None)
    evaluate_vintages_180_2026.add_argument("--max-jobs", type=int, default=0)
    verify_vintages_180_2026 = sub.add_parser(
        "verify-structured-180x35-2026-vintages",
        help="Verify the four-vintage Structured 180x35 2026 comparison.",
    )
    verify_vintages_180_2026.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {
        "prepare-structured-180x35-2026",
        "run-structured-180x35-2026",
        "evaluate-structured-180x35-2026",
        "status-structured-180x35-2026",
        "summarize-structured-180x35-2026",
        "verify-structured-180x35-2026",
        "diagnose-structured-180x35-2026",
        "verify-structured-180x35-2026-diagnostics",
        "evaluate-structured-180x35-2026-vintages",
        "verify-structured-180x35-2026-vintages",
    }:
        if args.command in {
            "evaluate-structured-180x35-2026-vintages",
            "verify-structured-180x35-2026-vintages",
        }:
            from daily_research.path_policy import seq100_structured_180x35_2026_vintages as vintages

            root = Path(args.output_root) if args.output_root else vintages.OUTPUT_ROOT
            result = (
                vintages.evaluate_vintages(
                    max_jobs=int(args.max_jobs), output_root=root
                )
                if args.command == "evaluate-structured-180x35-2026-vintages"
                else vintages.verify_vintages(output_root=root)
            )
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                    allow_nan=False,
                )
            )
            return 0
        if args.command in {
            "diagnose-structured-180x35-2026",
            "verify-structured-180x35-2026-diagnostics",
        }:
            from daily_research.path_policy import seq100_structured_180x35_2026_diagnostics as diagnostics

            root = Path(args.output_root) if args.output_root else diagnostics.OUTPUT_ROOT
            result = (
                diagnostics.run_diagnostics(output_root=root)
                if args.command == "diagnose-structured-180x35-2026"
                else diagnostics.verify_diagnostics(output_root=root)
            )
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                    allow_nan=False,
                )
            )
            return 0
        from daily_research.path_policy import seq100_structured_180x35_2026 as structured_180

        root = Path(args.study_root) if args.study_root else structured_180.STUDY_ROOT
        if args.command == "prepare-structured-180x35-2026":
            result = structured_180.prepare_structured_180x35_2026(study_root=root)
        elif args.command == "run-structured-180x35-2026":
            result = structured_180.run_structured_180x35_2026(
                study_root=root, max_tasks=int(args.max_tasks)
            )
        elif args.command == "evaluate-structured-180x35-2026":
            result = structured_180.evaluate_structured_180x35_2026(
                study_root=root, max_jobs=int(args.max_jobs)
            )
        elif args.command == "status-structured-180x35-2026":
            result = structured_180.status_structured_180x35_2026(study_root=root)
        elif args.command == "summarize-structured-180x35-2026":
            result = structured_180.summarize_structured_180x35_2026(study_root=root)
        else:
            result = structured_180.verify_structured_180x35_2026(
                study_root=root, require_complete=False
            )
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
                allow_nan=False,
            )
        )
        return 0
    if args.command in {
        "prepare-structured-training-window",
        "run-structured-training-window",
        "evaluate-structured-training-window",
        "status-structured-training-window",
        "summarize-structured-training-window",
        "verify-structured-training-window",
    }:
        from daily_research.path_policy import seq100_structured_training_window as window

        root = Path(args.study_root) if args.study_root else window.STUDY_ROOT
        if args.command == "prepare-structured-training-window":
            result = window.prepare_structured_training_window(study_root=root)
        elif args.command == "run-structured-training-window":
            result = window.run_structured_training_window(
                study_root=root, max_tasks=int(args.max_tasks)
            )
        elif args.command == "evaluate-structured-training-window":
            result = window.evaluate_structured_training_window(
                study_root=root, max_jobs=int(args.max_jobs)
            )
        elif args.command == "status-structured-training-window":
            result = window.status_structured_training_window(study_root=root)
        elif args.command == "summarize-structured-training-window":
            result = window.summarize_structured_training_window(study_root=root)
        else:
            result = window.verify_structured_training_window(study_root=root)
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
                allow_nan=False,
            )
        )
        return 0
    if args.command in {
        "prepare-structured-input-ablation",
        "run-structured-input-ablation",
        "status-structured-input-ablation",
        "review-structured-input-stage1-capital",
        "review-structured-input-stage-capital",
        "review-structured-input-capital-speed",
        "verify-structured-input-capital-speed",
        "summarize-structured-input-capital-speed",
        "summarize-structured-input-ablation",
        "verify-structured-input-ablation",
    }:
        from daily_research.path_policy import seq100_structured_input_ablation as ablation

        root = Path(args.study_root) if args.study_root else ablation.STUDY_ROOT
        if args.command == "prepare-structured-input-ablation":
            result = ablation.prepare_structured_input_ablation(study_root=root)
        elif args.command == "run-structured-input-ablation":
            result = ablation.run_structured_input_ablation(
                study_root=root, max_tasks=int(args.max_tasks)
            )
        elif args.command == "status-structured-input-ablation":
            result = ablation.status_structured_input_ablation(study_root=root)
        elif args.command == "review-structured-input-stage1-capital":
            result = ablation.run_stage1_capital_review(study_root=root)
        elif args.command == "review-structured-input-stage-capital":
            result = ablation.run_stage_capital_review(
                stage=int(args.stage), study_root=root
            )
        elif args.command in {
            "review-structured-input-capital-speed",
            "verify-structured-input-capital-speed",
            "summarize-structured-input-capital-speed",
        }:
            from daily_research.path_policy import seq100_structured_capital_speed

            if args.command == "review-structured-input-capital-speed":
                result = seq100_structured_capital_speed.run_capital_speed_review(
                    stage=int(args.stage),
                    study_root=root,
                    max_jobs=int(args.max_jobs),
                )
            elif args.command == "verify-structured-input-capital-speed":
                result = seq100_structured_capital_speed.verify_capital_speed_review(
                    stage=int(args.stage), study_root=root
                )
            else:
                result = seq100_structured_capital_speed.build_pre_stage3_summary(
                    study_root=root
                )
        elif args.command == "summarize-structured-input-ablation":
            result = ablation.summarize_structured_input_ablation(study_root=root)
        else:
            result = ablation.verify_structured_input_ablation(study_root=root)
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
                allow_nan=False,
            )
        )
        return 0
    if args.command in {
        "prepare-2026-fold",
        "run-2026-fold",
        "status-2026-fold",
        "evaluate-2026-vintages",
        "evaluate-2026-vintage-task",
        "migrate-2026-contract-v2",
        "build-compact-report",
        "verify-research-integrity",
    }:
        from daily_research.path_policy import seq100_2026_fold_comparison as comparison_2026

        if args.command == "prepare-2026-fold":
            result = comparison_2026.prepare_2026_fold(
                study_root=(
                    Path(args.study_root)
                    if args.study_root
                    else comparison_2026.STUDY_ROOT
                )
            )
        elif args.command == "run-2026-fold":
            result = comparison_2026.run_2026_fold(
                study_root=(
                    Path(args.study_root)
                    if args.study_root
                    else comparison_2026.STUDY_ROOT
                ),
                max_tasks=int(args.max_tasks),
            )
        elif args.command == "status-2026-fold":
            result = comparison_2026.status_2026_fold()
        elif args.command == "evaluate-2026-vintages":
            result = comparison_2026.evaluate_2026_vintages(
                output_root=(
                    Path(args.output_root)
                    if args.output_root
                    else comparison_2026.ANALYSIS_ROOT
                ),
                max_tasks=int(args.max_tasks),
            )
        elif args.command == "evaluate-2026-vintage-task":
            result = comparison_2026.evaluate_2026_vintage_task(
                suite_contract_path=Path(args.suite_contract),
                profile=str(args.profile),
                vintage=int(args.vintage),
                force_inference=bool(args.force_inference),
            )
        elif args.command == "migrate-2026-contract-v2":
            from daily_research.path_policy import seq100_integrity_v2 as integrity

            result = integrity.migrate_2026_contract_v2(
                output_root=(
                    Path(args.output_root)
                    if args.output_root
                    else comparison_2026.ANALYSIS_ROOT
                )
            )
        elif args.command == "build-compact-report":
            from daily_research.path_policy import build_seq100_research_report as report

            result = report.build_compact_report(
                artifact_path=(
                    Path(args.artifact_path)
                    if args.artifact_path
                    else report.ARTIFACT_PATH
                )
            )
        else:
            from daily_research.path_policy import seq100_integrity_v2 as integrity

            result = integrity.verify_research_integrity(
                output_root=(
                    Path(args.output_root)
                    if args.output_root
                    else comparison_2026.ANALYSIS_ROOT
                )
            )
        indent = 2
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=indent,
                default=_json_default,
                allow_nan=False,
            )
        )
        return 0
    structured = str(getattr(args, "experiment", "")) == "structured-path-v1"
    if structured:
        from daily_research.path_policy import seq100_structured_experiment as structured_experiment

        structured_root = Path(args.study_root) if getattr(args, "study_root", None) else structured_experiment.STUDY_ROOT
        if args.command == "contract":
            result = structured_experiment.contract()
            indent = None if args.compact else 2
        elif args.command == "diagnose":
            result = structured_experiment.diagnose(
                source_study_root=(
                    Path(args.source_study_root)
                    if args.source_study_root
                    else structured_experiment.SOURCE_STUDY_ROOT
                ),
                output_root=(
                    Path(args.output_root)
                    if args.output_root
                    else structured_experiment.STUDY_ROOT / "diagnostics" / "legacy_checkpoints"
                ),
            )
            indent = 2
        elif args.command == "prepare":
            result = structured_experiment.prepare(
                study_root=structured_root,
                dry_run=bool(args.dry_run),
            )
            indent = 2
        elif args.command == "run":
            result = structured_experiment.run(
                study_root=structured_root,
                max_tasks=int(args.max_tasks or args.max_folds),
            )
            indent = 2
        elif args.command == "status":
            result = structured_experiment.status(study_root=structured_root)
            indent = 2
        elif args.command == "summarize":
            result = structured_experiment.summarize(study_root=structured_root)
            indent = 2
        elif args.command == "evaluate-vintages":
            from daily_research.path_policy import seq100_checkpoint_freshness as freshness

            result = freshness.evaluate_vintages(
                study_root=structured_root,
                output_root=(
                    Path(args.output_root)
                    if args.output_root
                    else freshness.DEFAULT_OUTPUT_ROOT
                ),
                max_tasks=int(args.max_tasks),
            )
            indent = 2
        else:
            from daily_research.path_policy import seq100_checkpoint_freshness as freshness

            result = freshness.evaluate_vintage_task(
                suite_contract_path=Path(args.suite_contract),
                profile=str(args.profile),
                vintage=int(args.vintage),
            )
            indent = 2
    elif args.command == "contract":
        result = current_contract()
        indent = None if args.compact else 2
    elif args.command == "prepare" and args.dry_run:
        result = prepare_dry_run(qdp_root=args.qdp_root, store_root=args.store_root)
        indent = 2
    elif args.command == "prepare":
        result = prepare_current_study(
            qdp_root=args.qdp_root,
            study_root=args.study_root or STUDY_ROOT,
            store_root=args.store_root,
        )
        indent = 2
    elif args.command == "run":
        result = run_current_study(study_root=args.study_root or STUDY_ROOT, max_folds=args.max_folds)
        indent = 2
    elif args.command == "status":
        result = status_current_study(study_root=args.study_root or STUDY_ROOT)
        indent = 2
    else:
        result = summarize_current_study(study_root=args.study_root or STUDY_ROOT)
        indent = 2
    print(json.dumps(result, ensure_ascii=False, indent=indent, default=_json_default, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
