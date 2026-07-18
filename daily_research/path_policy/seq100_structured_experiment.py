from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import psutil
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy.seq100_development import (
    DEVELOPMENT_YEARS,
    EXPECTED_QDP_AS_OF,
    PYTHON,
    SEED,
    _assert_no_other_research_process,
    _guarded_command,
    _remove_path,
    _update_runtime,
    _validate_run_summary,
    _year_metrics,
)
from daily_research.path_policy.seq100_mainline import (
    PROFILE_SPECS,
    TodayClosePathOnlyProfile,
    build_todayclose_path_only_train_argv,
)
from daily_research.path_policy.seq100_candidate_execution import evaluate_candidate_execution
from daily_research.path_policy.seq100_walkforward import (
    _compute_development_fold_training_contract,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PACK_MANIFEST = WORKSPACE_ROOT / "daily_research/data/research_store/seq100_current/pack/manifest.json"
SOURCE_STUDY_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_corrected_rolling_2023_2025_v1"
)
STUDY_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_legal_structured_path_rolling_2023_2025_v1"
)
SUPPLEMENT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/data/research_store/seq100_current/supplements/relative_turnover_v1"
)
EXPERIMENT_ID = "structured-path-v1"
STUDY_ID = "seq100_legal_structured_path_rolling_2023_2025_v1"
PROFILE_ORDER = ("legal_flat_baseline", "structured_joint_turnover")
LEGAL_EXIT_CONTRACT = {
    "earliest_legal_exit_day": 2,
    "exit_argmax_domain": [2, 60],
    "tie_break": "earliest_legal_day",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_qdp_shards(pack: Mapping[str, Any], domain: str) -> tuple[dict[str, Any], list[Path]]:
    source = dict(dict(pack.get("qdp_source_manifests", {}) or {}).get(domain, {}) or {})
    manifest_path = Path(str(source.get("manifest_path", "") or "")).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing QDP {domain} manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    qdp_root = Path(str(pack["qdp_root"])).resolve()
    shards = [qdp_root / str(item["path"]) for item in list(manifest.get("shards", []) or [])]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing QDP {domain} shards: {missing[:3]}")
    return manifest, shards


def _duckdb_file_list(paths: Sequence[Path]) -> str:
    return "[" + ",".join("'" + str(path).replace("'", "''") + "'" for path in paths) + "]"


def _memmap_meta(path: Path, shape: Sequence[int]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "shape": [int(item) for item in shape],
        "dtype": "float32",
        "sha256": _file_sha256(path),
        "file_size": int(path.stat().st_size),
    }


def _past20_positive_median(values: np.ndarray) -> np.ndarray:
    """As-of median of the latest 20 strictly positive observations, carried forward."""

    source = np.asarray(values, dtype=np.float32).reshape(-1)
    result = np.full(source.shape, np.nan, dtype=np.float32)
    positive_positions = np.flatnonzero(np.isfinite(source) & (source > 0.0))
    if int(positive_positions.size) < 20:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(source[positive_positions], 20)
    medians = np.median(windows, axis=1).astype(np.float32)
    update_positions = positive_positions[19:]
    lookup = np.searchsorted(
        update_positions, np.arange(source.size, dtype=np.int64), side="right"
    ) - 1
    valid = lookup >= 0
    result[valid] = medians[lookup[valid]]
    return result


def _validate_supplement_manifest(path: Path, pack: Mapping[str, Any]) -> dict[str, Any]:
    payload = _read_json(path)
    expected_shape = [int(pack["date_count"]), int(pack["symbol_count"])]
    if payload.get("shape") != expected_shape:
        raise ValueError("relative turnover supplement shape does not match the current pack")
    if dict(payload.get("active_datasets", {}) or {}) != dict(pack.get("active_datasets", {}) or {}):
        raise ValueError("relative turnover supplement QDP datasets are stale")
    for key in ("log_turnover_pct", "past20_positive_median"):
        meta = dict(payload.get(key, {}) or {})
        data_path = Path(str(meta.get("path", "") or "")).resolve()
        if not data_path.is_file() or _file_sha256(data_path) != str(meta.get("sha256", "")):
            raise ValueError(f"relative turnover supplement {key} is missing or corrupted")
    audit = dict(payload.get("unit_audit", {}) or {})
    if int(audit.get("comparable_count", 0)) < 100:
        raise ValueError("relative turnover supplement unit audit is incomplete")
    return payload


def build_relative_turnover_supplement(
    *,
    pack_manifest: Path = PACK_MANIFEST,
    output_root: Path = SUPPLEMENT_ROOT,
) -> dict[str, Any]:
    """Build the only new material: two aligned ~48 MiB float32 panels."""

    pack = _read_json(pack_manifest.resolve())
    target = output_root.resolve()
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        return _validate_supplement_manifest(manifest_path, pack)

    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        _remove_path(staging)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        date_values = [str(item) for item in list(pack["date_values"])]
        symbol_values = [str(item) for item in list(pack["symbol_values"])]
        shape = (len(date_values), len(symbol_values))
        dates = pd.DataFrame(
            {
                "trade_date": date_values,
                "date_idx": np.arange(shape[0], dtype=np.int32),
                "year": np.asarray([int(value[:4]) for value in date_values], dtype=np.int16),
            }
        )
        symbols = pd.DataFrame(
            {"symbol": symbol_values, "symbol_idx": np.arange(shape[1], dtype=np.int32)}
        )
        daily_meta = dict(dict(pack["feature_channels"])["daily_raw"])
        daily_shape = tuple(int(item) for item in daily_meta["shape"])
        volume_idx = list(daily_meta["columns"]).index("volume")
        daily = np.memmap(
            str(Path(daily_meta["path"])), dtype=np.float32, mode="r", shape=daily_shape
        )
        share_manifest, share_shards = _resolve_qdp_shards(pack, "share_capital")
        log_path = staging / "log_turnover_pct.float32.dat"
        baseline_path = staging / "past20_positive_median.float32.dat"
        log_panel = np.memmap(str(log_path), dtype=np.float32, mode="w+", shape=shape)
        log_panel[:] = np.nan

        con = duckdb.connect()
        con.register("pack_dates", dates)
        con.register("pack_symbols", symbols)
        parquet_files = _duckdb_file_list(share_shards)
        missing_share_count = 0
        try:
            for year in sorted(dates["year"].unique().tolist()):
                date_slice = dates.index[dates["year"].eq(int(year))].to_numpy(dtype=np.int64)
                start = int(date_slice.min())
                stop = int(date_slice.max()) + 1
                frame = con.execute(
                    f"""
                    SELECT d.date_idx, s.symbol_idx, c.float_share
                    FROM read_parquet({parquet_files}) c
                    INNER JOIN pack_dates d ON c.trade_date = d.trade_date
                    INNER JOIN pack_symbols s ON c.symbol = s.symbol
                    WHERE d.year = ?
                    """,
                    [int(year)],
                ).fetch_df()
                shares = np.full((stop - start, shape[1]), np.nan, dtype=np.float64)
                if not frame.empty:
                    row = frame["date_idx"].to_numpy(dtype=np.int64) - start
                    col = frame["symbol_idx"].to_numpy(dtype=np.int64)
                    shares[row, col] = pd.to_numeric(
                        frame["float_share"], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                volume = np.asarray(daily[start:stop, :, volume_idx], dtype=np.float64)
                valid_volume = np.isfinite(volume) & (volume >= 0.0)
                valid = valid_volume & np.isfinite(shares) & (shares > 0.0)
                missing_share_count += int((valid_volume & ~valid).sum())
                result = np.full(volume.shape, np.nan, dtype=np.float32)
                result[valid] = np.log1p(100.0 * volume[valid] / shares[valid]).astype(
                    np.float32
                )
                log_panel[start:stop, :] = result
                log_panel.flush()
        finally:
            con.close()

        baseline_panel = np.memmap(
            str(baseline_path), dtype=np.float32, mode="w+", shape=shape
        )
        baseline_panel[:] = np.nan
        insufficient_cell_count = 0
        for symbol_idx in range(shape[1]):
            values = np.asarray(log_panel[:, symbol_idx], dtype=np.float32)
            baseline = _past20_positive_median(values)
            baseline_panel[:, symbol_idx] = baseline
            insufficient_cell_count += int((~np.isfinite(baseline)).sum())
        baseline_panel.flush()

        valuation_manifest, valuation_shards = _resolve_qdp_shards(pack, "valuation")
        con = duckdb.connect()
        con.register("pack_dates", dates[["trade_date", "date_idx"]])
        con.register("pack_symbols", symbols)
        try:
            valuation_files = _duckdb_file_list(valuation_shards)
            sample = con.execute(
                f"""
                SELECT d.date_idx, s.symbol_idx, v.turnover_rate
                FROM read_parquet({valuation_files}) v
                INNER JOIN pack_dates d ON v.trade_date = d.trade_date
                INNER JOIN pack_symbols s ON v.symbol = s.symbol
                WHERE v.turnover_rate IS NOT NULL
                ORDER BY hash(v.symbol || v.trade_date)
                LIMIT 2000
                """
            ).fetch_df()
        finally:
            con.close()
        row = sample["date_idx"].to_numpy(dtype=np.int64)
        col = sample["symbol_idx"].to_numpy(dtype=np.int64)
        computed = np.expm1(np.asarray(log_panel[row, col], dtype=np.float64))
        observed = pd.to_numeric(sample["turnover_rate"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        comparable = np.isfinite(computed) & np.isfinite(observed) & (observed > 0.0)
        relative_error = np.abs(computed[comparable] - observed[comparable]) / np.maximum(
            np.abs(observed[comparable]), 1.0e-12
        )
        ratio = computed[comparable] / observed[comparable]
        unit_audit = {
            "sample_size": int(len(sample)),
            "comparable_count": int(comparable.sum()),
            "median_computed_to_valuation_ratio": float(np.median(ratio)),
            "p99_relative_error": float(np.quantile(relative_error, 0.99)),
            "within_0_1pct_rate": float((relative_error <= 0.001).mean()),
            "systematic_multiplier_detected": bool(
                not (0.99 <= float(np.median(ratio)) <= 1.01)
            ),
        }
        if int(comparable.sum()) < 100 or unit_audit["systematic_multiplier_detected"]:
            raise ValueError(f"relative turnover unit audit failed: {unit_audit}")

        del baseline_panel, log_panel, daily
        payload = {
            "schema_version": 1,
            "artifact_type": "seq100_relative_turnover_supplement",
            "created_at": _now(),
            "formula": "log1p(100 * daily_volume / same_day_float_share)",
            "activity_target": (
                "future_log_turnover_pct - median(last_20_positive_observed_log_turnover_pct_"
                "through_signal_date)"
            ),
            "shape": list(shape),
            "date_start": date_values[0],
            "date_end": date_values[-1],
            "date_values_sha256": _canonical_digest(date_values),
            "symbol_values_sha256": _canonical_digest(symbol_values),
            "active_datasets": dict(pack.get("active_datasets", {}) or {}),
            "qdp_sources": {
                "share_capital": {
                    "dataset_id": share_manifest["dataset_id"],
                    "dataset_json_sha256": dict(pack["qdp_source_manifests"])[
                        "share_capital"
                    ]["dataset_json_sha256"],
                },
                "valuation": {
                    "dataset_id": valuation_manifest["dataset_id"],
                    "dataset_json_sha256": dict(pack["qdp_source_manifests"])[
                        "valuation"
                    ]["dataset_json_sha256"],
                },
            },
            "baseline_contract": {
                "positive_observation_count": 20,
                "uses_signal_date_or_earlier_only": True,
                "future_reference_count": 0,
                "insufficient_history_cell_count": int(insufficient_cell_count),
                "insufficient_history_behavior": "activity_target_nan_price_sample_retained",
            },
            "missing_share_for_nonnegative_volume_count": int(missing_share_count),
            "unit_audit": unit_audit,
            "log_turnover_pct": _memmap_meta(log_path, shape),
            "past20_positive_median": _memmap_meta(baseline_path, shape),
        }
        _write_json(staging / "manifest.json", payload)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return _validate_supplement_manifest(target / "manifest.json", pack)
    except Exception:
        if staging.exists():
            _remove_path(staging)
        raise


def _profile_contracts(batch_size: int) -> dict[str, dict[str, Any]]:
    baseline = replace(
        PROFILE_SPECS["train-daily-only-summary-v2-ohlcva-aux-low"].default_profile(),
        batch_size=int(batch_size),
        path_value_gradient_profile=training.PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
        early_stopping_metric=training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
        early_stopping_mode=training.EARLY_STOPPING_MODE_MIN,
    )
    structured = replace(
        baseline,
        model_type="gru_structured_joint_turnover",
        path_loss_weight=0.35,
        summary_loss_weight=0.20,
        value_loss_weight=0.15,
        rank_loss_weight=0.15,
        va_level_loss_weight=0.0,
        va_delta_loss_weight=0.0,
        geometry_loss_weight=0.10,
        utility_curve_loss_weight=0.05,
        turnover_level_loss_weight=0.02,
        turnover_delta_loss_weight=0.01,
    )
    return {
        "legal_flat_baseline": asdict(baseline),
        "structured_joint_turnover": asdict(structured),
    }


def contract(*, batch_size: int = 512) -> dict[str, Any]:
    profiles = _profile_contracts(batch_size)
    for payload in profiles.values():
        for key in ("store_view", "output_root", "run_tag"):
            payload.pop(key, None)
    semantic = {
        "schema_version": 1,
        "contract_id": STUDY_ID,
        "experiment": EXPERIMENT_ID,
        "data": {
            "qdp_expected_as_of": EXPECTED_QDP_AS_OF,
            "source_pack": str(PACK_MANIFEST.resolve()),
            "pack_mutated": False,
            "input_dim": 32,
            "lookback_days": 100,
            "forward_days": 60,
            "execution_tail_days": 20,
            "relative_turnover_supplement": str(SUPPLEMENT_ROOT.resolve()),
        },
        "legal_exit_contract": dict(LEGAL_EXIT_CONTRACT),
        "development_protocol": {
            "years": list(DEVELOPMENT_YEARS),
            "method": "purged_expanding_development_walkforward",
            "split_roles": {"fit": "train", "evaluation": "development"},
            "seed": SEED,
            "task_order": [
                f"{profile}:{year}" for profile in PROFILE_ORDER for year in DEVELOPMENT_YEARS
            ],
            "final_fit_in_scope": False,
        },
        "early_stopping": {
            "metric": training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
            "formula": "0.45*path + 0.20*summary + 0.20*legal_value + 0.15*rank",
            "excludes": ["volume", "amount", "turnover", "geometry", "utility_curve"],
            "mode": "min",
            "minimum_complete_epochs": 1,
            "maximum_epochs": 10,
            "patience": 2,
            "restore_best_checkpoint": True,
        },
        "profiles": profiles,
        "metric_contract": {
            "primary": "equal_year_mean_top3_net_realized_plan_return_base_alpha",
            "year_weighting": "equal",
            "top_k": [1, 3, 5, 10],
            "automatic_winner": False,
        },
        "protected_boundaries": {
            "update_qdp": False,
            "mutate_source_pack": False,
            "mutate_source_study": False,
            "change_active_execution": False,
            "create_final_model": False,
        },
    }
    return {
        "schema_version": 1,
        "workflow_id": "seq100_legal_structured_path_development",
        "operational_commands": ["contract", "diagnose", "prepare", "run", "status", "summarize"],
        "contract": semantic,
        "contract_sha256": _canonical_digest(semantic),
    }


def _probe_cuda_batch_size() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "selected_batch_size": 512, "attempts": []}
    attempts: list[dict[str, Any]] = []
    props = torch.cuda.get_device_properties(0)
    total = int(props.total_memory)
    for batch_size in (512, 256):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = training.SequencePathModel(
                input_dim=32,
                hidden_dim=128,
                layers=2,
                forward_days=60,
                summary_dim=12,
                dropout=0.1,
                model_type="gru_structured_joint_turnover",
            ).cuda()
            x = torch.randn(batch_size, 100, 32, device="cuda")
            target_raw = torch.randn(batch_size, 60, 4, device="cuda") * 0.1
            _geometry, y_path = training._structured_geometry_to_ohlc_torch(target_raw)
            y_activity = torch.randn(batch_size, 60, device="cuda")
            date_idx = torch.zeros(batch_size, dtype=torch.long, device="cuda")
            with torch.amp.autocast(device_type="cuda", enabled=True):
                output = model(x)
                loss, _parts = training._compute_loss(
                    output,
                    y_path.detach(),
                    torch.empty(batch_size, 0, device="cuda"),
                    date_idx,
                    y_activity_path=y_activity,
                    value_index=0,
                    path_weight=0.35,
                    summary_weight=0.20,
                    value_weight=0.15,
                    rank_weight=0.15,
                    geometry_weight=0.10,
                    utility_curve_weight=0.05,
                    turnover_level_weight=0.02,
                    turnover_delta_weight=0.01,
                    price_anchor="today_close",
                    summary_loss_profile=training.SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
                    path_value_gradient_profile=training.PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
                )
            loss.backward()
            peak = int(torch.cuda.max_memory_reserved())
            ratio = float(peak / max(total, 1))
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "ok",
                    "peak_reserved_bytes": peak,
                    "total_bytes": total,
                    "peak_reserved_ratio": ratio,
                }
            )
            del model, x, target_raw, y_path, y_activity, date_idx, output, loss
            torch.cuda.empty_cache()
            if batch_size == 512 and ratio > 0.90:
                continue
            return {
                "device": torch.cuda.get_device_name(0),
                "selected_batch_size": batch_size,
                "attempts": attempts,
            }
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch_size": batch_size, "status": "oom"})
            torch.cuda.empty_cache()
            if batch_size == 256:
                raise
    raise RuntimeError("CUDA batch-size probe did not select a batch size")


def prepare(
    *,
    study_root: Path = STUDY_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    pack = _read_json(PACK_MANIFEST)
    source_study = _read_json(SOURCE_STUDY_ROOT / "study.json")
    source_views = dict(dict(source_study.get("material", {}) or {}).get("fold_views", {}) or {})
    result = {
        "status": "dry_run" if dry_run else "prepared",
        "experiment": EXPERIMENT_ID,
        "study_root": str(study_root.resolve()),
        "source_pack": str(PACK_MANIFEST.resolve()),
        "source_pack_size_bytes": int(
            sum(Path(meta["path"]).stat().st_size for meta in dict(pack["feature_channels"]).values())
        ),
        "pack_will_be_copied": False,
        "qdp_will_be_updated": False,
        "supplement_estimated_bytes": int(pack["date_count"] * pack["symbol_count"] * 4 * 2),
        "fold_views": source_views,
        "tasks": [f"{profile}:{year}" for profile in PROFILE_ORDER for year in DEVELOPMENT_YEARS],
    }
    if dry_run:
        return result
    _assert_no_other_research_process()
    study_dir = study_root.resolve()
    study_path = study_dir / "study.json"
    if study_path.is_file():
        existing = _read_json(study_path)
        if str(existing.get("study_id", "")) != STUDY_ID:
            raise ValueError(f"refusing to reuse unrelated study root: {study_dir}")
        return existing

    supplement = build_relative_turnover_supplement()
    probe = _probe_cuda_batch_size()
    batch_size = int(probe["selected_batch_size"])
    contract_payload = contract(batch_size=batch_size)
    study = {
        "schema_version": 1,
        "artifact_type": "seq100_current_study",
        "study_id": STUDY_ID,
        "contract": contract_payload["contract"],
        "contract_sha256": contract_payload["contract_sha256"],
        "runtime": {"status": "preparing", "created_at": _now(), "updated_at": _now()},
        "material": {},
        "cuda_probe": probe,
    }
    _write_json(study_path, study)
    binding = {
        "contract_id": STUDY_ID,
        "contract_sha256": contract_payload["contract_sha256"],
        "contract_file_sha256": contract_payload["contract_sha256"],
        "path": str(study_path),
    }
    supplement_binding = {
        "manifest_path": str((SUPPLEMENT_ROOT / "manifest.json").resolve()),
        "manifest_sha256": _file_sha256(SUPPLEMENT_ROOT / "manifest.json"),
        "formula": supplement["formula"],
        "activity_target": supplement["activity_target"],
        "log_turnover_pct": supplement["log_turnover_pct"],
        "past20_positive_median": supplement["past20_positive_median"],
    }
    views: dict[str, str] = {}
    for year in DEVELOPMENT_YEARS:
        source_view = Path(str(source_views[str(year)])).resolve()
        view = _read_json(source_view)
        view["research_contract"] = dict(binding)
        view["development_contract"] = dict(binding)
        view["relative_turnover_supplement"] = dict(supplement_binding)
        view["legal_exit_contract"] = dict(LEGAL_EXIT_CONTRACT)
        view["artifact_view"] = f"{STUDY_ID}_development_{year}"
        view["development_fold_training_contract"] = _compute_development_fold_training_contract(
            view
        )
        output = study_dir / "views" / f"development_{year}.json"
        _write_json(output, view)
        views[str(year)] = str(output.resolve())
    study["material"] = {
        "pack_manifest": str(PACK_MANIFEST.resolve()),
        "source_study": str(SOURCE_STUDY_ROOT.resolve()),
        "fold_views": views,
        "relative_turnover_supplement": str((SUPPLEMENT_ROOT / "manifest.json").resolve()),
        "source_pack_mutated": False,
    }
    study["runtime"] = {
        "status": "prepared",
        "completed_tasks": [],
        "created_at": study["runtime"]["created_at"],
        "updated_at": _now(),
    }
    _write_json(study_path, study)
    return study


def _profile_from_study(study: Mapping[str, Any], profile_name: str) -> TodayClosePathOnlyProfile:
    payload = dict(dict(study["contract"])["profiles"][profile_name])
    for key in ("store_view", "output_root", "run_tag"):
        payload.pop(key, None)
    return TodayClosePathOnlyProfile(**payload)


def _run_dirs(study_dir: Path, profile_name: str, year: int) -> tuple[list[Path], list[Path]]:
    root = study_dir / "runs" / "development"
    matches = sorted(root.glob(f"seq100_structured_{profile_name}_{year}_seed{SEED}_*"))
    complete = [item for item in matches if (item / "sequence_path_training_summary.json").is_file()]
    partial = [item for item in matches if item not in complete]
    return complete, partial


def run(*, study_root: Path = STUDY_ROOT, max_tasks: int = 0) -> dict[str, Any]:
    _assert_no_other_research_process()
    study_path = study_root.resolve() / "study.json"
    study = _read_json(study_path)
    completed_tasks: list[str] = []
    launched = 0
    _update_runtime(study_path, study, status="running", error=None)
    try:
        views = dict(dict(study["material"])["fold_views"])
        for profile_name in PROFILE_ORDER:
            for year in DEVELOPMENT_YEARS:
                task = f"{profile_name}:{year}"
                complete, partial = _run_dirs(study_path.parent, profile_name, year)
                if len(complete) > 1:
                    raise ValueError(f"task {task} has multiple completed runs")
                if complete:
                    summary = _validate_run_summary(
                        complete[0] / "sequence_path_training_summary.json", year
                    )
                    if str(dict(summary["resolved_training_config"])["model_type"]) != (
                        "gru_ohlcva_aux_path_value"
                        if profile_name == "legal_flat_baseline"
                        else "gru_structured_joint_turnover"
                    ):
                        raise ValueError(f"task {task} completed with the wrong model")
                    completed_tasks.append(task)
                    continue
                for path in partial:
                    _remove_path(path)
                if max_tasks > 0 and launched >= int(max_tasks):
                    break
                base = _profile_from_study(study, profile_name)
                profile = replace(
                    base,
                    store_view=Path(views[str(year)]),
                    output_root=study_path.parent / "runs" / "development",
                    run_tag=f"seq100_structured_{profile_name}_{year}_seed{SEED}",
                    epochs=10,
                    seed=SEED,
                    top_k="1,3,5,10",
                    prediction_mode="compact",
                    evaluation_mode="development",
                    early_stopping_patience=2,
                    early_stopping_min_delta=0.0,
                    early_stopping_metric=training.EARLY_STOPPING_METRIC_DEVELOPMENT_PRICE_TOTAL_LOSS,
                    early_stopping_mode="min",
                    min_complete_epochs=1,
                    max_samples_per_split=0,
                )
                command = [
                    str(PYTHON),
                    "-m",
                    "daily_research.path_policy.qdp_v2_sequence_path_training",
                    *build_todayclose_path_only_train_argv(profile),
                    "--development-contract",
                    str(study_path),
                    "--json",
                ]
                _update_runtime(
                    study_path,
                    study,
                    status="running",
                    completed_tasks=completed_tasks,
                    current_task=task,
                )
                _guarded_command(
                    command,
                    log_path=study_path.parent / "logs" / f"memory_guard_{profile_name}_{year}.json",
                )
                complete, _partial = _run_dirs(study_path.parent, profile_name, year)
                if len(complete) != 1:
                    raise RuntimeError(f"task {task} did not produce exactly one completed run")
                _validate_run_summary(complete[0] / "sequence_path_training_summary.json", year)
                completed_tasks.append(task)
                launched += 1
                _update_runtime(
                    study_path,
                    study,
                    status="running",
                    completed_tasks=completed_tasks,
                    current_task=None,
                )
            if max_tasks > 0 and launched >= int(max_tasks):
                break
        final_status = "runs_completed" if len(completed_tasks) == 6 else "running"
        _update_runtime(
            study_path,
            study,
            status=final_status,
            completed_tasks=completed_tasks,
            current_task=None,
        )
    except Exception as exc:
        _update_runtime(
            study_path,
            study,
            status="run_failed",
            completed_tasks=completed_tasks,
            error=str(exc),
        )
        raise
    return status(study_root=study_root)


def status(*, study_root: Path = STUDY_ROOT) -> dict[str, Any]:
    study_path = study_root.resolve() / "study.json"
    study = _read_json(study_path)
    tasks: dict[str, Any] = {}
    for profile in PROFILE_ORDER:
        for year in DEVELOPMENT_YEARS:
            complete, partial = _run_dirs(study_path.parent, profile, year)
            progress = None
            if partial:
                progress_path = partial[-1] / "progress.json"
                if progress_path.is_file():
                    progress = _read_json(progress_path)
            tasks[f"{profile}:{year}"] = {
                "completed_run": str(complete[0]) if len(complete) == 1 else None,
                "partial_runs": [str(item) for item in partial],
                "progress": progress,
            }
    return {
        "study": str(study_path),
        "runtime": dict(study.get("runtime", {}) or {}),
        "cuda_probe": study.get("cuda_probe"),
        "tasks": tasks,
    }


def _load_checkpoint_model(
    run_dir: Path,
    dataset: training.SequencePathPackDataset,
    *,
    device: torch.device,
) -> tuple[training.SequencePathModel, dict[str, Any]]:
    summary = _read_json(run_dir / "sequence_path_training_summary.json")
    checkpoint = torch.load(Path(summary["best_checkpoint"]), map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("resolved_training_config", checkpoint.get("config", {})) or {})
    model = training.SequencePathModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(config.get("hidden_dim", 128)),
        layers=int(config.get("layers", 2)),
        forward_days=int(dataset.forward_days),
        summary_dim=len(dataset.path_summary_columns),
        dropout=float(config.get("dropout", 0.1)),
        model_type=str(config.get("model_type", "gru_ohlcva_aux_path_value")),
        symbol_count=int(dataset.symbol_count),
        symbol_embedding_dim=int(config.get("symbol_embedding_dim", 16)),
        richer_path_dim=int(dataset.richer_path_dim),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, summary


def _candidate_utility_curve_numpy(
    path: np.ndarray,
    *,
    price_anchor: str,
    earliest_exit_day: int,
    tradable: np.ndarray | None = None,
) -> np.ndarray:
    values = training._legacy_entry_relative_path_numpy(
        np.asarray(path, dtype=np.float32), price_anchor=price_anchor
    ).astype(np.float64, copy=False)
    close = values[:, :, 3]
    low = values[:, :, 2]
    drawdown = np.maximum(-np.minimum.accumulate(low, axis=1), 0.0)
    day = np.arange(values.shape[1], dtype=np.float64)
    candidate = (
        close
        - training.PATH_VALUE_V2_DRAWDOWN_PENALTY * drawdown
        - training.PATH_VALUE_V2_WAITING_PENALTY
        * np.sqrt((day + 1.0) / float(values.shape[1])).reshape(1, -1)
        - training.PATH_VALUE_V2_TRANSACTION_COST
    )
    candidate[:, : max(int(earliest_exit_day) - 1, 0)] = -np.inf
    if tradable is not None:
        candidate = np.where(np.asarray(tradable, dtype=bool), candidate, -np.inf)
    return candidate


def _entropy_from_counts(counts: Mapping[int, int]) -> float:
    values = np.asarray([int(value) for value in counts.values() if int(value) > 0], dtype=np.float64)
    if values.size == 0:
        return float("nan")
    probability = values / values.sum()
    return float(-(probability * np.log(probability)).sum())


def _increment_counts(target: dict[int, int], values: np.ndarray) -> None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)].astype(np.int64)
    keys, counts = np.unique(finite, return_counts=True)
    for key, count in zip(keys.tolist(), counts.tolist(), strict=True):
        target[int(key)] = int(target.get(int(key), 0) + int(count))


def _finite_corr(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return float("nan")
    x = x[finite]
    y = y[finite]
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _append_path_rows(
    rows: list[dict[str, Any]],
    *,
    trade_date: str,
    symbols: Sequence[str],
    indices: Sequence[int],
    category: str,
    pred_path: np.ndarray,
    true_path: np.ndarray,
    pred_activity: np.ndarray | None,
    true_activity: np.ndarray | None,
) -> None:
    for rank, idx in enumerate(indices, start=1):
        for day in range(int(pred_path.shape[1])):
            row: dict[str, Any] = {
                "trade_date": trade_date,
                "symbol": str(symbols[int(idx)]),
                "category": category,
                "selection_rank": int(rank),
                "future_day": int(day + 1),
            }
            for field_idx, field in enumerate(("open", "high", "low", "close")):
                row[f"pred_{field}"] = float(pred_path[int(idx), day, field_idx])
                value = float(true_path[int(idx), day, field_idx])
                row[f"true_{field}"] = value if math.isfinite(value) else None
            if pred_activity is not None:
                row["pred_activity"] = float(pred_activity[int(idx), day])
            if true_activity is not None:
                value = float(true_activity[int(idx), day])
                row["true_activity"] = value if math.isfinite(value) else None
            rows.append(row)


def stream_checkpoint_diagnostics(
    *,
    run_dir: Path,
    view_path: Path,
    output_dir: Path,
    year: int,
    compare_legacy_domain: bool,
    fixed_exit_comparison: bool,
) -> dict[str, Any]:
    """Re-infer one fold without ever writing a full-universe path prediction file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"checkpoint_path_diagnostics_{year}.json"
    if result_path.is_file():
        return _read_json(result_path)
    manifest = _read_json(view_path)
    dataset = training.SequencePathPackDataset(
        manifest,
        split="development",
        max_samples=0,
        input_channel_profile=training.INPUT_CHANNEL_PROFILE_DAILY_ONLY,
        index_role="candidate",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, run_summary = _load_checkpoint_model(run_dir, dataset, device=device)
    uses_activity = bool(getattr(model, "uses_structured_turnover", False))
    date_values = np.asarray(manifest["date_values"], dtype=object)
    groups = dataset.sample_index.groupby("date_idx", sort=True).indices
    raw_exit_counts: dict[int, int] = {}
    legal_exit_counts: dict[int, int] = {}
    top3_exit_counts: dict[int, int] = {}
    actual_exit_counts: dict[int, int] = {}
    d1_count = 0
    candidate_count = 0
    changed_after_legal_search = 0
    clamp_changed_count = 0
    tie_count = 0
    earliest_tie_count = 0
    margins: list[np.ndarray] = []
    rank_ic_old: list[float] = []
    rank_ic_legal: list[float] = []
    top3_overlap: list[float] = []
    opportunity_alphas: list[float] = []
    executable_alphas: list[float] = []
    oracle_executable_alphas: list[float] = []
    exact_raw_regret_sum = 0.0
    exact_raw_regret_count = 0
    utility_abs_sum = 0.0
    utility_count = 0
    utility_pred_values: list[np.ndarray] = []
    utility_true_values: list[np.ndarray] = []
    regret_sum = 0.0
    regret_count = 0
    bucket_edges = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 60))
    bucket_abs_sum = {f"D{start}_{end}": 0.0 for start, end in bucket_edges}
    bucket_count = {f"D{start}_{end}": 0 for start, end in bucket_edges}
    close_abs_sum = 0.0
    close_count = 0
    direction_correct = 0
    direction_count = 0
    path_correlations: list[float] = []
    geometry_violations = 0
    activity_level_sum = 0.0
    activity_level_count = 0
    activity_delta_sum = 0.0
    activity_delta_count = 0
    fixed_daily: dict[str, list[float]] = {
        f"D{day}": [] for day in (2, 5, 10, 20, 30, 40, 60)
    }
    fixed_daily["predicted"] = []
    fixed_daily["oracle_executable"] = []
    path_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED + int(year))
    progress_path = output_dir / f"checkpoint_path_diagnostics_{year}.progress.json"

    try:
        with torch.no_grad():
            for date_number, (date_idx_raw, positions_raw) in enumerate(groups.items(), start=1):
                if psutil.virtual_memory().available < int(0.5 * 1024**3):
                    raise MemoryError("available physical memory fell below 0.5 GiB")
                indices = np.asarray(positions_raw, dtype=np.int64)
                pred_parts: list[np.ndarray] = []
                true_parts: list[np.ndarray] = []
                tradable_parts: list[np.ndarray] = []
                entry_filled_parts: list[np.ndarray] = []
                entry_open_parts: list[np.ndarray] = []
                exit_close_parts: list[np.ndarray] = []
                exit_sellable_parts: list[np.ndarray] = []
                activity_pred_parts: list[np.ndarray] = []
                activity_true_parts: list[np.ndarray] = []
                symbols: list[str] = []
                for start in range(0, int(indices.size), 512):
                    chunk = indices[start : start + 512]
                    batch = dataset.get_batch(
                        chunk,
                        include_ohlcva_path=False,
                        include_richer_path=False,
                        include_summary=False,
                        include_activity_path=uses_activity,
                    )
                    x = batch["x"].to(device, non_blocking=device.type == "cuda")
                    symbol_idx = batch["symbol_idx"].to(
                        device, non_blocking=device.type == "cuda"
                    )
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        output = model(x, symbol_idx=symbol_idx)
                    pred_parts.append(output["future_path"].float().cpu().numpy())
                    true_parts.append(batch["y_path"].numpy())
                    tradable_parts.append(batch["y_tradable_path"].numpy().astype(bool))
                    entry_filled_parts.append(batch["entry_filled"].numpy().astype(bool))
                    entry_open_parts.append(batch["entry_open_raw"].numpy())
                    exit_close_parts.append(batch["exit_close_raw_path"].numpy())
                    exit_sellable_parts.append(batch["exit_sellable_path"].numpy().astype(bool))
                    if uses_activity:
                        activity_pred_parts.append(
                            output["future_activity_path"].float().cpu().numpy()
                        )
                        activity_true_parts.append(batch["y_activity_path"].numpy())
                    symbols.extend([str(item) for item in batch["symbol"]])
                    del batch, x, symbol_idx, output
                pred_path = np.concatenate(pred_parts, axis=0)
                true_path = np.concatenate(true_parts, axis=0)
                tradable = np.concatenate(tradable_parts, axis=0)
                entry_filled = np.concatenate(entry_filled_parts, axis=0)
                entry_open = np.concatenate(entry_open_parts, axis=0)
                exit_close = np.concatenate(exit_close_parts, axis=0)
                exit_sellable = np.concatenate(exit_sellable_parts, axis=0)
                pred_activity = (
                    np.concatenate(activity_pred_parts, axis=0) if activity_pred_parts else None
                )
                true_activity = (
                    np.concatenate(activity_true_parts, axis=0) if activity_true_parts else None
                )
                trade_date = str(dataset.trade_date_values[indices[0]])
                signal_date_idx = np.full(len(indices), int(date_idx_raw), dtype=np.int64)

                old_summary = training._derive_path_summary_numpy(
                    pred_path, price_anchor=dataset.price_anchor, earliest_exit_day=1
                )
                legal_summary = training._derive_path_summary_numpy(
                    pred_path, price_anchor=dataset.price_anchor, earliest_exit_day=2
                )
                true_summary = training._derive_path_summary_numpy(
                    true_path,
                    price_anchor=dataset.price_anchor,
                    tradable_path=tradable,
                    earliest_exit_day=2,
                )
                old_day = old_summary[:, 8]
                legal_day = legal_summary[:, 8]
                old_score = old_summary[:, -1]
                legal_score = legal_summary[:, -1]
                true_score = true_summary[:, -1]
                candidate_count += int(len(indices))
                d1_count += int((old_day == 1).sum())
                clamp_day = np.maximum(old_day, 2.0)
                clamp_changed_count += int(np.sum(np.isfinite(old_day) & (old_day != clamp_day)))
                changed_after_legal_search += int(
                    np.sum(np.isfinite(legal_day) & np.isfinite(clamp_day) & (legal_day != clamp_day))
                )
                _increment_counts(raw_exit_counts, old_day)
                _increment_counts(legal_exit_counts, legal_day)

                predicted_curve = _candidate_utility_curve_numpy(
                    pred_path, price_anchor=dataset.price_anchor, earliest_exit_day=2
                )
                true_curve = _candidate_utility_curve_numpy(
                    true_path,
                    price_anchor=dataset.price_anchor,
                    earliest_exit_day=2,
                    tradable=tradable,
                )
                legal_curve = predicted_curve[:, 1:]
                sorted_curve = np.sort(legal_curve, axis=1)
                margin = sorted_curve[:, -1] - sorted_curve[:, -2]
                finite_margin = margin[np.isfinite(margin)]
                margins.append(finite_margin.astype(np.float32))
                best = sorted_curve[:, -1]
                ties = np.isclose(legal_curve, best.reshape(-1, 1), atol=1.0e-8, rtol=0.0)
                tie_count += int((ties.sum(axis=1) > 1).sum())
                earliest_tie_count += int(
                    np.sum((ties.sum(axis=1) > 1) & (np.argmax(ties, axis=1) + 2 == legal_day))
                )
                curve_mask = np.isfinite(predicted_curve) & np.isfinite(true_curve)
                utility_abs_sum += float(
                    np.abs(predicted_curve[curve_mask] - true_curve[curve_mask]).sum()
                )
                utility_count += int(curve_mask.sum())
                if int(curve_mask.sum()):
                    utility_pred_values.append(predicted_curve[curve_mask].astype(np.float32))
                    utility_true_values.append(true_curve[curve_mask].astype(np.float32))
                planned_index = np.clip(legal_day.astype(np.int64) - 1, 0, 59)
                row_index = np.arange(len(indices))
                realized_at_plan = true_curve[row_index, planned_index]
                true_best = np.max(true_curve, axis=1)
                regret = true_best - realized_at_plan
                finite_regret = np.isfinite(regret)
                regret_sum += float(regret[finite_regret].sum())
                regret_count += int(finite_regret.sum())

                rank_ic_old.append(
                    float(pd.Series(old_score).corr(pd.Series(true_score), method="spearman"))
                )
                rank_ic_legal.append(
                    float(pd.Series(legal_score).corr(pd.Series(true_score), method="spearman"))
                )
                top3_old = np.argsort(-old_score, kind="stable")[:3]
                top3 = np.argsort(-legal_score, kind="stable")[:3]
                top3_overlap.append(float(len(set(top3_old.tolist()) & set(top3.tolist())) / 3.0))
                _increment_counts(top3_exit_counts, legal_day[top3])
                opportunity_alphas.append(
                    float(np.nanmean(true_score[top3]) - np.nanmean(true_score))
                )

                execution_frame = pd.DataFrame(
                    {
                        "trade_date": trade_date,
                        "symbol": symbols,
                        "score": legal_score,
                        "predicted_exit_day": legal_day,
                        "entry_filled": entry_filled,
                        "entry_open_raw": entry_open,
                    }
                )
                execution = evaluate_candidate_execution(
                    execution_frame,
                    exit_close,
                    exit_sellable,
                    manifest=manifest,
                    signal_date_idx=signal_date_idx,
                    date_values=date_values,
                    top_k_values=(3,),
                )
                daily_row = execution.daily_topk.iloc[0]
                executable_alphas.append(
                    float(daily_row["alpha_net_realized_plan_return_base"])
                )
                _increment_counts(
                    actual_exit_counts,
                    pd.to_numeric(
                        execution.candidates["realized_plan_resolved_exit_day"], errors="coerce"
                    ).to_numpy(dtype=np.float64),
                )
                fixed_daily["predicted"].append(executable_alphas[-1])

                oracle_frame = execution_frame.copy()
                oracle_frame["predicted_exit_day"] = true_summary[:, 8]
                oracle_execution = evaluate_candidate_execution(
                    oracle_frame,
                    exit_close,
                    exit_sellable,
                    manifest=manifest,
                    signal_date_idx=signal_date_idx,
                    date_values=date_values,
                    top_k_values=(3,),
                )
                oracle_alpha = float(
                    oracle_execution.daily_topk.iloc[0]["alpha_net_realized_plan_return_base"]
                )
                oracle_executable_alphas.append(oracle_alpha)
                fixed_daily["oracle_executable"].append(oracle_alpha)
                legal_net = pd.to_numeric(
                    execution.candidates["net_realized_plan_return_base"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                oracle_net = pd.to_numeric(
                    oracle_execution.candidates["net_realized_plan_return_base"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                net_mask = np.isfinite(legal_net) & np.isfinite(oracle_net)
                exact_raw_regret_sum += float((oracle_net[net_mask] - legal_net[net_mask]).sum())
                exact_raw_regret_count += int(net_mask.sum())
                if fixed_exit_comparison:
                    for day in (2, 5, 10, 20, 30, 40, 60):
                        fixed_frame = execution_frame.copy()
                        fixed_frame["predicted_exit_day"] = float(day)
                        fixed_execution = evaluate_candidate_execution(
                            fixed_frame,
                            exit_close,
                            exit_sellable,
                            manifest=manifest,
                            signal_date_idx=signal_date_idx,
                            date_values=date_values,
                            top_k_values=(3,),
                        )
                        fixed_daily[f"D{day}"].append(
                            float(
                                fixed_execution.daily_topk.iloc[0][
                                    "alpha_net_realized_plan_return_base"
                                ]
                            )
                        )

                finite_path = np.isfinite(pred_path) & np.isfinite(true_path)
                absolute = np.abs(pred_path - true_path)
                for start_day, end_day in bucket_edges:
                    key = f"D{start_day}_{end_day}"
                    bucket_mask = finite_path[:, start_day - 1 : end_day, :]
                    bucket_abs_sum[key] += float(
                        absolute[:, start_day - 1 : end_day, :][bucket_mask].sum()
                    )
                    bucket_count[key] += int(bucket_mask.sum())
                close_mask = finite_path[:, :, 3]
                close_abs_sum += float(absolute[:, :, 3][close_mask].sum())
                close_count += int(close_mask.sum())
                direction_correct += int(
                    (
                        np.sign(pred_path[:, :, 3][close_mask])
                        == np.sign(true_path[:, :, 3][close_mask])
                    ).sum()
                )
                direction_count += int(close_mask.sum())
                for row_idx in range(len(indices)):
                    corr = _finite_corr(pred_path[row_idx, :, 3], true_path[row_idx, :, 3])
                    if math.isfinite(corr):
                        path_correlations.append(corr)
                geometry_violations += int(
                    (
                        pred_path[:, :, 1]
                        < np.maximum(pred_path[:, :, 0], pred_path[:, :, 3])
                    ).sum()
                    + (
                        pred_path[:, :, 2]
                        > np.minimum(pred_path[:, :, 0], pred_path[:, :, 3])
                    ).sum()
                )
                if pred_activity is not None and true_activity is not None:
                    activity_mask = np.isfinite(pred_activity) & np.isfinite(true_activity)
                    difference = np.abs(pred_activity - true_activity)
                    activity_level_sum += float(difference[activity_mask].sum())
                    activity_level_count += int(activity_mask.sum())
                    pred_delta = np.diff(pred_activity, axis=1)
                    true_delta = np.diff(true_activity, axis=1)
                    delta_mask = np.isfinite(pred_delta) & np.isfinite(true_delta)
                    activity_delta_sum += float(
                        np.abs(pred_delta - true_delta)[delta_mask].sum()
                    )
                    activity_delta_count += int(delta_mask.sum())

                _append_path_rows(
                    path_rows,
                    trade_date=trade_date,
                    symbols=symbols,
                    indices=top3,
                    category="top3",
                    pred_path=pred_path,
                    true_path=true_path,
                    pred_activity=pred_activity,
                    true_activity=true_activity,
                )
                order = np.argsort(legal_score, kind="stable")
                strata = np.array_split(order, 3)
                sample_indices = [int(rng.choice(group)) for group in strata if len(group)]
                _append_path_rows(
                    path_rows,
                    trade_date=trade_date,
                    symbols=symbols,
                    indices=sample_indices,
                    category="seed7_score_tertile_sample",
                    pred_path=pred_path,
                    true_path=true_path,
                    pred_activity=pred_activity,
                    true_activity=true_activity,
                )
                if date_number == 1 or date_number % 25 == 0:
                    _write_json(
                        progress_path,
                        {
                            "status": "running",
                            "year": int(year),
                            "completed_dates": int(date_number),
                            "total_dates": int(len(groups)),
                            "updated_at": _now(),
                        },
                    )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    margin_values = np.concatenate(margins) if margins else np.asarray([], dtype=np.float32)
    utility_pred = (
        np.concatenate(utility_pred_values) if utility_pred_values else np.asarray([], dtype=np.float32)
    )
    utility_true = (
        np.concatenate(utility_true_values) if utility_true_values else np.asarray([], dtype=np.float32)
    )
    path_output = output_dir / f"checkpoint_path_samples_{year}.parquet"
    pd.DataFrame(path_rows).to_parquet(path_output, index=False)
    fixed_result = {
        key: float(np.mean(values)) if values else None for key, values in fixed_daily.items()
    }
    result = {
        "schema_version": 1,
        "artifact_type": "seq100_checkpoint_streaming_path_diagnostics",
        "created_at": _now(),
        "year": int(year),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(run_summary["best_checkpoint"]),
        "candidate_count": int(candidate_count),
        "date_count": int(len(groups)),
        "legal_exit_contract": dict(LEGAL_EXIT_CONTRACT),
        "compare_legacy_domain": bool(compare_legacy_domain),
        "legacy_d1_selected_rate": float(d1_count / max(candidate_count, 1)),
        "t1_clamp_changed_rate": float(clamp_changed_count / max(candidate_count, 1)),
        "legal_research_differs_from_simple_clamp_rate": float(
            changed_after_legal_search / max(candidate_count, 1)
        ),
        "raw_exit_day_distribution": {str(key): value for key, value in sorted(raw_exit_counts.items())},
        "legal_exit_day_distribution": {
            str(key): value for key, value in sorted(legal_exit_counts.items())
        },
        "top3_legal_exit_day_distribution": {
            str(key): value for key, value in sorted(top3_exit_counts.items())
        },
        "actual_exit_day_distribution": {
            str(key): value for key, value in sorted(actual_exit_counts.items())
        },
        "legal_exit_day_entropy": _entropy_from_counts(legal_exit_counts),
        "top3_exit_day_entropy": _entropy_from_counts(top3_exit_counts),
        "legal_exit_max_day_share": float(
            max(legal_exit_counts.values(), default=0) / max(sum(legal_exit_counts.values()), 1)
        ),
        "tie_rate": float(tie_count / max(candidate_count, 1)),
        "earliest_tie_break_count": int(earliest_tie_count),
        "tie_margin": {
            "mean": float(np.mean(margin_values)),
            "p10": float(np.quantile(margin_values, 0.10)),
            "median": float(np.median(margin_values)),
            "p90": float(np.quantile(margin_values, 0.90)),
        },
        "old_score_rank_ic": float(np.nanmean(rank_ic_old)),
        "legal_score_rank_ic": float(np.nanmean(rank_ic_legal)),
        "old_legal_top3_overlap_rate": float(np.mean(top3_overlap)),
        "legal_opportunity_alpha": float(np.mean(opportunity_alphas)),
        "legal_executable_alpha": float(np.mean(executable_alphas)),
        "opportunity_execution_gap": float(
            np.mean(opportunity_alphas) - np.mean(executable_alphas)
        ),
        "oracle_executable_alpha": float(np.mean(oracle_executable_alphas)),
        "exit_regret": float(regret_sum / max(regret_count, 1)),
        "exact_raw_oracle_executable_regret": float(
            exact_raw_regret_sum / max(exact_raw_regret_count, 1)
        ),
        "utility_curve_mae": float(utility_abs_sum / max(utility_count, 1)),
        "utility_curve_correlation": _finite_corr(utility_pred, utility_true),
        "fixed_exit_top3_alpha": fixed_result,
        "ohlc_path_mae_by_horizon": {
            key: float(bucket_abs_sum[key] / max(bucket_count[key], 1))
            for key in bucket_abs_sum
        },
        "close_cumulative_path_mae": float(close_abs_sum / max(close_count, 1)),
        "close_direction_accuracy": float(direction_correct / max(direction_count, 1)),
        "predicted_true_close_path_correlation": float(np.mean(path_correlations)),
        "ohlc_geometry_violation_count": int(geometry_violations),
        "turnover_level_mae": (
            float(activity_level_sum / activity_level_count) if activity_level_count else None
        ),
        "turnover_delta_mae": (
            float(activity_delta_sum / activity_delta_count) if activity_delta_count else None
        ),
        "score_coverage": 1.0,
        "path_samples": str(path_output.resolve()),
        "full_prediction_written": False,
    }
    _write_json(result_path, result)
    if progress_path.exists():
        progress_path.unlink()
    return result


def diagnose(
    *,
    source_study_root: Path = SOURCE_STUDY_ROOT,
    output_root: Path = STUDY_ROOT / "diagnostics" / "legacy_checkpoints",
) -> dict[str, Any]:
    source_study = _read_json(source_study_root.resolve() / "study.json")
    views = dict(dict(source_study["material"])["fold_views"])
    yearly: list[dict[str, Any]] = []
    for year in DEVELOPMENT_YEARS:
        matches = sorted(
            (source_study_root / "runs" / "development").glob(
                f"seq100_development_baseline_{year}_seed{SEED}_current_*"
            )
        )
        complete = [
            item for item in matches if (item / "sequence_path_training_summary.json").is_file()
        ]
        if len(complete) != 1:
            raise ValueError(f"legacy diagnose requires exactly one completed run for {year}")
        yearly.append(
            stream_checkpoint_diagnostics(
                run_dir=complete[0],
                view_path=Path(views[str(year)]),
                output_dir=output_root.resolve(),
                year=year,
                compare_legacy_domain=True,
                fixed_exit_comparison=True,
            )
        )
    numeric_keys = (
        "legacy_d1_selected_rate",
        "t1_clamp_changed_rate",
        "legal_research_differs_from_simple_clamp_rate",
        "old_score_rank_ic",
        "legal_score_rank_ic",
        "old_legal_top3_overlap_rate",
        "legal_opportunity_alpha",
        "legal_executable_alpha",
        "opportunity_execution_gap",
        "oracle_executable_alpha",
        "exit_regret",
        "exact_raw_oracle_executable_regret",
        "utility_curve_mae",
        "utility_curve_correlation",
    )
    aggregate = {
        key: float(np.mean([float(row[key]) for row in yearly])) for key in numeric_keys
    }
    result = {
        "schema_version": 1,
        "status": "completed",
        "source_study": str(source_study_root.resolve()),
        "legal_exit_contract": dict(LEGAL_EXIT_CONTRACT),
        "yearly": yearly,
        "equal_year": aggregate,
        "full_prediction_written": False,
    }
    _write_json(output_root.resolve() / "legacy_checkpoint_diagnostic.json", result)
    return result


def summarize(*, study_root: Path = STUDY_ROOT) -> dict[str, Any]:
    study_path = study_root.resolve() / "study.json"
    study = _read_json(study_path)
    profiles: dict[str, Any] = {}
    views = dict(dict(study["material"])["fold_views"])
    for profile in PROFILE_ORDER:
        yearly: list[dict[str, Any]] = []
        runs: dict[str, Any] = {}
        for year in DEVELOPMENT_YEARS:
            complete, _partial = _run_dirs(study_path.parent, profile, year)
            if len(complete) != 1:
                raise ValueError(f"summarize requires completed task {profile}:{year}")
            metrics, summary = _year_metrics(complete[0], year)
            path_diagnostics = stream_checkpoint_diagnostics(
                run_dir=complete[0],
                view_path=Path(views[str(year)]),
                output_dir=complete[0] / "path_diagnostics",
                year=year,
                compare_legacy_domain=False,
                fixed_exit_comparison=False,
            )
            metrics.update(
                {
                    "exit_regret": float(path_diagnostics["exit_regret"]),
                    "exact_raw_oracle_executable_regret": float(
                        path_diagnostics["exact_raw_oracle_executable_regret"]
                    ),
                    "utility_curve_mae": float(path_diagnostics["utility_curve_mae"]),
                    "utility_curve_correlation": float(
                        path_diagnostics["utility_curve_correlation"]
                    ),
                    "close_cumulative_path_mae": float(
                        path_diagnostics["close_cumulative_path_mae"]
                    ),
                    "close_direction_accuracy": float(
                        path_diagnostics["close_direction_accuracy"]
                    ),
                    "predicted_true_close_path_correlation": float(
                        path_diagnostics["predicted_true_close_path_correlation"]
                    ),
                    "legal_exit_day_entropy": float(
                        path_diagnostics["legal_exit_day_entropy"]
                    ),
                    "legal_exit_max_day_share": float(
                        path_diagnostics["legal_exit_max_day_share"]
                    ),
                    "ohlc_geometry_violation_count": int(
                        path_diagnostics["ohlc_geometry_violation_count"]
                    ),
                    "turnover_level_mae": path_diagnostics["turnover_level_mae"],
                    "turnover_delta_mae": path_diagnostics["turnover_delta_mae"],
                }
            )
            yearly.append(metrics)
            runs[str(year)] = {
                "run_dir": str(complete[0]),
                "summary": str(complete[0] / "sequence_path_training_summary.json"),
                "best_epoch": int(summary["best_epoch"]),
                "path_diagnostics": str(
                    complete[0]
                    / "path_diagnostics"
                    / f"checkpoint_path_diagnostics_{year}.json"
                ),
            }
        numeric = sorted(
            key
            for key in yearly[0]
            if key not in {
                "development_year",
                "best_epoch",
                "completed_epochs",
                "turnover_level_mae",
                "turnover_delta_mae",
            }
        )
        equal = {key: float(np.mean([float(row[key]) for row in yearly])) for key in numeric}
        equal["positive_top3_years"] = int(
            sum(float(row["top3_base_alpha"]) > 0.0 for row in yearly)
        )
        equal["worst_year_top3_base_alpha"] = float(
            min(float(row["top3_base_alpha"]) for row in yearly)
        )
        profiles[profile] = {
            "yearly_metrics": yearly,
            "equal_year_metrics": equal,
            "best_epoch_median": int(median([int(row["best_epoch"]) for row in yearly])),
            "runs": runs,
        }
        for optional in ("turnover_level_mae", "turnover_delta_mae"):
            values = [row[optional] for row in yearly if row.get(optional) is not None]
            profiles[profile]["equal_year_metrics"][optional] = (
                float(np.mean(values)) if values else None
            )
    base = dict(profiles["legal_flat_baseline"]["equal_year_metrics"])
    structured = dict(profiles["structured_joint_turnover"]["equal_year_metrics"])
    comparison = {
        "equal_year_top3_alpha_delta": float(
            structured["top3_base_alpha"] - base["top3_base_alpha"]
        ),
        "structured_three_positive_years": bool(structured["positive_top3_years"] == 3),
        "structured_worst_year_not_lower": bool(
            structured["worst_year_top3_base_alpha"]
            >= base["worst_year_top3_base_alpha"]
        ),
        "automatic_winner": None,
        "attribution": "combined_effect_only",
    }
    comparison.update(
        {
            "equal_year_exit_regret_lower": bool(
                structured["exit_regret"] < base["exit_regret"]
            ),
            "close_path_nonworse_year_count": int(
                sum(
                    float(structured_row["close_cumulative_path_mae"])
                    <= float(base_row["close_cumulative_path_mae"])
                    for structured_row, base_row in zip(
                        profiles["structured_joint_turnover"]["yearly_metrics"],
                        profiles["legal_flat_baseline"]["yearly_metrics"],
                        strict=True,
                    )
                )
            ),
            "structured_geometry_violation_count": int(
                sum(
                    int(row["ohlc_geometry_violation_count"])
                    for row in profiles["structured_joint_turnover"]["yearly_metrics"]
                )
            ),
            "new_exit_day_collapse": bool(
                structured["legal_exit_max_day_share"]
                > max(0.50, base["legal_exit_max_day_share"] + 0.05)
            ),
        }
    )
    comparison["consistent_improvement_evidence"] = bool(
        comparison["equal_year_top3_alpha_delta"] > 0.0
        and comparison["structured_three_positive_years"]
        and comparison["structured_worst_year_not_lower"]
        and comparison["equal_year_exit_regret_lower"]
        and comparison["close_path_nonworse_year_count"] >= 2
        and not comparison["new_exit_day_collapse"]
        and comparison["structured_geometry_violation_count"] == 0
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "study_id": STUDY_ID,
        "seed": SEED,
        "development_years": list(DEVELOPMENT_YEARS),
        "profiles": profiles,
        "comparison": comparison,
        "final_fit_performed": False,
        "winner": None,
        "qdp_changed": False,
        "source_pack_changed": False,
        "deployment_changed": False,
    }
    summary_json = study_path.parent / "research_summary.json"
    _write_json(summary_json, result)
    lines = [
        "# Seq100 Legal Exit and Structured Path Comparison",
        "",
        "- Report only; no final fit, QDP update, pack mutation, or deployment change.",
        "- All ranking and planned exits use the legal D2-D60 price-path domain.",
        "",
        "| profile | year | best epoch | Top3 alpha | Top3 absolute | universe | rank IC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILE_ORDER:
        for row in profiles[profile]["yearly_metrics"]:
            lines.append(
                f"| {profile} | {row['development_year']} | {row['best_epoch']} | "
                f"{row['top3_base_alpha']:.4%} | {row['top3_selected_base']:.4%} | "
                f"{row['top3_universe_base']:.4%} | {row['rank_ic']:.6f} |"
            )
    lines.extend(
        [
            "",
            f"Equal-year Top3 alpha delta (structured - legal flat): "
            f"`{comparison['equal_year_top3_alpha_delta']:.4%}`.",
            "",
            "No automatic winner is declared. The structured result is a combined architecture effect and cannot be attributed to one component.",
        ]
    )
    (study_path.parent / "research_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _update_runtime(
        study_path,
        study,
        status="completed",
        research_summary_json=str(summary_json),
        research_summary_md=str(study_path.parent / "research_summary.md"),
        error=None,
    )
    return result
