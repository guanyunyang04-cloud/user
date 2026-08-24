"""Orchestration for targeted repair from purchased local minute Parquet files."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.core.io import sha256_file, stable_hash, write_json
from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.minute_archive.quality import (
    PRICE_ABSOLUTE_TOLERANCE,
    PRICE_COMPARISON_EPSILON,
)

from .candidate import (
    DEFAULT_PRIORITY_RELATIVE_ERROR,
    PRICE_COLUMNS,
    MinuteRepairError,
    TargetBatch,
    load_priority_targets,
    write_candidate_artifacts,
)
from .install import install_accepted_changes, resume_interrupted_install
from .local_source import (
    LOCAL_PROVIDER_ID,
    LOCAL_REPAIR_REASON,
    capture_local_candidate_batches,
    evaluate_local_candidates,
    local_source_inventory,
    prefilter_local_candidates,
    prepare_local_aggregate_evidence,
    validate_extreme_timing_sample,
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _field_counts(decisions: pd.DataFrame) -> dict[str, int]:
    counts = {field: 0 for field in PRICE_COLUMNS}
    accepted = decisions.loc[decisions["status"].eq("accepted")]
    for value in accepted["repaired_fields"].astype(str):
        for field in filter(None, value.split(",")):
            counts[field] += 1
    return counts


def _reason_counts(decisions: pd.DataFrame) -> dict[str, int]:
    rejected = decisions.loc[decisions["status"].eq("rejected")]
    return {str(key): int(value) for key, value in rejected["reason"].value_counts().items()}


def _artifact_hashes(paths: dict[str, str]) -> dict[str, str]:
    return {key: sha256_file(value) for key, value in paths.items() if Path(value).is_file()}


def _normalize_excluded_ranges(
    values: tuple[tuple[str, str], ...] | list[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for start, end in values or ():
        start_date = pd.Timestamp(str(start)).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(str(end)).strftime("%Y-%m-%d")
        if start_date > end_date:
            raise ValueError(f"minute_repair_excluded_date_range_invalid:{start}:{end}")
        result.append((start_date, end_date))
    return tuple(sorted(set(result)))


def _split_excluded_ranges(
    targets: pd.DataFrame,
    ranges: tuple[tuple[str, str], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded = pd.Series(False, index=targets.index)
    dates = targets["trade_date"].astype(str)
    for start, end in ranges:
        excluded |= dates.between(start, end)
    return (
        targets.loc[~excluded].copy().reset_index(drop=True),
        targets.loc[excluded].copy().reset_index(drop=True),
    )


def _restrict_fields(targets: pd.DataFrame, selection_mode: str) -> pd.DataFrame:
    selected = targets.copy()
    mode = str(selection_mode).strip().lower()
    if mode == "open":
        for field in ("high", "low", "close"):
            selected[f"exclude_{field}"] = False
    elif mode == "high-low":
        selected["exclude_open"] = False
        selected["exclude_close"] = False
    else:
        raise ValueError(f"minute_repair_local_selection_mode_invalid:{selection_mode}")
    return selected


def _selection_id(
    targets: pd.DataFrame,
    *,
    selection_mode: str,
    minimum_relative_error: float,
    maximum_relative_error: float | None,
    excluded_ranges: tuple[tuple[str, str], ...],
    inventory_fingerprint: str,
) -> str:
    records = (
        targets.loc[
            :,
            [
                "symbol",
                "trade_date",
                "exclude_open",
                "exclude_high",
                "exclude_low",
                "exclude_close",
                "selection_reason",
            ],
        ]
        .sort_values(["symbol", "trade_date"], kind="stable")
        .to_dict("records")
    )
    digest = stable_hash(
        {
            "schema": "quantlab.local_parquet_minute_repair_selection/v1",
            "selection_mode": str(selection_mode),
            "minimum_relative_error": float(minimum_relative_error),
            "maximum_relative_error": maximum_relative_error,
            "excluded_ranges": excluded_ranges,
            "inventory_fingerprint": inventory_fingerprint,
            "targets": records,
        }
    )
    return f"{str(selection_mode).strip().lower().replace('-', '_')}_{digest[:16]}"


def _write_fallback_artifacts(
    run_dir: Path,
    decisions: pd.DataFrame,
    range_deferred: pd.DataFrame,
) -> dict[str, str]:
    fallback_path = run_dir / "tushare_fallback_targets.parquet"
    rejected = decisions.loc[
        decisions["status"].eq("rejected"),
        ["symbol", "trade_date", "reason", "selection_reason"],
    ].copy()
    rejected["fallback_provider"] = "tushare_compatible_stk_mins"
    rejected.to_parquet(fallback_path, index=False, engine="pyarrow", compression="zstd")
    deferred_path = run_dir / "date_range_deferred_targets.parquet"
    if range_deferred.empty:
        pd.DataFrame(
            columns=["symbol", "trade_date", "selection_reason", "fallback_provider", "reason"]
        ).to_parquet(deferred_path, index=False, engine="pyarrow", compression="zstd")
    else:
        deferred = range_deferred.loc[:, ["symbol", "trade_date", "selection_reason"]].copy()
        deferred["fallback_provider"] = "tushare_compatible_stk_mins"
        deferred["reason"] = "local_source_low_value_date_range_deferred"
        deferred.to_parquet(deferred_path, index=False, engine="pyarrow", compression="zstd")
    return {
        "tushare_fallback_targets": str(fallback_path),
        "date_range_deferred_targets": str(deferred_path),
    }


def _result_payload(
    *,
    status: str,
    workspace: Path,
    run_dir: Path,
    source_root: Path,
    selection_mode: str,
    minimum_relative_error: float,
    maximum_relative_error: float | None,
    excluded_ranges: tuple[tuple[str, str], ...],
    targets: pd.DataFrame,
    range_deferred: pd.DataFrame,
    aggregate_metadata: dict[str, Any],
    prefilter: pd.DataFrame,
    capture: dict[str, Any],
    decisions: pd.DataFrame,
    changes: pd.DataFrame,
    timing: dict[str, Any],
    artifacts: dict[str, str],
    installation: dict[str, Any],
) -> dict[str, Any]:
    accepted = decisions.loc[decisions["status"].eq("accepted")]
    return {
        "schema": "quantlab.local_parquet_minute_repair_result/v1",
        "status": status,
        "created_at": _utc_now(),
        "workspace": str(workspace),
        "run_dir": str(run_dir),
        "source": {
            "provider": LOCAL_PROVIDER_ID,
            "root": str(source_root),
            "canonical_base_files_only": True,
            "duplicate_parenthesized_files_excluded": True,
            "price_only": True,
            "volume_amount_policy": "retained_from_active_qdp",
            "aggregate_evidence": aggregate_metadata,
        },
        "selection": {
            "id": run_dir.name,
            "mode": str(selection_mode),
            "minimum_relative_error": float(minimum_relative_error),
            "maximum_relative_error": maximum_relative_error,
            "excluded_date_ranges": [list(value) for value in excluded_ranges],
            "target_stock_days": int(len(targets)),
            "date_range_deferred_stock_days": int(len(range_deferred)),
            "target_symbols": int(targets["symbol"].nunique()),
            "target_years": sorted({int(value) for value in targets["year"]}),
        },
        "prefilter": {
            "candidate_stock_days": int(prefilter["status"].eq("candidate").sum()),
            "rejected_stock_days": int(prefilter["status"].eq("rejected").sum()),
            "rejection_reasons": {
                str(key): int(value)
                for key, value in prefilter.loc[
                    prefilter["status"].eq("rejected"), "reason"
                ].value_counts().items()
            },
        },
        "capture": capture,
        "evaluation": {
            "accepted_stock_days": int(len(accepted)),
            "rejected_stock_days": int(decisions["status"].eq("rejected").sum()),
            "changed_rows": int(len(changes)),
            "repaired_field_stock_days": _field_counts(decisions),
            "rejection_reasons": _reason_counts(decisions),
            "flow_policy": "keep existing QDP volume and amount; external source prices only",
        },
        "extreme_timing_validation": timing,
        "artifacts": artifacts,
        "artifact_sha256": _artifact_hashes(artifacts),
        "installation": installation,
    }


def run_local_parquet_minute_repair(
    *,
    local_minute_root: str | Path,
    workspace_root: str | Path | None = None,
    minimum_relative_error: float = DEFAULT_PRIORITY_RELATIVE_ERROR,
    maximum_relative_error: float | None = None,
    selection_mode: str = "open",
    excluded_date_ranges: tuple[tuple[str, str], ...] | list[tuple[str, str]] | None = None,
    aggregate_seed_path: str | Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Evaluate and optionally install local-source historical minute repairs."""

    mode = str(selection_mode).strip().lower()
    if mode not in {"open", "high-low"}:
        raise ValueError(f"minute_repair_local_selection_mode_invalid:{selection_mode}")
    workspace = qdp_paths(workspace_root).workspace_root
    source_root = Path(local_minute_root).resolve()
    _, inventory = local_source_inventory(source_root)
    all_targets = load_priority_targets(
        workspace,
        minimum_relative_error=float(minimum_relative_error),
        maximum_relative_error=maximum_relative_error,
        include_known_probes=False,
        selection_mode=mode,
    )
    all_targets = _restrict_fields(all_targets, mode)
    ranges = _normalize_excluded_ranges(excluded_date_ranges)
    targets, range_deferred = _split_excluded_ranges(all_targets, ranges)
    if targets.empty:
        raise MinuteRepairError("minute_repair_local_targets_empty_after_date_filter")
    selection_id = _selection_id(
        targets,
        selection_mode=mode,
        minimum_relative_error=float(minimum_relative_error),
        maximum_relative_error=maximum_relative_error,
        excluded_ranges=ranges,
        inventory_fingerprint=str(inventory["fingerprint"]),
    )
    run_dir = (
        qdp_paths(workspace).source_archives_dir
        / "external_quant_data"
        / "minute_repair"
        / selection_id
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    aggregates, aggregate_metadata, aggregate_artifacts = prepare_local_aggregate_evidence(
        targets,
        source_root=source_root,
        run_dir=run_dir,
        seed_path=aggregate_seed_path,
    )
    candidates, prefilter_rejected, prefilter = prefilter_local_candidates(targets, aggregates)
    prefilter_path = run_dir / "local_prefilter_decisions.parquet"
    prefilter.to_parquet(prefilter_path, index=False, engine="pyarrow", compression="zstd")
    batches, capture = capture_local_candidate_batches(
        candidates,
        source_root=source_root,
        raw_dir=run_dir / "raw",
    )
    evaluated, changes = evaluate_local_candidates(
        candidates,
        batches,
        workspace_root=workspace,
    )
    decisions = pd.concat([prefilter_rejected, evaluated], ignore_index=True)
    decisions = decisions.sort_values(["trade_date", "symbol"], kind="stable").reset_index(drop=True)
    if len(decisions) != len(targets):
        raise MinuteRepairError(
            f"minute_repair_local_decision_count_invalid:{len(decisions)}!={len(targets)}"
        )
    artifacts = write_candidate_artifacts(
        run_dir,
        targets=targets,
        batches=batches,
        decisions=decisions,
        changes=changes,
    )
    artifacts.update(aggregate_artifacts)
    artifacts["local_prefilter"] = str(prefilter_path)
    artifacts.update(_write_fallback_artifacts(run_dir, decisions, range_deferred))

    timing: dict[str, Any] = {}
    if mode == "high-low" and not changes.empty:
        timing_frame, timing = validate_extreme_timing_sample(
            decisions,
            batches,
            workspace_root=workspace,
        )
        timing_path = run_dir / "extreme_timing_sample.parquet"
        timing_summary_path = run_dir / "extreme_timing_summary.json"
        timing_frame.to_parquet(timing_path, index=False, engine="pyarrow", compression="zstd")
        write_json(timing_summary_path, timing)
        artifacts.update(
            {
                "extreme_timing_sample": str(timing_path),
                "extreme_timing_summary": str(timing_summary_path),
            }
        )

    config = {
        "schema": "quantlab.local_parquet_minute_repair_run_config/v1",
        "provider": LOCAL_PROVIDER_ID,
        "repair_reason": LOCAL_REPAIR_REASON,
        "source_root": str(source_root),
        "selection_mode": mode,
        "minimum_relative_error": float(minimum_relative_error),
        "maximum_relative_error": maximum_relative_error,
        "excluded_date_ranges": [list(value) for value in ranges],
    }
    config_path = run_dir / "run_config.json"
    write_json(config_path, config)
    artifacts["run_config"] = str(config_path)

    installation: dict[str, Any] = {}
    if apply and not changes.empty:
        if mode == "high-low" and not bool(timing.get("gate_passed", False)):
            raise MinuteRepairError("minute_repair_extreme_timing_gate_failed")
        installation = install_accepted_changes(
            changes,
            decisions,
            run_dir=run_dir,
            workspace_root=workspace,
            provider=LOCAL_PROVIDER_ID,
            repair_reason=LOCAL_REPAIR_REASON,
        )
        status = "applied"
    elif changes.empty:
        status = "nothing_repairable"
    else:
        status = "would_apply"
    result = _result_payload(
        status=status,
        workspace=workspace,
        run_dir=run_dir,
        source_root=source_root,
        selection_mode=mode,
        minimum_relative_error=float(minimum_relative_error),
        maximum_relative_error=maximum_relative_error,
        excluded_ranges=ranges,
        targets=targets,
        range_deferred=range_deferred,
        aggregate_metadata=aggregate_metadata,
        prefilter=prefilter,
        capture=capture,
        decisions=decisions,
        changes=changes,
        timing=timing,
        artifacts=artifacts,
        installation=installation,
    )
    write_json(run_dir / "result.json", json_safe(result))
    return result


def resume_local_parquet_minute_repair(
    run_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Finish audits and manifests after a local-source install interruption."""

    output = Path(run_dir).resolve()
    config_path = output / "run_config.json"
    if not config_path.is_file():
        raise MinuteRepairError(f"minute_repair_local_run_config_missing:{config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("provider") != LOCAL_PROVIDER_ID:
        raise MinuteRepairError("minute_repair_local_run_provider_invalid")
    targets = pd.read_parquet(output / "targets.parquet")
    decisions = pd.read_parquet(output / "candidate_decisions.parquet")
    changes = pd.read_parquet(output / "accepted_row_changes.parquet")
    installation = resume_interrupted_install(
        changes,
        decisions,
        run_dir=output,
        workspace_root=workspace_root,
        provider=LOCAL_PROVIDER_ID,
        repair_reason=LOCAL_REPAIR_REASON,
    )
    result_path = output / "result.json"
    prior = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    prior.update(
        {
            "status": "applied",
            "resumed_at": _utc_now(),
            "installation": installation,
            "selection": {
                **dict(prior.get("selection", {}) or {}),
                "target_stock_days": int(len(targets)),
            },
        }
    )
    write_json(result_path, json_safe(prior))
    return prior


def refresh_local_extreme_timing_validation(
    run_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the small high/low timing sample without repeating candidate evaluation."""

    output = Path(run_dir).resolve()
    decisions = pd.read_parquet(output / "candidate_decisions.parquet")
    batch_payload = json.loads((output / "batches.json").read_text(encoding="utf-8"))
    batches = [TargetBatch(**dict(item)) for item in list(batch_payload.get("batches", []) or [])]
    timing_frame, timing = validate_extreme_timing_sample(
        decisions,
        batches,
        workspace_root=workspace_root,
    )
    timing_path = output / "extreme_timing_sample.parquet"
    timing_summary_path = output / "extreme_timing_summary.json"
    timing_frame.to_parquet(timing_path, index=False, engine="pyarrow", compression="zstd")
    write_json(timing_summary_path, timing)
    result_path = output / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        artifacts = dict(result.get("artifacts", {}) or {})
        artifacts.update(
            {
                "extreme_timing_sample": str(timing_path),
                "extreme_timing_summary": str(timing_summary_path),
            }
        )
        result["extreme_timing_validation"] = timing
        result["artifacts"] = artifacts
        result["artifact_sha256"] = _artifact_hashes(artifacts)
        write_json(result_path, json_safe(result))
    return timing


def apply_local_parquet_minute_repair_run(
    run_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Install an already evaluated local-source run without repeating its expensive scan."""

    output = Path(run_dir).resolve()
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    if config.get("provider") != LOCAL_PROVIDER_ID:
        raise MinuteRepairError("minute_repair_local_run_provider_invalid")
    decisions = pd.read_parquet(output / "candidate_decisions.parquet")
    changes = pd.read_parquet(output / "accepted_row_changes.parquet")
    if changes.empty:
        raise MinuteRepairError("minute_repair_local_run_changes_empty")
    mode = str(config.get("selection_mode", ""))
    timing: dict[str, Any] = {}
    if mode == "high-low":
        refresh_local_extreme_timing_validation(
            output,
            workspace_root=workspace_root,
        )
        timing = combine_extreme_timing_validation(output)
        if not bool(timing.get("gate_passed", False)):
            raise MinuteRepairError("minute_repair_extreme_timing_gate_failed")
    installation = install_accepted_changes(
        changes,
        decisions,
        run_dir=output,
        workspace_root=workspace_root,
        provider=LOCAL_PROVIDER_ID,
        repair_reason=LOCAL_REPAIR_REASON,
    )
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "status": "applied",
            "applied_at": _utc_now(),
            "installation": installation,
        }
    )
    if timing:
        result["extreme_timing_validation"] = timing
    write_json(result_path, json_safe(result))
    return result


def combine_extreme_timing_validation(run_dir: str | Path) -> dict[str, Any]:
    """Gate timing on independent 5m days whose own daily extreme is trustworthy."""

    output = Path(run_dir).resolve()
    active_summary = json.loads(
        (output / "extreme_timing_summary.json").read_text(encoding="utf-8")
    )
    baostock_path = output / "baostock_extreme_timing_sample.parquet"
    if not baostock_path.is_file():
        raise MinuteRepairError(f"minute_repair_baostock_timing_evidence_missing:{baostock_path}")
    baostock = pd.read_parquet(baostock_path)
    decisions = pd.read_parquet(output / "candidate_decisions.parquet")
    daily = decisions.loc[:, ["symbol", "trade_date", "daily_high", "daily_low"]]
    baostock = baostock.drop(
        columns=["daily_high", "daily_low", "daily_value", "price_reference_match"],
        errors="ignore",
    ).merge(
        daily,
        on=["symbol", "trade_date"],
        how="left",
        validate="many_to_one",
    )
    baostock["daily_value"] = baostock.apply(
        lambda row: row["daily_high"] if str(row["field"]) == "high" else row["daily_low"],
        axis=1,
    )
    baostock["price_reference_match"] = (
        baostock["baostock_extreme"].sub(baostock["daily_value"]).abs()
        <= PRICE_ABSOLUTE_TOLERANCE + PRICE_COMPARISON_EPSILON
    )
    baostock.to_parquet(baostock_path, index=False, engine="pyarrow", compression="zstd")
    reliable = baostock.loc[
        baostock["reference_available"].fillna(False).astype(bool)
        & baostock["price_reference_match"].fillna(False).astype(bool)
    ]
    baostock_rate = float(reliable["timing_match"].mean()) if not reliable.empty else 0.0

    external = pd.read_parquet(output / "extreme_timing_sample.parquet")
    provider = decisions.loc[
        :,
        ["symbol", "trade_date", "provider_high", "provider_low", "daily_high", "daily_low"],
    ]
    external = external.merge(
        provider,
        on=["symbol", "trade_date"],
        how="left",
        validate="many_to_one",
    )
    external["provider_value"] = external.apply(
        lambda row: row["provider_high"]
        if str(row["field"]) == "high"
        else row["provider_low"],
        axis=1,
    )
    external["daily_value"] = external.apply(
        lambda row: row["daily_high"] if str(row["field"]) == "high" else row["daily_low"],
        axis=1,
    )
    external_exact = external["provider_value"].eq(external["daily_value"])
    external_exact_rate = float(external_exact.mean()) if not external.empty else 0.0
    summary = {
        "schema": "quantlab.combined_extreme_timing_validation/v1",
        "created_at": _utc_now(),
        "gate_passed": bool(
            len(reliable) >= 30 and baostock_rate >= 0.98 and external_exact_rate >= 0.99
        ),
        "policy": (
            "external 1m must reproduce trusted daily extrema on at least 99% of the sample; "
            "BaoStock timing is counted only when BaoStock itself reproduces that daily extreme, "
            "then at least 98% of those independent 5m buckets must agree"
        ),
        "external_daily_exact": {
            "sample_rows": int(len(external)),
            "exact_rows": int(external_exact.sum()),
            "exact_rate": external_exact_rate,
            "threshold": 0.99,
        },
        "baostock_five_minute": {
            "sample_rows": int(len(baostock)),
            "complete_session_rows": int(
                baostock["reference_available"].fillna(False).astype(bool).sum()
            ),
            "reliable_price_reference_rows": int(len(reliable)),
            "timing_match_rows": int(reliable["timing_match"].sum()),
            "timing_match_rate": baostock_rate,
            "threshold": 0.98,
        },
        "active_five_minute_informational": active_summary,
    }
    summary_path = output / "combined_extreme_timing_summary.json"
    write_json(summary_path, summary)
    result_path = output / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        artifacts = dict(result.get("artifacts", {}) or {})
        artifacts.update(
            {
                "baostock_extreme_timing_sample": str(baostock_path),
                "baostock_extreme_timing_summary": str(
                    output / "baostock_extreme_timing_summary.json"
                ),
                "baostock_five_minute_raw": str(output / "baostock_5m_timing_raw.parquet"),
                "combined_extreme_timing_summary": str(summary_path),
            }
        )
        result["extreme_timing_validation"] = summary
        result["artifacts"] = artifacts
        result["artifact_sha256"] = _artifact_hashes(artifacts)
        write_json(result_path, json_safe(result))
    return summary


__all__ = [
    "apply_local_parquet_minute_repair_run",
    "combine_extreme_timing_validation",
    "refresh_local_extreme_timing_validation",
    "resume_local_parquet_minute_repair",
    "run_local_parquet_minute_repair",
]
