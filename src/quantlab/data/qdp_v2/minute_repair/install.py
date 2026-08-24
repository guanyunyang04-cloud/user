"""Transactional shard installation and post-repair minute quality audits."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import atomic_copy_file, sha256_file, stable_hash, write_json
from quantlab.data.core.paths import qdp_paths
from quantlab.data.minute_archive.quality import (
    PRICE_ABSOLUTE_TOLERANCE,
    PRICE_COMPARISON_EPSILON,
    parity_audit,
    qdp_daily_paths,
)
from quantlab.data.qdp_v2.manifest import (
    dataset_manifest_for_id,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quantlab.data.qdp_v2.repair import (
    mutate_active_shards_from_parquet,
    update_active_manifest_metadata,
)

from .candidate import KEY_COLUMNS, PRICE_COLUMNS, MinuteRepairError

_PARITY_ARTIFACT_NAMES = (
    "daily_parity_material_mismatches.parquet",
    "minute_feature_exclusions.parquet",
    "daily_reference_missing_minute.parquet",
)
_QUALITY_COUNT_KEYS = (
    "price_over_five_cent_rows",
    "price_warning_rows",
    "price_unreliable_rows",
    "price_severe_rows",
    "minute_feature_exclusion_rows",
    "total_minute_feature_exclusion_rows",
)


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sql_text(value: str | Path) -> str:
    return "'" + Path(value).as_posix().replace("'", "''") + "'"


def _active_context(
    domain: str,
    *,
    workspace_root: str | Path | None,
) -> tuple[Path, dict[str, Any], Path, Any]:
    root = qdp_paths(workspace_root).qdp_v2_dir.resolve()
    active = read_active_manifest(root)
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, ""))
    if not dataset_id:
        raise MinuteRepairError(f"minute_repair_active_domain_missing:{domain}")
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise MinuteRepairError(f"minute_repair_active_manifest_missing:{domain}")
    return root, active, manifest_path, read_dataset_manifest(manifest_path)


def _validate_change_frame(changes: pd.DataFrame) -> None:
    required = {
        *KEY_COLUMNS,
        *PRICE_COLUMNS,
        "volume",
        "amount",
        "source",
        "domain",
        "active_shard_path",
        *{f"old_{field}" for field in PRICE_COLUMNS},
    }
    missing = sorted(required.difference(changes.columns))
    if missing:
        raise MinuteRepairError(f"minute_repair_change_columns_missing:{','.join(missing)}")
    if changes.empty:
        raise MinuteRepairError("minute_repair_changes_empty")
    if changes.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteRepairError("minute_repair_change_primary_key_duplicate")
    allowed = {"market_intraday_1m", "market_opening_auction"}
    observed = set(changes["domain"].astype(str))
    if not observed.issubset(allowed):
        raise MinuteRepairError(f"minute_repair_change_domain_invalid:{sorted(observed)}")


def _prepared_path(run_dir: Path, domain: str, active_path: Path) -> Path:
    digest = stable_hash({"domain": domain, "active_path": str(active_path.resolve())})[:16]
    return run_dir / "prepared" / domain / f"{active_path.stem}_{digest}.parquet"


def _cas_validate_source_rows(active_path: Path, changes: pd.DataFrame) -> None:
    expected = changes.loc[:, [*KEY_COLUMNS, *[f"old_{field}" for field in PRICE_COLUMNS]]].copy()
    with duckdb.connect(":memory:") as connection:
        connection.execute("SET enable_progress_bar=false")
        connection.register("expected", expected)
        rows = connection.execute(
            f"""
            SELECT b.symbol,b.trade_date,b.bar_time,b.open,b.high,b.low,b.close,
                   e.old_open,e.old_high,e.old_low,e.old_close
            FROM read_parquet({_sql_text(active_path)}) b
            INNER JOIN expected e USING(symbol,trade_date,bar_time)
            """
        ).df()
    if len(rows) != len(expected):
        raise MinuteRepairError(
            f"minute_repair_source_cas_match_count_invalid:{active_path}:{len(rows)}!={len(expected)}"
        )
    for field in PRICE_COLUMNS:
        if not rows[field].astype("float64").eq(rows[f"old_{field}"].astype("float64")).all():
            raise MinuteRepairError(f"minute_repair_source_cas_expected_mismatch:{active_path}:{field}")


def _write_prepared_shard(
    active_path: Path,
    changes: pd.DataFrame,
    target: Path,
) -> dict[str, Any]:
    _cas_validate_source_rows(active_path, changes)
    patches = changes.loc[:, [*KEY_COLUMNS, *PRICE_COLUMNS, "source"]].copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with duckdb.connect(":memory:") as connection:
        connection.execute("SET enable_progress_bar=false")
        connection.execute("SET preserve_insertion_order=true")
        connection.register("patches", patches)
        connection.execute(
            f"""
            COPY (
                SELECT b.symbol,b.trade_date,b.bar_time,
                       coalesce(p.open,b.open)::DOUBLE AS open,
                       coalesce(p.high,b.high)::DOUBLE AS high,
                       coalesce(p.low,b.low)::DOUBLE AS low,
                       coalesce(p.close,b.close)::DOUBLE AS close,
                       b.volume,b.amount,
                       CASE WHEN p.symbol IS NULL THEN b.source ELSE p.source END AS source,
                       b.adjusted_flag
                FROM read_parquet({_sql_text(active_path)}) b
                LEFT JOIN patches p USING(symbol,trade_date,bar_time)
            ) TO {_sql_text(temporary)}
              (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
            """
        )
    old_parquet = pq.ParquetFile(active_path)
    new_parquet = pq.ParquetFile(temporary)
    if old_parquet.metadata is None or new_parquet.metadata is None:
        old_parquet.close()
        new_parquet.close()
        temporary.unlink(missing_ok=True)
        raise MinuteRepairError(f"minute_repair_prepared_parquet_metadata_missing:{active_path}")
    old_row_count = int(old_parquet.metadata.num_rows)
    new_row_count = int(new_parquet.metadata.num_rows)
    old_schema = old_parquet.schema_arrow
    new_schema = new_parquet.schema_arrow
    old_parquet.close()
    new_parquet.close()
    if old_row_count != new_row_count:
        temporary.unlink(missing_ok=True)
        raise MinuteRepairError(f"minute_repair_prepared_row_count_changed:{active_path}")
    if not new_schema.equals(old_schema, check_metadata=False):
        temporary.unlink(missing_ok=True)
        raise MinuteRepairError(f"minute_repair_prepared_schema_changed:{active_path}")
    os.replace(temporary, target)

    with duckdb.connect(":memory:") as connection:
        connection.execute("SET enable_progress_bar=false")
        connection.register("patches", patches)
        verified = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM read_parquet({_sql_text(target)}) b
                INNER JOIN patches p USING(symbol,trade_date,bar_time)
                WHERE b.open=p.open AND b.high=p.high AND b.low=p.low AND b.close=p.close
                  AND b.source=p.source
                """
            ).fetchone()[0]
        )
    if verified != len(patches):
        target.unlink(missing_ok=True)
        raise MinuteRepairError(f"minute_repair_prepared_patch_verification_failed:{active_path}")
    return {
        "domain": str(changes["domain"].iloc[0]),
        "active_path": str(active_path),
        "prepared_path": str(target),
        "old_row_count": old_row_count,
        "new_row_count": new_row_count,
        "changed_row_count": int(len(patches)),
        "prepared_sha256": sha256_file(target),
    }


def stage_active_shard_replacements(
    changes: pd.DataFrame,
    *,
    run_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    _validate_change_frame(changes)
    output = Path(run_dir).resolve()
    records: list[dict[str, Any]] = []
    for (domain, active_path_text), selected in changes.groupby(["domain", "active_shard_path"], sort=True):
        active_path = Path(str(active_path_text)).resolve()
        _, _, _, manifest = _active_context(str(domain), workspace_root=workspace_root)
        root = qdp_paths(workspace_root).qdp_v2_dir.resolve()
        known = {resolve_manifest_path(item.path, root=root).resolve() for item in manifest.shards}
        if active_path not in known:
            raise MinuteRepairError(f"minute_repair_change_shard_not_active:{active_path}")
        target = _prepared_path(output, str(domain), active_path)
        records.append(_write_prepared_shard(active_path, selected.reset_index(drop=True), target))
    write_json(
        output / "prepared" / "staged_replacements.json",
        {
            "schema": "quantlab.targeted_minute_repair_staged_shards/v1",
            "created_at": _utc_now(),
            "replacements": records,
        },
    )
    return records


def _backup_file(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    if target.exists():
        if sha256_file(target) != source_hash:
            raise MinuteRepairError(f"minute_repair_backup_conflict:{target}")
        return {"source": str(source), "backup": str(target), "method": "existing", "sha256": source_hash}
    try:
        os.link(source, target)
        method = "hardlink"
    except OSError:
        atomic_copy_file(source, target)
        method = "copy"
    if sha256_file(target) != source_hash:
        target.unlink(missing_ok=True)
        raise MinuteRepairError(f"minute_repair_backup_hash_mismatch:{target}")
    return {"source": str(source), "backup": str(target), "method": method, "sha256": source_hash}


def backup_active_state(
    staged: Sequence[Mapping[str, Any]],
    *,
    run_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    output = Path(run_dir).resolve() / "backup_pre_repair"
    qdp_root = qdp_paths(workspace_root).qdp_v2_dir.resolve()
    records = [_backup_file(qdp_root / "active" / "active.json", output / "active.json")]
    domains = sorted({str(item["domain"]) for item in staged})
    for domain in domains:
        _, _, manifest_path, _ = _active_context(domain, workspace_root=workspace_root)
        records.append(_backup_file(manifest_path, output / "manifests" / domain / "dataset.json"))
    for item in staged:
        source = Path(str(item["active_path"])).resolve()
        digest = stable_hash({"path": str(source)})[:16]
        target = output / "shards" / str(item["domain"]) / f"{source.stem}_{digest}.parquet"
        records.append(_backup_file(source, target))
    write_json(
        output / "backup_manifest.json",
        {
            "schema": "quantlab.targeted_minute_repair_backup/v1",
            "created_at": _utc_now(),
            "files": records,
        },
    )
    return records


def install_staged_replacements(
    staged: Sequence[Mapping[str, Any]],
    *,
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    by_domain: dict[str, list[tuple[str, str]]] = {}
    for item in staged:
        by_domain.setdefault(str(item["domain"]), []).append(
            (str(item["active_path"]), str(item["prepared_path"]))
        )
    for domain, replacements in sorted(by_domain.items()):
        results.append(
            mutate_active_shards_from_parquet(
                domain,
                replacements=replacements,
                reason="targeted Tushare-compatible historical minute price repair",
                workspace_root=workspace_root,
                primary_keys_prevalidated=True,
            )
        )
    return results


def _year_paths(
    domain: str,
    year: int,
    *,
    workspace_root: str | Path | None,
) -> list[Path]:
    root, _, _, manifest = _active_context(domain, workspace_root=workspace_root)
    start = f"{int(year):04d}-01-01"
    end = f"{int(year):04d}-12-31"
    return [
        resolve_manifest_path(item.path, root=root)
        for item in manifest.shards
        if str(item.start_date) <= end and str(item.end_date) >= start
    ]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MinuteRepairError(f"minute_repair_json_invalid:{path}:{exc}") from exc
    if not isinstance(payload, dict):
        raise MinuteRepairError(f"minute_repair_json_root_invalid:{path}")
    return payload


def _quality_directory(workspace_root: str | Path | None, year: int) -> Path:
    return qdp_paths(workspace_root).source_archives_dir / "minute" / "quality" / f"year={int(year)}"


def _backup_quality_year(
    year: int,
    *,
    run_dir: Path,
    workspace_root: str | Path | None,
) -> list[dict[str, Any]]:
    source_dir = _quality_directory(workspace_root, year)
    target_dir = run_dir / "backup_pre_repair" / "quality" / f"year={int(year)}"
    names = ("audit.json", *_PARITY_ARTIFACT_NAMES)
    return [_backup_file(source_dir / name, target_dir / name) for name in names]


def _quality_counts(quality: Mapping[str, Any]) -> dict[str, Any]:
    return {key: quality.get(key) for key in _QUALITY_COUNT_KEYS}


def _write_year_audit(
    *,
    audit_path: Path,
    audit: dict[str, Any],
    quality: dict[str, Any],
    run_id: str,
) -> None:
    session_exclusions = int(quality.get("session_feature_exclusion_rows", 0) or 0)
    quality["total_minute_feature_exclusion_rows"] = (
        int(quality.get("price_unreliable_rows", 0) or 0) + session_exclusions
    )
    quality["status"] = (
        "ready_with_exclusions" if int(quality["total_minute_feature_exclusion_rows"]) else "ready"
    )
    audit["quality"] = quality
    audit["quality_policy_updated_at"] = _utc_now()
    audit["last_targeted_minute_repair_at"] = _utc_now()
    audit["last_targeted_minute_repair_run"] = str(run_id)
    write_json(audit_path, audit)


def re_audit_affected_years(
    changes: pd.DataFrame,
    *,
    run_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    output = Path(run_dir).resolve()
    years = sorted({int(str(value)[:4]) for value in changes["trade_date"]})
    daily_paths = qdp_daily_paths(qdp_paths(workspace_root).workspace_root)
    results: list[dict[str, Any]] = []
    for year in years:
        backup_records = _backup_quality_year(
            year,
            run_dir=output,
            workspace_root=workspace_root,
        )
        quality_dir = _quality_directory(workspace_root, year)
        temporary_dir = output / "reaudit" / f"year={year}"
        temporary_dir.mkdir(parents=True, exist_ok=True)
        metrics = parity_audit(
            continuous_paths=_year_paths("market_intraday_1m", year, workspace_root=workspace_root),
            auction_paths=_year_paths("market_opening_auction", year, workspace_root=workspace_root),
            daily_paths=daily_paths,
            year=year,
            output_dir=temporary_dir,
            final_quality_dir=quality_dir,
        )
        for name in _PARITY_ARTIFACT_NAMES:
            atomic_copy_file(temporary_dir / name, quality_dir / name)
        audit_path = quality_dir / "audit.json"
        audit = _read_json(audit_path)
        quality = dict(audit.get("quality", {}) or {})
        before = _quality_counts(quality)
        quality.update(metrics)
        _write_year_audit(
            audit_path=audit_path,
            audit=audit,
            quality=quality,
            run_id=output.name,
        )
        results.append(
            {
                "year": year,
                "before": before,
                "after": _quality_counts(quality),
                "backup_files": backup_records,
                "audit_path": str(audit_path),
            }
        )
    return results


def _validate_reaudit_artifacts(directory: Path) -> None:
    for name in _PARITY_ARTIFACT_NAMES:
        path = directory / name
        if not path.is_file():
            raise MinuteRepairError(f"minute_repair_reaudit_artifact_missing:{path}")
        try:
            parquet = pq.ParquetFile(path)
            if parquet.metadata is None:
                raise MinuteRepairError(f"minute_repair_reaudit_artifact_metadata_missing:{path}")
            parquet.close()
        except MinuteRepairError:
            raise
        except Exception as exc:
            raise MinuteRepairError(f"minute_repair_reaudit_artifact_invalid:{path}:{exc}") from exc


def _reaudit_artifacts_valid(directory: Path) -> bool:
    try:
        _validate_reaudit_artifacts(directory)
    except MinuteRepairError:
        return False
    return True


def _adjust_exact_rates_from_decisions(
    quality: dict[str, Any],
    decisions: pd.DataFrame,
) -> None:
    common = int(quality.get("common_stock_days", 0) or 0)
    if not common or decisions.empty:
        return
    accepted = decisions.loc[decisions["status"].eq("accepted")]
    for field in PRICE_COLUMNS:
        columns = {f"baseline_{field}", f"post_{field}", f"daily_{field}"}
        if not columns.issubset(accepted.columns):
            continue
        before = accepted[f"baseline_{field}"].astype("float64").eq(
            accepted[f"daily_{field}"].astype("float64")
        )
        after = accepted[f"post_{field}"].astype("float64").eq(
            accepted[f"daily_{field}"].astype("float64")
        )
        delta = int(after.sum() - before.sum())
        key = f"{field}_exact_rate"
        old_count = round(float(quality.get(key, 0.0) or 0.0) * common)
        quality[key] = float((old_count + delta) / common)


def _metrics_from_reaudit_artifacts(
    directory: Path,
    *,
    final_quality_dir: Path,
) -> dict[str, Any]:
    _validate_reaudit_artifacts(directory)
    material = pd.read_parquet(directory / "daily_parity_material_mismatches.parquet")
    exclusions = pd.read_parquet(directory / "minute_feature_exclusions.parquet")
    missing = pd.read_parquet(directory / "daily_reference_missing_minute.parquet")
    price_over = int(
        material["price_max_abs_error"].gt(PRICE_ABSOLUTE_TOLERANCE + PRICE_COMPARISON_EPSILON).sum()
    )
    severe = int(exclusions["severity"].astype(str).eq("severe").sum())
    unreliable = int(len(exclusions))
    return {
        "material_daily_mismatch_rows": int(len(material)),
        "price_over_five_cent_rows": price_over,
        "price_warning_rows": price_over - unreliable,
        "price_unreliable_rows": unreliable,
        "price_severe_rows": severe,
        "minute_feature_exclusion_rows": unreliable,
        "minute_feature_exclusions_path": str(
            (final_quality_dir / "minute_feature_exclusions.parquet").resolve()
        ),
        "daily_reference_missing_minute_rows": int(len(missing)),
        "daily_reference_missing_minute_path": str(
            (final_quality_dir / "daily_reference_missing_minute.parquet").resolve()
        ),
    }


def finish_existing_reaudits(
    changes: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    run_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Finish or reuse year audits after an install-stage interruption."""

    output = Path(run_dir).resolve()
    years = sorted({int(str(value)[:4]) for value in changes["trade_date"]})
    daily_paths: list[Path] | None = None
    results: list[dict[str, Any]] = []
    for year in years:
        quality_dir = _quality_directory(workspace_root, year)
        temporary_dir = output / "reaudit" / f"year={year}"
        backup_dir = output / "backup_pre_repair" / "quality" / f"year={year}"
        backup_audit = _read_json(backup_dir / "audit.json")
        audit_path = quality_dir / "audit.json"
        audit = _read_json(audit_path)
        complete = str(audit.get("last_targeted_minute_repair_run", "")) == output.name
        if complete:
            _validate_reaudit_artifacts(temporary_dir)
            for name in _PARITY_ARTIFACT_NAMES:
                if sha256_file(quality_dir / name) != sha256_file(temporary_dir / name):
                    raise MinuteRepairError(f"minute_repair_completed_reaudit_hash_mismatch:{year}:{name}")
            quality = dict(audit.get("quality", {}) or {})
            status = "already_complete"
        else:
            full_metrics: dict[str, Any] | None = None
            if not _reaudit_artifacts_valid(temporary_dir):
                if daily_paths is None:
                    daily_paths = qdp_daily_paths(qdp_paths(workspace_root).workspace_root)
                temporary_dir.mkdir(parents=True, exist_ok=True)
                full_metrics = parity_audit(
                    continuous_paths=_year_paths("market_intraday_1m", year, workspace_root=workspace_root),
                    auction_paths=_year_paths("market_opening_auction", year, workspace_root=workspace_root),
                    daily_paths=daily_paths,
                    year=year,
                    output_dir=temporary_dir,
                    final_quality_dir=quality_dir,
                )
            for name in _PARITY_ARTIFACT_NAMES:
                atomic_copy_file(temporary_dir / name, quality_dir / name)
            quality = dict(audit.get("quality", {}) or {})
            quality.update(
                full_metrics
                or _metrics_from_reaudit_artifacts(
                    temporary_dir,
                    final_quality_dir=quality_dir,
                )
            )
            year_decisions = decisions.loc[decisions["trade_date"].astype(str).str.startswith(str(year))]
            if full_metrics is None:
                _adjust_exact_rates_from_decisions(quality, year_decisions)
            _write_year_audit(
                audit_path=audit_path,
                audit=audit,
                quality=quality,
                run_id=output.name,
            )
            status = "completed_from_existing_artifacts" if full_metrics is None else "completed_by_reaudit"
        results.append(
            {
                "year": year,
                "status": status,
                "before": _quality_counts(dict(backup_audit.get("quality", {}) or {})),
                "after": _quality_counts(quality),
                "audit_path": str(audit_path),
            }
        )
    return results


def _aggregate_quality(workspace_root: str | Path | None) -> dict[str, Any]:
    quality_root = qdp_paths(workspace_root).source_archives_dir / "minute" / "quality"
    audits = [_read_json(path) for path in sorted(quality_root.glob("year=*/audit.json"))]
    qualities = [dict(item.get("quality", {}) or {}) for item in audits]

    def total(key: str) -> int:
        return sum(int(item.get(key, 0) or 0) for item in qualities)

    coverage = [float(item.get("daily_reference_coverage_rate", 0.0) or 0.0) for item in qualities]
    return {
        "daily_reference_coverage_rate_min": min(coverage) if coverage else 0.0,
        "minute_feature_exclusion_rows": total("total_minute_feature_exclusion_rows"),
        "price_over_five_cent_rows": total("price_over_five_cent_rows"),
        "price_warning_rows": total("price_warning_rows"),
        "price_unreliable_rows": total("price_unreliable_rows"),
        "price_severe_rows": total("price_severe_rows"),
        "session_feature_exclusion_rows": total("session_feature_exclusion_rows"),
        "stock_day_count": total("stock_day_count"),
        "unexplained_incomplete_stock_days": total("unexplained_incomplete_stock_days"),
    }


def update_minute_manifests_after_repair(
    *,
    decisions: pd.DataFrame,
    changes: pd.DataFrame,
    run_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    quality = _aggregate_quality(workspace_root)
    accepted = decisions.loc[decisions["status"].eq("accepted")]
    repair_record = {
        "provider": "tushare_compatible_stk_mins",
        "record_path": str(Path(run_dir).resolve() / "result.json"),
        "price_only": True,
        "volume_amount_policy": "retained_from_active_qdp",
        "accepted_stock_days": int(len(accepted)),
        "changed_rows": int(len(changes)),
        "applied_at": _utc_now(),
    }
    results: list[dict[str, Any]] = []
    for domain in ("market_intraday_1m", "market_opening_auction"):
        _, _, _, manifest = _active_context(domain, workspace_root=workspace_root)
        history = [
            dict(item)
            for item in list(manifest.source.get("targeted_minute_repairs", []) or [])
            if isinstance(item, Mapping)
        ]
        prior = manifest.source.get("last_targeted_minute_repair")
        if isinstance(prior, Mapping) and not history:
            history.append(dict(prior))
        known_paths = {str(item.get("record_path", "")) for item in history}
        if repair_record["record_path"] not in known_paths:
            history.append(repair_record)
        results.append(
            update_active_manifest_metadata(
                domain,
                reason="record targeted historical minute repair and refreshed parity counts",
                workspace_root=workspace_root,
                source_updates={
                    "last_targeted_minute_repair": repair_record,
                    "targeted_minute_repairs": history,
                },
                quality_updates=quality,
            )
        )
    return results


def verify_installed_changes(
    changes: pd.DataFrame,
    staged: Sequence[Mapping[str, Any]],
    *,
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Verify that every staged shard and accepted row is active after interruption."""

    _validate_change_frame(changes)
    results: list[dict[str, Any]] = []
    for domain, staged_items in _group_records(staged, "domain").items():
        root, _, _, manifest = _active_context(domain, workspace_root=workspace_root)
        repair_paths = [
            resolve_manifest_path(item.path, root=root)
            for item in manifest.shards
            if str(dict(item.metadata or {}).get("reason", ""))
            == "targeted Tushare-compatible historical minute price repair"
        ]
        active_by_hash: dict[str, list[Path]] = {}
        for path in repair_paths:
            active_by_hash.setdefault(sha256_file(path), []).append(path)
        matched: list[Path] = []
        for item in staged_items:
            expected_hash = str(item.get("prepared_sha256", "") or sha256_file(item["prepared_path"]))
            candidates = active_by_hash.get(expected_hash, [])
            if len(candidates) != 1:
                raise MinuteRepairError(
                    f"minute_repair_installed_shard_hash_match_invalid:{domain}:{expected_hash}:{len(candidates)}"
                )
            matched.append(candidates[0])

        selected = changes.loc[changes["domain"].astype(str).eq(domain)].copy()
        paths_sql = ",".join(_sql_text(path) for path in matched)
        with duckdb.connect(":memory:") as connection:
            connection.execute("SET enable_progress_bar=false")
            connection.register("accepted_changes", selected)
            installed = connection.execute(
                f"""
                SELECT b.symbol,b.trade_date,b.bar_time,b.open,b.high,b.low,b.close,
                       b.volume,b.amount,b.source,
                       c.open expected_open,c.high expected_high,c.low expected_low,
                       c.close expected_close,c.volume expected_volume,
                       c.amount expected_amount,c.source expected_source
                FROM read_parquet([{paths_sql}],union_by_name=true) b
                INNER JOIN accepted_changes c USING(symbol,trade_date,bar_time)
                """
            ).df()
        if len(installed) != len(selected):
            raise MinuteRepairError(
                f"minute_repair_installed_change_match_count_invalid:{domain}:{len(installed)}!={len(selected)}"
            )
        for column in (*PRICE_COLUMNS, "volume", "amount", "source"):
            if not installed[column].eq(installed[f"expected_{column}"]).all():
                raise MinuteRepairError(f"minute_repair_installed_change_mismatch:{domain}:{column}")
        if installed.duplicated(list(KEY_COLUMNS)).any():
            raise MinuteRepairError(f"minute_repair_installed_change_duplicate_key:{domain}")
        valid_ohlc = (
            installed["low"].le(installed[["open", "close"]].min(axis=1))
            & installed["high"].ge(installed[["open", "close"]].max(axis=1))
            & installed["low"].le(installed["high"])
        )
        if not bool(valid_ohlc.all()):
            raise MinuteRepairError(f"minute_repair_installed_change_ohlc_invalid:{domain}")
        results.append(
            {
                "domain": domain,
                "status": "installed_verified",
                "active_shard_count": int(len(matched)),
                "changed_row_count": int(len(installed)),
                "active_shard_paths": [str(path) for path in matched],
            }
        )
    return results


def _group_records(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for item in records:
        result.setdefault(str(item[key]), []).append(item)
    return result


def resume_interrupted_install(
    changes: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    run_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resume after active shards were installed but final audits were interrupted."""

    output = Path(run_dir).resolve()
    staged_payload = _read_json(output / "prepared" / "staged_replacements.json")
    staged = [dict(item) for item in list(staged_payload.get("replacements", []) or [])]
    if not staged:
        raise MinuteRepairError("minute_repair_staged_replacements_missing")
    backup_payload = _read_json(output / "backup_pre_repair" / "backup_manifest.json")
    installed = verify_installed_changes(changes, staged, workspace_root=workspace_root)
    reaudits = finish_existing_reaudits(
        changes,
        decisions,
        run_dir=output,
        workspace_root=workspace_root,
    )
    metadata_updates = update_minute_manifests_after_repair(
        decisions=decisions,
        changes=changes,
        run_dir=output,
        workspace_root=workspace_root,
    )
    return {
        "resume_status": "completed",
        "staged_replacements": staged,
        "active_backups": list(backup_payload.get("files", []) or []),
        "installed_verification": installed,
        "year_reaudits": reaudits,
        "manifest_metadata_updates": metadata_updates,
    }


def install_accepted_changes(
    changes: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    run_dir: str | Path,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    staged = stage_active_shard_replacements(
        changes,
        run_dir=run_dir,
        workspace_root=workspace_root,
    )
    backups = backup_active_state(staged, run_dir=run_dir, workspace_root=workspace_root)
    mutations = install_staged_replacements(staged, workspace_root=workspace_root)
    reaudits = re_audit_affected_years(
        changes,
        run_dir=run_dir,
        workspace_root=workspace_root,
    )
    metadata_updates = update_minute_manifests_after_repair(
        decisions=decisions,
        changes=changes,
        run_dir=run_dir,
        workspace_root=workspace_root,
    )
    return {
        "staged_replacements": staged,
        "active_backups": backups,
        "mutations": mutations,
        "year_reaudits": reaudits,
        "manifest_metadata_updates": metadata_updates,
    }


__all__ = [
    "backup_active_state",
    "install_accepted_changes",
    "install_staged_replacements",
    "re_audit_affected_years",
    "resume_interrupted_install",
    "stage_active_shard_replacements",
    "update_minute_manifests_after_repair",
    "verify_installed_changes",
]
