"""Complete-outcome and candidate-gate audit for a narrow minute-v2 month.

The month builder deliberately labels only gated candidates.  That is useful
for storage, but it cannot answer whether the gate discarded the best rows.
This module materializes labels for the complementary base keys, combines
them with the already verified candidate labels, and writes a separate audit
bundle without changing the immutable month artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import DataContractError, read_json, sha256_file, write_json
from quantlab.data.qdp_v2.duckdb_resources import GIB, MIB, open_guarded_duckdb

from .builder import (
    BUILD_IMPLEMENTATION_REVISION,
    LABEL_BUCKET_COUNT,
    _artifact,
    _artifact_matches,
    _copy_query,
    _parquet_scan,
    _safe_clean_generated,
    verify_month,
)
from .contracts import KEY_COLUMNS, MinuteV2Config, MinuteV2Error
from .labels import LABEL_COLUMNS, MINUTE_LABEL_HORIZONS, label_query
from .sampling import (
    DEFAULT_RECALL_TOP_K,
    RECALL_TARGET_SEMANTICS,
    audit_candidate_recall,
    audit_candidate_recall_files,
)
from .source import register_source_views, resolve_source_snapshot

STAGE_ONE_IMPLEMENTATION_REVISION = "2026-08-29-1"
STAGE_ONE_OUTCOME_SCHEMA = "quantlab.minute_v2_complete_outcomes/1"
STAGE_ONE_REPORT_SCHEMA = "quantlab.minute_v2_stage_one_audit/1"
SEGMENT_REPORT_SCHEMA = "quantlab.minute_v2_candidate_recall_segments/1"
STAGE_ONE_TARGETS = (
    "label_net_return",
    "label_return_5m",
    "label_return_15m",
    "label_return_30m",
    "label_return_60m",
)


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _projection(alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    return ",".join(f'{prefix}"{name}"' for name in LABEL_COLUMNS)


def _file_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _read_object(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = read_json(path)
    except (DataContractError, OSError, TypeError, ValueError) as exc:
        raise MinuteV2Error(f"{error}:{path}") from exc
    if not isinstance(value, dict):
        raise MinuteV2Error(f"{error}:{path}")
    return value


def _default_output_directory(month_manifest: Path, selected_dates: list[str]) -> Path:
    month_directory = month_manifest.parent
    if len(month_directory.parents) < 3:
        raise MinuteV2Error(f"minute_v2_stage_one_month_path_invalid:{month_manifest}")
    dataset_root = month_directory.parents[2]
    selection = (
        f"date={selected_dates[0]}"
        if len(selected_dates) == 1
        else f"year={month_directory.parent.name.split('=', 1)[-1]}_month="
        f"{month_directory.name.split('=', 1)[-1]}"
    )
    return (dataset_root / "audits" / "stage_one" / selection).resolve()


def _stage_spec(
    month_manifest: Path,
    month: dict[str, Any],
    *,
    top_k: tuple[int, ...],
) -> dict[str, Any]:
    artifacts = month.get("artifacts")
    if not isinstance(artifacts, dict):
        raise MinuteV2Error("minute_v2_stage_one_month_artifacts_missing")
    required = ("base", "events", "labels")
    if any(not isinstance(artifacts.get(name), dict) for name in required):
        raise MinuteV2Error("minute_v2_stage_one_month_artifacts_missing")
    return {
        "schema": "quantlab.minute_v2_stage_one_spec/1",
        "implementation_revision": STAGE_ONE_IMPLEMENTATION_REVISION,
        "month_build_implementation_revision": BUILD_IMPLEMENTATION_REVISION,
        "month_manifest": str(month_manifest),
        "month_manifest_sha256": sha256_file(month_manifest),
        "base_sha256": artifacts["base"].get("sha256"),
        "events_sha256": artifacts["events"].get("sha256"),
        "candidate_labels_sha256": artifacts["labels"].get("sha256"),
        "selected_dates": list(month.get("date_selection", {}).get("selected_dates", [])),
        "targets": list(STAGE_ONE_TARGETS),
        "top_k": list(top_k),
    }


def _validate_active_source(month: dict[str, Any]) -> Any:
    source = month.get("source")
    build_spec = month.get("build_spec")
    if not isinstance(source, dict) or not isinstance(build_spec, dict):
        raise MinuteV2Error("minute_v2_stage_one_source_provenance_missing")
    workspace = source.get("workspace_root")
    if not isinstance(workspace, str):
        raise MinuteV2Error("minute_v2_stage_one_source_provenance_missing")
    snapshot = resolve_source_snapshot(workspace)
    expected_ids = source.get("dataset_ids")
    if snapshot.dataset_ids != expected_ids:
        raise MinuteV2Error("minute_v2_stage_one_active_source_dataset_mismatch")
    expected_hashes = build_spec.get("source_manifest_sha256")
    if not isinstance(expected_hashes, dict):
        raise MinuteV2Error("minute_v2_stage_one_source_manifest_hashes_missing")
    current_hashes = {
        name: sha256_file(Path(path)) for name, path in snapshot.manifests.items()
    }
    if current_hashes != expected_hashes:
        raise MinuteV2Error("minute_v2_stage_one_active_source_manifest_mismatch")
    return snapshot


def _support_path(month: dict[str, Any], *, stem: str) -> Path:
    support = month.get("support_cache")
    if not isinstance(support, dict) or not isinstance(support.get("directory"), str):
        raise MinuteV2Error("minute_v2_stage_one_support_cache_missing")
    directory = Path(support["directory"]).resolve()
    target = directory / f"{stem}.parquet"
    metadata_path = target.with_suffix(".json")
    if not target.is_file() or not metadata_path.is_file():
        raise MinuteV2Error(f"minute_v2_stage_one_support_artifact_missing:{target}")
    metadata = _read_object(
        metadata_path,
        error="minute_v2_stage_one_support_metadata_invalid",
    )
    if not _artifact_matches(target, metadata.get("artifact"), base_directory=directory):
        raise MinuteV2Error(f"minute_v2_stage_one_support_artifact_changed:{target}")
    return target


def _valid_label_part(path: Path, *, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        parquet = pq.ParquetFile(path)
        return (
            int(parquet.metadata.num_rows) == int(expected_rows)
            and not set(LABEL_COLUMNS).difference(parquet.schema_arrow.names)
        )
    except Exception:
        return False


def _complete_outcome_profile(connection: Any, outcome_path: Path) -> dict[str, Any]:
    scan = f"read_parquet({_sql_literal(outcome_path)})"
    observed_columns = [
        "entry_executable",
        "label_observed",
        *(f"label_{horizon}m_observed" for horizon in MINUTE_LABEL_HORIZONS),
        *(f"label_session_{horizon}m_observed" for horizon in MINUTE_LABEL_HORIZONS),
    ]
    observed_values = connection.execute(
        "SELECT "
        + ",".join(
            f"count(*) FILTER (WHERE COALESCE(\"{name}\", FALSE))" for name in observed_columns
        )
        + f" FROM {scan}"
    ).fetchone()
    finite_targets = connection.execute(
        "SELECT "
        + ",".join(
            f"count(*) FILTER (WHERE isfinite(TRY_CAST(\"{name}\" AS DOUBLE)))"
            for name in STAGE_ONE_TARGETS
        )
        + f" FROM {scan}"
    ).fetchone()
    invalid_reason_columns = [
        "entry_unfilled_reason",
        *(f"label_{horizon}m_invalid_reason" for horizon in MINUTE_LABEL_HORIZONS),
    ]
    invalid_reason_counts: dict[str, dict[str, int]] = {}
    for name in invalid_reason_columns:
        rows = connection.execute(
            f'SELECT COALESCE(NULLIF("{name}", \'\'), \'observed_or_unspecified\') AS reason,'
            f"count(*) AS rows FROM {scan} GROUP BY reason ORDER BY rows DESC, reason"
        ).fetchall()
        invalid_reason_counts[name] = {str(reason): int(count) for reason, count in rows}
    return {
        "rows": int(pq.ParquetFile(outcome_path).metadata.num_rows),
        "observed_rows": {
            name: int(value or 0)
            for name, value in zip(observed_columns, observed_values or (), strict=True)
        },
        "finite_target_rows": {
            name: int(value or 0)
            for name, value in zip(STAGE_ONE_TARGETS, finite_targets or (), strict=True)
        },
        "invalid_reason_counts": invalid_reason_counts,
    }


def build_complete_outcomes(
    month_manifest: str | Path,
    *,
    output_directory: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize labels for every retained base key in a separate audit bundle."""

    manifest_path = Path(month_manifest).resolve()
    verify_month(manifest_path)
    month = _read_object(manifest_path, error="minute_v2_stage_one_manifest_invalid")
    selected_dates = list(month.get("date_selection", {}).get("selected_dates", []))
    if not selected_dates:
        raise MinuteV2Error("minute_v2_stage_one_selected_dates_missing")
    output = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _default_output_directory(manifest_path, selected_dates)
    )
    output.mkdir(parents=True, exist_ok=True)
    outcome_path = output / "complete_outcomes.parquet"
    outcome_manifest_path = output / "complete_outcomes_manifest.json"
    spec = _stage_spec(manifest_path, month, top_k=DEFAULT_RECALL_TOP_K)
    if not force and outcome_manifest_path.is_file():
        existing = _read_object(
            outcome_manifest_path,
            error="minute_v2_stage_one_outcome_manifest_invalid",
        )
        artifact = existing.get("artifact")
        if existing.get("spec") == spec and _artifact_matches(
            outcome_path,
            artifact,
            base_directory=output,
        ):
            return existing
        raise MinuteV2Error(
            f"minute_v2_stage_one_existing_outcome_requires_force:{outcome_manifest_path}"
        )

    snapshot = _validate_active_source(month)
    config_values = month.get("config")
    if not isinstance(config_values, dict):
        raise MinuteV2Error("minute_v2_stage_one_config_missing")
    config = MinuteV2Config(**config_values)
    config.validate()
    artifacts = month["artifacts"]
    base_path = Path(str(artifacts["base"]["path"])).resolve()
    event_path = Path(str(artifacts["events"]["path"])).resolve()
    candidate_label_path = Path(str(artifacts["labels"]["path"])).resolve()
    start_date = str(month["start_date"])
    end_date = str(month["end_date"])
    extended_end = str(month["extended_label_end_date"])
    label_stock_days_path = _support_path(
        month,
        stem=f"label_stock_days__{start_date}__{extended_end}",
    )
    calendar_path = _support_path(
        month,
        stem=f"calendar_dates__{start_date}__{extended_end}",
    )

    runtime = output / "_runtime"
    parts = output / "_parts"
    if force:
        _safe_clean_generated(runtime, output, expected_name="_runtime")
        _safe_clean_generated(parts, output, expected_name="_parts")
    runtime.mkdir(parents=True, exist_ok=True)
    parts.mkdir(parents=True, exist_ok=True)
    parts_spec_path = parts / "spec.json"
    if parts_spec_path.is_file():
        parts_spec = _read_object(
            parts_spec_path,
            error="minute_v2_stage_one_parts_spec_invalid",
        )
        if parts_spec != spec:
            _safe_clean_generated(parts, output, expected_name="_parts")
            parts.mkdir(parents=True, exist_ok=True)
    write_json(parts_spec_path, spec)

    connection = open_guarded_duckdb(
        ":memory:",
        temp_directory=runtime,
        threads=config.duckdb_threads,
        floor_bytes=int(config.memory_floor_gib * GIB),
        minimum_limit_bytes=256 * MIB,
    )
    requested_memory = config.duckdb_memory_limit_gib
    effective_memory_limit_bytes = int(connection.settings.memory_limit_bytes)
    if not (
        isinstance(requested_memory, str)
        and requested_memory.strip().lower() == "auto"
    ):
        effective_memory_limit_bytes = min(
            effective_memory_limit_bytes,
            int(float(requested_memory) * GIB),
        )
    connection.execute(f"SET memory_limit='{effective_memory_limit_bytes}B'")
    completed = False
    try:
        register_source_views(
            connection,
            snapshot,
            start_date=start_date,
            end_date=end_date,
            minute_history_start_date=str(month["minute_history_start_date"]),
            minute_end_date=extended_end,
        )
        connection.execute(
            "CREATE OR REPLACE TEMP VIEW stage_base AS SELECT symbol,trade_date,bar_time "
            f"FROM read_parquet({_sql_literal(base_path)})"
        )
        connection.execute(
            "CREATE OR REPLACE TEMP VIEW stage_candidates AS SELECT symbol,trade_date,bar_time "
            f"FROM read_parquet({_sql_literal(event_path)})"
        )
        connection.execute(
            "CREATE OR REPLACE TEMP VIEW label_stock_days AS SELECT * "
            f"FROM read_parquet({_sql_literal(label_stock_days_path)})"
        )
        connection.execute(
            "CREATE OR REPLACE TEMP VIEW calendar_dates AS SELECT * "
            f"FROM read_parquet({_sql_literal(calendar_path)})"
        )
        missing_rows = int(
            connection.execute(
                "SELECT count(*) FROM stage_base b LEFT JOIN stage_candidates c "
                "USING(symbol,trade_date,bar_time) WHERE c.symbol IS NULL"
            ).fetchone()[0]
        )
        part_paths: list[Path] = []
        bucket_rows: list[dict[str, Any]] = []
        for bucket in range(LABEL_BUCKET_COUNT):
            expected_rows = int(
                connection.execute(
                    "SELECT count(*) FROM stage_base b LEFT JOIN stage_candidates c "
                    "USING(symbol,trade_date,bar_time) WHERE c.symbol IS NULL "
                    f"AND hash(b.symbol)%{LABEL_BUCKET_COUNT}={bucket}"
                ).fetchone()[0]
            )
            part = parts / f"missing_labels_{bucket:02d}_of_{LABEL_BUCKET_COUNT:02d}.parquet"
            reused = _valid_label_part(part, expected_rows=expected_rows)
            if not reused:
                connection.execute(
                    "CREATE OR REPLACE TEMP VIEW bucket_events AS "
                    "SELECT b.symbol,b.trade_date,b.bar_time FROM stage_base b "
                    "LEFT JOIN stage_candidates c USING(symbol,trade_date,bar_time) "
                    "WHERE c.symbol IS NULL "
                    f"AND hash(b.symbol)%{LABEL_BUCKET_COUNT}={bucket}"
                )
                connection.execute(
                    "CREATE OR REPLACE TEMP VIEW bucket_bars AS SELECT * "
                    "FROM minute_bars_extended "
                    f"WHERE hash(symbol)%{LABEL_BUCKET_COUNT}={bucket}"
                )
                _copy_query(
                    connection,
                    label_query(
                        event_view="bucket_events",
                        target_bars_view="minute_bars",
                        extended_bars_view="bucket_bars",
                        stock_days_view="label_stock_days",
                        calendar_view="calendar_dates",
                        config=config,
                    ),
                    part,
                    compression="SNAPPY",
                )
                if not _valid_label_part(part, expected_rows=expected_rows):
                    raise MinuteV2Error(
                        f"minute_v2_stage_one_bucket_row_mismatch:{bucket}:{expected_rows}"
                    )
            part_paths.append(part)
            bucket_rows.append(
                {"bucket": bucket, "rows": expected_rows, "reused": reused}
            )
        if sum(item["rows"] for item in bucket_rows) != missing_rows:
            raise MinuteV2Error("minute_v2_stage_one_missing_bucket_rows_incomplete")

        missing_scan = _parquet_scan(part_paths)
        candidate_projection = _projection("candidate_labels")
        missing_projection = _projection("missing_labels")
        _copy_query(
            connection,
            "SELECT * FROM ("
            f"SELECT {candidate_projection} FROM read_parquet({_sql_literal(candidate_label_path)}) "
            "AS candidate_labels UNION ALL BY NAME "
            f"SELECT {missing_projection} FROM {missing_scan} AS missing_labels"
            ") complete_labels ORDER BY trade_date,bar_time,symbol",
            outcome_path,
            compression="ZSTD",
        )

        outcome_rows = int(pq.ParquetFile(outcome_path).metadata.num_rows)
        base_rows = int(pq.ParquetFile(base_path).metadata.num_rows)
        candidate_rows = int(pq.ParquetFile(candidate_label_path).metadata.num_rows)
        if outcome_rows != base_rows or candidate_rows + missing_rows != base_rows:
            raise MinuteV2Error(
                "minute_v2_stage_one_complete_row_mismatch:"
                f"{base_rows}:{candidate_rows}:{missing_rows}:{outcome_rows}"
            )
        outcome_scan = f"read_parquet({_sql_literal(outcome_path)})"
        duplicate_keys = int(
            connection.execute(
                "SELECT count(*) FROM (SELECT symbol,trade_date,bar_time,count(*) AS row_count "
                f"FROM {outcome_scan} GROUP BY ALL HAVING row_count>1)"
            ).fetchone()[0]
        )
        null_keys = int(
            connection.execute(
                f"SELECT count(*) FROM {outcome_scan} WHERE symbol IS NULL "
                "OR trade_date IS NULL OR bar_time IS NULL"
            ).fetchone()[0]
        )
        missing_keys = int(
            connection.execute(
                "SELECT count(*) FROM stage_base b LEFT JOIN "
                f"{outcome_scan} o USING(symbol,trade_date,bar_time) WHERE o.symbol IS NULL"
            ).fetchone()[0]
        )
        extra_keys = int(
            connection.execute(
                f"SELECT count(*) FROM {outcome_scan} o LEFT JOIN stage_base b "
                "USING(symbol,trade_date,bar_time) WHERE b.symbol IS NULL"
            ).fetchone()[0]
        )
        if duplicate_keys or null_keys or missing_keys or extra_keys:
            raise MinuteV2Error(
                "minute_v2_stage_one_complete_key_contract_invalid:"
                f"{duplicate_keys}:{null_keys}:{missing_keys}:{extra_keys}"
            )
        explicit_columns = _projection()
        candidate_differences = int(
            connection.execute(
                "WITH existing AS ("
                f"SELECT {explicit_columns} FROM read_parquet({_sql_literal(candidate_label_path)})),"
                "complete_candidates AS ("
                f"SELECT {_projection('o')} FROM {outcome_scan} o JOIN stage_candidates c "
                "USING(symbol,trade_date,bar_time)),differences AS ("
                "(SELECT * FROM existing EXCEPT ALL SELECT * FROM complete_candidates) "
                "UNION ALL (SELECT * FROM complete_candidates EXCEPT ALL SELECT * FROM existing)) "
                "SELECT count(*) FROM differences"
            ).fetchone()[0]
        )
        if candidate_differences:
            raise MinuteV2Error(
                f"minute_v2_stage_one_candidate_labels_changed:{candidate_differences}"
            )
        profile = _complete_outcome_profile(connection, outcome_path)
        result = {
            "schema": STAGE_ONE_OUTCOME_SCHEMA,
            "status": "ok",
            "spec": spec,
            "artifact": _artifact(outcome_path),
            "verification": {
                "base_rows": base_rows,
                "candidate_label_rows_reused": candidate_rows,
                "non_candidate_label_rows_built": missing_rows,
                "complete_outcome_rows": outcome_rows,
                "duplicate_keys": duplicate_keys,
                "null_keys": null_keys,
                "missing_base_keys": missing_keys,
                "extra_outcome_keys": extra_keys,
                "candidate_label_differences": candidate_differences,
                "missing_label_column_count": len(
                    set(LABEL_COLUMNS).difference(
                        pq.ParquetFile(outcome_path).schema_arrow.names
                    )
                ),
            },
            "profile": profile,
            "build": {
                "label_bucket_count": LABEL_BUCKET_COUNT,
                "bucket_rows": bucket_rows,
                "effective_duckdb_memory_limit_bytes": effective_memory_limit_bytes,
                "candidate_labels_reused": True,
                "storage": "separate audit artifact; source month files unchanged",
            },
        }
        write_json(outcome_manifest_path, result)
        completed = True
        return result
    finally:
        connection.close()
        _safe_clean_generated(runtime, output, expected_name="_runtime")
        if completed:
            _safe_clean_generated(parts, output, expected_name="_parts")


def _time_bucket(bar_time: pd.Series) -> pd.Series:
    value = bar_time.astype(str)
    conditions = [
        value.le("100000000"),
        value.between("100100000", "110000000"),
        value.between("110100000", "112900000"),
        value.between("130100000", "140000000"),
        value.ge("140100000"),
    ]
    labels = [
        "opening_0931_1000",
        "mid_morning_1001_1100",
        "late_morning_1101_1129",
        "early_afternoon_1301_1400",
        "late_afternoon_1401_1455",
    ]
    return pd.Series(np.select(conditions, labels, default="outside_contract"), index=value.index)


def _segmented_recall(
    base_path: Path,
    candidate_path: Path,
    outcome_path: Path,
    *,
    target: str,
    top_k: tuple[int, ...],
) -> dict[str, Any]:
    context_columns = [
        *KEY_COLUMNS,
        "market_breadth_positive",
        "market_return_5m_mean",
    ]
    base = pd.read_parquet(base_path, columns=context_columns)
    candidates = pd.read_parquet(candidate_path, columns=list(KEY_COLUMNS))
    outcomes = pd.read_parquet(outcome_path, columns=[*KEY_COLUMNS, target])
    group_columns = ["trade_date", "bar_time"]
    group_consistency = base.groupby(group_columns, sort=False)[
        ["market_breadth_positive", "market_return_5m_mean"]
    ].nunique(dropna=False)
    inconsistent = {
        name: int((group_consistency[name] > 1).sum())
        for name in group_consistency.columns
    }
    if any(inconsistent.values()):
        raise MinuteV2Error(
            "minute_v2_stage_one_market_context_inconsistent:"
            + ":".join(f"{name}={rows}" for name, rows in inconsistent.items())
        )
    base["time_bucket"] = _time_bucket(base["bar_time"])
    breadth = pd.to_numeric(base["market_breadth_positive"], errors="coerce")
    base["market_breadth_state"] = np.select(
        [breadth.le(0.45), breadth.ge(0.55), breadth.notna()],
        ["risk_off_breadth_le_045", "risk_on_breadth_ge_055", "mixed_breadth"],
        default="missing_breadth",
    )
    trend = pd.to_numeric(base["market_return_5m_mean"], errors="coerce")
    base["market_trend_state"] = np.select(
        [trend.le(-0.0005), trend.ge(0.0005), trend.notna()],
        ["down_return_le_minus_5bp", "up_return_ge_5bp", "flat_between_5bp"],
        default="missing_return",
    )
    enriched_outcomes = base.loc[:, [*KEY_COLUMNS, "time_bucket", "market_breadth_state", "market_trend_state"]].merge(
        outcomes,
        how="left",
        on=list(KEY_COLUMNS),
        validate="one_to_one",
    )
    candidate_context = candidates.merge(
        base.loc[:, [*KEY_COLUMNS, "time_bucket", "market_breadth_state", "market_trend_state"]],
        how="left",
        on=list(KEY_COLUMNS),
        validate="one_to_one",
    )
    dimensions: dict[str, dict[str, Any]] = {}
    for dimension in ("time_bucket", "market_breadth_state", "market_trend_state"):
        reports: dict[str, Any] = {}
        for segment in sorted(str(value) for value in base[dimension].unique()):
            segment_base = base.loc[base[dimension].astype(str).eq(segment), list(KEY_COLUMNS)]
            segment_outcomes = enriched_outcomes.loc[
                enriched_outcomes[dimension].astype(str).eq(segment),
                [*KEY_COLUMNS, target],
            ]
            segment_candidates = candidate_context.loc[
                candidate_context[dimension].astype(str).eq(segment),
                list(KEY_COLUMNS),
            ]
            finite = pd.to_numeric(segment_outcomes[target], errors="coerce")
            if not np.isfinite(finite.to_numpy(dtype=float)).any():
                reports[segment] = {
                    "status": "no_finite_outcomes",
                    "base_rows": int(len(segment_base)),
                }
                continue
            reports[segment] = audit_candidate_recall(
                segment_base,
                segment_candidates,
                segment_outcomes,
                target=target,
                top_k=top_k,
            )
        dimensions[dimension] = reports
    return {
        "schema": SEGMENT_REPORT_SCHEMA,
        "target": target,
        "target_semantics": RECALL_TARGET_SEMANTICS,
        "segment_features_are_causal": True,
        "definitions": {
            "time_bucket": "fixed exchange-time intervals",
            "market_breadth_state": (
                "causal market_breadth_positive: risk-off <=0.45, risk-on >=0.55, mixed otherwise"
            ),
            "market_trend_state": (
                "causal market_return_5m_mean: down <=-5bp, up >=5bp, flat otherwise"
            ),
        },
        "market_context_inconsistent_group_counts": inconsistent,
        "dimensions": dimensions,
    }


def run_stage_one_audit(
    month_manifest: str | Path,
    *,
    output_directory: str | Path | None = None,
    top_k: tuple[int, ...] = DEFAULT_RECALL_TOP_K,
    force: bool = False,
) -> dict[str, Any]:
    """Build complete labels and all stage-one candidate-gate reports."""

    requested_k = tuple(sorted({int(value) for value in top_k if int(value) > 0}))
    if not requested_k:
        raise MinuteV2Error("minute_v2_recall_top_k_invalid")
    manifest_path = Path(month_manifest).resolve()
    month = _read_object(manifest_path, error="minute_v2_stage_one_manifest_invalid")
    selected_dates = list(month.get("date_selection", {}).get("selected_dates", []))
    if not selected_dates:
        raise MinuteV2Error("minute_v2_stage_one_selected_dates_missing")
    output = (
        Path(output_directory).resolve()
        if output_directory is not None
        else _default_output_directory(manifest_path, selected_dates)
    )
    outcome_result = build_complete_outcomes(
        manifest_path,
        output_directory=output,
        force=force,
    )
    artifacts = month["artifacts"]
    base_path = Path(str(artifacts["base"]["path"])).resolve()
    event_path = Path(str(artifacts["events"]["path"])).resolve()
    outcome_path = Path(str(outcome_result["artifact"]["path"])).resolve()
    recall_reports: dict[str, Any] = {}
    report_artifacts: dict[str, Any] = {}
    for target in STAGE_ONE_TARGETS:
        report = audit_candidate_recall_files(
            base_path,
            event_path,
            outcome_path,
            target=target,
            top_k=requested_k,
        )
        report_path = output / f"candidate_recall_{target}.json"
        write_json(report_path, report)
        recall_reports[target] = report
        report_artifacts[report_path.name] = _file_artifact(report_path)
    segments = _segmented_recall(
        base_path,
        event_path,
        outcome_path,
        target="label_net_return",
        top_k=requested_k,
    )
    segment_path = output / "candidate_recall_label_net_return_segments.json"
    write_json(segment_path, segments)
    report_artifacts[segment_path.name] = _file_artifact(segment_path)
    final_spec = _stage_spec(manifest_path, month, top_k=requested_k)
    result = {
        "schema": STAGE_ONE_REPORT_SCHEMA,
        "status": "ok",
        "spec": final_spec,
        "scope": {
            "selected_dates": selected_dates,
            "base_grain": "one stock x one causal decision minute",
            "decision_group_grain": "trade_date x bar_time",
            "profit_claim_allowed": False,
            "reason": (
                "single-day gate audit validates coverage and selection behavior; "
                "it is not an out-of-sample strategy backtest"
            ),
        },
        "complete_outcomes": outcome_result,
        "candidate_recall": recall_reports,
        "segmented_label_net_return_recall": segments,
        "artifacts": {
            "complete_outcomes": outcome_result["artifact"],
            **report_artifacts,
        },
    }
    manifest_output = output / "manifest.json"
    write_json(manifest_output, result)
    result["manifest"] = str(manifest_output)
    return result


__all__ = [
    "SEGMENT_REPORT_SCHEMA",
    "STAGE_ONE_IMPLEMENTATION_REVISION",
    "STAGE_ONE_OUTCOME_SCHEMA",
    "STAGE_ONE_REPORT_SCHEMA",
    "STAGE_ONE_TARGETS",
    "build_complete_outcomes",
    "run_stage_one_audit",
]
