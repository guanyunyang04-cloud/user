from __future__ import annotations

"""Compact the current 5-minute table into one Parquet file per year."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq

from quant_data_platform.core.json_io import json_safe, read_json
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.manifest import atomic_write_json
from quant_data_platform.qdp_v2.repair import (
    _sql_literal,
    mutate_active_shards_from_parquet,
    resolve_active_domain,
)


DOMAIN = "market_intraday_5m"
DATA_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "adjusted_flag",
)


class QdpCompactError(RuntimeError):
    pass


def plan_compact(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    context = resolve_active_domain(DOMAIN, workspace_root=workspace)
    years = _manifest_years(context.manifest.start_date, context.manifest.end_date)
    return {
        "status": "already_compact" if len(context.shard_paths) <= len(years) else "planned",
        "domain": DOMAIN,
        "dataset_id": context.dataset_id,
        "row_count": int(context.manifest.row_count),
        "start_date": context.manifest.start_date,
        "end_date": context.manifest.end_date,
        "current_shard_count": len(context.shard_paths),
        "target_shard_count": len(years),
        "logical_bytes": sum(path.stat().st_size for path in context.shard_paths),
        "layout": "natural_year",
    }


def run_compact(
    *,
    workspace_root: str | Path | None = None,
    keep_runtime: bool = False,
) -> dict[str, Any]:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    context = resolve_active_domain(DOMAIN, workspace_root=workspace)
    plan = plan_compact(workspace_root=workspace)
    if plan["status"] == "already_compact":
        return plan

    runtime = _runtime_dir(workspace, context.dataset_id)
    state_path = runtime / "state.json"
    prepared, source_summary = _load_prepared_state(
        state_path,
        workspace=workspace,
        dataset_id=context.dataset_id,
        source_paths=context.shard_paths,
    )
    runtime_reused = bool(prepared)
    if not prepared:
        prepared, source_summary = _build_year_files(
            context.shard_paths,
            runtime=runtime,
            workspace=workspace,
        )
        atomic_write_json(
            state_path,
            {
                "status": "prepared",
                "domain": DOMAIN,
                "dataset_id": context.dataset_id,
                "source_paths": [str(path.relative_to(workspace)) for path in context.shard_paths],
                "source_summary": source_summary,
                "prepared": [
                    {
                        "path": str(path.relative_to(workspace)),
                        "file_size": path.stat().st_size,
                        "row_count": int(pq.ParquetFile(path).metadata.num_rows),
                    }
                    for path in prepared
                ],
            },
        )

    expected_rows = sum(int(item["row_count"]) for item in source_summary.values())
    if expected_rows != int(context.manifest.row_count):
        raise QdpCompactError(
            f"compact_source_row_count_mismatch:{expected_rows}!={context.manifest.row_count}"
        )
    mutation = mutate_active_shards_from_parquet(
        DOMAIN,
        removals=context.shard_paths,
        appends=prepared,
        reason="compact current 5-minute table to one Parquet file per natural year",
        workspace_root=workspace,
        primary_keys_prevalidated=True,
    )
    current = resolve_active_domain(DOMAIN, workspace_root=workspace)
    if current.manifest.row_count != expected_rows or len(current.shard_paths) != len(prepared):
        raise QdpCompactError("compact_post_commit_layout_mismatch")

    result = {
        **plan,
        "status": "compacted",
        "new_shard_count": len(current.shard_paths),
        "deleted_shard_count": len(mutation.get("deleted_shard_paths", [])),
        "retained_old_shard_count": len(mutation.get("retained_old_shard_paths", [])),
        "manifest_row_count": int(current.manifest.row_count),
        "runtime_reused": runtime_reused,
    }
    if not keep_runtime:
        _remove_runtime(runtime, workspace=workspace)
        result["runtime_cleanup"] = "deleted_after_success"
    else:
        result["runtime_cleanup"] = "retained"
    return result


def _build_year_files(
    source_paths: Sequence[Path],
    *,
    runtime: Path,
    workspace: Path,
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    _remove_runtime(runtime, workspace=workspace)
    output_root = runtime / "prepared"
    spill = runtime / "spill"
    runtime.mkdir(parents=True, exist_ok=True)
    quoted_columns = ", ".join(f'"{item}"' for item in DATA_COLUMNS)
    source_sql = "[" + ",".join(_sql_literal(str(path)) for path in source_paths) + "]"
    with open_guarded_duckdb(temp_directory=spill, threads=4) as connection:
        source_summary = _year_summary(connection, source_paths)
        connection.execute(
            f"""
            COPY (
              SELECT {quoted_columns}, substr(cast(trade_date AS VARCHAR), 1, 4) AS qdp_year
              FROM read_parquet({source_sql}, union_by_name=false)
            ) TO {_sql_literal(str(output_root))} (
              FORMAT PARQUET,
              PARTITION_BY (qdp_year),
              COMPRESSION ZSTD,
              ROW_GROUP_SIZE 500000
            )
            """
        )
        prepared = sorted(output_root.rglob("*.parquet"))
        if not prepared:
            raise QdpCompactError("compact_prepared_files_missing")
        output_summary = _year_summary(connection, prepared)
    if output_summary != source_summary:
        raise QdpCompactError("compact_year_summary_mismatch")
    output_years = {
        path.parent.name.split("=", 1)[1]
        for path in prepared
        if path.parent.name.startswith("qdp_year=")
    }
    if len(prepared) != len(source_summary) or output_years != set(source_summary):
        raise QdpCompactError("compact_expected_one_file_per_year")
    return prepared, source_summary


def _year_summary(connection: Any, paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT substr(cast(trade_date AS VARCHAR), 1, 4) AS qdp_year,
               count(*) AS row_count,
               min(cast(trade_date AS VARCHAR)) AS start_date,
               max(cast(trade_date AS VARCHAR)) AS end_date,
               sum(try_cast(volume AS DOUBLE)) AS volume_sum,
               sum(try_cast(amount AS DOUBLE)) AS amount_sum
        FROM read_parquet(?, union_by_name=false)
        GROUP BY qdp_year ORDER BY qdp_year
        """,
        [[str(path) for path in paths]],
    ).fetchall()
    result = {
        str(year): {
            "row_count": int(row_count),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "volume_sum": float(volume_sum or 0.0),
            "amount_sum": float(amount_sum or 0.0),
        }
        for year, row_count, start_date, end_date, volume_sum, amount_sum in rows
    }
    if not result or any(len(year) != 4 or not year.isdigit() for year in result):
        raise QdpCompactError("compact_invalid_trade_date_year")
    return result


def _load_prepared_state(
    state_path: Path,
    *,
    workspace: Path,
    dataset_id: str,
    source_paths: Sequence[Path],
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    if not state_path.is_file():
        return [], {}
    state = read_json(state_path)
    expected_sources = [str(path.relative_to(workspace)) for path in source_paths]
    if (
        state.get("status") != "prepared"
        or state.get("dataset_id") != dataset_id
        or list(state.get("source_paths", []) or []) != expected_sources
    ):
        return [], {}
    prepared: list[Path] = []
    for item in list(state.get("prepared", []) or []):
        path = (workspace / str(item.get("path", ""))).resolve()
        path.relative_to(workspace)
        if not path.is_file():
            return [], {}
        parquet = pq.ParquetFile(path)
        if (
            path.stat().st_size != int(item.get("file_size", -1))
            or int(parquet.metadata.num_rows) != int(item.get("row_count", -1))
        ):
            return [], {}
        prepared.append(path)
    summary = {
        str(year): dict(values)
        for year, values in dict(state.get("source_summary", {}) or {}).items()
    }
    return prepared, summary


def _runtime_dir(workspace: Path, dataset_id: str) -> Path:
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in dataset_id)
    root = (
        workspace
        / "quant_data_platform"
        / "data"
        / "qdp_runtime"
        / "compact"
        / f"{DOMAIN}_{safe_id}"
    ).resolve()
    root.relative_to(workspace)
    return root


def _remove_runtime(path: Path, *, workspace: Path) -> None:
    resolved = path.resolve()
    runtime_root = (
        workspace / "quant_data_platform" / "data" / "qdp_runtime" / "compact"
    ).resolve()
    resolved.relative_to(runtime_root)
    if resolved.exists():
        shutil.rmtree(resolved)


def _manifest_years(start_date: str, end_date: str) -> tuple[str, ...]:
    if not start_date or not end_date:
        raise QdpCompactError("compact_manifest_date_range_missing")
    start = int(str(start_date)[:4])
    end = int(str(end_date)[:4])
    if start > end:
        raise QdpCompactError("compact_manifest_date_range_invalid")
    return tuple(str(year) for year in range(start, end + 1))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdp compact",
        description="Compact the current 5-minute table in place by natural year.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-runtime", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    kwargs = {"workspace_root": str(args.workspace_root or "") or None}
    result = plan_compact(**kwargs) if args.dry_run else run_compact(
        **kwargs, keep_runtime=bool(args.keep_runtime)
    )
    if args.json:
        print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0 if result.get("status") in {"planned", "already_compact", "compacted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
