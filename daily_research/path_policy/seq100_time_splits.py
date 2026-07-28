from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
INDEX_COLUMNS = {
    "split",
    "trade_date",
    "date_idx",
    "symbol_idx",
}
DEVELOPMENT_COLUMNS = {
    "year",
    "symbol",
    "entry_trade_date",
    "entry_filled",
    "label_valid",
    "price_label_valid",
    "va_aux_valid",
}


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def _read_index(
    manifest: Mapping[str, Any],
    key: str,
    frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if frame is not None:
        result = frame.copy()
    else:
        path = _workspace_path(str(manifest.get(key, "") or "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result = pd.read_parquet(path)
    missing = sorted(INDEX_COLUMNS.difference(result.columns))
    if missing:
        raise ValueError(f"{key} is missing required columns: {missing}")
    if result.empty:
        raise ValueError(f"{key} is empty")
    if bool(result.duplicated(["date_idx", "symbol_idx"]).any()):
        raise ValueError(f"{key} contains duplicate date/symbol rows")
    result["split"] = result["split"].astype(str)
    return result


def _validate_normalization(
    manifest: Mapping[str, Any],
    *,
    allowed_scopes: set[str],
    expected_cutoff: str,
    evaluation_count_key: str,
) -> dict[str, Any]:
    normalization = dict(manifest.get("normalization", {}) or {})
    if str(normalization.get("fit_scope", "")) not in allowed_scopes:
        raise ValueError("normalization must be fitted only on pre-evaluation dates")
    if str(normalization.get("fit_date_end_exclusive", "")) != expected_cutoff:
        raise ValueError("normalization cutoff does not match evaluation start")
    if int(normalization.get(evaluation_count_key, -1)) != 0:
        raise ValueError("normalization includes evaluation-period feature dates")
    for name, raw in dict(manifest.get("feature_channels", {}) or {}).items():
        columns = list(dict(raw).get("columns", []) or [])
        statistics = dict(normalization.get(name, {}) or {})
        if columns and (
            len(list(statistics.get("mean", []) or [])) != len(columns)
            or len(list(statistics.get("std", []) or [])) != len(columns)
        ):
            raise ValueError(f"normalization width does not match feature channel {name}")
    return normalization


def validate_fixed_oos_split(
    manifest: Mapping[str, Any],
    *,
    sample_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    frame = _read_index(manifest, "sample_index_path", sample_frame)
    if set(frame["split"].unique()) != {"train", "oos"}:
        raise ValueError("fixed OOS split must contain train and oos rows")
    train = frame[frame["split"].eq("train")]
    evaluation = frame[frame["split"].eq("oos")]
    forward_days = int(manifest.get("forward_days", 0) or 0)
    if train.empty or evaluation.empty or forward_days <= 0:
        raise ValueError("fixed OOS split has an empty side or invalid horizon")
    evaluation_start = int(pd.to_numeric(evaluation["date_idx"]).min())
    train_dates = pd.to_numeric(train["date_idx"]).to_numpy(dtype=np.int64)
    overlap_count = int(np.sum(train_dates + forward_days >= evaluation_start))
    if overlap_count:
        raise ValueError(f"training labels overlap OOS rows: {overlap_count}")
    years = sorted({int(str(value)[:4]) for value in evaluation["trade_date"]})
    split = dict(manifest.get("purged_walkforward", {}) or {})
    if len(years) != 1 or not split:
        raise ValueError("fixed OOS metadata must describe one evaluation year")
    if int(split.get("oos_year", 0)) != years[0]:
        raise ValueError("OOS year does not match the sample index")
    if int(split.get("oos_start_date_idx", -1)) != evaluation_start:
        raise ValueError("OOS start does not match the sample index")
    if int(split.get("label_overlap_count", -1)) != 0:
        raise ValueError("OOS metadata reports label overlap")
    _validate_normalization(
        manifest,
        allowed_scopes={"feature_dates_before_oos_start"},
        expected_cutoff=str(split.get("oos_start", "")),
        evaluation_count_key="oos_feature_date_count",
    )
    return {
        "mode": "fixed_oos",
        "forward_days": forward_days,
        "train_count": int(len(train)),
        "evaluation_count": int(len(evaluation)),
        "evaluation_year": years[0],
        "maximum_train_label_end_date_idx": int(
            np.max(train_dates + forward_days)
        ),
        "evaluation_start_date_idx": evaluation_start,
    }


def validate_development_split(
    manifest: Mapping[str, Any],
    *,
    sample_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    frame = _read_index(manifest, "sample_index_path", sample_frame)
    values = set(frame["split"].unique())
    if values != {"train", "development"}:
        raise ValueError("development split must contain train and development rows")
    train = frame[frame["split"].eq("train")]
    evaluation = frame[frame["split"].eq("development")]
    split = dict(manifest.get("development_walkforward", {}) or {})
    dependency_days = int(
        split.get(
            "training_label_dependency_days",
            split.get(
                "max_label_dependency_days",
                manifest.get("max_label_dependency_days", manifest.get("forward_days", 0)),
            ),
        )
        or 0
    )
    forward_days = int(manifest.get("forward_days", 0) or 0)
    if train.empty or evaluation.empty or dependency_days <= 0:
        raise ValueError("development split has an empty side or invalid dependency")
    if dependency_days != forward_days:
        raise ValueError("development training purge must equal forward_days")
    evaluation_start = int(pd.to_numeric(evaluation["date_idx"]).min())
    dependency_column = next(
        (
            name
            for name in (
                "training_label_end_date_idx",
                "label_end_date_idx",
                "dependency_end_date_idx",
            )
            if name in train.columns
        ),
        "",
    )
    if dependency_column:
        dependency_end = pd.to_numeric(
            train[dependency_column], errors="coerce"
        ).to_numpy(dtype=np.float64, na_value=np.nan)
        if not bool(np.isfinite(dependency_end).all()):
            raise ValueError(f"{dependency_column} contains non-finite values")
    else:
        dependency_end = (
            pd.to_numeric(train["date_idx"]).to_numpy(dtype=np.int64)
            + dependency_days
        )
    overlap_count = int(np.sum(dependency_end >= evaluation_start))
    if overlap_count:
        raise ValueError(f"training labels overlap development rows: {overlap_count}")
    declared_start = int(split.get("development_start_date_idx", -1))
    if declared_start != evaluation_start:
        raise ValueError("development start does not match the sample index")
    if int(split.get("label_dependency_overlap_count", -1)) != 0:
        raise ValueError("development metadata reports label overlap")
    expected_start = str(split.get("development_start", "") or "")
    _validate_normalization(
        manifest,
        allowed_scopes={"feature_dates_before_development_start"},
        expected_cutoff=expected_start,
        evaluation_count_key="development_feature_date_count",
    )
    return {
        "mode": "development",
        "forward_days": forward_days,
        "training_label_dependency_days": dependency_days,
        "execution_dependency_days": int(
            split.get(
                "execution_dependency_days",
                manifest.get("max_execution_dependency_days", dependency_days),
            )
        ),
        "train_count": int(len(train)),
        "evaluation_count": int(len(evaluation)),
        "evaluation_start_date_idx": evaluation_start,
        "maximum_train_label_end_date_idx": int(np.max(dependency_end)),
    }


def _parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _duckdb_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _fit_feature_normalization_before(
    metadata: Mapping[str, Any],
    *,
    end_date_idx_exclusive: int,
    date_chunk_size: int = 64,
) -> dict[str, list[float]]:
    path = _workspace_path(str(metadata["path"])).resolve()
    shape = tuple(int(value) for value in metadata["shape"])
    if len(shape) != 3 or str(metadata.get("dtype", "")) != "float32":
        raise ValueError(f"unsupported feature panel for normalization: {path}")
    cutoff = int(end_date_idx_exclusive)
    if cutoff <= 0 or cutoff > shape[0]:
        raise ValueError("normalization cutoff is outside the feature panel")
    panel = np.memmap(path, mode="r", dtype=np.float32, shape=shape)
    width = int(shape[2])
    count = np.zeros(width, dtype=np.int64)
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    for start in range(0, cutoff, max(1, int(date_chunk_size))):
        block = np.asarray(
            panel[start : min(cutoff, start + int(date_chunk_size))],
            dtype=np.float64,
        ).reshape(-1, width)
        finite = np.isfinite(block)
        count += finite.sum(axis=0, dtype=np.int64)
        clean = np.where(finite, block, 0.0)
        total += clean.sum(axis=0, dtype=np.float64)
        total_sq += np.square(clean).sum(axis=0, dtype=np.float64)
    mean = np.divide(total, count, out=np.zeros(width), where=count > 0)
    second = np.divide(total_sq, count, out=np.zeros(width), where=count > 0)
    std = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    std = np.where(np.isfinite(std) & (std > 1.0e-6), std, 1.0)
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_development_fold_view(
    *,
    source_manifest: str | Path,
    output_root: str | Path,
    development_year: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = _workspace_path(source_manifest).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if str(source.get("artifact_type", "")) != "qdp_v2_sequence_path_pack":
        raise ValueError("source manifest is not a sequence path pack")
    forward_days = int(source.get("forward_days", 0) or 0)
    execution_tail_days = int(source.get("execution_tail_days", 0) or 0)
    if forward_days <= 0 or execution_tail_days < 0:
        raise ValueError("source pack has invalid prediction or execution horizons")
    if int(source.get("max_label_dependency_days", forward_days)) != forward_days:
        raise ValueError("source label dependency differs from forward_days")
    execution_dependency_days = forward_days + execution_tail_days
    if int(
        source.get("max_execution_dependency_days", execution_dependency_days)
    ) != execution_dependency_days:
        raise ValueError("source execution dependency is inconsistent")

    source_sample_path = _workspace_path(str(source["sample_index_path"])).resolve()
    source_candidate_path = _workspace_path(
        str(source["candidate_index_path"])
    ).resolve()
    if not source_sample_path.is_file() or not source_candidate_path.is_file():
        raise FileNotFoundError("source sample or candidate index is missing")
    required = INDEX_COLUMNS | DEVELOPMENT_COLUMNS | {"sample_id"}
    if required.difference(_parquet_columns(source_sample_path)):
        raise ValueError("source sample index lacks required fold columns")
    if (INDEX_COLUMNS | DEVELOPMENT_COLUMNS).difference(
        _parquet_columns(source_candidate_path)
    ):
        raise ValueError("source candidate index lacks required fold columns")

    year = int(development_year)
    output = _workspace_path(output_root).resolve()
    index_dir = output / "indexes"
    view_dir = output / "views"
    sample_path = index_dir / f"development_{year}_purge{forward_days}.parquet"
    candidate_path = index_dir / f"candidates_{year}.parquet"
    view_path = view_dir / f"l35v2_pit_{year}.json"
    for target in (sample_path, candidate_path, view_path):
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        if target.exists():
            target.unlink()
    index_dir.mkdir(parents=True, exist_ok=True)
    view_dir.mkdir(parents=True, exist_ok=True)

    import duckdb  # type: ignore

    source_sample_sql = _duckdb_path(source_sample_path)
    source_candidate_sql = _duckdb_path(source_candidate_path)
    sample_output_sql = _duckdb_path(sample_path)
    candidate_output_sql = _duckdb_path(candidate_path)
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("set threads=2")
        connection.execute("set memory_limit='1GB'")
        candidate_count, development_start_idx, development_end_idx = connection.execute(
            f"""
            select count(*), min(date_idx), max(date_idx)
            from read_parquet({source_candidate_sql})
            where cast(year as integer) = {year}
            """
        ).fetchone()
        candidate_count = int(candidate_count or 0)
        if candidate_count <= 0:
            raise ValueError(f"source pack has no candidate rows for {year}")
        development_start_idx = int(development_start_idx)
        development_end_idx = int(development_end_idx)
        safe_train_signal_end_idx = development_start_idx - forward_days - 1
        if safe_train_signal_end_idx < 0:
            raise ValueError("development fold has no room for a strict purge")
        (
            train_count,
            supervised_count,
            purged_row_count,
            train_start_year,
        ) = connection.execute(
            f"""
            select
              count(*) filter (
                where cast(year as integer) < {year}
                  and cast(date_idx as bigint) <= {safe_train_signal_end_idx}
              ),
              count(*) filter (where cast(year as integer) = {year}),
              count(*) filter (
                where cast(year as integer) < {year}
                  and cast(date_idx as bigint) > {safe_train_signal_end_idx}
                  and cast(date_idx as bigint) < {development_start_idx}
              ),
              min(cast(year as integer)) filter (
                where cast(year as integer) < {year}
                  and cast(date_idx as bigint) <= {safe_train_signal_end_idx}
              )
            from read_parquet({source_sample_sql})
            """
        ).fetchone()
        train_count = int(train_count or 0)
        supervised_count = int(supervised_count or 0)
        purged_row_count = int(purged_row_count or 0)
        train_start_year = int(train_start_year or 0)
        if train_count <= 0 or supervised_count <= 0:
            raise ValueError("development fold has an empty supervised side")
        connection.execute(
            f"""
            copy (
              select
                sample_id,
                cast(split as varchar) as source_split,
                case when cast(year as integer) = {year}
                     then 'development' else 'train' end as split,
                cast(year as smallint) as year,
                cast(trade_date as varchar) as trade_date,
                cast(date_idx as bigint) as date_idx,
                cast(symbol_idx as integer) as symbol_idx,
                cast(symbol as varchar) as symbol,
                cast(entry_trade_date as varchar) as entry_trade_date,
                cast(entry_filled as boolean) as entry_filled,
                cast(label_valid as boolean) as label_valid,
                cast(price_label_valid as boolean) as price_label_valid,
                cast(va_aux_valid as boolean) as va_aux_valid,
                cast(date_idx as bigint) + {forward_days} as label_end_date_idx,
                cast(date_idx as bigint) + {forward_days}
                    as training_label_end_date_idx,
                cast(date_idx as bigint) + {execution_dependency_days}
                    as execution_dependency_end_date_idx
              from read_parquet({source_sample_sql})
              where (
                cast(year as integer) < {year}
                and cast(date_idx as bigint) <= {safe_train_signal_end_idx}
              ) or cast(year as integer) = {year}
              order by date_idx, symbol_idx
            ) to {sample_output_sql}
            (format parquet, compression zstd, row_group_size 250000)
            """
        )
        connection.execute(
            f"""
            copy (
              select
                row_number() over (order by date_idx, symbol_idx) - 1 as candidate_id,
                'development' as split,
                {year}::integer as year,
                cast(trade_date as varchar) as trade_date,
                cast(date_idx as bigint) as date_idx,
                cast(symbol_idx as integer) as symbol_idx,
                cast(symbol as varchar) as symbol,
                cast(entry_trade_date as varchar) as entry_trade_date,
                cast(entry_filled as boolean) as entry_filled,
                cast(label_valid as boolean) as label_valid,
                cast(price_label_valid as boolean) as price_label_valid,
                cast(va_aux_valid as boolean) as va_aux_valid,
                cast(split as varchar) as source_split
              from read_parquet({source_candidate_sql})
              where cast(year as integer) = {year}
              order by date_idx, symbol_idx
            ) to {candidate_output_sql}
            (format parquet, compression zstd, row_group_size 250000)
            """
        )
    finally:
        connection.close()

    date_values = [str(value) for value in source.get("date_values", [])]
    if development_end_idx + execution_dependency_days >= len(date_values):
        raise ValueError("source pack lacks the full execution tail for this fold")
    normalization: dict[str, Any] = {
        "fit_scope": "feature_dates_before_development_start",
        "fit_date_start": date_values[0],
        "fit_date_end": date_values[development_start_idx - 1],
        "fit_date_count": development_start_idx,
        "fit_date_end_exclusive": date_values[development_start_idx],
        "development_feature_date_count": 0,
    }
    for name, metadata in dict(source.get("feature_channels", {}) or {}).items():
        normalization[str(name)] = _fit_feature_normalization_before(
            dict(metadata),
            end_date_idx_exclusive=development_start_idx,
        )
    split = {
        "method": "expanding_train_development_walkforward",
        "split_roles": {"fit": "train", "evaluation": "development"},
        "train_start_year": train_start_year,
        "development_year": year,
        "development_start": date_values[development_start_idx],
        "development_end": date_values[development_end_idx],
        "development_start_date_idx": development_start_idx,
        "forward_days": forward_days,
        "execution_tail_days": execution_tail_days,
        "training_label_dependency_days": forward_days,
        "execution_dependency_days": execution_dependency_days,
        "safe_train_signal_end": date_values[safe_train_signal_end_idx],
        "max_train_dependency_end": date_values[
            safe_train_signal_end_idx + forward_days
        ],
        "purge_rule": "training_label_end_date_idx < development_start_date_idx",
        "purged_row_count": purged_row_count,
        "purged_signal_date_count": forward_days,
        "purged_signal_start": date_values[development_start_idx - forward_days],
        "purged_signal_end": date_values[development_start_idx - 1],
        "label_dependency_overlap_count": 0,
        "candidate_count": candidate_count,
        "supervised_development_count": supervised_count,
        "unsupervised_candidate_count": candidate_count - supervised_count,
    }
    view = dict(source)
    for key in (
        "research_contract",
        "development_contract",
        "source_view_provenance",
        "fold_training_contract",
        "development_fold_training_contract",
        "qdp_source_freshness_policy",
    ):
        view.pop(key, None)
    view.update(
        {
            "source_manifest_path": str(source_path),
            "sample_index_path": str(sample_path),
            "candidate_index_path": str(candidate_path),
            "sample_count": train_count + supervised_count,
            "sample_count_by_split": {
                "train": train_count,
                "development": supervised_count,
            },
            "candidate_count": candidate_count,
            "candidate_count_by_split": {"development": candidate_count},
            "normalization": normalization,
            "development_walkforward": split,
        }
    )
    semantics = dict(view.get("data_semantics", {}) or {})
    semantics["training_purge"] = {
        "days": forward_days,
        "uses_execution_tail": False,
        "execution_tail_days": execution_tail_days,
    }
    view["data_semantics"] = semantics
    _atomic_write_json(view_path, view)
    validate_development_split(view)
    return {
        "year": year,
        "view_path": str(view_path),
        "sample_index_path": str(sample_path),
        "candidate_index_path": str(candidate_path),
        "train_count": train_count,
        "development_supervised_count": supervised_count,
        "development_candidate_count": candidate_count,
        "purge_trading_days": forward_days,
        "execution_dependency_days": execution_dependency_days,
    }


def build_development_fold_views(
    *,
    source_manifest: str | Path,
    output_root: str | Path,
    years: Sequence[int],
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    return [
        build_development_fold_view(
            source_manifest=source_manifest,
            output_root=output_root,
            development_year=int(year),
            overwrite=overwrite,
        )
        for year in years
    ]


def _parse_years(value: str) -> list[int]:
    years: list[int] = []
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            years.extend(range(start, end + 1))
        else:
            years.append(int(item))
    return sorted(set(years))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build purged Seq100 development folds.")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = build_development_fold_views(
        source_manifest=args.source_manifest,
        output_root=args.output_root,
        years=_parse_years(args.years),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
