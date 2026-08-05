from __future__ import annotations

"""Derive the formal 2011-2025 research scope from the 2010 burn-in dataset."""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from daily_research.path_policy import seq100_quality_liquidity_data_prep as base

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_quality_liquidity_research_scope"
SOURCE_STUDY_ID = "seq100_quality_liquidity_data_prep"
DATA_HISTORY_START = "2010-01-01"
RESEARCH_START = "2011-01-01"
END_DATE = "2025-12-31"
BURN_IN_YEARS = (2010,)
RESEARCH_YEARS = tuple(range(2011, 2026))
OOS_YEARS = (2023, 2024, 2025)
FEATURE_FAMILIES = ("minute", "fundamental", "event")
EXPECTED_COMMON_SUPPORT_ROW_COUNT = 4_361_485
EXPECTED_FEATURE_COUNT = 518
EXPECTED_TRAINING_ROWS = {
    2023: 3_147_686,
    2024: 3_550_401,
    2025: 3_956_386,
}

DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_quality_liquidity_research_scope.json"
)
SOURCE_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_data_prep"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_research_scope"
)


class ResearchScopeError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResearchScopeError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _load_config(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    period = dict(payload.get("period", {}) or {})
    source = dict(payload.get("source_data_prep", {}) or {})
    burn_in = dict(payload.get("burn_in", {}) or {})
    training = dict(payload.get("training", {}) or {})
    if not str(payload.get("study_id", "")):
        raise ResearchScopeError("research_scope_study_id_changed")
    if not str(source.get("study_id", "")):
        raise ResearchScopeError("research_scope_source_study_changed")
    if (
        period.get("data_history_start") != DATA_HISTORY_START
        or period.get("research_start_date") != RESEARCH_START
        or period.get("end_date") != END_DATE
        or int(period.get("forbidden_year", 0)) != 2026
        or tuple(period.get("future_oos_prediction_years", ())) != OOS_YEARS
    ):
        raise ResearchScopeError("research_scope_date_boundary_changed")
    if (
        tuple(burn_in.get("years", ())) != BURN_IN_YEARS
        or burn_in.get("eligible_for_training") is not False
        or burn_in.get("eligible_for_evaluation") is not False
        or burn_in.get("eligible_for_atlas_statistics") is not False
        or burn_in.get("eligible_for_labels") is not False
        or burn_in.get("available_for_feature_history") is not True
    ):
        raise ResearchScopeError("research_scope_burn_in_semantics_changed")
    if training.get("performed") is not False:
        raise ResearchScopeError("research_scope_must_not_train")
    return payload


def _source_state(
    source_output_root: Path, *, expected_study_id: str = SOURCE_STUDY_ID
) -> dict[str, Any]:
    state = _read_json(source_output_root / "state.json")
    if state.get("study_id") != expected_study_id:
        raise ResearchScopeError("source_data_prep_study_mismatch")
    if state.get("status") != "completed":
        raise ResearchScopeError("source_data_prep_not_completed")
    if state.get("training_performed") is not False:
        raise ResearchScopeError("source_data_prep_trained_models")
    if dict(state.get("atlas", {}) or {}).get("status") != "completed":
        raise ResearchScopeError("source_data_prep_atlas_not_completed")
    return state


def _selected_records(
    source_state: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_membership = dict(source_state.get("membership_years", {}) or {})
    source_blocks = dict(source_state.get("feature_blocks", {}) or {})
    membership: dict[str, Any] = {}
    blocks: dict[str, Any] = {}
    for year in RESEARCH_YEARS:
        key = str(year)
        if key not in source_membership or key not in source_blocks:
            raise ResearchScopeError(f"source_year_missing:{year}")
        membership[key] = dict(source_membership[key])
        blocks[key] = {
            family: dict(source_blocks[key][family]) for family in FEATURE_FAMILIES
        }
    return membership, blocks


def _partition_input_fingerprint(
    *,
    config: Mapping[str, Any],
    source_state: Mapping[str, Any],
    membership: Mapping[str, Any],
    blocks: Mapping[str, Any],
) -> str:
    payload = {
        "config": config,
        "source_atlas_sha256": dict(source_state.get("atlas", {}) or {}).get(
            "manifest_sha256", ""
        ),
        "membership": {
            str(year): {
                "support_sha256": membership[str(year)].get("support_sha256", ""),
                "row_count": membership[str(year)].get("common_support_row_count", 0),
                "quality_support_sha256": membership[str(year)].get(
                    "quality_support_sha256", ""
                ),
                "quality_row_count": membership[str(year)].get(
                    "quality_support_row_count", 0
                ),
            }
            for year in RESEARCH_YEARS
        },
        "feature_blocks": {
            str(year): {
                family: {
                    "sha256": blocks[str(year)][family].get("sha256", ""),
                    "support_sha256": blocks[str(year)][family].get(
                        "support_sha256", ""
                    ),
                    "input_fingerprint": blocks[str(year)][family].get(
                        "input_fingerprint", ""
                    ),
                }
                for family in FEATURE_FAMILIES
            }
            for year in RESEARCH_YEARS
        },
    }
    serialized = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _validate_partition_files(
    membership: Mapping[str, Any], blocks: Mapping[str, Any]
) -> None:
    for year in RESEARCH_YEARS:
        key = str(year)
        support = dict(membership[key])
        support_path = Path(str(support["support_path"]))
        if not support_path.is_file():
            raise ResearchScopeError(f"support_partition_missing:{year}")
        support_rows = int(pq.ParquetFile(support_path).metadata.num_rows)
        expected_rows = int(support["common_support_row_count"])
        if support_rows != expected_rows:
            raise ResearchScopeError(f"support_partition_row_mismatch:{year}")
        for family in FEATURE_FAMILIES:
            record = dict(blocks[key][family])
            feature_path = Path(str(record["path"]))
            if not feature_path.is_file():
                raise ResearchScopeError(f"feature_partition_missing:{year}:{family}")
            if int(pq.ParquetFile(feature_path).metadata.num_rows) != expected_rows:
                raise ResearchScopeError(
                    f"feature_partition_row_mismatch:{year}:{family}"
                )
            if record.get("support_sha256") != support.get("support_sha256"):
                raise ResearchScopeError(
                    f"feature_partition_support_mismatch:{year}:{family}"
                )


def _support_semantics(
    *,
    output_root: Path,
    membership: Mapping[str, Any],
    path_key: str = "support_path",
) -> dict[str, Any]:
    paths = [Path(membership[str(year)][path_key]) for year in RESEARCH_YEARS]
    connection = base._connect(output_root)
    try:
        row = connection.execute(
            f"""
            SELECT count(*) AS row_count,
                   count(DISTINCT candidate_id) AS unique_candidate_count,
                   min(trade_date) AS min_date,
                   max(trade_date) AS max_date,
                   sum(CASE WHEN year<2011 OR trade_date<'2011-01-01'
                            THEN 1 ELSE 0 END) AS pre_scope_rows,
                   sum(CASE WHEN year>=2026 OR trade_date>='2026-01-01'
                            THEN 1 ELSE 0 END) AS forbidden_2026_rows,
                   sum(CASE WHEN year<>try_cast(substr(trade_date,1,4) AS INTEGER)
                            THEN 1 ELSE 0 END) AS year_date_mismatch_rows
            FROM {base._scan(paths)}
            """
        ).fetchone()
    finally:
        connection.close()
    return {
        "row_count": int(row[0]),
        "unique_candidate_count": int(row[1]),
        "min_date": str(row[2]),
        "max_date": str(row[3]),
        "pre_scope_rows": int(row[4] or 0),
        "forbidden_2026_rows": int(row[5] or 0),
        "year_date_mismatch_rows": int(row[6] or 0),
    }


def _rolling_oos_folds(membership: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for evaluation_year in OOS_YEARS:
        training_years = tuple(
            year for year in RESEARCH_YEARS if year < evaluation_year
        )
        folds.append(
            {
                "evaluation_year": evaluation_year,
                "training_years": list(training_years),
                "training_start_date": str(
                    membership[str(training_years[0])]["start_date"]
                ),
                "training_end_date_before_target_purge": str(
                    membership[str(training_years[-1])]["end_date"]
                ),
                "training_candidate_row_count_before_target_purge": sum(
                    int(membership[str(year)]["common_support_row_count"])
                    for year in training_years
                ),
                "evaluation_start_date": str(
                    membership[str(evaluation_year)]["start_date"]
                ),
                "evaluation_end_date": str(
                    membership[str(evaluation_year)]["end_date"]
                ),
                "evaluation_candidate_row_count_before_target_completeness": int(
                    membership[str(evaluation_year)]["common_support_row_count"]
                ),
            }
        )
    return folds


def _support_manifest(
    *,
    input_fingerprint: str,
    membership: Mapping[str, Any],
    common_support_hash: str,
    semantics: Mapping[str, Any],
    study_id: str = STUDY_ID,
    source_study_id: str = SOURCE_STUDY_ID,
    pool_name: str = "quality_liquidity_complete_pit",
    path_key: str = "support_path",
    hash_key: str = "support_sha256",
    row_count_key: str = "common_support_row_count",
    symbol_count_key: str = "common_support_symbol_count",
) -> dict[str, Any]:
    return {
        "schema": "seq100_quality_liquidity_common_support/v1",
        "status": "completed",
        "study_id": study_id,
        "source_study_id": source_study_id,
        "pool_name": pool_name,
        "input_fingerprint": input_fingerprint,
        "data_history_start": DATA_HISTORY_START,
        "research_start_date": RESEARCH_START,
        "end_date": END_DATE,
        "burn_in_years": list(BURN_IN_YEARS),
        "burn_in_eligible_for_training": False,
        "years": list(RESEARCH_YEARS),
        "common_support_hash": common_support_hash,
        **dict(semantics),
        "partitions": [
            {
                "year": year,
                "path": str(Path(membership[str(year)][path_key]).resolve()),
                "sha256": str(membership[str(year)][hash_key]),
                "row_count": int(membership[str(year)][row_count_key]),
                "symbol_count": int(membership[str(year)][symbol_count_key]),
                "start_date": str(membership[str(year)]["start_date"]),
                "end_date": str(membership[str(year)]["end_date"]),
            }
            for year in RESEARCH_YEARS
        ],
    }


def _compact_feature_partitions(blocks: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(year): {
            family: {
                "path": str(Path(blocks[str(year)][family]["path"]).resolve()),
                "sha256": str(blocks[str(year)][family]["sha256"]),
                "support_sha256": str(blocks[str(year)][family]["support_sha256"]),
                "row_count": int(blocks[str(year)][family]["row_count"]),
                "feature_count": len(blocks[str(year)][family]["feature_columns"]),
            }
            for family in FEATURE_FAMILIES
        }
        for year in RESEARCH_YEARS
    }


def _final_manifest(
    *,
    config: Mapping[str, Any],
    input_fingerprint: str,
    support_manifest_path: Path,
    support_manifest: Mapping[str, Any],
    blocks: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    atlas_manifest_path: Path,
    study_id: str = STUDY_ID,
    source_study_id: str = SOURCE_STUDY_ID,
    daily_support_manifest_path: Path | None = None,
    daily_support_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    atlas = _read_json(atlas_manifest_path)
    return {
        "schema": "seq100_quality_liquidity_research_scope/v1",
        "status": "completed",
        "study_id": study_id,
        "source_study_id": source_study_id,
        "input_fingerprint": input_fingerprint,
        "data_history": {
            "start_date": DATA_HISTORY_START,
            "burn_in_years": list(BURN_IN_YEARS),
            "burn_in_available_for_features": True,
            "burn_in_eligible_for_training": False,
            "burn_in_eligible_for_evaluation": False,
            "burn_in_eligible_for_atlas_statistics": False,
            "burn_in_eligible_for_labels": False,
        },
        "research_period": {
            "start_date": RESEARCH_START,
            "end_date": END_DATE,
            "years": list(RESEARCH_YEARS),
            "future_oos_prediction_years": list(OOS_YEARS),
        },
        "common_support": {
            "manifest_path": str(support_manifest_path.resolve()),
            "manifest_sha256": base._sha256(support_manifest_path),
            "row_count": int(support_manifest["row_count"]),
            "common_support_hash": str(support_manifest["common_support_hash"]),
        },
        "pools": {
            "quality_liquidity_complete_pit": {
                "manifest_path": str(support_manifest_path.resolve()),
                "manifest_sha256": base._sha256(support_manifest_path),
                "row_count": int(support_manifest["row_count"]),
                "support_hash": str(support_manifest["common_support_hash"]),
                "requires_complete_5m": True,
            },
            **(
                {
                    "quality_liquidity_pit": {
                        "manifest_path": str(daily_support_manifest_path.resolve()),
                        "manifest_sha256": base._sha256(daily_support_manifest_path),
                        "row_count": int(daily_support_manifest["row_count"]),
                        "support_hash": str(
                            daily_support_manifest["common_support_hash"]
                        ),
                        "requires_complete_5m": False,
                    }
                }
                if daily_support_manifest_path is not None
                and daily_support_manifest is not None
                else {}
            ),
        },
        "feature_partitions": _compact_feature_partitions(blocks),
        "feature_count": int(atlas["total_continuous_feature_count"]),
        "rolling_oos_folds": [dict(fold) for fold in folds],
        "atlas": {
            "manifest_path": str(atlas_manifest_path.resolve()),
            "manifest_sha256": base._sha256(atlas_manifest_path),
        },
        "source_coverage_semantics": {
            "missing_report_forecast_is_not_zero_reports": True,
            "eastmoney_metadata_does_not_replace_tushare_forecasts": True,
        },
        "training_performed": False,
        "feature_set_selected": False,
        "forbidden_2026_rows": int(support_manifest["forbidden_2026_rows"]),
        "config": dict(config),
    }


def _reusable_atlas_state(
    previous_state: Mapping[str, Any],
    *,
    input_fingerprint: str,
    study_id: str = STUDY_ID,
    source_study_id: str = SOURCE_STUDY_ID,
) -> dict[str, Any]:
    if (
        previous_state.get("study_id") != study_id
        or previous_state.get("source_study_id") != source_study_id
        or previous_state.get("input_fingerprint") != input_fingerprint
    ):
        return {}
    atlas = dict(previous_state.get("atlas", {}) or {})
    return atlas if atlas.get("status") == "completed" else {}


def prepare(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    source_output_root: Path = SOURCE_OUTPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config = _load_config(study_path)
    study_id = str(config["study_id"])
    source_config = dict(config.get("source_data_prep", {}) or {})
    source_study_id = str(source_config["study_id"])
    source_state = _source_state(
        source_output_root,
        expected_study_id=source_study_id,
    )
    membership, blocks = _selected_records(source_state)
    _validate_partition_files(membership, blocks)
    input_fingerprint = _partition_input_fingerprint(
        config=config,
        source_state=source_state,
        membership=membership,
        blocks=blocks,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    previous_state_path = output_root / "state.json"
    previous_state = (
        _read_json(previous_state_path) if previous_state_path.is_file() else {}
    )
    support_manifest_path = output_root / "common_support" / "manifest.json"
    existing_support = (
        _read_json(support_manifest_path) if support_manifest_path.is_file() else {}
    )
    if (
        existing_support.get("status") == "completed"
        and existing_support.get("input_fingerprint") == input_fingerprint
        and tuple(existing_support.get("years", ())) == RESEARCH_YEARS
    ):
        support_manifest = existing_support
    else:
        semantics = _support_semantics(output_root=output_root, membership=membership)
        common_support_hash = base._common_support_hash(
            membership, years=RESEARCH_YEARS
        )
        support_manifest = _support_manifest(
            input_fingerprint=input_fingerprint,
            membership=membership,
            common_support_hash=common_support_hash,
            semantics=semantics,
            study_id=study_id,
            source_study_id=source_study_id,
        )
        base._write_json(support_manifest_path, support_manifest)
    daily_support_manifest_path: Path | None = None
    daily_support_manifest: dict[str, Any] | None = None
    if all(
        membership[str(year)].get("quality_support_path") for year in RESEARCH_YEARS
    ):
        daily_support_manifest_path = (
            output_root / "quality_liquidity_pit" / "manifest.json"
        )
        existing_daily = (
            _read_json(daily_support_manifest_path)
            if daily_support_manifest_path.is_file()
            else {}
        )
        if (
            existing_daily.get("status") == "completed"
            and existing_daily.get("input_fingerprint") == input_fingerprint
            and tuple(existing_daily.get("years", ())) == RESEARCH_YEARS
        ):
            daily_support_manifest = existing_daily
        else:
            daily_semantics = _support_semantics(
                output_root=output_root,
                membership=membership,
                path_key="quality_support_path",
            )
            daily_support_hash = base._quality_support_hash(
                membership,
                years=RESEARCH_YEARS,
            )
            daily_support_manifest = _support_manifest(
                input_fingerprint=input_fingerprint,
                membership=membership,
                common_support_hash=daily_support_hash,
                semantics=daily_semantics,
                study_id=study_id,
                source_study_id=source_study_id,
                pool_name="quality_liquidity_pit",
                path_key="quality_support_path",
                hash_key="quality_support_sha256",
                row_count_key="quality_support_row_count",
                symbol_count_key="quality_liquidity_symbol_count",
            )
            base._write_json(
                daily_support_manifest_path,
                daily_support_manifest,
            )
    folds = _rolling_oos_folds(membership)
    state = {
        "study_id": study_id,
        "status": "scope_prepared",
        "source_study_id": source_study_id,
        "source_state_path": str((source_output_root / "state.json").resolve()),
        "source_state_sha256": base._sha256(source_output_root / "state.json"),
        "input_fingerprint": input_fingerprint,
        "data_history_start": DATA_HISTORY_START,
        "research_start_date": RESEARCH_START,
        "end_date": END_DATE,
        "burn_in_years": list(BURN_IN_YEARS),
        "research_years": list(RESEARCH_YEARS),
        "future_oos_prediction_years": list(OOS_YEARS),
        "membership_years": membership,
        "feature_blocks": blocks,
        "calendar_inputs": dict(source_state.get("calendar_inputs", {}) or {}),
        "qdp_dataset_ids": dict(source_state.get("qdp_dataset_ids", {}) or {}),
        "common_support_manifest_path": str(support_manifest_path.resolve()),
        "common_support_manifest_sha256": base._sha256(support_manifest_path),
        "common_support_row_count": int(support_manifest["row_count"]),
        "common_support_hash": str(support_manifest["common_support_hash"]),
        "quality_liquidity_pit": (
            {
                "manifest_path": str(daily_support_manifest_path.resolve()),
                "manifest_sha256": base._sha256(daily_support_manifest_path),
                "row_count": int(daily_support_manifest["row_count"]),
                "support_hash": str(daily_support_manifest["common_support_hash"]),
            }
            if daily_support_manifest_path is not None
            and daily_support_manifest is not None
            else {}
        ),
        "rolling_oos_folds": folds,
        "training_performed": False,
        "feature_set_selected": False,
        "config": config,
    }
    reusable_atlas = _reusable_atlas_state(
        previous_state,
        input_fingerprint=input_fingerprint,
        study_id=study_id,
        source_study_id=source_study_id,
    )
    if reusable_atlas:
        state["atlas"] = reusable_atlas
    base._write_state(output_root, state)
    report_coverage_value = str(
        dict(
            dict(source_state.get("config", {}) or {}).get("source_artifacts", {}) or {}
        ).get(
            "report_annual_statistics_path",
            base.DEFAULT_REPORT_COVERAGE_PATH,
        )
    )
    report_coverage_path = Path(report_coverage_value)
    if not report_coverage_path.is_absolute():
        report_coverage_path = WORKSPACE_ROOT / report_coverage_path
    base.prepare_atlas(
        output_root=output_root,
        state=state,
        years=RESEARCH_YEARS,
        study_id=study_id,
        start_date=RESEARCH_START,
        end_date=END_DATE,
        future_oos_prediction_years=OOS_YEARS,
        report_coverage_path=report_coverage_path,
    )
    atlas_manifest_path = output_root / "atlas" / "manifest.json"
    final_manifest = _final_manifest(
        config=config,
        input_fingerprint=input_fingerprint,
        support_manifest_path=support_manifest_path,
        support_manifest=support_manifest,
        blocks=blocks,
        folds=folds,
        atlas_manifest_path=atlas_manifest_path,
        study_id=study_id,
        source_study_id=source_study_id,
        daily_support_manifest_path=daily_support_manifest_path,
        daily_support_manifest=daily_support_manifest,
    )
    manifest_path = output_root / "manifest.json"
    base._write_json(manifest_path, final_manifest)
    state["status"] = "completed"
    state["manifest_path"] = str(manifest_path.resolve())
    state["manifest_sha256"] = base._sha256(manifest_path)
    state["training_performed"] = False
    state["feature_set_selected"] = False
    base._write_state(output_root, state)
    return status(output_root=output_root)


def status(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    path = output_root / "state.json"
    state = _read_json(path) if path.is_file() else {}
    return {
        "study_id": state.get("study_id", STUDY_ID),
        "status": state.get("status", "pending"),
        "data_history_start": state.get("data_history_start", DATA_HISTORY_START),
        "research_start_date": state.get("research_start_date", RESEARCH_START),
        "end_date": state.get("end_date", END_DATE),
        "burn_in_years": state.get("burn_in_years", list(BURN_IN_YEARS)),
        "research_year_count": len(state.get("research_years", [])),
        "common_support_row_count": int(state.get("common_support_row_count", 0)),
        "common_support_hash": state.get("common_support_hash", ""),
        "atlas_status": dict(state.get("atlas", {}) or {}).get("status", "pending"),
        "training_performed": state.get("training_performed", False),
    }


def evaluate(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    state = _read_json(output_root / "state.json")
    manifest = _read_json(output_root / "manifest.json")
    support_manifest = _read_json(output_root / "common_support" / "manifest.json")
    atlas_manifest = _read_json(output_root / "atlas" / "manifest.json")
    membership = dict(state.get("membership_years", {}) or {})
    blocks = dict(state.get("feature_blocks", {}) or {})
    _validate_partition_files(membership, blocks)
    semantics = _support_semantics(output_root=output_root, membership=membership)
    folds = _rolling_oos_folds(membership)
    config = dict(state.get("config", {}) or {})
    source_config = dict(config.get("source_data_prep", {}) or {})
    expected_source_study_id = str(source_config.get("study_id", SOURCE_STUDY_ID))
    expected_row_count = int(
        source_config.get(
            "expected_complete_support_rows",
            EXPECTED_COMMON_SUPPORT_ROW_COUNT,
        )
    )
    expected_feature_count = int(
        source_config.get("expected_existing_feature_count", EXPECTED_FEATURE_COUNT)
    )
    expected_training_rows = {
        int(year): int(count)
        for year, count in dict(
            config.get("expected_rolling_training_rows", EXPECTED_TRAINING_ROWS) or {}
        ).items()
    }
    daily_expected = source_config.get("expected_daily_support_rows")
    daily_manifest_record = dict(
        dict(manifest.get("pools", {}) or {}).get("quality_liquidity_pit", {}) or {}
    )
    daily_semantics: dict[str, Any] = {}
    if daily_manifest_record:
        daily_semantics = _support_semantics(
            output_root=output_root,
            membership=membership,
            path_key="quality_support_path",
        )
    checks = {
        "completed": state.get("status") == "completed",
        "source_prep_preserved": state.get("source_study_id")
        == expected_source_study_id,
        "burn_in_not_eligible": state.get("burn_in_years") == list(BURN_IN_YEARS)
        and 2010 not in state.get("research_years", []),
        "research_years_exact": tuple(state.get("research_years", ()))
        == RESEARCH_YEARS,
        "full_2011_start": semantics["min_date"] == "2011-01-04",
        "ends_in_2025": semantics["max_date"] == END_DATE,
        "row_count_matches": semantics["row_count"]
        == int(state["common_support_row_count"])
        == int(support_manifest["row_count"]),
        "expected_row_count": semantics["row_count"] == expected_row_count,
        "candidate_ids_unique": semantics["row_count"]
        == semantics["unique_candidate_count"],
        "no_pre_scope_rows": semantics["pre_scope_rows"] == 0,
        "forbidden_2026_rows": semantics["forbidden_2026_rows"] == 0,
        "year_dates_aligned": semantics["year_date_mismatch_rows"] == 0,
        "support_hash_matches": state.get("common_support_hash")
        == support_manifest.get("common_support_hash")
        == manifest.get("common_support", {}).get("common_support_hash"),
        "atlas_scope_exact": tuple(atlas_manifest.get("years", ())) == RESEARCH_YEARS
        and atlas_manifest.get("start_date") == RESEARCH_START
        and atlas_manifest.get("end_date") == END_DATE,
        "atlas_support_matches": atlas_manifest.get("common_support_hash")
        == state.get("common_support_hash"),
        "rolling_oos_folds_exact": state.get("rolling_oos_folds") == folds
        and manifest.get("rolling_oos_folds") == folds,
        "rolling_oos_training_rows_exact": {
            int(fold["evaluation_year"]): int(
                fold["training_candidate_row_count_before_target_purge"]
            )
            for fold in folds
        }
        == expected_training_rows,
        "feature_count_exact": int(atlas_manifest["total_continuous_feature_count"])
        == expected_feature_count,
        "dual_pool_contract": (
            daily_expected is None
            or (
                bool(daily_manifest_record)
                and daily_semantics.get("row_count") == int(daily_expected)
                and daily_semantics.get("unique_candidate_count") == int(daily_expected)
                and daily_semantics.get("pre_scope_rows") == 0
                and daily_semantics.get("forbidden_2026_rows") == 0
            )
        ),
        "training_not_performed": state.get("training_performed") is False
        and manifest.get("training_performed") is False
        and atlas_manifest.get("training_performed") is False,
        "feature_set_not_selected": state.get("feature_set_selected") is False
        and manifest.get("feature_set_selected") is False
        and atlas_manifest.get("feature_set_selected") is False,
        "report_source_gaps_explicit": manifest.get(
            "source_coverage_semantics", {}
        ).get("missing_report_forecast_is_not_zero_reports")
        is True,
    }
    if not all(checks.values()):
        raise ResearchScopeError(f"research_scope_evaluation_failed:{checks}")
    return {
        "status": "ok",
        "study_id": state.get("study_id", STUDY_ID),
        "checks": checks,
        "common_support_row_count": semantics["row_count"],
        "common_support_hash": state["common_support_hash"],
        "feature_count": int(atlas_manifest["total_continuous_feature_count"]),
        "quality_liquidity_pit_row_count": int(daily_semantics.get("row_count", 0)),
        "rolling_oos_folds": folds,
        "training_performed": False,
    }


def self_test() -> dict[str, Any]:
    if RESEARCH_YEARS != tuple(range(2011, 2026)):
        raise AssertionError("research year boundary changed")
    if BURN_IN_YEARS != (2010,) or set(BURN_IN_YEARS) & set(RESEARCH_YEARS):
        raise AssertionError("burn-in entered formal research years")
    if OOS_YEARS != (2023, 2024, 2025) or 2026 in RESEARCH_YEARS:
        raise AssertionError("OOS or forbidden year changed")
    return {
        "status": "ok",
        "checks": {
            "data_history_start": DATA_HISTORY_START,
            "research_start_date": RESEARCH_START,
            "burn_in_only": list(BURN_IN_YEARS),
            "burn_in_eligible_for_atlas_statistics": False,
            "burn_in_eligible_for_labels": False,
            "research_year_count": len(RESEARCH_YEARS),
            "future_oos_prediction_years": list(OOS_YEARS),
            "training_performed": False,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seq100-quality-liquidity-research-scope")
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--source-output-root", type=Path, default=SOURCE_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-pending", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.status:
        payload = status(output_root=args.output_root)
    elif args.prepare or args.run_pending:
        payload = prepare(
            study_path=args.study_path,
            source_output_root=args.source_output_root,
            output_root=args.output_root,
        )
    elif args.evaluate:
        payload = evaluate(output_root=args.output_root)
    else:
        payload = self_test()
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
