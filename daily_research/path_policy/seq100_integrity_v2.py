from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from daily_research.path_policy import seq100_2026_fold_comparison as comparison
from daily_research.path_policy import seq100_candidate_execution as candidate_execution
from daily_research.path_policy import seq100_checkpoint_freshness as freshness
from daily_research.path_policy import seq100_development as development
from quant_data_platform.qdp_v2.manifest import qdp_snapshot_sha256


INTEGRITY_SCHEMA = "seq100_integrity_v2"
METRIC_SEMANTIC_VERSION = "seq100_checkpoint_vintage_2026_metrics_v1"
LEGACY_ARCHIVE_NAME = "integrity_v1"
RESULT_OUTPUT_NAMES = (
    "topk_metrics_csv",
    "daily_rank_ic_csv",
    "daily_topk_metrics_csv",
    "topk_candidates_parquet",
)


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(
        (json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def candidate_semantic_sha256(path: Path) -> str:
    frame = pd.read_parquet(path, columns=["trade_date", "symbol"])
    if frame.empty or bool(frame.duplicated(["trade_date", "symbol"]).any()):
        raise ValueError("candidate semantic keys are empty or duplicated")
    ordered = frame.assign(
        trade_date=frame["trade_date"].astype(str),
        symbol=frame["symbol"].astype(str),
    ).sort_values(["trade_date", "symbol"], kind="mergesort")
    digest = hashlib.sha256()
    digest.update(b"seq100_candidate_semantic_v1\n")
    for row in ordered.itertuples(index=False):
        digest.update(f"{row.trade_date}\t{row.symbol}\n".encode("utf-8"))
    return digest.hexdigest()


def model_bundle_sha256(record: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "seq100_model_bundle_v1",
            "checkpoint_content_sha256": str(record["checkpoint_sha256"]),
            "normalization": dict(record["normalization"]),
            "architecture": dict(record["architecture"]),
        }
    )


def result_artifact_sha256(outputs: Mapping[str, str | Path]) -> str:
    files: list[dict[str, Any]] = []
    for name in sorted(RESULT_OUTPUT_NAMES):
        path = Path(str(outputs[name])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "logical_name": name,
                "file_size": int(path.stat().st_size),
                "sha256": file_sha256(path),
            }
        )
    return canonical_sha256(
        {"schema_version": "seq100_result_artifact_bundle_v1", "files": files}
    )


def evaluation_contract_sha256(
    source_view: Mapping[str, Any], evaluation_rules: Mapping[str, Any]
) -> str:
    return canonical_sha256(
        {
            "schema_version": "seq100_evaluation_contract_v2",
            "metric_semantic_version": METRIC_SEMANTIC_VERSION,
            "rules": dict(evaluation_rules),
            "execution_cost_contract": source_view.get("execution_cost_contract", {}),
            "terminal_execution_contract": source_view.get(
                "terminal_execution_contract", {}
            ),
        }
    )


def _data_payloads(source_view: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    label_payload = {
        "label_arrays": source_view.get("label_arrays", {}),
        "label_semantics": source_view.get("label_semantics", {}),
        "mask_views": source_view.get("mask_views", {}),
        "candidate_label_coverage": candidate.get("label_coverage_by_date", []),
    }
    execution_payload = {
        "execution_arrays": source_view.get("execution_arrays", {}),
        "execution_views": source_view.get("execution_views", {}),
        "execution_cost_contract": source_view.get("execution_cost_contract", {}),
        "terminal_execution_contract": source_view.get("terminal_execution_contract", {}),
        "execution_tail_days": source_view.get("execution_tail_days"),
        "exit_sellable_mask": dict(source_view.get("masks", {}) or {}).get(
            "exit_sellable", {}
        ),
        "entry_filled_mask": dict(source_view.get("masks", {}) or {}).get(
            "entry_filled", {}
        ),
    }
    auxiliary_payload = {
        "relative_turnover_supplement": source_view.get(
            "relative_turnover_supplement", {}
        ),
        "input_feature_channels": source_view.get("feature_channels", {}),
        "input_mask_features": dict(source_view.get("data_semantics", {}) or {}).get(
            "model_input_mask_features", []
        ),
    }
    return label_payload, execution_payload, auxiliary_payload


def _shared_integrity(source_view_path: Path) -> dict[str, Any]:
    frozen = comparison._assert_frozen_inputs()
    comparison._validate_overlay_material(comparison.OVERLAY_ROOT / "manifest.json")
    source_view = _read_json(source_view_path.resolve())
    candidate = comparison._candidate_material(source_view)
    candidate_path = Path(str(candidate["path"])).resolve()
    label_payload, execution_payload, auxiliary_payload = _data_payloads(
        source_view, candidate
    )
    data_snapshot = canonical_sha256(
        {
            "schema_version": "seq100_data_snapshot_v2",
            "qdp_snapshot_sha256": qdp_snapshot_sha256(Path(development.QDP_ROOT).resolve()),
            "qdp_full_audit_sha256": str(frozen["qdp_full_audit_sha256"]),
            "base_pack_manifest_sha256": file_sha256(comparison.BASE_PACK_MANIFEST),
            "overlay_manifest_sha256": file_sha256(
                comparison.OVERLAY_ROOT / "manifest.json"
            ),
            "date_values_sha256": canonical_sha256(source_view.get("date_values", [])),
            "symbol_values_sha256": canonical_sha256(
                source_view.get("symbol_values", [])
            ),
            "label_material_sha256": canonical_sha256(label_payload),
            "execution_material_sha256": canonical_sha256(execution_payload),
            "auxiliary_material_sha256": canonical_sha256(auxiliary_payload),
        }
    )
    evaluation_rules = {
        "candidate_membership": "exact_2026_candidate_index_without_label_filtering",
        "ranking": "predicted_price_path_only",
        "legal_exit": dict(comparison.LEGAL_EXIT_CONTRACT),
        "top_k": list(comparison.TOP_K_VALUES),
        "costs_and_execution": "exact_2026_view_contract",
        "terminal_recovery": source_view.get("terminal_execution_contract", {}),
        "score_only_tail_included": False,
    }
    evaluation_contract = evaluation_contract_sha256(source_view, evaluation_rules)
    return {
        "source_view_path": str(source_view_path.resolve()),
        "candidate_index_path": str(candidate_path),
        "candidate_count": int(candidate["row_count"]),
        "candidate_date_count": int(candidate["date_count"]),
        "candidate_trade_date_start": str(candidate["trade_date_start"]),
        "candidate_trade_date_end": str(candidate["trade_date_end"]),
        "symbol_count": int(candidate["pack_symbol_count"]),
        "metric_semantic_version": METRIC_SEMANTIC_VERSION,
        "evaluation_rules": evaluation_rules,
        "data_snapshot_sha256": data_snapshot,
        "candidate_semantic_sha256": candidate_semantic_sha256(candidate_path),
        "evaluation_contract_sha256": evaluation_contract,
    }


def _checkpoint_records(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for task_id, task in dict(contract["tasks"]).items():
        record = freshness._checkpoint_metadata(
            Path(str(task["run_dir"])),
            profile=str(task["profile"]),
            vintage=int(task["checkpoint_vintage"]),
        )
        records[str(task_id)] = record
    return records


def _legacy_metrics(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"task_id": str(result["task_id"]), "metrics": dict(result["metrics"])}
        for result in sorted(results, key=lambda row: str(row["task_id"]))
    ]


def _metrics_sha256(results: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(_legacy_metrics(results))


def _validate_legacy(
    root: Path, contract: Mapping[str, Any], *, require_complete: bool
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    declared = str(contract.get("contract_sha256", ""))
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if int(contract.get("schema_version", 0)) != 1 or declared != canonical_sha256(payload):
        raise ValueError("legacy 2026 comparison contract is not self-consistent")
    fairness = dict(contract["fairness_material"])
    source_view_path = Path(str(fairness["source_view_path"])).resolve()
    source_view = _read_json(source_view_path)
    candidate = comparison._candidate_material(source_view)
    observed_fairness = comparison._fairness_material(
        source_view_path=source_view_path,
        source_view=source_view,
        candidate=candidate,
    )
    for field in (
        "source_view_sha256",
        "overlay_manifest_sha256",
        "base_pack_sha256",
        "candidate_index_sha256",
        "candidate_key_sha256",
        "label_coverage_sha256",
        "label_material_sha256",
        "execution_material_sha256",
        "auxiliary_material_sha256",
        "overlay_material_files_sha256",
        "execution_cost_contract_sha256",
        "date_values_sha256",
        "symbol_values_sha256",
    ):
        if str(fairness.get(field, "")) != str(observed_fairness.get(field, "")):
            raise ValueError(f"legacy fairness material drifted: {field}")

    records = _checkpoint_records(contract)
    results: list[dict[str, Any]] = []
    for task_id, task in dict(contract["tasks"]).items():
        record = records[str(task_id)]
        for field, observed in (
            ("checkpoint_sha256", record["checkpoint_sha256"]),
            ("checkpoint_summary_sha256", record["summary_sha256"]),
            ("checkpoint_fold_contract_sha256", record["checkpoint_fold_contract_sha256"]),
            ("normalization_sha256", record["normalization_sha256"]),
            ("architecture_sha256", record["architecture_sha256"]),
        ):
            if str(task.get(field, "")) != str(observed):
                raise ValueError(f"legacy task material drifted: {task_id}/{field}")
        view_path = Path(str(task["view_path"])).resolve()
        if not view_path.is_file() or file_sha256(view_path) != str(task["view_sha256"]):
            raise ValueError(f"legacy evaluation view drifted: {task_id}")
        result_path = Path(str(task["result_path"])).resolve()
        if not result_path.is_file():
            if require_complete:
                raise FileNotFoundError(result_path)
            continue
        result = _read_json(result_path)
        if (
            str(result.get("status", "")) != "completed"
            or str(result.get("suite_contract_sha256", "")) != declared
            or str(result.get("task_id", "")) != str(task_id)
        ):
            raise ValueError(f"legacy result identity drifted: {task_id}")
        for field in (
            "candidate_index_sha256",
            "candidate_key_sha256",
            "label_coverage_sha256",
            "label_material_sha256",
            "execution_material_sha256",
            "auxiliary_material_sha256",
            "execution_cost_contract_sha256",
            "evaluation_contract_sha256",
        ):
            if str(result.get(field, "")) != str(fairness.get(field, "")):
                raise ValueError(f"legacy result fairness drifted: {task_id}/{field}")
        outputs = dict(result.get("outputs", {}) or {})
        for name in RESULT_OUTPUT_NAMES:
            path = Path(str(outputs.get(name, ""))).resolve()
            if not path.is_file() or file_sha256(path) != str(
                outputs.get(f"{name}_sha256", "")
            ):
                raise ValueError(f"legacy result artifact drifted: {task_id}/{name}")
        if any(
            bool(result.get(field, True))
            for field in (
                "qdp_changed",
                "provider_called",
                "base_pack_changed",
                "live_execution_changed",
            )
        ):
            raise ValueError(f"legacy result crossed a protected boundary: {task_id}")
        results.append(result)
    return results, records


def _archive_paths(root: Path, contract: Mapping[str, Any]) -> list[Path]:
    paths = [
        root / "contract.json",
        root / "state.json",
        root / "comparison_summary.json",
        root / "comparison_summary.md",
        root / "artifact_append_audit.json",
    ]
    for task in dict(contract["tasks"]).values():
        paths.append(Path(str(task["result_path"])).resolve())
    return [path for path in paths if path.is_file()]


def _verify_archive(archive_root: Path) -> dict[str, Any]:
    manifest_path = archive_root / "archive_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read_json(manifest_path)
    for item in list(manifest.get("files", []) or []):
        path = archive_root / str(item["archive_path"])
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(item["file_size"])
            or file_sha256(path) != str(item["sha256"])
        ):
            raise ValueError(f"legacy integrity archive drifted: {path}")
    return manifest


def _archive_legacy(
    root: Path, contract: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> Path:
    archive_root = root / "archive" / LEGACY_ARCHIVE_NAME
    if archive_root.is_dir():
        _verify_archive(archive_root)
        return archive_root
    staging = archive_root.with_name(f".{archive_root.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    for source in _archive_paths(root, contract):
        relative = source.relative_to(root)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files.append(
            {
                "source_path": relative.as_posix(),
                "archive_path": relative.as_posix(),
                "file_size": int(target.stat().st_size),
                "sha256": file_sha256(target),
            }
        )
    metrics_snapshot = {
        "schema_version": 1,
        "task_count": len(results),
        "metrics_sha256": _metrics_sha256(results),
        "tasks": _legacy_metrics(results),
    }
    _atomic_write_json(staging / "metrics_snapshot.json", metrics_snapshot)
    manifest = {
        "schema_version": 1,
        "artifact_type": "seq100_integrity_v1_archive",
        "created_at": _now(),
        "source_root": str(root.resolve()),
        "files": sorted(files, key=lambda item: str(item["archive_path"])),
        "metrics_sha256": metrics_snapshot["metrics_sha256"],
        "task_count": len(results),
    }
    _atomic_write_json(staging / "archive_manifest.json", manifest)
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, archive_root)
    _verify_archive(archive_root)
    return archive_root


def _v2_contract(
    legacy: Mapping[str, Any],
    shared: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task_id, raw in dict(legacy["tasks"]).items():
        task = dict(raw)
        record = records[str(task_id)]
        tasks[str(task_id)] = {
            "task_id": str(task_id),
            "profile": str(task["profile"]),
            "checkpoint_vintage": int(task["checkpoint_vintage"]),
            "target_year": int(task["target_year"]),
            "run_dir": str(Path(str(task["run_dir"])).resolve()),
            "checkpoint_path": str(Path(str(record["checkpoint_path"])).resolve()),
            "normalization_fit_date_end_exclusive": str(
                record["normalization_fit_date_end_exclusive"]
            ),
            "view_path": str(Path(str(task["view_path"])).resolve()),
            "output_dir": str(Path(str(task["output_dir"])).resolve()),
            "result_path": str(Path(str(task["result_path"])).resolve()),
            "inference_required": bool(task["inference_required"]),
            "result_source": str(task["result_source"]),
            "model_bundle_sha256": model_bundle_sha256(record),
        }
    return {
        "schema_version": 2,
        "integrity_schema": INTEGRITY_SCHEMA,
        "artifact_type": str(legacy["artifact_type"]),
        "analysis_id": str(legacy["analysis_id"]),
        "study_root": str(Path(str(legacy["study_root"])).resolve()),
        "target_year": int(legacy["target_year"]),
        "checkpoint_vintages": [int(value) for value in legacy["checkpoint_vintages"]],
        "profiles": [str(value) for value in legacy["profiles"]],
        "seed": int(legacy["seed"]),
        "top_k": [int(value) for value in legacy["top_k"]],
        "shared_material": dict(shared),
        "tasks": tasks,
        "gate_contract": dict(legacy["gate_contract"]),
        "uncertainty_contract": dict(legacy["uncertainty_contract"]),
        "protected_boundaries": dict(legacy["protected_boundaries"]),
    }


def _v2_result(
    legacy: Mapping[str, Any],
    *,
    shared: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    legacy_outputs = dict(legacy["outputs"])
    outputs = {name: str(Path(str(legacy_outputs[name])).resolve()) for name in RESULT_OUTPUT_NAMES}
    return {
        "schema_version": 2,
        "integrity_schema": INTEGRITY_SCHEMA,
        "artifact_type": str(legacy["artifact_type"]),
        "status": "completed",
        "completed_at": str(legacy["completed_at"]),
        "analysis_id": str(legacy["analysis_id"]),
        "task_id": str(task["task_id"]),
        "profile": str(task["profile"]),
        "checkpoint_vintage": int(task["checkpoint_vintage"]),
        "target_year": int(task["target_year"]),
        "result_source": str(legacy["result_source"]),
        "inference_performed": bool(legacy["inference_performed"]),
        "checkpoint_path": str(task["checkpoint_path"]),
        "normalization_fit_date_end_exclusive": str(
            task["normalization_fit_date_end_exclusive"]
        ),
        "data_snapshot_sha256": str(shared["data_snapshot_sha256"]),
        "candidate_semantic_sha256": str(shared["candidate_semantic_sha256"]),
        "model_bundle_sha256": str(task["model_bundle_sha256"]),
        "evaluation_contract_sha256": str(shared["evaluation_contract_sha256"]),
        "result_artifact_sha256": result_artifact_sha256(outputs),
        "metrics": dict(legacy["metrics"]),
        "outputs": outputs,
        "full_prediction_written": bool(legacy.get("full_prediction_written", False)),
        "qdp_changed": False,
        "provider_called": False,
        "base_pack_changed": False,
        "live_execution_changed": False,
    }


def _expected_output_paths(task: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Path]:
    expected = comparison._task_artifact_paths(task)
    if str(result.get("result_source", "")) == "reused_2026_training_evaluation":
        run_dir = Path(str(task["run_dir"])).resolve()
        expected = {name: run_dir / path.name for name, path in expected.items()}
    return expected


def _verify_v2_result(
    result_path: Path,
    *,
    contract: Mapping[str, Any],
    task: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = _read_json(result_path)
    shared = dict(contract["shared_material"])
    if (
        int(result.get("schema_version", 0)) != 2
        or str(result.get("integrity_schema", "")) != INTEGRITY_SCHEMA
        or str(result.get("status", "")) != "completed"
        or str(result.get("task_id", "")) != str(task["task_id"])
    ):
        raise ValueError(f"invalid v2 evaluation identity: {result_path}")
    expected_identity = {
        "profile": str(task["profile"]),
        "checkpoint_vintage": int(task["checkpoint_vintage"]),
        "target_year": int(task["target_year"]),
        "data_snapshot_sha256": str(shared["data_snapshot_sha256"]),
        "candidate_semantic_sha256": str(shared["candidate_semantic_sha256"]),
        "model_bundle_sha256": str(task["model_bundle_sha256"]),
        "evaluation_contract_sha256": str(shared["evaluation_contract_sha256"]),
    }
    for field, expected in expected_identity.items():
        if result.get(field) != expected:
            raise ValueError(f"v2 evaluation identity drift: {task['task_id']}/{field}")
    metrics = dict(result.get("metrics", {}) or {})
    if (
        int(metrics.get("candidate_count", -1)) != int(shared["candidate_count"])
        or int(metrics.get("date_count", -1)) != int(shared["candidate_date_count"])
        or float(metrics.get("candidate_score_coverage", -1.0)) != 1.0
        or float(metrics.get("execution_return_coverage", -1.0)) != 1.0
    ):
        raise ValueError(f"v2 evaluation coverage drift: {task['task_id']}")
    outputs = dict(result.get("outputs", {}) or {})
    expected_paths = _expected_output_paths(task, result)
    for name, expected_path in expected_paths.items():
        if Path(str(outputs.get(name, ""))).resolve() != expected_path.resolve():
            raise ValueError(f"v2 evaluation output path drift: {task['task_id']}/{name}")
    if result_artifact_sha256(outputs) != str(result["result_artifact_sha256"]):
        raise ValueError(f"v2 result bundle drift: {task['task_id']}")
    topk = pd.read_csv(expected_paths["topk_metrics_csv"])
    expected_cost = candidate_execution.execution_cost_contract_sha256(
        _read_json(Path(str(shared["source_view_path"])))
    )
    observed_costs = {
        str(value)
        for value in topk["execution_cost_contract_sha256"].dropna().tolist()
    }
    if observed_costs != {expected_cost}:
        raise ValueError(f"v2 evaluation cost contract drift: {task['task_id']}")
    daily = pd.read_csv(expected_paths["daily_topk_metrics_csv"])
    coverage = freshness._common_daily_coverage(daily).to_dict("records")
    if any(
        bool(result.get(field, True))
        for field in (
            "qdp_changed",
            "provider_called",
            "base_pack_changed",
            "live_execution_changed",
        )
    ):
        raise ValueError(f"v2 evaluation crossed a protected boundary: {task['task_id']}")
    return result, coverage


def _verify_v2_contract(
    contract_path: Path,
    *,
    allow_incomplete: bool = False,
    invalid_tasks: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = _read_json(contract_path)
    if (
        int(contract.get("schema_version", 0)) != 2
        or str(contract.get("integrity_schema", "")) != INTEGRITY_SCHEMA
        or str(contract.get("analysis_id", "")) != comparison.ANALYSIS_ID
    ):
        raise ValueError("wrong Seq100 integrity v2 contract")
    shared = dict(contract.get("shared_material", {}) or {})
    observed_shared = _shared_integrity(Path(str(shared["source_view_path"])))
    if observed_shared != shared:
        raise ValueError("Seq100 shared data, candidate, or evaluation identity drifted")
    records = _checkpoint_records(contract)
    for task_id, task in dict(contract["tasks"]).items():
        if model_bundle_sha256(records[str(task_id)]) != str(task["model_bundle_sha256"]):
            raise ValueError(f"Seq100 model bundle drifted: {task_id}")
    results: list[dict[str, Any]] = []
    coverages: list[list[dict[str, Any]]] = []
    for task_id, task in dict(contract["tasks"]).items():
        result_path = Path(str(task["result_path"])).resolve()
        if not result_path.is_file():
            continue
        try:
            result, coverage = _verify_v2_result(
                result_path, contract=contract, task=task
            )
        except (FileNotFoundError, ValueError):
            if not allow_incomplete:
                raise
            if invalid_tasks is not None:
                invalid_tasks.add(str(task_id))
            comparison._reset_task_output(task=task, suite_root=contract_path.parent)
            continue
        results.append(result)
        coverages.append(coverage)
    if coverages and any(coverage != coverages[0] for coverage in coverages[1:]):
        raise ValueError("daily execution coverage differs across v2 tasks")
    return contract, results


def _summary_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "profile": str(result["profile"]),
            "checkpoint_vintage": int(result["checkpoint_vintage"]),
            "target_year": int(result["target_year"]),
            "normalization_fit_date_end_exclusive": str(
                result["normalization_fit_date_end_exclusive"]
            ),
            "result_source": str(result["result_source"]),
            **dict(result["metrics"]),
        }
        for result in sorted(
            results,
            key=lambda value: (str(value["profile"]), int(value["checkpoint_vintage"])),
        )
    ]


def _rebuild_summary(
    root: Path,
    contract: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    *,
    rewrite_derived_csv: bool,
) -> dict[str, Any]:
    rows = _summary_rows(results)
    if rewrite_derived_csv:
        comparison._write_csv(root / "vintage_metrics.csv", pd.DataFrame(rows))
        daily = comparison._daily_pairwise_deltas(contract=contract, results=results)
        comparison._write_csv(root / "daily_pairwise_deltas.csv", daily)
        pairwise = (
            daily.groupby(["profile", "older_vintage", "metric"], sort=True)
            .agg(
                date_count=("trade_date", "size"),
                mean_improvement_delta=("improvement_delta", "mean"),
                median_improvement_delta=("improvement_delta", "median"),
                checkpoint_2026_win_rate=("checkpoint_2026_won", "mean"),
            )
            .reset_index()
        )
        comparison._write_csv(root / "pairwise_summary.csv", pairwise)
    pairwise = pd.read_csv(root / "pairwise_summary.csv")
    gate = comparison._expanded_freshness_gate(results)
    reuse_checks: dict[str, Any] = {}
    for profile in comparison.PROFILE_ORDER:
        result = next(
            row
            for row in results
            if str(row["profile"]) == profile
            and int(row["checkpoint_vintage"]) == comparison.TARGET_YEAR
        )
        reuse_checks[profile] = {
            "result_source": str(result["result_source"]),
            "inference_performed": bool(result["inference_performed"]),
            "pass": str(result["result_source"])
            in {"reused_2026_training_evaluation", "fresh_inference"},
        }
    coverage_checks = {
        "data_snapshot_identical": True,
        "candidate_keys_identical": True,
        "evaluation_contract_identical": True,
        "daily_execution_coverage_identical": True,
        "score_coverage_complete": all(
            float(dict(result["metrics"])["candidate_score_coverage"]) == 1.0
            for result in results
        ),
        "execution_coverage_complete": all(
            float(dict(result["metrics"])["execution_return_coverage"]) == 1.0
            for result in results
        ),
    }
    fairness_pass = all(coverage_checks.values())
    overall = bool(
        fairness_pass
        and all(value["pass"] for value in reuse_checks.values())
        and gate["metric_gate_pass"]
    )
    summary = {
        "schema_version": 2,
        "integrity_schema": INTEGRITY_SCHEMA,
        "artifact_type": "seq100_checkpoint_vintage_2026_summary",
        "status": "completed",
        "completed_at": _now(),
        "analysis_id": comparison.ANALYSIS_ID,
        "target_year": comparison.TARGET_YEAR,
        "signal_window": [comparison.SIGNAL_START, comparison.SIGNAL_END],
        "signal_date_count": comparison.EXPECTED_SIGNAL_DATE_COUNT,
        "candidate_count": comparison.EXPECTED_CANDIDATE_COUNT,
        "symbol_count": comparison.EXPECTED_SYMBOL_COUNT,
        "profiles": list(comparison.PROFILE_ORDER),
        "checkpoint_vintages": list(comparison.CHECKPOINT_VINTAGES),
        "fairness": {"checks": coverage_checks, "pass": fairness_pass},
        "existing_2026_consistency": {
            "profiles": reuse_checks,
            "pass": all(value["pass"] for value in reuse_checks.values()),
        },
        "gate": gate,
        "overall_2026_freshness_gate_pass": overall,
        "uncertainty": dict(contract["uncertainty_contract"]),
        "metrics": rows,
        "pairwise_summary": _json_records(pairwise),
        "outputs": {
            "vintage_metrics_csv": str((root / "vintage_metrics.csv").resolve()),
            "daily_pairwise_deltas_csv": str(
                (root / "daily_pairwise_deltas.csv").resolve()
            ),
            "pairwise_summary_csv": str((root / "pairwise_summary.csv").resolve()),
        },
        "qdp_changed": False,
        "provider_called": False,
        "base_pack_changed": False,
        "live_execution_changed": False,
        "score_only_tail_included": False,
        "finite_capital_cagr_reported": False,
    }
    _atomic_write_json(root / "comparison_summary.json", summary)
    lines = [
        "# Seq100 checkpoint vintages on the complete 2026 realized window",
        "",
        f"Evaluation window: {comparison.SIGNAL_START} through {comparison.SIGNAL_END}; "
        f"{comparison.EXPECTED_SIGNAL_DATE_COUNT} signal dates and "
        f"{comparison.EXPECTED_CANDIDATE_COUNT:,} candidates.",
        "",
        "| profile | checkpoint | Top1 alpha | Top3 alpha | Top5 alpha | Top10 alpha | Rank IC | exit regret | path MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['checkpoint_vintage']} | "
            f"{row['top1_base_alpha']:.4%} | {row['top3_base_alpha']:.4%} | "
            f"{row['top5_base_alpha']:.4%} | {row['top10_base_alpha']:.4%} | "
            f"{row['rank_ic']:.6f} | {row['exit_regret']:.6f} | {row['path_mae']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Fairness and coverage checks: `{fairness_pass}`.",
            f"2026 freshness gate: `{overall}`.",
            "No formal moving-block interval or finite-capital CAGR is reported for this 48-date window.",
        ]
    )
    _atomic_write_text(root / "comparison_summary.md", "\n".join(lines) + "\n")
    return summary


def migrate_2026_contract_v2(
    *, output_root: Path = comparison.ANALYSIS_ROOT, require_complete: bool = True
) -> dict[str, Any]:
    root = output_root.resolve()
    contract_path = root / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    current = _read_json(contract_path)
    if int(current.get("schema_version", 0)) == 2:
        return verify_research_integrity(output_root=root, include_report=False)
    results, records = _validate_legacy(root, current, require_complete=require_complete)
    archive_root = _archive_legacy(root, current, results)
    shared = _shared_integrity(
        Path(str(dict(current["fairness_material"])["source_view_path"]))
    )
    contract = _v2_contract(current, shared, records)
    converted: list[dict[str, Any]] = []
    for legacy_result in results:
        task = dict(contract["tasks"])[str(legacy_result["task_id"])]
        result = _v2_result(legacy_result, shared=shared, task=task)
        _atomic_write_json(Path(str(task["result_path"])), result)
        converted.append(result)
    _atomic_write_json(contract_path, contract)
    if require_complete and len(converted) != len(contract["tasks"]):
        raise ValueError("v2 migration did not cover all comparison tasks")
    if len(converted) == len(contract["tasks"]):
        summary = _rebuild_summary(
            root, contract, converted, rewrite_derived_csv=False
        )
        state = {
            "schema_version": 2,
            "integrity_schema": INTEGRITY_SCHEMA,
            "status": "completed",
            "analysis_id": comparison.ANALYSIS_ID,
            "completed_tasks": sorted(dict(contract["tasks"])),
            "reuse_fallback_tasks": [],
            "summary": str((root / "comparison_summary.json").resolve()),
            "updated_at": _now(),
        }
    else:
        summary = {}
        state = {
            "schema_version": 2,
            "integrity_schema": INTEGRITY_SCHEMA,
            "status": "prepared",
            "analysis_id": comparison.ANALYSIS_ID,
            "completed_tasks": sorted(str(result["task_id"]) for result in converted),
            "reuse_fallback_tasks": [],
            "updated_at": _now(),
        }
    _atomic_write_json(root / "state.json", state)
    verified = verify_research_integrity(output_root=root, include_report=False)
    snapshot = _read_json(archive_root / "metrics_snapshot.json")
    if int(snapshot["task_count"]) == len(converted) and str(
        snapshot["metrics_sha256"]
    ) != _metrics_sha256(converted):
        raise ValueError("metrics changed during v1 to v2 migration")
    return {
        "status": "completed" if summary else "prepared",
        "integrity_schema": INTEGRITY_SCHEMA,
        "archive": str(archive_root),
        "migrated_tasks": len(converted),
        "verified": verified,
    }


def _runtime_v1_contract(root: Path, v2: Mapping[str, Any]) -> Path:
    archive_root = root / "archive" / LEGACY_ARCHIVE_NAME
    legacy = _read_json(archive_root / "contract.json")
    fairness = dict(legacy["fairness_material"])
    implementation = dict(fairness["evaluation_implementation"])
    implementation["aggregation_module"] = str(Path(comparison.__file__).resolve())
    implementation["aggregation_module_sha256"] = file_sha256(
        Path(comparison.__file__).resolve()
    )
    fairness["evaluation_implementation"] = implementation
    fairness["evaluation_contract_sha256"] = canonical_sha256(
        {
            "rules": fairness["evaluation_rules"],
            "implementation": implementation,
            "execution_cost_contract_sha256": fairness[
                "execution_cost_contract_sha256"
            ],
            "label_material_sha256": fairness["label_material_sha256"],
            "execution_material_sha256": fairness["execution_material_sha256"],
            "auxiliary_material_sha256": fairness["auxiliary_material_sha256"],
        }
    )
    legacy["fairness_material"] = fairness
    records = _checkpoint_records(v2)
    for task_id, task in dict(legacy["tasks"]).items():
        record = records[str(task_id)]
        task.update(
            {
                "checkpoint_path": str(record["checkpoint_path"]),
                "checkpoint_sha256": str(record["checkpoint_sha256"]),
                "checkpoint_summary_sha256": str(record["summary_sha256"]),
                "checkpoint_fold_contract_sha256": str(
                    record["checkpoint_fold_contract_sha256"]
                ),
                "normalization_sha256": str(record["normalization_sha256"]),
                "architecture_sha256": str(record["architecture_sha256"]),
                "view_sha256": file_sha256(Path(str(task["view_path"]))),
            }
        )
    payload = {key: value for key, value in legacy.items() if key != "contract_sha256"}
    legacy = {**payload, "contract_sha256": canonical_sha256(payload)}
    path = root / f".integrity_v1_runtime_{os.getpid()}.json"
    _atomic_write_json(path, legacy)
    return path


def evaluate_v2_task(
    *, suite_contract_path: Path, profile: str, vintage: int, force_inference: bool = False
) -> dict[str, Any]:
    root = suite_contract_path.resolve().parent
    contract, _ = _verify_v2_contract(suite_contract_path.resolve())
    task_id = comparison._comparison_task_id(profile, vintage)
    task = dict(contract["tasks"])[task_id]
    result_path = Path(str(task["result_path"])).resolve()
    if result_path.is_file():
        result, _coverage = _verify_v2_result(
            result_path, contract=contract, task=task
        )
        return result
    runtime_path = _runtime_v1_contract(root, contract)
    try:
        legacy = comparison._evaluate_2026_vintage_task_v1(
            suite_contract_path=runtime_path,
            profile=profile,
            vintage=vintage,
            force_inference=force_inference,
        )
        result = _v2_result(
            legacy,
            shared=dict(contract["shared_material"]),
            task=task,
        )
        _atomic_write_json(result_path, result)
        _verify_v2_result(result_path, contract=contract, task=task)
        return result
    finally:
        runtime_path.unlink(missing_ok=True)


def evaluate_or_resume_2026_v2(
    *, output_root: Path = comparison.ANALYSIS_ROOT, max_tasks: int = 0
) -> dict[str, Any]:
    if int(max_tasks) < 0:
        raise ValueError("max_tasks must be non-negative")
    comparison._assert_no_other_research_process()
    root = output_root.resolve()
    contract_path = root / "contract.json"
    if not contract_path.is_file():
        comparison.prepare_2026_comparison(output_root=root)
    if int(_read_json(contract_path).get("schema_version", 0)) == 1:
        migrate_2026_contract_v2(output_root=root, require_complete=False)
    invalid_tasks: set[str] = set()
    contract, results = _verify_v2_contract(
        contract_path,
        allow_incomplete=True,
        invalid_tasks=invalid_tasks,
    )
    completed = {str(result["task_id"]) for result in results}
    launched = 0
    changed = False
    for task_id, task in dict(contract["tasks"]).items():
        if task_id in completed:
            continue
        if int(max_tasks) > 0 and launched >= int(max_tasks):
            break
        output_dir = Path(str(task["output_dir"])).resolve()
        if output_dir.exists():
            invalid_tasks.add(str(task_id))
            comparison._reset_task_output(task=task, suite_root=root)
        command = [
            str(comparison.PYTHON),
            "-m",
            "daily_research.path_policy.seq100_development",
            "evaluate-2026-vintage-task",
            "--suite-contract",
            str(contract_path),
            "--profile",
            str(task["profile"]),
            "--vintage",
            str(task["checkpoint_vintage"]),
        ]
        if (
            int(task["checkpoint_vintage"]) == comparison.TARGET_YEAR
            and task_id in invalid_tasks
        ):
            command.append("--force-inference")
        comparison._guarded_command(
            command,
            log_path=root / "logs" / f"memory_guard_{task_id}.json",
        )
        result, _coverage = _verify_v2_result(
            Path(str(task["result_path"])), contract=contract, task=task
        )
        completed.add(str(result["task_id"]))
        launched += 1
        changed = True
    contract, results = _verify_v2_contract(contract_path)
    if len(results) != len(contract["tasks"]):
        state = {
            "schema_version": 2,
            "integrity_schema": INTEGRITY_SCHEMA,
            "status": "running",
            "analysis_id": comparison.ANALYSIS_ID,
            "completed_tasks": sorted(completed),
            "reuse_fallback_tasks": sorted(
                task_id
                for task_id in invalid_tasks
                if int(dict(contract["tasks"])[task_id]["checkpoint_vintage"])
                == comparison.TARGET_YEAR
            ),
            "updated_at": _now(),
        }
        _atomic_write_json(root / "state.json", state)
        return state
    summary = _rebuild_summary(
        root, contract, results, rewrite_derived_csv=changed
    )
    _atomic_write_json(
        root / "state.json",
        {
            "schema_version": 2,
            "integrity_schema": INTEGRITY_SCHEMA,
            "status": "completed",
            "analysis_id": comparison.ANALYSIS_ID,
            "completed_tasks": sorted(completed),
            "reuse_fallback_tasks": sorted(
                task_id
                for task_id in invalid_tasks
                if int(dict(contract["tasks"])[task_id]["checkpoint_vintage"])
                == comparison.TARGET_YEAR
            ),
            "summary": str((root / "comparison_summary.json").resolve()),
            "updated_at": _now(),
        },
    )
    return summary


def verify_research_integrity(
    *,
    output_root: Path = comparison.ANALYSIS_ROOT,
    include_report: bool = True,
) -> dict[str, Any]:
    root = output_root.resolve()
    contract, results = _verify_v2_contract(root / "contract.json")
    archive = _verify_archive(root / "archive" / LEGACY_ARCHIVE_NAME)
    snapshot = _read_json(root / "archive" / LEGACY_ARCHIVE_NAME / "metrics_snapshot.json")
    if len(results) == int(snapshot["task_count"]):
        if _metrics_sha256(results) != str(snapshot["metrics_sha256"]):
            raise ValueError("v2 metrics differ from the archived v1 metrics")
    report_status: dict[str, Any] | None = None
    if include_report:
        from daily_research.path_policy import build_seq100_research_report as report

        report_status = report.verify_compact_report()
    return {
        "status": "ok",
        "integrity_schema": INTEGRITY_SCHEMA,
        "shared_hash_classes": 3,
        "task_count": len(contract["tasks"]),
        "completed_task_count": len(results),
        "score_coverage_complete": all(
            float(dict(result["metrics"])["candidate_score_coverage"]) == 1.0
            for result in results
        ),
        "execution_coverage_complete": all(
            float(dict(result["metrics"])["execution_return_coverage"]) == 1.0
            for result in results
        ),
        "archive_file_count": len(archive["files"]),
        "report": report_status,
    }


def is_v2_contract(path: Path) -> bool:
    return path.is_file() and int(_read_json(path).get("schema_version", 0)) == 2


__all__ = [
    "INTEGRITY_SCHEMA",
    "METRIC_SEMANTIC_VERSION",
    "candidate_semantic_sha256",
    "canonical_sha256",
    "evaluation_contract_sha256",
    "evaluate_or_resume_2026_v2",
    "evaluate_v2_task",
    "file_sha256",
    "is_v2_contract",
    "migrate_2026_contract_v2",
    "model_bundle_sha256",
    "result_artifact_sha256",
    "verify_research_integrity",
]
