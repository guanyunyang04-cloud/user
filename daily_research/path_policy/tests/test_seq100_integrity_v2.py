from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import seq100_integrity_v2 as integrity
from daily_research.path_policy.seq100_development import _parser


def _hash_field_names(value: object) -> list[str]:
    names: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if "sha256" in str(key):
                names.append(str(key))
            names.extend(_hash_field_names(item))
    elif isinstance(value, list):
        for item in value:
            names.extend(_hash_field_names(item))
    return names


def test_integrity_commands_are_registered() -> None:
    assert _parser().parse_args(["migrate-2026-contract-v2"]).command == (
        "migrate-2026-contract-v2"
    )
    assert _parser().parse_args(["build-compact-report"]).command == (
        "build-compact-report"
    )
    assert _parser().parse_args(["verify-research-integrity"]).command == (
        "verify-research-integrity"
    )


def test_candidate_semantic_digest_uses_sorted_date_symbol_keys(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    changed = tmp_path / "changed.parquet"
    rows = pd.DataFrame(
        {
            "trade_date": ["2026-01-06", "2026-01-05", "2026-01-05"],
            "symbol": ["B", "A", "B"],
            "label_valid": [False, True, False],
        }
    )
    rows.to_parquet(first, index=False)
    rows.iloc[[2, 0, 1]].to_parquet(second, index=False)
    rows.assign(symbol=["B", "A", "C"]).to_parquet(changed, index=False)

    assert integrity.candidate_semantic_sha256(first) == (
        integrity.candidate_semantic_sha256(second)
    )
    assert integrity.candidate_semantic_sha256(first) != (
        integrity.candidate_semantic_sha256(changed)
    )


def test_model_evaluation_and_result_bundles_change_only_with_their_semantics(
    tmp_path: Path,
) -> None:
    record = {
        "checkpoint_sha256": "checkpoint-content",
        "normalization": {"mean": [1.0], "fit_date_end_exclusive": "2026-01-05"},
        "architecture": {"name": "gru", "hidden": 512},
    }
    baseline = integrity.model_bundle_sha256(record)
    changed_normalization = copy.deepcopy(record)
    changed_normalization["normalization"]["mean"] = [2.0]
    changed_architecture = copy.deepcopy(record)
    changed_architecture["architecture"]["hidden"] = 256
    assert baseline != integrity.model_bundle_sha256(changed_normalization)
    assert baseline != integrity.model_bundle_sha256(changed_architecture)

    view = {
        "execution_cost_contract": {"commission": 0.0003},
        "terminal_execution_contract": {"recovery": "last_sellable_close"},
    }
    rules = {"legal_exit": {"domain": [2, 60]}, "top_k": [1, 3, 5, 10]}
    evaluation = integrity.evaluation_contract_sha256(view, rules)
    assert evaluation != integrity.evaluation_contract_sha256(
        {**view, "execution_cost_contract": {"commission": 0.0006}}, rules
    )
    assert evaluation != integrity.evaluation_contract_sha256(
        view, {**rules, "legal_exit": {"domain": [3, 60]}}
    )

    outputs: dict[str, Path] = {}
    for index, name in enumerate(integrity.RESULT_OUTPUT_NAMES):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(f"payload-{index}".encode("ascii"))
        outputs[name] = path
    result = integrity.result_artifact_sha256(outputs)
    outputs["daily_rank_ic_csv"].write_bytes(b"changed")
    assert result != integrity.result_artifact_sha256(outputs)


def test_v2_payload_exposes_only_the_five_registered_hash_classes(
    tmp_path: Path,
) -> None:
    task_id = "legal_flat_baseline_checkpoint_2023_on_2026"
    legacy = {
        "artifact_type": "seq100_checkpoint_vintage_2026_contract",
        "analysis_id": "analysis",
        "study_root": str(tmp_path / "study"),
        "target_year": 2026,
        "checkpoint_vintages": [2023],
        "profiles": ["legal_flat_baseline"],
        "seed": 7,
        "top_k": [1, 3, 5, 10],
        "tasks": {
            task_id: {
                "profile": "legal_flat_baseline",
                "checkpoint_vintage": 2023,
                "target_year": 2026,
                "run_dir": str(tmp_path / "run"),
                "view_path": str(tmp_path / "view.json"),
                "output_dir": str(tmp_path / "tasks" / task_id),
                "result_path": str(tmp_path / "tasks" / task_id / "evaluation.json"),
                "inference_required": True,
                "result_source": "fresh_inference",
            }
        },
        "gate_contract": {},
        "uncertainty_contract": {},
        "protected_boundaries": {},
    }
    shared = {
        "source_view_path": str(tmp_path / "source.json"),
        "candidate_index_path": str(tmp_path / "candidate.parquet"),
        "candidate_count": 2,
        "candidate_date_count": 1,
        "candidate_trade_date_start": "2026-01-05",
        "candidate_trade_date_end": "2026-01-05",
        "symbol_count": 2,
        "metric_semantic_version": integrity.METRIC_SEMANTIC_VERSION,
        "evaluation_rules": {},
        "data_snapshot_sha256": "data",
        "candidate_semantic_sha256": "candidate",
        "evaluation_contract_sha256": "evaluation",
    }
    record = {
        "checkpoint_path": str(tmp_path / "model.pt"),
        "checkpoint_sha256": "checkpoint",
        "normalization": {"fit_date_end_exclusive": "2023-01-03"},
        "normalization_fit_date_end_exclusive": "2023-01-03",
        "architecture": {"name": "gru"},
    }
    contract = integrity._v2_contract(legacy, shared, {task_id: record})
    assert set(_hash_field_names(contract)) == {
        "data_snapshot_sha256",
        "candidate_semantic_sha256",
        "evaluation_contract_sha256",
        "model_bundle_sha256",
    }

    outputs: dict[str, Path] = {}
    for name in integrity.RESULT_OUTPUT_NAMES:
        path = tmp_path / f"{name}.data"
        path.write_text(name, encoding="utf-8")
        outputs[name] = path
    legacy_result = {
        "artifact_type": "seq100_checkpoint_vintage_2026_evaluation",
        "completed_at": "2026-07-19T00:00:00+08:00",
        "analysis_id": "analysis",
        "result_source": "fresh_inference",
        "inference_performed": True,
        "metrics": {"candidate_count": 2},
        "outputs": {name: str(path) for name, path in outputs.items()},
        "full_prediction_written": False,
    }
    result = integrity._v2_result(
        legacy_result,
        shared=shared,
        task=contract["tasks"][task_id],
    )
    assert set(_hash_field_names(result)) == {
        "data_snapshot_sha256",
        "candidate_semantic_sha256",
        "model_bundle_sha256",
        "evaluation_contract_sha256",
        "result_artifact_sha256",
    }


def test_legacy_archive_is_atomic_idempotent_and_detects_drift(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks" / "task_a"
    task_dir.mkdir(parents=True)
    result_path = task_dir / "evaluation.json"
    result_path.write_text('{"status":"completed"}\n', encoding="utf-8")
    contract = {"tasks": {"task_a": {"result_path": str(result_path)}}}
    for name, payload in (
        ("contract.json", contract),
        ("state.json", {"status": "completed"}),
        ("comparison_summary.json", {"status": "completed"}),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    results = [{"task_id": "task_a", "metrics": {"alpha": 0.1}}]

    archive = integrity._archive_legacy(tmp_path, contract, results)
    assert integrity._archive_legacy(tmp_path, contract, results) == archive
    manifest = integrity._verify_archive(archive)
    assert manifest["task_count"] == 1

    (archive / "contract.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="archive drifted"):
        integrity._verify_archive(archive)
