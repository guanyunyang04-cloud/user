from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from daily_research.path_policy import seq100_signal_quality as quality


def _flat_path(rows: int = 1, *, close_return: float = 0.0) -> np.ndarray:
    path = np.zeros((rows, 20, 4), dtype=np.float32)
    path[:, :, :] = close_return
    return path


def _descriptor_fixture(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    count = int(values.shape[0])
    zeros = np.zeros(count, dtype=np.float32)
    return {
        "speed5_log_per_day": values[:, 0],
        "speed10_log_per_day": values[:, 1],
        "speed20_log_per_day": values[:, 2],
        "min_speed_5_10_20": np.min(values[:, :3], axis=1),
        "auc20_net_log": values[:, 3],
        "time_above_break_even20": values[:, 4],
        "mdd20": -values[:, 5],
        "post_peak_fade20": -values[:, 6],
        "r20_net": zeros.copy(),
    }


def _model_configs() -> dict[str, dict[str, object]]:
    freeze = quality.load_research_freeze(quality.load_study())["freeze"]
    return {
        str(item["model_id"]): dict(item["config"])
        for item in freeze["model_candidates"]
    }


def _budget_amendment_payload() -> dict[str, object]:
    study = quality.load_study()
    payload: dict[str, object] = {
        "schema": quality.TRAINING_BUDGET_AMENDMENT_VERSION,
        "study_id": quality.STUDY_ID,
        "status": "frozen",
        "base_study_contract_sha256": study["contract_sha256"],
        "model_ids": list(quality.NEURAL_MODEL_IDS),
        "config_changes": {
            "max_epochs": {"from": 3, "to": 10},
            "patience": {"from": 1, "to": 2},
        },
        "reason": "development_loss_budget_censoring",
    }
    payload["amendment_sha256"] = quality._canonical_json_sha256(payload)
    return payload


DESIGN_STATEMENT = (
    "The frozen objective measures a construct that cannot identify the causal "
    "quantity stated in the study question under the available observations."
)
DESIGN_ARGUMENT = (
    "Every admissible model and metric receives the same insufficient label "
    "information, so additional fitting cannot repair the structural mismatch."
)


def _design_invalidation_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    freeze = {
        "target_candidates": [
            {"target_id": target_id} for target_id in quality.TARGET_CANDIDATE_IDS
        ],
        "model_candidates": [
            {"model_id": model_id} for model_id in quality.MODEL_IDS
        ],
        "scientific_firewall": {"forbidden_years": [2026]},
    }
    freeze_payload = {
        "study_id": quality.STUDY_ID,
        "freeze": freeze,
        "freeze_sha256": quality._canonical_json_sha256(freeze),
    }
    freeze_path = workspace / "research_freeze.json"
    freeze_path.write_text(json.dumps(freeze_payload), encoding="utf-8")
    contract = {
        "contract_id": quality.STUDY_ID,
        "research_freeze": {
            "path": str(freeze_path.resolve()),
            "file_sha256": quality._file_sha256(freeze_path),
            "freeze_sha256": freeze_payload["freeze_sha256"],
        },
    }
    study = {
        "study_id": quality.STUDY_ID,
        "contract": contract,
        "contract_sha256": quality._canonical_json_sha256(contract),
    }
    study_path = workspace / "study.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    monkeypatch.setattr(quality, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(
        quality,
        "_verify_protected_bindings",
        lambda _study: {
            "qdp_active_manifest_sha256": "qdp-hash",
            "registered_model_registry_sha256": "registry-hash",
        },
    )
    return study_path, workspace / "output", workspace


def test_contract_research_freeze_and_source_bindings_are_valid() -> None:
    study = quality.load_study()
    assert study["contract_sha256"] == quality._canonical_json_sha256(
        study["contract"]
    )
    freeze = quality.load_research_freeze(study)
    assert freeze["freeze_sha256"] == quality._canonical_json_sha256(
        freeze["freeze"]
    )
    pack_path, manifest = quality._validate_source_bindings(study)
    assert pack_path.exists()
    assert int(manifest["lookback_days"]) == 180
    assert int(manifest["forward_days"]) >= 60
    assert len(study["contract"]["data"]["qdp_datasets"]) == 14


def test_cli_has_only_the_frozen_public_commands() -> None:
    parser = quality.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparsers.choices) == {
        "validate-research",
        "build-targets",
        "freeze-target",
        "build-view",
        "screen-models",
        "train",
        "evaluate",
        "invalidate-design",
        "closeout",
    }
    assert "build-atlas" not in subparsers.choices
    assert "feature-screen" not in subparsers.choices


def test_invalidate_study_design_writes_hashed_terminal_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path, output_root, workspace = _design_invalidation_fixture(
        tmp_path, monkeypatch
    )
    evidence_path = workspace / "evidence" / "design_analysis.txt"
    evidence_path.parent.mkdir()
    evidence_path.write_text("structural evidence", encoding="utf-8")
    first_attempt = (
        output_root
        / "training/model_a/fold_2023/seed_7/attempts/attempt_001"
    )
    first_attempt.mkdir(parents=True)
    (first_attempt / "predictions.parquet").write_bytes(b"predictions")
    second_attempt = first_attempt.with_name("attempt_002")
    second_attempt.mkdir()
    (
        output_root / "training/model_a/fold_2024/seed_7/attempts"
    ).mkdir(parents=True)

    summary = quality.invalidate_study_design(
        study_path=study_path,
        output_root=output_root,
        defect_class="objective_cannot_answer_stated_question",
        statement=DESIGN_STATEMENT,
        structural_argument=DESIGN_ARGUMENT,
        evidence_paths=[evidence_path],
        inspected_alternatives=["direct utility target", "survival evaluator"],
    )

    record_path = Path(summary["record"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    declared_hash = record.pop("design_invalidation_sha256")
    assert declared_hash == quality._canonical_json_sha256(record)
    assert summary["design_invalidation_sha256"] == declared_hash
    assert summary["burned_fold_years"] == [2023]
    assert summary["trained_cell_count"] == 2
    assert summary["completed_cell_count"] == 1
    assert record["burned_fold_years"] == [2023]
    assert record["trained_cells"] == [
        "model_a/fold_2023/seed_7/attempt_001",
        "model_a/fold_2023/seed_7/attempt_002",
    ]
    assert record["completed_cells"] == [
        "model_a/fold_2023/seed_7/attempt_001"
    ]
    assert record["evidence"] == [
        {
            "path": "evidence/design_analysis.txt",
            "sha256": quality._file_sha256(evidence_path),
        }
    ]
    pointer_path = output_root / "design_invalidation/current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert pointer["record"] == str(record_path.resolve())
    assert pointer["record_sha256"] == quality._file_sha256(record_path)


def test_invalidate_study_design_rejects_unknown_defect_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path, output_root, _workspace = _design_invalidation_fixture(
        tmp_path, monkeypatch
    )
    with pytest.raises(ValueError, match="defect_class must be one of"):
        quality.invalidate_study_design(
            study_path=study_path,
            output_root=output_root,
            defect_class="replace_the_target",
            statement=DESIGN_STATEMENT,
            structural_argument=DESIGN_ARGUMENT,
            evidence_paths=[study_path],
            inspected_alternatives=["alternative"],
        )


def test_invalidate_study_design_rejects_short_written_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path, output_root, _workspace = _design_invalidation_fixture(
        tmp_path, monkeypatch
    )
    common = {
        "study_path": study_path,
        "output_root": output_root,
        "defect_class": "primary_metric_misspecified",
        "evidence_paths": [study_path],
        "inspected_alternatives": ["alternative"],
    }
    with pytest.raises(ValueError, match="statement must contain at least 80"):
        quality.invalidate_study_design(
            **common,
            statement="too short",
            structural_argument=DESIGN_ARGUMENT,
        )
    with pytest.raises(
        ValueError, match="structural_argument must contain at least 80"
    ):
        quality.invalidate_study_design(
            **common,
            statement=DESIGN_STATEMENT,
            structural_argument="too short",
        )


def test_invalidate_study_design_rejects_missing_evidence_or_alternatives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path, output_root, workspace = _design_invalidation_fixture(
        tmp_path, monkeypatch
    )
    common = {
        "study_path": study_path,
        "output_root": output_root,
        "defect_class": "evidence_power_insufficient",
        "statement": DESIGN_STATEMENT,
        "structural_argument": DESIGN_ARGUMENT,
    }
    with pytest.raises(ValueError, match="evidence_paths must contain"):
        quality.invalidate_study_design(
            **common,
            evidence_paths=[],
            inspected_alternatives=["alternative"],
        )
    with pytest.raises(ValueError, match="evidence path must exist and be a file"):
        quality.invalidate_study_design(
            **common,
            evidence_paths=[workspace / "missing.txt"],
            inspected_alternatives=["alternative"],
        )
    with pytest.raises(ValueError, match="inspected_alternatives must contain"):
        quality.invalidate_study_design(
            **common,
            evidence_paths=[study_path],
            inspected_alternatives=[],
        )


def test_invalidate_study_design_rejects_completed_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path, output_root, _workspace = _design_invalidation_fixture(
        tmp_path, monkeypatch
    )
    report_path = output_root / "evaluation/attempt_001/evaluation_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    pointer_path = output_root / "evaluation/current.json"
    pointer_path.write_text(
        json.dumps(
            {
                "evaluation_report": str(report_path.resolve()),
                "evaluation_report_sha256": quality._file_sha256(report_path),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="illegal once a formal evaluation verdict exists"
    ):
        quality.invalidate_study_design(
            study_path=study_path,
            output_root=output_root,
            defect_class="evaluator_semantics_inconsistent",
            statement=DESIGN_STATEMENT,
            structural_argument=DESIGN_ARGUMENT,
            evidence_paths=[study_path],
            inspected_alternatives=["alternative"],
        )


def test_terminal_study_evidence_prefers_and_verifies_design_invalidation(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    record_path = (
        output_root
        / "design_invalidation/attempt_001/design_invalidation.json"
    )
    record_path.parent.mkdir(parents=True)
    record = {
        "artifact_type": "seq100_signal_quality_design_invalidation",
        "status": "research_design_insufficient",
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")
    pointer_path = output_root / "design_invalidation/current.json"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(
        json.dumps(
            {
                "record": str(record_path.resolve()),
                "record_sha256": quality._file_sha256(record_path),
            }
        ),
        encoding="utf-8",
    )

    status, evidence_path, evidence = quality._terminal_study_evidence(
        study={}, output_root=output_root
    )
    assert status == "research_design_insufficient"
    assert evidence_path == record_path.resolve()
    assert evidence == record

    record_path.write_text(json.dumps({**record, "mutated": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="design invalidation record changed"):
        quality._terminal_study_evidence(study={}, output_root=output_root)


def test_target_field_names_are_unique() -> None:
    assert len(quality.TARGET_FLOAT_FIELDS) == len(set(quality.TARGET_FLOAT_FIELDS))


def test_contract_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = quality.load_study()
    payload["contract"]["objective"] = "tampered"
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract SHA-256 mismatch"):
        quality.load_study(path)


def test_training_budget_amendment_is_exact_and_hash_bound(tmp_path: Path) -> None:
    study = quality.load_study()
    path = quality._training_budget_amendment_path(tmp_path)
    path.parent.mkdir(parents=True)
    payload = _budget_amendment_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = quality._load_training_budget_amendment(study, tmp_path)
    assert loaded == payload

    payload["config_changes"]["learning_rate"] = {"from": 0.001, "to": 0.002}
    payload["amendment_sha256"] = quality._canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "amendment_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported fields"):
        quality._load_training_budget_amendment(study, tmp_path)


def test_next_open_reanchor_uses_executed_open_denominator() -> None:
    path = np.zeros((1, 20, 4), dtype=np.float32)
    path[:, :, :] = 0.10
    path[:, 0, 0] = 0.05
    converted = quality.reanchor_to_actual_next_open(
        path, source_price_anchor="today_close"
    )
    assert converted[0, 0, 0] == pytest.approx(0.0, abs=2.0e-6)
    assert converted[0, 0, 3] == pytest.approx(1.10 / 1.05 - 1.0)

    invalid = path.copy()
    invalid[0, 0, 0] = np.nan
    tolerated = quality.reanchor_to_actual_next_open(
        invalid,
        source_price_anchor="today_close",
        required_mask=np.array([False]),
    )
    assert np.isnan(tolerated).all()
    with pytest.raises(ValueError, match="filled entry"):
        quality.reanchor_to_actual_next_open(
            invalid,
            source_price_anchor="today_close",
            required_mask=np.array([True]),
        )


def test_cost_t_plus_one_and_terminal_zero_recovery() -> None:
    path = _flat_path(close_return=0.10)
    multiplier = np.full((1, 20), 0.98, dtype=np.float32)
    sellable = np.zeros((1, 20), dtype=bool)
    sellable[:, 0] = True
    descriptor = quality.compute_path_descriptors(
        path,
        multiplier,
        exit_sellable=sellable,
        terminal_failure=np.array([False]),
    )
    expected_log = np.log(1.10 * 0.98)
    assert descriptor["close_net_log"][0, 19] == pytest.approx(expected_log)
    assert bool(descriptor["no_legal_sell"][0])

    delisted = np.zeros((1, 20), dtype=bool)
    delisted[0, 7:] = True
    recovered, failure = quality.apply_terminal_zero_recovery(path, delisted)
    assert bool(failure[0])
    np.testing.assert_array_equal(recovered[0, 7:, :4], -1.0)


def test_common_utility_and_four_target_candidates_keep_unfilled_cash() -> None:
    paths = np.stack(
        [
            _flat_path(close_return=0.15)[0],
            _flat_path(close_return=0.03)[0],
            _flat_path(close_return=-0.20)[0],
        ]
    )
    multiplier = np.ones((3, 20), dtype=np.float32)
    sellable = np.ones((3, 20), dtype=bool)
    delisted = np.zeros((3, 20), dtype=bool)
    descriptor = quality.compute_path_descriptors(
        paths,
        multiplier,
        exit_sellable=sellable,
        terminal_failure=np.zeros(3, dtype=bool),
    )
    result = quality.compute_target_candidate_qualities(
        descriptor,
        entry_relative_path=paths,
        growth_multiplier=multiplier,
        exit_sellable=sellable,
        delisted_path=delisted,
        entry_filled=np.array([True, True, False]),
        path_available=np.ones(3, dtype=bool),
        forced_low=np.zeros(3, dtype=bool),
        symbol_idx=np.array([1, 2, 3]),
        pareto_device="cpu",
    )
    assert result["common_sustained_action_utility"][2] == pytest.approx(0.0)
    for name in (
        "raw_path_distribution_quality",
        "pareto_ordinal_quality",
        "competing_risk_quality",
        "direct_listwise_utility_quality",
    ):
        assert result[name].shape == (3,)
        assert np.isfinite(result[name]).sum() >= 2


def test_exact_pareto_counts_and_deterministic_symbol_tie_break() -> None:
    values = np.array(
        [[3.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.0, 3.0]],
        dtype=np.float32,
    )
    dominated, dominating = quality.pareto_dominance_counts(
        values, device="cpu", block_size=2
    )
    np.testing.assert_array_equal(dominated, [3, 0, 0, 0])
    np.testing.assert_array_equal(dominating, [0, 1, 1, 1])

    tied = np.tile(np.arange(1.0, 8.0, dtype=np.float32), (3, 1))
    result = quality.compute_daily_relevance(
        _descriptor_fixture(tied),
        entry_filled=np.ones(3, dtype=bool),
        path_available=np.ones(3, dtype=bool),
        forced_low=np.zeros(3, dtype=bool),
        symbol_idx=np.array([30, 10, 20]),
        pareto_device="cpu",
    )
    np.testing.assert_allclose(result["conditional_relevance"], [1.0, 0.0, 0.5])


def test_d20_endpoint_cutoff_hard_rejects_2026() -> None:
    dates = [
        np.datetime_as_string(value, unit="D")
        for value in np.arange(
            np.datetime64("2025-11-20"),
            np.datetime64("2026-01-20"),
            dtype="datetime64[D]",
        )
    ]
    assert quality.label_endpoint_is_allowed(
        signal_date_idx=0,
        date_values=dates,
        horizon_days=20,
        cutoff="2025-12-31",
    )
    assert not quality.label_endpoint_is_allowed(
        signal_date_idx=30,
        date_values=dates,
        horizon_days=20,
        cutoff="2025-12-31",
    )
    with pytest.raises(ValueError, match="2026"):
        quality.label_endpoint_is_allowed(
            signal_date_idx=0,
            date_values=dates,
            horizon_days=20,
            cutoff="2026-01-01",
        )


def test_rolling_and_cross_section_features_are_causal() -> None:
    rng = np.random.default_rng(7)
    values = np.exp(
        np.cumsum(rng.normal(0.0, 0.01, size=(100, 3)), axis=0)
    ).astype(np.float32)
    changed = values.copy()
    changed[71:] *= 50.0
    before = quality._rolling_linear_stats(np.log(values), 20)[0]
    after = quality._rolling_linear_stats(np.log(changed), 20)[0]
    np.testing.assert_allclose(before[:71], after[:71], equal_nan=True)

    dates = np.array([1, 1, 1, 2, 2], dtype=np.int32)
    ranked = quality._candidate_cross_section_percentile(
        np.array([3.0, 1.0, 2.0, 100.0, -100.0], dtype=np.float32), dates
    )
    np.testing.assert_allclose(ranked, [1.0, 0.0, 0.5, 1.0, 0.0])


def test_feature_memmap_is_closed_before_atomic_publish(tmp_path: Path) -> None:
    source = tmp_path / "source.dat"
    destination = tmp_path / "destination.dat"
    values = np.memmap(source, dtype="float32", mode="w+", shape=(8,))
    values[:] = np.arange(8, dtype=np.float32)
    quality._close_memmap(values)
    source.replace(destination)
    assert destination.stat().st_size == 8 * 4


def test_lightgbm_sequence_uses_double_sampling_contract() -> None:
    continuous = np.arange(24, dtype=np.float32).reshape(8, 3)
    categorical = np.asarray([[1], [2], [1], [0], [2], [1], [0], [2]], dtype=np.int64)
    sequence = quality._make_lgb_sequence(
        continuous=continuous,
        categorical=categorical,
        row_ids=np.arange(8, dtype=np.int64),
        continuous_columns=np.asarray([0, 2], dtype=np.int32),
        categorical_columns=np.asarray([0], dtype=np.int32),
        category_vocabularies=[np.asarray([1, 2], dtype=np.int64)],
        batch_size=4,
    )
    assert sequence[0].dtype == np.float64
    assert sequence[:4].dtype == np.float64
    dataset = quality._lgb_dataset(
        sequence=sequence,
        label=np.arange(8, dtype=np.float64),
        feature_names=["left", "right", "category"],
        categorical_count=1,
    )
    dataset.construct()
    assert dataset.num_data() == 8


def test_completed_model_screen_task_is_reused_only_with_valid_hashes(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    output_dir = model_dir / "prescreen"
    output_dir.mkdir(parents=True)
    expected_config = {
        "model_id": "example",
        "resolved_config_sha256": "resolved",
        "resume_key_sha256": "resume",
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(expected_config), encoding="utf-8"
    )
    checkpoint = output_dir / "best_model.pt"
    prediction = output_dir / "predictions.parquet"
    daily = output_dir / "daily_metrics.parquet"
    checkpoint.write_bytes(b"checkpoint")
    prediction.write_bytes(b"prediction")
    daily.write_bytes(b"daily")
    summary = {
        "status": "completed",
        "model_id": "example",
        "fold_id": "screen",
        "fold_year": 2022,
        "seed": 7,
        "resume_key_sha256": "resume",
        "resolved_config_sha256": "resolved",
        "adapter": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": quality._file_sha256(checkpoint),
        },
        "prediction": {
            "path": str(prediction),
            "sha256": quality._file_sha256(prediction),
            "candidate_count": 11,
        },
        "daily_metrics": {
            "path": str(daily),
            "sha256": quality._file_sha256(daily),
        },
        "metrics": {"candidate_count": 11},
        "protection_before": {"protected": "same"},
        "protection_after": {"protected": "same"},
    }
    (output_dir / "task_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    loaded = quality._load_completed_screen_task(
        model_dir,
        model_id="example",
        expected_config=expected_config,
        expected_test_year=2022,
        expected_candidate_count=11,
    )
    assert loaded is not None
    assert loaded[0] == output_dir

    checkpoint.write_bytes(b"tampered")
    assert quality._load_completed_screen_task(
        model_dir,
        model_id="example",
        expected_config=expected_config,
        expected_test_year=2022,
        expected_candidate_count=11,
    ) is None


def test_screen_retry_paths_preserve_interrupted_outputs(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert quality._next_screen_retry_dir(model_dir, "prescreen") == model_dir / "prescreen"
    (model_dir / "prescreen").mkdir()
    assert quality._next_screen_retry_dir(model_dir, "prescreen") == (
        model_dir / "prescreen_retry_001"
    )
    (model_dir / "prescreen_retry_001").mkdir()
    assert quality._next_screen_retry_dir(model_dir, "prescreen") == (
        model_dir / "prescreen_retry_002"
    )


def test_budget_extension_accepts_only_the_frozen_budget_delta() -> None:
    amendment = _budget_amendment_payload()
    source = {
        "model_id": "patchtst_student_t_path",
        "fold_id": "screen",
        "seed": 7,
        "feature_sha256": "feature",
        "execution_semantics_version": "exact_effective_batch_global_loss_v2",
        "micro_batch": 128,
        "frozen_model_config": {
            "learning_rate": 0.0005,
            "max_epochs": 3,
            "patience": 1,
        },
        "resolved_config_sha256": "old-resolved",
        "resume_key_sha256": "old-resume",
    }
    expected = {
        **source,
        "frozen_model_config": {
            "learning_rate": 0.0005,
            "max_epochs": 10,
            "patience": 2,
        },
        "training_budget_amendment_sha256": amendment["amendment_sha256"],
        "training_budget_amended_fields": ["max_epochs", "patience"],
        "resolved_config_sha256": "new-resolved",
        "resume_key_sha256": "new-resume",
    }
    compatibility = quality._budget_extension_compatibility(
        source,
        expected,
        amendment,
    )
    assert compatibility is not None
    assert compatibility["config_changes"] == amendment["config_changes"]

    changed_learning_rate = json.loads(json.dumps(expected))
    changed_learning_rate["frozen_model_config"]["learning_rate"] = 0.001
    assert quality._budget_extension_compatibility(
        source,
        changed_learning_rate,
        amendment,
    ) is None
    changed_fold = dict(expected)
    changed_fold["fold_id"] = "2023"
    assert quality._budget_extension_compatibility(
        source,
        changed_fold,
        amendment,
    ) is None


def test_budget_extension_migrates_checkpoint_without_mutating_source(
    tmp_path: Path,
) -> None:
    amendment = _budget_amendment_payload()
    source_dir = tmp_path / "attempt_001" / "model" / "prescreen"
    source_dir.mkdir(parents=True)
    source_checkpoint = {
        "schema": quality.TRAINING_CHECKPOINT_VERSION,
        "saved_at": "before",
        "resume_key_sha256": "old-resume",
        "execution_semantics_version": "exact_effective_batch_global_loss_v2",
        "model_state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer_state_dict": {"state": {0: {"step": torch.tensor(3.0)}}},
        "scaler_state_dict": {"scale": 1.0},
        "epoch": 4,
        "next_step_index": 0,
        "best_epoch": 3,
        "best_development_loss": -1.25,
        "best_state": {"weight": torch.tensor([0.5, 1.5])},
        "epochs_without_improvement": 0,
        "training_log": [{"epoch": 3, "development_loss": -1.25}],
        "epoch_loss_sum": 0.0,
        "epoch_sample_count": 0,
        "rng_state": {"python": (3, (), None)},
    }
    quality._atomic_torch_save(source_dir / "last_checkpoint.pt", source_checkpoint)
    preprocessor = b'{"preprocessor_sha256":"same"}\n'
    (source_dir / "snapshot_preprocessor.json").write_bytes(preprocessor)
    source_hash = quality._file_sha256(source_dir / "last_checkpoint.pt")
    expected = {
        "resume_key_sha256": "new-resume",
        "training_budget_amendment_sha256": amendment["amendment_sha256"],
    }
    destination = tmp_path / "attempt_002" / "model" / "prescreen"
    provenance = quality._prepare_budget_extension_directory(
        output_dir=destination,
        source={
            "source_dir": source_dir,
            "source_config": {
                "resolved_config_sha256": "old-resolved",
                "resume_key_sha256": "old-resume",
            },
            "source_checkpoint": source_checkpoint,
            "compatibility": {
                "model_id": "patchtst_student_t_path",
                "fold_id": "screen",
                "seed": 7,
                "config_changes": amendment["config_changes"],
            },
        },
        expected_config=expected,
        amendment=amendment,
    )
    migrated = torch.load(
        destination / "last_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert migrated["resume_key_sha256"] == "new-resume"
    torch.testing.assert_close(
        migrated["model_state_dict"]["weight"],
        source_checkpoint["model_state_dict"]["weight"],
    )
    assert migrated["optimizer_state_dict"]["state"][0]["step"].item() == 3.0
    assert migrated["rng_state"] == source_checkpoint["rng_state"]
    assert (destination / "snapshot_preprocessor.json").read_bytes() == preprocessor
    assert quality._file_sha256(source_dir / "last_checkpoint.pt") == source_hash
    assert provenance["source_checkpoint_sha256"] == source_hash
    check = dict(provenance)
    declared = check.pop("provenance_sha256")
    assert declared == quality._canonical_json_sha256(check)


def test_only_complete_publish_failure_is_recoverable(tmp_path: Path) -> None:
    attempt = tmp_path / "feature_view" / "attempt_001"
    partial = attempt / "partial"
    partial.mkdir(parents=True)
    partial_paths = quality._feature_view_paths(partial)
    final_paths = quality._feature_view_paths(attempt)
    candidate_count = 3
    partial_paths.candidate_date_idx.write_bytes(bytes(candidate_count * 4))
    partial_paths.candidate_symbol_idx.write_bytes(bytes(candidate_count * 4))
    final_paths.continuous.write_bytes(bytes(candidate_count * 2 * 4))
    final_paths.categorical.write_bytes(bytes(candidate_count * 1 * 8))
    (attempt / "progress.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "error_type": "PermissionError",
                "error": "partial candidate_date_idx publish failed",
            }
        ),
        encoding="utf-8",
    )
    assert quality._recoverable_feature_publish_attempt(
        tmp_path,
        candidate_count=candidate_count,
        continuous_feature_count=2,
        categorical_feature_count=1,
    ) == attempt

    partial_paths.candidate_symbol_idx.write_bytes(b"short")
    assert quality._recoverable_feature_publish_attempt(
        tmp_path,
        candidate_count=candidate_count,
        continuous_feature_count=2,
        categorical_feature_count=1,
    ) is None


def test_qdp_source_date_after_signal_is_rejected(tmp_path: Path) -> None:
    shard = tmp_path / "domain.parquet"
    pd.DataFrame(
        {
            "trade_date": ["2020-01-02"],
            "symbol": ["000001.SZ"],
            "value": [1.0],
            "source_date": ["2020-01-03"],
        }
    ).to_parquet(shard, index=False)
    dataset_manifest = tmp_path / "dataset.json"
    dataset_manifest.write_text(
        json.dumps({"shards": [{"path": str(shard)}]}), encoding="utf-8"
    )
    study = {
        "contract": {
            "data": {
                "qdp_datasets": {
                    "fake": {"manifest_path": str(dataset_manifest)}
                }
            }
        }
    }
    manifest = {
        "date_values": ["2020-01-02"],
        "symbol_values": ["000001.SZ"],
    }
    with pytest.raises(ValueError, match="source_date after signal date"):
        quality._load_qdp_dense_domain(
            study,
            manifest,
            domain="fake",
            numeric_fields=("value",),
            source_guards={"value": "source_date"},
        )


def test_student_t_quality_and_quantiles_are_finite_and_monotone() -> None:
    location = np.zeros((4, 20), dtype=np.float32)
    scale = np.full((4, 20), 0.05, dtype=np.float32)
    terminal = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    result = quality.student_t_path_quality(location, scale, terminal)
    assert np.isfinite(result).all()
    assert result[-1] == pytest.approx(0.0)
    q = location[:, :, None] + scale[:, :, None] * np.asarray(
        [-1.5, 0.0, 1.5], dtype=np.float32
    )
    assert np.all(q[:, :, 0] <= q[:, :, 1])
    assert np.all(q[:, :, 1] <= q[:, :, 2])


def test_student_t_explicit_nll_matches_torch_distribution() -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(8, 20, generator=generator)
    location = torch.randn(8, 20, generator=generator)
    scale = torch.rand(8, 20, generator=generator) + 0.05
    expected = -torch.distributions.StudentT(
        df=torch.tensor(5.0),
        loc=location,
        scale=scale,
    ).log_prob(target)
    actual = quality.student_t_nll(target, location, scale, df=5.0)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_effective_batch_plan_is_exact_and_micro_batch_independent() -> None:
    rows = np.arange(13, dtype=np.int64)
    dates = np.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4], dtype=np.int32)
    first = quality.effective_batch_plan(
        rows,
        dates,
        effective_batch=5,
        seed=19,
    )
    second = quality.effective_batch_plan(
        rows,
        dates,
        effective_batch=5,
        seed=19,
    )
    assert [item.tolist() for item in first] == [item.tolist() for item in second]
    assert [len(item) for item in first] == [5, 5, 3]
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), rows)
    for effective in first:
        micro_two = np.concatenate(list(quality._micro_batches(effective, 2)))
        micro_four = np.concatenate(list(quality._micro_batches(effective, 4)))
        np.testing.assert_array_equal(micro_two, effective)
        np.testing.assert_array_equal(micro_four, effective)


def test_f0_cross_date_batch_preserves_requested_candidate_order() -> None:
    continuous = np.arange(182 * 2, dtype=np.float32).reshape(182, 2, 1)
    binary = (np.arange(182 * 2).reshape(182, 2) % 2 == 0)
    context = quality.ModelDataContext(
        study={},
        research_freeze={},
        pack_manifest={},
        feature_manifest={},
        target_manifest={},
        continuous=np.empty((0, 0), dtype=np.float32),
        categorical=np.empty((0, 0), dtype=np.int64),
        candidate_date_idx=np.asarray([180, 181], dtype=np.int32),
        candidate_symbol_idx=np.asarray([0, 1], dtype=np.int32),
        target=np.empty((0, 0), dtype=np.float32),
        grades=np.empty(0, dtype=np.uint8),
        flags=np.empty(0, dtype=np.uint8),
        candidate_index_path=Path("unused"),
    )
    rows = np.asarray([1, 0], dtype=np.int64)
    batch = quality._f0_batch(
        context,
        rows,
        {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        material=([continuous], [binary]),
    )
    assert batch.shape == (2, 180, 2)
    assert batch[0, -1, 0] == continuous[181, 1, 0]
    assert batch[1, -1, 0] == continuous[180, 0, 0]
    assert batch[0, -1, 1] == float(binary[181, 1])
    assert batch[1, -1, 1] == float(binary[180, 0])


def test_global_loss_denominator_makes_microbatch_gradients_equivalent() -> None:
    values = torch.linspace(-1.0, 1.0, 17)
    target = torch.linspace(0.5, -0.5, 17)
    mask = torch.tensor(
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        dtype=torch.float32,
    )
    denominator = float(mask.sum())

    full = values.clone().requires_grad_(True)
    full_loss = quality.global_normalized_component(
        (torch.nn.functional.smooth_l1_loss(full, target, reduction="none") * mask).sum(),
        denominator,
    )
    full_loss.backward()

    split = values.clone().requires_grad_(True)
    for start in range(0, len(split), 4):
        stop = min(start + 4, len(split))
        part = quality.global_normalized_component(
            (
                torch.nn.functional.smooth_l1_loss(
                    split[start:stop], target[start:stop], reduction="none"
                )
                * mask[start:stop]
            ).sum(),
            denominator,
        )
        part.backward()
    torch.testing.assert_close(split.grad, full.grad)


def test_complete_date_step_plan_never_splits_a_signal_date() -> None:
    rows = np.arange(12, dtype=np.int64)
    dates = np.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=np.int32)
    steps = quality.complete_date_step_plan(
        rows,
        dates,
        target_candidates=5,
        seed=7,
    )
    flattened = np.concatenate([group for step in steps for group in step])
    np.testing.assert_array_equal(np.sort(flattened), rows)
    for step in steps:
        for group in step:
            assert len(np.unique(dates[group])) == 1
            date = int(dates[group[0]])
            np.testing.assert_array_equal(np.sort(group), np.flatnonzero(dates == date))


def test_last_checkpoint_roundtrip_and_pause_sentinel(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    inputs = torch.randn(5, 3)
    loss = model(inputs).square().mean()
    loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    resolved = {
        "resume_key_sha256": "resume-key",
        "execution_semantics_version": "test-v1",
    }
    checkpoint = quality._save_last_training_checkpoint(
        output_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resolved_config=resolved,
        epoch=2,
        next_step_index=3,
        best_epoch=1,
        best_development_loss=0.25,
        best_state=expected,
        epochs_without_improvement=0,
        training_log=[{"epoch": 1, "development_loss": 0.25}],
    )
    assert checkpoint.exists()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    payload = quality._load_last_training_checkpoint(
        output_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resolved_config=resolved,
    )
    assert payload is not None
    assert payload["epoch"] == 2
    assert payload["next_step_index"] == 3
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    with pytest.raises(ValueError, match="resume key mismatch"):
        quality._load_last_training_checkpoint(
            output_dir=tmp_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            resolved_config={
                "resume_key_sha256": "different",
                "execution_semantics_version": "test-v1",
            },
        )

    monitor_dir = tmp_path / "monitor"
    monitor = quality._TrainingMonitor(
        output_dir=monitor_dir,
        base={"model_id": "test", "fold_id": "screen"},
        checkpoint_seconds=10_000,
    )
    monitor.pause_path.parent.mkdir(parents=True, exist_ok=True)
    monitor.pause_path.write_text("pause", encoding="utf-8")

    def _checkpoint_callback() -> Path:
        path = monitor_dir / "checkpoint.pt"
        path.write_bytes(b"safe")
        return path

    with pytest.raises(quality.TrainingPaused):
        quality._checkpoint_or_pause(
            monitor=monitor,
            force_checkpoint=False,
            checkpoint_callback=_checkpoint_callback,
            progress={
                "phase": "training",
                "samples_completed": 10,
                "samples_total": 100,
                "epoch": 1,
                "max_epochs": 3,
                "step": 2,
                "steps_total": 20,
            },
        )
    assert not monitor.pause_path.exists()
    progress = json.loads((monitor_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "paused"
    assert (monitor_dir / "training_events.jsonl").exists()


def test_training_monitor_trims_on_shared_cadence_and_records_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_calls: list[bool] = []
    monkeypatch.setattr(
        quality.sequence_training,
        "_available_physical_memory_gb",
        lambda: 1.5,
    )
    monkeypatch.setattr(
        quality.sequence_training,
        "_current_process_memory_gb",
        lambda: {"working_set_gb": 7.0, "private_gb": 4.0},
    )
    monkeypatch.setattr(
        quality.sequence_training,
        "_trim_working_set",
        lambda: trim_calls.append(True) or True,
    )
    monitor = quality._TrainingMonitor(output_dir=tmp_path, base={})

    for _ in range(31):
        assert monitor.relieve_memory_pressure() is None
    assert monitor.relieve_memory_pressure() is not None
    for _ in range(255):
        assert monitor.relieve_memory_pressure() is None
    assert monitor.relieve_memory_pressure() is not None

    events = [
        json.loads(line)
        for line in (tmp_path / "training_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(trim_calls) == 2
    assert len(events) == len(trim_calls)
    assert [event["event"] for event in events] == [
        "working_set_trim",
        "working_set_trim",
    ]
    assert [event["batch"] for event in events] == [32, 288]


def test_checkpoint_pauses_and_reports_persistent_low_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = quality._TrainingMonitor(
        output_dir=tmp_path,
        base={"model_id": "test", "fold_id": "screen"},
        checkpoint_seconds=10_000,
    )
    monitor.trim_step = 31
    trim_calls: list[bool] = []
    checkpoint_calls: list[Path] = []
    monkeypatch.setattr(
        quality.sequence_training,
        "_available_physical_memory_gb",
        lambda: 0.5,
    )
    monkeypatch.setattr(
        quality.sequence_training,
        "_current_process_memory_gb",
        lambda: {"working_set_gb": 8.0, "private_gb": 5.0},
    )
    monkeypatch.setattr(
        quality.sequence_training,
        "_trim_working_set",
        lambda: trim_calls.append(True) or True,
    )

    def _checkpoint_callback() -> Path:
        path = tmp_path / "checkpoint.pt"
        path.write_bytes(b"safe")
        checkpoint_calls.append(path)
        return path

    with pytest.raises(quality.TrainingPaused, match="low available memory"):
        quality._checkpoint_or_pause(
            monitor=monitor,
            force_checkpoint=False,
            checkpoint_callback=_checkpoint_callback,
            progress={"phase": "training", "samples_completed": 10},
        )

    assert checkpoint_calls == [tmp_path / "checkpoint.pt"]
    assert len(trim_calls) == 2
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "paused"
    assert progress["memory_pause"] == {
        "available_before_gb": 0.5,
        "available_after_gb": 0.5,
        "trim_succeeded": True,
        "working_set_gb": 8.0,
        "private_gb": 5.0,
        "pause_threshold_gb": quality.LOW_MEMORY_PAUSE_AVAILABLE_GB,
    }


def test_checkpoint_healthy_memory_and_requested_pause_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = quality._TrainingMonitor(
        output_dir=tmp_path,
        base={"model_id": "test", "fold_id": "screen"},
        checkpoint_seconds=10_000,
    )
    monitor.trim_step = 31
    monkeypatch.setattr(
        quality.sequence_training,
        "_available_physical_memory_gb",
        lambda: 3.0,
    )
    checkpoint_calls: list[Path] = []

    def _checkpoint_callback() -> Path:
        path = tmp_path / "checkpoint.pt"
        path.write_bytes(b"safe")
        checkpoint_calls.append(path)
        return path

    quality._checkpoint_or_pause(
        monitor=monitor,
        force_checkpoint=False,
        checkpoint_callback=_checkpoint_callback,
        progress={"phase": "training"},
    )
    assert checkpoint_calls == []

    monitor.pause_path.parent.mkdir(parents=True, exist_ok=True)
    monitor.pause_path.write_text("pause", encoding="utf-8")
    with pytest.raises(quality.TrainingPaused, match="training paused safely"):
        quality._checkpoint_or_pause(
            monitor=monitor,
            force_checkpoint=False,
            checkpoint_callback=_checkpoint_callback,
            progress={"phase": "training"},
        )
    assert checkpoint_calls == [tmp_path / "checkpoint.pt"]
    assert not monitor.pause_path.exists()
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "paused"
    assert "memory_pause" not in progress


def test_memory_exhaustion_ignores_missing_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trim_calls: list[bool] = []
    monkeypatch.setattr(
        quality.sequence_training,
        "_available_physical_memory_gb",
        lambda: None,
    )
    monkeypatch.setattr(
        quality.sequence_training,
        "_trim_working_set",
        lambda: trim_calls.append(True) or True,
    )
    monitor = quality._TrainingMonitor(output_dir=tmp_path, base={})

    assert monitor.memory_exhausted() is None
    assert trim_calls == []


def test_deephit_joint_distribution_is_finite_and_normalized() -> None:
    config = _model_configs()["deephit_competing_risk"]
    model = quality._DeepHitCompetingRisk(
        n_num_features=12,
        cat_cardinalities=[4, 7],
        config=config,
    )
    output = model(
        torch.randn(16, 12),
        torch.stack([torch.randint(0, 4, (16,)), torch.randint(0, 7, (16,))], 1),
    )
    probability = torch.softmax(output["joint_logits"], dim=1)
    torch.testing.assert_close(probability.sum(dim=1), torch.ones(16))
    score = quality.deephit_expected_event_score(probability)
    assert torch.isfinite(score).all()
    targets = quality.deephit_joint_targets(
        torch.tensor([0, 1, 5]), torch.tensor([20, 1, 20])
    )
    torch.testing.assert_close(targets, torch.tensor([100, 0, 99]))


def test_deepsets_is_permutation_equivariant() -> None:
    config = _model_configs()["market_industry_deepsets"]
    model = quality._MarketIndustryDeepSets(
        n_num_features=10,
        cat_cardinalities=[5, 3],
        industry_position=0,
        config=config,
    ).eval()
    x_num = torch.randn(24, 10)
    x_cat = torch.stack(
        [torch.randint(0, 5, (24,)), torch.randint(0, 3, (24,))], dim=1
    )
    permutation = torch.randperm(24)
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        original = model(x_num, x_cat)
        permuted = model(x_num[permutation], x_cat[permutation])
    torch.testing.assert_close(original["quality"], permuted["quality"][inverse])
    torch.testing.assert_close(
        original["fill_logit"], permuted["fill_logit"][inverse]
    )


def test_neural_sort_is_row_stochastic_and_loss_is_finite() -> None:
    scores = torch.randn(32, requires_grad=True)
    matrix = quality.neural_sort_matrix(scores, temperature=1.0)
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(32))
    loss = quality.neural_ndcg_loss(
        scores,
        torch.randint(0, 5, (32,)),
        temperature=1.0,
        gain=[0, 1, 3, 7, 15],
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_all_frozen_adapters_expose_the_unified_interface() -> None:
    assert tuple(quality.MODEL_ADAPTERS) == quality.MODEL_IDS
    for model_id in quality.MODEL_IDS:
        adapter = quality.get_model_adapter(model_id)
        assert adapter.model_id == model_id
        assert "fill" in adapter.capabilities
        assert callable(adapter.fit)
        assert callable(adapter.predict)


def test_stratified_block_bootstrap_is_deterministic() -> None:
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    left = pd.DataFrame(
        {
            "trade_date": dates.astype(str),
            "year": dates.year,
            "top1pct_u_mean": np.linspace(0.01, 0.03, len(dates)),
        }
    )
    right = left.copy()
    right["top1pct_u_mean"] -= 0.005
    first = quality._stratified_block_bootstrap_delta(
        left, right, iterations=200, block_days=20, seed=7
    )
    second = quality._stratified_block_bootstrap_delta(
        left, right, iterations=200, block_days=20, seed=7
    )
    assert first == second
    assert first["lower"] > 0.0
