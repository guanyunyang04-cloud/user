from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_quality_liquidity_model as model


def _fake_inputs() -> model.ModelInputs:
    inputs = object.__new__(model.ModelInputs)
    inputs.candidate_ids = np.arange(10, dtype=np.int64)
    inputs.date_idx = np.asarray([0, 1, 20, 21, 22, 23, 35, 36, 37, 38], dtype=np.int32)
    inputs.years = np.asarray(
        [2011, 2011, 2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
        dtype=np.int16,
    )
    inputs.labels = np.full((10, 15), np.nan, dtype=np.float32)
    inputs.labels[:, 4] = np.linspace(0.01, 0.10, 10)
    inputs.labels[:, 5] = -np.linspace(0.01, 0.10, 10)
    inputs.labels[:, 7] = np.linspace(0.02, 0.20, 10)
    inputs.labels[:, 8] = -np.linspace(0.02, 0.20, 10)
    inputs.flags = np.zeros((10, 5), dtype=np.uint16)
    inputs.flags[:, 1] = model.FLAG_MFE_PRE_PEAK_MAE_VALID | model.FLAG_STATE_ASSIGNED
    inputs.flags[:, 2] = model.FLAG_MFE_PRE_PEAK_MAE_VALID
    inputs.flags[3, 1] = 0
    inputs.states = np.full((10, 5), -1, dtype=np.int8)
    inputs.states[:, 1] = np.arange(10, dtype=np.int8) % 3
    inputs.label_manifest = {
        "label_columns": [
            "g_5",
            "mfe_5",
            "pre_peak_mae_5",
            "g_10",
            "mfe_10",
            "pre_peak_mae_10",
            "g_20",
            "mfe_20",
            "pre_peak_mae_20",
            "g_40",
            "mfe_40",
            "pre_peak_mae_40",
            "g_60",
            "mfe_60",
            "pre_peak_mae_60",
        ]
    }
    return inputs


def _metric_results(kind: str, *, improvement: float) -> tuple[dict, dict]:
    reference: dict[int, dict] = {}
    candidate: dict[int, dict] = {}
    for year in model.ROLLING_YEARS:
        if kind == "mfe":
            base = {
                "rank_ic": 0.05,
                "top5_mfe_mean": 0.08,
                "top5_capture": 0.20,
                "top1_capture": 0.10,
                "top5_adverse_median": 0.02,
            }
            changed = {
                **base,
                "rank_ic": base["rank_ic"] + improvement,
                "top5_mfe_mean": base["top5_mfe_mean"] + improvement,
                "top5_capture": base["top5_capture"] + improvement,
                "top1_capture": base["top1_capture"] + improvement,
            }
        elif kind == "risk":
            base = {"rank_ic": 0.05, "mae": 0.02, "deep_adverse_pr_auc": 0.30}
            changed = {
                "rank_ic": base["rank_ic"] + improvement,
                "mae": base["mae"] - improvement / 10,
                "deep_adverse_pr_auc": base["deep_adverse_pr_auc"] + improvement,
            }
        else:
            base = {
                "ordinal_ic": 0.05,
                "high_state_top5_lift": 0.10,
                "brier": 0.50,
                "logloss": 0.80,
            }
            changed = {
                "ordinal_ic": base["ordinal_ic"] + improvement,
                "high_state_top5_lift": base["high_state_top5_lift"] + improvement,
                "brier": base["brier"] - improvement / 10,
                "logloss": base["logloss"] - improvement / 10,
            }
        reference[year] = {"metrics": base}
        candidate[year] = {"metrics": changed}
    return candidate, reference


def test_real_feature_contract_has_exact_canonical_counts() -> None:
    ready_manifest = model._source_manifest()
    catalog, registry = model._catalog_and_registry(
        ready_root=model.DEFAULT_READY_ROOT,
        audit_root=model.DEFAULT_AUDIT_ROOT,
        ready_manifest=ready_manifest,
    )
    contract = model._feature_contract(
        catalog=catalog,
        registry=registry,
        base_manifest=model._base_manifest(),
    )

    assert len(contract["base_names"]) == 295
    assert len(contract["extra_names"]) == 333
    assert len(contract["metadata"]) == 13
    assert len(contract["groups"]["legacy_core"]) == 493
    assert len(contract["groups"]["full_core"]) == 587
    assert len(contract["groups"]["all_gated"]) == 41
    assert "listing_age_days" not in catalog["feature_name"].tolist()
    assert catalog["feature_name"].eq("listing_age_open_days").sum() == 1


def test_task_plan_is_exactly_78_without_parameter_search() -> None:
    tasks = model._task_plan(model._load_config())

    assert len(tasks) == 78
    assert sum(task["stage"] == "tuning" for task in tasks) == 9
    assert sum(task["stage"] == "mfe_core" for task in tasks) == 24
    assert sum(task["stage"] == "mfe_gated" for task in tasks) == 18
    assert sum(task["stage"] == "risk_state_core" for task in tasks) == 18
    assert sum(task["stage"] == "risk_state_gated" for task in tasks) == 9


def test_fold_uses_horizon_purge_and_target_flags() -> None:
    inputs = _fake_inputs()

    fold = inputs.fold(year=2023, horizon=10, target="mfe_10")

    assert fold["maximum_train_signal_date_idx"] == 24
    assert fold["train_rows"].tolist() == [0, 1, 2, 4, 5]
    assert fold["evaluation_rows"].tolist() == [6, 7, 8, 9]
    assert inputs.valid_mask("state_10")[3] == 0
    assert inputs.valid_mask("mfe_20").all()


def test_target_and_valid_arrays_are_cached() -> None:
    inputs = _fake_inputs()

    first_values = inputs.task_values("mfe_10")
    first_valid = inputs.valid_mask("mfe_10")

    assert inputs.task_values("mfe_10") is first_values
    assert inputs.valid_mask("mfe_10") is first_valid


def test_sequence_supports_arbitrary_rows_and_variant_layout() -> None:
    inputs = object.__new__(model.ModelInputs)
    inputs.candidate_ids = np.arange(4, dtype=np.int64)
    inputs.base = np.arange(12, dtype=np.float32).reshape(4, 3)
    inputs.extra = np.asarray([[10, 20, 30, 40], [11, 21, 31, 41]], dtype=np.float32)
    inputs.availability = np.asarray([[-1, 0, 1, 1]], dtype=np.int8)
    inputs.feature_records = [
        {"feature_name": "base_1", "storage": "base", "column_index": 1},
        {"feature_name": "extra_0", "storage": "extra", "column_index": 0},
        {
            "feature_name": "available",
            "storage": "availability",
            "column_index": 0,
        },
    ]
    inputs.feature_map = {item["feature_name"]: item for item in inputs.feature_records}
    layout = inputs.layout(["extra_0", "base_1", "available"])
    sequence = model._make_sequence(
        inputs=inputs,
        rows=np.asarray([3, 1], dtype=np.int64),
        layout=layout,
        batch_size=2,
    )

    assert np.array_equal(
        sequence[:],
        np.asarray([[40.0, 10.0, 1.0], [20.0, 4.0, 0.0]], dtype=np.float64),
    )
    assert layout["categorical_positions"] == [2]


def test_availability_encoding_preserves_unknown_as_minus_one() -> None:
    values = pd.Series(["eligible_observed", "known_ineligible", None, "unknown"])
    kind, mapping = model._metadata_mapping(values)
    encoded = model._encode_metadata(values, kind=kind, mapping=mapping)

    assert encoded[2:].tolist() == [-1, -1]
    assert encoded[0] != encoded[1]
    assert set(encoded[:2]) >= {0, 1}


def test_selection_gates_accept_consistent_improvement() -> None:
    config = model._load_config()
    mfe_candidate, mfe_reference = _metric_results("mfe", improvement=0.01)
    risk_candidate, risk_reference = _metric_results("risk", improvement=0.01)
    state_candidate, state_reference = _metric_results("state", improvement=0.01)

    assert model._mfe_gate(
        candidate=mfe_candidate, reference=mfe_reference, config=config
    )["passed"]
    assert model._risk_gate(
        candidate=risk_candidate, reference=risk_reference, config=config
    )["passed"]
    assert model._state_gate(
        candidate=state_candidate, reference=state_reference, config=config
    )["passed"]


def test_task_completion_is_resumable_and_hash_bound(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"stable")
    result_path = tmp_path / "task_result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "task_fingerprint": "fingerprint",
                "files": {"artifact": model._file_record(artifact)},
            }
        ),
        encoding="utf-8",
    )

    assert model._task_complete(result_path, fingerprint="fingerprint")
    artifact.write_bytes(b"changed")
    assert not model._task_complete(result_path, fingerprint="fingerprint")


def test_small_lightgbm_fixture_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(200, 4)).astype(np.float32)
    target = (features[:, 0] - features[:, 1] * 0.5).astype(np.float32)
    config = model._load_config()
    parameters = model._target_parameters(config, "mfe")
    parameters.update(
        {"num_threads": 1, "min_data_in_leaf": 2, "histogram_pool_size": 64}
    )

    first = lgb.train(
        parameters, lgb.Dataset(features, label=target), num_boost_round=5
    )
    second = lgb.train(
        parameters, lgb.Dataset(features, label=target), num_boost_round=5
    )

    assert np.array_equal(first.predict(features), second.predict(features))


def test_training_task_runs_through_sequence_and_resumes(tmp_path) -> None:
    rng = np.random.default_rng(7)
    rows_per_date = 25
    train_dates = np.arange(60, dtype=np.int32)
    evaluation_dates = np.arange(80, 92, dtype=np.int32)
    date_idx = np.repeat(np.concatenate([train_dates, evaluation_dates]), rows_per_date)
    row_count = len(date_idx)
    features = rng.normal(size=(row_count, 2)).astype(np.float32)

    inputs = object.__new__(model.ModelInputs)
    inputs.manifest = {"input_fingerprint": "fixture-input"}
    inputs.row_count = row_count
    inputs.candidate_ids = np.arange(row_count, dtype=np.int64)
    inputs.date_idx = date_idx
    inputs.years = np.where(date_idx < 80, 2022, 2023).astype(np.int16)
    inputs.date_values = np.asarray([f"date-{value:03d}" for value in range(100)])
    inputs.base = features
    inputs.extra = np.empty((0, row_count), dtype=np.float32)
    inputs.availability = np.full((2, row_count), -1, dtype=np.int8)
    inputs.labels = np.full((row_count, 15), np.nan, dtype=np.float32)
    inputs.labels[:, 4] = 0.08 + 0.02 * features[:, 0] - 0.01 * features[:, 1]
    inputs.labels[:, 5] = -0.02 - 0.005 * np.abs(features[:, 1])
    inputs.flags = np.zeros((row_count, 5), dtype=np.uint16)
    inputs.flags[:, 1] = model.FLAG_MFE_PRE_PEAK_MAE_VALID
    inputs.states = np.full((row_count, 5), -1, dtype=np.int8)
    inputs.label_manifest = {
        "label_columns": [
            "g_5",
            "mfe_5",
            "pre_peak_mae_5",
            "g_10",
            "mfe_10",
            "pre_peak_mae_10",
            "g_20",
            "mfe_20",
            "pre_peak_mae_20",
            "g_40",
            "mfe_40",
            "pre_peak_mae_40",
            "g_60",
            "mfe_60",
            "pre_peak_mae_60",
        ]
    }
    inputs.feature_records = [
        {
            "feature_name": "feature_0",
            "storage": "base",
            "column_index": 0,
            "block": "existing_seq100_base",
            "analytic_family": "fixture",
        },
        {
            "feature_name": "feature_1",
            "storage": "base",
            "column_index": 1,
            "block": "existing_seq100_base",
            "analytic_family": "fixture",
        },
        {
            "feature_name": "margin_eligibility_eligible",
            "storage": "availability",
            "column_index": 0,
            "block": "margin_features",
            "analytic_family": "availability_metadata",
        },
        {
            "feature_name": "balance_extension_source_conflict",
            "storage": "availability",
            "column_index": 1,
            "block": "financial_statement_extensions",
            "analytic_family": "availability_metadata",
        },
    ]
    inputs.feature_map = {item["feature_name"]: item for item in inputs.feature_records}
    inputs.feature_groups = {
        "legacy_core": ["feature_0", "feature_1"],
        "financial_extensions": [],
    }

    config = json.loads(json.dumps(model._load_config()))
    config["model"].update(
        {
            "fixed_mfe_rounds": 3,
            "min_data_in_leaf": 2,
            "num_threads": 1,
            "histogram_pool_size_mb": 64,
            "sequence_batch_size": 128,
        }
    )
    config["model"]["working_set_trim"]["enabled"] = False
    task = {
        "task_id": "mfe_core__mfe_10__legacy_core__2023",
        "stage": "mfe_core",
        "target": "mfe_10",
        "kind": "mfe",
        "horizon": 10,
        "year": 2023,
        "variant": "legacy_core",
        "gated_family": None,
    }

    first = model._run_training_task(
        task=task,
        inputs=inputs,
        config=config,
        output_root=tmp_path,
        config_sha256="fixture-config",
    )
    second = model._run_training_task(
        task=task,
        inputs=inputs,
        config=config,
        output_root=tmp_path,
        config_sha256="fixture-config",
    )

    assert first == second
    assert first["status"] == "completed"
    assert first["best_iteration"] == 3
    assert first["candidate_prediction_row_count"] == rows_per_date * len(
        evaluation_dates
    )
    assert all(record["sha256"] for record in first["files"].values())
    json.dumps(first, allow_nan=False)


def test_canonical_sources_exclude_2010_and_2026_from_model_rows() -> None:
    manifest = model._source_manifest()

    assert set(map(int, manifest["row_spine"])) == set(model.YEARS)
    assert "2010" not in manifest["row_spine"]
    assert "2026" not in manifest["row_spine"]
    assert int(manifest["forbidden_2026_rows"]) == 0


def test_forbidden_year_prefix_check_accepts_object_backed_dates() -> None:
    dates = np.asarray(np.asarray(["2025-12-31"], dtype=object), dtype=str)

    assert not np.char.startswith(dates, "2026-").any()
