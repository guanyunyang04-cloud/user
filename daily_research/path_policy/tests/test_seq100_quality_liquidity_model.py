from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

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

    compact = contract["groups"][model.COMPACT_VARIANT]
    assert len(compact) == 562
    assert len(set(compact)) == 562
    assert len(contract["decisions"]) == 28
    assert "balance_other_receivables" not in compact
    assert "balance_other_receivables_total" in compact
    assert "balance_other_payables" not in compact
    assert "balance_other_payables_total" in compact
    assert "balance_advances_from_customers" not in compact
    assert "balance_customer_advances_and_contract_liabilities" in compact
    assert set(model.COMPACT_CONSTANT_DROPS).isdisjoint(compact)
    assert set(model.COMPACT_REDUNDANCY_DROPS).isdisjoint(compact)
    assert "listing_age_days" not in catalog["feature_name"].tolist()
    assert catalog["feature_name"].eq("listing_age_open_days").sum() == 1


def test_refreshed_industry_features_use_current_industry_groups(
    monkeypatch, tmp_path
) -> None:
    diagnostics = tmp_path / "diagnostics.parquet"
    pd.DataFrame(
        {
            "candidate_id": [0, 1, 2, 3],
            "date_idx": [10, 10, 10, 10],
            "industry": ["A", "A", "B", "Unknown"],
        }
    ).to_parquet(diagnostics, index=False)
    monkeypatch.setattr(
        model.descriptive,
        "_diagnostics_path",
        lambda source_manifest, year: diagnostics,
    )
    base = np.asarray(
        [
            [0.10, 0.15, 0.30],
            [-0.10, 0.05, 0.10],
            [0.20, 0.25, 0.40],
            [0.30, 0.35, 0.50],
        ],
        dtype=np.float32,
    )

    frame = model._refreshed_industry_frame(
        ready_manifest={},
        year=2020,
        expected_ids=np.asarray([0, 1, 3], dtype=np.int64),
        base=base,
        base_index={"return_1d": 0, "return_5d": 1, "return_20d": 2},
    )

    assert frame.loc[0, "industry_ret1_mean"] == pytest.approx(0.0)
    assert frame.loc[0, "industry_breadth_ret1_positive"] == pytest.approx(0.5)
    assert frame.loc[0, "industry_ret1_dispersion"] == pytest.approx(0.1)
    assert frame.loc[0, "industry_relative_ret5"] == pytest.approx(0.05)
    assert frame.loc[1, "industry_relative_ret20"] == pytest.approx(-0.1)
    assert frame.loc[2, "industry_missing"] == 1
    assert np.isnan(frame.loc[2, "industry_ret1_mean"])


def test_task_plan_is_exactly_24_compact_core_tasks() -> None:
    tasks = model._task_plan(model._load_config())

    assert len(tasks) == 24
    assert sum(task["stage"] == "tuning" for task in tasks) == 9
    assert sum(task["stage"] == "mfe_core" for task in tasks) == 6
    assert sum(task["stage"] == "risk_state_core" for task in tasks) == 9
    assert {task["variant"] for task in tasks} == {model.COMPACT_VARIANT}


def test_partition_results_preserves_nonformal_tasks_as_diagnostics() -> None:
    tasks = model._task_plan(model._load_config())
    formal_task_id = str(tasks[0]["task_id"])
    diagnostic_task_id = "mfe_gated__mfe_10__selected_core_plus_margin__2023"
    results = {
        formal_task_id: {"task_id": formal_task_id, "status": "completed"},
        diagnostic_task_id: {
            "task_id": diagnostic_task_id,
            "status": "completed",
        },
    }

    formal, diagnostic = model._partition_results(results, tasks)

    assert set(formal) == {formal_task_id}
    assert set(diagnostic) == {diagnostic_task_id}


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
    inputs.compact = np.asarray(
        [[0, 10, -1], [3, 20, 0], [6, 30, 1], [9, 40, 1]],
        dtype=np.float32,
    )
    inputs.feature_records = [
        {"feature_name": "base_1", "storage": "compact", "column_index": 0},
        {"feature_name": "extra_0", "storage": "compact", "column_index": 1},
        {"feature_name": "available", "storage": "compact", "column_index": 2},
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
        np.asarray([[40.0, 9.0, 1.0], [20.0, 3.0, 0.0]], dtype=np.float64),
    )
    assert layout["categorical_positions"] == []


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
    inputs.compact = features
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
            "storage": "compact",
            "column_index": 0,
            "block": "existing_seq100_base",
            "analytic_family": "fixture",
        },
        {
            "feature_name": "feature_1",
            "storage": "compact",
            "column_index": 1,
            "block": "existing_seq100_base",
            "analytic_family": "fixture",
        },
    ]
    inputs.feature_map = {item["feature_name"]: item for item in inputs.feature_records}
    inputs.feature_groups = {
        model.COMPACT_VARIANT: ["feature_0", "feature_1"],
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
        "task_id": f"mfe_core__mfe_10__{model.COMPACT_VARIANT}__2023",
        "stage": "mfe_core",
        "target": "mfe_10",
        "kind": "mfe",
        "horizon": 10,
        "year": 2023,
        "variant": model.COMPACT_VARIANT,
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
