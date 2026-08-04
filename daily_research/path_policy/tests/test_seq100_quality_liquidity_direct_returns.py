from __future__ import annotations

import json

import numpy as np
import pytest

from daily_research.path_policy import seq100_quality_liquidity_model as model


def test_direct_return_plan_is_exactly_five_horizons_by_three_years() -> None:
    tasks = model._direct_return_task_plan()

    assert len(tasks) == model.RETURN_TASK_COUNT == 15
    assert {task["horizon"] for task in tasks} == {1, 3, 5, 10, 20}
    assert {task["year"] for task in tasks} == {2023, 2024, 2025}
    assert {task["kind"] for task in tasks} == {"return"}
    assert {task["variant"] for task in tasks} == {model.COMPACT_VARIANT}
    assert all(task["stage"] == model.RETURN_TASK_STAGE for task in tasks)


def test_winsorized_zscore_is_date_local_and_preserves_invalid_rows() -> None:
    first = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    first[-1] = 1000.0
    second = np.linspace(10.0, 20.0, 100, dtype=np.float32)
    values = np.concatenate([first, second, np.arange(5, dtype=np.float32)])
    dates = np.repeat(np.asarray([1, 2, 3], dtype=np.int32), [100, 100, 5])
    valid = np.ones(len(values), dtype=bool)
    valid[3] = False

    normalized, normalized_valid = model._winsorized_zscore_by_date(
        values=values, date_idx=dates, valid=valid
    )

    assert not normalized_valid[3]
    assert np.isnan(normalized[3])
    assert not normalized_valid[-5:].any()
    assert np.isnan(normalized[-5:]).all()
    for day in (1, 2):
        local = normalized[(dates == day) & normalized_valid]
        assert float(local.mean()) == pytest.approx(0.0, abs=2e-7)
        assert float(local.std(ddof=0)) == pytest.approx(1.0, abs=2e-7)
    assert float(normalized[99]) < 3.0


def test_return_accessor_unifies_short_and_main_label_stores() -> None:
    inputs = object.__new__(model.ModelInputs)
    inputs.candidate_ids = np.asarray([1, 3], dtype=np.int64)
    inputs.labels = np.full((5, 15), np.nan, dtype=np.float32)
    inputs.labels[:, 0] = np.arange(5, dtype=np.float32) + 0.05
    inputs.labels[:, 3] = np.arange(5, dtype=np.float32) + 0.10
    inputs.labels[:, 6] = np.arange(5, dtype=np.float32) + 0.20
    inputs.flags = np.zeros((5, 5), dtype=np.uint16)
    inputs.flags[:, :3] = model.FLAG_G_VALID
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
    inputs.short_labels = np.full((5, 4), np.nan, dtype=np.float32)
    inputs.short_labels[:, 0] = np.arange(5, dtype=np.float32) + 0.01
    inputs.short_labels[:, 1] = np.arange(5, dtype=np.float32) + 0.03
    inputs.short_flags = np.zeros((5, 2), dtype=np.uint16)
    inputs.short_flags[:, :] = model.SHORT_FLAG_G_VALID
    inputs.short_label_manifest = {
        "label_columns": ["g_1", "g_3", "mfe_3", "pre_peak_mae_3"]
    }

    assert inputs.raw_return_values(1).tolist() == pytest.approx([1.01, 3.01])
    assert inputs.raw_return_values(3).tolist() == pytest.approx([1.03, 3.03])
    assert inputs.raw_return_values(5).tolist() == pytest.approx([1.05, 3.05])
    assert inputs.raw_return_values(10).tolist() == pytest.approx([1.10, 3.10])
    assert inputs.raw_return_values(20).tolist() == pytest.approx([1.20, 3.20])
    assert inputs.raw_return_valid_mask(1).all()
    assert inputs.raw_return_valid_mask(20).all()


def test_return_metrics_use_raw_returns_for_rank_and_tail_capture() -> None:
    raw = np.linspace(-0.10, 0.20, 100, dtype=np.float32)
    zscore = (raw - raw.mean()) / raw.std(ddof=0)

    daily, summary = model._return_daily_metrics(
        date_idx=np.zeros(100, dtype=np.int32),
        actual_raw=raw,
        actual_zscore=zscore,
        prediction=zscore,
    )

    assert len(daily) == 1
    assert summary["rank_ic"] == pytest.approx(1.0)
    assert summary["top1_capture"] == pytest.approx(1.0)
    assert summary["top5_capture"] == pytest.approx(1.0)
    assert summary["top5_return_mean"] == pytest.approx(float(raw[-5:].mean()))
    assert summary["decile_spearman"] == pytest.approx(1.0)
    assert summary["zscore_mse"] == pytest.approx(0.0)


def test_return_parameters_use_mse() -> None:
    parameters = model._target_parameters(model._load_config(), "return")

    assert parameters["objective"] == "regression"
    assert parameters["metric"] == "l2"
    assert "alpha" not in parameters


def test_small_direct_return_task_is_deterministic_and_resumable(tmp_path) -> None:
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
    inputs.labels[:, 0] = 0.02 * features[:, 0] - 0.01 * features[:, 1]
    inputs.flags = np.zeros((row_count, 5), dtype=np.uint16)
    inputs.flags[:, 0] = model.FLAG_G_VALID
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
            "block": "fixture",
            "analytic_family": "fixture",
        },
        {
            "feature_name": "feature_1",
            "storage": "compact",
            "column_index": 1,
            "block": "fixture",
            "analytic_family": "fixture",
        },
    ]
    inputs.feature_map = {item["feature_name"]: item for item in inputs.feature_records}
    inputs.feature_groups = {model.COMPACT_VARIANT: ["feature_0", "feature_1"]}

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
        "task_id": f"return_zscore__g_05__{model.COMPACT_VARIANT}__2023",
        "stage": model.RETURN_TASK_STAGE,
        "target": "return_5",
        "source_label": "g_5",
        "kind": "return",
        "horizon": 5,
        "year": 2023,
        "variant": model.COMPACT_VARIANT,
        "gated_family": None,
        "label_transform": "signal_date_winsor_01_99_zscore",
    }

    first = model._run_training_task(
        task=task,
        inputs=inputs,
        config=config,
        output_root=tmp_path,
        config_sha256="fixture-config",
        experiment_fingerprint="fixture-return",
    )
    second = model._run_training_task(
        task=task,
        inputs=inputs,
        config=config,
        output_root=tmp_path,
        config_sha256="fixture-config",
        experiment_fingerprint="fixture-return",
    )

    assert first == second
    assert first["status"] == "completed"
    assert first["best_iteration"] == 3
    assert first["source_label"] == "g_5"
    assert first["label_transform"] == "signal_date_winsor_01_99_zscore"
    assert first["metrics"]["rank_ic"] > 0.8
    assert all(record["sha256"] for record in first["files"].values())
