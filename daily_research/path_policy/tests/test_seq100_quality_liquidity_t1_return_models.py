from __future__ import annotations

import json

import numpy as np
import pytest

from daily_research.path_policy import seq100_quality_liquidity_model as model
from daily_research.path_policy import (
    seq100_quality_liquidity_t1_return_models as return_models,
)


def _panels(*, days: int = 6, symbols: int = 2) -> dict[str, np.ndarray]:
    daily = np.full((days, symbols, 4), np.nan, dtype=np.float32)
    for day in range(days):
        for symbol in range(symbols):
            value = 10.0 + day + symbol
            daily[day, symbol, 0] = value
            daily[day, symbol, 1] = value + 1.0
            daily[day, symbol, 2] = value - 1.0
            daily[day, symbol, 3] = value + 0.5
    return {
        "daily_raw": daily,
        "price_observed": np.ones((days, symbols), dtype=bool),
    }


def _derive(
    panels: dict[str, np.ndarray],
    *,
    signal_date_idx: np.ndarray,
    symbol_idx: np.ndarray,
    source_open_to_open: np.ndarray,
    source_valid: np.ndarray,
    cutoff_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return return_models._derive_d2_batch(
        signal_date_idx=signal_date_idx,
        symbol_idx=symbol_idx,
        source_open_to_open=source_open_to_open,
        source_valid=source_valid,
        cutoff_idx=cutoff_idx,
        **panels,
    )


def test_study_and_task_plan_freeze_exactly_six_two_day_heads() -> None:
    study = return_models.load_study()
    tasks = return_models._task_plan(study)

    assert tuple(study["targets"]) == return_models.TARGETS
    assert len(tasks) == return_models.TASK_COUNT == 6
    assert {task["target"] for task in tasks} == set(return_models.TARGETS)
    assert {task["year"] for task in tasks} == {2023, 2024, 2025}
    assert {task["horizon"] for task in tasks} == {2}
    assert {task["variant"] for task in tasks} == {model.COMPACT_VARIANT}
    assert {task["study_id"] for task in tasks} == {return_models.STUDY_ID}


def test_open_to_d2_close_reconciles_with_open_to_open_and_exit_intraday() -> None:
    panels = _panels()
    panels["daily_raw"][1, 0, 0] = 10.0
    panels["daily_raw"][2, 0, 0] = 12.0
    panels["daily_raw"][2, 0, 3] = 15.0

    derived, valid, flags = _derive(
        panels,
        signal_date_idx=np.asarray([0], dtype=np.int32),
        symbol_idx=np.asarray([0], dtype=np.int32),
        source_open_to_open=np.asarray([0.20], dtype=np.float32),
        source_valid=np.asarray([True]),
        cutoff_idx=4,
    )

    assert valid.tolist() == [1]
    assert derived[0, 0] == pytest.approx(0.50)
    assert derived[0, 1] == pytest.approx(0.25)
    assert (1.0 + 0.20) * (1.0 + derived[0, 1]) - 1.0 == pytest.approx(
        derived[0, 0]
    )
    assert flags[0] & return_models.FLAG_OPEN_TO_CLOSE_2_VALID
    assert flags[0] & return_models.FLAG_DECOMPOSITION_VALID


def test_d2_close_label_does_not_require_source_exit_open_or_execution_state() -> None:
    panels = _panels()
    panels["daily_raw"][1, 0, 0] = 10.0
    panels["daily_raw"][2, 0, 0] = np.nan
    panels["daily_raw"][2, 0, 3] = 12.0

    derived, valid, flags = _derive(
        panels,
        signal_date_idx=np.asarray([0], dtype=np.int32),
        symbol_idx=np.asarray([0], dtype=np.int32),
        source_open_to_open=np.asarray([np.nan], dtype=np.float32),
        source_valid=np.asarray([False]),
        cutoff_idx=4,
    )

    assert valid.tolist() == [1]
    assert derived[0, 0] == pytest.approx(0.20)
    assert np.isnan(derived[0, 1])
    assert not flags[0] & return_models.FLAG_SOURCE_OPEN_TO_OPEN_VALID
    assert not flags[0] & return_models.FLAG_EXIT_OPEN_OBSERVED
    assert not flags[0] & return_models.FLAG_DECOMPOSITION_VALID


def test_d2_close_cutoff_blocks_any_t_plus_2_read() -> None:
    panels = _panels()
    panels["daily_raw"][4:, :, :] = 999.0

    derived, valid, flags = _derive(
        panels,
        signal_date_idx=np.asarray([2, 3], dtype=np.int32),
        symbol_idx=np.asarray([0, 0], dtype=np.int32),
        source_open_to_open=np.asarray([0.1, 0.1], dtype=np.float32),
        source_valid=np.asarray([True, True]),
        cutoff_idx=3,
    )

    assert valid.tolist() == [0, 0]
    assert np.isnan(derived).all()
    assert not (flags & return_models.FLAG_OUTCOME_WITHIN_CUTOFF).any()


def test_shared_raw_task_accessor_preserves_existing_return_behavior() -> None:
    inputs = object.__new__(model.ModelInputs)
    inputs.raw_return_values = lambda horizon: np.asarray(
        [float(horizon)], dtype=np.float32
    )

    assert inputs.raw_task_values("return_5").tolist() == pytest.approx([5.0])


def test_custom_target_accessor_normalizes_each_signal_date() -> None:
    inputs = object.__new__(return_models.T1ReturnModelInputs)
    values = np.concatenate(
        [
            np.linspace(-0.1, 0.1, 25, dtype=np.float32),
            np.linspace(0.2, 0.4, 25, dtype=np.float32),
        ]
    )
    inputs.date_idx = np.repeat(np.asarray([1, 2], dtype=np.int32), 25)
    inputs._raw_target_cache = {"open_to_open_1": values}
    inputs._raw_target_valid_cache = {
        "open_to_open_1": np.ones(len(values), dtype=bool)
    }
    inputs._normalized_target_cache = {}
    inputs._normalized_target_valid_cache = {}

    normalized = inputs.task_values("open_to_open_1")
    valid = inputs.valid_mask("open_to_open_1")

    assert valid.all()
    for date in (1, 2):
        local = normalized[inputs.date_idx == date]
        assert float(local.mean()) == pytest.approx(0.0, abs=2e-7)
        assert float(local.std(ddof=0)) == pytest.approx(1.0, abs=2e-7)


def test_small_custom_return_task_uses_two_day_purge_and_custom_study_id(
    tmp_path,
) -> None:
    rng = np.random.default_rng(7)
    rows_per_date = 25
    train_dates = np.arange(60, dtype=np.int32)
    evaluation_dates = np.arange(80, 92, dtype=np.int32)
    date_idx = np.repeat(np.concatenate([train_dates, evaluation_dates]), rows_per_date)
    row_count = len(date_idx)
    features = rng.normal(size=(row_count, 2)).astype(np.float32)
    raw_target = 0.02 * features[:, 0] - 0.01 * features[:, 1]

    inputs = object.__new__(return_models.T1ReturnModelInputs)
    inputs.manifest = {"input_fingerprint": "fixture-input"}
    inputs.row_count = row_count
    inputs.candidate_ids = np.arange(row_count, dtype=np.int64)
    inputs.date_idx = date_idx
    inputs.years = np.where(date_idx < 80, 2022, 2023).astype(np.int16)
    inputs.date_values = np.asarray([f"date-{value:03d}" for value in range(100)])
    inputs.compact = features
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
    inputs.feature_map = {
        item["feature_name"]: item for item in inputs.feature_records
    }
    inputs.feature_groups = {model.COMPACT_VARIANT: ["feature_0", "feature_1"]}
    inputs._raw_target_cache = {"open_to_open_1": raw_target}
    inputs._raw_target_valid_cache = {
        "open_to_open_1": np.ones(row_count, dtype=bool)
    }
    inputs._normalized_target_cache = {}
    inputs._normalized_target_valid_cache = {}

    config = json.loads(json.dumps(return_models.load_study()))
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
        "study_id": return_models.STUDY_ID,
        "task_id": "fixture_t1_return_2023",
        "stage": model.RETURN_TASK_STAGE,
        "target": "open_to_open_1",
        "source_label": "fixture",
        "kind": "return",
        "horizon": 2,
        "year": 2023,
        "variant": model.COMPACT_VARIANT,
        "gated_family": None,
        "label_transform": "signal_date_winsor_01_99_zscore",
    }

    result = model._run_training_task(
        task=task,
        inputs=inputs,
        config=config,
        output_root=tmp_path,
        config_sha256="fixture-config",
        experiment_fingerprint="fixture-t1-return",
    )

    assert result["status"] == "completed"
    assert result["study_id"] == return_models.STUDY_ID
    assert result["best_iteration"] == 3
    assert result["purge_days"] == 2
    assert result["maximum_train_signal_date_idx"] == 77
    assert result["metrics"]["rank_ic"] > 0.8
