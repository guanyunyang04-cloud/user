from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from daily_research.path_policy import (
    seq100_full_market_multitask_forecast as study,
)
from daily_research.path_policy.seq100_candidate_execution import ExecutionCosts


def _panels(days: int = 36, symbols: int = 2) -> dict[str, np.ndarray]:
    daily = np.full((days, symbols, 13), np.nan, dtype=np.float32)
    for day in range(days):
        for symbol in range(symbols):
            value = 9.0 + day + symbol
            daily[day, symbol, 0] = value
            daily[day, symbol, 1] = value + 1.0
            daily[day, symbol, 2] = value - 1.0
            daily[day, symbol, 3] = value + 0.5
    shape = (days, symbols)
    return {
        "daily_raw": daily,
        "entry_filled": np.ones(shape, dtype=bool),
        "exit_sellable": np.ones(shape, dtype=bool),
        "next_open_sellable": np.tile(
            np.arange(days, dtype=np.int32)[:, None], (1, symbols)
        ),
    }


def test_resource_plan_reserves_exactly_one_gib_and_uses_all_threads() -> None:
    plan = study.resource_plan(
        available_bytes=10 * (1 << 30),
        total_bytes=16 * (1 << 30),
        logical_cpus=16,
    )

    assert plan.reserve_bytes == 1 << 30
    assert plan.usable_bytes == 9 * (1 << 30)
    assert plan.cpu_threads == 16
    assert plan.histogram_pool_mb == 1536
    assert plan.sequence_batch_size == 65536


def test_strict_json_serializes_unavailable_metrics_as_null(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"

    study._write_json(path, {"metric": float("nan"), "nested": [float("inf")]})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "metric": None,
        "nested": [None],
    }


def test_full_market_study_does_not_gate_on_financing_or_chan() -> None:
    config = study.load_study()

    assert (
        config["population"]["candidate_gate"] == "none_beyond_the_frozen_pit_universe"
    )
    assert (
        config["population"]["financing_role"]
        == "optional_input_family_not_population_gate"
    )
    assert (
        config["population"]["chan_role"] == "optional_input_family_not_population_gate"
    )
    assert config["resources"]["reserve_system_gib"] == 1.0


def test_feature_ablation_variants_resolve_the_frozen_families() -> None:
    config = study.load_study()
    manifest = study._read_json(
        study._resolve(config["sources"]["model_input_manifest"])
    )

    all_names, _, _ = study._feature_variant_contract(
        config, manifest, "all_causal_557"
    )
    context_names, _, _ = study._feature_variant_contract(
        config, manifest, "price_path_context_364"
    )
    core_names, _, _ = study._feature_variant_contract(
        config, manifest, "price_path_core_183"
    )

    assert len(all_names) == 557
    assert len(context_names) == 364
    assert len(core_names) == 183
    assert set(core_names) < set(context_names) < set(all_names)


def test_task_ids_keep_feature_variants_and_profiles_isolated() -> None:
    default = study._task_id(
        fold=1,
        target="next_close_up",
        profile="strong_127",
        training_mode="causal_nested",
    )
    ablation = study._task_id(
        fold=1,
        target="next_close_up",
        profile="strong_127",
        feature_variant="price_path_core_183",
        training_mode="causal_nested",
    )

    assert default == "fold_01__next_close_up__strong_127__causal_nested"
    assert "price_path_core_183" in ablation
    assert default != ablation


def test_profile_ensemble_predictions_are_averaged(tmp_path: Path) -> None:
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    np.save(second, np.asarray([0.5, 0.4, 0.3], dtype=np.float32))
    records = {
        "1:target:first": {"files": {"prediction": {"path": str(first)}}},
        "1:target:second": {"files": {"prediction": {"path": str(second)}}},
    }

    prediction = study._load_ensemble_prediction(
        records,
        fold_number=1,
        target="target",
        model_profiles=("first", "second"),
        row_mask=np.asarray([True, False, True]),
    )

    np.testing.assert_allclose(prediction, np.asarray([0.3, 0.3]))


def test_candidate_replay_audits_concentrated_and_diversified_breadth() -> None:
    specs = study._account_specs(candidate_only=True)
    ordinary = [spec for spec in specs if "maximum_credited_gross_return" not in spec]

    assert {int(spec["top_k"]) for spec in ordinary} == {1, 3, 10, 30}
    assert {str(spec["cost_scenario"]) for spec in ordinary} == {"base", "stress"}


def test_multihorizon_target_uses_next_open_and_legal_d2_close() -> None:
    panels = _panels()
    panels["daily_raw"][0, 0, 3] = 9.0
    panels["daily_raw"][1, 0, 0] = 10.0
    panels["daily_raw"][1, 0, 2] = 9.0
    panels["daily_raw"][1, 0, 3] = 11.0
    panels["daily_raw"][2, 0, 1] = 13.0
    panels["daily_raw"][2, 0, 2] = 10.0
    panels["daily_raw"][2, 0, 3] = 12.0

    values, valid, fill_days = study._derive_rows_for_date(
        signal_idx=0,
        symbols=np.asarray([0], dtype=np.int32),
        cutoff_idx=35,
        **panels,
    )

    d2 = 1
    assert valid[0, 0] == 1
    assert values[0, 0] == pytest.approx(11.0 / 9.0 - 1.0)
    assert values[0, d2] == pytest.approx(0.20)
    assert values[0, d2 + 1] == pytest.approx(0.20)
    assert values[0, d2 + 3] == pytest.approx(0.30)
    assert values[0, d2 + 4] == pytest.approx(-0.10)
    assert values[0, d2 + 5] == pytest.approx(1.0)
    assert fill_days[0, 0] == 2


def test_blocked_d2_close_defers_to_next_sellable_open() -> None:
    panels = _panels()
    panels["daily_raw"][1, 0, 0] = 10.0
    panels["daily_raw"][2, 0, 3] = 12.0
    panels["daily_raw"][3, 0, 0] = 8.0
    panels["exit_sellable"][2, 0] = False
    panels["next_open_sellable"][3, 0] = 3

    values, valid, fill_days = study._derive_rows_for_date(
        signal_idx=0,
        symbols=np.asarray([0], dtype=np.int32),
        cutoff_idx=35,
        **panels,
    )

    assert valid[0, 2] == 1
    assert values[0, 2] == pytest.approx(-0.20)
    assert fill_days[0, 0] == 3


def test_cutoff_never_reads_or_labels_beyond_formal_history() -> None:
    panels = _panels()
    panels["daily_raw"][4:, :, :] = 999.0

    values, valid, fill_days = study._derive_rows_for_date(
        signal_idx=3,
        symbols=np.asarray([0], dtype=np.int32),
        cutoff_idx=3,
        **panels,
    )

    assert not valid.any()
    assert np.isnan(values).all()
    assert (fill_days == -1).all()


def test_forward_folds_use_all_prior_history_and_thirty_day_purge() -> None:
    unique_dates = np.arange(1000, 2500, dtype=np.int32)
    labels = np.repeat(
        np.asarray(
            [
                f"{2017 + index // 300:04d}-{index % 12 + 1:02d}-{index % 27 + 1:02d}"
                for index in range(len(unique_dates))
            ]
        ),
        2,
    )
    order = np.argsort(labels.reshape(-1, 2)[:, 0], kind="stable")
    unique_dates = unique_dates[order]
    unique_labels = labels.reshape(-1, 2)[order, 0]
    remapped = np.repeat(np.arange(len(unique_dates), dtype=np.int32), 2)
    repeated_labels = np.repeat(unique_labels, 2)

    folds = study.build_forward_folds(
        date_idx=remapped,
        trade_date=repeated_labels,
        validation_start_date=unique_labels[600],
        validation_end_date=unique_labels[-1],
        fold_count=5,
        purge_days=30,
    )

    assert len(folds) == 5
    for fold in folds:
        assert (
            fold["training_maximum_date_idx"] + 30 < fold["validation_start_date_idx"]
        )
        assert fold["training_row_count"] > 0


def test_nested_inner_split_never_uses_outer_fold_outcomes() -> None:
    dates = np.repeat(np.arange(1000, 1800, dtype=np.int32), 3)

    training, validation, metadata = study._nested_inner_rows(dates)

    assert dates[training].max() == 1478
    assert dates[validation].min() == 1509
    assert dates[validation].max() == 1799
    assert metadata["validation_date_count"] == 291
    assert metadata["purge_date_count"] == 30
    assert set(np.unique(dates[training])).isdisjoint(set(np.unique(dates[validation])))


def test_nested_lightgbm_subsets_preserve_complete_ranking_groups() -> None:
    rng = np.random.default_rng(17)
    dates = np.repeat(np.arange(400, dtype=np.int32), 5)
    matrix = rng.normal(size=(len(dates), 4)).astype(np.float32)
    raw = matrix[:, 0] + 0.1 * rng.normal(size=len(dates))
    valid = np.ones(len(dates), dtype=bool)
    labels = study._date_relevance_labels(dates, raw, valid)
    training, validation, _ = study._nested_inner_rows(
        dates,
        validation_date_count=40,
        purge_days=10,
    )
    parent = lgb.Dataset(matrix, free_raw_data=False).construct()
    parent.set_label(labels)
    inner_training = parent.subset(training).construct()
    inner_validation = parent.subset(validation).construct()
    inner_training.set_label(labels[training])
    inner_validation.set_label(labels[validation])
    inner_training.set_group(study._date_group_sizes(dates[training]))
    inner_validation.set_group(study._date_group_sizes(dates[validation]))

    booster = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": list(range(10)),
            "verbosity": -1,
            "num_threads": 2,
        },
        inner_training,
        num_boost_round=2,
        valid_sets=[inner_validation],
    )

    assert booster.current_iteration() >= 1


def test_ranking_labels_are_cross_sectional_and_groups_stay_by_date() -> None:
    dates = np.repeat(np.asarray([1, 2], dtype=np.int32), 10)
    values = np.tile(np.arange(10, dtype=np.float32), 2)
    valid = np.ones(20, dtype=bool)

    labels = study._date_relevance_labels(dates, values, valid)

    np.testing.assert_array_equal(labels[:10], np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(labels[10:], np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(
        study._date_group_sizes(dates), np.asarray([10, 10], dtype=np.int32)
    )


def test_exact_unit_return_applies_lots_fees_tax_and_double_slippage() -> None:
    daily = np.full((4, 1, 13), np.nan, dtype=np.float32)
    daily[1, 0, 0] = 10.0
    raw_open = np.full((4, 1), 10.0, dtype=np.float32)
    dates = np.asarray(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    costs = ExecutionCosts(
        lot_size=100,
        commission_bps=3.0,
        minimum_commission_cny=5.0,
        transfer_fee_bps=0.1,
        slippage_bps=7.0,
        stress_slippage_multiplier=2.0,
        stamp_tax_schedule=(("1900-01-01", 10.0), ("2023-08-28", 5.0)),
    )

    base, base_filled = study._exact_unit_net_return(
        signal_idx=0,
        symbol_idx=0,
        legal_gross_return=0.10,
        fill_day=2,
        daily_raw=daily,
        raw_open=raw_open,
        date_values=dates,
        costs=costs,
        notional_cny=100_000.0,
        slippage_multiplier=1.0,
    )
    stress, stress_filled = study._exact_unit_net_return(
        signal_idx=0,
        symbol_idx=0,
        legal_gross_return=0.10,
        fill_day=2,
        daily_raw=daily,
        raw_open=raw_open,
        date_values=dates,
        costs=costs,
        notional_cny=100_000.0,
        slippage_multiplier=2.0,
    )
    unfilled, unfilled_order = study._exact_unit_net_return(
        signal_idx=0,
        symbol_idx=0,
        legal_gross_return=0.10,
        fill_day=2,
        daily_raw=daily,
        raw_open=raw_open,
        date_values=dates,
        costs=costs,
        notional_cny=100_000.0,
        slippage_multiplier=1.0,
        entry_filled=np.zeros((4, 1), dtype=bool),
    )

    assert base_filled and stress_filled
    assert not unfilled_order and unfilled == 0.0
    assert 0.09 < base < 0.10
    assert stress < base


def test_newey_west_interval_is_finite_for_daily_returns() -> None:
    interval = study._newey_west_interval(
        np.asarray([0.01, -0.01, 0.02, 0.00, 0.01]), lag=2
    )

    assert interval["count"] == 5
    assert interval["lower"] < interval["mean"] < interval["upper"]


def test_binary_metrics_apply_the_executable_take_profit_threshold() -> None:
    _, metrics = study._daily_metrics(
        dates=np.asarray([1, 1, 2, 2], dtype=np.int32),
        actual=np.asarray([0.005, 0.015, 0.003, 0.020]),
        prediction=np.asarray([0.1, 0.9, 0.2, 0.8]),
        kind="binary",
        binary_threshold=0.01,
    )

    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["accuracy_at_0p5"] == pytest.approx(1.0)


def test_binned_dataset_cache_is_resumable_and_labels_are_replaceable(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(256, 4)).astype(np.float32)
    rows = np.arange(len(matrix), dtype=np.int64)
    binary = tmp_path / "train.bin"
    meta = tmp_path / "train.json"
    plan = study.resource_plan(
        available_bytes=8 * (1 << 30),
        total_bytes=16 * (1 << 30),
        logical_cpus=4,
        maximum_threads=4,
        histogram_pool_cap_mb=256,
        sequence_batch_cap=128,
    )
    fingerprint = "frozen-test"
    first = study.build_dataset_cache(
        matrix=matrix,
        rows=rows,
        feature_names=("a", "b", "c", "d"),
        binary_path=binary,
        meta_path=meta,
        fingerprint=fingerprint,
        max_bin=31,
        seed=7,
        resource=plan,
    )
    modified = binary.stat().st_mtime_ns
    second = study.build_dataset_cache(
        matrix=matrix,
        rows=rows,
        feature_names=("a", "b", "c", "d"),
        binary_path=binary,
        meta_path=meta,
        fingerprint=fingerprint,
        max_bin=31,
        seed=7,
        resource=plan,
    )
    dataset = lgb.Dataset(
        str(binary), params={"max_bin": 31}, free_raw_data=True
    ).construct()
    labels = (matrix[:, 0] > 0.0).astype(np.float32)
    dataset.set_label(labels)
    validation_binary = tmp_path / "validation.bin"
    validation_meta = tmp_path / "validation.json"
    validation_rows = np.arange(192, 256, dtype=np.int64)
    study.build_dataset_cache(
        matrix=matrix,
        rows=validation_rows,
        feature_names=("a", "b", "c", "d"),
        binary_path=validation_binary,
        meta_path=validation_meta,
        fingerprint="frozen-validation-test",
        max_bin=31,
        seed=7,
        resource=plan,
        reference=dataset,
    )
    validation = lgb.Dataset(
        str(validation_binary),
        reference=dataset,
        params={"max_bin": 31},
        free_raw_data=True,
    ).construct()
    validation.set_label(labels[validation_rows])
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "max_bin": 31,
            "num_threads": 2,
            "verbosity": -1,
        },
        dataset,
        num_boost_round=2,
        valid_sets=[validation],
    )

    assert first == second
    assert binary.stat().st_mtime_ns == modified
    np.testing.assert_array_equal(dataset.get_label(), labels)
    assert booster.current_iteration() == 2
