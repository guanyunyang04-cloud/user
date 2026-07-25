from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_path_relevance as relevance


def _flat_path(rows: int = 1, *, close_return: float = 0.0) -> np.ndarray:
    path = np.zeros((rows, 20, 4), dtype=np.float32)
    path[:, :, 0] = close_return
    path[:, :, 1] = close_return
    path[:, :, 2] = close_return
    path[:, :, 3] = close_return
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


def test_contract_hash_and_source_bindings_are_frozen() -> None:
    study = relevance.load_study()
    assert study["contract_sha256"] == relevance._canonical_json_sha256(
        study["contract"]
    )
    pack_path, manifest = relevance._validate_source_bindings(study)
    assert pack_path.exists()
    assert int(manifest["lookback_days"]) == 180
    assert int(manifest["forward_days"]) >= 60


def test_contract_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = relevance.load_study()
    payload["contract"]["objective"] = "tampered"
    path = tmp_path / "study.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract SHA-256 mismatch"):
        relevance.load_study(path)


def test_next_open_reanchor_uses_executed_open_denominator() -> None:
    path = np.zeros((1, 20, 4), dtype=np.float32)
    path[:, :, :] = 0.10
    path[:, 0, 0] = 0.05
    converted = relevance.reanchor_to_actual_next_open(
        path,
        source_price_anchor="today_close",
    )
    assert converted[0, 0, 0] == pytest.approx(0.0, abs=2.0e-6)
    assert converted[0, 0, 3] == pytest.approx(1.10 / 1.05 - 1.0)


def test_path_descriptors_apply_roundtrip_cost_and_t_plus_one_sellability() -> None:
    path = _flat_path(close_return=0.10)
    multiplier = np.full((1, 20), 0.98, dtype=np.float32)
    sellable = np.zeros((1, 20), dtype=bool)
    sellable[:, 0] = True  # Entry day cannot satisfy the T+1 exit requirement.
    descriptor = relevance.compute_path_descriptors(
        path,
        multiplier,
        exit_sellable=sellable,
        terminal_failure=np.array([False]),
    )
    expected_log = np.log(1.10 * 0.98)
    assert descriptor["close_net_log"][0, 19] == pytest.approx(expected_log)
    assert descriptor["r20_net"][0] == pytest.approx(np.expm1(expected_log))
    assert bool(descriptor["no_legal_sell"][0])


def test_terminal_zero_recovery_forces_all_later_ohlc_to_zero_wealth() -> None:
    path = _flat_path(close_return=0.20)
    delisted = np.zeros((1, 20), dtype=bool)
    delisted[0, 7:] = True
    recovered, failure = relevance.apply_terminal_zero_recovery(path, delisted)
    assert bool(failure[0])
    np.testing.assert_array_equal(recovered[0, 7:, :4], -1.0)
    descriptor = relevance.compute_path_descriptors(
        recovered,
        np.ones((1, 20), dtype=np.float32),
        exit_sellable=np.ones((1, 20), dtype=bool),
        terminal_failure=failure,
    )
    daily = relevance.compute_daily_relevance(
        descriptor,
        entry_filled=np.array([True]),
        path_available=descriptor["path_available"],
        forced_low=np.array([True]),
        pareto_device="cpu",
    )
    assert daily["conditional_relevance"][0] == pytest.approx(0.0)
    assert daily["relevance_grade"][0] == 0


def test_exact_pareto_counts_and_daily_tie_break() -> None:
    values = np.array(
        [
            [3.0, 3.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=np.float32,
    )
    dominated, dominating = relevance.pareto_dominance_counts(
        values,
        device="cpu",
        block_size=2,
    )
    np.testing.assert_array_equal(dominated, [3, 0, 0, 0])
    np.testing.assert_array_equal(dominating, [0, 1, 1, 1])

    tied = np.tile(np.arange(1.0, 8.0, dtype=np.float32), (3, 1))
    result = relevance.compute_daily_relevance(
        _descriptor_fixture(tied),
        entry_filled=np.ones(3, dtype=bool),
        path_available=np.ones(3, dtype=bool),
        forced_low=np.zeros(3, dtype=bool),
        symbol_idx=np.array([30, 10, 20]),
        pareto_device="cpu",
    )
    assert result["conditional_relevance"][1] == pytest.approx(0.0)
    assert result["conditional_relevance"][2] == pytest.approx(0.5)
    assert result["conditional_relevance"][0] == pytest.approx(1.0)


def test_grade_boundaries_and_unfilled_action_relevance() -> None:
    grades = relevance.relevance_grades(
        np.array([0.0, 0.4999, 0.5, 0.8, 0.95, 0.99, np.nan])
    )
    np.testing.assert_array_equal(grades, [0, 0, 1, 2, 3, 4, 255])

    values = np.tile(np.arange(1.0, 8.0, dtype=np.float32), (2, 1))
    result = relevance.compute_daily_relevance(
        _descriptor_fixture(values),
        entry_filled=np.array([True, False]),
        path_available=np.ones(2, dtype=bool),
        forced_low=np.zeros(2, dtype=bool),
        pareto_device="cpu",
    )
    assert np.isfinite(result["conditional_relevance"][0])
    assert np.isnan(result["conditional_relevance"][1])
    assert result["action_relevance"][1] == pytest.approx(0.0)


def test_pre_registered_trend_fallback_reuses_unified_target_fields(tmp_path: Path) -> None:
    count = 3
    paths = relevance._target_paths(tmp_path)
    target = np.memmap(
        paths.float_values,
        dtype="float32",
        mode="w+",
        shape=(count, len(relevance.TARGET_FLOAT_FIELDS)),
    )
    target[:] = np.nan
    target[:, relevance.TARGET_FIELD_INDEX["trend_consistency_v1"]] = [0.2, 0.8, np.nan]
    target[:, relevance.TARGET_FIELD_INDEX["conditional_relevance"]] = [0.1, 0.1, 0.1]
    target[:, relevance.TARGET_FIELD_INDEX["action_relevance"]] = [0.1, 0.1, 0.1]
    target.flush()
    del target
    grades = np.memmap(paths.relevance_grade, dtype="uint8", mode="w+", shape=(count,))
    grades[:] = 0
    grades.flush()
    del grades
    flags = np.memmap(paths.flags, dtype="uint8", mode="w+", shape=(count,))
    flags[:] = [relevance.TARGET_FLAG_ENTRY_FILLED, 0, relevance.TARGET_FLAG_ENTRY_FILLED]
    flags.flush()
    del flags

    relevance._rewrite_frozen_relevance_fields(
        paths,
        candidate_count=count,
        frozen_profile=relevance.TREND_FALLBACK_PROFILE,
        chunk_size=2,
    )
    target = np.memmap(
        paths.float_values,
        dtype="float32",
        mode="r",
        shape=(count, len(relevance.TARGET_FLOAT_FIELDS)),
    )
    np.testing.assert_allclose(
        target[:, relevance.TARGET_FIELD_INDEX["conditional_relevance"]],
        [0.2, 0.8, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        target[:, relevance.TARGET_FIELD_INDEX["action_relevance"]],
        [0.2, 0.0, np.nan],
        equal_nan=True,
    )
    grades = np.memmap(paths.relevance_grade, dtype="uint8", mode="r", shape=(count,))
    np.testing.assert_array_equal(grades, [0, 2, 255])


def test_d20_endpoint_cutoff_rejects_2026_and_out_of_range() -> None:
    dates = [
        np.datetime_as_string(value, unit="D")
        for value in np.arange(
            np.datetime64("2025-11-20"),
            np.datetime64("2026-01-20"),
            dtype="datetime64[D]",
        )
    ]
    assert relevance.label_endpoint_is_allowed(
        signal_date_idx=0,
        date_values=dates,
        horizon_days=20,
        cutoff="2025-12-31",
    )
    assert not relevance.label_endpoint_is_allowed(
        signal_date_idx=30,
        date_values=dates,
        horizon_days=20,
        cutoff="2025-12-31",
    )
    assert not relevance.label_endpoint_is_allowed(
        signal_date_idx=len(dates) - 5,
        date_values=dates,
        horizon_days=20,
        cutoff="2025-12-31",
    )
    with pytest.raises(ValueError, match="2026"):
        relevance.label_endpoint_is_allowed(
            signal_date_idx=0,
            date_values=dates,
            horizon_days=20,
            cutoff="2026-01-01",
        )


def test_rolling_features_are_causal_under_future_mutation() -> None:
    rng = np.random.default_rng(7)
    values = np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(100, 3)), axis=0)).astype(
        np.float32
    )
    changed = values.copy()
    changed[71:] *= 50.0
    before = {
        "return": relevance._lagged_return(values, 20),
        "mean": relevance._rolling_mean(values, 20),
        "corr": relevance._rolling_corr(values, np.log1p(values), 20),
        "trend": relevance._rolling_linear_stats(np.log(values), 20)[0],
        "high_age": relevance._rolling_extreme_age(values, 20, mode="max"),
    }
    after = {
        "return": relevance._lagged_return(changed, 20),
        "mean": relevance._rolling_mean(changed, 20),
        "corr": relevance._rolling_corr(changed, np.log1p(changed), 20),
        "trend": relevance._rolling_linear_stats(np.log(changed), 20)[0],
        "high_age": relevance._rolling_extreme_age(changed, 20, mode="max"),
    }
    for name in before:
        np.testing.assert_allclose(
            before[name][:71],
            after[name][:71],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
            err_msg=name,
        )


def test_cross_section_rank_is_daily_and_does_not_use_future_days() -> None:
    dates = np.array([1, 1, 1, 2, 2], dtype=np.int32)
    values = np.array([3.0, 1.0, 2.0, 100.0, -100.0], dtype=np.float32)
    ranked = relevance._candidate_cross_section_percentile(values, dates)
    np.testing.assert_allclose(ranked[:3], [1.0, 0.0, 0.5])
    np.testing.assert_allclose(ranked[3:], [1.0, 0.0])


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
        relevance._load_qdp_dense_domain(
            study,
            manifest,
            domain="fake",
            numeric_fields=("value",),
            source_guards={"value": "source_date"},
        )
