from __future__ import annotations

import json

import numpy as np
import pytest

from daily_research.path_policy import seq100_mfe_feature_family_audit as audit


def test_load_study_reads_plain_scientific_config(tmp_path) -> None:
    study = audit.load_study()

    assert study["folds"]["fold_years"] == [2023, 2024, 2025]
    assert study["folds"]["primary_horizons"] == [10, 20]
    assert study["feature_groups"]["negative_control"] == "deterministic_noise"

    changed = json.loads(json.dumps(study))
    changed["folds"]["primary_horizons"] = [10, 60]
    changed_path = tmp_path / "changed_config.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="target horizons changed"):
        audit.load_study(changed_path)


def test_feature_catalog_counts_match_the_scientific_config() -> None:
    expected = {
        "recent_kline_sequence": 100,
        "breakout_retest_levels": 40,
        "long_daily_context": 54,
        "completed_week_month_context": 48,
        "traditional_indicators": 27,
        "confirmed_swing_structure": 24,
        "turnover_cost_proxy": 5,
        audit.CONTROL_FAMILY: 32,
    }

    assert {
        family: len(features)
        for family, features in audit.feature_catalog().items()
    } == expected
    assert audit.load_study()["feature_groups"]["feature_counts"] == expected


def test_lag_and_confirmed_pivot_are_causal() -> None:
    source = np.arange(12, dtype=np.float32).reshape(6, 2)
    shifted = audit._shift_panel(source, 2)

    assert np.isnan(shifted[:2]).all()
    np.testing.assert_array_equal(shifted[2:], source[:-2])

    high = np.asarray([[1.0], [2.0], [5.0], [3.0], [2.0], [1.0]])
    events, levels = audit._confirmed_pivot_events(high, 2, mode="high")
    assert not events[:4].any()
    assert events[4, 0]
    assert levels[4, 0] == 5.0


def test_completed_period_mapping_never_uses_current_period() -> None:
    dates = [
        "2023-01-02",
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
        "2023-01-06",
        "2023-01-09",
        "2023-01-10",
        "2023-01-11",
        "2023-01-12",
        "2023-01-13",
    ]
    close = np.arange(1.0, 11.0, dtype=np.float32).reshape(-1, 1)
    period_high, period_low, period_close, completed = audit._aggregate_period_bars(
        date_values=dates,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        period="week",
    )

    assert completed.tolist() == [-1] * 5 + [0] * 5
    assert period_high[0, 0] == 5.5
    assert period_low[0, 0] == 0.5
    assert period_close[0, 0] == 5.0
    broadcast = audit._broadcast_completed_period(period_close, completed)
    assert np.isnan(broadcast[:5]).all()
    np.testing.assert_array_equal(broadcast[5:, 0], np.full(5, 5.0))


def test_turnover_cost_proxy_is_uniform_price_scale_invariant() -> None:
    close = np.asarray(
        [[10.0, 20.0], [11.0, 19.0], [12.0, 21.0], [11.5, 22.0]],
        dtype=np.float32,
    )
    turnover_pct = np.asarray(
        [[10.0, 5.0], [20.0, 8.0], [15.0, 12.0], [7.0, 10.0]],
        dtype=np.float32,
    )
    kwargs = {
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "log_turnover_pct": np.log1p(turnover_pct),
    }

    original = audit._turnover_cost_panels(**kwargs)
    scaled = audit._turnover_cost_panels(
        high=kwargs["high"] * 17.0,
        low=kwargs["low"] * 17.0,
        close=kwargs["close"] * 17.0,
        log_turnover_pct=kwargs["log_turnover_pct"],
    )

    for name in original:
        np.testing.assert_allclose(
            original[name], scaled[name], rtol=2.0e-6, atol=2.0e-6, equal_nan=True
        )


def test_hac_bh_and_negative_control_decisions_are_conservative() -> None:
    assert audit._one_sided_hac(np.zeros(20), 3)[
        "p_value_one_sided_positive"
    ] == pytest.approx(0.5)
    assert audit._one_sided_hac(-np.ones(20), 3)[
        "p_value_one_sided_positive"
    ] == pytest.approx(1.0)
    assert np.isnan(
        audit._one_sided_hac(np.asarray([1.0]), 0)[
            "p_value_one_sided_positive"
        ]
    )

    adjusted = audit._benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.04, "c": 0.04})

    head_status = {
        family: {"10": "omit", "20": "omit"}
        for family in audit.ALL_FAMILIES
    }
    head_status[audit.CONTROL_FAMILY]["10"] = "pass"
    decision = audit._compile_family_decision(
        head_status=head_status,
        evidence=[],
        thresholds={},
        non_selections=[],
    )
    assert decision["status"] == "invalid_negative_control"
    assert decision["family_status"] == {}


def test_no_incremental_family_has_distinct_completion_state() -> None:
    head_status = {
        family: {"10": "omit", "20": "omit"}
        for family in audit.ALL_FAMILIES
    }

    decision = audit._compile_family_decision(
        head_status=head_status,
        evidence=[],
        thresholds={},
        non_selections=["slot_count"],
    )

    assert decision["status"] == "completed_no_incremental_family"
    assert decision["next_step"] == "stop_and_reassess_base_inputs"


def test_prepared_family_checks_shape_path_and_candidate_count(tmp_path) -> None:
    family = "turnover_cost_proxy"
    family_root = tmp_path / "features" / family
    family_root.mkdir(parents=True)
    data_path = family_root / "features.float32.dat"
    values = np.zeros((5, 3), dtype=np.float32)
    values.tofile(data_path)
    manifest = {
        "schema": audit.PREPARED_FEATURE_SCHEMA,
        "family": family,
        "feature_names": list(audit.feature_catalog()[family]),
        "candidate_count": 3,
        "file": {
            "path": str(data_path),
            "size": int(data_path.stat().st_size),
            "shape": [5, 3],
            "dtype": "float32",
            "layout": "feature_major",
        },
    }
    (family_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    prepared = audit.load_prepared_family(
        tmp_path,
        family,
        candidate_count=3,
    )
    assert prepared.shape == (5, 3)
    with pytest.raises(ValueError, match="candidate count"):
        audit.load_prepared_family(
            tmp_path,
            family,
            candidate_count=4,
        )


def test_completed_task_loads_outputs_and_rejects_unreadable_prediction(tmp_path) -> None:
    import lightgbm as lgb
    import pandas as pd

    study = audit.load_study()
    task = audit._task_by_id("mfe10_2023_baseline_tuning")
    result_path = audit._task_result_path(tmp_path, task)
    result_path.parent.mkdir(parents=True)
    model_path = result_path.parent / "model.txt"
    prediction_path = result_path.parent / "prediction.npy"
    daily_path = result_path.parent / "daily_metrics.parquet"
    train = lgb.Dataset(
        np.arange(16, dtype=np.float32).reshape(8, 2),
        label=np.arange(8, dtype=np.float32),
    )
    lgb.train({"objective": "regression", "verbosity": -1}, train, 1).save_model(
        str(model_path)
    )
    np.save(prediction_path, np.arange(4, dtype=np.float32), allow_pickle=False)
    pd.DataFrame({"date_idx": [1], "rank_ic": [0.1]}).to_parquet(
        daily_path, index=False
    )
    parameters, _rounds, _patience = audit._model_parameters(study)
    result = {
        "schema": audit.TASK_RESULT_SCHEMA,
        "status": "completed",
        "study_id": audit.STUDY_ID,
        **task,
        "target": "mfe",
        "purge_days": 10,
        "parameters": parameters,
        "best_iteration": 1,
        "evaluation_row_count": 4,
        "files": {
            "model": {
                "path": str(model_path),
                "size": model_path.stat().st_size,
            },
            "prediction": {
                "path": str(prediction_path),
                "size": prediction_path.stat().st_size,
                "shape": [4],
            },
            "daily_metrics": {
                "path": str(daily_path),
                "size": daily_path.stat().st_size,
            },
        },
    }
    result["inner_validation_year"] = 2022
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert audit._task_complete(
        result_path, task=task, study=study, output_root=tmp_path
    )
    raw = bytearray(prediction_path.read_bytes())
    raw[0] ^= 0xFF
    prediction_path.write_bytes(raw)
    assert not audit._task_complete(
        result_path, task=task, study=study, output_root=tmp_path
    )
