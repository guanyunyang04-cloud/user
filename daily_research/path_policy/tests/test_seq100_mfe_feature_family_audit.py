from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from daily_research.path_policy import seq100_mfe_feature_family_audit as audit


def test_load_study_freezes_contract_and_2026_firewall(tmp_path) -> None:
    study = audit.load_study()

    assert study["contract_sha256"] == audit._canonical_json_sha256(
        study["contract"]
    )
    assert study["contract"]["protocol"]["fold_years"] == [2023, 2024, 2025]
    assert study["contract"]["protocol"]["primary_horizons"] == [10, 20]
    assert study["contract"]["scientific_firewall"]["forbidden_years"] == [2026]

    changed = json.loads(json.dumps(study))
    changed["contract"]["scientific_firewall"]["forbidden_years"] = [2027]
    changed["contract_sha256"] = audit._canonical_json_sha256(changed["contract"])
    changed_path = tmp_path / "changed_contract.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="2026 must remain forbidden"):
        audit.load_study(changed_path)


def test_feature_catalog_counts_and_hash_are_frozen() -> None:
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
    assert (
        audit.feature_catalog_sha256()
        == "0f91e81c7181e04336b5d43c151859d92ae7d31c40641f3905fe6a157dfbfc63"
    )


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


def test_prepared_family_is_bound_to_contract_and_candidate_count(tmp_path) -> None:
    family = "turnover_cost_proxy"
    family_root = tmp_path / "features" / family
    family_root.mkdir(parents=True)
    data_path = family_root / "features.float32.dat"
    values = np.zeros((5, 3), dtype=np.float32)
    values.tofile(data_path)
    manifest = {
        "schema": audit.PREPARED_FEATURE_SCHEMA,
        "family": family,
        "contract_sha256": "contract",
        "feature_catalog_sha256": audit.feature_catalog_sha256(),
        "feature_names": list(audit.feature_catalog()[family]),
        "candidate_count": 3,
        "file": {
            "path": str(data_path),
            "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
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
        verify_hash=True,
        contract_sha256="contract",
        candidate_count=3,
    )
    assert prepared.shape == (5, 3)
    with pytest.raises(ValueError, match="another contract"):
        audit.load_prepared_family(
            tmp_path,
            family,
            verify_hash=False,
            contract_sha256="other",
            candidate_count=3,
        )


def test_completed_task_rejects_same_size_file_corruption(tmp_path) -> None:
    data_path = tmp_path / "prediction.npy"
    data_path.write_bytes(b"abcd")
    result_path = tmp_path / "task_result.json"
    result = {
        "schema": audit.TASK_RESULT_SCHEMA,
        "status": "completed",
        "study_id": audit.STUDY_ID,
        "contract_sha256": "contract",
        "files": {
            "prediction": {
                "path": str(data_path),
                "size": 4,
                "sha256": hashlib.sha256(b"abcd").hexdigest(),
            }
        },
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert audit._task_complete(result_path, "contract")
    data_path.write_bytes(b"wxyz")
    assert not audit._task_complete(result_path, "contract")
