from __future__ import annotations

import json

import pandas as pd

from daily_research.path_policy import seq100_strict_chan_audit as audit
from daily_research.path_policy import seq100_strict_chan_stratified_audit as stratified


def _selection_spec() -> dict[str, object]:
    return {
        "selection": {
            "random_seed": 7,
            "globally_unique_symbols": True,
            "focal_date_bounds": {
                "earliest": "2012-10-01",
                "latest": "2013-09-15",
            },
            "window": {
                "parse_lookback_calendar_days": 270,
                "parse_lookahead_calendar_days": 75,
                "display_before_calendar_days": 30,
                "display_after_calendar_days": 30,
            },
            "strata": [
                {
                    "stratum_id": "sh_first",
                    "years": [2012],
                    "symbol_suffix": ".SH",
                    "samples": 1,
                },
                {
                    "stratum_id": "sh_second",
                    "years": [2013],
                    "symbol_suffix": ".SH",
                    "samples": 1,
                },
                {
                    "stratum_id": "sz_second",
                    "years": [2013],
                    "symbol_suffix": ".SZ",
                    "samples": 1,
                },
            ],
        }
    }


def test_frozen_stratified_contract_is_outcome_blind() -> None:
    spec = stratified.load_stratified_spec()
    selection = spec["selection"]
    assert spec["study_id"] == stratified.STUDY_ID
    assert tuple(selection["selection_columns"]) == stratified.SELECTION_COLUMNS
    assert "future_return" not in selection["selection_columns"]
    assert spec["boundaries"]["return_test_performed"] is False
    assert len(selection["strata"]) == 14


def test_selection_is_deterministic_and_ignores_outcome_columns() -> None:
    spec = _selection_spec()
    frames = {
        2012: pd.DataFrame(
            {
                "symbol": ["600001.SH", "600002.SH", "000001.SZ"],
                "trade_date": ["2012-10-03", "2012-11-06", "2012-12-03"],
                "future_return": [9.0, -9.0, 3.0],
            }
        ),
        2013: pd.DataFrame(
            {
                "symbol": ["600001.SH", "600003.SH", "000002.SZ"],
                "trade_date": ["2013-01-04", "2013-04-08", "2013-07-03"],
                "future_return": [-99.0, 99.0, 0.5],
            }
        ),
    }
    first_cases, first_statistics = stratified.select_cases_from_frames(spec, frames)
    changed = {
        year: frame.sample(frac=1.0, random_state=year).assign(future_return=0.0)
        for year, frame in frames.items()
    }
    second_cases, second_statistics = stratified.select_cases_from_frames(spec, changed)

    first_keys = [
        (case["case_id"], case["symbol"], case["focal_date"], case["selection_rank"])
        for case in first_cases
    ]
    second_keys = [
        (case["case_id"], case["symbol"], case["focal_date"], case["selection_rank"])
        for case in second_cases
    ]
    assert first_keys == second_keys
    assert first_statistics == second_statistics
    assert len({case["symbol"] for case in first_cases}) == len(first_cases)
    for case in first_cases:
        assert case["start_date"] <= case["display_start_date"] <= case["focal_date"]
        assert case["focal_date"] <= case["display_end_date"] <= case["end_date"]
        assert "future return" in case["selection_basis"].lower()


def test_materialized_contract_is_accepted_by_generic_audit(tmp_path) -> None:
    spec = stratified.load_stratified_spec()
    sample = {
        "selection_fingerprint": "sample-fingerprint",
        "cases": [
            {
                "case_id": "stratified_test_01",
                "category": "stratified_quality_pool",
                "stratum_id": "test",
                "symbol": "600000.SH",
                "start_date": "2024-01-01",
                "display_start_date": "2024-02-01",
                "focal_date": "2024-02-20",
                "display_end_date": "2024-02-29",
                "end_date": "2024-03-29",
                "quality_pool_expected": True,
                "selection_basis": "Outcome-blind test case.",
                "selection_fingerprint": "sample-fingerprint",
                "assertions": [],
            }
        ],
    }
    materialized = stratified._materialized_audit_spec(
        spec,
        sample,
        sample_manifest_path=tmp_path / "sample_manifest.json",
    )
    path = tmp_path / "frozen_audit_spec.json"
    path.write_text(json.dumps(materialized), encoding="utf-8")

    loaded = audit.load_audit_spec(path)
    assert loaded["study_id"] == stratified.STUDY_ID
    assert loaded["cases"][0]["case_id"] == "stratified_test_01"
