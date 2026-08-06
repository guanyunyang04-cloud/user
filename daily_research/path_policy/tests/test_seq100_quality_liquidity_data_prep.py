from __future__ import annotations

import json
import warnings

import duckdb
import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_quality_liquidity_data_prep as prep
from daily_research.path_policy.seq100_quality_liquidity_data_prep import (
    EXPECTED_5M_TIMES,
    YEARS,
    _calendar_position_frames,
    _combined_feature_sample,
    _coordinate_row_mapping,
    _feature_block_current,
    _load_config,
    _membership_input_fingerprint,
    _minute_input_fingerprint,
    _pct_rank_sql,
    _sha256,
    _status_eligible_sql,
    _validate_feature_block,
    _year_bounds,
)


def test_qdp_snapshot_rejects_stale_pinned_dataset(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(prep, "qdp_v2_root", lambda _: tmp_path)
    monkeypatch.setattr(prep, "read_active_manifest", lambda _: {"datasets": {}})
    monkeypatch.setattr(
        prep,
        "active_dataset_map",
        lambda _: {"balance_sheet_quarterly": "active-balance"},
    )

    with pytest.raises(prep.DataPreparationError, match="pinned_qdp_datasets_not_active"):
        prep._qdp_snapshot(
            tmp_path,
            pinned_dataset_ids={"balance_sheet_quarterly": "stale-balance"},
        )


def test_five_minute_contract_has_exact_exchange_end_labels() -> None:
    assert len(EXPECTED_5M_TIMES) == 48
    assert len(set(EXPECTED_5M_TIMES)) == 48
    assert EXPECTED_5M_TIMES[:2] == ("093500000", "094000000")
    assert EXPECTED_5M_TIMES[23:25] == ("113000000", "130500000")
    assert EXPECTED_5M_TIMES[-1] == "150000000"


def test_years_end_at_2025_and_buffers_never_cross_start() -> None:
    assert YEARS == tuple(range(2010, 2026))
    assert _year_bounds(2010, buffer_days=60)[0] == "2010-01-01"
    assert _year_bounds(2025)[1] == "2025-12-31"


def test_quality_rank_direction_rewards_low_debt() -> None:
    ordinary = _pct_rank_sql("roe", "trade_date")
    debt = _pct_rank_sql("debt", "trade_date", descending=True)

    assert "ORDER BY roe ASC" in ordinary
    assert "ORDER BY debt DESC" in debt


def test_status_eligibility_requires_every_status_field_to_be_known() -> None:
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT "
            + _status_eligible_sql("s")
            + " FROM (VALUES "
            "(false,false,false),(true,false,false),(NULL,false,false),"
            "(false,NULL,false),(false,false,NULL)) "
            "AS s(is_st,is_suspended,is_delisted)"
        ).fetchall()
    finally:
        connection.close()

    assert rows == [(True,), (False,), (False,), (False,), (False,)]


def test_membership_input_fingerprint_changes_with_qdp_partition(tmp_path) -> None:
    paths = {}
    for domain in (
        "market_daily_raw",
        "universe_snapshot",
        "security_status",
        "valuation",
        "industry_concept",
        "financial_quarterly",
        "trading_calendar",
        "security_identity",
    ):
        path = tmp_path / f"{domain}.parquet"
        path.write_bytes(domain.encode())
        paths[domain] = [path]
    minute_path = tmp_path / "minute.parquet"
    calendar_path = tmp_path / "calendar.parquet"
    listing_path = tmp_path / "listing.parquet"
    for path in (minute_path, calendar_path, listing_path):
        path.write_bytes(path.name.encode())
    original_candidate_index = prep.CANDIDATE_INDEX
    candidate_index = tmp_path / "candidate_index.parquet"
    candidate_index.write_bytes(b"candidate")
    prep.CANDIDATE_INDEX = candidate_index
    try:
        before = _membership_input_fingerprint(
            qdp_paths=paths,
            minute_path=minute_path,
            calendar_path=calendar_path,
            listing_path=listing_path,
        )
        paths["security_status"][0].write_bytes(b"security-status-repaired")
        after = _membership_input_fingerprint(
            qdp_paths=paths,
            minute_path=minute_path,
            calendar_path=calendar_path,
            listing_path=listing_path,
        )
    finally:
        prep.CANDIDATE_INDEX = original_candidate_index

    assert before != after


def test_minute_input_fingerprint_includes_candidate_index(
    monkeypatch, tmp_path
) -> None:
    intraday = tmp_path / "intraday.parquet"
    candidate = tmp_path / "candidate_index.parquet"
    intraday.write_bytes(b"intraday")
    candidate.write_bytes(b"candidate-old")
    monkeypatch.setattr(prep, "CANDIDATE_INDEX", candidate)

    before = _minute_input_fingerprint([intraday])
    candidate.write_bytes(b"candidate-rebuilt")
    after = _minute_input_fingerprint([intraday])

    assert before != after


def test_coordinate_row_mapping_uses_stable_panel_coordinates() -> None:
    actual = _coordinate_row_mapping(
        current_date_idx=np.asarray([2, 1, 3]),
        current_symbol_idx=np.asarray([1, 2, 0]),
        legacy_date_idx=np.asarray([1, 2, 2]),
        legacy_symbol_idx=np.asarray([2, 0, 1]),
        symbol_count=10,
    )

    assert actual.tolist() == [2, 0, -1]


def test_safe_ratio_parenthesizes_composite_expressions() -> None:
    expression = prep._safe_ratio_sql("a+b", "assets")
    connection = duckdb.connect()
    try:
        actual = connection.execute(
            f"SELECT {expression} FROM (SELECT 20.0 AS a, 10.0 AS b, 100.0 AS assets)"
        ).fetchone()[0]
    finally:
        connection.close()

    assert actual == 0.3
    assert "THEN (a+b)/(assets)" in expression


def test_config_forbids_symbol_deletion_and_daily_fallback(tmp_path) -> None:
    source = {
        "study_id": "seq100_quality_liquidity_data_prep",
        "period": {"end_date": "2025-12-31", "forbidden_year": 2026},
        "common_support": {
            "requires_current_day_complete_5m": True,
            "drop_entire_symbol": False,
            "daily_fallback": False,
        },
        "training": {"performed": False},
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    assert _load_config(path)["common_support"]["drop_entire_symbol"] is False


def test_combined_feature_sample_loads_each_modulus_independently(
    monkeypatch, tmp_path
) -> None:
    observed_moduli: list[int] = []

    def fake_feature_sample(*, output_root, state, modulus, years):
        del output_root, state
        assert tuple(years) == YEARS
        observed_moduli.append(modulus)
        candidate_ids = np.array([modulus, modulus * 2], dtype=np.int64)
        return pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "trade_date": ["2020-01-02", "2020-01-03"],
                "year": [2020, 2020],
                "date_idx": [1, 2],
                "symbol_idx": [1, 2],
                "symbol": ["000001.SZ", "000002.SZ"],
                "new_feature": [1.0, np.inf],
            }
        )

    def fake_base_sample(candidate_ids):
        return pd.DataFrame({"base_feature": candidate_ids.astype(float)}), []

    monkeypatch.setattr(prep, "_feature_sample", fake_feature_sample)
    monkeypatch.setattr(prep, "_base_sample", fake_base_sample)

    association_source, association = _combined_feature_sample(
        output_root=tmp_path,
        state={},
        modulus=223,
    )
    redundancy_source, redundancy = _combined_feature_sample(
        output_root=tmp_path,
        state={},
        modulus=887,
    )

    assert observed_moduli == [223, 887]
    assert association_source["candidate_id"].tolist() == [223, 446]
    assert redundancy_source["candidate_id"].tolist() == [887, 1774]
    assert association["base_feature"].tolist() == [223.0, 446.0]
    assert redundancy["base_feature"].tolist() == [887.0, 1774.0]
    assert np.isnan(association.loc[1, "new_feature"])


def test_listing_age_includes_open_days_before_research_start() -> None:
    open_dates = pd.bdate_range("2009-01-01", periods=300)
    calendar = pd.DataFrame({"trade_date": open_dates, "is_open": True})
    identity = pd.DataFrame(
        {"current_symbol": ["000001.SZ"], "list_date": ["2009-01-01"]}
    )

    positions, listings = _calendar_position_frames(calendar, identity)

    assert listings.loc[0, "list_open_index"] == 0
    assert (
        positions.loc[249, "open_index"] - listings.loc[0, "list_open_index"] + 1 == 250
    )


def test_feature_cache_rejects_changed_candidate_support(tmp_path) -> None:
    feature_path = tmp_path / "features.parquet"
    support_path = tmp_path / "support.parquet"
    pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "trade_date": ["2020-01-02", "2020-01-03"],
            "value": [0.1, 0.2],
        }
    ).to_parquet(feature_path, index=False)
    pd.DataFrame(
        {
            "candidate_id": [1, 3],
            "trade_date": ["2020-01-02", "2020-01-03"],
        }
    ).to_parquet(support_path, index=False)
    existing = {
        "status": "completed",
        "sha256": _sha256(feature_path),
    }
    connection = duckdb.connect()
    try:
        assert not _feature_block_current(
            connection,
            existing=existing,
            support_path=support_path,
            feature_path=feature_path,
            input_fingerprint="current-input",
        )
    finally:
        connection.close()


def test_feature_alignment_rejects_changed_symbol_coordinates(tmp_path) -> None:
    support_path = tmp_path / "support.parquet"
    feature_path = tmp_path / "features.parquet"
    common = {
        "candidate_id": [1],
        "year": [2020],
        "trade_date": ["2020-01-02"],
        "date_idx": [10],
    }
    pd.DataFrame(
        {**common, "symbol_idx": [1], "symbol": ["000001.SZ"]}
    ).to_parquet(support_path, index=False)
    pd.DataFrame(
        {
            **common,
            "symbol_idx": [2],
            "symbol": ["000002.SZ"],
            "value": [0.1],
        }
    ).to_parquet(feature_path, index=False)

    connection = duckdb.connect()
    try:
        alignment = _validate_feature_block(
            connection,
            support_path=support_path,
            feature_path=feature_path,
            raise_on_error=False,
        )
    finally:
        connection.close()

    assert alignment["support_rows"] == alignment["feature_rows"] == 1
    assert alignment["support_missing_rows"] == 1
    assert alignment["feature_extra_rows"] == 1


def test_reused_atlas_restores_completed_top_level_state(monkeypatch, tmp_path) -> None:
    atlas_root = tmp_path / "atlas"
    atlas_root.mkdir()
    manifest_path = atlas_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "builder_version": prep.ATLAS_BUILDER_VERSION,
                "common_support_hash": "support-hash",
                "feature_blocks_hash": "blocks-hash",
            }
        ),
        encoding="utf-8",
    )
    state = {
        "status": "feature_blocks_prepared",
        "common_support_hash": "support-hash",
        "atlas": {
            "status": "completed",
            "manifest_sha256": _sha256(manifest_path),
        },
    }
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        prep,
        "_feature_blocks_hash",
        lambda value, *, years: "blocks-hash",
    )
    monkeypatch.setattr(
        prep,
        "_write_state",
        lambda output_root, value: writes.append(dict(value)),
    )

    result = prep.prepare_atlas(output_root=tmp_path, state=state)

    assert result == state["atlas"]
    assert state["status"] == "completed"
    assert state["training_performed"] is False
    assert state["feature_set_selected"] is False
    assert writes[-1]["status"] == "completed"


def test_atlas_statistics_handle_constant_features_without_runtime_warning() -> None:
    features = pd.DataFrame(
        {
            "constant": np.ones(100),
            "varying": np.arange(100, dtype=np.float64),
        }
    )
    labels = pd.DataFrame(
        {
            "mfe_10": np.arange(100, dtype=np.float64),
            "mfe_20": np.arange(100, dtype=np.float64)[::-1],
        }
    )
    years = np.full(100, 2023)
    catalog = pd.DataFrame(
        {
            "name": ["constant", "varying"],
            "family": ["test", "test"],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        associations = prep._association_rows(features, labels, years, catalog)
        redundancy = prep._redundancy_rows(features, catalog)

    constant_rows = [row for row in associations if row["feature"] == "constant"]
    assert constant_rows
    assert all(np.isnan(row["spearman"]) for row in constant_rows)
    assert redundancy == []
