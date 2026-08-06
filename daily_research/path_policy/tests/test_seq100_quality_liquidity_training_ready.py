from __future__ import annotations

import json

import duckdb
import pandas as pd
import pytest

from daily_research.path_policy import seq100_quality_liquidity_training_ready as ready


def _config() -> dict[str, object]:
    return {
        "study_id": ready.STUDY_ID,
        "period": {
            "data_history_start": ready.DATA_HISTORY_START,
            "research_start_date": ready.RESEARCH_START,
            "end_date": ready.END_DATE,
            "forbidden_year": ready.FORBIDDEN_YEAR,
            "future_oos_prediction_years": list(ready.OOS_YEARS),
        },
        "burn_in": {
            "years": [2010, 2011],
            "available_for_feature_history": True,
            "eligible_for_training": False,
            "eligible_for_evaluation": False,
            "eligible_for_atlas_statistics": False,
            "eligible_for_labels": False,
        },
        "input_contract": {
            "row_spine_contract_version": ready.ROW_SPINE_CONTRACT_VERSION,
            "row_spine_projection": list(ready.ROW_SPINE_COLUMNS),
            "forbidden_inherited_metadata_columns": list(
                ready.FORBIDDEN_INHERITED_METADATA_COLUMNS
            ),
            "source_support_future_metadata_policy": "document_but_do_not_project",
            "symbol_history_cutoff_date": ready.END_DATE,
            "feature_loading": "feature_registry_whitelist_only",
            "label_validity_source": "cutoff_memmaps_and_label_flags_only",
        },
        "training": {"performed": False},
    }


def _spine_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "year": [2011, 2011],
            "trade_date": ["2011-01-05", "2011-01-05"],
            "date_idx": [1, 1],
            "symbol_idx": [10, 20],
            "symbol": ["000001.SZ", "600000.SH"],
            "security_id": ["SZ-1", "SH-1"],
        }
    )


def test_config_keeps_burn_in_out_of_formal_rows(tmp_path) -> None:
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")

    result = ready._load_config(path)

    assert result["period"]["research_start_date"] == "2012-01-01"
    assert result["burn_in"]["eligible_for_training"] is False


def test_config_rejects_training_or_2026(tmp_path) -> None:
    config = _config()
    config["training"]["performed"] = True
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ready.TrainingReadyError, match="must_not_train"):
        ready._load_config(path)


def test_config_rejects_unsafe_row_spine_projection(tmp_path) -> None:
    config = _config()
    config["input_contract"]["row_spine_projection"].append("entry_filled")
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ready.TrainingReadyError, match="input_contract_changed"):
        ready._load_config(path)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("close_qfq", "diagnostic_only"),
        ("rsi_qfq_6", "diagnostic_only"),
        ("close_hfq", "formal_candidate"),
        ("rsi_hfq_6", "formal_candidate"),
        ("macd_bfq", "formal_candidate"),
        ("close", "redundant_reference"),
        ("adj_factor", "redundant_reference"),
        ("ts_code", "rejected"),
    ],
)
def test_factor_eligibility(field, expected) -> None:
    eligibility, _ = ready.classify_factor_field(
        field,
        raw_price_validation_passed=True,
        hfq_validation_passed=True,
    )

    assert eligibility == expected


def test_failed_hfq_audit_removes_hfq_from_formal_candidates() -> None:
    eligibility, reason = ready.classify_factor_field(
        "rsi_hfq_6",
        raw_price_validation_passed=True,
        hfq_validation_passed=False,
    )

    assert eligibility == "rejected"
    assert reason == "hfq_adjustment_audit_failed"


@pytest.mark.parametrize(
    ("domain", "field", "expected"),
    [
        (ready.DataDomain.MARGIN_MARKET, "rzche", "diagnostic_only"),
        (ready.DataDomain.MARGIN_DETAIL, "rzche", "diagnostic_only"),
        (ready.DataDomain.MARGIN_DETAIL, "rqchl", "diagnostic_only"),
        (ready.DataDomain.MARGIN_DETAIL, "rzye", "formal_candidate"),
        (ready.DataDomain.MONEYFLOW_RAW, "net_mf_amount", "diagnostic_only"),
        (ready.DataDomain.MONEYFLOW_RAW, "buy_lg_amount", "formal_candidate"),
    ],
)
def test_extended_field_eligibility(domain, field, expected) -> None:
    assert ready.classify_extended_field(domain, field)[0] == expected


def test_row_spine_adds_exact_security_id_without_row_expansion(tmp_path) -> None:
    support = tmp_path / "support.parquet"
    history = tmp_path / "history.parquet"
    pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "year": [2011, 2011],
            "trade_date": ["2011-01-04", "2011-01-04"],
            "date_idx": [0, 0],
            "symbol_idx": [10, 20],
            "symbol": ["000001.SZ", "600000.SH"],
            "entry_trade_date": ["2011-01-05", "2011-01-05"],
            "entry_filled": [True, True],
            "label_valid": [True, True],
            "price_label_valid": [True, True],
            "va_aux_valid": [True, True],
        }
    ).to_parquet(support, index=False)
    pd.DataFrame(
        {
            "security_id": ["SZ-1", "SH-1", "SZ-FUTURE"],
            "symbol": ["000001.SZ", "600000.SH", "000001.SZ"],
            "effective_from": ["1991-01-01", "1999-01-01", "2026-01-02"],
        }
    ).to_parquet(history, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._row_spine_sql(support_path=support, history_paths=[history])
        ).fetchdf()

    assert len(result) == 2
    assert result["security_id"].tolist() == ["SZ-1", "SH-1"]
    assert tuple(result.columns) == ready.ROW_SPINE_COLUMNS
    assert not set(result.columns).intersection(
        ready.FORBIDDEN_INHERITED_METADATA_COLUMNS
    )
    history_profile = ready._history_mapping_profile([history])
    assert history_profile["mapping_row_count"] == 2
    assert history_profile["excluded_after_end_row_count"] == 1


def test_forbidden_date_detector_rejects_legacy_row_spine(tmp_path) -> None:
    path = tmp_path / "legacy-spine.parquet"
    frame = _spine_frame().iloc[:1].copy()
    frame["entry_trade_date"] = "2026-01-05"
    frame["entry_filled"] = True
    frame.to_parquet(path, index=False)

    assert ready._forbidden_date_row_count(path) == 1
    with pytest.raises(ready.TrainingReadyError, match="row_spine_schema_changed"):
        ready._assert_safe_row_spine_schema(path)


def test_source_support_boundary_documents_but_excludes_future_metadata(
    tmp_path, monkeypatch
) -> None:
    support = tmp_path / "support.parquet"
    frame = _spine_frame().iloc[:1].drop(columns="security_id")
    frame["entry_trade_date"] = "2026-01-05"
    frame["entry_filled"] = True
    frame["label_valid"] = True
    frame["price_label_valid"] = True
    frame["va_aux_valid"] = True
    frame.to_parquet(support, index=False)
    monkeypatch.setattr(ready, "RESEARCH_YEARS", (2025,))

    profile = ready._source_support_boundary_profile(
        {"membership_years": {"2025": {"support_path": str(support)}}}
    )

    assert profile["projection_columns"] == list(ready.SUPPORT_IDENTITY_COLUMNS)
    assert profile["explicit_future_date_row_count"] == 1
    assert profile["forbidden_metadata_columns_present"] == sorted(
        ready.FORBIDDEN_INHERITED_METADATA_COLUMNS
    )


def test_lagged_feature_joins_on_feature_available_date(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    source = tmp_path / "source.parquet"
    frame = _spine_frame().iloc[:1].copy()
    frame["entry_trade_date"] = "2026-01-05"
    frame["entry_filled"] = True
    frame.to_parquet(spine, index=False)
    pd.DataFrame(
        {
            "security_id": ["SZ-1"],
            "trade_date": ["2011-01-04"],
            "source_date": ["2011-01-04"],
            "feature_available_date": ["2011-01-05"],
            "rzye": [10.0],
        }
    ).to_parquet(source, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._feature_select_sql(
                spine_path=spine,
                source_paths=[source],
                fields=["rzye"],
                source_domain="margin_detail",
                lagged=True,
            )
        ).fetchdf()

    assert result.loc[0, "coverage_state"] == "observed"
    assert result.loc[0, "margin_detail_rzye"] == 10.0
    assert result.loc[0, "source_date"] == "2011-01-04"
    assert tuple(result.columns[: len(ready.ROW_SPINE_COLUMNS)]) == (
        ready.ROW_SPINE_COLUMNS
    )
    assert "entry_trade_date" not in result
    assert "entry_filled" not in result


def test_technical_block_marks_joined_all_null_row_as_warmup(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    source = tmp_path / "source.parquet"
    _spine_frame().iloc[:1].to_parquet(spine, index=False)
    pd.DataFrame(
        {
            "security_id": ["SZ-1"],
            "trade_date": ["2011-01-05"],
            "source_date": ["2011-01-05"],
            "feature_available_date": ["2011-01-05"],
            "rsi_hfq_6": [None],
        }
    ).to_parquet(source, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._feature_select_sql(
                spine_path=spine,
                source_paths=[source],
                fields=["rsi_hfq_6"],
                source_domain=ready.DataDomain.STK_FACTOR_PRO_RAW,
                lagged=False,
            )
        ).fetchdf()

    assert result.loc[0, "coverage_state"] == "warmup_missing"


def test_technical_block_gates_material_cross_provider_price_mismatch(
    tmp_path,
) -> None:
    spine = tmp_path / "spine.parquet"
    source = tmp_path / "source.parquet"
    daily = tmp_path / "daily.parquet"
    _spine_frame().iloc[:1].to_parquet(spine, index=False)
    pd.DataFrame(
        {
            "security_id": ["SZ-1"],
            "symbol": ["000001.SZ"],
            "trade_date": ["2011-01-05"],
            "source_date": ["2011-01-05"],
            "feature_available_date": ["2011-01-05"],
            "open": [11.0],
            "high": [11.0],
            "low": [11.0],
            "close": [11.0],
            "rsi_bfq_6": [55.0],
        }
    ).to_parquet(source, index=False)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": ["2011-01-05"],
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
        }
    ).to_parquet(daily, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._feature_select_sql(
                spine_path=spine,
                source_paths=[source],
                fields=["rsi_bfq_6"],
                source_domain=ready.DataDomain.STK_FACTOR_PRO_RAW,
                lagged=False,
                daily_paths=[daily],
            )
        ).fetchdf()

    assert result.loc[0, "coverage_state"] == "source_price_mismatch"
    assert pd.isna(result.loc[0, "rsi_bfq_6"])


def test_partition_profile_records_ordered_key_and_null_evidence(tmp_path) -> None:
    path = tmp_path / "block.parquet"
    profile_path = tmp_path / "block.profile.json"
    pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "trade_date": ["2011-01-04", "2011-01-04"],
            "security_id": ["SZ-1", "SH-1"],
            "value": [1.0, None],
        }
    ).to_parquet(path, index=False)

    row_key_hash = ready._row_key_hash(path)
    evidence = ready._partition_profile(
        path=path, profile_path=profile_path, row_key_hash=row_key_hash
    )

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert evidence["row_key_hash"] == row_key_hash
    assert profile["null_count"]["value"] == 1
    assert profile["null_rate"]["value"] == 0.5


def test_existing_registry_replaces_defective_listing_age_in_place(tmp_path) -> None:
    catalog_path = tmp_path / "feature_catalog.parquet"
    pd.DataFrame(
        {
            "name": ["feature_a", ready.LEGACY_LISTING_AGE_FIELD],
            "family": ["F1", "F5"],
        }
    ).to_parquet(catalog_path, index=False)
    atlas_path = tmp_path / "atlas_manifest.json"
    atlas_path.write_text(
        json.dumps({"files": {"feature_catalog": {"path": str(catalog_path)}}}),
        encoding="utf-8",
    )

    registry = ready._existing_registry(
        {"atlas": {"manifest_path": str(atlas_path)}}, tmp_path / "output"
    )

    assert len(registry) == 2
    assert not registry["feature_name"].eq(ready.LEGACY_LISTING_AGE_FIELD).any()
    listing = registry.loc[
        registry["feature_name"].eq(ready.LISTING_AGE_OPEN_DAYS_FIELD)
    ].iloc[0]
    assert listing["block"] == "membership_context"
    assert listing["physical_column"] == ready.LISTING_AGE_OPEN_DAYS_FIELD
    assert listing["source_field"] == "listed_open_days"
    assert registry["eligibility"].eq("formal_existing").sum() == 2


def test_membership_context_uses_complete_spine_and_open_day_age(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    diagnostics = tmp_path / "diagnostics.parquet"
    _spine_frame().to_parquet(spine, index=False)
    pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "listed_open_days": [250, 1234],
        }
    ).to_parquet(diagnostics, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._membership_context_sql(
                spine_path=spine, diagnostics_path=diagnostics
            )
        ).fetchdf()

    assert tuple(result.columns[: len(ready.ROW_SPINE_COLUMNS)]) == (
        ready.ROW_SPINE_COLUMNS
    )
    assert result[ready.LISTING_AGE_OPEN_DAYS_FIELD].tolist() == [250.0, 1234.0]
    assert result["coverage_state"].tolist() == ["observed", "observed"]


def test_margin_block_joins_detail_and_exchange_market_data(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    detail = tmp_path / "detail.parquet"
    market = tmp_path / "market.parquet"
    _spine_frame().to_parquet(spine, index=False)
    pd.DataFrame(
        {
            "security_id": ["SZ-1"],
            "source_date": ["2011-01-04"],
            "feature_available_date": ["2011-01-05"],
            "rzye": [10.0],
        }
    ).to_parquet(detail, index=False)
    pd.DataFrame(
        {
            "exchange_id": ["SZSE", "SSE"],
            "source_date": ["2011-01-04", "2011-01-04"],
            "feature_available_date": ["2011-01-05", "2011-01-05"],
            "rzrqye": [100.0, 200.0],
        }
    ).to_parquet(market, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._margin_block_sql(
                spine_path=spine,
                detail_paths=[detail],
                market_paths=[market],
                secs_paths=[],
                detail_fields=["rzye"],
                market_fields=["rzrqye"],
            )
        ).fetchdf()

    assert result["margin_detail_rzye"].tolist()[0] == 10.0
    assert pd.isna(result["margin_detail_rzye"].tolist()[1])
    assert result["margin_market_rzrqye"].tolist() == [100.0, 200.0]
    assert result["coverage_state"].tolist() == ["observed", "source_unavailable"]
    assert result["margin_detail_coverage_state"].tolist() == [
        "observed",
        "source_unavailable",
    ]
    assert result["margin_market_coverage_state"].tolist() == [
        "observed",
        "observed",
    ]
    assert set(result["margin_eligibility_state"]) == {"source_unavailable"}


def test_margin_block_preserves_exchange_tristate_metadata(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    detail = tmp_path / "detail.parquet"
    market = tmp_path / "market.parquet"
    eligibility = tmp_path / "eligibility.parquet"
    _spine_frame().to_parquet(spine, index=False)
    pd.DataFrame(
        {
            "security_id": ["SZ-1"],
            "source_date": ["2011-01-04"],
            "feature_available_date": ["2011-01-05"],
            "rzye": [10.0],
        }
    ).to_parquet(detail, index=False)
    pd.DataFrame(
        {
            "exchange_id": ["SZSE", "SSE"],
            "source_date": ["2011-01-04", "2011-01-04"],
            "feature_available_date": ["2011-01-05", "2011-01-05"],
            "rzrqye": [100.0, 200.0],
        }
    ).to_parquet(market, index=False)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "600000.SH"],
            "feature_available_date": ["2011-01-05", "2011-01-05"],
            "eligibility_state": ["eligible_observed", "known_ineligible"],
            "eligible": [True, False],
            "finance_eligible": [True, False],
            "securities_lending_eligible": [False, False],
            "detail_observed": [True, False],
            "source_available": [True, True],
            "eligibility_source_available": [True, True],
            "detail_source_available": [True, True],
        }
    ).to_parquet(eligibility, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._margin_block_sql(
                spine_path=spine,
                detail_paths=[detail],
                market_paths=[market],
                eligibility_paths=[eligibility],
                eligibility_domain=ready.DataDomain.MARGIN_ELIGIBILITY,
                detail_fields=["rzye"],
                market_fields=["rzrqye"],
            )
        ).fetchdf()

    assert result["margin_eligibility_state"].tolist() == [
        "eligible_observed",
        "known_ineligible",
    ]
    assert result["coverage_state"].tolist() == ["observed", "not_applicable"]
    assert result["margin_eligible"].tolist() == [True, False]


def test_balance_extension_uses_next_open_pit_asof_semantics(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    source = tmp_path / "balance.parquet"
    _spine_frame().to_parquet(spine, index=False)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "feature_available_date": ["2011-01-05"],
            "source_date": ["2011-01-04"],
            "report_date": ["2010-12-31"],
            "report_type": ["1"],
            "update_flag": [0],
            "other_receivables_total": [12.0],
            "other_payables_total": [8.0],
            "contract_liabilities": [3.0],
            "customer_advances_and_contract_liabilities": [5.0],
            "customer_liability_field_state": ["both_observed"],
            "other_receivables_total_field_state": ["observed"],
            "other_payables_total_field_state": ["observed"],
            "contract_liabilities_field_state": ["observed"],
            "balance_extension_source_conflict": [False],
            **{field: [1.0] for field in ready.BALANCE_SEMANTIC_NUMERIC_FIELDS},
            "trade_receivables_field_state": ["combined_observed_financing_unreported"],
            "fixed_assets_measure_field_state": ["component_fallback"],
            "construction_in_progress_measure_field_state": ["component_fallback"],
            "trade_payables_field_state": ["components_observed"],
            "balance_semantic_source_conflict": [False],
        }
    ).to_parquet(source, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._balance_extension_sql(
                spine_path=spine,
                source_paths=[source],
            )
        ).fetchdf()

    assert result["coverage_state"].tolist() == ["observed", "warmup_missing"]
    assert result.loc[0, "balance_contract_liabilities"] == 3.0
    assert result.loc[0, "balance_customer_liability_field_state"] == "both_observed"
    assert not bool(result.loc[0, "balance_extension_source_conflict"])


def test_balance_extension_conflict_gates_numeric_values(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    source = tmp_path / "balance.parquet"
    _spine_frame().to_parquet(spine, index=False)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "feature_available_date": ["2011-01-05"],
            "source_date": ["2011-01-04"],
            "report_date": ["2010-12-31"],
            "report_type": ["1"],
            "update_flag": [0],
            "other_receivables_total": [12.0],
            "other_payables_total": [8.0],
            "contract_liabilities": [3.0],
            "customer_advances_and_contract_liabilities": [5.0],
            "customer_liability_field_state": ["both_observed"],
            "other_receivables_total_field_state": ["observed"],
            "other_payables_total_field_state": ["observed"],
            "contract_liabilities_field_state": ["observed"],
            "balance_extension_source_conflict": [True],
            **{field: [1.0] for field in ready.BALANCE_SEMANTIC_NUMERIC_FIELDS},
            "trade_receivables_field_state": ["combined_observed_financing_unreported"],
            "fixed_assets_measure_field_state": ["component_fallback"],
            "construction_in_progress_measure_field_state": ["component_fallback"],
            "trade_payables_field_state": ["components_observed"],
            "balance_semantic_source_conflict": [False],
        }
    ).to_parquet(source, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._balance_extension_sql(
                spine_path=spine,
                source_paths=[source],
            )
        ).fetchdf()

    numeric = [f"balance_{field}" for field in ready.BALANCE_EXTENSION_NUMERIC_FIELDS]
    assert result.loc[0, numeric].isna().all()
    semantic = [f"balance_{field}" for field in ready.BALANCE_SEMANTIC_NUMERIC_FIELDS]
    assert result.loc[0, semantic].notna().all()
    assert bool(result.loc[0, "balance_extension_source_conflict"])


def test_balance_semantic_conflict_gates_only_semantic_values(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    source = tmp_path / "balance.parquet"
    _spine_frame().to_parquet(spine, index=False)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "feature_available_date": ["2011-01-05"],
            "source_date": ["2011-01-04"],
            "report_date": ["2010-12-31"],
            "report_type": ["1"],
            "update_flag": [0],
            **{field: [1.0] for field in ready.BALANCE_EXTENSION_NUMERIC_FIELDS},
            "customer_liability_field_state": ["both_observed"],
            "other_receivables_total_field_state": ["observed"],
            "other_payables_total_field_state": ["observed"],
            "contract_liabilities_field_state": ["observed"],
            "balance_extension_source_conflict": [False],
            **{field: [2.0] for field in ready.BALANCE_SEMANTIC_NUMERIC_FIELDS},
            "trade_receivables_field_state": ["combined_observed_financing_unreported"],
            "fixed_assets_measure_field_state": ["component_fallback"],
            "construction_in_progress_measure_field_state": ["component_fallback"],
            "trade_payables_field_state": ["components_observed"],
            "balance_semantic_source_conflict": [True],
        }
    ).to_parquet(source, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._balance_extension_sql(spine_path=spine, source_paths=[source])
        ).fetchdf()

    old_numeric = [
        f"balance_{field}" for field in ready.BALANCE_EXTENSION_NUMERIC_FIELDS
    ]
    semantic = [f"balance_{field}" for field in ready.BALANCE_SEMANTIC_NUMERIC_FIELDS]
    assert result.loc[0, old_numeric].notna().all()
    assert result.loc[0, semantic].isna().all()
    assert bool(result.loc[0, "balance_semantic_source_conflict"])


def test_self_test_locks_qfq_and_date_boundaries() -> None:
    result = ready.self_test()

    assert result["status"] == "ok"
    assert result["checks"]["qfq_formal"] is False
    assert result["checks"]["forbidden_2026"] is True
    assert (
        result["checks"]["row_spine_contract_version"]
        == ready.ROW_SPINE_CONTRACT_VERSION
    )
