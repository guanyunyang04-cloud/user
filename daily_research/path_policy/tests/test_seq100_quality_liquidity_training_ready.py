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
            "years": [2010],
            "available_for_feature_history": True,
            "eligible_for_training": False,
            "eligible_for_evaluation": False,
            "eligible_for_atlas_statistics": False,
            "eligible_for_labels": False,
        },
        "training": {"performed": False},
    }


def test_config_keeps_2010_out_of_formal_rows(tmp_path) -> None:
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")

    result = ready._load_config(path)

    assert result["period"]["research_start_date"] == "2011-01-01"
    assert result["burn_in"]["eligible_for_training"] is False


def test_config_rejects_training_or_2026(tmp_path) -> None:
    config = _config()
    config["training"]["performed"] = True
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ready.TrainingReadyError, match="must_not_train"):
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
            "symbol": ["000001.SZ", "600000.SH"],
        }
    ).to_parquet(support, index=False)
    pd.DataFrame(
        {
            "security_id": ["SZ-1", "SH-1"],
            "symbol": ["000001.SZ", "600000.SH"],
        }
    ).to_parquet(history, index=False)

    with duckdb.connect() as connection:
        result = connection.execute(
            ready._row_spine_sql(support_path=support, history_paths=[history])
        ).fetchdf()

    assert len(result) == 2
    assert result["security_id"].tolist() == ["SZ-1", "SH-1"]


def test_lagged_feature_joins_on_feature_available_date(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "candidate_id": [1],
            "trade_date": ["2011-01-05"],
            "security_id": ["SZ-1"],
        }
    ).to_parquet(spine, index=False)
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


def test_technical_block_marks_joined_all_null_row_as_warmup(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "candidate_id": [1],
            "trade_date": ["2011-01-05"],
            "security_id": ["SZ-1"],
        }
    ).to_parquet(spine, index=False)
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


def test_margin_block_joins_detail_and_exchange_market_data(tmp_path) -> None:
    spine = tmp_path / "spine.parquet"
    detail = tmp_path / "detail.parquet"
    market = tmp_path / "market.parquet"
    pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "trade_date": ["2011-01-05", "2011-01-05"],
            "security_id": ["SZ-1", "SH-1"],
            "symbol": ["000001.SZ", "600000.SH"],
        }
    ).to_parquet(spine, index=False)
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


def test_self_test_locks_qfq_and_date_boundaries() -> None:
    result = ready.self_test()

    assert result["status"] == "ok"
    assert result["checks"]["qfq_formal"] is False
    assert result["checks"]["forbidden_2026"] is True
