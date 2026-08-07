from __future__ import annotations

from pathlib import Path

from daily_research.path_policy import seq100_hot_path_neutral as neutral


def test_neutral_contract_separates_activation_from_low_amount() -> None:
    study = neutral.load_study()
    names = {item["name"] for item in study["contrasts"]}
    assert "activation_increment" in names
    assert "low_amount_increment" in names
    assert study["analysis"]["profit_claim_allowed"] is False
    assert study["neutralization"]["primary_match_spec"] == (
        "date_industry_size10"
    )
    assert any(
        item.get("turnover_bins")
        for item in study["neutralization"]["match_specs"]
    )


def test_cohort_conditions_keep_primary_coordinates_distinct() -> None:
    study = neutral.load_study()
    selected = neutral._cohort_condition("selected_pair", study)
    activation_control = neutral._cohort_condition(
        "same_amount_other_attention", study
    )
    low_amount = neutral._cohort_condition("low_amount", study)
    assert "attention_bin = 3" in selected
    assert "amount_bin = 0" in selected
    assert "attention_bin <> 3" in activation_control
    assert low_amount == "amount_bin = 0"


def test_primary_match_query_uses_signal_time_industry_and_size() -> None:
    study = neutral.load_study()
    contrast = next(
        item for item in study["contrasts"] if item["name"] == "activation_increment"
    )
    spec = next(
        item
        for item in study["neutralization"]["match_specs"]
        if item["name"] == "date_industry_size10"
    )
    query = neutral._matched_date_query(
        [Path("one.parquet")], study, contrast, spec
    )
    assert "industry" in query
    assert "size_bin_10" in query
    assert "outcome_match_coverage" in query
    assert "d5_match_coverage" in query
    assert "valid_close_days_5 = 5" in query
    assert "net_return_20_difference" in query


def test_turnover_match_adds_turnover_stratum() -> None:
    study = neutral.load_study()
    contrast = next(
        item for item in study["contrasts"] if item["name"] == "low_amount_increment"
    )
    spec = next(
        item
        for item in study["neutralization"]["match_specs"]
        if item["name"] == "date_size10_turnover10"
    )
    query = neutral._matched_date_query(
        [Path("one.parquet")], study, contrast, spec
    )
    assert "turnover_bin_10" in query
    assert "amount_to_circ_mv > 0" in query
