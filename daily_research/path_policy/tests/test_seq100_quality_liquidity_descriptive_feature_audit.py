from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import (
    seq100_quality_liquidity_descriptive_feature_audit as audit,
)


def test_period_for_year_covers_only_research_years() -> None:
    assert audit.period_for_year(2011) == "2011_2015"
    assert audit.period_for_year(2018) == "2016_2020"
    assert audit.period_for_year(2025) == "2021_2025"


def test_recent_reversal_requires_opposite_equal_or_larger_effect() -> None:
    effects = {year: 0.02 for year in range(2011, 2023)}
    effects.update({2023: -0.03, 2024: -0.02, 2025: -0.04})
    assert audit.recent_equal_or_larger_reversal(effects)

    effects.update({2023: -0.005, 2024: -0.004, 2025: -0.003})
    assert not audit.recent_equal_or_larger_reversal(effects)

    short_history = {2021: 0.01, 2022: 0.01}
    short_history.update({2023: -0.02, 2024: -0.02, 2025: -0.02})
    assert audit.recent_equal_or_larger_reversal(short_history)


def test_family_classification_respects_availability_gate() -> None:
    assert (
        audit.classify_family(
            stable_feature_count=2,
            stable_nonreversed_feature_count=1,
            availability_gated=False,
            formal_eligible_feature_count=5,
        )
        == "first_model_formal_family"
    )
    assert (
        audit.classify_family(
            stable_feature_count=2,
            stable_nonreversed_feature_count=1,
            availability_gated=True,
            formal_eligible_feature_count=5,
        )
        == "availability_gated_family"
    )
    assert (
        audit.classify_family(
            stable_feature_count=0,
            stable_nonreversed_feature_count=0,
            availability_gated=False,
            formal_eligible_feature_count=0,
        )
        == "diagnostic_only"
    )
    assert (
        audit.classify_family(
            stable_feature_count=5,
            stable_nonreversed_feature_count=1,
            availability_gated=False,
            formal_eligible_feature_count=20,
            minimum_independent_representatives=2,
        )
        == "defer_from_initial_model"
    )


def test_independent_representatives_do_not_count_redundant_twins() -> None:
    redundancy = pd.DataFrame(
        {
            "feature_left": ["a"],
            "feature_right": ["b"],
            "absolute_spearman": [0.999],
        }
    )

    representatives = audit._independent_representatives({"a", "b", "c"}, redundancy)

    assert "c" in representatives
    assert len(set(representatives) & {"a", "b"}) == 1
    assert len(representatives) == 2


def test_large_family_recommendation_uses_independent_representatives() -> None:
    features = [f"feature_{index}" for index in range(10)]
    catalog = pd.DataFrame(
        {
            "feature_name": features,
            "analytic_family": ["test"] * len(features),
            "eligibility": ["formal_candidate"] * len(features),
        }
    )
    registry = pd.DataFrame(
        columns=["feature_name", "eligibility", "block", "source_domain"]
    )
    stability = pd.DataFrame(
        {
            "feature": features[:2],
            "analytic_family": ["test", "test"],
            "stable_ten_year_relation": [True, True],
            "stable_without_recent_reversal": [True, True],
        }
    )
    redundancy = pd.DataFrame(
        {"feature_left": [features[0]], "feature_right": [features[1]]}
    )
    config = {
        "stability_gate": {
            "large_family_minimum_formal_features": 10,
            "minimum_independent_stable_members_large_family": 2,
        }
    }

    result = audit._recommendations(
        catalog, registry, stability, redundancy, config
    ).iloc[0]

    assert result["independent_stable_representative_count"] == 1
    assert result["classification"] == "defer_from_initial_model"


def test_relation_rows_keep_missing_values_out_of_real_zero() -> None:
    values = pd.DataFrame({"feature": [0.0, np.nan, 1.0, 2.0, 3.0]})
    bundle: dict[str, np.ndarray] = {"date_idx": np.ones(5, dtype=np.int32)}
    for horizon in audit.HORIZONS:
        bundle[f"mfe_{horizon}"] = np.arange(5, dtype=np.float64)
        bundle[f"mae_{horizon}"] = -np.arange(5, dtype=np.float64)
        bundle[f"state_{horizon}"] = np.array([0, 0, 1, 2, 2], dtype=np.int8)
        bundle[f"valid_{horizon}"] = np.ones(5, dtype=bool)
        bundle[f"state_valid_{horizon}"] = np.ones(5, dtype=bool)
        bundle[f"mfe_rank_{horizon}"] = np.arange(1, 6, dtype=np.float64) / 5
        bundle[f"top1_{horizon}"] = np.array([False, False, False, False, True])
        bundle[f"top5_{horizon}"] = np.array([False, False, False, False, True])
    catalog = pd.DataFrame(
        [
            {
                "feature_name": "feature",
                "analytic_family": "test",
                "eligibility": "formal_candidate",
            }
        ]
    )

    relation, coverage = audit._relation_rows_for_chunk(
        year=2025,
        values=values,
        bundle=bundle,
        catalog=catalog,
        tail_fraction=0.25,
        minimum_valid_rows=1,
    )

    assert coverage[0]["nonnull_count"] == 4
    assert coverage[0]["distinct_count"] == 4
    assert all(row["valid_count"] == 4 for row in relation)


def test_aggregate_relation_sums_tail_hits() -> None:
    rows = []
    for year in (2011, 2012):
        rows.append(
            {
                "year": year,
                "feature": "x",
                "analytic_family": "test",
                "eligibility": "formal_candidate",
                "horizon": 10,
                "relation_axis": "within_trade_date_cross_section",
                "valid_count": 100,
                "high_count": 10,
                "low_count": 10,
                "spearman_daily_rank": 0.1,
                "baseline_top1_hits": 2,
                "baseline_top5_hits": 10,
                "high_top1_hits": 1,
                "low_top1_hits": 0,
                "high_top5_hits": 2,
                "low_top5_hits": 1,
                "high_mfe_mean": 0.1,
                "low_mfe_mean": 0.05,
                "high_mfe_median": 0.08,
                "low_mfe_median": 0.04,
                "high_mae_mean": -0.02,
                "low_mae_mean": -0.03,
                "high_mae_median": -0.01,
                "low_mae_median": -0.02,
                "high_state_count": 10,
                "low_state_count": 10,
                "high_state_high_hits": 4,
                "low_state_high_hits": 2,
            }
        )

    result = audit._aggregate_relation(pd.DataFrame(rows), (2011, 2012), "all")

    assert len(result) == 1
    assert result.iloc[0]["high_top5_rate"] == 0.2
    assert result.iloc[0]["low_top5_rate"] == 0.1
    assert result.iloc[0]["high_top5_lift"] == 2.0


def test_market_wide_feature_uses_trade_date_time_axis() -> None:
    date_idx = np.repeat(np.arange(10, dtype=np.int32), 2)
    values = pd.DataFrame({"market": np.repeat(np.arange(10, dtype=float), 2)})
    bundle: dict[str, np.ndarray] = {"date_idx": date_idx}
    for horizon in audit.HORIZONS:
        bundle[f"mfe_{horizon}"] = np.linspace(0, 1, 20)
        bundle[f"mae_{horizon}"] = -np.linspace(0, 1, 20)
        bundle[f"state_{horizon}"] = np.tile([0, 2], 10).astype(np.int8)
        bundle[f"valid_{horizon}"] = np.ones(20, dtype=bool)
        bundle[f"state_valid_{horizon}"] = np.ones(20, dtype=bool)
        bundle[f"mfe_rank_{horizon}"] = np.tile([0.5, 1.0], 10)
        bundle[f"top1_{horizon}"] = np.tile([False, True], 10)
        bundle[f"top5_{horizon}"] = np.tile([False, True], 10)
    catalog = pd.DataFrame(
        [
            {
                "feature_name": "market",
                "analytic_family": "market_state",
                "eligibility": "formal_existing",
            }
        ]
    )

    relation, coverage = audit._relation_rows_for_chunk(
        year=2025,
        values=values,
        bundle=bundle,
        catalog=catalog,
        tail_fraction=0.2,
        minimum_valid_rows=1,
    )

    assert coverage[0]["relation_axis"] == "annual_trade_date_time_series"
    assert relation[0]["high_count"] > 0
    assert relation[0]["low_count"] > 0


def test_feature_catalog_uses_canonical_listing_age_and_keeps_count(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.parquet"
    legacy_rows = [
        {
            "name": f"legacy_{index}",
            "family": "F1",
            "source_block": "existing_seq100_base",
        }
        for index in range(517)
    ]
    legacy_rows.append(
        {
            "name": "listing_age_days",
            "family": "F5",
            "source_block": "existing_seq100_base",
        }
    )
    pd.DataFrame(legacy_rows).to_parquet(legacy_path, index=False)
    atlas_manifest = tmp_path / "atlas.json"
    atlas_manifest.write_text(
        json.dumps({"files": {"feature_catalog": {"path": str(legacy_path)}}}),
        encoding="utf-8",
    )
    existing_names = [f"legacy_{index}" for index in range(517)] + [
        "listing_age_open_days"
    ]
    registry_rows = [
        {
            "feature_name": name,
            "physical_column": name,
            "block": "membership_context"
            if name == "listing_age_open_days"
            else "existing_atlas_518",
            "source_domain": "quality_liquidity_membership"
            if name == "listing_age_open_days"
            else "existing_seq100_atlas",
            "source_field": "listed_open_days"
            if name == "listing_age_open_days"
            else name,
            "eligibility": "formal_existing",
            "eligibility_reason": "test",
        }
        for name in existing_names
    ]
    registry_rows.extend(
        {
            "feature_name": f"candidate_{index}",
            "physical_column": f"candidate_{index}",
            "block": "tushare_technical_candidates",
            "source_domain": "stk_factor_pro_raw",
            "source_field": f"candidate_{index}",
            "eligibility": "formal_candidate",
            "eligibility_reason": "test",
        }
        for index in range(110)
    )
    registry_path = tmp_path / "registry.parquet"
    pd.DataFrame(registry_rows).to_parquet(registry_path, index=False)
    source_manifest = {
        "existing_atlas_518": {"manifest_path": str(atlas_manifest)},
        "feature_registry": {"path": str(registry_path)},
    }

    catalog = audit._feature_catalog(source_manifest, tmp_path)

    assert len(catalog) == 628
    assert catalog["feature_name"].eq("listing_age_open_days").sum() == 1
    assert not catalog["feature_name"].eq("listing_age_days").any()
    listing = catalog.loc[catalog["feature_name"].eq("listing_age_open_days")].iloc[0]
    assert listing["block"] == "membership_context"
    assert listing["analytic_family"] == "size_liquidity_and_status"


def test_feature_catalog_rejects_legacy_listing_age_in_formal_registry(
    tmp_path,
) -> None:
    legacy_path = tmp_path / "legacy.parquet"
    pd.DataFrame(
        {
            "name": ["listing_age_days"],
            "family": ["F5"],
            "source_block": ["existing_seq100_base"],
        }
    ).to_parquet(legacy_path, index=False)
    atlas_manifest = tmp_path / "atlas.json"
    atlas_manifest.write_text(
        json.dumps({"files": {"feature_catalog": {"path": str(legacy_path)}}}),
        encoding="utf-8",
    )
    registry_path = tmp_path / "registry.parquet"
    pd.DataFrame(
        [
            {
                "feature_name": "listing_age_days",
                "physical_column": "listing_age_days",
                "block": "existing_atlas_518",
                "source_domain": "existing_seq100_atlas",
                "source_field": "listing_age_days",
                "eligibility": "formal_existing",
                "eligibility_reason": "test",
            }
        ]
    ).to_parquet(registry_path, index=False)

    with pytest.raises(
        audit.DescriptiveAuditError,
        match="canonical_listing_age_field_missing_or_duplicated",
    ):
        audit._feature_catalog(
            {
                "existing_atlas_518": {"manifest_path": str(atlas_manifest)},
                "feature_registry": {"path": str(registry_path)},
            },
            tmp_path,
        )
