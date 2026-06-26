from __future__ import annotations

from quant_data_platform.domains.contracts import DataDomain
from daily_research.path_policy import forecast_features as ff
from quant_data_platform.domains.contracts import (
    CANONICAL_BUNDLE_SIDECAR_DOMAINS,
    EXCLUDED_V1_DOMAINS,
    PROFILE_DOMAIN_POLICY,
)
from quant_data_platform.features.profiles import (
    MEDIUM_HORIZON_PROFILE,
    SHORT_HORIZON_CORE_PROFILE,
    STYLE_STRUCTURAL_ALPHA_PROFILE,
    STYLE_STRUCTURAL_PROFILE,
)


def test_canonical_v1_keeps_structural_domains_and_excludes_slow_disclosure_domains() -> None:
    assert DataDomain.VALUATION in CANONICAL_BUNDLE_SIDECAR_DOMAINS
    assert DataDomain.INDUSTRY_CONCEPT in CANONICAL_BUNDLE_SIDECAR_DOMAINS
    assert DataDomain.INDEX_CONSTITUENTS in CANONICAL_BUNDLE_SIDECAR_DOMAINS
    assert DataDomain.FINANCIAL_QUARTERLY in EXCLUDED_V1_DOMAINS
    assert DataDomain.PERFORMANCE_FORECAST in EXCLUDED_V1_DOMAINS
    assert DataDomain.PERFORMANCE_EXPRESS in EXCLUDED_V1_DOMAINS


def test_short_profile_excludes_structural_context_but_style_profiles_include_it() -> None:
    assert SHORT_HORIZON_CORE_PROFILE in ff.FORECAST_FEATURE_PROFILES
    assert STYLE_STRUCTURAL_PROFILE in ff.FORECAST_FEATURE_PROFILES
    assert STYLE_STRUCTURAL_ALPHA_PROFILE in ff.FORECAST_FEATURE_PROFILES
    assert MEDIUM_HORIZON_PROFILE in ff.FORECAST_FEATURE_PROFILES

    assert SHORT_HORIZON_CORE_PROFILE not in ff.VALUATION_CONTEXT_PROFILES
    assert SHORT_HORIZON_CORE_PROFILE not in ff.INDEX_CONTEXT_PROFILES
    assert SHORT_HORIZON_CORE_PROFILE not in ff.SECTOR_CONTEXT_PROFILES
    assert SHORT_HORIZON_CORE_PROFILE not in ff.FINANCE_CONTEXT_PROFILES

    assert STYLE_STRUCTURAL_PROFILE in ff.VALUATION_CONTEXT_PROFILES
    assert STYLE_STRUCTURAL_PROFILE in ff.INDEX_CONTEXT_PROFILES
    assert STYLE_STRUCTURAL_PROFILE in ff.SECTOR_CONTEXT_PROFILES
    assert STYLE_STRUCTURAL_ALPHA_PROFILE in ff.VALUATION_CONTEXT_PROFILES
    assert STYLE_STRUCTURAL_ALPHA_PROFILE in ff.INDEX_CONTEXT_PROFILES
    assert STYLE_STRUCTURAL_ALPHA_PROFILE in ff.SECTOR_CONTEXT_PROFILES
    assert STYLE_STRUCTURAL_ALPHA_PROFILE in ff.ALPHA_FORECAST_CLEAN_PROFILES
    assert MEDIUM_HORIZON_PROFILE in ff.VALUATION_CONTEXT_PROFILES
    assert MEDIUM_HORIZON_PROFILE in ff.INDEX_CONTEXT_PROFILES
    assert MEDIUM_HORIZON_PROFILE not in ff.FINANCE_CONTEXT_PROFILES


def test_profile_domain_policy_keeps_financial_domains_out_of_current_profiles() -> None:
    for profile in (SHORT_HORIZON_CORE_PROFILE, STYLE_STRUCTURAL_PROFILE, STYLE_STRUCTURAL_ALPHA_PROFILE, MEDIUM_HORIZON_PROFILE):
        policy = PROFILE_DOMAIN_POLICY[profile]
        assert DataDomain.FINANCIAL_QUARTERLY in policy["exclude"]
        assert DataDomain.PERFORMANCE_FORECAST in policy["exclude"]
        assert DataDomain.PERFORMANCE_EXPRESS in policy["exclude"]
    assert DataDomain.VALUATION not in PROFILE_DOMAIN_POLICY[SHORT_HORIZON_CORE_PROFILE]["include"]
    assert DataDomain.VALUATION in PROFILE_DOMAIN_POLICY[STYLE_STRUCTURAL_PROFILE]["include"]
    assert DataDomain.VALUATION in PROFILE_DOMAIN_POLICY[STYLE_STRUCTURAL_ALPHA_PROFILE]["include"]
