"""Auxiliary QDP domain updates with stable compatibility exports."""

import time

from .baostock import (
    _baostock_snapshot_columns,
    _baostock_snapshot_part_name,
    _baostock_snapshot_worker,
    _fetch_baostock_snapshots,
    _from_baostock_code,
    _provider_date,
    _valid_parquet_columns,
)

# Compatibility facade for existing QDP callers.
# ruff: noqa: F401
from .context import (
    AUXILIARY_DOMAINS,
    BAOSTOCK_WORKERS,
    DAILY_AUXILIARY_DOMAINS,
    FREE_SOURCE_POLICY,
    INDEX_SPECS,
    LEGACY_SOURCE_POLICY,
    SECONDARY_VALIDATION_SAMPLE_SIZE,
    SECONDARY_VALIDATION_SEED,
    SECONDARY_VALIDATION_WORKERS,
    TUSHARE_MAX_RPM,
    TUSHARE_WORKERS,
    AuxiliaryContext,
    AuxiliaryUpdateError,
    TushareRateLimitError,
    _context,
    _copy_query,
    _date_text,
    _external_with_retry,
    _identity_dependent_repair_required,
    _identity_scope_signature,
    _manifest,
    _MinuteLimiter,
    _open_dates,
    _parquet_schema_columns,
    _paths,
    _replace_domain,
    _resolve_tushare_api_url,
    _resolve_tushare_token,
    _runtime_state_path,
    _scan_sql,
    _snapshot_dates,
    _split_evenly,
    _TushareClient,
    _write_state,
    plan_auxiliary_update,
)
from .corporate import (
    CORPORATE_COLUMNS,
    DIVIDEND_FIELDS,
    MOOTDX_CORPORATE_VALIDATION_COLUMNS,
    _current_symbols,
    _fetch_dividend_parts,
    _fetch_mootdx_corporate_validation_parts,
    _mootdx_corporate_validation_worker,
    _normalize_cninfo_dividend,
    _normalize_dividend,
    repair_corporate_actions,
    validate_corporate_actions_secondary,
)
from .index import repair_index_constituents
from .industry import (
    CNINFO_INDUSTRY_COLUMNS,
    _fetch_cninfo_industry_parts,
    _industry_cninfo_symbols,
    _industry_query_dates,
    _normalize_cninfo_industry_history,
    repair_industry,
)
from .names import (
    NAME_CHANGE_FIELDS,
    NAME_INTERVAL_COLUMNS,
    _fetch_name_change_parts,
    _normalize_name_intervals,
    repair_name_change,
)
from .orchestrator import (
    _validate_auxiliary_domains,
    build_arg_parser,
    main,
    run_auxiliary_repair,
    run_auxiliary_update,
    run_auxiliary_validation,
)
from .secondary import (
    _normalize_comparison_text,
    _normalize_industry_comparison,
    validate_index_secondary,
    validate_industry_secondary,
    validate_share_capital_secondary,
)
from .shares import (
    CNINFO_A_SHARE_SOURCE,
    CNINFO_SHARE_NORMALIZED_COLUMNS,
    DAILY_BASIC_FIELDS,
    DAILY_BASIC_NORMALIZED_COLUMNS,
    LEGACY_CNINFO_SHARE_SOURCE,
    _fetch_cninfo_share_fallback_parts,
    _fetch_daily_basic_parts,
    _legacy_cninfo_share_symbols,
    _normalize_cninfo_share_change,
    _normalize_daily_basic,
    _repair_legacy_cninfo_share_rows,
    _share_repair_dates,
    _tushare_date_series,
    repair_share_capital,
)
from .valuation import (
    _repair_valuation_market_caps,
    _valuation_market_cap_error_count,
    _valuation_repair_dates,
    _valuation_secondary_policy,
    repair_valuation,
)

__all__ = [
    "AUXILIARY_DOMAINS",
    "DAILY_AUXILIARY_DOMAINS",
    "plan_auxiliary_update",
    "repair_corporate_actions",
    "repair_index_constituents",
    "repair_industry",
    "repair_name_change",
    "repair_share_capital",
    "repair_valuation",
    "run_auxiliary_repair",
    "run_auxiliary_update",
    "run_auxiliary_validation",
    "validate_corporate_actions_secondary",
    "validate_index_secondary",
    "validate_industry_secondary",
    "validate_share_capital_secondary",
]
