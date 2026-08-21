"""PIT history restore and lifecycle-normalization API."""

# Compatibility facade for existing QDP callers and tests.
# ruff: noqa: F401

from .config import (
    ARCHIVE_RECENT,
    ARCHIVE_RECOVERY,
    ARCHIVE_ROOT,
    CNINFO_SHARE_NORMALIZED_COLUMNS,
    DEFAULT_START_DATE,
    LIFECYCLE_NORMALIZE_DOMAINS,
    MAINBOARD_PREFIXES,
    NAME_INTERVAL_COLUMNS,
    PART_SEMANTIC_VERSION,
    PRE_ARCHIVE_SECURITIES,
    RESTORE_DOMAINS,
    SSE_FACTBOOK_EVIDENCE,
    SSE_ST_TRANSITIONS,
    _factor_rows,
    _historical_names,
    _identity_exchange,
    _is_mainboard,
    _name_implies_st,
    _normalize_cninfo_share_change,
    _normalize_eastmoney_history,
    _normalize_sina_factors,
    _security_id,
    _short_exchange,
)
from .context import (
    PitHistoryContext,
    PitHistoryError,
    _archive_paths,
    _atomic_parquet,
    _context,
    _local_stock_basic,
    _read_active_symbols,
    _read_symbol_history_symbols,
    _stock_basic,
    _valid_parquet,
    _workspace_relative_path,
    inventory_pit_history,
)
from .download import (
    _build_archive_cache,
    _download_market_supplements,
    _download_reference_parts,
    _download_sina_factors,
    _fetch_tencent_daily_metrics,
    _fill_missing_turnover_from_tencent,
    _historical_st_status,
    _load_sse_st_transitions,
    _load_sz_name_events,
    _materialize_history_parts,
    _name_intervals_from_events,
    _trade_calendar,
)
from .lifecycle_audit import (
    _lifecycle_domain_findings,
    _parquet_scan,
    _quoted_identifier,
    _symbol_lifecycle_tables,
    audit_symbol_lifecycle_effectivity,
)
from .lifecycle_factor import (
    _lifecycle_factor_scale_stitch_plan,
    _normalize_lifecycle_factor_scale_stitches,
)
from .lifecycle_prepare import (
    _cleanup_lifecycle_prepared_domain,
    _copy_lifecycle_query,
    _manifest_column_projection,
    _normalized_lifecycle_projection,
    _prepare_lifecycle_domain_mutation,
    _validate_lifecycle_prepared_schemas,
    _validate_lifecycle_source_schemas,
)
from .orchestrator import (
    normalize_symbol_lifecycle_effectivity,
    run_pit_history_restore,
)
from .prepare import (
    _combine_domain_parts,
    _create_composite_dataset,
    _prepare_symbol_parts,
    _validate_prepared,
)
