"""QDP database audit API grouped by check responsibility."""

from quantlab.data.qdp_v2.manifest import EXPECTED_BAR_TIMES

# Compatibility facade for existing audit callers.
# ruff: noqa: F401
from .asof import (
    _normalized_cutoff,
    _snapshot_manifest_as_of,
    audit_database_as_of,
)
from .auxiliary import (
    _auxiliary_semantics_check,
)
from .cli import (
    build_arg_parser,
    main,
)

__all__ = [
    "REQUIRED_DOMAINS",
    "audit_database",
    "audit_database_as_of",
    "audit_latest_keys",
]
from .common import (
    _audit_temp_directory,
    _finding,
    _format,
    _path_texts,
    _q,
)
from .config import (
    AS_OF_DATE_COLUMNS,
    REQUIRED_COLUMNS,
    REQUIRED_DOMAINS,
)
from .identity import (
    _active_manifest,
    _except_count,
    _identity_history_consistency_check,
    _long_suspension_check,
    _paths_for_date,
    _reopen_discontinuity_check,
)
from .latest import (
    audit_latest_keys,
)
from .market import (
    _daily_intraday_consistency_check,
    _factor_semantic_check,
    _latest_5m_stats,
    _status_daily_consistency_check,
    _status_daily_partition_check,
)
from .orchestrator import (
    _audit_dataset,
    audit_database,
)
from .physical import (
    _bar_day_check,
    _factor_check,
    _ohlc_check,
    _ordered_intraday_primary_key_check,
    _primary_key_check,
    _symbol_history_check,
)
