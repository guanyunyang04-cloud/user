"""Auxiliary Tail Update compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.domains.contracts import (
    DataDomain,
    DatePartitionFetchRequest,
)
from quantlab.data.providers import (
    BaostockProvider,
    MootdxOnlineProvider,
    _mootdx_xdxr_domain_frame,
)
from quantlab.data.qdp_v2 import normalization as _normalization
from quantlab.data.qdp_v2.auxiliary_update import (
    AUXILIARY_DOMAINS,
    CORPORATE_COLUMNS,
    SECONDARY_VALIDATION_WORKERS,
    AuxiliaryContext,
    _context,
    _current_symbols,
    _external_with_retry,
    _fetch_baostock_snapshots,
    _manifest,
    _paths,
    _scan_sql,
)
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.repair import (
    append_active_shard,
    update_active_manifest_metadata,
)

from .common import (
    _append_if_any,
    _checked_through,
    _mark_checked,
    _missing_daily_keys,
    _only_changed_share_events,
    _unconfirmed_share_detections,
)
from .config import (
    AuxiliaryTailUpdateError,
    _normalize_cninfo_dividend,
    _normalize_cninfo_share_change,
)
from .corporate import (
    update_corporate_actions_tail,
)
from .index import (
    _index_tail_dates,
    update_index_tail,
)
from .industry import (
    update_industry_tail,
)
from .names import (
    update_name_change_tail,
)
from .orchestrator import (
    run_auxiliary_tail_update,
)
from .shares import (
    _fetch_cninfo_share_events,
    _fetch_mootdx_evidence,
    update_share_capital_tail,
)
from .valuation import (
    update_valuation_tail,
)

__all__ = [
    "AuxiliaryTailUpdateError",
    "run_auxiliary_tail_update",
    "update_corporate_actions_tail",
    "update_index_tail",
    "update_industry_tail",
    "update_name_change_tail",
    "update_share_capital_tail",
    "update_valuation_tail",
]
