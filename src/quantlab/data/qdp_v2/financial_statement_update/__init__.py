"""Financial Statement Update compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.auxiliary_update import (
    _resolve_tushare_token,
    _TushareClient,
)
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
    _identity_symbols,
    _next_open_date,
    _open_dates,
    _sha256,
    _write_parquet,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    BALANCE_CONFLICT_REPAIR_ID,
    BALANCE_DERIVED_COLUMNS,
    BALANCE_EXTENSION_CONFLICT_COLUMN,
    BALANCE_EXTENSION_NUMERIC_COLUMNS,
    BALANCE_METRICS,
    BALANCE_SEMANTIC_CONFLICT_COLUMN,
    BALANCE_SEMANTIC_NUMERIC_COLUMNS,
    BALANCE_SEMANTIC_SOURCE_COLUMNS,
    BALANCE_SEMANTIC_STATE_COLUMNS,
    CASH_FLOW_METRICS,
    COMMON_FIELDS,
    COMMON_OUTPUT_COLUMNS,
    END_DATE,
    INCOME_METRICS,
    LEGACY_UPDATE_ID,
    MAX_REPORT_PERIOD,
    PRIMARY_KEY,
    SOURCE_SCHEMA_VERSION,
    START_DATE,
    STATEMENT_SPECS,
    TRAILING_OUTPUT_COLUMNS,
    UPDATE_ID,
    V2_STATEMENT_SPECS,
    FinancialStatementUpdateError,
    StatementSpec,
)
from .context import (
    _in_scope_provider_rows,
    _normalized_path,
    _provider_publish_date,
    _raw_path,
    _read_state,
    _report_periods,
    _runtime,
    _state_path,
    _workspace,
    _write_state,
)
from .download import (
    download,
)
from .install import (
    _install_domain,
    commit,
)
from .normalize import (
    _balance_component_state,
    _hash_rows,
    _normalize_statement_part,
    _populate_balance_semantic_fields,
    _safe_series_ratio,
)
from .prepare import (
    _align_balance_to_active,
    _balance_extension_hash_sql,
    _balance_semantic_hash_sql,
    _dataset_paths,
    _prepare_statement,
    _sql_paths,
    prepare,
)
from .repair import (
    repair_balance_extension_conflict_metadata,
)
from .workflow import (
    _active_domain_record,
    build_arg_parser,
    evaluate,
    main,
    run_pending,
    self_test,
    status,
)
