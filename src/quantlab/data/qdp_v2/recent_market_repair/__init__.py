"""Recent Market Repair compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts import DataDomain, DomainFetchRequest
from quantlab.data.providers import MootdxOnlineProvider
from quantlab.data.qdp_v2.duckdb_resources import (
    DEFAULT_LOW_MEMORY_SECONDS,
    DEFAULT_MEMORY_FLOOR_BYTES,
    DEFAULT_POLL_SECONDS,
    LOW_MEMORY_REASON,
    DuckDbMemoryFloorError,
    open_guarded_duckdb,
)
from quantlab.data.qdp_v2.manifest import (
    EXPECTED_BAR_TIMES,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quantlab.data.qdp_v2.repair import (
    bulk_append_active_shards_from_parquet,
    update_active_manifest_metadata,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .baostock import (
    _active_intraday_quality,
    _baostock_tasks,
    _build_baostock_daily_reference,
    _download_baostock_buckets,
    _fetch_baostock_bucket,
    _fetch_baostock_task,
    _run_baostock_bucket_process,
)
from .cli import (
    _build_parser,
    main,
)
from .commit import (
    _commit_bundles,
    _write_final_parquet,
)
from .config import (
    AMOUNT_UNIT_SCALES,
    BUCKET_COUNT,
    DAILY_AMOUNT_RELATIVE_TOLERANCE,
    DAILY_COLUMNS,
    DAILY_DOMAIN,
    DAILY_PRICE_RELATIVE_TOLERANCE,
    DAILY_SCHEMA,
    DAILY_VOLUME_RELATIVE_TOLERANCE,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_WORKERS,
    INTRADAY_COLUMNS,
    INTRADAY_DOMAIN,
    INTRADAY_SCHEMA,
    LOW_MEMORY_SECONDS,
    MEMORY_FLOOR_BYTES,
    NUMERIC_COLUMNS,
    POLL_SECONDS,
    REPAIR_VERSION,
    VOLUME_UNIT_SCALES,
    RecentMarketMemoryError,
    RecentMarketRepairError,
    _BaostockTask,
    _DomainInput,
    _SystemMemoryGuard,
)
from .entrypoints import (
    run_baostock_intraday_repair,
    run_recent_daily_repair,
    run_recent_intraday_repair,
)
from .inventory import (
    _active_inputs,
    _calendar_year_windows,
    _missing_daily_symbols,
    _missing_intraday_pairs,
    _missing_intraday_pairs_window,
)
from .mootdx import (
    _download_buckets,
    _fetch_bucket,
    _fetch_mootdx_chunk,
)
from .session import (
    _BAOSTOCK_PROCESS_SESSION,
    _baostock_process_session,
    _DirectBaostockSession,
    _normalize_baostock_raw_5m,
)
from .state import (
    _configure_workspace_runtime,
    _date_text,
    _file_sha256,
    _load_or_initialize_state,
    _pairs_sha256,
    _path_texts,
    _requested_pairs,
    _reused_bucket_is_valid,
    _run_directory,
    _runtime_root,
    _split_symbols,
    _stable_bucket,
    _unresolved_mapping,
    _utc_now,
)
from .validation import (
    _best_scale,
    _numeric_validity,
    _validate_daily_frame,
    _validate_intraday_against_daily,
    _validate_intraday_frame,
)
