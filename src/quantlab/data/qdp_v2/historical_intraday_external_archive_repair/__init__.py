"""Historical Intraday External Archive Repair compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import (
    EXPECTED_BAR_TIMES,
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.recent_market_repair import (
    INTRADAY_COLUMNS,
    INTRADAY_DOMAIN,
    _write_final_parquet,
)
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    _DECISION_COLUMNS,
    _MEMBER_PATTERN,
    _NUMERIC_COLUMNS,
    _PRICE_COLUMNS,
    AMOUNT_RELATIVE_TOLERANCE,
    BASELINE_ID,
    DEFAULT_ARCHIVE,
    DEFAULT_WORKERS,
    END_DATE,
    EXPECTED_ACCEPTED_DAYS,
    EXPECTED_CANDIDATE_DAYS,
    EXPECTED_CANDIDATE_SYMBOLS,
    EXPECTED_FORMAL_QUALITY_ACCEPTED_DAYS,
    EXPECTED_FORMAL_QUALITY_POOL_ROWS,
    EXPECTED_FORMAL_QUALITY_REMAINING_DAYS,
    EXPECTED_RECENT_QUALITY_REMAINING_DAYS,
    PRICE_RELATIVE_TOLERANCE,
    REPAIR_ID,
    SOURCE_NAME,
    START_DATE,
    VOLUME_RELATIVE_TOLERANCE,
    ArchiveMember,
    ExternalArchiveRepairError,
    SymbolResult,
)
from .context import (
    _baseline_manifest,
    _read_state,
    _runtime,
    _state_path,
    _workspace,
    _write_state,
)
from .install import (
    _dataset_id,
    _entry_for_installed,
    _install_immutable_version,
)
from .inventory import (
    _active_intraday,
    _gap_inventory,
    _scan_archive,
    _symbol_from_member,
)
from .normalize import (
    _normalize_day,
    _positive_finite,
    _relative_error,
    _validate_daily,
)
from .prepare import (
    _assemble_decisions,
    _bundle_accepted,
    _initialize,
    _no_file_decisions,
    _sql_paths,
    _validate_prepared,
)
from .process import (
    _atomic_frame_parquet,
    _load_reusable_result,
    _process_symbol,
    _read_candidate_rows,
    _symbol_paths,
)
from .workflow import (
    build_arg_parser,
    evaluate,
    main,
    run_pending,
)
