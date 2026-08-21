"""Margin Eligibility Update compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import time
import warnings
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts import DataDomain
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
    _sha256,
    _write_parquet,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    END_DATE,
    ENDPOINTS,
    MARGIN_DETAIL_COLUMNS,
    MARGIN_ELIGIBILITY_COLUMNS,
    MARGIN_ELIGIBILITY_CONTRACT_V1,
    MARGIN_ELIGIBILITY_CONTRACT_V2,
    MAX_WORKERS,
    OFFICIAL_COLUMNS,
    PROVENANCE_REPAIR_ID,
    SSE_SEMANTICS_REPAIR_ID,
    SSE_URL,
    START_DATE,
    SZSE_URL,
    UPDATE_ID,
    USER_AGENT,
    EndpointResult,
    MarginEligibilityUpdateError,
)
from .context import (
    _dataset_paths,
    _empty_official,
    _hash_bytes,
    _number,
    _raw_day_path,
    _read_state,
    _runtime,
    _sql_paths,
    _state_path,
    _trade_dates,
    _workspace,
    _write_state,
)
from .download import (
    download,
)
from .install import (
    _content_id,
    _install,
)
from .prepare import (
    _copy_query,
    _eligibility_query,
    prepare,
)
from .repair import (
    repair_margin_detail_provenance,
    repair_sse_eligibility_semantics,
)
from .sources import (
    _fetch_day,
    _fetch_endpoint,
    _fetch_sse_detail,
    _fetch_szse_excel,
    _official_frame,
)
from .workflow import (
    build_arg_parser,
    commit,
    evaluate,
    main,
    run_pending,
)
