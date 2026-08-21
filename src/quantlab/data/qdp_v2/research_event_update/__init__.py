"""Research Event Update compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
import unicodedata
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import datetime
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
from quantlab.data.providers import _fetch_cninfo_announcements
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
from quantlab.data.qdp_v2.provider_credentials import tushare_credential_values
from quantlab.data.qdp_v2.status import active_dataset_map

from .announcement import (
    _CATEGORY_KEYWORDS,
    _announcement_category,
    _announcement_raw_source_paths,
    _cninfo_announcement_path,
    _eastmoney_announcement_path,
    _fetch_cninfo_symbol,
    _fetch_eastmoney_announcements,
    _normalize_cninfo_announcements,
    _normalize_eastmoney_announcements,
    download_announcements,
    prepare_announcements,
)
from .config import (
    ANNOUNCEMENT_COLUMNS,
    END_DATE,
    FORECAST_COLUMNS,
    LEGACY_UPDATE_ID,
    MAX_WORKERS,
    REPORT_COLUMNS,
    REPORT_EASTMONEY_START,
    REPORT_RC_EMPTY_CONFIRMATIONS,
    REPORT_RC_FIELDS,
    REPORT_RC_PAGE_SIZE,
    START_DATE,
    UPDATE_ID,
    ResearchEventUpdateError,
)
from .context import (
    _active_paths,
    _assert_credential_free,
    _cached_report_day,
    _credential_values,
    _first,
    _frame_schema_hash,
    _identity_symbols,
    _json_list,
    _legacy_runtime,
    _next_open_date,
    _next_report_offset,
    _normalized_institution,
    _normalized_text,
    _now,
    _numeric,
    _open_dates,
    _parquet_safe_provider_frame,
    _read_state,
    _report_id,
    _report_rc_page_path,
    _report_request_dates,
    _report_source_key,
    _report_year_source_coverage,
    _request_json,
    _runtime,
    _sha256,
    _stable_hash,
    _state_path,
    _text,
    _workspace,
    _write_parquet,
    _write_state,
)
from .install import (
    _install_domain,
    commit_prepared,
)
from .report_download import (
    _download_symbol_files,
    _eastmoney_report_output_path,
    _eastmoney_report_path,
    _eastmoney_report_paths,
    _fetch_eastmoney_reports,
    download_eastmoney_reports,
    download_tushare_reports,
)
from .report_prepare import (
    _merge_reports,
    _normalize_eastmoney_reports,
    _normalize_tushare_report_year,
    prepare_reports,
)
from .workflow import (
    build_arg_parser,
    evaluate,
    main,
    run_pending,
    self_test,
    status,
)
