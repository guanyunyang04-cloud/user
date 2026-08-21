"""Tushare Extended Backfill compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import requests

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.auxiliary_update import (
    _MinuteLimiter,
    _resolve_tushare_token,
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
from quantlab.data.qdp_v2.provider_credentials import (
    ProviderCredentialError,
    resolve_tushare_api_url,
    resolve_tushare_rate_limit,
    tushare_credential_values,
    tushare_provider_status,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .audit import (
    _credential_file_hits,
    _legacy_year_cache_summary,
    _physical_domain_audit,
    audit,
    cleanup_prepared_cache,
)
from .config import (
    BURN_IN_START,
    END_DATE,
    EXPECTED_FACTOR_FIELD_COUNT,
    EXPECTED_FACTOR_SCHEMA_HASH,
    FORBIDDEN_YEAR,
    MARGIN_DETAIL_FIELDS,
    MARGIN_FIELDS,
    MARGIN_SECS_FIELDS,
    MAX_PAGES_PER_TASK,
    MAX_REQUESTS_PER_MINUTE,
    MAX_WORKERS,
    MONEYFLOW_FIELDS,
    PAGE_SIZE,
    PREPARED_TRANSFORM_VERSION,
    RESEARCH_START,
    SPECS,
    UPDATE_ID,
    EndpointSpec,
    ResilientTushareClient,
    TushareExtendedBackfillError,
    _resolve_tushare_api_url,
)
from .context import (
    _active_paths,
    _assert_credential_free,
    _credential_values,
    _open_dates,
    _provider_cache_frame,
    _read_state,
    _runtime,
    _scan,
    _schema_hash,
    _sha256,
    _stable_hash,
    _state_path,
    _validate_schema,
    _workspace,
    _write_parquet,
    _write_state,
)
from .download import (
    _download_inventory,
    _download_task,
    _page_paths,
    _request_fingerprint,
    _selected_specs,
    _success_path,
    _task_dir,
    _task_keys,
    _task_params,
    _valid_cached_page,
    _valid_success,
)
from .install import (
    _install_domain,
    prepare_and_install,
)
from .prepare import (
    _assert_uniform_prepared_schema,
    _copy_query,
    _factor_validation,
    _prepare_domain,
    _prepared_sql,
    _provider_fields,
    _provider_value_expression,
    _quoted,
    _sql_date,
    _validate_prepared,
    _year_page_paths,
)
from .probe import (
    _probe_params,
    probe,
)
from .workflow import (
    build_arg_parser,
    main,
    run_pending,
    self_test,
    status,
)
