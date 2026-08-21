"""Repair compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quantlab.data.core.json_io import json_safe, read_json
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    active_manifest_path,
    dataset_manifest_path,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .cli import (
    _parse_replacements,
    build_arg_parser,
    main,
)
from .common import (
    _date_column,
    _parquet_date_range,
    _parquet_list_sql,
    _quote_identifier,
    _sql_literal,
    _time_token,
    _utc_now,
)
from .model import (
    _DATE_COLUMNS,
    ActiveDomain,
    QdpV2RepairError,
    resolve_active_domain,
)
from .mutation import (
    append_active_shard,
    bulk_append_active_shards_from_parquet,
    mutate_active_shards_from_parquet,
    patch_active_cells,
    replace_active_table_from_parquet,
    update_active_manifest_metadata,
)
from .validation import (
    _commit_manifest,
    _common_schema,
    _ensure_unique_paths,
    _entry_for_parquet,
    _install_prepared_files,
    _new_shard_path,
    _read_patch_request,
    _reference_schema,
    _remove_files,
    _remove_old_shards,
    _require_columns,
    _resolve_active_shard,
    _resolve_prepared_path,
    _updated_manifest,
    _validate_append_keys,
    _validate_parquet,
    _validate_primary_keys,
)
