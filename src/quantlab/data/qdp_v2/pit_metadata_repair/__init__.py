"""Pit Metadata Repair compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.auxiliary_update import (
    _fetch_cninfo_industry_parts,
)
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quantlab.data.qdp_v2.repair import (
    _sql_literal,
    mutate_active_shards_from_parquet,
    update_active_manifest_metadata,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    REPAIR_ID,
    SUPPORTED_DOMAINS,
    PitMetadataRepairError,
)
from .context import (
    _active_paths,
    _assert_untargeted_columns_equal,
    _parquet_count,
    _root,
    _runtime,
    _safe_name,
    _schema_columns,
    _sha256,
    _sql_identifier,
    _workspace,
)
from .identity import (
    _build_identity_map,
    _prepare_list_date,
)
from .industry import (
    _prepare_industry,
    _unknown_symbols,
)
from .names import (
    _name_event_contract,
    _prepare_universe_name_pit,
    _resolved_universe_name_cte,
)
from .orchestrator import (
    main,
    run_repair,
)

__all__ = ["SUPPORTED_DOMAINS", "PitMetadataRepairError", "run_repair"]
