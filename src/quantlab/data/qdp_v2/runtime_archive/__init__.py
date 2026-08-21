"""Runtime Archive compatibility API."""

# Preserve the former module namespace while implementations live in focused modules.
# ruff: noqa: F401

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quantlab.data.core.json_io import json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import atomic_write_json, utc_now
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
)

from .archive import (
    _create_tar_zst,
    _safe_member_name,
    restore_archive,
    verify_archive,
)
from .cleanup import (
    _delete_exact_sources,
    _remove_empty_parents,
)
from .cli import (
    build_arg_parser,
    main,
)
from .config import (
    ARCHIVE_VERSION,
    DEFAULT_CLUSTER_BYTES,
    DEFAULT_SELECTIONS,
    LEDGER_COLUMNS,
    SENSITIVE_KEYS,
    YEAR_PATTERNS,
    ArchiveUnit,
    RuntimeArchiveError,
)
from .discovery import (
    _archive_root,
    _domain_for_file,
    _runtime_root,
    _safe_name,
    _unit_directory,
    _within,
    _workspace,
    _year_for_file,
    _zstd_executable,
    discover_units,
)
from .ledger import (
    _atomic_parquet,
    _json_metadata,
    _ledger_record,
    _redact,
    _state_snapshot,
    build_ledger,
)
from .seal import (
    _sample_restore,
    seal_completed_workflow,
    seal_unit,
    seal_workflows,
    verify_unit_manifest,
)
