"""Runtime Archive: config responsibilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ARCHIVE_VERSION = 1


DEFAULT_CLUSTER_BYTES = 1024 * 1024


YEAR_PATTERNS = (
    re.compile(r"^year=(20(?:0\d|1\d|2[0-6]))(?:$|[^0-9])", re.IGNORECASE),
    re.compile(r"^date=(20(?:0\d|1\d|2[0-6]))[-_]", re.IGNORECASE),
    re.compile(r"^task=(20(?:0\d|1\d|2[0-6]))(?:$|[^0-9])", re.IGNORECASE),
    re.compile(r"__(20(?:0\d|1\d|2[0-6]))\d{2}(?:\D|$)"),
)


SENSITIVE_KEYS = {"token", "api_key", "apikey", "authorization", "password", "secret"}


DEFAULT_SELECTIONS: dict[str, tuple[str, ...]] = {
    "research_report_rc_backfill_v1": ("raw", "normalized"),
    "research_report_rc_backfill_v2": ("raw", "normalized"),
    "historical_intraday_5m_repair_v1": ("parts",),
    "historical_intraday_5m_external_archive_repair_v2": ("symbols",),
    "pit_history_restore": (
        "domain_parts",
        "factor_parts",
        "history_parts",
        "name_interval_parts",
        "share_event_parts",
        "supplement_parts",
    ),
    "margin_eligibility_exchange_history_v1": ("raw",),
}


LEDGER_COLUMNS = (
    "workflow",
    "domain",
    "year",
    "relative_path",
    "file_size",
    "mtime_ns",
    "sha256",
    "suffix",
    "row_count",
    "schema_sha256",
    "request_status",
    "request_offset",
    "request_params_json",
    "response_sha256",
    "error_type",
    "error_message",
    "attempt_count",
    "empty_confirmation_count",
    "completed_at",
)


class RuntimeArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveUnit:
    workflow: str
    domain: str
    year: str
    workflow_root: Path
    files: tuple[Path, ...]

    @property
    def key(self) -> str:
        return f"{self.workflow}/{self.domain}/year={self.year}"
