"""Repair: common responsibilities."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

_DATE_COLUMNS = ("trade_date", "date", "datetime")


def _date_column(columns: Sequence[str]) -> str:
    return next((item for item in _DATE_COLUMNS if item in columns), "")


def _parquet_date_range(path: Path, column: str) -> tuple[str, str]:
    if not column:
        return "", ""
    values = pq.read_table(path, columns=[column])[column]
    if len(values) == 0:
        return "", ""
    return str(pc.min(values).as_py()), str(pc.max(values).as_py())


def _parquet_list_sql(paths: Sequence[Path]) -> str:
    return "[" + ",".join(_sql_literal(str(item)) for item in paths) + "]"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _time_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
