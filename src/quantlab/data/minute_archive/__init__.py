"""Canonical local minute-archive ingestion."""

from quantlab.data.minute_archive.contracts import (
    CORE_COLUMNS,
    OUTPUT_COLUMNS,
    ArchiveMember,
    MinuteArchiveError,
)
from quantlab.data.minute_archive.importer import import_year
from quantlab.data.minute_archive.reader import (
    extract_to_parquet,
    list_members,
    normalize_member_name,
    standardize_frame,
)

__all__ = [
    "ArchiveMember",
    "CORE_COLUMNS",
    "MinuteArchiveError",
    "OUTPUT_COLUMNS",
    "extract_to_parquet",
    "import_year",
    "list_members",
    "normalize_member_name",
    "standardize_frame",
]
