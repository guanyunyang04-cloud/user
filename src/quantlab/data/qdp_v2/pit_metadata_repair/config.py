"""Pit Metadata Repair: config responsibilities."""

from __future__ import annotations

REPAIR_ID = "pit_metadata_repair_v1"


SUPPORTED_DOMAINS = (
    "universe-list-date",
    "universe-name-pit",
    "industry-unknown",
)


class PitMetadataRepairError(RuntimeError):
    pass
