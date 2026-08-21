"""Margin Eligibility Update: install responsibilities."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from quantlab.core.io import atomic_copy_file
from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    qdp_v2_root,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
)

from .config import (
    END_DATE,
    MARGIN_ELIGIBILITY_CONTRACT_V2,
    START_DATE,
    UPDATE_ID,
)


def _content_id(domain: str, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(_sha256(path).encode("ascii"))
    return f"{domain}__{digest.hexdigest()[:24]}"


@dataclass(frozen=True)
class _InstallPolicy:
    contract: str
    primary_key: list[str]
    start_date: str
    end_date: str
    frequency: str
    source: dict[str, Any]
    notes: list[str]


def _dataset_id(domain: str, paths: Sequence[Path]) -> str:
    content_id = _content_id(domain, paths)
    if domain == DataDomain.MARGIN_ELIGIBILITY:
        return content_id
    provenance = "qdp_v2_tushare_margin_detail_raw_v3_mixed_provenance"
    digest = hashlib.sha256(f"{content_id}|{provenance}".encode()).hexdigest()[:24]
    return f"{domain}__{digest}"


def _install_shards(
    *,
    root: Path,
    target_root: Path,
    paths: Sequence[Path],
) -> list[ShardManifestEntry]:
    entries: list[ShardManifestEntry] = []
    for source_path in paths:
        year_partition = source_path.parent.name
        year = year_partition.split("=", 1)[1]
        target = target_root / "shards" / year_partition / "part-0000.parquet"
        if not target.is_file():
            atomic_copy_file(source_path, target)
        parquet = pq.ParquetFile(target)
        entries.append(
            ShardManifestEntry(
                path=str(target.relative_to(root)).replace("\\", "/"),
                row_count=int(parquet.metadata.num_rows),
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
                file_size=target.stat().st_size,
                metadata={"source_sha256": _sha256(source_path)},
            )
        )
    return entries


def _install_policy(
    domain: str,
    input_manifest: DatasetManifest | None,
) -> _InstallPolicy:
    if domain == DataDomain.MARGIN_ELIGIBILITY:
        return _InstallPolicy(
            contract=MARGIN_ELIGIBILITY_CONTRACT_V2,
            primary_key=["trade_date", "symbol"],
            start_date=START_DATE,
            end_date=END_DATE,
            frequency="1d",
            source={
                "provider": "sse+szse_official_exchange",
                "checked_through": END_DATE,
                "scope": "point_in_time_historical_mainboard",
                "availability_semantics": "next_exchange_open_day",
                "request_2026_count": 0,
                "credential_persisted": False,
            },
            notes=[
                "eligible_observed, known_ineligible, and source_unavailable are distinct",
                "SSE detail presence proves eligibility but detail absence is not an official negative eligibility list",
                "non-eligible securities never receive synthetic zero balances",
                "exchange data from date D is available on the next exchange-open date",
            ],
        )
    if input_manifest is None:
        raise ValueError(f"input manifest is required for domain: {domain}")
    return _InstallPolicy(
        contract="qdp_v2_tushare_margin_detail_raw_v2_exchange_gap_repair",
        primary_key=list(input_manifest.primary_key),
        start_date=input_manifest.start_date,
        end_date=input_manifest.end_date,
        frequency=input_manifest.frequency,
        source={
            **dict(input_manifest.source or {}),
            "provider": "tushare_compatible_primary+sse+szse_official_gap_repair",
            "query_granularity": "trade_date",
            "upstream_dataset_id": input_manifest.dataset_id,
            "exchange_gap_repair_update_id": UPDATE_ID,
            "exchange_gap_repair_row_count": 733,
            "availability_semantics": "next_exchange_open_day",
            "request_2026_count": 0,
            "credential_persisted": False,
        },
        notes=[
            *list(input_manifest.notes or []),
            "official exchange rows missing from QDP are appended without overwriting existing values",
            "cross-source differences are retained in a separate immutable conflict inventory",
        ],
    )


def _install(
    workspace: Path,
    *,
    domain: str,
    paths: Sequence[Path],
    input_manifest: DatasetManifest | None,
) -> tuple[str, dict[str, Any]]:
    root = qdp_v2_root(workspace)
    dataset_id = _dataset_id(domain, paths)
    entries = _install_shards(
        root=root,
        target_root=root / "datasets" / domain / dataset_id,
        paths=paths,
    )
    policy = _install_policy(domain, input_manifest)
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer="raw",
        frequency=policy.frequency,
        contract_version=policy.contract,
        primary_key=policy.primary_key,
        start_date=policy.start_date,
        end_date=policy.end_date,
        row_count=sum(item.row_count for item in entries),
        shards=entries,
        source=policy.source,
        quality={
            "primary_key_unique": True,
            "source_unavailable_is_not_known_ineligible": True,
            "synthetic_zero_balance_rows": 0,
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(paths[0])),
        notes=policy.notes,
    )
    _assert_credential_free(manifest.to_dict())
    write_dataset_manifest(root, manifest)
    return dataset_id, {
        "dataset_id": dataset_id,
        "row_count": manifest.row_count,
        "start_date": policy.start_date,
        "end_date": policy.end_date,
        "shard_count": len(entries),
    }
