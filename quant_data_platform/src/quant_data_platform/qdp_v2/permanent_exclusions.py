from __future__ import annotations

"""Permanent one-way ST and delisting-period exclusions for the mutable QDP store."""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    utc_now,
    write_active_manifest,
)
from quant_data_platform.qdp_v2.repair import (
    mutate_active_shards_from_parquet,
    resolve_active_domain,
)


REGISTRY_SCHEMA_VERSION = 1
POLICY_ID = "permanent_st_delisting_exclusion_v1"
REGISTRY_FILENAME = "permanent_exclusions.json"
INITIAL_AS_OF_DATE = "2026-07-16"
INITIAL_EXPECTED_COUNT = 151

SYMBOL_DOMAINS = (
    "adjust_factor",
    "corporate_actions",
    "index_constituents",
    "industry_concept",
    "market_daily_raw",
    "market_intraday_5m",
    "name_change",
    "security_status",
    "share_capital",
    "universe_snapshot",
    "valuation",
)
IDENTITY_DOMAINS = ("security_identity", "symbol_history")
PURGE_DOMAINS = (*SYMBOL_DOMAINS, *IDENTITY_DOMAINS)


class PermanentExclusionError(RuntimeError):
    pass


def registry_path(*, workspace_root: str | Path | None = None) -> Path:
    return (qdp_v2_root(workspace_root) / REGISTRY_FILENAME).resolve()


def empty_registry() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "policy": "permanent_one_way_never_restore",
        "trigger_rules": ["is_st", "name_ends_with_退", "is_delisted"],
        "exclusion_count": 0,
        "updated_at": utc_now(),
        "exclusions": [],
    }


def load_registry(
    *, workspace_root: str | Path | None = None, required: bool = False
) -> dict[str, Any]:
    path = registry_path(workspace_root=workspace_root)
    if not path.exists():
        if required:
            raise PermanentExclusionError(f"permanent_exclusion_registry_missing:{path}")
        return empty_registry()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PermanentExclusionError("permanent_exclusion_registry_not_object")
    if int(payload.get("schema_version", 0) or 0) != REGISTRY_SCHEMA_VERSION:
        raise PermanentExclusionError("permanent_exclusion_registry_version_invalid")
    if str(payload.get("policy_id", "") or "") != POLICY_ID:
        raise PermanentExclusionError("permanent_exclusion_policy_invalid")
    entries = normalize_entries(payload.get("exclusions", []))
    if int(payload.get("exclusion_count", -1)) != len(entries):
        raise PermanentExclusionError("permanent_exclusion_registry_count_mismatch")
    payload["exclusions"] = entries
    return payload


def normalize_entries(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    symbol_owner: dict[str, str] = {}
    for raw in values:
        security_id = str(raw.get("security_id", "") or "").strip()
        current_symbol = _symbol(raw.get("current_symbol", ""))
        symbols = sorted(
            {
                _symbol(item)
                for item in list(raw.get("symbols", []) or [])
                if _symbol(item)
            }
            | ({current_symbol} if current_symbol else set())
        )
        exclusion_date = _date_text(raw.get("exclusion_date", ""))
        reason = str(raw.get("reason", "") or "").strip()
        if not security_id or not current_symbol or not symbols or not exclusion_date:
            raise PermanentExclusionError(
                f"permanent_exclusion_entry_incomplete:{security_id or current_symbol}"
            )
        if reason not in {"st", "delisting_period"}:
            raise PermanentExclusionError(
                f"permanent_exclusion_reason_invalid:{security_id}:{reason}"
            )
        if security_id in seen_ids:
            raise PermanentExclusionError(
                f"permanent_exclusion_security_id_duplicate:{security_id}"
            )
        for symbol in symbols:
            owner = symbol_owner.setdefault(symbol, security_id)
            if owner != security_id:
                raise PermanentExclusionError(
                    f"permanent_exclusion_symbol_identity_conflict:{symbol}:{owner}:{security_id}"
                )
        seen_ids.add(security_id)
        entries.append(
            {
                "security_id": security_id,
                "current_symbol": current_symbol,
                "symbols": symbols,
                "exclusion_date": exclusion_date,
                "trigger_name": str(raw.get("trigger_name", "") or "").strip(),
                "reason": reason,
            }
        )
    return sorted(entries, key=lambda item: item["security_id"])


def registry_sets(
    registry: Mapping[str, Any] | None = None,
    *,
    workspace_root: str | Path | None = None,
) -> tuple[set[str], set[str]]:
    payload = dict(registry or load_registry(workspace_root=workspace_root))
    entries = normalize_entries(payload.get("exclusions", []))
    security_ids = {str(item["security_id"]) for item in entries}
    symbols = {
        str(symbol)
        for item in entries
        for symbol in list(item.get("symbols", []) or [])
    }
    return security_ids, symbols


def filter_excluded_candidates(
    frame: pd.DataFrame,
    *,
    domain: str,
    workspace_root: str | Path | None = None,
    registry: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty or str(domain) == "trading_calendar":
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    security_ids, symbols = registry_sets(
        registry, workspace_root=workspace_root
    )
    payload = dict(registry or load_registry(workspace_root=workspace_root))
    excluded_names = {
        _normalized_issuer_name(item.get("trigger_name", ""))
        for item in list(payload.get("exclusions", []) or [])
        if _normalized_issuer_name(item.get("trigger_name", ""))
    }
    if not security_ids and not symbols:
        return frame.copy().reset_index(drop=True)
    data = frame.copy()
    excluded = pd.Series(False, index=data.index)
    if "security_id" in data:
        excluded |= data["security_id"].astype(str).isin(security_ids)
    for column in ("symbol", "current_symbol"):
        if column in data:
            excluded |= data[column].astype(str).str.upper().isin(symbols)
    for column in ("name", "issuer_name"):
        if column in data and excluded_names:
            excluded |= data[column].map(_normalized_issuer_name).isin(excluded_names)
    return data.loc[~excluded].reset_index(drop=True)


def discover_current_exclusions(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    as_of = _date_text(as_of_date)
    status = resolve_active_domain("security_status", workspace_root=workspace)
    universe = resolve_active_domain("universe_snapshot", workspace_root=workspace)
    identity = resolve_active_domain("security_identity", workspace_root=workspace)
    history = resolve_active_domain("symbol_history", workspace_root=workspace)
    runtime = _runtime_root(workspace)
    with open_guarded_duckdb(
        temp_directory=runtime / "discover_spill", threads=2
    ) as con:
        latest = con.execute(
            """
            WITH latest_status AS (
              SELECT cast(symbol AS VARCHAR) AS symbol,
                     cast(trade_date AS VARCHAR) AS trade_date,
                     coalesce(try_cast(is_st AS BOOLEAN), false) AS is_st,
                     coalesce(try_cast(is_delisted AS BOOLEAN), false) AS is_delisted,
                     row_number() OVER (
                       PARTITION BY cast(symbol AS VARCHAR)
                       ORDER BY cast(trade_date AS VARCHAR) DESC
                     ) AS rn
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR) <= ?
            ), latest_universe AS (
              SELECT cast(symbol AS VARCHAR) AS symbol,
                     cast(name AS VARCHAR) AS name,
                     row_number() OVER (
                       PARTITION BY cast(symbol AS VARCHAR)
                       ORDER BY cast(trade_date AS VARCHAR) DESC
                     ) AS rn
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR) <= ?
            )
            SELECT s.symbol,s.trade_date,s.is_st,s.is_delisted,coalesce(u.name,'') AS name
            FROM latest_status s
            LEFT JOIN latest_universe u ON u.symbol=s.symbol AND u.rn=1
            WHERE s.rn=1
              AND (s.is_st OR s.is_delisted OR ends_with(trim(coalesce(u.name,'')), '退'))
            ORDER BY s.symbol
            """,
            [
                [str(item) for item in status.shard_paths],
                as_of,
                [str(item) for item in universe.shard_paths],
                as_of,
            ],
        ).fetchdf()
        identities = con.execute(
            """
            SELECT cast(security_id AS VARCHAR) AS security_id,
                   upper(trim(cast(current_symbol AS VARCHAR))) AS current_symbol
            FROM read_parquet(?, union_by_name=true)
            """,
            [[str(item) for item in identity.shard_paths]],
        ).fetchdf()
        histories = con.execute(
            """
            SELECT cast(security_id AS VARCHAR) AS security_id,
                   upper(trim(cast(symbol AS VARCHAR))) AS symbol
            FROM read_parquet(?, union_by_name=true)
            """,
            [[str(item) for item in history.shard_paths]],
        ).fetchdf()
    return _entries_from_current_frames(
        latest=latest,
        identities=identities,
        histories=histories,
        exclusion_date=as_of,
    )


def _entries_from_current_frames(
    *,
    latest: pd.DataFrame,
    identities: pd.DataFrame,
    histories: pd.DataFrame,
    exclusion_date: str,
) -> list[dict[str, Any]]:
    if latest.empty:
        return []
    identity_rows = identities.copy()
    identity_rows["current_symbol"] = identity_rows["current_symbol"].map(_symbol)
    history_rows = histories.copy()
    history_rows["symbol"] = history_rows["symbol"].map(_symbol)
    current_to_id = dict(
        identity_rows.loc[:, ["current_symbol", "security_id"]].itertuples(
            index=False, name=None
        )
    )
    symbol_to_id = dict(
        history_rows.loc[:, ["symbol", "security_id"]].itertuples(
            index=False, name=None
        )
    )
    symbols_by_id = (
        history_rows.groupby("security_id", sort=False)["symbol"]
        .agg(lambda values: sorted(set(str(item) for item in values if str(item))))
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for item in latest.to_dict("records"):
        current_symbol = _symbol(item.get("symbol", ""))
        security_id = str(
            current_to_id.get(current_symbol)
            or symbol_to_id.get(current_symbol)
            or ""
        )
        if not security_id:
            raise PermanentExclusionError(
                f"permanent_exclusion_identity_missing:{current_symbol}"
            )
        is_st = _bool(item.get("is_st", False))
        reason = "st" if is_st else "delisting_period"
        symbols = set(symbols_by_id.get(security_id, []))
        symbols.add(current_symbol)
        rows.append(
            {
                "security_id": security_id,
                "current_symbol": current_symbol,
                "symbols": sorted(symbols),
                "exclusion_date": _date_text(exclusion_date),
                "trigger_name": str(item.get("name", "") or "").strip(),
                "reason": reason,
            }
        )
    return normalize_entries(rows)


def merge_registry_entries(
    registry: Mapping[str, Any],
    discovered: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    existing = normalize_entries(registry.get("exclusions", []))
    additions = normalize_entries(discovered)
    by_id = {str(item["security_id"]): dict(item) for item in existing}
    added: list[dict[str, Any]] = []
    for item in additions:
        security_id = str(item["security_id"])
        prior = by_id.get(security_id)
        if prior is None:
            by_id[security_id] = dict(item)
            added.append(dict(item))
            continue
        prior_symbols = set(prior.get("symbols", []))
        prior_symbols.update(item.get("symbols", []))
        prior["symbols"] = sorted(prior_symbols)
        # The original trigger and date are immutable. New symbols are retained
        # so a later code change cannot reintroduce the same security.
        by_id[security_id] = prior
    merged = dict(registry)
    merged.setdefault("schema_version", REGISTRY_SCHEMA_VERSION)
    merged.setdefault("policy_id", POLICY_ID)
    merged.setdefault("policy", "permanent_one_way_never_restore")
    merged.setdefault(
        "trigger_rules", ["is_st", "name_ends_with_退", "is_delisted"]
    )
    merged["exclusions"] = normalize_entries(by_id.values())
    merged["exclusion_count"] = len(merged["exclusions"])
    merged["updated_at"] = utc_now()
    return merged, added


def register_current_exclusions(
    *,
    as_of_date: str,
    workspace_root: str | Path | None = None,
    expected_count: int = 0,
) -> dict[str, Any]:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    current = load_registry(workspace_root=workspace)
    discovered = discover_current_exclusions(
        as_of_date=as_of_date, workspace_root=workspace
    )
    merged, additions = merge_registry_entries(current, discovered)
    if int(expected_count) > 0 and int(merged["exclusion_count"]) != int(
        expected_count
    ):
        raise PermanentExclusionError(
            "permanent_exclusion_expected_count_mismatch:"
            f"expected={int(expected_count)}:actual={merged['exclusion_count']}"
        )
    path = registry_path(workspace_root=workspace)
    atomic_write_json(path, merged)
    return {
        "status": "registered" if additions else "unchanged",
        "registry_path": str(path),
        "exclusion_count": int(merged["exclusion_count"]),
        "new_exclusion_count": len(additions),
        "new_exclusions": additions,
    }


def purge_permanent_exclusions(
    *,
    workspace_root: str | Path | None = None,
    domains: Sequence[str] = PURGE_DOMAINS,
) -> dict[str, Any]:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    registry = load_registry(workspace_root=workspace, required=True)
    security_ids, symbols = registry_sets(registry)
    if not security_ids or not symbols:
        raise PermanentExclusionError("permanent_exclusion_registry_empty")
    active = read_active_manifest(qdp_v2_root(workspace))
    active_domains = _active_dataset_map(active)
    requested = [str(item) for item in domains if str(item) in active_domains]
    unknown = sorted(set(domains).difference(active_domains))
    if unknown:
        raise PermanentExclusionError(
            f"permanent_exclusion_active_domains_missing:{','.join(unknown)}"
        )
    runtime = _runtime_root(workspace)
    prepared = runtime / "prepared.parquet"
    progress = runtime / "progress.json"
    results: dict[str, Any] = {}
    total_removed = 0
    completed = False
    exclusion_rows = _exclusion_rows(registry)
    try:
        for domain_index, domain in enumerate(requested):
            context = resolve_active_domain(domain, workspace_root=workspace)
            initial_paths = list(context.shard_paths)
            domain_removed = 0
            domain_mutations: list[dict[str, Any]] = []
            for shard_index, old_path in enumerate(initial_paths):
                atomic_write_json(
                    progress,
                    {
                        "status": "filtering",
                        "domain": domain,
                        "domain_index": domain_index + 1,
                        "domain_count": len(requested),
                        "shard_index": shard_index + 1,
                        "shard_count": len(initial_paths),
                        "old_shard": str(old_path),
                        "updated_at": utc_now(),
                    },
                )
                prepared.unlink(missing_ok=True)
                removed, kept = _prepare_filtered_shard(
                    domain=domain,
                    source=old_path,
                    target=prepared,
                    exclusion_rows=exclusion_rows,
                    workspace=workspace,
                )
                if removed == 0:
                    prepared.unlink(missing_ok=True)
                    continue
                if kept <= 0:
                    mutation = mutate_active_shards_from_parquet(
                        domain,
                        removals=[old_path],
                        reason=f"apply {POLICY_ID}",
                        workspace_root=workspace,
                    )
                else:
                    mutation = mutate_active_shards_from_parquet(
                        domain,
                        replacements=[(old_path, prepared)],
                        reason=f"apply {POLICY_ID}",
                        workspace_root=workspace,
                    )
                prepared.unlink(missing_ok=True)
                domain_removed += int(removed)
                total_removed += int(removed)
                domain_mutations.append(
                    {
                        "removed_rows": int(removed),
                        "kept_rows": int(kept),
                        "mutation_id": str(mutation.get("mutation_id", "")),
                    }
                )
            results[domain] = {
                "removed_rows": domain_removed,
                "mutation_count": len(domain_mutations),
                "mutations": domain_mutations,
            }
        residual = audit_exclusion_residuals(
            workspace_root=workspace, registry=registry
        )
        if int(residual["total_residual_rows"]) != 0:
            raise PermanentExclusionError(
                "permanent_exclusion_residual_rows:"
                f"{residual['total_residual_rows']}"
            )
        _mark_manifests_and_scope(workspace=workspace, registry=registry)
        atomic_write_json(
            progress,
            {
                "status": "completed",
                "exclusion_count": int(registry["exclusion_count"]),
                "removed_rows": total_removed,
                "completed_at": utc_now(),
            },
        )
        result = {
            "status": "completed",
            "policy_id": POLICY_ID,
            "exclusion_count": int(registry["exclusion_count"]),
            "removed_rows": total_removed,
            "domains": results,
            "residuals": residual,
        }
        completed = True
        return result
    finally:
        prepared.unlink(missing_ok=True)
        if completed:
            progress.unlink(missing_ok=True)
            for child in runtime.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
            if runtime.exists() and not any(runtime.iterdir()):
                runtime.rmdir()


def audit_exclusion_residuals(
    *,
    workspace_root: str | Path | None = None,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    payload = dict(registry or load_registry(workspace_root=workspace, required=True))
    exclusion_rows = _exclusion_rows(payload)
    active = _active_dataset_map(read_active_manifest(qdp_v2_root(workspace)))
    counts: dict[str, int] = {}
    runtime = _runtime_root(workspace)
    with open_guarded_duckdb(
        temp_directory=runtime / "residual_spill", threads=2
    ) as con:
        con.register("qdp_permanent_exclusions", exclusion_rows)
        for domain in PURGE_DOMAINS:
            if domain not in active:
                continue
            context = resolve_active_domain(domain, workspace_root=workspace)
            predicate = _exclusion_predicate(domain, alias="d")
            count = con.execute(
                f"SELECT count(*) FROM read_parquet(?, union_by_name=true) d WHERE {predicate}",
                [[str(item) for item in context.shard_paths]],
            ).fetchone()[0]
            counts[domain] = int(count or 0)
    return {
        "status": "ok" if not sum(counts.values()) else "error",
        "residual_rows": counts,
        "total_residual_rows": sum(counts.values()),
    }


def registry_consistency(
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        entries = normalize_entries(registry.get("exclusions", []))
    except PermanentExclusionError as exc:
        entries = []
        errors.append(str(exc))
    expected = int(registry.get("exclusion_count", -1) or 0)
    if expected != len(entries):
        errors.append(
            f"exclusion_count_mismatch:expected={expected}:actual={len(entries)}"
        )
    return {
        "status": "error" if errors else "ok",
        "exclusion_count": len(entries),
        "errors": errors,
    }


def _prepare_filtered_shard(
    *,
    domain: str,
    source: Path,
    target: Path,
    exclusion_rows: pd.DataFrame,
    workspace: Path,
) -> tuple[int, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    predicate = _exclusion_predicate(domain, alias="d")
    order_by = _order_by(domain)
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace) / "rewrite_spill", threads=4
    ) as con:
        con.register("qdp_permanent_exclusions", exclusion_rows)
        row = con.execute(
            f"SELECT count(*) AS total, count(*) FILTER (WHERE {predicate}) AS removed "
            "FROM read_parquet(?, union_by_name=false) d",
            [str(source)],
        ).fetchone()
        total, removed = int(row[0] or 0), int(row[1] or 0)
        kept = total - removed
        if removed <= 0 or kept <= 0:
            return removed, kept
        source_sql = _sql_literal(source)
        temp_sql = _sql_literal(temporary)
        con.execute(
            f"COPY (SELECT d.* FROM read_parquet({source_sql}, union_by_name=false) d "
            f"WHERE NOT ({predicate}) ORDER BY {order_by}) TO {temp_sql} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)"
        )
    os.replace(temporary, target)
    return removed, kept


def _exclusion_predicate(domain: str, *, alias: str) -> str:
    if domain == "security_identity":
        return (
            "EXISTS (SELECT 1 FROM qdp_permanent_exclusions e "
            f"WHERE e.security_id=cast({alias}.security_id AS VARCHAR) "
            f"OR e.symbol=upper(trim(cast({alias}.current_symbol AS VARCHAR))))"
        )
    if domain == "symbol_history":
        return (
            "EXISTS (SELECT 1 FROM qdp_permanent_exclusions e "
            f"WHERE e.security_id=cast({alias}.security_id AS VARCHAR) "
            f"OR e.symbol=upper(trim(cast({alias}.symbol AS VARCHAR))))"
        )
    return (
        "EXISTS (SELECT 1 FROM qdp_permanent_exclusions e "
        f"WHERE e.symbol=upper(trim(cast({alias}.symbol AS VARCHAR))))"
    )


def _order_by(domain: str) -> str:
    if domain == "market_intraday_5m":
        return "symbol,trade_date,bar_time"
    if domain == "security_identity":
        return "security_id"
    if domain == "symbol_history":
        return "security_id,effective_from"
    if domain == "corporate_actions":
        return "symbol,trade_date,action_type,description,source"
    if domain == "name_change":
        return "symbol,trade_date,change_type,source"
    if domain == "index_constituents":
        return "trade_date,index_symbol,symbol"
    return "symbol,trade_date"


def _exclusion_rows(registry: Mapping[str, Any]) -> pd.DataFrame:
    entries = normalize_entries(registry.get("exclusions", []))
    rows = [
        {"security_id": str(item["security_id"]), "symbol": str(symbol)}
        for item in entries
        for symbol in item["symbols"]
    ]
    return pd.DataFrame(rows, columns=["security_id", "symbol"]).drop_duplicates()


def _mark_manifests_and_scope(
    *, workspace: Path, registry: Mapping[str, Any]
) -> None:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    datasets = _active_dataset_map(active)
    registry_ref = path_for_manifest(
        registry_path(workspace_root=workspace), root=root
    )
    for domain in PURGE_DOMAINS:
        dataset_id = datasets.get(domain, "")
        if not dataset_id:
            continue
        path = root / "datasets" / domain / dataset_id / "dataset.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        quality = dict(payload.get("quality", {}) or {})
        quality["permanent_exclusions"] = {
            "policy_id": POLICY_ID,
            "registry": registry_ref,
            "exclusion_count": int(registry["exclusion_count"]),
            "residual_rows": 0,
            "checked_at": utc_now(),
        }
        payload["quality"] = quality
        atomic_write_json(path, payload)

    identity_id = datasets.get("security_identity", "")
    identity_count = 0
    if identity_id:
        identity_manifest = read_dataset_manifest(
            root / "datasets" / "security_identity" / identity_id / "dataset.json"
        )
        identity_count = int(identity_manifest.row_count)
    scope = dict(active.get("scope", {}) or {})
    scope.update(
        {
            "name": "mutable_current_listed_mainboard_permanent_non_st_store",
            "universe": (
                "Current listed Shanghai/Shenzhen main-board A shares excluding "
                "every security permanently registered after an ST or delisting-period trigger."
            ),
            "security_id_count": identity_count,
            "symbol_count": identity_count,
            "permanent_exclusion_policy": POLICY_ID,
            "permanent_exclusion_registry": registry_ref,
            "permanent_exclusion_count": int(registry["exclusion_count"]),
        }
    )
    active["scope"] = scope
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)


def _runtime_root(workspace: Path) -> Path:
    root = (
        workspace
        / "quant_data_platform"
        / "data"
        / "qdp_runtime"
        / "permanent_exclusions"
    ).resolve()
    if workspace != root and workspace not in root.parents:
        raise PermanentExclusionError(f"permanent_exclusion_runtime_outside_workspace:{root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _active_dataset_map(active: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(domain): str(dataset_id)
        for domain, dataset_id in dict(active.get("datasets", {}) or {}).items()
        if str(dataset_id or "").strip()
    }


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalized_issuer_name(value: Any) -> str:
    name = str(value or "").strip().upper().replace(" ", "")
    while name.startswith("*"):
        name = name[1:]
    if name.startswith("ST"):
        name = name[2:]
    return name[:-1] if name.endswith("退") else name


def _date_text(value: Any) -> str:
    parsed = pd.Timestamp(str(value or "")).normalize()
    if pd.isna(parsed):
        raise PermanentExclusionError(f"permanent_exclusion_date_invalid:{value}")
    return parsed.strftime("%Y-%m-%d")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdp exclude",
        description="Register and permanently purge ST/delisting-period securities.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--as-of-date", default=INITIAL_AS_OF_DATE)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if not args.apply:
        discovered = discover_current_exclusions(
            as_of_date=str(args.as_of_date), workspace_root=workspace
        )
        payload: dict[str, Any] = {
            "status": "planned",
            "as_of_date": str(args.as_of_date),
            "discovered_count": len(discovered),
            "security_ids": [item["security_id"] for item in discovered],
        }
    else:
        registration = register_current_exclusions(
            as_of_date=str(args.as_of_date),
            workspace_root=workspace,
            expected_count=int(args.expected_count),
        )
        purge = purge_permanent_exclusions(workspace_root=workspace)
        payload = {
            "status": "completed",
            "registration": registration,
            "purge": purge,
        }
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0 if payload.get("status") in {"planned", "completed"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
