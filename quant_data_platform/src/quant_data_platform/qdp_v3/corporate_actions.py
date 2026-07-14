from __future__ import annotations

import re
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.providers import MootdxOnlineProvider
from quant_data_platform.qdp_v3.constants import MOOTDX_VERSION, QUALITY_PROVISIONAL, RAW_CORPORATE_ACTION_XDXR
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, normalize_symbol
from quant_data_platform.qdp_v3.manifest import atomic_write_json, sha256_file, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.storage import RawPartitionRef, iter_raw_partitions, read_raw_partition, write_raw_partition


XDXR_EVENT_COLUMNS = [
    "security_id",
    "event_date",
    "symbol_on_date",
    "provider_symbol",
    "category",
    "action_type",
    "cash_dividend_per_10",
    "rights_price",
    "bonus_transfer_per_10",
    "rights_share_per_10",
    "exercise_fraction",
    "exercise_price",
    "identity_mapping_status",
    "verification_status",
    "source",
]

SHARE_CAPITAL_EVENT_COLUMNS = [
    "security_id",
    "event_date",
    "symbol_on_date",
    "provider_symbol",
    "category",
    "change_reason",
    "float_share_before",
    "float_share_after",
    "total_share_before",
    "total_share_after",
    "identity_mapping_status",
    "verification_status",
    "source",
]

OFFICIAL_FACTOR_EVIDENCE_COLUMNS = [
    "security_id",
    "event_date",
    "provider_symbol",
    "official_source",
    "official_document_url",
    "official_document_sha256",
    "fore_adjust_factor",
    "back_adjust_factor",
    "adjust_factor",
    "price_adjustment_applicable",
    "factor_semantics",
    "holder_entitlement_ratio",
    "notes",
]

FACTOR_CANONICAL_COLUMNS = [
    "security_id",
    "divid_operate_date",
    "symbol_on_date",
    "provider_symbol",
    "fore_adjust_factor",
    "back_adjust_factor",
    "adjust_factor",
    "query_date",
    "source_method",
    "verification_status",
    "identity_mapping_status",
    "source",
    "arbitration_source",
    "arbitration_document_hash",
    "arbitration_xdxr_confirmed",
    "price_adjustment_applicable",
    "factor_semantics",
    "holder_entitlement_ratio",
    "semantic_evidence_source",
    "semantic_document_hash",
]

_OFFICIAL_SOURCES = {"cninfo", "sse", "szse", "bse"}
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def default_official_evidence_config(workspace_root: str | Path | None = None) -> Path:
    return qdp_paths(workspace_root).project_root / "configs" / "qdp_v3_corporate_action_evidence.json"


def default_factor_semantic_override_config(workspace_root: str | Path | None = None) -> Path:
    return qdp_paths(workspace_root).project_root / "configs" / "qdp_v3_factor_semantic_overrides.json"


def apply_factor_semantic_overrides(
    events: pd.DataFrame,
    *,
    identity_registry: SecurityIdentityRegistry,
    workspace_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply narrowly-scoped, document-hashed semantic classifications."""

    path = Path(config_path or default_factor_semantic_override_config(workspace_root))
    payload = read_json(path)
    if not payload:
        return events.copy(), pd.DataFrame(), pd.DataFrame()
    if payload.get("schema_version") != "qdp_v3_factor_semantic_overrides_v1":
        raise ValueError(f"factor_semantic_override_schema_mismatch:{path}")
    out = events.copy()
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, item in enumerate(list(payload.get("events", []) or [])):
        if not isinstance(item, dict):
            raise ValueError(f"factor_semantic_override_row_invalid:{index}")
        provider_symbol = normalize_symbol(item.get("provider_symbol", ""))
        security_id = str(item.get("security_id", "") or identity_registry.security_id_for_provider_symbol(provider_symbol))
        event_date = str(item.get("event_date", "") or "")[:10]
        source = str(item.get("official_source", "") or "").lower()
        document_hash = str(item.get("official_document_sha256", "") or "").lower()
        semantics = str(item.get("factor_semantics", "") or "")
        applicable = item.get("price_adjustment_applicable")
        holder_ratio = pd.to_numeric(item.get("holder_entitlement_ratio"), errors="coerce")
        if (
            not security_id
            or len(event_date) != 10
            or source not in _OFFICIAL_SOURCES
            or not _SHA256.fullmatch(document_hash)
            or not semantics
            or not isinstance(applicable, bool)
        ):
            raise ValueError(f"factor_semantic_override_key_invalid:{index}")
        mask = out["security_id"].astype(str).eq(security_id) & out["divid_operate_date"].astype(str).eq(event_date)
        evidence = {
            "security_id": security_id,
            "event_date": event_date,
            "provider_symbol": provider_symbol,
            "price_adjustment_applicable": bool(applicable),
            "factor_semantics": semantics,
            "holder_entitlement_ratio": float(holder_ratio) if np.isfinite(holder_ratio) else np.nan,
            "official_source": source,
            "official_document_url": str(item.get("official_document_url", "") or ""),
            "official_document_sha256": document_hash,
            "notes": str(item.get("notes", "") or ""),
        }
        rows.append(evidence)
        if not mask.any():
            missing.append({**evidence, "conflict_type": "semantic_override_event_missing"})
            continue
        out.loc[mask, "price_adjustment_applicable"] = bool(applicable)
        out.loc[mask, "factor_semantics"] = semantics
        out.loc[mask, "holder_entitlement_ratio"] = evidence["holder_entitlement_ratio"]
        out.loc[mask, "semantic_evidence_source"] = source
        out.loc[mask, "semantic_document_hash"] = document_hash
    return out, pd.DataFrame(rows), pd.DataFrame(missing)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce")


def _event_dates(raw: pd.DataFrame) -> pd.Series:
    required = {"year", "month", "day"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"mootdx_xdxr_date_fields_missing:{missing}")
    return pd.to_datetime(
        {
            "year": _numeric(raw, "year"),
            "month": _numeric(raw, "month"),
            "day": _numeric(raw, "day"),
        },
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")


def _deduplicate_xdxr(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    compare_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["security_id", "event_date", "category"]
    accepted: list[pd.Series] = []
    conflicts: list[dict[str, Any]] = []
    for values, group in frame.groupby(key, dropna=False, sort=False):
        mapped = group.loc[group["identity_mapping_status"].eq("mapped")]
        if mapped.empty:
            conflicts.extend({**row, "conflict_type": "xdxr_identity_unmapped"} for row in group.to_dict("records"))
            continue
        first = mapped.iloc[0]
        equal = True
        for column in compare_columns:
            numbers = pd.to_numeric(mapped[column], errors="coerce")
            non_null = numbers.dropna()
            if not non_null.empty and (non_null.max() - non_null.min()) > 1e-9:
                equal = False
                break
        if equal:
            accepted.append(first)
        else:
            conflicts.extend({**row, "conflict_type": "xdxr_identity_mapped_value_conflict", "key": list(values)} for row in mapped.to_dict("records"))
    out = pd.DataFrame(accepted, columns=columns) if accepted else pd.DataFrame(columns=columns)
    return out.sort_values(["event_date", "security_id", "category"]).reset_index(drop=True), pd.DataFrame(conflicts)


def canonicalize_mootdx_xdxr(
    raw_frames: Iterable[tuple[str, pd.DataFrame]],
    *,
    identity_registry: SecurityIdentityRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split lossless TDX xdxr records into action and share-capital events."""

    action_frames: list[pd.DataFrame] = []
    capital_frames: list[pd.DataFrame] = []
    malformed: list[dict[str, Any]] = []
    for partition_symbol, raw in raw_frames:
        if raw is None or raw.empty:
            continue
        required = {"category"}
        if not required.issubset(raw.columns):
            malformed.append({"provider_symbol": partition_symbol, "conflict_type": "xdxr_required_fields_missing", "missing": sorted(required - set(raw.columns))})
            continue
        work = raw.copy()
        provider = work.get("provider_symbol", pd.Series(normalize_symbol(partition_symbol), index=work.index)).map(normalize_symbol)
        event_date = _event_dates(work)
        category = _numeric(work, "category").astype("Int64")
        base = pd.DataFrame({"event_date": event_date, "provider_symbol": provider, "category": category})
        mapped = identity_registry.map_frame(base, date_column="event_date")
        name = work.get("name", pd.Series(index=work.index, dtype=str)).fillna("").astype(str).str.strip()
        actions = mapped.copy()
        actions["action_type"] = name
        actions["cash_dividend_per_10"] = _numeric(work, "fenhong")
        actions["rights_price"] = _numeric(work, "peigujia")
        actions["bonus_transfer_per_10"] = _numeric(work, "songzhuangu")
        actions["rights_share_per_10"] = _numeric(work, "peigu")
        actions["exercise_fraction"] = _numeric(work, "fenshu")
        actions["exercise_price"] = _numeric(work, "xingquanjia")
        actions["verification_status"] = "xdxr_unofficial_evidence"
        actions["source"] = "mootdx.xdxr"
        action_frames.append(actions.loc[category.eq(1).fillna(False), XDXR_EVENT_COLUMNS])

        capital = mapped.copy()
        capital["change_reason"] = name
        # The TDX protocol reports these four fields in ten-thousand shares.
        capital["float_share_before"] = _numeric(work, "panqianliutong") * 10_000.0
        capital["float_share_after"] = _numeric(work, "panhouliutong") * 10_000.0
        capital["total_share_before"] = _numeric(work, "qianzongguben") * 10_000.0
        capital["total_share_after"] = _numeric(work, "houzongguben") * 10_000.0
        capital["verification_status"] = "xdxr_unofficial_evidence"
        capital["source"] = "mootdx.xdxr"
        has_capital = capital[["float_share_before", "float_share_after", "total_share_before", "total_share_after"]].notna().any(axis=1)
        capital_frames.append(capital.loc[has_capital, SHARE_CAPITAL_EVENT_COLUMNS])

    actions = pd.concat(action_frames, ignore_index=True) if action_frames else pd.DataFrame(columns=XDXR_EVENT_COLUMNS)
    capitals = pd.concat(capital_frames, ignore_index=True) if capital_frames else pd.DataFrame(columns=SHARE_CAPITAL_EVENT_COLUMNS)
    accepted_actions, action_conflicts = _deduplicate_xdxr(
        actions,
        columns=XDXR_EVENT_COLUMNS,
        compare_columns=["cash_dividend_per_10", "rights_price", "bonus_transfer_per_10", "rights_share_per_10"],
    )
    accepted_capitals, capital_conflicts = _deduplicate_xdxr(
        capitals,
        columns=SHARE_CAPITAL_EVENT_COLUMNS,
        compare_columns=["float_share_before", "float_share_after", "total_share_before", "total_share_after"],
    )
    conflict_frames = [item for item in (pd.DataFrame(malformed), action_conflicts, capital_conflicts) if not item.empty]
    conflicts = pd.concat(conflict_frames, ignore_index=True, sort=False) if conflict_frames else pd.DataFrame()
    return accepted_actions, accepted_capitals, conflicts


def canonicalize_proxy_dividends(
    refs: Iterable[RawPartitionRef],
    *,
    identity_registry: SecurityIdentityRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    conflicts: list[pd.DataFrame] = []
    for ref in refs:
        raw = read_raw_partition(ref)
        if raw.empty:
            continue
        required = {"ts_code", "ex_date"}
        if not required.issubset(raw.columns):
            conflicts.append(
                pd.DataFrame(
                    [
                        {
                            "provider_symbol": normalize_symbol(ref.partition_value),
                            "conflict_type": "tushare_proxy_dividend_schema_invalid",
                            "missing": sorted(required - set(raw.columns)),
                        }
                    ]
                )
            )
            continue
        event_date = raw["ex_date"].fillna("").astype(str).str.strip().str.slice(0, 10)
        compact = event_date.str.fullmatch(r"\d{8}")
        event_date.loc[compact] = (
            event_date.loc[compact].str.slice(0, 4)
            + "-"
            + event_date.loc[compact].str.slice(4, 6)
            + "-"
            + event_date.loc[compact].str.slice(6, 8)
        )
        base = pd.DataFrame(
            {
                "event_date": event_date,
                "provider_symbol": raw["ts_code"].map(normalize_symbol),
                "category": 1,
            }
        )
        mapped = identity_registry.map_frame(base, provider_symbol_column="provider_symbol", date_column="event_date")
        mapped["action_type"] = raw.get("div_proc", pd.Series("dividend", index=raw.index)).fillna("dividend").astype(str)
        # Ex-right reference prices use the pre-tax cash entitlement.  Tushare
        # exposes that as cash_div_tax; cash_div is retained as fallback only.
        pretax = _numeric(raw, "cash_div_tax")
        fallback = _numeric(raw, "cash_div")
        mapped["cash_dividend_per_10"] = pretax.where(pretax.notna(), fallback) * 10.0
        mapped["rights_price"] = np.nan
        mapped["bonus_transfer_per_10"] = _numeric(raw, "stk_div") * 10.0
        mapped["rights_share_per_10"] = np.nan
        mapped["exercise_fraction"] = np.nan
        mapped["exercise_price"] = np.nan
        mapped["verification_status"] = "tushare_proxy_dividend_unreconciled"
        mapped["source"] = "tushare_proxy.dividend"
        invalid = mapped["identity_mapping_status"].ne("mapped") | mapped["event_date"].eq("")
        if invalid.any():
            bad = mapped.loc[invalid].copy()
            bad["conflict_type"] = "tushare_proxy_dividend_identity_or_date_invalid"
            conflicts.append(bad)
        frames.append(mapped.loc[~invalid, XDXR_EVENT_COLUMNS])
    actions = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=XDXR_EVENT_COLUMNS)
    if not actions.empty:
        actions = actions.drop_duplicates(
            [
                "security_id",
                "event_date",
                "cash_dividend_per_10",
                "bonus_transfer_per_10",
                "source",
            ],
            keep="last",
        ).sort_values(["event_date", "security_id"], kind="mergesort").reset_index(drop=True)
    conflict_frame = pd.concat(conflicts, ignore_index=True, sort=False) if conflicts else pd.DataFrame()
    return actions, conflict_frame


def corroborate_corporate_actions(
    xdxr_actions: pd.DataFrame,
    proxy_actions: pd.DataFrame,
    *,
    relative_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return action terms proven by both unofficial transport paths."""

    key = ["security_id", "event_date"]
    if xdxr_actions.empty or proxy_actions.empty:
        return pd.DataFrame(columns=XDXR_EVENT_COLUMNS), pd.DataFrame(), {
            "xdxr_event_count": int(len(xdxr_actions)),
            "proxy_dividend_event_count": int(len(proxy_actions)),
            "corroborated_event_count": 0,
            "value_conflict_count": 0,
        }
    numeric = ["cash_dividend_per_10", "bonus_transfer_per_10"]
    merged = xdxr_actions.merge(proxy_actions, on=key, how="inner", suffixes=("_xdxr", "_proxy"))
    accepted: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        mismatch: dict[str, float] = {}
        compared = 0
        for column in numeric:
            left = pd.to_numeric(row.get(f"{column}_xdxr"), errors="coerce")
            right = pd.to_numeric(row.get(f"{column}_proxy"), errors="coerce")
            if not np.isfinite(left) and not np.isfinite(right):
                continue
            if not np.isfinite(left) or not np.isfinite(right):
                mismatch[column] = float("inf")
                continue
            compared += 1
            error = abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), np.finfo(float).tiny)
            if error > relative_tolerance:
                mismatch[column] = float(error)
        if mismatch or compared == 0:
            conflicts.append(
                {
                    "security_id": row["security_id"],
                    "event_date": row["event_date"],
                    "conflict_type": "xdxr_tushare_proxy_action_terms_mismatch" if mismatch else "no_comparable_action_terms",
                    "relative_errors": mismatch,
                }
            )
            continue
        accepted.append(
            {
                "security_id": row["security_id"],
                "event_date": row["event_date"],
                "symbol_on_date": row.get("symbol_on_date_xdxr", ""),
                "provider_symbol": row.get("provider_symbol_xdxr", ""),
                "category": row.get("category_xdxr", 1),
                "action_type": row.get("action_type_xdxr", ""),
                **{column: row.get(f"{column}_xdxr") for column in XDXR_EVENT_COLUMNS if column in numeric},
                "rights_price": row.get("rights_price_xdxr"),
                "rights_share_per_10": row.get("rights_share_per_10_xdxr"),
                "exercise_fraction": row.get("exercise_fraction_xdxr"),
                "exercise_price": row.get("exercise_price_xdxr"),
                "identity_mapping_status": "mapped",
                "verification_status": "xdxr+tushare_proxy_corroborated",
                "source": "mootdx.xdxr+tushare_proxy.dividend",
            }
        )
    frame = pd.DataFrame(accepted, columns=XDXR_EVENT_COLUMNS)
    conflict_frame = pd.DataFrame(conflicts)
    return frame, conflict_frame, {
        "xdxr_event_count": int(len(xdxr_actions)),
        "proxy_dividend_event_count": int(len(proxy_actions)),
        "matched_key_count": int(len(merged)),
        "corroborated_event_count": int(len(frame)),
        "value_conflict_count": int(len(conflict_frame)),
        "relative_tolerance": float(relative_tolerance),
    }


def load_official_factor_evidence(
    *,
    identity_registry: SecurityIdentityRegistry,
    workspace_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    path = Path(config_path or default_official_evidence_config(workspace_root))
    payload = read_json(path)
    if not payload:
        return pd.DataFrame(columns=OFFICIAL_FACTOR_EVIDENCE_COLUMNS)
    if payload.get("schema_version") != "qdp_v3_official_corporate_action_evidence_v1":
        raise ValueError(f"official_corporate_action_evidence_schema_mismatch:{path}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(list(payload.get("events", []) or [])):
        if not isinstance(item, dict):
            raise ValueError(f"official_corporate_action_evidence_row_invalid:{index}")
        event_date = str(item.get("event_date", ""))[:10]
        provider_symbol = normalize_symbol(item.get("provider_symbol", ""))
        security_id = str(item.get("security_id", "") or identity_registry.security_id_for_provider_symbol(provider_symbol))
        source = str(item.get("official_source", "")).strip().lower()
        document_hash = str(item.get("official_document_sha256", "")).strip().lower()
        if not security_id or not event_date or source not in _OFFICIAL_SOURCES or not _SHA256.fullmatch(document_hash):
            raise ValueError(f"official_corporate_action_evidence_key_invalid:{index}")
        local_document = str(item.get("local_document_path", "") or "").strip()
        if local_document:
            local_path = Path(local_document)
            if not local_path.is_absolute():
                local_path = (path.parent / local_path).resolve()
            if not local_path.exists() or sha256_file(local_path).lower() != document_hash:
                raise ValueError(f"official_corporate_action_document_hash_mismatch:{index}:{local_path}")
        factor_values = [pd.to_numeric(item.get(column), errors="coerce") for column in ("fore_adjust_factor", "back_adjust_factor", "adjust_factor")]
        if any(not np.isfinite(value) or float(value) <= 0 for value in factor_values):
            raise ValueError(f"official_corporate_action_factor_values_invalid:{index}")
        rows.append(
            {
                "security_id": security_id,
                "event_date": event_date,
                "provider_symbol": provider_symbol,
                "official_source": source,
                "official_document_url": str(item.get("official_document_url", "") or ""),
                "official_document_sha256": document_hash,
                "fore_adjust_factor": float(factor_values[0]),
                "back_adjust_factor": float(factor_values[1]),
                "adjust_factor": float(factor_values[2]),
                "price_adjustment_applicable": bool(item.get("price_adjustment_applicable", True)),
                "factor_semantics": str(item.get("factor_semantics", "price_continuity_cumulative_factor") or "price_continuity_cumulative_factor"),
                "holder_entitlement_ratio": pd.to_numeric(item.get("holder_entitlement_ratio"), errors="coerce"),
                "notes": str(item.get("notes", "") or ""),
            }
        )
    frame = pd.DataFrame(rows, columns=OFFICIAL_FACTOR_EVIDENCE_COLUMNS)
    if frame.duplicated(["security_id", "event_date"]).any():
        raise ValueError("official_corporate_action_evidence_duplicate_key")
    return frame.sort_values(["event_date", "security_id"]).reset_index(drop=True)


def _candidate_from_dispute(row: dict[str, Any], suffix: str) -> dict[str, Any] | None:
    factors = {}
    for column in ("fore_adjust_factor", "back_adjust_factor", "adjust_factor"):
        value = pd.to_numeric(row.get(f"{column}_{suffix}"), errors="coerce")
        if not np.isfinite(value) or float(value) <= 0:
            return None
        factors[column] = float(value)
    return {
        "security_id": str(row.get("security_id", "")),
        "divid_operate_date": str(row.get("divid_operate_date", "")),
        "symbol_on_date": str(row.get(f"symbol_on_date_{suffix}", "") or ""),
        "provider_symbol": str(row.get(f"provider_symbol_{suffix}", "") or ""),
        **factors,
        "query_date": str(row.get(f"query_date_{suffix}", "") or row.get("divid_operate_date", "")),
        "identity_mapping_status": str(row.get(f"identity_mapping_status_{suffix}", "") or "mapped"),
        "candidate_path": "date_batch" if suffix == "batch" else "symbol_history",
    }


def arbitrate_adjust_factor_disputes(
    disputed: pd.DataFrame,
    *,
    xdxr_events: pd.DataFrame,
    official_evidence: pd.DataFrame,
    relative_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Resolve a dispute only when official factor values match one Bao path.

    TDX xdxr alone never promotes a factor event.  It only strengthens an
    official arbitration and is recorded in the verification status.
    """

    if disputed is None or disputed.empty:
        return pd.DataFrame(columns=FACTOR_CANONICAL_COLUMNS), pd.DataFrame(), {"arbitrated_count": 0, "remaining_count": 0, "xdxr_confirmed_count": 0}
    official_by_key = {
        (str(row.security_id), str(row.event_date)): row._asdict()
        for row in official_evidence.itertuples(index=False)
    } if official_evidence is not None and not official_evidence.empty else {}
    xdxr_keys = set(zip(xdxr_events.get("security_id", pd.Series(dtype=str)).astype(str), xdxr_events.get("event_date", pd.Series(dtype=str)).astype(str)))
    accepted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for row in disputed.to_dict("records"):
        key = (str(row.get("security_id", "")), str(row.get("divid_operate_date", "")))
        evidence = official_by_key.get(key)
        if evidence is None:
            remaining.append({**row, "arbitration_status": "official_evidence_missing", "xdxr_event_present": key in xdxr_keys})
            continue
        matching: list[dict[str, Any]] = []
        for suffix in ("batch", "symbol"):
            candidate = _candidate_from_dispute(row, suffix)
            if candidate is None:
                continue
            close = True
            for column in ("fore_adjust_factor", "back_adjust_factor", "adjust_factor"):
                left = float(candidate[column])
                right = float(evidence[column])
                close = close and abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny) <= float(relative_tolerance)
            if close:
                matching.append(candidate)
        unique_values = {
            (item["fore_adjust_factor"], item["back_adjust_factor"], item["adjust_factor"])
            for item in matching
        }
        if not matching or len(unique_values) != 1:
            remaining.append({**row, "arbitration_status": "official_values_match_no_unique_candidate", "xdxr_event_present": key in xdxr_keys})
            continue
        chosen = matching[0]
        has_xdxr = key in xdxr_keys
        accepted.append(
            {
                **{column: chosen[column] for column in FACTOR_CANONICAL_COLUMNS if column in chosen},
                "source_method": f"official_arbitration+{chosen['candidate_path']}",
                "verification_status": "xdxr_official_arbitrated" if has_xdxr else "official_arbitrated",
                "source": f"baostock.{chosen['candidate_path']}+{evidence['official_source']}",
                "arbitration_source": evidence["official_source"],
                "arbitration_document_hash": evidence["official_document_sha256"],
                "arbitration_xdxr_confirmed": bool(has_xdxr),
                "price_adjustment_applicable": (
                    True
                    if pd.isna(evidence.get("price_adjustment_applicable"))
                    else bool(evidence.get("price_adjustment_applicable"))
                ),
                "factor_semantics": str(
                    evidence.get("factor_semantics", "price_continuity_cumulative_factor")
                    or "price_continuity_cumulative_factor"
                ),
                "holder_entitlement_ratio": pd.to_numeric(
                    evidence.get("holder_entitlement_ratio"), errors="coerce"
                ),
            }
        )
    accepted_frame = pd.DataFrame(accepted, columns=FACTOR_CANONICAL_COLUMNS)
    remaining_frame = pd.DataFrame(remaining)
    metrics = {
        "arbitrated_count": int(len(accepted_frame)),
        "remaining_count": int(len(remaining_frame)),
        "xdxr_confirmed_count": int(accepted_frame.get("arbitration_xdxr_confirmed", pd.Series(dtype=bool)).fillna(False).sum()),
        "official_evidence_count": int(len(official_evidence)),
        "relative_tolerance": float(relative_tolerance),
    }
    return accepted_frame.sort_values(["divid_operate_date", "security_id"]).reset_index(drop=True), remaining_frame.reset_index(drop=True), metrics


def build_share_capital_daily(market_daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "security_id",
        "trade_date",
        "symbol_on_date",
        "total_share",
        "float_share",
        "restricted_share",
        "capital_event_date",
        "baseline_status",
        "source",
    ]
    if market_daily is None or market_daily.empty:
        return pd.DataFrame(columns=columns)
    event_groups = {key: group for key, group in events.groupby("security_id", sort=False)} if events is not None and not events.empty else {}
    rows: list[pd.DataFrame] = []
    for security_id, market_group in market_daily.groupby("security_id", sort=False):
        market = market_group[["security_id", "trade_date", "symbol_on_date"]].sort_values("trade_date").copy()
        market["_trade_ts"] = pd.to_datetime(market["trade_date"], errors="raise")
        capital = event_groups.get(security_id)
        if capital is None or capital.empty:
            market["total_share"] = np.nan
            market["float_share"] = np.nan
            market["capital_event_date"] = ""
        else:
            event = capital[["event_date", "total_share_after", "float_share_after"]].sort_values("event_date").copy()
            event["_event_ts"] = pd.to_datetime(event["event_date"], errors="raise")
            market = pd.merge_asof(market, event, left_on="_trade_ts", right_on="_event_ts", direction="backward", allow_exact_matches=True)
            market["total_share"] = market["total_share_after"]
            market["float_share"] = market["float_share_after"]
            market["capital_event_date"] = market["event_date"].fillna("").astype(str)
        market["restricted_share"] = (market["total_share"] - market["float_share"]).where(market["total_share"].ge(market["float_share"]))
        market["baseline_status"] = np.where(market["capital_event_date"].eq(""), "baseline_unproven", "event_initialized")
        market["source"] = "qdp_v3_forward_fill_xdxr_capital_events"
        rows.append(market)
    return pd.concat(rows, ignore_index=True).loc[:, columns].sort_values(["trade_date", "security_id"]).reset_index(drop=True)


def reconstruct_xdxr_reference_prices(market_daily: pd.DataFrame, actions: pd.DataFrame, *, tick_size: float = 0.01) -> pd.DataFrame:
    columns = ["security_id", "event_date", "previous_close", "provider_preclose", "reconstructed_reference", "absolute_error", "within_one_tick", "proof_status"]
    if market_daily is None or market_daily.empty or actions is None or actions.empty:
        return pd.DataFrame(columns=columns)
    market = market_daily.sort_values(["security_id", "trade_date"]).copy()
    market["previous_close"] = market.groupby("security_id", sort=False)["close"].shift(1)
    keyed = market.set_index(["security_id", "trade_date"], drop=False)
    rows: list[dict[str, Any]] = []
    for action in actions.to_dict("records"):
        key = (str(action.get("security_id", "")), str(action.get("event_date", "")))
        if key not in keyed.index:
            rows.append({"security_id": key[0], "event_date": key[1], "proof_status": "market_event_date_missing"})
            continue
        day = keyed.loc[key]
        if isinstance(day, pd.DataFrame):
            day = day.iloc[0]
        def zero_when_missing(value: Any) -> float:
            numeric = pd.to_numeric(value, errors="coerce")
            return float(numeric) if np.isfinite(numeric) else 0.0

        cash = zero_when_missing(action.get("cash_dividend_per_10"))
        rights = zero_when_missing(action.get("rights_share_per_10"))
        bonus = zero_when_missing(action.get("bonus_transfer_per_10"))
        rights_price = pd.to_numeric(action.get("rights_price"), errors="coerce")
        previous_close = pd.to_numeric(day.get("previous_close"), errors="coerce")
        provider_preclose = pd.to_numeric(day.get("preclose"), errors="coerce")
        if not np.isfinite(previous_close) or (rights > 0 and not np.isfinite(rights_price)):
            rows.append({"security_id": key[0], "event_date": key[1], "previous_close": previous_close, "provider_preclose": provider_preclose, "proof_status": "reference_inputs_unproven"})
            continue
        price = float(rights_price) if np.isfinite(rights_price) else 0.0
        reconstructed = (float(previous_close) - cash / 10.0 + price * rights / 10.0) / (1.0 + bonus / 10.0 + rights / 10.0)
        error = abs(reconstructed - float(provider_preclose)) if np.isfinite(provider_preclose) else np.nan
        rows.append(
            {
                "security_id": key[0],
                "event_date": key[1],
                "previous_close": float(previous_close),
                "provider_preclose": float(provider_preclose) if np.isfinite(provider_preclose) else np.nan,
                "reconstructed_reference": reconstructed,
                "absolute_error": error,
                "within_one_tick": bool(np.isfinite(error) and error <= float(tick_size) + 1e-12),
                "proof_status": "proved" if np.isfinite(error) and error <= float(tick_size) + 1e-12 else "reference_price_mismatch",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def ingest_mootdx_xdxr(
    *,
    symbols: Iterable[str],
    workspace_root: str | Path | None = None,
    provider: MootdxOnlineProvider | None = None,
    refresh: bool = False,
    chunk_size: int = 32,
    job_id: str = "",
) -> dict[str, Any]:
    ensure_qdp_v3_layout(workspace_root)
    runtime = importlib_metadata.version("mootdx")
    if runtime != MOOTDX_VERSION:
        raise RuntimeError(f"mootdx_version_mismatch:expected={MOOTDX_VERSION}:actual={runtime}")
    normalized = sorted({normalize_symbol(item) for item in symbols if normalize_symbol(item)})
    if not normalized:
        raise ValueError("mootdx_xdxr_symbols_empty")
    if not job_id:
        job_id = f"ingest_mootdx_xdxr__{stable_hash({'symbols': normalized}, length=20)}"
    job_path = qdp_v3_paths(workspace_root).jobs / f"{job_id}.json"
    state = read_json(job_path) or {
        "job_id": job_id,
        "provider": "mootdx_online",
        "mode": "corporate-action-xdxr",
        "symbols": normalized,
        "tasks": {symbol: {"status": "pending", "error": ""} for symbol in normalized},
        "created_at": utc_now(),
    }
    if state.get("symbols") != normalized:
        raise RuntimeError(f"mootdx_xdxr_job_contract_conflict:{job_id}")
    tasks = dict(state.get("tasks", {}) or {})
    existing = {ref.partition_value: ref for ref in iter_raw_partitions(RAW_CORPORATE_ACTION_XDXR, workspace_root=workspace_root)}
    pending: list[str] = []
    for symbol in normalized:
        if not refresh and symbol in existing:
            tasks[symbol] = {"status": "skipped", "content_sha256": existing[symbol].content_sha256, "row_count": existing[symbol].row_count, "error": ""}
        elif not refresh and str(dict(tasks.get(symbol, {}) or {}).get("status", "")) == "completed":
            continue
        else:
            pending.append(symbol)
    source = provider or MootdxOnlineProvider()
    state.update({"status": "running", "tasks": tasks, "updated_at": utc_now()})
    atomic_write_json(job_path, state)
    size = max(1, int(chunk_size))
    for offset in range(0, len(pending), size):
        chunk = pending[offset : offset + size]
        try:
            result = source.fetch_xdxr_raw(chunk)
            failed = {normalize_symbol(item.get("symbol", "")): item for item in result.error_report if isinstance(item, dict)}
            for symbol in chunk:
                if symbol in failed:
                    tasks[symbol] = {"status": "failed", "error": str(failed[symbol].get("message", "provider_symbol_error"))}
                    continue
                frame = result.data.loc[result.data.get("provider_symbol", pd.Series(dtype=str)).astype(str).eq(symbol)].copy() if not result.data.empty else result.data.copy()
                ref, created = write_raw_partition(
                    raw_domain=RAW_CORPORATE_ACTION_XDXR,
                    partition_field="provider_symbol",
                    partition_value=symbol,
                    frame=frame,
                    receipt={
                        "provider": "mootdx_online",
                        "endpoint": "xdxr/get_xdxr_info",
                        "package_version": runtime,
                        "request": {"symbol": symbol},
                        "error_code": "0",
                        "quality_tier": QUALITY_PROVISIONAL,
                        "quality_note": "Lossless TDX xdxr evidence; official disclosure is required for disputed factor promotion.",
                        "raw_share_unit": "10k_shares",
                    },
                    workspace_root=workspace_root,
                )
                tasks[symbol] = {"status": "completed", "content_sha256": ref.content_sha256, "row_count": ref.row_count, "created": bool(created), "error": ""}
        except Exception as exc:
            for symbol in chunk:
                tasks[symbol] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        state.update({"tasks": tasks, "updated_at": utc_now()})
        atomic_write_json(job_path, state)
    failed_tasks = [{"symbol": symbol, **dict(task)} for symbol, task in tasks.items() if str(dict(task).get("status", "")) == "failed"]
    state.update({"status": "partial" if failed_tasks else "completed", "tasks": tasks, "updated_at": utc_now()})
    atomic_write_json(job_path, state)
    if provider is None:
        source.close()
    return {
        "status": state["status"],
        "job_id": job_id,
        "symbol_count": len(normalized),
        "completed_count": sum(str(dict(item).get("status", "")) == "completed" for item in tasks.values()),
        "skipped_count": sum(str(dict(item).get("status", "")) == "skipped" for item in tasks.values()),
        "failed_count": len(failed_tasks),
        "failed": failed_tasks[:50],
        "job_path": str(job_path.resolve()),
    }


def load_xdxr_canonical(
    *,
    identity_registry: SecurityIdentityRegistry,
    workspace_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[RawPartitionRef]]:
    refs = iter_raw_partitions(RAW_CORPORATE_ACTION_XDXR, workspace_root=workspace_root)
    frames = [(ref.partition_value, read_raw_partition(ref)) for ref in refs]
    actions, capitals, conflicts = canonicalize_mootdx_xdxr(frames, identity_registry=identity_registry)
    return actions, capitals, conflicts, refs
