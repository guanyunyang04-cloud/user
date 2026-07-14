from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from quant_data_platform.qdp_v3.corporate_actions import FACTOR_CANONICAL_COLUMNS
from quant_data_platform.qdp_v3.identity import SecurityIdentityRegistry, normalize_symbol
from quant_data_platform.qdp_v3.quality import QualityFinding
from quant_data_platform.qdp_v3.storage import RawPartitionRef, read_raw_partition


PROXY_BASELINE_COLUMNS = [
    "security_id",
    "baseline_date",
    "symbol_on_date",
    "provider_symbol",
    "proxy_factor",
]

PROXY_FACTOR_EVENT_COLUMNS = [
    "security_id",
    "event_date",
    "symbol_on_date",
    "provider_symbol",
    "previous_proxy_factor",
    "proxy_factor",
    "factor_ratio",
]

TRUSTED_FACTOR_DAILY_COLUMNS = [
    "security_id",
    "trade_date",
    "symbol_on_date",
    "provider_symbol",
    "fore_adjust_factor",
    "back_adjust_factor",
    "adjust_factor",
    "baseline_status",
    "source",
]


def _relative_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), np.finfo(float).tiny)


def _deduplicate_evidence(
    frame: pd.DataFrame,
    *,
    key: list[str],
    value_column: str,
    date_column: str,
    conflict_type: str,
    relative_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted: list[pd.Series] = []
    conflicts: list[dict[str, Any]] = []
    if frame.empty:
        return frame.copy(), pd.DataFrame()
    for values, group in frame.groupby(key, dropna=False, sort=False):
        numeric = pd.to_numeric(group[value_column], errors="coerce")
        finite = numeric[np.isfinite(numeric) & numeric.gt(0)]
        if len(finite) != len(group):
            conflicts.extend(
                {**row, "conflict_type": "tushare_proxy_factor_nonpositive_or_nonfinite"}
                for row in group.to_dict("records")
            )
            continue
        first = float(finite.iloc[0])
        if any(_relative_error(first, float(item)) > relative_tolerance for item in finite.iloc[1:]):
            conflicts.extend(
                {**row, "conflict_type": conflict_type, "key": list(values) if isinstance(values, tuple) else [values]}
                for row in group.to_dict("records")
            )
            continue
        preferred = group.loc[group["provider_symbol"].eq(group["symbol_on_date"])]
        accepted.append((preferred.iloc[0] if not preferred.empty else group.iloc[0]).copy())
    out = pd.DataFrame(accepted, columns=frame.columns) if accepted else frame.iloc[0:0].copy()
    return out.sort_values([date_column, "security_id"], kind="mergesort").reset_index(drop=True), pd.DataFrame(conflicts)


def derive_proxy_factor_evidence(
    refs: Iterable[RawPartitionRef],
    *,
    identity_registry: SecurityIdentityRegistry,
    change_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reduce daily proxy factors to scale-invariant baselines and change events.

    The proxy's absolute factor is not treated as canonical.  Only the first
    positive observation and adjacent ratios are retained as third-path
    evidence, so a constant provider-specific scaling cannot create a false
    dispute.
    """

    baseline_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    conflict_frames: list[pd.DataFrame] = []
    raw_row_count = 0
    for ref in refs:
        raw = read_raw_partition(ref)
        raw_row_count += int(len(raw))
        if raw.empty:
            continue
        required = {"trade_date", "adj_factor"}
        if not required.issubset(raw.columns):
            conflict_frames.append(
                pd.DataFrame(
                    [
                        {
                            "provider_symbol": normalize_symbol(ref.partition_value),
                            "conflict_type": "tushare_proxy_factor_schema_invalid",
                            "missing": sorted(required - set(raw.columns)),
                        }
                    ]
                )
            )
            continue
        provider = (
            raw["ts_code"].map(normalize_symbol)
            if "ts_code" in raw.columns
            else pd.Series(normalize_symbol(ref.partition_value), index=raw.index)
        )
        base = pd.DataFrame(
            {
                "trade_date": raw["trade_date"].fillna("").astype(str).str.slice(0, 10),
                "provider_symbol": provider,
                "proxy_factor": pd.to_numeric(raw["adj_factor"], errors="coerce"),
            }
        )
        mapped = identity_registry.map_frame(base, provider_symbol_column="provider_symbol", date_column="trade_date")
        invalid = mapped.loc[
            mapped["identity_mapping_status"].ne("mapped")
            | ~np.isfinite(mapped["proxy_factor"])
            | mapped["proxy_factor"].le(0)
            | mapped["trade_date"].eq("")
        ].copy()
        if not invalid.empty:
            invalid["conflict_type"] = "tushare_proxy_factor_identity_or_value_invalid"
            conflict_frames.append(invalid)
        mapped = mapped.loc[
            mapped["identity_mapping_status"].eq("mapped")
            & np.isfinite(mapped["proxy_factor"])
            & mapped["proxy_factor"].gt(0)
            & mapped["trade_date"].ne("")
        ].copy()
        if mapped.empty:
            continue
        mapped = mapped.sort_values(["trade_date", "provider_symbol"], kind="mergesort")
        duplicate = mapped.duplicated(["security_id", "trade_date"], keep=False)
        if duplicate.any():
            same_day, conflicts = _deduplicate_evidence(
                mapped,
                key=["security_id", "trade_date"],
                value_column="proxy_factor",
                date_column="trade_date",
                conflict_type="tushare_proxy_factor_provider_symbol_restatement_conflict",
                relative_tolerance=relative_tolerance,
            )
            if not conflicts.empty:
                conflict_frames.append(conflicts)
            mapped = same_day
        for security_id, history in mapped.groupby("security_id", sort=False):
            history = history.sort_values("trade_date", kind="mergesort").copy()
            history["previous_proxy_factor"] = history["proxy_factor"].shift(1)
            history["factor_ratio"] = history["proxy_factor"] / history["previous_proxy_factor"]
            first = history.iloc[[0]].rename(columns={"trade_date": "baseline_date"})
            baseline_frames.append(first.loc[:, PROXY_BASELINE_COLUMNS])
            changed = history["previous_proxy_factor"].notna() & history["factor_ratio"].sub(1.0).abs().gt(change_tolerance)
            events = history.loc[changed].rename(columns={"trade_date": "event_date"})
            event_frames.append(events.loc[:, PROXY_FACTOR_EVENT_COLUMNS])

    baselines = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame(columns=PROXY_BASELINE_COLUMNS)
    proxy_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(columns=PROXY_FACTOR_EVENT_COLUMNS)
    baseline_conflicts = pd.DataFrame()
    event_conflicts = pd.DataFrame()
    if not baselines.empty:
        baselines, baseline_conflicts = _deduplicate_evidence(
            baselines,
            key=["security_id", "baseline_date"],
            value_column="proxy_factor",
            date_column="baseline_date",
            conflict_type="tushare_proxy_factor_baseline_duplicate_conflict",
            relative_tolerance=relative_tolerance,
        )
        # Multiple provider symbols may expose the same restated history.  One
        # earliest baseline per stable identity is enough evidence.
        baselines = baselines.sort_values(["security_id", "baseline_date"]).drop_duplicates("security_id", keep="first")
    if not proxy_events.empty:
        proxy_events, event_conflicts = _deduplicate_evidence(
            proxy_events,
            key=["security_id", "event_date"],
            value_column="factor_ratio",
            date_column="event_date",
            conflict_type="tushare_proxy_factor_ratio_duplicate_conflict",
            relative_tolerance=relative_tolerance,
        )
    conflicts = pd.concat(
        [item for item in [*conflict_frames, baseline_conflicts, event_conflicts] if not item.empty],
        ignore_index=True,
        sort=False,
    ) if any(not item.empty for item in [*conflict_frames, baseline_conflicts, event_conflicts]) else pd.DataFrame()
    metrics = {
        "raw_row_count": raw_row_count,
        "security_baseline_count": int(baselines["security_id"].nunique()) if not baselines.empty else 0,
        "factor_change_event_count": int(len(proxy_events)),
        "conflict_count": int(len(conflicts)),
        "comparison_semantics": "adjacent_ratio_constant_scale_invariant",
        "relative_tolerance": float(relative_tolerance),
    }
    return baselines.reset_index(drop=True), proxy_events.reset_index(drop=True), conflicts.reset_index(drop=True), metrics


def trusted_proxy_factor_daily_from_refs(
    refs: Iterable[RawPartitionRef],
    *,
    identity_registry: SecurityIdentityRegistry,
    start_date: str = "2010-01-01",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ref in refs:
        raw = read_raw_partition(ref)
        if raw.empty:
            continue
        if not {"trade_date", "adj_factor"}.issubset(raw.columns):
            raise ValueError(f"trusted_proxy_factor_raw_schema_invalid:{ref.partition_value}")
        provider = (
            raw["ts_code"].map(normalize_symbol)
            if "ts_code" in raw.columns
            else pd.Series(normalize_symbol(ref.partition_value), index=raw.index)
        )
        base = pd.DataFrame(
            {
                "trade_date": raw["trade_date"],
                "provider_symbol": provider,
                "adj_factor": raw["adj_factor"],
            }
        )
        mapped = identity_registry.map_frame(base, provider_symbol_column="provider_symbol", date_column="trade_date")
        unresolved = mapped["identity_mapping_status"].ne("mapped")
        if unresolved.any():
            raise ValueError(f"trusted_proxy_factor_identity_unmapped:{int(unresolved.sum())}")
        frames.append(mapped)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(
        columns=["security_id", "trade_date", "symbol_on_date", "provider_symbol", "adj_factor"]
    )
    return normalize_trusted_proxy_factors(combined, start_date=start_date)


def normalize_trusted_proxy_factors(
    frame: pd.DataFrame,
    *,
    start_date: str = "2010-01-01",
) -> pd.DataFrame:
    """Normalize each Tushare factor history to its first 2010+ observation."""

    required = {"security_id", "trade_date", "adj_factor"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"trusted_proxy_factor_fields_missing:{missing}")
    data = frame.copy()
    data["trade_date"] = data["trade_date"].fillna("").astype(str).str.replace("-", "", regex=False)
    compact = data["trade_date"].str.fullmatch(r"\d{8}")
    data.loc[compact, "trade_date"] = (
        data.loc[compact, "trade_date"].str.slice(0, 4)
        + "-"
        + data.loc[compact, "trade_date"].str.slice(4, 6)
        + "-"
        + data.loc[compact, "trade_date"].str.slice(6, 8)
    )
    data["adj_factor"] = pd.to_numeric(data["adj_factor"], errors="coerce")
    invalid = data["security_id"].fillna("").astype(str).eq("") | ~np.isfinite(data["adj_factor"]) | data["adj_factor"].le(0)
    if invalid.any():
        raise ValueError(f"trusted_proxy_factor_invalid_rows:{int(invalid.sum())}")
    data = data.loc[data["trade_date"].ge(str(start_date))].copy()
    duplicate = data.duplicated(["security_id", "trade_date"], keep=False)
    if duplicate.any():
        conflicts = data.loc[duplicate].groupby(["security_id", "trade_date"])["adj_factor"].nunique().gt(1)
        if conflicts.any():
            raise ValueError(f"trusted_proxy_factor_duplicate_conflict:{int(conflicts.sum())}")
        data = data.drop_duplicates(["security_id", "trade_date"], keep="last")
    data = data.sort_values(["security_id", "trade_date"], kind="mergesort")
    baseline = data.groupby("security_id", sort=False)["adj_factor"].transform("first")
    data["adjust_factor"] = data["adj_factor"] / baseline
    data["fore_adjust_factor"] = data["adjust_factor"]
    data["back_adjust_factor"] = data["adjust_factor"]
    data["baseline_status"] = "trusted_source_ratio"
    first_index = data.groupby("security_id", sort=False).head(1).index
    data.loc[first_index, "baseline_status"] = "trusted_source_2010_normalized"
    data["source"] = "tushare_proxy.adj_factor"
    for column in ("symbol_on_date", "provider_symbol"):
        if column not in data.columns:
            data[column] = ""
    return data.loc[:, TRUSTED_FACTOR_DAILY_COLUMNS].reset_index(drop=True)


def continue_trusted_factors_with_baostock(
    proxy_daily: pd.DataFrame,
    baostock_events: pd.DataFrame,
    *,
    cutoff: str,
    baostock_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extend proxy factors with post-cutoff BaoStock absolute-factor ratios.

    ``baostock_events`` is the authoritative post-cutoff event stream.  The
    optional ``baostock_history`` supplies the provider's complete absolute
    factor history used to anchor the first new event at the cutoff.  Keeping
    those roles separate prevents a pre-cutoff change already represented in
    the proxy series from being multiplied a second time.
    """

    required_proxy = {"security_id", "trade_date", "adjust_factor"}
    required_bao = {"security_id", "divid_operate_date", "adjust_factor"}
    if not required_proxy.issubset(proxy_daily.columns):
        raise ValueError(f"trusted_factor_proxy_fields_missing:{sorted(required_proxy - set(proxy_daily.columns))}")
    if baostock_events.empty:
        return pd.DataFrame(columns=TRUSTED_FACTOR_DAILY_COLUMNS)
    if not required_bao.issubset(baostock_events.columns):
        raise ValueError(f"trusted_factor_baostock_fields_missing:{sorted(required_bao - set(baostock_events.columns))}")
    proxy = proxy_daily.copy()
    proxy["trade_date"] = proxy["trade_date"].astype(str).str.slice(0, 10)
    proxy["adjust_factor"] = pd.to_numeric(proxy["adjust_factor"], errors="coerce")
    bao = baostock_events.copy()
    bao["divid_operate_date"] = bao["divid_operate_date"].astype(str).str.slice(0, 10)
    bao["adjust_factor"] = pd.to_numeric(bao["adjust_factor"], errors="coerce")
    absolute_history_all = (
        baostock_history if baostock_history is not None else baostock_events
    ).copy()
    if not required_bao.issubset(absolute_history_all.columns):
        raise ValueError(
            "trusted_factor_baostock_history_fields_missing:"
            f"{sorted(required_bao - set(absolute_history_all.columns))}"
        )
    absolute_history_all["divid_operate_date"] = (
        absolute_history_all["divid_operate_date"].astype(str).str.slice(0, 10)
    )
    absolute_history_all["adjust_factor"] = pd.to_numeric(
        absolute_history_all["adjust_factor"], errors="coerce"
    )
    invalid = ~np.isfinite(bao["adjust_factor"]) | bao["adjust_factor"].le(0)
    history_invalid = ~np.isfinite(absolute_history_all["adjust_factor"]) | absolute_history_all[
        "adjust_factor"
    ].le(0)
    if invalid.any() or history_invalid.any():
        raise ValueError(
            "trusted_factor_baostock_invalid_rows:"
            f"events={int(invalid.sum())}:history={int(history_invalid.sum())}"
        )
    for label, frame in (("events", bao), ("history", absolute_history_all)):
        duplicate = frame.duplicated(["security_id", "divid_operate_date"], keep=False)
        if duplicate.any():
            conflicts = (
                frame.loc[duplicate]
                .groupby(["security_id", "divid_operate_date"], dropna=False)["adjust_factor"]
                .nunique()
                .gt(1)
            )
            if conflicts.any():
                raise ValueError(f"trusted_factor_baostock_{label}_duplicate_conflict:{int(conflicts.sum())}")
            frame.drop_duplicates(["security_id", "divid_operate_date"], keep="last", inplace=True)
    rows: list[dict[str, Any]] = []
    post_events = bao.loc[bao["divid_operate_date"].gt(str(cutoff))].copy()
    for security_id, events in post_events.sort_values(
        ["security_id", "divid_operate_date"], kind="mergesort"
    ).groupby("security_id", sort=False):
        first_event_date = str(events.iloc[0]["divid_operate_date"])
        proxy_history = proxy.loc[
            proxy["security_id"].astype(str).eq(str(security_id))
            & proxy["trade_date"].lt(first_event_date)
        ].sort_values("trade_date", kind="mergesort")
        if proxy_history.empty:
            raise ValueError(f"trusted_factor_proxy_anchor_missing:{security_id}")
        current = float(proxy_history.iloc[-1]["adjust_factor"])
        absolute_history = absolute_history_all.loc[
            absolute_history_all["security_id"].astype(str).eq(str(security_id))
        ].sort_values("divid_operate_date", kind="mergesort")
        pre_event = absolute_history.loc[
            absolute_history["divid_operate_date"].lt(first_event_date)
        ]
        if pre_event.empty:
            raise ValueError(
                f"trusted_factor_baostock_ratio_anchor_missing:{security_id}:{first_event_date}"
            )
        previous = float(pre_event.iloc[-1]["adjust_factor"])
        absolute_by_date = {
            str(row.divid_operate_date): float(row.adjust_factor)
            for row in absolute_history.itertuples(index=False)
        }
        for event in events.itertuples(index=False):
            event_date = str(event.divid_operate_date)
            absolute = absolute_by_date.get(event_date)
            if absolute is None:
                raise ValueError(f"trusted_factor_baostock_event_history_missing:{security_id}:{event_date}")
            current *= float(absolute) / previous
            previous = float(absolute)
            rows.append(
                {
                    "security_id": str(security_id),
                    "trade_date": event_date,
                    "symbol_on_date": str(getattr(event, "symbol_on_date", "") or ""),
                    "provider_symbol": str(getattr(event, "provider_symbol", "") or ""),
                    "fore_adjust_factor": current,
                    "back_adjust_factor": current,
                    "adjust_factor": current,
                    "baseline_status": "trusted_source_continued",
                    "source": "baostock.adjust_factor_event_ratio",
                }
            )
    return pd.DataFrame(rows, columns=TRUSTED_FACTOR_DAILY_COLUMNS)


def build_trusted_factor_daily(
    market_daily: pd.DataFrame,
    proxy_daily: pd.DataFrame,
    continuation_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Align trusted factor observations to canonical market dates and forward-fill."""

    required = {"security_id", "trade_date"}
    if not required.issubset(market_daily.columns):
        raise ValueError(f"trusted_factor_market_fields_missing:{sorted(required - set(market_daily.columns))}")
    observations = [proxy_daily]
    if isinstance(continuation_events, pd.DataFrame) and not continuation_events.empty:
        observations.append(continuation_events)
    factors = pd.concat(observations, ignore_index=True, sort=False)
    factors["trade_date"] = factors["trade_date"].astype(str).str.slice(0, 10)
    factors = factors.sort_values(["security_id", "trade_date"], kind="mergesort").drop_duplicates(
        ["security_id", "trade_date"], keep="last"
    )
    market_columns = ["security_id", "trade_date"]
    for column in ("symbol_on_date", "provider_symbol"):
        if column in market_daily.columns:
            market_columns.append(column)
    market = market_daily.loc[:, market_columns].copy()
    market["trade_date"] = market["trade_date"].astype(str).str.slice(0, 10)
    value_columns = ["fore_adjust_factor", "back_adjust_factor", "adjust_factor", "baseline_status", "source"]
    market["_is_market_row"] = True
    factor_rows = factors[["security_id", "trade_date", *value_columns]].copy()
    factor_rows["_is_market_row"] = False
    combined = pd.concat([factor_rows, market], ignore_index=True, sort=False)
    combined = combined.sort_values(["security_id", "trade_date", "_is_market_row"], kind="mergesort")
    combined[value_columns] = combined.groupby("security_id", sort=False)[value_columns].ffill()
    carried = combined.loc[combined["_is_market_row"].fillna(False)].drop(columns=["_is_market_row"])
    merged = market.drop(columns=["_is_market_row"]).merge(
        carried[["security_id", "trade_date", *value_columns]],
        on=["security_id", "trade_date"],
        how="left",
    ).sort_values(["security_id", "trade_date"], kind="mergesort")
    missing = merged["adjust_factor"].isna()
    if missing.any():
        raise ValueError(f"trusted_factor_market_coverage_missing:{int(missing.sum())}")
    for column in ("symbol_on_date", "provider_symbol"):
        if column not in merged.columns:
            merged[column] = ""
    return merged.loc[:, TRUSTED_FACTOR_DAILY_COLUMNS].reset_index(drop=True)


def reconcile_proxy_factor_third_path(
    symbol_events: pd.DataFrame,
    *,
    proxy_baselines: pd.DataFrame,
    proxy_events: pd.DataFrame,
    comparable_start: str,
    comparable_end: str,
    relative_tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, list[QualityFinding], dict[str, Any]]:
    """Verify BaoStock history with proxy factor ratios and admit pre-start proof."""

    admitted: list[dict[str, Any]] = []
    disputes: list[dict[str, Any]] = []
    findings: list[QualityFinding] = []
    if symbol_events is None or symbol_events.empty:
        return pd.DataFrame(columns=FACTOR_CANONICAL_COLUMNS), pd.DataFrame(), findings, {
            "bao_symbol_event_count": 0,
            "proxy_event_count": int(len(proxy_events)),
            "verified_ratio_count": 0,
            "verified_pre_start_count": 0,
        }
    bao = symbol_events.copy().sort_values(["security_id", "divid_operate_date"], kind="mergesort")
    bao["adjust_factor"] = pd.to_numeric(bao["adjust_factor"], errors="coerce")
    bao["previous_adjust_factor"] = bao.groupby("security_id", sort=False)["adjust_factor"].shift(1)
    bao["factor_ratio"] = bao["adjust_factor"] / bao["previous_adjust_factor"]
    bao["is_baseline"] = bao["previous_adjust_factor"].isna()
    bao_changes = bao.loc[~bao["is_baseline"] & bao["factor_ratio"].sub(1.0).abs().gt(1e-12)].copy()
    proxy_by_key = {
        (str(row.security_id), str(row.event_date)): row._asdict()
        for row in proxy_events.itertuples(index=False)
    }
    bao_change_keys: set[tuple[str, str]] = set()
    verified_ratio_count = 0
    for row in bao_changes.to_dict("records"):
        key = (str(row.get("security_id", "")), str(row.get("divid_operate_date", "")))
        bao_change_keys.add(key)
        proxy = proxy_by_key.get(key)
        if proxy is None:
            if str(comparable_start) <= key[1] <= str(comparable_end) or key[1] < str(comparable_start):
                disputes.append({"security_id": key[0], "event_date": key[1], "dispute_reason": "tushare_proxy_factor_event_missing"})
            continue
        error = _relative_error(float(row["factor_ratio"]), float(proxy["factor_ratio"]))
        if error > relative_tolerance:
            disputes.append(
                {
                    "security_id": key[0],
                    "event_date": key[1],
                    "dispute_reason": "tushare_proxy_factor_ratio_mismatch",
                    "baostock_ratio": float(row["factor_ratio"]),
                    "proxy_ratio": float(proxy["factor_ratio"]),
                    "relative_error": float(error),
                }
            )
            continue
        verified_ratio_count += 1
        if key[1] < str(comparable_start):
            accepted = {column: row.get(column, "") for column in FACTOR_CANONICAL_COLUMNS}
            accepted.update(
                {
                    "source_method": "symbol_history+tushare_proxy_ratio",
                    "verification_status": "verified_third_path_ratio_pre_batch",
                    "source": "baostock.query_adjust_factor+tushare_proxy.adj_factor",
                    "price_adjustment_applicable": True,
                    "factor_semantics": "price_continuity_cumulative_factor",
                    "holder_entitlement_ratio": np.nan,
                }
            )
            admitted.append(accepted)

    for key, proxy in proxy_by_key.items():
        if key not in bao_change_keys and str(comparable_start) <= key[1] <= str(comparable_end):
            disputes.append(
                {
                    "security_id": key[0],
                    "event_date": key[1],
                    "dispute_reason": "tushare_proxy_only_factor_change",
                    "proxy_ratio": float(proxy["factor_ratio"]),
                }
            )

    baseline_by_security = {
        str(row.security_id): row._asdict() for row in proxy_baselines.itertuples(index=False)
    }
    for security_id, group in bao.groupby("security_id", sort=False):
        first = group.iloc[0]
        proxy = baseline_by_security.get(str(security_id))
        if proxy is None:
            disputes.append(
                {
                    "security_id": str(security_id),
                    "event_date": str(first["divid_operate_date"]),
                    "dispute_reason": "tushare_proxy_factor_baseline_missing",
                }
            )
            continue
        if str(proxy["baseline_date"]) != str(first["divid_operate_date"]):
            disputes.append(
                {
                    "security_id": str(security_id),
                    "event_date": str(first["divid_operate_date"]),
                    "proxy_baseline_date": str(proxy["baseline_date"]),
                    "dispute_reason": "factor_baseline_date_mismatch",
                }
            )
            continue
        if str(first["divid_operate_date"]) < str(comparable_start):
            accepted = {column: first.get(column, "") for column in FACTOR_CANONICAL_COLUMNS}
            accepted.update(
                {
                    "source_method": "symbol_history+tushare_proxy_baseline",
                    "verification_status": "verified_third_path_baseline_pre_batch",
                    "source": "baostock.query_adjust_factor+tushare_proxy.adj_factor",
                    "price_adjustment_applicable": True,
                    "factor_semantics": "price_continuity_baseline_state",
                    "holder_entitlement_ratio": np.nan,
                }
            )
            admitted.append(accepted)

    disputed = pd.DataFrame(disputes).drop_duplicates(
        [column for column in ("security_id", "event_date", "dispute_reason") if column in pd.DataFrame(disputes).columns]
    ) if disputes else pd.DataFrame()
    if not disputed.empty:
        findings.append(
            QualityFinding(
                code="tushare_proxy_factor_third_path_disputed",
                severity="blocker",
                message="Proxy adjacent factor ratios or baselines disagree with BaoStock symbol history.",
                domain="adjust_factor_event",
                count=int(len(disputed)),
                sample=disputed.head(20).to_dict("records"),
            )
        )
    admitted_frame = pd.DataFrame(admitted)
    if not admitted_frame.empty:
        for column in FACTOR_CANONICAL_COLUMNS:
            if column not in admitted_frame.columns:
                admitted_frame[column] = np.nan if column == "holder_entitlement_ratio" else ""
        admitted_frame = admitted_frame.loc[:, FACTOR_CANONICAL_COLUMNS]
        admitted_frame = admitted_frame.drop_duplicates(["security_id", "divid_operate_date"], keep="last")
    else:
        admitted_frame = pd.DataFrame(columns=FACTOR_CANONICAL_COLUMNS)
    metrics = {
        "bao_symbol_event_count": int(len(bao)),
        "bao_change_event_count": int(len(bao_changes)),
        "proxy_event_count": int(len(proxy_events)),
        "proxy_baseline_count": int(len(proxy_baselines)),
        "verified_ratio_count": int(verified_ratio_count),
        "verified_pre_start_count": int(len(admitted_frame)),
        "disputed_count": int(len(disputed)),
        "relative_tolerance": float(relative_tolerance),
    }
    return admitted_frame.reset_index(drop=True), disputed.reset_index(drop=True), findings, metrics
