from __future__ import annotations

"""Small, read-only adapters from trusted Tushare raw facts to QDP v2 rows.

The helpers in this module deliberately do not publish datasets or mutate an
active manifest.  They expose bounded batches (one natural year or one stable
security at a time) so a repair command can decide how to merge and publish the
rows later without loading the full market history into memory.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from quant_data_platform.qdp_v3.constants import (
    RAW_TUSHARE_PROXY_ADJ_FACTOR,
    RAW_TUSHARE_PROXY_DAILY,
    RAW_TUSHARE_PROXY_NAMECHANGE,
    RAW_TUSHARE_PROXY_STOCK_BASIC,
    RAW_TUSHARE_PROXY_SUSPEND,
)
from quant_data_platform.qdp_v3.identity import (
    IDENTITY_COLUMNS,
    SYMBOL_HISTORY_COLUMNS,
    SecurityIdentityRegistry,
    board_for_symbol,
    normalize_symbol,
)
from quant_data_platform.qdp_v3.storage import RawPartitionRef, iter_raw_partitions, read_raw_partition


TRUSTED_START_DATE = "2010-01-01"

V2_TRADING_CALENDAR_COLUMNS = (
    "trade_date",
    "is_open",
    "exchange",
    "source",
)

V2_MARKET_DAILY_COLUMNS = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "adjusted_flag",
)

V2_ADJUST_FACTOR_COLUMNS = (
    "symbol",
    "trade_date",
    "fore_adjust_factor",
    "back_adjust_factor",
    "adjust_factor",
    "factor_provider",
    "factor_semantics",
    "source",
    "factor_source_date",
    "ffill_days",
)

V2_SECURITY_STATUS_COLUMNS = (
    "symbol",
    "trade_date",
    "is_st",
    "is_suspended",
    "is_delisted",
    "status_reason",
    "source",
)

ExistingLoader = Callable[[str], pd.DataFrame | None]


@dataclass(frozen=True)
class IdentityRepairTables:
    security_identity: pd.DataFrame
    symbol_history: pd.DataFrame
    registry: SecurityIdentityRegistry


@dataclass(frozen=True)
class RepairBatch:
    partition_key: str
    rows: pd.DataFrame
    insert_count: int
    update_count: int
    source_partition_count: int


def _stable_symbol_map(identity_registry: SecurityIdentityRegistry) -> dict[str, str]:
    """Return the repository's stable research symbol for each identity.

    QDP v2 and the downloaded local minute archive already restate historical
    rows to the current code.  Direct repair must preserve that convention;
    PIT code history remains identity evidence rather than becoming the table
    primary key.
    """

    return {
        str(row.security_id): normalize_symbol(str(row.current_symbol))
        for row in identity_registry.identities.itertuples(index=False)
    }


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label}_fields_missing:{missing}")


def _iso_dates(values: pd.Series) -> pd.Series:
    dates = values.fillna("").astype(str).str.strip().str.replace("-", "", regex=False)
    valid = dates.str.fullmatch(r"\d{8}")
    out = pd.Series("", index=values.index, dtype="string")
    out.loc[valid] = (
        dates.loc[valid].str.slice(0, 4)
        + "-"
        + dates.loc[valid].str.slice(4, 6)
        + "-"
        + dates.loc[valid].str.slice(6, 8)
    )
    return out.astype(str)


def _refs(
    raw_domain: str,
    *,
    workspace_root: str | Path | None,
    supplied: Iterable[RawPartitionRef] | None,
) -> list[RawPartitionRef]:
    return list(supplied) if supplied is not None else iter_raw_partitions(raw_domain, workspace_root=workspace_root)


def _stock_basic_master(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("ts_code", "name", "list_date"), label="tushare_stock_basic")
    out = pd.DataFrame(
        {
            "provider_symbol": frame["ts_code"].map(normalize_symbol),
            "name": frame["name"].fillna("").astype(str).str.strip(),
            "list_date": _iso_dates(frame["list_date"]),
            "delist_date": _iso_dates(
                frame["delist_date"] if "delist_date" in frame.columns else pd.Series("", index=frame.index)
            ),
        }
    )
    out["board"] = out["provider_symbol"].map(board_for_symbol)
    out["identity_source"] = "tushare_proxy.stock_basic"
    out = out.loc[out["provider_symbol"].ne("")].drop_duplicates("provider_symbol", keep="last")
    return out.reset_index(drop=True)


def _is_mainboard_a_symbol(value: str) -> bool:
    symbol = normalize_symbol(value)
    code = symbol.split(".", 1)[0]
    if symbol.endswith(".SH"):
        return code.startswith(("600", "601", "603", "605"))
    if symbol.endswith(".SZ"):
        return code.startswith(("000", "001", "002", "003"))
    return False


def tushare_trade_calendar_to_v2(
    raw: pd.DataFrame,
    *,
    start_date: str = TRUSTED_START_DATE,
) -> pd.DataFrame:
    """Convert trusted Tushare ``trade_cal`` rows to the active v2 schema.

    QDP v2 stores one SSE calendar row per natural date and uses ``is_open``
    to select research dates.  Existing keys are deliberately not reconciled
    here; the direct-repair runner applies the repository-wide missing-only
    policy after this structural normalization.
    """

    _require_columns(raw, ("cal_date", "is_open"), label="tushare_trade_calendar")
    dates = _iso_dates(raw["cal_date"])
    open_numeric = pd.to_numeric(raw["is_open"], errors="coerce")
    valid_open = open_numeric.isin((0, 1))
    valid_date = dates.ne("") & dates.ge(str(start_date))
    invalid = dates.ne("") & dates.ge(str(start_date)) & ~valid_open
    if invalid.any():
        raise ValueError(
            f"tushare_trade_calendar_is_open_invalid:{int(invalid.sum())}"
        )

    exchange = (
        raw["exchange"].fillna("").astype(str).str.strip().str.upper()
        if "exchange" in raw.columns
        else pd.Series("SSE", index=raw.index, dtype="string")
    )
    exchange = exchange.mask(exchange.eq(""), "SSE")
    out = pd.DataFrame(
        {
            "trade_date": dates,
            "is_open": open_numeric.eq(1),
            "exchange": exchange.astype(str),
            "source": "tushare_proxy.trade_cal",
        }
    ).loc[valid_date & valid_open, V2_TRADING_CALENDAR_COLUMNS]
    if out.empty:
        return pd.DataFrame(columns=V2_TRADING_CALENDAR_COLUMNS)

    duplicated = out.duplicated("trade_date", keep=False)
    if duplicated.any():
        conflicts = (
            out.loc[duplicated]
            .groupby("trade_date", dropna=False)[["is_open", "exchange"]]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            raise ValueError(
                f"tushare_trade_calendar_duplicate_conflict:{int(conflicts.sum())}"
            )
        out = out.drop_duplicates("trade_date", keep="last")
    return out.sort_values("trade_date", kind="mergesort").reset_index(drop=True)


def build_mainboard_identity_tables(
    *,
    workspace_root: str | Path | None = None,
    stock_basic_refs: Iterable[RawPartitionRef] | None = None,
    config_path: str | Path | None = None,
) -> IdentityRepairTables:
    """Build the small stable-identity tables used by all v2 repair streams."""

    refs = _refs(
        RAW_TUSHARE_PROXY_STOCK_BASIC,
        workspace_root=workspace_root,
        supplied=stock_basic_refs,
    )
    if not refs:
        raise FileNotFoundError("tushare_stock_basic_raw_missing")
    master = _stock_basic_master(read_raw_partition(refs[-1]))
    master = master.loc[master["provider_symbol"].map(_is_mainboard_a_symbol)].copy()
    provider_symbols = master["provider_symbol"].tolist()
    registry = SecurityIdentityRegistry.from_sources(
        provider_symbols=provider_symbols,
        security_master=master,
        config_path=config_path,
        workspace_root=workspace_root,
    )
    requested_ids = {
        registry.security_id_for_provider_symbol(symbol)
        for symbol in provider_symbols
        if registry.security_id_for_provider_symbol(symbol)
    }
    history = registry.symbol_history.loc[
        registry.symbol_history["security_id"].isin(requested_ids)
        & (
            registry.symbol_history["board_on_date"].eq("MainBoard")
            | registry.symbol_history["symbol"].map(board_for_symbol).eq("MainBoard")
        )
        & registry.symbol_history["symbol"].map(_is_mainboard_a_symbol)
    ].copy()
    identity_ids = set(history["security_id"])
    identities = registry.identities.loc[registry.identities["security_id"].isin(identity_ids)].copy()
    mainboard_registry = SecurityIdentityRegistry(
        identities=identities.loc[:, IDENTITY_COLUMNS].reset_index(drop=True),
        symbol_history=history.loc[:, SYMBOL_HISTORY_COLUMNS].reset_index(drop=True),
    )
    return IdentityRepairTables(
        security_identity=mainboard_registry.identities.copy(),
        symbol_history=mainboard_registry.symbol_history.copy(),
        registry=mainboard_registry,
    )


def _deduplicate_trusted_rows(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str],
    facts: Sequence[str],
    label: str,
) -> pd.DataFrame:
    duplicated = frame.duplicated(list(keys), keep=False)
    if not duplicated.any():
        return frame.reset_index(drop=True)
    conflicts = (
        frame.loc[duplicated]
        .groupby(list(keys), dropna=False)[list(facts)]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        raise ValueError(f"{label}_duplicate_conflict:{int(conflicts.sum())}")
    return frame.drop_duplicates(list(keys), keep="last").reset_index(drop=True)


def tushare_daily_to_v2(
    raw: pd.DataFrame,
    *,
    identity_registry: SecurityIdentityRegistry,
    start_date: str = TRUSTED_START_DATE,
) -> pd.DataFrame:
    """Convert one or more Tushare daily partitions to the lean v2 schema."""

    required = ("ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount")
    _require_columns(raw, required, label="tushare_daily")
    base = pd.DataFrame(
        {
            "trade_date": _iso_dates(raw["trade_date"]),
            "provider_symbol": raw["ts_code"].map(normalize_symbol),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            # Tushare daily vol is in lots and amount is in thousand CNY.
            "volume": pd.to_numeric(raw["vol"], errors="coerce") * 100.0,
            "amount": pd.to_numeric(raw["amount"], errors="coerce") * 1_000.0,
        }
    )
    mapped = identity_registry.map_frame(base)
    mapped = mapped.loc[
        mapped["identity_mapping_status"].eq("mapped")
        & mapped["board_on_date"].eq("MainBoard")
        & mapped["trade_date"].ge(str(start_date))
    ].copy()
    if mapped.empty:
        return pd.DataFrame(columns=V2_MARKET_DAILY_COLUMNS)
    finite = np.isfinite(mapped[["open", "high", "low", "close", "volume", "amount"]]).all(axis=1)
    valid = (
        finite
        & mapped[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & mapped["volume"].ge(0)
        & mapped["amount"].ge(0)
        & mapped["high"].ge(mapped[["open", "low", "close"]].max(axis=1))
        & mapped["low"].le(mapped[["open", "high", "close"]].min(axis=1))
    )
    if not valid.all():
        raise ValueError(f"tushare_daily_structurally_invalid:{int((~valid).sum())}")
    out = mapped.copy()
    out["symbol"] = out["security_id"].map(_stable_symbol_map(identity_registry)).fillna("")
    out = out.loc[out["symbol"].ne("")].copy()
    out["source"] = "tushare_proxy.daily"
    out["adjusted_flag"] = "none"
    out = out.loc[:, V2_MARKET_DAILY_COLUMNS]
    out = _deduplicate_trusted_rows(
        out,
        keys=("trade_date", "symbol"),
        facts=("open", "high", "low", "close", "volume", "amount"),
        label="tushare_daily",
    )
    return out.sort_values(["trade_date", "symbol"], kind="mergesort").reset_index(drop=True)


def _filter_repairs(
    incoming: pd.DataFrame,
    existing: pd.DataFrame | None,
    *,
    columns: Sequence[str],
    compare_columns: Sequence[str],
    include_changed: bool = True,
) -> tuple[pd.DataFrame, int, int]:
    rows = incoming.loc[:, columns].copy()
    if existing is None or existing.empty:
        return rows.reset_index(drop=True), int(len(rows)), 0
    keys = ["trade_date", "symbol"]
    _require_columns(existing, (*keys, *compare_columns), label="existing_v2")
    if existing.duplicated(keys).any():
        raise ValueError("existing_v2_duplicate_key")
    left = rows.reset_index(names="_incoming_order")
    right = existing.loc[:, [*keys, *compare_columns]].copy()
    merged = left.merge(right, on=keys, how="left", suffixes=("", "__old"), indicator=True, sort=False)
    inserted = merged["_merge"].eq("left_only")
    changed = pd.Series(False, index=merged.index)
    for column in compare_columns:
        old = merged[f"{column}__old"]
        new = merged[column]
        if pd.api.types.is_numeric_dtype(new) or pd.api.types.is_bool_dtype(new):
            old_numeric = pd.to_numeric(old, errors="coerce")
            new_numeric = pd.to_numeric(new, errors="coerce")
            equal = np.isclose(new_numeric, old_numeric, rtol=1e-12, atol=1e-12, equal_nan=True)
            changed |= ~pd.Series(equal, index=merged.index)
        else:
            changed |= new.astype("string").fillna("<NULL>").ne(old.astype("string").fillna("<NULL>"))
    updated = ~inserted & changed
    selected_mask = inserted | updated if include_changed else inserted
    selected_orders = merged.loc[selected_mask, "_incoming_order"].astype(int)
    selected = rows.iloc[selected_orders].reset_index(drop=True)
    return selected, int(inserted.sum()), int(updated.sum()) if include_changed else 0


def _refs_by_year(refs: Iterable[RawPartitionRef], *, label: str) -> dict[str, list[RawPartitionRef]]:
    grouped: dict[str, list[RawPartitionRef]] = defaultdict(list)
    for ref in refs:
        value = str(ref.partition_value)
        if ref.partition_field != "trade_date" or len(value) < 4 or not value[:4].isdigit():
            raise ValueError(f"{label}_partition_not_trade_date:{ref.partition_field}={value}")
        grouped[value[:4]].append(ref)
    return dict(grouped)


def iter_market_daily_repairs_by_year(
    *,
    identity_registry: SecurityIdentityRegistry,
    workspace_root: str | Path | None = None,
    daily_refs: Iterable[RawPartitionRef] | None = None,
    years: Iterable[int | str] | None = None,
    existing_loader: ExistingLoader | None = None,
    start_date: str = TRUSTED_START_DATE,
    include_changed: bool = True,
) -> Iterator[RepairBatch]:
    """Yield only missing/changed v2 daily rows, one natural year at a time."""

    refs = _refs(RAW_TUSHARE_PROXY_DAILY, workspace_root=workspace_root, supplied=daily_refs)
    by_year = _refs_by_year(refs, label="tushare_daily")
    selected_years = {str(item) for item in years} if years is not None else set(by_year)
    for year in sorted(set(by_year).intersection(selected_years)):
        frames = [
            tushare_daily_to_v2(
                read_raw_partition(ref),
                identity_registry=identity_registry,
                start_date=start_date,
            )
            for ref in by_year[year]
        ]
        incoming = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=V2_MARKET_DAILY_COLUMNS)
        incoming = _deduplicate_trusted_rows(
            incoming,
            keys=("trade_date", "symbol"),
            facts=("open", "high", "low", "close", "volume", "amount"),
            label="tushare_daily_year",
        )
        existing = existing_loader(year) if existing_loader is not None else None
        rows, inserts, updates = _filter_repairs(
            incoming,
            existing,
            columns=V2_MARKET_DAILY_COLUMNS,
            compare_columns=("open", "high", "low", "close", "volume", "amount"),
            include_changed=include_changed,
        )
        yield RepairBatch(year, rows, inserts, updates, len(by_year[year]))


def tushare_adjust_factor_to_v2(
    raw: pd.DataFrame,
    *,
    identity_registry: SecurityIdentityRegistry,
    start_date: str = TRUSTED_START_DATE,
) -> pd.DataFrame:
    """Normalize one stable security's Tushare factor history for v2."""

    _require_columns(raw, ("ts_code", "trade_date", "adj_factor"), label="tushare_adj_factor")
    base = pd.DataFrame(
        {
            "provider_symbol": raw["ts_code"].map(normalize_symbol),
            "trade_date": _iso_dates(raw["trade_date"]),
            "adj_factor": pd.to_numeric(raw["adj_factor"], errors="coerce"),
        }
    )
    mapped = identity_registry.map_frame(base)
    mapped = mapped.loc[
        mapped["identity_mapping_status"].eq("mapped")
        & mapped["board_on_date"].eq("MainBoard")
        & mapped["trade_date"].ge(str(start_date))
    ].copy()
    if mapped.empty:
        return pd.DataFrame(columns=V2_ADJUST_FACTOR_COLUMNS)
    valid = np.isfinite(mapped["adj_factor"]) & mapped["adj_factor"].gt(0)
    if not valid.all():
        raise ValueError(f"tushare_adj_factor_invalid:{int((~valid).sum())}")
    mapped["_pit_symbol_match"] = mapped["provider_symbol"].eq(mapped["symbol_on_date"])
    mapped = mapped.sort_values(
        ["security_id", "trade_date", "_pit_symbol_match", "provider_symbol"],
        kind="mergesort",
    ).drop_duplicates(["security_id", "trade_date"], keep="last")
    if mapped["security_id"].nunique() != 1:
        raise ValueError(f"tushare_adj_factor_batch_not_one_security:{mapped['security_id'].nunique()}")
    mapped = mapped.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    baseline = float(mapped.iloc[0]["adj_factor"])
    mapped["adjust_factor"] = mapped["adj_factor"] / baseline

    stable_symbols = _stable_symbol_map(identity_registry)
    stable_symbol = mapped["security_id"].map(stable_symbols).fillna("")
    if stable_symbol.eq("").any():
        raise ValueError("tushare_adj_factor_stable_symbol_missing")

    out = pd.DataFrame(
        {
            "symbol": stable_symbol,
            "trade_date": mapped["trade_date"],
            "fore_adjust_factor": mapped["adjust_factor"].astype(float),
            "back_adjust_factor": mapped["adjust_factor"].astype(float),
            "adjust_factor": mapped["adjust_factor"].astype(float),
            "factor_provider": "tushare_proxy",
            "factor_semantics": "trusted_source_first_2010_observation_normalized_to_1",
            "source": "tushare_proxy.adj_factor",
            "factor_source_date": mapped["trade_date"],
            "ffill_days": 0,
        }
    )
    return out.loc[:, V2_ADJUST_FACTOR_COLUMNS].reset_index(drop=True)


def iter_adjust_factor_repairs_by_security(
    *,
    identity_registry: SecurityIdentityRegistry,
    workspace_root: str | Path | None = None,
    factor_refs: Iterable[RawPartitionRef] | None = None,
    existing_loader: ExistingLoader | None = None,
    start_date: str = TRUSTED_START_DATE,
) -> Iterator[RepairBatch]:
    """Yield one normalized stable-security factor sequence at a time."""

    refs = _refs(RAW_TUSHARE_PROXY_ADJ_FACTOR, workspace_root=workspace_root, supplied=factor_refs)
    grouped: dict[str, list[RawPartitionRef]] = defaultdict(list)
    for ref in refs:
        security_id = identity_registry.security_id_for_provider_symbol(ref.partition_value)
        if security_id:
            grouped[security_id].append(ref)
    compare = ("fore_adjust_factor", "back_adjust_factor", "adjust_factor")
    for security_id in sorted(grouped):
        raw = pd.concat([read_raw_partition(ref) for ref in grouped[security_id]], ignore_index=True, sort=False)
        incoming = tushare_adjust_factor_to_v2(
            raw,
            identity_registry=identity_registry,
            start_date=start_date,
        )
        existing = existing_loader(security_id) if existing_loader is not None else None
        rows, inserts, updates = _filter_repairs(
            incoming,
            existing,
            columns=V2_ADJUST_FACTOR_COLUMNS,
            compare_columns=compare,
        )
        yield RepairBatch(security_id, rows, inserts, updates, len(grouped[security_id]))


def _name_intervals(
    refs: Iterable[RawPartitionRef],
    *,
    identity_registry: SecurityIdentityRegistry,
) -> dict[str, list[tuple[str, str, bool]]]:
    grouped: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    for ref in refs:
        raw = read_raw_partition(ref)
        if raw.empty:
            continue
        _require_columns(raw, ("ts_code", "name", "start_date"), label="tushare_namechange")
        starts = _iso_dates(raw["start_date"])
        ends = _iso_dates(raw["end_date"] if "end_date" in raw.columns else pd.Series("", index=raw.index))
        for index, provider_symbol in raw["ts_code"].map(normalize_symbol).items():
            security_id = identity_registry.security_id_for_provider_symbol(provider_symbol)
            start = str(starts.loc[index])
            if not security_id or not start:
                continue
            end = str(ends.loc[index]) or "9999-12-31"
            is_st = "ST" in str(raw.loc[index, "name"] or "").upper()
            grouped[security_id].append((start, end, is_st))
    return {key: sorted(values) for key, values in grouped.items()}


def _suspend_keys(
    refs: Iterable[RawPartitionRef],
    *,
    identity_registry: SecurityIdentityRegistry,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ref in refs:
        raw = read_raw_partition(ref)
        if raw.empty:
            continue
        _require_columns(raw, ("ts_code", "trade_date"), label="tushare_suspend")
        base = pd.DataFrame(
            {
                "provider_symbol": raw["ts_code"].map(normalize_symbol),
                "trade_date": _iso_dates(raw["trade_date"]),
            }
        )
        mapped = identity_registry.map_frame(base)
        mapped = mapped.loc[
            mapped["identity_mapping_status"].eq("mapped")
            & mapped["board_on_date"].eq("MainBoard")
        ]
        mapped["symbol"] = mapped["security_id"].map(
            _stable_symbol_map(identity_registry)
        ).fillna("")
        frames.append(mapped.loc[mapped["symbol"].ne(""), ["security_id", "symbol", "trade_date"]])
    if not frames:
        return pd.DataFrame(columns=["security_id", "symbol", "trade_date"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["trade_date", "symbol"])


def iter_security_status_repairs_by_year(
    *,
    identity_registry: SecurityIdentityRegistry,
    workspace_root: str | Path | None = None,
    daily_refs: Iterable[RawPartitionRef] | None = None,
    suspend_refs: Iterable[RawPartitionRef] | None = None,
    namechange_refs: Iterable[RawPartitionRef] | None = None,
    years: Iterable[int | str] | None = None,
    existing_loader: ExistingLoader | None = None,
    start_date: str = TRUSTED_START_DATE,
    include_changed: bool = True,
) -> Iterator[RepairBatch]:
    """Yield a minimal v2 status table from daily, name and suspension facts."""

    daily = _refs(RAW_TUSHARE_PROXY_DAILY, workspace_root=workspace_root, supplied=daily_refs)
    suspends = _refs(RAW_TUSHARE_PROXY_SUSPEND, workspace_root=workspace_root, supplied=suspend_refs)
    names = _refs(RAW_TUSHARE_PROXY_NAMECHANGE, workspace_root=workspace_root, supplied=namechange_refs)
    daily_by_year = _refs_by_year(daily, label="tushare_daily")
    suspend_by_year = _refs_by_year(suspends, label="tushare_suspend")
    intervals = _name_intervals(names, identity_registry=identity_registry)
    available_years = set(daily_by_year) | set(suspend_by_year)
    selected_years = {str(item) for item in years} if years is not None else available_years
    compare = (
        "is_st",
        "is_suspended",
        "is_delisted",
        "status_reason",
    )
    for year in sorted(available_years.intersection(selected_years)):
        daily_frames = [
            tushare_daily_to_v2(
                read_raw_partition(ref),
                identity_registry=identity_registry,
                start_date=start_date,
            )
            for ref in daily_by_year.get(year, ())
        ]
        daily_rows = (
            pd.concat(daily_frames, ignore_index=True)
            if daily_frames
            else pd.DataFrame(columns=V2_MARKET_DAILY_COLUMNS)
        )
        daily_keys = daily_rows.loc[:, ["symbol", "trade_date"]].copy()
        daily_keys["security_id"] = daily_keys["symbol"].map(identity_registry.security_id_for_provider_symbol)
        daily_keys["has_bar"] = True
        suspended = _suspend_keys(
            suspend_by_year.get(year, ()),
            identity_registry=identity_registry,
        )
        suspended = suspended.loc[suspended["trade_date"].ge(str(start_date))].copy()
        suspended["is_suspended"] = True
        keys = daily_keys.merge(
            suspended,
            on=["security_id", "symbol", "trade_date"],
            how="outer",
            sort=False,
        )
        if keys.empty:
            yield RepairBatch(year, pd.DataFrame(columns=V2_SECURITY_STATUS_COLUMNS), 0, 0, 0)
            continue
        keys["has_bar"] = keys["has_bar"].eq(True)
        keys["is_suspended"] = keys["is_suspended"].eq(True)
        keys["is_st"] = False
        for security_id, indices in keys.groupby("security_id", sort=False).groups.items():
            dates = keys.loc[indices, "trade_date"]
            flags = pd.Series(False, index=indices)
            for interval_start, interval_end, is_st in intervals.get(str(security_id), ()):
                if is_st:
                    flags |= dates.between(interval_start, interval_end)
            keys.loc[indices, "is_st"] = flags
        keys["is_delisted"] = False
        keys["status_reason"] = np.select(
            [keys["is_st"], keys["is_suspended"], ~keys["has_bar"]],
            ["st", "suspended", "missing_or_invalid_bar"],
            default="tradeable",
        )
        keys["source"] = "tushare_proxy.daily+namechange+suspend_d"
        incoming = keys.loc[:, V2_SECURITY_STATUS_COLUMNS]
        incoming = _deduplicate_trusted_rows(
            incoming,
            keys=("trade_date", "symbol"),
            facts=compare,
            label="tushare_status_year",
        ).sort_values(["trade_date", "symbol"], kind="mergesort")
        existing = existing_loader(year) if existing_loader is not None else None
        rows, inserts, updates = _filter_repairs(
            incoming,
            existing,
            columns=V2_SECURITY_STATUS_COLUMNS,
            compare_columns=compare,
            include_changed=include_changed,
        )
        yield RepairBatch(
            year,
            rows,
            inserts,
            updates,
            len(daily_by_year.get(year, ())) + len(suspend_by_year.get(year, ())) + len(names),
        )


__all__ = [
    "IdentityRepairTables",
    "RepairBatch",
    "TRUSTED_START_DATE",
    "V2_ADJUST_FACTOR_COLUMNS",
    "V2_MARKET_DAILY_COLUMNS",
    "V2_SECURITY_STATUS_COLUMNS",
    "V2_TRADING_CALENDAR_COLUMNS",
    "build_mainboard_identity_tables",
    "iter_adjust_factor_repairs_by_security",
    "iter_market_daily_repairs_by_year",
    "iter_security_status_repairs_by_year",
    "tushare_adjust_factor_to_v2",
    "tushare_daily_to_v2",
    "tushare_trade_calendar_to_v2",
]
