from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = Path("daily_research/studies/seq100_payoff_episode_audit_v1.json")
SCHEMA = "seq100_payoff_episode_audit/1"
MAXIMUM_OUTCOME_DATE = "2025-12-31"
PRIMARY_TASK_ID = (
    "payoff_tree_sequence_mean_rank_d10__planned_close__always__top10__stress"
)
NO_OVERLAP_TASK_ID = PRIMARY_TASK_ID + "__no_overlapping_same_symbol"
SCENARIOS = {
    "overlap": PRIMARY_TASK_ID,
    "no_overlap": NO_OVERLAP_TASK_ID,
}
EPISODE_GAPS = {"overlap": 0, "adjacent": 1}
TOP_FRACTIONS = (0.01, 0.05, 0.10)


class PayoffEpisodeAuditError(RuntimeError):
    """Raised when the frozen payoff audit contract is violated."""


def _workspace_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _read_json(path: str | Path) -> dict[str, Any]:
    target = _workspace_path(path)
    payload = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise PayoffEpisodeAuditError(f"expected a JSON object: {target}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _write_parquet(path: str | Path, frame: pd.DataFrame) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)
    return target


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _workspace_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path, **extra: Any) -> dict[str, Any]:
    target = _workspace_path(path).resolve()
    record: dict[str, Any] = {
        "path": str(target),
        "size": int(target.stat().st_size),
        "sha256": _sha256(target),
    }
    record.update(extra)
    return record


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PayoffEpisodeAuditError(message)


def _date_text(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.strftime("%Y-%m-%d")


def _date_in_scope(values: pd.Series) -> bool:
    normalized = _date_text(values)
    return bool(normalized.le(MAXIMUM_OUTCOME_DATE).all())


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    return np.memmap(
        Path(str(meta["path"])),
        dtype=dtype,
        mode="r",
        shape=tuple(int(value) for value in meta["shape"]),
    )


def _dataset_manifest(
    *, qdp_root: Path, dataset_id: str, domain: str
) -> tuple[Path, dict[str, Any]]:
    path = qdp_root / "datasets" / domain / dataset_id / "dataset.json"
    payload = _read_json(path)
    _require(
        str(payload.get("dataset_id")) == str(dataset_id),
        f"dataset id drifted for {domain}",
    )
    return path, payload


def _historical_shards(
    manifest: Mapping[str, Any], *, qdp_root: Path, start_date: str
) -> list[Path]:
    paths: list[Path] = []
    for shard in manifest.get("shards", []):
        if str(shard.get("status", "stored")) != "stored":
            continue
        if str(shard["start_date"]) > MAXIMUM_OUTCOME_DATE:
            continue
        if str(shard["end_date"]) < str(start_date):
            continue
        target = (qdp_root / str(shard["path"])).resolve()
        _require(target.is_file(), f"missing active dataset shard: {target}")
        paths.append(target)
    _require(bool(paths), "no historical dataset shard selected")
    return paths


def _parquet_scan_sql(paths: Sequence[Path]) -> str:
    escaped = [str(path).replace("'", "''") for path in paths]
    return "read_parquet([" + ",".join(f"'{path}'" for path in escaped) + "])"


def _query_selected_keys(
    *,
    con: duckdb.DuckDBPyConnection,
    keys: pd.DataFrame,
    paths: Sequence[Path],
    columns: Sequence[str],
    key_date_column: str,
) -> pd.DataFrame:
    key_frame = keys[["symbol", key_date_column]].drop_duplicates().copy()
    key_frame = key_frame.rename(columns={key_date_column: "trade_date"})
    key_frame["symbol"] = key_frame["symbol"].astype(str)
    key_frame["trade_date"] = _date_text(key_frame["trade_date"])
    _require(_date_in_scope(key_frame["trade_date"]), "2026 key reached audit query")
    con.register("audit_keys", key_frame)
    selected = ", ".join(f"src.{column}" for column in columns)
    result = con.execute(
        f"""
        SELECT {selected}
        FROM {_parquet_scan_sql(paths)} AS src
        INNER JOIN audit_keys AS keys
          ON src.symbol = keys.symbol
         AND src.trade_date = keys.trade_date
        WHERE src.trade_date <= ?
        """,
        [MAXIMUM_OUTCOME_DATE],
    ).fetchdf()
    con.unregister("audit_keys")
    if not result.empty:
        result["trade_date"] = _date_text(result["trade_date"])
        _require(_date_in_scope(result["trade_date"]), "2026 row materialized from QDP")
    return result


def assign_episodes(trades: pd.DataFrame, *, gap_days: int) -> pd.DataFrame:
    """Assign deterministic same-symbol interval episodes without changing trades."""

    required = {
        "trade_id",
        "symbol",
        "signal_date_idx",
        "signal_date",
        "entry_date_idx",
        "entry_date",
        "exit_date_idx",
        "exit_date",
        "pnl",
        "trade_net_return",
        "net_cash_outflow",
        "selection_rank",
    }
    missing = required.difference(trades.columns)
    _require(not missing, f"episode input is missing columns: {sorted(missing)}")
    _require(int(gap_days) >= 0, "episode gap must be non-negative")

    ordered = trades.sort_values(
        ["symbol", "entry_date_idx", "exit_date_idx", "trade_id"],
        kind="mergesort",
    ).copy()
    episode_numbers = np.empty(len(ordered), dtype=np.int32)
    episode_number = -1
    previous_symbol: str | None = None
    current_exit_idx = -1
    for position, row in enumerate(ordered.itertuples(index=False)):
        symbol = str(row.symbol)
        entry_idx = int(row.entry_date_idx)
        exit_idx = int(row.exit_date_idx)
        if previous_symbol != symbol or entry_idx > current_exit_idx + int(gap_days):
            episode_number += 1
            current_exit_idx = exit_idx
        else:
            current_exit_idx = max(current_exit_idx, exit_idx)
        episode_numbers[position] = episode_number
        previous_symbol = symbol
    ordered["episode_number"] = episode_numbers
    ordered["episode_id"] = (
        ordered["symbol"].astype(str)
        + "__"
        + ordered["episode_number"].astype(str).str.zfill(7)
    )
    return ordered


def _max_concurrent_intervals(group: pd.DataFrame) -> int:
    events: list[tuple[int, int]] = []
    for entry_idx, exit_idx in zip(
        group["entry_date_idx"].astype(int),
        group["exit_date_idx"].astype(int),
        strict=True,
    ):
        events.append((int(entry_idx), 1))
        events.append((int(exit_idx) + 1, -1))
    active = 0
    maximum = 0
    for _, change in sorted(events, key=lambda value: (value[0], value[1])):
        active += int(change)
        maximum = max(maximum, active)
    return maximum


def summarize_episodes(assigned: pd.DataFrame, *, definition: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode_id, group in assigned.groupby("episode_id", sort=False):
        pnl = group["pnl"].astype(float)
        row: dict[str, Any] = {
            "episode_definition": str(definition),
            "episode_id": str(episode_id),
            "symbol": str(group["symbol"].iloc[0]),
            "start_signal_date": str(group["signal_date"].min()),
            "end_signal_date": str(group["signal_date"].max()),
            "entry_date": str(group["entry_date"].min()),
            "exit_date": str(group["exit_date"].max()),
            "entry_date_idx": int(group["entry_date_idx"].min()),
            "exit_date_idx": int(group["exit_date_idx"].max()),
            "episode_year": int(str(group["signal_date"].min())[:4]),
            "trade_count": len(group),
            "max_concurrent_cohorts": int(_max_concurrent_intervals(group)),
            "pnl": float(pnl.sum()),
            "positive_pnl": float(pnl.clip(lower=0.0).sum()),
            "negative_pnl": float(pnl.clip(upper=0.0).sum()),
            "mean_trade_net_return": float(
                group["trade_net_return"].astype(float).mean()
            ),
            "max_trade_net_return": float(
                group["trade_net_return"].astype(float).max()
            ),
            "net_cash_outflow_sum": float(
                group["net_cash_outflow"].astype(float).sum()
            ),
            "best_selection_rank": int(group["selection_rank"].astype(int).min()),
        }
        for column in (
            "fold",
            "industry_name",
            "industry_code",
            "total_mv",
            "circ_mv",
            "pe",
            "pb",
            "turnover_rate",
            "corporate_action_count",
            "factor_changed",
            "adjustment_log_change_abs",
        ):
            if column not in group.columns:
                continue
            if column in {"industry_name", "industry_code"}:
                values = group[column].dropna().astype(str)
                row[column] = None if values.empty else str(values.iloc[0])
            elif column == "fold":
                row[column] = int(group[column].astype(int).iloc[0])
            elif column == "factor_changed":
                row[column] = bool(group[column].astype(bool).any())
            elif column == "corporate_action_count":
                row[column] = int(group[column].fillna(0).astype(int).sum())
            else:
                values = pd.to_numeric(group[column], errors="coerce")
                row[column] = (
                    None if not bool(values.notna().any()) else float(values.iloc[0])
                )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["pnl", "episode_id"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def concentration_metrics(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, "concentration requires values")
    net_total = float(array.sum())
    positive = np.clip(array, 0.0, None)
    positive_total = float(positive.sum())
    positive_weights = (
        positive / positive_total
        if positive_total > 0.0
        else np.zeros(array.size, dtype=np.float64)
    )
    result: dict[str, Any] = {
        "unit_count": int(array.size),
        "net_pnl": net_total,
        "positive_pnl": positive_total,
        "negative_pnl": float(np.clip(array, None, 0.0).sum()),
        "effective_positive_contributor_count": (
            float(1.0 / np.square(positive_weights).sum())
            if bool(np.square(positive_weights).sum() > 0.0)
            else 0.0
        ),
    }
    order = np.argsort(array)[::-1]
    for fraction in TOP_FRACTIONS:
        count = max(1, math.ceil(array.size * float(fraction)))
        selected = array[order[:count]]
        prefix = f"top_{round(fraction * 100)}pct"
        result[f"{prefix}_unit_count"] = count
        result[f"{prefix}_net_pnl_share"] = (
            float(selected.sum() / net_total) if net_total != 0.0 else None
        )
        result[f"{prefix}_positive_pnl_share"] = (
            float(np.clip(selected, 0.0, None).sum() / positive_total)
            if positive_total > 0.0
            else None
        )
    return result


def _attribution(
    frame: pd.DataFrame, *, group_columns: Sequence[str], value_column: str = "pnl"
) -> pd.DataFrame:
    total = float(frame[value_column].astype(float).sum())
    positive_total = float(frame[value_column].astype(float).clip(lower=0.0).sum())
    grouped = frame.groupby(list(group_columns), dropna=False)[value_column]
    result = grouped.agg(unit_count="size", pnl="sum").reset_index()
    positive = grouped.apply(lambda values: float(values.clip(lower=0.0).sum()))
    negative = grouped.apply(lambda values: float(values.clip(upper=0.0).sum()))
    result["positive_pnl"] = positive.to_numpy(dtype=np.float64)
    result["negative_pnl"] = negative.to_numpy(dtype=np.float64)
    result["net_pnl_share"] = result["pnl"] / total if total != 0.0 else np.nan
    result["positive_pnl_share"] = (
        result["positive_pnl"] / positive_total if positive_total > 0.0 else np.nan
    )
    return result.sort_values(
        ["pnl", *group_columns], ascending=[False, *([True] * len(group_columns))]
    ).reset_index(drop=True)


def _load_selection_characteristics(
    *,
    trades: pd.DataFrame,
    selection_path: Path,
) -> pd.DataFrame:
    columns = [
        "date_idx",
        "symbol_idx",
        "fold",
        "sequence_prediction",
        "tree_prediction",
        "sequence_rank",
        "tree_rank",
        "ensemble_score",
        "selection_rank",
    ]
    selections = pd.read_parquet(selection_path, columns=[*columns, "top_k"])
    selections = selections.loc[selections["top_k"].astype(int).eq(10), columns]
    _require(
        not bool(selections.duplicated(["date_idx", "symbol_idx"]).any()),
        "top10 selection schedule is not unique",
    )
    merged = trades.merge(
        selections,
        left_on=["signal_date_idx", "symbol_idx"],
        right_on=["date_idx", "symbol_idx"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_selection"),
    ).drop(columns="date_idx")
    _require(bool(merged["fold"].notna().all()), "trade has no matching OOF selection")
    _require(
        bool(
            merged["selection_rank"]
            .astype(int)
            .eq(merged["selection_rank_selection"].astype(int))
            .all()
        ),
        "trade selection rank drifted from source schedule",
    )
    return merged.drop(columns="selection_rank_selection")


def _join_exact_characteristics(
    *,
    trades: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    qdp_root: Path,
    active_datasets: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    joined = trades.copy()
    source_records: dict[str, Any] = {}

    industry_path, industry_manifest = _dataset_manifest(
        qdp_root=qdp_root,
        dataset_id=str(active_datasets["industry_concept"]),
        domain="industry_concept",
    )
    industry_shards = _historical_shards(
        industry_manifest, qdp_root=qdp_root, start_date="2020-01-01"
    )
    industry = _query_selected_keys(
        con=con,
        keys=joined,
        paths=industry_shards,
        columns=(
            "symbol",
            "trade_date",
            "industry_code",
            "industry_name",
            "industry_source_date",
        ),
        key_date_column="signal_date",
    )
    _require(
        not bool(industry.duplicated(["symbol", "trade_date"]).any()),
        "industry PIT key is not unique",
    )
    if not industry.empty:
        source_dates = pd.to_datetime(industry["industry_source_date"], errors="coerce")
        trade_dates = pd.to_datetime(industry["trade_date"], errors="raise")
        _require(
            bool((source_dates.isna() | source_dates.le(trade_dates)).all()),
            "industry source date exceeds signal date",
        )
    joined = joined.merge(
        industry.rename(columns={"trade_date": "signal_date"}),
        on=["symbol", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    joined["industry_name"] = joined["industry_name"].fillna("Unknown")
    joined["industry_code"] = joined["industry_code"].fillna("Unknown")
    source_records["industry_concept"] = {
        "manifest": _file_record(industry_path),
        "historical_shards": [_file_record(path) for path in industry_shards],
        "matched_trade_count": int(joined["industry_source_date"].notna().sum()),
    }

    valuation_path, valuation_manifest = _dataset_manifest(
        qdp_root=qdp_root,
        dataset_id=str(active_datasets["valuation"]),
        domain="valuation",
    )
    valuation_shards = _historical_shards(
        valuation_manifest, qdp_root=qdp_root, start_date="2020-01-01"
    )
    valuation = _query_selected_keys(
        con=con,
        keys=joined,
        paths=valuation_shards,
        columns=(
            "symbol",
            "trade_date",
            "total_mv",
            "circ_mv",
            "pe",
            "pb",
            "turnover_rate",
        ),
        key_date_column="signal_date",
    )
    _require(
        not bool(valuation.duplicated(["symbol", "trade_date"]).any()),
        "valuation PIT key is not unique",
    )
    joined = joined.merge(
        valuation.rename(columns={"trade_date": "signal_date"}),
        on=["symbol", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    source_records["valuation"] = {
        "manifest": _file_record(valuation_path),
        "historical_shards": [_file_record(path) for path in valuation_shards],
        "matched_trade_count": int(joined["total_mv"].notna().sum()),
    }
    return joined, source_records


def _corporate_action_overlap(
    *,
    trades: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    qdp_root: Path,
    dataset_id: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    manifest_path, manifest = _dataset_manifest(
        qdp_root=qdp_root, dataset_id=dataset_id, domain="corporate_actions"
    )
    shards = _historical_shards(manifest, qdp_root=qdp_root, start_date="2020-01-01")
    symbols = pd.DataFrame({"symbol": sorted(trades["symbol"].astype(str).unique())})
    con.register("audit_symbols", symbols)
    actions = con.execute(
        f"""
        SELECT src.symbol, src.trade_date, src.announcement_date, src.ex_date,
               src.record_date, src.dividend_pay_date, src.action_type,
               src.cash_dividend_per_10, src.bonus_share_per_10,
               src.transfer_share_per_10, src.description, src.source
        FROM {_parquet_scan_sql(shards)} AS src
        INNER JOIN audit_symbols AS symbols ON src.symbol = symbols.symbol
        WHERE COALESCE(NULLIF(src.ex_date, ''), src.trade_date) <= ?
          AND COALESCE(NULLIF(src.ex_date, ''), src.trade_date) >= '2020-01-01'
        """,
        [MAXIMUM_OUTCOME_DATE],
    ).fetchdf()
    con.unregister("audit_symbols")
    if actions.empty:
        enriched = trades.copy()
        enriched["corporate_action_count"] = 0
        enriched["corporate_action_types"] = ""
        return (
            enriched,
            {
                "manifest": _file_record(manifest_path),
                "historical_shards": [_file_record(path) for path in shards],
                "matched_trade_count": 0,
            },
            actions,
        )

    actions["action_date"] = actions["ex_date"].where(
        actions["ex_date"].astype(str).str.len().gt(0), actions["trade_date"]
    )
    actions["action_date"] = _date_text(actions["action_date"])
    _require(
        _date_in_scope(actions["action_date"]), "2026 corporate action materialized"
    )
    con.register(
        "audit_trades",
        trades[["trade_id", "symbol", "entry_date", "exit_date"]],
    )
    con.register("audit_actions", actions)
    overlaps = con.execute(
        """
        SELECT trades.trade_id, actions.symbol, actions.action_date,
               actions.action_type, actions.cash_dividend_per_10,
               actions.bonus_share_per_10, actions.transfer_share_per_10,
               actions.description, actions.source
        FROM audit_trades AS trades
        INNER JOIN audit_actions AS actions
          ON trades.symbol = actions.symbol
         AND actions.action_date >= trades.entry_date
         AND actions.action_date <= trades.exit_date
        ORDER BY trades.trade_id, actions.action_date, actions.action_type
        """
    ).fetchdf()
    con.unregister("audit_trades")
    con.unregister("audit_actions")
    if overlaps.empty:
        counts = pd.DataFrame(
            columns=["trade_id", "corporate_action_count", "corporate_action_types"]
        )
    else:
        counts = (
            overlaps.groupby("trade_id", sort=False)
            .agg(
                corporate_action_count=("action_type", "size"),
                corporate_action_types=(
                    "action_type",
                    lambda values: "|".join(sorted(set(values.dropna().astype(str)))),
                ),
            )
            .reset_index()
        )
    enriched = trades.merge(counts, on="trade_id", how="left", validate="one_to_one")
    enriched["corporate_action_count"] = (
        enriched["corporate_action_count"].fillna(0).astype(np.int16)
    )
    enriched["corporate_action_types"] = enriched["corporate_action_types"].fillna("")
    source = {
        "manifest": _file_record(manifest_path),
        "historical_shards": [_file_record(path) for path in shards],
        "matched_trade_count": int(enriched["corporate_action_count"].gt(0).sum()),
        "overlap_row_count": len(overlaps),
    }
    return enriched, source, overlaps


def _execution_integrity(
    *, trades: pd.DataFrame, pack: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily_meta = pack["feature_channels"]["daily_raw"]
    execution = pack["execution_arrays"]
    masks = pack["masks"]
    daily = _open_memmap(daily_meta, dtype="float32")
    entry_open_raw = _open_memmap(execution["entry_open_raw"], dtype="float32")
    exit_close_raw = _open_memmap(execution["exit_close_raw"], dtype="float32")
    exit_down_limit_raw = _open_memmap(
        execution["exit_down_limit_raw"], dtype="float32"
    )
    entry_filled = _open_memmap(masks["entry_filled"], dtype="bool")
    exit_sellable = _open_memmap(masks["exit_sellable"], dtype="bool")
    status_valid = _open_memmap(masks["status_valid"], dtype="bool")
    suspended = _open_memmap(masks["is_suspended"], dtype="bool")
    delisted = _open_memmap(masks["is_delisted"], dtype="bool")

    signal_idx = trades["signal_date_idx"].to_numpy(dtype=np.int64)
    entry_idx = trades["entry_date_idx"].to_numpy(dtype=np.int64)
    exit_idx = trades["exit_date_idx"].to_numpy(dtype=np.int64)
    symbol_idx = trades["symbol_idx"].to_numpy(dtype=np.int64)
    phases = trades["exit_phase"].astype(str).to_numpy()
    planned = phases == "planned_close"
    delayed = phases == "delayed_open"
    _require(bool((planned | delayed).all()), "unknown exit phase in account trades")

    adjusted_entry = np.asarray(daily[entry_idx, symbol_idx, 0], dtype=np.float64)
    adjusted_exit = np.where(
        planned,
        np.asarray(daily[exit_idx, symbol_idx, 3], dtype=np.float64),
        np.asarray(daily[exit_idx, symbol_idx, 0], dtype=np.float64),
    )
    raw_entry = np.asarray(entry_open_raw[entry_idx, symbol_idx], dtype=np.float64)
    raw_exit = np.where(
        planned,
        np.asarray(exit_close_raw[exit_idx, symbol_idx], dtype=np.float64),
        np.asarray(entry_open_raw[exit_idx, symbol_idx], dtype=np.float64),
    )
    legal_return_recomputed = adjusted_exit / adjusted_entry - 1.0
    return_residual = (
        trades["legal_gross_return"].to_numpy(dtype=np.float64)
        - legal_return_recomputed
    )
    entry_price_from_account = trades["gross_entry_notional"].to_numpy(
        dtype=np.float64
    ) / trades["shares"].to_numpy(dtype=np.float64)
    entry_price_residual = entry_price_from_account - raw_entry
    factor_entry = adjusted_entry / raw_entry
    factor_exit = adjusted_exit / raw_exit
    adjustment_log_change = np.log(factor_exit / factor_entry)
    raw_exit_down_limit = np.asarray(
        exit_down_limit_raw[exit_idx, symbol_idx], dtype=np.float64
    )
    delayed_base = (
        np.isfinite(raw_exit)
        & (raw_exit > 0.0)
        & np.asarray(status_valid[exit_idx, symbol_idx], dtype=bool)
        & ~np.asarray(suspended[exit_idx, symbol_idx], dtype=bool)
        & ~np.asarray(delisted[exit_idx, symbol_idx], dtype=bool)
        & np.isfinite(adjusted_exit)
        & (adjusted_exit > 0.0)
    )
    delayed_blocked = (
        delayed_base
        & np.isfinite(raw_exit_down_limit)
        & (
            np.floor(raw_exit / 0.01 + 0.5)
            <= np.floor(raw_exit_down_limit / 0.01 + 0.5)
        )
    )
    legal_exit_mask = np.where(
        planned,
        np.asarray(exit_sellable[exit_idx, symbol_idx], dtype=bool),
        delayed_base & ~delayed_blocked,
    )

    enriched = trades.copy()
    enriched["adjusted_entry_price"] = adjusted_entry
    enriched["adjusted_exit_price"] = adjusted_exit
    enriched["raw_entry_price"] = raw_entry
    enriched["raw_exit_price"] = raw_exit
    enriched["legal_return_recomputed"] = legal_return_recomputed
    enriched["legal_return_residual"] = return_residual
    enriched["entry_price_residual"] = entry_price_residual
    enriched["implied_adjust_factor_entry"] = factor_entry
    enriched["implied_adjust_factor_exit"] = factor_exit
    enriched["adjustment_log_change"] = adjustment_log_change
    enriched["adjustment_log_change_abs"] = np.abs(adjustment_log_change)
    enriched["factor_changed"] = np.abs(adjustment_log_change) > 1.0e-6
    enriched["entry_filled_panel"] = np.asarray(
        entry_filled[signal_idx, symbol_idx], dtype=bool
    )
    enriched["exit_sellable_panel"] = legal_exit_mask

    _require(bool((entry_idx == signal_idx + 1).all()), "entry is not next trading day")
    _require(bool(np.isfinite(adjusted_entry).all()), "missing adjusted entry price")
    _require(bool(np.isfinite(adjusted_exit).all()), "missing adjusted exit price")
    _require(bool(np.isfinite(raw_entry).all()), "missing raw entry price")
    _require(bool(np.isfinite(raw_exit).all()), "missing raw exit price")
    _require(
        bool(enriched["entry_filled_panel"].all()), "recorded trade was not buyable"
    )
    _require(
        bool(enriched["exit_sellable_panel"].all()), "recorded exit was not sellable"
    )
    _require(
        float(np.max(np.abs(return_residual))) <= 5.0e-7,
        "legal return does not match adjusted OHLC path",
    )
    _require(
        float(np.max(np.abs(entry_price_residual))) <= 1.0e-10,
        "account entry price does not match raw execution panel",
    )
    summary = {
        "trade_count": len(enriched),
        "planned_close_count": int(planned.sum()),
        "delayed_open_count": int(delayed.sum()),
        "entry_filled_count": int(enriched["entry_filled_panel"].sum()),
        "exit_sellable_count": int(enriched["exit_sellable_panel"].sum()),
        "factor_changed_trade_count": int(enriched["factor_changed"].sum()),
        "maximum_absolute_legal_return_residual": float(
            np.max(np.abs(return_residual))
        ),
        "maximum_absolute_entry_price_residual": float(
            np.max(np.abs(entry_price_residual))
        ),
    }
    return enriched, summary


def _basic_integrity(
    *, trades: pd.DataFrame, task_result: Mapping[str, Any]
) -> dict[str, Any]:
    _require(not trades.empty, "account trade file is empty")
    for column in ("signal_date", "entry_date", "exit_date"):
        trades[column] = _date_text(trades[column])
        _require(_date_in_scope(trades[column]), f"{column} exceeds 2025 cutoff")
    _require(not bool(trades["trade_id"].duplicated().any()), "duplicate trade id")
    _require(
        bool((trades["signal_date_idx"] < trades["entry_date_idx"]).all()),
        "signal date must precede entry",
    )
    _require(
        bool((trades["entry_date_idx"] <= trades["exit_date_idx"]).all()),
        "entry date must not exceed exit",
    )
    _require(bool(trades["shares"].gt(0).all()), "non-positive share count")
    _require(
        bool(trades["shares"].astype(int).mod(100).eq(0).all()),
        "trade share count violates A-share board lot",
    )
    _require(
        bool(trades["selection_rank"].between(1, 10).all()),
        "selection rank is outside Top10",
    )
    _require(
        bool(trades[["buy_cost", "sell_cost"]].ge(0.0).all().all()),
        "negative execution cost",
    )
    pnl_residual = trades["pnl"].astype(float) - (
        trades["proceeds"].astype(float) - trades["net_cash_outflow"].astype(float)
    )
    _require(
        float(pnl_residual.abs().max()) <= 1.0e-8,
        "trade PnL does not equal proceeds minus cash outflow",
    )
    expected_pnl = float(task_result["ending_equity"]) - float(
        task_result["starting_cash"]
    )
    actual_pnl = float(trades["pnl"].astype(float).sum())
    _require(
        abs(actual_pnl - expected_pnl) <= 1.0e-5,
        "trade PnL does not reconcile to ending account equity",
    )
    _require(
        int(task_result.get("forbidden_2026_read_count", -1)) == 0,
        "source account task does not attest zero 2026 reads",
    )
    return {
        "trade_count": len(trades),
        "pnl": actual_pnl,
        "account_equity_change": expected_pnl,
        "maximum_absolute_pnl_cashflow_residual": float(pnl_residual.abs().max()),
        "forbidden_2026_read_count": 0,
    }


def _characteristic_comparison(episodes: pd.DataFrame) -> dict[str, Any]:
    winners = episodes.loc[episodes["pnl"].gt(0.0)]
    losers = episodes.loc[episodes["pnl"].le(0.0)]

    def summarize(frame: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {"episode_count": len(frame)}
        for column in ("total_mv", "circ_mv", "pe", "pb", "turnover_rate"):
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            result[column] = {
                "count": len(values),
                "median": None if values.empty else float(values.median()),
                "p25": None if values.empty else float(values.quantile(0.25)),
                "p75": None if values.empty else float(values.quantile(0.75)),
            }
        result["corporate_action_episode_fraction"] = (
            float(frame["corporate_action_count"].gt(0).mean())
            if len(frame) and "corporate_action_count" in frame
            else None
        )
        result["factor_changed_episode_fraction"] = (
            float(frame["factor_changed"].astype(bool).mean())
            if len(frame) and "factor_changed" in frame
            else None
        )
        return result

    return {"positive_pnl": summarize(winners), "nonpositive_pnl": summarize(losers)}


def _read_study(path: str | Path) -> tuple[Path, dict[str, Any]]:
    study_path = _workspace_path(path).resolve()
    study = _read_json(study_path)
    _require(str(study.get("study_id")), "study_id is required")
    source = study.get("sources", {})
    _require(
        str(source.get("maximum_outcome_date")) == MAXIMUM_OUTCOME_DATE,
        "audit study must freeze the 2025 outcome cutoff",
    )
    _require(
        int(source.get("forbidden_year", -1)) == 2026,
        "audit study must forbid 2026 outcomes",
    )
    return study_path, study


def run_audit(*, study_path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path, study = _read_study(study_path)
    source_spec = study["sources"]
    account_manifest_path = _workspace_path(source_spec["account_manifest"]).resolve()
    pack_manifest_path = _workspace_path(source_spec["pack_manifest"]).resolve()
    account = _read_json(account_manifest_path)
    pack = _read_json(pack_manifest_path)
    _require(account.get("status") == "completed", "source account is incomplete")
    _require(
        int(account.get("forbidden_2026_read_count", -1)) == 0,
        "source account manifest does not attest zero 2026 reads",
    )
    _require(int(account.get("horizon", -1)) == 10, "source account horizon drifted")
    _require(
        float(account.get("starting_cash", 0.0)) == 1_000_000.0,
        "source starting cash drifted",
    )
    selection_path = Path(str(account["files"]["selection_schedule"]["path"]))
    qdp_root = Path(str(pack["qdp_root"])).resolve()
    active_datasets = pack["active_datasets"]

    output_root = _workspace_path(study["outputs"]["output_root"]).resolve()
    fingerprint_payload = {
        "schema": SCHEMA,
        "implementation": _file_record(Path(__file__).resolve()),
        "study": _file_record(study_path),
        "account_manifest": _file_record(account_manifest_path),
        "pack_manifest": _file_record(pack_manifest_path),
        "selection_schedule": _file_record(selection_path),
        "scenarios": SCENARIOS,
        "episode_gaps": EPISODE_GAPS,
        "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
    }
    fingerprint = _stable_hash(fingerprint_payload)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        current = _read_json(manifest_path)
        if (
            current.get("status") == "completed"
            and current.get("fingerprint") == fingerprint
            and current.get("forbidden_2026_read_count") == 0
        ):
            return current

    con = duckdb.connect(database=":memory:")
    con.execute("SET threads=8")
    scenario_results: dict[str, Any] = {}
    source_records: dict[str, Any] = {}
    files: dict[str, Any] = {}
    all_event_rows: list[pd.DataFrame] = []
    try:
        for scenario, task_id in SCENARIOS.items():
            task_path = (
                account_manifest_path.parent / "tasks" / task_id / "task_result.json"
            )
            task_result = _read_json(task_path)
            trades_path = Path(str(task_result["files"]["trades"]["path"]))
            trades = pd.read_parquet(trades_path)
            basic = _basic_integrity(trades=trades, task_result=task_result)
            trades = _load_selection_characteristics(
                trades=trades, selection_path=selection_path
            )
            trades, execution = _execution_integrity(trades=trades, pack=pack)
            trades, characteristic_sources = _join_exact_characteristics(
                trades=trades,
                con=con,
                qdp_root=qdp_root,
                active_datasets=active_datasets,
            )
            trades, action_source, action_rows = _corporate_action_overlap(
                trades=trades,
                con=con,
                qdp_root=qdp_root,
                dataset_id=str(active_datasets["corporate_actions"]),
            )
            if scenario == "overlap":
                source_records.update(characteristic_sources)
                source_records["corporate_actions"] = action_source
                if not action_rows.empty:
                    action_path = _write_parquet(
                        output_root / "corporate_action_overlaps.parquet", action_rows
                    )
                    files["corporate_action_overlaps"] = _file_record(
                        action_path, row_count=len(action_rows)
                    )

            trade_path = _write_parquet(
                output_root / scenario / "trades_enriched.parquet", trades
            )
            scenario_files: dict[str, Any] = {
                "trades_enriched": _file_record(trade_path, row_count=len(trades))
            }
            definitions: dict[str, Any] = {}
            for definition, gap_days in EPISODE_GAPS.items():
                assigned = assign_episodes(trades, gap_days=gap_days)
                episodes = summarize_episodes(assigned, definition=definition)
                episode_path = _write_parquet(
                    output_root / scenario / f"episodes_{definition}.parquet", episodes
                )
                assigned_path = _write_parquet(
                    output_root / scenario / f"trade_episode_map_{definition}.parquet",
                    assigned[["trade_id", "episode_id"]],
                )
                scenario_files[f"episodes_{definition}"] = _file_record(
                    episode_path, row_count=len(episodes)
                )
                scenario_files[f"trade_episode_map_{definition}"] = _file_record(
                    assigned_path, row_count=len(assigned)
                )

                symbol_attribution = _attribution(episodes, group_columns=["symbol"])
                industry_attribution = _attribution(
                    episodes, group_columns=["industry_code", "industry_name"]
                )
                year_attribution = _attribution(
                    episodes, group_columns=["episode_year"]
                )
                fold_attribution = _attribution(episodes, group_columns=["fold"])
                for name, frame in (
                    ("symbol", symbol_attribution),
                    ("industry", industry_attribution),
                    ("year", year_attribution),
                    ("fold", fold_attribution),
                ):
                    path = _write_parquet(
                        output_root
                        / scenario
                        / f"{name}_attribution_{definition}.parquet",
                        frame,
                    )
                    scenario_files[f"{name}_attribution_{definition}"] = _file_record(
                        path, row_count=len(frame)
                    )

                top_events = episodes.head(
                    max(50, math.ceil(len(episodes) * 0.01))
                ).copy()
                top_events.insert(0, "scenario", scenario)
                all_event_rows.append(top_events)
                definitions[definition] = {
                    "gap_trading_days": int(gap_days),
                    "episode_count": len(episodes),
                    "mean_trades_per_episode": float(episodes["trade_count"].mean()),
                    "maximum_trades_per_episode": int(episodes["trade_count"].max()),
                    "maximum_concurrent_cohorts": int(
                        episodes["max_concurrent_cohorts"].max()
                    ),
                    "concentration": concentration_metrics(episodes["pnl"]),
                    "characteristics": _characteristic_comparison(episodes),
                }
            scenario_results[scenario] = {
                "task_id": task_id,
                "basic_integrity": basic,
                "execution_integrity": execution,
                "trade_concentration": concentration_metrics(trades["pnl"]),
                "episode_definitions": definitions,
                "industry_coverage_fraction": float(
                    trades["industry_source_date"].notna().mean()
                ),
                "valuation_coverage_fraction": float(trades["total_mv"].notna().mean()),
                "corporate_action_trade_fraction": float(
                    trades["corporate_action_count"].gt(0).mean()
                ),
                "factor_changed_trade_fraction": float(trades["factor_changed"].mean()),
                "files": scenario_files,
                "sources": {
                    "task_result": _file_record(task_path),
                    "trades": _file_record(trades_path, row_count=len(trades)),
                },
            }
    finally:
        con.close()

    top_events_frame = pd.concat(all_event_rows, ignore_index=True).sort_values(
        ["pnl", "scenario", "episode_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    top_events_path = _write_parquet(
        output_root / "top_events.parquet", top_events_frame
    )
    files["top_events"] = _file_record(top_events_path, row_count=len(top_events_frame))

    result = {
        "schema": SCHEMA,
        "status": "completed",
        "study_id": study["study_id"],
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "fingerprint": fingerprint,
        "forbidden_2026_read_count": 0,
        "maximum_outcome_date_read": MAXIMUM_OUTCOME_DATE,
        "audit_role": "diagnostic_only_no_trade_or_model_selection_changed",
        "scenario_results": scenario_results,
        "decision_boundary": {
            "episode_audit_is_post_outcome_diagnostic_only": True,
            "same_symbol_overlap_episode_definition": (
                "merge when next entry index is not after the current maximum exit index"
            ),
            "adjacent_episode_is_sensitivity_only": True,
            "corporate_actions_are_not_model_inputs": True,
            "valuation_and_industry_are_exact_signal_date_PIT_joins": True,
            "raw_and_adjusted_price_paths_are_reconciled": True,
            "stable_profit_claim_allowed": False,
            "forbidden_2026_read_count": 0,
        },
        "files": files,
        "sources": {
            "implementation": _file_record(Path(__file__).resolve()),
            "study": _file_record(study_path),
            "account_manifest": _file_record(account_manifest_path),
            "pack_manifest": _file_record(pack_manifest_path),
            "selection_schedule": _file_record(selection_path),
            "qdp_datasets": source_records,
        },
    }
    _write_json(manifest_path, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit payoff episodes and execution/data integrity."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_audit(study_path=args.study)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
