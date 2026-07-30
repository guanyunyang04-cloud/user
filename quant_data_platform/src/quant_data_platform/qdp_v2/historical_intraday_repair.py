from __future__ import annotations

"""Incrementally fill exact historical 5-minute stock-day gaps.

The repair never fabricates bars and never removes a security from the PIT
universe because a provider has no history for it.  Work is checkpointed by
symbol-month and committed through the existing active-manifest repair API.
"""

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.auxiliary_update import (
    _resolve_tushare_token,
    _TushareClient,
)
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quant_data_platform.qdp_v2.recent_market_repair import (
    INTRADAY_COLUMNS,
    INTRADAY_DOMAIN,
    _validate_intraday_frame,
    _write_final_parquet,
)
from quant_data_platform.qdp_v2.repair import (
    _sql_literal,
    bulk_append_active_shards_from_parquet,
    update_active_manifest_metadata,
)
from quant_data_platform.qdp_v2.status import active_dataset_map

REPAIR_ID = "historical_intraday_5m_repair_v1"
API_FIELDS = (
    "ts_code",
    "trade_time",
    "open",
    "close",
    "high",
    "low",
    "vol",
    "amount",
)
VOLUME_SCALES = (1.0, 100.0, 0.01)
AMOUNT_SCALES = (1.0, 1000.0, 0.001, 100.0, 0.01)


class HistoricalIntradayRepairError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime" / REPAIR_ID).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(workspace: Path) -> Path:
    return _runtime(workspace) / "state.json"


def _read_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(workspace: Path, state: Mapping[str, Any]) -> None:
    atomic_write_json(
        _state_path(workspace),
        {**dict(state), "updated_at": utc_now()},
    )


def _active_domain_paths(workspace: Path, domain: str) -> tuple[Path, ...]:
    root = qdp_v2_root(workspace)
    datasets = active_dataset_map(read_active_manifest(root))
    dataset_id = str(datasets.get(domain, ""))
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        raise HistoricalIntradayRepairError(
            f"historical_intraday_active_domain_missing:{domain}"
        )
    manifest = read_dataset_manifest(manifest_path)
    paths = tuple(
        resolve_manifest_path(item.path, root=root) for item in manifest.shards
    )
    if not paths or any(not path.is_file() for path in paths):
        raise HistoricalIntradayRepairError(
            f"historical_intraday_active_shards_missing:{domain}"
        )
    return paths


def _copy_inventory(source: Path, target: Path) -> pd.DataFrame:
    frame = pd.read_parquet(source, columns=["symbol", "trade_date"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["trade_date"] = frame["trade_date"].astype(str).str.slice(0, 10)
    frame = (
        frame.drop_duplicates(["symbol", "trade_date"])
        .sort_values(["symbol", "trade_date"], kind="stable")
        .reset_index(drop=True)
    )
    if frame.empty:
        raise HistoricalIntradayRepairError("historical_intraday_inventory_empty")
    if frame["trade_date"].str[:4].astype(int).gt(2026).any():
        raise HistoricalIntradayRepairError("historical_intraday_inventory_after_2026")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)
    return frame


def prepare_inventory(
    *,
    inventory_path: str | Path,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    runtime = _runtime(workspace)
    source = Path(inventory_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    inventory_path_out = runtime / "missing_positive_stock_days.parquet"
    frame = _copy_inventory(source, inventory_path_out)
    daily_reference = runtime / "missing_daily_reference.parquet"
    daily_paths = [
        str(item) for item in _active_domain_paths(workspace, "market_daily_raw")
    ]
    temporary = daily_reference.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    with open_guarded_duckdb(
        temp_directory=runtime / "prepare_spill",
        threads=4,
    ) as con:
        con.execute(
            f"""
            COPY (
              SELECT upper(cast(d.symbol AS VARCHAR)) AS symbol,
                     cast(d.trade_date AS VARCHAR) AS trade_date,
                     try_cast(d.open AS DOUBLE) AS open,
                     try_cast(d.high AS DOUBLE) AS high,
                     try_cast(d.low AS DOUBLE) AS low,
                     try_cast(d.close AS DOUBLE) AS close,
                     try_cast(d.volume AS DOUBLE) AS volume,
                     try_cast(d.amount AS DOUBLE) AS amount
              FROM read_parquet(?, union_by_name=true) d
              JOIN read_parquet(?) i
                ON upper(cast(d.symbol AS VARCHAR))=i.symbol
               AND cast(d.trade_date AS VARCHAR)=i.trade_date
              WHERE try_cast(d.volume AS DOUBLE)>0
              ORDER BY symbol,trade_date
            ) TO {_sql_literal(str(temporary))}
              (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """,
            [daily_paths, str(inventory_path_out)],
        )
        reference_count = int(
            con.execute(
                "SELECT count(*) FROM read_parquet(?)",
                [str(temporary)],
            ).fetchone()[0]
        )
    shutil.rmtree(runtime / "prepare_spill", ignore_errors=True)
    if reference_count != len(frame):
        temporary.unlink(missing_ok=True)
        raise HistoricalIntradayRepairError(
            "historical_intraday_daily_reference_mismatch:"
            f"{reference_count}!={len(frame)}"
        )
    os.replace(temporary, daily_reference)
    frame["month"] = frame["trade_date"].str[:7]
    state = {
        "repair_id": REPAIR_ID,
        "status": "prepared",
        "inventory_path": str(inventory_path_out),
        "inventory_sha256": _sha256(inventory_path_out),
        "daily_reference_path": str(daily_reference),
        "missing_day_count_before": len(frame),
        "missing_symbol_count_before": int(frame["symbol"].nunique()),
        "symbol_month_count_before": int(
            frame[["symbol", "month"]].drop_duplicates().shape[0]
        ),
        "preflight": {},
        "tasks": {},
    }
    _write_state(workspace, state)
    return state


def _normalize_stk_mins(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=INTRADAY_COLUMNS)
    frame = raw.rename(
        columns={
            "ts_code": "symbol",
            "trade_time": "timestamp",
            "vol": "volume",
        }
    ).copy()
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise HistoricalIntradayRepairError(
            f"stk_mins_columns_missing:{','.join(missing)}"
        )
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["symbol"] = str(symbol).upper()
    frame["trade_date"] = timestamp.dt.strftime("%Y-%m-%d")
    frame["bar_time"] = timestamp.dt.strftime("%H%M") + "00000"
    return frame


def _best_scale(
    observed: pd.Series,
    reference: pd.Series,
    candidates: Sequence[float],
) -> float:
    left = pd.to_numeric(observed, errors="coerce").to_numpy(dtype="float64")
    right = pd.to_numeric(reference, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    if not valid.any():
        return 1.0
    scores = {
        float(scale): float(
            np.median(
                np.abs(
                    np.log(
                        np.maximum(left[valid] * float(scale), 1e-12)
                        / np.maximum(right[valid], 1e-12)
                    )
                )
            )
        )
        for scale in candidates
    }
    return min(scores, key=lambda item: (scores[item], abs(np.log(item))))


def _validate_against_daily(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame, {
            "accepted_day_count": 0,
            "rejected_daily_consistency_day_count": 0,
        }
    aggregate = (
        frame.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
        )
        .merge(
            reference,
            on=["symbol", "trade_date"],
            how="inner",
            suffixes=("_five", "_day"),
            validate="one_to_one",
        )
    )
    volume_scale = _best_scale(
        aggregate["volume_five"],
        aggregate["volume_day"],
        VOLUME_SCALES,
    )
    amount_scale = _best_scale(
        aggregate["amount_five"],
        aggregate["amount_day"],
        AMOUNT_SCALES,
    )
    frame = frame.copy()
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * volume_scale
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce") * amount_scale
    for column in ("volume_five", "amount_five"):
        scale = volume_scale if column == "volume_five" else amount_scale
        aggregate[column] = aggregate[column] * scale
    price_rel = np.column_stack(
        [
            (aggregate[f"{column}_five"] - aggregate[f"{column}_day"]).abs()
            / aggregate[f"{column}_day"].abs().clip(lower=1.0)
            for column in ("open", "high", "low", "close")
        ]
    ).max(axis=1)
    volume_rel = (aggregate["volume_five"] - aggregate["volume_day"]).abs() / aggregate[
        "volume_day"
    ].abs().clip(lower=1.0)
    amount_rel = (aggregate["amount_five"] - aggregate["amount_day"]).abs() / aggregate[
        "amount_day"
    ].abs().clip(lower=1.0)
    volume_ratio = aggregate["volume_five"] / aggregate["volume_day"].replace(
        0.0,
        np.nan,
    )
    volume_100x = volume_ratio.between(95.0, 105.0) | volume_ratio.between(
        0.0095,
        0.0105,
    )
    accepted = (
        (price_rel <= 0.02)
        & ((volume_rel <= 0.05) | (amount_rel <= 0.05))
        & ~volume_100x.fillna(False)
    )
    accepted_pairs = pd.MultiIndex.from_frame(
        aggregate.loc[accepted, ["symbol", "trade_date"]]
    )
    frame_pairs = pd.MultiIndex.from_frame(frame[["symbol", "trade_date"]])
    output = frame.loc[frame_pairs.isin(accepted_pairs)].copy()
    diagnostics = {
        "accepted_day_count": int(accepted.sum()),
        "rejected_daily_consistency_day_count": int((~accepted).sum()),
        "rejected_volume_100x_day_count": int(
            volume_100x.fillna(False).sum()
        ),
        "volume_scale": float(volume_scale),
        "amount_scale": float(amount_scale),
        "maximum_accepted_price_relative_error": (
            float(price_rel[accepted].max()) if accepted.any() else None
        ),
        "maximum_accepted_volume_relative_error": (
            float(volume_rel[accepted].max()) if accepted.any() else None
        ),
        "maximum_accepted_amount_relative_error": (
            float(amount_rel[accepted].max()) if accepted.any() else None
        ),
    }
    return output.reset_index(drop=True), diagnostics


def _task_key(symbol: str, month: str) -> str:
    return f"{symbol}|{month}"


def _six_month_chunks(months: Sequence[str]) -> list[tuple[str, ...]]:
    chunks: list[list[str]] = []
    for month in sorted({str(item) for item in months}):
        if not chunks:
            chunks.append([month])
            continue
        first = pd.Period(chunks[-1][0], freq="M")
        current = pd.Period(month, freq="M")
        if int(current.ordinal - first.ordinal) >= 6:
            chunks.append([month])
        else:
            chunks[-1].append(month)
    return [tuple(item) for item in chunks]


def _period_reference(
    path: Path,
    *,
    symbol: str,
    months: Sequence[str],
) -> pd.DataFrame:
    ordered = sorted({str(item) for item in months})
    if not ordered:
        return pd.DataFrame()
    frame = pd.read_parquet(
        path,
        filters=[
            ("symbol", "==", symbol),
            ("trade_date", ">=", f"{ordered[0]}-01"),
            ("trade_date", "<=", f"{ordered[-1]}-31"),
        ],
    )
    return frame.loc[frame["trade_date"].astype(str).str[:7].isin(ordered)].copy()


def _fetch_period(
    client: _TushareClient,
    *,
    symbol: str,
    months: Sequence[str],
    reference_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    ordered_months = sorted({str(item) for item in months})
    reference = _period_reference(
        reference_path,
        symbol=symbol,
        months=ordered_months,
    )
    if reference.empty:
        raise HistoricalIntradayRepairError(
            f"historical_intraday_period_reference_empty:{symbol}:{ordered_months}"
        )
    wanted = sorted(reference["trade_date"].astype(str).unique())
    raw = client.fetch(
        "stk_mins",
        params={
            "ts_code": symbol,
            "freq": "5min",
            "start_date": f"{min(wanted)} 09:00:00",
            "end_date": f"{max(wanted)} 15:00:00",
        },
        fields=API_FIELDS,
    )
    normalized = _normalize_stk_mins(raw, symbol)
    valid, unresolved = _validate_intraday_frame(
        normalized,
        {symbol: wanted},
        source_name="tushare_compatible_stk_mins_historical_repair",
    )
    valid, daily_diagnostics = _validate_against_daily(valid, reference)
    accepted_days = int(valid["trade_date"].nunique()) if not valid.empty else 0
    if not valid.empty:
        _write_final_parquet(valid, output_path, domain=INTRADAY_DOMAIN)
    else:
        output_path.unlink(missing_ok=True)
    return {
        "status": "completed",
        "symbol": symbol,
        "months": ordered_months,
        "start_month": ordered_months[0],
        "end_month": ordered_months[-1],
        "requested_day_count": len(wanted),
        "provider_row_count": len(raw),
        "accepted_day_count": accepted_days,
        "unresolved_48_bar_day_count": len(unresolved),
        "output_path": str(output_path) if output_path.is_file() else "",
        "output_sha256": _sha256(output_path) if output_path.is_file() else "",
        **daily_diagnostics,
    }


def _fetch_month(
    client: _TushareClient,
    *,
    symbol: str,
    month: str,
    reference_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    result = _fetch_period(
        client,
        symbol=symbol,
        months=(month,),
        reference_path=reference_path,
        output_path=output_path,
    )
    return {**result, "month": month}


def run_pending(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if not state or state.get("status") == "applied":
        return state
    inventory = pd.read_parquet(state["inventory_path"])
    inventory["month"] = inventory["trade_date"].astype(str).str[:7]
    reference_path = Path(state["daily_reference_path"])
    token = _resolve_tushare_token()
    client = _TushareClient(token)
    parts = _runtime(workspace) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    preflight = dict(state.get("preflight", {}) or {})
    tasks = dict(state.get("tasks", {}) or {})
    period_tasks = dict(state.get("period_tasks", {}) or {})

    for number, (symbol, rows) in enumerate(
        inventory.groupby("symbol", sort=True),
        start=1,
    ):
        previous = dict(preflight.get(symbol, {}) or {})
        if previous.get("status") == "completed":
            continue
        month = str(rows["month"].max())
        key = _task_key(symbol, month)
        output = parts / f"{symbol.replace('.', '_')}__{month.replace('-', '')}.parquet"
        result = _fetch_month(
            client,
            symbol=symbol,
            month=month,
            reference_path=reference_path,
            output_path=output,
        )
        tasks[key] = result
        available = int(result["provider_row_count"]) > 0
        preflight[symbol] = {
            "status": "completed",
            "probe_month": month,
            "provider_available": available,
            "accepted_probe_day_count": int(result["accepted_day_count"]),
            "provider_row_count": int(result["provider_row_count"]),
        }
        state.update(
            {
                "status": "preflighting",
                "preflight": preflight,
                "tasks": tasks,
                "period_tasks": period_tasks,
            }
        )
        _write_state(workspace, state)
        if number % 25 == 0:
            print(
                json.dumps(
                    {
                        "preflight": number,
                        "symbols": int(inventory["symbol"].nunique()),
                        "available": sum(
                            bool(item.get("provider_available"))
                            for item in preflight.values()
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    available_symbols = {
        symbol
        for symbol, item in preflight.items()
        if bool(item.get("provider_available"))
    }
    completed_months: dict[str, set[str]] = {}
    for item in tasks.values():
        record = dict(item)
        if record.get("status") == "completed":
            completed_months.setdefault(str(record["symbol"]), set()).add(
                str(record["month"])
            )
    for item in period_tasks.values():
        record = dict(item)
        if record.get("status") == "completed":
            completed_months.setdefault(str(record["symbol"]), set()).update(
                str(month) for month in record.get("months", [])
            )
    pending_chunks: list[tuple[str, tuple[str, ...]]] = []
    for symbol, rows in inventory.loc[
        inventory["symbol"].isin(available_symbols)
    ].groupby("symbol", sort=True):
        months = [
            str(item)
            for item in sorted(rows["month"].unique())
            if str(item) not in completed_months.get(str(symbol), set())
        ]
        pending_chunks.extend(
            (str(symbol), chunk) for chunk in _six_month_chunks(months)
        )
    for number, (symbol, months) in enumerate(pending_chunks, start=1):
        key = f"{symbol}|{months[0]}|{months[-1]}"
        output = parts / (
            f"{symbol.replace('.', '_')}__"
            f"{months[0].replace('-', '')}-{months[-1].replace('-', '')}.parquet"
        )
        period_tasks[key] = _fetch_period(
            client,
            symbol=symbol,
            months=months,
            reference_path=reference_path,
            output_path=output,
        )
        state.update(
            {
                "status": "downloading",
                "preflight": preflight,
                "tasks": tasks,
                "period_tasks": period_tasks,
            }
        )
        _write_state(workspace, state)
        if number % 25 == 0:
            print(
                json.dumps(
                    {
                        "completed_period_tasks": number,
                        "pending_period_tasks": len(pending_chunks),
                        "accepted_days": sum(
                            int(item.get("accepted_day_count", 0) or 0)
                            for item in [
                                *tasks.values(),
                                *period_tasks.values(),
                            ]
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    state.update(
        {
            "status": "downloaded",
            "preflight": preflight,
            "tasks": tasks,
            "period_tasks": period_tasks,
            "provider_available_symbol_count": len(available_symbols),
            "provider_unavailable_symbol_count": (
                int(inventory["symbol"].nunique()) - len(available_symbols)
            ),
            "accepted_day_count": sum(
                int(item.get("accepted_day_count", 0) or 0)
                for item in [*tasks.values(), *period_tasks.values()]
            ),
        }
    )
    _write_state(workspace, state)
    return state


def _bundle_parts(workspace: Path, paths: Sequence[Path]) -> list[Path]:
    if not paths:
        return []
    runtime = _runtime(workspace)
    bundle_dir = runtime / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[Path]] = {}
    for path in paths:
        symbol = path.name.split("__", 1)[0]
        bucket = int(hashlib.sha256(symbol.encode("ascii")).hexdigest()[:8], 16) % 16
        grouped.setdefault(bucket, []).append(path)
    outputs: list[Path] = []
    with open_guarded_duckdb(
        temp_directory=runtime / "bundle_spill",
        threads=4,
    ) as con:
        for bucket, items in sorted(grouped.items()):
            target = bundle_dir / f"historical_5m_bucket_{bucket:02d}.parquet"
            temporary = target.with_suffix(".tmp.parquet")
            temporary.unlink(missing_ok=True)
            con.execute(
                f"""
                COPY (
                  SELECT symbol,trade_date,bar_time,open,high,low,close,
                         volume,amount,source,adjusted_flag
                  FROM read_parquet(?, union_by_name=true)
                  ORDER BY trade_date,symbol,bar_time
                ) TO {_sql_literal(str(temporary))} (
                  FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000
                )
                """,
                [[str(item) for item in items]],
            )
            os.replace(temporary, target)
            outputs.append(target)
    shutil.rmtree(runtime / "bundle_spill", ignore_errors=True)
    return outputs


def commit_repair(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if not state or state.get("status") == "applied":
        return state
    if state.get("status") != "downloaded":
        raise HistoricalIntradayRepairError(
            f"historical_intraday_not_downloaded:{state.get('status')}"
        )
    tasks = [
        dict(item)
        for item in [
            *dict(state.get("tasks", {})).values(),
            *dict(state.get("period_tasks", {})).values(),
        ]
    ]
    part_paths = sorted(
        {
            Path(str(item["output_path"]))
            for item in tasks
            if str(item.get("output_path", ""))
            and Path(str(item["output_path"])).is_file()
        }
    )
    bundles = _bundle_parts(workspace, part_paths)
    if bundles:
        commit = bulk_append_active_shards_from_parquet(
            INTRADAY_DOMAIN,
            bundles,
            (
                "fill exact historical positive-stock-day 5m gaps from a "
                "Tushare-compatible provider"
            ),
            workspace_root=workspace,
        )
    else:
        commit = {"status": "nothing_to_append", "appended_row_count": 0}
    recovered = int(state.get("accepted_day_count", 0) or 0)
    before = int(state.get("missing_day_count_before", 0) or 0)
    remaining = max(0, before - recovered)
    metadata = update_active_manifest_metadata(
        INTRADAY_DOMAIN,
        reason="record exact historical 5m repair coverage",
        workspace_root=workspace,
        source_updates={
            "historical_gap_repair_provider": "tushare_compatible_stk_mins",
            "historical_gap_repair_at": utc_now(),
        },
        quality_updates={
            "historical_missing_positive_days_before": before,
            "historical_recovered_positive_days": recovered,
            "historical_missing_positive_days_after": remaining,
            "historical_provider_available_symbol_count": int(
                state.get("provider_available_symbol_count", 0) or 0
            ),
            "historical_provider_unavailable_symbol_count": int(
                state.get("provider_unavailable_symbol_count", 0) or 0
            ),
            "intraday_5m_restored_for_historical_symbols": remaining == 0,
            "missing_history_is_not_an_eligibility_filter": True,
        },
    )
    state.update(
        {
            "status": "applied",
            "commit": commit,
            "metadata_commit": metadata,
            "missing_day_count_after": remaining,
        }
    )
    _write_state(workspace, state)
    audit_path = qdp_v2_root(workspace) / "audits" / f"{REPAIR_ID}.json"
    atomic_write_json(
        audit_path,
        {
            key: value
            for key, value in state.items()
            if key not in {"tasks", "period_tasks"}
        }
        | {
            "task_count": len(tasks),
            "accepted_task_count": sum(
                int(item.get("accepted_day_count", 0) or 0) > 0 for item in tasks
            ),
        },
    )
    return state


def status(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    state = _read_state(_workspace(workspace_root))
    tasks = [
        *dict(state.get("tasks", {}) or {}).values(),
        *dict(state.get("period_tasks", {}) or {}).values(),
    ]
    return {
        key: value
        for key, value in state.items()
        if key not in {"tasks", "period_tasks", "preflight"}
    } | {
        "preflight_completed": len(dict(state.get("preflight", {}) or {})),
        "task_completed": sum(
            dict(item).get("status") == "completed" for item in tasks
        ),
        "task_with_rows": sum(
            int(dict(item).get("accepted_day_count", 0) or 0) > 0 for item in tasks
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp historical-intraday-repair")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--inventory-path", default="")
    parser.add_argument(
        "--action",
        choices=("prepare", "run-pending", "commit", "status"),
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.action == "prepare":
        if not args.inventory_path:
            raise ValueError("inventory_path_required")
        payload = prepare_inventory(
            inventory_path=args.inventory_path,
            workspace_root=workspace,
        )
    elif args.action == "run-pending":
        payload = run_pending(workspace_root=workspace)
    elif args.action == "commit":
        payload = commit_repair(workspace_root=workspace)
    else:
        payload = status(workspace_root=workspace)
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
