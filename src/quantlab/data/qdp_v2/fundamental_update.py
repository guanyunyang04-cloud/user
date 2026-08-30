from __future__ import annotations

"""Build compact point-in-time quarterly fundamentals from announcement data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import atomic_copy_file, json_safe
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.auxiliary_update.context import _resolve_tushare_token, _TushareClient
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.status import active_dataset_map

UPDATE_ID = "fundamental_quarterly_pit_v1"
FINANCIAL_DOMAIN = "financial_quarterly"
FORECAST_DOMAIN = "performance_forecast"
FINANCIAL_FIELDS = (
    "ts_code",
    "ann_date",
    "end_date",
    "eps",
    "roe",
    "grossprofit_margin",
    "netprofit_margin",
    "netprofit_yoy",
    "or_yoy",
    "assets_turn",
    "debt_to_assets",
    "current_ratio",
    "ocfps",
    "update_flag",
)
FORECAST_FIELDS = (
    "ts_code",
    "ann_date",
    "end_date",
    "type",
    "p_change_min",
    "p_change_max",
    "net_profit_min",
    "net_profit_max",
    "summary",
    "change_reason",
)
FINANCIAL_COLUMNS = (
    "symbol",
    "trade_date",
    "report_date",
    "fiscal_year",
    "fiscal_quarter",
    "publish_date",
    "roe_avg",
    "net_profit_margin",
    "gross_profit_margin",
    "net_profit_yoy",
    "revenue_yoy",
    "eps",
    "net_profit",
    "revenue",
    "asset_turnover",
    "debt_to_asset",
    "current_ratio",
    "cash_flow_ps",
    "lag_policy",
    "source",
)
FORECAST_COLUMNS = (
    "symbol",
    "trade_date",
    "report_date",
    "fiscal_year",
    "fiscal_quarter",
    "publish_date",
    "forecast_type",
    "profit_min",
    "profit_max",
    "profit_change_min",
    "profit_change_max",
    "lag_policy",
    "source",
)


class FundamentalUpdateError(RuntimeError):
    pass


def _workspace(value: str | Path | None) -> Path:
    return Path(value or Path.cwd()).resolve()


def _runtime(workspace: Path) -> Path:
    path = (qdp_paths(workspace).data_dir / "qdp_runtime" / UPDATE_ID).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(workspace: Path) -> Path:
    return _runtime(workspace) / "state.json"


def _read_state(workspace: Path) -> dict[str, Any]:
    path = _state_path(workspace)
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_state(workspace: Path, state: dict[str, Any]) -> None:
    atomic_write_json(_state_path(workspace), {**state, "updated_at": utc_now()})


def _quarter_ends(target_date: str) -> list[str]:
    target = pd.Timestamp(target_date).normalize()
    return [
        item.strftime("%Y%m%d")
        for item in pd.date_range("2010-03-31", target, freq="QE-DEC")
    ]


def _identity_symbols(workspace: Path) -> set[str]:
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    dataset_id = active_dataset_map(active)["security_identity"]
    path = dataset_manifest_for_id(root, dataset_id, "security_identity")
    if path is None:
        raise FundamentalUpdateError("security_identity_manifest_missing")
    manifest = read_dataset_manifest(path)
    frames = [
        pd.read_parquet(
            resolve_manifest_path(item.path, root=root),
            columns=["current_symbol"],
        )
        for item in manifest.shards
    ]
    return {
        str(item).upper()
        for item in pd.concat(frames, ignore_index=True)["current_symbol"]
        if str(item).strip()
    }


def _date_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame.get(column), errors="coerce").dt.strftime("%Y-%m-%d")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _normalize_financial(
    raw: pd.DataFrame,
    *,
    identity_symbols: set[str],
    target_date: str,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=FINANCIAL_COLUMNS)
    data = raw.copy()
    data["symbol"] = data["ts_code"].fillna("").astype(str).str.upper()
    data["publish_date"] = _date_series(data, "ann_date")
    data["report_date"] = _date_series(data, "end_date")
    report = pd.to_datetime(data["report_date"], errors="coerce")
    data["fiscal_year"] = report.dt.year
    data["fiscal_quarter"] = report.dt.quarter
    data["trade_date"] = data["publish_date"]
    mappings = {
        "roe_avg": "roe",
        "net_profit_margin": "netprofit_margin",
        "gross_profit_margin": "grossprofit_margin",
        "net_profit_yoy": "netprofit_yoy",
        "revenue_yoy": "or_yoy",
        "eps": "eps",
        "asset_turnover": "assets_turn",
        "debt_to_asset": "debt_to_assets",
        "current_ratio": "current_ratio",
        "cash_flow_ps": "ocfps",
    }
    for target, source in mappings.items():
        data[target] = _numeric(data, source)
    data["net_profit"] = np.nan
    data["revenue"] = np.nan
    data["lag_policy"] = "publish_date_plus_1d_in_features"
    data["source"] = "tushare_compatible_fina_indicator_vip"
    data["_completeness"] = data.loc[:, list(mappings)].notna().sum(axis=1)
    data["_update_flag"] = _numeric(data, "update_flag").fillna(0)
    publish_ts = pd.to_datetime(data["publish_date"], errors="coerce")
    report_ts = pd.to_datetime(data["report_date"], errors="coerce")
    valid = (
        data["symbol"].isin(identity_symbols)
        & publish_ts.notna()
        & publish_ts.le(pd.Timestamp(target_date))
        & report_ts.notna()
        & report_ts.le(publish_ts)
    )
    return (
        data.loc[valid]
        .sort_values(
            [
                "symbol",
                "publish_date",
                "report_date",
                "_update_flag",
                "_completeness",
            ],
            kind="stable",
        )
        .drop_duplicates(
            ["symbol", "publish_date", "report_date"],
            keep="last",
        )
        .loc[:, list(FINANCIAL_COLUMNS)]
        .sort_values(["trade_date", "symbol", "report_date"], kind="stable")
        .reset_index(drop=True)
    )


def _normalize_forecast(
    raw: pd.DataFrame,
    *,
    identity_symbols: set[str],
    target_date: str,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=FORECAST_COLUMNS)
    data = raw.copy()
    data["symbol"] = data["ts_code"].fillna("").astype(str).str.upper()
    data["publish_date"] = _date_series(data, "ann_date")
    data["report_date"] = _date_series(data, "end_date")
    report = pd.to_datetime(data["report_date"], errors="coerce")
    data["fiscal_year"] = report.dt.year
    data["fiscal_quarter"] = report.dt.quarter
    data["trade_date"] = data["publish_date"]
    data["forecast_type"] = data.get("type", "").fillna("").astype(str)
    data["profit_min"] = _numeric(data, "net_profit_min")
    data["profit_max"] = _numeric(data, "net_profit_max")
    data["profit_change_min"] = _numeric(data, "p_change_min")
    data["profit_change_max"] = _numeric(data, "p_change_max")
    data["lag_policy"] = "publish_date_plus_1d_in_features"
    data["source"] = "tushare_compatible_forecast_vip"
    publish_ts = pd.to_datetime(data["publish_date"], errors="coerce")
    report_ts = pd.to_datetime(data["report_date"], errors="coerce")
    valid = (
        data["symbol"].isin(identity_symbols)
        & publish_ts.notna()
        & publish_ts.le(pd.Timestamp(target_date))
        & report_ts.notna()
    )
    return (
        data.loc[valid]
        .sort_values(
            ["symbol", "publish_date", "report_date"],
            kind="stable",
        )
        .drop_duplicates(
            ["symbol", "publish_date", "report_date"],
            keep="last",
        )
        .loc[:, list(FORECAST_COLUMNS)]
        .sort_values(["trade_date", "symbol", "report_date"], kind="stable")
        .reset_index(drop=True)
    )


def _write_part(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def download(
    *,
    target_date: str,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") in {"downloaded", "applied"}:
        return state
    target = pd.Timestamp(target_date).strftime("%Y-%m-%d")
    periods = _quarter_ends(target)
    token = _resolve_tushare_token(workspace)
    client = _TushareClient(token, workspace_root=workspace)
    runtime = _runtime(workspace)
    raw_dir = runtime / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    completed = dict(state.get("completed", {}) or {})
    for number, period in enumerate(periods, start=1):
        financial_path = raw_dir / f"financial_{period}.parquet"
        forecast_path = raw_dir / f"forecast_{period}.parquet"
        if not financial_path.is_file():
            _write_part(
                client.fetch(
                    "fina_indicator_vip",
                    params={"period": period},
                    fields=FINANCIAL_FIELDS,
                ),
                financial_path,
            )
        if not forecast_path.is_file():
            _write_part(
                client.fetch(
                    "forecast_vip",
                    params={"period": period},
                    fields=FORECAST_FIELDS,
                ),
                forecast_path,
            )
        completed[period] = {
            "status": "completed",
            "financial_path": str(financial_path),
            "financial_rows": int(pq.ParquetFile(financial_path).metadata.num_rows),
            "forecast_path": str(forecast_path),
            "forecast_rows": int(pq.ParquetFile(forecast_path).metadata.num_rows),
        }
        state = {
            "update_id": UPDATE_ID,
            "status": "downloading",
            "target_date": target,
            "period_count": len(periods),
            "completed": completed,
        }
        _write_state(workspace, state)
        if number % 8 == 0:
            print(
                json.dumps({"completed_periods": number, "period_count": len(periods)}),
                flush=True,
            )
    identity = _identity_symbols(workspace)
    financial = _normalize_financial(
        pd.concat(
            [pd.read_parquet(item["financial_path"]) for item in completed.values()],
            ignore_index=True,
        ),
        identity_symbols=identity,
        target_date=target,
    )
    forecast = _normalize_forecast(
        pd.concat(
            [pd.read_parquet(item["forecast_path"]) for item in completed.values()],
            ignore_index=True,
        ),
        identity_symbols=identity,
        target_date=target,
    )
    prepared = runtime / "prepared"
    financial_path = prepared / "financial_quarterly.parquet"
    forecast_path = prepared / "performance_forecast.parquet"
    _write_part(financial, financial_path)
    _write_part(forecast, forecast_path)
    state.update(
        {
            "status": "downloaded",
            "financial_path": str(financial_path),
            "financial_row_count": len(financial),
            "forecast_path": str(forecast_path),
            "forecast_row_count": len(forecast),
            "identity_symbol_count": len(identity),
        }
    )
    _write_state(workspace, state)
    return state


def _install_domain(
    workspace: Path,
    *,
    domain: str,
    prepared: Path,
    contract_version: str,
    primary_key: list[str],
    target_date: str,
) -> tuple[str, dict[str, Any]]:
    root = qdp_v2_root(workspace)
    digest = hashlib.sha256(prepared.read_bytes()).hexdigest()[:24]
    dataset_id = f"{domain}__{digest}"
    dataset_dir = root / "datasets" / domain / dataset_id
    shard = dataset_dir / "shards" / "part-0000.parquet"
    if not shard.is_file():
        atomic_copy_file(prepared, shard)
    parquet = pq.ParquetFile(shard)
    table = pd.read_parquet(
        shard,
        columns=["trade_date", "symbol", "report_date"],
    )
    duplicate_count = int(table.duplicated(primary_key).sum())
    future_count = int(table["trade_date"].astype(str).gt(target_date).sum())
    if duplicate_count or future_count:
        raise FundamentalUpdateError(
            f"fundamental_contract_failed:{domain}:"
            f"duplicates={duplicate_count}:future={future_count}"
        )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain=domain,
        layer="raw",
        frequency="quarterly_event",
        contract_version=contract_version,
        primary_key=primary_key,
        start_date=str(table["trade_date"].min()),
        end_date=str(table["trade_date"].max()),
        row_count=int(parquet.metadata.num_rows),
        shards=[
            ShardManifestEntry(
                path=str(shard.relative_to(root)).replace("\\", "/"),
                row_count=int(parquet.metadata.num_rows),
                start_date=str(table["trade_date"].min()),
                end_date=str(table["trade_date"].max()),
                file_size=shard.stat().st_size,
                metadata={"source_path": str(prepared)},
            )
        ],
        source={
            "provider": "tushare_compatible_announcement_api",
            "checked_through": target_date,
            "scope": "point_in_time_historical_mainboard",
            "availability_semantics": "announcement_date; consume from next day",
            "credential_persisted": False,
        },
        quality={
            "primary_key_unique": True,
            "future_announcement_rows": 0,
            "strict_point_in_time": True,
            "scope": "point_in_time_historical_mainboard",
            "null_values_are_provider_missing_not_zero": True,
        },
        schema=_manifest_schema_from_arrow(pq.read_schema(shard)),
        notes=[
            "event-level table; downstream daily features must apply the recorded lag_policy"
        ],
    )
    write_dataset_manifest(root, manifest)
    return dataset_id, {
        "dataset_id": dataset_id,
        "row_count": manifest.row_count,
        "start_date": manifest.start_date,
        "end_date": manifest.end_date,
        "duplicate_count": duplicate_count,
        "future_count": future_count,
    }


def commit(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    state = _read_state(workspace)
    if state.get("status") == "applied":
        return state
    if state.get("status") != "downloaded":
        raise FundamentalUpdateError(
            f"fundamental_not_downloaded:{state.get('status')}"
        )
    target = str(state["target_date"])
    financial_id, financial = _install_domain(
        workspace,
        domain=FINANCIAL_DOMAIN,
        prepared=Path(state["financial_path"]),
        contract_version="qdp_v2_financial_quarterly_announcement_pit_v1",
        primary_key=["trade_date", "symbol", "report_date"],
        target_date=target,
    )
    forecast_id, forecast = _install_domain(
        workspace,
        domain=FORECAST_DOMAIN,
        prepared=Path(state["forecast_path"]),
        contract_version="qdp_v2_performance_forecast_announcement_pit_v1",
        primary_key=["trade_date", "symbol", "report_date"],
        target_date=target,
    )
    root = qdp_v2_root(workspace)
    active = read_active_manifest(root)
    active["datasets"] = {
        **dict(active.get("datasets", {}) or {}),
        FINANCIAL_DOMAIN: financial_id,
        FORECAST_DOMAIN: forecast_id,
    }
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    state.update(
        {
            "status": "applied",
            "domains": {
                FINANCIAL_DOMAIN: financial,
                FORECAST_DOMAIN: forecast,
            },
        }
    )
    _write_state(workspace, state)
    atomic_write_json(root / "audits" / f"{UPDATE_ID}.json", state)
    return state


def status(
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    state = _read_state(_workspace(workspace_root))
    return {key: value for key, value in state.items() if key not in {"completed"}} | {
        "completed_period_count": len(dict(state.get("completed", {}) or {}))
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp fundamental-update")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--target-date", default="2026-07-21")
    parser.add_argument(
        "--action",
        choices=("download", "commit", "status"),
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.action == "download":
        payload = download(
            target_date=str(args.target_date),
            workspace_root=workspace,
        )
    elif args.action == "commit":
        payload = commit(workspace_root=workspace)
    else:
        payload = status(workspace_root=workspace)
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
