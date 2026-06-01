from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_lake.pool_views import load_pool_view
from daily_research.data_lake.v2_status_sidecar import build_v2_status_sidecar_frame
from daily_research.data_platform.contracts import DataDomain


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ID = "policy_input_bundle__45e3d8c059ba718426a9f887"
DEFAULT_POOL_VIEW_ID = "policy_pool_view__74f45f4f83263bccd64a8027"
DEFAULT_RUN_TAG = "v2_dataset_contract_audit_20260601_01"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "daily_research/output/data_lake/audits" / DEFAULT_RUN_TAG
ACTIVE_ARTIFACT = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
AMOUNT_SENSITIVE_FEATURE_KEYWORDS = ("amount", "adv", "liquid", "turnover", "money")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _active_artifact_has_diff() -> bool:
    result = subprocess.run(["git", "diff", "--", str(ACTIVE_ARTIFACT)], check=False, capture_output=True, text=True)
    return bool(str(result.stdout or "").strip() or str(result.stderr or "").strip())


def _read_policy_bundle_market(lake: ResearchDataLake, dataset_id: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = lake.describe_dataset(dataset_id)
    path = str(dict(metadata.get("content_paths", {}) or {}).get("bronze_market_data", "") or "")
    if not path or not Path(path).exists():
        raise ValueError(f"v2_dataset_contract_blocker: missing bronze_market_data for {dataset_id}")
    market = pd.read_parquet(path)
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    start = pd.Timestamp(start_date or metadata.get("start_date", "") or market["trade_date"].min())
    end = pd.Timestamp(end_date or metadata.get("end_date", "") or market["trade_date"].max())
    market = market.loc[(market["trade_date"] >= start) & (market["trade_date"] <= end)].copy()
    market["symbol"] = market["symbol"].astype(str).str.strip().str.upper()
    return market, metadata


def _read_benchmark(metadata: Mapping[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    path = str(dict(metadata.get("content_paths", {}) or {}).get("silver_benchmark", "") or "")
    if not path or not Path(path).exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    start = pd.Timestamp(start_date or frame["trade_date"].min())
    end = pd.Timestamp(end_date or frame["trade_date"].max())
    return frame.loc[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()


def _first_existing_dataset(lake: ResearchDataLake, dataset_kind: str) -> str:
    rows = lake.list_datasets(dataset_kind=dataset_kind)
    if rows.empty:
        return ""
    sort_cols = [column for column in ("end_date", "created_at") if column in rows.columns]
    if sort_cols:
        rows = rows.sort_values(sort_cols)
    return str(rows.iloc[-1]["dataset_id"])


def _read_domain_dataset(lake: ResearchDataLake, dataset_id: str) -> pd.DataFrame:
    if not str(dataset_id or "").strip():
        return pd.DataFrame()
    metadata = lake.describe_dataset(str(dataset_id))
    path = str(dict(metadata.get("content_paths", {}) or {}).get("silver_domain_data", "") or "")
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _resolve_domain_sidecar_id(lake: ResearchDataLake, metadata: Mapping[str, Any], domain: str) -> str:
    parameters = dict(metadata.get("parameters", {}) or {})
    sidecars = dict(parameters.get("sidecar_dataset_ids", {}) or {})
    explicit = str(sidecars.get(str(domain), "") or "").strip()
    if explicit:
        return explicit
    return _first_existing_dataset(lake, f"data_platform_{domain}")


def amount_unit_diagnostics(market: pd.DataFrame) -> dict[str, Any]:
    if market is None or market.empty:
        return {"status": "blocked", "blockers": ["empty_market"]}
    data = market.copy()
    for column in ("close", "volume", "amount"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    valid = data["close"].gt(0) & data["volume"].gt(0) & data["amount"].gt(0)
    ratio = data.loc[valid, "amount"].div(data.loc[valid, "close"].mul(data.loc[valid, "volume"]).replace(0, np.nan))
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return {"status": "blocked", "blockers": ["no_valid_amount_ratio"], "valid_ratio_rows": 0}
    median = float(ratio.median())
    mean = float(ratio.mean())
    log_error = np.log10(ratio.astype(float)).abs().replace([np.inf, -np.inf], np.nan).dropna()
    factor = 1.0
    policy = "as_is"
    risk = "ok"
    if 0.5 <= median <= 2.0:
        policy = "as_is"
    elif 5000.0 <= median <= 20000.0:
        factor = 1.0 / 10000.0
        policy = "divide_by_10000"
        risk = "unit_scale_mismatch"
    elif 0.00005 <= median <= 0.0002:
        factor = 10000.0
        policy = "multiply_by_10000"
        risk = "unit_scale_mismatch"
    else:
        policy = "unknown_unit"
        risk = "unit_unknown"
    return {
        "status": "ok" if risk == "ok" else "degraded",
        "risk": risk,
        "amount_unit_policy": policy,
        "amount_unit_factor": factor,
        "valid_ratio_rows": int(len(ratio)),
        "amount_to_close_volume_median": median,
        "amount_to_close_volume_mean": mean,
        "amount_consistency_p95_abs_log_error": float(log_error.quantile(0.95)) if not log_error.empty else 0.0,
        "amount_median": float(data["amount"].dropna().median()) if data["amount"].notna().any() else 0.0,
        "amount_mean": float(data["amount"].dropna().mean()) if data["amount"].notna().any() else 0.0,
    }


def missing_fill_diagnostics(market: pd.DataFrame) -> dict[str, Any]:
    if market is None or market.empty:
        return {"status": "blocked", "blockers": ["empty_market"]}
    fields = ["open", "high", "low", "close", "volume", "amount"]
    data = market.copy()
    for field in fields:
        data[field] = pd.to_numeric(data.get(field), errors="coerce")
    ohlc = data[["open", "high", "low", "close"]]
    row_count = max(int(len(data)), 1)
    single_field_missing = data[fields].isna().any(axis=1) & ~data[fields].isna().all(axis=1)
    all_ohlc_missing = ohlc.isna().all(axis=1)
    any_ohlc_missing = ohlc.isna().any(axis=1)
    zero_volume = data["volume"].fillna(0).le(0)
    zero_amount = data["amount"].fillna(0).le(0)
    blockers: list[str] = []
    if float(any_ohlc_missing.mean()) > 0.05:
        blockers.append("ohlc_missing_rate_gt_5pct")
    status = "ok" if not blockers else "degraded"
    return {
        "status": status,
        "row_count": int(len(data)),
        "any_ohlc_missing_rows": int(any_ohlc_missing.sum()),
        "all_ohlc_missing_rows": int(all_ohlc_missing.sum()),
        "single_field_missing_rows": int(single_field_missing.sum()),
        "zero_or_missing_volume_rows": int(zero_volume.sum()),
        "zero_or_missing_amount_rows": int(zero_amount.sum()),
        "any_ohlc_missing_rate": float(any_ohlc_missing.sum() / row_count),
        "all_ohlc_missing_rate": float(all_ohlc_missing.sum() / row_count),
        "single_field_missing_rate": float(single_field_missing.sum() / row_count),
        "zero_or_missing_volume_rate": float(zero_volume.sum() / row_count),
        "zero_or_missing_amount_rate": float(zero_amount.sum() / row_count),
        "blockers": blockers,
    }


def membership_diagnostics(market: pd.DataFrame, membership: pd.DataFrame) -> dict[str, Any]:
    if membership is None or membership.empty:
        return {"status": "blocked", "blockers": ["empty_membership"]}
    data = membership.fillna(False).astype(bool)
    daily_counts = data.sum(axis=1).astype(int)
    active_symbols = [str(column) for column in data.columns if bool(data[column].any())]
    return {
        "status": "ok" if int(daily_counts.max()) > 0 else "blocked",
        "membership_rows": int(len(data)),
        "membership_columns": int(len(data.columns)),
        "active_symbol_count": int(len(active_symbols)),
        "daily_member_min": int(daily_counts.min()) if len(daily_counts) else 0,
        "daily_member_median": float(daily_counts.median()) if len(daily_counts) else 0.0,
        "daily_member_max": int(daily_counts.max()) if len(daily_counts) else 0,
        "active_excluded_prefix_counts": _prefix_counts(active_symbols, ("300", "301", "688", "689")),
        "blockers": [] if int(daily_counts.max()) > 0 else ["empty_membership"],
    }


def pool_status_screen_diagnostics(
    *,
    lake: ResearchDataLake,
    market: pd.DataFrame,
    metadata: Mapping[str, Any],
    membership: pd.DataFrame,
) -> dict[str, Any]:
    if membership is None or membership.empty:
        return {"status": "blocked", "blockers": ["empty_membership"]}
    universe_id = _resolve_domain_sidecar_id(lake, metadata, DataDomain.UNIVERSE_SNAPSHOT)
    status_id = _resolve_domain_sidecar_id(lake, metadata, DataDomain.SECURITY_STATUS)
    universe = _read_domain_dataset(lake, universe_id)
    status = _read_domain_dataset(lake, status_id)
    if universe.empty and status.empty:
        return {
            "status": "degraded",
            "risk": "status_sidecars_missing",
            "universe_snapshot_dataset_id": str(universe_id),
            "security_status_dataset_id": str(status_id),
            "active_checked_rows": 0,
            "suspected_active_rows": 0,
            "blockers": [],
        }
    status_frame, sidecar_summary = build_v2_status_sidecar_frame(
        market=market,
        universe_snapshot=universe,
        security_status=status,
    )
    if status_frame.empty:
        return {
            "status": "degraded",
            "risk": "status_sidecar_empty",
            "universe_snapshot_dataset_id": str(universe_id),
            "security_status_dataset_id": str(status_id),
            "active_checked_rows": 0,
            "suspected_active_rows": 0,
            "blockers": [],
        }
    members = membership.fillna(False).astype(bool).copy()
    members.index = pd.to_datetime(members.index, errors="coerce").strftime("%Y-%m-%d")
    members.columns = [str(column).strip().upper() for column in members.columns]
    active = members.stack()
    active = active[active.astype(bool)].reset_index()
    active.columns = ["trade_date", "symbol", "in_pool"]
    active["symbol"] = active["symbol"].astype(str).str.strip().str.upper()
    merged = active.merge(status_frame, on=["trade_date", "symbol"], how="left")
    row_count = int(len(merged))
    if row_count <= 0:
        return {
            "status": "blocked",
            "risk": "no_active_membership_rows",
            "universe_snapshot_dataset_id": str(universe_id),
            "security_status_dataset_id": str(status_id),
            "active_checked_rows": 0,
            "suspected_active_rows": 0,
            "blockers": ["no_active_membership_rows"],
        }
    flag_columns = ["is_st", "is_suspended", "is_delisted"]
    counts = {column: int(merged.get(column, pd.Series(False, index=merged.index)).fillna(False).astype(bool).sum()) for column in flag_columns}
    not_listed = int((~merged.get("is_listed_on_date", pd.Series(True, index=merged.index)).fillna(False).astype(bool)).sum())
    missing_bar = int((~merged.get("has_bar", pd.Series(True, index=merged.index)).fillna(False).astype(bool)).sum())
    not_tradeable = int((~merged.get("is_tradeable", pd.Series(True, index=merged.index)).fillna(False).astype(bool)).sum())
    reject_counts = {
        str(key): int(value)
        for key, value in merged.get("reject_reason", pd.Series("", index=merged.index)).fillna("").value_counts().to_dict().items()
        if str(key)
    }
    suspected = int(max(not_tradeable, sum(counts.values()) + not_listed + missing_bar))
    return {
        "status": "ok" if suspected == 0 else "degraded",
        "risk": "ok" if suspected == 0 else "active_pool_status_violations",
        "universe_snapshot_dataset_id": str(universe_id),
        "security_status_dataset_id": str(status_id),
        "status_sidecar_summary": sidecar_summary,
        "active_checked_rows": row_count,
        "suspected_active_rows": suspected,
        "active_is_st_rows": counts["is_st"],
        "active_is_suspended_rows": counts["is_suspended"],
        "active_is_delisted_rows": counts["is_delisted"],
        "active_not_listed_rows": not_listed,
        "active_missing_bar_rows": missing_bar,
        "active_not_tradeable_rows": not_tradeable,
        "active_reject_reason_counts": reject_counts,
        "blockers": [],
    }


def pit_status_source_contract_diagnostics(
    *,
    lake: ResearchDataLake,
    metadata: Mapping[str, Any],
    market: pd.DataFrame,
) -> dict[str, Any]:
    universe_id = _resolve_domain_sidecar_id(lake, metadata, DataDomain.UNIVERSE_SNAPSHOT)
    status_id = _resolve_domain_sidecar_id(lake, metadata, DataDomain.SECURITY_STATUS)
    universe = _read_domain_dataset(lake, universe_id)
    status = _read_domain_dataset(lake, status_id)
    market_dates = pd.to_datetime(market.get("trade_date"), errors="coerce").dropna()
    market_trade_date_count = int(market_dates.dt.normalize().nunique()) if not market_dates.empty else 0
    risks: list[str] = []
    blockers: list[str] = []

    if universe.empty:
        risks.append("missing_universe_snapshot_sidecar")
        universe_date_count = 0
        list_date_nonblank = 0
        delist_date_nonblank = 0
        universe_symbol_count = 0
        list_status_counts: dict[str, int] = {}
    else:
        universe_dates = pd.to_datetime(universe.get("trade_date"), errors="coerce").dropna()
        universe_date_count = int(universe_dates.dt.normalize().nunique()) if not universe_dates.empty else 0
        universe_symbol_count = int(universe.get("symbol", pd.Series(dtype=object)).astype(str).nunique())
        list_date = universe.get("list_date", pd.Series("", index=universe.index)).fillna("").astype(str).str.strip()
        delist_date = universe.get("delist_date", pd.Series("", index=universe.index)).fillna("").astype(str).str.strip()
        list_date_nonblank = int(list_date.ne("").sum())
        delist_date_nonblank = int(delist_date.ne("").sum())
        list_status_counts = {
            str(key): int(value)
            for key, value in universe.get("list_status", pd.Series("", index=universe.index)).fillna("").astype(str).value_counts().head(20).to_dict().items()
        }
        if market_trade_date_count > 1 and universe_date_count <= 1:
            risks.append("universe_snapshot_single_date_not_pit_daily")
        if list_date_nonblank <= 0:
            risks.append("universe_missing_list_date")
        if delist_date_nonblank <= 0:
            risks.append("universe_missing_delist_date")

    if status.empty:
        risks.append("missing_security_status_sidecar")
        status_date_count = 0
        status_symbol_count = 0
        status_source_counts: dict[str, int] = {}
        st_true_rows = 0
        suspended_true_rows = 0
        delisted_true_rows = 0
    else:
        status_dates = pd.to_datetime(status.get("trade_date"), errors="coerce").dropna()
        status_date_count = int(status_dates.dt.normalize().nunique()) if not status_dates.empty else 0
        status_symbol_count = int(status.get("symbol", pd.Series(dtype=object)).astype(str).nunique())
        status_source_counts = {
            str(key): int(value)
            for key, value in status.get("source", pd.Series("", index=status.index)).fillna("").astype(str).value_counts().head(20).to_dict().items()
        }
        st_true_rows = int(status.get("is_st", pd.Series(False, index=status.index)).fillna(False).astype(bool).sum())
        suspended_true_rows = int(status.get("is_suspended", pd.Series(False, index=status.index)).fillna(False).astype(bool).sum())
        delisted_true_rows = int(status.get("is_delisted", pd.Series(False, index=status.index)).fillna(False).astype(bool).sum())
        if market_trade_date_count > 1 and status_date_count <= 1:
            risks.append("security_status_single_date_not_pit_daily")
        if suspended_true_rows <= 0 and market_trade_date_count > 1:
            risks.append("historical_suspension_status_unavailable")
        if delisted_true_rows <= 0 and delist_date_nonblank <= 0 and market_trade_date_count > 1:
            risks.append("historical_delist_status_unavailable")

    if blockers:
        status_value = "blocked"
    elif risks:
        status_value = "degraded"
    else:
        status_value = "ok"
    return {
        "status": status_value,
        "risk": "ok" if status_value == "ok" else "pit_status_source_contract_incomplete",
        "universe_snapshot_dataset_id": str(universe_id),
        "security_status_dataset_id": str(status_id),
        "market_trade_date_count": market_trade_date_count,
        "universe_snapshot_trade_date_count": universe_date_count,
        "security_status_trade_date_count": status_date_count,
        "universe_snapshot_rows": int(len(universe)),
        "security_status_rows": int(len(status)),
        "universe_symbol_count": universe_symbol_count,
        "security_status_symbol_count": status_symbol_count,
        "list_date_nonblank_rows": list_date_nonblank,
        "delist_date_nonblank_rows": delist_date_nonblank,
        "list_status_counts": list_status_counts,
        "security_status_source_counts": status_source_counts,
        "security_status_true_rows": {
            "is_st": st_true_rows,
            "is_suspended": suspended_true_rows,
            "is_delisted": delisted_true_rows,
        },
        "risks": risks,
        "blockers": blockers,
    }


def _prefix_counts(symbols: list[str], prefixes: tuple[str, ...]) -> dict[str, int]:
    counts = {prefix: 0 for prefix in prefixes}
    for symbol in symbols:
        code = str(symbol).split(".", 1)[0]
        for prefix in prefixes:
            if code.startswith(prefix):
                counts[prefix] += 1
    return counts


def feature_contract_diagnostics(metadata: Mapping[str, Any]) -> dict[str, Any]:
    paths = dict(metadata.get("content_paths", {}) or {})
    feature_path = str(paths.get("silver_feature_values", paths.get("silver_feature_panels", "")) or "")
    feature_dir = Path(str(feature_path).replace("*.parquet", "")) if feature_path else Path()
    if str(feature_path).endswith("*.parquet"):
        feature_dir = Path(feature_path).parent
    feature_names = sorted(path.stem for path in feature_dir.glob("*.parquet")) if feature_dir.exists() else []
    sensitive = [
        name
        for name in feature_names
        if any(keyword in name.lower() for keyword in AMOUNT_SENSITIVE_FEATURE_KEYWORDS)
    ]
    expected = ["adv20", "adv_ratio_5_20"]
    for item in expected:
        if item not in sensitive and item in feature_names:
            sensitive.append(item)
    return {
        "status": "ok",
        "feature_panel_count": int(len(feature_names)),
        "amount_sensitive_feature_count": int(len(set(sensitive))),
        "amount_sensitive_features": sorted(set(sensitive)),
    }


def benchmark_diagnostics(benchmark: pd.DataFrame) -> dict[str, Any]:
    if benchmark is None or benchmark.empty:
        return {"status": "blocked", "blockers": ["missing_benchmark"]}
    close_rows = int(pd.to_numeric(benchmark.get("close"), errors="coerce").notna().sum()) if "close" in benchmark else 0
    open_rows = int(pd.to_numeric(benchmark.get("open"), errors="coerce").notna().sum()) if "open" in benchmark else 0
    blockers = []
    if close_rows <= 0:
        blockers.append("missing_benchmark_close")
    if open_rows <= 0:
        blockers.append("missing_benchmark_open")
    return {
        "status": "ok" if not blockers else "blocked",
        "rows": int(len(benchmark)),
        "close_rows": close_rows,
        "open_rows": open_rows,
        "blockers": blockers,
    }


def audit_v2_dataset_contract(
    *,
    lake: ResearchDataLake | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    pool_view_id: str = DEFAULT_POOL_VIEW_ID,
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
) -> dict[str, Any]:
    resolved_lake = lake or ResearchDataLake()
    market, metadata = _read_policy_bundle_market(resolved_lake, dataset_id, start_date, end_date)
    benchmark = _read_benchmark(metadata, start_date, end_date)
    pool = load_pool_view(lake=resolved_lake, pool_view_id=pool_view_id)
    membership = pool.membership_frame
    reports = {
        "market_daily_coverage": {
            "status": "ok" if not market.empty else "blocked",
            "row_count": int(len(market)),
            "trade_date_count": int(market["trade_date"].nunique()) if not market.empty else 0,
            "symbol_count": int(market["symbol"].nunique()) if not market.empty else 0,
            "date_min": pd.Timestamp(market["trade_date"].min()).strftime("%Y-%m-%d") if not market.empty else "",
            "date_max": pd.Timestamp(market["trade_date"].max()).strftime("%Y-%m-%d") if not market.empty else "",
        },
        "benchmark_coverage": benchmark_diagnostics(benchmark),
        "membership_coverage": membership_diagnostics(market, membership),
        "pool_status_screen": pool_status_screen_diagnostics(
            lake=resolved_lake,
            market=market,
            metadata=metadata,
            membership=membership,
        ),
        "pit_status_source_contract": pit_status_source_contract_diagnostics(
            lake=resolved_lake,
            metadata=metadata,
            market=market,
        ),
        "amount_unit": amount_unit_diagnostics(market),
        "missing_fill": missing_fill_diagnostics(market),
        "feature_contract": feature_contract_diagnostics(metadata),
    }
    blockers: list[str] = []
    degraded: list[str] = []
    for name, report in reports.items():
        status = str(report.get("status", "ok"))
        if status == "blocked":
            blockers.append(str(name))
        elif status == "degraded":
            degraded.append(str(name))
    if blockers:
        verdict = "v2_dataset_contract_blocked"
    elif degraded:
        verdict = "v2_dataset_contract_degraded"
    else:
        verdict = "v2_dataset_contract_ok"
    return {
        "schema_version": 1,
        "status": verdict,
        "created_at": _now(),
        "dataset_id": str(dataset_id),
        "pool_view_id": str(pool_view_id),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "reports": reports,
        "blockers": blockers,
        "degraded": degraded,
        "boundary": {
            "research_only": True,
            "training_launched": False,
            "active_execution_artifact_expected_diff": "none",
        },
    }


def write_audit_outputs(payload: Mapping[str, Any], output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "v2_dataset_contract_audit.json", payload)
    lines = [
        "# V2 Dataset Contract Audit",
        "",
        f"- status: `{payload.get('status')}`",
        f"- dataset_id: `{payload.get('dataset_id')}`",
        f"- pool_view_id: `{payload.get('pool_view_id')}`",
        f"- blockers: `{payload.get('blockers')}`",
        f"- degraded: `{payload.get('degraded')}`",
        "",
    ]
    (root / "v2_dataset_contract_audit.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit daily_research v2 dataset contract quality.")
    parser.add_argument("--data-lake-root", default="")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--pool-view-id", default=DEFAULT_POOL_VIEW_ID)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _active_artifact_has_diff():
        raise ValueError(f"active_artifact_diff_blocker: {ACTIVE_ARTIFACT} has uncommitted diff.")
    lake = ResearchDataLake(str(args.data_lake_root or "").strip() or None)
    payload = audit_v2_dataset_contract(
        lake=lake,
        dataset_id=str(args.dataset_id),
        pool_view_id=str(args.pool_view_id),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
    )
    write_audit_outputs(payload, args.output_root)
    if args.json:
        print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(f"status={payload.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
