"""Outcome-blind inference for the frozen full-market D3 candidate.

The live path deliberately rebuilds only the 314 market/path fields that remain
available in the active QDP snapshot.  The other 243 fields are set to NaN, as
precommitted by the source-availability stress test.  This module never opens a
future-return, fill, or execution-label array.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import pickle
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import torch

from daily_research.path_policy import seq100_full_market_multitask_forecast as base
from daily_research.path_policy import seq100_full_market_sequence_challenger as seq
from daily_research.path_policy import seq100_quality_liquidity_data_prep as prep
from daily_research.path_policy import seq100_signal_quality as signal

SCHEMA = "seq100_full_market_live_inference/1"
VALIDATION_SCHEMA = "seq100_full_market_live_feature_validation/1"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research/data/research_store/seq100_pit_l35v2_v1/pack/manifest.json"
)
DEFAULT_MODEL_MANIFEST = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies/"
    "seq100_quality_liquidity_training_ready/model_inputs/manifest.json"
)
DEFAULT_OUTPUT_ROOT = base.DEFAULT_OUTPUT_ROOT / "live_inference"
DEFAULT_BUNDLE_MANIFEST = (
    base.DEFAULT_OUTPUT_ROOT / "forward_policy_bundle_v1/manifest.json"
)
DEFAULT_CORE_STRESS_MANIFEST = (
    base.DEFAULT_OUTPUT_ROOT / "core_source_availability_stress/manifest.json"
)
DEFAULT_CALENDAR_INPUT_MANIFEST = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies/"
    "seq100_quality_liquidity_data_prep/inputs/calendar_inputs_manifest.json"
)
HISTORICAL_VALIDATION_DATE = "2025-12-31"
FEATURE_WINDOWS = (5, 10, 20, 40, 60)
FEATURE_RETURN_HORIZONS = (1, 2, 5, 10, 20, 40, 60)
MARKET_GROUPS = ("all", "csi300", "csi500", "sse50")
MARKET_METRICS = (
    "ret1_mean",
    "ret5_mean",
    "ret20_mean",
    "vol20_mean",
    "drawdown60_mean",
    "breadth_ret1_positive",
    "breadth_ret5_positive",
    "breadth_above_ma20",
    "ret1_dispersion",
    "log_total_amount",
    "suspended_rate",
    "st_rate",
    "up_limit_rate",
    "down_limit_rate",
)
INDUSTRY_FEATURES = (
    "industry_ret1_mean",
    "industry_ret5_mean",
    "industry_breadth_ret1_positive",
    "industry_ret1_dispersion",
    "industry_relative_ret5",
    "industry_relative_ret20",
    "industry_member_count_log",
    "industry_source_age_days",
)
SIZE_STATUS_FEATURES = (
    "log_total_market_value",
    "log_circulating_market_value",
    "circulating_market_value_ratio",
    "signed_log_pe",
    "signed_log_pb",
    "log_turnover_rate",
    "log_total_share",
    "log_float_share",
    "float_share_ratio",
    "share_source_age_days",
    "listing_age_open_days",
    "is_csi300_member",
    "is_csi500_member",
    "is_sse50_member",
    "st_rate_20d",
    "suspended_rate_20d",
    "valuation_missing",
    "industry_missing",
)
MINUTE_FEATURES = tuple(prep.MINUTE_FEATURES)
QUALITY_METRICS = tuple(prep.QUALITY_METRICS)
CORE_FAMILIES = (
    "daily_price_volume_technical",
    "daily_cross_sectional_technical",
    "market_state",
    "industry_context",
    "size_liquidity_and_status",
    "same_day_5m",
)
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class LiveInferenceError(RuntimeError):
    """Raised when the frozen inference contract cannot be reproduced."""


@dataclass(frozen=True)
class SnapshotContract:
    pack: dict[str, Any]
    model: dict[str, Any]
    bundle: dict[str, Any]
    core_stress: dict[str, Any]
    feature_names: tuple[str, ...]
    feature_records: dict[str, dict[str, Any]]
    core_feature_names: tuple[str, ...]
    masked_feature_names: tuple[str, ...]


@dataclass
class PackArrays:
    daily_raw: np.memmap
    daily_state: np.memmap
    turnover: np.memmap
    candidate_eligible: np.memmap
    pit_universe_has_bar: np.memmap
    status_valid: np.memmap
    is_st: np.memmap
    is_suspended: np.memmap


@dataclass
class FeatureSnapshot:
    signal_date: str
    date_idx: int
    symbols: np.ndarray
    symbol_indices: np.ndarray
    names: np.ndarray
    static: np.ndarray
    market: np.ndarray
    sequence: np.ndarray
    membership: pd.DataFrame
    source_profile: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LiveInferenceError(f"required_json_missing:{path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "sha256": _sha256(path),
        **extra,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False, compression="zstd")
    os.replace(partial, path)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def daily_feature_names() -> tuple[str, ...]:
    names = [
        "open_gap_1d",
        "high_ret_prev_close_1d",
        "low_ret_prev_close_1d",
        "close_ret_1d",
        "intraday_range_1d",
        "body_to_range_1d",
        "upper_shadow_to_range_1d",
        "lower_shadow_to_range_1d",
        "close_location_1d",
        "log_amount_1d",
        "log_volume_1d",
        "log_turnover_pct_1d",
        "relative_turnover_20d",
    ]
    names.extend(f"return_{horizon}d" for horizon in FEATURE_RETURN_HORIZONS)
    per_window = (
        "ma_distance",
        "volatility",
        "downside_volatility",
        "atr",
        "trend_slope",
        "trend_t",
        "trend_r2",
        "trend_residual",
        "efficiency_ratio",
        "price_range_position",
        "high_age_fraction",
        "low_age_fraction",
        "up_day_ratio",
        "down_day_ratio",
        "price_amount_correlation",
        "amount_ratio",
        "volume_ratio",
    )
    for window in FEATURE_WINDOWS:
        names.extend(f"{name}_{window}d" for name in per_window)
    return tuple(names)


def _load_contract(
    *,
    pack_manifest: Path = DEFAULT_PACK_MANIFEST,
    model_manifest: Path = DEFAULT_MODEL_MANIFEST,
    bundle_manifest: Path = DEFAULT_BUNDLE_MANIFEST,
    core_stress_manifest: Path = DEFAULT_CORE_STRESS_MANIFEST,
) -> SnapshotContract:
    pack = _read_json(pack_manifest)
    model = _read_json(model_manifest)
    bundle = _read_json(bundle_manifest)
    stress = _read_json(core_stress_manifest)
    if bundle.get("status") != "completed":
        raise LiveInferenceError("final_bundle_not_completed")
    if (
        stress.get("status") != "completed"
        or stress.get("availability_profile") != "rebuildable_market_path_core"
        or not bool(stress.get("availability_gate", {}).get("passed"))
    ):
        raise LiveInferenceError("core_source_availability_gate_not_passed")
    feature_names = tuple(model["feature_groups"]["compact_core"])
    records = {str(row["feature_name"]): dict(row) for row in model["features"]}
    if len(feature_names) != 557 or set(feature_names) != set(records):
        raise LiveInferenceError("frozen_557_feature_contract_changed")
    core = tuple(
        name
        for name in feature_names
        if str(records[name]["analytic_family"]) in CORE_FAMILIES
    )
    masked = tuple(str(name) for name in stress["masked_features"])
    if (
        len(core) != 314
        or len(masked) != 243
        or set(core).intersection(masked)
        or set(core).union(masked) != set(feature_names)
    ):
        raise LiveInferenceError("core_314_mask_243_contract_changed")
    if tuple(feature_names[:314]) != core or tuple(feature_names[314:]) != masked:
        raise LiveInferenceError("core_features_are_not_the_frozen_prefix")
    expected_counts = {
        "daily_price_volume_technical": 104,
        "daily_cross_sectional_technical": 105,
        "market_state": 54,
        "industry_context": 8,
        "size_liquidity_and_status": 18,
        "same_day_5m": 25,
    }
    for family, expected in expected_counts.items():
        observed = sum(str(records[name]["analytic_family"]) == family for name in core)
        if observed != expected:
            raise LiveInferenceError(f"core_family_count_changed:{family}:{observed}")
    return SnapshotContract(
        pack=pack,
        model=model,
        bundle=bundle,
        core_stress=stress,
        feature_names=feature_names,
        feature_records=records,
        core_feature_names=core,
        masked_feature_names=masked,
    )


def _open_memmap(record: Mapping[str, Any], dtype: np.dtype[Any]) -> np.memmap:
    path = Path(str(record["path"]))
    shape = tuple(int(value) for value in record["shape"])
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if not path.is_file() or int(path.stat().st_size) != expected:
        raise LiveInferenceError(f"invalid_memmap:{path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _open_pack_arrays(pack: Mapping[str, Any]) -> PackArrays:
    channels = dict(pack["feature_channels"])
    masks = dict(pack["masks"])
    return PackArrays(
        daily_raw=_open_memmap(channels["daily_raw"], np.float32),
        daily_state=_open_memmap(channels["daily_state"], np.float32),
        turnover=_open_memmap(channels["turnover"], np.float32),
        candidate_eligible=_open_memmap(masks["candidate_eligible"], np.bool_),
        pit_universe_has_bar=_open_memmap(masks["pit_universe_has_bar"], np.bool_),
        status_valid=_open_memmap(masks["status_valid"], np.bool_),
        is_st=_open_memmap(masks["is_st"], np.bool_),
        is_suspended=_open_memmap(masks["is_suspended"], np.bool_),
    )


def resolve_signal_date(
    pack: Mapping[str, Any],
    *,
    active_as_of_date: str,
    requested: str | None,
) -> tuple[str, int]:
    dates = tuple(str(value) for value in pack["date_values"])
    signal_date = str(requested or min(str(active_as_of_date), dates[-1]))
    if not DATE_PATTERN.fullmatch(signal_date):
        raise LiveInferenceError(f"invalid_signal_date:{signal_date}")
    if signal_date > str(active_as_of_date):
        raise LiveInferenceError("signal_date_after_active_qdp_snapshot")
    try:
        date_idx = dates.index(signal_date)
    except ValueError as exc:
        raise LiveInferenceError(f"signal_date_absent_from_pack:{signal_date}") from exc
    if date_idx < max(FEATURE_WINDOWS):
        raise LiveInferenceError("signal_date_lacks_daily_lookback")
    return signal_date, int(date_idx)


def _connect(output_root: Path) -> tuple[duckdb.DuckDBPyConnection, dict[str, Any]]:
    memory = psutil.virtual_memory()
    reserve = 1 << 30
    if int(memory.available) <= reserve + (512 << 20):
        raise LiveInferenceError("insufficient_memory_for_live_snapshot")
    budget = min(int(memory.available) - reserve, 8 << 30)
    threads = min(int(os.cpu_count() or 1), 16)
    temporary = (output_root / "duckdb_tmp").resolve()
    temporary.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads={threads}")
    connection.execute(f"SET memory_limit='{max(budget >> 20, 512)}MB'")
    connection.execute(
        f"SET temp_directory='{str(temporary).replace(chr(39), chr(39) * 2)}'"
    )
    connection.execute("SET preserve_insertion_order=false")
    return connection, {
        "available_bytes_at_start": int(memory.available),
        "reserve_bytes": reserve,
        "duckdb_memory_limit_bytes": budget,
        "cpu_threads": threads,
    }


def _quality_membership_sql(*, signal_date: str, scans: Mapping[str, str]) -> str:
    buffer_start = (pd.Timestamp(signal_date) - pd.Timedelta(days=60)).strftime(
        "%Y-%m-%d"
    )
    rank_columns = ",\n".join(
        f"      {prep._pct_rank_sql(metric, 'quality_group', descending=metric == 'debt_to_asset')} AS {metric}_quality_rank,\n"
        f"      {prep._pct_rank_sql(metric, 'trade_date', descending=metric == 'debt_to_asset')} AS {metric}_global_rank"
        for metric in QUALITY_METRICS
    )
    quality_values = ",".join(f"{metric}_rank" for metric in QUALITY_METRICS)
    chosen = ",\n".join(
        f"      CASE WHEN quality_group='__ALL__' THEN {metric}_global_rank "
        f"ELSE {metric}_quality_rank END AS {metric}_rank"
        for metric in QUALITY_METRICS
    )
    status_eligible = prep._status_eligible_sql()
    expected_times = prep._expected_time_sql()
    return f"""
    WITH daily_window AS (
      SELECT symbol,trade_date,try_cast(amount AS DOUBLE) AS amount,
             count(*) FILTER (WHERE try_cast(amount AS DOUBLE)>0) OVER (
               PARTITION BY symbol ORDER BY trade_date
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
             ) AS valid_amount_20,
             quantile_cont(try_cast(amount AS DOUBLE),0.5) FILTER (
               WHERE try_cast(amount AS DOUBLE)>0
             ) OVER (
               PARTITION BY symbol ORDER BY trade_date
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
             ) AS amount_median_20
      FROM {scans["market_daily_raw"]}
      WHERE trade_date BETWEEN '{buffer_start}' AND '{signal_date}'
    ), financial_event AS (
      SELECT symbol,publish_date,report_date,{",".join(QUALITY_METRICS)}
      FROM {scans["financial_quarterly"]}
      WHERE publish_date<'{signal_date}'
      QUALIFY row_number() OVER (
        PARTITION BY symbol,publish_date ORDER BY report_date DESC
      )=1
    ), candidate_financial AS (
      SELECT c.*,f.* EXCLUDE(symbol)
      FROM candidate_current c ASOF LEFT JOIN financial_event f
        ON c.symbol=f.symbol AND c.trade_date>f.publish_date
    ), minute_complete AS (
      SELECT symbol,count(*) AS bar_count,count(DISTINCT bar_time) AS distinct_bar_count,
             count(*) FILTER (WHERE bar_time IN ({expected_times})) AS expected_bar_count
      FROM {scans["market_intraday_5m"]}
      WHERE trade_date='{signal_date}'
      GROUP BY symbol
    ), joined AS (
      SELECT c.*,u.name,u.list_status,u.list_date,
             s.is_st,s.is_suspended,s.is_delisted,
             try_cast(v.total_mv AS DOUBLE) AS total_mv,
             try_cast(v.circ_mv AS DOUBLE) AS circ_mv,
             try_cast(v.pe AS DOUBLE) AS pe,
             try_cast(v.pb AS DOUBLE) AS pb,
             try_cast(v.turnover_rate AS DOUBLE) AS turnover_rate,
             CASE WHEN coalesce(sc.total_share_source_date,'')<>''
                    AND sc.total_share_source_date<=c.trade_date
                  THEN try_cast(sc.total_share AS DOUBLE) END AS total_share,
             CASE WHEN coalesce(sc.float_share_source_date,'')<>''
                    AND sc.float_share_source_date<=c.trade_date
                  THEN try_cast(sc.float_share AS DOUBLE) END AS float_share,
             CASE WHEN coalesce(sc.float_share_source_date,'')<>''
                    AND sc.float_share_source_date<=c.trade_date
                  THEN sc.float_share_source_date END AS float_share_source_date,
             coalesce(i.industry_name,i.industry,'Unknown') AS quality_industry,
             i.industry,i.industry_source_date,
             d.valid_amount_20,d.amount_median_20,
             (SELECT count(*) FROM {scans["trading_calendar"]} cal
               WHERE cal.is_open
                 AND cal.trade_date BETWEEN u.list_date AND c.trade_date
             ) AS listed_open_days,
             coalesce(m.bar_count=48 AND m.distinct_bar_count=48
                      AND m.expected_bar_count=48,false) AS minute_complete
      FROM candidate_financial c
      LEFT JOIN {scans["universe_snapshot"]} u
        ON u.symbol=c.symbol AND u.trade_date=c.trade_date
      LEFT JOIN {scans["security_status"]} s
        ON s.symbol=c.symbol AND s.trade_date=c.trade_date
      LEFT JOIN {scans["valuation"]} v
        ON v.symbol=c.symbol AND v.trade_date=c.trade_date
      LEFT JOIN {scans["share_capital"]} sc
        ON sc.symbol=c.symbol AND sc.trade_date=c.trade_date
      LEFT JOIN {scans["industry_concept"]} i
        ON i.symbol=c.symbol AND i.trade_date=c.trade_date
      LEFT JOIN daily_window d
        ON d.symbol=c.symbol AND d.trade_date=c.trade_date
      LEFT JOIN minute_complete m ON m.symbol=c.symbol
    ), cross_rank AS (
      SELECT *,
        {prep._pct_rank_sql("amount_median_20", "trade_date")} AS amount_median_rank,
        {prep._pct_rank_sql("circ_mv", "trade_date")} AS circ_mv_rank,
        count(*) OVER (PARTITION BY trade_date,quality_industry) AS industry_group_count
      FROM joined
    ), quality_grouped AS (
      SELECT *,CASE WHEN industry_group_count>=20
        AND lower(quality_industry) NOT IN ('unknown','unavailable')
        THEN quality_industry ELSE '__ALL__' END AS quality_group
      FROM cross_rank
    ), component_rank AS (
      SELECT *,
{rank_columns}
      FROM quality_grouped
    ), chosen_rank AS (
      SELECT *,
{chosen}
      FROM component_rank
    ), scored AS (
      SELECT *,
             list_count(list_filter([{quality_values}],x->x IS NOT NULL)) AS quality_nonnull,
             list_avg(list_filter([{quality_values}],x->x IS NOT NULL)) AS quality_score,
             datediff('day',try_cast(report_date AS DATE),try_cast(trade_date AS DATE)) AS report_age_days
      FROM chosen_rank
    )
    SELECT *,(
      upper(coalesce(list_status,''))='L'
      AND {status_eligible}
      AND listed_open_days>=250 AND valid_amount_20>=15
      AND amount_median_rank>=0.30 AND circ_mv_rank>=0.20
      AND report_age_days BETWEEN 0 AND 550
      AND quality_nonnull>=3 AND quality_score>=0.30
      AND minute_complete
    ) AS quality_liquidity_keep
    FROM scored
    ORDER BY symbol_idx
    """


def _load_quality_membership(
    connection: duckdb.DuckDBPyConnection,
    *,
    signal_date: str,
    date_idx: int,
    pack: Mapping[str, Any],
    arrays: PackArrays,
    qdp_paths: Mapping[str, Sequence[Path]],
) -> pd.DataFrame:
    symbols = np.asarray(pack["symbol_values"], dtype=object)
    candidate_indices = np.flatnonzero(
        np.asarray(arrays.candidate_eligible[date_idx], dtype=bool)
    ).astype(np.int32)
    candidates = pd.DataFrame(
        {
            "symbol": symbols[candidate_indices].astype(str),
            "symbol_idx": candidate_indices,
            "trade_date": signal_date,
        }
    )
    connection.register("candidate_current", candidates)
    required = (
        "market_daily_raw",
        "market_intraday_5m",
        "financial_quarterly",
        "universe_snapshot",
        "security_status",
        "valuation",
        "share_capital",
        "industry_concept",
        "trading_calendar",
    )
    missing = sorted(set(required).difference(qdp_paths))
    if missing:
        raise LiveInferenceError(f"active_core_qdp_sources_missing:{missing}")
    scans = {name: prep._scan(qdp_paths[name]) for name in required}
    frame = connection.execute(
        _quality_membership_sql(signal_date=signal_date, scans=scans)
    ).fetchdf()
    if len(frame) != len(candidates) or frame["symbol"].duplicated().any():
        raise LiveInferenceError("quality_membership_candidate_alignment_failed")
    frame["quality_liquidity_keep"] = (
        frame["quality_liquidity_keep"].fillna(False).astype(bool)
    )
    frame["listed_open_days"] = _canonical_listing_open_days(
        connection,
        frame=frame,
        signal_date=signal_date,
        scans=scans,
    )
    return frame


def _canonical_listing_open_days(
    connection: duckdb.DuckDBPyConnection,
    *,
    frame: pd.DataFrame,
    signal_date: str,
    scans: Mapping[str, str],
) -> np.ndarray:
    """Use the research calendar's pre-2010 open-day index, not a truncated count."""

    calendar_path = (
        DEFAULT_CALENDAR_INPUT_MANIFEST.parent / "calendar_positions.parquet"
    )
    listing_path = DEFAULT_CALENDAR_INPUT_MANIFEST.parent / "listing_positions.parquet"
    if not calendar_path.is_file() or not listing_path.is_file():
        raise LiveInferenceError("canonical_calendar_inputs_missing")
    calendar = pd.read_parquet(calendar_path, columns=["trade_date", "open_index"])
    calendar["trade_date"] = calendar["trade_date"].astype(str)
    current = calendar.loc[calendar["trade_date"].eq(str(signal_date)), "open_index"]
    if len(current):
        current_open_index = int(current.iloc[0])
    else:
        last_date = str(calendar["trade_date"].max())
        last_index = int(calendar["open_index"].max())
        future_open_count = int(
            connection.execute(
                f"SELECT count(*) FROM {scans['trading_calendar']} "
                f"WHERE is_open AND trade_date>'{last_date}' "
                f"AND trade_date<='{signal_date!s}'"
            ).fetchone()[0]
        )
        current_open_index = last_index + future_open_count
    listing = pd.read_parquet(listing_path, columns=["symbol", "list_open_index"])
    listed = (
        frame["symbol"].astype(str).map(listing.set_index("symbol")["list_open_index"])
    )
    output = current_open_index - pd.to_numeric(listed, errors="coerce") + 1
    return output.to_numpy(dtype=np.float32)


def _load_index_and_industry(
    connection: duckdb.DuckDBPyConnection,
    *,
    signal_date: str,
    pack: Mapping[str, Any],
    qdp_paths: Mapping[str, Sequence[Path]],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    symbols = tuple(str(value) for value in pack["symbol_values"])
    symbol_map = {value: idx for idx, value in enumerate(symbols)}
    symbol_count = len(symbols)
    index_map = {
        "000300.SH": "csi300",
        "000905.SH": "csi500",
        "000016.SH": "sse50",
    }
    membership = {
        name: np.zeros(symbol_count, dtype=bool) for name in index_map.values()
    }
    index_frame = connection.execute(
        "SELECT symbol,index_symbol,source_snapshot_date "
        f"FROM {prep._scan(qdp_paths['index_constituents'])} "
        f"WHERE trade_date='{signal_date}' "
        "AND index_symbol IN ('000300.SH','000905.SH','000016.SH')"
    ).fetchdf()
    if not index_frame.empty:
        source = index_frame["source_snapshot_date"].fillna("").astype(str)
        if bool(((source == "") | (source > signal_date)).any()):
            raise LiveInferenceError("index_membership_source_time_violation")
        for index_symbol, name in index_map.items():
            values = index_frame.loc[
                index_frame["index_symbol"].astype(str).eq(index_symbol), "symbol"
            ].astype(str)
            positions = [symbol_map[value] for value in values if value in symbol_map]
            membership[name][positions] = True
    industry_frame = connection.execute(
        "SELECT symbol,industry,industry_source_date "
        f"FROM {prep._scan(qdp_paths['industry_concept'])} "
        f"WHERE trade_date='{signal_date}'"
    ).fetchdf()
    if industry_frame["symbol"].duplicated().any():
        raise LiveInferenceError("industry_current_key_duplicated")
    industry = np.zeros(symbol_count, dtype=np.int64)
    source_age = np.full(symbol_count, np.nan, dtype=np.float32)
    for row in industry_frame.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in symbol_map:
            continue
        source_date = str(row.industry_source_date or "")
        if not source_date or source_date > signal_date:
            raise LiveInferenceError("industry_source_time_violation")
        position = symbol_map[symbol]
        industry[position] = signal._stable_category_hash(row.industry)
        source_age[position] = np.float32(
            (pd.Timestamp(signal_date) - pd.Timestamp(source_date)).days
        )
    return membership, industry, source_age


def _last_daily_features(
    raw: np.ndarray, turnover: np.ndarray
) -> dict[str, np.ndarray]:
    if raw.ndim != 3 or raw.shape[0] < 61 or raw.shape[2] < 13:
        raise LiveInferenceError("daily_history_shape_invalid")
    open_price = np.asarray(raw[:, :, 0], dtype=np.float32)
    high = np.asarray(raw[:, :, 1], dtype=np.float32)
    low = np.asarray(raw[:, :, 2], dtype=np.float32)
    close = np.asarray(raw[:, :, 3], dtype=np.float32)
    volume = np.asarray(raw[:, :, 4], dtype=np.float32)
    amount = np.asarray(raw[:, :, 5], dtype=np.float32)
    previous_close = np.full(close.shape, np.nan, dtype=np.float32)
    previous_close[1:] = close[:-1]
    price_range = np.asarray(high - low, dtype=np.float32)
    output: dict[str, np.ndarray] = {}

    def keep(name: str, values: np.ndarray) -> None:
        output[name] = np.asarray(values[-1], dtype=np.float32).copy()

    keep("open_gap_1d", signal._safe_divide(open_price, previous_close) - 1.0)
    keep("high_ret_prev_close_1d", signal._safe_divide(high, previous_close) - 1.0)
    keep("low_ret_prev_close_1d", signal._safe_divide(low, previous_close) - 1.0)
    keep("close_ret_1d", signal._safe_divide(close, previous_close) - 1.0)
    keep("intraday_range_1d", signal._safe_divide(price_range, previous_close))
    keep(
        "body_to_range_1d",
        signal._safe_divide(np.abs(close - open_price), price_range),
    )
    keep(
        "upper_shadow_to_range_1d",
        signal._safe_divide(high - np.maximum(open_price, close), price_range),
    )
    keep(
        "lower_shadow_to_range_1d",
        signal._safe_divide(np.minimum(open_price, close) - low, price_range),
    )
    keep("close_location_1d", signal._safe_divide(close - low, price_range))
    keep("log_amount_1d", np.log1p(np.maximum(amount, 0.0)).astype(np.float32))
    keep("log_volume_1d", np.log1p(np.maximum(volume, 0.0)).astype(np.float32))
    keep("log_turnover_pct_1d", np.asarray(turnover[:, :, 0], dtype=np.float32))
    keep("relative_turnover_20d", np.asarray(turnover[:, :, 1], dtype=np.float32))
    for horizon in FEATURE_RETURN_HORIZONS:
        keep(f"return_{horizon}d", signal._lagged_return(close, horizon))
    log_close = np.where(close > 0.0, np.log(close), np.nan).astype(np.float32)
    log_return = np.full(close.shape, np.nan, dtype=np.float32)
    log_return[1:] = log_close[1:] - log_close[:-1]
    true_range = np.maximum.reduce(
        [
            np.asarray(high - low, dtype=np.float32),
            np.asarray(np.abs(high - previous_close), dtype=np.float32),
            np.asarray(np.abs(low - previous_close), dtype=np.float32),
        ]
    )
    true_range = signal._safe_divide(true_range, previous_close)
    log_amount = np.log1p(np.maximum(amount, 0.0)).astype(np.float32)
    abs_log_return = np.abs(log_return).astype(np.float32)
    for window in FEATURE_WINDOWS:
        ma = signal._rolling_mean(close, window)
        keep(f"ma_distance_{window}d", signal._safe_divide(close, ma) - 1.0)
        _mean_return, volatility = signal._rolling_mean_std(log_return, window)
        keep(f"volatility_{window}d", volatility)
        downside = np.where(
            np.isfinite(log_return), np.minimum(log_return, 0.0), np.nan
        ).astype(np.float32)
        downside_square_mean = signal._rolling_mean(np.square(downside), window)
        keep(
            f"downside_volatility_{window}d",
            np.sqrt(np.maximum(downside_square_mean, 0.0)).astype(np.float32),
        )
        keep(f"atr_{window}d", signal._rolling_mean(true_range, window))
        slope, t_value, r2, residual = signal._rolling_linear_stats(log_close, window)
        keep(f"trend_slope_{window}d", slope)
        keep(f"trend_t_{window}d", t_value)
        keep(f"trend_r2_{window}d", r2)
        keep(f"trend_residual_{window}d", residual)
        path_length = signal._rolling_mean(abs_log_return, window) * float(window)
        displacement = np.full(log_close.shape, np.nan, dtype=np.float32)
        displacement[window:] = np.abs(log_close[window:] - log_close[:-window])
        keep(
            f"efficiency_ratio_{window}d",
            signal._safe_divide(displacement, path_length),
        )
        rolling_low, rolling_high = signal._rolling_min_max(close, window)
        keep(
            f"price_range_position_{window}d",
            signal._safe_divide(close - rolling_low, rolling_high - rolling_low),
        )
        keep(
            f"high_age_fraction_{window}d",
            signal._rolling_extreme_age(high, window, mode="max")
            / float(max(window - 1, 1)),
        )
        keep(
            f"low_age_fraction_{window}d",
            signal._rolling_extreme_age(low, window, mode="min")
            / float(max(window - 1, 1)),
        )
        up = np.where(np.isfinite(log_return), log_return > 0.0, np.nan).astype(
            np.float32
        )
        down = np.where(np.isfinite(log_return), log_return < 0.0, np.nan).astype(
            np.float32
        )
        keep(f"up_day_ratio_{window}d", signal._rolling_mean(up, window))
        keep(f"down_day_ratio_{window}d", signal._rolling_mean(down, window))
        keep(
            f"price_amount_correlation_{window}d",
            signal._rolling_corr(log_return, log_amount, window),
        )
        amount_mean = signal._rolling_mean(amount, window)
        volume_mean = signal._rolling_mean(volume, window)
        keep(f"amount_ratio_{window}d", signal._safe_divide(amount, amount_mean))
        keep(f"volume_ratio_{window}d", signal._safe_divide(volume, volume_mean))
        del (
            ma,
            _mean_return,
            volatility,
            downside,
            downside_square_mean,
            slope,
            t_value,
            r2,
            residual,
            path_length,
            displacement,
            rolling_low,
            rolling_high,
            up,
            down,
            amount_mean,
            volume_mean,
        )
    expected = set(daily_feature_names())
    if set(output) != expected:
        raise LiveInferenceError("daily_feature_catalog_mismatch")
    return output


def _market_features(
    *,
    raw: np.ndarray,
    universe: np.ndarray,
    index_membership: Mapping[str, np.ndarray],
    is_suspended: np.ndarray,
    is_st: np.ndarray,
) -> dict[str, float]:
    close = np.asarray(raw[:, :, 3], dtype=np.float32)
    amount = np.asarray(raw[:, :, 5], dtype=np.float32)
    ret1 = signal._lagged_return(close, 1)
    ret5 = signal._lagged_return(close, 5)
    ret20 = signal._lagged_return(close, 20)
    log_close = np.where(close > 0.0, np.log(close), np.nan).astype(np.float32)
    log_return = np.full(close.shape, np.nan, dtype=np.float32)
    log_return[1:] = log_close[1:] - log_close[:-1]
    _mean, vol20 = signal._rolling_mean_std(log_return, 20)
    ma20 = signal._rolling_mean(close, 20)
    _low60, high60 = signal._rolling_min_max(close, 60)
    drawdown60 = 1.0 - signal._safe_divide(close, high60)
    above_ma20 = close > ma20
    output: dict[str, float] = {}
    groups = {"all": np.asarray(universe, dtype=bool)}
    for name in ("csi300", "csi500", "sse50"):
        groups[name] = groups["all"] & np.asarray(index_membership[name], dtype=bool)
    for group in MARKET_GROUPS:
        current_mask = groups[group]
        if not bool(current_mask.any()):
            continue

        def finite_mean(values: np.ndarray, mask: np.ndarray = current_mask) -> float:
            selected = np.asarray(values[-1, mask], dtype=np.float64)
            selected = selected[np.isfinite(selected)]
            return float(selected.mean()) if selected.size else np.nan

        current_ret1 = np.asarray(ret1[-1, current_mask], dtype=np.float64)
        current_ret5 = np.asarray(ret5[-1, current_mask], dtype=np.float64)
        current_amount = np.asarray(amount[-1, current_mask], dtype=np.float64)
        metrics = {
            "ret1_mean": finite_mean(ret1),
            "ret5_mean": finite_mean(ret5),
            "ret20_mean": finite_mean(ret20),
            "vol20_mean": finite_mean(vol20),
            "drawdown60_mean": finite_mean(drawdown60),
            "breadth_ret1_positive": float(np.nanmean(current_ret1 > 0.0)),
            "breadth_ret5_positive": float(np.nanmean(current_ret5 > 0.0)),
            "breadth_above_ma20": float(
                np.mean(np.asarray(above_ma20[-1, current_mask], dtype=bool))
            ),
            "ret1_dispersion": float(np.nanstd(current_ret1)),
            "log_total_amount": float(
                np.log1p(np.nansum(np.maximum(current_amount, 0.0)))
            ),
            "suspended_rate": float(
                np.mean(np.asarray(is_suspended[current_mask], dtype=bool))
            ),
            "st_rate": float(np.mean(np.asarray(is_st[current_mask], dtype=bool))),
            "up_limit_rate": float(np.nanmean(current_ret1 >= 0.095)),
            "down_limit_rate": float(np.nanmean(current_ret1 <= -0.095)),
        }
        for metric, value in metrics.items():
            output[f"market_{group}__{metric}"] = float(np.float32(value))
    return output


def _industry_features(
    *,
    raw: np.ndarray,
    universe: np.ndarray,
    industry: np.ndarray,
    industry_source_age: np.ndarray,
    selected_symbols: np.ndarray,
) -> dict[str, np.ndarray]:
    close = np.asarray(raw[:, :, 3], dtype=np.float32)
    ret1 = signal._lagged_return(close, 1)[-1]
    ret5 = signal._lagged_return(close, 5)[-1]
    ret20 = signal._lagged_return(close, 20)[-1]
    base_mask = np.asarray(universe, dtype=bool) & (industry != 0)
    output = {
        name: np.full(len(selected_symbols), np.nan, dtype=np.float32)
        for name in INDUSTRY_FEATURES
    }
    if not bool(base_mask.any()):
        return output
    codes = industry[base_mask]
    unique, inverse = np.unique(codes, return_inverse=True)
    group_count = np.bincount(inverse).astype(np.float64)

    def aggregate(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current = np.asarray(series[base_mask], dtype=np.float64)
        finite = np.isfinite(current)
        count = np.bincount(inverse[finite], minlength=len(unique)).astype(np.float64)
        total = np.bincount(
            inverse[finite], weights=current[finite], minlength=len(unique)
        ).astype(np.float64)
        mean = np.divide(
            total, count, out=np.full(len(unique), np.nan), where=count > 0
        )
        square = np.bincount(
            inverse[finite], weights=np.square(current[finite]), minlength=len(unique)
        ).astype(np.float64)
        variance = np.divide(
            square, count, out=np.full(len(unique), np.nan), where=count > 0
        ) - np.square(mean)
        return mean, np.sqrt(np.maximum(variance, 0.0))

    mean1, dispersion1 = aggregate(ret1)
    mean5, _ = aggregate(ret5)
    mean20, _ = aggregate(ret20)
    current_ret1 = np.asarray(ret1[base_mask], dtype=np.float64)
    finite_ret1 = np.isfinite(current_ret1)
    positive = np.bincount(
        inverse[finite_ret1],
        weights=(current_ret1[finite_ret1] > 0.0).astype(np.float64),
        minlength=len(unique),
    )
    valid_count = np.bincount(inverse[finite_ret1], minlength=len(unique)).astype(
        np.float64
    )
    breadth = np.divide(
        positive,
        valid_count,
        out=np.full(len(unique), np.nan),
        where=valid_count > 0,
    )
    candidate_codes = industry[selected_symbols]
    positions = np.searchsorted(unique, candidate_codes)
    mapped = np.minimum(positions, len(unique) - 1)
    known = (candidate_codes != 0) & (positions < len(unique))
    known &= unique[mapped] == candidate_codes
    output["industry_ret1_mean"][known] = mean1[mapped[known]]
    output["industry_ret5_mean"][known] = mean5[mapped[known]]
    output["industry_breadth_ret1_positive"][known] = breadth[mapped[known]]
    output["industry_ret1_dispersion"][known] = dispersion1[mapped[known]]
    output["industry_relative_ret5"][known] = (
        ret5[selected_symbols][known] - mean5[mapped[known]]
    )
    output["industry_relative_ret20"][known] = (
        ret20[selected_symbols][known] - mean20[mapped[known]]
    )
    output["industry_member_count_log"][known] = np.log1p(group_count[mapped[known]])
    output["industry_source_age_days"] = np.asarray(
        industry_source_age[selected_symbols], dtype=np.float32
    )
    return output


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (np.sign(array) * np.log1p(np.abs(array))).astype(np.float32)


def _size_status_features(
    *,
    membership: pd.DataFrame,
    selected: pd.DataFrame,
    selected_symbols: np.ndarray,
    index_membership: Mapping[str, np.ndarray],
    industry: np.ndarray,
    arrays: PackArrays,
    date_idx: int,
) -> dict[str, np.ndarray]:
    def values(name: str) -> np.ndarray:
        return pd.to_numeric(selected[name], errors="coerce").to_numpy(dtype=np.float32)

    total_mv = values("total_mv")
    circ_mv = values("circ_mv")
    pe = values("pe")
    pb = values("pb")
    turnover_rate = values("turnover_rate")
    total_share = values("total_share")
    float_share = values("float_share")
    trade_dates = pd.to_datetime(selected["trade_date"], errors="coerce")
    share_source_dates = pd.to_datetime(
        selected["float_share_source_date"], errors="coerce"
    )
    share_source_age = (trade_dates - share_source_dates).dt.days.to_numpy(
        dtype=np.float32
    )
    history = slice(date_idx - 19, date_idx + 1)
    status_valid = np.asarray(arrays.status_valid[history], dtype=bool)
    st = np.asarray(arrays.is_st[history], dtype=bool)
    suspended = np.asarray(arrays.is_suspended[history], dtype=bool)
    st_history = np.where(status_valid, st.astype(np.float32), np.nan).astype(
        np.float32
    )
    suspended_history = np.where(
        status_valid, suspended.astype(np.float32), np.nan
    ).astype(np.float32)
    st_rate = signal._rolling_mean(st_history, 20)[-1, selected_symbols]
    suspended_rate = signal._rolling_mean(suspended_history, 20)[-1, selected_symbols]
    if len(membership) < len(selected):
        raise LiveInferenceError("membership_selection_alignment_failed")
    return {
        "log_total_market_value": np.log1p(np.maximum(total_mv, 0.0)).astype(
            np.float32
        ),
        "log_circulating_market_value": np.log1p(np.maximum(circ_mv, 0.0)).astype(
            np.float32
        ),
        "circulating_market_value_ratio": signal._safe_divide(circ_mv, total_mv),
        "signed_log_pe": _signed_log1p(pe),
        "signed_log_pb": _signed_log1p(pb),
        "log_turnover_rate": np.log1p(np.maximum(turnover_rate, 0.0)).astype(
            np.float32
        ),
        "log_total_share": np.log1p(np.maximum(total_share, 0.0)).astype(np.float32),
        "log_float_share": np.log1p(np.maximum(float_share, 0.0)).astype(np.float32),
        "float_share_ratio": signal._safe_divide(float_share, total_share),
        "share_source_age_days": share_source_age,
        "listing_age_open_days": values("listed_open_days"),
        "is_csi300_member": index_membership["csi300"][selected_symbols].astype(
            np.float32
        ),
        "is_csi500_member": index_membership["csi500"][selected_symbols].astype(
            np.float32
        ),
        "is_sse50_member": index_membership["sse50"][selected_symbols].astype(
            np.float32
        ),
        "st_rate_20d": np.asarray(st_rate, dtype=np.float32),
        "suspended_rate_20d": np.asarray(suspended_rate, dtype=np.float32),
        "valuation_missing": (
            ~(
                np.isfinite(total_mv)
                & np.isfinite(circ_mv)
                & np.isfinite(pe)
                & np.isfinite(pb)
            )
        ).astype(np.float32),
        "industry_missing": (industry[selected_symbols] == 0).astype(np.float32),
    }


def _minute_feature_sql(*, signal_date: str, intraday_scan: str) -> str:
    expected_times = prep._expected_time_sql()
    return f"""
    WITH bars0 AS (
      SELECT c.symbol,b.bar_time,
             try_cast(b.open AS DOUBLE) AS open,
             try_cast(b.high AS DOUBLE) AS high,
             try_cast(b.low AS DOUBLE) AS low,
             try_cast(b.close AS DOUBLE) AS close,
             try_cast(b.volume AS DOUBLE) AS volume,
             try_cast(b.amount AS DOUBLE) AS amount,
             row_number() OVER (PARTITION BY b.symbol ORDER BY b.bar_time) AS bar_no,
             lag(try_cast(b.close AS DOUBLE)) OVER (
               PARTITION BY b.symbol ORDER BY b.bar_time
             ) AS previous_close
      FROM {intraday_scan} b
      JOIN selected_current c ON c.symbol=b.symbol
      WHERE b.trade_date='{signal_date}'
    ), bars AS (
      SELECT *,sum(amount) OVER (PARTITION BY symbol) AS total_amount,
             sum(volume) OVER (PARTITION BY symbol) AS total_volume,
              CASE WHEN previous_close>0 AND close>0
                THEN ln(close/previous_close) END AS log_return
      FROM bars0
    ), aggregate AS (
      SELECT symbol,count(*) AS bar_count,count(DISTINCT bar_time) AS distinct_bar_count,
             count(*) FILTER (WHERE bar_time IN ({expected_times})) AS expected_bar_count,
             arg_min(open,bar_time) AS day_open,
             max(close) AS max_close,min(close) AS min_close,
             arg_max(close,bar_time) AS day_close,max(high) AS day_high,min(low) AS day_low,
             max(CASE WHEN bar_no=6 THEN close END) AS close_bar_6,
             max(CASE WHEN bar_no=24 THEN close END) AS morning_close,
             max(CASE WHEN bar_no=25 THEN open END) AS afternoon_open,
             max(CASE WHEN bar_no=42 THEN close END) AS close_bar_42,
              sum(CASE
                WHEN bar_no=1 AND open>0 AND close>0 THEN abs(ln(close/open))
                ELSE abs(log_return)
              END) AS absolute_log_return_sum,
             sqrt(sum(log_return*log_return)) AS realized_volatility,
             sqrt(sum(CASE WHEN log_return<0 THEN log_return*log_return ELSE 0 END)) AS downside_semivolatility,
             sqrt(sum(CASE WHEN log_return>0 THEN log_return*log_return ELSE 0 END)) AS upside_semivolatility,
             max(log_return) AS max_log_return,min(log_return) AS min_log_return,
             avg(CASE WHEN log_return>0 THEN 1.0 ELSE 0.0 END)
               FILTER (WHERE log_return IS NOT NULL) AS positive_bar_fraction,
             sum(CASE WHEN bar_no<=12 THEN amount ELSE 0 END) AS first_hour_amount,
             sum(CASE WHEN bar_no>36 THEN amount ELSE 0 END) AS last_hour_amount,
             max(amount) AS max_bar_amount,
             -sum(CASE WHEN amount>0 AND total_amount>0
               THEN (amount/total_amount)*ln(amount/total_amount) ELSE 0 END)/ln(48.0) AS amount_entropy,
             sum(CASE WHEN bar_no<=24 THEN volume ELSE -volume END) AS volume_half_difference,
             arg_max(bar_no,high) AS high_bar_no,arg_min(bar_no,low) AS low_bar_no,
             sum(((high+low+close)/3.0)*volume) AS typical_value_volume,
             max(total_amount) AS total_amount,max(total_volume) AS total_volume
      FROM bars GROUP BY symbol
    )
    SELECT symbol,
      day_close/NULLIF(day_open,0)-1.0 AS minute_open_close_return,
      (day_high-day_low)/NULLIF(day_open,0) AS minute_range,
      (day_close-day_low)/NULLIF(day_high-day_low,0) AS minute_close_location,
      realized_volatility AS minute_realized_volatility,
      downside_semivolatility AS minute_downside_semivolatility,
      upside_semivolatility AS minute_upside_semivolatility,
      close_bar_6/NULLIF(day_open,0)-1.0 AS minute_first_30m_return,
      day_close/NULLIF(close_bar_42,0)-1.0 AS minute_last_30m_return,
      morning_close/NULLIF(day_open,0)-1.0 AS minute_morning_return,
      day_close/NULLIF(afternoon_open,0)-1.0 AS minute_afternoon_return,
      afternoon_open/NULLIF(morning_close,0)-1.0 AS minute_morning_afternoon_gap,
      day_close/NULLIF(typical_value_volume/NULLIF(total_volume,0),0)-1.0 AS minute_vwap_close_deviation,
      abs(ln(day_close/NULLIF(day_open,0)))/NULLIF(absolute_log_return_sum,0) AS minute_trend_efficiency,
      exp(max_log_return)-1.0 AS minute_max_5m_return,
      exp(min_log_return)-1.0 AS minute_min_5m_return,
      positive_bar_fraction AS minute_positive_bar_fraction,
      first_hour_amount/NULLIF(total_amount,0) AS minute_first_hour_amount_share,
      last_hour_amount/NULLIF(total_amount,0) AS minute_last_hour_amount_share,
      max_bar_amount/NULLIF(total_amount,0) AS minute_max_bar_amount_share,
      amount_entropy AS minute_amount_entropy,
      volume_half_difference/NULLIF(total_volume,0) AS minute_volume_half_imbalance,
      (high_bar_no-1.0)/47.0 AS minute_high_time_fraction,
      (low_bar_no-1.0)/47.0 AS minute_low_time_fraction,
      max_close/NULLIF(day_open,0)-1.0 AS minute_mfe_from_open,
      min_close/NULLIF(day_open,0)-1.0 AS minute_mae_from_open
    FROM aggregate
    WHERE bar_count=48 AND distinct_bar_count=48 AND expected_bar_count=48
      AND day_open>0 AND day_close>0 AND day_high>=day_low
    ORDER BY symbol
    """


def _load_minute_features(
    connection: duckdb.DuckDBPyConnection,
    *,
    signal_date: str,
    selected: pd.DataFrame,
    intraday_paths: Sequence[Path],
) -> dict[str, np.ndarray]:
    connection.register("selected_current", selected[["symbol"]])
    frame = connection.execute(
        _minute_feature_sql(
            signal_date=signal_date, intraday_scan=prep._scan(intraday_paths)
        )
    ).fetchdf()
    if len(frame) != len(selected) or frame["symbol"].duplicated().any():
        raise LiveInferenceError(
            f"selected_minute_support_incomplete:{len(frame)}:{len(selected)}"
        )
    aligned = selected[["symbol"]].merge(frame, on="symbol", how="left", validate="1:1")
    return {
        name: pd.to_numeric(aligned[name], errors="coerce").to_numpy(dtype=np.float32)
        for name in MINUTE_FEATURES
    }


def _build_sequence(
    *,
    arrays: PackArrays,
    date_idx: int,
    symbol_indices: np.ndarray,
    normalization: seq.Normalization,
    lookback: int,
) -> np.ndarray:
    start = date_idx - int(lookback) + 1
    if start < 0:
        raise LiveInferenceError("sequence_lookback_before_pack")
    raw = np.asarray(
        arrays.daily_raw[start : date_idx + 1, symbol_indices, :], dtype=np.float32
    )
    state = np.asarray(
        arrays.daily_state[start : date_idx + 1, symbol_indices, :], dtype=np.float32
    )
    turnover = np.asarray(
        arrays.turnover[start : date_idx + 1, symbol_indices, :], dtype=np.float32
    )
    current_close = raw[-1, :, 3]
    price_relative = np.divide(
        raw[:, :, :4],
        current_close[None, :, None],
        out=np.full_like(raw[:, :, :4], np.nan),
        where=np.isfinite(current_close[None, :, None])
        & (current_close[None, :, None] > 0.0),
    ) - np.float32(1.0)
    values = np.concatenate(
        [
            price_relative,
            raw[:, :, seq.DAILY_RAW_DERIVED_SLICE],
            state,
            turnover,
        ],
        axis=2,
    ).transpose(1, 0, 2)
    values = (
        values - normalization.sequence_mean[None, None, :]
    ) / normalization.sequence_scale[None, None, :]
    return np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0).clip(-10.0, 10.0)


def build_feature_snapshot(
    *,
    contract: SnapshotContract,
    arrays: PackArrays,
    signal_date: str,
    date_idx: int,
    output_root: Path,
    qdp_paths: Mapping[str, Sequence[Path]],
    qdp_dataset_ids: Mapping[str, str],
    normalization: seq.Normalization,
    lookback: int,
) -> FeatureSnapshot:
    connection, resource = _connect(output_root)
    try:
        membership = _load_quality_membership(
            connection,
            signal_date=signal_date,
            date_idx=date_idx,
            pack=contract.pack,
            arrays=arrays,
            qdp_paths=qdp_paths,
        )
        selected = membership.loc[membership["quality_liquidity_keep"]].copy()
        if len(selected) < 10:
            raise LiveInferenceError(f"quality_pool_too_small:{len(selected)}")
        selected = selected.sort_values("symbol_idx", kind="stable").reset_index(
            drop=True
        )
        selected_symbols = selected["symbol_idx"].to_numpy(dtype=np.int32)
        index_membership, industry, industry_source_age = _load_index_and_industry(
            connection,
            signal_date=signal_date,
            pack=contract.pack,
            qdp_paths=qdp_paths,
        )
        history = slice(date_idx - max(FEATURE_WINDOWS), date_idx + 1)
        raw = np.asarray(arrays.daily_raw[history], dtype=np.float32)
        turnover = np.asarray(arrays.turnover[history], dtype=np.float32)
        daily = _last_daily_features(raw, turnover)
        broad = np.flatnonzero(
            np.asarray(arrays.candidate_eligible[date_idx], dtype=bool)
        )
        daily_values: dict[str, np.ndarray] = {}
        for name, panel in daily.items():
            daily_values[name] = panel[selected_symbols]
            ranks = np.full(
                len(contract.pack["symbol_values"]), np.nan, dtype=np.float32
            )
            ranks[broad] = signal._rank_percentile(panel[broad])
            daily_values[f"cs_percentile__{name}"] = ranks[selected_symbols]
        market_values = _market_features(
            raw=raw,
            universe=np.asarray(arrays.pit_universe_has_bar[date_idx], dtype=bool),
            index_membership=index_membership,
            is_suspended=np.asarray(arrays.is_suspended[date_idx], dtype=bool),
            is_st=np.asarray(arrays.is_st[date_idx], dtype=bool),
        )
        industry_values = _industry_features(
            raw=raw,
            universe=np.asarray(arrays.pit_universe_has_bar[date_idx], dtype=bool),
            industry=industry,
            industry_source_age=industry_source_age,
            selected_symbols=selected_symbols,
        )
        size_values = _size_status_features(
            membership=membership,
            selected=selected,
            selected_symbols=selected_symbols,
            index_membership=index_membership,
            industry=industry,
            arrays=arrays,
            date_idx=date_idx,
        )
        minute_values = _load_minute_features(
            connection,
            signal_date=signal_date,
            selected=selected,
            intraday_paths=qdp_paths["market_intraday_5m"],
        )
    finally:
        connection.close()
    values_by_name = {
        **daily_values,
        **industry_values,
        **size_values,
        **minute_values,
    }
    for name, value in market_values.items():
        values_by_name[name] = np.full(len(selected), value, dtype=np.float32)
    static = np.full(
        (len(selected), len(contract.feature_names)), np.nan, dtype=np.float32
    )
    feature_index = {name: idx for idx, name in enumerate(contract.feature_names)}
    for name in contract.core_feature_names:
        if name not in values_by_name:
            raise LiveInferenceError(f"core_feature_not_built:{name}")
        values = np.asarray(values_by_name[name], dtype=np.float32)
        if values.shape != (len(selected),):
            raise LiveInferenceError(
                f"core_feature_shape_invalid:{name}:{values.shape}"
            )
        static[:, feature_index[name]] = values
    if bool(np.isfinite(static[:, 314:]).any()):
        raise LiveInferenceError("masked_243_fields_are_not_nan")
    market_names = tuple(
        name
        for name in contract.feature_names
        if contract.feature_records[name]["analytic_family"] == "market_state"
    )
    market = np.asarray(
        [market_values[name] for name in market_names], dtype=np.float32
    )
    sequence_values = _build_sequence(
        arrays=arrays,
        date_idx=date_idx,
        symbol_indices=selected_symbols,
        normalization=normalization,
        lookback=lookback,
    )
    return FeatureSnapshot(
        signal_date=signal_date,
        date_idx=date_idx,
        symbols=selected["symbol"].astype(str).to_numpy(),
        symbol_indices=selected_symbols,
        names=selected["name"].fillna("").astype(str).to_numpy(),
        static=static,
        market=market,
        sequence=sequence_values,
        membership=membership,
        source_profile={
            "candidate_eligible_count": len(membership),
            "quality_liquidity_pit_count": len(selected),
            "complete_5m_count": int(membership["minute_complete"].fillna(False).sum()),
            "active_qdp_dataset_ids": {
                name: str(qdp_dataset_ids[name])
                for name in sorted(
                    set(qdp_dataset_ids).intersection(
                        {
                            "market_daily_raw",
                            "market_intraday_5m",
                            "financial_quarterly",
                            "universe_snapshot",
                            "security_status",
                            "valuation",
                            "share_capital",
                            "industry_concept",
                            "trading_calendar",
                            "index_constituents",
                        }
                    )
                )
            },
            "resource_plan": resource,
        },
    )


def validate_against_frozen_history(
    snapshot: FeatureSnapshot,
    *,
    contract: SnapshotContract,
    model_manifest_path: Path = DEFAULT_MODEL_MANIFEST,
) -> dict[str, Any]:
    row_path = Path(str(contract.model["row_index"]["path"]))
    frame = pd.read_parquet(
        row_path,
        filters=[("trade_date", "=", snapshot.signal_date)],
        columns=["date_idx", "symbol_idx", "symbol"],
    ).sort_values("symbol_idx", kind="stable")
    expected_symbols = frame["symbol"].astype(str).to_numpy()
    pool_exact = np.array_equal(snapshot.symbols, expected_symbols)
    if not pool_exact:
        raise LiveInferenceError("historical_quality_pool_reproduction_failed")
    with duckdb.connect() as connection:
        start = int(
            connection.execute(
                "SELECT count(*) FROM read_parquet(?) WHERE date_idx<?",
                [str(row_path), int(snapshot.date_idx)],
            ).fetchone()[0]
        )
    positions = start + np.arange(len(frame), dtype=np.int64)
    matrix_record = contract.model["storage"]["compact"]
    matrix = _open_memmap(matrix_record, np.float32)
    expected = np.asarray(matrix[positions, :314], dtype=np.float32)
    actual = np.asarray(snapshot.static[:, :314], dtype=np.float32)
    expected_finite = np.isfinite(expected)
    actual_finite = np.isfinite(actual)
    finite_state_mismatch = expected_finite ^ actual_finite
    comparable = expected_finite & actual_finite
    difference = np.full(expected.shape, np.nan, dtype=np.float64)
    difference[comparable] = np.abs(
        actual[comparable].astype(np.float64) - expected[comparable].astype(np.float64)
    )
    close = np.zeros(expected.shape, dtype=bool)
    close[~expected_finite & ~actual_finite] = True
    close[comparable] = np.isclose(
        actual[comparable], expected[comparable], rtol=2.0e-5, atol=2.0e-6
    )
    feature_rows: list[dict[str, Any]] = []
    for column, name in enumerate(contract.core_feature_names):
        current = difference[:, column]
        finite_difference = current[np.isfinite(current)]
        feature_rows.append(
            {
                "feature_name": name,
                "finite_state_mismatch_count": int(
                    finite_state_mismatch[:, column].sum()
                ),
                "not_close_count": int((~close[:, column]).sum()),
                "maximum_absolute_difference": float(finite_difference.max())
                if finite_difference.size
                else 0.0,
            }
        )
    mismatch_count = int(finite_state_mismatch.sum())
    not_close_count = int((~close).sum())
    passed = mismatch_count == 0 and not_close_count == 0
    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "passed" if passed else "failed",
        "signal_date": snapshot.signal_date,
        "pool_exact_match": pool_exact,
        "row_count": len(frame),
        "core_feature_count": 314,
        "cell_count": int(expected.size),
        "finite_state_mismatch_count": mismatch_count,
        "not_close_count": not_close_count,
        "maximum_absolute_difference": float(np.nanmax(difference))
        if bool(np.isfinite(difference).any())
        else 0.0,
        "mean_absolute_difference": float(np.nanmean(difference))
        if bool(np.isfinite(difference).any())
        else 0.0,
        "largest_feature_differences": sorted(
            feature_rows,
            key=lambda row: (
                int(row["not_close_count"]),
                float(row["maximum_absolute_difference"]),
            ),
            reverse=True,
        )[:20],
        "sources": {
            "model_inputs": _file_record(model_manifest_path),
            "row_index": _file_record(row_path),
        },
        "no_outcome_or_fill_array_opened": True,
    }
    if not passed:
        raise LiveInferenceError(
            "historical_core_feature_reproduction_failed:"
            f"finite={mismatch_count}:not_close={not_close_count}"
        )
    return result


def dual_market_gate(
    *,
    ridge_prediction: float,
    linear_probability: float,
    neural_return_prediction: float,
    neural_probability: float,
) -> bool:
    return bool(
        ridge_prediction > 0.0
        and linear_probability > 0.5
        and neural_return_prediction > 0.0
        and neural_probability > 0.5
    )


@torch.no_grad()
def score_snapshot(
    snapshot: FeatureSnapshot,
    *,
    contract: SnapshotContract,
    checkpoint: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bundle = contract.bundle
    tree_path = Path(str(bundle["files"]["tree_model"]["path"]))
    sequence_path = Path(str(bundle["files"]["sequence_model"]["path"]))
    market_path = Path(str(bundle["files"]["market_models"]["path"]))
    for key, path in (
        ("tree_model", tree_path),
        ("sequence_model", sequence_path),
        ("market_models", market_path),
    ):
        if _sha256(path) != str(bundle["files"][key]["sha256"]):
            raise LiveInferenceError(f"final_bundle_file_hash_changed:{key}")
    booster = lgb.Booster(model_file=str(tree_path))
    tree_prediction = np.asarray(
        booster.predict(
            snapshot.static,
            num_iteration=int(bundle["stock_tree"]["final_iterations"]),
        ),
        dtype=np.float32,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = seq.PayoffSequenceModel().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    static_tensor = torch.from_numpy(snapshot.static).to(device)
    sequence_tensor = torch.from_numpy(snapshot.sequence).to(device)
    normalization = seq.Normalization.from_payload(checkpoint["normalization"])
    market_normalized = np.nan_to_num(
        (snapshot.market - normalization.market_center) / normalization.market_scale,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    ).clip(-10.0, 10.0)
    market_tensor = torch.from_numpy(market_normalized[None, :]).to(device)
    use_amp = device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        static_standardized, missing = seq.cross_sectional_standardize(
            static_tensor, (len(snapshot.symbols),)
        )
        sequence_prediction, neural_return, neural_logit = model(
            static_standardized,
            missing,
            sequence_tensor,
            market_tensor,
            (len(snapshot.symbols),),
        )
    sequence_score = sequence_prediction.float().cpu().numpy().astype(np.float32)
    neural_return_prediction = (
        float(neural_return[0].float().cpu()) * seq.MARKET_TARGET_SCALE
    )
    neural_probability = float(torch.sigmoid(neural_logit[0].float()).cpu())
    with market_path.open("rb") as stream:
        market_models = pickle.load(stream)
    expected_market_names = tuple(market_models["market_feature_names"])
    checkpoint_market_names = tuple(checkpoint["market_feature_names"])
    if expected_market_names != checkpoint_market_names:
        raise LiveInferenceError("final_market_feature_order_changed")
    market_input = snapshot.market.astype(np.float64, copy=False)[None, :]
    ridge_prediction = float(market_models["ridge"].predict(market_input)[0])
    linear_probability = float(
        market_models["logistic"].predict_proba(market_input)[0, 1]
    )
    gate = dual_market_gate(
        ridge_prediction=ridge_prediction,
        linear_probability=linear_probability,
        neural_return_prediction=neural_return_prediction,
        neural_probability=neural_probability,
    )
    tree_rank = pd.Series(tree_prediction).rank(pct=True).to_numpy(dtype=np.float32)
    sequence_rank = pd.Series(sequence_score).rank(pct=True).to_numpy(dtype=np.float32)
    ensemble = (tree_rank + sequence_rank) / np.float32(2.0)
    scores = pd.DataFrame(
        {
            "signal_date": snapshot.signal_date,
            "symbol": snapshot.symbols,
            "name": snapshot.names,
            "symbol_idx": snapshot.symbol_indices,
            "tree_prediction": tree_prediction,
            "sequence_prediction": sequence_score,
            "tree_rank_percentile": tree_rank,
            "sequence_rank_percentile": sequence_rank,
            "ensemble_score": ensemble,
        }
    ).sort_values(["ensemble_score", "symbol"], ascending=[False, True], kind="stable")
    scores["score_rank"] = np.arange(1, len(scores) + 1, dtype=np.int32)
    scores["market_gate_active"] = gate
    scores = scores.reset_index(drop=True)
    market_gate = {
        "ridge_prediction": ridge_prediction,
        "linear_positive_probability": linear_probability,
        "neural_return_prediction": neural_return_prediction,
        "neural_positive_probability": neural_probability,
        "ridge_above_zero": ridge_prediction > 0.0,
        "linear_probability_above_0p5": linear_probability > 0.5,
        "neural_return_above_zero": neural_return_prediction > 0.0,
        "neural_probability_above_0p5": neural_probability > 0.5,
        "dual_gate_active": gate,
        "inference_device": str(device),
    }
    del model, static_tensor, sequence_tensor, market_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return scores, market_gate


def requested_orders(scores: pd.DataFrame, *, top_k: int = 10) -> pd.DataFrame:
    columns = (
        "signal_date",
        "symbol",
        "name",
        "score_rank",
        "ensemble_score",
        "request",
        "entry_rule",
        "entry_trade_date",
        "exit_rule",
        "rank_substitution",
        "real_order_authority",
    )
    if scores.empty or not bool(scores["market_gate_active"].iloc[0]):
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
    selected = scores.head(int(top_k)).copy()
    selected["request"] = "paper_buy_request"
    selected["entry_rule"] = "next_legal_open"
    selected["entry_trade_date"] = pd.NA
    selected["exit_rule"] = "D3_close_request_then_first_legally_sellable_open"
    selected["rank_substitution"] = False
    selected["real_order_authority"] = False
    return selected[list(columns)].copy()


def _snapshot_frame(
    snapshot: FeatureSnapshot, contract: SnapshotContract
) -> pd.DataFrame:
    frame = pd.DataFrame(snapshot.static, columns=contract.feature_names)
    frame.insert(0, "name", snapshot.names)
    frame.insert(0, "symbol", snapshot.symbols)
    frame.insert(0, "symbol_idx", snapshot.symbol_indices)
    frame.insert(0, "signal_date", snapshot.signal_date)
    return frame


def run_live_inference(
    *,
    requested_signal_date: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    pack_manifest: Path = DEFAULT_PACK_MANIFEST,
    model_manifest: Path = DEFAULT_MODEL_MANIFEST,
    bundle_manifest: Path = DEFAULT_BUNDLE_MANIFEST,
    core_stress_manifest: Path = DEFAULT_CORE_STRESS_MANIFEST,
    validate_historical_features: bool = True,
) -> dict[str, Any]:
    contract = _load_contract(
        pack_manifest=pack_manifest,
        model_manifest=model_manifest,
        bundle_manifest=bundle_manifest,
        core_stress_manifest=core_stress_manifest,
    )
    active_manifest_path = (
        WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2/active/active.json"
    )
    active = _read_json(active_manifest_path)
    signal_date, date_idx = resolve_signal_date(
        contract.pack,
        active_as_of_date=str(active["active_as_of_date"]),
        requested=requested_signal_date,
    )
    qdp_dataset_ids, qdp_paths = prep._qdp_snapshot(WORKSPACE_ROOT)
    arrays = _open_pack_arrays(contract.pack)
    sequence_path = Path(str(contract.bundle["files"]["sequence_model"]["path"]))
    if _sha256(sequence_path) != str(
        contract.bundle["files"]["sequence_model"]["sha256"]
    ):
        raise LiveInferenceError("final_sequence_bundle_hash_changed")
    checkpoint = torch.load(sequence_path, map_location="cpu", weights_only=False)
    if tuple(checkpoint["feature_names"]) != contract.feature_names:
        raise LiveInferenceError("final_sequence_557_feature_order_changed")
    normalization = seq.Normalization.from_payload(checkpoint["normalization"])
    lookback = int(checkpoint["lookback"])
    fingerprint = _stable_hash(
        {
            "schema": SCHEMA,
            "signal_date": signal_date,
            "active_manifest_sha256": _sha256(active_manifest_path),
            "pack_manifest_sha256": _sha256(pack_manifest),
            "model_manifest_sha256": _sha256(model_manifest),
            "bundle_manifest_sha256": _sha256(bundle_manifest),
            "core_stress_manifest_sha256": _sha256(core_stress_manifest),
            "bundle_fingerprint": contract.bundle["fingerprint"],
            "availability_fingerprint": contract.core_stress["fingerprint"],
            "masked_feature_names": contract.masked_feature_names,
            "historical_validation_date": HISTORICAL_VALIDATION_DATE,
            "implementation_sha256": _sha256(Path(__file__)),
        }
    )
    destination = output_root / f"signal_date={signal_date}"
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if (
            existing.get("status") == "completed"
            and existing.get("fingerprint") == fingerprint
        ):
            return existing
        raise LiveInferenceError(
            "append_only_signal_snapshot_already_exists_with_other_fingerprint"
        )
    validation: dict[str, Any] | None = None
    if validate_historical_features:
        validation_idx = tuple(
            str(value) for value in contract.pack["date_values"]
        ).index(HISTORICAL_VALIDATION_DATE)
        historical = build_feature_snapshot(
            contract=contract,
            arrays=arrays,
            signal_date=HISTORICAL_VALIDATION_DATE,
            date_idx=int(validation_idx),
            output_root=output_root,
            qdp_paths=qdp_paths,
            qdp_dataset_ids=qdp_dataset_ids,
            normalization=normalization,
            lookback=lookback,
        )
        validation = validate_against_frozen_history(
            historical, contract=contract, model_manifest_path=model_manifest
        )
        del historical
        gc.collect()
    snapshot = build_feature_snapshot(
        contract=contract,
        arrays=arrays,
        signal_date=signal_date,
        date_idx=date_idx,
        output_root=output_root,
        qdp_paths=qdp_paths,
        qdp_dataset_ids=qdp_dataset_ids,
        normalization=normalization,
        lookback=lookback,
    )
    scores, market_gate = score_snapshot(
        snapshot, contract=contract, checkpoint=checkpoint
    )
    orders = requested_orders(
        scores, top_k=int(contract.bundle["live_inference_contract"]["top_k"])
    )
    feature_path = destination / "features.parquet"
    scores_path = destination / "scores.parquet"
    market_path = destination / "market_gate.parquet"
    orders_path = destination / "requested_orders.parquet"
    validation_path = destination / "historical_feature_validation.json"
    _write_parquet(_snapshot_frame(snapshot, contract), feature_path)
    _write_parquet(scores, scores_path)
    _write_parquet(
        pd.DataFrame([{"signal_date": signal_date, **market_gate}]), market_path
    )
    _write_parquet(orders, orders_path)
    if validation is not None:
        _write_json(validation_path, validation)
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "fingerprint": fingerprint,
        "study_id": base.STUDY_ID,
        "candidate_study_id": contract.bundle["candidate_study_id"],
        "signal_date": signal_date,
        "date_idx": date_idx,
        "active_qdp_as_of_date": str(active["active_as_of_date"]),
        "calendar_lag_at_run_days": int(
            (
                pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
                - pd.Timestamp(signal_date)
            ).days
        ),
        "universe": snapshot.source_profile,
        "feature_contract": {
            "total_feature_count": 557,
            "rebuilt_core_feature_count": 314,
            "masked_feature_count": 243,
            "masked_as_nan": True,
            "availability_profile": "rebuildable_market_path_core",
        },
        "market_gate": market_gate,
        "score_count": len(scores),
        "requested_order_count": len(orders),
        "requested_symbols": orders["symbol"].astype(str).tolist()
        if len(orders)
        else [],
        "historical_feature_validation": validation,
        "decision_boundary": {
            "outcome_blind": True,
            "future_return_array_opened": False,
            "future_fill_array_opened": False,
            "post_signal_price_read": False,
            "real_order_authority": False,
            "stale_snapshot_is_not_a_current_trading_recommendation": signal_date
            < pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d"),
            "stable_profit_claim_allowed": False,
        },
        "files": {
            "features": _file_record(feature_path, row_count=len(snapshot.symbols)),
            "scores": _file_record(scores_path, row_count=len(scores)),
            "market_gate": _file_record(market_path, row_count=1),
            "requested_orders": _file_record(orders_path, row_count=len(orders)),
            **(
                {"historical_feature_validation": _file_record(validation_path)}
                if validation is not None
                else {}
            ),
        },
        "sources": {
            "active_qdp": _file_record(active_manifest_path),
            "pack": _file_record(pack_manifest),
            "model_inputs": _file_record(model_manifest),
            "final_bundle": _file_record(bundle_manifest),
            "core_availability_stress": _file_record(core_stress_manifest),
        },
    }
    _write_json(manifest_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-date")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-historical-validation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_live_inference(
        requested_signal_date=args.signal_date,
        output_root=args.output_root,
        validate_historical_features=not args.skip_historical_validation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
