"""Causal margin-balance Top10 entry, exit, path, and continuation study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from daily_research.path_policy import seq100_stock_distribution as base
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
    ResolvedPlanBatch,
    cashflow_batch,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_margin_top10_adaptive_entry_v1"
SUMMARY_SCHEMA = "seq100_margin_top10_adaptive_entry_summary/1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_margin_top10_adaptive_entry_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies/"
    "seq100_margin_top10_adaptive_entry_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_study(
    path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path]:
    study_path = base._resolve_path(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    epistemic = dict(study["epistemic_contract"])
    expected_epistemic = {
        "prior_margin_growth_probe_was_read_before_this_study": True,
        "user_formula_and_entry_rules_are_new_user_specified_hypotheses": True,
        "all_2012_2025_results_are_adaptive_retrospective_evidence": True,
        "development_calibration_and_late_period_are_sequential_checks_not_untouched_global_confirmation": True,
        "no_2026_outcome_may_be_read": True,
        "production_claim_allowed": False,
    }
    if epistemic != expected_epistemic:
        raise ValueError("epistemic_contract_mismatch")
    if int(study["source"]["forbidden_year"]) != base.FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")
    selection = dict(study["margin_selection"])
    if selection != {
        "balance_field": "rzye",
        "source_observation_must_be_available_before_next_open": True,
        "increment_definition": (
            "current reported rzye minus prior reported rzye for the same symbol"
        ),
        "consecutive_source_trading_day_required": True,
        "minimum_consecutive_increases": 3,
        "ranking": "descending absolute CNY balance increment",
        "daily_top_k": 10,
        "tie_break": "symbol ascending",
        "universe": "quality_liquidity_pit on the signal date",
        "future_outcomes_used": False,
    }:
        raise ValueError("margin_selection_contract_mismatch")
    adaptive = dict(study["adaptive_line"])
    if (
        int(adaptive["efficiency_lookback"]) != 10
        or int(adaptive["fast_n"]) != 2
        or int(adaptive["slow_n"]) != 30
        or int(adaptive["smooth_n"]) != 2
    ):
        raise ValueError("adaptive_line_contract_mismatch")
    if tuple(study["evaluation"]["full_years"]) != tuple(range(2012, 2026)):
        raise ValueError("evaluation_year_contract_mismatch")
    if tuple(study["continuation_model"]["daily_selections"]) != (1, 3):
        raise ValueError("model_selection_contract_mismatch")
    if bool(study["decision_boundary"].get("production_claim_allowed", True)):
        raise ValueError("production_claim_forbidden")
    return study, study_path


def _source_contract(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    keys = (
        "quality_liquidity_pit_manifest",
        "model_input_manifest",
        "label_manifest",
        "pack_manifest",
        "qdp_active_manifest",
        "margin_detail_manifest",
        "market_daily_manifest",
        "adjust_factor_manifest",
        "prior_margin_probe_record",
    )
    paths: dict[str, Path] = {}
    for key in keys:
        path = base._resolve_path(source[key])
        expected = str(source[f"{key}_sha256"]).lower()
        if not path.is_file() or base._sha256_file(path).lower() != expected:
            raise ValueError(f"source_invalid:{key}")
        paths[key] = path
    pool = _read_json(paths["quality_liquidity_pit_manifest"])
    if (
        pool.get("status") != "completed"
        or pool.get("pool_name") != "quality_liquidity_pit"
        or int(pool.get("forbidden_2026_rows", -1)) != 0
    ):
        raise ValueError("pool_contract_invalid")
    active = _read_json(paths["qdp_active_manifest"])
    expected_ids = {
        "margin_detail": "margin_detail__2ed8aba21121548425f56b7c",
        "market_daily_raw": "market_daily_raw__0559f722eb9142e5f476854a",
        "adjust_factor": "adjust_factor__98fbecd479934dda685046f0",
    }
    if any(active["datasets"].get(key) != value for key, value in expected_ids.items()):
        raise ValueError("active_qdp_dataset_drift")
    for key, expected_id in (
        ("margin_detail_manifest", expected_ids["margin_detail"]),
        ("market_daily_manifest", expected_ids["market_daily_raw"]),
        ("adjust_factor_manifest", expected_ids["adjust_factor"]),
    ):
        if _read_json(paths[key]).get("dataset_id") != expected_id:
            raise ValueError(f"dataset_id_mismatch:{key}")
    CandidateCompleteAuditPack(paths["pack_manifest"])
    return paths


def _qdp_shard_paths(manifest_path: Path, *, qdp_root: Path) -> list[Path]:
    manifest = _read_json(manifest_path)
    paths: list[Path] = []
    for row in manifest.get("shards", []):
        raw = Path(str(row["path"]))
        path = raw if raw.is_absolute() else qdp_root / raw
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path.resolve())
    if not paths:
        raise ValueError(f"dataset_shards_empty:{manifest_path}")
    return paths


def _pool_paths(pool_manifest_path: Path) -> list[Path]:
    manifest = _read_json(pool_manifest_path)
    paths: list[Path] = []
    for row in manifest["partitions"]:
        path = Path(str(row["path"])).resolve()
        if not path.is_file() or base._sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"pool_partition_invalid:{row['year']}")
        paths.append(path)
    return paths


def _margin_top10(
    *,
    pool_paths: Sequence[Path],
    margin_paths: Sequence[Path],
    pack: CandidateCompleteAuditPack,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET threads=6")
    connection.execute("SET memory_limit='3GB'")
    connection.register(
        "date_map",
        pd.DataFrame(
            {
                "trade_date": np.asarray(pack.date_values, dtype=str),
                "date_idx": np.arange(len(pack.date_values), dtype=np.int32),
            }
        ),
    )
    connection.from_parquet([str(path) for path in pool_paths]).create_view(
        "pool_source"
    )
    connection.from_parquet([str(path) for path in margin_paths]).create_view(
        "margin_source"
    )
    maximum_date = str(study["source"]["maximum_outcome_date"])
    minimum_streak = int(study["margin_selection"]["minimum_consecutive_increases"])
    top_k = int(study["margin_selection"]["daily_top_k"])
    query = f"""
      WITH raw_aligned AS (
        SELECT m.symbol,
               m.trade_date AS margin_source_date,
               m.feature_available_date,
               d.date_idx AS source_date_idx,
               a.date_idx AS available_date_idx,
               CAST(m.rzye AS DOUBLE) AS margin_balance,
               CAST(m.rzmre AS DOUBLE) AS financing_buy,
               CAST(m.rzche AS DOUBLE) AS financing_repayment
        FROM margin_source m
        JOIN date_map d ON d.trade_date = m.trade_date
        JOIN date_map a ON a.trade_date = m.feature_available_date
        WHERE m.trade_date <= '{maximum_date}'
          AND m.feature_available_date <= '{maximum_date}'
          AND isfinite(m.rzye) AND m.rzye > 0
      ), lagged AS (
        SELECT *,
               lag(margin_balance) OVER w AS prior_margin_balance,
               lag(source_date_idx) OVER w AS prior_source_date_idx
        FROM raw_aligned
        WINDOW w AS (PARTITION BY symbol ORDER BY source_date_idx)
      ), classified AS (
        SELECT *,
               margin_balance - prior_margin_balance AS margin_increment,
               CASE WHEN source_date_idx = prior_source_date_idx + 1
                          AND margin_balance > prior_margin_balance
                    THEN 1 ELSE 0 END AS is_consecutive_increase
        FROM lagged
      ), grouped AS (
        SELECT *,
               sum(CASE WHEN is_consecutive_increase = 0 THEN 1 ELSE 0 END)
                 OVER (PARTITION BY symbol ORDER BY source_date_idx
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS reset_group
        FROM classified
      ), derived AS (
        SELECT *,
               sum(is_consecutive_increase)
                 OVER (PARTITION BY symbol, reset_group ORDER BY source_date_idx
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS increase_streak
        FROM grouped
      ), fresh AS (
        SELECT * EXCLUDE (fresh_rank)
        FROM (
          SELECT *, row_number() OVER (
            PARTITION BY symbol, available_date_idx
            ORDER BY source_date_idx DESC
          ) AS fresh_rank
          FROM derived
        )
        WHERE fresh_rank = 1
      ), eligible AS (
        SELECT p.candidate_id, p.year AS evaluation_year, p.trade_date,
               p.date_idx, p.symbol_idx, p.symbol,
               f.margin_source_date, f.feature_available_date,
               f.source_date_idx AS margin_source_date_idx,
               f.available_date_idx AS margin_available_date_idx,
               f.margin_balance, f.prior_margin_balance, f.margin_increment,
               CAST(f.increase_streak AS INTEGER) AS margin_increase_streak,
               f.financing_buy, f.financing_repayment,
               row_number() OVER (
                 PARTITION BY p.date_idx
                 ORDER BY f.margin_increment DESC, p.symbol ASC
               ) AS margin_rank
        FROM pool_source p
        JOIN fresh f
          ON f.symbol = p.symbol
         AND f.source_date_idx = p.date_idx
         AND f.available_date_idx = p.date_idx + 1
        WHERE f.margin_increment > 0
          AND f.increase_streak >= {minimum_streak}
          AND p.trade_date <= '{maximum_date}'
      )
      SELECT * FROM eligible WHERE margin_rank <= {top_k}
      ORDER BY date_idx, margin_rank
    """
    candidates = connection.execute(query).fetch_df()
    timeline_query = f"""
      WITH raw_aligned AS (
        SELECT m.symbol,
               m.trade_date AS margin_source_date,
               m.feature_available_date,
               d.date_idx AS source_date_idx,
               a.date_idx AS available_date_idx,
               CAST(m.rzye AS DOUBLE) AS margin_balance,
               CAST(m.rzmre AS DOUBLE) AS financing_buy,
               CAST(m.rzche AS DOUBLE) AS financing_repayment
        FROM margin_source m
        JOIN date_map d ON d.trade_date = m.trade_date
        JOIN date_map a ON a.trade_date = m.feature_available_date
        WHERE m.trade_date <= '{maximum_date}'
          AND m.feature_available_date <= '{maximum_date}'
          AND isfinite(m.rzye) AND m.rzye > 0
      )
      SELECT *, margin_balance - lag(margin_balance) OVER (
        PARTITION BY symbol ORDER BY source_date_idx
      ) AS margin_increment
      FROM raw_aligned
      ORDER BY symbol, available_date_idx, source_date_idx
    """
    timeline = connection.execute(timeline_query).fetch_df()
    timeline = (
        timeline.sort_values(
            ["symbol", "available_date_idx", "source_date_idx"], kind="stable"
        )
        .drop_duplicates(["symbol", "available_date_idx"], keep="last")
        .reset_index(drop=True)
    )
    connection.close()
    if candidates.empty:
        raise ValueError("margin_top10_empty")
    if bool(candidates["trade_date"].astype(str).str.startswith("2026-").any()):
        raise ValueError("forbidden_2026_candidate")
    if not bool(
        (
            candidates["margin_source_date_idx"].to_numpy(dtype=np.int64)
            == candidates["date_idx"].to_numpy(dtype=np.int64)
        ).all()
    ):
        raise ValueError("margin_source_kline_alignment_failed")
    if not bool(
        (
            candidates["margin_available_date_idx"].to_numpy(dtype=np.int64)
            == candidates["date_idx"].to_numpy(dtype=np.int64) + 1
        ).all()
    ):
        raise ValueError("margin_next_open_availability_failed")
    counts = candidates.groupby("date_idx").size()
    audit = {
        "candidate_count": len(candidates),
        "signal_date_count": int(candidates["date_idx"].nunique()),
        "first_signal_date": str(candidates["trade_date"].min()),
        "last_signal_date": str(candidates["trade_date"].max()),
        "daily_candidate_count_minimum": int(counts.min()),
        "daily_candidate_count_median": float(counts.median()),
        "daily_candidate_count_maximum": int(counts.max()),
        "margin_symbol_count": int(candidates["symbol"].nunique()),
        "minimum_streak_observed": int(candidates["margin_increase_streak"].min()),
        "margin_source_equals_signal_date": True,
        "margin_available_before_entry_open": True,
        "future_outcomes_used": False,
    }
    return candidates, timeline, audit


def _load_dense_fields(
    manifest_path: Path,
    *,
    pack: CandidateCompleteAuditPack,
    fields: Sequence[str],
    maximum_date: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    qdp_root = Path(str(pack.manifest["qdp_root"])).resolve()
    paths = _qdp_shard_paths(manifest_path, qdp_root=qdp_root)
    shape = (len(pack.date_values), len(pack.symbol_values))
    result = {str(field): np.full(shape, np.nan, dtype=np.float32) for field in fields}
    seen = np.zeros(shape, dtype=bool)
    dates = pd.Index(np.asarray(pack.date_values, dtype=str))
    symbols = pd.Index(np.asarray(pack.symbol_values, dtype=str))
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    scanner = dataset.scanner(
        columns=["trade_date", "symbol", *fields],
        filter=ds.field("trade_date") <= str(maximum_date),
        batch_size=262_144,
    )
    row_count = 0
    duplicate_count = 0
    invalid_identity_count = 0
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        date_idx = dates.get_indexer(frame["trade_date"].astype(str))
        symbol_idx = symbols.get_indexer(frame["symbol"].astype(str))
        identity_valid = (date_idx >= 0) & (symbol_idx >= 0)
        invalid_identity_count += int((~identity_valid).sum())
        if not bool(identity_valid.all()):
            frame = frame.loc[identity_valid].reset_index(drop=True)
            date_idx = date_idx[identity_valid]
            symbol_idx = symbol_idx[identity_valid]
        duplicate_count += int(seen[date_idx, symbol_idx].sum())
        if duplicate_count:
            raise ValueError(f"dense_source_duplicate_rows:{manifest_path}")
        for field in fields:
            values = pd.to_numeric(frame[field], errors="coerce").to_numpy(
                dtype=np.float64
            )
            result[str(field)][date_idx, symbol_idx] = values.astype(np.float32)
        seen[date_idx, symbol_idx] = True
        row_count += len(frame)
    return result, {
        "manifest": str(manifest_path),
        "loaded_row_count": row_count,
        "duplicate_row_count": duplicate_count,
        "invalid_identity_row_count": invalid_identity_count,
        "forbidden_2026_read_count": 0,
        "fields": list(fields),
    }


def _adaptive_line(
    close_raw: np.ndarray,
    factor: np.ndarray,
    *,
    symbol_indices: Sequence[int],
    maximum_date_idx: int,
    study: Mapping[str, Any],
) -> np.ndarray:
    config = dict(study["adaptive_line"])
    lookback = int(config["efficiency_lookback"])
    fast = 2.0 / (int(config["fast_n"]) + 1.0)
    slow = 2.0 / (int(config["slow_n"]) + 1.0)
    ema_alpha = 2.0 / (int(config["smooth_n"]) + 1.0)
    output = np.full(np.asarray(close_raw).shape, np.nan, dtype=np.float32)
    for symbol_idx in sorted({int(value) for value in symbol_indices}):
        raw = np.asarray(
            close_raw[: int(maximum_date_idx) + 1, symbol_idx], dtype=np.float64
        )
        factors = np.asarray(
            factor[: int(maximum_date_idx) + 1, symbol_idx], dtype=np.float64
        )
        valid_dates = np.flatnonzero(
            np.isfinite(raw) & (raw > 0.0) & np.isfinite(factors) & (factors > 0.0)
        )
        if not len(valid_dates):
            continue
        price = raw[valid_dates] * factors[valid_dates]
        absolute_change = np.abs(np.diff(price))
        cumulative_path = np.r_[0.0, np.cumsum(absolute_change)]
        abase = float(price[0])
        adp = float(price[0])
        values = np.empty(len(price), dtype=np.float64)
        values[0] = adp
        for position in range(1, len(price)):
            efficiency = 0.0
            if position >= lookback:
                path = float(
                    cumulative_path[position] - cumulative_path[position - lookback]
                )
                direction = abs(float(price[position] - price[position - lookback]))
                efficiency = 0.0 if path == 0.0 else direction / path
            scaling = slow + efficiency * (fast - slow)
            alpha = scaling * scaling
            abase = alpha * float(price[position]) + (1.0 - alpha) * abase
            adp = ema_alpha * abase + (1.0 - ema_alpha) * adp
            values[position] = adp
        output[valid_dates, symbol_idx] = values.astype(np.float32)
    return output


def _attach_technical(
    candidates: pd.DataFrame,
    *,
    pack: CandidateCompleteAuditPack,
    factor: np.ndarray,
    high_raw: np.ndarray,
    low_raw: np.ndarray,
    adp: np.ndarray,
) -> pd.DataFrame:
    result = candidates.reset_index(drop=True).copy()
    columns = {
        "adjusted_open": np.full(len(result), np.nan),
        "adjusted_high": np.full(len(result), np.nan),
        "adjusted_low": np.full(len(result), np.nan),
        "adjusted_close": np.full(len(result), np.nan),
        "adp": np.full(len(result), np.nan),
        "current_return_1d": np.full(len(result), np.nan),
        "current_candle_return": np.full(len(result), np.nan),
        "close_to_adp": np.full(len(result), np.nan),
        "low_to_adp": np.full(len(result), np.nan),
        "adp_slope_1d": np.full(len(result), np.nan),
        "adp_slope_3d": np.full(len(result), np.nan),
        "cross_support": np.zeros(len(result), dtype=bool),
        "down_close": np.zeros(len(result), dtype=bool),
        "bearish_candle": np.zeros(len(result), dtype=bool),
    }
    for symbol_idx, group in result.groupby("symbol_idx", sort=False):
        symbol_idx = int(symbol_idx)
        maximum_idx = int(group["date_idx"].max())
        raw_close = np.asarray(
            pack.exit_close_raw[: maximum_idx + 1, symbol_idx], dtype=np.float64
        )
        raw_open = np.asarray(
            pack.entry_open_raw[: maximum_idx + 1, symbol_idx], dtype=np.float64
        )
        factors = np.asarray(factor[: maximum_idx + 1, symbol_idx], dtype=np.float64)
        valid_dates = np.flatnonzero(
            np.isfinite(raw_close)
            & (raw_close > 0.0)
            & np.isfinite(factors)
            & (factors > 0.0)
            & np.isfinite(adp[: maximum_idx + 1, symbol_idx])
        )
        if not len(valid_dates):
            continue
        candidate_dates = group["date_idx"].to_numpy(dtype=np.int64)
        bar_positions = np.searchsorted(valid_dates, candidate_dates)
        valid_position = (
            (bar_positions < len(valid_dates))
            & (
                valid_dates[np.minimum(bar_positions, len(valid_dates) - 1)]
                == candidate_dates
            )
            & (bar_positions >= 3)
        )
        for row_position, date_idx, bar_position, usable in zip(
            group.index.to_numpy(dtype=np.int64),
            candidate_dates,
            bar_positions,
            valid_position,
            strict=True,
        ):
            if not usable:
                continue
            prior_idx = int(valid_dates[bar_position - 1])
            prior2_idx = int(valid_dates[bar_position - 2])
            prior3_idx = int(valid_dates[bar_position - 3])
            date_idx = int(date_idx)
            current_factor = float(factors[date_idx])
            current_close = float(raw_close[date_idx] * current_factor)
            current_open = float(raw_open[date_idx] * current_factor)
            current_low = float(low_raw[date_idx, symbol_idx] * current_factor)
            current_high = float(high_raw[date_idx, symbol_idx] * current_factor)
            current_adp = float(adp[date_idx, symbol_idx])
            prior_close = float(raw_close[prior_idx] * factors[prior_idx])
            prior2_close = float(raw_close[prior2_idx] * factors[prior2_idx])
            prior_adp = float(adp[prior_idx, symbol_idx])
            prior2_adp = float(adp[prior2_idx, symbol_idx])
            prior3_adp = float(adp[prior3_idx, symbol_idx])
            columns["adjusted_open"][row_position] = current_open
            columns["adjusted_high"][row_position] = current_high
            columns["adjusted_low"][row_position] = current_low
            columns["adjusted_close"][row_position] = current_close
            columns["adp"][row_position] = current_adp
            columns["current_return_1d"][row_position] = (
                current_close / prior_close - 1.0
            )
            columns["current_candle_return"][row_position] = (
                current_close / current_open - 1.0
            )
            columns["close_to_adp"][row_position] = current_close / current_adp - 1.0
            columns["low_to_adp"][row_position] = current_low / current_adp - 1.0
            columns["adp_slope_1d"][row_position] = current_adp / prior_adp - 1.0
            columns["adp_slope_3d"][row_position] = current_adp / prior3_adp - 1.0
            columns["cross_support"][row_position] = bool(
                prior_close > prior_adp
                and prior2_close <= prior2_adp
                and current_low > current_adp
            )
            columns["down_close"][row_position] = current_close < prior_close
            columns["bearish_candle"][row_position] = current_close < current_open
    for name, values in columns.items():
        result[name] = values
    result["user_union_close"] = result["cross_support"] | result["down_close"]
    result["user_union_candle"] = result["cross_support"] | result["bearish_candle"]
    result["both_cross_and_down"] = result["cross_support"] & result["down_close"]
    result["margin_increment_signed_log"] = np.log1p(
        np.maximum(result["margin_increment"].to_numpy(dtype=np.float64), 0.0)
    )
    result["margin_increment_to_balance"] = result["margin_increment"].to_numpy(
        dtype=np.float64
    ) / result["prior_margin_balance"].to_numpy(dtype=np.float64)
    result["reported_net_financing_flow_to_balance"] = (
        result["financing_buy"].to_numpy(dtype=np.float64)
        - result["financing_repayment"].to_numpy(dtype=np.float64)
    ) / result["prior_margin_balance"].to_numpy(dtype=np.float64)
    return result


def _first_decrease_request(
    candidates: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    maximum_request_day: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    requests = candidates["date_idx"].to_numpy(dtype=np.int64) + int(
        maximum_request_day
    )
    detected = np.zeros(len(candidates), dtype=bool)
    detection_offsets = np.full(len(candidates), -1, dtype=np.int16)
    timeline_groups = {
        str(symbol): group.sort_values(["available_date_idx", "source_date_idx"])
        for symbol, group in timeline.groupby("symbol", sort=False)
    }
    for symbol, group in candidates.groupby("symbol", sort=False):
        observations = timeline_groups.get(str(symbol))
        if observations is None:
            continue
        available = observations["available_date_idx"].to_numpy(dtype=np.int64)
        increments = observations["margin_increment"].to_numpy(dtype=np.float64)
        for row_position, signal_idx in zip(
            group.index.to_numpy(dtype=np.int64),
            group["date_idx"].to_numpy(dtype=np.int64),
            strict=True,
        ):
            start = int(np.searchsorted(available, signal_idx + 1, side="left"))
            stop = int(
                np.searchsorted(
                    available, signal_idx + int(maximum_request_day), side="right"
                )
            )
            local = np.flatnonzero(
                np.isfinite(increments[start:stop]) & (increments[start:stop] < 0.0)
            )
            if not len(local):
                continue
            detection_idx = int(available[start + int(local[0])])
            request_idx = max(signal_idx + 2, detection_idx)
            requests[row_position] = min(
                request_idx, signal_idx + int(maximum_request_day)
            )
            detected[row_position] = True
            detection_offsets[row_position] = int(detection_idx - signal_idx)
    valid_offsets = detection_offsets[detection_offsets >= 0]
    return requests, {
        "candidate_count": len(candidates),
        "decrease_detected_count": int(detected.sum()),
        "decrease_detected_fraction": float(detected.mean()),
        "timeout_count": int((~detected).sum()),
        "median_detection_offset": (
            float(np.median(valid_offsets)) if len(valid_offsets) else None
        ),
        "p90_detection_offset": (
            float(np.quantile(valid_offsets, 0.9)) if len(valid_offsets) else None
        ),
        "detection_offsets": detection_offsets,
        "detected": detected,
    }


def _exact_returns_for_requests(
    candidates: pd.DataFrame,
    requests: np.ndarray,
    *,
    pack: CandidateCompleteAuditPack,
    factor: np.ndarray,
    allocated_cash: float,
    maximum_outcome_date_idx: int,
) -> dict[str, np.ndarray]:
    signal_idx = candidates["date_idx"].to_numpy(dtype=np.int64)
    symbol_idx = candidates["symbol_idx"].to_numpy(dtype=np.int64)
    entry_idx = signal_idx + 1
    valid_entry_date = entry_idx <= int(maximum_outcome_date_idx)
    entry_raw = np.full(len(candidates), np.nan, dtype=np.float64)
    entry_filled = np.zeros(len(candidates), dtype=bool)
    entry_factor = np.full(len(candidates), np.nan, dtype=np.float64)
    entry_raw[valid_entry_date] = np.asarray(
        pack.entry_open_raw[entry_idx[valid_entry_date], symbol_idx[valid_entry_date]],
        dtype=np.float64,
    )
    entry_filled[valid_entry_date] = np.asarray(
        pack.entry_filled[entry_idx[valid_entry_date], symbol_idx[valid_entry_date]],
        dtype=bool,
    )
    entry_factor[valid_entry_date] = factor[
        entry_idx[valid_entry_date], symbol_idx[valid_entry_date]
    ].astype(np.float64)
    actual_exit = np.full(len(candidates), -1, dtype=np.int32)
    exit_raw = np.full(len(candidates), np.nan, dtype=np.float64)
    for position, (requested, symbol) in enumerate(
        zip(np.asarray(requests, dtype=np.int64), symbol_idx, strict=True)
    ):
        stop = min(int(requested) + 21, int(maximum_outcome_date_idx) + 1)
        if int(requested) < 0 or int(requested) >= stop:
            continue
        sellable = np.asarray(
            pack.exit_sellable[int(requested) : stop, int(symbol)], dtype=bool
        )
        close = np.asarray(
            pack.exit_close_raw[int(requested) : stop, int(symbol)], dtype=np.float64
        )
        local = np.flatnonzero(sellable & np.isfinite(close) & (close > 0.0))
        if not len(local):
            continue
        date_idx = int(requested) + int(local[0])
        actual_exit[position] = date_idx
        exit_raw[position] = float(pack.exit_close_raw[date_idx, int(symbol)])
    exit_factor = np.full(len(candidates), np.nan, dtype=np.float64)
    valid_exit = actual_exit >= 0
    exit_factor[valid_exit] = factor[
        actual_exit[valid_exit], symbol_idx[valid_exit]
    ].astype(np.float64)
    valid = (
        entry_filled
        & np.isfinite(entry_raw)
        & (entry_raw > 0.0)
        & np.isfinite(entry_factor)
        & (entry_factor > 0.0)
        & valid_exit
        & np.isfinite(exit_raw)
        & np.isfinite(exit_factor)
        & (exit_factor > 0.0)
    )
    economic_exit = np.where(valid, exit_raw * exit_factor / entry_factor, np.nan)
    planned = np.clip(np.asarray(requests) - signal_idx, 2, 60).astype(np.int16)
    exit_day = np.where(valid, actual_exit - signal_idx, -1).astype(np.int16)
    plan = ResolvedPlanBatch(
        planned_day=planned,
        exit_day=exit_day,
        exit_date_idx=actual_exit,
        exit_price=economic_exit,
        terminal_recovery=np.zeros(len(candidates), dtype=bool),
    )
    cashflow = cashflow_batch(
        allocated_cash=float(allocated_cash),
        entry_filled=entry_filled,
        entry_prices=entry_raw,
        plan=plan,
        date_values=pack.date_values,
        contract=pack.costs,
        slippage_multiplier=float(pack.costs.stress_slippage_multiplier),
    )
    gross_return = np.where(valid, economic_exit / entry_raw - 1.0, np.nan)
    return {
        "entry_date_idx": entry_idx.astype(np.int32),
        "entry_price_raw": entry_raw,
        "entry_filled": cashflow.order_filled,
        "requested_exit_date_idx": np.asarray(requests, dtype=np.int32),
        "actual_exit_date_idx": actual_exit,
        "exit_price_raw": exit_raw,
        "gross_adjusted_return": gross_return,
        "net_return": np.where(cashflow.order_filled, cashflow.net_return, np.nan),
        "shares": cashflow.shares,
        "total_cost_cny": cashflow.total_cost,
        "cash_utilization": cashflow.cash_utilization,
    }


def _take_profit_returns(
    candidates: pd.DataFrame,
    *,
    threshold: float,
    pack: CandidateCompleteAuditPack,
    allocated_cash: float,
    maximum_outcome_date_idx: int,
) -> dict[str, np.ndarray]:
    signal_idx = candidates["date_idx"].to_numpy(dtype=np.int64)
    symbol_idx = candidates["symbol_idx"].to_numpy(dtype=np.int64)
    entry_idx = signal_idx + 1
    entry_raw = candidates["one_day_entry_price_raw"].to_numpy(dtype=np.float64)
    raw_entry_filled = np.zeros(len(candidates), dtype=bool)
    valid_entry_date = entry_idx <= int(maximum_outcome_date_idx)
    raw_entry_filled[valid_entry_date] = np.asarray(
        pack.entry_filled[entry_idx[valid_entry_date], symbol_idx[valid_entry_date]],
        dtype=bool,
    )
    high_return = candidates["legal_d2_high_return"].to_numpy(dtype=np.float64)
    open_return = candidates["legal_d2_open_return"].to_numpy(dtype=np.float64)
    target_hit = (
        raw_entry_filled & np.isfinite(high_return) & (high_return >= float(threshold))
    )
    fill_return = np.fmax(open_return, float(threshold))
    fallback_return = candidates["one_day_gross_adjusted_return"].to_numpy(
        dtype=np.float64
    )
    gross_return = np.where(target_hit, fill_return, fallback_return)
    fallback_exit = candidates["one_day_actual_exit_date_idx"].to_numpy(dtype=np.int64)
    actual_exit = np.where(target_hit, signal_idx + 2, fallback_exit).astype(np.int32)
    valid = (
        raw_entry_filled
        & np.isfinite(entry_raw)
        & (entry_raw > 0.0)
        & np.isfinite(gross_return)
        & (actual_exit >= 0)
        & (actual_exit <= int(maximum_outcome_date_idx))
    )
    economic_exit = np.where(valid, entry_raw * (1.0 + gross_return), np.nan)
    plan = ResolvedPlanBatch(
        planned_day=np.full(len(candidates), 2, dtype=np.int16),
        exit_day=np.where(valid, actual_exit - signal_idx, -1).astype(np.int16),
        exit_date_idx=np.where(valid, actual_exit, -1).astype(np.int32),
        exit_price=economic_exit,
        terminal_recovery=np.zeros(len(candidates), dtype=bool),
    )
    cashflow = cashflow_batch(
        allocated_cash=float(allocated_cash),
        entry_filled=raw_entry_filled,
        entry_prices=entry_raw,
        plan=plan,
        date_values=pack.date_values,
        contract=pack.costs,
        slippage_multiplier=float(pack.costs.stress_slippage_multiplier),
    )
    return {
        "target_hit": target_hit,
        "gross_adjusted_return": np.where(cashflow.order_filled, gross_return, np.nan),
        "net_return": np.where(cashflow.order_filled, cashflow.net_return, np.nan),
        "actual_exit_date_idx": actual_exit,
    }


def _attach_paths_and_exits(
    candidates: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    pack: CandidateCompleteAuditPack,
    factor: np.ndarray,
    high_raw: np.ndarray,
    low_raw: np.ndarray,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = candidates.copy()
    signal_idx = result["date_idx"].to_numpy(dtype=np.int64)
    symbol_idx = result["symbol_idx"].to_numpy(dtype=np.int64)
    cutoff = int(
        np.flatnonzero(
            np.asarray(pack.date_values, dtype=str)
            == str(study["source"]["maximum_outcome_date"])
        )[0]
    )
    close_raw = pack.exit_close_raw
    signal_close = np.asarray(
        close_raw[signal_idx, symbol_idx], dtype=np.float64
    ) * factor[signal_idx, symbol_idx].astype(np.float64)
    entry_idx = signal_idx + 1
    entry_adjusted = np.full(len(result), np.nan, dtype=np.float64)
    valid_entry_date = entry_idx <= cutoff
    entry_adjusted[valid_entry_date] = np.asarray(
        pack.entry_open_raw[entry_idx[valid_entry_date], symbol_idx[valid_entry_date]],
        dtype=np.float64,
    ) * factor[entry_idx[valid_entry_date], symbol_idx[valid_entry_date]].astype(
        np.float64
    )
    legal_idx = signal_idx + 2
    valid_legal_date = legal_idx <= cutoff
    legal_factor = np.full(len(result), np.nan, dtype=np.float64)
    legal_open = np.full(len(result), np.nan, dtype=np.float64)
    legal_high = np.full(len(result), np.nan, dtype=np.float64)
    legal_low = np.full(len(result), np.nan, dtype=np.float64)
    legal_factor[valid_legal_date] = factor[
        legal_idx[valid_legal_date], symbol_idx[valid_legal_date]
    ].astype(np.float64)
    legal_open[valid_legal_date] = (
        np.asarray(
            pack.entry_open_raw[
                legal_idx[valid_legal_date], symbol_idx[valid_legal_date]
            ],
            dtype=np.float64,
        )
        * legal_factor[valid_legal_date]
    )
    legal_high[valid_legal_date] = (
        high_raw[legal_idx[valid_legal_date], symbol_idx[valid_legal_date]].astype(
            np.float64
        )
        * legal_factor[valid_legal_date]
    )
    legal_low[valid_legal_date] = (
        low_raw[legal_idx[valid_legal_date], symbol_idx[valid_legal_date]].astype(
            np.float64
        )
        * legal_factor[valid_legal_date]
    )
    result["legal_d2_open_return"] = legal_open / entry_adjusted - 1.0
    result["legal_d2_high_return"] = legal_high / entry_adjusted - 1.0
    result["legal_d2_low_return"] = legal_low / entry_adjusted - 1.0
    for horizon in study["path_diagnostics"]["close_horizons_from_signal"]:
        horizon = int(horizon)
        indices = signal_idx + horizon
        in_bounds = indices <= cutoff
        adjusted_close = np.full(len(result), np.nan, dtype=np.float64)
        adjusted_close[in_bounds] = np.asarray(
            close_raw[indices[in_bounds], symbol_idx[in_bounds]], dtype=np.float64
        ) * factor[indices[in_bounds], symbol_idx[in_bounds]].astype(np.float64)
        result[f"signal_close_return_d{horizon}"] = adjusted_close / signal_close - 1.0
        result[f"entry_to_close_return_d{horizon}"] = (
            adjusted_close / entry_adjusted - 1.0
        )
    for horizon in study["path_diagnostics"]["mfe_mae_horizons_from_signal"]:
        horizon = int(horizon)
        mfe = np.full(len(result), np.nan, dtype=np.float64)
        mae = np.full(len(result), np.nan, dtype=np.float64)
        for position, (date_idx, symbol) in enumerate(
            zip(signal_idx, symbol_idx, strict=True)
        ):
            stop = min(int(date_idx) + horizon + 1, cutoff + 1)
            dates = np.arange(int(date_idx) + 1, stop, dtype=np.int64)
            if not len(dates) or not math.isfinite(entry_adjusted[position]):
                continue
            factors = factor[dates, int(symbol)].astype(np.float64)
            highs = high_raw[dates, int(symbol)].astype(np.float64) * factors
            lows = low_raw[dates, int(symbol)].astype(np.float64) * factors
            if bool(np.isfinite(highs).any()):
                mfe[position] = float(np.nanmax(highs)) / entry_adjusted[position] - 1.0
            if bool(np.isfinite(lows).any()):
                mae[position] = float(np.nanmin(lows)) / entry_adjusted[position] - 1.0
        result[f"mfe_d{horizon}"] = mfe
        result[f"mae_d{horizon}"] = mae
    result["next_open_gap"] = entry_adjusted / signal_close - 1.0
    one_day_requests = signal_idx + 2
    one_day = _exact_returns_for_requests(
        result,
        one_day_requests,
        pack=pack,
        factor=factor,
        allocated_cash=float(study["exits"]["candidate_trade_cash_cny"]),
        maximum_outcome_date_idx=cutoff,
    )
    for name, values in one_day.items():
        result[f"one_day_{name}"] = values
    take_profit_config = study["path_diagnostics"]["adaptive_take_profit_sensitivity"]
    for threshold in take_profit_config["gross_return_thresholds"]:
        threshold = float(threshold)
        label = f"take_profit_{round(threshold * 10_000):04d}bp"
        target_result = _take_profit_returns(
            result,
            threshold=threshold,
            pack=pack,
            allocated_cash=float(study["exits"]["candidate_trade_cash_cny"]),
            maximum_outcome_date_idx=cutoff,
        )
        for name, values in target_result.items():
            result[f"{label}_{name}"] = values
    dynamic_requests, dynamic_audit = _first_decrease_request(
        result,
        timeline,
        maximum_request_day=int(
            study["exits"]["first_financing_decrease"]["maximum_request_day"]
        ),
    )
    dynamic = _exact_returns_for_requests(
        result,
        dynamic_requests,
        pack=pack,
        factor=factor,
        allocated_cash=float(study["exits"]["candidate_trade_cash_cny"]),
        maximum_outcome_date_idx=cutoff,
    )
    for name, values in dynamic.items():
        result[f"decrease_{name}"] = values
    result["decrease_detected"] = dynamic_audit.pop("detected")
    result["decrease_detection_offset"] = dynamic_audit.pop("detection_offsets")
    result["dynamic_primary_evaluable"] = signal_idx + 60 <= cutoff
    result["next_close_up"] = result["signal_close_return_d1"] > 0.0
    result["legal_one_day_net_positive"] = result["one_day_net_return"] > 0.0
    result["path_shape"] = _path_shapes(result)
    return result, dynamic_audit


def _path_shapes(frame: pd.DataFrame) -> np.ndarray:
    d1 = frame["signal_close_return_d1"].to_numpy(dtype=np.float64)
    entry_day = frame["entry_to_close_return_d1"].to_numpy(dtype=np.float64)
    d2 = frame["one_day_net_return"].to_numpy(dtype=np.float64)
    d5 = frame["entry_to_close_return_d5"].to_numpy(dtype=np.float64)
    d10 = frame["entry_to_close_return_d10"].to_numpy(dtype=np.float64)
    gap = frame["next_open_gap"].to_numpy(dtype=np.float64)
    result = np.full(len(frame), "mixed", dtype=object)
    result[(gap > 0.01) & (entry_day < 0.0)] = "gap_up_fade"
    result[(d1 > 0.0) & (d2 > 0.0)] = "immediate_continuation"
    result[(d1 > 0.0) & (d2 <= 0.0)] = "one_day_only_then_fail"
    result[(d1 <= 0.0) & (d2 > 0.0)] = "delayed_one_day_reversal"
    result[(d2 <= 0.0) & (d5 <= 0.0) & (d10 <= 0.0)] = "persistent_loss"
    result[(d2 <= 0.0) & (d10 > 0.0)] = "late_recovery"
    return result


def _periods(study: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        "full_history": tuple(
            int(value) for value in study["evaluation"]["full_years"]
        ),
        **{
            str(name): tuple(int(value) for value in values)
            for name, values in study["evaluation"]["periods"].items()
        },
    }


def _inference(
    values: np.ndarray, study: Mapping[str, Any], *, seed_add: int
) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "mean": math.nan,
            "standard_error": math.nan,
            "lcb_95": math.nan,
            "ucb_95": math.nan,
            "n": 0,
            "block": {"lcb_95": math.nan, "ucb_95": math.nan},
        }
    return {
        **base._hac_mean(x, lag=int(study["evaluation"]["hac_lag_days"])),
        "block": base._block_interval(
            x,
            block_length=int(study["evaluation"]["block_length_days"]),
            repetitions=int(study["evaluation"]["bootstrap_repetitions"]),
            seed=int(study["evaluation"]["seed"]) + int(seed_add),
        ),
    }


def _signal_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all_margin_top10": np.ones(len(frame), dtype=bool),
        "cross_support": frame["cross_support"].to_numpy(dtype=bool),
        "down_close": frame["down_close"].to_numpy(dtype=bool),
        "bearish_candle": frame["bearish_candle"].to_numpy(dtype=bool),
        "user_union_close": frame["user_union_close"].to_numpy(dtype=bool),
        "user_union_candle": frame["user_union_candle"].to_numpy(dtype=bool),
        "both_cross_and_down": frame["both_cross_and_down"].to_numpy(dtype=bool),
    }


def _daily_rule_frame(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    return_column: str,
    denominator_includes_unfilled: bool = True,
) -> pd.DataFrame:
    selected = frame.loc[mask].copy()
    selected["selected_return"] = selected[return_column]
    if denominator_includes_unfilled:
        selected["cash_return"] = selected["selected_return"].fillna(0.0)
    else:
        selected = selected.loc[selected["selected_return"].notna()]
        selected["cash_return"] = selected["selected_return"]
    return (
        selected.groupby(["date_idx", "trade_date", "evaluation_year"], as_index=False)
        .agg(
            selected_count=("candidate_id", "size"),
            filled_count=("selected_return", "count"),
            daily_net_return=("cash_return", "mean"),
            candidate_mean_net_return=("selected_return", "mean"),
            winning_candidate_fraction=(
                "selected_return",
                lambda values: (
                    float(np.mean(np.asarray(values.dropna()) > 0.0))
                    if len(values.dropna())
                    else math.nan
                ),
            ),
        )
        .sort_values("date_idx", kind="stable")
    )


def _rule_summaries(
    frame: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    masks = _signal_masks(frame)
    daily_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    seed_add = 0
    for exit_name, return_column, extra_mask in (
        ("legal_one_day", "one_day_net_return", np.ones(len(frame), dtype=bool)),
        (
            "first_financing_decrease_or_d60",
            "decrease_net_return",
            frame["dynamic_primary_evaluable"].to_numpy(dtype=bool),
        ),
    ):
        all_daily = _daily_rule_frame(
            frame,
            extra_mask,
            return_column=return_column,
        ).rename(columns={"daily_net_return": "baseline_daily_return"})
        for rule, mask in masks.items():
            selected_mask = mask & extra_mask
            daily = _daily_rule_frame(
                frame,
                selected_mask,
                return_column=return_column,
            )
            daily["exit"] = exit_name
            daily["rule"] = rule
            daily = daily.merge(
                all_daily[["date_idx", "baseline_daily_return"]],
                on="date_idx",
                how="left",
                validate="one_to_one",
            )
            daily["paired_delta_vs_all_top10"] = (
                daily["daily_net_return"] - daily["baseline_daily_return"]
            )
            daily_frames.append(daily)
            for period, years in _periods(study).items():
                local = daily.loc[daily["evaluation_year"].isin(set(years))]
                selected_candidates = frame.loc[
                    selected_mask
                    & frame["evaluation_year"].isin(set(years)).to_numpy(dtype=bool)
                ]
                valid_returns = (
                    selected_candidates[return_column]
                    .dropna()
                    .to_numpy(dtype=np.float64)
                )
                if local.empty:
                    continue
                annual = local.groupby("evaluation_year")["daily_net_return"].mean()
                summaries.append(
                    {
                        "period": period,
                        "exit": exit_name,
                        "rule": rule,
                        "signal_date_count": len(local),
                        "selected_candidate_count": len(selected_candidates),
                        "filled_candidate_count": len(valid_returns),
                        "fill_fraction": float(
                            len(valid_returns) / max(len(selected_candidates), 1)
                        ),
                        "candidate_mean_net_return": (
                            float(np.mean(valid_returns))
                            if len(valid_returns)
                            else math.nan
                        ),
                        "candidate_median_net_return": (
                            float(np.median(valid_returns))
                            if len(valid_returns)
                            else math.nan
                        ),
                        "candidate_winning_fraction": (
                            float(np.mean(valid_returns > 0.0))
                            if len(valid_returns)
                            else math.nan
                        ),
                        "daily_cash_return": _inference(
                            local["daily_net_return"].to_numpy(),
                            study,
                            seed_add=seed_add,
                        ),
                        "paired_delta_vs_all_top10": _inference(
                            local["paired_delta_vs_all_top10"].to_numpy(),
                            study,
                            seed_add=1000 + seed_add,
                        ),
                        "positive_year_count": int((annual > 0.0).sum()),
                        "year_count": len(annual),
                        "worst_annual_daily_mean": float(annual.min()),
                    }
                )
                seed_add += 1
    daily_output = pd.concat(daily_frames, ignore_index=True)
    primary = next(
        row
        for row in summaries
        if row["period"] == "full_history"
        and row["exit"] == "legal_one_day"
        and row["rule"] == "user_union_close"
    )
    late = next(
        row
        for row in summaries
        if row["period"] == "late"
        and row["exit"] == "legal_one_day"
        and row["rule"] == "user_union_close"
    )
    contract = dict(study["evaluation"]["entry_gate"])
    checks = {
        "full_history_hac_lower_bound_positive": float(
            primary["daily_cash_return"]["lcb_95"]
        )
        > 0.0,
        "full_history_block_lower_bound_positive": float(
            primary["daily_cash_return"]["block"]["lcb_95"]
        )
        > 0.0,
        "minimum_positive_years": int(primary["positive_year_count"])
        >= int(contract["minimum_positive_years"]),
        "late_period_mean_positive": float(late["daily_cash_return"]["mean"]) > 0.0,
        "paired_increment_hac_lower_bound_positive": float(
            primary["paired_delta_vs_all_top10"]["lcb_95"]
        )
        > 0.0,
    }
    return summaries, daily_output, {"passed": all(checks.values()), "checks": checks}


def _failure_summaries(
    frame: pd.DataFrame, study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    masks = _signal_masks(frame)
    for period, years in _periods(study).items():
        year_mask = frame["evaluation_year"].isin(set(years)).to_numpy(dtype=bool)
        for rule, rule_mask in masks.items():
            local = frame.loc[
                year_mask & rule_mask & frame["one_day_net_return"].notna()
            ]
            if local.empty:
                continue
            losses = local.loc[local["one_day_net_return"] <= 0.0]
            shape = local["path_shape"].value_counts(normalize=True)
            records.append(
                {
                    "period": period,
                    "rule": rule,
                    "candidate_count": len(local),
                    "loss_count": len(losses),
                    "loss_fraction": float(len(losses) / len(local)),
                    "mean_next_open_gap": float(local["next_open_gap"].mean()),
                    "mean_illegal_entry_day_return": float(
                        local["entry_to_close_return_d1"].mean()
                    ),
                    "mean_legal_d2_net_return": float(
                        local["one_day_net_return"].mean()
                    ),
                    "mean_d5_entry_return": float(
                        local["entry_to_close_return_d5"].mean()
                    ),
                    "loss_gap_up_fraction": float(
                        (losses["next_open_gap"] > 0.01).mean()
                    )
                    if len(losses)
                    else math.nan,
                    "loss_entry_day_positive_fraction": float(
                        (losses["entry_to_close_return_d1"] > 0.0).mean()
                    )
                    if len(losses)
                    else math.nan,
                    "loss_recovered_by_d5_fraction": float(
                        (losses["entry_to_close_return_d5"] > 0.0).mean()
                    )
                    if len(losses)
                    else math.nan,
                    "shape_fractions": {
                        str(name): float(value) for name, value in shape.items()
                    },
                }
            )
    return records


def _take_profit_summaries(
    frame: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    masks = _signal_masks(frame)
    thresholds = study["path_diagnostics"]["adaptive_take_profit_sensitivity"][
        "gross_return_thresholds"
    ]
    seed_add = 6000
    for threshold in thresholds:
        threshold = float(threshold)
        label = f"take_profit_{round(threshold * 10_000):04d}bp"
        return_column = f"{label}_net_return"
        hit_column = f"{label}_target_hit"
        for rule, mask in masks.items():
            target_daily = _daily_rule_frame(
                frame, mask, return_column=return_column
            ).rename(columns={"daily_net_return": "target_daily_net_return"})
            baseline_daily = _daily_rule_frame(
                frame, mask, return_column="one_day_net_return"
            )[["date_idx", "daily_net_return"]].rename(
                columns={"daily_net_return": "close_daily_net_return"}
            )
            daily = target_daily.merge(
                baseline_daily, on="date_idx", how="left", validate="one_to_one"
            )
            daily["paired_delta_vs_close"] = (
                daily["target_daily_net_return"] - daily["close_daily_net_return"]
            )
            daily["threshold"] = threshold
            daily["rule"] = rule
            daily_frames.append(daily)
            for period, years in _periods(study).items():
                local = daily.loc[daily["evaluation_year"].isin(set(years))]
                selected = frame.loc[
                    mask
                    & frame["evaluation_year"].isin(set(years)).to_numpy(dtype=bool)
                ]
                valid_returns = (
                    selected[return_column].dropna().to_numpy(dtype=np.float64)
                )
                if local.empty:
                    continue
                annual = local.groupby("evaluation_year")[
                    "target_daily_net_return"
                ].mean()
                records.append(
                    {
                        "period": period,
                        "rule": rule,
                        "gross_target_return": threshold,
                        "selected_candidate_count": len(selected),
                        "filled_candidate_count": len(valid_returns),
                        "target_hit_fraction": float(selected[hit_column].mean()),
                        "candidate_mean_net_return": (
                            float(np.mean(valid_returns))
                            if len(valid_returns)
                            else math.nan
                        ),
                        "candidate_median_net_return": (
                            float(np.median(valid_returns))
                            if len(valid_returns)
                            else math.nan
                        ),
                        "candidate_winning_fraction": (
                            float(np.mean(valid_returns > 0.0))
                            if len(valid_returns)
                            else math.nan
                        ),
                        "daily_cash_return": _inference(
                            local["target_daily_net_return"].to_numpy(),
                            study,
                            seed_add=seed_add,
                        ),
                        "paired_delta_vs_close": _inference(
                            local["paired_delta_vs_close"].to_numpy(),
                            study,
                            seed_add=seed_add + 1000,
                        ),
                        "positive_year_count": int((annual > 0.0).sum()),
                        "year_count": len(annual),
                        "adaptive_diagnostic_only": True,
                    }
                )
                seed_add += 1
    return records, pd.concat(daily_frames, ignore_index=True)


def _attach_compact_features(
    frame: pd.DataFrame,
    *,
    panel: base.StockPanel,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, tuple[str, ...]]:
    identity = panel.row_index[["candidate_id"]].copy()
    identity["model_row_position"] = np.arange(len(identity), dtype=np.int64)
    result = frame.merge(identity, on="candidate_id", how="left", validate="one_to_one")
    compact_names = tuple(
        str(value) for value in study["continuation_model"]["compact_features"]
    )
    compact_positions = base._feature_positions(panel, compact_names)
    matrix = np.full((len(result), len(compact_names)), np.nan, dtype=np.float32)
    valid = result["model_row_position"].notna().to_numpy(dtype=bool)
    rows = result.loc[valid, "model_row_position"].to_numpy(dtype=np.int64)
    matrix[valid] = base._feature_matrix(panel, rows, compact_positions)
    study_names = tuple(
        str(value) for value in study["continuation_model"]["study_features"]
    )
    study_matrix = np.column_stack(
        [result[name].to_numpy(dtype=np.float32) for name in study_names]
    ).astype(np.float32)
    full = np.column_stack([matrix, study_matrix]).astype(np.float32)
    return result, full, (*compact_names, *study_names)


def _date_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = (
        frame.groupby("date_idx")["candidate_id"]
        .transform("size")
        .to_numpy(dtype=np.float64)
    )
    return 1.0 / np.maximum(counts, 1.0)


def _platt_fit(
    probability: np.ndarray, target: np.ndarray, weight: np.ndarray
) -> LogisticRegression:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-5, 1.0 - 1.0e-5)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0e6, solver="lbfgs", max_iter=1000)
    model.fit(logit, np.asarray(target, dtype=np.int8), sample_weight=weight)
    return model


def _platt_predict(model: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-5, 1.0 - 1.0e-5)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return model.predict_proba(logit)[:, 1]


def _model_study(
    frame: pd.DataFrame,
    features: np.ndarray,
    feature_names: Sequence[str],
    *,
    study: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    config = dict(study["continuation_model"])
    development = {int(value) for value in config["development_years"]}
    calibration = {int(value) for value in config["platt_calibration_years"]}
    test = {int(value) for value in config["test_years"]}
    scores = frame[
        [
            "candidate_id",
            "date_idx",
            "trade_date",
            "evaluation_year",
            "symbol",
            "margin_rank",
        ]
    ].copy()
    importance_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    lgb_config = dict(config["lightgbm"])
    lgb_config["n_jobs"] = 6
    lgb_config["verbosity"] = -1
    lgb_config.pop("parameter_grid_performed")
    for target_name in config["targets"]:
        if target_name == "next_close_up":
            valid_target = frame["signal_close_return_d1"].notna().to_numpy(dtype=bool)
        else:
            valid_target = frame["one_day_net_return"].notna().to_numpy(dtype=bool)
        target = frame[target_name].to_numpy(dtype=bool).astype(np.int8)
        years = frame["evaluation_year"].to_numpy(dtype=np.int16)
        feature_valid = np.isfinite(features).any(axis=1)
        train_mask = np.isin(years, list(development)) & valid_target & feature_valid
        calibration_mask = (
            np.isin(years, list(calibration)) & valid_target & feature_valid
        )
        test_scoring_mask = np.isin(years, list(test)) & feature_valid
        test_metric_mask = test_scoring_mask & valid_target
        model = lgb.LGBMClassifier(**lgb_config)
        model.fit(
            features[train_mask],
            target[train_mask],
            sample_weight=_date_weights(frame.loc[train_mask]),
            feature_name=list(feature_names),
        )
        calibration_raw = model.predict_proba(features[calibration_mask])[:, 1]
        platt = _platt_fit(
            calibration_raw,
            target[calibration_mask],
            _date_weights(frame.loc[calibration_mask]),
        )
        all_probability = np.full(len(frame), np.nan, dtype=np.float64)
        predicted_mask = feature_valid
        raw = model.predict_proba(features[predicted_mask])[:, 1]
        all_probability[predicted_mask] = _platt_predict(platt, raw)
        scores[f"probability__{target_name}"] = all_probability
        test_metric_probability = all_probability[test_metric_mask]
        auc = roc_auc_score(target[test_metric_mask], test_metric_probability)
        brier = brier_score_loss(target[test_metric_mask], test_metric_probability)
        gain = model.booster_.feature_importance(importance_type="gain")
        split = model.booster_.feature_importance(importance_type="split")
        for name, gain_value, split_value in zip(
            feature_names, gain, split, strict=True
        ):
            importance_rows.append(
                {
                    "target": target_name,
                    "feature": str(name),
                    "gain": float(gain_value),
                    "split": int(split_value),
                }
            )
        # Candidate selection must use only information available on the signal
        # date.  In particular, do not remove an unfilled future entry before
        # ranking: an unfilled order remains cash with a zero return.
        test_frame = frame.loc[test_scoring_mask].copy()
        test_frame["probability"] = all_probability[test_scoring_mask]
        test_frame["target_value"] = target[test_scoring_mask]
        baseline_daily = (
            test_frame.assign(cash_return=test_frame["one_day_net_return"].fillna(0.0))
            .groupby("date_idx", as_index=False)["cash_return"]
            .mean()
            .rename(columns={"cash_return": "baseline_daily_return"})
        )
        for top_k in config["daily_selections"]:
            selected = (
                test_frame.sort_values(
                    ["date_idx", "probability", "symbol"],
                    ascending=[True, False, True],
                    kind="stable",
                )
                .groupby("date_idx", sort=False)
                .head(int(top_k))
                .copy()
            )
            selected["cash_return"] = selected["one_day_net_return"].fillna(0.0)
            selected["outcome_observed"] = selected["signal_close_return_d1"].notna()
            selected["entry_filled_observed"] = selected["one_day_net_return"].notna()
            daily = (
                selected.groupby(
                    ["date_idx", "trade_date", "evaluation_year"], as_index=False
                )
                .agg(
                    selected_count=("candidate_id", "size"),
                    filled_count=("entry_filled_observed", "sum"),
                    continuation_observed_count=("outcome_observed", "sum"),
                    continuation_fraction=("next_close_up", "mean"),
                    legal_win_fraction=("legal_one_day_net_positive", "mean"),
                    daily_net_return=("cash_return", "mean"),
                    mean_probability=("probability", "mean"),
                )
                .merge(baseline_daily, on="date_idx", how="left", validate="one_to_one")
            )
            daily["paired_delta"] = (
                daily["daily_net_return"] - daily["baseline_daily_return"]
            )
            annual = daily.groupby("evaluation_year")["daily_net_return"].mean()
            summaries.append(
                {
                    "target": target_name,
                    "test_period": "2023_2025",
                    "top_k": int(top_k),
                    "test_auc": float(auc),
                    "test_brier": float(brier),
                    "selected_candidate_count": len(selected),
                    "signal_date_count": int(selected["date_idx"].nunique()),
                    "fill_fraction": float(selected["entry_filled_observed"].mean()),
                    "continuation_observed_fraction": float(
                        selected["outcome_observed"].mean()
                    ),
                    "mean_predicted_probability": float(selected["probability"].mean()),
                    "continuation_fraction": float(selected["next_close_up"].mean()),
                    "legal_winning_fraction": float(
                        selected["legal_one_day_net_positive"].mean()
                    ),
                    "legal_candidate_mean_net_return": float(
                        selected["one_day_net_return"].mean()
                    ),
                    "daily_cash_return": _inference(
                        daily["daily_net_return"].to_numpy(),
                        study,
                        seed_add=3000 + len(summaries),
                    ),
                    "paired_delta_vs_all_top10": _inference(
                        daily["paired_delta"].to_numpy(),
                        study,
                        seed_add=4000 + len(summaries),
                    ),
                    "positive_year_count": int((annual > 0.0).sum()),
                    "year_count": len(annual),
                }
            )
    importance = pd.DataFrame(importance_rows).sort_values(
        ["target", "gain"], ascending=[True, False], kind="stable"
    )
    return scores, summaries, importance


def _representative_cases(
    frame: pd.DataFrame, study: Mapping[str, Any]
) -> pd.DataFrame:
    count = int(
        study["path_diagnostics"]["representative_best_and_worst_case_count_each"]
    )
    selected = frame.loc[
        frame["user_union_close"]
        & frame["one_day_net_return"].notna()
        & frame["evaluation_year"].isin({2023, 2024, 2025})
    ].copy()
    columns = [
        "candidate_id",
        "trade_date",
        "symbol",
        "margin_rank",
        "margin_increment",
        "margin_increase_streak",
        "cross_support",
        "down_close",
        "next_open_gap",
        "entry_to_close_return_d1",
        "one_day_net_return",
        "entry_to_close_return_d5",
        "mfe_d5",
        "mae_d5",
        "path_shape",
    ]
    best = selected.nlargest(count, "one_day_net_return")[columns].copy()
    best["case_side"] = "best"
    worst = selected.nsmallest(count, "one_day_net_return")[columns].copy()
    worst["case_side"] = "worst"
    return pd.concat([best, worst], ignore_index=True)


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, frozen_path = load_study(study_path)
    sources = _source_contract(study)
    root = base._resolve_path(output_root)
    summary_path = root / "summary.json"
    fingerprint = _payload_hash(
        {
            "study_sha256": base._sha256_file(frozen_path),
            "implementation_sha256": base._sha256_file(Path(__file__)),
            "source_hashes": {
                key: base._sha256_file(path) for key, path in sources.items()
            },
        }
    )
    if summary_path.is_file() and not force:
        current = _read_json(summary_path)
        if current.get("experiment_fingerprint") != fingerprint:
            raise ValueError("existing_output_fingerprint_mismatch")
        if all(
            base._record_valid(record, verify_hash=True)
            for record in current.get("files", {}).values()
        ):
            return current
    pack = CandidateCompleteAuditPack(sources["pack_manifest"])
    qdp_root = Path(str(pack.manifest["qdp_root"])).resolve()
    candidates, timeline, selection_audit = _margin_top10(
        pool_paths=_pool_paths(sources["quality_liquidity_pit_manifest"]),
        margin_paths=_qdp_shard_paths(
            sources["margin_detail_manifest"], qdp_root=qdp_root
        ),
        pack=pack,
        study=study,
    )
    factor_fields, factor_audit = _load_dense_fields(
        sources["adjust_factor_manifest"],
        pack=pack,
        fields=["adjust_factor"],
        maximum_date=str(study["source"]["maximum_outcome_date"]),
    )
    market_fields, market_audit = _load_dense_fields(
        sources["market_daily_manifest"],
        pack=pack,
        fields=["high", "low"],
        maximum_date=str(study["source"]["maximum_outcome_date"]),
    )
    factor = factor_fields["adjust_factor"]
    high_raw = market_fields["high"]
    low_raw = market_fields["low"]
    maximum_outcome_date_idx = int(
        pack.date_to_idx[str(study["source"]["maximum_outcome_date"])]
    )
    adp = _adaptive_line(
        pack.exit_close_raw,
        factor,
        symbol_indices=candidates["symbol_idx"].unique(),
        maximum_date_idx=maximum_outcome_date_idx,
        study=study,
    )
    candidates = _attach_technical(
        candidates,
        pack=pack,
        factor=factor,
        high_raw=high_raw,
        low_raw=low_raw,
        adp=adp,
    )
    candidates, decrease_audit = _attach_paths_and_exits(
        candidates,
        timeline,
        pack=pack,
        factor=factor,
        high_raw=high_raw,
        low_raw=low_raw,
        study=study,
    )
    rule_summaries, daily_rules, entry_gate = _rule_summaries(candidates, study)
    failure_summaries = _failure_summaries(candidates, study)
    take_profit_summaries, daily_take_profit = _take_profit_summaries(candidates, study)
    panel = base.load_panel(
        input_manifest_path=sources["model_input_manifest"],
        label_manifest_path=sources["label_manifest"],
    )
    candidates, model_features, model_feature_names = _attach_compact_features(
        candidates, panel=panel, study=study
    )
    model_scores, model_summaries, feature_importance = _model_study(
        candidates,
        model_features,
        model_feature_names,
        study=study,
    )
    cases = _representative_cases(candidates, study)
    files: dict[str, dict[str, Any]] = {}
    for name, frame in (
        ("candidates", candidates),
        ("daily_rule_returns", daily_rules),
        ("daily_take_profit_returns", daily_take_profit),
        ("model_scores", model_scores),
        ("model_feature_importance", feature_importance),
        ("representative_cases", cases),
    ):
        path = root / f"{name}.parquet"
        _write_parquet(path, frame)
        files[name] = base._file_record(path)
    account_authorized = bool(entry_gate["passed"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_adaptive_retrospective_margin_top10_study",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "study_sha256": base._sha256_file(frozen_path),
        "implementation_sha256": base._sha256_file(Path(__file__)),
        "selection_audit": selection_audit,
        "factor_audit": factor_audit,
        "market_field_audit": market_audit,
        "decrease_exit_audit": decrease_audit,
        "signal_counts": {
            name: int(mask.sum()) for name, mask in _signal_masks(candidates).items()
        },
        "rule_summaries": rule_summaries,
        "entry_gate": entry_gate,
        "failure_summaries": failure_summaries,
        "adaptive_take_profit_summaries": take_profit_summaries,
        "model_summaries": model_summaries,
        "decision": {
            "user_rule_entry_gate_passed": bool(entry_gate["passed"]),
            "finite_account_replay_authorized": account_authorized,
            "all_results_are_adaptive_retrospective": True,
            "production_claim_allowed": False,
            "forward_confirmation_required": True,
        },
        "forbidden_2026_read_count": 0,
        "files": files,
    }
    base._write_json(summary_path, summary)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary = _read_json(base._resolve_path(path))
    checks = {
        "schema": summary.get("schema") == SUMMARY_SCHEMA,
        "study": summary.get("study_id") == STUDY_ID,
        "completed": summary.get("status")
        == "completed_adaptive_retrospective_margin_top10_study",
        "no_2026": int(summary.get("forbidden_2026_read_count", -1)) == 0,
        "not_production": summary.get("decision", {}).get("production_claim_allowed")
        is False,
        "files": all(
            base._record_valid(record, verify_hash=True)
            for record in summary.get("files", {}).values()
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"summary_validation_failed:{checks}")
    return {"status": "ok", "checks": checks}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_study(
        study_path=args.study,
        output_root=args.output_root,
        force=args.force,
    )
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
