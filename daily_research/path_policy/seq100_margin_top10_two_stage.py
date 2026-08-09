"""Full-market next-day models followed by financing-Top10 reranking.

The study is deliberately retrospective.  It consumes the already audited
2012-2025 feature matrix and the corrected same-source-day financing Top10
panel, then emits expanding-window predictions for five contiguous 2020-2025
validation blocks.  No 2026 outcome is read.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score

from daily_research.path_policy import seq100_margin_top10_adaptive_entry as margin
from daily_research.path_policy import seq100_path_label_learnability as learnability

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_margin_top10_two_stage_v1"
SUMMARY_SCHEMA = "seq100_margin_top10_two_stage_summary/1"
PREPARED_SCHEMA = "seq100_margin_top10_two_stage_prepared/1"
TASK_SCHEMA = "seq100_margin_top10_two_stage_base_task/1"
FORBIDDEN_YEAR = 2026
MAXIMUM_OUTCOME_DATE = "2025-12-31"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_margin_top10_two_stage_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies" / STUDY_ID
)

TARGETS = ("next_close_up", "legal_gross_return")
VARIANTS = ("transparent_66", "compact_557")

CONTEXT_FEATURES = (
    "return_5d",
    "return_20d",
    "trend_slope_5d",
    "trend_slope_20d",
    "trend_r2_10d",
    "efficiency_ratio_5d",
    "efficiency_ratio_20d",
    "volatility_5d",
    "volatility_20d",
    "downside_volatility_10d",
    "intraday_range_1d",
    "close_location_1d",
    "open_gap_1d",
    "log_amount_1d",
    "log_circulating_market_value",
    "relative_turnover_20d",
    "market_all__ret1_mean",
    "market_all__breadth_ret1_positive",
    "market_all__up_limit_rate",
    "market_all__down_limit_rate",
    "industry_relative_ret5",
    "minute_vwap_close_deviation",
    "minute_last_30m_return",
    "minute_close_location",
    "minute_realized_volatility",
)

FINANCING_FEATURES = (
    "margin_rank",
    "margin_rank_fraction",
    "margin_increase_streak",
    "margin_increase_streak_log",
    "margin_increment_signed_log",
    "margin_increment_to_balance",
    "reported_net_financing_flow_to_balance",
    "margin_balance_log",
    "financing_buy_log",
    "financing_repayment_log",
    "financing_buy_to_balance",
    "financing_repayment_to_balance",
    "margin_increment_log_to_float_cap_log",
    "margin_balance_log_to_float_cap_log",
    "financing_buy_log_to_amount_log",
    "financing_repayment_log_to_amount_log",
    "prior_margin_increment_signed_log",
    "margin_increment_acceleration_scaled",
    "margin_increment_vs_prior",
    "margin_increment_z20",
    "margin_balance_change_5",
    "margin_balance_change_20",
    "financing_buy_z20",
    "financing_repayment_z20",
    "margin_source_gap_days",
    "short_balance_to_margin_balance",
    "total_margin_short_balance_log",
    "current_return_1d",
    "current_candle_return",
    "close_to_adp",
    "low_to_adp",
    "adp_slope_1d",
    "adp_slope_3d",
    "current_cross",
    "current_cross_rising",
    "down_close",
    "margin_observed",
    "margin_not_applicable",
    "margin_source_unavailable",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, default=_json_default
        ).encode("utf-8")
    ).hexdigest()


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size": int(path.stat().st_size),
        **extra,
    }


def _record_valid(record: Mapping[str, Any], *, verify_hash: bool = True) -> bool:
    try:
        path = Path(str(record["path"]))
        if not path.is_file() or int(path.stat().st_size) != int(record["size"]):
            return False
        return not verify_hash or _sha256(path) == str(record["sha256"])
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps({"event": event, "at": _now(), **payload}, ensure_ascii=False),
        flush=True,
    )


def load_study(
    path: str | Path = DEFAULT_STUDY_PATH,
) -> tuple[dict[str, Any], Path]:
    study_path = _resolve(path)
    study = _read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    epistemic = dict(study.get("epistemic_contract", {}) or {})
    if (
        not bool(
            epistemic.get("all_2012_2025_outcomes_are_consumed_development_evidence")
        )
        or not bool(
            epistemic.get("each_validation_prediction_uses_only_earlier_signal_dates")
        )
        or not bool(epistemic.get("no_2026_outcome_may_be_read"))
        or bool(epistemic.get("production_or_stable_profit_claim_allowed", True))
    ):
        raise ValueError("epistemic_contract_mismatch")
    source = dict(study["source"])
    if (
        int(source["forbidden_year"]) != FORBIDDEN_YEAR
        or str(source["maximum_outcome_date"]) != MAXIMUM_OUTCOME_DATE
    ):
        raise ValueError("outcome_boundary_mismatch")
    validation = dict(study["validation"])
    if (
        int(validation["forward_fold_count"]) != 5
        or int(validation["common_purge_trading_days"]) != 2
        or not bool(validation["same_date_stocks_never_split"])
        or not bool(validation["future_fold_training_forbidden"])
    ):
        raise ValueError("validation_contract_mismatch")
    if tuple(study["base_model"]["tasks"]) != TARGETS:
        raise ValueError("target_contract_mismatch")
    if tuple(study["base_model"]["variants"]) != VARIANTS:
        raise ValueError("variant_contract_mismatch")
    if int(study["features"]["transparent_variant"]["feature_count"]) != 66:
        raise ValueError("transparent_feature_count_mismatch")
    if int(study["features"]["compact_variant"]["feature_count"]) != 557:
        raise ValueError("compact_feature_count_mismatch")
    return study, study_path


def _source_paths(study: Mapping[str, Any]) -> dict[str, Path]:
    source = dict(study["source"])
    keys = (
        "model_input_manifest",
        "transparent_coordinate_manifest",
        "margin_candidate_panel",
        "margin_candidate_summary",
        "margin_candidate_study",
        "pack_manifest",
    )
    result: dict[str, Path] = {}
    for key in keys:
        path = _resolve(str(source[key]))
        if not path.is_file() or _sha256(path) != str(source[f"{key}_sha256"]):
            raise ValueError(f"source_hash_mismatch:{key}")
        result[key] = path
    active_path = _resolve(str(source["qdp_active_manifest"]))
    active = _read_json(active_path)
    margin_id = str(source["margin_detail_dataset_id"])
    if str(active.get("datasets", {}).get("margin_detail")) != margin_id:
        raise ValueError("active_margin_dataset_drift")
    result["qdp_active_manifest"] = active_path
    result["margin_detail_manifest"] = (
        active_path.parent.parent
        / "datasets"
        / "margin_detail"
        / margin_id
        / "dataset.json"
    ).resolve()
    if _read_json(result["margin_detail_manifest"]).get("dataset_id") != margin_id:
        raise ValueError("margin_detail_manifest_mismatch")
    return result


def _study_fingerprint(
    *, study_path: Path, paths: Mapping[str, Path], input_manifest: Mapping[str, Any]
) -> str:
    return _stable_hash(
        {
            "study_sha256": _sha256(study_path),
            "model_input_fingerprint": input_manifest["input_fingerprint"],
            "source_hashes": {
                name: _sha256(path)
                for name, path in paths.items()
                if name != "qdp_active_manifest"
            },
        }
    )


def _coordinate_names(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    names = tuple(
        str(value) for value in manifest["coordinate_audit"]["coordinate_names"]
    )
    if len(names) != 66 or len(set(names)) != 66:
        raise ValueError("coordinate_names_invalid")
    return names


def _build_coordinate_cache(
    *, manifest: Mapping[str, Any], row_count: int, output_root: Path
) -> dict[str, Any]:
    names = _coordinate_names(manifest)
    target = output_root / "prepared/transparent_66.float32.dat"
    partial = Path(str(target) + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)
    matrix = np.memmap(partial, dtype=np.float32, mode="w+", shape=(row_count, 66))
    seen = np.zeros(row_count, dtype=bool)
    loaded_rows = 0
    for record in manifest["coordinates"]:
        path = Path(str(record["path"])).resolve()
        if not path.is_file() or int(path.stat().st_size) != int(record["size"]):
            raise ValueError(f"coordinate_partition_invalid:{path}")
        frame = pd.read_parquet(path, columns=["input_row_idx", *names])
        rows = frame["input_row_idx"].to_numpy(dtype=np.int64)
        if (
            len(frame) != int(record["rows"])
            or bool((rows < 0).any())
            or bool((rows >= row_count).any())
            or bool(seen[rows].any())
        ):
            raise ValueError(f"coordinate_row_contract_failed:{path}")
        matrix[rows] = frame.loc[:, names].to_numpy(dtype=np.float32, copy=False)
        seen[rows] = True
        loaded_rows += len(rows)
    if loaded_rows != row_count or not bool(seen.all()):
        raise ValueError(f"coordinate_coverage_failed:{loaded_rows}:{int(seen.sum())}")
    matrix.flush()
    del matrix
    os.replace(partial, target)
    return _file_record(
        target,
        dtype="float32",
        shape=[row_count, 66],
        feature_names=list(names),
        loaded_rows=loaded_rows,
    )


def _pack_path_contract(
    pack_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    path_contract = dict(
        dict(pack_manifest.get("label_arrays", {}) or {}).get("future_ohlcva_path", {})
        or {}
    )
    fields = [str(value) for value in path_contract.get("fields", ())]
    semantics = str(
        dict(pack_manifest.get("label_semantics", {}) or {}).get("future_ohlc_path", "")
    )
    if (
        fields != ["open", "high", "low", "close", "volume", "amount"]
        or str(path_contract.get("anchor")) != "signal_day_close"
        or "back_adjust" not in semantics
    ):
        raise ValueError("future_path_contract_mismatch")
    return path_contract, fields


def _same_date_winsor(
    values: np.ndarray, valid: np.ndarray, date_idx: np.ndarray
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    dates = np.asarray(date_idx, dtype=np.int32)
    result = np.full(len(source), np.nan, dtype=np.float32)
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for left, right in pairwise(boundaries):
        local_valid = mask[left:right]
        local = source[left:right][local_valid]
        if len(local) < 20:
            continue
        lower, upper = np.quantile(local, [0.01, 0.99], method="nearest")
        positions = left + np.flatnonzero(local_valid)
        result[positions] = np.clip(local, lower, upper).astype(np.float32)
    return result


def _build_label_cache(
    *,
    row_index: pd.DataFrame,
    pack_manifest: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    row_count = len(row_index)
    path_contract, fields = _pack_path_contract(pack_manifest)
    close_column = fields.index("close")
    open_column = fields.index("open")
    values_path = output_root / "prepared/base_targets.float32.dat"
    valid_path = output_root / "prepared/base_target_valid.uint8.dat"
    winsor_path = output_root / "prepared/legal_gross_winsor.float32.dat"
    values_partial = Path(str(values_path) + ".partial")
    valid_partial = Path(str(valid_path) + ".partial")
    winsor_partial = Path(str(winsor_path) + ".partial")
    for path in (values_partial, valid_partial, winsor_partial):
        path.unlink(missing_ok=True)
    values = np.memmap(
        values_partial, dtype=np.float32, mode="w+", shape=(row_count, 2)
    )
    valid_store = np.memmap(
        valid_partial, dtype=np.uint8, mode="w+", shape=(row_count, 2)
    )
    values[:] = np.nan
    valid_store[:] = 0

    dates = row_index["date_idx"].to_numpy(dtype=np.int32)
    symbols = row_index["symbol_idx"].to_numpy(dtype=np.int32)
    pack_symbols = np.asarray(pack_manifest["symbol_values"], dtype=str)
    if (
        bool((symbols < 0).any())
        or int(symbols.max()) >= len(pack_symbols)
        or not bool(
            (
                pack_symbols[symbols]
                == row_index["symbol"].astype(str).to_numpy(dtype=str)
            ).all()
        )
    ):
        raise ValueError("row_index_pack_symbol_alignment_failed")
    date_values = np.asarray(pack_manifest["date_values"], dtype=str)
    cutoff_matches = np.flatnonzero(date_values == MAXIMUM_OUTCOME_DATE)
    if len(cutoff_matches) != 1:
        raise ValueError("maximum_outcome_date_missing_from_pack")
    cutoff = int(cutoff_matches[0])
    extracted_rows = 0
    for raw_record in path_contract.get("shards", ()):
        record = dict(raw_record)
        start = int(record["date_start_idx"])
        end = int(record["date_end_idx"])
        left = int(np.searchsorted(dates, start, side="left"))
        right = int(np.searchsorted(dates, end, side="right"))
        if left >= right:
            continue
        rows = np.arange(left, right, dtype=np.int64)
        path = Path(str(record["path"])).resolve()
        shape = tuple(int(value) for value in record["shape"])
        if not path.is_file() or int(path.stat().st_size) != int(
            np.prod(shape) * np.dtype(np.float32).itemsize
        ):
            raise ValueError(f"future_path_shard_invalid:{path}")
        source = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        local_date = dates[rows] - start
        local_symbol = symbols[rows]
        next_close = np.asarray(
            source[local_date, local_symbol, 0, close_column], dtype=np.float32
        )
        entry_open = np.asarray(
            source[local_date, local_symbol, 0, open_column], dtype=np.float32
        )
        legal_close = np.asarray(
            source[local_date, local_symbol, 1, close_column], dtype=np.float32
        )
        next_valid = (dates[rows] + 1 <= cutoff) & np.isfinite(next_close)
        legal_valid = (
            (dates[rows] + 2 <= cutoff)
            & np.isfinite(entry_open)
            & np.isfinite(legal_close)
            & (entry_open > -1.0)
            & (legal_close > -1.0)
        )
        legal_return = np.full(len(rows), np.nan, dtype=np.float32)
        legal_return[legal_valid] = (
            (1.0 + legal_close[legal_valid].astype(np.float64))
            / (1.0 + entry_open[legal_valid].astype(np.float64))
            - 1.0
        ).astype(np.float32)
        values[rows, 0] = np.where(next_valid, next_close, np.nan)
        values[rows, 1] = legal_return
        valid_store[rows, 0] = next_valid.astype(np.uint8)
        valid_store[rows, 1] = legal_valid.astype(np.uint8)
        extracted_rows += len(rows)
        del source
    if extracted_rows != row_count:
        raise ValueError(f"label_row_coverage_failed:{extracted_rows}:{row_count}")
    values.flush()
    valid_store.flush()
    legal_values = np.asarray(values[:, 1], dtype=np.float32)
    legal_valid_all = np.asarray(valid_store[:, 1], dtype=bool)
    winsor_values = _same_date_winsor(legal_values, legal_valid_all, dates)
    winsor = np.memmap(winsor_partial, dtype=np.float32, mode="w+", shape=(row_count,))
    winsor[:] = winsor_values
    winsor.flush()
    valid_counts = np.asarray(valid_store, dtype=np.uint8).sum(axis=0)
    del values, valid_store, winsor, winsor_values, legal_values
    os.replace(values_partial, values_path)
    os.replace(valid_partial, valid_path)
    os.replace(winsor_partial, winsor_path)
    return {
        "values": _file_record(
            values_path,
            dtype="float32",
            shape=[row_count, 2],
            columns=["next_close_return", "legal_gross_return"],
        ),
        "valid": _file_record(
            valid_path,
            dtype="uint8",
            shape=[row_count, 2],
            columns=["next_close_up", "legal_gross_return"],
        ),
        "legal_winsor": _file_record(
            winsor_path,
            dtype="float32",
            shape=[row_count],
            transform="same_date_q01_q99_nearest",
        ),
        "valid_counts": {
            "next_close_up": int(valid_counts[0]),
            "legal_gross_return": int(valid_counts[1]),
        },
        "maximum_outcome_date_idx": cutoff,
        "forbidden_2026_read_count": 0,
    }


def build_forward_folds(
    *,
    date_idx: np.ndarray,
    trade_date: np.ndarray,
    validation_start_date: str,
    validation_end_date: str,
    fold_count: int,
    purge_days: int,
) -> list[dict[str, Any]]:
    dates = np.asarray(date_idx, dtype=np.int32)
    labels = np.asarray(trade_date, dtype=str)
    if dates.ndim != 1 or labels.shape != dates.shape:
        raise ValueError("fold_arrays_invalid")
    if bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("fold_rows_not_date_ordered")
    validation_dates = np.unique(
        dates[
            (labels >= str(validation_start_date))
            & (labels <= str(validation_end_date))
        ]
    )
    if len(validation_dates) < int(fold_count):
        raise ValueError("insufficient_validation_dates")
    blocks = np.array_split(validation_dates, int(fold_count))
    result: list[dict[str, Any]] = []
    for fold_number, block in enumerate(blocks, start=1):
        start = int(block[0])
        end = int(block[-1])
        train_maximum = start - int(purge_days) - 1
        train_rows = np.flatnonzero(dates <= train_maximum)
        validation_rows = np.flatnonzero((dates >= start) & (dates <= end))
        if not len(train_rows) or not len(validation_rows):
            raise ValueError(f"fold_empty:{fold_number}")
        if int(dates[train_rows[-1]]) + int(purge_days) >= start:
            raise ValueError(f"fold_purge_failed:{fold_number}")
        if bool(np.intersect1d(dates[train_rows], block).size):
            raise ValueError(f"fold_date_overlap:{fold_number}")
        result.append(
            {
                "fold": fold_number,
                "validation_start_date_idx": start,
                "validation_end_date_idx": end,
                "validation_start_date": str(labels[validation_rows[0]]),
                "validation_end_date": str(labels[validation_rows[-1]]),
                "validation_date_count": len(block),
                "training_maximum_date_idx": train_maximum,
                "training_maximum_date": str(labels[train_rows[-1]]),
                "training_row_count": len(train_rows),
                "validation_row_count": len(validation_rows),
                "purge_days": int(purge_days),
            }
        )
    return result


def _qdp_margin_paths(manifest_path: Path) -> list[Path]:
    manifest = _read_json(manifest_path)
    qdp_root = manifest_path.parents[3]
    paths: list[Path] = []
    for record in manifest["shards"]:
        raw = Path(str(record["path"]))
        path = raw if raw.is_absolute() else qdp_root / raw
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path.resolve())
    return paths


def _margin_history_features(
    *,
    candidates: pd.DataFrame,
    margin_manifest_path: Path,
    pack_manifest: Mapping[str, Any],
) -> pd.DataFrame:
    date_map = pd.DataFrame(
        {
            "trade_date": np.asarray(pack_manifest["date_values"], dtype=str),
            "source_date_idx": np.arange(
                len(pack_manifest["date_values"]), dtype=np.int32
            ),
        }
    )
    keys = candidates[["candidate_id", "symbol", "trade_date"]].copy()
    connection = duckdb.connect(database=":memory:")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET threads=8")
    connection.execute("SET memory_limit='3GB'")
    connection.register("candidate_keys", keys)
    connection.register("date_map", date_map)
    connection.from_parquet(
        [str(path) for path in _qdp_margin_paths(margin_manifest_path)]
    ).create_view("margin_source")
    query = f"""
      WITH raw AS (
        SELECT m.symbol, m.trade_date, d.source_date_idx,
               CAST(m.rzye AS DOUBLE) AS margin_balance,
               CAST(m.rzmre AS DOUBLE) AS financing_buy,
               CAST(m.rzche AS DOUBLE) AS financing_repayment,
               CAST(m.rqye AS DOUBLE) AS short_balance,
               CAST(m.rzrqye AS DOUBLE) AS total_balance,
               lag(CAST(m.rzye AS DOUBLE), 1) OVER w AS balance_l1,
               lag(CAST(m.rzye AS DOUBLE), 2) OVER w AS balance_l2,
               lag(CAST(m.rzye AS DOUBLE), 5) OVER w AS balance_l5,
               lag(CAST(m.rzye AS DOUBLE), 20) OVER w AS balance_l20,
               lag(d.source_date_idx, 1) OVER w AS date_idx_l1
        FROM margin_source m
        JOIN date_map d ON d.trade_date = m.trade_date
        WHERE m.trade_date <= '{MAXIMUM_OUTCOME_DATE}'
        WINDOW w AS (PARTITION BY m.symbol ORDER BY d.source_date_idx)
      ), increments AS (
        SELECT *, margin_balance - balance_l1 AS margin_increment,
               balance_l1 - balance_l2 AS prior_margin_increment
        FROM raw
      ), history AS (
        SELECT *,
               avg(margin_increment) OVER w20 AS increment_mean_20,
               stddev_samp(margin_increment) OVER w20 AS increment_std_20,
               avg(financing_buy) OVER w20 AS buy_mean_20,
               stddev_samp(financing_buy) OVER w20 AS buy_std_20,
               avg(financing_repayment) OVER w20 AS repayment_mean_20,
               stddev_samp(financing_repayment) OVER w20 AS repayment_std_20
        FROM increments
        WINDOW w20 AS (
          PARTITION BY symbol ORDER BY source_date_idx
          ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        )
      )
      SELECT c.candidate_id,
             h.prior_margin_increment,
             h.margin_increment - h.prior_margin_increment AS margin_increment_acceleration,
             h.increment_mean_20, h.increment_std_20,
             h.balance_l5, h.balance_l20,
             h.buy_mean_20, h.buy_std_20,
             h.repayment_mean_20, h.repayment_std_20,
             h.source_date_idx - h.date_idx_l1 AS margin_source_gap_days,
             h.short_balance, h.total_balance
      FROM candidate_keys c
      JOIN history h ON h.symbol = c.symbol AND h.trade_date = c.trade_date
      ORDER BY c.candidate_id
    """
    result = connection.execute(query).fetch_df()
    connection.close()
    if len(result) != len(candidates) or result["candidate_id"].nunique() != len(
        candidates
    ):
        raise ValueError("margin_history_join_failed")
    return result


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    left = np.asarray(numerator, dtype=np.float64)
    right = np.asarray(denominator, dtype=np.float64)
    result = np.full(
        np.broadcast_shapes(left.shape, right.shape), np.nan, dtype=np.float64
    )
    np.divide(left, right, out=result, where=np.isfinite(right) & (right != 0.0))
    return result


def _signed_log(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    return np.sign(source) * np.log1p(np.abs(source))


def _prepare_candidates(
    *,
    source_path: Path,
    row_index: pd.DataFrame,
    compact: np.memmap,
    compact_names: Sequence[str],
    margin_manifest_path: Path,
    pack_manifest: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = pd.read_parquet(source_path)
    required = {
        "candidate_id",
        "date_idx",
        "trade_date",
        "symbol",
        "symbol_idx",
        "margin_source_date_idx",
        "margin_available_date_idx",
        "margin_increase_streak",
        "margin_rank",
        "margin_balance",
        "margin_increment",
        "financing_buy",
        "financing_repayment",
        "signal_close_return_d1",
        "one_day_net_return",
        "one_day_gross_adjusted_return",
        "one_day_entry_filled",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidate_columns_missing:{missing}")
    if (
        len(candidates) != 33877
        or int(candidates["margin_increase_streak"].min()) < 2
        or int(candidates["margin_rank"].max()) > 10
        or bool(candidates["trade_date"].astype(str).str.startswith("2026-").any())
        or not bool(
            (
                candidates["margin_source_date_idx"].to_numpy(dtype=np.int64)
                == candidates["date_idx"].to_numpy(dtype=np.int64)
            ).all()
        )
        or not bool(
            (
                candidates["margin_available_date_idx"].to_numpy(dtype=np.int64)
                == candidates["date_idx"].to_numpy(dtype=np.int64) + 1
            ).all()
        )
    ):
        raise ValueError("candidate_population_contract_failed")

    identity = row_index[["candidate_id", "trade_date", "symbol"]].copy()
    identity["model_row_position"] = np.arange(len(identity), dtype=np.int64)
    candidates = candidates.merge(
        identity,
        on="candidate_id",
        how="left",
        suffixes=("", "__model"),
        validate="one_to_one",
    )
    if (
        candidates["model_row_position"].isna().any()
        or not bool(
            candidates["trade_date"]
            .astype(str)
            .eq(candidates["trade_date__model"].astype(str))
            .all()
        )
        or not bool(
            candidates["symbol"]
            .astype(str)
            .eq(candidates["symbol__model"].astype(str))
            .all()
        )
    ):
        raise ValueError("candidate_model_row_alignment_failed")
    candidates = candidates.drop(columns=["trade_date__model", "symbol__model"])
    candidates["model_row_position"] = candidates["model_row_position"].astype(np.int64)
    history = _margin_history_features(
        candidates=candidates,
        margin_manifest_path=margin_manifest_path,
        pack_manifest=pack_manifest,
    )
    candidates = candidates.merge(
        history, on="candidate_id", how="left", validate="one_to_one"
    )

    name_to_position = {str(name): index for index, name in enumerate(compact_names)}
    for name in CONTEXT_FEATURES:
        if name not in name_to_position:
            raise ValueError(f"context_feature_missing:{name}")
    rows = candidates["model_row_position"].to_numpy(dtype=np.int64)
    context_positions = np.asarray(
        [name_to_position[name] for name in CONTEXT_FEATURES], dtype=np.int32
    )
    context = np.asarray(compact[np.ix_(rows, context_positions)], dtype=np.float32)
    for position, name in enumerate(CONTEXT_FEATURES):
        candidates[f"context__{name}"] = context[:, position]

    candidate_compact_path = output_root / "prepared/candidate_compact_557.float32.dat"
    candidate_compact_partial = Path(str(candidate_compact_path) + ".partial")
    candidate_compact_partial.unlink(missing_ok=True)
    candidate_compact = np.memmap(
        candidate_compact_partial,
        dtype=np.float32,
        mode="w+",
        shape=(len(candidates), len(compact_names)),
    )
    batch = 8192
    for left in range(0, len(candidates), batch):
        right = min(left + batch, len(candidates))
        candidate_compact[left:right] = np.asarray(
            compact[rows[left:right]], dtype=np.float32
        )
    candidate_compact.flush()
    del candidate_compact
    os.replace(candidate_compact_partial, candidate_compact_path)

    balance = candidates["margin_balance"].to_numpy(dtype=np.float64)
    increment = candidates["margin_increment"].to_numpy(dtype=np.float64)
    buy = candidates["financing_buy"].to_numpy(dtype=np.float64)
    repayment = candidates["financing_repayment"].to_numpy(dtype=np.float64)
    prior_increment = candidates["prior_margin_increment"].to_numpy(dtype=np.float64)
    increment_std = candidates["increment_std_20"].to_numpy(dtype=np.float64)
    increment_mean = candidates["increment_mean_20"].to_numpy(dtype=np.float64)
    buy_std = candidates["buy_std_20"].to_numpy(dtype=np.float64)
    repay_std = candidates["repayment_std_20"].to_numpy(dtype=np.float64)
    log_float_cap = candidates["context__log_circulating_market_value"].to_numpy(
        dtype=np.float64
    )
    log_amount = candidates["context__log_amount_1d"].to_numpy(dtype=np.float64)
    candidates["margin_rank_fraction"] = (
        candidates["margin_rank"].to_numpy(dtype=np.float64) - 1.0
    ) / 9.0
    candidates["margin_increase_streak_log"] = np.log1p(
        candidates["margin_increase_streak"].to_numpy(dtype=np.float64)
    )
    candidates["margin_balance_log"] = np.log1p(np.maximum(balance, 0.0))
    candidates["financing_buy_log"] = np.log1p(np.maximum(buy, 0.0))
    candidates["financing_repayment_log"] = np.log1p(np.maximum(repayment, 0.0))
    candidates["financing_buy_to_balance"] = _safe_divide(buy, balance)
    candidates["financing_repayment_to_balance"] = _safe_divide(repayment, balance)
    candidates["margin_increment_log_to_float_cap_log"] = (
        np.log1p(np.maximum(increment, 0.0)) - log_float_cap
    )
    candidates["margin_balance_log_to_float_cap_log"] = (
        np.log1p(np.maximum(balance, 0.0)) - log_float_cap
    )
    candidates["financing_buy_log_to_amount_log"] = (
        np.log1p(np.maximum(buy, 0.0)) - log_amount
    )
    candidates["financing_repayment_log_to_amount_log"] = (
        np.log1p(np.maximum(repayment, 0.0)) - log_amount
    )
    candidates["prior_margin_increment_signed_log"] = _signed_log(prior_increment)
    candidates["margin_increment_acceleration_scaled"] = _safe_divide(
        candidates["margin_increment_acceleration"].to_numpy(dtype=np.float64),
        np.abs(increment) + np.abs(prior_increment) + 1.0,
    )
    candidates["margin_increment_vs_prior"] = _safe_divide(
        increment, np.abs(prior_increment) + 1.0
    )
    candidates["margin_increment_z20"] = _safe_divide(
        increment - increment_mean, increment_std
    )
    candidates["margin_balance_change_5"] = (
        _safe_divide(balance, candidates["balance_l5"].to_numpy(dtype=np.float64)) - 1.0
    )
    candidates["margin_balance_change_20"] = (
        _safe_divide(balance, candidates["balance_l20"].to_numpy(dtype=np.float64))
        - 1.0
    )
    candidates["financing_buy_z20"] = _safe_divide(
        buy - candidates["buy_mean_20"].to_numpy(dtype=np.float64), buy_std
    )
    candidates["financing_repayment_z20"] = _safe_divide(
        repayment - candidates["repayment_mean_20"].to_numpy(dtype=np.float64),
        repay_std,
    )
    candidates["short_balance_to_margin_balance"] = _safe_divide(
        candidates["short_balance"].to_numpy(dtype=np.float64), balance
    )
    candidates["total_margin_short_balance_log"] = np.log1p(
        np.maximum(candidates["total_balance"].to_numpy(dtype=np.float64), 0.0)
    )
    candidates["margin_observed"] = np.float32(1.0)
    candidates["margin_not_applicable"] = np.float32(0.0)
    candidates["margin_source_unavailable"] = np.float32(0.0)
    for name in FINANCING_FEATURES:
        if name not in candidates.columns:
            raise ValueError(f"financing_feature_missing:{name}")

    keep = [
        column
        for column in candidates.columns
        if column
        not in {
            "increment_mean_20",
            "increment_std_20",
            "buy_mean_20",
            "buy_std_20",
            "repayment_mean_20",
            "repayment_std_20",
            "balance_l5",
            "balance_l20",
        }
    ]
    candidates = (
        candidates.loc[:, keep]
        .sort_values(["date_idx", "margin_rank", "symbol"], kind="stable")
        .reset_index(drop=True)
    )
    candidate_path = output_root / "prepared/candidates.parquet"
    _write_parquet(candidate_path, candidates)
    audit = {
        "candidate_count": len(candidates),
        "signal_date_count": int(candidates["date_idx"].nunique()),
        "first_signal_date": str(candidates["trade_date"].min()),
        "last_signal_date": str(candidates["trade_date"].max()),
        "minimum_streak": int(candidates["margin_increase_streak"].min()),
        "maximum_rank": int(candidates["margin_rank"].max()),
        "model_row_match_count": int(candidates["model_row_position"].notna().sum()),
        "margin_observed_count": len(candidates),
        "margin_not_applicable_count": 0,
        "margin_source_unavailable_count": 0,
        "forbidden_2026_count": 0,
        "source_date_equals_signal_date": True,
        "available_date_equals_next_trading_date": True,
    }
    return (
        _file_record(candidate_path, row_count=len(candidates)),
        {
            "candidate_compact": _file_record(
                candidate_compact_path,
                dtype="float32",
                shape=[len(candidates), len(compact_names)],
                feature_names=list(compact_names),
            ),
            "financing_feature_names": list(FINANCING_FEATURES),
            "context_feature_names": [f"context__{name}" for name in CONTEXT_FEATURES],
            "audit": audit,
        },
    )


def _prepared_valid(path: Path, *, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = _read_json(path)
        if (
            manifest.get("schema") != PREPARED_SCHEMA
            or manifest.get("status") != "completed"
            or manifest.get("study_fingerprint") != fingerprint
        ):
            return False
        records = [
            manifest["coordinate_cache"],
            manifest["labels"]["values"],
            manifest["labels"]["valid"],
            manifest["labels"]["legal_winsor"],
            manifest["candidate_panel"],
            manifest["candidate_features"]["candidate_compact"],
        ]
        return all(_record_valid(record, verify_hash=False) for record in records)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def prepare(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study, resolved_study_path = load_study(study_path)
    root = _resolve(output_root)
    paths = _source_paths(study)
    input_manifest = _read_json(paths["model_input_manifest"])
    if (
        input_manifest.get("status") != "completed"
        or int(input_manifest.get("row_count", -1)) != 4_191_476
        or str(input_manifest.get("input_fingerprint"))
        != str(study["source"]["model_input_fingerprint"])
        or int(input_manifest["storage"]["compact"]["shape"][1]) != 557
    ):
        raise ValueError("model_input_contract_mismatch")
    fingerprint = _study_fingerprint(
        study_path=resolved_study_path, paths=paths, input_manifest=input_manifest
    )
    manifest_path = root / "prepared/manifest.json"
    if not force and _prepared_valid(manifest_path, fingerprint=fingerprint):
        return _read_json(manifest_path)
    root.mkdir(parents=True, exist_ok=True)
    _emit("prepare_started", row_count=input_manifest["row_count"])
    row_index = pd.read_parquet(Path(input_manifest["row_index"]["path"]))
    row_index["trade_date"] = row_index["trade_date"].astype(str)
    if (
        len(row_index) != int(input_manifest["row_count"])
        or bool(row_index["trade_date"].str.startswith("2026-").any())
        or not bool(row_index["candidate_id"].is_unique)
    ):
        raise ValueError("row_index_contract_failed")
    coordinate_manifest = _read_json(paths["transparent_coordinate_manifest"])
    inherited = coordinate_manifest["source_contract"]["inherited_source_contract"]
    if (
        coordinate_manifest.get("status") != "completed"
        or int(coordinate_manifest["coordinate_audit"]["rows"]) != len(row_index)
        or str(inherited["model_input_manifest"]["sha256"])
        != str(study["source"]["model_input_manifest_sha256"])
    ):
        raise ValueError("coordinate_manifest_contract_failed")
    coordinate_cache = _build_coordinate_cache(
        manifest=coordinate_manifest, row_count=len(row_index), output_root=root
    )
    pack_manifest = _read_json(paths["pack_manifest"])
    labels = _build_label_cache(
        row_index=row_index, pack_manifest=pack_manifest, output_root=root
    )
    compact_record = input_manifest["storage"]["compact"]
    compact = np.memmap(
        Path(compact_record["path"]),
        dtype=np.float32,
        mode="r",
        shape=tuple(int(value) for value in compact_record["shape"]),
    )
    compact_names = tuple(
        str(value) for value in input_manifest["feature_groups"]["compact_core"]
    )
    candidate_panel, candidate_features = _prepare_candidates(
        source_path=paths["margin_candidate_panel"],
        row_index=row_index,
        compact=compact,
        compact_names=compact_names,
        margin_manifest_path=paths["margin_detail_manifest"],
        pack_manifest=pack_manifest,
        output_root=root,
    )
    del compact
    folds = build_forward_folds(
        date_idx=row_index["date_idx"].to_numpy(dtype=np.int32),
        trade_date=row_index["trade_date"].to_numpy(dtype=str),
        validation_start_date=str(study["validation"]["validation_start_date"]),
        validation_end_date=str(study["validation"]["validation_end_date"]),
        fold_count=int(study["validation"]["forward_fold_count"]),
        purge_days=int(study["validation"]["common_purge_trading_days"]),
    )
    manifest = {
        "schema": PREPARED_SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "study_id": STUDY_ID,
        "study_fingerprint": fingerprint,
        "row_count": len(row_index),
        "model_input_fingerprint": input_manifest["input_fingerprint"],
        "source": {
            name: _file_record(path)
            for name, path in paths.items()
            if name not in {"qdp_active_manifest", "margin_detail_manifest"}
        },
        "row_index": dict(input_manifest["row_index"]),
        "compact_557": dict(compact_record),
        "compact_feature_names": list(compact_names),
        "coordinate_cache": coordinate_cache,
        "labels": labels,
        "candidate_panel": candidate_panel,
        "candidate_features": candidate_features,
        "folds": folds,
        "forbidden_2026_read_count": 0,
    }
    _write_json(manifest_path, manifest)
    _emit(
        "prepare_completed", candidates=candidate_features["audit"]["candidate_count"]
    )
    return manifest


@dataclass
class PreparedData:
    manifest: dict[str, Any]
    row_index: pd.DataFrame
    compact: np.memmap
    transparent: np.memmap
    target_values: np.memmap
    target_valid: np.memmap
    legal_winsor: np.memmap
    candidates: pd.DataFrame
    candidate_compact: np.memmap


def load_prepared(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[dict[str, Any], Path, PreparedData]:
    study, resolved_study_path = load_study(study_path)
    root = _resolve(output_root)
    manifest_path = root / "prepared/manifest.json"
    manifest = _read_json(manifest_path)
    paths = _source_paths(study)
    input_manifest = _read_json(paths["model_input_manifest"])
    fingerprint = _study_fingerprint(
        study_path=resolved_study_path, paths=paths, input_manifest=input_manifest
    )
    if not _prepared_valid(manifest_path, fingerprint=fingerprint):
        raise ValueError("prepared_manifest_invalid_run_prepare")
    row_index = pd.read_parquet(Path(manifest["row_index"]["path"]))
    row_index["trade_date"] = row_index["trade_date"].astype(str)
    compact_record = manifest["compact_557"]
    coordinate_record = manifest["coordinate_cache"]
    label_records = manifest["labels"]
    candidate_record = manifest["candidate_features"]["candidate_compact"]
    prepared = PreparedData(
        manifest=manifest,
        row_index=row_index,
        compact=np.memmap(
            Path(compact_record["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(int(value) for value in compact_record["shape"]),
        ),
        transparent=np.memmap(
            Path(coordinate_record["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(int(value) for value in coordinate_record["shape"]),
        ),
        target_values=np.memmap(
            Path(label_records["values"]["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(int(value) for value in label_records["values"]["shape"]),
        ),
        target_valid=np.memmap(
            Path(label_records["valid"]["path"]),
            dtype=np.uint8,
            mode="r",
            shape=tuple(int(value) for value in label_records["valid"]["shape"]),
        ),
        legal_winsor=np.memmap(
            Path(label_records["legal_winsor"]["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(int(value) for value in label_records["legal_winsor"]["shape"]),
        ),
        candidates=pd.read_parquet(Path(manifest["candidate_panel"]["path"])),
        candidate_compact=np.memmap(
            Path(candidate_record["path"]),
            dtype=np.float32,
            mode="r",
            shape=tuple(int(value) for value in candidate_record["shape"]),
        ),
    )
    return study, resolved_study_path, prepared


class MatrixSequence(lgb.Sequence):
    def __init__(
        self,
        matrix: np.ndarray,
        rows: np.ndarray,
        *,
        batch_size: int,
    ) -> None:
        self.matrix = matrix
        self.rows = np.asarray(rows, dtype=np.int64)
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _block(self, local: np.ndarray) -> np.ndarray:
        selected = self.rows[np.asarray(local, dtype=np.int64)]
        if len(selected) and bool(np.all(np.diff(selected) == 1)):
            return np.asarray(
                self.matrix[int(selected[0]) : int(selected[-1]) + 1],
                dtype=np.float64,
            )
        # LightGBM's Sequence sampler requires double precision even when the
        # source memmap is float32.  Conversion stays batch-local, so this does
        # not duplicate the full 4.19-million-row matrix in memory.
        return np.asarray(self.matrix[selected], dtype=np.float64)

    def __getitem__(self, index: Any) -> np.ndarray:
        if isinstance(index, slice):
            start = 0 if index.start is None else int(index.start)
            stop = len(self.rows) if index.stop is None else int(index.stop)
            step = 1 if index.step is None else int(index.step)
            return self._block(np.arange(start, stop, step, dtype=np.int64))
        if isinstance(index, (list, tuple, np.ndarray)):
            return self._block(np.asarray(index, dtype=np.int64))
        if isinstance(index, (int, np.integer)):
            return self._block(np.asarray([int(index)], dtype=np.int64))[0]
        raise TypeError(type(index).__name__)


def _feature_source(
    prepared: PreparedData, variant: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    if variant == "transparent_66":
        return prepared.transparent, tuple(
            str(value)
            for value in prepared.manifest["coordinate_cache"]["feature_names"]
        )
    if variant == "compact_557":
        return prepared.compact, tuple(
            str(value) for value in prepared.manifest["compact_feature_names"]
        )
    raise ValueError(f"unknown_variant:{variant}")


def _target_arrays(
    prepared: PreparedData, target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target == "next_close_up":
        raw = np.asarray(prepared.target_values[:, 0], dtype=np.float32)
        valid = np.asarray(prepared.target_valid[:, 0], dtype=bool)
        train = (raw > 0.0).astype(np.int8)
        return train, raw, valid
    if target == "legal_gross_return":
        raw = np.asarray(prepared.target_values[:, 1], dtype=np.float32)
        train = np.asarray(prepared.legal_winsor, dtype=np.float32)
        valid = np.asarray(prepared.target_valid[:, 1], dtype=bool) & np.isfinite(train)
        return train, raw, valid
    raise ValueError(f"unknown_target:{target}")


def _base_parameters(study: Mapping[str, Any], target: str) -> dict[str, Any]:
    config = dict(study["base_model"])
    parameters: dict[str, Any] = {
        "boosting_type": "gbdt",
        "device_type": "cpu",
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]),
        "min_data_in_leaf": int(config["min_data_in_leaf"]),
        "lambda_l2": float(config["lambda_l2"]),
        "max_bin": int(config["max_bin"]),
        "feature_fraction": float(config["feature_fraction"]),
        "bagging_fraction": float(config["bagging_fraction"]),
        "bagging_freq": int(config["bagging_freq"]),
        "num_threads": int(config["num_threads"]),
        "histogram_pool_size": int(config["histogram_pool_size_mb"]),
        "deterministic": True,
        "force_col_wise": True,
        "seed": int(study["validation"]["seed"]),
        "feature_fraction_seed": int(study["validation"]["seed"]),
        "bagging_seed": int(study["validation"]["seed"]),
        "data_random_seed": int(study["validation"]["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    if target == "next_close_up":
        parameters.update({"objective": "binary", "metric": "binary_logloss"})
    else:
        parameters.update({"objective": "regression", "metric": "l2"})
    return parameters


def _task_id(fold: int, variant: str, target: str) -> str:
    return f"fold_{int(fold):02d}__{variant}__{target}"


def _task_fingerprint(
    *, study_path: Path, prepared: PreparedData, task: Mapping[str, Any]
) -> str:
    return _stable_hash(
        {
            "study_sha256": _sha256(study_path),
            "prepared_fingerprint": prepared.manifest["study_fingerprint"],
            "task": dict(task),
        }
    )


def _task_complete(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        result = _read_json(path)
        return (
            result.get("status") == "completed"
            and result.get("task_fingerprint") == fingerprint
            and all(
                _record_valid(record, verify_hash=False)
                for record in result["files"].values()
            )
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _predict_batches(
    booster: lgb.Booster,
    matrix: np.ndarray,
    rows: np.ndarray,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    selected = np.asarray(rows, dtype=np.int64)
    output = np.empty(len(selected), dtype=np.float32)
    for left in range(0, len(selected), int(batch_size)):
        right = min(left + int(batch_size), len(selected))
        local = selected[left:right]
        values = np.asarray(matrix[local], dtype=np.float32)
        output[left:right] = np.asarray(booster.predict(values), dtype=np.float32)
    return output


def _daily_rank_ic(
    date_idx: np.ndarray, prediction: np.ndarray, actual: np.ndarray
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "date_idx": np.asarray(date_idx, dtype=np.int32),
            "prediction": np.asarray(prediction, dtype=np.float64),
            "actual": np.asarray(actual, dtype=np.float64),
        }
    ).dropna()
    frame["prediction_rank"] = frame.groupby("date_idx")["prediction"].rank(pct=True)
    frame["actual_rank"] = frame.groupby("date_idx")["actual"].rank(pct=True)
    return (
        frame.groupby("date_idx")[["prediction_rank", "actual_rank"]]
        .corr()
        .iloc[0::2, -1]
        .to_numpy(dtype=np.float64)
    )


def _base_metrics(
    *,
    target: str,
    date_idx: np.ndarray,
    prediction: np.ndarray,
    raw_actual: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    mask = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(raw_actual)
    )
    dates = np.asarray(date_idx, dtype=np.int32)[mask]
    scores = np.asarray(prediction, dtype=np.float64)[mask]
    actual = np.asarray(raw_actual, dtype=np.float64)[mask]
    if not len(actual):
        raise ValueError("base_metric_support_empty")
    frame = pd.DataFrame({"date_idx": dates, "score": scores, "actual": actual})
    frame["score_rank"] = frame.groupby("date_idx")["score"].rank(pct=True)
    top = frame["score_rank"] >= 0.9
    daily = frame.groupby("date_idx").agg(baseline=("actual", "mean"))
    top_daily = frame.loc[top].groupby("date_idx")["actual"].mean().rename("top")
    daily = daily.join(top_daily, how="inner")
    rank_ic = _daily_rank_ic(dates, scores, actual)
    result: dict[str, Any] = {
        "row_count": len(actual),
        "date_count": int(np.unique(dates).size),
        "daily_rank_ic_mean": float(np.nanmean(rank_ic)),
        "daily_rank_ic_positive_fraction": float(np.nanmean(rank_ic > 0.0)),
        "top_decile_actual_mean": float(daily["top"].mean()),
        "baseline_actual_mean": float(daily["baseline"].mean()),
        "top_decile_uplift": float((daily["top"] - daily["baseline"]).mean()),
    }
    if target == "next_close_up":
        binary = (actual > 0.0).astype(np.int8)
        result.update(
            {
                "auc": float(roc_auc_score(binary, scores)),
                "brier": float(brier_score_loss(binary, scores)),
                "accuracy_at_0p5": float(((scores >= 0.5) == binary).mean()),
            }
        )
    else:
        result.update(
            {
                "mse": float(mean_squared_error(actual, scores)),
                "prediction_mean": float(np.mean(scores)),
                "actual_mean": float(np.mean(actual)),
            }
        )
    return result


def _run_base_task(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    prepared: PreparedData,
    fold: Mapping[str, Any],
    variant: str,
    target: str,
    output_root: Path,
) -> dict[str, Any]:
    task = {
        "fold": int(fold["fold"]),
        "variant": variant,
        "target": target,
        "training_maximum_date_idx": int(fold["training_maximum_date_idx"]),
        "validation_start_date_idx": int(fold["validation_start_date_idx"]),
        "validation_end_date_idx": int(fold["validation_end_date_idx"]),
    }
    task_id = _task_id(task["fold"], variant, target)
    task_dir = output_root / "base_tasks" / task_id
    result_path = task_dir / "task_result.json"
    fingerprint = _task_fingerprint(study_path=study_path, prepared=prepared, task=task)
    if _task_complete(result_path, fingerprint):
        return _read_json(result_path)

    matrix, feature_names = _feature_source(prepared, variant)
    train_values, raw_values, valid = _target_arrays(prepared, target)
    dates = prepared.row_index["date_idx"].to_numpy(dtype=np.int32)
    train_rows = np.flatnonzero(dates <= int(fold["training_maximum_date_idx"])).astype(
        np.int64
    )
    evaluation_rows = np.flatnonzero(
        (dates >= int(fold["validation_start_date_idx"]))
        & (dates <= int(fold["validation_end_date_idx"]))
    ).astype(np.int64)
    train_label = np.asarray(train_values[train_rows]).copy()
    train_valid = np.asarray(valid[train_rows], dtype=bool)
    train_label[~train_valid] = 0
    train_weights = np.zeros(len(train_rows), dtype=np.float32)
    train_weights[train_valid] = learnability.date_equal_weights(
        dates[train_rows][train_valid]
    )
    sequence = MatrixSequence(
        matrix,
        train_rows,
        batch_size=int(study["base_model"]["sequence_batch_size"]),
    )
    dataset = lgb.Dataset(
        sequence,
        label=train_label,
        weight=train_weights,
        feature_name=list(feature_names),
        free_raw_data=True,
        params={
            "max_bin": int(study["base_model"]["max_bin"]),
            "data_random_seed": int(study["validation"]["seed"]),
            "feature_pre_filter": False,
            "verbosity": -1,
        },
    )
    _emit(
        "base_task_training_started",
        task_id=task_id,
        train_rows=len(train_rows),
        features=len(feature_names),
    )
    started = time.perf_counter()
    dataset.construct()
    booster = lgb.train(
        _base_parameters(study, target),
        dataset,
        num_boost_round=int(study["base_model"]["num_boost_round"]),
    )
    training_seconds = float(time.perf_counter() - started)
    evaluation_prediction = _predict_batches(booster, matrix, evaluation_rows)
    evaluation_valid = np.asarray(valid[evaluation_rows], dtype=bool)
    metrics = _base_metrics(
        target=target,
        date_idx=dates[evaluation_rows],
        prediction=evaluation_prediction,
        raw_actual=np.asarray(raw_values[evaluation_rows]),
        valid=evaluation_valid,
    )
    oof = prepared.row_index.iloc[evaluation_rows][
        ["candidate_id", "date_idx", "trade_date", "symbol"]
    ].copy()
    oof["prediction"] = evaluation_prediction
    oof["actual"] = np.asarray(raw_values[evaluation_rows], dtype=np.float32)
    oof["target_valid"] = evaluation_valid
    oof["fold"] = int(fold["fold"])
    oof["variant"] = variant
    oof["target"] = target

    candidate_dates = prepared.candidates["date_idx"].to_numpy(dtype=np.int32)
    candidate_mask = candidate_dates <= int(fold["validation_end_date_idx"])
    candidate_positions = np.flatnonzero(candidate_mask)
    candidate_model_rows = prepared.candidates.loc[
        candidate_mask, "model_row_position"
    ].to_numpy(dtype=np.int64)
    candidate_prediction = _predict_batches(
        booster, matrix, candidate_model_rows, batch_size=32768
    )
    candidate_scores = prepared.candidates.loc[
        candidate_mask,
        ["candidate_id", "date_idx", "trade_date", "symbol", "margin_rank"],
    ].copy()
    candidate_scores["candidate_row_position"] = candidate_positions
    candidate_scores["prediction"] = candidate_prediction
    candidate_scores["is_base_training_period"] = candidate_scores["date_idx"].to_numpy(
        dtype=np.int32
    ) <= int(fold["training_maximum_date_idx"])
    candidate_scores["is_outer_validation"] = candidate_scores["date_idx"].to_numpy(
        dtype=np.int32
    ) >= int(fold["validation_start_date_idx"])

    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": gain.astype(np.float64),
            "split": split.astype(np.int64),
        }
    ).sort_values("gain", ascending=False, kind="stable")
    task_dir.mkdir(parents=True, exist_ok=True)
    model_path = task_dir / "model.txt"
    model_partial = Path(str(model_path) + ".partial")
    booster.save_model(str(model_partial))
    os.replace(model_partial, model_path)
    oof_path = task_dir / "oof_predictions.parquet"
    candidate_path = task_dir / "candidate_predictions.parquet"
    importance_path = task_dir / "feature_importance.parquet"
    _write_parquet(oof_path, oof)
    _write_parquet(candidate_path, candidate_scores)
    _write_parquet(importance_path, importance)
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": task_id,
        "task_fingerprint": fingerprint,
        **task,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "train_row_count": len(train_rows),
        "train_valid_count": int(train_valid.sum()),
        "evaluation_row_count": len(evaluation_rows),
        "evaluation_valid_count": int(evaluation_valid.sum()),
        "training_seconds": training_seconds,
        "parameters": _base_parameters(study, target),
        "metrics": metrics,
        "files": {
            "model": _file_record(model_path),
            "oof_predictions": _file_record(oof_path, row_count=len(oof)),
            "candidate_predictions": _file_record(
                candidate_path, row_count=len(candidate_scores)
            ),
            "feature_importance": _file_record(
                importance_path, row_count=len(importance)
            ),
        },
    }
    _write_json(result_path, result)
    _emit(
        "base_task_completed",
        task_id=task_id,
        seconds=round(training_seconds, 2),
        metrics=metrics,
    )
    dataset = None
    booster = None
    sequence = None
    gc.collect()
    return result


def train_base(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    variants: Sequence[str] = VARIANTS,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    targets: Sequence[str] = TARGETS,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    study, resolved_study_path, prepared = load_prepared(
        study_path=study_path, output_root=output_root
    )
    root = _resolve(output_root)
    selected_variants = tuple(str(value) for value in variants)
    selected_targets = tuple(str(value) for value in targets)
    if not set(selected_variants).issubset(VARIANTS):
        raise ValueError("unknown_requested_variant")
    if not set(selected_targets).issubset(TARGETS):
        raise ValueError("unknown_requested_target")
    selected_folds = {int(value) for value in folds}
    tasks: list[tuple[dict[str, Any], str, str]] = []
    for fold in prepared.manifest["folds"]:
        if int(fold["fold"]) not in selected_folds:
            continue
        for variant in selected_variants:
            for target in selected_targets:
                tasks.append((dict(fold), variant, target))
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]
    results: list[dict[str, Any]] = []
    for fold, variant, target in tasks:
        results.append(
            _run_base_task(
                study=study,
                study_path=resolved_study_path,
                prepared=prepared,
                fold=fold,
                variant=variant,
                target=target,
                output_root=root,
            )
        )
    return results


def _load_candidate_base_scores(
    *, output_root: Path, fold: int, candidates: pd.DataFrame
) -> pd.DataFrame:
    result = candidates[["candidate_id"]].copy()
    for variant in VARIANTS:
        for target in TARGETS:
            task_id = _task_id(fold, variant, target)
            task = _read_json(output_root / "base_tasks" / task_id / "task_result.json")
            record = task["files"]["candidate_predictions"]
            if not _record_valid(record, verify_hash=False):
                raise ValueError(f"candidate_base_prediction_invalid:{task_id}")
            scores = pd.read_parquet(
                Path(record["path"]), columns=["candidate_id", "prediction"]
            ).rename(columns={"prediction": f"base__{variant}__{target}"})
            result = result.merge(
                scores, on="candidate_id", how="left", validate="one_to_one"
            )
    return result


def _candidate_matrix(
    prepared: PreparedData,
    candidates: pd.DataFrame,
    *,
    variant: str,
    base_scores: pd.DataFrame,
) -> tuple[np.ndarray, tuple[str, ...]]:
    finance_names = tuple(
        str(value)
        for value in prepared.manifest["candidate_features"]["financing_feature_names"]
    )
    context_names = tuple(
        str(value)
        for value in prepared.manifest["candidate_features"]["context_feature_names"]
    )
    finance = candidates.loc[:, finance_names].to_numpy(dtype=np.float32, copy=True)
    if variant == "m2_direct_557_margin":
        matrix = np.column_stack(
            [np.asarray(prepared.candidate_compact), finance]
        ).astype(np.float32)
        names = (
            *tuple(str(value) for value in prepared.manifest["compact_feature_names"]),
            *finance_names,
        )
        return matrix, names
    if variant == "m3_two_stage":
        score_names = tuple(
            f"base__{base_variant}__{target}"
            for base_variant in VARIANTS
            for target in TARGETS
        )
        scores = base_scores.loc[:, score_names].to_numpy(dtype=np.float32, copy=True)
        context = candidates.loc[:, context_names].to_numpy(dtype=np.float32, copy=True)
        matrix = np.column_stack([scores, finance, context]).astype(np.float32)
        return matrix, (*score_names, *finance_names, *context_names)
    raise ValueError(f"unknown_candidate_model_variant:{variant}")


def _second_stage_parameters(study: Mapping[str, Any], target: str) -> dict[str, Any]:
    config = dict(study["second_stage"])
    parameters: dict[str, Any] = {
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]),
        "min_child_samples": int(config["min_data_in_leaf"]),
        "reg_lambda": float(config["lambda_l2"]),
        "n_estimators": int(config["n_estimators"]),
        "subsample": 1.0,
        "colsample_bytree": 0.8,
        "n_jobs": 8,
        "random_state": int(study["validation"]["seed"]),
        "deterministic": True,
        "verbosity": -1,
    }
    if target == "next_close_up":
        parameters["objective"] = "binary"
    else:
        parameters["objective"] = "regression"
    return parameters


def _fit_candidate_models(
    *,
    study: Mapping[str, Any],
    prepared: PreparedData,
    candidates: pd.DataFrame,
    base_scores: pd.DataFrame,
    fold: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = candidates[["candidate_id", "date_idx"]].copy()
    for column in base_scores.columns:
        if column != "candidate_id":
            scores[column] = base_scores[column].to_numpy()
    train_mask = candidates["date_idx"].to_numpy(dtype=np.int32) <= int(
        fold["training_maximum_date_idx"]
    )
    evaluation_mask = (
        candidates["date_idx"].to_numpy(dtype=np.int32)
        >= int(fold["validation_start_date_idx"])
    ) & (
        candidates["date_idx"].to_numpy(dtype=np.int32)
        <= int(fold["validation_end_date_idx"])
    )
    importance_rows: list[pd.DataFrame] = []
    for variant in ("m2_direct_557_margin", "m3_two_stage"):
        matrix, names = _candidate_matrix(
            prepared, candidates, variant=variant, base_scores=base_scores
        )
        feature_valid = np.isfinite(matrix).any(axis=1)
        for target in TARGETS:
            if target == "next_close_up":
                target_valid = candidates["signal_close_return_d1"].notna().to_numpy()
                target_values = (
                    candidates["signal_close_return_d1"].to_numpy(dtype=np.float64)
                    > 0.0
                ).astype(np.int8)
                estimator: Any = lgb.LGBMClassifier(
                    **_second_stage_parameters(study, target)
                )
            else:
                cutoff = int(prepared.manifest["labels"]["maximum_outcome_date_idx"])
                target_valid = (
                    candidates["date_idx"].to_numpy(dtype=np.int32) + 2 <= cutoff
                )
                target_values = (
                    candidates["one_day_net_return"]
                    .fillna(0.0)
                    .to_numpy(dtype=np.float64)
                )
                valid_train_values = target_values[train_mask & target_valid]
                lower, upper = np.quantile(
                    valid_train_values, [0.01, 0.99], method="nearest"
                )
                target_values = np.clip(target_values, lower, upper)
                estimator = lgb.LGBMRegressor(**_second_stage_parameters(study, target))
            fit_mask = train_mask & target_valid & feature_valid
            predict_mask = evaluation_mask & feature_valid
            estimator.fit(
                matrix[fit_mask],
                target_values[fit_mask],
                sample_weight=learnability.date_equal_weights(
                    candidates.loc[fit_mask, "date_idx"].to_numpy(dtype=np.int32)
                ),
                feature_name=list(names),
            )
            column = f"candidate__{variant}__{target}"
            values = np.full(len(candidates), np.nan, dtype=np.float32)
            if target == "next_close_up":
                values[predict_mask] = estimator.predict_proba(matrix[predict_mask])[
                    :, 1
                ]
            else:
                values[predict_mask] = estimator.predict(matrix[predict_mask])
            scores[column] = values
            gain = estimator.booster_.feature_importance(importance_type="gain")
            importance_rows.append(
                pd.DataFrame(
                    {
                        "fold": int(fold["fold"]),
                        "variant": variant,
                        "target": target,
                        "feature": names,
                        "gain": gain.astype(np.float64),
                    }
                )
            )
        del matrix
        gc.collect()
    return scores, pd.concat(importance_rows, ignore_index=True)


def _within_date_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date_idx")[column].rank(pct=True, method="average")


def _score_columns(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {
        "margin_rank": {"margin": "score__margin_rank"}
    }
    frame["score__margin_rank"] = -frame["margin_rank"].astype(float)
    mappings = {
        "m0_transparent_66": "base__transparent_66",
        "m1_compact_557": "base__compact_557",
        "m2_direct_557_margin": "candidate__m2_direct_557_margin",
        "m3_two_stage": "candidate__m3_two_stage",
    }
    for variant, prefix in mappings.items():
        up_source = f"{prefix}__next_close_up"
        legal_source = f"{prefix}__legal_gross_return"
        up = f"score__{variant}__direction"
        legal = f"score__{variant}__legal"
        composite = f"score__{variant}__composite"
        frame[up] = frame[up_source]
        frame[legal] = frame[legal_source]
        frame[composite] = (
            _within_date_rank(frame, up_source) + _within_date_rank(frame, legal_source)
        ) / 2.0
        result[variant] = {
            "direction": up,
            "legal": legal,
            "composite": composite,
        }
    return result


def _eligible_evaluation_frame(
    frame: pd.DataFrame, *, prepared: PreparedData, fold: Mapping[str, Any]
) -> pd.DataFrame:
    cutoff = int(prepared.manifest["labels"]["maximum_outcome_date_idx"])
    result = frame.loc[
        (frame["date_idx"] >= int(fold["validation_start_date_idx"]))
        & (frame["date_idx"] <= int(fold["validation_end_date_idx"]))
        & (frame["date_idx"] + 2 <= cutoff)
    ].copy()
    result["cash_net_return"] = result["one_day_net_return"].fillna(0.0)
    result["next_close_observed"] = result["signal_close_return_d1"].notna()
    result["next_close_up_observed"] = result["signal_close_return_d1"] > 0.0
    return result


def _select_top(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_k: int,
    no_trade: bool,
    score_type: str,
    variant: str,
) -> pd.DataFrame:
    ordered = frame.loc[np.isfinite(frame[score_column])].sort_values(
        ["date_idx", score_column, "symbol"],
        ascending=[True, False, True],
        kind="stable",
    )
    selected = ordered.groupby("date_idx", sort=False).head(int(top_k)).copy()
    if not no_trade or selected.empty or variant == "margin_rank":
        return selected
    top = selected.groupby("date_idx")[score_column].transform("max")
    if score_type == "direction":
        keep_date = top > 0.5
    elif score_type == "legal":
        keep_date = top > 0.0
    else:
        direction = f"score__{variant}__direction"
        legal = f"score__{variant}__legal"
        top_direction = selected.groupby("date_idx")[direction].transform("max")
        top_legal = selected.groupby("date_idx")[legal].transform("max")
        keep_date = (top_direction > 0.5) & (top_legal > 0.0)
    return selected.loc[keep_date].copy()


def _daily_selection_rows(
    frame: pd.DataFrame,
    *,
    fold: int,
    variant: str,
    score_type: str,
    score_column: str,
    top_k: int,
    no_trade: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = _select_top(
        frame,
        score_column=score_column,
        top_k=top_k,
        no_trade=no_trade,
        score_type=score_type,
        variant=variant,
    )
    baseline = frame.groupby(
        ["date_idx", "trade_date", "evaluation_year"], as_index=False
    ).agg(
        baseline_cash_net_return=("cash_net_return", "mean"),
        baseline_next_close_up=("next_close_up_observed", "mean"),
    )
    if selected.empty:
        daily = baseline.iloc[0:0].copy()
        for name in (
            "selected_count",
            "filled_count",
            "daily_net_return",
            "next_close_up_fraction",
            "mean_score",
            "paired_delta",
        ):
            daily[name] = pd.Series(dtype=float)
        return daily, selected
    selected["selected_rank"] = selected.groupby("date_idx")[score_column].rank(
        ascending=False, method="first"
    )
    daily = (
        selected.groupby(["date_idx", "trade_date", "evaluation_year"], as_index=False)
        .agg(
            selected_count=("candidate_id", "size"),
            filled_count=("one_day_entry_filled", "sum"),
            daily_net_return=("cash_net_return", "mean"),
            next_close_up_fraction=("next_close_up_observed", "mean"),
            mean_score=(score_column, "mean"),
        )
        .merge(
            baseline,
            on=["date_idx", "trade_date", "evaluation_year"],
            how="left",
            validate="one_to_one",
        )
    )
    daily["paired_delta"] = (
        daily["daily_net_return"] - daily["baseline_cash_net_return"]
    )
    daily["fold"] = int(fold)
    daily["variant"] = variant
    daily["score_type"] = score_type
    daily["top_k"] = int(top_k)
    daily["no_trade"] = bool(no_trade)
    selected["fold"] = int(fold)
    selected["variant"] = variant
    selected["score_type"] = score_type
    selected["top_k"] = int(top_k)
    selected["no_trade"] = bool(no_trade)
    selected["selection_score"] = selected[score_column]
    return daily, selected


def _selection_summary(
    daily: pd.DataFrame, selected: pd.DataFrame, study: Mapping[str, Any]
) -> list[dict[str, Any]]:
    keys = ["variant", "score_type", "top_k", "no_trade"]
    summaries: list[dict[str, Any]] = []
    gate = dict(study["evaluation"]["entry_gate"])
    validation = dict(study["validation"])
    inference_study = {
        "evaluation": {
            "hac_lag_days": validation["hac_lag_days"],
            "block_length_days": validation["block_length_days"],
            "bootstrap_repetitions": validation["bootstrap_repetitions"],
            "seed": validation["seed"],
        }
    }
    for values, group in daily.groupby(keys, sort=False, dropna=False):
        mask = np.ones(len(selected), dtype=bool)
        for key, value in zip(keys, values, strict=True):
            mask &= selected[key].to_numpy() == value
        chosen = selected.loc[mask]
        fold_net = group.groupby("fold")["daily_net_return"].mean()
        fold_delta = group.groupby("fold")["paired_delta"].mean()
        annual = group.groupby("evaluation_year")["daily_net_return"].mean()
        net_inference = margin._inference(
            group["daily_net_return"].to_numpy(dtype=np.float64),
            inference_study,
            seed_add=11000 + len(summaries),
        )
        delta_inference = margin._inference(
            group["paired_delta"].to_numpy(dtype=np.float64),
            inference_study,
            seed_add=12000 + len(summaries),
        )
        candidate_returns = chosen["cash_net_return"].to_numpy(dtype=np.float64)
        if len(candidate_returns):
            cutoff = float(np.quantile(candidate_returns, 0.99, method="nearest"))
            trimmed = chosen.loc[chosen["cash_net_return"] <= cutoff]
            trimmed_daily = trimmed.groupby("date_idx")["cash_net_return"].mean()
            trimmed_mean = float(trimmed_daily.mean()) if len(trimmed_daily) else None
        else:
            cutoff = float("nan")
            trimmed_mean = None
        passed = (
            int((fold_net > 0.0).sum()) >= int(gate["minimum_positive_legal_net_folds"])
            and int((fold_delta > 0.0).sum())
            >= int(gate["minimum_positive_paired_delta_folds"])
            and float(net_inference["lcb_95"]) > 0.0
            and float(delta_inference["lcb_95"]) > 0.0
            and int((annual > 0.0).sum())
            >= int(gate["minimum_positive_calendar_years"])
            and trimmed_mean is not None
            and trimmed_mean > 0.0
        )
        summaries.append(
            {
                **dict(zip(keys, values, strict=True)),
                "selected_candidate_count": len(chosen),
                "selected_date_count": int(group["date_idx"].nunique()),
                "available_validation_date_fraction": float(
                    group["date_idx"].nunique() / max(daily["date_idx"].nunique(), 1)
                ),
                "fill_fraction": float(chosen["one_day_entry_filled"].mean()),
                "next_close_up_fraction": float(
                    chosen.loc[
                        chosen["next_close_observed"], "next_close_up_observed"
                    ].mean()
                ),
                "daily_net_return": net_inference,
                "paired_delta_vs_top10": delta_inference,
                "positive_net_fold_count": int((fold_net > 0.0).sum()),
                "positive_delta_fold_count": int((fold_delta > 0.0).sum()),
                "positive_year_count": int((annual > 0.0).sum()),
                "year_count": len(annual),
                "top_one_percent_cutoff": cutoff,
                "remove_top_one_percent_daily_mean": trimmed_mean,
                "entry_gate_passed": bool(passed),
            }
        )
    return summaries


def evaluate(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study, _, prepared = load_prepared(study_path=study_path, output_root=output_root)
    root = _resolve(output_root)
    required_results: list[dict[str, Any]] = []
    for fold in prepared.manifest["folds"]:
        for variant in VARIANTS:
            for target in TARGETS:
                task_id = _task_id(int(fold["fold"]), variant, target)
                path = root / "base_tasks" / task_id / "task_result.json"
                if not path.is_file():
                    raise ValueError(f"base_task_missing:{task_id}")
                required_results.append(_read_json(path))

    candidate_scores_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    candidates = prepared.candidates.copy()
    for fold in prepared.manifest["folds"]:
        fold_number = int(fold["fold"])
        base_scores = _load_candidate_base_scores(
            output_root=root, fold=fold_number, candidates=candidates
        )
        model_scores, importance = _fit_candidate_models(
            study=study,
            prepared=prepared,
            candidates=candidates,
            base_scores=base_scores,
            fold=fold,
        )
        combined = candidates.merge(
            model_scores.drop(columns=["date_idx"]),
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        evaluation = _eligible_evaluation_frame(combined, prepared=prepared, fold=fold)
        mappings = _score_columns(evaluation)
        evaluation["fold"] = fold_number
        candidate_scores_parts.append(evaluation)
        importance_parts.append(importance)
        for variant, score_types in mappings.items():
            for score_type, column in score_types.items():
                for top_k in study["second_stage"]["daily_selections"]:
                    for no_trade in (
                        (False,) if variant == "margin_rank" else (False, True)
                    ):
                        daily, selected = _daily_selection_rows(
                            evaluation,
                            fold=fold_number,
                            variant=variant,
                            score_type=score_type,
                            score_column=column,
                            top_k=int(top_k),
                            no_trade=bool(no_trade),
                        )
                        daily_parts.append(daily)
                        selected_parts.append(selected)
    candidate_scores = pd.concat(candidate_scores_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    selected = pd.concat(selected_parts, ignore_index=True)
    importance = pd.concat(importance_parts, ignore_index=True)
    summaries = _selection_summary(daily, selected, study)
    base_metrics = [
        {
            "fold": int(result["fold"]),
            "variant": str(result["variant"]),
            "target": str(result["target"]),
            **dict(result["metrics"]),
        }
        for result in required_results
    ]
    passed = [row for row in summaries if bool(row["entry_gate_passed"])]
    candidate_scores_path = root / "candidate_oof_scores.parquet"
    daily_path = root / "daily_selection_metrics.parquet"
    selected_path = root / "selected_candidates.parquet"
    importance_path = root / "second_stage_feature_importance.parquet"
    base_metrics_path = root / "base_model_metrics.parquet"
    _write_parquet(candidate_scores_path, candidate_scores)
    _write_parquet(daily_path, daily)
    _write_parquet(selected_path, selected)
    _write_parquet(importance_path, importance)
    _write_parquet(base_metrics_path, pd.DataFrame(base_metrics))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "epistemic_status": "adaptive_retrospective_development_validation",
        "stable_profit_claim_allowed": False,
        "production_claim_allowed": False,
        "account_replay_performed": False,
        "account_replay_gate_passed": bool(passed),
        "candidate_count": len(candidates),
        "validation_candidate_count": len(candidate_scores),
        "validation_date_count": int(candidate_scores["date_idx"].nunique()),
        "folds": prepared.manifest["folds"],
        "base_model_metrics": base_metrics,
        "selection_summaries": summaries,
        "passed_selection_count": len(passed),
        "passed_selections": passed,
        "forbidden_2026_read_count": 0,
        "files": {
            "candidate_oof_scores": _file_record(
                candidate_scores_path, row_count=len(candidate_scores)
            ),
            "daily_selection_metrics": _file_record(daily_path, row_count=len(daily)),
            "selected_candidates": _file_record(selected_path, row_count=len(selected)),
            "second_stage_feature_importance": _file_record(
                importance_path, row_count=len(importance)
            ),
            "base_model_metrics": _file_record(
                base_metrics_path, row_count=len(base_metrics)
            ),
        },
    }
    summary_path = root / "summary.json"
    _write_json(summary_path, summary)
    validate_summary(summary_path)
    return summary


def validate_summary(path: str | Path) -> dict[str, Any]:
    summary_path = _resolve(path)
    summary = _read_json(summary_path)
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("status") != "completed"
        or summary.get("study_id") != STUDY_ID
        or bool(summary.get("stable_profit_claim_allowed", True))
        or bool(summary.get("production_claim_allowed", True))
        or int(summary.get("forbidden_2026_read_count", -1)) != 0
        or int(summary.get("candidate_count", -1)) != 33877
        or len(summary.get("folds", ())) != 5
    ):
        raise ValueError("summary_contract_failed")
    if not all(_record_valid(record) for record in summary["files"].values()):
        raise ValueError("summary_output_file_invalid")
    if bool(summary["account_replay_performed"]):
        raise ValueError("unexpected_account_replay")
    return summary


def run(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    prepare(study_path=study_path, output_root=output_root)
    train_base(study_path=study_path, output_root=output_root)
    return evaluate(study_path=study_path, output_root=output_root)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _parse_csv(value))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "train", "evaluate", "run", "validate")
    )
    parser.add_argument("--study-path", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--summary-path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        payload = prepare(
            study_path=args.study_path,
            output_root=args.output_root,
            force=bool(args.force),
        )
    elif args.command == "train":
        payload = train_base(
            study_path=args.study_path,
            output_root=args.output_root,
            variants=_parse_csv(args.variants),
            targets=_parse_csv(args.targets),
            folds=_parse_int_csv(args.folds),
            max_tasks=args.max_tasks,
        )
    elif args.command == "evaluate":
        payload = evaluate(study_path=args.study_path, output_root=args.output_root)
    elif args.command == "run":
        payload = run(study_path=args.study_path, output_root=args.output_root)
    else:
        summary_path = (
            _resolve(args.summary_path)
            if args.summary_path
            else _resolve(args.output_root) / "summary.json"
        )
        payload = validate_summary(summary_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
