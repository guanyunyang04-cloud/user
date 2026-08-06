"""Executable stock-level conditional-distribution research for Seq100.

The module keeps execution states separate from continuous path outcomes. A
signal is made at the close of date t; the hypothetical entry anchor is the
next trading-day open. A planned liquidation is the close of trading day H
and may be delayed by a bounded retry window when the close is not legally
sellable. The first implementation deliberately uses transparent Student-t
location/scale models and date-equal rolling evaluation. It is a research
contract, not a production trading policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_stock_distribution_v1.json"
)
DEFAULT_INPUT_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/"
    / "seq100_quality_liquidity_training_ready/model_inputs/manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_stock_distribution_v1"
)
DEFAULT_MEMBERSHIP_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/"
    / "seq100_quality_liquidity_data_prep/membership"
)
DEFAULT_PACK_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research/data/research_store/seq100_pit_l35v2_v1/pack/manifest.json"
)

STUDY_ID = "seq100_stock_distribution_v1"
LABEL_SCHEMA = "seq100_stock_executable_distribution_labels/1"
MANIFEST_SCHEMA = "seq100_stock_executable_distribution_manifest/1"
FORMAL_START_YEAR = 2012
FORBIDDEN_YEAR = 2026
MAXIMUM_OUTCOME_DATE = "2025-12-31"
DEVELOPMENT_YEARS = (2017, 2018, 2019, 2020, 2021, 2022)
RETROSPECTIVE_OOS_YEARS = (2023, 2024, 2025)
HORIZONS = (5, 10, 20)
PRIMARY_HORIZONS = (10, 20)
RETRY_DAYS = 20
EXPECTED_INPUT_FINGERPRINT = (
    "a2bea9ed174b0a04ea2e97af475e9b4f1da79d739288a0cd92114998e31784ae"
)
EXPECTED_ROW_COUNT = 4_191_476
EXPECTED_SYMBOL_COUNT = 3_419

UNKNOWN_STATE = -1
KNOWN_BLOCKED = 0
KNOWN_FILLED = 1

FLAG_OUTCOME_WITHIN_CUTOFF = np.uint16(1 << 0)
FLAG_ENTRY_PRICE_OBSERVED = np.uint16(1 << 1)
FLAG_ENTRY_STATE_KNOWN = np.uint16(1 << 2)
FLAG_ENTRY_FILLED = np.uint16(1 << 3)
FLAG_PLANNED_PATH_VALID = np.uint16(1 << 4)
FLAG_SELL_STATE_KNOWN = np.uint16(1 << 5)
FLAG_SELL_FILLED = np.uint16(1 << 6)
FLAG_EXECUTABLE_RETURN_VALID = np.uint16(1 << 7)
FLAG_MFE_VALID = np.uint16(1 << 8)
FLAG_MAE_VALID = np.uint16(1 << 9)
FLAG_INDUSTRY_KNOWN = np.uint16(1 << 10)

FLAG_DEFINITIONS = {
    "outcome_within_cutoff": int(FLAG_OUTCOME_WITHIN_CUTOFF),
    "entry_price_observed": int(FLAG_ENTRY_PRICE_OBSERVED),
    "entry_state_known": int(FLAG_ENTRY_STATE_KNOWN),
    "entry_filled": int(FLAG_ENTRY_FILLED),
    "planned_path_valid": int(FLAG_PLANNED_PATH_VALID),
    "sell_state_known": int(FLAG_SELL_STATE_KNOWN),
    "sell_filled": int(FLAG_SELL_FILLED),
    "executable_return_valid": int(FLAG_EXECUTABLE_RETURN_VALID),
    "mfe_valid": int(FLAG_MFE_VALID),
    "mae_valid": int(FLAG_MAE_VALID),
    "industry_known": int(FLAG_INDUSTRY_KNOWN),
}

CONTINUOUS_PER_HORIZON = (
    "shadow_log_return",
    "executable_log_return",
    "residual_log_return",
    "mfe_log",
    "mae_log",
    "market_component",
    "industry_component",
)
STATE_COLUMNS = ("entry_action",) + tuple(f"sell_action_{h}" for h in HORIZONS)
TAU_COLUMNS = tuple(f"tau_sell_{h}" for h in HORIZONS)
FLAG_COLUMNS = ("entry",) + tuple(f"h{h}" for h in HORIZONS)

FEATURE_BLOCK_FAMILIES: dict[str, tuple[str, ...]] = {
    "core": (
        "market_state",
        "industry_context",
        "daily_price_volume_technical",
        "daily_cross_sectional_technical",
        "size_liquidity_and_status",
    ),
    "minute": ("same_day_5m",),
    "pit": (
        "structured_financial_summary",
        "financial_statements",
        "financial_statement_extensions",
        "announcements",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _resolve_path(value: str | Path, *, base: Path = WORKSPACE_ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _file_record(
    path: Path,
    *,
    shape: Sequence[int] | None = None,
    dtype: str | None = None,
    columns: Sequence[str] | None = None,
    include_hash: bool = True,
) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {
        "path": str(path),
        "size": int(path.stat().st_size),
    }
    if include_hash:
        record["sha256"] = _sha256_file(path)
    if shape is not None:
        record["shape"] = [int(v) for v in shape]
    if dtype is not None:
        record["dtype"] = str(dtype)
    if columns is not None:
        record["columns"] = [str(v) for v in columns]
    return record


def _record_valid(record: Mapping[str, Any], *, verify_hash: bool = False) -> bool:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or int(record.get("size", -1)) != int(path.stat().st_size):
        return False
    return not (verify_hash and record.get("sha256") != _sha256_file(path))


def _open_memmap(
    record: Mapping[str, Any],
    *,
    dtype: np.dtype[Any] | type | str | None = None,
    mode: str = "r",
) -> np.memmap:
    path = _resolve_path(str(record["path"]))
    shape = tuple(int(v) for v in record["shape"])
    actual_dtype = np.dtype(dtype if dtype is not None else record["dtype"])
    expected_size = int(np.prod(shape)) * actual_dtype.itemsize
    if not path.is_file() or int(path.stat().st_size) != expected_size:
        raise ValueError(f"array_record_invalid:{path}")
    return np.memmap(path, dtype=actual_dtype, mode=mode, shape=shape)


def load_study(study_path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    path = _resolve_path(study_path)
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if str(source.get("expected_input_fingerprint")) != EXPECTED_INPUT_FINGERPRINT:
        raise ValueError("expected_input_fingerprint_mismatch")
    if int(source.get("expected_row_count", -1)) != EXPECTED_ROW_COUNT:
        raise ValueError("expected_row_count_mismatch")
    target = dict(study.get("target", {}) or {})
    if tuple(int(v) for v in target.get("horizons", ())) != HORIZONS:
        raise ValueError("target_horizons_mismatch")
    if int(target.get("retry_days", -1)) != RETRY_DAYS:
        raise ValueError("target_retry_window_mismatch")
    evaluation = dict(study.get("evaluation", {}) or {})
    if bool(evaluation.get("training_performed", True)):
        raise ValueError("study_must_start_without_training")
    return study


def feature_names_for_block(
    input_manifest: Mapping[str, Any], block: str
) -> tuple[str, ...]:
    if block not in {"core", "core_minute", "core_minute_pit"}:
        raise ValueError(f"unknown_feature_block:{block}")
    families = list(FEATURE_BLOCK_FAMILIES["core"])
    if block in {"core_minute", "core_minute_pit"}:
        families.extend(FEATURE_BLOCK_FAMILIES["minute"])
    if block == "core_minute_pit":
        families.extend(FEATURE_BLOCK_FAMILIES["pit"])
    wanted = set(families)
    names = [
        str(record["feature_name"])
        for record in input_manifest["features"]
        if str(record.get("analytic_family")) in wanted
    ]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"feature_block_invalid:{block}")
    return tuple(names)


@dataclass
class StockPanel:
    input_manifest: dict[str, Any]
    label_manifest: dict[str, Any]
    pack_manifest: dict[str, Any]
    row_index: pd.DataFrame
    compact: np.memmap
    feature_names: tuple[str, ...]
    date_idx: np.ndarray
    symbol_idx: np.ndarray
    trade_date: np.ndarray
    years: np.ndarray
    industry_code: np.ndarray
    continuous: np.memmap
    states: np.memmap
    tau: np.memmap
    flags: np.memmap
    continuous_columns: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return len(self.row_index)

    def rows_for_year(self, year: int) -> np.ndarray:
        rows = np.flatnonzero(self.years == int(year)).astype(np.int64, copy=False)
        if not len(rows):
            raise ValueError(f"model_input_year_missing:{year}")
        return rows

    def target(self, name: str) -> np.ndarray:
        try:
            column = self.continuous_columns.index(str(name))
        except ValueError as exc:
            raise ValueError(f"continuous_target_missing:{name}") from exc
        return np.asarray(self.continuous[:, column], dtype=np.float64)

    def state(self, name: str) -> np.ndarray:
        if name == "entry_action":
            return np.asarray(self.states[:, 0], dtype=np.int8)
        horizon = int(str(name).rsplit("_", 1)[1])
        if not name.startswith("sell_action_") or horizon not in HORIZONS:
            raise ValueError(f"state_target_missing:{name}")
        return np.asarray(self.states[:, 1 + HORIZONS.index(horizon)], dtype=np.int8)

    def target_valid(self, name: str) -> np.ndarray:
        values: np.ndarray
        if name.startswith("sell_action_"):
            values = self.state(name).astype(np.float64)
            bit = FLAG_SELL_STATE_KNOWN
            column = HORIZONS.index(int(name.rsplit("_", 1)[1]))
            return ((self.flags[:, column + 1] & bit) != 0) & (values >= 0)
        if name == "entry_action":
            values = self.state(name).astype(np.float64)
            return ((self.flags[:, 0] & FLAG_ENTRY_STATE_KNOWN) != 0) & (values >= 0)
        values = self.target(name)
        horizon = int(name.rsplit("_", 1)[1])
        column = HORIZONS.index(horizon) + 1
        if name.startswith(("executable_log_return_", "residual_log_return_")):
            bit = FLAG_EXECUTABLE_RETURN_VALID
        elif name.startswith("mfe_log_"):
            bit = FLAG_MFE_VALID
        elif name.startswith("mae_log_"):
            bit = FLAG_MAE_VALID
        elif name.startswith("shadow_log_return_"):
            bit = FLAG_PLANNED_PATH_VALID
        else:
            raise ValueError(f"target_validity_unknown:{name}")
        return ((self.flags[:, column] & bit) != 0) & np.isfinite(values)


def _load_input_contract(
    input_manifest_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = _read_json(input_manifest_path)
    if (
        manifest.get("status") != "completed"
        or manifest.get("study_id") != "seq100_quality_liquidity_model"
        or str(manifest.get("input_fingerprint")) != EXPECTED_INPUT_FINGERPRINT
        or int(manifest.get("row_count", -1)) != EXPECTED_ROW_COUNT
        or bool(manifest.get("training_performed", True))
    ):
        raise ValueError("model_input_contract_mismatch")
    row_record = dict(manifest["row_index"])
    row_path = _resolve_path(str(row_record["path"]))
    row_index = pd.read_parquet(row_path)
    expected = [
        "candidate_id",
        "date_idx",
        "symbol_idx",
        "trade_date",
        "symbol",
        "security_id",
        "legacy_candidate_row",
    ]
    if list(row_index.columns) != expected:
        raise ValueError("model_input_row_index_schema_mismatch")
    if len(row_index) != EXPECTED_ROW_COUNT:
        raise ValueError("model_input_row_count_mismatch")
    if bool(row_index["candidate_id"].duplicated().any()):
        raise ValueError("model_input_candidate_id_not_unique")
    dates = row_index["trade_date"].astype(str)
    if bool(dates.str.startswith("2026-").any()):
        raise ValueError("forbidden_2026_model_input")
    date_values = row_index["date_idx"].to_numpy(dtype=np.int64)
    if bool((date_values[1:] < date_values[:-1]).any()):
        raise ValueError("model_input_not_date_ordered")
    return manifest, row_index


def _load_industry_codes(
    row_index: pd.DataFrame, membership_root: Path
) -> np.ndarray:
    """Align PIT industry membership by candidate identity, never row offset."""

    codes = np.full(len(row_index), -1, dtype=np.int32)
    mapping: dict[str, int] = {}
    years = row_index["trade_date"].astype(str).str[:4].astype(int).to_numpy()
    for year in sorted(np.unique(years)):
        positions = np.flatnonzero(years == int(year))
        path = membership_root / f"year={int(year)}" / "diagnostics.parquet"
        if not path.is_file():
            raise ValueError(f"industry_membership_missing:{year}")
        source = pd.read_parquet(path, columns=["candidate_id", "industry"])
        if bool(source["candidate_id"].duplicated().any()):
            raise ValueError(f"industry_candidate_id_duplicate:{year}")
        selected = row_index.iloc[positions][["candidate_id"]].merge(
            source, on="candidate_id", how="left", validate="one_to_one"
        )
        values = selected["industry"].astype("string")
        result = np.full(len(values), -1, dtype=np.int32)
        for offset, value in enumerate(values):
            text = "" if pd.isna(value) else str(value).strip()
            if not text or text.lower() in {"nan", "none", "<na>"}:
                continue
            if text not in mapping:
                mapping[text] = len(mapping)
            result[offset] = mapping[text]
        codes[positions] = result
    return codes


def _pack_record(pack: Mapping[str, Any], section: str, name: str) -> Mapping[str, Any]:
    try:
        return pack[section][name]
    except KeyError as exc:
        raise ValueError(f"pack_record_missing:{section}:{name}") from exc


def _open_pack_array(
    pack: Mapping[str, Any],
    section: str,
    name: str,
    *,
    dtype: np.dtype[Any] | type | str,
) -> np.memmap:
    return _open_memmap(_pack_record(pack, section, name), dtype=dtype)


def _leave_one_out_factors(
    values: np.ndarray,
    valid: np.ndarray,
    date_idx: np.ndarray,
    industry_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute date and PIT-industry factors without using the stock itself."""

    y = np.asarray(values, dtype=np.float64)
    good = np.asarray(valid, dtype=bool) & np.isfinite(y)
    dates = np.asarray(date_idx, dtype=np.int64)
    codes = np.asarray(industry_code, dtype=np.int64)
    market = np.full(len(y), np.nan, dtype=np.float64)
    industry = np.full(len(y), np.nan, dtype=np.float64)
    residual = np.full(len(y), np.nan, dtype=np.float64)
    if not good.any():
        return residual, market, industry
    date_count = int(dates.max(initial=0)) + 1
    sums = np.bincount(dates[good], weights=y[good], minlength=date_count)
    counts = np.bincount(dates[good], minlength=date_count).astype(np.float64)
    base_count = counts[dates]
    market_excl = np.divide(
        sums[dates] - np.where(good, y, 0.0),
        np.maximum(base_count - good.astype(np.float64), 1.0),
        out=np.zeros(len(y), dtype=np.float64),
    )
    singleton = good & (base_count <= 1.0)
    market_excl[singleton] = y[singleton]
    known_group = good & (codes >= 0)
    if known_group.any():
        width = int(codes[known_group].max()) + 1
        keys = dates[known_group] * width + codes[known_group]
        group_size = int(date_count * width)
        group_sums = np.bincount(
            keys, weights=y[known_group], minlength=group_size
        )
        group_counts = np.bincount(keys, minlength=group_size).astype(np.float64)
        all_keys = dates * width + np.maximum(codes, 0)
        gcount = group_counts[all_keys]
        gsum = group_sums[all_keys]
        group_excl = np.divide(
            gsum - np.where(known_group, y, 0.0),
            np.maximum(gcount - known_group.astype(np.float64), 1.0),
            out=market_excl.copy(),
        )
        use_market = (~known_group) | (gcount <= known_group.astype(np.float64))
        group_excl[use_market] = market_excl[use_market]
    else:
        group_excl = market_excl
    market[good] = market_excl[good]
    industry[good] = group_excl[good] - market_excl[good]
    residual[good] = y[good] - group_excl[good]
    return residual, market, industry


def _continuous_columns() -> tuple[str, ...]:
    return tuple(f"{field}_{h}" for h in HORIZONS for field in CONTINUOUS_PER_HORIZON)


def _study_fingerprint(
    study_path: Path, input_manifest_path: Path, pack_manifest_path: Path
) -> str:
    study = _read_json(study_path)
    payload = {
        "schema": LABEL_SCHEMA,
        "study_id": study["study_id"],
        "source_contract": study["source"],
        "target_contract": study["target"],
        "input_manifest_sha256": _sha256_file(input_manifest_path),
        "pack_manifest_sha256": _sha256_file(pack_manifest_path),
        "input_fingerprint": EXPECTED_INPUT_FINGERPRINT,
        "maximum_outcome_date": MAXIMUM_OUTCOME_DATE,
        "retry_days": RETRY_DAYS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _coverage_frame(
    row_index: pd.DataFrame,
    states: np.ndarray,
    tau: np.ndarray,
    flags: np.ndarray,
) -> pd.DataFrame:
    years = row_index["trade_date"].astype(str).str[:4].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for year in sorted(np.unique(years)):
        selected = years == int(year)
        record: dict[str, Any] = {
            "signal_year": int(year),
            "row_count": int(selected.sum()),
            "entry_state_known_count": int(
                np.sum((flags[selected, 0] & FLAG_ENTRY_STATE_KNOWN) != 0)
            ),
            "entry_filled_count": int(np.sum(states[selected, 0] == KNOWN_FILLED)),
        }
        for pos, h in enumerate(HORIZONS):
            flag = flags[selected, pos + 1]
            state = states[selected, pos + 1]
            record[f"h{h}_outcome_within_count"] = int(
                np.sum((flag & FLAG_OUTCOME_WITHIN_CUTOFF) != 0)
            )
            record[f"h{h}_planned_path_count"] = int(
                np.sum((flag & FLAG_PLANNED_PATH_VALID) != 0)
            )
            record[f"h{h}_executable_return_count"] = int(
                np.sum((flag & FLAG_EXECUTABLE_RETURN_VALID) != 0)
            )
            record[f"h{h}_sell_state_known_count"] = int(np.sum(state != UNKNOWN_STATE))
            record[f"h{h}_sell_filled_count"] = int(np.sum(state == KNOWN_FILLED))
            record[f"h{h}_censored_count"] = int(
                np.sum((state == KNOWN_BLOCKED) & ((flag & FLAG_SELL_STATE_KNOWN) != 0))
            )
            tau_values = tau[selected, pos]
            record[f"h{h}_median_tau_sell"] = float(
                np.median(tau_values[tau_values >= 0])
                if np.any(tau_values >= 0)
                else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _prepared_labels_valid(manifest: Mapping[str, Any]) -> bool:
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") not in {
        "prepared",
        "audited",
    }:
        return False
    return all(
        _record_valid(record)
        for record in dict(manifest.get("files", {}) or {}).values()
    )


def prepare_labels(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    input_manifest_path: str | Path = DEFAULT_INPUT_MANIFEST,
    pack_manifest_path: str | Path = DEFAULT_PACK_MANIFEST,
    membership_root: str | Path = DEFAULT_MEMBERSHIP_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    """Build corrected-input executable path labels with explicit censoring."""

    study_path = _resolve_path(study_path)
    input_manifest_path = _resolve_path(input_manifest_path)
    pack_manifest_path = _resolve_path(pack_manifest_path)
    membership_root = _resolve_path(membership_root)
    output_root = _resolve_path(output_root)
    load_study(study_path)
    input_manifest, row_index = _load_input_contract(input_manifest_path)
    pack = _read_json(pack_manifest_path)
    if int(pack.get("symbol_count", -1)) != EXPECTED_SYMBOL_COUNT:
        raise ValueError("pack_symbol_count_mismatch")
    date_values = np.asarray(pack.get("date_values", ()), dtype=str)
    cutoff_positions = np.flatnonzero(date_values == MAXIMUM_OUTCOME_DATE)
    if len(cutoff_positions) != 1:
        raise ValueError("maximum_outcome_date_missing")
    cutoff_idx = int(cutoff_positions[0])
    fingerprint = _study_fingerprint(
        study_path, input_manifest_path, pack_manifest_path
    )
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not force:
        current = _read_json(manifest_path)
        if current.get("experiment_fingerprint") != fingerprint:
            raise ValueError("existing_label_fingerprint_mismatch")
        if _prepared_labels_valid(current):
            return current

    output_root.mkdir(parents=True, exist_ok=True)
    labels_root = output_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    n = len(row_index)
    continuous_columns = _continuous_columns()
    continuous_path = labels_root / "continuous.float32.dat"
    states_path = labels_root / "states.int8.dat"
    tau_path = labels_root / "tau_sell.int16.dat"
    flags_path = labels_root / "flags.uint16.dat"
    industry_path = labels_root / "industry_code.int32.dat"
    partials = {
        continuous_path: Path(str(continuous_path) + ".partial"),
        states_path: Path(str(states_path) + ".partial"),
        tau_path: Path(str(tau_path) + ".partial"),
        flags_path: Path(str(flags_path) + ".partial"),
        industry_path: Path(str(industry_path) + ".partial"),
    }
    for path in partials.values():
        path.unlink(missing_ok=True)

    continuous = np.memmap(
        partials[continuous_path],
        dtype=np.float32,
        mode="w+",
        shape=(n, len(continuous_columns)),
    )
    states = np.memmap(
        partials[states_path], dtype=np.int8, mode="w+", shape=(n, len(STATE_COLUMNS))
    )
    tau = np.memmap(
        partials[tau_path], dtype=np.int16, mode="w+", shape=(n, len(TAU_COLUMNS))
    )
    flags = np.memmap(
        partials[flags_path],
        dtype=np.uint16,
        mode="w+",
        shape=(n, len(FLAG_COLUMNS)),
    )
    industry_codes = np.memmap(
        partials[industry_path], dtype=np.int32, mode="w+", shape=(n,)
    )
    continuous[:] = np.nan
    states[:] = UNKNOWN_STATE
    tau[:] = -1
    flags[:] = 0
    industry_codes[:] = _load_industry_codes(row_index, membership_root)

    date_idx = row_index["date_idx"].to_numpy(dtype=np.int64)
    symbol_idx = row_index["symbol_idx"].to_numpy(dtype=np.int64)
    daily_raw = _open_pack_array(
        pack, "feature_channels", "daily_raw", dtype=np.float32
    )
    close = daily_raw[:, :, 3]
    price_observed = _open_pack_array(
        pack, "masks", "price_observed", dtype=np.bool_
    )
    status_valid = _open_pack_array(pack, "masks", "status_valid", dtype=np.bool_)
    suspended = _open_pack_array(pack, "masks", "is_suspended", dtype=np.bool_)
    delisted = _open_pack_array(pack, "masks", "is_delisted", dtype=np.bool_)
    entry_buyable = _open_pack_array(
        pack, "masks", "entry_buyable", dtype=np.bool_
    )
    exit_sellable = _open_pack_array(
        pack, "masks", "exit_sellable", dtype=np.bool_
    )

    entry_index = date_idx + 1
    entry_in_range = entry_index < len(date_values)
    entry_price = np.full(n, np.nan, dtype=np.float64)
    entry_price[entry_in_range] = np.asarray(
        daily_raw[entry_index[entry_in_range], symbol_idx[entry_in_range], 0],
        dtype=np.float64,
    )
    safe_entry = np.minimum(entry_index, len(date_values) - 1)
    entry_obs = (
        entry_in_range
        & np.asarray(price_observed[safe_entry, symbol_idx], dtype=bool)
        & np.isfinite(entry_price)
        & (entry_price > 0.0)
    )
    entry_known = np.zeros(n, dtype=bool)
    entry_known[entry_in_range] = np.asarray(
        status_valid[entry_index[entry_in_range], symbol_idx[entry_in_range]],
        dtype=bool,
    ) & (
        np.asarray(
            price_observed[entry_index[entry_in_range], symbol_idx[entry_in_range]],
            dtype=bool,
        )
        | np.asarray(
            suspended[entry_index[entry_in_range], symbol_idx[entry_in_range]],
            dtype=bool,
        )
        | np.asarray(
            delisted[entry_index[entry_in_range], symbol_idx[entry_in_range]],
            dtype=bool,
        )
    )
    states[:, 0] = np.where(
        entry_known,
        np.where(
            np.asarray(entry_buyable[safe_entry, symbol_idx], dtype=bool),
            KNOWN_FILLED,
            KNOWN_BLOCKED,
        ),
        UNKNOWN_STATE,
    ).astype(np.int8)
    flags[:, 0] |= np.where(entry_obs, FLAG_ENTRY_PRICE_OBSERVED, 0).astype(
        np.uint16
    )
    flags[:, 0] |= np.where(entry_known, FLAG_ENTRY_STATE_KNOWN, 0).astype(
        np.uint16
    )
    flags[:, 0] |= np.where(
        states[:, 0] == KNOWN_FILLED, FLAG_ENTRY_FILLED, 0
    ).astype(np.uint16)
    flags[:, 0] |= np.where(
        np.asarray(industry_codes) >= 0, FLAG_INDUSTRY_KNOWN, 0
    ).astype(np.uint16)

    for hpos, horizon in enumerate(HORIZONS):
        h = int(horizon)
        flag_column = hpos + 1
        within = date_idx + h + RETRY_DAYS <= cutoff_idx
        flags[within, flag_column] |= FLAG_OUTCOME_WITHIN_CUTOFF
        endpoint_index = date_idx + h
        safe_endpoint = np.minimum(endpoint_index, len(date_values) - 1)
        endpoint_close = np.asarray(close[safe_endpoint, symbol_idx], dtype=np.float64)
        shadow = np.full(n, np.nan, dtype=np.float64)
        shadow_valid = (
            within
            & entry_obs
            & np.isfinite(endpoint_close)
            & (endpoint_close > 0.0)
        )
        shadow[shadow_valid] = np.log(
            endpoint_close[shadow_valid] / entry_price[shadow_valid]
        )

        path_min = np.full(n, np.inf, dtype=np.float64)
        mfe_max = np.full(n, -np.inf, dtype=np.float64)
        mfe_seen = np.zeros(n, dtype=bool)
        path_seen = np.zeros(n, dtype=bool)
        for offset in range(1, h + 1):
            path_index = date_idx + offset
            safe_path = np.minimum(path_index, len(date_values) - 1)
            path_close = np.asarray(close[safe_path, symbol_idx], dtype=np.float64)
            path_valid = (
                within
                & entry_obs
                & np.isfinite(path_close)
                & (path_close > 0.0)
            )
            log_path = np.full(n, np.nan, dtype=np.float64)
            log_path[path_valid] = np.log(
                path_close[path_valid] / entry_price[path_valid]
            )
            path_min[path_valid] = np.minimum(
                path_min[path_valid], log_path[path_valid]
            )
            path_seen |= path_valid
            if offset >= 2:
                sellable = path_valid & np.asarray(
                    exit_sellable[safe_path, symbol_idx], dtype=bool
                )
                mfe_max[sellable] = np.maximum(
                    mfe_max[sellable], log_path[sellable]
                )
                mfe_seen |= sellable
        mae = np.full(n, np.nan, dtype=np.float64)
        mae_valid = path_seen
        mae[mae_valid] = np.minimum(path_min[mae_valid], 0.0)
        mfe = np.full(n, np.nan, dtype=np.float64)
        mfe[mfe_seen] = np.maximum(mfe_max[mfe_seen], 0.0)
        flags[mae_valid, flag_column] |= FLAG_MAE_VALID
        flags[mfe_seen, flag_column] |= FLAG_MFE_VALID
        planned_valid = shadow_valid & path_seen
        flags[planned_valid, flag_column] |= FLAG_PLANNED_PATH_VALID

        first_offset = np.full(n, RETRY_DAYS + 1, dtype=np.int16)
        sell_known = np.ones(n, dtype=bool)
        sell_known[~within] = False
        for offset in range(h, h + RETRY_DAYS + 1):
            exit_index = date_idx + offset
            safe_exit = np.minimum(exit_index, len(date_values) - 1)
            exit_close = np.asarray(close[safe_exit, symbol_idx], dtype=np.float64)
            day_known = np.asarray(
                status_valid[safe_exit, symbol_idx], dtype=bool
            ) & (
                np.asarray(price_observed[safe_exit, symbol_idx], dtype=bool)
                | np.asarray(suspended[safe_exit, symbol_idx], dtype=bool)
                | np.asarray(delisted[safe_exit, symbol_idx], dtype=bool)
            )
            sell_known &= np.where(within, day_known, True)
            can_sell = (
                within
                & np.isfinite(exit_close)
                & (exit_close > 0.0)
                & np.asarray(exit_sellable[safe_exit, symbol_idx], dtype=bool)
            )
            first_offset[(first_offset == RETRY_DAYS + 1) & can_sell] = np.int16(
                offset - h
            )
        state_col = hpos + 1
        known_sell = within & sell_known
        states[:, state_col] = UNKNOWN_STATE
        states[known_sell, state_col] = np.where(
            first_offset[known_sell] <= RETRY_DAYS, KNOWN_FILLED, KNOWN_BLOCKED
        ).astype(np.int8)
        tau[known_sell, hpos] = first_offset[known_sell]
        flags[known_sell, flag_column] |= FLAG_SELL_STATE_KNOWN
        flags[
            known_sell & (states[:, state_col] == KNOWN_FILLED), flag_column
        ] |= FLAG_SELL_FILLED

        exe = np.full(n, np.nan, dtype=np.float64)
        exe_valid = (
            known_sell
            & (states[:, 0] == KNOWN_FILLED)
            & (states[:, state_col] == KNOWN_FILLED)
            & entry_obs
        )
        fill_index = date_idx + h + first_offset.astype(np.int64)
        safe_fill = np.minimum(fill_index, len(date_values) - 1)
        fill_close = np.asarray(close[safe_fill, symbol_idx], dtype=np.float64)
        exe_valid &= np.isfinite(fill_close) & (fill_close > 0.0)
        exe[exe_valid] = np.log(fill_close[exe_valid] / entry_price[exe_valid])
        flags[exe_valid, flag_column] |= FLAG_EXECUTABLE_RETURN_VALID

        _, market_factor, industry_factor = _leave_one_out_factors(
            shadow,
            shadow_valid,
            date_idx,
            np.asarray(industry_codes, dtype=np.int64),
        )
        residual_exe = np.full(n, np.nan, dtype=np.float64)
        residual_exe[exe_valid] = (
            exe[exe_valid] - market_factor[exe_valid] - industry_factor[exe_valid]
        )
        base = hpos * len(CONTINUOUS_PER_HORIZON)
        continuous[:, base + 0] = shadow.astype(np.float32)
        continuous[:, base + 1] = exe.astype(np.float32)
        continuous[:, base + 2] = residual_exe.astype(np.float32)
        continuous[:, base + 3] = mfe.astype(np.float32)
        continuous[:, base + 4] = mae.astype(np.float32)
        continuous[:, base + 5] = market_factor.astype(np.float32)
        continuous[:, base + 6] = industry_factor.astype(np.float32)
        continuous.flush()
        flags.flush()
        states.flush()
        tau.flush()
        print(
            json.dumps(
                {
                    "event": "stock_distribution_horizon_prepared",
                    "horizon": h,
                    "outcome_within_count": int(within.sum()),
                    "shadow_count": int(shadow_valid.sum()),
                    "executable_count": int(exe_valid.sum()),
                    "sell_filled_count": int(
                        np.sum(states[:, state_col] == KNOWN_FILLED)
                    ),
                    "mfe_count": int(mfe_seen.sum()),
                    "mae_count": int(mae_valid.sum()),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    continuous.flush()
    states.flush()
    tau.flush()
    flags.flush()
    industry_codes.flush()
    del continuous, states, tau, flags, industry_codes
    for final, partial in partials.items():
        os.replace(partial, final)

    states = np.memmap(
        states_path, dtype=np.int8, mode="r", shape=(n, len(STATE_COLUMNS))
    )
    tau = np.memmap(tau_path, dtype=np.int16, mode="r", shape=(n, len(TAU_COLUMNS)))
    flags = np.memmap(
        flags_path, dtype=np.uint16, mode="r", shape=(n, len(FLAG_COLUMNS))
    )
    coverage = _coverage_frame(row_index, states, tau, flags)
    coverage_path = output_root / "label_coverage.parquet"
    coverage.to_parquet(coverage_path, index=False)

    label_contract = {
        "schema": LABEL_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "row_count": n,
        "horizons": list(HORIZONS),
        "primary_horizons": list(PRIMARY_HORIZONS),
        "retry_days": RETRY_DAYS,
        "signal_decision": "signal_day_close",
        "entry_anchor": "next_trading_day_adjusted_open",
        "planned_exit": "H_trading_day_adjusted_close",
        "execution_retry": (
            "first status-valid, non-suspended, non-delisted, exit_sellable "
            "close in H..H+retry_days"
        ),
        "tau_definition": (
            "tau_sell_h is the delay from planned H close to first legal sellable "
            "close; retry_days+1 is right-censored"
        ),
        "continuous_semantics": {
            "shadow_log_return": "log(adjusted_close[t+H]/adjusted_entry_open[t+1])",
            "executable_log_return": (
                "log(adjusted_close[first legal sellable close]/"
                "adjusted_entry_open[t+1]); only when A_buy=A_sell=1"
            ),
            "residual_log_return": (
                "executable return minus leave-one-out date market and PIT-industry "
                "components estimated from shadow returns"
            ),
            "mfe_log": (
                "max(0, log(adjusted_close/entry_open)) over legally sellable "
                "closes in D2..DH"
            ),
            "mae_log": (
                "min(0, log(adjusted_close/entry_open)) over D1..DH using the "
                "pack's explicit suspended-price carry semantics"
            ),
        },
        "state_semantics": {
            "entry_action": {
                "-1": "future next-open execution state unknown",
                "0": "known not buyable at next open",
                "1": "known buyable and filled at next open",
            },
            "sell_action_h": {
                "-1": "unknown or outcome outside cutoff",
                "0": "known unresolved after retry window",
                "1": "known sellable within retry window",
            },
        },
        "future_state_not_used_for_candidate_selection": True,
        "forbidden_2026_read_count": 0,
        "maximum_source_date_read": MAXIMUM_OUTCOME_DATE,
        "training_performed": False,
    }
    label_files = {
        "continuous": _file_record(
            continuous_path,
            shape=(n, len(continuous_columns)),
            dtype="float32",
            columns=continuous_columns,
        ),
        "states": _file_record(
            states_path,
            shape=(n, len(STATE_COLUMNS)),
            dtype="int8",
            columns=STATE_COLUMNS,
        ),
        "tau": _file_record(
            tau_path,
            shape=(n, len(TAU_COLUMNS)),
            dtype="int16",
            columns=TAU_COLUMNS,
        ),
        "flags": _file_record(
            flags_path,
            shape=(n, len(FLAG_COLUMNS)),
            dtype="uint16",
            columns=FLAG_COLUMNS,
        ),
        "industry_code": _file_record(
            industry_path, shape=(n,), dtype="int32"
        ),
        "coverage": _file_record(coverage_path),
    }
    label_contract["files"] = label_files
    label_contract["sources"] = {
        "study": _file_record(study_path),
        "model_input_manifest": _file_record(input_manifest_path),
        "input_fingerprint": EXPECTED_INPUT_FINGERPRINT,
        "pack_manifest": _file_record(pack_manifest_path),
        "membership_root": str(membership_root),
        "row_index": _file_record(
            _resolve_path(str(input_manifest["row_index"]["path"]))
        ),
    }
    _write_json(output_root / "label_contract.json", label_contract)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "study_id": STUDY_ID,
        "experiment_fingerprint": fingerprint,
        "input_fingerprint": EXPECTED_INPUT_FINGERPRINT,
        "row_count": n,
        "date_count": int(row_index["date_idx"].nunique()),
        "start_date": str(row_index["trade_date"].iloc[0]),
        "end_date": str(row_index["trade_date"].iloc[-1]),
        "training_performed": False,
        "label_contract": _file_record(output_root / "label_contract.json"),
        "files": label_files,
        "coverage_summary": coverage.to_dict(orient="records"),
    }
    _write_json(manifest_path, manifest)
    return manifest


def load_panel(
    *,
    input_manifest_path: str | Path = DEFAULT_INPUT_MANIFEST,
    label_manifest_path: str | Path | None = None,
) -> StockPanel:
    """Load the current input and prepared labels with fingerprint checks."""

    input_manifest_path = _resolve_path(input_manifest_path)
    input_manifest, row_index = _load_input_contract(input_manifest_path)
    if label_manifest_path is None:
        label_manifest_path = DEFAULT_OUTPUT_ROOT / "manifest.json"
    label_manifest_path = _resolve_path(label_manifest_path)
    label_manifest = _read_json(label_manifest_path)
    if (
        label_manifest.get("schema") != MANIFEST_SCHEMA
        or str(label_manifest.get("input_fingerprint")) != EXPECTED_INPUT_FINGERPRINT
        or int(label_manifest.get("row_count", -1)) != len(row_index)
        or not _prepared_labels_valid(label_manifest)
    ):
        raise ValueError("prepared_stock_label_contract_mismatch")
    contract = _read_json(
        _resolve_path(str(label_manifest["label_contract"]["path"]))
    )
    pack_manifest = _read_json(
        _resolve_path(str(contract["sources"]["pack_manifest"]["path"]))
    )
    feature_records = list(input_manifest["features"])
    feature_names = tuple(str(r["feature_name"]) for r in feature_records)
    compact = _open_memmap(input_manifest["storage"]["compact"], dtype=np.float32)
    files = label_manifest["files"]
    continuous = _open_memmap(files["continuous"], dtype=np.float32)
    states = _open_memmap(files["states"], dtype=np.int8)
    tau = _open_memmap(files["tau"], dtype=np.int16)
    flags = _open_memmap(files["flags"], dtype=np.uint16)
    industry = _open_memmap(files["industry_code"], dtype=np.int32)
    trade_date = row_index["trade_date"].astype(str).to_numpy()
    return StockPanel(
        input_manifest=input_manifest,
        label_manifest=label_manifest,
        pack_manifest=pack_manifest,
        row_index=row_index,
        compact=compact,
        feature_names=feature_names,
        date_idx=row_index["date_idx"].to_numpy(dtype=np.int32),
        symbol_idx=row_index["symbol_idx"].to_numpy(dtype=np.int32),
        trade_date=trade_date,
        years=pd.Series(trade_date).str[:4].astype(int).to_numpy(dtype=np.int16),
        industry_code=np.asarray(industry, dtype=np.int32),
        continuous=continuous,
        states=states,
        tau=tau,
        flags=flags,
        continuous_columns=tuple(files["continuous"]["columns"]),
    )


@dataclass
class StudentTModel:
    name: str
    feature_names: tuple[str, ...]
    feature_positions: np.ndarray
    center: np.ndarray
    spread: np.ndarray
    mean_model: Ridge | None
    scale_model: Ridge | None
    constant_location: float
    df: float
    residual_scale: float


@dataclass
class BinaryModel:
    name: str
    feature_names: tuple[str, ...]
    feature_positions: np.ndarray
    center: np.ndarray
    spread: np.ndarray
    model: LogisticRegression | None
    constant_probability: float


def _feature_positions(
    panel: StockPanel, feature_names: Sequence[str]
) -> np.ndarray:
    mapping = {name: pos for pos, name in enumerate(panel.feature_names)}
    missing = [name for name in feature_names if name not in mapping]
    if missing:
        raise ValueError(f"model_features_missing:{missing[:5]}")
    return np.asarray([mapping[name] for name in feature_names], dtype=np.int64)


def _feature_matrix(
    panel: StockPanel, rows: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    selected_rows = np.asarray(rows, dtype=np.int64)
    selected_columns = np.asarray(positions, dtype=np.int64)
    if selected_rows.ndim != 1 or selected_columns.ndim != 1:
        raise ValueError("feature_matrix_index_dimension")
    return np.asarray(
        panel.compact[selected_rows[:, None], selected_columns[None, :]],
        dtype=np.float32,
    )


def _sample_rows_by_date(
    rows: np.ndarray,
    date_idx: np.ndarray,
    *,
    maximum_per_date: int,
) -> np.ndarray:
    """Deterministic date-stratified sample for bounded baseline fitting."""

    selected = np.asarray(rows, dtype=np.int64)
    if int(maximum_per_date) <= 0:
        raise ValueError("maximum_per_date_must_be_positive")
    if not len(selected):
        return selected
    dates = np.asarray(date_idx[selected], dtype=np.int64)
    if bool((dates[1:] < dates[:-1]).any()):
        order = np.argsort(dates, kind="stable")
        selected = selected[order]
        dates = dates[order]
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(dates)]
    pieces: list[np.ndarray] = []
    for left, right in pairwise(boundaries):
        local = selected[left:right]
        if len(local) <= int(maximum_per_date):
            pieces.append(local)
        else:
            offsets = np.linspace(
                0, len(local) - 1, int(maximum_per_date), dtype=np.int64
            )
            pieces.append(local[offsets])
    return np.concatenate(pieces).astype(np.int64, copy=False)


def _date_equal_weights(date_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int64)
    if not len(dates):
        return np.empty(0, dtype=np.float64)
    _, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(np.float64)


def _fit_standardizer(
    x: np.ndarray, sample_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    weight = np.asarray(sample_weight, dtype=np.float64)
    finite = np.isfinite(values)
    weighted = finite * weight[:, None]
    denominator = np.sum(weighted, axis=0)
    center = np.divide(
        np.sum(np.where(finite, values, 0.0) * weight[:, None], axis=0),
        denominator,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=denominator > 0,
    )
    deviations = np.where(finite, values - center, 0.0)
    variance = np.divide(
        np.sum(weighted * np.square(deviations), axis=0),
        denominator,
        out=np.ones(values.shape[1], dtype=np.float64),
        where=denominator > 0,
    )
    spread = np.sqrt(np.maximum(variance, 0.0))
    spread = np.where(np.isfinite(spread) & (spread > 1.0e-8), spread, 1.0)
    return center.astype(np.float32), spread.astype(np.float32)


def _apply_standardizer(
    x: np.ndarray, center: np.ndarray, spread: np.ndarray
) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    z = (values - center) / spread
    z[~np.isfinite(z)] = 0.0
    return np.clip(z, -12.0, 12.0).astype(np.float32, copy=False)


def _safe_t_fit(values: np.ndarray, *, fixed_location: float = 0.0) -> tuple[float, float]:
    residual = np.asarray(values, dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    if len(residual) < 100:
        raise ValueError("too_few_values_for_student_t")
    if len(residual) > 50_000:
        positions = np.linspace(0, len(residual) - 1, 50_000, dtype=np.int64)
        residual = residual[positions]
    try:
        df, _, scale = stats.t.fit(residual, floc=float(fixed_location))
    except Exception:  # noqa: BLE001
        # scipy's optimizer can expose several numeric failure types here.
        excess = float(max(stats.kurtosis(residual, fisher=True, bias=False), 0.0))
        df = 200.0 if excess <= 1.0e-8 else 4.0 + 6.0 / excess
        standard_deviation = float(np.std(residual, ddof=1))
        scale = standard_deviation * math.sqrt(max((df - 2.0) / df, 1.0e-4))
    return float(np.clip(df, 2.05, 200.0)), float(max(scale, 1.0e-8))


def _build_fold(
    panel: StockPanel,
    *,
    evaluation_year: int,
    horizon: int,
    target: str,
) -> dict[str, Any]:
    if int(horizon) not in HORIZONS:
        raise ValueError(f"unsupported_horizon:{horizon}")
    year_rows = panel.rows_for_year(evaluation_year)
    start_date_idx = int(panel.date_idx[year_rows[0]])
    dependency_days = int(horizon) + RETRY_DAYS
    maximum_train_date_idx = start_date_idx - dependency_days - 1
    valid = panel.target_valid(target)
    if target == "entry_action" or target.startswith("sell_action_"):
        horizon_column = HORIZONS.index(int(horizon)) + 1
        valid &= (
            (panel.flags[:, horizon_column] & FLAG_OUTCOME_WITHIN_CUTOFF) != 0
        )
    train_rows = np.flatnonzero(
        (panel.date_idx <= maximum_train_date_idx) & valid
    ).astype(np.int64, copy=False)
    evaluation_rows = year_rows[valid[year_rows]]
    if not len(train_rows) or not len(evaluation_rows):
        raise ValueError(f"empty_fold:{evaluation_year}:{horizon}:{target}")
    if int(panel.date_idx[train_rows[-1]]) + dependency_days >= start_date_idx:
        raise ValueError("purge_contract_failed")
    return {
        "evaluation_year": int(evaluation_year),
        "horizon": int(horizon),
        "target": str(target),
        "dependency_days": dependency_days,
        "oos_start_date_idx": start_date_idx,
        "maximum_train_signal_date_idx": maximum_train_date_idx,
        "train_rows": train_rows,
        "evaluation_rows": evaluation_rows,
    }


def _fit_distribution_models(
    panel: StockPanel,
    *,
    train_rows: np.ndarray,
    target_values: np.ndarray,
    feature_block: str,
    penalty: float,
    maximum_train_rows_per_date: int,
) -> tuple[dict[str, StudentTModel], dict[str, Any]]:
    feature_names = feature_names_for_block(panel.input_manifest, feature_block)
    feature_positions = _feature_positions(panel, feature_names)
    sampled_rows = _sample_rows_by_date(
        train_rows,
        panel.date_idx,
        maximum_per_date=maximum_train_rows_per_date,
    )
    y = np.asarray(target_values[sampled_rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("sampled_distribution_target_not_finite")
    weights = _date_equal_weights(panel.date_idx[sampled_rows])
    x = _feature_matrix(panel, sampled_rows, feature_positions)
    center, spread = _fit_standardizer(x, weights)
    z = _apply_standardizer(x, center, spread)
    effective_dates = float(np.sum(weights))
    alpha = float(penalty) * effective_dates
    mean_model = Ridge(alpha=alpha, fit_intercept=True)
    mean_model.fit(z, y, sample_weight=weights)
    predicted_mean = np.asarray(mean_model.predict(z), dtype=np.float64)
    residual = y - predicted_mean
    constant_df, constant_scale = _safe_t_fit(residual)

    baseline_floor = max(float(np.nanmedian(np.abs(y))) ** 2 * 1.0e-4, 1.0e-10)
    zero_scale_model = Ridge(alpha=alpha, fit_intercept=True)
    zero_scale_model.fit(
        z, np.log(np.square(y) + baseline_floor), sample_weight=weights
    )
    zero_raw_scale = np.sqrt(
        np.exp(np.clip(zero_scale_model.predict(z), -30.0, 10.0))
    )
    zero_hetero_df, zero_hetero_scale = _safe_t_fit(
        y / np.maximum(zero_raw_scale, 1.0e-8)
    )

    floor = max(float(np.nanmedian(np.abs(residual))) ** 2 * 1.0e-4, 1.0e-10)
    log_residual_square = np.log(np.square(residual) + floor)
    scale_model = Ridge(alpha=alpha, fit_intercept=True)
    scale_model.fit(z, log_residual_square, sample_weight=weights)
    raw_scale = np.sqrt(
        np.exp(np.clip(scale_model.predict(z), -30.0, 10.0))
    )
    standardized_residual = residual / np.maximum(raw_scale, 1.0e-8)
    hetero_df, hetero_scale = _safe_t_fit(standardized_residual)
    baseline_df, baseline_scale = _safe_t_fit(y)
    empty = np.empty(0, dtype=np.float32)
    models = {
        "zero_mean_student_t": StudentTModel(
            name="zero_mean_student_t",
            feature_names=(),
            feature_positions=np.empty(0, dtype=np.int64),
            center=empty,
            spread=empty,
            mean_model=None,
            scale_model=None,
            constant_location=0.0,
            df=baseline_df,
            residual_scale=baseline_scale,
        ),
        "zero_mean_feature_scale_student_t": StudentTModel(
            name="zero_mean_feature_scale_student_t",
            feature_names=feature_names,
            feature_positions=feature_positions,
            center=center,
            spread=spread,
            mean_model=None,
            scale_model=zero_scale_model,
            constant_location=0.0,
            df=zero_hetero_df,
            residual_scale=zero_hetero_scale,
        ),
        "ridge_student_t": StudentTModel(
            name="ridge_student_t",
            feature_names=feature_names,
            feature_positions=feature_positions,
            center=center,
            spread=spread,
            mean_model=mean_model,
            scale_model=None,
            constant_location=0.0,
            df=constant_df,
            residual_scale=constant_scale,
        ),
        "ridge_hetero_student_t": StudentTModel(
            name="ridge_hetero_student_t",
            feature_names=feature_names,
            feature_positions=feature_positions,
            center=center,
            spread=spread,
            mean_model=mean_model,
            scale_model=scale_model,
            constant_location=0.0,
            df=hetero_df,
            residual_scale=hetero_scale,
        ),
    }
    coefficients = np.asarray(mean_model.coef_, dtype=np.float64)
    top = np.argsort(np.abs(coefficients))[::-1][:20]
    scale_coefficients = np.asarray(scale_model.coef_, dtype=np.float64)
    scale_top = np.argsort(np.abs(scale_coefficients))[::-1][:20]
    fit_summary = {
        "full_train_row_count": len(train_rows),
        "sampled_train_row_count": len(sampled_rows),
        "sampled_train_date_count": int(np.unique(panel.date_idx[sampled_rows]).size),
        "maximum_train_rows_per_date": int(maximum_train_rows_per_date),
        "feature_block": feature_block,
        "feature_count": len(feature_names),
        "penalty": float(penalty),
        "effective_date_weight": effective_dates,
        "target_mean": float(np.average(y, weights=weights)),
        "target_std": float(np.sqrt(np.average(np.square(y), weights=weights))),
        "top_absolute_location_coefficients": [
            {
                "feature": feature_names[int(pos)],
                "coefficient": float(coefficients[int(pos)]),
            }
            for pos in top
        ],
        "top_absolute_scale_coefficients": [
            {
                "feature": feature_names[int(pos)],
                "coefficient": float(scale_coefficients[int(pos)]),
            }
            for pos in scale_top
        ],
        "student_t_parameters": {
            name: {"df": model.df, "residual_scale": model.residual_scale}
            for name, model in models.items()
        },
    }
    return models, fit_summary


def _predict_distribution_models(
    panel: StockPanel,
    rows: np.ndarray,
    models: Mapping[str, StudentTModel],
    *,
    batch_size: int = 50_000,
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    selected = np.asarray(rows, dtype=np.int64)
    result = {
        name: (
            np.empty(len(selected), dtype=np.float64),
            np.empty(len(selected), dtype=np.float64),
            float(model.df),
        )
        for name, model in models.items()
    }
    feature_model = next(
        (model for model in models.values() if len(model.feature_positions)), None
    )
    for start in range(0, len(selected), int(batch_size)):
        stop = min(start + int(batch_size), len(selected))
        local_rows = selected[start:stop]
        if feature_model is not None:
            x = _feature_matrix(panel, local_rows, feature_model.feature_positions)
            z = _apply_standardizer(
                x, feature_model.center, feature_model.spread
            )
        else:
            z = np.empty((len(local_rows), 0), dtype=np.float32)
        for name, model in models.items():
            location_store, scale_store, _ = result[name]
            if model.mean_model is None:
                location = np.full(
                    len(local_rows), model.constant_location, dtype=np.float64
                )
            else:
                location = np.asarray(model.mean_model.predict(z), dtype=np.float64)
            if model.scale_model is None:
                scale = np.full(
                    len(local_rows), model.residual_scale, dtype=np.float64
                )
            else:
                raw_scale = np.sqrt(
                    np.exp(
                        np.clip(
                            np.asarray(model.scale_model.predict(z), dtype=np.float64),
                            -30.0,
                            10.0,
                        )
                    )
                )
                scale = raw_scale * model.residual_scale
            location_store[start:stop] = location
            scale_store[start:stop] = np.maximum(scale, 1.0e-8)
    return result


def _crps_from_quantiles(
    values: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    df: float,
    *,
    quantile_count: int = 64,
    batch_size: int = 20_000,
) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    loc = np.asarray(location, dtype=np.float64)
    scl = np.maximum(np.asarray(scale, dtype=np.float64), 1.0e-10)
    probabilities = (
        np.arange(int(quantile_count), dtype=np.float64) + 0.5
    ) / float(quantile_count)
    standardized = stats.t.ppf(probabilities, float(df))
    weights = (
        2.0 * np.arange(int(quantile_count), dtype=np.float64)
        - float(quantile_count)
        + 1.0
    )
    result = np.empty(len(y), dtype=np.float64)
    for start in range(0, len(y), int(batch_size)):
        stop = min(start + int(batch_size), len(y))
        draws = loc[start:stop, None] + scl[start:stop, None] * standardized[None, :]
        first = np.mean(np.abs(draws - y[start:stop, None]), axis=1)
        pairwise = (
            2.0
            * np.sum(draws * weights[None, :], axis=1)
            / float(quantile_count * quantile_count)
        )
        result[start:stop] = first - 0.5 * pairwise
    return result


def _pinball(values: np.ndarray, quantiles: np.ndarray, probability: float) -> np.ndarray:
    error = np.asarray(values, dtype=np.float64) - np.asarray(
        quantiles, dtype=np.float64
    )
    return np.maximum(float(probability) * error, (float(probability) - 1.0) * error)


def _daily_score_frame(
    *,
    panel: StockPanel,
    rows: np.ndarray,
    values: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    df: float,
    model: str,
    target: str,
    horizon: int,
    feature_block: str,
    evaluation_year: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = np.asarray(values, dtype=np.float64)
    loc = np.asarray(location, dtype=np.float64)
    scl = np.maximum(np.asarray(scale, dtype=np.float64), 1.0e-10)
    log_score = stats.t.logpdf(y, float(df), loc=loc, scale=scl)
    pit = stats.t.cdf(y, float(df), loc=loc, scale=scl)
    crps = _crps_from_quantiles(y, loc, scl, float(df))
    q05 = loc + scl * stats.t.ppf(0.05, float(df))
    q25 = loc + scl * stats.t.ppf(0.25, float(df))
    q50 = loc
    q75 = loc + scl * stats.t.ppf(0.75, float(df))
    q95 = loc + scl * stats.t.ppf(0.95, float(df))
    raw = pd.DataFrame(
        {
            "date_idx": panel.date_idx[rows],
            "log_score": log_score,
            "crps": crps,
            "squared_error": np.square(y - loc),
            "pinball_05": _pinball(y, q05, 0.05),
            "pinball_50": _pinball(y, q50, 0.50),
            "pinball_95": _pinball(y, q95, 0.95),
            "coverage_50": ((y >= q25) & (y <= q75)).astype(np.float64),
            "coverage_90": ((y >= q05) & (y <= q95)).astype(np.float64),
            "pit": pit,
        }
    )
    daily = raw.groupby("date_idx", sort=True, as_index=False).mean()
    date_map = (
        panel.row_index[["date_idx", "trade_date"]]
        .drop_duplicates("date_idx")
        .set_index("date_idx")["trade_date"]
    )
    daily.insert(1, "trade_date", daily["date_idx"].map(date_map).astype(str))
    daily["evaluation_year"] = int(evaluation_year)
    daily["horizon"] = int(horizon)
    daily["target"] = str(target)
    daily["feature_block"] = str(feature_block)
    daily["model"] = str(model)
    metric = {
        "evaluation_year": int(evaluation_year),
        "horizon": int(horizon),
        "target": str(target),
        "feature_block": str(feature_block),
        "model": str(model),
        "row_count": len(y),
        "date_count": len(daily),
        "log_score": float(daily["log_score"].mean()),
        "crps": float(daily["crps"].mean()),
        "rmse": float(math.sqrt(daily["squared_error"].mean())),
        "pinball_05": float(daily["pinball_05"].mean()),
        "pinball_50": float(daily["pinball_50"].mean()),
        "pinball_95": float(daily["pinball_95"].mean()),
        "coverage_50": float(daily["coverage_50"].mean()),
        "coverage_90": float(daily["coverage_90"].mean()),
        "pit_mean": float(daily["pit"].mean()),
        "pit_centered_second_moment_date_equal": float(
            np.average(
                np.square(pit - 0.5),
                weights=_date_equal_weights(panel.date_idx[rows]),
            )
        ),
        "student_t_df": float(df),
    }
    return daily, metric


def _fit_binary_models(
    panel: StockPanel,
    *,
    train_rows: np.ndarray,
    state_values: np.ndarray,
    feature_block: str,
    maximum_train_rows_per_date: int,
    regularization_c: float,
) -> tuple[dict[str, BinaryModel], dict[str, Any]]:
    feature_names = feature_names_for_block(panel.input_manifest, feature_block)
    positions = _feature_positions(panel, feature_names)
    sampled_rows = _sample_rows_by_date(
        train_rows,
        panel.date_idx,
        maximum_per_date=maximum_train_rows_per_date,
    )
    y = np.asarray(state_values[sampled_rows], dtype=np.int8)
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("binary_state_requires_two_known_classes")
    weights = _date_equal_weights(panel.date_idx[sampled_rows])
    full_values = np.asarray(state_values[train_rows], dtype=np.int8)
    full_weights = _date_equal_weights(panel.date_idx[train_rows])
    probability = float(np.average(full_values, weights=full_weights))
    unique_train_dates = np.unique(panel.date_idx[train_rows])
    recent_dates = unique_train_dates[-min(252, len(unique_train_dates)) :]
    recent_mask = np.isin(panel.date_idx[train_rows], recent_dates)
    recent_values = full_values[recent_mask]
    recent_weights = _date_equal_weights(panel.date_idx[train_rows][recent_mask])
    recent_probability = float(np.average(recent_values, weights=recent_weights))
    x = _feature_matrix(panel, sampled_rows, positions)
    center, spread = _fit_standardizer(x, weights)
    z = _apply_standardizer(x, center, spread)
    classifier = LogisticRegression(
        C=float(regularization_c),
        solver="lbfgs",
        max_iter=300,
        tol=1.0e-6,
    )
    classifier.fit(z, y, sample_weight=weights)
    empty = np.empty(0, dtype=np.float32)
    models = {
        "constant_probability": BinaryModel(
            name="constant_probability",
            feature_names=(),
            feature_positions=np.empty(0, dtype=np.int64),
            center=empty,
            spread=empty,
            model=None,
            constant_probability=probability,
        ),
        "recent_252d_probability": BinaryModel(
            name="recent_252d_probability",
            feature_names=(),
            feature_positions=np.empty(0, dtype=np.int64),
            center=empty,
            spread=empty,
            model=None,
            constant_probability=recent_probability,
        ),
        "logistic_ridge": BinaryModel(
            name="logistic_ridge",
            feature_names=feature_names,
            feature_positions=positions,
            center=center,
            spread=spread,
            model=classifier,
            constant_probability=probability,
        ),
    }
    coefficients = np.asarray(classifier.coef_[0], dtype=np.float64)
    top = np.argsort(np.abs(coefficients))[::-1][:20]
    summary = {
        "full_train_row_count": len(train_rows),
        "sampled_train_row_count": len(sampled_rows),
        "sampled_train_date_count": int(np.unique(panel.date_idx[sampled_rows]).size),
        "feature_count": len(feature_names),
        "constant_probability": probability,
        "recent_252d_probability": recent_probability,
        "class_counts": {
            "0": int(np.sum(y == 0)),
            "1": int(np.sum(y == 1)),
        },
        "top_absolute_logit_coefficients": [
            {
                "feature": feature_names[int(pos)],
                "coefficient": float(coefficients[int(pos)]),
            }
            for pos in top
        ],
    }
    return models, summary


def _predict_binary_models(
    panel: StockPanel,
    rows: np.ndarray,
    models: Mapping[str, BinaryModel],
    *,
    batch_size: int = 50_000,
) -> dict[str, np.ndarray]:
    selected = np.asarray(rows, dtype=np.int64)
    result = {
        name: np.empty(len(selected), dtype=np.float64) for name in models
    }
    feature_model = next(
        (model for model in models.values() if len(model.feature_positions)), None
    )
    for start in range(0, len(selected), int(batch_size)):
        stop = min(start + int(batch_size), len(selected))
        local_rows = selected[start:stop]
        if feature_model is not None:
            x = _feature_matrix(panel, local_rows, feature_model.feature_positions)
            z = _apply_standardizer(x, feature_model.center, feature_model.spread)
        else:
            z = np.empty((len(local_rows), 0), dtype=np.float32)
        for name, model in models.items():
            if model.model is None:
                probability = np.full(
                    len(local_rows), model.constant_probability, dtype=np.float64
                )
            else:
                probability = np.asarray(
                    model.model.predict_proba(z)[:, 1], dtype=np.float64
                )
            result[name][start:stop] = np.clip(
                probability, 1.0e-8, 1.0 - 1.0e-8
            )
    return result


def _daily_binary_score_frame(
    *,
    panel: StockPanel,
    rows: np.ndarray,
    values: np.ndarray,
    probability: np.ndarray,
    model: str,
    target: str,
    horizon: int,
    feature_block: str,
    evaluation_year: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = np.asarray(values, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-8, 1.0 - 1.0e-8)
    log_score = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    raw = pd.DataFrame(
        {
            "date_idx": panel.date_idx[rows],
            "log_score": log_score,
            "brier": np.square(y - p),
            "observed_rate": y,
            "predicted_rate": p,
        }
    )
    daily = raw.groupby("date_idx", sort=True, as_index=False).mean()
    date_map = (
        panel.row_index[["date_idx", "trade_date"]]
        .drop_duplicates("date_idx")
        .set_index("date_idx")["trade_date"]
    )
    daily.insert(1, "trade_date", daily["date_idx"].map(date_map).astype(str))
    daily["evaluation_year"] = int(evaluation_year)
    daily["horizon"] = int(horizon)
    daily["target"] = str(target)
    daily["feature_block"] = str(feature_block)
    daily["model"] = str(model)
    metric = {
        "evaluation_year": int(evaluation_year),
        "horizon": int(horizon),
        "target": str(target),
        "feature_block": str(feature_block),
        "model": str(model),
        "row_count": len(y),
        "date_count": len(daily),
        "log_score": float(daily["log_score"].mean()),
        "brier": float(daily["brier"].mean()),
        "observed_rate": float(daily["observed_rate"].mean()),
        "predicted_rate": float(daily["predicted_rate"].mean()),
    }
    return daily, metric


def _hac_mean(values: np.ndarray, lag: int = 20) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    if len(series) < 3:
        return {
            "mean": float("nan"),
            "standard_error": float("nan"),
            "lcb_95": float("nan"),
            "ucb_95": float("nan"),
            "n": len(series),
        }
    centered = series - float(series.mean())
    bandwidth = min(int(lag), len(series) - 1)
    long_run_variance = float(np.mean(np.square(centered)))
    for offset in range(1, bandwidth + 1):
        covariance = float(np.mean(centered[offset:] * centered[:-offset]))
        long_run_variance += 2.0 * (1.0 - offset / (bandwidth + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / len(series))
    mean = float(series.mean())
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lcb_95": mean - 1.959963984540054 * standard_error,
        "ucb_95": mean + 1.959963984540054 * standard_error,
        "n": len(series),
    }


def _moving_block_means(
    values: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    if not len(series):
        return np.empty(0, dtype=np.float64)
    block = min(max(int(block_length), 1), len(series))
    rng = np.random.default_rng(int(seed))
    result = np.empty(int(repetitions), dtype=np.float64)
    maximum_start = len(series) - block
    blocks_needed = math.ceil(len(series) / block)
    for repetition in range(int(repetitions)):
        starts = rng.integers(0, maximum_start + 1, size=blocks_needed)
        sample = np.concatenate([series[start : start + block] for start in starts])[
            : len(series)
        ]
        result[repetition] = float(sample.mean())
    return result


def _block_interval(
    values: np.ndarray,
    *,
    block_length: int = 20,
    repetitions: int = 1_000,
    seed: int = 20260806,
) -> dict[str, float]:
    draws = _moving_block_means(
        values,
        block_length=block_length,
        repetitions=repetitions,
        seed=seed,
    )
    if not len(draws):
        return {"lcb_95": float("nan"), "ucb_95": float("nan")}
    return {
        "lcb_95": float(np.quantile(draws, 0.025)),
        "ucb_95": float(np.quantile(draws, 0.975)),
    }


def _reality_check(
    differences: np.ndarray,
    model_names: Sequence[str],
    *,
    block_length: int = 20,
    repetitions: int = 1_000,
    seed: int = 20260806,
) -> dict[str, Any]:
    """White-style moving-block Reality Check for a declared challenger family."""

    matrix = np.asarray(differences, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(model_names):
        raise ValueError("reality_check_shape_mismatch")
    valid = np.isfinite(matrix).all(axis=1)
    matrix = matrix[valid]
    if not len(matrix):
        return {
            "winner": "",
            "observed_maximum_mean": float("nan"),
            "p_value": float("nan"),
            "date_count": 0,
        }
    observed_means = matrix.mean(axis=0)
    observed = float(observed_means.max())
    centered = matrix - observed_means
    block = min(max(int(block_length), 1), len(centered))
    rng = np.random.default_rng(int(seed))
    maximum_start = len(centered) - block
    blocks_needed = math.ceil(len(centered) / block)
    bootstrap_maxima = np.empty(int(repetitions), dtype=np.float64)
    for repetition in range(int(repetitions)):
        starts = rng.integers(0, maximum_start + 1, size=blocks_needed)
        sample = np.concatenate(
            [centered[start : start + block] for start in starts], axis=0
        )[: len(centered)]
        bootstrap_maxima[repetition] = float(sample.mean(axis=0).max())
    winner = str(model_names[int(np.argmax(observed_means))])
    return {
        "winner": winner,
        "observed_maximum_mean": observed,
        "p_value": float(
            (1 + np.sum(bootstrap_maxima >= observed)) / (len(bootstrap_maxima) + 1)
        ),
        "date_count": len(centered),
        "block_length": int(block),
        "repetitions": int(repetitions),
    }


def _comparison_tables(
    daily: pd.DataFrame,
    *,
    baseline: str,
    score_column: str,
    lower_is_better_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()
    key_columns = ["evaluation_year", "horizon", "target", "feature_block"]
    comparisons: list[dict[str, Any]] = []
    realities: list[dict[str, Any]] = []
    for keys, frame in daily.groupby(key_columns, sort=True):
        baseline_frame = frame.loc[frame["model"] == baseline]
        if baseline_frame.empty:
            continue
        base = baseline_frame.set_index("date_idx")
        challenger_differences: list[np.ndarray] = []
        challenger_names: list[str] = []
        for model in sorted(set(frame["model"]) - {baseline}):
            candidate = frame.loc[frame["model"] == model].set_index("date_idx")
            joined = base.join(candidate, how="inner", lsuffix="_base", rsuffix="_model")
            if joined.empty:
                continue
            score_delta = (
                joined[f"{score_column}_model"].to_numpy(dtype=np.float64)
                - joined[f"{score_column}_base"].to_numpy(dtype=np.float64)
            )
            hac = _hac_mean(score_delta)
            interval = _block_interval(score_delta, seed=20260806 + int(keys[0]))
            record = {
                "evaluation_year": int(keys[0]),
                "horizon": int(keys[1]),
                "target": str(keys[2]),
                "feature_block": str(keys[3]),
                "baseline": baseline,
                "model": model,
                "metric": score_column,
                "date_count": len(joined),
                "mean_difference": hac["mean"],
                "hac_standard_error": hac["standard_error"],
                "hac_lcb_95": hac["lcb_95"],
                "hac_ucb_95": hac["ucb_95"],
                "block_lcb_95": interval["lcb_95"],
                "block_ucb_95": interval["ucb_95"],
            }
            for column in lower_is_better_columns:
                if (
                    f"{column}_base" in joined.columns
                    and f"{column}_model" in joined.columns
                ):
                    delta = (
                        joined[f"{column}_base"].to_numpy(dtype=np.float64)
                        - joined[f"{column}_model"].to_numpy(dtype=np.float64)
                    )
                    record[f"{column}_improvement"] = float(np.mean(delta))
            comparisons.append(record)
            challenger_differences.append(score_delta)
            challenger_names.append(str(model))
        if challenger_differences:
            matrix = np.column_stack(challenger_differences)
            reality = _reality_check(
                matrix,
                challenger_names,
                seed=20260806 + int(keys[0]) * 17 + int(keys[1]),
            )
            realities.append(
                {
                    "evaluation_year": int(keys[0]),
                    "horizon": int(keys[1]),
                    "target": str(keys[2]),
                    "feature_block": str(keys[3]),
                    "baseline": baseline,
                    "metric": score_column,
                    **reality,
                }
            )
    return pd.DataFrame(comparisons), pd.DataFrame(realities)


def _location_scale_audit(daily: pd.DataFrame) -> pd.DataFrame:
    """Two-by-two Shapley decomposition of location and feature-scale gains."""

    required = {
        "zero_mean_student_t",
        "zero_mean_feature_scale_student_t",
        "ridge_student_t",
        "ridge_hetero_student_t",
    }
    key_columns = ["evaluation_year", "horizon", "target", "feature_block"]
    rows: list[dict[str, Any]] = []
    for keys, frame in daily.groupby(key_columns, sort=True):
        pivot = frame.pivot(index="date_idx", columns="model", values="log_score")
        if not required.issubset(pivot.columns):
            continue
        pivot = pivot.dropna(subset=sorted(required))
        if pivot.empty:
            continue
        l0s0 = pivot["zero_mean_student_t"].to_numpy(dtype=np.float64)
        l0s1 = pivot["zero_mean_feature_scale_student_t"].to_numpy(dtype=np.float64)
        l1s0 = pivot["ridge_student_t"].to_numpy(dtype=np.float64)
        l1s1 = pivot["ridge_hetero_student_t"].to_numpy(dtype=np.float64)
        total = l1s1 - l0s0
        location_constant = l1s0 - l0s0
        location_feature_scale = l1s1 - l0s1
        scale_zero_mean = l0s1 - l0s0
        scale_ridge_mean = l1s1 - l1s0
        location_shapley = 0.5 * (location_constant + location_feature_scale)
        scale_shapley = 0.5 * (scale_zero_mean + scale_ridge_mean)
        total_mean = float(np.mean(total))
        location_mean = float(np.mean(location_shapley))
        scale_mean = float(np.mean(scale_shapley))
        location_hac = _hac_mean(location_feature_scale)
        location_block = _block_interval(
            location_feature_scale,
            seed=20260806 + int(keys[0]) + int(keys[1]),
        )
        rows.append(
            {
                "evaluation_year": int(keys[0]),
                "horizon": int(keys[1]),
                "target": str(keys[2]),
                "feature_block": str(keys[3]),
                "date_count": len(pivot),
                "total_log_score_gain": total_mean,
                "location_gain_constant_scale": float(np.mean(location_constant)),
                "location_gain_feature_scale": float(np.mean(location_feature_scale)),
                "scale_gain_zero_mean": float(np.mean(scale_zero_mean)),
                "scale_gain_ridge_mean": float(np.mean(scale_ridge_mean)),
                "location_shapley_gain": location_mean,
                "scale_shapley_gain": scale_mean,
                "location_share_of_total": (
                    location_mean / total_mean if abs(total_mean) > 1.0e-12 else np.nan
                ),
                "scale_share_of_total": (
                    scale_mean / total_mean if abs(total_mean) > 1.0e-12 else np.nan
                ),
                "conditional_location_hac_lcb_95": location_hac["lcb_95"],
                "conditional_location_hac_ucb_95": location_hac["ucb_95"],
                "conditional_location_block_lcb_95": location_block["lcb_95"],
                "conditional_location_block_ucb_95": location_block["ucb_95"],
            }
        )
    return pd.DataFrame(rows)


def _run_continuous_target(
    *,
    panel: StockPanel,
    evaluation_year: int,
    horizon: int,
    feature_block: str,
    penalty: float,
    maximum_train_rows_per_date: int,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    target = f"residual_log_return_{int(horizon)}"
    fold = _build_fold(
        panel,
        evaluation_year=evaluation_year,
        horizon=horizon,
        target=target,
    )
    values = panel.target(target)
    models, fit_summary = _fit_distribution_models(
        panel,
        train_rows=fold["train_rows"],
        target_values=values,
        feature_block=feature_block,
        penalty=penalty,
        maximum_train_rows_per_date=maximum_train_rows_per_date,
    )
    predictions = _predict_distribution_models(
        panel,
        fold["evaluation_rows"],
        models,
    )
    daily_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for name, (location, scale, df) in predictions.items():
        daily, metric = _daily_score_frame(
            panel=panel,
            rows=fold["evaluation_rows"],
            values=values[fold["evaluation_rows"]],
            location=location,
            scale=scale,
            df=df,
            model=name,
            target=target,
            horizon=horizon,
            feature_block=feature_block,
            evaluation_year=evaluation_year,
        )
        daily_frames.append(daily)
        metrics.append(metric)
    fit_summary["fold"] = {
        key: value
        for key, value in fold.items()
        if key not in {"train_rows", "evaluation_rows"}
    }
    return daily_frames, metrics, fit_summary


def _run_state_target(
    *,
    panel: StockPanel,
    evaluation_year: int,
    horizon: int,
    feature_block: str,
    maximum_train_rows_per_date: int,
    regularization_c: float,
    target: str,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], dict[str, Any]]:
    fold = _build_fold(
        panel,
        evaluation_year=evaluation_year,
        horizon=horizon,
        target=target,
    )
    values = panel.state(target).astype(np.int8)
    models, fit_summary = _fit_binary_models(
        panel,
        train_rows=fold["train_rows"],
        state_values=values,
        feature_block=feature_block,
        maximum_train_rows_per_date=maximum_train_rows_per_date,
        regularization_c=regularization_c,
    )
    predictions = _predict_binary_models(panel, fold["evaluation_rows"], models)
    daily_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for name, probability in predictions.items():
        daily, metric = _daily_binary_score_frame(
            panel=panel,
            rows=fold["evaluation_rows"],
            values=values[fold["evaluation_rows"]],
            probability=probability,
            model=name,
            target=target,
            horizon=horizon,
            feature_block=feature_block,
            evaluation_year=evaluation_year,
        )
        daily_frames.append(daily)
        metrics.append(metric)
    fit_summary["fold"] = {
        key: value
        for key, value in fold.items()
        if key not in {"train_rows", "evaluation_rows"}
    }
    return daily_frames, metrics, fit_summary


def run_experiment(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    input_manifest_path: str | Path = DEFAULT_INPUT_MANIFEST,
    pack_manifest_path: str | Path = DEFAULT_PACK_MANIFEST,
    membership_root: str | Path = DEFAULT_MEMBERSHIP_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    evaluation_years: Sequence[int] = DEVELOPMENT_YEARS,
    horizons: Sequence[int] = PRIMARY_HORIZONS,
    feature_blocks: Sequence[str] = ("core",),
    maximum_train_rows_per_date: int = 64,
    penalty: float = 1.0e-3,
    regularization_c: float = 1.0,
    run_id: str = "development",
) -> dict[str, Any]:
    """Run frozen rolling forecast research; no portfolio is optimized here."""

    output_root = _resolve_path(output_root)
    label_manifest = prepare_labels(
        study_path=study_path,
        input_manifest_path=input_manifest_path,
        pack_manifest_path=pack_manifest_path,
        membership_root=membership_root,
        output_root=output_root,
    )
    panel = load_panel(
        input_manifest_path=input_manifest_path,
        label_manifest_path=output_root / "manifest.json",
    )
    requested_years = tuple(int(year) for year in evaluation_years)
    requested_horizons = tuple(int(h) for h in horizons)
    requested_blocks = tuple(str(block) for block in feature_blocks)
    if not requested_years or not requested_horizons or not requested_blocks:
        raise ValueError("experiment_matrix_empty")
    if any(h not in HORIZONS for h in requested_horizons):
        raise ValueError("experiment_horizon_not_in_contract")
    for block in requested_blocks:
        feature_names_for_block(panel.input_manifest, block)
    run_root = output_root / "experiments" / str(run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    continuous_daily: list[pd.DataFrame] = []
    continuous_metrics: list[dict[str, Any]] = []
    state_daily: list[pd.DataFrame] = []
    state_metrics: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []

    for evaluation_year in requested_years:
        for horizon in requested_horizons:
            for feature_block in requested_blocks:
                daily, metrics, summary = _run_continuous_target(
                    panel=panel,
                    evaluation_year=evaluation_year,
                    horizon=horizon,
                    feature_block=feature_block,
                    penalty=penalty,
                    maximum_train_rows_per_date=maximum_train_rows_per_date,
                )
                continuous_daily.extend(daily)
                continuous_metrics.extend(metrics)
                summary["head"] = "residual_return_distribution"
                fit_summaries.append(summary)
                for target in ("entry_action", f"sell_action_{int(horizon)}"):
                    daily, metrics, summary = _run_state_target(
                        panel=panel,
                        evaluation_year=evaluation_year,
                        horizon=horizon,
                        feature_block=feature_block,
                        maximum_train_rows_per_date=maximum_train_rows_per_date,
                        regularization_c=regularization_c,
                        target=target,
                    )
                    state_daily.extend(daily)
                    state_metrics.extend(metrics)
                    summary["head"] = target
                    fit_summaries.append(summary)
                print(
                    json.dumps(
                        {
                            "event": "stock_distribution_fold_complete",
                            "evaluation_year": evaluation_year,
                            "horizon": horizon,
                            "feature_block": feature_block,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    continuous_daily_frame = pd.concat(continuous_daily, ignore_index=True)
    continuous_metrics_frame = pd.DataFrame(continuous_metrics)
    state_daily_frame = pd.concat(state_daily, ignore_index=True)
    state_metrics_frame = pd.DataFrame(state_metrics)
    continuous_daily_path = run_root / "continuous_daily_scores.parquet"
    continuous_metrics_path = run_root / "continuous_metrics.parquet"
    state_daily_path = run_root / "state_daily_scores.parquet"
    state_metrics_path = run_root / "state_metrics.parquet"
    continuous_daily_frame.to_parquet(continuous_daily_path, index=False)
    continuous_metrics_frame.to_parquet(continuous_metrics_path, index=False)
    state_daily_frame.to_parquet(state_daily_path, index=False)
    state_metrics_frame.to_parquet(state_metrics_path, index=False)
    continuous_comparisons, continuous_reality = _comparison_tables(
        continuous_daily_frame,
        baseline="zero_mean_student_t",
        score_column="log_score",
        lower_is_better_columns=("crps", "squared_error"),
    )
    state_comparisons, state_reality = _comparison_tables(
        state_daily_frame,
        baseline="constant_probability",
        score_column="log_score",
        lower_is_better_columns=("brier",),
    )
    continuous_comparisons_path = run_root / "continuous_comparisons.parquet"
    continuous_reality_path = run_root / "continuous_reality_check.parquet"
    location_scale_path = run_root / "location_scale_audit.parquet"
    state_comparisons_path = run_root / "state_comparisons.parquet"
    state_reality_path = run_root / "state_reality_check.parquet"
    continuous_comparisons.to_parquet(continuous_comparisons_path, index=False)
    continuous_reality.to_parquet(continuous_reality_path, index=False)
    _location_scale_audit(continuous_daily_frame).to_parquet(
        location_scale_path, index=False
    )
    state_comparisons.to_parquet(state_comparisons_path, index=False)
    state_reality.to_parquet(state_reality_path, index=False)
    _write_json(
        run_root / "fit_summaries.json",
        {"fits": fit_summaries},
    )
    dev_only = set(requested_years).issubset(set(DEVELOPMENT_YEARS))
    summary = {
        "status": "completed_forecast_research",
        "study_id": STUDY_ID,
        "run_id": str(run_id),
        "label_manifest": str((output_root / "manifest.json").resolve()),
        "label_experiment_fingerprint": label_manifest["experiment_fingerprint"],
        "input_fingerprint": EXPECTED_INPUT_FINGERPRINT,
        "evaluation_years": list(requested_years),
        "horizons": list(requested_horizons),
        "feature_blocks": list(requested_blocks),
        "maximum_train_rows_per_date": int(maximum_train_rows_per_date),
        "ridge_penalty": float(penalty),
        "logistic_regularization_c": float(regularization_c),
        "development_only": bool(dev_only),
        "retrospective_oos_years_used": sorted(
            set(requested_years).intersection(RETROSPECTIVE_OOS_YEARS)
        ),
        "model_selection_performed": False,
        "portfolio_selection_performed": False,
        "execution_search_performed": False,
        "profit_claim_allowed": False,
        "forecast_gate": {
            "eligible_for_final_gate": False,
            "reason": (
                "This run is a fixed baseline comparison only. It omits the "
                "joint-path, LightGBM, and selection-nested challenger families "
                "required before any final forecast or utility gate."
            ),
            "required_final_condition": (
                "selection-adjusted 95% lower confidence bound of net utility "
                "improvement is positive on an unused confirmation period"
            ),
        },
        "files": {
            "continuous_daily_scores": _file_record(continuous_daily_path),
            "continuous_metrics": _file_record(continuous_metrics_path),
            "state_daily_scores": _file_record(state_daily_path),
            "state_metrics": _file_record(state_metrics_path),
            "continuous_comparisons": _file_record(continuous_comparisons_path),
            "continuous_reality_check": _file_record(continuous_reality_path),
            "location_scale_audit": _file_record(location_scale_path),
            "state_comparisons": _file_record(state_comparisons_path),
            "state_reality_check": _file_record(state_reality_path),
            "fit_summaries": _file_record(run_root / "fit_summaries.json"),
        },
    }
    _write_json(run_root / "summary.json", summary)
    return summary


def run_preflight(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """A bounded 2022 D10 core-block run before the wider fixed matrix."""

    return run_experiment(
        study_path=study_path,
        output_root=output_root,
        evaluation_years=(2022,),
        horizons=(10,),
        feature_blocks=("core",),
        maximum_train_rows_per_date=64,
        penalty=1.0e-3,
        regularization_c=1.0,
        run_id="preflight_2022_h10_core",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seq100 stock executable conditional-distribution research."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--pack-manifest", default=str(DEFAULT_PACK_MANIFEST))
    parser.add_argument("--membership-root", default=str(DEFAULT_MEMBERSHIP_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-development", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare:
        result = prepare_labels(
            study_path=args.study,
            input_manifest_path=args.input_manifest,
            pack_manifest_path=args.pack_manifest,
            membership_root=args.membership_root,
            output_root=args.output_root,
        )
    elif args.preflight:
        result = run_preflight(study_path=args.study, output_root=args.output_root)
    elif args.run_development:
        result = run_experiment(
            study_path=args.study,
            input_manifest_path=args.input_manifest,
            pack_manifest_path=args.pack_manifest,
            membership_root=args.membership_root,
            output_root=args.output_root,
            evaluation_years=DEVELOPMENT_YEARS,
            horizons=PRIMARY_HORIZONS,
            feature_blocks=("core",),
            run_id="development_core",
        )
    else:
        study = load_study(args.study)
        result = {
            "status": "ready",
            "study_id": study["study_id"],
            "input_manifest": str(_resolve_path(args.input_manifest)),
            "output_root": str(_resolve_path(args.output_root)),
            "horizons": list(HORIZONS),
            "primary_horizons": list(PRIMARY_HORIZONS),
            "development_years": list(DEVELOPMENT_YEARS),
            "retrospective_oos_years": list(RETROSPECTIVE_OOS_YEARS),
            "feature_blocks": {
                name: len(feature_names_for_block(
                    _load_input_contract(_resolve_path(args.input_manifest))[0], name
                ))
                for name in ("core", "core_minute", "core_minute_pit")
            },
            "training_performed": False,
            "model_selection_performed": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
