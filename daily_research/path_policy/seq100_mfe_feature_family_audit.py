from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import special, stats

from daily_research.path_policy import seq100_path_label_learnability as base
from daily_research.path_policy import seq100_signal_quality as signal_quality


WORKSPACE_ROOT = base.WORKSPACE_ROOT
STUDY_ID = "seq100_mfe_feature_family_audit_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_mfe_feature_family_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_mfe_feature_family_audit_v1"
)
TARGET_HORIZONS = (10, 20)
FOLD_YEARS = (2023, 2024, 2025)
LONG_DAILY_WINDOWS = (120, 180, 252)
BREAKOUT_WINDOWS = (20, 60, 120, 252)
BREAKOUT_EVENT_WINDOWS = (20, 60)
PERIOD_WINDOWS = {
    "week": (4, 13, 26, 52),
    "month": (3, 6, 12, 24),
}
REAL_FAMILIES = (
    "recent_kline_sequence",
    "breakout_retest_levels",
    "long_daily_context",
    "completed_week_month_context",
    "traditional_indicators",
    "confirmed_swing_structure",
    "turnover_cost_proxy",
)
CONTROL_FAMILY = "deterministic_noise"
ALL_FAMILIES = (*REAL_FAMILIES, CONTROL_FAMILY)
PREPARED_FEATURE_SCHEMA = "seq100_mfe_feature_family_features/v1"
TASK_RESULT_SCHEMA = "seq100_mfe_feature_family_task_result/v1"
SUMMARY_SCHEMA = "seq100_mfe_feature_family_summary/v1"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = WORKSPACE_ROOT / value
    return value.resolve()


def _record_for_file(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
        "size": int(path.stat().st_size),
    }


def _recent_catalog() -> tuple[str, ...]:
    channels = (
        "open_gap",
        "high_ret_prev_close",
        "low_ret_prev_close",
        "close_ret_prev_close",
        "intraday_range",
        "signed_body_to_range",
        "upper_shadow_to_range",
        "lower_shadow_to_range",
        "close_location",
        "log_amount_surprise_20",
        "relative_turnover_20",
    )
    names = [
        f"lag{lag:02d}__{channel}"
        for channel in channels
        for lag in range(1, 10)
    ]
    names.append("lag00__signed_body_to_range")
    return tuple(names)


def _breakout_catalog() -> tuple[str, ...]:
    level_metrics = (
        "close_vs_prior_high",
        "high_vs_prior_high",
        "low_vs_prior_high",
        "close_vs_prior_low",
        "high_vs_prior_low",
        "low_vs_prior_low",
        "prior_high_age_fraction",
        "prior_low_age_fraction",
    )
    names = [
        f"w{window:03d}__{metric}"
        for window in BREAKOUT_WINDOWS
        for metric in level_metrics
    ]
    event_metrics = (
        "last_breakout_age_fraction",
        "close_vs_last_breakout_level",
        "low_vs_last_breakout_level",
        "last_breakout_relative_turnover",
    )
    names.extend(
        f"w{window:03d}__{metric}"
        for window in BREAKOUT_EVENT_WINDOWS
        for metric in event_metrics
    )
    return tuple(names)


def _long_daily_catalog() -> tuple[str, ...]:
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
    names = [f"return_{window}d" for window in LONG_DAILY_WINDOWS]
    names.extend(
        f"{metric}_{window}d"
        for window in LONG_DAILY_WINDOWS
        for metric in per_window
    )
    return tuple(names)


def _completed_period_catalog() -> tuple[str, ...]:
    metrics = (
        "return",
        "ma_distance",
        "volatility",
        "atr",
        "trend_slope",
        "price_range_position",
    )
    return tuple(
        f"completed_{period}__{metric}_{window}p"
        for period, windows in PERIOD_WINDOWS.items()
        for window in windows
        for metric in metrics
    )


def _traditional_catalog() -> tuple[str, ...]:
    names: list[str] = []
    for er_period in (10, 20):
        names.extend(
            (
                f"kama_{er_period}_2_30__distance",
                f"kama_{er_period}_2_30__slope_1d",
                f"kama_{er_period}_2_30__slope_5d",
            )
        )
    names.extend(
        (
            "macd_12_26_9__dif_norm",
            "macd_12_26_9__signal_norm",
            "macd_12_26_9__hist_norm",
            "macd_12_26_9__hist_delta_1d",
            "macd_12_26_9__hist_delta_3d",
            "macd_12_26_9__bull_cross_age_fraction",
        )
    )
    for window in (9, 21):
        names.extend(
            (
                f"kdj_{window}_3_3__k",
                f"kdj_{window}_3_3__d",
                f"kdj_{window}_3_3__j",
                f"kdj_{window}_3_3__k_minus_d",
                f"kdj_{window}_3_3__k_delta_1d",
                f"kdj_{window}_3_3__j_delta_1d",
            )
        )
    names.extend(("rsi_6", "rsi_14", "rsi_24"))
    return tuple(names)


def _swing_catalog() -> tuple[str, ...]:
    metrics = (
        "close_vs_last_pivot_high",
        "close_vs_last_pivot_low",
        "last_pivot_high_age_fraction",
        "last_pivot_low_age_fraction",
        "last_pivot_high_change",
        "last_pivot_low_change",
        "confirmed_high_count_20",
        "confirmed_low_count_20",
        "confirmed_high_count_60",
        "confirmed_low_count_60",
        "swing_position",
        "last_swing_direction",
    )
    return tuple(
        f"radius{radius}__{metric}"
        for radius in (1, 2)
        for metric in metrics
    )


def feature_catalog() -> dict[str, tuple[str, ...]]:
    return {
        "recent_kline_sequence": _recent_catalog(),
        "breakout_retest_levels": _breakout_catalog(),
        "long_daily_context": _long_daily_catalog(),
        "completed_week_month_context": _completed_period_catalog(),
        "traditional_indicators": _traditional_catalog(),
        "confirmed_swing_structure": _swing_catalog(),
        "turnover_cost_proxy": (
            "survivor_log_close_minus_mean_cost",
            "survivor_log_cost_std",
            "survivor_profit_fraction_normal",
            "survivor_mass",
            "survivor_mean_age_days",
        ),
        CONTROL_FAMILY: tuple(f"noise_{index:02d}" for index in range(32)),
    }


def feature_catalog_payload() -> dict[str, Any]:
    catalog = feature_catalog()
    return {
        "schema": "seq100_mfe_feature_family_catalog/v1",
        "families": [
            {
                "family": family,
                "role": "negative_control" if family == CONTROL_FAMILY else "candidate",
                "feature_count": len(catalog[family]),
                "features": list(catalog[family]),
            }
            for family in ALL_FAMILIES
        ],
    }


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError("unexpected feature-family study id")
    protocol = dict(payload.get("folds", {}) or {})
    if tuple(int(value) for value in protocol.get("fold_years", [])) != FOLD_YEARS:
        raise ValueError("feature-family folds changed")
    if tuple(int(value) for value in protocol.get("primary_horizons", [])) != TARGET_HORIZONS:
        raise ValueError("feature-family target horizons changed")
    groups = dict(payload.get("feature_groups", {}) or {})
    if tuple(groups.get("candidates", [])) != REAL_FAMILIES:
        raise ValueError("candidate feature family order changed")
    if groups.get("negative_control") != CONTROL_FAMILY:
        raise ValueError("negative control changed")
    expected_counts = {name: len(values) for name, values in feature_catalog().items()}
    declared_counts = {
        str(name): int(value)
        for name, value in dict(groups.get("feature_counts", {}) or {}).items()
    }
    if declared_counts != expected_counts:
        raise ValueError("feature catalog does not match the configured groups")
    return payload


def _load_source_inputs(study: Mapping[str, Any]) -> base.LearnabilityInputs:
    return base.LearnabilityInputs(study)


def _verify_source_panels(
    study: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    records = dict(study["data"]["source_feature_panels"])
    channels = dict(pack["feature_channels"])
    verified: dict[str, dict[str, Any]] = {}
    for name in ("daily_raw", "turnover"):
        record = dict(records[name])
        channel = dict(channels[name])
        path = _resolve(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        if path != _resolve(channel["path"]):
            raise ValueError(f"source feature panel path changed: {name}")
        if tuple(int(value) for value in record["shape"]) != tuple(
            int(value) for value in channel["shape"]
        ):
            raise ValueError(f"source feature panel shape changed: {name}")
        if str(record["dtype"]) != str(channel.get("dtype", "float32")):
            raise ValueError(f"source feature panel dtype changed: {name}")
        verified[name] = {
            "path": record["path"],
            "size": int(path.stat().st_size),
        }
    return verified


def _load_pack(study: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    records = dict(study["data"])
    pack_path = _resolve(records["source_pack_manifest"]["path"])
    if not pack_path.is_file():
        raise FileNotFoundError(pack_path)
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    label_path = _resolve(records["label_manifest"]["path"])
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    label_manifest = json.loads(label_path.read_text(encoding="utf-8"))
    if _resolve(label_manifest["source"]["pack_manifest"]) != pack_path:
        raise ValueError("label source and feature source pack differ")
    return pack, {"source_feature_panels": _verify_source_panels(study, pack)}


def _model_task_plan() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for horizon in TARGET_HORIZONS:
        for year in FOLD_YEARS:
            tasks.append(
                {
                    "task_id": f"mfe{horizon}_{year}_baseline_tuning",
                    "stage": "iteration_tuning",
                    "variant": "baseline",
                    "horizon": horizon,
                    "fold_year": year,
                    "inner_validation_year": year - 1,
                }
            )
            for variant in ("baseline", *ALL_FAMILIES):
                tasks.append(
                    {
                        "task_id": f"mfe{horizon}_{year}_{variant}_outer",
                        "stage": "outer_evaluation",
                        "variant": variant,
                        "horizon": horizon,
                        "fold_year": year,
                    }
                )
    return tasks


def _load_validated_inputs(
    study: Mapping[str, Any],
) -> tuple[base.LearnabilityInputs, dict[str, Any]]:
    import lightgbm as lgb

    pack, _source = _load_pack(study)
    inputs = _load_source_inputs(study)
    if inputs.candidate_count != int(study["data"]["candidate_count"]):
        raise ValueError("candidate count differs from the configuration")
    expected_library = str(study["model"]["library"])
    if expected_library != f"lightgbm_{lgb.__version__}":
        raise ValueError("LightGBM runtime version differs from the configuration")
    if pack["data_semantics"]["price_adjustment"] != "back_adjust":
        raise ValueError("feature source must use a back-adjusted panel")
    maximum_outcome_date = str(study["folds"]["maximum_outcome_date"])
    if str(inputs.date_values[inputs.cutoff_date_idx]) != maximum_outcome_date:
        raise ValueError("input cutoff differs from the configured outcome boundary")
    catalog = feature_catalog_payload()
    names = [name for family in catalog["families"] for name in family["features"]]
    if len(names) != len(set(names)):
        raise ValueError("new feature names must be globally unique")
    if any(name in set(inputs.feature_names) for name in names):
        raise ValueError("new feature catalog duplicates a base feature name")
    return inputs, pack


@dataclass(frozen=True)
class PreparedFamily:
    family: str
    path: Path
    shape: tuple[int, int]
    feature_names: tuple[str, ...]
    manifest: dict[str, Any]

    def open(self) -> np.memmap:
        return np.memmap(
            self.path,
            mode="r",
            dtype=np.float32,
            shape=self.shape,
        )


class _FamilyWriter:
    def __init__(
        self,
        *,
        path: Path,
        feature_names: Sequence[str],
        candidate_date_idx: np.ndarray,
        candidate_symbol_idx: np.ndarray,
        cutoff_date_idx: int,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_names = tuple(str(value) for value in feature_names)
        self.index = {name: idx for idx, name in enumerate(self.feature_names)}
        if len(self.index) != len(self.feature_names):
            raise ValueError("family feature names must be unique")
        self.date_idx = np.asarray(candidate_date_idx, dtype=np.int32)
        self.symbol_idx = np.asarray(candidate_symbol_idx, dtype=np.int32)
        self.cutoff_date_idx = int(cutoff_date_idx)
        self.valid_count = int(
            np.searchsorted(self.date_idx, self.cutoff_date_idx, side="right")
        )
        self.values = np.memmap(
            self.path,
            mode="w+",
            dtype=np.float32,
            shape=(len(self.feature_names), len(self.date_idx)),
        )
        self.written: set[str] = set()

    def panel(self, name: str, values: np.ndarray) -> None:
        key = str(name)
        if key not in self.index:
            raise KeyError(f"unknown family feature: {key}")
        panel = np.asarray(values, dtype=np.float32)
        expected_rows = self.cutoff_date_idx + 1
        if panel.ndim != 2 or panel.shape[0] != expected_rows:
            raise ValueError(
                f"feature panel {key} has shape {panel.shape}; "
                f"expected ({expected_rows}, symbols)"
            )
        row = np.full(len(self.date_idx), np.nan, dtype=np.float32)
        dates = self.date_idx[: self.valid_count]
        symbols = self.symbol_idx[: self.valid_count]
        row[: self.valid_count] = panel[dates, symbols]
        self.values[self.index[key], :] = row
        self.written.add(key)

    def candidate(self, name: str, values: np.ndarray) -> None:
        key = str(name)
        if key not in self.index:
            raise KeyError(f"unknown family feature: {key}")
        row = np.asarray(values, dtype=np.float32)
        if row.shape != self.date_idx.shape:
            raise ValueError(f"candidate feature {key} is not row aligned")
        row = row.copy()
        row[self.valid_count :] = np.nan
        self.values[self.index[key], :] = row
        self.written.add(key)

    def finish(self) -> None:
        missing = sorted(set(self.feature_names) - self.written)
        if missing:
            raise ValueError(f"family features were not written: {missing}")
        self.values.flush()


def _open_panel(meta: Mapping[str, Any], *, cutoff_date_idx: int) -> np.memmap:
    shape = tuple(int(value) for value in meta["shape"])
    if shape[0] <= int(cutoff_date_idx):
        raise ValueError("source panel does not reach the 2025 cutoff")
    return np.memmap(
        _resolve(str(meta["path"])),
        mode="r",
        dtype=np.float32,
        shape=shape,
    )


def _shift_panel(values: np.ndarray, lag: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    amount = int(lag)
    output = np.full(source.shape, np.nan, dtype=np.float32)
    if amount == 0:
        output[:] = source
    elif 0 < amount < source.shape[0]:
        output[amount:] = source[:-amount]
    return output


def _one_day_delta(values: np.ndarray, lag: int = 1) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    output = np.full(source.shape, np.nan, dtype=np.float32)
    if int(lag) < source.shape[0]:
        output[int(lag) :] = source[int(lag) :] - source[: -int(lag)]
    return output


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return signal_quality._safe_divide(numerator, denominator)


def _raw_views(
    pack: Mapping[str, Any],
    cutoff_date_idx: int,
) -> tuple[np.memmap, np.memmap, dict[str, int], dict[str, int]]:
    channels = dict(pack["feature_channels"])
    raw_meta = dict(channels["daily_raw"])
    turnover_meta = dict(channels["turnover"])
    raw = _open_panel(raw_meta, cutoff_date_idx=cutoff_date_idx)
    turnover = _open_panel(turnover_meta, cutoff_date_idx=cutoff_date_idx)
    raw_columns = {str(name): idx for idx, name in enumerate(raw_meta["columns"])}
    turnover_columns = {
        str(name): idx for idx, name in enumerate(turnover_meta["columns"])
    }
    return raw, turnover, raw_columns, turnover_columns


def _generate_recent(
    writer: _FamilyWriter,
    *,
    raw: np.ndarray,
    turnover: np.ndarray,
    raw_columns: Mapping[str, int],
    turnover_columns: Mapping[str, int],
) -> None:
    stop = writer.cutoff_date_idx + 1
    open_price = np.asarray(raw[:stop, :, raw_columns["open"]], dtype=np.float32)
    high = np.asarray(raw[:stop, :, raw_columns["high"]], dtype=np.float32)
    low = np.asarray(raw[:stop, :, raw_columns["low"]], dtype=np.float32)
    close = np.asarray(raw[:stop, :, raw_columns["close"]], dtype=np.float32)
    amount = np.asarray(raw[:stop, :, raw_columns["amount"]], dtype=np.float32)
    price_range = high - low
    log_amount = np.log1p(np.maximum(amount, 0.0)).astype(np.float32)
    amount_mean = signal_quality._rolling_mean(log_amount, 20)
    components: list[tuple[str, np.ndarray]] = [
        (
            "open_gap",
            np.asarray(
                raw[:stop, :, raw_columns["open_ret_prev_close"]],
                dtype=np.float32,
            ),
        ),
        (
            "high_ret_prev_close",
            np.asarray(
                raw[:stop, :, raw_columns["high_ret_prev_close"]],
                dtype=np.float32,
            ),
        ),
        (
            "low_ret_prev_close",
            np.asarray(
                raw[:stop, :, raw_columns["low_ret_prev_close"]],
                dtype=np.float32,
            ),
        ),
        (
            "close_ret_prev_close",
            np.asarray(
                raw[:stop, :, raw_columns["close_ret_prev_close"]],
                dtype=np.float32,
            ),
        ),
        (
            "intraday_range",
            np.asarray(
                raw[:stop, :, raw_columns["intraday_range_raw"]],
                dtype=np.float32,
            ),
        ),
        ("signed_body_to_range", _safe_ratio(close - open_price, price_range)),
        (
            "upper_shadow_to_range",
            _safe_ratio(high - np.maximum(open_price, close), price_range),
        ),
        (
            "lower_shadow_to_range",
            _safe_ratio(np.minimum(open_price, close) - low, price_range),
        ),
        ("close_location", _safe_ratio(close - low, price_range)),
        ("log_amount_surprise_20", log_amount - amount_mean),
        (
            "relative_turnover_20",
            np.asarray(
                turnover[:stop, :, turnover_columns["relative_turnover_20"]],
                dtype=np.float32,
            ),
        ),
    ]
    for channel, panel in components:
        for lag in range(1, 10):
            writer.panel(f"lag{lag:02d}__{channel}", _shift_panel(panel, lag))
    writer.panel("lag00__signed_body_to_range", components[5][1])


def _last_breakout_panels(
    *,
    close: np.ndarray,
    low: np.ndarray,
    prior_high: np.ndarray,
    relative_turnover: np.ndarray,
    age_scale: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates, symbols = close.shape
    age = np.full(close.shape, np.nan, dtype=np.float32)
    close_gap = np.full(close.shape, np.nan, dtype=np.float32)
    low_gap = np.full(close.shape, np.nan, dtype=np.float32)
    event_turnover = np.full(close.shape, np.nan, dtype=np.float32)
    last_date = np.full(symbols, -1, dtype=np.int32)
    last_level = np.full(symbols, np.nan, dtype=np.float32)
    last_turnover = np.full(symbols, np.nan, dtype=np.float32)
    for date in range(dates):
        event = (
            np.isfinite(close[date])
            & np.isfinite(prior_high[date])
            & (close[date] > prior_high[date])
        )
        last_date[event] = date
        last_level[event] = prior_high[date, event]
        last_turnover[event] = relative_turnover[date, event]
        valid = (last_date >= 0) & np.isfinite(last_level)
        age[date, valid] = np.minimum(
            (date - last_date[valid]) / float(max(age_scale, 1)),
            1.0,
        )
        close_gap[date, valid] = (
            close[date, valid] / last_level[valid] - 1.0
        )
        low_gap[date, valid] = low[date, valid] / last_level[valid] - 1.0
        event_turnover[date, valid] = last_turnover[valid]
    return age, close_gap, low_gap, event_turnover


def _generate_breakout(
    writer: _FamilyWriter,
    *,
    raw: np.ndarray,
    turnover: np.ndarray,
    raw_columns: Mapping[str, int],
    turnover_columns: Mapping[str, int],
) -> None:
    stop = writer.cutoff_date_idx + 1
    high = np.asarray(raw[:stop, :, raw_columns["high"]], dtype=np.float32)
    low = np.asarray(raw[:stop, :, raw_columns["low"]], dtype=np.float32)
    close = np.asarray(raw[:stop, :, raw_columns["close"]], dtype=np.float32)
    relative_turnover = np.asarray(
        turnover[:stop, :, turnover_columns["relative_turnover_20"]],
        dtype=np.float32,
    )
    shifted_high = _shift_panel(high, 1)
    shifted_low = _shift_panel(low, 1)
    event_sources: dict[int, np.ndarray] = {}
    for window in BREAKOUT_WINDOWS:
        _unused_low, prior_high = signal_quality._rolling_min_max(
            shifted_high, window
        )
        prior_low, _unused_high = signal_quality._rolling_min_max(
            shifted_low, window
        )
        writer.panel(f"w{window:03d}__close_vs_prior_high", _safe_ratio(close, prior_high) - 1.0)
        writer.panel(f"w{window:03d}__high_vs_prior_high", _safe_ratio(high, prior_high) - 1.0)
        writer.panel(f"w{window:03d}__low_vs_prior_high", _safe_ratio(low, prior_high) - 1.0)
        writer.panel(f"w{window:03d}__close_vs_prior_low", _safe_ratio(close, prior_low) - 1.0)
        writer.panel(f"w{window:03d}__high_vs_prior_low", _safe_ratio(high, prior_low) - 1.0)
        writer.panel(f"w{window:03d}__low_vs_prior_low", _safe_ratio(low, prior_low) - 1.0)
        writer.panel(
            f"w{window:03d}__prior_high_age_fraction",
            signal_quality._rolling_extreme_age(
                shifted_high, window, mode="max"
            )
            / float(max(window - 1, 1)),
        )
        writer.panel(
            f"w{window:03d}__prior_low_age_fraction",
            signal_quality._rolling_extreme_age(
                shifted_low, window, mode="min"
            )
            / float(max(window - 1, 1)),
        )
        if window in BREAKOUT_EVENT_WINDOWS:
            event_sources[window] = prior_high
        else:
            del prior_high
        del prior_low, _unused_low, _unused_high
        gc.collect()
    for window in BREAKOUT_EVENT_WINDOWS:
        age, close_gap, low_gap, event_turnover = _last_breakout_panels(
            close=close,
            low=low,
            prior_high=event_sources[window],
            relative_turnover=relative_turnover,
            age_scale=window,
        )
        writer.panel(f"w{window:03d}__last_breakout_age_fraction", age)
        writer.panel(f"w{window:03d}__close_vs_last_breakout_level", close_gap)
        writer.panel(f"w{window:03d}__low_vs_last_breakout_level", low_gap)
        writer.panel(
            f"w{window:03d}__last_breakout_relative_turnover",
            event_turnover,
        )


def _generate_long_daily(
    writer: _FamilyWriter,
    *,
    raw: np.ndarray,
    raw_columns: Mapping[str, int],
) -> None:
    stop = writer.cutoff_date_idx + 1
    high = np.asarray(raw[:stop, :, raw_columns["high"]], dtype=np.float32)
    low = np.asarray(raw[:stop, :, raw_columns["low"]], dtype=np.float32)
    close = np.asarray(raw[:stop, :, raw_columns["close"]], dtype=np.float32)
    amount = np.asarray(raw[:stop, :, raw_columns["amount"]], dtype=np.float32)
    volume = np.asarray(raw[:stop, :, raw_columns["volume"]], dtype=np.float32)
    previous_close = _shift_panel(close, 1)
    log_close = np.where(close > 0.0, np.log(close), np.nan).astype(np.float32)
    log_return = _one_day_delta(log_close)
    true_range = np.maximum.reduce(
        [
            np.asarray(high - low, dtype=np.float32),
            np.asarray(np.abs(high - previous_close), dtype=np.float32),
            np.asarray(np.abs(low - previous_close), dtype=np.float32),
        ]
    )
    true_range = _safe_ratio(true_range, previous_close)
    log_amount = np.log1p(np.maximum(amount, 0.0)).astype(np.float32)
    abs_log_return = np.abs(log_return).astype(np.float32)
    for window in LONG_DAILY_WINDOWS:
        writer.panel(
            f"return_{window}d",
            signal_quality._lagged_return(close, window),
        )
        ma = signal_quality._rolling_mean(close, window)
        writer.panel(f"ma_distance_{window}d", _safe_ratio(close, ma) - 1.0)
        del ma
        _mean_return, volatility = signal_quality._rolling_mean_std(
            log_return, window
        )
        writer.panel(f"volatility_{window}d", volatility)
        del _mean_return, volatility
        downside = np.where(
            np.isfinite(log_return), np.minimum(log_return, 0.0), np.nan
        ).astype(np.float32)
        downside_square_mean = signal_quality._rolling_mean(
            np.square(downside), window
        )
        writer.panel(
            f"downside_volatility_{window}d",
            np.sqrt(np.maximum(downside_square_mean, 0.0)).astype(np.float32),
        )
        writer.panel(
            f"atr_{window}d", signal_quality._rolling_mean(true_range, window)
        )
        slope, t_value, r2, residual = signal_quality._rolling_linear_stats(
            log_close, window
        )
        writer.panel(f"trend_slope_{window}d", slope)
        writer.panel(f"trend_t_{window}d", t_value)
        writer.panel(f"trend_r2_{window}d", r2)
        writer.panel(f"trend_residual_{window}d", residual)
        path_length = signal_quality._rolling_mean(abs_log_return, window) * float(
            window
        )
        displacement = np.full(log_close.shape, np.nan, dtype=np.float32)
        displacement[window:] = np.abs(
            log_close[window:] - log_close[:-window]
        )
        writer.panel(
            f"efficiency_ratio_{window}d",
            _safe_ratio(displacement, path_length),
        )
        rolling_low, rolling_high = signal_quality._rolling_min_max(close, window)
        writer.panel(
            f"price_range_position_{window}d",
            _safe_ratio(close - rolling_low, rolling_high - rolling_low),
        )
        writer.panel(
            f"high_age_fraction_{window}d",
            signal_quality._rolling_extreme_age(high, window, mode="max")
            / float(max(window - 1, 1)),
        )
        writer.panel(
            f"low_age_fraction_{window}d",
            signal_quality._rolling_extreme_age(low, window, mode="min")
            / float(max(window - 1, 1)),
        )
        writer.panel(
            f"up_day_ratio_{window}d",
            signal_quality._rolling_mean(
                np.where(np.isfinite(log_return), log_return > 0.0, np.nan),
                window,
            ),
        )
        writer.panel(
            f"down_day_ratio_{window}d",
            signal_quality._rolling_mean(
                np.where(np.isfinite(log_return), log_return < 0.0, np.nan),
                window,
            ),
        )
        writer.panel(
            f"price_amount_correlation_{window}d",
            signal_quality._rolling_corr(log_close, log_amount, window),
        )
        writer.panel(
            f"amount_ratio_{window}d",
            _safe_ratio(amount, signal_quality._rolling_mean(amount, window)),
        )
        writer.panel(
            f"volume_ratio_{window}d",
            _safe_ratio(volume, signal_quality._rolling_mean(volume, window)),
        )
        del (
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
        )
        gc.collect()


def _period_key(values: Sequence[str], period: str) -> np.ndarray:
    dates = pd.to_datetime(np.asarray(values, dtype=str))
    frequency = "W-FRI" if period == "week" else "M"
    return np.asarray(pd.PeriodIndex(dates, freq=frequency).astype(str), dtype=str)


def _aggregate_period_bars(
    *,
    date_values: Sequence[str],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = _period_key(date_values, period)
    unique, inverse = np.unique(keys, return_inverse=True)
    symbols = close.shape[1]
    period_high = np.full((len(unique), symbols), np.nan, dtype=np.float32)
    period_low = np.full((len(unique), symbols), np.nan, dtype=np.float32)
    period_close = np.full((len(unique), symbols), np.nan, dtype=np.float32)
    for index in range(len(unique)):
        rows = np.flatnonzero(inverse == index)
        high_value = np.asarray(high[rows[0]], dtype=np.float32).copy()
        low_value = np.asarray(low[rows[0]], dtype=np.float32).copy()
        for row in rows[1:]:
            high_value = np.fmax(high_value, high[row])
            low_value = np.fmin(low_value, low[row])
        period_high[index] = high_value
        period_low[index] = low_value
        period_close[index] = close[rows[-1]]
    completed_index = inverse.astype(np.int32) - 1
    return period_high, period_low, period_close, completed_index


def _broadcast_completed_period(
    values: np.ndarray,
    completed_index: np.ndarray,
) -> np.ndarray:
    output = np.full(
        (len(completed_index), values.shape[1]), np.nan, dtype=np.float32
    )
    valid = completed_index >= 0
    output[valid] = values[completed_index[valid]]
    return output


def _generate_completed_periods(
    writer: _FamilyWriter,
    *,
    pack: Mapping[str, Any],
    raw: np.ndarray,
    raw_columns: Mapping[str, int],
) -> None:
    stop = writer.cutoff_date_idx + 1
    dates = list(pack["date_values"][:stop])
    high = np.asarray(raw[:stop, :, raw_columns["high"]], dtype=np.float32)
    low = np.asarray(raw[:stop, :, raw_columns["low"]], dtype=np.float32)
    close = np.asarray(raw[:stop, :, raw_columns["close"]], dtype=np.float32)
    for period, windows in PERIOD_WINDOWS.items():
        period_high, period_low, period_close, completed_index = (
            _aggregate_period_bars(
                date_values=dates,
                high=high,
                low=low,
                close=close,
                period=period,
            )
        )
        previous_close = _shift_panel(period_close, 1)
        log_close = np.where(
            period_close > 0.0, np.log(period_close), np.nan
        ).astype(np.float32)
        log_return = _one_day_delta(log_close)
        true_range = np.maximum.reduce(
            [
                period_high - period_low,
                np.abs(period_high - previous_close),
                np.abs(period_low - previous_close),
            ]
        )
        true_range = _safe_ratio(true_range, previous_close)
        for window in windows:
            metrics: dict[str, np.ndarray] = {
                "return": signal_quality._lagged_return(period_close, window),
                "ma_distance": _safe_ratio(
                    period_close,
                    signal_quality._rolling_mean(period_close, window),
                )
                - 1.0,
                "volatility": signal_quality._rolling_mean_std(
                    log_return, window
                )[1],
                "atr": signal_quality._rolling_mean(true_range, window),
                "trend_slope": signal_quality._rolling_linear_stats(
                    log_close, window
                )[0],
            }
            rolling_low, rolling_high = signal_quality._rolling_min_max(
                period_close, window
            )
            metrics["price_range_position"] = _safe_ratio(
                period_close - rolling_low, rolling_high - rolling_low
            )
            for metric, values in metrics.items():
                writer.panel(
                    f"completed_{period}__{metric}_{window}p",
                    _broadcast_completed_period(values, completed_index),
                )
            del metrics, rolling_low, rolling_high
            gc.collect()
        del period_high, period_low, period_close, completed_index
        gc.collect()


def _recursive_ema(values: np.ndarray, span: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    output = np.full(source.shape, np.nan, dtype=np.float32)
    state = np.full(source.shape[1], np.nan, dtype=np.float64)
    alpha = 2.0 / float(int(span) + 1)
    for date in range(source.shape[0]):
        current = np.asarray(source[date], dtype=np.float64)
        finite = np.isfinite(current)
        initialize = finite & ~np.isfinite(state)
        state[initialize] = current[initialize]
        update = finite & np.isfinite(state) & ~initialize
        state[update] += alpha * (current[update] - state[update])
        output[date, finite] = state[finite].astype(np.float32)
    return output


def _kama(values: np.ndarray, er_period: int, fast: int = 2, slow: int = 30) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    output = np.full(source.shape, np.nan, dtype=np.float32)
    change = np.abs(source - _shift_panel(source, int(er_period)))
    one_day = np.abs(_one_day_delta(source))
    volatility = signal_quality._rolling_mean(one_day, int(er_period)) * float(
        er_period
    )
    efficiency = _safe_ratio(change, volatility)
    fast_sc = 2.0 / float(fast + 1)
    slow_sc = 2.0 / float(slow + 1)
    state = np.full(source.shape[1], np.nan, dtype=np.float64)
    for date in range(source.shape[0]):
        current = np.asarray(source[date], dtype=np.float64)
        er = np.asarray(efficiency[date], dtype=np.float64)
        finite = np.isfinite(current)
        initialize = finite & ~np.isfinite(state)
        state[initialize] = current[initialize]
        update = finite & np.isfinite(state) & np.isfinite(er) & ~initialize
        smoothing = np.square(er * (fast_sc - slow_sc) + slow_sc)
        state[update] += smoothing[update] * (current[update] - state[update])
        output[date, finite] = state[finite].astype(np.float32)
    return output


def _rsi(values: np.ndarray, window: int) -> np.ndarray:
    delta = _one_day_delta(values)
    gain = np.where(np.isfinite(delta), np.maximum(delta, 0.0), np.nan).astype(
        np.float32
    )
    loss = np.where(np.isfinite(delta), np.maximum(-delta, 0.0), np.nan).astype(
        np.float32
    )
    average_gain = _recursive_ema(gain, 2 * int(window) - 1)
    average_loss = _recursive_ema(loss, 2 * int(window) - 1)
    denominator = average_gain + average_loss
    return _safe_ratio(average_gain, denominator)


def _kdj(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rolling_low, rolling_high = signal_quality._rolling_min_max(close * 0.0 + low, window)
    _unused_low, high_max = signal_quality._rolling_min_max(close * 0.0 + high, window)
    rsv = _safe_ratio(close - rolling_low, high_max - rolling_low)
    k = np.full(close.shape, np.nan, dtype=np.float32)
    d = np.full(close.shape, np.nan, dtype=np.float32)
    k_state = np.full(close.shape[1], 0.5, dtype=np.float64)
    d_state = np.full(close.shape[1], 0.5, dtype=np.float64)
    started = np.zeros(close.shape[1], dtype=bool)
    for date in range(close.shape[0]):
        current = np.asarray(rsv[date], dtype=np.float64)
        valid = np.isfinite(current)
        k_state[valid] = (2.0 * k_state[valid] + current[valid]) / 3.0
        d_state[valid] = (2.0 * d_state[valid] + k_state[valid]) / 3.0
        started |= valid
        k[date, started] = k_state[started].astype(np.float32)
        d[date, started] = d_state[started].astype(np.float32)
    j = (3.0 * k - 2.0 * d).astype(np.float32)
    del rolling_low, rolling_high, _unused_low, high_max
    return k, d, j


def _last_event_age(event: np.ndarray, scale: int = 60) -> np.ndarray:
    flags = np.asarray(event, dtype=bool)
    output = np.full(flags.shape, np.nan, dtype=np.float32)
    last = np.full(flags.shape[1], -1, dtype=np.int32)
    for date in range(flags.shape[0]):
        last[flags[date]] = date
        valid = last >= 0
        output[date, valid] = np.minimum(
            (date - last[valid]) / float(max(scale, 1)), 1.0
        )
    return output


def _generate_traditional(
    writer: _FamilyWriter,
    *,
    raw: np.ndarray,
    raw_columns: Mapping[str, int],
) -> None:
    stop = writer.cutoff_date_idx + 1
    high = np.asarray(raw[:stop, :, raw_columns["high"]], dtype=np.float32)
    low = np.asarray(raw[:stop, :, raw_columns["low"]], dtype=np.float32)
    close = np.asarray(raw[:stop, :, raw_columns["close"]], dtype=np.float32)
    for er_period in (10, 20):
        current = _kama(close, er_period)
        writer.panel(
            f"kama_{er_period}_2_30__distance", _safe_ratio(close, current) - 1.0
        )
        writer.panel(
            f"kama_{er_period}_2_30__slope_1d",
            _safe_ratio(current, _shift_panel(current, 1)) - 1.0,
        )
        writer.panel(
            f"kama_{er_period}_2_30__slope_5d",
            _safe_ratio(current, _shift_panel(current, 5)) - 1.0,
        )
        del current
    ema12 = _recursive_ema(close, 12)
    ema26 = _recursive_ema(close, 26)
    dif = ema12 - ema26
    macd_signal = _recursive_ema(dif, 9)
    histogram = dif - macd_signal
    writer.panel("macd_12_26_9__dif_norm", _safe_ratio(dif, close))
    writer.panel("macd_12_26_9__signal_norm", _safe_ratio(macd_signal, close))
    writer.panel("macd_12_26_9__hist_norm", _safe_ratio(histogram, close))
    writer.panel(
        "macd_12_26_9__hist_delta_1d", _safe_ratio(_one_day_delta(histogram), close)
    )
    writer.panel(
        "macd_12_26_9__hist_delta_3d", _safe_ratio(_one_day_delta(histogram, 3), close)
    )
    previous_histogram = _shift_panel(histogram, 1)
    writer.panel(
        "macd_12_26_9__bull_cross_age_fraction",
        _last_event_age(
            np.isfinite(histogram)
            & np.isfinite(previous_histogram)
            & (histogram >= 0.0)
            & (previous_histogram < 0.0)
        ),
    )
    for window in (9, 21):
        k, d, j = _kdj(high, low, close, window)
        writer.panel(f"kdj_{window}_3_3__k", k)
        writer.panel(f"kdj_{window}_3_3__d", d)
        writer.panel(f"kdj_{window}_3_3__j", j)
        writer.panel(f"kdj_{window}_3_3__k_minus_d", k - d)
        writer.panel(f"kdj_{window}_3_3__k_delta_1d", _one_day_delta(k))
        writer.panel(f"kdj_{window}_3_3__j_delta_1d", _one_day_delta(j))
    for window in (6, 14, 24):
        writer.panel(f"rsi_{window}", _rsi(close, window))


def _confirmed_pivot_events(
    values: np.ndarray,
    radius: int,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values, dtype=np.float32)
    amount = int(radius)
    events = np.zeros(source.shape, dtype=bool)
    levels = np.full(source.shape, np.nan, dtype=np.float32)
    if source.shape[0] <= 2 * amount:
        return events, levels
    center = source[amount : source.shape[0] - amount]
    valid = np.isfinite(center)
    for offset in range(1, amount + 1):
        left = source[amount - offset : source.shape[0] - amount - offset]
        right = source[amount + offset : source.shape[0] - amount + offset]
        if mode == "high":
            valid &= np.isfinite(left) & np.isfinite(right) & (center > left) & (center > right)
        elif mode == "low":
            valid &= np.isfinite(left) & np.isfinite(right) & (center < left) & (center < right)
        else:
            raise ValueError("pivot mode must be high or low")
    confirmation = np.arange(amount, source.shape[0] - amount) + amount
    events[confirmation] = valid
    levels[confirmation] = np.where(valid, center, np.nan)
    return events, levels


def _pivot_state_panels(
    *,
    close: np.ndarray,
    high_event: np.ndarray,
    high_level: np.ndarray,
    low_event: np.ndarray,
    low_level: np.ndarray,
) -> dict[str, np.ndarray]:
    dates, symbols = close.shape
    outputs = {
        name: np.full(close.shape, np.nan, dtype=np.float32)
        for name in (
            "close_vs_last_pivot_high",
            "close_vs_last_pivot_low",
            "last_pivot_high_age_fraction",
            "last_pivot_low_age_fraction",
            "last_pivot_high_change",
            "last_pivot_low_change",
            "swing_position",
            "last_swing_direction",
        )
    }
    last_high = np.full(symbols, np.nan, dtype=np.float32)
    previous_high = np.full(symbols, np.nan, dtype=np.float32)
    last_low = np.full(symbols, np.nan, dtype=np.float32)
    previous_low = np.full(symbols, np.nan, dtype=np.float32)
    last_high_date = np.full(symbols, -1, dtype=np.int32)
    last_low_date = np.full(symbols, -1, dtype=np.int32)
    for date in range(dates):
        high_mask = high_event[date]
        low_mask = low_event[date]
        previous_high[high_mask] = last_high[high_mask]
        last_high[high_mask] = high_level[date, high_mask]
        last_high_date[high_mask] = date
        previous_low[low_mask] = last_low[low_mask]
        last_low[low_mask] = low_level[date, low_mask]
        last_low_date[low_mask] = date
        valid_high = np.isfinite(last_high)
        valid_low = np.isfinite(last_low)
        outputs["close_vs_last_pivot_high"][date, valid_high] = (
            close[date, valid_high] / last_high[valid_high] - 1.0
        )
        outputs["close_vs_last_pivot_low"][date, valid_low] = (
            close[date, valid_low] / last_low[valid_low] - 1.0
        )
        outputs["last_pivot_high_age_fraction"][date, valid_high] = np.minimum(
            (date - last_high_date[valid_high]) / 60.0, 1.0
        )
        outputs["last_pivot_low_age_fraction"][date, valid_low] = np.minimum(
            (date - last_low_date[valid_low]) / 60.0, 1.0
        )
        high_change = valid_high & np.isfinite(previous_high)
        low_change = valid_low & np.isfinite(previous_low)
        outputs["last_pivot_high_change"][date, high_change] = (
            last_high[high_change] / previous_high[high_change] - 1.0
        )
        outputs["last_pivot_low_change"][date, low_change] = (
            last_low[low_change] / previous_low[low_change] - 1.0
        )
        both = valid_high & valid_low & (last_high > last_low)
        outputs["swing_position"][date, both] = (
            (close[date, both] - last_low[both])
            / (last_high[both] - last_low[both])
        )
        latest = both & (last_high_date != last_low_date)
        outputs["last_swing_direction"][date, latest] = np.where(
            last_low_date[latest] > last_high_date[latest], 1.0, -1.0
        )
    return outputs


def _generate_swing(
    writer: _FamilyWriter,
    *,
    raw: np.ndarray,
    raw_columns: Mapping[str, int],
) -> None:
    stop = writer.cutoff_date_idx + 1
    high = np.asarray(raw[:stop, :, raw_columns["high"]], dtype=np.float32)
    low = np.asarray(raw[:stop, :, raw_columns["low"]], dtype=np.float32)
    close = np.asarray(raw[:stop, :, raw_columns["close"]], dtype=np.float32)
    for radius in (1, 2):
        high_event, high_level = _confirmed_pivot_events(
            high, radius, mode="high"
        )
        low_event, low_level = _confirmed_pivot_events(low, radius, mode="low")
        panels = _pivot_state_panels(
            close=close,
            high_event=high_event,
            high_level=high_level,
            low_event=low_event,
            low_level=low_level,
        )
        for window in (20, 60):
            panels[f"confirmed_high_count_{window}"] = (
                signal_quality._rolling_mean(high_event.astype(np.float32), window)
                * float(window)
            )
            panels[f"confirmed_low_count_{window}"] = (
                signal_quality._rolling_mean(low_event.astype(np.float32), window)
                * float(window)
            )
        for metric, panel in panels.items():
            writer.panel(f"radius{radius}__{metric}", panel)


def _turnover_cost_panels(
    *,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    log_turnover_pct: np.ndarray,
) -> dict[str, np.ndarray]:
    shape = close.shape
    outputs = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in feature_catalog()["turnover_cost_proxy"]
    }
    mass = np.zeros(shape[1], dtype=np.float64)
    mean_cost = np.full(shape[1], np.nan, dtype=np.float64)
    centered_moment2 = np.zeros(shape[1], dtype=np.float64)
    age_mass = np.zeros(shape[1], dtype=np.float64)
    for date in range(shape[0]):
        typical = (high[date] + low[date] + close[date]) / 3.0
        log_price = np.where(typical > 0.0, np.log(typical), np.nan)
        turnover = np.expm1(log_turnover_pct[date]) / 100.0
        turnover = np.clip(turnover, 0.0, 1.0)
        valid = (
            np.isfinite(log_price)
            & np.isfinite(turnover)
            & np.isfinite(close[date])
            & (close[date] > 0.0)
        )
        old_mass = mass.copy()
        survival = 1.0 - turnover[valid]
        surviving_mass = survival * old_mass[valid]
        added_mass = turnover[valid]
        combined_mass = surviving_mass + added_mass
        age_mass[valid] = survival * (age_mass[valid] + old_mass[valid])
        old_mean = mean_cost[valid]
        old_centered_moment2 = centered_moment2[valid]
        has_survivors = (surviving_mass > 1.0e-12) & np.isfinite(old_mean)
        next_mean = np.asarray(log_price[valid], dtype=np.float64).copy()
        next_centered_moment2 = np.zeros_like(next_mean)
        if bool(has_survivors.any()):
            delta = log_price[valid][has_survivors] - old_mean[has_survivors]
            total = combined_mass[has_survivors]
            next_mean[has_survivors] = old_mean[has_survivors] + (
                added_mass[has_survivors] / total
            ) * delta
            next_centered_moment2[has_survivors] = (
                survival[has_survivors]
                * old_centered_moment2[has_survivors]
                + surviving_mass[has_survivors]
                * added_mass[has_survivors]
                / total
                * np.square(delta)
            )
        has_mass = combined_mass > 1.0e-12
        mass[valid] = combined_mass
        mean_cost[valid] = np.where(has_mass, next_mean, np.nan)
        centered_moment2[valid] = np.where(
            has_mass, next_centered_moment2, 0.0
        )
        usable = valid & (mass > 1.0e-8)
        mean = np.full(shape[1], np.nan, dtype=np.float64)
        variance = np.full(shape[1], np.nan, dtype=np.float64)
        mean[usable] = mean_cost[usable]
        variance[usable] = np.maximum(
            centered_moment2[usable] / mass[usable], 0.0
        )
        standard_deviation = np.sqrt(variance)
        log_close = np.where(close[date] > 0.0, np.log(close[date]), np.nan)
        distance = log_close - mean
        profit = np.full(shape[1], np.nan, dtype=np.float64)
        dispersed = usable & (standard_deviation > 1.0e-8)
        profit[dispersed] = special.ndtr(
            distance[dispersed] / standard_deviation[dispersed]
        )
        concentrated = usable & ~dispersed
        profit[concentrated] = (distance[concentrated] >= 0.0).astype(np.float64)
        outputs["survivor_log_close_minus_mean_cost"][date] = distance.astype(
            np.float32
        )
        outputs["survivor_log_cost_std"][date] = standard_deviation.astype(
            np.float32
        )
        outputs["survivor_profit_fraction_normal"][date] = profit.astype(
            np.float32
        )
        outputs["survivor_mass"][date, usable] = mass[usable].astype(np.float32)
        outputs["survivor_mean_age_days"][date, usable] = (
            age_mass[usable] / mass[usable]
        ).astype(np.float32)
    return outputs


def _generate_turnover_cost(
    writer: _FamilyWriter,
    *,
    raw: np.ndarray,
    turnover: np.ndarray,
    raw_columns: Mapping[str, int],
    turnover_columns: Mapping[str, int],
) -> None:
    stop = writer.cutoff_date_idx + 1
    panels = _turnover_cost_panels(
        high=np.asarray(raw[:stop, :, raw_columns["high"]], dtype=np.float32),
        low=np.asarray(raw[:stop, :, raw_columns["low"]], dtype=np.float32),
        close=np.asarray(raw[:stop, :, raw_columns["close"]], dtype=np.float32),
        log_turnover_pct=np.asarray(
            turnover[:stop, :, turnover_columns["log_turnover_pct"]],
            dtype=np.float32,
        ),
    )
    for name, values in panels.items():
        writer.panel(name, values)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64).copy()
    x = (x + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(
        np.uint64
    )
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(
        np.uint64
    )
    return x ^ (x >> np.uint64(31))


def _generate_noise(writer: _FamilyWriter) -> None:
    candidate_id = np.arange(len(writer.date_idx), dtype=np.uint64)
    for index, name in enumerate(feature_catalog()[CONTROL_FAMILY]):
        seed = np.uint64(index + 1) * np.uint64(0xD1342543DE82EF95)
        raw = _splitmix64(candidate_id ^ seed)
        values = (
            (raw >> np.uint64(11)).astype(np.float64)
            * (1.0 / float(1 << 53))
            * 2.0
            - 1.0
        ).astype(np.float32)
        writer.candidate(name, values)


def _family_manifest_path(output_root: Path, family: str) -> Path:
    return output_root / "features" / family / "manifest.json"


def load_prepared_family(
    output_root: Path,
    family: str,
    *,
    candidate_count: int,
) -> PreparedFamily:
    path = _family_manifest_path(output_root, family)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PREPARED_FEATURE_SCHEMA:
        raise ValueError(f"unexpected prepared feature schema: {family}")
    if manifest.get("family") != family:
        raise ValueError(f"prepared family id changed: {family}")
    feature_names = tuple(str(value) for value in manifest["feature_names"])
    if feature_names != feature_catalog()[family]:
        raise ValueError(f"prepared feature order changed: {family}")
    record = dict(manifest["file"])
    feature_path = _resolve(record["path"])
    expected_path = (
        output_root / "features" / family / "features.float32.dat"
    ).resolve()
    if feature_path != expected_path:
        raise ValueError(f"prepared feature path changed: {family}")
    expected_shape = (len(feature_names), int(candidate_count))
    shape = tuple(int(value) for value in record["shape"])
    if int(manifest.get("candidate_count", -1)) != int(candidate_count):
        raise ValueError(f"prepared family candidate count changed: {family}")
    if shape != expected_shape:
        raise ValueError(f"prepared feature shape changed: {family}")
    if record.get("dtype") != "float32" or record.get("layout") != "feature_major":
        raise ValueError(f"prepared feature storage format changed: {family}")
    expected_size = int(np.prod(expected_shape, dtype=np.int64)) * 4
    if int(record["size"]) != expected_size:
        raise ValueError(f"prepared feature declared size changed: {family}")
    if feature_path.stat().st_size != int(record["size"]):
        raise ValueError(f"prepared feature size changed: {family}")
    return PreparedFamily(
        family=family,
        path=feature_path,
        shape=shape,
        feature_names=feature_names,
        manifest=manifest,
    )


def prepare_family(
    *,
    study_path: Path,
    output_root: Path,
    family: str,
) -> dict[str, Any]:
    if family not in ALL_FAMILIES:
        raise ValueError(f"unknown feature family: {family}")
    study = load_study(study_path)
    candidate_count = int(study["data"]["candidate_count"])
    existing = _family_manifest_path(output_root, family)
    if existing.exists():
        return load_prepared_family(
            output_root,
            family,
            candidate_count=candidate_count,
        ).manifest
    started = time.perf_counter()
    inputs, pack = _load_validated_inputs(study)
    if inputs.candidate_count != candidate_count:
        raise ValueError("prepared feature candidate count differs from configuration")
    raw, turnover, raw_columns, turnover_columns = _raw_views(
        pack, inputs.cutoff_date_idx
    )
    names = feature_catalog()[family]
    family_root = output_root / "features" / family
    feature_path = family_root / "features.float32.dat"
    writer = _FamilyWriter(
        path=feature_path,
        feature_names=names,
        candidate_date_idx=inputs.candidate_date_idx,
        candidate_symbol_idx=inputs.candidate_symbol_idx,
        cutoff_date_idx=inputs.cutoff_date_idx,
    )
    if family == "recent_kline_sequence":
        _generate_recent(
            writer,
            raw=raw,
            turnover=turnover,
            raw_columns=raw_columns,
            turnover_columns=turnover_columns,
        )
    elif family == "breakout_retest_levels":
        _generate_breakout(
            writer,
            raw=raw,
            turnover=turnover,
            raw_columns=raw_columns,
            turnover_columns=turnover_columns,
        )
    elif family == "long_daily_context":
        _generate_long_daily(writer, raw=raw, raw_columns=raw_columns)
    elif family == "completed_week_month_context":
        _generate_completed_periods(
            writer,
            pack=pack,
            raw=raw,
            raw_columns=raw_columns,
        )
    elif family == "traditional_indicators":
        _generate_traditional(writer, raw=raw, raw_columns=raw_columns)
    elif family == "confirmed_swing_structure":
        _generate_swing(writer, raw=raw, raw_columns=raw_columns)
    elif family == "turnover_cost_proxy":
        _generate_turnover_cost(
            writer,
            raw=raw,
            turnover=turnover,
            raw_columns=raw_columns,
            turnover_columns=turnover_columns,
        )
    else:
        _generate_noise(writer)
    writer.finish()
    after_boundary = writer.values[:, writer.valid_count :]
    if after_boundary.size and not bool(np.isnan(after_boundary).all()):
        raise AssertionError("prepared feature family exceeds the configured date boundary")
    file_record = _record_for_file(feature_path)
    file_record.update(
        {
            "shape": [len(names), inputs.candidate_count],
            "dtype": "float32",
            "layout": "feature_major",
        }
    )
    manifest = {
        "schema": PREPARED_FEATURE_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "family": family,
        "role": "negative_control" if family == CONTROL_FAMILY else "candidate",
        "feature_names": list(names),
        "candidate_count": inputs.candidate_count,
        "candidate_count_through_boundary": writer.valid_count,
        "maximum_source_date": study["folds"]["maximum_feature_source_date"],
        "rows_after_boundary_all_missing": True,
        "file": file_record,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _atomic_write_json(existing, manifest)
    return manifest


def prepare_features(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    family: str | None = None,
) -> dict[str, Any]:
    load_study(study_path)
    selected = ALL_FAMILIES if family is None else (str(family),)
    manifests = [
        prepare_family(
            study_path=study_path,
            output_root=output_root,
            family=current,
        )
        for current in selected
    ]
    result = {
        "schema": "seq100_mfe_feature_family_preparation_summary/v1",
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "families": [
            {
                "family": item["family"],
                "feature_count": len(item["feature_names"]),
                "path": item["file"]["path"],
                "size": item["file"]["size"],
            }
            for item in manifests
        ],
    }
    return result


def _model_parameters(study: Mapping[str, Any]) -> tuple[dict[str, Any], int, int]:
    model = dict(study["model"])
    parameters = {
        "objective": "huber",
        "metric": str(model["early_stopping_metric"]),
        "alpha": float(model["huber_alpha"]),
        "boosting_type": "gbdt",
        "device_type": "cpu",
        "learning_rate": float(model["learning_rate"]),
        "num_leaves": int(model["num_leaves"]),
        "max_depth": int(model["max_depth"]),
        "min_data_in_leaf": int(model["min_data_in_leaf"]),
        "feature_fraction": float(model["feature_fraction"]),
        "bagging_fraction": float(model["bagging_fraction"]),
        "bagging_freq": int(model["bagging_freq"]),
        "lambda_l1": float(model["lambda_l1"]),
        "lambda_l2": float(model["lambda_l2"]),
        "max_bin": int(model["max_bin"]),
        "deterministic": bool(model["deterministic"]),
        "force_col_wise": bool(model["force_col_wise"]),
        "num_threads": int(model["num_threads"]),
        "histogram_pool_size": int(model["histogram_pool_size_mb"]),
        "seed": int(model["seed"]),
        "feature_fraction_seed": int(model["seed"]),
        "bagging_seed": int(model["seed"]),
        "data_random_seed": int(model["seed"]),
        "drop_seed": int(model["seed"]),
        "extra_seed": int(model["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    return (
        parameters,
        int(model["num_boost_round"]),
        int(model["early_stopping_rounds"]),
    )


def _combined_sequence(
    *,
    inputs: base.LearnabilityInputs,
    row_ids: np.ndarray,
    category_vocabularies: Sequence[np.ndarray],
    batch_size: int,
    extra: np.ndarray | None,
) -> Any:
    import lightgbm as lgb

    rows = np.asarray(row_ids, dtype=np.int64)
    continuous_columns = np.asarray(inputs.continuous_columns, dtype=np.int32)
    categorical_columns = np.asarray(inputs.categorical_columns, dtype=np.int32)

    class CombinedSequence(lgb.Sequence):
        def __init__(self) -> None:
            self.batch_size = int(batch_size)

        def __len__(self) -> int:
            return int(len(rows))

        def _block(self, local: np.ndarray) -> np.ndarray:
            global_rows = rows[local]
            parts = [
                np.asarray(
                    inputs.continuous[np.ix_(global_rows, continuous_columns)],
                    dtype=np.float64,
                )
            ]
            if extra is not None:
                parts.append(
                    np.asarray(extra[:, global_rows].T, dtype=np.float64)
                )
            for position, column in enumerate(categorical_columns):
                parts.append(
                    signal_quality._map_categories(
                        inputs.categorical[global_rows, int(column)],
                        category_vocabularies[position],
                    ).reshape(-1, 1)
                )
            return np.concatenate(parts, axis=1).astype(np.float64, copy=False)

        def __getitem__(self, index: Any) -> np.ndarray:
            if isinstance(index, slice):
                local = np.arange(
                    0 if index.start is None else int(index.start),
                    len(rows) if index.stop is None else int(index.stop),
                    1 if index.step is None else int(index.step),
                    dtype=np.int64,
                )
                return self._block(local)
            if isinstance(index, (list, tuple, np.ndarray)):
                return self._block(np.asarray(index, dtype=np.int64))
            if isinstance(index, (int, np.integer)):
                return self._block(np.asarray([int(index)], dtype=np.int64))[0]
            raise TypeError(
                f"unsupported LightGBM sequence index: {type(index).__name__}"
            )

    return CombinedSequence()


@dataclass
class _Datasets:
    train_rows: np.ndarray
    evaluation_rows: np.ndarray
    train_sequence: Any
    evaluation_sequence: Any
    train_set: Any
    evaluation_set: Any
    feature_names: list[str]


def _aligned_mfe(
    *,
    inputs: base.LearnabilityInputs,
    rows: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(inputs.label_values("mfe", horizon)[rows], dtype=np.float32)
    valid = np.isfinite(raw)
    label, weight = base.aligned_target_weights(
        values=raw,
        date_idx=inputs.candidate_date_idx[rows],
        valid=valid,
    )
    return label.astype(np.float32, copy=False), weight, raw


def _build_datasets(
    *,
    study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    horizon: int,
    extra: np.ndarray | None,
    extra_names: Sequence[str],
) -> tuple[_Datasets, np.ndarray, np.ndarray]:
    import lightgbm as lgb

    signal_quality.sequence_training._trim_working_set()
    category_vocabularies = signal_quality._fit_category_vocabularies(
        inputs.categorical,
        np.asarray(train_rows, dtype=np.int64),
        inputs.categorical_columns,
    )
    batch_size = int(study["model"]["sequence_batch_size"])
    train_sequence = _combined_sequence(
        inputs=inputs,
        row_ids=train_rows,
        category_vocabularies=category_vocabularies,
        batch_size=batch_size,
        extra=extra,
    )
    evaluation_sequence = _combined_sequence(
        inputs=inputs,
        row_ids=evaluation_rows,
        category_vocabularies=category_vocabularies,
        batch_size=batch_size,
        extra=extra,
    )
    train_sequence = base._memory_trimmed_sequence(train_sequence, study)
    evaluation_sequence = base._memory_trimmed_sequence(evaluation_sequence, study)
    train_label, train_weight, _train_raw = _aligned_mfe(
        inputs=inputs, rows=train_rows, horizon=horizon
    )
    evaluation_label, evaluation_weight, evaluation_raw = _aligned_mfe(
        inputs=inputs, rows=evaluation_rows, horizon=horizon
    )
    feature_names = [
        str(item["name"]) for item in inputs.continuous_catalog
    ] + list(extra_names) + [
        str(item["name"]) for item in inputs.categorical_catalog
    ]
    categorical_count = len(inputs.categorical_columns)
    categorical_positions = list(
        range(len(feature_names) - categorical_count, len(feature_names))
    )
    construction = {
        "max_bin": int(study["model"]["max_bin"]),
        "data_random_seed": int(study["model"]["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(
        train_sequence,
        label=train_label,
        weight=train_weight,
        feature_name=feature_names,
        categorical_feature=categorical_positions,
        free_raw_data=True,
        params=construction,
    )
    evaluation_set = lgb.Dataset(
        evaluation_sequence,
        label=evaluation_label,
        weight=evaluation_weight,
        feature_name=feature_names,
        categorical_feature=categorical_positions,
        reference=train_set,
        free_raw_data=True,
        params=construction,
    )
    train_set.construct()
    signal_quality.sequence_training._trim_working_set()
    evaluation_set.construct()
    signal_quality.sequence_training._trim_working_set()
    return (
        _Datasets(
            train_rows=np.asarray(train_rows, dtype=np.int64),
            evaluation_rows=np.asarray(evaluation_rows, dtype=np.int64),
            train_sequence=train_sequence,
            evaluation_sequence=evaluation_sequence,
            train_set=train_set,
            evaluation_set=evaluation_set,
            feature_names=feature_names,
        ),
        evaluation_weight,
        evaluation_raw,
    )


def _release_datasets(datasets: _Datasets) -> None:
    datasets.train_set = None
    datasets.evaluation_set = None
    datasets.train_sequence = None
    datasets.evaluation_sequence = None
    gc.collect()
    signal_quality.sequence_training._trim_working_set()


def _predict(model: Any, sequence: Any, iterations: int) -> np.ndarray:
    output = np.empty(len(sequence), dtype=np.float32)
    for start in range(0, len(sequence), 250_000):
        stop = min(start + 250_000, len(sequence))
        output[start:stop] = np.asarray(
            model.predict(sequence[start:stop], num_iteration=int(iterations)),
            dtype=np.float32,
        )
    return output


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    os.replace(temporary, path)


def _task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    output_root: Path,
) -> bool:
    if not path.is_file():
        return False
    try:
        import lightgbm as lgb

        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema") != TASK_RESULT_SCHEMA:
            return False
        if result.get("study_id") != STUDY_ID:
            return False
        if result.get("status") != "completed":
            return False
        for key in ("task_id", "stage", "variant", "fold_year", "horizon"):
            if result.get(key) != task.get(key):
                return False
        if result.get("target") != "mfe" or int(result.get("purge_days", -1)) != int(
            task["horizon"]
        ):
            return False
        expected_parameters, _rounds, _patience = _model_parameters(study)
        if dict(result.get("parameters", {}) or {}) != expected_parameters:
            return False
        files = dict(result.get("files", {}) or {})
        for record in files.values():
            file_path = _resolve(record["path"])
            if not file_path.is_file() or file_path.stat().st_size != int(
                record["size"]
            ):
                return False
        lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        prediction = np.load(_resolve(files["prediction"]["path"]), allow_pickle=False)
        if tuple(prediction.shape) != tuple(files["prediction"]["shape"]):
            return False
        pd.read_parquet(_resolve(files["daily_metrics"]["path"]))
        if task["stage"] == "outer_evaluation":
            rows = np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False)
            if tuple(rows.shape) != tuple(files["evaluation_rows"]["shape"]):
                return False
            tuning_task = _task_by_id(
                f"mfe{int(task['horizon'])}_{int(task['fold_year'])}_baseline_tuning"
            )
            tuning_path = _task_result_path(output_root, tuning_task)
            if not tuning_path.is_file():
                return False
            tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
            if tuning.get("status") != "completed" or tuning.get(
                "task_id"
            ) != tuning_task["task_id"]:
                return False
            if int(result["fixed_iteration_source"]["best_iteration"]) != int(
                tuning["best_iteration"]
            ):
                return False
        return True
    except Exception:
        return False


def _tuning_dir(output_root: Path, year: int, horizon: int) -> Path:
    return output_root / "tuning" / f"fold_{year}" / f"h{horizon:02d}"


def _outer_dir(
    output_root: Path, year: int, horizon: int, variant: str
) -> Path:
    return (
        output_root
        / "outer"
        / f"fold_{year}"
        / f"h{horizon:02d}"
        / variant
    )


def _task_by_id(task_id: str) -> dict[str, Any]:
    for task in _model_task_plan():
        if task["task_id"] == task_id:
            return task
    raise KeyError(f"unknown task id: {task_id}")


def _task_result_path(output_root: Path, task: Mapping[str, Any]) -> Path:
    if task["stage"] == "iteration_tuning":
        directory = _tuning_dir(
            output_root, int(task["fold_year"]), int(task["horizon"])
        )
    else:
        directory = _outer_dir(
            output_root,
            int(task["fold_year"]),
            int(task["horizon"]),
            str(task["variant"]),
        )
    return directory / "task_result.json"


def _run_tuning_task(
    *,
    study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
    year: int,
    horizon: int,
) -> dict[str, Any]:
    import lightgbm as lgb

    output_dir = _tuning_dir(output_root, year, horizon)
    result_path = output_dir / "task_result.json"
    task = _task_by_id(f"mfe{horizon}_{year}_baseline_tuning")
    if _task_complete(
        result_path,
        task=task,
        study=study,
        output_root=output_root,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    inner_year = int(year) - 1
    fold = inputs.common_path_rows(inner_year, horizon)
    datasets, _evaluation_weight, evaluation_raw = _build_datasets(
        study=study,
        inputs=inputs,
        train_rows=fold.train_rows,
        evaluation_rows=fold.evaluation_rows,
        horizon=horizon,
        extra=None,
        extra_names=(),
    )
    parameters, maximum_rounds, patience = _model_parameters(study)
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        datasets.train_set,
        num_boost_round=maximum_rounds,
        valid_sets=[datasets.evaluation_set],
        valid_names=["inner_validation"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=patience,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )
    elapsed = float(time.perf_counter() - started)
    best_iteration = int(model.best_iteration)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.txt"
    model.save_model(str(model_path), num_iteration=best_iteration)
    prediction = _predict(model, datasets.evaluation_sequence, best_iteration)
    prediction_path = output_dir / "prediction.npy"
    _save_npy(prediction_path, prediction)
    daily, metrics = base.regression_metrics(
        date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
        actual=evaluation_raw,
        prediction=prediction,
        date_values=inputs.date_values,
        horizon=horizon,
    )
    daily_path = output_dir / "daily_metrics.parquet"
    daily.to_parquet(daily_path, index=False, compression="zstd")
    result = {
        "schema": TASK_RESULT_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": task["task_id"],
        "target": "mfe",
        "purge_days": int(horizon),
        "stage": "iteration_tuning",
        "variant": "baseline",
        "fold_year": int(year),
        "inner_validation_year": inner_year,
        "horizon": int(horizon),
        "train_row_count": int(len(fold.train_rows)),
        "evaluation_row_count": int(len(fold.evaluation_rows)),
        "best_iteration": best_iteration,
        "training_seconds": elapsed,
        "parameters": parameters,
        "metrics": metrics,
        "files": {
            "model": _record_for_file(model_path),
            "prediction": {
                **_record_for_file(prediction_path),
                "shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
            },
            "daily_metrics": _record_for_file(daily_path),
        },
    }
    _atomic_write_json(result_path, result)
    _release_datasets(datasets)
    del model, prediction, fold
    return result


def _run_outer_task(
    *,
    study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
    year: int,
    horizon: int,
    variant: str,
) -> dict[str, Any]:
    import lightgbm as lgb

    output_dir = _outer_dir(output_root, year, horizon, variant)
    result_path = output_dir / "task_result.json"
    task = _task_by_id(f"mfe{horizon}_{year}_{variant}_outer")
    if _task_complete(
        result_path,
        task=task,
        study=study,
        output_root=output_root,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    tuning_task = _task_by_id(f"mfe{horizon}_{year}_baseline_tuning")
    tuning_path = _task_result_path(output_root, tuning_task)
    if not _task_complete(
        tuning_path,
        task=tuning_task,
        study=study,
        output_root=output_root,
    ):
        raise RuntimeError(
            f"iteration tuning must complete before outer task: {tuning_task['task_id']}"
        )
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    iterations = int(tuning["best_iteration"])
    extra = None
    extra_names: tuple[str, ...] = ()
    prepared = None
    if variant != "baseline":
        prepared = load_prepared_family(
            output_root,
            variant,
            candidate_count=inputs.candidate_count,
        )
        extra = prepared.open()
        extra_names = prepared.feature_names
    fold = inputs.common_path_rows(year, horizon)
    datasets, _evaluation_weight, evaluation_raw = _build_datasets(
        study=study,
        inputs=inputs,
        train_rows=fold.train_rows,
        evaluation_rows=fold.evaluation_rows,
        horizon=horizon,
        extra=extra,
        extra_names=extra_names,
    )
    parameters, _maximum_rounds, _patience = _model_parameters(study)
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        datasets.train_set,
        num_boost_round=iterations,
        valid_sets=[datasets.evaluation_set],
        valid_names=["outer_evaluation"],
    )
    elapsed = float(time.perf_counter() - started)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.txt"
    model.save_model(str(model_path), num_iteration=iterations)
    prediction = _predict(model, datasets.evaluation_sequence, iterations)
    prediction_path = output_dir / "prediction.npy"
    _save_npy(prediction_path, prediction)
    evaluation_rows_path = output_dir / "evaluation_rows.npy"
    _save_npy(evaluation_rows_path, datasets.evaluation_rows)
    daily, metrics = base.regression_metrics(
        date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
        actual=evaluation_raw,
        prediction=prediction,
        date_values=inputs.date_values,
        horizon=horizon,
    )
    daily_path = output_dir / "daily_metrics.parquet"
    daily.to_parquet(daily_path, index=False, compression="zstd")
    result = {
        "schema": TASK_RESULT_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": task["task_id"],
        "target": "mfe",
        "purge_days": int(horizon),
        "stage": "outer_evaluation",
        "variant": variant,
        "fold_year": int(year),
        "horizon": int(horizon),
        "train_row_count": int(len(fold.train_rows)),
        "evaluation_row_count": int(len(fold.evaluation_rows)),
        "fixed_iteration_source": {
            "inner_validation_year": int(tuning["inner_validation_year"]),
            "best_iteration": iterations,
            "task_id": tuning_task["task_id"],
        },
        "training_seconds": elapsed,
        "parameters": parameters,
        "feature_count": len(datasets.feature_names),
        "added_feature_count": len(extra_names),
        "added_feature_source": (
            None
            if prepared is None
            else {
                "family": prepared.family,
                "manifest": _family_manifest_path(
                    output_root, prepared.family
                ).relative_to(WORKSPACE_ROOT).as_posix(),
            }
        ),
        "metrics": metrics,
        "files": {
            "model": _record_for_file(model_path),
            "prediction": {
                **_record_for_file(prediction_path),
                "shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
            },
            "evaluation_rows": {
                **_record_for_file(evaluation_rows_path),
                "shape": list(datasets.evaluation_rows.shape),
                "dtype": str(datasets.evaluation_rows.dtype),
            },
            "daily_metrics": _record_for_file(daily_path),
        },
    }
    _atomic_write_json(result_path, result)
    _release_datasets(datasets)
    del model, prediction, fold, extra, prepared
    return result


def _require_training_inputs(
    study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
) -> None:
    for family in ALL_FAMILIES:
        load_prepared_family(
            output_root,
            family,
            candidate_count=inputs.candidate_count,
        )


def task_status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    groups: dict[str, list[str]] = {"completed": [], "pending": [], "failed": []}
    for task in _model_task_plan():
        path = _task_result_path(output_root, task)
        if _task_complete(
            path,
            task=task,
            study=study,
            output_root=output_root,
        ):
            state = "completed"
        elif path.is_file():
            state = "failed"
        else:
            state = "pending"
        groups[state].append(str(task["task_id"]))
    result = {
        "study_id": STUDY_ID,
        "total_task_count": len(_model_task_plan()),
        "completed_count": len(groups["completed"]),
        "pending_count": len(groups["pending"]),
        "failed_count": len(groups["failed"]),
        **groups,
    }
    return result


def run_task(
    *,
    task_id: str,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    task = _task_by_id(task_id)
    result_path = _task_result_path(output_root, task)
    if _task_complete(
        result_path,
        task=task,
        study=study,
        output_root=output_root,
    ):
        return {"task_id": task_id, "status": "completed", "skipped": True}
    try:
        inputs, _pack = _load_validated_inputs(study)
        if task["stage"] == "iteration_tuning":
            _run_tuning_task(
                study=study,
                inputs=inputs,
                output_root=output_root,
                year=int(task["fold_year"]),
                horizon=int(task["horizon"]),
            )
        else:
            if task["variant"] != "baseline":
                load_prepared_family(
                    output_root,
                    str(task["variant"]),
                    candidate_count=inputs.candidate_count,
                )
            _run_outer_task(
                study=study,
                inputs=inputs,
                output_root=output_root,
                year=int(task["fold_year"]),
                horizon=int(task["horizon"]),
                variant=str(task["variant"]),
            )
    except Exception as exc:
        failed = {
            "schema": TASK_RESULT_SCHEMA,
            "study_id": STUDY_ID,
            "task_id": task_id,
            "target": "mfe",
            "purge_days": int(task["horizon"]),
            "stage": task["stage"],
            "variant": task["variant"],
            "fold_year": int(task["fold_year"]),
            "horizon": int(task["horizon"]),
            "status": "failed",
            "failed_at": _now(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _atomic_write_json(result_path, failed)
        raise
    return {"task_id": task_id, "status": "completed", "skipped": False}


def run_pending(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = study_path.resolve()
    output_root = output_root.resolve()
    initial = task_status(study_path=study_path, output_root=output_root)
    runnable = set(initial["pending"]) | set(initial["failed"])
    completed_now: list[str] = []
    for task in _model_task_plan():
        task_id = str(task["task_id"])
        if task_id not in runnable:
            continue
        print(json.dumps({"event": "task_started", "task_id": task_id}), flush=True)
        command = [
            sys.executable,
            "-m",
            "daily_research.path_policy.seq100_mfe_feature_family_audit",
            "--config",
            str(study_path),
            "--output-root",
            str(output_root),
            "--task-id",
            task_id,
            "run-task",
        ]
        child = subprocess.run(command, cwd=WORKSPACE_ROOT, check=False)
        if child.returncode != 0:
            raise RuntimeError(
                f"task {task_id} exited with code {child.returncode}; rerun continues here"
            )
        completed_now.append(task_id)
        print(json.dumps({"event": "task_completed", "task_id": task_id}), flush=True)
    final = task_status(study_path=study_path, output_root=output_root)
    final["newly_completed"] = completed_now
    return final


def _load_outer_result(
    output_root: Path,
    *,
    year: int,
    horizon: int,
    variant: str,
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    task = _task_by_id(f"mfe{horizon}_{year}_{variant}_outer")
    result_path = _task_result_path(output_root, task)
    if not _task_complete(
        result_path,
        task=task,
        study=study,
        output_root=output_root,
    ):
        raise RuntimeError(f"outer task is incomplete: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    files = dict(result["files"])
    prediction = np.load(_resolve(files["prediction"]["path"]), allow_pickle=False)
    rows = np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False)
    return result, np.asarray(prediction, dtype=np.float32), np.asarray(rows, dtype=np.int64)


def _top_indices(values: np.ndarray, fraction: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    count = max(1, int(math.ceil(float(fraction) * len(source))))
    order = np.argsort(-source, kind="mergesort")
    return order[:count]


def paired_daily_increment(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    baseline_score: np.ndarray,
    variant_score: np.ndarray,
) -> pd.DataFrame:
    dates = np.asarray(date_idx, dtype=np.int32)
    outcome = np.asarray(actual, dtype=np.float64)
    baseline_values = np.asarray(baseline_score, dtype=np.float64)
    variant_values = np.asarray(variant_score, dtype=np.float64)
    valid = (
        np.isfinite(outcome)
        & np.isfinite(baseline_values)
        & np.isfinite(variant_values)
    )
    rows: list[dict[str, Any]] = []
    for date in np.unique(dates[valid]):
        mask = valid & (dates == int(date))
        current_actual = outcome[mask]
        current_baseline = baseline_values[mask]
        current_variant = variant_values[mask]
        if len(current_actual) < 20:
            continue
        baseline_ic = stats.spearmanr(
            current_baseline, current_actual
        ).statistic
        variant_ic = stats.spearmanr(current_variant, current_actual).statistic
        baseline_top = _top_indices(current_baseline, 0.05)
        variant_top = _top_indices(current_variant, 0.05)
        actual_tail = np.zeros(len(current_actual), dtype=bool)
        actual_tail[_top_indices(current_actual, 0.20)] = True
        rows.append(
            {
                "date_idx": int(date),
                "candidate_count": int(len(current_actual)),
                "rank_ic_delta": float(variant_ic - baseline_ic),
                "top5_mean_delta": float(
                    np.mean(current_actual[variant_top])
                    - np.mean(current_actual[baseline_top])
                ),
                "tail_top5_rate_delta": float(
                    np.mean(actual_tail[variant_top])
                    - np.mean(actual_tail[baseline_top])
                ),
            }
        )
    return pd.DataFrame(rows)


def _one_sided_hac(values: np.ndarray, lag: int) -> dict[str, Any]:
    result = base._hac_mean_test(values, maximum_lag=lag)
    count = int(result["count"])
    mean = float(result["mean"])
    standard_error = float(result["standard_error"])
    if count < 2 or not math.isfinite(mean):
        statistic = math.nan
        p_value = math.nan
    elif standard_error == 0.0:
        statistic = math.inf if mean > 0.0 else -math.inf if mean < 0.0 else 0.0
        p_value = float(stats.norm.sf(statistic))
    else:
        statistic = float(result["t_statistic"])
        p_value = (
            float(stats.norm.sf(statistic))
            if math.isfinite(statistic)
            else math.nan
        )
    result["t_statistic"] = statistic
    result["p_value_one_sided_positive"] = p_value
    return result


def _combined_stouffer(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    z_values: list[float] = []
    weights: list[float] = []
    for record in records:
        p_value = float(record["rank_ic_hac"]["p_value_one_sided_positive"])
        count = int(record["rank_ic_hac"]["count"])
        if not math.isfinite(p_value) or count <= 0:
            continue
        z_values.append(float(stats.norm.isf(np.clip(p_value, 1.0e-15, 1.0 - 1.0e-15))))
        weights.append(math.sqrt(float(count)))
    if not z_values:
        return {"z_statistic": math.nan, "p_value_one_sided": math.nan}
    weight = np.asarray(weights, dtype=np.float64)
    z = float(np.dot(weight, np.asarray(z_values)) / np.sqrt(np.dot(weight, weight)))
    return {"z_statistic": z, "p_value_one_sided": float(stats.norm.sf(z))}


def _benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    finite = sorted(
        (
            (key, float(value))
            for key, value in p_values.items()
            if math.isfinite(float(value))
        ),
        key=lambda item: item[1],
    )
    count = len(finite)
    result = {key: math.nan for key in p_values}
    running = 1.0
    for rank in range(count, 0, -1):
        key, p_value = finite[rank - 1]
        running = min(running, p_value * count / float(rank))
        result[key] = float(min(max(running, 0.0), 1.0))
    return result


def _classify_head(
    annual: Sequence[Mapping[str, Any]],
    *,
    fdr_q_value: float,
    thresholds: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    ic = [float(row["rank_ic_delta_mean"]) for row in annual]
    top5 = [float(row["top5_mean_delta_mean"]) for row in annual]
    tail = [float(row["tail_top5_rate_delta_mean"]) for row in annual]
    checks = {
        "fdr_q_pass": bool(
            math.isfinite(fdr_q_value)
            and fdr_q_value <= float(thresholds["maximum_fdr_q_value"])
        ),
        "positive_ic_years": int(sum(value > 0.0 for value in ic)),
        "median_ic_delta": float(np.median(ic)),
        "worst_ic_delta": float(min(ic)),
        "positive_top5_years": int(sum(value > 0.0 for value in top5)),
        "worst_top5_delta": float(min(top5)),
        "positive_tail_years": int(sum(value > 0.0 for value in tail)),
        "worst_tail_delta": float(min(tail)),
    }
    passed = (
        checks["fdr_q_pass"]
        and checks["positive_ic_years"]
        >= int(thresholds["minimum_positive_ic_years"])
        and checks["median_ic_delta"]
        >= float(thresholds["minimum_median_rank_ic_delta"])
        and checks["worst_ic_delta"]
        >= float(thresholds["minimum_worst_year_rank_ic_delta"])
        and checks["positive_top5_years"]
        >= int(thresholds["minimum_positive_top5_years"])
        and checks["worst_top5_delta"]
        >= float(thresholds["minimum_worst_year_top5_mean_delta"])
        and checks["positive_tail_years"]
        >= int(thresholds["minimum_positive_tail_years"])
        and checks["worst_tail_delta"]
        >= float(thresholds["minimum_worst_year_tail_rate_delta"])
    )
    return ("pass" if passed else "omit"), checks


def _compile_family_decision(
    *,
    head_status: Mapping[str, Mapping[str, str]],
    evidence: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    non_selections: Sequence[str],
) -> dict[str, Any]:
    control_invalid = any(
        value == "pass" for value in head_status[CONTROL_FAMILY].values()
    )
    family_status: dict[str, str] = {}
    for family in REAL_FAMILIES:
        passed = [
            horizon
            for horizon in TARGET_HORIZONS
            if head_status[family][str(horizon)] == "pass"
        ]
        if len(passed) == 2:
            family_status[family] = "core"
        elif len(passed) == 1:
            other = next(value for value in TARGET_HORIZONS if value not in passed)
            row = next(
                item
                for item in evidence
                if item["variant"] == family and int(item["horizon"]) == other
            )
            checks = row["gate_checks"]
            no_material_harm = (
                checks["worst_ic_delta"]
                >= float(thresholds["minimum_worst_year_rank_ic_delta"])
                and checks["worst_top5_delta"]
                >= float(thresholds["minimum_worst_year_top5_mean_delta"])
            )
            family_status[family] = "specialized" if no_material_harm else "omit"
        else:
            family_status[family] = "omit"
    advancing = any(value != "omit" for value in family_status.values())
    if control_invalid:
        decision_status = "invalid_negative_control"
    elif advancing:
        decision_status = "completed_feature_family_screen"
    else:
        decision_status = "completed_no_incremental_family"
    return {
        "status": decision_status,
        "family_status": family_status if not control_invalid else {},
        "head_status": {key: dict(value) for key, value in head_status.items()},
        "negative_control_passed_any_head": control_invalid,
        "next_step": (
            "audit_state10_only_for_non_omitted_families"
            if not control_invalid and advancing
            else "stop_and_diagnose_false_positive_control"
            if control_invalid
            else "stop_and_reassess_base_inputs"
        ),
        "does_not_select": list(non_selections),
    }


def evaluate_results(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    current = task_status(study_path=study_path, output_root=output_root)
    if current["completed_count"] != current["total_task_count"]:
        raise RuntimeError(
            "training is incomplete: "
            f"{current['completed_count']} completed, "
            f"{current['pending_count']} pending, {current['failed_count']} failed"
        )
    inputs = _load_source_inputs(study)
    evidence: list[dict[str, Any]] = []
    for variant in ALL_FAMILIES:
        for horizon in TARGET_HORIZONS:
            annual: list[dict[str, Any]] = []
            for year in FOLD_YEARS:
                _baseline_result, baseline_prediction, baseline_rows = _load_outer_result(
                    output_root,
                    year=year,
                    horizon=horizon,
                    variant="baseline",
                    study=study,
                )
                _variant_result, variant_prediction, variant_rows = _load_outer_result(
                    output_root,
                    year=year,
                    horizon=horizon,
                    variant=variant,
                    study=study,
                )
                if not np.array_equal(baseline_rows, variant_rows):
                    raise ValueError("baseline and variant evaluation rows differ")
                actual = np.asarray(
                    inputs.label_values("mfe", horizon)[baseline_rows],
                    dtype=np.float32,
                )
                daily = paired_daily_increment(
                    date_idx=inputs.candidate_date_idx[baseline_rows],
                    actual=actual,
                    baseline_score=baseline_prediction,
                    variant_score=variant_prediction,
                )
                daily_path = (
                    output_root
                    / "evaluation"
                    / variant
                    / f"h{horizon:02d}"
                    / f"fold_{year}_paired_daily.parquet"
                )
                daily_path.parent.mkdir(parents=True, exist_ok=True)
                daily.to_parquet(daily_path, index=False, compression="zstd")
                annual.append(
                    {
                        "fold_year": int(year),
                        "daily_count": int(len(daily)),
                        "rank_ic_delta_mean": float(daily["rank_ic_delta"].mean()),
                        "top5_mean_delta_mean": float(
                            daily["top5_mean_delta"].mean()
                        ),
                        "tail_top5_rate_delta_mean": float(
                            daily["tail_top5_rate_delta"].mean()
                        ),
                        "rank_ic_hac": _one_sided_hac(
                            daily["rank_ic_delta"].to_numpy(), horizon - 1
                        ),
                        "top5_mean_hac": _one_sided_hac(
                            daily["top5_mean_delta"].to_numpy(), horizon - 1
                        ),
                        "paired_daily": _record_for_file(daily_path),
                    }
                )
            combined = _combined_stouffer(annual)
            evidence.append(
                {
                    "variant": variant,
                    "horizon": int(horizon),
                    "annual": annual,
                    "combined_rank_ic_test": combined,
                }
            )
    p_values = {
        f"{row['variant']}__h{int(row['horizon']):02d}": float(
            row["combined_rank_ic_test"]["p_value_one_sided"]
        )
        for row in evidence
    }
    q_values = _benjamini_hochberg(p_values)
    thresholds = dict(study["decision"]["head_thresholds"])
    head_status: dict[str, dict[str, Any]] = {}
    for row in evidence:
        key = f"{row['variant']}__h{int(row['horizon']):02d}"
        status, checks = _classify_head(
            row["annual"],
            fdr_q_value=q_values[key],
            thresholds=thresholds,
        )
        row["fdr_q_value"] = q_values[key]
        row["status"] = status
        row["gate_checks"] = checks
        head_status.setdefault(str(row["variant"]), {})[
            str(int(row["horizon"]))
        ] = status
    decision = _compile_family_decision(
        head_status=head_status,
        evidence=evidence,
        thresholds=thresholds,
        non_selections=study["non_selections"],
    )
    decision_status = str(decision["status"])
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": decision_status,
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "scope": {
            "fold_years": list(FOLD_YEARS),
            "fold_role": study["folds"]["fold_role"],
            "primary_horizons": list(TARGET_HORIZONS),
            "maximum_consumed_outcome_date": study["folds"][
                "maximum_outcome_date"
            ],
        },
        "evidence": evidence,
        "decision": decision,
    }
    _atomic_write_json(output_root / "summary.json", summary)
    _atomic_write_json(output_root / "decision.json", decision)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit incremental causal feature families for Seq100 MFE D10/D20."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--family", choices=ALL_FAMILIES)
    parser.add_argument("--task-id")
    parser.add_argument(
        "command",
        choices=("status", "prepare", "run", "evaluate", "run-task"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    study_path = args.config.resolve()
    output_root = args.output_root.resolve()
    if args.command == "status":
        result = task_status(study_path=study_path, output_root=output_root)
    elif args.command == "prepare":
        result = prepare_features(
            study_path=study_path,
            output_root=output_root,
            family=args.family,
        )
    elif args.command == "run":
        result = run_pending(study_path=study_path, output_root=output_root)
    elif args.command == "run-task":
        if not args.task_id:
            raise ValueError("--task-id is required for run-task")
        result = run_task(
            task_id=str(args.task_id),
            study_path=study_path,
            output_root=output_root,
        )
    else:
        result = evaluate_results(study_path=study_path, output_root=output_root)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
