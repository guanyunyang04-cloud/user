from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.continuous_policy.pipeline_utils import select_feature_columns
from daily_research.continuous_policy.portfolio_simulator import PortfolioState
from daily_research.continuous_policy.runtime import write_json
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs, build_cross_section_state
from daily_research.path_policy.labels import (
    PATH20_CUMULATIVE_HORIZONS,
    PATH20_HORIZON,
    build_path20_labels,
)


NON_FORECAST_FEATURE_COLUMNS = {
    "date",
    "stock",
    "in_universe",
}


@dataclass(frozen=True)
class ForecastSequenceDataset:
    x: np.ndarray
    y_daily_excess: np.ndarray
    y_cum_excess: np.ndarray
    y_rank_20d: np.ndarray
    y_max_drawdown_20d: np.ndarray
    y_worst_1d_20d: np.ndarray
    y_upside_20d: np.ndarray
    date: np.ndarray
    stock: np.ndarray
    role: np.ndarray
    sequence_start_dates: np.ndarray
    label_end_dates: np.ndarray
    feature_columns: list[str]
    normalization_manifest: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def dates_by_role(self) -> dict[str, list[pd.Timestamp]]:
        result: dict[str, list[pd.Timestamp]] = {}
        for role_name in sorted({str(item) for item in self.role.tolist()}):
            mask = self.role == role_name
            result[role_name] = sorted({pd.Timestamp(item).normalize() for item in self.date[mask].tolist()})
        return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _role_for_year(
    year: int,
    *,
    train_start_year: int,
    train_end_year: int,
    validation_year: int,
    test_year: int,
) -> str | None:
    if int(train_start_year) <= int(year) <= int(train_end_year):
        return "train"
    if int(year) == int(validation_year):
        return "validation"
    if int(year) == int(test_year):
        return "test"
    return None


def _date_role_boundaries(
    dates: list[pd.Timestamp],
    *,
    train_start_year: int,
    train_end_year: int,
    validation_year: int,
    test_year: int,
    purge_trading_days: int,
) -> dict[str, list[pd.Timestamp]]:
    by_role: dict[str, list[pd.Timestamp]] = {"train": [], "validation": [], "test": []}
    for dt in dates:
        role = _role_for_year(
            int(dt.year),
            train_start_year=train_start_year,
            train_end_year=train_end_year,
            validation_year=validation_year,
            test_year=test_year,
        )
        if role:
            by_role[role].append(dt)
    eligible: dict[str, list[pd.Timestamp]] = {}
    for role, role_dates in by_role.items():
        if len(role_dates) > int(purge_trading_days):
            eligible[role] = role_dates[: -int(purge_trading_days)]
        else:
            eligible[role] = []
    return eligible


def _state_feature_panels(
    prepared: PreparedPolicyInputs,
    dates: list[pd.Timestamp],
    *,
    max_feature_columns: int,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], list[str]]:
    empty_portfolio = PortfolioState()
    panels: dict[pd.Timestamp, pd.DataFrame] = {}
    feature_columns: list[str] = []
    for dt in dates:
        state_frame = build_cross_section_state(prepared, date=dt, portfolio_state=empty_portfolio)
        state_frame = state_frame.copy()
        if not feature_columns:
            feature_columns = [
                column
                for column in select_feature_columns(state_frame)
                if column not in NON_FORECAST_FEATURE_COLUMNS
            ][: max(int(max_feature_columns), 1)]
        numeric = (
            state_frame.set_index("stock")
            .reindex(index=list(prepared.universe), columns=feature_columns)
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        panels[pd.Timestamp(dt).normalize()] = numeric.astype(float)
    return panels, feature_columns


def _safe_label_value(frame: pd.DataFrame, date: pd.Timestamp, stock: str) -> float:
    try:
        value = frame.loc[date, stock]
    except KeyError:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _empty_dataset(
    *,
    feature_columns: list[str],
    lookback_days: int,
    horizon: int,
    manifest: dict[str, Any],
    normalization_manifest: dict[str, Any],
) -> ForecastSequenceDataset:
    x = np.empty((0, int(lookback_days), int(len(feature_columns))), dtype=np.float32)
    return ForecastSequenceDataset(
        x=x,
        y_daily_excess=np.empty((0, int(horizon)), dtype=np.float32),
        y_cum_excess=np.empty((0, len(PATH20_CUMULATIVE_HORIZONS)), dtype=np.float32),
        y_rank_20d=np.empty((0,), dtype=np.float32),
        y_max_drawdown_20d=np.empty((0,), dtype=np.float32),
        y_worst_1d_20d=np.empty((0,), dtype=np.float32),
        y_upside_20d=np.empty((0,), dtype=np.float32),
        date=np.array([], dtype=object),
        stock=np.array([], dtype=object),
        role=np.array([], dtype=object),
        sequence_start_dates=np.array([], dtype=object),
        label_end_dates=np.array([], dtype=object),
        feature_columns=list(feature_columns),
        normalization_manifest=dict(normalization_manifest),
        manifest=dict(manifest),
    )


def build_forecast_sequence_dataset(
    prepared: PreparedPolicyInputs,
    *,
    train_start_year: int = 2019,
    train_end_year: int = 2022,
    validation_year: int = 2023,
    test_year: int = 2024,
    lookback_days: int = 252,
    horizon: int = PATH20_HORIZON,
    execution_mode: str = "next_open",
    max_samples_per_role: int = 0,
    max_feature_columns: int = 96,
) -> ForecastSequenceDataset:
    lookback_days = int(lookback_days)
    horizon = int(horizon)
    if horizon != PATH20_HORIZON:
        raise ValueError("Path20 forecast dataset currently requires horizon=20.")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if int(train_start_year) > int(train_end_year):
        raise ValueError("train_start_year must be <= train_end_year.")

    dates = [pd.Timestamp(dt).normalize() for dt in prepared.close.index]
    date_to_pos = {dt: idx for idx, dt in enumerate(dates)}
    next_open_extra_day = 1 if str(execution_mode or "next_open").strip().lower() == "next_open" else 0
    label_forward_offset = horizon + next_open_extra_day
    eligible_dates_by_role = _date_role_boundaries(
        dates,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        validation_year=validation_year,
        test_year=test_year,
        purge_trading_days=label_forward_offset,
    )
    panels, feature_columns = _state_feature_panels(
        prepared,
        dates,
        max_feature_columns=max_feature_columns,
    )
    labels = build_path20_labels(prepared, execution_mode=execution_mode, horizon=horizon)

    x_rows: list[np.ndarray] = []
    y_daily_rows: list[list[float]] = []
    y_cum_rows: list[list[float]] = []
    y_rank_rows: list[float] = []
    y_drawdown_rows: list[float] = []
    y_worst_rows: list[float] = []
    y_upside_rows: list[float] = []
    date_rows: list[pd.Timestamp] = []
    stock_rows: list[str] = []
    role_rows: list[str] = []
    sequence_start_rows: list[pd.Timestamp] = []
    label_end_rows: list[pd.Timestamp] = []
    dropped_target_nan = 0
    dropped_missing_lookback = 0
    capped = int(max_samples_per_role) > 0
    sample_count_by_role = {"train": 0, "validation": 0, "test": 0}

    for role in ("train", "validation", "test"):
        for signal_dt in eligible_dates_by_role.get(role, []):
            signal_pos = date_to_pos.get(signal_dt)
            if signal_pos is None or signal_pos < lookback_days - 1:
                dropped_missing_lookback += len(prepared.universe)
                continue
            label_end_pos = signal_pos + label_forward_offset
            if label_end_pos >= len(dates):
                dropped_missing_lookback += len(prepared.universe)
                continue
            sequence_dates = dates[signal_pos - lookback_days + 1 : signal_pos + 1]
            membership = prepared.membership_frame.reindex(index=[signal_dt], columns=list(prepared.universe))
            membership_row = membership.iloc[0].fillna(False) if not membership.empty else pd.Series(False, index=prepared.universe)
            for stock in prepared.universe:
                if capped and sample_count_by_role[role] >= int(max_samples_per_role):
                    break
                if not bool(membership_row.get(stock, False)):
                    continue
                daily_target = [
                    _safe_label_value(labels.daily_excess_return[step], signal_dt, str(stock))
                    for step in range(1, horizon + 1)
                ]
                cum_target = [
                    _safe_label_value(labels.cumulative_excess_return[step], signal_dt, str(stock))
                    for step in PATH20_CUMULATIVE_HORIZONS
                ]
                rank_target = _safe_label_value(labels.forward_rank[horizon], signal_dt, str(stock))
                drawdown_target = _safe_label_value(labels.path_max_drawdown_20d, signal_dt, str(stock))
                worst_target = _safe_label_value(labels.path_worst_1d_20d, signal_dt, str(stock))
                upside_target = _safe_label_value(labels.path_upside_capture_20d, signal_dt, str(stock))
                all_targets = [*daily_target, *cum_target, rank_target, drawdown_target, worst_target, upside_target]
                if not np.isfinite(np.asarray(all_targets, dtype=float)).all():
                    dropped_target_nan += 1
                    continue
                try:
                    sequence = np.stack(
                        [
                            panels[dt].reindex(index=list(prepared.universe), columns=feature_columns).loc[str(stock)].to_numpy(
                                dtype=np.float32,
                                copy=False,
                            )
                            for dt in sequence_dates
                        ],
                        axis=0,
                    )
                except KeyError:
                    dropped_missing_lookback += 1
                    continue
                x_rows.append(sequence)
                y_daily_rows.append(daily_target)
                y_cum_rows.append(cum_target)
                y_rank_rows.append(rank_target)
                y_drawdown_rows.append(drawdown_target)
                y_worst_rows.append(worst_target)
                y_upside_rows.append(upside_target)
                date_rows.append(signal_dt)
                stock_rows.append(str(stock))
                role_rows.append(role)
                sequence_start_rows.append(sequence_dates[0])
                label_end_rows.append(dates[label_end_pos])
                sample_count_by_role[role] += 1

    normalization_manifest: dict[str, Any] = {
        "fit_role": "train_only",
        "method": "zscore",
        "feature_count": int(len(feature_columns)),
    }
    base_manifest: dict[str, Any] = {
        "status": "completed",
        "stage": "forecast_sequence_dataset",
        "lookback_days": int(lookback_days),
        "horizon": int(horizon),
        "execution_mode": str(execution_mode),
        "label_semantics": dict(labels.metadata),
        "role_years": {
            "train_start_year": int(train_start_year),
            "train_end_year": int(train_end_year),
            "validation_year": int(validation_year),
            "test_year": int(test_year),
        },
        "role_purge_trading_days": int(label_forward_offset),
        "next_open_label_extra_trading_day": int(next_open_extra_day),
        "feature_columns": list(feature_columns),
        "sample_count_by_role": {key: int(value) for key, value in sample_count_by_role.items()},
        "dropped_target_nan": int(dropped_target_nan),
        "dropped_missing_lookback": int(dropped_missing_lookback),
        "max_samples_per_role": int(max_samples_per_role),
    }

    if not x_rows:
        base_manifest["status"] = "insufficient_or_incomplete"
        base_manifest["reason"] = "no_forecast_samples"
        return _empty_dataset(
            feature_columns=feature_columns,
            lookback_days=lookback_days,
            horizon=horizon,
            manifest=base_manifest,
            normalization_manifest=normalization_manifest,
        )

    x_raw = np.stack(x_rows, axis=0).astype(np.float32, copy=False)
    feature_nan_ratio = float(np.isnan(x_raw).mean()) if x_raw.size else 0.0
    roles_np = np.array(role_rows, dtype=object)
    train_mask = roles_np == "train"
    if bool(train_mask.any()):
        train_values = x_raw[train_mask].reshape(-1, x_raw.shape[-1])
        feature_mean = np.nanmean(train_values, axis=0)
        feature_std = np.nanstd(train_values, axis=0)
    else:
        feature_mean = np.zeros((x_raw.shape[-1],), dtype=np.float32)
        feature_std = np.ones((x_raw.shape[-1],), dtype=np.float32)
        base_manifest["status"] = "insufficient_or_incomplete"
        base_manifest["reason"] = "no_train_samples"
    feature_mean = np.where(np.isfinite(feature_mean), feature_mean, 0.0).astype(np.float32)
    feature_std = np.where(np.isfinite(feature_std) & (np.abs(feature_std) > 1.0e-8), feature_std, 1.0).astype(np.float32)
    x = ((x_raw - feature_mean.reshape(1, 1, -1)) / feature_std.reshape(1, 1, -1)).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    normalization_manifest.update(
        {
            "feature_mean": feature_mean.astype(float).tolist(),
            "feature_std": feature_std.astype(float).tolist(),
            "raw_feature_nan_ratio": feature_nan_ratio,
        }
    )
    base_manifest.update(
        {
            "sample_count": int(x.shape[0]),
            "feature_count": int(x.shape[-1]),
            "sequence_shape": [int(x.shape[0]), int(x.shape[1]), int(x.shape[2])],
            "target_shape": [int(x.shape[0]), int(horizon)],
            "normalization": normalization_manifest,
        }
    )

    return ForecastSequenceDataset(
        x=x,
        y_daily_excess=np.asarray(y_daily_rows, dtype=np.float32),
        y_cum_excess=np.asarray(y_cum_rows, dtype=np.float32),
        y_rank_20d=np.asarray(y_rank_rows, dtype=np.float32),
        y_max_drawdown_20d=np.asarray(y_drawdown_rows, dtype=np.float32),
        y_worst_1d_20d=np.asarray(y_worst_rows, dtype=np.float32),
        y_upside_20d=np.asarray(y_upside_rows, dtype=np.float32),
        date=np.array(date_rows, dtype=object),
        stock=np.array(stock_rows, dtype=object),
        role=roles_np,
        sequence_start_dates=np.array(sequence_start_rows, dtype=object),
        label_end_dates=np.array(label_end_rows, dtype=object),
        feature_columns=list(feature_columns),
        normalization_manifest=normalization_manifest,
        manifest=base_manifest,
    )


def save_forecast_sequence_dataset(dataset: ForecastSequenceDataset, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / "forecast_dataset.npz"
    manifest_path = root / "forecast_dataset_manifest.json"
    np.savez_compressed(
        npz_path,
        x=dataset.x,
        y_daily_excess=dataset.y_daily_excess,
        y_cum_excess=dataset.y_cum_excess,
        y_rank_20d=dataset.y_rank_20d,
        y_max_drawdown_20d=dataset.y_max_drawdown_20d,
        y_worst_1d_20d=dataset.y_worst_1d_20d,
        y_upside_20d=dataset.y_upside_20d,
        date=np.array([pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.date.tolist()], dtype=object),
        stock=dataset.stock.astype(str),
        role=dataset.role.astype(str),
        sequence_start_dates=np.array(
            [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.sequence_start_dates.tolist()],
            dtype=object,
        ),
        label_end_dates=np.array(
            [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dataset.label_end_dates.tolist()],
            dtype=object,
        ),
        feature_columns=np.array(dataset.feature_columns, dtype=object),
    )
    manifest = {
        **dataset.manifest,
        "dataset_npz": str(npz_path.resolve()),
        "manifest_json": str(manifest_path.resolve()),
        "normalization": dataset.normalization_manifest,
    }
    manifest = _json_ready(manifest)
    write_json(manifest_path, manifest)
    return manifest
