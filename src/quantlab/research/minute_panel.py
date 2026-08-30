"""Build causal morning features and executable minute windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantlab.data.qdp_v2.active import resolve_active_domain

FEATURE_NAMES = (
    "morning_return",
    "last_5m_return",
    "morning_range",
    "morning_volatility",
    "close_location",
    "volume_acceleration",
    "last_5m_amount_log",
)
DEFAULT_DECISION_BAR = "100000000"
DEFAULT_ENTRY_BAR = "100100000"
DEFAULT_ENTRY_END_BAR = "101000000"
REFERENCE_COLUMNS = (
    "symbol",
    "trade_date",
    "adjust_factor",
    "daily_close",
    "is_st",
    "is_suspended",
    "is_delisted",
)


class MinuteResearchError(ValueError):
    """Raised when minute inputs violate the research contract."""


@dataclass(frozen=True)
class MinuteReference:
    frame: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MinuteDataset:
    panel: pd.DataFrame
    windows: pd.DataFrame
    quality: dict[str, Any]


def _domain_frame(
    connection: duckdb.DuckDBPyConnection,
    *,
    domain: str,
    columns: tuple[str, ...],
    keys: pd.DataFrame,
    workspace_root: str | Path | None,
) -> tuple[pd.DataFrame, str]:
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    relation = connection.from_parquet(
        [str(path) for path in context.shard_paths],
        union_by_name=True,
    )
    view_name = f"active_{domain}"
    relation.create_view(view_name)
    connection.register("wanted_minute_keys", keys)
    selected = ", ".join(f"q.{column}" for column in columns)
    frame = connection.execute(
        f"SELECT {selected} FROM {view_name} q "
        "JOIN wanted_minute_keys k USING(symbol, trade_date) "
        "ORDER BY q.symbol, q.trade_date"
    ).fetchdf()
    if frame.duplicated(["symbol", "trade_date"]).any():
        raise MinuteResearchError(f"qdp_reference_duplicate_keys:{domain}")
    return frame, context.dataset_id


def load_minute_reference(
    keys: pd.DataFrame,
    *,
    workspace_root: str | Path | None = None,
) -> MinuteReference:
    """Load adjustment factors and same-day status for selected stock-days."""

    required = {"symbol", "trade_date"}
    if not required.issubset(keys.columns):
        raise MinuteResearchError("minute_reference_keys_incomplete")
    wanted = keys.loc[:, ["symbol", "trade_date"]].copy()
    wanted["symbol"] = wanted["symbol"].astype(str)
    wanted["trade_date"] = wanted["trade_date"].astype(str)
    wanted = wanted.drop_duplicates().sort_values(["symbol", "trade_date"], kind="stable")
    with duckdb.connect(":memory:") as connection:
        factors, factor_id = _domain_frame(
            connection,
            domain="adjust_factor",
            columns=("symbol", "trade_date", "adjust_factor"),
            keys=wanted,
            workspace_root=workspace_root,
        )
        status, status_id = _domain_frame(
            connection,
            domain="security_status",
            columns=("symbol", "trade_date", "is_st", "is_suspended", "is_delisted"),
            keys=wanted,
            workspace_root=workspace_root,
        )
        daily, daily_id = _domain_frame(
            connection,
            domain="market_daily_raw",
            columns=("symbol", "trade_date", "close"),
            keys=wanted,
            workspace_root=workspace_root,
        )
    daily = daily.rename(columns={"close": "daily_close"})
    result = wanted.merge(factors, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    result = result.merge(status, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    result = result.merge(daily, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    valid_factor = np.isfinite(pd.to_numeric(result["adjust_factor"], errors="coerce")) & result[
        "adjust_factor"
    ].gt(0.0)
    status_known = result[["is_st", "is_suspended", "is_delisted"]].notna().all(axis=1)
    metadata = {
        "requested_key_count": int(len(wanted)),
        "factor_dataset_id": factor_id,
        "status_dataset_id": status_id,
        "daily_dataset_id": daily_id,
        "valid_factor_count": int(valid_factor.sum()),
        "valid_factor_rate": float(valid_factor.mean()) if len(result) else 0.0,
        "known_status_count": int(status_known.sum()),
        "known_status_rate": float(status_known.mean()) if len(result) else 0.0,
        "daily_close_count": int(result["daily_close"].notna().sum()),
        "daily_close_rate": float(result["daily_close"].notna().mean()) if len(result) else 0.0,
        "st_count": int(result["is_st"].eq(True).sum()),
        "suspended_count": int(result["is_suspended"].eq(True).sum()),
        "delisted_count": int(result["is_delisted"].eq(True).sum()),
    }
    return MinuteReference(frame=result.loc[:, list(REFERENCE_COLUMNS)], metadata=metadata)


def _normalize_bars(bars: pd.DataFrame, *, last_bar: str) -> pd.DataFrame:
    required = {"symbol", "trade_date", "bar_time", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise MinuteResearchError(f"minute_panel_columns_missing:{','.join(missing)}")
    working = bars.loc[:, sorted(required)].copy()
    working["symbol"] = working["symbol"].astype(str)
    working["trade_date"] = working["trade_date"].astype(str)
    working["bar_time"] = working["bar_time"].astype(str)
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    working[numeric] = working[numeric].apply(pd.to_numeric, errors="coerce")
    working[numeric] = working[numeric].replace([np.inf, -np.inf], np.nan)
    working = working.loc[
        (working["bar_time"] > "093000000") & (working["bar_time"] <= str(last_bar))
    ].copy()
    if working.duplicated(["symbol", "trade_date", "bar_time"]).any():
        raise MinuteResearchError("minute_bars_duplicate_keys")
    working = working.sort_values(["symbol", "trade_date", "bar_time"], kind="stable")
    if working.empty:
        raise MinuteResearchError("minute_bars_are_empty")
    return working


def _morning_features(working: pd.DataFrame, *, decision_bar: str) -> pd.DataFrame:
    keys = ["symbol", "trade_date"]
    decision = working.loc[working["bar_time"] <= str(decision_bar)].copy()
    grouped = decision.groupby(keys, sort=True, observed=True)
    decision["position"] = grouped.cumcount()
    decision["group_size"] = grouped["close"].transform("size")
    close = decision["close"].astype(float)
    decision["log_return"] = np.log(close.where(close > 0.0)).groupby(
        [decision["symbol"], decision["trade_date"]]
    ).diff()
    summary = grouped.agg(
        feature_bar_count=("bar_time", "size"),
        first_close=("close", "first"),
        last_close=("close", "last"),
        morning_high=("high", "max"),
        morning_low=("low", "min"),
        morning_volatility=("log_return", "std"),
    )
    tail6 = decision.loc[decision["position"] >= decision["group_size"] - 6]
    tail6_summary = tail6.groupby(keys, sort=True, observed=True)["close"].agg(
        close_5m_ago="first", last_close_tail="last"
    )
    recent = decision.loc[decision["position"] >= decision["group_size"] - 5]
    recent_summary = recent.groupby(keys, sort=True, observed=True).agg(
        recent_volume=("volume", "mean"),
        last_5m_amount_log=("amount", lambda values: float(np.log1p(np.maximum(values, 0.0)).mean())),
    )
    previous = decision.loc[
        (decision["position"] >= decision["group_size"] - 25)
        & (decision["position"] < decision["group_size"] - 5)
    ]
    previous_volume = previous.groupby(keys, sort=True, observed=True)["volume"].mean().rename("previous_volume")
    last = grouped.tail(1).set_index(keys)[["high", "low", "close"]].rename(
        columns={"high": "last_high", "low": "last_low", "close": "last_bar_close"}
    )
    features = summary.join([tail6_summary, recent_summary, previous_volume, last], how="inner")
    features["morning_return"] = features["last_close"] / features["first_close"] - 1.0
    features["last_5m_return"] = features["last_close_tail"] / features["close_5m_ago"] - 1.0
    features["morning_range"] = features["morning_high"] / features["morning_low"] - 1.0
    denominator = features["last_high"] - features["last_low"]
    features["close_location"] = np.where(
        denominator > 0.0,
        (features["last_bar_close"] - features["last_low"]) / denominator,
        0.5,
    )
    features["volume_acceleration"] = features["recent_volume"] / features["previous_volume"] - 1.0
    return features.reset_index()


def _execution_windows(
    working: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    entry_bar: str,
    entry_end_bar: str,
    expected_session_bars: int,
) -> pd.DataFrame:
    keys = ["symbol", "trade_date"]
    session_counts = working.groupby(keys, sort=True).size().rename("session_bar_count")
    selected = working.loc[
        (working["bar_time"] >= str(entry_bar)) & (working["bar_time"] <= str(entry_end_bar))
    ]
    windows = selected.groupby(keys, sort=True, observed=True).agg(
        window_bar_count=("bar_time", "size"),
        window_open=("open", "first"),
        window_high=("high", "max"),
        window_low=("low", "min"),
        window_close=("close", "last"),
        window_volume=("volume", "sum"),
        window_amount=("amount", "sum"),
    ).reset_index()
    windows["window_price"] = windows["window_amount"] / windows["window_volume"]
    windows = windows.merge(session_counts.reset_index(), on=keys, how="left", validate="one_to_one")
    reference_with_prior = reference.sort_values(keys, kind="stable").copy()
    reference_with_prior["previous_close"] = reference_with_prior.groupby("symbol", sort=False)[
        "daily_close"
    ].transform(lambda values: values.ffill().shift())
    windows = windows.merge(reference_with_prior, on=keys, how="left", validate="one_to_one")
    factor_valid = np.isfinite(windows["adjust_factor"]) & windows["adjust_factor"].gt(0.0)
    status_known = windows[["is_suspended", "is_delisted"]].notna().all(axis=1)
    numeric_columns = [
        "window_open",
        "window_high",
        "window_low",
        "window_close",
        "window_volume",
        "window_amount",
        "window_price",
    ]
    numeric_valid = np.isfinite(windows[numeric_columns].to_numpy(dtype=np.float64)).all(axis=1)
    positive = windows[["window_open", "window_high", "window_low", "window_close"]].gt(0.0).all(axis=1)
    liquid = windows["window_volume"].gt(0.0) & windows["window_amount"].gt(0.0)
    one_price = np.isclose(windows["window_high"], windows["window_low"], rtol=0.0, atol=1.0e-12)
    one_price_down = (
        one_price
        & windows["previous_close"].notna()
        & windows["window_price"].lt(windows["previous_close"] * (1.0 - 1.0e-6))
    )
    complete = windows["session_bar_count"].eq(int(expected_session_bars)) & windows["window_bar_count"].eq(10)
    active = windows["is_suspended"].eq(False).fillna(False) & windows["is_delisted"].eq(False).fillna(False)
    windows["one_price_window"] = one_price
    windows["one_price_down_window"] = one_price_down
    base_executable = complete & factor_valid & status_known & numeric_valid & positive & liquid & active
    windows["entry_executable"] = base_executable & ~one_price
    windows["exit_executable"] = base_executable & ~one_price_down
    windows["entry_reason"] = np.select(
        [
            ~complete,
            ~factor_valid,
            ~status_known,
            ~numeric_valid | ~positive | ~liquid,
            one_price,
            ~active,
        ],
        ["incomplete", "factor_missing", "status_unknown", "invalid_bar", "one_price", "inactive"],
        default="",
    )
    windows["exit_reason"] = np.select(
        [
            ~complete,
            ~factor_valid,
            ~status_known,
            ~numeric_valid | ~positive | ~liquid,
            one_price_down,
            ~active,
        ],
        ["incomplete", "factor_missing", "status_unknown", "invalid_bar", "one_price_down", "inactive"],
        default="",
    )
    windows["adjusted_price"] = windows["window_price"] * windows["adjust_factor"]
    return windows.sort_values(["trade_date", "symbol"], kind="stable").reset_index(drop=True)


def _eligible_features(
    features: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    expected_feature_bars: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    merged = features.merge(reference, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    factor_valid = np.isfinite(merged["adjust_factor"]) & merged["adjust_factor"].gt(0.0)
    status_known = merged[["is_st", "is_suspended", "is_delisted"]].notna().all(axis=1)
    active = (
        merged["is_st"].eq(False).fillna(False)
        & merged["is_suspended"].eq(False).fillna(False)
        & merged["is_delisted"].eq(False).fillna(False)
    )
    finite = np.isfinite(merged[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)).all(axis=1)
    positive = merged[["first_close", "last_close", "morning_high", "morning_low"]].gt(0.0).all(axis=1)
    complete = merged["feature_bar_count"].eq(int(expected_feature_bars))
    keep = factor_valid & status_known & active & finite & positive & complete
    quality = {
        "raw_feature_stock_days": int(len(merged)),
        "incomplete_feature_stock_days": int((~complete).sum()),
        "invalid_feature_stock_days": int((~finite | ~positive).sum()),
        "missing_factor_feature_stock_days": int((~factor_valid).sum()),
        "unknown_status_feature_stock_days": int((~status_known).sum()),
        "st_feature_stock_days": int(merged["is_st"].eq(True).sum()),
        "suspended_feature_stock_days": int(merged["is_suspended"].eq(True).sum()),
        "delisted_feature_stock_days": int(merged["is_delisted"].eq(True).sum()),
        "eligible_feature_stock_days": int(keep.sum()),
    }
    return merged.loc[keep].copy(), quality


def _next_market_dates(values: pd.Series) -> dict[str, str]:
    dates = sorted(values.astype(str).unique().tolist())
    return {current: following for current, following in zip(dates[:-1], dates[1:], strict=True)}


def _attach_outcomes(panel: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    outcome_columns = {
        "label_exit_date": object,
        "label_exit_price": float,
        "label_exit_adjust_factor": float,
        "label_exit_adjusted_price": float,
        "label_exit_window_amount": float,
        "label_exit_window_volume": float,
    }
    for column, dtype in outcome_columns.items():
        result[column] = pd.Series(index=result.index, dtype=dtype)
    valid_windows = windows.loc[windows["exit_executable"]].sort_values(
        ["symbol", "trade_date"], kind="stable"
    )
    by_symbol = {str(symbol): group.reset_index(drop=True) for symbol, group in valid_windows.groupby("symbol")}
    for symbol, rows in result.loc[result["entry_filled"]].groupby("symbol", sort=False):
        available = by_symbol.get(str(symbol))
        if available is None or available.empty:
            continue
        dates = available["trade_date"].astype(str).to_numpy()
        positions = np.searchsorted(dates, rows["planned_exit_date"].astype(str).to_numpy(), side="left")
        valid = positions < len(available)
        if not valid.any():
            continue
        row_indexes = rows.index.to_numpy()[valid]
        selected = available.iloc[positions[valid]]
        result.loc[row_indexes, "label_exit_date"] = selected["trade_date"].astype(str).to_numpy()
        mapping = {
            "label_exit_price": "window_price",
            "label_exit_adjust_factor": "adjust_factor",
            "label_exit_adjusted_price": "adjusted_price",
            "label_exit_window_amount": "window_amount",
            "label_exit_window_volume": "window_volume",
        }
        for target, source in mapping.items():
            result.loc[row_indexes, target] = selected[source].to_numpy(dtype=np.float64)
    result["entry_adjusted_price"] = result["entry_price"] * result["entry_adjust_factor"]
    observed = result["entry_filled"] & result["label_exit_adjusted_price"].notna()
    result["label_gross_return"] = np.nan
    result.loc[observed, "label_gross_return"] = (
        result.loc[observed, "label_exit_adjusted_price"]
        / result.loc[observed, "entry_adjusted_price"]
        - 1.0
    )
    result["label_observed"] = observed
    return result


def build_minute_dataset(
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    decision_bar: str = DEFAULT_DECISION_BAR,
    entry_bar: str = DEFAULT_ENTRY_BAR,
    entry_end_bar: str = DEFAULT_ENTRY_END_BAR,
    expected_feature_bars: int = 30,
    expected_session_bars: int = 40,
) -> MinuteDataset:
    """Build candidates without filtering on future entry-window outcomes."""

    last_bar = max(str(decision_bar), str(entry_end_bar))
    working = _normalize_bars(bars, last_bar=last_bar)
    ref = reference.loc[:, list(REFERENCE_COLUMNS)].copy()
    if ref.duplicated(["symbol", "trade_date"]).any():
        raise MinuteResearchError("minute_reference_duplicate_keys")
    features = _morning_features(working, decision_bar=decision_bar)
    features, feature_quality = _eligible_features(
        features,
        ref,
        expected_feature_bars=expected_feature_bars,
    )
    windows = _execution_windows(
        working,
        ref,
        entry_bar=entry_bar,
        entry_end_bar=entry_end_bar,
        expected_session_bars=expected_session_bars,
    )
    next_dates = _next_market_dates(working["trade_date"])
    panel = features.rename(columns={"trade_date": "signal_date"})
    panel["planned_exit_date"] = panel["signal_date"].map(next_dates)
    panel = panel.loc[panel["planned_exit_date"].notna()].copy()
    entry = windows.rename(
        columns={
            "trade_date": "signal_date",
            "window_price": "entry_price",
            "adjust_factor": "entry_adjust_factor",
            "adjusted_price": "entry_adjusted_price",
            "window_volume": "entry_window_volume",
            "window_amount": "entry_window_amount",
            "entry_executable": "entry_filled",
            "entry_reason": "entry_unfilled_reason",
            "one_price_window": "entry_one_price_window",
        }
    )
    entry_columns = [
        "symbol",
        "signal_date",
        "entry_price",
        "entry_adjust_factor",
        "entry_adjusted_price",
        "entry_window_volume",
        "entry_window_amount",
        "entry_filled",
        "entry_unfilled_reason",
        "entry_one_price_window",
    ]
    panel = panel.merge(entry[entry_columns], on=["symbol", "signal_date"], how="left", validate="one_to_one")
    panel["entry_filled"] = panel["entry_filled"].fillna(False).astype(bool)
    panel["entry_unfilled_reason"] = panel["entry_unfilled_reason"].fillna("window_missing")
    panel["entry_one_price_window"] = panel["entry_one_price_window"].fillna(False).astype(bool)
    panel = _attach_outcomes(panel, windows)
    panel = panel.sort_values(["signal_date", "symbol"], kind="stable").reset_index(drop=True)
    if panel.empty:
        raise MinuteResearchError("minute_panel_is_empty")
    zero_bar = working[["open", "high", "low", "close", "volume", "amount"]].fillna(0.0).eq(0.0).all(axis=1)
    quality = {
        **feature_quality,
        "source_bar_count": int(len(working)),
        "source_symbol_count": int(working["symbol"].nunique()),
        "source_date_count": int(working["trade_date"].nunique()),
        "zero_placeholder_bar_count": int(zero_bar.sum()),
        "execution_window_count": int(len(windows)),
        "executable_entry_window_count": int(windows["entry_executable"].sum()),
        "executable_exit_window_count": int(windows["exit_executable"].sum()),
        "one_price_window_count": int(windows["one_price_window"].sum()),
        "one_price_down_window_count": int(windows["one_price_down_window"].sum()),
        "candidate_count": int(len(panel)),
        "entry_fill_count": int(panel["entry_filled"].sum()),
        "entry_fill_rate": float(panel["entry_filled"].mean()),
        "observed_label_count": int(panel["label_observed"].sum()),
        "observed_label_rate": float(panel["label_observed"].mean()),
        "label_price_mode": "raw_price_times_qdp_back_adjust_factor",
        "candidate_policy": "known non-ST active status at the 10:00 decision; entry outcome does not filter candidates",
    }
    return MinuteDataset(panel=panel, windows=windows, quality=quality)


def build_minute_panel(
    bars: pd.DataFrame,
    reference: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """Compatibility wrapper returning only the model panel."""

    return build_minute_dataset(bars, reference, **kwargs).panel


__all__ = [
    "DEFAULT_DECISION_BAR",
    "DEFAULT_ENTRY_BAR",
    "DEFAULT_ENTRY_END_BAR",
    "FEATURE_NAMES",
    "MinuteDataset",
    "MinuteReference",
    "MinuteResearchError",
    "build_minute_dataset",
    "build_minute_panel",
    "load_minute_reference",
]
