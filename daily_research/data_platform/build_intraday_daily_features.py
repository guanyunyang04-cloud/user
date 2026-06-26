from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.data_lake.canonical import DEFAULT_CANONICAL_START_DATE
from daily_research.data_lake.catalog import DEFAULT_DATA_LAKE_ROOT, ResearchDataLake
from daily_research.data_platform.contracts import (
    DataDomain,
    DOMAIN_STANDARD_COLUMNS,
    normalize_intraday_5m_frame,
    normalize_intraday_daily_features_frame,
    normalize_domain,
)


_PROGRESS_LOCK = threading.Lock()
_INTRADAY_5M_REQUIRED_COLUMNS = {
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}
INTRADAY_DAILY_FEATURE_CONTRACT_VERSION = "intraday_daily_features_v2"
LAST_5M_RET_POLICY = "last_close_to_previous_5m_close_return"


@dataclass(frozen=True)
class BuildIntradayDailyFeaturesConfig:
    lake_root: Path = DEFAULT_DATA_LAKE_ROOT
    source_dataset_id: str = ""
    start_date: str = DEFAULT_CANONICAL_START_DATE
    end_date: str = ""
    years: tuple[int, ...] = ()
    source_name: str = "external_1m_derived_5m"
    adjusted_flag: str = "none"
    workers: int = 1
    max_shards: int = 0
    progress_path: Path | None = None
    dry_run: bool = False
    resume: bool = True
    reuse: bool = True

    def normalized(self) -> "BuildIntradayDailyFeaturesConfig":
        return BuildIntradayDailyFeaturesConfig(
            lake_root=Path(self.lake_root),
            source_dataset_id=str(self.source_dataset_id or "").strip(),
            start_date=str(self.start_date or DEFAULT_CANONICAL_START_DATE),
            end_date=str(self.end_date or ""),
            years=tuple(sorted({int(item) for item in self.years})),
            source_name=str(self.source_name or "external_1m_derived_5m").strip().lower(),
            adjusted_flag=str(self.adjusted_flag or "none").strip().lower(),
            workers=max(1, int(self.workers or 1)),
            max_shards=max(0, int(self.max_shards or 0)),
            progress_path=Path(self.progress_path) if self.progress_path else None,
            dry_run=bool(self.dry_run),
            resume=bool(self.resume),
            reuse=bool(self.reuse),
        )


@dataclass(frozen=True)
class BuildIntradayDailyFeaturesResult:
    status: str
    dataset_id: str = ""
    shard_count: int = 0
    row_count: int = 0
    error_count: int = 0
    progress_path: Path | None = None
    errors: list[str] = field(default_factory=list)


def build_intraday_daily_features(config: BuildIntradayDailyFeaturesConfig) -> BuildIntradayDailyFeaturesResult:
    cfg = config.normalized()
    if not cfg.source_dataset_id:
        raise ValueError("source_dataset_id_required")
    lake = ResearchDataLake(cfg.lake_root)
    source_metadata = lake.describe_dataset(cfg.source_dataset_id)
    source_domain = normalize_domain(str(dict(source_metadata.get("parameters", {}) or {}).get("domain", "") or ""))
    if source_domain != DataDomain.MARKET_INTRADAY_5M:
        raise ValueError(f"source_domain_mismatch: {source_domain}; expected={DataDomain.MARKET_INTRADAY_5M}")
    source_manifest_path = Path(str(dict(source_metadata.get("content_paths", {}) or {}).get("shard_manifest", "") or ""))
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"source_shard_manifest_not_found: {source_manifest_path}")

    progress_path = cfg.progress_path or _default_progress_path(lake)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_shards = [
        dict(item)
        for item in list(source_manifest.get("shards", []) or [])
        if str(dict(item).get("status", "") or "") == "stored" and int(dict(item).get("row_count", 0) or 0) > 0
    ]
    if cfg.years:
        allowed_years = set(int(item) for item in cfg.years)
        source_shards = [record for record in source_shards if _shard_intersects_years(record, allowed_years)]
    if cfg.max_shards:
        source_shards = source_shards[: cfg.max_shards]

    target_spec = _feature_dataset_spec(cfg, source_metadata=source_metadata)
    identity_spec = {**target_spec, "domain": DataDomain.INTRADAY_DAILY_FEATURES, "sharded": True}
    identity = lake.build_domain_dataset_identity(domain=DataDomain.INTRADAY_DAILY_FEATURES, spec=identity_spec)
    shard_dir = Path(identity["dataset_dir"]) / "shards"
    _write_progress(
        progress_path,
        {
            "event": "start",
            "source_dataset_id": cfg.source_dataset_id,
            "source_shard_count": len(source_shards),
            "workers": cfg.workers,
            "dry_run": cfg.dry_run,
        },
    )
    if cfg.dry_run:
        _write_progress(progress_path, {"event": "completed", "dry_run": True, "shard_count": len(source_shards)})
        return BuildIntradayDailyFeaturesResult(
            status="dry_run",
            shard_count=len(source_shards),
            progress_path=progress_path,
        )

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if cfg.workers <= 1 or len(source_shards) <= 1:
        for idx, source_record in enumerate(source_shards, start=1):
            records.append(_build_one_shard(lake=lake, cfg=cfg, shard_dir=shard_dir, source_record=source_record, progress_path=progress_path, index=idx, total=len(source_shards)))
    else:
        with ThreadPoolExecutor(max_workers=min(cfg.workers, len(source_shards))) as executor:
            futures = {
                executor.submit(
                    _build_one_shard,
                    lake=lake,
                    cfg=cfg,
                    shard_dir=shard_dir,
                    source_record=source_record,
                    progress_path=progress_path,
                    index=idx,
                    total=len(source_shards),
                ): source_record
                for idx, source_record in enumerate(source_shards, start=1)
            }
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as exc:
                    source_record = futures[future]
                    message = f"{source_record.get('path', '')}: {exc}"
                    errors.append(message)
                    _write_progress(progress_path, {"event": "shard_error", "source_path": str(source_record.get("path", "") or ""), "error": str(exc)})

    records = sorted(records, key=lambda item: str(item.get("path", "")))
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.INTRADAY_DAILY_FEATURES,
        spec=target_spec,
        shard_records=records,
        source="intraday_daily_features_from_5m",
        reuse=cfg.reuse,
    )
    row_count = sum(int(item.get("row_count", 0) or 0) for item in records)
    error_count = len(errors) + sum(int(item.get("error_count", 0) or 0) for item in records)
    _write_progress(
        progress_path,
        {
            "event": "dataset_registered",
            "dataset_id": record.dataset_id,
            "shard_count": len(records),
            "row_count": row_count,
            "error_count": error_count,
        },
    )
    _write_progress(progress_path, {"event": "completed", "dataset_id": record.dataset_id})
    return BuildIntradayDailyFeaturesResult(
        status="completed",
        dataset_id=record.dataset_id,
        shard_count=len(records),
        row_count=row_count,
        error_count=error_count,
        progress_path=progress_path,
        errors=errors,
    )


def _build_one_shard(
    *,
    lake: ResearchDataLake,
    cfg: BuildIntradayDailyFeaturesConfig,
    shard_dir: Path,
    source_record: dict[str, Any],
    progress_path: Path,
    index: int,
    total: int,
) -> dict[str, Any]:
    source_path = Path(str(source_record.get("path", "") or ""))
    if not source_path.exists():
        raise FileNotFoundError(f"source_shard_not_found: {source_path}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    target_path = shard_dir / f"{_safe_stem(source_path.stem)}__intraday_daily_features.parquet"
    _write_progress(progress_path, {"event": "shard_start", "index": index, "total": total, "source_path": str(source_path.resolve())})
    if cfg.resume and target_path.exists():
        features = pd.read_parquet(target_path, columns=["trade_date", "symbol"])
        return _feature_shard_record(
            target_path=target_path,
            features=features,
            source_record=source_record,
            source_path=source_path,
            status="stored",
            materialization="resume_hit",
        )
    raw = pd.read_parquet(source_path)
    features = _build_intraday_daily_feature_frame_fast(raw, source=cfg.source_name, adjusted_flag=cfg.adjusted_flag)
    features.to_parquet(target_path, index=False)
    record = _feature_shard_record(
        target_path=target_path,
        features=features,
        source_record=source_record,
        source_path=source_path,
        status="stored",
        materialization="built",
    )
    _write_progress(
        progress_path,
        {
            "event": "shard_stored",
            "index": index,
            "total": total,
            "path": str(target_path.resolve()),
            "source_path": str(source_path.resolve()),
            "row_count": record["row_count"],
            "start_date": record["start_date"],
            "end_date": record["end_date"],
        },
    )
    return record


def _build_intraday_daily_feature_frame_fast(
    intraday_frame: pd.DataFrame,
    *,
    source: str,
    adjusted_flag: str,
) -> pd.DataFrame:
    bars = _prepare_intraday_5m_bars(intraday_frame, source=source, adjusted_flag=adjusted_flag)
    if bars.empty:
        return pd.DataFrame(columns=DOMAIN_STANDARD_COLUMNS[DataDomain.INTRADAY_DAILY_FEATURES])

    keys = ["trade_date", "symbol"]
    bars = bars.sort_values(keys + ["bar_time"]).reset_index(drop=True)
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric_columns:
        bars[column] = pd.to_numeric(bars[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    grouped = bars.groupby(keys, sort=True, dropna=False)
    bars["_pos"] = grouped.cumcount()
    bars["_revpos"] = grouped.cumcount(ascending=False)
    bars["_clock"] = pd.to_numeric(bars["bar_time"].astype(str).str[:6], errors="coerce").fillna(0).astype(int)

    base = grouped.agg(
        first_open=("open", "first"),
        last_close=("close", "last"),
        high_max=("high", "max"),
        low_min=("low", "min"),
        total_volume=("volume", "sum"),
        total_amount=("amount", "sum"),
        bar_count=("close", "size"),
    )
    out = pd.DataFrame(index=base.index)
    out["bar_count"] = base["bar_count"].astype(float)
    out["intraday_vwap"] = _safe_ratio(base["total_amount"], base["total_volume"])
    out["intraday_ret"] = _safe_return_series(base["last_close"], base["first_open"])
    out["close_to_vwap"] = _safe_return_series(base["last_close"], out["intraday_vwap"])
    out["intraday_range"] = _safe_return_series(base["high_max"], base["low_min"])
    out["close_position"] = (base["last_close"] - base["low_min"]) / (base["high_max"] - base["low_min"])
    out.loc[(base["high_max"] <= base["low_min"]) | base["last_close"].isna(), "close_position"] = np.nan
    out["open_to_high_ret"] = _safe_return_series(base["high_max"], base["first_open"])
    out["open_to_low_ret"] = _safe_return_series(base["low_min"], base["first_open"])
    out["high_to_close_ret"] = _safe_return_series(base["last_close"], base["high_max"])
    out["low_to_close_ret"] = _safe_return_series(base["last_close"], base["low_min"])

    pos0 = _rows_at(bars, bars["_pos"].eq(0), keys)
    pos2 = _rows_at(bars, bars["_pos"].eq(2), keys)
    pos5 = _rows_at(bars, bars["_pos"].eq(5), keys)
    last = _rows_at(bars, bars["_revpos"].eq(0), keys)
    tail5 = _rows_at(bars, bars["_revpos"].eq(5), keys)
    out["first_5m_ret"] = _safe_return_series(pos0["close"], pos0["open"]).reindex(out.index)
    out["first_15m_ret"] = _safe_return_series(pos2["close"], pos0["open"]).reindex(out.index)
    out["first_30m_ret"] = _safe_return_series(pos5["close"], pos0["open"]).reindex(out.index)
    out["last_5m_ret"] = _last_close_to_previous_close_return(bars, keys).reindex(out.index)
    out["last_30m_ret"] = _safe_return_series(last["close"], tail5["open"]).reindex(out.index)

    head6 = bars.loc[bars["_pos"].lt(6)]
    tail6 = bars.loc[bars["_revpos"].lt(6)]
    head6_amount = head6.groupby(keys, sort=True, dropna=False)["amount"].sum().reindex(out.index)
    tail6_amount = tail6.groupby(keys, sort=True, dropna=False)["amount"].sum().reindex(out.index)
    first_amount = pos0["amount"].reindex(out.index)
    last_amount = last["amount"].reindex(out.index)
    out["first_30m_amount_share"] = _safe_ratio(head6_amount, base["total_amount"])
    out["last_30m_amount_share"] = _safe_ratio(tail6_amount, base["total_amount"])
    out["first_5m_amount_share"] = _safe_ratio(first_amount, base["total_amount"])
    out["last_5m_amount_share"] = _safe_ratio(last_amount, base["total_amount"])

    head_range = head6.groupby(keys, sort=True, dropna=False).agg(high=("high", "max"), low=("low", "min")).reindex(out.index)
    tail_range = tail6.groupby(keys, sort=True, dropna=False).agg(high=("high", "max"), low=("low", "min")).reindex(out.index)
    enough = base["bar_count"].ge(6)
    out["first_30m_range"] = _safe_return_series(head_range["high"], head_range["low"])
    out["last_30m_range"] = _safe_return_series(tail_range["high"], tail_range["low"])
    out.loc[~enough, ["first_30m_range", "last_30m_range"]] = np.nan

    prev_close = base["last_close"].groupby(level="symbol", sort=False).shift(1)
    out["open_gap"] = _safe_return_series(base["first_open"], prev_close)
    gap_sign = np.sign(out["open_gap"])
    out["open_gap_first_30m_follow_through"] = gap_sign * out["first_30m_ret"]
    out["open_gap_first_30m_reversal"] = -gap_sign * out["first_30m_ret"]
    out.loc[out["open_gap"].isna() | out["first_30m_ret"].isna(), ["open_gap_first_30m_follow_through", "open_gap_first_30m_reversal"]] = np.nan

    close_ret = grouped["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    bars["_close_ret"] = close_ret
    out["intraday_realized_vol"] = close_ret.groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).std().reindex(out.index)
    out["intraday_price_volume_corr"] = _group_corr(bars, keys, "_close_ret", "volume").reindex(out.index)

    valid_high = bars["high"].notna()
    valid_low = bars["low"].notna()
    high_pos = _extreme_pos(bars.loc[valid_high], keys, "high", "max").reindex(out.index)
    low_pos = _extreme_pos(bars.loc[valid_low], keys, "low", "min").reindex(out.index)
    denom = (base["bar_count"] - 1).replace(0, np.nan)
    out["high_time_frac"] = high_pos / denom
    out["low_time_frac"] = low_pos / denom
    out.loc[base["bar_count"].eq(1) & high_pos.notna(), "high_time_frac"] = 0.0
    out.loc[base["bar_count"].eq(1) & low_pos.notna(), "low_time_frac"] = 0.0
    out["high_before_low"] = np.where(
        high_pos.notna() & low_pos.notna(),
        np.where(high_pos.eq(low_pos), 0.5, (high_pos < low_pos).astype(float)),
        np.nan,
    )

    valid_close_count = grouped["close"].count().reindex(out.index)
    running_max = grouped["close"].cummax()
    running_min = grouped["close"].cummin()
    drawdown = bars["close"] / running_max - 1.0
    runup = bars["close"] / running_min - 1.0
    out["intraday_max_drawdown"] = drawdown.groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).min().reindex(out.index)
    out["intraday_max_runup"] = runup.groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).max().reindex(out.index)
    out.loc[valid_close_count.lt(2), ["intraday_max_drawdown", "intraday_max_runup"]] = np.nan

    vwap_by_row = out["intraday_vwap"].reindex(pd.MultiIndex.from_frame(bars[keys])).to_numpy()
    above = pd.Series(bars["close"].to_numpy() > vwap_by_row, index=bars.index)
    out["price_above_vwap_share"] = above.groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).mean().reindex(out.index)
    out.loc[out["intraday_vwap"].isna(), "price_above_vwap_share"] = np.nan

    amount_share = _safe_ratio(bars["amount"], base["total_amount"].reindex(pd.MultiIndex.from_frame(bars[keys])).to_numpy())
    bars["_amount_share"] = amount_share.to_numpy() if isinstance(amount_share, pd.Series) else amount_share
    out["amount_top_bar_share"] = bars["_amount_share"].groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).max().reindex(out.index)
    out["amount_concentration_hhi"] = (bars["_amount_share"] ** 2).groupby([bars["trade_date"], bars["symbol"]], sort=True, dropna=False).sum().reindex(out.index)
    out.loc[base["total_amount"].le(0) | base["total_amount"].isna(), ["amount_top_bar_share", "amount_concentration_hhi"]] = np.nan

    cum_amount = grouped["amount"].cumsum()
    cum_volume = grouped["volume"].cumsum()
    bars["_cum_vwap"] = cum_amount / cum_volume.where(cum_volume > 0)
    out["cum_vwap_slope"] = _linear_slope_by_group(bars, keys, "_cum_vwap").reindex(out.index)

    am = bars.loc[bars["_clock"].le(113000)]
    pm = bars.loc[bars["_clock"].ge(130000)]
    am_first = _rows_at(am.assign(_session_pos=am.groupby(keys, sort=True, dropna=False).cumcount()), am.groupby(keys, sort=True, dropna=False).cumcount().eq(0), keys)
    am_last = _rows_at(am.assign(_session_revpos=am.groupby(keys, sort=True, dropna=False).cumcount(ascending=False)), am.groupby(keys, sort=True, dropna=False).cumcount(ascending=False).eq(0), keys)
    pm_first = _rows_at(pm.assign(_session_pos=pm.groupby(keys, sort=True, dropna=False).cumcount()), pm.groupby(keys, sort=True, dropna=False).cumcount().eq(0), keys)
    pm_last = _rows_at(pm.assign(_session_revpos=pm.groupby(keys, sort=True, dropna=False).cumcount(ascending=False)), pm.groupby(keys, sort=True, dropna=False).cumcount(ascending=False).eq(0), keys)
    out["am_ret"] = _safe_return_series(am_last["close"], am_first["open"]).reindex(out.index)
    out["pm_ret"] = _safe_return_series(pm_last["close"], pm_first["open"]).reindex(out.index)
    out["lunch_gap_ret"] = _safe_return_series(pm_first["open"], am_last["close"]).reindex(out.index)
    out["am_pm_ret_spread"] = out["pm_ret"] - out["am_ret"]
    am_close_ret = am.groupby(keys, sort=True, dropna=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    pm_close_ret = pm.groupby(keys, sort=True, dropna=False)["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    am_vol = am_close_ret.groupby([am["trade_date"], am["symbol"]], sort=True, dropna=False).std().reindex(out.index)
    pm_vol = pm_close_ret.groupby([pm["trade_date"], pm["symbol"]], sort=True, dropna=False).std().reindex(out.index)
    out["am_pm_vol_spread"] = pm_vol - am_vol
    am_amount = am.groupby(keys, sort=True, dropna=False)["amount"].sum().reindex(out.index)
    out["am_amount_share"] = _safe_ratio(am_amount, base["total_amount"])
    pm_amount_share = 1.0 - out["am_amount_share"]
    out["am_pm_amount_spread"] = out["am_amount_share"] - pm_amount_share

    out["early_strength_late_weak"] = out["first_30m_ret"] - out["last_30m_ret"]
    out["close_pressure_30m"] = out["last_30m_ret"] * out["last_30m_amount_share"]
    out["source"] = source
    out["adjusted_flag"] = str(adjusted_flag or "none")
    out = out.reset_index()
    return normalize_intraday_daily_features_frame(out, source=source, adjusted_flag=adjusted_flag, require_columns=False)


def _prepare_intraday_5m_bars(intraday_frame: pd.DataFrame, *, source: str, adjusted_flag: str) -> pd.DataFrame:
    if intraday_frame is None or intraday_frame.empty:
        return pd.DataFrame(columns=list(_INTRADAY_5M_REQUIRED_COLUMNS) + ["source", "adjusted_flag"])
    if not _INTRADAY_5M_REQUIRED_COLUMNS.issubset(set(intraday_frame.columns)):
        return normalize_intraday_5m_frame(
            intraday_frame,
            source=source,
            adjusted_flag=adjusted_flag,
            require_columns=False,
        )

    columns = [
        "symbol",
        "trade_date",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    data = intraday_frame.loc[:, columns].copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    data["trade_date"] = data["trade_date"].astype(str).str.strip().str.slice(0, 10)
    data["bar_time"] = _fast_bar_time(data["bar_time"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = source
    data["adjusted_flag"] = str(adjusted_flag or "none")
    valid = (
        data["symbol"].str.len().gt(0)
        & data["trade_date"].str.lower().ne("nat")
        & data["trade_date"].str.lower().ne("nan")
        & data["bar_time"].str.len().gt(0)
    )
    return data.loc[valid].drop_duplicates(["symbol", "trade_date", "bar_time", "source"]).reset_index(drop=True)


def _fast_bar_time(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    has_colon = text.str.contains(":", regex=False, na=False)
    if bool(has_colon.any()):
        text = text.str.replace(":", "", regex=False)
    text = text.str.replace(".", "", regex=False)
    text = text.str.slice(0, 9)
    short = text.str.len().eq(6)
    if bool(short.any()):
        text.loc[short] = text.loc[short] + "000"
    return text.str.zfill(9)


def _rows_at(frame: pd.DataFrame, mask: pd.Series, keys: list[str]) -> pd.DataFrame:
    value_columns = ["open", "high", "low", "close", "volume", "amount", "_pos"]
    if frame.empty:
        return pd.DataFrame(columns=keys + value_columns).set_index(keys)
    subset = frame.loc[mask, keys + value_columns].drop_duplicates(keys, keep="first")
    return subset.set_index(keys)


def _safe_ratio(numerator: Any, denominator: Any) -> pd.Series:
    num = pd.Series(numerator, copy=False)
    den = pd.Series(denominator, index=num.index if not isinstance(denominator, pd.Series) else denominator.index, copy=False)
    if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
        den = denominator.reindex(numerator.index)
    out = pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[pd.to_numeric(den, errors="coerce").le(0) | pd.to_numeric(den, errors="coerce").isna()] = np.nan
    return out


def _safe_return_series(close_value: Any, open_value: Any) -> pd.Series:
    close = pd.Series(close_value, copy=False)
    open_ = pd.Series(open_value, index=close.index if not isinstance(open_value, pd.Series) else open_value.index, copy=False)
    if isinstance(close_value, pd.Series) and isinstance(open_value, pd.Series):
        open_ = open_value.reindex(close_value.index)
    out = pd.to_numeric(close, errors="coerce") / pd.to_numeric(open_, errors="coerce") - 1.0
    out = out.replace([np.inf, -np.inf], np.nan)
    out.loc[pd.to_numeric(open_, errors="coerce").le(0) | pd.to_numeric(open_, errors="coerce").isna() | pd.to_numeric(close, errors="coerce").isna()] = np.nan
    return out


def _last_close_to_previous_close_return(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    data = frame.loc[:, [*keys, "close"]].copy()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["_previous_close"] = data.groupby(keys, sort=True, dropna=False)["close"].shift(1)
    last_rows = data.groupby(keys, sort=True, dropna=False).tail(1).set_index(keys)
    return _safe_return_series(last_rows["close"], last_rows["_previous_close"])


def _extreme_pos(frame: pd.DataFrame, keys: list[str], column: str, mode: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    idx = frame.groupby(keys, sort=True, dropna=False)[column].idxmax() if mode == "max" else frame.groupby(keys, sort=True, dropna=False)[column].idxmin()
    return frame.loc[idx.dropna().astype(int), keys + ["_pos"]].drop_duplicates(keys, keep="first").set_index(keys)["_pos"].astype(float)


def _group_corr(frame: pd.DataFrame, keys: list[str], x_col: str, y_col: str) -> pd.Series:
    valid = frame[[*keys, x_col, y_col]].copy()
    valid[x_col] = pd.to_numeric(valid[x_col], errors="coerce")
    valid[y_col] = pd.to_numeric(valid[y_col], errors="coerce")
    valid = valid.loc[valid[x_col].notna() & valid[y_col].notna()]
    if valid.empty:
        return pd.Series(dtype=float)
    valid["_xy"] = valid[x_col] * valid[y_col]
    valid["_x2"] = valid[x_col] * valid[x_col]
    valid["_y2"] = valid[y_col] * valid[y_col]
    agg = valid.groupby(keys, sort=True, dropna=False).agg(
        n=(x_col, "size"),
        sx=(x_col, "sum"),
        sy=(y_col, "sum"),
        sxy=("_xy", "sum"),
        sx2=("_x2", "sum"),
        sy2=("_y2", "sum"),
    )
    num = agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]
    den_sq = (agg["n"] * agg["sx2"] - agg["sx"] ** 2) * (agg["n"] * agg["sy2"] - agg["sy"] ** 2)
    den = np.sqrt(den_sq.where(den_sq > 0))
    corr = num / den
    corr = corr.replace([np.inf, -np.inf], np.nan)
    corr.loc[agg["n"].lt(2)] = np.nan
    return corr


def _linear_slope_by_group(frame: pd.DataFrame, keys: list[str], column: str) -> pd.Series:
    valid = frame.loc[pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).notna(), keys + [column]].copy()
    if valid.empty:
        return pd.Series(dtype=float)
    grouped = valid.groupby(keys, sort=True, dropna=False)
    valid["_x"] = grouped.cumcount().astype(float)
    first = grouped[column].transform("first").astype(float)
    scale = first.abs().where(first.ne(0.0), 1.0)
    valid["_y"] = valid[column].astype(float) / scale
    valid["_xy"] = valid["_x"] * valid["_y"]
    valid["_x2"] = valid["_x"] * valid["_x"]
    agg = valid.groupby(keys, sort=True, dropna=False).agg(
        n=("_y", "size"),
        sx=("_x", "sum"),
        sy=("_y", "sum"),
        sxy=("_xy", "sum"),
        sx2=("_x2", "sum"),
    )
    denom = agg["n"] * agg["sx2"] - agg["sx"] ** 2
    slope = (agg["n"] * agg["sxy"] - agg["sx"] * agg["sy"]) / denom
    slope = slope.replace([np.inf, -np.inf], np.nan)
    slope.loc[agg["n"].lt(2) | denom.eq(0)] = np.nan
    return slope


def _feature_shard_record(
    *,
    target_path: Path,
    features: pd.DataFrame,
    source_record: dict[str, Any],
    source_path: Path,
    status: str,
    materialization: str,
) -> dict[str, Any]:
    starts = pd.to_datetime(features["trade_date"], errors="coerce") if "trade_date" in features.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "domain": DataDomain.INTRADAY_DAILY_FEATURES,
        "status": status,
        "path": str(target_path.resolve()),
        "row_count": int(len(features)),
        "start_date": starts.min().strftime("%Y-%m-%d") if starts.notna().any() else str(source_record.get("start_date", "") or ""),
        "end_date": starts.max().strftime("%Y-%m-%d") if starts.notna().any() else str(source_record.get("end_date", "") or ""),
        "symbol_count": int(features["symbol"].nunique()) if "symbol" in features.columns and not features.empty else 0,
        "source_raw_path": str(source_path.resolve()),
        "source_raw_row_count": int(source_record.get("row_count", 0) or 0),
        "source_zip": str(source_record.get("source_zip", "") or ""),
        "source_member": str(source_record.get("source_member", "") or ""),
        "error_count": 0,
        "error": "",
        "materialization": materialization,
        "auction_process_policy": "unobservable_use_open_as_opening_result_only",
    }


def _feature_dataset_spec(cfg: BuildIntradayDailyFeaturesConfig, *, source_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "intraday_daily_features_from_5m",
        "domain": DataDomain.INTRADAY_DAILY_FEATURES,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "years": list(cfg.years),
        "source_5m_dataset_id": cfg.source_dataset_id,
        "source_5m_fingerprint": str(source_metadata.get("fingerprint", "") or ""),
        "source_5m_dataset_kind": str(source_metadata.get("dataset_kind", "") or ""),
        "feature_source_name": cfg.source_name,
        "feature_contract_version": INTRADAY_DAILY_FEATURE_CONTRACT_VERSION,
        "last_5m_ret_policy": LAST_5M_RET_POLICY,
        "adjusted_flag": cfg.adjusted_flag,
        "raw_ohlcv_policy": "raw_ohlcv_never_overwritten",
        "auction_process_policy": "unobservable_use_open_as_opening_result_only",
        "prediction_policy": "daily_features_for_next_day_or_multi_day_prediction_not_intraday_realtime",
    }


def _shard_intersects_years(record: dict[str, Any], years: set[int]) -> bool:
    if not years:
        return True
    start = pd.to_datetime(str(record.get("start_date", "") or ""), errors="coerce")
    end = pd.to_datetime(str(record.get("end_date", "") or ""), errors="coerce")
    if pd.isna(start) and pd.isna(end):
        return True
    if pd.isna(end):
        end = start
    if pd.isna(start):
        start = end
    return any(int(start.year) <= year <= int(end.year) for year in years)


def _safe_stem(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(raw or "shard"))
    return safe[:180] if len(safe) > 180 else safe


def _default_progress_path(lake: ResearchDataLake) -> Path:
    return lake.root / "canonical" / "imports" / f"intraday_daily_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}_progress.jsonl"


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), **payload}
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with _PROGRESS_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build daily intraday feature sidecar from a sharded 5m dataset.")
    parser.add_argument("--lake-root", default=str(DEFAULT_DATA_LAKE_ROOT))
    parser.add_argument("--source-dataset-id", required=True)
    parser.add_argument("--start-date", default=DEFAULT_CANONICAL_START_DATE)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--years", default="")
    parser.add_argument("--source-name", default="external_1m_derived_5m")
    parser.add_argument("--adjusted-flag", default="none")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--progress-path", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--reuse", dest="reuse", action="store_true", default=True)
    parser.add_argument("--no-reuse", dest="reuse", action="store_false")
    return parser


def _parse_years(text: str) -> tuple[int, ...]:
    years: list[int] = []
    for raw_item in str(text or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"invalid_year_range: {item}")
            years.extend(range(start, end + 1))
        else:
            years.append(int(item))
    return tuple(dict.fromkeys(years))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_intraday_daily_features(
        BuildIntradayDailyFeaturesConfig(
            lake_root=Path(args.lake_root),
            source_dataset_id=args.source_dataset_id,
            start_date=args.start_date,
            end_date=args.end_date,
            years=_parse_years(args.years),
            source_name=args.source_name,
            adjusted_flag=args.adjusted_flag,
            workers=args.workers,
            max_shards=args.max_shards,
            progress_path=Path(args.progress_path) if str(args.progress_path or "").strip() else None,
            dry_run=args.dry_run,
            resume=args.resume,
            reuse=args.reuse,
        )
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "dataset_id": result.dataset_id,
                "shard_count": result.shard_count,
                "row_count": result.row_count,
                "error_count": result.error_count,
                "progress_path": str(result.progress_path.resolve()) if result.progress_path else "",
                "errors": result.errors,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
