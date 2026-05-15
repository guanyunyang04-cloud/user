from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from daily_research.baseline.advanced_ml_runtime import HistoryWindow
from daily_research.continuous_policy.state_builder import (
    DEFAULT_SCORE_BLEND_WEIGHTS,
    PreparedPolicyInputs,
    _build_alpha_prior_frames,
)
from daily_research.data_lake.catalog import ResearchDataLake

DEFAULT_POLICY_INPUT_LAKE_DATASET_ID = "policy_input_bundle__0f116a9b78c92ff045a6853d"


def _read_feature_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "trade_date" in frame.columns:
        frame = frame.rename(columns={"trade_date": "date"})
    if "date" not in frame.columns:
        raise ValueError(f"Feature panel has no date/trade_date column: {path}")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").sort_index()


def _slice_wide(frame: pd.DataFrame, *, universe: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]
    return out.reindex(columns=universe).sort_index()


def _load_feature_panels(feature_glob: str) -> dict[str, pd.DataFrame]:
    pattern = Path(str(feature_glob).replace("*.parquet", ""))
    if pattern.name:
        feature_dir = pattern
    else:
        feature_dir = pattern.parent
    if str(feature_glob).endswith("*.parquet"):
        feature_dir = Path(str(feature_glob)).parent
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(feature_dir.glob("*.parquet")):
        frames[path.stem] = _read_feature_panel(path)
    return frames


def _normalize_symbol_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_lake_universe(
    *,
    available_symbols: list[str],
    requested_symbols: list[str] | None,
    extra_stocks: list[str] | None,
    max_universe_size: int,
) -> list[str]:
    available = _normalize_symbol_list(available_symbols)
    available_set = set(available)
    requested = _normalize_symbol_list(requested_symbols or [])
    extras = _normalize_symbol_list(extra_stocks or [])
    if requested:
        universe = [stock for stock in requested if stock in available_set]
        missing_requested = [stock for stock in requested if stock not in available_set]
    else:
        universe = list(available)
        missing_requested = []
    if int(max_universe_size or 0) > 0:
        capped = universe[: int(max_universe_size)]
        universe = _normalize_symbol_list([*capped, *extras])
    else:
        universe = _normalize_symbol_list([*universe, *extras])
    missing_extras = [stock for stock in extras if stock not in available_set]
    universe = [stock for stock in universe if stock in available_set]
    if not universe:
        raise ValueError(
            "lake_coverage_blocker: no requested symbols are available in lake market data; "
            f"missing_requested={missing_requested[:10]}, missing_extras={missing_extras[:10]}"
        )
    return universe


def _lake_coverage_report(
    *,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
    benchmark_close: pd.Series,
    membership_frame: pd.DataFrame,
    start_date: str,
    end_date: str,
    dataset_id: str,
    min_trading_days: int,
) -> dict[str, Any]:
    price_frames = {
        "close": close,
        "open": open_,
        "high": high,
        "low": low,
        "volume": volume,
        "amount": amount,
    }
    missing_fields = [
        name
        for name, frame in price_frames.items()
        if frame.empty or int(frame.notna().sum().sum()) <= 0
    ]
    frame_nan_cells = {
        name: int(frame.isna().sum().sum()) if not frame.empty else 0
        for name, frame in price_frames.items()
    }
    trading_days = int(len(close.index))
    membership_true_rows = int(membership_frame.any(axis=1).sum()) if not membership_frame.empty else 0
    blockers: list[str] = []
    if trading_days < int(min_trading_days or 1):
        blockers.append("insufficient_trading_days")
    if missing_fields:
        blockers.append("missing_market_fields")
    if benchmark_close.empty or int(benchmark_close.notna().sum()) < max(1, trading_days):
        blockers.append("missing_benchmark")
    if membership_frame.empty or membership_true_rows <= 0:
        blockers.append("missing_membership")
    return {
        "status": "ok" if not blockers else "lake_coverage_blocker",
        "dataset_id": str(dataset_id),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "trading_days": trading_days,
        "min_trading_days": int(min_trading_days or 1),
        "universe_size": int(len(close.columns)),
        "missing_fields": missing_fields,
        "market_nan_cells": frame_nan_cells,
        "benchmark_rows": int(benchmark_close.notna().sum()) if not benchmark_close.empty else 0,
        "membership_rows": int(len(membership_frame.index)),
        "membership_true_rows": membership_true_rows,
        "blockers": blockers,
    }


def _raise_lake_coverage_blocker(report: dict[str, Any]) -> None:
    if str(report.get("status", "")) == "ok":
        return
    raise ValueError(
        "lake_coverage_blocker: "
        f"dataset_id={report.get('dataset_id')} "
        f"start_date={report.get('start_date')} end_date={report.get('end_date')} "
        f"blockers={report.get('blockers')} missing_fields={report.get('missing_fields')} "
        f"trading_days={report.get('trading_days')} benchmark_rows={report.get('benchmark_rows')} "
        f"membership_true_rows={report.get('membership_true_rows')}"
    )


def load_policy_inputs_from_lake(
    *,
    lake: ResearchDataLake,
    dataset_id: str,
    start_date: str,
    end_date: str,
    pool_name: str = "",
    benchmark: str = "000300.SH",
    universe: list[str] | None = None,
    extra_stocks: list[str] | None = None,
    max_universe_size: int = 0,
    min_trading_days: int = 2,
    alpha_prior_source: str = "none",
    alpha_prior_score_panel: str = "",
    alpha_prior_target_weight_panel: str = "",
) -> PreparedPolicyInputs:
    dataset_id = str(dataset_id or DEFAULT_POLICY_INPUT_LAKE_DATASET_ID)
    metadata = lake.describe_dataset(dataset_id)
    start_date = str(start_date or metadata.get("start_date", "") or "").strip()
    end_date = str(end_date or metadata.get("end_date", "") or start_date).strip()
    if not start_date or not end_date:
        raise ValueError(f"lake_coverage_blocker: start/end date is required for lake dataset {dataset_id}")
    paths = dict(metadata.get("content_paths", {}) or {})
    market_path = str(paths.get("bronze_market_data", "") or "")
    if not market_path:
        raise ValueError(f"lake_coverage_blocker: dataset has no bronze_market_data path: {dataset_id}")
    market = pd.read_parquet(market_path)
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_trade_dates = pd.Index(sorted(pd.to_datetime(market["trade_date"].dropna().unique())))
    market = market.loc[(market["trade_date"] >= start_ts) & (market["trade_date"] <= end_ts)].copy()
    if market.empty:
        raise ValueError(f"lake_coverage_blocker: no market rows in lake dataset {dataset_id} for {start_date} -> {end_date}")
    available_symbols = sorted(str(item).strip().upper() for item in market["symbol"].dropna().astype(str).unique())
    resolved_universe = _resolve_lake_universe(
        available_symbols=available_symbols,
        requested_symbols=universe,
        extra_stocks=extra_stocks,
        max_universe_size=max_universe_size,
    )
    market["symbol"] = market["symbol"].astype(str).str.strip().str.upper()
    market = market.loc[market["symbol"].isin(resolved_universe)].copy()
    if market.empty:
        raise ValueError(f"lake_coverage_blocker: no selected market rows in lake dataset {dataset_id} for {start_date} -> {end_date}")

    def pivot(column: str) -> pd.DataFrame:
        out = market.pivot(index="trade_date", columns="symbol", values=column)
        out = out.reindex(columns=resolved_universe).sort_index()
        out.index.name = None
        out.columns.name = None
        return out

    open_ = pivot("open")
    high = pivot("high")
    low = pivot("low")
    close = pivot("close")
    volume = pivot("volume")
    amount = pivot("amount")

    benchmark_path = str(paths.get("silver_benchmark", "") or "")
    if not benchmark_path:
        raise ValueError(f"lake_coverage_blocker: dataset has no silver_benchmark path: {dataset_id}")
    benchmark_frame = pd.read_parquet(benchmark_path)
    benchmark_frame["trade_date"] = pd.to_datetime(benchmark_frame["trade_date"])
    benchmark_frame = benchmark_frame.loc[(benchmark_frame["trade_date"] >= start_ts) & (benchmark_frame["trade_date"] <= end_ts)].copy()
    if "benchmark" in benchmark_frame.columns and str(benchmark or "").strip():
        requested_benchmark = str(benchmark or "").strip().upper()
        matched = benchmark_frame.loc[benchmark_frame["benchmark"].astype(str).str.upper() == requested_benchmark].copy()
        if not matched.empty:
            benchmark_frame = matched
    benchmark_close = pd.Series(dtype=float, name=str(benchmark or metadata.get("benchmark", "") or "000300.SH"))
    if "close" in benchmark_frame.columns and not benchmark_frame.empty:
        benchmark_close = pd.Series(
            pd.to_numeric(benchmark_frame["close"], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(benchmark_frame["trade_date"]),
            name=str(benchmark or metadata.get("benchmark", "") or "000300.SH"),
        ).sort_index()
    if "open" in benchmark_frame.columns and not benchmark_frame.empty:
        benchmark_open = pd.Series(
            pd.to_numeric(benchmark_frame["open"], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(benchmark_frame["trade_date"]),
            name=str(benchmark_close.name),
        ).sort_index()
    else:
        benchmark_open = benchmark_close.copy()

    membership_path = str(paths.get("silver_membership", "") or "")
    if not membership_path:
        raise ValueError(f"lake_coverage_blocker: dataset has no silver_membership path: {dataset_id}")
    membership_raw = pd.read_parquet(membership_path)
    if "trade_date" in membership_raw.columns:
        membership_raw = membership_raw.rename(columns={"trade_date": "date"})
    if "date" in membership_raw.columns:
        membership_raw["date"] = pd.to_datetime(membership_raw["date"])
        membership_frame = membership_raw.set_index("date").sort_index()
    else:
        membership_frame = membership_raw.copy()
        if len(membership_frame) == len(all_trade_dates):
            membership_frame.index = all_trade_dates
        elif len(membership_frame) == len(close.index):
            membership_frame.index = close.index
        else:
            raise ValueError(
                "Silver membership has no date column and row count does not match market dates: "
                f"membership_rows={len(membership_frame)}, all_trade_dates={len(all_trade_dates)}, selected_dates={len(close.index)}"
            )
    membership_frame = _slice_wide(membership_frame, universe=resolved_universe, start_date=start_date, end_date=end_date).fillna(False).astype(bool)

    panels = _load_feature_panels(str(paths["silver_feature_panels"]))
    sliced_panels = {
        name: _slice_wide(frame, universe=resolved_universe, start_date=start_date, end_date=end_date)
        for name, frame in panels.items()
    }
    score_none = sliced_panels.get("score_none", pd.DataFrame(index=close.index, columns=resolved_universe, dtype=float)).copy()
    score_v2 = sliced_panels.get("score_v2", pd.DataFrame(index=close.index, columns=resolved_universe, dtype=float)).copy()
    score_blend = sliced_panels.get("score_blend")
    if score_blend is None:
        score_blend = score_none * float(DEFAULT_SCORE_BLEND_WEIGHTS[0]) + score_v2 * float(DEFAULT_SCORE_BLEND_WEIGHTS[1])

    coverage_report = _lake_coverage_report(
        close=close,
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        benchmark_close=benchmark_close,
        membership_frame=membership_frame,
        start_date=start_date,
        end_date=end_date,
        dataset_id=dataset_id,
        min_trading_days=int(min_trading_days or 1),
    )
    _raise_lake_coverage_blocker(coverage_report)

    feature_names = {
        "z_score_none",
        "z_score_v2",
        "adv20_rank",
        "price_rank",
        "ma20_gap",
        "ma60_gap",
        "volume_rank",
    }
    feature_frames = {name: frame for name, frame in sliced_panels.items() if name in feature_names}
    alpha_prior_frames, alpha_prior_summary = _build_alpha_prior_frames(
        close=close,
        source=alpha_prior_source,
        score_panel=alpha_prior_score_panel,
        target_weight_panel=alpha_prior_target_weight_panel,
    )
    derived_frames = {name: frame for name, frame in sliced_panels.items() if name not in {"score_none", "score_v2"}}
    derived_frames.update(alpha_prior_frames)
    derived_frames["score_blend"] = score_blend

    history_window = HistoryWindow(
        mode="train",
        requested_start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        effective_start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        required_trading_days=int(len(close.index)),
    )
    return PreparedPolicyInputs(
        universe=tuple(resolved_universe),
        pool_name=str(pool_name or metadata.get("universe_name", "") or metadata.get("parameters", {}).get("pool_name", "") or metadata.get("parameters", {}).get("universe", "") or "learned_all_a"),
        benchmark=str(benchmark or metadata.get("benchmark", "") or "000300.SH"),
        data_source="lake",
        csv_folder="",
        start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        requested_start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        history_window=history_window,
        raw_cache_meta={
            "cache_hit": True,
            "source": "data_lake",
            "dataset_id": dataset_id,
            "data_lake_root": str(lake.root.resolve()),
            "lake_coverage_report": coverage_report,
        },
        prepared_cache_meta={
            "cache_hit": True,
            "source": "data_lake",
            "dataset_id": dataset_id,
            "data_lake_root": str(lake.root.resolve()),
            "lake_coverage_report": coverage_report,
        },
        close=close,
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        score_none=score_none,
        score_v2=score_v2,
        score_blend=score_blend,
        feature_frames=feature_frames,
        market_features={},
        membership_frame=membership_frame.reindex(index=close.index, columns=close.columns, fill_value=False),
        rolling_pool_summary={"source": "data_lake", "dataset_id": dataset_id},
        alpha_prior_summary=alpha_prior_summary,
        derived_frames=derived_frames,
    )
