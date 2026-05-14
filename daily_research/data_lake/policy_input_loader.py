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


def load_policy_inputs_from_lake(
    *,
    lake: ResearchDataLake,
    dataset_id: str,
    start_date: str,
    end_date: str,
    alpha_prior_source: str = "none",
    alpha_prior_score_panel: str = "",
    alpha_prior_target_weight_panel: str = "",
) -> PreparedPolicyInputs:
    metadata = lake.describe_dataset(dataset_id)
    paths = dict(metadata.get("content_paths", {}) or {})
    market_path = str(paths.get("bronze_market_data", "") or "")
    if not market_path:
        raise ValueError(f"Dataset has no bronze_market_data path: {dataset_id}")
    market = pd.read_parquet(market_path)
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_trade_dates = pd.Index(sorted(pd.to_datetime(market["trade_date"].dropna().unique())))
    market = market.loc[(market["trade_date"] >= start_ts) & (market["trade_date"] <= end_ts)].copy()
    if market.empty:
        raise ValueError(f"No market rows in lake dataset {dataset_id} for {start_date} -> {end_date}")
    universe = sorted(str(item) for item in market["symbol"].dropna().astype(str).unique())

    def pivot(column: str) -> pd.DataFrame:
        out = market.pivot(index="trade_date", columns="symbol", values=column)
        out = out.reindex(columns=universe).sort_index()
        out.index.name = None
        out.columns.name = None
        return out

    open_ = pivot("open")
    high = pivot("high")
    low = pivot("low")
    close = pivot("close")
    volume = pivot("volume")
    amount = pivot("amount")

    benchmark = pd.read_parquet(str(paths["silver_benchmark"]))
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark = benchmark.loc[(benchmark["trade_date"] >= start_ts) & (benchmark["trade_date"] <= end_ts)].copy()
    benchmark_close = pd.Series(
        pd.to_numeric(benchmark["close"], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(benchmark["trade_date"]),
        name=str(metadata.get("benchmark", "") or "000300.SH"),
    ).sort_index()
    benchmark_open = benchmark_close.copy()

    membership_raw = pd.read_parquet(str(paths["silver_membership"]))
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
    membership_frame = _slice_wide(membership_frame, universe=universe, start_date=start_date, end_date=end_date).fillna(False).astype(bool)

    panels = _load_feature_panels(str(paths["silver_feature_panels"]))
    sliced_panels = {
        name: _slice_wide(frame, universe=universe, start_date=start_date, end_date=end_date)
        for name, frame in panels.items()
    }
    score_none = sliced_panels.get("score_none", pd.DataFrame(index=close.index, columns=universe, dtype=float)).copy()
    score_v2 = sliced_panels.get("score_v2", pd.DataFrame(index=close.index, columns=universe, dtype=float)).copy()
    score_blend = sliced_panels.get("score_blend")
    if score_blend is None:
        score_blend = score_none * float(DEFAULT_SCORE_BLEND_WEIGHTS[0]) + score_v2 * float(DEFAULT_SCORE_BLEND_WEIGHTS[1])

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
        universe=tuple(universe),
        pool_name=str(metadata.get("universe_name", "") or metadata.get("parameters", {}).get("universe", "") or "learned_all_a"),
        benchmark=str(metadata.get("benchmark", "") or "000300.SH"),
        data_source=str(metadata.get("source", "") or "lake"),
        csv_folder="",
        start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        requested_start_date=pd.Timestamp(start_date).strftime("%Y%m%d"),
        history_window=history_window,
        raw_cache_meta={"cache_hit": True, "source": "data_lake", "dataset_id": dataset_id},
        prepared_cache_meta={"cache_hit": True, "source": "data_lake", "dataset_id": dataset_id},
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
