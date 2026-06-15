from __future__ import annotations

import hashlib
import json
import gc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_data_platform.core.json_io import json_safe, write_json
from quant_data_platform.core.paths import QdpPaths, qdp_paths


DEFAULT_EVENT_PACK_TAG = "traditional_event_alpha_v1_candidate"
DEFAULT_SELL_WINDOWS = (1, 3, 5, 10, 20)
DEFAULT_TRAIN_END_YEAR = 2023
DEFAULT_VALIDATION_YEAR = 2024
DEFAULT_TEST_START_YEAR = 2025
DEFAULT_MIN_SIGNAL_AMOUNT = 1.0e8
DEFAULT_FEE_BPS = 30.0
DEFAULT_BIG_LOSS_THRESHOLD_PCT = -5.0
DEFAULT_TRADITIONAL_PIT_ROOT = Path("traditional_quant_research/data/raw/baostock_daily_mainboard_v2_pit")

EXECUTION_STATE_EXACT_FEATURES = {
    "current_weight",
    "holding_flag",
    "hold_days",
    "unrealized_pnl",
    "drawdown_from_peak",
    "hold_days_clip20",
    "position_age_phase",
    "reentry_cooldown",
    "recent_buy_flag",
    "recent_sell_flag",
    "last_action_is_open",
    "last_action_is_hold",
    "last_action_is_add",
    "last_action_is_reduce",
    "last_action_is_exit",
    "pnl_to_vol20",
    "pnl_from_entry",
    "pnl_rank_in_portfolio",
    "drawdown_rank_in_portfolio",
    "portfolio_cash_pressure",
    "exit_reentry_pressure",
    "cash_regime_pressure",
}
EXECUTION_STATE_PREFIXES = (
    "holding_age_",
    "portfolio_",
    "recent_reversal_",
    "recent_reduce_",
    "recent_exit_",
    "recent_add_",
    "recent_open_",
)


@dataclass(frozen=True)
class EventPackConfig:
    source_training_pack: str = ""
    pit_root: str = ""
    output_root: str = ""
    tag: str = DEFAULT_EVENT_PACK_TAG
    start_year: int = 2024
    end_year: int = 2024
    max_events_per_year: int = 512
    warmup_years: int = 1
    sell_windows: str = "1,3,5,10,20"
    min_signal_amount: float = DEFAULT_MIN_SIGNAL_AMOUNT
    fee_bps: float = DEFAULT_FEE_BPS
    slippage_bps: float = 0.0
    big_loss_threshold_pct: float = DEFAULT_BIG_LOSS_THRESHOLD_PCT
    feature_chunk_rows: int = 4096
    write_event_cache: bool = False
    resume: bool = True
    candidate_code_prefilter: bool = True

    def normalized(self) -> "EventPackConfig":
        start = int(self.start_year or 0)
        end = int(self.end_year or 0)
        if start <= 0 and end <= 0:
            start = end = DEFAULT_VALIDATION_YEAR
        elif start <= 0:
            start = end
        elif end <= 0:
            end = start
        if end < start:
            raise ValueError("end_year must be greater than or equal to start_year")
        return EventPackConfig(
            source_training_pack=str(self.source_training_pack or "").strip(),
            pit_root=str(self.pit_root or "").strip(),
            output_root=str(self.output_root or "").strip(),
            tag=str(self.tag or DEFAULT_EVENT_PACK_TAG).strip(),
            start_year=start,
            end_year=end,
            max_events_per_year=max(int(self.max_events_per_year or 0), 0),
            warmup_years=max(int(self.warmup_years or 0), 0),
            sell_windows=",".join(str(item) for item in parse_int_values(self.sell_windows)),
            min_signal_amount=float(self.min_signal_amount),
            fee_bps=float(self.fee_bps),
            slippage_bps=float(self.slippage_bps),
            big_loss_threshold_pct=float(self.big_loss_threshold_pct),
            feature_chunk_rows=max(int(self.feature_chunk_rows or 4096), 1),
            write_event_cache=bool(self.write_event_cache),
            resume=bool(self.resume),
            candidate_code_prefilter=bool(self.candidate_code_prefilter),
        )


def parse_int_values(values: str | Sequence[int] | Sequence[str]) -> tuple[int, ...]:
    if isinstance(values, str):
        items = [item.strip() for item in values.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in values if str(item).strip()]
    parsed = tuple(int(item) for item in items)
    if not parsed:
        raise ValueError("at least one integer value is required")
    return parsed


def is_execution_state_feature(name: str) -> bool:
    column = str(name)
    return column in EXECUTION_STATE_EXACT_FEATURES or any(column.startswith(prefix) for prefix in EXECUTION_STATE_PREFIXES)


def clean_qdp_feature_columns(columns: Sequence[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for column in columns:
        if is_execution_state_feature(str(column)):
            dropped.append(str(column))
        else:
            kept.append(str(column))
    return kept, dropped


def build_traditional_event_alpha_pack(
    config: EventPackConfig,
    *,
    paths: QdpPaths | None = None,
) -> dict[str, Any]:
    cfg = config.normalized()
    resolved_paths = paths or qdp_paths()
    source_pack_dir = resolve_training_pack(resolved_paths, cfg.source_training_pack)
    source_manifest_path = source_pack_dir / "qdp_training_pack_manifest.json"
    source_manifest = read_json(source_manifest_path)
    all_feature_columns = [str(item) for item in source_manifest.get("feature_columns", [])]
    if not all_feature_columns:
        raise ValueError(f"source training pack has no feature_columns: {source_manifest_path}")
    feature_columns, dropped_features = clean_qdp_feature_columns(all_feature_columns)
    output_root = Path(cfg.output_root).resolve() if cfg.output_root else (resolved_paths.data_dir / "event_packs" / cfg.tag).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    parts_dir = output_root / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    years = tuple(range(int(cfg.start_year), int(cfg.end_year) + 1))
    sell_windows = parse_int_values(cfg.sell_windows)
    pack_reader = QdpTrainingPackReader(source_manifest_path, feature_columns=feature_columns)

    partitions: list[dict[str, Any]] = []
    quality_accumulator = new_quality_accumulator()
    for year in years:
        out_path = parts_dir / f"year={int(year)}" / "events.parquet"
        if bool(cfg.resume) and out_path.exists():
            existing = pd.read_parquet(out_path)
            stats = summarize_event_partition(existing)
            merge_quality_stats(quality_accumulator, stats)
            partitions.append(partition_record(year=int(year), path=out_path, stats=stats, resumed=True))
            del existing
            gc.collect()
            continue
        events = build_year_events(
            cfg,
            year=year,
            sell_windows=sell_windows,
            output_root=output_root,
            workspace_root=resolved_paths.workspace_root,
        )
        if cfg.max_events_per_year and len(events) > int(cfg.max_events_per_year):
            events = events.sort_values(["date", "code"]).head(int(cfg.max_events_per_year)).reset_index(drop=True)
        if events.empty:
            continue
        enriched = attach_qdp_features(events, pack_reader, chunk_rows=int(cfg.feature_chunk_rows))
        enriched = add_derived_execution_labels(
            enriched,
            sell_windows=sell_windows,
            fee_bps=float(cfg.fee_bps),
            slippage_bps=float(cfg.slippage_bps),
            big_loss_threshold_pct=float(cfg.big_loss_threshold_pct),
        )
        enriched["split_role"] = split_role_for_year(int(year))
        enriched["dataset_year"] = int(year)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(out_path, index=False)
        stats = summarize_event_partition(enriched)
        merge_quality_stats(quality_accumulator, stats)
        partitions.append(partition_record(year=int(year), path=out_path, stats=stats, resumed=False))
        del events, enriched
        gc.collect()

    feature_schema = {
        "feature_view": "traditional_event_alpha_v1_clean_qdp",
        "source_feature_count": int(len(all_feature_columns)),
        "qdp_feature_count": int(len(feature_columns)),
        "dropped_execution_state_feature_count": int(len(dropped_features)),
        "dropped_execution_state_features": dropped_features,
        "qdp_feature_columns": feature_columns,
        "qdp_feature_output_columns": [qdp_feature_output_name(col) for col in feature_columns],
        "feature_values": "normalized_qdp_training_pack_values",
    }
    label_schema = build_label_schema(sell_windows=sell_windows, fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps)
    event_schema = build_event_schema()
    quality_report = build_quality_report(
        quality_accumulator,
    )
    dataset_id = event_pack_dataset_id(
        tag=cfg.tag,
        source_training_pack=str(source_manifest_path),
        years=years,
        feature_columns=feature_columns,
        row_count=int(quality_accumulator["total_rows"]),
    )
    manifest = {
        "artifact_type": "qdp_event_pack",
        "dataset_id": dataset_id,
        "dataset_name": cfg.tag,
        "status": "candidate",
        "created_at": utc_now(),
        "config": asdict(cfg),
        "years": [int(year) for year in years],
        "row_count": int(quality_accumulator["total_rows"]),
        "partition_count": int(len(partitions)),
        "partitions": partitions,
        "source_training_pack_manifest": str(source_manifest_path),
        "source_training_pack_profile": source_manifest.get("profile") or source_manifest.get("feature_profile"),
        "source_training_pack_feature_schema_hash": source_manifest.get("feature_schema_hash", ""),
        "source_qdp_sharded_manifest_json": source_manifest.get("source_qdp_sharded_manifest_json", ""),
        "source_canonical_dataset_id": source_manifest.get("canonical_dataset_id", ""),
        "feature_schema_path": str(output_root / "feature_schema.json"),
        "label_schema_path": str(output_root / "label_schema.json"),
        "event_schema_path": str(output_root / "event_schema.json"),
        "quality_report_path": str(output_root / "quality_report.json"),
        "notes": [
            "Candidate event-level dataset for traditional_quant_research consumption.",
            "QDP execution/portfolio state features are excluded because current alpha research has no execution-state loop.",
            "This command does not activate or mutate shared QDP registries.",
        ],
    }
    write_json(output_root / "feature_schema.json", feature_schema)
    write_json(output_root / "label_schema.json", label_schema)
    write_json(output_root / "event_schema.json", event_schema)
    write_json(output_root / "quality_report.json", quality_report)
    write_json(output_root / "manifest.json", manifest)
    return {
        "status": "completed",
        "dataset_id": dataset_id,
        "output_root": str(output_root),
        "manifest": str(output_root / "manifest.json"),
        "row_count": int(quality_accumulator["total_rows"]),
        "partition_count": int(len(partitions)),
        "qdp_feature_count": int(len(feature_columns)),
        "dropped_execution_state_feature_count": int(len(dropped_features)),
        "qdp_feature_missing_rows": int(quality_accumulator["qdp_feature_missing_rows"]),
        "primary_event_type_counts": {
            str(k): int(v) for k, v in sorted(dict(quality_accumulator["event_type_counts"]).items())
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json root must be object: {path}")
    return payload


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_training_pack(paths: QdpPaths, source_training_pack: str) -> Path:
    if str(source_training_pack or "").strip():
        candidate = Path(source_training_pack).expanduser()
        if candidate.is_file():
            candidate = candidate.parent
        if not (candidate / "qdp_training_pack_manifest.json").exists():
            raise FileNotFoundError(f"training pack manifest not found under: {candidate}")
        return candidate.resolve()
    pack_root = paths.memmap_dir / "training_pack"
    candidates: list[tuple[int, Path]] = []
    for manifest_path in pack_root.glob("*/qdp_training_pack_manifest.json"):
        try:
            payload = read_json(manifest_path)
        except Exception:
            continue
        if str(payload.get("status", "")) != "completed":
            continue
        if str(payload.get("profile") or payload.get("feature_profile") or "") != "style_structural_v1":
            continue
        name = manifest_path.parent.name.lower()
        if "smoke" in name or "bench" in name:
            continue
        candidates.append((int(payload.get("sample_count", 0) or 0), manifest_path.parent))
    if not candidates:
        raise FileNotFoundError(f"no completed full style_structural_v1 training pack found under: {pack_root}")
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1].resolve()


class QdpTrainingPackReader:
    def __init__(self, manifest_path: Path, *, feature_columns: Sequence[str]) -> None:
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = read_json(self.manifest_path)
        self.all_feature_columns = [str(item) for item in self.manifest.get("feature_columns", [])]
        self.feature_columns = [str(item) for item in feature_columns]
        column_pos = {column: idx for idx, column in enumerate(self.all_feature_columns)}
        missing_columns = [column for column in self.feature_columns if column not in column_pos]
        if missing_columns:
            raise ValueError(f"feature columns missing from training pack: {missing_columns[:10]}")
        self.feature_indices = np.asarray([column_pos[column] for column in self.feature_columns], dtype=np.int64)
        self.stock_values = [str(item) for item in self.manifest.get("stock_values", [])]
        self.date_values = [str(item) for item in self.manifest.get("date_values", [])]
        self.stock_pos = {stock: idx for idx, stock in enumerate(self.stock_values)}
        self.date_pos = {date: idx for idx, date in enumerate(self.date_values)}
        self.feature_shape = tuple(int(item) for item in self.manifest.get("feature_panel_shape", []))
        self.feature_dtype = np.dtype(str(self.manifest.get("feature_dtype", "float16")))
        feature_path = Path(str(self.manifest.get("feature_panel_path", "") or ""))
        if not feature_path.exists():
            raise FileNotFoundError(f"feature panel not found: {feature_path}")
        self.feature_panel = np.memmap(feature_path, dtype=self.feature_dtype, mode="r", shape=self.feature_shape)

    def lookup_positions(self, events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dates = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
        symbols = events["code"].astype(str).to_numpy(dtype=object)
        stock_pos = np.asarray([self.stock_pos.get(str(symbol), -1) for symbol in symbols], dtype=np.int64)
        date_pos = np.asarray([self.date_pos.get(str(date), -1) for date in dates], dtype=np.int64)
        valid = (stock_pos >= 0) & (date_pos >= 0)
        return stock_pos, date_pos, valid


def attach_qdp_features(events: pd.DataFrame, reader: QdpTrainingPackReader, *, chunk_rows: int) -> pd.DataFrame:
    output = events.copy()
    n_rows = int(len(output))
    qdp_columns = [qdp_feature_output_name(column) for column in reader.feature_columns]
    feature_values = np.full((n_rows, len(reader.feature_indices)), np.nan, dtype=np.float32)
    stock_pos, date_pos, valid = reader.lookup_positions(output)
    valid_idx = np.flatnonzero(valid)
    for start in range(0, len(valid_idx), int(chunk_rows)):
        rows = valid_idx[start : start + int(chunk_rows)]
        panel_slice = reader.feature_panel[stock_pos[rows], date_pos[rows], :]
        feature_values[rows, :] = np.asarray(panel_slice[:, reader.feature_indices], dtype=np.float32)
    features = pd.DataFrame(feature_values, columns=qdp_columns, index=output.index)
    output = pd.concat([output, features], axis=1)
    output["qdp_feature_missing"] = ~pd.Series(valid, index=output.index)
    output["qdp_source_training_pack_manifest"] = str(reader.manifest_path)
    return output


def qdp_feature_output_name(column: str) -> str:
    return "qdp_" + str(column)


def build_year_events(
    cfg: EventPackConfig,
    *,
    year: int,
    sell_windows: Sequence[int],
    output_root: Path,
    workspace_root: Path | None = None,
) -> pd.DataFrame:
    from traditional_quant_research.experiments.generalized_strong_event_pool_research import build_generalized_event_feature_panel
    from traditional_quant_research.experiments.short_open_known_factor_rebuild import DEFAULT_DATA_START_YEAR

    start_year = max(int(DEFAULT_DATA_START_YEAR), int(year) - int(cfg.warmup_years))
    raw_panel = load_event_tradeable_panel(
        cfg.pit_root or None,
        start_date=f"{start_year}-01-01",
        end_date=f"{int(year)}-12-31",
        workspace_root=workspace_root,
    )
    market_context = build_market_industry_context(raw_panel)
    if bool(cfg.candidate_code_prefilter):
        candidate_codes = event_candidate_codes(raw_panel, year=year, min_signal_amount=float(cfg.min_signal_amount))
        if not candidate_codes:
            return pd.DataFrame()
        raw_panel = raw_panel.loc[raw_panel["code"].isin(candidate_codes)].reset_index(drop=True)
    events = build_generalized_event_feature_panel(
        raw_panel,
        sell_windows=sell_windows,
        min_signal_amount=float(cfg.min_signal_amount),
    )
    del raw_panel
    gc.collect()
    events = events.loc[pd.to_datetime(events["date"]).dt.year.eq(int(year))].sort_values(["date", "code"]).reset_index(drop=True)
    events = apply_market_industry_context(events, market_context)
    if bool(cfg.write_event_cache):
        cache_dir = output_root / "_event_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        events.to_parquet(cache_dir / f"events_year={int(year)}.parquet", index=False)
    return events


def event_candidate_codes(frame: pd.DataFrame, *, year: int, min_signal_amount: float) -> set[str]:
    if frame.empty:
        return set()
    dates = pd.to_datetime(frame["date"])
    target_year = dates.dt.year.eq(int(year))
    pct = pd.to_numeric(frame["pctChg"], errors="coerce")
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    close_position = (close - low) / (high - low).replace(0, np.nan)
    limit_like = pct.ge(9.5) & close.ge(high * 0.999)
    non_limit_candidate = amount.ge(float(min_signal_amount)) & pct.ge(3.0) & close_position.ge(0.65)
    codes = frame.loc[target_year & (limit_like | non_limit_candidate), "code"].astype(str)
    return set(codes.unique().tolist())


def build_market_industry_context(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {"market": pd.DataFrame(), "industry": pd.DataFrame()}
    context = frame.loc[:, ["date", "code", "industry", "high", "close", "pctChg"]].copy()
    context["date"] = pd.to_datetime(context["date"])
    pct = pd.to_numeric(context["pctChg"], errors="coerce")
    high = pd.to_numeric(context["high"], errors="coerce")
    close = pd.to_numeric(context["close"], errors="coerce")
    context["limit_up_like"] = pct.ge(9.5) & close.ge(high * 0.999)
    context["pctChg"] = pct
    market = context.groupby("date", as_index=False).agg(
        market_limitup_count=("limit_up_like", "sum"),
        market_tradeable_count=("code", "count"),
        market_breadth=("pctChg", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
    )
    market["market_limitup_rate"] = market["market_limitup_count"] / market["market_tradeable_count"].replace(0, np.nan)
    market = market.sort_values("date")
    market["market_limitup_count_ma5"] = market["market_limitup_count"].rolling(5, min_periods=2).mean()
    market["market_limitup_rate_ma20"] = market["market_limitup_rate"].rolling(20, min_periods=5).mean()
    market["market_breadth_5d"] = market["market_breadth"].rolling(5, min_periods=2).mean()
    market["market_breadth_20d"] = market["market_breadth"].rolling(20, min_periods=5).mean()
    market = market[
        [
            "date",
            "market_limitup_count",
            "market_limitup_rate",
            "market_limitup_count_ma5",
            "market_limitup_rate_ma20",
            "market_breadth_5d",
            "market_breadth_20d",
        ]
    ]
    industry = context.groupby(["industry", "date"], as_index=False).agg(
        industry_limitup_count=("limit_up_like", "sum"),
        industry_tradeable_count=("code", "count"),
        industry_ret_mean=("pctChg", "mean"),
    )
    industry["industry_limitup_rate"] = industry["industry_limitup_count"] / industry["industry_tradeable_count"].replace(0, np.nan)
    industry = industry.sort_values(["industry", "date"])
    industry["industry_limitup_count_ma5"] = (
        industry.groupby("industry", sort=False)["industry_limitup_count"]
        .rolling(5, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    industry["industry_ret5_mean"] = (
        industry.groupby("industry", sort=False)["industry_ret_mean"]
        .rolling(5, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    industry = industry[
        [
            "industry",
            "date",
            "industry_limitup_count",
            "industry_limitup_rate",
            "industry_limitup_count_ma5",
            "industry_ret5_mean",
        ]
    ]
    return {"market": market, "industry": industry}


def apply_market_industry_context(events: pd.DataFrame, context: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if events.empty:
        return events
    output = events.copy()
    output["date"] = pd.to_datetime(output["date"])
    market = context.get("market", pd.DataFrame())
    if market is not None and not market.empty:
        market_columns = [column for column in market.columns if column != "date"]
        output = output.drop(columns=[column for column in market_columns if column in output.columns])
        output = output.merge(market, on="date", how="left")
    industry = context.get("industry", pd.DataFrame())
    if industry is not None and not industry.empty and "industry" in output.columns:
        industry_columns = [column for column in industry.columns if column not in {"date", "industry"}]
        output = output.drop(columns=[column for column in industry_columns if column in output.columns])
        output = output.merge(industry, on=["industry", "date"], how="left")
    return output


def load_event_tradeable_panel(
    root: str | Path | None,
    *,
    start_date: str,
    end_date: str,
    workspace_root: Path | None = None,
) -> pd.DataFrame:
    snapshot_root = resolve_traditional_pit_root(root, workspace_root=workspace_root)
    universe = read_parquet_date_range(
        snapshot_root / "daily_universe.parquet",
        columns=["date", "code", "is_tradeable"],
        start_date=start_date,
        end_date=end_date,
        extra_filters=[("is_tradeable", "==", True)],
    )
    if universe.empty:
        return pd.DataFrame()
    universe = universe.loc[universe["is_tradeable"]].reset_index(drop=True)
    bars = read_parquet_date_range(
        snapshot_root / "daily_bars.parquet",
        columns=["date", "code", "open", "high", "low", "close", "volume", "amount"],
        start_date=start_date,
        end_date=end_date,
    )
    metrics = read_parquet_date_range(
        snapshot_root / "daily_metrics.parquet",
        columns=["date", "code", "turn", "pctChg"],
        start_date=start_date,
        end_date=end_date,
    )
    panel = universe.merge(bars, on=["date", "code"], how="inner")
    panel = panel.merge(metrics, on=["date", "code"], how="left")
    industry_path = snapshot_root / "stock_industry.parquet"
    if industry_path.exists():
        industry = read_parquet_date_range(
            industry_path,
            columns=["date", "code", "industry"],
            start_date=start_date,
            end_date=end_date,
        )
        if not industry.empty:
            panel = panel.merge(industry, on=["date", "code"], how="left")
    if "industry" not in panel.columns:
        panel["industry"] = "__unknown__"
    numeric_columns = ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]
    for column in numeric_columns:
        if column in panel.columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce", downcast="float")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype(str)
    panel["industry"] = panel["industry"].fillna("__unknown__").astype(str)
    return panel.sort_values(["date", "code"]).reset_index(drop=True)


def resolve_traditional_pit_root(root: str | Path | None, *, workspace_root: Path | None = None) -> Path:
    base = Path(root).expanduser() if str(root or "").strip() else DEFAULT_TRADITIONAL_PIT_ROOT
    base = first_existing_path(base, workspace_root=workspace_root)
    latest = base / "latest_manifest.json"
    if latest.exists():
        payload = json.loads(latest.read_text(encoding="utf-8-sig"))
        snapshot_path = str(payload.get("snapshot_path") or "").strip()
        if snapshot_path:
            return first_existing_path(Path(snapshot_path), workspace_root=workspace_root, fallback_base=base)
    return base.resolve()


def first_existing_path(path: Path, *, workspace_root: Path | None = None, fallback_base: Path | None = None) -> Path:
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if workspace_root is not None:
            candidates.append(Path(workspace_root) / path)
        candidates.append(Path.cwd() / path)
        if fallback_base is not None:
            candidates.append(Path(fallback_base) / path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def read_parquet_date_range(
    path: Path,
    *,
    columns: Sequence[str],
    start_date: str,
    end_date: str,
    extra_filters: Sequence[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: {path}")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    filters: list[tuple[str, str, Any]] = [("date", ">=", start), ("date", "<=", end)]
    if extra_filters:
        filters.extend(extra_filters)
    try:
        frame = pd.read_parquet(path, columns=list(columns), filters=filters)
    except (TypeError, ValueError, NotImplementedError):
        frame = pd.read_parquet(path, columns=list(columns))
        dates = pd.to_datetime(frame["date"])
        frame = frame.loc[(dates >= start) & (dates <= end)].reset_index(drop=True)
        for column, op, value in extra_filters or ():
            if op == "==":
                frame = frame.loc[frame[column] == value].reset_index(drop=True)
            else:
                raise ValueError(f"unsupported fallback parquet filter: {(column, op, value)}")
    return frame.reset_index(drop=True)


def add_derived_execution_labels(
    frame: pd.DataFrame,
    *,
    sell_windows: Sequence[int],
    fee_bps: float,
    slippage_bps: float,
    big_loss_threshold_pct: float,
) -> pd.DataFrame:
    output = frame.copy()
    cost_pct = (float(fee_bps) + float(slippage_bps)) / 100.0
    output["label_entry_tradeable_next_open"] = bool_series(output, "executable_entry")
    output["label_entry_limit_up_buy_blocked"] = (
        bool_series(output, "entry_open_near_limit")
        | bool_series(output, "entry_one_word_limit")
    )
    output["label_entry_suspended_or_no_open"] = False
    output["label_entry_day_close_ret_after_cost_pct"] = numeric_series(output, "entry_day_close_ret_pct") - cost_pct
    for window in sell_windows:
        raw_col = f"sell{int(window)}_close_ret_pct"
        high_col = f"sell{int(window)}_max_high_pct"
        low_col = f"sell{int(window)}_min_low_pct"
        raw = numeric_series(output, raw_col)
        high = numeric_series(output, high_col)
        low = numeric_series(output, low_col)
        output[f"label_ret_raw_d{int(window)}_pct"] = raw
        output[f"label_ret_after_cost_d{int(window)}_pct"] = raw - cost_pct
        output[f"label_mfe_1_{int(window)}_pct"] = high
        output[f"label_mae_1_{int(window)}_pct"] = low
        output[f"label_big_loss_d{int(window)}"] = low.le(float(big_loss_threshold_pct))
    return output


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def split_role_for_year(year: int) -> str:
    if int(year) <= DEFAULT_TRAIN_END_YEAR:
        return "train"
    if int(year) == DEFAULT_VALIDATION_YEAR:
        return "validation"
    if int(year) >= DEFAULT_TEST_START_YEAR:
        return "test"
    return "holdout"


def build_label_schema(*, sell_windows: Sequence[int], fee_bps: float, slippage_bps: float) -> dict[str, Any]:
    label_columns = [
        "label_entry_tradeable_next_open",
        "label_entry_limit_up_buy_blocked",
        "label_entry_suspended_or_no_open",
        "label_entry_day_close_ret_after_cost_pct",
    ]
    for window in sell_windows:
        label_columns.extend(
            [
                f"label_ret_raw_d{int(window)}_pct",
                f"label_ret_after_cost_d{int(window)}_pct",
                f"label_mfe_1_{int(window)}_pct",
                f"label_mae_1_{int(window)}_pct",
                f"label_big_loss_d{int(window)}",
            ]
        )
    return {
        "label_schema_name": "traditional_event_alpha_v1_execution_labels",
        "label_schema_version": 1,
        "execution_mode": "next_open_entry_to_future_close",
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "label_columns": label_columns,
        "notes": [
            "Labels are derived from traditional generalized strong-event event windows.",
            "Forward tradeability path labels are not included in v1; use alpha_v2 label_v2 when finalized.",
        ],
    }


def build_event_schema() -> dict[str, Any]:
    return {
        "event_schema_name": "traditional_generalized_strong_event_v1",
        "event_schema_version": 1,
        "id_columns": ["date", "code", "entry_date"],
        "event_type_column": "primary_event_type",
        "event_flag_columns": [
            "event_limit_up_core",
            "event_near_limit",
            "event_big_up",
            "event_volume_atr_breakout",
            "event_new_high_breakout",
            "event_kama_breakout",
            "event_trend_accel",
        ],
        "source": "traditional_quant_research.experiments.generalized_strong_event_pool_research",
    }


def build_quality_report(
    accumulator: Mapping[str, Any],
) -> dict[str, Any]:
    total_rows = int(accumulator.get("total_rows", 0) or 0)
    missing_feature_rows = int(accumulator.get("qdp_feature_missing_rows", 0) or 0)
    qdp_cells = int(accumulator.get("qdp_feature_cells", 0) or 0)
    qdp_nan_cells = int(accumulator.get("qdp_feature_nan_cells", 0) or 0)
    split_counts = {str(k): int(v) for k, v in sorted(dict(accumulator.get("split_counts", {}) or {}).items())}
    event_type_counts = dict(accumulator.get("event_type_counts", {}) or {})
    return {
        "status": "ok" if total_rows > 0 else "empty",
        "row_count": int(total_rows),
        "qdp_feature_missing_rows": int(missing_feature_rows),
        "qdp_feature_missing_rate": float(missing_feature_rows / total_rows) if total_rows else None,
        "qdp_feature_nan_ratio": float(qdp_nan_cells / qdp_cells) if qdp_cells else 1.0,
        "primary_event_type_counts": {str(k): int(v) for k, v in sorted(event_type_counts.items())},
        "split_role_counts": split_counts,
    }


def new_quality_accumulator() -> dict[str, Any]:
    return {
        "total_rows": 0,
        "qdp_feature_missing_rows": 0,
        "qdp_feature_nan_cells": 0,
        "qdp_feature_cells": 0,
        "event_type_counts": {},
        "split_counts": {},
    }


def summarize_event_partition(frame: pd.DataFrame) -> dict[str, Any]:
    row_count = int(len(frame))
    qdp_missing = int(frame["qdp_feature_missing"].sum()) if "qdp_feature_missing" in frame else 0
    event_type_counts = (
        frame["primary_event_type"].fillna("unknown").astype(str).value_counts().to_dict()
        if "primary_event_type" in frame
        else {}
    )
    split_counts = frame["split_role"].fillna("unknown").astype(str).value_counts().to_dict() if "split_role" in frame else {}
    qdp_feature_cols = [
        column
        for column in frame.columns
        if column.startswith("qdp_") and column not in {"qdp_feature_missing", "qdp_source_training_pack_manifest"}
    ]
    if qdp_feature_cols:
        qdp_values = frame[qdp_feature_cols]
        qdp_feature_cells = int(qdp_values.shape[0] * qdp_values.shape[1])
        qdp_feature_nan_cells = int(qdp_values.isna().to_numpy().sum())
    else:
        qdp_feature_cells = 0
        qdp_feature_nan_cells = 0
    return {
        "row_count": row_count,
        "qdp_feature_missing_rows": qdp_missing,
        "qdp_feature_cells": qdp_feature_cells,
        "qdp_feature_nan_cells": qdp_feature_nan_cells,
        "primary_event_type_counts": {str(k): int(v) for k, v in sorted(event_type_counts.items())},
        "split_role_counts": {str(k): int(v) for k, v in sorted(split_counts.items())},
    }


def merge_quality_stats(accumulator: dict[str, Any], stats: Mapping[str, Any]) -> None:
    accumulator["total_rows"] = int(accumulator.get("total_rows", 0) or 0) + int(stats.get("row_count", 0) or 0)
    accumulator["qdp_feature_missing_rows"] = int(accumulator.get("qdp_feature_missing_rows", 0) or 0) + int(
        stats.get("qdp_feature_missing_rows", 0) or 0
    )
    accumulator["qdp_feature_cells"] = int(accumulator.get("qdp_feature_cells", 0) or 0) + int(stats.get("qdp_feature_cells", 0) or 0)
    accumulator["qdp_feature_nan_cells"] = int(accumulator.get("qdp_feature_nan_cells", 0) or 0) + int(
        stats.get("qdp_feature_nan_cells", 0) or 0
    )
    for key, value in dict(stats.get("primary_event_type_counts", {}) or {}).items():
        accumulator["event_type_counts"][str(key)] = int(accumulator["event_type_counts"].get(str(key), 0)) + int(value)
    for key, value in dict(stats.get("split_role_counts", {}) or {}).items():
        accumulator["split_counts"][str(key)] = int(accumulator["split_counts"].get(str(key), 0)) + int(value)


def partition_record(*, year: int, path: Path, stats: Mapping[str, Any], resumed: bool) -> dict[str, Any]:
    return {
        "year": int(year),
        "path": str(path),
        "row_count": int(stats.get("row_count", 0) or 0),
        "qdp_feature_missing_rows": int(stats.get("qdp_feature_missing_rows", 0) or 0),
        "primary_event_type_counts": {
            str(k): int(v) for k, v in sorted(dict(stats.get("primary_event_type_counts", {}) or {}).items())
        },
        "resumed_existing_partition": bool(resumed),
    }


def event_pack_dataset_id(
    *,
    tag: str,
    source_training_pack: str,
    years: Sequence[int],
    feature_columns: Sequence[str],
    row_count: int,
) -> str:
    payload = {
        "tag": str(tag),
        "source_training_pack": str(source_training_pack),
        "years": [int(year) for year in years],
        "feature_columns": list(feature_columns),
        "row_count": int(row_count),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]
    return f"{tag}__{digest}"
