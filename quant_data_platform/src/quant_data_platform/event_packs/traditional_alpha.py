from __future__ import annotations

import hashlib
import json
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
    frames_for_quality: list[pd.DataFrame] = []
    total_rows = 0
    missing_feature_rows = 0
    event_type_counts: dict[str, int] = {}
    for year in years:
        events = build_year_events(
            cfg,
            year=year,
            sell_windows=sell_windows,
            output_root=output_root,
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
        out_path = parts_dir / f"year={int(year)}" / "events.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(out_path, index=False)
        row_count = int(len(enriched))
        qdp_missing = int(enriched["qdp_feature_missing"].sum()) if "qdp_feature_missing" in enriched else 0
        missing_feature_rows += qdp_missing
        total_rows += row_count
        type_counts = enriched["primary_event_type"].fillna("unknown").astype(str).value_counts().to_dict()
        for key, value in type_counts.items():
            event_type_counts[key] = int(event_type_counts.get(key, 0) + int(value))
        partitions.append(
            {
                "year": int(year),
                "path": str(out_path),
                "row_count": row_count,
                "qdp_feature_missing_rows": qdp_missing,
                "primary_event_type_counts": {str(k): int(v) for k, v in sorted(type_counts.items())},
            }
        )
        frames_for_quality.append(enriched)

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
        frames_for_quality,
        total_rows=total_rows,
        missing_feature_rows=missing_feature_rows,
        event_type_counts=event_type_counts,
    )
    dataset_id = event_pack_dataset_id(
        tag=cfg.tag,
        source_training_pack=str(source_manifest_path),
        years=years,
        feature_columns=feature_columns,
        row_count=total_rows,
    )
    manifest = {
        "artifact_type": "qdp_event_pack",
        "dataset_id": dataset_id,
        "dataset_name": cfg.tag,
        "status": "candidate",
        "created_at": utc_now(),
        "config": asdict(cfg),
        "years": [int(year) for year in years],
        "row_count": int(total_rows),
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
        "row_count": int(total_rows),
        "partition_count": int(len(partitions)),
        "qdp_feature_count": int(len(feature_columns)),
        "dropped_execution_state_feature_count": int(len(dropped_features)),
        "qdp_feature_missing_rows": int(missing_feature_rows),
        "primary_event_type_counts": {str(k): int(v) for k, v in sorted(event_type_counts.items())},
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
) -> pd.DataFrame:
    from traditional_quant_research.dataset_v2 import load_tradeable_panel
    from traditional_quant_research.experiments.generalized_strong_event_pool_research import build_generalized_event_feature_panel
    from traditional_quant_research.experiments.short_open_known_factor_rebuild import DEFAULT_DATA_START_YEAR

    start_year = max(int(DEFAULT_DATA_START_YEAR), int(year) - int(cfg.warmup_years))
    raw_panel = load_tradeable_panel(
        cfg.pit_root or None,
        start_date=f"{start_year}-01-01",
        end_date=f"{int(year)}-12-31",
        include_metrics=True,
        include_industry=True,
    )
    events = build_generalized_event_feature_panel(
        raw_panel,
        sell_windows=sell_windows,
        min_signal_amount=float(cfg.min_signal_amount),
    )
    events = events.loc[pd.to_datetime(events["date"]).dt.year.eq(int(year))].sort_values(["date", "code"]).reset_index(drop=True)
    if bool(cfg.write_event_cache):
        cache_dir = output_root / "_event_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        events.to_parquet(cache_dir / f"events_year={int(year)}.parquet", index=False)
    return events


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
    frames: Sequence[pd.DataFrame],
    *,
    total_rows: int,
    missing_feature_rows: int,
    event_type_counts: Mapping[str, int],
) -> dict[str, Any]:
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        qdp_feature_cols = [
            column
            for column in frame.columns
            if column.startswith("qdp_") and column not in {"qdp_feature_missing", "qdp_source_training_pack_manifest"}
        ]
        qdp_nan_ratio = float(frame[qdp_feature_cols].isna().to_numpy().mean()) if qdp_feature_cols else 1.0
        split_counts = {str(k): int(v) for k, v in frame["split_role"].value_counts().sort_index().items()} if "split_role" in frame else {}
    else:
        qdp_nan_ratio = 1.0
        split_counts = {}
    return {
        "status": "ok" if total_rows > 0 else "empty",
        "row_count": int(total_rows),
        "qdp_feature_missing_rows": int(missing_feature_rows),
        "qdp_feature_missing_rate": float(missing_feature_rows / total_rows) if total_rows else None,
        "qdp_feature_nan_ratio": qdp_nan_ratio,
        "primary_event_type_counts": {str(k): int(v) for k, v in sorted(event_type_counts.items())},
        "split_role_counts": split_counts,
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
