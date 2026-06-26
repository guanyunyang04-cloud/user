from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import numpy as np
import pandas as pd

from quant_data_platform.lake import ResearchDataLake, load_pool_view
from quant_data_platform.lake.policy_input_loader import load_policy_inputs_from_lake
from daily_research.path_policy.decision_score_proxy import add_path_proxy_decision_scores
from daily_research.path_policy.labels import build_path20_dataset_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = PROJECT_ROOT / "daily_research/output/path_policy/studies"
RUN_TAG = "tq_baostock_lineage_audit_20260601_01"
NEW_DATASET_ID = "policy_input_bundle__45e3d8c059ba718426a9f887"
CORRECTED_POOL_VIEW_ID = "policy_pool_view__74f45f4f83263bccd64a8027"
TQCENTER_PATH = Path("H:/new_tdx64/PYPlugins/user/t0_project/tqcenter.py")
TQ_CONNECTION_PATH = TQCENTER_PATH
CURRENT_MANIFEST_PATH = (
    STUDIES_ROOT
    / "mh_rebuild_mainboard_target_norm_head_constraint_raw_seed7_20260601_01"
    / "forecast_dataset_manifest.json"
)
ACTIVE_EXECUTION_ARTIFACT = PROJECT_ROOT / "daily_research/output/active_execution_strategy.json"
DEFAULT_HORIZONS = (1, 2, 3, 5, 8, 10, 15, 20, 30)
MARKET_FIELDS = ("Open", "High", "Low", "Close", "Volume", "Amount")
PRICE_FIELDS = ("Open", "High", "Low", "Close")
SIZE_FIELDS = ("Volume", "Amount")


class TQMarketAdapter(Protocol):
    def get_market_data(
        self,
        *,
        field_list: list[str],
        stock_list: list[str],
        period: str,
        start_time: str,
        end_time: str,
        count: int,
        dividend_type: str,
        fill_data: bool,
    ) -> Mapping[str, pd.DataFrame]:
        ...


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [_normalize_symbol(column) for column in out.columns]
    return out


def _wide_to_long(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    working = _normalize_date_index(frame)
    try:
        stacked = working.stack(future_stack=True)
    except TypeError:
        stacked = working.stack(dropna=False)
    out = stacked.rename(str(field)).reset_index()
    out.columns = ["trade_date", "symbol", str(field)]
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y-%m-%d")
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out[str(field)] = pd.to_numeric(out[str(field)], errors="coerce")
    return out


def market_frames_to_long(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for field in MARKET_FIELDS:
        frame = frames.get(field)
        if frame is None:
            continue
        long = _wide_to_long(frame, field)
        merged = long if merged is None else merged.merge(long, on=["trade_date", "symbol"], how="outer")
    if merged is None:
        merged = pd.DataFrame(columns=["trade_date", "symbol", *MARKET_FIELDS])
    for field in MARKET_FIELDS:
        if field not in merged.columns:
            merged[field] = np.nan
    return merged[["trade_date", "symbol", *MARKET_FIELDS]].sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _parse_feature_columns_from_text(text: str) -> list[str]:
    # Conservative fallback for JSON-like references. It intentionally ignores loose "156 features" prose.
    matches = re.findall(r'"feature_columns"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    for raw in matches:
        columns = re.findall(r'"([^"]+)"', raw)
        if len(columns) == 156:
            return [str(item) for item in columns]
    return []


def _manifest_feature_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    columns = [str(item) for item in payload.get("feature_columns", []) or []]
    shape = [int(item) for item in payload.get("feature_store_shape", []) or [] if str(item).strip()]
    if len(shape) == 3 and int(shape[2]) == 156 and len(columns) == 156:
        return {
            "source_type": "forecast_dataset_manifest",
            "path": str(path),
            "source_market_dataset_id": str(payload.get("source_market_dataset_id", "")),
            "source_pool_view_id": str(payload.get("source_pool_view_id", "")),
            "feature_store_shape": shape,
            "feature_group_counts": dict(payload.get("feature_group_counts", {}) or {}),
            "feature_columns": columns,
            "evidence_level": "manifest_exact",
        }
    return {}


def _fallback_feature_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".json", ".md", ".txt", ".csv"}:
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    columns = _parse_feature_columns_from_text(text)
    if len(columns) != 156:
        return {}
    return {
        "source_type": "text_feature_columns",
        "path": str(path),
        "source_market_dataset_id": "",
        "source_pool_view_id": "",
        "feature_store_shape": [],
        "feature_group_counts": {},
        "feature_columns": columns,
        "evidence_level": "text_columns_trace_required",
    }


def _git_manifest_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    try:
        proc = subprocess.run(
            ["git", "log", "--all", "--name-only", "--pretty=format:", "--", "*forecast_dataset_manifest.json"],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except Exception:
        return candidates
    paths = sorted({line.strip() for line in proc.stdout.splitlines() if line.strip()})
    for rel in paths[:200]:
        try:
            show = subprocess.run(
                ["git", "show", f"HEAD:{rel}"],
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except Exception:
            continue
        if show.returncode != 0:
            continue
        try:
            payload = json.loads(show.stdout)
        except Exception:
            continue
        columns = [str(item) for item in payload.get("feature_columns", []) or []] if isinstance(payload, dict) else []
        shape = [int(item) for item in payload.get("feature_store_shape", []) or []] if isinstance(payload, dict) else []
        if len(shape) == 3 and int(shape[2]) == 156 and len(columns) == 156:
            candidates.append(
                {
                    "source_type": "git_head_manifest",
                    "path": f"HEAD:{rel}",
                    "source_market_dataset_id": str(payload.get("source_market_dataset_id", "")),
                    "source_pool_view_id": str(payload.get("source_pool_view_id", "")),
                    "feature_store_shape": shape,
                    "feature_group_counts": dict(payload.get("feature_group_counts", {}) or {}),
                    "feature_columns": columns,
                    "evidence_level": "git_manifest_exact",
                }
            )
    return candidates


def recover_legacy156_features(
    *,
    current_manifest_path: str | Path = CURRENT_MANIFEST_PATH,
    search_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    roots = list(
        search_roots
        or (
            PROJECT_ROOT / "daily_research/output",
            PROJECT_ROOT / "daily_research/cache",
            PROJECT_ROOT / "daily_research/archive",
            PROJECT_ROOT / "daily_research/brain/references",
            Path("H:/new_tdx64/PYPlugins/user/t0_project"),
        )
    )
    candidates: list[dict[str, Any]] = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            payload = _manifest_feature_payload(path) if path.name == "forecast_dataset_manifest.json" else {}
            if not payload:
                payload = _fallback_feature_payload(path)
            if payload:
                candidates.append(payload)
    candidates.extend(_git_manifest_candidates())
    current = _read_json(Path(current_manifest_path))
    current_columns = [str(item) for item in current.get("feature_columns", []) or []]
    if not candidates:
        return {
            "schema_version": 1,
            "status": "legacy156_not_recovered",
            "legacy156_profile_allowed": False,
            "feature_schema_shift": "unresolved_root_cause",
            "current_manifest_path": str(current_manifest_path),
            "current_feature_count": int(len(current_columns)),
            "current_feature_store_shape": [int(item) for item in current.get("feature_store_shape", []) or []],
            "search_roots": [str(root) for root in roots],
            "candidate_count": 0,
            "reason": "No traceable artifact with exactly 156 feature columns was found; loose prose mentioning 156 features is ignored.",
        }
    legacy = candidates[0]
    legacy_columns = [str(item) for item in legacy.get("feature_columns", []) or []]
    current_set = set(current_columns)
    legacy_set = set(legacy_columns)
    return {
        "schema_version": 1,
        "status": "legacy156_recovered",
        "legacy156_profile_allowed": True,
        "feature_schema_shift": "resolved_columns_recovered",
        "current_manifest_path": str(current_manifest_path),
        "legacy_manifest_path": str(legacy.get("path", "")),
        "current_feature_count": int(len(current_columns)),
        "legacy_feature_count": int(len(legacy_columns)),
        "current_feature_store_shape": [int(item) for item in current.get("feature_store_shape", []) or []],
        "legacy_feature_store_shape": list(legacy.get("feature_store_shape", []) or []),
        "current_feature_group_counts": dict(current.get("feature_group_counts", {}) or {}),
        "legacy_feature_group_counts": dict(legacy.get("feature_group_counts", {}) or {}),
        "shared_feature_count": int(len(current_set & legacy_set)),
        "old_only_feature_count": int(len(legacy_set - current_set)),
        "new_only_feature_count": int(len(current_set - legacy_set)),
        "old_only_features": sorted(legacy_set - current_set),
        "new_only_features": sorted(current_set - legacy_set),
        "shared_features": sorted(current_set & legacy_set),
        "candidate_count": int(len(candidates)),
        "candidates": [
            {
                "path": str(item.get("path", "")),
                "source_type": str(item.get("source_type", "")),
                "evidence_level": str(item.get("evidence_level", "")),
                "source_market_dataset_id": str(item.get("source_market_dataset_id", "")),
                "source_pool_view_id": str(item.get("source_pool_view_id", "")),
            }
            for item in candidates[:20]
        ],
    }


class TQCenterAdapter:
    def __init__(
        self,
        tqcenter_path: str | Path = TQCENTER_PATH,
        *,
        connection_path: str | Path | None = TQ_CONNECTION_PATH,
    ) -> None:
        path = Path(tqcenter_path)
        if not path.exists():
            raise FileNotFoundError(f"tqcenter.py not found: {path}")
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("_legacy_tqcenter_for_audit", str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import tqcenter.py: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.tq = getattr(module, "tq")
        self.connection_path = str(connection_path or path)
        self.tq.initialize(self.connection_path)

    def get_market_data(
        self,
        *,
        field_list: list[str],
        stock_list: list[str],
        period: str,
        start_time: str,
        end_time: str,
        count: int,
        dividend_type: str,
        fill_data: bool,
    ) -> Mapping[str, pd.DataFrame]:
        return self.tq.get_market_data(
            field_list=field_list,
            stock_list=stock_list,
            period=period,
            start_time=start_time,
            end_time=end_time,
            count=count,
            dividend_type=dividend_type,
            fill_data=fill_data,
        )


def select_audit_symbols(
    *,
    open_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
    amount_frame: pd.DataFrame,
    membership_frame: pd.DataFrame,
    max_symbols: int = 120,
    seed: int = 20260601,
) -> list[str]:
    membership = membership_frame.fillna(False).astype(bool)
    active = [str(column) for column in membership.columns if bool(membership[column].any())]
    amount = amount_frame.reindex(columns=active)
    adv = amount.tail(min(len(amount), 252)).mean(axis=0, skipna=True)
    last_price = close_frame.reindex(columns=active).ffill().tail(1).iloc[0] if active and not close_frame.empty else pd.Series(dtype=float)
    sh = [symbol for symbol in active if symbol.endswith(".SH")]
    sz = [symbol for symbol in active if symbol.endswith(".SZ")]
    high_sh = list(adv.reindex(sh).sort_values(ascending=False).dropna().index[:40])
    high_sz = list(adv.reindex(sz).sort_values(ascending=False).dropna().index[:40])
    low_score = (adv.rank(pct=True).fillna(1.0) + last_price.reindex(active).rank(pct=True).fillna(1.0)).sort_values()
    low_boundary = [symbol for symbol in low_score.index if symbol not in set(high_sh + high_sz)][:20]
    selected = list(dict.fromkeys([*high_sh, *high_sz, *low_boundary]))
    remaining = [symbol for symbol in active if symbol not in set(selected)]
    rng = np.random.default_rng(int(seed))
    random_pick = list(rng.choice(remaining, size=min(20, len(remaining)), replace=False)) if remaining else []
    selected = list(dict.fromkeys([*selected, *map(str, random_pick)]))
    return selected[: int(max_symbols)]


def fetch_tq_market_frames(
    *,
    adapter: TQMarketAdapter,
    symbols: list[str],
    start_date: str,
    end_date: str,
    period: str,
    dividend_type: str,
) -> dict[str, pd.DataFrame]:
    raw = adapter.get_market_data(
        field_list=list(MARKET_FIELDS),
        stock_list=symbols,
        period=period,
        start_time=start_date,
        end_time=end_date,
        count=-1,
        dividend_type=dividend_type,
        fill_data=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    for field in MARKET_FIELDS:
        frame = raw.get(field)
        if isinstance(frame, pd.DataFrame):
            frames[field] = _normalize_date_index(frame)
        else:
            frames[field] = pd.DataFrame()
    return frames


def _diff_stats(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {"mean": float(clean.mean()), "p95": float(clean.quantile(0.95)), "max": float(clean.max())}


def compare_ohlcv_frames(
    *,
    baostock_frames: Mapping[str, pd.DataFrame],
    tq_frames_by_dividend: Mapping[str, Mapping[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bao = market_frames_to_long(baostock_frames).rename(columns={field: f"bao_{field}" for field in MARKET_FIELDS})
    rows: list[pd.DataFrame] = []
    dividend_summaries: dict[str, Any] = {}
    for dividend_type, tq_frames in tq_frames_by_dividend.items():
        tq_long = market_frames_to_long(tq_frames).rename(columns={field: f"tq_{field}" for field in MARKET_FIELDS})
        merged = bao.merge(tq_long, on=["trade_date", "symbol"], how="outer", indicator=True)
        merged.insert(0, "dividend_type", str(dividend_type))
        for field in MARKET_FIELDS:
            tq_col = f"tq_{field}"
            bao_col = f"bao_{field}"
            merged[f"abs_diff_{field}"] = (pd.to_numeric(merged[tq_col], errors="coerce") - pd.to_numeric(merged[bao_col], errors="coerce")).abs()
            denom = pd.to_numeric(merged[bao_col], errors="coerce").abs().replace(0.0, np.nan)
            merged[f"rel_diff_{field}"] = merged[f"abs_diff_{field}"].div(denom)
        rows.append(merged)
        inner = merged.loc[merged["_merge"] == "both"].copy()
        total_union = max(int(len(merged)), 1)
        field_summary: dict[str, Any] = {}
        for field in MARKET_FIELDS:
            abs_stats = _diff_stats(inner[f"abs_diff_{field}"])
            rel_stats = _diff_stats(inner[f"rel_diff_{field}"])
            tq_col = pd.to_numeric(inner[f"tq_{field}"], errors="coerce")
            bao_col = pd.to_numeric(inner[f"bao_{field}"], errors="coerce")
            exact = tq_col.eq(bao_col) | (tq_col.isna() & bao_col.isna())
            missing_mismatch = tq_col.isna() ^ bao_col.isna()
            item = {
                "inner_rows": int(len(inner)),
                "exact_match_rate": float(exact.mean()) if len(exact) else 0.0,
                "missing_mismatch_rate": float(missing_mismatch.mean()) if len(missing_mismatch) else 0.0,
                "abs_diff": abs_stats,
                "rel_diff": rel_stats,
                "mismatch_rate_1e-6": float((inner[f"rel_diff_{field}"] > 1.0e-6).mean()) if len(inner) else 0.0,
                "mismatch_rate_1e-4": float((inner[f"rel_diff_{field}"] > 1.0e-4).mean()) if len(inner) else 0.0,
                "mismatch_rate_1e-3": float((inner[f"rel_diff_{field}"] > 1.0e-3).mean()) if len(inner) else 0.0,
            }
            if field in SIZE_FIELDS:
                item["zero_nonzero_flip_rate"] = float(((tq_col.fillna(0.0).eq(0.0)) ^ (bao_col.fillna(0.0).eq(0.0))).mean()) if len(inner) else 0.0
            field_summary[field] = item
        price_p95 = float(np.nanmean([field_summary[field]["rel_diff"]["p95"] for field in PRICE_FIELDS]))
        missing_rate = float(np.nanmean([field_summary[field]["missing_mismatch_rate"] for field in MARKET_FIELDS]))
        open_p95 = _finite_float(field_summary["Open"]["rel_diff"]["p95"])
        close_p95 = _finite_float(field_summary["Close"]["rel_diff"]["p95"])
        flags: list[str] = []
        if price_p95 <= 1.0e-4 and missing_rate <= 0.005:
            status = "tq_baostock_ohlcv_equivalent"
            flags.append("ohlcv_close_enough_for_price_labels")
        else:
            status = "material_data_source_shift"
        if open_p95 > max(close_p95 * 2.0, close_p95 + 1.0e-4):
            flags.append("open_source_shift")
        dividend_summaries[str(dividend_type)] = {
            "status": status,
            "flags": flags,
            "union_rows": int(len(merged)),
            "inner_rows": int(len(inner)),
            "left_only_rate": float((merged["_merge"] == "left_only").sum() / total_union),
            "right_only_rate": float((merged["_merge"] == "right_only").sum() / total_union),
            "price_p95_relative_diff_mean": price_p95,
            "missing_mismatch_rate_mean": missing_rate,
            "field_summary": field_summary,
        }
    if len(dividend_summaries) >= 2 and {"none", "front"}.issubset(dividend_summaries):
        none_p95 = _finite_float(dividend_summaries["none"].get("price_p95_relative_diff_mean"))
        front_p95 = _finite_float(dividend_summaries["front"].get("price_p95_relative_diff_mean"))
        if front_p95 < none_p95 * 0.5:
            dividend_summaries["front"].setdefault("flags", []).append("likely_adjustment_mismatch")
    diff = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    overall_status = "tq_baostock_ohlcv_equivalent" if any(
        item.get("status") == "tq_baostock_ohlcv_equivalent" for item in dividend_summaries.values()
    ) else "material_data_source_shift"
    return diff, {"schema_version": 1, "status": overall_status, "dividend_summaries": dividend_summaries}


def _prepared_with_market_frames(prepared: Any, frames: Mapping[str, pd.DataFrame]) -> Any:
    return replace(
        prepared,
        open_=_normalize_date_index(frames["Open"]),
        high=_normalize_date_index(frames["High"]),
        low=_normalize_date_index(frames["Low"]),
        close=_normalize_date_index(frames["Close"]),
        volume=_normalize_date_index(frames["Volume"]),
        amount=_normalize_date_index(frames["Amount"]),
    )


def _label_frame(prepared: Any, *, start_date: str, end_date: str, horizons: tuple[int, ...]) -> pd.DataFrame:
    frame, _ = build_path20_dataset_frame(
        prepared,
        start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        execution_mode="next_open",
        include_state_features=False,
    )
    frame = add_path_proxy_decision_scores(
        frame,
        cost_bps=20.0,
        hit_threshold_bps=10.0,
        drawdown_penalty=0.10,
        cumulative_horizons=horizons,
    )
    missing_future_hits = [horizon for horizon in horizons if f"future_hit_label_{int(horizon)}d" not in frame.columns]
    if missing_future_hits:
        future_utilities: list[pd.Series] = []
        used_horizons: list[int] = []
        max_horizon = max(int(item) for item in horizons)
        for horizon in horizons:
            horizon_int = int(horizon)
            return_col = f"future_cum_excess_return_{horizon_int}d"
            if return_col not in frame.columns:
                continue
            drawdown_col = f"future_path_max_drawdown_{horizon_int}d"
            drawdown = pd.to_numeric(frame.get(drawdown_col, frame.get("future_path_max_drawdown_20d", 0.0)), errors="coerce").fillna(0.0)
            horizon_scale = math.sqrt(horizon_int / max(float(max_horizon), 1.0))
            utility = (
                pd.to_numeric(frame[return_col], errors="coerce")
                - 20.0 / 10000.0
                - 0.10 * drawdown.mul(-1.0).clip(lower=0.0) * horizon_scale
            )
            frame[f"future_decision_utility_{horizon_int}d"] = utility
            frame[f"future_hit_label_{horizon_int}d"] = (utility > (10.0 / 10000.0)).astype(int)
            future_utilities.append(utility)
            used_horizons.append(horizon_int)
        if future_utilities:
            matrix = np.column_stack([series.to_numpy(dtype=float) for series in future_utilities])
            horizon_values = np.asarray(used_horizons, dtype=int)
            idx = np.nanargmax(np.where(np.isfinite(matrix), matrix, -np.inf), axis=1)
            frame["future_best_horizon"] = horizon_values[idx]
            frame["future_decision_score"] = np.nanmax(matrix, axis=1)
    keep = [
        "date",
        "stock",
        "future_decision_score",
        "future_best_horizon",
        *[f"future_cum_excess_return_{int(h)}d" for h in horizons if f"future_cum_excess_return_{int(h)}d" in frame.columns],
        *[f"future_hit_label_{int(h)}d" for h in horizons if f"future_hit_label_{int(h)}d" in frame.columns],
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def compare_next_open_labels(
    *,
    prepared: Any,
    baostock_frames: Mapping[str, pd.DataFrame],
    tq_frames: Mapping[str, pd.DataFrame],
    start_date: str,
    end_date: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    bao_prepared = _prepared_with_market_frames(prepared, baostock_frames)
    tq_prepared = _prepared_with_market_frames(prepared, tq_frames)
    bao = _label_frame(bao_prepared, start_date=start_date, end_date=end_date, horizons=horizons).add_prefix("bao_")
    tq = _label_frame(tq_prepared, start_date=start_date, end_date=end_date, horizons=horizons).add_prefix("tq_")
    merged = bao.merge(
        tq,
        left_on=["bao_date", "bao_stock"],
        right_on=["tq_date", "tq_stock"],
        how="inner",
    )
    if merged.empty:
        return merged, {"schema_version": 1, "status": "blocked_no_common_label_rows", "row_count": 0}
    rows: list[dict[str, Any]] = []
    hit_flip_rates: list[float] = []
    for horizon in horizons:
        bao_hit = f"bao_future_hit_label_{int(horizon)}d"
        tq_hit = f"tq_future_hit_label_{int(horizon)}d"
        bao_ret = f"bao_future_cum_excess_return_{int(horizon)}d"
        tq_ret = f"tq_future_cum_excess_return_{int(horizon)}d"
        if bao_hit in merged.columns and tq_hit in merged.columns:
            hit_a = pd.to_numeric(merged[bao_hit], errors="coerce")
            hit_b = pd.to_numeric(merged[tq_hit], errors="coerce")
            valid = hit_a.notna() & hit_b.notna()
            flip = float(hit_a.loc[valid].ne(hit_b.loc[valid]).mean()) if bool(valid.any()) else 0.0
            hit_flip_rates.append(flip)
        else:
            flip = 0.0
        if bao_ret in merged.columns and tq_ret in merged.columns:
            a = pd.to_numeric(merged[bao_ret], errors="coerce")
            b = pd.to_numeric(merged[tq_ret], errors="coerce")
            diff = (a - b).abs()
            denom = a.abs().replace(0.0, np.nan)
            rel = diff.div(denom)
            rel_mean = float(rel.replace([np.inf, -np.inf], np.nan).dropna().mean()) if rel.notna().any() else 0.0
        else:
            rel_mean = 0.0
        rows.append({"horizon": int(horizon), "hit_label_flip_rate": flip, "label_relative_diff_mean": rel_mean})
    score_a = pd.to_numeric(merged.get("bao_future_decision_score", pd.Series(dtype=float)), errors="coerce")
    score_b = pd.to_numeric(merged.get("tq_future_decision_score", pd.Series(dtype=float)), errors="coerce")
    valid_score = score_a.notna() & score_b.notna()
    score_diff = (score_a - score_b).abs()
    score_corr = float(score_a.loc[valid_score].corr(score_b.loc[valid_score])) if int(valid_score.sum()) > 2 else 0.0
    top20_flip = 0.0
    if "bao_future_hit_label_30d" in merged.columns and "tq_future_hit_label_30d" in merged.columns:
        top20_flip = float(
            pd.to_numeric(merged["bao_future_hit_label_30d"], errors="coerce")
            .ne(pd.to_numeric(merged["tq_future_hit_label_30d"], errors="coerce"))
            .mean()
        )
    date_rows: list[dict[str, Any]] = []
    if "bao_future_hit_label_30d" in merged.columns and "tq_future_hit_label_30d" in merged.columns:
        for date, group in merged.groupby("bao_date", sort=True):
            bao_rate = float(pd.to_numeric(group["bao_future_hit_label_30d"], errors="coerce").mean())
            tq_rate = float(pd.to_numeric(group["tq_future_hit_label_30d"], errors="coerce").mean())
            if abs(bao_rate - tq_rate) > 0.02:
                date_rows.append({"date": str(date), "bao_hit_rate_30d": bao_rate, "tq_hit_rate_30d": tq_rate, "abs_diff": abs(bao_rate - tq_rate)})
    max_flip = max(hit_flip_rates) if hit_flip_rates else 0.0
    if max_flip <= 0.01:
        status = "label_equivalent"
    elif max_flip <= 0.05:
        status = "label_shift_minor_but_relevant"
    else:
        status = "material_label_shift"
    summary = {
        "schema_version": 1,
        "status": status,
        "row_count": int(len(merged)),
        "horizon_summary": rows,
        "max_hit_label_flip_rate": float(max_flip),
        "future_decision_score_corr": score_corr,
        "future_decision_score_mae": float(score_diff.loc[valid_score].mean()) if bool(valid_score.any()) else 0.0,
        "future_decision_score_p95_abs_diff": float(score_diff.loc[valid_score].quantile(0.95)) if bool(valid_score.any()) else 0.0,
        "top20_hit_label_flip_rate": top20_flip,
        "days_all_pool_hit_rate_diff_gt_2pp": date_rows[:100],
        "top_contributing_symbol_dates": merged.assign(
            score_abs_diff=score_diff,
        )
        .sort_values("score_abs_diff", ascending=False)
        .head(100)[["bao_date", "bao_stock", "score_abs_diff"]]
        .rename(columns={"bao_date": "date", "bao_stock": "symbol"})
        .to_dict("records"),
    }
    out = pd.DataFrame(rows)
    return out, summary


def _audit_markdown(payload: Mapping[str, Any]) -> str:
    legacy = dict(payload.get("legacy156_feature_recovery", {}) or {})
    ohlcv = dict(payload.get("ohlcv_diff_summary", {}) or {})
    label = dict(payload.get("label_diff_summary", {}) or {})
    return "\n".join(
        [
            "# TQ vs BaoStock Lineage Audit",
            "",
            f"- status: `{payload.get('status')}`",
            f"- run_tag: `{payload.get('run_tag')}`",
            f"- legacy156: `{legacy.get('status')}`",
            f"- ohlcv: `{ohlcv.get('status')}`",
            f"- label: `{label.get('status')}`",
            f"- replay verdict: `{payload.get('old_stage28_replay_verdict')}`",
            f"- tq status: `{payload.get('tq_status')}`",
            "",
            "## Boundary",
            "",
            "- research-only / shadow-only",
            "- no training launched",
            "- active execution artifact unchanged by this CLI",
            "",
        ]
    )


def write_legacy_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Legacy 156 Feature Recovery Report",
        "",
        f"- status: `{report.get('status')}`",
        f"- legacy156_profile_allowed: `{report.get('legacy156_profile_allowed')}`",
        f"- current feature count: `{report.get('current_feature_count')}`",
        f"- candidate_count: `{report.get('candidate_count')}`",
        f"- feature_schema_shift: `{report.get('feature_schema_shift')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    *,
    output_root: str | Path | None = None,
    dataset_id: str = NEW_DATASET_ID,
    pool_view_id: str = CORRECTED_POOL_VIEW_ID,
    tqcenter_path: str | Path = TQCENTER_PATH,
    tq_connection_path: str | Path | None = TQ_CONNECTION_PATH,
    start_date: str = "20180101",
    end_date: str = "20241231",
    period: str = "1d",
    dividend_type_matrix: tuple[str, ...] = ("none", "front"),
    adapter: TQMarketAdapter | None = None,
) -> dict[str, Any]:
    root = Path(output_root) if output_root is not None else STUDIES_ROOT / RUN_TAG
    root.mkdir(parents=True, exist_ok=True)
    legacy = recover_legacy156_features(current_manifest_path=CURRENT_MANIFEST_PATH)
    _write_json(root / "legacy156_feature_recovery_report.json", legacy)
    write_legacy_report_markdown(root / "legacy156_feature_recovery_report.md", legacy)
    prepared = load_policy_inputs_from_lake(
        lake=ResearchDataLake(),
        dataset_id=dataset_id,
        start_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        pool_view_id=pool_view_id,
        benchmark="000300.SH",
        require_benchmark_open=True,
    )
    symbols = select_audit_symbols(
        open_frame=prepared.open_,
        close_frame=prepared.close,
        amount_frame=prepared.amount,
        membership_frame=prepared.membership_frame,
    )
    baostock_frames = {
        "Open": prepared.open_.reindex(columns=symbols),
        "High": prepared.high.reindex(columns=symbols),
        "Low": prepared.low.reindex(columns=symbols),
        "Close": prepared.close.reindex(columns=symbols),
        "Volume": prepared.volume.reindex(columns=symbols),
        "Amount": prepared.amount.reindex(columns=symbols),
    }
    tq_status = "ok"
    tq_error = ""
    tq_frames_by_dividend: dict[str, dict[str, pd.DataFrame]] = {}
    try:
        resolved_adapter = adapter or TQCenterAdapter(tqcenter_path, connection_path=tq_connection_path)
        for dividend_type in dividend_type_matrix:
            tq_frames_by_dividend[str(dividend_type)] = fetch_tq_market_frames(
                adapter=resolved_adapter,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                period=period,
                dividend_type=str(dividend_type),
            )
    except Exception as exc:
        tq_status = "blocked_tq_unavailable"
        tq_error = str(exc)
    if tq_status == "ok":
        diff, ohlcv_summary = compare_ohlcv_frames(baostock_frames=baostock_frames, tq_frames_by_dividend=tq_frames_by_dividend)
        diff.to_csv(root / "tq_baostock_ohlcv_diff.csv", index=False)
        _write_json(root / "tq_baostock_ohlcv_diff_summary.json", ohlcv_summary)
        best_dividend = min(
            ohlcv_summary["dividend_summaries"],
            key=lambda item: _finite_float(ohlcv_summary["dividend_summaries"][item].get("price_p95_relative_diff_mean"), 1.0e9),
        )
        label_diff, label_summary = compare_next_open_labels(
            prepared=prepared,
            baostock_frames=baostock_frames,
            tq_frames=tq_frames_by_dividend[str(best_dividend)],
            start_date=start_date,
            end_date=end_date,
            horizons=DEFAULT_HORIZONS,
        )
        label_diff.to_csv(root / "tq_baostock_next_open_label_diff.csv", index=False)
        _write_json(root / "tq_baostock_next_open_label_diff_summary.json", label_summary)
    else:
        ohlcv_summary = {"schema_version": 1, "status": "blocked_tq_unavailable", "error": tq_error}
        label_summary = {"schema_version": 1, "status": "blocked_tq_unavailable", "error": tq_error}
        pd.DataFrame().to_csv(root / "tq_baostock_ohlcv_diff.csv", index=False)
        pd.DataFrame().to_csv(root / "tq_baostock_next_open_label_diff.csv", index=False)
        _write_json(root / "tq_baostock_ohlcv_diff_summary.json", ohlcv_summary)
        _write_json(root / "tq_baostock_next_open_label_diff_summary.json", label_summary)
    replay_verdict = (
        "old_stage28_replay_possible"
        if legacy.get("status") == "legacy156_recovered" and label_summary.get("status") in {"label_equivalent", "label_shift_minor_but_relevant"}
        else "old_stage28_still_text_only"
    )
    status = "completed" if tq_status == "ok" else "completed_with_tq_blocker"
    audit = {
        "schema_version": 1,
        "status": status,
        "run_tag": RUN_TAG,
        "created_at": _now(),
        "dataset_id": dataset_id,
        "pool_view_id": pool_view_id,
        "tqcenter_path": str(tqcenter_path),
        "tq_connection_path": str(tq_connection_path or tqcenter_path),
        "tq_status": tq_status,
        "tq_error": tq_error,
        "sample_symbol_count": int(len(symbols)),
        "sample_symbols": symbols,
        "legacy156_feature_recovery": legacy,
        "ohlcv_diff_summary": ohlcv_summary,
        "label_diff_summary": label_summary,
        "old_stage28_replay_verdict": replay_verdict,
        "boundary": {
            "research_only": True,
            "shadow_only": True,
            "training_launched": False,
            "active_execution_artifact_expected_diff": "none",
        },
    }
    _write_json(root / "tq_baostock_lineage_audit.json", audit)
    (root / "tq_baostock_lineage_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit TQ legacy source vs BaoStock-first lake lineage.")
    parser.add_argument("--output-root", default=str(STUDIES_ROOT / RUN_TAG))
    parser.add_argument("--dataset-id", default=NEW_DATASET_ID)
    parser.add_argument("--pool-view-id", default=CORRECTED_POOL_VIEW_ID)
    parser.add_argument("--tqcenter-path", default=str(TQCENTER_PATH))
    parser.add_argument("--tq-connection-path", default=str(TQ_CONNECTION_PATH))
    parser.add_argument("--start-date", default="20180101")
    parser.add_argument("--end-date", default="20241231")
    parser.add_argument("--period", default="1d")
    parser.add_argument("--dividend-type-matrix", default="none,front")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run_audit(
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        pool_view_id=args.pool_view_id,
        tqcenter_path=args.tqcenter_path,
        tq_connection_path=args.tq_connection_path,
        start_date=args.start_date,
        end_date=args.end_date,
        period=args.period,
        dividend_type_matrix=tuple(item.strip() for item in str(args.dividend_type_matrix).split(",") if item.strip()),
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(f"status={payload.get('status')} run_tag={RUN_TAG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
