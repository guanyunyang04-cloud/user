from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from daily_research.path_policy.qdp_v2_raw_rising_path_atlas import (
    DAILY_RAW_COLUMNS,
    DEFAULT_QDP_ROOT,
    INTRADAY_COLUMNS,
    INTRADAY_SIGNAL_COLUMNS,
    LABELS,
    LIMIT_COLUMNS,
    LIMIT_SIGNAL_COLUMNS,
    RAW_SIGNAL_COLUMNS,
    _add_forward_labels,
    _add_raw_daily_signals,
    _relative_shard_paths,
    _read_dataset,
)


DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/rising_type_strategy_lab")
DEFAULT_EVENT_START = "2012-01-01"
DEFAULT_EVENT_END = "2025-12-02"
DEFAULT_DEVELOPMENT_YEARS = tuple(range(2012, 2025))
DEFAULT_FOLD_YEARS = tuple(range(2018, 2025))
DEFAULT_TEST_YEAR = 2025
DEFAULT_TOP_K = (20, 50, 100)
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_POOL_K = (300,)
DEFAULT_COST_BPS = 20.0
DEFAULT_RANDOM_SEED = 7


RISING_TYPES: dict[str, dict[str, Any]] = {
    "heat_top_gain": {
        "description": "high-intensity top-return profile; allowed to use limit-up strength.",
        "components": (("top10_return_20d", 1.0), ("rise_then_fade_10pct_20d", -0.15)),
        "target": "top10_return_20d",
    },
    "quiet_trade_path": {
        "description": "quieter path profile: reach +10% before -5%, while penalizing fade.",
        "components": (("hit_up10_before_down5_20d", 1.0), ("rise_then_fade_10pct_20d", -0.40)),
        "target": "stable_upside_target",
    },
    "persistent_trend": {
        "description": "persistent upside profile; favors gains that remain near their peak.",
        "components": (("persistent_upside_10pct_20d", 1.0), ("rise_then_fade_10pct_20d", -0.35)),
        "target": "persistent_upside_10pct_20d",
    },
    "anti_fade_blend": {
        "description": "positive path minus fade profile.",
        "components": (
            ("hit_up10_before_down5_20d", 0.45),
            ("persistent_upside_10pct_20d", 0.45),
            ("rise_then_fade_10pct_20d", -0.55),
        ),
        "target": "stable_upside_target",
    },
    "balanced_blend": {
        "description": "balanced profile across top return, tradable path, persistence, and fade control.",
        "components": (
            ("top10_return_20d", 0.30),
            ("hit_up10_before_down5_20d", 0.35),
            ("persistent_upside_10pct_20d", 0.25),
            ("rise_then_fade_10pct_20d", -0.25),
        ),
        "target": "stable_upside_target",
    },
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return str(path.resolve())


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _parse_year_range(raw: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in str(raw or "").split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start, end = [int(part) for part in item.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    return tuple(sorted(set(values))) or default


def _parse_csv_ints(raw: str | Iterable[int] | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = [int(item) for item in raw]
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item <= 0 or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return tuple(out or default)


def _chunk_slices(length: int, chunk_size: int) -> Iterable[slice]:
    chunk = max(1, int(chunk_size))
    for start in range(0, int(length), chunk):
        yield slice(start, min(start + chunk, int(length)))


def _downcast_numeric(frame: pd.DataFrame, *, skip: set[str] | None = None) -> pd.DataFrame:
    skip = set(skip or set())
    for col in frame.columns:
        if col in skip:
            continue
        if pd.api.types.is_bool_dtype(frame[col]):
            frame[col] = frame[col].astype("int8")
        elif pd.api.types.is_float_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col], downcast="float")
        elif pd.api.types.is_integer_dtype(frame[col]):
            frame[col] = pd.to_numeric(frame[col], downcast="integer")
    return frame


def _read_active(root: Path) -> dict[str, Any]:
    return json.loads((root / "active" / "active.json").read_text(encoding="utf-8"))


def _read_dataset_date_range(root: Path, active: Mapping[str, Any], domain: str, columns: list[str], start: str, end: str) -> pd.DataFrame:
    dataset_id = str(dict(active.get("datasets", {}) or {}).get(domain, "") or "")
    if not dataset_id:
        raise KeyError(f"active dataset missing: {domain}")
    manifest_path = root / "datasets" / domain / dataset_id / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available = {str(item.get("name", "")) for item in list(manifest.get("schema", []) or [])}
    selected_columns = [col for col in columns if col in available]
    paths = _relative_shard_paths(root, manifest)
    filt = (ds.field("trade_date") >= str(start)) & (ds.field("trade_date") <= str(end))
    table = ds.dataset([str(path) for path in paths], format="parquet").to_table(columns=selected_columns, filter=filt)
    frame = table.to_pandas()
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str)
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    return frame


def _add_entry_constraints(
    *,
    root: Path,
    active: Mapping[str, Any],
    daily: pd.DataFrame,
) -> pd.DataFrame:
    out = daily.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    groups = out.groupby("symbol", sort=False, group_keys=False)
    out["entry_trade_date"] = groups["trade_date"].shift(-1).astype(str)

    calendar = _read_dataset(root, active, "trading_calendar", ["trade_date", "is_open"])
    open_days = calendar[calendar["is_open"].astype(bool)]["trade_date"].astype(str).sort_values(kind="mergesort").to_numpy()
    next_map = {str(open_days[idx]): str(open_days[idx + 1]) for idx in range(len(open_days) - 1)}
    out["next_market_trade_date"] = out["trade_date"].map(next_map)
    out["entry_next_market_day_ok"] = out["entry_trade_date"].eq(out["next_market_trade_date"])

    limit_status = _read_dataset(root, active, "limit_status", ["symbol", "trade_date", "up_limit", "is_limit_up"])
    limit_status = limit_status.rename(columns={"trade_date": "entry_trade_date", "up_limit": "entry_up_limit", "is_limit_up": "entry_close_limit_up"})
    out = out.merge(limit_status, on=["symbol", "entry_trade_date"], how="left", validate="many_to_one")
    entry_open = pd.to_numeric(out["entry_open_next"], errors="coerce")
    up_limit = pd.to_numeric(out["entry_up_limit"], errors="coerce")
    out["entry_limit_up_open_blocked"] = up_limit.notna() & entry_open.notna() & (entry_open >= up_limit * 0.999)
    return out


def _add_horizon_returns(daily: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = daily.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    groups = out.groupby("symbol", sort=False, group_keys=False)
    entry = out["entry_open_next"].astype("float64")
    for horizon in sorted(set(int(item) for item in horizons)):
        out[f"return_{horizon}d"] = groups["close"].shift(-int(horizon)).astype("float64").div(entry).sub(1.0)
    out["stable_upside_target"] = (
        (out["hit_up10_before_down5_20d"].astype(bool) | out["persistent_upside_10pct_20d"].astype(bool))
        & (~out["rise_then_fade_10pct_20d"].astype(bool))
    )
    return out


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    raw = [col for col in RAW_SIGNAL_COLUMNS if col in frame.columns]
    intraday = [col for col in INTRADAY_SIGNAL_COLUMNS if col in frame.columns and col != "bar_count"]
    limit_cols = [col for col in LIMIT_SIGNAL_COLUMNS if col in frame.columns]
    out: list[str] = []
    seen: set[str] = set()
    for col in [*raw, *intraday, *limit_cols]:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _rank_normalize_features(frame: pd.DataFrame, feature_cols: list[str], progress_path: Path) -> pd.DataFrame:
    out = frame
    dates = out["trade_date"]
    total = len(feature_cols)
    for idx, col in enumerate(feature_cols, start=1):
        values = pd.to_numeric(out[col], errors="coerce")
        ranked = values.groupby(dates, sort=False).rank(method="average", pct=True)
        out[col] = ranked.sub(0.5).fillna(0.0).astype("float32")
        if idx % 10 == 0 or idx == total:
            _write_json(progress_path, {"status": "rank_normalize_features", "completed_features": idx, "total_features": total, "updated_at": _now()})
    return out


def _load_research_frame(
    *,
    root: Path,
    event_start: str,
    event_end: str,
    horizons: tuple[int, ...],
    progress_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    active = _read_active(root)
    _write_json(progress_path, {"status": "load_daily_raw", "updated_at": _now()})
    daily = _read_dataset(root, active, "market_daily_raw", DAILY_RAW_COLUMNS)
    daily = daily[(daily["trade_date"] >= "2011-11-22") & (daily["trade_date"] <= "2026-06-26")].copy()
    daily = _add_raw_daily_signals(daily)
    daily = _add_forward_labels(daily, forward_days=20)
    daily = _add_horizon_returns(daily, horizons)
    daily = _add_entry_constraints(root=root, active=active, daily=daily)
    daily["_row_pos"] = np.arange(len(daily), dtype=np.int64)

    entry_ok = (
        daily["entry_next_market_day_ok"].astype(bool)
        & daily["entry_open_next"].notna()
        & daily["path_valid_20d"].astype(bool)
        & (~daily["entry_limit_up_open_blocked"].astype(bool))
    )
    candidate = daily[
        (daily["trade_date"] >= str(event_start))
        & (daily["trade_date"] <= str(event_end))
        & entry_ok
    ].copy()
    candidate["year"] = candidate["trade_date"].str.slice(0, 4).astype("int16")
    del daily

    _write_json(progress_path, {"status": "load_industry", "candidate_rows": int(len(candidate)), "updated_at": _now()})
    industry = _read_dataset_date_range(root, active, "industry_concept", ["symbol", "trade_date", "industry"], event_start, event_end)
    candidate = candidate.merge(industry, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    candidate["industry"] = candidate["industry"].fillna("UNKNOWN")
    del industry

    _write_json(progress_path, {"status": "load_intraday_features", "candidate_rows": int(len(candidate)), "updated_at": _now()})
    intraday_cols = [col for col in INTRADAY_COLUMNS if col in {"symbol", "trade_date"} or col in INTRADAY_SIGNAL_COLUMNS]
    intraday = _read_dataset_date_range(root, active, "intraday_daily_features", intraday_cols, event_start, event_end)
    intraday = _downcast_numeric(intraday, skip={"symbol", "trade_date"})
    candidate = candidate.merge(intraday, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    del intraday

    _write_json(progress_path, {"status": "load_limit_intraday_features", "candidate_rows": int(len(candidate)), "updated_at": _now()})
    limit_cols = [col for col in LIMIT_COLUMNS if col in {"symbol", "trade_date"} or col in LIMIT_SIGNAL_COLUMNS]
    limit_features = _read_dataset_date_range(root, active, "limit_intraday_features", limit_cols, event_start, event_end)
    limit_features = _downcast_numeric(limit_features, skip={"symbol", "trade_date"})
    candidate = candidate.merge(limit_features, on=["symbol", "trade_date"], how="left", validate="one_to_one")
    del limit_features

    for label in LABELS:
        if label in candidate.columns:
            candidate[label] = candidate[label].astype(bool)
    candidate["stable_upside_target"] = candidate["stable_upside_target"].astype(bool)
    feature_cols = _feature_columns(candidate)
    for col in feature_cols:
        candidate[col] = pd.to_numeric(candidate[col], errors="coerce")
    candidate = _downcast_numeric(candidate, skip={"symbol", "trade_date", "entry_trade_date", "next_market_trade_date", "industry"})
    candidate = _rank_normalize_features(candidate, feature_cols, progress_path)
    keep_cols = [
        "symbol",
        "trade_date",
        "year",
        "industry",
        *feature_cols,
        *[label for label in LABELS if label in candidate.columns],
        "stable_upside_target",
        *[f"return_{int(h)}d" for h in horizons if f"return_{int(h)}d" in candidate.columns],
        "future_max_high_20d",
        "future_min_low_20d",
        "future_final_close_20d",
        "future_peak_day_20d",
    ]
    candidate = candidate[[col for col in keep_cols if col in candidate.columns]].copy()
    return candidate, feature_cols, active


def _feature_effects(frame: pd.DataFrame, feature_cols: list[str], labels: tuple[str, ...], years: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year in years:
        year_frame = frame[frame["year"].eq(int(year))]
        if year_frame.empty:
            continue
        all_mean = year_frame[feature_cols].mean(numeric_only=True)
        for label in labels:
            if label not in year_frame.columns:
                continue
            event = year_frame[year_frame[label].astype(bool)]
            if event.empty:
                continue
            event_mean = event[feature_cols].mean(numeric_only=True)
            diff = event_mean.sub(all_mean)
            for feature in feature_cols:
                value = diff.get(feature, np.nan)
                rows.append(
                    {
                        "year": int(year),
                        "label": label,
                        "feature": feature,
                        "sample_count": int(len(year_frame)),
                        "event_count": int(len(event)),
                        "event_rate": float(len(event) / len(year_frame)),
                        "all_mean": float(all_mean.get(feature, np.nan)),
                        "event_mean": float(event_mean.get(feature, np.nan)),
                        "effect": float(value) if math.isfinite(float(value)) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _strategy_weights(
    *,
    effects: pd.DataFrame,
    strategy: str,
    train_years: tuple[int, ...],
    top_features: int,
    min_same_sign_share: float,
) -> pd.DataFrame:
    spec = RISING_TYPES[strategy]
    combined: dict[str, dict[str, Any]] = {}
    for label, scale in spec["components"]:
        sub = effects[(effects["label"].eq(label)) & (effects["year"].isin(train_years))].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="feature", columns="year", values="effect", aggfunc="mean")
        if pivot.empty:
            continue
        mean_effect = pivot.mean(axis=1, skipna=True)
        signs = np.sign(pivot)
        target_sign = np.sign(mean_effect)
        same_sign_share = signs.eq(target_sign, axis=0).mean(axis=1)
        valid = mean_effect.notna() & mean_effect.ne(0.0) & same_sign_share.ge(float(min_same_sign_share))
        ranked = pd.DataFrame(
            {
                "feature": mean_effect.index,
                "mean_effect": mean_effect.to_numpy(dtype=np.float64),
                "same_sign_share": same_sign_share.to_numpy(dtype=np.float64),
            }
        )
        ranked = ranked[valid.to_numpy()]
        if ranked.empty:
            continue
        ranked["abs_effect"] = ranked["mean_effect"].abs()
        ranked = ranked.sort_values(["same_sign_share", "abs_effect"], ascending=[False, False], kind="mergesort").head(int(top_features))
        for row in ranked.to_dict("records"):
            feature = str(row["feature"])
            slot = combined.setdefault(
                feature,
                {
                    "strategy": strategy,
                    "feature": feature,
                    "weight": 0.0,
                    "components": [],
                    "max_same_sign_share": 0.0,
                },
            )
            slot["weight"] += float(row["mean_effect"]) * float(scale)
            slot["components"].append(f"{label}:{scale:+.2f}")
            slot["max_same_sign_share"] = max(float(slot["max_same_sign_share"]), float(row["same_sign_share"]))
    weights = pd.DataFrame(list(combined.values()))
    if weights.empty:
        return weights
    weights["components"] = weights["components"].map(lambda items: ",".join(items))
    weight_sum = float(weights["weight"].abs().sum())
    if weight_sum > 0:
        weights["weight"] = weights["weight"] / weight_sum
    weights["abs_weight"] = weights["weight"].abs()
    return weights.sort_values("abs_weight", ascending=False, kind="mergesort").reset_index(drop=True)


def _score_with_weights(frame: pd.DataFrame, weights: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    out = frame
    if weights.empty:
        out[score_col] = np.nan
        return out
    cols = [str(col) for col in weights["feature"].to_list() if str(col) in out.columns]
    if not cols:
        out[score_col] = np.nan
        return out
    aligned = weights.set_index("feature").loc[cols, "weight"].to_numpy(dtype=np.float32, copy=True)
    matrix = out[cols].to_numpy(dtype=np.float32, copy=True)
    out[score_col] = (np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0) @ aligned).astype(np.float32)
    return out


def _daily_topk(frame: pd.DataFrame, *, score_col: str, top_k: int) -> pd.DataFrame:
    work = frame[np.isfinite(pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=np.float32, copy=False))].copy()
    if work.empty:
        return work
    work = work.sort_values(["trade_date", score_col], ascending=[True, False], kind="mergesort")
    work["rank_in_day"] = work.groupby("trade_date", sort=False).cumcount() + 1
    return work[work["rank_in_day"] <= int(top_k)].copy()


def _first_hit_day(path: np.ndarray, threshold: float, *, direction: str) -> np.ndarray:
    if direction == "up":
        hit = np.asarray(path >= float(threshold), dtype=bool)
    elif direction == "down":
        hit = np.asarray(path <= -abs(float(threshold)), dtype=bool)
    else:
        raise ValueError(direction)
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1).astype(np.int16) + 1
    first[~any_hit] = 0
    return first


def _evaluate_selection(
    *,
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    strategy: str,
    method: str,
    top_k: int,
    horizon: int,
    cost_bps: float,
    fold_kind: str,
    fold_year: int,
    pool_k: int | None = None,
    model_target: str = "",
) -> dict[str, Any]:
    metric: dict[str, Any] = {
        "fold_kind": fold_kind,
        "fold_year": int(fold_year),
        "method": method,
        "strategy": strategy,
        "pool_k": int(pool_k or 0),
        "model_target": model_target,
        "top_k": int(top_k),
        "horizon": int(horizon),
        "selected_count": int(len(selected)),
    }
    return_col = f"return_{int(horizon)}d"
    if universe.empty or selected.empty or return_col not in universe.columns:
        return metric
    base = universe[["trade_date", return_col, "hit_up5_before_down3_20d", "hit_up10_before_down5_20d"]].copy()
    sel = selected[["trade_date", return_col, "hit_up5_before_down3_20d", "hit_up10_before_down5_20d", "rise_then_fade_10pct_20d", "persistent_upside_10pct_20d"]].copy()
    cost = float(cost_bps) / 10000.0
    base["net_ret"] = pd.to_numeric(base[return_col], errors="coerce") - cost
    sel["net_ret"] = pd.to_numeric(sel[return_col], errors="coerce") - cost
    base_daily = base.groupby("trade_date", sort=True)["net_ret"].mean()
    selected_daily = sel.groupby("trade_date", sort=True)["net_ret"].mean()
    aligned = pd.concat([selected_daily.rename("selected_ret"), base_daily.rename("base_ret")], axis=1, join="inner").dropna()
    if aligned.empty:
        return metric
    excess = aligned["selected_ret"] - aligned["base_ret"]
    metric.update(
        {
            "date_count": int(len(aligned)),
            "mean_net_return": float(aligned["selected_ret"].mean()),
            "mean_base_return": float(aligned["base_ret"].mean()),
            "mean_excess_return": float(excess.mean()),
            "mean_excess_bps": float(excess.mean() * 10000.0),
            "std_daily_return": float(aligned["selected_ret"].std(ddof=1)),
            "win_day_rate": float((aligned["selected_ret"] > 0).mean()),
            "excess_day_positive_rate": float((excess > 0).mean()),
            "selected_success_up5_down3": float(sel["hit_up5_before_down3_20d"].astype(bool).mean()),
            "base_success_up5_down3": float(base["hit_up5_before_down3_20d"].astype(bool).mean()),
            "selected_success_up10_down5": float(sel["hit_up10_before_down5_20d"].astype(bool).mean()),
            "base_success_up10_down5": float(base["hit_up10_before_down5_20d"].astype(bool).mean()),
            "selected_persistent": float(sel["persistent_upside_10pct_20d"].astype(bool).mean()),
            "selected_fade": float(sel["rise_then_fade_10pct_20d"].astype(bool).mean()),
        }
    )
    std = metric["std_daily_return"]
    metric["cohort_ir"] = float(metric["mean_net_return"] / std) if math.isfinite(std) and std > 0 else float("nan")
    return metric


def _daily_return_frame(universe: pd.DataFrame, selected: pd.DataFrame, *, horizon: int, cost_bps: float) -> pd.DataFrame:
    return_col = f"return_{int(horizon)}d"
    if universe.empty or selected.empty or return_col not in universe.columns:
        return pd.DataFrame()
    cost = float(cost_bps) / 10000.0
    base = universe[["trade_date", return_col]].copy()
    sel = selected[["trade_date", return_col]].copy()
    base["base_ret"] = pd.to_numeric(base[return_col], errors="coerce") - cost
    sel["selected_ret"] = pd.to_numeric(sel[return_col], errors="coerce") - cost
    out = pd.concat(
        [
            sel.groupby("trade_date", sort=True)["selected_ret"].mean(),
            base.groupby("trade_date", sort=True)["base_ret"].mean(),
        ],
        axis=1,
        join="inner",
    ).dropna().reset_index()
    out["excess_ret"] = out["selected_ret"] - out["base_ret"]
    out["cum_selected_ret_index"] = (1.0 + out["selected_ret"].fillna(0.0)).cumprod()
    out["cum_excess_ret_index"] = (1.0 + out["excess_ret"].fillna(0.0)).cumprod()
    return out


def _development_score(results: pd.DataFrame) -> pd.DataFrame:
    dev = results[results["fold_kind"].eq("rolling_validation")].copy()
    if dev.empty:
        return pd.DataFrame()
    grouped = dev.groupby(["method", "strategy", "pool_k", "model_target", "top_k", "horizon"], sort=True, dropna=False)
    out = grouped.agg(
        fold_count=("fold_year", "nunique"),
        mean_excess_bps=("mean_excess_bps", "mean"),
        median_excess_bps=("mean_excess_bps", "median"),
        std_excess_bps=("mean_excess_bps", "std"),
        positive_year_rate=("mean_excess_bps", lambda s: float((s > 0).mean())),
        mean_net_return=("mean_net_return", "mean"),
        mean_success_up10_down5=("selected_success_up10_down5", "mean"),
        mean_base_success_up10_down5=("base_success_up10_down5", "mean"),
        mean_persistent=("selected_persistent", "mean"),
        mean_fade=("selected_fade", "mean"),
    ).reset_index()
    out["success_up10_down5_lift"] = out["mean_success_up10_down5"] - out["mean_base_success_up10_down5"]
    out["development_score"] = (
        out["mean_excess_bps"].fillna(-999.0)
        + 12.0 * (out["positive_year_rate"].fillna(0.0) - 0.5)
        - 0.12 * out["std_excess_bps"].fillna(0.0)
        + 85.0 * out["success_up10_down5_lift"].fillna(0.0)
        + 20.0 * out["mean_persistent"].fillna(0.0)
        - 10.0 * out["mean_fade"].fillna(0.0)
    )
    out = out.sort_values("development_score", ascending=False, kind="mergesort").reset_index(drop=True)
    out["development_rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def _fit_sgd_model(
    train_pool: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    seed: int,
    epochs: int,
    chunk_size: int,
):
    from sklearn.linear_model import SGDClassifier

    y_all = train_pool[target_col].astype(bool).to_numpy()
    if len(np.unique(y_all)) < 2:
        return None
    pos = max(1, int(y_all.sum()))
    neg = max(1, int((~y_all).sum()))
    pos_w = len(y_all) / (2.0 * pos)
    neg_w = len(y_all) / (2.0 * neg)
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.05,
        random_state=int(seed),
        fit_intercept=True,
        learning_rate="optimal",
        max_iter=1,
        tol=None,
        shuffle=False,
    )
    classes = np.array([False, True])
    order = np.arange(len(train_pool), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    first = True
    for epoch in range(int(epochs)):
        rng.shuffle(order)
        for slc in _chunk_slices(len(order), chunk_size):
            idx = order[slc]
            batch = train_pool.iloc[idx]
            x = batch[feature_cols].to_numpy(dtype=np.float32, copy=True)
            y = batch[target_col].astype(bool).to_numpy()
            sample_weight = np.where(y, pos_w, neg_w).astype(np.float32)
            if first:
                model.partial_fit(x, y, classes=classes, sample_weight=sample_weight)
                first = False
            else:
                model.partial_fit(x, y, sample_weight=sample_weight)
    return model


def _model_score(frame: pd.DataFrame, model: Any, *, feature_cols: list[str], score_col: str, chunk_size: int) -> pd.DataFrame:
    out = frame
    scores = np.full(len(out), np.nan, dtype=np.float32)
    for slc in _chunk_slices(len(out), chunk_size):
        x = out.iloc[slc][feature_cols].to_numpy(dtype=np.float32, copy=True)
        if hasattr(model, "predict_proba"):
            scores[slc] = model.predict_proba(x)[:, 1].astype(np.float32)
        else:
            raw = model.decision_function(x)
            scores[slc] = (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
    out[score_col] = scores
    return out


def _plot_outputs(output_dir: Path, all_results: pd.DataFrame, test_results: pd.DataFrame, daily_returns: pd.DataFrame) -> list[str]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    if not all_results.empty:
        dev = all_results[all_results["fold_kind"].eq("rolling_validation")].copy()
        if not dev.empty:
            top = _development_score(dev).head(10)
            labels = [f"{r.method}:{r.strategy}:K{int(r.top_k)}:H{int(r.horizon)}" for r in top.itertuples()]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(range(len(top)), top["mean_excess_bps"], color="#2f6f9f")
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.set_xticks(range(len(top)))
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
            ax.set_ylabel("Mean rolling excess bps")
            ax.set_title("Top development configurations")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            path = chart_dir / "top_development_configs.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            outputs.append(str(path.resolve()))
    if not test_results.empty:
        top = test_results.sort_values("development_rank").head(12)
        labels = [f"r{int(r.development_rank)} {r.method}:{r.strategy}:K{int(r.top_k)}:H{int(r.horizon)}" for r in top.itertuples()]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(top)), top["mean_excess_bps"], color="#926c15")
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("2025 frozen excess bps")
        ax.set_title("Frozen test result by development rank")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = chart_dir / "frozen_test_by_dev_rank.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    if not daily_returns.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily_returns["trade_date"], daily_returns["cum_selected_ret_index"], label="selected")
        ax.plot(daily_returns["trade_date"], daily_returns["cum_excess_ret_index"], label="excess")
        step = max(1, len(daily_returns) // 8)
        ax.set_xticks(daily_returns["trade_date"].iloc[::step])
        ax.tick_params(axis="x", rotation=35)
        ax.set_title("Frozen test daily cohort index")
        ax.set_ylabel("Index")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = chart_dir / "winner_frozen_test_daily_index.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs


def _build_report(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    config_selection: pd.DataFrame,
    test_results: pd.DataFrame,
    winner_weights: pd.DataFrame,
    chart_paths: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# QDP v2 Rising Type Strategy Lab")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("This run uses QDP v2 raw daily bars plus 1m-derived intraday/limit structure. It does not use symbol ids or the old 307-feature training pack.")
    lines.append("Signals are formed after the signal-day close, entries are next-market-day open, open-limit-up entries are blocked, and 2025 is kept as frozen test.")
    lines.append("")
    lines.append("## Development Winner")
    lines.append("")
    if not config_selection.empty:
        row = config_selection.iloc[0].to_dict()
        lines.append(
            f"- rank=1 method={row['method']} strategy={row['strategy']} pool_k={int(row['pool_k'])} "
            f"top_k={int(row['top_k'])} horizon={int(row['horizon'])} "
            f"dev_excess={row['mean_excess_bps']:.2f} bps positive_year={row['positive_year_rate']:.2%} "
            f"up10/down5_lift={row['success_up10_down5_lift']:.2%}"
        )
    lines.append("")
    lines.append("## Frozen Test")
    lines.append("")
    if not test_results.empty:
        for row in test_results.sort_values("development_rank").head(12).to_dict("records"):
            lines.append(
                f"- dev_rank={int(row['development_rank'])} {row['method']} {row['strategy']} "
                f"pool_k={int(row.get('pool_k', 0))} top{int(row['top_k'])} H{int(row['horizon'])}: "
                f"net={row.get('mean_net_return', float('nan')) * 100:.3f}% "
                f"base={row.get('mean_base_return', float('nan')) * 100:.3f}% "
                f"excess={row.get('mean_excess_bps', float('nan')):.2f} bps "
                f"up10/down5={row.get('selected_success_up10_down5', float('nan')):.2%} "
                f"fade={row.get('selected_fade', float('nan')):.2%}"
            )
    lines.append("")
    lines.append("## Winner Rule Weights")
    lines.append("")
    for row in winner_weights.head(25).to_dict("records"):
        lines.append(f"- {row['feature']}: weight={row['weight']:.5f}, components={row.get('components', '')}")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for path in chart_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Boundaries")
    lines.append("")
    lines.append("- This is daily cohort validation, not a cash-constrained overlapping portfolio simulator.")
    lines.append("- The model branch is trained only inside a rule-generated candidate pool; if 2025 is used for further changes, a later unseen period must become the new final test.")
    path = output_dir / "rising_type_strategy_lab_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


@dataclass(frozen=True)
class LabConfig:
    qdp_root: Path
    output_root: Path
    run_tag: str
    event_start: str
    event_end: str
    development_years: tuple[int, ...]
    fold_years: tuple[int, ...]
    test_year: int
    top_k: tuple[int, ...]
    horizons: tuple[int, ...]
    pool_k: tuple[int, ...]
    cost_bps: float
    top_features: int
    min_same_sign_share: float
    model_epochs: int
    chunk_size: int
    seed: int
    skip_model: bool


def run_lab(config: LabConfig) -> dict[str, Any]:
    root = config.qdp_root.resolve()
    output_dir = config.output_root / f"{config.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "starting", "updated_at": _now()})

    frame, feature_cols, active = _load_research_frame(
        root=root,
        event_start=config.event_start,
        event_end=config.event_end,
        horizons=config.horizons,
        progress_path=progress_path,
    )
    label_cols = tuple(label for label in (*LABELS, "stable_upside_target") if label in frame.columns)
    effects = _feature_effects(frame, feature_cols, label_cols, tuple(sorted(set(config.development_years))))
    _write_json(progress_path, {"status": "feature_effects_complete", "rows": int(len(effects)), "updated_at": _now()})

    all_weights: list[pd.DataFrame] = []
    result_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for fold_year in config.fold_years:
        train_years = tuple(year for year in config.development_years if int(year) < int(fold_year))
        if len(train_years) < 3:
            continue
        train = frame[frame["year"].isin(train_years)].copy()
        validation = frame[frame["year"].eq(int(fold_year))].copy()
        if train.empty or validation.empty:
            continue
        strategy_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        for strategy in RISING_TYPES:
            weights = _strategy_weights(
                effects=effects,
                strategy=strategy,
                train_years=train_years,
                top_features=config.top_features,
                min_same_sign_share=config.min_same_sign_share,
            )
            if weights.empty:
                continue
            tmp_weights = weights.copy()
            tmp_weights["fold_kind"] = "rolling_validation"
            tmp_weights["fold_year"] = int(fold_year)
            tmp_weights["train_years"] = f"{min(train_years)}-{max(train_years)}"
            all_weights.append(tmp_weights)
            score_col = f"rule_score_{strategy}"
            train_scored = _score_with_weights(train.copy(), weights, score_col=score_col)
            val_scored = _score_with_weights(validation.copy(), weights, score_col=score_col)
            strategy_cache[strategy] = (weights, train_scored, val_scored)
            for top_k in config.top_k:
                selected = _daily_topk(val_scored, score_col=score_col, top_k=int(top_k))
                for horizon in config.horizons:
                    result_rows.append(
                        _evaluate_selection(
                            universe=validation,
                            selected=selected,
                            strategy=strategy,
                            method="rule",
                            top_k=int(top_k),
                            horizon=int(horizon),
                            cost_bps=config.cost_bps,
                            fold_kind="rolling_validation",
                            fold_year=int(fold_year),
                        )
                    )
        if not config.skip_model:
            for strategy, (_weights, train_scored, val_scored) in strategy_cache.items():
                target = str(RISING_TYPES[strategy]["target"])
                if target not in train_scored.columns:
                    continue
                for pool_k in config.pool_k:
                    score_col = f"rule_score_{strategy}"
                    train_pool = _daily_topk(train_scored, score_col=score_col, top_k=int(pool_k))
                    val_pool = _daily_topk(val_scored, score_col=score_col, top_k=int(pool_k))
                    if train_pool.empty or val_pool.empty:
                        continue
                    model = _fit_sgd_model(
                        train_pool,
                        feature_cols=feature_cols,
                        target_col=target,
                        seed=config.seed + int(fold_year),
                        epochs=config.model_epochs,
                        chunk_size=config.chunk_size,
                    )
                    if model is None:
                        continue
                    model_score_col = f"model_score_{strategy}_{pool_k}"
                    val_pool = _model_score(val_pool.copy(), model, feature_cols=feature_cols, score_col=model_score_col, chunk_size=config.chunk_size)
                    for top_k in config.top_k:
                        selected = _daily_topk(val_pool, score_col=model_score_col, top_k=int(top_k))
                        for horizon in config.horizons:
                            model_rows.append(
                                _evaluate_selection(
                                    universe=validation,
                                    selected=selected,
                                    strategy=strategy,
                                    method="model_sgd_in_rule_pool",
                                    pool_k=int(pool_k),
                                    model_target=target,
                                    top_k=int(top_k),
                                    horizon=int(horizon),
                                    cost_bps=config.cost_bps,
                                    fold_kind="rolling_validation",
                                    fold_year=int(fold_year),
                                )
                            )
        _write_json(progress_path, {"status": "fold_complete", "fold_year": int(fold_year), "updated_at": _now()})

    rolling_results = pd.DataFrame([*result_rows, *model_rows])
    if rolling_results.empty:
        raise RuntimeError("no rolling validation results were produced")
    config_selection = _development_score(rolling_results)
    if config_selection.empty:
        raise RuntimeError("no valid development config selection was produced")

    final_train_years = tuple(int(year) for year in config.development_years)
    train_full = frame[frame["year"].isin(final_train_years)].copy()
    test = frame[frame["year"].eq(int(config.test_year))].copy()
    final_weights_by_strategy: dict[str, pd.DataFrame] = {}
    test_rows: list[dict[str, Any]] = []
    daily_returns = pd.DataFrame()
    winner_weights = pd.DataFrame()
    final_rule_score_cache: dict[str, pd.DataFrame] = {}
    final_model_cache: dict[tuple[str, int], pd.DataFrame] = {}

    for strategy in RISING_TYPES:
        weights = _strategy_weights(
            effects=effects,
            strategy=strategy,
            train_years=final_train_years,
            top_features=config.top_features,
            min_same_sign_share=config.min_same_sign_share,
        )
        final_weights_by_strategy[strategy] = weights
        if not weights.empty:
            tmp = weights.copy()
            tmp["fold_kind"] = "final_train"
            tmp["fold_year"] = int(config.test_year)
            tmp["train_years"] = f"{min(final_train_years)}-{max(final_train_years)}"
            all_weights.append(tmp)

    for row in config_selection.head(50).to_dict("records"):
        method = str(row["method"])
        strategy = str(row["strategy"])
        top_k = int(row["top_k"])
        horizon = int(row["horizon"])
        pool_k = int(row.get("pool_k", 0) or 0)
        target = str(row.get("model_target", "") or "")
        weights = final_weights_by_strategy.get(strategy, pd.DataFrame())
        if weights.empty or test.empty:
            continue
        score_col = f"final_rule_score_{strategy}"
        if strategy not in final_rule_score_cache:
            final_rule_score_cache[strategy] = _score_with_weights(test.copy(), weights, score_col=score_col)
        if method == "rule":
            selected = _daily_topk(final_rule_score_cache[strategy], score_col=score_col, top_k=top_k)
        else:
            key = (strategy, pool_k)
            model_score_col = f"final_model_score_{strategy}_{pool_k}"
            if key not in final_model_cache:
                train_scored = _score_with_weights(train_full.copy(), weights, score_col=score_col)
                train_pool = _daily_topk(train_scored, score_col=score_col, top_k=pool_k)
                test_pool = _daily_topk(final_rule_score_cache[strategy].copy(), score_col=score_col, top_k=pool_k)
                model = _fit_sgd_model(
                    train_pool,
                    feature_cols=feature_cols,
                    target_col=target or str(RISING_TYPES[strategy]["target"]),
                    seed=config.seed + 1000,
                    epochs=config.model_epochs,
                    chunk_size=config.chunk_size,
                )
                if model is None:
                    continue
                final_model_cache[key] = _model_score(test_pool, model, feature_cols=feature_cols, score_col=model_score_col, chunk_size=config.chunk_size)
            selected = _daily_topk(final_model_cache[key], score_col=model_score_col, top_k=top_k)
        metrics = _evaluate_selection(
            universe=test,
            selected=selected,
            strategy=strategy,
            method=method,
            pool_k=pool_k,
            model_target=target,
            top_k=top_k,
            horizon=horizon,
            cost_bps=config.cost_bps,
            fold_kind="frozen_test",
            fold_year=int(config.test_year),
        )
        metrics["development_rank"] = int(row["development_rank"])
        metrics["development_score"] = float(row["development_score"])
        test_rows.append(metrics)
        if int(row["development_rank"]) == 1:
            winner_weights = weights.copy()
            daily_returns = _daily_return_frame(test, selected, horizon=horizon, cost_bps=config.cost_bps)

    test_results = pd.DataFrame(test_rows).sort_values("development_rank", kind="mergesort") if test_rows else pd.DataFrame()
    weights_frame = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    outputs = {
        "feature_effects_csv": _write_csv(output_dir / "feature_effects_by_year.csv", effects),
        "rolling_results_csv": _write_csv(output_dir / "rolling_validation_results.csv", rolling_results),
        "development_config_selection_csv": _write_csv(output_dir / "development_config_selection.csv", config_selection),
        "frozen_test_results_csv": _write_csv(output_dir / "frozen_test_results.csv", test_results),
        "rule_weights_csv": _write_csv(output_dir / "rule_weights_by_fold.csv", weights_frame),
        "winner_rule_weights_csv": _write_csv(output_dir / "winner_rule_weights.csv", winner_weights),
        "winner_frozen_daily_returns_csv": _write_csv(output_dir / "winner_frozen_daily_returns.csv", daily_returns),
    }
    chart_paths = _plot_outputs(output_dir, rolling_results, test_results, daily_returns)
    outputs["charts"] = chart_paths

    summary: dict[str, Any] = {
        "artifact_type": "qdp_v2_rising_type_strategy_lab",
        "generated_at": _now(),
        "qdp_root": str(root),
        "active_manifest": str((root / "active" / "active.json").resolve()),
        "active_datasets": dict(active.get("datasets", {}) or {}),
        "output_dir": str(output_dir.resolve()),
        "event_start": config.event_start,
        "event_end": config.event_end,
        "candidate_count": int(len(frame)),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "development_years": list(config.development_years),
        "fold_years": list(config.fold_years),
        "test_year": int(config.test_year),
        "top_k": list(config.top_k),
        "horizons": list(config.horizons),
        "pool_k": list(config.pool_k),
        "cost_bps": float(config.cost_bps),
        "rising_types": RISING_TYPES,
        "winner": config_selection.iloc[0].to_dict() if not config_selection.empty else {},
        "winner_test_result": (
            test_results[test_results["development_rank"].eq(1)].iloc[0].to_dict()
            if (not test_results.empty and test_results["development_rank"].eq(1).any())
            else {}
        ),
        "outputs": outputs,
    }
    report_path = _build_report(
        output_dir=output_dir,
        summary=summary,
        config_selection=config_selection,
        test_results=test_results,
        winner_weights=winner_weights,
        chart_paths=chart_paths,
    )
    summary["outputs"]["report_md"] = report_path
    summary_path = _write_json(output_dir / "rising_type_strategy_lab_summary.json", summary)
    summary["outputs"]["summary_json"] = summary_path
    _write_json(progress_path, {"status": "completed", "summary_json": summary_path, "report_md": report_path, "updated_at": _now()})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover rising-stock types from QDP v2 raw data and validate rule/model selectors on frozen test.")
    parser.add_argument("--qdp-root", default=str(DEFAULT_QDP_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-tag", default="qdp_v2_rising_type_strategy_lab")
    parser.add_argument("--event-start", default=DEFAULT_EVENT_START)
    parser.add_argument("--event-end", default=DEFAULT_EVENT_END)
    parser.add_argument("--development-years", default="2012-2024")
    parser.add_argument("--fold-years", default="2018-2024")
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    parser.add_argument("--top-k", default="20,50,100")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--pool-k", default="300")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--top-features", type=int, default=25)
    parser.add_argument("--min-same-sign-share", type=float, default=0.60)
    parser.add_argument("--model-epochs", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_lab(
        LabConfig(
            qdp_root=Path(args.qdp_root),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            event_start=str(args.event_start),
            event_end=str(args.event_end),
            development_years=_parse_year_range(args.development_years, default=DEFAULT_DEVELOPMENT_YEARS),
            fold_years=_parse_year_range(args.fold_years, default=DEFAULT_FOLD_YEARS),
            test_year=int(args.test_year),
            top_k=_parse_csv_ints(args.top_k, default=DEFAULT_TOP_K),
            horizons=_parse_csv_ints(args.horizons, default=DEFAULT_HORIZONS),
            pool_k=_parse_csv_ints(args.pool_k, default=DEFAULT_POOL_K),
            cost_bps=float(args.cost_bps),
            top_features=int(args.top_features),
            min_same_sign_share=float(args.min_same_sign_share),
            model_epochs=int(args.model_epochs),
            chunk_size=int(args.chunk_size),
            seed=int(args.seed),
            skip_model=bool(args.skip_model),
        )
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(summary.get("outputs", {}).get("report_md", ""))
    return summary


if __name__ == "__main__":
    main()
