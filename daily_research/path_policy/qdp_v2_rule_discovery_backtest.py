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


DEFAULT_TRAINING_PACK_MANIFEST = Path(
    "daily_research/data/research_store/training_pack/"
    "qdp_v2_alpha_v2_full_contract_2012_2025_20260702_01_training_pack/"
    "qdp_training_pack_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/rule_discovery_backtest")
DEFAULT_DEVELOPMENT_YEARS = tuple(range(2012, 2025))
DEFAULT_FOLD_YEARS = tuple(range(2018, 2025))
DEFAULT_TEST_YEAR = 2025
DEFAULT_TOP_K = (20, 50, 100)
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_COST_BPS = 20.0


TARGET_LABELS = (
    "top10_return_20d",
    "hit_up10_before_down5_20d",
    "persistent_upside_10pct_20d",
    "rise_then_fade_10pct_20d",
)


STRATEGY_COMPONENTS: dict[str, tuple[tuple[str, float], ...]] = {
    "top20d_profile": (("top10_return_20d", 1.0),),
    "trade_path_profile": (("hit_up10_before_down5_20d", 1.0),),
    "persistent_profile": (("persistent_upside_10pct_20d", 1.0),),
    "anti_fade_profile": (("persistent_upside_10pct_20d", 1.0), ("rise_then_fade_10pct_20d", -1.0)),
    "blend_profile": (
        ("top10_return_20d", 0.35),
        ("hit_up10_before_down5_20d", 0.35),
        ("persistent_upside_10pct_20d", 0.25),
        ("rise_then_fade_10pct_20d", -0.20),
    ),
}


EXCLUDED_FEATURE_PREFIXES = ("market_", "benchmark_")
EXCLUDED_FEATURES = {
    "current_price",
    "in_pool",
    "history_valid_days_252",
    "history_valid_ratio_252",
    "core_ohlcv_valid_ratio_252",
    "is_low_history_like",
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


def _resolve_path(path: str | Path, *, root: Path | None = None) -> Path:
    out = Path(path)
    if not out.is_absolute() and root is not None:
        out = root / out
    return out


def _open_memmap(meta: Mapping[str, Any]) -> np.memmap:
    path = Path(str(meta.get("path", "") or ""))
    dtype = str(meta.get("dtype", "float32") or "float32")
    shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
    if not path.exists():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _chunk_slices(length: int, chunk_size: int) -> Iterable[slice]:
    chunk = max(1, int(chunk_size))
    for start in range(0, int(length), chunk):
        yield slice(start, min(start + chunk, int(length)))


def _first_hit_day(cumulative_path: np.ndarray, threshold: float, *, direction: str) -> np.ndarray:
    if direction == "up":
        hit = np.asarray(cumulative_path >= float(threshold), dtype=bool)
    elif direction == "down":
        hit = np.asarray(cumulative_path <= -abs(float(threshold)), dtype=bool)
    else:
        raise ValueError(f"unsupported direction: {direction}")
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1).astype(np.int16) + 1
    first[~any_hit] = 0
    return first


def _label_masks(labels: Mapping[str, np.memmap], positions: np.ndarray) -> dict[str, np.ndarray]:
    ranks = np.asarray(labels["rank_by_horizon"][positions], dtype=np.float32)
    cumulative_path = np.asarray(labels["cumulative_return_1to20"][positions], dtype=np.float32)
    valid_path = np.isfinite(cumulative_path).all(axis=1)
    max_upside = np.nanmax(cumulative_path, axis=1)
    final_return = cumulative_path[:, -1]
    masks = {
        "top10_return_20d": np.isfinite(ranks[:, 4]) & (ranks[:, 4] >= 0.90),
        "hit_up10_before_down5_20d": valid_path
        & (_first_hit_day(cumulative_path, 0.10, direction="up") > 0)
        & (
            (_first_hit_day(cumulative_path, 0.05, direction="down") == 0)
            | (_first_hit_day(cumulative_path, 0.10, direction="up") <= _first_hit_day(cumulative_path, 0.05, direction="down"))
        ),
        "persistent_upside_10pct_20d": valid_path & (max_upside >= 0.10) & (final_return >= (max_upside - 0.02)),
        "rise_then_fade_10pct_20d": valid_path & (max_upside >= 0.10) & (final_return <= (max_upside - 0.03)),
    }
    return masks


def _tradable_mask(labels: Mapping[str, np.memmap], positions: np.ndarray) -> np.ndarray:
    tradeable = np.asarray(labels["entry_tradeable"][positions], dtype=np.float32) > 0.5
    blocked = np.asarray(labels["entry_limit_up_buy_blocked"][positions], dtype=np.float32) > 0.5
    suspended = np.asarray(labels["entry_suspended_or_no_open"][positions], dtype=np.float32) > 0.5
    return tradeable & (~blocked) & (~suspended)


def _feature_allowed(name: str) -> bool:
    if name in EXCLUDED_FEATURES:
        return False
    return not any(name.startswith(prefix) for prefix in EXCLUDED_FEATURE_PREFIXES)


@dataclass
class EffectStore:
    years: tuple[int, ...]
    labels: tuple[str, ...]
    feature_columns: list[str]
    sample_count: np.ndarray
    feature_count_all: np.ndarray
    feature_sum_all: np.ndarray
    label_sample_count: np.ndarray
    label_feature_count: np.ndarray
    label_feature_sum: np.ndarray

    @classmethod
    def create(cls, *, years: tuple[int, ...], labels: tuple[str, ...], feature_columns: list[str]) -> "EffectStore":
        y = len(years)
        l = len(labels)
        f = len(feature_columns)
        return cls(
            years=years,
            labels=labels,
            feature_columns=feature_columns,
            sample_count=np.zeros(y, dtype=np.int64),
            feature_count_all=np.zeros((y, f), dtype=np.int64),
            feature_sum_all=np.zeros((y, f), dtype=np.float64),
            label_sample_count=np.zeros((l, y), dtype=np.int64),
            label_feature_count=np.zeros((l, y, f), dtype=np.int64),
            label_feature_sum=np.zeros((l, y, f), dtype=np.float64),
        )

    def effects_for(self, label: str, years: Iterable[int]) -> np.ndarray:
        label_idx = self.labels.index(label)
        year_indices = [self.years.index(int(year)) for year in years if int(year) in self.years]
        if not year_indices:
            return np.empty((0, len(self.feature_columns)), dtype=np.float64)
        all_mean = np.divide(
            self.feature_sum_all[year_indices],
            np.maximum(self.feature_count_all[year_indices], 1),
            out=np.full((len(year_indices), len(self.feature_columns)), np.nan, dtype=np.float64),
            where=self.feature_count_all[year_indices] > 0,
        )
        label_mean = np.divide(
            self.label_feature_sum[label_idx, year_indices],
            np.maximum(self.label_feature_count[label_idx, year_indices], 1),
            out=np.full((len(year_indices), len(self.feature_columns)), np.nan, dtype=np.float64),
            where=self.label_feature_count[label_idx, year_indices] > 0,
        )
        return label_mean - all_mean


def _build_effect_store(
    *,
    feature_panel: np.memmap,
    labels: Mapping[str, np.memmap],
    sample_index: pd.DataFrame,
    feature_columns: list[str],
    years: tuple[int, ...],
    chunk_size: int,
    progress_path: Path,
) -> EffectStore:
    store = EffectStore.create(years=years, labels=TARGET_LABELS, feature_columns=feature_columns)
    year_to_idx = {int(year): idx for idx, year in enumerate(years)}
    label_to_idx = {label: idx for idx, label in enumerate(TARGET_LABELS)}
    selected = sample_index[sample_index["year"].isin(year_to_idx)].copy()
    sample_pos_all = selected["_sample_pos"].to_numpy(dtype=np.int64, copy=True)
    stock_pos_all = selected["global_stock_pos"].to_numpy(dtype=np.int64, copy=True)
    date_pos_all = selected["global_date_pos"].to_numpy(dtype=np.int64, copy=True)
    year_all = selected["year"].to_numpy(dtype=np.int16, copy=True)
    feature_indices = np.arange(len(feature_columns), dtype=np.int64)
    processed = 0
    for chunk_idx, slc in enumerate(_chunk_slices(len(selected), chunk_size), start=1):
        sample_pos = sample_pos_all[slc]
        stock_pos = stock_pos_all[slc]
        date_pos = date_pos_all[slc]
        row_year = year_all[slc]
        valid = (
            (sample_pos >= 0)
            & (stock_pos >= 0)
            & (date_pos >= 0)
            & (stock_pos < feature_panel.shape[0])
            & (date_pos < feature_panel.shape[1])
        )
        if not np.all(valid):
            sample_pos = sample_pos[valid]
            stock_pos = stock_pos[valid]
            date_pos = date_pos[valid]
            row_year = row_year[valid]
        if len(sample_pos) == 0:
            continue
        features = np.asarray(feature_panel[stock_pos, date_pos, :][:, feature_indices], dtype=np.float32)
        finite = np.isfinite(features)
        masks = _label_masks(labels, sample_pos)
        for year in np.unique(row_year):
            yidx = year_to_idx[int(year)]
            year_mask = row_year == int(year)
            if not np.any(year_mask):
                continue
            store.sample_count[yidx] += int(year_mask.sum())
            yf = features[year_mask]
            yfinite = finite[year_mask]
            store.feature_count_all[yidx] += yfinite.sum(axis=0, dtype=np.int64)
            store.feature_sum_all[yidx] += np.where(yfinite, yf, 0.0).sum(axis=0, dtype=np.float64)
            for label, label_mask in masks.items():
                selected_mask = year_mask & label_mask
                count = int(selected_mask.sum())
                if count <= 0:
                    continue
                lidx = label_to_idx[label]
                lf = finite[selected_mask]
                lvalues = features[selected_mask]
                store.label_sample_count[lidx, yidx] += count
                store.label_feature_count[lidx, yidx] += lf.sum(axis=0, dtype=np.int64)
                store.label_feature_sum[lidx, yidx] += np.where(lf, lvalues, 0.0).sum(axis=0, dtype=np.float64)
        processed += int(len(sample_pos))
        if chunk_idx % 8 == 0:
            _write_json(progress_path, {"status": "effect_scan", "processed_rows": processed, "total_rows": int(len(selected)), "updated_at": _now()})
    return store


def _component_weights(
    *,
    store: EffectStore,
    label: str,
    train_years: tuple[int, ...],
    top_features: int,
    min_same_sign_share: float,
) -> pd.DataFrame:
    effects = store.effects_for(label, train_years)
    if effects.size == 0:
        return pd.DataFrame(columns=["feature", "feature_index", "weight", "mean_effect", "same_sign_share", "abs_mean_effect"])
    mean_effect = np.nanmean(effects, axis=0)
    sign = np.sign(mean_effect)
    same_sign = np.nanmean(np.sign(effects) == sign[None, :], axis=0)
    same_sign = np.asarray(same_sign, dtype=np.float64)
    same_sign[sign == 0] = 0.0
    allowed = np.asarray([_feature_allowed(name) for name in store.feature_columns], dtype=bool)
    valid = allowed & np.isfinite(mean_effect) & (np.abs(mean_effect) > 0) & (same_sign >= float(min_same_sign_share))
    rows: list[dict[str, Any]] = []
    for idx in np.where(valid)[0]:
        rows.append(
            {
                "feature": store.feature_columns[int(idx)],
                "feature_index": int(idx),
                "weight": float(mean_effect[int(idx)]),
                "mean_effect": float(mean_effect[int(idx)]),
                "same_sign_share": float(same_sign[int(idx)]),
                "abs_mean_effect": float(abs(mean_effect[int(idx)])),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["same_sign_share", "abs_mean_effect"], ascending=[False, False], kind="mergesort").head(int(top_features))


def _strategy_weights(
    *,
    store: EffectStore,
    strategy_name: str,
    train_years: tuple[int, ...],
    top_features: int,
    min_same_sign_share: float,
) -> pd.DataFrame:
    combined: dict[int, dict[str, Any]] = {}
    components = STRATEGY_COMPONENTS[strategy_name]
    for label, scale in components:
        comp = _component_weights(
            store=store,
            label=label,
            train_years=train_years,
            top_features=top_features,
            min_same_sign_share=min_same_sign_share,
        )
        for row in comp.to_dict("records"):
            idx = int(row["feature_index"])
            if idx not in combined:
                combined[idx] = {
                    "strategy": strategy_name,
                    "feature": row["feature"],
                    "feature_index": idx,
                    "weight": 0.0,
                    "components": [],
                }
            combined[idx]["weight"] += float(row["weight"]) * float(scale)
            combined[idx]["components"].append(f"{label}:{scale:+.2f}")
    frame = pd.DataFrame(list(combined.values()))
    if frame.empty:
        return frame
    weight_sum = float(frame["weight"].abs().sum())
    if weight_sum > 0:
        frame["weight"] = frame["weight"] / weight_sum
    frame["abs_weight"] = frame["weight"].abs()
    frame["components"] = frame["components"].map(lambda values: ",".join(values))
    return frame.sort_values("abs_weight", ascending=False, kind="mergesort")


def _score_subset(
    *,
    feature_panel: np.memmap,
    subset: pd.DataFrame,
    weights: pd.DataFrame,
    chunk_size: int,
) -> np.ndarray:
    if subset.empty or weights.empty:
        return np.full(len(subset), np.nan, dtype=np.float32)
    feature_idx = weights["feature_index"].to_numpy(dtype=np.int64, copy=True)
    weight = weights["weight"].to_numpy(dtype=np.float32, copy=True)
    stock_pos = subset["global_stock_pos"].to_numpy(dtype=np.int64, copy=True)
    date_pos = subset["global_date_pos"].to_numpy(dtype=np.int64, copy=True)
    out = np.full(len(subset), np.nan, dtype=np.float32)
    for slc in _chunk_slices(len(subset), chunk_size):
        sp = stock_pos[slc]
        dp = date_pos[slc]
        valid = (
            (sp >= 0)
            & (dp >= 0)
            & (sp < feature_panel.shape[0])
            & (dp < feature_panel.shape[1])
        )
        if not np.any(valid):
            continue
        values = np.asarray(feature_panel[sp[valid], dp[valid], :][:, feature_idx], dtype=np.float32)
        score = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) @ weight
        local = np.arange(slc.start, slc.stop, dtype=np.int64)
        out[local[valid]] = score.astype(np.float32, copy=False)
    return out


def _daily_topk(subset: pd.DataFrame, *, score: np.ndarray, top_k: int) -> pd.DataFrame:
    if subset.empty:
        return subset.copy()
    work = subset[["_sample_pos", "date", "year", "role"]].copy()
    work["score"] = score
    work = work[np.isfinite(work["score"].to_numpy(dtype=np.float32, copy=False))].copy()
    if work.empty:
        return work
    work = work.sort_values(["date", "score"], ascending=[True, False], kind="mergesort")
    work["rank_in_day"] = work.groupby("date", sort=False).cumcount() + 1
    return work[work["rank_in_day"] <= int(top_k)].copy()


def _outcome_arrays(labels: Mapping[str, np.memmap], positions: np.ndarray, *, cost_bps: float, horizon: int) -> np.ndarray:
    path = labels["cumulative_return_1to20"]
    idx = int(horizon) - 1
    return np.asarray(path[positions, idx], dtype=np.float32) - float(cost_bps) / 10000.0


def _dynamic_success(labels: Mapping[str, np.memmap], positions: np.ndarray, *, profit: float, stop: float) -> np.ndarray:
    path = np.asarray(labels["cumulative_return_1to20"][positions], dtype=np.float32)
    valid_path = np.isfinite(path).all(axis=1)
    profit_day = _first_hit_day(path, profit, direction="up")
    stop_day = _first_hit_day(path, stop, direction="down")
    return valid_path & (profit_day > 0) & ((stop_day == 0) | (profit_day <= stop_day))


def _evaluate_selection(
    *,
    labels: Mapping[str, np.memmap],
    subset: pd.DataFrame,
    selected: pd.DataFrame,
    strategy: str,
    top_k: int,
    horizon: int,
    cost_bps: float,
    fold_kind: str,
    fold_year: int,
) -> dict[str, Any]:
    if subset.empty or selected.empty:
        return {
            "fold_kind": fold_kind,
            "fold_year": int(fold_year),
            "strategy": strategy,
            "top_k": int(top_k),
            "horizon": int(horizon),
            "selected_count": 0,
        }
    all_pos = subset["_sample_pos"].to_numpy(dtype=np.int64, copy=True)
    sel_pos = selected["_sample_pos"].to_numpy(dtype=np.int64, copy=True)
    all_ret = _outcome_arrays(labels, all_pos, cost_bps=cost_bps, horizon=horizon)
    sel_ret = _outcome_arrays(labels, sel_pos, cost_bps=cost_bps, horizon=horizon)
    base = pd.DataFrame({"date": subset["date"].to_numpy(), "base_ret": all_ret})
    sel = pd.DataFrame({"date": selected["date"].to_numpy(), "selected_ret": sel_ret})
    base_daily = base.groupby("date", sort=True)["base_ret"].mean()
    selected_daily = sel.groupby("date", sort=True)["selected_ret"].mean()
    aligned = pd.concat([selected_daily, base_daily], axis=1, join="inner").dropna()
    if aligned.empty:
        mean_ret = mean_base = mean_excess = std_ret = float("nan")
        win_day_rate = excess_day_positive_rate = float("nan")
    else:
        excess = aligned["selected_ret"] - aligned["base_ret"]
        mean_ret = float(aligned["selected_ret"].mean())
        mean_base = float(aligned["base_ret"].mean())
        mean_excess = float(excess.mean())
        std_ret = float(aligned["selected_ret"].std(ddof=1))
        win_day_rate = float((aligned["selected_ret"] > 0).mean())
        excess_day_positive_rate = float((excess > 0).mean())
    success_5_3 = _dynamic_success(labels, sel_pos, profit=0.05, stop=0.03)
    success_10_5 = _dynamic_success(labels, sel_pos, profit=0.10, stop=0.05)
    base_success_5_3 = _dynamic_success(labels, all_pos, profit=0.05, stop=0.03)
    base_success_10_5 = _dynamic_success(labels, all_pos, profit=0.10, stop=0.05)
    return {
        "fold_kind": fold_kind,
        "fold_year": int(fold_year),
        "strategy": strategy,
        "top_k": int(top_k),
        "horizon": int(horizon),
        "date_count": int(aligned.shape[0]),
        "selected_count": int(len(selected)),
        "mean_net_return": mean_ret,
        "mean_base_return": mean_base,
        "mean_excess_return": mean_excess,
        "mean_excess_bps": mean_excess * 10000.0 if math.isfinite(mean_excess) else float("nan"),
        "std_daily_return": std_ret,
        "cohort_ir": float(mean_ret / std_ret) if math.isfinite(mean_ret) and math.isfinite(std_ret) and std_ret > 0 else float("nan"),
        "win_day_rate": win_day_rate,
        "excess_day_positive_rate": excess_day_positive_rate,
        "selected_success_up5_down3": float(success_5_3.mean()) if len(success_5_3) else float("nan"),
        "base_success_up5_down3": float(base_success_5_3.mean()) if len(base_success_5_3) else float("nan"),
        "selected_success_up10_down5": float(success_10_5.mean()) if len(success_10_5) else float("nan"),
        "base_success_up10_down5": float(base_success_10_5.mean()) if len(base_success_10_5) else float("nan"),
    }


def _daily_return_frame(
    *,
    labels: Mapping[str, np.memmap],
    subset: pd.DataFrame,
    selected: pd.DataFrame,
    horizon: int,
    cost_bps: float,
) -> pd.DataFrame:
    if subset.empty or selected.empty:
        return pd.DataFrame()
    all_pos = subset["_sample_pos"].to_numpy(dtype=np.int64, copy=True)
    sel_pos = selected["_sample_pos"].to_numpy(dtype=np.int64, copy=True)
    base_ret = _outcome_arrays(labels, all_pos, cost_bps=cost_bps, horizon=horizon)
    sel_ret = _outcome_arrays(labels, sel_pos, cost_bps=cost_bps, horizon=horizon)
    base = pd.DataFrame({"date": subset["date"].to_numpy(), "base_ret": base_ret}).groupby("date", sort=True)["base_ret"].mean()
    sel = pd.DataFrame({"date": selected["date"].to_numpy(), "selected_ret": sel_ret}).groupby("date", sort=True)["selected_ret"].mean()
    out = pd.concat([sel, base], axis=1, join="inner").reset_index()
    out["excess_ret"] = out["selected_ret"] - out["base_ret"]
    out["cum_selected_ret_index"] = (1.0 + out["selected_ret"].fillna(0.0)).cumprod()
    out["cum_excess_ret_index"] = (1.0 + out["excess_ret"].fillna(0.0)).cumprod()
    return out


def _aggregate_config_selection(fold_results: pd.DataFrame) -> pd.DataFrame:
    dev = fold_results[fold_results["fold_kind"] == "rolling_validation"].copy()
    if dev.empty:
        return pd.DataFrame()
    grouped = dev.groupby(["strategy", "top_k", "horizon"], sort=True)
    out = grouped.agg(
        fold_count=("fold_year", "nunique"),
        mean_excess_bps=("mean_excess_bps", "mean"),
        median_excess_bps=("mean_excess_bps", "median"),
        std_excess_bps=("mean_excess_bps", "std"),
        positive_year_rate=("mean_excess_bps", lambda s: float((s > 0).mean())),
        mean_net_return=("mean_net_return", "mean"),
        mean_base_return=("mean_base_return", "mean"),
        mean_success_up10_down5=("selected_success_up10_down5", "mean"),
        mean_base_success_up10_down5=("base_success_up10_down5", "mean"),
        mean_success_up5_down3=("selected_success_up5_down3", "mean"),
        mean_base_success_up5_down3=("base_success_up5_down3", "mean"),
    ).reset_index()
    out["success_up10_down5_lift"] = out["mean_success_up10_down5"] - out["mean_base_success_up10_down5"]
    out["success_up5_down3_lift"] = out["mean_success_up5_down3"] - out["mean_base_success_up5_down3"]
    out["development_score"] = (
        out["mean_excess_bps"].fillna(-999.0)
        + 15.0 * (out["positive_year_rate"].fillna(0.0) - 0.5)
        - 0.15 * out["std_excess_bps"].fillna(0.0)
        + 100.0 * out["success_up10_down5_lift"].fillna(0.0)
    )
    return out.sort_values("development_score", ascending=False, kind="mergesort")


def _plot_outputs(*, output_dir: Path, fold_results: pd.DataFrame, daily_returns: pd.DataFrame, winner: Mapping[str, Any]) -> list[str]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    if not fold_results.empty and winner:
        mask = (
            (fold_results["strategy"] == winner["strategy"])
            & (fold_results["top_k"] == int(winner["top_k"]))
            & (fold_results["horizon"] == int(winner["horizon"]))
            & (fold_results["fold_kind"] == "rolling_validation")
        )
        frame = fold_results[mask].sort_values("fold_year")
        if not frame.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(frame["fold_year"].astype(str), frame["mean_excess_bps"], color="#2f6f9f")
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.set_title("Rolling validation excess return by year")
            ax.set_xlabel("Validation year")
            ax.set_ylabel("Mean cohort excess return (bps)")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            path = chart_dir / "winner_rolling_excess_bps.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            outputs.append(str(path.resolve()))
    if not daily_returns.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(daily_returns["date"], daily_returns["cum_selected_ret_index"], label="selected")
        ax.plot(daily_returns["date"], daily_returns["cum_excess_ret_index"], label="excess_vs_all")
        ax.set_title("Frozen 2025 daily cohort return index")
        ax.set_xlabel("Date")
        ax.set_ylabel("Index")
        ax.grid(True, alpha=0.25)
        ax.legend()
        step = max(1, len(daily_returns) // 8)
        ax.set_xticks(daily_returns["date"].iloc[::step])
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        path = chart_dir / "winner_2025_daily_return_index.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs


def _build_markdown_report(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    config_selection: pd.DataFrame,
    fold_results: pd.DataFrame,
    test_results: pd.DataFrame,
    winner_weights: pd.DataFrame,
    chart_paths: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# QDP v2 画像规则发现与冻结验收")
    lines.append("")
    lines.append("## 方法")
    lines.append("")
    lines.append("开发集为 2012-2024，内部用 2018-2024 做滚动年份验证；2025 只用于冻结规则验收。规则来自上涨画像特征的稳定效应，不使用 symbol。")
    lines.append("")
    lines.append("## 最优开发配置")
    lines.append("")
    if not config_selection.empty:
        row = config_selection.iloc[0].to_dict()
        lines.append(
            f"- strategy={row['strategy']}, top_k={int(row['top_k'])}, horizon={int(row['horizon'])}, "
            f"dev_mean_excess={row['mean_excess_bps']:.2f} bps, "
            f"positive_year_rate={row['positive_year_rate']:.2%}, "
            f"up10/down5_lift={row['success_up10_down5_lift']:.2%}"
        )
    lines.append("")
    lines.append("## 2025 冻结验收")
    lines.append("")
    if not test_results.empty:
        top = test_results.sort_values("development_rank").head(10)
        for row in top.to_dict("records"):
            lines.append(
                "- "
                f"rank={int(row['development_rank'])} {row['strategy']} top{int(row['top_k'])} H{int(row['horizon'])}: "
                f"net={row['mean_net_return'] * 100:.2f}%, "
                f"base={row['mean_base_return'] * 100:.2f}%, "
                f"excess={row['mean_excess_bps']:.2f} bps, "
                f"win_day={row['win_day_rate']:.2%}, "
                f"up10/down5={row['selected_success_up10_down5']:.2%} vs base {row['base_success_up10_down5']:.2%}"
            )
    lines.append("")
    lines.append("## 最优规则权重")
    lines.append("")
    for row in winner_weights.head(20).to_dict("records"):
        lines.append(f"- {row['feature']}: weight={row['weight']:.4f}, components={row.get('components', '')}")
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    for path in chart_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## 边界")
    lines.append("")
    lines.append("- 这是画像规则的第一版可交易性证明，仍是入场 cohort 评估，不是完整持仓组合撮合。")
    lines.append("- 涨停买入限制、停牌和无开盘已经用 training pack 的 entry labels 过滤；更细滑点、仓位重叠、资金容量需要下一步组合回测。")
    lines.append("- 2025 验收如果被用来继续改规则，就必须把后续新数据作为新的最终验收。")
    path = output_dir / "rule_discovery_backtest_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


def run_rule_discovery_backtest(
    *,
    manifest_json: str | Path = DEFAULT_TRAINING_PACK_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_tag: str = "qdp_v2_rule_discovery_backtest",
    development_years: tuple[int, ...] = DEFAULT_DEVELOPMENT_YEARS,
    fold_years: tuple[int, ...] = DEFAULT_FOLD_YEARS,
    test_year: int = DEFAULT_TEST_YEAR,
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    cost_bps: float = DEFAULT_COST_BPS,
    top_features: int = 25,
    min_same_sign_share: float = 0.60,
    chunk_size: int = 150_000,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pack_root = manifest_path.parent
    output_dir = Path(output_root) / f"{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "starting", "started_at": _now()})

    feature_columns = [str(item) for item in list(manifest.get("feature_columns", []) or [])]
    feature_shape = tuple(int(item) for item in list(manifest.get("feature_panel_shape", []) or []))
    feature_dtype = str(manifest.get("feature_dtype", "float16") or "float16")
    feature_path = _resolve_path(str(manifest.get("feature_panel_path", "") or ""), root=pack_root)
    feature_panel = np.memmap(feature_path, dtype=feature_dtype, mode="r", shape=feature_shape)

    label_meta = dict(manifest.get("label_arrays", {}) or {})
    labels = {
        name: _open_memmap(label_meta[name])
        for name in [
            "rank_by_horizon",
            "cumulative_return_1to20",
            "entry_tradeable",
            "entry_limit_up_buy_blocked",
            "entry_suspended_or_no_open",
        ]
    }

    sample_index_path = _resolve_path(str(manifest.get("sample_index_path", "") or ""), root=pack_root)
    sample_index = pd.read_parquet(
        sample_index_path,
        columns=["date", "role", "global_date_pos", "global_stock_pos", "_sample_pos"],
    )
    sample_index["date"] = sample_index["date"].astype(str)
    sample_index["year"] = sample_index["date"].str.slice(0, 4).astype("int16")
    for col in ["global_date_pos", "global_stock_pos", "_sample_pos"]:
        sample_index[col] = pd.to_numeric(sample_index[col], errors="coerce").fillna(-1).astype("int64")

    effect_store = _build_effect_store(
        feature_panel=feature_panel,
        labels=labels,
        sample_index=sample_index,
        feature_columns=feature_columns,
        years=tuple(sorted(set(development_years))),
        chunk_size=int(chunk_size),
        progress_path=progress_path,
    )

    all_weights: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_year in fold_years:
        train_years = tuple(year for year in development_years if int(year) < int(fold_year))
        if len(train_years) < 3:
            continue
        subset = sample_index[sample_index["year"].eq(int(fold_year))].copy()
        tradable = _tradable_mask(labels, subset["_sample_pos"].to_numpy(dtype=np.int64, copy=True))
        subset = subset[tradable].copy()
        for strategy in STRATEGY_COMPONENTS:
            weights = _strategy_weights(
                store=effect_store,
                strategy_name=strategy,
                train_years=train_years,
                top_features=int(top_features),
                min_same_sign_share=float(min_same_sign_share),
            )
            if weights.empty:
                continue
            tmp_weights = weights.copy()
            tmp_weights["fold_kind"] = "rolling_validation"
            tmp_weights["fold_year"] = int(fold_year)
            tmp_weights["train_years"] = f"{min(train_years)}-{max(train_years)}"
            all_weights.append(tmp_weights)
            score = _score_subset(feature_panel=feature_panel, subset=subset, weights=weights, chunk_size=int(chunk_size))
            for top_k in top_k_values:
                selected = _daily_topk(subset, score=score, top_k=int(top_k))
                for horizon in horizons:
                    fold_rows.append(
                        _evaluate_selection(
                            labels=labels,
                            subset=subset,
                            selected=selected,
                            strategy=strategy,
                            top_k=int(top_k),
                            horizon=int(horizon),
                            cost_bps=float(cost_bps),
                            fold_kind="rolling_validation",
                            fold_year=int(fold_year),
                        )
                    )
        _write_json(progress_path, {"status": "rolling_validation", "completed_fold_year": int(fold_year), "updated_at": _now()})

    fold_results = pd.DataFrame(fold_rows)
    config_selection = _aggregate_config_selection(fold_results)
    if config_selection.empty:
        raise RuntimeError("no valid rolling validation config was produced")
    config_selection["development_rank"] = np.arange(1, len(config_selection) + 1, dtype=np.int32)

    test_subset = sample_index[sample_index["year"].eq(int(test_year))].copy()
    test_tradable = _tradable_mask(labels, test_subset["_sample_pos"].to_numpy(dtype=np.int64, copy=True))
    test_subset = test_subset[test_tradable].copy()
    final_train_years = tuple(int(year) for year in development_years)
    test_rows: list[dict[str, Any]] = []
    daily_output = pd.DataFrame()
    winner_weights = pd.DataFrame()
    final_weights_by_strategy: dict[str, pd.DataFrame] = {}
    for strategy in STRATEGY_COMPONENTS:
        weights = _strategy_weights(
            store=effect_store,
            strategy_name=strategy,
            train_years=final_train_years,
            top_features=int(top_features),
            min_same_sign_share=float(min_same_sign_share),
        )
        final_weights_by_strategy[strategy] = weights
        if not weights.empty:
            tmp = weights.copy()
            tmp["fold_kind"] = "final_train"
            tmp["fold_year"] = int(test_year)
            tmp["train_years"] = f"{min(final_train_years)}-{max(final_train_years)}"
            all_weights.append(tmp)

    # Evaluate all development-ranked configs on the frozen test year.
    score_cache: dict[str, np.ndarray] = {}
    for row in config_selection.to_dict("records"):
        strategy = str(row["strategy"])
        weights = final_weights_by_strategy.get(strategy, pd.DataFrame())
        if weights.empty:
            continue
        if strategy not in score_cache:
            score_cache[strategy] = _score_subset(feature_panel=feature_panel, subset=test_subset, weights=weights, chunk_size=int(chunk_size))
        selected = _daily_topk(test_subset, score=score_cache[strategy], top_k=int(row["top_k"]))
        metrics = _evaluate_selection(
            labels=labels,
            subset=test_subset,
            selected=selected,
            strategy=strategy,
            top_k=int(row["top_k"]),
            horizon=int(row["horizon"]),
            cost_bps=float(cost_bps),
            fold_kind="frozen_test",
            fold_year=int(test_year),
        )
        metrics["development_rank"] = int(row["development_rank"])
        metrics["development_score"] = float(row["development_score"])
        test_rows.append(metrics)
    test_results = pd.DataFrame(test_rows).sort_values("development_rank", kind="mergesort")

    winner = config_selection.iloc[0].to_dict()
    winner_weights = final_weights_by_strategy.get(str(winner["strategy"]), pd.DataFrame()).copy()
    if str(winner["strategy"]) in score_cache:
        winner_score = score_cache[str(winner["strategy"])]
    else:
        winner_score = _score_subset(
            feature_panel=feature_panel,
            subset=test_subset,
            weights=winner_weights,
            chunk_size=int(chunk_size),
        )
    winner_selected = _daily_topk(test_subset, score=winner_score, top_k=int(winner["top_k"]))
    daily_output = _daily_return_frame(
        labels=labels,
        subset=test_subset,
        selected=winner_selected,
        horizon=int(winner["horizon"]),
        cost_bps=float(cost_bps),
    )

    weights_frame = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    outputs = {
        "fold_results_csv": _write_csv(output_dir / "rolling_fold_results.csv", fold_results),
        "config_selection_csv": _write_csv(output_dir / "development_config_selection.csv", config_selection),
        "frozen_test_results_csv": _write_csv(output_dir / "frozen_2025_test_results.csv", test_results),
        "rule_weights_csv": _write_csv(output_dir / "rule_weights_by_fold.csv", weights_frame),
        "winner_weights_csv": _write_csv(output_dir / "winner_final_rule_weights.csv", winner_weights),
        "winner_2025_daily_returns_csv": _write_csv(output_dir / "winner_2025_daily_returns.csv", daily_output),
    }
    chart_paths = _plot_outputs(output_dir=output_dir, fold_results=fold_results, daily_returns=daily_output, winner=winner)
    outputs["charts"] = chart_paths
    summary: dict[str, Any] = {
        "artifact_type": "qdp_v2_rule_discovery_backtest",
        "generated_at": _now(),
        "manifest_json": str(manifest_path),
        "output_dir": str(output_dir.resolve()),
        "sample_count": int(len(sample_index)),
        "development_years": list(development_years),
        "rolling_fold_years": list(fold_years),
        "test_year": int(test_year),
        "cost_bps": float(cost_bps),
        "top_k_values": list(top_k_values),
        "horizons": list(horizons),
        "top_features_per_component": int(top_features),
        "min_same_sign_share": float(min_same_sign_share),
        "winner": winner,
        "winner_test_result": test_results[test_results["development_rank"].eq(1)].iloc[0].to_dict() if not test_results.empty else {},
        "outputs": outputs,
    }
    report_path = _build_markdown_report(
        output_dir=output_dir,
        summary=summary,
        config_selection=config_selection,
        fold_results=fold_results,
        test_results=test_results,
        winner_weights=winner_weights,
        chart_paths=chart_paths,
    )
    summary["outputs"]["report_md"] = report_path
    summary_path = _write_json(output_dir / "rule_discovery_backtest_summary.json", summary)
    summary["outputs"]["summary_json"] = summary_path
    _write_json(progress_path, {"status": "completed", "summary_json": summary_path, "report_md": report_path, "updated_at": _now()})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover simple rising-stock profile rules on 2012-2024 and validate frozen configs on 2025.")
    parser.add_argument("--manifest-json", default=str(DEFAULT_TRAINING_PACK_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-tag", default="qdp_v2_rule_discovery_backtest")
    parser.add_argument("--development-years", default="2012-2024")
    parser.add_argument("--fold-years", default="2018-2024")
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    parser.add_argument("--top-k", default="20,50,100")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--top-features", type=int, default=25)
    parser.add_argument("--min-same-sign-share", type=float, default=0.60)
    parser.add_argument("--chunk-size", type=int, default=150_000)
    parser.add_argument("--json", action="store_true")
    return parser


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


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_rule_discovery_backtest(
        manifest_json=args.manifest_json,
        output_root=args.output_root,
        run_tag=args.run_tag,
        development_years=_parse_year_range(args.development_years, default=DEFAULT_DEVELOPMENT_YEARS),
        fold_years=_parse_year_range(args.fold_years, default=DEFAULT_FOLD_YEARS),
        test_year=int(args.test_year),
        top_k_values=_parse_csv_ints(args.top_k, default=DEFAULT_TOP_K),
        horizons=_parse_csv_ints(args.horizons, default=DEFAULT_HORIZONS),
        cost_bps=float(args.cost_bps),
        top_features=int(args.top_features),
        min_same_sign_share=float(args.min_same_sign_share),
        chunk_size=int(args.chunk_size),
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(summary.get("outputs", {}).get("report_md", ""))
    return summary


if __name__ == "__main__":
    main()
