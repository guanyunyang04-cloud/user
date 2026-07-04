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
    "quant_data_platform/data/qdp_v2/research/training_pack/"
    "qdp_v2_alpha_v2_full_contract_2012_2025_20260702_01_training_pack/"
    "qdp_training_pack_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/rising_stock_profile")
DEFAULT_ROLES = ("train", "validation", "test")
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_TOP_QUANTILES = (0.90, 0.95)
DEFAULT_PROFIT_STOP_PAIRS = ((0.05, 0.03), (0.10, 0.05))


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


def _parse_csv_strings(raw: str | Iterable[str] | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out or default)


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
        seen.add(item)
        out.append(item)
    return tuple(out or default)


def _parse_csv_floats(raw: str | Iterable[float] | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = [float(item) for item in raw]
    out: list[float] = []
    seen: set[float] = set()
    for value in values:
        item = float(value)
        if not math.isfinite(item) or item <= 0 or item >= 1 or item in seen:
            continue
        seen.add(item)
        out.append(item)
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
    if not shape:
        raise ValueError(f"memmap shape missing for {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _feature_family(feature: str) -> str:
    name = str(feature)
    if name.startswith("intraday_") or "intraday" in name:
        return "intraday"
    if name.startswith("raw_"):
        return "daily_bar"
    if name.startswith("ret_") or name.startswith("vol_") or name.startswith("turn"):
        return "daily_price_volume"
    if name.startswith("cs_rank_") or name.startswith("cs_z_"):
        return "cross_section"
    if name.startswith("market_") or name.startswith("benchmark_"):
        return "market_regime"
    if name.startswith("industry_") or "industry" in name:
        return "industry"
    if name.startswith("peer_"):
        return "peer"
    if name.startswith("valuation_"):
        return "valuation"
    if name.startswith("adjust_"):
        return "adjustment"
    if name.startswith("index_"):
        return "index_membership"
    if name.startswith("local_"):
        return "local_path"
    return "other"


@dataclass(frozen=True)
class LabelSpec:
    name: str
    family: str
    description: str


@dataclass
class ScopeFeatureAccumulator:
    feature_count: int
    scope_sample_count: int = 0
    feature_count_all: np.ndarray | None = None
    feature_sum_all: np.ndarray | None = None
    label_sample_count: dict[str, int] | None = None
    label_feature_count: dict[str, np.ndarray] | None = None
    label_feature_sum: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.feature_count_all = np.zeros(self.feature_count, dtype=np.int64)
        self.feature_sum_all = np.zeros(self.feature_count, dtype=np.float64)
        self.label_sample_count = {}
        self.label_feature_count = {}
        self.label_feature_sum = {}

    def update_all(self, features: np.ndarray, finite: np.ndarray) -> None:
        self.scope_sample_count += int(features.shape[0])
        assert self.feature_count_all is not None
        assert self.feature_sum_all is not None
        self.feature_count_all += finite.sum(axis=0, dtype=np.int64)
        self.feature_sum_all += np.where(finite, features, 0.0).sum(axis=0, dtype=np.float64)

    def update_label(self, label: str, features: np.ndarray, finite: np.ndarray, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        count = int(mask.sum())
        if count <= 0:
            return
        assert self.label_sample_count is not None
        assert self.label_feature_count is not None
        assert self.label_feature_sum is not None
        self.label_sample_count[label] = int(self.label_sample_count.get(label, 0) + count)
        if label not in self.label_feature_count:
            self.label_feature_count[label] = np.zeros(self.feature_count, dtype=np.int64)
            self.label_feature_sum[label] = np.zeros(self.feature_count, dtype=np.float64)
        selected_finite = finite[mask]
        selected_features = features[mask]
        self.label_feature_count[label] += selected_finite.sum(axis=0, dtype=np.int64)
        self.label_feature_sum[label] += np.where(selected_finite, selected_features, 0.0).sum(axis=0, dtype=np.float64)


@dataclass
class PathAccumulator:
    day_count: int = 20
    count: int = 0
    path_sum: np.ndarray | None = None
    peak_day_hist: np.ndarray | None = None
    max_upside_sum: float = 0.0
    final_return_sum: float = 0.0
    pre_peak_drawdown_sum: float = 0.0
    post_peak_fade_sum: float = 0.0

    def __post_init__(self) -> None:
        self.path_sum = np.zeros(self.day_count, dtype=np.float64)
        self.peak_day_hist = np.zeros(self.day_count, dtype=np.int64)

    def update(self, cumulative_path: np.ndarray) -> None:
        if cumulative_path.size == 0:
            return
        path = np.asarray(cumulative_path, dtype=np.float32)
        finite_rows = np.isfinite(path).all(axis=1)
        if not np.any(finite_rows):
            return
        path = path[finite_rows]
        self.count += int(path.shape[0])
        assert self.path_sum is not None
        assert self.peak_day_hist is not None
        self.path_sum += path.sum(axis=0, dtype=np.float64)
        peak_idx = np.argmax(path, axis=1)
        max_upside = path[np.arange(path.shape[0]), peak_idx]
        final_ret = path[:, -1]
        self.max_upside_sum += float(max_upside.sum(dtype=np.float64))
        self.final_return_sum += float(final_ret.sum(dtype=np.float64))
        for item in peak_idx:
            self.peak_day_hist[int(item)] += 1
        pre_peak_drawdown = np.zeros(path.shape[0], dtype=np.float32)
        for row_idx, top_idx in enumerate(peak_idx):
            pre_peak_drawdown[row_idx] = float(np.nanmin(path[row_idx, : int(top_idx) + 1]))
        self.pre_peak_drawdown_sum += float(pre_peak_drawdown.sum(dtype=np.float64))
        self.post_peak_fade_sum += float((max_upside - final_ret).sum(dtype=np.float64))


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


def _label_masks(
    *,
    labels: Mapping[str, np.memmap],
    positions: np.ndarray,
    cumulative_horizons: tuple[int, ...],
    top_quantiles: tuple[float, ...],
    profit_stop_pairs: tuple[tuple[float, float], ...],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    masks: dict[str, np.ndarray] = {}
    event_extra: dict[str, np.ndarray] = {}
    ranks = np.asarray(labels["rank_by_horizon"][positions], dtype=np.float32)
    cumulative_path = np.asarray(labels["cumulative_return_1to20"][positions], dtype=np.float32)
    for horizon in cumulative_horizons:
        idx = cumulative_horizons.index(int(horizon))
        rank = ranks[:, idx]
        valid_rank = np.isfinite(rank)
        for quantile in top_quantiles:
            top_pct = int(round((1.0 - float(quantile)) * 100.0))
            name = f"top{top_pct}_return_{horizon}d"
            masks[name] = valid_rank & (rank >= float(quantile))
    max_upside = np.nanmax(cumulative_path, axis=1)
    final_return = cumulative_path[:, -1]
    peak_day = np.argmax(np.nan_to_num(cumulative_path, nan=-999.0), axis=1).astype(np.int16) + 1
    valid_path = np.isfinite(cumulative_path).all(axis=1)
    masks["large_upside_5pct_20d"] = valid_path & (max_upside >= 0.05)
    masks["large_upside_10pct_20d"] = valid_path & (max_upside >= 0.10)
    masks["persistent_upside_10pct_20d"] = valid_path & (max_upside >= 0.10) & (final_return >= (max_upside - 0.02))
    masks["rise_then_fade_10pct_20d"] = valid_path & (max_upside >= 0.10) & (final_return <= (max_upside - 0.03))
    masks["fast_peak_10pct_20d"] = valid_path & (max_upside >= 0.10) & (peak_day <= 5)
    masks["late_peak_10pct_20d"] = valid_path & (max_upside >= 0.10) & (peak_day >= 15)
    event_extra["peak_day"] = peak_day
    event_extra["max_upside"] = max_upside
    event_extra["final_return"] = final_return
    for profit, stop in profit_stop_pairs:
        profit_day = _first_hit_day(cumulative_path, float(profit), direction="up")
        stop_day = _first_hit_day(cumulative_path, float(stop), direction="down")
        success = valid_path & (profit_day > 0) & ((stop_day == 0) | (profit_day <= stop_day))
        name = f"hit_up{int(round(profit * 100))}_before_down{int(round(stop * 100))}_20d"
        masks[name] = success
        event_extra[f"{name}_profit_day"] = profit_day
        event_extra[f"{name}_stop_day"] = stop_day
    return masks, cumulative_path, event_extra


def _label_specs(cumulative_horizons: tuple[int, ...], top_quantiles: tuple[float, ...]) -> list[LabelSpec]:
    specs: list[LabelSpec] = []
    for horizon in cumulative_horizons:
        for quantile in top_quantiles:
            top_pct = int(round((1.0 - float(quantile)) * 100.0))
            specs.append(
                LabelSpec(
                    name=f"top{top_pct}_return_{horizon}d",
                    family="fixed_horizon_rank",
                    description=f"future {horizon} trading day cumulative return is in same-day top {top_pct} percent",
                )
            )
    specs.extend(
        [
            LabelSpec("large_upside_5pct_20d", "dynamic_path", "future 20 day max cumulative return reaches +5%"),
            LabelSpec("large_upside_10pct_20d", "dynamic_path", "future 20 day max cumulative return reaches +10%"),
            LabelSpec("persistent_upside_10pct_20d", "dynamic_path", "+10% runup and keeps most gains through day 20"),
            LabelSpec("rise_then_fade_10pct_20d", "dynamic_path", "+10% runup followed by at least 3% fade by day 20"),
            LabelSpec("fast_peak_10pct_20d", "dynamic_path", "+10% runup with peak in first 5 days"),
            LabelSpec("late_peak_10pct_20d", "dynamic_path", "+10% runup with peak on day 15 or later"),
            LabelSpec("hit_up5_before_down3_20d", "trade_path", "hits +5% before -3% in the next 20 days"),
            LabelSpec("hit_up10_before_down5_20d", "trade_path", "hits +10% before -5% in the next 20 days"),
        ]
    )
    return specs


def _make_feature_stats_frame(
    *,
    accumulators: Mapping[str, ScopeFeatureAccumulator],
    feature_columns: list[str],
    label_specs: list[LabelSpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_families = [_feature_family(item) for item in feature_columns]
    label_spec_by_name = {item.name: item for item in label_specs}
    for scope, acc in accumulators.items():
        assert acc.feature_count_all is not None
        assert acc.feature_sum_all is not None
        assert acc.label_sample_count is not None
        assert acc.label_feature_count is not None
        assert acc.label_feature_sum is not None
        all_mean = np.divide(
            acc.feature_sum_all,
            np.maximum(acc.feature_count_all, 1),
            out=np.full_like(acc.feature_sum_all, np.nan, dtype=np.float64),
            where=acc.feature_count_all > 0,
        )
        for label in sorted(acc.label_sample_count):
            label_count = int(acc.label_sample_count.get(label, 0))
            label_base_rate = float(label_count / acc.scope_sample_count) if acc.scope_sample_count else float("nan")
            lf_count = acc.label_feature_count.get(label)
            lf_sum = acc.label_feature_sum.get(label)
            if lf_count is None or lf_sum is None:
                continue
            label_mean = np.divide(
                lf_sum,
                np.maximum(lf_count, 1),
                out=np.full_like(lf_sum, np.nan, dtype=np.float64),
                where=lf_count > 0,
            )
            effect = label_mean - all_mean
            spec = label_spec_by_name.get(label)
            for idx, feature in enumerate(feature_columns):
                if int(lf_count[idx]) <= 0 or int(acc.feature_count_all[idx]) <= 0:
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "label": label,
                        "label_family": spec.family if spec else "",
                        "label_description": spec.description if spec else "",
                        "feature": feature,
                        "feature_index": int(idx),
                        "feature_family": feature_families[idx],
                        "scope_sample_count": int(acc.scope_sample_count),
                        "label_sample_count": int(label_count),
                        "label_base_rate": label_base_rate,
                        "feature_count_all": int(acc.feature_count_all[idx]),
                        "feature_count_label": int(lf_count[idx]),
                        "feature_mean_all": float(all_mean[idx]),
                        "feature_mean_label": float(label_mean[idx]),
                        "effect_label_minus_all": float(effect[idx]),
                        "abs_effect": float(abs(effect[idx])) if math.isfinite(float(effect[idx])) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _make_stability_frame(feature_stats: pd.DataFrame) -> pd.DataFrame:
    role_stats = feature_stats[feature_stats["scope"].isin(["role=train", "role=validation", "role=test"])].copy()
    if role_stats.empty:
        return pd.DataFrame()
    pivot = role_stats.pivot_table(
        index=["label", "feature", "feature_family"],
        columns="scope",
        values="effect_label_minus_all",
        aggfunc="mean",
    ).reset_index()
    for col in ["role=train", "role=validation", "role=test"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    effects = pivot[["role=train", "role=validation", "role=test"]].to_numpy(dtype=np.float64)
    signs = np.sign(effects)
    finite = np.isfinite(effects)
    same_positive = finite.all(axis=1) & (signs > 0).all(axis=1)
    same_negative = finite.all(axis=1) & (signs < 0).all(axis=1)
    pivot["stable_direction"] = np.where(same_positive, "positive", np.where(same_negative, "negative", "mixed"))
    pivot["min_abs_effect"] = np.nanmin(np.abs(effects), axis=1)
    pivot["mean_abs_effect"] = np.nanmean(np.abs(effects), axis=1)
    pivot["validation_test_same_sign"] = np.sign(pivot["role=validation"]) == np.sign(pivot["role=test"])
    pivot = pivot.sort_values(["stable_direction", "min_abs_effect"], ascending=[True, False], kind="mergesort")
    return pivot


def _make_path_frames(
    *,
    path_accumulators: Mapping[str, PathAccumulator],
    label_specs: list[LabelSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_spec_by_name = {item.name: item for item in label_specs}
    path_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for key, acc in path_accumulators.items():
        scope, label = key.split("|", 1)
        assert acc.path_sum is not None
        assert acc.peak_day_hist is not None
        mean_path = acc.path_sum / max(int(acc.count), 1)
        spec = label_spec_by_name.get(label)
        for day_idx, value in enumerate(mean_path, start=1):
            path_rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "label_family": spec.family if spec else "",
                    "day": int(day_idx),
                    "mean_cumulative_return": float(value),
                    "sample_count": int(acc.count),
                }
            )
        if int(acc.count) > 0:
            peak_mode = int(np.argmax(acc.peak_day_hist) + 1)
            peak_hist_total = int(acc.peak_day_hist.sum())
            summary_rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "label_family": spec.family if spec else "",
                    "sample_count": int(acc.count),
                    "mean_final_return_20d": float(acc.final_return_sum / acc.count),
                    "mean_max_upside_20d": float(acc.max_upside_sum / acc.count),
                    "mean_pre_peak_min_return": float(acc.pre_peak_drawdown_sum / acc.count),
                    "mean_post_peak_fade": float(acc.post_peak_fade_sum / acc.count),
                    "peak_day_mode": peak_mode,
                    "peak_day_mode_share": float(acc.peak_day_hist[peak_mode - 1] / peak_hist_total) if peak_hist_total else float("nan"),
                }
            )
    return pd.DataFrame(path_rows), pd.DataFrame(summary_rows)


def _make_exit_frame(exit_counts: Mapping[str, Mapping[str, int]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, metrics in sorted(exit_counts.items()):
        denominator = int(metrics.get("sample_count", 0))
        for metric, value in sorted(metrics.items()):
            if metric == "sample_count":
                continue
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "count": int(value),
                    "sample_count": denominator,
                    "rate": float(value / denominator) if denominator else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _plot_path_profiles(path_frame: pd.DataFrame, output_dir: Path) -> list[str]:
    outputs: list[str] = []
    if path_frame.empty:
        return outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    target_labels = [
        "top10_return_20d",
        "top5_return_20d",
        "hit_up10_before_down5_20d",
        "persistent_upside_10pct_20d",
        "rise_then_fade_10pct_20d",
    ]
    for scope in ["all", "role=validation", "role=test"]:
        frame = path_frame[(path_frame["scope"] == scope) & (path_frame["label"].isin(target_labels))].copy()
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, group in frame.groupby("label", sort=False):
            group = group.sort_values("day")
            ax.plot(group["day"], group["mean_cumulative_return"] * 100.0, label=label)
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_title(f"Future 20D path profile: {scope}")
        ax.set_xlabel("Future trading day")
        ax.set_ylabel("Mean cumulative return (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / f"path_profile_{scope.replace('=', '_')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs


def _plot_feature_bars(stability_frame: pd.DataFrame, output_dir: Path) -> list[str]:
    outputs: list[str] = []
    if stability_frame.empty:
        return outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in ["top10_return_20d", "hit_up10_before_down5_20d", "persistent_upside_10pct_20d"]:
        frame = stability_frame[(stability_frame["label"] == label) & (stability_frame["stable_direction"] == "positive")].copy()
        frame = frame.sort_values("min_abs_effect", ascending=False).head(15)
        if frame.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(frame))
        ax.barh(y, frame["min_abs_effect"], color="#2f6f9f")
        ax.set_yticks(y)
        ax.set_yticklabels(frame["feature"], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"Stable positive entry features: {label}")
        ax.set_xlabel("Minimum absolute effect across train/validation/test")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"stable_features_{label}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs


def _plot_exit_rates(exit_frame: pd.DataFrame, output_dir: Path) -> list[str]:
    outputs: list[str] = []
    if exit_frame.empty:
        return outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = exit_frame[exit_frame["metric"].str.startswith("success_")].copy()
    frame = frame[frame["scope"].isin(["all", "role=validation", "role=test"])]
    if frame.empty:
        return outputs
    pivot = frame.pivot_table(index="metric", columns="scope", values="rate", aggfunc="mean").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=np.float64) * 100.0, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index), fontsize=8)
    ax.set_title("Profit-before-stop success rate (%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iat[i, j] * 100:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    fig.tight_layout()
    path = output_dir / "exit_success_rates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(str(path.resolve()))
    return outputs


def _top_rows(frame: pd.DataFrame, *, label: str, direction: str, limit: int = 12) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame[(frame["label"] == label) & (frame["stable_direction"] == direction)].copy()
    return out.sort_values("min_abs_effect", ascending=False).head(limit)


def _build_markdown_report(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    feature_stats: pd.DataFrame,
    stability_frame: pd.DataFrame,
    path_summary: pd.DataFrame,
    exit_frame: pd.DataFrame,
    chart_paths: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# QDP v2 上涨股票画像研究")
    lines.append("")
    lines.append("## 研究定位")
    lines.append("")
    lines.append("这份报告是上涨股票画像基线，不是模型训练结果，也不是最终交易策略。固定 1/3/5/10/20 日标签用于观察不同持有尺度，动态路径标签用于研究上涨潜力、见顶衰竭和止盈止损先后。")
    lines.append("")
    lines.append("## 输入范围")
    lines.append("")
    lines.append(f"- training pack: `{summary.get('manifest_json', '')}`")
    lines.append(f"- sample_count: {summary.get('sample_count', 0):,}")
    lines.append(f"- feature_count: {summary.get('feature_count', 0):,}")
    lines.append(f"- roles: {', '.join(summary.get('roles', []))}")
    lines.append(f"- generated_at: {summary.get('generated_at', '')}")
    lines.append("")
    lines.append("## 关键路径结果")
    lines.append("")
    if not path_summary.empty:
        focus = path_summary[
            (path_summary["scope"].isin(["all", "role=validation", "role=test"]))
            & (path_summary["label"].isin(["top10_return_20d", "hit_up10_before_down5_20d", "persistent_upside_10pct_20d", "rise_then_fade_10pct_20d"]))
        ].copy()
        focus = focus.sort_values(["scope", "label"]).head(20)
        for row in focus.to_dict("records"):
            lines.append(
                "- "
                f"{row['scope']} / {row['label']}: "
                f"n={int(row['sample_count']):,}, "
                f"final20={row['mean_final_return_20d'] * 100:.2f}%, "
                f"max_upside={row['mean_max_upside_20d'] * 100:.2f}%, "
                f"post_peak_fade={row['mean_post_peak_fade'] * 100:.2f}%, "
                f"peak_day_mode=D+{int(row['peak_day_mode'])}"
            )
    lines.append("")
    lines.append("## 稳定入场特征候选")
    lines.append("")
    for label in ["top10_return_20d", "hit_up10_before_down5_20d", "persistent_upside_10pct_20d"]:
        rows = _top_rows(stability_frame, label=label, direction="positive", limit=10)
        if rows.empty:
            continue
        lines.append(f"### {label}")
        for row in rows.to_dict("records"):
            lines.append(
                "- "
                f"{row['feature']} ({row['feature_family']}): "
                f"train={row.get('role=train', float('nan')):.4f}, "
                f"val={row.get('role=validation', float('nan')):.4f}, "
                f"test={row.get('role=test', float('nan')):.4f}"
            )
        lines.append("")
    lines.append("## 止盈止损路径")
    lines.append("")
    if not exit_frame.empty:
        focus_exit = exit_frame[
            (exit_frame["scope"].isin(["all", "role=validation", "role=test"]))
            & (exit_frame["metric"].str.startswith("success_"))
        ].copy()
        for row in focus_exit.sort_values(["metric", "scope"]).to_dict("records"):
            lines.append(f"- {row['scope']} / {row['metric']}: {row['rate'] * 100:.2f}% ({int(row['count']):,}/{int(row['sample_count']):,})")
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    for path in chart_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## 解释边界")
    lines.append("")
    lines.append("- 这里的 307 特征是加工特征画像，不能证明 raw K 线模型一定不需要。")
    lines.append("- 画像发现的是入场日前可观察特征与未来路径的统计关系，不是事后基本面解释。")
    lines.append("- 固定周期标签是研究标尺；动态止盈止损和见顶衰竭标签更接近交易，但仍需要独立回测验证。")
    lines.append("- 下一步可以把稳定画像转成候选股票池，再训练 gate/rank/exit 三层模型。")
    path = output_dir / "rising_stock_profile_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


def build_rising_stock_profile(
    *,
    manifest_json: str | Path = DEFAULT_TRAINING_PACK_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_tag: str = "qdp_v2_rising_stock_profile",
    roles: str | Iterable[str] = DEFAULT_ROLES,
    horizons: str | Iterable[int] = DEFAULT_HORIZONS,
    top_quantiles: str | Iterable[float] = DEFAULT_TOP_QUANTILES,
    profit_stop_pairs: tuple[tuple[float, float], ...] = DEFAULT_PROFIT_STOP_PAIRS,
    chunk_size: int = 100_000,
    max_rows: int = 0,
    feature_limit: int = 0,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pack_root = manifest_path.parent
    output_dir = Path(output_root) / f"{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "starting", "generated_at": _now()})

    role_values = set(_parse_csv_strings(roles, default=DEFAULT_ROLES))
    requested_horizons = _parse_csv_ints(horizons, default=DEFAULT_HORIZONS)
    available_horizons = tuple(int(item) for item in list(manifest.get("cumulative_horizons", []) or DEFAULT_HORIZONS))
    cumulative_horizons = tuple(item for item in requested_horizons if item in set(available_horizons))
    if not cumulative_horizons:
        raise ValueError(f"no requested horizons found in manifest cumulative_horizons={available_horizons}")
    quantiles = _parse_csv_floats(top_quantiles, default=DEFAULT_TOP_QUANTILES)
    label_specs = _label_specs(cumulative_horizons, quantiles)

    feature_columns = [str(item) for item in list(manifest.get("feature_columns", []) or [])]
    if feature_limit and int(feature_limit) > 0:
        feature_columns = feature_columns[: int(feature_limit)]
    feature_count = len(feature_columns)
    if feature_count <= 0:
        raise ValueError("manifest has no feature columns")
    feature_shape = tuple(int(item) for item in list(manifest.get("feature_panel_shape", []) or []))
    feature_dtype = str(manifest.get("feature_dtype", "float16") or "float16")
    feature_path = _resolve_path(str(manifest.get("feature_panel_path", "") or ""), root=pack_root)
    feature_panel = np.memmap(feature_path, dtype=feature_dtype, mode="r", shape=feature_shape)
    feature_indices = np.arange(feature_count, dtype=np.int64)

    label_arrays_meta = dict(manifest.get("label_arrays", {}) or {})
    needed_label_names = ["rank_by_horizon", "cumulative_return_1to20"]
    labels = {name: _open_memmap(label_arrays_meta[name]) for name in needed_label_names}

    sample_index_path = _resolve_path(str(manifest.get("sample_index_path", "") or ""), root=pack_root)
    index_columns = ["role", "date", "global_date_pos", "global_stock_pos", "_sample_pos"]
    sample_index = pd.read_parquet(sample_index_path, columns=index_columns)
    selected = sample_index[sample_index["role"].astype(str).isin(role_values)].copy()
    if max_rows and int(max_rows) > 0:
        selected = selected.iloc[: int(max_rows)].copy()
    selected["_row_pos"] = np.arange(len(selected), dtype=np.int64)
    selected_sample_pos = pd.to_numeric(selected["_sample_pos"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    selected_stock_pos = pd.to_numeric(selected["global_stock_pos"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    selected_date_pos = pd.to_numeric(selected["global_date_pos"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    selected_roles = selected["role"].astype(str).to_numpy()
    role_scopes = [f"role={role}" for role in sorted(role_values)]
    scope_names = ["all", *role_scopes]
    accumulators = {scope: ScopeFeatureAccumulator(feature_count=feature_count) for scope in scope_names}
    path_accumulators: dict[str, PathAccumulator] = {}
    exit_counts: dict[str, dict[str, int]] = {scope: {"sample_count": 0} for scope in scope_names}

    processed_rows = 0
    for chunk_idx, slc in enumerate(_chunk_slices(len(selected), int(chunk_size)), start=1):
        sample_pos = selected_sample_pos[slc]
        stock_pos = selected_stock_pos[slc]
        date_pos = selected_date_pos[slc]
        role_chunk = selected_roles[slc]
        valid_positions = (
            (sample_pos >= 0)
            & (stock_pos >= 0)
            & (date_pos >= 0)
            & (stock_pos < feature_shape[0])
            & (date_pos < feature_shape[1])
        )
        if not np.all(valid_positions):
            sample_pos = sample_pos[valid_positions]
            stock_pos = stock_pos[valid_positions]
            date_pos = date_pos[valid_positions]
            role_chunk = role_chunk[valid_positions]
        if len(sample_pos) == 0:
            continue
        features = np.asarray(feature_panel[stock_pos, date_pos, :][:, feature_indices], dtype=np.float32)
        finite = np.isfinite(features)
        masks, cumulative_path, _event_extra = _label_masks(
            labels=labels,
            positions=sample_pos,
            cumulative_horizons=cumulative_horizons,
            top_quantiles=quantiles,
            profit_stop_pairs=profit_stop_pairs,
        )
        scope_masks = {"all": np.ones(len(sample_pos), dtype=bool)}
        for role in sorted(role_values):
            scope_masks[f"role={role}"] = role_chunk == role
        for scope, scope_mask in scope_masks.items():
            if not np.any(scope_mask):
                continue
            scoped_features = features[scope_mask]
            scoped_finite = finite[scope_mask]
            accumulators[scope].update_all(scoped_features, scoped_finite)
            exit_counts[scope]["sample_count"] = int(exit_counts[scope].get("sample_count", 0) + int(scope_mask.sum()))
            for label, mask in masks.items():
                scoped_label_mask = mask[scope_mask]
                accumulators[scope].update_label(label, scoped_features, scoped_finite, scoped_label_mask)
                key = f"{scope}|{label}"
                if key not in path_accumulators:
                    path_accumulators[key] = PathAccumulator(day_count=20)
                path_accumulators[key].update(cumulative_path[scope_mask][scoped_label_mask])
            scoped_path = cumulative_path[scope_mask]
            for profit, stop in profit_stop_pairs:
                profit_day = _first_hit_day(scoped_path, profit, direction="up")
                stop_day = _first_hit_day(scoped_path, stop, direction="down")
                valid_path = np.isfinite(scoped_path).all(axis=1)
                success = valid_path & (profit_day > 0) & ((stop_day == 0) | (profit_day <= stop_day))
                fail_stop_first = valid_path & (stop_day > 0) & ((profit_day == 0) | (stop_day < profit_day))
                prefix = f"up{int(round(profit * 100))}_down{int(round(stop * 100))}_20d"
                exit_counts[scope][f"success_{prefix}"] = int(exit_counts[scope].get(f"success_{prefix}", 0) + int(success.sum()))
                exit_counts[scope][f"stop_first_{prefix}"] = int(exit_counts[scope].get(f"stop_first_{prefix}", 0) + int(fail_stop_first.sum()))
                exit_counts[scope][f"no_touch_{prefix}"] = int(
                    exit_counts[scope].get(f"no_touch_{prefix}", 0)
                    + int((valid_path & (profit_day == 0) & (stop_day == 0)).sum())
                )
        processed_rows += int(len(sample_pos))
        if chunk_idx % 10 == 0:
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "processed_rows": int(processed_rows),
                    "selected_rows": int(len(selected)),
                    "chunk_idx": int(chunk_idx),
                    "generated_at": _now(),
                },
            )

    feature_stats = _make_feature_stats_frame(accumulators=accumulators, feature_columns=feature_columns, label_specs=label_specs)
    stability_frame = _make_stability_frame(feature_stats)
    path_frame, path_summary = _make_path_frames(path_accumulators=path_accumulators, label_specs=label_specs)
    exit_frame = _make_exit_frame(exit_counts)

    feature_stats_path = _write_csv(output_dir / "feature_label_effects.csv", feature_stats)
    stability_path = _write_csv(output_dir / "stable_feature_candidates.csv", stability_frame)
    path_frame_path = _write_csv(output_dir / "future_path_profiles.csv", path_frame)
    path_summary_path = _write_csv(output_dir / "future_path_summary.csv", path_summary)
    exit_path = _write_csv(output_dir / "exit_outcome_rates.csv", exit_frame)

    chart_dir = output_dir / "charts"
    chart_paths: list[str] = []
    chart_paths.extend(_plot_path_profiles(path_frame, chart_dir))
    chart_paths.extend(_plot_feature_bars(stability_frame, chart_dir))
    chart_paths.extend(_plot_exit_rates(exit_frame, chart_dir))

    summary: dict[str, Any] = {
        "artifact_type": "qdp_v2_rising_stock_profile",
        "generated_at": _now(),
        "manifest_json": str(manifest_path),
        "output_dir": str(output_dir.resolve()),
        "sample_count": int(len(selected)),
        "processed_rows": int(processed_rows),
        "feature_count": int(feature_count),
        "roles": sorted(role_values),
        "horizons": list(cumulative_horizons),
        "top_quantiles": list(quantiles),
        "profit_stop_pairs": [{"profit": float(p), "stop": float(s)} for p, s in profit_stop_pairs],
        "feature_schema_hash": manifest.get("feature_schema_hash", ""),
        "feature_profile": manifest.get("feature_profile", ""),
        "label_schema_name": manifest.get("label_schema_name", ""),
        "label_schema_version": manifest.get("label_schema_version", ""),
        "outputs": {
            "feature_label_effects_csv": feature_stats_path,
            "stable_feature_candidates_csv": stability_path,
            "future_path_profiles_csv": path_frame_path,
            "future_path_summary_csv": path_summary_path,
            "exit_outcome_rates_csv": exit_path,
            "charts": chart_paths,
        },
    }
    report_path = _build_markdown_report(
        output_dir=output_dir,
        summary=summary,
        feature_stats=feature_stats,
        stability_frame=stability_frame,
        path_summary=path_summary,
        exit_frame=exit_frame,
        chart_paths=chart_paths,
    )
    summary["outputs"]["report_md"] = report_path
    summary_path = _write_json(output_dir / "rising_stock_profile_summary.json", summary)
    summary["outputs"]["summary_json"] = summary_path
    _write_json(progress_path, {"status": "completed", "summary_json": summary_path, "report_md": report_path, "generated_at": _now()})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a QDP v2 rising-stock profile study from the repaired training pack.")
    parser.add_argument("--manifest-json", default=str(DEFAULT_TRAINING_PACK_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-tag", default="qdp_v2_rising_stock_profile")
    parser.add_argument("--roles", default="train,validation,test")
    parser.add_argument("--horizons", default="1,3,5,10,20")
    parser.add_argument("--top-quantiles", default="0.90,0.95")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--feature-limit", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = build_rising_stock_profile(
        manifest_json=args.manifest_json,
        output_root=args.output_root,
        run_tag=args.run_tag,
        roles=args.roles,
        horizons=args.horizons,
        top_quantiles=args.top_quantiles,
        chunk_size=int(args.chunk_size),
        max_rows=int(args.max_rows),
        feature_limit=int(args.feature_limit),
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(summary.get("outputs", {}).get("report_md", ""))
    return summary


if __name__ == "__main__":
    main()
