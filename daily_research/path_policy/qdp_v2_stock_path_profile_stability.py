from __future__ import annotations

import argparse
import gc
import itertools
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

from daily_research.path_policy.qdp_v2_stock_path_profile_atlas import (
    PATH_CLUSTER_FEATURES,
    _json_default,
    _label_clusters,
    _write_csv,
    _write_json,
)


DEFAULT_ATLAS_DIR = Path(
    "daily_research/output/path_policy/stock_path_profile_atlas/"
    "qdp_v2_stock_path_profile_atlas_full_20260704_214229"
)
DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/stock_path_profile_stability")
DEFAULT_CLUSTER_COUNTS = (6, 8, 10, 12)
DEFAULT_SEED = 7
DEFAULT_REPRO_SEEDS = (7, 17, 29)

SUMMARY_METRICS = [
    "buyable_rate",
    "future_max_return_60d",
    "future_min_return_60d",
    "future_final_return_60d",
    "future_peak_day_60d",
    "future_trough_day_60d",
    "drawdown_after_peak_60d",
    "runup_after_trough_60d",
    "path_range_60d",
    "path_efficiency_60d",
    "path_trade_value_60d",
    "time_above_zero_60d",
    "time_below_zero_60d",
]

READ_COLUMNS = [
    *PATH_CLUSTER_FEATURES,
    "path_valid_60d",
    "entry_valid",
    "entry_buyable",
    "year",
    *[col for col in SUMMARY_METRICS if col != "buyable_rate"],
]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_int_list(raw: str | None, *, default: Iterable[int]) -> tuple[int, ...]:
    if not raw:
        return tuple(default)
    values: list[int] = []
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start, end = [int(part.strip()) for part in item.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    return tuple(sorted(set(values))) or tuple(default)


def _find_shards(atlas_dir: Path) -> list[Path]:
    shard_dir = atlas_dir / "path_metric_shards"
    if not shard_dir.exists():
        raise FileNotFoundError(f"path metric shard directory does not exist: {shard_dir}")
    paths = sorted(shard_dir.glob("path_metrics_year_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no path metric shards found under: {shard_dir}")
    return paths


def _read_metric_frame(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=columns)
    if "year" in frame.columns:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("int16")
    return frame


def _valid_mask(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    return (
        frame["path_valid_60d"].astype(bool).to_numpy()
        & frame["entry_valid"].astype(bool).to_numpy()
        & np.isfinite(values).all(axis=1)
    )


def _weighted_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        weights = group["sample_count"].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {name: value for name, value in zip(keys, key)}
        row["sample_count"] = int(weights.sum())
        for col in [c for c in frame.columns if c not in {*keys, "sample_count"}]:
            values = group[col].to_numpy(dtype=np.float64)
            finite = np.isfinite(values) & np.isfinite(weights)
            if finite.any() and weights[finite].sum() > 0:
                row[col] = float(np.average(values[finite], weights=weights[finite]))
            else:
                row[col] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_model(
    shard_paths: list[Path],
    *,
    cluster_count: int,
    seed: int,
    batch_size: int,
    epochs: int,
    progress_path: Path,
) -> tuple[Any, Any, int]:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    fitted_rows = 0
    read_cols = [*PATH_CLUSTER_FEATURES, "path_valid_60d", "entry_valid"]
    for path in shard_paths:
        frame = _read_metric_frame(path, columns=read_cols)
        values = frame[PATH_CLUSTER_FEATURES].to_numpy(dtype=np.float32, copy=True)
        valid = _valid_mask(frame, values)
        if valid.any():
            scaler.partial_fit(values[valid])
            fitted_rows += int(valid.sum())
        _write_json(
            progress_path,
            {
                "status": "scaler_fit",
                "cluster_count": int(cluster_count),
                "seed": int(seed),
                "fitted_rows": int(fitted_rows),
                "last_shard": str(path),
                "updated_at": _now(),
            },
        )
        del frame, values, valid
        gc.collect()

    model = MiniBatchKMeans(
        n_clusters=int(cluster_count),
        random_state=int(seed),
        batch_size=int(batch_size),
        n_init=3,
    )
    trained_rows = 0
    for epoch in range(int(epochs)):
        epoch_rows = 0
        for path in shard_paths:
            frame = _read_metric_frame(path, columns=read_cols)
            values = frame[PATH_CLUSTER_FEATURES].to_numpy(dtype=np.float32, copy=True)
            valid = _valid_mask(frame, values)
            if valid.any():
                model.partial_fit(scaler.transform(values[valid]))
                epoch_rows += int(valid.sum())
            del frame, values, valid
            gc.collect()
        trained_rows += epoch_rows
        _write_json(
            progress_path,
            {
                "status": "model_fit",
                "cluster_count": int(cluster_count),
                "seed": int(seed),
                "epoch": int(epoch + 1),
                "epoch_rows": int(epoch_rows),
                "trained_rows": int(trained_rows),
                "updated_at": _now(),
            },
        )
    return scaler, model, int(fitted_rows)


def _aggregate_assignment(
    shard_paths: list[Path],
    *,
    scaler: Any,
    model: Any,
    cluster_count: int,
    seed: int,
    collect_labels: bool,
    progress_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    cluster_frames: list[pd.DataFrame] = []
    year_cluster_frames: list[pd.DataFrame] = []
    label_parts: list[np.ndarray] = []

    read_cols = list(dict.fromkeys(READ_COLUMNS))
    for path in shard_paths:
        frame = _read_metric_frame(path, columns=read_cols)
        values = frame[PATH_CLUSTER_FEATURES].to_numpy(dtype=np.float32, copy=True)
        valid = _valid_mask(frame, values)
        clusters = np.full(len(frame), -1, dtype=np.int16)
        if valid.any():
            clusters[valid] = model.predict(scaler.transform(values[valid])).astype(np.int16)
        if collect_labels:
            label_parts.append(clusters[valid].astype(np.int16, copy=True))

        frame["path_cluster"] = clusters
        work = frame[frame["path_cluster"] >= 0].copy()
        if not work.empty:
            work["entry_buyable"] = work["entry_buyable"].astype(bool)
            agg = work.groupby("path_cluster", sort=True).agg(
                sample_count=("path_cluster", "size"),
                buyable_rate=("entry_buyable", "mean"),
                future_max_return_60d=("future_max_return_60d", "mean"),
                future_min_return_60d=("future_min_return_60d", "mean"),
                future_final_return_60d=("future_final_return_60d", "mean"),
                future_peak_day_60d=("future_peak_day_60d", "mean"),
                future_trough_day_60d=("future_trough_day_60d", "mean"),
                drawdown_after_peak_60d=("drawdown_after_peak_60d", "mean"),
                runup_after_trough_60d=("runup_after_trough_60d", "mean"),
                path_range_60d=("path_range_60d", "mean"),
                path_efficiency_60d=("path_efficiency_60d", "mean"),
                path_trade_value_60d=("path_trade_value_60d", "mean"),
                time_above_zero_60d=("time_above_zero_60d", "mean"),
                time_below_zero_60d=("time_below_zero_60d", "mean"),
            ).reset_index()
            cluster_frames.append(agg)
            year_agg = work.groupby(["year", "path_cluster"], sort=True).agg(
                sample_count=("path_cluster", "size"),
                buyable_rate=("entry_buyable", "mean"),
                future_max_return_60d=("future_max_return_60d", "mean"),
                future_min_return_60d=("future_min_return_60d", "mean"),
                future_final_return_60d=("future_final_return_60d", "mean"),
                future_peak_day_60d=("future_peak_day_60d", "mean"),
                future_trough_day_60d=("future_trough_day_60d", "mean"),
                drawdown_after_peak_60d=("drawdown_after_peak_60d", "mean"),
                runup_after_trough_60d=("runup_after_trough_60d", "mean"),
                path_range_60d=("path_range_60d", "mean"),
                path_efficiency_60d=("path_efficiency_60d", "mean"),
                path_trade_value_60d=("path_trade_value_60d", "mean"),
                time_above_zero_60d=("time_above_zero_60d", "mean"),
                time_below_zero_60d=("time_below_zero_60d", "mean"),
            ).reset_index()
            year_cluster_frames.append(year_agg)

        _write_json(
            progress_path,
            {
                "status": "assign",
                "cluster_count": int(cluster_count),
                "seed": int(seed),
                "last_shard": str(path),
                "updated_at": _now(),
            },
        )
        del frame, work, values, valid, clusters
        gc.collect()

    raw_cluster = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame()
    cluster_summary = _weighted_rows(raw_cluster, ["path_cluster"])
    if cluster_summary.empty:
        return cluster_summary, pd.DataFrame(), pd.DataFrame(), None

    labels = _label_clusters(cluster_summary.rename(columns={metric: f"{metric}_mean" for metric in SUMMARY_METRICS if metric != "buyable_rate"}))
    # _label_clusters expects *_mean names for all path metrics except sample_count.
    if not labels:
        rename_for_label = cluster_summary.rename(
            columns={metric: f"{metric}_mean" for metric in cluster_summary.columns if metric not in {"path_cluster", "sample_count"}}
        )
        labels = _label_clusters(rename_for_label)

    cluster_summary["cluster_count"] = int(cluster_count)
    cluster_summary["seed"] = int(seed)
    cluster_summary["path_type"] = cluster_summary["path_cluster"].map(labels).fillna("unknown")
    cluster_summary = cluster_summary.rename(columns={metric: f"{metric}_mean" for metric in SUMMARY_METRICS})

    raw_year_cluster = pd.concat(year_cluster_frames, ignore_index=True) if year_cluster_frames else pd.DataFrame()
    year_cluster_summary = _weighted_rows(raw_year_cluster, ["year", "path_cluster"])
    year_cluster_summary["cluster_count"] = int(cluster_count)
    year_cluster_summary["seed"] = int(seed)
    year_cluster_summary["path_type"] = year_cluster_summary["path_cluster"].map(labels).fillna("unknown")
    year_cluster_summary = year_cluster_summary.rename(columns={metric: f"{metric}_mean" for metric in SUMMARY_METRICS})

    type_work = cluster_summary.rename(columns={f"{metric}_mean": metric for metric in SUMMARY_METRICS})
    type_summary = _weighted_rows(type_work, ["path_type"])
    total_count = float(type_summary["sample_count"].sum()) if not type_summary.empty else 0.0
    type_summary["sample_share"] = np.where(total_count > 0, type_summary["sample_count"] / total_count, np.nan)
    type_summary["cluster_count"] = int(cluster_count)
    type_summary["seed"] = int(seed)
    type_summary = type_summary.rename(columns={metric: f"{metric}_mean" for metric in SUMMARY_METRICS})

    year_type_work = year_cluster_summary.rename(columns={f"{metric}_mean": metric for metric in SUMMARY_METRICS})
    year_type = _weighted_rows(year_type_work, ["year", "path_type"])
    year_totals = year_type.groupby("year", sort=True)["sample_count"].sum().rename("year_total").reset_index()
    year_type = year_type.merge(year_totals, on="year", how="left", validate="many_to_one")
    year_type["year_share"] = np.where(year_type["year_total"] > 0, year_type["sample_count"] / year_type["year_total"], np.nan)
    year_type["cluster_count"] = int(cluster_count)
    year_type["seed"] = int(seed)
    year_type = year_type.rename(columns={metric: f"{metric}_mean" for metric in SUMMARY_METRICS})

    labels_array = np.concatenate(label_parts).astype(np.int16, copy=False) if collect_labels and label_parts else None
    return cluster_summary, type_summary, year_type, labels_array


def _stability_by_year(year_type: pd.DataFrame) -> pd.DataFrame:
    if year_type.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (cluster_count, seed, path_type), group in year_type.groupby(["cluster_count", "seed", "path_type"], sort=True):
        shares = group["year_share"].to_numpy(dtype=np.float64)
        mean_share = float(np.nanmean(shares))
        std_share = float(np.nanstd(shares))
        rows.append(
            {
                "cluster_count": int(cluster_count),
                "seed": int(seed),
                "path_type": str(path_type),
                "n_years": int(group["year"].nunique()),
                "mean_year_share": mean_share,
                "std_year_share": std_share,
                "cv_year_share": float(std_share / mean_share) if mean_share > 0 else np.nan,
                "min_year_share": float(np.nanmin(shares)),
                "max_year_share": float(np.nanmax(shares)),
                "sample_count": int(group["sample_count"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_outputs(
    output_dir: Path,
    *,
    type_summary: pd.DataFrame,
    year_type: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    reference_cluster_count: int,
    reference_seed: int,
) -> list[str]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    ref_types = type_summary[type_summary["seed"].eq(int(reference_seed))].copy()
    if not ref_types.empty:
        pivot = ref_types.pivot_table(index="cluster_count", columns="path_type", values="sample_share", aggfunc="sum").fillna(0.0)
        pivot = pivot.sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        bottom = np.zeros(len(pivot), dtype=np.float64)
        x = np.arange(len(pivot))
        for path_type in sorted(pivot.columns):
            values = pivot[path_type].to_numpy(dtype=np.float64)
            ax.bar(x, values * 100.0, bottom=bottom * 100.0, label=path_type)
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in pivot.index])
        ax.set_xlabel("Cluster count")
        ax.set_ylabel("Sample share (%)")
        ax.set_title("Path type share stability across cluster counts")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = chart_dir / "path_type_share_by_cluster_count.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))

    ref_year = year_type[
        year_type["cluster_count"].eq(int(reference_cluster_count)) & year_type["seed"].eq(int(reference_seed))
    ].copy()
    if not ref_year.empty:
        pivot = ref_year.pivot_table(index="path_type", columns="year", values="year_share", aggfunc="sum").fillna(0.0)
        pivot = pivot.sort_index()
        fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(pivot))))
        image = ax.imshow(pivot.to_numpy(dtype=np.float64) * 100.0, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(int(col)) for col in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(v) for v in pivot.index])
        ax.set_title(f"Path type year share heatmap (k={reference_cluster_count}, seed={reference_seed})")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Year share (%)")
        fig.tight_layout()
        path = chart_dir / "path_type_year_share_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))

    ref_cluster = cluster_summary[
        cluster_summary["cluster_count"].eq(int(reference_cluster_count)) & cluster_summary["seed"].eq(int(reference_seed))
    ].copy()
    if not ref_cluster.empty:
        frame = ref_cluster.sort_values("future_final_return_60d_mean")
        labels = [f"{int(row.path_cluster)}:{row.path_type}" for row in frame.itertuples()]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, frame["future_final_return_60d_mean"] * 100.0, label="final")
        ax.scatter(labels, frame["future_max_return_60d_mean"] * 100.0, color="#c44e52", label="max high", zorder=3)
        ax.scatter(labels, frame["future_min_return_60d_mean"] * 100.0, color="#4c72b0", label="min low", zorder=3)
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_title(f"Reference path outcomes (k={reference_cluster_count}, seed={reference_seed})")
        ax.set_ylabel("Return (%)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = chart_dir / "reference_path_outcomes.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs


def _build_report(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    type_summary: pd.DataFrame,
    year_stability: pd.DataFrame,
    seed_reproducibility: pd.DataFrame,
    chart_paths: list[str],
) -> str:
    reference_cluster_count = int(summary["reference_cluster_count"])
    reference_seed = int(summary["reference_seed"])
    ref = type_summary[
        type_summary["cluster_count"].eq(reference_cluster_count) & type_summary["seed"].eq(reference_seed)
    ].copy()
    lines: list[str] = []
    lines.append("# QDP v2 Stock Path Profile Stability")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "This report refits future-path clustering on the existing full path metric shards. "
        "It tests whether the stock path atlas is stable across cluster counts and random seeds, "
        "without changing the QDP active data base."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- atlas_dir: `{summary.get('atlas_dir')}`")
    lines.append(f"- shards: {summary.get('shard_count')}")
    lines.append(f"- clustered rows per fit: {summary.get('clustered_rows'):,}")
    lines.append(f"- cluster counts: {summary.get('cluster_counts')}")
    lines.append(f"- seed reproducibility: {summary.get('seed_reproducibility_seeds')}")
    lines.append("")
    lines.append("## Reference Path Types")
    lines.append("")
    if not ref.empty:
        for row in ref.sort_values("future_final_return_60d_mean", ascending=False).to_dict("records"):
            lines.append(
                f"- {row['path_type']}: share={row['sample_share'] * 100:.2f}%, "
                f"max={row['future_max_return_60d_mean'] * 100:.2f}%, "
                f"final={row['future_final_return_60d_mean'] * 100:.2f}%, "
                f"min={row['future_min_return_60d_mean'] * 100:.2f}%, "
                f"trade_value={row['path_trade_value_60d_mean'] * 100:.2f}%"
            )
    lines.append("")
    lines.append("## Seed Reproducibility")
    lines.append("")
    if seed_reproducibility.empty:
        lines.append("- Not run.")
    else:
        for row in seed_reproducibility.to_dict("records"):
            lines.append(
                f"- k={int(row['cluster_count'])}, seeds {int(row['seed_a'])}/{int(row['seed_b'])}: "
                f"adjusted_rand_index={row['adjusted_rand_index']:.4f}"
            )
    lines.append("")
    lines.append("## Year Stability")
    lines.append("")
    focus = year_stability[
        year_stability["cluster_count"].eq(reference_cluster_count) & year_stability["seed"].eq(reference_seed)
    ].copy()
    if not focus.empty:
        for row in focus.sort_values("mean_year_share", ascending=False).to_dict("records"):
            lines.append(
                f"- {row['path_type']}: mean_share={row['mean_year_share'] * 100:.2f}%, "
                f"range={row['min_year_share'] * 100:.2f}%..{row['max_year_share'] * 100:.2f}%, "
                f"cv={row['cv_year_share']:.3f}"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This validates the existence of broad future-path regimes: durable up, trend up, mixed/spike-fade, sideways compression, and down paths.")
    lines.append("- It does not prove we can predict those regimes before they happen; it only proves the target space is structured enough to model.")
    lines.append("- Exact subtype boundaries are expected to move as k changes. The robust object is the broad path family, not a single cluster id.")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for path in chart_paths:
        lines.append(f"- `{path}`")
    lines.append("")
    path = output_dir / "stock_path_profile_stability_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


@dataclass(frozen=True)
class StabilityConfig:
    atlas_dir: Path
    output_root: Path
    run_tag: str
    cluster_counts: tuple[int, ...]
    seed: int
    seed_reproducibility_seeds: tuple[int, ...]
    reference_cluster_count: int
    batch_size: int
    epochs: int


def build_stock_path_profile_stability(config: StabilityConfig) -> dict[str, Any]:
    atlas_dir = config.atlas_dir.resolve()
    shard_paths = _find_shards(atlas_dir)
    output_dir = config.output_root / f"{config.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "started", "updated_at": _now()})

    cluster_frames: list[pd.DataFrame] = []
    type_frames: list[pd.DataFrame] = []
    year_type_frames: list[pd.DataFrame] = []
    label_cache: dict[tuple[int, int], np.ndarray] = {}
    clustered_rows = 0

    planned_fit_map: dict[tuple[int, int], bool] = {(int(k), int(config.seed)): False for k in config.cluster_counts}
    repro_seeds = tuple(int(seed) for seed in config.seed_reproducibility_seeds)
    for seed in repro_seeds:
        key = (int(config.reference_cluster_count), int(seed))
        planned_fit_map[key] = True
    planned_fits = [(k, seed, collect) for (k, seed), collect in sorted(planned_fit_map.items())]

    for cluster_count, seed, collect_labels in planned_fits:
        scaler, model, fitted_rows = _fit_model(
            shard_paths,
            cluster_count=int(cluster_count),
            seed=int(seed),
            batch_size=int(config.batch_size),
            epochs=int(config.epochs),
            progress_path=progress_path,
        )
        clustered_rows = max(clustered_rows, int(fitted_rows))
        cluster_summary, type_summary, year_type, labels = _aggregate_assignment(
            shard_paths,
            scaler=scaler,
            model=model,
            cluster_count=int(cluster_count),
            seed=int(seed),
            collect_labels=bool(collect_labels),
            progress_path=progress_path,
        )
        cluster_frames.append(cluster_summary)
        type_frames.append(type_summary)
        year_type_frames.append(year_type)
        if labels is not None:
            label_cache[(int(cluster_count), int(seed))] = labels
        del scaler, model
        gc.collect()

    cluster_summary = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame()
    type_summary = pd.concat(type_frames, ignore_index=True) if type_frames else pd.DataFrame()
    year_type = pd.concat(year_type_frames, ignore_index=True) if year_type_frames else pd.DataFrame()
    year_stability = _stability_by_year(year_type)

    repro_rows: list[dict[str, Any]] = []
    if len(repro_seeds) >= 2:
        from sklearn.metrics import adjusted_rand_score

        for seed_a, seed_b in itertools.combinations(repro_seeds, 2):
            labels_a = label_cache.get((int(config.reference_cluster_count), int(seed_a)))
            labels_b = label_cache.get((int(config.reference_cluster_count), int(seed_b)))
            if labels_a is None or labels_b is None or len(labels_a) != len(labels_b):
                continue
            repro_rows.append(
                {
                    "cluster_count": int(config.reference_cluster_count),
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "adjusted_rand_index": float(adjusted_rand_score(labels_a, labels_b)),
                    "label_count": int(len(labels_a)),
                }
            )
    seed_reproducibility = pd.DataFrame(repro_rows)

    chart_paths = _plot_outputs(
        output_dir,
        type_summary=type_summary,
        year_type=year_type,
        cluster_summary=cluster_summary,
        reference_cluster_count=int(config.reference_cluster_count),
        reference_seed=int(config.seed),
    )

    outputs = {
        "cluster_summary_csv": _write_csv(output_dir / "cluster_summary_by_k_seed.csv", cluster_summary),
        "type_summary_csv": _write_csv(output_dir / "path_type_summary_by_k_seed.csv", type_summary),
        "year_type_csv": _write_csv(output_dir / "path_type_year_share_by_k_seed.csv", year_type),
        "year_stability_csv": _write_csv(output_dir / "path_type_year_stability.csv", year_stability),
        "seed_reproducibility_csv": _write_csv(output_dir / "seed_reproducibility.csv", seed_reproducibility),
        "charts": chart_paths,
    }
    summary: dict[str, Any] = {
        "artifact_type": "qdp_v2_stock_path_profile_stability",
        "generated_at": _now(),
        "atlas_dir": str(atlas_dir),
        "output_dir": str(output_dir.resolve()),
        "shard_count": int(len(shard_paths)),
        "clustered_rows": int(clustered_rows),
        "cluster_counts": list(config.cluster_counts),
        "seed": int(config.seed),
        "reference_seed": int(config.seed),
        "seed_reproducibility_seeds": list(repro_seeds),
        "reference_cluster_count": int(config.reference_cluster_count),
        "batch_size": int(config.batch_size),
        "epochs": int(config.epochs),
        "outputs": outputs,
    }
    report_path = _build_report(
        output_dir,
        summary=summary,
        type_summary=type_summary,
        year_stability=year_stability,
        seed_reproducibility=seed_reproducibility,
        chart_paths=chart_paths,
    )
    summary["outputs"]["report_md"] = report_path
    summary_path = _write_json(output_dir / "stock_path_profile_stability_summary.json", summary)
    summary["outputs"]["summary_json"] = summary_path
    _write_json(progress_path, {"status": "completed", "summary_json": summary_path, "report_md": report_path, "updated_at": _now()})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate stock path profile stability across cluster counts and seeds.")
    parser.add_argument("--atlas-dir", default=str(DEFAULT_ATLAS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-tag", default="qdp_v2_stock_path_profile_stability")
    parser.add_argument("--cluster-counts", default=",".join(str(v) for v in DEFAULT_CLUSTER_COUNTS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seed-reproducibility-seeds", default=",".join(str(v) for v in DEFAULT_REPRO_SEEDS))
    parser.add_argument("--reference-cluster-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = build_stock_path_profile_stability(
        StabilityConfig(
            atlas_dir=Path(args.atlas_dir),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            cluster_counts=_parse_int_list(str(args.cluster_counts), default=DEFAULT_CLUSTER_COUNTS),
            seed=int(args.seed),
            seed_reproducibility_seeds=_parse_int_list(str(args.seed_reproducibility_seeds), default=DEFAULT_REPRO_SEEDS),
            reference_cluster_count=int(args.reference_cluster_count),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
        )
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(summary.get("outputs", {}).get("report_md", ""))
    return summary


if __name__ == "__main__":
    main()
