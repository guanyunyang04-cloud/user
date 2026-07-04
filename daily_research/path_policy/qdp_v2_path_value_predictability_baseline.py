from __future__ import annotations

import argparse
import gc
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

from daily_research.path_policy.qdp_v2_raw_rising_path_atlas import (
    DAILY_RAW_COLUMNS,
    INTRADAY_SIGNAL_COLUMNS,
    LIMIT_SIGNAL_COLUMNS,
    RAW_SIGNAL_COLUMNS,
    _add_raw_daily_signals,
    _read_dataset,
    _write_csv,
    _write_json,
)
from daily_research.path_policy.qdp_v2_stock_path_profile_atlas import (
    DEFAULT_FORWARD_DAYS,
    DEFAULT_QDP_ROOT,
    _horizon_col,
    _json_default,
    _read_active,
    _read_dataset_date_range,
    path_target_columns,
)


DEFAULT_ATLAS_DIR = Path(
    "daily_research/output/path_policy/stock_path_profile_atlas/"
    "qdp_v2_stock_path_profile_atlas_full_20260704_214229"
)
DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/path_value_predictability")
DEFAULT_TRAIN_YEARS = tuple(range(2012, 2024))
DEFAULT_VALIDATION_YEARS = (2024,)
DEFAULT_TEST_YEARS = (2025,)
DEFAULT_TOP_K = (20, 50, 100)
DEFAULT_SEED = 7

PAST_CLOSE_DAYS = 100
PAST_LIQUIDITY_LAGS = (1, 2, 3, 5, 10, 20, 40, 60, 100)
TARGET_COLUMN = _horizon_col("path_trade_value", DEFAULT_FORWARD_DAYS)
TARGET_COLUMNS = path_target_columns(DEFAULT_FORWARD_DAYS)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_years(raw: str | Iterable[int] | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    if not isinstance(raw, str):
        return tuple(sorted({int(value) for value in raw}))
    values: list[int] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start, end = [int(part.strip()) for part in item.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    return tuple(sorted(set(values))) or default


def _parse_int_list(raw: str | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    return tuple(sorted(set(values))) or default


def _path_metric_shards(atlas_dir: Path) -> list[Path]:
    paths = sorted((atlas_dir / "path_metric_shards").glob("path_metrics_year_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no path metric shards found under {atlas_dir / 'path_metric_shards'}")
    return paths


def _add_past_path_features(daily: pd.DataFrame, *, past_close_days: int = PAST_CLOSE_DAYS) -> tuple[pd.DataFrame, list[str]]:
    daily = daily.sort_values(["symbol", "trade_date"], kind="mergesort").reset_index(drop=True)
    symbol_key = daily["symbol"]
    close = pd.to_numeric(daily["close"], errors="coerce").astype("float64")
    amount = pd.to_numeric(daily["amount"], errors="coerce").astype("float64")
    volume = pd.to_numeric(daily["volume"], errors="coerce").astype("float64")
    out_cols = ["amount_log", "volume_log"]
    feature_data: dict[str, pd.Series | np.ndarray] = {
        "amount_log": np.log1p(amount),
        "volume_log": np.log1p(volume),
    }
    for lag in range(1, int(past_close_days) + 1):
        close_lag = close.groupby(symbol_key, sort=False).shift(int(lag)).astype("float64")
        c_name = f"past_close_ret_{int(lag)}d"
        feature_data[c_name] = close.div(close_lag.replace(0.0, np.nan)).sub(1.0)
        out_cols.append(c_name)
    for lag in PAST_LIQUIDITY_LAGS:
        amount_lag = amount.groupby(symbol_key, sort=False).shift(int(lag)).astype("float64")
        volume_lag = volume.groupby(symbol_key, sort=False).shift(int(lag)).astype("float64")
        a_name = f"past_amount_log_{int(lag)}d"
        v_name = f"past_volume_log_{int(lag)}d"
        feature_data[a_name] = np.log1p(amount_lag)
        feature_data[v_name] = np.log1p(volume_lag)
        out_cols.extend([a_name, v_name])
    feature_frame = pd.DataFrame(feature_data, index=daily.index).replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.astype("float32")
    return pd.concat([daily, feature_frame], axis=1, copy=False), out_cols


def _to_numeric_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
    return frame


def _build_feature_shards(
    *,
    qdp_root: Path,
    atlas_dir: Path,
    output_dir: Path,
    progress_path: Path,
    years: tuple[int, ...],
    forward_days: int,
    past_close_days: int,
) -> tuple[list[Path], list[str]]:
    root = qdp_root.resolve()
    active = _read_active(root)
    _warmup_daily = pd.DataFrame(columns=DAILY_RAW_COLUMNS)
    _warmup_daily, past_path_cols = _add_past_path_features(_warmup_daily, past_close_days=past_close_days)
    daily_feature_cols = list(dict.fromkeys([*RAW_SIGNAL_COLUMNS, *past_path_cols]))
    feature_cols = list(dict.fromkeys([*daily_feature_cols, *INTRADAY_SIGNAL_COLUMNS, *LIMIT_SIGNAL_COLUMNS]))
    shard_dir = output_dir / "feature_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    path_shards = _path_metric_shards(atlas_dir)
    target_column = _horizon_col("path_trade_value", forward_days)
    target_columns = path_target_columns(forward_days)
    valid_col = _horizon_col("path_valid", forward_days)
    for shard_path in path_shards:
        year = int(shard_path.stem.rsplit("_", 1)[-1])
        if year not in years:
            continue
        _write_json(progress_path, {"status": "build_feature_shard", "year": int(year), "updated_at": _now()})
        metric_cols = [
            "symbol",
            "trade_date",
            "entry_buyable",
            "entry_valid",
            valid_col,
            *target_columns,
        ]
        metrics = pd.read_parquet(shard_path, columns=metric_cols)
        metrics["trade_date"] = metrics["trade_date"].astype(str)
        metrics["symbol"] = metrics["symbol"].astype(str).str.upper().str.strip()
        metrics = metrics[
            metrics["entry_valid"].astype(bool)
            & metrics[valid_col].astype(bool)
            & metrics["entry_buyable"].astype(bool)
            & pd.to_numeric(metrics[target_column], errors="coerce").notna()
        ].copy()
        # Compute rolling and lagged daily features on a small warmup window rather
        # than expanding the entire daily history into a wide in-memory table.
        daily_start = f"{max(2011, year - 1)}-01-01"
        daily_end = f"{year}-12-31"
        daily_window = _read_dataset_date_range(root, active, "market_daily_raw", DAILY_RAW_COLUMNS, daily_start, daily_end)
        daily_window = _add_raw_daily_signals(daily_window)
        daily_window, _past_cols = _add_past_path_features(daily_window, past_close_days=past_close_days)
        daily_window["year"] = daily_window["trade_date"].str.slice(0, 4).astype("int16")
        base = daily_window[daily_window["year"].eq(int(year))][["symbol", "trade_date", *daily_feature_cols]].copy()
        frame = metrics.merge(base, on=["symbol", "trade_date"], how="left", validate="one_to_one")
        start, end = f"{year}-01-01", f"{year}-12-31"
        intraday = _read_dataset_date_range(root, active, "intraday_daily_features", ["symbol", "trade_date", *INTRADAY_SIGNAL_COLUMNS], start, end)
        limit = _read_dataset_date_range(root, active, "limit_intraday_features", ["symbol", "trade_date", *LIMIT_SIGNAL_COLUMNS], start, end)
        if not intraday.empty:
            frame = frame.merge(intraday, on=["symbol", "trade_date"], how="left", validate="one_to_one")
        if not limit.empty:
            frame = frame.merge(limit, on=["symbol", "trade_date"], how="left", validate="one_to_one")
        frame["year"] = int(year)
        frame = _to_numeric_feature_frame(frame, feature_cols)
        for col in target_columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
        keep_cols = ["symbol", "trade_date", "year", *target_columns, *feature_cols]
        out_path = shard_dir / f"path_value_features_year_{year}.parquet"
        frame[keep_cols].to_parquet(out_path, index=False)
        out_paths.append(out_path)
        del metrics, daily_window, base, intraday, limit, frame
        gc.collect()
    return out_paths, feature_cols


def _load_split(
    shard_paths: list[Path],
    *,
    years: tuple[int, ...],
    feature_cols: list[str],
    target_column: str,
    target_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames: list[pd.DataFrame] = []
    for path in shard_paths:
        year = int(path.stem.rsplit("_", 1)[-1])
        if year in years:
            frames.append(pd.read_parquet(path, columns=["symbol", "trade_date", "year", *target_columns, *feature_cols]))
    if not frames:
        raise ValueError(f"no feature shards for years={years}")
    meta = pd.concat(frames, ignore_index=True)
    values = meta[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=True)
    target = pd.to_numeric(meta[target_column], errors="coerce").to_numpy(dtype=np.float32, copy=True)
    valid = np.isfinite(target)
    meta = meta.loc[valid].reset_index(drop=True)
    values = values[valid]
    target = target[valid]
    return meta, values, target


def _load_xy_split(
    shard_paths: list[Path],
    *,
    years: tuple[int, ...],
    feature_cols: list[str],
    target_column: str,
) -> tuple[int, np.ndarray, np.ndarray]:
    selected_paths = [path for path in shard_paths if int(path.stem.rsplit("_", 1)[-1]) in years]
    row_count = 0
    for path in selected_paths:
        target_only = pd.read_parquet(path, columns=[target_column])
        target = pd.to_numeric(target_only[target_column], errors="coerce").to_numpy(dtype=np.float32, copy=True)
        row_count += int(np.isfinite(target).sum())
        del target_only, target
        gc.collect()
    if row_count <= 0:
        raise ValueError(f"no training rows for years={years}")

    x = np.empty((int(row_count), len(feature_cols)), dtype=np.float32)
    y = np.empty(int(row_count), dtype=np.float32)
    cursor = 0
    for path in selected_paths:
        year = int(path.stem.rsplit("_", 1)[-1])
        frame = pd.read_parquet(path, columns=[target_column, *feature_cols])
        values = frame[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=True)
        target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.float32, copy=True)
        valid = np.isfinite(target)
        if valid.any():
            n = int(valid.sum())
            x[cursor : cursor + n] = values[valid]
            y[cursor : cursor + n] = target[valid]
            cursor += n
        del frame, values, target, valid
        gc.collect()
    return int(cursor), x[:cursor], y[:cursor]


def _daily_spearman(frame: pd.DataFrame, *, score_col: str, target_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("trade_date", sort=True):
        if len(group) < 20:
            continue
        score = pd.to_numeric(group[score_col], errors="coerce")
        target = pd.to_numeric(group[target_col], errors="coerce")
        corr = score.corr(target, method="spearman")
        if math.isfinite(float(corr)):
            rows.append({"trade_date": str(date), "rank_ic": float(corr), "sample_count": int(len(group))})
    return pd.DataFrame(rows)


def _topk_metrics(frame: pd.DataFrame, *, top_k_values: tuple[int, ...], score_col: str, forward_days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        _horizon_col("future_max_return", forward_days),
        _horizon_col("future_min_return", forward_days),
        _horizon_col("future_final_return", forward_days),
        _horizon_col("drawdown_after_peak", forward_days),
        _horizon_col("path_trade_value", forward_days),
    ]
    for top_k in top_k_values:
        daily_rows: list[dict[str, Any]] = []
        for date, group in frame.groupby("trade_date", sort=True):
            if len(group) < top_k:
                continue
            top = group.nlargest(int(top_k), score_col)
            row: dict[str, Any] = {"trade_date": str(date), "top_k": int(top_k), "universe_count": int(len(group))}
            for col in metric_cols:
                row[f"selected_{col}"] = float(pd.to_numeric(top[col], errors="coerce").mean())
                row[f"universe_{col}"] = float(pd.to_numeric(group[col], errors="coerce").mean())
                row[f"alpha_{col}"] = row[f"selected_{col}"] - row[f"universe_{col}"]
            row["selected_positive_final_rate"] = float((pd.to_numeric(top[_horizon_col("future_final_return", forward_days)], errors="coerce") > 0).mean())
            row["selected_hit_10pct_rate"] = float((pd.to_numeric(top[_horizon_col("future_max_return", forward_days)], errors="coerce") >= 0.10).mean())
            row["selected_loss_5pct_rate"] = float((pd.to_numeric(top[_horizon_col("future_min_return", forward_days)], errors="coerce") <= -0.05).mean())
            daily_rows.append(row)
        daily = pd.DataFrame(daily_rows)
        if daily.empty:
            continue
        out: dict[str, Any] = {"top_k": int(top_k), "day_count": int(len(daily))}
        for col in [c for c in daily.columns if c not in {"trade_date", "top_k"}]:
            out[col] = float(pd.to_numeric(daily[col], errors="coerce").mean())
        rows.append(out)
    return pd.DataFrame(rows)


def _evaluate_split(
    meta: pd.DataFrame,
    pred: np.ndarray,
    *,
    split: str,
    top_k_values: tuple[int, ...],
    target_column: str,
    target_columns: list[str],
    forward_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = meta[["symbol", "trade_date", "year", *target_columns]].copy()
    frame["score"] = pred.astype("float32")
    ic = _daily_spearman(frame, score_col="score", target_col=target_column)
    topk = _topk_metrics(frame, top_k_values=top_k_values, score_col="score", forward_days=forward_days)
    if not topk.empty:
        topk.insert(0, "split", split)
    metrics = {
        "split": split,
        "row_count": int(len(frame)),
        "date_count": int(frame["trade_date"].nunique()),
        "rank_ic_mean": float(ic["rank_ic"].mean()) if not ic.empty else np.nan,
        "rank_ic_median": float(ic["rank_ic"].median()) if not ic.empty else np.nan,
        "rank_ic_positive_day_rate": float((ic["rank_ic"] > 0).mean()) if not ic.empty else np.nan,
        "target_mean": float(frame[target_column].mean()),
        "prediction_mean": float(frame["score"].mean()),
    }
    return ic, topk, metrics


def _predict_with_feature_names(model: Any, values: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    frame = pd.DataFrame(values, columns=feature_cols, copy=False)
    return model.predict(frame, num_iteration=model.best_iteration_).astype("float32")


def _plot_outputs(output_dir: Path, topk: pd.DataFrame, feature_importance: pd.DataFrame, *, forward_days: int) -> list[str]:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    if not topk.empty:
        for metric in [
            f"alpha_{_horizon_col('path_trade_value', forward_days)}",
            f"alpha_{_horizon_col('future_final_return', forward_days)}",
            f"alpha_{_horizon_col('future_max_return', forward_days)}",
        ]:
            col = metric if metric in topk.columns else f"{metric}"
            if col not in topk.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            for split, group in topk.groupby("split", sort=True):
                group = group.sort_values("top_k")
                ax.plot(group["top_k"], group[col] * 100.0, marker="o", label=split)
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.set_xlabel("Top K per day")
            ax.set_ylabel("Alpha vs same-day universe (%)")
            ax.set_title(col)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            path = chart_dir / f"{col}_by_topk.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            outputs.append(str(path.resolve()))
    if not feature_importance.empty:
        focus = feature_importance.head(40).copy().iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(focus["feature"], focus["importance_gain"])
        ax.set_title("LightGBM feature importance by gain")
        ax.set_xlabel("Gain")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path = chart_dir / "feature_importance_gain_top40.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs


def _build_report(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    split_metrics: pd.DataFrame,
    topk: pd.DataFrame,
    feature_importance: pd.DataFrame,
    chart_paths: list[str],
) -> str:
    forward_days = int(summary.get("forward_days", DEFAULT_FORWARD_DAYS))
    value_alpha_col = f"alpha_{_horizon_col('path_trade_value', forward_days)}"
    final_alpha_col = f"alpha_{_horizon_col('future_final_return', forward_days)}"
    max_alpha_col = f"alpha_{_horizon_col('future_max_return', forward_days)}"
    lines: list[str] = []
    lines.append("# QDP v2 Path Value Predictability Baseline")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        f"This baseline predicts next-open anchored future {forward_days}-day path_trade_value from pre-signal daily state, "
        "full past close-path lags, intraday summaries, and limit-board structure. It is a predictability bridge, not the final raw sequence model."
    )
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- train_years: {summary.get('train_years')}")
    lines.append(f"- validation_years: {summary.get('validation_years')}")
    lines.append(f"- test_years: {summary.get('test_years')}")
    lines.append(f"- feature_count: {summary.get('feature_count')}")
    lines.append(f"- output_dir: `{summary.get('output_dir')}`")
    lines.append("")
    lines.append("## Split Metrics")
    lines.append("")
    for row in split_metrics.to_dict("records"):
        lines.append(
            f"- {row['split']}: rows={int(row['row_count']):,}, "
            f"rank_ic_mean={row['rank_ic_mean']:.4f}, "
            f"rank_ic_positive_day_rate={row['rank_ic_positive_day_rate']:.2%}, "
            f"target_mean={row['target_mean'] * 100:.2f}%"
        )
    lines.append("")
    lines.append("## Top-K Validation")
    lines.append("")
    if not topk.empty:
        focus = topk[topk["split"].isin(["validation", "test"])].copy()
        for row in focus.sort_values(["split", "top_k"]).to_dict("records"):
            lines.append(
                f"- {row['split']} top{int(row['top_k'])}: "
                f"path_value_alpha={row[value_alpha_col] * 100:.2f}%, "
                f"final_alpha={row[final_alpha_col] * 100:.2f}%, "
                f"max_alpha={row[max_alpha_col] * 100:.2f}%, "
                f"hit10={row['selected_hit_10pct_rate']:.2%}, "
                f"loss5={row['selected_loss_5pct_rate']:.2%}"
            )
    lines.append("")
    lines.append("## Top Features")
    lines.append("")
    for row in feature_importance.head(20).to_dict("records"):
        lines.append(f"- {row['feature']}: gain={row['importance_gain']:.2f}, split={int(row['importance_split'])}")
    lines.append("")
    lines.append("## Boundaries")
    lines.append("")
    lines.append("- This uses summary/path-state features, not full raw N-day sequence tensors.")
    lines.append("- A positive top-K spread means the feature set contains predictive information; it is not yet a complete execution strategy.")
    lines.append("- The next step is to build the unified sequence training pack and compare it against this baseline and the old 307-feature model.")
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    for path in chart_paths:
        lines.append(f"- `{path}`")
    path = output_dir / "path_value_predictability_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


@dataclass(frozen=True)
class BaselineConfig:
    qdp_root: Path
    atlas_dir: Path
    output_root: Path
    run_tag: str
    train_years: tuple[int, ...]
    validation_years: tuple[int, ...]
    test_years: tuple[int, ...]
    top_k: tuple[int, ...]
    seed: int
    forward_days: int
    past_close_days: int
    n_estimators: int
    learning_rate: float
    num_leaves: int
    n_jobs: int


def build_path_value_predictability_baseline(config: BaselineConfig) -> dict[str, Any]:
    output_dir = config.output_root / f"{config.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "started", "updated_at": _now()})
    all_years = tuple(sorted(set(config.train_years + config.validation_years + config.test_years)))
    target_column = _horizon_col("path_trade_value", config.forward_days)
    target_columns = path_target_columns(config.forward_days)
    shard_paths, feature_cols = _build_feature_shards(
        qdp_root=config.qdp_root,
        atlas_dir=config.atlas_dir,
        output_dir=output_dir,
        progress_path=progress_path,
        years=all_years,
        forward_days=config.forward_days,
        past_close_days=config.past_close_days,
    )
    _write_json(progress_path, {"status": "loading_splits", "updated_at": _now()})
    train_row_count, x_train, y_train = _load_xy_split(shard_paths, years=config.train_years, feature_cols=feature_cols, target_column=target_column)
    val_meta, x_val, y_val = _load_split(shard_paths, years=config.validation_years, feature_cols=feature_cols, target_column=target_column, target_columns=target_columns)
    test_meta, x_test, y_test = _load_split(shard_paths, years=config.test_years, feature_cols=feature_cols, target_column=target_column, target_columns=target_columns)

    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    model = LGBMRegressor(
        objective="regression_l2",
        n_estimators=int(config.n_estimators),
        learning_rate=float(config.learning_rate),
        num_leaves=int(config.num_leaves),
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=int(config.seed),
        n_jobs=int(config.n_jobs),
        verbosity=-1,
    )
    _write_json(progress_path, {"status": "training", "train_rows": int(train_row_count), "validation_rows": int(len(val_meta)), "updated_at": _now()})
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l2",
        feature_name=feature_cols,
        callbacks=[early_stopping(60, verbose=False), log_evaluation(100)],
    )

    train_metrics = {
        "split": "train",
        "row_count": int(train_row_count),
        "date_count": 0,
        "rank_ic_mean": np.nan,
        "rank_ic_median": np.nan,
        "rank_ic_positive_day_rate": np.nan,
        "target_mean": float(np.nanmean(y_train)),
        "prediction_mean": np.nan,
        "note": "train symbol/date metadata not kept to reduce memory; validation/test carry ranking metrics",
    }
    del x_train, y_train
    gc.collect()
    pred_val = _predict_with_feature_names(model, x_val, feature_cols)
    pred_test = _predict_with_feature_names(model, x_test, feature_cols)

    val_ic, val_topk, val_metrics = _evaluate_split(
        val_meta,
        pred_val,
        split="validation",
        top_k_values=config.top_k,
        target_column=target_column,
        target_columns=target_columns,
        forward_days=config.forward_days,
    )
    test_ic, test_topk, test_metrics = _evaluate_split(
        test_meta,
        pred_test,
        split="test",
        top_k_values=config.top_k,
        target_column=target_column,
        target_columns=target_columns,
        forward_days=config.forward_days,
    )
    split_metrics = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    topk = pd.concat([val_topk, test_topk], ignore_index=True)
    daily_ic = pd.concat(
        [
            val_ic.assign(split="validation"),
            test_ic.assign(split="test"),
        ],
        ignore_index=True,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["importance_gain", "importance_split"], ascending=False, kind="mergesort")

    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    val_pred = val_meta[["symbol", "trade_date", "year", *target_columns]].copy()
    val_pred["score"] = pred_val
    test_pred = test_meta[["symbol", "trade_date", "year", *target_columns]].copy()
    test_pred["score"] = pred_test
    outputs = {
        "split_metrics_csv": _write_csv(output_dir / "split_metrics.csv", split_metrics),
        "topk_metrics_csv": _write_csv(output_dir / "topk_metrics.csv", topk),
        "daily_ic_csv": _write_csv(output_dir / "daily_rank_ic.csv", daily_ic),
        "feature_importance_csv": _write_csv(output_dir / "feature_importance.csv", feature_importance),
        "validation_predictions_csv": _write_csv(pred_dir / "validation_predictions.csv", val_pred),
        "test_predictions_csv": _write_csv(pred_dir / "test_predictions.csv", test_pred),
        "feature_shards": [str(path.resolve()) for path in shard_paths],
    }
    chart_paths = _plot_outputs(output_dir, topk, feature_importance, forward_days=config.forward_days)
    outputs["charts"] = chart_paths
    summary: dict[str, Any] = {
        "artifact_type": "qdp_v2_path_value_predictability_baseline",
        "generated_at": _now(),
        "qdp_root": str(config.qdp_root.resolve()),
        "atlas_dir": str(config.atlas_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "train_years": list(config.train_years),
        "validation_years": list(config.validation_years),
        "test_years": list(config.test_years),
        "target": target_column,
        "forward_days": int(config.forward_days),
        "past_close_days": int(config.past_close_days),
        "feature_count": int(len(feature_cols)),
        "features": feature_cols,
        "model": {
            "type": "LightGBM LGBMRegressor",
            "best_iteration": int(model.best_iteration_ or config.n_estimators),
            "n_estimators": int(config.n_estimators),
            "learning_rate": float(config.learning_rate),
            "num_leaves": int(config.num_leaves),
            "n_jobs": int(config.n_jobs),
        },
        "split_metrics": split_metrics.to_dict("records"),
        "outputs": outputs,
    }
    report_path = _build_report(
        output_dir,
        summary=summary,
        split_metrics=split_metrics,
        topk=topk,
        feature_importance=feature_importance,
        chart_paths=chart_paths,
    )
    summary["outputs"]["report_md"] = report_path
    summary_path = _write_json(output_dir / "path_value_predictability_summary.json", summary)
    summary["outputs"]["summary_json"] = summary_path
    _write_json(progress_path, {"status": "completed", "summary_json": summary_path, "report_md": report_path, "updated_at": _now()})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a first path-value predictability baseline from QDP v2 path atlas features.")
    parser.add_argument("--qdp-root", default=str(DEFAULT_QDP_ROOT))
    parser.add_argument("--atlas-dir", default=str(DEFAULT_ATLAS_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-tag", default="qdp_v2_path_value_predictability_baseline")
    parser.add_argument("--train-years", default="2012-2023")
    parser.add_argument("--validation-years", default="2024")
    parser.add_argument("--test-years", default="2025")
    parser.add_argument("--top-k", default=",".join(str(v) for v in DEFAULT_TOP_K))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--forward-days", type=int, default=DEFAULT_FORWARD_DAYS)
    parser.add_argument("--past-close-days", type=int, default=PAST_CLOSE_DAYS)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = build_path_value_predictability_baseline(
        BaselineConfig(
            qdp_root=Path(args.qdp_root),
            atlas_dir=Path(args.atlas_dir),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            train_years=_parse_years(str(args.train_years), default=DEFAULT_TRAIN_YEARS),
            validation_years=_parse_years(str(args.validation_years), default=DEFAULT_VALIDATION_YEARS),
            test_years=_parse_years(str(args.test_years), default=DEFAULT_TEST_YEARS),
            top_k=_parse_int_list(str(args.top_k), default=DEFAULT_TOP_K),
            seed=int(args.seed),
            forward_days=int(args.forward_days),
            past_close_days=int(args.past_close_days),
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            num_leaves=int(args.num_leaves),
            n_jobs=int(args.n_jobs),
        )
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(summary.get("outputs", {}).get("report_md", ""))
    return summary


if __name__ == "__main__":
    main()
