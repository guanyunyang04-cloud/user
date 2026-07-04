from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler

from daily_research.path_policy.qdp_v2_sequence_path_pack import (
    DEFAULT_FORWARD_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    PATH_OHLC_FIELDS,
    PATH_SUMMARY_COLUMNS,
    _json_default,
    _write_json,
)


DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/sequence_path_training")
DEFAULT_TOP_K = (20, 50, 100)
DEFAULT_SEED = 7
VALUE_COLUMN = "path_trade_value_20d"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_int_list(raw: str | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    return tuple(sorted(set(values))) or default


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(value: str) -> torch.device:
    raw = str(value or "auto").strip().lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if raw not in {"cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    return torch.device(raw)


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    path = Path(str(meta.get("path", "") or ""))
    shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
    if not path.exists():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


class SequencePathPackDataset(Dataset):
    def __init__(self, manifest: Mapping[str, Any], *, split: str, max_samples: int = 0) -> None:
        self.manifest = dict(manifest)
        self.lookback_days = int(self.manifest.get("lookback_days", DEFAULT_LOOKBACK_DAYS) or DEFAULT_LOOKBACK_DAYS)
        self.forward_days = int(self.manifest.get("forward_days", DEFAULT_FORWARD_DAYS) or DEFAULT_FORWARD_DAYS)
        sample_index = pd.read_parquet(str(self.manifest["sample_index_path"]))
        self.sample_index = sample_index[sample_index["split"].astype(str).eq(str(split))].reset_index(drop=True)
        if int(max_samples) > 0 and len(self.sample_index) > int(max_samples):
            self.sample_index = self.sample_index.head(int(max_samples)).reset_index(drop=True)
        self.split = str(split)
        channels = dict(self.manifest.get("feature_channels", {}) or {})
        self.channel_order = ["daily_raw", "daily_state", "intraday_summary", "limit_structure"]
        self.feature_arrays = {name: _open_memmap(channels[name], dtype="float32") for name in self.channel_order}
        self.feature_columns = {name: list(channels[name].get("columns", []) or []) for name in self.channel_order}
        self.normalization = dict(self.manifest.get("normalization", {}) or {})
        labels = dict(self.manifest.get("label_arrays", {}) or {})
        self.future_path = _open_memmap(labels["future_ohlc_path"], dtype="float32")
        self.path_summary = _open_memmap(labels["path_summary"], dtype="float32")
        self.path_summary_columns = list(labels["path_summary"].get("columns", []) or PATH_SUMMARY_COLUMNS)
        self.value_index = self.path_summary_columns.index(VALUE_COLUMN)
        self.input_dim = int(sum(len(self.feature_columns[name]) for name in self.channel_order))

    def __len__(self) -> int:
        return int(len(self.sample_index))

    def _normalize(self, name: str, values: np.ndarray) -> np.ndarray:
        stats = dict(self.normalization.get(name, {}) or {})
        mean = np.asarray(stats.get("mean", [0.0] * values.shape[-1]), dtype=np.float32)
        std = np.asarray(stats.get("std", [1.0] * values.shape[-1]), dtype=np.float32)
        out = (values.astype(np.float32, copy=False) - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1.0e-6)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.sample_index.iloc[int(idx)]
        date_idx = int(row["date_idx"])
        symbol_idx = int(row["symbol_idx"])
        start = date_idx - self.lookback_days + 1
        end = date_idx + 1
        parts = [
            self._normalize(name, np.asarray(self.feature_arrays[name][start:end, symbol_idx, :], dtype=np.float32))
            for name in self.channel_order
        ]
        x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        y_path = np.asarray(self.future_path[date_idx, symbol_idx, :, :], dtype=np.float32).copy()
        y_summary = np.asarray(self.path_summary[date_idx, symbol_idx, :], dtype=np.float32).copy()
        return {
            "x": torch.from_numpy(x),
            "y_path": torch.from_numpy(y_path),
            "y_summary": torch.from_numpy(y_summary),
            "date_idx": int(date_idx),
            "symbol_idx": int(symbol_idx),
            "trade_date": str(row["trade_date"]),
            "symbol": str(row["symbol"]),
        }


class DateGroupedBatchSampler(BatchSampler):
    def __init__(self, sample_index: pd.DataFrame, *, batch_size: int, shuffle: bool, seed: int) -> None:
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[int, list[int]] = {}
        for idx, date_idx in enumerate(sample_index["date_idx"].astype(int).to_numpy()):
            groups.setdefault(int(date_idx), []).append(int(idx))
        self.groups = groups
        self.date_indices = list(groups.keys())
        self._length = sum((len(items) + self.batch_size - 1) // self.batch_size for items in self.groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        dates = list(self.date_indices)
        if self.shuffle:
            rng.shuffle(dates)
        for date_idx in dates:
            items = list(self.groups[date_idx])
            if self.shuffle:
                rng.shuffle(items)
            for start in range(0, len(items), self.batch_size):
                yield items[start : start + self.batch_size]

    def __len__(self) -> int:
        return int(self._length)


class SequencePathModel(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, layers: int, forward_days: int, summary_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=max(int(layers), 1),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.path_head = nn.Linear(hidden_dim, int(forward_days) * 4)
        self.summary_head = nn.Linear(hidden_dim, int(summary_dim))
        self.score_head = nn.Linear(hidden_dim, 1)
        self.forward_days = int(forward_days)
        self.summary_dim = int(summary_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.input_norm(x)
        z = F.gelu(self.proj(z))
        encoded, _ = self.encoder(z)
        pooled = self.dropout(encoded[:, -1, :])
        return {
            "future_path": self.path_head(pooled).view(-1, self.forward_days, 4),
            "path_summary": self.summary_head(pooled).view(-1, self.summary_dim),
            "score": self.score_head(pooled).squeeze(-1),
        }


def _finite_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask], reduction="mean")


def _rank_loss_by_date(score: torch.Tensor, target: torch.Tensor, date_idx: torch.Tensor, *, max_per_side: int = 64) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for date in torch.unique(date_idx):
        mask = (date_idx == date) & torch.isfinite(target)
        if int(mask.sum().item()) < 4:
            continue
        s = score[mask]
        y = target[mask]
        order = torch.argsort(y)
        n = int(order.numel())
        side = min(int(max_per_side), n // 2)
        low = order[:side]
        high = order[-side:]
        diff = s[high].view(-1, 1) - s[low].view(1, -1)
        losses.append(F.softplus(-diff).mean())
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def _compute_loss(
    outputs: Mapping[str, torch.Tensor],
    y_path: torch.Tensor,
    y_summary: torch.Tensor,
    date_idx: torch.Tensor,
    *,
    value_index: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    path_loss = _finite_smooth_l1(outputs["future_path"], y_path)
    summary_loss = _finite_smooth_l1(outputs["path_summary"], y_summary)
    value_target = y_summary[:, int(value_index)]
    value_loss = _finite_smooth_l1(outputs["score"], value_target)
    rank_loss = _rank_loss_by_date(outputs["score"], value_target, date_idx)
    total = 0.35 * path_loss + 0.35 * summary_loss + 0.15 * value_loss + 0.15 * rank_loss
    return total, {
        "loss": float(total.detach().cpu().item()),
        "path_loss": float(path_loss.detach().cpu().item()),
        "summary_loss": float(summary_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "rank_loss": float(rank_loss.detach().cpu().item()),
    }


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=device.type == "cuda")
    y_path = batch["y_path"].to(device, non_blocking=device.type == "cuda")
    y_summary = batch["y_summary"].to(device, non_blocking=device.type == "cuda")
    date_idx = batch["date_idx"].to(device, non_blocking=device.type == "cuda")
    return x, y_path, y_summary, date_idx


def _daily_spearman(frame: pd.DataFrame, *, score_col: str, target_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        if len(group) < 5:
            continue
        corr = group[[score_col, target_col]].corr(method="spearman").iloc[0, 1]
        rows.append({"trade_date": str(trade_date), "rank_ic": float(corr) if pd.notna(corr) else np.nan, "count": int(len(group))})
    return pd.DataFrame(rows)


def _topk_metrics(frame: pd.DataFrame, *, top_k_values: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "future_max_return_20d",
        "future_min_return_20d",
        "future_final_return_20d",
        "drawdown_after_peak_20d",
        "path_trade_value_20d",
    ]
    for top_k in top_k_values:
        daily_rows: list[dict[str, Any]] = []
        for trade_date, group in frame.groupby("trade_date", sort=True):
            group = group.dropna(subset=["score", VALUE_COLUMN])
            if group.empty:
                continue
            top = group.sort_values("score", ascending=False, kind="mergesort").head(int(top_k))
            row: dict[str, Any] = {"trade_date": str(trade_date), "top_k": int(top_k)}
            for col in metric_cols:
                universe_mean = pd.to_numeric(group[col], errors="coerce").mean()
                selected_mean = pd.to_numeric(top[col], errors="coerce").mean()
                row[f"selected_{col}"] = float(selected_mean)
                row[f"universe_{col}"] = float(universe_mean)
                row[f"alpha_{col}"] = float(selected_mean - universe_mean)
            row["selected_hit_5pct_rate"] = float((pd.to_numeric(top["future_max_return_20d"], errors="coerce") >= 0.05).mean())
            row["selected_hit_10pct_rate"] = float((pd.to_numeric(top["future_max_return_20d"], errors="coerce") >= 0.10).mean())
            row["selected_hit_20pct_rate"] = float((pd.to_numeric(top["future_max_return_20d"], errors="coerce") >= 0.20).mean())
            row["selected_loss_3pct_rate"] = float((pd.to_numeric(top["future_min_return_20d"], errors="coerce") <= -0.03).mean())
            row["selected_loss_5pct_rate"] = float((pd.to_numeric(top["future_min_return_20d"], errors="coerce") <= -0.05).mean())
            row["selected_loss_10pct_rate"] = float((pd.to_numeric(top["future_min_return_20d"], errors="coerce") <= -0.10).mean())
            row["selected_peak_day_mean"] = float(pd.to_numeric(top["future_peak_day_20d"], errors="coerce").mean())
            daily_rows.append(row)
        daily = pd.DataFrame(daily_rows)
        if daily.empty:
            continue
        out: dict[str, Any] = {"top_k": int(top_k), "day_count": int(len(daily))}
        for col in [c for c in daily.columns if c not in {"trade_date", "top_k"}]:
            out[col] = float(pd.to_numeric(daily[col], errors="coerce").mean())
        rows.append(out)
    return pd.DataFrame(rows)


def _find_latest_baseline_summary() -> dict[str, Any]:
    root = Path("daily_research/output/path_policy/path_value_predictability")
    paths = sorted(root.glob("qdp_v2_path20_value_predictability_lookback100_full_*/path_value_predictability_summary.json"))
    if not paths:
        return {}
    return json.loads(paths[-1].read_text(encoding="utf-8"))


@torch.no_grad()
def _predict_split(
    *,
    model: nn.Module,
    dataset: SequencePathPackDataset,
    device: torch.device,
    output_dir: Path,
    split: str,
    batch_size: int,
    amp_enabled: bool,
    top_k: tuple[int, ...],
    write_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=max(int(batch_size), 1), shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    pred_dir = output_dir / "predictions"
    if write_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{split}_predictions.csv"
    if write_predictions and pred_path.exists():
        pred_path.unlink()
    summary_rows: list[pd.DataFrame] = []
    first_write = True
    model.eval()
    for batch in loader:
        x, y_path, y_summary, _date_idx = _batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(x)
        pred_path_np = out["future_path"].detach().float().cpu().numpy()
        pred_summary_np = out["path_summary"].detach().float().cpu().numpy()
        score_np = out["score"].detach().float().cpu().numpy()
        true_path_np = y_path.detach().float().cpu().numpy()
        true_summary_np = y_summary.detach().float().cpu().numpy()
        rows: dict[str, Any] = {
            "trade_date": list(batch["trade_date"]),
            "symbol": list(batch["symbol"]),
            "score": score_np,
        }
        for idx, col in enumerate(dataset.path_summary_columns):
            rows[f"true_{col}"] = true_summary_np[:, idx]
            rows[f"pred_{col}"] = pred_summary_np[:, idx]
        chunk = pd.DataFrame(rows)
        if write_predictions:
            path_rows: dict[str, Any] = {}
            for day in range(dataset.forward_days):
                for field_idx, field in enumerate(PATH_OHLC_FIELDS):
                    path_rows[f"true_{field}_ret_d{day + 1}"] = true_path_np[:, day, field_idx]
                    path_rows[f"pred_{field}_ret_d{day + 1}"] = pred_path_np[:, day, field_idx]
            chunk = pd.concat([chunk, pd.DataFrame(path_rows)], axis=1, copy=False)
            chunk.to_csv(pred_path, index=False, mode="w" if first_write else "a", header=first_write, encoding="utf-8-sig")
            first_write = False
        metric_frame = chunk[["trade_date", "symbol", "score", *[f"true_{c}" for c in dataset.path_summary_columns]]].copy()
        metric_frame = metric_frame.rename(columns={f"true_{c}": c for c in dataset.path_summary_columns})
        summary_rows.append(metric_frame)
        del x, y_path, y_summary, out, pred_path_np, pred_summary_np, score_np, true_path_np, true_summary_np, chunk
        gc.collect()
    frame = pd.concat(summary_rows, ignore_index=True)
    ic = _daily_spearman(frame, score_col="score", target_col=VALUE_COLUMN)
    topk = _topk_metrics(frame, top_k_values=top_k)
    metrics = {
        "split": split,
        "row_count": int(len(frame)),
        "date_count": int(frame["trade_date"].nunique()),
        "rank_ic_mean": float(ic["rank_ic"].mean()) if not ic.empty else np.nan,
        "rank_ic_median": float(ic["rank_ic"].median()) if not ic.empty else np.nan,
        "rank_ic_positive_day_rate": float((ic["rank_ic"] > 0).mean()) if not ic.empty else np.nan,
        "target_mean": float(frame[VALUE_COLUMN].mean()),
        "prediction_mean": float(frame["score"].mean()),
        "prediction_csv": str(pred_path.resolve()) if write_predictions else "",
    }
    return ic, topk, metrics


def _write_report(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    split_metrics: pd.DataFrame,
    topk: pd.DataFrame,
    baseline_summary: Mapping[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# QDP v2 Sequence Path Model")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "This model reads past 100-day multi-channel sequences from QDP v2 and predicts future 20-day OHLC paths, path summaries, and a path value score."
    )
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
    lines.append("## Top-K")
    lines.append("")
    if not topk.empty:
        for row in topk.sort_values(["split", "top_k"]).to_dict("records"):
            lines.append(
                f"- {row['split']} top{int(row['top_k'])}: "
                f"path_value_alpha={row['alpha_path_trade_value_20d'] * 100:.2f}%, "
                f"max_alpha={row['alpha_future_max_return_20d'] * 100:.2f}%, "
                f"final_alpha={row['alpha_future_final_return_20d'] * 100:.2f}%, "
                f"hit10={row['selected_hit_10pct_rate']:.2%}, "
                f"loss5={row['selected_loss_5pct_rate']:.2%}"
            )
    if baseline_summary:
        lines.append("")
        lines.append("## Baseline")
        lines.append("")
        lines.append(f"- 191-feature baseline summary: `{baseline_summary.get('output_dir', '')}`")
        for item in list(baseline_summary.get("split_metrics", []) or []):
            if str(item.get("split", "")) in {"validation", "test"}:
                lines.append(
                    f"- baseline {item.get('split')}: rank_ic_mean={float(item.get('rank_ic_mean', float('nan'))):.4f}, "
                    f"positive_day_rate={float(item.get('rank_ic_positive_day_rate', float('nan'))):.2%}"
                )
    lines.append("")
    lines.append("## Boundaries")
    lines.append("")
    lines.append("- No symbol id or symbol embedding is used.")
    lines.append("- Path types are explanation labels; the model trains on numeric paths and path value.")
    lines.append("- This is a first sequence model; failed improvement over baseline should be diagnosed, not hidden by test-set tuning.")
    path = output_dir / "sequence_path_training_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


@dataclass(frozen=True)
class TrainConfig:
    pack_manifest: Path
    output_root: Path
    run_tag: str
    epochs: int
    batch_size: int
    hidden_dim: int
    layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    device: str
    amp: bool
    seed: int
    top_k: tuple[int, ...]
    max_samples_per_split: int


def train_sequence_path_model(config: TrainConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    manifest = json.loads(Path(config.pack_manifest).read_text(encoding="utf-8"))
    device = _resolve_device(config.device)
    amp_enabled = bool(config.amp and device.type == "cuda")
    output_dir = config.output_root / f"{config.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "loading_datasets", "updated_at": _now()})
    train_ds = SequencePathPackDataset(manifest, split="train", max_samples=int(config.max_samples_per_split))
    val_ds = SequencePathPackDataset(manifest, split="validation", max_samples=int(config.max_samples_per_split))
    test_ds = SequencePathPackDataset(manifest, split="test", max_samples=int(config.max_samples_per_split))
    train_sampler = DateGroupedBatchSampler(train_ds.sample_index, batch_size=int(config.batch_size), shuffle=True, seed=int(config.seed))
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0, pin_memory=device.type == "cuda")
    model = SequencePathModel(
        input_dim=train_ds.input_dim,
        hidden_dim=int(config.hidden_dim),
        layers=int(config.layers),
        forward_days=train_ds.forward_days,
        summary_dim=len(train_ds.path_summary_columns),
        dropout=float(config.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val_ic = -1e9
    best_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config.epochs) + 1):
        _write_json(progress_path, {"status": "training", "epoch": epoch, "updated_at": _now()})
        model.train()
        loss_totals: dict[str, float] = {"loss": 0.0, "path_loss": 0.0, "summary_loss": 0.0, "value_loss": 0.0, "rank_loss": 0.0}
        batch_count = 0
        sample_count = 0
        for batch in train_loader:
            x, y_path, y_summary, date_idx = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x)
                loss, parts = _compute_loss(out, y_path, y_summary, date_idx, value_index=train_ds.value_index)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                loss_totals[key] += float(value)
            batch_count += 1
            sample_count += int(x.shape[0])
            del x, y_path, y_summary, date_idx, out, loss
        train_row = {
            "epoch": int(epoch),
            "train_sample_count": int(sample_count),
            **{key: float(value / max(batch_count, 1)) for key, value in loss_totals.items()},
        }
        val_ic, val_topk, val_metrics = _predict_split(
            model=model,
            dataset=val_ds,
            device=device,
            output_dir=output_dir / f"epoch_{epoch:03d}",
            split="validation",
            batch_size=int(config.batch_size),
            amp_enabled=amp_enabled,
            top_k=config.top_k,
            write_predictions=False,
        )
        train_row["validation_rank_ic_mean"] = float(val_metrics["rank_ic_mean"])
        history.append(train_row)
        if float(val_metrics["rank_ic_mean"]) > best_val_ic:
            best_val_ic = float(val_metrics["rank_ic_mean"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "input_dim": train_ds.input_dim,
                    "summary_columns": train_ds.path_summary_columns,
                    "feature_channels": manifest.get("feature_channels", {}),
                    "best_epoch": int(epoch),
                    "best_validation_rank_ic_mean": best_val_ic,
                },
                best_path,
            )
        del val_ic, val_topk
        gc.collect()
    if best_path.exists():
        payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
    _write_json(progress_path, {"status": "final_evaluation", "updated_at": _now()})
    val_ic, val_topk, val_metrics = _predict_split(
        model=model,
        dataset=val_ds,
        device=device,
        output_dir=output_dir,
        split="validation",
        batch_size=int(config.batch_size),
        amp_enabled=amp_enabled,
        top_k=config.top_k,
        write_predictions=True,
    )
    test_ic, test_topk, test_metrics = _predict_split(
        model=model,
        dataset=test_ds,
        device=device,
        output_dir=output_dir,
        split="test",
        batch_size=int(config.batch_size),
        amp_enabled=amp_enabled,
        top_k=config.top_k,
        write_predictions=True,
    )
    split_metrics = pd.DataFrame([val_metrics, test_metrics])
    topk = pd.concat([val_topk.assign(split="validation"), test_topk.assign(split="test")], ignore_index=True)
    daily_ic = pd.concat([val_ic.assign(split="validation"), test_ic.assign(split="test")], ignore_index=True)
    split_metrics_path = output_dir / "split_metrics.csv"
    topk_path = output_dir / "topk_metrics.csv"
    daily_ic_path = output_dir / "daily_rank_ic.csv"
    history_path = output_dir / "training_history.csv"
    split_metrics.to_csv(split_metrics_path, index=False, encoding="utf-8-sig")
    topk.to_csv(topk_path, index=False, encoding="utf-8-sig")
    daily_ic.to_csv(daily_ic_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    baseline_summary = _find_latest_baseline_summary()
    summary = {
        "artifact_type": "qdp_v2_sequence_path_training",
        "generated_at": _now(),
        "pack_manifest": str(Path(config.pack_manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "max_samples_per_split": int(config.max_samples_per_split),
        "model": {
            "type": "SequencePathModel_GRU",
            "input_dim": int(train_ds.input_dim),
            "hidden_dim": int(config.hidden_dim),
            "layers": int(config.layers),
            "dropout": float(config.dropout),
        },
        "best_checkpoint": str(best_path.resolve()),
        "history": history,
        "split_metrics": split_metrics.to_dict("records"),
        "outputs": {
            "split_metrics_csv": str(split_metrics_path.resolve()),
            "topk_metrics_csv": str(topk_path.resolve()),
            "daily_rank_ic_csv": str(daily_ic_path.resolve()),
            "training_history_csv": str(history_path.resolve()),
            "validation_predictions_csv": str((output_dir / "predictions" / "validation_predictions.csv").resolve()),
            "test_predictions_csv": str((output_dir / "predictions" / "test_predictions.csv").resolve()),
        },
        "baseline_191_feature_summary": baseline_summary.get("output_dir", ""),
    }
    report_path = _write_report(output_dir, summary=summary, split_metrics=split_metrics, topk=topk, baseline_summary=baseline_summary)
    summary["outputs"]["report_md"] = report_path
    summary_path = output_dir / "sequence_path_training_summary.json"
    _write_json(summary_path, summary)
    _write_json(progress_path, {"status": "completed", "summary_json": str(summary_path.resolve()), "updated_at": _now()})
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train QDP v2 sequence path models.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--pack-manifest", type=Path, required=True)
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--run-tag", default="qdp_v2_seq100_path20_model_v1")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.10)
    train.add_argument("--learning-rate", type=float, default=1.0e-3)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    train.add_argument("--amp", dest="amp", action="store_true", default=True)
    train.add_argument("--no-amp", dest="amp", action="store_false")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--top-k", default="20,50,100")
    train.add_argument("--max-samples-per-split", type=int, default=0)
    train.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = TrainConfig(
        pack_manifest=Path(args.pack_manifest),
        output_root=Path(args.output_root),
        run_tag=str(args.run_tag),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        layers=int(args.layers),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        device=str(args.device),
        amp=bool(args.amp),
        seed=int(args.seed),
        top_k=_parse_int_list(args.top_k, default=DEFAULT_TOP_K),
        max_samples_per_split=int(args.max_samples_per_split),
    )
    result = train_sequence_path_model(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) if bool(args.json) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
