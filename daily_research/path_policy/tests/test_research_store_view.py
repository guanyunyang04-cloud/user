from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import research_store_view as view


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_float(path: Path, shape: tuple[int, ...], values: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(path, dtype="float32", mode="w+", shape=shape)
    arr[:] = values if values is not None else 0.0
    arr.flush()


def _write_bool(path: Path, shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(path, dtype="bool", mode="w+", shape=shape)
    arr[:] = True
    arr.flush()


def _manifest(
    root: Path,
    *,
    name: str,
    symbols: list[str],
    forward_days: int,
    label_dim: int,
    panel_values: dict[str, np.ndarray],
    future_path: np.ndarray | None = None,
    future_ohlcva: np.ndarray | None = None,
) -> dict:
    pack = root / name
    dates = [f"2025-01-0{idx + 1}" for idx in range(4)]
    feature_channels = {}
    for channel, values in panel_values.items():
        path = pack / "panels" / f"{channel}.float32.dat"
        _write_float(path, values.shape, values)
        feature_channels[channel] = {"path": str(path.resolve()), "shape": list(values.shape), "columns": [f"{channel}_x"]}
    masks = {}
    for mask in ["input_valid", "entry_buyable", "label_valid"]:
        path = pack / "masks" / f"{mask}.bool.dat"
        _write_bool(path, (4, len(symbols)))
        masks[mask] = {"path": str(path.resolve()), "shape": [4, len(symbols)]}
    labels = {}
    if future_path is not None:
        path = pack / "labels/future_ohlc_path.float32.dat"
        _write_float(path, future_path.shape, future_path)
        labels["future_ohlc_path"] = {
            "path": str(path.resolve()),
            "shape": list(future_path.shape),
            "fields": ["open", "high", "low", "close"],
            "price_anchor": "next_open",
        }
    if future_ohlcva is not None:
        path = pack / "labels/future_ohlcva_path.float32.dat"
        _write_float(path, future_ohlcva.shape, future_ohlcva)
        labels["future_ohlcva_path"] = {
            "path": str(path.resolve()),
            "shape": list(future_ohlcva.shape),
            "fields": ["open", "high", "low", "close", "volume", "amount"][:label_dim],
            "price_anchor": "next_open",
        }
    summary_path = pack / "labels/path_summary.float32.dat"
    _write_float(summary_path, (4, len(symbols), 9))
    labels["path_summary"] = {
        "path": str(summary_path.resolve()),
        "shape": [4, len(symbols), 9],
        "columns": [
            "future_max_return_60d",
            "future_min_return_60d",
            "future_final_return_60d",
            "future_peak_day_60d",
            "future_trough_day_60d",
            "drawdown_after_peak_60d",
            "time_above_zero_60d",
            "time_below_zero_60d",
            "path_trade_value_60d",
        ],
    }
    sample = pd.DataFrame(
        {
            "trade_date": [dates[1], dates[2]],
            "date_idx": [1, 2],
            "symbol": [symbols[0], symbols[-1]],
            "symbol_idx": [0, len(symbols) - 1],
            "split": ["train", "test"],
        }
    )
    sample_path = pack / "sample_index.parquet"
    sample.to_parquet(sample_path, index=False)
    manifest = {
        "artifact_type": "qdp_v2_sequence_path_pack",
        "lookback_days": 1,
        "forward_days": forward_days,
        "date_values": dates,
        "symbol_values": symbols,
        "date_count": 4,
        "symbol_count": len(symbols),
        "sample_count": len(sample),
        "sample_count_by_split": {"train": 1, "test": 1},
        "feature_channels": feature_channels,
        "label_arrays": labels,
        "masks": masks,
        "sample_index_path": str(sample_path.resolve()),
        "normalization": {channel: {"mean": [0.0], "std": [1.0]} for channel in panel_values},
    }
    _write_json(pack / "manifest.json", manifest)
    return manifest


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(view, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


def test_build_and_verify_unified_store_views_remaps_path20(workspace: Path) -> None:
    root = workspace / "daily_research/data/research_store/sequence_pack"
    channels = ["daily_raw", "daily_state", "intraday_summary", "limit_structure"]
    canonical_symbols = ["AAA", "BBB", "CCC"]
    path20_symbols = ["AAA", "CCC"]
    canonical_panels = {}
    path20_panels = {}
    for channel in channels:
        values = np.zeros((4, 3, 1), dtype=np.float32)
        for date_idx in range(4):
            for symbol_idx in range(3):
                values[date_idx, symbol_idx, 0] = date_idx * 10 + symbol_idx
        canonical_panels[channel] = values
        path20_panels[channel] = values[:, [0, 2], :]
    path60_label = np.zeros((4, 3, 60, 4), dtype=np.float32)
    for date_idx in range(4):
        for symbol_idx in range(3):
            for day in range(60):
                path60_label[date_idx, symbol_idx, day, :] = date_idx * 1000 + symbol_idx * 100 + day * 10 + np.arange(4)
    path20_label = path60_label[:, [0, 2], :20, :]
    ohlcva_label = np.concatenate([path60_label, np.zeros((4, 3, 60, 2), dtype=np.float32)], axis=3)

    _manifest(
        root,
        name="qdp_v2_seq100_path60_full",
        symbols=canonical_symbols,
        forward_days=60,
        label_dim=4,
        panel_values=canonical_panels,
        future_path=path60_label,
    )
    _manifest(
        root,
        name="qdp_v2_seq100_ohlcva_path60_full",
        symbols=canonical_symbols,
        forward_days=60,
        label_dim=6,
        panel_values=canonical_panels,
        future_ohlcva=ohlcva_label,
    )
    _manifest(
        root,
        name="qdp_v2_seq100_path60_todayclose_full",
        symbols=canonical_symbols,
        forward_days=60,
        label_dim=6,
        panel_values=canonical_panels,
        future_ohlcva=ohlcva_label,
    )
    _manifest(
        root,
        name="qdp_v2_seq100_path20_full",
        symbols=path20_symbols,
        forward_days=20,
        label_dim=4,
        panel_values=path20_panels,
        future_path=path20_label,
    )

    result = view.build_unified_store_from_legacy_packs(store_root=workspace / "daily_research/data/research_store", legacy_root=root)
    verification = view.verify_unified_store_views(
        store_root=workspace / "daily_research/data/research_store",
        legacy_root=root,
        max_checks=2,
    )

    assert result["status"] == "built"
    assert verification["status"] == "ok"
    path20_view = json.loads(
        (workspace / "daily_research/data/research_store/views/seq100_path20_nextopen_ohlc_from_path60.json").read_text(
            encoding="utf-8"
        )
    )
    remapped = pd.read_parquet(path20_view["sample_index_path"])
    assert remapped["source_symbol_idx"].tolist() == [0, 1]
    assert remapped["label_symbol_idx"].tolist() == [0, 1]
    assert remapped["symbol_idx"].tolist() == [0, 2]

    for pack_name in view.LEGACY_PACK_NAMES:
        shutil.rmtree(root / pack_name)
    post_delete = view.verify_unified_store_views(
        store_root=workspace / "daily_research/data/research_store",
        legacy_root=root,
        max_checks=2,
    )
    assert post_delete["status"] == "ok"
    assert post_delete["path20_label_source"] == "moved_label_store_no_legacy_manifest"
