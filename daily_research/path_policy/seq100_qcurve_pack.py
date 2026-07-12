from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from daily_research.path_policy.seq100_qcurve import (
    MA_STATE_FEATURES,
    QCURVE_ENTRY_HORIZONS,
    QCURVE_HOLD_HORIZONS,
    QCurveCostContract,
    build_open_execution_masks,
    build_qcurve_targets,
)


DEFAULT_SOURCE_MANIFEST = Path(
    "daily_research/data/research_store/sequence_pack/"
    "qdp_v2_seq100_path60_todayclose_candidate_complete_2012_2025_v7/manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("daily_research/data/research_store/sequence_pack")
DEFAULT_RUN_TAG = "qdp_v2_seq100_dynamic_qcurve_candidate_complete_2012_2025_v8"
DEFAULT_CONTRACT = Path(
    "daily_research/brain/references/seq100_dynamic_qcurve_development_contract_20260712.json"
)
QCURVE_TARGET_SEMANTICS_VERSION = "per_horizon_exit_tail20_v1"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_binding(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    raw = resolved.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    semantic = dict(payload)
    semantic.pop("contract_sha256", None)
    digest = hashlib.sha256(
        json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    declared = str(payload.get("contract_sha256", "") or "")
    if declared and declared != digest:
        raise ValueError("Q-curve contract semantic digest mismatch")
    return {
        "path": str(resolved),
        "contract_id": str(payload["contract_id"]),
        "contract_sha256": digest,
        "contract_file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    return np.memmap(
        Path(str(meta["path"])),
        dtype=dtype,
        mode="r",
        shape=tuple(int(value) for value in meta["shape"]),
    )


def _new_memmap(path: Path, shape: Sequence[int], *, dtype: str) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=dtype, mode="w+", shape=tuple(int(value) for value in shape))


def _hardlink(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        source_stat = source.stat()
        target_stat = target.stat()
        if (
            source_stat.st_size == target_stat.st_size
            and source_stat.st_mtime_ns == target_stat.st_mtime_ns
        ):
            return target
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
        if target.stat().st_size != source.stat().st_size or _sha256_file(target) != _sha256_file(source):
            target.unlink(missing_ok=True)
            raise OSError(f"verified index copy failed: {source} -> {target}")
    return target


def _memmap_metadata(path: Path, shape: Sequence[int], *, dtype: str, layout: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "shape": [int(value) for value in shape],
        "dtype": str(dtype),
    }
    if layout:
        result["layout"] = str(layout)
    return result


def _stage_marker(target: Path, stage: str) -> Path:
    return target / "stages" / f"{stage}.json"


def _mark_stage(target: Path, stage: str, payload: Mapping[str, Any]) -> None:
    _write_json(
        _stage_marker(target, stage),
        {"stage": str(stage), "status": "completed", "completed_at": _now(), **dict(payload)},
    )


def _stage_is_complete(target: Path, stage: str, required_paths: Sequence[Path]) -> bool:
    marker = _stage_marker(target, stage)
    if not marker.is_file() or any(not path.is_file() for path in required_paths):
        return False


def _trim_process_working_set() -> None:
    gc.collect()
    if os.name != "nt":
        return
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetProcessWorkingSetSize(handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    except (AttributeError, OSError):
        return


def _existing_or_new_memmap(
    path: Path,
    shape: Sequence[int],
    *,
    dtype: str,
    resume: bool,
) -> np.memmap:
    expected = int(np.prod(tuple(int(value) for value in shape), dtype=np.int64)) * np.dtype(dtype).itemsize
    if bool(resume) and path.is_file() and path.stat().st_size == expected:
        return np.memmap(path, dtype=dtype, mode="r+", shape=tuple(int(value) for value in shape))
    return _new_memmap(path, shape, dtype=dtype)


def _infer_qcurve_resume_row(dependency: np.memmap, *, checkpoint_rows: int) -> int:
    first_zero = int(dependency.shape[0])
    scan = 1_000_000
    for start in range(0, int(dependency.shape[0]), scan):
        stop = min(start + scan, int(dependency.shape[0]))
        zero = np.flatnonzero(np.asarray(dependency[start:stop], dtype=np.int32) == 0)
        if zero.size:
            first_zero = start + int(zero[0])
            break
    if first_zero >= int(dependency.shape[0]):
        return int(dependency.shape[0])
    return max(0, first_zero // int(checkpoint_rows) * int(checkpoint_rows))
    try:
        return str(_read_json(marker).get("status", "")) == "completed"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = np.full(source.shape, np.nan, dtype=np.float64)
    state = np.full(source.shape[1], np.nan, dtype=np.float64)
    alpha = 2.0 / (float(span) + 1.0)
    for date_idx in range(source.shape[0]):
        current = source[date_idx]
        finite = np.isfinite(current)
        initialize = finite & (~np.isfinite(state))
        update = finite & np.isfinite(state)
        state[initialize] = current[initialize]
        state[update] = alpha * current[update] + (1.0 - alpha) * state[update]
        output[date_idx] = state
    return output


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.asarray(numerator, dtype=np.float64) / np.asarray(denominator, dtype=np.float64) - 1.0
    value[~np.isfinite(value)] = np.nan
    return value


def _lag_ratio(values: np.ndarray, lag: int) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    output[lag:] = _ratio(values[lag:], values[:-lag]) / float(lag)
    return output


def _rolling_vwap(amount: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    amount_values = np.nan_to_num(np.asarray(amount, dtype=np.float64), nan=0.0)
    volume_values = np.nan_to_num(np.asarray(volume, dtype=np.float64), nan=0.0)
    amount_cum = np.vstack([np.zeros((1, amount_values.shape[1])), np.cumsum(amount_values, axis=0)])
    volume_cum = np.vstack([np.zeros((1, volume_values.shape[1])), np.cumsum(volume_values, axis=0)])
    starts = np.maximum(np.arange(amount_values.shape[0]) + 1 - int(window), 0)
    ends = np.arange(amount_values.shape[0]) + 1
    amount_sum = amount_cum[ends] - amount_cum[starts]
    volume_sum = volume_cum[ends] - volume_cum[starts]
    with np.errstate(divide="ignore", invalid="ignore"):
        result = amount_sum / volume_sum
    result[~np.isfinite(result)] = np.nan
    return result


def _fit_stats(panel: np.memmap, date_mask: np.ndarray, *, symbol_chunk: int = 256) -> dict[str, list[float]]:
    count = np.zeros(panel.shape[2], dtype=np.float64)
    total = np.zeros(panel.shape[2], dtype=np.float64)
    total_sq = np.zeros(panel.shape[2], dtype=np.float64)
    date_indices = np.flatnonzero(np.asarray(date_mask, dtype=bool))
    for start in range(0, panel.shape[1], int(symbol_chunk)):
        stop = min(start + int(symbol_chunk), panel.shape[1])
        block = np.asarray(panel[date_indices, start:stop], dtype=np.float64)
        finite = np.isfinite(block)
        count += finite.sum(axis=(0, 1))
        safe = np.where(finite, block, 0.0)
        total += safe.sum(axis=(0, 1))
        total_sq += np.square(safe).sum(axis=(0, 1))
    mean = total / np.maximum(count, 1.0)
    variance = np.maximum(total_sq / np.maximum(count, 1.0) - np.square(mean), 1.0e-12)
    return {"mean": mean.astype(float).tolist(), "std": np.sqrt(variance).astype(float).tolist()}


def _build_ma_panel(
    source: Mapping[str, Any],
    output_path: Path,
    *,
    symbol_chunk: int = 128,
) -> np.memmap:
    daily_meta = dict(source["feature_channels"]["daily_raw"])
    daily = _open_memmap(daily_meta, dtype="float32")
    columns = list(daily_meta["columns"])
    raw_close = _open_memmap(source["execution_arrays"]["exit_close_raw"], dtype="float32")
    output = _new_memmap(output_path, (daily.shape[0], daily.shape[1], len(MA_STATE_FEATURES)), dtype="float32")
    output[:] = np.nan
    close_idx = columns.index("close")
    volume_idx = columns.index("volume")
    amount_idx = columns.index("amount")
    for start in range(0, daily.shape[1], int(symbol_chunk)):
        stop = min(start + int(symbol_chunk), daily.shape[1])
        close = np.asarray(daily[:, start:stop, close_idx], dtype=np.float64)
        volume = np.asarray(daily[:, start:stop, volume_idx], dtype=np.float64)
        amount = np.asarray(daily[:, start:stop, amount_idx], dtype=np.float64)
        ema_close = {span: _ema(close, span) for span in (5, 10, 20, 60)}
        ema_volume = {span: _ema(volume, span) for span in (5, 20, 60)}
        ema_amount = {span: _ema(amount, span) for span in (5, 20, 60)}
        vwap20 = _rolling_vwap(amount, volume, 20)
        raw_close_block = np.asarray(raw_close[:, start:stop], dtype=np.float64)
        features = [
            _ratio(close, ema_close[5]),
            _ratio(close, ema_close[10]),
            _ratio(close, ema_close[20]),
            _ratio(close, ema_close[60]),
            _ratio(ema_close[5], ema_close[20]),
            _ratio(ema_close[20], ema_close[60]),
            _lag_ratio(ema_close[5], 3),
            _lag_ratio(ema_close[20], 5),
            _lag_ratio(ema_close[60], 10),
            _ratio(ema_volume[5], ema_volume[20]),
            _ratio(ema_volume[20], ema_volume[60]),
            _ratio(ema_amount[5], ema_amount[20]),
            _ratio(ema_amount[20], ema_amount[60]),
            _ratio(raw_close_block, vwap20),
        ]
        output[:, start:stop, :] = np.stack(features, axis=2).astype(np.float32)
        output.flush()
    return output


def _write_candidate_arrays(
    candidate_path: Path,
    target: Path,
    *,
    candidate_count: int,
    open_buyable: np.memmap,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, int]], int]:
    import pyarrow.parquet as pq

    specs = {
        "date_idx": "int32",
        "symbol_idx": "int32",
        "year": "int16",
        "label_valid": "bool",
        "price_label_valid": "bool",
        "va_aux_valid": "bool",
        "entry_filled": "bool",
    }
    arrays: dict[str, np.memmap] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for name, dtype in specs.items():
        path = target / "candidate_arrays" / f"{name}.{dtype}.dat"
        arrays[name] = _new_memmap(path, (int(candidate_count),), dtype=dtype)
        metadata[name] = _memmap_metadata(path, (int(candidate_count),), dtype=dtype)

    parquet = pq.ParquetFile(candidate_path)
    offset = 0
    entry_mismatch_count = 0
    columns = [
        "candidate_id",
        "date_idx",
        "symbol_idx",
        "year",
        "label_valid",
        "price_label_valid",
        "va_aux_valid",
        "entry_filled",
    ]
    for batch in parquet.iter_batches(batch_size=131_072, columns=columns):
        data = batch.to_pydict()
        count = int(batch.num_rows)
        stop = offset + count
        ids = np.asarray(data["candidate_id"], dtype=np.int64)
        if not np.array_equal(ids, np.arange(offset, stop, dtype=np.int64)):
            raise ValueError("candidate_id must be dense and aligned with parquet row order")
        date_idx = np.asarray(data["date_idx"], dtype=np.int32)
        symbol_idx = np.asarray(data["symbol_idx"], dtype=np.int32)
        derived_entry = np.asarray(open_buyable[date_idx + 1, symbol_idx], dtype=bool)
        recorded_entry = np.asarray(data["entry_filled"], dtype=bool)
        entry_mismatch_count += int(np.count_nonzero(derived_entry != recorded_entry))
        arrays["date_idx"][offset:stop] = date_idx
        arrays["symbol_idx"][offset:stop] = symbol_idx
        arrays["year"][offset:stop] = np.asarray(data["year"], dtype=np.int16)
        arrays["label_valid"][offset:stop] = np.asarray(data["label_valid"], dtype=bool)
        arrays["price_label_valid"][offset:stop] = np.asarray(data["price_label_valid"], dtype=bool)
        arrays["va_aux_valid"][offset:stop] = np.asarray(data["va_aux_valid"], dtype=bool)
        arrays["entry_filled"][offset:stop] = derived_entry
        offset = stop
    if offset != int(candidate_count):
        raise ValueError("candidate parquet row count differs from manifest")
    for array in arrays.values():
        array.flush()

    date_values = np.asarray(arrays["date_idx"], dtype=np.int32)
    boundaries = np.flatnonzero(np.r_[True, date_values[1:] != date_values[:-1], True])
    spans = [
        {
            "date_idx": int(date_values[int(start)]),
            "candidate_start": int(start),
            "candidate_stop": int(stop),
            "candidate_count": int(stop - start),
            "supervised_count": int(np.count_nonzero(arrays["label_valid"][start:stop])),
        }
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]
    return metadata, spans, entry_mismatch_count


def _materialize_qcurve_targets(
    *,
    source: Mapping[str, Any],
    target: Path,
    candidate_arrays: Mapping[str, Mapping[str, Any]],
    open_raw: np.memmap,
    open_sellable: np.memmap,
    batch_size: int = 16_384,
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_count = int(source["candidate_count"])
    date_idx = _open_memmap(candidate_arrays["date_idx"], dtype="int32")
    symbol_idx = _open_memmap(candidate_arrays["symbol_idx"], dtype="int32")
    entry_filled = _open_memmap(candidate_arrays["entry_filled"], dtype="bool")
    signal_close = _open_memmap(source["execution_arrays"]["exit_close_raw"], dtype="float32")
    trade_dates = np.asarray(source["date_values"], dtype="datetime64[D]")
    future_offsets = np.arange(1, 81, dtype=np.int32)
    cost_contract = QCurveCostContract.from_manifest(source)

    target_meta: dict[str, Any] = {}
    target_maps: dict[tuple[str, str], np.memmap] = {}
    for cost_name in ("base", "stress"):
        for action, horizons in (("enter", QCURVE_ENTRY_HORIZONS), ("hold", QCURVE_HOLD_HORIZONS)):
            path = target / "qcurve_targets" / f"{cost_name}_{action}_net_log_return.float32.dat"
            shape = (len(horizons), candidate_count)
            target_maps[(cost_name, action)] = _existing_or_new_memmap(
                path,
                shape,
                dtype="float32",
                resume=bool(resume),
            )
            target_meta[f"{cost_name}_{action}_net_log_return"] = {
                **_memmap_metadata(path, shape, dtype="float32", layout="horizon_major_candidate_aligned"),
                "horizons": [int(horizons[0]), int(horizons[-1])],
            }
    dependency_path = target / "qcurve_targets" / "max_resolved_dependency_date_idx.int32.dat"
    dependency = _existing_or_new_memmap(
        dependency_path,
        (candidate_count,),
        dtype="int32",
        resume=bool(resume),
    )

    batch_size = int(batch_size)
    checkpoint_rows = batch_size * 8
    cursor_path = target / "qcurve_targets" / "materialization_progress.json"
    start_row = 0
    if bool(resume):
        if cursor_path.is_file():
            cursor = _read_json(cursor_path)
            start_row = int(cursor.get("next_candidate_row", 0) or 0)
        else:
            start_row = _infer_qcurve_resume_row(dependency, checkpoint_rows=checkpoint_rows)
        start_row = max(0, min(start_row, candidate_count))
        start_row = start_row // batch_size * batch_size
    if start_row > 0:
        _write_json(
            cursor_path,
            {
                "status": "resuming",
                "target_semantics_version": QCURVE_TARGET_SEMANTICS_VERSION,
                "next_candidate_row": start_row,
                "candidate_count": candidate_count,
                "updated_at": _now(),
            },
        )

    finite_counts = {
        name: int(start_row) * int(meta["shape"][0])
        for name, meta in target_meta.items()
    }
    for start in range(start_row, candidate_count, batch_size):
        stop = min(start + batch_size, candidate_count)
        rows = np.arange(start, stop, dtype=np.int64)
        signal_indices = np.asarray(date_idx[rows], dtype=np.int32)
        symbols = np.asarray(symbol_idx[rows], dtype=np.int32)
        future_indices = signal_indices[:, None] + future_offsets[None, :]
        future_symbols = symbols[:, None]
        future_opens = np.asarray(open_raw[future_indices, future_symbols], dtype=np.float64)
        future_sellable = np.asarray(open_sellable[future_indices, future_symbols], dtype=bool)
        dates = trade_dates[future_indices]
        signal_closes = np.asarray(signal_close[signal_indices, symbols], dtype=np.float64)
        fills = np.asarray(entry_filled[rows], dtype=bool)
        batch_results: dict[str, dict[str, np.ndarray]] = {}
        for cost_name, multiplier in (
            ("base", 1.0),
            ("stress", float(cost_contract.stress_slippage_multiplier)),
        ):
            result = build_qcurve_targets(
                signal_close_raw=signal_closes,
                future_open_raw=future_opens,
                future_open_sellable=future_sellable,
                future_trade_dates=dates,
                entry_filled=fills,
                contract=cost_contract,
                slippage_multiplier=float(multiplier),
            )
            batch_results[cost_name] = result
            for action in ("enter", "hold"):
                name = f"{cost_name}_{action}_net_log_return"
                values = np.asarray(result[f"{action}_net_log_return"], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"non-finite materialized Q target: {name} rows {start}:{stop}")
                target_maps[(cost_name, action)][:, start:stop] = values.T
                finite_counts[name] += int(values.size)
        base = batch_results["base"]
        max_days = np.maximum(
            np.max(np.asarray(base["enter_resolved_exit_day"], dtype=np.int32), axis=1),
            np.max(np.asarray(base["hold_resolved_exit_day"], dtype=np.int32), axis=1),
        )
        dependency[start:stop] = signal_indices + max_days
        if stop == candidate_count or stop % checkpoint_rows == 0:
            for array in target_maps.values():
                array.flush()
            dependency.flush()
            _write_json(
                cursor_path,
                {
                    "status": "materializing",
                    "target_semantics_version": QCURVE_TARGET_SEMANTICS_VERSION,
                    "next_candidate_row": stop,
                    "candidate_count": candidate_count,
                    "percent_complete": float(stop / candidate_count),
                    "updated_at": _now(),
                },
            )
            if stop < candidate_count:
                for key, array in tuple(target_maps.items()):
                    array._mmap.close()
                    meta = target_meta[f"{key[0]}_{key[1]}_net_log_return"]
                    target_maps[key] = np.memmap(
                        Path(str(meta["path"])),
                        dtype="float32",
                        mode="r+",
                        shape=tuple(int(value) for value in meta["shape"]),
                    )
                dependency._mmap.close()
                dependency = np.memmap(
                    dependency_path,
                    dtype="int32",
                    mode="r+",
                    shape=(candidate_count,),
                )
            del (
                future_indices,
                future_opens,
                future_sellable,
                dates,
                signal_closes,
                batch_results,
                result,
                values,
                base,
                max_days,
                rows,
                signal_indices,
                symbols,
                future_symbols,
                fills,
            )
            _trim_process_working_set()

    for array in target_maps.values():
        array.flush()
    dependency.flush()
    _write_json(
        cursor_path,
        {
            "status": "completed",
            "target_semantics_version": QCURVE_TARGET_SEMANTICS_VERSION,
            "next_candidate_row": candidate_count,
            "candidate_count": candidate_count,
            "percent_complete": 1.0,
            "updated_at": _now(),
        },
    )
    target_meta["max_resolved_dependency_date_idx"] = _memmap_metadata(
        dependency_path,
        (candidate_count,),
        dtype="int32",
        layout="candidate_aligned",
    )
    audit = {
        "candidate_count": candidate_count,
        "target_semantics_version": QCURVE_TARGET_SEMANTICS_VERSION,
        "finite_value_counts": finite_counts,
        "q_label_coverage": 1.0,
        "maximum_resolved_dependency_date_idx": int(np.max(dependency)),
    }
    return target_meta, audit


def derive_qcurve_pack(
    *,
    source_manifest: str | Path = DEFAULT_SOURCE_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_tag: str = DEFAULT_RUN_TAG,
    contract_path: str | Path = DEFAULT_CONTRACT,
    overwrite: bool = False,
    resume: bool = False,
    rebuild_qcurve_targets: bool = False,
    label_batch_size: int = 16_384,
) -> dict[str, Any]:
    source_path = Path(source_manifest).resolve()
    source = _read_json(source_path)
    if source.get("artifact_type") != "qdp_v2_sequence_path_pack":
        raise ValueError("Q-curve source must be a qdp_v2_sequence_path_pack")
    if int(source.get("forward_days", 0)) != 60 or int(source.get("execution_tail_days", 0)) != 20:
        raise ValueError("Q-curve source must provide forward60 plus a 20-day execution tail")
    target = Path(output_root).resolve() / str(run_tag)
    manifest_path = target / "manifest.json"
    if target.exists() and not (bool(overwrite) or bool(resume)):
        raise FileExistsError(target)
    completed_cursor_path = target / "qcurve_targets" / "materialization_progress.json"
    cursor_completed = False
    if completed_cursor_path.is_file():
        cursor_payload = _read_json(completed_cursor_path)
        cursor_completed = (
            str(cursor_payload.get("status", "")) == "completed"
            and str(cursor_payload.get("target_semantics_version", ""))
            == QCURVE_TARGET_SEMANTICS_VERSION
        )
    if (
        manifest_path.is_file()
        and bool(resume)
        and not bool(overwrite)
        and not bool(rebuild_qcurve_targets)
        and _stage_is_complete(target, "qcurve_targets", ())
        and cursor_completed
    ):
        validation = validate_qcurve_pack(manifest_path)
        if validation["status"] == "ok":
            return {"status": "completed", "manifest": str(manifest_path.resolve()), "resumed": True}
    target.mkdir(parents=True, exist_ok=True)
    progress_path = target / "progress.json"
    ma_path = target / "panels" / "ma_state.float32.dat"
    daily_shape = tuple(int(value) for value in source["feature_channels"]["daily_raw"]["shape"])
    ma_shape = (daily_shape[0], daily_shape[1], len(MA_STATE_FEATURES))
    if _stage_is_complete(target, "ma_state", (ma_path,)) and not bool(overwrite):
        ma_panel = np.memmap(ma_path, dtype="float32", mode="r", shape=ma_shape)
    else:
        _write_json(progress_path, {"status": "building_ma_state", "updated_at": _now()})
        ma_panel = _build_ma_panel(source, ma_path)
        _mark_stage(target, "ma_state", {"path": str(ma_path.resolve()), "shape": list(ma_shape)})

    execution = dict(source["execution_arrays"])
    masks = dict(source["masks"])
    raw_open = _open_memmap(execution["entry_open_raw"], dtype="float32")
    up_limit = _open_memmap(execution["entry_up_limit_raw"], dtype="float32")
    down_limit = _open_memmap(execution["exit_down_limit_raw"], dtype="float32")
    observed = _open_memmap(masks["has_bar"], dtype="bool")
    status_valid = _open_memmap(masks["status_valid"], dtype="bool")
    suspended = _open_memmap(masks["is_suspended"], dtype="bool")
    delisted = _open_memmap(masks["is_delisted"], dtype="bool")
    open_buyable_path = target / "masks" / "open_buyable.bool.dat"
    open_sellable_path = target / "masks" / "open_sellable.bool.dat"
    if _stage_is_complete(target, "open_execution_masks", (open_buyable_path, open_sellable_path)) and not bool(overwrite):
        open_buyable = np.memmap(open_buyable_path, dtype="bool", mode="r", shape=raw_open.shape)
        open_sellable = np.memmap(open_sellable_path, dtype="bool", mode="r", shape=raw_open.shape)
        mask_audit = dict(_read_json(_stage_marker(target, "open_execution_masks")).get("audit", {}) or {})
    else:
        _write_json(progress_path, {"status": "building_open_execution_masks", "updated_at": _now()})
        buyable, sellable = build_open_execution_masks(
            raw_open,
            up_limit,
            down_limit,
            observed=observed,
            status_valid=status_valid,
            suspended=suspended,
            delisted=delisted,
        )
        open_buyable = _new_memmap(open_buyable_path, raw_open.shape, dtype="bool")
        open_sellable = _new_memmap(open_sellable_path, raw_open.shape, dtype="bool")
        open_buyable[:] = buyable
        open_sellable[:] = sellable
        open_buyable.flush()
        open_sellable.flush()
        mask_audit = {
            "open_buyable_count": int(np.count_nonzero(open_buyable)),
            "open_sellable_count": int(np.count_nonzero(open_sellable)),
        }
        _mark_stage(target, "open_execution_masks", {"audit": mask_audit})

    sample_path = _hardlink(Path(str(source["sample_index_path"])), target / "sample_index.parquet")
    candidate_path = _hardlink(Path(str(source["candidate_index_path"])), target / "candidate_index.parquet")
    candidate_count = int(source["candidate_count"])
    candidate_array_paths = tuple(
        target / "candidate_arrays" / f"{name}.{dtype}.dat"
        for name, dtype in (
            ("date_idx", "int32"),
            ("symbol_idx", "int32"),
            ("year", "int16"),
            ("label_valid", "bool"),
            ("price_label_valid", "bool"),
            ("va_aux_valid", "bool"),
            ("entry_filled", "bool"),
        )
    )
    spans_path = target / "candidate_date_spans.json"
    if _stage_is_complete(target, "candidate_arrays", (*candidate_array_paths, spans_path)) and not bool(overwrite):
        index_stage = _read_json(_stage_marker(target, "candidate_arrays"))
        candidate_arrays = dict(index_stage["candidate_arrays"])
        candidate_spans = list(_read_json(spans_path)["spans"])
        entry_mismatch_count = int(index_stage.get("entry_filled_mismatch_count", 0))
    else:
        _write_json(progress_path, {"status": "building_candidate_arrays", "updated_at": _now()})
        candidate_arrays, candidate_spans, entry_mismatch_count = _write_candidate_arrays(
            candidate_path,
            target,
            candidate_count=candidate_count,
            open_buyable=open_buyable,
        )
        _write_json(spans_path, {"candidate_count": candidate_count, "spans": candidate_spans})
        _mark_stage(
            target,
            "candidate_arrays",
            {
                "candidate_arrays": candidate_arrays,
                "date_spans_path": str(spans_path.resolve()),
                "entry_filled_mismatch_count": entry_mismatch_count,
            },
        )

    qtarget_required = tuple(
        target / "qcurve_targets" / f"{cost}_{action}_net_log_return.float32.dat"
        for cost in ("base", "stress")
        for action in ("enter", "hold")
    ) + (target / "qcurve_targets" / "max_resolved_dependency_date_idx.int32.dat",)
    if (
        _stage_is_complete(target, "qcurve_targets", qtarget_required)
        and not bool(overwrite)
        and not bool(rebuild_qcurve_targets)
    ):
        target_stage = _read_json(_stage_marker(target, "qcurve_targets"))
        qcurve_views = dict(target_stage["qcurve_target_views"])
        qcurve_audit = dict(target_stage["audit"])
    else:
        _write_json(progress_path, {"status": "materializing_qcurve_targets", "updated_at": _now()})
        target_cursor_path = target / "qcurve_targets" / "materialization_progress.json"
        resume_rebuild = False
        if bool(rebuild_qcurve_targets) and target_cursor_path.is_file():
            cursor = _read_json(target_cursor_path)
            cursor_status = str(cursor.get("status", ""))
            cursor_version = str(cursor.get("target_semantics_version", ""))
            resume_rebuild = (
                cursor_status in {"materializing", "resuming"}
                and cursor_version in {"", QCURVE_TARGET_SEMANTICS_VERSION}
            ) or (
                cursor_status == "completed"
                and cursor_version == QCURVE_TARGET_SEMANTICS_VERSION
            )
        if bool(rebuild_qcurve_targets):
            _write_json(
                _stage_marker(target, "qcurve_targets"),
                {
                    "stage": "qcurve_targets",
                    "status": "rebuilding",
                    "target_semantics_version": QCURVE_TARGET_SEMANTICS_VERSION,
                    "updated_at": _now(),
                },
            )
        qcurve_views, qcurve_audit = _materialize_qcurve_targets(
            source=source,
            target=target,
            candidate_arrays=candidate_arrays,
            open_raw=raw_open,
            open_sellable=open_sellable,
            batch_size=int(label_batch_size),
            resume=(
                bool(resume)
                and not bool(overwrite)
                and (not bool(rebuild_qcurve_targets) or bool(resume_rebuild))
            ),
        )
        _mark_stage(
            target,
            "qcurve_targets",
            {"qcurve_target_views": qcurve_views, "audit": qcurve_audit},
        )

    years = np.asarray([int(str(date)[:4]) for date in source["date_values"]], dtype=np.int16)
    train_mask = np.isin(years, np.asarray(source.get("train_years", []), dtype=np.int16))
    ma_stats = _fit_stats(ma_panel, train_mask)
    binding = _contract_binding(contract_path)

    manifest = dict(source)
    feature_channels = dict(source["feature_channels"])
    feature_channels["ma_state"] = {
        "path": str(ma_path.resolve()),
        "shape": [int(value) for value in ma_panel.shape],
        "columns": list(MA_STATE_FEATURES),
        "semantics": "PIT-only relative EMA, slope, volume/amount regime, and VWAP features",
    }
    manifest_masks = dict(masks)
    manifest_masks["open_buyable"] = {"path": str(open_buyable_path.resolve()), "shape": list(raw_open.shape)}
    manifest_masks["open_sellable"] = {"path": str(open_sellable_path.resolve()), "shape": list(raw_open.shape)}
    normalization = dict(source.get("normalization", {}) or {})
    normalization["ma_state"] = ma_stats
    execution_arrays = dict(execution)
    execution_arrays["open_raw"] = dict(execution_arrays["entry_open_raw"])
    execution_arrays["signal_close_raw"] = dict(execution_arrays["exit_close_raw"])
    manifest.update(
        {
            "created_at": _now(),
            "run_tag": str(run_tag),
            "manifest_path": str(manifest_path.resolve()),
            "source_pack": {
                "manifest_path": str(source_path),
                "manifest_sha256": _sha256_file(source_path),
            },
            "research_contract": binding,
            "development_contract": binding,
            "feature_channels": feature_channels,
            "execution_arrays": execution_arrays,
            "masks": manifest_masks,
            "normalization": normalization,
            "sample_index_path": str(sample_path.resolve()),
            "candidate_index_path": str(candidate_path.resolve()),
            "candidate_arrays": candidate_arrays,
            "candidate_date_spans_path": str(spans_path.resolve()),
            "qcurve_target_views": qcurve_views,
            "qcurve_pack_version": 8,
            "qcurve_build_audit": {
                **mask_audit,
                **qcurve_audit,
                "source_candidate_entry_filled_mismatch_count": int(entry_mismatch_count),
            },
            "qcurve_contract": {
                "decision_time": "signal_day_after_close",
                "entry_execution": "D+1 raw open",
                "exit_execution": "planned raw open, deferred to first sellable open for at most 20 additional trading days",
                "maximum_exit_dependency": "D+80 only for the h=60 curve point",
                "entry_horizons": [2, 60],
                "hold_horizons": [1, 60],
                "reference_portfolio_cash_cny": 1000000.0,
                "reference_position_cash_cny": 1000000.0 / 3.0,
                "maximum_positions": 3,
                "maximum_position_weight": 0.60,
                "q20_gate": 0.0,
                "target_storage": "horizon-major, candidate-id aligned, base and double-slippage stress",
                "purge_dependency": "candidate-aligned maximum actual resolved exit date index",
            },
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(
        progress_path,
        {
            "status": "completed",
            "manifest": str(manifest_path.resolve()),
            "contract_sha256": binding["contract_sha256"],
            "updated_at": _now(),
        },
    )
    return {
        "status": "completed",
        "manifest": str(manifest_path.resolve()),
        "ma_state_shape": list(ma_panel.shape),
        **mask_audit,
        "q_label_coverage": float(qcurve_audit["q_label_coverage"]),
        "contract": binding,
    }


def validate_qcurve_pack(manifest_path: str | Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    blockers: list[str] = []
    if "ma_state" not in dict(manifest.get("feature_channels", {}) or {}):
        blockers.append("missing_ma_state")
    if not {"open_buyable", "open_sellable"}.issubset(dict(manifest.get("masks", {}) or {})):
        blockers.append("missing_open_execution_masks")
    required_targets = {
        "base_enter_net_log_return",
        "base_hold_net_log_return",
        "stress_enter_net_log_return",
        "stress_hold_net_log_return",
        "max_resolved_dependency_date_idx",
    }
    if not required_targets.issubset(
        dict(manifest.get("qcurve_target_views", {}) or {})
    ):
        blockers.append("missing_qcurve_target_views")
    if int(manifest.get("max_label_dependency_days", 0) or 0) != 80:
        blockers.append("invalid_dependency_days")
    for section, names in (
        ("feature_channels", ("ma_state",)),
        ("masks", ("open_buyable", "open_sellable")),
        ("candidate_arrays", ("date_idx", "symbol_idx", "label_valid", "entry_filled")),
        ("qcurve_target_views", tuple(sorted(required_targets))),
    ):
        values = dict(manifest.get(section, {}) or {})
        for name in names:
            path = Path(str(dict(values.get(name, {}) or {}).get("path", "")))
            if not path.is_file():
                blockers.append(f"missing_file:{section}.{name}")
                continue
            meta = dict(values.get(name, {}) or {})
            dtype = str(meta.get("dtype", "bool" if section == "masks" else "float32"))
            shape = tuple(int(value) for value in meta.get("shape", []) or [])
            if shape and path.stat().st_size != int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize:
                blockers.append(f"size_mismatch:{section}.{name}")
    audit = dict(manifest.get("qcurve_build_audit", {}) or {})
    if str(audit.get("target_semantics_version", "")) != QCURVE_TARGET_SEMANTICS_VERSION:
        blockers.append("qcurve_target_semantics_version_mismatch")
    if float(audit.get("q_label_coverage", 0.0) or 0.0) != 1.0:
        blockers.append("q_label_coverage_not_complete")
    if int(manifest.get("qcurve_pack_version", 0) or 0) != 8:
        blockers.append("invalid_qcurve_pack_version")
    return {"status": "ok" if not blockers else "blocked", "blockers": blockers, "manifest": str(Path(manifest_path).resolve())}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the additive Seq100 dynamic Q-curve v8 pack.")
    sub = parser.add_subparsers(dest="command", required=True)
    derive = sub.add_parser("derive")
    derive.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    derive.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    derive.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    derive.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    derive.add_argument("--overwrite", action="store_true")
    derive.add_argument("--resume", action="store_true")
    derive.add_argument("--rebuild-qcurve-targets", action="store_true")
    derive.add_argument("--label-batch-size", type=int, default=16_384)
    derive.add_argument("--json", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "derive":
        result = derive_qcurve_pack(
            source_manifest=args.source_manifest,
            output_root=args.output_root,
            run_tag=str(args.run_tag),
            contract_path=args.contract,
            overwrite=bool(args.overwrite),
            resume=bool(args.resume),
            rebuild_qcurve_targets=bool(args.rebuild_qcurve_targets),
            label_batch_size=int(args.label_batch_size),
        )
    else:
        result = validate_qcurve_pack(args.manifest)
    if bool(args.json):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result)
    return 0 if result.get("status") in {"ok", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
