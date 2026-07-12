from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from daily_research.path_policy.seq100_qcurve_pack import (
    DEFAULT_CONTRACT,
    DEFAULT_RUN_TAG,
    DEFAULT_OUTPUT_ROOT,
    validate_qcurve_pack,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK_MANIFEST = DEFAULT_OUTPUT_ROOT / DEFAULT_RUN_TAG / "manifest.json"
DEFAULT_FOLD_ROOT = Path("daily_research/data/research_store/views/seq100_dynamic_qcurve_v8")
DEVELOPMENT_YEARS = (2022, 2023, 2024, 2025)
TRAIN_START_YEAR = 2012
LOOKBACK_DAYS = 100

PRICE_DAILY_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "open_ret_prev_close",
    "high_ret_prev_close",
    "low_ret_prev_close",
    "close_ret_prev_close",
    "intraday_range_raw",
)
VA_DAILY_COLUMNS = ("volume", "amount", "volume_log", "amount_log")
PRICE_MA_COLUMNS = (
    "close_to_ema_5",
    "close_to_ema_10",
    "close_to_ema_20",
    "close_to_ema_60",
    "ema_5_to_20",
    "ema_20_to_60",
    "ema_5_slope_3d",
    "ema_20_slope_5d",
    "ema_60_slope_10d",
    "close_to_vwap_20",
)
VA_MA_COLUMNS = (
    "volume_ema_5_to_20",
    "volume_ema_20_to_60",
    "amount_ema_5_to_20",
    "amount_ema_20_to_60",
)
BASE_MASKS = ("has_bar", "status_valid", "price_observed", "tradable")


def _workspace_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: str | Path) -> dict[str, Any]:
    result = json.loads(_workspace_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _workspace_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _open_memmap(meta: Mapping[str, Any], *, mode: str = "r") -> np.memmap:
    return np.memmap(
        Path(str(meta["path"])),
        dtype=str(meta.get("dtype", "float32")),
        mode=mode,
        shape=tuple(int(value) for value in meta["shape"]),
    )


def _column_indices(meta: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    columns = [str(value) for value in meta["columns"]]
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"feature panel is missing columns: {missing}")
    return np.asarray([columns.index(name) for name in names], dtype=np.int64)


@dataclass(frozen=True)
class QCurveFeatureProfile:
    name: str
    use_ma: bool
    grouped: bool


FEATURE_PROFILES = {
    "qcurve_gru": QCurveFeatureProfile("qcurve_gru", use_ma=False, grouped=False),
    "qcurve_multiscale_ma": QCurveFeatureProfile("qcurve_multiscale_ma", use_ma=True, grouped=True),
}


class QCurvePack:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = _workspace_path(manifest_path).resolve()
        validation = validate_qcurve_pack(self.manifest_path)
        if validation["status"] != "ok":
            raise ValueError(f"invalid Q-curve pack: {validation['blockers']}")
        self.manifest = _read_json(self.manifest_path)
        self.date_values = np.asarray(self.manifest["date_values"], dtype="datetime64[D]")
        self.symbol_values = np.asarray(self.manifest["symbol_values"], dtype=object)
        self.daily_raw_meta = dict(self.manifest["feature_channels"]["daily_raw"])
        self.daily_state_meta = dict(self.manifest["feature_channels"]["daily_state"])
        self.ma_state_meta = dict(self.manifest["feature_channels"]["ma_state"])
        self.daily_raw = _open_memmap({**self.daily_raw_meta, "dtype": "float32"})
        self.daily_state = _open_memmap({**self.daily_state_meta, "dtype": "float32"})
        self.ma_state = _open_memmap({**self.ma_state_meta, "dtype": "float32"})
        self.masks = {
            name: _open_memmap({**dict(self.manifest["masks"][name]), "dtype": "bool"})
            for name in BASE_MASKS + ("open_buyable", "open_sellable", "signal_eligible")
        }
        self.execution = {
            name: _open_memmap({**dict(self.manifest["execution_arrays"][name]), "dtype": "float32"})
            for name in ("open_raw", "signal_close_raw", "entry_up_limit_raw", "exit_down_limit_raw")
        }
        self.candidate_arrays = {
            name: _open_memmap(meta)
            for name, meta in dict(self.manifest["candidate_arrays"]).items()
        }
        self.target_arrays = {
            name: _open_memmap(meta)
            for name, meta in dict(self.manifest["qcurve_target_views"]).items()
        }
        self.date_spans = list(_read_json(self.manifest["candidate_date_spans_path"])["spans"])
        self.span_by_date = {int(item["date_idx"]): dict(item) for item in self.date_spans}
        self._path_shard: tuple[int, int, np.memmap] | None = None

        self.price_daily_idx = _column_indices(self.daily_raw_meta, PRICE_DAILY_COLUMNS)
        self.va_daily_idx = _column_indices(self.daily_raw_meta, VA_DAILY_COLUMNS)
        self.price_ma_idx = _column_indices(self.ma_state_meta, PRICE_MA_COLUMNS)
        self.va_ma_idx = _column_indices(self.ma_state_meta, VA_MA_COLUMNS)

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_arrays["date_idx"].shape[0])

    def span(self, date_idx: int) -> tuple[int, int]:
        item = self.span_by_date[int(date_idx)]
        return int(item["candidate_start"]), int(item["candidate_stop"])

    def candidate_symbols(self, start: int, stop: int) -> np.ndarray:
        return np.asarray(self.candidate_arrays["symbol_idx"][int(start) : int(stop)], dtype=np.int32)

    @staticmethod
    def _normalize(values: np.ndarray, stats: Mapping[str, Any], indices: np.ndarray) -> np.ndarray:
        mean = np.asarray(stats["mean"], dtype=np.float32)[indices]
        std = np.asarray(stats["std"], dtype=np.float32)[indices]
        normalized = (np.asarray(values, dtype=np.float32) - mean) / np.maximum(std, 1.0e-6)
        return np.where(np.isfinite(normalized), normalized, 0.0).astype(np.float32, copy=False)

    def sequence_batch(
        self,
        *,
        date_idx: int,
        start: int,
        stop: int,
        profile: str,
        normalization: Mapping[str, Any],
    ) -> np.ndarray:
        symbols = self.candidate_symbols(start, stop)
        return self.sequence_symbols(
            date_idx=date_idx,
            symbols=symbols,
            profile=profile,
            normalization=normalization,
        )

    def sequence_symbols(
        self,
        *,
        date_idx: int,
        symbols: np.ndarray,
        profile: str,
        normalization: Mapping[str, Any],
    ) -> np.ndarray:
        feature_profile = FEATURE_PROFILES[str(profile)]
        symbols = np.asarray(symbols, dtype=np.int32).reshape(-1)
        window_start = int(date_idx) - LOOKBACK_DAYS + 1
        if window_start < 0:
            raise ValueError("candidate does not have a complete 100-day input window")
        dates = slice(window_start, int(date_idx) + 1)
        daily = np.asarray(self.daily_raw[dates, symbols], dtype=np.float32).transpose(1, 0, 2)
        state = np.asarray(self.daily_state[dates, symbols], dtype=np.float32).transpose(1, 0, 2)
        price_raw = daily[..., self.price_daily_idx]
        va_raw = daily[..., self.va_daily_idx]
        price_available = np.isfinite(price_raw).all(axis=-1, keepdims=True)
        va_available = np.isfinite(va_raw).all(axis=-1, keepdims=True)
        price = self._normalize(price_raw, normalization["daily_raw"], self.price_daily_idx)
        va = self._normalize(va_raw, normalization["daily_raw"], self.va_daily_idx)
        state_numeric = self._normalize(
            state,
            normalization["daily_state"],
            np.arange(state.shape[-1], dtype=np.int64),
        )
        mask_values = [
            np.asarray(self.masks[name][dates, symbols], dtype=np.float32).T[..., None]
            for name in BASE_MASKS
        ]
        availability = [price_available.astype(np.float32), va_available.astype(np.float32)]

        if feature_profile.use_ma:
            ma = np.asarray(self.ma_state[dates, symbols], dtype=np.float32).transpose(1, 0, 2)
            price_ma_raw = ma[..., self.price_ma_idx]
            va_ma_raw = ma[..., self.va_ma_idx]
            ma_available = np.isfinite(ma).all(axis=-1, keepdims=True)
            price = np.concatenate(
                [price, self._normalize(price_ma_raw, normalization["ma_state"], self.price_ma_idx)],
                axis=-1,
            )
            va = np.concatenate(
                [va, self._normalize(va_ma_raw, normalization["ma_state"], self.va_ma_idx)],
                axis=-1,
            )
            availability.append(ma_available.astype(np.float32))
        state_group = np.concatenate([state_numeric, *mask_values, *availability], axis=-1)
        result = np.concatenate([price, va, state_group], axis=-1)
        return np.ascontiguousarray(result, dtype=np.float32)

    def sequence_group_dims(self, profile: str) -> tuple[int, int, int]:
        feature_profile = FEATURE_PROFILES[str(profile)]
        price = len(PRICE_DAILY_COLUMNS) + (len(PRICE_MA_COLUMNS) if feature_profile.use_ma else 0)
        va = len(VA_DAILY_COLUMNS) + (len(VA_MA_COLUMNS) if feature_profile.use_ma else 0)
        state = int(self.daily_state.shape[2]) + len(BASE_MASKS) + 2 + int(feature_profile.use_ma)
        return price, va, state

    def current_lgbm_features(
        self,
        *,
        date_idx: int,
        symbols: np.ndarray,
        normalization: Mapping[str, Any],
    ) -> np.ndarray:
        symbols = np.asarray(symbols, dtype=np.int32).reshape(-1)
        state = np.asarray(self.daily_state[int(date_idx), symbols], dtype=np.float32)
        ma = np.asarray(self.ma_state[int(date_idx), symbols], dtype=np.float32)
        state_norm = self._normalize(
            state,
            normalization["daily_state"],
            np.arange(state.shape[-1], dtype=np.int64),
        )
        ma_norm = self._normalize(
            ma,
            normalization["ma_state"],
            np.arange(ma.shape[-1], dtype=np.int64),
        )
        numeric = np.concatenate([state_norm, ma_norm], axis=1)
        ranks = np.zeros_like(numeric, dtype=np.float32)
        for column in range(numeric.shape[1]):
            values = numeric[:, column]
            order = np.argsort(values, kind="mergesort")
            ranks[order, column] = (
                np.arange(values.size, dtype=np.float32) + 0.5
            ) / max(values.size, 1)
        return np.ascontiguousarray(np.concatenate([numeric, ranks], axis=1), dtype=np.float32)

    def q_targets(self, start: int, stop: int, *, cost: str = "base") -> tuple[np.ndarray, np.ndarray]:
        enter = np.asarray(
            self.target_arrays[f"{cost}_enter_net_log_return"][:, int(start) : int(stop)],
            dtype=np.float32,
        ).T
        hold = np.asarray(
            self.target_arrays[f"{cost}_hold_net_log_return"][:, int(start) : int(stop)],
            dtype=np.float32,
        ).T
        return np.ascontiguousarray(enter), np.ascontiguousarray(hold)

    def future_ohlcva(self, date_idx: int, symbols: np.ndarray) -> np.ndarray:
        meta = dict(self.manifest["label_arrays"]["future_ohlcva_path"])
        shards = list(meta.get("shards", []) or [])
        if not shards:
            panel = _open_memmap({**meta, "dtype": "float32"})
            return np.asarray(panel[int(date_idx), symbols], dtype=np.float32)
        cached = self._path_shard
        if cached is None or not (cached[0] <= int(date_idx) <= cached[1]):
            shard = next(
                item
                for item in shards
                if int(item["date_start_idx"]) <= int(date_idx) <= int(item["date_end_idx"])
            )
            panel = np.memmap(
                Path(str(shard["path"])),
                dtype="float32",
                mode="r",
                shape=tuple(int(value) for value in shard["shape"]),
            )
            cached = (int(shard["date_start_idx"]), int(shard["date_end_idx"]), panel)
            self._path_shard = cached
        return np.asarray(cached[2][int(date_idx) - cached[0], symbols], dtype=np.float32)


def _input_normalization(pack: QCurvePack, start: int, stop: int, *, block_size: int = 131_072) -> dict[str, Any]:
    panels = {
        "daily_raw": pack.daily_raw,
        "daily_state": pack.daily_state,
        "ma_state": pack.ma_state,
    }
    accumulators = {
        name: {
            "count": np.zeros(panel.shape[2], dtype=np.float64),
            "sum": np.zeros(panel.shape[2], dtype=np.float64),
            "sum_sq": np.zeros(panel.shape[2], dtype=np.float64),
        }
        for name, panel in panels.items()
    }
    for block_start in range(int(start), int(stop), int(block_size)):
        block_stop = min(block_start + int(block_size), int(stop))
        dates = np.asarray(pack.candidate_arrays["date_idx"][block_start:block_stop], dtype=np.int32)
        symbols = np.asarray(pack.candidate_arrays["symbol_idx"][block_start:block_stop], dtype=np.int32)
        for name, panel in panels.items():
            values = np.asarray(panel[dates, symbols], dtype=np.float64)
            finite = np.isfinite(values)
            safe = np.where(finite, values, 0.0)
            accumulators[name]["count"] += finite.sum(axis=0)
            accumulators[name]["sum"] += safe.sum(axis=0)
            accumulators[name]["sum_sq"] += np.square(safe).sum(axis=0)
    result: dict[str, Any] = {}
    for name, values in accumulators.items():
        count = np.maximum(values["count"], 1.0)
        mean = values["sum"] / count
        variance = np.maximum(values["sum_sq"] / count - np.square(mean), 1.0e-12)
        result[name] = {
            "mean": mean.astype(float).tolist(),
            "std": np.sqrt(variance).astype(float).tolist(),
            "finite_count": values["count"].astype(np.int64).tolist(),
        }
    return result


def _target_normalization(pack: QCurvePack, start: int, stop: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for action in ("enter", "hold"):
        panel = pack.target_arrays[f"base_{action}_net_log_return"]
        medians: list[float] = []
        mads: list[float] = []
        scales: list[float] = []
        for horizon_idx in range(int(panel.shape[0])):
            values = np.asarray(panel[horizon_idx, int(start) : int(stop)], dtype=np.float32)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(f"empty target horizon: {action}[{horizon_idx}]")
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            medians.append(median)
            mads.append(mad)
            scales.append(max(mad, 1.0e-4))
        result[action] = {"median": medians, "mad": mads, "scale": scales, "scale_floor": 1.0e-4}
    return result


def _contract_binding(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    semantic = {key: value for key, value in payload.items() if key != "contract_sha256"}
    digest = _canonical_sha256(semantic)
    if str(payload.get("contract_sha256", "")) != digest:
        raise ValueError("Q-curve development contract digest mismatch")
    resolved = _workspace_path(path).resolve()
    return {
        "path": str(resolved),
        "contract_id": str(payload["contract_id"]),
        "contract_sha256": digest,
        "contract_file_sha256": _sha256_file(resolved),
    }


def build_qcurve_development_fold(
    *,
    pack_manifest: str | Path,
    development_year: int,
    output_root: str | Path = DEFAULT_FOLD_ROOT,
    contract_path: str | Path = DEFAULT_CONTRACT,
    overwrite: bool = False,
) -> dict[str, Any]:
    year = int(development_year)
    if year not in DEVELOPMENT_YEARS:
        raise ValueError(f"development_year must be one of {DEVELOPMENT_YEARS}")
    pack = QCurvePack(pack_manifest)
    target = _workspace_path(output_root) / f"development_{year}.json"
    if target.exists() and not bool(overwrite):
        return verify_qcurve_development_fold(target)

    date_years = np.asarray([int(str(value)[:4]) for value in pack.manifest["date_values"]], dtype=np.int16)
    development_dates = [
        int(item["date_idx"])
        for item in pack.date_spans
        if int(date_years[int(item["date_idx"])]) == year
    ]
    if not development_dates:
        raise ValueError(f"pack has no candidates in development year {year}")
    development_start_idx = int(min(development_dates))
    dependency = pack.target_arrays["max_resolved_dependency_date_idx"]
    label_valid = pack.candidate_arrays["label_valid"]
    train_spans: list[dict[str, int]] = []
    purged_spans: list[dict[str, int]] = []
    for raw in pack.date_spans:
        date_idx = int(raw["date_idx"])
        date_year = int(date_years[date_idx])
        if date_year < TRAIN_START_YEAR or date_idx >= development_start_idx:
            continue
        start = int(raw["candidate_start"])
        stop = int(raw["candidate_stop"])
        max_dependency = int(np.max(np.asarray(dependency[start:stop], dtype=np.int32)))
        item = {
            "date_idx": date_idx,
            "candidate_start": start,
            "candidate_stop": stop,
            "max_resolved_dependency_date_idx": max_dependency,
        }
        if max_dependency < development_start_idx:
            train_spans.append(item)
        else:
            purged_spans.append(item)
    if not train_spans:
        raise ValueError("fold has no safe training dates")
    train_start = int(train_spans[0]["candidate_start"])
    train_stop = int(train_spans[-1]["candidate_stop"])
    expected_dates = [int(item["date_idx"]) for item in pack.date_spans if train_start <= int(item["candidate_start"]) < train_stop]
    if expected_dates != [int(item["date_idx"]) for item in train_spans]:
        raise ValueError("safe full-day train spans must be contiguous")
    development_spans = [
        {
            "date_idx": int(pack.span_by_date[date]["date_idx"]),
            "candidate_start": int(pack.span_by_date[date]["candidate_start"]),
            "candidate_stop": int(pack.span_by_date[date]["candidate_stop"]),
        }
        for date in development_dates
    ]
    input_normalization = _input_normalization(pack, train_start, train_stop)
    target_normalization = _target_normalization(pack, train_start, train_stop)
    contract = _contract_binding(contract_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "seq100_dynamic_qcurve_development_fold",
        "created_at": _now(),
        "development_year": year,
        "train_start_year": TRAIN_START_YEAR,
        "pack_manifest": str(pack.manifest_path),
        "pack_manifest_sha256": _sha256_file(pack.manifest_path),
        "candidate_index_sha256": _sha256_file(pack.manifest["candidate_index_path"]),
        "contract": contract,
        "split_roles": {"fit": "train", "evaluation": "development"},
        "train_spans": train_spans,
        "development_spans": development_spans,
        "purged_spans": purged_spans,
        "purge": {
            "rule": "entire signal date retained only when every Q-supervised eligible candidate max_resolved_dependency_date_idx < development_start_date_idx",
            "development_start_date_idx": development_start_idx,
            "development_start_date": str(pack.date_values[development_start_idx]),
            "safe_train_end_date_idx": int(train_spans[-1]["date_idx"]),
            "safe_train_end_date": str(pack.date_values[int(train_spans[-1]["date_idx"])]),
            "purged_date_count": len(purged_spans),
            "dependency_overlap_count": 0,
        },
        "counts": {
            "train_date_count": len(train_spans),
            "development_date_count": len(development_spans),
            "train_candidate_count": int(train_stop - train_start),
            "train_q_supervised_count": int(train_stop - train_start),
            "train_path_aux_supervised_count": int(np.count_nonzero(label_valid[train_start:train_stop])),
            "development_candidate_count": int(
                sum(item["candidate_stop"] - item["candidate_start"] for item in development_spans)
            ),
            "development_q_supervised_count": int(
                sum(item["candidate_stop"] - item["candidate_start"] for item in development_spans)
            ),
            "development_path_aux_supervised_count": int(
                sum(np.count_nonzero(label_valid[item["candidate_start"] : item["candidate_stop"]]) for item in development_spans)
            ),
        },
        "normalization": {
            "fit_scope": "safe full-day training candidate rows only",
            "inputs": input_normalization,
            "targets": target_normalization,
        },
        "training": {
            "seed": 7,
            "minimum_complete_epochs": 1,
            "maximum_epochs": 10,
            "early_stopping_metric": "development_total_loss",
            "early_stopping_mode": "min",
            "early_stopping_patience": 2,
            "topk_selects_checkpoint": False,
        },
    }
    payload["fold_contract_sha256"] = _canonical_sha256(payload)
    _write_json(target, payload)
    return verify_qcurve_development_fold(target)


def verify_qcurve_development_fold(path: str | Path) -> dict[str, Any]:
    fold = _read_json(path)
    blockers: list[str] = []
    stored = str(fold.get("fold_contract_sha256", ""))
    semantic = {key: value for key, value in fold.items() if key != "fold_contract_sha256"}
    if stored != _canonical_sha256(semantic):
        blockers.append("fold_contract_digest_mismatch")
    pack_path = Path(str(fold.get("pack_manifest", "")))
    if not pack_path.is_file() or _sha256_file(pack_path) != str(fold.get("pack_manifest_sha256", "")):
        blockers.append("pack_manifest_provenance_mismatch")
    train = list(fold.get("train_spans", []) or [])
    development = list(fold.get("development_spans", []) or [])
    if not train or not development:
        blockers.append("empty_train_or_development_spans")
    start_idx = int(dict(fold.get("purge", {}) or {}).get("development_start_date_idx", -1))
    if any(int(item["max_resolved_dependency_date_idx"]) >= start_idx for item in train):
        blockers.append("label_dependency_overlap")
    if any(key in fold for key in ("test_spans", "oos_spans", "test_years")):
        blockers.append("historical_test_semantics_present")
    return {
        "status": "ok" if not blockers else "blocked",
        "blockers": blockers,
        "path": str(_workspace_path(path).resolve()),
        "development_year": int(fold.get("development_year", 0) or 0),
        "counts": dict(fold.get("counts", {}) or {}),
        "fold_contract_sha256": stored,
    }


def build_qcurve_development_folds(
    *,
    pack_manifest: str | Path = DEFAULT_PACK_MANIFEST,
    years: Iterable[int] = DEVELOPMENT_YEARS,
    output_root: str | Path = DEFAULT_FOLD_ROOT,
    contract_path: str | Path = DEFAULT_CONTRACT,
    overwrite: bool = False,
) -> dict[str, Any]:
    folds = [
        build_qcurve_development_fold(
            pack_manifest=pack_manifest,
            development_year=int(year),
            output_root=output_root,
            contract_path=contract_path,
            overwrite=bool(overwrite),
        )
        for year in years
    ]
    return {"status": "ok", "folds": folds}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate Seq100 dynamic Q-curve development folds.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--pack-manifest", type=Path, default=DEFAULT_PACK_MANIFEST)
    build.add_argument("--output-root", type=Path, default=DEFAULT_FOLD_ROOT)
    build.add_argument("--years", default=",".join(str(value) for value in DEVELOPMENT_YEARS))
    build.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    build.add_argument("--overwrite", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--fold", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        years = tuple(int(value.strip()) for value in str(args.years).split(",") if value.strip())
        result = build_qcurve_development_folds(
            pack_manifest=args.pack_manifest,
            years=years,
            output_root=args.output_root,
            contract_path=args.contract,
            overwrite=bool(args.overwrite),
        )
    else:
        result = verify_qcurve_development_fold(args.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
