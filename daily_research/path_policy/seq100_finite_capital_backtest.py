from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
    _buy_order,
    _portfolio_equity,
    _sell_order,
)


WORKSPACE = Path(r"H:\quant_project")
DEFAULT_STUDY_ROOT = WORKSPACE / (
    "daily_research/output/path_policy/studies/"
    "seq100_legal_structured_path_rolling_2023_2025_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_STUDY_ROOT / "analysis" / "finite_capital_20260719"
PROFILES = ("legal_flat_baseline", "structured_joint_turnover")
YEARS = (2023, 2024, 2025)
SLOT_COUNTS = (3, 6, 12, 24, 48)
STARTING_CASH_CNY = 1_000_000.0
MINIMUM_AVAILABLE_BYTES = int(0.5 * 1024**3)
LOW_MEMORY_SECONDS = 2.0
TOP_K = 3
SCHEMA_VERSION = 1


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class _MemoryGuard:
    def __init__(
        self,
        *,
        minimum_available_bytes: int = MINIMUM_AVAILABLE_BYTES,
        low_seconds: float = LOW_MEMORY_SECONDS,
    ) -> None:
        self.minimum_available_bytes = int(minimum_available_bytes)
        self.low_seconds = float(low_seconds)
        self._low_since: float | None = None

    def check(self) -> None:
        available = int(psutil.virtual_memory().available)
        now = time.monotonic()
        if available >= self.minimum_available_bytes:
            self._low_since = None
            return
        if self._low_since is None:
            self._low_since = now
            return
        if now - self._low_since >= self.low_seconds:
            raise MemoryError(
                "available physical memory remained below 0.5 GiB for at least 2 seconds"
            )


@dataclass(frozen=True)
class ForecastDay:
    symbol_idx: np.ndarray
    score: np.ndarray
    planned_day: np.ndarray
    top3_symbol_idx: tuple[int, ...]
    ranked_symbol_idx: tuple[int, ...] = ()


class ForecastBook:
    """Compact, date-indexed forecasts without a multi-million-key Python dict."""

    def __init__(
        self,
        profile: str,
        *,
        top_k: int = TOP_K,
        candidate_scan_k: int | None = None,
    ) -> None:
        self.profile = str(profile)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("forecast-book top_k must be positive")
        self.candidate_scan_k = int(
            self.top_k if candidate_scan_k is None else candidate_scan_k
        )
        if self.candidate_scan_k < self.top_k:
            raise ValueError("candidate_scan_k must be at least top_k")
        self.days: dict[int, ForecastDay] = {}

    def add_day(
        self,
        *,
        date_idx: int,
        symbol_idx: np.ndarray,
        score: np.ndarray,
        planned_day: np.ndarray,
        selection_mask: np.ndarray | None = None,
    ) -> None:
        symbols = np.asarray(symbol_idx, dtype=np.int32)
        scores = np.asarray(score, dtype=np.float64)
        planned = np.asarray(planned_day, dtype=np.int16)
        eligible = (
            np.ones(len(symbols), dtype=bool)
            if selection_mask is None
            else np.asarray(selection_mask, dtype=bool)
        )
        if not (symbols.ndim == scores.ndim == planned.ndim == 1):
            raise ValueError("forecast day arrays must be one-dimensional")
        if eligible.ndim != 1 or not (
            len(symbols) == len(scores) == len(planned) == len(eligible)
        ):
            raise ValueError("forecast day arrays have different lengths")
        if int(date_idx) in self.days:
            raise ValueError(f"duplicate forecast date_idx={date_idx} for {self.profile}")
        order = np.argsort(symbols, kind="mergesort")
        symbols = symbols[order]
        scores = scores[order]
        planned = planned[order]
        eligible = eligible[order]
        if len(symbols) > 1 and bool(np.any(symbols[1:] == symbols[:-1])):
            raise ValueError(f"duplicate symbol forecast on date_idx={date_idx}")
        if not bool(np.isfinite(scores).all()):
            raise ValueError(f"non-finite score on date_idx={date_idx}")
        if bool(np.any((planned < 2) | (planned > 60))):
            raise ValueError(f"illegal planned day on date_idx={date_idx}")
        # Candidate rows are symbol sorted; stable descending score therefore gives
        # symbol order as the deterministic tie-break.
        eligible_positions = np.flatnonzero(eligible)
        ranked_local = np.argsort(-scores[eligible_positions], kind="mergesort")[
            : self.candidate_scan_k
        ]
        ranked_positions = eligible_positions[ranked_local]
        top_positions = ranked_positions[: self.top_k]
        top3 = tuple(int(value) for value in symbols[top_positions])
        ranked = tuple(int(value) for value in symbols[ranked_positions])
        self.days[int(date_idx)] = ForecastDay(
            symbol_idx=symbols,
            score=scores,
            planned_day=planned,
            top3_symbol_idx=top3,
            ranked_symbol_idx=ranked,
        )

    def lookup(self, date_idx: int, symbol_idx: int) -> tuple[float, int] | None:
        day = self.days.get(int(date_idx))
        if day is None:
            return None
        position = int(np.searchsorted(day.symbol_idx, int(symbol_idx)))
        if position >= len(day.symbol_idx) or int(day.symbol_idx[position]) != int(symbol_idx):
            return None
        return float(day.score[position]), int(day.planned_day[position])

    @property
    def signal_date_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self.days))


@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: str
    fixed_day: int | None = None
    stop_mode: str | None = None
    take_profit: float | None = None
    stop_loss: float | None = None

    def validate(self) -> None:
        if self.kind == "fixed":
            if self.fixed_day is None or not 2 <= int(self.fixed_day) <= 60:
                raise ValueError(f"invalid fixed policy: {self}")
        elif self.kind in {"model_plan", "rolling"}:
            if self.fixed_day is not None:
                raise ValueError(f"unexpected fixed day: {self}")
        elif self.kind == "stop":
            if self.stop_mode not in {"intraday_conservative", "close_confirmed"}:
                raise ValueError(f"invalid stop mode: {self}")
            if self.take_profit is None or float(self.take_profit) <= 0.0:
                raise ValueError(f"invalid take profit: {self}")
            if self.stop_loss is None or float(self.stop_loss) <= 0.0:
                raise ValueError(f"invalid stop loss: {self}")
        else:
            raise ValueError(f"unknown policy kind: {self.kind}")


@dataclass(frozen=True)
class StopPlan:
    requested_date_idx: int
    trigger_date_idx: int | None
    trigger_price: float | None
    event: str
    ambiguous: bool


def detect_stop_plan(
    *,
    signal_date_idx: int,
    entry_price: float,
    raw_ohlc: np.ndarray,
    take_profit: float,
    stop_loss: float,
    mode: str,
    forward_days: int = 60,
) -> StopPlan:
    """Resolve a causal bracket/close stop request from D2 through D60.

    The returned date is only the request date. Sellability and any deferral are
    handled by the portfolio simulator using the authoritative execution mask.
    """

    values = np.asarray(raw_ohlc, dtype=np.float64)
    if values.shape != (int(forward_days), 4):
        raise ValueError(f"raw_ohlc must have shape [{forward_days}, 4]")
    if not math.isfinite(entry_price) or entry_price <= 0.0:
        raise ValueError("entry price must be finite and positive")
    if mode not in {"intraday_conservative", "close_confirmed"}:
        raise ValueError(f"unknown stop mode: {mode}")
    take_level = float(entry_price) * (1.0 + float(take_profit))
    stop_level = float(entry_price) * (1.0 - float(stop_loss))
    for offset in range(1, int(forward_days)):  # zero-based D2..D60
        open_price, high_price, low_price, close_price = values[offset]
        if not bool(np.isfinite(values[offset]).all()):
            continue
        event = ""
        price = math.nan
        ambiguous = False
        if mode == "close_confirmed":
            if close_price >= take_level:
                event, price = "take_profit", float(close_price)
            elif close_price <= stop_level:
                event, price = "stop_loss", float(close_price)
        else:
            if open_price >= take_level:
                event, price = "take_profit", float(open_price)
            elif open_price <= stop_level:
                event, price = "stop_loss", float(open_price)
            else:
                take_hit = bool(high_price >= take_level)
                stop_hit = bool(low_price <= stop_level)
                ambiguous = take_hit and stop_hit
                if stop_hit:  # conservative ordering when both occur in one bar
                    event, price = "stop_loss", float(stop_level)
                elif take_hit:
                    event, price = "take_profit", float(take_level)
        if event:
            trigger_idx = int(signal_date_idx) + offset + 1
            return StopPlan(
                requested_date_idx=trigger_idx,
                trigger_date_idx=trigger_idx,
                trigger_price=float(price),
                event=event,
                ambiguous=bool(ambiguous),
            )
    return StopPlan(
        requested_date_idx=int(signal_date_idx) + int(forward_days),
        trigger_date_idx=None,
        trigger_price=None,
        event="time_exit",
        ambiguous=False,
    )


def rolling_plan_update(
    *,
    current_date_idx: int,
    current_requested_date_idx: int,
    hard_cap_date_idx: int,
    score: float,
    planned_day: int,
) -> tuple[int, str]:
    """Use a fresh close-of-day forecast without same-close lookahead.

    A non-positive fresh path value requests the next trading-day close. A
    positive forecast may only accelerate the existing request; it cannot keep
    extending a losing position indefinitely. D60 from the original signal is
    an unconditional hard cap before sellability deferral.
    """

    if not math.isfinite(score):
        return int(current_requested_date_idx), "missing"
    if score <= 0.0:
        candidate = int(current_date_idx) + 1
        reason = "nonpositive_value"
    else:
        candidate = int(current_date_idx) + int(np.clip(round(planned_day), 2, 60))
        reason = "earlier_reforecast"
    updated = min(
        int(current_requested_date_idx),
        int(candidate),
        int(hard_cap_date_idx),
    )
    if updated >= int(current_requested_date_idx):
        return int(current_requested_date_idx), "unchanged"
    return int(updated), reason


@dataclass
class _PendingOrder:
    signal_date_idx: int
    symbol_idx: int
    symbol: str
    entry_filled: bool
    initial_score: float
    initial_planned_day: int


@dataclass
class _Position:
    signal_date_idx: int
    symbol_idx: int
    symbol: str
    shares: int
    entry_price_raw: float
    buy_cash: float
    entry_date_idx: int
    requested_exit_date_idx: int
    hard_cap_date_idx: int
    terminal_date_idx: int
    last_mark_price: float
    stop_plan: StopPlan | None = None


@dataclass(frozen=True)
class BacktestMarket:
    date_values: np.ndarray
    symbol_values: np.ndarray
    entry_open_raw: np.ndarray
    exit_close_raw: np.ndarray
    exit_sellable: np.ndarray
    entry_filled: np.ndarray
    contract: Any
    terminal_recovery_fraction: float
    forward_days: int = 60
    execution_days: int = 80


def _stop_policy_name(mode: str, take_profit: float, stop_loss: float) -> str:
    prefix = "stop_intraday" if mode == "intraday_conservative" else "stop_close"
    return f"{prefix}_tp{int(round(take_profit * 100)):02d}_sl{int(round(stop_loss * 100)):02d}"


def _policy_grid(profile: str) -> list[PolicySpec]:
    policies: list[PolicySpec] = []
    fixed = (3, 4, 5) if profile == "legal_flat_baseline" else (14, 15, 16, 17, 18)
    for day in fixed:
        policies.append(PolicySpec(name=f"fixed_d{day}", kind="fixed", fixed_day=day))
    policies.extend(
        [
            PolicySpec(name="model_plan", kind="model_plan"),
            PolicySpec(name="rolling_reforecast", kind="rolling"),
        ]
    )
    for mode in ("intraday_conservative", "close_confirmed"):
        for take_profit in (0.05, 0.10, 0.15, 0.20):
            for stop_loss in (0.03, 0.05, 0.08, 0.10):
                policies.append(
                    PolicySpec(
                        name=_stop_policy_name(mode, take_profit, stop_loss),
                        kind="stop",
                        stop_mode=mode,
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                    )
                )
    for policy in policies:
        policy.validate()
    return policies


def _stress_selected(profile: str, policy: PolicySpec) -> bool:
    preferred_fixed = 4 if profile == "legal_flat_baseline" else 16
    return bool(
        (policy.kind == "fixed" and policy.fixed_day == preferred_fixed)
        or policy.kind in {"model_plan", "rolling"}
        or policy.name in {"stop_intraday_tp10_sl05", "stop_close_tp10_sl05"}
    )


def _profile_from_run(path: Path) -> str:
    if "structured_joint_turnover" in path.name:
        return "structured_joint_turnover"
    if "legal_flat_baseline" in path.name:
        return "legal_flat_baseline"
    raise ValueError(f"cannot identify profile from {path}")


def _prediction_paths(study_root: Path, year: int) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for run_dir in sorted((study_root / "runs" / "development").glob(f"*_{year}_*")):
        candidate = run_dir / "predictions" / "development_predictions.csv"
        if candidate.is_file():
            found[_profile_from_run(run_dir)] = candidate
    if set(found) != set(PROFILES):
        raise RuntimeError(f"missing prediction files for {year}: {found}")
    return found


def _take_price_path(
    dataset: training.SequencePathPackDataset,
    date_indices: np.ndarray,
    symbol_indices: np.ndarray,
) -> np.ndarray:
    source = dataset.future_path if dataset.future_path is not None else dataset.future_ohlcva_path
    if source is None:
        raise RuntimeError("future OHLC path is unavailable")
    dates = np.asarray(date_indices, dtype=np.int64)
    symbols = np.asarray(symbol_indices, dtype=np.int64)
    if hasattr(source, "take"):
        values = source.take(dates, symbols, field_slice=slice(0, 4))
        return np.asarray(values[:, : dataset.forward_days, :4], dtype=np.float64)
    return np.asarray(source[dates, symbols, : dataset.forward_days, :4], dtype=np.float64)


def _reconstruct_raw_ohlc(adjusted_returns: np.ndarray, raw_close: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(adjusted_returns, dtype=np.float64)
    close = np.asarray(raw_close[:, : adjusted.shape[1]], dtype=np.float64)
    denominator = 1.0 + adjusted[:, :, 3]
    output = np.full_like(adjusted, np.nan, dtype=np.float64)
    valid = np.isfinite(close) & np.isfinite(denominator) & (denominator > 0.0)
    for field in range(4):
        numerator = 1.0 + adjusted[:, :, field]
        output[:, :, field] = np.where(
            valid & np.isfinite(numerator),
            close * numerator / denominator,
            np.nan,
        )
    return output


@dataclass(frozen=True)
class LoadedMaterial:
    study: dict[str, Any]
    manifest_path: Path
    market: BacktestMarket
    books: Mapping[str, ForecastBook]
    raw_top3_paths: Mapping[tuple[int, int], np.ndarray]
    first_signal_date_idx: int
    last_signal_date_idx: int
    source_hashes: Mapping[str, str]


def _load_material(
    study_root: Path,
    *,
    include_raw_top3_paths: bool,
    memory_guard: _MemoryGuard,
) -> LoadedMaterial:
    study = _read_json(study_root / "study.json")
    fold_views = dict(study["material"]["fold_views"])
    manifest_path = Path(str(fold_views[str(YEARS[0])])).resolve()
    audit_pack = CandidateCompleteAuditPack(manifest_path)
    market = BacktestMarket(
        date_values=np.asarray(audit_pack.date_values, dtype=object),
        symbol_values=np.asarray(audit_pack.symbol_values, dtype=object),
        entry_open_raw=audit_pack.entry_open_raw,
        exit_close_raw=audit_pack.exit_close_raw,
        exit_sellable=audit_pack.exit_sellable,
        entry_filled=audit_pack.entry_filled,
        contract=audit_pack.contract,
        terminal_recovery_fraction=float(audit_pack.terminal_recovery_fraction),
        forward_days=int(audit_pack.forward_days),
        execution_days=int(audit_pack.execution_days),
    )
    books = {profile: ForecastBook(profile) for profile in PROFILES}
    raw_paths: dict[tuple[int, int], np.ndarray] = {}
    source_hashes: dict[str, str] = {}

    for year in YEARS:
        memory_guard.check()
        view_path = Path(str(fold_views[str(year)])).resolve()
        manifest = _read_json(view_path)
        dataset = training.SequencePathPackDataset(
            manifest,
            split="development",
            max_samples=0,
            input_channel_profile=training.INPUT_CHANNEL_PROFILE_DAILY_ONLY,
            index_role="candidate",
        )
        candidates = dataset.sample_index
        expected_dates = candidates["trade_date"].astype(str).to_numpy()
        expected_symbols = candidates["symbol"].astype(str).to_numpy()
        date_indices = candidates["date_idx"].to_numpy(dtype=np.int64, copy=False)
        symbol_indices = candidates["symbol_idx"].to_numpy(dtype=np.int64, copy=False)
        prediction_paths = _prediction_paths(study_root, year)
        top_row_positions: list[np.ndarray] = []
        groups = candidates.groupby("date_idx", sort=True).indices

        for profile, prediction_path in prediction_paths.items():
            source_hashes[f"{profile}:{year}"] = _sha256_file(prediction_path)
            frame = pd.read_csv(
                prediction_path,
                usecols=["trade_date", "symbol", "score", "predicted_exit_day"],
                dtype={
                    "trade_date": str,
                    "symbol": str,
                    "score": np.float64,
                    "predicted_exit_day": np.float64,
                },
            )
            if len(frame) != len(candidates):
                raise RuntimeError(f"prediction/sample count mismatch: {profile}:{year}")
            if not np.array_equal(frame["trade_date"].to_numpy(), expected_dates):
                raise RuntimeError(f"prediction date order mismatch: {profile}:{year}")
            if not np.array_equal(frame["symbol"].to_numpy(), expected_symbols):
                raise RuntimeError(f"prediction symbol order mismatch: {profile}:{year}")
            scores = frame["score"].to_numpy(dtype=np.float64, copy=True)
            planned = np.clip(
                np.rint(frame["predicted_exit_day"].to_numpy(dtype=np.float64, copy=True)),
                2,
                60,
            ).astype(np.int16)
            if not bool(np.isfinite(scores).all()):
                raise RuntimeError(f"non-finite score: {profile}:{year}")
            selected_positions: list[int] = []
            for date_idx_raw, positions_raw in groups.items():
                positions = np.asarray(positions_raw, dtype=np.int64)
                books[profile].add_day(
                    date_idx=int(date_idx_raw),
                    symbol_idx=symbol_indices[positions],
                    score=scores[positions],
                    planned_day=planned[positions],
                )
                local_top = np.argsort(-scores[positions], kind="mergesort")[:TOP_K]
                selected_positions.extend(int(value) for value in positions[local_top])
            top_row_positions.append(np.asarray(selected_positions, dtype=np.int64))
            del frame, scores, planned

        if include_raw_top3_paths:
            union_positions = np.unique(np.concatenate(top_row_positions))
            selected_dates = date_indices[union_positions]
            selected_symbols = symbol_indices[union_positions]
            adjusted = _take_price_path(dataset, selected_dates, selected_symbols)
            raw_close = dataset._future_float_panel_batch(
                dataset.exit_close_raw_panel,
                selected_dates,
                selected_symbols,
                days=dataset.forward_days,
            )
            if raw_close is None:
                raise RuntimeError("raw close execution panel is unavailable")
            raw_ohlc = _reconstruct_raw_ohlc(adjusted, np.asarray(raw_close, dtype=np.float64))
            for date_idx, symbol_idx, path in zip(
                selected_dates,
                selected_symbols,
                raw_ohlc,
                strict=True,
            ):
                key = (int(date_idx), int(symbol_idx))
                if key in raw_paths:
                    previous = raw_paths[key]
                    if not bool(np.allclose(previous, path, equal_nan=True, atol=1.0e-7, rtol=1.0e-7)):
                        raise RuntimeError(f"raw Top3 path drift for key={key}")
                else:
                    raw_paths[key] = np.asarray(path, dtype=np.float32).copy()
        del dataset

    signal_sets = [set(book.signal_date_indices) for book in books.values()]
    if not signal_sets or any(values != signal_sets[0] for values in signal_sets[1:]):
        raise RuntimeError("profile signal-date coverage differs")
    signal_indices = sorted(signal_sets[0])
    if not signal_indices:
        raise RuntimeError("no signal dates were loaded")
    return LoadedMaterial(
        study=study,
        manifest_path=manifest_path,
        market=market,
        books=books,
        raw_top3_paths=raw_paths,
        first_signal_date_idx=int(signal_indices[0]),
        last_signal_date_idx=int(signal_indices[-1]),
        source_hashes=source_hashes,
    )


def _initial_requested_exit(
    *,
    order: _PendingOrder,
    policy: PolicySpec,
    entry_price: float,
    raw_top3_paths: Mapping[tuple[int, int], np.ndarray],
    forward_days: int,
    execution_days: int,
) -> tuple[int, StopPlan | None]:
    hard_cap = int(order.signal_date_idx) + int(forward_days)
    if policy.kind == "fixed":
        return int(order.signal_date_idx) + int(policy.fixed_day or 0), None
    if policy.kind in {"model_plan", "rolling"}:
        requested = int(order.signal_date_idx) + int(order.initial_planned_day)
        return min(requested, hard_cap), None
    if policy.kind == "stop":
        key = (int(order.signal_date_idx), int(order.symbol_idx))
        path = raw_top3_paths.get(key)
        if path is None:
            raise RuntimeError(f"missing raw Top3 path for stop policy key={key}")
        stop_plan = detect_stop_plan(
            signal_date_idx=int(order.signal_date_idx),
            entry_price=float(entry_price),
            raw_ohlc=path,
            take_profit=float(policy.take_profit or 0.0),
            stop_loss=float(policy.stop_loss or 0.0),
            mode=str(policy.stop_mode),
            forward_days=int(forward_days),
        )
        return int(stop_plan.requested_date_idx), stop_plan
    raise ValueError(policy.kind)


def _calendar_year_metrics(
    equity_frame: pd.DataFrame,
    *,
    starting_cash: float,
    years: Sequence[int] = YEARS,
) -> list[dict[str, Any]]:
    if equity_frame.empty:
        return []
    dates = pd.to_datetime(equity_frame["trade_date"], errors="raise")
    equity = equity_frame["equity"].to_numpy(dtype=np.float64)
    output: list[dict[str, Any]] = []
    previous_equity = float(starting_cash)
    for year in years:
        mask = dates.dt.year.to_numpy() == int(year)
        if not bool(mask.any()):
            continue
        before = dates.dt.year.to_numpy() < int(year)
        if bool(before.any()):
            previous_equity = float(equity[np.flatnonzero(before)[-1]])
        year_frame = equity_frame.loc[mask]
        year_equity = year_frame["equity"].to_numpy(dtype=np.float64)
        path = np.r_[previous_equity, year_equity]
        drawdown = path / np.maximum.accumulate(path) - 1.0
        daily_return = path[1:] / path[:-1] - 1.0
        volatility = float(np.std(daily_return, ddof=1) * math.sqrt(252.0)) if len(daily_return) > 1 else 0.0
        mean_return = float(np.mean(daily_return)) if len(daily_return) else 0.0
        sharpe = (
            float(mean_return / np.std(daily_return, ddof=1) * math.sqrt(252.0))
            if len(daily_return) > 1 and float(np.std(daily_return, ddof=1)) > 0.0
            else 0.0
        )
        output.append(
            {
                "year": int(year),
                "starting_equity_cny": float(previous_equity),
                "ending_equity_cny": float(year_equity[-1]),
                "net_return": float(year_equity[-1] / previous_equity - 1.0),
                "maximum_drawdown": float(drawdown.min()),
                "annualized_volatility": volatility,
                "sharpe_zero_rate": sharpe,
                "mean_position_count": float(year_frame["position_count"].mean()),
                "mean_capital_utilization": float(year_frame["capital_utilization"].mean()),
                "trading_session_count": int(len(year_frame)),
            }
        )
        previous_equity = float(year_equity[-1])
    return output


def _simulation_metric(
    *,
    equity_frame: pd.DataFrame,
    trade_frame: pd.DataFrame,
    profile: str,
    policy: PolicySpec,
    slots: int,
    cost_scenario: str,
    counters: Mapping[str, int | float],
    first_signal_date_idx: int,
    last_signal_date_idx: int,
    starting_cash: float,
    allow_pyramiding: bool,
    daily_selection_count: int,
    replace_rejected_from_ranked_candidates: bool,
    candidate_scan_k: int,
    calendar_years: Sequence[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    signal_frame = equity_frame[
        equity_frame["date_idx"].astype(int).between(
            int(first_signal_date_idx), int(last_signal_date_idx), inclusive="both"
        )
    ]
    if signal_frame.empty:
        raise RuntimeError("signal-period equity frame is empty")
    signal_equity = signal_frame["equity"].to_numpy(dtype=np.float64)
    signal_path = np.r_[float(starting_cash), signal_equity]
    signal_drawdown = signal_path / np.maximum.accumulate(signal_path) - 1.0
    daily_returns = signal_path[1:] / signal_path[:-1] - 1.0
    daily_std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    total_return = float(signal_equity[-1] / float(starting_cash) - 1.0)
    sessions = int(len(signal_frame))
    if total_return <= -1.0:
        annualized = -1.0
    else:
        annualized = float(np.expm1(np.log1p(total_return) * 252.0 / max(sessions, 1)))
    liquidated_equity = float(equity_frame["equity"].iloc[-1])
    full_equity = equity_frame["equity"].to_numpy(dtype=np.float64)
    full_path = np.r_[float(starting_cash), full_equity]
    full_drawdown = full_path / np.maximum.accumulate(full_path) - 1.0
    annual = _calendar_year_metrics(
        equity_frame,
        starting_cash=float(starting_cash),
        years=YEARS if calendar_years is None else tuple(int(value) for value in calendar_years),
    )
    win_rate = 0.0
    mean_occupied = 0.0
    median_occupied = 0.0
    if not trade_frame.empty:
        win_rate = float((trade_frame["net_pnl_cny"].astype(float) > 0.0).mean())
        mean_occupied = float(trade_frame["occupied_sessions"].astype(float).mean())
        median_occupied = float(trade_frame["occupied_sessions"].astype(float).median())
    replacement_contract = (
        f"scan_ranked_top{int(candidate_scan_k)}_for_rejected_signals"
        if replace_rejected_from_ranked_candidates
        else f"no_top{int(daily_selection_count) + 1}"
    )
    metric: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "profile": str(profile),
        "policy": str(policy.name),
        "policy_kind": str(policy.kind),
        "policy_spec": asdict(policy),
        "slot_count": int(slots),
        "cost_scenario": str(cost_scenario),
        "starting_cash_cny": float(starting_cash),
        "signal_period_ending_equity_cny": float(signal_equity[-1]),
        "signal_period_total_return": total_return,
        "signal_period_cagr_trading_days": annualized,
        "signal_period_maximum_drawdown": float(signal_drawdown.min()),
        "signal_period_annualized_volatility": float(daily_std * math.sqrt(252.0)),
        "signal_period_sharpe_zero_rate": (
            float(np.mean(daily_returns) / daily_std * math.sqrt(252.0))
            if daily_std > 0.0
            else 0.0
        ),
        "liquidated_ending_equity_cny": liquidated_equity,
        "liquidated_total_return": float(liquidated_equity / float(starting_cash) - 1.0),
        "liquidation_tail_return_effect": float(
            liquidated_equity / float(signal_equity[-1]) - 1.0
        ),
        "full_path_maximum_drawdown": float(full_drawdown.min()),
        "minimum_cash_cny": float(equity_frame["cash"].min()),
        "minimum_equity_cny": float(equity_frame["equity"].min()),
        "mean_signal_position_count": float(signal_frame["position_count"].mean()),
        "maximum_position_count": int(equity_frame["position_count"].max()),
        "mean_signal_capital_utilization": float(signal_frame["capital_utilization"].mean()),
        "mean_slot_fill_ratio": float(signal_frame["position_count"].mean() / int(slots)),
        "closed_trade_count": int(len(trade_frame)),
        "winning_trade_rate": win_rate,
        "mean_occupied_sessions": mean_occupied,
        "median_occupied_sessions": median_occupied,
        "signal_session_count": sessions,
        "daily_selection_count": int(daily_selection_count),
        "equity_path_session_count": int(len(equity_frame)),
        "portfolio_contract": (
            f"continuous_1m_daily_top{int(daily_selection_count)}_equal_slot_target_"
            f"{'per_signal_lots' if allow_pyramiding else 'no_pyramiding'}_"
            f"{replacement_contract}_"
            "next_open_entry_close_exit_entries_before_same_day_exits"
        ),
        "allow_pyramiding": bool(allow_pyramiding),
        "replace_rejected_from_ranked_candidates": bool(
            replace_rejected_from_ranked_candidates
        ),
        "candidate_scan_k": int(candidate_scan_k),
        **{str(key): value for key, value in counters.items()},
    }
    return metric, annual


def simulate_portfolio(
    *,
    market: BacktestMarket,
    book: ForecastBook,
    raw_top3_paths: Mapping[tuple[int, int], np.ndarray],
    policy: PolicySpec,
    slots: int,
    cost_scenario: str,
    first_signal_date_idx: int,
    last_signal_date_idx: int,
    starting_cash: float = STARTING_CASH_CNY,
    memory_guard: _MemoryGuard | None = None,
    allow_pyramiding: bool = False,
    replace_rejected_from_ranked_candidates: bool = False,
    calendar_years: Sequence[int] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    policy.validate()
    if int(slots) <= 0:
        raise ValueError("slots must be positive")
    if cost_scenario not in {"base", "double_slippage"}:
        raise ValueError(f"unknown cost scenario: {cost_scenario}")
    if not book.days:
        raise ValueError("forecast book must contain at least one signal date")
    selection_sizes = [len(day.top3_symbol_idx) for day in book.days.values()]
    selection_counts = set(selection_sizes)
    if any(count < 0 or count > int(book.top_k) for count in selection_counts):
        raise ValueError(f"forecast book contains an invalid daily selection count: {selection_counts}")
    # Preserve the established explicitly-truncated Top-K contract when every
    # day has the same positive width.  V4 cash filtering may vary by day (and
    # may select nobody), in which case book.top_k remains the configured cap.
    daily_selection_count = (
        int(next(iter(selection_counts)))
        if len(selection_counts) == 1 and int(next(iter(selection_counts))) > 0
        else int(book.top_k)
    )
    guard = memory_guard or _MemoryGuard()
    multiplier = (
        1.0
        if cost_scenario == "base"
        else float(market.contract.stress_slippage_multiplier)
    )
    final_date_idx = int(last_signal_date_idx) + int(market.execution_days)
    if final_date_idx >= len(market.date_values):
        raise ValueError("execution tail is incomplete")

    cash = float(starting_cash)
    positions: dict[int, _Position] = {}
    next_position_id = 0
    pending: list[_PendingOrder] = []
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    counters: dict[str, int | float] = {
        "buy_count": 0,
        "sell_count": 0,
        "failed_entry_count": 0,
        "skipped_duplicate_signal_count": 0,
        "skipped_no_slot_signal_count": 0,
        "skipped_duplicate_pending_count": 0,
        "terminal_recovery_count": 0,
        "deferred_exit_count": 0,
        "stop_take_profit_count": 0,
        "stop_loss_count": 0,
        "stop_time_exit_count": 0,
        "stop_ambiguous_count": 0,
        "rolling_forecast_observation_count": 0,
        "rolling_missing_forecast_count": 0,
        "rolling_acceleration_count": 0,
        "rolling_nonpositive_exit_request_count": 0,
        "replacement_candidate_scan_count": 0,
        "replacement_order_count": 0,
        "replacement_exhausted_signal_count": 0,
        "fees_and_slippage_cny": 0.0,
        "turnover_notional_cny": 0.0,
    }

    for date_idx in range(int(first_signal_date_idx), final_date_idx + 1):
        guard.check()
        trade_date = str(market.date_values[date_idx])

        # Open: execute only the orders generated after the previous close.
        if pending:
            equity_open = _portfolio_equity(
                cash=cash,
                positions=positions,
                mark_panel=market.entry_open_raw,
                date_idx=date_idx,
            )
            for order in pending:
                duplicate_active = any(
                    position.symbol_idx == order.symbol_idx for position in positions.values()
                )
                if duplicate_active and not allow_pyramiding:
                    counters["skipped_duplicate_pending_count"] += 1
                    continue
                if len(positions) >= int(slots):
                    counters["skipped_no_slot_signal_count"] += 1
                    continue
                entry_price = float(market.entry_open_raw[date_idx, order.symbol_idx])
                if (
                    not order.entry_filled
                    or not math.isfinite(entry_price)
                    or entry_price <= 0.0
                ):
                    counters["failed_entry_count"] += 1
                    continue
                allocation = min(cash, max(float(equity_open), 0.0) / float(slots))
                shares, buy_cash, buy_cost, buy_notional = _buy_order(
                    available_cash=cash,
                    allocated_cash=allocation,
                    entry_price=entry_price,
                    contract=market.contract,
                    slippage_multiplier=multiplier,
                )
                if shares <= 0:
                    counters["failed_entry_count"] += 1
                    continue
                requested, stop_plan = _initial_requested_exit(
                    order=order,
                    policy=policy,
                    entry_price=entry_price,
                    raw_top3_paths=raw_top3_paths,
                    forward_days=market.forward_days,
                    execution_days=market.execution_days,
                )
                if stop_plan is not None:
                    if stop_plan.event == "take_profit":
                        counters["stop_take_profit_count"] += 1
                    elif stop_plan.event == "stop_loss":
                        counters["stop_loss_count"] += 1
                    else:
                        counters["stop_time_exit_count"] += 1
                    if stop_plan.ambiguous:
                        counters["stop_ambiguous_count"] += 1
                cash -= float(buy_cash)
                counters["fees_and_slippage_cny"] += float(buy_cost)
                counters["turnover_notional_cny"] += float(buy_notional)
                counters["buy_count"] += 1
                positions[next_position_id] = _Position(
                    signal_date_idx=int(order.signal_date_idx),
                    symbol_idx=int(order.symbol_idx),
                    symbol=str(order.symbol),
                    shares=int(shares),
                    entry_price_raw=float(entry_price),
                    buy_cash=float(buy_cash),
                    entry_date_idx=int(date_idx),
                    requested_exit_date_idx=int(requested),
                    hard_cap_date_idx=int(order.signal_date_idx) + int(market.forward_days),
                    terminal_date_idx=int(order.signal_date_idx) + int(market.execution_days),
                    last_mark_price=float(entry_price),
                    stop_plan=stop_plan,
                )
                next_position_id += 1
            pending = []

        # Close marks are observed before an executable close exit.
        for position in positions.values():
            mark = float(market.exit_close_raw[date_idx, position.symbol_idx])
            if math.isfinite(mark) and mark >= 0.0:
                position.last_mark_price = mark

        for position_id, position in list(positions.items()):
            symbol_idx = int(position.symbol_idx)
            if date_idx < int(position.requested_exit_date_idx):
                continue
            close_price = float(market.exit_close_raw[date_idx, symbol_idx])
            sellable = bool(market.exit_sellable[date_idx, symbol_idx]) and math.isfinite(
                close_price
            ) and close_price >= 0.0
            terminal = date_idx >= int(position.terminal_date_idx) and not sellable
            if not sellable and not terminal:
                continue
            exit_price = close_price
            exit_reason = "time_exit"
            if terminal:
                exit_price = float(position.entry_price_raw) * float(
                    market.terminal_recovery_fraction
                )
                exit_reason = "terminal_recovery"
                counters["terminal_recovery_count"] += 1
            elif date_idx > int(position.requested_exit_date_idx):
                exit_reason = (
                    f"deferred_after_{position.stop_plan.event}"
                    if position.stop_plan is not None
                    else "deferred_sellable_close"
                )
                counters["deferred_exit_count"] += 1
            elif position.stop_plan is not None:
                stop_plan = position.stop_plan
                exit_reason = str(stop_plan.event)
                if (
                    stop_plan.trigger_date_idx == date_idx
                    and stop_plan.trigger_price is not None
                    and math.isfinite(float(stop_plan.trigger_price))
                ):
                    exit_price = float(stop_plan.trigger_price)
            proceeds, sell_cost, sell_notional = _sell_order(
                shares=position.shares,
                exit_price=float(exit_price),
                exit_date_idx=date_idx,
                date_values=market.date_values,
                contract=market.contract,
                slippage_multiplier=multiplier,
            )
            cash += float(proceeds)
            counters["fees_and_slippage_cny"] += float(sell_cost)
            counters["turnover_notional_cny"] += float(sell_notional)
            counters["sell_count"] += 1
            net_pnl = float(proceeds - position.buy_cash)
            trade_rows.append(
                {
                    "profile": book.profile,
                    "policy": policy.name,
                    "slot_count": int(slots),
                    "cost_scenario": cost_scenario,
                    "symbol": position.symbol,
                    "signal_date": str(market.date_values[position.signal_date_idx]),
                    "entry_date": str(market.date_values[position.entry_date_idx]),
                    "exit_date": trade_date,
                    "signal_date_idx": int(position.signal_date_idx),
                    "entry_date_idx": int(position.entry_date_idx),
                    "exit_date_idx": int(date_idx),
                    "entry_price_raw": float(position.entry_price_raw),
                    "exit_price_raw": float(exit_price),
                    "shares": int(position.shares),
                    "buy_cash_cny": float(position.buy_cash),
                    "sell_proceeds_cny": float(proceeds),
                    "net_pnl_cny": net_pnl,
                    "net_return_on_buy_cash": (
                        float(proceeds / position.buy_cash - 1.0)
                        if position.buy_cash > 0.0
                        else 0.0
                    ),
                    "occupied_sessions": int(date_idx - position.entry_date_idx + 1),
                    "requested_exit_date": str(
                        market.date_values[
                            min(position.requested_exit_date_idx, len(market.date_values) - 1)
                        ]
                    ),
                    "exit_reason": exit_reason,
                }
            )
            positions.pop(position_id)

        equity = _portfolio_equity(
            cash=cash,
            positions=positions,
            mark_panel=market.exit_close_raw,
            date_idx=date_idx,
        )
        utilization = 0.0 if equity <= 0.0 else float(1.0 - cash / equity)
        equity_rows.append(
            {
                "profile": book.profile,
                "policy": policy.name,
                "slot_count": int(slots),
                "cost_scenario": cost_scenario,
                "trade_date": trade_date,
                "date_idx": int(date_idx),
                "equity": float(equity),
                "cash": float(cash),
                "position_count": int(len(positions)),
                "capital_utilization": utilization,
                "inside_signal_period": bool(date_idx <= int(last_signal_date_idx)),
            }
        )

        # A close-of-day rolling forecast may only affect a future close.
        if policy.kind == "rolling":
            for position in positions.values():
                refreshed = book.lookup(date_idx, position.symbol_idx)
                if refreshed is None:
                    counters["rolling_missing_forecast_count"] += 1
                    continue
                counters["rolling_forecast_observation_count"] += 1
                score, planned_day = refreshed
                old_requested = int(position.requested_exit_date_idx)
                updated, reason = rolling_plan_update(
                    current_date_idx=date_idx,
                    current_requested_date_idx=old_requested,
                    hard_cap_date_idx=position.hard_cap_date_idx,
                    score=score,
                    planned_day=planned_day,
                )
                position.requested_exit_date_idx = int(updated)
                if updated < old_requested:
                    counters["rolling_acceleration_count"] += 1
                    if reason == "nonpositive_value":
                        counters["rolling_nonpositive_exit_request_count"] += 1

        # After the close, today's target Top-K creates next-open orders.  The
        # opt-in ranked scan can replace only signal-time rejections such as an
        # already-held symbol; it never looks ahead to next-open fill status.
        forecast_day = book.days.get(date_idx) if date_idx <= int(last_signal_date_idx) else None
        if forecast_day is not None:
            target_order_count = len(forecast_day.top3_symbol_idx)
            ranked_candidates = (
                forecast_day.ranked_symbol_idx or forecast_day.top3_symbol_idx
                if replace_rejected_from_ranked_candidates
                else forecast_day.top3_symbol_idx
            )
            scheduled_order_count = 0
            for candidate_rank, symbol_idx in enumerate(ranked_candidates, start=1):
                if scheduled_order_count >= target_order_count:
                    break
                if replace_rejected_from_ranked_candidates:
                    counters["replacement_candidate_scan_count"] += 1
                duplicate_active = any(
                    position.symbol_idx == symbol_idx for position in positions.values()
                )
                if duplicate_active and not allow_pyramiding:
                    counters["skipped_duplicate_signal_count"] += 1
                    continue
                if len(positions) + len(pending) >= int(slots):
                    if replace_rejected_from_ranked_candidates:
                        counters["skipped_no_slot_signal_count"] += int(
                            target_order_count - scheduled_order_count
                        )
                        break
                    counters["skipped_no_slot_signal_count"] += 1
                    continue
                forecast = book.lookup(date_idx, symbol_idx)
                if forecast is None:
                    raise AssertionError("Top3 forecast lookup failed")
                score, planned_day = forecast
                pending.append(
                    _PendingOrder(
                        signal_date_idx=int(date_idx),
                        symbol_idx=int(symbol_idx),
                        symbol=str(market.symbol_values[symbol_idx]),
                        entry_filled=bool(market.entry_filled[date_idx, symbol_idx]),
                        initial_score=float(score),
                        initial_planned_day=int(planned_day),
                    )
                )
                scheduled_order_count += 1
                if candidate_rank > target_order_count:
                    counters["replacement_order_count"] += 1
            if (
                replace_rejected_from_ranked_candidates
                and scheduled_order_count < target_order_count
                and len(positions) + len(pending) < int(slots)
            ):
                counters["replacement_exhausted_signal_count"] += int(
                    target_order_count - scheduled_order_count
                )

    if positions or pending:
        raise AssertionError(
            f"portfolio did not drain by D+{market.execution_days}: "
            f"positions={len(positions)} pending={len(pending)}"
        )
    counters["turnover_to_starting_cash"] = float(
        counters["turnover_notional_cny"] / float(starting_cash)
    )
    counters["cash_filtered_signal_days"] = int(
        sum(len(day.top3_symbol_idx) < int(book.top_k) for day in book.days.values())
    )
    counters["configured_top_k"] = int(book.top_k)
    counters["selected_name_count_min"] = int(min(selection_sizes))
    counters["selected_name_count_max"] = int(max(selection_sizes))
    counters["selected_name_count_mean"] = float(np.mean(selection_sizes))
    equity_frame = pd.DataFrame(equity_rows)
    trade_frame = pd.DataFrame(trade_rows)
    metric, annual = _simulation_metric(
        equity_frame=equity_frame,
        trade_frame=trade_frame,
        profile=book.profile,
        policy=policy,
        slots=int(slots),
        cost_scenario=cost_scenario,
        counters=counters,
        first_signal_date_idx=int(first_signal_date_idx),
        last_signal_date_idx=int(last_signal_date_idx),
        starting_cash=float(starting_cash),
        allow_pyramiding=bool(allow_pyramiding),
        daily_selection_count=daily_selection_count,
        replace_rejected_from_ranked_candidates=bool(
            replace_rejected_from_ranked_candidates
        ),
        candidate_scan_k=int(book.candidate_scan_k),
        calendar_years=calendar_years,
    )
    return metric, equity_frame, trade_frame, annual


def _job_id(profile: str, policy: str, slots: int, cost_scenario: str) -> str:
    return f"{profile}__{policy}__slots{int(slots):02d}__{cost_scenario}"


def _all_jobs() -> list[tuple[str, PolicySpec, int, str]]:
    jobs: list[tuple[str, PolicySpec, int, str]] = []
    for profile in PROFILES:
        for policy in _policy_grid(profile):
            for slots in SLOT_COUNTS:
                jobs.append((profile, policy, int(slots), "base"))
                if _stress_selected(profile, policy):
                    jobs.append((profile, policy, int(slots), "double_slippage"))
    return jobs


def _selected_detail_policy(profile: str, policy: PolicySpec, cost_scenario: str) -> bool:
    if cost_scenario != "base":
        return False
    preferred_fixed = 4 if profile == "legal_flat_baseline" else 16
    return bool(
        (policy.kind == "fixed" and policy.fixed_day == preferred_fixed)
        or policy.kind in {"model_plan", "rolling"}
        or policy.name in {"stop_intraday_tp10_sl05", "stop_close_tp10_sl05"}
    )


def _contract_payload(material: LoadedMaterial) -> dict[str, Any]:
    policy_contract = {
        profile: [asdict(policy) for policy in _policy_grid(profile)] for profile in PROFILES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "seq100_finite_capital_backtest_contract",
        "source_study_root": str(DEFAULT_STUDY_ROOT),
        "source_study_id": str(material.study.get("study_id", "")),
        "source_study_contract_sha256": str(material.study.get("contract_sha256", "")),
        "source_prediction_sha256": dict(sorted(material.source_hashes.items())),
        "source_pack_manifest": str(material.manifest_path),
        "profiles": list(PROFILES),
        "years": list(YEARS),
        "starting_cash_cny": STARTING_CASH_CNY,
        "daily_selection": "model_score_top3_stable_symbol_tie_break",
        "slot_counts": list(SLOT_COUNTS),
        "position_contract": {
            "entry": "signal_D1_raw_open",
            "exit": "requested_raw_close_or_causal_stop_event_price",
            "same_day_cash_order": "entries_before_exits",
            "target_allocation": "open_equity_divided_by_slot_count",
            "pyramiding": False,
            "top4_replacement": False,
            "cash_when_unfilled": True,
            "terminal_tail_days": 20,
        },
        "rolling_contract": {
            "forecast_observed": "each_close",
            "nonpositive_score": "request_next_trading_day_close",
            "positive_score": "accelerate_only_to_fresh_absolute_planned_day",
            "extension": False,
            "original_signal_hard_cap_day": 60,
            "same_close_execution": False,
        },
        "stop_contract": {
            "domain": "D2_D60",
            "intraday_same_bar_tie": "stop_loss_first_conservative",
            "gap_execution": "raw_open",
            "single_threshold_execution": "threshold",
            "close_confirmed_execution": "raw_close",
            "unsellable": "next_sellable_raw_close",
        },
        "cost_scenarios": ["base", "double_slippage_selected_policies"],
        "policies": policy_contract,
        "protected_boundaries": {
            "qdp_mutated": False,
            "source_pack_mutated": False,
            "models_retrained": False,
            "live_state_mutated": False,
        },
    }


def _load_completed_job(path: Path, contract_sha256: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("contract_sha256") != contract_sha256:
        return None
    if not isinstance(payload.get("metric"), dict):
        return None
    if not isinstance(payload.get("annual_metrics"), list):
        return None
    return payload


def run_backtests(
    *,
    study_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    jobs_dir = output_root / "jobs"
    curves_dir = output_root / "curves"
    trades_dir = output_root / "trades"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    trades_dir.mkdir(parents=True, exist_ok=True)
    guard = _MemoryGuard()
    print("finite-capital: loading forecasts and Top3 raw execution paths", flush=True)
    material = _load_material(
        study_root.resolve(),
        include_raw_top3_paths=True,
        memory_guard=guard,
    )
    contract = _contract_payload(material)
    contract["source_study_root"] = str(study_root.resolve())
    contract_sha256 = _sha256_json(contract)
    contract_document = {**contract, "contract_sha256": contract_sha256}
    contract_path = output_root / "contract.json"
    if contract_path.is_file():
        existing = _read_json(contract_path)
        if existing.get("contract_sha256") != contract_sha256:
            raise RuntimeError(
                "finite-capital output root contains a different source/contract; "
                "choose a new output root"
            )
    else:
        _atomic_write_json(contract_path, contract_document)

    jobs = _all_jobs()
    completed = 0
    resumed = 0
    started_at = time.monotonic()
    for number, (profile, policy, slots, cost_scenario) in enumerate(jobs, start=1):
        guard.check()
        identifier = _job_id(profile, policy.name, slots, cost_scenario)
        job_path = jobs_dir / f"{identifier}.json"
        existing = _load_completed_job(job_path, contract_sha256)
        if existing is not None:
            completed += 1
            resumed += 1
            continue
        metric, equity, trades, annual = simulate_portfolio(
            market=material.market,
            book=material.books[profile],
            raw_top3_paths=material.raw_top3_paths,
            policy=policy,
            slots=slots,
            cost_scenario=cost_scenario,
            first_signal_date_idx=material.first_signal_date_idx,
            last_signal_date_idx=material.last_signal_date_idx,
            starting_cash=STARTING_CASH_CNY,
            memory_guard=guard,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "seq100_finite_capital_backtest_job",
            "contract_sha256": contract_sha256,
            "job_id": identifier,
            "metric": metric,
            "annual_metrics": annual,
        }
        _atomic_write_json(job_path, payload)
        if _selected_detail_policy(profile, policy, cost_scenario):
            equity.to_csv(curves_dir / f"{identifier}.csv", index=False)
            trades.to_csv(trades_dir / f"{identifier}.csv", index=False)
        completed += 1
        if number == 1 or number % 10 == 0 or number == len(jobs):
            elapsed = time.monotonic() - started_at
            state = {
                "schema_version": SCHEMA_VERSION,
                "status": "running" if completed < len(jobs) else "completed",
                "contract_sha256": contract_sha256,
                "completed_job_count": int(completed),
                "total_job_count": int(len(jobs)),
                "resumed_job_count": int(resumed),
                "last_job_id": identifier,
                "elapsed_seconds_this_run": float(elapsed),
                "available_memory_gib": float(psutil.virtual_memory().available / 1024**3),
            }
            _atomic_write_json(output_root / "run_state.json", state)
            print(
                f"finite-capital: {completed}/{len(jobs)} jobs; "
                f"last={identifier}; available={state['available_memory_gib']:.2f} GiB",
                flush=True,
            )
    final_state = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "contract_sha256": contract_sha256,
        "completed_job_count": int(completed),
        "total_job_count": int(len(jobs)),
        "resumed_job_count": int(resumed),
        "elapsed_seconds_this_run": float(time.monotonic() - started_at),
        "available_memory_gib": float(psutil.virtual_memory().available / 1024**3),
    }
    _atomic_write_json(output_root / "run_state.json", final_state)
    return final_state


def _markdown_percent(value: Any) -> str:
    return f"{float(value) * 100.0:+.2f}%"


def summarize_backtests(output_root: Path) -> dict[str, Any]:
    contract = _read_json(output_root / "contract.json")
    contract_sha256 = str(contract["contract_sha256"])
    payloads: list[dict[str, Any]] = []
    for path in sorted((output_root / "jobs").glob("*.json")):
        payload = _load_completed_job(path, contract_sha256)
        if payload is not None:
            payloads.append(payload)
    expected_jobs = _all_jobs()
    if len(payloads) != len(expected_jobs):
        raise RuntimeError(
            f"cannot summarize incomplete run: {len(payloads)}/{len(expected_jobs)} jobs"
        )
    metrics = pd.DataFrame([payload["metric"] for payload in payloads])
    annual_rows: list[dict[str, Any]] = []
    for payload in payloads:
        identity = {
            key: payload["metric"][key]
            for key in ("profile", "policy", "policy_kind", "slot_count", "cost_scenario")
        }
        for row in payload["annual_metrics"]:
            annual_rows.append({**identity, **row})
    annual = pd.DataFrame(annual_rows)
    metrics = metrics.sort_values(
        ["profile", "cost_scenario", "policy", "slot_count"], kind="mergesort"
    ).reset_index(drop=True)
    annual = annual.sort_values(
        ["profile", "cost_scenario", "policy", "slot_count", "year"],
        kind="mergesort",
    ).reset_index(drop=True)
    metrics.to_csv(output_root / "portfolio_metrics.csv", index=False)
    annual.to_csv(output_root / "annual_metrics.csv", index=False)

    preferred = {
        "legal_flat_baseline": "fixed_d4",
        "structured_joint_turnover": "fixed_d16",
    }
    selected_names = {
        "fixed_d4",
        "fixed_d16",
        "model_plan",
        "rolling_reforecast",
        "stop_intraday_tp10_sl05",
        "stop_close_tp10_sl05",
    }
    selected = metrics[
        metrics["policy"].isin(selected_names) & metrics["cost_scenario"].eq("base")
    ].copy()
    selected.to_csv(output_root / "selected_policy_comparison.csv", index=False)

    fixed = metrics[
        metrics["policy_kind"].eq("fixed") & metrics["cost_scenario"].eq("base")
    ].copy()
    fixed.to_csv(output_root / "fixed_exit_comparison.csv", index=False)

    stop_base = metrics[
        metrics["policy_kind"].eq("stop") & metrics["cost_scenario"].eq("base")
    ].copy()
    stop_base["stop_family"] = np.where(
        stop_base["policy"].str.startswith("stop_intraday"),
        "intraday_conservative",
        "close_confirmed",
    )
    stop_ranked = stop_base.sort_values(
        ["profile", "slot_count", "stop_family", "signal_period_cagr_trading_days"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    stop_best = stop_ranked.groupby(
        ["profile", "slot_count", "stop_family"], as_index=False, sort=True
    ).head(1)
    stop_best.to_csv(output_root / "stop_grid_best.csv", index=False)

    fixed_rows: list[dict[str, Any]] = []
    for profile, policy_name in preferred.items():
        frame = metrics[
            metrics["profile"].eq(profile)
            & metrics["policy"].eq(policy_name)
            & metrics["cost_scenario"].eq("base")
        ].copy()
        fixed_rows.extend(frame.to_dict("records"))
    preferred_fixed = pd.DataFrame(fixed_rows).sort_values(
        ["profile", "slot_count"], kind="mergesort"
    )
    preferred_fixed.to_csv(output_root / "preferred_fixed_slot_curve.csv", index=False)

    stress = metrics[metrics["cost_scenario"].eq("double_slippage")].copy()
    stress.to_csv(output_root / "selected_policy_stress.csv", index=False)

    headline: dict[str, Any] = {}
    for profile, policy_name in preferred.items():
        frame = preferred_fixed[preferred_fixed["profile"].eq(profile)].copy()
        best = frame.sort_values(
            ["signal_period_cagr_trading_days", "signal_period_maximum_drawdown"],
            ascending=[False, False],
            kind="mergesort",
        ).iloc[0]
        rolling = metrics[
            metrics["profile"].eq(profile)
            & metrics["policy"].eq("rolling_reforecast")
            & metrics["cost_scenario"].eq("base")
        ].sort_values("signal_period_cagr_trading_days", ascending=False).iloc[0]
        stop = stop_best[stop_best["profile"].eq(profile)].sort_values(
            "signal_period_cagr_trading_days", ascending=False
        ).iloc[0]
        headline[profile] = {
            "preferred_fixed_policy": policy_name,
            "preferred_fixed_best_slot_count": int(best["slot_count"]),
            "preferred_fixed_best_cagr": float(best["signal_period_cagr_trading_days"]),
            "preferred_fixed_best_total_return": float(best["signal_period_total_return"]),
            "preferred_fixed_best_max_drawdown": float(best["signal_period_maximum_drawdown"]),
            "preferred_fixed_best_utilization": float(best["mean_signal_capital_utilization"]),
            "rolling_best_slot_count": int(rolling["slot_count"]),
            "rolling_best_cagr": float(rolling["signal_period_cagr_trading_days"]),
            "best_observed_stop_policy": str(stop["policy"]),
            "best_observed_stop_slot_count": int(stop["slot_count"]),
            "best_observed_stop_cagr": float(stop["signal_period_cagr_trading_days"]),
        }

    comparison_rows: list[dict[str, Any]] = []
    for slots in SLOT_COUNTS:
        legal = preferred_fixed[
            preferred_fixed["profile"].eq("legal_flat_baseline")
            & preferred_fixed["slot_count"].eq(slots)
        ].iloc[0]
        structured = preferred_fixed[
            preferred_fixed["profile"].eq("structured_joint_turnover")
            & preferred_fixed["slot_count"].eq(slots)
        ].iloc[0]
        comparison_rows.append(
            {
                "slot_count": int(slots),
                "legal_flat_d4_cagr": float(legal["signal_period_cagr_trading_days"]),
                "structured_d16_cagr": float(
                    structured["signal_period_cagr_trading_days"]
                ),
                "structured_minus_legal_cagr": float(
                    structured["signal_period_cagr_trading_days"]
                    - legal["signal_period_cagr_trading_days"]
                ),
                "legal_flat_d4_max_drawdown": float(
                    legal["signal_period_maximum_drawdown"]
                ),
                "structured_d16_max_drawdown": float(
                    structured["signal_period_maximum_drawdown"]
                ),
                "legal_flat_d4_utilization": float(
                    legal["mean_signal_capital_utilization"]
                ),
                "structured_d16_utilization": float(
                    structured["mean_signal_capital_utilization"]
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_root / "legal_d4_vs_structured_d16.csv", index=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "seq100_finite_capital_backtest_summary",
        "contract_sha256": contract_sha256,
        "job_count": int(len(metrics)),
        "base_job_count": int(metrics["cost_scenario"].eq("base").sum()),
        "stress_job_count": int(metrics["cost_scenario"].eq("double_slippage").sum()),
        "headline": headline,
        "interpretation_guardrails": [
            "Continuous-account returns are absolute strategy returns, not candidate-pool alpha.",
            "Stop-grid maxima are in-sample choices across the three development years and are not deployment winners.",
            "Rolling refresh uses saved daily score/planned-day outputs, accelerates only, and never uses same-close execution.",
            "The 2025 year-end metric is mark-to-market; liquidation-tail impact is reported separately.",
        ],
    }
    _atomic_write_json(output_root / "summary.json", summary)

    lines = [
        "# Seq100 finite-capital continuous-account backtest",
        "",
        "This analysis replays one CNY 1,000,000 account with daily model Top3 signals, "
        "next-open entries, exact lot/cost rules, no pyramiding, and 3/6/12/24/48 slots.",
        "",
        "## Headline",
        "",
        "| Profile | Reference fixed exit | Best observed slot count | CAGR | Total return | Max drawdown | Mean utilization |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, policy_name in preferred.items():
        item = headline[profile]
        lines.append(
            f"| {profile} | {policy_name} | {item['preferred_fixed_best_slot_count']} | "
            f"{_markdown_percent(item['preferred_fixed_best_cagr'])} | "
            f"{_markdown_percent(item['preferred_fixed_best_total_return'])} | "
            f"{_markdown_percent(item['preferred_fixed_best_max_drawdown'])} | "
            f"{_markdown_percent(item['preferred_fixed_best_utilization'])} |"
        )
    lines.extend(
        [
            "",
            "## Legal D4 versus Structured D16",
            "",
            "| Slots | Legal D4 CAGR | Structured D16 CAGR | Difference | Legal utilization | Structured utilization |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison.to_dict("records"):
        lines.append(
            f"| {int(row['slot_count'])} | {_markdown_percent(row['legal_flat_d4_cagr'])} | "
            f"{_markdown_percent(row['structured_d16_cagr'])} | "
            f"{_markdown_percent(row['structured_minus_legal_cagr'])} | "
            f"{_markdown_percent(row['legal_flat_d4_utilization'])} | "
            f"{_markdown_percent(row['structured_d16_utilization'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These are path-dependent account returns, so they answer the capital-capacity question that cohort averages cannot.",
            "- Stop-grid best rows are descriptive and require a later untouched-period or live check before use.",
            "- Rolling refresh is causal: today's close may request an exit no earlier than the next trading-day close.",
            "- QDP, the source pack, model checkpoints, and live execution state were not modified.",
            "",
        ]
    )
    (output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _dry_run(study_root: Path, output_root: Path) -> dict[str, Any]:
    study = _read_json(study_root / "study.json")
    jobs = _all_jobs()
    by_profile: dict[str, int] = {}
    for profile, _policy, _slots, _cost in jobs:
        by_profile[profile] = by_profile.get(profile, 0) + 1
    return {
        "study_id": study.get("study_id"),
        "study_status": study.get("runtime", {}).get("status"),
        "output_root": str(output_root),
        "total_job_count": len(jobs),
        "jobs_by_profile": by_profile,
        "slot_counts": list(SLOT_COUNTS),
        "qdp_update": False,
        "model_retraining": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "run", "summarize"))
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    study_root = Path(args.study_root).resolve()
    output_root = Path(args.output_root).resolve()
    if args.command == "dry-run":
        print(json.dumps(_dry_run(study_root, output_root), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        try:
            state = run_backtests(study_root=study_root, output_root=output_root)
        except MemoryError as exc:
            output_root.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(
                output_root / "run_state.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "stopped_low_memory",
                    "error": str(exc),
                    "available_memory_gib": float(
                        psutil.virtual_memory().available / 1024**3
                    ),
                },
            )
            print(str(exc), flush=True)
            return 2
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    summary = summarize_backtests(output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
