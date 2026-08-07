"""Causal paired action-value study for hold versus exit-and-replace.

The target is a relative decision value, not a fixed-horizon return label.  A
state is sampled from a frozen, legally executable reference account.  At its
close, the hold action keeps the audited shares until the second following
close.  The alternative requests the next legal close sale and, only after
that close, chooses the next-date causal candidate for the following open.
All estimates are updated from outcomes whose common resolution precedes the
prediction date.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_daily_phase_execution as phase
from daily_research.path_policy import seq100_finite_capital_backtest as backtest
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
    _buy_order,
    _sell_order,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / (
    "daily_research/studies/seq100_daily_action_value_v1.json"
)
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / (
    "daily_research/output/path_policy/studies/seq100_daily_action_value_v1"
)
STUDY_ID = "seq100_daily_action_value_v1"
SCHEMA_VERSION = 1
BUILDER_VERSION = 1
COST_SCENARIOS = ("base", "double_slippage")
MODEL_NAMES = ("prior", "geometry", "position", "geometry_position")
REFERENCE_NAMES = ("rolling_phase_boundary", "fixed_d20_reference")
AGE_BIN_COUNT = 5
POSITION_CODE_COUNT = 8
GEOMETRY_CLUSTER_COUNT = 128
OBSERVABLE_STATE_GROUP_COUNT = 7  # six frozen groups plus an explicit unknown state


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_resolve(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path, *, include_hash: bool = True) -> dict[str, Any]:
    target = _resolve(path)
    record: dict[str, Any] = {
        "path": str(target.resolve()),
        "size": int(target.stat().st_size),
    }
    if include_hash:
        record["sha256"] = _sha256(target)
    return record


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_frame(path: str | Path, frame: pd.DataFrame) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd", row_group_size=100_000)
    os.replace(temporary, target)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("action_value_study_id_mismatch")
    source = dict(study["source"])
    if source.get("quality_pool_name") != "quality_liquidity_pit":
        raise ValueError("action_value_quality_pool_mismatch")
    if int(source.get("forbidden_year", 0)) != 2026:
        raise ValueError("action_value_forbidden_year_mismatch")
    if bool(dict(study["counterfactual"]).get("fixed_total_holding_horizon_used")):
        raise ValueError("action_value_fixed_horizon_forbidden")
    if float(dict(study["state"])["smoothing_strength"]) != 200.0:
        raise ValueError("action_value_smoothing_contract_mismatch")
    if int(dict(study["replacement"])["candidate_scan_k"]) != 20:
        raise ValueError("action_value_candidate_scan_contract_mismatch")
    return study


@dataclass(frozen=True)
class Material:
    study: dict[str, Any]
    source: dict[str, Any]
    market: backtest.BacktestMarket
    amount_panel: np.ndarray
    audit_pack: CandidateCompleteAuditPack
    symbol_to_idx: dict[str, int]


def _source_contract(study: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(study["source"])
    phase_manifest_path = _resolve(source["phase_analysis_manifest"])
    if _sha256(phase_manifest_path) != source["expected_phase_analysis_sha256"]:
        raise ValueError("action_value_phase_analysis_hash_mismatch")
    phase_manifest = _read_json(phase_manifest_path)
    if phase_manifest.get("status") != "completed":
        raise ValueError("action_value_phase_analysis_not_completed")
    belief_path = _resolve(source["phase_belief_manifest"])
    if _sha256(belief_path) != source["expected_phase_belief_sha256"]:
        raise ValueError("action_value_phase_belief_hash_mismatch")
    belief_manifest = _read_json(belief_path)
    if belief_manifest.get("status") != "completed":
        raise ValueError("action_value_phase_belief_not_completed")
    continuation_path = _resolve(source["continuation_prediction_manifest"])
    if _sha256(continuation_path) != source["expected_continuation_prediction_sha256"]:
        raise ValueError("action_value_continuation_prediction_hash_mismatch")
    continuation_manifest = _read_json(continuation_path)
    if continuation_manifest.get("status") != "completed":
        raise ValueError("action_value_continuation_prediction_not_completed")
    pack_path = _resolve(source["corrected_pack_manifest"])
    if _sha256(pack_path) != source["expected_corrected_pack_sha256"]:
        raise ValueError("action_value_pack_hash_mismatch")
    pack = _read_json(pack_path)
    belief_paths = {
        int(item["year"]): _resolve(item["path"])
        for item in belief_manifest["years"]
    }
    prediction_paths = {
        int(item["year"]): _resolve(item["path"])
        for item in continuation_manifest["prediction_years"]
    }
    # State clusters and phase groups are already carried by the frozen
    # continuation predictions; no raw future panel is needed here.
    trade_files = {
        str(item["name"]): _resolve(item["trade_file"])
        for item in study["reference_positions"]
    }
    return {
        "phase_manifest_path": phase_manifest_path,
        "phase_manifest": phase_manifest,
        "belief_manifest_path": belief_path,
        "belief_manifest": belief_manifest,
        "belief_paths": belief_paths,
        "continuation_manifest_path": continuation_path,
        "continuation_manifest": continuation_manifest,
        "prediction_paths": prediction_paths,
        "pack_manifest_path": pack_path,
        "pack_manifest": pack,
        "trade_files": trade_files,
    }


def _open_material(study: Mapping[str, Any], source: Mapping[str, Any]) -> Material:
    market, amount_panel, audit_pack = phase._open_market(
        _resolve(source["pack_manifest_path"])
    )
    symbols = {str(value): int(index) for index, value in enumerate(market.symbol_values)}
    return Material(
        study=dict(study),
        source=dict(source),
        market=market,
        amount_panel=np.asarray(amount_panel),
        audit_pack=audit_pack,
        symbol_to_idx=symbols,
    )


def age_bin(age: int | np.ndarray) -> np.ndarray | int:
    """Natural log2 duration bins used without a tuned return horizon."""
    values = np.asarray(age, dtype=np.int64)
    result = np.select(
        [values <= 1, values <= 3, values <= 7, values <= 15],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int8)
    return int(result) if result.ndim == 0 else result


def position_code(
    current_price: float, entry_price: float, running_max: float, running_min: float
) -> int:
    """Encode only sign/order information, avoiding arbitrary percentage cuts."""
    if not all(
        math.isfinite(float(value)) and float(value) > 0.0
        for value in (current_price, entry_price, running_max, running_min)
    ):
        return -1
    profitable = int(float(current_price) >= float(entry_price))
    at_high = int(float(current_price) >= float(running_max) * (1.0 - 1e-9))
    adverse_seen = int(float(running_min) < float(entry_price) * (1.0 - 1e-9))
    return profitable + 2 * at_high + 4 * adverse_seen


def _mark_close(
    close_panel: np.ndarray, date_idx: int, symbol_idx: int, *, minimum_date_idx: int
) -> float:
    """Use the last observed close when a later session has no authoritative bar."""
    for index in range(int(date_idx), int(minimum_date_idx) - 1, -1):
        value = float(close_panel[index, int(symbol_idx)])
        if math.isfinite(value) and value > 0.0:
            return value
    return math.nan


def _choose_replacement(
    *,
    date_idx: int,
    current_symbol_idx: int,
    rankings: Mapping[int, Sequence[int]],
    active_symbols: Mapping[int, set[int]],
) -> tuple[int, int]:
    """Return the first causal candidate not already held at the sale date."""
    active = set(active_symbols.get(int(date_idx), set()))
    active.add(int(current_symbol_idx))
    for rank, symbol_idx in enumerate(rankings.get(int(date_idx), ()), start=1):
        if int(symbol_idx) not in active:
            return int(symbol_idx), int(rank)
    return -1, -1


class ActionStats:
    """Online empirical-Bayes mean with a global fallback and cluster shrinkage."""

    def __init__(self, group_count: int, cluster_count: int = GEOMETRY_CLUSTER_COUNT):
        self.group_count = int(group_count)
        self.cluster_count = int(cluster_count)
        self.count = np.zeros(self.group_count, dtype=np.int64)
        self.total = np.zeros(self.group_count, dtype=np.float64)
        self.squared = np.zeros(self.group_count, dtype=np.float64)
        size = self.group_count * self.cluster_count
        self.cluster_count_seen = np.zeros(size, dtype=np.int64)
        self.cluster_total = np.zeros(size, dtype=np.float64)
        self.global_count = 0
        self.global_total = 0.0

    def update(self, group: np.ndarray, cluster: np.ndarray, value: np.ndarray) -> None:
        groups = np.asarray(group, dtype=np.int64)
        clusters = np.asarray(cluster, dtype=np.int64)
        values = np.asarray(value, dtype=np.float64)
        valid = (
            (groups >= 0)
            & (groups < self.group_count)
            & (clusters >= 0)
            & (clusters < self.cluster_count)
            & np.isfinite(values)
        )
        if not bool(valid.any()):
            return
        groups = groups[valid]
        clusters = clusters[valid]
        values = values[valid]
        np.add.at(self.count, groups, 1)
        np.add.at(self.total, groups, values)
        np.add.at(self.squared, groups, values * values)
        keys = groups * self.cluster_count + clusters
        np.add.at(self.cluster_count_seen, keys, 1)
        np.add.at(self.cluster_total, keys, values)
        self.global_count += len(values)
        self.global_total += float(values.sum())

    def predict(
        self,
        group: np.ndarray,
        cluster: np.ndarray,
        *,
        smoothing: float,
        use_cluster: bool,
    ) -> np.ndarray:
        groups = np.asarray(group, dtype=np.int64)
        clusters = np.asarray(cluster, dtype=np.int64)
        fallback = (
            float(self.global_total / self.global_count)
            if self.global_count
            else 0.0
        )
        prior = np.full(len(groups), fallback, dtype=np.float64)
        valid_group = (groups >= 0) & (groups < self.group_count)
        if bool(valid_group.any()):
            counts = self.count[groups[valid_group]]
            totals = self.total[groups[valid_group]]
            prior[valid_group] = np.where(
                counts > 0, totals / np.maximum(counts, 1), fallback
            )
        if not use_cluster:
            return prior
        valid = valid_group & (clusters >= 0) & (clusters < self.cluster_count)
        if not bool(valid.any()):
            return prior
        keys = groups[valid] * self.cluster_count + clusters[valid]
        counts = self.cluster_count_seen[keys]
        totals = self.cluster_total[keys]
        denom = counts.astype(np.float64) + float(smoothing)
        predicted = (totals + float(smoothing) * prior[valid]) / np.maximum(denom, 1.0)
        output = prior.copy()
        output[valid] = predicted
        return output


def _hac_mean(values: np.ndarray, lag: int = 20) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    if len(series) == 0:
        return {"mean": math.nan, "se": math.nan, "lcb_95": math.nan, "ucb_95": math.nan}
    centered = series - float(series.mean())
    gamma0 = float(np.dot(centered, centered) / len(series))
    long_var = gamma0
    for offset in range(1, min(int(lag), len(series) - 1) + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / len(series))
        long_var += 2.0 * (1.0 - offset / (int(lag) + 1.0)) * covariance
    se = math.sqrt(max(long_var, 0.0) / len(series))
    mean = float(series.mean())
    return {"mean": mean, "se": se, "lcb_95": mean - 1.96 * se, "ucb_95": mean + 1.96 * se}


def _safe_rank_corr(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(p)
    if int(valid.sum()) < 3:
        return math.nan
    av = a[valid]
    pv = p[valid]
    if np.all(av == av[0]) or np.all(pv == pv[0]):
        return math.nan
    return float(pd.Series(av).corr(pd.Series(pv), method="spearman"))


def _expand_reference_positions(
    *,
    material: Material,
    reference_name: str,
    trade_path: Path,
    reference_index: int,
) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    """Expand audited trades into close-time states while they are still held."""
    trades = pd.read_parquet(trade_path)
    required = {
        "symbol",
        "symbol_idx",
        "entry_date_idx",
        "exit_date_idx",
        "entry_price_raw",
        "shares",
        "buy_cash_cny",
        "signal_date_idx",
    }
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"action_value_reference_trade_columns_missing:{sorted(missing)}")
    period = dict(material.study["period"])
    first_idx = int(
        np.searchsorted(material.market.date_values, str(period["first_state_date"]))
    )
    last_idx = int(
        np.searchsorted(material.market.date_values, str(period["last_state_date"]))
    )
    maximum_resolution_idx = int(
        np.searchsorted(
            material.market.date_values, str(period["maximum_resolution_date"])
        )
    )
    if last_idx + 2 != maximum_resolution_idx:
        raise ValueError("action_value_two_session_resolution_tail_mismatch")
    active: dict[int, set[int]] = defaultdict(set)
    rows: list[dict[str, Any]] = []
    for trade_id, row in enumerate(trades.itertuples(index=False)):
        symbol_idx = int(row.symbol_idx)
        entry_idx = int(row.entry_date_idx)
        exit_idx = int(row.exit_date_idx)
        shares = int(row.shares)
        entry_price = float(row.entry_price_raw)
        if shares <= 0 or not math.isfinite(entry_price) or entry_price <= 0.0:
            continue
        # A state is observable after the close while the position remains in
        # the account. The reference exit date itself is excluded.
        start = max(entry_idx, first_idx)
        stop = min(exit_idx - 1, last_idx)
        if stop < start:
            continue
        running_max = -math.inf
        running_min = math.inf
        last_mark = entry_price
        for date_idx in range(entry_idx, stop + 1):
            observed = float(material.market.exit_close_raw[date_idx, symbol_idx])
            if math.isfinite(observed) and observed > 0.0:
                last_mark = observed
                running_max = max(running_max, observed)
                running_min = min(running_min, observed)
            if date_idx < start:
                continue
            active[date_idx].add(symbol_idx)
            if not (
                math.isfinite(last_mark)
                and math.isfinite(running_max)
                and math.isfinite(running_min)
            ):
                continue
            trade_date = str(material.market.date_values[date_idx])
            state_year = int(trade_date[:4])
            age = int(date_idx - entry_idx)
            rows.append(
                {
                    "reference_name": str(reference_name),
                    "reference_index": int(reference_index),
                    "reference_trade_id": int(trade_id),
                    "symbol": str(row.symbol),
                    "symbol_idx": symbol_idx,
                    "trade_date": trade_date,
                    "state_year": state_year,
                    "date_idx": int(date_idx),
                    "signal_date_idx": int(row.signal_date_idx),
                    "entry_date_idx": entry_idx,
                    "reference_exit_date_idx": exit_idx,
                    "holding_age": age,
                    "age_bin": int(age_bin(age)),
                    "shares": shares,
                    "entry_price_raw": entry_price,
                    "current_mark_raw": float(last_mark),
                    "running_max_close_raw": float(running_max),
                    "running_min_close_raw": float(running_min),
                    "floating_return": float(last_mark / entry_price - 1.0),
                    "close_mfe": float(running_max / entry_price - 1.0),
                    "close_mae": float(running_min / entry_price - 1.0),
                    "position_code": int(
                        position_code(last_mark, entry_price, running_max, running_min)
                    ),
                    "reference_buy_cash_cny": float(row.buy_cash_cny),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"action_value_empty_reference_states:{reference_name}")
    if bool(frame.duplicated(["reference_name", "reference_trade_id", "date_idx"]).any()):
        raise ValueError(f"action_value_duplicate_reference_states:{reference_name}")
    return frame, dict(active)


def _load_state_features(
    *,
    material: Material,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only frozen causal state coordinates to the audited holdings."""
    pieces: list[pd.DataFrame] = []
    for year in (int(value) for value in material.study["period"]["state_years"]):
        section = states.loc[states["state_year"].eq(year)].copy()
        if section.empty:
            continue
        prediction = pd.read_parquet(
            material.source["prediction_paths"][year],
            columns=[
                "symbol",
                "date_idx",
                "state_group",
                "geometry_cluster",
                "sequence_cluster",
                "geometry_expected_continuation",
                "geometry_positive_probability",
            ],
        )
        belief = pd.read_parquet(
            material.source["belief_paths"][year],
            columns=[
                "symbol",
                "date_idx",
                "geometry_long_probability",
                "blend01_long_probability",
                "entry_setup",
            ],
        )
        feature = prediction.merge(
            belief,
            on=["symbol", "date_idx"],
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        section = section.merge(
            feature,
            on=["symbol", "date_idx"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        missing = section["state_group"].isna()
        # A held name can leave the current PIT pool after entry. It remains a
        # valid position state; represent the absent current belief explicitly
        # instead of silently filtering it or fabricating a future fill.
        section.loc[missing, "state_group"] = OBSERVABLE_STATE_GROUP_COUNT - 1
        section.loc[missing, "geometry_cluster"] = -1
        section.loc[missing, "sequence_cluster"] = -1
        section.loc[missing, "geometry_long_probability"] = np.nan
        section.loc[missing, "entry_setup"] = "unknown"
        pieces.append(section)
        del prediction, belief, feature, section
        gc.collect()
    result = pd.concat(pieces, ignore_index=True)
    result["state_group"] = result["state_group"].astype(np.int8)
    result["geometry_cluster"] = result["geometry_cluster"].astype(np.int16)
    result["sequence_cluster"] = result["sequence_cluster"].astype(np.int16)
    if not bool(
        result["state_group"].between(0, OBSERVABLE_STATE_GROUP_COUNT - 1).all()
    ):
        raise ValueError("action_value_observable_state_group_out_of_range")
    if not bool(
        result["geometry_cluster"].between(-1, GEOMETRY_CLUSTER_COUNT - 1).all()
    ):
        raise ValueError("action_value_geometry_cluster_out_of_range")
    base_group = (
        (
            result["reference_index"].astype(np.int32)
            * OBSERVABLE_STATE_GROUP_COUNT
            + result["state_group"]
        )
        * AGE_BIN_COUNT
        + result["age_bin"].astype(np.int32)
    )
    result["base_group"] = base_group.astype(np.int16)
    result["position_group"] = (
        base_group.astype(np.int32) * POSITION_CODE_COUNT
        + result["position_code"].astype(np.int32)
    ).astype(np.int16)
    return result.sort_values(
        ["date_idx", "reference_index", "reference_trade_id"], kind="mergesort"
    ).reset_index(drop=True)


def _build_replacement_rankings(material: Material) -> dict[int, tuple[int, ...]]:
    replacement = dict(material.study["replacement"])
    probability_column = str(replacement["probability_column"])
    allowed_setups = {str(value) for value in replacement["entry_setups"]}
    minimum = float(replacement["minimum_probability"])
    scan_k = int(replacement["candidate_scan_k"])
    rankings: dict[int, tuple[int, ...]] = {}
    for year in (int(value) for value in material.study["period"]["state_years"]):
        frame = pd.read_parquet(
            material.source["belief_paths"][year],
            columns=["symbol", "date_idx", probability_column, "entry_setup"],
        )
        frame["symbol_idx"] = frame["symbol"].map(material.symbol_to_idx)
        if bool(frame["symbol_idx"].isna().any()):
            raise ValueError(f"action_value_replacement_symbol_alignment:{year}")
        eligible = (
            frame["entry_setup"].isin(allowed_setups)
            & frame[probability_column].astype(float).gt(minimum)
            & np.isfinite(frame[probability_column].astype(float))
        )
        selected = frame.loc[eligible, ["date_idx", "symbol_idx", probability_column]].copy()
        selected["symbol_idx"] = selected["symbol_idx"].astype(np.int32)
        selected = selected.sort_values(
            ["date_idx", probability_column, "symbol_idx"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        selected = selected.groupby("date_idx", sort=False, as_index=False).head(scan_k)
        for date_idx, group in selected.groupby("date_idx", sort=False):
            rankings[int(date_idx)] = tuple(int(value) for value in group["symbol_idx"])
        del frame, selected
        gc.collect()
    return rankings


def _slippage_multiplier(material: Material, cost_scenario: str) -> float:
    if str(cost_scenario) == "base":
        return 1.0
    if str(cost_scenario) == "double_slippage":
        return float(material.market.costs.stress_slippage_multiplier)
    raise ValueError(f"unknown action-value cost scenario: {cost_scenario}")


def _paired_action_wealth(
    *,
    material: Material,
    date_idx: int,
    current_symbol_idx: int,
    current_shares: int,
    replacement_symbol_idx: int,
    cost_scenario: str,
) -> dict[str, Any]:
    """Resolve the paired wealth at the common second-following close."""
    t = int(date_idx)
    t1 = t + 1
    t2 = t + 2
    symbol = int(current_symbol_idx)
    shares = int(current_shares)
    hold_mark = _mark_close(
        material.market.exit_close_raw, t2, symbol, minimum_date_idx=t
    )
    current_mark = _mark_close(
        material.market.exit_close_raw, t, symbol, minimum_date_idx=t
    )
    valid_hold = (
        shares > 0
        and math.isfinite(hold_mark)
        and hold_mark > 0.0
        and math.isfinite(current_mark)
        and current_mark > 0.0
    )
    if not valid_hold:
        return {
            "action_valid": False,
            "blocked_action_identity": False,
            "replacement_status": "invalid_hold_mark",
            "sale_log_return": math.nan,
            "candidate_gross_log_return": math.nan,
            "sell_cost_cny": math.nan,
            "buy_cost_cny": math.nan,
        }
    hold_wealth = float(shares * hold_mark)
    hold_log_return = float(math.log(hold_mark / current_mark))
    sale_price = float(material.market.exit_close_raw[t1, symbol])
    sellable = bool(material.market.exit_sellable[t1, symbol])
    if not sellable or not math.isfinite(sale_price) or sale_price <= 0.0:
        return {
            "action_valid": True,
            "blocked_action_identity": True,
            "replacement_status": "sale_blocked_identity",
            "hold_wealth_cny": hold_wealth,
            "replace_wealth_cny": hold_wealth,
            "hold_log_return": hold_log_return,
            "replace_log_return": hold_log_return,
            "delta_log_value": 0.0,
            "sale_proceeds_cny": 0.0,
            "replacement_buy_cash_cny": 0.0,
            "replacement_shares": 0,
            "replacement_capacity_cny": 0.0,
            "sale_log_return": math.nan,
            "candidate_gross_log_return": math.nan,
            "sell_cost_cny": 0.0,
            "buy_cost_cny": 0.0,
        }
    multiplier = _slippage_multiplier(material, cost_scenario)
    proceeds, sell_cost, sell_notional = _sell_order(
        shares=shares,
        exit_price=sale_price,
        exit_date_idx=t1,
        date_values=material.market.date_values,
        contract=material.market.costs,
        slippage_multiplier=multiplier,
    )
    replace_wealth = float(max(proceeds, 0.0))
    buy_cash = 0.0
    buy_cost = 0.0
    replacement_shares = 0
    capacity = 0.0
    status = "cash_no_candidate"
    candidate_gross_log_return = math.nan
    replacement_mark = math.nan
    replacement_entry_price = math.nan
    candidate = int(replacement_symbol_idx)
    if candidate >= 0:
        entry_price = float(material.market.entry_open_raw[t2, candidate])
        replacement_entry_price = entry_price
        entry_filled = bool(material.market.entry_filled[t1, candidate])
        signal_amount = float(material.amount_panel[t1, candidate])
        if not math.isfinite(signal_amount) or signal_amount <= 0.0:
            status = "cash_missing_capacity"
        elif not entry_filled or not math.isfinite(entry_price) or entry_price <= 0.0:
            status = "cash_entry_blocked"
        else:
            capacity = signal_amount * float(
                material.study["execution"]["maximum_signal_day_amount_fraction"]
            )
            replacement_shares, buy_cash, buy_cost, _ = _buy_order(
                available_cash=replace_wealth,
                allocated_cash=min(replace_wealth, capacity),
                entry_price=entry_price,
                contract=material.market.costs,
                slippage_multiplier=multiplier,
            )
            if replacement_shares > 0:
                replacement_mark = _mark_close(
                    material.market.exit_close_raw,
                    t2,
                    candidate,
                    minimum_date_idx=t2,
                )
                if not math.isfinite(replacement_mark) or replacement_mark <= 0.0:
                    replacement_mark = entry_price
                    status = "bought_entry_mark_fallback"
                else:
                    status = "bought_and_marked"
                candidate_gross_log_return = float(
                    math.log(replacement_mark / entry_price)
                )
                replace_wealth = (
                    replace_wealth - buy_cash + replacement_shares * replacement_mark
                )
            else:
                status = "cash_lot_or_fee_blocked"
    if replace_wealth <= 0.0 or not math.isfinite(replace_wealth):
        return {
            "action_valid": False,
            "blocked_action_identity": False,
            "replacement_status": "invalid_replace_wealth",
        }
    current_wealth = float(shares * current_mark)
    replace_log_return = float(math.log(replace_wealth / current_wealth))
    delta = float(math.log(replace_wealth / hold_wealth))
    return {
        "action_valid": True,
        "blocked_action_identity": False,
        "replacement_status": status,
        "hold_wealth_cny": hold_wealth,
        "replace_wealth_cny": float(replace_wealth),
        "hold_log_return": hold_log_return,
        "replace_log_return": replace_log_return,
        "delta_log_value": delta,
        "sale_proceeds_cny": float(proceeds),
        "replacement_buy_cash_cny": float(buy_cash),
        "replacement_shares": int(replacement_shares),
        "replacement_capacity_cny": float(capacity),
        "sale_log_return": float(math.log(sale_price / current_mark)),
        "candidate_gross_log_return": candidate_gross_log_return,
        "sell_cost_cny": float(sell_cost),
        "buy_cost_cny": float(buy_cost),
        "sale_notional_cny": float(sell_notional),
        "replacement_entry_price_raw": replacement_entry_price,
        "replacement_mark_raw": replacement_mark,
    }


def build_outcomes(
    *,
    material: Material,
    states: pd.DataFrame,
    rankings: Mapping[int, Sequence[int]],
    active_by_reference: Mapping[str, Mapping[int, set[int]]],
    output_root: Path,
) -> dict[str, Any]:
    """Create the paired counterfactual table, partitioned by state year."""
    output_dir = output_root / "action_outcomes"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    all_rows: list[pd.DataFrame] = []
    year_frames: dict[int, list[pd.DataFrame]] = defaultdict(list)
    for (reference_name, year), group in states.groupby(
        ["reference_name", "state_year"], sort=True
    ):
        rows: list[dict[str, Any]] = []
        active_map = active_by_reference[str(reference_name)]
        for state in group.itertuples(index=False):
            t = int(state.date_idx)
            if t + 2 >= len(material.market.date_values):
                continue
            candidate, candidate_rank = _choose_replacement(
                date_idx=t + 1,
                current_symbol_idx=int(state.symbol_idx),
                rankings=rankings,
                active_symbols=active_map,
            )
            base = {
                "reference_name": str(state.reference_name),
                "reference_index": int(state.reference_index),
                "reference_trade_id": int(state.reference_trade_id),
                "symbol": str(state.symbol),
                "symbol_idx": int(state.symbol_idx),
                "trade_date": str(state.trade_date),
                "state_year": int(state.state_year),
                "date_idx": t,
                "resolution_date_idx": t + 2,
                "holding_age": int(state.holding_age),
                "age_bin": int(state.age_bin),
                "shares": int(state.shares),
                "entry_price_raw": float(state.entry_price_raw),
                "current_mark_raw": float(state.current_mark_raw),
                "floating_return": float(state.floating_return),
                "close_mfe": float(state.close_mfe),
                "close_mae": float(state.close_mae),
                "position_code": int(state.position_code),
                "state_group": int(state.state_group),
                "geometry_cluster": int(state.geometry_cluster),
                "sequence_cluster": int(state.sequence_cluster),
                "base_group": int(state.base_group),
                "position_group": int(state.position_group),
                "geometry_long_probability": float(state.geometry_long_probability),
                "entry_setup": str(state.entry_setup),
                "candidate_symbol_idx": int(candidate),
                "candidate_rank": int(candidate_rank),
                "candidate_available": bool(candidate >= 0),
            }
            for cost_scenario in COST_SCENARIOS:
                outcome = _paired_action_wealth(
                    material=material,
                    date_idx=t,
                    current_symbol_idx=int(state.symbol_idx),
                    current_shares=int(state.shares),
                    replacement_symbol_idx=int(candidate),
                    cost_scenario=cost_scenario,
                )
                suffix = str(cost_scenario)
                for key, value in outcome.items():
                    base[f"{key}_{suffix}"] = value
            rows.append(base)
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        year_frames[int(year)].append(frame)
        all_rows.append(frame)
        del rows
        gc.collect()
    for year, frames in sorted(year_frames.items()):
        frame = pd.concat(frames, ignore_index=True)
        target = output_dir / f"year={int(year)}" / "part-0000.parquet"
        _write_frame(target, frame)
        records.append(
            {
                "year": int(year),
                "path": str(target.resolve()),
                "rows": len(frame),
                "valid_base": int(frame["action_valid_base"].sum()),
                "valid_double_slippage": int(
                    frame["action_valid_double_slippage"].sum()
                ),
                "blocked_identity": int(frame["blocked_action_identity_base"].sum()),
            }
        )
        del frame
    combined = pd.concat(all_rows, ignore_index=True)
    expected_rows = len(states)
    if len(combined) != expected_rows:
        raise ValueError(f"action_value_outcome_row_count_mismatch:{len(combined)}:{expected_rows}")
    manifest = {
        "schema": "seq100_daily_action_outcomes/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "years": records,
        "rows": len(combined),
        "valid_base": int(combined["action_valid_base"].sum()),
        "valid_double_slippage": int(combined["action_valid_double_slippage"].sum()),
        "blocked_identity": int(combined["blocked_action_identity_base"].sum()),
        "forbidden_2026_rows": int(
            (combined["trade_date"].astype(str) > "2025-12-31").sum()
        ),
        "fixed_total_holding_horizon_used": False,
    }
    _write_json(output_root / "outcome_manifest.json", manifest)
    del all_rows, combined
    gc.collect()
    return manifest


def _outcome_paths(output_root: Path, study: Mapping[str, Any]) -> dict[int, Path]:
    manifest = _read_json(output_root / "outcome_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("action_value_outcomes_not_completed")
    return {int(item["year"]): _resolve(item["path"]) for item in manifest["years"]}


def _model_group_and_cluster(
    frame: pd.DataFrame, model: str
) -> tuple[np.ndarray, np.ndarray, bool]:
    if model == "prior":
        return frame["base_group"].to_numpy(np.int64), np.zeros(len(frame), dtype=np.int64), False
    if model == "geometry":
        return (
            frame["base_group"].to_numpy(np.int64),
            frame["geometry_cluster"].to_numpy(np.int64),
            True,
        )
    if model == "position":
        return (
            frame["position_group"].to_numpy(np.int64),
            np.zeros(len(frame), dtype=np.int64),
            False,
        )
    if model == "geometry_position":
        return (
            frame["position_group"].to_numpy(np.int64),
            frame["geometry_cluster"].to_numpy(np.int64),
            True,
        )
    raise KeyError(model)


def _prediction_group_count(model: str) -> int:
    return (
        len(REFERENCE_NAMES) * OBSERVABLE_STATE_GROUP_COUNT * AGE_BIN_COUNT
        if model in {"prior", "geometry"}
        else len(REFERENCE_NAMES)
        * OBSERVABLE_STATE_GROUP_COUNT
        * AGE_BIN_COUNT
        * POSITION_CODE_COUNT
    )


def _apply_outcome_updates(
    *,
    states: Mapping[tuple[str, str], ActionStats],
    outcome_rows: pd.DataFrame,
    cursor: int,
    before_date_idx: int,
    cost_scenario: str,
    smoothing: float,
) -> int:
    if outcome_rows.empty:
        return cursor
    resolution = outcome_rows["resolution_date_idx"].to_numpy(np.int64)
    next_cursor = int(cursor)
    while next_cursor < len(outcome_rows) and int(resolution[next_cursor]) < int(
        before_date_idx
    ):
        row = outcome_rows.iloc[next_cursor]
        valid_key = f"{row.reference_name}:{cost_scenario}"
        value = float(row[f"delta_log_value_{cost_scenario}"])
        if bool(row[f"action_valid_{cost_scenario}"]) and math.isfinite(value):
            for model in MODEL_NAMES:
                group, cluster, _ = _model_group_and_cluster(
                    outcome_rows.iloc[[next_cursor]], model
                )
                states[f"{model}:{valid_key}"].update(
                    group, cluster, np.asarray([value], dtype=np.float64)
                )
        next_cursor += 1
    return next_cursor


def _build_prediction_rows(
    *,
    material: Material,
    states_frame: pd.DataFrame,
    outcome_paths: Mapping[int, Path],
    output_root: Path,
) -> dict[str, Any]:
    first_prediction_idx = int(
        np.searchsorted(
            material.market.date_values,
            str(material.study["period"]["first_prediction_date"]),
        )
    )
    smoothing = float(material.study["state"]["smoothing_strength"])
    output_dir = output_root / "oos_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    # The outcome table is small relative to the full universe. Keep one sorted
    # copy in memory so online updates are a simple monotone cursor operation.
    outcome_parts = [pd.read_parquet(path) for path in outcome_paths.values()]
    outcomes = pd.concat(outcome_parts, ignore_index=True).sort_values(
        ["resolution_date_idx", "date_idx", "reference_index", "reference_trade_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    del outcome_parts
    stats: dict[str, ActionStats] = {}
    for cost in COST_SCENARIOS:
        for model in MODEL_NAMES:
            stats[f"{model}:rolling_phase_boundary:{cost}"] = ActionStats(
                _prediction_group_count(model)
            )
            stats[f"{model}:fixed_d20_reference:{cost}"] = ActionStats(
                _prediction_group_count(model)
            )
    cursor_by_cost = {cost: 0 for cost in COST_SCENARIOS}
    records: list[dict[str, Any]] = []
    for year, frame in states_frame.groupby("state_year", sort=True):
        if int(year) < 2020:
            continue
        rows: list[pd.DataFrame] = []
        for date_idx, day in frame.groupby("date_idx", sort=True):
            date_idx = int(date_idx)
            if date_idx < first_prediction_idx:
                continue
            for cost in COST_SCENARIOS:
                cursor_by_cost[cost] = _apply_outcome_updates(
                    states=stats,
                    outcome_rows=outcomes,
                    cursor=cursor_by_cost[cost],
                    before_date_idx=date_idx,
                    cost_scenario=cost,
                    smoothing=smoothing,
                )
            prediction = day.copy()
            prediction["maximum_training_resolution_date_idx"] = np.where(
                prediction["reference_name"].eq("rolling_phase_boundary"),
                outcomes["resolution_date_idx"].iloc[
                    max(0, min(cursor_by_cost["base"] - 1, len(outcomes) - 1))
                ]
                if cursor_by_cost["base"] > 0
                else -1,
                outcomes["resolution_date_idx"].iloc[
                    max(0, min(cursor_by_cost["base"] - 1, len(outcomes) - 1))
                ]
                if cursor_by_cost["base"] > 0
                else -1,
            ).astype(np.int32)
            for cost in COST_SCENARIOS:
                for model in MODEL_NAMES:
                    group_code, cluster, use_cluster = _model_group_and_cluster(
                        prediction, model
                    )
                    values = stats[f"{model}:rolling_phase_boundary:{cost}"]
                    # Predictions must use the matching reference account's
                    # stats; slice by reference below rather than sharing one
                    # global state object.
                    output = np.zeros(len(prediction), dtype=np.float64)
                    for reference_name in REFERENCE_NAMES:
                        mask = prediction["reference_name"].eq(reference_name).to_numpy()
                        if not bool(mask.any()):
                            continue
                        values = stats[f"{model}:{reference_name}:{cost}"]
                        output[mask] = values.predict(
                            group_code[mask],
                            cluster[mask],
                            smoothing=smoothing,
                            use_cluster=use_cluster,
                        )
                    prediction[f"predicted_{model}_{cost}"] = output.astype(np.float32)
                    actual = prediction[f"delta_log_value_{cost}"].astype(float)
                    valid = prediction[f"action_valid_{cost}"].astype(bool)
                    action = (output > 0.0) & valid.to_numpy() & ~prediction[
                        f"blocked_action_identity_{cost}"
                    ].astype(bool).to_numpy()
                    prediction[f"policy_gain_{model}_{cost}"] = np.where(
                        action, actual.to_numpy(), 0.0
                    ).astype(np.float32)
                    prediction[f"policy_replace_{model}_{cost}"] = action
            prediction["causal_update_violation"] = prediction[
                "maximum_training_resolution_date_idx"
            ] >= date_idx
            rows.append(prediction)
        if not rows:
            continue
        result = pd.concat(rows, ignore_index=True)
        if bool(result["causal_update_violation"].any()):
            raise ValueError(f"action_value_causal_update_violation:{year}")
        target = output_dir / f"year={int(year)}" / "part-0000.parquet"
        _write_frame(target, result)
        records.append(
            {
                "year": int(year),
                "path": str(target.resolve()),
                "rows": len(result),
                "causal_update_violations": int(result["causal_update_violation"].sum()),
            }
        )
        del rows, result
        gc.collect()
    manifest = {
        "schema": "seq100_daily_action_predictions/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "years": records,
        "rows": int(sum(item["rows"] for item in records)),
        "causal_update_violations": int(
            sum(item["causal_update_violations"] for item in records)
        ),
        "first_prediction_date": str(material.study["period"]["first_prediction_date"]),
        "fixed_total_holding_horizon_used": False,
    }
    _write_json(output_root / "prediction_manifest.json", manifest)
    del outcomes
    gc.collect()
    return manifest


def _prediction_paths(output_root: Path) -> dict[int, Path]:
    manifest = _read_json(output_root / "prediction_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("action_value_predictions_not_completed")
    return {int(item["year"]): _resolve(item["path"]) for item in manifest["years"]}


def _daily_action_metrics(
    frame: pd.DataFrame, *, cost_scenario: str, model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_column = f"delta_log_value_{cost_scenario}"
    prediction_column = f"predicted_{model}_{cost_scenario}"
    policy_column = f"policy_gain_{model}_{cost_scenario}"
    replace_column = f"policy_replace_{model}_{cost_scenario}"
    valid_column = f"action_valid_{cost_scenario}"
    valid = frame[valid_column].astype(bool)
    selected = frame.loc[valid].copy()
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected["actual"] = selected[actual_column].astype(float)
    selected["predicted"] = selected[prediction_column].astype(float)
    selected["policy_gain"] = selected[policy_column].astype(float)
    selected["policy_replace"] = selected[replace_column].astype(bool)
    daily = (
        selected.groupby(["reference_name", "date_idx", "state_year"], sort=True)
        .agg(
            rows=("actual", "size"),
            actual_mean=("actual", "mean"),
            predicted_mean=("predicted", "mean"),
            squared_error=("actual", lambda value: float(np.mean(np.square(value)))),
            absolute_error=("actual", lambda value: float(np.mean(np.abs(value)))),
            policy_gain=("policy_gain", "mean"),
            replace_fraction=("policy_replace", "mean"),
            oracle_gain=("actual", lambda value: float(np.mean(np.maximum(value, 0.0)))),
        )
        .reset_index()
    )
    # Grouped reductions above need the prediction explicitly for MSE and rank.
    score_rows: list[dict[str, Any]] = []
    for (reference, date_idx, year), group in selected.groupby(
        ["reference_name", "date_idx", "state_year"], sort=True
    ):
        actual = group["actual"].to_numpy(float)
        predicted = group["predicted"].to_numpy(float)
        score_rows.append(
            {
                "reference_name": str(reference),
                "date_idx": int(date_idx),
                "state_year": int(year),
                "rows": len(group),
                "actual_mean": float(np.mean(actual)),
                "predicted_mean": float(np.mean(predicted)),
                "mse": float(np.mean(np.square(actual - predicted))),
                "mae": float(np.mean(np.abs(actual - predicted))),
                "rank_correlation": _safe_rank_corr(actual, predicted),
                "policy_gain": float(group["policy_gain"].mean()),
                "replace_fraction": float(group["policy_replace"].mean()),
                "oracle_gain": float(np.maximum(actual, 0.0).mean()),
                "always_replace_gain": float(actual.mean()),
            }
        )
    return daily, pd.DataFrame(score_rows)


def evaluate_predictions(
    *,
    material: Material,
    prediction_paths: Mapping[int, Path],
    output_root: Path,
) -> dict[str, Any]:
    frames = [pd.read_parquet(path) for path in prediction_paths.values()]
    predictions = pd.concat(frames, ignore_index=True)
    del frames
    decomposition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    scopes = {
        "strict_2020_2023": {
            int(v) for v in material.study["period"]["strict_oos_years"]
        },
        "near_complete_2020_2024": {
            int(v) for v in material.study["period"]["near_complete_years"]
        },
        "all_2020_2025": {
            int(v)
            for v in material.study["period"]["strict_oos_years"]
            + material.study["period"]["provisional_years"]
        }
        | {2024},
    }
    # The prior model is the date-matched group mean under the same reference.
    for scope_name, years in scopes.items():
        scoped = predictions[predictions["state_year"].isin(years)]
        for reference in REFERENCE_NAMES:
            reference_frame = scoped[scoped["reference_name"].eq(reference)]
            for cost in COST_SCENARIOS:
                _, prior_scores = _daily_action_metrics(
                    reference_frame, cost_scenario=cost, model="prior"
                )
                if prior_scores.empty:
                    continue
                prior_mse = float(prior_scores["mse"].mean())
                for model in MODEL_NAMES:
                    _, scores = _daily_action_metrics(
                        reference_frame, cost_scenario=cost, model=model
                    )
                    if scores.empty:
                        continue
                    # Use equal-date weighting for headline errors and policy
                    # utility; this prevents a large cross-section dominating.
                    error_gain = prior_mse - float(scores["mse"].mean())
                    policy_hac = _hac_mean(scores["policy_gain"].to_numpy(float))
                    summary_rows.append(
                        {
                            "scope": scope_name,
                            "reference_name": reference,
                            "cost_scenario": cost,
                            "model": model,
                            "dates": len(scores),
                            "rows": int(scores["rows"].sum()),
                            "mean_squared_error": float(scores["mse"].mean()),
                            "mse_gain_vs_prior": float(error_gain),
                            "mean_absolute_error": float(scores["mae"].mean()),
                            "mean_daily_rank_correlation": float(
                                scores["rank_correlation"].mean()
                            ),
                            "mean_policy_gain": float(scores["policy_gain"].mean()),
                            "policy_gain_hac_se": float(policy_hac["se"]),
                            "policy_gain_lcb_95": float(policy_hac["lcb_95"]),
                            "policy_gain_ucb_95": float(policy_hac["ucb_95"]),
                            "mean_oracle_gain": float(scores["oracle_gain"].mean()),
                            "mean_always_replace_gain": float(
                                scores["always_replace_gain"].mean()
                            ),
                            "mean_replace_fraction": float(
                                scores["replace_fraction"].mean()
                            ),
                            "blocked_identity_fraction": float(
                                reference_frame[
                                    f"blocked_action_identity_{cost}"
                                ].astype(bool).mean()
                            ),
                            "profit_claim_allowed": False,
                        }
                    )
                _, model_scores = _daily_action_metrics(
                    reference_frame, cost_scenario=cost, model="geometry_position"
                )
                if not model_scores.empty:
                    model_scores["scope"] = scope_name
                    model_scores["cost_scenario"] = cost
                    daily_rows.append(model_scores)
        # Economic decomposition is descriptive and uses the same resolved
        # action table. It is never used to choose a model.
        for reference in REFERENCE_NAMES:
            reference_frame = predictions[
                predictions["reference_name"].eq(reference)
                & predictions["state_year"].isin(years)
            ]
            for cost in COST_SCENARIOS:
                valid = reference_frame[
                    reference_frame[f"action_valid_{cost}"].astype(bool)
                ]
                if valid.empty:
                    continue
                decomposition_rows.append(
                    {
                        "scope": scope_name,
                        "reference_name": reference,
                        "cost_scenario": cost,
                        "rows": len(valid),
                        "hold_log_return": float(
                            valid[f"hold_log_return_{cost}"].mean()
                        ),
                        "net_replace_log_return": float(
                            valid[f"replace_log_return_{cost}"].mean()
                        ),
                        "delta_log_value": float(
                            valid[f"delta_log_value_{cost}"].mean()
                        ),
                        "sale_log_return": float(
                            valid[f"sale_log_return_{cost}"].mean()
                        ),
                        "candidate_gross_log_return": float(
                            valid[f"candidate_gross_log_return_{cost}"].mean()
                        ),
                        "sell_cost_cny_mean": float(
                            valid[f"sell_cost_cny_{cost}"].mean()
                        ),
                        "buy_cost_cny_mean": float(
                            valid[f"buy_cost_cny_{cost}"].mean()
                        ),
                        "blocked_identity_fraction": float(
                            valid[f"blocked_action_identity_{cost}"].astype(bool).mean()
                        ),
                        "cash_no_candidate_fraction": float(
                            valid[f"replacement_status_{cost}"]
                            .astype(str)
                            .eq("cash_no_candidate")
                            .mean()
                        ),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    decomposition = pd.DataFrame(decomposition_rows)
    _write_frame(output_root / "evaluation_summary.parquet", summary)
    _write_frame(output_root / "daily_metrics.parquet", daily)
    _write_frame(output_root / "decomposition.parquet", decomposition)
    manifest = {
        "schema": "seq100_daily_action_evaluation/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "prediction_manifest": _file_record(output_root / "prediction_manifest.json"),
        "summary": _file_record(output_root / "evaluation_summary.parquet"),
        "daily_metrics": _file_record(output_root / "daily_metrics.parquet"),
        "decomposition": _file_record(output_root / "decomposition.parquet"),
        "strict_rows": int(
            predictions["state_year"].isin(
                {int(v) for v in material.study["period"]["strict_oos_years"]}
            ).sum()
        ),
        "causal_update_violations": int(predictions["causal_update_violation"].sum()),
        "profit_claim_allowed": False,
        "production_policy_selected": False,
    }
    _write_json(output_root / "evaluation_manifest.json", manifest)
    del predictions
    gc.collect()
    return manifest


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = _resolve(study_path)
    output_root = _resolve(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    study = load_study(study_path)
    source = _source_contract(study)
    material = _open_material(study, source)
    state_parts: list[pd.DataFrame] = []
    active_by_reference: dict[str, dict[int, set[int]]] = {}
    for reference_index, reference in enumerate(study["reference_positions"]):
        name = str(reference["name"])
        states, active = _expand_reference_positions(
            material=material,
            reference_name=name,
            trade_path=_resolve(reference["trade_file"]),
            reference_index=reference_index,
        )
        state_parts.append(states)
        active_by_reference[name] = active
    states = pd.concat(state_parts, ignore_index=True)
    states = _load_state_features(material=material, states=states)
    rankings = _build_replacement_rankings(material)
    outcome_manifest = build_outcomes(
        material=material,
        states=states,
        rankings=rankings,
        active_by_reference=active_by_reference,
        output_root=output_root,
    )
    outcome_paths = _outcome_paths(output_root, study)
    # Re-read the frozen outcome partitions so prediction and validation use
    # precisely the persisted state, rather than an in-memory intermediate.
    outcome_frames = [pd.read_parquet(path) for path in outcome_paths.values()]
    outcome_frame = pd.concat(outcome_frames, ignore_index=True)
    prediction_manifest = _build_prediction_rows(
        material=material,
        states_frame=outcome_frame,
        outcome_paths=outcome_paths,
        output_root=output_root,
    )
    prediction_paths = _prediction_paths(output_root)
    evaluate_predictions(
        material=material,
        prediction_paths=prediction_paths,
        output_root=output_root,
    )
    manifest = {
        "schema": "seq100_daily_action_value/1",
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "study": _file_record(study_path),
        "source": {
            "phase_analysis_manifest": _file_record(source["phase_manifest_path"]),
            "phase_belief_manifest": _file_record(source["belief_manifest_path"]),
            "continuation_prediction_manifest": _file_record(
                source["continuation_manifest_path"]
            ),
            "corrected_pack_manifest": _file_record(source["pack_manifest_path"]),
            "quality_pool_name": "quality_liquidity_pit",
        },
        "period": dict(study["period"]),
        "reference_positions": [dict(value) for value in study["reference_positions"]],
        "outcome_manifest": _file_record(output_root / "outcome_manifest.json"),
        "prediction_manifest": _file_record(output_root / "prediction_manifest.json"),
        "evaluation_manifest": _file_record(output_root / "evaluation_manifest.json"),
        "audit": {
            "state_rows": len(states),
            "outcome_rows": int(outcome_manifest["rows"]),
            "prediction_rows": int(prediction_manifest["rows"]),
            "causal_update_violations": int(
                prediction_manifest["causal_update_violations"]
            ),
            "forbidden_2026_rows": int(outcome_manifest["forbidden_2026_rows"]),
            "candidate_ranking_dates": len(rankings),
        },
        "fixed_total_holding_horizon_used": False,
        "training_performed": True,
        "portfolio_execution_performed": False,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    _write_json(output_root / "analysis_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_study(study_path=args.study, output_root=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "study_id": result["study_id"],
                "state_rows": result["audit"]["state_rows"],
                "outcome_rows": result["audit"]["outcome_rows"],
                "prediction_rows": result["audit"]["prediction_rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
