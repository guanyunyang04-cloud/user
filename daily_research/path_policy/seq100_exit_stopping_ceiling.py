"""Future-informed legal stopping ceiling for frozen causal entries.

The ceiling is a diagnostic teacher only.  It asks how much each already
causal entry could earn if a perfect observer selected the best close-time
sale request inside the bounded episode.  It never becomes a policy, label
availability claim, or profit estimate.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil

from daily_research.path_policy import seq100_causal_exit_baselines as baselines
from daily_research.path_policy import seq100_dynamic_oracle as dynamic_oracle
from daily_research.path_policy import seq100_market_replay as replay
from daily_research.path_policy.seq100_exit_policy_audit import (
    _buy_order,
    _stamp_tax_bps_by_date_idx,
)

WORKSPACE_ROOT = baselines.WORKSPACE_ROOT
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_exit_stopping_ceiling_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_exit_stopping_ceiling_v1"
)
COST_SCENARIOS = replay.COST_SCENARIOS


def _resolve(value: str | Path) -> Path:
    return baselines._resolve(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    return baselines._read_json(path)


def _sha256(path: str | Path) -> str:
    return baselines._sha256(path)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    baselines._write_json(path, payload)


def _write_frame(path: str | Path, frame: pd.DataFrame) -> dict[str, Any]:
    return baselines._write_frame(path, frame)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study = _read_json(path)
    if study.get("study_id") != "seq100_exit_stopping_ceiling_v1":
        raise ValueError("stopping_ceiling_study_id")
    source = dict(study.get("source", {}))
    if source.get("quality_pool_name") != "quality_liquidity_pit":
        raise ValueError("stopping_ceiling_quality_pool")
    if int(source.get("forbidden_year", -1)) != baselines.FORBIDDEN_YEAR:
        raise ValueError("stopping_ceiling_forbidden_year")
    if tuple(dict(study.get("execution", {})).get("cost_scenarios", ())) != COST_SCENARIOS:
        raise ValueError("stopping_ceiling_cost_contract")
    if int(dict(study.get("episodes", {})).get("maximum_sessions", 0)) != 40:
        raise ValueError("stopping_ceiling_episode_contract")
    return study


def _legal_request_exit_pairs(
    *,
    market: replay.ReplayMarket,
    entry_abs: int,
    symbol: int,
    liquidation_abs: int,
    maximum_sessions: int,
) -> tuple[np.ndarray, np.ndarray]:
    end_abs = min(int(market.end_idx), int(entry_abs) + int(maximum_sessions))
    days = np.arange(int(entry_abs), end_abs + 1, dtype=np.int32)
    if len(days) < 2:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    last_request = min(int(maximum_sessions) - 1, len(days) - 2)
    terminal_positions = np.flatnonzero(days >= int(liquidation_abs))
    if len(terminal_positions):
        last_request = min(last_request, int(terminal_positions[0]))
    requests = np.arange(0, last_request + 1, dtype=np.int32)
    sellable = np.asarray(market.exit_sellable[days, int(symbol)], dtype=bool)
    prices = np.asarray(market.exit_close_raw[days, int(symbol)], dtype=np.float64)
    legal_positions = np.flatnonzero(sellable & np.isfinite(prices) & (prices > 0.0))
    insertion = np.searchsorted(legal_positions, requests + 1, side="left")
    valid = insertion < len(legal_positions)
    return days[requests[valid]], days[legal_positions[insertion[valid]]]


def _exact_net_returns(
    *,
    market: replay.ReplayMarket,
    entry_price: float,
    exit_days: np.ndarray,
    symbol: int,
    starting_cash: float,
    cost_scenario: str,
    stamp_bps: np.ndarray,
) -> tuple[np.ndarray, bool]:
    multiplier = (
        1.0
        if cost_scenario == "base"
        else float(market.costs.stress_slippage_multiplier)
    )
    shares, buy_cash, _, _ = _buy_order(
        available_cash=float(starting_cash),
        allocated_cash=float(starting_cash),
        entry_price=float(entry_price),
        contract=market.costs,
        slippage_multiplier=multiplier,
    )
    if shares <= 0 or buy_cash <= 0.0:
        return np.zeros(len(exit_days), dtype=np.float64), False
    raw = np.asarray(market.exit_close_raw[exit_days, int(symbol)], dtype=np.float64)
    slippage_rate = float(market.costs.slippage_bps) * multiplier / 10_000.0
    sell_price = raw * (1.0 - slippage_rate)
    notional = int(shares) * sell_price
    commission = np.maximum(
        float(market.costs.minimum_commission_cny),
        notional * float(market.costs.commission_bps) / 10_000.0,
    )
    transfer = notional * float(market.costs.transfer_fee_bps) / 10_000.0
    stamp = notional * stamp_bps[exit_days] / 10_000.0
    proceeds = notional - commission - transfer - stamp
    returns = (float(starting_cash) - buy_cash + proceeds) / float(starting_cash) - 1.0
    return returns.astype(np.float64), True


def _mechanical_best(
    *, root: Path, study: Mapping[str, Any]
) -> pd.DataFrame:
    result_glob = (
        _resolve(study["source"]["mechanical_policy_results_root"])
        / "**"
        / "*.parquet"
    ).as_posix()
    available_mb = int(psutil.virtual_memory().available / 1024**2)
    reserve_mb = int(dict(study["resources"])["reserve_memory_mb"])
    memory_mb = max(min(int(available_mb * 0.6), available_mb - reserve_mb), 512)
    connection = duckdb.connect(database=":memory:")
    connection.execute(f"SET memory_limit='{memory_mb}MB'")
    connection.execute(
        f"SET threads={min(int(dict(study['resources'])['maximum_threads']), max(int(psutil.cpu_count() or 1) - 2, 1))}"
    )
    frame = connection.execute(
        f"""
        SELECT entry_pattern, signal_date_idx, symbol_idx,
               max(net_return_base) AS best_mechanical_base,
               max(net_return_double_slippage) AS best_mechanical_double_slippage
        FROM read_parquet('{result_glob}', hive_partitioning=false)
        GROUP BY entry_pattern, signal_date_idx, symbol_idx
        """
    ).fetchdf()
    connection.close()
    _write_frame(root / "mechanical_best.parquet", frame)
    return frame


def build_ceiling(
    *,
    study: Mapping[str, Any],
    market: replay.ReplayMarket,
    entries: pd.DataFrame,
) -> pd.DataFrame:
    maximum_sessions = int(study["episodes"]["maximum_sessions"])
    liquidation_abs = int(
        np.flatnonzero(
            market.date_values.astype(str)
            == str(study["period"]["terminal_liquidation_start"])
        )[0]
    )
    starting_cash = float(study["execution"]["starting_cash_cny_per_entry"])
    absolute_dates = np.arange(len(market.date_values), dtype=np.int64)
    stamp_bps = _stamp_tax_bps_by_date_idx(
        absolute_dates,
        date_values=market.date_values,
        contract=market.costs,
    ).astype(np.float64)
    records: list[dict[str, Any]] = []
    for position, row in enumerate(entries.to_dict("records")):
        entry_abs = int(row["entry_date_idx"])
        symbol = int(row["symbol_idx"])
        request_days, exit_days = _legal_request_exit_pairs(
            market=market,
            entry_abs=entry_abs,
            symbol=symbol,
            liquidation_abs=liquidation_abs,
            maximum_sessions=maximum_sessions,
        )
        record: dict[str, Any] = {
            "entry_pattern": str(row["entry_pattern"]),
            "symbol_idx": symbol,
            "symbol": str(row["symbol"]),
            "signal_date_idx": int(row["signal_date_idx"]),
            "entry_date_idx": entry_abs,
            "signal_date": str(row["signal_date"]),
            "entry_date": str(row["entry_date"]),
            "signal_year": int(row["signal_year"]),
            "candidate_request_days": len(request_days),
        }
        for cost in COST_SCENARIOS:
            returns, finite_filled = _exact_net_returns(
                market=market,
                entry_price=float(row["entry_price_raw"]),
                exit_days=exit_days,
                symbol=symbol,
                starting_cash=starting_cash,
                cost_scenario=cost,
                stamp_bps=stamp_bps,
            )
            if len(returns) == 0:
                record[f"ceiling_net_return_{cost}"] = -1.0
                record[f"ceiling_request_date_idx_{cost}"] = -1
                record[f"ceiling_exit_date_idx_{cost}"] = -1
                record[f"ceiling_request_session_{cost}"] = -1
                record[f"ceiling_holding_sessions_{cost}"] = -1
                record[f"finite_order_filled_{cost}"] = bool(finite_filled)
                continue
            best = int(np.argmax(returns))
            record[f"ceiling_net_return_{cost}"] = float(returns[best])
            record[f"ceiling_request_date_idx_{cost}"] = int(request_days[best])
            record[f"ceiling_exit_date_idx_{cost}"] = int(exit_days[best])
            record[f"ceiling_request_session_{cost}"] = int(
                request_days[best] - entry_abs + 1
            )
            record[f"ceiling_holding_sessions_{cost}"] = int(
                exit_days[best] - entry_abs + 1
            )
            record[f"finite_order_filled_{cost}"] = bool(finite_filled)
        records.append(record)
        if (position + 1) % 25_000 == 0:
            print(
                json.dumps(
                    {
                        "processed_entries": position + 1,
                        "available_memory_mb": int(
                            psutil.virtual_memory().available / 1024**2
                        ),
                    }
                ),
                flush=True,
            )
    return pd.DataFrame.from_records(records)


def _summaries(
    ceiling: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = {int(value) for value in study["period"]["primary_years"]}
    work = ceiling.loc[ceiling["signal_year"].isin(primary)].copy()
    lag = int(study["evaluation"]["hac_lag"])
    summary_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for pattern, group in work.groupby("entry_pattern", sort=True):
        for cost in COST_SCENARIOS:
            column = f"ceiling_net_return_{cost}"
            daily = group.groupby("signal_date", sort=True)[column].mean()
            mean, se = baselines._hac(daily.to_numpy(float), lag)
            annual = daily.groupby(daily.index.astype(str).str[:4].astype(int)).mean()
            for year, value in annual.items():
                annual_rows.append(
                    {
                        "entry_pattern": pattern,
                        "cost_scenario": cost,
                        "signal_year": int(year),
                        "mean_ceiling_net_return": float(value),
                    }
                )
            summary_rows.append(
                {
                    "entry_pattern": pattern,
                    "cost_scenario": cost,
                    "entries": len(group),
                    "dates": int(group["signal_date"].nunique()),
                    "mean_ceiling_net_return": mean,
                    "hac_se": se,
                    "hac_lcb95": mean - 1.96 * se
                    if math.isfinite(se)
                    else math.nan,
                    "trade_weighted_ceiling_net_return": float(group[column].mean()),
                    "positive_entry_fraction": float((group[column] > 0.0).mean()),
                    "median_best_request_session": float(
                        group[f"ceiling_request_session_{cost}"].replace(-1, np.nan).median()
                    ),
                    "median_best_holding_sessions": float(
                        group[f"ceiling_holding_sessions_{cost}"].replace(-1, np.nan).median()
                    ),
                    "positive_year_count": int((annual > 0.0).sum()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(annual_rows)


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    study_file = _resolve(study_path)
    study = load_study(study_file)
    root = _resolve(output_root or study["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not force:
        return _read_json(manifest_path)
    baseline_manifest = _resolve(study["source"]["mechanical_manifest"])
    if _sha256(baseline_manifest) != str(
        study["source"]["expected_mechanical_manifest_sha256"]
    ):
        raise ValueError("stopping_ceiling_mechanical_manifest_hash")
    entry_path = _resolve(study["source"]["entry_records"])
    if _sha256(entry_path) != str(study["source"]["expected_entry_records_sha256"]):
        raise ValueError("stopping_ceiling_entry_hash")
    policy_manifest = _resolve(study["source"]["mechanical_policy_results_manifest"])
    if _sha256(policy_manifest) != str(
        study["source"]["expected_mechanical_policy_results_manifest_sha256"]
    ):
        raise ValueError("stopping_ceiling_policy_manifest_hash")
    oracle_study = dynamic_oracle.load_study(
        _resolve(study["source"]["dynamic_oracle_study"])
    )
    market, market_audit = replay.load_market(oracle_study)
    entries = pd.read_parquet(entry_path)
    available_start = int(psutil.virtual_memory().available / 1024**2)
    ceiling = build_ceiling(study=study, market=market, entries=entries)
    mechanical = _mechanical_best(root=root, study=study)
    ceiling = ceiling.merge(
        mechanical,
        on=["entry_pattern", "signal_date_idx", "symbol_idx"],
        how="left",
        validate="one_to_one",
    )
    for cost in COST_SCENARIOS:
        ceiling[f"ceiling_increment_vs_mechanical_{cost}"] = (
            ceiling[f"ceiling_net_return_{cost}"]
            - ceiling[f"best_mechanical_{cost}"]
        )
    summary, annual = _summaries(ceiling, study)
    outputs = {
        "ceiling": _write_frame(root / "stopping_ceiling.parquet", ceiling),
        "summary": _write_frame(root / "summary.parquet", summary),
        "annual_summary": _write_frame(root / "annual_summary.parquet", annual),
    }
    gate = {
        "status": "diagnostic_only",
        "account_replay_allowed": False,
        "production_policy_allowed": False,
        "reason": "future-informed stopping ceiling; not a causal policy or profit estimate",
    }
    _write_json(root / "promotion_gate.json", gate)
    manifest = {
        "schema": "seq100_exit_stopping_ceiling_manifest/1",
        "status": "completed",
        "study_id": study["study_id"],
        "study": {"path": str(study_file), "sha256": _sha256(study_file)},
        "market_audit": market_audit,
        "entries": len(entries),
        "outputs": outputs,
        "runtime": {
            "elapsed_seconds": float(time.perf_counter() - started),
            "minimum_available_memory_mb": min(
                available_start, int(psutil.virtual_memory().available / 1024**2)
            ),
            "available_memory_mb_at_end": int(
                psutil.virtual_memory().available / 1024**2
            ),
        },
        "boundaries": dict(study["boundaries"]),
        "promotion_gate": gate,
        "future_informed_teacher": True,
        "causal_policy_evaluated": False,
        "profit_claim_allowed": False,
    }
    _write_json(manifest_path, manifest)
    del market
    gc.collect()
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build legal stopping ceiling.")
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = run_study(
        study_path=args.study,
        output_root=args.output_root,
        force=bool(args.force),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=baselines._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
