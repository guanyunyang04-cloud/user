from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from daily_research.path_policy.seq100_qcurve import QCurveCostContract
from daily_research.path_policy.seq100_qcurve_backtest import (
    PortfolioState,
    _benchmark_log_return,
    _buy_order,
    _equity,
    _sell_order,
)
from daily_research.path_policy.seq100_qcurve_data import DEFAULT_PACK_MANIFEST, QCurvePack


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OLD_STUDY_ROOT = Path(
    "daily_research/output/path_policy/studies/"
    "seq100_candidate_complete_development_walkforward_20260712_v3/runs/development"
)
DEFAULT_OUTPUT = Path(
    "daily_research/output/path_policy/studies/seq100_dynamic_qcurve_2022_2025/"
    "zero_training_old_baseline_exit_audit.json"
)
YEARS = (2022, 2023, 2024, 2025)
TOPKS = (1, 3, 5, 10)
FIXED_HORIZONS = (2, 5, 10, 20, 40, 60)


def _workspace_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _old_topk_path(year: int) -> Path:
    matches = sorted(
        _workspace_path(OLD_STUDY_ROOT).glob(
            f"seq100_development_baseline_{int(year)}_seed7_*/topk_candidates.parquet"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"expected exactly one frozen baseline TopK file for {year}, found {len(matches)}")
    return matches[0]


def _date_candidate_lookup(pack: QCurvePack, date_idx: int) -> tuple[int, dict[int, int]]:
    start, stop = pack.span(date_idx)
    symbols = pack.candidate_symbols(start, stop)
    return start, {int(symbol): start + offset for offset, symbol in enumerate(symbols)}


def _selected_policy_values(
    *,
    rows: pd.DataFrame,
    actual: np.ndarray,
    universe: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if policy.startswith("fixed_h"):
        horizon = int(policy.removeprefix("fixed_h"))
        indices = np.full(len(rows), horizon - 2, dtype=np.int64)
        realized = actual[np.arange(len(rows)), indices]
        benchmark = np.full(len(rows), float(np.mean(universe[:, horizon - 2])), dtype=np.float64)
    elif policy == "old_predicted_exit":
        horizons = np.clip(np.rint(rows["predicted_exit_day"].to_numpy(dtype=np.float64)), 2, 60).astype(np.int64)
        indices = horizons - 2
        realized = actual[np.arange(len(rows)), indices]
        benchmark = np.asarray([np.mean(universe[:, index]) for index in indices], dtype=np.float64)
    elif policy == "oracle_exit":
        indices = np.argmax(actual, axis=1)
        realized = actual[np.arange(len(rows)), indices]
        benchmark = np.full(len(rows), float(np.mean(np.max(universe, axis=1))), dtype=np.float64)
    else:
        raise ValueError(f"unsupported zero-training exit policy: {policy}")
    return realized, realized - benchmark, indices + 2


@dataclass
class _PlannedPosition:
    planned_exit_date_idx: int
    terminal_date_idx: int


def _stateful_policy_replay(
    *,
    pack: QCurvePack,
    topk: pd.DataFrame,
    policy: str,
    slippage_multiplier: float,
) -> dict[str, Any]:
    contract = QCurveCostContract.from_manifest(pack.manifest)
    state = PortfolioState()
    plans: dict[int, _PlannedPosition] = {}
    equities: list[float] = []
    logs: list[float] = []
    benchmarks: list[float] = []
    cash_weights: list[float] = []
    position_counts: list[int] = []
    terminal_losses = 0
    previous_execution_idx: int | None = None
    date_map = {str(value): idx for idx, value in enumerate(pack.date_values.astype(str))}
    symbol_map = {str(value): idx for idx, value in enumerate(pack.symbol_values)}

    for trade_date, raw_rows in topk.groupby("trade_date", sort=True):
        date_idx = int(date_map[str(trade_date)])
        execution_idx = date_idx + 1
        rows = raw_rows.sort_values("score_rank", kind="mergesort")
        for symbol in list(state.positions):
            plan = plans[symbol]
            if execution_idx >= plan.terminal_date_idx and not bool(pack.masks["open_sellable"][execution_idx, symbol]):
                position = state.positions.pop(symbol)
                state.realized_pnls.append(-float(position.basis_cash))
                state.holding_days.append(execution_idx - int(position.entry_date_idx))
                plans.pop(symbol)
                terminal_losses += 1
                continue
            if execution_idx < plan.planned_exit_date_idx or not bool(pack.masks["open_sellable"][execution_idx, symbol]):
                continue
            raw_open = float(pack.execution["open_raw"][execution_idx, symbol])
            _sell_order(
                state,
                symbol=symbol,
                shares=int(state.positions[symbol].shares),
                raw_open=raw_open,
                trade_date=pack.date_values[execution_idx],
                date_idx=execution_idx,
                contract=contract,
                slippage_multiplier=slippage_multiplier,
            )
            plans.pop(symbol)

        vacancies = max(3 - len(state.positions), 0)
        if vacancies:
            proposed = rows[~rows["symbol"].map(symbol_map).isin(state.positions)].head(vacancies)
            start, lookup = _date_candidate_lookup(pack, date_idx)
            span_start, span_stop = pack.span(date_idx)
            enter_curve, _ = pack.q_targets(span_start, span_stop, cost="base")
            equity_before = _equity(pack, state, execution_idx, at_open=True)
            for row in proposed.itertuples(index=False):
                symbol = int(symbol_map[str(row.symbol)])
                candidate_id = lookup.get(symbol)
                if candidate_id is None or not bool(pack.masks["open_buyable"][execution_idx, symbol]):
                    continue
                curve = enter_curve[candidate_id - start]
                if policy.startswith("fixed_h"):
                    horizon = int(policy.removeprefix("fixed_h"))
                elif policy == "old_predicted_exit":
                    horizon = int(np.clip(round(float(row.predicted_exit_day)), 2, 60))
                elif policy == "oracle_exit":
                    horizon = int(np.argmax(curve) + 2)
                else:
                    raise ValueError(policy)
                raw_open = float(pack.execution["open_raw"][execution_idx, symbol])
                desired = int(math.floor((equity_before / 3.0) / raw_open / contract.lot_size) * contract.lot_size)
                trade = _buy_order(
                    state,
                    symbol=symbol,
                    desired_shares=desired,
                    raw_open=raw_open,
                    date_idx=execution_idx,
                    contract=contract,
                    slippage_multiplier=slippage_multiplier,
                )
                if trade:
                    plans[symbol] = _PlannedPosition(
                        planned_exit_date_idx=date_idx + horizon,
                        terminal_date_idx=date_idx + horizon + 20,
                    )
        equity_after = _equity(pack, state, execution_idx, at_open=True)
        previous_equity = equities[-1] if equities else 1_000_000.0
        logs.append(float(math.log(equity_after / previous_equity)))
        benchmarks.append(
            0.0
            if previous_execution_idx is None
            else _benchmark_log_return(pack, previous_execution_idx, execution_idx)
        )
        equities.append(equity_after)
        cash_weights.append(float(state.cash / max(equity_after, 1.0e-12)))
        position_counts.append(len(state.positions))
        previous_execution_idx = execution_idx

    equity_path = np.r_[1_000_000.0, np.asarray(equities, dtype=np.float64)]
    drawdown = equity_path / np.maximum.accumulate(equity_path) - 1.0
    log_values = np.asarray(logs, dtype=np.float64)
    benchmark_values = np.asarray(benchmarks, dtype=np.float64)
    return {
        "ending_equity": float(equities[-1]),
        "net_return": float(equities[-1] / 1_000_000.0 - 1.0),
        "total_log_return": float(log_values.sum()),
        "alpha_total_log_return": float((log_values - benchmark_values).sum()),
        "maximum_drawdown": float(drawdown.min()),
        "fees_paid": float(state.fees_paid),
        "turnover_notional": float(state.traded_notional),
        "average_cash_weight": float(np.mean(cash_weights)),
        "average_position_count": float(np.mean(position_counts)),
        "maximum_position_count": int(max(position_counts, default=0)),
        "terminal_loss_count": int(terminal_losses),
        "closed_trade_count": len(state.realized_pnls),
    }


def run_zero_training_audit(
    *,
    pack_manifest: str | Path = DEFAULT_PACK_MANIFEST,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    pack = QCurvePack(pack_manifest)
    date_map = {str(value): idx for idx, value in enumerate(pack.date_values.astype(str))}
    symbol_map = {str(value): idx for idx, value in enumerate(pack.symbol_values)}
    policies = tuple(f"fixed_h{value}" for value in FIXED_HORIZONS) + ("old_predicted_exit", "oracle_exit")
    yearly: dict[str, Any] = {}
    for year in YEARS:
        source_path = _old_topk_path(year)
        topk = pd.read_parquet(source_path)
        daily_values = {
            policy: {topk_value: [] for topk_value in TOPKS}
            for policy in policies
        }
        for trade_date, raw_rows in topk.groupby("trade_date", sort=True):
            date_idx = int(date_map[str(trade_date)])
            span_start, span_stop = pack.span(date_idx)
            candidate_symbols = pack.candidate_symbols(span_start, span_stop)
            lookup = {int(symbol): offset for offset, symbol in enumerate(candidate_symbols)}
            universe, _ = pack.q_targets(span_start, span_stop, cost="base")
            rows = raw_rows.sort_values("score_rank", kind="mergesort").copy()
            rows["symbol_idx"] = rows["symbol"].map(symbol_map)
            rows = rows[rows["symbol_idx"].isin(lookup)].reset_index(drop=True)
            row_indices = np.asarray([lookup[int(symbol)] for symbol in rows["symbol_idx"]], dtype=np.int64)
            actual = universe[row_indices]
            for policy in policies:
                realized, alpha, horizons = _selected_policy_values(
                    rows=rows,
                    actual=actual,
                    universe=universe,
                    policy=policy,
                )
                for topk_value in TOPKS:
                    count = min(topk_value, len(rows))
                    daily_values[policy][topk_value].append(
                        (float(np.mean(realized[:count])), float(np.mean(alpha[:count])), float(np.mean(horizons[:count])))
                    )
        metric_summary = {
            policy: {
                f"top{topk_value}": {
                    "cost_adjusted_absolute_mean_log_return": float(np.mean(daily_values[policy][topk_value], axis=0)[0]),
                    "cost_adjusted_alpha_mean_log_return": float(np.mean(daily_values[policy][topk_value], axis=0)[1]),
                    "mean_exit_horizon": float(np.mean(daily_values[policy][topk_value], axis=0)[2]),
                }
                for topk_value in TOPKS
            }
            for policy in policies
        }
        stateful = {
            policy: {
                "base": _stateful_policy_replay(pack=pack, topk=topk, policy=policy, slippage_multiplier=1.0),
                "stress": _stateful_policy_replay(
                    pack=pack,
                    topk=topk,
                    policy=policy,
                    slippage_multiplier=QCurveCostContract.from_manifest(pack.manifest).stress_slippage_multiplier,
                ),
            }
            for policy in policies
        }
        yearly[str(year)] = {
            "source_topk_path": str(source_path.resolve()),
            "daily_ranking_exit_comparison": metric_summary,
            "stateful_max3": stateful,
        }
    result = {
        "schema_version": 1,
        "status": "completed",
        "audit": "frozen old baseline ranking under new next-open Q contract; no training",
        "policies": list(policies),
        "years": yearly,
        "notes": {
            "old_predicted_exit_translation": "old close-path predicted day is mapped to the same numbered raw-open Q horizon",
            "no_rank_replacement": True,
            "oracle_role": "diagnostic_only",
            "maximum_positions": 3,
            "position_budget": "one third of current equity per new position",
        },
        "created_at": _now(),
    }
    target = _write_json(output_path, result)
    return {"status": "completed", "output_path": str(target.resolve())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-training old-baseline exit-policy audit for Q-curve v8.")
    parser.add_argument("--pack-manifest", type=Path, default=DEFAULT_PACK_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_zero_training_audit(pack_manifest=args.pack_manifest, output_path=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
