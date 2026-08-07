"""Independent cash-flow validation and D20 exposure diagnostics.

The validator treats the completed phase-execution run as an external artifact.
It recomputes transaction cash flows, daily account states, annual metrics, and
dynamic-control differences from the retained parquet files.  It also compares
the apparent D20 return with the contemporaneous point-in-time quality pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_daily_phase_execution as phase
from daily_research.path_policy.seq100_exit_policy_audit import (
    CandidateCompleteAuditPack,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_ROOT = phase.DEFAULT_OUTPUT_ROOT
VALIDATION_SCHEMA = "seq100_daily_phase_execution_validation/1"
FLOAT_TOLERANCE = 1.0e-5


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    return {
        "path": str(target),
        "size": int(target.stat().st_size),
        "sha256": _sha256(target),
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_frame(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)


def _declared_file_errors(record: Mapping[str, Any]) -> list[str]:
    path = Path(str(record["path"]))
    errors: list[str] = []
    if not path.is_file():
        return [f"missing:{path}"]
    if int(path.stat().st_size) != int(record["size"]):
        errors.append(f"size:{path}")
    if str(record.get("sha256", "")) and _sha256(path) != str(record["sha256"]):
        errors.append(f"sha256:{path}")
    return errors


def _stamp_tax_bps(
    exit_dates: np.ndarray, schedule: Sequence[Sequence[Any]]
) -> np.ndarray:
    dates = np.asarray(exit_dates, dtype=str)
    rates = np.full(dates.shape, np.nan, dtype=np.float64)
    for effective_date, rate in schedule:
        rates = np.where(dates >= str(effective_date), float(rate), rates)
    if bool(np.isnan(rates).any()):
        raise ValueError("validation_stamp_schedule_does_not_cover_all_exits")
    return rates


def _trade_cashflows(
    trades: pd.DataFrame,
    *,
    costs: Any,
    cost_scenario: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if trades.empty:
        return trades.copy(), {
            "maximum_trade_field_difference": 0.0,
            "recomputed_total_cost_cny": 0.0,
            "recomputed_turnover_notional_cny": 0.0,
        }
    multiplier = (
        1.0 if cost_scenario == "base" else float(costs.stress_slippage_multiplier)
    )
    slippage = float(costs.slippage_bps) * multiplier / 10_000.0
    shares = trades["shares"].to_numpy(np.int64)
    entry_raw = trades["entry_price_raw"].to_numpy(np.float64)
    exit_raw = trades["exit_price_raw"].to_numpy(np.float64)

    buy_price = entry_raw * (1.0 + slippage)
    buy_notional = shares * buy_price
    buy_commission = np.maximum(
        float(costs.minimum_commission_cny),
        buy_notional * float(costs.commission_bps) / 10_000.0,
    )
    buy_transfer = buy_notional * float(costs.transfer_fee_bps) / 10_000.0
    buy_cash = buy_notional + buy_commission + buy_transfer

    sell_price = exit_raw * (1.0 - slippage)
    sell_notional = shares * sell_price
    sell_commission = np.maximum(
        float(costs.minimum_commission_cny),
        sell_notional * float(costs.commission_bps) / 10_000.0,
    )
    sell_transfer = sell_notional * float(costs.transfer_fee_bps) / 10_000.0
    stamp_bps = _stamp_tax_bps(
        trades["exit_date"].astype(str).to_numpy(), costs.stamp_tax_schedule
    )
    stamp_tax = sell_notional * stamp_bps / 10_000.0
    proceeds = sell_notional - sell_commission - sell_transfer - stamp_tax
    net_pnl = proceeds - buy_cash
    net_return = proceeds / buy_cash - 1.0
    explicit_cost = (
        buy_commission + buy_transfer + sell_commission + sell_transfer + stamp_tax
    )
    slippage_cost = shares * ((buy_price - entry_raw) + (exit_raw - sell_price))
    total_cost = explicit_cost + slippage_cost

    expected = {
        "buy_notional_cny": buy_notional,
        "buy_cash_cny": buy_cash,
        "sell_proceeds_cny": proceeds,
        "net_pnl_cny": net_pnl,
        "net_return_on_buy_cash": net_return,
    }
    differences = [
        np.max(np.abs(trades[column].to_numpy(np.float64) - values))
        for column, values in expected.items()
    ]
    result = trades.copy()
    result["recomputed_total_cost_cny"] = total_cost
    result["recomputed_turnover_notional_cny"] = buy_notional + sell_notional
    return result, {
        "maximum_trade_field_difference": float(max(differences, default=0.0)),
        "recomputed_total_cost_cny": float(total_cost.sum()),
        "recomputed_turnover_notional_cny": float((buy_notional + sell_notional).sum()),
    }


def _cash_and_position_paths(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    starting_cash: float,
) -> tuple[np.ndarray, np.ndarray]:
    dates = equity["date_idx"].to_numpy(np.int64)
    if len(dates) == 0 or not np.array_equal(dates, np.arange(dates[0], dates[-1] + 1)):
        raise ValueError("validation_equity_dates_are_not_contiguous")
    first = int(dates[0])
    cash_event = np.zeros(len(dates), dtype=np.float64)
    position_delta = np.zeros(len(dates) + 1, dtype=np.int64)
    for row in trades.itertuples(index=False):
        entry_offset = int(row.entry_date_idx) - first
        exit_offset = int(row.exit_date_idx) - first
        cash_event[entry_offset] -= float(row.buy_cash_cny)
        cash_event[exit_offset] += float(row.sell_proceeds_cny)
        position_delta[entry_offset] += 1
        position_delta[exit_offset] -= 1
    return (
        float(starting_cash) + np.cumsum(cash_event),
        np.cumsum(position_delta[:-1]),
    )


def _equity_path_from_trades(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    expected_cash: np.ndarray,
    pack: CandidateCompleteAuditPack,
) -> np.ndarray:
    first = int(equity["date_idx"].iloc[0])
    position_value = np.zeros(len(equity), dtype=np.float64)
    for row in trades.itertuples(index=False):
        entry_idx = int(row.entry_date_idx)
        exit_idx = int(row.exit_date_idx)
        if exit_idx <= entry_idx:
            raise ValueError("validation_nonpositive_holding_interval")
        raw_marks = np.asarray(
            pack.exit_close_raw[entry_idx:exit_idx, int(row.symbol_idx)],
            dtype=np.float64,
        )
        valid = np.isfinite(raw_marks) & (raw_marks >= 0.0)
        marks = np.empty(len(raw_marks), dtype=np.float64)
        last_mark = float(row.entry_price_raw)
        for index, value in enumerate(raw_marks):
            if bool(valid[index]):
                last_mark = float(value)
            marks[index] = last_mark
        start = entry_idx - first
        position_value[start : start + len(marks)] += int(row.shares) * marks
    return expected_cash + position_value


def _annual_metrics(
    equity: pd.DataFrame, *, starting_cash: float, years: Sequence[int]
) -> pd.DataFrame:
    dates = pd.to_datetime(equity["trade_date"], errors="raise")
    values = equity["equity"].to_numpy(np.float64)
    rows: list[dict[str, Any]] = []
    previous = float(starting_cash)
    for year in years:
        mask = dates.dt.year.to_numpy() == int(year)
        if not bool(mask.any()):
            continue
        before = dates.dt.year.to_numpy() < int(year)
        if bool(before.any()):
            previous = float(values[np.flatnonzero(before)[-1]])
        year_frame = equity.loc[mask]
        year_values = year_frame["equity"].to_numpy(np.float64)
        path = np.r_[previous, year_values]
        returns = path[1:] / path[:-1] - 1.0
        drawdown = path / np.maximum.accumulate(path) - 1.0
        volatility = (
            float(np.std(returns, ddof=1) * math.sqrt(252.0))
            if len(returns) > 1
            else 0.0
        )
        sharpe = (
            float(np.mean(returns) / np.std(returns, ddof=1) * math.sqrt(252.0))
            if len(returns) > 1 and float(np.std(returns, ddof=1)) > 0.0
            else 0.0
        )
        rows.append(
            {
                "year": int(year),
                "starting_equity_cny": previous,
                "ending_equity_cny": float(year_values[-1]),
                "net_return": float(year_values[-1] / previous - 1.0),
                "log_growth": float(math.log(year_values[-1] / previous)),
                "maximum_drawdown": float(drawdown.min()),
                "annualized_volatility": volatility,
                "sharpe_zero_rate": sharpe,
                "mean_position_count": float(year_frame["position_count"].mean()),
                "mean_capital_utilization": float(
                    year_frame["capital_utilization"].mean()
                ),
                "trading_session_count": len(year_frame),
            }
        )
        previous = float(year_values[-1])
    return pd.DataFrame(rows)


def _maximum_frame_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    keys: Sequence[str],
    columns: Sequence[str],
) -> float:
    merged = left[list(keys) + list(columns)].merge(
        right[list(keys) + list(columns)],
        on=list(keys),
        how="outer",
        suffixes=("_left", "_right"),
        indicator=True,
    )
    if not bool(merged["_merge"].eq("both").all()):
        return math.inf
    differences: list[float] = []
    for column in columns:
        first = pd.to_numeric(merged[f"{column}_left"], errors="coerce")
        second = pd.to_numeric(merged[f"{column}_right"], errors="coerce")
        both_nan = first.isna() & second.isna()
        mismatch = (first.isna() ^ second.isna()) & ~both_nan
        if bool(mismatch.any()):
            return math.inf
        finite = ~(first.isna() | second.isna())
        if bool(finite.any()):
            differences.append(
                float(
                    np.max(np.abs(first[finite].to_numpy() - second[finite].to_numpy()))
                )
            )
    return float(max(differences, default=0.0))


def _hac_mean(values: np.ndarray, *, lag: int = 10) -> dict[str, float]:
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    if len(series) == 0:
        return {
            "mean": math.nan,
            "se": math.nan,
            "lcb_95": math.nan,
            "ucb_95": math.nan,
        }
    centered = series - float(series.mean())
    long_variance = float(np.dot(centered, centered) / len(series))
    for offset in range(1, min(int(lag), len(series) - 1) + 1):
        weight = 1.0 - offset / (int(lag) + 1.0)
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / len(series))
        long_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_variance, 0.0) / len(series))
    mean = float(series.mean())
    return {
        "mean": mean,
        "se": standard_error,
        "lcb_95": mean - 1.96 * standard_error,
        "ucb_95": mean + 1.96 * standard_error,
    }


def _newey_west_regression(
    outcome: np.ndarray, market_return: np.ndarray, *, lag: int = 10
) -> dict[str, float]:
    y = np.asarray(outcome, dtype=np.float64)
    x = np.asarray(market_return, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(x)
    y = y[valid]
    x = x[valid]
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    inverse = np.linalg.pinv(design.T @ design)
    coefficient = inverse @ design.T @ y
    residual = y - design @ coefficient
    score = design * residual[:, None]
    meat = score.T @ score
    for offset in range(1, min(int(lag), len(y) - 1) + 1):
        weight = 1.0 - offset / (int(lag) + 1.0)
        cross = score[offset:].T @ score[:-offset]
        meat += weight * (cross + cross.T)
    covariance = inverse @ meat @ inverse
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "daily_alpha": float(coefficient[0]),
        "daily_alpha_hac_se": float(standard_error[0]),
        "market_beta": float(coefficient[1]),
        "market_beta_hac_se": float(standard_error[1]),
    }


def _quality_pool_benchmark(panel_manifest: Mapping[str, Any]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for record in panel_manifest["panels"]:
        year = int(record["year"])
        if year < 2019:
            continue
        frame = pd.read_parquet(
            Path(str(record["path"])),
            columns=["trade_date", "date_idx", "cumret_1"],
        )
        frame["simple_return"] = np.expm1(frame["cumret_1"].astype(np.float64))
        parts.append(
            frame.groupby(["date_idx", "trade_date"], sort=True, as_index=False)[
                "simple_return"
            ].mean()
        )
    benchmark = pd.concat(parts, ignore_index=True).sort_values("date_idx")
    if bool(benchmark["date_idx"].duplicated().any()):
        raise ValueError("validation_quality_benchmark_duplicate_dates")
    return benchmark.reset_index(drop=True)


def _market_exposure_rows(
    *,
    analysis_root: Path,
    tasks: Sequence[Mapping[str, Any]],
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for record in tasks:
        if str(record["profile"]) not in {"geometry_combined", "blend01_combined"}:
            continue
        if str(record["policy"]) not in {"rolling_phase_boundary", "fixed_d20"}:
            continue
        task_root = Path(str(record["path"]))
        metric = _read_json(task_root / "metric.json")
        equity = pd.read_parquet(task_root / "equity.parquet")
        merged = equity.merge(
            benchmark[["date_idx", "simple_return"]],
            on="date_idx",
            how="left",
            validate="one_to_one",
        )
        if bool(merged["simple_return"].isna().any()):
            raise ValueError(f"validation_missing_benchmark_dates:{record['job_id']}")
        strategy = merged["equity"].pct_change().to_numpy(np.float64)
        strategy[0] = (
            float(merged["equity"].iloc[0]) / float(metric["starting_cash_cny"]) - 1.0
        )
        market = merged["simple_return"].to_numpy(np.float64)
        exposure = (
            merged["capital_utilization"].shift(1).fillna(0.0).clip(0.0, 1.0)
        ).to_numpy(np.float64)
        matched = exposure * market
        regression = _newey_west_regression(strategy, market, lag=10)
        rows.append(
            {
                "job_id": str(record["job_id"]),
                "profile": str(record["profile"]),
                "policy": str(record["policy"]),
                "cost_scenario": str(record["cost_scenario"]),
                "sessions": len(merged),
                "strategy_total_return": float(np.prod(1.0 + strategy) - 1.0),
                "quality_pool_total_return": float(np.prod(1.0 + market) - 1.0),
                "exposure_matched_pool_total_return": float(
                    np.prod(1.0 + matched) - 1.0
                ),
                "strategy_minus_matched_log_growth": float(
                    np.log1p(strategy).sum() - np.log1p(matched).sum()
                ),
                "daily_return_correlation": float(np.corrcoef(strategy, market)[0, 1]),
                "annualized_alpha": float(regression["daily_alpha"] * 252.0),
                "annualized_alpha_hac_se": float(
                    regression["daily_alpha_hac_se"] * 252.0
                ),
                "annualized_alpha_lcb_95": float(
                    (
                        regression["daily_alpha"]
                        - 1.96 * regression["daily_alpha_hac_se"]
                    )
                    * 252.0
                ),
                "annualized_alpha_ucb_95": float(
                    (
                        regression["daily_alpha"]
                        + 1.96 * regression["daily_alpha_hac_se"]
                    )
                    * 252.0
                ),
                "market_beta": float(regression["market_beta"]),
                "market_beta_hac_se": float(regression["market_beta_hac_se"]),
            }
        )
        merged["year"] = pd.to_datetime(merged["trade_date"]).dt.year
        merged["strategy_return"] = strategy
        merged["matched_return"] = matched
        for year, group in merged.groupby("year", sort=True):
            annual_rows.append(
                {
                    "job_id": str(record["job_id"]),
                    "profile": str(record["profile"]),
                    "policy": str(record["policy"]),
                    "cost_scenario": str(record["cost_scenario"]),
                    "year": int(year),
                    "strategy_return": float(
                        np.prod(1.0 + group["strategy_return"]) - 1.0
                    ),
                    "exposure_matched_pool_return": float(
                        np.prod(1.0 + group["matched_return"]) - 1.0
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(annual_rows)


def _resolved_d20_pool_return(
    *,
    date_idx: int,
    symbols: np.ndarray,
    pack: CandidateCompleteAuditPack,
) -> tuple[np.ndarray, np.ndarray]:
    symbol_idx = np.asarray(symbols, dtype=np.int64)
    entry_idx = int(date_idx) + 1
    requested_idx = int(date_idx) + 20
    terminal_idx = int(date_idx) + int(pack.execution_days)
    entry = np.asarray(pack.entry_open_raw[entry_idx, symbol_idx], dtype=np.float64)
    valid = (
        np.asarray(pack.entry_filled[int(date_idx), symbol_idx], dtype=bool)
        & np.isfinite(entry)
        & (entry > 0.0)
    )
    exit_price = np.asarray(
        pack.exit_close_raw[requested_idx, symbol_idx], dtype=np.float64
    ).copy()
    resolved = (
        np.asarray(pack.exit_sellable[requested_idx, symbol_idx], dtype=bool)
        & np.isfinite(exit_price)
        & (exit_price >= 0.0)
    )
    for exit_idx in range(requested_idx + 1, terminal_idx + 1):
        pending = valid & ~resolved
        if not bool(pending.any()):
            break
        candidates = np.flatnonzero(pending)
        prices = np.asarray(
            pack.exit_close_raw[exit_idx, symbol_idx[candidates]], dtype=np.float64
        )
        sellable = np.asarray(
            pack.exit_sellable[exit_idx, symbol_idx[candidates]], dtype=bool
        )
        accepted = sellable & np.isfinite(prices) & (prices >= 0.0)
        if bool(accepted.any()):
            positions = candidates[accepted]
            exit_price[positions] = prices[accepted]
            resolved[positions] = True
    valid &= resolved & np.isfinite(exit_price) & (exit_price > 0.0)
    gross_log_return = np.full(len(symbol_idx), np.nan, dtype=np.float64)
    gross_log_return[valid] = np.log(exit_price[valid] / entry[valid])
    return gross_log_return, valid


def _d20_selection_excess(
    *,
    tasks: Sequence[Mapping[str, Any]],
    panel_manifest: Mapping[str, Any],
    pack: CandidateCompleteAuditPack,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_trades: dict[str, pd.DataFrame] = {}
    required_dates: set[int] = set()
    for record in tasks:
        if (
            str(record["policy"]) != "fixed_d20"
            or str(record["cost_scenario"]) != "base"
        ):
            continue
        trades = pd.read_parquet(Path(str(record["path"])) / "trades.parquet")
        task_trades[str(record["profile"])] = trades
        required_dates.update(
            int(value) for value in trades["signal_date_idx"].unique()
        )

    benchmark_rows: list[dict[str, Any]] = []
    symbol_map = {str(value): index for index, value in enumerate(pack.symbol_values)}
    for record in panel_manifest["panels"]:
        year = int(record["year"])
        if year < 2019:
            continue
        panel = pd.read_parquet(
            Path(str(record["path"])), columns=["symbol", "date_idx"]
        )
        panel = panel[panel["date_idx"].isin(required_dates)]
        if panel.empty:
            continue
        panel["symbol_idx"] = panel["symbol"].map(symbol_map)
        if bool(panel["symbol_idx"].isna().any()):
            raise ValueError("validation_quality_symbol_missing_from_execution_pack")
        for date_idx, group in panel.groupby("date_idx", sort=True):
            values, valid = _resolved_d20_pool_return(
                date_idx=int(date_idx),
                symbols=group["symbol_idx"].to_numpy(np.int64),
                pack=pack,
            )
            benchmark_rows.append(
                {
                    "signal_date_idx": int(date_idx),
                    "signal_year": year,
                    "quality_pool_rows": len(group),
                    "quality_pool_valid_rows": int(valid.sum()),
                    "quality_pool_mean_gross_log_return": float(np.nanmean(values)),
                    "quality_pool_median_gross_log_return": float(np.nanmedian(values)),
                }
            )
    benchmark = pd.DataFrame(benchmark_rows)
    date_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for profile, trades in task_trades.items():
        selected = trades.copy()
        selected["selected_gross_log_return"] = np.log(
            selected["exit_price_raw"].astype(float)
            / selected["entry_price_raw"].astype(float)
        )
        daily = (
            selected.groupby("signal_date_idx", sort=True)
            .agg(
                selected_trades=("selected_gross_log_return", "size"),
                selected_mean_gross_log_return=(
                    "selected_gross_log_return",
                    "mean",
                ),
            )
            .reset_index()
            .merge(benchmark, on="signal_date_idx", how="left", validate="one_to_one")
        )
        if bool(daily["quality_pool_mean_gross_log_return"].isna().any()):
            raise ValueError(f"validation_missing_d20_date_benchmark:{profile}")
        daily.insert(0, "profile", profile)
        daily["selected_minus_pool_log_return"] = (
            daily["selected_mean_gross_log_return"]
            - daily["quality_pool_mean_gross_log_return"]
        )
        date_rows.append(daily)
        for scope, years in {
            "strict_2019_2023": set(range(2019, 2024)),
            "all_2019_2025": set(range(2019, 2026)),
        }.items():
            scoped = daily[daily["signal_year"].isin(years)]
            estimate = _hac_mean(
                scoped["selected_minus_pool_log_return"].to_numpy(np.float64),
                lag=10,
            )
            annual = scoped.groupby("signal_year", sort=True)[
                "selected_minus_pool_log_return"
            ].mean()
            summary_rows.append(
                {
                    "profile": profile,
                    "scope": scope,
                    "signal_dates": len(scoped),
                    "trades": int(scoped["selected_trades"].sum()),
                    "selected_mean_gross_log_return": float(
                        scoped["selected_mean_gross_log_return"].mean()
                    ),
                    "quality_pool_mean_gross_log_return": float(
                        scoped["quality_pool_mean_gross_log_return"].mean()
                    ),
                    "selected_minus_pool_log_return": float(estimate["mean"]),
                    "excess_hac_se": float(estimate["se"]),
                    "excess_lcb_95": float(estimate["lcb_95"]),
                    "excess_ucb_95": float(estimate["ucb_95"]),
                    "positive_years": int(annual.gt(0.0).sum()),
                    "evaluated_years": len(annual),
                }
            )
    return pd.concat(date_rows, ignore_index=True), pd.DataFrame(summary_rows)


def _trade_concentration(tasks: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in tasks:
        if (
            str(record["policy"]) != "fixed_d20"
            or str(record["cost_scenario"]) != "base"
        ):
            continue
        trades = pd.read_parquet(Path(str(record["path"])) / "trades.parquet")
        pnl = trades["net_pnl_cny"].to_numpy(np.float64)
        positive = pnl[pnl > 0.0]
        negative = pnl[pnl < 0.0]
        ordered = np.sort(positive)[::-1]
        positive_total = float(positive.sum())
        row: dict[str, Any] = {
            "profile": str(record["profile"]),
            "trades": len(trades),
            "total_net_pnl_cny": float(pnl.sum()),
            "positive_pnl_cny": positive_total,
            "negative_pnl_cny": float(negative.sum()),
            "profit_factor": (
                float(positive_total / -negative.sum())
                if len(negative) and float(negative.sum()) < 0.0
                else math.inf
            ),
            "winner_pnl_hhi": (
                float(np.square(positive / positive_total).sum())
                if positive_total > 0.0
                else math.nan
            ),
        }
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, math.ceil(len(trades) * fraction))
            top = float(ordered[:count].sum())
            suffix = f"top_{int(fraction * 100)}pct"
            row[f"{suffix}_positive_pnl_share"] = (
                top / positive_total if positive_total > 0.0 else math.nan
            )
            row[f"net_pnl_without_{suffix}_cny"] = float(pnl.sum() - top)
        rows.append(row)
    return pd.DataFrame(rows)


def run_validation(
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
) -> dict[str, Any]:
    analysis_root = Path(analysis_root).resolve()
    manifest_path = analysis_root / "analysis_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("phase_execution_analysis_is_not_completed")
    if manifest.get("study_id") != phase.STUDY_ID:
        raise ValueError("phase_execution_validation_study_id_mismatch")

    declared_errors: list[str] = []
    for record in manifest["tasks"]:
        for field in ("metric", "equity", "trades"):
            declared_errors.extend(_declared_file_errors(record[field]))
    for record in manifest["outputs"].values():
        declared_errors.extend(_declared_file_errors(record))

    pack_manifest_path = Path(
        str(manifest["source"]["corrected_pack_manifest"]["path"])
    )
    pack = CandidateCompleteAuditPack(pack_manifest_path)
    years = tuple(int(value) for value in manifest["period"]["belief_years"])
    first_signal_idx = int(manifest["period"]["first_signal_idx"])
    last_signal_idx = int(manifest["period"]["last_signal_idx"])

    task_rows: list[dict[str, Any]] = []
    annual_recomputed: list[pd.DataFrame] = []
    paired_event_frames: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for record in manifest["tasks"]:
        task_root = Path(str(record["path"]))
        metric = _read_json(task_root / "metric.json")
        equity = pd.read_parquet(task_root / "equity.parquet")
        trades = pd.read_parquet(task_root / "trades.parquet")
        _recomputed_trades, cashflow = _trade_cashflows(
            trades, costs=pack.costs, cost_scenario=str(record["cost_scenario"])
        )
        expected_cash, expected_positions = _cash_and_position_paths(
            equity, trades, starting_cash=float(metric["starting_cash_cny"])
        )
        expected_equity = _equity_path_from_trades(
            equity,
            trades,
            expected_cash=expected_cash,
            pack=pack,
        )
        annual = _annual_metrics(
            equity, starting_cash=float(metric["starting_cash_cny"]), years=years
        )
        stored_annual = pd.read_parquet(task_root / "annual.parquet")
        annual_difference = _maximum_frame_difference(
            stored_annual,
            annual,
            keys=["year"],
            columns=[
                "starting_equity_cny",
                "ending_equity_cny",
                "net_return",
                "log_growth",
                "maximum_drawdown",
                "annualized_volatility",
                "sharpe_zero_rate",
                "mean_position_count",
                "mean_capital_utilization",
                "trading_session_count",
            ],
        )
        signal = equity[equity["date_idx"].between(first_signal_idx, last_signal_idx)]
        signal_path = np.r_[float(metric["starting_cash_cny"]), signal["equity"]]
        full_path = np.r_[float(metric["starting_cash_cny"]), equity["equity"]]
        metric_checks = {
            "signal_period_ending_equity_cny": float(signal["equity"].iloc[-1]),
            "signal_period_total_return": float(
                signal["equity"].iloc[-1] / float(metric["starting_cash_cny"]) - 1.0
            ),
            "liquidated_ending_equity_cny": float(equity["equity"].iloc[-1]),
            "liquidated_total_return": float(
                equity["equity"].iloc[-1] / float(metric["starting_cash_cny"]) - 1.0
            ),
            "signal_period_maximum_drawdown": float(
                (signal_path / np.maximum.accumulate(signal_path) - 1.0).min()
            ),
            "full_path_maximum_drawdown": float(
                (full_path / np.maximum.accumulate(full_path) - 1.0).min()
            ),
            "minimum_cash_cny": float(equity["cash"].min()),
            "minimum_equity_cny": float(equity["equity"].min()),
            "closed_trade_count": float(len(trades)),
        }
        metric_difference = max(
            abs(float(metric[key]) - value) for key, value in metric_checks.items()
        )
        total_cost_difference = abs(
            float(metric["fees_and_slippage_cny"])
            - float(cashflow["recomputed_total_cost_cny"])
        )
        turnover_difference = abs(
            float(metric["turnover_notional_cny"])
            - float(cashflow["recomputed_turnover_notional_cny"])
        )
        task_rows.append(
            {
                "job_id": str(record["job_id"]),
                "profile": str(record["profile"]),
                "policy": str(record["policy"]),
                "cost_scenario": str(record["cost_scenario"]),
                "trades": len(trades),
                **cashflow,
                "maximum_cash_path_difference": float(
                    np.max(np.abs(equity["cash"].to_numpy(np.float64) - expected_cash))
                ),
                "maximum_position_count_difference": int(
                    np.max(
                        np.abs(
                            equity["position_count"].to_numpy(np.int64)
                            - expected_positions
                        )
                    )
                ),
                "maximum_equity_path_difference": float(
                    np.max(
                        np.abs(equity["equity"].to_numpy(np.float64) - expected_equity)
                    )
                ),
                "metric_recompute_maximum_difference": float(metric_difference),
                "annual_recompute_maximum_difference": float(annual_difference),
                "total_cost_difference_cny": float(total_cost_difference),
                "turnover_difference_cny": float(turnover_difference),
                "terminal_cash_identity_difference": float(
                    abs(
                        float(equity["cash"].iloc[-1])
                        - (
                            float(metric["starting_cash_cny"])
                            + float(trades["net_pnl_cny"].sum())
                        )
                    )
                ),
            }
        )
        annual.insert(0, "job_id", str(record["job_id"]))
        annual_recomputed.append(annual)
        event_columns = [
            "symbol_idx",
            "signal_date_idx",
            "entry_date_idx",
            "exit_date_idx",
            "exit_reason",
        ]
        paired_event_frames.setdefault(
            (str(record["profile"]), str(record["policy"])), {}
        )[str(record["cost_scenario"])] = (
            trades[event_columns]
            .sort_values(event_columns, kind="mergesort")
            .reset_index(drop=True)
        )

    task_reconciliation = pd.DataFrame(task_rows)
    annual_reconciliation = pd.concat(annual_recomputed, ignore_index=True)
    paired_mismatches = 0
    paired_rows: list[dict[str, Any]] = []
    event_columns = [
        "symbol_idx",
        "signal_date_idx",
        "entry_date_idx",
        "exit_date_idx",
        "exit_reason",
    ]
    for (profile, policy), scenarios in paired_event_frames.items():
        if set(scenarios) != {"base", "double_slippage"}:
            paired_mismatches += 1
            paired_rows.append(
                {
                    "profile": profile,
                    "policy": policy,
                    "base_events": len(scenarios.get("base", [])),
                    "double_slippage_events": len(scenarios.get("double_slippage", [])),
                    "shared_events": 0,
                    "event_jaccard": 0.0,
                    "identical_event_path": False,
                }
            )
            continue
        base = scenarios["base"]
        stress = scenarios["double_slippage"]
        identical = base.equals(stress)
        if not identical:
            paired_mismatches += 1
        shared = len(base.merge(stress, on=event_columns, how="inner"))
        union = len(base) + len(stress) - shared
        paired_rows.append(
            {
                "profile": profile,
                "policy": policy,
                "base_events": len(base),
                "double_slippage_events": len(stress),
                "shared_events": shared,
                "event_jaccard": float(shared / union) if union else 1.0,
                "identical_event_path": bool(identical),
            }
        )
    paired_event_comparison = pd.DataFrame(paired_rows)

    stored_metrics = pd.read_parquet(
        Path(str(manifest["outputs"]["configuration_metrics"]["path"]))
    )
    stored_annual = pd.read_parquet(
        Path(str(manifest["outputs"]["annual_metrics"]["path"]))
    )
    aggregate_annual_difference = _maximum_frame_difference(
        stored_annual.assign(
            job_id=lambda value: (
                value["profile"].astype(str)
                + "__"
                + value["policy"].astype(str)
                + "__"
                + value["cost_scenario"].astype(str)
            )
        ),
        annual_reconciliation,
        keys=["job_id", "year"],
        columns=["net_return", "maximum_drawdown", "annualized_volatility"],
    )
    metric_keys = ["profile", "policy", "cost_scenario"]
    task_metric_frame = pd.DataFrame(
        [
            {
                **{key: metric[key] for key in metric_keys},
                "liquidated_total_return": metric["liquidated_total_return"],
                "liquidated_ending_equity_cny": metric["liquidated_ending_equity_cny"],
            }
            for metric in (
                _read_json(Path(str(record["path"])) / "metric.json")
                for record in manifest["tasks"]
            )
        ]
    )
    aggregate_metric_difference = _maximum_frame_difference(
        stored_metrics,
        task_metric_frame,
        keys=metric_keys,
        columns=["liquidated_total_return", "liquidated_ending_equity_cny"],
    )

    stored_dynamic = pd.read_parquet(
        Path(str(manifest["outputs"]["dynamic_vs_fixed"]["path"]))
    )
    dynamic_rows: list[dict[str, Any]] = []
    for (profile, cost), group in stored_metrics.groupby(
        ["profile", "cost_scenario"], sort=True
    ):
        dynamic = group[group["policy"].eq("rolling_phase_boundary")].iloc[0]
        for control in group[group["policy"].str.startswith("fixed_")].itertuples():
            dynamic_rows.append(
                {
                    "profile": profile,
                    "cost_scenario": cost,
                    "dynamic_policy": "rolling_phase_boundary",
                    "control_policy": str(control.policy),
                    "delta_liquidated_log_growth": float(
                        dynamic["liquidated_log_growth"] - control.liquidated_log_growth
                    ),
                    "delta_signal_period_log_growth": float(
                        math.log1p(dynamic["signal_period_total_return"])
                        - math.log1p(control.signal_period_total_return)
                    ),
                    "dynamic_signal_period_maximum_drawdown": float(
                        dynamic["signal_period_maximum_drawdown"]
                    ),
                    "control_signal_period_maximum_drawdown": float(
                        control.signal_period_maximum_drawdown
                    ),
                }
            )
    dynamic_recomputed = pd.DataFrame(dynamic_rows)
    dynamic_difference = _maximum_frame_difference(
        stored_dynamic,
        dynamic_recomputed,
        keys=["profile", "cost_scenario", "dynamic_policy", "control_policy"],
        columns=[
            "delta_liquidated_log_growth",
            "delta_signal_period_log_growth",
            "dynamic_signal_period_maximum_drawdown",
            "control_signal_period_maximum_drawdown",
        ],
    )

    neighbor_manifest_path = Path(
        str(manifest["source"]["path_neighbor_manifest"]["path"])
    )
    panel_manifest = _read_json(neighbor_manifest_path.parent / "panel_manifest.json")
    benchmark = _quality_pool_benchmark(panel_manifest)
    market_exposure, market_annual = _market_exposure_rows(
        analysis_root=analysis_root,
        tasks=manifest["tasks"],
        benchmark=benchmark,
    )
    d20_daily, d20_selection = _d20_selection_excess(
        tasks=manifest["tasks"], panel_manifest=panel_manifest, pack=pack
    )
    concentration = _trade_concentration(manifest["tasks"])

    output_root = analysis_root / "validation"
    outputs = {
        "task_reconciliation": output_root / "task_reconciliation.parquet",
        "annual_reconciliation": output_root / "annual_reconciliation.parquet",
        "dynamic_recomputed": output_root / "dynamic_recomputed.parquet",
        "paired_event_comparison": output_root / "paired_event_comparison.parquet",
        "quality_pool_benchmark": output_root / "quality_pool_benchmark.parquet",
        "market_exposure": output_root / "market_exposure.parquet",
        "market_exposure_annual": output_root / "market_exposure_annual.parquet",
        "d20_selection_daily": output_root / "d20_selection_daily.parquet",
        "d20_selection_summary": output_root / "d20_selection_summary.parquet",
        "d20_trade_concentration": output_root / "d20_trade_concentration.parquet",
    }
    frames = {
        "task_reconciliation": task_reconciliation,
        "annual_reconciliation": annual_reconciliation,
        "dynamic_recomputed": dynamic_recomputed,
        "paired_event_comparison": paired_event_comparison,
        "quality_pool_benchmark": benchmark,
        "market_exposure": market_exposure,
        "market_exposure_annual": market_annual,
        "d20_selection_daily": d20_daily,
        "d20_selection_summary": d20_selection,
        "d20_trade_concentration": concentration,
    }
    for key, path in outputs.items():
        _write_frame(path, frames[key])

    numeric_columns = [
        "maximum_trade_field_difference",
        "maximum_cash_path_difference",
        "maximum_equity_path_difference",
        "metric_recompute_maximum_difference",
        "annual_recompute_maximum_difference",
        "total_cost_difference_cny",
        "turnover_difference_cny",
        "terminal_cash_identity_difference",
    ]
    maxima = {
        column: float(task_reconciliation[column].max()) for column in numeric_columns
    }
    blocking = {
        "declared_file_errors": len(declared_errors),
        "position_count_mismatches": int(
            (task_reconciliation["maximum_position_count_difference"] != 0).sum()
        ),
        "task_numeric_mismatches": int(
            (task_reconciliation[numeric_columns].max(axis=1) > FLOAT_TOLERANCE).sum()
        ),
        "aggregate_metric_mismatch": int(
            not math.isfinite(aggregate_metric_difference)
            or aggregate_metric_difference > FLOAT_TOLERANCE
        ),
        "aggregate_annual_mismatch": int(
            not math.isfinite(aggregate_annual_difference)
            or aggregate_annual_difference > FLOAT_TOLERANCE
        ),
        "dynamic_comparison_mismatch": int(
            not math.isfinite(dynamic_difference)
            or dynamic_difference > FLOAT_TOLERANCE
        ),
    }
    if any(blocking.values()):
        raise ValueError(f"daily_phase_execution_validation_failed:{blocking}:{maxima}")

    combined_market = market_exposure[
        market_exposure["profile"].isin(["geometry_combined", "blend01_combined"])
        & market_exposure["policy"].eq("fixed_d20")
        & market_exposure["cost_scenario"].eq("base")
    ]
    combined_selection = d20_selection[
        d20_selection["profile"].isin(["geometry_combined", "blend01_combined"])
        & d20_selection["scope"].eq("strict_2019_2023")
    ]
    validation = {
        "schema": VALIDATION_SCHEMA,
        "status": "completed",
        "study_id": phase.STUDY_ID,
        "analysis_manifest": _file_record(manifest_path),
        "audit": {
            "task_count": len(task_reconciliation),
            "trade_rows": int(task_reconciliation["trades"].sum()),
            "blocking": blocking,
            "paired_base_stress_event_path_differences": int(paired_mismatches),
            "paired_minimum_event_jaccard": float(
                paired_event_comparison["event_jaccard"].min()
            ),
            "maxima": maxima,
            "aggregate_metric_maximum_difference": float(aggregate_metric_difference),
            "aggregate_annual_maximum_difference": float(aggregate_annual_difference),
            "dynamic_comparison_maximum_difference": float(dynamic_difference),
        },
        "d20_diagnosis": {
            "combined_base_strategy_total_returns": {
                str(row.profile): float(row.strategy_total_return)
                for row in combined_market.itertuples()
            },
            "combined_base_exposure_matched_pool_total_returns": {
                str(row.profile): float(row.exposure_matched_pool_total_return)
                for row in combined_market.itertuples()
            },
            "combined_base_annualized_alphas": {
                str(row.profile): float(row.annualized_alpha)
                for row in combined_market.itertuples()
            },
            "strict_selection_excess_log_returns": {
                str(row.profile): float(row.selected_minus_pool_log_return)
                for row in combined_selection.itertuples()
            },
            "interpretation": (
                "fixed_d20_absolute_profit_is_not_incremental_alpha; compare the "
                "retained exposure and same-date selection diagnostics"
            ),
        },
        "outputs": {key: _file_record(path) for key, path in outputs.items()},
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    validation_path = output_root / "validation_manifest.json"
    _write_json(validation_path, validation)
    return validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_validation(args.analysis_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
