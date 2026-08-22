"""Local minute-level model and walk-forward research orchestration."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.core.io import write_json
from quantlab.research.minute_account import simulate_minute_account
from quantlab.research.minute_panel import (
    DEFAULT_DECISION_BAR,
    DEFAULT_ENTRY_BAR,
    DEFAULT_ENTRY_END_BAR,
    FEATURE_NAMES,
    MinuteDataset,
    MinuteResearchError,
    build_minute_dataset,
    build_minute_panel,
    load_minute_reference,
)

DEFAULT_PARTICIPATION_RATE = 0.01
BAR_COLUMNS = ("symbol", "trade_date", "bar_time", "open", "high", "low", "close", "volume", "amount")
ACCOUNT_SCENARIOS = (
    ("base", 1.0, None),
    ("stress", 2.0, None),
    ("stress_cap10", 2.0, 0.10),
    ("stress_cap5", 2.0, 0.05),
)


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def fit_minute_model(
    panel: pd.DataFrame,
    *,
    train_end_date: str,
    random_state: int = 7,
) -> tuple[Any, dict[str, Any]]:
    """Fit one frozen LightGBM regressor without parameter search."""

    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover
        raise MinuteResearchError("lightgbm_is_required_for_minute_baseline") from exc
    train = panel.loc[panel["label_exit_date"].astype(str) <= str(train_end_date)].dropna(
        subset=[*FEATURE_NAMES, "label_gross_return"]
    )
    if train["signal_date"].nunique() < 30 or len(train) < 500:
        raise MinuteResearchError(
            f"minute_training_support_too_small:{len(train)}:{train['signal_date'].nunique()}"
        )
    model = LGBMRegressor(
        objective="regression",
        n_estimators=220,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=100,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=int(random_state),
        verbosity=-1,
    )
    model.fit(train[list(FEATURE_NAMES)], train["label_gross_return"])
    importance = {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, model.feature_importances_, strict=True)
    }
    return model, {
        "train_row_count": int(len(train)),
        "train_date_count": int(train["signal_date"].nunique()),
        "train_end_date": str(train_end_date),
        "maximum_training_outcome_date": str(train["label_exit_date"].max()),
        "random_state": int(random_state),
        "feature_names": list(FEATURE_NAMES),
        "feature_importance": importance,
    }


def _rank_ic(frame: pd.DataFrame, prediction: str, actual: str) -> pd.Series:
    values: dict[str, float] = {}
    for date, group in frame.groupby("signal_date", sort=True):
        current = group[[prediction, actual]].dropna()
        if len(current) < 5:
            continue
        values[str(date)] = float(current[prediction].rank().corr(current[actual].rank()))
    return pd.Series(values, dtype=float)


def _score_with_model(
    panel: pd.DataFrame,
    model: Any,
    *,
    start_date: str,
    end_date: str = "",
) -> pd.DataFrame:
    result = panel.copy()
    result["prediction"] = np.nan
    mask = result["signal_date"] >= str(start_date)
    if end_date:
        mask &= result["signal_date"] <= str(end_date)
    valid = result.loc[mask].dropna(subset=list(FEATURE_NAMES))
    if not valid.empty:
        result.loc[valid.index, "prediction"] = model.predict(valid[list(FEATURE_NAMES)])
    return result


def evaluate_scored_panel(
    scored: pd.DataFrame,
    *,
    evaluation_start_date: str,
    evaluation_end_date: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    mask = (scored["signal_date"] >= str(evaluation_start_date)) & scored["prediction"].notna()
    if evaluation_end_date:
        mask &= scored["signal_date"] <= str(evaluation_end_date)
    evaluated = scored.loc[mask].copy()
    if evaluated.empty:
        raise MinuteResearchError("minute_evaluation_is_empty")
    daily_ic = _rank_ic(evaluated, "prediction", "label_gross_return")
    rule_ic = _rank_ic(evaluated, "morning_return", "label_gross_return")
    top = evaluated.sort_values(
        ["signal_date", "prediction", "symbol"], ascending=[True, False, True], kind="stable"
    ).groupby("signal_date", sort=True).head(int(top_k)).copy()
    rule_top = evaluated.sort_values(
        ["signal_date", "morning_return", "symbol"], ascending=[True, False, True], kind="stable"
    ).groupby("signal_date", sort=True).head(int(top_k)).copy()
    for selected in (top, rule_top):
        selected["action_gross_return"] = np.where(
            selected["entry_filled"],
            selected["label_gross_return"],
            0.0,
        )
    daily = pd.DataFrame(
        {
            "model_topk_action_gross_return": top.groupby("signal_date")["action_gross_return"].mean(),
            "rule_topk_action_gross_return": rule_top.groupby("signal_date")["action_gross_return"].mean(),
            "model_topk_entry_fill_rate": top.groupby("signal_date")["entry_filled"].mean(),
            "universe_entry_fill_rate": evaluated.groupby("signal_date")["entry_filled"].mean(),
            "universe_conditional_gross_return": evaluated.groupby("signal_date")["label_gross_return"].mean(),
            "rank_ic": daily_ic,
            "rule_rank_ic": rule_ic,
        }
    ).reset_index()
    if "index" in daily.columns:
        daily = daily.rename(columns={"index": "signal_date"})
    daily["year"] = daily["signal_date"].astype(str).str[:4].astype(int)
    annual = [
        {
            "year": int(year),
            "date_count": int(len(group)),
            "model_rank_ic_mean": _finite(group["rank_ic"].mean()),
            "rule_rank_ic_mean": _finite(group["rule_rank_ic"].mean()),
            "model_topk_mean_action_gross_return": _finite(
                group["model_topk_action_gross_return"].mean()
            ),
            "rule_topk_mean_action_gross_return": _finite(
                group["rule_topk_action_gross_return"].mean()
            ),
            "model_topk_entry_fill_rate": _finite(group["model_topk_entry_fill_rate"].mean()),
        }
        for year, group in daily.groupby("year", sort=True)
    ]
    metrics = {
        "evaluation_start_date": str(evaluation_start_date),
        "evaluation_requested_end_date": str(evaluation_end_date),
        "evaluation_end_date": str(evaluated["signal_date"].max()),
        "row_count": int(len(evaluated)),
        "date_count": int(evaluated["signal_date"].nunique()),
        "observed_label_count": int(evaluated["label_observed"].sum()),
        "top_k": int(top_k),
        "model_rank_ic_mean": _finite(daily_ic.mean()) if len(daily_ic) else None,
        "rule_rank_ic_mean": _finite(rule_ic.mean()) if len(rule_ic) else None,
        "model_topk_mean_action_gross_return": _finite(top["action_gross_return"].mean()),
        "rule_topk_mean_action_gross_return": _finite(rule_top["action_gross_return"].mean()),
        "model_topk_entry_fill_rate": _finite(top["entry_filled"].mean()),
        "universe_entry_fill_rate": _finite(evaluated["entry_filled"].mean()),
        "universe_mean_conditional_gross_return": _finite(evaluated["label_gross_return"].mean()),
        "annual": annual,
    }
    return {"metrics": metrics, "daily": daily}


def evaluate_minute_panel(
    panel: pd.DataFrame,
    model: Any,
    *,
    evaluation_start_date: str,
    evaluation_end_date: str = "",
    top_k: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored = _score_with_model(
        panel,
        model,
        start_date=evaluation_start_date,
        end_date=evaluation_end_date,
    )
    return scored, evaluate_scored_panel(
        scored,
        evaluation_start_date=evaluation_start_date,
        evaluation_end_date=evaluation_end_date,
        top_k=top_k,
    )


def fit_walk_forward(
    panel: pd.DataFrame,
    *,
    evaluation_start_year: int,
    evaluation_end_year: int,
    random_state: int = 7,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if int(evaluation_end_year) < int(evaluation_start_year):
        raise MinuteResearchError("minute_walk_forward_years_invalid")
    result = panel.copy()
    result["prediction"] = np.nan
    fits: list[dict[str, Any]] = []
    for year in range(int(evaluation_start_year), int(evaluation_end_year) + 1):
        model, fit = fit_minute_model(
            panel,
            train_end_date=f"{year - 1}-12-31",
            random_state=random_state,
        )
        mask = result["signal_date"].astype(str).str.startswith(f"{year}-")
        valid = result.loc[mask].dropna(subset=list(FEATURE_NAMES))
        result.loc[valid.index, "prediction"] = model.predict(valid[list(FEATURE_NAMES)])
        fits.append({"evaluation_year": year, **fit, "predicted_row_count": int(len(valid))})
    return result, fits


def _parquet_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    values = [value] if isinstance(value, (str, Path)) else list(value)
    paths = [Path(item).resolve() for item in values]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise MinuteResearchError(f"minute_parquet_missing:{','.join(missing)}")
    return paths


def _read_bar_frames(paths: Sequence[Path], *, maximum_trade_date: str = "") -> pd.DataFrame:
    filters: list[tuple[str, str, str]] = [
        ("bar_time", ">", "093000000"),
        ("bar_time", "<=", DEFAULT_ENTRY_END_BAR),
    ]
    if maximum_trade_date:
        filters.append(("trade_date", "<=", str(maximum_trade_date)))
    bars = pd.concat(
        [pd.read_parquet(path, columns=list(BAR_COLUMNS), filters=filters) for path in paths],
        ignore_index=True,
    )
    if maximum_trade_date and bars["trade_date"].astype(str).gt(str(maximum_trade_date)).any():
        raise MinuteResearchError("minute_parquet_cutoff_violation")
    return bars


def _load_dataset(
    parquet_path: str | Path | Sequence[str | Path],
    *,
    workspace_root: str | Path | None,
    maximum_trade_date: str = "",
) -> tuple[list[Path], MinuteDataset, dict[str, Any]]:
    paths = _parquet_paths(parquet_path)
    bars = _read_bar_frames(paths, maximum_trade_date=maximum_trade_date)
    if bars.duplicated(["symbol", "trade_date", "bar_time"]).any():
        raise MinuteResearchError("minute_parquet_sources_overlap")
    keys = bars.loc[:, ["symbol", "trade_date"]].drop_duplicates()
    reference = load_minute_reference(keys, workspace_root=workspace_root)
    dataset = build_minute_dataset(bars, reference.frame)
    return paths, dataset, reference.metadata


def _run_account_scenarios(
    scored: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    evaluation_start_date: str,
    evaluation_end_date: str,
    output: Path,
    top_k: int,
    maximum_participation_rate: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    scenarios: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, str]] = {}
    for name, multiplier, cap in ACCOUNT_SCENARIOS:
        account, equity, trades, fills = simulate_minute_account(
            scored,
            windows,
            evaluation_start_date=evaluation_start_date,
            evaluation_end_date=evaluation_end_date,
            top_k=top_k,
            slippage_multiplier=multiplier,
            maximum_credited_gross_return=cap,
            maximum_participation_rate=maximum_participation_rate,
        )
        account["scenario"] = name
        scenarios[name] = account
        current = {}
        for artifact_name, frame in (("equity", equity), ("trades", trades), ("exit_fills", fills)):
            path = output / f"{artifact_name}_{name}.parquet"
            frame.to_parquet(path, index=False)
            current[artifact_name] = str(path)
        artifacts[name] = current
    return scenarios, artifacts


def _write_outputs(
    *,
    output: Path,
    scored: pd.DataFrame,
    windows: pd.DataFrame,
    evaluation: dict[str, Any],
    account_artifacts: dict[str, dict[str, str]],
) -> dict[str, Any]:
    paths = {
        "scored_panel": output / "scored_panel.parquet",
        "execution_windows": output / "execution_windows.parquet",
        "daily_metrics": output / "daily_metrics.parquet",
    }
    scored.to_parquet(paths["scored_panel"], index=False)
    windows.to_parquet(paths["execution_windows"], index=False)
    evaluation["daily"].to_parquet(paths["daily_metrics"], index=False)
    return {name: str(path) for name, path in paths.items()} | {"scenarios": account_artifacts}


def _result_contract(
    maximum_participation_rate: float,
    *,
    maximum_trade_date: str = "",
) -> dict[str, Any]:
    return {
        "decision_bar": DEFAULT_DECISION_BAR,
        "entry_bar": DEFAULT_ENTRY_BAR,
        "entry_end_bar": DEFAULT_ENTRY_END_BAR,
        "planned_holding": "ordinary-share T+1; exits begin next market day and delay until executable",
        "candidate_policy": "rank all 10:00-eligible candidates before observing the 10:01-10:10 fill window",
        "price_mode": "raw execution price; QDP back-adjust factor for cross-day return and valuation",
        "one_price_window_policy": "all one-price entries are rejected; only one-price-down exits are delayed",
        "maximum_participation_rate": float(maximum_participation_rate),
        "maximum_trade_date": str(maximum_trade_date),
        "unresolved_policy": "retain and mark at the last valid adjusted window price; never force liquidation",
    }


def run_minute_baseline(
    parquet_path: str | Path | Sequence[str | Path],
    *,
    train_end_date: str,
    evaluation_start_date: str,
    evaluation_end_date: str = "",
    output_dir: str | Path,
    top_k: int = 5,
    maximum_participation_rate: float = DEFAULT_PARTICIPATION_RATE,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    paths, dataset, reference = _load_dataset(
        parquet_path,
        workspace_root=workspace_root,
        maximum_trade_date=evaluation_end_date,
    )
    model, fit = fit_minute_model(dataset.panel, train_end_date=train_end_date)
    scored, evaluation = evaluate_minute_panel(
        dataset.panel,
        model,
        evaluation_start_date=evaluation_start_date,
        evaluation_end_date=evaluation_end_date,
        top_k=top_k,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    accounts, account_artifacts = _run_account_scenarios(
        scored,
        dataset.windows,
        evaluation_start_date=evaluation_start_date,
        evaluation_end_date=evaluation_end_date,
        output=output,
        top_k=top_k,
        maximum_participation_rate=maximum_participation_rate,
    )
    result = {
        "schema": "quantlab.minute_baseline/2",
        "status": "ok",
        "source_parquet": [str(path) for path in paths],
        "contract": _result_contract(
            maximum_participation_rate,
            maximum_trade_date=evaluation_end_date,
        ),
        "reference": reference,
        "data_quality": dataset.quality,
        "fit": fit,
        "evaluation": evaluation["metrics"],
        "account": accounts["stress"],
        "account_scenarios": accounts,
        "artifacts": _write_outputs(
            output=output,
            scored=scored,
            windows=dataset.windows,
            evaluation=evaluation,
            account_artifacts=account_artifacts,
        ),
    }
    write_json(output / "result.json", result)
    return result


def run_minute_walk_forward(
    parquet_path: str | Path | Sequence[str | Path],
    *,
    evaluation_start_year: int,
    evaluation_end_year: int,
    output_dir: str | Path,
    top_k: int = 5,
    maximum_participation_rate: float = DEFAULT_PARTICIPATION_RATE,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    start_date = f"{int(evaluation_start_year)}-01-01"
    end_date = f"{int(evaluation_end_year)}-12-31"
    paths, dataset, reference = _load_dataset(
        parquet_path,
        workspace_root=workspace_root,
        maximum_trade_date=end_date,
    )
    scored, fits = fit_walk_forward(
        dataset.panel,
        evaluation_start_year=evaluation_start_year,
        evaluation_end_year=evaluation_end_year,
    )
    evaluation = evaluate_scored_panel(
        scored,
        evaluation_start_date=start_date,
        evaluation_end_date=end_date,
        top_k=top_k,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    accounts, account_artifacts = _run_account_scenarios(
        scored,
        dataset.windows,
        evaluation_start_date=start_date,
        evaluation_end_date=end_date,
        output=output,
        top_k=top_k,
        maximum_participation_rate=maximum_participation_rate,
    )
    result = {
        "schema": "quantlab.minute_walk_forward/1",
        "status": "ok",
        "source_parquet": [str(path) for path in paths],
        "contract": _result_contract(
            maximum_participation_rate,
            maximum_trade_date=end_date,
        ),
        "reference": reference,
        "data_quality": dataset.quality,
        "fits": fits,
        "evaluation": evaluation["metrics"],
        "account": accounts["stress"],
        "account_scenarios": accounts,
        "artifacts": _write_outputs(
            output=output,
            scored=scored,
            windows=dataset.windows,
            evaluation=evaluation,
            account_artifacts=account_artifacts,
        ),
    }
    write_json(output / "result.json", result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local T+1 minute baseline")
    parser.add_argument("--parquet", action="append", required=True)
    parser.add_argument("--train-end-date", required=True)
    parser.add_argument("--evaluation-start-date", required=True)
    parser.add_argument("--evaluation-end-date", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--maximum-participation-rate", type=float, default=DEFAULT_PARTICIPATION_RATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_minute_baseline(
        args.parquet,
        train_end_date=args.train_end_date,
        evaluation_start_date=args.evaluation_start_date,
        evaluation_end_date=args.evaluation_end_date,
        output_dir=args.output_dir,
        top_k=args.top_k,
        maximum_participation_rate=args.maximum_participation_rate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PARTICIPATION_RATE",
    "FEATURE_NAMES",
    "MinuteResearchError",
    "build_minute_panel",
    "evaluate_minute_panel",
    "fit_minute_model",
    "fit_walk_forward",
    "run_minute_baseline",
    "run_minute_walk_forward",
    "simulate_minute_account",
]
