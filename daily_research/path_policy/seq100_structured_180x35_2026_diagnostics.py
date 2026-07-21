"""Checkpoint and execution diagnostics for the Structured 180x35 2026 study.

This module is deliberately outside the preselected-model study contract. It
compares the 2025 checkpoint on the same 2026 material and scans execution
policies for the 2026 checkpoint without changing the deployment decision.
"""

from __future__ import annotations

import gc
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_2026_fold_comparison as comparison_2026
from daily_research.path_policy import seq100_checkpoint_freshness as freshness
from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import seq100_integrity_v2 as integrity
from daily_research.path_policy import seq100_structured_180x35_2026 as study
from daily_research.path_policy import seq100_structured_experiment as structured
from daily_research.path_policy.seq100_exit_policy_audit import CandidateCompleteAuditPack


DIAGNOSTIC_ID = "structured-180x35-2026-checkpoint-execution-diagnostic-v1"
OUTPUT_ROOT = study.STUDY_ROOT / "analysis/checkpoint_execution_diagnostic_20260721"
MODEL_2025 = "structured_180x35_2025_on_2026"
MODEL_2026 = study.MODEL_2026_180
MODEL_ORDER = (MODEL_2025, MODEL_2026)
VIEW_2025 = OUTPUT_ROOT / "views/evaluation_2025_on_2026_native_180.json"
SCAN_POLICIES = tuple(
    [finite.PolicySpec(name=f"fixed_d{day}", kind="fixed", fixed_day=day) for day in range(2, 61)]
    + [
        finite.PolicySpec(name="model_plan", kind="model_plan"),
        finite.PolicySpec(name="rolling_path", kind="rolling"),
    ]
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    study._write_json(path, payload)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    study._write_csv(path, frame)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    study._write_parquet(path, frame)


def _canonical_digest(payload: Any) -> str:
    return study._canonical_digest(payload)


def _file_sha256(path: Path) -> str:
    return study._file_sha256(path)


def _new_run() -> Path:
    complete, partial = study._completed_runs()
    if len(complete) != 1 or partial:
        raise ValueError("the Structured 180x35 2026 training task is incomplete")
    return complete[0]


def _model_material(model_id: str) -> tuple[Path, Path, int]:
    if model_id == MODEL_2025:
        return study._old_2025_run(), VIEW_2025, 2025
    if model_id == MODEL_2026:
        return _new_run(), study.STUDY_ROOT / "fold_view_2026.json", 2026
    raise ValueError(f"unknown diagnostic model: {model_id}")


def _checkpoint_normalization(run_dir: Path) -> tuple[dict[str, Any], str]:
    summary = _read_json(run_dir / "sequence_path_training_summary.json")
    checkpoint_path = Path(str(summary["best_checkpoint"])).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        return freshness._normalization_from_checkpoint(checkpoint)
    finally:
        del checkpoint
        gc.collect()


def prepare_diagnostics(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    root = output_root.resolve()
    if root != OUTPUT_ROOT.resolve():
        raise ValueError("diagnostics require the canonical output root")
    study.verify_structured_180x35_2026(require_complete=True)
    root.mkdir(parents=True, exist_ok=True)
    old_run = study._old_2025_run()
    normalization, normalization_sha256 = _checkpoint_normalization(old_run)
    source_view = _read_json(study.STUDY_ROOT / "fold_view_2026.json")
    view = json.loads(json.dumps(source_view, ensure_ascii=False))
    source_fold = dict(view.pop("development_fold_training_contract", {}) or {})
    view["normalization"] = normalization
    artifact = dict(view.get("artifact_view", {}) or {})
    artifact.update(
        {
            "schema_version": 1,
            "view_id": "seq100_structured_180x35_2025_checkpoint_on_2026_native_180",
            "view_type": "checkpoint_normalization_evaluation_view",
        }
    )
    view["artifact_view"] = artifact
    view["checkpoint_diagnostic"] = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "checkpoint_vintage": 2025,
        "target_year": 2026,
        "normalization_source": "checkpoint.fold_training_contract.payload.normalization",
        "normalization_sha256": normalization_sha256,
        "source_2026_fold_contract_sha256": str(source_fold.get("sha256", "")),
        "candidate_mode": "native_180",
        "candidate_membership_uses_future_information": False,
    }
    if VIEW_2025.is_file():
        if _read_json(VIEW_2025) != view:
            raise ValueError("the 2025-on-2026 diagnostic view drifted")
    else:
        _write_json(VIEW_2025, view)

    dataset = training.SequencePathPackDataset(
        view,
        split="development",
        max_samples=0,
        input_channel_profile=study.INPUT_CHANNEL_PROFILE,
        index_role="candidate",
    )
    try:
        if (
            len(dataset) != study.EXPECTED_NATIVE_FULL_LABEL_CANDIDATES
            or int(dataset.lookback_days) != study.LOOKBACK_DAYS
            or int(dataset.input_dim) != study.INPUT_DIM
        ):
            raise ValueError("the diagnostic view does not preserve the 180x35 candidate contract")
    finally:
        del dataset
        gc.collect()

    candidate_path = Path(str(view["candidate_index_path"])).resolve()
    contract = {
        "schema_version": 1,
        "diagnostic_id": DIAGNOSTIC_ID,
        "source_study_contract_sha256": str(_read_json(study.STUDY_ROOT / "study.json")["contract_sha256"]),
        "candidate_semantic_sha256": integrity.candidate_semantic_sha256(candidate_path),
        "candidate_count": study.EXPECTED_NATIVE_FULL_LABEL_CANDIDATES,
        "full_label_window": [study.SIGNAL_START, study.FULL_LABEL_END],
        "extended_d7_window": [study.SIGNAL_START, study.EXTENDED_SIGNAL_END],
        "checkpoint_2025_sha256": _file_sha256(old_run / "best_model.pt"),
        "checkpoint_2025_normalization_sha256": normalization_sha256,
        "checkpoint_2026_sha256": _file_sha256(_new_run() / "best_model.pt"),
        "execution_scan": {
            "model_id": MODEL_2026,
            "top_k_slots": [[1, 1], [3, 3]],
            "policies": [asdict(policy) for policy in SCAN_POLICIES],
            "cost_scenarios": list(study.COST_SCENARIOS),
            "selection_window": [study.SIGNAL_START, study.FULL_LABEL_END],
            "primary_metric": "signal_period_annualized_log_growth",
            "terminal_sensitivity_metric": "liquidated_annualized_log_growth",
            "interpretation": "same-window post-hoc diagnostic, not a deployment promotion",
        },
        "protected_boundaries": {
            "qdp_changed": False,
            "provider_called": False,
            "checkpoint_changed": False,
            "live_state_changed": False,
        },
    }
    payload = {"contract": contract, "contract_sha256": _canonical_digest(contract)}
    contract_path = root / "contract.json"
    if contract_path.is_file():
        if _read_json(contract_path) != payload:
            raise ValueError("the diagnostic contract drifted")
    else:
        _write_json(contract_path, payload)
    return payload


def _load_model_dataset(
    model_id: str, *, device: torch.device
) -> tuple[torch.nn.Module, training.SequencePathPackDataset, Path, dict[str, Any]]:
    run_dir, view_path, _vintage = _model_material(model_id)
    view = _read_json(view_path)
    dataset = training.SequencePathPackDataset(
        view,
        split="development",
        max_samples=0,
        input_channel_profile=study.INPUT_CHANNEL_PROFILE,
        index_role="candidate",
    )
    if len(dataset) != study.EXPECTED_NATIVE_FULL_LABEL_CANDIDATES:
        raise ValueError(f"candidate coverage drifted for {model_id}")
    model, summary = structured._load_checkpoint_model(run_dir, dataset, device=device)
    return model, dataset, run_dir, summary


def _full_label_result_path(model_id: str) -> Path:
    return OUTPUT_ROOT / "full_label" / model_id / "evaluation.json"


def _evaluate_old_full_label() -> dict[str, Any]:
    result_path = _full_label_result_path(MODEL_2025)
    if result_path.is_file():
        return _read_json(result_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset, run_dir, summary = _load_model_dataset(MODEL_2025, device=device)
    output_dir = result_path.parent / "inference"
    try:
        daily_ic, topk, daily_topk, topk_candidates, split_metrics = training._predict_split(
            model=model,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            split="development",
            batch_size=study.BATCH_SIZE,
            amp_enabled=bool(device.type == "cuda"),
            top_k=study.TOP_K_VALUES,
            write_predictions=False,
            write_path_predictions=False,
            direct_value_horizon=0,
        )
    finally:
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "topk_metrics": result_path.parent / "topk_metrics.csv",
        "daily_rank_ic": result_path.parent / "daily_rank_ic.csv",
        "daily_topk_metrics": result_path.parent / "daily_topk_metrics.csv",
        "topk_candidates": result_path.parent / "topk_candidates.parquet",
    }
    _write_csv(paths["topk_metrics"], topk)
    _write_csv(paths["daily_rank_ic"], daily_ic)
    _write_csv(paths["daily_topk_metrics"], daily_topk)
    _write_parquet(paths["topk_candidates"], topk_candidates)
    metrics = comparison_2026._evaluation_metrics_2026(
        topk=topk, daily_topk=daily_topk, split_metrics=split_metrics
    )
    normalization, normalization_sha256 = _checkpoint_normalization(run_dir)
    result = {
        "schema_version": 1,
        "status": "completed",
        "diagnostic_id": DIAGNOSTIC_ID,
        "model_id": MODEL_2025,
        "checkpoint_vintage": 2025,
        "checkpoint_sha256": _file_sha256(run_dir / "best_model.pt"),
        "normalization_sha256": normalization_sha256,
        "model_bundle_sha256": integrity.model_bundle_sha256(
            {
                "checkpoint_sha256": _file_sha256(run_dir / "best_model.pt"),
                "normalization": normalization,
                "architecture": study._architecture_signature(run_dir),
            }
        ),
        "metrics": study._json_safe(metrics),
        "candidate_count": int(split_metrics["row_count"]),
        "date_count": int(split_metrics["date_count"]),
        "outputs": {key: str(path.resolve()) for key, path in paths.items()},
        "output_sha256": {key: _file_sha256(path) for key, path in paths.items()},
        "completed_at": study._now(),
    }
    _write_json(result_path, result)
    return result


def _extended_prediction_path(model_id: str) -> Path:
    return OUTPUT_ROOT / "predictions" / f"extended_score_and_exit_{model_id}.parquet"


def _evaluate_extended(model_id: str) -> Path:
    target = _extended_prediction_path(model_id)
    if target.is_file():
        frame = pd.read_parquet(target, columns=["score", "predicted_exit_day", "trade_date"])
        if (
            len(frame) == 0
            or frame["trade_date"].nunique() != study.EXPECTED_EXTENDED_DATES
            or not bool(np.isfinite(frame[["score", "predicted_exit_day"]]).all().all())
        ):
            raise ValueError(f"stored extended diagnostics are incomplete for {model_id}")
        return target
    suffix = _read_json(study.OVERLAY_ROOT / "manifest.json")
    candidate_path = Path(
        str(dict(suffix["candidate_indexes"])["extended_native_180"]["path"])
    ).resolve()
    candidates = pd.read_parquet(candidate_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dataset, _run_dir, _summary = _load_model_dataset(model_id, device=device)
    accessor = study.ExtendedFeatureAccessor(dataset, suffix)
    guard = finite._MemoryGuard()
    rows: list[pd.DataFrame] = []
    next_bucket = 10
    try:
        for start in range(0, len(candidates), study.BATCH_SIZE):
            guard.check()
            stop = min(start + study.BATCH_SIZE, len(candidates))
            part = candidates.iloc[start:stop]
            dates = part["date_idx"].to_numpy(dtype=np.int64)
            symbols = part["symbol_idx"].to_numpy(dtype=np.int64)
            x = torch.from_numpy(accessor.batch_x(dates, symbols)).to(device)
            symbol_tensor = torch.from_numpy(symbols.astype(np.int64, copy=False)).to(device)
            with torch.no_grad(), torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                output = model(x, symbol_idx=symbol_tensor)
            path = output["future_path"].detach().float().cpu().numpy()
            summary = training._derive_path_summary_numpy(
                path,
                price_anchor=dataset.price_anchor,
                earliest_exit_day=2,
            )
            columns = training.derived_path_summary_columns(dataset.forward_days, path_dim=4)
            score = summary[:, columns.index(training.value_column_for_path(dataset.forward_days, path_dim=4))]
            planned = summary[:, columns.index(f"best_exit_day_{dataset.forward_days}d")]
            if not bool(np.isfinite(score).all()) or not bool(np.isfinite(planned).all()):
                raise ValueError(f"non-finite extended prediction for {model_id}")
            rows.append(
                part[
                    [
                        "candidate_id",
                        "trade_date",
                        "date_idx",
                        "symbol_idx",
                        "symbol",
                        "entry_filled",
                    ]
                ].assign(score=score.astype(np.float64), predicted_exit_day=planned.astype(np.int16))
            )
            percent = int(stop * 100 / len(candidates))
            if percent >= next_bucket:
                print(f"{model_id}: extended inference {next_bucket}%", flush=True)
                next_bucket += 10
            del x, symbol_tensor, output, path, summary, score, planned
    finally:
        accessor.close()
        del model, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    forecast = pd.concat(rows, ignore_index=True).sort_values(
        ["date_idx", "symbol_idx"], kind="mergesort"
    ).reset_index(drop=True)
    if (
        len(forecast) != len(candidates)
        or forecast["trade_date"].nunique() != study.EXPECTED_EXTENDED_DATES
        or bool(forecast.duplicated(["date_idx", "symbol_idx"]).any())
    ):
        raise ValueError(f"extended candidate coverage drifted for {model_id}")
    _write_parquet(target, forecast)
    return target


def _d7_job_dir(
    model_id: str, spec: study.AccountSpec, cost: str, window: str
) -> Path:
    return OUTPUT_ROOT / "d7_accounts" / model_id / spec.strategy_id / cost / window


def _run_d7_accounts() -> pd.DataFrame:
    audit = CandidateCompleteAuditPack(study.FROZEN_2026_OVERLAY)
    metrics: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        forecast = pd.read_parquet(_evaluate_extended(model_id))
        for spec in study.ACCOUNT_SPECS:
            for cost in study.COST_SCENARIOS:
                for window, end in (
                    ("strict_101", study.STRICT_SIGNAL_END),
                    ("extended_121", study.EXTENDED_SIGNAL_END),
                ):
                    root = _d7_job_dir(model_id, spec, cost, window)
                    result_path = root / "evaluation.json"
                    if result_path.is_file():
                        result = _read_json(result_path)
                    else:
                        metric, daily, trades, terminal = study.simulate_partial_d7_account(
                            model_id=model_id,
                            forecast=forecast,
                            spec=spec,
                            cost_scenario=cost,
                            last_signal_date=end,
                            audit_pack=audit,
                        )
                        root.mkdir(parents=True, exist_ok=True)
                        paths = {
                            "daily": root / "daily.parquet",
                            "trades": root / "trades.parquet",
                            "terminal": root / "terminal.parquet",
                        }
                        _write_parquet(paths["daily"], daily)
                        _write_parquet(paths["trades"], trades)
                        _write_parquet(paths["terminal"], terminal)
                        result = {
                            "schema_version": 1,
                            "status": "completed",
                            "diagnostic_id": DIAGNOSTIC_ID,
                            "model_id": model_id,
                            "strategy_id": spec.strategy_id,
                            "cost_scenario": cost,
                            "window_name": window,
                            "metrics": study._json_safe(metric),
                            "outputs": {key: str(path.resolve()) for key, path in paths.items()},
                            "completed_at": study._now(),
                        }
                        _write_json(result_path, result)
                    row = dict(result["metrics"])
                    row["window_name"] = window
                    metrics.append(row)
    frame = pd.DataFrame(metrics)
    _write_csv(OUTPUT_ROOT / "d7_checkpoint_metrics.csv", frame)
    return frame


def _correct_entry_filled() -> np.ndarray:
    suffix = _read_json(study.OVERLAY_ROOT / "manifest.json")
    meta = dict(suffix["execution_masks"]["entry_filled_extended"])
    extended = np.memmap(
        str(meta["path"]),
        dtype="bool",
        mode="r",
        shape=tuple(int(value) for value in meta["shape"]),
    )
    # The compact overlay stores only the 121 extended signal dates.  The
    # finite-account engine uses the full 4014-date market index, so map by
    # date value rather than treating overlay row numbers as global date_idx.
    full_dates = [str(value) for value in _read_json(study.FROZEN_2026_OVERLAY)["date_values"]]
    candidate_path = Path(
        str(dict(suffix["candidate_indexes"])["extended_native_180"]["path"])
    ).resolve()
    suffix_dates = sorted(
        pd.read_parquet(candidate_path, columns=["trade_date"])["trade_date"]
        .astype(str)
        .unique()
        .tolist()
    )
    if len(suffix_dates) != int(extended.shape[0]):
        raise ValueError("extended entry-mask date mapping is incomplete")
    full = np.zeros((len(full_dates), int(extended.shape[1])), dtype=bool)
    date_to_idx = {value: idx for idx, value in enumerate(full_dates)}
    for row_idx, date in enumerate(suffix_dates):
        if date in date_to_idx:
            full[date_to_idx[date], :] = extended[row_idx, :]
    return full


def _market() -> finite.BacktestMarket:
    audit = CandidateCompleteAuditPack(study.FROZEN_2026_OVERLAY)
    entry_filled = _correct_entry_filled()
    if entry_filled.shape != audit.exit_sellable.shape:
        raise ValueError("corrected entry mask and execution panel shapes differ")
    return finite.BacktestMarket(
        date_values=np.asarray(audit.date_values, dtype=object),
        symbol_values=np.asarray(audit.symbol_values, dtype=object),
        entry_open_raw=audit.entry_open_raw,
        exit_close_raw=audit.exit_close_raw,
        exit_sellable=audit.exit_sellable,
        entry_filled=entry_filled,
        contract=audit.contract,
        terminal_recovery_fraction=float(audit.terminal_recovery_fraction),
        forward_days=int(audit.forward_days),
        execution_days=int(audit.execution_days),
    )


def _forecast_book(frame: pd.DataFrame, *, top_k: int) -> finite.ForecastBook:
    book = finite.ForecastBook(f"{MODEL_2026}_top{top_k}")
    for date_idx_raw, group in frame.groupby("date_idx", sort=True):
        book.add_day(
            date_idx=int(date_idx_raw),
            symbol_idx=group["symbol_idx"].to_numpy(dtype=np.int32),
            score=group["score"].to_numpy(dtype=np.float64),
            planned_day=group["predicted_exit_day"].to_numpy(dtype=np.int16),
        )
    for date_idx, day in list(book.days.items()):
        selected = np.argsort(-day.score, kind="mergesort")[: int(top_k)]
        book.days[date_idx] = finite.ForecastDay(
            symbol_idx=day.symbol_idx,
            score=day.score,
            planned_day=day.planned_day,
            top3_symbol_idx=tuple(int(value) for value in day.symbol_idx[selected]),
        )
    return book


def _log_growth(total_return: float, sessions: int) -> float:
    if int(sessions) <= 0 or float(total_return) <= -1.0:
        return float("-inf")
    return float(math.log1p(float(total_return)) * 252.0 / float(sessions))


def _scan_execution() -> pd.DataFrame:
    target = OUTPUT_ROOT / "execution_scan_metrics.csv"
    if target.is_file():
        frame = pd.read_csv(target)
        if len(frame) != 244:
            raise ValueError("stored execution scan row count drifted")
        return frame
    forecast = pd.read_parquet(_evaluate_extended(MODEL_2026))
    market = _market()
    date_to_idx = {str(value): idx for idx, value in enumerate(market.date_values)}
    first_idx = int(date_to_idx[study.SIGNAL_START])
    last_idx = int(date_to_idx[study.FULL_LABEL_END])
    guard = finite._MemoryGuard()
    rows: list[dict[str, Any]] = []
    total = 2 * len(SCAN_POLICIES) * len(study.COST_SCENARIOS)
    completed = 0
    for top_k, slots in ((1, 1), (3, 3)):
        book = _forecast_book(forecast, top_k=top_k)
        for policy in SCAN_POLICIES:
            for cost in study.COST_SCENARIOS:
                metric, _daily, trades, _annual = finite.simulate_portfolio(
                    market=market,
                    book=book,
                    raw_top3_paths={},
                    policy=policy,
                    slots=slots,
                    cost_scenario=cost,
                    first_signal_date_idx=first_idx,
                    last_signal_date_idx=last_idx,
                    starting_cash=study.STARTING_CASH_CNY,
                    memory_guard=guard,
                )
                occupied = (
                    float(
                        np.sum(
                            trades["buy_cash_cny"].to_numpy(dtype=np.float64)
                            * trades["occupied_sessions"].to_numpy(dtype=np.float64)
                        )
                    )
                    if not trades.empty
                    else 0.0
                )
                net_pnl = float(trades["net_pnl_cny"].sum()) if not trades.empty else 0.0
                row = {
                    "model_id": MODEL_2026,
                    "top_k": top_k,
                    "slot_count": slots,
                    "policy_name": policy.name,
                    "policy_kind": policy.kind,
                    "fixed_day": policy.fixed_day,
                    "cost_scenario": cost,
                    **metric,
                    "signal_period_annualized_log_growth": _log_growth(
                        float(metric["signal_period_total_return"]),
                        int(metric["signal_session_count"]),
                    ),
                    "liquidated_annualized_log_growth": _log_growth(
                        float(metric["liquidated_total_return"]),
                        int(metric["equity_path_session_count"]),
                    ),
                    "net_pnl_per_deployed_capital_session": (
                        float(net_pnl / occupied) if occupied > 0.0 else None
                    ),
                }
                rows.append(study._json_safe(row))
                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(f"execution scan {completed}/{total}", flush=True)
    frame = pd.DataFrame(rows)
    _write_csv(target, frame)
    return frame


def _best_scan_rows(scan: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (top_k, slots, cost), group in scan.groupby(
        ["top_k", "slot_count", "cost_scenario"], sort=True
    ):
        for objective in (
            "signal_period_annualized_log_growth",
            "liquidated_annualized_log_growth",
        ):
            row = group.sort_values(
                [objective, "policy_name"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0]
            output.append(
                {
                    "top_k": int(top_k),
                    "slot_count": int(slots),
                    "cost_scenario": str(cost),
                    "objective": objective,
                    "policy_name": str(row["policy_name"]),
                    "policy_kind": str(row["policy_kind"]),
                    "fixed_day": (
                        None if pd.isna(row["fixed_day"]) else int(row["fixed_day"])
                    ),
                    "objective_value": float(row[objective]),
                    "signal_period_total_return": float(row["signal_period_total_return"]),
                    "liquidated_total_return": float(row["liquidated_total_return"]),
                    "signal_period_maximum_drawdown": float(
                        row["signal_period_maximum_drawdown"]
                    ),
                    "full_path_maximum_drawdown": float(row["full_path_maximum_drawdown"]),
                    "closed_trade_count": int(row["closed_trade_count"]),
                    "winning_trade_rate": float(row["winning_trade_rate"]),
                    "mean_occupied_sessions": float(row["mean_occupied_sessions"]),
                }
            )
    return output


def _pairwise_forecasts() -> tuple[pd.DataFrame, dict[str, Any]]:
    old = pd.read_parquet(_evaluate_extended(MODEL_2025))
    new = pd.read_parquet(_evaluate_extended(MODEL_2026))
    keys = ["candidate_id", "trade_date", "date_idx", "symbol_idx", "symbol"]
    if not old[keys].equals(new[keys]):
        raise ValueError("2025 and 2026 checkpoint candidate keys differ")
    rows: list[dict[str, Any]] = []
    for date_idx_raw, positions in old.groupby("date_idx", sort=True).indices.items():
        idx = np.asarray(positions, dtype=np.int64)
        old_scores = old.loc[idx, "score"].to_numpy(dtype=np.float64)
        new_scores = new.loc[idx, "score"].to_numpy(dtype=np.float64)
        old_order = np.argsort(-old_scores, kind="mergesort")
        new_order = np.argsort(-new_scores, kind="mergesort")
        rows.append(
            {
                "trade_date": str(old.loc[idx[0], "trade_date"]),
                "date_idx": int(date_idx_raw),
                "score_spearman": float(
                    pd.Series(old_scores).corr(pd.Series(new_scores), method="spearman")
                ),
                "top1_same": bool(old.loc[idx[old_order[0]], "symbol_idx"] == new.loc[idx[new_order[0]], "symbol_idx"]),
                "top3_overlap_fraction": float(
                    len(set(old.loc[idx[old_order[:3]], "symbol_idx"]) & set(new.loc[idx[new_order[:3]], "symbol_idx"]))
                    / 3.0
                ),
            }
        )
    daily = pd.DataFrame(rows)
    _write_csv(OUTPUT_ROOT / "checkpoint_pairwise_daily.csv", daily)
    summary = {
        "mean_daily_score_spearman": float(daily["score_spearman"].mean()),
        "top1_same_day_rate": float(daily["top1_same"].mean()),
        "mean_top3_overlap_fraction": float(daily["top3_overlap_fraction"].mean()),
        "date_count": int(len(daily)),
    }
    return daily, summary


def _full_label_rows() -> pd.DataFrame:
    old = _evaluate_old_full_label()
    new = _read_json(study._full_label_task_dir(MODEL_2026) / "evaluation.json")
    rows: list[dict[str, Any]] = []
    for result in (old, new):
        metrics = dict(result["metrics"])
        rows.append(
            {
                "model_id": str(result["model_id"]),
                "checkpoint_vintage": int(result["checkpoint_vintage"]),
                **{
                    key: metrics.get(key)
                    for key in (
                        "rank_ic",
                        "rank_ic_positive_day_rate",
                        "path_mae",
                        "path_close_mae",
                        "exit_regret",
                        "top1_base_alpha",
                        "top1_stress_alpha",
                        "top3_base_alpha",
                        "top3_stress_alpha",
                        "candidate_score_coverage",
                        "execution_return_coverage",
                    )
                },
            }
        )
    frame = pd.DataFrame(rows)
    _write_csv(OUTPUT_ROOT / "full_label_checkpoint_metrics.csv", frame)
    return frame


def _d7_late_trade_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        for spec in study.ACCOUNT_SPECS:
            root = _d7_job_dir(model_id, spec, "double_slippage", "extended_121")
            trades = pd.read_parquet(root / "trades.parquet")
            late = trades[trades["signal_date"].astype(str).gt(study.STRICT_SIGNAL_END)]
            returns = late["net_return_on_buy_cash"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "model_id": model_id,
                    "strategy_id": spec.strategy_id,
                    "late_trade_count": int(len(late)),
                    "late_mean_trade_return": float(np.mean(returns)) if len(returns) else None,
                    "late_winning_trade_rate": float(np.mean(returns > 0.0)) if len(returns) else None,
                    "late_compounded_trade_return": float(np.prod(1.0 + returns) - 1.0) if len(returns) else None,
                }
            )
    frame = pd.DataFrame(rows)
    _write_csv(OUTPUT_ROOT / "late_trade_diagnostics.csv", frame)
    return frame


def summarize_diagnostics(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError("diagnostics require the canonical output root")
    full = _full_label_rows()
    d7 = _run_d7_accounts()
    scan = _scan_execution()
    _daily_pairwise, pairwise = _pairwise_forecasts()
    late = _d7_late_trade_summary()
    best = _best_scan_rows(scan)
    summary = {
        "schema_version": 1,
        "status": "completed",
        "diagnostic_id": DIAGNOSTIC_ID,
        "created_at": study._now(),
        "full_label_checkpoint_metrics": study._json_safe(full.to_dict("records")),
        "d7_checkpoint_metrics": study._json_safe(d7.to_dict("records")),
        "late_trade_diagnostics": study._json_safe(late.to_dict("records")),
        "checkpoint_pairwise": pairwise,
        "execution_scan_best": best,
        "execution_scan_interpretation": {
            "primary": "signal-period annualized log growth on the common 48-signal window",
            "sensitivity": "fully liquidated growth through the common 80-session execution tail",
            "selection_status": "post_hoc_diagnostic_only",
            "formal_promotion": False,
        },
        "historical_d7_contract_note": {
            "selected_on": "joint 2023-2025 after-cost annualized log account growth",
            "not_selected_per_year": True,
            "historical_terminal_boundary": "signal_D80",
            "current_d7_terminal_boundary": "requested_D7_plus_20_sessions",
            "observed_material_effect": "checked separately from policy selection",
        },
        "outputs": {
            "full_label_checkpoint_metrics": str((OUTPUT_ROOT / "full_label_checkpoint_metrics.csv").resolve()),
            "d7_checkpoint_metrics": str((OUTPUT_ROOT / "d7_checkpoint_metrics.csv").resolve()),
            "execution_scan_metrics": str((OUTPUT_ROOT / "execution_scan_metrics.csv").resolve()),
            "checkpoint_pairwise_daily": str((OUTPUT_ROOT / "checkpoint_pairwise_daily.csv").resolve()),
            "late_trade_diagnostics": str((OUTPUT_ROOT / "late_trade_diagnostics.csv").resolve()),
        },
        "protected_boundaries": {
            "qdp_changed": False,
            "provider_called": False,
            "checkpoint_changed": False,
            "live_state_changed": False,
            "deployment_changed": False,
        },
    }
    _write_json(OUTPUT_ROOT / "diagnostic_summary.json", summary)
    verify_diagnostics(output_root=output_root)
    return summary


def run_diagnostics(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    prepare_diagnostics(output_root=output_root)
    print("2025 checkpoint full-label inference", flush=True)
    _evaluate_old_full_label()
    for model_id in MODEL_ORDER:
        print(f"{model_id} extended score and exit inference", flush=True)
        _evaluate_extended(model_id)
    print("2025/2026 fixed-D7 account replay", flush=True)
    _run_d7_accounts()
    print("2026 checkpoint execution-policy scan", flush=True)
    _scan_execution()
    return summarize_diagnostics(output_root=output_root)


def verify_diagnostics(*, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    root = output_root.resolve()
    if root != OUTPUT_ROOT.resolve():
        raise ValueError("diagnostics require the canonical output root")
    study.verify_structured_180x35_2026(require_complete=True)
    contract = _read_json(root / "contract.json")
    if str(contract.get("contract_sha256", "")) != _canonical_digest(contract["contract"]):
        raise ValueError("diagnostic contract digest drifted")
    old_run = study._old_2025_run()
    normalization, normalization_sha256 = _checkpoint_normalization(old_run)
    view = _read_json(VIEW_2025)
    if (
        dict(view.get("normalization", {}) or {}) != normalization
        or str(dict(view["checkpoint_diagnostic"])["normalization_sha256"])
        != normalization_sha256
    ):
        raise ValueError("2025 checkpoint normalization binding drifted")
    for model_id in MODEL_ORDER:
        forecast = pd.read_parquet(_extended_prediction_path(model_id))
        if (
            forecast["trade_date"].nunique() != study.EXPECTED_EXTENDED_DATES
            or not bool(np.isfinite(forecast[["score", "predicted_exit_day"]]).all().all())
            or bool(forecast.duplicated(["date_idx", "symbol_idx"]).any())
        ):
            raise ValueError(f"extended prediction verification failed for {model_id}")
    old_keys = pd.read_parquet(_extended_prediction_path(MODEL_2025), columns=["date_idx", "symbol_idx"])
    new_keys = pd.read_parquet(_extended_prediction_path(MODEL_2026), columns=["date_idx", "symbol_idx"])
    if not old_keys.equals(new_keys):
        raise ValueError("checkpoint candidate keys are not identical")
    d7 = pd.read_csv(root / "d7_checkpoint_metrics.csv")
    if len(d7) != 16 or not bool(
        d7[d7["window_name"].astype(str).eq("strict_101")]["all_positions_resolved"].astype(bool).all()
    ):
        raise ValueError("D7 checkpoint comparison is incomplete")
    core = pd.read_csv(study.STUDY_ROOT / "d7_account_metrics.csv")
    observed = d7[d7["model_id"].astype(str).eq(MODEL_2026)].sort_values(
        ["strategy_id", "cost_scenario", "window_name"], kind="mergesort"
    )
    expected = core.sort_values(
        ["strategy_id", "cost_scenario", "window_name"], kind="mergesort"
    )
    if not np.allclose(
        observed["mark_to_market_equity"].to_numpy(dtype=np.float64),
        expected["mark_to_market_equity"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("unified D7 replay does not reproduce the core 2026 account")
    scan = pd.read_csv(root / "execution_scan_metrics.csv")
    if (
        len(scan) != 244
        or set(scan["top_k"].astype(int)) != {1, 3}
        or set(scan["cost_scenario"].astype(str)) != set(study.COST_SCENARIOS)
        or set(scan["policy_name"].astype(str))
        != {policy.name for policy in SCAN_POLICIES}
    ):
        raise ValueError("execution scan coverage is incomplete")
    if not (root / "diagnostic_summary.json").is_file():
        raise FileNotFoundError(root / "diagnostic_summary.json")
    return {
        "status": "ok",
        "checkpoint_models": list(MODEL_ORDER),
        "extended_candidate_rows": int(len(old_keys)),
        "d7_account_rows": int(len(d7)),
        "execution_scan_rows": int(len(scan)),
        "core_d7_reproduced": True,
        "qdp_changed": False,
        "provider_called": False,
        "checkpoint_changed": False,
        "live_state_changed": False,
    }
