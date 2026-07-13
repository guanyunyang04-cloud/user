from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from daily_research.path_policy.seq100_exit_policy_audit import (
    ACTIVE_EXECUTION_PATH,
    ORACLE_POLICY,
    PREDICTED_POLICY,
    PROFILES,
    QDP_ACTIVE_PATH,
    SOURCE_PACK_MANIFEST,
    SOURCE_STUDY_ROOT,
    YEARS,
    CandidateCompleteAuditPack,
    _next_valid_exit_indices,
    _sha256_file,
    _source_run_dir,
    _workspace_path,
    _write_csv,
    _write_json,
    cashflow_batch,
    oracle_executable_outcome_batch,
    resolve_planned_exit_batch,
)


DEFAULT_AUDIT_ROOT = Path(
    "daily_research/output/path_policy/studies/"
    "seq100_candidate_complete_exit_policy_audit_2022_2025_v1"
)
COMMON_ALLOCATION_CNY = 100_000.0
FIXED_POLICY = "fixed_h2"


def _rank_bucket(score_rank: int) -> str:
    rank = int(score_rank)
    if rank == 1:
        return "rank1"
    if 2 <= rank <= 3:
        return "rank2_3"
    if 4 <= rank <= 5:
        return "rank4_5"
    if 6 <= rank <= 10:
        return "rank6_10"
    raise ValueError(f"rank bucket is defined only for score ranks 1..10: {rank}")


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    return float(finite.mean()) if finite.size else math.nan


def _finite_rate(values: pd.Series, predicate) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    return float(predicate(finite).mean()) if finite.size else math.nan


def _dataset_shards(*, qdp_root: Path, domain: str, dataset_id: str) -> list[Path]:
    manifest_path = qdp_root / "datasets" / domain / dataset_id / "dataset.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    shards = [qdp_root / str(item["path"]) for item in payload.get("shards", [])]
    if not shards or any(not path.is_file() for path in shards):
        raise FileNotFoundError(f"dataset shards are incomplete: {domain}/{dataset_id}")
    return shards


def _read_filtered_shards(
    paths: Sequence[Path],
    *,
    filters: list[tuple[str, str, Any]],
) -> pd.DataFrame:
    frames = [pd.read_parquet(path, filters=filters) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _candidate_bridge_rows(
    *,
    pack: CandidateCompleteAuditPack,
    source_study_root: str | Path,
    years: Sequence[int],
    profiles: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for year in years:
            run_dir = _source_run_dir(source_study_root, profile, int(year))
            topk = pd.read_parquet(run_dir / "topk_candidates.parquet")
            topk = topk[topk["score_rank"].astype(int).le(10)].copy()
            topk["trade_date"] = pd.to_datetime(topk["trade_date"], errors="raise").dt.strftime(
                "%Y-%m-%d"
            )
            topk = topk.sort_values(["trade_date", "score_rank"], kind="mergesort")
            for trade_date, raw_group in topk.groupby("trade_date", sort=True):
                group = raw_group.sort_values("score_rank", kind="mergesort").reset_index(drop=True)
                if len(group) != 10:
                    raise ValueError(f"frozen Top10 is incomplete for {profile}/{year}/{trade_date}")
                date_idx = int(pack.date_to_idx[str(trade_date)])
                symbol_idx = np.asarray(
                    [pack.symbol_to_idx[str(symbol)] for symbol in group["symbol"]],
                    dtype=np.int64,
                )
                entry_prices, exit_prices, exit_sellable = pack.execution_paths(date_idx, symbol_idx)
                entry_filled = np.asarray(pack.entry_filled[date_idx, symbol_idx], dtype=bool)
                recorded_fill = group["realized_plan_entry_filled"].astype(bool).to_numpy()
                if not bool(np.array_equal(entry_filled, recorded_fill)):
                    raise ValueError(f"entry-fill identity drift for {profile}/{year}/{trade_date}")
                next_valid = _next_valid_exit_indices(exit_prices, exit_sellable)

                fixed_plan = resolve_planned_exit_batch(
                    signal_date_idx=date_idx,
                    entry_filled=entry_filled,
                    entry_prices=entry_prices,
                    exit_prices=exit_prices,
                    next_valid_exit_idx=next_valid,
                    planned_days=2,
                    forward_days=pack.forward_days,
                    execution_days=pack.execution_days,
                    terminal_recovery_fraction=pack.terminal_recovery_fraction,
                )
                predicted_plan = resolve_planned_exit_batch(
                    signal_date_idx=date_idx,
                    entry_filled=entry_filled,
                    entry_prices=entry_prices,
                    exit_prices=exit_prices,
                    next_valid_exit_idx=next_valid,
                    planned_days=pd.to_numeric(
                        group["predicted_exit_day"], errors="raise"
                    ).to_numpy(dtype=np.float64),
                    forward_days=pack.forward_days,
                    execution_days=pack.execution_days,
                    terminal_recovery_fraction=pack.terminal_recovery_fraction,
                )
                fixed_cash = cashflow_batch(
                    allocated_cash=COMMON_ALLOCATION_CNY,
                    entry_filled=entry_filled,
                    entry_prices=entry_prices,
                    plan=fixed_plan,
                    date_values=pack.date_values,
                    contract=pack.contract,
                    slippage_multiplier=1.0,
                )
                predicted_cash = cashflow_batch(
                    allocated_cash=COMMON_ALLOCATION_CNY,
                    entry_filled=entry_filled,
                    entry_prices=entry_prices,
                    plan=predicted_plan,
                    date_values=pack.date_values,
                    contract=pack.contract,
                    slippage_multiplier=1.0,
                )
                oracle_plan, oracle_cash = oracle_executable_outcome_batch(
                    signal_date_idx=date_idx,
                    allocated_cash=COMMON_ALLOCATION_CNY,
                    entry_filled=entry_filled,
                    entry_prices=entry_prices,
                    exit_prices=exit_prices,
                    next_valid_exit_idx=next_valid,
                    date_values=pack.date_values,
                    contract=pack.contract,
                    slippage_multiplier=1.0,
                    forward_days=pack.forward_days,
                    execution_days=pack.execution_days,
                    terminal_recovery_fraction=pack.terminal_recovery_fraction,
                )

                raw_returns = exit_prices[:, 1 : pack.forward_days] / entry_prices[:, None] - 1.0
                raw_valid = np.isfinite(raw_returns)
                raw_unconstrained = np.max(
                    np.where(raw_valid, raw_returns, -np.inf), axis=1
                )
                raw_unconstrained[~raw_valid.any(axis=1)] = np.nan
                sellable_valid = raw_valid & exit_sellable[:, 1 : pack.forward_days]
                raw_sellable = np.max(
                    np.where(sellable_valid, raw_returns, -np.inf), axis=1
                )
                raw_sellable[~sellable_valid.any(axis=1)] = np.nan

                label_days = pd.to_numeric(group["best_exit_day_60d"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                label_returns = pd.to_numeric(
                    group["best_exit_close_return_60d"], errors="coerce"
                ).to_numpy(dtype=np.float64)
                raw_at_label = np.full(len(group), np.nan, dtype=np.float64)
                valid_label = (
                    np.isfinite(label_days)
                    & np.isfinite(label_returns)
                    & (np.rint(label_days) >= 1)
                    & (np.rint(label_days) <= pack.forward_days)
                )
                valid_rows = np.where(valid_label)[0]
                valid_days = np.rint(label_days[valid_rows]).astype(np.int64) - 1
                raw_at_label[valid_rows] = (
                    exit_prices[valid_rows, valid_days] / entry_prices[valid_rows] - 1.0
                )
                implied_ratio = np.where(
                    valid_label & np.isfinite(raw_at_label) & ((1.0 + raw_at_label) > 0.0),
                    (1.0 + label_returns) / (1.0 + raw_at_label),
                    np.nan,
                )

                for row_idx, item in group.iterrows():
                    rank = int(item["score_rank"])
                    rows.append(
                        {
                            "profile": str(profile),
                            "development_year": int(year),
                            "trade_date": str(trade_date),
                            "symbol": str(item["symbol"]),
                            "score_rank": rank,
                            "rank_bucket": _rank_bucket(rank),
                            "entry_filled": bool(entry_filled[row_idx]),
                            "adjusted_best_exit_return": float(label_returns[row_idx]),
                            "best_exit_day": float(label_days[row_idx]),
                            "raw_return_at_label_exit_day": float(raw_at_label[row_idx]),
                            "implied_adjustment_ratio": float(implied_ratio[row_idx]),
                            "raw_unconstrained_max_return": float(raw_unconstrained[row_idx]),
                            "raw_sellable_max_return": float(raw_sellable[row_idx]),
                            "fixed_h2_net_return": float(fixed_cash.net_return[row_idx]),
                            "predicted_exit_net_return": float(predicted_cash.net_return[row_idx]),
                            "oracle_executable_net_return": float(oracle_cash.net_return[row_idx]),
                            "fixed_h2_terminal_recovery": bool(
                                fixed_plan.terminal_recovery[row_idx]
                                and fixed_cash.order_filled[row_idx]
                            ),
                            "predicted_exit_terminal_recovery": bool(
                                predicted_plan.terminal_recovery[row_idx]
                                and predicted_cash.order_filled[row_idx]
                            ),
                            "oracle_executable_terminal_recovery": bool(
                                oracle_plan.terminal_recovery[row_idx]
                                and oracle_cash.order_filled[row_idx]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _basis_summaries(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for key, group in rows.groupby(
        ["profile", "development_year", "rank_bucket"], sort=True
    ):
        adjusted = pd.to_numeric(group["adjusted_best_exit_return"], errors="coerce")
        raw_max = pd.to_numeric(group["raw_unconstrained_max_return"], errors="coerce")
        records.append(
            {
                "profile": str(key[0]),
                "development_year": int(key[1]),
                "rank_bucket": str(key[2]),
                "adjusted_best_exit_return": _safe_mean(adjusted),
                "raw_unconstrained_max_return": _safe_mean(raw_max),
                "raw_sellable_max_return": _safe_mean(group["raw_sellable_max_return"]),
                "adjusted_minus_raw_return": _safe_mean(adjusted - raw_max),
                "large_basis_gap_rate": _finite_rate(adjusted - raw_max, lambda value: value > 0.10),
                "mean_implied_adjustment_ratio": _safe_mean(group["implied_adjustment_ratio"]),
                "median_implied_adjustment_ratio": float(
                    pd.to_numeric(group["implied_adjustment_ratio"], errors="coerce").median()
                ),
                "implied_ratio_gt_1_1_rate": _finite_rate(
                    group["implied_adjustment_ratio"], lambda value: value > 1.10
                ),
                "entry_fill_rate": float(group["entry_filled"].mean()),
            }
        )
    yearly = pd.DataFrame(records)
    numeric = [
        column
        for column in yearly.columns
        if column not in {"profile", "development_year", "rank_bucket"}
    ]
    equal_year = yearly.groupby(
        ["profile", "rank_bucket"], as_index=False, sort=True
    )[numeric].mean()
    return yearly, equal_year


def _policy_bucket_summaries(
    rows: pd.DataFrame,
    *,
    audit_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = {
        FIXED_POLICY: ("fixed_h2_net_return", "fixed_h2_terminal_recovery"),
        PREDICTED_POLICY: (
            "predicted_exit_net_return",
            "predicted_exit_terminal_recovery",
        ),
        ORACLE_POLICY: (
            "oracle_executable_net_return",
            "oracle_executable_terminal_recovery",
        ),
    }
    records: list[dict[str, Any]] = []
    group_columns = ["profile", "development_year", "trade_date", "rank_bucket"]
    for policy, (return_column, terminal_column) in mapping.items():
        daily = rows.groupby(group_columns, as_index=False, sort=True).agg(
            bucket_mean_net_return=(return_column, "mean"),
            bucket_entry_fill_rate=("entry_filled", "mean"),
            bucket_terminal_recovery_rate=(terminal_column, "mean"),
        )
        yearly = daily.groupby(
            ["profile", "development_year", "rank_bucket"], as_index=False, sort=True
        )[
            [
                "bucket_mean_net_return",
                "bucket_entry_fill_rate",
                "bucket_terminal_recovery_rate",
            ]
        ].mean()
        yearly["policy"] = policy
        records.extend(yearly.to_dict("records"))
    yearly = pd.DataFrame(records)
    universe = pd.read_csv(audit_root / "year_policy_metrics.csv")
    universe = universe[
        universe["top_k"].astype(int).eq(10)
        & universe["cost_scenario"].astype(str).eq("base")
        & universe["policy"].astype(str).isin(mapping)
    ][
        ["profile", "development_year", "policy", "universe_mean_net_return"]
    ]
    yearly = yearly.merge(
        universe,
        on=["profile", "development_year", "policy"],
        how="left",
        validate="many_to_one",
    )
    if bool(yearly["universe_mean_net_return"].isna().any()):
        raise ValueError("audit universe benchmark is incomplete")
    yearly["bucket_alpha"] = (
        yearly["bucket_mean_net_return"] - yearly["universe_mean_net_return"]
    )
    numeric = [
        "bucket_mean_net_return",
        "universe_mean_net_return",
        "bucket_alpha",
        "bucket_entry_fill_rate",
        "bucket_terminal_recovery_rate",
    ]
    equal_year = yearly.groupby(
        ["profile", "policy", "rank_bucket"], as_index=False, sort=True
    )[numeric].mean()
    return yearly, equal_year


def _counterexample_evidence(
    *,
    rows: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    finite = rows[np.isfinite(rows["implied_adjustment_ratio"].to_numpy(dtype=np.float64))]
    if finite.empty:
        raise ValueError("price-basis bridge has no finite adjustment-ratio samples")
    sample = finite.sort_values("implied_adjustment_ratio", ascending=False).iloc[0]
    symbol = str(sample["symbol"])
    qdp_active = json.loads(_workspace_path(QDP_ACTIVE_PATH).read_text(encoding="utf-8-sig"))
    datasets = dict(qdp_active.get("datasets", {}) or {})
    factor_id = str(datasets["adjust_factor"])
    actions_id = str(datasets["corporate_actions"])
    qdp_root = _workspace_path(QDP_ACTIVE_PATH).resolve().parents[1]
    factor = _read_filtered_shards(
        _dataset_shards(qdp_root=qdp_root, domain="adjust_factor", dataset_id=factor_id),
        filters=[("symbol", "=", symbol)],
    )
    actions = _read_filtered_shards(
        _dataset_shards(
            qdp_root=qdp_root,
            domain="corporate_actions",
            dataset_id=actions_id,
        ),
        filters=[("symbol", "=", symbol)],
    )
    factor["trade_date"] = pd.to_datetime(factor["trade_date"], errors="raise")
    start = pd.Timestamp(str(sample["trade_date"]))
    end = start + pd.Timedelta(days=120)
    window = factor[factor["trade_date"].between(start, end)].copy()
    window = window.sort_values("trade_date", kind="mergesort")
    change = pd.Series(
        ~np.isclose(
            pd.to_numeric(window["back_adjust_factor"], errors="coerce"),
            pd.to_numeric(window["back_adjust_factor"], errors="coerce").shift(),
            rtol=0.0,
            atol=1.0e-12,
            equal_nan=True,
        ),
        index=window.index,
    )
    if len(change):
        change.iloc[0] = False
    action_count = 0
    if not actions.empty and "trade_date" in actions.columns:
        action_dates = pd.to_datetime(actions["trade_date"], errors="coerce")
        action_count = int(action_dates.between(start, end).sum())
    _write_csv(output_root / "adjust_factor_counterexample_window.csv", window)
    _write_csv(output_root / "corporate_action_counterexample_all.csv", actions)
    return {
        "symbol": symbol,
        "profile": str(sample["profile"]),
        "development_year": int(sample["development_year"]),
        "trade_date": str(sample["trade_date"]),
        "score_rank": int(sample["score_rank"]),
        "best_exit_day": float(sample["best_exit_day"]),
        "adjusted_best_exit_return": float(sample["adjusted_best_exit_return"]),
        "raw_return_at_label_exit_day": float(sample["raw_return_at_label_exit_day"]),
        "implied_adjustment_ratio": float(sample["implied_adjustment_ratio"]),
        "active_adjust_factor_dataset_id": factor_id,
        "active_corporate_actions_dataset_id": actions_id,
        "factor_window_row_count": int(len(window)),
        "factor_change_count": int(change.sum()),
        "corporate_action_count_in_window": action_count,
    }


def _metric_lookup(
    frame: pd.DataFrame,
    *,
    profile: str,
    rank_bucket: str,
) -> dict[str, float]:
    selected = frame[
        frame["profile"].astype(str).eq(profile)
        & frame["rank_bucket"].astype(str).eq(rank_bucket)
    ]
    if len(selected) != 1:
        raise ValueError(f"metric lookup is not unique: {profile}/{rank_bucket}")
    row = selected.iloc[0]
    return {
        key: float(row[key])
        for key in (
            "adjusted_best_exit_return",
            "raw_unconstrained_max_return",
            "raw_sellable_max_return",
            "implied_ratio_gt_1_1_rate",
            "entry_fill_rate",
        )
    }


def run_price_basis_diagnostic(
    *,
    source_study_root: str | Path = SOURCE_STUDY_ROOT,
    pack_manifest: str | Path = SOURCE_PACK_MANIFEST,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    years: Sequence[int] = YEARS,
    profiles: Sequence[str] = PROFILES,
) -> dict[str, Any]:
    output = _workspace_path(audit_root).resolve()
    if not (output / "year_policy_metrics.csv").is_file():
        raise FileNotFoundError("run the primary exit-policy audit before this diagnostic")
    qdp_before = _sha256_file(QDP_ACTIVE_PATH)
    active_path = _workspace_path(ACTIVE_EXECUTION_PATH)
    active_before = active_path.exists()
    active_sha_before = _sha256_file(active_path) if active_before else None
    pack = CandidateCompleteAuditPack(pack_manifest)
    rows = _candidate_bridge_rows(
        pack=pack,
        source_study_root=source_study_root,
        years=years,
        profiles=profiles,
    )
    expected_rows = sum(
        int(
            pd.read_parquet(
                _source_run_dir(source_study_root, profile, int(year))
                / "topk_candidates.parquet",
                columns=["score_rank"],
            )["score_rank"].astype(int).le(10).sum()
        )
        for profile in profiles
        for year in years
    )
    if len(rows) != expected_rows:
        raise AssertionError(f"selected candidate bridge dropped rows: {len(rows)} != {expected_rows}")
    if bool(rows.duplicated(["profile", "development_year", "trade_date", "score_rank"]).any()):
        raise AssertionError("selected candidate bridge contains duplicate rank keys")

    basis_year, basis_equal = _basis_summaries(rows)
    policy_year, policy_equal = _policy_bucket_summaries(rows, audit_root=output)
    counterexample = _counterexample_evidence(rows=rows, output_root=output)
    rows_path = output / "price_basis_selected_candidate_bridge.parquet"
    rows.to_parquet(rows_path, index=False)
    _write_csv(output / "price_basis_rank_bucket_year.csv", basis_year)
    _write_csv(output / "price_basis_rank_bucket_equal_year.csv", basis_equal)
    _write_csv(output / "common_100k_rank_policy_year.csv", policy_year)
    _write_csv(output / "common_100k_rank_policy_equal_year.csv", policy_equal)

    qdp_after = _sha256_file(QDP_ACTIVE_PATH)
    active_after = active_path.exists()
    active_sha_after = _sha256_file(active_path) if active_after else None
    if qdp_after != qdp_before:
        raise RuntimeError("QDP active changed during the read-only price-basis diagnostic")
    if active_after != active_before or active_sha_after != active_sha_before:
        raise RuntimeError("active execution changed during the read-only price-basis diagnostic")
    summary = {
        "schema_version": 1,
        "status": "completed",
        "diagnostic_id": "seq100_exit_policy_price_basis_diagnostic_2022_2025_v1",
        "training_performed": False,
        "profiles": list(profiles),
        "development_years": [int(year) for year in years],
        "common_per_name_allocation_cny": COMMON_ALLOCATION_CNY,
        "selected_candidate_row_count": int(len(rows)),
        "key_basis_metrics": {
            "baseline_rank1": _metric_lookup(
                basis_equal, profile="baseline", rank_bucket="rank1"
            ),
            "baseline_rank6_10": _metric_lookup(
                basis_equal, profile="baseline", rank_bucket="rank6_10"
            ),
            "hard_st_rank1": _metric_lookup(
                basis_equal, profile="hard_st", rank_bucket="rank1"
            ),
            "hard_st_rank6_10": _metric_lookup(
                basis_equal, profile="hard_st", rank_bucket="rank6_10"
            ),
        },
        "counterexample": counterexample,
        "artifacts": {
            "selected_candidate_bridge": str(rows_path.resolve()),
            "basis_rank_bucket_year": str((output / "price_basis_rank_bucket_year.csv").resolve()),
            "basis_rank_bucket_equal_year": str(
                (output / "price_basis_rank_bucket_equal_year.csv").resolve()
            ),
            "common_100k_rank_policy_year": str(
                (output / "common_100k_rank_policy_year.csv").resolve()
            ),
            "common_100k_rank_policy_equal_year": str(
                (output / "common_100k_rank_policy_equal_year.csv").resolve()
            ),
            "factor_counterexample": str(
                (output / "adjust_factor_counterexample_window.csv").resolve()
            ),
            "corporate_actions_counterexample": str(
                (output / "corporate_action_counterexample_all.csv").resolve()
            ),
        },
        "protected_objects": {
            "qdp_active_sha256_before": qdp_before.upper(),
            "qdp_active_sha256_after": qdp_after.upper(),
            "qdp_active_changed": False,
            "active_execution_present_before": active_before,
            "active_execution_present_after": active_after,
            "active_execution_sha256_before": active_sha_before,
            "active_execution_sha256_after": active_sha_after,
            "active_execution_changed": False,
        },
    }
    _write_json(output / "price_basis_diagnostic_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-training selected-rank price-basis diagnostic for the Seq100 exit audit."
    )
    parser.add_argument("--source-study-root", type=Path, default=SOURCE_STUDY_ROOT)
    parser.add_argument("--pack-manifest", type=Path, default=SOURCE_PACK_MANIFEST)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_price_basis_diagnostic(
        source_study_root=args.source_study_root,
        pack_manifest=args.pack_manifest,
        audit_root=args.audit_root,
        years=args.years,
        profiles=args.profiles,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_price_basis_diagnostic"]
