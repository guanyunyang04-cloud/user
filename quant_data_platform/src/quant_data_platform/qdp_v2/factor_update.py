from __future__ import annotations

"""Keep the mutable daily adjustment-factor table aligned with market daily."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_data_platform.domains.contracts import (
    DataDomain,
    DatePartitionFetchRequest,
    DomainFetchRequest,
)
from quant_data_platform.providers import BaostockProvider
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.repair import append_active_shard, resolve_active_domain


FACTOR_DOMAIN = "adjust_factor"
DAILY_DOMAIN = "market_daily_raw"
FACTOR_SEMANTICS = "trusted_source_back_adjust_factor_ratio_normalized_to_qdp_asof"


class FactorTailUpdateError(RuntimeError):
    pass


def missing_factor_keys(
    *,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None = None,
) -> pd.DataFrame:
    daily = resolve_active_domain(DAILY_DOMAIN, workspace_root=workspace_root)
    factor = resolve_active_domain(FACTOR_DOMAIN, workspace_root=workspace_root)
    with open_guarded_duckdb(
        temp_directory=_runtime_root(workspace_root) / "factor_inventory_spill",
        threads=4,
    ) as connection:
        return connection.execute(
            """
            WITH missing AS (
              SELECT DISTINCT cast(d.symbol AS VARCHAR) AS symbol,
                              cast(d.trade_date AS VARCHAR) AS trade_date
              FROM read_parquet(?, union_by_name=true) d
              LEFT JOIN read_parquet(?, union_by_name=true) f
                ON cast(f.symbol AS VARCHAR)=cast(d.symbol AS VARCHAR)
               AND cast(f.trade_date AS VARCHAR)=cast(d.trade_date AS VARCHAR)
              WHERE cast(d.trade_date AS VARCHAR) BETWEEN ? AND ?
                AND f.symbol IS NULL
            ), prior AS (
              SELECT m.symbol,
                     m.trade_date,
                     arg_max(cast(f.trade_date AS VARCHAR), cast(f.trade_date AS VARCHAR)) AS prior_date,
                     arg_max(try_cast(f.fore_adjust_factor AS DOUBLE), cast(f.trade_date AS VARCHAR)) AS prior_fore,
                     arg_max(try_cast(f.back_adjust_factor AS DOUBLE), cast(f.trade_date AS VARCHAR)) AS prior_back,
                     arg_max(try_cast(f.adjust_factor AS DOUBLE), cast(f.trade_date AS VARCHAR)) AS prior_adjust
              FROM missing m
              LEFT JOIN read_parquet(?, union_by_name=true) f
                ON cast(f.symbol AS VARCHAR)=m.symbol
               AND cast(f.trade_date AS VARCHAR)<m.trade_date
              GROUP BY m.symbol,m.trade_date
            )
            SELECT * FROM prior ORDER BY symbol,trade_date
            """,
            [
                [str(item) for item in daily.shard_paths],
                [str(item) for item in factor.shard_paths],
                _date_text(start_date),
                _date_text(end_date),
                [str(item) for item in factor.shard_paths],
            ],
        ).fetchdf()


def build_factor_tail_rows(
    missing: pd.DataFrame,
    *,
    provider: BaostockProvider,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = (
        "symbol",
        "trade_date",
        "fore_adjust_factor",
        "back_adjust_factor",
        "adjust_factor",
        "factor_provider",
        "factor_semantics",
        "source",
        "factor_source_date",
        "ffill_days",
    )
    if missing.empty:
        return pd.DataFrame(columns=columns), {
            "missing_key_count": 0,
            "event_count": 0,
            "continued_event_count": 0,
            "initialized_symbol_count": 0,
        }
    required = {"symbol", "trade_date", "prior_date", "prior_fore", "prior_back", "prior_adjust"}
    absent = sorted(required.difference(missing.columns))
    if absent:
        raise FactorTailUpdateError(f"factor_tail_missing_inventory_columns:{absent}")
    event_frames: list[pd.DataFrame] = []
    for trade_date in sorted(missing["trade_date"].astype(str).unique()):
        result = provider.fetch_date_partition(
            DatePartitionFetchRequest(
                domain=DataDomain.ADJUST_FACTOR_EVENT,
                trade_date=trade_date,
                fetch_mode="date_events",
            )
        )
        frame = result.data.copy()
        if not frame.empty:
            event_frames.append(frame)
    events = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    missing_symbols = set(missing["symbol"].astype(str))
    if not events.empty:
        events = events.loc[events["provider_symbol"].astype(str).isin(missing_symbols)].copy()
        events["divid_operate_date"] = events["divid_operate_date"].astype(str).str.slice(0, 10)
        for column in ("adjust_factor", "back_adjust_factor"):
            events[column] = pd.to_numeric(events[column], errors="coerce")
        if (
            events[["adjust_factor", "back_adjust_factor"]].isna().any().any()
            or events[["adjust_factor", "back_adjust_factor"]].le(0).any().any()
        ):
            raise FactorTailUpdateError("factor_tail_baostock_event_factor_invalid")
        duplicate = events.duplicated(["provider_symbol", "divid_operate_date"], keep=False)
        if duplicate.any():
            conflicts = (
                events.loc[duplicate]
                .groupby(["provider_symbol", "divid_operate_date"])[
                    ["adjust_factor", "back_adjust_factor"]
                ]
                .nunique()
                .gt(1)
                .any(axis=1)
            )
            if conflicts.any():
                raise FactorTailUpdateError("factor_tail_baostock_event_duplicate_conflict")
            events = events.drop_duplicates(["provider_symbol", "divid_operate_date"], keep="last")

    event_symbols = tuple(sorted(set(events.get("provider_symbol", pd.Series(dtype=str)).astype(str))))
    history = pd.DataFrame()
    if event_symbols:
        result = provider.fetch_domain(
            DomainFetchRequest(
                domain=DataDomain.ADJUST_FACTOR,
                symbols=event_symbols,
                start_date="1990-01-01",
                end_date=str(missing["trade_date"].max()),
            )
        )
        if result.error_report:
            raise FactorTailUpdateError(
                f"factor_tail_baostock_history_errors:{len(result.error_report)}"
            )
        history = result.data.copy()
        history["trade_date"] = history["trade_date"].astype(str).str.slice(0, 10)
        for column in ("adjust_factor", "back_adjust_factor"):
            history[column] = pd.to_numeric(history[column], errors="coerce")
        history = history.loc[
            np.isfinite(history["adjust_factor"])
            & history["adjust_factor"].gt(0)
            & np.isfinite(history["back_adjust_factor"])
            & history["back_adjust_factor"].gt(0)
        ]

    event_by_key = {
        (str(row.provider_symbol), str(row.divid_operate_date)): {
            "adjust_factor": float(row.adjust_factor),
            "back_adjust_factor": float(row.back_adjust_factor),
        }
        for row in events.itertuples(index=False)
    } if not events.empty else {}
    history_by_symbol = {
        str(symbol): group.sort_values("trade_date", kind="stable").reset_index(drop=True)
        for symbol, group in history.groupby("symbol", sort=False)
    } if not history.empty else {}

    generated: dict[str, tuple[str, float]] = {}
    rows: list[dict[str, Any]] = []
    continued_event_count = 0
    initialized_symbol_count = 0
    for item in missing.sort_values(["symbol", "trade_date"], kind="stable").itertuples(index=False):
        symbol = str(item.symbol)
        trade_date = str(item.trade_date)
        prior_available = bool(
            pd.notna(item.prior_date)
            and np.isfinite(float(item.prior_adjust))
            and float(item.prior_adjust) > 0
        )
        prior_date = str(item.prior_date) if prior_available else ""
        current = float(item.prior_adjust) if prior_available else 1.0
        previous_generated = generated.get(symbol)
        if previous_generated is not None and (
            not prior_date or previous_generated[0] > prior_date
        ):
            current = float(previous_generated[1])
        has_qdp_baseline = prior_available or previous_generated is not None
        if not has_qdp_baseline:
            initialized_symbol_count += 1
        event = event_by_key.get((symbol, trade_date))
        if event is not None and has_qdp_baseline:
            symbol_history = history_by_symbol.get(symbol)
            if symbol_history is None or symbol_history.empty:
                raise FactorTailUpdateError(f"factor_tail_baostock_history_missing:{symbol}")
            prior_history = symbol_history.loc[symbol_history["trade_date"].lt(trade_date)]
            current_history = symbol_history.loc[symbol_history["trade_date"].eq(trade_date)]
            if prior_history.empty or current_history.empty:
                raise FactorTailUpdateError(
                    f"factor_tail_baostock_ratio_anchor_missing:{symbol}:{trade_date}"
                )
            history_current_adjust = float(current_history.iloc[-1]["adjust_factor"])
            if not np.isclose(
                history_current_adjust,
                float(event["adjust_factor"]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise FactorTailUpdateError(
                    f"factor_tail_baostock_event_history_mismatch:{symbol}:{trade_date}"
                )
            provider_current = float(current_history.iloc[-1]["back_adjust_factor"])
            if not np.isclose(
                provider_current,
                float(event["back_adjust_factor"]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise FactorTailUpdateError(
                    f"factor_tail_baostock_back_factor_history_mismatch:{symbol}:{trade_date}"
                )
            provider_prior = float(prior_history.iloc[-1]["back_adjust_factor"])
            current *= provider_current / provider_prior
            continued_event_count += 1
        if not np.isfinite(current) or current <= 0:
            raise FactorTailUpdateError(f"factor_tail_result_invalid:{symbol}:{trade_date}")
        generated[symbol] = (trade_date, current)
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "fore_adjust_factor": current,
                "back_adjust_factor": current,
                "adjust_factor": current,
                "factor_provider": "baostock",
                "factor_semantics": FACTOR_SEMANTICS,
                "source": (
                    "baostock.back_adjust_factor_ratio+qdp_prior_carry"
                    if has_qdp_baseline
                    else "qdp_first_observation_normalized_to_1"
                ),
                "factor_source_date": trade_date,
                "ffill_days": 0,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    return frame, {
        "missing_key_count": int(len(missing)),
        "event_count": int(len(events)),
        "continued_event_count": int(continued_event_count),
        "initialized_symbol_count": int(initialized_symbol_count),
    }


def run_factor_tail_update(
    *,
    start_date: str,
    end_date: str,
    workspace_root: str | Path | None = None,
    provider: BaostockProvider | None = None,
    apply: bool = True,
) -> dict[str, Any]:
    missing = missing_factor_keys(
        start_date=start_date,
        end_date=end_date,
        workspace_root=workspace_root,
    )
    source = provider or BaostockProvider()
    try:
        frame, metrics = build_factor_tail_rows(missing, provider=source)
    finally:
        if provider is None:
            source.close()
    if frame.empty:
        return {"status": "already_complete", **metrics, "row_count": 0}
    if frame.duplicated(["trade_date", "symbol"]).any():
        raise FactorTailUpdateError("factor_tail_duplicate_output_keys")
    result: dict[str, Any] = {"status": "planned", **metrics, "row_count": int(len(frame))}
    if apply:
        result["commit"] = append_active_shard(
            FACTOR_DOMAIN,
            frame,
            f"fill missing daily adjustment factors from BaoStock event ratios {_date_text(start_date)}..{_date_text(end_date)}",
            workspace_root=workspace_root,
        )
        remaining = missing_factor_keys(
            start_date=start_date,
            end_date=end_date,
            workspace_root=workspace_root,
        )
        if not remaining.empty:
            raise FactorTailUpdateError(f"factor_tail_post_commit_missing:{len(remaining)}")
        result["status"] = "updated"
        result["remaining_missing_key_count"] = 0
    return result


def _runtime_root(workspace_root: str | Path | None) -> Path:
    workspace = Path(workspace_root or Path.cwd()).resolve()
    root = (workspace / "quant_data_platform" / "data" / "qdp_runtime").resolve()
    if workspace not in root.parents:
        raise FactorTailUpdateError(f"factor_tail_runtime_outside_workspace:{root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _date_text(value: str) -> str:
    return pd.Timestamp(str(value)).strftime("%Y-%m-%d")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill missing current QDP adjustment-factor keys.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_factor_tail_update(
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        workspace_root=str(args.workspace_root or "") or None,
        apply=not bool(args.dry_run),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") in {"planned", "updated", "already_complete"} else 2


__all__ = [
    "FactorTailUpdateError",
    "build_factor_tail_rows",
    "missing_factor_keys",
    "run_factor_tail_update",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
