"""Causal stock-attention gate evaluated at every decision minute."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .contracts import KEY_COLUMNS, MinuteV2Config, MinuteV2Error

EVENT_FLAG_COLUMNS = (
    "event_extreme_return",
    "event_volume_shock",
    "event_industry_leadership",
    "event_vwap_cross",
    "event_liquidity_anchor",
    "event_background_control",
)

# Event files are an index into the full causal feature table.  Keeping the
# gate inputs and decisions here makes the event artifact cheap to scan while
# preserving enough provenance to audit why a row was selected.  Model
# features are joined back by KEY_COLUMNS when a training/replay consumer
# needs them.
EVENT_CONTEXT_COLUMNS = (
    "industry_name",
    "adjust_factor",
    "valid_open",
    "valid_high",
    "valid_low",
    "valid_close",
    "daily_liquidity_rank",
    "stock_return_rank",
    "stock_return_5m_rank",
    "stock_volume_acceleration_rank",
    "stock_amount_rank",
    "industry_strength_rank",
    "industry_stock_return_rank",
    "industry_stock_amount_rank",
    "market_breadth_positive",
    "vwap_deviation",
    "previous_vwap_deviation",
    "return_5m",
)

DEFAULT_RECALL_TOP_K = (1, 3, 5)

CANDIDATE_COLUMNS = (
    "candidate_attention",
    "candidate_background_control",
    "candidate_selected",
    "candidate_reason_mask",
)


def event_query(
    *,
    feature_view: str = "minute_features",
    config: MinuteV2Config,
) -> str:
    """Keep causal stock candidates while preserving every decision-time group."""

    config.validate()
    background_percent = int(config.candidate_background_percent)
    liquidity = float(config.minimum_daily_liquidity_rank)
    stock_floor = float(config.candidate_stock_rank_floor)
    industry_floor = float(config.candidate_industry_rank_floor)
    industry_stock_floor = float(config.candidate_industry_stock_rank_floor)
    volume_floor = float(config.candidate_volume_rank_floor)
    return f"""
        WITH flags AS (
            SELECT
                *,
                (
                    stock_return_rank >= {stock_floor}
                    OR stock_return_5m_rank >= {stock_floor}
                ) AS event_extreme_return,
                (
                    stock_volume_acceleration_rank >= {volume_floor}
                    AND stock_amount_rank >= 0.50
                    AND ABS(COALESCE(return_from_previous_close, 0.0)) >= 0.001
                ) AS event_volume_shock,
                (
                    industry_strength_rank >= {industry_floor}
                    AND industry_stock_return_rank >= {industry_stock_floor}
                    AND COALESCE(industry_return_mean, 0.0) > 0.0
                ) AS event_industry_leadership,
                (
                    previous_vwap_deviation IS NOT NULL
                    AND vwap_deviation IS NOT NULL
                    AND SIGN(previous_vwap_deviation) <> SIGN(vwap_deviation)
                    AND ABS(COALESCE(return_5m, 0.0)) >= 0.001
                ) AS event_vwap_cross,
                stock_amount_rank >= 0.90 AS event_liquidity_anchor,
                (HASH(symbol, trade_date, bar_time) % 100) < {background_percent}
                    AS event_background_control
            FROM {feature_view}
        ),
        candidates AS (
            SELECT
                *,
                (
                    daily_liquidity_rank >= {liquidity}
                    AND (
                        event_extreme_return
                        OR event_volume_shock
                        OR event_industry_leadership
                        OR event_vwap_cross
                        OR event_liquidity_anchor
                    )
                ) AS candidate_attention,
                (
                    daily_liquidity_rank >= {liquidity}
                    AND event_background_control
                ) AS candidate_background_control
            FROM flags
        ),
        encoded AS (
            SELECT
                *,
                candidate_attention OR candidate_background_control AS candidate_selected,
                CAST(event_extreme_return AS INTEGER)
                    + 2 * CAST(event_volume_shock AS INTEGER)
                    + 4 * CAST(event_industry_leadership AS INTEGER)
                    + 8 * CAST(event_vwap_cross AS INTEGER)
                    + 16 * CAST(event_liquidity_anchor AS INTEGER)
                    + 32 * CAST(event_background_control AS INTEGER)
                    AS candidate_reason_mask
            FROM candidates
        )
        SELECT
            symbol,
            trade_date,
            bar_time,
            {", ".join(EVENT_CONTEXT_COLUMNS)},
            {", ".join(EVENT_FLAG_COLUMNS)},
            candidate_attention,
            candidate_background_control,
            candidate_selected,
            candidate_reason_mask,
            candidate_reason_mask AS event_mask
        FROM encoded
        WHERE candidate_selected
        ORDER BY trade_date, bar_time, symbol
    """


def build_event_frame(
    features: pd.DataFrame,
    *,
    config: MinuteV2Config | None = None,
    connection: Any | None = None,
) -> pd.DataFrame:
    current = config or MinuteV2Config()
    missing = sorted(set(KEY_COLUMNS).difference(features.columns))
    if missing:
        raise MinuteV2Error(f"minute_v2_event_keys_missing:{','.join(missing)}")
    owned = connection is None
    con = duckdb.connect(":memory:") if owned else connection
    try:
        con.register("minute_features", features)
        result = con.execute(event_query(config=current)).fetchdf()
    finally:
        if owned:
            con.close()
    if result.empty:
        raise MinuteV2Error("minute_v2_candidate_gate_empty")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_duplicate_event_keys")
    source_sizes = features.groupby(["trade_date", "bar_time"], sort=False).size()
    result_sizes = result.groupby(["trade_date", "bar_time"], sort=False).size()
    if len(result_sizes) != len(source_sizes):
        raise MinuteV2Error(
            f"minute_v2_candidate_time_groups_missing:{len(result_sizes)}:{len(source_sizes)}"
        )
    expected_sizes = source_sizes.reindex(result_sizes.index)
    if result_sizes.gt(expected_sizes).any():
        raise MinuteV2Error("minute_v2_candidate_group_exceeds_universe")
    return result


def _validate_recall_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    require_target: str | None = None,
) -> None:
    missing = sorted(set(KEY_COLUMNS).difference(frame.columns))
    if require_target is not None and require_target not in frame.columns:
        missing.append(require_target)
    if missing:
        raise MinuteV2Error(
            f"minute_v2_recall_{name}_columns_missing:{','.join(sorted(set(missing)))}"
        )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error(f"minute_v2_recall_{name}_duplicate_keys")


def audit_candidate_recall(
    base: pd.DataFrame,
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame | None = None,
    *,
    target: str = "label_return_5m",
    top_k: Sequence[int] = DEFAULT_RECALL_TOP_K,
) -> dict[str, Any]:
    """Measure whether the causal gate retains the eventual best rows.

    ``base`` and ``outcomes`` represent the complete universe.  ``candidates``
    is treated only as a set of primary keys; no outcome column is consulted
    when determining membership.  The function is therefore an audit of the
    gate, not a way to tune it using future returns.
    """

    _validate_recall_frame(base, name="base")
    _validate_recall_frame(candidates, name="candidates")
    outcome_frame = base if outcomes is None and target in base.columns else outcomes
    if outcome_frame is None:
        raise MinuteV2Error("minute_v2_recall_outcomes_required")
    _validate_recall_frame(outcome_frame, name="outcomes", require_target=target)
    requested_k = tuple(sorted({int(value) for value in top_k if int(value) > 0}))
    if not requested_k:
        raise MinuteV2Error("minute_v2_recall_top_k_invalid")

    keys = list(KEY_COLUMNS)
    universe = base.loc[:, keys].merge(
        outcome_frame.loc[:, [*keys, target]], how="left", on=keys, validate="one_to_one"
    )
    candidate_keys = candidates.loc[:, keys].drop_duplicates().copy()
    candidate_keys["_candidate_selected"] = True
    universe = universe.merge(candidate_keys, how="left", on=keys, validate="one_to_one")
    universe["_candidate_selected"] = (
        universe["_candidate_selected"].astype("boolean").fillna(False).astype(bool)
    )
    universe[target] = pd.to_numeric(universe[target], errors="coerce")
    finite = np.isfinite(
        pd.to_numeric(universe[target], errors="coerce").to_numpy(dtype=float)
    )
    universe = universe.loc[finite].copy()
    if universe.empty:
        raise MinuteV2Error("minute_v2_recall_no_finite_outcomes")

    group_columns = ["trade_date", "bar_time"]
    group_count = int(universe.groupby(group_columns, sort=False).ngroups)
    candidate_count = int(universe["_candidate_selected"].sum())
    group_sizes = universe.groupby(group_columns, sort=False).size()
    recalls: dict[str, float | None] = {}
    hits: dict[str, int] = {}
    eligible_groups: dict[str, int] = {}
    for k in requested_k:
        hit_count = 0
        eligible = 0
        for _, group in universe.groupby(group_columns, sort=False):
            if group.empty:
                continue
            take = min(int(k), len(group))
            ranked = group.sort_values(
                [target, "symbol"], ascending=[False, True], kind="stable"
            ).head(take)
            eligible += 1
            hit_count += int(ranked["_candidate_selected"].any())
        key = f"top_{k}"
        hits[key] = hit_count
        eligible_groups[key] = eligible
        recalls[key] = hit_count / eligible if eligible else None

    candidate_group_counts = universe.loc[universe["_candidate_selected"]].groupby(
        group_columns, sort=False
    ).size()
    return {
        "schema": "quantlab.minute_v2_candidate_recall/1",
        "target": str(target),
        "base_rows": int(len(base)),
        "outcome_rows": int(len(outcome_frame)),
        "finite_outcome_rows": int(len(universe)),
        "candidate_rows": candidate_count,
        "candidate_fraction_of_finite_outcomes": candidate_count / len(universe),
        "group_count": group_count,
        "groups_without_candidates": int(group_count - len(candidate_group_counts)),
        "minimum_candidate_rows_per_group": int(candidate_group_counts.min())
        if len(candidate_group_counts)
        else 0,
        "maximum_candidate_rows_per_group": int(candidate_group_counts.max())
        if len(candidate_group_counts)
        else 0,
        "minimum_universe_rows_per_group": int(group_sizes.min()),
        "maximum_universe_rows_per_group": int(group_sizes.max()),
        "top_k_hits": hits,
        "top_k_eligible_groups": eligible_groups,
        "top_k_recall": recalls,
        "candidate_membership_source": "primary-key-only; target excluded from gate membership",
    }


def audit_candidate_recall_files(
    base_path: str | Path,
    candidate_path: str | Path,
    outcome_path: str | Path,
    *,
    target: str = "label_return_5m",
    top_k: Sequence[int] = DEFAULT_RECALL_TOP_K,
) -> dict[str, Any]:
    """Run the recall audit from Parquet without materializing feature columns."""

    connection = duckdb.connect(":memory:")
    try:
        def literal(path: str | Path) -> str:
            return "'" + str(Path(path).resolve()).replace("'", "''") + "'"

        base_scan = f"read_parquet({literal(base_path)})"
        candidate_scan = f"read_parquet({literal(candidate_path)})"
        outcome_scan = f"read_parquet({literal(outcome_path)})"
        key_projection = ",".join(KEY_COLUMNS)
        top_values = tuple(sorted({int(value) for value in top_k if int(value) > 0}))
        if not top_values:
            raise MinuteV2Error("minute_v2_recall_top_k_invalid")
        query = f"""
            WITH base_keys AS (
                SELECT {key_projection} FROM {base_scan}
            ),
            candidates AS (
                SELECT DISTINCT {key_projection} FROM {candidate_scan}
            ),
            outcomes AS (
                SELECT {key_projection}, CAST({target} AS DOUBLE) AS target
                FROM {outcome_scan}
            ),
            universe AS (
                SELECT b.*, o.target,
                       c.symbol IS NOT NULL AS candidate_selected
                FROM base_keys b
                JOIN outcomes o USING({key_projection})
                LEFT JOIN candidates c USING({key_projection})
                WHERE isfinite(o.target)
            ),
            ranked AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY trade_date, bar_time
                    ORDER BY target DESC, symbol
                ) AS rank_in_group
                FROM universe
            )
            SELECT
                (SELECT count(*) FROM base_keys) AS base_rows,
                (SELECT count(*) FROM outcomes) AS outcome_rows,
                count(*) AS finite_outcome_rows,
                count(*) FILTER (WHERE candidate_selected) AS candidate_rows,
                count(DISTINCT trade_date || 'T' || bar_time) AS group_count,
                count(DISTINCT trade_date || 'T' || bar_time)
                    FILTER (WHERE candidate_selected) AS candidate_group_count
            FROM ranked
        """
        summary = connection.execute(query).fetchone()
        if not summary:
            raise MinuteV2Error("minute_v2_recall_file_summary_empty")
        base_rows, outcome_rows, finite_rows, candidate_rows, groups, candidate_groups = summary
        report: dict[str, Any] = {
            "schema": "quantlab.minute_v2_candidate_recall/1",
            "target": str(target),
            "base_rows": int(base_rows),
            "outcome_rows": int(outcome_rows),
            "finite_outcome_rows": int(finite_rows),
            "candidate_rows": int(candidate_rows),
            "candidate_fraction_of_finite_outcomes": float(candidate_rows / finite_rows),
            "group_count": int(groups),
            "groups_without_candidates": int(groups - candidate_groups),
            "candidate_membership_source": "primary-key-only; target excluded from gate membership",
            "top_k_recall": {},
        }
        for k in top_values:
            row = connection.execute(
                f"""
                WITH base_keys AS (SELECT {key_projection} FROM {base_scan}),
                candidates AS (SELECT DISTINCT {key_projection} FROM {candidate_scan}),
                outcomes AS (
                    SELECT {key_projection}, CAST({target} AS DOUBLE) AS target
                    FROM {outcome_scan} WHERE isfinite(CAST({target} AS DOUBLE))
                ),
                ranked AS (
                    SELECT o.*, c.symbol IS NOT NULL AS candidate_selected,
                           ROW_NUMBER() OVER (
                               PARTITION BY o.trade_date, o.bar_time
                               ORDER BY o.target DESC, o.symbol
                           ) AS rank_in_group
                    FROM outcomes o
                    JOIN base_keys b USING({key_projection})
                    LEFT JOIN candidates c USING({key_projection})
                ),
                groups AS (
                    SELECT trade_date,bar_time,
                           bool_or(candidate_selected) AS hit
                    FROM ranked WHERE rank_in_group <= {int(k)}
                    GROUP BY trade_date,bar_time
                )
                SELECT count(*) FILTER(WHERE hit), count(*) FROM groups
                """
            ).fetchone()
            hits, eligible = row or (0, 0)
            report["top_k_recall"][f"top_{k}"] = (
                float(hits / eligible) if eligible else None
            )
        return report
    finally:
        connection.close()


__all__ = [
    "CANDIDATE_COLUMNS",
    "DEFAULT_RECALL_TOP_K",
    "EVENT_CONTEXT_COLUMNS",
    "EVENT_FLAG_COLUMNS",
    "audit_candidate_recall",
    "audit_candidate_recall_files",
    "build_event_frame",
    "event_query",
]
