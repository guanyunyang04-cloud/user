"""Quality-liquidity decision universe for the continuous turning-path atlas.

The full adjusted price history remains the object used to identify path turns.
The point-in-time pool is applied only at the causal probe-onset date, so pool
membership cannot turn suspensions or temporary ineligibility into fake adjacent
price observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_hot_path_atlas as atlas
from daily_research.path_policy import seq100_turning_path_atlas as turning

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_turning_path_quality_pool_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_turning_path_quality_pool_v1"
)
STUDY_ID = "seq100_turning_path_quality_pool_v1"
MANIFEST_SCHEMA = "seq100_turning_path_quality_pool_analysis/1"
BUILDER_VERSION = 2
FUTURE_RESOLUTION_FILTER = "NOT right_censored AND resolution_date_idx > onset_date_idx"


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = atlas._resolve_path(path)
    study = atlas._read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("turning_quality_pool_study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if int(source.get("forbidden_year", -1)) != 2026:
        raise ValueError("turning_quality_pool_forbidden_year_contract_missing")
    if str(source.get("maximum_outcome_date")) != "2025-12-31":
        raise ValueError("turning_quality_pool_outcome_cutoff_mismatch")
    if str(source.get("quality_pool_name")) != "quality_liquidity_pit":
        raise ValueError("turning_quality_pool_name_mismatch")
    semantics = dict(study.get("universe_semantics", {}) or {})
    if str(semantics.get("membership_time")) != "probe onset date":
        raise ValueError("turning_quality_pool_membership_time_mismatch")
    if bool(dict(study.get("analysis", {}) or {}).get("profit_claim_allowed")):
        raise ValueError("turning_quality_pool_profit_claim_forbidden")
    return study


def _records_by_year(records: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for year in range(2012, 2026):
        record = dict(records.get(str(year), {}) or {})
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"turning_quality_pool_spine_missing:{year}:{path}")
        paths.append(path)
    return paths


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    source = dict(study["source"])
    turning_manifest_path = atlas._resolve_path(str(source["turning_manifest"]))
    turning_analysis_path = atlas._resolve_path(
        str(source["turning_analysis_manifest"])
    )
    training_ready_path = atlas._resolve_path(str(source["training_ready_manifest"]))
    quality_pool_path = atlas._resolve_path(str(source["quality_pool_manifest"]))
    for path in (
        turning_manifest_path,
        turning_analysis_path,
        training_ready_path,
        quality_pool_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"turning_quality_pool_source_missing:{path}")

    turning_manifest = atlas._read_json(turning_manifest_path)
    turning_analysis = atlas._read_json(turning_analysis_path)
    expected_turning = str(source["expected_turning_fingerprint"])
    if str(turning_manifest.get("experiment_fingerprint")) != expected_turning:
        raise ValueError("turning_quality_pool_turning_fingerprint_mismatch")
    if str(turning_analysis.get("experiment_fingerprint")) != expected_turning:
        raise ValueError("turning_quality_pool_analysis_fingerprint_mismatch")

    training_ready = atlas._read_json(training_ready_path)
    if training_ready.get("schema") != "seq100_quality_liquidity_training_ready/v1":
        raise ValueError("turning_quality_pool_training_ready_schema_mismatch")
    row_contract = dict(training_ready.get("row_spine_contract", {}) or {})
    if int(row_contract.get("consumed_forbidden_dependency_row_count", -1)) != 0:
        raise ValueError("turning_quality_pool_forbidden_dependency_consumed")
    spine_paths = _records_by_year(
        dict(training_ready.get("daily_quality_row_spine", {}) or {})
    )

    pool_manifest = atlas._read_json(quality_pool_path)
    if str(pool_manifest.get("pool_name")) != str(source["quality_pool_name"]):
        raise ValueError("turning_quality_pool_manifest_name_mismatch")
    if str(pool_manifest.get("common_support_hash")) != str(
        source["expected_quality_pool_hash"]
    ):
        raise ValueError("turning_quality_pool_hash_mismatch")
    if int(pool_manifest.get("row_count", -1)) != int(
        source["expected_quality_pool_rows"]
    ):
        raise ValueError("turning_quality_pool_row_count_mismatch")
    if int(pool_manifest.get("forbidden_2026_rows", -1)) != 0:
        raise ValueError("turning_quality_pool_contains_2026")

    turning_source = dict(turning_manifest["source_contract"])
    dense_path = Path(str(dict(turning_source["dense_base"])["path"]))
    events_path = Path(str(dict(turning_manifest["outputs"])["events"]))
    episodes_path = Path(str(dict(turning_manifest["outputs"])["episodes"]))
    rule_manifest_path = Path(str(dict(turning_source["rule_manifest"])["path"]))
    neutral_manifest_path = Path(str(dict(turning_source["neutral_manifest"])["path"]))
    for path in (
        dense_path,
        events_path,
        episodes_path,
        rule_manifest_path,
        neutral_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"turning_quality_pool_path_missing:{path}")
    rule_paths = [
        Path(value) for _, value in sorted(dict(turning_source["rule_panels"]).items())
    ]
    neutral_paths = [
        Path(value)
        for _, value in sorted(dict(turning_source["neutral_panels"]).items())
    ]

    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": atlas._sha256_file(study_path),
        "turning_manifest_sha256": atlas._sha256_file(turning_manifest_path),
        "turning_analysis_sha256": atlas._sha256_file(turning_analysis_path),
        "training_ready_sha256": atlas._sha256_file(training_ready_path),
        "quality_pool_sha256": atlas._sha256_file(quality_pool_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    contract = {
        "turning_manifest": atlas._file_record(turning_manifest_path),
        "turning_analysis_manifest": atlas._file_record(turning_analysis_path),
        "training_ready_manifest": atlas._file_record(training_ready_path),
        "quality_pool_manifest": atlas._file_record(quality_pool_path),
        "quality_pool_hash": str(pool_manifest["common_support_hash"]),
        "quality_pool_rows": int(pool_manifest["row_count"]),
        "quality_spines": [
            atlas._file_record(path, include_hash=False) for path in spine_paths
        ],
        "dense_base": atlas._file_record(dense_path, include_hash=False),
        "turning_events": atlas._file_record(events_path, include_hash=False),
        "turning_episodes": atlas._file_record(episodes_path, include_hash=False),
        "rule_manifest": atlas._file_record(rule_manifest_path),
        "rule_panels": [
            atlas._file_record(path, include_hash=False) for path in rule_paths
        ],
        "neutral_manifest": atlas._file_record(neutral_manifest_path),
        "neutral_panels": [
            atlas._file_record(path, include_hash=False) for path in neutral_paths
        ],
        "fingerprint_payload": payload,
    }
    return contract, fingerprint


def _quality_episode_query(
    episodes_path: Path,
    spine_paths: Sequence[Path],
    rule_paths: Sequence[Path],
    neutral_paths: Sequence[Path],
    features: Sequence[str],
) -> str:
    feature_columns = ",\n        ".join(
        f"r.{name}_rank AS {name}_rank" for name in features
    )
    complete = " AND ".join(
        f"r.{name}_rank IS NOT NULL AND isfinite(r.{name}_rank)" for name in features
    )
    return f"""
    WITH pool AS (
        SELECT candidate_id, year, trade_date, date_idx, symbol_idx,
               symbol, security_id
        FROM {atlas._parquet_scan(spine_paths)}
    )
    SELECT
        e.*,
        p.candidate_id AS pool_candidate_id,
        p.security_id,
        p.symbol_idx AS pool_symbol_idx,
        p.date_idx AS pool_date_idx,
        p.year AS signal_year,
        (r.symbol IS NOT NULL AND n.symbol IS NOT NULL) AS legacy_panel_covered,
        coalesce(({complete}), false) AS all_rank_features_complete,
        n.industry,
        n.circ_mv,
        n.total_mv,
        n.turnover_proxy_rank,
        n.size_bin_5,
        n.size_bin_10,
        {feature_columns}
    FROM read_parquet({atlas._sql_quote(episodes_path)}) e
    INNER JOIN pool p
      ON e.symbol = p.symbol AND e.onset_date = p.trade_date
    LEFT JOIN {atlas._parquet_scan(rule_paths)} r
      ON e.symbol = r.symbol AND e.onset_date = r.trade_date
    LEFT JOIN {atlas._parquet_scan(neutral_paths)} n
      ON e.symbol = n.symbol AND e.onset_date = n.trade_date
    """


def _episode_summary(
    connection: duckdb.DuckDBPyConnection, panel_path: Path
) -> pd.DataFrame:
    return connection.execute(
        f"""
        SELECT
            probe_multiplier,
            major_threshold_multiplier,
            episode_type,
            outcome,
            signal_year AS evaluation_year,
            count(*) AS episodes,
            count(*) FILTER (WHERE resolution_date_idx = onset_date_idx)
                AS same_day_resolved,
            count(*) FILTER (WHERE resolution_date_idx > onset_date_idx)
                AS future_resolved,
            same_day_resolved / NULLIF(episodes, 0)::DOUBLE
                AS same_day_share,
            count(DISTINCT onset_date) AS dates,
            median(resolution_delay_market_days) AS median_resolution_days,
            median(onset_reversal_log_return) AS median_onset_reversal,
            median(signed_leg_move_to_anchor) AS median_leg_move
        FROM read_parquet({atlas._sql_quote(panel_path)})
        WHERE NOT right_censored
        GROUP BY ALL
        ORDER BY probe_multiplier, major_threshold_multiplier,
                 episode_type, outcome, evaluation_year
        """
    ).fetchdf()


def _feature_contrasts(
    connection: duckdb.DuckDBPyConnection,
    panel_path: Path,
    feature_names: Sequence[str],
    *,
    hac_lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode_type in ("up_pullback", "down_rebound"):
        target = turning._target_expression(episode_type)
        for feature in feature_names:
            cells = connection.execute(
                f"""
                WITH labeled AS (
                    SELECT *, {target} AS target
                    FROM read_parquet({atlas._sql_quote(panel_path)})
                    WHERE episode_type = '{episode_type}'
                      AND {FUTURE_RESOLUTION_FILTER}
                      AND {feature} IS NOT NULL
                      AND isfinite({feature})
                )
                SELECT
                    probe_multiplier,
                    major_threshold_multiplier,
                    onset_date,
                    avg({feature}) FILTER (WHERE target = 1.0) AS positive_mean,
                    avg({feature}) FILTER (WHERE target = 0.0) AS negative_mean,
                    count(*) FILTER (WHERE target = 1.0) AS positive_rows,
                    count(*) FILTER (WHERE target = 0.0) AS negative_rows
                FROM labeled
                GROUP BY probe_multiplier, major_threshold_multiplier, onset_date
                HAVING positive_rows > 0 AND negative_rows > 0
                ORDER BY probe_multiplier, major_threshold_multiplier, onset_date
                """
            ).fetchdf()
            for keys, group in cells.groupby(
                ["probe_multiplier", "major_threshold_multiplier"], sort=True
            ):
                probe, major = keys
                difference = (group["positive_mean"] - group["negative_mean"]).to_numpy(
                    dtype=np.float64
                )
                estimate = atlas._hac_mean(difference, lag=int(hac_lag))
                p_value = float(
                    min(
                        1.0,
                        2.0
                        * turning.rules._one_sided_positive_p(
                            abs(float(estimate["mean"])),
                            float(estimate["se"]),
                        ),
                    )
                )
                rows.append(
                    {
                        "probe_multiplier": int(probe),
                        "major_threshold_multiplier": int(major),
                        "episode_type": episode_type,
                        "feature": feature,
                        "matched_dates": len(group),
                        "positive_rows": int(group["positive_rows"].sum()),
                        "negative_rows": int(group["negative_rows"].sum()),
                        **{
                            f"difference_{key}": value
                            for key, value in estimate.items()
                        },
                        "p_two_sided": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["bh_q"] = np.nan
    for positions in result.groupby(
        ["episode_type", "probe_multiplier", "major_threshold_multiplier"],
        sort=True,
    ).groups.values():
        index = list(positions)
        result.loc[index, "bh_q"] = turning.rules._benjamini_hochberg(
            result.loc[index, "p_two_sided"].to_numpy(float)
        )
    return result


def _annual_feature_contrasts(
    connection: duckdb.DuckDBPyConnection,
    panel_path: Path,
    feature_names: Sequence[str],
    *,
    hac_lag: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode_type in ("up_pullback", "down_rebound"):
        target = turning._target_expression(episode_type)
        for feature in feature_names:
            cells = connection.execute(
                f"""
                WITH labeled AS (
                    SELECT *, {target} AS target
                    FROM read_parquet({atlas._sql_quote(panel_path)})
                    WHERE episode_type = '{episode_type}'
                      AND {FUTURE_RESOLUTION_FILTER}
                      AND {feature} IS NOT NULL
                      AND isfinite({feature})
                )
                SELECT
                    signal_year,
                    probe_multiplier,
                    major_threshold_multiplier,
                    onset_date,
                    avg({feature}) FILTER (WHERE target = 1.0) AS positive_mean,
                    avg({feature}) FILTER (WHERE target = 0.0) AS negative_mean,
                    count(*) FILTER (WHERE target = 1.0) AS positive_rows,
                    count(*) FILTER (WHERE target = 0.0) AS negative_rows
                FROM labeled
                GROUP BY signal_year, probe_multiplier,
                         major_threshold_multiplier, onset_date
                HAVING positive_rows > 0 AND negative_rows > 0
                ORDER BY signal_year, probe_multiplier,
                         major_threshold_multiplier, onset_date
                """
            ).fetchdf()
            for keys, group in cells.groupby(
                [
                    "signal_year",
                    "probe_multiplier",
                    "major_threshold_multiplier",
                ],
                sort=True,
            ):
                year, probe, major = keys
                difference = (group["positive_mean"] - group["negative_mean"]).to_numpy(
                    dtype=np.float64
                )
                estimate = atlas._hac_mean(difference, lag=int(hac_lag))
                rows.append(
                    {
                        "episode_type": episode_type,
                        "feature": feature,
                        "evaluation_year": int(year),
                        "probe_multiplier": int(probe),
                        "major_threshold_multiplier": int(major),
                        "matched_dates": len(group),
                        "positive_rows": int(group["positive_rows"].sum()),
                        "negative_rows": int(group["negative_rows"].sum()),
                        **{
                            f"difference_{key}": value
                            for key, value in estimate.items()
                        },
                    }
                )
    return pd.DataFrame(rows)


def _path_profile_query(
    panel_path: Path,
    dense_path: Path,
    study: Mapping[str, Any],
) -> str:
    analysis = dict(study["analysis"])
    start = int(analysis["profile_offset_start"])
    end = int(analysis["profile_offset_end"])
    cap = int(analysis["maximum_profile_events_per_date_outcome"])
    reference = dict(analysis["profile_reference_scale_pair"])
    probe = int(reference["probe_multiplier"])
    major = int(reference["major_multiplier"])
    return f"""
    WITH eligible AS (
        SELECT *, row_number() OVER (
            PARTITION BY episode_type, outcome, onset_date
            ORDER BY hash(symbol || '|' || onset_date || '|' || outcome)
        ) AS sample_order
        FROM read_parquet({atlas._sql_quote(panel_path)})
        WHERE {FUTURE_RESOLUTION_FILTER}
          AND probe_multiplier = {probe}
          AND major_threshold_multiplier = {major}
    ), sampled AS (
        SELECT * FROM eligible WHERE sample_order <= {cap}
    ), paths AS (
        SELECT e.episode_type, e.outcome,
               d.date_idx - e.onset_date_idx AS relative_market_day,
               ln(d.adj_close) - e.onset_log_price AS relative_log_price,
               ln(d.adj_close) - e.anchor_log_price AS anchor_relative_log_price
        FROM sampled e
        INNER JOIN read_parquet({atlas._sql_quote(dense_path)}) d
          ON e.symbol = d.symbol
         AND d.date_idx BETWEEN e.onset_date_idx + {start}
                            AND e.onset_date_idx + {end}
        WHERE d.bar_valid AND d.adj_close > 0 AND isfinite(d.adj_close)
    )
    SELECT episode_type, outcome, relative_market_day, count(*) AS rows,
           quantile_cont(relative_log_price, 0.25) AS relative_log_price_q25,
           median(relative_log_price) AS relative_log_price_median,
           quantile_cont(relative_log_price, 0.75) AS relative_log_price_q75,
           median(anchor_relative_log_price) AS anchor_relative_log_price_median
    FROM paths
    GROUP BY episode_type, outcome, relative_market_day
    ORDER BY episode_type, outcome, relative_market_day
    """


def _contrast_stability(
    annual: pd.DataFrame,
    discovery_years: Sequence[int],
    evaluation_years: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "episode_type",
        "feature",
        "probe_multiplier",
        "major_threshold_multiplier",
    ]
    discovery_set = {int(value) for value in discovery_years}
    evaluation_set = {int(value) for value in evaluation_years}
    for values, group in annual.groupby(keys, sort=True):
        discovery = group[group["evaluation_year"].isin(discovery_set)]
        evaluation = group[group["evaluation_year"].isin(evaluation_set)]
        discovery_mean = float(discovery["difference_mean"].mean())
        evaluation_mean = float(evaluation["difference_mean"].mean())
        direction = 1 if discovery_mean > 0 else -1 if discovery_mean < 0 else 0
        same_sign = int(
            (np.sign(evaluation["difference_mean"].to_numpy(float)) == direction).sum()
        )
        rows.append(
            {
                **dict(zip(keys, values)),
                "discovery_years": len(discovery),
                "discovery_difference_mean": discovery_mean,
                "discovery_positive_years": int(
                    discovery["difference_mean"].gt(0).sum()
                ),
                "evaluation_years": len(evaluation),
                "evaluation_difference_mean": evaluation_mean,
                "evaluation_positive_years": int(
                    evaluation["difference_mean"].gt(0).sum()
                ),
                "evaluation_same_discovery_sign_years": same_sign,
                "evaluation_min_difference": float(evaluation["difference_mean"].min()),
                "evaluation_max_difference": float(evaluation["difference_mean"].max()),
            }
        )
    return pd.DataFrame(rows)


def _representative_examples(
    connection: duckdb.DuckDBPyConnection,
    panel_path: Path,
    study: Mapping[str, Any],
) -> pd.DataFrame:
    analysis = dict(study["analysis"])
    years = ",".join(
        str(int(value)) for value in analysis["representative_example_years"]
    )
    reference = dict(analysis["profile_reference_scale_pair"])
    return connection.execute(
        f"""
        WITH base AS (
            SELECT *
            FROM read_parquet({atlas._sql_quote(panel_path)})
            WHERE signal_year IN ({years})
              AND {FUTURE_RESOLUTION_FILTER}
              AND probe_multiplier = {int(reference["probe_multiplier"])}
              AND major_threshold_multiplier = {int(reference["major_multiplier"])}
        ), centers AS (
            SELECT episode_type, outcome,
                   median(onset_reversal_log_return) AS m_reversal,
                   median(signed_leg_move_to_anchor) AS m_leg,
                   median(resolution_delay_market_days) AS m_delay,
                   stddev_samp(onset_reversal_log_return) AS s_reversal,
                   stddev_samp(signed_leg_move_to_anchor) AS s_leg,
                   stddev_samp(resolution_delay_market_days) AS s_delay
            FROM base GROUP BY episode_type, outcome
        ), deviations AS (
            SELECT b.*, c.* EXCLUDE (episode_type, outcome),
                   abs(b.onset_reversal_log_return-c.m_reversal)
                     / greatest(coalesce(c.s_reversal, 0), 1e-9)
                   + abs(b.signed_leg_move_to_anchor-c.m_leg)
                     / greatest(coalesce(c.s_leg, 0), 1e-9)
                   + abs(b.resolution_delay_market_days-c.m_delay)
                     / greatest(coalesce(c.s_delay, 0), 1e-9) AS distance
            FROM base b INNER JOIN centers c USING (episode_type, outcome)
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY episode_type, outcome
                ORDER BY distance, onset_date, symbol
            ) AS row_number
            FROM deviations
        )
        SELECT * EXCLUDE (m_reversal, m_leg, m_delay, s_reversal, s_leg,
                          s_delay, distance, row_number)
        FROM ranked WHERE row_number = 1
        ORDER BY episode_type, outcome
        """
    ).fetchdf()


def _runtime_record(
    connection: duckdb.DuckDBPyConnection, study: Mapping[str, Any]
) -> dict[str, Any]:
    settings = connection.execute(
        "SELECT current_setting('memory_limit'), current_setting('threads')"
    ).fetchone()
    resolved = atlas._duckdb_runtime_resources(study)
    return {
        **resolved,
        "active_memory_limit": str(settings[0]),
        "active_threads": int(settings[1]),
    }


def _research_record(
    audit: Mapping[str, Any],
    episode_summary: pd.DataFrame,
    stability: pd.DataFrame,
    runtime: Mapping[str, Any],
    study: Mapping[str, Any],
) -> str:
    totals = (
        episode_summary.groupby(
            [
                "probe_multiplier",
                "major_threshold_multiplier",
                "episode_type",
                "outcome",
            ],
            sort=True,
        )[["episodes", "same_day_resolved", "future_resolved"]]
        .sum()
        .reset_index()
    )
    reference = dict(dict(study["analysis"])["profile_reference_scale_pair"])
    candidates = stability[
        stability["probe_multiplier"].eq(int(reference["probe_multiplier"]))
        & stability["major_threshold_multiplier"].eq(int(reference["major_multiplier"]))
    ].copy()
    candidates["discovery_abs"] = candidates["discovery_difference_mean"].abs()
    leading = (
        candidates.sort_values(
            ["episode_type", "discovery_abs", "feature"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("episode_type", sort=True)
        .head(10)
    )
    lines = [
        "# Quality-Liquidity Turning-Path Atlas V1",
        "",
        "The decision universe is the established point-in-time daily quality-liquidity pool. Turning geometry is still computed from each stock's complete valid adjusted-close history. Pool membership is applied only at the causal probe onset, so temporary non-membership, suspensions, or missing data cannot become artificial adjacent path observations.",
        "",
        "The daily pool is used here. Complete five-minute coverage is not part of stock selection and will be imposed only when testing whether intraday data adds information.",
        "",
        "## Resolved multi-scale outcomes",
        "",
        totals.to_markdown(index=False),
        "",
        "## Discovery-selected same-date feature differences and later stability",
        "",
        "Positive differences mean terminal-top minus recovered-pullback for up-pullback episodes, and confirmed-bottom minus continued-downtrend for down-rebound episodes. Same-day resolutions are excluded because their outcome is already known at the probe close. Features below are ranked only by 2012-2018 absolute difference; 2019-2025 columns are a separate chronological stability check.",
        "",
        leading[
            [
                "episode_type",
                "feature",
                "probe_multiplier",
                "major_threshold_multiplier",
                "discovery_difference_mean",
                "evaluation_difference_mean",
                "evaluation_same_discovery_sign_years",
                "evaluation_years",
                "evaluation_min_difference",
                "evaluation_max_difference",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Data audit",
        "",
        f"- Point-in-time pool rows: {int(audit['quality_pool_rows']):,}",
        f"- Pool-filtered path episodes: {int(audit['episode_rows']):,}",
        f"- Resolved episodes: {int(audit['resolved_rows']):,}",
        f"- Same-day already-resolved episodes: {int(audit['same_day_resolved_rows']):,}",
        f"- Future-resolved prediction episodes: {int(audit['future_resolved_rows']):,}",
        f"- Right-censored episodes: {int(audit['right_censored_rows']):,}",
        f"- Duplicate episode keys: {int(audit['duplicate_keys']):,}",
        f"- Onset/pool date-index mismatches: {int(audit['date_index_mismatches']):,}",
        f"- 2026 rows: {int(audit['forbidden_rows']):,}",
        f"- Legacy signal-panel coverage: {float(audit['legacy_panel_coverage']):.2%}",
        f"- All legacy rank features complete: {float(audit['all_rank_complete_share']):.2%}",
        "",
        "Missing legacy ranks remain in the panel with explicit flags. They are not removed from path counts and can be imputed or given missing indicators only inside a later chronological model.",
        "",
        "## Runtime",
        "",
        f"- Adaptive resource policy: {bool(runtime['adaptive'])}",
        f"- Available memory observed: {runtime.get('available_memory_mb')} MB",
        f"- DuckDB memory limit used: {runtime['active_memory_limit']}",
        f"- DuckDB threads used: {int(runtime['active_threads'])}",
        "",
        "## Boundary",
        "",
        "This remains a descriptive atlas, not a trading result. Same-day completed transitions are deterministic state recognitions, not predictions. For unresolved episodes, outcome identities and post-onset paths use future information. A feature difference is useful only if a model fitted exclusively on already resolved earlier episodes improves later-year log score, calibration, and eventual executable utility. No fixed directional-change scale is declared to be the true top or bottom.",
        "",
    ]
    return "\n".join(lines)


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    contract, fingerprint = _source_contract(study_path, study)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    atlas._write_json(
        progress_path,
        {"status": "building_quality_episode_panel", "fingerprint": fingerprint},
    )

    episodes_path = Path(str(dict(contract["turning_episodes"])["path"]))
    events_path = Path(str(dict(contract["turning_events"])["path"]))
    dense_path = Path(str(dict(contract["dense_base"])["path"]))
    spine_paths = [Path(str(record["path"])) for record in contract["quality_spines"]]
    rule_paths = [Path(str(record["path"])) for record in contract["rule_panels"]]
    neutral_paths = [Path(str(record["path"])) for record in contract["neutral_panels"]]
    rule_manifest = atlas._read_json(Path(str(dict(contract["rule_manifest"])["path"])))
    features = [str(value) for value in rule_manifest["feature_ranks"]]
    feature_columns = [f"{name}_rank" for name in features]
    panel_path = output_root / "quality_episode_features.parquet"
    connection = atlas._connect(output_root, study)
    try:
        runtime = _runtime_record(connection, study)
        atlas._copy_query(
            connection,
            _quality_episode_query(
                episodes_path,
                spine_paths,
                rule_paths,
                neutral_paths,
                features,
            ),
            panel_path,
        )
        audit_row = (
            connection.execute(
                f"""
            SELECT
                count(*) AS episode_rows,
                count(*) - count(DISTINCT
                    symbol || '|' || major_threshold_multiplier::VARCHAR || '|'
                    || probe_multiplier::VARCHAR || '|' || episode_order::VARCHAR
                ) AS duplicate_keys,
                count(*) FILTER (WHERE NOT right_censored) AS resolved_rows,
                count(*) FILTER (
                    WHERE NOT right_censored
                      AND resolution_date_idx = onset_date_idx
                ) AS same_day_resolved_rows,
                count(*) FILTER (
                    WHERE NOT right_censored
                      AND resolution_date_idx > onset_date_idx
                ) AS future_resolved_rows,
                count(*) FILTER (WHERE right_censored) AS right_censored_rows,
                count(*) FILTER (WHERE onset_date_idx != pool_date_idx)
                    AS date_index_mismatches,
                count(*) FILTER (WHERE signal_year = 2026) AS forbidden_rows,
                count(*) FILTER (WHERE legacy_panel_covered)
                    AS legacy_panel_rows,
                count(*) FILTER (WHERE all_rank_features_complete)
                    AS all_rank_complete_rows,
                count(DISTINCT symbol) AS symbols,
                count(DISTINCT onset_date) AS dates,
                min(onset_date) AS minimum_date,
                max(onset_date) AS maximum_date
            FROM read_parquet({atlas._sql_quote(panel_path)})
            """
            )
            .fetchdf()
            .iloc[0]
            .to_dict()
        )
        audit = {
            key: (int(value) if isinstance(value, (np.integer, int)) else str(value))
            for key, value in audit_row.items()
        }
        audit["quality_pool_rows"] = int(contract["quality_pool_rows"])
        audit["legacy_panel_coverage"] = int(audit["legacy_panel_rows"]) / max(
            int(audit["episode_rows"]), 1
        )
        audit["all_rank_complete_share"] = int(audit["all_rank_complete_rows"]) / max(
            int(audit["episode_rows"]), 1
        )
        if int(audit["duplicate_keys"]) != 0:
            raise ValueError("turning_quality_pool_duplicate_episode_keys")
        if int(audit["date_index_mismatches"]) != 0:
            raise ValueError("turning_quality_pool_date_index_mismatch")
        if int(audit["forbidden_rows"]) != 0:
            raise ValueError("turning_quality_pool_forbidden_rows")

        episode_summary = _episode_summary(connection, panel_path)
        contrasts = _feature_contrasts(
            connection,
            panel_path,
            [*turning.EPISODE_STATE_FEATURES, *feature_columns],
            hac_lag=int(dict(study["analysis"])["hac_lag"]),
        )
        annual = _annual_feature_contrasts(
            connection,
            panel_path,
            [*turning.EPISODE_STATE_FEATURES, *feature_columns],
            hac_lag=int(dict(study["analysis"])["hac_lag"]),
        )
        period = dict(study["period"])
        stability = _contrast_stability(
            annual,
            period["discovery_years"],
            period["rolling_evaluation_years"],
        )
        profiles = connection.execute(
            _path_profile_query(panel_path, dense_path, study)
        ).fetchdf()
        examples = _representative_examples(connection, panel_path, study)

        episode_summary_path = output_root / "episode_summary.csv"
        contrasts_path = output_root / "feature_contrasts.csv"
        annual_path = output_root / "annual_feature_contrasts.csv"
        stability_path = output_root / "feature_stability.csv"
        profile_path = output_root / "path_profiles.csv"
        examples_path = output_root / "representative_examples.csv"
        profile_figure = output_root / "figures/path_profiles.png"
        examples_figure = output_root / "figures/representative_examples.png"
        episode_summary.to_csv(episode_summary_path, index=False)
        contrasts.to_csv(contrasts_path, index=False)
        annual.to_csv(annual_path, index=False)
        stability.to_csv(stability_path, index=False)
        profiles.to_csv(profile_path, index=False)
        examples.to_csv(examples_path, index=False)
        turning._plot_profiles(profiles, profile_figure)
        turning._plot_examples(
            connection, examples, dense_path, events_path, examples_figure
        )
    finally:
        connection.close()

    record_path = output_root / "research_record.md"
    record_path.write_text(
        _research_record(audit, episode_summary, stability, runtime, study),
        encoding="utf-8",
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": fingerprint,
        "study": atlas._file_record(study_path),
        "source_contract": contract,
        "runtime": runtime,
        "audit": audit,
        "outputs": {
            "quality_episode_features": str(panel_path.resolve()),
            "episode_summary": str(episode_summary_path.resolve()),
            "feature_contrasts": str(contrasts_path.resolve()),
            "annual_feature_contrasts": str(annual_path.resolve()),
            "feature_stability": str(stability_path.resolve()),
            "path_profiles": str(profile_path.resolve()),
            "representative_examples": str(examples_path.resolve()),
            "path_profile_figure": str(profile_figure.resolve()),
            "representative_examples_figure": str(examples_figure.resolve()),
            "research_record": str(record_path.resolve()),
        },
        "training_performed": False,
        "portfolio_selection_performed": False,
        "profit_claim_allowed": False,
    }
    atlas._write_json(output_root / "analysis_manifest.json", manifest)
    atlas._write_json(
        progress_path,
        {
            "status": "completed",
            "analysis_manifest": str(
                (output_root / "analysis_manifest.json").resolve()
            ),
        },
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the quality-liquidity continuous turning-path atlas."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_study(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
