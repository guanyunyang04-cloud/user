"""Nonparametric complete-panel state maps and chronological rule probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_hot_path_atlas as atlas


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_hot_path_rules_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_hot_path_rules_v1"
)
STUDY_ID = "seq100_hot_path_rules_v1"
MANIFEST_SCHEMA = "seq100_hot_path_rules_manifest/1"
ANALYSIS_SCHEMA = "seq100_hot_path_rules_analysis/1"
BUILDER_VERSION = 1

FEATURE_EXPRESSIONS: dict[str, str] = {
    "attention": "p.attention_score",
    "attention_mean5": "p.attention_mean5",
    "amount_shock": "p.amount_shock_rank",
    "volume_shock": "p.volume_shock_rank",
    "range_shock": "p.range_shock_rank",
    "amount_cross_section": "p.amount_cross_section_rank",
    "ret1": "p.ret1_rank",
    "ret5": "p.ret5_rank",
    "position20": "p.position20_rank",
    "ret20": "p.ret_20d",
    "trend_efficiency20": "p.trend_efficiency20",
    "volatility20": "p.volatility20_prev",
    "pullback5": "p.pullback_from_high5",
    "distance_ma20": "p.distance_ma20",
    "close_location": "p.close_location_1d",
    "bar_open_close": "p.adj_close / NULLIF(p.adj_open, 0) - 1.0",
    "bar_range": "p.range_1d",
    "bar_mfe": "p.adj_high / NULLIF(p.adj_open, 0) - 1.0",
    "bar_mae": "p.adj_low / NULLIF(p.adj_open, 0) - 1.0",
    "bar_vwap_close": (
        "p.raw_close / NULLIF(p.amount / NULLIF(p.volume, 0), 0) - 1.0"
    ),
}

SUMMARY_METRICS = (
    "signal_share",
    "fill_rate",
    "complete20_rate",
    "selected_net_return_5",
    "selected_net_return_20",
    "d20_excess",
    "selected_up10_before_down5",
    "up10_before_down5_lift",
    "selected_mfe20",
    "mfe20_lift",
    "selected_mae20",
    "mae20_lift",
)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = atlas._resolve_path(path)
    study = atlas._read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("hot_path_rules_study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if int(source.get("forbidden_year", -1)) != 2026:
        raise ValueError("hot_path_rules_forbidden_year_contract_missing")
    if str(source.get("maximum_outcome_date")) != "2025-12-31":
        raise ValueError("hot_path_rules_outcome_cutoff_mismatch")
    features = [str(value) for value in study.get("feature_ranks", [])]
    unknown = sorted(set(features).difference(FEATURE_EXPRESSIONS))
    if not features or unknown:
        raise ValueError(f"hot_path_rules_feature_contract_invalid:{unknown}")
    return study


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], list[Path], str]:
    source = dict(study.get("source", {}) or {})
    atlas_manifest_path = atlas._resolve_path(str(source["atlas_manifest"]))
    if not atlas_manifest_path.is_file():
        raise FileNotFoundError("hot_path_rules_atlas_manifest_missing")
    atlas_manifest = atlas._read_json(atlas_manifest_path)
    expected = str(source.get("expected_atlas_fingerprint", ""))
    if str(atlas_manifest.get("experiment_fingerprint")) != expected:
        raise ValueError("hot_path_rules_atlas_fingerprint_mismatch")
    panel_paths = atlas._panel_paths(atlas_manifest)
    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": atlas._sha256_file(study_path),
        "atlas_manifest_sha256": atlas._sha256_file(atlas_manifest_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    contract = {
        "atlas_manifest": atlas._file_record(atlas_manifest_path),
        "atlas_experiment_fingerprint": expected,
        "source_panels": [
            atlas._file_record(path, include_hash=False) for path in panel_paths
        ],
        "source_formal_signal_rows": int(
            atlas_manifest.get("formal_signal_rows", 0)
        ),
        "fingerprint_payload": payload,
    }
    return contract, panel_paths, fingerprint


def _rank_expression(expression: str, alias: str) -> str:
    valid = f"({expression}) IS NOT NULL AND isfinite({expression})"
    return f"""
    CASE WHEN {valid} THEN
        (rank() OVER (
            PARTITION BY p.trade_date
            ORDER BY CASE WHEN {valid} THEN {expression} ELSE NULL END NULLS LAST
        ) - 1)::DOUBLE
        / NULLIF(
            count(*) FILTER (WHERE {valid}) OVER (PARTITION BY p.trade_date) - 1,
            0
        )
    ELSE NULL END AS {alias}_rank
    """.strip()


def _compact_panel_query(panel_path: Path, features: Sequence[str]) -> str:
    ranks = ",\n        ".join(
        _rank_expression(FEATURE_EXPRESSIONS[name], name) for name in features
    )
    return f"""
    SELECT
        p.symbol,
        p.trade_date,
        p.date_idx,
        CAST(left(p.trade_date, 4) AS INTEGER) AS signal_year,
        {ranks},
        p.entry_observed_legal,
        p.entry_buyable_approx,
        p.valid_close_days_5,
        p.valid_close_days_20,
        p.terminal_log_return_5,
        p.terminal_log_return_20,
        p.mfe_20,
        p.mae_20,
        p.up10_before_down5,
        p.up10_down5_same_day_ambiguous
    FROM read_parquet({atlas._sql_quote(panel_path)}) p
    """


def _manifest_valid(manifest: Mapping[str, Any], fingerprint: str) -> bool:
    records = list(manifest.get("compact_panels", []) or [])
    return (
        manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("status") == "prepared"
        and str(manifest.get("experiment_fingerprint")) == str(fingerprint)
        and len(records) == 14
        and all(Path(str(record.get("path", ""))).is_file() for record in records)
    )


def prepare_rule_panels(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    contract, panel_paths, fingerprint = _source_contract(study_path, study)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not force:
        current = atlas._read_json(manifest_path)
        if _manifest_valid(current, fingerprint):
            return current
        if str(current.get("experiment_fingerprint", "")) not in {"", fingerprint}:
            raise ValueError("hot_path_rules_existing_fingerprint_mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    compact_root = output_root / "compact_panels"
    compact_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    features = [str(value) for value in study["feature_ranks"]]
    connection = atlas._connect(output_root, study)
    records: list[dict[str, Any]] = []
    try:
        for source_path in panel_paths:
            year = int(source_path.stem.rsplit("_", 1)[-1])
            if year < 2012 or year > 2025:
                continue
            output_path = compact_root / f"rule_state_{year}.parquet"
            if force or not output_path.is_file():
                atlas._write_json(
                    progress_path,
                    {
                        "status": "building_compact_rule_panel",
                        "year": year,
                        "completed_years": [int(item["year"]) for item in records],
                        "experiment_fingerprint": fingerprint,
                    },
                )
                atlas._copy_query(
                    connection,
                    _compact_panel_query(source_path, features),
                    output_path,
                )
            audit = connection.execute(
                f"""
                SELECT
                    count(*) AS rows,
                    count(*) - count(DISTINCT symbol || '|' || trade_date)
                        AS duplicate_keys,
                    count(*) FILTER (WHERE signal_year = 2026) AS forbidden_rows,
                    min(trade_date) AS minimum_date,
                    max(trade_date) AS maximum_date
                FROM read_parquet({atlas._sql_quote(output_path)})
                """
            ).fetchdf().iloc[0].to_dict()
            if int(audit["duplicate_keys"]) != 0:
                raise ValueError(f"hot_path_rules_duplicate_keys:{year}")
            if int(audit["forbidden_rows"]) != 0:
                raise ValueError(f"hot_path_rules_forbidden_rows:{year}")
            records.append(
                {
                    "year": year,
                    **atlas._file_record(output_path, include_hash=False),
                    "rows": int(audit["rows"]),
                    "duplicate_keys": int(audit["duplicate_keys"]),
                    "minimum_date": str(audit["minimum_date"]),
                    "maximum_date": str(audit["maximum_date"]),
                }
            )
    finally:
        connection.close()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": fingerprint,
        "study": atlas._file_record(study_path),
        "source_contract": contract,
        "compact_panels": records,
        "rows": int(sum(item["rows"] for item in records)),
        "feature_ranks": features,
        "training_performed": False,
        "portfolio_selection_performed": False,
    }
    expected_rows = int(contract["source_formal_signal_rows"])
    if int(manifest["rows"]) != expected_rows:
        raise ValueError(
            f"hot_path_rules_row_count_mismatch:{manifest['rows']}:{expected_rows}"
        )
    atlas._write_json(manifest_path, manifest)
    atlas._write_json(
        progress_path,
        {"status": "prepared", "manifest": str(manifest_path.resolve())},
    )
    return manifest


def _marginal_date_query(
    paths: Sequence[Path],
    feature: str,
    *,
    quintile_count: int,
    cost_bps: float,
) -> str:
    scan = atlas._parquet_scan(paths)
    column = f"{feature}_rank"
    cost = float(cost_bps) / 10_000.0
    return f"""
    WITH valid AS (
        SELECT *,
            CAST(least({int(quintile_count) - 1},
                 greatest(0, floor({column} * {int(quintile_count)}))) AS INTEGER)
                AS feature_bin
        FROM {scan}
        WHERE {column} IS NOT NULL AND isfinite({column})
    ), universe AS (
        SELECT
            trade_date,
            signal_year,
            count(*) AS universe_signal_rows,
            avg(exp(terminal_log_return_5) - 1.0 - {cost:.12g}) FILTER (
                WHERE entry_buyable_approx
                  AND valid_close_days_5 = 5
                  AND terminal_log_return_5 IS NOT NULL
            ) AS universe_net_return_5,
            avg(exp(terminal_log_return_20) - 1.0 - {cost:.12g}) FILTER (
                WHERE entry_buyable_approx
                  AND valid_close_days_20 = 20
                  AND terminal_log_return_20 IS NOT NULL
            ) AS universe_net_return_20,
            avg(CAST(up10_before_down5 AS DOUBLE)) FILTER (
                WHERE entry_buyable_approx
                  AND valid_close_days_20 = 20
                  AND NOT up10_down5_same_day_ambiguous
            ) AS universe_up10_before_down5,
            avg(mfe_20) FILTER (
                WHERE entry_buyable_approx AND valid_close_days_20 = 20
            ) AS universe_mfe20,
            avg(mae_20) FILTER (
                WHERE entry_buyable_approx AND valid_close_days_20 = 20
            ) AS universe_mae20
        FROM valid
        GROUP BY trade_date, signal_year
    ), cells AS (
        SELECT
            trade_date,
            signal_year,
            feature_bin,
            count(*) AS signal_rows,
            count(*) FILTER (WHERE entry_buyable_approx) AS filled_rows,
            count(*) FILTER (
                WHERE entry_buyable_approx
                  AND valid_close_days_20 = 20
                  AND terminal_log_return_20 IS NOT NULL
            ) AS complete20_rows,
            avg(exp(terminal_log_return_5) - 1.0 - {cost:.12g}) FILTER (
                WHERE entry_buyable_approx
                  AND valid_close_days_5 = 5
                  AND terminal_log_return_5 IS NOT NULL
            ) AS selected_net_return_5,
            avg(exp(terminal_log_return_20) - 1.0 - {cost:.12g}) FILTER (
                WHERE entry_buyable_approx
                  AND valid_close_days_20 = 20
                  AND terminal_log_return_20 IS NOT NULL
            ) AS selected_net_return_20,
            avg(CAST(up10_before_down5 AS DOUBLE)) FILTER (
                WHERE entry_buyable_approx
                  AND valid_close_days_20 = 20
                  AND NOT up10_down5_same_day_ambiguous
            ) AS selected_up10_before_down5,
            avg(mfe_20) FILTER (
                WHERE entry_buyable_approx AND valid_close_days_20 = 20
            ) AS selected_mfe20,
            avg(mae_20) FILTER (
                WHERE entry_buyable_approx AND valid_close_days_20 = 20
            ) AS selected_mae20
        FROM valid
        GROUP BY trade_date, signal_year, feature_bin
    )
    SELECT
        '{feature}' AS feature,
        c.feature_bin,
        c.trade_date,
        c.signal_year,
        c.signal_rows / NULLIF(u.universe_signal_rows, 0)::DOUBLE AS signal_share,
        c.filled_rows / NULLIF(c.signal_rows, 0)::DOUBLE AS fill_rate,
        c.complete20_rows / NULLIF(c.filled_rows, 0)::DOUBLE AS complete20_rate,
        c.selected_net_return_5,
        c.selected_net_return_20,
        c.selected_net_return_20 - u.universe_net_return_20 AS d20_excess,
        c.selected_up10_before_down5,
        c.selected_up10_before_down5 - u.universe_up10_before_down5
            AS up10_before_down5_lift,
        c.selected_mfe20,
        c.selected_mfe20 - u.universe_mfe20 AS mfe20_lift,
        c.selected_mae20,
        c.selected_mae20 - u.universe_mae20 AS mae20_lift
    FROM cells c
    INNER JOIN universe u USING (trade_date, signal_year)
    ORDER BY c.trade_date, c.feature_bin
    """


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return result
    finite = values[finite_positions]
    order = np.argsort(finite, kind="mergesort")
    ranked = finite[order]
    count = len(ranked)
    adjusted = ranked * count / np.arange(1, count + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    ordered_positions = finite_positions[order]
    result[ordered_positions] = adjusted
    return result


def _one_sided_positive_p(mean: float, se: float) -> float:
    if not math.isfinite(mean) or not math.isfinite(se):
        return float("nan")
    if se <= 0:
        return 0.0 if mean > 0 else 1.0
    z_value = mean / se
    return float(0.5 * math.erfc(z_value / math.sqrt(2.0)))


def _summarize_period(
    date_cells: pd.DataFrame,
    *,
    period_name: str,
    years: set[int],
    hac_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = date_cells[date_cells["signal_year"].isin(years)].copy()
    summary_rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(["feature", "feature_bin"], sort=True):
        feature, feature_bin = keys
        row: dict[str, Any] = {
            "period": period_name,
            "feature": str(feature),
            "feature_bin": int(feature_bin),
            "dates": int(group["trade_date"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            estimate = atlas._hac_mean(
                group[metric].to_numpy(dtype=np.float64), lag=int(hac_lag)
            )
            for name, value in estimate.items():
                row[f"{metric}_{name}"] = value
        row["d20_excess_p_one_sided"] = _one_sided_positive_p(
            float(row["d20_excess_mean"]), float(row["d20_excess_se"])
        )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["d20_excess_bh_q"] = _benjamini_hochberg(
            summary["d20_excess_p_one_sided"].to_numpy(dtype=np.float64)
        )
    annual = (
        selected.groupby(["feature", "feature_bin", "signal_year"], sort=True)[
            list(SUMMARY_METRICS)
        ]
        .mean()
        .reset_index()
        .rename(columns={"signal_year": "evaluation_year"})
    )
    annual.insert(0, "period", period_name)
    return summary, annual


def _eligible_candidates(
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    positive_years = (
        annual.groupby(["feature", "feature_bin"], sort=True)["d20_excess"]
        .agg(
            positive_excess_years=lambda values: int((values > 0).sum()),
            evaluated_years="count",
        )
        .reset_index()
    )
    work = summary.merge(
        positive_years, on=["feature", "feature_bin"], how="left", validate="one_to_one"
    )
    mask = (
        work["selected_net_return_20_mean"].gt(0)
        & work["d20_excess_mean"].gt(0)
        & work["up10_before_down5_lift_mean"].gt(0)
        & work["positive_excess_years"].ge(
            int(contract.get("minimum_positive_excess_years", 4))
        )
        & work["signal_share_mean"].between(
            float(contract.get("minimum_mean_signal_share", 0.10)),
            float(contract.get("maximum_mean_signal_share", 0.30)),
            inclusive="both",
        )
    )
    maximum_q = contract.get("maximum_one_sided_bh_q")
    if maximum_q is not None:
        mask &= work["d20_excess_bh_q"].le(float(maximum_q))
    return work.loc[mask].sort_values(
        ["d20_excess_lcb_95", "d20_excess_mean"],
        ascending=False,
        kind="mergesort",
    )


def _rolling_rule_audit(
    date_cells: pd.DataFrame,
    study: Mapping[str, Any],
    *,
    candidate_contract: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    periods = dict(study.get("period", {}) or {})
    evaluation_years = [int(value) for value in periods["rolling_evaluation_years"]]
    analysis = dict(study.get("analysis", {}) or {})
    contract = dict(
        candidate_contract
        if candidate_contract is not None
        else (analysis.get("candidate_contract", {}) or {})
    )
    hac_lag = int(analysis.get("hac_lag", 20))
    selection_rows: list[dict[str, Any]] = []
    evaluation_rows: list[pd.DataFrame] = []
    for evaluation_year in evaluation_years:
        training_years = set(range(2012, int(evaluation_year)))
        train_summary, train_annual = _summarize_period(
            date_cells,
            period_name=f"train_through_{evaluation_year - 1}",
            years=training_years,
            hac_lag=hac_lag,
        )
        eligible = _eligible_candidates(train_summary, train_annual, contract)
        if eligible.empty:
            selection_rows.append(
                {
                    "evaluation_year": evaluation_year,
                    "status": "cash_no_rule_passed",
                    "eligible_rules": 0,
                }
            )
            continue
        chosen = eligible.iloc[0]
        evaluation = date_cells[
            date_cells["signal_year"].eq(evaluation_year)
            & date_cells["feature"].eq(chosen["feature"])
            & date_cells["feature_bin"].eq(int(chosen["feature_bin"]))
        ].copy()
        evaluation["selected_feature"] = str(chosen["feature"])
        evaluation["selected_feature_bin"] = int(chosen["feature_bin"])
        evaluation_rows.append(evaluation)
        evaluation_summary, _ = _summarize_period(
            evaluation,
            period_name=f"rolling_oos_{evaluation_year}",
            years={evaluation_year},
            hac_lag=hac_lag,
        )
        evaluated = evaluation_summary.iloc[0]
        selection_rows.append(
            {
                "evaluation_year": evaluation_year,
                "status": "rule_selected",
                "eligible_rules": int(len(eligible)),
                "selected_feature": str(chosen["feature"]),
                "selected_feature_bin": int(chosen["feature_bin"]),
                "train_d20_excess_mean": float(chosen["d20_excess_mean"]),
                "train_d20_excess_lcb_95": float(chosen["d20_excess_lcb_95"]),
                "train_d20_excess_bh_q": float(chosen["d20_excess_bh_q"]),
                "oos_selected_net_return_20": float(
                    evaluated["selected_net_return_20_mean"]
                ),
                "oos_d20_excess": float(evaluated["d20_excess_mean"]),
                "oos_up10_before_down5_lift": float(
                    evaluated["up10_before_down5_lift_mean"]
                ),
                "oos_signal_share": float(evaluated["signal_share_mean"]),
                "oos_fill_rate": float(evaluated["fill_rate_mean"]),
                "oos_complete20_rate": float(evaluated["complete20_rate_mean"]),
            }
        )
    selections = pd.DataFrame(selection_rows)
    if evaluation_rows:
        oos_dates = pd.concat(evaluation_rows, ignore_index=True)
        pooled_rows: dict[str, Any] = {
            "selected_years": int(oos_dates["signal_year"].nunique()),
            "dates": int(oos_dates["trade_date"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            estimate = atlas._hac_mean(
                oos_dates[metric].to_numpy(dtype=np.float64), lag=hac_lag
            )
            for name, value in estimate.items():
                pooled_rows[f"{metric}_{name}"] = value
        pooled = pd.DataFrame([pooled_rows])
    else:
        oos_dates = pd.DataFrame()
        pooled = pd.DataFrame()
    return selections, pooled, oos_dates


def analyze_marginal_rules(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("hot_path_rules_manifest_missing")
    manifest = atlas._read_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("hot_path_rules_manifest_schema_mismatch")
    analysis_root = output_root / "analysis"
    analysis_manifest_path = analysis_root / "manifest.json"
    if analysis_manifest_path.is_file() and not force:
        current = atlas._read_json(analysis_manifest_path)
        if (
            current.get("schema") == ANALYSIS_SCHEMA
            and current.get("status") == "completed"
            and current.get("experiment_fingerprint")
            == manifest.get("experiment_fingerprint")
        ):
            return current
    analysis_root.mkdir(parents=True, exist_ok=True)
    paths = [Path(str(record["path"])) for record in manifest["compact_panels"]]
    analysis_config = dict(study.get("analysis", {}) or {})
    quintile_count = int(analysis_config.get("quintile_count", 5))
    cost_bps = float(analysis_config.get("round_trip_cost_bps", 60.0))
    hac_lag = int(analysis_config.get("hac_lag", 20))
    connection = atlas._connect(output_root, study)
    frames: list[pd.DataFrame] = []
    try:
        for feature in [str(value) for value in study["feature_ranks"]]:
            frames.append(
                connection.execute(
                    _marginal_date_query(
                        paths,
                        feature,
                        quintile_count=quintile_count,
                        cost_bps=cost_bps,
                    )
                ).fetchdf()
            )
    finally:
        connection.close()
    date_cells = pd.concat(frames, ignore_index=True)
    date_cells.to_parquet(
        analysis_root / "marginal_date_cells.parquet",
        index=False,
        compression="zstd",
    )
    periods = dict(study.get("period", {}) or {})
    discovery_years = set(int(value) for value in periods["discovery_years"])
    fixed_years = set(int(value) for value in periods["fixed_evaluation_years"])
    discovery_summary, discovery_annual = _summarize_period(
        date_cells,
        period_name="discovery_2012_2018",
        years=discovery_years,
        hac_lag=hac_lag,
    )
    fixed_summary, fixed_annual = _summarize_period(
        date_cells,
        period_name="fixed_2019_2025",
        years=fixed_years,
        hac_lag=hac_lag,
    )
    marginal_summary = pd.concat(
        [discovery_summary, fixed_summary], ignore_index=True
    )
    marginal_annual = pd.concat([discovery_annual, fixed_annual], ignore_index=True)
    candidates = _eligible_candidates(
        discovery_summary,
        discovery_annual,
        dict(analysis_config.get("candidate_contract", {}) or {}),
    )
    fixed_columns = [
        "feature",
        "feature_bin",
        "selected_net_return_20_mean",
        "d20_excess_mean",
        "d20_excess_lcb_95",
        "d20_excess_bh_q",
        "up10_before_down5_lift_mean",
        "mae20_lift_mean",
    ]
    candidate_evaluation = candidates.merge(
        fixed_summary[fixed_columns],
        on=["feature", "feature_bin"],
        how="left",
        validate="one_to_one",
        suffixes=("_discovery", "_fixed"),
    )
    rolling_selections, rolling_pooled, rolling_dates = _rolling_rule_audit(
        date_cells, study
    )
    atlas._write_csv(analysis_root / "marginal_summary.csv", marginal_summary)
    atlas._write_csv(analysis_root / "marginal_annual.csv", marginal_annual)
    atlas._write_csv(
        analysis_root / "discovery_candidate_fixed_evaluation.csv",
        candidate_evaluation,
    )
    atlas._write_csv(
        analysis_root / "rolling_rule_selections.csv", rolling_selections
    )
    atlas._write_csv(analysis_root / "rolling_oos_pooled.csv", rolling_pooled)
    if not rolling_dates.empty:
        rolling_dates.to_parquet(
            analysis_root / "rolling_oos_date_cells.parquet",
            index=False,
            compression="zstd",
        )
    report_lines = [
        "# Complete-Panel Hot-Path Rule Audit",
        "",
        "Every coordinate is converted to a same-date cross-sectional rank before fixed quintile bins are formed. Discovery uses 2012-2018; fixed evaluation uses 2019-2025. The rolling audit selects one quintile rule using only earlier years.",
        "",
        "## Discovery candidates and fixed evaluation",
        "",
        candidate_evaluation.head(20).to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Chronological rolling selections",
        "",
        rolling_selections.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Pooled rolling cohort diagnostics",
        "",
        rolling_pooled.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Boundary",
        "",
        "These are overlapping equal-weight signal cohorts, not a continuous account. Return means condition on approximate next-open fill and complete legal D20 paths; fill and completion rates are reported because that conditioning is not known at signal close. A quintile is a broad state gate, not a final stock ranking, position-sizing rule, or exit policy.",
    ]
    report_path = analysis_root / "research_record.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    result = {
        "schema": ANALYSIS_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "experiment_fingerprint": manifest.get("experiment_fingerprint"),
        "panel_rows": int(manifest.get("rows", 0)),
        "date_cell_rows": int(len(date_cells)),
        "discovery_candidate_rows": int(len(candidate_evaluation)),
        "rolling_selected_years": int(
            rolling_selections.get("status", pd.Series(dtype=str))
            .eq("rule_selected")
            .sum()
        ),
        "outputs": {
            "marginal_summary": str(
                (analysis_root / "marginal_summary.csv").resolve()
            ),
            "candidate_evaluation": str(
                (
                    analysis_root / "discovery_candidate_fixed_evaluation.csv"
                ).resolve()
            ),
            "rolling_selections": str(
                (analysis_root / "rolling_rule_selections.csv").resolve()
            ),
            "rolling_oos_pooled": str(
                (analysis_root / "rolling_oos_pooled.csv").resolve()
            ),
            "research_record": str(report_path.resolve()),
        },
        "training_role": "nonparametric_chronological_rule_probe",
        "portfolio_selection_performed": False,
        "profit_claim_allowed": False,
    }
    atlas._write_json(analysis_manifest_path, result)
    return result


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force_prepare: bool = False,
    force_analysis: bool = False,
) -> dict[str, Any]:
    prepared = prepare_rule_panels(
        study_path=study_path, output_root=output_root, force=force_prepare
    )
    analysis = analyze_marginal_rules(
        study_path=study_path, output_root=output_root, force=force_analysis
    )
    return {"prepared": prepared, "analysis": analysis}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build complete-panel nonparametric hot-path state rules."
    )
    parser.add_argument(
        "command", choices=("prepare", "analyze", "run"), nargs="?", default="run"
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_rule_panels(
            study_path=args.study,
            output_root=args.output_root,
            force=bool(args.force_prepare),
        )
    elif args.command == "analyze":
        result = analyze_marginal_rules(
            study_path=args.study,
            output_root=args.output_root,
            force=bool(args.force_analysis),
        )
    else:
        result = run_study(
            study_path=args.study,
            output_root=args.output_root,
            force_prepare=bool(args.force_prepare),
            force_analysis=bool(args.force_analysis),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=atlas._json_default))
    return result


if __name__ == "__main__":
    main()
