"""Industry- and size-neutral audit of the rolling hot-path pair rule.

The upstream pair result mixes two economically different coordinates: a
relative activation state and the lowest same-date absolute-amount quintile.
This study keeps the upstream selection frozen, then estimates date-equal
coarsened exact-matching contrasts so the two coordinates are not credited to
each other.
"""

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
from daily_research.path_policy import seq100_hot_path_pairs as pairs
from daily_research.path_policy import seq100_hot_path_rules as rules


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_hot_path_neutral_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_hot_path_neutral_v1"
)
STUDY_ID = "seq100_hot_path_neutral_v1"
MANIFEST_SCHEMA = "seq100_hot_path_neutral_manifest/1"
ANALYSIS_SCHEMA = "seq100_hot_path_neutral_analysis/1"
BUILDER_VERSION = 2

MATCHED_METRICS = (
    "fill_rate",
    "complete20_probability",
    "net_return_5",
    "net_return_20",
    "up10_before_down5",
    "mfe20",
    "mae20",
)

VALID_COHORTS = {
    "selected_pair",
    "all_nonpair",
    "same_amount_other_attention",
    "same_amount_less_active",
    "same_amount_neighbor",
    "low_amount",
    "higher_amount",
}


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = atlas._resolve_path(path)
    study = atlas._read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("hot_path_neutral_study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if int(source.get("forbidden_year", -1)) != 2026:
        raise ValueError("hot_path_neutral_forbidden_year_contract_missing")
    if str(source.get("maximum_outcome_date")) != "2025-12-31":
        raise ValueError("hot_path_neutral_outcome_cutoff_mismatch")

    primary = dict(study.get("primary_rule", {}) or {})
    quintiles = int(primary.get("quintile_count", 0))
    activation_bin = int(primary.get("activation_bin", -1))
    quality_bin = int(primary.get("quality_bin", -1))
    if quintiles < 2 or not (0 <= activation_bin < quintiles):
        raise ValueError("hot_path_neutral_activation_bin_invalid")
    if not (0 <= quality_bin < quintiles):
        raise ValueError("hot_path_neutral_quality_bin_invalid")

    contrasts = list(study.get("contrasts", []) or [])
    names = [str(item.get("name", "")) for item in contrasts]
    if not names or len(names) != len(set(names)):
        raise ValueError("hot_path_neutral_contrast_names_invalid")
    cohorts = {
        str(value)
        for item in contrasts
        for value in (item.get("treatment"), item.get("control"))
    }
    unknown = sorted(cohorts.difference(VALID_COHORTS))
    if unknown:
        raise ValueError(f"hot_path_neutral_unknown_cohorts:{unknown}")

    neutral = dict(study.get("neutralization", {}) or {})
    specs = list(neutral.get("match_specs", []) or [])
    spec_names = [str(item.get("name", "")) for item in specs]
    if not spec_names or len(spec_names) != len(set(spec_names)):
        raise ValueError("hot_path_neutral_match_specs_invalid")
    if str(neutral.get("primary_match_spec", "")) not in spec_names:
        raise ValueError("hot_path_neutral_primary_match_spec_missing")
    for spec in specs:
        size_bins = spec.get("size_bins")
        if size_bins is not None and int(size_bins) not in {5, 10}:
            raise ValueError("hot_path_neutral_size_bins_invalid")
        turnover_bins = spec.get("turnover_bins")
        if turnover_bins is not None and int(turnover_bins) not in {5, 10}:
            raise ValueError("hot_path_neutral_turnover_bins_invalid")
    analysis = dict(study.get("analysis", {}) or {})
    if str(analysis.get("primary_contrast", "")) not in names:
        raise ValueError("hot_path_neutral_primary_contrast_missing")
    return study


def _records_by_year(records: Sequence[Mapping[str, Any]]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for record in records:
        year = int(record["year"])
        path = Path(str(record["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"hot_path_neutral_source_panel_missing:{path}")
        result[year] = path
    return result


def _valuation_paths(
    study: Mapping[str, Any],
) -> tuple[dict[int, Path], Path, dict[str, Any], Path]:
    source = dict(study.get("source", {}) or {})
    active_path = atlas._resolve_path(str(source["qdp_active_manifest"]))
    active = atlas._read_json(active_path)
    domain = str(source.get("valuation_domain", "valuation"))
    qdp_root = active_path.parents[1]
    paths, manifest_path, manifest = atlas._dataset_paths_and_manifest(
        qdp_root, active, domain
    )
    by_year: dict[int, Path] = {}
    for path in paths:
        year_tokens = [part for part in path.parts if part.startswith("year=")]
        if not year_tokens:
            continue
        by_year[int(year_tokens[-1].split("=", 1)[1])] = path
    return by_year, manifest_path, manifest, active_path


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[int, tuple[Path, Path, Path]], str]:
    source = dict(study.get("source", {}) or {})
    pair_manifest_path = atlas._resolve_path(str(source["pair_manifest"]))
    pair_manifest = atlas._read_json(pair_manifest_path)
    if pair_manifest.get("schema") != pairs.ANALYSIS_SCHEMA:
        raise ValueError("hot_path_neutral_pair_manifest_schema_mismatch")
    if str(pair_manifest.get("experiment_fingerprint")) != str(
        source.get("expected_pair_fingerprint", "")
    ):
        raise ValueError("hot_path_neutral_pair_fingerprint_mismatch")
    selection_path = Path(str(dict(pair_manifest.get("outputs", {}))["rolling_selections"]))
    if not selection_path.is_file():
        raise FileNotFoundError("hot_path_neutral_rolling_selections_missing")

    rule_manifest_path = atlas._resolve_path(str(source["rule_manifest"]))
    rule_manifest = atlas._read_json(rule_manifest_path)
    if rule_manifest.get("schema") != rules.MANIFEST_SCHEMA:
        raise ValueError("hot_path_neutral_rule_manifest_schema_mismatch")
    if str(rule_manifest.get("experiment_fingerprint")) != str(
        source.get("expected_rule_fingerprint", "")
    ):
        raise ValueError("hot_path_neutral_rule_fingerprint_mismatch")

    atlas_manifest_path = atlas._resolve_path(str(source["atlas_manifest"]))
    atlas_manifest = atlas._read_json(atlas_manifest_path)
    if atlas_manifest.get("schema") != atlas.MANIFEST_SCHEMA:
        raise ValueError("hot_path_neutral_atlas_manifest_schema_mismatch")
    if str(atlas_manifest.get("experiment_fingerprint")) != str(
        source.get("expected_atlas_fingerprint", "")
    ):
        raise ValueError("hot_path_neutral_atlas_fingerprint_mismatch")

    rule_by_year = _records_by_year(rule_manifest["compact_panels"])
    atlas_by_year = _records_by_year(atlas_manifest["panels"])
    valuation_by_year, valuation_manifest_path, valuation_manifest, active_path = (
        _valuation_paths(study)
    )
    years = [int(value) for value in dict(study["period"])["all_research_years"]]
    missing = sorted(
        set(years).difference(rule_by_year)
        | set(years).difference(atlas_by_year)
        | set(years).difference(valuation_by_year)
    )
    if missing:
        raise FileNotFoundError(f"hot_path_neutral_year_inputs_missing:{missing}")
    year_paths = {
        year: (rule_by_year[year], atlas_by_year[year], valuation_by_year[year])
        for year in years
    }
    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": atlas._sha256_file(study_path),
        "pair_manifest_sha256": atlas._sha256_file(pair_manifest_path),
        "rolling_selections_sha256": atlas._sha256_file(selection_path),
        "rule_manifest_sha256": atlas._sha256_file(rule_manifest_path),
        "atlas_manifest_sha256": atlas._sha256_file(atlas_manifest_path),
        "active_manifest_sha256": atlas._sha256_file(active_path),
        "valuation_manifest_sha256": atlas._sha256_file(valuation_manifest_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    contract = {
        "pair_manifest": atlas._file_record(pair_manifest_path),
        "rolling_selections": atlas._file_record(selection_path),
        "rule_manifest": atlas._file_record(rule_manifest_path),
        "atlas_manifest": atlas._file_record(atlas_manifest_path),
        "qdp_active_manifest": atlas._file_record(active_path),
        "valuation_manifest": atlas._file_record(valuation_manifest_path),
        "valuation_dataset_id": str(valuation_manifest.get("dataset_id", "")),
        "valuation_row_count": int(valuation_manifest.get("row_count", 0)),
        "fingerprint_payload": payload,
    }
    return contract, year_paths, fingerprint


def _verify_rolling_rule(
    study: Mapping[str, Any], selection_path: Path
) -> pd.DataFrame:
    selections = pd.read_csv(selection_path)
    rolling_years = [
        int(value) for value in dict(study["period"])["rolling_evaluation_years"]
    ]
    selected = selections[
        selections["evaluation_year"].isin(rolling_years)
        & selections["status"].eq("rule_selected")
    ].copy()
    if selected["evaluation_year"].astype(int).tolist() != rolling_years:
        raise ValueError("hot_path_neutral_rolling_years_incomplete")
    primary = dict(study["primary_rule"])
    expected_feature = (
        f"{primary['activation_feature']}|{primary['quality_feature']}"
    )
    expected = (
        selected["selected_feature"].eq(expected_feature)
        & selected["activation_bin"].astype(int).eq(int(primary["activation_bin"]))
        & selected["quality_bin"].astype(int).eq(int(primary["quality_bin"]))
    )
    if not bool(expected.all()):
        raise ValueError("hot_path_neutral_primary_rule_not_rolling_frozen")
    return selected


def _neutral_panel_query(
    rule_path: Path,
    atlas_path: Path,
    valuation_path: Path,
    *,
    quintile_count: int,
) -> str:
    maximum_bin = int(quintile_count) - 1
    return f"""
    WITH joined AS (
        SELECT
            r.*,
            coalesce(a.industry, 'UNKNOWN') AS industry,
            a.amount,
            v.total_mv,
            v.circ_mv,
            v.turnover_rate
        FROM read_parquet({atlas._sql_quote(rule_path)}) r
        INNER JOIN read_parquet({atlas._sql_quote(atlas_path)}) a
          USING(symbol, trade_date)
        LEFT JOIN read_parquet({atlas._sql_quote(valuation_path)}) v
          USING(symbol, trade_date)
    ), ranked AS (
        SELECT *,
            CASE WHEN circ_mv > 0 AND isfinite(circ_mv) THEN
                percent_rank() OVER (
                    PARTITION BY trade_date
                    ORDER BY CASE WHEN circ_mv > 0 AND isfinite(circ_mv)
                                  THEN circ_mv ELSE NULL END NULLS LAST
                )
            ELSE NULL END AS circ_mv_rank,
            CASE WHEN amount > 0 AND circ_mv > 0
                           AND isfinite(amount) AND isfinite(circ_mv) THEN
                percent_rank() OVER (
                    PARTITION BY trade_date
                    ORDER BY CASE WHEN amount > 0 AND circ_mv > 0
                                            AND isfinite(amount)
                                            AND isfinite(circ_mv)
                                  THEN amount / circ_mv ELSE NULL END NULLS LAST
                )
            ELSE NULL END AS turnover_proxy_rank
        FROM joined
    )
    SELECT
        symbol,
        trade_date,
        signal_year,
        industry,
        CAST(least({maximum_bin}, greatest(0,
             floor(attention_rank * {int(quintile_count)}))) AS INTEGER)
            AS attention_bin,
        CAST(least({maximum_bin}, greatest(0,
             floor(amount_cross_section_rank * {int(quintile_count)}))) AS INTEGER)
            AS amount_bin,
        circ_mv_rank,
        CAST(least(4, greatest(0, floor(circ_mv_rank * 5))) AS INTEGER)
            AS size_bin_5,
        CAST(least(9, greatest(0, floor(circ_mv_rank * 10))) AS INTEGER)
            AS size_bin_10,
        turnover_proxy_rank,
        CAST(least(4, greatest(0, floor(turnover_proxy_rank * 5))) AS INTEGER)
            AS turnover_bin_5,
        CAST(least(9, greatest(0, floor(turnover_proxy_rank * 10))) AS INTEGER)
            AS turnover_bin_10,
        amount,
        total_mv,
        circ_mv,
        turnover_rate,
        amount / NULLIF(circ_mv, 0) AS amount_to_circ_mv,
        entry_observed_legal,
        entry_buyable_approx,
        valid_close_days_5,
        valid_close_days_20,
        terminal_log_return_5,
        terminal_log_return_20,
        mfe_20,
        mae_20,
        up10_before_down5,
        up10_down5_same_day_ambiguous
    FROM ranked
    """


def _manifest_valid(manifest: Mapping[str, Any], fingerprint: str) -> bool:
    records = list(manifest.get("neutral_panels", []) or [])
    return (
        manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("status") == "prepared"
        and str(manifest.get("experiment_fingerprint")) == str(fingerprint)
        and len(records) == 14
        and all(Path(str(record.get("path", ""))).is_file() for record in records)
    )


def prepare_neutral_panels(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    contract, year_paths, fingerprint = _source_contract(study_path, study)
    selection_path = Path(str(contract["rolling_selections"]["path"]))
    selected = _verify_rolling_rule(study, selection_path)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not force:
        current = atlas._read_json(manifest_path)
        if _manifest_valid(current, fingerprint):
            return current
        if str(current.get("experiment_fingerprint", "")) not in {"", fingerprint}:
            raise ValueError("hot_path_neutral_existing_fingerprint_mismatch")

    output_root.mkdir(parents=True, exist_ok=True)
    panel_root = output_root / "neutral_panels"
    panel_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    quintiles = int(dict(study["primary_rule"])["quintile_count"])
    connection = atlas._connect(output_root, study)
    records: list[dict[str, Any]] = []
    try:
        for year in sorted(year_paths):
            rule_path, atlas_path, valuation_path = year_paths[year]
            output_path = panel_root / f"neutral_state_{year}.parquet"
            if force or not output_path.is_file():
                atlas._write_json(
                    progress_path,
                    {
                        "status": "building_neutral_panel",
                        "year": int(year),
                        "completed_years": [int(item["year"]) for item in records],
                        "experiment_fingerprint": fingerprint,
                    },
                )
                atlas._copy_query(
                    connection,
                    _neutral_panel_query(
                        rule_path,
                        atlas_path,
                        valuation_path,
                        quintile_count=quintiles,
                    ),
                    output_path,
                )
            audit = connection.execute(
                f"""
                SELECT
                    count(*) AS rows,
                    count(*) - count(DISTINCT symbol || '|' || trade_date)
                        AS duplicate_keys,
                    count(*) FILTER (WHERE signal_year = 2026) AS forbidden_rows,
                    count(*) FILTER (WHERE industry IS NULL) AS missing_industry,
                    count(*) FILTER (
                        WHERE circ_mv IS NULL OR circ_mv <= 0 OR NOT isfinite(circ_mv)
                    ) AS invalid_circ_mv,
                    min(trade_date) AS minimum_date,
                    max(trade_date) AS maximum_date
                FROM read_parquet({atlas._sql_quote(output_path)})
                """
            ).fetchdf().iloc[0].to_dict()
            source_rows = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({atlas._sql_quote(rule_path)})"
                ).fetchone()[0]
            )
            if int(audit["rows"]) != source_rows:
                raise ValueError(f"hot_path_neutral_row_count_mismatch:{year}")
            if int(audit["duplicate_keys"]) != 0:
                raise ValueError(f"hot_path_neutral_duplicate_keys:{year}")
            if int(audit["forbidden_rows"]) != 0:
                raise ValueError(f"hot_path_neutral_forbidden_rows:{year}")
            records.append(
                {
                    "year": int(year),
                    **atlas._file_record(output_path, include_hash=False),
                    "rows": int(audit["rows"]),
                    "duplicate_keys": int(audit["duplicate_keys"]),
                    "missing_industry": int(audit["missing_industry"]),
                    "invalid_circ_mv": int(audit["invalid_circ_mv"]),
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
        "rolling_rule_verification": {
            "years": selected["evaluation_year"].astype(int).tolist(),
            "selected_feature": str(selected.iloc[0]["selected_feature"]),
            "activation_bin": int(selected.iloc[0]["activation_bin"]),
            "quality_bin": int(selected.iloc[0]["quality_bin"]),
        },
        "neutral_panels": records,
        "rows": int(sum(item["rows"] for item in records)),
        "training_performed": False,
        "portfolio_selection_performed": False,
    }
    atlas._write_json(manifest_path, manifest)
    atlas._write_json(
        progress_path,
        {"status": "prepared", "manifest": str(manifest_path.resolve())},
    )
    return manifest


def _cohort_condition(name: str, study: Mapping[str, Any]) -> str:
    primary = dict(study["primary_rule"])
    activation_bin = int(primary["activation_bin"])
    quality_bin = int(primary["quality_bin"])
    selected = f"attention_bin = {activation_bin} AND amount_bin = {quality_bin}"
    conditions = {
        "selected_pair": f"({selected})",
        "all_nonpair": f"NOT ({selected})",
        "same_amount_other_attention": (
            f"amount_bin = {quality_bin} AND attention_bin <> {activation_bin}"
        ),
        "same_amount_less_active": (
            f"amount_bin = {quality_bin} AND attention_bin < {activation_bin}"
        ),
        "same_amount_neighbor": (
            f"amount_bin = {quality_bin} AND attention_bin = {activation_bin - 1}"
        ),
        "low_amount": f"amount_bin = {quality_bin}",
        "higher_amount": f"amount_bin <> {quality_bin}",
    }
    if name not in conditions:
        raise ValueError(f"hot_path_neutral_unknown_cohort:{name}")
    return conditions[name]


def _match_columns(spec: Mapping[str, Any]) -> list[str]:
    columns = ["trade_date", "signal_year"]
    if bool(spec.get("match_industry", False)):
        columns.append("industry")
    size_bins = spec.get("size_bins")
    if size_bins is not None:
        columns.append(f"size_bin_{int(size_bins)}")
    turnover_bins = spec.get("turnover_bins")
    if turnover_bins is not None:
        columns.append(f"turnover_bin_{int(turnover_bins)}")
    return columns


def _matched_date_query(
    paths: Sequence[Path],
    study: Mapping[str, Any],
    contrast: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> str:
    treatment = _cohort_condition(str(contrast["treatment"]), study)
    control = _cohort_condition(str(contrast["control"]), study)
    group_columns = _match_columns(spec)
    group_sql = ", ".join(group_columns)
    cost = float(dict(study["analysis"]).get("round_trip_cost_bps", 60)) / 10_000
    matching_filter = ""
    if spec.get("size_bins") is not None:
        matching_filter += " AND circ_mv > 0 AND isfinite(circ_mv)"
    if spec.get("turnover_bins") is not None:
        matching_filter += (
            " AND amount_to_circ_mv > 0 AND isfinite(amount_to_circ_mv)"
        )
    return f"""
    WITH base AS (
        SELECT *,
            CAST(({treatment}) AS BOOLEAN) AS is_treatment,
            CAST(({control}) AS BOOLEAN) AS is_control,
            CAST(entry_buyable_approx AS DOUBLE) AS fill_value,
            CAST(
                entry_buyable_approx
                AND valid_close_days_5 = 5
                AND terminal_log_return_5 IS NOT NULL
                AS DOUBLE
            ) AS complete5_value,
            CAST(
                entry_buyable_approx
                AND valid_close_days_20 = 20
                AND terminal_log_return_20 IS NOT NULL
                AS DOUBLE
            ) AS complete20_value,
            CASE WHEN entry_buyable_approx
                       AND valid_close_days_5 = 5
                       AND terminal_log_return_5 IS NOT NULL
                 THEN exp(terminal_log_return_5) - 1.0 - {cost:.12g}
                 ELSE NULL END AS net_return_5_value,
            CASE WHEN entry_buyable_approx
                       AND valid_close_days_20 = 20
                       AND terminal_log_return_20 IS NOT NULL
                 THEN exp(terminal_log_return_20) - 1.0 - {cost:.12g}
                 ELSE NULL END AS net_return_20_value,
            CASE WHEN entry_buyable_approx
                       AND valid_close_days_20 = 20
                       AND NOT up10_down5_same_day_ambiguous
                 THEN CAST(up10_before_down5 AS DOUBLE)
                 ELSE NULL END AS hit_value,
            CASE WHEN entry_buyable_approx AND valid_close_days_20 = 20
                 THEN mfe_20 ELSE NULL END AS mfe20_value,
            CASE WHEN entry_buyable_approx AND valid_close_days_20 = 20
                 THEN mae_20 ELSE NULL END AS mae20_value
        FROM {atlas._parquet_scan(paths)}
        WHERE (({treatment}) OR ({control})) {matching_filter}
    ), cells AS (
        SELECT
            {group_sql},
            count(*) FILTER (WHERE is_treatment) AS nt_signal,
            count(*) FILTER (WHERE is_control) AS nc_signal,
            avg(fill_value) FILTER (WHERE is_treatment) AS t_fill_rate,
            avg(fill_value) FILTER (WHERE is_control) AS c_fill_rate,
            avg(complete20_value) FILTER (WHERE is_treatment)
                AS t_complete20_probability,
            avg(complete20_value) FILTER (WHERE is_control)
                AS c_complete20_probability,
            count(net_return_5_value) FILTER (WHERE is_treatment) AS nt_d5,
            count(net_return_5_value) FILTER (WHERE is_control) AS nc_d5,
            count(net_return_20_value) FILTER (WHERE is_treatment) AS nt_outcome,
            count(net_return_20_value) FILTER (WHERE is_control) AS nc_outcome,
            avg(net_return_5_value) FILTER (WHERE is_treatment)
                AS t_net_return_5,
            avg(net_return_5_value) FILTER (WHERE is_control)
                AS c_net_return_5,
            avg(net_return_20_value) FILTER (WHERE is_treatment)
                AS t_net_return_20,
            avg(net_return_20_value) FILTER (WHERE is_control)
                AS c_net_return_20,
            count(hit_value) FILTER (WHERE is_treatment) AS nt_hit,
            count(hit_value) FILTER (WHERE is_control) AS nc_hit,
            avg(hit_value) FILTER (WHERE is_treatment) AS t_up10_before_down5,
            avg(hit_value) FILTER (WHERE is_control) AS c_up10_before_down5,
            avg(mfe20_value) FILTER (WHERE is_treatment) AS t_mfe20,
            avg(mfe20_value) FILTER (WHERE is_control) AS c_mfe20,
            avg(mae20_value) FILTER (WHERE is_treatment) AS t_mae20,
            avg(mae20_value) FILTER (WHERE is_control) AS c_mae20
        FROM base
        GROUP BY {group_sql}
    ), totals AS (
        SELECT
            trade_date,
            max(signal_year) AS signal_year,
            sum(nt_signal) AS total_treatment_signal_rows,
            sum(nt_d5) AS total_treatment_d5_rows,
            sum(nt_outcome) AS total_treatment_outcome_rows
        FROM cells
        GROUP BY trade_date
    )
    SELECT
        '{str(contrast['name'])}' AS contrast,
        '{str(spec['name'])}' AS match_spec,
        c.trade_date,
        max(c.signal_year) AS signal_year,
        max(t.total_treatment_signal_rows) AS total_treatment_signal_rows,
        sum(c.nt_signal) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ) AS matched_treatment_signal_rows,
        sum(c.nt_signal) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ) / NULLIF(max(t.total_treatment_signal_rows), 0)::DOUBLE
            AS signal_match_coverage,
        max(t.total_treatment_d5_rows) AS total_treatment_d5_rows,
        sum(c.nt_d5) FILTER (
            WHERE c.nt_d5 > 0 AND c.nc_d5 > 0
        ) AS matched_treatment_d5_rows,
        sum(c.nt_d5) FILTER (
            WHERE c.nt_d5 > 0 AND c.nc_d5 > 0
        ) / NULLIF(max(t.total_treatment_d5_rows), 0)::DOUBLE
            AS d5_match_coverage,
        max(t.total_treatment_outcome_rows) AS total_treatment_outcome_rows,
        sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) AS matched_treatment_outcome_rows,
        sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(max(t.total_treatment_outcome_rows), 0)::DOUBLE
            AS outcome_match_coverage,
        sum(c.nt_signal * c.t_fill_rate) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ) / NULLIF(sum(c.nt_signal) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ), 0) AS treatment_fill_rate,
        sum(c.nt_signal * c.c_fill_rate) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ) / NULLIF(sum(c.nt_signal) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ), 0) AS control_fill_rate,
        sum(c.nt_signal * (c.t_fill_rate - c.c_fill_rate)) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ) / NULLIF(sum(c.nt_signal) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ), 0) AS fill_rate_difference,
        sum(c.nt_signal * c.t_complete20_probability) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ) / NULLIF(sum(c.nt_signal) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ), 0) AS treatment_complete20_probability,
        sum(c.nt_signal * c.c_complete20_probability) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ) / NULLIF(sum(c.nt_signal) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ), 0) AS control_complete20_probability,
        sum(c.nt_signal * (
            c.t_complete20_probability - c.c_complete20_probability
        )) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ) / NULLIF(sum(c.nt_signal) FILTER (
            WHERE c.nt_signal > 0 AND c.nc_signal > 0
        ), 0) AS complete20_probability_difference,
        sum(c.nt_d5 * c.t_net_return_5) FILTER (
            WHERE c.nt_d5 > 0 AND c.nc_d5 > 0
        ) / NULLIF(sum(c.nt_d5) FILTER (
            WHERE c.nt_d5 > 0 AND c.nc_d5 > 0
        ), 0) AS treatment_net_return_5,
        sum(c.nt_d5 * c.c_net_return_5) FILTER (
            WHERE c.nt_d5 > 0 AND c.nc_d5 > 0
        ) / NULLIF(sum(c.nt_d5) FILTER (
            WHERE c.nt_d5 > 0 AND c.nc_d5 > 0
        ), 0) AS control_net_return_5,
        sum(c.nt_d5 * (c.t_net_return_5 - c.c_net_return_5)) FILTER (
            WHERE c.nt_d5 > 0 AND c.nc_d5 > 0
        ) / NULLIF(sum(c.nt_d5) FILTER (
            WHERE c.nt_d5 > 0 AND c.nc_d5 > 0
        ), 0) AS net_return_5_difference,
        sum(c.nt_outcome * c.t_net_return_20) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ), 0) AS treatment_net_return_20,
        sum(c.nt_outcome * c.c_net_return_20) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ), 0) AS control_net_return_20,
        sum(c.nt_outcome * (c.t_net_return_20 - c.c_net_return_20)) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ), 0) AS net_return_20_difference,
        sum(c.nt_hit * c.t_up10_before_down5) FILTER (
            WHERE c.nt_hit > 0 AND c.nc_hit > 0
        ) / NULLIF(sum(c.nt_hit) FILTER (
            WHERE c.nt_hit > 0 AND c.nc_hit > 0
        ), 0) AS treatment_up10_before_down5,
        sum(c.nt_hit * c.c_up10_before_down5) FILTER (
            WHERE c.nt_hit > 0 AND c.nc_hit > 0
        ) / NULLIF(sum(c.nt_hit) FILTER (
            WHERE c.nt_hit > 0 AND c.nc_hit > 0
        ), 0) AS control_up10_before_down5,
        sum(c.nt_hit * (
            c.t_up10_before_down5 - c.c_up10_before_down5
        )) FILTER (
            WHERE c.nt_hit > 0 AND c.nc_hit > 0
        ) / NULLIF(sum(c.nt_hit) FILTER (
            WHERE c.nt_hit > 0 AND c.nc_hit > 0
        ), 0) AS up10_before_down5_difference,
        sum(c.nt_outcome * c.t_mfe20) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ), 0) AS treatment_mfe20,
        sum(c.nt_outcome * c.c_mfe20) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ), 0) AS control_mfe20,
        sum(c.nt_outcome * (c.t_mfe20 - c.c_mfe20)) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ), 0) AS mfe20_difference,
        sum(c.nt_outcome * c.t_mae20) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ), 0) AS treatment_mae20,
        sum(c.nt_outcome * c.c_mae20) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ), 0) AS control_mae20,
        sum(c.nt_outcome * (c.t_mae20 - c.c_mae20)) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ) / NULLIF(sum(c.nt_outcome) FILTER (
            WHERE c.nt_outcome > 0 AND c.nc_outcome > 0
        ), 0) AS mae20_difference
    FROM cells c
    INNER JOIN totals t USING(trade_date)
    GROUP BY c.trade_date
    HAVING max(t.total_treatment_signal_rows) > 0
    ORDER BY c.trade_date
    """


def _period_masks(study: Mapping[str, Any]) -> dict[str, set[int]]:
    period = dict(study["period"])
    return {
        "discovery_2012_2018": set(int(value) for value in period["discovery_years"]),
        "rolling_2019_2025": set(
            int(value) for value in period["rolling_evaluation_years"]
        ),
        "all_2012_2025": set(int(value) for value in period["all_research_years"]),
    }


def _summarize_daily(
    daily: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hac_lag = int(dict(study["analysis"]).get("hac_lag", 20))
    summary_rows: list[dict[str, Any]] = []
    annual_rows: list[pd.DataFrame] = []
    value_columns = [
        "signal_match_coverage",
        "d5_match_coverage",
        "outcome_match_coverage",
        *[
            f"{prefix}_{metric}"
            for metric in MATCHED_METRICS
            for prefix in ("treatment", "control")
        ],
        *[f"{metric}_difference" for metric in MATCHED_METRICS],
    ]
    for period_name, years in _period_masks(study).items():
        selected = daily[daily["signal_year"].isin(years)].copy()
        for keys, group in selected.groupby(["contrast", "match_spec"], sort=True):
            contrast, match_spec = keys
            row: dict[str, Any] = {
                "period": period_name,
                "contrast": str(contrast),
                "match_spec": str(match_spec),
                "dates": int(group["trade_date"].nunique()),
                "d5_dates": int(
                    group.loc[
                        group["net_return_5_difference"].notna(), "trade_date"
                    ].nunique()
                ),
                "outcome_dates": int(
                    group.loc[
                        group["net_return_20_difference"].notna(), "trade_date"
                    ].nunique()
                ),
                "treatment_signal_rows": int(
                    group["total_treatment_signal_rows"].sum()
                ),
                "matched_treatment_signal_rows": int(
                    group["matched_treatment_signal_rows"].fillna(0).sum()
                ),
                "treatment_d5_rows": int(group["total_treatment_d5_rows"].sum()),
                "matched_treatment_d5_rows": int(
                    group["matched_treatment_d5_rows"].fillna(0).sum()
                ),
                "treatment_outcome_rows": int(
                    group["total_treatment_outcome_rows"].sum()
                ),
                "matched_treatment_outcome_rows": int(
                    group["matched_treatment_outcome_rows"].fillna(0).sum()
                ),
            }
            for column in value_columns:
                estimate = atlas._hac_mean(
                    group[column].to_numpy(dtype=np.float64), lag=hac_lag
                )
                for name, value in estimate.items():
                    row[f"{column}_{name}"] = value
            row["net_return_20_p_one_sided"] = rules._one_sided_positive_p(
                float(row["net_return_20_difference_mean"]),
                float(row["net_return_20_difference_se"]),
            )
            row["net_return_5_p_one_sided"] = rules._one_sided_positive_p(
                float(row["net_return_5_difference_mean"]),
                float(row["net_return_5_difference_se"]),
            )
            summary_rows.append(row)
        annual = (
            selected.groupby(
                ["contrast", "match_spec", "signal_year"], sort=True
            )[value_columns]
            .mean()
            .reset_index()
            .rename(columns={"signal_year": "evaluation_year"})
        )
        annual.insert(0, "period", period_name)
        annual_rows.append(annual)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["net_return_20_bh_q"] = np.nan
        summary["net_return_5_bh_q"] = np.nan
        for _, positions in summary.groupby(
            ["period", "match_spec"], sort=True
        ).groups.items():
            index = list(positions)
            summary.loc[index, "net_return_20_bh_q"] = rules._benjamini_hochberg(
                summary.loc[index, "net_return_20_p_one_sided"].to_numpy(float)
            )
            summary.loc[index, "net_return_5_bh_q"] = rules._benjamini_hochberg(
                summary.loc[index, "net_return_5_p_one_sided"].to_numpy(float)
            )
    annual = pd.concat(annual_rows, ignore_index=True) if annual_rows else pd.DataFrame()
    if not annual.empty:
        positive = (
            annual[annual["period"].eq("rolling_2019_2025")]
            .groupby(["contrast", "match_spec"], sort=True)[
                "net_return_20_difference"
            ]
            .agg(
                positive_rolling_years=lambda values: int((values > 0).sum()),
                rolling_years="count",
            )
            .reset_index()
        )
        positive_d5 = (
            annual[annual["period"].eq("rolling_2019_2025")]
            .groupby(["contrast", "match_spec"], sort=True)[
                "net_return_5_difference"
            ]
            .agg(positive_d5_rolling_years=lambda values: int((values > 0).sum()))
            .reset_index()
        )
        positive = positive.merge(
            positive_d5,
            on=["contrast", "match_spec"],
            how="inner",
            validate="one_to_one",
        )
        summary = summary.merge(
            positive,
            on=["contrast", "match_spec"],
            how="left",
            validate="many_to_one",
        )
    return summary, annual


def _exposure_query(paths: Sequence[Path], study: Mapping[str, Any]) -> str:
    primary = dict(study["primary_rule"])
    a_bin = int(primary["activation_bin"])
    q_bin = int(primary["quality_bin"])
    cost = float(dict(study["analysis"])["round_trip_cost_bps"]) / 10_000
    return f"""
    WITH base AS (
        SELECT *,
            CASE
                WHEN attention_bin = {a_bin} AND amount_bin = {q_bin}
                    THEN 'selected_pair'
                WHEN amount_bin = {q_bin} THEN 'low_amount_other'
                ELSE 'higher_amount'
            END AS cohort,
            CASE WHEN entry_buyable_approx
                       AND valid_close_days_20 = 20
                       AND terminal_log_return_20 IS NOT NULL
                 THEN exp(terminal_log_return_20) - 1.0 - {cost:.12g}
                 ELSE NULL END AS net_return_20
        FROM {atlas._parquet_scan(paths)}
    ), date_cells AS (
        SELECT
            trade_date,
            signal_year,
            cohort,
            count(*) AS rows,
            median(circ_mv) AS median_circ_mv,
            median(total_mv) AS median_total_mv,
            median(amount) AS median_amount,
            median(turnover_rate) AS median_turnover_rate,
            median(amount_to_circ_mv) AS median_amount_to_circ_mv,
            avg(CAST(size_bin_5 = 0 AS DOUBLE)) AS smallest_size_quintile_share,
            avg(net_return_20) AS net_return_20
        FROM base
        GROUP BY trade_date, signal_year, cohort
    )
    SELECT * FROM date_cells ORDER BY trade_date, cohort
    """


def _summarize_exposure(
    date_cells: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hac_lag = int(dict(study["analysis"])["hac_lag"])
    metrics = [
        "rows",
        "median_circ_mv",
        "median_total_mv",
        "median_amount",
        "median_turnover_rate",
        "median_amount_to_circ_mv",
        "smallest_size_quintile_share",
        "net_return_20",
    ]
    rows: list[dict[str, Any]] = []
    for period_name, years in _period_masks(study).items():
        selected = date_cells[date_cells["signal_year"].isin(years)]
        for cohort, group in selected.groupby("cohort", sort=True):
            row: dict[str, Any] = {
                "period": period_name,
                "cohort": str(cohort),
                "dates": int(group["trade_date"].nunique()),
            }
            for metric in metrics:
                estimate = atlas._hac_mean(group[metric].to_numpy(float), lag=hac_lag)
                for name, value in estimate.items():
                    row[f"{metric}_{name}"] = value
            rows.append(row)
    annual = (
        date_cells.groupby(["cohort", "signal_year"], sort=True)[metrics]
        .mean()
        .reset_index()
        .rename(columns={"signal_year": "evaluation_year"})
    )
    return pd.DataFrame(rows), annual


def _size_gradient_query(paths: Sequence[Path], study: Mapping[str, Any]) -> str:
    primary = dict(study["primary_rule"])
    a_bin = int(primary["activation_bin"])
    q_bin = int(primary["quality_bin"])
    cost = float(dict(study["analysis"])["round_trip_cost_bps"]) / 10_000
    return f"""
    SELECT
        trade_date,
        signal_year,
        size_bin_5,
        count(*) AS rows,
        avg(exp(terminal_log_return_20) - 1.0 - {cost:.12g}) FILTER (
            WHERE entry_buyable_approx
              AND valid_close_days_20 = 20
              AND terminal_log_return_20 IS NOT NULL
        ) AS net_return_20
    FROM {atlas._parquet_scan(paths)}
    WHERE attention_bin = {a_bin}
      AND amount_bin = {q_bin}
      AND size_bin_5 IS NOT NULL
    GROUP BY trade_date, signal_year, size_bin_5
    ORDER BY trade_date, size_bin_5
    """


def _summarize_size_gradient(
    date_cells: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hac_lag = int(dict(study["analysis"])["hac_lag"])
    rows: list[dict[str, Any]] = []
    for period_name, years in _period_masks(study).items():
        selected = date_cells[date_cells["signal_year"].isin(years)]
        for size_bin, group in selected.groupby("size_bin_5", sort=True):
            estimate = atlas._hac_mean(group["net_return_20"].to_numpy(float), lag=hac_lag)
            rows.append(
                {
                    "period": period_name,
                    "size_bin_5": int(size_bin),
                    "dates": int(group["trade_date"].nunique()),
                    "rows_mean": float(group["rows"].mean()),
                    **{f"net_return_20_{key}": value for key, value in estimate.items()},
                }
            )
    annual = (
        date_cells.groupby(["size_bin_5", "signal_year"], sort=True)[
            ["rows", "net_return_20"]
        ]
        .mean()
        .reset_index()
        .rename(columns={"signal_year": "evaluation_year"})
    )
    return pd.DataFrame(rows), annual


def _decision(summary: pd.DataFrame, study: Mapping[str, Any]) -> dict[str, Any]:
    analysis = dict(study["analysis"])
    neutral = dict(study["neutralization"])
    primary = summary[
        summary["period"].eq("rolling_2019_2025")
        & summary["contrast"].eq(str(analysis["primary_contrast"]))
        & summary["match_spec"].eq(str(neutral["primary_match_spec"]))
    ]
    if len(primary) != 1:
        raise ValueError("hot_path_neutral_primary_result_missing")
    row = primary.iloc[0]
    checks = {
        "positive_d20_hac_lower_bound": bool(
            float(row["net_return_20_difference_lcb_95"]) > 0
        ),
        "positive_year_count": bool(
            int(row["positive_rolling_years"])
            >= int(analysis["minimum_positive_rolling_years"])
        ),
        "outcome_match_coverage": bool(
            float(row["outcome_match_coverage_mean"])
            >= float(neutral["minimum_primary_outcome_match_coverage"])
        ),
    }
    return {
        "primary_period": "rolling_2019_2025",
        "primary_contrast": str(analysis["primary_contrast"]),
        "primary_match_spec": str(neutral["primary_match_spec"]),
        "checks": checks,
        "activation_gate_passed": bool(all(checks.values())),
        "short_horizon_d5_information_supported": bool(
            float(row["net_return_5_difference_lcb_95"]) > 0
            and int(row["positive_d5_rolling_years"])
            >= int(analysis["minimum_positive_rolling_years"])
            and float(row["d5_match_coverage_mean"])
            >= float(neutral["minimum_primary_outcome_match_coverage"])
        ),
        "estimate": {
            "d5_difference_mean": float(row["net_return_5_difference_mean"]),
            "d5_difference_lcb_95": float(
                row["net_return_5_difference_lcb_95"]
            ),
            "d5_difference_ucb_95": float(
                row["net_return_5_difference_ucb_95"]
            ),
            "d5_treatment_net_return": float(row["treatment_net_return_5_mean"]),
            "d5_break_even_round_trip_cost_bps": float(
                (
                    row["treatment_net_return_5_mean"]
                    + float(analysis["round_trip_cost_bps"]) / 10_000
                )
                * 10_000
            ),
            "positive_d5_rolling_years": int(
                row["positive_d5_rolling_years"]
            ),
            "d5_match_coverage": float(row["d5_match_coverage_mean"]),
            "d20_difference_mean": float(row["net_return_20_difference_mean"]),
            "d20_difference_lcb_95": float(
                row["net_return_20_difference_lcb_95"]
            ),
            "d20_difference_ucb_95": float(
                row["net_return_20_difference_ucb_95"]
            ),
            "positive_rolling_years": int(row["positive_rolling_years"]),
            "rolling_years": int(row["rolling_years"]),
            "outcome_match_coverage": float(row["outcome_match_coverage_mean"]),
        },
    }


def _format_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return frame.loc[:, list(columns)].to_markdown(index=False, floatfmt=".5f")


def _research_record(
    summary: pd.DataFrame,
    annual: pd.DataFrame,
    exposure: pd.DataFrame,
    size_gradient: pd.DataFrame,
    decision: Mapping[str, Any],
    study: Mapping[str, Any],
) -> str:
    analysis = dict(study["analysis"])
    neutral = dict(study["neutralization"])
    rolling = summary[summary["period"].eq("rolling_2019_2025")].copy()
    primary_spec = str(neutral["primary_match_spec"])
    primary_rows = rolling[rolling["match_spec"].eq(primary_spec)].copy()
    primary_rows = primary_rows.sort_values("contrast", kind="mergesort")
    exposure_rows = exposure[exposure["period"].eq("rolling_2019_2025")].copy()
    gradient_rows = size_gradient[
        size_gradient["period"].eq("rolling_2019_2025")
    ].copy()
    liquidity_rows = rolling[
        rolling["contrast"].eq("low_amount_increment")
        & rolling["match_spec"].str.contains("turnover", regex=False)
    ].copy()
    primary_annual = annual[
        annual["period"].eq("rolling_2019_2025")
        & annual["match_spec"].eq(primary_spec)
        & annual["contrast"].isin(
            [str(analysis["primary_contrast"]), "low_amount_increment"]
        )
    ].copy()
    lines = [
        "# Hot-Path Industry and Size Neutral Audit",
        "",
        "The upstream rolling audit selected the same rule in every 2019-2025 year: attention quintile 3 together with the lowest same-date absolute-amount quintile. This audit freezes that result and separates the pair's total effect from the low-amount coordinate and from the incremental activation coordinate.",
        "",
        "The primary estimator is a date-equal coarsened exact-matching ATT. Controls are matched on signal date, point-in-time industry, and same-date float-market-cap decile. A stratum contributes only when treatment and control outcomes both exist; coverage is reported rather than silently extrapolating outside common support.",
        "",
        "## Primary rolling-period matched contrasts",
        "",
        _format_table(
            primary_rows,
            [
                "contrast",
                "dates",
                "d5_dates",
                "outcome_dates",
                "outcome_match_coverage_mean",
                "d5_match_coverage_mean",
                "treatment_net_return_5_mean",
                "control_net_return_5_mean",
                "net_return_5_difference_mean",
                "net_return_5_difference_lcb_95",
                "treatment_net_return_20_mean",
                "control_net_return_20_mean",
                "net_return_20_difference_mean",
                "net_return_20_difference_lcb_95",
                "net_return_20_difference_ucb_95",
                "positive_rolling_years",
                "net_return_20_bh_q",
                "up10_before_down5_difference_mean",
                "mfe20_difference_mean",
                "mae20_difference_mean",
            ],
        ),
        "",
        "## Turnover-matched low-amount sensitivity",
        "",
        _format_table(
            liquidity_rows,
            [
                "match_spec",
                "outcome_match_coverage_mean",
                "net_return_5_difference_mean",
                "net_return_5_difference_lcb_95",
                "net_return_20_difference_mean",
                "net_return_20_difference_lcb_95",
                "positive_rolling_years",
            ],
        ),
        "",
        "## Annual primary and low-amount contrasts",
        "",
        _format_table(
            primary_annual,
            [
                "contrast",
                "evaluation_year",
                "outcome_match_coverage",
                "net_return_20_difference",
                "up10_before_down5_difference",
                "mfe20_difference",
                "mae20_difference",
            ],
        ),
        "",
        "## Cohort exposure",
        "",
        _format_table(
            exposure_rows,
            [
                "cohort",
                "rows_mean",
                "median_circ_mv_mean",
                "median_amount_mean",
                "median_turnover_rate_mean",
                "median_amount_to_circ_mv_mean",
                "smallest_size_quintile_share_mean",
                "net_return_20_mean",
            ],
        ),
        "",
        "## Selected-pair size gradient",
        "",
        _format_table(
            gradient_rows,
            [
                "size_bin_5",
                "dates",
                "rows_mean",
                "net_return_20_mean",
                "net_return_20_lcb_95",
                "net_return_20_ucb_95",
            ],
        ),
        "",
        "## Decision",
        "",
        f"Activation gate passed: `{str(bool(decision['activation_gate_passed'])).lower()}`.",
        "",
        f"Primary rolling D20 activation increment: {float(decision['estimate']['d20_difference_mean']):.4%}; HAC 95% interval [{float(decision['estimate']['d20_difference_lcb_95']):.4%}, {float(decision['estimate']['d20_difference_ucb_95']):.4%}]. Positive years: {int(decision['estimate']['positive_rolling_years'])}/{int(decision['estimate']['rolling_years'])}; matched outcome coverage: {float(decision['estimate']['outcome_match_coverage']):.1%}.",
        "",
        f"The independently sampled D5 activation increment is positive in {int(decision['estimate']['positive_d5_rolling_years'])}/{int(decision['estimate']['rolling_years'])} years: {float(decision['estimate']['d5_difference_mean']):.4%}, HAC 95% interval [{float(decision['estimate']['d5_difference_lcb_95']):.4%}, {float(decision['estimate']['d5_difference_ucb_95']):.4%}], with {float(decision['estimate']['d5_match_coverage']):.1%} matched D5 coverage. The matched selected cohort itself earns only {float(decision['estimate']['d5_treatment_net_return']):.4%} after the frozen 60 bp proxy; its implied break-even round-trip cost is about {float(decision['estimate']['d5_break_even_round_trip_cost_bps']):.1f} bp. This is conditional-information evidence with a narrow cost budget, not an executable profit result.",
        "",
        "## Boundary",
        "",
        "This audit identifies conditional return structure; it is not a continuous account, an executable exit policy, or a capacity claim. Exact matching removes observed industry and coarse size composition, not unobserved news, order-flow identity, or all liquidity risk. The 60 bp cost proxy is not a state-dependent market-impact model. A positive low-amount contrast therefore cannot be promoted to a strategy without a separate capacity-aware account test, and a failed activation gate means the current evidence cannot credit relative activity with the pair result.",
        "",
    ]
    return "\n".join(lines)


def analyze_neutral_audit(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    prepared = prepare_neutral_panels(
        study_path=study_path, output_root=output_root, force=False
    )
    paths = [Path(str(record["path"])) for record in prepared["neutral_panels"]]
    connection = atlas._connect(output_root, study)
    daily_frames: list[pd.DataFrame] = []
    try:
        for contrast in list(study["contrasts"]):
            for spec in list(dict(study["neutralization"])["match_specs"]):
                daily_frames.append(
                    connection.execute(
                        _matched_date_query(paths, study, contrast, spec)
                    ).fetchdf()
                )
        exposure_dates = connection.execute(
            _exposure_query(paths, study)
        ).fetchdf()
        size_dates = connection.execute(
            _size_gradient_query(paths, study)
        ).fetchdf()
    finally:
        connection.close()
    daily = pd.concat(daily_frames, ignore_index=True)
    if bool((daily["signal_year"] == 2026).any()):
        raise ValueError("hot_path_neutral_analysis_read_forbidden_year")
    summary, annual = _summarize_daily(daily, study)
    exposure, exposure_annual = _summarize_exposure(exposure_dates, study)
    size_gradient, size_gradient_annual = _summarize_size_gradient(size_dates, study)
    decision = _decision(summary, study)

    output_root.mkdir(parents=True, exist_ok=True)
    daily_path = output_root / "matched_date_contrasts.parquet"
    summary_path = output_root / "matched_contrast_summary.csv"
    annual_path = output_root / "matched_contrast_annual.csv"
    exposure_dates_path = output_root / "cohort_exposure_dates.parquet"
    exposure_path = output_root / "cohort_exposure_summary.csv"
    exposure_annual_path = output_root / "cohort_exposure_annual.csv"
    size_dates_path = output_root / "selected_size_gradient_dates.parquet"
    size_path = output_root / "selected_size_gradient_summary.csv"
    size_annual_path = output_root / "selected_size_gradient_annual.csv"
    decision_path = output_root / "decision.json"
    record_path = output_root / "research_record.md"
    daily.to_parquet(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    annual.to_csv(annual_path, index=False)
    exposure_dates.to_parquet(exposure_dates_path, index=False)
    exposure.to_csv(exposure_path, index=False)
    exposure_annual.to_csv(exposure_annual_path, index=False)
    size_dates.to_parquet(size_dates_path, index=False)
    size_gradient.to_csv(size_path, index=False)
    size_gradient_annual.to_csv(size_annual_path, index=False)
    atlas._write_json(decision_path, dict(decision))
    record_path.write_text(
        _research_record(
            summary, annual, exposure, size_gradient, decision, study
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": str(prepared["experiment_fingerprint"]),
        "study": atlas._file_record(study_path),
        "source_manifest": atlas._file_record(output_root / "manifest.json"),
        "contrasts": [str(item["name"]) for item in study["contrasts"]],
        "match_specs": [
            str(item["name"])
            for item in dict(study["neutralization"])["match_specs"]
        ],
        "date_contrast_rows": int(len(daily)),
        "decision": dict(decision),
        "outputs": {
            "matched_date_contrasts": str(daily_path.resolve()),
            "matched_contrast_summary": str(summary_path.resolve()),
            "matched_contrast_annual": str(annual_path.resolve()),
            "cohort_exposure_summary": str(exposure_path.resolve()),
            "selected_size_gradient_summary": str(size_path.resolve()),
            "decision": str(decision_path.resolve()),
            "research_record": str(record_path.resolve()),
        },
        "training_performed": False,
        "portfolio_selection_performed": False,
        "profit_claim_allowed": False,
    }
    atlas._write_json(output_root / "analysis_manifest.json", manifest)
    atlas._write_json(
        output_root / "progress.json",
        {
            "status": "completed",
            "analysis_manifest": str((output_root / "analysis_manifest.json").resolve()),
        },
    )
    return manifest


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force_prepare: bool = False,
) -> dict[str, Any]:
    prepare_neutral_panels(
        study_path=study_path,
        output_root=output_root,
        force=force_prepare,
    )
    return analyze_neutral_audit(study_path=study_path, output_root=output_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit industry- and size-neutral hot-path effects."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force-prepare", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_study(
        study_path=args.study,
        output_root=args.output_root,
        force_prepare=bool(args.force_prepare),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=atlas._json_default))
    return result


if __name__ == "__main__":
    main()
