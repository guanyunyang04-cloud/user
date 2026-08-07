"""Two-coordinate active-state rule audit on the complete legal panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from daily_research.path_policy import seq100_hot_path_atlas as atlas
from daily_research.path_policy import seq100_hot_path_rules as rules


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_hot_path_pairs_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_hot_path_pairs_v1"
)
STUDY_ID = "seq100_hot_path_pairs_v1"
ANALYSIS_SCHEMA = "seq100_hot_path_pairs_analysis/1"
BUILDER_VERSION = 1


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = atlas._resolve_path(path)
    study = atlas._read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("hot_path_pairs_study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if int(source.get("forbidden_year", -1)) != 2026:
        raise ValueError("hot_path_pairs_forbidden_year_contract_missing")
    if str(source.get("maximum_outcome_date")) != "2025-12-31":
        raise ValueError("hot_path_pairs_outcome_cutoff_mismatch")
    search = dict(study.get("pair_search", {}) or {})
    activation = [str(value) for value in search.get("activation_features", [])]
    quality = [str(value) for value in search.get("quality_features", [])]
    unknown = sorted(set(activation + quality).difference(rules.FEATURE_EXPRESSIONS))
    if not activation or not quality or unknown:
        raise ValueError(f"hot_path_pairs_feature_contract_invalid:{unknown}")
    allowed = [int(value) for value in search.get("allowed_activation_bins", [])]
    quintiles = int(search.get("quintile_count", 5))
    if not allowed or min(allowed) < 0 or max(allowed) >= quintiles:
        raise ValueError("hot_path_pairs_activation_bins_invalid")
    return study


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], list[Path], str]:
    source = dict(study.get("source", {}) or {})
    manifest_path = atlas._resolve_path(str(source["rule_manifest"]))
    if not manifest_path.is_file():
        raise FileNotFoundError("hot_path_pairs_rule_manifest_missing")
    manifest = atlas._read_json(manifest_path)
    if manifest.get("schema") != rules.MANIFEST_SCHEMA:
        raise ValueError("hot_path_pairs_rule_manifest_schema_mismatch")
    expected = str(source.get("expected_rule_fingerprint", ""))
    if str(manifest.get("experiment_fingerprint")) != expected:
        raise ValueError("hot_path_pairs_rule_fingerprint_mismatch")
    paths = [Path(str(record["path"])) for record in manifest["compact_panels"]]
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError("hot_path_pairs_compact_panels_missing")
    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": atlas._sha256_file(study_path),
        "rule_manifest_sha256": atlas._sha256_file(manifest_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    contract = {
        "rule_manifest": atlas._file_record(manifest_path),
        "rule_experiment_fingerprint": expected,
        "compact_panels": [
            atlas._file_record(path, include_hash=False) for path in paths
        ],
        "panel_rows": int(manifest.get("rows", 0)),
        "fingerprint_payload": payload,
    }
    return contract, paths, fingerprint


def _encode_pair_bin(
    activation_bin: int, quality_bin: int, *, quintile_count: int
) -> int:
    return int(activation_bin) * int(quintile_count) + int(quality_bin)


def _decode_pair_bin(value: int, *, quintile_count: int) -> tuple[int, int]:
    return divmod(int(value), int(quintile_count))


def _pair_date_query(
    paths: Sequence[Path],
    activation_feature: str,
    quality_feature: str,
    *,
    allowed_activation_bins: Sequence[int],
    quintile_count: int,
    cost_bps: float,
) -> str:
    scan = atlas._parquet_scan(paths)
    activation_column = f"{activation_feature}_rank"
    quality_column = f"{quality_feature}_rank"
    allowed = ",".join(str(int(value)) for value in allowed_activation_bins)
    cost = float(cost_bps) / 10_000.0
    pair_name = f"{activation_feature}|{quality_feature}"
    return f"""
    WITH ranked AS (
        SELECT *,
            CAST(least({int(quintile_count) - 1}, greatest(0,
                floor({activation_column} * {int(quintile_count)}))) AS INTEGER)
                AS activation_bin,
            CAST(least({int(quintile_count) - 1}, greatest(0,
                floor({quality_column} * {int(quintile_count)}))) AS INTEGER)
                AS quality_bin
        FROM {scan}
        WHERE {activation_column} IS NOT NULL
          AND isfinite({activation_column})
          AND {quality_column} IS NOT NULL
          AND isfinite({quality_column})
    ), universe AS (
        SELECT
            trade_date,
            signal_year,
            count(*) AS universe_signal_rows,
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
        FROM ranked
        GROUP BY trade_date, signal_year
    ), cells AS (
        SELECT
            trade_date,
            signal_year,
            activation_bin,
            quality_bin,
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
        FROM ranked
        WHERE activation_bin IN ({allowed})
        GROUP BY trade_date, signal_year, activation_bin, quality_bin
    )
    SELECT
        '{pair_name}' AS feature,
        c.activation_bin * {int(quintile_count)} + c.quality_bin AS feature_bin,
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
    ORDER BY c.trade_date, c.activation_bin, c.quality_bin
    """


def _decode_columns(frame: pd.DataFrame, *, quintile_count: int) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    split = result["feature"].str.split("|", n=1, expand=True)
    result.insert(2, "activation_feature", split[0])
    result.insert(3, "quality_feature", split[1])
    bins = result["feature_bin"].map(
        lambda value: _decode_pair_bin(value, quintile_count=quintile_count)
    )
    result.insert(5, "activation_bin", bins.map(lambda value: value[0]))
    result.insert(6, "quality_bin", bins.map(lambda value: value[1]))
    return result


def analyze_pair_rules(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    contract, paths, fingerprint = _source_contract(study_path, study)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not force:
        current = atlas._read_json(manifest_path)
        if (
            current.get("schema") == ANALYSIS_SCHEMA
            and current.get("status") == "completed"
            and current.get("experiment_fingerprint") == fingerprint
        ):
            return current
    search = dict(study.get("pair_search", {}) or {})
    activation_features = [
        str(value) for value in search["activation_features"]
    ]
    quality_features = [str(value) for value in search["quality_features"]]
    allowed_bins = [int(value) for value in search["allowed_activation_bins"]]
    quintile_count = int(search.get("quintile_count", 5))
    analysis = dict(study.get("analysis", {}) or {})
    cost_bps = float(analysis.get("round_trip_cost_bps", 60.0))
    hac_lag = int(analysis.get("hac_lag", 20))
    connection = atlas._connect(output_root, study)
    frames: list[pd.DataFrame] = []
    try:
        for activation_feature in activation_features:
            for quality_feature in quality_features:
                frames.append(
                    connection.execute(
                        _pair_date_query(
                            paths,
                            activation_feature,
                            quality_feature,
                            allowed_activation_bins=allowed_bins,
                            quintile_count=quintile_count,
                            cost_bps=cost_bps,
                        )
                    ).fetchdf()
                )
    finally:
        connection.close()
    date_cells = pd.concat(frames, ignore_index=True)
    date_cells.to_parquet(
        output_root / "pair_date_cells.parquet",
        index=False,
        compression="zstd",
    )
    periods = dict(study.get("period", {}) or {})
    discovery_years = set(int(value) for value in periods["discovery_years"])
    fixed_years = set(int(value) for value in periods["fixed_evaluation_years"])
    discovery_summary, discovery_annual = rules._summarize_period(
        date_cells,
        period_name="discovery_2012_2018",
        years=discovery_years,
        hac_lag=hac_lag,
    )
    fixed_summary, fixed_annual = rules._summarize_period(
        date_cells,
        period_name="fixed_2019_2025",
        years=fixed_years,
        hac_lag=hac_lag,
    )
    candidate_contract = dict(analysis.get("candidate_contract", {}) or {})
    candidates = rules._eligible_candidates(
        discovery_summary, discovery_annual, candidate_contract
    )
    fixed_columns = [
        "feature",
        "feature_bin",
        "selected_net_return_20_mean",
        "d20_excess_mean",
        "d20_excess_lcb_95",
        "d20_excess_bh_q",
        "up10_before_down5_lift_mean",
        "mfe20_lift_mean",
        "mae20_lift_mean",
    ]
    candidate_evaluation = candidates.merge(
        fixed_summary[fixed_columns],
        on=["feature", "feature_bin"],
        how="left",
        validate="one_to_one",
        suffixes=("_discovery", "_fixed"),
    )
    rolling_selections, rolling_pooled, rolling_dates = rules._rolling_rule_audit(
        date_cells,
        study,
        candidate_contract=candidate_contract,
    )
    decoded_summary = _decode_columns(
        pd.concat([discovery_summary, fixed_summary], ignore_index=True),
        quintile_count=quintile_count,
    )
    decoded_annual = _decode_columns(
        pd.concat([discovery_annual, fixed_annual], ignore_index=True),
        quintile_count=quintile_count,
    )
    decoded_candidates = _decode_columns(
        candidate_evaluation, quintile_count=quintile_count
    )
    if not rolling_selections.empty and "selected_feature" in rolling_selections:
        split = rolling_selections["selected_feature"].str.split(
            "|", n=1, expand=True
        )
        rolling_selections["activation_feature"] = split[0]
        rolling_selections["quality_feature"] = split[1]
        decoded_bins = rolling_selections["selected_feature_bin"].map(
            lambda value: _decode_pair_bin(value, quintile_count=quintile_count)
            if pd.notna(value)
            else (None, None)
        )
        rolling_selections["activation_bin"] = decoded_bins.map(
            lambda value: value[0]
        )
        rolling_selections["quality_bin"] = decoded_bins.map(lambda value: value[1])
    atlas._write_csv(output_root / "pair_summary.csv", decoded_summary)
    atlas._write_csv(output_root / "pair_annual.csv", decoded_annual)
    atlas._write_csv(
        output_root / "discovery_candidate_fixed_evaluation.csv",
        decoded_candidates,
    )
    atlas._write_csv(
        output_root / "rolling_pair_selections.csv", rolling_selections
    )
    atlas._write_csv(output_root / "rolling_pair_oos_pooled.csv", rolling_pooled)
    if not rolling_dates.empty:
        rolling_dates.to_parquet(
            output_root / "rolling_pair_oos_dates.parquet",
            index=False,
            compression="zstd",
        )
    report_lines = [
        "# Active-State Pair Rule Audit",
        "",
        "Every pair contains one activation coordinate at or above its middle quintile and one independent quality, position, or crowding coordinate. All bins are same-date ranks. Selection uses only earlier years in the rolling audit.",
        "",
        "## Discovery candidates and fixed evaluation",
        "",
        decoded_candidates.head(20).to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Rolling selections",
        "",
        rolling_selections.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Pooled rolling cohort diagnostics",
        "",
        rolling_pooled.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Boundary",
        "",
        "This is a two-coordinate state gate over overlapping D20 signal cohorts, not a continuous account, ranking within the gate, exit rule, or capacity claim. Fixed 2019-2025 results are retrospective because that period has already informed the broader research conversation.",
    ]
    report_path = output_root / "research_record.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    result = {
        "schema": ANALYSIS_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": fingerprint,
        "study": atlas._file_record(study_path),
        "source_contract": contract,
        "pair_family_size": int(
            len(activation_features)
            * len(quality_features)
            * len(allowed_bins)
            * quintile_count
        ),
        "date_cell_rows": int(len(date_cells)),
        "discovery_candidate_rows": int(len(decoded_candidates)),
        "rolling_selected_years": int(
            rolling_selections.get("status", pd.Series(dtype=str))
            .eq("rule_selected")
            .sum()
        ),
        "outputs": {
            "pair_summary": str((output_root / "pair_summary.csv").resolve()),
            "candidate_evaluation": str(
                (
                    output_root / "discovery_candidate_fixed_evaluation.csv"
                ).resolve()
            ),
            "rolling_selections": str(
                (output_root / "rolling_pair_selections.csv").resolve()
            ),
            "rolling_oos_pooled": str(
                (output_root / "rolling_pair_oos_pooled.csv").resolve()
            ),
            "research_record": str(report_path.resolve()),
        },
        "training_role": "nonparametric_active_state_pair_probe",
        "portfolio_selection_performed": False,
        "profit_claim_allowed": False,
    }
    atlas._write_json(manifest_path, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit active-state feature pairs.")
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = analyze_pair_rules(
        study_path=args.study,
        output_root=args.output_root,
        force=bool(args.force),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=atlas._json_default))
    return result


if __name__ == "__main__":
    main()
