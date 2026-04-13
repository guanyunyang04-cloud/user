from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"

DEFAULT_FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_policy_v5_family_formal_review_20260412_r1" / "runs" / "short_expert_policy_v5b"
)
DEFAULT_RECENT_RUN_DIR = OUTPUT_ROOT / "short_alpha_recent_model_protocol_20260412_r1__short_expert_policy_v5b"
DEFAULT_CONSTRAINED_MODEL_DIR = (
    OUTPUT_ROOT / "short_alpha_policy_v5_family_constrained_execution_review_20260412_r1" / "models" / "short_expert_policy_v5b"
)
DEFAULT_CURRENT_FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_short_horizon_expert_review_20260406_r2_fullbudget" / "runs" / "short_expert_monthly_v1"
)
DEFAULT_CURRENT_RECENT_RUN_DIR = OUTPUT_ROOT / "short_alpha_recent_model_protocol_20260412_r1__short_expert_monthly_v1"
DEFAULT_ROOT_TAG = "short_alpha_policy_v5b_bridge_sensitivity_audit_20260412_r1"
DEFAULT_VARIANTS = "raw_1d,k1_3d,k1_5d,k1_20d,k2_5d,k2_20d,cap6_k1_5d,cap4_g092_098_k1_5d"
ANCHOR_VARIANT = "k1_20d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit policy_v5b bridge sensitivity, monthly attribution, and already-failed external concentration repairs."
    )
    parser.add_argument("--formal-run-dir", default=str(DEFAULT_FORMAL_RUN_DIR))
    parser.add_argument("--recent-run-dir", default=str(DEFAULT_RECENT_RUN_DIR))
    parser.add_argument("--constrained-model-dir", default=str(DEFAULT_CONSTRAINED_MODEL_DIR))
    parser.add_argument("--current-formal-run-dir", default=str(DEFAULT_CURRENT_FORMAL_RUN_DIR))
    parser.add_argument("--current-recent-run-dir", default=str(DEFAULT_CURRENT_RECENT_RUN_DIR))
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_monthly_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["month"] = df["month"].astype(str)
    return df.sort_values("month").reset_index(drop=True)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _format_num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def _monthly_diag(run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    diag = metrics.get("primary_research_monthly_diagnostics")
    if isinstance(diag, dict) and diag:
        return dict(diag)
    fallback = run_dir / "primary_research_monthly_diagnostics.json"
    return _load_json(fallback) if fallback.exists() else {}


def _monthly_robust_score(monthly_diag: dict[str, Any]) -> float:
    positive_ratio = _safe_float(monthly_diag.get("positive_month_ratio"))
    median_monthly_return = _safe_float(monthly_diag.get("median_monthly_return"))
    mean_monthly_return = _safe_float(monthly_diag.get("mean_monthly_return"))
    worst_monthly_return = _safe_float(monthly_diag.get("worst_monthly_return"))
    top3_positive_share = _safe_float(monthly_diag.get("top3_positive_month_share"))
    longest_negative_streak = int(monthly_diag.get("longest_negative_streak", 0) or 0)
    downside_penalty = max(-worst_monthly_return, 0.0)
    concentration_penalty = max(top3_positive_share - 0.60, 0.0)
    streak_penalty = max(longest_negative_streak - 2, 0)
    return float(
        mean_monthly_return
        + median_monthly_return
        + 0.05 * (positive_ratio - 0.50)
        - 0.35 * downside_penalty
        - 0.05 * concentration_penalty
        - 0.01 * float(streak_penalty)
    )


def _backtest_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    backtest = metrics.get("primary_research_backtest")
    return dict(backtest) if isinstance(backtest, dict) else {}


def _run_snapshot(run_dir: Path, label: str) -> dict[str, Any]:
    monthly_diag = _monthly_diag(run_dir)
    backtest = _backtest_metrics(run_dir)
    execution_rows = pd.read_csv(run_dir / "execution_alignment_objective_rows.csv")
    execution_rows = execution_rows.sort_values(["annual_return", "excess_annual_return"], ascending=False)
    top_profiles = execution_rows.head(5)[
        ["profile_name", "annual_return", "excess_annual_return", "excess_sharpe", "avg_turnover"]
    ].to_dict("records")
    return {
        "label": label,
        "run_dir": str(run_dir),
        "monthly_diag": monthly_diag,
        "monthly_robust_score": _monthly_robust_score(monthly_diag),
        "backtest": backtest,
        "top_execution_profiles": top_profiles,
    }


def _variant_payload(constrained_model_dir: Path, variant: str) -> dict[str, Any]:
    payload = _load_json(constrained_model_dir / "variants" / variant / "variant_summary.json")
    row = payload.get("row")
    variant_payload = payload.get("variant_payload")
    return {
        "variant": variant,
        "row": row if isinstance(row, dict) else {},
        "variant_payload": variant_payload if isinstance(variant_payload, dict) else {},
    }


def _variant_monthly(constrained_model_dir: Path, variant: str) -> pd.DataFrame:
    return _load_monthly_summary(constrained_model_dir / "variants" / variant / "monthly_backtest_summary.csv")


def _variant_table(variant_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in variant_payloads:
        row = payload.get("row", {})
        rows.append(
            {
                "variant": payload["variant"],
                "profile_name": str(row.get("profile_name", "") or ""),
                "monthly_robust_score": _safe_float(row.get("monthly_robust_score")),
                "annual_return": _safe_float(row.get("annual_return")),
                "excess_annual_return": _safe_float(row.get("excess_annual_return")),
                "avg_turnover": _safe_float(row.get("avg_turnover")),
                "worst_monthly_return": _safe_float(row.get("worst_monthly_return")),
                "aligned_avg_positive_count": _safe_float(row.get("aligned_avg_positive_count")),
                "aligned_avg_top1_weight": _safe_float(row.get("aligned_avg_top1_weight")),
                "aligned_avg_top2_share": _safe_float(row.get("aligned_avg_top2_share")),
                "aligned_avg_hhi": _safe_float(row.get("aligned_avg_hhi")),
            }
        )
    return sorted(rows, key=lambda item: item["monthly_robust_score"], reverse=True)


def _compare_vs_anchor(constrained_model_dir: Path, variant: str, anchor_variant: str) -> dict[str, Any]:
    variant_df = _variant_monthly(constrained_model_dir, variant)
    anchor_df = _variant_monthly(constrained_model_dir, anchor_variant)
    merged = anchor_df[["month", "excess_return", "avg_turnover", "avg_holding_count"]].merge(
        variant_df[["month", "excess_return", "avg_turnover", "avg_holding_count"]],
        on="month",
        suffixes=("_anchor", "_variant"),
    )
    merged["delta_excess_return"] = merged["excess_return_variant"] - merged["excess_return_anchor"]
    merged["delta_turnover"] = merged["avg_turnover_variant"] - merged["avg_turnover_anchor"]
    merged["delta_holding_count"] = merged["avg_holding_count_variant"] - merged["avg_holding_count_anchor"]
    negative = merged.sort_values("delta_excess_return").head(3)
    positive = merged.sort_values("delta_excess_return", ascending=False).head(3)
    return {
        "variant": variant,
        "anchor_variant": anchor_variant,
        "month_count": int(len(merged)),
        "mean_delta_excess_return": _safe_float(merged["delta_excess_return"].mean()),
        "median_delta_excess_return": _safe_float(merged["delta_excess_return"].median()),
        "worst_delta_excess_return": _safe_float(merged["delta_excess_return"].min()),
        "best_delta_excess_return": _safe_float(merged["delta_excess_return"].max()),
        "months_variant_better": int((merged["delta_excess_return"] > 0).sum()),
        "months_variant_worse": int((merged["delta_excess_return"] < 0).sum()),
        "months_worse_by_3pct": int((merged["delta_excess_return"] <= -0.03).sum()),
        "mean_delta_turnover": _safe_float(merged["delta_turnover"].mean()),
        "mean_delta_holding_count": _safe_float(merged["delta_holding_count"].mean()),
        "worst_months": negative[
            ["month", "delta_excess_return", "excess_return_variant", "excess_return_anchor", "delta_turnover", "delta_holding_count"]
        ].to_dict("records"),
        "best_months": positive[
            ["month", "delta_excess_return", "excess_return_variant", "excess_return_anchor", "delta_turnover", "delta_holding_count"]
        ].to_dict("records"),
    }


def _overlap_recent_vs_anchor(recent_run_dir: Path, constrained_model_dir: Path, anchor_variant: str) -> dict[str, Any]:
    recent_df = _load_monthly_summary(recent_run_dir / "primary_research_monthly_summary.csv")
    anchor_df = _variant_monthly(constrained_model_dir, anchor_variant)
    merged = anchor_df[["month", "excess_return", "avg_turnover", "avg_holding_count"]].merge(
        recent_df[["month", "excess_return", "avg_turnover", "avg_holding_count"]],
        on="month",
        suffixes=("_anchor", "_recent"),
    )
    merged["delta_excess_return"] = merged["excess_return_recent"] - merged["excess_return_anchor"]
    merged["delta_turnover"] = merged["avg_turnover_recent"] - merged["avg_turnover_anchor"]
    merged["delta_holding_count"] = merged["avg_holding_count_recent"] - merged["avg_holding_count_anchor"]
    return {
        "anchor_variant": anchor_variant,
        "overlap_month_count": int(len(merged)),
        "mean_delta_excess_return": _safe_float(merged["delta_excess_return"].mean()),
        "median_delta_excess_return": _safe_float(merged["delta_excess_return"].median()),
        "worst_delta_excess_return": _safe_float(merged["delta_excess_return"].min()),
        "best_delta_excess_return": _safe_float(merged["delta_excess_return"].max()),
        "months_recent_better": int((merged["delta_excess_return"] > 0).sum()),
        "months_recent_worse": int((merged["delta_excess_return"] < 0).sum()),
        "worst_months": merged.sort_values("delta_excess_return").head(4)[
            ["month", "delta_excess_return", "excess_return_recent", "excess_return_anchor", "delta_turnover", "delta_holding_count"]
        ].to_dict("records"),
        "best_months": merged.sort_values("delta_excess_return", ascending=False).head(4)[
            ["month", "delta_excess_return", "excess_return_recent", "excess_return_anchor", "delta_turnover", "delta_holding_count"]
        ].to_dict("records"),
    }


def _no_op_repairs(variant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {row["variant"]: row for row in variant_rows}
    findings: list[dict[str, Any]] = []
    comparisons = [
        ("cap6_k1_5d", "k1_5d"),
        ("cap4_g092_098_k1_5d", "k1_5d"),
        ("cap6_k1_3d", "k1_3d"),
    ]
    for left, right in comparisons:
        left_row = by_variant.get(left)
        right_row = by_variant.get(right)
        if not left_row or not right_row:
            continue
        same_metrics = (
            abs(left_row["monthly_robust_score"] - right_row["monthly_robust_score"]) < 1e-9
            and abs(left_row["avg_turnover"] - right_row["avg_turnover"]) < 1e-9
            and abs(left_row["aligned_avg_positive_count"] - right_row["aligned_avg_positive_count"]) < 1e-9
        )
        if same_metrics:
            findings.append(
                {
                    "variant": left,
                    "baseline_variant": right,
                    "reason": "external cap/gross wrapper was a no-op against the existing learned panel",
                }
            )
    return findings


def _design_implications(
    *,
    recent_vs_anchor: dict[str, Any],
    compare_rows: list[dict[str, Any]],
    no_op_repairs: list[dict[str, Any]],
) -> list[str]:
    implications: list[str] = []
    k1_3d = next((row for row in compare_rows if row["variant"] == "k1_3d"), {})
    k1_5d = next((row for row in compare_rows if row["variant"] == "k1_5d"), {})
    if k1_3d and _safe_float(k1_3d.get("worst_delta_excess_return")) <= -0.08:
        implications.append(
            "Fast 3d bridge is the main tail-risk source under constrained replay; next branch should damp bridge-speed sensitivity before chasing more recent upside."
        )
    if k1_5d and _safe_float(k1_5d.get("months_worse_by_3pct")) >= 2:
        implications.append(
            "5d bridge still leaves multiple material down-gap months versus k1_20d, so 'just slow down a bit' is not enough without stronger stability regularization."
        )
    if recent_vs_anchor and _safe_float(recent_vs_anchor.get("mean_delta_turnover")) > 0.40:
        implications.append(
            "Recent winner relies on much higher turnover and fewer names than the deployable anchor; successor branches should stabilize holdings and candidate-count behavior inside the score head."
        )
    if no_op_repairs:
        implications.append(
            "External cap wrappers already failed as no-op repairs, so the next round should internalize concentration control instead of adding more capped bridge variants."
        )
    implications.append(
        "Keep k1_20d deployable quality as the anchor and try to pull fresh formal upward by balancing v5b's concentration objective with v5a-style stability pressure."
    )
    return implications


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    formal_run_dir = Path(args.formal_run_dir).resolve()
    recent_run_dir = Path(args.recent_run_dir).resolve()
    constrained_model_dir = Path(args.constrained_model_dir).resolve()
    current_formal_run_dir = Path(args.current_formal_run_dir).resolve()
    current_recent_run_dir = Path(args.current_recent_run_dir).resolve()
    variants = [part.strip() for part in str(args.variants).split(",") if part.strip()]

    formal_snapshot = _run_snapshot(formal_run_dir, "formal_selected")
    recent_snapshot = _run_snapshot(recent_run_dir, "recent_selected")
    current_formal_snapshot = _run_snapshot(current_formal_run_dir, "current_mainline_formal")
    current_recent_snapshot = _run_snapshot(current_recent_run_dir, "current_mainline_recent")

    variant_payloads = [_variant_payload(constrained_model_dir, variant) for variant in variants]
    variant_rows = _variant_table(variant_payloads)
    compare_rows = [
        _compare_vs_anchor(constrained_model_dir, variant, ANCHOR_VARIANT) for variant in variants if variant != ANCHOR_VARIANT
    ]
    recent_vs_anchor = _overlap_recent_vs_anchor(recent_run_dir, constrained_model_dir, ANCHOR_VARIANT)
    no_op_repairs = _no_op_repairs(variant_rows)
    design_implications = _design_implications(
        recent_vs_anchor=recent_vs_anchor,
        compare_rows=compare_rows,
        no_op_repairs=no_op_repairs,
    )

    summary = {
        "formal_snapshot": formal_snapshot,
        "recent_snapshot": recent_snapshot,
        "current_formal_snapshot": current_formal_snapshot,
        "current_recent_snapshot": current_recent_snapshot,
        "anchor_variant": ANCHOR_VARIANT,
        "variant_rows": variant_rows,
        "compare_vs_anchor": compare_rows,
        "recent_vs_anchor": recent_vs_anchor,
        "no_op_repairs": no_op_repairs,
        "design_implications": design_implications,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Policy V5b Bridge Sensitivity Audit",
        "",
        "## Direct Answer",
        f"- current fresh-formal selected profile: `{formal_snapshot['backtest'].get('execution_alignment_profile', formal_snapshot['monthly_diag'].get('profile_name', 'n/a'))}`",
        f"- current fresh-formal monthly robust: `{_format_num(formal_snapshot.get('monthly_robust_score'))}`",
        f"- constrained deployable anchor: `{ANCHOR_VARIANT}` with monthly robust `{_format_num(next((row['monthly_robust_score'] for row in variant_rows if row['variant'] == ANCHOR_VARIANT), 0.0))}`",
        f"- recent selected bridge is effectively much faster, with overlap mean delta vs `{ANCHOR_VARIANT}` = `{_format_num(recent_vs_anchor.get('mean_delta_excess_return'))}` and turnover delta `{_format_num(recent_vs_anchor.get('mean_delta_turnover'))}`",
        "",
        "## Variant Ladder",
    ]
    for row in variant_rows:
        lines.append(
            f"- `{row['variant']}` [{row['profile_name'] or 'raw'}]: robust `{_format_num(row['monthly_robust_score'])}`, "
            f"excess annual `{_format_pct(row['excess_annual_return'])}`, turnover `{_format_num(row['avg_turnover'])}`, "
            f"worst month `{_format_pct(row['worst_monthly_return'])}`, aligned names `{_format_num(row['aligned_avg_positive_count'])}`, "
            f"top1 `{_format_pct(row['aligned_avg_top1_weight'])}`, top2 `{_format_pct(row['aligned_avg_top2_share'])}`, HHI `{_format_num(row['aligned_avg_hhi'])}`"
        )
    lines.extend(["", f"## Bridge Comparisons vs `{ANCHOR_VARIANT}`"])
    for row in compare_rows:
        lines.append(
            f"- `{row['variant']}`: mean delta `{_format_pct(row['mean_delta_excess_return'])}`, median delta `{_format_pct(row['median_delta_excess_return'])}`, "
            f"worst gap `{_format_pct(row['worst_delta_excess_return'])}`, better months `{row['months_variant_better']}`, worse months `{row['months_variant_worse']}`, "
            f"`<= -3%` months `{row['months_worse_by_3pct']}`, turnover delta `{_format_num(row['mean_delta_turnover'])}`, holding-count delta `{_format_num(row['mean_delta_holding_count'])}`"
        )
        worst_months = ", ".join(
            f"{item['month']}({_format_pct(item['delta_excess_return'])})" for item in row.get("worst_months", [])
        )
        if worst_months:
            lines.append(f"  - worst months: {worst_months}")
    lines.extend(
        [
            "",
            "## Recent vs Deployable Anchor",
            f"- overlap months: `{recent_vs_anchor.get('overlap_month_count', 0)}`",
            f"- mean monthly excess delta: `{_format_pct(recent_vs_anchor.get('mean_delta_excess_return'))}`",
            f"- median monthly excess delta: `{_format_pct(recent_vs_anchor.get('median_delta_excess_return'))}`",
            f"- worst overlap month delta: `{_format_pct(recent_vs_anchor.get('worst_delta_excess_return'))}`",
            f"- recent better months: `{recent_vs_anchor.get('months_recent_better', 0)}`",
            f"- recent worse months: `{recent_vs_anchor.get('months_recent_worse', 0)}`",
            "",
            "## Already-Failed External Repairs",
        ]
    )
    if no_op_repairs:
        for row in no_op_repairs:
            lines.append(f"- `{row['variant']}` vs `{row['baseline_variant']}`: {row['reason']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Design Implications"])
    for note in design_implications:
        lines.append(f"- {note}")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
