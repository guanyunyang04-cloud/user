from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
DEFAULT_ROOT_TAG = "short_alpha_policy_v2_family_constrained_execution_review_20260411_r2"
DEFAULT_FAMILY_FORMAL_SUMMARY = (
    OUTPUT_ROOT / "short_alpha_policy_v2_family_formal_review_20260411_r1" / "summary.json"
)
DEFAULT_CURRENT_FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_short_horizon_expert_review_20260406_r2_fullbudget" / "runs" / "short_expert_monthly_v1"
)
DEFAULT_FAMILY_PROFILES = (
    "short_expert_policy_v2",
    "short_expert_policy_v2b",
    "short_expert_policy_v2c",
)
DEFAULT_EXISTING_RUN_DIRS = {
    "short_expert_policy_v2": OUTPUT_ROOT / "short_alpha_policy_v2_review_20260410_r1" / "runs" / "short_expert_policy_v2",
    "short_expert_policy_v2b": OUTPUT_ROOT / "short_alpha_policy_v2_family_formal_review_20260411_r1" / "runs" / "short_expert_policy_v2b",
    "short_expert_policy_v2c": OUTPUT_ROOT / "short_alpha_policy_v2_family_formal_review_20260411_r1" / "runs" / "short_expert_policy_v2c",
}


def _load_single_review_module():
    module_path = Path(__file__).with_name("policy_v2_constrained_execution_review.py")
    spec = importlib.util.spec_from_file_location("policy_v2_constrained_execution_review", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load constrained review helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


single_review = _load_single_review_module()
VariantSpec = single_review.VariantSpec

VARIANT_SPECS = (
    VariantSpec("raw_1d", "Original learned target weights with no execution bridge.", "raw_1d"),
    VariantSpec("k1_5d", "Core fast bridge: k1 5d ensemble.", "regoff_k1_5d_ensemble_native_anchor"),
    VariantSpec("k1_20d", "Core slow bridge: k1 20d ensemble.", "regoff_k1_20d_ensemble_native_anchor"),
    VariantSpec("k2_5d", "Current deployable comparator: k2 5d ensemble.", "regoff_k2_5d_ensemble_native_anchor"),
    VariantSpec("k2_20d", "Current slow deployable comparator: k2 20d ensemble.", "regoff_k2_20d_ensemble_native_anchor"),
    VariantSpec("cap6_k1_5d", "Cap learned candidates at 6 names before k1 5d bridge.", "regoff_k1_5d_ensemble_native_anchor", candidate_cap=6),
    VariantSpec(
        "cap4_g092_098_k1_5d",
        "Cap learned candidates at 4 names and clip gross to 0.92-0.98 before k1 5d bridge.",
        "regoff_k1_5d_ensemble_native_anchor",
        candidate_cap=4,
        gross_floor=0.92,
        gross_cap=0.98,
    ),
)

_MARKET_CACHE: dict[tuple[Any, ...], Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run constrained formal review for policy_v2 family under a narrow k1/k2 bridge matrix."
    )
    parser.add_argument("--family-formal-summary", default=str(DEFAULT_FAMILY_FORMAL_SUMMARY))
    parser.add_argument("--family-profiles", default=",".join(DEFAULT_FAMILY_PROFILES))
    parser.add_argument("--current-formal-run-dir", default=str(DEFAULT_CURRENT_FORMAL_RUN_DIR))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _model_key(profile_name: str) -> str:
    return str(profile_name).replace("short_expert_", "")


def _resolve_source_runs(summary_payload: dict[str, Any], family_profiles: list[str]) -> dict[str, Path]:
    source_runs = summary_payload.get("source_runs", {})
    resolved: dict[str, Path] = {}
    for profile_name in family_profiles:
        source_path = str(source_runs.get(profile_name, "") or "").strip()
        if source_path:
            candidate = Path(source_path).resolve()
            if (candidate / "metrics.json").exists():
                resolved[profile_name] = candidate
                continue
        fallback = DEFAULT_EXISTING_RUN_DIRS.get(profile_name)
        if fallback is not None and (fallback / "metrics.json").exists():
            resolved[profile_name] = fallback.resolve()
            continue
        raise FileNotFoundError(f"Unable to resolve run_dir for {profile_name}.")
    return resolved


def _load_market_bundle(*, stocks: list[str], start_date: str, end_date: str, benchmark: str):
    cache_key = (tuple(sorted(stocks)), start_date, end_date, benchmark)
    cached = _MARKET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    bundle = single_review._load_formal_market_data(
        stocks=stocks,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
    )
    _MARKET_CACHE[cache_key] = bundle
    return bundle


def _load_cached_variant(variant_root: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    summary_path = variant_root / "variant_summary.json"
    if not summary_path.exists():
        return None
    payload = single_review._load_json(summary_path)
    row = payload.get("row")
    variant_payload = payload.get("variant_payload")
    if not isinstance(row, dict) or not isinstance(variant_payload, dict):
        return None
    return row, variant_payload


def _select_variant_row(scoreboard: pd.DataFrame, selected_profile: str) -> dict[str, Any]:
    subset = scoreboard.loc[
        (scoreboard["profile_name"] == selected_profile)
        & (scoreboard["candidate_cap"] == 0)
        & (scoreboard["gross_floor"] == 0.0)
        & (scoreboard["gross_cap"] == 0.0)
    ]
    return subset.iloc[0].to_dict() if not subset.empty else {}


def _format_pct(value: Any) -> str:
    return single_review._format_pct(value)


def _format_num(value: Any) -> str:
    return single_review._format_num(value)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    family_profiles = [part.strip() for part in str(args.family_profiles).split(",") if part.strip()]
    if not family_profiles:
        raise ValueError("family_profiles is empty.")

    family_summary_path = Path(args.family_formal_summary).resolve()
    family_summary = single_review._load_json(family_summary_path)
    if not family_summary:
        raise FileNotFoundError(f"Missing family formal summary: {family_summary_path}")

    source_runs = _resolve_source_runs(family_summary, family_profiles)
    current_formal_run_dir = Path(args.current_formal_run_dir).resolve()
    current_reference = single_review._reference_row(current_formal_run_dir)

    model_summaries: dict[str, Any] = {}
    combined_rows: list[dict[str, Any]] = []

    for profile_name in family_profiles:
        run_dir = source_runs[profile_name]
        metrics = single_review._load_json(run_dir / "metrics.json")
        if not metrics:
            raise FileNotFoundError(f"Missing metrics.json under {run_dir}")

        raw_target_weights = single_review._load_long_panel(run_dir / "daily_target_weight_panel.csv", "target_weight")
        raw_score_frame = single_review._load_long_panel(run_dir / "daily_score_panel.csv", "score")
        common_columns = sorted(set(raw_target_weights.columns) | set(raw_score_frame.columns))
        raw_target_weights = raw_target_weights.reindex(columns=common_columns).fillna(0.0)
        raw_score_frame = raw_score_frame.reindex(index=raw_target_weights.index, columns=common_columns).fillna(0.0)

        benchmark = str(metrics.get("benchmark", "000300.SH") or "000300.SH")
        valid_start = str(metrics.get("valid_start", "") or "").replace("-", "")
        valid_end = str(metrics.get("valid_end", "") or "").replace("-", "")
        if not valid_start or not valid_end:
            raise ValueError(f"Run {run_dir} is missing valid_start/valid_end metadata.")

        close, open_df, benchmark_close, benchmark_open = _load_market_bundle(
            stocks=list(raw_target_weights.columns),
            start_date=valid_start,
            end_date=valid_end,
            benchmark=benchmark,
        )
        holding_count = int(metrics.get("score_head_extra", {}).get("holding_count", metrics.get("holding_count", 5)) or 5)
        max_weight = float(metrics.get("max_weight", 0.25) or 0.25)
        min_adv20 = float(metrics.get("min_adv20", 50000.0) or 50000.0)
        min_price = float(metrics.get("min_price", 2.0) or 2.0)
        max_price = float(metrics.get("max_price", 300.0) or 300.0)
        transaction_cost_bps = float(metrics.get("execution_alignment_transaction_cost_bps", 3.0) or 3.0)
        slippage_bps = float(metrics.get("execution_alignment_slippage_bps", 7.0) or 7.0)
        sell_tax_bps = float(metrics.get("execution_alignment_sell_tax_bps", 10.0) or 10.0)
        raw_stats = single_review._panel_stats(raw_target_weights)
        selected_formal_profile = str(metrics.get("execution_alignment_profile", "") or "")

        model_root = output_dir / "models" / profile_name
        model_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        variant_payloads: dict[str, dict[str, Any]] = {}
        model_key = _model_key(profile_name)

        for spec in VARIANT_SPECS:
            variant_root = model_root / "variants" / spec.name
            cached = None if args.force_rerun else _load_cached_variant(variant_root)
            if cached is not None:
                row, variant_payload = cached
                rows.append(row)
                variant_payloads[spec.name] = variant_payload
                continue

            prepared_target_weights = raw_target_weights.copy()
            if spec.candidate_cap > 0:
                prepared_target_weights = single_review._cap_candidate_count(prepared_target_weights, spec.candidate_cap)
            if spec.gross_floor > 0.0 or spec.gross_cap > 0.0:
                prepared_target_weights = single_review._clip_gross_band(prepared_target_weights, spec.gross_floor, spec.gross_cap)
            prepared_stats = single_review._panel_stats(prepared_target_weights)
            profile = single_review.exec_align.get_profile(spec.profile_name)
            variant_metrics, monthly_summary, monthly_diag, aligned_stats = single_review._evaluate_variant(
                raw_target_weights=prepared_target_weights.reindex(index=raw_score_frame.index).fillna(0.0),
                raw_score_frame=raw_score_frame,
                close=close,
                benchmark_close=benchmark_close,
                open_df=open_df,
                benchmark_open=benchmark_open,
                benchmark=benchmark,
                holding_count=holding_count,
                max_weight=max_weight,
                min_adv20=min_adv20,
                min_price=min_price,
                max_price=max_price,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
                sell_tax_bps=sell_tax_bps,
                profile=profile,
            )

            variant_root.mkdir(parents=True, exist_ok=True)
            monthly_summary.to_csv(variant_root / "monthly_backtest_summary.csv", index=False, encoding="utf-8-sig")
            _write_json(variant_root / "monthly_backtest_diagnostics.json", monthly_diag)
            row = {
                "model_profile_name": profile_name,
                "model_key": model_key,
                "run_dir": str(run_dir),
                "selected_formal_profile": selected_formal_profile,
                "variant_name": f"{model_key}__{spec.name}",
                "variant_short_name": spec.name,
                "description": spec.description,
                "profile_name": profile.name,
                "candidate_cap": int(spec.candidate_cap or 0),
                "gross_floor": float(spec.gross_floor or 0.0),
                "gross_cap": float(spec.gross_cap or 0.0),
                "annual_return": float(variant_metrics.get("annual_return", 0.0) or 0.0),
                "excess_annual_return": float(variant_metrics.get("excess_annual_return", 0.0) or 0.0),
                "excess_sharpe": float(variant_metrics.get("excess_sharpe", 0.0) or 0.0),
                "avg_turnover": float(variant_metrics.get("avg_turnover", 0.0) or 0.0),
                "positive_month_ratio": float(monthly_diag.get("positive_month_ratio", 0.0) or 0.0),
                "median_monthly_return": float(monthly_diag.get("median_monthly_return", 0.0) or 0.0),
                "mean_monthly_return": float(monthly_diag.get("mean_monthly_return", 0.0) or 0.0),
                "worst_monthly_return": float(monthly_diag.get("worst_monthly_return", 0.0) or 0.0),
                "top3_positive_month_share": float(monthly_diag.get("top3_positive_month_share", 0.0) or 0.0),
                "longest_negative_streak": int(monthly_diag.get("longest_negative_streak", 0) or 0),
                "monthly_robust_score": single_review._monthly_robust_score(monthly_diag),
                "prepared_avg_gross_exposure": float(prepared_stats["avg_gross_exposure"]),
                "prepared_avg_positive_count": float(prepared_stats["avg_positive_count"]),
                "aligned_avg_gross_exposure": float(aligned_stats["avg_gross_exposure"]),
                "aligned_avg_positive_count": float(aligned_stats["avg_positive_count"]),
                "aligned_avg_top1_weight": float(aligned_stats["avg_top1_weight"]),
                "aligned_avg_top2_share": float(aligned_stats["avg_top2_share"]),
                "aligned_avg_hhi": float(aligned_stats["avg_hhi"]),
            }
            variant_payload = {
                "metrics": variant_metrics,
                "monthly_diagnostics": monthly_diag,
                "prepared_panel_stats": prepared_stats,
                "aligned_panel_stats": aligned_stats,
                "raw_panel_stats": raw_stats,
            }
            _write_json(variant_root / "variant_summary.json", {"row": row, "variant_payload": variant_payload})
            rows.append(row)
            variant_payloads[spec.name] = variant_payload

        scoreboard = pd.DataFrame(rows).sort_values(
            ["monthly_robust_score", "positive_month_ratio", "median_monthly_return", "excess_annual_return", "excess_sharpe"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        scoreboard.to_csv(model_root / "variant_scoreboard.csv", index=False, encoding="utf-8-sig")

        best_row = scoreboard.iloc[0].to_dict()
        selected_row = _select_variant_row(scoreboard, selected_formal_profile)
        model_summary = {
            "profile_name": profile_name,
            "run_dir": str(run_dir),
            "selected_formal_profile": selected_formal_profile,
            "best_variant": best_row,
            "selected_formal_variant": selected_row,
            "k1_5d_variant": scoreboard.loc[scoreboard["variant_short_name"] == "k1_5d"].iloc[0].to_dict(),
            "k1_20d_variant": scoreboard.loc[scoreboard["variant_short_name"] == "k1_20d"].iloc[0].to_dict(),
            "k2_5d_variant": scoreboard.loc[scoreboard["variant_short_name"] == "k2_5d"].iloc[0].to_dict(),
            "k2_20d_variant": scoreboard.loc[scoreboard["variant_short_name"] == "k2_20d"].iloc[0].to_dict(),
            "scoreboard": scoreboard.to_dict("records"),
            "variant_payloads": variant_payloads,
        }
        model_summaries[profile_name] = model_summary
        combined_rows.extend(scoreboard.to_dict("records"))

    combined_scoreboard = pd.DataFrame(combined_rows).sort_values(
        ["monthly_robust_score", "positive_month_ratio", "median_monthly_return", "excess_annual_return", "excess_sharpe"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    combined_scoreboard.to_csv(output_dir / "variant_scoreboard.csv", index=False, encoding="utf-8-sig")

    family_best_variant = combined_scoreboard.iloc[0].to_dict() if not combined_scoreboard.empty else {}
    summary_payload = {
        "family_formal_summary_path": str(family_summary_path),
        "family_profiles": family_profiles,
        "current_formal_run_dir": str(current_formal_run_dir),
        "current_mainline_reference": current_reference,
        "family_best_model_profile_name": str(family_best_variant.get("model_profile_name", "")),
        "family_best_variant": family_best_variant,
        "model_summaries": model_summaries,
        "combined_scoreboard": combined_scoreboard.to_dict("records"),
    }
    _write_json(output_dir / "summary.json", summary_payload)

    lines = [
        "# Policy V2 Family Constrained Execution Review",
        "",
        "## Scope",
        f"- family formal summary: `{family_summary_path}`",
        f"- compared profiles: `{', '.join(family_profiles)}`",
        "- constrained matrix: `raw_1d / k1_5d / k1_20d / k2_5d / k2_20d / cap6_k1_5d / cap4_g092_098_k1_5d`",
        "",
        "## Direct Answer",
        f"- family best constrained variant: `{family_best_variant.get('variant_name', 'n/a')}`",
        f"- family best constrained robust: `{_format_num(family_best_variant.get('monthly_robust_score'))}`",
        f"- family best constrained profile: `{family_best_variant.get('profile_name', 'n/a')}`",
        f"- current mainline formal robust: `{_format_num(current_reference.get('monthly_robust_score'))}`",
        f"- delta vs current mainline: `{_format_num(float(family_best_variant.get('monthly_robust_score', 0.0) or 0.0) - float(current_reference.get('monthly_robust_score', 0.0) or 0.0))}`",
        "",
        "## Per-Model Readout",
    ]
    for profile_name in family_profiles:
        model_summary = model_summaries[profile_name]
        best_row = model_summary.get("best_variant", {})
        selected_row = model_summary.get("selected_formal_variant", {})
        lines.append(
            f"- `{profile_name}`: best `{best_row.get('variant_name', 'n/a')}` / profile `{best_row.get('profile_name', 'n/a')}` / "
            f"robust `{_format_num(best_row.get('monthly_robust_score'))}`; selected formal profile "
            f"`{model_summary.get('selected_formal_profile', 'n/a')}` / selected-variant robust "
            f"`{_format_num(selected_row.get('monthly_robust_score'))}` / delta "
            f"`{_format_num(float(best_row.get('monthly_robust_score', 0.0) or 0.0) - float(selected_row.get('monthly_robust_score', 0.0) or 0.0))}`"
        )
    lines.extend(["", "## Combined Top Rows"])
    for _, row in combined_scoreboard.head(10).iterrows():
        lines.append(
            f"- `{row['variant_name']}`: excess annual `{_format_pct(row['excess_annual_return'])}`, "
            f"excess Sharpe `{_format_num(row['excess_sharpe'])}`, positive-month `{_format_pct(row['positive_month_ratio'])}`, "
            f"median monthly `{_format_pct(row['median_monthly_return'])}`, worst month `{_format_pct(row['worst_monthly_return'])}`, "
            f"monthly robust `{_format_num(row['monthly_robust_score'])}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "- 只有在 `v2b / v2c` 的 constrained formal 证据补齐之后，recent frontier 才允许进入下一轮 promotion 讨论。",
            "- 如果最优 constrained 结果继续停在 `k1_20d / k2_20d` 一类慢桥，则 formal gap 更像 execution stability 问题；如果 `k1_5d` 明显更强，则 formal gap 更像 profile drift 问题。",
            "- 如果 capped `k1_5d` 同步改善，则下一轮 learned-control 应优先把 concentration / candidate control 内生化，而不是继续堆外部手工桥。",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
