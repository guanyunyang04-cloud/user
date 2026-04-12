from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.tools.current_default_followup_repair_verdict import (
    DEFAULT_ROOT_TAG as FOLLOWUP_ROOT_TAG,
    _blend_to_base_gross,
    _build_attack_defense_month_map,
    _label_series,
    _mean_gross_by_label,
    _month_state_series,
    _row_gross,
    _target_label_gross,
)
from daily_research.tools.current_default_signal_cash_repair_verdict import (
    DEFAULT_ACTIVE_MANIFEST,
    DEFAULT_STRONGEST_SUMMARY,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    TRADE_PLAN_SCRIPT,
    _apply_month_plan,
    _bridge_k2_5d,
    _bridge_summary,
    _build_month_features,
    _build_score_weight_raw_panel,
    _collect_run_metrics,
    _derive_thresholds,
    _generate_live_preview,
    _load_json,
    _load_long_panel,
    _load_regime_state,
    _num,
    _pct,
    _rank_scoreboard,
    _rowwise_power,
    _run_external_backtest,
    _sanitize_panel,
    _variant_kind_summary,
    _wide_to_long,
)
from daily_research.tools.recent_model_protocol import load_recent_protocol_bundle, resolve_repair_companion_entry


DEFAULT_ROOT_TAG = "short_alpha_current_default_gross_control_sweep_20260410_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Narrow gross-control sweep for the current short_expert + k2 default chain. "
            "First sweep a small family of state gross maps, then layer month-trigger and "
            "narrow score-to-weight overlays on the best gross-control candidate."
        )
    )
    parser.add_argument("--strongest-summary", default=str(DEFAULT_STRONGEST_SUMMARY))
    parser.add_argument("--active-manifest", default=str(DEFAULT_ACTIVE_MANIFEST))
    parser.add_argument("--followup-summary", default=str(OUTPUT_ROOT / FOLLOWUP_ROOT_TAG / "summary.json"))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--preview-cash", type=float, default=100000.0)
    return parser.parse_args()


def _build_state_map(up: float, flat: float, down: float) -> dict[str, float]:
    return {
        "trend_up_vol_low": min(max(float(up), 0.0), 1.0),
        "trend_flat_vol_low": min(max(float(flat), 0.0), 1.0),
        "trend_down_vol_low": min(max(float(down), 0.0), 1.0),
    }


def _best_gross_map_rows(scoreboard: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gross_rows = scoreboard.loc[
        scoreboard["variant_name"].astype(str).str.startswith("winner_gross_map_")
    ].copy()
    gross_rows = gross_rows.sort_values(
        ["monthly_robust_score", "median_monthly_return", "excess_annual_return"],
        ascending=[False, False, False],
    )
    for _, row in gross_rows.iterrows():
        rows.append(
            {
                "variant_name": str(row["variant_name"]),
                "monthly_robust_score": float(row["monthly_robust_score"]),
                "excess_annual_return": float(row["excess_annual_return"]),
                "excess_sharpe": float(row["excess_sharpe"]),
                "positive_month_ratio": float(row["positive_month_ratio"]),
                "median_monthly_return": float(row["median_monthly_return"]),
                "avg_turnover": float(row["avg_turnover"]),
            }
        )
    return rows


def _write_summary(
    output_dir: Path,
    *,
    scoreboard: pd.DataFrame,
    base_gross_scoreboard: pd.DataFrame,
    thresholds: dict[str, float],
    month_plan: pd.DataFrame,
    state_map_catalog: dict[str, Any],
    repair_winner: dict[str, Any],
    live_preview_payload: dict[str, Any],
    companion_source: str,
    companion_profile_name: str,
) -> None:
    winner = scoreboard.iloc[0].to_dict() if not scoreboard.empty else {}
    current_row = scoreboard.loc[scoreboard["variant_name"].eq("winner_current_target_static")].iloc[0].to_dict()
    companion_row = scoreboard.loc[scoreboard["variant_name"].eq("companion_current_target_static")].iloc[0].to_dict()
    gross_best = base_gross_scoreboard.iloc[0].to_dict() if not base_gross_scoreboard.empty else {}
    month_state_counts = month_plan["month_state"].value_counts().sort_index().to_dict() if not month_plan.empty else {}
    promotion_ready = bool(
        float(repair_winner.get("monthly_robust_score", 0.0) or 0.0)
        > float(companion_row.get("monthly_robust_score", 0.0) or 0.0)
    )
    summary_payload = {
        "overall_winner_variant": str(winner.get("variant_name", "")),
        "overall_winner_kind": _variant_kind_summary(str(winner.get("variant_name", ""))),
        "best_gross_variant": str(gross_best.get("variant_name", "")),
        "repair_winner_variant": str(repair_winner.get("variant_name", "")),
        "repair_winner_kind": _variant_kind_summary(str(repair_winner.get("variant_name", ""))),
        "promotion_ready": promotion_ready,
        "thresholds": thresholds,
        "month_state_counts": month_state_counts,
        "state_map_catalog": state_map_catalog,
        "live_preview": live_preview_payload,
        "best_gross_rows": _best_gross_map_rows(base_gross_scoreboard),
        "companion_source": str(companion_source),
        "companion_profile_name": str(companion_profile_name),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Current Default Gross-Control Sweep Verdict",
        "",
        "## Objective",
        "- 在 `market_state_guard_v2_balance` 的基础上继续做窄版 gross-control 微调，优先回答 current default 能否仅靠控制层继续追近或超过 companion。",
        "- gross-only winner 选出后，再叠加轻量 `month-trigger` 与 narrow `score-to-weight overlay`；所有变体都按 recent 一年同协议回放。",
        "",
        "## Month Plan",
        f"- aggressive_top1 threshold: `{_num(thresholds.get('aggressive_top1'))}`",
        f"- aggressive_top2 threshold: `{_num(thresholds.get('aggressive_top2'))}`",
        f"- defensive_top1 threshold: `{_num(thresholds.get('defensive_top1'))}`",
        f"- defensive_gross threshold: `{_pct(thresholds.get('defensive_gross'))}`",
        f"- month_state counts: `{month_state_counts}`",
        "",
        "## Overall Winner",
        f"- overall winner: `{winner.get('variant_name', 'n/a')}` ({_variant_kind_summary(str(winner.get('variant_name', '')))}).",
        f"- overall winner monthly_robust_score: `{_num(winner.get('monthly_robust_score'))}`",
        f"- overall winner excess annual: `{_pct(winner.get('excess_annual_return'))}`",
        f"- overall winner excess Sharpe: `{_num(winner.get('excess_sharpe'))}`",
        "",
        "## Gross-Control Best",
        f"- best gross-only winner-side variant: `{gross_best.get('variant_name', 'n/a')}`",
        f"- gross-only best monthly_robust_score: `{_num(gross_best.get('monthly_robust_score'))}`",
        f"- gross-only best excess annual: `{_pct(gross_best.get('excess_annual_return'))}`",
        f"- gross-only best excess Sharpe: `{_num(gross_best.get('excess_sharpe'))}`",
        "",
        "## Final Repair Winner",
        f"- repair winner: `{repair_winner.get('variant_name', 'n/a')}` ({_variant_kind_summary(str(repair_winner.get('variant_name', '')))}).",
        f"- repair winner monthly_robust_score: `{_num(repair_winner.get('monthly_robust_score'))}`",
        f"- repair winner excess annual: `{_pct(repair_winner.get('excess_annual_return'))}`",
        f"- repair winner excess Sharpe: `{_num(repair_winner.get('excess_sharpe'))}`",
        f"- repair winner positive month ratio: `{_pct(repair_winner.get('positive_month_ratio'))}`",
        f"- repair winner median monthly return: `{_pct(repair_winner.get('median_monthly_return'))}`",
        "",
        "## Baseline Compare",
        f"- current default monthly_robust_score: `{_num(current_row.get('monthly_robust_score'))}`",
        f"- companion baseline monthly_robust_score: `{_num(companion_row.get('monthly_robust_score'))}`",
        f"- repair winner vs current default robust delta: `{_num(float(repair_winner.get('monthly_robust_score', 0.0) or 0.0) - float(current_row.get('monthly_robust_score', 0.0) or 0.0))}`",
        f"- repair winner vs companion robust delta: `{_num(float(repair_winner.get('monthly_robust_score', 0.0) or 0.0) - float(companion_row.get('monthly_robust_score', 0.0) or 0.0))}`",
        "",
        "## Decision",
    ]
    if promotion_ready:
        lines.append("- current default 的 winner-side repair 已在 recent 一年口径下超过 companion，具备直接讨论默认 promotion 的资格。")
    else:
        lines.append("- current default 的 winner-side repair 仍未超过 companion，因此本轮不直接切默认。")
    if str(gross_best.get("variant_name", "")) == str(repair_winner.get("variant_name", "")):
        lines.append("- 这说明当前提升主要仍来自 gross-control 本身，month-trigger 和 narrow score overlay 没有进一步推翻 gross-only winner。")
    else:
        lines.append("- 这说明在 best gross-control 之上，light overlay 还能继续提供增益。")
    if live_preview_payload.get("plan_path"):
        lines.append(f"- 已输出 repair winner 的当前 live 预览计划：`{live_preview_payload.get('plan_path')}`。")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preview_live_plan(
    *,
    output_dir: Path,
    variant_name: str,
    panel: pd.DataFrame,
    active_manifest: dict[str, Any],
    python_executable: str,
    preview_cash: float,
) -> dict[str, Any]:
    preview_dir = output_dir / "live_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_target_csv = preview_dir / f"{variant_name}_target_weight_panel.csv"
    _wide_to_long(panel, "target_weight").to_csv(preview_target_csv, index=False, encoding="utf-8-sig")
    trade_plan_dir = preview_dir / "trade_plan"
    trade_plan_dir.mkdir(parents=True, exist_ok=True)
    score_panel_csv = str(
        active_manifest.get("trade_plan_score_panel_csv")
        or active_manifest.get("trade_plan_score_panel")
        or ""
    ).strip()
    preview_error = ""
    if score_panel_csv:
        cmd = [
            str(python_executable),
            str(TRADE_PLAN_SCRIPT),
            "--external-score-csv",
            score_panel_csv,
            "--external-target-weight-csv",
            str(preview_target_csv),
            "--candidate-label",
            variant_name,
            "--cash",
            str(float(preview_cash)),
            "--output-dir",
            str(trade_plan_dir),
        ]
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        except subprocess.CalledProcessError as exc:
            preview_error = str(exc)
    else:
        preview_error = "missing trade_plan_score_panel path in active manifest"
    return {
        "variant_name": variant_name,
        "preview_target_weight_panel_csv": str(preview_target_csv),
        "plan_path": str(trade_plan_dir / "latest_trade_plan.txt"),
        "preview_error": preview_error,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).expanduser() / str(args.root_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    recent_runs_dir = output_dir / "recent_runs"
    recent_runs_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = output_dir / "variant_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    strongest_summary = _load_json(Path(args.strongest_summary).expanduser())
    active_manifest = _load_json(Path(args.active_manifest).expanduser())
    followup_summary = _load_json(Path(args.followup_summary).expanduser())

    winner_recent = strongest_summary.get("winner_recent", {})
    companion_recent, companion_source = resolve_repair_companion_entry(
        strongest_summary,
        winner_entry=winner_recent if isinstance(winner_recent, dict) else {},
    )
    winner_recent_bundle = load_recent_protocol_bundle(winner_recent if isinstance(winner_recent, dict) else {})
    companion_recent_bundle = load_recent_protocol_bundle(companion_recent if isinstance(companion_recent, dict) else {})
    winner_recent_dir = Path(winner_recent_bundle["run_dir"])
    companion_recent_dir = Path(companion_recent_bundle["run_dir"])
    recent_start = str(strongest_summary.get("recent_start_date", "")).strip()
    recent_end = str(strongest_summary.get("recent_end_date", "")).strip()
    if not winner_recent_dir.exists() or not companion_recent_dir.exists():
        raise FileNotFoundError("Winner/companion corrected recent dirs are missing.")

    winner_score_panel = winner_recent_bundle["score_panel"]
    winner_target_panel = winner_recent_bundle["target_panel"]
    companion_score_panel = companion_recent_bundle["score_panel"]
    companion_target_panel = companion_recent_bundle["target_panel"]
    winner_regime_state = winner_recent_bundle["regime_state"]
    if winner_score_panel.empty or winner_target_panel.empty or companion_target_panel.empty:
        raise RuntimeError("Corrected recent panels are empty.")

    month_features = _build_month_features(winner_score_panel, winner_target_panel, winner_regime_state)
    thresholds = _derive_thresholds(month_features)
    month_plan = _apply_month_plan(month_features, thresholds)
    month_plan.to_csv(output_dir / "month_plan.csv", index=False, encoding="utf-8-sig")

    companion_market_state_map = _mean_gross_by_label(companion_target_panel, winner_regime_state, selector="market_state")
    companion_month_map = followup_summary.get("month_state_maps", {}).get("companion_month_state_gross", {})
    base_state_up = float(companion_market_state_map.get("trend_up_vol_low", 0.953))
    base_state_flat = float(companion_market_state_map.get("trend_flat_vol_low", 0.959))

    gross_map_catalog: dict[str, dict[str, float]] = {
        "winner_gross_map_d084": _build_state_map(base_state_up, base_state_flat, 0.84),
        "winner_gross_map_d086": _build_state_map(base_state_up, base_state_flat, 0.86),
        "winner_gross_map_d088": _build_state_map(base_state_up, base_state_flat, 0.88),
        "winner_gross_map_d090": _build_state_map(base_state_up, base_state_flat, 0.90),
        "winner_gross_map_d092": _build_state_map(base_state_up, base_state_flat, 0.92),
        "winner_gross_map_u096_f096_d088": _build_state_map(0.96, 0.96, 0.88),
        "winner_gross_map_u097_f098_d088": _build_state_map(0.97, 0.98, 0.88),
    }

    variant_panels: dict[str, pd.DataFrame] = {
        "winner_current_target_static": _sanitize_panel(winner_target_panel),
        "companion_current_target_static": _sanitize_panel(companion_target_panel),
    }
    market_state_labels = _label_series(winner_regime_state, winner_target_panel.index, "market_state")
    for variant_name, gross_map in gross_map_catalog.items():
        variant_panels[variant_name] = _target_label_gross(
            _sanitize_panel(winner_target_panel),
            market_state_labels,
            gross_map,
        )

    scoreboard_rows: list[dict[str, Any]] = []
    for variant_name, panel in variant_panels.items():
        panel_csv = panels_dir / f"{variant_name}.csv"
        _wide_to_long(panel, "target_weight").to_csv(panel_csv, index=False, encoding="utf-8-sig")
        score_panel_csv = Path(winner_recent_bundle["score_panel_csv"])
        score_panel = winner_score_panel
        if variant_name.startswith("companion_"):
            score_panel_csv = Path(companion_recent_bundle["score_panel_csv"])
            score_panel = companion_score_panel
        run_dir = _run_external_backtest(
            python_executable=str(args.python_executable),
            output_root=recent_runs_dir,
            variant_name=variant_name,
            score_panel_csv=score_panel_csv,
            target_weight_panel_csv=panel_csv,
            start_date=recent_start,
            end_date=recent_end,
            transaction_cost_bps=float(args.transaction_cost_bps),
            slippage_bps=float(args.slippage_bps),
            sell_tax_bps=float(args.sell_tax_bps),
        )
        scoreboard_rows.append(
            _collect_run_metrics(run_dir, variant_name, _bridge_summary(score_panel, panel))
        )

    first_pass = _rank_scoreboard(pd.DataFrame(scoreboard_rows))
    first_pass.to_csv(output_dir / "first_pass_scoreboard.csv", index=False, encoding="utf-8-sig")
    gross_only = first_pass.loc[first_pass["variant_name"].astype(str).str.startswith("winner_gross_map_")].copy()
    if gross_only.empty:
        raise RuntimeError("Gross-control sweep produced no winner-side gross variants.")
    best_gross_name = str(gross_only.iloc[0]["variant_name"])
    best_gross_panel = variant_panels[best_gross_name]

    best_gross_month_map = {
        label: float(value)
        for label, value in _row_gross(best_gross_panel)
        .groupby(_month_state_series(best_gross_panel.index, month_plan))
        .mean()
        .items()
        if str(label).strip()
    }
    mild_month_map = {
        "aggressive": min(1.0, float(best_gross_month_map.get("aggressive", 0.95)) + 0.02),
        "base": float(best_gross_month_map.get("base", 0.94)),
        "defensive": max(0.0, float(best_gross_month_map.get("defensive", 0.88)) - 0.05),
    }
    defensive_month_map = {
        "aggressive": min(1.0, float(best_gross_month_map.get("aggressive", 0.95)) + 0.01),
        "base": float(best_gross_month_map.get("base", 0.94)),
        "defensive": max(0.0, float(best_gross_month_map.get("defensive", 0.88)) - 0.08),
    }
    companion_attack_map = _build_attack_defense_month_map(companion_month_map)
    month_state_labels = _month_state_series(best_gross_panel.index, month_plan)
    score_weight_k2 = _bridge_k2_5d(_build_score_weight_raw_panel(winner_score_panel))

    overlay_panels: dict[str, pd.DataFrame] = {
        f"{best_gross_name}_month_overlay_mild": _target_label_gross(best_gross_panel, month_state_labels, mild_month_map),
        f"{best_gross_name}_month_overlay_defensive": _target_label_gross(best_gross_panel, month_state_labels, defensive_month_map),
        f"{best_gross_name}_month_overlay_companion_attack": _target_label_gross(best_gross_panel, month_state_labels, companion_attack_map),
        f"{best_gross_name}_scoreblend05": _blend_to_base_gross(best_gross_panel, score_weight_k2, alpha=0.05),
        f"{best_gross_name}_scoreblend10": _blend_to_base_gross(best_gross_panel, score_weight_k2, alpha=0.10),
        f"{best_gross_name}_power103": _rowwise_power(best_gross_panel, power=1.03, preserve_row_gross=True),
    }
    variant_panels.update(overlay_panels)

    for variant_name, panel in overlay_panels.items():
        panel_csv = panels_dir / f"{variant_name}.csv"
        _wide_to_long(panel, "target_weight").to_csv(panel_csv, index=False, encoding="utf-8-sig")
        run_dir = _run_external_backtest(
            python_executable=str(args.python_executable),
            output_root=recent_runs_dir,
            variant_name=variant_name,
            score_panel_csv=Path(winner_recent_bundle["score_panel_csv"]),
            target_weight_panel_csv=panel_csv,
            start_date=recent_start,
            end_date=recent_end,
            transaction_cost_bps=float(args.transaction_cost_bps),
            slippage_bps=float(args.slippage_bps),
            sell_tax_bps=float(args.sell_tax_bps),
        )
        scoreboard_rows.append(
            _collect_run_metrics(run_dir, variant_name, _bridge_summary(winner_score_panel, panel))
        )

    scoreboard = _rank_scoreboard(pd.DataFrame(scoreboard_rows))
    scoreboard.to_csv(output_dir / "recent_scoreboard.csv", index=False, encoding="utf-8-sig")

    repair_scoreboard = scoreboard.loc[scoreboard["variant_name"].astype(str).str.startswith("winner_")].copy()
    repair_winner = repair_scoreboard.iloc[0].to_dict() if not repair_scoreboard.empty else {}
    live_preview_payload = _generate_live_preview(
        output_dir=output_dir,
        variant_name=str(repair_winner.get("variant_name", "")),
        panel_builder=None,
        active_manifest=active_manifest,
        thresholds=thresholds,
        regime_state=winner_regime_state,
        python_executable=str(args.python_executable),
        preview_cash=float(args.preview_cash),
    )
    if str(repair_winner.get("variant_name", "")) in variant_panels:
        live_preview_payload = _preview_live_plan(
            output_dir=output_dir,
            variant_name=str(repair_winner["variant_name"]),
            panel=variant_panels[str(repair_winner["variant_name"])],
            active_manifest=active_manifest,
            python_executable=str(args.python_executable),
            preview_cash=float(args.preview_cash),
        )

    catalog = {
        "gross_maps": gross_map_catalog,
        "best_gross_month_map": best_gross_month_map,
        "mild_month_map": mild_month_map,
        "defensive_month_map": defensive_month_map,
        "companion_attack_map": companion_attack_map,
    }
    (output_dir / "gross_control_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(
        output_dir=output_dir,
        scoreboard=scoreboard,
        base_gross_scoreboard=gross_only,
        thresholds=thresholds,
        month_plan=month_plan,
        state_map_catalog=catalog,
        repair_winner=repair_winner,
        live_preview_payload=live_preview_payload,
        companion_source=companion_source,
        companion_profile_name=str(companion_recent.get("profile_name", "") or strongest_summary.get("recent_winner_profile_name", "")),
    )


if __name__ == "__main__":
    main()
