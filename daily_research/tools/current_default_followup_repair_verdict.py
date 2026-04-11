from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.baseline.regime import resolve_regime_label_series
from daily_research.tools.current_default_signal_cash_repair_verdict import (
    DEFAULT_ACTIVE_MANIFEST,
    DEFAULT_STRONGEST_SUMMARY,
    OUTPUT_ROOT,
    _apply_month_plan,
    _apply_soft_state,
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
    PROJECT_ROOT,
)
from daily_research.tools.recent_model_protocol import load_recent_protocol_bundle, resolve_repair_companion_entry


DEFAULT_ROOT_TAG = "short_alpha_current_default_followup_repair_20260410_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-round same-protocol recent-year repair verdict for the current short_expert + k2 "
            "default chain. This round expands cash sizing, month-trigger, control-layer transfer, "
            "and narrow score-to-weight experiments."
        )
    )
    parser.add_argument("--strongest-summary", default=str(DEFAULT_STRONGEST_SUMMARY))
    parser.add_argument("--active-manifest", default=str(DEFAULT_ACTIVE_MANIFEST))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default=DEFAULT_ROOT_TAG)
    parser.add_argument("--transaction-cost-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=7.0)
    parser.add_argument("--sell-tax-bps", type=float, default=10.0)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--preview-cash", type=float, default=100000.0)
    return parser.parse_args()


def _row_gross(panel: pd.DataFrame) -> pd.Series:
    if panel.empty:
        return pd.Series(dtype=float)
    return panel.sum(axis=1).astype(float)


def _rescale_row_to_gross(row: pd.Series, desired_gross: float) -> pd.Series:
    weights = row.fillna(0.0).astype(float).clip(lower=0.0)
    current_gross = float(weights.sum())
    desired = min(max(float(desired_gross), 0.0), 1.0)
    if current_gross <= 0.0 or desired <= 0.0:
        return pd.Series(0.0, index=weights.index, name=row.name, dtype=float)
    return (weights / current_gross * desired).rename(row.name)


def _target_row_gross(base_panel: pd.DataFrame, desired_gross: pd.Series) -> pd.DataFrame:
    if base_panel.empty:
        return base_panel.copy()
    aligned_gross = desired_gross.reindex(base_panel.index).fillna(_row_gross(base_panel))
    rows = [
        _rescale_row_to_gross(base_panel.loc[dt], float(aligned_gross.loc[dt]))
        for dt in base_panel.index
    ]
    return _sanitize_panel(pd.DataFrame(rows, index=base_panel.index, columns=base_panel.columns))


def _label_series(regime_state: pd.DataFrame, index: pd.Index, selector: str) -> pd.Series:
    labels = resolve_regime_label_series(regime_state.reindex(index), selector)
    return labels.astype("string").str.lower().reindex(index)


def _mean_gross_by_label(
    panel: pd.DataFrame,
    regime_state: pd.DataFrame,
    *,
    selector: str,
) -> dict[str, float]:
    if panel.empty:
        return {}
    gross = _row_gross(panel)
    labels = _label_series(regime_state, panel.index, selector)
    frame = pd.DataFrame({"gross": gross, "label": labels}).dropna(subset=["label"])
    if frame.empty:
        return {}
    return {
        str(label): float(value)
        for label, value in frame.groupby("label", dropna=False)["gross"].mean().items()
        if str(label).strip()
    }


def _month_state_series(index: pd.Index, month_plan: pd.DataFrame) -> pd.Series:
    plan_map = {
        str(row["month"]): str(row.get("month_state", "base"))
        for row in month_plan.to_dict("records")
    }
    return pd.Series(
        [plan_map.get(str(pd.Timestamp(dt).to_period("M")), "base") for dt in index],
        index=index,
        dtype="string",
    ).astype("string").str.lower()


def _mean_gross_by_month_state(panel: pd.DataFrame, month_plan: pd.DataFrame) -> dict[str, float]:
    if panel.empty:
        return {}
    gross = _row_gross(panel)
    labels = _month_state_series(panel.index, month_plan)
    frame = pd.DataFrame({"gross": gross, "label": labels}).dropna(subset=["label"])
    if frame.empty:
        return {}
    return {
        str(label): float(value)
        for label, value in frame.groupby("label", dropna=False)["gross"].mean().items()
        if str(label).strip()
    }


def _target_label_gross(
    base_panel: pd.DataFrame,
    label_series: pd.Series,
    gross_map: dict[str, float],
) -> pd.DataFrame:
    if base_panel.empty:
        return base_panel.copy()
    labels = label_series.reindex(base_panel.index).astype("string").str.lower()
    rows: list[pd.Series] = []
    base_gross = _row_gross(base_panel)
    for dt in base_panel.index:
        label_value = labels.loc[dt]
        label = "" if pd.isna(label_value) else str(label_value).strip().lower()
        desired = float(gross_map.get(label, float(base_gross.loc[dt])))
        rows.append(_rescale_row_to_gross(base_panel.loc[dt], desired))
    return _sanitize_panel(pd.DataFrame(rows, index=base_panel.index, columns=base_panel.columns))


def _blend_to_base_gross(
    base_panel: pd.DataFrame,
    alt_panel: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    if base_panel.empty:
        return base_panel.copy()
    alpha = min(max(float(alpha), 0.0), 1.0)
    columns = sorted(set(base_panel.columns).union(set(alt_panel.columns)))
    base = _sanitize_panel(base_panel.reindex(columns=columns).fillna(0.0))
    alt = _sanitize_panel(alt_panel.reindex(index=base.index, columns=columns).fillna(0.0))
    rows: list[pd.Series] = []
    for dt in base.index:
        base_row = base.loc[dt]
        alt_row = alt.loc[dt]
        base_gross = float(base_row.sum())
        if base_gross <= 0.0:
            rows.append(pd.Series(0.0, index=columns, name=dt, dtype=float))
            continue
        base_norm = base_row / base_gross
        alt_gross = float(alt_row.sum())
        alt_norm = alt_row / alt_gross if alt_gross > 0.0 else base_norm
        mixed = (1.0 - alpha) * base_norm + alpha * alt_norm
        rows.append((mixed / float(mixed.sum()) * base_gross).rename(dt))
    return _sanitize_panel(pd.DataFrame(rows, index=base.index, columns=columns))


def _build_balance_state_map(
    winner_map: dict[str, float],
    companion_map: dict[str, float],
) -> dict[str, float]:
    labels = sorted(set(winner_map).union(set(companion_map)))
    out: dict[str, float] = {}
    for label in labels:
        winner_value = float(winner_map.get(label, 0.92))
        companion_value = float(companion_map.get(label, winner_value))
        if label.startswith("trend_up"):
            target = max(companion_value, winner_value + 0.03)
        elif label.startswith("trend_flat"):
            target = max(companion_value, winner_value + 0.02)
        elif label.startswith("trend_down"):
            target = min(companion_value, winner_value - 0.08)
        else:
            target = companion_value
        out[label] = min(max(float(target), 0.0), 1.0)
    return out


def _build_attack_defense_month_map(companion_month_map: dict[str, float]) -> dict[str, float]:
    aggressive = min(1.0, float(companion_month_map.get("aggressive", 0.95)) + 0.03)
    base = min(1.0, float(companion_month_map.get("base", 0.94)))
    defensive = max(0.0, float(companion_month_map.get("defensive", 0.90)) - 0.08)
    return {
        "aggressive": aggressive,
        "base": base,
        "defensive": defensive,
    }


def _migration_rows(scoreboard: pd.DataFrame) -> list[dict[str, Any]]:
    lookup = {str(row["variant_name"]): row for row in scoreboard.to_dict("records")}
    rows: list[dict[str, Any]] = []
    pairs = [
        (
            "winner_current_target_static",
            "winner_current_target_companion_row_gross_transfer",
            "winner_with_companion_row_gross",
        ),
        (
            "winner_current_target_static",
            "winner_current_target_market_state_guard_v2_companion_transfer",
            "winner_with_companion_state_gross",
        ),
        (
            "companion_current_target_static",
            "companion_current_target_winner_row_gross_transfer",
            "companion_with_winner_row_gross",
        ),
        (
            "companion_current_target_static",
            "companion_current_target_winner_state_gross_transfer",
            "companion_with_winner_state_gross",
        ),
    ]
    for baseline_name, challenger_name, label in pairs:
        if baseline_name not in lookup or challenger_name not in lookup:
            continue
        base = lookup[baseline_name]
        challenger = lookup[challenger_name]
        rows.append(
            {
                "migration_case": label,
                "baseline_variant": baseline_name,
                "challenger_variant": challenger_name,
                "baseline_monthly_robust_score": float(base.get("monthly_robust_score", 0.0) or 0.0),
                "challenger_monthly_robust_score": float(challenger.get("monthly_robust_score", 0.0) or 0.0),
                "robust_delta": float(challenger.get("monthly_robust_score", 0.0) or 0.0)
                - float(base.get("monthly_robust_score", 0.0) or 0.0),
                "baseline_excess_annual_return": float(base.get("excess_annual_return", 0.0) or 0.0),
                "challenger_excess_annual_return": float(challenger.get("excess_annual_return", 0.0) or 0.0),
                "excess_annual_delta": float(challenger.get("excess_annual_return", 0.0) or 0.0)
                - float(base.get("excess_annual_return", 0.0) or 0.0),
            }
        )
    return rows


def _write_summary(
    output_dir: Path,
    *,
    scoreboard: pd.DataFrame,
    thresholds: dict[str, float],
    month_plan: pd.DataFrame,
    state_maps: dict[str, dict[str, float]],
    month_state_maps: dict[str, dict[str, float]],
    repair_winner: dict[str, Any],
    migration_rows: list[dict[str, Any]],
    live_preview_payload: dict[str, Any],
    companion_source: str,
    companion_profile_name: str,
) -> None:
    winner = scoreboard.iloc[0].to_dict() if not scoreboard.empty else {}
    current_row = scoreboard.loc[scoreboard["variant_name"].eq("winner_current_target_static")].iloc[0].to_dict()
    companion_row = scoreboard.loc[scoreboard["variant_name"].eq("companion_current_target_static")].iloc[0].to_dict()
    month_state_counts = month_plan["month_state"].value_counts().sort_index().to_dict() if not month_plan.empty else {}
    migration_lookup = {str(row["migration_case"]): row for row in migration_rows}
    summary_payload = {
        "overall_winner_variant": str(winner.get("variant_name", "")),
        "overall_winner_kind": _variant_kind_summary(str(winner.get("variant_name", ""))),
        "repair_winner_variant": str(repair_winner.get("variant_name", "")),
        "repair_winner_kind": _variant_kind_summary(str(repair_winner.get("variant_name", ""))),
        "thresholds": thresholds,
        "month_state_counts": month_state_counts,
        "state_maps": state_maps,
        "month_state_maps": month_state_maps,
        "migration_rows": migration_rows,
        "live_preview": live_preview_payload,
        "companion_source": str(companion_source),
        "companion_profile_name": str(companion_profile_name),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Current Default Follow-Up Repair Verdict",
        "",
        "## Objective",
        "- 围绕当前 `short_expert + k2` 默认链，优先做 `cash sizing / month-trigger`，再做窄版 `signal-to-weight`。",
        "- 第二轮实验同时补齐 `market_state_guard_v2`、控制层迁移、以及低波动 score blend；所有变体都按 recent 一年同协议回放。",
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
        "## Winner-Side Repair Winner",
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
        "## Control-Layer Transfer",
    ]
    winner_row_transfer = migration_lookup.get("winner_with_companion_row_gross", {})
    winner_state_transfer = migration_lookup.get("winner_with_companion_state_gross", {})
    companion_row_transfer = migration_lookup.get("companion_with_winner_row_gross", {})
    companion_state_transfer = migration_lookup.get("companion_with_winner_state_gross", {})
    if winner_row_transfer:
        lines.append(
            f"- winner 换成 companion 的逐日 gross 后，monthly_robust delta: `{_num(winner_row_transfer.get('robust_delta'))}` | excess annual delta: `{_pct(winner_row_transfer.get('excess_annual_delta'))}`"
        )
    if winner_state_transfer:
        lines.append(
            f"- winner 换成 companion 的状态 gross map 后，monthly_robust delta: `{_num(winner_state_transfer.get('robust_delta'))}` | excess annual delta: `{_pct(winner_state_transfer.get('excess_annual_delta'))}`"
        )
    if companion_row_transfer:
        lines.append(
            f"- companion 换成 winner 的逐日 gross 后，monthly_robust delta: `{_num(companion_row_transfer.get('robust_delta'))}` | excess annual delta: `{_pct(companion_row_transfer.get('excess_annual_delta'))}`"
        )
    if companion_state_transfer:
        lines.append(
            f"- companion 换成 winner 的状态 gross map 后，monthly_robust delta: `{_num(companion_state_transfer.get('robust_delta'))}` | excess annual delta: `{_pct(companion_state_transfer.get('excess_annual_delta'))}`"
        )
    lines.extend(
        [
            "",
            "## Decision",
        ]
    )
    repair_winner_name = str(repair_winner.get("variant_name", ""))
    if repair_winner_name == "winner_current_target_static":
        lines.append("- 第二轮 follow-up 里，winner-side 仍没有推翻 current default；说明这条默认链需要更深一层的 learned control 才可能继续抬升。")
    else:
        lines.append(f"- 第二轮 winner-side repair winner 已收口到 `{repair_winner_name}`，说明 current default 的下一步修补方向已经进一步明确。")
    if str(winner.get("variant_name", "")).startswith("companion_"):
        lines.append("- overall winner 仍在 companion 侧，说明 current default 还没有完成 recent 一年的追平。")
    if winner_row_transfer and companion_row_transfer:
        if float(winner_row_transfer.get("robust_delta", 0.0) or 0.0) > 0 and float(companion_row_transfer.get("robust_delta", 0.0) or 0.0) < 0:
            lines.append("- 控制层迁移结果是同向的：winner 借用 companion 的 gross 变好、companion 借用 winner 的 gross 变差，说明控制层确实是当前差距的重要来源。")
    if live_preview_payload.get("plan_path"):
        lines.append(f"- 已输出 repair winner 的当前 live 预览计划：`{live_preview_payload.get('plan_path')}`。")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).expanduser() / str(args.root_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    recent_runs_dir = output_dir / "recent_runs"
    recent_runs_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = output_dir / "variant_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_json(Path(args.strongest_summary).expanduser())
    active_manifest = _load_json(Path(args.active_manifest).expanduser())
    winner_recent = summary.get("winner_recent", {})
    companion_recent, companion_source = resolve_repair_companion_entry(
        summary,
        winner_entry=winner_recent if isinstance(winner_recent, dict) else {},
    )
    winner_recent_bundle = load_recent_protocol_bundle(winner_recent if isinstance(winner_recent, dict) else {})
    companion_recent_bundle = load_recent_protocol_bundle(companion_recent if isinstance(companion_recent, dict) else {})
    winner_recent_dir = Path(winner_recent_bundle["run_dir"])
    companion_recent_dir = Path(companion_recent_bundle["run_dir"])

    recent_start = str(summary.get("recent_start_date", "")).strip()
    recent_end = str(summary.get("recent_end_date", "")).strip()
    if not recent_start or not recent_end:
        raise RuntimeError("Strongest-model summary is missing recent_start_date / recent_end_date.")

    winner_score_panel = winner_recent_bundle["score_panel"]
    winner_target_panel = winner_recent_bundle["target_panel"]
    companion_score_panel = companion_recent_bundle["score_panel"]
    companion_target_panel = companion_recent_bundle["target_panel"]
    winner_regime_state = winner_recent_bundle["regime_state"]

    month_features = _build_month_features(winner_score_panel, winner_target_panel, winner_regime_state)
    thresholds = _derive_thresholds(month_features)
    month_plan = _apply_month_plan(month_features, thresholds)
    month_plan.to_csv(output_dir / "month_plan.csv", index=False, encoding="utf-8-sig")
    (output_dir / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    winner_market_state_map = _mean_gross_by_label(winner_target_panel, winner_regime_state, selector="market_state")
    companion_market_state_map = _mean_gross_by_label(companion_target_panel, winner_regime_state, selector="market_state")
    winner_month_state_map = _mean_gross_by_month_state(winner_target_panel, month_plan)
    companion_month_state_map = _mean_gross_by_month_state(companion_target_panel, month_plan)
    balance_market_state_map = _build_balance_state_map(winner_market_state_map, companion_market_state_map)
    attack_defense_month_map = _build_attack_defense_month_map(companion_month_state_map)

    winner_row_gross = _row_gross(winner_target_panel)
    companion_row_gross = _row_gross(companion_target_panel)
    market_state_labels = _label_series(winner_regime_state, winner_target_panel.index, "market_state")
    month_state_labels = _month_state_series(winner_target_panel.index, month_plan)
    score_weight_k2 = _bridge_k2_5d(_build_score_weight_raw_panel(winner_score_panel))

    winner_current_target_static = _sanitize_panel(winner_target_panel)
    winner_current_target_guard_v1 = _apply_soft_state(
        winner_current_target_static,
        winner_regime_state,
        profile="market_state_guard_v1",
    )[0]
    winner_current_target_companion_row_gross = _target_row_gross(winner_current_target_static, companion_row_gross)
    winner_current_target_companion_state_gross = _target_label_gross(
        winner_current_target_static,
        market_state_labels,
        companion_market_state_map,
    )
    winner_current_target_balance_state_gross = _target_label_gross(
        winner_current_target_static,
        market_state_labels,
        balance_market_state_map,
    )
    winner_current_target_month_state_companion = _target_label_gross(
        winner_current_target_static,
        month_state_labels,
        companion_month_state_map,
    )
    winner_current_target_month_state_attack_defense_v2 = _target_label_gross(
        winner_current_target_static,
        month_state_labels,
        attack_defense_month_map,
    )
    winner_current_target_balance_state_power105 = _rowwise_power(
        winner_current_target_balance_state_gross,
        power=1.05,
        preserve_row_gross=True,
    )
    winner_current_target_balance_state_scoreblend15 = _blend_to_base_gross(
        winner_current_target_balance_state_gross,
        score_weight_k2,
        alpha=0.15,
    )
    winner_current_target_companion_state_scoreblend15 = _blend_to_base_gross(
        winner_current_target_companion_state_gross,
        score_weight_k2,
        alpha=0.15,
    )

    companion_current_target_static = _sanitize_panel(companion_target_panel)
    companion_current_target_winner_row_gross = _target_row_gross(companion_current_target_static, winner_row_gross)
    companion_current_target_winner_state_gross = _target_label_gross(
        companion_current_target_static,
        market_state_labels,
        winner_market_state_map,
    )

    variant_builders: dict[str, Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame] | None] = {
        "winner_current_target_static": lambda score, target, regime, plan: _sanitize_panel(target),
        "winner_current_target_market_state_guard_v1_reference": lambda score, target, regime, plan: _apply_soft_state(
            _sanitize_panel(target),
            regime,
            profile="market_state_guard_v1",
        )[0],
        "winner_current_target_companion_row_gross_transfer": lambda score, target, regime, plan: _target_row_gross(
            _sanitize_panel(target),
            companion_row_gross.reindex(target.index).fillna(_row_gross(_sanitize_panel(target))),
        ),
        "winner_current_target_market_state_guard_v2_companion_transfer": lambda score, target, regime, plan: _target_label_gross(
            _sanitize_panel(target),
            _label_series(regime, target.index, "market_state"),
            companion_market_state_map,
        ),
        "winner_current_target_market_state_guard_v2_balance": lambda score, target, regime, plan: _target_label_gross(
            _sanitize_panel(target),
            _label_series(regime, target.index, "market_state"),
            balance_market_state_map,
        ),
        "winner_current_target_month_state_guard_v2_companion_transfer": lambda score, target, regime, plan: _target_label_gross(
            _sanitize_panel(target),
            _month_state_series(target.index, plan),
            companion_month_state_map,
        ),
        "winner_current_target_month_state_attack_defense_v2": lambda score, target, regime, plan: _target_label_gross(
            _sanitize_panel(target),
            _month_state_series(target.index, plan),
            attack_defense_month_map,
        ),
        "winner_current_target_market_state_guard_v2_balance_power105": lambda score, target, regime, plan: _rowwise_power(
            _target_label_gross(_sanitize_panel(target), _label_series(regime, target.index, "market_state"), balance_market_state_map),
            power=1.05,
            preserve_row_gross=True,
        ),
        "winner_current_target_market_state_guard_v2_balance_scoreblend15": lambda score, target, regime, plan: _blend_to_base_gross(
            _target_label_gross(_sanitize_panel(target), _label_series(regime, target.index, "market_state"), balance_market_state_map),
            _bridge_k2_5d(_build_score_weight_raw_panel(score)),
            alpha=0.15,
        ),
        "winner_current_target_market_state_guard_v2_companion_transfer_scoreblend15": lambda score, target, regime, plan: _blend_to_base_gross(
            _target_label_gross(_sanitize_panel(target), _label_series(regime, target.index, "market_state"), companion_market_state_map),
            _bridge_k2_5d(_build_score_weight_raw_panel(score)),
            alpha=0.15,
        ),
        "winner_score_weight_k2_static_reference": lambda score, target, regime, plan: _bridge_k2_5d(_build_score_weight_raw_panel(score)),
        "companion_current_target_static": None,
        "companion_current_target_winner_row_gross_transfer": None,
        "companion_current_target_winner_state_gross_transfer": None,
    }

    variant_panels: dict[str, pd.DataFrame] = {
        "winner_current_target_static": winner_current_target_static,
        "winner_current_target_market_state_guard_v1_reference": winner_current_target_guard_v1,
        "winner_current_target_companion_row_gross_transfer": winner_current_target_companion_row_gross,
        "winner_current_target_market_state_guard_v2_companion_transfer": winner_current_target_companion_state_gross,
        "winner_current_target_market_state_guard_v2_balance": winner_current_target_balance_state_gross,
        "winner_current_target_month_state_guard_v2_companion_transfer": winner_current_target_month_state_companion,
        "winner_current_target_month_state_attack_defense_v2": winner_current_target_month_state_attack_defense_v2,
        "winner_current_target_market_state_guard_v2_balance_power105": winner_current_target_balance_state_power105,
        "winner_current_target_market_state_guard_v2_balance_scoreblend15": winner_current_target_balance_state_scoreblend15,
        "winner_current_target_market_state_guard_v2_companion_transfer_scoreblend15": winner_current_target_companion_state_scoreblend15,
        "winner_score_weight_k2_static_reference": score_weight_k2,
        "companion_current_target_static": companion_current_target_static,
        "companion_current_target_winner_row_gross_transfer": companion_current_target_winner_row_gross,
        "companion_current_target_winner_state_gross_transfer": companion_current_target_winner_state_gross,
    }

    state_maps = {
        "winner_market_state_gross": winner_market_state_map,
        "companion_market_state_gross": companion_market_state_map,
        "balance_market_state_gross": balance_market_state_map,
    }
    month_state_maps = {
        "winner_month_state_gross": winner_month_state_map,
        "companion_month_state_gross": companion_month_state_map,
        "attack_defense_month_state_gross": attack_defense_month_map,
    }
    (output_dir / "state_maps.json").write_text(json.dumps(state_maps, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "month_state_maps.json").write_text(json.dumps(month_state_maps, ensure_ascii=False, indent=2), encoding="utf-8")

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
        scoreboard_rows.append(_collect_run_metrics(run_dir, variant_name, _bridge_summary(score_panel, panel)))

    scoreboard = _rank_scoreboard(pd.DataFrame(scoreboard_rows))
    scoreboard.to_csv(output_dir / "recent_scoreboard.csv", index=False, encoding="utf-8-sig")

    repair_scoreboard = scoreboard.loc[scoreboard["variant_name"].astype(str).str.startswith("winner_")].copy()
    repair_winner = repair_scoreboard.iloc[0].to_dict() if not repair_scoreboard.empty else {}
    repair_winner_name = str(repair_winner.get("variant_name", ""))
    live_preview_payload = _generate_live_preview(
        output_dir=output_dir,
        variant_name=repair_winner_name,
        panel_builder=variant_builders.get(repair_winner_name),
        active_manifest=active_manifest,
        thresholds=thresholds,
        regime_state=winner_regime_state,
        python_executable=str(args.python_executable),
        preview_cash=float(args.preview_cash),
    )

    migration_rows = _migration_rows(scoreboard)
    pd.DataFrame(migration_rows).to_csv(output_dir / "migration_scoreboard.csv", index=False, encoding="utf-8-sig")

    _write_summary(
        output_dir,
        scoreboard=scoreboard,
        thresholds=thresholds,
        month_plan=month_plan,
        state_maps=state_maps,
        month_state_maps=month_state_maps,
        repair_winner=repair_winner,
        migration_rows=migration_rows,
        live_preview_payload=live_preview_payload,
        companion_source=companion_source,
        companion_profile_name=str(companion_recent.get("profile_name", "") or summary.get("recent_winner_profile_name", "")),
    )


if __name__ == "__main__":
    main()
