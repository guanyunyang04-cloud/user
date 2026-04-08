from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.run_short_alpha_targeted_weak_month_repair_review import (
    DEFAULT_STATIC_PROFILE,
    _build_hybrid_target_weight_panel,
    _discover_window,
    _load_window_audit_monthly,
    _mapping_label,
    _run_recent_h2h,
    _run_replay,
)
from daily_research.execution.output_root_resolver import (
    OUTPUT_ROOT,
    STATIC_PRODUCTION_ROOT,
    resolve_formal_execution_audit_roots,
    resolve_recent_execution_audit_root,
    resolve_single_mapping_review_root,
    resolve_single_mapping_root_tag,
)
from daily_research.progress import StageProgress, progress_write


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FALLBACK_LIVE_TARGET_WEIGHT_PANEL = STATIC_PRODUCTION_ROOT / "daily_live_target_weight_panel.csv"
DEFAULT_FALLBACK_LIVE_SCORE_PANEL = STATIC_PRODUCTION_ROOT / "daily_live_score_panel.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Operationalize a single short-alpha execution mapping candidate by materializing full formal "
            "candidate/static panels, trigger coverage summaries, and live trigger monitoring."
        )
    )
    parser.add_argument("--review-root", default="")
    parser.add_argument(
        "--audit-roots",
        default="",
        help="Comma-separated formal execution-policy audit roots.",
    )
    parser.add_argument(
        "--recent-audit-root",
        default="",
        help="Recent execution-policy audit root used for live monitoring.",
    )
    parser.add_argument("--static-profile", default=DEFAULT_STATIC_PROFILE)
    parser.add_argument("--candidate-label", default="candidate_expand_stable_topk3")
    parser.add_argument("--bridge-start", default="2025-03-18")
    parser.add_argument("--weak-start", default="2025-09-05")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--fallback-live-target-weight-panel-csv",
        default=str(DEFAULT_FALLBACK_LIVE_TARGET_WEIGHT_PANEL),
        help="Optional freshest static live target-weight panel used when the candidate is inactive in the latest month.",
    )
    parser.add_argument(
        "--fallback-live-score-panel-csv",
        default=str(DEFAULT_FALLBACK_LIVE_SCORE_PANEL),
        help="Static live score panel copied into the candidate pipeline root as the companion reference score view.",
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default="")
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Skip heavy formal replay/H2H regeneration and only refresh live monitor + live panels.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _resolve_output_dir(output_root: str, root_tag: str) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root / root_tag


def _load_review_context(review_root: Path) -> tuple[dict[str, str], str, str]:
    mapping_payload = _load_json(review_root / "full_training_mapping.json")
    mapping = mapping_payload.get("mapping", {})
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Review root does not contain a non-empty mapping: {review_root}")
    mapping = {str(key): str(value) for key, value in mapping.items()}
    mapping_label = str(mapping_payload.get("mapping_label", "")).strip() or _mapping_label(mapping)
    trigger_mode = str(mapping_payload.get("trigger_mode", "")).strip() or "unknown"
    return mapping, mapping_label, trigger_mode


def _stitch_panels(panels: list[pd.DataFrame]) -> pd.DataFrame:
    if not panels:
        raise ValueError("No panels were provided for stitching.")
    stitched = pd.concat(panels).sort_index()
    stitched = stitched.loc[~stitched.index.duplicated(keep="last")]
    return stitched.fillna(0.0)


def _build_month_outcomes(
    audit_monthly: pd.DataFrame,
    month_choice_df: pd.DataFrame,
    *,
    static_profile: str,
) -> pd.DataFrame:
    selected = audit_monthly.merge(
        month_choice_df[["window_key", "month", "selected_profile"]],
        left_on=["window_key", "month", "profile_name"],
        right_on=["window_key", "month", "selected_profile"],
        how="inner",
    )
    selected = (
        selected.rename(
            columns={
                "profile_name": "targeted_profile_name",
                "excess_return": "targeted_excess_return",
                "avg_turnover": "targeted_avg_turnover",
            }
        )[
            [
                "window_key",
                "month",
                "targeted_profile_name",
                "targeted_excess_return",
                "targeted_avg_turnover",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    static = (
        audit_monthly.loc[audit_monthly["profile_name"].astype(str).eq(static_profile), ["window_key", "month", "excess_return", "avg_turnover"]]
        .rename(columns={"excess_return": "static_excess_return", "avg_turnover": "static_avg_turnover"})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    merged = month_choice_df.merge(selected, on=["window_key", "month"], how="left").merge(static, on=["window_key", "month"], how="left")
    merged["is_triggered"] = merged["selected_profile"].astype(str).ne(str(static_profile))
    merged["delta_excess_return"] = merged["targeted_excess_return"].astype(float) - merged["static_excess_return"].astype(float)
    merged["delta_avg_turnover"] = merged["targeted_avg_turnover"].astype(float) - merged["static_avg_turnover"].astype(float)
    merged["year"] = merged["month"].astype(str).str.slice(0, 4)
    return merged.sort_values(["window_key", "month"]).reset_index(drop=True)


def _build_trigger_summary(frame: pd.DataFrame) -> dict[str, Any]:
    triggered = frame.loc[frame["is_triggered"].astype(bool)].copy()
    summary = {
        "month_count": int(len(frame)),
        "triggered_month_count": int(triggered.shape[0]),
        "trigger_coverage_ratio": float(triggered.shape[0] / len(frame)) if len(frame) else 0.0,
        "triggered_mean_monthly_delta": float(triggered["delta_excess_return"].mean()) if not triggered.empty else 0.0,
        "triggered_median_monthly_delta": float(triggered["delta_excess_return"].median()) if not triggered.empty else 0.0,
        "triggered_positive_delta_ratio": float((triggered["delta_excess_return"].astype(float) > 0.0).mean()) if not triggered.empty else 0.0,
        "triggered_mean_targeted_excess_return": float(triggered["targeted_excess_return"].mean()) if not triggered.empty else 0.0,
        "triggered_mean_static_excess_return": float(triggered["static_excess_return"].mean()) if not triggered.empty else 0.0,
        "non_triggered_month_count": int((~frame["is_triggered"].astype(bool)).sum()),
        "non_triggered_mean_monthly_delta": float(
            frame.loc[~frame["is_triggered"].astype(bool), "delta_excess_return"].mean()
        )
        if len(frame)
        else 0.0,
    }
    return summary


def _empty_trigger_summary() -> dict[str, Any]:
    return {
        "month_count": 0,
        "triggered_month_count": 0,
        "trigger_coverage_ratio": 0.0,
        "triggered_mean_monthly_delta": 0.0,
        "triggered_median_monthly_delta": 0.0,
        "triggered_positive_delta_ratio": 0.0,
        "triggered_mean_targeted_excess_return": 0.0,
        "triggered_mean_static_excess_return": 0.0,
        "non_triggered_month_count": 0,
        "non_triggered_mean_monthly_delta": 0.0,
    }


def _extract_trigger_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = _empty_trigger_summary()
    for key in summary:
        if key in payload:
            summary[key] = payload[key]
    return summary


def _copy_csv(src: Path, dst: Path) -> None:
    frame = pd.read_csv(src)
    frame.to_csv(dst, index=False, encoding="utf-8-sig")


def _refresh_live_outputs(
    *,
    output_dir: Path,
    candidate_panels_dir: Path,
    mapping: dict[str, str],
    mapping_label: str,
    trigger_mode: str,
    recent_choices: pd.DataFrame,
    recent_gate_summary: dict[str, Any],
    recent_audit_root: Path,
    static_profile: str,
    fallback_live_target_weight_panel: Path,
    fallback_live_score_panel: Path,
) -> dict[str, Any]:
    live_target_weight_panel_path = output_dir / "daily_live_target_weight_panel.csv"
    live_score_panel_path = output_dir / "daily_live_score_panel.csv"
    score_reference_meta_path = output_dir / "daily_live_score_reference.json"

    if not recent_choices.empty:
        recent_choices = recent_choices.sort_values("month").reset_index(drop=True)
        recent_choices.to_csv(candidate_panels_dir / "recent_month_policy_choices.csv", index=False, encoding="utf-8-sig")
        recent_window = _discover_window(recent_audit_root)
        recent_candidate_panel = _build_hybrid_target_weight_panel(recent_window, recent_choices)
        recent_candidate_panel.to_csv(
            candidate_panels_dir / "recent_candidate_target_weight_panel.csv",
            index_label="date",
            encoding="utf-8-sig",
        )
        recent_candidate_panel.to_csv(
            live_target_weight_panel_path,
            index_label="date",
            encoding="utf-8-sig",
        )
        recent_static_choice = recent_choices.copy()
        recent_static_choice["selected_profile"] = str(static_profile)
        recent_static_panel = _build_hybrid_target_weight_panel(recent_window, recent_static_choice)
        recent_static_panel.to_csv(
            candidate_panels_dir / "recent_static_target_weight_panel.csv",
            index_label="date",
            encoding="utf-8-sig",
        )
        latest = recent_choices.iloc[-1].to_dict()
        live_triggered = recent_choices["selected_profile"].astype(str).ne(str(static_profile))
        live_monitor: dict[str, Any] = {
            "mapping": mapping,
            "mapping_label": mapping_label,
            "trigger_mode": trigger_mode,
            "latest_month": str(latest.get("month", "")),
            "latest_trigger_key": str(latest.get("trigger_key", "")),
            "latest_regime": str(latest.get("month_start_regime", "")),
            "latest_selected_profile": str(latest.get("selected_profile", "")),
            "candidate_active_now": bool(str(latest.get("selected_profile", "")) != str(static_profile)),
            "live_month_count": int(len(recent_choices)),
            "live_triggered_month_count": int(live_triggered.sum()),
            "live_trigger_coverage_ratio": float(live_triggered.mean()) if len(recent_choices) else 0.0,
            "recent_gate_summary": recent_gate_summary,
        }
        if not bool(live_monitor["candidate_active_now"]) and fallback_live_target_weight_panel.exists():
            _copy_csv(fallback_live_target_weight_panel, live_target_weight_panel_path)
            live_monitor["daily_live_panel_source"] = str(fallback_live_target_weight_panel)
            live_monitor["live_target_weight_mode"] = "static_fallback"
        else:
            live_monitor["daily_live_panel_source"] = str(live_target_weight_panel_path)
            live_monitor["live_target_weight_mode"] = "candidate_triggered"
    else:
        live_monitor = {
            "mapping": mapping,
            "mapping_label": mapping_label,
            "trigger_mode": trigger_mode,
            "candidate_active_now": False,
            "live_month_count": 0,
            "live_triggered_month_count": 0,
            "live_trigger_coverage_ratio": 0.0,
            "recent_gate_summary": recent_gate_summary,
            "live_target_weight_mode": "static_fallback" if fallback_live_target_weight_panel.exists() else "missing",
            "daily_live_panel_source": str(live_target_weight_panel_path),
        }
        if fallback_live_target_weight_panel.exists():
            _copy_csv(fallback_live_target_weight_panel, live_target_weight_panel_path)
            live_monitor["daily_live_panel_source"] = str(fallback_live_target_weight_panel)

    score_reference_meta: dict[str, Any] = {
        "role": "static_reference_score_panel",
        "source_score_panel_csv": "",
        "local_score_panel_csv": str(live_score_panel_path),
        "candidate_active_now": bool(live_monitor.get("candidate_active_now", False)),
        "latest_month": str(live_monitor.get("latest_month", "")),
        "latest_selected_profile": str(live_monitor.get("latest_selected_profile", "")),
        "note": "Companion live score panel remains a static reference view; target weights drive actual execution.",
    }
    if fallback_live_score_panel.exists():
        _copy_csv(fallback_live_score_panel, live_score_panel_path)
        score_reference_meta["source_score_panel_csv"] = str(fallback_live_score_panel)
    elif live_score_panel_path.exists():
        score_reference_meta["role"] = "existing_local_score_panel"
        score_reference_meta["source_score_panel_csv"] = str(live_score_panel_path)
    else:
        score_reference_meta["role"] = "missing"

    score_reference_meta_path.write_text(
        json.dumps(score_reference_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    live_monitor["daily_live_score_panel_source"] = str(score_reference_meta.get("source_score_panel_csv", ""))
    live_monitor["daily_live_score_reference_mode"] = str(score_reference_meta.get("role", ""))
    return live_monitor


def _write_summary_markdown(
    output_dir: Path,
    *,
    mapping_label: str,
    trigger_mode: str,
    candidate_label: str,
    static_profile: str,
    trigger_summary: dict[str, Any],
    live_monitor: dict[str, Any],
    fullbridge_summary_path: Path,
) -> None:
    def pct(value: Any) -> str:
        try:
            return f"{float(value):.2%}"
        except Exception:
            return "n/a"

    lines = [
        "# Short Alpha Execution Single-Mapping Candidate Pipeline",
        "",
        "## Candidate",
        f"- mapping: `{mapping_label}`",
        f"- trigger_mode: `{trigger_mode}`",
        f"- candidate_label: `{candidate_label}`",
        f"- static_profile: `{static_profile}`",
        "",
        "## Trigger Tradeoff",
        f"- formal month_count: `{trigger_summary['month_count']}`",
        f"- triggered_month_count: `{trigger_summary['triggered_month_count']}`",
        f"- trigger_coverage_ratio: `{pct(trigger_summary['trigger_coverage_ratio'])}`",
        f"- triggered_mean_monthly_delta: `{pct(trigger_summary['triggered_mean_monthly_delta'])}`",
        f"- triggered_median_monthly_delta: `{pct(trigger_summary['triggered_median_monthly_delta'])}`",
        f"- triggered_positive_delta_ratio: `{pct(trigger_summary['triggered_positive_delta_ratio'])}`",
        "",
        "## Live Monitor",
        f"- latest_month: `{live_monitor.get('latest_month', 'n/a')}`",
        f"- latest_trigger_key: `{live_monitor.get('latest_trigger_key', 'n/a')}`",
        f"- latest_selected_profile: `{live_monitor.get('latest_selected_profile', 'n/a')}`",
        f"- candidate_active_now: `{bool(live_monitor.get('candidate_active_now', False))}`",
        f"- live_trigger_coverage_ratio: `{pct(live_monitor.get('live_trigger_coverage_ratio', 0.0))}`",
        f"- live_target_weight_mode: `{live_monitor.get('live_target_weight_mode', 'n/a')}`",
        f"- live_target_weight_source: `{live_monitor.get('daily_live_panel_source', 'n/a')}`",
        f"- live_score_reference_mode: `{live_monitor.get('daily_live_score_reference_mode', 'n/a')}`",
        f"- live_score_reference_source: `{live_monitor.get('daily_live_score_panel_source', 'n/a')}`",
        "",
        "## Full Formal Head-to-Head",
        f"- summary: `{fullbridge_summary_path}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    review_root = resolve_single_mapping_review_root(args.review_root)
    if not review_root.exists():
        raise FileNotFoundError(f"Review root does not exist: {review_root}")
    output_dir = _resolve_output_dir(str(args.output_root), resolve_single_mapping_root_tag(str(args.root_tag)))
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_panels_dir = output_dir / "candidate_panels"
    candidate_panels_dir.mkdir(parents=True, exist_ok=True)

    mapping, mapping_label, trigger_mode = _load_review_context(review_root)
    formal_choices = pd.read_csv(review_root / "month_policy_choices.csv")
    recent_choices_path = review_root / "recent_month_policy_choices.csv"
    recent_choices = pd.read_csv(recent_choices_path) if recent_choices_path.exists() else pd.DataFrame()
    recent_gate_summary = _load_json(review_root / "recent_gate_summary.json") if (review_root / "recent_gate_summary.json").exists() else {}
    recent_audit_root = resolve_recent_execution_audit_root(args.recent_audit_root)
    fallback_live_target_weight_panel = Path(str(args.fallback_live_target_weight_panel_csv or "")).resolve()
    fallback_live_score_panel = Path(str(args.fallback_live_score_panel_csv or "")).resolve()
    trigger_summary = _empty_trigger_summary()
    fullbridge_summary_path = output_dir / "fullbridge_h2h" / "summary.md"

    candidate_panels: list[pd.DataFrame] = []
    static_panels: list[pd.DataFrame] = []
    outcome_rows: list[pd.DataFrame] = []
    windows: list[Any] = []
    if not bool(args.live_only):
        audit_roots = resolve_formal_execution_audit_roots(args.audit_roots)
        windows = [_discover_window(path) for path in audit_roots]

    total_stages = 2 if bool(args.live_only) else len(windows) + 4
    with StageProgress(total=total_stages, label="SingleMappingCandidate") as progress:
        if not bool(args.live_only):
            for index, window in enumerate(windows, start=1):
                with progress.stage("Materialize formal window", f"{index}/{len(windows)} {window.window_key}"):
                    month_choice = (
                        formal_choices.loc[formal_choices["window_key"].astype(str).eq(window.window_key)]
                        .copy()
                        .sort_values("month")
                        .reset_index(drop=True)
                    )
                    if month_choice.empty:
                        raise RuntimeError(f"Review root is missing month choices for {window.window_key}")
                    audit_monthly = _load_window_audit_monthly(window)
                    month_outcomes = _build_month_outcomes(audit_monthly, month_choice, static_profile=str(args.static_profile))
                    outcome_rows.append(month_outcomes)
                    candidate_panel = _build_hybrid_target_weight_panel(window, month_choice)
                    candidate_panels.append(candidate_panel)
                    static_choice = month_choice.copy()
                    static_choice["selected_profile"] = str(args.static_profile)
                    static_panels.append(_build_hybrid_target_weight_panel(window, static_choice))

            with progress.stage("Write candidate panels", output_dir.name):
                full_candidate_panel = _stitch_panels(candidate_panels)
                full_static_panel = _stitch_panels(static_panels)
                full_candidate_panel.to_csv(candidate_panels_dir / "formal_candidate_target_weight_panel.csv", index_label="date", encoding="utf-8-sig")
                progress_write(f"Write artifact: {candidate_panels_dir / 'formal_candidate_target_weight_panel.csv'}")
                full_static_panel.to_csv(candidate_panels_dir / "formal_static_target_weight_panel.csv", index_label="date", encoding="utf-8-sig")
                progress_write(f"Write artifact: {candidate_panels_dir / 'formal_static_target_weight_panel.csv'}")
                formal_choices.to_csv(candidate_panels_dir / "formal_month_policy_choices.csv", index=False, encoding="utf-8-sig")

            with progress.stage("Run formal candidate replay", output_dir.name):
                formal_start = min(window.start_date for window in windows)
                formal_end = max(window.end_date for window in windows)
                candidate_run_dir = _run_replay(
                    python_executable=str(args.python_executable),
                    hybrid_panel_path=candidate_panels_dir / "formal_candidate_target_weight_panel.csv",
                    start_date=formal_start,
                    end_date=formal_end,
                    output_dir=output_dir,
                    experiment_tag=f"formal_replays/{args.candidate_label}",
                    candidate_label=str(args.candidate_label),
                )
                static_run_dir = _run_replay(
                    python_executable=str(args.python_executable),
                    hybrid_panel_path=candidate_panels_dir / "formal_static_target_weight_panel.csv",
                    start_date=formal_start,
                    end_date=formal_end,
                    output_dir=output_dir,
                    experiment_tag="formal_replays/static_reference",
                    candidate_label=f"static_{args.static_profile}",
                )
                _run_recent_h2h(
                    python_executable=str(args.python_executable),
                    candidate_run_dir=candidate_run_dir,
                    static_run_dir=static_run_dir,
                    output_dir=output_dir / "fullbridge_h2h",
                    label_a=str(args.candidate_label),
                    label_b=f"static_{args.static_profile}".replace("_ensemble_native_anchor", ""),
                    bridge_start=str(args.bridge_start),
                    weak_start=str(args.weak_start),
                )

            with progress.stage("Summarize tradeoff", output_dir.name):
                formal_outcomes = pd.concat(outcome_rows, ignore_index=True).sort_values(["window_key", "month"]).reset_index(drop=True)
                formal_outcomes.to_csv(output_dir / "formal_month_outcomes.csv", index=False, encoding="utf-8-sig")
                trigger_summary = _build_trigger_summary(formal_outcomes)
                by_window = (
                    formal_outcomes.groupby("window_key", dropna=False)
                    .agg(
                        month_count=("month", "count"),
                        triggered_month_count=("is_triggered", "sum"),
                        mean_monthly_delta=("delta_excess_return", "mean"),
                        median_monthly_delta=("delta_excess_return", "median"),
                    )
                    .reset_index()
                )
                by_window["trigger_coverage_ratio"] = by_window["triggered_month_count"].astype(float) / by_window["month_count"].astype(float)
                by_window.to_csv(output_dir / "formal_trigger_coverage_by_window.csv", index=False, encoding="utf-8-sig")
                by_year = (
                    formal_outcomes.groupby("year", dropna=False)
                    .agg(
                        month_count=("month", "count"),
                        triggered_month_count=("is_triggered", "sum"),
                        mean_monthly_delta=("delta_excess_return", "mean"),
                        median_monthly_delta=("delta_excess_return", "median"),
                    )
                    .reset_index()
                )
                by_year["trigger_coverage_ratio"] = by_year["triggered_month_count"].astype(float) / by_year["month_count"].astype(float)
                by_year.to_csv(output_dir / "formal_trigger_coverage_by_year.csv", index=False, encoding="utf-8-sig")
                by_key = (
                    formal_outcomes.groupby(["trigger_key", "selected_profile"], dropna=False)
                    .agg(
                        month_count=("month", "count"),
                        triggered_month_count=("is_triggered", "sum"),
                        mean_monthly_delta=("delta_excess_return", "mean"),
                        median_monthly_delta=("delta_excess_return", "median"),
                        positive_delta_ratio=("delta_excess_return", lambda series: float((series.astype(float) > 0.0).mean())),
                    )
                    .reset_index()
                    .sort_values(["triggered_month_count", "mean_monthly_delta"], ascending=[False, False])
                    .reset_index(drop=True)
                )
                by_key["trigger_coverage_ratio"] = by_key["triggered_month_count"].astype(float) / by_key["month_count"].astype(float)
                by_key.to_csv(output_dir / "formal_trigger_summary_by_key.csv", index=False, encoding="utf-8-sig")
                (output_dir / "formal_trigger_tradeoff_summary.json").write_text(
                    json.dumps(
                        {
                            "mapping": mapping,
                            "mapping_label": mapping_label,
                            "trigger_mode": trigger_mode,
                            **trigger_summary,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        else:
            with progress.stage("Load formal references", output_dir.name):
                trigger_summary = _extract_trigger_summary(_load_json(output_dir / "formal_trigger_tradeoff_summary.json"))
                if trigger_summary["month_count"] <= 0:
                    progress_write("Live-only refresh is reusing an output root without formal trigger summary; keeping zeroed tradeoff summary.")

        with progress.stage("Write live monitor", output_dir.name):
            live_monitor = _refresh_live_outputs(
                output_dir=output_dir,
                candidate_panels_dir=candidate_panels_dir,
                mapping=mapping,
                mapping_label=mapping_label,
                trigger_mode=trigger_mode,
                recent_choices=recent_choices,
                recent_gate_summary=recent_gate_summary,
                recent_audit_root=recent_audit_root,
                static_profile=str(args.static_profile),
                fallback_live_target_weight_panel=fallback_live_target_weight_panel,
                fallback_live_score_panel=fallback_live_score_panel,
            )
            (output_dir / "live_trigger_monitor.json").write_text(
                json.dumps(live_monitor, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _write_summary_markdown(
                output_dir,
                mapping_label=mapping_label,
                trigger_mode=trigger_mode,
                candidate_label=str(args.candidate_label),
                static_profile=str(args.static_profile),
                trigger_summary=trigger_summary,
                live_monitor=live_monitor,
                fullbridge_summary_path=fullbridge_summary_path,
            )

    print(json.dumps({"output_dir": str(output_dir), "mapping_label": mapping_label}, ensure_ascii=False))


if __name__ == "__main__":
    main()
