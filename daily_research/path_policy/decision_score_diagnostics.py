from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy.labels import PATH20_CUMULATIVE_HORIZONS


STUDIES_ROOT = Path("daily_research/output/path_policy/studies")
DEFAULT_SCORE_NAMES = (
    "trade_utility_score",
    "max_pred_utility",
    "horizon_selected_utility",
    "hit_weighted_utility",
    "short_horizon_blend",
    "forecast_5d_mu",
    "forecast_20d_mu",
    "decision_forecast_blend",
)
CONTEXT_COLUMNS = (
    "history_bucket",
    "stock_seen_in_train",
    "model_family",
    "pred_best_horizon",
    "future_best_horizon",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float(default)
    return resolved if np.isfinite(resolved) else float(default)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"decision_score_diagnostics missing required columns: {missing}")


def _infer_horizons(frame: pd.DataFrame) -> tuple[tuple[int, ...], str]:
    horizons = tuple(
        sorted(
            int(match.group(1))
            for column in frame.columns
            for match in [re.match(r"pred_decision_utility_(\d+)d$", str(column))]
            if match is not None
        )
    )
    if horizons:
        return tuple(dict.fromkeys(horizons)), "columns"
    return PATH20_CUMULATIVE_HORIZONS, "default"


def _hit_label_columns(frame: pd.DataFrame, horizons: tuple[int, ...]) -> list[str]:
    return [f"future_hit_label_{int(horizon)}d" for horizon in horizons if f"future_hit_label_{int(horizon)}d" in frame.columns]


def _rank_ic_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    values: list[float] = []
    for _, group in frame[["date", score_column, target_column]].dropna().groupby("date", sort=True):
        if len(group) < 3:
            continue
        corr = group[score_column].rank().corr(group[target_column].rank())
        if pd.notna(corr) and np.isfinite(float(corr)):
            values.append(float(corr))
    return float(np.mean(values)) if values else 0.0


def _daily_spreads(frame: pd.DataFrame, score_column: str, target_column: str, frac: float = 0.20) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in frame[["date", score_column, target_column]].dropna().groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * float(frac)), 1)
        top = float(group.nlargest(k, score_column)[target_column].mean())
        bottom = float(group.nsmallest(k, score_column)[target_column].mean())
        rows.append({"date": pd.Timestamp(date), "spread": top - bottom})
    return pd.DataFrame(rows)


def _top_bottom_spread_by_date(frame: pd.DataFrame, score_column: str, target_column: str) -> float:
    daily = _daily_spreads(frame, score_column, target_column)
    return float(daily["spread"].mean()) if len(daily) else 0.0


def _monthly_spread_summary(frame: pd.DataFrame, score_column: str, target_column: str) -> dict[str, Any]:
    daily = _daily_spreads(frame, score_column, target_column)
    if daily.empty:
        return {
            "monthly_spread": {},
            "monthly_spread_positive_rate": 0.0,
            "negative_months": [],
            "worst_month": "",
            "worst_month_spread": 0.0,
        }
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = daily.groupby("month", sort=True)["spread"].mean()
    monthly_dict = {str(key): float(value) for key, value in monthly.items()}
    negative_months = [month for month, value in monthly_dict.items() if float(value) <= 0.0]
    worst_month = str(monthly.idxmin()) if len(monthly) else ""
    return {
        "monthly_spread": monthly_dict,
        "monthly_spread_positive_rate": float((monthly > 0.0).mean()) if len(monthly) else 0.0,
        "negative_months": negative_months,
        "worst_month": worst_month,
        "worst_month_spread": float(monthly.loc[worst_month]) if worst_month else 0.0,
    }


def _hit_lift_top20(frame: pd.DataFrame, score_column: str, horizons: tuple[int, ...]) -> float:
    hit_cols = _hit_label_columns(frame, horizons)
    _require_columns(frame, ["date", score_column, *hit_cols])
    work = frame[["date", score_column, *hit_cols]].copy()
    for column in [score_column, *hit_cols]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["_hit_any"] = work[hit_cols].max(axis=1)
    lifts: list[float] = []
    for _, group in work.dropna(subset=[score_column, "_hit_any"]).groupby("date", sort=True):
        if len(group) < 2:
            continue
        k = max(int(len(group) * 0.20), 1)
        top_hit = float(group.nlargest(k, score_column)["_hit_any"].mean())
        all_hit = float(group["_hit_any"].mean())
        lifts.append(top_hit - all_hit)
    return float(np.mean(lifts)) if lifts else 0.0


def _top_bottom_payload(frame: pd.DataFrame, score_column: str, target_column: str, horizons: tuple[int, ...]) -> dict[str, float]:
    work = frame[[score_column, target_column]].dropna()
    if len(work) < 2:
        return {
            "top20_future_utility_mean": 0.0,
            "bottom20_future_utility_mean": 0.0,
            "top20_hit_rate": 0.0,
        }
    k = max(int(len(work) * 0.20), 1)
    top = frame.loc[work.nlargest(k, score_column).index]
    bottom = frame.loc[work.nsmallest(k, score_column).index]
    hit_cols = _hit_label_columns(frame, horizons)
    hit_rate = 0.0
    if set(hit_cols).issubset(frame.columns):
        hit_rate = float(top[hit_cols].apply(pd.to_numeric, errors="coerce").max(axis=1).mean())
    return {
        "top20_future_utility_mean": float(pd.to_numeric(top[target_column], errors="coerce").mean()),
        "bottom20_future_utility_mean": float(pd.to_numeric(bottom[target_column], errors="coerce").mean()),
        "top20_hit_rate": hit_rate,
    }


def _value_counts_payload(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.astype(str).value_counts(dropna=False).sort_index().items()}


def _negative_month_context(frame: pd.DataFrame, score_column: str, target_column: str, negative_months: list[str]) -> dict[str, Any]:
    if not negative_months:
        return {}
    work = frame.copy()
    work["month"] = pd.to_datetime(work["date"]).dt.to_period("M").astype(str)
    work = work[work["month"].isin(negative_months)].dropna(subset=[score_column, target_column])
    if work.empty:
        return {}
    rows: dict[str, Any] = {}
    for month, month_frame in work.groupby("month", sort=True):
        k = max(int(len(month_frame) * 0.20), 1)
        top = month_frame.nlargest(k, score_column)
        context: dict[str, Any] = {"top20_count": int(len(top))}
        for column in CONTEXT_COLUMNS:
            if column in top.columns:
                context[column] = _value_counts_payload(top[column])
        alpha_cols = [column for column in top.columns if column.startswith("alpha_prior_")]
        if alpha_cols:
            context["alpha_prior_means"] = {
                column: _finite_float(pd.to_numeric(top[column], errors="coerce").mean())
                for column in alpha_cols[:12]
            }
        rows[str(month)] = context
    return rows


def add_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    horizons, _ = _infer_horizons(frame)
    required = [
        "date",
        "pred_decision_score",
        "future_decision_score",
        "pred_best_horizon",
    ]
    required.extend(f"pred_decision_utility_{int(horizon)}d" for horizon in horizons)
    required.extend(f"pred_hit_prob_{int(horizon)}d" for horizon in horizons)
    required.extend(f"future_hit_label_{int(horizon)}d" for horizon in horizons)
    _require_columns(frame, required)

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    utility_cols = [f"pred_decision_utility_{int(horizon)}d" for horizon in horizons]
    hit_cols = [f"pred_hit_prob_{int(horizon)}d" for horizon in horizons]
    out[utility_cols + hit_cols] = out[utility_cols + hit_cols].apply(pd.to_numeric, errors="coerce")
    out["max_pred_utility"] = pd.to_numeric(out["pred_decision_score"], errors="coerce")
    out["trade_utility_score"] = (
        pd.to_numeric(out["trade_utility_score"], errors="coerce")
        if "trade_utility_score" in out.columns
        else out[utility_cols].max(axis=1)
    )

    horizon_values = list(horizons)
    horizon_to_col = {int(horizon): f"pred_decision_utility_{int(horizon)}d" for horizon in horizon_values}
    out["horizon_selected_utility"] = [
        _finite_float(row.get(horizon_to_col.get(int(_finite_float(row.get("pred_best_horizon"), 1)), ""), np.nan), np.nan)
        for _, row in out.iterrows()
    ]
    out["hit_weighted_utility"] = (out[utility_cols].to_numpy(dtype=float) * out[hit_cols].to_numpy(dtype=float)).max(axis=1)
    short_weights = (0.50, 0.30, 0.20)
    short_horizons = [item for item in (1, 3, 5) if f"pred_decision_utility_{item}d" in out.columns]
    if len(short_horizons) < 3:
        short_horizons = list(horizons[: min(3, len(horizons))])
    weight_sum = sum(short_weights[: len(short_horizons)]) or 1.0
    out["short_horizon_blend"] = sum(
        (short_weights[pos] / weight_sum) * out[f"pred_decision_utility_{int(horizon)}d"]
        for pos, horizon in enumerate(short_horizons)
    )
    out["forecast_5d_mu"] = pd.to_numeric(out["pred_cum_mu_5d"], errors="coerce") if "pred_cum_mu_5d" in out.columns else out["short_horizon_blend"]
    out["forecast_20d_mu"] = pd.to_numeric(out["pred_cum_mu_20d"], errors="coerce") if "pred_cum_mu_20d" in out.columns else out["max_pred_utility"]
    out["decision_forecast_blend"] = (
        0.50 * out["max_pred_utility"]
        + 0.30 * out["forecast_5d_mu"]
        + 0.20 * out["forecast_20d_mu"]
    )
    return out


def summarize_score(
    frame: pd.DataFrame,
    score_column: str,
    *,
    target_column: str = "future_decision_score",
    horizons: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    _require_columns(frame, ["date", score_column, target_column])
    resolved_horizons = horizons or _infer_horizons(frame)[0]
    work = frame.copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    monthly = _monthly_spread_summary(work, score_column, target_column)
    negative_months = list(monthly["negative_months"])
    return {
        "score": score_column,
        "rank_ic": _rank_ic_by_date(work, score_column, target_column),
        "top_bottom_spread": _top_bottom_spread_by_date(work, score_column, target_column),
        "hit_lift_top20_mean": _hit_lift_top20(work, score_column, resolved_horizons),
        **monthly,
        **_top_bottom_payload(work, score_column, target_column, resolved_horizons),
        "negative_month_context": _negative_month_context(work, score_column, target_column, negative_months),
    }


def _horizon_discovery(scored: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for horizon in horizons:
        score_col = f"pred_decision_utility_{int(horizon)}d"
        target_col = f"future_decision_utility_{int(horizon)}d"
        hit_col = f"future_hit_label_{int(horizon)}d"
        if score_col not in scored.columns or target_col not in scored.columns:
            continue
        hit_lift = 0.0
        if hit_col in scored.columns:
            work = scored[["date", score_col, hit_col]].copy()
            work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
            work[hit_col] = pd.to_numeric(work[hit_col], errors="coerce")
            lifts: list[float] = []
            for _, group in work.dropna().groupby("date", sort=True):
                if len(group) < 2:
                    continue
                k = max(int(len(group) * 0.20), 1)
                lifts.append(float(group.nlargest(k, score_col)[hit_col].mean() - group[hit_col].mean()))
            hit_lift = float(np.mean(lifts)) if lifts else 0.0
        rows[str(int(horizon))] = {
            "rank_ic": _rank_ic_by_date(scored, score_col, target_col),
            "top_bottom_spread": _top_bottom_spread_by_date(scored, score_col, target_col),
            "hit_lift_top20_mean": hit_lift,
        }
    return rows


def summarize_frame(frame: pd.DataFrame, *, score_names: tuple[str, ...] = DEFAULT_SCORE_NAMES) -> dict[str, Any]:
    horizons, horizon_source = _infer_horizons(frame)
    scored = add_score_columns(frame)
    score_candidates = tuple(score_name for score_name in score_names if score_name in scored.columns)
    payload = {
        "row_count": int(len(scored)),
        "date_count": int(scored["date"].nunique()),
        "horizons": [int(item) for item in horizons],
        "horizon_source": horizon_source,
        "horizon_discovery": _horizon_discovery(scored, horizons),
        "pred_best_horizon_distribution": _value_counts_payload(scored["pred_best_horizon"]),
        "future_best_horizon_distribution": _value_counts_payload(scored["future_best_horizon"])
        if "future_best_horizon" in scored.columns
        else {},
        "scores": {},
    }
    for score_name in score_candidates:
        payload["scores"][score_name] = summarize_score(scored, score_name, horizons=horizons)
    return payload


def score_passes_gate(validation_score: dict[str, Any], test_score: dict[str, Any], baseline_negative_count: int) -> dict[str, Any]:
    negative_month_count = len(test_score.get("negative_months", []) or [])
    negative_month_check = negative_month_count < int(baseline_negative_count)
    if negative_month_count == 0:
        negative_month_check = True
    checks = {
        "validation_rank_ic_positive": _finite_float(validation_score.get("rank_ic")) > 0.0,
        "validation_spread_positive": _finite_float(validation_score.get("top_bottom_spread")) > 0.0,
        "validation_hit_lift_positive": _finite_float(validation_score.get("hit_lift_top20_mean")) > 0.0,
        "test_rank_ic_positive": _finite_float(test_score.get("rank_ic")) > 0.0,
        "test_spread_positive": _finite_float(test_score.get("top_bottom_spread")) > 0.0,
        "test_hit_lift_positive": _finite_float(test_score.get("hit_lift_top20_mean")) > 0.0,
        "test_monthly_spread_positive_rate_ge_60pct": _finite_float(test_score.get("monthly_spread_positive_rate")) >= 0.60,
        "negative_month_count_less_than_baseline": negative_month_check,
    }
    return {"passed": all(checks.values()), "checks": checks}


def summarize_study(tag: str, *, studies_root: Path = STUDIES_ROOT) -> dict[str, Any]:
    study_root = studies_root / tag
    if not study_root.exists():
        raise FileNotFoundError(f"study tag not found: {tag}")
    summary_path = study_root / "study_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"study summary not found for tag={tag}: {summary_path}")
    study_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(study_summary.get("status", "")).lower() != "completed":
        raise ValueError(f"study tag is not completed: {tag}")

    validation = summarize_frame(pd.read_csv(study_root / "forecast_predictions_validation.csv"))
    test = summarize_frame(pd.read_csv(study_root / "forecast_predictions_test.csv"))
    baseline_negative_count = len(test["scores"]["max_pred_utility"].get("negative_months", []))
    gates = {
        score_name: score_passes_gate(validation["scores"][score_name], test["scores"][score_name], baseline_negative_count)
        for score_name in DEFAULT_SCORE_NAMES
    }
    return {
        "tag": tag,
        "status": study_summary.get("status"),
        "evidence_verdict": study_summary.get("evidence_verdict"),
        "dataset_ids": [
            item
            for item in [
                study_summary.get("lake_dataset_id"),
                study_summary.get("dataset_manifest", {}).get("source_pool_view_id"),
                study_summary.get("dataset_manifest", {}).get("source_sector_board_view_id"),
            ]
            if item
        ],
        "model_families": study_summary.get("model_families", []),
        "role_years": study_summary.get("dataset_manifest", {}).get("role_years", {}),
        "validation": validation,
        "test": test,
        "score_gates": gates,
        "any_score_passed_gate": any(item.get("passed") for item in gates.values()),
    }


def _load_target_audit_payload(target_audit: str | Path | dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if target_audit is None:
        return None, ""
    if isinstance(target_audit, dict):
        return dict(target_audit), ""
    path = Path(target_audit)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("target audit payload must be a JSON object.")
    return payload, str(path.resolve())


def _target_audit_context(target_audit: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    payload, source_path = _load_target_audit_payload(target_audit)
    if payload is None:
        return {}
    study_hints: list[dict[str, Any]] = []
    for study in payload.get("studies", []) or []:
        if not isinstance(study, dict):
            continue
        for target in study.get("targets", []) or []:
            if not isinstance(target, dict):
                continue
            target_info = target.get("target", {}) if isinstance(target.get("target"), dict) else {}
            validation = target.get("validation", {}) if isinstance(target.get("validation"), dict) else {}
            test = target.get("test", {}) if isinstance(target.get("test"), dict) else {}
            gate = target.get("gate_a", {}) if isinstance(target.get("gate_a"), dict) else {}
            study_hints.append(
                {
                    "tag": str(study.get("tag", "")),
                    "target": str(target_info.get("name", "")),
                    "gate_a_passed": bool(gate.get("passed", False)),
                    "validation_hint": str(validation.get("conclusion_hint", "")),
                    "test_hint": str(test.get("conclusion_hint", "")),
                }
            )
    gate = payload.get("gate_a", {}) if isinstance(payload.get("gate_a"), dict) else {}
    return {
        "source_path": source_path,
        "schema_version": payload.get("schema_version"),
        "gate_a_passed": bool(gate.get("passed", False)),
        "conclusion_hints": [str(item) for item in payload.get("conclusion_hints", []) or []],
        "study_hints": study_hints,
    }


def _score_target_inconsistency_reasons(target_context: dict[str, Any]) -> list[str]:
    if not target_context:
        return []
    hints = {
        str(item)
        for item in target_context.get("conclusion_hints", []) or []
        if str(item).strip()
    }
    for row in target_context.get("study_hints", []) or []:
        if not isinstance(row, dict):
            continue
        for key in ("validation_hint", "test_hint"):
            value = str(row.get(key, "")).strip()
            if value:
                hints.add(value)
    if "ranking_signal_exists_but_hit_target_is_miscalibrated" in hints:
        return ["ranking_signal_exists_but_hit_target_is_miscalibrated"]
    if "ranking_signal_exists_but_hit_or_utility_target_is_miscalibrated" in hints:
        return ["ranking_signal_exists_but_hit_or_utility_target_is_miscalibrated"]
    priority = (
        "hit_label_base_rate_out_of_range",
        "score_selection_or_regime_instability",
        "target_calibration_inconclusive",
    )
    return [item for item in priority if item in hints]


def build_diagnostics(
    tags: list[str],
    *,
    studies_root: Path = STUDIES_ROOT,
    target_audit: str | Path | dict[str, Any] | None = None,
) -> dict[str, Any]:
    studies = [summarize_study(tag, studies_root=studies_root) for tag in tags]
    passed = [
        {"tag": study["tag"], "score": score_name}
        for study in studies
        for score_name, gate in study["score_gates"].items()
        if gate.get("passed")
    ]
    stock_mixer = next((study for study in studies if "stock_mixer" in study["tag"]), None)
    stock_mixer_near_miss = False
    if stock_mixer:
        max_score = stock_mixer["test"]["scores"]["max_pred_utility"]
        stock_mixer_near_miss = (
            _finite_float(max_score.get("rank_ic")) > 0.0
            and _finite_float(max_score.get("top_bottom_spread")) > 0.0
            and _finite_float(max_score.get("monthly_spread_positive_rate")) >= 0.60
            and _finite_float(max_score.get("hit_lift_top20_mean")) <= 0.0
        )
    target_context = _target_audit_context(target_audit)
    return {
        "schema_version": 1,
        "purpose": "Path20 decision score diagnostics and selection calibration audit.",
        "score_names": list(DEFAULT_SCORE_NAMES),
        "studies": studies,
        "passed_score_candidates": passed,
        "stock_mixer_near_miss": bool(stock_mixer_near_miss),
        "target_audit_context": target_context,
        "score_target_inconsistency_reasons": _score_target_inconsistency_reasons(target_context),
        "conclusion_hint": _conclusion_hint(passed, stock_mixer_near_miss),
    }


def _conclusion_hint(passed: list[dict[str, str]], stock_mixer_near_miss: bool) -> str:
    if passed:
        return "selection_calibration_candidate_found"
    if stock_mixer_near_miss:
        return "ranking_signal_exists_but_hit_or_utility_target_is_miscalibrated"
    return "no_selection_only_fix_found; inspect target_definition_or_input_regime_contamination"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Path20 decision score diagnostics.")
    parser.add_argument("--tags", required=True, help="Comma-separated completed study tags.")
    parser.add_argument("--studies-root", default=str(STUDIES_ROOT))
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--target-audit", default="", help="Optional target_calibration_audit.py output JSON path.")
    args = parser.parse_args(argv)

    tags = [item.strip() for item in str(args.tags).split(",") if item.strip()]
    if not tags:
        parser.error("--tags must include at least one study tag.")
    payload = build_diagnostics(
        tags,
        studies_root=Path(args.studies_root),
        target_audit=Path(args.target_audit) if str(args.target_audit).strip() else None,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "conclusion_hint": payload["conclusion_hint"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
