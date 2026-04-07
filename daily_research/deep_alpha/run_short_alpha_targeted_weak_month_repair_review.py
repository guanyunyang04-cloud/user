from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
WEAK_MONTH_REVIEW_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_short_alpha_weak_month_review.py"
EXTERNAL_REPLAY_SCRIPT = PROJECT_ROOT / "daily_research" / "baseline" / "backtest_external_score_panel.py"
DEFAULT_FORMAL_ROOT = OUTPUT_ROOT / "short_alpha_formal_head2head_20260404_monthly_budgetnorm_r1"
DEFAULT_AUDIT_ROOTS = (
    OUTPUT_ROOT / "short_alpha_formal_execution_policy_audit_20230216_20240229_20260405_r1",
    OUTPUT_ROOT / "short_alpha_formal_execution_policy_audit_20240301_20250317_20260405_r1",
    OUTPUT_ROOT / "short_alpha_formal_execution_policy_audit_20260405_r1",
)
DEFAULT_STATIC_PROFILE = "regoff_k1_5d_ensemble_native_anchor"


@dataclass(frozen=True)
class AuditWindow:
    window_key: str
    audit_root: Path
    profiles_root: Path
    start_date: str
    end_date: str
    month_regime_map: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a narrow targeted weak-month repair plan for liquid500 short_alpha by "
            "learning small regime-specific overrides from training windows only and "
            "evaluating them leave-window-out."
        )
    )
    parser.add_argument("--formal-root", default=str(DEFAULT_FORMAL_ROOT))
    parser.add_argument(
        "--audit-roots",
        default=",".join(str(path) for path in DEFAULT_AUDIT_ROOTS),
        help="Comma-separated formal execution-policy audit roots.",
    )
    parser.add_argument("--candidate-profile", default="state_liquidity_listwise_v1")
    parser.add_argument("--baseline-profile", default="baseline_current")
    parser.add_argument("--static-profile", default=DEFAULT_STATIC_PROFILE)
    parser.add_argument("--current-policy", default=DEFAULT_STATIC_PROFILE)
    parser.add_argument(
        "--trigger-mode",
        choices=[
            "regime",
            "market_state",
            "trend_vol",
            "regime_market_state",
            "regime_weight_count",
            "regime_signal_shape",
            "regime_firstweek_weight_drift",
            "regime_firstweek_score_followthrough",
            "regime_firstweek_combo",
        ],
        default="regime",
        help="Which month-start trigger key the targeted repair plan should learn on.",
    )
    parser.add_argument("--min-regime-support", type=int, default=2)
    parser.add_argument("--min-regime-lift", type=float, default=0.02)
    parser.add_argument("--max-regimes-considered", type=int, default=3)
    parser.add_argument("--max-targeted-regimes", type=int, default=2)
    parser.add_argument("--max-profiles-per-regime", type=int, default=2)
    parser.add_argument("--selection-objective", choices=["mean_excess_return", "monthly_robust_score"], default="mean_excess_return")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--root-tag", default="short_alpha_targeted_weak_month_repair_review_20260406_r1")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_long_panel(path: Path, value_name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["stock"] = raw["stock"].astype(str).str.upper().str.strip()
    raw[value_name] = pd.to_numeric(raw[value_name], errors="coerce")
    raw = raw.dropna(subset=["date", value_name])
    return (
        raw.sort_values(["date", "stock"])
        .drop_duplicates(subset=["date", "stock"], keep="last")
        .pivot(index="date", columns="stock", values=value_name)
        .sort_index()
        .fillna(0.0)
    )


def _parse_profile_name(profile_run_dir: Path) -> str:
    parts = profile_run_dir.name.split("_")
    if len(parts) >= 3 and all(token.isdigit() and len(token) == 8 for token in parts[-2:]):
        return "_".join(parts[:-2])
    if len(parts) < 3:
        return profile_run_dir.name
    return profile_run_dir.name


def _resolve_month_regime_map(profile_run_dir: Path) -> dict[str, str]:
    regime = pd.read_csv(profile_run_dir / "regime_state.csv")
    regime["Date"] = pd.to_datetime(regime["Date"], errors="coerce")
    regime = regime.dropna(subset=["Date"]).copy()
    regime["month"] = regime["Date"].dt.to_period("M").astype(str)
    out: dict[str, str] = {}
    for month, frame in regime.groupby("month", sort=True):
        values = frame["quadrant"].dropna()
        out[str(month)] = str(values.iloc[0]).strip() if not values.empty and str(values.iloc[0]).strip() else "not_ready"
    return out


def _discover_window(audit_root: Path) -> AuditWindow:
    profiles_root = audit_root / "profiles"
    first_profile_run = next(path for path in sorted(profiles_root.iterdir()) if path.is_dir())
    monthly = pd.read_csv(first_profile_run / "monthly_backtest_summary.csv")
    start_date = pd.to_datetime(monthly["start_date"], errors="coerce").min().strftime("%Y%m%d")
    end_date = pd.to_datetime(monthly["end_date"], errors="coerce").max().strftime("%Y%m%d")
    return AuditWindow(
        window_key=f"{start_date}_{end_date}",
        audit_root=audit_root,
        profiles_root=profiles_root,
        start_date=start_date,
        end_date=end_date,
        month_regime_map=_resolve_month_regime_map(first_profile_run),
    )


def _load_window_audit_monthly(window: AuditWindow) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for profile_run_dir in sorted(window.profiles_root.iterdir()):
        if not profile_run_dir.is_dir():
            continue
        profile_name = _parse_profile_name(profile_run_dir)
        monthly = pd.read_csv(profile_run_dir / "monthly_backtest_summary.csv")
        monthly["month"] = monthly["month"].astype(str)
        monthly.insert(0, "window_key", window.window_key)
        monthly.insert(1, "profile_name", profile_name)
        monthly["month_start_regime"] = monthly["month"].map(window.month_regime_map).fillna("not_ready")
        rows.append(monthly)
    if not rows:
        raise RuntimeError(f"No audit monthly summaries were found under {window.audit_root}")
    return pd.concat(rows, ignore_index=True)


def _run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))


def _run_weak_month_review(args: argparse.Namespace, output_dir: Path) -> Path:
    weak_month_tag = f"{args.root_tag}/weak_month_review"
    cmd = [
        str(args.python_executable),
        str(WEAK_MONTH_REVIEW_SCRIPT),
        "--formal-root",
        str(Path(args.formal_root).resolve()),
        "--audit-roots",
        str(args.audit_roots),
        "--candidate-profile",
        str(args.candidate_profile),
        "--baseline-profile",
        str(args.baseline_profile),
        "--current-policy",
        str(args.current_policy),
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--root-tag",
        weak_month_tag,
    ]
    _run_command(cmd)
    return Path(args.output_root).resolve() / weak_month_tag


def _normalize_text(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def _apply_trigger_key(frame: pd.DataFrame, *, trigger_mode: str) -> pd.DataFrame:
    enriched = frame.copy()
    if "month_start_regime" not in enriched.columns:
        enriched["month_start_regime"] = "not_ready"
    if "month_start_market_state" not in enriched.columns:
        enriched["month_start_market_state"] = ""
    if "month_start_trend_bucket" not in enriched.columns:
        enriched["month_start_trend_bucket"] = ""
    if "month_start_vol_bucket" not in enriched.columns:
        enriched["month_start_vol_bucket"] = ""
    if "month_start_weight_count" not in enriched.columns:
        enriched["month_start_weight_count"] = "count0"
    if "month_start_signal_shape" not in enriched.columns:
        enriched["month_start_signal_shape"] = "zero"
    if "first_week_weight_drift" not in enriched.columns:
        enriched["first_week_weight_drift"] = "stable"
    if "first_week_score_followthrough" not in enriched.columns:
        enriched["first_week_score_followthrough"] = "flat"
    if "first_week_combo" not in enriched.columns:
        enriched["first_week_combo"] = "flat|stable"
    enriched["month_start_regime"] = enriched["month_start_regime"].map(lambda x: _normalize_text(x, "not_ready"))
    enriched["month_start_market_state"] = enriched["month_start_market_state"].map(lambda x: _normalize_text(x, "unknown"))
    enriched["month_start_trend_bucket"] = enriched["month_start_trend_bucket"].map(lambda x: _normalize_text(x, "unknown"))
    enriched["month_start_vol_bucket"] = enriched["month_start_vol_bucket"].map(lambda x: _normalize_text(x, "unknown"))
    enriched["month_start_weight_count"] = enriched["month_start_weight_count"].map(lambda x: _normalize_text(x, "count0"))
    enriched["month_start_signal_shape"] = enriched["month_start_signal_shape"].map(lambda x: _normalize_text(x, "zero"))
    enriched["first_week_weight_drift"] = enriched["first_week_weight_drift"].map(lambda x: _normalize_text(x, "stable"))
    enriched["first_week_score_followthrough"] = enriched["first_week_score_followthrough"].map(lambda x: _normalize_text(x, "flat"))
    enriched["first_week_combo"] = enriched["first_week_combo"].map(lambda x: _normalize_text(x, "flat|stable"))

    if trigger_mode == "market_state":
        enriched["trigger_key"] = enriched["month_start_market_state"]
    elif trigger_mode == "trend_vol":
        enriched["trigger_key"] = (
            enriched["month_start_trend_bucket"].astype(str)
            + "|"
            + enriched["month_start_vol_bucket"].astype(str)
        )
    elif trigger_mode == "regime_market_state":
        enriched["trigger_key"] = (
            enriched["month_start_regime"].astype(str)
            + "|"
            + enriched["month_start_market_state"].astype(str)
        )
    elif trigger_mode == "regime_weight_count":
        enriched["trigger_key"] = (
            enriched["month_start_regime"].astype(str)
            + "|"
            + enriched["month_start_weight_count"].astype(str)
        )
    elif trigger_mode == "regime_signal_shape":
        enriched["trigger_key"] = (
            enriched["month_start_regime"].astype(str)
            + "|"
            + enriched["month_start_signal_shape"].astype(str)
        )
    elif trigger_mode == "regime_firstweek_weight_drift":
        enriched["trigger_key"] = (
            enriched["month_start_regime"].astype(str)
            + "|"
            + enriched["first_week_weight_drift"].astype(str)
        )
    elif trigger_mode == "regime_firstweek_score_followthrough":
        enriched["trigger_key"] = (
            enriched["month_start_regime"].astype(str)
            + "|"
            + enriched["first_week_score_followthrough"].astype(str)
        )
    elif trigger_mode == "regime_firstweek_combo":
        enriched["trigger_key"] = (
            enriched["month_start_regime"].astype(str)
            + "|"
            + enriched["first_week_combo"].astype(str)
        )
    else:
        enriched["trigger_key"] = enriched["month_start_regime"]
    return enriched


def _top3_positive_share(series: pd.Series) -> float:
    positive = series.loc[series > 0.0].sort_values(ascending=False)
    if positive.empty:
        return 0.0
    return float(positive.head(3).sum() / positive.sum())


def _longest_negative_streak(series: pd.Series) -> int:
    longest = 0
    current = 0
    for value in series.tolist():
        if float(value) < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _monthly_robust_score(series: pd.Series) -> float:
    if series.empty:
        return float("-inf")
    positive_ratio = float((series > 0.0).mean())
    median_monthly_return = float(series.median())
    mean_monthly_return = float(series.mean())
    worst_monthly_return = float(series.min())
    top3_positive_share = _top3_positive_share(series)
    longest_negative_streak = _longest_negative_streak(series)
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


def _build_plan_metrics(plan_frame: pd.DataFrame) -> dict[str, Any]:
    ordered = plan_frame.sort_values(["window_key", "month"]).reset_index(drop=True)
    monthly_excess = ordered["excess_return"].astype(float)
    return {
        "month_count": int(len(ordered)),
        "mean_excess_return": float(monthly_excess.mean()) if not ordered.empty else float("nan"),
        "median_monthly_return": float(monthly_excess.median()) if not ordered.empty else float("nan"),
        "positive_month_ratio": float((monthly_excess > 0.0).mean()) if not ordered.empty else float("nan"),
        "worst_monthly_return": float(monthly_excess.min()) if not ordered.empty else float("nan"),
        "top3_positive_month_share": _top3_positive_share(monthly_excess),
        "longest_negative_streak": _longest_negative_streak(monthly_excess),
        "monthly_robust_score": _monthly_robust_score(monthly_excess),
        "avg_turnover": float(ordered["avg_turnover"].astype(float).mean()) if "avg_turnover" in ordered.columns and not ordered.empty else float("nan"),
    }


def _training_rank_tuple(metrics: dict[str, Any], selection_objective: str) -> tuple[float, float, float, float, float]:
    if selection_objective == "monthly_robust_score":
        return (
            float(metrics.get("monthly_robust_score", float("-inf"))),
            float(metrics.get("mean_excess_return", float("-inf"))),
            float(metrics.get("positive_month_ratio", float("-inf"))),
            float(metrics.get("median_monthly_return", float("-inf"))),
            -float(metrics.get("avg_turnover", float("inf"))),
        )
    return (
        float(metrics.get("mean_excess_return", float("-inf"))),
        float(metrics.get("monthly_robust_score", float("-inf"))),
        float(metrics.get("positive_month_ratio", float("-inf"))),
        float(metrics.get("median_monthly_return", float("-inf"))),
        -float(metrics.get("avg_turnover", float("inf"))),
    )


def _mapping_label(mapping: dict[str, str]) -> str:
    if not mapping:
        return "static_only"
    items = [f"{key}->{mapping[key]}" for key in sorted(mapping)]
    return ";".join(items)


def _evaluate_plan_on_monthly(
    audit_monthly: pd.DataFrame,
    *,
    static_profile: str,
    mapping: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    monthly_meta = (
        audit_monthly[["window_key", "month", "month_start_regime", "trigger_key"]]
        .drop_duplicates()
        .sort_values(["window_key", "month"])
        .reset_index(drop=True)
    )
    monthly_meta["selected_profile"] = monthly_meta["trigger_key"].map(mapping).fillna(static_profile)
    selected = audit_monthly.merge(
        monthly_meta[["window_key", "month", "selected_profile"]],
        left_on=["window_key", "month", "profile_name"],
        right_on=["window_key", "month", "selected_profile"],
        how="inner",
    )
    if len(selected) != len(monthly_meta):
        missing = monthly_meta.merge(
            selected[["window_key", "month", "selected_profile"]],
            on=["window_key", "month", "selected_profile"],
            how="left",
            indicator=True,
        )
        missing = missing.loc[missing["_merge"] != "both"]
        raise RuntimeError(f"Plan selection missing audit monthly rows: {missing.head(10).to_dict(orient='records')}")
    metrics = _build_plan_metrics(selected)
    return selected, metrics


def _derive_trigger_candidates(
    weak_df: pd.DataFrame,
    *,
    min_regime_support: int,
    min_regime_lift: float,
    max_regimes_considered: int,
    max_profiles_per_regime: int,
    static_profile: str,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    trigger_summary = (
        weak_df.groupby("trigger_key", dropna=False)
        .agg(
            weak_month_count=("month", "count"),
            avg_candidate_delta_vs_baseline=("delta_excess_return", "mean"),
            avg_best_policy_delta_vs_current=("best_policy_delta_vs_current", "mean"),
            avg_candidate_gap_vs_best_policy=("candidate_gap_vs_best_policy", "mean"),
        )
        .reset_index()
        .sort_values(
            ["weak_month_count", "avg_best_policy_delta_vs_current", "avg_candidate_gap_vs_best_policy"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    eligible = trigger_summary.loc[
        trigger_summary["weak_month_count"].astype(int).ge(int(min_regime_support))
        & trigger_summary["avg_best_policy_delta_vs_current"].astype(float).ge(float(min_regime_lift))
    ].copy()
    if max_regimes_considered > 0:
        eligible = eligible.head(int(max_regimes_considered)).copy()

    profile_choices: dict[str, list[str]] = {}
    for trigger_key in eligible["trigger_key"].astype(str).tolist():
        trigger_profiles = (
            weak_df.loc[weak_df["trigger_key"].astype(str).eq(trigger_key)]
            .groupby("best_policy_name", dropna=False)
            .agg(
                weak_month_wins=("month", "count"),
                avg_lift_vs_current=("best_policy_delta_vs_current", "mean"),
                avg_candidate_gap=("candidate_gap_vs_best_policy", "mean"),
            )
            .reset_index()
            .rename(columns={"best_policy_name": "profile_name"})
            .sort_values(["weak_month_wins", "avg_lift_vs_current"], ascending=[False, False])
        )
        candidates = [
            str(name)
            for name in trigger_profiles["profile_name"].astype(str).tolist()
            if str(name).strip() and str(name) != static_profile
        ]
        if max_profiles_per_regime > 0:
            candidates = candidates[: int(max_profiles_per_regime)]
        if candidates:
            profile_choices[trigger_key] = candidates
    eligible = eligible.loc[eligible["trigger_key"].astype(str).isin(profile_choices.keys())].copy()
    return eligible.reset_index(drop=True), profile_choices


def _generate_mappings(
    trigger_candidates: list[str],
    profile_choices: dict[str, list[str]],
    *,
    max_targeted_regimes: int,
) -> list[dict[str, str]]:
    plans: list[dict[str, str]] = [{}]
    if not trigger_candidates:
        return plans
    max_size = min(max_targeted_regimes, len(trigger_candidates))
    for size in range(1, max_size + 1):
        for subset in combinations(trigger_candidates, size):
            choice_lists = [profile_choices[trigger_key] for trigger_key in subset]
            for profile_combo in product(*choice_lists):
                mapping = {trigger_key: profile for trigger_key, profile in zip(subset, profile_combo, strict=True)}
                plans.append(mapping)
    return plans


def _select_best_training_plan(
    train_audit_monthly: pd.DataFrame,
    train_weak_df: pd.DataFrame,
    *,
    static_profile: str,
    trigger_mode: str,
    min_regime_support: int,
    min_regime_lift: float,
    max_regimes_considered: int,
    max_targeted_regimes: int,
    max_profiles_per_regime: int,
    selection_objective: str,
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    train_audit_monthly = _apply_trigger_key(train_audit_monthly, trigger_mode=trigger_mode)
    train_weak_df = _apply_trigger_key(train_weak_df, trigger_mode=trigger_mode)
    trigger_summary, profile_choices = _derive_trigger_candidates(
        train_weak_df,
        min_regime_support=min_regime_support,
        min_regime_lift=min_regime_lift,
        max_regimes_considered=max_regimes_considered,
        max_profiles_per_regime=max_profiles_per_regime,
        static_profile=static_profile,
    )
    trigger_candidates = trigger_summary["trigger_key"].astype(str).tolist()
    plan_rows: list[dict[str, Any]] = []
    best_mapping: dict[str, str] = {}
    best_rank: tuple[float, float, float, float, float] | None = None

    for mapping in _generate_mappings(
        trigger_candidates,
        profile_choices,
        max_targeted_regimes=max_targeted_regimes,
    ):
        _, metrics = _evaluate_plan_on_monthly(
            train_audit_monthly,
            static_profile=static_profile,
            mapping=mapping,
        )
        row = {
            "plan_label": _mapping_label(mapping),
            "targeted_regime_count": int(len(mapping)),
            "mapping_json": json.dumps(mapping, ensure_ascii=False, sort_keys=True),
            **metrics,
        }
        plan_rows.append(row)
        rank = _training_rank_tuple(metrics, selection_objective)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_mapping = dict(mapping)

    plan_frame = pd.DataFrame(plan_rows).sort_values(
        by=[
            "mean_excess_return" if selection_objective == "mean_excess_return" else "monthly_robust_score",
            "monthly_robust_score" if selection_objective == "mean_excess_return" else "mean_excess_return",
            "positive_month_ratio",
            "median_monthly_return",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return best_mapping, plan_frame, trigger_summary


def _build_hybrid_target_weight_panel(window: AuditWindow, month_choice_df: pd.DataFrame) -> pd.DataFrame:
    panels: dict[str, pd.DataFrame] = {}
    for profile_name in sorted(month_choice_df["selected_profile"].astype(str).unique()):
        profile_dirs = [path for path in window.profiles_root.iterdir() if path.is_dir() and _parse_profile_name(path) == profile_name]
        if not profile_dirs:
            raise FileNotFoundError(f"Profile run not found for {profile_name} under {window.audit_root}")
        panels[profile_name] = _load_long_panel(profile_dirs[0] / "aligned_daily_target_weight_panel.csv", "target_weight")

    stitched: list[pd.DataFrame] = []
    for _, row in month_choice_df.sort_values("month").iterrows():
        period = pd.Period(str(row["month"]), freq="M")
        panel = panels[str(row["selected_profile"])]
        month_panel = panel.loc[panel.index.to_period("M") == period]
        if not month_panel.empty:
            stitched.append(month_panel)
    if not stitched:
        raise RuntimeError(f"No hybrid target-weight panels were constructed for {window.window_key}")
    hybrid = pd.concat(stitched).sort_index()
    hybrid = hybrid.loc[~hybrid.index.duplicated(keep="last")]
    return hybrid.fillna(0.0)


def _load_static_metrics(window: AuditWindow, static_profile: str) -> tuple[dict[str, Any], Path]:
    for path in window.profiles_root.iterdir():
        if path.is_dir() and _parse_profile_name(path) == static_profile:
            return _load_json(path / "metrics.json"), path
    raise FileNotFoundError(f"Static profile {static_profile} not found under {window.audit_root}")


def _derive_signal_shape(nonzero_weight_count: int) -> str:
    if nonzero_weight_count <= 0:
        return "zero"
    if nonzero_weight_count <= 3:
        return "tight"
    if nonzero_weight_count == 4:
        return "mid"
    return "broad"


def _derive_first_week_weight_drift(
    start_weight_count: int,
    end_weight_count: int,
    start_top1_weight: float,
    end_top1_weight: float,
) -> str:
    count_delta = int(end_weight_count) - int(start_weight_count)
    top1_delta = float(end_top1_weight) - float(start_top1_weight)
    if count_delta <= -1 or top1_delta >= 0.02:
        return "tighten"
    if count_delta >= 1 or top1_delta <= -0.02:
        return "loosen"
    return "stable"


def _derive_first_week_score_followthrough(
    start_positive_share: float,
    end_positive_share: float,
    start_gap_1_5: float,
    end_gap_1_5: float,
) -> str:
    breadth_delta = float(end_positive_share) - float(start_positive_share)
    gap_delta = float(end_gap_1_5) - float(start_gap_1_5)
    if breadth_delta >= 0.08 or gap_delta >= 0.03:
        return "expand"
    if breadth_delta <= -0.08 or gap_delta <= -0.03:
        return "fade"
    return "flat"


def _summarize_score_panel(score_frame: pd.DataFrame) -> tuple[float, float]:
    ordered = score_frame["score"].astype(float).sort_values(ascending=False).reset_index(drop=True)
    positive_share = float((ordered > 0.0).mean()) if not ordered.empty else 0.0
    gap_1_5 = float(ordered.iloc[0] - ordered.iloc[min(4, len(ordered) - 1)]) if not ordered.empty else 0.0
    return positive_share, gap_1_5


def _summarize_weight_panel(weight_frame: pd.DataFrame) -> tuple[int, float, float]:
    ordered = weight_frame["target_weight"].astype(float).sort_values(ascending=False).reset_index(drop=True)
    nonzero_weight_count = int((ordered > 0.0).sum())
    top1_weight = float(ordered.iloc[0]) if not ordered.empty else 0.0
    top2_weight_sum = float(ordered.head(2).sum()) if not ordered.empty else 0.0
    return nonzero_weight_count, top1_weight, top2_weight_sum


def _load_month_start_signal_diagnostics(static_run_dir: Path) -> pd.DataFrame:
    score = pd.read_csv(static_run_dir / "aligned_daily_score_panel.csv")
    weight = pd.read_csv(static_run_dir / "aligned_daily_target_weight_panel.csv")
    score["date"] = pd.to_datetime(score["date"], errors="coerce")
    weight["date"] = pd.to_datetime(weight["date"], errors="coerce")
    score = score.dropna(subset=["date"]).copy()
    weight = weight.dropna(subset=["date"]).copy()

    rows: list[dict[str, Any]] = []
    for month, month_scores in score.groupby(score["date"].dt.to_period("M"), sort=True):
        month_dates = sorted(pd.to_datetime(month_scores["date"], errors="coerce").dropna().unique().tolist())
        if not month_dates:
            continue
        first_date = pd.Timestamp(month_dates[0])
        first_week_dates = [pd.Timestamp(item) for item in month_dates[:5]]
        last_first_week_date = first_week_dates[-1]

        first_score_frame = month_scores.loc[month_scores["date"].eq(first_date), ["stock", "score"]].copy()
        first_weight_frame = weight.loc[weight["date"].eq(first_date), ["stock", "target_weight"]].copy()
        first_positive_share, first_gap_1_5 = _summarize_score_panel(first_score_frame)
        first_nonzero_weight_count, first_top1_weight, first_top2_weight_sum = _summarize_weight_panel(first_weight_frame)

        week_score_daily: list[dict[str, Any]] = []
        week_weight_daily: list[dict[str, Any]] = []
        for current_date in first_week_dates:
            score_day = month_scores.loc[month_scores["date"].eq(current_date), ["stock", "score"]].copy()
            weight_day = weight.loc[weight["date"].eq(current_date), ["stock", "target_weight"]].copy()
            positive_share, gap_1_5 = _summarize_score_panel(score_day)
            nonzero_weight_count, top1_weight, top2_weight_sum = _summarize_weight_panel(weight_day)
            week_score_daily.append(
                {
                    "date": current_date,
                    "positive_share": positive_share,
                    "gap_1_5": gap_1_5,
                }
            )
            week_weight_daily.append(
                {
                    "date": current_date,
                    "nonzero_weight_count": nonzero_weight_count,
                    "top1_weight": top1_weight,
                    "top2_weight_sum": top2_weight_sum,
                }
            )

        last_score_daily = week_score_daily[-1]
        last_weight_daily = week_weight_daily[-1]
        first_week_score_positive_share_mean = float(pd.DataFrame(week_score_daily)["positive_share"].mean())
        first_week_score_gap_1_5_mean = float(pd.DataFrame(week_score_daily)["gap_1_5"].mean())
        first_week_weight_count_mean = float(pd.DataFrame(week_weight_daily)["nonzero_weight_count"].mean())
        first_week_top1_weight_mean = float(pd.DataFrame(week_weight_daily)["top1_weight"].mean())
        first_week_top2_weight_sum_mean = float(pd.DataFrame(week_weight_daily)["top2_weight_sum"].mean())
        first_week_weight_drift = _derive_first_week_weight_drift(
            start_weight_count=first_nonzero_weight_count,
            end_weight_count=int(last_weight_daily["nonzero_weight_count"]),
            start_top1_weight=first_top1_weight,
            end_top1_weight=float(last_weight_daily["top1_weight"]),
        )
        first_week_score_followthrough = _derive_first_week_score_followthrough(
            start_positive_share=first_positive_share,
            end_positive_share=float(last_score_daily["positive_share"]),
            start_gap_1_5=first_gap_1_5,
            end_gap_1_5=float(last_score_daily["gap_1_5"]),
        )
        rows.append(
            {
                "month": str(month),
                "month_start_weight_count": f"count{first_nonzero_weight_count}",
                "month_start_signal_shape": _derive_signal_shape(first_nonzero_weight_count),
                "month_start_top1_weight": first_top1_weight,
                "month_start_top2_weight_sum": first_top2_weight_sum,
                "month_start_score_positive_share": first_positive_share,
                "month_start_score_gap_1_5": first_gap_1_5,
                "first_week_last_date": str(last_first_week_date.date()),
                "first_week_weight_count_mean": first_week_weight_count_mean,
                "first_week_top1_weight_mean": first_week_top1_weight_mean,
                "first_week_top2_weight_sum_mean": first_week_top2_weight_sum_mean,
                "first_week_score_positive_share_mean": first_week_score_positive_share_mean,
                "first_week_score_gap_1_5_mean": first_week_score_gap_1_5_mean,
                "first_week_weight_drift": first_week_weight_drift,
                "first_week_score_followthrough": first_week_score_followthrough,
                "first_week_combo": f"{first_week_score_followthrough}|{first_week_weight_drift}",
            }
        )
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _run_replay(
    *,
    python_executable: str,
    hybrid_panel_path: Path,
    start_date: str,
    end_date: str,
    output_dir: Path,
    experiment_tag: str,
    candidate_label: str,
) -> Path:
    cmd = [
        str(python_executable),
        str(EXTERNAL_REPLAY_SCRIPT),
        "--target-weight-panel-csv",
        str(hybrid_panel_path),
        "--data-source",
        "tq",
        "--benchmark",
        "000300.SH",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--rebalance-freq",
        "1d",
        "--rebalance-offset-mode",
        "single",
        "--transaction-cost-bps",
        "3",
        "--slippage-bps",
        "7",
        "--sell-tax-bps",
        "10",
        "--no-market-regime-filter",
        "--output-dir",
        str(output_dir),
        "--experiment-tag",
        experiment_tag,
        "--candidate-label",
        candidate_label,
    ]
    _run_command(cmd)
    return output_dir / experiment_tag


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except Exception:
        return "n/a"


def _format_num(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "n/a"


def _write_report(
    output_dir: Path,
    *,
    summary: dict[str, Any],
    compare_df: pd.DataFrame,
) -> None:
    lines = [
        "# Short Alpha Targeted Weak-Month Repair Review",
        "",
        f"- static_profile: `{summary['static_profile']}`",
        f"- trigger_mode: `{summary['trigger_mode']}`",
        f"- window_count: `{summary['window_count']}`",
        f"- targeted mean excess annual: `{_format_pct(summary['targeted_mean_excess_annual_return'])}`",
        f"- targeted mean excess Sharpe: `{_format_num(summary['targeted_mean_excess_sharpe'])}`",
        f"- static mean excess annual: `{_format_pct(summary['static_mean_excess_annual_return'])}`",
        f"- static mean excess Sharpe: `{_format_num(summary['static_mean_excess_sharpe'])}`",
        f"- delta mean excess annual: `{_format_pct(summary['delta_mean_excess_annual_return'])}`",
        f"- delta mean excess Sharpe: `{_format_num(summary['delta_mean_excess_sharpe'])}`",
        f"- annual wins: `{summary['wins_excess_annual_return']}/{summary['window_count']}`",
        f"- Sharpe wins: `{summary['wins_excess_sharpe']}/{summary['window_count']}`",
        "",
        "## Window Results",
    ]
    for _, row in compare_df.iterrows():
        lines.append(
            "- "
            f"`{row['window_key']}`: targeted `{_format_pct(row['targeted_excess_annual_return'])}` vs static `{_format_pct(row['static_excess_annual_return'])}` "
            f"(delta `{_format_pct(row['delta_excess_annual_return'])}`), mapping `{row['selected_plan_label']}`"
        )
    lines.extend(["", "## Direct Answer"])
    if float(summary["delta_mean_excess_annual_return"]) > 0.0:
        lines.append(
            "- A narrow weak-month repair plan now beats the static policy on average, so the next step should move from broad execution-policy search to a limited regime-scoped repair branch."
        )
    else:
        lines.append(
            "- Even after narrowing the mapping to weak-month regimes only, the static policy still holds up better on average, so the remaining repair work should move deeper into signal-to-weight or month-trigger design instead of simple regime swapping."
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root).resolve() / str(args.root_tag).strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    static_profile = str(args.static_profile).strip()

    weak_month_dir = _run_weak_month_review(args, output_dir)
    weak_df = pd.read_csv(weak_month_dir / "weak_month_review.csv")
    weak_df["month"] = weak_df["month"].astype(str)

    audit_roots = [Path(token.strip()).resolve() for token in str(args.audit_roots or "").split(",") if token.strip()]
    windows = [_discover_window(path) for path in audit_roots]
    monthly_by_window = {window.window_key: _load_window_audit_monthly(window) for window in windows}
    all_audit_monthly = pd.concat(monthly_by_window.values(), ignore_index=True)
    all_audit_monthly["month"] = all_audit_monthly["month"].astype(str)

    diagnostic_frames: list[pd.DataFrame] = []
    for window in windows:
        _, static_run_dir = _load_static_metrics(window, static_profile)
        signal_diag = _load_month_start_signal_diagnostics(static_run_dir)
        signal_diag.insert(0, "window_key", window.window_key)
        diagnostic_frames.append(signal_diag)
    month_start_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    month_start_diagnostics.to_csv(output_dir / "month_start_signal_diagnostics.csv", index=False, encoding="utf-8-sig")

    diagnostic_cols = [
        "month",
        "month_start_weight_count",
        "month_start_signal_shape",
        "month_start_top1_weight",
        "month_start_top2_weight_sum",
        "month_start_score_positive_share",
        "month_start_score_gap_1_5",
        "first_week_last_date",
        "first_week_weight_count_mean",
        "first_week_top1_weight_mean",
        "first_week_top2_weight_sum_mean",
        "first_week_score_positive_share_mean",
        "first_week_score_gap_1_5_mean",
        "first_week_weight_drift",
        "first_week_score_followthrough",
        "first_week_combo",
    ]
    weak_df = weak_df.merge(month_start_diagnostics[diagnostic_cols], on="month", how="left")
    all_audit_monthly = all_audit_monthly.merge(month_start_diagnostics[diagnostic_cols], on="month", how="left")
    weak_df = _apply_trigger_key(weak_df, trigger_mode=str(args.trigger_mode))
    all_audit_monthly = _apply_trigger_key(all_audit_monthly, trigger_mode=str(args.trigger_mode))

    compare_rows: list[dict[str, Any]] = []
    month_choice_exports: list[pd.DataFrame] = []
    plan_search_exports: list[pd.DataFrame] = []
    regime_export_rows: list[pd.DataFrame] = []

    for window in windows:
        test_months = sorted(monthly_by_window[window.window_key]["month"].astype(str).unique().tolist())
        train_audit_monthly = all_audit_monthly.loc[~all_audit_monthly["month"].isin(test_months)].copy()
        train_weak_df = weak_df.loc[~weak_df["month"].isin(test_months)].copy()
        if train_audit_monthly.empty or train_weak_df.empty:
            raise RuntimeError(f"Training split became empty for {window.window_key}")

        best_mapping, plan_search_df, regime_summary = _select_best_training_plan(
            train_audit_monthly,
            train_weak_df,
            static_profile=static_profile,
            trigger_mode=str(args.trigger_mode),
            min_regime_support=int(args.min_regime_support),
            min_regime_lift=float(args.min_regime_lift),
            max_regimes_considered=int(args.max_regimes_considered),
            max_targeted_regimes=int(args.max_targeted_regimes),
            max_profiles_per_regime=int(args.max_profiles_per_regime),
            selection_objective=str(args.selection_objective),
        )
        plan_search_df.insert(0, "test_window_key", window.window_key)
        plan_search_exports.append(plan_search_df)

        regime_summary.insert(0, "test_window_key", window.window_key)
        regime_export_rows.append(regime_summary)

        test_month_meta = (
            _apply_trigger_key(
                monthly_by_window[window.window_key].merge(month_start_diagnostics[diagnostic_cols], on="month", how="left"),
                trigger_mode=str(args.trigger_mode),
            )[
                [
                    "window_key",
                    "month",
                    "month_start_regime",
                    "month_start_weight_count",
                    "month_start_signal_shape",
                    "first_week_weight_drift",
                    "first_week_score_followthrough",
                    "first_week_combo",
                    "trigger_key",
                ]
            ]
            .drop_duplicates()
            .sort_values("month")
            .reset_index(drop=True)
        )
        test_month_meta["selected_profile"] = test_month_meta["trigger_key"].map(best_mapping).fillna(static_profile)
        test_month_meta.insert(1, "selected_plan_label", _mapping_label(best_mapping))
        month_choice_exports.append(test_month_meta.copy())

        static_metrics, static_run_dir = _load_static_metrics(window, static_profile)
        if best_mapping:
            hybrid_panel = _build_hybrid_target_weight_panel(window, test_month_meta)
            hybrid_panel_path = output_dir / f"hybrid_target_weight_panel_{window.window_key}.csv"
            hybrid_panel.to_csv(hybrid_panel_path, index_label="date", encoding="utf-8-sig")
            replay_run_dir = _run_replay(
                python_executable=str(args.python_executable),
                hybrid_panel_path=hybrid_panel_path,
                start_date=window.start_date,
                end_date=window.end_date,
                output_dir=output_dir,
                experiment_tag=f"targeted_replays/{window.window_key}",
                candidate_label=f"targeted_weak_month_repair_{window.window_key}",
            )
            targeted_metrics = _load_json(replay_run_dir / "metrics.json")
        else:
            replay_run_dir = static_run_dir
            targeted_metrics = dict(static_metrics)

        selected_train_plan = plan_search_df.iloc[0].to_dict()
        compare_rows.append(
            {
                "window_key": window.window_key,
                "selected_plan_label": _mapping_label(best_mapping),
                "selected_mapping_json": json.dumps(best_mapping, ensure_ascii=False, sort_keys=True),
                "train_mean_excess_return": float(selected_train_plan.get("mean_excess_return", 0.0) or 0.0),
                "train_monthly_robust_score": float(selected_train_plan.get("monthly_robust_score", 0.0) or 0.0),
                "targeted_excess_annual_return": float(targeted_metrics.get("excess_annual_return", 0.0) or 0.0),
                "targeted_excess_sharpe": float(targeted_metrics.get("excess_sharpe", 0.0) or 0.0),
                "targeted_avg_turnover": float(targeted_metrics.get("avg_turnover", 0.0) or 0.0),
                "static_profile": static_profile,
                "static_excess_annual_return": float(static_metrics.get("excess_annual_return", 0.0) or 0.0),
                "static_excess_sharpe": float(static_metrics.get("excess_sharpe", 0.0) or 0.0),
                "static_avg_turnover": float(static_metrics.get("avg_turnover", 0.0) or 0.0),
                "delta_excess_annual_return": float(targeted_metrics.get("excess_annual_return", 0.0) or 0.0) - float(static_metrics.get("excess_annual_return", 0.0) or 0.0),
                "delta_excess_sharpe": float(targeted_metrics.get("excess_sharpe", 0.0) or 0.0) - float(static_metrics.get("excess_sharpe", 0.0) or 0.0),
                "delta_avg_turnover": float(targeted_metrics.get("avg_turnover", 0.0) or 0.0) - float(static_metrics.get("avg_turnover", 0.0) or 0.0),
                "wins_excess_annual_return": bool(float(targeted_metrics.get("excess_annual_return", 0.0) or 0.0) > float(static_metrics.get("excess_annual_return", 0.0) or 0.0)),
                "wins_excess_sharpe": bool(float(targeted_metrics.get("excess_sharpe", 0.0) or 0.0) > float(static_metrics.get("excess_sharpe", 0.0) or 0.0)),
                "targeted_run_dir": str(replay_run_dir),
                "static_run_dir": str(static_run_dir),
            }
        )

    compare_df = pd.DataFrame(compare_rows).sort_values("window_key").reset_index(drop=True)
    compare_df.to_csv(output_dir / "window_compare.csv", index=False, encoding="utf-8-sig")
    pd.concat(month_choice_exports, ignore_index=True).to_csv(output_dir / "month_policy_choices.csv", index=False, encoding="utf-8-sig")
    pd.concat(plan_search_exports, ignore_index=True).to_csv(output_dir / "plan_search_summary.csv", index=False, encoding="utf-8-sig")
    if regime_export_rows:
        pd.concat(regime_export_rows, ignore_index=True).to_csv(output_dir / "training_regime_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "static_profile": static_profile,
        "window_count": int(len(compare_df)),
        "targeted_mean_excess_annual_return": float(compare_df["targeted_excess_annual_return"].mean()),
        "targeted_mean_excess_sharpe": float(compare_df["targeted_excess_sharpe"].mean()),
        "static_mean_excess_annual_return": float(compare_df["static_excess_annual_return"].mean()),
        "static_mean_excess_sharpe": float(compare_df["static_excess_sharpe"].mean()),
        "delta_mean_excess_annual_return": float(compare_df["delta_excess_annual_return"].mean()),
        "delta_mean_excess_sharpe": float(compare_df["delta_excess_sharpe"].mean()),
        "wins_excess_annual_return": int(compare_df["wins_excess_annual_return"].sum()),
        "wins_excess_sharpe": int(compare_df["wins_excess_sharpe"].sum()),
        "selection_objective": str(args.selection_objective),
        "trigger_mode": str(args.trigger_mode),
        "review_root": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(output_dir, summary=summary, compare_df=compare_df)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
