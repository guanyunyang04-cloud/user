from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from daily_research.baseline.cli_utils import parse_csv_list


DEFAULT_LABELS = (
    "trend_up_low_vol_ml25_none25_v250",
    "trend_up_low_vol_ml25_none20_v255",
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    group: str
    better: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a reusable head-to-head summary for the final advanced_ml shortlist from existing formal output directories."
    )
    parser.add_argument(
        "--run-dirs",
        required=True,
        help="Comma-separated formal output directories that contain strict_compare_summary.csv and run_config.json.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Exactly two shortlist labels to compare.",
    )
    parser.add_argument("--train-window-days", type=int, default=504)
    parser.add_argument("--retrain-every-days", type=int, default=21)
    parser.add_argument("--lgbm-n-estimators", type=int, default=520)
    parser.add_argument("--focus-state", default="", help="Optional override. Defaults to the first run_config focus_state.")
    parser.add_argument(
        "--weak-window-name",
        default="",
        help="Optional override. Defaults to the first weak window found in run_config windows.",
    )
    parser.add_argument("--output-dir", default="", help="Optional explicit output directory.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_weak_window_name(run_config: dict[str, Any]) -> str:
    windows = run_config.get("windows") or []
    for item in windows:
        name = str(item.get("name", "")).strip()
        if name.startswith("weak_window_"):
            return name
    for item in windows:
        name = str(item.get("name", "")).strip()
        if name and name != "recent_full":
            return name
    raise ValueError("Could not infer weak window name from run_config.json")


def _resolve_output_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    output_root = Path("daily_research") / "output"
    prefix = f"advanced_ml_shortlist_head_to_head_{datetime.now():%Y%m%d}_formal_r"
    existing = sorted(output_root.glob(f"{prefix}*"))
    next_index = 1
    for path in existing:
        suffix = path.name.replace(prefix, "", 1)
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)
    return output_root / f"{prefix}{next_index}"


def _normalise_numeric_column(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def _load_run_context(
    run_dir: Path,
    *,
    labels: list[str],
    train_window_days: int,
    retrain_every_days: int,
    lgbm_n_estimators: int,
    focus_state_override: str,
    weak_window_name_override: str,
) -> dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    strict_compare_path = run_dir / "strict_compare_summary.csv"
    run_config_path = run_dir / "run_config.json"
    if not strict_compare_path.exists():
        raise FileNotFoundError(strict_compare_path)
    if not run_config_path.exists():
        raise FileNotFoundError(run_config_path)

    run_config = _load_json(run_config_path)
    df = pd.read_csv(strict_compare_path)
    for column in ("ml_train_window_days", "ml_retrain_every_days", "lgbm_n_estimators"):
        _normalise_numeric_column(df, column)

    filtered = df[
        (df["label"].isin(labels))
        & (df["ml_train_window_days"] == train_window_days)
        & (df["ml_retrain_every_days"] == retrain_every_days)
        & (df["lgbm_n_estimators"] == lgbm_n_estimators)
    ].copy()
    if filtered.empty:
        raise ValueError(
            f"No shortlist rows found in {run_dir} for "
            f"{train_window_days}/{retrain_every_days}/{lgbm_n_estimators}"
        )

    label_counts = filtered["label"].value_counts().to_dict()
    missing = [label for label in labels if label_counts.get(label, 0) != 1]
    if missing:
        raise ValueError(f"Run {run_dir} is missing unique rows for labels: {missing}")

    label_rank = {label: index for index, label in enumerate(labels)}
    filtered["label_rank"] = filtered["label"].map(label_rank)
    filtered = filtered.sort_values("label_rank").drop(columns=["label_rank"]).reset_index(drop=True)

    history_window = run_config.get("history_window") or {}
    focus_state = focus_state_override or str(run_config.get("focus_state", "")).strip()
    weak_window_name = weak_window_name_override or _detect_weak_window_name(run_config)
    if not focus_state:
        raise ValueError(f"Could not infer focus_state from {run_config_path}")

    recent_window = next(
        (item for item in run_config.get("windows", []) if str(item.get("name", "")).strip() == "recent_full"),
        {},
    )
    weak_window = next(
        (item for item in run_config.get("windows", []) if str(item.get("name", "")).strip() == weak_window_name),
        {},
    )
    return {
        "run_dir": run_dir,
        "run_name": run_dir.name,
        "run_config": run_config,
        "rows": filtered,
        "focus_state": focus_state,
        "weak_window_name": weak_window_name,
        "latest_data_date": str(run_config.get("latest_data_date", "")),
        "history_mode": str(history_window.get("mode", "")),
        "history_start": str(history_window.get("effective_start_date", "")),
        "history_end": str(history_window.get("end_date", "")),
        "recent_end": str(recent_window.get("end", "")),
        "weak_start": str(weak_window.get("start", "")),
        "weak_end": str(weak_window.get("end", "")),
    }


def _build_metric_specs(weak_window_name: str, focus_state: str) -> list[MetricSpec]:
    return [
        MetricSpec("full_excess_sharpe", "offense", "higher"),
        MetricSpec("recent_full_excess_sharpe", "offense", "higher"),
        MetricSpec(f"{weak_window_name}_excess_sharpe", "defense", "higher"),
        MetricSpec(f"{focus_state}_{weak_window_name}_excess_sharpe", "defense", "higher"),
        MetricSpec("full_excess_max_drawdown", "defense", "higher"),
        MetricSpec("recent_full_excess_max_drawdown", "defense", "higher"),
        MetricSpec("full_avg_turnover", "cost", "lower"),
    ]


def _safe_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _pick_winner(value_a: float, value_b: float, better: str) -> str:
    if pd.isna(value_a) or pd.isna(value_b):
        return "unknown"
    if better == "higher":
        if value_a > value_b:
            return "label_a"
        if value_b > value_a:
            return "label_b"
    elif better == "lower":
        if value_a < value_b:
            return "label_a"
        if value_b < value_a:
            return "label_b"
    else:
        raise ValueError(f"Unsupported better mode: {better}")
    return "tie"


def _build_run_overview(contexts: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for context in contexts:
        rows.append(
            {
                "run_name": context["run_name"],
                "run_dir": str(context["run_dir"]),
                "latest_data_date": context["latest_data_date"],
                "history_mode": context["history_mode"],
                "history_start": context["history_start"],
                "history_end": context["history_end"],
                "recent_end": context["recent_end"],
                "weak_window_name": context["weak_window_name"],
                "weak_start": context["weak_start"],
                "weak_end": context["weak_end"],
                "focus_state": context["focus_state"],
            }
        )
    return pd.DataFrame(rows)


def _build_selected_rows(contexts: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for context in contexts:
        frame = context["rows"].copy()
        frame.insert(0, "run_name", context["run_name"])
        frame.insert(1, "run_dir", str(context["run_dir"]))
        frame.insert(2, "history_start", context["history_start"])
        frame.insert(3, "history_end", context["history_end"])
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _build_metric_comparison(
    contexts: list[dict[str, Any]],
    *,
    labels: list[str],
    metric_specs: list[MetricSpec],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_a = labels[0]
    label_b = labels[1]
    for context in contexts:
        indexed = context["rows"].set_index("label")
        for spec in metric_specs:
            if spec.name not in indexed.columns:
                raise KeyError(f"Metric {spec.name!r} not found in {context['run_dir']}")
            value_a = _safe_float(indexed.at[label_a, spec.name])
            value_b = _safe_float(indexed.at[label_b, spec.name])
            winner_key = _pick_winner(value_a, value_b, spec.better)
            if winner_key == "label_a":
                winner_label = label_a
            elif winner_key == "label_b":
                winner_label = label_b
            else:
                winner_label = winner_key
            rows.append(
                {
                    "run_name": context["run_name"],
                    "run_dir": str(context["run_dir"]),
                    "history_start": context["history_start"],
                    "history_end": context["history_end"],
                    "metric": spec.name,
                    "metric_group": spec.group,
                    "better_when": spec.better,
                    "label_a": label_a,
                    "label_b": label_b,
                    "label_a_value": value_a,
                    "label_b_value": value_b,
                    "label_b_minus_label_a": value_b - value_a,
                    "winner": winner_label,
                }
            )
    return pd.DataFrame(rows)


def _majority_winner(metric_df: pd.DataFrame, *, group: str) -> str:
    subset = metric_df[(metric_df["metric_group"] == group) & (~metric_df["winner"].isin(["tie", "unknown"]))]
    if subset.empty:
        return "unknown"
    counts = subset["winner"].value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "tie"
    return str(counts.index[0])


def _build_group_winners(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run_df in metric_df.groupby("run_name", sort=False):
        for group in ("offense", "defense", "cost"):
            rows.append(
                {
                    "run_name": run_name,
                    "metric_group": group,
                    "winner": _majority_winner(run_df, group=group),
                }
            )
    return pd.DataFrame(rows)


def _resolve_single_upgrade_winner(group_winners: pd.DataFrame) -> str:
    primary = group_winners[group_winners["metric_group"].isin(["offense", "defense"])]
    winners = set(primary["winner"].tolist())
    winners.discard("tie")
    winners.discard("unknown")
    if len(winners) != 1:
        return "none"
    if "tie" in set(primary["winner"].tolist()) or "unknown" in set(primary["winner"].tolist()):
        return "none"
    return next(iter(winners))


def _build_verdict(
    *,
    contexts: list[dict[str, Any]],
    labels: list[str],
    metric_df: pd.DataFrame,
    group_winners: pd.DataFrame,
    train_window_days: int,
    retrain_every_days: int,
    lgbm_n_estimators: int,
) -> dict[str, Any]:
    label_a = labels[0]
    label_b = labels[1]

    metric_consistency: list[dict[str, Any]] = []
    for metric, metric_rows in metric_df.groupby("metric", sort=False):
        unique_winners = {item for item in metric_rows["winner"].tolist() if item not in {"unknown"}}
        consistent_winner = next(iter(unique_winners)) if len(unique_winners) == 1 else "mixed"
        metric_consistency.append(
            {
                "metric": metric,
                "consistent_winner": consistent_winner,
            }
        )

    single_upgrade_winner = _resolve_single_upgrade_winner(group_winners)
    if single_upgrade_winner == label_a:
        direct_answer = f"{label_a} is the single upgrade winner."
        recommended_action = f"Promote {label_a} under the current formal rules."
    elif single_upgrade_winner == label_b:
        direct_answer = f"{label_b} is the single upgrade winner."
        recommended_action = f"Promote {label_b} under the current formal rules."
    else:
        direct_answer = "No single upgrade winner."
        recommended_action = (
            "Keep the current default unchanged and preserve both shortlist candidates until an explicit tie-break rule "
            "or a new formal comparator can align offense and defense into one winner."
        )

    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_window_days": train_window_days,
            "retrain_every_days": retrain_every_days,
            "lgbm_n_estimators": lgbm_n_estimators,
        },
        "labels": labels,
        "source_run_dirs": [str(context["run_dir"]) for context in contexts],
        "metric_consistency": metric_consistency,
        "group_winners": group_winners.to_dict(orient="records"),
        "single_upgrade_winner": single_upgrade_winner,
        "direct_answer": direct_answer,
        "recommended_action": recommended_action,
    }


def _format_num(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    return f"{float(value):.3f}"


def _render_summary(
    *,
    contexts: list[dict[str, Any]],
    labels: list[str],
    metric_df: pd.DataFrame,
    group_winners: pd.DataFrame,
    verdict: dict[str, Any],
) -> str:
    label_a = labels[0]
    label_b = labels[1]
    lines: list[str] = []
    config = verdict["config"]
    lines.append("# Advanced ML Shortlist Head-to-Head")
    lines.append("")
    lines.append(f"- generated_at: {verdict['created_at']}")
    lines.append(
        f"- exact_config: train_window_days={config['train_window_days']}, "
        f"retrain_every_days={config['retrain_every_days']}, lgbm_n_estimators={config['lgbm_n_estimators']}"
    )
    lines.append(f"- label_a: {label_a}")
    lines.append(f"- label_b: {label_b}")
    lines.append("")
    lines.append("## Run snapshots")
    for context in contexts:
        indexed = context["rows"].set_index("label")
        weak_metric = f"{context['weak_window_name']}_excess_sharpe"
        focus_weak_metric = f"{context['focus_state']}_{context['weak_window_name']}_excess_sharpe"
        lines.append(f"### {context['run_name']}")
        lines.append(
            f"- history_window: {context['history_start']} -> {context['history_end']} "
            f"(mode={context['history_mode']}, latest_data_date={context['latest_data_date']})"
        )
        lines.append(
            f"- {label_a}: full_excess_sharpe={_format_num(indexed.at[label_a, 'full_excess_sharpe'])}, "
            f"recent_full_excess_sharpe={_format_num(indexed.at[label_a, 'recent_full_excess_sharpe'])}, "
            f"{weak_metric}={_format_num(indexed.at[label_a, weak_metric])}, "
            f"{focus_weak_metric}={_format_num(indexed.at[label_a, focus_weak_metric])}, "
            f"full_excess_max_drawdown={_format_num(indexed.at[label_a, 'full_excess_max_drawdown'])}, "
            f"full_avg_turnover={_format_num(indexed.at[label_a, 'full_avg_turnover'])}"
        )
        lines.append(
            f"- {label_b}: full_excess_sharpe={_format_num(indexed.at[label_b, 'full_excess_sharpe'])}, "
            f"recent_full_excess_sharpe={_format_num(indexed.at[label_b, 'recent_full_excess_sharpe'])}, "
            f"{weak_metric}={_format_num(indexed.at[label_b, weak_metric])}, "
            f"{focus_weak_metric}={_format_num(indexed.at[label_b, focus_weak_metric])}, "
            f"full_excess_max_drawdown={_format_num(indexed.at[label_b, 'full_excess_max_drawdown'])}, "
            f"full_avg_turnover={_format_num(indexed.at[label_b, 'full_avg_turnover'])}"
        )
        group_subset = group_winners[group_winners["run_name"] == context["run_name"]]
        offense = group_subset[group_subset["metric_group"] == "offense"]["winner"].iloc[0]
        defense = group_subset[group_subset["metric_group"] == "defense"]["winner"].iloc[0]
        lines.append(f"- group_winners: offense={offense}, defense={defense}")
        lines.append("")

    lines.append("## Metric consistency")
    for metric, metric_rows in metric_df.groupby("metric", sort=False):
        winners = metric_rows["winner"].tolist()
        unique_winners = {item for item in winners if item != "unknown"}
        if len(unique_winners) == 1:
            winner = next(iter(unique_winners))
            lines.append(f"- {metric}: {winner} wins in every compared formal run.")
        else:
            lines.append(f"- {metric}: mixed winners across compared formal runs.")
    lines.append("")
    lines.append("## Direct answer")
    lines.append(f"- {verdict['direct_answer']}")
    lines.append(f"- Recommended action: {verdict['recommended_action']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dirs = [Path(item) for item in parse_csv_list(args.run_dirs, normalizer=None)]
    labels = parse_csv_list(args.labels, normalizer=None)
    if len(labels) != 2:
        raise ValueError("--labels must contain exactly two labels.")
    if not run_dirs:
        raise ValueError("--run-dirs must not be empty.")

    contexts = [
        _load_run_context(
            run_dir,
            labels=labels,
            train_window_days=args.train_window_days,
            retrain_every_days=args.retrain_every_days,
            lgbm_n_estimators=args.lgbm_n_estimators,
            focus_state_override=args.focus_state.strip(),
            weak_window_name_override=args.weak_window_name.strip(),
        )
        for run_dir in run_dirs
    ]

    focus_state = contexts[0]["focus_state"]
    weak_window_name = contexts[0]["weak_window_name"]
    metric_specs = _build_metric_specs(weak_window_name, focus_state)
    run_overview_df = _build_run_overview(contexts)
    selected_rows_df = _build_selected_rows(contexts)
    metric_df = _build_metric_comparison(contexts, labels=labels, metric_specs=metric_specs)
    group_winners_df = _build_group_winners(metric_df)
    verdict = _build_verdict(
        contexts=contexts,
        labels=labels,
        metric_df=metric_df,
        group_winners=group_winners_df,
        train_window_days=args.train_window_days,
        retrain_every_days=args.retrain_every_days,
        lgbm_n_estimators=args.lgbm_n_estimators,
    )
    summary_text = _render_summary(
        contexts=contexts,
        labels=labels,
        metric_df=metric_df,
        group_winners=group_winners_df,
        verdict=verdict,
    )

    output_dir = _resolve_output_dir(args.output_dir.strip())
    output_dir.mkdir(parents=True, exist_ok=True)
    run_overview_df.to_csv(output_dir / "run_overview.csv", index=False, encoding="utf-8-sig")
    selected_rows_df.to_csv(output_dir / "selected_rows.csv", index=False, encoding="utf-8-sig")
    metric_df.to_csv(output_dir / "metric_comparison.csv", index=False, encoding="utf-8-sig")
    group_winners_df.to_csv(output_dir / "group_winners.csv", index=False, encoding="utf-8-sig")
    (output_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(summary_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "single_upgrade_winner": verdict["single_upgrade_winner"],
                "direct_answer": verdict["direct_answer"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
