from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_LEFT_PROFILE = "legacy_v7"
DEFAULT_LEFT_LABEL = "trend_up_low_vol_ml25_none20_v255"
DEFAULT_RIGHT_PROFILE = "expanded_v24"
DEFAULT_RIGHT_LABEL = "trend_up_low_vol_ml25_none25_v250"
DEFAULT_FOCUS_STATE = "trend_up_low_vol"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a direct stack-vs-stack A/B verdict from market feature profile compare outputs."
    )
    parser.add_argument("--profile-results-csv", required=True)
    parser.add_argument("--left-profile", default=DEFAULT_LEFT_PROFILE)
    parser.add_argument("--left-label", default=DEFAULT_LEFT_LABEL)
    parser.add_argument("--right-profile", default=DEFAULT_RIGHT_PROFILE)
    parser.add_argument("--right-label", default=DEFAULT_RIGHT_LABEL)
    parser.add_argument("--focus-state", default=DEFAULT_FOCUS_STATE)
    parser.add_argument("--weak-window", default="")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _infer_weak_window(df: pd.DataFrame) -> str:
    prefixes = sorted(
        {
            col[: -len("_excess_sharpe")]
            for col in df.columns
            if col.startswith("weak_window_") and col.endswith("_excess_sharpe")
        }
    )
    if not prefixes:
        raise ValueError("Unable to infer weak_window_* columns from profile results CSV.")
    return prefixes[0]


def _row_for(df: pd.DataFrame, *, label: str, profile: str) -> pd.Series:
    matched = df.loc[df["label"].eq(label) & df["market_feature_profile"].eq(profile)]
    if matched.empty:
        raise ValueError(f"Missing row for label={label}, market_feature_profile={profile}")
    return matched.iloc[0]


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _format_pct(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.2%}"


def _format_num(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.3f}"


def _metric_snapshot(row: pd.Series, weak_window: str, focus_state: str) -> dict[str, float | None]:
    return {
        "full_total_return": _to_float(row.get("full_total_return")),
        "full_annual_return": _to_float(row.get("full_annual_return")),
        "full_excess_total_return": _to_float(row.get("full_excess_total_return")),
        "full_excess_annual_return": _to_float(row.get("full_excess_annual_return")),
        "full_excess_sharpe": _to_float(row.get("full_excess_sharpe")),
        "recent_full_total_return": _to_float(row.get("recent_full_total_return")),
        "recent_full_annual_return": _to_float(row.get("recent_full_annual_return")),
        "recent_full_excess_total_return": _to_float(row.get("recent_full_excess_total_return")),
        "recent_full_excess_annual_return": _to_float(row.get("recent_full_excess_annual_return")),
        "recent_full_excess_sharpe": _to_float(row.get("recent_full_excess_sharpe")),
        f"{weak_window}_total_return": _to_float(row.get(f"{weak_window}_total_return")),
        f"{weak_window}_annual_return": _to_float(row.get(f"{weak_window}_annual_return")),
        f"{weak_window}_excess_total_return": _to_float(row.get(f"{weak_window}_excess_total_return")),
        f"{weak_window}_excess_annual_return": _to_float(row.get(f"{weak_window}_excess_annual_return")),
        f"{weak_window}_excess_sharpe": _to_float(row.get(f"{weak_window}_excess_sharpe")),
        f"{focus_state}_{weak_window}_excess_sharpe": _to_float(row.get(f"{focus_state}_{weak_window}_excess_sharpe")),
    }


def _delta_map(left: dict[str, float | None], right: dict[str, float | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in left:
        left_value = left.get(key)
        right_value = right.get(key)
        out[key] = None if left_value is None or right_value is None else right_value - left_value
    return out


def _make_metric_rows(name: str, metrics: dict[str, float | None], weak_window: str, focus_state: str) -> list[dict[str, Any]]:
    ordered_keys = [
        "full_total_return",
        "full_annual_return",
        "full_excess_total_return",
        "full_excess_annual_return",
        "full_excess_sharpe",
        "recent_full_total_return",
        "recent_full_annual_return",
        "recent_full_excess_total_return",
        "recent_full_excess_annual_return",
        "recent_full_excess_sharpe",
        f"{weak_window}_total_return",
        f"{weak_window}_annual_return",
        f"{weak_window}_excess_total_return",
        f"{weak_window}_excess_annual_return",
        f"{weak_window}_excess_sharpe",
        f"{focus_state}_{weak_window}_excess_sharpe",
    ]
    return [{"series": name, "metric": key, "value": metrics.get(key)} for key in ordered_keys]


def _render_summary(
    *,
    left_name: str,
    right_name: str,
    weak_window: str,
    focus_state: str,
    direct_left: dict[str, float | None],
    direct_right: dict[str, float | None],
    direct_delta: dict[str, float | None],
    offense_profile_delta: dict[str, float | None] | None,
    live_profile_delta: dict[str, float | None] | None,
    left_profile_internal_delta: dict[str, float | None] | None,
    right_profile_internal_delta: dict[str, float | None] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Market Feature Stack A/B")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- left_stack: {left_name}")
    lines.append(f"- right_stack: {right_name}")
    lines.append(f"- weak_window: {weak_window}")
    lines.append(f"- focus_state: {focus_state}")
    lines.append("")

    lines.append("## Direct stack comparison")
    lines.append(
        f"- {left_name}: full annual {_format_pct(direct_left['full_annual_return'])}, "
        f"full excess annual {_format_pct(direct_left['full_excess_annual_return'])}, "
        f"full excess Sharpe {_format_num(direct_left['full_excess_sharpe'])}, "
        f"{weak_window} excess Sharpe {_format_num(direct_left[f'{weak_window}_excess_sharpe'])}, "
        f"{focus_state}_{weak_window} excess Sharpe {_format_num(direct_left[f'{focus_state}_{weak_window}_excess_sharpe'])}"
    )
    lines.append(
        f"- {right_name}: full annual {_format_pct(direct_right['full_annual_return'])}, "
        f"full excess annual {_format_pct(direct_right['full_excess_annual_return'])}, "
        f"full excess Sharpe {_format_num(direct_right['full_excess_sharpe'])}, "
        f"{weak_window} excess Sharpe {_format_num(direct_right[f'{weak_window}_excess_sharpe'])}, "
        f"{focus_state}_{weak_window} excess Sharpe {_format_num(direct_right[f'{focus_state}_{weak_window}_excess_sharpe'])}"
    )
    lines.append("")

    lines.append("## Right minus left")
    lines.append(
        f"- full annual {_format_pct(direct_delta['full_annual_return'])}, "
        f"full excess annual {_format_pct(direct_delta['full_excess_annual_return'])}, "
        f"full excess Sharpe {_format_num(direct_delta['full_excess_sharpe'])}"
    )
    lines.append(
        f"- recent_full annual {_format_pct(direct_delta['recent_full_annual_return'])}, "
        f"recent_full excess annual {_format_pct(direct_delta['recent_full_excess_annual_return'])}, "
        f"recent_full excess Sharpe {_format_num(direct_delta['recent_full_excess_sharpe'])}"
    )
    lines.append(
        f"- {weak_window} annual {_format_pct(direct_delta[f'{weak_window}_annual_return'])}, "
        f"{weak_window} excess annual {_format_pct(direct_delta[f'{weak_window}_excess_annual_return'])}, "
        f"{weak_window} excess Sharpe {_format_num(direct_delta[f'{weak_window}_excess_sharpe'])}"
    )
    lines.append(
        f"- {focus_state}_{weak_window} excess Sharpe {_format_num(direct_delta[f'{focus_state}_{weak_window}_excess_sharpe'])}"
    )
    lines.append("")

    lines.append("## Decomposition controls")
    if offense_profile_delta is not None:
        lines.append(
            f"- offense row profile-only change (`{right_name.split(' | ')[0]}` vs `{left_name.split(' | ')[0]}` on `v255`): "
            f"full excess Sharpe {_format_num(offense_profile_delta['full_excess_sharpe'])}, "
            f"weak excess Sharpe {_format_num(offense_profile_delta[f'{weak_window}_excess_sharpe'])}"
        )
    if live_profile_delta is not None:
        lines.append(
            f"- live row profile-only change (`{right_name.split(' | ')[0]}` vs `{left_name.split(' | ')[0]}` on `v250`): "
            f"full excess Sharpe {_format_num(live_profile_delta['full_excess_sharpe'])}, "
            f"weak excess Sharpe {_format_num(live_profile_delta[f'{weak_window}_excess_sharpe'])}"
        )
    if left_profile_internal_delta is not None:
        lines.append(
            f"- legacy profile internal switch (`v250 - v255` under `{left_name.split(' | ')[0]}`): "
            f"full excess Sharpe {_format_num(left_profile_internal_delta['full_excess_sharpe'])}, "
            f"weak excess Sharpe {_format_num(left_profile_internal_delta[f'{weak_window}_excess_sharpe'])}"
        )
    if right_profile_internal_delta is not None:
        lines.append(
            f"- expanded profile internal switch (`v250 - v255` under `{right_name.split(' | ')[0]}`): "
            f"full excess Sharpe {_format_num(right_profile_internal_delta['full_excess_sharpe'])}, "
            f"weak excess Sharpe {_format_num(right_profile_internal_delta[f'{weak_window}_excess_sharpe'])}"
        )
    lines.append("")

    full_delta = direct_delta["full_excess_sharpe"]
    weak_delta = direct_delta[f"{weak_window}_excess_sharpe"]
    focus_delta = direct_delta[f"{focus_state}_{weak_window}_excess_sharpe"]
    recent_delta = direct_delta["recent_full_excess_sharpe"]
    if (full_delta or 0.0) < 0 and min((weak_delta or 0.0), (focus_delta or 0.0), (recent_delta or 0.0)) > 0:
        verdict = (
            "当前 `expanded_v24 + v250` 不是单纯变强，也不是单纯变弱；它牺牲了旧 `legacy_v7 + v255` 的 full 端进攻上沿，"
            "换来了 recent / weak / focus-weak 三层更强的 live 稳健性。"
        )
    elif min((full_delta or 0.0), (weak_delta or 0.0), (focus_delta or 0.0)) > 0:
        verdict = "当前 `expanded_v24 + v250` 在 full 与弱窗口两侧都更强，可视为整体升级。"
    elif max((full_delta or 0.0), (weak_delta or 0.0), (focus_delta or 0.0)) < 0:
        verdict = "当前 `expanded_v24 + v250` 在 full 与弱窗口两侧都更弱，可视为整体变弱。"
    else:
        verdict = "两套栈呈现结构性分工，不能简单归纳成单向升级或单向变弱。"
    lines.append("## Verdict")
    lines.append(f"- {verdict}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    profile_results_csv = Path(args.profile_results_csv).resolve()
    if not profile_results_csv.exists():
        raise FileNotFoundError(f"profile_results.csv not found: {profile_results_csv}")

    df = pd.read_csv(profile_results_csv, encoding="utf-8-sig")
    weak_window = str(args.weak_window or _infer_weak_window(df))
    focus_state = str(args.focus_state)

    left_row = _row_for(df, label=args.left_label, profile=args.left_profile)
    right_row = _row_for(df, label=args.right_label, profile=args.right_profile)
    left_name = f"{args.left_profile} | {args.left_label}"
    right_name = f"{args.right_profile} | {args.right_label}"

    direct_left = _metric_snapshot(left_row, weak_window, focus_state)
    direct_right = _metric_snapshot(right_row, weak_window, focus_state)
    direct_delta = _delta_map(direct_left, direct_right)

    offense_profile_delta = None
    live_profile_delta = None
    left_profile_internal_delta = None
    right_profile_internal_delta = None

    try:
        offense_right_row = _row_for(df, label=args.left_label, profile=args.right_profile)
        offense_profile_delta = _delta_map(
            _metric_snapshot(left_row, weak_window, focus_state),
            _metric_snapshot(offense_right_row, weak_window, focus_state),
        )
    except ValueError:
        offense_profile_delta = None

    try:
        live_left_row = _row_for(df, label=args.right_label, profile=args.left_profile)
        live_profile_delta = _delta_map(
            _metric_snapshot(live_left_row, weak_window, focus_state),
            _metric_snapshot(right_row, weak_window, focus_state),
        )
    except ValueError:
        live_profile_delta = None

    try:
        live_left_row = _row_for(df, label=args.right_label, profile=args.left_profile)
        left_profile_internal_delta = _delta_map(
            _metric_snapshot(left_row, weak_window, focus_state),
            _metric_snapshot(live_left_row, weak_window, focus_state),
        )
    except ValueError:
        left_profile_internal_delta = None

    try:
        offense_right_row = _row_for(df, label=args.left_label, profile=args.right_profile)
        right_profile_internal_delta = _delta_map(
            _metric_snapshot(offense_right_row, weak_window, focus_state),
            _metric_snapshot(right_row, weak_window, focus_state),
        )
    except ValueError:
        right_profile_internal_delta = None

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else profile_results_csv.parent / "stack_ab"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, Any]] = []
    metrics_rows.extend(_make_metric_rows(left_name, direct_left, weak_window, focus_state))
    metrics_rows.extend(_make_metric_rows(right_name, direct_right, weak_window, focus_state))
    metrics_rows.extend(_make_metric_rows(f"{right_name} - {left_name}", direct_delta, weak_window, focus_state))
    pd.DataFrame(metrics_rows).to_csv(output_dir / "stack_metrics.csv", index=False, encoding="utf-8-sig")

    summary_text = _render_summary(
        left_name=left_name,
        right_name=right_name,
        weak_window=weak_window,
        focus_state=focus_state,
        direct_left=direct_left,
        direct_right=direct_right,
        direct_delta=direct_delta,
        offense_profile_delta=offense_profile_delta,
        live_profile_delta=live_profile_delta,
        left_profile_internal_delta=left_profile_internal_delta,
        right_profile_internal_delta=right_profile_internal_delta,
    )
    (output_dir / "summary.md").write_text(summary_text, encoding="utf-8")

    verdict = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile_results_csv": str(profile_results_csv),
        "left_stack": {"profile": args.left_profile, "label": args.left_label, "metrics": direct_left},
        "right_stack": {"profile": args.right_profile, "label": args.right_label, "metrics": direct_right},
        "right_minus_left": direct_delta,
        "offense_profile_delta": offense_profile_delta,
        "live_profile_delta": live_profile_delta,
        "left_profile_internal_delta": left_profile_internal_delta,
        "right_profile_internal_delta": right_profile_internal_delta,
        "weak_window": weak_window,
        "focus_state": focus_state,
    }
    (output_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
