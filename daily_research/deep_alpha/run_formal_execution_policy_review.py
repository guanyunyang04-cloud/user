from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.deep_alpha.run_execution_policy_audit import _rank_tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
AUDIT_SCRIPT = PROJECT_ROOT / "daily_research" / "deep_alpha" / "run_execution_policy_audit.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same execution-policy audit across multiple formal window runs and "
            "aggregate a candidate's best profit-max execution policy."
        )
    )
    parser.add_argument("--run-dir", action="append", required=True, help="Formal window run directory. Repeat for multiple windows.")
    parser.add_argument("--candidate-name", required=True, help="User-facing label written into the review summary.")
    parser.add_argument("--universe", default="", help="Optional universe label such as liquid500 or liquid800.")
    parser.add_argument("--panel-scope", choices=["formal", "live", "auto"], default="formal")
    parser.add_argument("--profile-set", default="profit_max_v1")
    parser.add_argument(
        "--selection-objective",
        choices=["excess_annual_return", "excess_sharpe", "monthly_robust_score"],
        default="excess_annual_return",
    )
    parser.add_argument("--current-profile", default="", help="Optional current execution profile label. Empty means infer from run metrics.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _infer_window_label(run_dir: Path) -> str:
    matched = re.search(r"(\d{8}_\d{8})$", run_dir.name)
    if matched:
        return matched.group(1)
    metrics = _load_json(run_dir / "metrics.json")
    valid_start = str(metrics.get("valid_start", "") or "").replace("-", "")
    valid_end = str(metrics.get("valid_end", "") or "").replace("-", "")
    if len(valid_start) == 8 and len(valid_end) == 8:
        return f"{valid_start}_{valid_end}"
    return run_dir.name


def _current_profile_for_run(run_dir: Path, explicit_current_profile: str) -> str:
    if explicit_current_profile:
        return str(explicit_current_profile).strip()
    metrics = _load_json(run_dir / "metrics.json")
    return str(metrics.get("execution_alignment_profile", "") or "").strip()


def _run_single_audit(
    *,
    run_dir: Path,
    panel_scope: str,
    profile_set: str,
    selection_objective: str,
    python_executable: str,
    output_root: Path,
    experiment_tag: str,
) -> Path:
    cmd = [
        str(python_executable),
        str(AUDIT_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--panel-scope",
        panel_scope,
        "--profile-set",
        profile_set,
        "--selection-objective",
        selection_objective,
        "--output-root",
        str(output_root),
        "--experiment-tag",
        experiment_tag,
        "--python-executable",
        str(python_executable),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    return output_root / experiment_tag


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    run_dirs = [Path(item).resolve() for item in args.run_dir]
    output_root = Path(args.output_root).resolve()
    root_tag = args.experiment_tag.strip() or f"formal_execution_policy_review_{args.candidate_name}_{datetime.now():%Y%m%d_%H%M%S}"
    review_root = output_root / root_tag
    review_root.mkdir(parents=True, exist_ok=True)

    window_frames: list[pd.DataFrame] = []
    best_rows: list[dict[str, Any]] = []
    current_profiles: list[str] = []

    for run_dir in run_dirs:
        window_label = _infer_window_label(run_dir)
        audit_tag = f"{args.candidate_name}_{window_label}_{datetime.now():%Y%m%d_%H%M%S}"
        audit_root = _run_single_audit(
            run_dir=run_dir,
            panel_scope=args.panel_scope,
            profile_set=args.profile_set,
            selection_objective=args.selection_objective,
            python_executable=args.python_executable,
            output_root=output_root,
            experiment_tag=audit_tag,
        )
        summary_csv = audit_root / "execution_policy_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Execution policy audit summary missing: {summary_csv}")
        frame = pd.read_csv(summary_csv)
        if frame.empty:
            raise RuntimeError(f"Execution policy audit returned no rows for {run_dir}")
        frame["audit_root"] = str(audit_root)
        frame["window"] = window_label
        window_frames.append(frame)
        rows = frame.to_dict(orient="records")
        best_row = max(rows, key=lambda row: _rank_tuple(row, args.selection_objective))
        best_row["window"] = window_label
        best_rows.append(best_row)
        current_profiles.append(_current_profile_for_run(run_dir, args.current_profile))

    window_profile_summary = pd.concat(window_frames, ignore_index=True)
    window_profile_summary.to_csv(review_root / "window_profile_summary.csv", index=False, encoding="utf-8-sig")

    current_profile_set = sorted({item for item in current_profiles if item})
    current_profile_text = current_profile_set[0] if len(current_profile_set) == 1 else ",".join(current_profile_set)

    grouped_rows: list[dict[str, Any]] = []
    for profile_name, frame in window_profile_summary.groupby("profile_name", sort=False):
        grouped_rows.append(
            {
                "profile_name": str(profile_name),
                "mean_excess_annual_return": float(frame["excess_annual_return"].mean()),
                "mean_excess_sharpe": float(frame["excess_sharpe"].mean()),
                "mean_positive_month_ratio": float(frame["positive_month_ratio"].mean()),
                "mean_median_monthly_return": float(frame["median_monthly_return"].mean()),
                "best_window_count": int(frame["window"].nunique()),
                "current_profile_count": int(frame["is_current_profile"].sum()) if "is_current_profile" in frame.columns else 0,
                "window_win_count": int(sum(1 for row in best_rows if str(row.get("profile_name", "")) == str(profile_name))),
            }
        )

    profile_mean_summary = pd.DataFrame(grouped_rows).sort_values(
        by=[
            "mean_excess_annual_return",
            "mean_excess_sharpe",
            "mean_positive_month_ratio",
            "mean_median_monthly_return",
        ],
        ascending=[False, False, False, False],
    )
    profile_mean_summary.to_csv(review_root / "profile_mean_summary.csv", index=False, encoding="utf-8-sig")

    best_profile_row = profile_mean_summary.iloc[0].to_dict()
    current_profile_row = (
        profile_mean_summary.loc[profile_mean_summary["profile_name"].eq(current_profile_text)].iloc[0].to_dict()
        if current_profile_text and profile_mean_summary["profile_name"].astype(str).eq(current_profile_text).any()
        else {}
    )
    summary_payload = {
        "candidate_name": str(args.candidate_name),
        "universe": str(args.universe or ""),
        "profile_set": str(args.profile_set),
        "selection_objective": str(args.selection_objective),
        "current_profile": current_profile_text,
        "best_profile": str(best_profile_row.get("profile_name", "")),
        "best_mean_excess_annual_return": _safe_float(best_profile_row.get("mean_excess_annual_return")),
        "best_mean_excess_sharpe": _safe_float(best_profile_row.get("mean_excess_sharpe")),
        "window_win_count": int(best_profile_row.get("window_win_count", 0) or 0),
        "run_count": int(len(run_dirs)),
        "review_root": str(review_root),
    }
    if current_profile_row:
        summary_payload.update(
            {
                "current_mean_excess_annual_return": _safe_float(current_profile_row.get("mean_excess_annual_return")),
                "current_mean_excess_sharpe": _safe_float(current_profile_row.get("mean_excess_sharpe")),
                "delta_mean_excess_annual_return": _safe_float(best_profile_row.get("mean_excess_annual_return"))
                - _safe_float(current_profile_row.get("mean_excess_annual_return")),
                "delta_mean_excess_sharpe": _safe_float(best_profile_row.get("mean_excess_sharpe"))
                - _safe_float(current_profile_row.get("mean_excess_sharpe")),
            }
        )

    lines = [
        f"# {args.candidate_name} Formal Execution Policy Review",
        "",
        f"- universe: `{args.universe or 'unspecified'}`",
        f"- scope: `{args.candidate_name}` {len(run_dirs)} formal windows",
        f"- objective: `{args.selection_objective}`",
        f"- profile_set: `{args.profile_set}`",
        f"- best_profile: `{summary_payload['best_profile']}`",
        f"- best_mean_excess_annual_return: `{summary_payload['best_mean_excess_annual_return']:.2%}`",
        f"- best_mean_excess_sharpe: `{summary_payload['best_mean_excess_sharpe']:.3f}`",
        f"- window_win_count: `{summary_payload['window_win_count']}/{len(run_dirs)}`",
        f"- current_profile: `{current_profile_text or 'none'}`",
    ]
    if current_profile_row:
        lines.extend(
            [
                f"- current_mean_excess_annual_return: `{summary_payload['current_mean_excess_annual_return']:.2%}`",
                f"- current_mean_excess_sharpe: `{summary_payload['current_mean_excess_sharpe']:.3f}`",
                f"- delta_mean_excess_annual_return: `{summary_payload['delta_mean_excess_annual_return']:.2%}`",
                f"- delta_mean_excess_sharpe: `{summary_payload['delta_mean_excess_sharpe']:.3f}`",
            ]
        )
    (review_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (review_root / "leaderboard.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
