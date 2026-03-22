from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = WORKSPACE_ROOT / "daily_research" / "output"


@dataclass
class HealthThresholds:
    min_overall_excess_sharpe: float = 0.80
    min_recent_full_excess_sharpe: float = 0.50
    max_allowed_excess_drawdown: float = -0.35


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_by_mtime(pattern: str) -> Path:
    matches = list(OUTPUT_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No output matched pattern: {pattern}")
    return max(matches, key=lambda item: item.stat().st_mtime)


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def holdout_sort_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("holdout_start", "")), str(item.get("holdout_end", "")))


def summarize_holdout(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if "holdout_start" not in data or "holdout_end" not in data:
        stem = path.stem.replace("metrics_holdout_", "")
        start, end = stem.split("_", 1)
        data["holdout_start"] = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
        data["holdout_end"] = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
    data["source_file"] = str(path.relative_to(WORKSPACE_ROOT))
    return data


def pick_best_pool(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (parse_float(row.get("excess_sharpe")), parse_float(row.get("excess_total_return"))))


def build_report(
    formal_dir: Path,
    pool_compare_csv: Path,
    thresholds: HealthThresholds,
) -> dict[str, Any]:
    overall = load_json(formal_dir / "metrics.json")
    holdouts = [summarize_holdout(path) for path in sorted(formal_dir.glob("metrics_holdout_*.json"))]
    holdouts_sorted = sorted(holdouts, key=holdout_sort_key)
    latest_window = max(holdouts_sorted, key=holdout_sort_key) if holdouts_sorted else {}
    recent_full = {}
    if holdouts_sorted:
        latest_end = max(str(item.get("holdout_end", "")) for item in holdouts_sorted)
        same_end = [item for item in holdouts_sorted if str(item.get("holdout_end", "")) == latest_end]
        if same_end:
            recent_full = min(same_end, key=lambda item: str(item.get("holdout_start", "")))

    pool_rows = load_csv_rows(pool_compare_csv)
    best_pool = pick_best_pool(pool_rows)
    liquid500_row = next((row for row in pool_rows if row.get("pool") == "liquid500"), None)

    issues: list[str] = []
    overall_excess_sharpe = parse_float(overall.get("excess_sharpe"))
    overall_excess_drawdown = parse_float(overall.get("excess_max_drawdown"))
    latest_excess_sharpe = parse_float(latest_window.get("excess_sharpe"))
    latest_excess_return = parse_float(latest_window.get("excess_total_return"))
    recent_full_excess_sharpe = parse_float(recent_full.get("excess_sharpe"))

    if overall_excess_sharpe < thresholds.min_overall_excess_sharpe:
        issues.append("overall_excess_sharpe_below_threshold")
    if overall_excess_drawdown <= thresholds.max_allowed_excess_drawdown:
        issues.append("overall_excess_drawdown_too_deep")
    if recent_full and recent_full_excess_sharpe < thresholds.min_recent_full_excess_sharpe:
        issues.append("recent_full_holdout_excess_sharpe_below_threshold")
    if latest_window and latest_excess_return <= 0:
        issues.append("latest_window_negative_excess_return")
    if latest_window and latest_excess_sharpe <= 0:
        issues.append("latest_window_negative_excess_sharpe")

    verdict = "keep_current_execution_line"
    decision = "继续冻结当前执行主线，不重启执行方向研究。"
    if latest_window and latest_excess_return <= 0 and latest_excess_sharpe <= 0:
        if overall_excess_sharpe >= thresholds.min_overall_excess_sharpe:
            verdict = "restart_parallel_execution_research_keep_current_frozen"
            decision = "长期主线仍有效，但近期窗口明显转弱，重启执行方向研究，同时继续冻结当前执行主线。"
        else:
            verdict = "restart_execution_research"
            decision = "执行主线整体与近期都不理想，需要重启执行方向研究。"
    elif issues:
        verdict = "restart_parallel_execution_research_keep_current_frozen"
        decision = "执行主线存在明显薄弱点，建议在保持当前主线冻结的前提下，重启执行方向研究。"

    pool_comment = ""
    if best_pool and liquid500_row:
        if best_pool.get("pool") == "liquid500":
            pool_comment = "当前 `liquid500` 仍是三档高流动性池里超额 Sharpe 最优的执行池，暂不先改股票池。"
        else:
            pool_comment = (
                f"当前最优执行池已变成 `{best_pool.get('pool', '')}`，应把股票池本身纳入执行方向研究。"
            )

    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "formal_dir": str(formal_dir.relative_to(WORKSPACE_ROOT)),
        "pool_compare_csv": str(pool_compare_csv.relative_to(WORKSPACE_ROOT)),
        "thresholds": {
            "min_overall_excess_sharpe": thresholds.min_overall_excess_sharpe,
            "min_recent_full_excess_sharpe": thresholds.min_recent_full_excess_sharpe,
            "max_allowed_excess_drawdown": thresholds.max_allowed_excess_drawdown,
        },
        "overall_metrics": overall,
        "holdout_metrics": holdouts_sorted,
        "latest_window": latest_window,
        "recent_full_holdout": recent_full,
        "pool_compare_rows": pool_rows,
        "best_pool": best_pool,
        "liquid500_pool": liquid500_row,
        "issues": issues,
        "verdict": verdict,
        "decision": decision,
        "pool_comment": pool_comment,
    }


def print_report(report: dict[str, Any]) -> None:
    overall = report["overall_metrics"]
    latest_window = report.get("latest_window") or {}
    recent_full = report.get("recent_full_holdout") or {}
    print(f"正式执行主线目录: {report['formal_dir']}")
    print(f"股票池对照文件: {report['pool_compare_csv']}")
    print()
    print("总体正式框架结果:")
    print(f"  - 总超额收益: {parse_float(overall.get('excess_total_return')):.2%}")
    print(f"  - 总超额 Sharpe: {parse_float(overall.get('excess_sharpe')):.3f}")
    print(f"  - 总超额最大回撤: {parse_float(overall.get('excess_max_drawdown')):.2%}")
    print(f"  - 执行口径: {overall.get('execution_mode', '')}")
    print(f"  - 股票池: {overall.get('rolling_liquidity_pool', overall.get('universe_scope', ''))}")
    print()
    if recent_full:
        print("最近完整 holdout 结果:")
        print(f"  - 区间: {recent_full.get('holdout_start', '')} -> {recent_full.get('holdout_end', '')}")
        print(f"  - 超额收益: {parse_float(recent_full.get('excess_total_return')):.2%}")
        print(f"  - 超额 Sharpe: {parse_float(recent_full.get('excess_sharpe')):.3f}")
        print(f"  - 超额最大回撤: {parse_float(recent_full.get('excess_max_drawdown')):.2%}")
        print()
    if latest_window:
        print("最新子窗口结果:")
        print(f"  - 区间: {latest_window.get('holdout_start', '')} -> {latest_window.get('holdout_end', '')}")
        print(f"  - 超额收益: {parse_float(latest_window.get('excess_total_return')):.2%}")
        print(f"  - 超额 Sharpe: {parse_float(latest_window.get('excess_sharpe')):.3f}")
        print(f"  - 超额最大回撤: {parse_float(latest_window.get('excess_max_drawdown')):.2%}")
        print()

    best_pool = report.get("best_pool")
    liquid500 = report.get("liquid500_pool")
    if best_pool and liquid500:
        print("高流动性池对照:")
        print(
            f"  - 当前最优池: {best_pool.get('pool', '')}, "
            f"超额 Sharpe={parse_float(best_pool.get('excess_sharpe')):.3f}, "
            f"超额收益={parse_float(best_pool.get('excess_total_return')):.2%}"
        )
        print(
            f"  - liquid500: 超额 Sharpe={parse_float(liquid500.get('excess_sharpe')):.3f}, "
            f"超额收益={parse_float(liquid500.get('excess_total_return')):.2%}"
        )
        print(f"  - 说明: {report.get('pool_comment', '')}")
        print()

    print(f"体检结论: {report['verdict']}")
    print(f"动作建议: {report['decision']}")
    if report["issues"]:
        print("触发问题:")
        for issue in report["issues"]:
            print(f"  - {issue}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check execution-line health from formal advanced_ml outputs.")
    parser.add_argument(
        "--formal-dir",
        default="",
        help="Optional formal advanced_ml output directory. Defaults to latest advanced_ml_rolling_liq500_formal_*.",
    )
    parser.add_argument(
        "--pool-compare-csv",
        default="",
        help="Optional pool comparison CSV. Defaults to latest advanced_ml_liquidity_pool_compare_*.csv.",
    )
    parser.add_argument("--json-out", default="", help="Optional UTF-8 JSON output path.")
    parser.add_argument("--min-overall-excess-sharpe", type=float, default=0.80)
    parser.add_argument("--min-recent-full-excess-sharpe", type=float, default=0.50)
    parser.add_argument("--max-allowed-excess-drawdown", type=float, default=-0.35)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    formal_dir = Path(args.formal_dir) if args.formal_dir else latest_by_mtime("advanced_ml_rolling_liq500_formal_*")
    pool_compare_csv = Path(args.pool_compare_csv) if args.pool_compare_csv else latest_by_mtime("advanced_ml_liquidity_pool_compare_*.csv")
    if not formal_dir.is_absolute():
        formal_dir = (WORKSPACE_ROOT / formal_dir).resolve()
    if not pool_compare_csv.is_absolute():
        pool_compare_csv = (WORKSPACE_ROOT / pool_compare_csv).resolve()

    thresholds = HealthThresholds(
        min_overall_excess_sharpe=args.min_overall_excess_sharpe,
        min_recent_full_excess_sharpe=args.min_recent_full_excess_sharpe,
        max_allowed_excess_drawdown=args.max_allowed_excess_drawdown,
    )
    report = build_report(formal_dir, pool_compare_csv, thresholds)
    print_report(report)

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = WORKSPACE_ROOT / out_path
        write_json(out_path, report)
        print()
        print(f"JSON report written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
