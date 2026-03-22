from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ma-boundary and light-risk scan outputs into one report."
    )
    parser.add_argument("--scan-dirs", nargs="+", required=True, help="Scan output directories")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output dir. Defaults to daily_research/output/ma_boundary_risk_<timestamp>",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.2%}"


def _safe_num(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{float(value):.3f}"


def _scan_rows(scan_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(child for child in scan_dir.iterdir() if child.is_dir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _load_json(metrics_path)
        recent_full = _load_json(run_dir / "metrics_recent_full.json")
        latest_weak = _load_json(run_dir / "metrics_latest_weak.json")
        rows.append(
            {
                "scan_dir": scan_dir.name,
                "label": run_dir.name,
                "regime_ma_window": metrics.get("regime_ma_window"),
                "regime_vol_window": metrics.get("regime_vol_window"),
                "regime_max_annual_vol": metrics.get("regime_max_annual_vol"),
                "stop_loss": metrics.get("stop_loss"),
                "take_profit": metrics.get("take_profit"),
                "full_excess_total_return": metrics.get("excess_total_return"),
                "full_excess_sharpe": metrics.get("excess_sharpe"),
                "full_excess_max_drawdown": metrics.get("excess_max_drawdown"),
                "full_avg_turnover": metrics.get("avg_turnover"),
                "recent_full_excess_total_return": recent_full.get("excess_total_return"),
                "recent_full_excess_sharpe": recent_full.get("excess_sharpe"),
                "recent_full_excess_max_drawdown": recent_full.get("excess_max_drawdown"),
                "latest_weak_excess_total_return": latest_weak.get("excess_total_return"),
                "latest_weak_excess_sharpe": latest_weak.get("excess_sharpe"),
                "latest_weak_excess_max_drawdown": latest_weak.get("excess_max_drawdown"),
            }
        )
    return rows


def _build_markdown(output_path: Path, summary_df: pd.DataFrame) -> None:
    ranked = summary_df.sort_values(
        ["latest_weak_excess_sharpe", "latest_weak_excess_total_return", "full_excess_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top = ranked.iloc[0].to_dict()
    best_by_ma = (
        summary_df.sort_values(
            ["regime_ma_window", "latest_weak_excess_sharpe", "full_excess_sharpe"],
            ascending=[True, False, False],
        )
        .groupby("regime_ma_window", as_index=False)
        .head(1)
        .sort_values("regime_ma_window")
    )
    lines = [
        "# ma 边界与轻量风控复验汇总",
        "",
        "## 当前最强候选",
        (
            f"- `{top['label']}` @ ma{int(top['regime_ma_window'])}"
            f" | 全样本超额 Sharpe {_safe_num(top['full_excess_sharpe'])}"
            f" | 最新弱窗口 {_safe_pct(top['latest_weak_excess_total_return'])} / {_safe_num(top['latest_weak_excess_sharpe'])}"
        ),
        (
            f"- 风控参数：stop_loss={top['stop_loss']}, take_profit={top['take_profit']}"
        ),
        "",
        "## 各边界最优候选",
    ]
    for row in best_by_ma.to_dict(orient="records"):
        lines.append(
            f"- ma{int(row['regime_ma_window'])}: `{row['label']}`"
            f" | 全样本超额 Sharpe {_safe_num(row['full_excess_sharpe'])}"
            f" | 最新弱窗口 {_safe_pct(row['latest_weak_excess_total_return'])} / {_safe_num(row['latest_weak_excess_sharpe'])}"
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "- 排序优先看最新弱窗口超额 Sharpe，其次看最新弱窗口超额收益，再看全样本超额 Sharpe。",
            "- 目标不是找最激进参数，而是确认 ma50 邻域是否存在更稳的状态边界，以及轻量风控是否提供净增益。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("daily_research/output") / f"ma_boundary_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for raw_dir in args.scan_dirs:
        scan_dir = Path(raw_dir)
        rows.extend(_scan_rows(scan_dir))

    if not rows:
        raise RuntimeError("No candidate runs found in the provided scan dirs.")

    summary_df = pd.DataFrame(rows)
    ranked_df = summary_df.sort_values(
        ["latest_weak_excess_sharpe", "latest_weak_excess_total_return", "full_excess_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    summary_df.to_csv(output_dir / "boundary_risk_summary.csv", index=False, encoding="utf-8-sig")
    ranked_df.to_csv(output_dir / "boundary_risk_ranked.csv", index=False, encoding="utf-8-sig")
    _build_markdown(output_dir / "boundary_risk_report.md", summary_df)

    report = {
        "top_candidate": ranked_df.iloc[0].to_dict(),
        "rows": summary_df.to_dict(orient="records"),
    }
    (output_dir / "boundary_risk_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"output: {output_dir}")
    print(ranked_df.to_string(index=False))


if __name__ == "__main__":
    main()
