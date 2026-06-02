"""Audit v2 PIT data and forward-return labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from traditional_quant_research.data_audit import V2DataLabelAudit, audit_v2_data_labels


DEFAULT_OUTPUT_DIR = Path("traditional_quant_research/output/experiments/v2_data_label_audit")
DEFAULT_RESEARCH_LOG = Path("traditional_quant_research/research_log/2026-06-02_v2_data_label_audit.md")


def run_v2_data_label_audit(
    *,
    root: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    horizons: tuple[int, ...] = (1, 5, 20),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_research_log: bool = False,
    research_log_path: Path = DEFAULT_RESEARCH_LOG,
) -> dict[str, Any]:
    run_id = f"v2_data_label_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    audit = audit_v2_data_labels(root, start_date=start_date, end_date=end_date, horizons=horizons)
    write_audit_artifacts(audit, run_dir)
    markdown = render_audit_markdown(audit, run_dir)
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    if write_research_log:
        research_log_path.write_text(markdown, encoding="utf-8")
    result = {
        **audit.summary,
        "run_id": run_id,
        "output_dir": str(run_dir),
        "research_log": str(research_log_path) if write_research_log else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_audit_artifacts(audit: V2DataLabelAudit, run_dir: Path) -> None:
    audit.universe_yearly.to_csv(run_dir / "universe_yearly.csv", index=False, encoding="utf-8-sig")
    audit.reject_reason_counts.to_csv(run_dir / "reject_reason_counts.csv", index=False, encoding="utf-8-sig")
    audit.bar_quality.to_csv(run_dir / "bar_quality.csv", index=False, encoding="utf-8-sig")
    audit.label_summary.to_csv(run_dir / "label_summary.csv", index=False, encoding="utf-8-sig")
    audit.label_yearly.to_csv(run_dir / "label_yearly.csv", index=False, encoding="utf-8-sig")
    audit.extreme_label_samples.to_csv(run_dir / "extreme_label_samples.csv", index=False, encoding="utf-8-sig")


def render_audit_markdown(audit: V2DataLabelAudit, run_dir: Path) -> str:
    summary = audit.summary
    label_flags = summary.get("label_quality_flags", {})
    lines = [
        "# 2026-06-02 V2 Data Label Audit",
        "",
        f"- Data Scope: snapshot `{summary.get('snapshot_id')}`, `{summary.get('start_date')}` to `{summary.get('end_date')}`.",
        f"- Universe: `{summary.get('universe', {}).get('rows')}` rows, `{summary.get('universe', {}).get('date_count')}` dates, `{summary.get('universe', {}).get('security_count')}` securities.",
        f"- Tradeable Panel: `{summary.get('tradeable_panel', {}).get('rows')}` rows, `{summary.get('tradeable_panel', {}).get('date_count')}` dates, `{summary.get('tradeable_panel', {}).get('security_count')}` securities.",
        f"- Bar Quality Flags: `{summary.get('bar_quality_flags')}`.",
        f"- Label Convention: {summary.get('label_convention')}",
        "- Key Label Flags: "
        + ", ".join(
            f"`{label}` available `{_fmt(values.get('available_rate'))}`, abs>0.2 rows `{values.get('abs_gt_0.2_rows')}`, abs>0.5 rows `{values.get('abs_gt_0.5_rows')}`"
            for label, values in label_flags.items()
        ),
        "- Assessment: 审计用于识别数据/标签风险，不构成因子有效性结论。",
        "- Next Step: 对极端标签样本做复权/除权核查，并在全周期单因子诊断中按年份引用本审计结果。",
        "",
        "## Label Summary",
        "",
        _markdown_table(audit.label_summary),
        "",
        "## Bar Quality",
        "",
        _markdown_table(audit.bar_quality),
        "",
        "## Reject Reasons",
        "",
        _markdown_table(audit.reject_reason_counts),
        "",
        "## Universe By Year",
        "",
        _markdown_table(audit.universe_yearly),
        "",
        f"Artifacts: `{run_dir}`",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(_fmt)
    return view.to_markdown(index=False)


def _fmt(value: Any) -> str:
    if value is None:
        return "nan"
    try:
        if pd.isna(value):
            return "nan"
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="Optional v2 snapshot root.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizons", default="1,5,20", help="Comma-separated forward label horizons.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-research-log", action="store_true")
    parser.add_argument("--research-log-path", type=Path, default=DEFAULT_RESEARCH_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    result = run_v2_data_label_audit(
        root=args.root,
        start_date=args.start_date,
        end_date=args.end_date,
        horizons=horizons,
        output_dir=args.output_dir,
        write_research_log=args.write_research_log,
        research_log_path=args.research_log_path,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
