from __future__ import annotations

"""Append the authorized Seq100 2026 fold comparison to the complete report.

The report artifact is an append-only audit surface.  This script validates the
comparison and its frozen inputs, then adds a suffix to the existing manifest
and snapshot without rebuilding or reordering any earlier report object.
"""

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "daily_research/output/path_policy/studies/"
    "seq100_legal_structured_path_rolling_2023_2025_v1/analysis/"
    "unit_time_exit_20260719/artifact.json"
)
COMPARISON_ROOT = (
    PROJECT_ROOT
    / "daily_research/output/path_policy/studies/"
    "seq100_legal_structured_path_2026_fold_v1/analysis/"
    "checkpoint_vintage_2026_20260719"
)
STUDY_ROOT = (
    PROJECT_ROOT
    / "daily_research/output/path_policy/studies/"
    "seq100_legal_structured_path_2026_fold_v1"
)
STUDY_PATH = STUDY_ROOT / "study.json"
CONTRACT_PATH = COMPARISON_ROOT / "contract.json"
SUMMARY_PATH = COMPARISON_ROOT / "comparison_summary.json"
METRICS_PATH = COMPARISON_ROOT / "vintage_metrics.csv"
PAIRWISE_PATH = COMPARISON_ROOT / "pairwise_summary.csv"
DAILY_PAIRWISE_PATH = COMPARISON_ROOT / "daily_pairwise_deltas.csv"
AUDIT_PATH = COMPARISON_ROOT / "artifact_append_audit.json"

EXPECTED_PROFILE_ORDER = ["legal_flat_baseline", "structured_joint_turnover"]
PROFILE_LABELS = {
    "legal_flat_baseline": "Legal flat",
    "structured_joint_turnover": "Structured joint",
}
EXPECTED_VINTAGES = [2023, 2024, 2025, 2026]
EXPECTED_OLD_COUNTS = {
    "cards": 19,
    "charts": 11,
    "tables": 15,
    "blocks": 55,
    "sources": 16,
    "datasets": 29,
}
MAX_OVERLAY_BYTES = int(1.25 * 1024**3)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _parse_csv_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer() and "." not in text and "e" not in lowered:
        return int(number)
    return number


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: _parse_csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("missing required comparison files: " + ", ".join(missing))


def _validate_inputs(
    summary: dict[str, Any], contract: dict[str, Any], study: dict[str, Any]
) -> dict[str, Any]:
    if summary.get("status") != "completed":
        raise RuntimeError("comparison summary is not completed")
    contract_payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    declared_contract_hash = str(contract.get("contract_sha256", ""))
    if not declared_contract_hash or _digest(contract_payload) != declared_contract_hash:
        raise RuntimeError("comparison contract digest is invalid")
    if str(summary.get("suite_contract_sha256", "")) != declared_contract_hash:
        raise RuntimeError("summary and comparison contract hashes disagree")
    if int(summary.get("target_year", -1)) != 2026:
        raise RuntimeError("comparison target year drifted")
    if list(summary.get("checkpoint_vintages", [])) != EXPECTED_VINTAGES:
        raise RuntimeError("comparison vintage list drifted")
    if list(summary.get("profiles", [])) != EXPECTED_PROFILE_ORDER:
        raise RuntimeError("comparison profile list drifted")
    if list(summary.get("signal_window", [])) != ["2026-01-05", "2026-03-19"]:
        raise RuntimeError("comparison signal window drifted")
    if int(summary.get("signal_date_count", -1)) != 48:
        raise RuntimeError("comparison signal date count drifted")
    if int(summary.get("candidate_count", -1)) != 143_840:
        raise RuntimeError("comparison candidate count drifted")
    if int(summary.get("symbol_count", -1)) != 3_034:
        raise RuntimeError("comparison symbol count drifted")
    fairness = dict(summary.get("fairness", {}) or {})
    if fairness.get("pass") is not True:
        raise RuntimeError("comparison fairness checks did not pass")
    checks = dict(fairness.get("checks", {}) or {})
    if checks and not all(bool(value) for value in checks.values()):
        raise RuntimeError("comparison fairness check contains a failure")
    if not summary.get("existing_2026_consistency", {}).get("pass", False):
        raise RuntimeError("2026 training evaluation consistency did not pass")
    if summary.get("score_only_tail_included") is not False:
        raise RuntimeError("score-only tail was included")
    if summary.get("finite_capital_cagr_reported") is not False:
        raise RuntimeError("finite-capital CAGR was reported for the short window")
    uncertainty = dict(summary.get("uncertainty", {}) or {})
    if uncertainty.get("formal_moving_block_interval") is not False:
        raise RuntimeError("formal moving-block interval unexpectedly enabled")

    metrics = _read_csv(METRICS_PATH)
    expected_keys = {
        (profile, vintage)
        for profile in EXPECTED_PROFILE_ORDER
        for vintage in EXPECTED_VINTAGES
    }
    actual_keys = {
        (str(row.get("profile")), int(row.get("checkpoint_vintage"))) for row in metrics
    }
    if actual_keys != expected_keys or len(metrics) != 8:
        raise RuntimeError("vintage_metrics.csv does not contain exactly eight results")
    pairwise = _read_csv(PAIRWISE_PATH)
    if len(pairwise) != 30:
        raise RuntimeError("pairwise_summary.csv should contain 30 rows")
    if not DAILY_PAIRWISE_PATH.is_file():
        raise RuntimeError("daily_pairwise_deltas.csv is missing")
    output_hashes = {
        METRICS_PATH: str(summary.get("outputs", {}).get("vintage_metrics_csv_sha256", "")),
        PAIRWISE_PATH: str(summary.get("outputs", {}).get("pairwise_summary_csv_sha256", "")),
        DAILY_PAIRWISE_PATH: str(
            summary.get("outputs", {}).get("daily_pairwise_deltas_csv_sha256", "")
        ),
    }
    for output_path, expected_hash in output_hashes.items():
        if not expected_hash or _sha256_file(output_path) != expected_hash:
            raise RuntimeError(f"comparison output hash drifted: {output_path}")

    material = dict(study.get("material", {}) or {})
    overlay_size = int(material.get("overlay_size_bytes", -1))
    if overlay_size <= 0 or overlay_size > MAX_OVERLAY_BYTES:
        raise RuntimeError(f"overlay size violates the 1.25 GiB guard: {overlay_size}")
    runtime = dict(study.get("runtime", {}) or {})
    if runtime.get("status") != "runs_completed":
        raise RuntimeError("2026 study runtime is not complete")
    if set(runtime.get("completed_tasks", [])) != {
        "legal_flat_baseline:2026",
        "structured_joint_turnover:2026",
    }:
        raise RuntimeError("2026 study task completion is incomplete")
    study_contract = dict(study.get("contract", {}) or {})
    if dict(study_contract.get("protected_boundaries", {}) or {}) != {
        "update_qdp": False,
        "call_provider": False,
        "mutate_base_pack": False,
        "change_active_execution": False,
        "create_final_model": False,
    }:
        raise RuntimeError("study protected boundaries drifted")
    if material.get("base_pack_sha256") and _sha256_file(Path(material["base_pack_manifest"])) != str(
        material["base_pack_sha256"]
    ):
        raise RuntimeError("base pack manifest changed")
    frozen = dict(material.get("frozen_inputs", {}) or {})
    # The QDP hash is recorded in the fold contract; re-read it only for the
    # final immutability assertion, never through a provider or QDP writer.
    qdp_manifest = frozen.get("qdp_active_manifest")
    qdp_hash = frozen.get("qdp_active_manifest_sha256")
    if qdp_manifest and qdp_hash and _sha256_file(Path(qdp_manifest)) != str(qdp_hash):
        raise RuntimeError("QDP active manifest changed")
    return {
        "metrics": metrics,
        "pairwise": pairwise,
        "overlay_size_bytes": overlay_size,
        "contract_sha256": declared_contract_hash,
        "base_pack_sha256": str(material.get("base_pack_sha256", "")),
        "qdp_active_manifest_sha256": str(qdp_hash or ""),
        "study_contract_sha256": str(study.get("contract_sha256", "")),
    }


def _source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    executed_at: str,
    tables: list[str],
    filters: list[str],
    definitions: list[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": executed_at,
            "tables_used": tables,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def _metric_rows(raw_rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    gate_profiles = dict(summary.get("gate", {}).get("profiles", {}) or {})
    rows: list[dict[str, Any]] = []
    for raw in sorted(
        raw_rows,
        key=lambda row: (
            EXPECTED_PROFILE_ORDER.index(str(row["profile"])),
            int(row["checkpoint_vintage"]),
        ),
    ):
        profile = str(raw["profile"])
        vintage = int(raw["checkpoint_vintage"])
        row = dict(raw)
        row.update(
            {
                "profile_label": PROFILE_LABELS[profile],
                "profile_vintage": f"{PROFILE_LABELS[profile]} / {vintage}",
                "checkpoint_label": str(vintage),
                "evaluation_mode_label": (
                    "复用 2026 训练期评价"
                    if vintage == 2026
                    else "跨 checkpoint 统一推理"
                ),
                "profile_gate_status": (
                    "通过" if gate_profiles.get(profile, {}).get("profile_pass") else "未通过"
                ),
            }
        )
        rows.append(row)
    return rows


def _pairwise_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {
        "rank_ic": "Rank IC",
        "top3_base_alpha": "Top3 executable alpha (base)",
        "top3_stress_alpha": "Top3 executable alpha (stress)",
        "top3_opportunity_alpha": "Top3 opportunity alpha",
        "top3_exit_regret": "Top3 exit regret",
    }
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        profile = str(raw["profile"])
        older = int(raw["older_vintage"])
        metric = str(raw["metric"])
        row = dict(raw)
        row.update(
            {
                "profile_label": PROFILE_LABELS[profile],
                "comparison_label": f"2026 vs {older}",
                "metric_label": labels.get(metric, metric),
                "direction": "越低越好" if "regret" in metric else "越高越好",
                "win_rate": float(raw["checkpoint_2026_win_rate"]),
                "mean_delta_display": (
                    f"{float(raw['mean_improvement_delta']):.6f}"
                    if metric == "rank_ic"
                    else f"{float(raw['mean_improvement_delta']):.4%}"
                ),
                "median_delta_display": (
                    f"{float(raw['median_improvement_delta']):.6f}"
                    if metric == "rank_ic"
                    else f"{float(raw['median_improvement_delta']):.4%}"
                ),
            }
        )
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            EXPECTED_PROFILE_ORDER.index(str(row["profile"])),
            int(row["older_vintage"]),
            str(row["metric"]),
        ),
    )


def _gate_rows(metrics: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_key = {(str(row["profile"]), int(row["checkpoint_vintage"])): row for row in metrics}
    gate_profiles = dict(summary.get("gate", {}).get("profiles", {}) or {})
    specs = [
        ("primary_top3_pass", "主门槛：2026 Top3 alpha 高于三个旧 checkpoint"),
        ("breadth_pass", "广度：Top1/Top5/Top10 至少两项高于三个旧 checkpoint"),
        ("ranking_support_pass", "排名支持：Rank IC 或 Top3 opportunity alpha"),
        ("path_support_pass", "路径支持：Exit regret 或 Path MAE"),
        ("profile_pass", "Profile 综合门槛"),
    ]
    rows: list[dict[str, Any]] = []
    for profile in EXPECTED_PROFILE_ORDER:
        target = by_key[(profile, 2026)]
        older = [by_key[(profile, vintage)] for vintage in (2023, 2024, 2025)]
        for key, definition in specs:
            if key == "primary_top3_pass":
                value = float(target["top3_base_alpha"])
                baseline = max(float(row["top3_base_alpha"]) for row in older)
                unit = "alpha"
            elif key == "breadth_pass":
                wins = sum(
                    float(target[field]) > max(float(row[field]) for row in older)
                    for field in ("top1_base_alpha", "top5_base_alpha", "top10_base_alpha")
                )
                value, baseline, unit = wins, 2, "count"
            elif key == "ranking_support_pass":
                value = float(target["rank_ic"])
                baseline = max(float(row["rank_ic"]) for row in older)
                unit = "Rank IC"
            elif key == "path_support_pass":
                value = float(target["path_mae"])
                baseline = min(float(row["path_mae"]) for row in older)
                unit = "Path MAE; lower is better"
            else:
                value = 1 if gate_profiles[profile].get("profile_pass") else 0
                baseline = 1
                unit = "Boolean"
            rows.append(
                {
                    "profile": profile,
                    "profile_label": PROFILE_LABELS[profile],
                    "evidence_key": key,
                    "evidence_definition": definition,
                    "target_value": value,
                    "older_baseline": baseline,
                    "unit": unit,
                    "status": "通过" if gate_profiles[profile].get(key) else "未通过",
                }
            )
    rows.append(
        {
            "profile": "overall",
            "profile_label": "Overall",
            "evidence_key": "overall_2026_freshness_gate_pass",
            "evidence_definition": "两个 profile、统一材料、coverage 与合同检查全部通过",
            "target_value": 1 if summary.get("overall_2026_freshness_gate_pass") else 0,
            "older_baseline": 1,
            "unit": "Boolean",
            "status": "通过" if summary.get("overall_2026_freshness_gate_pass") else "未通过",
        }
    )
    return rows


def _training_rows(study: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    development_root = STUDY_ROOT / "runs/development"
    for profile in EXPECTED_PROFILE_ORDER:
        candidates = sorted(
            development_root.glob(f"seq100_structured_{profile}_2026_seed7_*/sequence_path_training_summary.json")
        )
        if len(candidates) != 1:
            raise RuntimeError(f"expected one completed 2026 training summary for {profile}")
        record = _read_json(candidates[0])
        history = list(record.get("history", []) or [])
        best = next((row for row in history if row.get("is_best")), None)
        if best is None:
            raise RuntimeError(f"missing best epoch in training summary for {profile}")
        fold_contract = dict(record.get("fold_training_contract", {}) or {})
        rows.append(
            {
                "profile": profile,
                "profile_label": PROFILE_LABELS[profile],
                "checkpoint_vintage": 2026,
                "seed": int(record.get("seed", -1)),
                "train_row_count": int(record["sample_selection"]["train"]["selected_row_count"]),
                "development_candidate_count": int(
                    record["sample_selection"]["development_supervised"]["selected_row_count"]
                ),
                "best_epoch": int(record["best_epoch"]),
                "completed_epochs": int(record["completed_epochs"]),
                "best_development_price_total_loss": float(
                    best["development_price_total_loss"]
                ),
                "normalization_cutoff": str(record["normalization_cutoff"]),
                "fold_contract_sha256": str(fold_contract.get("sha256", "")),
                "normalization_sha256": str(fold_contract.get("normalization_sha256", "")),
                "checkpoint_sha256": str(record.get("best_checkpoint_sha256", "")),
                "architecture": str(record.get("model", {}).get("type", "")),
            }
        )
    if len(rows) != 2 or any(row["seed"] != 7 for row in rows):
        raise RuntimeError("2026 training contract does not match seed 7")
    return rows


def _build_additions(
    summary: dict[str, Any], contract: dict[str, Any], study: dict[str, Any], checked: dict[str, Any]
) -> dict[str, Any]:
    metrics = _metric_rows(checked["metrics"], summary)
    pairwise = _pairwise_rows(checked["pairwise"])
    gate = _gate_rows(metrics, summary)
    training = _training_rows(study)
    completed_at = str(summary.get("completed_at", ""))
    summary_rel = _relative(SUMMARY_PATH)
    metrics_rel = _relative(METRICS_PATH)
    contract_rel = _relative(CONTRACT_PATH)
    pairwise_rel = _relative(PAIRWISE_PATH)
    daily_rel = _relative(DAILY_PAIRWISE_PATH)
    study_rel = _relative(STUDY_PATH)

    comparison_source = _source(
        "checkpoint_vintage_2026_source",
        "2026 fold 与四年代统一评价汇总、合同和训练记录",
        summary_rel,
        (
            "SELECT target_year, signal_window, signal_date_count, candidate_count, "
            "symbol_count, fairness, gate, uncertainty, metrics "
            f"FROM read_json_auto('{summary_rel}')"
        ),
        "读取 2026 完整标签窗口的八组 vintage 指标、统一材料哈希、训练 fold 合同和 freshness gate。",
        completed_at,
        [summary_rel, metrics_rel, contract_rel, study_rel],
        [
            "signal window: 2026-01-05..2026-03-19",
            "48 signal dates, 143,840 candidate rows, 3,034 symbols",
            "four checkpoint vintages per profile; 2026 rows reuse completed training evaluation",
            "score-only dates after 2026-03-19 are excluded",
            "base pack and QDP are read-only; no provider calls",
        ],
        [
            "Executable alpha = selected net realized plan return minus same-date executable universe return after the frozen cost contract.",
            "Rank IC = daily cross-sectional Spearman correlation; positive-day rate is the share of covered signal dates with positive IC.",
            "Path MAE is mean absolute error over the 60-day OHLC path in relative-price space; channel MAE is reported separately.",
            "Exit regret is the legal opportunity value minus the model-planned executable return on covered candidates.",
            "Freshness gate requires 2026 Top3 alpha to exceed all three older vintages, breadth in at least two of Top1/5/10, and ranking/path support.",
        ],
    )
    pairwise_source = _source(
        "checkpoint_vintage_2026_pairwise_source",
        "2026 checkpoint 对旧 vintage 的逐日配对差异",
        pairwise_rel,
        (
            "SELECT profile, older_vintage, metric, date_count, mean_improvement_delta, "
            "median_improvement_delta, checkpoint_2026_win_rate "
            f"FROM read_csv_auto('{pairwise_rel}', header=true) "
            "ORDER BY profile, older_vintage, metric"
        ),
        "读取 2026 与 2023/2024/2025 的逐日配对汇总；原始逐日行保留在 daily_pairwise_deltas.csv。",
        completed_at,
        [pairwise_rel, daily_rel, summary_rel],
        [
            "48 paired signal dates for every profile/older-vintage/metric combination",
            "higher-is-better metrics use checkpoint_2026_value > older_checkpoint_value",
            "exit regret is interpreted as lower-is-better",
            "paired deltas are descriptive because the window is shorter than the 60-day dependency",
        ],
        [
            "Mean and median improvement delta are checkpoint 2026 minus the older checkpoint under the stored metric direction.",
            "Win rate is the share of paired dates on which the 2026 value wins under the metric direction.",
            "No moving-block confidence interval is estimated for 48 signal dates versus a 60-day outcome dependency.",
        ],
    )

    fairness_checks = dict(summary["fairness"]["checks"])
    headline = [
        {
            "overall_gate_pass_numeric": 1
            if summary.get("overall_2026_freshness_gate_pass")
            else 0,
            "overall_gate_status": "通过"
            if summary.get("overall_2026_freshness_gate_pass")
            else "未通过",
            "passing_profile_count": sum(
                bool(value.get("profile_pass"))
                for value in summary["gate"]["profiles"].values()
            ),
            "total_profile_count": 2,
            "signal_date_count": 48,
            "candidate_count": 143_840,
            "symbol_count": 3_034,
            "score_coverage": min(float(row["candidate_score_coverage"]) for row in metrics),
            "execution_coverage": min(float(row["execution_return_coverage"]) for row in metrics),
            "fairness_failure_count": sum(not bool(value) for value in fairness_checks.values()),
            "training_task_count": 2,
            "overlay_size_gib": checked["overlay_size_bytes"] / 1024**3,
            "train_row_count": 7_773_480,
            "purged_row_count": 239_081,
            "purged_date_count": 80,
            "normalization_cutoff": "2026-01-05",
            "contract_sha256": checked["contract_sha256"],
        }
    ]
    top3_pairwise = [
        row for row in pairwise if str(row["metric"]) == "top3_base_alpha"
    ]
    if len(top3_pairwise) != 6:
        raise RuntimeError("expected six Top3 pairwise rows")
    metric_lookup = {
        (str(row["profile"]), int(row["checkpoint_vintage"])): row
        for row in metrics
    }
    pairwise_lookup = {
        (str(row["profile"]), int(row["older_vintage"])): row
        for row in top3_pairwise
    }
    legal_alpha_text = "、".join(
        f"**{float(metric_lookup[('legal_flat_baseline', vintage)]['top3_base_alpha']):.4%}**（{vintage}）"
        for vintage in EXPECTED_VINTAGES
    )
    structured_alpha_text = "、".join(
        f"**{float(metric_lookup[('structured_joint_turnover', vintage)]['top3_base_alpha']):.4%}**（{vintage}）"
        for vintage in EXPECTED_VINTAGES
    )
    legal_2026 = metric_lookup[("legal_flat_baseline", 2026)]
    structured_2026 = metric_lookup[("structured_joint_turnover", 2026)]
    legal_pairwise_text = " / ".join(
        f"{float(pairwise_lookup[('legal_flat_baseline', vintage)]['win_rate']):.2%}"
        for vintage in (2023, 2024, 2025)
    )
    structured_pairwise_text = " / ".join(
        f"{float(pairwise_lookup[('structured_joint_turnover', vintage)]['win_rate']):.2%}"
        for vintage in (2023, 2024, 2025)
    )

    cards = [
        {
            "id": "checkpoint_vintage_2026_gate_card",
            "dataset": "checkpoint_vintage_2026_headline",
            "sourceId": "checkpoint_vintage_2026_source",
            "description": "四年代统一评价的预注册 freshness gate；未通过不产生 promotion 或实盘结论。",
            "metrics": [
                {"label": "Gate 通过（1=是）", "field": "overall_gate_pass_numeric", "format": "number"},
                {"label": "通过 profile", "field": "passing_profile_count", "format": "number"},
            ],
        },
        {
            "id": "checkpoint_vintage_2026_scope_card",
            "dataset": "checkpoint_vintage_2026_headline",
            "sourceId": "checkpoint_vintage_2026_source",
            "description": "只统计完整标签和 20 日执行尾部都可用的 2026 信号窗口。",
            "metrics": [
                {"label": "信号日", "field": "signal_date_count", "format": "number"},
                {"label": "候选行", "field": "candidate_count", "format": "number"},
                {"label": "symbol", "field": "symbol_count", "format": "number"},
            ],
        },
        {
            "id": "checkpoint_vintage_2026_coverage_card",
            "dataset": "checkpoint_vintage_2026_headline",
            "sourceId": "checkpoint_vintage_2026_source",
            "description": "八组结果共享同一 candidate、label、execution、cost 和评价合同。",
            "metrics": [
                {"label": "Score 覆盖", "field": "score_coverage", "format": "percent"},
                {"label": "Execution 覆盖", "field": "execution_coverage", "format": "percent"},
                {"label": "公平性失败项", "field": "fairness_failure_count", "format": "number"},
            ],
        },
        {
            "id": "checkpoint_vintage_2026_training_card",
            "dataset": "checkpoint_vintage_2026_headline",
            "sourceId": "checkpoint_vintage_2026_source",
            "description": "两个 profile 在同一 2026 fold 上串行完成训练；这是研究 checkpoint，不是 final/deployment model。",
            "metrics": [
                {"label": "2026 训练任务", "field": "training_task_count", "format": "number"},
                {"label": "Overlay GiB", "field": "overlay_size_gib", "format": "number"},
                {"label": "训练行", "field": "train_row_count", "format": "number"},
            ],
        },
    ]

    metric_columns = [
        {"field": "profile_vintage", "label": "Profile / checkpoint", "type": "text"},
        {"field": "evaluation_mode_label", "label": "评价方式", "type": "text"},
        {"field": "normalization_fit_date_end_exclusive", "label": "Normalization 截止（不含）", "type": "date"},
        {"field": "candidate_count", "label": "候选行", "format": "number"},
        {"field": "rank_ic", "label": "Rank IC", "format": "number"},
        {"field": "rank_ic_positive_day_rate", "label": "IC 正日占比", "format": "percent"},
        {"field": "path_mae", "label": "Path MAE", "format": "percent"},
        {"field": "path_open_mae", "label": "Open MAE", "format": "percent"},
        {"field": "path_high_mae", "label": "High MAE", "format": "percent"},
        {"field": "path_low_mae", "label": "Low MAE", "format": "percent"},
        {"field": "path_close_mae", "label": "Close MAE", "format": "percent"},
        {"field": "top1_base_alpha", "label": "Top1 alpha", "format": "percent"},
        {"field": "top3_base_alpha", "label": "Top3 alpha", "format": "percent"},
        {"field": "top5_base_alpha", "label": "Top5 alpha", "format": "percent"},
        {"field": "top10_base_alpha", "label": "Top10 alpha", "format": "percent"},
        {"field": "top3_stress_alpha", "label": "Top3 stress alpha", "format": "percent"},
        {"field": "top3_opportunity_alpha", "label": "Top3 opportunity alpha", "format": "percent"},
        {"field": "exit_regret", "label": "Exit regret", "format": "percent"},
        {"field": "candidate_score_coverage", "label": "Score 覆盖", "format": "percent"},
        {"field": "execution_return_coverage", "label": "Execution 覆盖", "format": "percent"},
        {"field": "profile_gate_status", "label": "Profile 门槛", "type": "text"},
    ]
    pairwise_columns = [
        {"field": "profile_label", "label": "Profile", "type": "text"},
        {"field": "comparison_label", "label": "比较", "type": "text"},
        {"field": "metric_label", "label": "指标", "type": "text"},
        {"field": "direction", "label": "方向", "type": "text"},
        {"field": "date_count", "label": "配对日", "format": "number"},
        {"field": "mean_delta_display", "label": "平均差异", "type": "text"},
        {"field": "median_delta_display", "label": "中位差异", "type": "text"},
        {"field": "win_rate", "label": "2026 胜出日占比", "format": "percent"},
    ]
    gate_columns = [
        {"field": "profile_label", "label": "Profile", "type": "text"},
        {"field": "evidence_key", "label": "证据项", "type": "text"},
        {"field": "evidence_definition", "label": "定义", "type": "text"},
        {"field": "target_value", "label": "2026 值", "format": "number"},
        {"field": "older_baseline", "label": "旧版本基线", "format": "number"},
        {"field": "unit", "label": "单位/方向", "type": "text"},
        {"field": "status", "label": "状态", "type": "text"},
    ]
    training_columns = [
        {"field": "profile_label", "label": "Profile", "type": "text"},
        {"field": "checkpoint_vintage", "label": "Checkpoint", "format": "number"},
        {"field": "train_row_count", "label": "训练行", "format": "number"},
        {"field": "development_candidate_count", "label": "Development candidate", "format": "number"},
        {"field": "best_epoch", "label": "最佳 epoch", "format": "number"},
        {"field": "completed_epochs", "label": "完成 epoch", "format": "number"},
        {"field": "best_development_price_total_loss", "label": "最佳 development price loss", "format": "number"},
        {"field": "normalization_cutoff", "label": "Normalization 截止", "type": "date"},
        {"field": "architecture", "label": "Architecture", "type": "text"},
    ]

    charts = [
        {
            "id": "checkpoint_vintage_2026_alpha_chart",
            "title": "2026 窗口 Top3 executable alpha 与 checkpoint 年代",
            "subtitle": "2026 checkpoint 未同时超过三个旧版本；分组显示两个 profile，48 个信号日等权聚合。",
            "intent": "comparison",
            "question": "不同 checkpoint 年代在同一 2026 候选集上的 Top3 alpha 如何变化？",
            "rationale": "分组柱形图保留 profile 这一第二分类维度，直接比较四个 checkpoint 年代的同口径 Top3 可执行 alpha。",
            "comparisonContext": {"grain": "profile × checkpoint vintage", "unit": "executable alpha", "normalization": "same 2026 candidate and execution view"},
            "type": "bar",
            "dataset": "checkpoint_vintage_2026_metrics",
            "sourceId": "checkpoint_vintage_2026_source",
            "encodings": {
                "x": {"field": "checkpoint_label", "type": "ordinal", "label": "Checkpoint 年代"},
                "y": {"field": "top3_base_alpha", "type": "quantitative", "label": "Top3 executable alpha"},
                "color": {"field": "profile_label", "type": "nominal", "label": "Profile"},
                "tooltip": [
                    {"field": "profile_label", "label": "Profile"},
                    {"field": "checkpoint_label", "label": "Checkpoint"},
                    {"field": "top3_base_alpha", "label": "Top3 base alpha", "format": "percent"},
                    {"field": "top3_stress_alpha", "label": "Top3 stress alpha", "format": "percent"},
                    {"field": "candidate_score_coverage", "label": "Score coverage", "format": "percent"},
                ],
            },
            "valueFormat": "percent",
            "palette": {"kind": "categorical", "name": "profile comparison"},
            "layout": "full",
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "all"},
            "settings": {"groupMode": "grouped"},
            "surface": {"surface": "card", "viewMode": "visualization"},
        },
        {
            "id": "checkpoint_vintage_2026_path_mae_chart",
            "title": "2026 窗口 OHLC path MAE 与 checkpoint 年代",
            "subtitle": "Legal flat 的 2026 path MAE 低于 Structured joint，但该路径优势没有转化为整体 freshness gate 通过。",
            "intent": "comparison",
            "question": "四个 checkpoint 年代的路径误差是否支持 2026 更新？",
            "rationale": "沿用同一标签材料，按 profile 与 checkpoint 分组呈现总体 OHLC path MAE，避免把单一收益指标当作充分证据。",
            "comparisonContext": {"grain": "profile × checkpoint vintage", "unit": "relative-price MAE", "normalization": "same 2026 label material"},
            "type": "bar",
            "dataset": "checkpoint_vintage_2026_metrics",
            "sourceId": "checkpoint_vintage_2026_source",
            "encodings": {
                "x": {"field": "checkpoint_label", "type": "ordinal", "label": "Checkpoint 年代"},
                "y": {"field": "path_mae", "type": "quantitative", "label": "OHLC path MAE"},
                "color": {"field": "profile_label", "type": "nominal", "label": "Profile"},
                "tooltip": [
                    {"field": "profile_label", "label": "Profile"},
                    {"field": "checkpoint_label", "label": "Checkpoint"},
                    {"field": "path_mae", "label": "Path MAE", "format": "percent"},
                    {"field": "path_open_mae", "label": "Open MAE", "format": "percent"},
                    {"field": "path_close_mae", "label": "Close MAE", "format": "percent"},
                    {"field": "exit_regret", "label": "Exit regret", "format": "percent"},
                ],
            },
            "valueFormat": "percent",
            "palette": {"kind": "categorical", "name": "profile comparison"},
            "layout": "full",
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "all"},
            "settings": {"groupMode": "grouped"},
            "surface": {"surface": "card", "viewMode": "visualization"},
        },
        {
            "id": "checkpoint_vintage_2026_pairwise_win_chart",
            "title": "2026 Top3 alpha 对旧 checkpoint 的逐日胜出比例",
            "subtitle": "Legal flat 相对 2025 的胜出日占比最高，但相对 2024 仍低于 50%；这些是描述性配对结果。",
            "intent": "comparison",
            "question": "2026 checkpoint 在逐日 Top3 alpha 配对中有多常胜出？",
            "rationale": "用旧 checkpoint 年代作为横轴、profile 作为分组，显示 48 个配对信号日中的胜出比例。",
            "comparisonContext": {"grain": "profile × older checkpoint", "unit": "paired-day win rate", "normalization": "same 2026 candidates and costs"},
            "type": "bar",
            "dataset": "checkpoint_vintage_2026_pairwise_top3",
            "sourceId": "checkpoint_vintage_2026_pairwise_source",
            "encodings": {
                "x": {"field": "older_vintage_label", "type": "ordinal", "label": "旧 checkpoint"},
                "y": {"field": "win_rate", "type": "quantitative", "label": "2026 胜出日占比"},
                "color": {"field": "profile_label", "type": "nominal", "label": "Profile"},
                "tooltip": [
                    {"field": "profile_label", "label": "Profile"},
                    {"field": "comparison_label", "label": "比较"},
                    {"field": "date_count", "label": "配对日"},
                    {"field": "mean_improvement_delta", "label": "平均差异", "format": "percent"},
                    {"field": "win_rate", "label": "胜出日占比", "format": "percent"},
                ],
            },
            "valueFormat": "percent",
            "referenceLines": [{"value": 0.5, "axis": "y", "color": "neutral", "lineStyle": "dashed", "label": "50%"}],
            "palette": {"kind": "categorical", "name": "profile comparison"},
            "layout": "full",
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "all"},
            "settings": {"groupMode": "grouped"},
            "surface": {"surface": "card", "viewMode": "visualization"},
        },
    ]

    tables = [
        {
            "id": "checkpoint_vintage_2026_metrics_table_asset",
            "title": "四年代 checkpoint 在统一 2026 窗口上的完整指标",
            "subtitle": "两个 profile × 2023/2024/2025/2026；所有行共享 143,840 候选和同一执行费用合同。",
            "dataset": "checkpoint_vintage_2026_metrics",
            "defaultSort": {"field": "profile_vintage", "direction": "asc"},
            "density": "dense",
            "sourceId": "checkpoint_vintage_2026_source",
            "columns": metric_columns,
        },
        {
            "id": "checkpoint_vintage_2026_pairwise_table_asset",
            "title": "2026 checkpoint 与旧版本的逐日配对摘要",
            "subtitle": "每行 48 个信号日；均值/中位数为 checkpoint 2026 相对旧版本的差异。",
            "dataset": "checkpoint_vintage_2026_pairwise_summary",
            "defaultSort": {"field": "profile_label", "direction": "asc"},
            "density": "dense",
            "sourceId": "checkpoint_vintage_2026_pairwise_source",
            "columns": pairwise_columns,
        },
        {
            "id": "checkpoint_vintage_2026_gate_table_asset",
            "title": "2026 freshness gate 证据",
            "subtitle": "主门槛、广度、排名、路径和 overall 判定均按预先锁定的四年代规则计算。",
            "dataset": "checkpoint_vintage_2026_gate_evidence",
            "defaultSort": {"field": "profile_label", "direction": "asc"},
            "density": "dense",
            "sourceId": "checkpoint_vintage_2026_source",
            "columns": gate_columns,
        },
        {
            "id": "checkpoint_vintage_2026_training_table_asset",
            "title": "2026 fold 训练合同与 checkpoint",
            "subtitle": "训练行、最佳 epoch、normalization 截止和 fold/checkpoint 哈希用于复核研究 checkpoint 身份。",
            "dataset": "checkpoint_vintage_2026_training",
            "defaultSort": {"field": "profile_label", "direction": "asc"},
            "density": "dense",
            "sourceId": "checkpoint_vintage_2026_source",
            "columns": training_columns,
        },
    ]

    blocks = [
        {
            "id": "checkpoint_vintage_2026_summary_section",
            "type": "markdown",
            "sourceId": "checkpoint_vintage_2026_source",
            "layout": "full",
            "body": """## 用户授权的 2026 扩展仍未支持 freshness promotion

在保留此前“2025 freshness 门槛失败、未触发 2026”的历史结论后，本次按用户随后授权追加了一个共享 2026 fold，并在完全相同的 2026 候选集上评价 2023、2024、2025、2026 四个 checkpoint。2026 fold 的整体 freshness gate 仍为 **未通过**：Legal flat 与 Structured joint 都没有同时超过三个旧 checkpoint，且没有得到完整的排名和路径支持。因此这组结果不支持 promotion、deployment 或 live trading。

评价只覆盖 **2026-01-05..2026-03-19** 的 48 个信号日、143,840 个候选行和 3,034 个 symbol。3 月 20 日之后的 score-only 日期没有进入实现收益指标；该窗口也不生成有限资金 CAGR。""",
        },
        {
            "id": "checkpoint_vintage_2026_headline_metric_strip",
            "type": "metric-strip",
            "cardIds": [
                "checkpoint_vintage_2026_gate_card",
                "checkpoint_vintage_2026_scope_card",
                "checkpoint_vintage_2026_coverage_card",
                "checkpoint_vintage_2026_training_card",
            ],
            "layout": "full",
        },
        {
            "id": "checkpoint_vintage_2026_alpha_section",
            "type": "markdown",
            "sourceId": "checkpoint_vintage_2026_source",
            "layout": "full",
            "body": f"""## 2026 Top3 alpha 没有形成跨旧版本的一致领先

Legal flat 的四年代 Top3 executable alpha 依次为 {legal_alpha_text}；Structured joint 依次为 {structured_alpha_text}。完整精确值见下表，图表只承担年代与 profile 的比较。2026 的主门槛要求严格高于 2023、2024、2025 三个旧值；该条件在两个 profile 均未满足，不能用单个较好的对照年替代。""",
        },
        {
            "id": "checkpoint_vintage_2026_alpha_chart_block",
            "type": "chart",
            "chartId": "checkpoint_vintage_2026_alpha_chart",
            "layout": "full",
        },
        {
            "id": "checkpoint_vintage_2026_metrics_table_block",
            "type": "table",
            "tableId": "checkpoint_vintage_2026_metrics_table_asset",
            "layout": "full",
        },
        {
            "id": "checkpoint_vintage_2026_path_section",
            "type": "markdown",
            "sourceId": "checkpoint_vintage_2026_source",
            "layout": "full",
            "body": f"""## 路径指标提供局部信息，但不足以支持更新

2026 Legal flat 的总体 path MAE 为 **{float(legal_2026['path_mae']):.4%}**，exit regret 为 **{float(legal_2026['exit_regret']):.4%}**；Structured joint 分别为 **{float(structured_2026['path_mae']):.4%}** 和 **{float(structured_2026['exit_regret']):.4%}**。这说明 2026 fold 的路径预测误差与合法退出损失仍然显著，且 profile 间方向并不等价于可执行 alpha 的全面改善。Open/High/Low/Close 分通道误差保留在完整表中，避免把总体 MAE 当成唯一模型选择依据。""",
        },
        {
            "id": "checkpoint_vintage_2026_path_chart_block",
            "type": "chart",
            "chartId": "checkpoint_vintage_2026_path_mae_chart",
            "layout": "full",
        },
        {
            "id": "checkpoint_vintage_2026_pairwise_section",
            "type": "markdown",
            "sourceId": "checkpoint_vintage_2026_pairwise_source",
            "layout": "full",
            "body": f"""## 逐日配对结果显示胜出并不稳定

逐日配对把每个信号日的 2026 分数、排名和 Top3 执行结果与同日旧 checkpoint 对齐。Legal flat 相对 2023/2024/2025 的 Top3 alpha 胜出日占比分别为 **{legal_pairwise_text}**；Structured joint 分别为 **{structured_pairwise_text}**。这些比例是描述性稳定性证据，不是独立样本置信区间。原始逐日差异仍保留在 `daily_pairwise_deltas.csv`。""",
        },
        {
            "id": "checkpoint_vintage_2026_pairwise_chart_block",
            "type": "chart",
            "chartId": "checkpoint_vintage_2026_pairwise_win_chart",
            "layout": "full",
        },
        {
            "id": "checkpoint_vintage_2026_pairwise_table_block",
            "type": "table",
            "tableId": "checkpoint_vintage_2026_pairwise_table_asset",
            "layout": "full",
        },
        {
            "id": "checkpoint_vintage_2026_scope_section",
            "type": "markdown",
            "sourceId": "checkpoint_vintage_2026_source",
            "layout": "full",
            "body": """## 共享候选、标签和执行合同定义了可比边界

八组结果使用同一个冻结 candidate key/index、label material、entry/exit execution material、relative turnover material、费用合同和评价实现。每个旧 checkpoint 保留其训练时 normalization；2026 fold 的 normalization 在 `2026-01-05` 前拟合。候选集合来自冻结 base pack 的 `candidate_eligible` mask，不因未来标签完整性或 entry fill 再筛选。

这不是 QDP 更新或部署模型：QDP active_as_of 仍为 `2026-07-16`，full audit 为 `ok`，原 pack manifest 哈希保持不变，补充层为约 **0.821 GiB** 的独立 overlay。""",
        },
        {
            "id": "checkpoint_vintage_2026_gate_section",
            "type": "markdown",
            "sourceId": "checkpoint_vintage_2026_source",
            "layout": "full",
            "body": """## 预注册 freshness gate 的逐项判定

Gate 要求 2026 Top3 executable alpha 同时高于三个旧 checkpoint；Top1、Top5、Top10 中至少两项满足同一条件；Rank IC 或 Top3 opportunity alpha 提供排名支持；Exit regret 或 path MAE 提供路径支持；最后两个 profile 都必须通过。当前每个 profile 的主门槛、广度、排名和路径组合均未通过，overall 也为未通过。逐项的目标值、旧版本基线和方向见下表。""",
        },
        {
            "id": "checkpoint_vintage_2026_gate_table_block",
            "type": "table",
            "tableId": "checkpoint_vintage_2026_gate_table_asset",
            "layout": "full",
        },
        {
            "id": "checkpoint_vintage_2026_training_section",
            "type": "markdown",
            "sourceId": "checkpoint_vintage_2026_source",
            "layout": "full",
            "body": """## 2026 fold 训练严格沿用原研究协议

训练样本为 **7,773,480** 行，development candidate 为 **143,805** 个有监督行；2025-09-03..2025-12-31 的 purge 为 **239,081** 行、80 个交易日。两个 profile 均使用 seed 7、最多 10 epoch、patience 2、D2–D60 合法退出和原有损失权重，最佳 epoch 都为 1，并在 epoch 3 早停。它们是研究 fold checkpoint，不是 final/deployment model。""",
        },
        {
            "id": "checkpoint_vintage_2026_training_table_block",
            "type": "table",
            "tableId": "checkpoint_vintage_2026_training_table_asset",
            "layout": "full",
        },
        {
            "id": "checkpoint_vintage_2026_limitations_section",
            "type": "markdown",
            "sourceId": "checkpoint_vintage_2026_source",
            "layout": "full",
            "body": """## 窗口长度限制了不确定性解释

只有 48 个信号日，而每个信号的结果依赖最长 60 个未来交易日，因此不生成正式 moving-block 置信区间。逐日配对胜出比例和均值差异只用于描述跨 vintage 稳定性；不能把它们解释成独立日样本的显著性检验。实现收益窗口也不年化为有限资金 CAGR，且不把 3 月 20 日后的 score-only 日期混入任何收益指标。""",
        },
        {
            "id": "checkpoint_vintage_2026_next_steps_section",
            "type": "markdown",
            "sourceId": "checkpoint_vintage_2026_source",
            "layout": "full",
            "body": """## 下一步仍应保持研究用途而非自动晋级

当前证据支持冻结四年代比较和失败 gate，不支持选择 2026 checkpoint 进入部署或实盘。后续若继续研究，应在新的预注册窗口中延长可评价信号日、保持同一候选与执行合同，并把有限资金账户结果作为独立问题；任何模型晋级都需要重新定义门槛、验证窗口和风险预算。""",
        },
        {
            "id": "checkpoint_vintage_2026_questions_section",
            "type": "markdown",
            "layout": "full",
            "body": """## 待回答的问题

旧 checkpoint 在不同市场状态下的稳定性是否能解释 2026 的 profile 差异？如果扩展到更长、标签完整的窗口，Top3 alpha、路径误差和 exit regret 是否仍呈现同一排序？这些问题需要新的冻结合同和独立评价，不应从本次 48 日描述性结果外推。""",
        },
    ]

    return {
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "blocks": blocks,
        "sources": [comparison_source, pairwise_source],
        "datasets": {
            "checkpoint_vintage_2026_headline": headline,
            "checkpoint_vintage_2026_metrics": metrics,
            "checkpoint_vintage_2026_pairwise_summary": pairwise,
            "checkpoint_vintage_2026_pairwise_top3": [
                {
                    **row,
                    "older_vintage_label": str(row["older_vintage"]),
                }
                for row in top3_pairwise
            ],
            "checkpoint_vintage_2026_gate_evidence": gate,
            "checkpoint_vintage_2026_training": training,
        },
    }


def _assert_unique_ids(items: list[dict[str, Any]], label: str) -> None:
    ids = [str(item["id"]) for item in items]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate {label} IDs: {duplicates}")


def _validate_references(artifact: dict[str, Any]) -> dict[str, int]:
    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]
    datasets = snapshot["datasets"]
    collections = {
        "blocks": manifest["blocks"],
        "cards": manifest.get("cards", []),
        "charts": manifest.get("charts", []),
        "tables": manifest.get("tables", []),
        "sources": manifest.get("sources", []),
    }
    for label, items in collections.items():
        _assert_unique_ids(items, label)
    source_ids = {str(item["id"]) for item in collections["sources"]}
    card_ids = {str(item["id"]) for item in collections["cards"]}
    chart_ids = {str(item["id"]) for item in collections["charts"]}
    table_ids = {str(item["id"]) for item in collections["tables"]}
    for label in ("cards", "charts", "tables"):
        for item in collections[label]:
            dataset = item.get("dataset")
            if dataset is not None and str(dataset) not in datasets:
                raise RuntimeError(f"{label} {item['id']} references missing dataset {dataset}")
            source_id = item.get("sourceId")
            if source_id is not None and str(source_id) not in source_ids:
                raise RuntimeError(f"{label} {item['id']} references missing source {source_id}")
    for block in collections["blocks"]:
        source_id = block.get("sourceId")
        if source_id is not None and str(source_id) not in source_ids:
            raise RuntimeError(f"block {block['id']} references missing source {source_id}")
        for card_id in block.get("cardIds", []):
            if str(card_id) not in card_ids:
                raise RuntimeError(f"block {block['id']} references missing card {card_id}")
        chart_id = block.get("chartId")
        if chart_id is not None and str(chart_id) not in chart_ids:
            raise RuntimeError(f"block {block['id']} references missing chart {chart_id}")
        table_id = block.get("tableId")
        if table_id is not None and str(table_id) not in table_ids:
            raise RuntimeError(f"block {block['id']} references missing table {table_id}")
    if artifact.get("sources") != collections["sources"]:
        raise RuntimeError("top-level sources do not match manifest sources")
    for dataset_id, rows in datasets.items():
        if not isinstance(rows, list) or len(rows) > 2_000:
            raise RuntimeError(f"dataset {dataset_id} exceeds the bounded snapshot limit")
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f"dataset {dataset_id} contains a non-object row")
            for value in row.values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise RuntimeError(f"dataset {dataset_id} contains a non-finite value")
    return {key: len(value) for key, value in collections.items()} | {"datasets": len(datasets)}


def _assert_append_only(
    before: dict[str, Any], after: dict[str, Any], additions: dict[str, Any]
) -> None:
    if list(before) != list(after):
        raise RuntimeError("top-level artifact keys changed")
    for key in before:
        if key not in {"manifest", "snapshot", "sources"} and before[key] != after[key]:
            raise RuntimeError(f"top-level field changed: {key}")
    for key in before["manifest"]:
        if key in {"blocks", "cards", "charts", "tables", "sources"}:
            old = before["manifest"].get(key, [])
            new = after["manifest"].get(key, [])
            if new[: len(old)] != old:
                raise RuntimeError(f"existing manifest.{key} objects changed")
            if new[len(old) :] != additions[key]:
                raise RuntimeError(f"manifest.{key} was not extended by the expected suffix")
        elif before["manifest"][key] != after["manifest"][key]:
            raise RuntimeError(f"existing manifest field changed: {key}")
    for key in before["snapshot"]:
        if key != "datasets" and before["snapshot"][key] != after["snapshot"][key]:
            raise RuntimeError(f"existing snapshot field changed: {key}")
    old_datasets = before["snapshot"]["datasets"]
    new_datasets = after["snapshot"]["datasets"]
    old_ids = list(old_datasets)
    if list(new_datasets)[: len(old_ids)] != old_ids:
        raise RuntimeError("existing snapshot dataset order changed")
    for dataset_id, rows in old_datasets.items():
        if new_datasets[dataset_id] != rows:
            raise RuntimeError(f"existing dataset changed: {dataset_id}")
    new_ids = list(additions["datasets"])
    if list(new_datasets)[len(old_ids) :] != new_ids:
        raise RuntimeError("snapshot datasets were not appended in the expected order")
    for dataset_id, rows in additions["datasets"].items():
        if new_datasets[dataset_id] != rows:
            raise RuntimeError(f"new dataset differs from expected: {dataset_id}")
    if after["sources"][: len(before["sources"])] != before["sources"]:
        raise RuntimeError("existing top-level sources changed")
    if after["sources"][len(before["sources"]) :] != additions["sources"]:
        raise RuntimeError("top-level sources were not extended by the expected suffix")


def _collection_hashes(artifact: dict[str, Any]) -> dict[str, Any]:
    manifest = artifact["manifest"]
    result: dict[str, Any] = {}
    for key in ("cards", "charts", "tables", "blocks", "sources"):
        values = list(manifest.get(key, []))
        result[key] = {
            "count": len(values),
            "collection_sha256": _digest(values),
            "object_sha256": [_digest(value) for value in values],
        }
    datasets = artifact["snapshot"]["datasets"]
    result["datasets"] = {
        "count": len(datasets),
        "collection_sha256": _digest(datasets),
        "object_sha256": {key: _digest(value) for key, value in datasets.items()},
    }
    return result


def _suffix_is_present(artifact: dict[str, Any], additions: dict[str, Any]) -> bool:
    for key in ("cards", "charts", "tables", "blocks", "sources"):
        suffix = additions[key]
        if suffix and artifact["manifest"].get(key, [])[-len(suffix) :] != suffix:
            return False
    dataset_ids = list(additions["datasets"])
    existing_ids = list(artifact["snapshot"]["datasets"])
    if existing_ids[-len(dataset_ids) :] != dataset_ids:
        return False
    return all(
        artifact["snapshot"]["datasets"].get(key) == value
        for key, value in additions["datasets"].items()
    )


def main() -> int:
    _require_files(
        [ARTIFACT_PATH, SUMMARY_PATH, CONTRACT_PATH, STUDY_PATH, METRICS_PATH, PAIRWISE_PATH, DAILY_PAIRWISE_PATH]
    )
    original_bytes = ARTIFACT_PATH.read_bytes()
    artifact = json.loads(original_bytes.decode("utf-8"))
    summary = _read_json(SUMMARY_PATH)
    contract = _read_json(CONTRACT_PATH)
    study = _read_json(STUDY_PATH)
    checked = _validate_inputs(summary, contract, study)
    additions = _build_additions(summary, contract, study, checked)

    current_counts = {
        "cards": len(artifact["manifest"].get("cards", [])),
        "charts": len(artifact["manifest"].get("charts", [])),
        "tables": len(artifact["manifest"].get("tables", [])),
        "blocks": len(artifact["manifest"].get("blocks", [])),
        "sources": len(artifact["manifest"].get("sources", [])),
        "datasets": len(artifact["snapshot"].get("datasets", {})),
    }
    expected_after_counts = {
        key: EXPECTED_OLD_COUNTS[key] + len(additions[key])
        for key in ("cards", "charts", "tables", "blocks", "sources")
    } | {"datasets": EXPECTED_OLD_COUNTS["datasets"] + len(additions["datasets"])}
    if current_counts == expected_after_counts and _suffix_is_present(artifact, additions):
        validated_counts = _validate_references(artifact)
        audit = _read_json(AUDIT_PATH) if AUDIT_PATH.is_file() else {}
        if audit:
            disk_bytes = ARTIFACT_PATH.read_bytes()
            audit["after_size_bytes"] = len(disk_bytes)
            audit["after_sha256"] = _sha256_bytes(disk_bytes)
            audit["after_collection_hashes"] = _collection_hashes(artifact)
            audit["counts_after"] = validated_counts
            _write_json_atomic(AUDIT_PATH, audit)
        print(
            json.dumps(
                {
                    "status": "already_present",
                    "counts": current_counts,
                    "artifact_sha256": _sha256_bytes(ARTIFACT_PATH.read_bytes()),
                    "audit_refreshed": bool(audit),
                },
                ensure_ascii=False,
            )
        )
        return 0
    if current_counts != EXPECTED_OLD_COUNTS:
        raise RuntimeError(
            f"artifact prefix counts drifted; expected {EXPECTED_OLD_COUNTS}, got {current_counts}"
        )
    for key in ("cards", "charts", "tables", "blocks", "sources"):
        existing_ids = {str(row["id"]) for row in artifact["manifest"].get(key, [])}
        overlap = existing_ids.intersection(str(row["id"]) for row in additions[key])
        if overlap:
            raise RuntimeError(f"partial or conflicting append IDs in {key}: {sorted(overlap)}")
    overlap_datasets = set(artifact["snapshot"]["datasets"]).intersection(additions["datasets"])
    if overlap_datasets:
        raise RuntimeError(f"partial or conflicting dataset IDs: {sorted(overlap_datasets)}")

    before = copy.deepcopy(artifact)
    before_hashes = _collection_hashes(before)
    for key in ("cards", "charts", "tables", "blocks", "sources"):
        artifact["manifest"].setdefault(key, []).extend(additions[key])
    artifact["sources"].extend(additions["sources"])
    artifact["snapshot"]["datasets"].update(additions["datasets"])
    _assert_append_only(before, artifact, additions)
    after_counts = _validate_references(artifact)
    serialized = (json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(serialized) <= len(original_bytes):
        raise RuntimeError("artifact did not grow after append")

    after_hashes = _collection_hashes(artifact)
    audit = {
        "schema_version": 1,
        "status": "passed",
        "artifact_path": str(ARTIFACT_PATH),
        "comparison_summary_path": str(SUMMARY_PATH),
        "comparison_contract_sha256": checked["contract_sha256"],
        "study_contract_sha256": checked["study_contract_sha256"],
        "base_pack_manifest_sha256": checked["base_pack_sha256"],
        "qdp_active_manifest_sha256": checked["qdp_active_manifest_sha256"],
        "overlay_size_bytes": checked["overlay_size_bytes"],
        "overlay_size_gib": checked["overlay_size_bytes"] / 1024**3,
        "before_size_bytes": len(original_bytes),
        "after_size_bytes": len(serialized),
        "before_sha256": _sha256_bytes(original_bytes),
        "after_sha256": _sha256_bytes(serialized),
        "before_collection_hashes": before_hashes,
        "after_collection_hashes": after_hashes,
        "all_existing_objects_unchanged": True,
        "new_objects_are_exact_suffixes": True,
        "new_counts": {
            key: len(additions[key]) for key in ("cards", "charts", "tables", "blocks", "sources")
        },
        "new_dataset_ids": list(additions["datasets"]),
        "counts_before": current_counts,
        "counts_after": after_counts,
        "signal_window": ["2026-01-05", "2026-03-19"],
        "signal_date_count": 48,
        "candidate_count": 143_840,
        "symbol_count": 3_034,
        "score_coverage": 1.0,
        "execution_coverage": 1.0,
        "overall_2026_freshness_gate_pass": bool(summary.get("overall_2026_freshness_gate_pass")),
        "historical_gate_failure_preserved": True,
        "authorized_extension": True,
    }

    _write_json_atomic(ARTIFACT_PATH, artifact)
    reloaded = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    if reloaded != artifact:
        raise RuntimeError("atomically written artifact differs from validated payload")
    _assert_append_only(before, reloaded, additions)
    _validate_references(reloaded)
    written_bytes = ARTIFACT_PATH.read_bytes()
    audit["after_size_bytes"] = len(written_bytes)
    audit["after_sha256"] = _sha256_bytes(written_bytes)
    _write_json_atomic(AUDIT_PATH, audit)
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
