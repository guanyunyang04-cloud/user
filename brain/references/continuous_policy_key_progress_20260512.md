# Continuous Policy Key Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current r54 continuous-policy research front from a positive but blocked screening result into a diagnosable r55 line that attacks cash timing, source release, reduce/exit quality, and abnormal-run reliability without changing the live policy_v5b production anchor.

**Architecture:** Keep production execution immutable and build the next research line inside `daily_research/continuous_policy`. First make failed trials inspectable, then add a behavior bottleneck report, then introduce an r55 cash-timing/release-intent controller with tests before any new safe screening. Large rewrites are allowed inside continuous-policy modules, but promotion/live/default artifacts stay untouched until evidence gates pass.

**Tech Stack:** Python 3.10 in `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`, pandas, numpy, torch, unittest/pytest, existing `daily_research.continuous_policy` protocol/study runner.

---

## Scope And Non-Negotiables

Facts:
- Current branch must remain `main`.
- Current live/default strategy is `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- `continuous_policy` is research/shadow only.
- r54 best completed screening trial was positive on composite score but blocked by insufficient training evidence, negative cash timing, weak reduce/exit quality, and one abnormal trial exit.

Inferences:
- The highest-leverage work is not another broad confirmatory run. It is making r54/r55 failure modes measurable and directly training the missing intent surfaces.
- Allocator closure and receiver headroom are no longer the primary bottlenecks; cash timing and sell/reduce/exit semantics are.

Assumptions:
- The immediate objective is key research progress, not live promotion.
- The active production artifact must remain unchanged unless a later explicit promotion decision is made.

Hard boundaries:
- Do not create or switch branches.
- Do not modify `daily_research/output/active_execution_strategy.json`.
- Do not treat malformed/failed trial artifacts as completed evidence.
- Do not use `latest_*` artifacts to judge strategy truth when an explicit tag exists.

## File Map

Create:
- `daily_research/continuous_policy/protocol_artifact_diagnostics.py`  
  Single-purpose inspection of protocol/study artifacts, including malformed JSON and abnormal exit diagnostics.
- `daily_research/continuous_policy/behavior_bottleneck_report.py`  
  Summarizes cash timing, source release, reduce/exit, exposure, and training-edge blockers from explicit protocol summaries.
- `daily_research/continuous_policy/semantic_budget_intent.py`  
  Pure functions for cash-timing intent and release/reduce/exit intent derived from target-weight deltas and risk/opportunity signals.
- `daily_research/continuous_policy/tests/test_protocol_artifact_diagnostics.py`
- `daily_research/continuous_policy/tests/test_behavior_bottleneck_report.py`
- `daily_research/continuous_policy/tests/test_semantic_budget_intent.py`
- `daily_research/brain/references/r55_cash_timing_release_controller_status_20260512.md`

Modify:
- `daily_research/continuous_policy/run_self_optimizing_study.py`  
  Preserve artifact-health details for failed trials; register r55 profile; tighten r55 resource gate.
- `daily_research/continuous_policy/model_seq_v3.py`  
  Register `alpha_result_value_budget_split_v45`; export r55 intent columns; route r55 loss terms to day-set/native losses.
- `daily_research/continuous_policy/native_allocation.py`  
  Add directional cash-timing and source-release intent losses.
- `daily_research/continuous_policy/portfolio_simulator.py`  
  Under `allocation_intent_v2_mode`, derive sell/reduce/exit hints from negative target-delta intent and expose realized intent diagnostics.
- `daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py`  
  Add r55 registry, scoring, gate, and simulator-contract tests.
- `daily_research/continuous_policy/tests/test_allocation_closure_diagnostics.py`  
  Extend loss/gate tests for r55 cash/release intent.
- `daily_research/brain/continuous_policy_design_contract.md`  
  Record r55 contract after code passes tests.
- `daily_research/brain/state_center.md` and `daily_research/brain/knowledge_center.md`  
  Record final evidence only after a real screening run completes.

Validation commands:
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_protocol_artifact_diagnostics.py -q`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_behavior_bottleneck_report.py -q`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_semantic_budget_intent.py -q`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_allocation_core_v2.py daily_research/continuous_policy/tests/test_allocation_closure_diagnostics.py daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py -q`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`
- `git diff -- daily_research/output/active_execution_strategy.json`

---

### Task 1: Artifact Health Diagnostics

**Files:**
- Create: `daily_research/continuous_policy/protocol_artifact_diagnostics.py`
- Create: `daily_research/continuous_policy/tests/test_protocol_artifact_diagnostics.py`

- [ ] **Step 1: Write failing tests for JSON health and protocol diagnostics**

Create `daily_research/continuous_policy/tests/test_protocol_artifact_diagnostics.py`:

```python
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from daily_research.continuous_policy.protocol_artifact_diagnostics import (
    inspect_json_artifact,
    summarize_protocol_artifacts,
)


class ProtocolArtifactDiagnosticsTest(unittest.TestCase):
    def test_inspect_json_artifact_reports_malformed_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "protocol_summary.json"
            path.write_text('{"run_tag": "bad", "items": [1, 2,}', encoding="utf-8")

            health = inspect_json_artifact(path)

        self.assertTrue(health["exists"])
        self.assertFalse(health["parse_ok"])
        self.assertGreater(health["size_bytes"], 0)
        self.assertIn("parse_error", health)
        self.assertIn("tail_preview", health)
        self.assertEqual(health["kind"], "json")

    def test_inspect_json_artifact_reports_valid_top_level_keys(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "protocol_summary.json"
            path.write_text(json.dumps({"run_tag": "ok", "promotion_gate": {"status": "shadow_only"}}), encoding="utf-8")

            health = inspect_json_artifact(path)

        self.assertTrue(health["parse_ok"])
        self.assertEqual(health["top_level_keys"], ["promotion_gate", "run_tag"])
        self.assertEqual(health["run_tag"], "ok")

    def test_summarize_protocol_artifacts_marks_failed_exit_as_diagnostic_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protocol_root = root / "protocols" / "trial_03"
            protocol_root.mkdir(parents=True)
            (protocol_root / "protocol_summary.json").write_text(
                json.dumps(
                    {
                        "run_tag": "trial_03",
                        "training_evidence": {"status": "insufficient", "best_epoch": 8, "completed_epochs": 8},
                        "promotion_gate": {"status": "shadow_only", "failed_checks": ["cash_timing_quality_1d"]},
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_protocol_artifacts(protocol_root, exit_code=3221226505)

        self.assertEqual(summary["exit_code"], 3221226505)
        self.assertFalse(summary["completed_evidence"])
        self.assertEqual(summary["protocol_summary"]["run_tag"], "trial_03")
        self.assertEqual(summary["training_evidence_status"], "insufficient")
        self.assertIn("abnormal_exit", summary["diagnostic_flags"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_protocol_artifact_diagnostics.py -q
```

Expected: FAIL because `daily_research.continuous_policy.protocol_artifact_diagnostics` does not exist.

- [ ] **Step 3: Implement the diagnostics module**

Create `daily_research/continuous_policy/protocol_artifact_diagnostics.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ABNORMAL_WINDOWS_EXIT_CODES = {3221226505, 3221225477, 3221225786}


def _preview(text: str, *, limit: int = 500) -> str:
    cleaned = text.replace("\x00", "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[-limit:]


def inspect_json_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    payload: dict[str, Any] = {
        "kind": "json",
        "path": str(artifact_path.resolve()) if artifact_path.exists() else str(artifact_path),
        "exists": artifact_path.exists(),
        "parse_ok": False,
        "size_bytes": 0,
        "top_level_keys": [],
    }
    if not artifact_path.exists():
        payload["missing_reason"] = "not_found"
        return payload

    raw = artifact_path.read_text(encoding="utf-8", errors="replace")
    payload["size_bytes"] = artifact_path.stat().st_size
    payload["tail_preview"] = _preview(raw)
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        payload["parse_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    if not isinstance(parsed, dict):
        payload["parse_error"] = f"top_level_type={type(parsed).__name__}"
        return payload

    payload["parse_ok"] = True
    payload["top_level_keys"] = sorted(str(key) for key in parsed.keys())
    payload["run_tag"] = str(parsed.get("run_tag", ""))
    payload["promotion_status"] = str((parsed.get("promotion_gate", {}) or {}).get("status", ""))
    return payload


def summarize_protocol_artifacts(protocol_root: str | Path, *, exit_code: int = 0) -> dict[str, Any]:
    root = Path(protocol_root)
    protocol_summary_path = root / "protocol_summary.json"
    protocol_health = inspect_json_artifact(protocol_summary_path)
    protocol_summary: dict[str, Any] = {}
    if protocol_health.get("parse_ok"):
        protocol_summary = json.loads(protocol_summary_path.read_text(encoding="utf-8"))

    training_evidence = dict(protocol_summary.get("training_evidence", {}) or {})
    promotion_gate = dict(protocol_summary.get("promotion_gate", {}) or {})
    diagnostic_flags: list[str] = []
    if int(exit_code) != 0:
        diagnostic_flags.append("nonzero_exit")
    if int(exit_code) in ABNORMAL_WINDOWS_EXIT_CODES:
        diagnostic_flags.append("abnormal_exit")
    if not protocol_health.get("parse_ok"):
        diagnostic_flags.append("protocol_summary_unparseable")

    return {
        "protocol_root": str(root.resolve()) if root.exists() else str(root),
        "exit_code": int(exit_code),
        "completed_evidence": int(exit_code) == 0 and bool(protocol_health.get("parse_ok")),
        "diagnostic_flags": diagnostic_flags,
        "protocol_summary_health": protocol_health,
        "protocol_summary": protocol_summary,
        "training_evidence_status": str(training_evidence.get("status", "")),
        "best_epoch": int(training_evidence.get("best_epoch", 0) or 0),
        "completed_epochs": int(training_evidence.get("completed_epochs", 0) or 0),
        "promotion_status": str(promotion_gate.get("status", "")),
        "promotion_failed_checks": list(promotion_gate.get("failed_checks", []) or []),
    }
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_protocol_artifact_diagnostics.py -q
```

Expected: PASS.

---

### Task 2: Preserve Failed-Trial Evidence Without Counting It

**Files:**
- Modify: `daily_research/continuous_policy/run_self_optimizing_study.py`
- Modify: `daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py`

- [ ] **Step 1: Write a focused failing test for failed trial health**

Append to `PortfolioDailyStrategyContractsTest` in `daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py`:

```python
    def test_failed_trial_result_preserves_artifact_health_without_completed_evidence(self) -> None:
        from daily_research.continuous_policy.run_self_optimizing_study import _build_failed_trial_result

        result = _build_failed_trial_result(
            trial_id=3,
            trial_tag="study__trial_03",
            phase="screening",
            source_trial_tag="",
            trial_config={"epochs": 8, "min_epochs": 6},
            protocol_summary_path="H:/missing/protocol_summary.json",
            exit_code=3221226505,
            exception_message="protocol exited with code 3221226505",
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.composite_score, -999.0)
        self.assertIn("trial_execution_failed", result.failed_checks)
        self.assertEqual(result.primary_metrics["exit_code"], 3221226505)
        self.assertEqual(result.primary_metrics["completed_evidence"], 0.0)
        self.assertIn("abnormal_exit", result.score_breakdown["artifact_health"]["diagnostic_flags"])
```

- [ ] **Step 2: Run the targeted test and verify it fails**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py::PortfolioDailyStrategyContractsTest::test_failed_trial_result_preserves_artifact_health_without_completed_evidence -q
```

Expected: FAIL because `_build_failed_trial_result` does not exist.

- [ ] **Step 3: Add `_build_failed_trial_result` and call it from failure paths**

In `daily_research/continuous_policy/run_self_optimizing_study.py`, import:

```python
from daily_research.continuous_policy.protocol_artifact_diagnostics import summarize_protocol_artifacts
```

Add this helper near `TrialResult` helpers:

```python
def _build_failed_trial_result(
    *,
    trial_id: int,
    trial_tag: str,
    phase: str,
    source_trial_tag: str,
    trial_config: dict[str, Any],
    protocol_summary_path: str,
    exit_code: int,
    exception_message: str,
) -> TrialResult:
    health = summarize_protocol_artifacts(Path(protocol_summary_path).parent, exit_code=exit_code)
    primary_metrics = {
        "exit_code": float(exit_code),
        "completed_evidence": 1.0 if bool(health.get("completed_evidence")) else 0.0,
        "protocol_summary_parse_ok": 1.0
        if bool((health.get("protocol_summary_health", {}) or {}).get("parse_ok"))
        else 0.0,
        "best_epoch": float(health.get("best_epoch", 0) or 0),
        "completed_epochs": float(health.get("completed_epochs", 0) or 0),
    }
    return TrialResult(
        trial_id=int(trial_id),
        trial_tag=str(trial_tag),
        status="failed",
        phase=str(phase),
        role="",
        source_trial_tag=str(source_trial_tag or ""),
        trial_config=dict(trial_config),
        protocol_summary_path=str(protocol_summary_path),
        performance_score=-999.0,
        stability_score=-999.0,
        composite_score=-999.0,
        score_breakdown={"artifact_health": health, "exception_message": str(exception_message)},
        primary_metrics=primary_metrics,
        promotion_status="failed",
        failed_checks=["trial_execution_failed"],
        gate_pass_ratio=0.0,
        passed_check_count=0,
        total_check_count=1,
    )
```

Replace both inline failed `TrialResult(...)` construction blocks in screening and confirmatory exception handlers with `_build_failed_trial_result(...)`.

- [ ] **Step 4: Run the focused test and then full continuous-policy tests**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py::PortfolioDailyStrategyContractsTest::test_failed_trial_result_preserves_artifact_health_without_completed_evidence -q
```

Expected: PASS.

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_protocol_artifact_diagnostics.py daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py -q
```

Expected: PASS.

---

### Task 3: Behavior Bottleneck Report

**Files:**
- Create: `daily_research/continuous_policy/behavior_bottleneck_report.py`
- Create: `daily_research/continuous_policy/tests/test_behavior_bottleneck_report.py`

- [ ] **Step 1: Write failing tests for bottleneck classification**

Create `daily_research/continuous_policy/tests/test_behavior_bottleneck_report.py`:

```python
from __future__ import annotations

import unittest

from daily_research.continuous_policy.behavior_bottleneck_report import build_behavior_bottleneck_report


class BehaviorBottleneckReportTest(unittest.TestCase):
    def test_report_prioritizes_cash_timing_and_sell_intent_blockers(self) -> None:
        protocol_summary = {
            "run_tag": "trial_02",
            "evaluation": {
                "continuous_policy_metrics": {
                    "annual_return": 0.153993,
                    "monthly_return_mean": 0.010115,
                    "max_drawdown": -0.106873,
                },
                "continuity_metrics": {
                    "cash_timing_quality_1d": -0.168248,
                    "reduce_success_rate_5d": 0.0,
                    "exit_timeliness_rate_5d": 0.0,
                    "portfolio_daily_source_target_count": 0.0,
                    "portfolio_daily_source_realized_sell_rate": 0.0,
                    "portfolio_daily_actual_cash_weight_mean": 0.188532,
                    "portfolio_daily_exposure_utilization": 1.082835,
                    "portfolio_daily_target_sum_gap": 0.004812,
                    "intent_translation_conflict_rate": 0.0,
                },
            },
            "training_evidence": {"status": "insufficient", "best_epoch": 8, "completed_epochs": 8},
            "promotion_gate": {
                "status": "shadow_only",
                "failed_checks": [
                    "training_evidence_sufficient",
                    "reduce_success_rate_5d",
                    "exit_timeliness_rate_5d",
                    "cash_timing_quality_1d",
                ],
            },
        }

        report = build_behavior_bottleneck_report(protocol_summary)

        self.assertEqual(report["run_tag"], "trial_02")
        self.assertEqual(report["primary_blocker"], "cash_timing_negative")
        self.assertIn("sell_intent_dead", report["blockers"])
        self.assertIn("best_epoch_at_edge", report["blockers"])
        self.assertGreater(report["progress_assets"]["allocation_closure_score"], 0.90)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_behavior_bottleneck_report.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement report builder**

Create `daily_research/continuous_policy/behavior_bottleneck_report.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _section(summary: dict[str, Any], *keys: str) -> dict[str, Any]:
    value: Any = summary
    for key in keys:
        value = (value or {}).get(key, {})
    return dict(value or {}) if isinstance(value, dict) else {}


def _float(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(mapping.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def build_behavior_bottleneck_report(protocol_summary: dict[str, Any]) -> dict[str, Any]:
    evaluation_metrics = _section(protocol_summary, "evaluation", "continuous_policy_metrics")
    continuity = _section(protocol_summary, "evaluation", "continuity_metrics")
    training = _section(protocol_summary, "training_evidence")
    promotion = _section(protocol_summary, "promotion_gate")
    failed_checks = list(promotion.get("failed_checks", []) or [])

    cash_timing = _float(continuity, "cash_timing_quality_1d")
    reduce_success = _float(continuity, "reduce_success_rate_5d")
    exit_timeliness = _float(continuity, "exit_timeliness_rate_5d")
    source_count = _float(continuity, "portfolio_daily_source_target_count")
    source_sell_rate = _float(continuity, "portfolio_daily_source_realized_sell_rate")
    exposure_utilization = _float(continuity, "portfolio_daily_exposure_utilization")
    target_sum_gap = _float(continuity, "portfolio_daily_target_sum_gap")
    intent_conflict = _float(continuity, "intent_translation_conflict_rate")
    best_epoch = int(training.get("best_epoch", 0) or 0)
    completed_epochs = int(training.get("completed_epochs", 0) or 0)

    blockers: list[str] = []
    if cash_timing < -0.05 or "cash_timing_quality_1d" in failed_checks:
        blockers.append("cash_timing_negative")
    if source_count < 1.0 or source_sell_rate < 0.20:
        blockers.append("sell_intent_dead")
    if reduce_success < 0.45 or "reduce_success_rate_5d" in failed_checks:
        blockers.append("reduce_quality_weak")
    if exit_timeliness < 0.45 or "exit_timeliness_rate_5d" in failed_checks:
        blockers.append("exit_timeliness_weak")
    if str(training.get("status", "")) != "sufficient":
        blockers.append("training_evidence_insufficient")
    if completed_epochs > 0 and best_epoch > max(completed_epochs - 2, 0):
        blockers.append("best_epoch_at_edge")

    allocation_closure_score = max(0.0, min(1.0, exposure_utilization)) * max(0.0, 1.0 - min(target_sum_gap, 1.0))
    if intent_conflict == 0.0:
        allocation_closure_score = max(allocation_closure_score, 0.90)

    primary_blocker = blockers[0] if blockers else "none"
    return {
        "run_tag": str(protocol_summary.get("run_tag", "")),
        "primary_blocker": primary_blocker,
        "blockers": blockers,
        "metrics": {
            "annual_return": _float(evaluation_metrics, "annual_return"),
            "monthly_return_mean": _float(evaluation_metrics, "monthly_return_mean"),
            "max_drawdown": _float(evaluation_metrics, "max_drawdown"),
            "cash_timing_quality_1d": cash_timing,
            "reduce_success_rate_5d": reduce_success,
            "exit_timeliness_rate_5d": exit_timeliness,
            "portfolio_daily_source_target_count": source_count,
            "portfolio_daily_source_realized_sell_rate": source_sell_rate,
            "portfolio_daily_exposure_utilization": exposure_utilization,
            "portfolio_daily_target_sum_gap": target_sum_gap,
            "intent_translation_conflict_rate": intent_conflict,
        },
        "progress_assets": {
            "allocation_closure_score": allocation_closure_score,
            "intent_translation_clean": intent_conflict <= 0.01,
            "exposure_closed": exposure_utilization >= 0.60 and target_sum_gap <= 0.05,
        },
        "next_focus": [
            "cash_timing_direction",
            "source_release_intent",
            "reduce_exit_realization",
            "strict_resume_training_evidence",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a continuous-policy behavior bottleneck report.")
    parser.add_argument("--protocol-summary-json", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    protocol_path = Path(args.protocol_summary_json)
    summary = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = build_behavior_bottleneck_report(summary)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the report tests**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_behavior_bottleneck_report.py -q
```

Expected: PASS.

---

### Task 4: Semantic Budget Intent Pure Functions

**Files:**
- Create: `daily_research/continuous_policy/semantic_budget_intent.py`
- Create: `daily_research/continuous_policy/tests/test_semantic_budget_intent.py`

- [ ] **Step 1: Write failing tests for cash and release intent**

Create `daily_research/continuous_policy/tests/test_semantic_budget_intent.py`:

```python
from __future__ import annotations

import unittest

import pandas as pd

from daily_research.continuous_policy.semantic_budget_intent import (
    derive_cash_timing_intent,
    derive_release_intent_from_target_delta,
)


class SemanticBudgetIntentTest(unittest.TestCase):
    def test_cash_timing_intent_increases_for_downside_and_falls_for_deploy_opportunity(self) -> None:
        defensive = derive_cash_timing_intent(
            risk_score=0.80,
            deploy_score=0.15,
            benchmark_downside=0.70,
            alpha_alignment=0.10,
        )
        opportunistic = derive_cash_timing_intent(
            risk_score=0.20,
            deploy_score=0.85,
            benchmark_downside=0.05,
            alpha_alignment=0.75,
        )

        self.assertGreater(defensive, 0.60)
        self.assertLess(opportunistic, 0.30)

    def test_release_intent_uses_negative_target_delta_for_held_names_only(self) -> None:
        frame = pd.DataFrame(
            {
                "stock": ["KEEP", "REDUCE", "EXIT", "NEW"],
                "current_weight": [0.20, 0.20, 0.12, 0.00],
                "portfolio_daily_target_delta_intent": [0.00, -0.04, -0.12, -0.02],
                "exit_hazard": [0.10, 0.20, 0.80, 0.90],
                "reduce_quality": [0.10, 0.70, 0.60, 0.90],
            }
        ).set_index("stock")

        result = derive_release_intent_from_target_delta(frame, deadband=0.01)

        self.assertEqual(result.loc["KEEP", "release_intent_action"], "hold")
        self.assertEqual(result.loc["REDUCE", "release_intent_action"], "reduce")
        self.assertEqual(result.loc["EXIT", "release_intent_action"], "exit")
        self.assertEqual(result.loc["NEW", "release_intent_action"], "hold")
        self.assertGreater(result.loc["EXIT", "release_intent_score"], result.loc["REDUCE", "release_intent_score"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_semantic_budget_intent.py -q
```

Expected: FAIL because `semantic_budget_intent.py` does not exist.

- [ ] **Step 3: Implement pure intent helpers**

Create `daily_research/continuous_policy/semantic_budget_intent.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def derive_cash_timing_intent(
    *,
    risk_score: float,
    deploy_score: float,
    benchmark_downside: float,
    alpha_alignment: float,
) -> float:
    raw = (
        0.46 * float(risk_score)
        + 0.28 * max(float(risk_score) - float(deploy_score), 0.0)
        + 0.26 * float(benchmark_downside)
        - 0.24 * float(deploy_score)
        - 0.18 * float(alpha_alignment)
    )
    return float(np.clip(raw, 0.0, 1.0))


def derive_release_intent_from_target_delta(policy_frame: pd.DataFrame, *, deadband: float = 0.003) -> pd.DataFrame:
    result = pd.DataFrame(index=policy_frame.index)
    current_weight = pd.to_numeric(policy_frame.get("current_weight", 0.0), errors="coerce").fillna(0.0)
    target_delta = pd.to_numeric(
        policy_frame.get("portfolio_daily_target_delta_intent", 0.0),
        errors="coerce",
    ).fillna(0.0)
    exit_hazard = pd.to_numeric(policy_frame.get("exit_hazard", 0.0), errors="coerce").fillna(0.0)
    reduce_quality = pd.to_numeric(policy_frame.get("reduce_quality", 0.0), errors="coerce").fillna(0.0)

    held = current_weight > float(deadband)
    release_size = (-target_delta).clip(lower=0.0)
    release_score = (release_size / current_weight.clip(lower=float(deadband))).clip(lower=0.0, upper=1.0)
    release_score = (0.62 * release_score + 0.22 * exit_hazard.clip(0.0, 1.0) + 0.16 * reduce_quality.clip(0.0, 1.0)).where(held, 0.0)

    action = pd.Series("hold", index=policy_frame.index, dtype=object)
    reduce_mask = held & (target_delta < -float(deadband))
    exit_mask = reduce_mask & ((exit_hazard >= 0.62) | (release_size >= current_weight * 0.72))
    action.loc[reduce_mask] = "reduce"
    action.loc[exit_mask] = "exit"

    result["release_intent_score"] = release_score.astype(float)
    result["release_intent_action"] = action
    result["release_intent_delta"] = (-release_size).where(held, 0.0).astype(float)
    return result
```

- [ ] **Step 4: Run tests**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_semantic_budget_intent.py -q
```

Expected: PASS.

---

### Task 5: r55 Registry And Gates

**Files:**
- Modify: `daily_research/continuous_policy/run_self_optimizing_study.py`
- Modify: `daily_research/continuous_policy/model_seq_v3.py`
- Modify: `daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py`

- [ ] **Step 1: Add failing registry test**

Append a r55 case to `test_search_profile_registry_contracts_cover_research_lines` in `daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py`:

```python
            {
                "profile": "split_heads_portfolio_daily_cash_timing_release_controller_r55",
                "loss_profile": "alpha_result_value_budget_split_v45",
                "objective_profile": "end_to_end_allocation_layer_v1",
                "base_equals": {"epochs": 16, "min_epochs": 10, "batch_size": 1},
                "forbid_true_solver": True,
                "multi_equals": {
                    "action_total": 0.0,
                    "duration_total": 0.0,
                    "portfolio_cvxpy_convex_allocation_total": 0.0,
                    "portfolio_full_universe_convex_allocation_total": 0.0,
                },
                "multi_greater": {
                    "portfolio_day_set_native_allocation_vector_total": 4.20,
                    "portfolio_cash_timing_directional_total": 0.0,
                    "portfolio_source_release_intent_total": 0.0,
                },
                "no_full_universe_train_solver": True,
                "auto_resource_profile": "balanced",
            },
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py::PortfolioDailyStrategyContractsTest::test_search_profile_registry_contracts_cover_research_lines -q
```

Expected: FAIL because r55 profile and loss profile are not registered.

- [ ] **Step 3: Register r55 study profile**

In `daily_research/continuous_policy/run_self_optimizing_study.py`, add to `SEARCH_PROFILE_BASE_TRIALS`:

```python
    "split_heads_portfolio_daily_cash_timing_release_controller_r55": {
        "label_preset": "holdcash_v3",
        "decoder_profile": "budget_v3",
        "loss_profile": "alpha_result_value_budget_split_v45",
        "budget_semantics": BUDGET_SEMANTICS_ALLOCATION_LAYER,
        "budget_calibration": BUDGET_CALIBRATION_END_TO_END_ALLOCATION_LAYER,
        "budget_objective": "result_value_v10",
        "alpha_prior_source": "active_execution_strategy",
        "daily_head_layout": "split_v2",
        "learning_rate": 7.0e-5,
        "hidden_dim": 192,
        "sequence_layers": 2,
        "daily_hidden_dim": 128,
        "dropout": 0.30,
        "daily_dropout": 0.16,
        "batch_size": 1,
        "epochs": 16,
        "min_epochs": 10,
    },
```

Add to `SEARCH_PROFILE_DEFAULT_OBJECTIVES`:

```python
    "split_heads_portfolio_daily_cash_timing_release_controller_r55": "end_to_end_allocation_layer_v1",
```

Add to `RESOURCE_GATED_SEARCH_PROFILES`:

```python
    "split_heads_portfolio_daily_cash_timing_release_controller_r55": {
        "min_completed_screening": 1,
        "source_count_floor": 1.0,
        "source_sell_rate_floor": 0.20,
        "cash_timing_floor": -0.02,
        "drawdown_floor": -0.135,
        "monthly_return_floor": 0.003,
        "annual_return_floor": 0.12,
        "receiver_unrealized_cap": 0.025,
        "exposure_utilization_floor": 0.60,
        "actual_cash_idle_cap": 0.20,
        "actual_cash_weight_cap": 0.45,
        "cash_funded_deploy_floor": 0.08,
        "unused_receiver_headroom_cap": 0.22,
        "target_sum_gap_cap": 0.05,
        "cash_first_source_gate": True,
    },
```

- [ ] **Step 4: Register r55 loss profile**

In `daily_research/continuous_policy/model_seq_v3.py`, add `alpha_result_value_budget_split_v45` by copying v44 as the base and changing only the new r55 intent weights:

```python
LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v45"] = {
    **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v44"],
    "multi_objective_loss_weights": {
        **LOSS_PROFILE_CONFIGS["alpha_result_value_budget_split_v44"]["multi_objective_loss_weights"],
        "portfolio_day_set_native_allocation_vector_total": 4.35,
        "portfolio_cash_timing_directional_total": 0.42,
        "portfolio_source_release_intent_total": 0.54,
        "portfolio_reduce_exit_intent_total": 0.48,
        "action_total": 0.0,
        "duration_total": 0.0,
        "portfolio_cvxpy_convex_allocation_total": 0.0,
        "portfolio_full_universe_convex_allocation_total": 0.0,
    },
}
```

If `LOSS_PROFILE_CONFIGS` is built as a literal dictionary rather than mutated after definition, add the same dictionary entry inside the literal at the existing loss-profile section.

- [ ] **Step 5: Run registry test**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py::PortfolioDailyStrategyContractsTest::test_search_profile_registry_contracts_cover_research_lines -q
```

Expected: PASS.

---

### Task 6: r55 Intent Loss Terms

**Files:**
- Modify: `daily_research/continuous_policy/native_allocation.py`
- Modify: `daily_research/continuous_policy/tests/test_allocation_closure_diagnostics.py`

- [ ] **Step 1: Write failing test for r55 loss terms**

Append to `AllocationClosureDiagnosticsTest` in `daily_research/continuous_policy/tests/test_allocation_closure_diagnostics.py`:

```python
    def test_r55_native_loss_penalizes_wrong_cash_timing_and_dead_source_release(self) -> None:
        from daily_research.continuous_policy.native_allocation import _portfolio_native_allocation_vector_loss

        row_count = 4
        targets = {
            "date_code": torch.zeros(row_count, dtype=torch.float32),
            "current_weight": torch.tensor([0.25, 0.20, 0.0, 0.0], dtype=torch.float32),
            "portfolio_daily_receiver_candidate_mask": torch.tensor([0, 0, 1, 1], dtype=torch.float32),
            "portfolio_daily_source_candidate_mask": torch.tensor([1, 1, 0, 0], dtype=torch.float32),
            "portfolio_daily_receiver_executable_candidate": torch.tensor([0, 0, 1, 1], dtype=torch.float32),
            "portfolio_daily_source_executable_candidate": torch.tensor([1, 1, 0, 0], dtype=torch.float32),
            "gross_exposure_target": torch.full((row_count,), 0.80, dtype=torch.float32),
            "turnover_budget": torch.full((row_count,), 1.0, dtype=torch.float32),
            "max_position_weight_target": torch.full((row_count,), 0.30, dtype=torch.float32),
            "budget_cash_timing_signal_target": torch.full((row_count,), 0.90, dtype=torch.float32),
            "portfolio_daily_allocation_cash_deployment_target": torch.full((row_count,), 0.20, dtype=torch.float32),
            "portfolio_daily_allocation_net_utility_target": torch.full((row_count,), 0.80, dtype=torch.float32),
            "portfolio_daily_allocation_final_objective": torch.full((row_count,), 0.80, dtype=torch.float32),
            "portfolio_daily_unified_receiver_score": torch.tensor([0, 0, 0.55, 0.52], dtype=torch.float32),
            "portfolio_daily_unified_source_score": torch.tensor([0.90, 0.86, 0, 0], dtype=torch.float32),
        }
        wrong_outputs = {
            "portfolio_daily_allocation_weight_logit": torch.tensor([4.0, 4.0, 4.0, 4.0], dtype=torch.float32),
            "portfolio_daily_cash_reserve_logit": torch.full((row_count,), -5.0, dtype=torch.float32),
            "portfolio_daily_allocation_risk_buffer_logit": torch.full((row_count,), -5.0, dtype=torch.float32),
        }
        defensive_outputs = {
            **wrong_outputs,
            "portfolio_daily_cash_reserve_logit": torch.full((row_count,), 5.0, dtype=torch.float32),
            "portfolio_daily_allocation_weight_logit": torch.tensor([-3.0, -3.0, 0.0, 0.0], dtype=torch.float32),
        }

        wrong_terms = _portfolio_native_allocation_vector_loss(wrong_outputs, targets, return_terms=True)
        defensive_terms = _portfolio_native_allocation_vector_loss(defensive_outputs, targets, return_terms=True)

        self.assertIn("cash_timing_directional_loss", wrong_terms)
        self.assertIn("source_release_intent_loss", wrong_terms)
        self.assertGreater(float(wrong_terms["cash_timing_directional_loss"]), float(defensive_terms["cash_timing_directional_loss"]))
        self.assertGreater(float(wrong_terms["source_release_intent_loss"]), float(defensive_terms["source_release_intent_loss"]))
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_allocation_closure_diagnostics.py::AllocationClosureDiagnosticsTest::test_r55_native_loss_penalizes_wrong_cash_timing_and_dead_source_release -q
```

Expected: FAIL because terms are not emitted.

- [ ] **Step 3: Add r55 terms inside native allocation loss**

In `daily_research/continuous_policy/native_allocation.py`, add term names to exported term lists:

```python
"cash_timing_directional_loss",
"source_release_intent_loss",
"reduce_exit_intent_loss",
```

Inside the day loop of `_portfolio_native_allocation_vector_loss`, compute:

```python
cash_timing_directional_loss = torch.square(
    torch.relu(cash_timing_day - raw_cash_reserve)
    + torch.relu(deploy_day - (1.0 - raw_cash_reserve))
)
source_release_pressure = torch.clamp(source_score[day_mask].mean(), 0.0, 1.0)
held_weight_day = current_weight_day.sum()
sell_flow_day = torch.relu(current_weight_day - target_weight_day).sum() / torch.clamp(held_weight_day, min=1.0e-6)
source_release_intent_loss = torch.square(torch.relu(source_release_pressure * 0.55 - sell_flow_day))
reduce_exit_intent_loss = torch.square(torch.relu(source_release_pressure * cash_timing_day * 0.35 - sell_flow_day))
```

Append them to `term_values` and add them to the total:

```python
total = (
    total
    + 0.42 * cash_timing_directional_loss
    + 0.54 * source_release_intent_loss
    + 0.48 * reduce_exit_intent_loss
)
```

Preserve existing r54 behavior for profiles that do not assign these terms positive weights.

- [ ] **Step 4: Run native loss tests**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_allocation_closure_diagnostics.py::AllocationClosureDiagnosticsTest::test_r55_native_loss_penalizes_wrong_cash_timing_and_dead_source_release -q
```

Expected: PASS.

---

### Task 7: Simulator Uses Negative Target-Delta Intent For Sell/Reduce/Exit Hints

**Files:**
- Modify: `daily_research/continuous_policy/portfolio_simulator.py`
- Modify: `daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py`

- [ ] **Step 1: Add failing simulator contract test**

Append to `PortfolioDailyStrategyContractsTest`:

```python
    def test_allocation_intent_v2_negative_delta_creates_release_actions(self) -> None:
        from daily_research.continuous_policy.portfolio_simulator import HoldingState, PortfolioState

        state = PortfolioState(
            cash_weight=0.70,
            holdings={"HELD": HoldingState(weight=0.30, entry_price=10.0, peak_price=11.0)},
            max_positions=4,
            max_position_weight=0.50,
            turnover_limit=1.00,
        )
        policy = pd.DataFrame(
            {
                "stock": ["HELD", "RECV"],
                "current_weight": [0.30, 0.0],
                "action_label": ["hold", "open"],
                "portfolio_daily_target_weight_intent": [0.10, 0.20],
                "portfolio_daily_target_delta_intent": [-0.20, 0.20],
                "allocation_intent_v2_mode": [1.0, 1.0],
                "exit_hazard": [0.75, 0.0],
                "reduce_quality": [0.80, 0.0],
                "receiver": [0, 1],
                "source": [1, 0],
                "portfolio_daily_receiver_executable_candidate": [0, 1],
                "portfolio_daily_source_executable_candidate": [1, 0],
            }
        ).set_index("stock")

        result = state.step(
            date="2026-05-12",
            prices=pd.Series({"HELD": 10.0, "RECV": 10.0}),
            policy_frame=policy,
            global_targets={
                "gross_exposure_target": 0.50,
                "turnover_budget": 1.00,
                "max_position_weight_target": 0.50,
                "cash_reserve_target": 0.05,
                "allocation_intent_v2_mode": 1.0,
            },
            budget_semantics="allocation_layer_v1",
            budget_calibration="end_to_end_allocation_layer_v1",
        )

        self.assertGreater(result.diagnostics["allocation_intent_release_target_count"], 0.0)
        self.assertGreater(result.sell_turnover, 0.0)
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py::PortfolioDailyStrategyContractsTest::test_allocation_intent_v2_negative_delta_creates_release_actions -q
```

Expected: FAIL because simulator does not expose `allocation_intent_release_target_count` or does not realize sell turnover from negative delta.

- [ ] **Step 3: Integrate semantic intent into simulator**

In `daily_research/continuous_policy/portfolio_simulator.py`, import:

```python
from daily_research.continuous_policy.semantic_budget_intent import derive_release_intent_from_target_delta
```

Inside `PortfolioState.step`, after `allocation_intent_v2_mode` is resolved and before action-derived allocation logic consumes `action_names`, add:

```python
if allocation_intent_v2_mode and "portfolio_daily_target_delta_intent" in policy.columns:
    release_intent = derive_release_intent_from_target_delta(policy, deadband=deadband)
    release_actions = release_intent["release_intent_action"].reindex(policy.index).fillna("hold")
    release_mask = release_actions.isin({"reduce", "exit"})
    action_names = action_names.where(~release_mask, release_actions)
    allocation_intent_release_target_count = float(release_mask.sum())
else:
    allocation_intent_release_target_count = 0.0
```

Add to diagnostics:

```python
"allocation_intent_release_target_count": float(allocation_intent_release_target_count),
```

Keep lifecycle action labels as reporting hints; the target-weight delta remains the primary allocation path under r54/r55.

- [ ] **Step 4: Run simulator contract test**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py::PortfolioDailyStrategyContractsTest::test_allocation_intent_v2_negative_delta_creates_release_actions -q
```

Expected: PASS.

---

### Task 8: r55 Dry Run And Safe Screening

**Files:**
- Modify after evidence: `daily_research/brain/references/r55_cash_timing_release_controller_status_20260512.md`
- Modify after evidence: `daily_research/brain/continuous_policy_design_contract.md`
- Modify after evidence: `daily_research/brain/state_center.md`
- Modify after evidence: `daily_research/brain/knowledge_center.md`

- [ ] **Step 1: Run full unit/regression suite for continuous policy**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests/test_protocol_artifact_diagnostics.py daily_research/continuous_policy/tests/test_behavior_bottleneck_report.py daily_research/continuous_policy/tests/test_semantic_budget_intent.py daily_research/continuous_policy/tests/test_allocation_core_v2.py daily_research/continuous_policy/tests/test_allocation_closure_diagnostics.py daily_research/continuous_policy/tests/test_portfolio_daily_strategy_contracts.py -q
```

Expected: PASS.

- [ ] **Step 2: Verify production anchor is untouched**

Run:

```powershell
git diff -- daily_research/output/active_execution_strategy.json
```

Expected: no output.

- [ ] **Step 3: Run r55 dry run**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_cash_timing_release_controller_r55 --study-tag self_opt_study_r55_cash_timing_release_controller_dryrun_20260512_01 --trial-count 2 --disable-confirmatory --resource-profile safe --dry-run
```

Expected:
- dry-run exits 0;
- generated study plan includes `alpha_result_value_budget_split_v45`;
- `confirmatory_enabled=false`;
- `full_universe_train_solver_effective=false`.

- [ ] **Step 4: Run bounded safe screening**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_self_optimizing_study --search-profile split_heads_portfolio_daily_cash_timing_release_controller_r55 --study-tag self_opt_study_r55_cash_timing_release_controller_screening_safe_20260512_01 --trial-count 2 --disable-confirmatory --resource-profile safe
```

Expected:
- no abnormal exit;
- all failed trials, if any, include artifact-health diagnostics;
- at least one completed screening trial writes `protocol_summary.json`;
- no confirmatory runs.

- [ ] **Step 5: Generate behavior bottleneck report for the best completed r55 trial**

Use the best completed `protocol_summary.json` path from the r55 `study_summary.json`.

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.behavior_bottleneck_report --protocol-summary-json H:/new_tdx64/PYPlugins/user/daily_research/output/continuous_policy/protocols/self_opt_study_r55_cash_timing_release_controller_screening_safe_20260512_01__trial_01/protocol_summary.json --output-json H:/new_tdx64/PYPlugins/user/daily_research/output/continuous_policy/analysis/r55_cash_timing_release_controller_trial_01_bottleneck_report.json
```

Expected:
- report exists;
- `primary_blocker` is explicit;
- `progress_assets.exposure_closed` is true or the report explains why not;
- blockers distinguish cash timing from sell-intent/reduce/exit.

- [ ] **Step 6: Decide whether strict resume is justified**

Strict resume is allowed only if:
- r55 has no abnormal exit;
- best completed trial has `training_evidence.status="insufficient"` only because `best_epoch_at_edge`;
- composite score is positive or materially better than r54 on blockers;
- cash timing is no worse than r54 trial 02;
- source/reduce/exit are no longer dead.

If allowed, run same explicit tag with larger epoch budget:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.continuous_policy.run_continuous_policy_protocol --pool-name learned_all_a --benchmark 000300.SH --data-source tq --pool-rebalance-days 21 --pool-adv-window 20 --max-universe-size 1200 --train-start-date 20240102 --train-end-date 20251231 --eval-start-date 20260102 --eval-end-date 20260512 --shadow-start-date 20260428 --shadow-end-date 20260512 --transaction-cost-bps 3.0 --slippage-bps 7.0 --sell-tax-bps 10.0 --random-seed 7 --skip-multiplier 2.0 --execution-semantics semantic_preserving_v1 --budget-semantics allocation_layer_v1 --budget-calibration end_to_end_allocation_layer_v1 --budget-objective result_value_v10 --alpha-prior-source active_execution_strategy --label-preset holdcash_v3 --trainer-backend formal_torch_seq_v3 --decoder-profile budget_v3 --loss-profile alpha_result_value_budget_split_v45 --epochs 32 --min-epochs 24 --batch-size 1 --learning-rate 7.0e-5 --hidden-dim 192 --sequence-layers 2 --daily-hidden-dim 128 --daily-head-layout split_v2 --dropout 0.30 --daily-dropout 0.16 --early-stop-patience 10 --resume-mode strict --tag self_opt_study_r55_cash_timing_release_controller_screening_safe_20260512_01__trial_01
```

Expected:
- training evidence becomes sufficient or gives a concrete new blocker;
- best epoch is not at edge;
- protocol remains shadow only unless all gates pass later.

---

### Task 9: Evidence Writeback

**Files:**
- Create/modify: `daily_research/brain/references/r55_cash_timing_release_controller_status_20260512.md`
- Modify: `daily_research/brain/continuous_policy_design_contract.md`
- Modify: `daily_research/brain/state_center.md`
- Modify: `daily_research/brain/knowledge_center.md`

- [ ] **Step 1: Write r55 status reference**

Create `daily_research/brain/references/r55_cash_timing_release_controller_status_20260512.md` with:

```markdown
# r55 Cash Timing Release Controller Status

Date: `2026-05-12`

## Scope
- Research line: `r55 cash-timing release-intent controller`.
- Branch: `main`.
- Production boundary: no live/default/promotion switch and no change to `daily_research/output/active_execution_strategy.json`.
- Execution boundary: safe screening only unless explicit later promotion decision.

## Code Contract
- Search profile: `split_heads_portfolio_daily_cash_timing_release_controller_r55`.
- Loss profile: `alpha_result_value_budget_split_v45`.
- Core change: target-weight intent remains the primary allocation path, while negative target-delta intent now produces explicit source/reduce/exit intent diagnostics.

## Evidence
- At plan-writing time, no r55 dry run or r55 screening has been executed.
- Create this status file only after Task 8 completes.
- Use explicit run tags from `study_summary.json`; if a run did not execute, write `not executed under this plan`.

## Verdict
- Before evidence exists: `r55 is planned only and has no strategy verdict`.
- After Task 8: replace this sentence with the observed shadow-only verdict and the exact blocker list from the r55 study summary.
```

Replace the empty evidence bullets with actual tag names and metrics from r55 outputs before finalizing.

- [ ] **Step 2: Update design contract and state after evidence**

Add a short r55 entry to `daily_research/brain/continuous_policy_design_contract.md`:

```markdown
### r55 cash-timing release-intent controller
- Status: research / shadow-only.
- Search profile: `split_heads_portfolio_daily_cash_timing_release_controller_r55`.
- Loss profile: `alpha_result_value_budget_split_v45`.
- Purpose: test whether explicit cash-timing direction and negative target-delta release intent can repair r54 cash timing and reduce/exit deadness while preserving allocation closure.
- Evidence reference: `daily_research/brain/references/r55_cash_timing_release_controller_status_20260512.md`.
```

Update `daily_research/brain/state_center.md` only with completed evidence. Use explicit tags; do not use `latest_*`.

- [ ] **Step 3: Run doc guard**

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check
```

Expected: PASS.

---

## Promotion And Stop Gates

Do not enter confirmatory unless all are true:
- no abnormal exit in r55 safe screening;
- all completed trials have parseable `protocol_summary.json`;
- best trial has `training_evidence.status="sufficient"` or a strict-resume plan has already fixed edge-only insufficiency;
- `cash_timing_quality_1d >= -0.02` for screening and target `>= 0.0` before confirmatory;
- `portfolio_daily_source_target_count >= 1.0` and `portfolio_daily_source_realized_sell_rate >= 0.20` when source release is required;
- `reduce_success_rate_5d` and `exit_timeliness_rate_5d` are non-zero and improving versus r54;
- `portfolio_daily_actual_cash_weight_mean <= 0.45`;
- `portfolio_daily_target_sum_gap <= 0.05`;
- `intent_translation_conflict_rate <= 0.01`;
- `max_drawdown >= -0.135`.

Stop immediately and write back a failed verdict if:
- cash timing becomes more negative than r54 trial 02 while allocation closure remains clean;
- r55 still produces zero source/reduce/exit actions;
- abnormal exit reproduces with parseable but failed artifacts;
- active production artifact changes unexpectedly.

## Self-Review

Spec coverage:
- Deep plan provided for large rewrite/replacement.
- Current project facts are respected: main branch only, policy_v5b active, continuous_policy shadow.
- r54 blockers are directly mapped to tasks: abnormal exit, cash timing, training evidence, source/reduce/exit.

Plan hygiene scan:
- No deferred implementation markers remain.
- Future evidence fields are explicitly represented as `not executed under this plan` until real tagged artifacts exist.

Type consistency:
- New helper names are stable across tests and implementation: `inspect_json_artifact`, `summarize_protocol_artifacts`, `build_behavior_bottleneck_report`, `derive_cash_timing_intent`, `derive_release_intent_from_target_delta`.
- r55 names are stable: `split_heads_portfolio_daily_cash_timing_release_controller_r55` and `alpha_result_value_budget_split_v45`.
