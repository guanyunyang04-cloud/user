from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
RUNTIME = ROOT / "brain/skills/workspace-brain/scripts/brain_runtime.py"
REFLECTION = ROOT / "brain/skills/workspace-brain/scripts/reflection_learning.py"


def load_reflection_module():
    spec = importlib.util.spec_from_file_location("workspace_brain_reflection_learning", REFLECTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_trace(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json")
    with tmp:
        json.dump(payload, tmp, ensure_ascii=False)
    return Path(tmp.name)


class ReflectionLearningTest(unittest.TestCase):
    def test_required_step_skipped_generates_completion_gate_candidate(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "继续实施计划，到最后启动长训练",
            "planned_steps": [
                {"id": "train", "title": "启动训练", "expected_outcome": "pid logged", "required": True}
            ],
            "events": [{"type": "step_skipped", "step_id": "train", "summary": "ended before training"}],
            "final_state": {"completed": True, "skipped_steps": ["train"], "unresolved_blockers": [], "user_nudges": [], "verification": []},
        }

        payload = module.analyze_trace(trace)

        self.assertEqual(payload["analysis_mode"], "structured_trace")
        self.assertTrue(payload["learning_candidates"])
        candidate = payload["learning_candidates"][0]
        self.assertEqual(candidate["target_layer"], "execution_completion_gate")
        self.assertEqual(candidate["confidence"], "high")
        self.assertIn("train", candidate["supporting_events"][0]["step_id"])
        self.assertTrue(candidate["suggested_tests"])

    def test_tool_failure_followed_by_manual_workaround_generates_tool_contract_gap(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "登记 path_policy evidence",
            "planned_steps": [{"id": "writeback", "title": "run writeback-plan", "expected_outcome": "route found", "required": True}],
            "events": [
                {"type": "tool_failure", "step_id": "writeback", "summary": "writeback-plan searched wrong evidence domain", "command": "writeback-plan"},
                {"type": "manual_workaround", "step_id": "writeback", "summary": "manually read path_policy study summary"},
            ],
            "final_state": {"completed": True, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": [], "verification": []},
        }

        payload = module.analyze_trace(trace)

        candidates = payload["learning_candidates"]
        self.assertIn("tool_contract_gap", {candidate["target_layer"] for candidate in candidates})
        candidate = next(candidate for candidate in candidates if candidate["target_layer"] == "tool_contract_gap")
        self.assertIn("anti_overfit_check", candidate)
        self.assertIn("manual_workaround", {event["type"] for event in candidate["supporting_events"]})

    def test_user_nudge_about_unstarted_training_uses_trace_structure(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "执行计划，最后启动训练",
            "planned_steps": [{"id": "train", "title": "启动训练", "expected_outcome": "pid logged", "required": True}],
            "events": [{"type": "user_nudge", "step_id": "train", "summary": "还没启动训练，需要继续实施计划"}],
            "final_state": {"completed": True, "skipped_steps": ["train"], "unresolved_blockers": [], "user_nudges": ["还没启动训练"], "verification": []},
        }

        payload = module.analyze_trace(trace)

        candidate = payload["learning_candidates"][0]
        self.assertEqual(candidate["target_layer"], "execution_completion_gate")
        self.assertEqual(candidate["confidence"], "high")

    def test_completed_trace_with_passing_verification_has_no_learning_candidates(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "完成普通脑区维护",
            "planned_steps": [{"id": "test", "title": "run tests", "expected_outcome": "pass", "required": True}],
            "events": [{"type": "verification_passed", "step_id": "test", "summary": "tests passed"}],
            "final_state": {"completed": True, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": [], "verification": [{"name": "tests", "status": "passed"}]},
        }

        payload = module.analyze_trace(trace)

        self.assertEqual(payload["learning_candidates"], [])
        self.assertEqual(payload["next_actions"], ["no_runtime_learning_needed"])

    def test_one_off_environment_failure_does_not_create_persistent_learning(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "run check",
            "planned_steps": [{"id": "check", "title": "run check", "expected_outcome": "pass", "required": True}],
            "events": [{"type": "tool_failure", "step_id": "check", "summary": "network hiccup", "evidence": "one_off_environment_failure"}],
            "final_state": {"completed": False, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": [], "verification": []},
        }

        payload = module.analyze_trace(trace)

        self.assertEqual(payload["learning_candidates"], [])
        self.assertEqual(payload["next_actions"], ["no_persistent_learning"])

    def test_review_cli_accepts_trace_json(self) -> None:
        trace_path = write_trace(
            {
                "task": "继续实施计划，到最后启动长训练",
                "planned_steps": [{"id": "train", "title": "启动训练", "expected_outcome": "pid logged", "required": True}],
                "events": [{"type": "step_skipped", "step_id": "train", "summary": "ended before training"}],
                "final_state": {"completed": True, "skipped_steps": ["train"], "unresolved_blockers": [], "user_nudges": [], "verification": []},
            }
        )
        try:
            result = subprocess.run(
                [PYTHON, str(RUNTIME), "review", "--cwd", str(ROOT), "--trace-json", str(trace_path), "--json"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
        finally:
            trace_path.unlink(missing_ok=True)
        payload = json.loads(result.stdout)

        self.assertEqual(payload["analysis_mode"], "structured_trace")
        self.assertTrue(payload["learning_candidates"])

    def test_freeform_review_is_low_confidence_fallback(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "review",
                "--cwd",
                str(ROOT),
                "--observation",
                "用户指出 Start-Sleep 被用作长任务轮询",
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["analysis_mode"], "freeform_fallback")
        self.assertEqual(payload["evidence_gaps"], ["structured_trace_missing"])
        self.assertTrue(payload["learning_candidates"])
        self.assertEqual(payload["learning_candidates"][0]["confidence"], "low")

    def test_reflection_template_cli_outputs_trace_schema(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "reflection-template", "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("planned_steps", payload)
        self.assertIn("events", payload)
        self.assertIn("final_state", payload)


if __name__ == "__main__":
    unittest.main()
