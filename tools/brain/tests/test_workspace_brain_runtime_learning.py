from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
SKILL = ROOT / "brain/skills/workspace-brain/SKILL.md"
RUNTIME = ROOT / "brain/skills/workspace-brain/scripts/brain_runtime.py"


class WorkspaceBrainRuntimeLearningTest(unittest.TestCase):
    def test_brain_runtime_writes_learning_proposal_outside_core_brain(self) -> None:
        tmp_root = ROOT / "daily_research/output/test_learning_proposal_project"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        subprocess.run(
            [PYTHON, str(RUNTIME), "init", "--cwd", str(tmp_root), "--brain-id", "learning_demo"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )

        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "proposal",
                "--cwd",
                str(tmp_root),
                "--title",
                "branch blocker learning",
                "--trigger",
                "mutation attempted away from main",
                "--evidence",
                "git branch was feature",
                "--recommendation",
                "add a preflight blocker",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertTrue(Path(payload["json_path"]).exists())
        self.assertTrue(Path(payload["markdown_path"]).exists())
        self.assertIn("brain/output/agent_learning", payload["json_path"].replace("\\", "/"))
        proposal_payload = json.loads(Path(payload["json_path"]).read_text(encoding="utf-8"))
        self.assertEqual(proposal_payload["schema_version"], 3)
        self.assertIn("requires_user_confirmation", proposal_payload["authority"])
        self.assertEqual(proposal_payload["severity"], "info")
        self.assertEqual(proposal_payload["owner_brain"], "learning_demo")
        self.assertEqual(proposal_payload["writeback_target"], "brain/references/")
        self.assertTrue(proposal_payload["requires_user_confirmation"])
        self.assertEqual(proposal_payload["proposal_creation_policy"], "auto_create_low_risk_proposed_status")
        self.assertEqual(proposal_payload["proposal_status"], "proposed")
        self.assertTrue(proposal_payload["implementation_requires_user_confirmation"])
        self.assertIn("lesson", proposal_payload)
        self.assertIn("root_cause", proposal_payload)
        self.assertIn("supporting_events", proposal_payload)
        self.assertIn("anti_overfit_check", proposal_payload)

    def test_brain_runtime_review_detects_user_correction_learning_candidate(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "review",
                "--cwd",
                str(ROOT),
                "--task",
                "脑区中的规则明明写了，为什么还是没有执行？",
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

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["analysis_mode"], "freeform_fallback")
        self.assertTrue(payload["reflection_review_required"])
        self.assertTrue(payload["learning_candidates"])
        candidate = payload["learning_candidates"][0]
        self.assertIn(candidate["target_layer"], {"workflow_selector", "capsule_contract", "skill", "tests_guard"})
        self.assertTrue(candidate["requires_user_confirmation"])
        self.assertEqual(candidate["proposal_creation_policy"], "auto_create_low_risk_proposed_status")
        self.assertEqual(candidate["proposal_status"], "proposed")
        self.assertTrue(candidate["implementation_requires_user_confirmation"])

    def test_brain_runtime_review_detects_run_evidence_domain_mismatch(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "review",
                "--cwd",
                str(ROOT),
                "--task",
                "writeback-plan 对 path_policy run tag 查 continuous_policy/studies",
                "--observation",
                "工具被手工绕过：证据域错路由导致 path_policy/studies 证据查成 continuous_policy/studies",
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        candidates = payload["learning_candidates"]
        self.assertTrue(candidates)
        self.assertIn("run_evidence_resolver", {candidate["target_layer"] for candidate in candidates})

    def test_brain_runtime_review_detects_brain_rule_selector_miss(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "review",
                "--cwd",
                str(ROOT),
                "--task",
                "修改脑区规则",
                "--observation",
                "capsule selected brain_handoff for a brain rule mutation",
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        candidates = payload["learning_candidates"]
        self.assertTrue(candidates)
        self.assertIn("workflow_selector", {candidate["target_layer"] for candidate in candidates})

    def test_brain_runtime_review_does_not_flag_completed_generic_adapter_retirement(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "review",
                "--cwd",
                str(ROOT),
                "--task",
                "彻底清理脑区 Generic Workflow Adapter",
                "--observation",
                (
                    "已移除 generic workflow fallback adapter hot path；registry/playbook/selector/capsule/"
                    "skill 文档与 evidence registry 已清理；local skills 重新成为通用方法论真源"
                ),
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["learning_candidates"], [])
        self.assertEqual(payload["next_actions"], ["no_agent_learning_needed"])

    def test_brain_runtime_proposal_queue_can_be_listed_and_marked(self) -> None:
        tmp_root = ROOT / "daily_research/output/test_learning_queue_project"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        subprocess.run(
            [PYTHON, str(RUNTIME), "init", "--cwd", str(tmp_root), "--brain-id", "learning_queue"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        proposal = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "proposal",
                "--cwd",
                str(tmp_root),
                "--title",
                "workflow selector learning",
                "--trigger",
                "mutate plan title selected " + "writing_" + "plan",
                "--evidence",
                "capsule workflow mismatch",
                "--recommendation",
                "make intent participate in selector",
                "--target-layer",
                "workflow_selector",
                "--suggested-test",
                "capsule mutate plan title selects " + "executing_" + "plan",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        proposal_payload = json.loads(proposal.stdout)
        listed = subprocess.run(
            [PYTHON, str(RUNTIME), "list-proposals", "--cwd", str(tmp_root)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        listed_payload = json.loads(listed.stdout)

        self.assertEqual(len(listed_payload["proposals"]), 1)
        self.assertEqual(listed_payload["proposals"][0]["status"], "proposed")

        marked = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "mark-proposal",
                "--cwd",
                str(tmp_root),
                "--proposal-id",
                proposal_payload["proposal_id"],
                "--status",
                "approved",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        marked_payload = json.loads(marked.stdout)

        self.assertEqual(marked_payload["status"], "ok")
        self.assertEqual(marked_payload["proposal"]["status"], "approved")

    def test_brain_runtime_review_detects_planned_training_step_skipped(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "review",
                "--cwd",
                str(ROOT),
                "--task",
                "继续实施计划，到最后启动长训练",
                "--observation",
                "计划包含训练但 agent 结束任务，需要用户提示继续实施计划",
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertTrue(payload["learning_candidates"])
        targets = {candidate["target_layer"] for candidate in payload["learning_candidates"]}
        self.assertIn("execution_completion_gate", targets)

    def test_brain_runtime_review_outputs_meta_question_candidates(self) -> None:
        trace_path = ROOT / "daily_research/output/test_trace_meta_question.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(
                {
                    "task": "判断 agent 是否遗漏元问题",
                    "planned_steps": [
                        {"id": "review", "title": "review closure", "expected_outcome": "meta issue found", "required": True}
                    ],
                    "events": [
                        {
                            "type": "human_feedback_overrode_tool_clear",
                            "step_id": "review",
                            "summary": "audit was clear but user feedback showed the closure judgment was wrong",
                            "evidence": "agent-meta-audit clear; user says meta-cognition missing",
                        }
                    ],
                    "final_state": {
                        "completed": True,
                        "skipped_steps": [],
                        "unresolved_blockers": [],
                        "user_nudges": ["工具 clear 但人类反馈不 clear"],
                        "verification": [],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
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

        self.assertTrue(payload["meta_question_candidates"])
        self.assertIn("create_agent_learning_proposal", payload["next_actions"])
        self.assertNotIn("ask_user_for_evolution", payload["next_actions"])
        self.assertTrue(payload["meta_question_candidates"][0]["auto_create_proposal"])
        self.assertTrue(payload["meta_question_candidates"][0]["implementation_requires_user_confirmation"])

    def test_brain_runtime_proposal_schema_v3_has_lifecycle_fields(self) -> None:
        tmp_root = ROOT / "daily_research/output/test_learning_schema_v3_project"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        subprocess.run(
            [PYTHON, str(RUNTIME), "init", "--cwd", str(tmp_root), "--brain-id", "learning_schema_v3"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )

        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "proposal",
                "--cwd",
                str(tmp_root),
                "--title",
                "agent meta protocol learning",
                "--trigger",
                "learning opportunity missed",
                "--evidence",
                "agent skipped the required meta pass",
                "--recommendation",
                "add agent meta protocol contract",
                "--target-layer",
                "agent_meta_protocol",
                "--suggested-test",
                "capsule includes agent_meta",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)
        proposal_payload = json.loads(Path(payload["json_path"]).read_text(encoding="utf-8"))

        self.assertEqual(proposal_payload["schema_version"], 3)
        self.assertEqual(proposal_payload["target_layer"], "agent_meta_protocol")
        self.assertEqual(proposal_payload["lifecycle_status"], "proposed")
        self.assertIn("verification_required", proposal_payload)
        self.assertIn("writeback_route", proposal_payload)
        self.assertEqual(proposal_payload["proposal_creation_policy"], "auto_create_low_risk_proposed_status")
        self.assertEqual(proposal_payload["proposal_status"], "proposed")
        self.assertTrue(proposal_payload["implementation_requires_user_confirmation"])

    def test_brain_runtime_mark_proposal_accepts_verified(self) -> None:
        tmp_root = ROOT / "daily_research/output/test_learning_verified_project"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        subprocess.run(
            [PYTHON, str(RUNTIME), "init", "--cwd", str(tmp_root), "--brain-id", "learning_verified"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        proposal = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "proposal",
                "--cwd",
                str(tmp_root),
                "--title",
                "verified lifecycle",
                "--trigger",
                "proposal needs verified state",
                "--evidence",
                "mark-proposal",
                "--recommendation",
                "allow verified",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        proposal_payload = json.loads(proposal.stdout)

        marked = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "mark-proposal",
                "--cwd",
                str(tmp_root),
                "--proposal-id",
                proposal_payload["proposal_id"],
                "--status",
                "verified",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        marked_payload = json.loads(marked.stdout)

        self.assertEqual(marked_payload["proposal"]["status"], "verified")



if __name__ == "__main__":
    unittest.main()
