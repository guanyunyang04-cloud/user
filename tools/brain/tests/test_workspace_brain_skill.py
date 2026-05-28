from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.brain import skill_install


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
SKILL = ROOT / "brain/skills/workspace-brain/SKILL.md"
RUNTIME = ROOT / "brain/skills/workspace-brain/scripts/brain_runtime.py"


class WorkspaceBrainSkillTest(unittest.TestCase):
    def test_skill_frontmatter_and_body_are_procedural(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("---\nname: workspace-brain"))
        self.assertIn("description: Use the workspace main brain", text)
        self.assertIn("脑区", text)
        self.assertIn("agent learning", text.lower())
        self.assertIn("-m tools.brain.workflow capsule", text)
        self.assertIn("-m tools.brain.workflow route", text)
        self.assertIn("-m tools.brain.workflow query", text)
        self.assertIn("writeback-plan", text)
        self.assertIn("brain_runtime.py", text)
        self.assertIn("brain_runtime.py health", text)
        self.assertNotIn("r10-r52", text)

    def test_skill_uses_program_family_run_evidence_model(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("research_programs", text)
        self.assertIn("study_families", text)
        self.assertIn("run_tags", text)
        self.assertIn("program/family/run", text)
        self.assertNotIn("Prefer explicit study tags", text)

    def test_skill_mentions_research_scope_grade_alignment_before_final(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("scope-grade alignment", text)
        self.assertIn("universe_scope", text)
        self.assertIn("seed count", text)
        self.assertIn("gate result", text)
        self.assertIn("diagnostic-vs-evidence-grade", text)

    def test_skill_does_not_offer_generic_workflow_fallback(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertNotIn("API Fallback", text)
        self.assertNotIn("workflow-guide", text)
        self.assertNotIn("--intent " + "long_" + "task", text)
        self.assertNotIn("Superpowers plugin body", text)

    def test_install_dry_run_does_not_copy_skill(self) -> None:
        target_root = ROOT / "daily_research/output/test_skill_install_target"
        before_exists = target_root.exists()
        result = subprocess.run(
            [PYTHON, "-m", "tools.brain.skill_install", "--dry-run", "--target-root", str(target_root)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["skill_count"], 1)
        self.assertEqual(payload["skills"][0]["skill"], "workspace-brain")
        self.assertEqual(target_root.exists(), before_exists)

    def test_install_check_reports_hashes_and_missing_target(self) -> None:
        target_root = ROOT / "daily_research/output/test_skill_check_target"
        if target_root.exists():
            shutil.rmtree(target_root)
        result = subprocess.run(
            [PYTHON, "-m", "tools.brain.skill_install", "--check", "--target-root", str(target_root)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)
        item = payload["skills"][0]

        self.assertTrue(payload["check"])
        self.assertFalse(payload["all_in_sync"])
        self.assertEqual(item["skill"], "workspace-brain")
        self.assertTrue(item["source_sha256"])
        self.assertEqual(item["target_sha256"], "")
        self.assertFalse(item["target_exists"])
        self.assertFalse(item["in_sync"])

    def test_install_reports_post_install_hash_sync(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            result = subprocess.run(
                [PYTHON, "-m", "tools.brain.skill_install", "--install", "--target-root", raw_tmp],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertTrue(payload["installed"])
        self.assertTrue(payload["all_in_sync"])
        self.assertTrue(payload["skills"][0]["target_exists"])
        self.assertTrue(payload["skills"][0]["in_sync"])

    def test_install_ignores_generated_python_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_root = Path(raw_tmp)
            source_root = tmp_root / "source"
            target_root = tmp_root / "target"
            skill = source_root / "workspace-brain"
            cache_dir = skill / "scripts" / "__pycache__"
            cache_dir.mkdir(parents=True)
            (skill / "SKILL.md").write_text("---\nname: workspace-brain\ndescription: Use when testing\n---\n", encoding="utf-8")
            (skill / "scripts" / "brain_runtime.py").write_text("print('ok')\n", encoding="utf-8")
            (cache_dir / "brain_runtime.cpython-311.pyc").write_bytes(b"compiled")

            with patch.object(skill_install, "PROJECT_SKILLS_ROOT", source_root):
                payload = skill_install.install_project_skills(target_root)

            copied_cache = target_root / "workspace-brain" / "scripts" / "__pycache__"
            copied_cache_exists = copied_cache.exists()

        self.assertTrue(payload["all_in_sync"])
        self.assertFalse(copied_cache_exists)

    def test_workspace_brain_source_has_no_python_cache_files(self) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "brain/skills/workspace-brain"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        tracked_paths = [line.strip().replace("\\", "/") for line in tracked.stdout.splitlines() if line.strip()]

        self.assertFalse(any("__pycache__" in path or path.endswith(".pyc") for path in tracked_paths), tracked_paths)
        self.assertFalse((ROOT / "brain/skills/workspace-brain/scripts/__pycache__").exists())

    def test_brain_runtime_detects_project_without_brain(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_root = Path(raw_tmp) / "test_no_brain_project"
            tmp_root.mkdir(parents=True)
            result = subprocess.run(
                [PYTHON, str(RUNTIME), "detect", "--cwd", str(tmp_root)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertFalse(payload["has_brain"])
        self.assertFalse(payload["has_brain_tools"])
        self.assertIn("init_brain_available", payload["next_actions"])

    def test_brain_runtime_init_minimal_project(self) -> None:
        tmp_root = ROOT / "daily_research/output/test_minimal_brain_project"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)

        result = subprocess.run(
            [PYTHON, str(RUNTIME), "init", "--cwd", str(tmp_root), "--brain-id", "demo_project"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertTrue((tmp_root / "brain/brain_manifest.json").exists())
        self.assertTrue((tmp_root / "brain/state_center.md").read_text(encoding="utf-8").startswith("# Demo Project 状态中枢"))

        detect = subprocess.run(
            [PYTHON, str(RUNTIME), "detect", "--cwd", str(tmp_root)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        detected = json.loads(detect.stdout)
        self.assertTrue(detected["has_brain"])
        self.assertEqual(detected["brain_manifest"], str((tmp_root / "brain/brain_manifest.json").resolve()))

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

    def test_brain_runtime_health_reports_brain_guard_summary(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "health", "--cwd", str(ROOT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["mode"], "compact")
        self.assertIn("detect", payload)
        self.assertIn("skill_sync", payload)
        self.assertIn("doc_guard_status", payload)
        self.assertIn("integrity", payload)
        self.assertIn("frontier", payload)
        self.assertIn("catalog", payload)
        self.assertIn("next_actions", payload)

    def test_brain_runtime_health_compact_separates_acknowledged_info(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "health", "--cwd", str(ROOT), "--mode", "compact"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["mode"], "compact")
        self.assertIn("catalog", payload)
        self.assertIn("acknowledged_info", payload["catalog"])
        self.assertEqual(payload["catalog"]["actionable_warning_count"], 0)
        self.assertNotIn("doc_guard", payload)
        self.assertIn("doc_guard_status", payload)

    def test_brain_runtime_health_full_keeps_detailed_payload(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "health", "--cwd", str(ROOT), "--mode", "full"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["mode"], "full")
        self.assertIn("doc_guard", payload)
        self.assertIn("unregistered_latest_run_details", payload["frontier"])

    def test_workspace_brain_skill_first_move_is_lite(self) -> None:
        text = SKILL.read_text(encoding="utf-8")
        line_count = len(text.splitlines())

        self.assertIn("capsule --task", text)
        self.assertIn("--verbosity lite", text)
        self.assertIn("health --cwd . --mode compact", text)
        self.assertIn("health --cwd . --mode full", text)
        self.assertIn("Agent Meta Protocol", text)
        self.assertIn("agent-meta-audit", text)
        self.assertIn("reflection-template", text)
        self.assertIn("review --trace-json", text)
        self.assertIn("agent learning proposal", text)
        self.assertIn("auto-create low-risk proposals", text)
        self.assertIn("implementation requires user approval", text)
        self.assertIn("pending agent learning approvals", text)
        self.assertIn("proposed` or `approved", text)
        self.assertLessEqual(line_count, 100)

    def test_workspace_brain_skill_points_long_tasks_to_monitor_without_template_bloat(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("tools.brain.long_task_monitor", text)
        self.assertIn("trace-poll", text)
        self.assertNotIn("Wait-Process -Id <pid> -Timeout 7200", text)
        self.assertNotIn("Start-Sleep", text)
        self.assertNotIn("must not be used as the primary " + "long-task polling mechanism", text)
        self.assertNotIn("start_" + "training", text)
        self.assertNotIn("Do not " + "modify", text)
        self.assertNotIn("\u4e0d\u5f97", text)
        self.assertNotIn("\u7981\u6b62", text)

    def test_brain_runtime_capsule_compat_command_is_removed(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "capsule",
                "--cwd",
                str(ROOT),
                "--task",
                "长训练",
                "--intent",
                "long_" + "task",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_workflow_capsule_mutate_intent_still_works(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "tools.brain.workflow",
                "capsule",
                "--task",
                "修改脑区规则",
                "--intent",
                "mutate",
                "--workflow",
                "auto",
                "--verbosity",
                "lite",
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["workflow"], "brain_maintenance")

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
        self.assertTrue({"execution_completion_gate", "long_task_execution_closure"}.intersection(targets))

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

    def test_brain_runtime_agent_meta_audit_compact_reports_contract(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "agent-meta-audit", "--cwd", str(ROOT), "--mode", "compact"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["mode"], "compact")
        self.assertIn("agent_learning", payload)
        self.assertIn("agent_meta_contract", payload)
        self.assertIn("closure_meta_review", payload["agent_meta_contract"])
        self.assertIn("daily_research_evidence_quality", payload)
        self.assertIn("actionable_items", payload)

    def test_brain_runtime_burden_audit_compact_reports_budget_contract(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "brain-burden-audit", "--cwd", str(ROOT), "--mode", "compact"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertIn("brain_burden", payload)
        self.assertEqual(payload["brain_burden"]["blocked_count"], 0)
        self.assertLessEqual(payload["brain_burden"]["hot_path_files"]["brain/skills/workspace-brain/SKILL.md"]["line_count"], 100)
        self.assertIn("compatibility", payload["brain_burden"])
        self.assertIn("tracked_non_source_files", payload["brain_burden"])

    def test_agent_meta_audit_reports_proposed_and_approved_learning_items(self) -> None:
        tmp_root = ROOT / "daily_research/output/test_agent_learning_pending_project"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        subprocess.run(
            [PYTHON, str(RUNTIME), "init", "--cwd", str(tmp_root), "--brain-id", "pending_learning"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        for title, status in (("needs approval", "proposed"), ("already approved", "approved")):
            subprocess.run(
                [
                    PYTHON,
                    str(RUNTIME),
                    "proposal",
                    "--cwd",
                    str(tmp_root),
                    "--title",
                    title,
                    "--trigger",
                    "agent learning item needs user-visible follow-up",
                    "--evidence",
                    "agent learning queue",
                    "--recommendation",
                    "surface pending learning approvals proactively",
                    "--status",
                    status,
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )

        result = subprocess.run(
            [PYTHON, str(RUNTIME), "agent-meta-audit", "--cwd", str(tmp_root), "--mode", "compact"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["agent_learning"]["pending_approval_count"], 2)
        self.assertEqual(payload["agent_learning"]["proposed_count"], 1)
        self.assertEqual(payload["agent_learning"]["approved_count"], 1)
        actionable = [item for item in payload["actionable_items"] if item["type"] == "agent_learning_pending_approval"]
        self.assertEqual(len(actionable), 1)
        self.assertIn("2", actionable[0]["summary"])
        self.assertIn("proposed or approved", actionable[0]["recommended_action"])

    def test_brain_runtime_meta_audit_command_is_removed(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "meta-audit", "--cwd", str(ROOT), "--mode", "compact"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)

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
