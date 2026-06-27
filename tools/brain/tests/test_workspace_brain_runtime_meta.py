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


class WorkspaceBrainRuntimeMetaTest(unittest.TestCase):
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

    def test_brain_runtime_burden_audit_command_is_removed(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "brain-burden-audit", "--cwd", str(ROOT), "--mode", "compact"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_brain_runtime_structure_audit_compact_reports_contract(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "brain-structure-audit", "--cwd", str(ROOT), "--mode", "compact"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertIn("brain_structure", payload)
        self.assertNotIn("brain_burden", payload)
        self.assertEqual(payload["brain_structure"]["blocked_count"], 0)
        skill_entry = payload["brain_structure"]["hot_path_files"]["brain/skills/workspace-brain/SKILL.md"]
        self.assertEqual(skill_entry["line_count_policy"], "diagnostic_only_not_blocking")
        self.assertIn("structural_signals", skill_entry)
        self.assertIn("legacy_entrypoints", payload["brain_structure"])
        self.assertEqual(payload["brain_structure"]["legacy_entrypoints"]["unsupported_legacy_entries"], [])
        self.assertIn("tracked_non_source_files", payload["brain_structure"])

    def test_brain_runtime_multi_paradigm_lint_reports_attached_brains(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "multi-paradigm-lint", "--cwd", str(ROOT), "--scope", "attached"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["error_count"], 0)
        self.assertGreaterEqual(len(payload["brains"]), 6)
        self.assertTrue(all(item["ok"] for item in payload["brains"]))

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



if __name__ == "__main__":
    unittest.main()
