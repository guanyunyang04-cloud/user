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
        self.assertIn("自进化", text)
        self.assertIn("-m tools.brain.workflow capsule", text)
        self.assertIn("-m tools.brain.workflow route", text)
        self.assertIn("-m tools.brain.workflow query", text)
        self.assertIn("writeback-plan", text)
        self.assertIn("brain_runtime.py", text)
        self.assertIn("brain_runtime.py health", text)
        self.assertNotIn("r10-r52", text)

    def test_skill_mentions_api_fallback(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("API Fallback", text)
        self.assertIn("workflow auto", text)
        self.assertIn("workflow-guide", text)
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
        self.assertIn("brain/output/runtime_learning", payload["json_path"].replace("\\", "/"))
        proposal_payload = json.loads(Path(payload["json_path"]).read_text(encoding="utf-8"))
        self.assertIn("requires_user_confirmation", proposal_payload["authority"])
        self.assertEqual(proposal_payload["severity"], "info")
        self.assertEqual(proposal_payload["owner_brain"], "learning_demo")
        self.assertEqual(proposal_payload["writeback_target"], "brain/references/")
        self.assertTrue(proposal_payload["requires_user_confirmation"])

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
        self.assertIn("detect", payload)
        self.assertIn("skill_sync", payload)
        self.assertIn("doc_guard", payload)
        self.assertIn("integrity", payload)
        self.assertIn("frontier", payload)
        self.assertIn("catalog", payload)
        self.assertIn("next_actions", payload)


if __name__ == "__main__":
    unittest.main()
