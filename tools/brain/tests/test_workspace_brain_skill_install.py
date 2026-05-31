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


class WorkspaceBrainSkillInstallTest(unittest.TestCase):
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



if __name__ == "__main__":
    unittest.main()
