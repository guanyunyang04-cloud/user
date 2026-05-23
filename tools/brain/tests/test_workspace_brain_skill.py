from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
SKILL = ROOT / "brain/skills/workspace-brain/SKILL.md"


class WorkspaceBrainSkillTest(unittest.TestCase):
    def test_skill_frontmatter_and_body_are_procedural(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("---\nname: workspace-brain"))
        self.assertIn("description: Use the workspace main brain", text)
        self.assertIn("-m tools.brain.workflow capsule", text)
        self.assertIn("-m tools.brain.workflow route", text)
        self.assertIn("-m tools.brain.workflow query", text)
        self.assertIn("writeback-plan", text)
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


if __name__ == "__main__":
    unittest.main()

