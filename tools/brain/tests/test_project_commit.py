from __future__ import annotations

import unittest

from tools.brain.project_commit import build_commit_message, check_commit_scope, stage_and_commit_project
from tools.brain.project_profiles import load_project_profile


class ProjectCommitTest(unittest.TestCase):
    def test_build_commit_message_uses_project_prefix_and_trailers(self) -> None:
        message = build_commit_message(
            project_id="traditional_quant_research",
            task_summary="repair factor tests",
            verified=["pytest traditional_quant_research/tests -q", "git diff --check"],
        )

        self.assertTrue(message.startswith("traditional_quant_research: repair factor tests"))
        self.assertIn("Project: traditional_quant_research", message)
        self.assertIn("Agent-Task: repair factor tests", message)
        self.assertIn("Verified: pytest traditional_quant_research/tests -q; git diff --check", message)

    def test_scope_check_allows_project_and_declared_brain_routes(self) -> None:
        payload = check_commit_scope(
            project_id="traditional_quant_research",
            changed_paths=[
                "traditional_quant_research/backtest.py",
                "traditional_quant_research/brain/state_center.md",
            ],
            allowed_prefixes=[
                "traditional_quant_research/",
                "traditional_quant_research/brain/",
            ],
            baseline_dirty_paths=[],
        )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["blocked_paths"], [])

    def test_scope_check_blocks_cross_project_paths(self) -> None:
        payload = check_commit_scope(
            project_id="traditional_quant_research",
            changed_paths=[
                "traditional_quant_research/backtest.py",
                "daily_research/brain/state_center.md",
            ],
            allowed_prefixes=["traditional_quant_research/"],
            baseline_dirty_paths=[],
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_commit_scope_conflict")
        self.assertIn("daily_research/brain/state_center.md", payload["blocked_paths"])

    def test_scope_check_blocks_baseline_dirty_overlap(self) -> None:
        payload = check_commit_scope(
            project_id="traditional_quant_research",
            changed_paths=["traditional_quant_research/backtest.py"],
            allowed_prefixes=["traditional_quant_research/"],
            baseline_dirty_paths=["traditional_quant_research/backtest.py"],
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_commit_scope_conflict")
        self.assertIn("traditional_quant_research/backtest.py", payload["baseline_overlap_paths"])

    def test_stage_and_commit_blocks_without_verification_evidence(self) -> None:
        payload = stage_and_commit_project(
            project_id="traditional_quant_research",
            task_summary="repair factor tests",
            verified=[],
            dry_run=True,
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_commit_unverified")

    def test_workspace_brain_commit_scope_allows_child_manifest_profiles_only(self) -> None:
        profile = load_project_profile("workspace-brain")
        allowed_prefixes = list(profile["commit_policy"]["allowed_prefixes"])
        payload = check_commit_scope(
            project_id="workspace-brain",
            changed_paths=[
                "brain/workflows/playbooks/core.json",
                "daily_research/brain/brain_manifest.json",
                "traditional_quant_research/brain/brain_manifest.json",
            ],
            allowed_prefixes=allowed_prefixes,
            baseline_dirty_paths=[],
        )

        self.assertEqual(payload["status"], "ok")

        blocked = check_commit_scope(
            project_id="workspace-brain",
            changed_paths=["daily_research/tools/train.py"],
            allowed_prefixes=allowed_prefixes,
            baseline_dirty_paths=[],
        )
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("daily_research/tools/train.py", blocked["blocked_paths"])


if __name__ == "__main__":
    unittest.main()
