from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from tools.brain.project_commit import build_commit_message, check_commit_scope, split_commit_paths, stage_and_commit_project
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

    def test_split_commit_paths_keeps_external_parallel_work_out_of_scope(self) -> None:
        project_paths, external_paths = split_commit_paths(
            paths=[
                "daily_research/brain/state_center.md",
                "traditional_quant_research/research_log/scout.md",
                "daily_research/path_policy/model.py",
            ],
            allowed_prefixes=["daily_research/"],
        )

        self.assertEqual(
            project_paths,
            ["daily_research/brain/state_center.md", "daily_research/path_policy/model.py"],
        )
        self.assertEqual(external_paths, ["traditional_quant_research/research_log/scout.md"])

    def test_stage_and_commit_blocks_without_verification_evidence(self) -> None:
        payload = stage_and_commit_project(
            project_id="traditional_quant_research",
            task_summary="repair factor tests",
            verified=[],
            dry_run=True,
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_commit_unverified")

    def test_stage_and_commit_dry_run_ignores_external_parallel_dirty_paths(self) -> None:
        with (
            patch(
                "tools.brain.project_commit.load_project_profile",
                return_value={
                    "commit_policy": {
                        "allowed_prefixes": ["daily_research/"],
                        "message_prefix": "daily_research",
                    }
                },
            ),
            patch(
                "tools.brain.project_commit.changed_paths",
                return_value=[
                    "daily_research/brain/state_center.md",
                    "traditional_quant_research/research_log/scout.md",
                ],
            ),
        ):
            payload = stage_and_commit_project(
                project_id="daily_research",
                task_summary="write high return scout",
                verified=["pytest daily_research/path_policy/tests -q"],
                dry_run=True,
            )

        self.assertEqual(payload["status"], "dry_run")
        scope = payload["scope"]
        self.assertEqual(scope["changed_paths"], ["daily_research/brain/state_center.md"])
        self.assertEqual(scope["blocked_paths"], [])
        self.assertEqual(scope["ignored_external_paths"], ["traditional_quant_research/research_log/scout.md"])

    def test_stage_and_commit_blocks_when_only_external_parallel_paths_exist(self) -> None:
        with (
            patch(
                "tools.brain.project_commit.load_project_profile",
                return_value={"commit_policy": {"allowed_prefixes": ["daily_research/"]}},
            ),
            patch(
                "tools.brain.project_commit.changed_paths",
                return_value=["traditional_quant_research/research_log/scout.md"],
            ),
        ):
            payload = stage_and_commit_project(
                project_id="daily_research",
                task_summary="write high return scout",
                verified=["pytest daily_research/path_policy/tests -q"],
                dry_run=True,
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "no_project_changes")
        self.assertEqual(payload["scope"]["ignored_external_paths"], ["traditional_quant_research/research_log/scout.md"])

    def test_stage_and_commit_uses_project_pathspec_only(self) -> None:
        git_calls: list[list[str]] = []

        def fake_run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
            git_calls.append(args)
            if args == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            patch(
                "tools.brain.project_commit.load_project_profile",
                return_value={
                    "commit_policy": {
                        "allowed_prefixes": ["daily_research/"],
                        "message_prefix": "daily_research",
                    }
                },
            ),
            patch(
                "tools.brain.project_commit.changed_paths",
                return_value=[
                    "daily_research/brain/state_center.md",
                    "traditional_quant_research/research_log/scout.md",
                ],
            ),
            patch("tools.brain.project_commit._run_git", side_effect=fake_run_git),
        ):
            payload = stage_and_commit_project(
                project_id="daily_research",
                task_summary="write high return scout",
                verified=["pytest daily_research/path_policy/tests -q"],
            )

        self.assertEqual(payload["status"], "committed")
        self.assertIn(["add", "--", "daily_research/brain/state_center.md"], git_calls)
        commit_calls = [args for args in git_calls if args and args[0] == "commit"]
        self.assertEqual(len(commit_calls), 1)
        self.assertEqual(commit_calls[0][-2:], ["--", "daily_research/brain/state_center.md"])
        self.assertNotIn("traditional_quant_research/research_log/scout.md", commit_calls[0])

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
