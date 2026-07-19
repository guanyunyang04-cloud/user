from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from tools.brain.project_commit import (
    _git_output,
    build_commit_message,
    check_commit_scope,
    inspect_push_target,
    split_commit_paths,
    stage_and_commit_project,
)
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

    def test_stage_and_commit_blocks_when_expected_path_is_ignored(self) -> None:
        with (
            patch(
                "tools.brain.project_commit.load_project_profile",
                return_value={
                    "commit_policy": {
                        "allowed_prefixes": ["brain/"],
                        "message_prefix": "workspace-brain",
                    }
                },
            ),
            patch(
                "tools.brain.project_commit.changed_paths",
                return_value=[
                    "brain/state_center.md",
                    "quant_data_platform/README.md",
                ],
            ),
        ):
            payload = stage_and_commit_project(
                project_id="workspace-brain",
                task_summary="register data platform",
                verified=["git diff --check"],
                expected_paths=["quant_data_platform/README.md"],
                dry_run=True,
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_commit_expected_paths_out_of_scope")
        self.assertEqual(payload["scope"]["ignored_expected_paths"], ["quant_data_platform/README.md"])

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
        pathspec_calls: list[tuple[list[str], list[str]]] = []
        git_calls: list[list[str]] = []

        def fake_run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
            git_calls.append(args)
            if args == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        def fake_git_path_list(args: list[str]) -> list[str]:
            if args == ["diff", "--name-only"]:
                return ["daily_research/brain/state_center.md"]
            if args == ["ls-files", "--others", "--exclude-standard"]:
                return []
            if args == ["diff", "--cached", "--name-only"]:
                return ["daily_research/brain/state_center.md"]
            return []

        def fake_run_git_with_pathspec(args: list[str], paths: list[str]) -> subprocess.CompletedProcess[str]:
            pathspec_calls.append((args, paths))
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
            patch("tools.brain.project_commit._git_path_list", side_effect=fake_git_path_list),
            patch("tools.brain.project_commit._run_git", side_effect=fake_run_git),
            patch("tools.brain.project_commit._run_git_with_pathspec", side_effect=fake_run_git_with_pathspec),
        ):
            payload = stage_and_commit_project(
                project_id="daily_research",
                task_summary="write high return scout",
                verified=["pytest daily_research/path_policy/tests -q"],
            )

        self.assertEqual(payload["status"], "committed")
        self.assertEqual(
            pathspec_calls[0],
            (["add", "--update"], ["daily_research/brain/state_center.md"]),
        )
        commit_calls = [item for item in git_calls if item and item[0] == "commit"]
        self.assertEqual(len(commit_calls), 1)
        self.assertEqual(commit_calls[0][:2], ["commit", "-m"])
        self.assertFalse(any(item[0] == ["commit"] for item in pathspec_calls))

    def test_stage_and_commit_does_not_readd_cached_ignored_deletions(self) -> None:
        pathspec_calls: list[tuple[list[str], list[str]]] = []
        git_calls: list[list[str]] = []

        def fake_run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
            git_calls.append(args)
            if args == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        def fake_git_path_list(args: list[str]) -> list[str]:
            if args == ["diff", "--name-only"]:
                return []
            if args == ["ls-files", "--others", "--exclude-standard"]:
                return []
            if args == ["diff", "--cached", "--name-only"]:
                return ["node_modules/pkg/index.js"]
            return []

        def fake_run_git_with_pathspec(args: list[str], paths: list[str]) -> subprocess.CompletedProcess[str]:
            pathspec_calls.append((args, paths))
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with (
            patch(
                "tools.brain.project_commit.load_project_profile",
                return_value={
                    "commit_policy": {
                        "allowed_prefixes": ["node_modules/"],
                        "message_prefix": "workspace-brain",
                    }
                },
            ),
            patch("tools.brain.project_commit.changed_paths", return_value=["node_modules/pkg/index.js"]),
            patch("tools.brain.project_commit._git_path_list", side_effect=fake_git_path_list),
            patch("tools.brain.project_commit._run_git", side_effect=fake_run_git),
            patch("tools.brain.project_commit._run_git_with_pathspec", side_effect=fake_run_git_with_pathspec),
        ):
            payload = stage_and_commit_project(
                project_id="workspace-brain",
                task_summary="untrack dependency directory",
                verified=["git diff --check"],
            )

        self.assertEqual(payload["status"], "committed")
        self.assertEqual(pathspec_calls, [])
        self.assertTrue(any(item and item[0] == "commit" for item in git_calls))

    def test_push_preflight_blocks_when_remote_branch_is_ahead(self) -> None:
        calls: list[list[str]] = []

        def fake_run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            outputs = {
                ("symbolic-ref", "--quiet", "--short", "HEAD"): "main\n",
                ("remote", "get-url", "origin"): "https://example.invalid/repo.git\n",
                ("rev-list", "--left-right", "--count", "HEAD...refs/remotes/origin/main"): "0\t1\n",
            }
            return subprocess.CompletedProcess(args, 0, stdout=outputs.get(tuple(args), ""), stderr="")

        with patch("tools.brain.project_commit._run_git", side_effect=fake_run_git):
            payload = inspect_push_target(remote="origin")

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "git_remote_branch_ahead")
        self.assertEqual(payload["remote_ahead"], 1)
        self.assertIn(["fetch", "--prune", "origin"], calls)

    def test_stage_commit_and_push_is_one_verified_operation(self) -> None:
        git_calls: list[list[str]] = []

        def fake_run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
            git_calls.append(args)
            if args == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        push_target = {
            "status": "ok",
            "remote": "origin",
            "branch": "main",
            "remote_branch_exists": True,
            "local_ahead": 0,
            "remote_ahead": 0,
        }
        published = {
            "status": "ok",
            "remote": "origin",
            "branch": "main",
            "head": "abc123",
            "remote_head": "abc123",
        }
        staged = subprocess.CompletedProcess([], 0, stdout="", stderr="")
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
                    "daily_research/new_module.py",
                    "daily_research/unrelated_work.py",
                ],
            ),
            patch(
                "tools.brain.project_commit._stage_project_paths",
                return_value=staged,
            ) as stage_mock,
            patch(
                "tools.brain.project_commit._git_path_list",
                side_effect=[[], ["daily_research/new_module.py"]],
            ),
            patch("tools.brain.project_commit._run_git", side_effect=fake_run_git),
            patch("tools.brain.project_commit.inspect_push_target", return_value=push_target),
            patch("tools.brain.project_commit.push_current_branch", return_value=published) as push_mock,
        ):
            payload = stage_and_commit_project(
                project_id="daily_research",
                task_summary="add module",
                verified=["pytest focused_test.py -q", "git diff --check"],
                expected_paths=["daily_research/new_module.py"],
                push=True,
            )

        self.assertEqual(payload["status"], "published")
        self.assertEqual(payload["commit"], "abc123")
        self.assertTrue(payload["committed_now"])
        self.assertTrue(any(item and item[0] == "commit" for item in git_calls))
        self.assertEqual(
            payload["scope"]["ignored_project_paths"],
            ["daily_research/unrelated_work.py"],
        )
        stage_mock.assert_called_once_with(["daily_research/new_module.py"])
        push_mock.assert_called_once_with(remote="origin", branch="main")

    def test_explicit_paths_block_previously_staged_unrelated_project_file(self) -> None:
        with (
            patch(
                "tools.brain.project_commit.load_project_profile",
                return_value={"commit_policy": {"allowed_prefixes": ["daily_research/"]}},
            ),
            patch(
                "tools.brain.project_commit.changed_paths",
                return_value=[
                    "daily_research/new_module.py",
                    "daily_research/unrelated_work.py",
                ],
            ),
            patch(
                "tools.brain.project_commit._git_path_list",
                return_value=["daily_research/unrelated_work.py"],
            ),
            patch("tools.brain.project_commit._stage_project_paths") as stage_mock,
        ):
            payload = stage_and_commit_project(
                project_id="daily_research",
                task_summary="add module",
                verified=["git diff --check"],
                expected_paths=["daily_research/new_module.py"],
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "git_staged_unexpected_paths")
        self.assertEqual(
            payload["staged_unexpected_paths"],
            ["daily_research/unrelated_work.py"],
        )
        stage_mock.assert_not_called()

    def test_expected_untracked_path_missing_from_git_inventory_is_blocked(self) -> None:
        with (
            patch(
                "tools.brain.project_commit.load_project_profile",
                return_value={"commit_policy": {"allowed_prefixes": ["daily_research/"]}},
            ),
            patch("tools.brain.project_commit.changed_paths", return_value=[]),
            patch("tools.brain.project_commit._git_path_is_tracked", return_value=False),
        ):
            payload = stage_and_commit_project(
                project_id="daily_research",
                task_summary="add module",
                verified=["git diff --check"],
                expected_paths=["daily_research/missing_module.py"],
                push=True,
            )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["reason"], "project_commit_expected_paths_missing")
        self.assertEqual(
            payload["scope"]["missing_expected_paths"],
            ["daily_research/missing_module.py"],
        )

    def test_push_retry_publishes_existing_local_commit_without_new_changes(self) -> None:
        push_target = {
            "status": "ok",
            "remote": "origin",
            "branch": "main",
            "remote_branch_exists": True,
            "local_ahead": 1,
            "remote_ahead": 0,
        }
        published = {
            "status": "ok",
            "remote": "origin",
            "branch": "main",
            "head": "abc123",
            "remote_head": "abc123",
        }
        with (
            patch(
                "tools.brain.project_commit.load_project_profile",
                return_value={"commit_policy": {"allowed_prefixes": ["daily_research/"]}},
            ),
            patch("tools.brain.project_commit.changed_paths", return_value=[]),
            patch("tools.brain.project_commit.inspect_push_target", return_value=push_target),
            patch("tools.brain.project_commit.push_current_branch", return_value=published),
        ):
            payload = stage_and_commit_project(
                project_id="daily_research",
                task_summary="retry publish",
                verified=["git diff --check"],
                push=True,
            )

        self.assertEqual(payload["status"], "published")
        self.assertFalse(payload["committed_now"])
        self.assertEqual(payload["commit"], "abc123")

    def test_git_failure_output_redacts_url_credentials_and_query_tokens(self) -> None:
        result = subprocess.CompletedProcess(
            [],
            1,
            stdout="",
            stderr="https://private-value@example.invalid/repo?token=another-private-value",
        )

        output = _git_output(result)

        self.assertNotIn("private-value", output)
        self.assertNotIn("another-private-value", output)
        self.assertEqual(output, "https://***@example.invalid/repo?token=***")

    def test_workspace_brain_commit_scope_allows_root_governance_and_child_brain_docs(self) -> None:
        profile = load_project_profile("workspace-brain")
        allowed_prefixes = list(profile["commit_policy"]["allowed_prefixes"])
        payload = check_commit_scope(
            project_id="workspace-brain",
            changed_paths=[
                ".gitignore",
                "AGENTS.md",
                "README.md",
                "node_modules/package.json",
                "daily_research/README.md",
                "daily_stock_analysis-main/AGENTS.md",
                "daily_stock_analysis-main/README.md",
                "quant_data_platform/README.md",
                "traditional_quant_research/README.md",
                "brain/workflows/playbooks/core.json",
                "daily_research/brain/state_center.md",
                "traditional_quant_research/brain/knowledge_center.md",
                "quant_data_platform/brain/identity_layer.md",
            ],
            allowed_prefixes=allowed_prefixes,
            baseline_dirty_paths=[],
        )

        self.assertEqual(payload["status"], "ok")

        blocked = check_commit_scope(
            project_id="workspace-brain",
            changed_paths=[
                "daily_research/tools/train.py",
                "daily_stock_analysis-main/docs/FAQ.md",
                "traditional_quant_research/research_log/README.md",
                "docs/testing_governance.md",
            ],
            allowed_prefixes=allowed_prefixes,
            baseline_dirty_paths=[],
        )
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("daily_research/tools/train.py", blocked["blocked_paths"])
        self.assertIn("daily_stock_analysis-main/docs/FAQ.md", blocked["blocked_paths"])
        self.assertIn("traditional_quant_research/research_log/README.md", blocked["blocked_paths"])
        self.assertIn("docs/testing_governance.md", blocked["blocked_paths"])


if __name__ == "__main__":
    unittest.main()
