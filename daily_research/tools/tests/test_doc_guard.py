from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from daily_research.tools import doc_guard


class DocGuardTest(unittest.TestCase):
    def test_brain_docs_reject_codex_thread_uri(self) -> None:
        forbidden_uri = "codex" + "://threads/019df224-1780-7b43-9855-cf5b41550c4a"
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brain_dir = root / "brain"
            brain_dir.mkdir()
            (brain_dir / "state_center.md").write_text(
                f"# 状态\n\n不要保存 {forbidden_uri}\n",
                encoding="utf-8",
            )

            with patch.object(doc_guard, "WORKSPACE_ROOT", root):
                issues = doc_guard._check_brain_thread_deeplinks()

        self.assertEqual(len(issues), 1)
        self.assertIn("brain/state_center.md:3", issues[0])

    def test_non_codex_brain_links_are_allowed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            brain_dir = root / "brain"
            brain_dir.mkdir()
            (brain_dir / "state_center.md").write_text(
                "# 状态\n\n参考 https://example.com/thread-123.html\n",
                encoding="utf-8",
            )

            with patch.object(doc_guard, "WORKSPACE_ROOT", root):
                issues = doc_guard._check_brain_thread_deeplinks()

        self.assertEqual(issues, [])

    def test_pytest_cache_readme_is_not_treated_as_project_doc(self) -> None:
        self.assertTrue(doc_guard._is_allowed_doc_path(".pytest_cache/README.md"))

    def test_generated_dependency_docs_are_not_treated_as_project_docs(self) -> None:
        self.assertTrue(
            doc_guard._is_allowed_doc_path(
                "daily_stock_analysis-main/apps/dsa-desktop/node_modules/@types/node/README.md"
            )
        )
        self.assertTrue(
            doc_guard._is_allowed_doc_path(
                "daily_stock_analysis-main/apps/dsa-web/test-results/smoke/error-context.md"
            )
        )


if __name__ == "__main__":
    unittest.main()
