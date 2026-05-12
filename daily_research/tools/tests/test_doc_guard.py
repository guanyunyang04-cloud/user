from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from daily_research.tools import doc_guard


class DocGuardTest(unittest.TestCase):
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

    def test_tracked_large_files_require_allowlist(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            large_file = root / "research" / "output" / "artifact.pkl"
            allowed_file = root / "daily_stock_analysis-main" / "sources" / "demo.gif"
            large_file.parent.mkdir(parents=True)
            allowed_file.parent.mkdir(parents=True)
            large_file.write_bytes(b"x" * 2048)
            allowed_file.write_bytes(b"y" * 2048)
            allowlist = root / "brain" / "tracked_large_file_allowlist.json"
            allowlist.parent.mkdir(parents=True)
            allowlist.write_text(
                (
                    '{"version":1,"max_tracked_file_size_bytes":1024,'
                    '"allowed":[{"path":"daily_stock_analysis-main/sources/demo.gif",'
                    '"max_size_bytes":4096,"reason":"test asset"}]}'
                ),
                encoding="utf-8",
            )

            with patch.object(doc_guard, "WORKSPACE_ROOT", root):
                issues = doc_guard._check_tracked_large_files(
                    tracked_paths=[
                        "research/output/artifact.pkl",
                        "daily_stock_analysis-main/sources/demo.gif",
                    ]
                )

        self.assertEqual(len(issues), 1)
        self.assertIn("tracked_large_file_not_allowlisted:research/output/artifact.pkl", issues[0])


if __name__ == "__main__":
    unittest.main()
