from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.brain import doc_guard


class DocGuardTest(unittest.TestCase):
    def test_registered_child_project_output_markdown_is_allowed(self) -> None:
        self.assertTrue(
            doc_guard._is_allowed_doc_path(
                "traditional_quant_research/output/experiments/example_run/summary.md"
            )
        )

    def test_unregistered_output_markdown_is_not_allowed_by_default(self) -> None:
        self.assertFalse(
            doc_guard._is_allowed_doc_path(
                "unregistered_project/output/experiments/example_run/summary.md"
            )
        )

    def test_workspace_markdown_paths_uses_git_list_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            tracked = root / "brain/state_center.md"
            ignored = root / "node_modules/ignored.md"
            tracked.parent.mkdir(parents=True)
            ignored.parent.mkdir(parents=True)
            tracked.write_text("# state\n", encoding="utf-8")
            ignored.write_text("# ignored\n", encoding="utf-8")

            with mock.patch.object(doc_guard, "_git_markdown_paths", return_value=["brain/state_center.md"]):
                paths = doc_guard._workspace_markdown_paths(root)

        self.assertEqual(paths, [tracked])


if __name__ == "__main__":
    unittest.main()
