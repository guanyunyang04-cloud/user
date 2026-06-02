from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
