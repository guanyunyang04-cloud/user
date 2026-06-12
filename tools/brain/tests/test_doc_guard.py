from __future__ import annotations

import contextlib
import io
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

    def test_external_docs_require_canonical_marker(self) -> None:
        self.assertTrue(doc_guard._requires_canonical_marker("README.md"))
        self.assertEqual(
            doc_guard._canonical_marker_issue("README.md", "# Root\n"),
            "external_doc_missing_canonical_marker:README.md",
        )
        self.assertEqual(
            doc_guard._canonical_marker_issue(
                "README.md",
                "# Root\n\nCanonical brain source: `brain/master_brain.md`.\n",
            ),
            "",
        )

    def test_generated_and_output_markdown_do_not_require_canonical_marker(self) -> None:
        self.assertFalse(
            doc_guard._requires_canonical_marker(
                "traditional_quant_research/output/experiments/example_run/summary.md"
            )
        )
        self.assertFalse(doc_guard._requires_canonical_marker("node_modules/package/README.md"))

    def test_changed_guard_files_skips_generated_dependency_deletions(self) -> None:
        with mock.patch.object(
            doc_guard,
            "_git_changed_paths",
            return_value=[
                "node_modules/package/README.md",
                "brain/state_center.md",
            ],
        ):
            self.assertEqual(doc_guard._changed_guard_files(), ["brain/state_center.md"])

    def test_public_docs_skip_strict_question_line_heuristic(self) -> None:
        self.assertFalse(doc_guard._uses_strict_brain_text_heuristics("daily_stock_analysis-main/docs/FAQ_EN.md"))
        self.assertTrue(doc_guard._uses_strict_brain_text_heuristics("brain/state_center.md"))

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

    def test_check_files_scope_skips_global_guards(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "note.md"
            path.write_text("# note\n", encoding="utf-8")
            args = doc_guard.build_parser().parse_args(["check", "--files", str(path)])

            with (
                mock.patch.object(doc_guard, "_check_document_layout", side_effect=AssertionError("layout")),
                mock.patch.object(doc_guard, "_check_active_execution_brain_alignment", side_effect=AssertionError("active")),
                mock.patch.object(doc_guard, "_check_tracked_large_files", side_effect=AssertionError("large")),
                mock.patch.object(doc_guard, "run_brain_integrity_checks", side_effect=AssertionError("integrity")),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                result = doc_guard.cmd_check(args)

        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn("[scope] mode=files", output)
        self.assertIn("[layout] skipped", output)
        self.assertIn("[brain-integrity] skipped", output)

    def test_check_changed_scope_uses_changed_guard_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "changed.md"
            path.write_text("# changed\n", encoding="utf-8")
            args = doc_guard.build_parser().parse_args(["check", "--scope", "changed"])

            with (
                mock.patch.object(doc_guard, "_changed_guard_files", return_value=[str(path)]),
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                result = doc_guard.cmd_check(args)

        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn("[scope] mode=changed file_count=1", output)
        self.assertIn("[layout] skipped", output)

    def test_bare_check_keeps_full_global_guard_semantics(self) -> None:
        calls: list[str] = []

        def mark(name: str):
            def _inner():
                calls.append(name)
                return []

            return _inner

        args = doc_guard.build_parser().parse_args(["check"])
        with (
            mock.patch.object(doc_guard, "_default_docs", return_value=[]),
            mock.patch.object(doc_guard, "_check_document_layout", side_effect=mark("layout")),
            mock.patch.object(doc_guard, "_check_active_execution_brain_alignment", side_effect=mark("active")),
            mock.patch.object(doc_guard, "_check_tracked_large_files", side_effect=mark("large")),
            mock.patch.object(doc_guard, "run_brain_integrity_checks", side_effect=mark("integrity")),
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            result = doc_guard.cmd_check(args)

        self.assertEqual(result, 0)
        self.assertEqual(calls, ["layout", "active", "large", "integrity"])
        self.assertIn("[scope] mode=full", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
