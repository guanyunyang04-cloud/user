from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.brain.adapters.daily_research_frontier import build_frontier_report


class FrontierScannerTest(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _touch(self, path: Path, ts: int) -> None:
        os.utime(path, (ts, ts))

    def test_output_newer_than_brain_sets_stale_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "daily_research/output/path_policy/studies/path20_new/study_summary.json"
            self._write_json(
                summary,
                {"run_tag": "path20_new", "status": "completed", "evidence_verdict": "forecast_test_confirmed"},
            )
            ref = root / "daily_research/brain/references/current.md"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("# older\n", encoding="utf-8")
            registry = root / "daily_research/brain/references/evidence_registry.json"
            self._write_json(registry, {"records": [{"id": "path20_new", "run_tags": ["path20_new"]}]})
            self._touch(ref, 100)
            self._touch(registry, 100)
            self._touch(summary, 200)

            report = build_frontier_report(workspace_root=root)

        self.assertTrue(report["brain_may_be_stale"])
        self.assertTrue(report["output_newer_than_brain"])
        self.assertIn("output_newer_than_brain_references", report["warnings"])

    def test_unregistered_latest_tag_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "daily_research/output/path_policy/studies/path20_unregistered/study_summary.json"
            self._write_json(summary, {"run_tag": "path20_unregistered", "status": "completed"})
            ref = root / "daily_research/brain/references/current.md"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("# newer\n", encoding="utf-8")
            registry = root / "daily_research/brain/references/evidence_registry.json"
            self._write_json(registry, {"records": []})
            self._touch(summary, 100)
            self._touch(ref, 200)
            self._touch(registry, 200)

            report = build_frontier_report(workspace_root=root)

        self.assertTrue(report["brain_may_be_stale"])
        self.assertIn("path20_unregistered", report["unregistered_latest_run_tags"])
        self.assertIn("unregistered_latest_run_tags", report["warnings"])
        details = {item["run_tag"]: item for item in report["unregistered_latest_run_details"]}
        self.assertIn("path20_unregistered", details)
        detail = details["path20_unregistered"]
        self.assertEqual(detail["path"], "daily_research/output/path_policy/studies/path20_unregistered/study_summary.json")
        self.assertEqual(detail["mtime_epoch"], 100)
        self.assertFalse(detail["reference_exists"])
        self.assertEqual(detail["suggested_action"], "create_reconciliation_proposal")

    def test_failed_or_interrupted_summary_is_not_upgraded_to_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "daily_research/output/path_policy/studies/path20_failed/study_summary.json"
            self._write_json(
                summary,
                {"run_tag": "path20_failed", "status": "failed", "evidence_verdict": "forecast_failed"},
            )
            ref = root / "daily_research/brain/references/current.md"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("# current\n", encoding="utf-8")
            registry = root / "daily_research/brain/references/evidence_registry.json"
            self._write_json(registry, {"records": [{"run_tags": ["path20_failed"]}]})

            report = build_frontier_report(workspace_root=root)

        latest = report["latest_output_runs"][0]
        self.assertEqual(latest["status"], "failed")
        self.assertEqual(latest["evidence_verdict"], "forecast_failed")
        self.assertFalse(latest["completed_evidence"])

    def test_registered_output_with_newer_brain_reference_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "daily_research/output/path_policy/studies/path20_registered/study_summary.json"
            self._write_json(summary, {"run_tag": "path20_registered", "status": "completed"})
            ref = root / "daily_research/brain/references/current.md"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("# newer\n", encoding="utf-8")
            registry = root / "daily_research/brain/references/evidence_registry.json"
            self._write_json(registry, {"records": [{"run_tags": ["path20_registered"]}]})
            self._touch(summary, 100)
            self._touch(ref, 200)
            self._touch(registry, 200)

            report = build_frontier_report(workspace_root=root)

        self.assertFalse(report["brain_may_be_stale"])
        self.assertFalse(report["output_newer_than_brain"])
        self.assertEqual(report["unregistered_latest_run_tags"], [])
        self.assertEqual(report["registered_run_tag_count"], 1)

    def test_legacy_study_tags_do_not_register_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "daily_research/output/path_policy/studies/path20_legacy/study_summary.json"
            self._write_json(summary, {"study_tag": "path20_legacy", "status": "completed"})
            ref = root / "daily_research/brain/references/current.md"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("# newer\n", encoding="utf-8")
            registry = root / "daily_research/brain/references/evidence_registry.json"
            self._write_json(registry, {"records": [{"study_tags": ["path20_legacy"]}]})
            self._touch(summary, 100)
            self._touch(ref, 200)
            self._touch(registry, 200)

            report = build_frontier_report(workspace_root=root)

        self.assertIn("path20_legacy", report["unregistered_latest_run_tags"])
        self.assertEqual(report["registered_run_tag_count"], 0)

    def test_removed_study_tag_field_does_not_define_current_output_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "daily_research/output/path_policy/studies/path20_dir_name/study_summary.json"
            self._write_json(summary, {"study_tag": "path20_legacy_field", "status": "completed"})
            ref = root / "daily_research/brain/references/current.md"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("# newer\n", encoding="utf-8")
            registry = root / "daily_research/brain/references/evidence_registry.json"
            self._write_json(registry, {"records": [{"run_tags": ["path20_legacy_field"]}]})
            self._touch(summary, 100)
            self._touch(ref, 200)
            self._touch(registry, 200)

            report = build_frontier_report(workspace_root=root)

        self.assertIn("path20_dir_name", report["unregistered_latest_run_tags"])
        self.assertNotIn("path20_legacy_field", report["unregistered_latest_run_tags"])

    def test_registry_id_does_not_register_output_without_run_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "daily_research/output/path_policy/studies/path20_id_only/study_summary.json"
            self._write_json(summary, {"run_tag": "path20_id_only", "status": "completed"})
            ref = root / "daily_research/brain/references/current.md"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("# newer\n", encoding="utf-8")
            registry = root / "daily_research/brain/references/evidence_registry.json"
            self._write_json(registry, {"records": [{"id": "path20_id_only"}]})
            self._touch(summary, 100)
            self._touch(ref, 200)
            self._touch(registry, 200)

            report = build_frontier_report(workspace_root=root)

        self.assertIn("path20_id_only", report["unregistered_latest_run_tags"])
        self.assertEqual(report["registered_run_tag_count"], 0)

    def test_brain_rules_can_report_frontier_staleness_as_warning(self) -> None:
        from tools.brain import rules as brain_rules

        with patch.object(
            brain_rules.daily_research_adapter,
            "build_frontier_report",
            return_value={"brain_may_be_stale": True, "warnings": ["output_newer_than_brain_references"]},
        ):
            payload = brain_rules.run_brain_rules(has_explicit_run_tag=True, check_control_plane_lengths=False)

        self.assertEqual(payload["status"], "ok")
        findings = {item["code"]: item for item in payload["findings"]}
        self.assertIn("frontier_brain_may_be_stale", findings)
        self.assertEqual(findings["frontier_brain_may_be_stale"]["severity"], "warning")


if __name__ == "__main__":
    unittest.main()
