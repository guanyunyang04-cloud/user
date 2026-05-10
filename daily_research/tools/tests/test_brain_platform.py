from __future__ import annotations

import unittest
from pathlib import Path

from daily_research.tools.brain_platform import (
    check_text_encoding,
    check_text_encoding_text,
    load_workflow_registry,
    resolve_artifact_freshness,
    resolve_bootstrap,
)


class BrainPlatformTest(unittest.TestCase):
    def test_resolve_bootstrap_builds_daily_research_handoff_capsule(self) -> None:
        state = resolve_bootstrap("daily_research")
        payload = state.to_dict()

        self.assertEqual(payload["child_brain"], "daily_research")
        self.assertIn("brain/brain_manifest.json", payload["main_boot_order"])
        self.assertIn("daily_research/brain/brain_manifest.json", payload["child_boot_order"])
        self.assertIn("daily_research/brain/state_center.md", payload["child_fast_handoff_paths"])
        self.assertEqual(payload["child_write_routes"]["state"], "daily_research/brain/state_center.md")
        self.assertIn("artifact_freshness", payload)
        self.assertIn("encoding_report", payload)

    def test_encoding_report_distinguishes_valid_utf8_from_real_mojibake(self) -> None:
        valid = check_text_encoding(Path("daily_research/brain/operations_center.md"))
        damaged = check_text_encoding_text("# title\n鎿嶄綔涓灑 �\n", label="synthetic")

        self.assertTrue(valid.is_utf8)
        self.assertFalse(valid.has_replacement_char)
        self.assertEqual(valid.suspicious_mojibake_count, 0)
        self.assertGreater(damaged.suspicious_mojibake_count, 0)
        self.assertTrue(damaged.has_replacement_char)

    def test_artifact_freshness_reports_latest_tag_mismatch(self) -> None:
        report = resolve_artifact_freshness()
        payload = report.to_dict()

        self.assertIn("latest_study_summary", payload["artifacts"])
        self.assertIn("latest_protocol_summary", payload["artifacts"])
        self.assertIn("is_stale_risk", payload)
        if payload["artifacts"]["latest_study_summary"]["tag"] and payload["artifacts"]["latest_protocol_summary"]["tag"]:
            self.assertEqual(
                payload["is_stale_risk"],
                payload["artifacts"]["latest_study_summary"]["tag"]
                != payload["artifacts"]["latest_protocol_summary"]["tag"],
            )

    def test_workflow_registry_declares_required_contract_fields(self) -> None:
        registry = load_workflow_registry()
        expected = {
            "brain_handoff",
            "brain_maintenance",
            "continuous_policy_safe_screening",
            "continuous_policy_result_review",
            "brain_writeback",
        }
        self.assertTrue(expected.issubset(set(registry)))
        for workflow_id in expected:
            workflow = registry[workflow_id]
            for key in ("preflight", "artifacts", "writeback_routes", "forbidden_actions"):
                self.assertIn(key, workflow, workflow_id)


if __name__ == "__main__":
    unittest.main()
