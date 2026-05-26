from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.brain import rules as brain_rules
from tools.brain.rules import (
    finding_for_full_gold_claim_without_catalog,
    finding_for_stale_latest_without_explicit_tag,
    findings_for_shadow_promotional_claims,
    findings_for_control_plane_doc_lengths,
    findings_for_failed_completed_trials,
    findings_for_realtime_completed_evidence,
)


class BrainRulesTest(unittest.TestCase):
    def test_failed_trial_cannot_be_completed_evidence(self) -> None:
        findings = findings_for_failed_completed_trials(
            {
                "trials": [
                    {"trial_id": 1, "status": "failed", "training_evidence_status": "completed"},
                    {"trial_id": 2, "status": "completed", "training_evidence_status": "sufficient"},
                ]
            }
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "failed_trial_marked_completed_evidence")

    def test_stale_latest_requires_explicit_tag(self) -> None:
        class Freshness:
            is_stale_risk = True
            mismatch_reason = "mismatch"

        with patch.object(brain_rules.daily_research_adapter, "resolve_artifact_freshness", return_value=Freshness()):
            finding = finding_for_stale_latest_without_explicit_tag(has_explicit_run_tag=False)

        self.assertIsNotNone(finding)
        self.assertEqual(finding.code, "loose_latest_stale_requires_explicit_tag")

    def test_active_artifact_diff_is_hard_failure(self) -> None:
        with patch.object(brain_rules, "_run_git_diff_name", return_value="diff --git ..."):
            payload = brain_rules.run_brain_rules(has_explicit_run_tag=True, check_control_plane_lengths=False)

        self.assertEqual(payload["status"], "failed")
        self.assertIn("active_artifact_diff", {finding["code"] for finding in payload["findings"]})

    def test_full_gold_claim_requires_catalog_entry(self) -> None:
        class EmptyLake:
            def __init__(self, _root=None) -> None:
                pass

            def list_datasets(self, **_kwargs):
                import pandas as pd

                return pd.DataFrame([])

        with patch("daily_research.data_lake.ResearchDataLake", EmptyLake):
            finding = finding_for_full_gold_claim_without_catalog("full Gold training dataset is complete")

        self.assertIsNotNone(finding)
        self.assertEqual(finding.code, "full_gold_claim_without_catalog_entry")

    def test_realtime_tail_is_not_training_evidence(self) -> None:
        findings = findings_for_realtime_completed_evidence(
            {"zone": "realtime_research", "unobserved_label_rows": 3, "is_training_safe": False}
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "realtime_tail_counted_as_training_evidence")

    def test_shadow_only_summary_cannot_be_marked_live_ready(self) -> None:
        findings = findings_for_shadow_promotional_claims(
            [
                {
                    "run_tag": "shadow_protocol",
                    "shadow_only": True,
                    "promotion_allowed": False,
                    "promotion_gate": {"status": "shadow_only"},
                    "verdict": "promotion_ready",
                    "live_default_change": True,
                }
            ]
        )

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "shadow_evidence_marked_promotional")

    def test_run_brain_rules_checks_shadow_promotional_evidence(self) -> None:
        payload = brain_rules.run_brain_rules(
            has_explicit_run_tag=True,
            check_control_plane_lengths=False,
            evidence_summaries=[
                {
                    "shadow_only": True,
                    "promotion_allowed": False,
                    "promotion_gate": {"status": "research_shadow_only"},
                    "decision": "activate_live",
                }
            ],
        )

        self.assertEqual(payload["status"], "failed")
        self.assertIn("shadow_evidence_marked_promotional", {finding["code"] for finding in payload["findings"]})

    def test_control_plane_doc_length_warning(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "too_long.md"
            path.write_text("a\nb\nc\n", encoding="utf-8")
            with patch.object(brain_rules, "WORKSPACE_ROOT", Path(tmp)):
                findings = findings_for_control_plane_doc_lengths({Path("too_long.md"): 2})

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "warning")
        self.assertEqual(findings[0].code, "brain_control_plane_doc_too_long")


if __name__ == "__main__":
    unittest.main()
