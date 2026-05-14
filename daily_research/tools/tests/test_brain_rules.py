from __future__ import annotations

import unittest
from unittest.mock import patch

from daily_research.tools import brain_rules
from daily_research.tools.brain_rules import (
    finding_for_full_gold_claim_without_catalog,
    finding_for_stale_latest_without_explicit_tag,
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

        with patch.object(brain_rules, "resolve_artifact_freshness", return_value=Freshness()):
            finding = finding_for_stale_latest_without_explicit_tag(has_explicit_study_tag=False)

        self.assertIsNotNone(finding)
        self.assertEqual(finding.code, "loose_latest_stale_requires_explicit_tag")

    def test_active_artifact_diff_is_hard_failure(self) -> None:
        with patch.object(brain_rules, "_run_git_diff_name", return_value="diff --git ..."):
            payload = brain_rules.run_brain_rules(has_explicit_study_tag=True)

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


if __name__ == "__main__":
    unittest.main()
