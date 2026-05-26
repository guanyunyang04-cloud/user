from __future__ import annotations

import unittest

from tools.brain.integrity_check import run_checks
from tools.brain.platform import load_manifest


class BrainIntegrityCatalogTest(unittest.TestCase):
    def test_catalog_warnings_do_not_block_attached_brain_integrity(self) -> None:
        findings = run_checks()
        errors = [finding for finding in findings if finding.severity == "error"]
        warning_codes = {finding.code for finding in findings if finding.severity == "warning"}

        self.assertEqual(errors, [])
        self.assertIn("catalog_noncanonical_brain", warning_codes)
        self.assertNotIn("brain_catalog_discovered_entry_missing", warning_codes)
        self.assertNotIn("route_target_unbootstrapable", warning_codes)
        self.assertNotIn("workflow_category_matches_child_brain_id", warning_codes)

    def test_main_manifest_declares_hot_handoff_contract(self) -> None:
        manifest = load_manifest("brain/brain_manifest.json")
        contract = manifest["hot_handoff_contract"]

        self.assertEqual(contract["workspace_default_paths"], [
            "brain/state_center.md",
            "brain/operations_center.md",
            "brain/governance_layer.md",
        ])
        self.assertEqual(contract["child_default_modules"], ["state_center", "operations_center"])
        self.assertIn("episodic_memory", contract["never_default_modules"])
        self.assertEqual(contract["line_budgets"]["workspace_core_doc"], 80)

    def test_non_truth_catalog_warnings_are_acknowledged_boundaries(self) -> None:
        findings = run_checks()
        warnings = [finding for finding in findings if finding.code == "catalog_noncanonical_brain"]

        self.assertTrue(warnings)
        self.assertTrue(
            all("acknowledged non-truth" in finding.detail for finding in warnings),
            [finding.detail for finding in warnings],
        )


if __name__ == "__main__":
    unittest.main()
