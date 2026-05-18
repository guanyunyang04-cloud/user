from __future__ import annotations

import unittest

from daily_research.tools.brain_integrity_check import run_checks


class BrainIntegrityCatalogTest(unittest.TestCase):
    def test_catalog_warnings_do_not_block_attached_brain_integrity(self) -> None:
        findings = run_checks()
        errors = [finding for finding in findings if finding.severity == "error"]
        warning_codes = {finding.code for finding in findings if finding.severity == "warning"}

        self.assertEqual(errors, [])
        self.assertIn("catalog_noncanonical_brain", warning_codes)


if __name__ == "__main__":
    unittest.main()
