from __future__ import annotations

import unittest
from datetime import date

from tools.brain.platform import build_brain_catalog


class BrainCatalogTest(unittest.TestCase):
    def test_catalog_discovers_canonical_and_noncanonical_brains(self) -> None:
        catalog = build_brain_catalog()
        brains = {item["brain_id"]: item for item in catalog["brains"]}

        self.assertEqual(brains["workspace_root"]["status"], "canonical_root")
        for brain_id in ("daily_research", "t0_project", "daily_stock_analysis-main"):
            self.assertEqual(brains[brain_id]["status"], "attached")
            self.assertTrue(brains[brain_id]["manifest_path"].endswith("brain_manifest.json"))

        self.assertEqual(brains["daily_research_cache_legacy"]["status"], "cache_legacy")
        self.assertEqual(brains["daily_research_cache_legacy"]["root"], "daily_research/cache/brain")

        self.assertEqual(brains["tools"]["status"], "non_truth_tooling")
        self.assertEqual(brains["tools"]["root"], "tools/brain")
        self.assertEqual(brains["tools"]["manifest_path"], "")

        self.assertIn("a_stock_daily_selection", brains)
        self.assertEqual(brains["a_stock_daily_selection"]["status"], "external_or_inactive_missing_manifest")

    def test_catalog_records_language_policy_and_guard_status(self) -> None:
        catalog = build_brain_catalog()

        self.assertEqual(catalog["language_policy"], "zh_semantic_en_identifiers_v1")
        for item in catalog["brains"]:
            self.assertIn("language_policy", item)
            self.assertIn("last_guard_status", item)

    def test_catalog_generated_at_uses_current_date(self) -> None:
        catalog = build_brain_catalog()

        self.assertEqual(catalog["generated_at"], date.today().isoformat())


if __name__ == "__main__":
    unittest.main()
