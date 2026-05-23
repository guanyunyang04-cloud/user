from __future__ import annotations

import unittest

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
        self.assertIn("a_stock_daily_selection", brains)
        self.assertIn(brains["a_stock_daily_selection"]["status"], {"discovered_untracked", "missing_manifest"})

    def test_catalog_records_language_policy_and_guard_status(self) -> None:
        catalog = build_brain_catalog()

        self.assertEqual(catalog["language_policy"], "zh_semantic_en_identifiers_v1")
        for item in catalog["brains"]:
            self.assertIn("language_policy", item)
            self.assertIn("last_guard_status", item)


if __name__ == "__main__":
    unittest.main()

