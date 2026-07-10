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

        self.assertEqual(brains["tools"]["status"], "non_truth_tooling")
        self.assertEqual(brains["tools"]["root"], "tools/brain")
        self.assertEqual(brains["tools"]["manifest_path"], "")

        self.assertNotIn("daily_research_cache_legacy", brains)
        self.assertNotIn("a_stock_daily_selection", brains)

    def test_catalog_records_only_stable_registry_fields(self) -> None:
        catalog = build_brain_catalog()

        self.assertEqual(catalog["schema_version"], 2)
        self.assertEqual(catalog["language_policy"], "zh_semantic_en_identifiers_v1")
        self.assertNotIn("generated_at", catalog)
        for item in catalog["brains"]:
            self.assertIn("language_policy", item)
            self.assertNotIn("references_count", item)
            self.assertNotIn("last_guard_status", item)


if __name__ == "__main__":
    unittest.main()
