from __future__ import annotations

import unittest
from pathlib import Path

from daily_research.tools.brain_evidence_registry import build_evidence_registry, query_evidence_registry


ROOT = Path(__file__).resolve().parents[3]


class BrainEvidenceRegistryTest(unittest.TestCase):
    def test_registry_indexes_recent_status_docs_with_existing_paths(self) -> None:
        payload = build_evidence_registry()

        self.assertEqual(payload["status"], "ok")
        ids = {record["id"] for record in payload["records"]}
        self.assertIn("r61", ids)
        self.assertIn("r62", ids)
        self.assertIn("r55_cash_timing_release_controller_status_20260512", ids)
        self.assertIn("r55_gpu_training_runtime_acceleration_status_20260513", ids)
        for record in payload["records"]:
            self.assertTrue((ROOT / record["path"]).exists(), record["path"])

    def test_query_finds_r62_and_dataset_ids(self) -> None:
        payload = query_evidence_registry("r62")

        self.assertEqual(payload["match_count"], 1)
        match = payload["matches"][0]
        self.assertEqual(match["id"], "r62")
        self.assertIn("daily_research/brain/references/r62_research_data_lake_status_20260514.md", match["path"])
        self.assertTrue(any("policy_input_bundle" in item for item in match["dataset_ids"]))


if __name__ == "__main__":
    unittest.main()
