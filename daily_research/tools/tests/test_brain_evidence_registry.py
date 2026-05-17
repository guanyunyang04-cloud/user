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

    def test_registry_indexes_data_lake_contract_docs(self) -> None:
        registry = build_evidence_registry()
        ids = {record["id"] for record in registry["records"]}

        self.assertIn("data_lake_universal_repair_contract_20260517", ids)

    def test_registry_uses_section_aware_tags_and_next_actions(self) -> None:
        registry = build_evidence_registry()
        matches = [record for record in registry["records"] if record["id"] == "r65"]

        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertIn("portfolio_set_v5", match["tags"])
        self.assertNotIn("core_v4", match["tags"])
        self.assertTrue(any("continuous_policy_training_matrices__strict_train" in item for item in match["dataset_ids"]))
        self.assertFalse(
            any("replaces MLP-style core-v4" in item for item in match["next_allowed_actions"]),
            match["next_allowed_actions"],
        )

    def test_query_finds_r64_strict_gold_dataset(self) -> None:
        payload = query_evidence_registry("continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1")

        self.assertGreaterEqual(payload["match_count"], 1)
        ids = {match["id"] for match in payload["matches"]}
        self.assertIn("r64", ids)
        self.assertIn("r65", ids)

    def test_brain_maintenance_records_are_brain_workflow(self) -> None:
        registry = build_evidence_registry()
        matches = [record for record in registry["records"] if record["id"] == "r66"]

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["workflow"], "brain")
        self.assertIn("brain", matches[0]["tags"])
        self.assertNotIn("data_lake", matches[0]["tags"])
        self.assertFalse(
            any("Evidence registry no longer treats early summary bullets" in item for item in matches[0]["next_allowed_actions"]),
            matches[0]["next_allowed_actions"],
        )


if __name__ == "__main__":
    unittest.main()
