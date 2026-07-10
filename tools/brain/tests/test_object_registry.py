from __future__ import annotations

import unittest

from tools.brain.object_registry import (
    load_object_registry,
    match_objects_for_paths,
    match_objects_for_task,
    registry_validation_commands,
    registry_writeback_targets,
)


class ObjectRegistryTest(unittest.TestCase):
    def test_registry_declares_core_cross_project_objects(self) -> None:
        payload = load_object_registry()
        object_ids = {item["object_id"] for item in payload["objects"]}

        self.assertIn("qdp_v2_active_data_base", object_ids)
        self.assertIn("sequence_training_pack", object_ids)
        self.assertIn("model_research_evidence", object_ids)
        self.assertIn("active_execution_artifact", object_ids)
        self.assertIn("brain_sync_surface", object_ids)

    def test_task_match_finds_qdp_and_research_objects(self) -> None:
        matches = match_objects_for_task("qdp_v2 sequence pack GRU 训练")
        by_id = {item.object_id: item for item in matches}

        self.assertIn("qdp_v2_active_data_base", by_id)
        self.assertIn("sequence_training_pack", by_id)
        self.assertIn("model_research_evidence", by_id)
        self.assertEqual(by_id["qdp_v2_active_data_base"].owner, "quant_data_platform")
        self.assertEqual(by_id["sequence_training_pack"].owner, "daily_research")

    def test_generic_write_verb_does_not_select_an_unmentioned_object(self) -> None:
        self.assertEqual(match_objects_for_task("清理脑区治理规则")[0].object_id, "brain_sync_surface")
        self.assertFalse(any(item.object_id == "qdp_v2_active_data_base" for item in match_objects_for_task("清理脑区治理规则")))

    def test_path_match_finds_protected_active_artifact(self) -> None:
        matches = match_objects_for_paths(["daily_research/output/active_execution_strategy.json"])
        by_id = {item.object_id: item for item in matches}

        self.assertIn("active_execution_artifact", by_id)
        self.assertTrue(by_id["active_execution_artifact"].protected)

    def test_registry_helpers_return_writeback_and_validation(self) -> None:
        targets = registry_writeback_targets(["qdp_v2_active_data_base", "brain_sync_surface"])
        commands = registry_validation_commands(["qdp_v2_active_data_base", "brain_sync_surface"])

        self.assertIn("quant_data_platform/brain/state_center.md", targets)
        self.assertIn("brain/state_center.md", targets)
        self.assertIn("qdp check --quick --json", commands)
        self.assertTrue(any("integrity_check" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
