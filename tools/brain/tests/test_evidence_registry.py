from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from tools.brain import evidence_registry as brain_evidence_registry
from tools.brain.adapters import daily_research_evidence
from tools.brain.evidence_registry import build_evidence_registry, query_evidence_registry


ROOT = Path(__file__).resolve().parents[3]


class BrainEvidenceRegistryTest(unittest.TestCase):
    def test_daily_research_evidence_adapter_declares_project_sources(self) -> None:
        self.assertIn(Path("daily_research/brain/references"), daily_research_evidence.REFERENCE_ROOTS)
        self.assertEqual(
            daily_research_evidence.REGISTRY_PATH,
            Path("daily_research/brain/references/evidence_registry.json"),
        )

    def test_workspace_evidence_registry_does_not_embed_daily_research_patterns(self) -> None:
        source = inspect.getsource(brain_evidence_registry)

        self.assertNotIn("daily_research/brain/references", source)
        self.assertNotIn("continuous_policy", source)
        self.assertNotIn("alpha_path20", source)

    def test_registry_indexes_recent_status_docs_with_existing_paths(self) -> None:
        payload = build_evidence_registry()

        self.assertEqual(payload["schema_version"], 3)
        self.assertEqual(payload["status"], "ok")
        ids = {record["id"] for record in payload["records"]}
        self.assertIn("r61", ids)
        self.assertIn("r62", ids)
        self.assertIn("r55_cash_timing_release_controller_status_20260512", ids)
        self.assertIn("r55_gpu_training_runtime_acceleration_status_20260513", ids)
        for record in payload["records"]:
            self.assertTrue((ROOT / record["path"]).exists(), record["path"])
            self.assertNotIn("study_tags", record)

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

    def test_registry_no_longer_indexes_generic_workflow_contract(self) -> None:
        registry = build_evidence_registry()
        retired_id = "brain_native_" + "superpowers_contract_20260518"
        matches = [record for record in registry["records"] if record["id"] == retired_id]

        self.assertEqual(matches, [])

    def test_registry_indexes_path20_run_tags(self) -> None:
        registry = build_evidence_registry()
        matches = [record for record in registry["records"] if record["id"] == "alpha_path20_stage1_formal_cap80_result_20260518"]

        self.assertEqual(len(matches), 1)
        self.assertIn("path20_stage1_formal_cap80_repaired_20260518_01", matches[0]["run_tags"])
        self.assertNotIn("study_tags", matches[0])
        payload = query_evidence_registry("path20_stage1_formal_cap80_repaired_20260518_01")
        self.assertIn(
            "alpha_path20_stage1_formal_cap80_result_20260518",
            {match["id"] for match in payload["matches"]},
        )

    def test_registry_indexes_multi_horizon_utility_mainline(self) -> None:
        registry = build_evidence_registry()
        matches = [
            record
            for record in registry["records"]
            if record["id"] == "alpha_multi_horizon_utility_policy_mainline_rename_20260523"
        ]

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["workflow"], "path_policy")
        self.assertIn("path_policy", matches[0]["tags"])
        payload = query_evidence_registry("alpha_multi_horizon_utility_policy_v1")
        self.assertIn(
            "alpha_multi_horizon_utility_policy_mainline_rename_20260523",
            {match["id"] for match in payload["matches"]},
        )

    def test_registry_separates_research_program_study_family_and_run_tags(self) -> None:
        registry = build_evidence_registry()
        matches = [
            record
            for record in registry["records"]
            if record["id"] == "alpha_multi_horizon_stage25_stability_calibration_20260526"
        ]

        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertIn("alpha_multi_horizon_utility_policy_v1", match["research_programs"])
        self.assertIn("stage25_stability_calibration", match["study_families"])
        self.assertIn("mh25_path_aux_daily1_45_multiseed_seed19_20260526_01", match["run_tags"])
        self.assertNotIn("study_tags", match)

    def test_query_can_find_multi_horizon_evidence_by_program_and_family(self) -> None:
        program_payload = query_evidence_registry("alpha_multi_horizon_utility_policy_v1")
        program_ids = {match["id"] for match in program_payload["matches"]}
        self.assertIn("alpha_multi_horizon_output_aux_grid_stage1_shadow_20260525", program_ids)
        self.assertIn("alpha_multi_horizon_stage25_stability_calibration_20260526", program_ids)

        family_payload = query_evidence_registry("stage25_stability_calibration")
        family_ids = {match["id"] for match in family_payload["matches"]}
        self.assertIn("alpha_multi_horizon_stage25_stability_calibration_20260526", family_ids)

    def test_stage1_reference_does_not_inherit_stage2_family_from_negative_mention(self) -> None:
        registry = build_evidence_registry()
        matches = [
            record
            for record in registry["records"]
            if record["id"] == "alpha_multi_horizon_output_aux_grid_stage1_shadow_20260525"
        ]

        self.assertEqual(len(matches), 1)
        self.assertIn("alpha_multi_horizon_utility_policy_v1", matches[0]["research_programs"])
        self.assertEqual(matches[0]["study_families"], ["stage1_output_aux_grid"])

    def test_adapter_does_not_treat_target_function_names_as_run_tags(self) -> None:
        text = """
        - `mh_short_utility_1_3_5d_v1`
        - `mh_mid_utility_5_10_20d_v1`
        - `mh_long_utility_15_20_30d_v1`
        """

        self.assertEqual(daily_research_evidence.run_tags(text), [])

    def test_adapter_indexes_mh_output_aux_grid_run_tags(self) -> None:
        text = """
        ## Run Tags

        - `mh_out_decision_utility_path_aux_v1_fullgrid_seed7_20260525_01`
        - `mh_out_forecast_path_v1_baseline_fullgrid_seed19_20260525_01`
        - `mh_grid_decision_utility_path_aux_v1_daily1_45_feas_seed7_20260525_02`
        - `mh_grid_decision_utility_v1_baseline_sparse_long_seed19_20260525_02`
        - `mh25_path_aux_daily1_45_multiseed_seed19_20260526_01`
        """

        tags = daily_research_evidence.run_tags(text)

        self.assertIn("mh_out_decision_utility_path_aux_v1_fullgrid_seed7_20260525_01", tags)
        self.assertIn("mh_out_forecast_path_v1_baseline_fullgrid_seed19_20260525_01", tags)
        self.assertIn("mh_grid_decision_utility_path_aux_v1_daily1_45_feas_seed7_20260525_02", tags)
        self.assertIn("mh_grid_decision_utility_v1_baseline_sparse_long_seed19_20260525_02", tags)
        self.assertIn("mh25_path_aux_daily1_45_multiseed_seed19_20260526_01", tags)

    def test_adapter_ignores_removed_study_tag_section_for_run_instances(self) -> None:
        text = """
        - Mainline: `alpha_multi_horizon_utility_policy_v1`.

        ## Study Tags

        - `mh26_path_aux_future_family_seed7_20260601_01`
        """

        self.assertEqual(daily_research_evidence.run_tags(text), [])

    def test_adapter_treats_run_tag_section_entries_as_run_instances_without_prefix_whitelist(self) -> None:
        text = """
        - Mainline: `alpha_multi_horizon_utility_policy_v1`.

        ## Run Tags

        - `mh26_path_aux_future_family_seed7_20260601_01`
        """

        self.assertEqual(daily_research_evidence.run_tags(text), ["mh26_path_aux_future_family_seed7_20260601_01"])

    def test_adapter_does_not_treat_research_pointers_as_run_instances(self) -> None:
        text = """
        - Current research mainline pointer: `alpha_multi_horizon_utility_policy_v1`.
        - Previous current pointer: `alpha_path20_neural_policy_v1`.
        - Secondary route: `alpha_path20_sequence_policy_v1`.
        - Architecture route: `alpha_path20_sequence_policy_v5`.
        - Completed study: `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01`.
        """

        self.assertEqual(
            daily_research_evidence.run_tags(text),
            ["path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01"],
        )

    def test_adapter_classifies_multi_horizon_program_family_and_runs(self) -> None:
        text = """
        - Mainline: `alpha_multi_horizon_utility_policy_v1`.
        - Study family: `stage25_stability_calibration`.
        - `mh25_path_aux_daily1_45_multiseed_seed19_20260526_01`
        """

        self.assertEqual(
            daily_research_evidence.research_programs(text),
            ["alpha_multi_horizon_utility_policy_v1"],
        )
        self.assertEqual(
            daily_research_evidence.study_families(text),
            ["stage25_stability_calibration"],
        )
        self.assertEqual(
            daily_research_evidence.run_tags(text),
            ["mh25_path_aux_daily1_45_multiseed_seed19_20260526_01"],
        )

    def test_adapter_indexes_path_policy_reference_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "path_policy_execution_issue_learning_20260523.md"
            path.write_text("# Path Policy Execution Issue Learning\n", encoding="utf-8")

            self.assertTrue(daily_research_evidence.is_reference_file(path))

    def test_adapter_indexes_tdx_free_data_platform_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tdx_free_data_platform_decision_20260523.md"
            path.write_text("# TDX-Free Data Platform Decision\n", encoding="utf-8")

            self.assertTrue(daily_research_evidence.is_reference_file(path))

    def test_adapter_indexes_execution_reference_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "execution_signal_refresh_closure_20260524.md"
            path.write_text("# Execution Signal Refresh Closure\n", encoding="utf-8")

            self.assertTrue(daily_research_evidence.is_reference_file(path))

    def test_registry_indexes_tdx_free_data_platform_decision(self) -> None:
        registry = build_evidence_registry()
        matches = [
            record
            for record in registry["records"]
            if record["id"] == "tdx_free_data_platform_decision_20260523"
        ]

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["workflow"], "research_data_lake")
        self.assertIn("data_platform", matches[0]["tags"])

    def test_registry_indexes_tdx_free_data_platform_v2(self) -> None:
        registry = build_evidence_registry()
        matches = [
            record
            for record in registry["records"]
            if record["id"] == "tdx_free_data_platform_v2_20260523"
        ]

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["workflow"], "research_data_lake")
        self.assertIn("data_platform", matches[0]["tags"])

    def test_registry_indexes_execution_signal_refresh_closure(self) -> None:
        registry = build_evidence_registry()
        matches = [
            record
            for record in registry["records"]
            if record["id"] == "execution_signal_refresh_closure_20260524"
        ]

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["workflow"], "execution")
        self.assertIn("execution", matches[0]["tags"])
        self.assertIn("signal_panel", matches[0]["tags"])

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
