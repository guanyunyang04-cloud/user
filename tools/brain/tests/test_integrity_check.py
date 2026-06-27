from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from tools.brain import integrity_check
from tools.brain.integrity_check import run_checks
from tools.brain.platform import load_manifest


class BrainIntegrityCatalogTest(unittest.TestCase):
    def test_acknowledged_non_truth_catalog_entries_do_not_warn(self) -> None:
        findings = run_checks()
        errors = [finding for finding in findings if finding.severity == "error"]
        warning_codes = {finding.code for finding in findings if finding.severity == "warning"}

        self.assertEqual(errors, [])
        self.assertNotIn("catalog_noncanonical_brain", warning_codes)
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
        self.assertIn("current workspace object interfaces", contract["hot_path_semantics"]["workspace_core_doc"])
        self.assertIn("runtime object instances", contract["hot_path_semantics"]["child_state_center"])

    def test_main_manifest_declares_agent_meta_protocol(self) -> None:
        manifest = load_manifest("brain/brain_manifest.json")
        contract = manifest["agent_meta_protocol"]

        self.assertEqual(contract["actor"], "agent")
        self.assertEqual(contract["substrate"], "brain")
        self.assertEqual(contract["tool_role"], "sensor")
        self.assertEqual(contract["authority"], "propose_only")
        self.assertEqual(contract["required_passes"], ["task_start", "decision_boundary", "before_final"])
        self.assertEqual(contract["before_final_mode"], "closure_boundary_meta_question_discovery")
        self.assertEqual(contract["before_final_trigger_policy"], "low_noise")
        self.assertEqual(contract["proposal_creation_policy"], "auto_create_low_risk_proposed_status")
        self.assertIn("proposed status", contract["proposal_creation_boundary"])
        self.assertEqual(
            contract["implementation_approval_policy"],
            "requires_explicit_user_approval_for_protocol_or_behavior_changes",
        )
        self.assertIn("tool clear", contract["before_final_human_override_rule"].lower())
        self.assertNotIn("fixed checklist", contract["before_final_human_override_rule"].lower())
        self.assertIn("universe_scope", contract["before_final_research_scope_grade_rule"])
        self.assertIn("gate result", contract["before_final_research_scope_grade_rule"])
        self.assertIn("diagnostic-vs-evidence-grade", contract["before_final_research_scope_grade_rule"])
        self.assertEqual(contract["proposal_queue"], "brain/output/agent_learning/")
        self.assertEqual(contract["pending_approval_statuses"], ["proposed", "approved"])
        self.assertIn("proactively", contract["pending_approval_surface_rule"])
        self.assertIn("proposed", contract["pending_approval_surface_rule"])
        self.assertIn("approved", contract["pending_approval_surface_rule"])

    def test_main_manifest_declares_brain_burden_contract(self) -> None:
        manifest = load_manifest("brain/brain_manifest.json")
        contract = manifest["brain_burden_contract"]

        self.assertIn("brain/skills/workspace-brain/SKILL.md", contract["hot_path_files"])
        self.assertEqual(contract["line_count_policy"], "diagnostic_only_not_blocking")
        self.assertIn("legacy_global_rule_terms", contract["structural_signals"])
        self.assertEqual(contract["rule_classes"], ["hard_safety", "operating_default", "deep_dive", "deprecated"])
        self.assertIn("owner", contract["compatibility_entry_required_fields"])
        self.assertIn("delete_by", contract["compatibility_entry_required_fields"])

    def test_agent_meta_contract_docs_do_not_reintroduce_legacy_public_terms(self) -> None:
        findings = run_checks()
        error_codes = {finding.code for finding in findings if finding.severity == "error"}

        self.assertNotIn("agent_meta_legacy_contract_text", error_codes)
        self.assertNotIn("agent_meta_capsule_schema_invalid", error_codes)
        self.assertNotIn("agent_meta_legacy_capsule_field_present", error_codes)

    def test_action_needed_noncanonical_catalog_entries_still_warn(self) -> None:
        fake_catalog = deepcopy(integrity_check._read_json(integrity_check.BRAIN_CATALOG))
        fake_catalog["brains"].append(
            {
                "brain_id": "unregistered_workspace",
                "root": "unregistered_workspace/brain",
                "manifest_path": "",
                "status": "missing_manifest",
                "body_root": "unregistered_workspace",
                "references_path": "",
                "references_count": 0,
                "language_policy": "zh_semantic_en_identifiers_v1",
                "last_guard_status": "not_guarded",
            }
        )
        findings: list[integrity_check.Finding] = []

        with (
            patch.object(integrity_check, "_read_json", return_value=fake_catalog),
            patch.object(integrity_check, "build_brain_catalog", return_value=fake_catalog),
        ):
            integrity_check._validate_brain_catalog(findings, load_manifest("brain/brain_manifest.json"))
        warnings = [finding for finding in findings if finding.code == "catalog_noncanonical_brain"]

        self.assertTrue(warnings)
        self.assertIn("unregistered_workspace needs review", warnings[0].detail)


if __name__ == "__main__":
    unittest.main()
