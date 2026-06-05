from __future__ import annotations

import time
import unittest
import inspect
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from tools.brain import platform as brain_platform
from tools.brain.platform import (
    PYTHON_EXECUTABLE,
    audit_brain_system,
    build_brain_catalog,
    _run_check,
    check_brain_health,
    check_text_encoding,
    check_text_encoding_text,
    load_workflow_registry,
    resolve_artifact_freshness,
    resolve_workflow_child_brain,
    resolve_bootstrap,
    resolve_run_evidence,
)
from tools.brain.adapters import daily_research as daily_research_adapter

ROOT = Path(__file__).resolve().parents[3]
REMOVED_WORKFLOWS = {
    "brainstorming_" + "design",
    "writing_" + "plan",
    "executing_" + "plan",
    "systematic_" + "debugging",
    "verification_before_" + "completion",
    "long_task",
}


def _write_json(root: Path, relative_path: str, payload: dict) -> str:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return relative_path


def _write_continuous_policy_run_fixture(root: Path, tag: str) -> None:
    trials = []
    for trial_id in range(1, 4):
        trial_tag = f"{tag}_trial{trial_id}"
        base = f"daily_research/output/continuous_policy/studies/{tag}/trial_{trial_id}"
        diagnostics_path = _write_json(
            root,
            f"{base}/training_diagnostics.json",
            {
                "sample_model_type": "temporal_day_set",
                "portfolio_day_set_native_allocation_vector_terms": {
                    "native_source_threshold_loss": 0.01,
                },
            },
        )
        evaluation_path = _write_json(
            root,
            f"{base}/evaluation_summary.json",
            {"continuity_metrics": {"cash_timing_quality_1d": 0.1}},
        )
        protocol_path = _write_json(
            root,
            f"{base}/protocol_summary.json",
            {
                "run_tag": trial_tag,
                "loss_profile": "alpha_result_value_budget_split_v37",
                "evaluation_summary_json": evaluation_path,
                "train": {"training_diagnostics": {"training_diagnostics_json": diagnostics_path}},
                "training_evidence": {"status": "sufficient"},
            },
        )
        trials.append(
            {
                "trial_id": trial_id,
                "trial_tag": trial_tag,
                "status": "completed",
                "phase": "screening",
                "protocol_summary_json": protocol_path,
                "trial_config": {"loss_profile": "alpha_result_value_budget_split_v37"},
                "primary_metrics": {"native_target_valid": True},
            }
        )

    _write_json(
        root,
        f"daily_research/output/continuous_policy/studies/{tag}/study_summary.json",
        {
            "run_tag": tag,
            "status": "completed",
            "stage": "safe_screening",
            "trial_count": 3,
            "completed_trial_count": 3,
            "failed_trial_count": 0,
            "screening_trials": trials,
        },
    )
    _write_json(root, "daily_research/output/continuous_policy/latest_study_summary.json", {"run_tag": tag})
    _write_json(root, "daily_research/output/continuous_policy/latest_protocol_summary.json", {"protocol_tag": f"{tag}_stale"})


class BrainPlatformTest(unittest.TestCase):
    def test_resolve_bootstrap_builds_daily_research_handoff_capsule(self) -> None:
        state = resolve_bootstrap("daily_research")
        payload = state.to_dict()

        self.assertEqual(payload["target_kind"], "child")
        self.assertEqual(payload["workflow_domain"], "daily_research")
        self.assertEqual(payload["child_brain"], "daily_research")
        self.assertIn("brain/brain_manifest.json", payload["main_boot_order"])
        self.assertIn("daily_research/brain/brain_manifest.json", payload["child_boot_order"])
        self.assertIn("daily_research/brain/state_center.md", payload["child_fast_handoff_paths"])
        self.assertEqual(payload["child_write_routes"]["state"], "daily_research/brain/state_center.md")
        self.assertIn("artifact_freshness", payload)
        self.assertIn("project_profile", payload)
        self.assertTrue(payload["project_profile"]["guard_profile"]["active_artifact_guard"])
        self.assertIn("encoding_report", payload)

    def test_resolve_bootstrap_traditional_quant_uses_generic_project_profile(self) -> None:
        payload = resolve_bootstrap("traditional_quant_research").to_dict()
        encoded = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(payload["target_kind"], "child")
        self.assertEqual(payload["workflow_domain"], "traditional_quant_research")
        self.assertEqual(payload["child_brain"], "traditional_quant_research")
        self.assertIn("project_profile", payload)
        self.assertEqual(payload["project_profile"]["project_id"], "traditional_quant_research")
        self.assertFalse(payload["project_profile"]["guard_profile"]["active_artifact_guard"])
        self.assertEqual(payload["artifact_freshness"], {})
        self.assertNotIn("daily_research/output/active_execution_strategy.json", encoded)
        self.assertNotIn("traditional_quant_research/tests -q", encoded)
        self.assertEqual(payload["project_profile"]["verification_profile"]["default_test_commands"], [])

    def test_workspace_governance_alias_bootstraps_workspace(self) -> None:
        for alias in ("workspace", "workspace_root", "workspace_governance"):
            with self.subTest(alias=alias):
                payload = resolve_bootstrap(alias).to_dict()

                self.assertEqual(payload["target_kind"], "workspace")
                self.assertEqual(payload["workflow_domain"], "workspace_governance")
                self.assertEqual(payload["child_brain"], "")
                self.assertEqual(payload["main_manifest"], "brain/brain_manifest.json")
                self.assertIn("brain/brain_manifest.json", payload["boot_order"])

    def test_encoding_report_distinguishes_valid_utf8_from_real_mojibake(self) -> None:
        valid = check_text_encoding(Path("daily_research/brain/operations_center.md"))
        damaged = check_text_encoding_text("# title\n鎿嶄綔涓灑 �\n", label="synthetic")

        self.assertTrue(valid.is_utf8)
        self.assertFalse(valid.has_replacement_char)
        self.assertEqual(valid.suspicious_mojibake_count, 0)
        self.assertGreater(damaged.suspicious_mojibake_count, 0)
        self.assertTrue(damaged.has_replacement_char)

    def test_artifact_freshness_reports_latest_tag_mismatch(self) -> None:
        report = resolve_artifact_freshness()
        payload = report.to_dict()

        self.assertIn("latest_study_summary", payload["artifacts"])
        self.assertIn("latest_protocol_summary", payload["artifacts"])
        self.assertIn("is_stale_risk", payload)
        if payload["artifacts"]["latest_study_summary"]["tag"] and payload["artifacts"]["latest_protocol_summary"]["tag"]:
            self.assertEqual(
                payload["is_stale_risk"],
                payload["artifacts"]["latest_study_summary"]["tag"]
                != payload["artifacts"]["latest_protocol_summary"]["tag"],
            )

    def test_workflow_registry_declares_required_contract_fields(self) -> None:
        registry = load_workflow_registry()
        expected = {
            "brain_handoff",
            "brain_maintenance",
            "brain_writeback",
            "brain_system_audit",
            "brain_architecture_refactor",
            "brain_writeback_verified",
        }
        removed = REMOVED_WORKFLOWS
        self.assertTrue(expected.issubset(set(registry)))
        self.assertFalse(removed.intersection(registry))
        for workflow_id in expected:
            workflow = registry[workflow_id]
            for key in ("preflight", "artifacts", "writeback_routes", "capability_hints", "risk_signals", "verification_hints"):
                self.assertIn(key, workflow, workflow_id)
            self.assertNotIn("forbidden_" + "actions", workflow, workflow_id)
            self.assertNotIn("start_" + "training", json.dumps(workflow, ensure_ascii=False), workflow_id)
        child_registry = load_workflow_registry("daily_research")
        self.assertIn("continuous_policy_safe_screening", child_registry)
        self.assertIn("continuous_policy_result_review", child_registry)

    def test_workspace_registry_has_no_generic_adapter_category(self) -> None:
        registry_path = ROOT / "brain/workflows/registry.json"
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        categories = {str(item.get("category", "")) for item in payload.get("workflows", [])}

        self.assertNotIn("brain_native_" + "superpowers", categories)
        self.assertNotIn("skill " + "fallback", str(payload.get("description", "")).lower())
        self.assertIn("guards", str(payload.get("description", "")).lower())

    def test_split_workflow_registry_and_catalog_are_available(self) -> None:
        registry = load_workflow_registry()
        catalog = build_brain_catalog()

        self.assertIn("brain_system_audit", registry)
        self.assertEqual(catalog["schema_version"], 1)
        self.assertIn("brains", catalog)

    def test_child_workflow_registry_path_is_derived_from_manifest(self) -> None:
        main = {
            "child_brains": [
                {
                    "id": "custom_child",
                    "path": "custom_child/brain/brain_manifest.json",
                }
            ]
        }
        child_registry = {
            "workflows": [
                {
                    "id": "custom_workflow",
                    "title": "Custom Workflow",
                    "category": "custom",
                    "playbook": "custom_child/brain/workflows/playbooks/core.json#custom_workflow",
                }
            ]
        }
        playbook = {
            "workflows": {
                "custom_workflow": {
                    "description": "custom",
                    "preflight": [],
                    "artifacts": [],
                    "writeback_routes": {},
                    "capability_hints": [],
                    "risk_signals": [],
                    "verification_hints": [],
                }
            }
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "brain/workflows/playbooks").mkdir(parents=True)
            (root / "custom_child/brain/workflows/playbooks").mkdir(parents=True)
            (root / "brain/brain_manifest.json").write_text(json.dumps(main), encoding="utf-8")
            (root / "brain/workflows/registry.json").write_text(json.dumps({"workflows": []}), encoding="utf-8")
            (root / "custom_child/brain/workflows/registry.json").write_text(json.dumps(child_registry), encoding="utf-8")
            (root / "custom_child/brain/workflows/playbooks/core.json").write_text(json.dumps(playbook), encoding="utf-8")
            with patch.object(brain_platform, "WORKSPACE_ROOT", root):
                registry = load_workflow_registry("custom_child")

        self.assertIn("custom_workflow", registry)

    def test_workflow_child_resolution_uses_child_registry(self) -> None:
        main = {
            "child_brains": [
                {
                    "id": "custom_child",
                    "path": "custom_child/brain/brain_manifest.json",
                }
            ]
        }
        child_registry = {
            "workflows": [
                {
                    "id": "custom_workflow",
                    "title": "Custom Workflow",
                    "category": "custom",
                    "playbook": "custom_child/brain/workflows/playbooks/core.json#custom_workflow",
                }
            ]
        }
        playbook = {
            "workflows": {
                "custom_workflow": {
                    "description": "custom",
                    "preflight": [],
                    "artifacts": [],
                    "writeback_routes": {},
                    "capability_hints": [],
                    "risk_signals": [],
                    "verification_hints": [],
                }
            }
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "brain/workflows").mkdir(parents=True)
            (root / "custom_child/brain/workflows/playbooks").mkdir(parents=True)
            (root / "brain/brain_manifest.json").write_text(json.dumps(main), encoding="utf-8")
            (root / "brain/workflows/registry.json").write_text(json.dumps({"workflows": []}), encoding="utf-8")
            (root / "custom_child/brain/workflows/registry.json").write_text(json.dumps(child_registry), encoding="utf-8")
            (root / "custom_child/brain/workflows/playbooks/core.json").write_text(json.dumps(playbook), encoding="utf-8")
            with patch.object(brain_platform, "WORKSPACE_ROOT", root):
                child_id = resolve_workflow_child_brain("custom_workflow")

        self.assertEqual(child_id, "custom_child")

    def test_brain_system_audit_reports_verdict_and_noncanonical_brains(self) -> None:
        payload = audit_brain_system(scope="all")

        self.assertIn(payload["verdict"], {"clean", "usable_with_warnings", "needs_refactor", "blocked"})
        self.assertIn("catalog", payload)
        self.assertIn("workflow_registry", payload)
        self.assertIn("language", payload)
        self.assertIn("active_artifact_guard", payload)
        self.assertIn("loose_latest", payload)
        self.assertNotIn("noncanonical_brains_discovered", payload["warnings"])
        self.assertEqual(payload["discovered_noncanonical_brains"], [])
        self.assertEqual(payload["acknowledged_non_truth_brains"][0]["brain_id"], "tools")

    def test_explicit_run_evidence_capsule_reads_coherent_r52_trials(self) -> None:
        tag = "self_opt_study_r52_native_source_delta_closure_screening_safe_20260510_02"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_continuous_policy_run_fixture(root, tag)
            with patch.object(brain_platform, "WORKSPACE_ROOT", root), patch.object(
                daily_research_adapter,
                "WORKSPACE_ROOT",
                root,
            ):
                report = resolve_run_evidence(tag)
                payload = report.to_dict()

        self.assertEqual(payload["run_tag"], tag)
        self.assertEqual(
            payload["run_summary_json"],
            f"daily_research/output/continuous_policy/studies/{tag}/study_summary.json",
        )
        self.assertNotIn("study_summary_json", payload)
        self.assertNotIn("study_tag", payload)
        self.assertEqual(payload["completed_trial_count"], 3)
        self.assertEqual(len(payload["trials"]), 3)
        self.assertTrue(payload["artifact_freshness"]["is_stale_risk"])
        first = payload["trials"][0]
        self.assertEqual(first["loss_profile"], "alpha_result_value_budget_split_v37")
        self.assertEqual(first["sample_model_type"], "temporal_day_set")
        self.assertIn("native_target_valid", first["metrics"])
        self.assertIn("native_source_threshold_loss", first["training_terms"])

    def test_run_evidence_requires_run_tag_field_when_summary_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tag = "path20_removed_study_tag_field"
            summary = root / f"daily_research/output/path_policy/studies/{tag}/study_summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps({"study_tag": tag, "status": "completed"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with patch.object(brain_platform, "WORKSPACE_ROOT", root), patch.object(
                daily_research_adapter,
                "WORKSPACE_ROOT",
                root,
            ):
                report = resolve_run_evidence(tag, workflow="path_policy").to_dict()

        self.assertTrue(report["exists"])
        self.assertFalse(report["coherent"])
        self.assertIn("run summary missing run_tag", report["evidence_gaps"])

    def test_evidence_metric_registry_includes_r52c_receiver_mask_metrics(self) -> None:
        for key in (
            "native_receiver_executable_mask_count",
            "native_positive_delta_count",
            "native_positive_delta_unsupported_share",
            "native_receiver_mask_mismatch_count",
        ):
            self.assertIn(key, daily_research_adapter.STUDY_SUMMARY_METRIC_KEYS)

    def test_run_check_reports_elapsed_seconds(self) -> None:
        result = _run_check("quick", [PYTHON_EXECUTABLE, "-c", "print('ok')"])

        self.assertTrue(result["ok"])
        self.assertIn("elapsed_seconds", result)
        self.assertGreaterEqual(result["elapsed_seconds"], 0.0)

    def test_health_checks_run_in_parallel(self) -> None:
        from tools.brain import platform as brain_platform

        def slow_ok(name: str, _command: list[str], *, env: dict[str, str] | None = None) -> dict:
            time.sleep(0.15)
            return {
                "name": name,
                "returncode": 0,
                "ok": True,
                "stdout_tail": "",
                "stderr_tail": "",
                "elapsed_seconds": 0.15,
            }

        started = time.perf_counter()
        with patch.object(brain_platform, "_run_check", side_effect=slow_ok):
            payload = check_brain_health().to_dict()
        elapsed = time.perf_counter() - started

        self.assertEqual(payload["status"], "ok")
        self.assertLess(elapsed, 0.45)
        self.assertIn("elapsed_seconds", payload)
        for check in payload["checks"].values():
            self.assertIn("elapsed_seconds", check)

    def test_health_openmp_lane_sanitizes_parent_workaround_env(self) -> None:
        captured_envs: dict[str, dict[str, str] | None] = {}

        def ok_with_env(name: str, _command: list[str], *, env: dict[str, str] | None = None) -> dict:
            captured_envs[name] = env
            return {
                "name": name,
                "returncode": 0,
                "ok": True,
                "stdout_tail": "",
                "stderr_tail": "",
                "elapsed_seconds": 0.001,
            }

        with patch.dict("os.environ", {"KMP_DUPLICATE_LIB_OK": "True"}):
            with patch.object(brain_platform, "_run_check", side_effect=ok_with_env):
                payload = check_brain_health("daily_research").to_dict()

        self.assertEqual(payload["status"], "ok")
        openmp_env = captured_envs["openmp_strict"]
        self.assertIsNotNone(openmp_env)
        self.assertNotIn("KMP_DUPLICATE_LIB_OK", openmp_env or {})
        self.assertEqual((openmp_env or {})["PYTHONUTF8"], "1")

    def test_daily_research_facts_are_served_by_adapter(self) -> None:
        freshness = daily_research_adapter.resolve_artifact_freshness().to_dict()
        active_guard = daily_research_adapter.active_artifact_diff_status()

        self.assertIn("latest_study_summary", freshness["artifacts"])
        self.assertEqual(active_guard["path"], "daily_research/output/active_execution_strategy.json")

    def test_platform_source_does_not_embed_daily_research_fact_constants(self) -> None:
        source = inspect.getsource(brain_platform)

        self.assertNotIn("LATEST_ARTIFACTS =", source)
        self.assertNotIn("STUDY_SUMMARY_METRIC_KEYS =", source)
        self.assertNotIn("daily_research/output/active_execution_strategy.json", source)


if __name__ == "__main__":
    unittest.main()
