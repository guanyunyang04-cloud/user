from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import patch

from daily_research.tools.brain_platform import (
    PYTHON_EXECUTABLE,
    audit_brain_system,
    build_brain_catalog,
    _run_check,
    check_brain_health,
    check_text_encoding,
    check_text_encoding_text,
    load_workflow_registry,
    resolve_artifact_freshness,
    resolve_bootstrap,
    resolve_study_evidence,
)


class BrainPlatformTest(unittest.TestCase):
    def test_resolve_bootstrap_builds_daily_research_handoff_capsule(self) -> None:
        state = resolve_bootstrap("daily_research")
        payload = state.to_dict()

        self.assertEqual(payload["child_brain"], "daily_research")
        self.assertIn("brain/brain_manifest.json", payload["main_boot_order"])
        self.assertIn("daily_research/brain/brain_manifest.json", payload["child_boot_order"])
        self.assertIn("daily_research/brain/state_center.md", payload["child_fast_handoff_paths"])
        self.assertEqual(payload["child_write_routes"]["state"], "daily_research/brain/state_center.md")
        self.assertIn("artifact_freshness", payload)
        self.assertIn("encoding_report", payload)

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
            "continuous_policy_safe_screening",
            "continuous_policy_result_review",
            "brain_writeback",
            "brain_system_audit",
            "brain_architecture_refactor",
        }
        self.assertTrue(expected.issubset(set(registry)))
        for workflow_id in expected:
            workflow = registry[workflow_id]
            for key in ("preflight", "artifacts", "writeback_routes", "forbidden_actions"):
                self.assertIn(key, workflow, workflow_id)

    def test_split_workflow_registry_and_catalog_are_available(self) -> None:
        registry = load_workflow_registry()
        catalog = build_brain_catalog()

        self.assertIn("brain_system_audit", registry)
        self.assertEqual(catalog["schema_version"], 1)
        self.assertIn("brains", catalog)

    def test_brain_system_audit_reports_verdict_and_noncanonical_brains(self) -> None:
        payload = audit_brain_system(scope="all")

        self.assertIn(payload["verdict"], {"clean", "usable_with_warnings", "needs_refactor", "blocked"})
        self.assertIn("catalog", payload)
        self.assertIn("workflow_registry", payload)
        self.assertIn("language", payload)
        self.assertIn("active_artifact_guard", payload)
        self.assertIn("loose_latest", payload)
        self.assertTrue(payload["discovered_noncanonical_brains"])

    def test_explicit_study_evidence_capsule_reads_coherent_r52_trials(self) -> None:
        tag = "self_opt_study_r52_native_source_delta_closure_screening_safe_20260510_02"
        report = resolve_study_evidence(tag)
        payload = report.to_dict()

        self.assertEqual(payload["study_tag"], tag)
        self.assertEqual(payload["completed_trial_count"], 3)
        self.assertEqual(len(payload["trials"]), 3)
        self.assertTrue(payload["artifact_freshness"]["is_stale_risk"])
        first = payload["trials"][0]
        self.assertEqual(first["loss_profile"], "alpha_result_value_budget_split_v37")
        self.assertEqual(first["sample_model_type"], "temporal_day_set")
        self.assertIn("native_target_valid", first["metrics"])
        self.assertIn("native_source_threshold_loss", first["training_terms"])

    def test_evidence_metric_registry_includes_r52c_receiver_mask_metrics(self) -> None:
        from daily_research.tools import brain_platform

        for key in (
            "native_receiver_executable_mask_count",
            "native_positive_delta_count",
            "native_positive_delta_unsupported_share",
            "native_receiver_mask_mismatch_count",
        ):
            self.assertIn(key, brain_platform.STUDY_SUMMARY_METRIC_KEYS)

    def test_run_check_reports_elapsed_seconds(self) -> None:
        result = _run_check("quick", [PYTHON_EXECUTABLE, "-c", "print('ok')"])

        self.assertTrue(result["ok"])
        self.assertIn("elapsed_seconds", result)
        self.assertGreaterEqual(result["elapsed_seconds"], 0.0)

    def test_health_checks_run_in_parallel(self) -> None:
        from daily_research.tools import brain_platform

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
        from daily_research.tools import brain_platform

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
                payload = check_brain_health().to_dict()

        self.assertEqual(payload["status"], "ok")
        openmp_env = captured_envs["openmp_strict"]
        self.assertIsNotNone(openmp_env)
        self.assertNotIn("KMP_DUPLICATE_LIB_OK", openmp_env or {})
        self.assertEqual((openmp_env or {})["PYTHONUTF8"], "1")


if __name__ == "__main__":
    unittest.main()
