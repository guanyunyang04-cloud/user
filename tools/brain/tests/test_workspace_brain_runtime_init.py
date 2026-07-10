from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
import importlib.util
from pathlib import Path
from unittest.mock import patch

from tools.brain import routing


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
SKILL = ROOT / "brain/skills/workspace-brain/SKILL.md"
RUNTIME = ROOT / "brain/skills/workspace-brain/scripts/brain_runtime.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("workspace_brain_runtime_under_test", RUNTIME)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load workspace brain runtime")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkspaceBrainRuntimeInitTest(unittest.TestCase):
    def test_brain_runtime_detects_project_without_brain(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_root = Path(raw_tmp) / "test_no_brain_project"
            tmp_root.mkdir(parents=True)
            result = subprocess.run(
                [PYTHON, str(RUNTIME), "detect", "--cwd", str(tmp_root)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            payload = json.loads(result.stdout)

        self.assertIn(payload["status"], {"ok", "warning"})
        self.assertFalse(payload["has_brain"])
        self.assertFalse(payload["has_brain_tools"])
        self.assertIn("init_brain_available", payload["next_actions"])

    def test_brain_runtime_init_minimal_project(self) -> None:
        tmp_root = ROOT / "daily_research/output/test_minimal_brain_project"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)

        result = subprocess.run(
            [PYTHON, str(RUNTIME), "init", "--cwd", str(tmp_root), "--brain-id", "demo_project"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertTrue((tmp_root / "brain/brain_manifest.json").exists())
        self.assertTrue((tmp_root / "brain/state_center.md").read_text(encoding="utf-8").startswith("# Demo Project 状态中枢"))

        detect = subprocess.run(
            [PYTHON, str(RUNTIME), "detect", "--cwd", str(tmp_root)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        detected = json.loads(detect.stdout)
        self.assertTrue(detected["has_brain"])
        self.assertEqual(detected["brain_manifest"], str((tmp_root / "brain/brain_manifest.json").resolve()))

    def test_brain_runtime_health_reports_takeover_summary(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "health", "--cwd", str(ROOT)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["mode"], "compact")
        self.assertEqual(payload["health_level"], "takeover")
        self.assertIn("detect", payload)
        self.assertIn("manifest", payload)
        self.assertIn("capsule_command", payload)
        self.assertIn("deferred_checks", payload)
        self.assertIn("doc_guard", payload["deferred_checks"])
        self.assertNotIn("skill_sync", payload)
        self.assertNotIn("doc_guard_status", payload)
        self.assertNotIn("integrity", payload)
        self.assertNotIn("frontier", payload)
        self.assertIn("next_actions", payload)

    def test_brain_runtime_health_compact_does_not_run_deep_subprocesses(self) -> None:
        runtime = load_runtime_module()
        with patch.object(runtime, "_run_command", side_effect=AssertionError("compact health must not run deep checks")):
            payload = runtime.health(ROOT, mode="compact")

        self.assertEqual(payload["mode"], "compact")
        self.assertEqual(payload["health_level"], "takeover")
        self.assertNotIn("doc_guard", payload)
        self.assertNotIn("frontier", payload)

    def test_brain_runtime_health_compact_reports_project_profile_deferment(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "health", "--cwd", str(ROOT), "--mode", "compact"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["mode"], "compact")
        self.assertIn("project_profile_note", payload)
        self.assertIn("project profile", payload["project_profile_note"])
        self.assertIn("full_health_command", payload)

    def test_brain_runtime_health_full_keeps_detailed_payload(self) -> None:
        runtime = load_runtime_module()

        def fake_run_command(cwd: Path, command: list[str], *, timeout_sec: float | None = None) -> dict[str, object]:
            self.assertEqual(timeout_sec, 17)
            joined = " ".join(command)
            if "skill_install" in joined:
                stdout = json.dumps({"all_in_sync": True, "skills": []})
            elif "integrity_check" in joined:
                stdout = json.dumps({"status": "ok", "error_count": 0, "warning_count": 0, "findings": []})
            elif "current-frontier" in joined:
                stdout = json.dumps(
                    {
                        "brain_may_be_stale": False,
                        "warnings": [],
                        "unregistered_latest_run_tags": [],
                        "unregistered_latest_run_details": [{"run_tag": "kept-in-full"}],
                    }
                )
            else:
                stdout = "doc guard ok"
            return {
                "command": command,
                "returncode": 0,
                "ok": True,
                "timed_out": False,
                "timeout_sec": timeout_sec,
                "stdout": stdout,
                "stderr": "",
                "stdout_tail": stdout,
                "stderr_tail": "",
            }

        with (
            patch.object(runtime, "_run_command", side_effect=fake_run_command) as run_command,
            patch("tools.brain.platform.build_brain_catalog", return_value={"brains": [], "generated_at": "test"}),
        ):
            payload = runtime.health(ROOT, mode="full", timeout_sec=17)

        self.assertEqual(payload["mode"], "full")
        self.assertIn("doc_guard", payload)
        self.assertIn("skill_sync", payload)
        self.assertIn("integrity", payload)
        self.assertIn("unregistered_latest_run_details", payload["frontier"])
        self.assertEqual(run_command.call_count, 4)

    def test_brain_runtime_full_health_frontier_timeout_is_structured(self) -> None:
        runtime = load_runtime_module()

        def fake_run_command(cwd: Path, command: list[str], *, timeout_sec: float | None = None) -> dict[str, object]:
            joined = " ".join(command)
            if "skill_install" in joined:
                stdout = json.dumps({"all_in_sync": True, "skills": []})
                return {
                    "command": command,
                    "returncode": 0,
                    "ok": True,
                    "timed_out": False,
                    "timeout_sec": timeout_sec,
                    "stdout": stdout,
                    "stderr": "",
                    "stdout_tail": stdout,
                    "stderr_tail": "",
                }
            if "integrity_check" in joined:
                stdout = json.dumps({"status": "ok", "error_count": 0, "warning_count": 0, "findings": []})
                return {
                    "command": command,
                    "returncode": 0,
                    "ok": True,
                    "timed_out": False,
                    "timeout_sec": timeout_sec,
                    "stdout": stdout,
                    "stderr": "",
                    "stdout_tail": stdout,
                    "stderr_tail": "",
                }
            if "current-frontier" in joined:
                return {
                    "command": command,
                    "returncode": 124,
                    "ok": False,
                    "timed_out": True,
                    "timeout_sec": timeout_sec,
                    "stdout": "",
                    "stderr": "frontier slow",
                    "stdout_tail": "",
                    "stderr_tail": "frontier slow",
                }
            return {
                "command": command,
                "returncode": 0,
                "ok": True,
                "timed_out": False,
                "timeout_sec": timeout_sec,
                "stdout": "doc guard ok",
                "stderr": "",
                "stdout_tail": "doc guard ok",
                "stderr_tail": "",
            }

        with (
            patch.object(runtime, "_run_command", side_effect=fake_run_command),
            patch("tools.brain.platform.build_brain_catalog", return_value={"brains": [], "generated_at": "test"}),
        ):
            payload = runtime.health(ROOT, mode="full", timeout_sec=11)

        self.assertEqual(payload["status"], "warning")
        self.assertTrue(payload["frontier"]["timed_out"])
        self.assertEqual(payload["frontier"]["timeout_sec"], 11)
        self.assertIn("rerun_frontier_with_more_time", payload["next_actions"])

    def test_brain_runtime_full_health_fails_when_doc_guard_fails(self) -> None:
        runtime = load_runtime_module()

        def fake_run_command(cwd: Path, command: list[str], *, timeout_sec: float | None = None) -> dict[str, object]:
            joined = " ".join(command)
            if "skill_install" in joined:
                stdout = json.dumps({"all_in_sync": True, "skills": []})
                return _runtime_result(command, timeout_sec, stdout=stdout)
            if "integrity_check" in joined:
                stdout = json.dumps({"status": "ok", "error_count": 0, "warning_count": 0, "findings": []})
                return _runtime_result(command, timeout_sec, stdout=stdout)
            if "current-frontier" in joined:
                stdout = json.dumps({"brain_may_be_stale": False, "warnings": []})
                return _runtime_result(command, timeout_sec, stdout=stdout)
            return _runtime_result(command, timeout_sec, returncode=1, stderr="layout failed")

        def _runtime_result(
            command: list[str],
            timeout_sec: float | None,
            *,
            returncode: int = 0,
            stdout: str = "",
            stderr: str = "",
        ) -> dict[str, object]:
            return {
                "command": command,
                "returncode": returncode,
                "ok": returncode == 0,
                "timed_out": False,
                "timeout_sec": timeout_sec,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_tail": stdout,
                "stderr_tail": stderr,
            }

        with (
            patch.object(runtime, "_run_command", side_effect=fake_run_command),
            patch("tools.brain.platform.build_brain_catalog", return_value={"brains": []}),
        ):
            payload = runtime.health(ROOT, mode="full", timeout_sec=17)

        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["doc_guard"]["status"], "failed")
        self.assertIn("fix_doc_guard_errors", payload["next_actions"])

    def test_brain_runtime_deep_check_forces_utf8_and_replaces_decode_errors(self) -> None:
        runtime = load_runtime_module()
        completed = subprocess.CompletedProcess(["check"], 0, stdout="ok", stderr="")
        with patch.object(runtime.subprocess, "run", return_value=completed) as run:
            result = runtime._run_command(ROOT, ["check"], timeout_sec=1)

        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")
        self.assertEqual(kwargs["env"]["PYTHONUTF8"], "1")
        self.assertEqual(kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertTrue(result["ok"])

    def test_brain_runtime_deep_check_timeout_is_structured(self) -> None:
        runtime = load_runtime_module()
        with patch.object(
            runtime.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd=["slow-check"], timeout=1, output="partial", stderr="still running"),
        ):
            result = runtime._run_command(ROOT, ["slow-check"], timeout_sec=1)

        self.assertEqual(result["returncode"], 124)
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["timeout_sec"], 1)


    def test_brain_runtime_init_infers_brain_id_and_health_for_standalone_project(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_root = Path(raw_tmp) / "auto_named_project"
            result = subprocess.run(
                [PYTHON, str(RUNTIME), "init", "--cwd", str(tmp_root)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            payload = json.loads(result.stdout)
            health = subprocess.run(
                [PYTHON, str(RUNTIME), "health", "--cwd", str(tmp_root), "--mode", "compact"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            health_payload = json.loads(health.stdout)

        self.assertEqual(payload["brain_id"], "auto_named_project")
        self.assertEqual(health_payload["status"], "ok")
        self.assertIn("register_brain", health_payload["next_actions"])

    def test_brain_runtime_register_attaches_project_to_workspace_manifest_and_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            workspace = Path(raw_tmp) / "workspace"
            project = workspace / "child_alpha"
            (workspace / "brain").mkdir(parents=True)
            for name in (
                "identity_layer.md",
                "state_center.md",
                "knowledge_center.md",
                "master_brain.md",
                "brain_architecture.md",
                "operations_center.md",
                "governance_layer.md",
            ):
                (workspace / "brain" / name).write_text(f"# {name}\n", encoding="utf-8")
            (workspace / "brain/brain_manifest.json").write_text(
                json.dumps(
                    {
                        "brain_type": "main",
                        "entrypoint": "brain/identity_layer.md",
                        "read_order": [
                            "brain/identity_layer.md",
                            "brain/state_center.md",
                            "brain/knowledge_center.md",
                            "brain/master_brain.md",
                            "brain/brain_architecture.md",
                            "brain/operations_center.md",
                            "brain/governance_layer.md",
                        ],
                        "shared_regional_brain_contract": {
                            "default_module_order": [
                                "identity_layer",
                                "state_center",
                                "knowledge_center",
                                "brain_architecture",
                                "operations_center",
                                "governance_layer",
                                "episodic_memory",
                            ],
                            "required_modules": [
                                "identity_layer",
                                "state_center",
                                "knowledge_center",
                                "brain_architecture",
                                "operations_center",
                                "governance_layer",
                                "episodic_memory",
                            ],
                            "fast_handoff_modules": ["identity_layer", "state_center", "knowledge_center", "operations_center"],
                            "region_bindings": {},
                        },
                        "child_brains": [],
                        "write_routes": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [PYTHON, str(RUNTIME), "init", "--cwd", str(project), "--brain-id", "child_alpha"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            result = subprocess.run(
                [PYTHON, str(RUNTIME), "register", "--cwd", str(project), "--workspace-root", str(workspace)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            payload = json.loads(result.stdout)
            main_manifest = json.loads((workspace / "brain/brain_manifest.json").read_text(encoding="utf-8"))
            child_manifest = json.loads((project / "brain/brain_manifest.json").read_text(encoding="utf-8"))
            catalog = json.loads((workspace / "brain/brain_catalog.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(main_manifest["child_brains"][0]["id"], "child_alpha")
        self.assertEqual(child_manifest["brain_type"], "sub_brain")
        self.assertEqual(child_manifest["shared_contract_source"], "brain/brain_manifest.json#shared_regional_brain_contract")
        self.assertIn("child_alpha", {item["brain_id"] for item in catalog["brains"]})

    def test_registered_project_can_be_routed_from_manifest_hints(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            workspace = Path(raw_tmp) / "workspace"
            project = workspace / "child_beta"
            (workspace / "brain").mkdir(parents=True)
            for name in (
                "identity_layer.md",
                "state_center.md",
                "knowledge_center.md",
                "master_brain.md",
                "brain_architecture.md",
                "operations_center.md",
                "governance_layer.md",
            ):
                (workspace / "brain" / name).write_text(f"# {name}\n", encoding="utf-8")
            (workspace / "brain/brain_manifest.json").write_text(
                json.dumps(
                    {
                        "brain_type": "main",
                        "entrypoint": "brain/identity_layer.md",
                        "read_order": ["brain/identity_layer.md"],
                        "shared_regional_brain_contract": {
                            "default_module_order": [
                                "identity_layer",
                                "state_center",
                                "knowledge_center",
                                "brain_architecture",
                                "operations_center",
                                "governance_layer",
                                "episodic_memory",
                            ],
                            "required_modules": [
                                "identity_layer",
                                "state_center",
                                "knowledge_center",
                                "brain_architecture",
                                "operations_center",
                                "governance_layer",
                                "episodic_memory",
                            ],
                            "fast_handoff_modules": ["identity_layer", "state_center", "knowledge_center", "operations_center"],
                            "region_bindings": {},
                        },
                        "child_brains": [],
                        "write_routes": {},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [PYTHON, str(RUNTIME), "init", "--cwd", str(project), "--brain-id", "child_beta"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            child_manifest_path = project / "brain/brain_manifest.json"
            child_manifest = json.loads(child_manifest_path.read_text(encoding="utf-8"))
            child_manifest["routing_hints"]["terms"] = ["beta-special"]
            child_manifest_path.write_text(json.dumps(child_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            subprocess.run(
                [PYTHON, str(RUNTIME), "register", "--cwd", str(project), "--workspace-root", str(workspace)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
            with patch.object(routing, "WORKSPACE_ROOT", workspace), patch.object(routing, "MAIN_MANIFEST", Path("brain/brain_manifest.json")):
                payload = routing.route_task_to_brain("beta-special 修复")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "child_beta")


if __name__ == "__main__":
    unittest.main()
