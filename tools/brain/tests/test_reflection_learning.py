from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
RUNTIME = ROOT / "brain/skills/workspace-brain/scripts/brain_runtime.py"
REFLECTION = ROOT / "brain/skills/workspace-brain/scripts/reflection_learning.py"


def load_reflection_module():
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("workspace_brain_reflection_learning", REFLECTION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_trace(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json")
    with tmp:
        json.dump(payload, tmp, ensure_ascii=False)
    return Path(tmp.name)


class ReflectionLearningTest(unittest.TestCase):
    def test_required_step_skipped_generates_completion_gate_candidate(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "继续实施计划，到最后启动长训练",
            "planned_steps": [
                {"id": "train", "title": "启动训练", "expected_outcome": "pid logged", "required": True}
            ],
            "events": [{"type": "step_skipped", "step_id": "train", "summary": "ended before training"}],
            "final_state": {"completed": True, "skipped_steps": ["train"], "unresolved_blockers": [], "user_nudges": [], "verification": []},
        }

        payload = module.analyze_trace(trace)

        self.assertEqual(payload["analysis_mode"], "structured_trace")
        self.assertTrue(payload["learning_candidates"])
        candidate = payload["learning_candidates"][0]
        self.assertEqual(candidate["target_layer"], "execution_completion_gate")
        self.assertEqual(candidate["confidence"], "high")
        self.assertIn("train", candidate["supporting_events"][0]["step_id"])
        self.assertTrue(candidate["suggested_tests"])
        self.assertEqual(candidate["proposal_creation_policy"], "auto_create_low_risk_proposed_status")
        self.assertEqual(candidate["proposal_status"], "proposed")
        self.assertTrue(candidate["implementation_requires_user_confirmation"])

    def test_tool_failure_followed_by_manual_workaround_generates_tool_contract_gap(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "登记 path_policy evidence",
            "planned_steps": [{"id": "writeback", "title": "run writeback-plan", "expected_outcome": "route found", "required": True}],
            "events": [
                {"type": "tool_failure", "step_id": "writeback", "summary": "writeback-plan searched wrong evidence domain", "command": "writeback-plan"},
                {"type": "manual_workaround", "step_id": "writeback", "summary": "manually read path_policy study summary"},
            ],
            "final_state": {"completed": True, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": [], "verification": []},
        }

        payload = module.analyze_trace(trace)

        candidates = payload["learning_candidates"]
        self.assertIn("tool_contract_gap", {candidate["target_layer"] for candidate in candidates})
        candidate = next(candidate for candidate in candidates if candidate["target_layer"] == "tool_contract_gap")
        self.assertIn("anti_overfit_check", candidate)
        self.assertIn("manual_workaround", {event["type"] for event in candidate["supporting_events"]})

    def test_user_nudge_about_unstarted_training_uses_trace_structure(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "执行计划，最后启动训练",
            "planned_steps": [{"id": "train", "title": "启动训练", "expected_outcome": "pid logged", "required": True}],
            "events": [{"type": "user_nudge", "step_id": "train", "summary": "还没启动训练，需要继续实施计划"}],
            "final_state": {"completed": True, "skipped_steps": ["train"], "unresolved_blockers": [], "user_nudges": ["还没启动训练"], "verification": []},
        }

        payload = module.analyze_trace(trace)

        candidate = payload["learning_candidates"][0]
        self.assertEqual(candidate["target_layer"], "execution_completion_gate")
        self.assertEqual(candidate["confidence"], "high")

    def test_completed_trace_with_passing_verification_has_no_learning_candidates(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "完成普通脑区维护",
            "planned_steps": [{"id": "test", "title": "run tests", "expected_outcome": "pass", "required": True}],
            "events": [{"type": "verification_passed", "step_id": "test", "summary": "tests passed"}],
            "final_state": {"completed": True, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": [], "verification": [{"name": "tests", "status": "passed"}]},
        }

        payload = module.analyze_trace(trace)

        self.assertEqual(payload["learning_candidates"], [])
        self.assertEqual(payload["meta_question_candidates"], [])

    def test_review_trace_normalizes_legacy_poll_events(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "polling training",
            "planned_steps": [{"id": "train", "title": "train", "expected_outcome": "completed run", "required": True}],
            "events": [
                {
                    "type": "long_task_poll",
                    "step_id": "train",
                    "summary": "poll returned while training was still running",
                    "evidence": "progress_percent=50 eta_status=estimated",
                    "run_tag": "run_01",
                    "pid": 123,
                    "poll_window_seconds": 0,
                    "progress_path": "forecast_progress.json",
                    "eta_status": "estimated",
                    "decision": "continue_adaptive_polling",
                }
            ],
            "final_state": {"completed": False},
        }

        normalized = module.normalize_trace(trace)
        payload = module.analyze_trace(trace)

        self.assertEqual(normalized["events"][0]["type"], "polling_task_poll")
        self.assertEqual(normalized["events"][0]["run_tag"], "run_01")
        self.assertEqual(normalized["events"][0]["poll_window_seconds"], "0")
        self.assertEqual(payload["next_actions"], ["no_agent_learning_needed"])

    def test_human_feedback_overrode_tool_clear_generates_evaluation_meta_candidate(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "评估 agent meta audit 是否说明无问题",
            "planned_steps": [{"id": "review", "title": "review meta audit", "expected_outcome": "closure judged", "required": True}],
            "events": [
                {
                    "type": "human_feedback_overrode_tool_clear",
                    "step_id": "review",
                    "summary": "agent_meta_audit returned clear, but user pointed out the agent missed the meta problem",
                    "evidence": "工具 clear 与人类反馈冲突",
                }
            ],
            "final_state": {"completed": True, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": ["工具 clear 不是免责"], "verification": []},
        }

        payload = module.analyze_trace(trace)

        candidates = payload["meta_question_candidates"]
        self.assertTrue(candidates)
        candidate = candidates[0]
        self.assertEqual(candidate["meta_layer"], "evaluation")
        self.assertEqual(candidate["object_level_issue"], "agent_meta_audit returned clear, but user pointed out the agent missed the meta problem")
        self.assertIn("clear", candidate["meta_question"])
        self.assertTrue(candidate["auto_create_proposal"])
        self.assertTrue(candidate["implementation_requires_user_confirmation"])
        self.assertEqual(candidate["confidence"], "high")
        self.assertIn("create_agent_learning_proposal", payload["next_actions"])
        self.assertNotIn("ask_user_for_evolution", payload["next_actions"])

    def test_meta_question_missed_generates_learning_salience_candidate(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "回答用户指出的执行边界误读",
            "planned_steps": [{"id": "answer", "title": "explain mistake", "expected_outcome": "meta issue recognized", "required": True}],
            "events": [
                {
                    "type": "meta_question_missed",
                    "step_id": "answer",
                    "summary": "agent fixed the object-level answer but did not recognize this as a learnable meta-cognitive failure",
                    "evidence": "用户指出：你没有意识到问题本身",
                }
            ],
            "final_state": {"completed": True, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": ["没有意识到元问题"], "verification": []},
        }

        payload = module.analyze_trace(trace)

        candidates = payload["meta_question_candidates"]
        self.assertTrue(candidates)
        candidate = candidates[0]
        self.assertEqual(candidate["meta_layer"], "learning_salience")
        self.assertIn("可泛化", candidate["why_it_matters"])
        self.assertTrue(candidate["auto_create_proposal"])
        self.assertTrue(candidate["implementation_requires_user_confirmation"])
        self.assertEqual(candidate["confidence"], "high")

    def test_handoff_conflict_is_authority_order_example_not_special_case(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "执行用户明确要求的训练计划",
            "planned_steps": [{"id": "train", "title": "run planned training", "expected_outcome": "training launched", "required": True}],
            "events": [
                {
                    "type": "closure_boundary_misread",
                    "step_id": "train",
                    "summary": "agent treated handoff conservatism as higher authority than the user's explicit plan",
                    "evidence": "handoff said no training; user plan said default full training",
                    "target_layer": "authority_order",
                }
            ],
            "final_state": {"completed": True, "skipped_steps": ["train"], "unresolved_blockers": [], "user_nudges": ["为什么不训练"], "verification": []},
        }

        payload = module.analyze_trace(trace)

        candidates = payload["meta_question_candidates"]
        self.assertTrue(candidates)
        candidate = candidates[0]
        self.assertEqual(candidate["meta_layer"], "authority")
        self.assertNotIn("handoff", candidate["meta_layer"])
        self.assertTrue(candidate["auto_create_proposal"])
        self.assertTrue(candidate["implementation_requires_user_confirmation"])

    def test_one_off_environment_failure_does_not_create_persistent_learning(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "run check",
            "planned_steps": [{"id": "check", "title": "run check", "expected_outcome": "pass", "required": True}],
            "events": [{"type": "tool_failure", "step_id": "check", "summary": "network hiccup", "evidence": "one_off_environment_failure"}],
            "final_state": {"completed": False, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": [], "verification": []},
        }

        payload = module.analyze_trace(trace)

        self.assertEqual(payload["learning_candidates"], [])
        self.assertEqual(payload["next_actions"], ["no_persistent_learning"])

    def test_budget_reliability_gap_generates_experiment_governance_candidate(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "审阅 multi-horizon 实验结果",
            "planned_steps": [{"id": "review", "title": "review evidence", "expected_outcome": "budget graded", "required": True}],
            "events": [
                {
                    "type": "budget_reliability_gap",
                    "step_id": "review",
                    "summary": "epochs=2 max_epochs_reached best_epoch=2 single seed used as model-quality evidence",
                    "owner_brain": "daily_research",
                    "target_layer": "experiment_governance",
                    "evidence_grade": "scout_only",
                }
            ],
            "final_state": {"completed": True, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": [], "verification": []},
        }

        payload = module.analyze_trace(trace)

        candidates = payload["learning_candidates"]
        self.assertIn("experiment_governance", {candidate["target_layer"] for candidate in candidates})
        candidate = next(candidate for candidate in candidates if candidate["target_layer"] == "experiment_governance")
        self.assertEqual(candidate["confidence"], "high")
        self.assertEqual(candidate["supporting_events"][0]["owner_brain"], "daily_research")

    def test_research_scope_grade_mismatch_generates_conclusion_gate_candidate(self) -> None:
        module = load_reflection_module()
        trace = {
            "task": "审阅 multi-horizon 研究结论是否被错误证据支撑",
            "planned_steps": [
                {
                    "id": "audit",
                    "title": "audit evidence scope and grade",
                    "expected_outcome": "conclusion wording downgraded when scope is diagnostic",
                    "required": True,
                }
            ],
            "events": [
                {
                    "type": "research_scope_grade_mismatch",
                    "step_id": "audit",
                    "summary": "cap80_diagnostic metrics were at risk of being read as full-pool model-quality evidence",
                    "evidence": "universe_scope=cap80_diagnostic; claimed_conclusion=full_rolling_liquid500 gate pass",
                    "owner_brain": "daily_research",
                    "target_layer": "research_conclusion_gate",
                    "evidence_grade": "diagnostic_only",
                    "confidence": "high",
                }
            ],
            "final_state": {"completed": True, "skipped_steps": [], "unresolved_blockers": [], "user_nudges": [], "verification": []},
        }

        payload = module.analyze_trace(trace)

        candidates = payload["learning_candidates"]
        self.assertTrue(candidates)
        candidate = candidates[0]
        self.assertEqual(candidate["target_layer"], "research_conclusion_gate")
        self.assertIn("scope", candidate["lesson"])
        self.assertIn("universe_scope", candidate["recommended_change"])
        self.assertEqual(candidate["confidence"], "high")
        self.assertEqual(candidate["supporting_events"][0]["evidence_grade"], "diagnostic_only")
        self.assertTrue(payload["meta_question_candidates"])
        self.assertEqual(payload["meta_question_candidates"][0]["meta_layer"], "criterion")

    def test_analyze_meta_signals_detects_learning_opportunity_missed(self) -> None:
        module = load_reflection_module()

        payload = module.analyze_agent_meta_signals(
            task="这应该学会，为什么没提示，以后都要自动发现",
            capsule_context={"target_kind": "workspace", "workflow_domain": "workspace_governance"},
        )

        self.assertEqual(payload["status"], "opportunity")
        self.assertEqual(payload["authority"], "propose_only")
        self.assertIn("learning_opportunity_missed", payload["signals"])
        self.assertIn("create_proposal", payload["next_actions"])
        self.assertTrue(payload["learning_opportunities"])
        opportunity = payload["learning_opportunities"][0]
        self.assertEqual(opportunity["target_layer"], "agent_meta_protocol")
        self.assertEqual(opportunity["proposal_creation_policy"], "auto_create_low_risk_proposed_status")
        self.assertEqual(opportunity["proposal_status"], "proposed")
        self.assertTrue(opportunity["implementation_requires_user_confirmation"])

    def test_analyze_agent_meta_signals_detects_actor_boundary_mismatch(self) -> None:
        module = load_reflection_module()

        payload = module.analyze_agent_meta_signals(
            task="脑区只是载体，没有思考能力，真正思考并执行的是 agent",
            capsule_context={"target_kind": "workspace", "workflow_domain": "workspace_governance"},
        )

        self.assertEqual(payload["status"], "opportunity")
        self.assertIn("actor_boundary_mismatch", payload["signals"])
        opportunity = next(item for item in payload["learning_opportunities"] if item["target_layer"] == "agent_meta_protocol")
        self.assertEqual(opportunity["source_signal"], "actor_boundary_mismatch")

    def test_analyze_agent_meta_signals_detects_brain_rule_obstruction(self) -> None:
        module = load_reflection_module()

        payload = module.analyze_agent_meta_signals(
            task="脑区规则太多，测试过细，旧兼容入口让 agent 变成 checklist runner，需要判断规则是否阻碍任务",
            capsule_context={"target_kind": "workspace", "workflow_domain": "workspace_governance"},
        )

        self.assertEqual(payload["status"], "opportunity")
        self.assertIn("brain_rule_obstruction", payload["signals"])
        opportunity = next(item for item in payload["learning_opportunities"] if item["target_layer"] == "brain_burden_governance")
        self.assertEqual(opportunity["source_signal"], "brain_rule_obstruction")
        self.assertEqual(opportunity["owner_brain"], "workspace")

    def test_analyze_meta_signals_detects_research_scope_grade_alignment_gap(self) -> None:
        module = load_reflection_module()

        payload = module.analyze_agent_meta_signals(
            task="审阅研究结论，确认 cap80 diagnostic 没有被当作 full-pool evidence 或 evidence-grade gate pass",
            capsule_context={
                "target_kind": "child",
                "workflow_domain": "daily_research",
                "routing": {"selected_brain_id": "daily_research"},
            },
        )

        self.assertEqual(payload["status"], "opportunity")
        self.assertIn("research_scope_grade_alignment_gap", payload["signals"])
        opportunity = next(item for item in payload["learning_opportunities"] if item["target_layer"] == "research_conclusion_gate")
        self.assertEqual(opportunity["owner_brain"], "workspace")
        self.assertEqual(opportunity["source_signal"], "research_scope_grade_alignment_gap")
        self.assertIn("scope", opportunity["recommended_action"])
        self.assertEqual(opportunity["proposal_creation_policy"], "auto_create_low_risk_proposed_status")
        self.assertTrue(opportunity["implementation_requires_user_confirmation"])

    def test_analyze_meta_signals_detects_low_budget_evidence_pollution(self) -> None:
        module = load_reflection_module()

        payload = module.analyze_agent_meta_signals(
            task="审阅 multi-horizon 低预算实验是否可作模型质量结论",
            capsule_context={
                "target_kind": "child",
                "workflow_domain": "daily_research",
                "routing": {"selected_brain_id": "daily_research"},
            },
        )

        self.assertEqual(payload["status"], "opportunity")
        self.assertIn("low_budget_evidence_pollution", payload["signals"])
        opportunity = next(item for item in payload["learning_opportunities"] if item["target_layer"] == "experiment_governance")
        self.assertEqual(opportunity["owner_brain"], "daily_research")

    def test_review_cli_accepts_trace_json(self) -> None:
        trace_path = write_trace(
            {
                "task": "继续实施计划，到最后启动长训练",
                "planned_steps": [{"id": "train", "title": "启动训练", "expected_outcome": "pid logged", "required": True}],
                "events": [{"type": "step_skipped", "step_id": "train", "summary": "ended before training"}],
                "final_state": {"completed": True, "skipped_steps": ["train"], "unresolved_blockers": [], "user_nudges": [], "verification": []},
            }
        )
        try:
            result = subprocess.run(
                [PYTHON, str(RUNTIME), "review", "--cwd", str(ROOT), "--trace-json", str(trace_path), "--json"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
        finally:
            trace_path.unlink(missing_ok=True)
        payload = json.loads(result.stdout)

        self.assertEqual(payload["analysis_mode"], "structured_trace")
        self.assertTrue(payload["learning_candidates"])

    def test_freeform_review_is_low_confidence_fallback(self) -> None:
        result = subprocess.run(
            [
                PYTHON,
                str(RUNTIME),
                "review",
                "--cwd",
                str(ROOT),
                "--observation",
                "用户指出 Start-Sleep 被用作长任务轮询",
                "--json",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["analysis_mode"], "freeform_fallback")
        self.assertEqual(payload["evidence_gaps"], ["structured_trace_missing"])
        self.assertTrue(payload["learning_candidates"])
        self.assertEqual(payload["learning_candidates"][0]["confidence"], "low")

    def test_freeform_completion_review_does_not_match_eta_inside_targeted_or_completed_sync(self) -> None:
        module = load_reflection_module()

        payload = module.analyze_freeform(
            task="simplify agent learning proposal flow",
            observation=(
                "Implemented policy split. Targeted tests, skill sync, doc guard, "
                "integrity check, and agent meta audit completed."
            ),
        )

        self.assertEqual(payload["learning_candidates"], [])
        self.assertEqual(payload["next_actions"], ["no_agent_learning_needed"])

    def test_reflection_template_cli_outputs_trace_schema(self) -> None:
        result = subprocess.run(
            [PYTHON, str(RUNTIME), "reflection-template", "--json"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        payload = json.loads(result.stdout)

        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("agent_meta_passes", payload)
        self.assertIn("planned_steps", payload)
        self.assertIn("events", payload)
        self.assertIn("final_state", payload)


if __name__ == "__main__":
    unittest.main()
