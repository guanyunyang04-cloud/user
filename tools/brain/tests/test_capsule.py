from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from tools.brain.capsule import build_task_capsule
from tools.brain.routing import route_task_to_brain


class BrainCapsuleTest(unittest.TestCase):
    def test_route_workspace_governance_task(self) -> None:
        payload = route_task_to_brain("审阅主脑和分脑接管治理")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "workspace")
        self.assertEqual(payload["target"]["id"], "workspace")
        self.assertEqual(payload["target"]["kind"], "workspace")
        self.assertEqual(payload["target"]["domain"], "workspace_governance")
        self.assertIn("workspace_governance", payload["target"]["bootstrap_aliases"])
        self.assertFalse(payload["decision_required"])
        self.assertEqual(payload["confidence"], "high")

    def test_route_brain_routing_policy_stays_workspace_despite_registry_brain_term(self) -> None:
        payload = route_task_to_brain("brain路由机制是否应该由agent思考决定而不是根据提示词关键词决定")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "workspace")
        self.assertEqual(payload["target"]["kind"], "workspace")
        self.assertFalse(payload["decision_required"])
        self.assertEqual(payload["recommended_default"], "workspace")

    def test_route_path20_task_to_daily_research(self) -> None:
        payload = route_task_to_brain("Path20 多 horizon 训练计划")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "daily_research")
        self.assertFalse(payload["decision_required"])
        self.assertEqual(payload["confidence"], "high")

    def test_route_multi_horizon_and_path_policy_terms_to_daily_research(self) -> None:
        cases = (
            "alpha_multi_horizon_utility_policy_v1 根因审计",
            "Alpha Multi-Horizon 长短周期根因评估",
            "path_policy horizon root cause audit",
            "daily_research/path_policy/horizon_root_cause_audit.py 修复",
        )
        for task in cases:
            with self.subTest(task=task):
                payload = route_task_to_brain(task)
                self.assertEqual(payload["status"], "selected")
                self.assertEqual(payload["selected_brain_id"], "daily_research")
                self.assertEqual(payload["target"]["id"], "daily_research")
                self.assertEqual(payload["target"]["kind"], "child")
                self.assertEqual(payload["target"]["domain"], "daily_research")
                self.assertTrue(payload.get("routing_sources"))

    def test_route_qdp_research_pack_training_to_daily_research_with_qdp_support(self) -> None:
        payload = route_task_to_brain("qdp_v2 sequence pack GRU 训练")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "daily_research")
        self.assertEqual(payload["primary_brain_id"], "daily_research")
        self.assertIn("quant_data_platform", payload["supporting_brain_ids"])
        self.assertIn("daily_research/brain/state_center.md", payload["writeback_targets"])
        self.assertIn("brain/state_center.md", payload["writeback_targets"])
        self.assertIn(
            {
                "object": "qdp_v2_active_data_base",
                "owner": "quant_data_platform",
                "mode": "read_only",
                "reason": "source facts",
            },
            payload["object_routes"],
        )
        self.assertTrue(
            any(
                item["object"] == "sequence_training_pack"
                and item["owner"] == "daily_research"
                and item["mode"] == "write"
                for item in payload["object_routes"]
            )
        )

    def test_route_pure_qdp_quality_task_stays_qdp(self) -> None:
        payload = route_task_to_brain("qdp check active.json dataset quality")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "quant_data_platform")
        self.assertEqual(payload["primary_brain_id"], "quant_data_platform")
        self.assertNotIn("daily_research", payload["supporting_brain_ids"])

    def test_route_qdp_update_stays_qdp_even_when_research_is_mentioned(self) -> None:
        payload = route_task_to_brain("qdp update provider ingest for daily_research sequence pack")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "quant_data_platform")
        self.assertEqual(payload["primary_brain_id"], "quant_data_platform")
        self.assertTrue(
            any(
                item["owner"] == "quant_data_platform" and item["mode"] == "write"
                for item in payload["object_routes"]
            )
        )

    def test_route_daily_research_model_using_qdp_active_data_is_cross_project(self) -> None:
        payload = route_task_to_brain("daily_research 模型使用 QDP active 数据")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "daily_research")
        self.assertEqual(payload["primary_brain_id"], "daily_research")
        self.assertIn("quant_data_platform", payload["supporting_brain_ids"])
        self.assertTrue(
            any(
                item["owner"] == "quant_data_platform" and item["mode"] == "read_only"
                for item in payload["object_routes"]
            )
        )

    def test_route_daily_research_execution_terms(self) -> None:
        cases = (
            "执行端交易计划没有动作",
            "数据刷新后 signal panel 过期",
            "模型页 active manifest 状态",
            "daily_research/execution/app_service.py 修复",
            "production signal refresh 闭环",
        )
        for task in cases:
            with self.subTest(task=task):
                payload = route_task_to_brain(task)
                self.assertEqual(payload["status"], "selected")
                self.assertEqual(payload["selected_brain_id"], "daily_research")

    def test_route_preserves_actual_research_and_execution_objects(self) -> None:
        research = route_task_to_brain("评估默认研究主线模型")
        research_objects = {item["object"]: item for item in research["object_routes"]}
        self.assertEqual(research["selected_brain_id"], "daily_research")
        self.assertIn("model_research_evidence", research_objects)
        self.assertNotIn("sequence_training_pack", research_objects)

        execution = route_task_to_brain("切换默认执行策略并写入 active_execution_strategy")
        execution_objects = {item["object"]: item for item in execution["object_routes"]}
        self.assertEqual(execution["selected_brain_id"], "daily_research")
        self.assertEqual(execution_objects["active_execution_artifact"]["mode"], "write")

    def test_capsule_fails_closed_for_missing_active_artifact_on_execution_write(self) -> None:
        missing = {
            "path": "daily_research/output/active_execution_strategy.json",
            "status": "missing",
            "exists": False,
            "tracked": False,
            "ignored": True,
            "clean": False,
            "returncode": 1,
            "diff_line_count": 0,
        }
        with patch("tools.brain.adapters.daily_research.active_artifact_diff_status", return_value=missing):
            payload = build_task_capsule(
                task="切换默认执行策略并写入 active_execution_strategy",
                workflow="auto",
                intent="mutate",
            )

        self.assertFalse(payload["mutation_allowed"])
        self.assertIn("active_artifact_missing", payload["preflight_blockers"])

    def test_capsule_allows_research_mutation_when_missing_active_artifact_is_read_only(self) -> None:
        missing = {
            "path": "daily_research/output/active_execution_strategy.json",
            "status": "missing",
            "exists": False,
            "tracked": False,
            "ignored": True,
            "clean": False,
            "returncode": 1,
            "diff_line_count": 0,
        }
        with patch("tools.brain.adapters.daily_research.active_artifact_diff_status", return_value=missing):
            payload = build_task_capsule(
                task="评估默认研究主线模型",
                workflow="auto",
                intent="mutate",
            )

        self.assertTrue(payload["mutation_allowed"])
        self.assertNotIn("active_artifact_missing", payload["preflight_blockers"])

    def test_route_intraday_task_to_t0_project(self) -> None:
        payload = route_task_to_brain("盘中 RL 原型接管")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "t0_project")

    def test_route_daily_stock_analysis_task_to_product_brain(self) -> None:
        payload = route_task_to_brain("daily_stock_analysis 多市场产品修复")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "daily_stock_analysis-main")

    def test_route_ambiguous_task_does_not_default_to_daily_research(self) -> None:
        payload = route_task_to_brain("Path20 和盘中 RL 联合接管")

        self.assertEqual(payload["status"], "ambiguous")
        self.assertEqual(payload["selected_brain_id"], "")
        self.assertEqual(payload["target"]["kind"], "ambiguous")
        self.assertEqual(payload["target"]["domain"], "")
        self.assertTrue(payload["decision_required"])
        self.assertEqual(payload["recommended_default"], "none")

    def test_route_governance_wording_with_single_hard_child_selects_child(self) -> None:
        payload = route_task_to_brain("接管 traditional_quant_research，检查该分脑是否需要维护")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "traditional_quant_research")
        self.assertEqual(payload["target"]["kind"], "child")
        self.assertEqual(payload["target"]["domain"], "traditional_quant_research")
        self.assertFalse(payload["decision_required"])
        self.assertEqual(payload["recommended_default"], "traditional_quant_research")
        self.assertIn("workspace_governance_signal", payload)

    def test_route_soft_terms_need_agent_decision(self) -> None:
        for task in ("training 复盘", "当前数据集不完整"):
            with self.subTest(task=task):
                payload = route_task_to_brain(task)

                self.assertEqual(payload["status"], "needs_agent_decision")
                self.assertEqual(payload["selected_brain_id"], "")
                self.assertEqual(payload["target"]["kind"], "ambiguous")
                self.assertTrue(payload["decision_required"])
                self.assertEqual(payload["recommended_default"], "workspace")

    def test_capsule_without_child_returns_schema_v4_workspace_context(self) -> None:
        payload = build_task_capsule(task="审阅主脑接管规则", workflow="auto")

        self.assertEqual(payload["schema_version"], 4)
        self.assertIn("agent_meta", payload)
        self.assertIn("agent_review", payload)
        self.assertNotIn("meta_cognition", payload)
        self.assertNotIn("runtime_learning_hooks", payload)
        self.assertEqual(payload["agent_meta"]["actor"], "agent")
        self.assertEqual(payload["agent_meta"]["substrate"], "brain")
        self.assertEqual(payload["agent_meta"]["tool_role"], "sensor")
        self.assertEqual(payload["agent_meta"]["authority"], "propose_only")
        self.assertEqual(payload["agent_meta"]["proposal_creation_policy"], "auto_create_low_risk_proposed_status")
        self.assertEqual(
            payload["agent_meta"]["implementation_approval_policy"],
            "requires_explicit_user_approval_for_protocol_or_behavior_changes",
        )
        self.assertEqual(payload["agent_meta"]["required_passes"], ["task_start", "decision_boundary", "before_final"])
        self.assertEqual(payload["agent_meta"]["review"]["status"], "clear")
        self.assertEqual(payload["agent_meta"]["review"]["next_actions"], ["no_learning_needed"])
        self.assertFalse(payload["agent_review"]["before_final_required"])
        self.assertFalse(payload["agent_review"]["closure_meta_review_required"])
        self.assertIn("closure_review_command", payload["agent_review"])
        self.assertIn("main_context", payload)
        self.assertIn("workspace_context", payload)
        self.assertEqual(payload["routing"]["selected_brain_id"], "workspace")
        self.assertEqual(payload["routing"]["target"]["kind"], "workspace")
        self.assertEqual(payload["target_kind"], "workspace")
        self.assertEqual(payload["workflow_domain"], "workspace_governance")
        self.assertNotIn("child", payload)
        self.assertNotIn("child_context", payload)
        self.assertNotIn("state_summary", payload)
        self.assertNotIn("hard_rules", payload)
        self.assertNotIn("fast_handoff_paths", payload)

    def test_capsule_path20_attaches_daily_research_child_context(self) -> None:
        payload = build_task_capsule(task="Path20 当前状态", workflow="auto")

        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["context_profile"], "lite")
        self.assertEqual(payload["routing"]["selected_brain_id"], "daily_research")
        self.assertEqual(payload["routing"]["target"]["kind"], "child")
        self.assertEqual(payload["target_kind"], "child")
        self.assertEqual(payload["workflow_domain"], "daily_research")
        self.assertIn("child_context", payload)
        self.assertEqual(payload["child_context"]["brain_id"], "daily_research")
        self.assertIn("daily_research/brain/state_center.md", payload["child_context"]["fast_handoff_paths"])
        self.assertIn("active_artifact_guard", payload["guards"])
        self.assertNotIn("state_summary", payload["child_context"])
        self.assertNotIn("hard_rules", payload["child_context"])
        self.assertNotIn("latest_output_runs", payload["guards"].get("frontier_report", {}))

    def test_capsule_qdp_research_task_exposes_brain_orchestration(self) -> None:
        payload = build_task_capsule(task="qdp_v2 sequence pack GRU 训练", workflow="auto")

        self.assertEqual(payload["routing"]["selected_brain_id"], "daily_research")
        self.assertEqual(payload["brain_orchestration"]["primary_brain_id"], "daily_research")
        self.assertIn("quant_data_platform", payload["brain_orchestration"]["supporting_brain_ids"])
        self.assertIn("child_context", payload)
        self.assertEqual(payload["child_context"]["brain_id"], "daily_research")
        self.assertIn("supporting_contexts", payload)
        self.assertTrue(any(item["brain_id"] == "quant_data_platform" for item in payload["supporting_contexts"]))
        self.assertTrue(
            any(
                item["owner"] == "quant_data_platform" and item["mode"] == "read_only"
                for item in payload["brain_orchestration"]["object_routes"]
            )
        )

    def test_capsule_multi_horizon_registry_fallback_attaches_daily_research_child_context(self) -> None:
        payload = build_task_capsule(task="alpha_multi_horizon_utility_policy_v1 根因审计", workflow="auto")

        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(payload["routing"]["selected_brain_id"], "daily_research")
        self.assertEqual(payload["routing"]["target"]["kind"], "child")
        self.assertIn("child_context", payload)
        self.assertEqual(payload["child_context"]["brain_id"], "daily_research")
        self.assertIn("active_artifact_guard", payload["guards"])
        self.assertIn("frontier_report", payload["guards"])
        self.assertIn("registry_exact_match", payload["routing"].get("routing_sources", []))

    def test_route_multi_horizon_study_family_from_registry_to_daily_research(self) -> None:
        payload = route_task_to_brain("stage25_stability_calibration 复盘")

        self.assertEqual(payload["status"], "selected")
        self.assertEqual(payload["selected_brain_id"], "daily_research")
        self.assertEqual(payload["target"]["kind"], "child")
        self.assertIn("registry_exact_match", payload["routing_sources"])

    def test_capsule_lite_exposes_deep_dive_commands(self) -> None:
        payload = build_task_capsule(task="Path20 当前状态", workflow="auto")

        self.assertEqual(payload["context_profile"], "lite")
        self.assertIn("summary_budget", payload)
        commands = "\n".join(payload["available_deep_dive_commands"])
        self.assertIn("current-frontier", commands)
        self.assertIn("query", commands)
        self.assertIn("health --mode full", commands)

    def test_capsule_full_preserves_deep_context(self) -> None:
        payload = build_task_capsule(task="Path20 当前状态", workflow="auto", verbosity="full")

        self.assertEqual(payload["context_profile"], "full")
        self.assertIn("state_summary", payload["child_context"])
        self.assertIn("hard_rules", payload["child_context"])
        self.assertIn("latest_output_runs", payload["guards"]["frontier_report"])

    def test_capsule_workspace_governance_has_no_daily_research_state_summary(self) -> None:
        payload = build_task_capsule(task="主脑 capsule 重构", workflow="auto")

        self.assertEqual(payload["routing"]["selected_brain_id"], "workspace")
        self.assertEqual(payload["routing"]["target"]["domain"], "workspace_governance")
        self.assertNotIn("child_context", payload)
        self.assertIn("workspace_context", payload)
        self.assertNotIn("summary", payload["main_context"])
        self.assertNotIn("hard_rules", payload["main_context"])

    def test_capsule_needs_agent_decision_does_not_attach_child_context(self) -> None:
        payload = build_task_capsule(task="training 复盘", workflow="auto")

        self.assertEqual(payload["routing"]["status"], "needs_agent_decision")
        self.assertEqual(payload["target_kind"], "ambiguous")
        self.assertNotIn("route_needs_agent_decision", payload["preflight_blockers"])
        self.assertIn("agent_selected_brain_id", payload)
        self.assertEqual(payload["agent_selected_brain_id"], "")
        self.assertIn("routing_evidence", payload)
        self.assertNotIn("child_context", payload)
        self.assertNotIn("workspace_context", payload)
        self.assertTrue(any("candidate only:" in command and "daily_research" in command for command in payload["available_deep_dive_commands"]))

    def test_capsule_traditional_quant_governance_wording_uses_project_profile(self) -> None:
        payload = build_task_capsule(task="接管 traditional_quant_research，检查该分脑是否需要维护", workflow="auto")
        encoded_guards = json.dumps(payload["guards"], ensure_ascii=False)
        encoded_payload = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(payload["routing"]["status"], "selected")
        self.assertEqual(payload["agent_selected_brain_id"], "traditional_quant_research")
        self.assertEqual(payload["target_kind"], "child")
        self.assertEqual(payload["workflow_domain"], "traditional_quant_research")
        self.assertIn("child_context", payload)
        self.assertIn("project_profile", payload)
        self.assertEqual(payload["project_profile"]["project_id"], "traditional_quant_research")
        self.assertNotIn("active_artifact_guard", payload["guards"])
        self.assertNotIn("frontier_report", payload["guards"])
        self.assertNotIn("daily_research/output/active_execution_strategy.json", encoded_guards)
        self.assertNotIn("daily_research/output/active_execution_strategy.json", encoded_payload)
        self.assertNotIn("traditional_quant_research/tests -q", encoded_guards)
        self.assertEqual(payload["project_profile"]["verification_profile"]["default_test_commands"], [])

    def test_capsule_polling_wording_stays_in_brain_handoff(self) -> None:
        payload = build_task_capsule(
            task="三组 shadow-only seed7 长训练轮询",
            workflow="auto",
            intent="mutate",
            verbosity="lite",
        )

        self.assertEqual(payload["workflow"], "brain_handoff")
        self.assertNotIn("long_task_contract", payload)
        self.assertIn("guards", payload)
        self.assertIn("preflight_blockers", payload)
        self.assertIn("capability_hints", payload)
        encoded = json.dumps(
            {
                "capability_hints": payload["capability_hints"],
                "risk_signals": payload["risk_signals"],
                "verification_hints": payload["verification_hints"],
            },
            ensure_ascii=False,
        )
        self.assertIn("polling/async task", encoded)
        self.assertIn("best observable handle", encoded)
        self.assertIn("polling_task_without_observable_handle", encoded)
        self.assertNotIn("long_task_monitor", encoded)
        self.assertNotIn("Wait-Process", encoded)

    def test_capsule_training_mixed_with_brain_maintenance_is_not_forbidden(self) -> None:
        payload = build_task_capsule(
            task="修改脑区规则并启动训练",
            workflow="auto",
            intent="mutate",
            verbosity="lite",
        )
        encoded = json.dumps(
            {
                "capability_hints": payload["capability_hints"],
                "risk_signals": payload["risk_signals"],
                "verification_hints": payload["verification_hints"],
            },
            ensure_ascii=False,
        )

        self.assertEqual(payload["workflow"], "brain_maintenance")
        self.assertNotIn("forbidden_" + "actions", payload)
        self.assertNotIn("start_" + "training", encoded)
        self.assertIn("risk_signals", payload)
        self.assertIn("verification_hints", payload)
        self.assertIn("polling/async task", encoded)
        self.assertNotIn("long_task_monitor", encoded)

    def test_capsule_mutate_plan_title_stays_in_brain_handoff(self) -> None:
        payload = build_task_capsule(
            task="将所有未完成计划结合在一起，全部完成",
            workflow="auto",
            intent="mutate",
            verbosity="lite",
        )

        self.assertEqual(payload["workflow"], "brain_handoff")
        self.assertIn("external_skill_signal", payload["workflow_selection"]["decision_sources"])
        self.assertNotIn("self_" + "evolution_hooks", payload)
        self.assertNotIn("runtime_learning_hooks", payload)
        self.assertIn("agent_review", payload)
        self.assertFalse(payload["agent_review"]["before_final_required"])
        self.assertIn("trace_template_command", payload["agent_review"])

    def test_capsule_user_learning_question_exposes_meta_opportunity(self) -> None:
        payload = build_task_capsule(
            task="为什么这个应该学会却没提示，以后都要自动发现",
            workflow="auto",
            intent="read",
            verbosity="lite",
        )

        review = payload["agent_meta"]["review"]
        self.assertEqual(review["status"], "opportunity")
        self.assertEqual(payload["agent_meta"]["authority"], "propose_only")
        self.assertIn("user_correction", review["signals"])
        self.assertIn("learning_opportunity_missed", review["signals"])
        layers = {item["target_layer"] for item in review["learning_opportunities"]}
        self.assertIn("agent_meta_protocol", layers)
        self.assertIn("create_proposal", review["next_actions"])
        opportunity = next(item for item in review["learning_opportunities"] if item["target_layer"] == "agent_meta_protocol")
        self.assertEqual(opportunity["proposal_creation_policy"], "auto_create_low_risk_proposed_status")
        self.assertEqual(opportunity["proposal_status"], "proposed")
        self.assertTrue(opportunity["implementation_requires_user_confirmation"])
        self.assertTrue(payload["agent_review"]["before_final_required"])
        self.assertIn("agent_meta_opportunity", payload["agent_review"]["reason_codes"])
        self.assertTrue(payload["agent_review"]["closure_meta_review_required"])

    def test_capsule_meta_question_request_exposes_closure_review(self) -> None:
        payload = build_task_capsule(
            task="你真的理解我说的元问题吗？重点是任务结束前要意识到问题本身",
            workflow="auto",
            intent="read",
            verbosity="lite",
        )

        review = payload["agent_meta"]["review"]
        self.assertEqual(review["status"], "opportunity")
        self.assertIn("meta_question_discovery_gap", review["signals"])
        self.assertIn("create_proposal", review["next_actions"])
        self.assertNotIn("ask_user_for_evolution", review["next_actions"])
        self.assertTrue(payload["agent_review"]["closure_meta_review_required"])
        self.assertIn("closure_meta_review", payload["agent_review"]["reason_codes"])
        self.assertIn(" review ", payload["agent_review"]["closure_review_command"])
        self.assertIn("--trace-json", payload["agent_review"]["closure_review_command"])

    def test_capsule_actor_boundary_mismatch_exposes_agent_meta_opportunity(self) -> None:
        payload = build_task_capsule(
            task="脑区只是载体，没有思考能力，agent 应通过脑区获得元能力并作用于脑区",
            workflow="auto",
            intent="read",
            verbosity="lite",
        )

        review = payload["agent_meta"]["review"]
        self.assertEqual(payload["schema_version"], 4)
        self.assertEqual(review["status"], "opportunity")
        self.assertIn("actor_boundary_mismatch", review["signals"])
        opportunity = next(
            item for item in review["learning_opportunities"]
            if item["target_layer"] == "agent_meta_protocol"
        )
        self.assertEqual(opportunity["owner_brain"], "workspace")
        self.assertEqual(opportunity["source_signal"], "actor_boundary_mismatch")
        self.assertTrue(payload["agent_review"]["before_final_required"])

    def test_capsule_brain_rule_obstruction_exposes_burden_governance(self) -> None:
        payload = build_task_capsule(
            task="当前脑区规则是否过多，是否阻碍 agent 判断，需要清理冗余兼容和过细测试",
            workflow="auto",
            intent="read",
            verbosity="lite",
        )

        review = payload["agent_meta"]["review"]
        self.assertEqual(review["status"], "opportunity")
        self.assertIn("brain_rule_obstruction", review["signals"])
        opportunity = next(
            item for item in review["learning_opportunities"]
            if item["target_layer"] == "brain_structure_governance"
        )
        self.assertEqual(opportunity["owner_brain"], "workspace")
        self.assertEqual(opportunity["source_signal"], "brain_rule_obstruction")
        self.assertNotIn("actor_boundary_mismatch", review["signals"])

    def test_capsule_multi_horizon_low_budget_review_exposes_domain_signal(self) -> None:
        payload = build_task_capsule(
            task="审阅 multi-horizon 低预算实验是否可作模型质量结论",
            workflow="auto",
            intent="read",
            verbosity="lite",
        )

        review = payload["agent_meta"]["review"]
        self.assertEqual(review["status"], "opportunity")
        self.assertIn("low_budget_evidence_pollution", review["signals"])
        opportunity = next(
            item for item in review["learning_opportunities"]
            if item["target_layer"] == "experiment_governance"
        )
        self.assertEqual(opportunity["owner_brain"], "daily_research")
        self.assertEqual(opportunity["writeback_route"], "daily_research/brain/knowledge_center.md")
        self.assertTrue(opportunity["verification_required"])

    def test_capsule_research_scope_grade_review_exposes_closure_signal(self) -> None:
        payload = build_task_capsule(
            task="详细审阅 multi-horizon 研究结论，确认 cap80 diagnostic 没有被当作 full-pool evidence 或 evidence-grade gate pass",
            workflow="auto",
            intent="read",
            verbosity="lite",
        )

        review = payload["agent_meta"]["review"]
        self.assertEqual(review["status"], "opportunity")
        self.assertIn("research_scope_grade_alignment_gap", review["signals"])
        opportunity = next(
            item for item in review["learning_opportunities"]
            if item["target_layer"] == "research_conclusion_gate"
        )
        self.assertEqual(opportunity["owner_brain"], "workspace")
        self.assertEqual(opportunity["writeback_route"], "brain/governance_layer.md")
        self.assertTrue(payload["agent_review"]["before_final_required"])
        self.assertTrue(payload["agent_review"]["closure_meta_review_required"])

    def test_capsule_brain_rule_mutation_uses_brain_maintenance(self) -> None:
        payload = build_task_capsule(
            task="修改脑区规则",
            workflow="auto",
            intent="mutate",
            verbosity="lite",
        )

        self.assertEqual(payload["workflow"], "brain_maintenance")
        self.assertEqual(payload["routing"]["selected_brain_id"], "workspace")
        self.assertEqual(payload["routing"]["target"]["kind"], "workspace")
        self.assertFalse(payload["agent_review"]["before_final_required"])
        self.assertNotIn("workflow_completion_review", payload["agent_review"]["reason_codes"])
        self.assertFalse(payload["agent_review"]["closure_meta_review_required"])


if __name__ == "__main__":
    unittest.main()
