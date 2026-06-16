from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"
SKILL = ROOT / "brain/skills/workspace-brain/SKILL.md"
RUNTIME = ROOT / "brain/skills/workspace-brain/scripts/brain_runtime.py"


class WorkspaceBrainSkillContractTest(unittest.TestCase):
    def test_skill_frontmatter_and_body_are_procedural(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertTrue(text.startswith("---\nname: workspace-brain"))
        self.assertIn("description: Use when", text)
        self.assertIn("脑区", text)
        self.assertIn("agent learning", text.lower())
        self.assertIn("-m tools.brain.workflow capsule", text)
        self.assertIn("-m tools.brain.workflow route", text)
        self.assertIn("-m tools.brain.workflow bootstrap", text)
        self.assertIn("brain_runtime.py register", text)
        self.assertIn("brain_runtime.py", text)
        self.assertIn("brain_runtime.py health", text)
        self.assertNotIn("r10-r52", text)

    def test_skill_treats_route_as_sensor_and_agent_decision_boundary(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("`route` is a sensor", text)
        self.assertIn("not the final thinker", text)
        self.assertIn("use agent judgment", text)
        self.assertIn("no route result is required", text)

    def test_skill_keeps_stable_workspace_memory_in_skill(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("QDP owns the shared canonical data substrate", text)
        self.assertIn("daily_research", text)
        self.assertIn("active artifact evidence", text)

    def test_skill_keeps_project_helpers_optional(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("Optional Diagnostics", text)
        self.assertIn("Use these when they reduce uncertainty", text)
        self.assertIn("tools.brain.project_commit", text)
        self.assertIn("optional helpers", text)
        self.assertNotIn("Every task first binds", text)
        self.assertNotIn("project_profile", text)

    def test_daily_and_qdp_docs_keep_minimal_fact_boundary(self) -> None:
        daily_ops = (ROOT / "daily_research/brain/operations_center.md").read_text(encoding="utf-8")
        qdp_state = (ROOT / "quant_data_platform/brain/state_center.md").read_text(encoding="utf-8")

        self.assertIn("QDP", daily_ops)
        self.assertIn("canonical", daily_ops)
        self.assertIn("canonical", qdp_state)
        self.assertNotIn("Start-Process -PassThru", daily_ops)

    def test_skill_exposes_personal_researcher_direct_change_default(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("Direct Change And Verification", text)
        self.assertIn("objective-first direct changes", text)
        self.assertIn("wrappers, fallback modes, compatibility layers", text)

    def test_skill_keeps_self_evolution_proposal_only(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("Proposal-Only Evolution", text)
        self.assertIn("proposed", text)
        self.assertIn("explicit user approval", text)

    def test_skill_does_not_offer_generic_workflow_fallback(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertNotIn("API Fallback", text)
        self.assertNotIn("workflow-guide", text)
        self.assertNotIn("--intent " + "long_" + "task", text)
        self.assertNotIn("Superpowers plugin body", text)

    def test_workspace_brain_skill_keeps_diagnostics_optional_and_light(self) -> None:
        text = SKILL.read_text(encoding="utf-8")
        line_count = len(text.splitlines())

        self.assertIn("capsule --task", text)
        self.assertIn("--verbosity lite", text)
        self.assertIn("health --cwd . --mode compact", text)
        self.assertIn("health --cwd . --mode full --timeout-sec 60", text)
        self.assertIn("Use compact health for quick orientation", text)
        self.assertIn("not as mandatory first moves", text)
        self.assertIn("agent-meta-audit", text)
        self.assertIn("brain-burden-audit", text)
        self.assertIn("list-proposals", text)
        self.assertIn("implementation", text)
        self.assertIn("approval", text)
        self.assertLessEqual(line_count, 65)

    def test_workspace_brain_skill_has_no_domain_policy_bloat(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertNotIn("research_programs", text)
        self.assertNotIn("scope-grade alignment", text)
        self.assertIn("agent-meta-audit", text)
        self.assertNotIn("Wait-Process -Id <pid> -Timeout 7200", text)
        self.assertNotIn("Start-Sleep", text)
        self.assertNotIn("start_" + "training", text)
        self.assertNotIn("Do not " + "modify", text)
        self.assertNotIn("\u4e0d\u5f97", text)
        self.assertNotIn("\u7981\u6b62", text)

    def test_workspace_entry_docs_keep_capsule_optional(self) -> None:
        paths = [
            ROOT / "README.md",
            ROOT / "daily_research/README.md",
            ROOT / "brain/skills/workspace-brain/agents/openai.yaml",
        ]

        for path in paths:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                self.assertNotIn("Run the workspace brain capsule first", text)
                self.assertNotIn("接管必须先从工作区主脑进入", text)
                self.assertNotIn("先运行主脑 capsule", text)

        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        daily_readme = (ROOT / "daily_research/README.md").read_text(encoding="utf-8")
        agent_prompt = (ROOT / "brain/skills/workspace-brain/agents/openai.yaml").read_text(encoding="utf-8")

        self.assertIn("可选诊断入口", root_readme)
        self.assertIn("capsule / route 可辅助判断", daily_readme)
        self.assertIn("optional diagnostics", agent_prompt)



if __name__ == "__main__":
    unittest.main()
