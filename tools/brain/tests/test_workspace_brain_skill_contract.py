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
        self.assertIn("needs_agent_decision", text)
        self.assertIn("Do not read a child brain because of one generic term", text)

    def test_skill_keeps_manifest_as_truth_source(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("brain/brain_manifest.json", text)
        self.assertIn("contract truth", text)
        self.assertIn("manifest and catalog", text)

    def test_skill_exposes_project_scope_for_parallel_agents(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("Project Scope", text)
        self.assertIn("project_profile", text)
        self.assertIn("Every task first binds to one `project_id`", text)
        self.assertIn("temporary reports, logs, short-run artifacts, and diagnostics", text)
        self.assertIn("status summaries", text)
        self.assertIn("operating contract", text)
        self.assertIn("external parallel work", text)
        self.assertIn("tools.brain.project_commit", text)
        self.assertIn("ignored_external_paths", text)
        self.assertIn("<project>/output/agent_runs/<run_id>/", text)
        self.assertIn("explicit lease", text)

    def test_child_brain_operations_inherit_project_namespace_contract(self) -> None:
        child_paths = [
            ROOT / "daily_research/brain/operations_center.md",
            ROOT / "t0_project/brain/operations_center.md",
            ROOT / "daily_stock_analysis-main/brain/operations_center.md",
            ROOT / "traditional_quant_research/brain/operations_center.md",
        ]
        for path in child_paths:
            text = path.read_text(encoding="utf-8")
            self.assertIn("项目任务命名空间", text, msg=str(path))
            self.assertIn("不下钻", text, msg=str(path))
            self.assertIn("lease", text, msg=str(path))

        daily_ops = (ROOT / "daily_research/brain/operations_center.md").read_text(encoding="utf-8")
        self.assertIn("tools.brain.agent_run", daily_ops)
        self.assertNotIn("Start-Process -PassThru", daily_ops)

    def test_skill_exposes_personal_researcher_direct_change_default(self) -> None:
        text = SKILL.read_text(encoding="utf-8")

        self.assertIn("Personal Researcher Direct Change", text)
        self.assertIn("direct rewrite", text)
        self.assertIn("Compatibility is evidence-gated", text)
        self.assertIn("stale shells/tests/helpers", text)

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

    def test_workspace_brain_skill_first_move_is_lite(self) -> None:
        text = SKILL.read_text(encoding="utf-8")
        line_count = len(text.splitlines())

        self.assertIn("capsule --task", text)
        self.assertIn("--verbosity lite", text)
        self.assertIn("health --cwd . --mode compact", text)
        self.assertIn("health --cwd . --mode full --timeout-sec 60", text)
        self.assertIn("fast takeover summary", text)
        self.assertIn("first hop stays `detect` + lite capsule", text)
        self.assertIn("project-profile guards only for maintenance", text)
        self.assertIn("agent-meta-audit", text)
        self.assertIn("brain-burden-audit", text)
        self.assertIn("list-proposals", text)
        self.assertIn("implementation", text)
        self.assertIn("approval", text)
        self.assertIn("proposed or approved", text)
        self.assertLessEqual(line_count, 75)

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



if __name__ == "__main__":
    unittest.main()
