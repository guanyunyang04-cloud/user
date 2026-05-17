from __future__ import annotations

import unittest

from daily_research.tools.brain_capsule import build_task_capsule


class BrainCapsuleTest(unittest.TestCase):
    def test_capsule_contains_operating_context_and_guard_fields(self) -> None:
        payload = build_task_capsule(
            child="daily_research",
            task="continue r63 brain skill operating system",
            workflow="brain_handoff",
        )

        self.assertEqual(payload["child"], "daily_research")
        self.assertIn("daily_research/brain/state_center.md", payload["fast_handoff_paths"])
        self.assertIn("facts", payload)
        self.assertIn("inferences", payload)
        self.assertIn("assumptions", payload)
        self.assertIn("forbidden_actions", payload)
        self.assertIn("active_artifact_guard", payload)
        self.assertEqual(payload["active_artifact_guard"]["path"], "daily_research/output/active_execution_strategy.json")
        self.assertIn("branch_line", payload["git"])
        self.assertIsInstance(payload["git"]["on_main"], bool)

    def test_capsule_links_relevant_reference_records(self) -> None:
        payload = build_task_capsule(
            child="daily_research",
            task="r62 data lake status",
            workflow="brain_handoff",
        )

        paths = {item["path"] for item in payload["related_references"]}
        self.assertIn("daily_research/brain/references/r62_research_data_lake_status_20260514.md", paths)


if __name__ == "__main__":
    unittest.main()
