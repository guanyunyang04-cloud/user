from __future__ import annotations

import unittest

from tools.brain.adapters.daily_research import detect_low_budget_evidence


class DailyResearchMetaCognitionTest(unittest.TestCase):
    def test_low_budget_training_summary_is_scout_only_signal(self) -> None:
        payload = detect_low_budget_evidence(
            task="审阅 multi-horizon 低预算实验是否可作模型质量结论",
            training_summary={
                "status": "completed",
                "training_config": {"epochs": 2},
                "models": {
                    "gru_sequence_static_context": {
                        "seed_summaries": {
                            "7": {
                                "seed": 7,
                                "epochs_ran": 2,
                                "best_epoch": 2,
                                "stopped_reason": "max_epochs_reached",
                            }
                        }
                    }
                },
            },
        )

        self.assertEqual(payload["status"], "opportunity")
        self.assertIn("low_budget_evidence_pollution", payload["signals"])
        self.assertEqual(payload["evidence_grade"], "scout_only")
        opportunity = payload["learning_opportunities"][0]
        self.assertEqual(opportunity["owner_brain"], "daily_research")
        self.assertEqual(opportunity["target_layer"], "experiment_governance")
        self.assertTrue(opportunity["verification_required"])

    def test_unrelated_task_does_not_emit_domain_signal(self) -> None:
        payload = detect_low_budget_evidence(
            task="审阅主脑结构",
            training_summary={
                "training_config": {"epochs": 2},
                "models": {
                    "m": {
                        "seed_summaries": {
                            "7": {"epochs_ran": 2, "best_epoch": 2, "stopped_reason": "max_epochs_reached"}
                        }
                    }
                },
            },
        )

        self.assertEqual(payload["status"], "clear")
        self.assertEqual(payload["signals"], [])


if __name__ == "__main__":
    unittest.main()
