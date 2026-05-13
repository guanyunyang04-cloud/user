from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from daily_research.continuous_policy.behavior_bottleneck_report import (
    build_behavior_bottleneck_report,
)


class BehaviorBottleneckReportTest(unittest.TestCase):
    def test_report_prioritizes_cash_timing_and_sell_intent_blockers(self) -> None:
        protocol_summary = {
            "run_tag": "trial_02",
            "evaluation": {
                "continuous_policy_metrics": {
                    "annual_return": 0.153993,
                    "monthly_return_mean": 0.010115,
                    "max_drawdown": -0.106873,
                },
                "continuity_metrics": {
                    "cash_timing_quality_1d": -0.168248,
                    "reduce_success_rate_5d": 0.0,
                    "exit_timeliness_rate_5d": 0.0,
                    "portfolio_daily_source_target_count": 0.0,
                    "portfolio_daily_source_realized_sell_rate": 0.0,
                    "portfolio_daily_actual_cash_weight_mean": 0.188532,
                    "portfolio_daily_exposure_utilization": 1.082835,
                    "portfolio_daily_target_sum_gap": 0.004812,
                    "intent_translation_conflict_rate": 0.0,
                },
            },
            "training_evidence": {
                "status": "insufficient",
                "best_epoch": 8,
                "completed_epochs": 8,
            },
            "promotion_gate": {
                "status": "shadow_only",
                "failed_checks": [
                    "training_evidence_sufficient",
                    "reduce_success_rate_5d",
                    "exit_timeliness_rate_5d",
                    "cash_timing_quality_1d",
                ],
            },
        }

        report = build_behavior_bottleneck_report(protocol_summary)

        self.assertEqual(report["run_tag"], "trial_02")
        self.assertEqual(report["primary_blocker"], "cash_timing_negative")
        self.assertIn("sell_intent_dead", report["blockers"])
        self.assertIn("best_epoch_at_edge", report["blockers"])
        self.assertGreater(report["progress_assets"]["allocation_closure_score"], 0.90)

    def test_study_helper_writes_report_next_to_protocol_summary(self) -> None:
        from daily_research.continuous_policy.run_self_optimizing_study import _write_behavior_bottleneck_report

        with TemporaryDirectory() as tmpdir:
            protocol_path = Path(tmpdir) / "protocol_summary.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "run_tag": "r55_trial",
                        "evaluation": {
                            "continuity_metrics": {
                                "cash_timing_quality_1d": -0.12,
                                "portfolio_daily_source_target_count": 0.0,
                                "portfolio_daily_source_realized_sell_rate": 0.0,
                            }
                        },
                        "training_evidence": {"status": "insufficient"},
                        "promotion_gate": {"failed_checks": ["cash_timing_quality_1d"]},
                    }
                ),
                encoding="utf-8",
            )

            output_path = _write_behavior_bottleneck_report(protocol_path)

            self.assertTrue(output_path.endswith("behavior_bottleneck_report.json"))
            report = json.loads(Path(output_path).read_text(encoding="utf-8"))
            self.assertEqual(report["primary_blocker"], "cash_timing_negative")

    def test_missing_intent_translation_metric_is_not_reported_as_clean(self) -> None:
        report = build_behavior_bottleneck_report(
            {
                "run_tag": "missing_diagnostics",
                "evaluation": {
                    "continuity_metrics": {
                        "portfolio_daily_source_target_count": 0.0,
                        "portfolio_daily_source_realized_sell_rate": 0.0,
                    }
                },
                "training_evidence": {"status": "insufficient"},
                "promotion_gate": {"failed_checks": []},
            }
        )

        self.assertIn("sell_intent_dead", report["blockers"])
        self.assertFalse(report["progress_assets"]["intent_translation_clean"])

    def test_enriched_intent_translation_metric_drives_report(self) -> None:
        report = build_behavior_bottleneck_report(
            {
                "run_tag": "enriched_diagnostics",
                "evaluation": {
                    "continuity_metrics": {
                        "cash_timing_quality_1d": -0.01,
                        "reduce_success_rate_5d": 0.5,
                        "exit_timeliness_rate_5d": 0.5,
                        "portfolio_daily_source_target_count": 1.0,
                        "portfolio_daily_source_realized_sell_rate": 0.5,
                        "portfolio_daily_exposure_utilization": 0.9,
                        "portfolio_daily_target_sum_gap": 0.02,
                        "intent_translation_conflict_rate": 0.25,
                    }
                },
                "training_evidence": {"status": "sufficient", "best_epoch": 4, "completed_epochs": 8},
                "promotion_gate": {"failed_checks": []},
            }
        )

        self.assertEqual(report["metrics"]["intent_translation_conflict_rate"], 0.25)
        self.assertFalse(report["progress_assets"]["intent_translation_clean"])


if __name__ == "__main__":
    unittest.main()
