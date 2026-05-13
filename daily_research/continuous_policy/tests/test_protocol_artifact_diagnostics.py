from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from daily_research.continuous_policy.protocol_artifact_diagnostics import (
    inspect_json_artifact,
    summarize_protocol_artifacts,
)


class ProtocolArtifactDiagnosticsTest(unittest.TestCase):
    def test_inspect_json_artifact_reports_malformed_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "protocol_summary.json"
            path.write_text('{"run_tag": "bad", "items": [1, 2,}', encoding="utf-8")

            health = inspect_json_artifact(path)

        self.assertTrue(health["exists"])
        self.assertFalse(health["parse_ok"])
        self.assertGreater(health["size_bytes"], 0)
        self.assertIn("parse_error", health)
        self.assertIn("tail_preview", health)
        self.assertEqual(health["kind"], "json")

    def test_inspect_json_artifact_reports_valid_top_level_keys(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "protocol_summary.json"
            path.write_text(
                json.dumps({"run_tag": "ok", "promotion_gate": {"status": "shadow_only"}}),
                encoding="utf-8",
            )

            health = inspect_json_artifact(path)

        self.assertTrue(health["parse_ok"])
        self.assertEqual(health["top_level_keys"], ["promotion_gate", "run_tag"])
        self.assertEqual(health["run_tag"], "ok")

    def test_summarize_protocol_artifacts_marks_failed_exit_as_diagnostic_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protocol_root = root / "protocols" / "trial_03"
            protocol_root.mkdir(parents=True)
            (protocol_root / "protocol_summary.json").write_text(
                json.dumps(
                    {
                        "run_tag": "trial_03",
                        "training_evidence": {
                            "status": "insufficient",
                            "best_epoch": 8,
                            "completed_epochs": 8,
                        },
                        "promotion_gate": {
                            "status": "shadow_only",
                            "failed_checks": ["cash_timing_quality_1d"],
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_protocol_artifacts(protocol_root, exit_code=3221226505)

        self.assertEqual(summary["exit_code"], 3221226505)
        self.assertFalse(summary["completed_evidence"])
        self.assertEqual(summary["protocol_summary"]["run_tag"], "trial_03")
        self.assertEqual(summary["training_evidence_status"], "insufficient")
        self.assertIn("abnormal_exit", summary["diagnostic_flags"])


if __name__ == "__main__":
    unittest.main()
