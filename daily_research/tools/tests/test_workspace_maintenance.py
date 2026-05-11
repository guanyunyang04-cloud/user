from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from daily_research.tools import workspace_maintenance


class WorkspaceMaintenanceTest(unittest.TestCase):
    def test_archive_prune_plan_defaults_to_cache_items_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_payload = root / "daily_research" / "archive" / "cache" / "deep_alpha" / "batch" / "old.pkl"
            output_payload = root / "daily_research" / "archive" / "output" / "runs" / "batch" / "study"
            cache_payload.parent.mkdir(parents=True)
            output_payload.mkdir(parents=True)
            cache_payload.write_bytes(b"x" * 10)
            (output_payload / "summary.json").write_text("{}", encoding="utf-8")
            manifest = root / "daily_research" / "archive" / "manifests" / "archive_batch.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "batch_id": "batch",
                        "items": [
                            {
                                "rule": "deep_alpha_corpus_cache",
                                "source_path": "daily_research/cache/deep_alpha/corpus/old.pkl",
                                "archived_path": "daily_research/archive/cache/deep_alpha/batch/old.pkl",
                                "size_bytes": 10,
                                "is_dir": False,
                                "latest_mtime": "2026-05-11T00:00:00",
                            },
                            {
                                "rule": "output_runs",
                                "source_path": "daily_research/output/study",
                                "archived_path": "daily_research/archive/output/runs/batch/study",
                                "size_bytes": 2,
                                "is_dir": True,
                                "latest_mtime": "2026-05-11T00:00:00",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            plan = workspace_maintenance.build_archive_prune_plan(root, manifest, selected_rules=())

        self.assertEqual(plan["candidate_count"], 1)
        self.assertEqual(plan["candidate_size_bytes"], 10)
        self.assertEqual(
            plan["candidates"][0]["archived_path"],
            "daily_research/archive/cache/deep_alpha/batch/old.pkl",
        )
