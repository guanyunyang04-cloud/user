import json
import errno
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from daily_research.continuous_policy.runtime import _replace_json_with_retry
from daily_research.continuous_policy.runtime_progress import (
    JsonlProgressSink,
    summarize_progress_log,
)


class RuntimeProgressTest(unittest.TestCase):
    def test_json_replace_retries_transient_file_lock(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "progress.json.tmp"
            final_path = Path(temp_dir) / "progress.json"
            temp_path.write_text('{"status":"ok"}', encoding="utf-8")
            attempts = {"count": 0}

            class FlakyTempPath:
                def replace(self, target: Path) -> None:
                    attempts["count"] += 1
                    if attempts["count"] == 1:
                        raise PermissionError(errno.EACCES, "temporarily locked")
                    temp_path.replace(target)

            _replace_json_with_retry(FlakyTempPath(), final_path)  # type: ignore[arg-type]

            self.assertEqual(attempts["count"], 2)
            self.assertEqual(json.loads(final_path.read_text(encoding="utf-8"))["status"], "ok")

    def test_jsonl_sink_updates_latest_json_and_jsonl(self) -> None:
        with TemporaryDirectory() as temp_dir:
            progress_path = Path(temp_dir) / "protocol_progress.jsonl"
            sink = JsonlProgressSink(progress_path, run_tag="runtime_test", stage="train")

            event = sink.emit(
                "train_epoch_complete",
                epoch=2,
                completed_epochs=2,
                amp_enabled=True,
                device="cuda",
                train_seconds=1.25,
                validation_seconds=0.75,
            )

            latest = json.loads(progress_path.with_name("protocol_progress.json").read_text(encoding="utf-8"))
            lines = progress_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(event["run_tag"], "runtime_test")
        self.assertEqual(event["stage"], "train")
        self.assertEqual(event["event"], "train_epoch_complete")
        self.assertEqual(event["epoch"], 2)
        self.assertEqual(event["completed_epochs"], 2)
        self.assertIn("elapsed_seconds", event)
        self.assertIn("pid", event)
        self.assertIn("cuda_memory_allocated_mb", event)
        self.assertEqual(latest["event"], "train_epoch_complete")
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["completed_epochs"], 2)

    def test_summarize_progress_log_returns_last_event_and_staleness(self) -> None:
        with TemporaryDirectory() as temp_dir:
            progress_path = Path(temp_dir) / "protocol_progress.jsonl"
            sink = JsonlProgressSink(progress_path, run_tag="runtime_test", stage="train")
            sink.emit("train_epoch_start", epoch=1)
            time.sleep(0.01)
            sink.emit("train_epoch_complete", epoch=1, completed_epochs=1)

            summary = summarize_progress_log(progress_path)

        self.assertTrue(summary["exists"])
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["last_event"]["event"], "train_epoch_complete")
        self.assertEqual(summary["last_epoch"], 1)
        self.assertEqual(summary["completed_epochs"], 1)
        self.assertGreaterEqual(summary["stale_seconds"], 0.0)

    def test_noop_sink_is_safe_for_legacy_callers(self) -> None:
        sink = JsonlProgressSink(None, run_tag="runtime_test", stage="train")

        event = sink.emit("ignored", epoch=1)

        self.assertEqual(event, {})
        self.assertEqual(sink.latest(), {})


if __name__ == "__main__":
    unittest.main()
