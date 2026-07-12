from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.brain import brain_sync_audit


class BrainSyncAuditTest(unittest.TestCase):
    def test_removed_brain_command_is_rejected_in_child_hot_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for relative_path in brain_sync_audit.HOT_PATHS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("current\n", encoding="utf-8")
            target = root / "daily_research/brain/governance_layer.md"
            target.write_text("brain-burden-audit\n", encoding="utf-8")

            payload = brain_sync_audit.audit(workspace_root=root)

        self.assertEqual(payload["status"], "needs_sync")
        self.assertTrue(any(item["pattern"] == "brain-burden-audit" for item in payload["findings"]))

    def test_daily_research_current_semantics_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for relative_path in brain_sync_audit.HOT_PATHS:
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                required = brain_sync_audit.REQUIRED_BY_PATH.get(relative_path, {})
                path.write_text("\n".join(required) or "current\n", encoding="utf-8")
            target = root / "daily_research/brain/state_center.md"
            target.write_text(
                target.read_text(encoding="utf-8").replace("winner=null", "winner unresolved"),
                encoding="utf-8",
            )

            payload = brain_sync_audit.audit(workspace_root=root)

        self.assertEqual(payload["status"], "needs_sync")
        self.assertTrue(any(item["pattern"] == "missing:winner=null" for item in payload["findings"]))


if __name__ == "__main__":
    unittest.main()
