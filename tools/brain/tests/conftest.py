from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SKILL_CACHE = ROOT / "brain" / "skills" / "workspace-brain" / "scripts" / "__pycache__"


sys.dont_write_bytecode = True


def pytest_sessionfinish(session, exitstatus) -> None:  # type: ignore[no-untyped-def]
    if SKILL_CACHE.exists():
        shutil.rmtree(SKILL_CACHE)
