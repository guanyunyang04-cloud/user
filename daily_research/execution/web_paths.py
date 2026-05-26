from __future__ import annotations

from pathlib import Path


def ui_paths() -> dict[str, Path]:
    execution_root = Path(__file__).resolve().parent
    return {
        "ui_root": execution_root / "webapp",
        "react_dist": execution_root / "webapp" / "dist",
    }
