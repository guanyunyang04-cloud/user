from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.update_model import main as update_model_main


def main():
    update_model_main()


if __name__ == "__main__":
    main()
