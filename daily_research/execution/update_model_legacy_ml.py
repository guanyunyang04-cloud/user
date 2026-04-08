from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.update_model import main as update_model_main


def main():
    print(
        "warning=update_model_legacy_ml.py is deprecated; prefer update_model.py and use this wrapper only for intentional legacy maintenance.",
        file=sys.stderr,
    )
    update_model_main()


if __name__ == "__main__":
    main()
