from __future__ import annotations

import subprocess
import sys

from quantlab import cli as root_cli
from quantlab.research import __main__ as research_cli


def test_run_tree_comparison_trains_both_feature_views_before_comparing(monkeypatch) -> None:
    from quantlab.research import data, tree

    events: list[tuple[str, int | None]] = []

    monkeypatch.setattr(data, "verify_current_data", lambda: {"status": "ok"})
    monkeypatch.setattr(
        tree,
        "train_all",
        lambda features: events.append(("train", features)) or {"features": features},
    )
    monkeypatch.setattr(
        tree,
        "evaluate",
        lambda features: events.append(("evaluate", features)) or {"features": features},
    )
    monkeypatch.setattr(
        tree,
        "compare",
        lambda: events.append(("compare", None)) or {"status": "ok"},
    )
    monkeypatch.setattr(research_cli, "_print", lambda value: None)

    assert research_cli.main(["run-tree-comparison"]) == 0
    assert events == [
        ("train", 158),
        ("evaluate", 158),
        ("train", 183),
        ("evaluate", 183),
        ("compare", None),
    ]


def test_nested_minute_v2_options_are_forwarded_without_outer_parsing(monkeypatch) -> None:
    forwarded: list[list[str]] = []
    monkeypatch.setattr(
        root_cli,
        "research_main",
        lambda argv: forwarded.append(list(argv)) or 0,
    )
    assert root_cli.main(
        [
            "research",
            "minute-v2",
            "--workspace-root",
            "H:/quant_project",
            "verify-month",
            "--manifest",
            "manifest.json",
        ]
    ) == 0
    assert forwarded == [
        [
            "minute-v2",
            "--workspace-root",
            "H:/quant_project",
            "verify-month",
            "--manifest",
            "manifest.json",
        ]
    ]

    inner: list[list[str]] = []
    from quantlab.research.minute_v2 import cli as minute_v2_cli

    monkeypatch.setattr(minute_v2_cli, "main", lambda argv: inner.append(list(argv)) or 0)
    assert research_cli.main(
        ["minute-v2", "--workspace-root", "H:/quant_project", "verify-month"]
    ) == 0
    assert inner == [["--workspace-root", "H:/quant_project", "verify-month"]]


def test_data_cli_does_not_import_machine_learning_runtimes() -> None:
    script = """
import contextlib
import io
import sys

sys.path.insert(0, "src")

from quantlab import cli

assert "torch" not in sys.modules
assert "lightgbm" not in sys.modules
with contextlib.redirect_stdout(io.StringIO()):
    assert cli.main(["data", "--help"]) == 0
assert "torch" not in sys.modules
assert "lightgbm" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
