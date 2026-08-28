from __future__ import annotations

from quantlab import cli as root_cli
from quantlab.research import __main__ as research_cli


def test_run_tree_comparison_trains_both_feature_views_before_comparing(monkeypatch) -> None:
    events: list[tuple[str, int | None]] = []

    monkeypatch.setattr(research_cli, "verify_current_data", lambda: {"status": "ok"})
    monkeypatch.setattr(
        research_cli,
        "train_all",
        lambda features: events.append(("train", features)) or {"features": features},
    )
    monkeypatch.setattr(
        research_cli,
        "evaluate",
        lambda features: events.append(("evaluate", features)) or {"features": features},
    )
    monkeypatch.setattr(
        research_cli,
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
    monkeypatch.setattr(research_cli, "minute_v2_main", lambda argv: inner.append(list(argv)) or 0)
    assert research_cli.main(
        ["minute-v2", "--workspace-root", "H:/quant_project", "verify-month"]
    ) == 0
    assert inner == [["--workspace-root", "H:/quant_project", "verify-month"]]
