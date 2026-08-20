from __future__ import annotations

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
