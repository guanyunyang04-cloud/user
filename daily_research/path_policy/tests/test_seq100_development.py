from __future__ import annotations

import json
from pathlib import Path

import pytest

from daily_research.path_policy import seq100_development as development


def test_current_contract_is_small_and_marks_winner_null() -> None:
    contract = development.current_contract()

    assert contract["operational_commands"] == ["register", "run", "select", "freeze"]
    assert contract["winner"] is None
    assert contract["freeze_allowed"] is False
    assert contract["profile_roles"]["baseline"] == "control_only_not_champion"
    assert contract["profile_roles"]["hard_st"] == "rejected_candidate"
    assert contract["training"]["early_stopping_metric"] == "development_total_loss"
    assert contract["memory_guard_min_available_gib"] == 1.0
    assert contract["historical_fixed_oos_is_current"] is False


def test_current_parser_exposes_four_operational_commands_without_legacy_flow() -> None:
    parser = development._parser()
    help_text = parser.format_help()

    for command in development.OPERATIONAL_COMMANDS:
        assert command in help_text
    assert "advance-confirmation" not in help_text
    assert "begin-outer-audit" not in help_text
    assert "finalize-outer-audit" not in help_text


def test_register_dispatches_frozen_current_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict = {}

    def fake_register(**kwargs):
        captured.update(kwargs)
        return {"status": "registered"}

    monkeypatch.setattr(development.generation, "initialize_development_registry", fake_register)
    assert (
        development.main(
            [
                "register",
                "--root",
                "study",
                "--store-root",
                "folds",
                "--profiles",
                "baseline,hard_st",
            ]
        )
        == 0
    )

    assert captured["root"] == Path("study")
    assert captured["store_root"] == Path("folds")
    assert captured["source_view"] == development.CURRENT_SOURCE_VIEW
    assert captured["profiles"] == ("baseline", "hard_st")
    assert json.loads(capsys.readouterr().out)["status"] == "registered"


def test_current_facade_rejects_global_tail_profile() -> None:
    with pytest.raises(SystemExit):
        development._parser().parse_args(
            [
                "register",
                "--root",
                "study",
                "--store-root",
                "folds",
                "--profiles",
                "hard_st_global_tail",
            ]
        )


def test_register_has_no_implicit_profile() -> None:
    with pytest.raises(SystemExit):
        development._parser().parse_args(
            ["register", "--root", "study", "--store-root", "folds"]
        )
