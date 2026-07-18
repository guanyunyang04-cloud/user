from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import qdp_v2_sequence_path_training as training
from daily_research.path_policy import seq100_development as development
from daily_research.path_policy.qdp_v2_sequence_path_pack import _bind_research_contract


def _dry_run_payload() -> dict[str, Any]:
    return {
        "status": "dry_run",
        "writes_performed": False,
        "qdp_active_as_of": development.EXPECTED_QDP_AS_OF,
        "signal_window": [development.SIGNAL_START, development.SIGNAL_END],
        "first_valid_lookback_signal": "2010-06-01",
        "dependency_padding_end": "2026-05-08",
        "symbol_count": 3041,
        "safe_train_signal_end": {
            "2023": "2022-09-01",
            "2024": "2023-08-31",
            "2025": "2024-08-30",
        },
    }


def test_current_contract_exposes_only_the_single_current_workflow() -> None:
    contract = development.current_contract()
    semantics = contract["contract"]

    assert contract["operational_commands"] == [
        "contract",
        "prepare",
        "run",
        "status",
        "summarize",
    ]
    assert contract["source_pack_required_for_contract"] is False
    assert semantics["profile"]["command"] == development.PROFILE_COMMAND
    assert semantics["profile"]["input_dim"] == 32
    assert semantics["development_protocol"]["years"] == [2023, 2024, 2025]
    assert semantics["development_protocol"]["seed"] == 7
    assert semantics["development_protocol"]["final_fit_in_scope"] is False
    assert semantics["early_stopping"]["metric"] == "development_total_loss"
    assert semantics["metric_contract"]["year_weighting"] == "equal"
    assert contract["memory_guard"] == {
        "minimum_available_gib": 0.5,
        "interval_seconds": 1.0,
        "consecutive_breaches": 2,
    }


def test_current_parser_has_no_registry_selection_or_freeze_commands() -> None:
    help_text = development._parser().format_help()

    for command in development.OPERATIONAL_COMMANDS:
        assert command in help_text
    for legacy in ("register", "select", "freeze", "promotion"):
        assert legacy not in help_text


def test_mutable_study_runtime_does_not_change_semantic_contract_binding(
    tmp_path: Path,
) -> None:
    study_path = tmp_path / "study.json"
    payload = development._new_study_payload(_dry_run_payload())
    development._write_json(study_path, payload)

    first_pack = _bind_research_contract(study_path)
    first_training = training._validated_development_contract(study_path)
    payload["runtime"].update({"status": "running", "current_year": 2024})
    development._write_json(study_path, payload)
    second_pack = _bind_research_contract(study_path)
    second_training = training._validated_development_contract(study_path)

    assert first_pack == second_pack
    assert first_training == second_training
    assert first_pack["contract_sha256"] == payload["contract_sha256"]
    assert first_pack["contract_file_sha256"] == payload["contract_sha256"]


def test_relocate_pack_manifest_updates_nested_staging_paths(tmp_path: Path) -> None:
    staging = tmp_path / ".seq100_current.staging"
    final = tmp_path / "seq100_current"
    pack = final / "pack"
    pack.mkdir(parents=True)
    old_file = staging / "pack" / "labels" / "future.float32.dat"
    manifest_path = pack / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "feature_channels": {"daily_raw": {"path": str(old_file)}},
                "label_arrays": {
                    "future_ohlcva_path": {
                        "shards": [{"path": str(staging / "pack" / "labels" / "s0.dat")}]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (pack / "progress.json").write_text(
        json.dumps({"manifest_json": str(staging / "pack" / "manifest.json")}),
        encoding="utf-8",
    )

    relocated = development._relocate_pack_manifest(manifest_path, staging, final)

    serialized = json.dumps(relocated)
    assert str(staging.resolve()) not in serialized
    assert Path(relocated["feature_channels"]["daily_raw"]["path"]) == (
        final / "pack" / "labels" / "future.float32.dat"
    )
    progress = json.loads((pack / "progress.json").read_text(encoding="utf-8"))
    assert progress["manifest_json"] == str((final / "pack" / "manifest.json").resolve())


def test_prepare_is_idempotent_when_the_single_pack_is_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root = tmp_path / "study"
    store_root = tmp_path / "research_store" / "seq100_current"
    manifest = store_root / "pack" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    fold_views: dict[str, str] = {}
    for year in development.DEVELOPMENT_YEARS:
        view = store_root / "folds" / "views" / f"development_{year}.json"
        view.parent.mkdir(parents=True, exist_ok=True)
        view.write_text("{}", encoding="utf-8")
        fold_views[str(year)] = str(view)
    payload = development._new_study_payload(_dry_run_payload())
    payload["runtime"]["status"] = "prepared"
    payload["material"] = {
        "pack_manifest": str(manifest),
        "fold_views": fold_views,
    }
    development._write_json(study_root / "study.json", payload)

    monkeypatch.setattr(development, "_assert_no_other_research_process", lambda: None)
    monkeypatch.setattr(development, "prepare_dry_run", lambda **_: _dry_run_payload())
    monkeypatch.setattr(
        development.pack_builder,
        "validate_sequence_pack",
        lambda _: {"status": "ok"},
    )
    monkeypatch.setattr(development, "_pack_preflight", lambda **_: {})
    monkeypatch.setattr(
        development,
        "_guarded_command",
        lambda *_, **__: pytest.fail("valid prepare rerun must not rebuild the pack"),
    )

    result = development.prepare_current_study(
        qdp_root=tmp_path / "qdp",
        study_root=study_root,
        store_root=store_root,
        cleanup=False,
    )

    assert result["runtime"]["status"] == "prepared"
    assert result["material"]["pack_manifest"] == str(manifest)


def test_three_year_summary_is_strictly_equal_year_weighted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_root = tmp_path / "study"
    pack_manifest = tmp_path / "pack" / "manifest.json"
    pack_manifest.parent.mkdir(parents=True)
    pack_manifest.write_text("{}", encoding="utf-8")
    payload = development._new_study_payload(_dry_run_payload())
    payload["runtime"]["status"] = "runs_completed"
    payload["material"] = {"pack_manifest": str(pack_manifest)}
    development._write_json(study_root / "study.json", payload)

    for year in development.DEVELOPMENT_YEARS:
        run = (
            study_root
            / "runs"
            / "development"
            / f"seq100_development_baseline_{year}_seed7_fixture"
        )
        run.mkdir(parents=True)
        (run / "sequence_path_training_summary.json").write_text("{}", encoding="utf-8")

    yearly_alpha = {2023: 0.03, 2024: -0.02, 2025: 0.06}

    def fake_year_metrics(_run: Path, year: int):
        return (
            {
                "development_year": year,
                "best_epoch": {2023: 2, 2024: 5, 2025: 3}[year],
                "completed_epochs": 5,
                "top1_base_alpha": yearly_alpha[year] + 0.01,
                "top3_base_alpha": yearly_alpha[year],
                "top5_base_alpha": yearly_alpha[year] - 0.01,
                "top10_base_alpha": yearly_alpha[year] - 0.02,
                "top3_stress_alpha": yearly_alpha[year] - 0.005,
                "rank_ic": 0.01 * (year - 2022),
            },
            {"optimizer_steps": 10},
        )

    monkeypatch.setattr(development, "_year_metrics", fake_year_metrics)
    monkeypatch.setattr(development, "CandidateCompleteAuditPack", lambda _: object())
    monkeypatch.setattr(
        development,
        "_candidate_bridge_rows",
        lambda **_: pd.DataFrame([{"year": 2023, "rank": 1, "raw": 0.1}]),
    )
    monkeypatch.setattr(
        development,
        "_basis_summaries",
        lambda rows: (rows.copy(), rows.copy()),
    )

    result = development.summarize_current_study(study_root=study_root)

    assert result["equal_year_metrics"]["top3_base_alpha"] == pytest.approx(
        (0.03 - 0.02 + 0.06) / 3.0
    )
    assert result["equal_year_metrics"]["positive_top3_years"] == 2
    assert result["equal_year_metrics"]["worst_year_top3_base_alpha"] == -0.02
    assert result["suggested_final_fit_epochs"] == 3
    assert result["winner"] is None
    assert result["final_fit_performed"] is False


def test_optional_opportunity_diagnostic_reports_finite_coverage() -> None:
    frame = pd.DataFrame({"alpha_opportunity_value": [0.12, np.nan, 0.06]})

    mean, coverage = development._diagnostic_mean_and_coverage(
        frame, "alpha_opportunity_value"
    )

    assert mean == pytest.approx(0.09)
    assert coverage == pytest.approx(2.0 / 3.0)
    with pytest.raises(ValueError, match="non-finite"):
        development._finite_mean(frame, "alpha_opportunity_value")


def test_unfilled_cash_day_does_not_fabricate_an_exit_day() -> None:
    frame = pd.DataFrame({"selected_realized_plan_exit_day": [2.0, np.nan, 4.0]})

    mean, coverage = development._diagnostic_mean_and_coverage(
        frame, "selected_realized_plan_exit_day"
    )

    assert mean == pytest.approx(3.0)
    assert coverage == pytest.approx(2.0 / 3.0)
