from __future__ import annotations

import json
from pathlib import Path

import pytest

from daily_research.path_policy.seq100_research_generation import (
    INNER_OOS_YEARS,
    METRIC_NAMES,
    OUTER_OOS_YEARS,
    PROFILE_COMMANDS,
    begin_outer_audit,
    create_confirmation_registry,
    finalize_outer_audit,
    freeze_champion,
    initialize_candidate_registry,
    register_result,
    select_profiles,
)


def _source_view(tmp_path: Path) -> tuple[Path, Path]:
    store = tmp_path / "store"
    source = store / "views" / "source.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"artifact_type": "tiny_seq100_source"}), encoding="utf-8")
    return source, store


def _metrics(
    opportunity: float,
    *,
    realized: float | None = None,
    regret: float = 0.04,
    top10: float | None = None,
    rank_ic: float = 0.05,
    fill_rate: float = 0.90,
    path_mae: float = 0.08,
) -> dict[str, float]:
    payload = {
        "top3_opportunity_alpha": opportunity,
        "top3_realized_plan_alpha": opportunity - 0.01 if realized is None else realized,
        "oracle_regret": regret,
        "top10_opportunity_alpha": opportunity - 0.005 if top10 is None else top10,
        "daily_rank_ic": rank_ic,
        "fill_rate": fill_rate,
        "path_mae": path_mae,
    }
    assert set(payload) == set(METRIC_NAMES)
    return payload


def _register_matrix(
    root: Path,
    registry_path: Path,
    profile_metrics: dict[str, dict[str, float]],
) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for job in registry["jobs"]:
        base = dict(profile_metrics[job["profile"]])
        # Preserve a deterministic fold ordering so worst-fold behavior is tested.
        base["top3_opportunity_alpha"] += (int(job["oos_year"]) - min(INNER_OOS_YEARS)) * 0.001
        register_result(
            root=root,
            registry_path=registry_path,
            job_id=job["job_id"],
            metrics=base,
        )


def test_candidate_registry_freezes_inner_matrix_and_training_commands(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    registry = initialize_candidate_registry(
        root=root,
        source_view=source,
        store_root=store,
        screen_seeds=(7,),
        screen_epochs=1,
    )

    assert registry["inner_oos_years"] == list(INNER_OOS_YEARS)
    assert registry["historical_outer_oos_years"] == list(OUTER_OOS_YEARS)
    assert len(registry["jobs"]) == len(PROFILE_COMMANDS) * len(INNER_OOS_YEARS)
    assert {job["profile"] for job in registry["jobs"]} == set(PROFILE_COMMANDS)
    assert all(job["evaluation_mode"] == "fixed_oos" for job in registry["jobs"])
    assert all(job["checkpoint_policy"] == "final_epoch" for job in registry["jobs"])
    assert all("--early-stopping-patience" in job["command"] for job in registry["jobs"])
    assert all("--prediction-mode" in job["command"] for job in registry["jobs"])
    assert registry["protected_boundaries"] == {
        "active_execution_changed": False,
        "qdp_active_changed": False,
        "legacy_artifacts_overwritten": False,
    }

    with pytest.raises(FileExistsError, match="immutable artifact"):
        initialize_candidate_registry(
            root=root,
            source_view=source,
            store_root=store,
            screen_seeds=(7,),
            screen_epochs=2,
        )


def test_screen_confirmation_freeze_and_one_time_outer_audit(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    candidate = initialize_candidate_registry(root=root, source_view=source, store_root=store)
    candidate_path = Path(candidate["registry_path"])
    ledger_path = root / "result_ledger.json"
    _register_matrix(
        root,
        candidate_path,
        {
            "baseline": _metrics(0.020),
            "hard_st": _metrics(0.026),
            "hard_st_global_tail": _metrics(0.034),
        },
    )

    screen = select_profiles(registry_path=candidate_path, ledger_path=ledger_path)
    assert screen["winner"] == "hard_st_global_tail"
    confirmation = create_confirmation_registry(
        root=root,
        candidate_registry_path=candidate_path,
        ledger_path=ledger_path,
        confirm_seeds=(7, 17, 29),
        confirm_epochs=3,
        shortlist_size=2,
    )
    assert confirmation["shortlist"] == ["baseline", "hard_st_global_tail"]
    assert len(confirmation["jobs"]) == 2 * len(INNER_OOS_YEARS) * 3
    confirmation_path = Path(confirmation["registry_path"])
    _register_matrix(
        root,
        confirmation_path,
        {
            "baseline": _metrics(0.021),
            "hard_st_global_tail": _metrics(0.036),
        },
    )

    with pytest.raises(ValueError, match="not the policy winner"):
        freeze_champion(
            root=root,
            confirmation_registry_path=confirmation_path,
            ledger_path=ledger_path,
            champion="baseline",
        )
    frozen = freeze_champion(
        root=root,
        confirmation_registry_path=confirmation_path,
        ledger_path=ledger_path,
        champion="hard_st_global_tail",
    )
    assert frozen["outer_evidence_grade"] == "selection_aware_historical_oos"
    assert frozen["active_execution_changed"] is False

    outer = begin_outer_audit(
        root=root,
        freeze_path=frozen["freeze_path"],
        source_view=source,
        store_root=store,
        epochs=4,
        seed=7,
    )
    assert outer["historical_outer_oos_years"] == list(OUTER_OOS_YEARS)
    assert outer["evidence_grade"] == "selection_aware_historical_oos"
    assert outer["selection_aware"] is True
    assert outer["repeat_allowed"] is False
    with pytest.raises(RuntimeError, match="already been consumed or claimed"):
        begin_outer_audit(
            root=root,
            freeze_path=frozen["freeze_path"],
            source_view=source,
            store_root=store,
        )

    outer_path = Path(outer["registry_path"])
    outer_registry = json.loads(outer_path.read_text(encoding="utf-8"))
    for job in outer_registry["jobs"]:
        register_result(
            root=root,
            registry_path=outer_path,
            job_id=job["job_id"],
            metrics=_metrics(0.038),
        )
    result = finalize_outer_audit(root=root, ledger_path=ledger_path)
    assert result["status"] == "completed"
    assert result["evidence_grade"] == "selection_aware_historical_oos"
    assert result["aggregate"]["fold_count"] == 4
    assert result["active_execution_changed"] is False
    assert result["qdp_active_changed"] is False


def test_guardrail_failure_keeps_baseline_even_with_higher_primary(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    candidate = initialize_candidate_registry(root=root, source_view=source, store_root=store)
    path = Path(candidate["registry_path"])
    _register_matrix(
        root,
        path,
        {
            "baseline": _metrics(0.020, fill_rate=0.90),
            "hard_st": _metrics(0.030, fill_rate=0.40),
            "hard_st_global_tail": _metrics(0.025, regret=0.20),
        },
    )
    selection = select_profiles(registry_path=path, ledger_path=root / "result_ledger.json")
    assert selection["winner"] == "baseline"
    verdicts = {item["profile"]: item["guardrails"] for item in selection["ranking"]}
    assert verdicts["hard_st"]["checks"]["fill_rate"] is False
    assert verdicts["hard_st_global_tail"]["checks"]["oracle_regret"] is False


def test_incomplete_or_unknown_result_is_rejected(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    registry = initialize_candidate_registry(root=root, source_view=source, store_root=store)
    registry_path = Path(registry["registry_path"])
    with pytest.raises(KeyError, match="not registered"):
        register_result(
            root=root,
            registry_path=registry_path,
            job_id="screen:unknown:oos2018:seed7",
            metrics=_metrics(0.1),
        )
    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        select_profiles(registry_path=registry_path, ledger_path=root / "result_ledger.json")


def test_real_run_import_validates_registered_provenance(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    registry = initialize_candidate_registry(root=root, source_view=source, store_root=store)
    registry_path = Path(registry["registry_path"])
    job = registry["jobs"][0]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "evaluation_mode": "fixed_oos",
        "checkpoint_policy": "final_epoch",
        "run_tag": job["command"][job["command"].index("--run-tag") + 1],
        "seed": job["seed"] + 1,
        "fold_year": job["oos_year"],
        "pack_manifest": job["view_path"],
        "path_value_gradient_profile": "smooth_current",
        "rank_training_profile": "local_chunk",
    }
    (run_dir / "sequence_path_training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="registered job for seed"):
        register_result(
            root=root,
            registry_path=registry_path,
            job_id=job["job_id"],
            run_dir=run_dir,
        )


def test_outer_audit_rejects_tampered_freeze(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    candidate = initialize_candidate_registry(root=root, source_view=source, store_root=store)
    candidate_path = Path(candidate["registry_path"])
    ledger_path = root / "result_ledger.json"
    _register_matrix(
        root,
        candidate_path,
        {
            "baseline": _metrics(0.02),
            "hard_st": _metrics(0.03),
            "hard_st_global_tail": _metrics(0.04),
        },
    )
    confirmation = create_confirmation_registry(
        root=root,
        candidate_registry_path=candidate_path,
        ledger_path=ledger_path,
        confirm_seeds=(7,),
        confirm_epochs=1,
        shortlist_size=2,
    )
    confirmation_path = Path(confirmation["registry_path"])
    _register_matrix(
        root,
        confirmation_path,
        {"baseline": _metrics(0.02), "hard_st_global_tail": _metrics(0.04)},
    )
    frozen = freeze_champion(
        root=root,
        confirmation_registry_path=confirmation_path,
        ledger_path=ledger_path,
        champion="hard_st_global_tail",
    )
    freeze_path = Path(frozen["freeze_path"])
    tampered = json.loads(freeze_path.read_text(encoding="utf-8"))
    tampered["champion"] = "baseline"
    freeze_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        begin_outer_audit(
            root=root,
            freeze_path=freeze_path,
            source_view=source,
            store_root=store,
        )
