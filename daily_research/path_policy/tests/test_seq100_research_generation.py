from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy.seq100_research_generation import (
    EVIDENCE_POLICY_SYNTHETIC_ALLOWED,
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


def _initialize_test_registry(**kwargs):
    kwargs.setdefault("screen_epochs", 2)
    return initialize_candidate_registry(
        **kwargs,
        require_corrected_source=False,
        require_existing_folds=False,
        evidence_policy=EVIDENCE_POLICY_SYNTHETIC_ALLOWED,
    )


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


def _write_valid_run_dir(tmp_path: Path, job: dict, *, name: str = "run") -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "final_model.pt"
    checkpoint_path.write_bytes(b"tiny-final-checkpoint")
    resolved = dict(job["expected_resolved_training_config"])
    max_samples = int(job["max_samples_per_split"])
    train_original = 1000
    train_selected = min(train_original, max_samples) if max_samples > 0 else train_original
    train_policy = "date_complete_even_spread_v1" if 0 < max_samples < train_original else "all_rows"
    summary = {
        "evaluation_mode": "fixed_oos",
        "evaluation_splits": ["oos"],
        "evaluation_split": "oos",
        "checkpoint_policy": "final_epoch",
        "run_tag": job["command"][job["command"].index("--run-tag") + 1],
        "seed": int(job["seed"]),
        "fold_year": int(job["oos_year"]),
        "pack_manifest": str(Path(job["view_path"]).resolve()),
        "path_value_gradient_profile": resolved["path_value_gradient_profile"],
        "rank_training_profile": resolved["rank_training_profile"],
        "epochs": int(job["epochs"]),
        "completed_epochs": int(job["epochs"]),
        "best_epoch": int(job["epochs"]),
        "max_samples_per_split": max_samples,
        "resolved_training_config": resolved,
        "sample_selection": {
            "train": {
                "policy": train_policy,
                "requested_max_samples": max_samples,
                "original_row_count": train_original,
                "selected_row_count": train_selected,
                "original_date_count": 20,
                "selected_date_count": 10 if train_policy != "all_rows" else 20,
                "date_complete": True,
            },
            "oos": {
                "policy": "all_rows",
                "requested_max_samples": 0,
                "original_row_count": 100,
                "selected_row_count": 100,
                "original_date_count": 2,
                "selected_date_count": 2,
                "date_complete": True,
            },
        },
        "output_dir": str(run_dir.resolve()),
        "best_checkpoint": str(checkpoint_path.resolve()),
    }
    (run_dir / "sequence_path_training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "split": "oos",
                "top_k": 3,
                "alpha_opportunity_value": 0.03,
                "alpha_realized_plan_value": 0.02,
                "selected_oracle_regret": 0.04,
                "selected_entry_fill_rate": 0.90,
            },
            {
                "split": "oos",
                "top_k": 10,
                "alpha_opportunity_value": 0.025,
                "alpha_realized_plan_value": 0.018,
                "selected_oracle_regret": 0.05,
                "selected_entry_fill_rate": 0.88,
            },
        ]
    ).to_csv(run_dir / "topk_metrics.csv", index=False)
    pd.DataFrame([{"split": "oos", "rank_ic_mean": 0.05, "path_mae": 0.08}]).to_csv(
        run_dir / "split_metrics.csv", index=False
    )
    pd.DataFrame([{"split": "oos", "trade_date": "2020-01-02", "top_k": 3}]).to_csv(
        run_dir / "daily_topk_metrics.csv", index=False
    )
    pd.DataFrame([{"split": "oos", "trade_date": "2020-01-02", "rank_ic": 0.05}]).to_csv(
        run_dir / "daily_rank_ic.csv", index=False
    )
    pd.DataFrame([{"epoch": int(job["epochs"])}]).to_csv(run_dir / "training_history.csv", index=False)
    pd.DataFrame([{"trade_date": "2020-01-02", "symbol": "000001.SZ", "score_rank": 1}]).to_parquet(
        run_dir / "topk_candidates.parquet", index=False
    )
    return run_dir


def test_candidate_registry_freezes_inner_matrix_and_training_commands(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    registry = _initialize_test_registry(
        root=root,
        source_view=source,
        store_root=store,
        screen_seeds=(7,),
        screen_epochs=2,
    )

    assert registry["inner_oos_years"] == list(INNER_OOS_YEARS)
    assert registry["historical_outer_oos_years"] == list(OUTER_OOS_YEARS)
    assert len(registry["jobs"]) == len(PROFILE_COMMANDS) * len(INNER_OOS_YEARS)
    assert {job["profile"] for job in registry["jobs"]} == set(PROFILE_COMMANDS)
    assert all(job["evaluation_mode"] == "fixed_oos" for job in registry["jobs"])
    assert all(job["checkpoint_policy"] == "final_epoch" for job in registry["jobs"])
    assert all(job["expected_resolved_training_config"] for job in registry["jobs"])
    assert len(registry["code_provenance"]["sha256"]) == 64
    assert all("--early-stopping-patience" in job["command"] for job in registry["jobs"])
    assert all("--prediction-mode" in job["command"] for job in registry["jobs"])
    assert registry["protected_boundaries"] == {
        "active_execution_changed": False,
        "qdp_active_changed": False,
        "legacy_artifacts_overwritten": False,
    }
    with pytest.raises(ValueError, match="study root"):
        register_result(
            root=tmp_path / "different_study",
            registry_path=Path(registry["registry_path"]),
            job_id=registry["jobs"][0]["job_id"],
            metrics=_metrics(0.02),
        )
    with pytest.raises(ValueError, match="canonical study"):
        select_profiles(
            registry_path=Path(registry["registry_path"]),
            ledger_path=tmp_path / "other_ledger.json",
        )

    with pytest.raises(FileExistsError, match="immutable artifact"):
        _initialize_test_registry(
            root=root,
            source_view=source,
            store_root=store,
            screen_seeds=(7,),
            screen_epochs=3,
        )


def test_screen_confirmation_freeze_and_one_time_outer_audit(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    candidate = _initialize_test_registry(root=root, source_view=source, store_root=store)
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
    assert confirmation["source_view"] == candidate["source_view"]
    assert confirmation["source_view_sha256"] == candidate["source_view_sha256"]
    assert confirmation["store_root"] == candidate["store_root"]
    assert confirmation["fold_bindings"] == candidate["fold_bindings"]
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
    assert frozen["source_view_sha256"] == candidate["source_view_sha256"]
    assert frozen["store_root"] == candidate["store_root"]
    assert frozen["fold_bindings"] == candidate["fold_bindings"]

    with pytest.raises(ValueError, match="study root"):
        begin_outer_audit(
            root=tmp_path / "second_outer_root",
            freeze_path=frozen["freeze_path"],
            source_view=source,
            store_root=store,
            require_corrected_source=False,
            require_existing_folds=False,
        )

    copied_source = tmp_path / "copied_source.json"
    copied_source.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="source view must match"):
        begin_outer_audit(
            root=root,
            freeze_path=frozen["freeze_path"],
            source_view=copied_source,
            store_root=store,
            require_corrected_source=False,
            require_existing_folds=False,
        )
    with pytest.raises(ValueError, match="store root must match"):
        begin_outer_audit(
            root=root,
            freeze_path=frozen["freeze_path"],
            source_view=source,
            store_root=tmp_path / "wrong_store",
            require_corrected_source=False,
            require_existing_folds=False,
        )

    outer = begin_outer_audit(
        root=root,
        freeze_path=frozen["freeze_path"],
        source_view=source,
        store_root=store,
        epochs=4,
        seed=7,
        require_corrected_source=False,
        require_existing_folds=False,
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
            require_corrected_source=False,
            require_existing_folds=False,
        )

    outer_path = Path(outer["registry_path"])
    outer_registry = json.loads(outer_path.read_text(encoding="utf-8"))
    for job in outer_registry["jobs"]:
        binding = outer_registry["fold_bindings"][str(job["oos_year"])]
        assert job["view_sha256"] == binding["view_sha256"]
        assert job["fold_training_contract_sha256"] == binding["fold_training_contract_sha256"]
        assert job["source_view_sha256"] == binding["source_view_sha256"]
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
    assert len(result["outer_result_sha256s"]) == 4
    assert len(result["ledger_sha256"]) == 64
    assert result["active_execution_changed"] is False
    assert result["qdp_active_changed"] is False


def test_guardrail_failure_keeps_baseline_even_with_higher_primary(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    candidate = _initialize_test_registry(root=root, source_view=source, store_root=store)
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
    registry = _initialize_test_registry(root=root, source_view=source, store_root=store)
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
    registry = _initialize_test_registry(root=root, source_view=source, store_root=store)
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
    candidate = _initialize_test_registry(root=root, source_view=source, store_root=store)
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
        confirm_epochs=2,
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
            require_corrected_source=False,
            require_existing_folds=False,
        )


@pytest.mark.parametrize(
    "tamper",
    ("completed_epochs", "top_level_max", "resolved_model", "train_limit", "oos_policy"),
)
def test_run_artifact_registration_rejects_job_contract_mismatch(tmp_path: Path, tamper: str) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    registry = _initialize_test_registry(root=root, source_view=source, store_root=store)
    registry_path = Path(registry["registry_path"])
    job = registry["jobs"][0]
    run_dir = _write_valid_run_dir(tmp_path, job, name=f"run_{tamper}")
    summary_path = run_dir / "sequence_path_training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if tamper == "completed_epochs":
        summary["completed_epochs"] -= 1
    elif tamper == "top_level_max":
        summary["max_samples_per_split"] += 1
    elif tamper == "resolved_model":
        summary["resolved_training_config"]["model_type"] = "wrong_model"
    elif tamper == "train_limit":
        summary["sample_selection"]["train"]["requested_max_samples"] += 1
    else:
        summary["sample_selection"]["oos"]["policy"] = "date_complete_even_spread_v1"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError):
        register_result(
            root=root,
            registry_path=registry_path,
            job_id=job["job_id"],
            run_dir=run_dir,
        )


def test_registered_run_rejects_later_metric_file_change(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    registry = _initialize_test_registry(root=root, source_view=source, store_root=store)
    registry_path = Path(registry["registry_path"])
    first = registry["jobs"][0]
    run_dir = _write_valid_run_dir(tmp_path, first, name="bound_run")
    register_result(root=root, registry_path=registry_path, job_id=first["job_id"], run_dir=run_dir)
    for job in registry["jobs"][1:]:
        register_result(
            root=root,
            registry_path=registry_path,
            job_id=job["job_id"],
            metrics=_metrics(0.02),
        )
    topk_path = run_dir / "topk_metrics.csv"
    topk_path.write_text(topk_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="result evidence file changed"):
        select_profiles(registry_path=registry_path, ledger_path=root / "result_ledger.json")


def test_registry_and_result_digest_tampering_is_rejected(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    registry = _initialize_test_registry(root=root, source_view=source, store_root=store)
    registry_path = Path(registry["registry_path"])
    tampered_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    tampered_registry["status"] = "tampered"
    registry_path.write_text(json.dumps(tampered_registry), encoding="utf-8")
    with pytest.raises(ValueError, match="registry digest mismatch"):
        register_result(
            root=root,
            registry_path=registry_path,
            job_id=registry["jobs"][0]["job_id"],
            metrics=_metrics(0.02),
        )

    second_root = tmp_path / "second_study"
    second = _initialize_test_registry(root=second_root, source_view=source, store_root=store)
    second_path = Path(second["registry_path"])
    _register_matrix(
        second_root,
        second_path,
        {profile: _metrics(0.02) for profile in PROFILE_COMMANDS},
    )
    ledger_path = second_root / "result_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    first_record = next(iter(ledger["results"].values()))
    first_record["metrics"]["path_mae"] = 999.0
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(ValueError, match="result digest mismatch"):
        select_profiles(registry_path=second_path, ledger_path=ledger_path)


def test_candidate_source_change_blocks_confirmation(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    candidate = _initialize_test_registry(root=root, source_view=source, store_root=store)
    candidate_path = Path(candidate["registry_path"])
    _register_matrix(
        root,
        candidate_path,
        {profile: _metrics(0.02) for profile in PROFILE_COMMANDS},
    )
    source.write_text(json.dumps({"artifact_type": "changed_source"}), encoding="utf-8")

    with pytest.raises(ValueError, match="source view changed"):
        create_confirmation_registry(
            root=root,
            candidate_registry_path=candidate_path,
            ledger_path=root / "result_ledger.json",
            confirm_seeds=(7,),
            confirm_epochs=2,
        )


def test_production_policy_cannot_disable_source_or_fold_verification(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    with pytest.raises(ValueError, match="run-artifact registries require"):
        initialize_candidate_registry(
            root=tmp_path / "study",
            source_view=source,
            store_root=store,
            screen_epochs=2,
            require_corrected_source=False,
            require_existing_folds=False,
        )
    with pytest.raises(ValueError, match="screen_max_samples_per_split"):
        _initialize_test_registry(
            root=tmp_path / "negative_cap",
            source_view=source,
            store_root=store,
            screen_max_samples_per_split=-1,
        )
    with pytest.raises(ValueError, match="screen_epochs"):
        _initialize_test_registry(
            root=tmp_path / "one_epoch",
            source_view=source,
            store_root=store,
            screen_epochs=1,
        )


def test_result_ledger_update_uses_exclusive_lock(tmp_path: Path) -> None:
    source, store = _source_view(tmp_path)
    root = tmp_path / "study"
    registry = _initialize_test_registry(root=root, source_view=source, store_root=store)
    lock_path = root / "result_ledger.lock"
    lock_path.write_text("held", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already being updated"):
        register_result(
            root=root,
            registry_path=Path(registry["registry_path"]),
            job_id=registry["jobs"][0]["job_id"],
            metrics=_metrics(0.02),
        )
