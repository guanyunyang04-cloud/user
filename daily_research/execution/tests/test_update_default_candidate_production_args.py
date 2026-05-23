from __future__ import annotations


def test_update_default_candidate_production_accepts_start_date() -> None:
    from daily_research.execution.update_default_candidate_production import parse_args

    args = parse_args(["--start-date", "20250101", "--end-date", "20260522"])

    assert args.start_date == "20250101"
    assert args.end_date == "20260522"


def test_build_retrain_command_uses_explicit_start_date(monkeypatch) -> None:
    from daily_research.execution import update_default_candidate_production as module

    monkeypatch.setattr(
        module,
        "_load_source_config",
        lambda source_run_dir: (
            {"epochs": 8},
            {
                "start_date": "20210101",
                "benchmark": "000300.SH",
                "epochs": 8,
                "prediction_horizons": [1, 5, 20],
            },
        ),
    )
    monkeypatch.setattr(module, "_infer_family_key_for_source", lambda source_run_dir, cfg: "family")
    monkeypatch.setattr(module, "resolve_epoch_budget_for_family", lambda *args, **kwargs: 8)
    monkeypatch.setattr(module, "resolve_project_python_executable", lambda executable: executable)

    cmd = module._build_retrain_command(
        source_run_dir=module.Path("source"),
        family_epoch_budget_manifest="manifest.json",
        latest_completed_date="20260522",
        latest_trainable_date="20260520",
        internal_monitor_start_date="20260518",
        internal_monitor_days=3,
        experiment_tag="exp",
        train_start_date_override="20250101",
    )

    assert cmd[cmd.index("--start-date") + 1] == "20250101"
