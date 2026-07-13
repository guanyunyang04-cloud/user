from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from daily_research.path_policy.seq100_qcurve_data import (
    DEFAULT_FOLD_ROOT,
    DEFAULT_PACK_MANIFEST,
    DEVELOPMENT_YEARS,
    verify_qcurve_development_fold,
)
from daily_research.path_policy.seq100_qcurve_pack import validate_qcurve_pack
from daily_research.path_policy.seq100_qcurve_qonly_training import (
    QONLY_IMPLEMENTATION_VERSION,
    QONLY_LOSS_WEIGHT_MAP,
    QONLY_OBJECTIVE_VERSION,
    QONLY_PROFILES,
    _objective_digest,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
DEFAULT_STUDY_ROOT = Path(
    "daily_research/output/path_policy/studies/seq100_dynamic_qcurve_qonly_2022_2025"
)
OLD_STUDY_REGISTRY = Path(
    "daily_research/output/path_policy/studies/seq100_dynamic_qcurve_2022_2025/registry.json"
)
DEFAULT_CONTRACT_PATH = Path(
    "daily_research/brain/references/seq100_dynamic_qcurve_qonly_contract_20260713.json"
)
PROFILES = ("qcurve_lgbm", *QONLY_PROFILES)
SEED = 7
MAX_EPOCHS = 10
PATIENCE = 2


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _workspace_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(_workspace_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = _workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _preflight_path(root: Path, profile: str) -> Path:
    return root / "preflight" / f"{profile}.json"


def current_contract(*, root: str | Path = DEFAULT_STUDY_ROOT) -> dict[str, Any]:
    root_path = _workspace_path(root)
    precision: dict[str, Any] = {}
    for profile in QONLY_PROFILES:
        path = _preflight_path(root_path, profile)
        if path.is_file():
            data = _read_json(path)
            precision[profile] = {
                "precision": data["precision"],
                "microbatch_size": int(data["microbatch_size"]),
                "fast_cudnn": True,
                "bitwise_deterministic": False,
                "preflight_path": str(path.resolve()),
            }
    return {
        "schema_version": 1,
        "workflow_id": "seq100_dynamic_qcurve_qonly_development",
        "profiles": list(PROFILES),
        "development_years": list(DEVELOPMENT_YEARS),
        "seed": SEED,
        "objective": {
            "version": QONLY_OBJECTIVE_VERSION,
            "loss_weights": QONLY_LOSS_WEIGHT_MAP,
            "auxiliary_heads": False,
            "full_day_rank": True,
        },
        "training": {
            "implementation_version": QONLY_IMPLEMENTATION_VERSION,
            "full_eligible_rows": True,
            "complete_train_epoch_required": True,
            "maximum_epochs": MAX_EPOCHS,
            "development_loss_early_stopping_patience": PATIENCE,
            "topk_selects_checkpoint": False,
            "low_budget_screen": False,
            "historical_test": None,
            "multi_seed": False,
            "single_concurrency": True,
            "in_epoch_checkpoint_date_interval": 50,
            "in_epoch_checkpoint_seconds": 600,
        },
        "precision": precision,
        "deployment_epoch": {
            "source": "four_fold_development_loss_curves",
            "selection": "minimum_mean_relative_regret",
            "tie_tolerance": 1.0e-4,
            "tie_break": "smaller_epoch",
            "fixed_epoch_fold_replay": False,
        },
        "selection": {
            "winner": None,
            "winner_null_until_all_hard_gates_pass": True,
            "active_execution_change_allowed": False,
            "qdp_active_change_allowed": False,
        },
        "contract_path": str(_workspace_path(DEFAULT_CONTRACT_PATH).resolve()),
    }


def initialize_registry(
    *,
    root: str | Path = DEFAULT_STUDY_ROOT,
    pack_manifest: str | Path = DEFAULT_PACK_MANIFEST,
    fold_root: str | Path = DEFAULT_FOLD_ROOT,
    old_registry_path: str | Path = OLD_STUDY_REGISTRY,
    overwrite: bool = False,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    registry_path = root_path / "registry.json"
    if registry_path.is_file() and not overwrite:
        return _read_json(registry_path)
    validation = validate_qcurve_pack(_workspace_path(pack_manifest))
    if validation["status"] != "ok":
        raise ValueError(f"invalid Q-curve pack: {validation['blockers']}")
    folds: dict[str, str] = {}
    fold_contracts: dict[str, str] = {}
    for year in DEVELOPMENT_YEARS:
        fold_path = _workspace_path(fold_root) / f"development_{year}.json"
        verification = verify_qcurve_development_fold(fold_path)
        if verification["status"] != "ok":
            raise ValueError(f"invalid Q-curve fold {year}: {verification['blockers']}")
        folds[str(year)] = str(fold_path.resolve())
        fold_contracts[str(year)] = str(verification["fold_contract_sha256"])
    preflights: dict[str, dict[str, Any]] = {}
    for profile in QONLY_PROFILES:
        path = _preflight_path(root_path, profile)
        if not path.is_file():
            raise ValueError(f"missing Q-only preflight: {path}")
        preflight = _read_json(path)
        if profile == "qcurve_multiscale_ma_qonly" and not bool(preflight.get("performance_gate_passed")):
            raise ValueError("multiscale Q-only performance gate did not pass")
        if not bool(preflight.get("one_step_finite")):
            raise ValueError(f"Q-only one-step finite gate failed: {profile}")
        preflights[profile] = preflight
    old_registry = _read_json(old_registry_path)
    entries: dict[str, Any] = {}
    for year in DEVELOPMENT_YEARS:
        old_key = f"qcurve_lgbm:development{year}:seed{SEED}"
        old_entry = dict(old_registry["entries"][old_key])
        if old_entry.get("status") != "completed" or not Path(str(old_entry["summary_path"])).is_file():
            raise ValueError(f"imported LGBM entry is incomplete: {old_key}")
        entries[old_key] = {
            **old_entry,
            "status": "completed",
            "source_mode": "imported",
            "imported_from_registry": str(_workspace_path(old_registry_path).resolve()),
        }
    for profile in QONLY_PROFILES:
        preflight = preflights[profile]
        for year in DEVELOPMENT_YEARS:
            key = f"{profile}:development{year}:seed{SEED}"
            output = root_path / "runs" / profile / str(year)
            cache = root_path / "cache" / profile / str(year)
            entries[key] = {
                "status": "pending",
                "source_mode": "train",
                "profile": profile,
                "development_year": int(year),
                "seed": SEED,
                "fold_path": folds[str(year)],
                "fold_contract_sha256": fold_contracts[str(year)],
                "objective_digest": _objective_digest(profile),
                "implementation_version": QONLY_IMPLEMENTATION_VERSION,
                "precision": str(preflight["precision"]),
                "microbatch_size": int(preflight["microbatch_size"]),
                "fast_cudnn": True,
                "output_dir": str(output.resolve()),
                "cache_dir": str(cache.resolve()),
                "preflight_path": str(_preflight_path(root_path, profile).resolve()),
            }
    registry = {
        "schema_version": 2,
        "study_id": "seq100_dynamic_qcurve_qonly_2022_2025_v1",
        "status": "registered",
        "created_at": _now(),
        "root": str(root_path.resolve()),
        "pack_manifest": str(_workspace_path(pack_manifest).resolve()),
        "contract": current_contract(root=root_path),
        "profiles": list(PROFILES),
        "development_years": list(DEVELOPMENT_YEARS),
        "folds": folds,
        "fold_contracts": fold_contracts,
        "entries": entries,
        "winner": None,
        "active_execution_changed": False,
        "qdp_active_changed": False,
    }
    _write_json(registry_path, registry)
    return registry


def _entry_process_alive(entry: Mapping[str, Any]) -> bool:
    try:
        import psutil
    except ImportError:
        return False
    output = str(Path(str(entry["output_dir"])).resolve()).lower()
    for process in psutil.process_iter(["cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or []).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "seq100_qcurve_qonly_training" in command and output in command:
            return True
    return False


def _job_command(entry: Mapping[str, Any], *, python: str | Path) -> list[str]:
    return [
        str(_workspace_path(python)),
        "-m",
        "daily_research.path_policy.seq100_qcurve_qonly_training",
        "--fold",
        str(entry["fold_path"]),
        "--profile",
        str(entry["profile"]),
        "--output-dir",
        str(entry["output_dir"]),
        "--cache-dir",
        str(entry["cache_dir"]),
        "--device",
        "cuda",
        "--precision",
        str(entry["precision"]),
        "--microbatch-size",
        str(int(entry["microbatch_size"])),
        "--seed",
        str(SEED),
        "--max-epochs",
        str(MAX_EPOCHS),
        "--patience",
        str(PATIENCE),
    ]


def run_registry(
    *,
    registry_path: str | Path,
    max_jobs: int = 0,
    python: str | Path = DEFAULT_PYTHON,
) -> dict[str, Any]:
    path = _workspace_path(registry_path)
    registry = _read_json(path)
    entries = dict(registry["entries"])
    completed_this_run = 0
    for key, raw in entries.items():
        entry = dict(raw)
        if entry.get("source_mode") == "imported":
            continue
        summary_path = Path(str(entry["output_dir"])) / "summary.json"
        if entry.get("status") == "completed" and summary_path.is_file():
            continue
        if entry.get("status") == "running" and _entry_process_alive(entry):
            return {
                "status": "already_running",
                "entry": key,
                "completed_total": sum(item.get("status") == "completed" for item in entries.values()),
                "total": len(entries),
            }
        if entry.get("status") == "running":
            entry.update(
                {
                    "status": "interrupted",
                    "interruption_reason": "registry_running_without_live_training_process",
                    "interrupted_at": _now(),
                }
            )
        if int(max_jobs) > 0 and completed_this_run >= int(max_jobs):
            break
        command = _job_command(entry, python=python)
        log_path = Path(str(registry["root"])) / "logs" / f"{key.replace(':', '_')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry.update(
            {
                "status": "running",
                "command": command,
                "log_path": str(log_path.resolve()),
                "started_at": _now(),
            }
        )
        entries[key] = entry
        registry["entries"] = entries
        registry["status"] = "running"
        registry["updated_at"] = _now()
        _write_json(path, registry)
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=WORKSPACE_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0 or not summary_path.is_file():
            entry.update({"status": "failed", "returncode": int(completed.returncode), "finished_at": _now()})
            entries[key] = entry
            registry["entries"] = entries
            registry["status"] = "failed"
            registry["updated_at"] = _now()
            _write_json(path, registry)
            raise RuntimeError(f"Q-only job failed: {key}; see {log_path}")
        summary = _read_json(summary_path)
        if (
            summary.get("profile") != entry["profile"]
            or int(summary.get("development_year", 0)) != int(entry["development_year"])
            or int(summary.get("seed", -1)) != SEED
            or summary.get("fold_contract_sha256") != entry["fold_contract_sha256"]
            or summary.get("objective_digest") != entry["objective_digest"]
            or summary.get("implementation_version") != entry["implementation_version"]
            or summary.get("precision") != entry["precision"]
            or int(summary.get("microbatch_size", 0)) != int(entry["microbatch_size"])
        ):
            raise ValueError(f"completed Q-only job provenance mismatch: {key}")
        entry.update(
            {
                "status": "completed",
                "returncode": 0,
                "summary_path": str(summary_path.resolve()),
                "evaluation_path": str(Path(summary["evaluation_path"]).resolve()),
                "history_path": str(Path(summary["history_path"]).resolve()),
                "checkpoint_path": str(Path(summary["checkpoint_path"]).resolve()),
                "input_cache_digest": summary["input_cache_digest"],
                "best_epoch": int(summary["best_epoch"]),
                "finished_at": _now(),
            }
        )
        entries[key] = entry
        completed_this_run += 1
        registry["entries"] = entries
        registry["updated_at"] = _now()
        _write_json(path, registry)
    all_completed = all(item.get("status") == "completed" for item in entries.values())
    registry["status"] = "completed" if all_completed else "running"
    registry["updated_at"] = _now()
    _write_json(path, registry)
    return {
        "status": registry["status"],
        "completed_this_run": completed_this_run,
        "completed_total": sum(item.get("status") == "completed" for item in entries.values()),
        "total": len(entries),
        "registry_path": str(path.resolve()),
    }


def _year_metrics(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    base = dict(evaluation["portfolio"]["base"])
    stress = dict(evaluation["portfolio"]["stress"])
    top3 = dict(evaluation["topk"]["top3"])
    calibration = dict(evaluation["calibration"]["quantile_coverage"]["enter"]["q20"])
    diagnostics = dict(evaluation["diagnostics"])
    return {
        "base_total_log_return": float(base["total_log_return"]),
        "base_net_return": float(base["net_return"]),
        "base_alpha_total_log_return": float(base["alpha_total_log_return"]),
        "stress_total_log_return": float(stress["total_log_return"]),
        "stress_net_return": float(stress["net_return"]),
        "top3_absolute_mean_log_return": float(top3["cost_adjusted_absolute_mean_log_return"]),
        "top3_alpha_mean_log_return": float(top3["cost_adjusted_alpha_mean_log_return"]),
        "q20_breach_rate": float(calibration["actual_breach_rate"]),
        "q20_count": int(calibration["count"]),
        "turnover": float(base["turnover"]),
        "maximum_drawdown": float(base["maximum_drawdown"]),
        "candidate_coverage": float(diagnostics["candidate_coverage"]),
        "execution_mask_coverage": float(diagnostics["execution_mask_coverage"]),
        "q_label_coverage": float(diagnostics["q_label_coverage"]),
        "prediction_coverage": float(diagnostics["prediction_coverage"]),
    }


def _deployment_epoch(profile: str, entries: Mapping[str, Any]) -> dict[str, Any]:
    curves: dict[int, dict[int, float]] = {}
    best_epochs: dict[str, int] = {}
    for year in DEVELOPMENT_YEARS:
        key = f"{profile}:development{year}:seed{SEED}"
        history_payload = _read_json(entries[key]["history_path"])
        rows = list(history_payload.get("history", []) or [])
        curve = {int(row["epoch"]): float(row["development_total_loss"]) for row in rows}
        if not curve:
            raise ValueError(f"missing loss curve: {key}")
        curves[int(year)] = curve
        best_epochs[str(year)] = min(curve, key=curve.get)
    common = sorted(set.intersection(*(set(curve) for curve in curves.values())))
    if not common:
        raise ValueError(f"no common completed epoch across folds: {profile}")
    scores: list[dict[str, Any]] = []
    for epoch in common:
        regrets = {}
        for year, curve in curves.items():
            best = min(curve.values())
            regrets[str(year)] = (curve[epoch] - best) / max(abs(best), 1.0e-8)
        scores.append(
            {
                "epoch": int(epoch),
                "relative_regret_by_year": regrets,
                "deployment_score": float(np.mean(list(regrets.values()))),
            }
        )
    minimum = min(float(row["deployment_score"]) for row in scores)
    selected = min(
        int(row["epoch"])
        for row in scores
        if float(row["deployment_score"]) <= minimum + 1.0e-4
    )
    return {
        "profile": profile,
        "objective_digest": _objective_digest(profile),
        "best_epoch_by_year": best_epochs,
        "loss_curve_by_year": {
            str(year): {str(epoch): loss for epoch, loss in curve.items()}
            for year, curve in curves.items()
        },
        "common_epochs": common,
        "scores": scores,
        "deployment_epoch": selected,
        "formula": "mean((loss_y(e)-best_loss_y)/max(abs(best_loss_y),1e-8))",
        "tie_tolerance": 1.0e-4,
        "tie_break": "smaller_epoch",
        "fixed_epoch_fold_replay": False,
    }


def select_profiles(*, registry_path: str | Path) -> dict[str, Any]:
    path = _workspace_path(registry_path)
    registry = _read_json(path)
    entries = dict(registry["entries"])
    if not all(item.get("status") == "completed" for item in entries.values()):
        raise ValueError("all 12 logical Q-only development jobs must complete before selection")
    root = Path(str(registry["root"]))
    deployment = {
        "schema_version": 1,
        "status": "completed",
        "profiles": {
            profile: _deployment_epoch(profile, entries)
            for profile in QONLY_PROFILES
        },
        "fixed_epoch_fold_replay": False,
        "real_training_note": (
            "train all fully matured labels through approximately D-80 for the selected fixed epoch"
        ),
        "created_at": _now(),
    }
    deployment_path = _write_json(root / "deployment_epoch.json", deployment)
    ledger_rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    eligible_profiles: list[str] = []
    for profile in PROFILES:
        rows: list[dict[str, Any]] = []
        for year in DEVELOPMENT_YEARS:
            key = f"{profile}:development{year}:seed{SEED}"
            evaluation = _read_json(entries[key]["evaluation_path"])
            metrics = _year_metrics(evaluation)
            row = {
                "profile": profile,
                "development_year": int(year),
                "source_mode": entries[key]["source_mode"],
                "best_epoch": int(entries[key]["best_epoch"]),
                **metrics,
            }
            rows.append(row)
            ledger_rows.append(row)
        base = np.asarray([row["base_total_log_return"] for row in rows], dtype=np.float64)
        alpha = np.asarray([row["base_alpha_total_log_return"] for row in rows], dtype=np.float64)
        stress = np.asarray([row["stress_total_log_return"] for row in rows], dtype=np.float64)
        top3 = np.asarray([row["top3_absolute_mean_log_return"] for row in rows], dtype=np.float64)
        top3_alpha = np.asarray([row["top3_alpha_mean_log_return"] for row in rows], dtype=np.float64)
        q20_count = sum(int(row["q20_count"]) for row in rows)
        q20_rate = sum(float(row["q20_breach_rate"]) * int(row["q20_count"]) for row in rows) / max(q20_count, 1)
        coverage = min(
            min(float(row[name]) for row in rows)
            for name in ("candidate_coverage", "execution_mask_coverage", "q_label_coverage", "prediction_coverage")
        )
        gates = {
            "base_absolute_positive_at_least_3_of_4": int(np.count_nonzero(base > 0.0)) >= 3,
            "base_alpha_positive_at_least_3_of_4": int(np.count_nonzero(alpha > 0.0)) >= 3,
            "base_equal_year_mean_positive": float(base.mean()) > 0.0,
            "base_year_median_positive": float(np.median(base)) > 0.0,
            "stress_equal_year_mean_positive": float(stress.mean()) > 0.0,
            "stress_positive_at_least_2_of_4": int(np.count_nonzero(stress > 0.0)) >= 2,
            "top3_absolute_positive_at_least_3_of_4": int(np.count_nonzero(top3 > 0.0)) >= 3,
            "top3_alpha_positive_at_least_3_of_4": int(np.count_nonzero(top3_alpha > 0.0)) >= 3,
            "q20_breach_rate_within_15_25_percent": 0.15 <= q20_rate <= 0.25,
            "candidate_execution_q_prediction_coverage_100_percent": coverage == 1.0,
        }
        eligible = all(gates.values())
        if eligible:
            eligible_profiles.append(profile)
        aggregates[profile] = {
            "eligible": eligible,
            "gates": gates,
            "primary_equal_year_mean_portfolio_log_return": float(base.mean()),
            "year_median_portfolio_log_return": float(np.median(base)),
            "worst_year_portfolio_log_return": float(base.min()),
            "equal_year_mean_top3_alpha": float(top3_alpha.mean()),
            "equal_year_mean_turnover": float(np.mean([row["turnover"] for row in rows])),
            "q20_breach_rate": float(q20_rate),
            "minimum_coverage": float(coverage),
            "deployment_epoch": (
                None if profile == "qcurve_lgbm" else deployment["profiles"][profile]["deployment_epoch"]
            ),
        }
    ranked = sorted(
        eligible_profiles,
        key=lambda profile: (
            -float(aggregates[profile]["primary_equal_year_mean_portfolio_log_return"]),
            -float(aggregates[profile]["worst_year_portfolio_log_return"]),
            -float(aggregates[profile]["equal_year_mean_top3_alpha"]),
            float(aggregates[profile]["equal_year_mean_turnover"]),
            profile,
        ),
    )
    winner = ranked[0] if ranked else None
    old_registry = _read_json(OLD_STUDY_REGISTRY)
    old_gru_rows = []
    for year in DEVELOPMENT_YEARS:
        key = f"qcurve_gru:development{year}:seed{SEED}"
        old_gru_rows.append(
            {
                "profile": "qcurve_gru_qplusaux_historical_reference",
                "development_year": int(year),
                "best_epoch": int(old_registry["entries"][key]["best_epoch"]),
                **_year_metrics(_read_json(old_registry["entries"][key]["evaluation_path"])),
            }
        )
    comparison_path = _write_json(
        root / "historical_qplusaux_comparison.json",
        {
            "schema_version": 1,
            "status": "reference_only_not_selection_candidate",
            "rows": old_gru_rows,
            "source_registry": str(_workspace_path(OLD_STUDY_REGISTRY).resolve()),
            "created_at": _now(),
        },
    )
    ledger_path = _write_json(
        root / "ledger.json",
        {
            "schema_version": 2,
            "status": "completed",
            "rows": ledger_rows,
            "aggregates": aggregates,
            "winner": winner,
            "selection_order": ranked,
            "deployment_epoch_path": str(deployment_path.resolve()),
            "historical_comparison_path": str(comparison_path.resolve()),
            "fixed_epoch_fold_replay": False,
            "maximum_drawdown_role": "report_only_not_gate",
            "active_execution_changed": False,
            "qdp_active_changed": False,
            "created_at": _now(),
        },
    )
    selection_path = _write_json(
        root / "selection.json",
        {
            "schema_version": 2,
            "winner": winner,
            "eligible_profiles": eligible_profiles,
            "ranking": ranked,
            "winner_null_reason": None if winner else "no profile passed every frozen hard gate",
            "freeze_allowed": False,
            "promotion_allowed": False,
            "active_execution_changed": False,
            "qdp_active_changed": False,
            "ledger_path": str(ledger_path.resolve()),
            "deployment_epoch_path": str(deployment_path.resolve()),
            "created_at": _now(),
        },
    )
    registry["winner"] = winner
    registry["ledger_path"] = str(ledger_path.resolve())
    registry["selection_path"] = str(selection_path.resolve())
    registry["deployment_epoch_path"] = str(deployment_path.resolve())
    registry["status"] = "selected"
    registry["updated_at"] = _now()
    _write_json(path, registry)
    return {
        "status": "ok",
        "winner": winner,
        "ledger_path": str(ledger_path.resolve()),
        "selection_path": str(selection_path.resolve()),
        "deployment_epoch_path": str(deployment_path.resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seq100 Q-only development workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    contract = sub.add_parser("contract")
    contract.add_argument("--root", type=Path, default=DEFAULT_STUDY_ROOT)
    register = sub.add_parser("register")
    register.add_argument("--root", type=Path, default=DEFAULT_STUDY_ROOT)
    register.add_argument("--pack-manifest", type=Path, default=DEFAULT_PACK_MANIFEST)
    register.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
    register.add_argument("--old-registry", type=Path, default=OLD_STUDY_REGISTRY)
    register.add_argument("--overwrite", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--registry", type=Path, required=True)
    run.add_argument("--max-jobs", type=int, default=0)
    run.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    select = sub.add_parser("select")
    select.add_argument("--registry", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "contract":
        result = current_contract(root=args.root)
    elif args.command == "register":
        result = initialize_registry(
            root=args.root,
            pack_manifest=args.pack_manifest,
            fold_root=args.fold_root,
            old_registry_path=args.old_registry,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "run":
        result = run_registry(
            registry_path=args.registry,
            max_jobs=int(args.max_jobs),
            python=args.python,
        )
    else:
        result = select_profiles(registry_path=args.registry)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
