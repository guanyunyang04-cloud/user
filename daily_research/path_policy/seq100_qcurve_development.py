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
    build_qcurve_development_folds,
    verify_qcurve_development_fold,
)
from daily_research.path_policy.seq100_qcurve_pack import DEFAULT_CONTRACT, validate_qcurve_pack


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
DEFAULT_STUDY_ROOT = Path("daily_research/output/path_policy/studies/seq100_dynamic_qcurve_2022_2025")
PROFILES = ("qcurve_lgbm", "qcurve_gru", "qcurve_multiscale_ma")
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


def current_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow_id": "seq100_dynamic_qcurve_development",
        "profiles": list(PROFILES),
        "development_years": list(DEVELOPMENT_YEARS),
        "seed": SEED,
        "training": {
            "full_eligible_rows": True,
            "complete_train_epoch_required": True,
            "maximum_epochs": MAX_EPOCHS,
            "development_loss_early_stopping_patience": PATIENCE,
            "topk_selects_checkpoint": False,
            "low_budget_screen": False,
            "historical_test": None,
            "multi_seed": False,
        },
        "selection": {
            "winner": None,
            "winner_null_until_all_hard_gates_pass": True,
            "active_execution_change_allowed": False,
            "qdp_active_change_allowed": False,
        },
        "contract_path": str(_workspace_path(DEFAULT_CONTRACT).resolve()),
    }


def initialize_registry(
    *,
    root: str | Path = DEFAULT_STUDY_ROOT,
    pack_manifest: str | Path = DEFAULT_PACK_MANIFEST,
    fold_root: str | Path = DEFAULT_FOLD_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    registry_path = root_path / "registry.json"
    if registry_path.is_file() and not bool(overwrite):
        return _read_json(registry_path)
    pack_validation = validate_qcurve_pack(_workspace_path(pack_manifest))
    if pack_validation["status"] != "ok":
        raise ValueError(f"invalid Q-curve pack: {pack_validation['blockers']}")
    folds: dict[str, str] = {}
    fold_contracts: dict[str, str] = {}
    for year in DEVELOPMENT_YEARS:
        fold_path = _workspace_path(fold_root) / f"development_{year}.json"
        verification = verify_qcurve_development_fold(fold_path)
        if verification["status"] != "ok":
            raise ValueError(f"invalid Q-curve fold {year}: {verification['blockers']}")
        folds[str(year)] = str(fold_path.resolve())
        fold_contracts[str(year)] = str(verification["fold_contract_sha256"])
    entries: dict[str, Any] = {}
    for profile in PROFILES:
        for year in DEVELOPMENT_YEARS:
            key = f"{profile}:development{year}:seed{SEED}"
            output = root_path / "runs" / profile / str(year)
            entries[key] = {
                "status": "pending",
                "profile": profile,
                "development_year": int(year),
                "seed": SEED,
                "fold_path": folds[str(year)],
                "fold_contract_sha256": fold_contracts[str(year)],
                "output_dir": str(output.resolve()),
            }
    registry = {
        "schema_version": 1,
        "study_id": "seq100_dynamic_qcurve_2022_2025_v1",
        "status": "registered",
        "created_at": _now(),
        "root": str(root_path.resolve()),
        "pack_manifest": str(_workspace_path(pack_manifest).resolve()),
        "contract": current_contract(),
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


def _job_command(entry: Mapping[str, Any], *, python: str | Path) -> list[str]:
    module = (
        "daily_research.path_policy.seq100_qcurve_lgbm"
        if str(entry["profile"]) == "qcurve_lgbm"
        else "daily_research.path_policy.seq100_qcurve_training"
    )
    command = [
        str(_workspace_path(python)),
        "-m",
        module,
        "--fold",
        str(entry["fold_path"]),
        "--output-dir",
        str(entry["output_dir"]),
        "--seed",
        str(SEED),
        "--max-epochs",
        str(MAX_EPOCHS),
        "--patience",
        str(PATIENCE),
    ]
    if str(entry["profile"]) != "qcurve_lgbm":
        command.extend(["--profile", str(entry["profile"]), "--device", "cuda", "--microbatch-size", "512"])
    return command


def run_registry(
    *,
    registry_path: str | Path,
    max_jobs: int = 0,
    python: str | Path = DEFAULT_PYTHON,
) -> dict[str, Any]:
    path = _workspace_path(registry_path)
    registry = _read_json(path)
    entries = dict(registry["entries"])
    completed_jobs = 0
    for key, raw in entries.items():
        entry = dict(raw)
        summary_path = Path(str(entry["output_dir"])) / "summary.json"
        if str(entry.get("status", "")) == "completed" and summary_path.is_file():
            continue
        if int(max_jobs) > 0 and completed_jobs >= int(max_jobs):
            break
        command = _job_command(entry, python=python)
        log_path = _workspace_path(registry["root"]) / "logs" / f"{key.replace(':', '_')}.log"
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
        if int(completed.returncode) != 0 or not summary_path.is_file():
            entry.update({"status": "failed", "returncode": int(completed.returncode), "finished_at": _now()})
            entries[key] = entry
            registry["entries"] = entries
            registry["status"] = "failed"
            registry["updated_at"] = _now()
            _write_json(path, registry)
            raise RuntimeError(f"Q-curve job failed: {key}; see {log_path}")
        summary = _read_json(summary_path)
        if (
            str(summary.get("profile", "")) != str(entry["profile"])
            or int(summary.get("development_year", 0)) != int(entry["development_year"])
            or int(summary.get("seed", -1)) != SEED
            or str(summary.get("fold_contract_sha256", "")) != str(entry["fold_contract_sha256"])
        ):
            raise ValueError(f"completed Q-curve job provenance mismatch: {key}")
        entry.update(
            {
                "status": "completed",
                "returncode": 0,
                "summary_path": str(summary_path.resolve()),
                "evaluation_path": str(Path(summary["evaluation_path"]).resolve()),
                "best_epoch": int(summary["best_epoch"]),
                "finished_at": _now(),
            }
        )
        entries[key] = entry
        completed_jobs += 1
        registry["entries"] = entries
        registry["updated_at"] = _now()
        _write_json(path, registry)
    all_completed = all(str(item.get("status", "")) == "completed" for item in entries.values())
    registry["status"] = "completed" if all_completed else "running"
    registry["updated_at"] = _now()
    _write_json(path, registry)
    return {
        "status": registry["status"],
        "completed_this_run": completed_jobs,
        "completed_total": sum(str(item.get("status", "")) == "completed" for item in entries.values()),
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


def select_profiles(*, registry_path: str | Path) -> dict[str, Any]:
    path = _workspace_path(registry_path)
    registry = _read_json(path)
    entries = dict(registry["entries"])
    if not all(str(item.get("status", "")) == "completed" for item in entries.values()):
        raise ValueError("all 12 Q-curve development jobs must complete before selection")
    ledger_rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    eligible_profiles: list[str] = []
    for profile in PROFILES:
        rows: list[dict[str, Any]] = []
        for year in DEVELOPMENT_YEARS:
            key = f"{profile}:development{year}:seed{SEED}"
            evaluation = _read_json(entries[key]["evaluation_path"])
            metrics = _year_metrics(evaluation)
            row = {"profile": profile, "development_year": year, **metrics}
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
            "maximum_drawdown_by_year": {
                str(row["development_year"]): float(row["maximum_drawdown"]) for row in rows
            },
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
    ledger = {
        "schema_version": 1,
        "status": "completed",
        "rows": ledger_rows,
        "aggregates": aggregates,
        "winner": winner,
        "selection_order": ranked,
        "maximum_drawdown_role": "report_only_not_gate",
        "active_execution_changed": False,
        "qdp_active_changed": False,
        "created_at": _now(),
    }
    root = _workspace_path(registry["root"])
    ledger_path = _write_json(root / "ledger.json", ledger)
    selection_path = _write_json(
        root / "selection.json",
        {
            "schema_version": 1,
            "winner": winner,
            "eligible_profiles": eligible_profiles,
            "ranking": ranked,
            "winner_null_reason": None if winner else "no profile passed every frozen hard gate",
            "freeze_allowed": bool(winner),
            "active_execution_changed": False,
            "qdp_active_changed": False,
            "ledger_path": str(ledger_path.resolve()),
            "created_at": _now(),
        },
    )
    registry["winner"] = winner
    registry["ledger_path"] = str(ledger_path.resolve())
    registry["selection_path"] = str(selection_path.resolve())
    registry["status"] = "selected"
    registry["updated_at"] = _now()
    _write_json(path, registry)
    return {"status": "ok", "winner": winner, "ledger_path": str(ledger_path), "selection_path": str(selection_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seq100 dynamic Q-curve development workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    folds = sub.add_parser("build-folds")
    folds.add_argument("--pack-manifest", type=Path, default=DEFAULT_PACK_MANIFEST)
    folds.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
    folds.add_argument("--overwrite", action="store_true")
    audit = sub.add_parser("zero-audit")
    audit.add_argument("--pack-manifest", type=Path, default=DEFAULT_PACK_MANIFEST)
    audit.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STUDY_ROOT / "zero_training_old_baseline_exit_audit.json",
    )
    register = sub.add_parser("register")
    register.add_argument("--root", type=Path, default=DEFAULT_STUDY_ROOT)
    register.add_argument("--pack-manifest", type=Path, default=DEFAULT_PACK_MANIFEST)
    register.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
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
        result = current_contract()
    elif args.command == "build-folds":
        result = build_qcurve_development_folds(
            pack_manifest=args.pack_manifest,
            output_root=args.fold_root,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "zero-audit":
        from daily_research.path_policy.seq100_qcurve_zero_training_audit import run_zero_training_audit

        result = run_zero_training_audit(
            pack_manifest=args.pack_manifest,
            output_path=args.output,
        )
    elif args.command == "register":
        result = initialize_registry(
            root=args.root,
            pack_manifest=args.pack_manifest,
            fold_root=args.fold_root,
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
