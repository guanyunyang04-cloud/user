from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from daily_research.path_policy.seq100_walkforward import (
    DEFAULT_PYTHON,
    WORKSPACE_ROOT,
    _validated_fold_training_contract,
    purged_view_path,
    verify_purged_walkforward_view,
)


CONTRACT_ID = "seq100_pit_adjusted_global_tail_contract_20260711"
INNER_OOS_YEARS = (2018, 2019, 2020, 2021)
OUTER_OOS_YEARS = (2022, 2023, 2024, 2025)
SCREEN_SEEDS = (7,)
CONFIRM_SEEDS = (7, 17, 29)
DEFAULT_ROOT = Path("daily_research/output/path_policy/studies/seq100_pit_adjusted_global_tail_generation_20260711_v1")
DEFAULT_SOURCE_VIEW = Path(
    "daily_research/data/research_store/sequence_pack/"
    "qdp_v2_seq100_path60_todayclose_pit_adjusted_2012_2025_v1/manifest.json"
)
DEFAULT_STORE_ROOT = Path(
    "daily_research/data/research_store/walkforward/"
    "seq100_pit_adjusted_global_tail_contract_20260711_v1"
)
REGISTRY_SCHEMA_VERSION = 2
EVIDENCE_POLICY_RUN_ARTIFACTS = "run_artifacts_only"
EVIDENCE_POLICY_SYNTHETIC_ALLOWED = "synthetic_metrics_allowed_for_tests"

PROFILE_COMMANDS = {
    "baseline": "train-daily-only-summary-v2-ohlcva-aux-low",
    "hard_st": "train-daily-only-summary-v2-ohlcva-aux-low-hard-st",
    "hard_st_global_tail": "train-daily-only-summary-v2-ohlcva-aux-low-hard-st-global-tail",
}
PROFILE_ORDER = tuple(PROFILE_COMMANDS)
METRIC_NAMES = (
    "top3_opportunity_alpha",
    "top3_realized_plan_alpha",
    "oracle_regret",
    "top10_opportunity_alpha",
    "daily_rank_ic",
    "fill_rate",
    "path_mae",
)
SELECTION_POLICY = {
    "primary": "top3_opportunity_alpha",
    "direction": "maximize",
    "tie_breakers": ["worst_fold_top3_opportunity_alpha", "top3_realized_plan_alpha", "daily_rank_ic"],
    "guardrails_relative_to_baseline": {
        "top3_realized_plan_alpha_min_delta": -0.005,
        "oracle_regret_max_delta": 0.01,
        "top10_opportunity_alpha_min_delta": -0.005,
        "daily_rank_ic_min_delta": -0.01,
        "worst_fold_top3_opportunity_alpha_min_delta": -0.01,
        "fill_rate_min_delta": -0.05,
        "path_mae_max_ratio": 1.10,
    },
    "finite_required": list(METRIC_NAMES),
    "folds_required": list(INNER_OOS_YEARS),
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _workspace_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else WORKSPACE_ROOT / raw


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_workspace_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _workspace_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(registry)
    declared = str(payload.pop("registry_sha256", ""))
    if not declared or _payload_sha256(payload) != declared:
        raise ValueError("registry digest mismatch")
    return dict(registry)


def _validate_source_view(
    source_view: str | Path,
    *,
    require_corrected_contract: bool,
) -> tuple[Path, dict[str, Any], str]:
    source_path = _workspace_path(source_view).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source = _read_json(source_path)
    source_sha256 = _file_sha256(source_path)
    if not require_corrected_contract:
        return source_path, source, source_sha256
    if str(source.get("artifact_type", "")) != "qdp_v2_sequence_path_pack":
        raise ValueError("research generation requires a qdp_v2_sequence_path_pack source")
    if int(source.get("lookback_days", 0) or 0) != 100 or int(source.get("forward_days", 0) or 0) != 60:
        raise ValueError("research generation source must use lookback=100 and forward=60")
    semantics = dict(source.get("data_semantics", {}) or {})
    expected_semantics = {
        "price_adjustment": "back_adjust",
        "entry_rule": "open_below_limit_tick",
        "pit_universe_domain": "pit_signal_universe",
    }
    for key, expected in expected_semantics.items():
        if semantics.get(key) != expected:
            raise ValueError(f"research generation source data_semantics.{key} must equal {expected!r}")
    if semantics.get("unfilled_samples_retained") is not True:
        raise ValueError("research generation source must retain unfilled samples")
    labels = dict(source.get("label_arrays", {}) or {})
    ohlcva = dict(labels.get("future_ohlcva_path", {}) or {})
    if str(ohlcva.get("price_anchor", "")) != "today_close":
        raise ValueError("research generation source must use today_close path anchoring")
    masks = dict(source.get("masks", {}) or {})
    required_masks = {"entry_filled", "signal_eligible", "tradable", "price_observed", "va_aux_valid"}
    missing_masks = sorted(required_masks.difference(masks))
    if missing_masks:
        raise ValueError(f"research generation source is missing corrected masks: {missing_masks}")
    return source_path, source, source_sha256


def _fold_bindings(
    *,
    years: Sequence[int],
    store_root: Path,
    source_view_sha256: str,
    require_existing: bool,
) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for year in years:
        view_path = purged_view_path(int(year), store_root=store_root).resolve()
        if not view_path.is_file():
            if require_existing:
                raise FileNotFoundError(
                    f"missing prebuilt fold for {year}: {view_path}; build and verify folds before registry init"
                )
            bindings[str(int(year))] = {
                "oos_year": int(year),
                "view_path": str(view_path),
                "view_sha256": "",
                "fold_training_contract_sha256": "",
                "source_view_sha256": str(source_view_sha256),
            }
            continue
        verification = verify_purged_walkforward_view(view_path)
        if str(verification.get("status", "")) != "ok":
            raise ValueError(f"fold verification failed for {year}: {verification.get('blockers', [])}")
        manifest = _read_json(view_path)
        provenance = dict(manifest.get("source_view_provenance", {}) or {})
        if str(provenance.get("manifest_sha256", "")) != str(source_view_sha256):
            raise ValueError(f"fold {year} is not derived from the registered source view")
        fold_contract = _validated_fold_training_contract(manifest)
        bindings[str(int(year))] = {
            "oos_year": int(year),
            "view_path": str(view_path),
            "view_sha256": _file_sha256(view_path),
            "fold_training_contract_sha256": str(fold_contract["sha256"]),
            "source_view_sha256": str(source_view_sha256),
        }
    return bindings


def _write_json(path: str | Path, payload: Mapping[str, Any], *, immutable: bool) -> Path:
    target = _workspace_path(path)
    serialized = json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if target.exists() and immutable:
        if target.read_text(encoding="utf-8") == serialized:
            return target
        raise FileExistsError(f"immutable artifact already exists with different content: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, target)
    return target


def _parse_ints(raw: str | Iterable[int]) -> tuple[int, ...]:
    if not isinstance(raw, str):
        values = tuple(int(item) for item in raw)
    else:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("seeds must be a non-empty unique sequence")
    return values


def _validate_exact_years(years: Sequence[int], expected: Sequence[int], *, label: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in years)
    if values != tuple(expected):
        raise ValueError(f"{label} years are frozen at {list(expected)}")
    return values


def _job_id(phase: str, profile: str, year: int, seed: int) -> str:
    return f"{phase}:{profile}:oos{int(year)}:seed{int(seed)}"


def _training_command(
    *,
    phase: str,
    profile: str,
    year: int,
    seed: int,
    epochs: int,
    max_samples_per_split: int,
    root: Path,
    store_root: Path,
    python_executable: Path,
) -> list[str]:
    view_path = purged_view_path(int(year), store_root=store_root)
    command = [
        str(_workspace_path(python_executable)),
        "-m",
        "daily_research.path_policy.seq100_mainline",
        PROFILE_COMMANDS[profile],
        "--store-view",
        str(view_path.resolve()),
        "--output-root",
        str((root / "runs" / phase).resolve()),
        "--run-tag",
        f"seq100_pit_{phase}_{profile}_oos{int(year)}_seed{int(seed)}",
        "--epochs",
        str(int(epochs)),
        "--device",
        "cuda",
        "--seed",
        str(int(seed)),
        "--early-stopping-patience",
        "0",
        "--prediction-mode",
        "none",
        "--evaluation-mode",
        "fixed_oos",
    ]
    if int(max_samples_per_split) > 0:
        command.extend(["--max-samples-per-split", str(int(max_samples_per_split))])
    command.append("--json")
    return command


def _fold_build_commands(
    *,
    years: Sequence[int],
    source_view: Path,
    store_root: Path,
    train_start_year: int,
    python_executable: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "oos_year": int(year),
            "view_path": str(purged_view_path(int(year), store_root=store_root).resolve()),
            "command": [
                str(_workspace_path(python_executable)),
                "-m",
                "daily_research.path_policy.seq100_walkforward",
                "build-fold",
                "--source-view",
                str(_workspace_path(source_view).resolve()),
                "--oos-year",
                str(int(year)),
                "--train-start-year",
                str(int(train_start_year)),
                "--store-root",
                str(_workspace_path(store_root).resolve()),
                "--json",
            ],
        }
        for year in years
    ]


def _build_jobs(
    *,
    phase: str,
    profiles: Sequence[str],
    years: Sequence[int],
    seeds: Sequence[int],
    epochs: int,
    max_samples_per_split: int,
    root: Path,
    store_root: Path,
    python_executable: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for profile in profiles:
        if profile not in PROFILE_COMMANDS:
            raise ValueError(f"unsupported profile: {profile}")
        for year in years:
            for seed in seeds:
                job_id = _job_id(phase, profile, int(year), int(seed))
                jobs.append(
                    {
                        "job_id": job_id,
                        "phase": phase,
                        "profile": profile,
                        "profile_command": PROFILE_COMMANDS[profile],
                        "oos_year": int(year),
                        "seed": int(seed),
                        "epochs": int(epochs),
                        "max_samples_per_split": int(max_samples_per_split),
                        "evaluation_mode": "fixed_oos",
                        "checkpoint_policy": "final_epoch",
                        "view_path": str(purged_view_path(int(year), store_root=store_root).resolve()),
                        "command": _training_command(
                            phase=phase,
                            profile=profile,
                            year=int(year),
                            seed=int(seed),
                            epochs=int(epochs),
                            max_samples_per_split=int(max_samples_per_split),
                            root=root,
                            store_root=store_root,
                            python_executable=python_executable,
                        ),
                    }
                )
    return jobs


def initialize_candidate_registry(
    *,
    root: str | Path = DEFAULT_ROOT,
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    screen_seeds: Sequence[int] = SCREEN_SEEDS,
    screen_epochs: int = 1,
    screen_max_samples_per_split: int = 0,
    train_start_year: int = 2012,
    python_executable: str | Path = DEFAULT_PYTHON,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    source_path = _workspace_path(source_view)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    years = _validate_exact_years(INNER_OOS_YEARS, INNER_OOS_YEARS, label="inner OOS")
    seeds = _parse_ints(screen_seeds)
    jobs = _build_jobs(
        phase="screen",
        profiles=PROFILE_ORDER,
        years=years,
        seeds=seeds,
        epochs=int(screen_epochs),
        max_samples_per_split=int(screen_max_samples_per_split),
        root=root_path,
        store_root=_workspace_path(store_root),
        python_executable=_workspace_path(python_executable),
    )
    body = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "artifact_type": "seq100_candidate_registry",
        "contract_id": CONTRACT_ID,
        "status": "registered",
        "source_view": str(source_path.resolve()),
        "source_view_sha256": _file_sha256(source_path),
        "train_start_year": int(train_start_year),
        "inner_oos_years": list(years),
        "historical_outer_oos_years": list(OUTER_OOS_YEARS),
        "outer_evidence_grade": "selection_aware_historical_oos",
        "profiles": [
            {"profile": profile, "profile_command": PROFILE_COMMANDS[profile]}
            for profile in PROFILE_ORDER
        ],
        "phase": {
            "name": "screen",
            "seeds": list(seeds),
            "epochs": int(screen_epochs),
            "max_samples_per_split": int(screen_max_samples_per_split),
        },
        "selection_policy": SELECTION_POLICY,
        "fold_builds": _fold_build_commands(
            years=years,
            source_view=source_path,
            store_root=_workspace_path(store_root),
            train_start_year=int(train_start_year),
            python_executable=_workspace_path(python_executable),
        ),
        "jobs": jobs,
        "protected_boundaries": {
            "active_execution_changed": False,
            "qdp_active_changed": False,
            "legacy_artifacts_overwritten": False,
        },
    }
    registry = {**body, "registry_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "candidate_registry.json", registry, immutable=True)
    _initialize_ledger(root_path)
    return {**registry, "registry_path": str(path.resolve())}


def _initialize_ledger(root: Path) -> dict[str, Any]:
    path = root / "result_ledger.json"
    if path.exists():
        return _read_json(path)
    payload = {
        "schema_version": 1,
        "artifact_type": "seq100_research_result_ledger",
        "created_at": _now(),
        "updated_at": _now(),
        "results": {},
    }
    _write_json(path, payload, immutable=False)
    return payload


def _registry_jobs(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    jobs = {str(item["job_id"]): dict(item) for item in list(registry.get("jobs", []) or [])}
    if len(jobs) != len(list(registry.get("jobs", []) or [])):
        raise ValueError("registry contains duplicate job ids")
    return jobs


def _finite_float(value: Any, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"metric {name} must be finite")
    return parsed


def _command_option(command: Sequence[str], name: str) -> str:
    try:
        index = list(command).index(name)
    except ValueError as exc:
        raise ValueError(f"registered command is missing {name}") from exc
    if index + 1 >= len(command):
        raise ValueError(f"registered command has no value for {name}")
    return str(command[index + 1])


def _validate_run_job_binding(summary: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    command = list(job.get("command", []) or [])
    expected = {
        "run_tag": _command_option(command, "--run-tag"),
        "seed": int(job["seed"]),
        "fold_year": int(job["oos_year"]),
        "pack_manifest": str(Path(str(job["view_path"])).resolve()),
    }
    actual = {
        "run_tag": str(summary.get("run_tag", "")),
        "seed": int(summary.get("seed", -1)),
        "fold_year": int(summary.get("fold_year", -1)),
        "pack_manifest": str(Path(str(summary.get("pack_manifest", ""))).resolve()),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(f"run summary does not match registered job for {key}")
    if str(summary.get("evaluation_mode", "")) != "fixed_oos":
        raise ValueError("research-generation results require fixed_oos evaluation")
    if str(summary.get("checkpoint_policy", "")) != "final_epoch":
        raise ValueError("research-generation results require final_epoch checkpoints")
    expected_value_profile = "smooth_current"
    expected_rank_profile = "local_chunk"
    if str(job["profile"]) in {"hard_st", "hard_st_global_tail"}:
        expected_value_profile = "hard_st"
    if str(job["profile"]) == "hard_st_global_tail":
        expected_rank_profile = "global_tail_512"
    if str(summary.get("path_value_gradient_profile", "")) != expected_value_profile:
        raise ValueError("run path-value gradient profile does not match registered profile")
    if str(summary.get("rank_training_profile", "")) != expected_rank_profile:
        raise ValueError("run ranking profile does not match registered profile")


def _metrics_from_run_dir(run_dir: str | Path, *, job: Mapping[str, Any]) -> tuple[dict[str, float], str, str]:
    root = _workspace_path(run_dir)
    summary_path = root / "sequence_path_training_summary.json"
    summary = _read_json(summary_path)
    _validate_run_job_binding(summary, job)
    topk = pd.read_csv(root / "topk_metrics.csv")
    split = pd.read_csv(root / "split_metrics.csv")
    if "split" in topk.columns:
        topk = topk.loc[topk["split"].astype(str).eq("oos")]
    if "split" in split.columns:
        split = split.loc[split["split"].astype(str).eq("oos")]
    top3 = topk.loc[pd.to_numeric(topk["top_k"], errors="coerce").eq(3)]
    top10 = topk.loc[pd.to_numeric(topk["top_k"], errors="coerce").eq(10)]
    if len(top3) != 1 or len(top10) != 1 or len(split) != 1:
        raise ValueError("run outputs must contain exactly one OOS Top3, Top10, and split row")
    row3, row10, split_row = top3.iloc[0], top10.iloc[0], split.iloc[0]

    def pick(row: pd.Series, names: Sequence[str], label: str) -> float:
        for name in names:
            if name in row.index and pd.notna(row[name]):
                return _finite_float(row[name], name=label)
        raise ValueError(f"run output is missing {label}; tried {list(names)}")

    metrics = {
        "top3_opportunity_alpha": pick(row3, ["alpha_opportunity_value"], "top3_opportunity_alpha"),
        "top3_realized_plan_alpha": pick(row3, ["alpha_realized_plan_value"], "top3_realized_plan_alpha"),
        "oracle_regret": pick(row3, ["selected_oracle_regret", "alpha_oracle_regret"], "oracle_regret"),
        "top10_opportunity_alpha": pick(row10, ["alpha_opportunity_value"], "top10_opportunity_alpha"),
        "daily_rank_ic": pick(split_row, ["rank_ic_mean"], "daily_rank_ic"),
        "fill_rate": pick(row3, ["selected_entry_fill_rate", "selected_realized_fill_rate"], "fill_rate"),
        "path_mae": pick(split_row, ["path_mae"], "path_mae"),
    }
    return metrics, str(summary_path.resolve()), _file_sha256(summary_path)


def register_result(
    *,
    root: str | Path,
    registry_path: str | Path,
    job_id: str,
    metrics: Mapping[str, Any] | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    registry = _read_json(registry_path)
    job = _registry_jobs(registry).get(str(job_id))
    if job is None:
        raise KeyError(f"job is not registered: {job_id}")
    if (metrics is None) == (run_dir is None):
        raise ValueError("provide exactly one of metrics or run_dir")
    if run_dir is not None:
        normalized, summary_path, summary_sha256 = _metrics_from_run_dir(run_dir, job=job)
    else:
        assert metrics is not None
        missing = sorted(set(METRIC_NAMES) - set(metrics))
        extra = sorted(set(metrics) - set(METRIC_NAMES))
        if missing or extra:
            raise ValueError(f"metrics mismatch: missing={missing}, extra={extra}")
        normalized = {name: _finite_float(metrics[name], name=name) for name in METRIC_NAMES}
        summary_path = ""
        summary_sha256 = ""
    record_body = {
        "job_id": str(job_id),
        "registry_sha256": str(registry["registry_sha256"]),
        "profile": str(job["profile"]),
        "phase": str(job["phase"]),
        "oos_year": int(job["oos_year"]),
        "seed": int(job["seed"]),
        "metrics": normalized,
        "summary_path": summary_path,
        "summary_sha256": summary_sha256,
    }
    record = {**record_body, "result_sha256": _payload_sha256(record_body)}
    ledger_path = root_path / "result_ledger.json"
    ledger = _initialize_ledger(root_path)
    results = dict(ledger.get("results", {}) or {})
    existing = results.get(str(job_id))
    if existing is not None and existing != record:
        raise ValueError(f"result already registered with different evidence: {job_id}")
    results[str(job_id)] = record
    ledger["results"] = results
    ledger["updated_at"] = _now()
    _write_json(ledger_path, ledger, immutable=False)
    return record


def _completed_results(
    registry: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    jobs = _registry_jobs(registry)
    results = dict(ledger.get("results", {}) or {})
    missing = sorted(set(jobs) - set(results))
    if missing:
        raise RuntimeError(f"registered matrix is incomplete; missing {len(missing)} jobs, first={missing[:3]}")
    records: list[dict[str, Any]] = []
    for job_id, job in jobs.items():
        record = dict(results[job_id])
        if str(record.get("registry_sha256", "")) != str(registry.get("registry_sha256", "")):
            raise ValueError(f"result registry binding mismatch: {job_id}")
        for key in ("profile", "phase", "oos_year", "seed"):
            if record.get(key) != job.get(key):
                raise ValueError(f"result job binding mismatch for {job_id}: {key}")
        records.append(record)
    return records


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = [
        {
            "profile": str(item["profile"]),
            "oos_year": int(item["oos_year"]),
            "seed": int(item["seed"]),
            **{name: float(dict(item["metrics"])[name]) for name in METRIC_NAMES},
        }
        for item in records
    ]
    frame = pd.DataFrame(rows)
    output: dict[str, dict[str, Any]] = {}
    for profile, group in frame.groupby("profile", sort=False):
        fold_values = group.groupby("oos_year", sort=True)["top3_opportunity_alpha"].mean()
        output[str(profile)] = {
            "profile": str(profile),
            "job_count": int(len(group)),
            "seed_count": int(group["seed"].nunique()),
            "fold_count": int(group["oos_year"].nunique()),
            **{name: float(group[name].mean()) for name in METRIC_NAMES},
            "worst_fold_top3_opportunity_alpha": float(fold_values.min()),
            "fold_top3_opportunity_alpha": {str(int(year)): float(value) for year, value in fold_values.items()},
        }
    return output


def _guardrail_verdict(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    policy = dict(SELECTION_POLICY["guardrails_relative_to_baseline"])
    checks = {
        "top3_realized_plan_alpha": float(candidate["top3_realized_plan_alpha"])
        >= float(baseline["top3_realized_plan_alpha"]) + float(policy["top3_realized_plan_alpha_min_delta"]),
        "oracle_regret": float(candidate["oracle_regret"])
        <= float(baseline["oracle_regret"]) + float(policy["oracle_regret_max_delta"]),
        "top10_opportunity_alpha": float(candidate["top10_opportunity_alpha"])
        >= float(baseline["top10_opportunity_alpha"]) + float(policy["top10_opportunity_alpha_min_delta"]),
        "daily_rank_ic": float(candidate["daily_rank_ic"])
        >= float(baseline["daily_rank_ic"]) + float(policy["daily_rank_ic_min_delta"]),
        "worst_fold": float(candidate["worst_fold_top3_opportunity_alpha"])
        >= float(baseline["worst_fold_top3_opportunity_alpha"])
        + float(policy["worst_fold_top3_opportunity_alpha_min_delta"]),
        "fill_rate": float(candidate["fill_rate"])
        >= float(baseline["fill_rate"]) + float(policy["fill_rate_min_delta"]),
        "path_mae": float(candidate["path_mae"])
        <= float(baseline["path_mae"]) * float(policy["path_mae_max_ratio"]),
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def select_profiles(
    *,
    registry_path: str | Path,
    ledger_path: str | Path,
) -> dict[str, Any]:
    registry = _read_json(registry_path)
    records = _completed_results(registry, _read_json(ledger_path))
    aggregates = _aggregate_records(records)
    baseline = aggregates.get("baseline")
    if baseline is None:
        raise RuntimeError("baseline is required for guardrail comparison")
    ranked: list[dict[str, Any]] = []
    for profile in PROFILE_ORDER:
        aggregate = aggregates.get(profile)
        if aggregate is None:
            continue
        verdict = {"passed": True, "checks": {"baseline_reference": True}} if profile == "baseline" else _guardrail_verdict(aggregate, baseline)
        ranked.append({**aggregate, "guardrails": verdict})
    ranked.sort(
        key=lambda item: (
            bool(dict(item["guardrails"])["passed"]),
            float(item["top3_opportunity_alpha"]),
            float(item["worst_fold_top3_opportunity_alpha"]),
            float(item["top3_realized_plan_alpha"]),
            float(item["daily_rank_ic"]),
        ),
        reverse=True,
    )
    return {
        "registry_sha256": str(registry["registry_sha256"]),
        "phase": str(dict(registry.get("phase", {}))["name"]),
        "ranking": ranked,
        "winner": str(ranked[0]["profile"]),
    }


def create_confirmation_registry(
    *,
    root: str | Path,
    candidate_registry_path: str | Path,
    ledger_path: str | Path,
    confirm_seeds: Sequence[int] = CONFIRM_SEEDS,
    confirm_epochs: int = 10,
    shortlist_size: int = 2,
    python_executable: str | Path = DEFAULT_PYTHON,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    candidate = _read_json(candidate_registry_path)
    selection = select_profiles(registry_path=candidate_registry_path, ledger_path=ledger_path)
    eligible = [
        str(item["profile"])
        for item in list(selection["ranking"])
        if bool(dict(item["guardrails"])["passed"]) and str(item["profile"]) != "baseline"
    ]
    if int(shortlist_size) < 1:
        raise ValueError("shortlist_size must be at least 1")
    shortlist = ["baseline", *eligible[: max(int(shortlist_size) - 1, 0)]]
    seeds = _parse_ints(confirm_seeds)
    store_root = Path(str(Path(list(candidate["jobs"])[0]["view_path"]).parents[1]))
    jobs = _build_jobs(
        phase="confirmation",
        profiles=shortlist,
        years=INNER_OOS_YEARS,
        seeds=seeds,
        epochs=int(confirm_epochs),
        max_samples_per_split=0,
        root=root_path,
        store_root=store_root,
        python_executable=_workspace_path(python_executable),
    )
    body = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "artifact_type": "seq100_confirmation_registry",
        "contract_id": CONTRACT_ID,
        "status": "registered",
        "parent_candidate_registry": str(_workspace_path(candidate_registry_path).resolve()),
        "parent_registry_sha256": str(candidate["registry_sha256"]),
        "screen_selection": selection,
        "shortlist": shortlist,
        "inner_oos_years": list(INNER_OOS_YEARS),
        "phase": {"name": "confirmation", "seeds": list(seeds), "epochs": int(confirm_epochs)},
        "selection_policy": SELECTION_POLICY,
        "jobs": jobs,
    }
    registry = {**body, "registry_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "confirmation_registry.json", registry, immutable=True)
    return {**registry, "registry_path": str(path.resolve())}


def freeze_champion(
    *,
    root: str | Path,
    confirmation_registry_path: str | Path,
    ledger_path: str | Path,
    champion: str,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    registry = _read_json(confirmation_registry_path)
    ledger = _read_json(ledger_path)
    records = _completed_results(registry, ledger)
    selection = select_profiles(registry_path=confirmation_registry_path, ledger_path=ledger_path)
    selected = str(selection["winner"])
    if str(champion) != selected:
        raise ValueError(f"explicit champion {champion!r} is not the policy winner {selected!r}")
    body = {
        "schema_version": 1,
        "artifact_type": "seq100_frozen_champion",
        "contract_id": CONTRACT_ID,
        "status": "frozen",
        "champion": selected,
        "profile_command": PROFILE_COMMANDS[selected],
        "confirmation_registry": str(_workspace_path(confirmation_registry_path).resolve()),
        "confirmation_registry_sha256": str(registry["registry_sha256"]),
        "confirmation_result_sha256s": {
            str(item["job_id"]): str(item["result_sha256"])
            for item in sorted(records, key=lambda value: str(value["job_id"]))
        },
        "selection": selection,
        "inner_oos_years_consumed": list(INNER_OOS_YEARS),
        "historical_outer_oos_years_unconsumed": list(OUTER_OOS_YEARS),
        "outer_evidence_grade": "selection_aware_historical_oos",
        "active_execution_changed": False,
        "qdp_active_changed": False,
    }
    frozen = {**body, "freeze_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "frozen_champion.json", frozen, immutable=True)
    return {**frozen, "freeze_path": str(path.resolve())}


def begin_outer_audit(
    *,
    root: str | Path,
    freeze_path: str | Path,
    source_view: str | Path = DEFAULT_SOURCE_VIEW,
    store_root: str | Path = DEFAULT_STORE_ROOT,
    epochs: int = 10,
    seed: int = 7,
    train_start_year: int = 2012,
    python_executable: str | Path = DEFAULT_PYTHON,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    frozen = _read_json(freeze_path)
    if str(frozen.get("status", "")) != "frozen" or not str(frozen.get("freeze_sha256", "")):
        raise ValueError("outer audit requires an explicit immutable frozen champion")
    declared_freeze_sha256 = str(frozen["freeze_sha256"])
    freeze_body = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if _payload_sha256(freeze_body) != declared_freeze_sha256:
        raise ValueError("frozen champion digest mismatch")
    claim_path = root_path / "outer_audit_consumption.json"
    if claim_path.exists():
        raise RuntimeError("historical outer audit has already been consumed or claimed")
    profile = str(frozen["champion"])
    jobs = _build_jobs(
        phase="outer_audit",
        profiles=(profile,),
        years=OUTER_OOS_YEARS,
        seeds=(int(seed),),
        epochs=int(epochs),
        max_samples_per_split=0,
        root=root_path,
        store_root=_workspace_path(store_root),
        python_executable=_workspace_path(python_executable),
    )
    body = {
        "schema_version": 1,
        "artifact_type": "seq100_outer_audit_registry",
        "contract_id": CONTRACT_ID,
        "status": "claimed",
        "freeze_path": str(_workspace_path(freeze_path).resolve()),
        "freeze_sha256": declared_freeze_sha256,
        "champion": profile,
        "historical_outer_oos_years": list(OUTER_OOS_YEARS),
        "evidence_grade": "selection_aware_historical_oos",
        "selection_aware": True,
        "repeat_allowed": False,
        "fold_builds": _fold_build_commands(
            years=OUTER_OOS_YEARS,
            source_view=_workspace_path(source_view),
            store_root=_workspace_path(store_root),
            train_start_year=int(train_start_year),
            python_executable=_workspace_path(python_executable),
        ),
        "jobs": jobs,
        "claimed_at": _now(),
    }
    claim = {**body, "registry_sha256": _payload_sha256(body)}
    path = _write_json(claim_path, claim, immutable=True)
    return {**claim, "registry_path": str(path.resolve())}


def finalize_outer_audit(
    *,
    root: str | Path,
    ledger_path: str | Path,
) -> dict[str, Any]:
    root_path = _workspace_path(root)
    claim_path = root_path / "outer_audit_consumption.json"
    claim = _read_json(claim_path)
    records = _completed_results(claim, _read_json(ledger_path))
    aggregates = _aggregate_records(records)
    champion = str(claim["champion"])
    body = {
        "schema_version": 1,
        "artifact_type": "seq100_outer_audit_result",
        "contract_id": CONTRACT_ID,
        "status": "completed",
        "outer_registry": str(claim_path.resolve()),
        "outer_registry_sha256": str(claim["registry_sha256"]),
        "freeze_sha256": str(claim["freeze_sha256"]),
        "champion": champion,
        "historical_outer_oos_years": list(OUTER_OOS_YEARS),
        "evidence_grade": "selection_aware_historical_oos",
        "selection_aware": True,
        "repeat_allowed": False,
        "aggregate": aggregates[champion],
        "completed_at": _now(),
        "active_execution_changed": False,
        "qdp_active_changed": False,
    }
    result = {**body, "result_sha256": _payload_sha256(body)}
    path = _write_json(root_path / "outer_audit_result.json", result, immutable=True)
    return {**result, "result_path": str(path.resolve())}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Selection-safe seq100 research-generation orchestration.")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    init.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    init.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    init.add_argument("--screen-seeds", default=",".join(map(str, SCREEN_SEEDS)))
    init.add_argument("--screen-epochs", type=int, default=1)
    init.add_argument(
        "--screen-max-samples-per-split",
        type=int,
        default=0,
        help="Training-only complete-date cap for the screen; inner OOS metrics remain full-universe.",
    )

    register = sub.add_parser("register-result")
    register.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    register.add_argument("--registry", type=Path, required=True)
    register.add_argument("--job-id", required=True)
    register.add_argument("--run-dir", type=Path)
    register.add_argument("--metrics-json", type=Path)

    advance = sub.add_parser("advance-confirmation")
    advance.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    advance.add_argument("--candidate-registry", type=Path, required=True)
    advance.add_argument("--ledger", type=Path, required=True)
    advance.add_argument("--confirm-seeds", default=",".join(map(str, CONFIRM_SEEDS)))
    advance.add_argument("--confirm-epochs", type=int, default=10)
    advance.add_argument("--shortlist-size", type=int, default=2)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    freeze.add_argument("--confirmation-registry", type=Path, required=True)
    freeze.add_argument("--ledger", type=Path, required=True)
    freeze.add_argument("--champion", choices=PROFILE_ORDER, required=True)

    outer = sub.add_parser("begin-outer-audit")
    outer.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    outer.add_argument("--freeze", type=Path, required=True)
    outer.add_argument("--source-view", type=Path, default=DEFAULT_SOURCE_VIEW)
    outer.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    outer.add_argument("--epochs", type=int, default=10)
    outer.add_argument("--seed", type=int, default=7)

    finalize = sub.add_parser("finalize-outer-audit")
    finalize.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    finalize.add_argument("--ledger", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init":
        result = initialize_candidate_registry(
            root=args.root,
            source_view=args.source_view,
            store_root=args.store_root,
            screen_seeds=_parse_ints(args.screen_seeds),
            screen_epochs=args.screen_epochs,
            screen_max_samples_per_split=args.screen_max_samples_per_split,
        )
    elif args.command == "register-result":
        metrics = _read_json(args.metrics_json) if args.metrics_json else None
        result = register_result(
            root=args.root,
            registry_path=args.registry,
            job_id=args.job_id,
            metrics=metrics,
            run_dir=args.run_dir,
        )
    elif args.command == "advance-confirmation":
        result = create_confirmation_registry(
            root=args.root,
            candidate_registry_path=args.candidate_registry,
            ledger_path=args.ledger,
            confirm_seeds=_parse_ints(args.confirm_seeds),
            confirm_epochs=args.confirm_epochs,
            shortlist_size=args.shortlist_size,
        )
    elif args.command == "freeze":
        result = freeze_champion(
            root=args.root,
            confirmation_registry_path=args.confirmation_registry,
            ledger_path=args.ledger,
            champion=args.champion,
        )
    elif args.command == "begin-outer-audit":
        result = begin_outer_audit(
            root=args.root,
            freeze_path=args.freeze,
            source_view=args.source_view,
            store_root=args.store_root,
            epochs=args.epochs,
            seed=args.seed,
        )
    else:
        result = finalize_outer_audit(root=args.root, ledger_path=args.ledger)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
