"""Two-fold, bounded-memory baseline training for minute-v2 events."""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil

from quantlab.core.io import read_json, write_json

from .contracts import KEY_COLUMNS, MODEL_FEATURE_COLUMNS, MinuteV2Error
from .mining import mine_formula_features, write_mining_result
from .models import evaluate_scores, fit_lightgbm_ranker, fit_ridge, rule_score
from .replay import EventReplayConfig, replay_events


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_start_year: int
    train_end_year: int
    validation_year: int
    test_start_year: int
    test_end_year: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


TWO_FOLDS = (
    FoldSpec(1, 2012, 2017, 2018, 2019, 2020),
    FoldSpec(2, 2012, 2019, 2020, 2021, 2022),
)

EXECUTION_LABEL_COLUMNS = (
    "planned_exit_date",
    "entry_bar_time",
    "entry_price",
    "entry_amount",
    "entry_executable",
    "actual_exit_date",
    "exit_amount",
    "label_observed",
)


def _literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _scan(paths: list[Path]) -> str:
    if not paths:
        raise MinuteV2Error("minute_v2_training_paths_empty")
    values = ",".join(_literal(path) for path in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def _period_paths(dataset_root: Path, start_year: int, end_year: int) -> tuple[list[Path], list[Path]]:
    events: list[Path] = []
    labels: list[Path] = []
    for year in range(int(start_year), int(end_year) + 1):
        for month in range(1, 13):
            directory = dataset_root / "months" / f"year={year:04d}" / f"month={month:02d}"
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                raise MinuteV2Error(f"minute_v2_training_manifest_missing:{year:04d}-{month:02d}")
            manifest = read_json(manifest_path)
            if manifest.get("status") != "ok":
                raise MinuteV2Error(f"minute_v2_training_manifest_not_ok:{manifest_path}")
            artifacts = dict(manifest.get("artifacts", {}))
            event_path = Path(str(dict(artifacts.get("events", {})).get("path", "")))
            label_path = Path(str(dict(artifacts.get("labels", {})).get("path", "")))
            if not event_path.is_file() or not label_path.is_file():
                raise MinuteV2Error(f"minute_v2_training_artifact_missing:{year:04d}-{month:02d}")
            events.append(event_path.resolve())
            labels.append(label_path.resolve())
    return events, labels


def _load_period(
    dataset_root: Path,
    *,
    start_year: int,
    end_year: int,
    sample_basis_points: int,
    include_execution: bool,
    temp_directory: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 1 <= int(sample_basis_points) <= 10_000:
        raise MinuteV2Error("minute_v2_training_sample_basis_points_invalid")
    event_paths, label_paths = _period_paths(dataset_root, start_year, end_year)
    event_scan = _scan(event_paths)
    label_scan = _scan(label_paths)
    feature_projection = ",".join(f"e.{name}" for name in MODEL_FEATURE_COLUMNS)
    execution_projection = (
        "," + ",".join(f"l.{name}" for name in EXECUTION_LABEL_COLUMNS)
        if include_execution
        else ""
    )
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"SET temp_directory={_literal(temp_directory)}")
        query = f"""
            SELECT
                CAST(e.symbol AS VARCHAR) AS symbol,
                CAST(e.trade_date AS VARCHAR) AS trade_date,
                CAST(e.bar_time AS VARCHAR) AS bar_time,
                {feature_projection},
                CAST(l.label_net_return AS DOUBLE) AS label_net_return
                {execution_projection}
            FROM {event_scan} e
            JOIN {label_scan} l USING(symbol,trade_date,bar_time)
            WHERE l.label_observed
              AND HASH(CAST(e.trade_date AS VARCHAR),CAST(e.bar_time AS VARCHAR)) % 10000
                  < {int(sample_basis_points)}
            ORDER BY e.trade_date,e.bar_time,e.symbol
        """
        frame = connection.execute(query).fetchdf()
    finally:
        connection.close()
    if frame.empty:
        raise MinuteV2Error(
            f"minute_v2_training_period_empty:{start_year}:{end_year}:{sample_basis_points}"
        )
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_training_duplicate_keys")
    for name in MODEL_FEATURE_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float32")
    frame["label_net_return"] = pd.to_numeric(frame["label_net_return"], errors="coerce")
    metadata = {
        "start_year": int(start_year),
        "end_year": int(end_year),
        "sample_basis_points": int(sample_basis_points),
        "sampling_unit": "complete (trade_date, bar_time) cross-sectional groups",
        "row_count": int(len(frame)),
        "group_count": int(frame.groupby(["trade_date", "bar_time"]).ngroups),
        "event_file_count": len(event_paths),
        "label_file_count": len(label_paths),
    }
    return frame, metadata


def _bounded_group_sample(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    sizes = (
        frame.groupby(["trade_date", "bar_time"], sort=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    sizes["hash"] = pd.util.hash_pandas_object(
        sizes[["trade_date", "bar_time"]], index=False
    ).astype("uint64")
    sizes = sizes.sort_values(["hash", "trade_date", "bar_time"], kind="stable")
    selected = sizes.loc[sizes["rows"].cumsum() <= int(maximum_rows), ["trade_date", "bar_time"]]
    if selected.empty:
        selected = sizes.head(1)[["trade_date", "bar_time"]]
    return frame.merge(selected, on=["trade_date", "bar_time"], how="inner", validate="many_to_one")


def _evaluate_and_replay(
    test: pd.DataFrame,
    scores: np.ndarray | pd.Series,
    *,
    model_name: str,
    output_directory: Path,
) -> dict[str, Any]:
    scored = test.copy()
    scored["score"] = np.asarray(scores, dtype=float)
    metrics = evaluate_scores(scored)
    replay_result, daily, trades = replay_events(
        scored,
        replay_config=EventReplayConfig(score_threshold=-1.0e30),
    )
    daily.to_parquet(output_directory / f"{model_name}_replay_daily.parquet", index=False)
    trades.to_parquet(output_directory / f"{model_name}_replay_trades.parquet", index=False)
    return {"score_metrics": metrics, "sampled_event_replay": replay_result}


def _run_fold(
    dataset_root: Path,
    output_root: Path,
    spec: FoldSpec,
    *,
    train_sample_basis_points: int,
    evaluation_sample_basis_points: int,
    mining_result: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    directory = output_root / f"fold_{spec.fold}"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"fold_{spec.fold}_", dir=output_root) as temporary:
        temp = Path(temporary)
        train, train_meta = _load_period(
            dataset_root,
            start_year=spec.train_start_year,
            end_year=spec.train_end_year,
            sample_basis_points=train_sample_basis_points,
            include_execution=False,
            temp_directory=temp,
        )
        validation, validation_meta = _load_period(
            dataset_root,
            start_year=spec.validation_year,
            end_year=spec.validation_year,
            sample_basis_points=evaluation_sample_basis_points,
            include_execution=False,
            temp_directory=temp,
        )
        test, test_meta = _load_period(
            dataset_root,
            start_year=spec.test_start_year,
            end_year=spec.test_end_year,
            sample_basis_points=evaluation_sample_basis_points,
            include_execution=True,
            temp_directory=temp,
        )
        if spec.fold == 1 and mining_result is None:
            mining_train = _bounded_group_sample(train, 120_000)
            mining_validation = _bounded_group_sample(validation, 80_000)
            mining_result = mine_formula_features(
                mining_train,
                mining_validation,
                maximum_candidates=120,
                maximum_selected=12,
                minimum_coverage=0.85,
            )
            mining_result["bounded_train_rows"] = int(len(mining_train))
            mining_result["bounded_validation_rows"] = int(len(mining_validation))
            write_mining_result(mining_result, output_root / "formula_mining_fold_1.json")
        ridge = fit_ridge(train, alpha=10.0)
        ridge.save(directory / "ridge.json")
        lightgbm, lightgbm_meta = fit_lightgbm_ranker(train, validation=validation)
        lightgbm.booster_.save_model(str(directory / "lightgbm.txt"))
        evaluations = {
            "rule": _evaluate_and_replay(
                test,
                rule_score(test),
                model_name="rule",
                output_directory=directory,
            ),
            "ridge": _evaluate_and_replay(
                test,
                ridge.predict(test),
                model_name="ridge",
                output_directory=directory,
            ),
            "lightgbm": _evaluate_and_replay(
                test,
                lightgbm.predict(test.loc[:, list(MODEL_FEATURE_COLUMNS)]),
                model_name="lightgbm",
                output_directory=directory,
            ),
        }
        result = {
            "schema": "quantlab.minute_v2_fold/1",
            "status": "ok",
            "fold": spec.as_dict(),
            "period_samples": {
                "train": train_meta,
                "validation": validation_meta,
                "test": test_meta,
            },
            "feature_names": list(MODEL_FEATURE_COLUMNS),
            "lightgbm": lightgbm_meta,
            "evaluations": evaluations,
            "available_memory_gib_after_fold": psutil.virtual_memory().available / 1024**3,
        }
        write_json(directory / "result.json", result)
    del train, validation, test, ridge, lightgbm
    gc.collect()
    return result, mining_result


def run_two_fold_baselines(
    dataset_root: str | Path,
    *,
    output_root: str | Path,
    train_sample_basis_points: int = 500,
    evaluation_sample_basis_points: int = 1000,
) -> dict[str, Any]:
    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise MinuteV2Error("minute_v2_training_requires_four_gib_available_memory")
    results: list[dict[str, Any]] = []
    mining_result: dict[str, Any] | None = None
    for spec in TWO_FOLDS:
        result, mining_result = _run_fold(
            dataset,
            output,
            spec,
            train_sample_basis_points=train_sample_basis_points,
            evaluation_sample_basis_points=evaluation_sample_basis_points,
            mining_result=mining_result,
        )
        results.append(result)
    aggregate = {
        "schema": "quantlab.minute_v2_two_fold/1",
        "status": "ok",
        "dataset_root": str(dataset),
        "fold_count": len(results),
        "train_sample_basis_points": int(train_sample_basis_points),
        "evaluation_sample_basis_points": int(evaluation_sample_basis_points),
        "sampling_note": "sampling keeps whole minute cross-sections; it never samples individual stocks",
        "formula_mining": {
            "path": str(output / "formula_mining_fold_1.json"),
            "selected_count": int((mining_result or {}).get("selected_count", 0)),
            "used_final_fold": False,
        },
        "fold_results": [str(output / f"fold_{spec.fold}" / "result.json") for spec in TWO_FOLDS],
    }
    write_json(output / "result.json", aggregate)
    return aggregate


__all__ = ["FoldSpec", "TWO_FOLDS", "run_two_fold_baselines"]
