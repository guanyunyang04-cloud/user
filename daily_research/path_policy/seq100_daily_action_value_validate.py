"""Independent validation for the paired action-value study."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_daily_action_value as action

VALIDATION_SCHEMA = "seq100_daily_action_value_validation/1"


def _resolve(path: str | Path) -> Path:
    return action._resolve(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    return action._read_json(path)


def _sample_indices(length: int, maximum: int = 2000) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.unique(
        np.concatenate(
            [
                np.arange(0, maximum // 2, dtype=np.int64),
                np.linspace(length // 2, length - 1, maximum // 2, dtype=np.int64),
            ]
        )
    )


def validate(
    *,
    output_root: str | Path = action.DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    root = _resolve(output_root)
    study_path = action.DEFAULT_STUDY_PATH
    study = action.load_study(study_path)
    source = action._source_contract(study)
    material = action._open_material(study, source)
    outcome_manifest = _read_json(root / "outcome_manifest.json")
    prediction_manifest = _read_json(root / "prediction_manifest.json")
    if outcome_manifest.get("status") != "completed":
        raise ValueError("action_value_validation_outcomes_not_completed")
    if prediction_manifest.get("status") != "completed":
        raise ValueError("action_value_validation_predictions_not_completed")
    outcome_frames = [
        pd.read_parquet(_resolve(item["path"])) for item in outcome_manifest["years"]
    ]
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    prediction_frames = [
        pd.read_parquet(_resolve(item["path"])) for item in prediction_manifest["years"]
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_outcomes = int(outcome_manifest["rows"])
    expected_predictions = int(prediction_manifest["rows"])
    if len(outcomes) != expected_outcomes:
        raise ValueError("action_value_validation_outcome_count_mismatch")
    if len(predictions) != expected_predictions:
        raise ValueError("action_value_validation_prediction_count_mismatch")
    key_columns = ["reference_name", "reference_trade_id", "date_idx"]
    duplicate_outcomes = int(outcomes.duplicated(key_columns).sum())
    duplicate_predictions = int(
        predictions.duplicated(key_columns).sum()
    )
    if duplicate_outcomes or duplicate_predictions:
        raise ValueError(
            f"action_value_validation_duplicate_keys:{duplicate_outcomes}:{duplicate_predictions}"
        )
    cutoff = str(study["source"]["maximum_outcome_date"])
    forbidden_rows = int((outcomes["trade_date"].astype(str) > cutoff).sum())
    forbidden_rows += int(
        (predictions["trade_date"].astype(str) > cutoff).sum()
    )
    if forbidden_rows:
        raise ValueError(f"action_value_validation_forbidden_rows:{forbidden_rows}")
    resolution_mismatch = int(
        (outcomes["resolution_date_idx"].astype(int) != outcomes["date_idx"].astype(int) + 2).sum()
    )
    if resolution_mismatch:
        raise ValueError("action_value_validation_resolution_mismatch")
    causal_violations = int(predictions["causal_update_violation"].astype(bool).sum())
    causal_violations += int(
        (
            predictions["maximum_training_resolution_date_idx"].astype(int)
            >= predictions["date_idx"].astype(int)
        ).sum()
    )
    if causal_violations:
        raise ValueError(f"action_value_validation_causal_violations:{causal_violations}")
    blocked = outcomes["blocked_action_identity_base"].astype(bool)
    blocked_difference = float(
        np.abs(outcomes.loc[blocked, "delta_log_value_base"].astype(float)).max()
        if bool(blocked.any())
        else 0.0
    )
    if blocked_difference > 1e-10:
        raise ValueError("action_value_validation_blocked_identity_failed")
    prediction_identity_violations = 0
    for cost in action.COST_SCENARIOS:
        for model in action.MODEL_NAMES:
            # Prediction rows are a strict subset of outcomes but retain the
            # same key; align before checking the identity.
            merged = predictions[key_columns + [f"policy_gain_{model}_{cost}", f"policy_replace_{model}_{cost}"]].merge(
                outcomes[
                    key_columns
                    + [
                        f"delta_log_value_{cost}",
                        f"action_valid_{cost}",
                        f"blocked_action_identity_{cost}",
                    ]
                ],
                on=key_columns,
                how="left",
                validate="one_to_one",
            )
            expected = np.where(
                merged[f"policy_replace_{model}_{cost}"].astype(bool)
                & merged[f"action_valid_{cost}"].astype(bool)
                & ~merged[f"blocked_action_identity_{cost}"].astype(bool),
                merged[f"delta_log_value_{cost}"].astype(float),
                0.0,
            )
            difference = float(
                np.abs(
                    merged[f"policy_gain_{model}_{cost}"].astype(float).to_numpy()
                    - expected
                ).max()
            )
            if difference > 1e-7:
                prediction_identity_violations += 1
    if prediction_identity_violations:
        raise ValueError(
            f"action_value_validation_policy_identity:{prediction_identity_violations}"
        )
    sample = outcomes.iloc[_sample_indices(len(outcomes))]
    maximum_delta_difference = {cost: 0.0 for cost in action.COST_SCENARIOS}
    for row in sample.itertuples(index=False):
        for cost in action.COST_SCENARIOS:
            recomputed = action._paired_action_wealth(
                material=material,
                date_idx=int(row.date_idx),
                current_symbol_idx=int(row.symbol_idx),
                current_shares=int(row.shares),
                replacement_symbol_idx=int(row.candidate_symbol_idx),
                cost_scenario=cost,
            )
            if bool(getattr(row, f"action_valid_{cost}")):
                difference = abs(
                    float(recomputed["delta_log_value"])
                    - float(getattr(row, f"delta_log_value_{cost}"))
                )
                maximum_delta_difference[cost] = max(
                    maximum_delta_difference[cost], difference
                )
    if max(maximum_delta_difference.values()) > 1e-7:
        raise ValueError(
            f"action_value_validation_recompute_difference:{maximum_delta_difference}"
        )
    audit = {
        "outcome_rows": len(outcomes),
        "prediction_rows": len(predictions),
        "duplicate_outcome_keys": duplicate_outcomes,
        "duplicate_prediction_keys": duplicate_predictions,
        "forbidden_rows": forbidden_rows,
        "resolution_mismatches": resolution_mismatch,
        "causal_update_violations": causal_violations,
        "blocked_identity_rows": int(blocked.sum()),
        "maximum_blocked_identity_difference": blocked_difference,
        "policy_identity_violations": prediction_identity_violations,
        "sample_rows": len(sample),
        "maximum_delta_difference": maximum_delta_difference,
        "minimum_candidate_rank": int(outcomes["candidate_rank"].min()),
        "maximum_candidate_rank": int(outcomes["candidate_rank"].max()),
        "unknown_state_rows": int(
            (outcomes["state_group"].astype(int) == action.OBSERVABLE_STATE_GROUP_COUNT - 1).sum()
        ),
    }
    output = {
        "schema": VALIDATION_SCHEMA,
        "status": "completed",
        "study_id": action.STUDY_ID,
        "study": action._file_record(study_path),
        "outcome_manifest": action._file_record(root / "outcome_manifest.json"),
        "prediction_manifest": action._file_record(root / "prediction_manifest.json"),
        "audit": audit,
        "profit_claim_allowed": False,
        "production_policy_selected": False,
        "report_generation_performed": False,
    }
    action._write_json(root / "validation_manifest.json", output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(action.DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = validate(output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
