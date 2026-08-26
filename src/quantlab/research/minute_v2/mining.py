"""Bounded, auditable formula-feature discovery on development data only."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import MinuteV2Error


@dataclass(frozen=True)
class FormulaFeature:
    name: str
    operator: str
    left: str
    right: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_MINING_SEEDS = (
    "return_1m",
    "return_5m",
    "return_15m",
    "return_from_previous_close",
    "vwap_deviation",
    "volume_acceleration_5_20",
    "amount_curve_surprise",
    "trend_efficiency",
    "drawdown_from_day_high",
    "rebound_from_day_low",
    "auction_gap",
    "previous_return_5d",
    "previous_volatility_20d",
    "market_breadth_positive",
    "industry_strength_rank",
    "market_residual_return",
    "industry_residual_return",
    "industry_residual_return_5m",
)


def generate_formulas(
    seed_features: Sequence[str] = DEFAULT_MINING_SEEDS,
    *,
    maximum_pair_formulas: int = 180,
) -> list[FormulaFeature]:
    seeds = tuple(dict.fromkeys(str(value) for value in seed_features))
    formulas: list[FormulaFeature] = []
    for name in seeds:
        formulas.extend(
            [
                FormulaFeature(f"abs__{name}", "abs", name),
                FormulaFeature(f"signed_square__{name}", "signed_square", name),
                FormulaFeature(f"log_abs__{name}", "log_abs", name),
            ]
        )
    pair_count = 0
    for left, right in combinations(seeds, 2):
        for operator, prefix in (
            ("difference", "diff"),
            ("product", "product"),
            ("safe_ratio", "ratio"),
        ):
            if pair_count >= int(maximum_pair_formulas):
                return formulas
            formulas.append(FormulaFeature(f"{prefix}__{left}__{right}", operator, left, right))
            pair_count += 1
    return formulas


def apply_formula(frame: pd.DataFrame, formula: FormulaFeature) -> pd.Series:
    if formula.left not in frame:
        raise MinuteV2Error(f"minute_v2_formula_input_missing:{formula.left}")
    left = pd.to_numeric(frame[formula.left], errors="coerce").astype(float)
    if formula.operator == "abs":
        result = left.abs()
    elif formula.operator == "signed_square":
        result = left * left.abs()
    elif formula.operator == "log_abs":
        result = np.sign(left) * np.log1p(left.abs())
    else:
        if formula.right not in frame:
            raise MinuteV2Error(f"minute_v2_formula_input_missing:{formula.right}")
        right = pd.to_numeric(frame[formula.right], errors="coerce").astype(float)
        if formula.operator == "difference":
            result = left - right
        elif formula.operator == "product":
            result = left * right
        elif formula.operator == "safe_ratio":
            result = left / (right.abs() + 1.0e-6)
        else:
            raise MinuteV2Error(f"minute_v2_formula_operator_unknown:{formula.operator}")
    return pd.Series(result, index=frame.index, name=formula.name).replace([np.inf, -np.inf], np.nan)


def _grouped_ic(frame: pd.DataFrame, values: pd.Series, target: str) -> pd.Series:
    working = pd.DataFrame(
        {
            "trade_date": frame["trade_date"].astype(str),
            "bar_time": frame["bar_time"].astype(str),
            "value": values,
            "target": pd.to_numeric(frame[target], errors="coerce"),
        }
    ).dropna(subset=["value", "target"])
    if working.empty:
        return pd.Series(dtype=float)
    sizes = working.groupby(["trade_date", "bar_time"], sort=False)["value"].transform("size")
    working = working.loc[sizes >= 5].copy()
    if working.empty:
        return pd.Series(dtype=float)
    working["value_rank"] = working.groupby(["trade_date", "bar_time"], sort=False)["value"].rank()
    working["target_rank"] = working.groupby(["trade_date", "bar_time"], sort=False)["target"].rank()
    def correlation(group: pd.DataFrame) -> float:
        if group["value_rank"].nunique() < 2 or group["target_rank"].nunique() < 2:
            return float("nan")
        return float(group["value_rank"].corr(group["target_rank"]))

    result = working.groupby(["trade_date", "bar_time"], sort=True).apply(
        correlation,
        include_groups=False,
    )
    return result.dropna().astype(float)


def _formula_metrics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    formula: FormulaFeature,
    *,
    target: str,
) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    train_value = apply_formula(train, formula)
    valid_value = apply_formula(validation, formula)
    train_ic = _grouped_ic(train, train_value, target)
    valid_ic = _grouped_ic(validation, valid_value, target)
    train_mean = float(train_ic.mean()) if len(train_ic) else float("nan")
    valid_mean = float(valid_ic.mean()) if len(valid_ic) else float("nan")
    same_sign = bool(np.isfinite(train_mean) and np.isfinite(valid_mean) and train_mean * valid_mean > 0.0)
    valid_year = (
        valid_ic.groupby(valid_ic.index.get_level_values(0).astype(str).str[:4]).mean()
        if len(valid_ic)
        else pd.Series(dtype=float)
    )
    sign = np.sign(train_mean) if np.isfinite(train_mean) else 0.0
    stable_fraction = (
        float((np.sign(valid_year) == sign).mean()) if len(valid_year) and sign != 0.0 else 0.0
    )
    coverage = float(valid_value.notna().mean()) if len(valid_value) else 0.0
    score = min(abs(train_mean), abs(valid_mean)) * stable_fraction if same_sign else 0.0
    return (
        {
            **formula.as_dict(),
            "train_group_count": int(len(train_ic)),
            "validation_group_count": int(len(valid_ic)),
            "train_rank_ic_mean": train_mean if np.isfinite(train_mean) else None,
            "validation_rank_ic_mean": valid_mean if np.isfinite(valid_mean) else None,
            "validation_coverage": coverage,
            "validation_year_sign_stability": stable_fraction,
            "same_sign": same_sign,
            "selection_score": float(score),
        },
        train_value,
        valid_value,
    )


def mine_formula_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    target: str = "label_net_return",
    seed_features: Sequence[str] = DEFAULT_MINING_SEEDS,
    maximum_candidates: int = 240,
    maximum_selected: int = 20,
    minimum_coverage: float = 0.90,
    maximum_redundancy: float = 0.95,
) -> dict[str, Any]:
    required = {"trade_date", "bar_time", target, *seed_features}
    for label, frame in (("train", train), ("validation", validation)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise MinuteV2Error(f"minute_v2_mining_{label}_columns_missing:{','.join(missing)}")
    candidates = generate_formulas(seed_features)[: int(maximum_candidates)]
    evaluated: list[tuple[dict[str, Any], FormulaFeature, pd.Series]] = []
    for formula in candidates:
        metrics, _, valid_values = _formula_metrics(
            train,
            validation,
            formula,
            target=target,
        )
        evaluated.append((metrics, formula, valid_values))
    evaluated.sort(key=lambda item: float(item[0]["selection_score"]), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_values: list[pd.Series] = []
    for metrics, _formula, values in evaluated:
        if len(selected) >= int(maximum_selected):
            break
        if not metrics["same_sign"] or float(metrics["validation_coverage"]) < float(minimum_coverage):
            continue
        redundant = False
        for accepted in selected_values:
            paired = pd.concat([values, accepted], axis=1).dropna()
            if len(paired) < 100:
                continue
            left_rank = paired.iloc[:, 0].rank()
            right_rank = paired.iloc[:, 1].rank()
            if left_rank.nunique() < 2 or right_rank.nunique() < 2:
                continue
            correlation = left_rank.corr(right_rank)
            if np.isfinite(correlation) and abs(float(correlation)) >= float(maximum_redundancy):
                redundant = True
                break
        metrics["redundant_with_selected"] = redundant
        if redundant:
            continue
        selected.append(metrics)
        selected_values.append(values)
    return {
        "schema": "quantlab.minute_v2_formula_mining/1",
        "target": target,
        "seed_features": list(seed_features),
        "candidate_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "selection_uses": "development train and validation only; no final-fold data",
        "selected": selected,
        "all_candidates": [metrics for metrics, _, _ in evaluated],
    }


def write_mining_result(result: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "DEFAULT_MINING_SEEDS",
    "FormulaFeature",
    "apply_formula",
    "generate_formulas",
    "mine_formula_features",
    "write_mining_result",
]
