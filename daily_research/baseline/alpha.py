from typing import Dict, Tuple

import numpy as np
import pandas as pd

from daily_research.baseline.config import ResearchConfig


def _clip_cross_section(df: pd.DataFrame, clip_value: float) -> pd.DataFrame:
    return df.clip(lower=-abs(float(clip_value)), upper=abs(float(clip_value)))


def build_filter_mask(factor_bundle: Dict[str, Dict[str, pd.DataFrame]], config: ResearchConfig) -> pd.DataFrame:
    raw_inputs = factor_bundle["raw_inputs"]
    close = raw_inputs["Close"]
    amount = raw_inputs["Amount"]
    adv20 = amount.rolling(20).mean()

    mask = (
        close.notna()
        & amount.notna()
        & (close >= float(config.min_price))
        & (close <= float(config.max_price))
        & (adv20 >= float(config.min_adv20))
    )
    return mask


def combine_scores(
    factor_bundle: Dict[str, Dict[str, pd.DataFrame]],
    config: ResearchConfig,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    zscore_factors = factor_bundle["zscore_factors"]
    clipped_factors = {
        name: _clip_cross_section(value, config.score_clip)
        for name, value in zscore_factors.items()
    }

    group_scores: Dict[str, pd.DataFrame] = {}
    for group_name, factor_names in config.factor_groups.items():
        weighted_sum = None
        total_weight = 0.0
        for factor_name in factor_names:
            if factor_name not in clipped_factors:
                continue
            factor_weight = float(config.factor_weights.get(factor_name, 0.0))
            if factor_weight == 0.0:
                continue
            term = clipped_factors[factor_name] * factor_weight
            weighted_sum = term if weighted_sum is None else weighted_sum.add(term, fill_value=0.0)
            total_weight += abs(factor_weight)
        if weighted_sum is None or total_weight == 0:
            continue
        group_scores[group_name] = weighted_sum / total_weight

    score = None
    group_weight_sum = 0.0
    for group_name, group_df in group_scores.items():
        group_weight = float(config.factor_group_weights.get(group_name, 0.0))
        if group_weight == 0.0:
            continue
        term = group_df * group_weight
        score = term if score is None else score.add(term, fill_value=0.0)
        group_weight_sum += abs(group_weight)

    if score is None or group_weight_sum == 0.0:
        raise ValueError("No valid grouped factors to combine.")

    score = score / group_weight_sum
    filter_mask = build_filter_mask(factor_bundle, config)
    score = score.where(filter_mask)
    return score, group_scores, filter_mask


def combine_scores_by_state(
    factor_bundle: Dict[str, Dict[str, pd.DataFrame]],
    config: ResearchConfig,
    quadrant_series: pd.Series | None,
    state_configs: Dict[str, ResearchConfig] | None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    if quadrant_series is None or not state_configs:
        return combine_scores(factor_bundle, config)

    base_score, base_group_scores, base_filter_mask = combine_scores(factor_bundle, config)
    score = base_score.copy()
    filter_mask = base_filter_mask.copy()
    group_scores = {name: df.copy() for name, df in base_group_scores.items()}
    cached_results: Dict[str, Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]] = {}

    quadrant_series = quadrant_series.reindex(score.index)
    for quadrant_name, state_cfg in state_configs.items():
        row_mask = quadrant_series.eq(quadrant_name).fillna(False)
        if not bool(row_mask.any()):
            continue
        if quadrant_name not in cached_results:
            cached_results[quadrant_name] = combine_scores(factor_bundle, state_cfg)
        state_score, state_group_scores, state_filter_mask = cached_results[quadrant_name]

        score.loc[row_mask] = state_score.loc[row_mask]
        filter_mask.loc[row_mask] = state_filter_mask.loc[row_mask]

        all_group_names = set(group_scores) | set(state_group_scores)
        for group_name in all_group_names:
            if group_name not in group_scores:
                group_scores[group_name] = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
            if group_name not in state_group_scores:
                continue
            group_scores[group_name].loc[row_mask] = state_group_scores[group_name].loc[row_mask]

    return score, group_scores, filter_mask
