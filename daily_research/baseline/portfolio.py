import pandas as pd

from daily_research.baseline.config import ResearchConfig


def _industry_name(stock: str, industry_map: pd.Series | None) -> str:
    if industry_map is None:
        return "__all__"
    return str(industry_map.get(stock, "__unknown__"))


def _stock_styles(stock: str, style_map: pd.DataFrame | None) -> list[str]:
    if style_map is None or stock not in style_map.index:
        return []
    row = style_map.loc[stock]
    return [style for style, flag in row.items() if bool(flag)]


def _apply_weight_cap(weights: pd.Series, max_weight: float) -> pd.Series:
    if weights.empty:
        return weights
    max_weight = float(max_weight)
    if max_weight <= 0:
        return weights * 0.0

    remaining = weights.clip(lower=0.0).copy()
    capped = pd.Series(0.0, index=weights.index, dtype=float)
    free = remaining.index.tolist()
    budget = 1.0

    while free and budget > 1e-12:
        free_weights = remaining.loc[free]
        total = float(free_weights.sum())
        if total <= 0:
            break
        scaled = free_weights / total * budget
        hit_cap = scaled >= max_weight - 1e-12
        if not hit_cap.any():
            capped.loc[free] = scaled
            budget = 0.0
            break
        capped_names = scaled.index[hit_cap]
        capped.loc[capped_names] = max_weight
        budget -= max_weight * len(capped_names)
        free = [name for name in free if name not in set(capped_names)]
        remaining.loc[capped_names] = 0.0

    if budget > 1e-12 and free:
        free_weights = remaining.loc[free]
        total = float(free_weights.sum())
        if total > 0:
            capped.loc[free] += free_weights / total * budget

    return capped.clip(lower=0.0)


def _select_with_exposure_caps(
    ranked_scores: pd.Series,
    config: ResearchConfig,
    industry_map: pd.Series | None,
    style_map: pd.DataFrame | None,
) -> pd.Series:
    candidate_count = max(config.holding_count * max(int(config.industry_candidate_buffer), 1), config.holding_count)
    ranked_pool = ranked_scores.iloc[:candidate_count]
    approx_name_weight = min(float(config.max_weight), 1.0 / max(int(config.holding_count), 1))
    max_names_per_industry = max(1, int(float(config.max_industry_weight) / max(approx_name_weight, 1e-8)))
    max_names_per_style = max(1, int(float(config.max_style_weight) / max(approx_name_weight, 1e-8)))

    selected = []
    industry_counts: dict[str, int] = {}
    style_counts: dict[str, int] = {}

    for stock in ranked_pool.index:
        industry_ok = True
        if config.enable_industry_cap and industry_map is not None:
            industry = _industry_name(stock, industry_map)
            if industry_counts.get(industry, 0) >= max_names_per_industry:
                industry_ok = False
        if not industry_ok:
            continue

        style_ok = True
        if config.enable_style_cap and style_map is not None:
            for style in _stock_styles(stock, style_map):
                if style_counts.get(style, 0) >= max_names_per_style:
                    style_ok = False
                    break
        if not style_ok:
            continue

        selected.append(stock)
        if config.enable_industry_cap and industry_map is not None:
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if config.enable_style_cap and style_map is not None:
            for style in _stock_styles(stock, style_map):
                style_counts[style] = style_counts.get(style, 0) + 1
        if len(selected) >= config.holding_count:
            break

    if not selected:
        return ranked_scores.iloc[: config.holding_count]
    return ranked_scores.reindex(selected).dropna()


def _current_style_weight(style: str, weights: pd.Series, style_map: pd.DataFrame | None) -> float:
    if style_map is None or weights.empty:
        return 0.0
    members = [stock for stock in weights.index if stock in style_map.index and bool(style_map.at[stock, style])]
    if not members:
        return 0.0
    return float(weights.reindex(members).fillna(0.0).sum())


def _enforce_exposure_caps(
    weights: pd.Series,
    scores: pd.Series,
    config: ResearchConfig,
    industry_map: pd.Series | None,
    style_map: pd.DataFrame | None,
) -> pd.Series:
    if weights.empty:
        return weights

    weights = weights.copy()
    priorities = scores.reindex(weights.index).fillna(0.0).sort_values(ascending=False)
    max_industry_weight = float(config.max_industry_weight)
    max_style_weight = float(config.max_style_weight)
    max_weight = float(config.max_weight)

    for _ in range(5):
        changed = False
        if config.enable_industry_cap and industry_map is not None:
            industry_labels = pd.Series({_stock: _industry_name(_stock, industry_map) for _stock in weights.index})
            industry_sums = weights.groupby(industry_labels).sum()
            violations = industry_sums[industry_sums > max_industry_weight + 1e-12]
            for industry, total_weight in violations.items():
                members = industry_labels[industry_labels == industry].index
                weights.loc[members] *= max_industry_weight / float(total_weight)
                changed = True
        if config.enable_style_cap and style_map is not None and not style_map.empty:
            for style in style_map.columns:
                members = [stock for stock in weights.index if stock in style_map.index and bool(style_map.at[stock, style])]
                if not members:
                    continue
                style_total = float(weights.reindex(members).fillna(0.0).sum())
                if style_total > max_style_weight + 1e-12:
                    weights.loc[members] *= max_style_weight / style_total
                    changed = True
        if not changed:
            break

    residual = max(0.0, 1.0 - float(weights.sum()))
    if residual <= 1e-12:
        return weights

    for stock in priorities.index:
        stock_slack = max(0.0, max_weight - float(weights.get(stock, 0.0)))
        if stock_slack <= 1e-12:
            continue

        add_limit = stock_slack
        if config.enable_industry_cap and industry_map is not None:
            industry = _industry_name(stock, industry_map)
            industry_total = float(weights.loc[[s for s in weights.index if _industry_name(s, industry_map) == industry]].sum())
            add_limit = min(add_limit, max(0.0, max_industry_weight - industry_total))
        if config.enable_style_cap and style_map is not None:
            for style in _stock_styles(stock, style_map):
                add_limit = min(add_limit, max(0.0, max_style_weight - _current_style_weight(style, weights, style_map)))

        add_weight = min(residual, add_limit)
        if add_weight <= 1e-12:
            continue
        weights.loc[stock] = float(weights.get(stock, 0.0)) + add_weight
        residual -= add_weight
        if residual <= 1e-12:
            break

    return weights


def build_target_weights(
    scores: pd.DataFrame,
    config: ResearchConfig,
    industry_map: pd.Series | None = None,
    style_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    targets = []
    for dt, row in scores.iterrows():
        s = row.dropna()
        s = s[s > float(config.score_threshold)]
        if s.empty:
            targets.append(pd.Series(dtype=float, name=dt))
            continue

        s = s.sort_values(ascending=False)
        s = _select_with_exposure_caps(s, config, industry_map, style_map)

        if config.weighting_method == "score":
            min_v = s.min()
            adj = s - min_v + 1e-6
            w = adj / adj.sum()
        else:
            w = pd.Series(1.0 / len(s), index=s.index, name=dt)

        w = _apply_weight_cap(w, config.max_weight)
        w = _enforce_exposure_caps(w, s, config, industry_map, style_map)
        w.name = dt
        targets.append(w)

    weights = pd.DataFrame(targets).fillna(0.0)
    weights.index = scores.index
    return weights
