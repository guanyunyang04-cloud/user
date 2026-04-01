from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_rebalance_freq_step(rebalance_freq: str) -> int:
    freq = str(rebalance_freq or "1d").strip().lower()
    if not freq.endswith("d"):
        raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq}")
    step = int(freq[:-1] or "1")
    if step <= 0:
        raise ValueError(f"rebalance_freq must resolve to a positive trading-day step, got {rebalance_freq}")
    return step


def _resolve_anchor_position(index: pd.Index, rebalance_anchor_date: str | None) -> tuple[int, str]:
    if len(index) == 0:
        return 0, ""
    if rebalance_anchor_date in {None, ""}:
        return 0, str(pd.Timestamp(index[0]).date())

    anchor_ts = pd.Timestamp(rebalance_anchor_date)
    dt_index = pd.DatetimeIndex(index)
    pos = int(dt_index.searchsorted(anchor_ts, side="left"))
    if pos >= len(dt_index):
        raise ValueError(
            f"rebalance_anchor_date {anchor_ts.date()} is after the available trading calendar ending at {dt_index.max().date()}"
        )
    resolved = pd.Timestamp(dt_index[pos])
    return pos, str(resolved.date())


def apply_rebalance_frequency(
    frame: pd.DataFrame,
    rebalance_freq: str,
    rebalance_offset: int = 0,
    rebalance_anchor_date: str | None = None,
) -> pd.DataFrame:
    step = parse_rebalance_freq_step(rebalance_freq)
    out = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).copy()
    if step <= 1:
        return out
    offset = int(rebalance_offset or 0)
    if offset < 0:
        raise ValueError(f"rebalance_offset must be >= 0, got {rebalance_offset}")
    offset = offset % step
    anchor_pos, _ = _resolve_anchor_position(out.index, rebalance_anchor_date)
    positions = [idx for idx in range(len(out.index)) if ((idx - anchor_pos) % step) == offset]
    rebalance_idx = out.index[positions]
    rebalanced = out.loc[rebalance_idx].reindex(out.index).ffill()
    return rebalanced.fillna(0.0)


def apply_rebalance_schedule(
    frame: pd.DataFrame,
    *,
    rebalance_freq: str,
    rebalance_offset: int = 0,
    rebalance_offset_mode: str = "single",
    rebalance_anchor_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    step = parse_rebalance_freq_step(rebalance_freq)
    mode = str(rebalance_offset_mode or "single").strip().lower()
    if mode not in {"single", "all"}:
        raise ValueError(f"Unsupported rebalance_offset_mode: {rebalance_offset_mode}")

    base = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    normalized_offset = int(rebalance_offset or 0)
    if normalized_offset < 0:
        raise ValueError(f"rebalance_offset must be >= 0, got {rebalance_offset}")
    _, resolved_anchor_date = _resolve_anchor_position(base.index, rebalance_anchor_date)

    if step <= 1:
        return (
            base.copy(),
            {
                "rebalance_freq": f"{step}d",
                "rebalance_step": int(step),
                "rebalance_offset_mode": "single",
                "rebalance_offset": 0,
                "rebalance_offsets": [0],
                "rebalance_sleeve_count": 1,
                "rebalance_anchor_date": resolved_anchor_date,
            },
        )

    if mode == "single":
        normalized_offset = normalized_offset % step
        return (
            apply_rebalance_frequency(
                base,
                rebalance_freq,
                normalized_offset,
                rebalance_anchor_date=rebalance_anchor_date,
            ),
            {
                "rebalance_freq": str(rebalance_freq).strip().lower(),
                "rebalance_step": int(step),
                "rebalance_offset_mode": "single",
                "rebalance_offset": int(normalized_offset),
                "rebalance_offsets": [int(normalized_offset)],
                "rebalance_sleeve_count": 1,
                "rebalance_anchor_date": resolved_anchor_date,
            },
        )

    sleeves = [
        apply_rebalance_frequency(
            base,
            rebalance_freq,
            offset,
            rebalance_anchor_date=rebalance_anchor_date,
        )
        for offset in range(step)
    ]
    ensemble = sleeves[0].copy()
    for sleeve in sleeves[1:]:
        ensemble = ensemble.add(sleeve, fill_value=0.0)
    ensemble = (ensemble / float(step)).fillna(0.0)
    return (
        ensemble,
        {
            "rebalance_freq": str(rebalance_freq).strip().lower(),
            "rebalance_step": int(step),
            "rebalance_offset_mode": "all",
            "rebalance_offset": None,
            "rebalance_offsets": list(range(step)),
            "rebalance_sleeve_count": int(step),
            "rebalance_anchor_date": resolved_anchor_date,
        },
    )


def load_value_panel(
    path: Path,
    *,
    panel_format: str,
    date_column: str,
    stock_column: str,
    value_column: str,
    panel_label: str,
    value_name: str,
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{panel_label} CSV is empty: {path}")

    fmt = str(panel_format).lower()
    columns_lower = {str(col).strip().lower(): str(col) for col in raw.columns}
    date_key = str(date_column).strip().lower()
    stock_key = str(stock_column).strip().lower()
    value_key = str(value_column).strip().lower()

    if fmt == "auto":
        if date_key in columns_lower and stock_key in columns_lower and value_key in columns_lower:
            fmt = "long"
        else:
            fmt = "wide"

    if fmt == "long":
        missing = [name for name in (date_column, stock_column, value_column) if str(name).strip().lower() not in columns_lower]
        if missing:
            raise ValueError(f"Missing required columns for long {panel_label}: {missing}")
        date_col = columns_lower[date_key]
        stock_col = columns_lower[stock_key]
        value_col = columns_lower[value_key]
        panel = raw[[date_col, stock_col, value_col]].copy()
        panel.columns = ["date", "stock", value_name]
        panel["date"] = pd.to_datetime(panel["date"])
        panel["stock"] = panel["stock"].astype(str).str.upper().str.strip()
        panel[value_name] = pd.to_numeric(panel[value_name], errors="coerce")
        panel = panel[(panel["stock"] != "") & panel[value_name].notna()].copy()
        if panel.empty:
            raise ValueError(f"No usable rows found in {panel_label}: {path}")
        return (
            panel.sort_values(["date", "stock"])
            .drop_duplicates(subset=["date", "stock"], keep="last")
            .pivot(index="date", columns="stock", values=value_name)
            .sort_index()
        )

    if date_key not in columns_lower:
        raise ValueError(f"Wide {panel_label} requires a date column named {date_column}. Available: {list(raw.columns)}")
    date_col = columns_lower[date_key]
    panel = raw.copy()
    panel[date_col] = pd.to_datetime(panel[date_col])
    value_panel = panel.set_index(date_col)
    value_panel.columns = [str(col).upper().strip() for col in value_panel.columns]
    value_panel = value_panel.apply(pd.to_numeric, errors="coerce").sort_index()
    value_panel = value_panel.loc[:, [col for col in value_panel.columns if col]]
    if value_panel.empty:
        raise ValueError(f"No usable stock columns found in wide {panel_label}: {path}")
    return value_panel


def sanitize_target_weights(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    row_sums = clean.sum(axis=1)
    overweight = row_sums > 1.0 + 1e-8
    if bool(overweight.any()):
        clean.loc[overweight] = clean.loc[overweight].div(row_sums.loc[overweight], axis=0).fillna(0.0)
    return clean


def sanitize_target_weight_row(row: pd.Series) -> pd.Series:
    clean = row.fillna(0.0).astype(float).clip(lower=0.0)
    total = float(clean.sum())
    if total > 1.0 + 1e-8:
        clean = clean / total
    return clean


def transform_target_weights(
    frame: pd.DataFrame,
    *,
    top_k: int,
    min_weight: float,
    power: float,
    full_invest: bool,
) -> pd.DataFrame:
    clean = sanitize_target_weights(frame)
    top_k = max(int(top_k or 0), 0)
    min_weight = max(float(min_weight or 0.0), 0.0)
    power = max(float(power or 1.0), 1e-8)

    out_rows: list[pd.Series] = []
    for dt, row in clean.iterrows():
        weights = row.fillna(0.0).astype(float).clip(lower=0.0)
        positive = weights[weights > 0.0].copy()
        if positive.empty:
            out_rows.append(pd.Series(0.0, index=clean.columns, name=dt, dtype=float))
            continue

        if min_weight > 0.0:
            positive = positive[positive >= min_weight]
        if positive.empty:
            out_rows.append(pd.Series(0.0, index=clean.columns, name=dt, dtype=float))
            continue

        positive = positive.sort_values(ascending=False)
        if top_k > 0:
            positive = positive.iloc[:top_k]
        if positive.empty:
            out_rows.append(pd.Series(0.0, index=clean.columns, name=dt, dtype=float))
            continue

        if abs(power - 1.0) > 1e-12:
            positive = positive.pow(power)

        total = float(positive.sum())
        if total <= 0.0:
            out_rows.append(pd.Series(0.0, index=clean.columns, name=dt, dtype=float))
            continue

        if full_invest or top_k > 0 or min_weight > 0.0 or abs(power - 1.0) > 1e-12:
            positive = positive / total

        aligned = pd.Series(0.0, index=clean.columns, name=dt, dtype=float)
        aligned.loc[positive.index] = positive.values
        out_rows.append(aligned)

    transformed = pd.DataFrame(out_rows, index=clean.index, columns=clean.columns).fillna(0.0)
    return sanitize_target_weights(transformed)


def build_target_weight_bridge(
    frame: pd.DataFrame,
    *,
    rebalance_freq: str,
    rebalance_offset: int = 0,
    rebalance_offset_mode: str = "single",
    rebalance_anchor_date: str | None = None,
    top_k: int = 0,
    min_weight: float = 0.0,
    power: float = 1.0,
    full_invest: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    transformed = transform_target_weights(
        frame,
        top_k=top_k,
        min_weight=min_weight,
        power=power,
        full_invest=full_invest,
    )
    bridged, schedule_meta = apply_rebalance_schedule(
        transformed,
        rebalance_freq=rebalance_freq,
        rebalance_offset=rebalance_offset,
        rebalance_offset_mode=rebalance_offset_mode,
        rebalance_anchor_date=rebalance_anchor_date,
    )
    bridged = sanitize_target_weights(bridged)
    meta = dict(schedule_meta)
    meta.update(
        {
            "target_weight_top_k": int(top_k or 0),
            "target_weight_min_weight": float(min_weight or 0.0),
            "target_weight_power": float(power or 1.0),
            "target_weight_full_invest": bool(full_invest),
        }
    )
    return bridged, meta
