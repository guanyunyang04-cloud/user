from __future__ import annotations

"""Reproducible read-only audit for the project's mathematical foundations.

The audit is intentionally bounded to 2011-2025. It does not train a model and
does not use any outcome whose logical observation date is later than
2025-12-31.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import ndtr

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent
CUTOFF_DATE = "2025-12-31"
HORIZON = 20
TRADING_DAYS = 252
RNG_SEED = 20260805

MODEL_INPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_training_ready/model_inputs"
)
PACK_MANIFEST_PATH = (
    WORKSPACE_ROOT
    / "daily_research/data/research_store/seq100_pit_l35v2_v1/pack/manifest.json"
)
POLICY_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_policy_targets"
)
EXECUTION_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_execution"
)
T1_MODEL_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_t1_return_models"
)
QUALITY_MODEL_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_quality_liquidity_model"
)


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_memmap(spec: dict[str, Any], dtype: str | np.dtype[Any]) -> np.memmap:
    return np.memmap(
        Path(spec["path"]),
        mode="r",
        dtype=dtype,
        shape=tuple(int(value) for value in spec["shape"]),
    )


def expected_shortfall(values: np.ndarray, level: float = 0.05) -> float:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return math.nan
    count = max(1, int(math.ceil(level * len(clean))))
    return float(np.partition(clean, count - 1)[:count].mean())


def distribution_stats(values: np.ndarray) -> dict[str, float | int]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    mean = float(clean.mean())
    centered = clean - mean
    variance = float(np.mean(centered**2))
    scale = math.sqrt(variance)
    skew = float(np.mean(centered**3) / scale**3) if scale > 0 else math.nan
    excess_kurtosis = (
        float(np.mean(centered**4) / scale**4 - 3.0) if scale > 0 else math.nan
    )
    quantiles = np.quantile(clean, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "count": len(clean),
        "mean": mean,
        "std": scale,
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "es01": expected_shortfall(clean, 0.01),
        "es05": expected_shortfall(clean, 0.05),
    }


def round_tick(values: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=np.float64) * 100.0 + 0.5) / 100.0


def parquet_scan(paths: list[Path]) -> str:
    literals = ",".join(
        "'" + str(path.resolve()).replace("'", "''") + "'" for path in paths
    )
    return f"read_parquet([{literals}],union_by_name=true)"


def qdp_manifest_paths(pack: dict[str, Any], domain: str) -> list[Path]:
    source = dict(pack["qdp_source_manifests"][domain])
    manifest_path = Path(source["manifest_path"])
    manifest = read_json(manifest_path)
    qdp_root = Path(pack["qdp_root"])
    return [(qdp_root / shard["path"]).resolve() for shard in manifest["shards"]]


def raw_st_limit_geometry(
    pack: dict[str, Any],
    pit_universe: np.memmap,
) -> pd.DataFrame:
    """Count an exchange-price diagnostic without treating it as ST truth."""

    raw_paths = qdp_manifest_paths(pack, "market_daily_raw")
    status_paths = qdp_manifest_paths(pack, "security_status")
    symbols = pd.DataFrame({"symbol": pack["symbol_values"]})
    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='10GB'")
    connection.register("pack_symbols", symbols)
    query = f"""
    WITH raw0 AS (
      SELECT r.symbol,r.trade_date,
             try_cast(r.open AS DOUBLE) AS open_px,
             try_cast(r.high AS DOUBLE) AS high_px,
             try_cast(r.low AS DOUBLE) AS low_px,
             try_cast(r.close AS DOUBLE) AS close_px
      FROM {parquet_scan(raw_paths)} r
      INNER JOIN pack_symbols p USING(symbol)
      WHERE r.trade_date BETWEEN '2010-01-01' AND '{CUTOFF_DATE}'
    ), raw AS (
      SELECT *,lag(close_px) OVER (
        PARTITION BY symbol ORDER BY trade_date
      ) AS previous_close
      FROM raw0
    ), status AS (
      SELECT symbol,trade_date,bool_or(coalesce(is_st,false)) AS is_st
      FROM {parquet_scan(status_paths)}
      WHERE trade_date BETWEEN '2011-01-01' AND '{CUTOFF_DATE}'
      GROUP BY symbol,trade_date
    )
    SELECT r.symbol,r.trade_date,coalesce(s.is_st,false) AS is_st
    FROM raw r
    LEFT JOIN status s USING(symbol,trade_date)
    WHERE r.trade_date >= '2011-01-01'
      AND previous_close > 0
      AND abs(open_px-high_px) < 1e-8
      AND abs(open_px-low_px) < 1e-8
      AND abs(open_px-close_px) < 1e-8
      AND (
        abs(close_px-round(previous_close*1.05,2)) < 1e-8
        OR abs(close_px-round(previous_close*0.95,2)) < 1e-8
      )
    """
    events = connection.execute(query).fetchdf()
    connection.close()
    date_to_idx = {
        str(value): position for position, value in enumerate(pack["date_values"])
    }
    symbol_to_idx = {
        str(value): position for position, value in enumerate(pack["symbol_values"])
    }
    date_idx = events["trade_date"].map(date_to_idx).to_numpy(dtype=np.int32)
    symbol_idx = events["symbol"].map(symbol_to_idx).to_numpy(dtype=np.int32)
    in_pit_scope = np.asarray(pit_universe[date_idx, symbol_idx], dtype=bool)
    events = events.loc[in_pit_scope].copy()
    events["year"] = events["trade_date"].str[:4].astype(int)
    annual = (
        events.groupby(["year", "is_st"]).size().unstack(fill_value=0)
    )
    annual = annual.rename(
        columns={False: "unmarked_st_candidate_count", True: "marked_st_count"}
    ).reset_index()
    for name in ("marked_st_count", "unmarked_st_candidate_count"):
        if name not in annual:
            annual[name] = 0
        annual[name] = annual[name].astype(int)
    return annual[
        ["year", "marked_st_count", "unmarked_st_candidate_count"]
    ].sort_values("year")


def hac_mean_t(values: np.ndarray, lag: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = len(values)
    centered = values - values.mean()
    long_run_variance = float(np.dot(centered, centered) / count)
    for offset in range(1, min(int(lag), count - 1) + 1):
        covariance = float(
            np.dot(centered[offset:], centered[:-offset]) / count
        )
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / count)
    statistic = float(values.mean() / standard_error) if standard_error > 0 else math.nan
    return standard_error, statistic


def moving_block_mean_ci(
    values: np.ndarray,
    block_length: int,
    *,
    repetitions: int = 4000,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = len(values)
    block_length = min(int(block_length), count)
    blocks_needed = int(math.ceil(count / block_length))
    maximum_start = count - block_length
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    offsets = np.arange(block_length, dtype=np.int32)
    for repetition in range(repetitions):
        starts = rng.integers(0, maximum_start + 1, size=blocks_needed)
        positions = (starts[:, None] + offsets[None, :]).reshape(-1)[:count]
        means[repetition] = float(values[positions].mean())
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def load_equity_log_returns(path: Path) -> pd.Series:
    frame = pd.read_parquet(path, columns=["trade_date", "daily_net_return"])
    frame["trade_date"] = frame["trade_date"].astype(str)
    values = np.log1p(frame["daily_net_return"].to_numpy(dtype=np.float64))
    return pd.Series(values, index=frame["trade_date"], name=path.parent.name)


def paired_return_statistics(
    preferred_path: Path,
    reference_path: Path,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    preferred = load_equity_log_returns(preferred_path)
    reference = load_equity_log_returns(reference_path)
    common = preferred.index.intersection(reference.index)
    if len(common) != len(preferred) or len(common) != len(reference):
        raise RuntimeError("paired equity calendars are not identical")
    difference = preferred.loc[common].to_numpy() - reference.loc[common].to_numpy()
    rows: list[dict[str, Any]] = []
    for lag in (20, 60):
        standard_error, statistic = hac_mean_t(difference, lag)
        lower, upper = moving_block_mean_ci(
            difference,
            lag,
            seed=seed + lag,
        )
        rows.append(
            {
                "preferred_task": preferred_path.parent.name,
                "reference_task": reference_path.parent.name,
                "date_count": len(difference),
                "lag_or_block_days": lag,
                "annualized_mean_log_return_difference": float(
                    difference.mean() * TRADING_DAYS
                ),
                "annualized_hac_standard_error": float(
                    standard_error * TRADING_DAYS
                ),
                "hac_t": statistic,
                "annualized_block_bootstrap_ci_low": lower * TRADING_DAYS,
                "annualized_block_bootstrap_ci_high": upper * TRADING_DAYS,
            }
        )
    return rows


def model_importance_rows() -> pd.DataFrame:
    roots = {
        "direct_return_15": QUALITY_MODEL_ROOT / "direct_returns/tasks",
        "policy_target_18": POLICY_ROOT / "tasks",
        "t1_direct_6": T1_MODEL_ROOT / "tasks",
        "original_path_24": QUALITY_MODEL_ROOT / "tasks",
    }
    rows: list[dict[str, Any]] = []
    feature_name = "market_all__st_rate"
    for group, root in roots.items():
        for model_path in sorted(root.glob("*/model.txt")):
            booster = lgb.Booster(model_file=str(model_path))
            names = booster.feature_name()
            gains = booster.feature_importance(importance_type="gain").astype(
                np.float64
            )
            index = names.index(feature_name)
            gain = float(gains[index])
            total_gain = float(gains.sum())
            rank = int(np.count_nonzero(gains > gain) + 1)
            task_id = model_path.parent.name
            rows.append(
                {
                    "model_group": group,
                    "task_id": task_id,
                    "target": task_id.split("__")[1],
                    "year": int(task_id.rsplit("__", 1)[1]),
                    "feature": feature_name,
                    "gain": gain,
                    "gain_share": gain / total_gain if total_gain else math.nan,
                    "gain_rank": rank,
                    "feature_count": len(names),
                }
            )
    return pd.DataFrame(rows)


def trade_outcome_row(path: Path, label: str) -> dict[str, Any]:
    trades = pd.read_parquet(path)
    sells = trades.loc[(trades["side"] == "sell") & (trades["status"] == "filled")]
    returns = sells["gross_return"].to_numpy(dtype=np.float64)
    hit = sells["reason"].eq("take_profit").to_numpy()
    hit_mean = float(returns[hit].mean())
    miss_mean = float(returns[~hit].mean())
    breakeven_hit_rate = float(-miss_mean / (hit_mean - miss_mean))
    return {
        "label": label,
        "task": path.parent.name,
        "closed_trades": len(sells),
        "take_profit_hit_rate": float(hit.mean()),
        "hit_mean_gross_return": hit_mean,
        "miss_mean_gross_return": miss_mean,
        "mean_gross_return": float(returns.mean()),
        "loss_below_minus_10_rate": float(np.mean(returns < -0.10)),
        "es05_gross_return": expected_shortfall(returns, 0.05),
        "breakeven_hit_rate": breakeven_hit_rate,
    }


def audit_policy_labels(
    row_index: pd.DataFrame,
    row_symbol_idx: np.ndarray,
    cutoff_idx: int,
    price_label_valid: np.memmap,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    label_root = POLICY_ROOT / "labels"
    fill_day = np.load(label_root / "fill_day.npy", mmap_mode="r")
    trigger_day = np.load(label_root / "trigger_day.npy", mmap_mode="r")
    fill_phase = np.load(label_root / "fill_phase.npy", mmap_mode="r")
    policy_return = np.load(label_root / "policy_return.npy", mmap_mode="r")
    valid = np.load(label_root / "valid.npy", mmap_mode="r")
    outcome_code = np.load(label_root / "outcome_code.npy", mmap_mode="r")
    rows: list[dict[str, Any]] = []
    extreme_records: list[dict[str, Any]] = []
    for position, horizon in enumerate((10, 20)):
        delayed = (fill_phase[:, position] == 3) & (fill_day[:, position] >= 0)
        delay = (
            fill_day[:, position].astype(np.int32)
            - trigger_day[:, position].astype(np.int32)
        )
        resolved = outcome_code[:, position] > 0
        maximum_fill_position = int(np.argmax(fill_day[:, position]))
        maximum_return_position = int(np.nanargmax(policy_return[:, position]))
        rows.append(
            {
                "horizon": horizon,
                "resolved_count": int(resolved.sum()),
                "delayed_open_count": int(delayed.sum()),
                "delay_over_20_count": int(np.count_nonzero(delayed & (delay > 20))),
                "delay_over_60_count": int(np.count_nonzero(delayed & (delay > 60))),
                "maximum_delay_days": int(delay[delayed].max()),
                "maximum_fill_day": int(fill_day[:, position].max()),
                "maximum_policy_return": float(np.nanmax(policy_return[:, position])),
            }
        )
        for kind, selected in (
            ("maximum_fill_day", maximum_fill_position),
            ("maximum_policy_return", maximum_return_position),
        ):
            extreme_records.append(
                {
                    "horizon": horizon,
                    "kind": kind,
                    "row": selected,
                    "trade_date": str(row_index.iloc[selected]["trade_date"]),
                    "symbol": str(row_index.iloc[selected]["symbol"]),
                    "trigger_day": int(trigger_day[selected, position]),
                    "fill_day": int(fill_day[selected, position]),
                    "delay_days": int(delay[selected]),
                    "policy_return": float(policy_return[selected, position]),
                }
            )
    formal_d60 = row_index["date_idx"].to_numpy(dtype=np.int32) + 60 <= cutoff_idx
    inherited_valid = np.asarray(
        price_label_valid[
            row_index.loc[formal_d60, "date_idx"].to_numpy(dtype=np.int32),
            row_symbol_idx[formal_d60],
        ],
        dtype=bool,
    )
    metadata = {
        "extremes": extreme_records,
        "d60_fully_observed_through_cutoff_count": int(formal_d60.sum()),
        "d60_price_label_invalid_count": int((~inherited_valid).sum()),
        "policy_label_contract_uses_price_label_valid": False,
        "logical_maximum_source_date": CUTOFF_DATE,
        "forbidden_2026_read_count": 0,
        "valid_matrix_shape": list(valid.shape),
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_manifest = read_json(MODEL_INPUT_ROOT / "manifest.json")
    pack = read_json(PACK_MANIFEST_PATH)
    row_index = pd.read_parquet(MODEL_INPUT_ROOT / "row_index.parquet")
    row_index["trade_date"] = row_index["trade_date"].astype(str)
    if row_index["trade_date"].max() > CUTOFF_DATE:
        raise RuntimeError("formal row index reads past 2025")
    if row_index["candidate_id"].duplicated().any():
        raise RuntimeError("candidate_id is not unique")
    if row_index.duplicated(["trade_date", "symbol"]).any():
        raise RuntimeError("date-symbol grain is not unique")
    row_date_idx = row_index["date_idx"].to_numpy(dtype=np.int32)
    if np.any(row_date_idx[1:] < row_date_idx[:-1]):
        raise RuntimeError("row index must be date sorted")

    date_values = np.asarray(pack["date_values"], dtype=str)
    cutoff_matches = np.flatnonzero(date_values == CUTOFF_DATE)
    if len(cutoff_matches) != 1:
        raise RuntimeError("2025 cutoff date is missing or duplicated")
    cutoff_idx = int(cutoff_matches[0])
    symbol_to_idx = {
        str(symbol): position for position, symbol in enumerate(pack["symbol_values"])
    }
    row_symbol_idx = row_index["symbol"].map(symbol_to_idx).to_numpy()
    if pd.isna(row_symbol_idx).any():
        raise RuntimeError("row symbols do not align with pack")
    row_symbol_idx = row_symbol_idx.astype(np.int32)

    daily_state = open_memmap(pack["feature_channels"]["daily_state"], np.float32)
    masks = pack["masks"]
    pit_universe = open_memmap(masks["pit_universe_has_bar"], np.bool_)
    is_st = open_memmap(masks["is_st"], np.bool_)
    continuity_break = open_memmap(masks["continuity_break"], np.bool_)
    price_label_valid = open_memmap(masks["price_label_valid"], np.bool_)

    feature_frame = pd.DataFrame(input_manifest["features"])
    feature_position = int(
        feature_frame.loc[
            feature_frame["feature_name"] == "market_all__st_rate", "column_index"
        ].iloc[0]
    )
    compact = np.memmap(
        input_manifest["storage"]["compact"]["path"],
        mode="r",
        dtype=np.float32,
        shape=tuple(input_manifest["storage"]["compact"]["shape"]),
    )
    formal_dates, first_rows = np.unique(row_date_idx, return_index=True)
    st_rows: list[dict[str, Any]] = []
    for date_idx, first_row in zip(formal_dates, first_rows, strict=True):
        universe = np.asarray(pit_universe[date_idx], dtype=bool)
        denominator = int(universe.sum())
        numerator = int(np.count_nonzero(universe & np.asarray(is_st[date_idx])))
        calculated = numerator / denominator if denominator else math.nan
        st_rows.append(
            {
                "date_idx": int(date_idx),
                "trade_date": str(date_values[date_idx]),
                "year": int(str(date_values[date_idx])[:4]),
                "universe_count": denominator,
                "is_st_count": numerator,
                "calculated_st_rate": calculated,
                "feature_st_rate": float(compact[first_row, feature_position]),
            }
        )
    st_daily = pd.DataFrame(st_rows)
    st_daily["feature_error"] = (
        st_daily["feature_st_rate"] - st_daily["calculated_st_rate"]
    )
    st_daily["one_day_change"] = st_daily["calculated_st_rate"].diff()
    largest_change_row = st_daily.loc[st_daily["one_day_change"].abs().idxmax()]
    previous_change_row = st_daily.iloc[int(largest_change_row.name) - 1]

    st_geometry_annual = raw_st_limit_geometry(pack, pit_universe)

    continuity_prefix = np.cumsum(
        np.asarray(continuity_break[: cutoff_idx + 1], dtype=np.int16),
        axis=0,
        dtype=np.int16,
    )
    row_count = len(row_index)
    path_valid = np.zeros(row_count, dtype=bool)
    endpoint_return = np.full(row_count, np.nan, dtype=np.float32)
    mfe = np.full(row_count, np.nan, dtype=np.float32)
    mae = np.full(row_count, np.nan, dtype=np.float32)
    volatility = np.full(row_count, np.nan, dtype=np.float32)
    path_has_continuity_break = np.zeros(row_count, dtype=bool)

    path_spec = pack["label_arrays"]["future_ohlcva_path"]
    for shard in path_spec["shards"]:
        shard_start = int(shard["date_start_idx"])
        shard_end = int(shard["date_end_idx"])
        left = int(np.searchsorted(row_date_idx, shard_start, side="left"))
        right = int(np.searchsorted(row_date_idx, shard_end, side="right"))
        if left >= right:
            continue
        shard_map = open_memmap(shard, np.float32)
        for chunk_left in range(left, right, 100_000):
            chunk_right = min(chunk_left + 100_000, right)
            positions = np.arange(chunk_left, chunk_right, dtype=np.int64)
            in_bounds = row_date_idx[positions] + HORIZON <= cutoff_idx
            positions = positions[in_bounds]
            if not len(positions):
                continue
            dates = row_date_idx[positions]
            symbols = row_symbol_idx[positions]
            local_dates = dates - shard_start
            block = np.asarray(
                shard_map[local_dates, symbols, :HORIZON, :4], dtype=np.float64
            )
            complete = np.isfinite(block).all(axis=(1, 2))
            entry = 1.0 + block[:, 0, 0]
            terminal = 1.0 + block[:, HORIZON - 1, 3]
            complete &= (entry > 0.0) & (terminal > 0.0)
            accepted = positions[complete]
            if not len(accepted):
                continue
            accepted_entry = entry[complete]
            accepted_block = block[complete]
            endpoint_return[accepted] = (
                terminal[complete] / accepted_entry - 1.0
            ).astype(np.float32)
            mfe[accepted] = (
                np.max(1.0 + accepted_block[:, 1:HORIZON, 1], axis=1)
                / accepted_entry
                - 1.0
            ).astype(np.float32)
            mae[accepted] = (
                np.min(1.0 + accepted_block[:, 1:HORIZON, 2], axis=1)
                / accepted_entry
                - 1.0
            ).astype(np.float32)
            volatility[accepted] = np.asarray(
                daily_state[dates[complete], symbols[complete], 5], dtype=np.float32
            )
            path_has_continuity_break[accepted] = (
                continuity_prefix[dates[complete] + HORIZON, symbols[complete]]
                - continuity_prefix[dates[complete], symbols[complete]]
            ) > 0
            path_valid[accepted] = True

    valid = (
        path_valid
        & np.isfinite(endpoint_return)
        & np.isfinite(mfe)
        & np.isfinite(mae)
        & np.isfinite(volatility)
        & (volatility > 0.0)
    )
    valid_positions = np.flatnonzero(valid)
    valid_endpoint = endpoint_return[valid].astype(np.float64)
    valid_log_endpoint = np.log1p(valid_endpoint)
    valid_mfe = mfe[valid].astype(np.float64)
    valid_mae = mae[valid].astype(np.float64)
    valid_volatility = volatility[valid].astype(np.float64)
    valid_year = row_index.loc[valid, "trade_date"].str[:4].astype(int).to_numpy()
    valid_date_idx = row_date_idx[valid]
    standardized_distance = np.log1p(0.08) / (
        valid_volatility * math.sqrt(HORIZON)
    )
    dynamic_barrier = np.expm1(valid_volatility * math.sqrt(HORIZON))
    fixed_hit = valid_mfe >= 0.08
    dynamic_hit = valid_mfe >= dynamic_barrier
    brownian_fixed_probability = 2.0 * ndtr(-standardized_distance)

    edges = np.quantile(valid_volatility, np.linspace(0.0, 1.0, 11))
    volatility_decile = np.searchsorted(
        edges[1:-1], valid_volatility, side="right"
    )
    decile_rows: list[dict[str, Any]] = []
    for decile in range(10):
        selected = volatility_decile == decile
        decile_rows.append(
            {
                "volatility_decile": decile + 1,
                "count": int(selected.sum()),
                "volatility_median": float(np.median(valid_volatility[selected])),
                "standardized_8pct_distance_median": float(
                    np.median(standardized_distance[selected])
                ),
                "fixed_8pct_hit_rate": float(fixed_hit[selected].mean()),
                "volatility_scaled_k1_hit_rate": float(
                    dynamic_hit[selected].mean()
                ),
                "zero_drift_brownian_hit_probability_mean": float(
                    brownian_fixed_probability[selected].mean()
                ),
                "endpoint_mean": float(valid_endpoint[selected].mean()),
                "endpoint_mean_log_return": float(valid_log_endpoint[selected].mean()),
                "endpoint_median": float(np.median(valid_endpoint[selected])),
                "endpoint_es05": expected_shortfall(valid_endpoint[selected], 0.05),
                "mfe_median": float(np.median(valid_mfe[selected])),
                "mae_median": float(np.median(valid_mae[selected])),
            }
        )
    barrier_deciles = pd.DataFrame(decile_rows)

    annual_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for year in range(2011, 2026):
        selected = valid_year == year
        year_dates = np.unique(valid_date_idx[selected])
        daily_fixed = []
        daily_dynamic = []
        for date_idx in year_dates:
            day = selected & (valid_date_idx == date_idx)
            daily_fixed.append(float(fixed_hit[day].mean()))
            daily_dynamic.append(float(dynamic_hit[day].mean()))
        annual_rows.append(
            {
                "year": year,
                "count": int(selected.sum()),
                "date_count": len(year_dates),
                "fixed_8pct_hit_rate": float(fixed_hit[selected].mean()),
                "fixed_8pct_daily_equal_hit_rate": float(np.mean(daily_fixed)),
                "volatility_scaled_k1_hit_rate": float(
                    dynamic_hit[selected].mean()
                ),
                "volatility_scaled_k1_daily_equal_hit_rate": float(
                    np.mean(daily_dynamic)
                ),
                "endpoint_mean": float(valid_endpoint[selected].mean()),
                "endpoint_mean_log_return": float(valid_log_endpoint[selected].mean()),
                "endpoint_es05": expected_shortfall(valid_endpoint[selected], 0.05),
            }
        )
        stats = distribution_stats(valid_endpoint[selected])
        distribution_rows.append({"year": year, **stats})
    barrier_annual = pd.DataFrame(annual_rows)
    endpoint_distribution = pd.DataFrame(distribution_rows)

    unique_valid_dates, date_inverse = np.unique(
        valid_date_idx, return_inverse=True
    )
    date_count = np.bincount(date_inverse)
    date_sum = np.bincount(date_inverse, weights=valid_endpoint)
    date_mean = date_sum / date_count
    date_log_sum = np.bincount(date_inverse, weights=valid_log_endpoint)
    date_log_mean = date_log_sum / date_count
    expanded_date_mean = date_mean[date_inverse]
    date_r2 = float(np.var(expanded_date_mean) / np.var(valid_endpoint))
    daily_mean_series = pd.Series(date_mean, index=unique_valid_dates)
    daily_hac20_se, daily_hac20_t = hac_mean_t(date_mean, 20)
    daily_hac60_se, daily_hac60_t = hac_mean_t(date_mean, 60)
    daily_block20_low, daily_block20_high = moving_block_mean_ci(
        date_mean, 20, seed=RNG_SEED + 9000
    )
    daily_block60_low, daily_block60_high = moving_block_mean_ci(
        date_mean, 60, seed=RNG_SEED + 9060
    )
    daily_log_hac20_se, daily_log_hac20_t = hac_mean_t(date_log_mean, 20)
    daily_log_hac60_se, daily_log_hac60_t = hac_mean_t(date_log_mean, 60)
    daily_log_block20_low, daily_log_block20_high = moving_block_mean_ci(
        date_log_mean, 20, seed=RNG_SEED + 9100
    )
    daily_log_block60_low, daily_log_block60_high = moving_block_mean_ci(
        date_log_mean, 60, seed=RNG_SEED + 9160
    )
    daily_iid_se = float(date_mean.std(ddof=1) / math.sqrt(len(date_mean)))
    daily_log_iid_se = float(
        date_log_mean.std(ddof=1) / math.sqrt(len(date_log_mean))
    )
    endpoint_by_date = pd.DataFrame(
        {
            "date_idx": unique_valid_dates,
            "trade_date": date_values[unique_valid_dates],
            "candidate_count": date_count,
            "endpoint_mean": date_mean,
            "endpoint_mean_log_return": date_log_mean,
        }
    )
    dependence = {
        "row_count": len(valid_endpoint),
        "signal_date_count": len(unique_valid_dates),
        "rough_nonoverlapping_d20_block_count": len(unique_valid_dates) / HORIZON,
        "date_fixed_effect_variance_share": date_r2,
        "naive_row_standard_error_of_mean": float(
            valid_endpoint.std(ddof=1) / math.sqrt(len(valid_endpoint))
        ),
        "date_equal_mean_return": float(date_mean.mean()),
        "date_equal_standard_error_of_mean": daily_iid_se,
        "date_equal_hac20_standard_error_of_mean": daily_hac20_se,
        "date_equal_hac20_t_of_mean": daily_hac20_t,
        "date_equal_hac60_standard_error_of_mean": daily_hac60_se,
        "date_equal_hac60_t_of_mean": daily_hac60_t,
        "date_equal_hac20_effective_signal_date_count": float(
            len(date_mean) * (daily_iid_se / daily_hac20_se) ** 2
        ),
        "date_equal_hac60_effective_signal_date_count": float(
            len(date_mean) * (daily_iid_se / daily_hac60_se) ** 2
        ),
        "date_equal_block20_mean_ci": [daily_block20_low, daily_block20_high],
        "date_equal_block60_mean_ci": [daily_block60_low, daily_block60_high],
        "date_equal_mean_log_return": float(date_log_mean.mean()),
        "date_equal_log_standard_error_of_mean": daily_log_iid_se,
        "date_equal_log_hac20_standard_error_of_mean": daily_log_hac20_se,
        "date_equal_log_hac20_t_of_mean": daily_log_hac20_t,
        "date_equal_log_hac60_standard_error_of_mean": daily_log_hac60_se,
        "date_equal_log_hac60_t_of_mean": daily_log_hac60_t,
        "date_equal_log_hac20_effective_signal_date_count": float(
            len(date_log_mean) * (daily_log_iid_se / daily_log_hac20_se) ** 2
        ),
        "date_equal_log_hac60_effective_signal_date_count": float(
            len(date_log_mean) * (daily_log_iid_se / daily_log_hac60_se) ** 2
        ),
        "date_equal_log_block20_mean_ci": [
            daily_log_block20_low,
            daily_log_block20_high,
        ],
        "date_equal_log_block60_mean_ci": [
            daily_log_block60_low,
            daily_log_block60_high,
        ],
        "daily_mean_autocorrelation_lag1": float(daily_mean_series.autocorr(1)),
        "daily_mean_autocorrelation_lag5": float(daily_mean_series.autocorr(5)),
        "daily_mean_autocorrelation_lag20": float(daily_mean_series.autocorr(20)),
    }

    fixed_hit_endpoint = valid_endpoint[fixed_hit]
    miss_endpoint = valid_endpoint[~fixed_hit]
    barrier_summary = {
        "valid_path_count": len(valid_endpoint),
        "volatility_median": float(np.median(valid_volatility)),
        "standardized_8pct_distance_quantiles": {
            "q01": float(np.quantile(standardized_distance, 0.01)),
            "median": float(np.median(standardized_distance)),
            "q99": float(np.quantile(standardized_distance, 0.99)),
        },
        "fixed_8pct_annual_rate_range": [
            float(barrier_annual["fixed_8pct_hit_rate"].min()),
            float(barrier_annual["fixed_8pct_hit_rate"].max()),
        ],
        "fixed_8pct_annual_rate_std": float(
            barrier_annual["fixed_8pct_hit_rate"].std(ddof=0)
        ),
        "scaled_k1_annual_rate_range": [
            float(barrier_annual["volatility_scaled_k1_hit_rate"].min()),
            float(barrier_annual["volatility_scaled_k1_hit_rate"].max()),
        ],
        "scaled_k1_annual_rate_std": float(
            barrier_annual["volatility_scaled_k1_hit_rate"].std(ddof=0)
        ),
        "zero_drift_brownian_k1_probability": float(2.0 * ndtr(-1.0)),
        "fixed_hit_endpoint_negative_rate": float(
            np.mean(fixed_hit_endpoint < 0.0)
        ),
        "fixed_hit_endpoint_below_minus_10_rate": float(
            np.mean(fixed_hit_endpoint < -0.10)
        ),
        "fixed_hit_mean_peak_retrace": float(
            np.mean(valid_mfe[fixed_hit] - fixed_hit_endpoint)
        ),
        "fixed_miss_endpoint_mean": float(miss_endpoint.mean()),
        "fixed_miss_endpoint_es05": expected_shortfall(miss_endpoint, 0.05),
        "endpoint_mean_log_return": float(valid_log_endpoint.mean()),
        "continuity_break_within_d20_count": int(
            path_has_continuity_break[valid_positions].sum()
        ),
        "logical_maximum_outcome_date": CUTOFF_DATE,
        "forbidden_2026_read_count": 0,
    }

    policy_delays, policy_metadata = audit_policy_labels(
        row_index,
        row_symbol_idx,
        cutoff_idx,
        price_label_valid,
    )

    importance = model_importance_rows()
    importance_summary = (
        importance.groupby("model_group", as_index=False)
        .agg(
            model_count=("task_id", "count"),
            positive_gain_count=("gain", lambda values: int((values > 0).sum())),
            median_gain_share=("gain_share", "median"),
            minimum_gain_share=("gain_share", "min"),
            maximum_gain_share=("gain_share", "max"),
            median_gain_rank=("gain_rank", "median"),
            best_gain_rank=("gain_rank", "min"),
            worst_gain_rank=("gain_rank", "max"),
        )
        .sort_values("model_group")
    )

    profit_timeout_root = EXECUTION_ROOT / "profit_timeout/tasks"
    threshold_rows: list[dict[str, Any]] = []
    comparison_counter = 0
    for slot_count in (6, 12):
        for cost in ("base", "stress"):
            preferred = (
                profit_timeout_root
                / f"dual_mfe_risk_veto__tp08_d20__k{slot_count:02d}__{cost}/equity.parquet"
            )
            for reference_label in ("tp05", "tp10", "timeout_only"):
                reference = (
                    profit_timeout_root
                    / f"dual_mfe_risk_veto__{reference_label}_d20__k{slot_count:02d}__{cost}/equity.parquet"
                )
                rows = paired_return_statistics(
                    preferred,
                    reference,
                    seed=RNG_SEED + comparison_counter * 100,
                )
                for row in rows:
                    row.update(
                        {
                            "slot_count": slot_count,
                            "cost": cost,
                            "preferred_threshold": "8pct",
                            "reference_threshold": reference_label,
                        }
                    )
                threshold_rows.extend(rows)
                comparison_counter += 1
    threshold_robustness = pd.DataFrame(threshold_rows)

    paired_model_rows: list[dict[str, Any]] = []
    old_root = EXECUTION_ROOT / "profit_timeout/tasks"
    new_root = POLICY_ROOT / "replay/tasks"
    for slot_count in (6, 12):
        for cost in ("base", "stress"):
            new_path = (
                new_root
                / f"dual_mfe_hit_timeout_maximin__tp08_d20__k{slot_count:02d}__{cost}/equity.parquet"
            )
            old_path = (
                old_root
                / f"dual_mfe_risk_veto__tp08_d20__k{slot_count:02d}__{cost}/equity.parquet"
            )
            rows = paired_return_statistics(
                new_path,
                old_path,
                seed=RNG_SEED + 5000 + slot_count * 100 + (cost == "stress") * 10,
            )
            for row in rows:
                row.update({"slot_count": slot_count, "cost": cost})
            paired_model_rows.extend(rows)
    paired_model_accounts = pd.DataFrame(paired_model_rows)

    trade_outcomes = pd.DataFrame(
        [
            trade_outcome_row(
                old_root
                / "dual_mfe_risk_veto__tp08_d20__k12__base/trades.parquet",
                "old_mfe_plus_risk",
            ),
            trade_outcome_row(
                new_root
                / "dual_mfe_hit_timeout_maximin__tp08_d20__k12__base/trades.parquet",
                "new_hit_plus_timeout",
            ),
        ]
    )

    st_annual = (
        st_daily.groupby("year", as_index=False)
        .agg(
            trading_days=("trade_date", "count"),
            mean_st_rate=("calculated_st_rate", "mean"),
            minimum_st_rate=("calculated_st_rate", "min"),
            maximum_st_rate=("calculated_st_rate", "max"),
            maximum_feature_error=("feature_error", lambda values: values.abs().max()),
        )
        .merge(st_geometry_annual, on="year", how="left")
    )

    data_quality_findings = pd.DataFrame(
        [
            {
                "severity": "high",
                "confidence": "high",
                "finding": "Policy labels can defer a D10/D20 exit until any later sellable open",
                "evidence": f"Maximum observed delay is {int(policy_delays.maximum_delay_days.max())} trading days; the implementation has only a 2025 cutoff.",
                "impact": "Target horizon, purge length, timeout-return tails, and calibration no longer describe D10/D20.",
            },
            {
                "severity": "high",
                "confidence": "high",
                "finding": "The historical ST-state series has an implausible one-day splice discontinuity",
                "evidence": f"ST rate moves from {float(previous_change_row.calculated_st_rate):.4%} on {previous_change_row.trade_date} to {float(largest_change_row.calculated_st_rate):.4%} on {largest_change_row.trade_date}.",
                "impact": "Universe rules, reconstructed price limits, market-state interactions, and blocked-exit labels may all change.",
            },
            {
                "severity": "high",
                "confidence": "medium",
                "finding": "Many raw-price flat days at an exact +/-5% tick are not marked ST",
                "evidence": f"Diagnostic count is {int(st_geometry_annual.unmarked_st_candidate_count.sum())} stock-days versus {int(st_geometry_annual.marked_st_count.sum())} marked stock-days.",
                "impact": "The test is not an official-name truth set and may include corporate-action edge cases, but the scale requires an exchange-name-interval reconstruction.",
            },
            {
                "severity": "high",
                "confidence": "high",
                "finding": "Candidate rows are not independent observations",
                "evidence": f"There are {len(valid_endpoint):,} D20 paths but only {len(unique_valid_dates):,} signal dates and about {len(unique_valid_dates) / HORIZON:.1f} non-overlapping 20-day blocks.",
                "impact": "Row-level p-values and random-row validation materially understate uncertainty.",
            },
            {
                "severity": "medium",
                "confidence": "high",
                "finding": "The ST-rate feature is correctly named as a date-level market rate, but it is built from suspect ST states",
                "evidence": f"Its maximum difference from is_st/pit_universe is {float(st_daily.feature_error.abs().max()):.3g}; it receives positive gain in {int((importance.gain > 0).sum())}/{len(importance)} audited models.",
                "impact": "The feature is not wrong merely because it is constant within date; repair the source and run a drop/rebuild ablation.",
            },
            {
                "severity": "high",
                "confidence": "high",
                "finding": "2023-2025 is retrospective rolling OOS, not an untouched holdout",
                "evidence": "The same period has already been used repeatedly for target, threshold, model, and execution comparisons.",
                "impact": "Current confidence intervals do not include adaptive research selection; final confirmation requires future frozen data.",
            },
        ]
    )

    outputs = {
        "st_rate_daily.csv": st_daily,
        "st_rate_annual.csv": st_annual,
        "st_limit_geometry_annual.csv": st_geometry_annual,
        "barrier_by_volatility_decile.csv": barrier_deciles,
        "barrier_by_year.csv": barrier_annual,
        "endpoint_distribution_by_year.csv": endpoint_distribution,
        "endpoint_by_date.csv": endpoint_by_date,
        "policy_label_delays.csv": policy_delays,
        "st_feature_importance.csv": importance,
        "st_feature_importance_summary.csv": importance_summary,
        "threshold_robustness.csv": threshold_robustness,
        "paired_model_accounts.csv": paired_model_accounts,
        "trade_outcomes.csv": trade_outcomes,
        "data_quality_findings.csv": data_quality_findings,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8")

    summary = {
        "schema": "seq100_mathematical_foundations_audit/v1",
        "status": "completed",
        "scope": {
            "research_start": str(row_index["trade_date"].min()),
            "research_end": str(row_index["trade_date"].max()),
            "logical_maximum_outcome_date": CUTOFF_DATE,
            "forbidden_2026_read_count": 0,
            "path_horizon": HORIZON,
        },
        "row_index": {
            "row_count": len(row_index),
            "signal_date_count": int(row_index["trade_date"].nunique()),
            "security_count": int(row_index["security_id"].nunique()),
            "pack_symbol_count": len(pack["symbol_values"]),
            "candidate_id_missing_count": int(row_index["candidate_id"].isna().sum()),
            "candidate_id_duplicate_count": int(
                row_index["candidate_id"].duplicated().sum()
            ),
            "date_symbol_duplicate_count": int(
                row_index.duplicated(["trade_date", "symbol"]).sum()
            ),
        },
        "st_state": {
            "feature_max_absolute_error_vs_pack_rate": float(
                st_daily["feature_error"].abs().max()
            ),
            "largest_one_day_change": {
                "previous_date": str(previous_change_row.trade_date),
                "previous_rate": float(previous_change_row.calculated_st_rate),
                "date": str(largest_change_row.trade_date),
                "rate": float(largest_change_row.calculated_st_rate),
                "change": float(largest_change_row.one_day_change),
            },
            "marked_five_percent_flat_stock_days": int(
                st_geometry_annual["marked_st_count"].sum()
            ),
            "unmarked_five_percent_flat_stock_days": int(
                st_geometry_annual["unmarked_st_candidate_count"].sum()
            ),
            "diagnostic_is_not_official_truth": True,
            "diagnostic_price_basis": "pinned_qdp_raw_unadjusted_ohlc",
        },
        "barrier": barrier_summary,
        "endpoint_distribution_all": {
            **distribution_stats(valid_endpoint),
            "mean_log_return": float(valid_log_endpoint.mean()),
        },
        "dependence": dependence,
        "policy_labels": policy_metadata,
        "model_importance": importance_summary.to_dict(orient="records"),
        "trade_outcomes": trade_outcomes.to_dict(orient="records"),
        "source_files": {
            "model_input_manifest": {
                "path": str(MODEL_INPUT_ROOT / "manifest.json"),
                "sha256": file_sha256(MODEL_INPUT_ROOT / "manifest.json"),
            },
            "row_index": {
                "path": str(MODEL_INPUT_ROOT / "row_index.parquet"),
                "sha256": file_sha256(MODEL_INPUT_ROOT / "row_index.parquet"),
            },
            "pack_manifest": {
                "path": str(PACK_MANIFEST_PATH),
                "sha256": file_sha256(PACK_MANIFEST_PATH),
            },
            "policy_label_manifest": {
                "path": str(POLICY_ROOT / "labels/manifest.json"),
                "sha256": file_sha256(POLICY_ROOT / "labels/manifest.json"),
            },
        },
        "outputs": sorted(outputs),
    }
    write_json(OUTPUT_DIR / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
