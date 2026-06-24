from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_TRAINING_PACK_MANIFEST = Path(
    "quant_data_platform/data/memmap/training_pack/"
    "mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/"
    "qdp_training_pack_manifest.json"
)
DEFAULT_ROLES = ("train", "validation", "test")
DEFAULT_LABEL_HORIZONS = (1, 2, 4)
DEFAULT_TARGET_KINDS = ("net_abs", "net_excess")
DEFAULT_GROUP_SCOPES = ("all", "role", "year", "role_year")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return str(path.resolve())


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def _parse_csv_ints(raw: str | Iterable[int] | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = [int(item) for item in raw]
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out or default)


def _parse_csv_strings(raw: str | Iterable[str] | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out or default)


def _resolve_path(path: str | Path, *, root: Path | None = None) -> Path:
    out = Path(path)
    if not out.is_absolute() and root is not None:
        out = root / out
    return out


def _exit_label(label_horizon: int) -> str:
    return f"D+{int(label_horizon) + 1}_open"


def _safe_div(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, np.nan)


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _feature_percentile_ranks(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(features)
    ranks = pd.DataFrame(features).rank(axis=0, method="average", pct=True, na_option="keep").to_numpy(dtype=np.float32)
    ranks = np.nan_to_num(ranks, nan=0.5, posinf=0.5, neginf=0.5)
    return np.clip(ranks, 0.0, 1.0), finite


def _target_percentile_rank(target: np.ndarray) -> np.ndarray:
    ranks = pd.Series(np.asarray(target, dtype="float64")).rank(method="average", pct=True, na_option="keep").to_numpy(dtype=np.float32)
    ranks = np.nan_to_num(ranks, nan=0.5, posinf=0.5, neginf=0.5)
    return np.clip(ranks, 0.0, 1.0)


@dataclass(frozen=True)
class FeatureSemantics:
    feature: str
    semantic_type: str
    source_family: str
    value_kind: str
    ordinal_rank_profile: bool
    preferred_diagnostic: str
    interpretation_note: str


def _feature_source_family(feature: str) -> str:
    name = str(feature)
    if name.startswith("market_"):
        return "market"
    if name.startswith("industry_") or "industry" in name:
        return "industry"
    if name.startswith("peer_") or "peer_" in name:
        return "peer"
    if name.startswith("index_"):
        return "index_membership"
    if name.startswith("valuation_"):
        return "valuation"
    if "intraday" in name:
        return "intraday"
    if name.startswith("raw_"):
        return "raw_daily_bar"
    if name.startswith("cs_rank_") or name.startswith("cs_z_"):
        return "cross_section_transform"
    if name.startswith("local_"):
        return "local_path"
    if name.startswith("ret_") or name.startswith("vol_") or name.startswith("turn"):
        return "daily_price_volume"
    if name.startswith("adjust_"):
        return "event_quality"
    return "other"


def classify_feature_semantics(feature: str) -> FeatureSemantics:
    name = str(feature)
    family = _feature_source_family(name)
    if name.startswith("market_"):
        return FeatureSemantics(
            feature=name,
            semantic_type="market_level",
            source_family=family,
            value_kind="date_context_scalar",
            ordinal_rank_profile=False,
            preferred_diagnostic="date_level_target_correlation",
            interpretation_note="Same-date cross-sectional rank is usually weak for market-level context; inspect relation across dates/regimes.",
        )
    if name.startswith("index_") and "_member" in name:
        return FeatureSemantics(
            feature=name,
            semantic_type="membership_flag",
            source_family=family,
            value_kind="binary",
            ordinal_rank_profile=False,
            preferred_diagnostic="group_0_vs_1",
            interpretation_note="Interpret as membership group comparison, not a graded high/low signal.",
        )
    if name.endswith("_flag") or "_flag_" in name:
        return FeatureSemantics(
            feature=name,
            semantic_type="binary_flag",
            source_family=family,
            value_kind="binary",
            ordinal_rank_profile=False,
            preferred_diagnostic="group_0_vs_1",
            interpretation_note="Interpret as flag=1 versus flag=0; deciles are only a coarse fallback.",
        )
    if name.endswith("_bucket_id"):
        return FeatureSemantics(
            feature=name,
            semantic_type="ordinal_bucket",
            source_family=family,
            value_kind="bucket_id",
            ordinal_rank_profile=True,
            preferred_diagnostic="bucket_group_profile",
            interpretation_note="Interpret by bucket groups; high/low is valid only if the bucket id ordering is contractual.",
        )
    if name.endswith("_id"):
        return FeatureSemantics(
            feature=name,
            semantic_type="nominal_id",
            source_family=family,
            value_kind="category_id",
            ordinal_rank_profile=False,
            preferred_diagnostic="category_group_profile",
            interpretation_note="Category ids are not ordinal unless separately proven; avoid high/low interpretation.",
        )
    if name.startswith("cs_rank_") or name.endswith("_rank") or "_rank_" in name:
        return FeatureSemantics(
            feature=name,
            semantic_type="already_ranked",
            source_family=family,
            value_kind="cross_section_rank",
            ordinal_rank_profile=True,
            preferred_diagnostic="rank_ic_decile",
            interpretation_note="Already a rank-like feature; re-ranking is monotonic but redundant.",
        )
    if name.startswith("cs_z_") or name.endswith("_z") or "_z" in name:
        return FeatureSemantics(
            feature=name,
            semantic_type="zscore_continuous",
            source_family=family,
            value_kind="standardized_continuous",
            ordinal_rank_profile=True,
            preferred_diagnostic="rank_ic_decile",
            interpretation_note="Standardized continuous feature; rank profile is valid but absolute thresholds are not used here.",
        )
    if name.endswith("_count") or name.endswith("_count_log"):
        return FeatureSemantics(
            feature=name,
            semantic_type="ordinal_count",
            source_family=family,
            value_kind="count_or_log_count",
            ordinal_rank_profile=True,
            preferred_diagnostic="rank_ic_decile",
            interpretation_note="Ordered count-like context; rank profile is valid but economic meaning depends on the count definition.",
        )
    return FeatureSemantics(
        feature=name,
        semantic_type="continuous_ordinal",
        source_family=family,
        value_kind="continuous_scalar",
        ordinal_rank_profile=True,
        preferred_diagnostic="rank_ic_decile",
        interpretation_note="Continuous or ordered scalar; high/low means same-date numeric rank, not automatically a trading rule.",
    )


def _feature_semantics_frame(feature_semantics: Iterable[FeatureSemantics]) -> pd.DataFrame:
    return pd.DataFrame([sem.__dict__ for sem in feature_semantics])


def _category_value_key(value: Any) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(out):
        return "NA"
    rounded = round(out)
    if abs(out - float(rounded)) <= 1.0e-6:
        return str(int(rounded))
    return f"{out:.8g}"


def _row_scopes(role: str, date: str, group_scopes: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    year = str(pd.Timestamp(date).year)
    scopes: list[tuple[str, str]] = []
    for scope in group_scopes:
        if scope == "all":
            scopes.append(("all", "all"))
        elif scope == "role":
            scopes.append(("role", str(role)))
        elif scope == "year":
            scopes.append(("year", year))
        elif scope == "role_year":
            scopes.append(("role_year", f"{role}:{year}"))
        else:
            scopes.append((scope, f"{role}:{year}"))
    return tuple(scopes)


@dataclass
class FeatureProfileAccumulator:
    feature_columns: tuple[str, ...]
    feature_semantics: tuple[FeatureSemantics, ...] | None = None
    n_deciles: int = 10

    def __post_init__(self) -> None:
        n_features = len(self.feature_columns)
        n_deciles = int(self.n_deciles)
        if self.feature_semantics is None:
            self.feature_semantics = tuple(classify_feature_semantics(item) for item in self.feature_columns)
        if len(self.feature_semantics) != n_features:
            raise ValueError("feature_semantics length must match feature_columns length.")
        self._categorical_feature_indices = [
            idx
            for idx, sem in enumerate(self.feature_semantics)
            if sem.semantic_type in {"binary_flag", "membership_flag", "ordinal_bucket", "nominal_id"}
        ]
        self._market_feature_indices = [
            idx
            for idx, sem in enumerate(self.feature_semantics)
            if sem.semantic_type == "market_level"
        ]
        self.date_count = 0
        self.row_count = 0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.target_hit_sum = 0.0
        self.target_top5_sum = 0.0
        self.target_top5_count = 0
        self.target_bottom5_sum = 0.0
        self.target_bottom5_count = 0
        self.target_top10_sum = 0.0
        self.target_top10_count = 0
        self.target_bottom10_sum = 0.0
        self.target_bottom10_count = 0
        self.feature_finite_count = np.zeros(n_features, dtype=np.float64)
        self.feature_rank_all_sum = np.zeros(n_features, dtype=np.float64)
        self.feature_rank_all_count = 0
        self.rank_ic_sum = np.zeros(n_features, dtype=np.float64)
        self.rank_ic_sq_sum = np.zeros(n_features, dtype=np.float64)
        self.rank_ic_positive_count = np.zeros(n_features, dtype=np.float64)
        self.rank_ic_valid_count = np.zeros(n_features, dtype=np.float64)
        self.top5_rank_sum = np.zeros(n_features, dtype=np.float64)
        self.top5_rank_count = 0
        self.bottom5_rank_sum = np.zeros(n_features, dtype=np.float64)
        self.bottom5_rank_count = 0
        self.top10_rank_sum = np.zeros(n_features, dtype=np.float64)
        self.top10_rank_count = 0
        self.bottom10_rank_sum = np.zeros(n_features, dtype=np.float64)
        self.bottom10_rank_count = 0
        self.positive_rank_sum = np.zeros(n_features, dtype=np.float64)
        self.positive_rank_count = 0
        self.negative_rank_sum = np.zeros(n_features, dtype=np.float64)
        self.negative_rank_count = 0
        self.decile_count = np.zeros((n_features, n_deciles), dtype=np.float64)
        self.decile_target_sum = np.zeros((n_features, n_deciles), dtype=np.float64)
        self.decile_hit_sum = np.zeros((n_features, n_deciles), dtype=np.float64)
        self._decile_offsets = (np.arange(n_features, dtype=np.int64) * n_deciles)[None, :]
        self.category_count: dict[tuple[int, str], float] = defaultdict(float)
        self.category_target_sum: dict[tuple[int, str], float] = defaultdict(float)
        self.category_hit_sum: dict[tuple[int, str], float] = defaultdict(float)
        self.market_feature_sum = np.zeros(n_features, dtype=np.float64)
        self.market_feature_sq_sum = np.zeros(n_features, dtype=np.float64)
        self.market_target_sum = np.zeros(n_features, dtype=np.float64)
        self.market_target_sq_sum = np.zeros(n_features, dtype=np.float64)
        self.market_feature_target_sum = np.zeros(n_features, dtype=np.float64)
        self.market_valid_date_count = np.zeros(n_features, dtype=np.float64)
        self.market_value_unique_sum = np.zeros(n_features, dtype=np.float64)

    def update(
        self,
        *,
        features: np.ndarray,
        feature_ranks: np.ndarray,
        feature_finite: np.ndarray,
        target: np.ndarray,
    ) -> None:
        y = np.asarray(target, dtype=np.float64)
        valid_y = np.isfinite(y)
        if int(valid_y.sum()) < 5:
            return
        raw_features = np.asarray(features[valid_y], dtype=np.float32)
        ranks = np.asarray(feature_ranks[valid_y], dtype=np.float32)
        finite = np.asarray(feature_finite[valid_y], dtype=bool)
        y = y[valid_y]
        n_rows = int(y.shape[0])
        target_rank = _target_percentile_rank(y)
        hit = (y > 0.0).astype(np.float64)

        self.date_count += 1
        self.row_count += n_rows
        self.target_sum += float(y.sum())
        self.target_sq_sum += float(np.square(y).sum())
        self.target_hit_sum += float(hit.sum())
        self.feature_finite_count += finite.sum(axis=0)
        self.feature_rank_all_sum += ranks.sum(axis=0)
        self.feature_rank_all_count += n_rows

        centered_target = target_rank.astype(np.float64) - float(target_rank.mean())
        centered_features = ranks.astype(np.float64) - ranks.astype(np.float64).mean(axis=0)
        numerator = np.sum(centered_features * centered_target[:, None], axis=0)
        denom = np.sqrt(np.sum(np.square(centered_features), axis=0) * float(np.sum(np.square(centered_target))))
        with np.errstate(divide="ignore", invalid="ignore"):
            rank_ic = numerator / denom
        valid_ic = np.isfinite(rank_ic)
        self.rank_ic_sum[valid_ic] += rank_ic[valid_ic]
        self.rank_ic_sq_sum[valid_ic] += np.square(rank_ic[valid_ic])
        self.rank_ic_positive_count[valid_ic] += (rank_ic[valid_ic] > 0.0).astype(np.float64)
        self.rank_ic_valid_count[valid_ic] += 1.0

        self._update_target_group(ranks, y, target_rank >= 0.95, "top5")
        self._update_target_group(ranks, y, target_rank <= 0.05, "bottom5")
        self._update_target_group(ranks, y, target_rank >= 0.90, "top10")
        self._update_target_group(ranks, y, target_rank <= 0.10, "bottom10")
        self._update_rank_group(ranks, y > 0.0, "positive")
        self._update_rank_group(ranks, y <= 0.0, "negative")
        self._update_deciles(ranks, y, hit)
        self._update_categories(raw_features, y, hit)
        self._update_market_dates(raw_features, y)

    def _update_target_group(self, ranks: np.ndarray, y: np.ndarray, mask: np.ndarray, name: str) -> None:
        count = int(mask.sum())
        if count <= 0:
            return
        rank_sum = np.asarray(ranks[mask], dtype=np.float64).sum(axis=0)
        target_sum = float(np.asarray(y[mask], dtype=np.float64).sum())
        if name == "top5":
            self.top5_rank_sum += rank_sum
            self.top5_rank_count += count
            self.target_top5_sum += target_sum
            self.target_top5_count += count
        elif name == "bottom5":
            self.bottom5_rank_sum += rank_sum
            self.bottom5_rank_count += count
            self.target_bottom5_sum += target_sum
            self.target_bottom5_count += count
        elif name == "top10":
            self.top10_rank_sum += rank_sum
            self.top10_rank_count += count
            self.target_top10_sum += target_sum
            self.target_top10_count += count
        elif name == "bottom10":
            self.bottom10_rank_sum += rank_sum
            self.bottom10_rank_count += count
            self.target_bottom10_sum += target_sum
            self.target_bottom10_count += count

    def _update_rank_group(self, ranks: np.ndarray, mask: np.ndarray, name: str) -> None:
        count = int(mask.sum())
        if count <= 0:
            return
        rank_sum = np.asarray(ranks[mask], dtype=np.float64).sum(axis=0)
        if name == "positive":
            self.positive_rank_sum += rank_sum
            self.positive_rank_count += count
        elif name == "negative":
            self.negative_rank_sum += rank_sum
            self.negative_rank_count += count

    def _update_deciles(self, ranks: np.ndarray, y: np.ndarray, hit: np.ndarray) -> None:
        deciles = np.clip((ranks * float(self.n_deciles)).astype(np.int64), 0, self.n_deciles - 1)
        flat = (deciles + self._decile_offsets).ravel()
        size = len(self.feature_columns) * self.n_deciles
        y_weights = np.broadcast_to(y[:, None], deciles.shape).ravel()
        hit_weights = np.broadcast_to(hit[:, None], deciles.shape).ravel()
        self.decile_count += np.bincount(flat, minlength=size).reshape(len(self.feature_columns), self.n_deciles)
        self.decile_target_sum += np.bincount(flat, weights=y_weights, minlength=size).reshape(len(self.feature_columns), self.n_deciles)
        self.decile_hit_sum += np.bincount(flat, weights=hit_weights, minlength=size).reshape(len(self.feature_columns), self.n_deciles)

    def _update_categories(self, features: np.ndarray, y: np.ndarray, hit: np.ndarray) -> None:
        for feature_idx in self._categorical_feature_indices:
            values = np.asarray(features[:, feature_idx], dtype=np.float64)
            finite = np.isfinite(values)
            if int(finite.sum()) <= 0:
                continue
            keys = np.asarray([_category_value_key(value) for value in values], dtype=object)
            for key in sorted(set(keys[finite].tolist())):
                mask = finite & np.equal(keys, key)
                count = int(mask.sum())
                if count <= 0:
                    continue
                storage_key = (int(feature_idx), str(key))
                self.category_count[storage_key] += float(count)
                self.category_target_sum[storage_key] += float(np.asarray(y[mask], dtype=np.float64).sum())
                self.category_hit_sum[storage_key] += float(np.asarray(hit[mask], dtype=np.float64).sum())

    def _update_market_dates(self, features: np.ndarray, y: np.ndarray) -> None:
        if not self._market_feature_indices:
            return
        target_mean = float(np.asarray(y, dtype=np.float64).mean())
        for feature_idx in self._market_feature_indices:
            values = np.asarray(features[:, feature_idx], dtype=np.float64)
            finite = np.isfinite(values)
            if int(finite.sum()) <= 0:
                continue
            mean_value = float(values[finite].mean())
            unique_count = float(len(np.unique(np.round(values[finite], 10))))
            self.market_feature_sum[feature_idx] += mean_value
            self.market_feature_sq_sum[feature_idx] += mean_value * mean_value
            self.market_target_sum[feature_idx] += target_mean
            self.market_target_sq_sum[feature_idx] += target_mean * target_mean
            self.market_feature_target_sum[feature_idx] += mean_value * target_mean
            self.market_valid_date_count[feature_idx] += 1.0
            self.market_value_unique_sum[feature_idx] += unique_count

    def feature_rows(self, *, scope: str, scope_value: str, target_kind: str, label_horizon: int) -> list[dict[str, Any]]:
        n_features = len(self.feature_columns)
        rank_ic_mean = _safe_div(self.rank_ic_sum, self.rank_ic_valid_count)
        rank_ic_second = _safe_div(self.rank_ic_sq_sum, self.rank_ic_valid_count)
        rank_ic_std = np.sqrt(np.maximum(rank_ic_second - np.square(rank_ic_mean), 0.0))
        rank_ic_positive_rate = _safe_div(self.rank_ic_positive_count, self.rank_ic_valid_count)
        all_rank_mean = _safe_div(self.feature_rank_all_sum, float(self.feature_rank_all_count))
        top5_rank_mean = _safe_div(self.top5_rank_sum, float(self.top5_rank_count))
        bottom5_rank_mean = _safe_div(self.bottom5_rank_sum, float(self.bottom5_rank_count))
        top10_rank_mean = _safe_div(self.top10_rank_sum, float(self.top10_rank_count))
        bottom10_rank_mean = _safe_div(self.bottom10_rank_sum, float(self.bottom10_rank_count))
        positive_rank_mean = _safe_div(self.positive_rank_sum, float(self.positive_rank_count))
        negative_rank_mean = _safe_div(self.negative_rank_sum, float(self.negative_rank_count))
        finite_rate = _safe_div(self.feature_finite_count, float(self.row_count))
        decile_mean = _safe_div(self.decile_target_sum, self.decile_count)
        decile_hit_rate = _safe_div(self.decile_hit_sum, self.decile_count)
        target_mean = float(self.target_sum / self.row_count) if self.row_count else float("nan")
        rows: list[dict[str, Any]] = []
        decile_index = np.arange(self.n_deciles, dtype=np.float64)
        decile_index_centered = decile_index - float(decile_index.mean())
        decile_index_ss = float(np.sum(np.square(decile_index_centered)))
        for idx, feature in enumerate(self.feature_columns):
            semantics = self.feature_semantics[idx]
            means = decile_mean[idx]
            valid_deciles = np.isfinite(means)
            monotonic_corr = float("nan")
            if int(valid_deciles.sum()) >= 3:
                x = decile_index_centered[valid_deciles]
                y = means[valid_deciles] - float(np.nanmean(means[valid_deciles]))
                denom = math.sqrt(float(np.sum(np.square(x))) * float(np.sum(np.square(y))))
                monotonic_corr = float(np.sum(x * y) / denom) if denom > 0.0 else float("nan")
            top_decile_mean = float(means[-1]) if np.isfinite(means[-1]) else float("nan")
            bottom_decile_mean = float(means[0]) if np.isfinite(means[0]) else float("nan")
            top_decile_hit = float(decile_hit_rate[idx, -1]) if np.isfinite(decile_hit_rate[idx, -1]) else float("nan")
            bottom_decile_hit = float(decile_hit_rate[idx, 0]) if np.isfinite(decile_hit_rate[idx, 0]) else float("nan")
            signed_spread = top_decile_mean - bottom_decile_mean if math.isfinite(top_decile_mean) and math.isfinite(bottom_decile_mean) else float("nan")
            extreme_values = [value for value in (top_decile_mean, bottom_decile_mean) if math.isfinite(value)]
            best_extreme = max(extreme_values) if extreme_values else float("nan")
            mean_ic = _safe_float(rank_ic_mean[idx])
            std_ic = _safe_float(rank_ic_std[idx])
            positive_ic_rate = _safe_float(rank_ic_positive_rate[idx])
            if not math.isfinite(mean_ic):
                mean_ic = 0.0
            rows.append(
                {
                    "scope": scope,
                    "scope_value": scope_value,
                    "target_kind": target_kind,
                    "label_horizon": int(label_horizon),
                    "exit_label": _exit_label(int(label_horizon)),
                    "feature_index": int(idx),
                    "feature": feature,
                    "semantic_type": semantics.semantic_type,
                    "source_family": semantics.source_family,
                    "value_kind": semantics.value_kind,
                    "preferred_diagnostic": semantics.preferred_diagnostic,
                    "ordinal_rank_profile": bool(semantics.ordinal_rank_profile),
                    "interpretation_note": semantics.interpretation_note,
                    "date_count": int(self.date_count),
                    "row_count": int(self.row_count),
                    "finite_rate": float(finite_rate[idx]),
                    "rank_ic_mean": mean_ic,
                    "rank_ic_std": std_ic,
                    "rank_ic_positive_date_rate": positive_ic_rate,
                    "direction": "high_is_good" if mean_ic >= 0.0 else "low_is_good",
                    "feature_rank_all_mean": float(all_rank_mean[idx]),
                    "future_top5_feature_rank_mean": float(top5_rank_mean[idx]),
                    "future_top5_lift_vs_all": float(top5_rank_mean[idx] - all_rank_mean[idx]),
                    "future_bottom5_feature_rank_mean": float(bottom5_rank_mean[idx]),
                    "future_bottom5_lift_vs_all": float(bottom5_rank_mean[idx] - all_rank_mean[idx]),
                    "future_top10_feature_rank_mean": float(top10_rank_mean[idx]),
                    "future_top10_lift_vs_all": float(top10_rank_mean[idx] - all_rank_mean[idx]),
                    "future_bottom10_feature_rank_mean": float(bottom10_rank_mean[idx]),
                    "future_bottom10_lift_vs_all": float(bottom10_rank_mean[idx] - all_rank_mean[idx]),
                    "positive_return_feature_rank_mean": float(positive_rank_mean[idx]),
                    "negative_return_feature_rank_mean": float(negative_rank_mean[idx]),
                    "positive_minus_negative_feature_rank": float(positive_rank_mean[idx] - negative_rank_mean[idx]),
                    "top_decile_target_mean": top_decile_mean,
                    "bottom_decile_target_mean": bottom_decile_mean,
                    "top_minus_bottom_decile_target_mean": float(signed_spread),
                    "best_extreme_decile_target_mean": best_extreme,
                    "top_decile_hit_rate": top_decile_hit,
                    "bottom_decile_hit_rate": bottom_decile_hit,
                    "top_decile_lift_vs_all": float(top_decile_mean - target_mean) if math.isfinite(top_decile_mean) else float("nan"),
                    "bottom_decile_lift_vs_all": float(bottom_decile_mean - target_mean) if math.isfinite(bottom_decile_mean) else float("nan"),
                    "decile_monotonic_corr": monotonic_corr,
                    "abs_rank_ic_mean": abs(mean_ic),
                    "abs_top_bottom_decile_spread": abs(float(signed_spread)) if math.isfinite(signed_spread) else float("nan"),
                }
            )
        return rows

    def decile_rows(self, *, scope: str, scope_value: str, target_kind: str, label_horizon: int) -> list[dict[str, Any]]:
        decile_mean = _safe_div(self.decile_target_sum, self.decile_count)
        decile_hit_rate = _safe_div(self.decile_hit_sum, self.decile_count)
        target_mean = float(self.target_sum / self.row_count) if self.row_count else float("nan")
        rows: list[dict[str, Any]] = []
        for feature_idx, feature in enumerate(self.feature_columns):
            semantics = self.feature_semantics[feature_idx]
            for decile in range(self.n_deciles):
                target_value = float(decile_mean[feature_idx, decile]) if np.isfinite(decile_mean[feature_idx, decile]) else float("nan")
                rows.append(
                    {
                        "scope": scope,
                        "scope_value": scope_value,
                        "target_kind": target_kind,
                        "label_horizon": int(label_horizon),
                        "exit_label": _exit_label(int(label_horizon)),
                        "feature_index": int(feature_idx),
                        "feature": feature,
                        "semantic_type": semantics.semantic_type,
                        "source_family": semantics.source_family,
                        "decile": int(decile),
                        "count": int(self.decile_count[feature_idx, decile]),
                        "target_mean": target_value,
                        "target_lift_vs_all": float(target_value - target_mean) if math.isfinite(target_value) else float("nan"),
                        "hit_rate": float(decile_hit_rate[feature_idx, decile]) if np.isfinite(decile_hit_rate[feature_idx, decile]) else float("nan"),
                    }
                )
        return rows

    def categorical_rows(self, *, scope: str, scope_value: str, target_kind: str, label_horizon: int) -> list[dict[str, Any]]:
        target_mean = float(self.target_sum / self.row_count) if self.row_count else float("nan")
        rows: list[dict[str, Any]] = []
        by_feature: dict[int, list[tuple[str, float, float, float]]] = defaultdict(list)
        for (feature_idx, category), count in self.category_count.items():
            by_feature[int(feature_idx)].append(
                (
                    str(category),
                    float(count),
                    float(self.category_target_sum[(feature_idx, category)]),
                    float(self.category_hit_sum[(feature_idx, category)]),
                )
            )
        for feature_idx, items in sorted(by_feature.items()):
            semantics = self.feature_semantics[feature_idx]
            if not items:
                continue
            parsed_items: list[tuple[float, str, float, float, float]] = []
            for category, count, target_sum, hit_sum in items:
                try:
                    sort_value = float(category)
                except ValueError:
                    sort_value = float("inf")
                parsed_items.append((sort_value, category, count, target_sum, hit_sum))
            parsed_items.sort(key=lambda item: (item[0], item[1]))
            means = [target_sum / count for _, _, count, target_sum, _ in parsed_items if count > 0]
            best_mean = max(means) if means else float("nan")
            worst_mean = min(means) if means else float("nan")
            for _, category, count, target_sum, hit_sum in parsed_items:
                target_value = target_sum / count if count > 0 else float("nan")
                rows.append(
                    {
                        "scope": scope,
                        "scope_value": scope_value,
                        "target_kind": target_kind,
                        "label_horizon": int(label_horizon),
                        "exit_label": _exit_label(int(label_horizon)),
                        "feature_index": int(feature_idx),
                        "feature": self.feature_columns[feature_idx],
                        "semantic_type": semantics.semantic_type,
                        "source_family": semantics.source_family,
                        "category_value": category,
                        "count": int(count),
                        "target_mean": float(target_value),
                        "target_lift_vs_all": float(target_value - target_mean) if math.isfinite(target_value) else float("nan"),
                        "hit_rate": float(hit_sum / count) if count > 0 else float("nan"),
                        "best_category_target_mean": float(best_mean),
                        "worst_category_target_mean": float(worst_mean),
                        "category_spread": float(best_mean - worst_mean) if math.isfinite(best_mean) and math.isfinite(worst_mean) else float("nan"),
                    }
                )
        return rows

    def market_rows(self, *, scope: str, scope_value: str, target_kind: str, label_horizon: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for feature_idx in self._market_feature_indices:
            count = float(self.market_valid_date_count[feature_idx])
            if count < 3:
                corr = float("nan")
            else:
                feature_mean = self.market_feature_sum[feature_idx] / count
                target_mean = self.market_target_sum[feature_idx] / count
                cov = self.market_feature_target_sum[feature_idx] / count - feature_mean * target_mean
                feature_var = self.market_feature_sq_sum[feature_idx] / count - feature_mean * feature_mean
                target_var = self.market_target_sq_sum[feature_idx] / count - target_mean * target_mean
                denom = math.sqrt(max(feature_var, 0.0) * max(target_var, 0.0))
                corr = cov / denom if denom > 0.0 else float("nan")
            semantics = self.feature_semantics[feature_idx]
            rows.append(
                {
                    "scope": scope,
                    "scope_value": scope_value,
                    "target_kind": target_kind,
                    "label_horizon": int(label_horizon),
                    "exit_label": _exit_label(int(label_horizon)),
                    "feature_index": int(feature_idx),
                    "feature": self.feature_columns[feature_idx],
                    "semantic_type": semantics.semantic_type,
                    "source_family": semantics.source_family,
                    "date_count": int(count),
                    "date_level_corr": float(corr) if math.isfinite(corr) else float("nan"),
                    "avg_unique_values_per_date": float(self.market_value_unique_sum[feature_idx] / count) if count > 0 else float("nan"),
                    "interpretation_note": semantics.interpretation_note,
                }
            )
        return rows

    def target_row(self, *, scope: str, scope_value: str, target_kind: str, label_horizon: int) -> dict[str, Any]:
        mean = float(self.target_sum / self.row_count) if self.row_count else float("nan")
        variance = self.target_sq_sum / self.row_count - mean * mean if self.row_count else float("nan")
        return {
            "scope": scope,
            "scope_value": scope_value,
            "target_kind": target_kind,
            "label_horizon": int(label_horizon),
            "exit_label": _exit_label(int(label_horizon)),
            "date_count": int(self.date_count),
            "row_count": int(self.row_count),
            "target_mean": mean,
            "target_std": math.sqrt(max(float(variance), 0.0)) if math.isfinite(float(variance)) else float("nan"),
            "hit_rate": float(self.target_hit_sum / self.row_count) if self.row_count else float("nan"),
            "future_top5_target_mean": float(self.target_top5_sum / self.target_top5_count) if self.target_top5_count else float("nan"),
            "future_bottom5_target_mean": float(self.target_bottom5_sum / self.target_bottom5_count) if self.target_bottom5_count else float("nan"),
            "future_top10_target_mean": float(self.target_top10_sum / self.target_top10_count) if self.target_top10_count else float("nan"),
            "future_bottom10_target_mean": float(self.target_bottom10_sum / self.target_bottom10_count) if self.target_bottom10_count else float("nan"),
        }


def _get_accumulator(
    accumulators: dict[tuple[str, str, str, int], FeatureProfileAccumulator],
    *,
    key: tuple[str, str, str, int],
    feature_columns: tuple[str, ...],
    feature_semantics: tuple[FeatureSemantics, ...],
    n_deciles: int,
) -> FeatureProfileAccumulator:
    if key not in accumulators:
        accumulators[key] = FeatureProfileAccumulator(
            feature_columns=feature_columns,
            feature_semantics=feature_semantics,
            n_deciles=n_deciles,
        )
    return accumulators[key]


def _target_from_columns(frame: pd.DataFrame, *, target_kind: str, label_horizon: int, round_trip_cost_bps: float) -> np.ndarray:
    cost = float(round_trip_cost_bps) / 10000.0
    if target_kind == "net_abs":
        column = f"future_cum_return_{int(label_horizon)}d"
    elif target_kind == "net_excess":
        column = f"future_cum_excess_return_{int(label_horizon)}d"
    else:
        raise ValueError(f"unsupported target kind: {target_kind}")
    if column not in frame.columns:
        return np.full((len(frame),), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64) - cost


def _build_outputs(
    *,
    accumulators: dict[tuple[str, str, str, int], FeatureProfileAccumulator],
    output_root: Path,
    report_base: dict[str, Any],
    leaderboard_limit: int,
) -> dict[str, Any]:
    feature_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []
    market_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for (scope, scope_value, target_kind, label_horizon), accumulator in sorted(accumulators.items()):
        feature_rows.extend(
            accumulator.feature_rows(
                scope=scope,
                scope_value=scope_value,
                target_kind=target_kind,
                label_horizon=label_horizon,
            )
        )
        decile_rows.extend(
            accumulator.decile_rows(
                scope=scope,
                scope_value=scope_value,
                target_kind=target_kind,
                label_horizon=label_horizon,
            )
        )
        categorical_rows.extend(
            accumulator.categorical_rows(
                scope=scope,
                scope_value=scope_value,
                target_kind=target_kind,
                label_horizon=label_horizon,
            )
        )
        market_rows.extend(
            accumulator.market_rows(
                scope=scope,
                scope_value=scope_value,
                target_kind=target_kind,
                label_horizon=label_horizon,
            )
        )
        target_rows.append(
            accumulator.target_row(
                scope=scope,
                scope_value=scope_value,
                target_kind=target_kind,
                label_horizon=label_horizon,
            )
        )
    feature_metrics = pd.DataFrame(feature_rows)
    decile_profile = pd.DataFrame(decile_rows)
    categorical_profile = pd.DataFrame(categorical_rows)
    market_profile = pd.DataFrame(market_rows)
    target_summary = pd.DataFrame(target_rows)
    feature_semantics = _feature_semantics_frame(
        next(iter(accumulators.values())).feature_semantics if accumulators else ()
    )
    categorical_summary = _categorical_summary(categorical_profile)
    if not feature_metrics.empty:
        feature_metrics["selection_strength"] = feature_metrics["abs_rank_ic_mean"].fillna(0.0) + 2.0 * feature_metrics[
            "abs_top_bottom_decile_spread"
        ].fillna(0.0)
        leaderboard = feature_metrics.sort_values(
            ["scope", "scope_value", "target_kind", "label_horizon", "selection_strength"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
        stable_outputs = _stable_feature_outputs(feature_metrics)
    else:
        leaderboard = pd.DataFrame()
        stable_outputs = {
            "stable_same_direction_features": pd.DataFrame(),
            "common_stable_features_all_targets": pd.DataFrame(),
            "stable_feature_counts": pd.DataFrame(),
        }
    feature_metrics_csv = _write_frame(output_root / "shortline_upside_feature_metrics.csv", feature_metrics)
    decile_profile_csv = _write_frame(output_root / "shortline_upside_decile_profile.csv", decile_profile)
    categorical_profile_csv = _write_frame(output_root / "shortline_upside_categorical_profile.csv", categorical_profile)
    categorical_summary_csv = _write_frame(output_root / "shortline_upside_categorical_summary.csv", categorical_summary)
    market_profile_csv = _write_frame(output_root / "shortline_upside_market_level_profile.csv", market_profile)
    feature_semantics_csv = _write_frame(output_root / "shortline_upside_feature_semantics.csv", feature_semantics)
    target_summary_csv = _write_frame(output_root / "shortline_upside_target_summary.csv", target_summary)
    leaderboard_csv = _write_frame(output_root / "shortline_upside_feature_leaderboard.csv", leaderboard)
    stable_same_direction_csv = _write_frame(
        output_root / "shortline_upside_stable_same_direction_features.csv",
        stable_outputs["stable_same_direction_features"],
    )
    common_stable_csv = _write_frame(
        output_root / "shortline_upside_common_stable_features_all_targets.csv",
        stable_outputs["common_stable_features_all_targets"],
    )
    stable_counts_csv = _write_frame(
        output_root / "shortline_upside_stable_feature_counts.csv",
        stable_outputs["stable_feature_counts"],
    )
    semantic_counts = (
        feature_semantics.groupby(["semantic_type", "source_family"], dropna=False)
        .size()
        .reset_index(name="feature_count")
        .sort_values(["semantic_type", "source_family"], kind="mergesort")
        .to_dict("records")
        if not feature_semantics.empty
        else []
    )
    report = {
        **report_base,
        "status": "completed" if not feature_metrics.empty else "empty",
        "created_at": _now(),
        "semantic_counts": semantic_counts,
        "outputs": {
            "feature_metrics_csv": feature_metrics_csv,
            "decile_profile_csv": decile_profile_csv,
            "categorical_profile_csv": categorical_profile_csv,
            "categorical_summary_csv": categorical_summary_csv,
            "market_profile_csv": market_profile_csv,
            "feature_semantics_csv": feature_semantics_csv,
            "target_summary_csv": target_summary_csv,
            "feature_leaderboard_csv": leaderboard_csv,
            "stable_same_direction_features_csv": stable_same_direction_csv,
            "common_stable_features_all_targets_csv": common_stable_csv,
            "stable_feature_counts_csv": stable_counts_csv,
        },
        "leaderboard": _leaderboard_rows(feature_metrics, limit=leaderboard_limit),
    }
    report_json = _write_json(output_root / "shortline_upside_feature_profile_report.json", report)
    report["outputs"]["report_json"] = report_json
    _write_markdown(output_root / "shortline_upside_feature_profile_report.md", report)
    report["outputs"]["report_md"] = str((output_root / "shortline_upside_feature_profile_report.md").resolve())
    _write_json(output_root / "shortline_upside_feature_profile_report.json", report)
    return report


def _signed_direction(value: Any) -> str:
    out = _safe_float(value)
    if not math.isfinite(out) or abs(out) <= 1.0e-12:
        return "flat_or_nan"
    return "positive" if out > 0.0 else "negative"


def _stable_feature_outputs(feature_metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    role_metrics = feature_metrics[
        feature_metrics["scope"].astype(str).eq("role")
        & feature_metrics["scope_value"].astype(str).isin(["train", "validation", "test"])
    ].copy()
    if role_metrics.empty:
        empty = pd.DataFrame()
        return {
            "stable_same_direction_features": empty,
            "common_stable_features_all_targets": empty,
            "stable_feature_counts": empty,
        }
    role_metrics["rank_ic_sign"] = role_metrics["rank_ic_mean"].map(_signed_direction)
    role_metrics["spread_sign"] = role_metrics["top_minus_bottom_decile_target_mean"].map(_signed_direction)
    keys = [
        "target_kind",
        "label_horizon",
        "exit_label",
        "feature_index",
        "feature",
        "semantic_type",
        "source_family",
        "value_kind",
        "preferred_diagnostic",
    ]
    rows: list[dict[str, Any]] = []
    for key_values, group in role_metrics.groupby(keys, dropna=False, sort=True):
        roles = set(group["scope_value"].astype(str).tolist())
        if roles != {"train", "validation", "test"}:
            continue
        rank_signs = set(group["rank_ic_sign"].astype(str).tolist())
        spread_signs = set(group["spread_sign"].astype(str).tolist())
        if len(rank_signs) != 1 or len(spread_signs) != 1:
            continue
        rank_sign = next(iter(rank_signs))
        spread_sign = next(iter(spread_signs))
        if rank_sign == "flat_or_nan" or spread_sign == "flat_or_nan":
            continue
        row = dict(zip(keys, key_values, strict=True))
        row.update(
            {
                "direction": "high_is_good" if rank_sign == "positive" else "low_is_good",
                "rank_ic_sign": rank_sign,
                "spread_sign": spread_sign,
                "train_rank_ic": float(group.loc[group["scope_value"].eq("train"), "rank_ic_mean"].iloc[0]),
                "validation_rank_ic": float(group.loc[group["scope_value"].eq("validation"), "rank_ic_mean"].iloc[0]),
                "test_rank_ic": float(group.loc[group["scope_value"].eq("test"), "rank_ic_mean"].iloc[0]),
                "train_top_bottom_spread": float(group.loc[group["scope_value"].eq("train"), "top_minus_bottom_decile_target_mean"].iloc[0]),
                "validation_top_bottom_spread": float(
                    group.loc[group["scope_value"].eq("validation"), "top_minus_bottom_decile_target_mean"].iloc[0]
                ),
                "test_top_bottom_spread": float(group.loc[group["scope_value"].eq("test"), "top_minus_bottom_decile_target_mean"].iloc[0]),
                "mean_abs_rank_ic": float(group["rank_ic_mean"].abs().mean()),
                "mean_abs_top_bottom_spread": float(group["top_minus_bottom_decile_target_mean"].abs().mean()),
            }
        )
        rows.append(row)
    stable = pd.DataFrame(rows)
    if stable.empty:
        empty = pd.DataFrame()
        return {
            "stable_same_direction_features": empty,
            "common_stable_features_all_targets": empty,
            "stable_feature_counts": empty,
        }
    stable = stable.sort_values(
        ["target_kind", "label_horizon", "mean_abs_rank_ic", "mean_abs_top_bottom_spread"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )
    counts = (
        stable.groupby(["target_kind", "label_horizon", "exit_label", "direction", "semantic_type"], dropna=False)
        .size()
        .reset_index(name="feature_count")
        .sort_values(["target_kind", "label_horizon", "direction", "semantic_type"], kind="mergesort")
    )
    combo_count = stable[["target_kind", "label_horizon"]].drop_duplicates().shape[0]
    common_features = (
        stable.groupby(["feature_index", "feature", "semantic_type", "source_family", "value_kind", "direction"], dropna=False)
        .agg(
            stable_combo_count=("target_kind", "count"),
            mean_abs_rank_ic=("mean_abs_rank_ic", "mean"),
            mean_abs_top_bottom_spread=("mean_abs_top_bottom_spread", "mean"),
        )
        .reset_index()
    )
    common = common_features.loc[common_features["stable_combo_count"].eq(combo_count)].sort_values(
        ["direction", "mean_abs_rank_ic", "mean_abs_top_bottom_spread"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    return {
        "stable_same_direction_features": stable,
        "common_stable_features_all_targets": common,
        "stable_feature_counts": counts,
    }


def _parse_category_number(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _categorical_summary(categorical_profile: pd.DataFrame) -> pd.DataFrame:
    if categorical_profile.empty:
        return pd.DataFrame()
    keys = [
        "scope",
        "scope_value",
        "target_kind",
        "label_horizon",
        "exit_label",
        "feature_index",
        "feature",
        "semantic_type",
        "source_family",
    ]
    rows: list[dict[str, Any]] = []
    for key_values, group in categorical_profile.groupby(keys, dropna=False, sort=True):
        work = group.copy()
        work["target_mean"] = pd.to_numeric(work["target_mean"], errors="coerce")
        work["count"] = pd.to_numeric(work["count"], errors="coerce")
        valid = work.loc[work["target_mean"].notna() & work["count"].gt(0)].copy()
        if valid.empty:
            continue
        best = valid.sort_values(["target_mean", "count"], ascending=[False, False], kind="mergesort").iloc[0]
        worst = valid.sort_values(["target_mean", "count"], ascending=[True, False], kind="mergesort").iloc[0]
        parsed = valid.assign(_category_num=valid["category_value"].map(_parse_category_number))
        numeric_valid = parsed.loc[parsed["_category_num"].notna()].copy()
        high_minus_low = float("nan")
        highest_category_value: str | float = ""
        lowest_category_value: str | float = ""
        if not numeric_valid.empty:
            highest = numeric_valid.sort_values(["_category_num", "count"], ascending=[False, False], kind="mergesort").iloc[0]
            lowest = numeric_valid.sort_values(["_category_num", "count"], ascending=[True, False], kind="mergesort").iloc[0]
            high_minus_low = float(highest["target_mean"] - lowest["target_mean"])
            highest_category_value = highest["category_value"]
            lowest_category_value = lowest["category_value"]
        by_value = {str(row["category_value"]): float(row["target_mean"]) for _, row in valid.iterrows()}
        one_minus_zero = by_value.get("1", float("nan")) - by_value.get("0", float("nan")) if "1" in by_value and "0" in by_value else float("nan")
        row = dict(zip(keys, key_values, strict=True))
        row.update(
            {
                "category_count": int(valid["category_value"].nunique()),
                "sample_count": int(valid["count"].sum()),
                "best_category_value": best["category_value"],
                "best_category_target_mean": float(best["target_mean"]),
                "worst_category_value": worst["category_value"],
                "worst_category_target_mean": float(worst["target_mean"]),
                "best_minus_worst_target_mean": float(best["target_mean"] - worst["target_mean"]),
                "highest_category_value": highest_category_value,
                "lowest_category_value": lowest_category_value,
                "highest_minus_lowest_target_mean": high_minus_low,
                "category_1_minus_0_target_mean": float(one_minus_zero) if math.isfinite(one_minus_zero) else float("nan"),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["scope", "scope_value", "target_kind", "label_horizon", "feature_index"],
        kind="mergesort",
    )


def _leaderboard_rows(feature_metrics: pd.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    if feature_metrics.empty:
        return []
    preferred = feature_metrics[
        (feature_metrics["scope"].astype(str).eq("role"))
        & (feature_metrics["scope_value"].astype(str).isin(["train", "validation", "test"]))
    ].copy()
    if preferred.empty:
        preferred = feature_metrics.copy()
    preferred["selection_strength"] = preferred["abs_rank_ic_mean"].fillna(0.0) + 2.0 * preferred[
        "abs_top_bottom_decile_spread"
    ].fillna(0.0)
    cols = [
        "scope",
        "scope_value",
        "target_kind",
        "label_horizon",
        "exit_label",
        "feature",
        "direction",
        "rank_ic_mean",
        "top_minus_bottom_decile_target_mean",
        "future_top10_lift_vs_all",
        "positive_minus_negative_feature_rank",
        "top_decile_target_mean",
        "bottom_decile_target_mean",
        "selection_strength",
    ]
    rows = preferred.sort_values("selection_strength", ascending=False, kind="mergesort")[cols].head(int(limit)).to_dict("records")
    return [{key: _json_default(value) for key, value in row.items()} for row in rows]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Shortline Upside Feature Profile",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Research-only: `{report.get('contract', {}).get('research_only', True)}`",
        f"- Not model training: `{report.get('contract', {}).get('not_model_training', True)}`",
        f"- Feature count: `{report.get('input', {}).get('feature_count', 0)}`",
        f"- Features are normalized: `{report.get('input', {}).get('features_are_normalized', False)}`",
        f"- Selected rows: `{report.get('input', {}).get('selected_row_count', 0)}`",
        f"- Date groups: `{report.get('input', {}).get('selected_group_count', 0)}`",
        "",
        "## Semantics",
        "",
        "- This profile asks which D-close features describe future shortline upside potential.",
        "- It does not choose topK portfolios and does not train a model.",
        "- Feature statistics use same-date cross-sectional percentile ranks across all selected features.",
        "- T+1 labels are open-to-open: horizon 1 means D+1 open entry to D+2 open exit.",
        "- `direction` is a statistical numeric-rank shorthand, not a universal economic high/low interpretation.",
        "- Type-aware outputs split flags, buckets, membership features, and market-level date context from continuous ordinal scans.",
        "- If features are normalized, categorical/bucket profiles group standardized feature values; they are diagnostic groups, not raw category ids.",
        "",
        "## Leaderboard",
        "",
    ]
    leaderboard = list(report.get("leaderboard", []) or [])
    if leaderboard:
        lines.append("| scope | target | exit | feature | dir | rank_ic | top-bottom | top10_lift |")
        lines.append("|---|---|---|---|---|---:|---:|---:|")
        for row in leaderboard[:20]:
            lines.append(
                "| {scope}:{scope_value} | {target_kind} | {exit_label} | {feature} | {direction} | {rank_ic:.5f} | {spread:.5f} | {top10:.5f} |".format(
                    scope=row.get("scope", ""),
                    scope_value=row.get("scope_value", ""),
                    target_kind=row.get("target_kind", ""),
                    exit_label=row.get("exit_label", ""),
                    feature=row.get("feature", ""),
                    direction=row.get("direction", ""),
                    rank_ic=_safe_float(row.get("rank_ic_mean")),
                    spread=_safe_float(row.get("top_minus_bottom_decile_target_mean")),
                    top10=_safe_float(row.get("future_top10_lift_vs_all")),
                )
            )
    else:
        lines.append("- No usable rows.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Feature metrics: `{report.get('outputs', {}).get('feature_metrics_csv', '')}`",
            f"- Decile profile: `{report.get('outputs', {}).get('decile_profile_csv', '')}`",
            f"- Feature semantics: `{report.get('outputs', {}).get('feature_semantics_csv', '')}`",
            f"- Categorical profile: `{report.get('outputs', {}).get('categorical_profile_csv', '')}`",
            f"- Categorical summary: `{report.get('outputs', {}).get('categorical_summary_csv', '')}`",
            f"- Market-level profile: `{report.get('outputs', {}).get('market_profile_csv', '')}`",
            f"- Target summary: `{report.get('outputs', {}).get('target_summary_csv', '')}`",
            f"- Leaderboard: `{report.get('outputs', {}).get('feature_leaderboard_csv', '')}`",
            f"- Report JSON: `{report.get('outputs', {}).get('report_json', '')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_shortline_upside_feature_profile_from_frame(
    *,
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    output_root: str | Path,
    run_tag: str = "shortline_upside_feature_profile_frame",
    roles: str | Iterable[str] | None = None,
    label_horizons: str | Iterable[int] | None = None,
    target_kinds: str | Iterable[str] | None = None,
    group_scopes: str | Iterable[str] | None = None,
    round_trip_cost_bps: float = 20.0,
    n_deciles: int = 10,
    leaderboard_limit: int = 40,
) -> dict[str, Any]:
    resolved_features = tuple(str(item) for item in feature_columns)
    resolved_roles = _parse_csv_strings(roles, default=tuple(sorted(frame["role"].astype(str).unique().tolist())) if "role" in frame.columns else ("all",))
    resolved_horizons = _parse_csv_ints(label_horizons, default=DEFAULT_LABEL_HORIZONS)
    resolved_targets = _parse_csv_strings(target_kinds, default=DEFAULT_TARGET_KINDS)
    resolved_scopes = _parse_csv_strings(group_scopes, default=DEFAULT_GROUP_SCOPES)
    feature_semantics = tuple(classify_feature_semantics(item) for item in resolved_features)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    if "role" not in work.columns:
        work["role"] = "all"
    work["role"] = work["role"].astype(str)
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work = work.loc[work["role"].isin(set(resolved_roles))].copy()
    accumulators: dict[tuple[str, str, str, int], FeatureProfileAccumulator] = {}
    for (role, date), group in work.groupby(["role", "date"], sort=True, dropna=False):
        features = group.loc[:, list(resolved_features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        feature_ranks, feature_finite = _feature_percentile_ranks(features)
        scopes = _row_scopes(str(role), str(date), resolved_scopes)
        for target_kind in resolved_targets:
            for horizon in resolved_horizons:
                y = _target_from_columns(group, target_kind=target_kind, label_horizon=int(horizon), round_trip_cost_bps=float(round_trip_cost_bps))
                for scope, scope_value in scopes:
                    key = (scope, scope_value, str(target_kind), int(horizon))
                    accumulator = _get_accumulator(
                        accumulators,
                        key=key,
                        feature_columns=resolved_features,
                        feature_semantics=feature_semantics,
                        n_deciles=int(n_deciles),
                    )
                    accumulator.update(features=features, feature_ranks=feature_ranks, feature_finite=feature_finite, target=y)
    report_base = {
        "run_tag": str(run_tag),
        "contract": {
            "diagnostic_kind": "shortline_after_close_upside_feature_profile",
            "research_only": True,
            "not_model_training": True,
            "active_artifact_impact": "unchanged",
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "roles": list(resolved_roles),
            "label_horizons": [int(item) for item in resolved_horizons],
            "target_kinds": list(resolved_targets),
            "group_scopes": list(resolved_scopes),
            "n_deciles": int(n_deciles),
            "feature_semantics": "type-aware diagnostics are emitted for categorical/bucket/flag and market-level features; rank/decile remains available as numeric ordinal scan.",
        },
        "input": {
            "selected_row_count": int(len(work)),
            "selected_group_count": int(work.groupby(["role", "date"]).ngroups if not work.empty else 0),
            "feature_count": int(len(resolved_features)),
            "features_are_normalized": False,
            "source": "frame",
        },
    }
    return _build_outputs(accumulators=accumulators, output_root=root, report_base=report_base, leaderboard_limit=int(leaderboard_limit))


def _open_label_array(manifest: dict[str, Any], name: str, *, root: Path) -> np.memmap:
    meta = dict(dict(manifest.get("label_arrays", {}) or {}).get(name, {}) or {})
    if not meta:
        raise ValueError(f"training pack missing required label array: {name}")
    path = _resolve_path(str(meta.get("path", "") or ""), root=root)
    shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
    dtype = str(meta.get("dtype", "float32") or "float32")
    if not path.exists():
        raise FileNotFoundError(f"label array not found: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _load_feature_panel(manifest: dict[str, Any], *, root: Path) -> np.memmap:
    meta = dict(manifest.get("date_major_feature_panel", {}) or {})
    path_raw = str(meta.get("path", "") or manifest.get("date_major_feature_panel_path", "") or "")
    shape_raw = meta.get("shape") or manifest.get("date_major_feature_panel_shape") or []
    dtype = str(meta.get("dtype") or manifest.get("date_major_feature_dtype") or manifest.get("feature_dtype", "float16") or "float16")
    path = _resolve_path(path_raw, root=root)
    shape = tuple(int(item) for item in list(shape_raw or []))
    if len(shape) != 3:
        raise ValueError("training pack missing date-major feature panel shape.")
    if not path.exists():
        raise FileNotFoundError(f"date-major feature panel not found: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def _choose_dates(sample_index: pd.DataFrame, *, role: str, max_dates: int) -> list[str]:
    dates = sorted(sample_index.loc[sample_index["role"].astype(str).eq(str(role)), "date"].astype(str).unique().tolist())
    if int(max_dates) <= 0:
        return dates
    return dates[: int(max_dates)]


def _target_from_arrays(
    *,
    labels: dict[str, np.memmap],
    sample_pos: np.ndarray,
    target_kind: str,
    label_horizon: int,
    round_trip_cost_bps: float,
) -> np.ndarray:
    label_idx = int(label_horizon) - 1
    cost = float(round_trip_cost_bps) / 10000.0
    if target_kind == "net_abs":
        source = labels["cumulative_return_1to20"]
    elif target_kind == "net_excess":
        source = labels["cumulative_excess_return_1to20"]
    else:
        raise ValueError(f"unsupported target kind: {target_kind}")
    if label_idx < 0 or label_idx >= int(source.shape[1]):
        return np.full((len(sample_pos),), np.nan, dtype=np.float64)
    return np.asarray(source[sample_pos, label_idx], dtype=np.float64) - cost


def build_shortline_upside_feature_profile(
    *,
    manifest_json: str | Path = DEFAULT_TRAINING_PACK_MANIFEST,
    output_root: str | Path | None = None,
    run_tag: str = "shortline_upside_feature_profile",
    roles: str | Iterable[str] | None = None,
    max_dates_per_role: int = 0,
    label_horizons: str | Iterable[int] | None = None,
    target_kinds: str | Iterable[str] | None = None,
    group_scopes: str | Iterable[str] | None = None,
    round_trip_cost_bps: float = 20.0,
    n_deciles: int = 10,
    leaderboard_limit: int = 40,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    if str(manifest.get("artifact_type", "")) != "qdp_training_pack_v1":
        raise ValueError(f"upside feature profile requires qdp_training_pack_v1: {manifest_path}")
    resolved_roles = _parse_csv_strings(roles, default=DEFAULT_ROLES)
    resolved_horizons = _parse_csv_ints(label_horizons, default=DEFAULT_LABEL_HORIZONS)
    resolved_targets = _parse_csv_strings(target_kinds, default=DEFAULT_TARGET_KINDS)
    resolved_scopes = _parse_csv_strings(group_scopes, default=DEFAULT_GROUP_SCOPES)
    feature_columns = tuple(str(item) for item in list(manifest.get("feature_columns", []) or []))
    if not feature_columns:
        raise ValueError("training pack manifest has no feature_columns.")
    feature_semantics = tuple(classify_feature_semantics(item) for item in feature_columns)
    output_dir = Path(output_root) if output_root is not None else Path("daily_research/output/path_policy/shortline_upside_feature_profile") / str(run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_index_path = _resolve_path(str(manifest.get("sample_index_path", "") or ""), root=root)
    sample_index = pd.read_parquet(sample_index_path)
    sample_index["date"] = pd.to_datetime(sample_index["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    selected_dates_by_role = {
        role: set(_choose_dates(sample_index, role=role, max_dates=int(max_dates_per_role)))
        for role in resolved_roles
    }
    mask = sample_index["role"].astype(str).isin(set(resolved_roles))
    if int(max_dates_per_role) > 0:
        date_allowed = pd.Series(False, index=sample_index.index)
        for role, dates in selected_dates_by_role.items():
            date_allowed |= sample_index["role"].astype(str).eq(role) & sample_index["date"].isin(dates)
        mask &= date_allowed
    selected_index = sample_index.loc[mask].copy()
    if selected_index.empty:
        raise ValueError(f"no samples selected for roles={resolved_roles}, max_dates_per_role={max_dates_per_role}")

    feature_panel = _load_feature_panel(manifest, root=root)
    labels = {
        "cumulative_return_1to20": _open_label_array(manifest, "cumulative_return_1to20", root=root),
        "cumulative_excess_return_1to20": _open_label_array(manifest, "cumulative_excess_return_1to20", root=root),
    }
    accumulators: dict[tuple[str, str, str, int], FeatureProfileAccumulator] = {}
    grouped = list(selected_index.groupby(["role", "date"], sort=True))
    progress_path = output_dir / "shortline_upside_feature_profile_progress.json"
    processed_groups = 0
    processed_rows = 0
    for (role, date), group in grouped:
        sample_pos = pd.to_numeric(group["_sample_pos"], errors="coerce").to_numpy(dtype=np.int64, copy=True)
        date_pos_values = pd.to_numeric(group["global_date_pos"], errors="coerce").to_numpy(dtype=np.int64, copy=True)
        stock_pos = pd.to_numeric(group["global_stock_pos"], errors="coerce").to_numpy(dtype=np.int64, copy=True)
        if len(np.unique(date_pos_values)) != 1:
            raise ValueError("date group contains multiple global_date_pos values.")
        date_pos = int(date_pos_values[0])
        features = np.asarray(feature_panel[date_pos, stock_pos[:, None], np.arange(len(feature_columns), dtype=np.int64)[None, :]], dtype=np.float32)
        feature_ranks, feature_finite = _feature_percentile_ranks(features)
        scopes = _row_scopes(str(role), str(date), resolved_scopes)
        for target_kind in resolved_targets:
            for horizon in resolved_horizons:
                target = _target_from_arrays(
                    labels=labels,
                    sample_pos=sample_pos,
                    target_kind=str(target_kind),
                    label_horizon=int(horizon),
                    round_trip_cost_bps=float(round_trip_cost_bps),
                )
                for scope, scope_value in scopes:
                    key = (scope, scope_value, str(target_kind), int(horizon))
                    accumulator = _get_accumulator(
                        accumulators,
                        key=key,
                        feature_columns=feature_columns,
                        feature_semantics=feature_semantics,
                        n_deciles=int(n_deciles),
                    )
                    accumulator.update(features=features, feature_ranks=feature_ranks, feature_finite=feature_finite, target=target)
        processed_groups += 1
        processed_rows += int(len(group))
        if processed_groups % 50 == 0 or processed_groups == len(grouped):
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "processed_groups": int(processed_groups),
                    "total_groups": int(len(grouped)),
                    "processed_rows": int(processed_rows),
                    "selected_rows": int(len(selected_index)),
                },
            )
    report_base = {
        "run_tag": str(run_tag),
        "manifest_json": str(manifest_path.resolve()),
        "output_root": str(output_dir.resolve()),
        "contract": {
            "diagnostic_kind": "shortline_after_close_upside_feature_profile",
            "research_only": True,
            "not_model_training": True,
            "active_artifact_impact": "unchanged",
            "entry_semantics": "D close signal; future labels use D+1 open fixed-entry baseline from QDP pack",
            "t_plus_1_semantics": "label_horizon=1 means D+1 open entry to D+2 open exit; label_horizon=2 means D+3 open exit; label_horizon=4 means D+5 open exit",
            "do_not_claim": "D+1 open to D+1 close is not executable under A-share T+1",
            "profile_semantics": "all feature statistics use same-date cross-sectional percentile ranks; this diagnoses upside potential before topK/entry/exit policy selection",
            "feature_semantics": "rank/decile outputs are numeric ordinal scans; categorical/bucket/flag and market-level outputs are emitted separately to avoid semantic over-interpretation.",
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "roles": list(resolved_roles),
            "label_horizons": [int(item) for item in resolved_horizons],
            "target_kinds": list(resolved_targets),
            "group_scopes": list(resolved_scopes),
            "n_deciles": int(n_deciles),
            "max_dates_per_role": int(max_dates_per_role),
        },
        "input": {
            "selected_row_count": int(len(selected_index)),
            "selected_group_count": int(len(grouped)),
            "roles": list(resolved_roles),
            "selected_dates_by_role": {role: sorted(dates) for role, dates in selected_dates_by_role.items()},
            "feature_count": int(len(feature_columns)),
            "features_are_normalized": bool(manifest.get("features_are_normalized", False)),
            "normalization_method": str(dict(manifest.get("normalization", {}) or {}).get("method", "")),
            "normalization_fit_role": str(dict(manifest.get("normalization", {}) or {}).get("fit_role", "")),
            "label_schema_name": str(manifest.get("label_schema_name", "")),
            "label_schema_version": int(manifest.get("label_schema_version", 0) or 0),
            "source": "qdp_training_pack",
        },
    }
    report = _build_outputs(accumulators=accumulators, output_root=output_dir, report_base=report_base, leaderboard_limit=int(leaderboard_limit))
    _write_json(progress_path, {"status": "completed", "report_json": report.get("outputs", {}).get("report_json", "")})
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile all QDP alpha_v2 features against shortline upside potential labels.")
    parser.add_argument("--manifest-json", default=str(DEFAULT_TRAINING_PACK_MANIFEST))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-tag", default="shortline_upside_feature_profile")
    parser.add_argument("--roles", default="train,validation,test")
    parser.add_argument("--max-dates-per-role", type=int, default=0)
    parser.add_argument("--label-horizons", default="1,2,4")
    parser.add_argument("--target-kinds", default="net_abs,net_excess")
    parser.add_argument("--group-scopes", default="all,role,year,role_year")
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--n-deciles", type=int, default=10)
    parser.add_argument("--leaderboard-limit", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_shortline_upside_feature_profile(
        manifest_json=args.manifest_json,
        output_root=args.output_root or None,
        run_tag=args.run_tag,
        roles=args.roles,
        max_dates_per_role=int(args.max_dates_per_role),
        label_horizons=args.label_horizons,
        target_kinds=args.target_kinds,
        group_scopes=args.group_scopes,
        round_trip_cost_bps=float(args.round_trip_cost_bps),
        n_deciles=int(args.n_deciles),
        leaderboard_limit=int(args.leaderboard_limit),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(report.get("outputs", {}).get("report_json", ""))
    return report


if __name__ == "__main__":
    main()
