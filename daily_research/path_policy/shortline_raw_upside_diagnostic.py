from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_SHARDED_MANIFEST = Path(
    "quant_data_platform/data/memmap/sharded/"
    "mainboard_style_structural_alpha_v2_label_v2_full_2010_2026_20260616_01/"
    "sharded_memmap_manifest.json"
)
DEFAULT_ROLES = ("train", "validation", "test")
DEFAULT_ROLE_YEARS = {
    "train": tuple(range(2012, 2024)),
    "validation": (2024,),
    "test": (2025,),
}
DEFAULT_LABEL_HORIZONS = (1, 2, 4)
DEFAULT_TARGET_KINDS = ("net_abs", "net_excess")
DEFAULT_FEATURES = (
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "turn",
    "turn_z20",
    "raw_open_gap_1d",
    "raw_intraday_range_1d",
    "raw_body_to_range_1d",
    "raw_upper_shadow_to_range_1d",
    "raw_lower_shadow_to_range_1d",
    "raw_limit_up_like_1d",
    "raw_limit_down_like_1d",
    "distance_to_20d_high",
    "distance_to_20d_low",
    "local_drawdown_20d",
    "intraday_first_30m_ret",
    "intraday_last_30m_ret",
    "intraday_close_to_vwap",
    "intraday_close_position",
    "intraday_low_time_frac",
    "intraday_high_before_low",
    "intraday_intraday_range",
    "intraday_intraday_realized_vol",
    "market_positive_share_1d",
    "market_above_ma20_share",
    "market_amount_expansion_share",
    "industry_ret_5_excess",
    "industry_ret_20_excess",
    "stock_ret_5_minus_industry",
    "stock_ret_20_minus_industry",
)


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return str(path.resolve())


def _write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


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


def _parse_role_years(raw: str | None) -> dict[str, tuple[int, ...]]:
    if raw is None or not str(raw).strip():
        return dict(DEFAULT_ROLE_YEARS)
    out: dict[str, tuple[int, ...]] = {}
    for chunk in str(raw).split(";"):
        if not chunk.strip():
            continue
        role, _, years_raw = chunk.partition(":")
        years: list[int] = []
        for item in years_raw.split(","):
            token = item.strip()
            if not token:
                continue
            if "-" in token:
                start, end = [int(part) for part in token.split("-", 1)]
                years.extend(range(start, end + 1))
            else:
                years.append(int(token))
        if role.strip() and years:
            out[role.strip()] = tuple(sorted(set(years)))
    return out or dict(DEFAULT_ROLE_YEARS)


def _exit_label(label_horizon: int) -> str:
    return f"D+{int(label_horizon) + 1}_open"


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _target_array(arrays: Mapping[str, np.memmap], *, target_kind: str, label_horizon: int, cost_bps: float) -> np.ndarray:
    idx = int(label_horizon) - 1
    if target_kind == "net_abs":
        source = arrays["cumulative_return_1to20"]
    elif target_kind == "net_excess":
        source = arrays["cumulative_excess_return_1to20"]
    else:
        raise ValueError(f"unsupported target kind: {target_kind}")
    if idx < 0 or idx >= int(source.shape[2]):
        raise ValueError(f"label_horizon={label_horizon} outside label shape {source.shape}")
    return np.asarray(source[:, :, idx], dtype=np.float32) - float(cost_bps) / 10000.0


def _open_label_arrays(label_manifest: Mapping[str, Any], names: Iterable[str]) -> dict[str, np.memmap]:
    arrays_meta = dict(label_manifest.get("arrays", {}) or {})
    out: dict[str, np.memmap] = {}
    for name in names:
        meta = dict(arrays_meta.get(str(name), {}) or {})
        if not meta:
            raise ValueError(f"label manifest missing array: {name}")
        path = Path(str(meta.get("path", "") or ""))
        shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
        dtype = str(meta.get("dtype", "float32") or "float32")
        out[str(name)] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    return out


def _role_for_year(year: int, role_years: Mapping[str, tuple[int, ...]]) -> str | None:
    for role, years in role_years.items():
        if int(year) in set(int(item) for item in years):
            return str(role)
    return None


def _selected_shards(manifest: Mapping[str, Any], *, role_years: Mapping[str, tuple[int, ...]], roles: tuple[str, ...]) -> list[dict[str, Any]]:
    allowed_roles = set(roles)
    selected: list[dict[str, Any]] = []
    for shard in list(manifest.get("shards", []) or []):
        if str(shard.get("status", "")) != "completed":
            continue
        year = int(shard.get("year", 0))
        role = _role_for_year(year, role_years)
        if role is None or role not in allowed_roles:
            continue
        row = dict(shard)
        row["_role"] = role
        selected.append(row)
    return selected


@dataclass
class RawFeatureAccumulator:
    feature: str
    n_bins: int = 10
    max_samples: int = 250_000

    def __post_init__(self) -> None:
        self.total_count = 0
        self.finite_count = 0
        self.feature_sum = 0.0
        self.feature_sq_sum = 0.0
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.feature_target_sum = 0.0
        self.hit_sum = 0.0
        self.top10_feature_sum = 0.0
        self.top10_count = 0
        self.bottom10_feature_sum = 0.0
        self.bottom10_count = 0
        self.positive_feature_sum = 0.0
        self.positive_count = 0
        self.negative_feature_sum = 0.0
        self.negative_count = 0
        self.sample_values: list[np.ndarray] = []
        self.sample_targets: list[np.ndarray] = []
        self.sample_hits: list[np.ndarray] = []
        self.sample_count = 0

    def update(self, feature_values: np.ndarray, target: np.ndarray) -> None:
        x = np.asarray(feature_values, dtype=np.float64).reshape(-1)
        y = np.asarray(target, dtype=np.float64).reshape(-1)
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) <= 0:
            return
        x = x[valid]
        y = y[valid]
        count = int(len(x))
        self.total_count += int(len(valid))
        self.finite_count += count
        self.feature_sum += float(x.sum())
        self.feature_sq_sum += float(np.square(x).sum())
        self.target_sum += float(y.sum())
        self.target_sq_sum += float(np.square(y).sum())
        self.feature_target_sum += float(np.sum(x * y))
        hit = (y > 0.0).astype(np.float64)
        self.hit_sum += float(hit.sum())
        if count >= 10:
            ranks = pd.Series(y).rank(method="average", pct=True).to_numpy(dtype=np.float64)
            top = ranks >= 0.90
            bottom = ranks <= 0.10
            if int(top.sum()) > 0:
                self.top10_feature_sum += float(x[top].sum())
                self.top10_count += int(top.sum())
            if int(bottom.sum()) > 0:
                self.bottom10_feature_sum += float(x[bottom].sum())
                self.bottom10_count += int(bottom.sum())
        pos = y > 0.0
        neg = ~pos
        if int(pos.sum()) > 0:
            self.positive_feature_sum += float(x[pos].sum())
            self.positive_count += int(pos.sum())
        if int(neg.sum()) > 0:
            self.negative_feature_sum += float(x[neg].sum())
            self.negative_count += int(neg.sum())
        self._append_sample(x, y, hit)

    def _append_sample(self, x: np.ndarray, y: np.ndarray, hit: np.ndarray) -> None:
        remaining = int(self.max_samples) - int(self.sample_count)
        if remaining <= 0:
            return
        if len(x) > remaining:
            idx = np.linspace(0, len(x) - 1, num=remaining, dtype=np.int64)
            x = x[idx]
            y = y[idx]
            hit = hit[idx]
        self.sample_values.append(np.asarray(x, dtype=np.float32))
        self.sample_targets.append(np.asarray(y, dtype=np.float32))
        self.sample_hits.append(np.asarray(hit, dtype=np.float32))
        self.sample_count += int(len(x))

    def metric_row(self, *, role: str, target_kind: str, label_horizon: int) -> dict[str, Any]:
        count = float(self.finite_count)
        mean_x = _safe_div(self.feature_sum, count)
        mean_y = _safe_div(self.target_sum, count)
        var_x = _safe_div(self.feature_sq_sum, count) - mean_x * mean_x if count else float("nan")
        var_y = _safe_div(self.target_sq_sum, count) - mean_y * mean_y if count else float("nan")
        cov = _safe_div(self.feature_target_sum, count) - mean_x * mean_y if count else float("nan")
        denom = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0)) if math.isfinite(var_x) and math.isfinite(var_y) else 0.0
        corr = cov / denom if denom > 0.0 else float("nan")
        top_mean = _safe_div(self.top10_feature_sum, float(self.top10_count))
        bottom_mean = _safe_div(self.bottom10_feature_sum, float(self.bottom10_count))
        pos_mean = _safe_div(self.positive_feature_sum, float(self.positive_count))
        neg_mean = _safe_div(self.negative_feature_sum, float(self.negative_count))
        sample_x = np.concatenate(self.sample_values) if self.sample_values else np.asarray([], dtype=np.float32)
        quantiles = {}
        if len(sample_x):
            for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
                quantiles[f"feature_q{int(q * 100):02d}"] = float(np.nanquantile(sample_x, q))
        return {
            "role": role,
            "target_kind": target_kind,
            "label_horizon": int(label_horizon),
            "exit_label": _exit_label(int(label_horizon)),
            "feature": self.feature,
            "row_count": int(self.total_count),
            "finite_count": int(self.finite_count),
            "finite_rate": _safe_div(float(self.finite_count), float(self.total_count)),
            "feature_mean": mean_x,
            "feature_std": math.sqrt(max(var_x, 0.0)) if math.isfinite(var_x) else float("nan"),
            "target_mean": mean_y,
            "target_std": math.sqrt(max(var_y, 0.0)) if math.isfinite(var_y) else float("nan"),
            "hit_rate": _safe_div(self.hit_sum, count),
            "pearson_corr_raw": corr,
            "future_top10_feature_mean": top_mean,
            "future_bottom10_feature_mean": bottom_mean,
            "future_top10_minus_bottom10_feature_mean": top_mean - bottom_mean if math.isfinite(top_mean) and math.isfinite(bottom_mean) else float("nan"),
            "positive_return_feature_mean": pos_mean,
            "negative_return_feature_mean": neg_mean,
            "positive_minus_negative_feature_mean": pos_mean - neg_mean if math.isfinite(pos_mean) and math.isfinite(neg_mean) else float("nan"),
            **quantiles,
        }

    def bin_rows(self, *, role: str, target_kind: str, label_horizon: int) -> list[dict[str, Any]]:
        if not self.sample_values:
            return []
        x = np.concatenate(self.sample_values).astype(np.float64)
        y = np.concatenate(self.sample_targets).astype(np.float64)
        hit = np.concatenate(self.sample_hits).astype(np.float64)
        if len(x) < max(20, self.n_bins):
            return []
        quantile_grid = np.linspace(0.0, 1.0, int(self.n_bins) + 1)
        edges = np.nanquantile(x, quantile_grid)
        edges = np.unique(edges[np.isfinite(edges)])
        if len(edges) < 2:
            edges = np.asarray([float(np.nanmin(x)), float(np.nanmax(x))], dtype=np.float64)
            if not np.isfinite(edges).all() or edges[0] == edges[1]:
                return []
        bins = np.searchsorted(edges[1:-1], x, side="right")
        rows: list[dict[str, Any]] = []
        all_mean = float(np.nanmean(y))
        for bin_id in range(len(edges) - 1):
            mask = bins == bin_id
            count = int(mask.sum())
            if count <= 0:
                continue
            target_mean = float(np.nanmean(y[mask]))
            rows.append(
                {
                    "role": role,
                    "target_kind": target_kind,
                    "label_horizon": int(label_horizon),
                    "exit_label": _exit_label(int(label_horizon)),
                    "feature": self.feature,
                    "bin": int(bin_id),
                    "bin_left": float(edges[bin_id]),
                    "bin_right": float(edges[bin_id + 1]),
                    "sample_count": count,
                    "feature_mean": float(np.nanmean(x[mask])),
                    "target_mean": target_mean,
                    "target_lift_vs_sample_all": target_mean - all_mean,
                    "hit_rate": float(np.nanmean(hit[mask])),
                }
            )
        return rows


def _build_outputs(
    *,
    accumulators: Mapping[tuple[str, str, int, str], RawFeatureAccumulator],
    output_root: Path,
    report_base: Mapping[str, Any],
) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for (role, target_kind, horizon, feature), acc in sorted(accumulators.items()):
        metric_rows.append(acc.metric_row(role=role, target_kind=target_kind, label_horizon=horizon))
        bin_rows.extend(acc.bin_rows(role=role, target_kind=target_kind, label_horizon=horizon))
    metrics = pd.DataFrame(metric_rows)
    bins = pd.DataFrame(bin_rows)
    if not metrics.empty:
        metrics["abs_corr"] = metrics["pearson_corr_raw"].abs()
        metrics["abs_top_bottom_feature_gap"] = metrics["future_top10_minus_bottom10_feature_mean"].abs()
        leaderboard = metrics.sort_values(
            ["role", "target_kind", "label_horizon", "abs_corr", "abs_top_bottom_feature_gap"],
            ascending=[True, True, True, False, False],
            kind="mergesort",
        )
    else:
        leaderboard = pd.DataFrame()
    stable_corr, common_stable_corr = _stable_corr_frames(metrics)
    metrics_csv = _write_frame(output_root / "shortline_raw_upside_feature_metrics.csv", metrics)
    bins_csv = _write_frame(output_root / "shortline_raw_upside_feature_bins.csv", bins)
    leaderboard_csv = _write_frame(output_root / "shortline_raw_upside_feature_leaderboard.csv", leaderboard)
    stable_corr_csv = _write_frame(output_root / "shortline_raw_upside_stable_corr_features.csv", stable_corr)
    common_stable_corr_csv = _write_frame(
        output_root / "shortline_raw_upside_common_stable_corr_features.csv",
        common_stable_corr,
    )
    report = {
        **dict(report_base),
        "status": "completed" if not metrics.empty else "empty",
        "created_at": _now(),
        "outputs": {
            "feature_metrics_csv": metrics_csv,
            "feature_bins_csv": bins_csv,
            "feature_leaderboard_csv": leaderboard_csv,
            "stable_corr_features_csv": stable_corr_csv,
            "common_stable_corr_features_csv": common_stable_corr_csv,
        },
        "stable_corr_summary": {
            "stable_same_sign_rows": int(len(stable_corr)),
            "common_stable_feature_rows": int(len(common_stable_corr)),
            "target_horizon_combo_count": int(
                metrics[["target_kind", "label_horizon"]].drop_duplicates().shape[0]
            )
            if not metrics.empty
            else 0,
        },
        "leaderboard": _leaderboard_rows(metrics),
    }
    report_json = _write_json(output_root / "shortline_raw_upside_diagnostic_report.json", report)
    report["outputs"]["report_json"] = report_json
    _write_markdown(output_root / "shortline_raw_upside_diagnostic_report.md", report)
    report["outputs"]["report_md"] = str((output_root / "shortline_raw_upside_diagnostic_report.md").resolve())
    _write_json(output_root / "shortline_raw_upside_diagnostic_report.json", report)
    return report


def _stable_corr_frames(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stable_columns = [
        "target_kind",
        "label_horizon",
        "feature",
        "sign",
        "train_corr",
        "validation_corr",
        "test_corr",
        "mean_abs_corr",
    ]
    common_columns = ["feature", "sign", "stable_combo_count", "mean_abs_corr"]
    if metrics.empty:
        return pd.DataFrame(columns=stable_columns), pd.DataFrame(columns=common_columns)
    required_roles = ("train", "validation", "test")
    combo_count = int(metrics[["target_kind", "label_horizon"]].drop_duplicates().shape[0])
    stable_rows: list[dict[str, Any]] = []
    for (target_kind, horizon, feature), group in metrics.groupby(["target_kind", "label_horizon", "feature"], sort=True):
        by_role = {
            str(row["role"]): _safe_float(row["pearson_corr_raw"])
            for row in group[["role", "pearson_corr_raw"]].to_dict("records")
        }
        if not all(role in by_role and math.isfinite(by_role[role]) for role in required_roles):
            continue
        corrs = [float(by_role[role]) for role in required_roles]
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in corrs]
        if signs[0] == 0 or any(sign != signs[0] for sign in signs):
            continue
        stable_rows.append(
            {
                "target_kind": str(target_kind),
                "label_horizon": int(horizon),
                "feature": str(feature),
                "sign": "pos" if signs[0] > 0 else "neg",
                "train_corr": corrs[0],
                "validation_corr": corrs[1],
                "test_corr": corrs[2],
                "mean_abs_corr": float(np.mean(np.abs(corrs))),
            }
        )
    stable = pd.DataFrame(stable_rows, columns=stable_columns)
    if stable.empty:
        return stable, pd.DataFrame(columns=common_columns)
    stable = stable.sort_values(
        ["target_kind", "label_horizon", "mean_abs_corr", "feature"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    common_rows: list[dict[str, Any]] = []
    for (feature, sign), group in stable.groupby(["feature", "sign"], sort=True):
        stable_combo_count = int(group[["target_kind", "label_horizon"]].drop_duplicates().shape[0])
        if combo_count > 0 and stable_combo_count == combo_count:
            common_rows.append(
                {
                    "feature": str(feature),
                    "sign": str(sign),
                    "stable_combo_count": stable_combo_count,
                    "mean_abs_corr": float(group["mean_abs_corr"].mean()),
                }
            )
    common = pd.DataFrame(common_rows, columns=common_columns)
    if not common.empty:
        common = common.sort_values(["mean_abs_corr", "feature"], ascending=[False, True], kind="mergesort")
    return stable, common


def _leaderboard_rows(metrics: pd.DataFrame, *, limit: int = 40) -> list[dict[str, Any]]:
    if metrics.empty:
        return []
    preferred = metrics.loc[metrics["role"].astype(str).isin(["train", "validation", "test"])].copy()
    if preferred.empty:
        preferred = metrics.copy()
    preferred["abs_corr"] = preferred["pearson_corr_raw"].abs()
    rows = preferred.sort_values(["role", "target_kind", "label_horizon", "abs_corr"], ascending=[True, True, True, False], kind="mergesort").head(limit)
    cols = [
        "role",
        "target_kind",
        "exit_label",
        "feature",
        "pearson_corr_raw",
        "feature_mean",
        "feature_std",
        "target_mean",
        "hit_rate",
        "future_top10_minus_bottom10_feature_mean",
        "positive_minus_negative_feature_mean",
    ]
    return [{key: _json_default(value) for key, value in row.items()} for row in rows[cols].to_dict("records")]


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Shortline Raw Upside Diagnostic",
        "",
        f"- Status: `{report.get('status', '')}`",
        f"- Run tag: `{report.get('run_tag', '')}`",
        f"- Research-only: `{report.get('contract', {}).get('research_only', True)}`",
        f"- Feature source: `{report.get('contract', {}).get('feature_source', '')}`",
        f"- Rows scanned: `{report.get('input', {}).get('scanned_rows', 0)}`",
        f"- Shards scanned: `{report.get('input', {}).get('scanned_shards', 0)}`",
        f"- Feature count: `{report.get('input', {}).get('feature_count', 0)}`",
        "",
        "## Semantics",
        "",
        "- This diagnostic reads unnormalized QDP sharded engineering features, not the normalized training pack.",
        "- Labels use next-open entry to future-open exits; horizon 1 means D+1 open entry to D+2 open exit.",
        "- It is a mechanism/profile diagnostic, not topK strategy selection and not model training.",
        "",
        "## Leaderboard",
        "",
    ]
    leaderboard = list(report.get("leaderboard", []) or [])
    if leaderboard:
        lines.append("| role | target | exit | feature | raw_corr | target_mean | hit | top10-bottom10 feature |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for row in leaderboard[:20]:
            lines.append(
                "| {role} | {target_kind} | {exit_label} | {feature} | {corr:.5f} | {target:.5f} | {hit:.4f} | {gap:.5f} |".format(
                    role=row.get("role", ""),
                    target_kind=row.get("target_kind", ""),
                    exit_label=row.get("exit_label", ""),
                    feature=row.get("feature", ""),
                    corr=_safe_float(row.get("pearson_corr_raw")),
                    target=_safe_float(row.get("target_mean")),
                    hit=_safe_float(row.get("hit_rate")),
                    gap=_safe_float(row.get("future_top10_minus_bottom10_feature_mean")),
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
            f"- Feature bins: `{report.get('outputs', {}).get('feature_bins_csv', '')}`",
            f"- Leaderboard: `{report.get('outputs', {}).get('feature_leaderboard_csv', '')}`",
            f"- Stable corr features: `{report.get('outputs', {}).get('stable_corr_features_csv', '')}`",
            f"- Common stable corr features: `{report.get('outputs', {}).get('common_stable_corr_features_csv', '')}`",
            f"- Report JSON: `{report.get('outputs', {}).get('report_json', '')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_shortline_raw_upside_diagnostic(
    *,
    sharded_manifest_json: str | Path = DEFAULT_SHARDED_MANIFEST,
    output_root: str | Path | None = None,
    run_tag: str = "shortline_raw_upside_diagnostic_v1",
    roles: str | Iterable[str] | None = None,
    role_years: str | None = None,
    features: str | Iterable[str] | None = None,
    label_horizons: str | Iterable[int] | None = None,
    target_kinds: str | Iterable[str] | None = None,
    round_trip_cost_bps: float = 20.0,
    n_bins: int = 10,
    max_sample_per_feature: int = 250_000,
    max_shards_per_role: int = 0,
) -> dict[str, Any]:
    manifest_path = Path(sharded_manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("artifact_type", "")) != "qdp_sharded_memmap":
        raise ValueError(f"requires qdp_sharded_memmap manifest: {manifest_path}")
    resolved_roles = _parse_csv_strings(roles, default=DEFAULT_ROLES)
    resolved_role_years = _parse_role_years(role_years)
    resolved_horizons = _parse_csv_ints(label_horizons, default=DEFAULT_LABEL_HORIZONS)
    resolved_targets = _parse_csv_strings(target_kinds, default=DEFAULT_TARGET_KINDS)
    all_features = list(str(item) for item in list(manifest.get("feature_columns", []) or []))
    requested_features = _parse_csv_strings(features, default=DEFAULT_FEATURES)
    selected_features = tuple(feature for feature in requested_features if feature in set(all_features))
    missing = [feature for feature in requested_features if feature not in set(all_features)]
    if not selected_features:
        raise ValueError(f"none of requested features are in manifest. missing={missing[:10]}")
    feature_indices = [all_features.index(feature) for feature in selected_features]
    output_dir = Path(output_root) if output_root is not None else Path("daily_research/output/path_policy/shortline_raw_upside_diagnostic") / str(run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = _selected_shards(manifest, role_years=resolved_role_years, roles=resolved_roles)
    if int(max_shards_per_role) > 0:
        kept: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for shard in shards:
            role = str(shard["_role"])
            counts.setdefault(role, 0)
            if counts[role] >= int(max_shards_per_role):
                continue
            kept.append(shard)
            counts[role] += 1
        shards = kept
    if not shards:
        raise ValueError("no completed shards selected.")
    accumulators: dict[tuple[str, str, int, str], RawFeatureAccumulator] = {}
    scanned_rows = 0
    progress_path = output_dir / "shortline_raw_upside_diagnostic_progress.json"
    for shard_idx, shard in enumerate(shards, start=1):
        role = str(shard["_role"])
        feature_shape = tuple(int(item) for item in list(shard.get("feature_store_shape", []) or []))
        feature_store = np.memmap(str(shard["feature_store_path"]), dtype="float32", mode="r", shape=feature_shape)
        label_manifest = json.loads(Path(str(shard["label_manifest_json"])).read_text(encoding="utf-8"))
        arrays = _open_label_arrays(label_manifest, ("cumulative_return_1to20", "cumulative_excess_return_1to20"))
        selected_feature_values = np.asarray(feature_store[:, :, feature_indices], dtype=np.float32)
        scanned_rows += int(feature_shape[0] * feature_shape[1])
        for target_kind in resolved_targets:
            for horizon in resolved_horizons:
                target = _target_array(
                    arrays,
                    target_kind=str(target_kind),
                    label_horizon=int(horizon),
                    cost_bps=float(round_trip_cost_bps),
                )
                for local_idx, feature in enumerate(selected_features):
                    key = (role, str(target_kind), int(horizon), str(feature))
                    if key not in accumulators:
                        accumulators[key] = RawFeatureAccumulator(
                            feature=str(feature),
                            n_bins=int(n_bins),
                            max_samples=int(max_sample_per_feature),
                        )
                    accumulators[key].update(selected_feature_values[:, :, local_idx], target)
        if shard_idx % 20 == 0 or shard_idx == len(shards):
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "processed_shards": int(shard_idx),
                    "total_shards": int(len(shards)),
                    "scanned_rows": int(scanned_rows),
                },
            )
    report_base = {
        "run_tag": str(run_tag),
        "sharded_manifest_json": str(manifest_path.resolve()),
        "contract": {
            "diagnostic_kind": "shortline_raw_unnormalized_upside_profile",
            "research_only": True,
            "not_model_training": True,
            "active_artifact_impact": "unchanged",
            "feature_source": "qdp_sharded_memmap_forecast_feature_store_unnormalized_engineering_features",
            "label_semantics": "next_open_entry_to_future_open",
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "roles": list(resolved_roles),
            "role_years": {role: list(years) for role, years in resolved_role_years.items()},
            "label_horizons": [int(item) for item in resolved_horizons],
            "target_kinds": list(resolved_targets),
            "n_bins": int(n_bins),
        },
        "input": {
            "scanned_shards": int(len(shards)),
            "scanned_rows": int(scanned_rows),
            "feature_count": int(len(selected_features)),
            "features": list(selected_features),
            "missing_requested_features": missing,
            "source_feature_profile": str(manifest.get("feature_profile", "")),
            "source_label_schema": str(manifest.get("label_schema_name", "")),
            "source_label_schema_version": int(manifest.get("label_schema_version", 0) or 0),
        },
    }
    report = _build_outputs(accumulators=accumulators, output_root=output_dir, report_base=report_base)
    _write_json(progress_path, {"status": "completed", "report_json": report.get("outputs", {}).get("report_json", "")})
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile unnormalized QDP sharded features against shortline upside labels.")
    parser.add_argument("--sharded-manifest-json", default=str(DEFAULT_SHARDED_MANIFEST))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-tag", default="shortline_raw_upside_diagnostic_v1")
    parser.add_argument("--roles", default="train,validation,test")
    parser.add_argument("--role-years", default="")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--label-horizons", default="1,2,4")
    parser.add_argument("--target-kinds", default="net_abs,net_excess")
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--max-sample-per-feature", type=int, default=250_000)
    parser.add_argument("--max-shards-per-role", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_shortline_raw_upside_diagnostic(
        sharded_manifest_json=args.sharded_manifest_json,
        output_root=args.output_root or None,
        run_tag=args.run_tag,
        roles=args.roles,
        role_years=args.role_years or None,
        features=args.features,
        label_horizons=args.label_horizons,
        target_kinds=args.target_kinds,
        round_trip_cost_bps=float(args.round_trip_cost_bps),
        n_bins=int(args.n_bins),
        max_sample_per_feature=int(args.max_sample_per_feature),
        max_shards_per_role=int(args.max_shards_per_role),
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(report.get("outputs", {}).get("report_json", ""))
    return report


if __name__ == "__main__":
    main()
