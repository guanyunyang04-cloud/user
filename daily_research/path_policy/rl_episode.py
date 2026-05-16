from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import torch

from daily_research.continuous_policy.pipeline_utils import select_feature_columns, signal_dates_between
from daily_research.continuous_policy.portfolio_simulator import PortfolioState
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs, build_cross_section_state
from daily_research.path_policy import ALPHA_PATH20_SEQUENCE_POLICY_VERSION
from daily_research.path_policy.rl_dataset import (
    DEFAULT_RL_REWARD_PROFILE,
    Path20TrajectoryDataset,
    _next_return_panels,
    validate_no_oracle_or_future_inputs,
)


EPISODE_REWARD_COLUMNS = ["next_open_return", "benchmark_return", "next_open_excess_return"]
EPISODE_PORTFOLIO_FEATURE_COLUMNS = [
    "cash_weight",
    "holding_count",
    "gross_exposure",
    "turnover_5d_mean",
    "realized_return_5d",
    "realized_vol_20d",
    "cash_change_5d",
    "cash_deficit",
]
EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS = ["rollout_current_weight"]
EPISODE_NON_INPUT_COLUMNS = {
    "date",
    "stock",
    "in_universe",
    "current_weight",
    "tradable_mask",
    "next_open_return",
    "benchmark_return",
    "next_open_excess_return",
    "reward_profile",
}


@dataclass(frozen=True)
class Path20MarketEpisode:
    daily_frames: list[pd.DataFrame]
    feature_columns: list[str]
    dates: list[pd.Timestamp]
    manifest: dict[str, Any]

    @property
    def empty(self) -> bool:
        return not self.daily_frames

    def to_long_frame(self) -> pd.DataFrame:
        return pd.concat(self.daily_frames, ignore_index=True) if self.daily_frames else pd.DataFrame()

    def to_trajectory(self) -> Path20TrajectoryDataset:
        frames: list[pd.DataFrame] = []
        for daily in self.daily_frames:
            frame = daily.copy()
            if "current_weight" not in frame.columns:
                frame["current_weight"] = 0.0
            for column in EPISODE_PORTFOLIO_FEATURE_COLUMNS:
                frame[f"portfolio_{column}"] = 0.0
            frames.append(frame)
        return Path20TrajectoryDataset(
            daily_frames=frames,
            feature_columns=list(self.feature_columns),
            dates=list(self.dates),
            manifest=dict(self.manifest),
        )


def stable_market_episode_id(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:24]
    return f"alpha_path20_market_episode__{digest}"


def _resolve_episode_feature_columns(sample: pd.DataFrame, max_feature_columns: int) -> list[str]:
    candidates = []
    for column in select_feature_columns(sample):
        name = str(column)
        if name in EPISODE_NON_INPUT_COLUMNS:
            continue
        if name.startswith("portfolio_"):
            continue
        candidates.append(name)
    validate_no_oracle_or_future_inputs(candidates)
    return candidates[: int(max_feature_columns)]


def build_path20_market_episode(
    prepared: PreparedPolicyInputs,
    *,
    start_date: str,
    end_date: str,
    lake_dataset_id: str = "",
    year: int | str | None = None,
    sequence_length: int = 20,
    execution_mode: str = "next_open",
    reward_profile: str = DEFAULT_RL_REWARD_PROFILE,
    max_feature_columns: int = 96,
    min_trading_days: int = 180,
) -> Path20MarketEpisode:
    stock_next_return, benchmark_next_return = _next_return_panels(prepared, execution_mode=execution_mode)
    dates = signal_dates_between(prepared, start_date=start_date, end_date=end_date, max_forward_horizon=1)
    empty_portfolio = PortfolioState()
    daily_frames: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None
    for dt in dates:
        state_frame = build_cross_section_state(prepared, date=dt, portfolio_state=empty_portfolio).copy()
        if feature_columns is None:
            feature_columns = _resolve_episode_feature_columns(state_frame, max_feature_columns=max_feature_columns)
        membership = prepared.membership_frame.reindex(index=[dt], columns=prepared.close.columns).iloc[0].fillna(False).astype(bool)
        next_ret = stock_next_return.loc[dt].reindex(state_frame["stock"].astype(str)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        bench_ret = float(benchmark_next_return.loc[dt]) if dt in benchmark_next_return.index else 0.0
        frame = state_frame[["date", "stock", *feature_columns]].copy()
        frame["tradable_mask"] = frame["stock"].astype(str).map(lambda stock: bool(membership.get(stock, False))).astype(float)
        frame["next_open_return"] = frame["stock"].astype(str).map(lambda stock: float(next_ret.get(stock, 0.0)))
        frame["benchmark_return"] = float(bench_ret if np.isfinite(bench_ret) else 0.0)
        frame["next_open_excess_return"] = frame["next_open_return"].astype(float) - float(frame["benchmark_return"].iloc[0])
        frame["reward_profile"] = str(reward_profile)
        daily_frames.append(frame)
    resolved_features = feature_columns or []
    leakage_guard = validate_no_oracle_or_future_inputs(resolved_features)
    actual_start = dates[0].strftime("%Y-%m-%d") if dates else ""
    actual_end = dates[-1].strftime("%Y-%m-%d") if dates else ""
    dataset_id = stable_market_episode_id(
        {
            "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
            "lake_dataset_id": str(lake_dataset_id or ""),
            "year": int(year) if year is not None and str(year).strip() else "",
            "requested_start_date": str(start_date),
            "requested_end_date": str(end_date),
            "actual_signal_start_date": actual_start,
            "actual_signal_end_date": actual_end,
            "sequence_length": int(sequence_length),
            "feature_count": int(len(resolved_features)),
            "trading_day_count": int(len(dates)),
            "reward_profile": str(reward_profile),
        }
    )
    incomplete = int(len(dates)) < int(min_trading_days)
    manifest = {
        "status": "incomplete" if incomplete else "completed",
        "incomplete_reason": "trading_day_count_below_180" if incomplete else "",
        "artifact_type": "market_episode",
        "policy_version": ALPHA_PATH20_SEQUENCE_POLICY_VERSION,
        "dataset_id": dataset_id,
        "lake_dataset_id": str(lake_dataset_id or ""),
        "stage": "path20_market_episode_dataset",
        "year": int(year) if year is not None and str(year).strip() else "",
        "requested_start_date": str(start_date),
        "requested_end_date": str(end_date),
        "actual_signal_start_date": actual_start,
        "actual_signal_end_date": actual_end,
        "trading_day_count": int(len(dates)),
        "sequence_length": int(sequence_length),
        "feature_columns": resolved_features,
        "dynamic_stock_feature_columns": list(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS),
        "model_input_feature_columns": [*resolved_features, *EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS],
        "feature_count": int(len(resolved_features)),
        "portfolio_feature_columns": list(EPISODE_PORTFOLIO_FEATURE_COLUMNS),
        "reward_columns": list(EPISODE_REWARD_COLUMNS),
        "reward_profile": str(reward_profile),
        "execution_mode": str(execution_mode),
        "normalization_scope": "train_years_only_when_training",
        "leakage_guard": leakage_guard,
        "loose_latest_allowed": False,
        "shadow_only": True,
        "promotion_allowed": False,
    }
    return Path20MarketEpisode(
        daily_frames=daily_frames,
        feature_columns=resolved_features,
        dates=dates,
        manifest=manifest,
    )


def episode_to_arrays(episode: Path20MarketEpisode) -> dict[str, np.ndarray]:
    if episode.empty:
        return {
            "state": np.zeros((0, 0, 0), dtype=np.float32),
            "mask": np.zeros((0, 0), dtype=bool),
            "future_return": np.zeros((0, 0), dtype=np.float32),
            "benchmark_return": np.zeros((0,), dtype=np.float32),
            "dates": np.asarray([], dtype=object),
            "stocks": np.asarray([], dtype=object),
        }
    stocks = [str(item) for item in episode.daily_frames[0]["stock"].astype(str).tolist()]
    states = []
    masks = []
    future_returns = []
    benchmark_returns = []
    dates = []
    for daily in episode.daily_frames:
        indexed = daily.set_index("stock").reindex(stocks)
        features = indexed[episode.feature_columns].apply(pd.to_numeric, errors="coerce")
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        states.append(features.to_numpy(dtype=np.float32))
        masks.append(indexed["tradable_mask"].astype(float).fillna(0.0).to_numpy(dtype=float) > 0.5)
        future_returns.append(indexed["next_open_excess_return"].astype(float).fillna(0.0).to_numpy(dtype=np.float32))
        benchmark_returns.append(float(indexed["benchmark_return"].iloc[0]))
        dates.append(str(indexed["date"].iloc[0]))
    return {
        "state": np.stack(states, axis=0).astype(np.float32),
        "mask": np.stack(masks, axis=0).astype(bool),
        "future_return": np.stack(future_returns, axis=0).astype(np.float32),
        "benchmark_return": np.asarray(benchmark_returns, dtype=np.float32),
        "dates": np.asarray(dates, dtype=object),
        "stocks": np.asarray(stocks, dtype=object),
    }


def fit_episode_normalization(episodes: list[Path20MarketEpisode]) -> dict[str, Any]:
    completed = [episode for episode in episodes if not episode.empty]
    if not completed:
        return {"scope": "train_years", "feature_columns": [], "mean": [], "std": []}
    feature_columns = list(completed[0].feature_columns)
    blocks = []
    for episode in completed:
        if list(episode.feature_columns) != feature_columns:
            raise ValueError("All market episodes must share feature columns for normalization.")
        blocks.append(episode_to_arrays(episode)["state"].reshape(-1, len(feature_columns)))
    matrix = np.concatenate(blocks, axis=0) if blocks else np.zeros((0, len(feature_columns)), dtype=np.float32)
    mean = np.nanmean(matrix, axis=0) if len(matrix) else np.zeros(len(feature_columns), dtype=np.float32)
    std = np.nanstd(matrix, axis=0) if len(matrix) else np.ones(len(feature_columns), dtype=np.float32)
    std = np.where(np.isfinite(std) & (std > 1.0e-8), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return {
        "scope": "train_years",
        "feature_columns": feature_columns,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def normalize_episode_arrays(arrays: dict[str, np.ndarray], normalization: dict[str, Any]) -> dict[str, np.ndarray]:
    state = np.asarray(arrays["state"], dtype=np.float32)
    mean = np.asarray(normalization.get("mean", []), dtype=np.float32)
    std = np.asarray(normalization.get("std", []), dtype=np.float32)
    if state.size and len(mean) == state.shape[-1] and len(std) == state.shape[-1]:
        state = (state - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    result = dict(arrays)
    result["state"] = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return result


def project_target_weights_torch(
    raw_target_weight: torch.Tensor,
    current_weight: torch.Tensor,
    tradable_mask: torch.Tensor,
    *,
    max_position_weight: float,
    max_gross_exposure: float,
    max_positions: int,
    turnover_budget: float,
) -> dict[str, torch.Tensor]:
    raw = torch.nan_to_num(raw_target_weight.float(), nan=0.0, posinf=0.0, neginf=0.0)
    current = torch.nan_to_num(current_weight.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    tradable = tradable_mask.bool()
    raw_long = raw.clamp_min(0.0)
    clipped = raw_long.clamp(max=float(max_position_weight))
    locked = torch.where((~tradable) & (current > 1.0e-12), current, torch.zeros_like(current))
    tradable_target = torch.where(tradable, clipped, torch.zeros_like(clipped))
    if int(max_positions) > 0:
        locked_count = int((locked > 1.0e-12).detach().sum().item())
        remaining_slots = max(int(max_positions) - locked_count, 0)
        positive = tradable_target > 1.0e-12
        if remaining_slots <= 0:
            tradable_target = torch.zeros_like(tradable_target)
        elif int(positive.detach().sum().item()) > remaining_slots:
            _, indices = torch.topk(tradable_target, k=remaining_slots)
            keep = torch.zeros_like(tradable_target, dtype=torch.bool)
            keep[indices] = True
            tradable_target = torch.where(keep, tradable_target, torch.zeros_like(tradable_target))
    projected = (locked + tradable_target).clamp_min(0.0)
    raw_gross = raw_long.sum()
    projected_gross = projected.sum()
    max_gross = float(np.clip(float(max_gross_exposure), 0.0, 1.0))
    locked_sum = locked.sum()
    tradable_sum = tradable_target.sum()
    available = torch.clamp(torch.as_tensor(max_gross, device=raw.device, dtype=raw.dtype) - locked_sum, min=0.0)
    gross_scale = torch.where(
        (projected_gross > max_gross) & (tradable_sum > 1.0e-12),
        available / tradable_sum.clamp_min(1.0e-12),
        torch.ones_like(tradable_sum),
    )
    tradable_target = tradable_target * gross_scale.clamp(max=1.0)
    projected = (locked + tradable_target).clamp_min(0.0)
    current_delta = projected - current
    raw_turnover = (raw_long - current).abs().sum()
    projected_turnover = current_delta.abs().sum()
    budget = max(float(turnover_budget), 0.0)
    turnover_scale = torch.where(
        (projected_turnover > budget) & (projected_turnover > 1.0e-12) & (torch.as_tensor(budget, device=raw.device, dtype=raw.dtype) > 0.0),
        torch.as_tensor(budget, device=raw.device, dtype=raw.dtype) / projected_turnover.clamp_min(1.0e-12),
        torch.ones_like(projected_turnover),
    )
    projected = (current + current_delta * turnover_scale.clamp(max=1.0)).clamp(min=0.0, max=float(max_position_weight))
    projected_gross = projected.sum()
    projected_turnover = (projected - current).abs().sum()
    diagnostics = {
        "raw_gross_exposure": raw_gross,
        "projected_gross_exposure": projected_gross,
        "raw_turnover": raw_turnover,
        "projected_turnover": projected_turnover,
        "projection_l1_distance": (projected - raw_long).abs().sum(),
        "cash_weight": (1.0 - projected_gross).clamp_min(0.0),
        "clipped_count": ((raw < 0.0) | (raw > float(max_position_weight))).float().sum(),
        "masked_count": ((~tradable) & ((clipped - current).abs() > 1.0e-12)).float().sum(),
        "gross_scaled_count": (gross_scale < 1.0 - 1.0e-8).float(),
        "turnover_scaled_count": (turnover_scale < 1.0 - 1.0e-8).float(),
    }
    return {"projected_target_weight": projected, "diagnostics": diagnostics}


def _portfolio_feature_tensor(
    current_weight: torch.Tensor,
    recent_turnovers: list[torch.Tensor],
    recent_rewards: list[torch.Tensor],
) -> torch.Tensor:
    gross = current_weight.clamp_min(0.0).sum()
    cash = (1.0 - gross).clamp_min(0.0)
    holding_count = (current_weight > 1.0e-8).float().sum()
    turnover_5d = torch.stack(recent_turnovers[-5:]).mean() if recent_turnovers else torch.zeros((), device=current_weight.device)
    if recent_rewards:
        rewards_5 = torch.stack(recent_rewards[-5:])
        rewards_20 = torch.stack(recent_rewards[-20:])
        realized_5 = rewards_5.mean()
        realized_vol_20 = rewards_20.std(unbiased=False) if rewards_20.numel() > 1 else torch.zeros((), device=current_weight.device)
    else:
        realized_5 = torch.zeros((), device=current_weight.device)
        realized_vol_20 = torch.zeros((), device=current_weight.device)
    cash_change_5d = torch.zeros((), device=current_weight.device)
    cash_deficit = torch.relu(gross - 1.0)
    return torch.stack(
        [
            cash,
            holding_count,
            gross,
            turnover_5d,
            realized_5,
            realized_vol_20,
            cash_change_5d,
            cash_deficit,
        ]
    ).float()


def episode_policy_rollout_loss(
    model: torch.nn.Module,
    episode: Path20MarketEpisode,
    *,
    sequence_length: int,
    normalization: dict[str, Any],
    model_family: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    sell_tax_bps: float,
    max_position_weight: float,
    max_gross_exposure: float,
    max_positions: int,
    turnover_budget: float,
    turnover_penalty: float = 0.20,
    concentration_penalty: float = 0.02,
    projection_penalty: float = 0.05,
    drawdown_penalty: float = 0.10,
    entropy_bonus: float = 0.001,
    detach_rollout_state: bool = True,
) -> dict[str, Any]:
    arrays = normalize_episode_arrays(episode_to_arrays(episode), normalization)
    state_np = arrays["state"]
    if state_np.shape[0] < int(sequence_length):
        zero = torch.zeros((), dtype=torch.float32)
        return {"status": "skipped", "reason": "insufficient_sequence_length", "loss": zero}
    device = next(model.parameters()).device
    state = torch.tensor(state_np, dtype=torch.float32, device=device)
    mask = torch.tensor(arrays["mask"], dtype=torch.bool, device=device)
    future_return = torch.tensor(arrays["future_return"], dtype=torch.float32, device=device)
    seq_len = int(sequence_length)
    stock_count = int(state.shape[1])
    current_weight = torch.zeros(stock_count, dtype=torch.float32, device=device)
    weight_context: list[torch.Tensor] = []
    reward_context: list[torch.Tensor] = []
    recent_turnovers: list[torch.Tensor] = []
    recent_rewards: list[torch.Tensor] = []
    portfolio_context: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    diagnostics_rows: list[dict[str, float]] = []
    equity = torch.ones((), dtype=torch.float32, device=device)
    peak_equity = torch.ones((), dtype=torch.float32, device=device)
    previous_reward = torch.zeros((), dtype=torch.float32, device=device)
    for idx in range(int(state.shape[0])):
        current_for_context = current_weight.detach() if detach_rollout_state else current_weight
        weight_context.append(current_for_context)
        reward_context.append(previous_reward.detach() if detach_rollout_state else previous_reward)
        portfolio_context.append(_portfolio_feature_tensor(current_for_context, recent_turnovers, recent_rewards))
        if idx < seq_len - 1:
            previous_reward = torch.zeros((), dtype=torch.float32, device=device)
            continue
        weight_window = torch.stack(weight_context[idx - seq_len + 1 : idx + 1], dim=0)
        state_window = state[idx - seq_len + 1 : idx + 1]
        dynamic_weight = weight_window.unsqueeze(-1)
        model_state = torch.cat([state_window, dynamic_weight], dim=-1).unsqueeze(0)
        portfolio_window = torch.stack(portfolio_context[idx - seq_len + 1 : idx + 1], dim=0).unsqueeze(0)
        reward_window = torch.stack(reward_context[idx - seq_len + 1 : idx + 1], dim=0).unsqueeze(0)
        mask_now = mask[idx].unsqueeze(0)
        if str(model_family).strip().lower() == "decision_transformer":
            pred = model(
                model_state,
                portfolio_window,
                previous_weight_sequence=weight_window.unsqueeze(0),
                previous_reward_sequence=reward_window,
                tradable_mask=mask_now,
            )
        else:
            pred = model(model_state, portfolio_window, tradable_mask=mask_now)
        raw = pred["raw_target_weight"].squeeze(0)
        projection = project_target_weights_torch(
            raw,
            current_weight,
            mask[idx],
            max_position_weight=max_position_weight,
            max_gross_exposure=max_gross_exposure,
            max_positions=max_positions,
            turnover_budget=turnover_budget,
        )
        projected = projection["projected_target_weight"]
        diag = projection["diagnostics"]
        buy_turnover = (projected - current_weight).clamp_min(0.0).sum()
        sell_turnover = (current_weight - projected).clamp_min(0.0).sum()
        cost = (
            buy_turnover * (float(transaction_cost_bps) + float(slippage_bps))
            + sell_turnover * (float(transaction_cost_bps) + float(slippage_bps) + float(sell_tax_bps))
        ) / 10_000.0
        gross_return = (projected * future_return[idx].nan_to_num(0.0)).sum()
        net_reward = gross_return - cost
        equity = equity * (1.0 + net_reward).clamp_min(0.05)
        peak_equity = torch.maximum(peak_equity, equity.detach() if detach_rollout_state else equity)
        drawdown = (equity / peak_equity.clamp_min(1.0e-8) - 1.0).clamp(max=0.0)
        concentration = (projected**2).sum()
        entropy = -(projected.clamp_min(1.0e-12) * projected.clamp_min(1.0e-12).log()).sum()
        loss = -torch.log1p(net_reward.clamp(min=-0.95))
        loss = loss + float(turnover_penalty) * diag["projected_turnover"]
        loss = loss + float(concentration_penalty) * concentration
        loss = loss + float(projection_penalty) * diag["projection_l1_distance"]
        loss = loss + float(drawdown_penalty) * (-drawdown)
        loss = loss - float(entropy_bonus) * entropy
        losses.append(loss)
        diagnostics_rows.append({key: float(value.detach().cpu()) for key, value in diag.items()})
        recent_turnovers.append(diag["projected_turnover"].detach())
        recent_rewards.append(net_reward.detach())
        previous_reward = net_reward
        current_weight = projected.detach() if detach_rollout_state else projected
    if not losses:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return {"status": "skipped", "reason": "empty_rollout_loss", "loss": zero}
    loss_tensor = torch.stack(losses).mean()
    return {
        "status": "completed",
        "loss": loss_tensor,
        "step_count": int(len(losses)),
        "diagnostics": _summarize_diagnostics(diagnostics_rows),
        "used_projected_weights_for_loss": True,
        "rolled_current_weight": True,
        "decision_transformer_previous_context_nonzero": bool(
            str(model_family).strip().lower() == "decision_transformer"
            and any(float(item.abs().sum().detach().cpu()) > 0.0 for item in reward_context)
        ),
    }


def _summarize_diagnostics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    return {
        f"avg_{column}": float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean())
        for column in frame.columns
    }


def predict_episode_targets(
    model: torch.nn.Module,
    episode: Path20MarketEpisode,
    *,
    sequence_length: int,
    normalization: dict[str, Any],
    model_family: str,
    max_position_weight: float,
    max_gross_exposure: float,
    max_positions: int,
    turnover_budget: float,
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    arrays = normalize_episode_arrays(episode_to_arrays(episode), normalization)
    if arrays["state"].shape[0] < int(sequence_length):
        return {}, {}
    device = next(model.parameters()).device
    state = torch.tensor(arrays["state"], dtype=torch.float32, device=device)
    mask = torch.tensor(arrays["mask"], dtype=torch.bool, device=device)
    dates = [str(item) for item in arrays["dates"].tolist()]
    stocks = [str(item) for item in arrays["stocks"].tolist()]
    seq_len = int(sequence_length)
    current_weight = torch.zeros(int(state.shape[1]), dtype=torch.float32, device=device)
    weight_context: list[torch.Tensor] = []
    reward_context: list[torch.Tensor] = []
    recent_turnovers: list[torch.Tensor] = []
    recent_rewards: list[torch.Tensor] = []
    portfolio_context: list[torch.Tensor] = []
    target_by_date: dict[str, pd.Series] = {}
    diagnostics_rows: list[dict[str, float]] = []
    previous_reward = torch.zeros((), dtype=torch.float32, device=device)
    future_return = torch.tensor(arrays["future_return"], dtype=torch.float32, device=device)
    with torch.no_grad():
        for idx in range(int(state.shape[0])):
            weight_context.append(current_weight)
            reward_context.append(previous_reward)
            portfolio_context.append(_portfolio_feature_tensor(current_weight, recent_turnovers, recent_rewards))
            if idx < seq_len - 1:
                previous_reward = torch.zeros((), dtype=torch.float32, device=device)
                continue
            weight_window = torch.stack(weight_context[idx - seq_len + 1 : idx + 1], dim=0)
            state_window = torch.cat([state[idx - seq_len + 1 : idx + 1], weight_window.unsqueeze(-1)], dim=-1).unsqueeze(0)
            portfolio_window = torch.stack(portfolio_context[idx - seq_len + 1 : idx + 1], dim=0).unsqueeze(0)
            reward_window = torch.stack(reward_context[idx - seq_len + 1 : idx + 1], dim=0).unsqueeze(0)
            if str(model_family).strip().lower() == "decision_transformer":
                pred = model(
                    state_window,
                    portfolio_window,
                    previous_weight_sequence=weight_window.unsqueeze(0),
                    previous_reward_sequence=reward_window,
                    tradable_mask=mask[idx].unsqueeze(0),
                )
            else:
                pred = model(state_window, portfolio_window, tradable_mask=mask[idx].unsqueeze(0))
            raw = pred["raw_target_weight"].squeeze(0)
            target_by_date[dates[idx]] = pd.Series(raw.detach().cpu().numpy(), index=stocks, dtype=float)
            projection = project_target_weights_torch(
                raw,
                current_weight,
                mask[idx],
                max_position_weight=max_position_weight,
                max_gross_exposure=max_gross_exposure,
                max_positions=max_positions,
                turnover_budget=turnover_budget,
            )
            projected = projection["projected_target_weight"]
            diag = projection["diagnostics"]
            diagnostics_rows.append({key: float(value.detach().cpu()) for key, value in diag.items()})
            net_reward = (projected * future_return[idx].nan_to_num(0.0)).sum()
            recent_turnovers.append(diag["projected_turnover"].detach())
            recent_rewards.append(net_reward.detach())
            previous_reward = net_reward.detach()
            current_weight = projected.detach()
    return target_by_date, _summarize_diagnostics(diagnostics_rows)

