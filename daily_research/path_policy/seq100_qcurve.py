from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


QCURVE_FORWARD_DAYS = 60
QCURVE_EXECUTION_TAIL_DAYS = 20
QCURVE_ENTRY_HORIZONS = tuple(range(2, QCURVE_FORWARD_DAYS + 1))
QCURVE_HOLD_HORIZONS = tuple(range(1, QCURVE_FORWARD_DAYS + 1))
QCURVE_RANK_ANCHORS = (2, 5, 10, 20, 40, 60)
QCURVE_REFERENCE_PORTFOLIO_CASH_CNY = 1_000_000.0
QCURVE_REFERENCE_POSITION_CASH_CNY = QCURVE_REFERENCE_PORTFOLIO_CASH_CNY / 3.0
QCURVE_MAX_POSITIONS = 3
QCURVE_MAX_POSITION_WEIGHT = 0.60
QCURVE_Q20_GATE = 0.0
QCURVE_SOFT_EXIT_TEMPERATURE = 0.03

MA_STATE_FEATURES = (
    "close_to_ema_5",
    "close_to_ema_10",
    "close_to_ema_20",
    "close_to_ema_60",
    "ema_5_to_20",
    "ema_20_to_60",
    "ema_5_slope_3d",
    "ema_20_slope_5d",
    "ema_60_slope_10d",
    "volume_ema_5_to_20",
    "volume_ema_20_to_60",
    "amount_ema_5_to_20",
    "amount_ema_20_to_60",
    "close_to_vwap_20",
)


@dataclass(frozen=True)
class QCurveLossWeights:
    mean: float = 0.25
    quantile: float = 0.20
    positive: float = 0.05
    soft_exit: float = 0.10
    rank: float = 0.25
    price_aux: float = 0.10
    va_aux: float = 0.05

    def validate(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values())
        if any((not math.isfinite(value)) or value < 0.0 for value in values):
            raise ValueError("Q-curve loss weights must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Q-curve loss weights must sum to 1")


@dataclass(frozen=True)
class QCurveCostContract:
    lot_size: int
    commission_bps: float
    minimum_commission_cny: float
    transfer_fee_bps: float
    slippage_bps: float
    stress_slippage_multiplier: float
    stamp_tax_schedule: tuple[tuple[str, float], ...]
    terminal_recovery_fraction: float = 0.0

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "QCurveCostContract":
        raw = dict(manifest.get("execution_cost_contract", {}) or {})
        if not raw:
            raise ValueError("manifest is missing execution_cost_contract")
        schedule_raw = raw.get("stamp_tax_schedule") or [
            {"effective_date": "1900-01-01", "stamp_tax_bps": raw.get("stamp_tax_bps", 0.0)}
        ]
        schedule: list[tuple[str, float]] = []
        for item in schedule_raw:
            if isinstance(item, Mapping):
                date = str(item["effective_date"])
                rate = float(item["stamp_tax_bps"])
            else:
                date, rate = str(item[0]), float(item[1])
            schedule.append((pd.Timestamp(date).strftime("%Y-%m-%d"), rate))
        terminal = dict(manifest.get("terminal_execution_contract", {}) or {})
        result = cls(
            lot_size=int(raw["lot_size"]),
            commission_bps=float(raw["commission_bps"]),
            minimum_commission_cny=float(raw["minimum_commission_cny"]),
            transfer_fee_bps=float(raw["transfer_fee_bps"]),
            slippage_bps=float(raw["slippage_bps"]),
            stress_slippage_multiplier=float(raw["stress_slippage_multiplier"]),
            stamp_tax_schedule=tuple(schedule),
            terminal_recovery_fraction=float(
                terminal.get("recovery_fraction_of_entry_notional", 0.0) or 0.0
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        numeric = (
            self.commission_bps,
            self.minimum_commission_cny,
            self.transfer_fee_bps,
            self.slippage_bps,
            self.stress_slippage_multiplier,
            self.terminal_recovery_fraction,
        )
        if self.lot_size <= 0 or any((not math.isfinite(float(v))) or float(v) < 0.0 for v in numeric):
            raise ValueError("invalid Q-curve execution cost contract")
        if not 0.0 <= self.terminal_recovery_fraction <= 1.0:
            raise ValueError("terminal recovery fraction must be within [0, 1]")
        if self.stress_slippage_multiplier < 1.0:
            raise ValueError("stress slippage multiplier must be at least 1")


def price_to_tick_units(values: np.ndarray, *, tick_size: float = 0.01) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=np.float64) / float(tick_size) + 0.5)


def build_open_execution_masks(
    open_raw: np.ndarray,
    up_limit_raw: np.ndarray,
    down_limit_raw: np.ndarray,
    *,
    observed: np.ndarray,
    status_valid: np.ndarray,
    suspended: np.ndarray,
    delisted: np.ndarray,
    tick_size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    open_values = np.asarray(open_raw, dtype=np.float64)
    up_values = np.asarray(up_limit_raw, dtype=np.float64)
    down_values = np.asarray(down_limit_raw, dtype=np.float64)
    shapes = {
        open_values.shape,
        up_values.shape,
        down_values.shape,
        np.asarray(observed).shape,
        np.asarray(status_valid).shape,
        np.asarray(suspended).shape,
        np.asarray(delisted).shape,
    }
    if len(shapes) != 1:
        raise ValueError("open execution arrays and masks must share shape")
    base = (
        np.isfinite(open_values)
        & np.asarray(observed, dtype=bool)
        & np.asarray(status_valid, dtype=bool)
        & (~np.asarray(suspended, dtype=bool))
        & (~np.asarray(delisted, dtype=bool))
    )
    buyable = (
        base
        & np.isfinite(up_values)
        & (price_to_tick_units(open_values, tick_size=tick_size) < price_to_tick_units(up_values, tick_size=tick_size))
    )
    sellable = (
        base
        & np.isfinite(down_values)
        & (price_to_tick_units(open_values, tick_size=tick_size) > price_to_tick_units(down_values, tick_size=tick_size))
    )
    return buyable, sellable


def _stamp_tax_rates(
    exit_dates: np.ndarray,
    schedule: Sequence[tuple[str, float]],
) -> np.ndarray:
    dates = np.asarray(exit_dates, dtype="datetime64[D]")
    rates = np.full(dates.shape, np.nan, dtype=np.float64)
    for effective_date, rate in sorted(schedule):
        rates[dates >= np.datetime64(effective_date)] = float(rate)
    if not np.isfinite(rates).all():
        raise ValueError("Q-curve exit date predates the stamp-tax schedule")
    return rates


def _next_sellable_indices(sellable: np.ndarray) -> np.ndarray:
    mask = np.asarray(sellable, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("sellable must be [batch, future_day]")
    batch, days = mask.shape
    result = np.full((batch, days), -1, dtype=np.int16)
    nearest = np.full(batch, -1, dtype=np.int16)
    for day in range(days - 1, -1, -1):
        nearest = np.where(mask[:, day], np.int16(day), nearest)
        result[:, day] = nearest
    return result


def _shares_for_cash(
    cash: float,
    entry_price: np.ndarray,
    *,
    contract: QCurveCostContract,
    include_buy_cost: bool,
    slippage_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    entry = np.asarray(entry_price, dtype=np.float64)
    slippage = contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    buy_price = entry * (1.0 + slippage if include_buy_cost else 1.0)
    lot = int(contract.lot_size)
    valid_entry = np.isfinite(buy_price) & (buy_price > 0.0)
    denominator = np.where(valid_entry, np.maximum(buy_price * lot, 1.0e-12), np.inf)
    shares = np.floor(float(cash) / denominator).astype(np.int64) * lot
    commission_rate = contract.commission_bps / 10_000.0
    transfer_rate = contract.transfer_fee_bps / 10_000.0
    for _ in range(4):
        notional = shares * buy_price
        buy_cost = (
            np.maximum(contract.minimum_commission_cny, notional * commission_rate)
            + notional * transfer_rate
            if include_buy_cost
            else np.zeros_like(notional)
        )
        required = notional + buy_cost
        over = (shares > 0) & (required > float(cash) + 1.0e-9)
        if not bool(over.any()):
            break
        shares[over] -= lot
    notional = shares * buy_price
    buy_cost = (
        np.maximum(contract.minimum_commission_cny, notional * commission_rate)
        + notional * transfer_rate
        if include_buy_cost
        else np.zeros_like(notional)
    )
    cash_after_buy = float(cash) - notional - buy_cost
    invalid = (~np.isfinite(entry)) | (entry <= 0.0) | (shares <= 0)
    shares[invalid] = 0
    cash_after_buy[invalid] = float(cash)
    return shares, cash_after_buy


def _net_log_curve(
    *,
    reference_cash: float,
    basis_price: np.ndarray,
    exit_prices: np.ndarray,
    exit_dates: np.ndarray,
    entry_filled: np.ndarray,
    exit_resolved: np.ndarray,
    contract: QCurveCostContract,
    include_buy_cost: bool,
    slippage_multiplier: float,
) -> np.ndarray:
    basis = np.asarray(basis_price, dtype=np.float64).reshape(-1)
    exits = np.asarray(exit_prices, dtype=np.float64)
    if exits.ndim != 2 or exits.shape[0] != basis.size:
        raise ValueError("exit_prices must be [batch, horizon]")
    shares, cash_after_buy = _shares_for_cash(
        float(reference_cash),
        basis,
        contract=contract,
        include_buy_cost=include_buy_cost,
        slippage_multiplier=slippage_multiplier,
    )
    slippage = contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    sell_price = exits * np.maximum(0.0, 1.0 - slippage)
    sell_notional = shares[:, None] * sell_price
    commission_rate = contract.commission_bps / 10_000.0
    transfer_rate = contract.transfer_fee_bps / 10_000.0
    sell_commission = np.maximum(contract.minimum_commission_cny, sell_notional * commission_rate)
    sell_transfer = sell_notional * transfer_rate
    stamp_rate = _stamp_tax_rates(exit_dates, contract.stamp_tax_schedule) / 10_000.0
    stamp_tax = sell_notional * stamp_rate
    ending_cash = cash_after_buy[:, None] + sell_notional - sell_commission - sell_transfer - stamp_tax
    entry_mask = np.asarray(entry_filled, dtype=bool).reshape(-1, 1)
    resolved_mask = np.asarray(exit_resolved, dtype=bool)
    if resolved_mask.shape != exits.shape:
        raise ValueError("exit_resolved must match exit_prices")
    position_valid = (shares[:, None] > 0) & np.isfinite(basis[:, None]) & (basis[:, None] > 0.0)
    executable_exit = resolved_mask & np.isfinite(exits) & (exits > 0.0)
    terminal_proceeds = (
        shares[:, None]
        * basis[:, None]
        * float(contract.terminal_recovery_fraction)
    )
    resolved_proceeds = sell_notional - sell_commission - sell_transfer - stamp_tax
    ending_cash = cash_after_buy[:, None] + np.where(
        executable_exit,
        resolved_proceeds,
        terminal_proceeds,
    )
    valid = entry_mask & position_valid & (executable_exit | (~resolved_mask))
    net_return = ending_cash / float(reference_cash) - 1.0
    if include_buy_cost:
        net_return = np.where(entry_mask, net_return, 0.0)
        valid = valid | (~entry_mask)
    net_return = np.where(valid, net_return, np.nan)
    return np.log(np.maximum(1.0 + net_return, 1.0e-4)).astype(np.float32)


def build_qcurve_targets(
    *,
    signal_close_raw: np.ndarray,
    future_open_raw: np.ndarray,
    future_open_sellable: np.ndarray,
    future_trade_dates: np.ndarray,
    entry_filled: np.ndarray,
    contract: QCurveCostContract,
    reference_cash: float = QCURVE_REFERENCE_POSITION_CASH_CNY,
    slippage_multiplier: float = 1.0,
) -> dict[str, np.ndarray]:
    opens = np.asarray(future_open_raw, dtype=np.float64)
    sellable = np.asarray(future_open_sellable, dtype=bool)
    dates = np.asarray(future_trade_dates)
    expected_days = QCURVE_FORWARD_DAYS + QCURVE_EXECUTION_TAIL_DAYS
    if opens.ndim != 2 or opens.shape[1] < expected_days:
        raise ValueError(f"future_open_raw must provide at least {expected_days} days")
    if sellable.shape != opens.shape or dates.shape != opens.shape:
        raise ValueError("future open prices, sellability, and dates must share shape")
    batch = opens.shape[0]
    if np.asarray(signal_close_raw).reshape(-1).size != batch:
        raise ValueError("signal_close_raw must match batch size")
    next_sellable = _next_sellable_indices(sellable)

    def resolved(horizons: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        planned = np.asarray(horizons, dtype=np.int64) - 1
        indices = next_sellable[:, planned]
        terminal_idx = np.minimum(
            planned + int(QCURVE_EXECUTION_TAIL_DAYS),
            expected_days - 1,
        )[None, :]
        unresolved = (indices < 0) | (indices > terminal_idx)
        terminal_idx = np.broadcast_to(terminal_idx, indices.shape)
        safe_indices = np.where(unresolved, terminal_idx, indices)
        rows = np.arange(batch, dtype=np.int64)[:, None]
        prices = opens[rows, safe_indices]
        if bool(unresolved.any()):
            recovery_basis = np.asarray(signal_close_raw, dtype=np.float64).reshape(-1, 1)
            prices = np.where(unresolved, recovery_basis * contract.terminal_recovery_fraction, prices)
        resolved_dates = dates[rows, safe_indices]
        return prices, resolved_dates, safe_indices + 1, ~unresolved

    entry_prices, entry_dates, entry_resolved_days, entry_resolved = resolved(QCURVE_ENTRY_HORIZONS)
    hold_prices, hold_dates, hold_resolved_days, hold_resolved = resolved(QCURVE_HOLD_HORIZONS)
    entry_curve = _net_log_curve(
        reference_cash=float(reference_cash),
        basis_price=opens[:, 0],
        exit_prices=entry_prices,
        exit_dates=entry_dates,
        entry_filled=np.asarray(entry_filled, dtype=bool),
        exit_resolved=entry_resolved,
        contract=contract,
        include_buy_cost=True,
        slippage_multiplier=float(slippage_multiplier),
    )
    hold_curve = _net_log_curve(
        reference_cash=float(reference_cash),
        basis_price=np.asarray(signal_close_raw, dtype=np.float64),
        exit_prices=hold_prices,
        exit_dates=hold_dates,
        entry_filled=np.ones(batch, dtype=bool),
        exit_resolved=hold_resolved,
        contract=contract,
        include_buy_cost=False,
        slippage_multiplier=float(slippage_multiplier),
    )
    return {
        "enter_net_log_return": entry_curve,
        "hold_net_log_return": hold_curve,
        "enter_resolved_exit_day": entry_resolved_days.astype(np.int16),
        "hold_resolved_exit_day": hold_resolved_days.astype(np.int16),
        "enter_exit_resolved": entry_resolved,
        "hold_exit_resolved": hold_resolved,
        "enter_horizons": np.asarray(QCURVE_ENTRY_HORIZONS, dtype=np.int16),
        "hold_horizons": np.asarray(QCURVE_HOLD_HORIZONS, dtype=np.int16),
    }


class _ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=int(dilation), dilation=int(dilation))
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + self.dropout(F.gelu(z)))


class QCurveModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        layers: int = 2,
        dropout: float = 0.10,
        encoder_type: str = "gru",
        horizon_embedding_dim: int = 32,
        input_group_dims: tuple[int, int, int] | None = None,
        enable_auxiliary_heads: bool = True,
    ) -> None:
        super().__init__()
        encoder = str(encoder_type).strip().lower()
        if encoder not in {"gru", "multiscale_tcn"}:
            raise ValueError("encoder_type must be gru or multiscale_tcn")
        self.encoder_type = encoder
        self.enable_auxiliary_heads = bool(enable_auxiliary_heads)
        self.input_group_dims = tuple(int(value) for value in input_group_dims) if input_group_dims else None
        if self.input_group_dims is not None:
            if encoder != "multiscale_tcn":
                raise ValueError("grouped price/VA/state projections are supported only by multiscale_tcn")
            if len(self.input_group_dims) != 3 or sum(self.input_group_dims) != int(input_dim):
                raise ValueError("input_group_dims must contain price, VA, and state widths summing to input_dim")
        self.input_norm = nn.LayerNorm(int(input_dim))
        if self.input_group_dims is None:
            self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
            self.price_proj = None
            self.va_proj = None
            self.state_proj = None
            self.group_fusion = None
        else:
            price_dim, va_dim, state_dim = self.input_group_dims
            self.input_proj = None
            self.price_proj = nn.Linear(price_dim, 48)
            self.va_proj = nn.Linear(va_dim, 48)
            self.state_proj = nn.Linear(state_dim, 32)
            self.group_fusion = nn.Linear(128, int(hidden_dim))
        if encoder == "gru":
            self.encoder = nn.GRU(
                input_size=int(hidden_dim),
                hidden_size=int(hidden_dim),
                num_layers=max(int(layers), 1),
                batch_first=True,
                dropout=float(dropout) if int(layers) > 1 else 0.0,
            )
            self.temporal_blocks = nn.ModuleList()
        else:
            self.encoder = None
            self.temporal_blocks = nn.ModuleList(
                [_ResidualTemporalBlock(int(hidden_dim), dilation, float(dropout)) for dilation in (1, 2, 4, 8, 16, 32)]
            )
        self.attention = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.horizon_embedding = nn.Embedding(QCURVE_FORWARD_DAYS + 1, int(horizon_embedding_dim))
        self.action_embedding = nn.Embedding(2, int(horizon_embedding_dim))
        head_input = int(hidden_dim) + 2 * int(horizon_embedding_dim)
        self.q_head = nn.Sequential(
            nn.Linear(head_input, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 5),
        )
        if self.enable_auxiliary_heads:
            self.path_aux_head: nn.Module | None = nn.Sequential(
                nn.Linear(int(hidden_dim) + int(horizon_embedding_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 6),
            )
            self.trend_aux_head: nn.Module | None = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), 5),
            )
        else:
            self.path_aux_head = None
            self.trend_aux_head = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(x)
        if self.input_group_dims is None:
            assert self.input_proj is not None
            z = F.gelu(self.input_proj(normalized))
        else:
            assert self.price_proj is not None
            assert self.va_proj is not None
            assert self.state_proj is not None
            assert self.group_fusion is not None
            price, va, state = torch.split(normalized, self.input_group_dims, dim=-1)
            grouped = torch.cat(
                [
                    F.gelu(self.price_proj(price)),
                    F.gelu(self.va_proj(va)),
                    F.gelu(self.state_proj(state)),
                ],
                dim=-1,
            )
            z = F.gelu(self.group_fusion(grouped))
        if self.encoder_type == "gru":
            assert self.encoder is not None
            encoded, _ = self.encoder(z)
            return encoded[:, -1]
        for block in self.temporal_blocks:
            z = block(z)
        weights = torch.softmax(self.attention(z).squeeze(-1), dim=1)
        return torch.sum(z * weights.unsqueeze(-1), dim=1)

    @staticmethod
    def _ordered_outputs(raw: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = raw[..., 0]
        q50 = raw[..., 1]
        q20 = q50 - F.softplus(raw[..., 2])
        q80 = q50 + F.softplus(raw[..., 3])
        p_positive = torch.sigmoid(raw[..., 4])
        return {"mean": mean, "q20": q20, "q50": q50, "q80": q80, "p_positive": p_positive}

    def decode(self, pooled: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = int(pooled.shape[0])
        horizons = torch.arange(1, QCURVE_FORWARD_DAYS + 1, device=pooled.device, dtype=torch.long)
        horizon_emb = self.horizon_embedding(horizons).unsqueeze(0).expand(batch, -1, -1)
        pooled_expanded = pooled.unsqueeze(1).expand(-1, QCURVE_FORWARD_DAYS, -1)

        outputs: dict[str, torch.Tensor] = {}
        for action_idx, action_name in enumerate(("hold", "enter")):
            action = self.action_embedding(
                torch.full((batch, QCURVE_FORWARD_DAYS), action_idx, device=pooled.device, dtype=torch.long)
            )
            raw = self.q_head(torch.cat([pooled_expanded, horizon_emb, action], dim=-1))
            ordered = self._ordered_outputs(raw)
            start = 0 if action_name == "hold" else 1
            for name, value in ordered.items():
                outputs[f"{action_name}_{name}"] = value[:, start:]

        if self.path_aux_head is None or self.trend_aux_head is None:
            return outputs

        aux_raw = self.path_aux_head(torch.cat([pooled_expanded, horizon_emb], dim=-1))
        close_delta = aux_raw[..., 0]
        close_log_return = torch.cumsum(close_delta, dim=1)
        previous_close = torch.cat(
            [torch.zeros_like(close_log_return[:, :1]), close_log_return[:, :-1]],
            dim=1,
        )
        open_log_return = previous_close + aux_raw[..., 1]
        upper = F.softplus(aux_raw[..., 2])
        lower = F.softplus(aux_raw[..., 3])
        high_log_return = torch.maximum(open_log_return, close_log_return) + upper
        low_log_return = torch.minimum(open_log_return, close_log_return) - lower
        outputs["path_close_log_return"] = close_log_return
        outputs["path_open_gap"] = aux_raw[..., 1]
        outputs["path_open_log_return"] = open_log_return
        outputs["path_high_log_return"] = high_log_return
        outputs["path_low_log_return"] = low_log_return
        outputs["path_upper_spread"] = upper
        outputs["path_lower_spread"] = lower
        outputs["path_log_volume"] = aux_raw[..., 4]
        outputs["path_log_vwap"] = aux_raw[..., 5]
        outputs["path_log_amount"] = aux_raw[..., 4] + aux_raw[..., 5]
        outputs["path_trend_mean"] = self.trend_aux_head(pooled)
        return outputs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(x))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe = torch.where(mask, values, torch.zeros_like(values))
    return safe.sum() / mask.sum().clamp_min(1).to(dtype=values.dtype)


def _pinball(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    error = target - pred
    return torch.maximum(float(quantile) * error, (float(quantile) - 1.0) * error)


def qcurve_distribution_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    enter_target: torch.Tensor,
    hold_target: torch.Tensor,
    enter_raw_target: torch.Tensor | None = None,
    hold_raw_target: torch.Tensor | None = None,
    enter_location: torch.Tensor | None = None,
    enter_scale: torch.Tensor | None = None,
    hold_location: torch.Tensor | None = None,
    hold_scale: torch.Tensor | None = None,
    weights: QCurveLossWeights = QCurveLossWeights(),
    temperature: float = QCURVE_SOFT_EXIT_TEMPERATURE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights.validate()
    component_values: dict[str, list[torch.Tensor]] = {
        "mean_loss": [],
        "quantile_loss": [],
        "positive_loss": [],
        "soft_exit_loss": [],
    }
    action_inputs = (
        ("enter", enter_target, enter_raw_target, enter_location, enter_scale),
        ("hold", hold_target, hold_raw_target, hold_location, hold_scale),
    )
    for action, target, raw_target_value, location_value, scale_value in action_inputs:
        finite = torch.isfinite(target)
        safe_target = torch.where(finite, target, torch.zeros_like(target))
        mean = outputs[f"{action}_mean"]
        q20 = outputs[f"{action}_q20"]
        q50 = outputs[f"{action}_q50"]
        q80 = outputs[f"{action}_q80"]
        probability = outputs[f"{action}_p_positive"].clamp(1.0e-6, 1.0 - 1.0e-6)
        component_values["mean_loss"].append(
            _masked_mean(F.smooth_l1_loss(mean, safe_target, reduction="none"), finite)
        )
        quantile_raw = (
            _pinball(q20, safe_target, 0.20)
            + _pinball(q50, safe_target, 0.50)
            + _pinball(q80, safe_target, 0.80)
        ) / 3.0
        component_values["quantile_loss"].append(_masked_mean(quantile_raw, finite))
        raw_target = target if raw_target_value is None else raw_target_value
        raw_finite = finite & torch.isfinite(raw_target)
        safe_raw_target = torch.where(raw_finite, raw_target, torch.zeros_like(raw_target))
        location = (
            torch.zeros(target.shape[-1], dtype=mean.dtype, device=mean.device)
            if location_value is None
            else location_value.to(dtype=mean.dtype, device=mean.device)
        )
        scale = (
            torch.ones(target.shape[-1], dtype=mean.dtype, device=mean.device)
            if scale_value is None
            else scale_value.to(dtype=mean.dtype, device=mean.device).clamp_min(1.0e-8)
        )
        raw_mean = mean * scale + location
        positive_target = (safe_raw_target > 0.0).to(dtype=mean.dtype)
        component_values["positive_loss"].append(
            _masked_mean(F.binary_cross_entropy(probability, positive_target, reduction="none"), raw_finite)
        )
        logits = torch.where(raw_finite, raw_mean / float(temperature), torch.full_like(mean, -1.0e9))
        policy = torch.softmax(logits, dim=1)
        component_values["soft_exit_loss"].append(
            -torch.mean(torch.sum(policy * safe_raw_target, dim=1))
        )
    parts = {name: torch.stack(values).mean() for name, values in component_values.items()}
    total = (
        float(weights.mean) * parts["mean_loss"]
        + float(weights.quantile) * parts["quantile_loss"]
        + float(weights.positive) * parts["positive_loss"]
        + float(weights.soft_exit) * parts["soft_exit_loss"]
    )
    parts["distribution_loss"] = total
    return total, parts


def denormalize_qcurve_outputs(
    outputs: Mapping[str, torch.Tensor],
    *,
    enter_location: torch.Tensor,
    enter_scale: torch.Tensor,
    hold_location: torch.Tensor,
    hold_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = dict(outputs)
    for action, location, scale in (
        ("enter", enter_location, enter_scale),
        ("hold", hold_location, hold_scale),
    ):
        loc = location.to(device=outputs[f"{action}_mean"].device, dtype=outputs[f"{action}_mean"].dtype)
        scl = scale.to(device=loc.device, dtype=loc.dtype).clamp_min(1.0e-8)
        for name in ("mean", "q20", "q50", "q80"):
            result[f"{action}_{name}"] = outputs[f"{action}_{name}"] * scl + loc
    return result


def structured_qcurve_auxiliary_loss(
    outputs: Mapping[str, torch.Tensor],
    future_ohlcva_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    target = future_ohlcva_target
    if target.ndim != 3 or target.shape[1:] != (QCURVE_FORWARD_DAYS, 6):
        raise ValueError("future_ohlcva_target must be [batch, 60, 6]")
    price_simple = target[..., :4]
    price_finite = torch.isfinite(price_simple) & (price_simple > -1.0)
    safe_price = torch.where(price_finite, price_simple, torch.zeros_like(price_simple))
    price_log = torch.log1p(safe_price)
    open_log, high_log, low_log, close_log = torch.unbind(price_log, dim=-1)
    previous_close = torch.cat([torch.zeros_like(close_log[:, :1]), close_log[:, :-1]], dim=1)
    open_gap = open_log - previous_close
    upper_spread = high_log - torch.maximum(open_log, close_log)
    lower_spread = torch.minimum(open_log, close_log) - low_log

    price_terms = (
        (outputs["path_close_log_return"], close_log, price_finite[..., 3]),
        (outputs["path_open_gap"], open_gap, price_finite[..., 0] & price_finite[..., 3]),
        (outputs["path_upper_spread"], upper_spread, price_finite.all(dim=-1)),
        (outputs["path_lower_spread"], lower_spread, price_finite.all(dim=-1)),
    )
    price_losses = [
        _masked_mean(F.smooth_l1_loss(pred, truth, reduction="none"), mask)
        for pred, truth, mask in price_terms
    ]
    trend_horizons = (5, 10, 20, 40, 60)
    trend_target = torch.stack(
        [torch.nanmean(torch.where(price_finite[:, :h, 3], close_log[:, :h], torch.nan), dim=1) for h in trend_horizons],
        dim=1,
    )
    trend_mask = torch.isfinite(trend_target)
    price_losses.append(
        _masked_mean(
            F.smooth_l1_loss(outputs["path_trend_mean"], torch.nan_to_num(trend_target), reduction="none"),
            trend_mask,
        )
    )
    price_loss = torch.stack(price_losses).mean()

    volume = target[..., 4]
    amount = target[..., 5]
    va_finite = torch.isfinite(volume) & torch.isfinite(amount)
    safe_volume = torch.where(va_finite, volume, torch.zeros_like(volume))
    safe_amount = torch.where(va_finite, amount, torch.zeros_like(amount))
    vwap = safe_amount - safe_volume
    va_loss = torch.stack(
        [
            _masked_mean(F.smooth_l1_loss(outputs["path_log_volume"], safe_volume, reduction="none"), va_finite),
            _masked_mean(F.smooth_l1_loss(outputs["path_log_vwap"], vwap, reduction="none"), va_finite),
            _masked_mean(F.smooth_l1_loss(outputs["path_log_amount"], safe_amount, reduction="none"), va_finite),
        ]
    ).mean()
    return price_loss, va_loss, {
        "structured_price_trend_aux_loss": price_loss,
        "va_vwap_aux_loss": va_loss,
    }


def full_day_topk_rank_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    *,
    top_count: int = 32,
    top3_weight: float = 4.0,
) -> torch.Tensor:
    if score.ndim != 2 or target.shape != score.shape:
        raise ValueError("full-day rank score and target must have shape [stocks, horizons]")
    losses: list[torch.Tensor] = []
    for horizon in range(int(score.shape[1])):
        finite = torch.isfinite(score[:, horizon]) & torch.isfinite(target[:, horizon])
        if int(finite.sum()) < 4:
            continue
        s = score[finite, horizon].float()
        y = target[finite, horizon].float()
        order = torch.argsort(y, descending=True)
        count = min(int(top_count), int(order.numel()) - 1)
        top = order[:count]
        rest = order[count:]
        if int(rest.numel()) == 0:
            continue
        gap = y[top].unsqueeze(1) - y[rest].unsqueeze(0)
        positive = gap > 0.0
        gap_scale = gap[positive].mean().clamp_min(1.0e-6) if bool(positive.any()) else gap.new_tensor(1.0)
        gap_weight = torch.clamp(gap / gap_scale, min=0.25, max=4.0)
        rank_weight = torch.ones((count, 1), device=score.device, dtype=torch.float32)
        rank_weight[: min(3, count)] = float(top3_weight)
        diff = s[top].unsqueeze(1) - s[rest].unsqueeze(0)
        raw = F.softplus(-diff) * gap_weight * rank_weight
        losses.append(torch.where(positive, raw, torch.zeros_like(raw)).sum() / positive.sum().clamp_min(1))
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def _capped_value_weights(values: pd.Series, *, available: float, cap: float) -> pd.Series:
    positive = values.clip(lower=0.0).astype(float)
    result = pd.Series(0.0, index=positive.index, dtype=float)
    remaining = list(positive[positive > 0.0].index)
    remaining_weight = float(max(available, 0.0))
    while remaining and remaining_weight > 1.0e-12:
        scores = positive.loc[remaining]
        allocation = scores / max(float(scores.sum()), 1.0e-12) * remaining_weight
        hit = allocation[allocation > float(cap) + 1.0e-12]
        if hit.empty:
            result.loc[remaining] = allocation
            break
        for symbol in hit.index:
            result.loc[symbol] = float(cap)
            remaining_weight -= float(cap)
            remaining.remove(symbol)
    return result.clip(lower=0.0, upper=float(cap))


def allocate_dynamic_qcurve_portfolio(
    frame: pd.DataFrame,
    *,
    max_positions: int = QCURVE_MAX_POSITIONS,
    max_position_weight: float = QCURVE_MAX_POSITION_WEIGHT,
) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {
        "symbol",
        "is_held",
        "current_weight",
        "sellable_next_open",
        "enter_mean_curve",
        "enter_q20_curve",
        "hold_mean_curve",
        "hold_q20_curve",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Q-curve allocation frame is missing columns: {missing}")
    working = frame.copy().reset_index(drop=True)
    working["symbol"] = working["symbol"].astype(str)
    if bool(working["symbol"].duplicated().any()):
        raise ValueError("Q-curve allocation requires unique symbols")

    action_rates: list[float] = []
    action_values: list[float] = []
    best_horizons: list[int] = []
    eligible_flags: list[bool] = []
    for row in working.itertuples(index=False):
        held = bool(row.is_held)
        mean = np.asarray(row.hold_mean_curve if held else row.enter_mean_curve, dtype=np.float64)
        q20 = np.asarray(row.hold_q20_curve if held else row.enter_q20_curve, dtype=np.float64)
        horizons = np.asarray(QCURVE_HOLD_HORIZONS if held else QCURVE_ENTRY_HORIZONS, dtype=np.int64)
        holding_days = horizons if held else horizons - 1
        eligible = np.isfinite(mean) & np.isfinite(q20) & (q20 > QCURVE_Q20_GATE)
        if not bool(eligible.any()):
            action_rates.append(float("-inf"))
            action_values.append(float("-inf"))
            best_horizons.append(0)
            eligible_flags.append(False)
            continue
        rate = np.where(eligible, mean / np.maximum(holding_days, 1), -np.inf)
        action_rates.append(float(np.max(rate)))
        action_values.append(0.0)
        best_horizons.append(0)
        eligible_flags.append(True)

    finite_rates = sorted((rate for rate in action_rates if math.isfinite(rate) and rate > 0.0), reverse=True)
    rho = max(0.0, float(finite_rates[3])) if len(finite_rates) >= 4 else 0.0
    for idx, row in enumerate(working.itertuples(index=False)):
        if not eligible_flags[idx]:
            continue
        held = bool(row.is_held)
        mean = np.asarray(row.hold_mean_curve if held else row.enter_mean_curve, dtype=np.float64)
        q20 = np.asarray(row.hold_q20_curve if held else row.enter_q20_curve, dtype=np.float64)
        horizons = np.asarray(QCURVE_HOLD_HORIZONS if held else QCURVE_ENTRY_HORIZONS, dtype=np.int64)
        holding_days = horizons if held else horizons - 1
        eligible = np.isfinite(mean) & np.isfinite(q20) & (q20 > QCURVE_Q20_GATE)
        values = np.where(eligible, mean - rho * holding_days, -np.inf)
        best = int(np.argmax(values))
        action_values[idx] = float(values[best])
        best_horizons[idx] = int(horizons[best])

    working["compound_rate"] = action_rates
    working["path_value"] = action_values
    working["recommended_exit_horizon"] = best_horizons
    locked = working[working["is_held"].astype(bool) & (~working["sellable_next_open"].astype(bool))]
    locked_symbols = set(locked["symbol"])
    locked_weight = float(pd.to_numeric(locked["current_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).sum())
    slots = max(int(max_positions) - len(locked_symbols), 0)
    selectable = working[
        (~working["symbol"].isin(locked_symbols))
        & np.isfinite(working["path_value"])
        & working["path_value"].gt(0.0)
    ].sort_values(["path_value", "symbol"], ascending=[False, True], kind="mergesort")
    selected = selectable.head(slots)
    selected_values = selected.set_index("symbol")["path_value"]
    target = pd.Series(0.0, index=working["symbol"], dtype=float)
    if locked_symbols:
        locked_current = locked.set_index("symbol")["current_weight"].astype(float).clip(lower=0.0)
        target.loc[locked_current.index] = locked_current
    allocated = _capped_value_weights(
        selected_values,
        available=max(0.0, 1.0 - locked_weight),
        cap=float(max_position_weight),
    )
    if not allocated.empty:
        target.loc[allocated.index] = allocated
    working["target_weight"] = working["symbol"].map(target).fillna(0.0).astype(float)
    working["selected"] = working["target_weight"].gt(1.0e-12)
    diagnostics = {
        "rho": float(rho),
        "target_count": float(working["selected"].sum()),
        "gross_exposure": float(working["target_weight"].sum()),
        "cash_weight": float(max(0.0, 1.0 - working["target_weight"].sum())),
        "locked_count": float(len(locked_symbols)),
    }
    return working, diagnostics
