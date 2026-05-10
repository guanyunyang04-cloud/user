from __future__ import annotations

import torch


_NATIVE_ALLOCATION_VECTOR_NAMES: tuple[str, ...] = (
    "portfolio_daily_target_weight",
    "portfolio_daily_target_delta",
    "portfolio_daily_target_cash_weight",
    "portfolio_daily_target_turnover",
    "portfolio_daily_native_receiver_score",
    "portfolio_daily_native_source_score",
    "portfolio_daily_native_cash_score",
)

_NATIVE_ALLOCATION_TERM_NAMES: tuple[str, ...] = (
    "allocation_sum_error",
    "cash_reserve_error",
    "position_cap_violation",
    "turnover_violation",
    "unsupported_receiver_weight",
    "sell_nonheld_violation",
    "receiver_flow_mean",
    "source_flow_mean",
    "funding_shortfall_loss",
    "cash_timing_loss",
    "decision_utility_loss",
    "risk_cost_loss",
    "source_breadth_loss",
    "exposure_utilization_loss",
    "source_dead_loss",
    "native_source_threshold_loss",
    "legacy_mask_block_loss",
    "native_negative_delta_count",
    "native_source_candidate_count",
    "native_source_flow_without_sellable_support",
    "native_source_blocked_by_legacy_mask",
    "native_source_audit_threshold_gap",
    "total",
)


def _project_native_allocation_vector(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    return_terms: bool = False,
) -> dict[str, torch.Tensor]:
    required_outputs = {
        "portfolio_daily_allocation_weight_logit",
        "portfolio_daily_cash_reserve_logit",
        "portfolio_daily_allocation_risk_buffer_logit",
    }
    required_targets = {
        "date_code",
        "current_weight",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
    }
    device = next(iter(outputs.values())).device
    dtype = next(iter(outputs.values())).dtype
    row_count = int(next(iter(outputs.values())).numel())

    def _zero_projection() -> dict[str, torch.Tensor]:
        zero_vec = torch.zeros(row_count, device=device, dtype=dtype)
        result = {
            "portfolio_daily_target_weight": zero_vec,
            "portfolio_daily_target_delta": zero_vec,
            "portfolio_daily_target_cash_weight": zero_vec,
            "portfolio_daily_target_turnover": zero_vec,
            "portfolio_daily_native_receiver_score": zero_vec,
            "portfolio_daily_native_source_score": zero_vec,
            "portfolio_daily_native_cash_score": zero_vec,
        }
        if return_terms:
            zero = torch.tensor(0.0, device=device, dtype=dtype)
            result.update({name: zero for name in _NATIVE_ALLOCATION_TERM_NAMES})
        return result

    if not required_outputs.issubset(outputs) or not required_targets.issubset(targets):
        return _zero_projection()

    weight_logit = outputs["portfolio_daily_allocation_weight_logit"].to(device).flatten()
    cash_logit = outputs["portfolio_daily_cash_reserve_logit"].to(device).flatten()
    risk_buffer_logit = outputs["portfolio_daily_allocation_risk_buffer_logit"].to(device).flatten()
    date_code = targets["date_code"].to(device).flatten()
    current_weight = torch.clamp(targets["current_weight"].to(device).flatten(), 0.0, 1.0)
    receiver_mask = torch.clamp(targets["portfolio_daily_receiver_candidate_mask"].to(device).flatten(), 0.0, 1.0)
    source_mask = torch.clamp(targets["portfolio_daily_source_candidate_mask"].to(device).flatten(), 0.0, 1.0)
    if weight_logit.numel() == 0 or date_code.numel() != weight_logit.numel():
        return _zero_projection()

    def _target(name: str, default: float | torch.Tensor) -> torch.Tensor:
        if name in targets:
            value = targets[name].to(device).flatten()
        elif torch.is_tensor(default):
            value = default.to(device).flatten()
        else:
            value = torch.full_like(weight_logit, float(default))
        if value.numel() == 1 and weight_logit.numel() != 1:
            value = torch.full_like(weight_logit, float(value.detach().cpu()))
        if value.numel() != weight_logit.numel():
            value = torch.full_like(weight_logit, float(torch.nanmean(value).detach().cpu()) if value.numel() else float(default))
        return value

    held_support = (current_weight > 1.0e-8).to(dtype=dtype, device=device)
    receiver_support = receiver_mask
    if "portfolio_daily_receiver_executable_candidate" in targets:
        receiver_support = receiver_support * torch.clamp(
            targets["portfolio_daily_receiver_executable_candidate"].to(device).flatten(),
            0.0,
            1.0,
        )
    legacy_source_support = torch.clamp(source_mask * held_support, 0.0, 1.0)
    source_support = held_support
    receiver_support = torch.clamp(receiver_support, 0.0, 1.0)
    source_support = torch.clamp(source_support, 0.0, 1.0)
    eligible_support = torch.clamp(held_support + receiver_support, 0.0, 1.0)

    gross_exposure_target = torch.clamp(_target("gross_exposure_target", 0.60), 0.0, 1.0)
    turnover_budget = torch.clamp(_target("turnover_budget", 0.36), 0.02, 1.0)
    max_position_weight_target = torch.clamp(_target("max_position_weight_target", 0.20), 0.02, 0.50)
    cash_timing_target = torch.clamp(_target("budget_cash_timing_signal_target", 0.0), 0.0, 1.0)
    deploy_target = torch.clamp(_target("portfolio_daily_allocation_cash_deployment_target", 0.0), 0.0, 1.0)
    net_utility_target = torch.clamp(_target("portfolio_daily_allocation_net_utility_target", deploy_target), 0.0, 1.0)
    final_objective = torch.clamp(_target("portfolio_daily_allocation_final_objective", net_utility_target), 0.0, 1.0)
    target_receiver = torch.clamp(_target("portfolio_daily_unified_receiver_score", receiver_support), 0.0, 1.0)
    target_source = torch.clamp(_target("portfolio_daily_unified_source_score", source_support), 0.0, 1.0)
    receiver_forward = torch.clamp(_target("portfolio_daily_receiver_forward_excess_5d", 0.0), -0.20, 0.20)
    source_forward = torch.clamp(_target("portfolio_daily_source_forward_excess_5d", 0.0), -0.20, 0.20)
    risk_pressure = torch.stack(
        [
            torch.clamp(_target("portfolio_daily_allocation_uncertainty_pressure_target", 0.0), 0.0, 1.0),
            torch.clamp(_target("portfolio_daily_allocation_tail_risk_control_target", 0.0), 0.0, 1.0),
            torch.clamp(_target("portfolio_daily_allocation_drawdown_control_target", 0.0), 0.0, 1.0),
            torch.clamp(_target("market_downside_pressure", 0.0), 0.0, 1.0),
            torch.clamp(_target("cash_regime_pressure", 0.0), 0.0, 1.0),
        ],
        dim=0,
    ).amax(dim=0)

    target_weight = torch.zeros_like(weight_logit) + weight_logit * 0.0
    target_delta = torch.zeros_like(weight_logit) + weight_logit * 0.0
    target_cash_weight = torch.zeros_like(weight_logit) + weight_logit * 0.0
    target_turnover = torch.zeros_like(weight_logit) + weight_logit * 0.0
    native_receiver_score = torch.zeros_like(weight_logit) + weight_logit * 0.0
    native_source_score = torch.zeros_like(weight_logit) + weight_logit * 0.0
    native_cash_score = torch.zeros_like(weight_logit) + weight_logit * 0.0

    term_values: dict[str, list[torch.Tensor]] = {
        name: []
        for name in _NATIVE_ALLOCATION_TERM_NAMES
        if name not in {"receiver_flow_mean", "source_flow_mean", "total"}
    }
    receiver_flow_values: list[torch.Tensor] = []
    source_flow_values: list[torch.Tensor] = []
    day_losses: list[torch.Tensor] = []

    unique_dates = torch.unique(date_code.detach())
    for day_value in unique_dates:
        day_mask = date_code == day_value
        if not bool(day_mask.any()):
            continue
        current_day = current_weight[day_mask]
        weight_logit_day = weight_logit[day_mask]
        cash_logit_day = cash_logit[day_mask]
        risk_buffer_day = risk_buffer_logit[day_mask]
        receiver_support_day = receiver_support[day_mask]
        source_support_day = source_support[day_mask]
        legacy_source_support_day = legacy_source_support[day_mask]
        eligible_day = eligible_support[day_mask]
        held_day = held_support[day_mask]
        gross_day = torch.clamp(gross_exposure_target[day_mask].mean(), 0.0, 1.0)
        turnover_limit = torch.clamp(turnover_budget[day_mask].mean(), 0.02, 1.0)
        cap_day = torch.clamp(max_position_weight_target[day_mask].mean(), 0.02, 0.50)
        risk_day = torch.clamp(risk_pressure[day_mask].mean(), 0.0, 1.0)
        cash_timing_day = torch.clamp(cash_timing_target[day_mask].mean(), 0.0, 1.0)
        deploy_day = torch.clamp(
            0.45 * deploy_target[day_mask].mean()
            + 0.35 * net_utility_target[day_mask].mean()
            + 0.20 * final_objective[day_mask].mean(),
            0.0,
            1.0,
        )
        raw_cash_reserve = torch.sigmoid(cash_logit_day).mean()
        raw_risk_buffer = torch.sigmoid(risk_buffer_day).mean()
        cash_reserve_target = torch.clamp(
            0.03
            + 0.24 * raw_cash_reserve
            + 0.18 * raw_risk_buffer
            + 0.26 * risk_day
            + 0.16 * cash_timing_day
            - 0.18 * deploy_day,
            0.02,
            0.80,
        )
        stock_budget = torch.minimum(gross_day, torch.clamp(1.0 - cash_reserve_target, 0.0, 1.0))

        if float(eligible_day.detach().sum().cpu()) <= 0.0 or float(stock_budget.detach().cpu()) <= 1.0e-8:
            target_day = torch.zeros_like(current_day)
        else:
            masked_logits = torch.where(
                eligible_day > 0.0,
                weight_logit_day,
                torch.full_like(weight_logit_day, -1.0e9),
            )
            base_weight = torch.softmax(masked_logits, dim=0) * eligible_day
            base_weight = base_weight / torch.clamp(base_weight.sum(), min=1.0e-8)
            target_day = torch.clamp(base_weight * stock_budget, 0.0, cap_day) * eligible_day

        delta_day = target_day - current_day
        turnover_day = torch.abs(delta_day).sum()
        if float(turnover_day.detach().cpu()) > float(turnover_limit.detach().cpu()) + 1.0e-8:
            scale = torch.clamp(turnover_limit / torch.clamp(turnover_day, min=1.0e-8), 0.0, 1.0)
            target_day = current_day + delta_day * scale
            target_day = torch.clamp(target_day, 0.0, cap_day) * eligible_day
            delta_day = target_day - current_day
            turnover_day = torch.abs(delta_day).sum()

        source_audit_threshold = torch.maximum(
            torch.full_like(current_day, 0.00125),
            torch.clamp(current_day, min=0.0) * 0.010,
        )
        clamped_current_day = torch.clamp(current_day, 0.0, cap_day)
        preliminary_release = torch.relu(clamped_current_day - target_day) * held_day
        preliminary_source_count = (
            (preliminary_release > source_audit_threshold).to(dtype=dtype, device=device) * held_day
        ).sum()
        held_count = held_day.sum()
        total_preliminary_release = preliminary_release.sum()
        desired_source_count = torch.minimum(held_count, torch.tensor(3.0, device=device, dtype=dtype))
        if (
            float(total_preliminary_release.detach().cpu()) > 1.0e-8
            and float(held_count.detach().cpu()) > 0.0
            and float(preliminary_source_count.detach().cpu()) < float(desired_source_count.detach().cpu())
        ):
            selected_count = max(1, min(3, int(float(held_count.detach().cpu()))))
            source_order_score = (
                preliminary_release
                + 0.030 * torch.clamp(target_source[day_mask], 0.0, 1.0)
                + 0.020 * torch.relu(-source_forward[day_mask])
                + 0.010 * legacy_source_support_day
            )
            source_order_score = torch.where(
                held_day > 0.0,
                source_order_score,
                torch.full_like(source_order_score, -1.0e9),
            )
            selected_indices = torch.topk(source_order_score, k=selected_count).indices
            concentrated_release = torch.zeros_like(current_day)
            remaining_release = total_preliminary_release
            release_floor = torch.maximum(
                source_audit_threshold * 1.35,
                total_preliminary_release / float(selected_count),
            )
            for index_tensor in selected_indices:
                index = int(index_tensor.detach().cpu())
                capacity = clamped_current_day[index]
                proposed_release = torch.minimum(
                    capacity,
                    torch.maximum(preliminary_release[index], release_floor[index]),
                )
                assigned_release = torch.minimum(proposed_release, remaining_release)
                concentrated_release[index] = assigned_release
                remaining_release = torch.relu(remaining_release - assigned_release)
            if float(remaining_release.detach().cpu()) > 1.0e-8:
                for index_tensor in selected_indices:
                    if float(remaining_release.detach().cpu()) <= 1.0e-8:
                        break
                    index = int(index_tensor.detach().cpu())
                    extra_capacity = torch.relu(clamped_current_day[index] - concentrated_release[index])
                    extra_release = torch.minimum(extra_capacity, remaining_release)
                    concentrated_release[index] = concentrated_release[index] + extra_release
                    remaining_release = torch.relu(remaining_release - extra_release)
            releasable_rows = preliminary_release > 1.0e-8
            target_day = torch.where(releasable_rows, clamped_current_day, target_day)
            target_day = target_day - concentrated_release
            target_day = torch.clamp(target_day, 0.0, cap_day) * eligible_day
            delta_day = target_day - current_day
            turnover_day = torch.abs(delta_day).sum()

        cost_proxy = torch.clamp(0.0015 * turnover_day, 0.0, 0.03)
        total_with_cost = target_day.sum() + cost_proxy
        if float(total_with_cost.detach().cpu()) > 1.0 + 1.0e-8:
            target_day = target_day * torch.clamp((1.0 - cost_proxy) / torch.clamp(target_day.sum(), min=1.0e-8), 0.0, 1.0)
            target_day = torch.clamp(target_day, 0.0, cap_day) * eligible_day
            delta_day = target_day - current_day
            turnover_day = torch.abs(delta_day).sum()
            cost_proxy = torch.clamp(0.0015 * turnover_day, 0.0, 0.03)
        cash_after = torch.clamp(1.0 - target_day.sum() - cost_proxy, 0.0, 1.0)

        receiver_headroom = torch.clamp(cap_day - current_day, min=1.0e-6)
        receiver_score_day = torch.clamp(torch.relu(delta_day) / receiver_headroom, 0.0, 1.0) * receiver_support_day
        source_score_day = torch.clamp(torch.relu(-delta_day) / torch.clamp(current_day, min=1.0e-6), 0.0, 1.0) * source_support_day
        cash_score_day = torch.ones_like(current_day) * cash_after
        release_amount = torch.relu(-delta_day) * held_day
        audit_threshold = torch.maximum(
            torch.full_like(current_day, 0.00125),
            torch.clamp(current_day, min=0.0) * 0.010,
        )
        native_source_active = (release_amount > audit_threshold).to(dtype=dtype, device=device) * held_day
        native_negative_delta = (release_amount > 1.0e-8).to(dtype=dtype, device=device) * held_day
        legacy_blocked_source_flow = release_amount * (1.0 - legacy_source_support_day)
        native_source_threshold_gap = (
            torch.relu(audit_threshold - release_amount)
            * (release_amount > 1.0e-8).to(dtype=dtype, device=device)
            * held_day
        ).sum()

        target_weight[day_mask] = target_day
        target_delta[day_mask] = delta_day
        target_cash_weight[day_mask] = cash_score_day
        target_turnover[day_mask] = torch.ones_like(current_day) * turnover_day
        native_receiver_score[day_mask] = receiver_score_day
        native_source_score[day_mask] = source_score_day
        native_cash_score[day_mask] = cash_score_day

        allocation_sum_error = torch.square(torch.relu(target_day.sum() + cash_after + cost_proxy - 1.0))
        cash_reserve_error = torch.square(torch.relu(cash_reserve_target - cash_after))
        position_cap_violation = torch.square(torch.relu(target_day - cap_day)).mean()
        turnover_violation = torch.square(torch.relu(turnover_day - turnover_limit))
        unsupported_receiver_weight = (torch.relu(delta_day) * (1.0 - receiver_support_day)).sum()
        sell_nonheld_violation = (torch.relu(-delta_day) * (1.0 - held_day)).sum()
        receiver_flow = torch.relu(delta_day).sum()
        source_flow = release_amount.sum()
        current_cash = torch.clamp(1.0 - current_day.sum(), 0.0, 1.0)
        cash_release = torch.relu(current_cash - cash_after)
        raw_receiver_demand = (
            torch.sigmoid(weight_logit_day)
            * receiver_support_day
            * torch.clamp(cap_day - current_day, min=0.0)
            * (0.55 + 0.45 * torch.clamp(target_receiver[day_mask], 0.0, 1.0))
        ).sum()
        funding_shortfall_loss = torch.square(torch.relu(raw_receiver_demand - source_flow - cash_release))
        over_cash_loss = torch.square(cash_after) * torch.relu(deploy_day - risk_day - cash_timing_day - 0.04)
        risk_cash_floor = torch.clamp(
            0.04 + 0.36 * risk_day + 0.22 * cash_timing_day - 0.20 * deploy_day,
            0.02,
            0.80,
        )
        under_cash_loss = torch.square(torch.relu(risk_cash_floor - cash_after)) * torch.clamp(
            0.30 + risk_day + cash_timing_day,
            0.0,
            1.5,
        )
        cash_timing_loss = 0.55 * over_cash_loss + 0.45 * under_cash_loss
        receiver_alpha = (
            torch.relu(delta_day)
            * receiver_support_day
            * torch.clamp(0.50 * receiver_forward[day_mask] + 0.025 * target_receiver[day_mask], -0.10, 0.20)
        ).sum()
        source_alpha = (
            torch.relu(-delta_day)
            * source_support_day
            * torch.clamp(-source_forward[day_mask] + 0.025 * target_source[day_mask], -0.10, 0.20)
        ).sum()
        protected_positive_source = (
            torch.relu(-delta_day)
            * source_support_day
            * torch.relu(source_forward[day_mask])
        ).sum()
        decision_utility_loss = torch.relu(0.006 - receiver_alpha - source_alpha + protected_positive_source).square()
        risk_cost_loss = (
            torch.square(torch.relu(risk_day - cash_after))
            + 0.05 * turnover_day
            + 0.04 * (target_day.square().sum() * torch.clamp(0.50 + risk_day, 0.50, 1.50))
        )
        source_count = source_support_day.sum()
        if float(source_count.detach().cpu()) > 0.0:
            soft_source_breadth = native_source_active.sum()
            target_source_breadth = torch.minimum(source_count, torch.tensor(3.0, device=device, dtype=dtype))
            source_breadth_loss = torch.square(torch.relu(target_source_breadth - soft_source_breadth)) / torch.clamp(
                target_source_breadth.square(),
                min=1.0,
            )
        else:
            source_breadth_loss = torch.tensor(0.0, device=device, dtype=dtype)
        exposure_utilization_loss = torch.square(torch.relu(stock_budget - target_day.sum())) * torch.clamp(
            0.35 + deploy_day - risk_day,
            0.0,
            1.25,
        )
        native_source_candidate_count = native_source_active.sum()
        native_negative_delta_count = native_negative_delta.sum()
        native_source_flow_without_sellable_support = (torch.relu(-delta_day) * (1.0 - held_day)).sum()
        native_source_blocked_by_legacy_mask = legacy_blocked_source_flow.sum()
        receiver_pressure_for_source = torch.clamp(
            raw_receiver_demand + torch.relu(stock_budget - target_day.sum()),
            0.0,
            1.0,
        )
        source_dead_loss = (
            torch.square(torch.relu(torch.minimum(source_count, torch.tensor(3.0, device=device, dtype=dtype)) - native_source_candidate_count))
            / torch.clamp(torch.minimum(source_count, torch.tensor(3.0, device=device, dtype=dtype)).square(), min=1.0)
            if float(source_count.detach().cpu()) > 0.0
            else torch.tensor(1.0, device=device, dtype=dtype)
        ) * receiver_pressure_for_source
        native_source_threshold_loss = (
            torch.square(native_source_threshold_gap)
            + torch.square(torch.relu(torch.tensor(1.0, device=device, dtype=dtype) - native_source_candidate_count))
            * receiver_pressure_for_source
        )
        legacy_mask_block_loss = torch.square(native_source_blocked_by_legacy_mask)

        component_loss = (
            0.10 * allocation_sum_error
            + 0.16 * cash_reserve_error
            + 0.16 * position_cap_violation
            + 0.18 * turnover_violation
            + 0.24 * unsupported_receiver_weight
            + 0.22 * sell_nonheld_violation
            + 0.34 * funding_shortfall_loss
            + 0.22 * cash_timing_loss
            + 0.28 * decision_utility_loss
            + 0.18 * risk_cost_loss
            + 0.20 * source_breadth_loss
            + 0.24 * exposure_utilization_loss
            + 0.24 * source_dead_loss
            + 0.18 * native_source_threshold_loss
            + 0.08 * legacy_mask_block_loss
        )
        day_losses.append(component_loss)
        receiver_flow_values.append(receiver_flow)
        source_flow_values.append(source_flow)
        term_values["allocation_sum_error"].append(allocation_sum_error)
        term_values["cash_reserve_error"].append(cash_reserve_error)
        term_values["position_cap_violation"].append(position_cap_violation)
        term_values["turnover_violation"].append(turnover_violation)
        term_values["unsupported_receiver_weight"].append(unsupported_receiver_weight)
        term_values["sell_nonheld_violation"].append(sell_nonheld_violation)
        term_values["funding_shortfall_loss"].append(funding_shortfall_loss)
        term_values["cash_timing_loss"].append(cash_timing_loss)
        term_values["decision_utility_loss"].append(decision_utility_loss)
        term_values["risk_cost_loss"].append(risk_cost_loss)
        term_values["source_breadth_loss"].append(source_breadth_loss)
        term_values["exposure_utilization_loss"].append(exposure_utilization_loss)
        term_values["source_dead_loss"].append(source_dead_loss)
        term_values["native_source_threshold_loss"].append(native_source_threshold_loss)
        term_values["legacy_mask_block_loss"].append(legacy_mask_block_loss)
        term_values["native_negative_delta_count"].append(native_negative_delta_count)
        term_values["native_source_candidate_count"].append(native_source_candidate_count)
        term_values["native_source_flow_without_sellable_support"].append(native_source_flow_without_sellable_support)
        term_values["native_source_blocked_by_legacy_mask"].append(native_source_blocked_by_legacy_mask)
        term_values["native_source_audit_threshold_gap"].append(native_source_threshold_gap)

    result = {
        "portfolio_daily_target_weight": target_weight,
        "portfolio_daily_target_delta": target_delta,
        "portfolio_daily_target_cash_weight": target_cash_weight,
        "portfolio_daily_target_turnover": target_turnover,
        "portfolio_daily_native_receiver_score": native_receiver_score,
        "portfolio_daily_native_source_score": native_source_score,
        "portfolio_daily_native_cash_score": native_cash_score,
    }
    if return_terms:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        terms = {
            name: (torch.stack(values).mean() if values else zero)
            for name, values in term_values.items()
        }
        terms["receiver_flow_mean"] = torch.stack(receiver_flow_values).mean() if receiver_flow_values else zero
        terms["source_flow_mean"] = torch.stack(source_flow_values).mean() if source_flow_values else zero
        terms["total"] = torch.stack(day_losses).mean() if day_losses else zero
        result.update(terms)
    return result


def _portfolio_native_allocation_vector_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    return_terms: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    projection = _project_native_allocation_vector(outputs, targets, return_terms=True)
    if return_terms:
        return {
            name: projection.get(
                name,
                torch.tensor(0.0, device=next(iter(outputs.values())).device),
            )
            for name in _NATIVE_ALLOCATION_TERM_NAMES
        }
    return projection.get("total", torch.tensor(0.0, device=next(iter(outputs.values())).device))

_DAY_SET_NATIVE_ALLOCATION_TERM_NAMES: tuple[str, ...] = (
    "day_set_full_day_batch",
    "day_set_sample_mask_coverage",
    "day_set_padding_weight_violation",
    "allocation_sum_error",
    "cash_reserve_error",
    "position_cap_violation",
    "turnover_violation",
    "unsupported_receiver_weight",
    "sell_nonheld_violation",
    "receiver_flow_mean",
    "source_flow_mean",
    "funding_shortfall_loss",
    "cash_timing_loss",
    "decision_utility_loss",
    "risk_cost_loss",
    "source_breadth_loss",
    "exposure_utilization_loss",
    "source_dead_loss",
    "native_source_threshold_loss",
    "legacy_mask_block_loss",
    "native_negative_delta_count",
    "native_source_candidate_count",
    "native_source_flow_without_sellable_support",
    "native_source_blocked_by_legacy_mask",
    "native_source_audit_threshold_gap",
    "total",
)


def _project_day_set_native_allocation_vector(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    sample_mask: torch.Tensor,
    *,
    return_terms: bool = False,
) -> dict[str, torch.Tensor]:
    device = next(iter(outputs.values())).device
    dtype = next(iter(outputs.values())).dtype
    mask = sample_mask.to(device=device, dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError("day-set native allocation projection requires sample_mask with shape [B,N].")
    batch_size, max_rows = int(mask.shape[0]), int(mask.shape[1])

    def _output_matrix(name: str) -> torch.Tensor:
        value = outputs[name].to(device=device, dtype=dtype)
        if value.ndim == 1 and value.numel() == batch_size:
            value = value[:, None].expand(batch_size, max_rows)
        if value.shape != mask.shape:
            value = value.reshape(batch_size, max_rows)
        return value

    def _target_matrix(name: str, default: float) -> torch.Tensor:
        if name not in targets:
            return torch.full((batch_size, max_rows), float(default), device=device, dtype=dtype)
        value = targets[name].to(device=device, dtype=dtype)
        if value.ndim == 0:
            return torch.full((batch_size, max_rows), float(value.detach().cpu()), device=device, dtype=dtype)
        if value.ndim == 1:
            if value.numel() == batch_size:
                return value[:, None].expand(batch_size, max_rows)
            if value.numel() == batch_size * max_rows:
                return value.reshape(batch_size, max_rows)
        if tuple(value.shape) == (batch_size, max_rows):
            return value
        return torch.full((batch_size, max_rows), float(default), device=device, dtype=dtype)

    weight_logit = _output_matrix("portfolio_daily_allocation_weight_logit")
    cash_logit = _output_matrix("portfolio_daily_cash_reserve_logit")
    risk_buffer_logit = _output_matrix("portfolio_daily_allocation_risk_buffer_logit")
    masked_weight_logit = torch.where(mask, weight_logit, torch.full_like(weight_logit, -1.0e9))

    flat_outputs = {
        "portfolio_daily_allocation_weight_logit": masked_weight_logit.reshape(-1),
        "portfolio_daily_cash_reserve_logit": cash_logit.reshape(-1),
        "portfolio_daily_allocation_risk_buffer_logit": risk_buffer_logit.reshape(-1),
    }
    flat_targets = {
        "date_code": torch.arange(batch_size, device=device, dtype=dtype)[:, None].expand(batch_size, max_rows).reshape(-1),
        "current_weight": _target_matrix("current_weight", 0.0).reshape(-1),
        "portfolio_daily_receiver_candidate_mask": (_target_matrix("portfolio_daily_receiver_candidate_mask", 0.0) * mask.to(dtype)).reshape(-1),
        "portfolio_daily_source_candidate_mask": (_target_matrix("portfolio_daily_source_candidate_mask", 0.0) * mask.to(dtype)).reshape(-1),
    }
    for name, default in (
        ("portfolio_daily_receiver_executable_candidate", 0.0),
        ("portfolio_daily_source_executable_candidate", 0.0),
        ("gross_exposure_target", 0.60),
        ("turnover_budget", 0.36),
        ("max_position_weight_target", 0.20),
        ("budget_cash_timing_signal_target", 0.0),
        ("portfolio_daily_allocation_cash_deployment_target", 0.0),
        ("portfolio_daily_allocation_net_utility_target", 0.0),
        ("portfolio_daily_allocation_final_objective", 0.0),
        ("portfolio_daily_unified_receiver_score", 0.0),
        ("portfolio_daily_unified_source_score", 0.0),
        ("portfolio_daily_receiver_forward_excess_5d", 0.0),
        ("portfolio_daily_source_forward_excess_5d", 0.0),
        ("portfolio_daily_allocation_uncertainty_pressure_target", 0.0),
        ("portfolio_daily_allocation_tail_risk_control_target", 0.0),
        ("portfolio_daily_allocation_drawdown_control_target", 0.0),
        ("market_downside_pressure", 0.0),
        ("cash_regime_pressure", 0.0),
    ):
        value = _target_matrix(name, default)
        if name.endswith("candidate") or name.endswith("mask"):
            value = value * mask.to(dtype)
        flat_targets[name] = value.reshape(-1)

    flat_projection = _project_native_allocation_vector(flat_outputs, flat_targets, return_terms=True)

    def _reshape(name: str) -> torch.Tensor:
        return flat_projection[name].reshape(batch_size, max_rows) * mask.to(dtype)

    target_weight = _reshape("portfolio_daily_target_weight")
    target_delta = _reshape("portfolio_daily_target_delta")
    native_receiver_score = _reshape("portfolio_daily_native_receiver_score")
    native_source_score = _reshape("portfolio_daily_native_source_score")
    native_cash_score = _reshape("portfolio_daily_native_cash_score")
    target_cash_matrix = flat_projection["portfolio_daily_target_cash_weight"].reshape(batch_size, max_rows)
    target_turnover_matrix = flat_projection["portfolio_daily_target_turnover"].reshape(batch_size, max_rows)
    valid_counts = torch.clamp(mask.to(dtype).sum(dim=1), min=1.0)
    target_cash_weight = (target_cash_matrix * mask.to(dtype)).sum(dim=1) / valid_counts
    target_turnover = (target_turnover_matrix * mask.to(dtype)).sum(dim=1) / valid_counts
    padding_weight_violation = torch.square(target_weight * (~mask).to(dtype)).mean()

    result = {
        "portfolio_daily_target_weight": target_weight,
        "portfolio_daily_target_delta": target_delta,
        "portfolio_daily_target_cash_weight": target_cash_weight,
        "portfolio_daily_target_turnover": target_turnover,
        "portfolio_daily_native_receiver_score": native_receiver_score,
        "portfolio_daily_native_source_score": native_source_score,
        "portfolio_daily_native_cash_score": native_cash_score,
        "padding_weight_violation": padding_weight_violation,
    }
    if return_terms:
        zero = torch.tensor(0.0, device=device, dtype=dtype)
        result.update(
            {
                "day_set_full_day_batch": torch.tensor(1.0, device=device, dtype=dtype),
                "day_set_sample_mask_coverage": mask.to(dtype).mean() if mask.numel() else zero,
                "day_set_padding_weight_violation": padding_weight_violation,
            }
        )
        for name in _NATIVE_ALLOCATION_TERM_NAMES:
            result[name] = flat_projection.get(name, zero)
        result["total"] = result["total"] + 0.10 * padding_weight_violation
    return result


def _portfolio_day_set_native_allocation_vector_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    sample_mask: torch.Tensor,
    *,
    return_terms: bool = False,
) -> torch.Tensor | dict[str, torch.Tensor]:
    projection = _project_day_set_native_allocation_vector(outputs, targets, sample_mask, return_terms=True)
    if return_terms:
        return {
            name: projection.get(
                name,
                torch.tensor(0.0, device=next(iter(outputs.values())).device),
            )
            for name in _DAY_SET_NATIVE_ALLOCATION_TERM_NAMES
        }
    return projection.get("total", torch.tensor(0.0, device=next(iter(outputs.values())).device))
