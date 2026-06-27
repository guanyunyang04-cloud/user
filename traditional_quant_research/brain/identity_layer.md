# Traditional Quant Research 身份对象

## object `traditional_quant_research`
`type`: traditional_quant_lab_brain
`definition`: 可解释、可复现、低复杂度的传统量化方法研究实验室。
`focus`: 因子研究、回测协议、组合构建、风险控制、成本建模、事件/日历效应和传统 ML/tabular 方法。
`not_default`: deep learning、reinforcement learning and production execution systems.
`north_star`: Baostock-only personal quant strategy research with explicit evidence grade and user-discretion endpoint.
`methods`: `inspect_frontier_state()`；`run_diagnostic_or_formal_protocol()`；`update_research_log()`；`select_personal_candidate()`。

## object `evidence_identity`
`type`: research_evidence_model
`truth_dimensions`: data range, sample split, fee/slippage, holding constraints, out-of-sample state, gate status.
`candidate_boundary`: agent work stops at `personal_backtest_candidate`; paper/live/trading decisions belong to user discretion.

## Pure Functions
- `classify_research_result(run) -> diagnostic|personal_research|personal_backtest_candidate|strategy_candidate`
- `requires_true_size_gate(task) -> bool`
- `select_research_protocol(task) -> factor|backtest|frontier|ml|shortline|data_audit`
