# v2 Industry/Size Source Audit

- run_id: `v2_industry_size_source_audit_20260602_185451`
- snapshot_id: `baostock_v2_pit_20160101_20260601_stockbasic_fixed`
- snapshot_root: `traditional_quant_research\data\raw\baostock_daily_mainboard_v2_pit\baostock_v2_pit_20160101_20260601_stockbasic_fixed`
- status: `field_audit`
- candidate_count: `0`

## Field Summary

- `true_industry_in_snapshot`: `False`
- `true_market_cap_in_snapshot`: `False`
- `share_base_in_snapshot`: `False`
- `turnover_in_snapshot`: `False`
- `size_liquidity_proxy_in_snapshot`: `True`
- `pit_status_fields_in_snapshot`: `True`
- `baostock_query_stock_industry_available`: `True`
- `baostock_history_query_available`: `True`

## Key Evidence

- true industry: not available
- true market cap: not available
- share base: not available
- turnover: not available
- size liquidity proxy: amount, volume
- baostock client: query_stock_industry, query_history_k_data_plus, query_stock_basic, query_profit_data, query_operation_data, query_growth_data, query_balance_data, query_cash_flow_data, query_dupont_data, query_performance_express_report

## Recommendations

- Extend v2 with a cached Baostock industry table before running true industry-neutral tests.
- Do not claim market-cap neutralization from the current snapshot; add vetted market-cap/float-cap data or keep size controls explicitly proxy-only.
- Continue using amount/volume-derived fields only as liquidity/size proxies until true cap fields are added.
- Audit whether Baostock daily bars can add turnover before using turnover-based liquidity controls.
- Keep frontier signals at candidate-frontier/backtest_only until true exposure controls are audited.

## Interpretation

This audit is a source/schema gate. It does not promote any strategy candidate. It decides whether the current v2 snapshot can support true industry and size neutralization, or whether the dataset must be extended first.
